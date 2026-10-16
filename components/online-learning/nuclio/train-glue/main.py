import hashlib
import hmac
import json
import os
import threading

import numpy as np
import tritonclient.http as triton_http
from cvat_sdk import make_client
from PIL import Image


def init_context(context):
    context.user_data.lock = threading.Lock()
    context.user_data.running = set()  # jobs being exported
    context.user_data.dirty = set()  # jobs saved again during their export
    context.user_data.sent = {}  # job id -> {frame: annotation digest}, skips unchanged frames
    context.user_data.states = {}  # job id -> state from the latest webhook


def handler(context, event):
    body = event.body if isinstance(event.body, dict) else json.loads(event.body)

    secret = os.environ.get("WEBHOOK_SECRET")
    if secret:
        # CVAT signs json.dumps(payload); Nuclio has already parsed the JSON body, re-dump it in order
        expected = (
            "sha256="
            + hmac.new(secret.encode(), json.dumps(body).encode(), hashlib.sha256).hexdigest()
        )
        if not hmac.compare_digest(expected, event.headers.get("X-Signature-256", "")):
            return context.Response(body="invalid signature", status_code=401)

    # update:job is sent on every annotation save, not only on state changes
    if body.get("event") != "update:job" or body["job"]["type"] != "annotation":
        return context.Response(body="ignored", status_code=200)

    job = body["job"]
    # e.g. TRIGGER_STATES=completed to learn from finished jobs only; empty means every save
    trigger_states = [s for s in os.environ.get("TRIGGER_STATES", "").split(",") if s]
    required_stage = os.environ.get("REQUIRED_STAGE")  # e.g. "acceptance"
    if (trigger_states and job["state"] not in trigger_states) or (
        required_stage and job["stage"] != required_stage
    ):
        return context.Response(body="ignored", status_code=200)

    with context.user_data.lock:
        context.user_data.states[job["id"]] = job["state"]
        if job["id"] in context.user_data.running:
            # export again when the current one finishes, so the latest save is never lost
            context.user_data.dirty.add(job["id"])
            return context.Response(body="queued", status_code=202)
        context.user_data.running.add(job["id"])

    # CVAT waits at most 10 seconds for a webhook response, so the export runs in background
    threading.Thread(target=_export_job, args=(context, job["id"]), daemon=True).start()
    return context.Response(body="accepted", status_code=202)


def _export_job(context, job_id):
    while True:
        try:
            _send_job(context, job_id)
        except Exception as e:
            context.logger.error_with("Failed to send job to trainer", job_id=job_id, error=str(e))

        with context.user_data.lock:
            if job_id not in context.user_data.dirty:
                context.user_data.running.discard(job_id)
                return
            context.user_data.dirty.discard(job_id)


def _send_job(context, job_id):
    triton = triton_http.InferenceServerClient(os.environ.get("TRITON_URL", "pytriton:8000"))
    with make_client(
        os.environ.get("CVAT_URL", "http://cvat_server:8080"),
        credentials=(os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"]),
    ) as cvat:
        job = cvat.jobs.retrieve(job_id)
        label_names = {label.id: label.name for label in job.get_labels()}
        deleted_frames = set(job.get_meta().deleted_frames)

        boxes_per_frame = {}
        for shape in job.get_annotations().shapes:
            # Rectangles only: other shape types need a model with a different head
            if shape.type.value != "rectangle" or shape.rotation or shape.frame in deleted_frames:
                continue
            boxes_per_frame.setdefault(shape.frame, []).append(
                (shape.points, label_names[shape.label_id])
            )

        def sample_id(frame):
            # stable id per frame: re-annotated frames replace the previous sample
            return f"task{job.task_id}-frame{frame}".encode()

        digests = {
            frame: hashlib.sha1(json.dumps(sorted(items)).encode()).hexdigest()
            for frame, items in boxes_per_frame.items()
        }
        sent = context.user_data.sent.setdefault(job_id, {})
        changed = [frame for frame, digest in digests.items() if sent.get(frame) != digest]

        for frame in sorted(changed):
            items = boxes_per_frame[frame]
            image = np.asarray(Image.open(job.get_frame(frame, quality="original")).convert("RGB"))
            values = (
                image,
                np.asarray([points for points, _ in items], dtype=np.float32),
                np.asarray([name.encode() for _, name in items], dtype=np.object_),
                np.asarray([sample_id(frame)], dtype=np.object_),
            )
            inputs = [
                triton_http.InferInput(name, list(value.shape), datatype)
                for name, value, datatype in zip(
                    ("image", "boxes", "labels", "sample_id"),
                    values,
                    ("UINT8", "FP32", "BYTES", "BYTES"),
                )
            ]
            for triton_input, value in zip(inputs, values):
                triton_input.set_data_from_numpy(value)
            triton.infer("DetTrain", inputs=inputs)
            sent[frame] = digests[frame]

        # frames without boxes (never annotated, cleared or deleted) must not stay in the buffer
        empty = [
            sample_id(frame)
            for frame in range(job.start_frame, job.stop_frame + 1)
            if frame not in boxes_per_frame
        ]
        removed = 0
        if empty:
            forget = triton_http.InferInput("sample_ids", [len(empty)], "BYTES")
            forget.set_data_from_numpy(np.asarray(empty, dtype=np.object_))
            removed = int(triton.infer("DetForget", inputs=[forget]).as_numpy("removed")[0])
        for frame in set(sent) - set(boxes_per_frame):
            del sent[frame]

        # finishing a job (by default) also asks for a full round over all collected samples
        full_states = os.environ.get("FULL_TRAIN_STATES", "completed").split(",")
        with context.user_data.lock:
            state = context.user_data.states.get(job_id)
        full = 0
        if state in full_states:
            reason = triton_http.InferInput("reason", [1], "BYTES")
            reason.set_data_from_numpy(
                np.asarray([f"job {job_id} {state}".encode()], dtype=np.object_)
            )
            full = int(triton.infer("DetConsolidate", inputs=[reason]).as_numpy("scheduled")[0])

        context.logger.info_with(
            "Job sent to trainer",
            job_id=job_id,
            changed_frames=len(changed),
            removed=removed,
            full_round=full,
        )
