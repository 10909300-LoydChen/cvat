import hashlib
import hmac
import json
import os
import threading

import numpy as np
import tritonclient.http as triton_http
from cvat_sdk import make_client
from PIL import Image

MODEL_NAME = "DetTrain"


def init_context(context):
    context.user_data.jobs_in_progress = set()
    context.user_data.lock = threading.Lock()


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

    if body.get("event") != "update:job":
        return context.Response(body="ignored", status_code=200)

    job = body["job"]
    required_stage = os.environ.get(
        "REQUIRED_STAGE"
    )  # e.g. "acceptance" to train on reviewed jobs only
    if job["state"] != "completed" or (required_stage and job["stage"] != required_stage):
        return context.Response(body="ignored", status_code=200)

    with context.user_data.lock:
        if job["id"] in context.user_data.jobs_in_progress:
            return context.Response(body="already in progress", status_code=202)
        context.user_data.jobs_in_progress.add(job["id"])

    # CVAT waits at most 10 seconds for a webhook response, so the export runs in background
    threading.Thread(target=_send_job, args=(context, job["id"]), daemon=True).start()
    return context.Response(body="accepted", status_code=202)


def _send_job(context, job_id):
    try:
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
                if (
                    shape.type.value != "rectangle"
                    or shape.rotation
                    or shape.frame in deleted_frames
                ):
                    continue
                boxes_per_frame.setdefault(shape.frame, []).append(
                    (shape.points, label_names[shape.label_id])
                )

            triton = triton_http.InferenceServerClient(
                os.environ.get("TRITON_URL", "pytriton:8000")
            )
            for frame, items in sorted(boxes_per_frame.items()):
                image = np.asarray(
                    Image.open(job.get_frame(frame, quality="original")).convert("RGB")
                )
                boxes = np.asarray([points for points, _ in items], dtype=np.float32)
                labels = np.asarray([name.encode() for _, name in items], dtype=np.object_)
                # stable id per frame: re-annotated frames replace the previous sample
                sample_id = np.asarray(
                    [f"task{job.task_id}-frame{frame}".encode()], dtype=np.object_
                )

                inputs = [
                    triton_http.InferInput("image", list(image.shape), "UINT8"),
                    triton_http.InferInput("boxes", list(boxes.shape), "FP32"),
                    triton_http.InferInput("labels", list(labels.shape), "BYTES"),
                    triton_http.InferInput("sample_id", [1], "BYTES"),
                ]
                for triton_input, value in zip(inputs, (image, boxes, labels, sample_id)):
                    triton_input.set_data_from_numpy(value)
                triton.infer(MODEL_NAME, inputs=inputs)

            context.logger.info_with(
                "Job sent to trainer", job_id=job_id, frames=len(boxes_per_frame)
            )
    except Exception as e:
        context.logger.error_with("Failed to send job to trainer", job_id=job_id, error=str(e))
    finally:
        with context.user_data.lock:
            context.user_data.jobs_in_progress.discard(job_id)
