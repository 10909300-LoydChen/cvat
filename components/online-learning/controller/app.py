"""Active learning controller: the browser-facing API of the online learning PyTriton service.

It is served under /al/ on the CVAT host (see docker-compose.online-learning.yml), so the CVAT UI
calls it with the user's CVAT session. Every request is authorized by CVAT itself: the controller
forwards the user's cookies to CVAT, so users can only score jobs they can open in CVAT.
"""

import io
import json
import logging
import os
import threading
from urllib.parse import urlparse

import numpy as np
import requests
import tritonclient.http as triton_http
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

CVAT_URL = os.environ.get("CVAT_URL", "http://cvat_server:8080")
TRITON_URL = os.environ.get("TRITON_URL", "pytriton:8000")
TOP_K = 3  # most ambiguous detections averaged into the frame score

LOGGER = logging.getLogger("al_controller")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
)

app = FastAPI()
scoring: dict[int, dict] = {}  # job id -> scoring state
scoring_lock = threading.Lock()


def triton_json(model: str, name: str, value: bytes, output: str) -> dict:
    client = triton_http.InferenceServerClient(TRITON_URL)
    data = triton_http.InferInput(name, [1], "BYTES")
    data.set_data_from_numpy(np.asarray([value], dtype=np.object_))
    return json.loads(client.infer(model, inputs=[data]).as_numpy(output)[0])


def cvat_session(request: Request, path: str) -> tuple[requests.Session, dict]:
    """Return a CVAT session acting as the requesting user, after CVAT authorized `path`."""
    if request.method != "GET":
        # cross-site requests must not trigger actions with the user's cookies
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Cross-origin request")

    session = requests.Session()
    session.trust_env = False
    session.headers["Cookie"] = request.headers.get("cookie", "")
    # Django rejects the internal "cvat_server" name (underscores are invalid in host names),
    # so present the public host name the browser used, which CVAT is configured to accept
    session.headers["Host"] = request.headers.get("host", "localhost")
    session.params = {"org": request.query_params.get("org", "")}
    response = session.get(f"{CVAT_URL}/api/{path}", timeout=10)
    if response.status_code in (401, 403, 404):
        raise HTTPException(response.status_code, "Not allowed by CVAT")
    response.raise_for_status()
    return session, response.json()


def snapshot(job_id: int) -> dict:
    with scoring_lock:
        state = scoring.get(job_id)
        if state is None:
            return {"state": "idle", "progress": 0, "total": 0, "model_version": None, "frames": []}
        return {**state, "frames": list(state["frames"])}


def frame_score(detections: list[dict]) -> float:
    """Uncertainty of a frame: mean ambiguity of its most ambiguous detections.

    A detection with confidence 0.5 is the most ambiguous (1.0), confident ones score close to 0.
    """
    ambiguity = sorted((1 - abs(2 * d["confidence"] - 1) for d in detections), reverse=True)
    return float(np.mean(ambiguity[:TOP_K])) if ambiguity else 0.0


def score_job(
    job_id: int, session: requests.Session, frames: list[int], annotated: set[int]
) -> None:
    state = scoring[job_id]
    client = triton_http.InferenceServerClient(TRITON_URL)
    try:
        for frame in frames:
            response = session.get(
                f"{CVAT_URL}/api/jobs/{job_id}/data",
                params={"type": "frame", "quality": "compressed", "number": frame},
                timeout=60,
            )
            response.raise_for_status()
            pixels = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
            image = triton_http.InferInput("image", list(pixels.shape), "UINT8")
            image.set_data_from_numpy(pixels)
            detections = json.loads(
                client.infer("DetInfer", inputs=[image]).as_numpy("detections")[0]
            )
            state["frames"].append(
                {
                    "frame": frame,
                    "score": round(frame_score(detections), 3),
                    "detections": sum(d["confidence"] >= 0.5 for d in detections),
                    "annotated": frame in annotated,
                }
            )
            state["progress"] = len(state["frames"])
        state["state"] = "done"
    except Exception as e:
        LOGGER.exception("Scoring of job %d failed", job_id)
        state["state"], state["error"] = "failed", str(e)


@app.get("/al/status")
def status(request: Request) -> dict:
    cvat_session(request, "users/self")
    try:
        return triton_json("DetStatus", "query", b"?", "status")
    except Exception as e:
        raise HTTPException(503, f"Model service is not reachable: {e}") from e


class TrainingAction(BaseModel):
    action: str  # pause, resume or train


@app.post("/al/training")
def training(request: Request, body: TrainingAction) -> dict:
    cvat_session(request, "users/self")
    if body.action not in ("pause", "resume", "train"):
        raise HTTPException(400, f"Unknown action: {body.action}")
    return triton_json("DetControl", "command", body.action.encode(), "status")


@app.post("/al/jobs/{job_id}/scores")
def start_scoring(request: Request, job_id: int) -> dict:
    session, _ = cvat_session(request, f"jobs/{job_id}")
    meta = session.get(f"{CVAT_URL}/api/jobs/{job_id}/data/meta", timeout=30).json()
    deleted = set(meta.get("deleted_frames", []))
    frames = [
        f
        for f in (meta.get("included_frames") or range(meta["start_frame"], meta["stop_frame"] + 1))
        if f not in deleted
    ]
    annotations = session.get(f"{CVAT_URL}/api/jobs/{job_id}/annotations", timeout=60).json()
    annotated = {s["frame"] for s in annotations["shapes"]} | {
        s["frame"] for t in annotations["tracks"] for s in t["shapes"]
    }
    model_version = triton_json("DetStatus", "query", b"?", "status")["version"]

    with scoring_lock:
        if scoring.get(job_id, {}).get("state") != "running":
            scoring[job_id] = {
                "state": "running",
                "progress": 0,
                "total": len(frames),
                "model_version": model_version,
                "frames": [],
            }
            threading.Thread(
                target=score_job, args=(job_id, session, frames, annotated), daemon=True
            ).start()
    return snapshot(job_id)


@app.get("/al/jobs/{job_id}/scores")
def get_scores(request: Request, job_id: int) -> dict:
    cvat_session(request, f"jobs/{job_id}")
    return snapshot(job_id)
