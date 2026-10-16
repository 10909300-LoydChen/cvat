"""PyTriton server exposing the online learning detector to CVAT glue functions.

Models:
- DetTrain:  (image, boxes, labels, sample_id) -> accepted   (upserts a sample, training runs in background)
- DetForget: sample_ids -> removed                           (drops samples whose annotations were deleted)
- DetConsolidate: reason -> scheduled                        (requests a full round, e.g. on job completion)
- DetInfer:  image -> detections (JSON)                      (always served by the last promoted model)
- DetStatus: query -> status (JSON)                          (version, labels, metrics, buffer size)

DetInfer returns JSON instead of typed tensors: PyTriton cannot serialize zero-size
tensors, and "no detections" is a normal result.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from detector import OnlineDetector, Sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

LOGGER = logging.getLogger("online_learning.server")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    )

    detector = OnlineDetector(
        base_weights=os.environ.get("BASE_WEIGHTS", "yolo11n.pt"),
        state_dir=Path(os.environ.get("STATE_DIR", "/models")),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        imgsz=int(os.environ.get("IMGSZ", "640")),
        batch_size=int(os.environ.get("BATCH_SIZE", "8")),
        lr=float(os.environ.get("LR", "0.001")),
        epochs_per_round=int(os.environ.get("EPOCHS_PER_ROUND", "20")),
        max_steps_per_round=int(os.environ.get("MAX_STEPS_PER_ROUND", "200")),
        full_epochs=int(os.environ.get("FULL_EPOCHS", "60")),
        full_max_steps=int(os.environ.get("FULL_MAX_STEPS", "3000")),
        min_new_samples=int(os.environ.get("MIN_NEW_SAMPLES", "1")),
        settle_seconds=float(os.environ.get("SETTLE_SECONDS", "3")),
        max_rejections=int(os.environ.get("MAX_REJECTIONS", "3")),
        val_fraction=float(os.environ.get("VAL_FRACTION", "0.2")),
    )

    def train(requests):
        responses = []
        for request in requests:
            changed = detector.add_sample(
                request["sample_id"][0].decode(),
                Sample(
                    # inputs are views of PyTriton shared memory, reused after the call: keep a copy
                    image=request["image"].copy(),
                    boxes=request["boxes"].astype(np.float32).reshape(-1, 4),
                    labels=[name.decode() for name in request["labels"]],
                ),
            )
            responses.append({"accepted": np.array([int(changed)], dtype=np.int32)})
        return responses

    def forget(requests):
        return [
            {
                "removed": np.array(
                    [detector.remove_samples([sid.decode() for sid in request["sample_ids"]])],
                    dtype=np.int32,
                )
            }
            for request in requests
        ]

    def consolidate(requests):
        responses = []
        for request in requests:
            scheduled = detector.request_full_round()
            LOGGER.info(
                "Full round %s (%s)",
                "scheduled" if scheduled else "skipped, nothing changed",
                request["reason"][0].decode(),
            )
            responses.append({"scheduled": np.array([int(scheduled)], dtype=np.int32)})
        return responses

    def infer(requests):
        responses = []
        for request in requests:
            boxes, scores, labels = detector.infer(request["image"])
            detections = [
                {"label": label, "confidence": float(score), "points": [float(v) for v in box]}
                for box, score, label in zip(boxes, scores, labels)
            ]
            responses.append(
                {"detections": np.array([json.dumps(detections).encode()], dtype=np.object_)}
            )
        return responses

    def status(requests):
        payload = json.dumps(detector.status()).encode()
        return [{"status": np.array([payload], dtype=np.object_)} for _ in requests]

    detector.start()
    try:
        with Triton(
            config=TritonConfig(http_port=8000, grpc_port=8001, metrics_port=8002)
        ) as triton:
            triton.bind(
                model_name="DetTrain",
                infer_func=train,
                inputs=[
                    Tensor(name="image", dtype=np.uint8, shape=(-1, -1, 3)),
                    Tensor(name="boxes", dtype=np.float32, shape=(-1, 4)),  # xyxy, pixels
                    Tensor(name="labels", dtype=bytes, shape=(-1,)),
                    Tensor(name="sample_id", dtype=bytes, shape=(1,)),
                ],
                outputs=[Tensor(name="accepted", dtype=np.int32, shape=(1,))],
                config=ModelConfig(batching=False),
                strict=True,
            )
            triton.bind(
                model_name="DetForget",
                infer_func=forget,
                inputs=[Tensor(name="sample_ids", dtype=bytes, shape=(-1,))],
                outputs=[Tensor(name="removed", dtype=np.int32, shape=(1,))],
                config=ModelConfig(batching=False),
                strict=True,
            )
            triton.bind(
                model_name="DetConsolidate",
                infer_func=consolidate,
                inputs=[Tensor(name="reason", dtype=bytes, shape=(1,))],
                outputs=[Tensor(name="scheduled", dtype=np.int32, shape=(1,))],
                config=ModelConfig(batching=False),
                strict=True,
            )
            triton.bind(
                model_name="DetInfer",
                infer_func=infer,
                inputs=[Tensor(name="image", dtype=np.uint8, shape=(-1, -1, 3))],
                outputs=[Tensor(name="detections", dtype=bytes, shape=(1,))],
                config=ModelConfig(batching=False),
                strict=True,
            )
            triton.bind(
                model_name="DetStatus",
                infer_func=status,
                inputs=[Tensor(name="query", dtype=bytes, shape=(1,))],
                outputs=[Tensor(name="status", dtype=bytes, shape=(1,))],
                config=ModelConfig(batching=False),
                strict=True,
            )
            LOGGER.info("Serving models")
            triton.serve()
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
