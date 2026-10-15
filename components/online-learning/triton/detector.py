"""Online learning YOLO detector: double-buffered train/infer models with a validation gate.

Follows the pattern of pytriton/examples/online_learning_mnist, extended for detection:
- samples are upserted by id (re-annotated frames replace older versions)
- the whole sample buffer is replayed every round to limit catastrophic forgetting
- the inference model is replaced only if the validation F1 does not degrade
- new label names grow the detection head (pretrained weights are transferred)
"""

import copy
import hashlib
import io
import json
import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

LOGGER = logging.getLogger(__name__)


@dataclass
class Sample:
    image: np.ndarray  # HxWx3 uint8
    boxes: np.ndarray  # Nx4 float32, xyxy in pixels
    labels: list[str]


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size. Returns image, scale and (pad_x, pad_y)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0


def build_model(base: torch.nn.Module, num_classes: int) -> DetectionModel:
    """Create a detector with num_classes outputs, transferring all shape-compatible weights."""
    model = DetectionModel(copy.deepcopy(base.yaml), nc=num_classes, verbose=False)
    model.load(base, verbose=False)

    # load() skips class-dependent head layers; keep the channels of already known classes
    state, base_state = model.state_dict(), base.state_dict()
    for key, value in state.items():
        old = base_state.get(key)
        if old is not None and old.shape != value.shape and old.shape[1:] == value.shape[1:]:
            n = min(old.shape[0], value.shape[0])
            value[:n] = old[:n]

    model.args = DEFAULT_CFG  # loss hyperparameters (box, cls, dfl)
    return model


class OnlineDetector(threading.Thread):
    def __init__(
        self,
        *,
        base_weights: str,
        state_dir: Path,
        device: torch.device,
        imgsz: int,
        batch_size: int,
        lr: float,
        epochs_per_round: int,
        min_new_samples: int,
        val_fraction: float,
    ):
        super().__init__(daemon=True)
        self.device = device
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.lr = lr
        self.epochs_per_round = epochs_per_round
        self.min_new_samples = min_new_samples
        self.val_fraction = val_fraction

        self.state_dir = state_dir
        self.samples_dir = state_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.base = YOLO(base_weights).model
        self.samples: dict[str, Sample] = {}
        self.new_sample_ids: set[str] = set()
        self.samples_lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stopped = False

        self.labels: list[str] = []
        self.train_model: DetectionModel | None = None
        self.infer_model: DetectionModel | None = None
        self.infer_labels: list[str] = []
        self.infer_lock = threading.Lock()
        self.version = 0
        self.metrics: dict = {}
        self.training = False

        self._restore()

    # ---------- persistence ----------

    def _restore(self) -> None:
        for meta_path in sorted(self.samples_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            image = np.asarray(Image.open(meta_path.with_suffix(".png")).convert("RGB"))
            self.samples[meta["id"]] = Sample(
                image, np.asarray(meta["boxes"], dtype=np.float32).reshape(-1, 4), meta["labels"]
            )

        checkpoint_path = self.state_dir / "latest.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.labels = list(checkpoint["labels"])
            self.version = checkpoint["version"]
            self.metrics = checkpoint["metrics"]
            self.train_model = build_model(self.base, len(self.labels))
            self.train_model.load_state_dict(checkpoint["state_dict"])
            self.train_model.to(self.device)
            self._publish()
            LOGGER.info("Restored model v%d with labels %s", self.version, self.labels)

        LOGGER.info("Restored %d samples", len(self.samples))

    def _save_sample(self, sample_id: str, sample: Sample) -> None:
        stem = hashlib.sha1(sample_id.encode()).hexdigest()
        Image.fromarray(sample.image).save(self.samples_dir / f"{stem}.png")
        (self.samples_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "boxes": sample.boxes.tolist(),
                    "labels": sample.labels,
                }
            )
        )

    def _save_checkpoint(self) -> None:
        checkpoint = {
            "labels": self.labels,
            "version": self.version,
            "metrics": self.metrics,
            "state_dict": self.train_model.state_dict(),
        }
        torch.save(checkpoint, self.state_dir / f"v{self.version}.pt")
        torch.save(checkpoint, self.state_dir / "latest.pt")

    # ---------- triton entry points ----------

    def add_sample(self, sample_id: str, sample: Sample) -> None:
        with self.samples_lock:
            self.samples[sample_id] = sample
            self.new_sample_ids.add(sample_id)
        self._save_sample(sample_id, sample)
        self.wakeup.set()

    def infer(
        self, image: np.ndarray, conf: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with self.infer_lock:
            model, labels = self.infer_model, self.infer_labels
        if model is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []

        detections = self._predict(model, [image], conf)[0]
        return detections[:, :4], detections[:, 4], [labels[int(c)] for c in detections[:, 5]]

    def status(self) -> dict:
        with self.samples_lock:
            pending = len(self.new_sample_ids)
            total = len(self.samples)
        return {
            "version": self.version,
            "labels": self.infer_labels,
            "metrics": self.metrics,
            "samples": total,
            "pending_new_samples": pending,
            "training": self.training,
        }

    # ---------- training loop ----------

    def run(self) -> None:
        while not self.stopped:
            self.wakeup.wait(timeout=5)
            self.wakeup.clear()
            with self.samples_lock:
                ready = len(self.new_sample_ids) >= self.min_new_samples
            if ready and not self.stopped:
                try:
                    self._train_round()
                except Exception:  # keep serving inference even if a round fails
                    LOGGER.exception("Training round failed")

    def stop(self) -> None:
        self.stopped = True
        self.wakeup.set()
        self.join()

    def _is_val(self, sample_id: str) -> bool:
        digest = int(hashlib.sha1(sample_id.encode()).hexdigest(), 16)
        return (digest % 1000) < self.val_fraction * 1000

    def _sync_labels(self, samples: dict[str, Sample]) -> None:
        names = sorted({name for s in samples.values() for name in s.labels} - set(self.labels))
        if not names and self.train_model is not None:
            return

        # Appending keeps existing class indices stable, so learned head channels survive
        self.labels.extend(names)
        if not self.labels:
            return
        previous = self.train_model if self.train_model is not None else self.base
        self.train_model = build_model(previous, len(self.labels)).to(self.device)
        LOGGER.info("Detection head rebuilt for labels %s", self.labels)

    def _train_round(self) -> None:
        self.training = True
        with self.samples_lock:
            samples = dict(self.samples)
            self.new_sample_ids.clear()

        self._sync_labels(samples)
        if not self.labels:
            self.training = False
            return
        train_ids = [sid for sid in samples if not self._is_val(sid)]
        val_ids = [sid for sid in samples if self._is_val(sid)]
        LOGGER.info("Training round: %d train / %d val samples", len(train_ids), len(val_ids))

        model = self.train_model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=5e-4)
        losses = []
        for _ in range(self.epochs_per_round):
            random.shuffle(train_ids)
            for start in range(0, len(train_ids), self.batch_size):
                batch = self._make_batch(
                    [samples[sid] for sid in train_ids[start : start + self.batch_size]]
                )
                loss, _ = model.loss(batch)
                optimizer.zero_grad()
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(loss.sum().item())

        val_samples = [samples[sid] for sid in val_ids]
        candidate_f1 = self._f1(model, self.labels, val_samples)
        with self.infer_lock:
            current_model, current_labels = self.infer_model, self.infer_labels
        current_f1 = (
            self._f1(current_model, current_labels, val_samples)
            if current_model is not None
            else -1.0
        )

        mean_loss = float(np.mean(losses)) if losses else 0.0
        if candidate_f1 >= current_f1:
            self.version += 1
            self.metrics = {
                "val_f1": candidate_f1,
                "train_loss": mean_loss,
                "val_samples": len(val_samples),
            }
            self._save_checkpoint()
            self._publish()
            LOGGER.info(
                "Promoted v%d: val F1 %.3f (previous %.3f)", self.version, candidate_f1, current_f1
            )
        else:
            # Keep training from the candidate but keep serving the better model
            LOGGER.info("Rejected candidate: val F1 %.3f < current %.3f", candidate_f1, current_f1)
        self.training = False

    def _publish(self) -> None:
        candidate = copy.deepcopy(self.train_model).eval()
        with self.infer_lock:
            self.infer_model, self.infer_labels = candidate, list(self.labels)

    # ---------- helpers ----------

    def _make_batch(self, samples: list[Sample]) -> dict:
        images, cls, bboxes, batch_idx = [], [], [], []
        for idx, sample in enumerate(samples):
            image, scale, (pad_x, pad_y) = letterbox(sample.image, self.imgsz)
            boxes = sample.boxes * scale + np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
            if random.random() < 0.5:  # horizontal flip augmentation
                image = image[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]
            images.append(image)
            for box, name in zip(boxes, sample.labels):
                x1, y1, x2, y2 = np.clip(box, 0, self.imgsz) / self.imgsz
                bboxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                cls.append([self.labels.index(name)])
                batch_idx.append(idx)
        return {
            "img": to_tensor(images).to(self.device),
            "cls": torch.tensor(cls, dtype=torch.float32).reshape(-1, 1).to(self.device),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4).to(self.device),
            "batch_idx": torch.tensor(batch_idx, dtype=torch.float32).to(self.device),
        }

    @torch.no_grad()
    def _predict(
        self, model: DetectionModel, images: list[np.ndarray], conf: float
    ) -> list[np.ndarray]:
        letterboxed = [letterbox(image, self.imgsz) for image in images]
        was_training = model.training
        model.eval()
        preds = model(to_tensor([lb[0] for lb in letterboxed]).to(self.device))
        model.train(was_training)
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        results = []
        for det, (_, scale, (pad_x, pad_y)) in zip(
            non_max_suppression(preds, conf_thres=conf, iou_thres=0.5), letterboxed
        ):
            det = det.cpu().numpy().astype(np.float32)
            det[:, [0, 2]] = (det[:, [0, 2]] - pad_x) / scale
            det[:, [1, 3]] = (det[:, [1, 3]] - pad_y) / scale
            results.append(det)
        return results

    def _f1(
        self, model: DetectionModel, labels: list[str], samples: list[Sample], conf: float = 0.25
    ) -> float:
        """F1 at IoU 0.5 over the validation samples (class-aware)."""
        if not samples:
            return 0.0
        tp = fp = fn = 0
        for sample, det in zip(samples, self._predict(model, [s.image for s in samples], conf)):
            matched = set()
            gt_boxes = torch.from_numpy(sample.boxes)
            ious = (
                box_iou(torch.from_numpy(det[:, :4]), gt_boxes)
                if len(det) and len(gt_boxes)
                else None
            )
            for i, row in enumerate(det):
                name = labels[int(row[5])]
                best, best_j = 0.0, -1
                for j, gt_name in enumerate(sample.labels):
                    if (
                        ious is not None
                        and j not in matched
                        and gt_name == name
                        and ious[i, j] > best
                    ):
                        best, best_j = float(ious[i, j]), j
                if best >= 0.5:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            fn += len(sample.labels) - len(matched)
        return 2 * tp / max(2 * tp + fp + fn, 1)
