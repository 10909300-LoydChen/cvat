"""Online learning YOLO detector: double-buffered train/infer models with a validation gate.

Follows the pattern of pytriton/examples/online_learning_mnist, extended for detection:
- samples are upserted by id (re-annotated frames replace older versions); unchanged
  content is ignored, so repeated saves of the same annotations do not retrain
- a round starts once changes stop arriving for a short time (one CVAT save -> one round)
- every batch mixes the changed samples with replayed older ones to limit forgetting,
  and the number of steps per round is capped so that rounds stay short
- the inference model is replaced only if the validation mAP50 does not degrade; after several
  rejected rounds in a row, training restarts from the served model to stop drifting
- new label names grow the detection head (pretrained weights are transferred)

Besides these incremental rounds, a full round can be requested (e.g. when a CVAT job is
completed): it starts from the served model, passes over the whole buffer for more epochs with
a cosine learning rate schedule and stronger augmentation, and also has to pass the gate.
"""

import copy
import hashlib
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

LOGGER = logging.getLogger(__name__)


@dataclass
class Sample:
    image: np.ndarray  # HxWx3 uint8
    boxes: np.ndarray  # Nx4 float32, xyxy in pixels
    labels: list[str]

    def digest(self) -> str:
        content = hashlib.sha1(self.image.tobytes())
        content.update(self.boxes.astype(np.float32).tobytes())
        content.update(json.dumps(self.labels).encode())
        return content.hexdigest()


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
        max_steps_per_round: int,
        full_epochs: int,
        full_max_steps: int,
        min_new_samples: int,
        settle_seconds: float,
        max_rejections: int,
        val_fraction: float,
    ):
        super().__init__(daemon=True)
        self.device = device
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.lr = lr
        self.epochs_per_round = epochs_per_round
        self.max_steps_per_round = max_steps_per_round
        self.full_epochs = full_epochs
        self.full_max_steps = full_max_steps
        self.min_new_samples = min_new_samples
        self.settle_seconds = settle_seconds
        self.max_rejections = max_rejections
        self.val_fraction = val_fraction

        self.state_dir = state_dir
        self.samples_dir = state_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.base = YOLO(base_weights).model
        self.samples: dict[str, Sample] = {}
        self.digests: dict[str, str] = {}
        self.new_sample_ids: set[str] = set()  # added, changed or removed since the last round
        self.unconsolidated_ids: set[str] = set()  # ... since the last full round
        self.full_requested = False
        self.last_change = 0.0
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
        self.rejections = 0

        self._restore()

    # ---------- persistence ----------

    def _restore(self) -> None:
        checkpoint_path = self.state_dir / "latest.pt"
        checkpoint_time = checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0.0

        for meta_path in sorted(self.samples_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            image = np.asarray(Image.open(meta_path.with_suffix(".png")).convert("RGB"))
            sample = Sample(
                image, np.asarray(meta["boxes"], dtype=np.float32).reshape(-1, 4), meta["labels"]
            )
            self.samples[meta["id"]] = sample
            self.digests[meta["id"]] = sample.digest()
            # samples received after the last promoted model were not learned yet
            if meta_path.stat().st_mtime > checkpoint_time:
                self.new_sample_ids.add(meta["id"])

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

        self.unconsolidated_ids = set(self.new_sample_ids)
        LOGGER.info(
            "Restored %d samples (%d not learned yet)", len(self.samples), len(self.new_sample_ids)
        )

    def _sample_path(self, sample_id: str, suffix: str) -> Path:
        return self.samples_dir / (hashlib.sha1(sample_id.encode()).hexdigest() + suffix)

    def _save_sample(self, sample_id: str, sample: Sample) -> None:
        Image.fromarray(sample.image).save(self._sample_path(sample_id, ".png"))
        self._sample_path(sample_id, ".json").write_text(
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

    def add_sample(self, sample_id: str, sample: Sample) -> bool:
        """Upsert a sample. Returns False if the same content is already known."""
        digest = sample.digest()
        with self.samples_lock:
            if self.digests.get(sample_id) == digest:
                return False
            self.samples[sample_id] = sample
            self.digests[sample_id] = digest
            self.new_sample_ids.add(sample_id)
            self.unconsolidated_ids.add(sample_id)
            self.last_change = time.monotonic()
        self._save_sample(sample_id, sample)
        self.wakeup.set()
        return True

    def remove_samples(self, sample_ids: list[str]) -> int:
        """Forget samples whose frames no longer have annotations. Returns the number removed."""
        with self.samples_lock:
            removed = [sid for sid in sample_ids if self.samples.pop(sid, None) is not None]
            for sid in removed:
                del self.digests[sid]
                self.new_sample_ids.add(sid)
                self.unconsolidated_ids.add(sid)
            if removed:
                self.last_change = time.monotonic()
        for sid in removed:
            self._sample_path(sid, ".png").unlink(missing_ok=True)
            self._sample_path(sid, ".json").unlink(missing_ok=True)
        if removed:
            self.wakeup.set()
        return len(removed)

    def request_full_round(self) -> bool:
        """Schedule a full round if anything changed since the last one. Returns True if scheduled."""
        with self.samples_lock:
            if not self.unconsolidated_ids:
                return False
            self.full_requested = True
        self.wakeup.set()
        return True

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
            "full_round_requested": self.full_requested,
            "training": self.training,
        }

    # ---------- training loop ----------

    def run(self) -> None:
        while not self.stopped:
            self.wakeup.wait(timeout=1)
            self.wakeup.clear()
            with self.samples_lock:
                # wait until a save has been fully received before starting the round
                settled = time.monotonic() - self.last_change >= self.settle_seconds
                full = self.full_requested and settled
                ready = full or (settled and len(self.new_sample_ids) >= self.min_new_samples)
            if ready and not self.stopped:
                try:
                    self._train_round(full)
                except Exception:  # keep serving inference even if a round fails
                    LOGGER.exception("Training round failed")
                finally:
                    self.training = False

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

    def _train_round(self, full: bool) -> None:
        self.training = True
        with self.samples_lock:
            samples = dict(self.samples)
            changed_ids = set(self.unconsolidated_ids if full else self.new_sample_ids)
            self.new_sample_ids.clear()
            if full:
                self.unconsolidated_ids.clear()
                self.full_requested = False

        self._sync_labels(samples)
        train_ids = [sid for sid in samples if not self._is_val(sid)]
        val_ids = [sid for sid in samples if self._is_val(sid)]
        if not self.labels or not train_ids:
            self.training = False
            return

        with self.infer_lock:
            current_model, current_labels = self.infer_model, self.infer_labels
        kind = "full" if full else "incremental"
        if full:
            # start from the validated model rather than the incremental one, which may have drifted
            model = (
                build_model(current_model, len(self.labels)).to(self.device)
                if current_model is not None
                else copy.deepcopy(self.train_model)
            )
            epochs, max_steps = self.full_epochs, self.full_max_steps
        else:
            model = self.train_model
            epochs, max_steps = self.epochs_per_round, self.max_steps_per_round
        steps = min(max_steps, max(1, math.ceil(epochs * len(train_ids) / self.batch_size)))
        LOGGER.info(
            "Training round (%s): %d changed / %d train / %d val samples, %d steps",
            kind,
            len(changed_ids),
            len(train_ids),
            len(val_ids),
            steps,
        )

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=5e-4)
        scheduler = None
        if full:
            warmup = max(1, steps // 20)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(1.0, (step + 1) / warmup)
                * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / steps))),
            )
        fresh_ids = [sid for sid in train_ids if sid in changed_ids]
        queue = []
        losses = []
        for _ in range(steps):
            if full:
                # epoch-style passes over the whole buffer
                if len(queue) < self.batch_size:
                    queue += random.sample(train_ids, len(train_ids))
                batch_ids, queue = queue[: self.batch_size], queue[self.batch_size :]
            else:
                # half of every batch is reserved for changed samples, the rest replays the buffer
                batch_ids = random.sample(fresh_ids, min(len(fresh_ids), self.batch_size // 2))
                replay_ids = [sid for sid in train_ids if sid not in batch_ids]
                batch_ids += random.sample(
                    replay_ids, min(len(replay_ids), self.batch_size - len(batch_ids))
                )
            batch = self._make_batch([samples[sid] for sid in batch_ids], strong=full)
            loss, _ = model.loss(batch)
            optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.sum().item())

        val_samples = [samples[sid] for sid in val_ids]
        candidate_map = self._map50(model, self.labels, val_samples)
        current_map = (
            self._map50(current_model, current_labels, val_samples)
            if current_model is not None
            else -1.0
        )

        mean_loss = float(np.mean(losses)) if losses else 0.0
        if candidate_map >= current_map:
            self.rejections = 0
            self.train_model = model
            self.version += 1
            self.metrics = {
                "round": kind,
                "val_map50": candidate_map,
                "train_loss": mean_loss,
                "val_samples": len(val_samples),
            }
            self._save_checkpoint()
            self._publish()
            LOGGER.info(
                "Promoted v%d (%s): val mAP50 %.3f (previous %.3f)",
                self.version,
                kind,
                candidate_map,
                current_map,
            )
        elif full:
            # incremental training continues from its own model
            LOGGER.info(
                "Rejected full round: val mAP50 %.3f < current %.3f", candidate_map, current_map
            )
        else:
            # Keep serving the better model; keep training from the candidate unless it keeps losing
            self.rejections += 1
            LOGGER.info(
                "Rejected candidate: val mAP50 %.3f < current %.3f", candidate_map, current_map
            )
            if self.rejections >= self.max_rejections:
                self.train_model = build_model(current_model, len(self.labels)).to(self.device)
                self.rejections = 0
                LOGGER.info("Training restarted from the served model v%d", self.version)
        self.training = False

    def _publish(self) -> None:
        candidate = copy.deepcopy(self.train_model).eval()
        with self.infer_lock:
            self.infer_model, self.infer_labels = candidate, list(self.labels)

    # ---------- helpers ----------

    def _make_batch(self, samples: list[Sample], strong: bool = False) -> dict:
        images, cls, bboxes, batch_idx = [], [], [], []
        for idx, sample in enumerate(samples):
            image, scale, (pad_x, pad_y) = letterbox(sample.image, self.imgsz)
            boxes = sample.boxes * scale + np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
            if random.random() < 0.5:  # horizontal flip augmentation
                image = image[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]
            if strong:
                # photometric jitter, then zoom out to a random position (smaller objects)
                gain, bias = random.uniform(0.75, 1.25), random.uniform(-20, 20)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
                size = round(self.imgsz * random.uniform(0.6, 1.0))
                off_x = random.randint(0, self.imgsz - size)
                off_y = random.randint(0, self.imgsz - size)
                canvas = np.full_like(image, 114)
                canvas[off_y : off_y + size, off_x : off_x + size] = np.asarray(
                    Image.fromarray(image).resize((size, size), Image.BILINEAR)
                )
                image = canvas
                boxes = boxes * (size / self.imgsz) + np.array(
                    [off_x, off_y, off_x, off_y], dtype=np.float32
                )
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

    def _map50(self, model: DetectionModel, labels: list[str], samples: list[Sample]) -> float:
        """Mean AP at IoU 0.5 over the validation samples (classes present in the samples)."""
        if not samples:
            return 0.0
        tp, conf, pred_cls, target_cls = [], [], [], []
        for sample, det in zip(samples, self._predict(model, [s.image for s in samples], 0.001)):
            target_cls += [self.labels.index(name) for name in sample.labels]
            ious = (
                box_iou(torch.from_numpy(det[:, :4]), torch.from_numpy(sample.boxes)).numpy()
                if len(det) and len(sample.labels)
                else None
            )
            matched = set()
            for i in np.argsort(-det[:, 4]):  # greedy matching, most confident first
                name = labels[int(det[i, 5])]
                candidates = [
                    j
                    for j, gt_name in enumerate(sample.labels)
                    if gt_name == name and j not in matched and ious[i, j] >= 0.5
                ]
                best = max(candidates, key=lambda j: ious[i, j], default=None)
                if best is not None:
                    matched.add(best)
                tp.append(best is not None)
                conf.append(det[i, 4])
                pred_cls.append(self.labels.index(name))
        if not target_cls or not tp:
            return 0.0
        ap = ap_per_class(
            np.array(tp)[:, None], np.array(conf), np.array(pred_cls), np.array(target_cls)
        )[5]
        return float(ap[:, 0].mean())
