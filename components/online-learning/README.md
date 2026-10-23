# Online learning with PyTriton

Fine-tunes a detector every time annotations are saved in CVAT and serves it back to CVAT
auto-annotation. Based on the pattern of
[pytriton/examples/online_learning_mnist](https://github.com/triton-inference-server/pytriton/tree/main/examples/online_learning_mnist),
extended for object detection (YOLO11n).

```
                    ┌──────────────────────── Nuclio (glue) ────────────────────────┐
CVAT auto-annotation│ online-learning-detector     base64 image → DetInfer → CVAT JSON│──┐
                    │                                                                 │  │ HTTP
CVAT webhook        │ online-learning-trainer-hook update:job (sent on every save)    │  ▼
(update:job)  ─────▶│   → changed frames → DetTrain, emptied frames → DetForget       │ ┌──────────────────────┐
                    └─────────────────────────────────────────────────────────────────┘ │ PyTriton             │
                                                                                         │ train model ─┐ SGD   │
                                                                                         │              │ gate  │
                                                                                         │ infer model ◀┘ swap  │
                                                                                         └──────────────────────┘
```

## How it works

PyTriton (`triton/`) keeps two copies of the model in one process:

| Model       | Inputs                                         | Output             | Purpose                                     |
|-------------|------------------------------------------------|--------------------|---------------------------------------------|
| `DetTrain`  | `image` uint8 HxWx3, `boxes` xyxy, `labels`, `sample_id` | `accepted` | Upserts a training sample, returns immediately (0 if the content is unchanged) |
| `DetForget` | `sample_ids`                                   | `removed`          | Drops samples of frames that no longer have boxes |
| `DetConsolidate` | `reason`                                  | `scheduled`        | Requests a full round (0 if nothing changed since the last one) |
| `DetInfer`  | `image` uint8 HxWx3                            | `detections` JSON  | Served by the last promoted model           |
| `DetStatus` | `query`                                        | `status` JSON      | Version, labels, metrics, buffer size       |

CVAT sends `update:job` on every annotation save (saving updates the job), so each save is a
training trigger:

- The trainer hook exports only frames whose boxes changed since its last export, and asks
  PyTriton to forget frames that became empty. Saves that arrive during an export are coalesced
  into one more export, so the latest save is never lost.
- PyTriton ignores samples whose content (image, boxes, labels) is already known, so re-sending a
  job (e.g. after a hook restart, or "Finish the job" right after a save) does not retrain.
- A round starts once at least `MIN_NEW_SAMPLES` samples changed and no change arrived for
  `SETTLE_SECONDS`, i.e. one save results in one round. Saves made while a round runs are
  learned in the next round.
- When a job reaches a state in `FULL_TRAIN_STATES` (default `completed`, e.g. "Finish the job"),
  the trainer hook also requests a **full round**, which runs before any pending incremental round.

There are two kinds of rounds:

| | Incremental (every save) | Full (job finished) |
|---|---|---|
| Starts from | the training model of the previous round | the served (validated) model, discarding incremental drift |
| Batches | half changed samples, half replay | epoch passes over the whole buffer |
| Length | `EPOCHS_PER_ROUND`, at most `MAX_STEPS_PER_ROUND` steps | `FULL_EPOCHS`, at most `FULL_MAX_STEPS` steps |
| Learning rate | constant | warmup + cosine decay |
| Augmentation | horizontal flip | flip, brightness/contrast jitter, random zoom-out |
| If rejected by the gate | counts towards `MAX_REJECTIONS` | discarded, incremental training continues |

Both kinds go through the same steps:

1. **Label growth**: unknown label names extend the detection head. Pretrained weights and the
   channels of already known classes are kept, only the new class channels start from scratch.
2. **Replay**: half of every batch is taken from the changed samples, the rest from the whole buffer,
   to limit catastrophic forgetting. Rounds run `EPOCHS_PER_ROUND` passes over the buffer, capped
   at `MAX_STEPS_PER_ROUND` steps.
3. **Validation gate**: ~`VAL_FRACTION` of samples (chosen by a hash of `sample_id`, so the split is
   stable) are never trained on. The candidate replaces the served model only if its mAP@IoU0.5
   on them is not worse than the current one. A rejected candidate keeps training in the next
   rounds; after `MAX_REJECTIONS` rejections in a row, training restarts from the served model.
4. **Persistence**: promoted models are saved to `/models/v<N>.pt` and `/models/latest.pt`, samples to
   `/models/samples/`. Both are restored on restart; samples saved after the last promoted model
   are learned in the first round after the restart.

Samples are keyed by `task<id>-frame<n>`: when a frame is re-annotated, the new annotations replace
the old ones instead of being learned twice.

## Deployment

1. Start CVAT with serverless support and the PyTriton service:

   ```bash
   docker compose \
     -f docker-compose.yml \
     -f components/serverless/docker-compose.serverless.yml \
     -f components/online-learning/docker-compose.online-learning.yml \
     up -d --build
   ```

   For GPU training, uncomment the `deploy` section of `pytriton` in
   `docker-compose.online-learning.yml`.

2. Deploy the glue functions (`nuctl` must match the Nuclio dashboard version, 1.16.3):

   ```bash
   nuctl create project cvat --platform local
   nuctl deploy --project-name cvat --path components/online-learning/nuclio/infer-glue \
     --file components/online-learning/nuclio/infer-glue/function.yaml --platform local \
     --platform-config '{"attributes": {"network": "cvat_cvat"}}'
   nuctl deploy --project-name cvat --path components/online-learning/nuclio/train-glue \
     --file components/online-learning/nuclio/train-glue/function.yaml --platform local \
     --platform-config '{"attributes": {"network": "cvat_cvat"}}' \
     --env CVAT_USERNAME=<user> --env CVAT_PASSWORD=<password> --env WEBHOOK_SECRET=<secret>
   ```

3. In CVAT, open the project → Actions → Setup webhooks, and create a webhook:
   - Target URL: `http://nuclio-nuclio-online-learning-trainer-hook:8080`
   - Events: `Job` → `update`
   - Secret: the same value as `WEBHOOK_SECRET`

4. Annotate and save. After the first training round the "Online learning detector (PyTriton)"
   model in CVAT starts returning predictions. Finish the job to run a full round.

Training on every save also learns from frames that are still being annotated: objects that are
not boxed yet are learned as background until the frame is saved again. To learn only from
finished or reviewed work, set `TRIGGER_STATES=completed` and/or `REQUIRED_STAGE=acceptance` on
the trainer hook. Without a review stage the model may also learn its own mistakes from accepted
pre-annotations.

### Configuration (`pytriton` service)

| Variable           | Default | Meaning                                       |
|--------------------|---------|-----------------------------------------------|
| `IMGSZ`            | 640     | Training/inference letterbox size             |
| `EPOCHS_PER_ROUND` | 20      | Passes over the sample buffer per round       |
| `MAX_STEPS_PER_ROUND` | 200  | Upper bound of optimizer steps per round      |
| `MIN_NEW_SAMPLES`  | 1       | Changed samples that trigger a round          |
| `SETTLE_SECONDS`   | 3       | Quiet time after the last change before a round |
| `MAX_REJECTIONS`   | 3       | Rejected rounds in a row before restarting from the served model |
| `FULL_EPOCHS`      | 60      | Passes over the sample buffer in a full round |
| `FULL_MAX_STEPS`   | 3000    | Upper bound of optimizer steps in a full round |
| `VAL_FRACTION`     | 0.2     | Share of samples held out for the gate        |
| `LR`, `BATCH_SIZE` | 0.001, 8 | AdamW learning rate, batch size              |

## Limitations

- Only axis-aligned rectangles are used for training; other shapes and tracks are ignored.
- The label list in `infer-glue/function.yaml` is static (CVAT reads it from Nuclio). After adding a
  label in CVAT, add it there and redeploy the function. The model learns the label by itself.
- Training and inference share one device. On a single GPU, inference latency increases during rounds.
- The sample buffer grows without bound; for large projects add eviction.
- Until a few samples land in the validation split, the gate has nothing to compare and accepts
  every candidate, so the first rounds can produce many false positives.
- CVAT webhooks have a 10s timeout, so the trainer hook answers `202` and exports the job in background.
  A failed export is only logged; the next save of the job sends the changes again.
- Each save downloads the annotations of the whole job; frames are downloaded only when changed.
