# Online learning with PyTriton

Continuously fine-tunes a detector on annotations finished in CVAT and serves it back to CVAT
auto-annotation. Based on the pattern of
[pytriton/examples/online_learning_mnist](https://github.com/triton-inference-server/pytriton/tree/main/examples/online_learning_mnist),
extended for object detection (YOLO11n).

```
                    ┌──────────────────────── Nuclio (glue) ────────────────────────┐
CVAT auto-annotation│ online-learning-detector     base64 image → DetInfer → CVAT JSON│──┐
                    │                                                                 │  │ HTTP
CVAT webhook        │ online-learning-trainer-hook update:job (completed)             │  ▼
(update:job)  ─────▶│   → cvat-sdk: annotations + frames → DetTrain (per frame)       │ ┌──────────────────────┐
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
| `DetTrain`  | `image` uint8 HxWx3, `boxes` xyxy, `labels`, `sample_id` | `accepted` | Upserts a training sample, returns immediately |
| `DetInfer`  | `image` uint8 HxWx3                            | `detections` JSON  | Served by the last promoted model           |
| `DetStatus` | `query`                                        | `status` JSON      | Version, labels, metrics, buffer size       |

A background thread runs a training round once `MIN_NEW_SAMPLES` new samples arrived:

1. **Label growth**: unknown label names extend the detection head. Pretrained weights and the
   channels of already known classes are kept, only the new class channels start from scratch.
2. **Replay**: the round trains on the whole sample buffer, not only new samples, to limit
   catastrophic forgetting.
3. **Validation gate**: ~`VAL_FRACTION` of samples (chosen by a hash of `sample_id`, so the split is
   stable) are never trained on. The candidate replaces the served model only if its F1@IoU0.5
   on them is not worse than the current one.
4. **Persistence**: promoted models are saved to `/models/v<N>.pt` and `/models/latest.pt`, samples to
   `/models/samples/`. Both are restored on restart.

Samples are keyed by `task<id>-frame<n>`: when a job is fixed and completed again, the new
annotations replace the old ones instead of being learned twice.

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

4. Annotate and finish a job (state `completed`). After the first training round the
   "Online learning detector (PyTriton)" model in CVAT starts returning predictions.

Set `REQUIRED_STAGE=acceptance` on the trainer hook to learn only from reviewed jobs.
Without it the model may learn its own mistakes from accepted-but-unreviewed pre-annotations.

### Configuration (`pytriton` service)

| Variable           | Default | Meaning                                       |
|--------------------|---------|-----------------------------------------------|
| `IMGSZ`            | 640     | Training/inference letterbox size             |
| `EPOCHS_PER_ROUND` | 20      | Passes over the sample buffer per round       |
| `MIN_NEW_SAMPLES`  | 8       | New/updated samples that trigger a round      |
| `VAL_FRACTION`     | 0.2     | Share of samples held out for the gate        |
| `LR`, `BATCH_SIZE` | 0.001, 8 | AdamW learning rate, batch size              |

## Limitations

- Only axis-aligned rectangles are used for training; other shapes and tracks are ignored.
- The label list in `infer-glue/function.yaml` is static (CVAT reads it from Nuclio). After adding a
  label in CVAT, add it there and redeploy the function. The model learns the label by itself.
- Training and inference share one device. On a single GPU, inference latency increases during rounds.
- The sample buffer grows without bound; for large projects add eviction or sampling.
- CVAT webhooks have a 10s timeout, so the trainer hook answers `202` and exports the job in background.
  A failed export is only logged; re-completing the job sends it again.
