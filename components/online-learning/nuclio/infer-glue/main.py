import base64
import io
import json
import os

import numpy as np
import tritonclient.http as triton_http
from PIL import Image

MODEL_NAME = "DetInfer"


def init_context(context):
    # One client per Nuclio worker: tritonclient HTTP clients are not thread-safe
    context.user_data.client = triton_http.InferenceServerClient(
        url=os.environ.get("TRITON_URL", "pytriton:8000"),
        network_timeout=float(os.environ.get("TRITON_TIMEOUT", "25")),
    )


def handler(context, event):
    data = event.body
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    threshold = float(data.get("threshold", 0.5))

    pixels = np.asarray(image, dtype=np.uint8)
    image_input = triton_http.InferInput("image", list(pixels.shape), "UINT8")
    image_input.set_data_from_numpy(pixels)
    result = context.user_data.client.infer(MODEL_NAME, inputs=[image_input])

    detections = [
        {
            "confidence": str(det["confidence"]),
            "label": det["label"],
            "points": det["points"],
            "type": "rectangle",
        }
        for det in json.loads(result.as_numpy("detections")[0])
        if det["confidence"] >= threshold
    ]

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
