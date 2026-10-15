FROM python:3.11-slim

WORKDIR /opt/online-learning

# ultralytics pulls opencv-python, which needs libGL; the headless build provides the same cv2 without it.
# Resolve everything together so that pytriton's numpy<2 constraint holds, then drop opencv-python and
# reinstall the same headless version, because both packages share the cv2 directory.
RUN pip install --no-cache-dir nvidia-pytriton ultralytics opencv-python-headless && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps \
        "opencv-python-headless==$(pip show opencv-python-headless | sed -n 's/^Version: //p')" && \
    python -c "import cv2, numpy; assert numpy.__version__ < '2', numpy.__version__"

# Bake the pretrained base weights into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY detector.py server.py ./

ENV STATE_DIR=/models \
    BASE_WEIGHTS=/opt/online-learning/yolo11n.pt

EXPOSE 8000 8001 8002
CMD ["python", "server.py"]
