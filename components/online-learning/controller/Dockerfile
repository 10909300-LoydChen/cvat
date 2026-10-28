FROM python:3.11-slim

WORKDIR /opt/al-controller
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" requests "tritonclient[http]" numpy pillow

COPY app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
