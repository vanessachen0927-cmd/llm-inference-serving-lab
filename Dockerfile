# CPU/API-only image; real CUDA inference remains in the Colab experiment.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BACKEND=fake BATCH_WINDOW_MS=20 MAX_QUEUE_REQUESTS=64 INFERENCE_TIMEOUT_SECONDS=30
WORKDIR /srv/app
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt \
    && groupadd --system appuser && useradd --system --gid appuser --no-create-home appuser
COPY app/ ./app/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health',timeout=2).close()" || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
