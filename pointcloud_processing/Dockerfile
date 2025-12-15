FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep build tools available as a fallback when wheels aren't available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pointcloud_processing/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Vendored pointcloudlib (avoids network/auth during docker builds).
COPY pointcloud_processing/vendor/pointcloudlib /app/pointcloudlib

# Copy shared storage helpers to enable S3/MinIO reads/writes from pointcloud tasks.
COPY storage /app/storage

ENV PYTHONPATH=/app

CMD ["python", "-c", "import pointcloudlib; print('pointcloudlib ok')"]
