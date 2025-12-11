FROM python:3.10-slim

# Install runtime/build deps for rasterio wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY processing/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the code needed for processing tasks
COPY processing /app/processing
COPY storage /app/storage

ENV PYTHONPATH=/app

# The command is overridden by DockerOperator; keep a harmless default.
CMD ["python", "-m", "processing.ingest", "--help"]
