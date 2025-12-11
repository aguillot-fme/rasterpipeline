ARG AIRFLOW_VERSION=3.1.3
ARG PYTHON_VERSION=3.10
# Separate constraint Python version to avoid picking up patch versions from the base image env
ARG CONSTRAINT_PY=3.10
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}
ARG AIRFLOW_VERSION
ARG CONSTRAINT_PY

USER root
# Install system dependencies for rasterio/gdal if needed
# For simplicity, we rely on binary wheels which usually work for rasterio
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${CONSTRAINT_PY}.txt"

# Copy project code
COPY --chown=airflow:root . /opt/airflow/dags/repo
