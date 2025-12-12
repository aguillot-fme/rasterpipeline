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
# Install project requirements using the official Airflow constraints to pin provider transitive deps.
# This reduces install breakage across environments.
RUN pip install --no-cache-dir -r /requirements.txt \
    -c https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${CONSTRAINT_PY}.txt

# Copy project code
COPY --chown=airflow:root . /opt/airflow/dags/repo

# Ensure user-level installs (airflow, pytest, etc.) are on PATH
ENV PATH="/home/airflow/.local/bin:${PATH}"
