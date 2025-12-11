ARG AIRFLOW_VERSION=3.1.3
ARG PYTHON_VERSION=3.10
ARG CONSTRAINT_PY=3.10
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}
ARG AIRFLOW_VERSION
ARG CONSTRAINT_PY

USER root
# Basic build tooling for rasterio wheels and future native deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow/dags/repo

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pull in processing code and shared libs (storage, helpers, etc.)
COPY --chown=airflow:root . /opt/airflow/dags/repo

ENV PYTHONPATH=/opt/airflow/dags/repo
ENV PATH="/home/airflow/.local/bin:${PATH}"

# The command is overridden by DockerOperator; keep a harmless default.
CMD ["python", "-m", "processing.ingest", "--help"]
