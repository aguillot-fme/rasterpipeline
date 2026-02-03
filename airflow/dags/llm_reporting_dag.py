import json
import os
import posixpath
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import PythonOperator
from docker.types import Mount

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

ANALYTICS_IMAGE = os.getenv("ANALYTICS_IMAGE", "analytics-embeddings:latest")
PROCESSING_NETWORK = os.getenv("DOCKER_NETWORK", "rasterpipeline_default")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET") or os.getenv("DOCKER_HOST") or "unix://var/run/docker.sock"
DEV_CODE_MOUNT = os.getenv("DEV_CODE_MOUNT", "").lower() in {"1", "true", "yes"}
HOST_REPO_PATH = os.getenv("HOST_REPO_PATH")

COMMON_ENV = {
    "STORAGE_TYPE": os.getenv("STORAGE_TYPE", "s3"),
    "S3_BUCKET": os.getenv("S3_BUCKET", "raster-data"),
    "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "LLM_PLANNER_MODEL": os.getenv("LLM_PLANNER_MODEL", "gpt-4o-mini"),
    "LLM_SQL_MODEL": os.getenv("LLM_SQL_MODEL", "gpt-4o-mini"),
    "LLM_ANALYST_MODEL": os.getenv("LLM_ANALYST_MODEL", "gpt-4o"),
}
COMMON_ENV = {k: v for k, v in COMMON_ENV.items() if v is not None}

STORAGE_TYPE = COMMON_ENV.get("STORAGE_TYPE", "s3").lower()
S3_BUCKET = COMMON_ENV.get("S3_BUCKET", "raster-data")
S3_PREFIX = COMMON_ENV.get("S3_PREFIX", "")
S3_ENDPOINT_URL = COMMON_ENV.get("S3_ENDPOINT_URL", "http://minio:9000")
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "/opt/airflow/dags/repo")
if STORAGE_TYPE in {"s3", "minio"}:
    fs_args = {
        "AWS_ENDPOINT_URL": S3_ENDPOINT_URL,
        "AWS_ACCESS_KEY_ID": COMMON_ENV.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": COMMON_ENV.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": COMMON_ENV.get("AWS_DEFAULT_REGION"),
    }
    fs_args = {k: v for k, v in fs_args.items() if v}
    FS_ARGS_JSON = json.dumps(fs_args)
else:
    FS_ARGS_JSON = "{}"


def _join_s3(prefix: str, *parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p]
    if prefix:
        cleaned = [prefix.strip("/")] + cleaned
    return f"s3://{S3_BUCKET}/{posixpath.join(*cleaned)}"


def _output_dir(base_dir: str, dataset_id: str, run_id: str) -> str:
    if STORAGE_TYPE in {"s3", "minio"}:
        return _join_s3(S3_PREFIX, base_dir, dataset_id, run_id)
    return os.path.join(LOCAL_STORAGE_PATH, base_dir, dataset_id, run_id)


def get_docker_operator(task_id, image, command, environment=None, mounts=None, **kwargs):
    merged_env = {**COMMON_ENV, **(environment or {})}
    final_mounts = []
    if DEV_CODE_MOUNT and HOST_REPO_PATH:
        final_mounts.append(Mount(target="/app", source=HOST_REPO_PATH, type="bind", read_only=False))
    if mounts:
        final_mounts.extend(mounts)

    return DockerOperator(
        task_id=task_id,
        image=image,
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        xcom_all=False,
        command=command,
        docker_url=DOCKER_SOCKET,
        network_mode=PROCESSING_NETWORK,
        environment=merged_env,
        mounts=final_mounts or None,
        mem_limit="4g",
        cpus=2.0,
        **kwargs,
    )


def prepare_inputs(**kwargs):
    dag_run = kwargs.get("dag_run")
    conf = (dag_run.conf if dag_run and getattr(dag_run, "conf", None) else {}) or {}
    dataset_id = conf.get("dataset_id") or "sample_dataset"
    run_id = getattr(dag_run, "run_id", "") or "manual_run"

    input_path = conf.get("input_path") or _join_s3(S3_PREFIX, "datasets", dataset_id, "input.parquet")
    user_goal = conf.get("user_goal") or "Summarize the dataset and highlight key trends."

    output_dir = conf.get("output_dir") or _output_dir("llm_reports", dataset_id, run_id)
    profile_path = conf.get("profile_path") or os.path.join(output_dir, "profile.json")
    plan_path = conf.get("plan_path") or os.path.join(output_dir, "plan.json")
    queries_path = conf.get("queries_path") or os.path.join(output_dir, "queries.json")
    results_dir = conf.get("results_dir") or os.path.join(output_dir, "results")
    report_dir = conf.get("report_dir") or os.path.join(output_dir, "report")

    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "input_path": input_path,
        "user_goal": user_goal,
        "output_dir": output_dir,
        "profile_path": profile_path,
        "plan_path": plan_path,
        "queries_path": queries_path,
        "results_dir": results_dir,
        "report_dir": report_dir,
    }


def write_manifest(**kwargs):
    ti = kwargs["ti"]
    data = ti.xcom_pull(task_ids="prepare_inputs")
    output_dir = data["output_dir"]
    manifest = {
        "dataset_id": data["dataset_id"],
        "run_id": data["run_id"],
        "input_path": data["input_path"],
        "profile_path": data["profile_path"],
        "plan_path": data["plan_path"],
        "queries_path": data["queries_path"],
        "results_dir": data["results_dir"],
        "report_dir": data["report_dir"],
    }

    if output_dir.startswith("s3://"):
        import fsspec

        fs = fsspec.filesystem("s3", **json.loads(FS_ARGS_JSON))
        manifest_path = posixpath.join(output_dir, "run_manifest.json")
        with fs.open(manifest_path, "w") as f:
            f.write(json.dumps(manifest, indent=2))
    else:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


dag = DAG(
    "duckdb_llm_reporting",
    default_args=default_args,
    description="Profile data, plan questions, generate SQL, execute, and report.",
    schedule=None,
    catchup=False,
    tags=["duckdb", "llm", "reporting"],
)


with dag:
    prepare_task = PythonOperator(
        task_id="prepare_inputs",
        python_callable=prepare_inputs,
    )

    profile_task = get_docker_operator(
        task_id="profile_dataset",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/llm_reporting/profile.py",
            "--input_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['input_path'] }}",
            "--output_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['profile_path'] }}",
            "--fs_args",
            FS_ARGS_JSON,
        ],
    )

    planner_task = get_docker_operator(
        task_id="plan_questions",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/llm_reporting/planner.py",
            "--profile_json",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['profile_path'] }}",
            "--user_goal",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['user_goal'] }}",
            "--output_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['plan_path'] }}",
        ],
    )

    sql_task = get_docker_operator(
        task_id="generate_sql",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/llm_reporting/sql_generator.py",
            "--plan_json",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['plan_path'] }}",
            "--table_name",
            "reporting_table",
            "--output_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['queries_path'] }}",
        ],
    )

    execute_task = get_docker_operator(
        task_id="execute_queries",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/llm_reporting/executor.py",
            "--queries_json",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['queries_path'] }}",
            "--input_path",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['input_path'] }}",
            "--output_dir",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['results_dir'] }}",
            "--fs_args",
            FS_ARGS_JSON,
        ],
    )

    report_task = get_docker_operator(
        task_id="generate_report",
        image=ANALYTICS_IMAGE,
        command=[
            "python",
            "scripts/llm_reporting/reporter.py",
            "--results_index_json",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['results_dir'] }}/results_index.json",
            "--user_goal",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['user_goal'] }}",
            "--output_dir",
            "{{ ti.xcom_pull(task_ids='prepare_inputs')['report_dir'] }}",
        ],
    )

    manifest_task = PythonOperator(
        task_id="write_manifest",
        python_callable=write_manifest,
    )

    prepare_task >> profile_task >> planner_task >> sql_task >> execute_task >> report_task >> manifest_task
