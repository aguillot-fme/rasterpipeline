import time

import requests

# Configuration
AIRFLOW_API_URL = "http://localhost:8080/api/v1"
DAG_ID = "raster_ingest"
USERNAME = "airflow"
PASSWORD = "airflow"


def trigger_dag(source_path: str, skip_validation: bool = False) -> str | None:
    """
    Trigger the Airflow DAG via REST API using basic auth.
    Mirrors the official curl examples for Airflow's API v1.
    """
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dag_runs"
    payload = {
        "conf": {
            "source_path": source_path,
            "skip_validation": skip_validation,
        }
    }

    print(f"[client] POST {url}")
    print(f"[client] Payload: {payload}")
    print(f"[client] Using basic auth user={USERNAME}")

    print(f"Triggering DAG {DAG_ID} for file: {source_path}")
    response = requests.post(url, json=payload, auth=(USERNAME, PASSWORD))

    if response.status_code not in (200, 201):
        print(f"[client] Error triggering DAG ({response.status_code})")
        print(f"[client] Response headers: {response.headers}")
        print(f"[client] Response body: {response.text}")
        return None

    run_data = response.json()
    dag_run_id = run_data["dag_run_id"]
    print(f"DAG Run started! ID: {dag_run_id}")
    return dag_run_id


def check_status(dag_run_id: str) -> str | None:
    """
    Poll the status of the DAG run using basic auth.
    """
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dag_runs/{dag_run_id}"

    while True:
        print(f"[client] GET {url}")
        response = requests.get(url, auth=(USERNAME, PASSWORD))
        if response.status_code != 200:
            print(f"[client] Error fetching status ({response.status_code})")
            print(f"[client] Response headers: {response.headers}")
            print(f"[client] Response body: {response.text}")
            break

        data = response.json()
        state = data["state"]
        print(f"Current Status: {state}")

        if state in ["success", "failed"]:
            print(f"Run finished with state: {state}")
            return state

        time.sleep(2)


if __name__ == "__main__":
    # Ensure you are running this from a context where 'tests/data/sample.tif' exists
    # relative to the Airflow worker, OR use an absolute path readable by Docker.
    FILE_PATH = "data/raw/HEL_0_0.tif"

    print("--- 1. Triggering Standard Run ---")
    run_id_1 = trigger_dag(FILE_PATH)
    if run_id_1:
        check_status(run_id_1)

    print("\n--- 2. Triggering Skipped Validation Run (Branching Demo) ---")
    run_id_2 = trigger_dag(FILE_PATH, skip_validation=True)
    if run_id_2:
        check_status(run_id_2)
