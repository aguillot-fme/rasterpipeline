import json
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_docker_compose_defines_prometheus_and_grafana():
    compose_path = REPO_ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = compose["services"]
    assert "prometheus" in services
    assert "grafana" in services
    assert "statsd-exporter" in services

    assert "9090:9090" in services["prometheus"].get("ports", [])
    assert "3000:3000" in services["grafana"].get("ports", [])

    airflow_env = compose["x-airflow-common"]["environment"]
    assert airflow_env["AIRFLOW__METRICS__STATSD_ON"] == "true"
    assert airflow_env["AIRFLOW__METRICS__STATSD_HOST"] == "statsd-exporter"
    assert airflow_env["AIRFLOW__METRICS__STATSD_PORT"] == "8125"


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_prometheus_config_scrapes_statsd_exporter():
    prometheus_cfg_path = REPO_ROOT / "observability" / "prometheus" / "prometheus.yml"
    cfg = yaml.safe_load(prometheus_cfg_path.read_text(encoding="utf-8"))

    jobs = {job["job_name"]: job for job in cfg.get("scrape_configs", [])}
    assert "prometheus" in jobs
    assert "statsd-exporter" in jobs

    targets = jobs["statsd-exporter"]["static_configs"][0]["targets"]
    assert "statsd-exporter:9102" in targets


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_grafana_provisioning_points_to_prometheus():
    ds_path = (
        REPO_ROOT
        / "observability"
        / "grafana"
        / "provisioning"
        / "datasources"
        / "prometheus.yml"
    )
    ds_cfg = yaml.safe_load(ds_path.read_text(encoding="utf-8"))
    datasource = ds_cfg["datasources"][0]

    assert datasource["type"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["isDefault"] is True

    dashboards_cfg_path = (
        REPO_ROOT
        / "observability"
        / "grafana"
        / "provisioning"
        / "dashboards"
        / "dashboards.yml"
    )
    dashboards_cfg = yaml.safe_load(dashboards_cfg_path.read_text(encoding="utf-8"))
    provider = dashboards_cfg["providers"][0]
    assert provider["type"] == "file"
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"

    dashboard_json_path = (
        REPO_ROOT / "observability" / "grafana" / "dashboards" / "rasterpipeline-overview.json"
    )
    dashboard = json.loads(dashboard_json_path.read_text(encoding="utf-8"))
    assert dashboard["title"] == "Raster Pipeline - Overview"

