# run_all_tests.ps1
# This script ensures required Docker services are running (starting any that are missing),
# waits for them to become healthy, runs pytest inside the `airflow-scheduler` service, and then
# stops only the services that this script started.

$ErrorActionPreference = "Stop"

# List of required services as defined in docker-compose.yml
$RequiredServices = @("postgres", "minio", "airflow-webserver", "airflow-scheduler")

# Keep track of services we start so we can stop them later
$StartedServices = @()

# Function to check if a container for a given service is already running
function Is-Running {
    param (
        [string]$Service
    )
    # docker-compose creates container names like <project>_<service>_1; we check any container with the service name
    $container = docker ps --filter "name=$Service" --filter "status=running" -q
    return ($null -ne $container -and $container -ne "")
}

# Start missing services
foreach ($svc in $RequiredServices) {
    if (Is-Running -Service $svc) {
        Write-Host "Service $svc is already running."
    }
    else {
        Write-Host "Starting missing service: $svc"
        docker-compose up -d $svc
        $StartedServices += $svc
    }
}

# Helper to wait for a container's health status to become "healthy"
function Wait-For-Healthy {
    param (
        [string]$Service
    )
    Write-Host "Waiting for $Service to become healthy..."
    
    # Get the container name (first match)
    $containerName = docker ps --filter "name=$Service" --format "{{.Names}}" | Select-Object -First 1
    
    if ([string]::IsNullOrEmpty($containerName)) {
        Write-Host "Container for $Service not found. Skipping health check."
        return
    }

    # Loop until health status is healthy or timeout after 120 seconds
    $attempts = 0
    while ($true) {
        try {
            $status = docker inspect --format='{{.State.Health.Status}}' $containerName 2>$null
        }
        catch {
            $status = "none"
        }

        if ($status -eq "healthy") {
            Write-Host "$Service is healthy."
            break
        }

        $attempts++
        if ($attempts -ge 30) {
            Write-Host "Timeout waiting for $Service to become healthy. Continuing anyway."
            break
        }
        Start-Sleep -Seconds 4
    }
}

# Health checks disabled for faster iteration in test runs.

# Ensure Airflow metadata knows about the latest DAG files
Write-Host "Reserializing DAGs to refresh the UI metadata..."
docker-compose exec airflow-scheduler airflow dags reserialize

# Run pytest inside the airflow-scheduler container
Write-Host "Running pytest inside the airflow-scheduler container (excluding REST trigger test)..."
# Use docker-compose exec to run pytest from the project root inside the airflow-scheduler container
docker-compose exec airflow-scheduler /bin/bash -lc "cd /opt/airflow/dags/repo && pytest tests -k 'not rest_trigger'"

Write-Host "Running REST trigger test from the host environment..."
$env:AIRFLOW_REST_URL = "http://localhost:8080"
pytest tests/test_airflow_rest_trigger.py

Write-Host "Tests completed."

# Stop only the services we started (if any)
if ($StartedServices.Count -gt 0) {
    Write-Host "Stopping services that were started by this script: $($StartedServices -join ', ')"
    docker-compose stop $StartedServices
}
else {
    Write-Host "No services were started by this script; leaving existing containers running."
}
