# Check if Docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker is not running. Please start Docker Desktop."
    exit 1
}

Write-Host "Building and starting containers..."
docker-compose up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "Environment setup complete!"
    Write-Host "Airflow: http://localhost:8080"
    Write-Host "MinIO: http://localhost:9001"
    Write-Host "API: http://localhost:8000/docs"
} else {
    Write-Host "Error: Failed to start containers."
    exit 1
}
