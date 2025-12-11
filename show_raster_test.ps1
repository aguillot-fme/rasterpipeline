param(
    [string]$RasterPath = "./data/raw/HEL_0_0.tif",
    [string]$RedBand = "",
    [string]$NirBand = "",
    [switch]$KeepArtifacts
)

# Build env passthrough for docker-compose exec
$execArgs = @()
if ($RasterPath) {
    $execArgs += "-e"
    $execArgs += "TEST_RASTER_PATH=$RasterPath"
}
if ($RedBand) {
    $execArgs += "-e"
    $execArgs += "TEST_RED_BAND=$RedBand"
}
if ($NirBand) {
    $execArgs += "-e"
    $execArgs += "TEST_NIR_BAND=$NirBand"
}
if ($KeepArtifacts) {
    $execArgs += "-e"
    $execArgs += "KEEP_TEST_RASTER=1"
}

# Run pytest with -s to show print output
$cmd = "cd /opt/airflow/dags/repo && pytest -s tests/test_raster_pipeline_real.py tests/test_airflow_rest_trigger.py"
Write-Host "Running: docker-compose exec $($execArgs -join ' ') airflow-scheduler /bin/bash -lc `"$cmd`""
docker-compose exec @execArgs airflow-scheduler /bin/bash -lc $cmd
