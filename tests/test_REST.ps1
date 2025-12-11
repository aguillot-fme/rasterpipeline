param(
    [string]$AirflowUrl = "http://localhost:8080",
    [string]$Username   = "airflow",
    [string]$Password   = "airflow"
)

# -----------------------------
# 1. Get JWT token from /auth/token
# -----------------------------

$tokenEndpoint = "$AirflowUrl/auth/token"

$tokenBody = @{
    username = $Username
    password = $Password
} | ConvertTo-Json

$tokenHeaders = @{
    "Content-Type" = "application/json"
}

Write-Host "Requesting token from $tokenEndpoint ..."

try {
    $tokenResponse = Invoke-RestMethod -Uri $tokenEndpoint `
                                       -Method POST `
                                       -Headers $tokenHeaders `
                                       -Body $tokenBody
} catch {
    Write-Error "Failed to get token from $tokenEndpoint"
    Write-Error $_
    exit 1
}

if (-not $tokenResponse.access_token) {
    Write-Error "No access_token field in token response. Raw response:"
    $tokenResponse | ConvertTo-Json -Depth 5
    exit 1
}

$accessToken = $tokenResponse.access_token
Write-Host "Successfully obtained JWT token."

# -----------------------------
# 2. Call Airflow REST API with Bearer token
# -----------------------------

# Example: list DAGs
$apiEndpoint = "$AirflowUrl/api/v2/dags"

$apiHeaders = @{
    "Authorization" = "Bearer $accessToken"
}

Write-Host "Calling Airflow API: $apiEndpoint ..."

try {
    $apiResponse = Invoke-RestMethod -Uri $apiEndpoint `
                                     -Method GET `
                                     -Headers $apiHeaders
} catch {
    Write-Error "Failed to call Airflow API at $apiEndpoint"
    Write-Error $_
    exit 1
}

Write-Host "Airflow API response (DAG list):"
$apiResponse | ConvertTo-Json -Depth 5
