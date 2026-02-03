const map = L.map("map", {
  center: [60.1699, 24.9384],
  zoom: 7,
});

map.createPane("patches");
map.getPane("patches").style.zIndex = 650;

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const logOutput = document.getElementById("logOutput");
const statusChip = document.getElementById("statusChip");
const coordX = document.getElementById("coordX");
const coordY = document.getElementById("coordY");
const geotiffFile = document.getElementById("geotiffFile");
const geotiffUrl = document.getElementById("geotiffUrl");
const loadUrlBtn = document.getElementById("loadUrlBtn");
const triggerBtn = document.getElementById("triggerBtn");
const proj4Def = document.getElementById("proj4Def");
const loadResultsBtn = document.getElementById("loadResultsBtn");
const drawPatchesBtn = document.getElementById("drawPatchesBtn");
const clearPatchesBtn = document.getElementById("clearPatchesBtn");
const resultsUrlInput = document.getElementById("resultsUrl");
const minioHttpInput = document.getElementById("minioHttp");
const resultsTableBody = document.querySelector("#resultsTable tbody");
const nativeZoomBtn = document.getElementById("nativeZoomBtn");
const pickPointBtn = document.getElementById("pickPointBtn");

let rasterLayer = null;
let rasterMeta = null;
let nativeZoomTarget = null;
let pickEnabled = false;
let lastResultsUrl = "";
let lastResultsRows = [];
let resultsLayer = L.layerGroup({ pane: "patches" }).addTo(map);

function logLine(message) {
  const now = new Date().toLocaleTimeString();
  logOutput.textContent = `[${now}] ${message}\n` + logOutput.textContent;
}

function setStatus(text, tone = "idle") {
  statusChip.textContent = text;
  if (tone === "error") {
    statusChip.style.background = "#ffe4e0";
    statusChip.style.color = "#b42318";
  } else if (tone === "success") {
    statusChip.style.background = "#e9f7ef";
    statusChip.style.color = "#1e6f5c";
  } else {
    statusChip.style.background = "#fff3e6";
    statusChip.style.color = "#9f4b1e";
  }
}

function s3ToHttpUrl(s3Path) {
  if (!s3Path || !s3Path.startsWith("s3://")) return "";
  const parts = s3Path.replace("s3://", "").split("/");
  const bucket = parts.shift();
  const key = parts.join("/");
  const base = minioHttpInput.value.trim().replace(/\/$/, "");
  return `${base}/${bucket}/${key}`;
}

function getDefaultDatasetId() {
  const datasetId = document.getElementById("datasetId").value.trim();
  return datasetId || "cf1eb9ac-859d-42e7-8a23-848be4ba1046";
}

function deriveS3Dir(kind) {
  const datasetId = getDefaultDatasetId();
  return `s3://raster-data/${kind}/${datasetId}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchXcomReturn(baseUrl, token, dagId, dagRunId) {
  const url = `${baseUrl}/api/v1/dags/${dagId}/dagRuns/${dagRunId}/taskInstances/prepare_inputs/xcomEntries/return_value?deserialize=true`;
  const resp = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!resp.ok) {
    throw new Error(`XCom fetch failed: ${resp.status}`);
  }
  const data = await resp.json();
  if (!data || data.value === undefined) {
    throw new Error("XCom response missing value");
  }
  return data.value;
}

async function fetchTaskState(baseUrl, token, dagId, dagRunId, taskId) {
  const url = `${baseUrl}/api/v1/dags/${dagId}/dagRuns/${dagRunId}/taskInstances`;
  const resp = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!resp.ok) {
    if (resp.status === 404) {
      return null;
    }
    throw new Error(`Task state fetch failed: ${resp.status}`);
  }
  const data = await resp.json();
  if (!data || !Array.isArray(data.task_instances)) {
    return null;
  }
  const match = data.task_instances.find((item) => item.task_id === taskId);
  return match ? match.state : null;
}

async function fetchXcomWithRetry(baseUrl, token, dagId, dagRunId) {
  const attempts = 10;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const state = await fetchTaskState(baseUrl, token, dagId, dagRunId, "prepare_inputs");
      if (state === "failed") {
        throw new Error("prepare_inputs failed");
      }
      if (state !== "success") {
        await sleep(1500);
        continue;
      }
      return await fetchXcomReturn(baseUrl, token, dagId, dagRunId);
    } catch (err) {
      if (i === attempts - 1) throw err;
      await sleep(1500);
    }
  }
  return null;
}

function updateResultsTable(rows) {
  resultsTableBody.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "No results yet.";
    tr.appendChild(td);
    resultsTableBody.appendChild(tr);
    return;
  }
  const formatNumber = (value) => (Number.isFinite(value) ? value.toFixed(3) : "");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    [
      "rank",
      "score",
      "match_patch_id",
      "match_min_x",
      "match_min_y",
      "match_max_x",
      "match_max_y",
      "query_x",
      "query_y",
    ].forEach((key) => {
      const td = document.createElement("td");
      if (key === "match_patch_id") {
        td.textContent = row[key] || "";
      } else if (key.endsWith("_x") || key.endsWith("_y") || key === "score") {
        td.textContent = formatNumber(row[key]);
      } else {
        td.textContent = row[key] ?? "";
      }
      tr.appendChild(td);
    });
    resultsTableBody.appendChild(tr);
  });
}

async function fetchCsv(url) {
  if (url.includes(":9001") || url.includes("/download-shared-object/")) {
    throw new Error("MinIO console download URLs (port 9001) are blocked by CORS. Use a presigned URL from the S3 API on port 9000.");
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to load results: ${resp.status}`);
  }
  const text = await resp.text();
  const lines = text.trim().split("\n");
  const headers = lines.shift().split(",").map((h) => h.trim());
  return lines.map((line) => {
    const values = line.split(",").map((value) => value.trim());
    const row = {};
    headers.forEach((h, i) => {
      row[h] = values[i];
    });
    return row;
  });
}

function normalizeResults(rows) {
  const numericFields = [
    "rank",
    "score",
    "query_x",
    "query_y",
    "match_min_x",
    "match_min_y",
    "match_max_x",
    "match_max_y",
  ];
  return rows.map((row) => {
    const normalized = { ...row };
    numericFields.forEach((field) => {
      if (normalized[field] === undefined || normalized[field] === "") return;
      const value = Number.parseFloat(normalized[field]);
      normalized[field] = Number.isFinite(value) ? value : normalized[field];
    });
    return normalized;
  });
}

function clearPatchOverlays() {
  resultsLayer.clearLayers();
}

function drawPatches(rows) {
  if (!rasterMeta) {
    throw new Error("Load a GeoTIFF first to draw patches.");
  }
  clearPatchOverlays();

  let drawn = 0;
  rows.forEach((row) => {
    const minX = row.match_min_x;
    const minY = row.match_min_y;
    const maxX = row.match_max_x;
    const maxY = row.match_max_y;
    if (
      !Number.isFinite(minX) ||
      !Number.isFinite(minY) ||
      !Number.isFinite(maxX) ||
      !Number.isFinite(maxY)
    ) {
      return;
    }

    const sw = fromRasterCoords(minX, minY);
    const ne = fromRasterCoords(maxX, maxY);
    const bounds = L.latLngBounds(sw, ne);
    const rank = Number.isFinite(row.rank) ? row.rank : null;
    const color = rank === 1 ? "#b42318" : rank === 2 ? "#e07a2f" : "#2563eb";
    const rect = L.rectangle(bounds, {
      color,
      weight: 2,
      fillOpacity: 0.08,
    });
    const labelParts = [];
    if (row.match_patch_id) labelParts.push(`patch_id=${row.match_patch_id}`);
    if (row.match_tile_id) labelParts.push(`tile_id=${row.match_tile_id}`);
    if (Number.isFinite(row.score)) labelParts.push(`score=${row.score.toFixed(4)}`);
    if (Number.isFinite(row.rank)) labelParts.push(`rank=${row.rank}`);
    if (labelParts.length) rect.bindTooltip(labelParts.join(" · "));
    rect.addTo(resultsLayer);
    drawn += 1;
  });

  if (!drawn) {
    throw new Error("No patch bounds available in results. Ensure tiles_dir is provided.");
  }
}

async function loadGeoTiff(arrayBuffer) {
  const parser =
    window.parseGeoraster ||
    (window.georaster ? window.georaster.parseGeoraster : null);
  if (!parser) {
    throw new Error("GeoRaster parser not available. Check that georaster scripts loaded.");
  }
  if (rasterLayer) {
    map.removeLayer(rasterLayer);
  }
  const georaster = await parser(arrayBuffer);
  rasterMeta = georaster;
  rasterLayer = new GeoRasterLayer({
    georaster,
    opacity: 0.85,
  });
  rasterLayer.addTo(map);
  map.fitBounds(rasterLayer.getBounds());
  if (georaster && georaster.pixelWidth && georaster.pixelHeight) {
    nativeZoomTarget = {
      center: rasterLayer.getBounds().getCenter(),
      zoom: map.getZoom(),
    };
  }
  logLine(`Loaded GeoTIFF: ${georaster.width}x${georaster.height}, CRS=${georaster.projection || "unknown"}`);
  setStatus("Raster loaded", "success");
}

function getRasterProjection() {
  if (!rasterMeta) return null;
  if (rasterMeta.projection) {
    const raw = String(rasterMeta.projection).trim();
    if (/^\d+$/.test(raw)) {
      return `EPSG:${raw}`;
    }
    return raw;
  }
  if (rasterMeta.srid) return `EPSG:${rasterMeta.srid}`;
  return null;
}

function resolveProjection() {
  const projection = getRasterProjection();
  const customDef = proj4Def.value.trim();
  if (!projection || projection === "EPSG:4326") {
    return "EPSG:4326";
  }

  if (projection === "EPSG:3857") {
    return "EPSG:3857";
  }

  if (projection === "EPSG:3067") {
    proj4.defs(
      "EPSG:3067",
      "+proj=utm +zone=35 +datum=WGS84 +units=m +no_defs +type=crs",
    );
  }

  if (projection.startsWith("EPSG:") && customDef) {
    proj4.defs(projection, customDef);
    return projection;
  }

  if (projection.startsWith("+proj") || projection.includes("+")) {
    return projection;
  }

  return projection;
}

function toRasterCoords(latlng) {
  const projection = getRasterProjection();
  const customDef = proj4Def.value.trim();

  if (!projection || projection === "EPSG:4326") {
    return { x: latlng.lng, y: latlng.lat };
  }

  if (projection === "EPSG:3857") {
    const projected = L.Projection.SphericalMercator.project(latlng);
    return { x: projected.x, y: projected.y };
  }

  if (projection === "EPSG:3067") {
    proj4.defs(
      "EPSG:3067",
      "+proj=utm +zone=35 +datum=WGS84 +units=m +no_defs +type=crs",
    );
  }

  let targetProj = projection;
  if (projection.startsWith("EPSG:") && customDef) {
    proj4.defs(projection, customDef);
    targetProj = projection;
  }

  if (projection.startsWith("+proj") || projection.includes("+")) {
    targetProj = projection;
  }

  if (!proj4.defs(targetProj)) {
    throw new Error(
      `Projection ${projection} is unknown. Provide a proj4 definition to convert coordinates.`,
    );
  }

  const [x, y] = proj4("EPSG:4326", targetProj, [latlng.lng, latlng.lat]);
  return { x, y };
}

function fromRasterCoords(x, y) {
  const projection = resolveProjection();
  if (projection === "EPSG:4326") {
    return L.latLng(y, x);
  }
  if (projection === "EPSG:3857") {
    return L.Projection.SphericalMercator.unproject(L.point(x, y));
  }
  if (!proj4.defs(projection) && !(projection.startsWith("+proj") || projection.includes("+"))) {
    throw new Error(
      `Projection ${projection} is unknown. Provide a proj4 definition to convert coordinates.`,
    );
  }
  const [lng, lat] = proj4(projection, "EPSG:4326", [x, y]);
  return L.latLng(lat, lng);
}

geotiffFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  setStatus("Loading raster...");
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      await loadGeoTiff(reader.result);
    } catch (err) {
      console.error(err);
      setStatus("Failed to load", "error");
      logLine(`Load failed: ${err.message}`);
    }
  };
  reader.readAsArrayBuffer(file);
});

loadUrlBtn.addEventListener("click", async () => {
  if (!geotiffUrl.value.trim()) return;
  setStatus("Loading raster...");
  try {
    const response = await fetch(geotiffUrl.value.trim());
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    await loadGeoTiff(buffer);
  } catch (err) {
    console.error(err);
    setStatus("Failed to load", "error");
    logLine(`Load failed: ${err.message}`);
  }
});

map.on("click", (event) => {
  if (!pickEnabled) {
    return;
  }
  if (!rasterMeta) {
    logLine("Load a GeoTIFF first.");
    return;
  }
  try {
    const { x, y } = toRasterCoords(event.latlng);
    coordX.value = x.toFixed(3);
    coordY.value = y.toFixed(3);
    logLine(`Selected point X=${coordX.value}, Y=${coordY.value}`);
    pickEnabled = false;
    pickPointBtn.textContent = "Enable Point Pick";
  } catch (err) {
    setStatus("Projection error", "error");
    logLine(err.message);
  }
});

async function getToken(baseUrl, username, password) {
  const resp = await fetch(`${baseUrl}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    throw new Error(`Auth failed: ${resp.status}`);
  }
  const data = await resp.json();
  if (!data.access_token) {
    throw new Error("No access_token in response");
  }
  return data.access_token;
}

async function triggerDag() {
  const xVal = coordX.value;
  const yVal = coordY.value;
  if (!xVal || !yVal) {
    throw new Error("Select a point on the map first.");
  }

  const baseUrl = document.getElementById("airflowUrl").value.trim();
  const user = document.getElementById("airflowUser").value.trim();
  const pass = document.getElementById("airflowPassword").value;
  const dagId = document.getElementById("dagId").value.trim();
  const datasetId = getDefaultDatasetId();
  const topK = parseInt(document.getElementById("topK").value, 10) || 5;
  const tilesDir = deriveS3Dir("tiles");
  const embeddingsDir = deriveS3Dir("embeddings");

  const token = await getToken(baseUrl, user, pass);
  const conf = {
    top_k: topK,
    query_coords: [{ x: parseFloat(xVal), y: parseFloat(yVal) }],
  };
  if (datasetId) conf.dataset_id = datasetId;
  conf.tiles_dir = tilesDir;
  conf.embeddings_dir = embeddingsDir;

  const payload = {
    dag_run_id: `ui-${Date.now()}`,
    logical_date: new Date().toISOString(),
    conf,
  };

  const resp = await fetch(`${baseUrl}/api/v2/dags/${dagId}/dagRuns`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Trigger failed: ${resp.status} ${text}`);
  }
  const data = await resp.json();
  const dagRunId = data.dag_run_id;
  logLine(`Triggered DAG ${dagId} run ${dagRunId}`);
  setStatus("DAG triggered", "success");

  try {
    const xcomValue = await fetchXcomWithRetry(baseUrl, token, dagId, dagRunId);
    if (xcomValue && (xcomValue.output_path || xcomValue.embeddings_dir)) {
      let csvPath = "";
      if (xcomValue.output_path) {
        csvPath = xcomValue.output_path.replace(/\.parquet$/i, ".csv");
      } else {
        const resolvedBase = xcomValue.embeddings_dir.endsWith("/")
          ? xcomValue.embeddings_dir.slice(0, -1)
          : xcomValue.embeddings_dir;
        csvPath = `${resolvedBase}/similarity_results.csv`;
      }
      lastResultsUrl = s3ToHttpUrl(csvPath);
      if (lastResultsUrl) {
        resultsUrlInput.value = lastResultsUrl;
      }
      if (xcomValue.dataset_id) {
        logLine(`Resolved dataset_id=${xcomValue.dataset_id}`);
      }
      return;
    }
  } catch (err) {
    logLine(`XCom lookup failed: ${err.message}`);
  }

  const fallbackBase = embeddingsDir.endsWith("/") ? embeddingsDir.slice(0, -1) : embeddingsDir;
  const fallbackCsv = `${fallbackBase}/similarity_results.csv`;
  lastResultsUrl = s3ToHttpUrl(fallbackCsv);
  if (lastResultsUrl) {
    resultsUrlInput.value = lastResultsUrl;
  }
}

triggerBtn.addEventListener("click", async () => {
  setStatus("Triggering...");
  try {
    await triggerDag();
  } catch (err) {
    console.error(err);
    setStatus("Trigger failed", "error");
    logLine(err.message);
  }
});

loadResultsBtn.addEventListener("click", async () => {
  const url = resultsUrlInput.value.trim() || lastResultsUrl;
  if (!url) {
    logLine("Provide a results CSV URL first.");
    return;
  }
  setStatus("Loading results...");
  try {
    const rows = normalizeResults(await fetchCsv(url));
    lastResultsRows = rows;
    updateResultsTable(rows);
    setStatus("Results loaded", "success");
    logLine(`Loaded ${rows.length} results from ${url}`);
  } catch (err) {
    console.error(err);
    setStatus("Results failed", "error");
    logLine(err.message);
  }
});

drawPatchesBtn.addEventListener("click", async () => {
  const url = resultsUrlInput.value.trim() || lastResultsUrl;
  setStatus("Drawing patches...");
  try {
    if (!lastResultsRows.length) {
      if (!url) {
        throw new Error("Provide a results CSV URL first.");
      }
      const rows = normalizeResults(await fetchCsv(url));
      lastResultsRows = rows;
      updateResultsTable(rows);
      logLine(`Loaded ${rows.length} results from ${url}`);
    }
    drawPatches(lastResultsRows);
    setStatus("Patches drawn", "success");
    logLine("Patch polygons rendered.");
  } catch (err) {
    console.error(err);
    setStatus("Draw failed", "error");
    logLine(err.message);
  }
});

clearPatchesBtn.addEventListener("click", () => {
  clearPatchOverlays();
  logLine("Cleared patch overlays.");
});

nativeZoomBtn.addEventListener("click", () => {
  zoomToNativeResolution();
});

pickPointBtn.addEventListener("click", () => {
  pickEnabled = !pickEnabled;
  pickPointBtn.textContent = pickEnabled ? "Click a Point..." : "Enable Point Pick";
  logLine(pickEnabled ? "Point picking enabled." : "Point picking disabled.");
});

logLine("Ready. Load a GeoTIFF to begin.");
function zoomToNativeResolution() {
  if (!rasterLayer || !rasterMeta) {
    logLine("Load a GeoTIFF first.");
    return;
  }

  const mapSize = map.getSize();
  const bounds = rasterLayer.getBounds();
  const targetResX = bounds.getEast() - bounds.getWest();
  const targetResY = bounds.getNorth() - bounds.getSouth();
  const pixelWidth = rasterMeta.pixelWidth || rasterMeta.pixelWidthInMeters || 0;
  const pixelHeight = rasterMeta.pixelHeight || rasterMeta.pixelHeightInMeters || 0;

  if (!pixelWidth || !pixelHeight) {
    logLine("Raster pixel size unavailable for native zoom.");
    return;
  }

  const currentResX = targetResX / mapSize.x;
  const currentResY = targetResY / mapSize.y;
  const scaleX = currentResX / pixelWidth;
  const scaleY = currentResY / Math.abs(pixelHeight);
  const scale = Math.max(scaleX, scaleY);
  const zoomDelta = Math.log2(scale);
  const targetZoom = map.getZoom() + zoomDelta;

  map.setView(bounds.getCenter(), targetZoom, { animate: true });
  logLine("Zoomed to native resolution.");
}
