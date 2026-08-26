const positionText = document.querySelector("#position");
const result = document.querySelector("#result");
const accuracyText = document.querySelector("#accuracy");
const resultLabel = document.querySelector("#result-label");
const confidenceEl = document.querySelector("#confidence");
const confidenceLabelEl = document.querySelector("#confidence-label");
const confidenceScoreEl = document.querySelector("#confidence-score");
const confidenceGaugeFillEl = document.querySelector("#confidence-gauge-fill");
const confidenceGaugeThresholdEl = document.querySelector("#confidence-gauge-threshold");
const confidenceSparkLineEl = document.querySelector("#confidence-spark-line");
const confidenceSparkThresholdEl = document.querySelector("#confidence-spark-threshold");
const CONFIDENCE_HISTORY_MS = 30000;
const DEFAULT_MIN_ACCURACY = 60;
let confidenceHistory = [];
const status = document.querySelector("#status");
const debugButton = document.querySelector("#debug");
const detectorToggle = document.querySelector("#detector-toggle");
const eventLogBody = document.querySelector("#event-log-body");
const maxLogRows = 100;
const queryParams = new URLSearchParams(window.location.search);
const mockMode = queryParams.get("mock") === "1";
// 62.232360729710024, 25.736908900974168
// Nuppitie
const lat = 62.300957299457714;
const lng = 25.785837367223287;
const DEFAULT_CENTER = { lat: lat, lng: lng };

let map;
let sensorMarker;
let hideTimer;
let markerImpactTimer;
let mockTimer;

const sensorIcon = L.divIcon({
  className: "sensor-marker-wrap",
  html: "<div class=\"sensor-marker\"><span class=\"sensor-marker-core\"></span></div>",
  iconSize: [34, 34],
  iconAnchor: [17, 17],
});

// Dummy sensors along a line through the real one, forming an evenly-spaced
// "fence" -- for demo purposes only, not backed by any real detector.
// Counts are independent per side; FENCE_BEARING_DEGREES is the compass
// bearing (0=N, 90=E, 180=S, 270=W) of the "positive" side -- the other
// side runs the opposite direction automatically.
const DUMMY_SENSORS_NW = 5;
const DUMMY_SENSORS_SE = 3;
const DUMMY_SENSOR_SPACING_METERS = 150;
const FENCE_BEARING_DEGREES = 135; // SE; the NW side is just the reverse of this

const dummySensorIcon = L.divIcon({
  className: "sensor-marker-wrap",
  html: "<div class=\"sensor-marker-dummy\"><span class=\"sensor-marker-dummy-core\"></span></div>",
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

// meters may be negative to go the opposite way along the same bearing.
function offsetByBearingMeters(lat, lng, bearingDegrees, meters) {
  const bearingRad = (bearingDegrees * Math.PI) / 180;
  const metersPerDegreeLat = 111320;
  const metersPerDegreeLng = 111320 * Math.cos((lat * Math.PI) / 180);
  return {
    lat: lat + (meters * Math.cos(bearingRad)) / metersPerDegreeLat,
    lng: lng + (meters * Math.sin(bearingRad)) / metersPerDegreeLng,
  };
}

function dummySensorOffsets() {
  const offsets = [];
  for (let i = DUMMY_SENSORS_NW; i >= 1; i--) offsets.push(-i * DUMMY_SENSOR_SPACING_METERS);
  for (let i = 1; i <= DUMMY_SENSORS_SE; i++) offsets.push(i * DUMMY_SENSOR_SPACING_METERS);
  return offsets;
}

// No GPS at the demo site: the real sensor gets dragged to the right spot
// on the map by eye, so the dummy "fence" needs to follow it (it's defined
// relative to the real sensor, not fixed to DEFAULT_CENTER), and the chosen
// position needs to survive a page reload since there's no backend to save
// it to. localStorage covers both -- per-browser, but that's fine since the
// demo always runs from the same laptop.
const SENSOR_POSITION_STORAGE_KEY = "detector-sensor-position";
let dummyMarkers = [];

function loadSavedSensorPosition() {
  try {
    if (queryParams.get("resetPosition") === "1") {
      localStorage.removeItem(SENSOR_POSITION_STORAGE_KEY);
      return null;
    }
    const raw = localStorage.getItem(SENSOR_POSITION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Number.isFinite(parsed.lat) && Number.isFinite(parsed.lng)) return parsed;
  } catch (e) {
    // localStorage unavailable (private browsing etc.) -- just use the default.
  }
  return null;
}

function saveSensorPosition(lat, lng) {
  try {
    localStorage.setItem(SENSOR_POSITION_STORAGE_KEY, JSON.stringify({ lat, lng }));
  } catch (e) {
    // Persistence is a convenience, not required for the page to work.
  }
}

function updateDummySensorPositions(centerLat, centerLng) {
  dummySensorOffsets().forEach((meters, index) => {
    const pos = offsetByBearingMeters(centerLat, centerLng, FENCE_BEARING_DEGREES, meters);
    if (dummyMarkers[index]) {
      dummyMarkers[index].setLatLng([pos.lat, pos.lng]);
    } else if (map) {
      dummyMarkers[index] = L.marker([pos.lat, pos.lng], {
        icon: dummySensorIcon,
        title: `Sensor ${String(index + 2).padStart(2, "0")} (demo)`,
        interactive: false,
      }).addTo(map);
    }
  });
}

function getAccuracyTone(accuracyValue) {
  const accuracy = Number(accuracyValue);
  if (accuracy >= 95) return { color: "#e60000", className: "impact-high" };
  if (accuracy >= 85) return { color: "#e63946", className: "impact-mid" };
  return { color: "#ff5a3c", className: "impact-low" };
}

function getConfidenceColor(score, threshold) {
  if (score >= threshold) return "#b22222";
  if (score >= threshold - 20) return "#d98a1f";
  return "#138a62";
}

function setSensorPosition(lat, lng, pan = false) {
  if (!sensorMarker) return;
  sensorMarker.setLatLng([lat, lng]);
  if (pan && map) map.panTo([lat, lng], { animate: true, duration: 0.5 });
  positionText.textContent = `Position ${lat.toFixed(4)}, ${lng.toFixed(4)}`;
  saveSensorPosition(lat, lng);
  updateDummySensorPositions(lat, lng);
}

function addEventLogRow(event) {
  if (!eventLogBody) return;
  const accuracy = Number(event.accuracy);
  const accuracyValue = Number.isFinite(accuracy) ? `${accuracy.toFixed(1)}%` : "-";
  const detector = event.detector || event.label || "Detection";
  // Same-session demo log: time-of-day is what matters, date would just
  // crowd the narrow side-column table.
  const time = new Date().toLocaleTimeString("fi-FI", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const row = document.createElement("tr");
  row.innerHTML = `<td>${time}</td><td>${accuracyValue}</td><td>${detector}</td>`;
  eventLogBody.prepend(row);

  while (eventLogBody.rows.length > maxLogRows) {
    eventLogBody.deleteRow(eventLogBody.rows.length - 1);
  }
}

function showMapImpact(lat, lng, accuracyValue) {
  if (!map) return;
  const tone = getAccuracyTone(accuracyValue);
  const markerEl = sensorMarker?.getElement()?.querySelector(".sensor-marker");

  if (markerEl) {
    markerEl.classList.remove("sensor-impact-high", "sensor-impact-mid", "sensor-impact-low");
    markerEl.classList.add(`sensor-impact-${tone.className.split("-")[1]}`);
    clearTimeout(markerImpactTimer);
    markerImpactTimer = setTimeout(() => {
      markerEl.classList.remove("sensor-impact-high", "sensor-impact-mid", "sensor-impact-low");
    }, 1400);
  }

  const flash = L.circleMarker([lat, lng], {
    radius: 26,
    color: tone.color,
    weight: 5,
    fillColor: tone.color,
    fillOpacity: 0.7,
    opacity: 1,
    pane: "markerPane",
  }).addTo(map);

  const ring = L.circle([lat, lng], {
    radius: 55,
    color: tone.color,
    weight: 7,
    opacity: 1,
    fillColor: tone.color,
    fillOpacity: 0.18,
    pane: "overlayPane",
  }).addTo(map);

  const ring2 = L.circle([lat, lng], {
    radius: 55,
    color: tone.color,
    weight: 4,
    opacity: 0.9,
    fillOpacity: 0,
    pane: "overlayPane",
  }).addTo(map);

  const start = performance.now();
  const duration = 2000;

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(1, elapsed / duration);
    const eased = 1 - (1 - progress) * (1 - progress);

    ring.setRadius(55 + eased * 620);
    ring.setStyle({ opacity: 1 - eased, fillOpacity: 0.18 * (1 - eased) });

    // Toinen rengas seuraa hieman jäljessä voimakkaamman aaltoefektin luomiseksi.
    const eased2 = Math.max(0, eased - 0.18);
    ring2.setRadius(55 + eased2 * 620);
    ring2.setStyle({ opacity: 0.9 * (1 - eased) });

    flash.setStyle({ opacity: 1 - eased, fillOpacity: 0.7 - eased * 0.7 });

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      map.removeLayer(ring);
      map.removeLayer(ring2);
      map.removeLayer(flash);
    }
  }

  requestAnimationFrame(tick);
}

function renderConfidenceSpark(threshold) {
  const now = performance.now();
  confidenceHistory = confidenceHistory.filter(p => now - p.t <= CONFIDENCE_HISTORY_MS);

  const width = 200;
  const height = 36;
  const thresholdY = height - (threshold / 100) * height;
  confidenceSparkThresholdEl.setAttribute("y1", thresholdY.toFixed(1));
  confidenceSparkThresholdEl.setAttribute("y2", thresholdY.toFixed(1));
  confidenceSparkThresholdEl.removeAttribute("hidden");

  if (confidenceHistory.length < 2) {
    confidenceSparkLineEl.setAttribute("points", "");
    return;
  }

  const oldest = confidenceHistory[0].t;
  const span = Math.max(1, now - oldest);
  const points = confidenceHistory
    .map(p => {
      const x = (width * (p.t - oldest)) / span;
      const y = height - (Math.max(0, Math.min(100, p.score)) / 100) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  confidenceSparkLineEl.setAttribute("points", points);
}

function showConfidence(event) {
  const isDrone = Boolean(event.is_drone);
  const score = Number(event.confidence);
  const threshold = Number.isFinite(Number(event.min_accuracy))
    ? Number(event.min_accuracy)
    : DEFAULT_MIN_ACCURACY;

  confidenceLabelEl.textContent = isDrone ? "DRONE" : "NO DRONE";
  confidenceScoreEl.textContent = `${score.toFixed(1)}% probability of drone`;
  confidenceEl.classList.toggle("confidence-drone", isDrone);
  confidenceEl.classList.toggle("confidence-clear", !isDrone);

  const clamped = Math.max(0, Math.min(100, score));
  const color = getConfidenceColor(clamped, threshold);
  confidenceGaugeFillEl.style.width = `${clamped}%`;
  confidenceGaugeFillEl.style.backgroundColor = color;
  confidenceGaugeThresholdEl.style.left = `${Math.max(0, Math.min(100, threshold))}%`;
  confidenceGaugeThresholdEl.removeAttribute("hidden");

  confidenceHistory.push({ t: performance.now(), score: clamped });
  renderConfidenceSpark(threshold);
}

function showDetection(event) {
  let targetLat = sensorMarker.getLatLng().lat;
  let targetLng = sensorMarker.getLatLng().lng;

  if (typeof event.latitude === "number" && typeof event.longitude === "number") {
    targetLat = event.latitude;
    targetLng = event.longitude;
    setSensorPosition(targetLat, targetLng, true);
  }

  const accuracyValue = Number(event.accuracy);
  const accuracyTextValue = Number.isFinite(accuracyValue) ? accuracyValue.toFixed(1) : "-";

  showMapImpact(targetLat, targetLng, accuracyValue);
  accuracyText.textContent = `Accuracy: ${accuracyTextValue}%`;
  resultLabel.textContent = (event.label || "Detection").toUpperCase();
  addEventLogRow(event);

  result.classList.add("show");
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => result.classList.remove("show"), 6000);
}

function getRandomInRange(min, max) {
  return Math.random() * (max - min) + min;
}

function createMockDetection() {
  const detectorOptions = ["sensor-01", "yolov8", "yolo-nano", "edge-tflite"];
  const labelOptions = ["person", "dog", "car", "bird", "detection"];
  const base = sensorMarker.getLatLng();

  return {
    accuracy: Number(getRandomInRange(84, 99.9).toFixed(1)),
    label: labelOptions[Math.floor(Math.random() * labelOptions.length)],
    detector: detectorOptions[Math.floor(Math.random() * detectorOptions.length)],
    latitude: Number((base.lat + getRandomInRange(-0.00045, 0.00045)).toFixed(7)),
    longitude: Number((base.lng + getRandomInRange(-0.00045, 0.00045)).toFixed(7)),
  };
}

function startMockStream() {
  status.innerHTML = "<span></span> SIMULATION ONLINE";
  status.classList.add("online");

  showConfidence({ is_drone: false, confidence: 0 });
  showDetection(createMockDetection());
  clearInterval(mockTimer);
  mockTimer = setInterval(() => {
    const isDrone = Math.random() > 0.55;
    showConfidence({ is_drone: isDrone, confidence: isDrone ? Number(getRandomInRange(55, 98).toFixed(1)) : 0 });
    showDetection(createMockDetection());
  }, 2600);
}

function connect() {
  if (mockMode) {
    startMockStream();
    return;
  }

  const source = new EventSource("api/events");
  source.addEventListener("connected", () => {
    status.innerHTML = "<span></span> SYSTEM ONLINE";
    status.classList.add("online");
  });
  source.addEventListener("detection", message => showDetection(JSON.parse(message.data)));
  source.addEventListener("confidence", message => showConfidence(JSON.parse(message.data)));
  source.onerror = () => {
    status.innerHTML = "<span></span> RECONNECTING";
    status.classList.remove("online");
  };
}

function initMap() {
  const startCenter = loadSavedSensorPosition() || DEFAULT_CENTER;

  map = L.map("leaflet-map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([startCenter.lat, startCenter.lng], 17);

  // Tiles are proxied and cached by the Pi; works offline after first online load.
  L.tileLayer("/tiles/{z}/{x}/{y}.png", {
    maxZoom: 20,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  sensorMarker = L.marker([startCenter.lat, startCenter.lng], {
    draggable: true,
    title: "Sensor 01",
    icon: sensorIcon,
  }).addTo(map);

  sensorMarker.on("drag", e => {
    const pos = e.target.getLatLng();
    positionText.textContent = `Position ${pos.lat.toFixed(4)}, ${pos.lng.toFixed(4)}`;
    updateDummySensorPositions(pos.lat, pos.lng);
  });

  sensorMarker.on("dragend", e => {
    const pos = e.target.getLatLng();
    setSensorPosition(pos.lat, pos.lng, false);
  });

  // Creates the dummy markers on first call, and keeps them in sync with
  // the real sensor's position (dragged, loaded from storage, or moved by a
  // detection event) on every call after.
  setSensorPosition(startCenter.lat, startCenter.lng, false);
}

fetch("api/config").then(r => r.json()).then(config => {
  debugButton.hidden = !config.debug;
  detectorToggle.hidden = !config.showStopDetector;
});
debugButton.addEventListener("click", () => fetch("api/debug/detection", { method: "POST" }));

function setDetectorToggleState(running) {
  detectorToggle.textContent = running ? "STOP DETECTOR" : "START DETECTOR";
  detectorToggle.classList.toggle("running", running);
  detectorToggle.disabled = false;
}

function refreshDetectorStatus() {
  fetch("api/detector/status")
    .then(r => r.json())
    .then(data => setDetectorToggleState(data.running));
}

detectorToggle.addEventListener("click", () => {
  const isRunning = detectorToggle.classList.contains("running");
  detectorToggle.disabled = true;
  fetch(`api/detector/${isRunning ? "stop" : "start"}`, { method: "POST" })
    .then(() => refreshDetectorStatus())
    .catch(() => { detectorToggle.disabled = false; });
});

initMap();
connect();
refreshDetectorStatus();
