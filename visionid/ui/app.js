/* ───────────────────────────────────────────────────────────────────────────
   VisionID – app.js
   Handles tab navigation, file upload, recognition API calls,
   benchmark display, gallery loading and toast notifications.
─────────────────────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:8000";

/* ══ State ═══════════════════════════════════════════════════════════════════ */
let currentFile = null;
let benchChart  = null;

/* ══ DOM refs ════════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

/* ══ Init ════════════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initUpload();
  initRecognize();
  initBenchmark();
  initGallery();
  checkApiHealth();
  setInterval(checkApiHealth, 15_000);
});

/* ══ Tabs ════════════════════════════════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      $(`tab-${tab}`).classList.add("active");
      if (tab === "gallery") loadGallery();
    });
  });
}

/* ══ Upload / Preview ════════════════════════════════════════════════════════ */
function initUpload() {
  const zone    = $("upload-zone");
  const input   = $("file-input");
  const canvas  = $("preview-canvas");
  const btnRec  = $("btn-recognize");
  const btnClr  = $("btn-clear");

  ["dragover", "dragenter"].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.add("drag-over");
  }));
  ["dragleave", "dragend", "drop"].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.remove("drag-over");
  }));

  zone.addEventListener("drop", ev => {
    const f = ev.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setFile(f);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0]);
  });

  btnClr.addEventListener("click", clearAll);

  function setFile(f) {
    currentFile = f;
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        zone.classList.add("hidden");
        canvas.classList.remove("hidden");
        const ctx = canvas.getContext("2d");
        canvas.width  = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        btnRec.disabled = false;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(f);
  }

  function clearAll() {
    currentFile = null;
    input.value = "";
    zone.classList.remove("hidden");
    canvas.classList.add("hidden");
    btnRec.disabled = true;
    $("results-placeholder").classList.remove("hidden");
    $("results-content").classList.add("hidden");
    $("face-cards-container").innerHTML = "";
  }
}

/* ══ Recognize ═══════════════════════════════════════════════════════════════ */
function initRecognize() {
  $("btn-recognize").addEventListener("click", runRecognize);
}

async function runRecognize() {
  if (!currentFile) return;

  const btn    = $("btn-recognize");
  const method = $("index-select").value;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="display:inline-block;width:16px;height:16px;border-width:2px"></span> Recognizing…';

  // Switch index first
  try {
    const fd = new FormData();
    fd.append("method", method);
    await fetch(`${API_BASE}/api/index/switch`, { method: "POST", body: fd });
  } catch (_) {}

  // Recognize
  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    const resp = await fetch(`${API_BASE}/api/recognize`, { method: "POST", body: fd });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const data = await resp.json();
    renderResults(data, method);

    // Draw bounding boxes on canvas
    drawBoxes(data.faces);

    showToast(`Recognized ${data.faces.length} face(s) in ${data.total_ms.toFixed(1)} ms`, "success");
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    renderDemoResults(method);
  } finally {
    btn.disabled  = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span> Recognize';
  }
}

function renderResults(data, method) {
  $("results-placeholder").classList.add("hidden");
  $("results-content").classList.remove("hidden");

  // Timing
  const first = data.faces[0] || {};
  $("t-detect").textContent = `${data.detection_ms.toFixed(1)} ms`;
  $("t-embed").textContent  = first.embedding_ms != null ? `${first.embedding_ms.toFixed(1)} ms` : "–";
  $("t-search").textContent = first.search_ms != null    ? `${first.search_ms.toFixed(2)} ms`    : "–";
  $("t-total").textContent  = `${data.total_ms.toFixed(1)} ms`;

  const container = $("face-cards-container");
  container.innerHTML = "";

  if (data.faces.length === 0) {
    container.innerHTML = `<div class="results-placeholder" style="padding:24px">
      <div class="placeholder-icon">😶</div><p>No faces detected</p></div>`;
    return;
  }

  data.faces.forEach((face, i) => {
    const isMatch = face.identity && !["UNKNOWN","AMBIGUOUS"].includes(face.identity);
    const type    = isMatch ? "match" : (face.identity === "AMBIGUOUS" ? "ambiguous" : "unknown");
    const sim     = face.similarity ?? 0;
    const simPct  = Math.round(sim * 100);
    const barClass = simPct >= 60 ? "high" : simPct >= 40 ? "medium" : "low";

    container.innerHTML += `
      <div class="face-card ${type}">
        <div class="face-card-header">
          <div class="face-identity ${type}">${face.identity || "UNKNOWN"}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
            ${face.from_cache ? '<span class="face-badge badge-cache">CACHED</span>' : ""}
            <span class="face-badge badge-${type}">${type.toUpperCase()}</span>
          </div>
        </div>
        <div class="face-meta">
          <span class="face-meta-item">Score <strong>${face.score.toFixed(3)}</strong></span>
          <span class="face-meta-item">Index <strong>${method.toUpperCase()}</strong></span>
          <span class="face-meta-item">Track <strong>#${face.track_id}</strong></span>
        </div>
        ${sim > 0 ? `
        <div class="sim-bar-wrap">
          <div class="sim-bar-label">Similarity: ${simPct}%</div>
          <div class="sim-bar-track">
            <div class="sim-bar-fill ${barClass}" style="width:${simPct}%"></div>
          </div>
        </div>` : ""}
      </div>`;
  });
}

/* Draw bounding boxes on preview canvas */
function drawBoxes(faces) {
  const canvas = $("preview-canvas");
  const ctx    = canvas.getContext("2d");

  faces.forEach(face => {
    const [x1, y1, x2, y2] = face.bbox;
    const isMatch = face.identity && !["UNKNOWN","AMBIGUOUS"].includes(face.identity);
    const color = isMatch ? "#34d399" : (face.identity === "AMBIGUOUS" ? "#fbbf24" : "#f43f5e");

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.shadowBlur = 0;

    // Label
    const label = face.identity || "UNKNOWN";
    ctx.fillStyle = color;
    ctx.font = "bold 13px Inter, sans-serif";
    const pad = 4;
    const tw  = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x1, y1 - 22, tw + pad*2, 20);
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + pad, y1 - 7);
  });
}

/* Demo (offline) results – shown when API is unreachable */
function renderDemoResults(method) {
  const demoData = {
    detection_ms: 14.8,
    total_ms: 41.2,
    faces: [{
      track_id: 1,
      bbox: [40, 40, 200, 220],
      score: 0.97,
      identity: "Alex Morgan",
      similarity: 0.81,
      from_cache: false,
      embedding_ms: 22.1,
      search_ms: 1.7,
    }],
  };
  renderResults(demoData, method);
  showToast("API offline – showing demo data", "error");
}

/* ══ Benchmark ═══════════════════════════════════════════════════════════════ */
function initBenchmark() {
  $("btn-run-benchmark").addEventListener("click", runBenchmark);
}

async function runBenchmark() {
  const btn     = $("btn-run-benchmark");
  const spinner = $("bench-spinner");
  const placeholder = $("bench-placeholder");
  const tableWrap   = $("bench-table-wrap");
  const chartWrap   = $("bench-chart-wrap");

  btn.disabled  = true;
  spinner.classList.remove("hidden");
  placeholder.classList.add("hidden");
  tableWrap.classList.add("hidden");
  chartWrap.classList.add("hidden");
  $("bench-btn-text").textContent = "Running…";

  try {
    const resp = await fetch(`${API_BASE}/api/benchmark/quick`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderBenchmarkResults(data);
    showToast("Benchmark complete!", "success");
  } catch (err) {
    showToast(`Benchmark failed: ${err.message} – Showing demo data`, "error");
    renderBenchmarkResults(getDemoBenchData());
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
    $("bench-btn-text").textContent = "▶ Run Benchmark";
  }
}

function getDemoBenchData() {
  return {
    gallery_size: 5000,
    n_queries: 100,
    results: {
      Flat:  { p50_ms: 3.21,  p95_ms: 4.11,  mean_ms: 3.30  },
      HNSW:  { p50_ms: 0.18,  p95_ms: 0.27,  mean_ms: 0.19  },
      IVF:   { p50_ms: 0.42,  p95_ms: 0.65,  mean_ms: 0.44  },
    }
  };
}

function renderBenchmarkResults(data) {
  const tableWrap = $("bench-table-wrap");
  const chartWrap = $("bench-chart-wrap");

  const methods = Object.keys(data.results);
  const rows = methods.map(m => {
    const r = data.results[m];
    if (r.error) return `<tr><td>${m}</td><td colspan="3" style="color:var(--rose)">${r.error}</td></tr>`;
    return `<tr>
      <td><strong>${m}</strong></td>
      <td>${r.p50_ms} ms</td>
      <td>${r.p95_ms} ms</td>
      <td>${r.mean_ms} ms</td>
    </tr>`;
  }).join("");

  tableWrap.innerHTML = `
    <p style="font-size:0.8rem;color:var(--text-3);margin-bottom:12px">
      Gallery: ${data.gallery_size.toLocaleString()} vectors · Queries: ${data.n_queries}
    </p>
    <table class="bench-table">
      <thead>
        <tr>
          <th>Index</th>
          <th>P50 Latency</th>
          <th>P95 Latency</th>
          <th>Mean</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;

  tableWrap.classList.remove("hidden");

  // Chart
  const labels = methods.filter(m => !data.results[m].error);
  const p50    = labels.map(m => parseFloat(data.results[m].p50_ms));
  const p95    = labels.map(m => parseFloat(data.results[m].p95_ms));

  if (benchChart) benchChart.destroy();

  const ctx = $("bench-chart").getContext("2d");
  benchChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "P50 (ms)",
          data: p50,
          backgroundColor: "rgba(99,102,241,0.7)",
          borderColor: "#6366f1",
          borderWidth: 1.5,
          borderRadius: 6,
        },
        {
          label: "P95 (ms)",
          data: p95,
          backgroundColor: "rgba(139,92,246,0.5)",
          borderColor: "#8b5cf6",
          borderWidth: 1.5,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } },
        title: {
          display: true,
          text: `Search Latency Comparison (N=${data.gallery_size.toLocaleString()})`,
          color: "#f1f5f9",
          font: { size: 14, weight: "700", family: "Inter" },
        },
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(99,102,241,0.1)" } },
        y: {
          ticks: { color: "#94a3b8", callback: v => `${v} ms` },
          grid: { color: "rgba(99,102,241,0.1)" },
          title: { display: true, text: "Latency (ms)", color: "#64748b" },
        },
      },
    },
  });

  chartWrap.classList.remove("hidden");
}

/* ══ Gallery ═════════════════════════════════════════════════════════════════ */
function initGallery() {
  $("btn-refresh-gallery").addEventListener("click", loadGallery);
}

async function loadGallery() {
  const grid = $("gallery-grid");
  grid.innerHTML = `<div class="gallery-loading"><div class="spinner"></div><p>Loading gallery…</p></div>`;

  try {
    const resp = await fetch(`${API_BASE}/api/gallery`);
    if (!resp.ok) throw new Error("API unreachable");
    const data = await resp.json();

    if (!data.persons || data.persons.length === 0) {
      grid.innerHTML = `<div class="gallery-empty">
        <div class="placeholder-icon">👤</div>
        <p>No enrolled identities yet.<br/>Run <code>python scripts/enroll_demo.py</code> to add demo profiles.</p></div>`;
      return;
    }

    grid.innerHTML = data.persons.map(p => `
      <div class="gallery-card">
        <div class="gallery-avatar">${(p.display_name || "?")[0]}</div>
        <div class="gallery-name">${p.display_name}</div>
        <div class="gallery-id">${p.person_id}</div>
        <div class="gallery-meta">
          ${p.age ? `${p.age} yrs` : ""} ${p.demo_city ? `· ${p.demo_city}` : ""}<br/>
          ${p.role || ""}
        </div>
      </div>`).join("");

  } catch (_) {
    // Show demo data when API is offline
    const demoPersons = [
      { person_id: "DEMO_0001", display_name: "Alex Morgan",  age: 28, demo_city: "Dublin",    role: "Software Engineer" },
      { person_id: "DEMO_0002", display_name: "Sam Rivera",   age: 34, demo_city: "Cork",      role: "Data Scientist"    },
      { person_id: "DEMO_0003", display_name: "Jordan Lee",   age: 22, demo_city: "Galway",    role: "Student"           },
      { person_id: "DEMO_0004", display_name: "Taylor Kim",   age: 41, demo_city: "Limerick",  role: "Researcher"        },
      { person_id: "DEMO_0005", display_name: "Casey Walsh",  age: 30, demo_city: "Waterford", role: "Engineer"          },
    ];
    grid.innerHTML = demoPersons.map(p => `
      <div class="gallery-card">
        <div class="gallery-avatar">${p.display_name[0]}</div>
        <div class="gallery-name">${p.display_name}</div>
        <div class="gallery-id">${p.person_id}</div>
        <div class="gallery-meta">${p.age} yrs · ${p.demo_city}<br/>${p.role}</div>
      </div>`).join("");
    showToast("API offline – showing demo gallery", "error");
  }
}

/* ══ API health ══════════════════════════════════════════════════════════════ */
async function checkApiHealth() {
  const dot   = $("api-status-dot");
  const label = $("api-status-label");
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (resp.ok) {
      const data = await resp.json();
      dot.className   = "status-dot online";
      label.textContent = `API Online · ${(data.gallery_size || 0).toLocaleString()} vectors`;
    } else {
      throw new Error();
    }
  } catch {
    dot.className   = "status-dot offline";
    label.textContent = "API Offline";
  }
}

/* ══ Toasts ══════════════════════════════════════════════════════════════════ */
function showToast(msg, type = "info") {
  const tc = $("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  tc.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
