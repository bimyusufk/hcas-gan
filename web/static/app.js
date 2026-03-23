const imageInput = document.getElementById('imageInput');
const annotationInput = document.getElementById('annotationInput');
const canvas = document.getElementById('canvas');
const canvasWrap = document.getElementById('canvasWrap');
const ctx = canvas.getContext('2d');
const closePolygonBtn = document.getElementById('closePolygonBtn');
const clearBtn = document.getElementById('clearBtn');
const inferBtn = document.getElementById('inferBtn');
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomResetBtn = document.getElementById('zoomResetBtn');
const zoomLabel = document.getElementById('zoomLabel');
const composeModeEl = document.getElementById('composeMode');
const tileSizeEl = document.getElementById('tileSize');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');

const outEnv = document.getElementById('outEnv');
const outMask = document.getElementById('outMask');
const outPattern = document.getElementById('outPattern');
const outPatternCanvas = document.getElementById('outPatternCanvas');
const outComposite = document.getElementById('outComposite');

let uploadedFile = null;
let baseImage = null;
let points = [];
let closed = false;
let zoom = 1;
let annotationPayload = null;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4.0;
const ZOOM_STEP = 0.1;

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateZoomLabel() {
  if (zoomLabel) {
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  }
}

function setCanvasDisplaySize() {
  if (!canvas || !baseImage) return;
  canvas.style.width = `${Math.round(canvas.width * zoom)}px`;
  canvas.style.height = `${Math.round(canvas.height * zoom)}px`;
  updateZoomLabel();
}

function setZoom(nextZoom) {
  zoom = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
  setCanvasDisplaySize();
}

function mapClientToCanvas(e) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }
  const x = ((e.clientX - rect.left) * canvas.width) / rect.width;
  const y = ((e.clientY - rect.top) * canvas.height) / rect.height;
  return {
    x: clamp(x, 0, canvas.width),
    y: clamp(y, 0, canvas.height),
  };
}

function applyAnnotationFromPayload() {
  if (!annotationPayload || !baseImage) return;

  let payload = annotationPayload;
  let rawPoints = [];
  let srcW = null;
  let srcH = null;

  if (Array.isArray(payload)) {
    rawPoints = payload;
  } else if (payload && typeof payload === 'object') {
    rawPoints = Array.isArray(payload.points) ? payload.points : [];
    srcW = Number(payload.imageWidth ?? payload.width ?? payload.image_w ?? payload.w);
    srcH = Number(payload.imageHeight ?? payload.height ?? payload.image_h ?? payload.h);
  }

  if (!Array.isArray(rawPoints) || rawPoints.length === 0) {
    setStatus('JSON anotasi tidak valid: points kosong.');
    return;
  }

  const hasSrcSize = Number.isFinite(srcW) && Number.isFinite(srcH) && srcW > 0 && srcH > 0;
  const sx = hasSrcSize ? canvas.width / srcW : 1;
  const sy = hasSrcSize ? canvas.height / srcH : 1;

  const mapped = [];
  for (const p of rawPoints) {
    if (!Array.isArray(p) || p.length < 2) continue;
    const x = Number(p[0]);
    const y = Number(p[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    mapped.push({
      x: clamp(x * sx, 0, canvas.width),
      y: clamp(y * sy, 0, canvas.height),
    });
  }

  if (mapped.length < 3) {
    setStatus('JSON anotasi valid tapi titik kurang dari 3.');
    return;
  }

  points = mapped;
  closed = true;
  draw();
  setStatus(`Anotasi JSON dimuat (${points.length} titik).`);
}

function draw() {
  if (!baseImage) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

  if (points.length > 0) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00ff99';
    ctx.fillStyle = 'rgba(0,255,153,0.2)';

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    if (closed && points.length >= 3) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();

    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#00ff99';
      ctx.fill();
    }
  }
}

imageInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      baseImage = img;
      const maxW = 900;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      zoom = 1;
      setCanvasDisplaySize();
      if (canvasWrap) {
        canvasWrap.scrollLeft = 0;
        canvasWrap.scrollTop = 0;
      }
      points = [];
      closed = false;
      draw();
      setStatus('Gambar siap. Klik untuk menambahkan titik polygon camou.');
      applyAnnotationFromPayload();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

annotationInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      annotationPayload = JSON.parse(String(reader.result || '{}'));
      setStatus('JSON anotasi dibaca.');
      applyAnnotationFromPayload();
    } catch (err) {
      setStatus(`Gagal parse JSON anotasi: ${err.message}`);
    }
  };
  reader.readAsText(file);
});

canvas.addEventListener('click', (e) => {
  if (!baseImage || closed) return;
  const { x, y } = mapClientToCanvas(e);
  points.push({ x, y });
  draw();
  setStatus(`Titik: ${points.length}.`);
});

zoomInBtn.addEventListener('click', () => {
  if (!baseImage) return;
  setZoom(zoom + ZOOM_STEP);
});

zoomOutBtn.addEventListener('click', () => {
  if (!baseImage) return;
  setZoom(zoom - ZOOM_STEP);
});

zoomResetBtn.addEventListener('click', () => {
  if (!baseImage) return;
  setZoom(1);
});

closePolygonBtn.addEventListener('click', () => {
  if (points.length < 3) {
    setStatus('Minimal 3 titik untuk menutup polygon.');
    return;
  }
  closed = true;
  draw();
  setStatus('Polygon tertutup. Siap inferensi.');
});

clearBtn.addEventListener('click', () => {
  points = [];
  closed = false;
  draw();
  setStatus('Anotasi dihapus.');
});

inferBtn.addEventListener('click', async () => {
  if (!uploadedFile || !baseImage) {
    setStatus('Upload gambar dulu.');
    return;
  }
  if (points.length < 3) {
    setStatus('Buat polygon minimal 3 titik.');
    return;
  }

  const sx = baseImage.width / canvas.width;
  const sy = baseImage.height / canvas.height;
  const pointsOriginal = points.map((p) => [p.x * sx, p.y * sy]);

  const formData = new FormData();
  formData.append('image', uploadedFile);
  formData.append('points', JSON.stringify(pointsOriginal));
  formData.append('compose_mode', composeModeEl?.value || 'tile');
  formData.append('tile_size', String(tileSizeEl?.value || '64'));

  setStatus('Inferensi berjalan...');
  inferBtn.disabled = true;

  try {
    const res = await fetch('/api/infer', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    outEnv.src = `data:image/png;base64,${data.environment}`;
    outMask.src = `data:image/png;base64,${data.mask}`;
    outPattern.src = `data:image/png;base64,${data.pattern}`;
    outPatternCanvas.src = `data:image/png;base64,${data.pattern_canvas}`;
    outComposite.src = `data:image/png;base64,${data.composite}`;

    window.__latestPatternTexture = outPattern.src;
    if (typeof window.update3DTexture === 'function') {
      window.update3DTexture(window.__latestPatternTexture);
    }

    const ratioPct = (Number(data.mask_area_ratio || 0) * 100).toFixed(2);
    statsEl.textContent = `Output: ${data.width}x${data.height} · Mask area: ${ratioPct}% · Mode: ${data.compose_mode} · Tile: ${data.tile_size} · RandomTile: ${data.random_tiling} · ColorMatch: ${data.color_match} · Feather: k=${data.feather_kernel}, s=${Number(data.feather_sigma).toFixed(2)}`;

    setStatus('Inferensi selesai.');
  } catch (err) {
    setStatus(`Gagal inferensi: ${err.message}`);
  } finally {
    inferBtn.disabled = false;
  }
});

updateZoomLabel();
