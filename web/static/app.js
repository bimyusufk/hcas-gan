const imageInput = document.getElementById('imageInput');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const closePolygonBtn = document.getElementById('closePolygonBtn');
const clearBtn = document.getElementById('clearBtn');
const inferBtn = document.getElementById('inferBtn');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');

const outEnv = document.getElementById('outEnv');
const outMask = document.getElementById('outMask');
const outPattern = document.getElementById('outPattern');
const outComposite = document.getElementById('outComposite');

let uploadedFile = null;
let baseImage = null;
let points = [];
let closed = false;

function setStatus(msg) {
  statusEl.textContent = msg;
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
      points = [];
      closed = false;
      draw();
      setStatus('Gambar siap. Klik untuk menambahkan titik polygon camou.');
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

canvas.addEventListener('click', (e) => {
  if (!baseImage || closed) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  points.push({ x, y });
  draw();
  setStatus(`Titik: ${points.length}.`);
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
    outComposite.src = `data:image/png;base64,${data.composite}`;

    const ratioPct = (Number(data.mask_area_ratio || 0) * 100).toFixed(2);
    statsEl.textContent = `Output: ${data.width}x${data.height} · Mask area: ${ratioPct}%`;

    setStatus('Inferensi selesai.');
  } catch (err) {
    setStatus(`Gagal inferensi: ${err.message}`);
  } finally {
    inferBtn.disabled = false;
  }
});
