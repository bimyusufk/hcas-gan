const sampleSelect = document.getElementById('sampleSelect');
const annotationSelect = document.getElementById('annotationSelect');
const deviceSelect = document.getElementById('deviceSelect');
const composeModeEl = document.getElementById('composeMode');
const tileSizeEl = document.getElementById('tileSize');

const refreshSamplesBtn = document.getElementById('refreshSamplesBtn');
const refreshAnnotationsBtn = document.getElementById('refreshAnnotationsBtn');
const refreshCheckpointsBtn = document.getElementById('refreshCheckpointsBtn');
const selectLatestCheckpointsBtn = document.getElementById('selectLatestCheckpointsBtn');
const selectAllCheckpointsBtn = document.getElementById('selectAllCheckpointsBtn');
const clearCheckpointsBtn = document.getElementById('clearCheckpointsBtn');
const inferBtn = document.getElementById('inferBtn');

const selectedBackgroundPreview = document.getElementById('selectedBackgroundPreview');
const checkpointListEl = document.getElementById('checkpointList');
const checkpointCountLabel = document.getElementById('checkpointCountLabel');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const resultCardsEl = document.getElementById('resultCards');

let environmentSamples = [];
let checkpointOptions = [];
let availableModelConfigs = [];
let defaultCheckpointConfigMap = {};

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

function toDataUrl(base64Png) {
  return `data:image/png;base64,${base64Png}`;
}

function updateCheckpointCountLabel() {
  if (!checkpointCountLabel) return;
  const selectedCount = getSelectedCheckpointNames().length;
  checkpointCountLabel.textContent = `${selectedCount} model dipilih`;
}

function getSelectedCheckpointNames() {
  if (!checkpointListEl) return [];
  return Array.from(checkpointListEl.querySelectorAll('input[type="checkbox"]:checked')).map((input) => input.value);
}

function getCheckpointConfigSelectionMap(selectedCheckpointNames = []) {
  if (!checkpointListEl) return {};
  const selectedSet = new Set(Array.isArray(selectedCheckpointNames) ? selectedCheckpointNames : []);
  const out = {};
  const selects = checkpointListEl.querySelectorAll('select[data-checkpoint-name]');
  selects.forEach((selectEl) => {
    const checkpointName = String(selectEl.dataset.checkpointName || '').trim();
    if (!checkpointName || !selectedSet.has(checkpointName)) {
      return;
    }
    const configName = String(selectEl.value || '').trim();
    if (configName) {
      out[checkpointName] = configName;
    }
  });
  return out;
}

function setSelectedCheckpointNames(names) {
  if (!checkpointListEl) return;
  const target = new Set(names);
  const boxes = checkpointListEl.querySelectorAll('input[type="checkbox"]');
  boxes.forEach((input) => {
    input.checked = target.has(input.value);
  });
  updateCheckpointCountLabel();
}

function createOption(value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

function renderSampleSelect(defaultSampleId = '') {
  if (!sampleSelect) return;
  sampleSelect.innerHTML = '';

  if (environmentSamples.length === 0) {
    sampleSelect.appendChild(createOption('', 'Tidak ada sample tersedia'));
    return;
  }

  environmentSamples.forEach((sample) => {
    const label = `${sample.image_name} (${sample.annotation_count} annotation)`;
    sampleSelect.appendChild(createOption(sample.sample_id, label));
  });

  const candidate = defaultSampleId && environmentSamples.some((s) => s.sample_id === defaultSampleId)
    ? defaultSampleId
    : environmentSamples[0].sample_id;
  sampleSelect.value = candidate;
}

async function loadEnvironmentSamples() {
  const res = await fetch('/api/environment/samples');
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }

  environmentSamples = Array.isArray(data.samples) ? data.samples : [];
  renderSampleSelect(data.default_sample_id || '');
  await refreshSelectedSamplePreview();
  await loadAnnotationsForSelectedSample();
}

async function refreshSelectedSamplePreview() {
  const sampleId = String(sampleSelect?.value || '').trim();
  if (!sampleId || !selectedBackgroundPreview) {
    return;
  }
  selectedBackgroundPreview.src = `/api/environment/image?sample_id=${encodeURIComponent(sampleId)}`;
}

async function loadAnnotationsForSelectedSample() {
  const sampleId = String(sampleSelect?.value || '').trim();
  if (!sampleId || !annotationSelect) {
    return;
  }

  const res = await fetch(`/api/environment/annotations?sample_id=${encodeURIComponent(sampleId)}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }

  const annotations = Array.isArray(data.annotations) ? data.annotations : [];
  annotationSelect.innerHTML = '';
  if (annotations.length === 0) {
    annotationSelect.appendChild(createOption('', 'Tidak ada annotation'));
    return;
  }

  annotations.forEach((annotation) => {
    const label = `${annotation.annotation_id} · ${annotation.label} · ${annotation.point_count} titik`;
    annotationSelect.appendChild(createOption(annotation.annotation_id, label));
  });

  const defAnn = String(data.default_annotation_id || '').trim();
  const firstAnn = annotations[0].annotation_id;
  annotationSelect.value = defAnn || firstAnn;
}

function createCheckpointItem(option) {
  const label = document.createElement('label');
  label.className = 'checkpoint-item';

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.value = option.name;
  checkbox.checked = Boolean(option.checked);

  const meta = document.createElement('div');
  meta.className = 'checkpoint-item__meta';

  const titleRow = document.createElement('div');
  titleRow.className = 'checkpoint-item__title-row';

  const title = document.createElement('strong');
  title.textContent = option.label || option.name;
  titleRow.appendChild(title);

  if (option.is_default) {
    const badge = document.createElement('span');
    badge.className = 'checkpoint-badge';
    badge.textContent = 'default';
    titleRow.appendChild(badge);
  }

  const path = document.createElement('span');
  path.className = 'checkpoint-item__path hint';
  path.textContent = option.path || option.name;

  const configRow = document.createElement('div');
  configRow.className = 'checkpoint-item__config-row';

  const configLabel = document.createElement('span');
  configLabel.className = 'checkpoint-item__config-label';
  configLabel.textContent = 'Config model';

  const configSelect = document.createElement('select');
  configSelect.className = 'checkpoint-item__config-select';
  configSelect.dataset.checkpointName = option.name;

  const configOptions = Array.isArray(availableModelConfigs) ? availableModelConfigs : [];
  configOptions.forEach((cfg) => {
    configSelect.appendChild(createOption(cfg.name, cfg.name));
  });

  const fallbackConfig =
    String(option.default_config_name || defaultCheckpointConfigMap[option.name] || '').trim()
    || String(configOptions[0]?.name || '').trim();
  if (fallbackConfig) {
    configSelect.value = fallbackConfig;
  }

  const syncConfigEnabled = () => {
    const isEnabled = Boolean(checkbox.checked);
    configSelect.disabled = !isEnabled;
    configRow.classList.toggle('is-disabled', !isEnabled);
  };

  checkbox.addEventListener('change', () => {
    updateCheckpointCountLabel();
    syncConfigEnabled();
  });

  configRow.appendChild(configLabel);
  configRow.appendChild(configSelect);

  meta.appendChild(titleRow);
  meta.appendChild(path);
  meta.appendChild(configRow);
  label.appendChild(checkbox);
  label.appendChild(meta);
  syncConfigEnabled();
  return label;
}

function renderCheckpointList(options, defaultSelectedNames = []) {
  checkpointOptions = Array.isArray(options) ? options.slice() : [];
  if (!checkpointListEl) return;

  checkpointListEl.innerHTML = '';
  if (checkpointOptions.length === 0) {
    checkpointListEl.textContent = 'Tidak ada checkpoint ditemukan.';
    updateCheckpointCountLabel();
    return;
  }

  const selected = new Set(defaultSelectedNames);
  checkpointOptions.forEach((option) => {
    checkpointListEl.appendChild(
      createCheckpointItem({
        ...option,
        checked: selected.size > 0 ? selected.has(option.name) : Boolean(option.is_default),
      })
    );
  });

  if (getSelectedCheckpointNames().length === 0) {
    const fallback = checkpointOptions.slice(-Math.min(2, checkpointOptions.length)).map((item) => item.name);
    setSelectedCheckpointNames(fallback);
  } else {
    updateCheckpointCountLabel();
  }
}

async function loadCheckpointOptions() {
  const fallbackName = String(window.HCAS_DEFAULT_CHECKPOINT || '').trim();
  const res = await fetch('/api/checkpoints');
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }

  const available = Array.isArray(data.available_checkpoints) ? data.available_checkpoints : [];
  availableModelConfigs = Array.isArray(data.available_configs) ? data.available_configs : [];
  defaultCheckpointConfigMap =
    data.default_checkpoint_config_map && typeof data.default_checkpoint_config_map === 'object'
      ? data.default_checkpoint_config_map
      : {};
  const defaults = Array.isArray(data.default_selected_checkpoints) ? data.default_selected_checkpoints : [];
  if (available.length === 0 && fallbackName) {
    renderCheckpointList(
      [{ name: fallbackName, label: fallbackName, path: fallbackName, is_default: true }],
      [fallbackName],
    );
    return;
  }

  renderCheckpointList(available, defaults);
}

function clearResults() {
  if (resultCardsEl) {
    resultCardsEl.innerHTML = '';
  }
  if (typeof window.HCAS3D?.disposeAll === 'function') {
    window.HCAS3D.disposeAll();
  }
}

function buildMetricList(result) {
  const meta = document.createElement('ul');
  meta.className = 'result-card__meta-list';
  const items = [
    `Mode: ${result.compose_mode}`,
    `Tile: ${result.tile_size}`,
    `Device: ${result.runtime_device}`,
    `Config: ${result.model_config_name || '-'}`,
  ];
  items.forEach((text) => {
    const li = document.createElement('li');
    li.textContent = text;
    meta.appendChild(li);
  });
  return meta;
}

function buildFigure(title, src, alt) {
  const figure = document.createElement('figure');
  figure.className = 'result-card__figure';
  const caption = document.createElement('figcaption');
  caption.textContent = title;
  const img = document.createElement('img');
  img.src = src;
  img.alt = alt;
  figure.appendChild(caption);
  figure.appendChild(img);
  return figure;
}

function renderResultCards(data) {
  clearResults();
  if (!resultCardsEl) return;

  const results = Array.isArray(data.results) ? data.results : [];
  results.forEach((result, index) => {
    const card = document.createElement('article');
    card.className = 'result-card';

    const header = document.createElement('div');
    header.className = 'result-card__header';

    const title = document.createElement('h3');
    title.textContent = result.checkpoint_label || result.checkpoint_name || `Model ${index + 1}`;
    const subtitle = document.createElement('p');
    subtitle.className = 'hint';
    subtitle.textContent = result.checkpoint_path || result.checkpoint_name || '';

    header.appendChild(title);
    header.appendChild(subtitle);

    const figureGrid = document.createElement('div');
    figureGrid.className = 'result-card__figure-grid';
    figureGrid.appendChild(buildFigure('Generated Pattern', toDataUrl(result.pattern), 'generated pattern'));
    figureGrid.appendChild(buildFigure('Pattern Canvas', toDataUrl(result.pattern_canvas), 'pattern canvas'));
    figureGrid.appendChild(buildFigure('Composite', toDataUrl(result.composite), 'composite'));

    const previewWrap = document.createElement('div');
    previewWrap.className = 'result-card__preview3d';
    const previewTitle = document.createElement('p');
    previewTitle.className = 'hint result-card__preview-label';
    previewTitle.textContent = '3D Preview (klik untuk fullscreen)';
    const previewViewport = document.createElement('div');
    previewViewport.className = 'preview3d-mini';
    previewWrap.appendChild(previewTitle);
    previewWrap.appendChild(previewViewport);

    card.appendChild(header);
    card.appendChild(buildMetricList(result));
    card.appendChild(figureGrid);
    card.appendChild(previewWrap);
    resultCardsEl.appendChild(card);

    if (typeof window.HCAS3D?.createPreview === 'function') {
      const handle = window.HCAS3D.createPreview(previewViewport, toDataUrl(result.pattern), {
        modelUrl: window.HCAS_MODEL_GLB_URL || '/model/long_sleeve_t-_shirt.glb',
      });
      
      // Add controls UI
      if (handle && typeof handle.getControlsUI === 'function') {
        const controlsUI = handle.getControlsUI();
        previewWrap.insertBefore(controlsUI, previewViewport);
      }
    }
  });
}

async function runInference() {
  const sampleId = String(sampleSelect?.value || '').trim();
  const annotationId = String(annotationSelect?.value || '').trim();
  const selectedCheckpoints = getSelectedCheckpointNames();
  const checkpointConfigMap = getCheckpointConfigSelectionMap(selectedCheckpoints);

  if (!sampleId) {
    setStatus('Pilih background image terlebih dahulu.');
    return;
  }
  if (!annotationId) {
    setStatus('Pilih annotation terlebih dahulu.');
    return;
  }
  if (selectedCheckpoints.length < 1) {
    setStatus('Pilih minimal 1 checkpoint.');
    return;
  }

  let tileSize = 64;
  try {
    tileSize = Math.max(4, Number.parseInt(String(tileSizeEl?.value || '64'), 10));
  } catch {
    tileSize = 64;
  }

  const payload = {
    sample_id: sampleId,
    annotation_id: annotationId,
    tile_size: tileSize,
    device: String(deviceSelect?.value || window.HCAS_DEFAULT_DEVICE || 'auto').trim().toLowerCase(),
    compose_mode: String(composeModeEl?.value || 'tile').trim().toLowerCase(),
    selected_checkpoints: selectedCheckpoints,
    checkpoint_config_map: checkpointConfigMap,
  };

  inferBtn.disabled = true;
  setStatus(`Inferensi berjalan untuk ${selectedCheckpoints.length} model...`);

  try {
    const res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    renderResultCards(data);

    const ratioPct = (Number(data.mask_area_ratio || 0) * 100).toFixed(2);
    const resultCount = Array.isArray(data.results) ? data.results.length : 0;
    statsEl.textContent = `Image: ${data.image_name || sampleId} · Mask area: ${ratioPct}% · Models: ${resultCount} · Runtime device: ${data.runtime_device}`;
    setStatus('Inferensi selesai.');
  } catch (err) {
    setStatus(`Gagal inferensi: ${err.message}`);
  } finally {
    inferBtn.disabled = false;
  }
}

async function bootstrap() {
  try {
    await loadEnvironmentSamples();
    await loadCheckpointOptions();
    setStatus('Siap untuk inference.');
  } catch (err) {
    setStatus(`Gagal bootstrap UI: ${err.message}`);
  }
}

if (refreshSamplesBtn) {
  refreshSamplesBtn.addEventListener('click', async () => {
    try {
      await loadEnvironmentSamples();
      setStatus('Daftar sample diperbarui.');
    } catch (err) {
      setStatus(`Gagal refresh sample: ${err.message}`);
    }
  });
}

if (sampleSelect) {
  sampleSelect.addEventListener('change', async () => {
    try {
      await refreshSelectedSamplePreview();
      await loadAnnotationsForSelectedSample();
      setStatus('Sample dan annotation diperbarui.');
    } catch (err) {
      setStatus(`Gagal memuat annotation: ${err.message}`);
    }
  });
}

if (refreshAnnotationsBtn) {
  refreshAnnotationsBtn.addEventListener('click', async () => {
    try {
      await loadAnnotationsForSelectedSample();
      setStatus('Daftar annotation diperbarui.');
    } catch (err) {
      setStatus(`Gagal refresh annotation: ${err.message}`);
    }
  });
}

if (refreshCheckpointsBtn) {
  refreshCheckpointsBtn.addEventListener('click', async () => {
    try {
      await loadCheckpointOptions();
      setStatus('Daftar checkpoint diperbarui.');
    } catch (err) {
      setStatus(`Gagal refresh checkpoint: ${err.message}`);
    }
  });
}

if (selectLatestCheckpointsBtn) {
  selectLatestCheckpointsBtn.addEventListener('click', () => {
    const latest = checkpointOptions.slice(-Math.min(2, checkpointOptions.length)).map((item) => item.name);
    setSelectedCheckpointNames(latest);
  });
}

if (selectAllCheckpointsBtn) {
  selectAllCheckpointsBtn.addEventListener('click', () => {
    setSelectedCheckpointNames(checkpointOptions.map((item) => item.name));
  });
}

if (clearCheckpointsBtn) {
  clearCheckpointsBtn.addEventListener('click', () => {
    setSelectedCheckpointNames([]);
  });
}

if (inferBtn) {
  inferBtn.addEventListener('click', runInference);
}

bootstrap();