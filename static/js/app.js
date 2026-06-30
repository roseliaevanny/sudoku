/* Global state */
let activeTab    = 'upload';
let selectedFile = null;      // File from upload tab
let capturedBlob = null;      // Blob from camera tab
let cameraStream = null;
let solveAbort   = null;      // AbortController for in-flight fetch

/* Tab switching */
function switchTab(tab) {
  activeTab = tab;

  document.querySelectorAll('.tab').forEach(t => {
    const isActive = t.id === `tab-${tab}`;
    t.classList.toggle('active', isActive);
    t.setAttribute('aria-selected', isActive);
  });
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === `panel-${tab}`)
  );

  if (tab !== 'camera') stopCamera();
  updateSolveButton();
}

/* Upload— drag-and-drop + file picker */
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('drag-over');
}
function handleDragLeave() {
  document.getElementById('drop-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) applyFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) applyFile(file);
}

function applyFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select a valid image file (PNG, JPG, JPEG).');
    return;
  }
  selectedFile = file;
  capturedBlob = null;

  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById('preview-img');
    img.src = ev.target.result;
    img.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
  updateSolveButton();
}

/* Camera */
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }
    });
    const video = document.getElementById('camera-feed');
    video.srcObject = cameraStream;
    video.classList.remove('hidden');
    document.getElementById('captured-img').classList.add('hidden');

    document.getElementById('btn-start-cam').classList.add('hidden');
    document.getElementById('btn-capture').classList.remove('hidden');
    document.getElementById('btn-retake').classList.add('hidden');

    capturedBlob = null;
    updateSolveButton();
  } catch (err) {
    showError('Camera access denied or unavailable.');
  }
}

function capturePhoto() {
  const video  = document.getElementById('camera-feed');
  const canvas = document.getElementById('capture-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    capturedBlob = blob;
    selectedFile = null;

    const url = URL.createObjectURL(blob);
    const img  = document.getElementById('captured-img');
    img.src = url;
    img.classList.remove('hidden');
    document.getElementById('camera-feed').classList.add('hidden');

    stopCamera();
    document.getElementById('btn-capture').classList.add('hidden');
    document.getElementById('btn-retake').classList.remove('hidden');
    updateSolveButton();
  }, 'image/jpeg', 0.92);
}

function retake() {
  capturedBlob = null;
  document.getElementById('captured-img').classList.add('hidden');
  document.getElementById('btn-start-cam').classList.remove('hidden');
  document.getElementById('btn-capture').classList.add('hidden');
  document.getElementById('btn-retake').classList.add('hidden');
  updateSolveButton();
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

/* Solve button state */
function updateSolveButton() {
  const hasImage =
    (activeTab === 'upload' && selectedFile) ||
    (activeTab === 'camera' && capturedBlob);
  document.getElementById('btn-solve').disabled = !hasImage;
}

/* Submit */
async function submitSolve() {
  const blob = activeTab === 'upload' ? selectedFile : capturedBlob;
  if (!blob) return;

  // Show loading UI
  hideError();
  document.getElementById('input-card').classList.add('hidden');
  document.getElementById('results-card').classList.add('hidden');
  document.getElementById('loading-card').classList.remove('hidden');

  // Reset step states
  const STEP_IDS = ['step-1', 'step-2', 'step-3'];
  STEP_IDS.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });

  function activateStep(idx) {
    if (idx > 0) document.getElementById(STEP_IDS[idx - 1]).classList.replace('active', 'done');
    document.getElementById(STEP_IDS[idx]).classList.add('active');
  }
  activateStep(0);
  const t1 = setTimeout(() => activateStep(1), 300);
  const t2 = setTimeout(() => activateStep(2), 600);
  const startTime = Date.now();

  // Fetch with timeout
  const TIMEOUT_MS = 120_000;
  solveAbort = new AbortController();
  const timeoutId = setTimeout(() => solveAbort.abort(), TIMEOUT_MS);

  try {
    const form = new FormData();
    form.append('file', blob, 'sudoku.jpg');

    const res = await fetch(`${API_URL}/solve`, {
      method: 'POST',
      body: form,
      signal: solveAbort.signal,
    });

    clearTimeout(timeoutId);
    clearTimeout(t1);
    clearTimeout(t2);

    // Mark all steps done
    STEP_IDS.forEach(id => {
      const el = document.getElementById(id);
      el.classList.remove('active');
      el.classList.add('done');
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errBody.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    
    // Ensure the loading screen is visible for at least 900ms total
    // so the user can see all 3 steps even if the AI solves it instantly
    const elapsed = Date.now() - startTime;
    if (elapsed < 900) {
      await new Promise(r => setTimeout(r, 900 - elapsed));
    }

    showResults(data);

  } catch (err) {
    clearTimeout(timeoutId);
    clearTimeout(t1);
    clearTimeout(t2);

    document.getElementById('loading-card').classList.add('hidden');
    document.getElementById('input-card').classList.remove('hidden');

    if (err.name === 'AbortError') {
      showError(
        'The request timed out after 2 minutes. ' +
        'Make sure the AI service is running at ' + API_URL +
        ' and try again.'
      );
    } else {
      showError(err.message);
    }
  } finally {
    solveAbort = null;
  }
}

/* Called by the Cancel button in the loading screen */
function cancelSolve() {
  if (solveAbort) solveAbort.abort();
}

/* Render results */
function showResults(data) {
  document.getElementById('loading-card').classList.add('hidden');
  const card = document.getElementById('results-card');
  card.classList.remove('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Title + meta
  const title = document.getElementById('results-title');
  const meta  = document.getElementById('results-meta');
  if (data.solved) {
    title.textContent = 'Puzzle solved!';
    title.style.color = 'var(--solved)';
  } else {
    title.textContent = 'Could not solve.';
    title.style.color = 'var(--warn)';
  }
  meta.textContent = `${data.clues_found} clue${data.clues_found !== 1 ? 's' : ''} detected`;

  // Warning
  const warnBox = document.getElementById('warning-box');
  if (data.warning) {
    warnBox.textContent = '⚠ ' + data.warning;
    warnBox.classList.remove('hidden');
  } else {
    warnBox.classList.add('hidden');
  }

  renderGrid('grid-initial', data.initial_board, data.initial_board, false);
  renderGrid('grid-solved',  data.initial_board, data.solved_board,  data.solved);
}

function renderGrid(containerId, initialBoard, displayBoard, isSolved) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const cell    = document.createElement('div');
      cell.className = 'cell';

      const initVal = initialBoard[r][c];
      const dispVal = displayBoard[r][c];

      if (initVal !== 0) {
        cell.classList.add('given');
        cell.textContent = initVal;
      } else if (dispVal !== 0) {
        if (isSolved) {
          cell.classList.add('solved');
          cell.style.animationDelay = `${(r * 9 + c) * 12}ms`;
        } else {
          cell.classList.add('unsolved-fail');
        }
        cell.textContent = dispVal;
      } else {
        cell.classList.add('empty');
      }

      container.appendChild(cell);
    }
  }
}

/* Error banner */
function showError(msg) {
  let banner = document.getElementById('error-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'error-banner';
    banner.className = 'error-banner';
    banner.innerHTML = `
      <span id="error-msg"></span>
      <button class="error-close" onclick="hideError()">✕</button>
    `;
    document.querySelector('.main').prepend(banner);
  }
  document.getElementById('error-msg').textContent = msg;
  banner.style.display = 'flex';
  banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  const banner = document.getElementById('error-banner');
  if (banner) banner.style.display = 'none';
}

/* Reset */
function reset() {
  selectedFile = null;
  capturedBlob = null;

  const previewImg = document.getElementById('preview-img');
  previewImg.classList.add('hidden');
  previewImg.src = '';
  document.getElementById('file-input').value = '';
  document.getElementById('captured-img').classList.add('hidden');
  document.getElementById('btn-start-cam').classList.remove('hidden');
  document.getElementById('btn-capture').classList.add('hidden');
  document.getElementById('btn-retake').classList.add('hidden');

  ['step-1', 'step-2', 'step-3'].forEach(id =>
    document.getElementById(id).classList.remove('active', 'done')
  );

  document.getElementById('results-card').classList.add('hidden');
  document.getElementById('loading-card').classList.add('hidden');
  document.getElementById('input-card').classList.remove('hidden');
  hideError();
  switchTab('upload');
}

window.addEventListener('beforeunload', stopCamera);
