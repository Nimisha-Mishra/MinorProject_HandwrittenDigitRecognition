/* ── Canvas drawing logic ───────────────────────────────────────── */
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const result = document.getElementById('result');
const predictedDigit = document.getElementById('predictedDigit');
const confidenceEl = document.getElementById('confidence');
const barChart = document.getElementById('barChart');
const errorMsg = document.getElementById('errorMsg');

let drawing = false;
let lastX = 0;
let lastY = 0;

// ── Drawing helpers ───────────────────────────────────────────────

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  if (e.touches) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY,
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  const { x, y } = getPos(e);
  lastX = x;
  lastY = y;
}

function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x;
  lastY = y;
}

function stopDraw() { drawing = false; }

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDraw);

// ── Clear ─────────────────────────────────────────────────────────

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  result.classList.add('hidden');
  errorMsg.classList.add('hidden');
}

clearBtn.addEventListener('click', clearCanvas);

// ── Predict ───────────────────────────────────────────────────────

predictBtn.addEventListener('click', async () => {
  errorMsg.classList.add('hidden');
  const imageData = canvas.toDataURL('image/png');

  predictBtn.textContent = 'Predicting…';
  predictBtn.disabled = true;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    showResult(data);
  } catch (err) {
    showError('Could not reach the server. Is the Flask app running?');
  } finally {
    predictBtn.textContent = 'Predict';
    predictBtn.disabled = false;
  }
});

// ── Display helpers ───────────────────────────────────────────────

function showResult(data) {
  predictedDigit.textContent = data.digit;
  confidenceEl.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;

  barChart.innerHTML = '';
  const topDigit = data.digit;

  data.probabilities
    .sort((a, b) => b.probability - a.probability)
    .forEach((item) => {
      const isTop = item.digit === topDigit;

      const row = document.createElement('div');
      row.className = 'bar-row';

      const digitLabel = document.createElement('span');
      digitLabel.className = 'bar-digit';
      digitLabel.textContent = item.digit;

      const track = document.createElement('div');
      track.className = 'bar-track';

      const fill = document.createElement('div');
      fill.className = `bar-fill${isTop ? ' top' : ''}`;
      fill.style.width = `${item.probability}%`;

      const pct = document.createElement('span');
      pct.className = 'bar-pct';
      pct.textContent = `${item.probability.toFixed(1)}%`;

      track.appendChild(fill);
      row.appendChild(digitLabel);
      row.appendChild(track);
      row.appendChild(pct);
      barChart.appendChild(row);
    });

  result.classList.remove('hidden');
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.classList.remove('hidden');
}
