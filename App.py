#!/usr/bin/env python3
"""
Sand Mining Detection Web App
Upload a satellite image → get probability + demarcation of mining areas.

Run: python app.py
Open: http://localhost:5000

Requirements: pip install flask pillow numpy scikit-learn joblib
"""

import os
import sys
import json
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add parent to path so src imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, render_template_string
from src import config
from src.features import extract_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sand Mining Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0a1628;
    color: #fff;
    min-height: 100vh;
  }
  header {
    background: #1b3a6b;
    padding: 20px 40px;
    border-bottom: 3px solid #0d7377;
  }
  header h1 { font-size: 1.6rem; color: #14a8ae; }
  header p  { color: #8899aa; font-size: 0.9rem; margin-top: 4px; }

  .container { max-width: 1100px; margin: 0 auto; padding: 30px 20px; }

  /* River selector */
  .river-bar {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px;
  }
  .river-btn {
    padding: 8px 20px; border-radius: 20px; border: 2px solid #0d7377;
    background: transparent; color: #14a8ae; cursor: pointer;
    font-size: 0.9rem; transition: all 0.2s;
  }
  .river-btn.active, .river-btn:hover {
    background: #0d7377; color: #fff;
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #0d7377; border-radius: 12px;
    padding: 40px; text-align: center; cursor: pointer;
    background: #0f1f3d; transition: border-color 0.2s;
    margin-bottom: 24px;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: #14a8ae; }
  .upload-zone input { display: none; }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-zone p { color: #8899aa; }
  .upload-zone .hint { font-size: 0.8rem; margin-top: 8px; color: #556677; }

  /* Results */
  .results { display: none; }
  .results.show { display: block; }

  .prob-banner {
    border-radius: 12px; padding: 24px 32px;
    margin-bottom: 24px; text-align: center;
  }
  .prob-banner .prob-val {
    font-size: 3.5rem; font-weight: 700; line-height: 1;
  }
  .prob-banner .prob-lbl {
    font-size: 1.1rem; margin-top: 8px;
  }
  .prob-banner.high   { background: linear-gradient(135deg, #7b0000, #c0392b); }
  .prob-banner.medium { background: linear-gradient(135deg, #7d5a00, #e67e22); }
  .prob-banner.low    { background: linear-gradient(135deg, #0d4420, #27ae60); }

  .images-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 20px; margin-bottom: 24px;
  }
  @media (max-width: 700px) { .images-grid { grid-template-columns: 1fr; } }

  .img-card {
    background: #0f1f3d; border-radius: 12px;
    border: 1px solid #1b3a6b; overflow: hidden;
  }
  .img-card .img-title {
    padding: 10px 16px; font-size: 0.85rem;
    color: #8899aa; border-bottom: 1px solid #1b3a6b;
  }
  .img-card img { width: 100%; display: block; }

  /* Feature table */
  .features-box {
    background: #0f1f3d; border-radius: 12px;
    border: 1px solid #1b3a6b; padding: 20px;
    margin-bottom: 24px;
  }
  .features-box h3 { color: #14a8ae; margin-bottom: 14px; }
  .feat-row {
    display: flex; justify-content: space-between;
    padding: 6px 0; border-bottom: 1px solid #162440;
    font-size: 0.88rem;
  }
  .feat-row:last-child { border-bottom: none; }
  .feat-name { color: #8899aa; }
  .feat-val  { color: #fff; font-weight: 600; }
  .feat-bar-wrap { width: 100px; background: #162440; border-radius: 4px; height: 8px; margin-top: 4px; }
  .feat-bar { height: 8px; border-radius: 4px; background: #0d7377; }

  /* Spinner */
  .spinner {
    display: none; text-align: center; padding: 40px;
  }
  .spinner.show { display: block; }
  .spinner-ring {
    width: 50px; height: 50px; border: 4px solid #1b3a6b;
    border-top: 4px solid #14a8ae; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box {
    background: #3d0000; border: 1px solid #c0392b;
    border-radius: 8px; padding: 16px; margin-bottom: 20px;
    display: none; color: #ff6b6b;
  }
  .error-box.show { display: block; }

  .model-selector {
    margin-bottom: 20px;
  }
  .model-selector label { color: #8899aa; font-size: 0.9rem; margin-right: 10px; }
  .model-selector select {
    background: #0f1f3d; color: #fff; border: 1px solid #0d7377;
    padding: 6px 12px; border-radius: 6px; font-size: 0.9rem;
  }
</style>
</head>
<body>

<header>
  <h1>🛰️ Sand Mining Detection</h1>
  <p>Upload a satellite image to detect sand mining probability and highlight mining regions</p>
</header>

<div class="container">

  <!-- River / Model selector -->
  <div class="model-selector">
    <label>River model:</label>
    <select id="modelSelect">
      <option value="auto">Auto (best available)</option>
    </select>
  </div>

  <!-- Upload zone -->
  <div class="upload-zone" id="dropZone">
    <input type="file" id="fileInput" accept="image/*">
    <div class="icon">📡</div>
    <p>Drop satellite image here or <strong>click to upload</strong></p>
    <p class="hint">Supports PNG, JPG, TIF — ideally Sentinel-2 RGB composite</p>
  </div>

  <!-- Error -->
  <div class="error-box" id="errorBox"></div>

  <!-- Spinner -->
  <div class="spinner" id="spinner">
    <div class="spinner-ring"></div>
    <p style="color:#8899aa">Analysing image for sand mining signatures...</p>
  </div>

  <!-- Results -->
  <div class="results" id="results">

    <div class="prob-banner" id="probBanner">
      <div class="prob-val" id="probVal">—</div>
      <div class="prob-lbl" id="probLbl">—</div>
    </div>

    <div class="images-grid">
      <div class="img-card">
        <div class="img-title">Original image</div>
        <img id="origImg" src="" alt="Original">
      </div>
      <div class="img-card">
        <div class="img-title">Mining regions highlighted</div>
        <img id="annotImg" src="" alt="Annotated">
      </div>
    </div>

    <div class="features-box">
      <h3>Key features</h3>
      <div id="featList"></div>
    </div>

  </div>
</div>

<script>
// ── Load available models ──────────────────────────────────────────────────
fetch('/models').then(r=>r.json()).then(data=>{
  const sel = document.getElementById('modelSelect');
  data.models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    sel.appendChild(opt);
  });
});

// ── Drag and drop ──────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) processFile(fileInput.files[0]);
});

// ── Process file ───────────────────────────────────────────────────────────
function processFile(file) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model', document.getElementById('modelSelect').value);

  document.getElementById('results').classList.remove('show');
  document.getElementById('errorBox').classList.remove('show');
  document.getElementById('spinner').classList.add('show');

  fetch('/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      document.getElementById('spinner').classList.remove('show');
      if (data.error) {
        showError(data.error); return;
      }
      showResults(data);
    })
    .catch(err => {
      document.getElementById('spinner').classList.remove('show');
      showError('Server error: ' + err.message);
    });
}

function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = '❌ ' + msg;
  box.classList.add('show');
}

function showResults(data) {
  const prob = data.probability;
  const pct  = Math.round(prob * 100);

  // Banner colour
  const banner = document.getElementById('probBanner');
  banner.className = 'prob-banner ' + (prob >= 0.65 ? 'high' : prob >= 0.4 ? 'medium' : 'low');
  document.getElementById('probVal').textContent = pct + '%';
  document.getElementById('probLbl').textContent = data.classification;

  // Images
  document.getElementById('origImg').src  = 'data:image/png;base64,' + data.original_b64;
  document.getElementById('annotImg').src = 'data:image/png;base64,' + data.annotated_b64;

  // Features
  const featList = document.getElementById('featList');
  featList.innerHTML = '';
  data.top_features.forEach(f => {
    const pct = Math.min(100, Math.round(f.importance * 100));
    featList.innerHTML += `
      <div class="feat-row">
        <div>
          <div class="feat-name">${f.name}</div>
          <div class="feat-bar-wrap"><div class="feat-bar" style="width:${pct}%"></div></div>
        </div>
        <div class="feat-val">${f.value !== null ? f.value.toFixed(4) : 'N/A'}</div>
      </div>`;
  });

  document.getElementById('results').classList.add('show');
}
</script>
</body>
</html>
"""

# ── Helper: load best available model ────────────────────────────────────────

def get_available_models():
    """Return list of river models available."""
    models = ['auto']
    models_dir = config.MODELS_DIR
    if os.path.exists(models_dir):
        for entry in os.scandir(models_dir):
            if entry.is_dir():
                model_file = os.path.join(entry.path, 'sand_mining_model.joblib')
                if os.path.exists(model_file):
                    models.append(entry.name)
    return models


def load_model_for_river(river='auto'):
    """Load model + scaler + feature_names for a given river (or best available)."""
    import joblib

    search_dirs = []

    if river != 'auto':
        search_dirs.append(os.path.join(config.MODELS_DIR, river))

    # Also search per-river subdirs
    if os.path.exists(config.MODELS_DIR):
        for entry in os.scandir(config.MODELS_DIR):
            if entry.is_dir():
                search_dirs.append(entry.path)

    # Fall back to main models dir
    search_dirs.append(config.MODELS_DIR)

    for d in search_dirs:
        mf = os.path.join(d, 'sand_mining_model.joblib')
        sf = os.path.join(d, 'feature_scaler.joblib')
        ff = os.path.join(d, 'feature_names.json')
        if os.path.exists(mf) and os.path.exists(sf):
            model  = joblib.load(mf)
            scaler = joblib.load(sf)
            feature_names = []
            if os.path.exists(ff):
                with open(ff) as f:
                    feature_names = json.load(f)
            fi_path = os.path.join(d, 'feature_importance.json')
            feature_importance = {}
            if os.path.exists(fi_path):
                with open(fi_path) as f:
                    feature_importance = json.load(f)
            river_name = os.path.basename(d)
            print(f"[WebApp] Loaded model from: {d}")
            return model, scaler, feature_names, feature_importance, river_name

    return None, None, [], {}, None


def annotate_image(img, probability, feature_vec, feature_names):
    """
    Draw mining probability heatmap overlay on the image.
    Highlights high-BSI / low-NDVI regions as likely mining zones.
    Returns annotated PIL Image.
    """
    img_draw = img.copy().convert('RGBA')
    draw = ImageDraw.Draw(img_draw, 'RGBA')
    w, h = img.size

    # ── Simple spatial heuristic: divide image into 4x4 grid ─────────────
    # For each cell compute a local "mining score" based on pixel brightness
    # (high red, low green = sandy/disturbed soil)
    img_rgb = np.array(img.convert('RGB'), dtype=float)
    cell_w = w // 4
    cell_h = h // 4

    cell_scores = []
    for row in range(4):
        for col in range(4):
            y1, y2 = row*cell_h, (row+1)*cell_h
            x1, x2 = col*cell_w, (col+1)*cell_w
            patch = img_rgb[y1:y2, x1:x2]
            # Mining signal: high red, moderate blue, low green ratio
            r_mean = patch[:,:,0].mean() / 255
            g_mean = patch[:,:,1].mean() / 255
            b_mean = patch[:,:,2].mean() / 255
            # BSI proxy: high SWIR/red, low NIR/green
            bsi_proxy = (r_mean - g_mean + 0.1) / (r_mean + g_mean + 0.1)
            brightness = (r_mean + g_mean + b_mean) / 3
            # Sand mining: bright + reddish + low green
            score = bsi_proxy * brightness * (1 - g_mean) * 4
            score = float(np.clip(score, 0, 1))
            cell_scores.append((row, col, score))

    # Scale scores relative to overall probability
    max_score = max(s for _,_,s in cell_scores) if cell_scores else 1
    threshold = max_score * 0.6  # only highlight top cells

    for row, col, score in cell_scores:
        if score >= threshold and probability >= 0.4:
            y1, y2 = row*cell_h, (row+1)*cell_h
            x1, x2 = col*cell_w, (col+1)*cell_w
            alpha = int(100 + 80 * (score / max_score))
            if probability >= 0.65:
                fill = (255, 50, 50, alpha)
                outline = (255, 0, 0, 200)
            else:
                fill = (255, 165, 0, alpha)
                outline = (255, 140, 0, 200)
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline)
            draw.rectangle([x1, y1, x1+2, y2], fill=outline)

    # ── Probability banner at top ─────────────────────────────────────────
    pct = int(probability * 100)
    banner_h = max(36, h // 14)
    if probability >= 0.65:
        banner_col = (192, 0, 0, 220)
        text_col   = (255, 200, 200, 255)
    elif probability >= 0.4:
        banner_col = (180, 100, 0, 220)
        text_col   = (255, 230, 180, 255)
    else:
        banner_col = (0, 100, 30, 220)
        text_col   = (180, 255, 200, 255)

    draw.rectangle([0, 0, w, banner_h], fill=banner_col)

    label = {True: "⚠ Sand Mining Likely", False: "✓ No Mining Detected"}[probability >= 0.65]
    if 0.4 <= probability < 0.65:
        label = "~ Possible Sand Mining"

    try:
        font = ImageFont.truetype("arial.ttf", banner_h - 10)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 6), f"{pct}%  {label}", fill=text_col, font=font)

    # Flatten RGBA → RGB
    background = Image.new('RGB', img_draw.size, (10, 22, 40))
    background.paste(img_draw, mask=img_draw.split()[3])
    return background


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/models')
def list_models():
    return jsonify({'models': get_available_models()})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    river = request.form.get('model', 'auto')

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    # Resize to 1024x1024 for feature extraction (same as training)
    img_resized = img.resize((1024, 1024), Image.Resampling.LANCZOS)

    # Save to temp file for feature extractor
    tmp_path = os.path.join(config.TEMP_DIR, 'webapp_upload.png')
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    img_resized.save(tmp_path)

    # Load model
    model, scaler, feature_names, feature_importance, model_river = load_model_for_river(river)
    if model is None:
        return jsonify({
            'error': 'No trained model found. Run the pipeline first to train a model.'
        }), 400

    # Extract features
    try:
        feat_dict = extract_features(tmp_path)
        if feat_dict is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        # Align features to model's expected order
        if feature_names:
            feat_vec = np.array([feat_dict.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)
        else:
            feat_vec = np.array(list(feat_dict.values())).reshape(1, -1)

        feat_vec_scaled = scaler.transform(feat_vec)
        probability = float(model.predict_proba(feat_vec_scaled)[0][1])

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Classification
    if probability >= 0.65:
        classification = 'Sand Mining Likely'
    elif probability >= 0.4:
        classification = 'Possible Sand Mining'
    else:
        classification = 'No Sand Mining Likely'

    # Annotate image
    annotated = annotate_image(img_resized, probability, feat_vec[0], feature_names)

    # Top features
    top_features = []
    fi_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, fimp in fi_items:
        fval = feat_dict.get(fname, None)
        top_features.append({
            'name':       fname,
            'importance': float(fimp),
            'value':      float(fval) if fval is not None else None
        })

    # If no feature importance, just show top values
    if not top_features and feat_dict:
        for fname, fval in list(feat_dict.items())[:10]:
            top_features.append({'name': fname, 'importance': 0.1, 'value': float(fval)})

    return jsonify({
        'probability':    round(probability, 4),
        'classification': classification,
        'model_river':    model_river,
        'original_b64':   pil_to_b64(img_resized),
        'annotated_b64':  pil_to_b64(annotated),
        'top_features':   top_features,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Sand Mining Detection Web App")
    print("="*60)
    print(f"  Models directory: {config.MODELS_DIR}")
    available = get_available_models()
    print(f"  Available models: {available}")
    if len(available) <= 1:
        print("  ⚠️  No river models found yet.")
        print("     Run the pipeline first to train models.")
    print("\n  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)