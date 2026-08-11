"""
app.py

Web app for testing the knee MRI abnormality classifier, deployable
on Render's free tier. Upload one or more DICOM slices from a study;
get back per-finding probabilities, averaged across all uploaded
slices, with TTA (horizontal flip) applied automatically.
"""

import io
import logging
import os
from pathlib import Path

import numpy as np
import pydicom
import torch
from flask import Flask, render_template_string, request, send_file
from PIL import Image, ImageDraw

from models.baseline_cnn import KneeMRIClassifier
from preprocessing.transforms import preprocess_slice
from preprocessing.augmentations import apply_augmentation, get_val_augmentations
from inference.tta import predict_with_tta
from datasets.knee_dataset import FINDING_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

DEVICE = torch.device("cpu")  # Render free tier has no GPU

CHECKPOINT_CANDIDATES = [
    Path("checkpoints/best_model_resnet18.pt"),
    Path("checkpoints/best_model.pt"),
]

ICON_DIR = Path("static_icons")
ICON_DIR.mkdir(exist_ok=True)

_model = None
_backbone_name = None


def load_model():
    """Load the trained model once, from whichever checkpoint exists."""
    global _model, _backbone_name
    if _model is not None:
        return _model

    checkpoint_path = next((p for p in CHECKPOINT_CANDIDATES if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Expected one of: "
            + ", ".join(str(p) for p in CHECKPOINT_CANDIDATES)
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone_name = checkpoint.get("backbone_name", "resnet18")
    dropout_rate = checkpoint.get("hyperparameters", {}).get("dropout_rate", 0.3)

    model = KneeMRIClassifier(
        backbone_name=backbone_name,
        pretrained=False,
        dropout_rate=dropout_rate,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _backbone_name = backbone_name
    logger.info("Loaded model | backbone=%s | checkpoint=%s", backbone_name, checkpoint_path)
    return _model


def ensure_icons_exist() -> None:
    """Generate simple placeholder app icons (192x192, 512x512) if missing."""
    for size in (192, 512):
        icon_path = ICON_DIR / f"icon-{size}.png"
        if icon_path.exists():
            continue
        img = Image.new("RGB", (size, size), color="#2563eb")
        draw = ImageDraw.Draw(img)
        margin = size // 5
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            outline="white",
            width=max(size // 25, 4),
        )
        img.save(icon_path)
        logger.info("Generated icon: %s", icon_path)


UPLOAD_FORM = """
<!doctype html>
<html>
<head>
    <title>Knee MRI Abnormality Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="manifest" href="/manifest.json">
    <link rel="apple-touch-icon" href="/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="Knee MRI">
    <meta name="theme-color" content="#2563eb">
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
        h1 { font-size: 22px; }
        .sub { color: #666; margin-bottom: 24px; }
        input[type=file] { margin: 16px 0; }
        button { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        table { border-collapse: collapse; width: 100%; margin-top: 24px; }
        th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e5e5; }
        th { color: #666; font-weight: 600; font-size: 13px; text-transform: uppercase; }
        .bar-bg { background: #eee; border-radius: 4px; height: 10px; width: 160px; overflow: hidden; }
        .bar-fill { background: #2563eb; height: 100%; }
        .pos { color: #b91c1c; font-weight: 600; }
        .neg { color: #16a34a; }
        .note { color: #888; font-size: 13px; margin-top: 20px; }
        .error { background: #fee2e2; color: #b91c1c; padding: 12px; border-radius: 6px; margin-top: 20px; }
        #install-btn { display: none; margin-top: 16px; background: #16a34a; }
    </style>
</head>
<body>
    <h1>Knee MRI Abnormality Detector</h1>
    <div class="sub">Upload one or more DICOM slices (.dcm) from a single study. Predictions are averaged across all uploaded slices.</div>

    <button id="install-btn">Install App</button>

    <form method="post" enctype="multipart/form-data">
        <input type="file" name="dicom_files" accept=".dcm" multiple required>
        <br>
        <button type="submit">Run Prediction</button>
    </form>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if results %}
    <p class="note">Model: {{ backbone_name }} &middot; Slices analyzed: {{ num_slices }}</p>
    <table>
        <tr><th>Finding</th><th>Probability</th><th></th><th>Status</th></tr>
        {% for name, prob in results %}
        <tr>
            <td>{{ name }}</td>
            <td>{{ "%.1f"|format(prob * 100) }}%</td>
            <td><div class="bar-bg"><div class="bar-fill" style="width: {{ prob * 100 }}%"></div></div></td>
            <td class="{{ 'pos' if prob >= 0.5 else 'neg' }}">{{ 'Suspected' if prob >= 0.5 else 'Unlikely' }}</td>
        </tr>
        {% endfor %}
    </table>
    <p class="note">This is a research prototype for testing purposes only, not a diagnostic tool. The 50% threshold shown is illustrative, not clinically validated.</p>
    {% endif %}

    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            document.getElementById('install-btn').style.display = 'inline-block';
        });
        document.getElementById('install-btn').addEventListener('click', async () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                await deferredPrompt.userChoice;
                deferredPrompt = null;
                document.getElementById('install-btn').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


def run_prediction(files) -> np.ndarray:
    """Load uploaded DICOM files, preprocess, run TTA inference, average results."""
    model = load_model()
    pipeline = get_val_augmentations()

    all_probs = []
    for file in files:
        ds = pydicom.dcmread(io.BytesIO(file.read()))
        image = preprocess_slice(ds.pixel_array)
        image = apply_augmentation(image, pipeline)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        probs = predict_with_tta(model, image_tensor, use_flip=True)
        all_probs.append(probs.squeeze(0).cpu().numpy())

    return np.mean(all_probs, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    """Serve the upload form and handle prediction requests."""
    results = None
    error = None
    num_slices = 0

    if request.method == "POST":
        files = request.files.getlist("dicom_files")
        files = [f for f in files if f.filename]

        if not files:
            error = "Please select at least one .dcm file."
        else:
            try:
                avg_probs = run_prediction(files)
                results = sorted(
                    zip(FINDING_COLUMNS, avg_probs.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                num_slices = len(files)
            except Exception as exc:
                logger.exception("Prediction failed")
                error = f"Prediction failed: {exc}"

    return render_template_string(
        UPLOAD_FORM,
        results=results,
        error=error,
        backbone_name=_backbone_name,
        num_slices=num_slices,
    )


@app.route("/manifest.json")
def manifest():
    """Serve the PWA manifest, telling the browser this is installable."""
    return {
        "name": "Knee MRI Abnormality Detector",
        "short_name": "Knee MRI",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2563eb",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }


@app.route("/sw.js")
def service_worker():
    """Minimal service worker -- required for Chrome's install prompt."""
    js = "self.addEventListener('fetch', (event) => {});"
    return app.response_class(js, mimetype="application/javascript")


@app.route("/icon-192.png")
def icon_192():
    return send_file(ICON_DIR / "icon-192.png", mimetype="image/png")


@app.route("/icon-512.png")
def icon_512():
    return send_file(ICON_DIR / "icon-512.png", mimetype="image/png")


if __name__ == "__main__":
    load_model()
    ensure_icons_exist()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)