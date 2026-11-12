"""Flask demo: upload a chest CT slice and get the predicted class.

Research/portfolio demo only — not a medical device and not for diagnosis.

Run:  python app.py   (set LUNG_MODEL_PATH to use a model other than the default)
"""

import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from lungct import Classifier

ALLOWED = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
classifier = Classifier()


def read_upload():
    """Return (PIL image, error message) from the 'image' form field."""
    file = request.files.get("image")
    if file is None or not file.filename:
        return None, "Please choose an image file."
    if os.path.splitext(file.filename)[1].lower() not in ALLOWED:
        return None, "Supported formats: PNG, JPG."
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()
        return img, None
    except (UnidentifiedImageError, OSError):
        return None, "That file could not be read as an image."


def run_prediction(img):
    try:
        return classifier.predict(img), None
    except FileNotFoundError as exc:
        return None, str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"model_ready": classifier.available, "model_path": str(classifier.model_path)}
    if request.method == "POST":
        img, error = read_upload()
        if img is not None:
            result, error = run_prediction(img)
            if result:
                preview = io.BytesIO()
                thumb = img.convert("RGB")
                thumb.thumbnail((320, 320))
                thumb.save(preview, format="PNG")
                ctx.update(result=result, preview=base64.b64encode(preview.getvalue()).decode())
        ctx["error"] = error
    return render_template("index.html", **ctx)


@app.post("/api/predict")
def api_predict():
    img, error = read_upload()
    if error:
        return jsonify({"error": error}), 400
    result, error = run_prediction(img)
    if error:
        return jsonify({"error": error}), 503
    return jsonify(result)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": classifier.available}


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File larger than 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
