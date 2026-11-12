"""Tests for the web app and preprocessing. TensorFlow is not needed:
the model is replaced by a stub that returns fixed probabilities."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as webapp  # noqa: E402
from lungct import CLASSES, ModelMeta, canonical_class, load_image, preprocess  # noqa: E402


class StubModel:
    def __init__(self, probs):
        self.probs = np.array([probs], dtype=float)
        self.inputs = []

    def predict(self, x, verbose=0):
        self.inputs.append(x)
        return self.probs


def png_bytes(size=(64, 48), color=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client(monkeypatch):
    stub = StubModel([0.1, 0.05, 0.8, 0.05])
    monkeypatch.setattr(webapp.classifier, "_model", stub)
    monkeypatch.setattr(webapp.classifier, "meta", ModelMeta(preprocess="rescale"))
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        c.stub = stub
        yield c


@pytest.mark.parametrize("folder, expected", [
    ("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib", "adenocarcinoma"),
    ("adenocarcinoma", "adenocarcinoma"),
    ("large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa", "large_cell_carcinoma"),
    ("large.cell.carcinoma", "large_cell_carcinoma"),
    ("normal", "normal"),
    ("squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa", "squamous_cell_carcinoma"),
    ("README", None),
])
def test_canonical_class(folder, expected):
    assert canonical_class(folder) == expected


def test_load_image_resizes_to_rgb_224():
    arr = load_image(png_bytes(size=(300, 120)))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.max() <= 255


def test_rescale_preprocessing():
    batch = np.full((1, 4, 4, 3), 255.0, dtype=np.float32)
    assert np.allclose(preprocess(batch, "rescale"), 1.0)


def test_meta_roundtrip(tmp_path):
    model_path = tmp_path / "m.keras"
    ModelMeta(preprocess="resnet50").save(model_path)
    meta = ModelMeta.for_model(model_path)
    assert meta.preprocess == "resnet50"
    assert meta.classes == CLASSES


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Lung CT Classifier" in resp.data


def test_api_predict_returns_top_class(client):
    resp = client.post("/api/predict", data={"image": (png_bytes(), "scan.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["label"] == "normal"
    assert body["display_name"] == "Normal"
    assert body["confidence"] == pytest.approx(0.8)
    probs = [row["probability"] for row in body["probabilities"]]
    assert body["probabilities"][0]["name"] == "Normal"
    assert probs == sorted(probs, reverse=True)  # highest first
    x = client.stub.inputs[-1]
    assert x.shape == (1, 224, 224, 3) and x.max() <= 1.0  # rescaled input reached the model


def test_html_form_shows_result(client):
    resp = client.post("/", data={"image": (png_bytes(), "scan.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Normal" in resp.data and b"80.0%" in resp.data


def test_rejects_wrong_extension(client):
    resp = client.post("/api/predict", data={"image": (io.BytesIO(b"hello"), "notes.txt")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_rejects_corrupt_image(client):
    resp = client.post("/api/predict", data={"image": (io.BytesIO(b"not a png"), "scan.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_missing_model_gives_clear_error(monkeypatch, tmp_path):
    clf = webapp.Classifier(tmp_path / "missing.keras")
    monkeypatch.setattr(webapp, "classifier", clf)
    with webapp.app.test_client() as c:
        resp = c.post("/api/predict", data={"image": (png_bytes(), "scan.png")},
                      content_type="multipart/form-data")
    assert resp.status_code == 503
    assert "train.py" in resp.get_json()["error"]
