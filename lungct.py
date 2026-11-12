"""Shared inference code for the lung CT classifier (used by app.py and train.py).

A trained model is a Keras file plus a small JSON sidecar (`<model>.json`)
that records the class order and how input pixels must be prepared, so the
web app always preprocesses images the same way the model was trained.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224

# Canonical class keys, in the alphabetical order Keras/os.listdir produce for
# the Kaggle "Chest CT-Scan images" folders.
CLASSES = ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"]
DISPLAY_NAMES = {
    "adenocarcinoma": "Adenocarcinoma",
    "large_cell_carcinoma": "Large cell carcinoma",
    "normal": "Normal",
    "squamous_cell_carcinoma": "Squamous cell carcinoma",
}

DEFAULT_MODEL_PATH = Path(os.environ.get("LUNG_MODEL_PATH", "models/lung_ct_resnet50.keras"))


def canonical_class(folder_name: str) -> str | None:
    """Map a dataset folder name to a class key.

    The dataset's train/valid folders use long names such as
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib' while the test folders use
    short ones such as 'adenocarcinoma'; both map to the same key.
    """
    name = folder_name.lower()
    if name.startswith("adenocarcinoma"):
        return "adenocarcinoma"
    if name.startswith("large.cell") or name.startswith("large_cell"):
        return "large_cell_carcinoma"
    if name.startswith("normal"):
        return "normal"
    if name.startswith("squamous"):
        return "squamous_cell_carcinoma"
    return None


@dataclass
class ModelMeta:
    """How to feed a model. `preprocess` is one of:
    'rescale'  - pixels / 255 (the custom CNN in the notebooks)
    'resnet50' | 'vgg16' | 'inception_v3' - keras.applications preprocess_input
    """

    preprocess: str = "rescale"
    classes: list[str] = field(default_factory=lambda: list(CLASSES))
    image_size: int = IMAGE_SIZE

    @classmethod
    def for_model(cls, model_path: Path) -> "ModelMeta":
        sidecar = Path(model_path).with_suffix(".json")
        if sidecar.exists():
            return cls(**json.loads(sidecar.read_text()))
        return cls()

    def save(self, model_path: Path) -> None:
        Path(model_path).with_suffix(".json").write_text(json.dumps(self.__dict__, indent=2))


def load_image(source, size: int = IMAGE_SIZE) -> np.ndarray:
    """Read an image (path, file object or PIL image) as an RGB float array (H, W, 3) in 0-255."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def preprocess(batch: np.ndarray, method: str) -> np.ndarray:
    """Apply the model-specific input transform to a (N, H, W, 3) 0-255 RGB batch."""
    if method == "rescale":
        return batch / 255.0
    from tensorflow.keras import applications  # heavy import only when needed

    fn = {
        "resnet50": applications.resnet50.preprocess_input,
        "vgg16": applications.vgg16.preprocess_input,
        "inception_v3": applications.inception_v3.preprocess_input,
    }.get(method)
    if fn is None:
        raise ValueError(f"Unknown preprocess method: {method}")
    return fn(batch.copy())


class Classifier:
    """Lazy-loading wrapper so the web app starts even before a model exists."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.meta = ModelMeta.for_model(self.model_path)
        self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None or self.model_path.exists()

    def _load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No model at {self.model_path}. Train one with `python train.py` "
                    "or set LUNG_MODEL_PATH."
                )
            from tensorflow import keras

            self._model = keras.models.load_model(self.model_path)
        return self._model

    def predict(self, image) -> dict:
        """Return {'label', 'display_name', 'confidence', 'probabilities': [{name, probability}]}
        with probabilities sorted from most to least likely."""
        x = load_image(image, self.meta.image_size)[None, ...]
        x = preprocess(x, self.meta.preprocess)
        probs = np.asarray(self._load().predict(x, verbose=0))[0].astype(float)
        order = np.argsort(probs)[::-1]
        top = self.meta.classes[int(order[0])]
        return {
            "label": top,
            "display_name": DISPLAY_NAMES.get(top, top),
            "confidence": float(probs[order[0]]),
            "probabilities": [
                {"name": DISPLAY_NAMES.get(self.meta.classes[i], self.meta.classes[i]),
                 "probability": float(probs[i])}
                for i in order
            ],
        }
