"""Reproducible training script for the best model from the notebooks.

Transfer learning on the Kaggle "Chest CT-Scan images" dataset with a
pretrained backbone (ResNet50 by default):

  phase 1 - train a new classification head on the frozen backbone
  phase 2 - fine-tune the top of the backbone with a low learning rate

Unlike the notebooks, this script uses the dataset's own train/valid/test
split, so the reported test accuracy comes from images never used for
training or model selection.

Usage:
    python train.py --data data/ --backbone resnet50
Outputs (in --out, default models/):
    lung_ct_<backbone>.keras   trained model
    lung_ct_<backbone>.json    preprocessing + class order (read by the app)
    metrics.json               test accuracy and per-class report
    confusion_matrix.png
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from lungct import CLASSES, DISPLAY_NAMES, IMAGE_SIZE, ModelMeta, canonical_class, load_image, preprocess

BACKBONES = {
    "resnet50": ("ResNet50", "resnet50"),
    "vgg16": ("VGG16", "vgg16"),
    "inception_v3": ("InceptionV3", "inception_v3"),
}


def load_split(split_dir: Path):
    """Load every image in split_dir/<class folder>/ as (X 0-255 RGB, y int)."""
    xs, ys = [], []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        key = canonical_class(folder.name)
        if key is None:
            print(f"  skipping unknown folder {folder.name}")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                xs.append(load_image(f, IMAGE_SIZE))
                ys.append(CLASSES.index(key))
    return np.stack(xs), np.array(ys)


def build_model(backbone: str):
    from tensorflow import keras

    app_name, _ = BACKBONES[backbone]
    base = getattr(keras.applications, app_name)(
        weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base.trainable = False

    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.05)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"), help="folder containing train/ valid/ test/")
    ap.add_argument("--backbone", choices=BACKBONES, default="resnet50")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--finetune-layers", type=int, default=30, help="top backbone layers to unfreeze")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    method = BACKBONES[args.backbone][1]
    print("Loading data ...")
    splits = {}
    for name in ("train", "valid", "test"):
        X, y = load_split(args.data / name)
        splits[name] = (preprocess(X, method), y)
        print(f"  {name:5s}: {len(y):4d} images  class counts {np.bincount(y, minlength=len(CLASSES)).tolist()}")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["valid"], splits["test"]

    model, base = build_model(args.backbone)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\nPhase 1: training the classification head (backbone frozen)")
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.head_epochs,
              batch_size=args.batch_size, callbacks=callbacks, verbose=2)

    print(f"\nPhase 2: fine-tuning the top {args.finetune_layers} backbone layers")
    base.trainable = True
    for layer in base.layers[:-args.finetune_layers]:
        layer.trainable = False
    for layer in base.layers:  # keep BatchNorm statistics frozen on a small dataset
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.finetune_epochs,
              batch_size=args.batch_size, callbacks=callbacks, verbose=2)

    # ---- evaluate on the untouched test split --------------------------------
    from sklearn.metrics import classification_report, confusion_matrix

    pred = model.predict(Xte, verbose=0).argmax(axis=1)
    names = [DISPLAY_NAMES[c] for c in CLASSES]
    report = classification_report(yte, pred, target_names=names, output_dict=True, zero_division=0)
    print("\nTest set\n" + classification_report(yte, pred, target_names=names, zero_division=0))

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"lung_ct_{args.backbone}.keras"
    model.save(model_path)
    ModelMeta(preprocess=method).save(model_path)
    (args.out / "metrics.json").write_text(json.dumps(
        {"backbone": args.backbone, "test_accuracy": report["accuracy"], "report": report}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=f"{args.backbone} — test set")
    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=130)

    print(f"Saved {model_path} (+ .json, metrics.json, confusion_matrix.png)")


if __name__ == "__main__":
    main()
