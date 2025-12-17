import argparse
import json
import os
from collections import Counter

import numpy as np
import piexif
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def _get_image_labels(image_path: str) -> dict:
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get("exif", b""))
        description = (
            exif_dict.get("0th", {})
            .get(piexif.ImageIFD.ImageDescription, b"{}")
            .decode("utf-8")
        )
        labels = json.loads(description)
        if not isinstance(labels, dict):
            return {}
        return labels
    except Exception:
        return {}


def _filter_excluded_samples(image_paths: list[str], labels_list: list[dict]) -> tuple[list[str], list[dict], int]:
    kept_paths: list[str] = []
    kept_labels: list[dict] = []
    excluded = 0
    for p, labels in zip(image_paths, labels_list):
        if labels.get("cat") and labels.get("morris") and (not labels.get("entering")):
            excluded += 1
            continue
        kept_paths.append(p)
        kept_labels.append(labels)
    return kept_paths, kept_labels, excluded


def _filter_by_negative_policy(
    image_paths: list[str],
    labels_list: list[dict],
    negative_policy: str,
) -> tuple[list[str], list[dict], int]:
    policy = str(negative_policy or "all")
    if policy == "all":
        return image_paths, labels_list, 0
    if policy != "cat_entering_only":
        return image_paths, labels_list, 0

    kept_paths: list[str] = []
    kept_labels: list[dict] = []
    dropped = 0
    for p, labels in zip(image_paths, labels_list):
        prey = bool(labels.get("prey", False))
        cat = bool(labels.get("cat", False))
        enter = bool(labels.get("entering", False))
        if prey or (cat and enter):
            kept_paths.append(p)
            kept_labels.append(labels)
        else:
            dropped += 1
    return kept_paths, kept_labels, dropped


def _convert_labels(labels_list: list[dict], class_count: int) -> tuple[list[str], list[int]]:
    if class_count == 2:
        classes = ["prey", "not_prey"]
    elif class_count == 3:
        classes = ["prey", "not_prey", "not_cat"]
    else:
        raise ValueError("class_count must be 2 or 3")

    encoded: list[int] = []
    for labels in labels_list:
        prey = bool(labels.get("prey", False))
        cat = bool(labels.get("cat", False))
        enter = bool(labels.get("entering", False))
        if prey:
            label = "prey"
        elif cat and enter:
            label = "not_prey"
        elif class_count == 3:
            label = "not_cat"
        else:
            label = "not_prey"
        encoded.append(classes.index(label))
    return classes, encoded


def _load_split_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("split_manifest must be a JSON object")
    out: dict = {}
    for k in ("train", "val", "test"):
        v = data.get(k)
        if v is None:
            continue
        if not isinstance(v, list):
            raise ValueError(f"split_manifest key '{k}' must be a list")
        out[k] = [str(x) for x in v]
    if "train" not in out or "val" not in out:
        raise ValueError("split_manifest must contain at least 'train' and 'val'")
    return out


def _preprocess_u8(path: str) -> np.ndarray:
    # Match train_simple_model.py preprocessing:
    # - grayscale
    # - resolution-dependent center crop (384 for >=480, 192 for >=240)
    # - resize to 96x96 with nearest
    with Image.open(path) as img:
        img = img.convert("L")
        w, h = img.size
        min_dim = min(w, h)
        if min_dim >= 480:
            crop = 384
        elif min_dim >= 240:
            crop = 192
        else:
            crop = min_dim
        crop = min(crop, min_dim)
        left = (w - crop) // 2
        top = (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((96, 96), resample=Image.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
        return arr[:, :, None]


def _load_u8_array(paths: list[str]) -> np.ndarray:
    arr = np.stack([_preprocess_u8(p) for p in paths], axis=0)
    return arr.astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved local (.keras) model on a split manifest.")
    parser.add_argument("--model", required=True, help="Path to a .keras model (e.g. best_recall_model.keras)")
    parser.add_argument("--split_manifest", required=True, help="Path to split manifest JSON")
    parser.add_argument("--dataset_dir", default="dataset/images", help="Dataset images directory")
    parser.add_argument("--class_count", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--negative_policy",
        choices=["all", "cat_entering_only"],
        default="all",
        help="Must match how the model was trained.",
    )
    parser.add_argument(
        "--prey_threshold",
        type=float,
        default=None,
        help="Threshold on P(prey) for binary classification. If omitted, will read from --metrics_json.",
    )
    parser.add_argument(
        "--metrics_json",
        default=None,
        help="Optional path to a metrics.json produced by train_simple_model.py (to pull prey_threshold).",
    )
    args = parser.parse_args()

    if args.prey_threshold is None and not args.metrics_json:
        raise SystemExit("Provide --prey_threshold or --metrics_json")

    prey_threshold = args.prey_threshold
    if prey_threshold is None:
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            m = json.load(f)
        prey_threshold = float(m["training_params"]["prey_threshold"])

    split = _load_split_manifest(args.split_manifest)

    # Load dataset basenames -> (path, label)
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)

    image_paths: list[str] = []
    labels_list: list[dict] = []
    for fn in os.listdir(dataset_dir):
        if fn.lower().endswith((".jpg", ".jpeg")):
            p = os.path.join(dataset_dir, fn)
            image_paths.append(p)
            labels_list.append(_get_image_labels(p))

    image_paths, labels_list, excluded = _filter_excluded_samples(image_paths, labels_list)
    image_paths, labels_list, dropped = _filter_by_negative_policy(image_paths, labels_list, args.negative_policy)
    classes, labels_encoded = _convert_labels(labels_list, args.class_count)

    by_name = {os.path.basename(p): (p, lbl) for p, lbl in zip(image_paths, labels_encoded)}

    def resolve(names: list[str]) -> tuple[list[str], list[int]]:
        paths: list[str] = []
        labels: list[int] = []
        missing = 0
        for n in names:
            key = os.path.basename(str(n))
            if key in by_name:
                p, lbl = by_name[key]
                paths.append(p)
                labels.append(int(lbl))
            else:
                missing += 1
        if missing:
            print(f"Warning: {missing} manifest files missing after filters")
        return paths, labels

    test_paths, test_labels = resolve(split.get("test", []))
    if not test_paths:
        raise SystemExit("No test samples resolved from split manifest")

    # Load model
    # We only need inference; avoid requiring custom metrics/losses at load time.
    model = tf.keras.models.load_model(args.model, compile=False)

    x = _load_u8_array(test_paths)
    probs = model.predict(x, verbose=0)

    prey_index = classes.index("prey")
    prey_probs = np.asarray(probs[:, prey_index], dtype=np.float32)

    if args.class_count == 2:
        not_prey_index = classes.index("not_prey")
        pred = np.where(prey_probs >= float(prey_threshold), prey_index, not_prey_index).astype(int)
        labels = [0, 1]
    else:
        pred = np.argmax(probs, axis=1).astype(int)
        labels = list(range(len(classes)))

    report = classification_report(
        np.asarray(test_labels, dtype=np.int32),
        pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(np.asarray(test_labels, dtype=np.int32), pred, labels=labels)

    # Binary AUCs
    try:
        y_true_bin = (np.asarray(test_labels, dtype=np.int32) == prey_index).astype(np.int32)
        pos = prey_probs[y_true_bin == 1]
        neg = prey_probs[y_true_bin == 0]
        roc_auc = float(roc_auc_score(y_true_bin, prey_probs)) if (pos.size and neg.size) else None
        pr_auc = float(average_precision_score(y_true_bin, prey_probs)) if (pos.size and neg.size) else None
    except Exception:
        roc_auc, pr_auc = None, None

    print("\n=== EVAL (test) ===")
    print(f"excluded={excluded} dropped_by_policy={dropped}")
    print(f"test_size={len(test_paths)} class_counts={dict(Counter(test_labels))}")
    if args.class_count == 2:
        print(f"prey_threshold={float(prey_threshold):.6f}")
    print(f"roc_auc={roc_auc} pr_auc={pr_auc}")
    print("prey:", report.get("prey"))
    print("confusion_matrix:\n", cm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
