import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import piexif
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image


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
        return labels if isinstance(labels, dict) else {}
    except Exception:
        return {}


def _filter_excluded_samples(image_paths: list[str], labels_list: list[dict]) -> tuple[list[str], list[dict]]:
    kept_paths: list[str] = []
    kept_labels: list[dict] = []
    for p, labels in zip(image_paths, labels_list):
        if labels.get("cat") and labels.get("morris") and (not labels.get("entering")):
            continue
        kept_paths.append(p)
        kept_labels.append(labels)
    return kept_paths, kept_labels


def _filter_by_negative_policy(
    image_paths: list[str],
    labels_list: list[dict],
    negative_policy: str,
) -> tuple[list[str], list[dict]]:
    policy = str(negative_policy or "all")
    if policy == "all":
        return image_paths, labels_list
    if policy != "cat_entering_only":
        return image_paths, labels_list

    kept_paths: list[str] = []
    kept_labels: list[dict] = []
    for p, labels in zip(image_paths, labels_list):
        prey = bool(labels.get("prey", False))
        cat = bool(labels.get("cat", False))
        enter = bool(labels.get("entering", False))
        if prey or (cat and enter):
            kept_paths.append(p)
            kept_labels.append(labels)
    return kept_paths, kept_labels


def _convert_binary_label(labels: dict, negative_policy: str) -> int | None:
    # Return 0=prey, 1=not_prey, or None if excluded by policy.
    prey = bool(labels.get("prey", False))
    if prey:
        return 0

    policy = str(negative_policy or "all")
    if policy == "cat_entering_only":
        cat = bool(labels.get("cat", False))
        enter = bool(labels.get("entering", False))
        if cat and enter:
            return 1
        return None

    return 1


def _fast_mean_u8(path: str) -> float | None:
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            img = img.resize((32, 32))
            arr = np.asarray(img, dtype=np.uint8)
            return float(arr.mean())
    except Exception:
        return None


def _preprocess_u8(path: str) -> np.ndarray:
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


def representative_data_gen(
    dataset_dir: str,
    negative_policy: str,
    seed: int,
    rep_bins: int = 6,
    rep_per_cell: int = 10,
):
    rng = random.Random(seed)

    image_paths: list[str] = []
    labels_list: list[dict] = []
    for fn in os.listdir(dataset_dir):
        if fn.lower().endswith((".jpg", ".jpeg")):
            p = os.path.join(dataset_dir, fn)
            image_paths.append(p)
            labels_list.append(_get_image_labels(p))

    image_paths, labels_list = _filter_excluded_samples(image_paths, labels_list)
    image_paths, labels_list = _filter_by_negative_policy(image_paths, labels_list, negative_policy)

    # Bucket by (label, brightness_bin)
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for p, labels in zip(image_paths, labels_list):
        y = _convert_binary_label(labels, negative_policy)
        if y is None:
            continue
        mean_u8 = _fast_mean_u8(p)
        if mean_u8 is None:
            continue
        b = int(mean_u8 * rep_bins / 256.0)
        b = max(0, min(rep_bins - 1, b))
        buckets[(int(y), int(b))].append(p)

    chosen: list[str] = []
    for (y, b), paths in buckets.items():
        rng.shuffle(paths)
        chosen.extend(paths[:rep_per_cell])

    rng.shuffle(chosen)

    def _gen():
        for p in chosen:
            x = _preprocess_u8(p)[None, ...]
            yield [x.astype(np.uint8)]

    return _gen


def make_export_model(model: tf.keras.Model, export_logit_scale: float, export_output: str) -> tf.keras.Model:
    """Create an eval/export model (Keras 3 safe).

    Modes:
    - probs: softmax probabilities
    - logits_margin: prey_logit - not_prey_logit (binary only)
    """

    export_output = str(export_output or "probs")
    if export_output not in ("probs", "logits_margin"):
        raise ValueError(f"Unsupported export_output: {export_output}")

    if export_output == "logits_margin":
        # Only meaningful for 2-class models.
        out_shape = getattr(model, "output_shape", None)
        out_dim = None
        if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2:
            out_dim = out_shape[-1]
        if out_dim is not None and int(out_dim) != 2:
            raise ValueError("logits_margin export requires a 2-class model")

    try:
        logits_tensor = model.get_layer("logits").output
    except Exception as e:
        raise RuntimeError("Model does not expose a 'logits' layer") from e

    scaled_logits = logits_tensor
    if export_logit_scale is not None and abs(float(export_logit_scale) - 1.0) >= 1e-9:
        scaled_logits = layers.Rescaling(
            float(export_logit_scale),
            offset=0.0,
            name="export_logit_rescale",
        )(logits_tensor)

    if export_output == "probs":
        probs = layers.Activation("softmax", name="probs")(scaled_logits)
        if scaled_logits is logits_tensor:
            return model
        return tf.keras.Model(model.input, probs, name="export_probs_scaled")

    prey_logit = scaled_logits[:, 0:1]
    not_prey_logit = scaled_logits[:, 1:2]
    margin = layers.Subtract(name="logits_margin")([prey_logit, not_prey_logit])
    return tf.keras.Model(model.input, margin, name="export_logits_margin")


def write_cc_from_tflite(tflite_path: str, cc_path: str, var_name: str = "g_model_data") -> None:
    data = open(tflite_path, "rb").read()
    with open(cc_path, "w", encoding="utf-8") as f:
        f.write('#include <cstdint>\n\n')
        f.write(f'alignas(16) const unsigned char {var_name}[] = {{\n')
        for i in range(0, len(data), 12):
            chunk = data[i : i + 12]
            f.write("  " + ", ".join(str(b) for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f'const unsigned int {var_name}_len = {len(data)};\n')


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained local model (.keras) to full-int8 TFLite.")
    parser.add_argument("--model", required=True, help="Path to a trained .keras model")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--dataset_dir", default="dataset/images")
    parser.add_argument("--negative_policy", choices=["all", "cat_entering_only"], default="all")
    parser.add_argument("--export_logit_scale", type=float, default=1.0)
    parser.add_argument("--export_output", choices=["probs", "logits_margin"], default="probs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = tf.keras.models.load_model(args.model, compile=False)
    export_model = make_export_model(
        model,
        export_logit_scale=float(args.export_logit_scale),
        export_output=args.export_output,
    )

    savedmodel_dir = os.path.join(args.out_dir, "my_model")
    export_model.export(savedmodel_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)

    converter.representative_dataset = representative_data_gen(
        dataset_dir=dataset_dir,
        negative_policy=args.negative_policy,
        seed=int(args.seed),
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8 if args.export_output == "logits_margin" else tf.uint8

    tflite_model = converter.convert()

    tflite_path = os.path.join(args.out_dir, "model_quant.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    cc_path = os.path.join(args.out_dir, "model_quant.cc")
    write_cc_from_tflite(tflite_path, cc_path)

    print(f"Wrote {tflite_path} ({len(tflite_model)/1024:.2f} KB)")
    print(f"Wrote {cc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
