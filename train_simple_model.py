
import os
# Disable CUDA if not using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import piexif
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
import argparse
from collections import Counter
from collections import defaultdict
import random
import datetime

# -----------------------------
# Args & constants
# -----------------------------
parser = argparse.ArgumentParser(description='Train a small grayscale 96x96 model (prey-first output) and export TFLite + .cc')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for initial training.')
parser.add_argument('--learning_rate', type=str, default='1e-3', help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--class_count', type=int, choices=[2, 3], default=2,
                    help='Number of output classes. Use 2 for [prey, not_prey] or 3 for [prey, not_prey, not_cat].')
parser.add_argument('--prefer_recall', action='store_true',
                    help='Use the best recall checkpoint (binary only). Defaults to best accuracy.')
parser.add_argument('--export_logit_scale', type=float, default=1.0,
                    help='Scale logits before softmax at export/eval time (improves probability spread for quantized uint8 output).')
parser.add_argument('--export_output', choices=['probs', 'logits_margin'], default='probs',
                    help='Export output tensor. probs=softmax probabilities (uint8 output). logits_margin=prey_logit-not_prey_logit (int8 output). Only supported for 2-class.')
parser.add_argument('--max_samples_per_class', type=int, default=0,
                    help='Optional cap per class for training/validation (0 = use all samples).')
parser.add_argument('--run_id', type=str, default=None,
                    help='Optional explicit run/version id (used by app.py).')
parser.add_argument('--width_mult', type=float, default=0.75,
                    help='Width multiplier to shrink/expand channel counts for a lighter/heavier model.')
parser.add_argument('--dropout', type=float, default=0.30,
                    help='Dropout rate applied before the classifier head.')
parser.add_argument('--val_split', type=float, default=0.20, help='Fraction for validation split (0-0.5).')
parser.add_argument('--early_stop_patience', type=int, default=5, help='Early stopping patience (epochs).')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW weight decay.')
parser.add_argument('--augment', choices=['off', 'light', 'medium'], default='light', help='Data augmentation level.')
parser.add_argument('--use_class_weights', choices=['on', 'off'], default='on', help='Toggle class weighting.')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0-0.2).')
parser.add_argument('--lr_schedule', choices=['constant', 'cosine', 'step'], default='cosine', help='LR schedule type.')
parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup epochs for LR schedule.')
args = parser.parse_args()

EPOCHS = args.epochs
INIT_LR = float(args.learning_rate)
BATCH_SIZE = args.batch_size
SEED = args.seed
CLASS_COUNT = args.class_count
PREFER_RECALL = bool(args.prefer_recall and CLASS_COUNT == 2)
EXPORT_LOGIT_SCALE = max(1e-6, float(args.export_logit_scale))
EXPORT_OUTPUT = str(args.export_output)
MAX_SAMPLES_PER_CLASS = max(0, args.max_samples_per_class)
RUN_ID = args.run_id
WIDTH_MULT = max(0.4, float(args.width_mult))  # clamp to avoid degenerate shapes
DROPOUT_RATE = min(max(0.0, float(args.dropout)), 0.8)
VAL_SPLIT = min(max(args.val_split, 0.05), 0.4)
EARLY_STOP_PATIENCE = max(1, min(int(args.early_stop_patience), 20))
WEIGHT_DECAY = max(0.0, float(args.weight_decay))
AUGMENT = args.augment
USE_CLASS_WEIGHTS = (args.use_class_weights == 'on')
LABEL_SMOOTHING = min(max(args.label_smoothing, 0.0), 0.2)
LR_SCHEDULE = args.lr_schedule
WARMUP_EPOCHS = max(0, min(int(args.warmup_epochs), 20))
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths (per-version under models/local/<run_id>)
# -----------------------------
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_NAME = 'my_simple_model_quant'
MODELS_ROOT = 'models'
LOCAL_MODELS_DIR = os.path.join(MODELS_ROOT, 'local')

# -----------------------------
# Task setup
# -----------------------------
if CLASS_COUNT == 3:
    CLASSES = ['prey', 'not_prey', 'not_cat']
else:
    CLASSES = ['prey', 'not_prey']

if CLASS_COUNT != 2 and EXPORT_OUTPUT != 'probs':
    print("Warning: --export_output logits_margin only supported for 2-class; forcing probs.")
    EXPORT_OUTPUT = 'probs'
IMG_SIZE = (96, 96)

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.total_epochs * 100)
        print(f'\nPROGRESS:{progress}', flush=True)

# -----------------------------
# Data I/O
# -----------------------------
def get_image_labels(image_path):
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        description = exif_dict['0th'].get(piexif.ImageIFD.ImageDescription, b'{}').decode('utf-8')
        labels = json.loads(description)
    except Exception as e:
        print(f"Error reading labels from {image_path}: {e}")
        labels = {"cat": False, "morris": False, "entering": False, "prey": False}
    return labels

def load_dataset(dataset_dir):
    image_paths = []
    labels_list = []
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(dataset_dir, filename)
            labels = get_image_labels(image_path)
            image_paths.append(image_path)
            labels_list.append(labels)
    return image_paths, labels_list


def filter_excluded_samples(image_paths, labels_list):
    """Exclude samples we don't want the simple model to learn.

    Currently drops "cat_morris_leaving" frames (cat=True, morris=True, entering=False),
    since those will be handled by an ESP-side heuristic.
    """
    kept_paths = []
    kept_labels = []
    excluded = 0
    for p, labels in zip(image_paths, labels_list):
        is_cat = bool(labels.get('cat', False))
        is_morris = bool(labels.get('morris', False))
        is_entering = bool(labels.get('entering', False))
        if is_cat and is_morris and (not is_entering):
            excluded += 1
            continue
        kept_paths.append(p)
        kept_labels.append(labels)
    return kept_paths, kept_labels, excluded

def convert_labels(labels_list):
    labels_encoded = []
    for labels in labels_list:
        prey = bool(labels.get('prey', False))
        cat = bool(labels.get('cat', False))
        enter = bool(labels.get('entering', False))
        if prey:
            label = 'prey'
        elif cat and enter:
            label = 'not_prey'
        elif CLASS_COUNT == 3:
            label = 'not_cat'
        else:
            label = 'not_prey'
        labels_encoded.append(CLASSES.index(label))
    return labels_encoded


def limit_samples_per_class(image_paths, labels_encoded, max_per_class, seed=0):
    if max_per_class <= 0:
        return image_paths, labels_encoded

    grouped = defaultdict(list)
    for path, label in zip(image_paths, labels_encoded):
        grouped[label].append((path, label))

    rng = random.Random(seed)
    limited_pairs = []
    for label, items in grouped.items():
        rng.shuffle(items)
        limited_pairs.extend(items[:max_per_class])

    rng.shuffle(limited_pairs)
    limited_paths = [p for p, _ in limited_pairs]
    limited_labels = [lbl for _, lbl in limited_pairs]
    return limited_paths, limited_labels

# -----------------------------
# Preprocessing (no external /255 — use model Rescaling layer instead)
# -----------------------------
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # grayscale
    # Resolution-dependent center crop to match how datasets were captured:
    # - legacy 640x480 images used a 384x384 center crop
    # - new 320x240 images use a 192x192 center crop
    # Then resize to 96x96 (nearest) like the ESP32 local model input.
    image = tf.cast(image, tf.uint8)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    min_dim = tf.minimum(h, w)

    desired_crop = tf.where(
        min_dim >= 480,
        tf.constant(384, dtype=min_dim.dtype),
        tf.where(
            min_dim >= 240,
            tf.constant(192, dtype=min_dim.dtype),
            min_dim,
        ),
    )
    crop_size = tf.minimum(desired_crop, min_dim)

    offset_y = (h - crop_size) // 2
    offset_x = (w - crop_size) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_size, crop_size)
    image = tf.image.resize(image, IMG_SIZE, method='nearest')
    image = tf.cast(image, tf.uint8)
    # keep dtype uint8; Rescaling layer will scale to 0..1
    return image, label

@tf.function
def adjust_gamma(img):
    # gamma in [0.8, 1.25]
    g = tf.random.uniform([], 0.8, 1.25)
    img = tf.image.adjust_gamma(tf.cast(img, tf.float32)/255.0, gamma=g)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.cast(img*255.0, tf.uint8)
    return img

def preprocess_image_train(image_path, label):
    image, label = preprocess_image(image_path, label)
    if AUGMENT != 'off':
        image = tf.image.random_brightness(image, 0.04)
        image = tf.image.random_contrast(image, 0.95, 1.05)
        if AUGMENT == 'medium':
            image = adjust_gamma(image)
    return image, label

def preprocess_image_val(image_path, label):
    image, label = preprocess_image(image_path, label)
    return image, label

# Representative dataset for INT8: feed uint8 in [0..255]
def representative_data_gen():
    # Keep this self-contained so importing the module works.
    # For quantization stability, stratify the representative set across
    # brightness (day/night/IR) while reusing the exact preprocess pipeline.
    rep_bins = 6
    rep_per_cell = 10  # per (label, brightness bin)
    rng = random.Random(SEED)

    def _fast_mean_u8(path: str) -> float | None:
        """Fast approximate brightness for binning (0..255).

        Using PIL here is *much* faster than running the full TF preprocess
        for every image during TFLite conversion.
        """
        try:
            with Image.open(path) as img:
                img = img.convert('L')
                img = img.resize((32, 32))
                arr = np.asarray(img, dtype=np.uint8)
                return float(arr.mean())
        except Exception:
            return None

    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    # Apply the same exclusions as training so calibration matches the task.
    image_paths, labels_list, _excluded = filter_excluded_samples(image_paths, labels_list)
    labels_encoded = convert_labels(labels_list)

    # Bucket by (label, brightness_bin)
    buckets = defaultdict(list)
    bin_counts = [0] * rep_bins
    for idx, (path, label) in enumerate(zip(image_paths, labels_encoded)):
        mean_u8 = _fast_mean_u8(path)
        if mean_u8 is None:
            continue

        bin_idx = int(mean_u8 * rep_bins / 256.0)
        bin_idx = max(0, min(rep_bins - 1, bin_idx))
        bin_counts[bin_idx] += 1
        buckets[(int(label), bin_idx)].append(path)

        # Light progress output so long conversions don't look stuck.
        if (idx + 1) % 200 == 0:
            print(f"Representative scan: {idx + 1}/{len(image_paths)}", flush=True)

    sampled_paths = []
    for (label, bin_idx), paths in buckets.items():
        rng.shuffle(paths)
        sampled_paths.extend(paths[:min(len(paths), rep_per_cell)])

    rng.shuffle(sampled_paths)
    if sampled_paths:
        # One-line debug summary during conversion.
        print(f"Representative scan complete: {len(image_paths)} images, brightness bins={bin_counts}", flush=True)
        print(f"Representative set: {len(sampled_paths)} samples (target per cell={rep_per_cell})", flush=True)

    for image_path in sampled_paths:
        img, _ = preprocess_image(image_path, 0)
        arr = tf.cast(img, tf.uint8).numpy()  # (96,96,1)
        arr = np.expand_dims(arr, axis=0)     # (1,96,96,1)
        yield [arr]

# -----------------------------
# Model (tiny depthwise-separable CNN)
# -----------------------------
def _dw_sep_block(filters, stride=2, dropout=0.0):
    # Depthwise + pointwise with stride for downsampling; avoids extra pooling ops.
    def apply(x):
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        return x
    return apply

def _scaled_channels(base):
    return max(8, int(round(base * WIDTH_MULT)))

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.uint8)
    x = layers.Rescaling(1./255.0)(inputs)  # keep uint8 input for INT8 quantization

    # Lightweight stem
    x = layers.Conv2D(_scaled_channels(16), 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Slimmed depthwise blocks
    x = _dw_sep_block(_scaled_channels(24), stride=2, dropout=DROPOUT_RATE * 0.3)(x)
    x = _dw_sep_block(_scaled_channels(32), stride=2, dropout=DROPOUT_RATE * 0.3)(x)
    x = _dw_sep_block(_scaled_channels(40), stride=1, dropout=DROPOUT_RATE * 0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    logits = layers.Dense(len(CLASSES), activation=None, name='logits')(x)
    outputs = layers.Activation('softmax', name='probs')(logits)
    return tf.keras.Model(inputs, outputs)


def make_export_model(
    base_model: tf.keras.Model,
    *,
    export_logit_scale: float,
    export_output: str = 'probs',
) -> tf.keras.Model:
    """Create an eval/export model.

    Modes:
    - probs: softmax probabilities (useful for metrics + backwards-compatible uint8 output).
    - logits_margin: prey_logit - not_prey_logit (better for quantized output resolution).
    """

    export_output = str(export_output or 'probs')
    if export_output not in ('probs', 'logits_margin'):
        raise ValueError(f"Unsupported export_output: {export_output}")

    if export_output == 'logits_margin' and CLASS_COUNT != 2:
        raise ValueError("logits_margin export requires 2-class model")

    try:
        logits_tensor = base_model.get_layer('logits').output
    except Exception as e:
        raise RuntimeError(
            "Model does not expose a 'logits' layer; cannot build export model. "
            "Re-train with the updated script."
        ) from e

    scaled_logits = logits_tensor
    if export_logit_scale is not None and abs(float(export_logit_scale) - 1.0) >= 1e-9:
        scaled_logits = layers.Rescaling(float(export_logit_scale), offset=0.0, name='export_logit_rescale')(logits_tensor)

    if export_output == 'probs':
        probs = layers.Activation('softmax', name='probs')(scaled_logits)
        if scaled_logits is logits_tensor:
            return base_model
        return tf.keras.Model(base_model.input, probs, name=f"export_probs_scaled_x{float(export_logit_scale):g}")

    # logits_margin
    prey_index = CLASSES.index('prey')
    not_prey_index = CLASSES.index('not_prey')
    prey_logit = scaled_logits[:, prey_index:prey_index + 1]
    not_prey_logit = scaled_logits[:, not_prey_index:not_prey_index + 1]
    margin = layers.Subtract(name='logits_margin')([prey_logit, not_prey_logit])
    return tf.keras.Model(base_model.input, margin, name=f"export_logits_margin_x{float(export_logit_scale):g}")

# -----------------------------
# Training
# -----------------------------
if __name__ == '__main__':
    # Create versioned output directories for this run (use provided run_id if any)
    run_id = RUN_ID or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(LOCAL_MODELS_DIR, run_id)
    reports_dir = os.path.join(version_dir, 'reports')
    images_dir = os.path.join(reports_dir, 'images')
    model_dir = os.path.join(version_dir, 'model')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Single JSON report file for this version
    metrics_json_path = os.path.join(reports_dir, 'metrics.json')

    # Load dataset
    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    image_paths, labels_list, excluded = filter_excluded_samples(image_paths, labels_list)
    if excluded:
        print(f"Excluded {excluded} cat_morris_leaving samples from training dataset.")
    labels_encoded = convert_labels(labels_list)

    original_class_counts = Counter(labels_encoded)
    print("Original class distribution:", {CLASSES[label]: count for label, count in original_class_counts.items()})

    if MAX_SAMPLES_PER_CLASS > 0:
        image_paths, labels_encoded = limit_samples_per_class(image_paths, labels_encoded, MAX_SAMPLES_PER_CLASS, seed=SEED)
        print(f"Applied max {MAX_SAMPLES_PER_CLASS} samples per class for training.")

    # Class distribution
    class_counts = Counter(labels_encoded)
    print("Class distribution (encoded):", class_counts)
    class_distribution = {CLASSES[label]: count for label, count in class_counts.items()}
    print("Class distribution (named):", class_distribution)

    # Split (stratified)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels_encoded, test_size=VAL_SPLIT, random_state=SEED, stratify=labels_encoded)

    # Make split sizes explicit (helps catch accidental tiny validation sets)
    try:
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        train_named = {CLASSES[int(k)]: int(v) for k, v in train_counts.items()}
        val_named = {CLASSES[int(k)]: int(v) for k, v in val_counts.items()}
        print(f"Train/val sizes: train={len(train_paths)} val={len(val_paths)} (val_split={VAL_SPLIT})")
        print(f"Train distribution: {train_named}")
        print(f"Val distribution:   {val_named}")
    except Exception:
        train_named = None
        val_named = None

    # Class weights
    class_weight_dict = None
    if USE_CLASS_WEIGHTS:
        unique_labels = np.unique(train_labels)
        class_weights_arr = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=train_labels
        )
        class_weight_dict = {int(label): weight for label, weight in zip(unique_labels, class_weights_arr)}
        for idx in range(len(CLASSES)):
            class_weight_dict.setdefault(idx, 1.0)

    # Datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))\
        .shuffle(buffer_size=2048, seed=SEED, reshuffle_each_iteration=True)\
        .map(preprocess_image_train, num_parallel_calls=AUTOTUNE)\
        .batch(BATCH_SIZE)\
        .prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))\
        .map(preprocess_image_val, num_parallel_calls=AUTOTUNE)\
        .batch(BATCH_SIZE)\
        .prefetch(AUTOTUNE)

    # Optional: dump a snapshot of preprocessed images for visual inspection
    try:
        train_dump_dir = os.path.join(images_dir, 'train_set')
        val_dump_dir = os.path.join(images_dir, 'val_set')
        os.makedirs(train_dump_dir, exist_ok=True)
        os.makedirs(val_dump_dir, exist_ok=True)

        def _save_sample_batch(dataset, dump_dir, prefix, max_batches=3):
            batch_idx = 0
            for batch_images, batch_labels in dataset.take(max_batches):
                batch_images_np = batch_images.numpy()
                batch_labels_np = batch_labels.numpy()
                for i in range(batch_images_np.shape[0]):
                    img = batch_images_np[i]
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img[:, :, 0]
                    img_pil = Image.fromarray(img.astype(np.uint8), mode='L')
                    label_idx = int(batch_labels_np[i])
                    label_name = CLASSES[label_idx] if 0 <= label_idx < len(CLASSES) else 'unknown'
                    filename = f"{prefix}_b{batch_idx}_i{i}_{label_name}.png"
                    img_pil.save(os.path.join(dump_dir, filename))
                batch_idx += 1

        _save_sample_batch(train_ds, train_dump_dir, 'train')
        _save_sample_batch(val_ds, val_dump_dir, 'val')
    except Exception as e:
        print(f"Warning: failed to dump preprocessed images: {e}")

    # Model & optimizer
    model = build_model()

    steps_per_epoch = max(1, len(train_paths)//BATCH_SIZE)
    total_steps = max(1, steps_per_epoch * EPOCHS)
    warmup_steps = max(0, min(WARMUP_EPOCHS, EPOCHS) * steps_per_epoch)

    def make_lr_schedule():
        if LR_SCHEDULE == 'cosine':
            base = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=INIT_LR, decay_steps=total_steps, alpha=1e-2
            )
        elif LR_SCHEDULE == 'step':
            boundaries = [int(total_steps * 0.5), int(total_steps * 0.75)]
            values = [INIT_LR, INIT_LR * 0.5, INIT_LR * 0.1]
            base = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        else:
            # Use a proper schedule object (not a Python lambda), since some
            # Keras optimizer paths may call the LR callable with no args.
            base = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[total_steps + 1],
                values=[INIT_LR, INIT_LR]
            )

        if warmup_steps <= 0:
            return base

        def schedule(step):
            step = tf.cast(step, tf.float32)
            base_val = base(step)
            warm = tf.constant(INIT_LR, dtype=tf.float32) * tf.minimum(1.0, step / float(warmup_steps))
            return tf.cond(step < warmup_steps, lambda: warm, lambda: base_val)

        return schedule

    lr_schedule = make_lr_schedule()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)

    # Metrics focused on 'prey' class
    prey_index = CLASSES.index('prey')
    precision_prey = tf.keras.metrics.Precision(class_id=prey_index, name='precision_prey')
    recall_prey = tf.keras.metrics.Recall(class_id=prey_index, name='recall_prey')

    def make_loss():
        try:
            return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
        except TypeError:
            # Fallback for older TF/Keras builds without label_smoothing support on sparse CCE
            if LABEL_SMOOTHING > 0:
                print("SparseCategoricalCrossentropy lacks label_smoothing; applying manual smoothing.")

                def smoothed_sparse_cce(y_true, y_pred):
                    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
                    y_true_one_hot = tf.one_hot(y_true, depth=CLASS_COUNT)
                    smooth = LABEL_SMOOTHING
                    y_true_smooth = y_true_one_hot * (1.0 - smooth) + smooth / float(CLASS_COUNT)
                    return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

                return smoothed_sparse_cce
            return tf.keras.losses.SparseCategoricalCrossentropy()

    loss = make_loss()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                 precision_prey, recall_prey]
    )

    # Callbacks - save checkpoints inside this run's versioned model directory
    checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_acc_model.keras'),
        monitor='val_accuracy', mode='max', save_best_only=True
    )
    checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_recall_model.keras'),
        monitor='val_recall_prey', mode='max', save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOP_PATIENCE, mode='min', restore_best_weights=True
    )
    progress_callback = ProgressCallback(total_epochs=EPOCHS)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_acc, checkpoint_recall, early_stopping, progress_callback],
        verbose=2
    )

    # We'll consider multiple candidates for export/evaluation:
    # - current in-memory weights (already restored to best val_loss by EarlyStopping)
    # - best-accuracy checkpoint
    # - best-recall checkpoint
    # Then choose the one that best matches our deployment objective.
    best_acc_path = os.path.join(model_dir, 'best_acc_model.keras')
    best_recall_path = os.path.join(model_dir, 'best_recall_model.keras')
    model_val_loss = model

    # Collect curves for metrics.json (no separate plot files)
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])

    # ---------------------------------
    # Evaluation & threshold selection
    # ---------------------------------
    # Build full val tensors for thresholding using the same preprocessing
    # as training/device (single source of truth).
    def load_val_array(paths):
        X = []
        for p in paths:
            img, _ = preprocess_image(p, 0)
            X.append(tf.cast(img, tf.uint8).numpy())
        X = np.stack(X, axis=0)  # (N,96,96,1) uint8
        return X
    val_images_u8 = load_val_array(val_paths)
    y_true_prey = (np.array(val_labels) == prey_index).astype(int)
    y_true = np.asarray(y_true_prey, dtype=np.int32)

    def _select_threshold(scores, y_true, *, mode: str = 'min_fp'):
        """Return (chosen_thr, best_stats) selecting a prey threshold.

        Modes:
        - 'min_fp' (default): minimize FP, then maximize recall, prefer smaller thresholds.
        - 'recall': maximize recall subject to an FP cap, avoiding trivial "all prey" when possible.
        """
        scores = np.asarray(scores, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.int32)

        n_total = int(y_true.size)
        n_neg = int(np.sum(y_true == 0))

        base = np.unique(scores).astype(np.float32)
        base.sort()
        if base.size == 0:
            return 0.0, None
        eps = 1e-6
        if base.size >= 2:
            mids = ((base[:-1] + base[1:]) * 0.5).astype(np.float32)
            thresholds = np.unique(np.concatenate((base, mids, [base[0] - eps, base[-1] + eps]))).astype(np.float32)
        else:
            thresholds = np.unique(np.array([base[0] - eps, base[0], base[0] + eps], dtype=np.float32))

        candidates = []
        for t in thresholds:
            pred = scores >= t
            fp = int(np.sum(pred & (y_true == 0)))
            tp = int(np.sum(pred & (y_true == 1)))
            fn = int(np.sum((~pred) & (y_true == 1)))
            pred_pos = int(np.sum(pred))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            candidates.append((float(t), fp, tp, fn, float(precision), float(recall), pred_pos))

        if not candidates:
            return 0.5, None

        def _stats_from_tuple(item, *, fp_cap_used=None):
            t, fp, tp, fn, precision, recall, pred_pos = item
            return {
                "threshold": float(t),
                "fp": int(fp),
                "tp": int(tp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "pred_pos": int(pred_pos),
                "pred_pos_rate": float(pred_pos / n_total) if n_total > 0 else None,
                "mode": str(mode),
                "fp_cap_used": int(fp_cap_used) if fp_cap_used is not None else None,
                "n_neg": int(n_neg),
                "n_total": int(n_total),
            }

        if mode == 'recall':
            # Avoid the trivial "everything prey" threshold when possible.
            non_trivial = [c for c in candidates if c[2] > 0 and c[6] < n_total]
            if not non_trivial:
                non_trivial = [c for c in candidates if c[2] > 0]

            # Start with a conservative FP cap and relax if needed.
            caps = [0.15, 0.25, 0.5, 1.0]
            for frac in caps:
                fp_cap = int(round(n_neg * frac))
                pool = [c for c in non_trivial if c[1] <= fp_cap]
                if not pool:
                    continue
                # Max recall, then fewer FP, then fewer predicted positives, then higher threshold.
                pool_sorted = sorted(pool, key=lambda c: (-c[5], c[1], c[6], -c[0]))
                best_item = pool_sorted[0]
                stats = _stats_from_tuple(best_item, fp_cap_used=fp_cap)
                return float(stats["threshold"]), stats

            # Fallback: best recall overall, but still prefer fewer FP and fewer predicted positives.
            pool_sorted = sorted(non_trivial, key=lambda c: (-c[5], c[1], c[6], -c[0])) if non_trivial else sorted(candidates, key=lambda c: (-c[5], c[1], c[6], -c[0]))
            best_item = pool_sorted[0]
            stats = _stats_from_tuple(best_item)
            return float(stats["threshold"]), stats

        # Default: minimize FP, but require TP>0 when possible to avoid all-not_prey.
        candidates_with_tp = [c for c in candidates if c[2] > 0]
        if candidates_with_tp:
            min_fp = min(c[1] for c in candidates_with_tp)
            pool = [c for c in candidates_with_tp if c[1] == min_fp]
        else:
            min_fp = min(c[1] for c in candidates)
            pool = [c for c in candidates if c[1] == min_fp]

        # Within the chosen FP group: maximize recall then prefer smaller thresholds.
        pool_sorted = sorted(pool, key=lambda c: (-c[5], c[0]))
        best_item = pool_sorted[0]
        stats = _stats_from_tuple(best_item, fp_cap_used=min_fp)
        return float(stats["threshold"]), stats

    def _eval_candidate(candidate_model, name):
        # Always evaluate selection metrics in probability-space.
        candidate_export_model = make_export_model(
            candidate_model,
            export_logit_scale=EXPORT_LOGIT_SCALE,
            export_output='probs',
        )

        x = val_images_u8
        if candidate_export_model.inputs[0].dtype == tf.float32:
            x = x.astype('float32') / 255.0

        probs_all = candidate_export_model.predict(x, batch_size=BATCH_SIZE, verbose=0)
        prey_probs = np.asarray(probs_all[:, prey_index], dtype=np.float32)
        chosen_thr, stats = _select_threshold(prey_probs, y_true, mode=('recall' if PREFER_RECALL else 'min_fp'))

        if CLASS_COUNT == 2:
            not_prey_index = CLASSES.index('not_prey')
            pred_labels_used = np.where(prey_probs >= chosen_thr, prey_index, not_prey_index).astype(int)
            cm_used = confusion_matrix(val_labels, pred_labels_used, labels=[0, 1])
        else:
            pred_labels_used = np.argmax(probs_all, axis=1)
            cm_used = confusion_matrix(val_labels, pred_labels_used, labels=list(range(len(CLASSES))))

        fp = int(stats["fp"]) if stats else 0
        tp = int(stats["tp"]) if stats else 0
        fn = int(stats["fn"]) if stats else 0
        recall = float(stats["recall"]) if stats else 0.0
        pred_pos = int(stats.get("pred_pos")) if isinstance(stats, dict) and stats.get("pred_pos") is not None else 0
        has_tp = tp > 0
        if PREFER_RECALL:
            # Prefer any model that yields at least one TP. Then maximize recall under FP cap.
            key = (0 if has_tp else 1, -recall, fp, pred_pos, float(chosen_thr))
        else:
            # Prefer any model that yields at least one TP. Then minimize FP, maximize recall.
            key = (0 if has_tp else 1, fp, -recall, -tp, float(chosen_thr))
        return {
            "name": str(name),
            "key": key,
            "threshold": float(chosen_thr),
            "stats": stats,
            "cm": cm_used,
            "probs": probs_all,
        }

    # Build candidate list.
    candidates = []
    candidates.append((model_val_loss, 'val_loss'))
    if os.path.exists(best_acc_path):
        try:
            candidates.append((tf.keras.models.load_model(best_acc_path), 'accuracy'))
        except Exception as e:
            print(f"Warning: failed to load accuracy checkpoint: {e}")
    if os.path.exists(best_recall_path):
        try:
            candidates.append((tf.keras.models.load_model(best_recall_path), 'recall'))
        except Exception as e:
            print(f"Warning: failed to load recall checkpoint: {e}")

    # If user explicitly requested recall, honor that (with fallback).
    selected = None
    if PREFER_RECALL:
        for m_cand, name in candidates:
            if name == 'recall':
                selected = _eval_candidate(m_cand, name)
                model = m_cand
                break

    # Otherwise, choose the best candidate based on our objective.
    if selected is None:
        evaluated = []
        for m_cand, name in candidates:
            try:
                evaluated.append(_eval_candidate(m_cand, name))
            except Exception as e:
                print(f"Warning: failed to evaluate candidate '{name}': {e}")

        if evaluated:
            selected = sorted(evaluated, key=lambda d: d['key'])[0]
            chosen_name = selected['name']
            # pick the corresponding model instance
            for m_cand, name in candidates:
                if name == chosen_name:
                    model = m_cand
                    break

    # If evaluation failed for some reason, fall back to val_loss weights.
    if selected is None:
        model = model_val_loss
        selected = _eval_candidate(model, 'val_loss')

    checkpoint_choice = selected['name']
    val_probs = selected['probs']
    val_pred_labels = np.argmax(val_probs, axis=1)
    chosen_thr = float(selected['threshold'])
    best_stats = selected['stats']
    cm = selected['cm']
    export_model = make_export_model(model, export_logit_scale=EXPORT_LOGIT_SCALE, export_output=EXPORT_OUTPUT)

    print(
        f"Selected model for export: {checkpoint_choice} | prey_threshold={chosen_thr} | "
        f"fp={best_stats['fp'] if best_stats else 'n/a'} tp={best_stats['tp'] if best_stats else 'n/a'} fn={best_stats['fn'] if best_stats else 'n/a'}"
    )

    # For binary runs, also build thresholded predictions that match ESP usage
    if CLASS_COUNT == 2:
        not_prey_index = CLASSES.index('not_prey')
        prey_probs = np.asarray(val_probs[:, prey_index], dtype=np.float32)

        # Majority baseline (useful when val_accuracy looks "stuck")
        try:
            _val_counts = Counter(val_labels)
            _total = int(len(val_labels))
            _maj = int(max(_val_counts.values())) if _val_counts else 0
            majority_baseline_accuracy = float(_maj / _total) if _total > 0 else None
        except Exception:
            majority_baseline_accuracy = None

        # Helpful diagnostics: if these distributions overlap heavily, any low-FP threshold
        # will necessarily have poor recall.
        try:
            pos = prey_probs[y_true == 1]
            neg = prey_probs[y_true == 0]

            def _q(a, q):
                return float(np.quantile(a, q)) if a.size else None

            prey_prob_summary = {
                "pos": {
                    "n": int(pos.size),
                    "min": float(np.min(pos)) if pos.size else None,
                    "p50": _q(pos, 0.50),
                    "p90": _q(pos, 0.90),
                    "max": float(np.max(pos)) if pos.size else None,
                },
                "neg": {
                    "n": int(neg.size),
                    "min": float(np.min(neg)) if neg.size else None,
                    "p50": _q(neg, 0.50),
                    "p90": _q(neg, 0.90),
                    "max": float(np.max(neg)) if neg.size else None,
                },
            }

            print(
                "Val prey_prob summary | "
                f"pos(n={prey_prob_summary['pos']['n']} p50={prey_prob_summary['pos']['p50']} p90={prey_prob_summary['pos']['p90']} max={prey_prob_summary['pos']['max']}) "
                f"neg(n={prey_prob_summary['neg']['n']} p50={prey_prob_summary['neg']['p50']} p90={prey_prob_summary['neg']['p90']} max={prey_prob_summary['neg']['max']})"
            )

            # Scalar diagnostics
            try:
                roc_auc = float(roc_auc_score(y_true, prey_probs)) if (pos.size and neg.size) else None
            except Exception:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_true, prey_probs)) if (pos.size and neg.size) else None
            except Exception:
                pr_auc = None

            # Plot diagnostics (saved under reports/images)
            try:
                # Probability histogram
                plt.figure(figsize=(6, 4))
                bins = 30
                plt.hist(neg, bins=bins, alpha=0.65, label='not_prey', density=True)
                plt.hist(pos, bins=bins, alpha=0.65, label='prey', density=True)
                plt.axvline(chosen_thr, color='k', linestyle='--', linewidth=1, label=f'thr={chosen_thr:.3f}')
                plt.title('Validation prey probability distribution')
                plt.xlabel('P(prey)')
                plt.ylabel('Density')
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(images_dir, 'val_prey_prob_hist.png'))
                plt.close()

                # ROC + PR curves
                if pos.size and neg.size:
                    fpr, tpr, _ = roc_curve(y_true, prey_probs)
                    pr_precision, pr_recall, _ = precision_recall_curve(y_true, prey_probs)

                    plt.figure(figsize=(10, 4))

                    plt.subplot(1, 2, 1)
                    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}' if roc_auc is not None else 'ROC')
                    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC curve')
                    plt.legend(loc='lower right')

                    plt.subplot(1, 2, 2)
                    plt.plot(pr_recall, pr_precision, label=f'PR AUC={pr_auc:.3f}' if pr_auc is not None else 'PR')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall curve')
                    plt.legend(loc='lower left')

                    plt.tight_layout()
                    plt.savefig(os.path.join(images_dir, 'val_prey_prob_curves.png'))
                    plt.close()
            except Exception as e:
                print(f"Warning: failed to write probability diagnostic plots: {e}")
        except Exception:
            prey_prob_summary = None
            roc_auc = None
            pr_auc = None
            majority_baseline_accuracy = None

        val_pred_used = np.where(prey_probs >= chosen_thr, prey_index, not_prey_index).astype(int)
        label_indices = [0, 1]
    else:
        prey_prob_summary = None
        roc_auc = None
        pr_auc = None
        majority_baseline_accuracy = None
        val_pred_used = val_pred_labels
        label_indices = list(range(len(CLASSES)))

    # Classification report based on the chosen prediction scheme
    report = classification_report(
        val_labels,
        val_pred_used,
        labels=label_indices,
        target_names=CLASSES,
        zero_division=0
    )
    report_dict = classification_report(
        val_labels,
        val_pred_used,
        labels=label_indices,
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    # Prepare training/eval parameters for the report
    def _prob_to_logit_margin(p: float) -> float:
        try:
            p = float(p)
        except Exception:
            p = 0.5
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    report_params = {
        "epochs": EPOCHS,
        "learning_rate": INIT_LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "class_count": CLASS_COUNT,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "train_distribution": train_named,
        "val_distribution": val_named,
        "width_mult": WIDTH_MULT,
        "dropout_rate": DROPOUT_RATE,
        "val_split": VAL_SPLIT,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "augment": AUGMENT,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "label_smoothing": LABEL_SMOOTHING,
        "lr_schedule": LR_SCHEDULE,
        "warmup_epochs": WARMUP_EPOCHS,
        "export_logit_scale": float(EXPORT_LOGIT_SCALE),
        "export_output": str(EXPORT_OUTPUT),
        "checkpoint_choice": checkpoint_choice,
        "prey_threshold": chosen_thr,
        "prey_logit_margin_threshold": _prob_to_logit_margin(chosen_thr) if CLASS_COUNT == 2 else None,
        "prey_threshold_stats": best_stats,
        "val_prey_prob_summary": prey_prob_summary,
        "val_majority_baseline_accuracy": majority_baseline_accuracy,
        "val_roc_auc": roc_auc,
        "val_pr_auc": pr_auc,
    }

    # Confusion matrix, aligned with the same predictions used in the report
    # (cm may already be computed above for the selected candidate, but keep this
    # as the source of truth for the report output.)
    cm = confusion_matrix(val_labels, val_pred_used, labels=label_indices)
    # (Confusion matrix values are stored directly in metrics.json)

    # Capture a structured model summary into metrics.json (no separate txt file)
    from io import StringIO
    _buf = StringIO()
    with redirect_stdout(_buf):
        model.summary()
    summary_lines = _buf.getvalue().splitlines()

    # Very lightweight parser: keep raw text plus a simple per-layer list
    layers_summary = []
    for layer in model.layers:
        cfg = layer.get_config() if hasattr(layer, 'get_config') else {}
        layers_summary.append({
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "output_shape": str(getattr(layer, 'output_shape', 'unknown')),
            "params": int(getattr(layer, 'count_params', lambda: 0)() or 0),
            "config": cfg,
        })

    structured_summary = {
        "text": "\n".join(summary_lines),
        "layers": layers_summary,
        "total_params": int(model.count_params()),
    }

    # -----------------------------
    # Aggregate metrics into JSON for Flask
    # -----------------------------
    def _to_native(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    metrics = {
        "classes": CLASSES,
        "label_indices": label_indices,
        "report": report_dict,
        "confusion_matrix": cm.tolist(),
        "training_params": report_params,
        "model_summary": structured_summary,
        "curves": {
            "accuracy": list(map(float, acc)),
            "val_accuracy": list(map(float, val_acc)),
            "loss": list(map(float, loss_hist)),
            "val_loss": list(map(float, val_loss_hist))
        }
    }

    with open(metrics_json_path, 'w') as f:
        json.dump(_to_native(metrics), f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_path}")

    # -----------------------------
    # Export: SavedModel -> INT8 TFLite -> .cc (into version folder)
    # -----------------------------
    model_save_path = os.path.join(model_dir, 'my_model')
    print(f"Exporting the model to {model_save_path}...", flush=True)
    _export_start = datetime.datetime.now()
    # Keras 3 export prints a very long "Captures:" dump; silence it to avoid
    # giving the impression that the process is stuck.
    from io import StringIO
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        export_model.export(model_save_path)
    _export_end = datetime.datetime.now()
    print(f"Export complete in {( _export_end - _export_start ).total_seconds():.1f}s", flush=True)

    if EXPORT_OUTPUT == 'logits_margin':
        print("Converting and quantizing the model (full INT8, uint8 input / int8 output)...", flush=True)
    else:
        print("Converting and quantizing the model (full INT8 with uint8 I/O)...", flush=True)
    _convert_start = datetime.datetime.now()
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = (tf.int8 if EXPORT_OUTPUT == 'logits_margin' else tf.uint8)
    tflite_quant_model = converter.convert()
    _convert_end = datetime.datetime.now()
    print(f"TFLite conversion complete in {( _convert_end - _convert_start ).total_seconds():.1f}s", flush=True)

    quant_model_path = os.path.join(model_dir, f'{MODEL_NAME}.tflite')
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized model saved to {quant_model_path}")

    # Extract TFLite model details
    def get_tflite_details(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get ops used
        ops = set()
        for op in interpreter._get_ops_details():
            ops.add(op['op_name'])
        
        # Get input/output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        # Estimate arena size (sum of all tensor sizes)
        arena_estimate = 0
        for t in interpreter.get_tensor_details():
            size = 1
            for dim in t['shape']:
                size *= dim
            arena_estimate += size
        
        # Get file size
        file_size = os.path.getsize(model_path)
        
        return {
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2),
            "ops": sorted(list(ops)),
            "input_shape": input_details['shape'].tolist(),
            "input_dtype": str(input_details['dtype']),
            "output_shape": output_details['shape'].tolist(),
            "output_dtype": str(output_details['dtype']),
            "arena_estimate_bytes": arena_estimate,
            "arena_estimate_kb": round(arena_estimate / 1024, 2),
            "input_quantization": input_details.get('quantization', None),
            "output_quantization": output_details.get('quantization', None),
        }
    
    tflite_details = get_tflite_details(quant_model_path)
    print(f"TFLite model size: {tflite_details['file_size_kb']} KB")
    print(f"TFLite ops: {', '.join(tflite_details['ops'])}")
    print(f"Estimated arena: {tflite_details['arena_estimate_kb']} KB")
    
    # Update metrics with TFLite details
    metrics["tflite_details"] = tflite_details

    # If exporting logits margin, compute an int-domain threshold to use on-device.
    try:
        if EXPORT_OUTPUT == 'logits_margin' and CLASS_COUNT == 2:
            out_q = tflite_details.get('output_quantization')
            if isinstance(out_q, (list, tuple)) and len(out_q) == 2:
                out_scale = float(out_q[0])
                out_zp = int(out_q[1])
                margin_thr = float(report_params.get('prey_logit_margin_threshold'))
                if out_scale > 0:
                    q_thr = int(round(margin_thr / out_scale) + out_zp)
                    metrics['tflite_output_threshold'] = {
                        'mode': 'logits_margin',
                        'margin_threshold': margin_thr,
                        'quantized_threshold': q_thr,
                        'scale': out_scale,
                        'zero_point': out_zp,
                    }
    except Exception as e:
        print(f"Warning: failed to compute quantized output threshold: {e}")

    with open(metrics_json_path, 'w') as f:
        json.dump(_to_native(metrics), f, indent=2)
    print(f"Updated metrics JSON with TFLite details")

    # Write a .cc file for ESP32
    def convert_tflite_to_cc(tflite_model_path, cc_output_path):
        with open(tflite_model_path, 'rb') as f:
            model_bytes = f.read()
        with open(cc_output_path, 'w') as f:
            f.write('const unsigned char my_model_quant_tflite[] = {\n')
            hex_array = [f'0x{b:02x}' for b in model_bytes]
            for i in range(0, len(hex_array), 12):
                f.write('  ' + ', '.join(hex_array[i:i+12]) + ',\n')
            f.write('};\n')
            f.write(f'const unsigned int my_model_quant_tflite_len = {len(model_bytes)};\n')

    cc_output_path = os.path.join(model_dir, f'{MODEL_NAME}.cc')
    convert_tflite_to_cc(quant_model_path, cc_output_path)
    print(f"C model source file saved to {cc_output_path}")
