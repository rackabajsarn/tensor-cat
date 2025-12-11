
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout
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
    # square center crop
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    shorter_side = tf.minimum(h, w)
    image = tf.image.resize_with_crop_or_pad(image, shorter_side, shorter_side)
    image = tf.image.resize_with_crop_or_pad(image, 384, 384)
    image = tf.image.resize(image, IMG_SIZE, method='nearest')
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
    # Group by label
    class_to_images = defaultdict(list)
    for path, label in zip(image_paths, labels_encoded):
        class_to_images[label].append(path)

    sampled_paths = []
    for images in class_to_images.values():
        sampled_paths.extend(random.sample(images, min(len(images), 20)))

    for image_path in sampled_paths[:100]:
        img = Image.open(image_path).convert("L").resize(IMG_SIZE)
        arr = np.array(img).astype(np.uint8)  # (96,96)
        arr = np.expand_dims(arr, axis=-1)    # (96,96,1)
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
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

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
            base = lambda step: tf.constant(INIT_LR, dtype=tf.float32)

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

    loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
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

    # Load preferred checkpoint from this run's versioned model directory
    best_acc_path = os.path.join(model_dir, 'best_acc_model.keras')
    best_recall_path = os.path.join(model_dir, 'best_recall_model.keras')

    checkpoint_choice = 'recall' if PREFER_RECALL else 'accuracy'
    checkpoint_path = best_recall_path if PREFER_RECALL else best_acc_path

    if not os.path.exists(checkpoint_path):
        # Fallback to whichever checkpoint exists
        fallback_path = best_acc_path if checkpoint_choice == 'recall' else best_recall_path
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
            checkpoint_choice = 'recall' if fallback_path == best_recall_path else 'accuracy'

    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        print(f"Loaded best {checkpoint_choice} model from {checkpoint_path}")
    else:
        print("No best-checkpoint files found; continuing with current model weights.")

    # Collect curves for metrics.json (no separate plot files)
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])

    # ---------------------------------
    # Evaluation & threshold selection
    # ---------------------------------
    # Build full val tensors for thresholding
    def load_val_array(paths):
        X = []
        for p in paths:
            img = Image.open(p).convert('L')
            shorter = min(img.size)
            left = (img.width - shorter)//2
            top = (img.height - shorter)//2
            img = img.crop((left, top, left+shorter, top+shorter))
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            X.append(np.array(img, dtype=np.uint8))
        X = np.array(X)
        X = np.expand_dims(X, -1)  # (N,96,96,1) uint8
        return X
    val_images = load_val_array(val_paths)
    # If the model expects float32, scale to [0,1]. If it expects uint8, keep as-is.
    if model.inputs[0].dtype == tf.float32:
        val_images = val_images.astype('float32') / 255.0
    # Predict probabilities
    val_probs = model.predict(val_images, batch_size=BATCH_SIZE, verbose=0)
    val_pred_labels = np.argmax(val_probs, axis=1)

    # threshold for prey (binary prey vs everything else)
    y_true_prey = (np.array(val_labels) == prey_index).astype(int)
    prey_probs = val_probs[:, prey_index]

    prec, rec, thr = precision_recall_curve(y_true_prey, prey_probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1)) if len(f1) > 0 else 0
    chosen_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    print("Chosen prey threshold (max F1):", chosen_thr)

    # For binary runs, also build thresholded predictions that match ESP usage
    if CLASS_COUNT == 2:
        # prey=1, not_prey=0 using the chosen threshold
        val_pred_used = (prey_probs >= chosen_thr).astype(int)
        label_indices = [0, 1]
    else:
        # For 3-class, fall back to standard argmax multiclass predictions
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
    report_params = {
        "epochs": EPOCHS,
        "learning_rate": INIT_LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "class_count": CLASS_COUNT,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
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
        "checkpoint_choice": checkpoint_choice,
        "prey_threshold": chosen_thr,
    }

    # Confusion matrix, aligned with the same predictions used in the report
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
    print(f"Exporting the model to {model_save_path}...")
    model.export(model_save_path)

    print("Converting and quantizing the model (full INT8 with uint8 I/O)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

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
            "quantization": input_details.get('quantization', None),
        }
    
    tflite_details = get_tflite_details(quant_model_path)
    print(f"TFLite model size: {tflite_details['file_size_kb']} KB")
    print(f"TFLite ops: {', '.join(tflite_details['ops'])}")
    print(f"Estimated arena: {tflite_details['arena_estimate_kb']} KB")
    
    # Update metrics with TFLite details
    metrics["tflite_details"] = tflite_details
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
