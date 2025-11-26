
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from contextlib import redirect_stdout
import argparse
from collections import Counter
from collections import defaultdict
import random

# -----------------------------
# Args & constants
# -----------------------------
parser = argparse.ArgumentParser(description='Train a small grayscale 96x96 model (binary prey vs not_prey) and export TFLite + .cc')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for initial training.')
parser.add_argument('--learning_rate', type=str, default='1e-3', help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
args = parser.parse_args()

EPOCHS = args.epochs
INIT_LR = float(args.learning_rate)
BATCH_SIZE = args.batch_size
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths
# -----------------------------
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_DIR = 'simple_model'
MODEL_NAME = 'my_simple_model_quant'
STATIC_DIR = 'static'
REPORTS_DIR = os.path.join(STATIC_DIR, 'reports')
IMAGES_DIR = os.path.join(REPORTS_DIR, 'images')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Report/plot files
report_filename = os.path.join(REPORTS_DIR, 'classification_report.html')
accuracy_plot_filename = os.path.join(IMAGES_DIR, 'accuracy_plot.png')
loss_plot_filename = os.path.join(IMAGES_DIR, 'loss_plot.png')
confusion_matrix_filename = os.path.join(IMAGES_DIR, 'confusion_matrix.png')
class_weights_filename = os.path.join(REPORTS_DIR, 'class_weights.json')
model_summary_filename = os.path.join(REPORTS_DIR, 'model_summary.txt')
threshold_filename = os.path.join(REPORTS_DIR, 'prey_threshold.txt')

# -----------------------------
# Task setup
# -----------------------------
CLASSES = ['not_prey', 'prey']  # binary
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
        # robust access
        prey = bool(labels.get('prey', False))
        label = 'prey' if prey else 'not_prey'
        labels_encoded.append(CLASSES.index(label))
    return labels_encoded

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
    image = tf.image.resize(image, IMG_SIZE, method='bilinear', antialias=True)
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
    # Photometric-only augmentations; tiny translations
    image = tf.image.random_brightness(image, 0.07)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = adjust_gamma(image)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    image = tf.pad(image, [[2,2],[2,2],[0,0]], mode='REFLECT')
    image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 1])
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
def ds_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2)
    ])

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.uint8)
    x = layers.Rescaling(1./255.0)(inputs)  # move normalization inside graph
    x = ds_block(16)(x)
    x = ds_block(32)(x)
    x = ds_block(64)(x)
    x = layers.Conv2D(96, 1, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# -----------------------------
# Training
# -----------------------------
if __name__ == '__main__':
    # Load dataset
    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    labels_encoded = convert_labels(labels_list)

    # Class distribution
    class_counts = Counter(labels_encoded)
    print("Class distribution (encoded):", class_counts)
    class_distribution = {CLASSES[label]: count for label, count in class_counts.items()}
    print("Class distribution (named):", class_distribution)

    # Split (stratified)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels_encoded, test_size=0.2, random_state=SEED, stratify=labels_encoded)

    # Class weights
    class_weights_arr = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    # Emphasize 'prey' a bit more
    class_weight_dict[CLASSES.index('prey')] *= 4.0

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

    # Model & optimizer
    model = build_model()

    # AdamW + cosine decay
    steps_per_epoch = max(1, len(train_paths)//BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INIT_LR, decay_steps=total_steps, alpha=1e-2
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)

    # Metrics focused on 'prey' class
    prey_index = CLASSES.index('prey')
    precision_prey = tf.keras.metrics.Precision(class_id=prey_index, name='precision_prey')
    recall_prey = tf.keras.metrics.Recall(class_id=prey_index, name='recall_prey')

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                 precision_prey, recall_prey]
    )

    # Callbacks
    checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_acc_model.keras'),
        monitor='val_accuracy', mode='max', save_best_only=True
    )
    checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_recall_model.keras'),
        monitor='val_recall_prey', mode='max', save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_recall_prey', patience=5, mode='max', restore_best_weights=True
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

    # Load best recall model
    best_recall_path = os.path.join(MODEL_DIR, 'best_recall_model.keras')
    if os.path.exists(best_recall_path):
        model = tf.keras.models.load_model(best_recall_path)

    # ---------------------------------
    # Plots
    # ---------------------------------
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    if len(acc) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.savefig(accuracy_plot_filename)
        plt.close()
        print(f"Accuracy plot saved to {accuracy_plot_filename}")

    if len(loss_hist) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, loss_hist, label='Training Loss')
        plt.plot(epochs_range, val_loss_hist, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(loss_plot_filename)
        plt.close()
        print(f"Loss plot saved to {loss_plot_filename}")

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

    val_images_u8 = load_val_array(val_paths)
    # Predict probabilities
    val_probs = model.predict(val_images_u8, batch_size=BATCH_SIZE, verbose=0)
    val_pred_labels = np.argmax(val_probs, axis=1)

    # threshold for prey
    y_true_prey = (np.array(val_labels) == prey_index).astype(int)
    prey_probs = val_probs[:, prey_index]
    prec, rec, thr = precision_recall_curve(y_true_prey, prey_probs)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    best_idx = np.argmax(f1[:-1]) if len(f1) > 1 else 0
    chosen_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    with open(threshold_filename, 'w') as f:
        f.write(str(chosen_thr))
    print("Chosen prey threshold:", chosen_thr)

    # Classification report (default argmax)
    report = classification_report(
        val_labels, val_pred_labels, target_names=CLASSES, zero_division=0
    )
    report_dict = classification_report(
        val_labels, val_pred_labels, target_names=CLASSES, zero_division=0, output_dict=True
    )

    # Save HTML report
    report_template = """
    <html>
    <head>
        <title>Classification Report</title>
        <link rel="stylesheet" type="text/css" href="/static/css/style.css">
    </head>
    <body class="dark-theme">
        <table>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
            {% for label, metrics in report.items() if label in classes %}
            <tr>
                <td>{{ label }}</td>
                <td>{{ '{0:.2f}'.format(metrics['precision']) }}</td>
                <td>{{ '{0:.2f}'.format(metrics['recall']) }}</td>
                <td>{{ '{0:.2f}'.format(metrics['f1-score']) }}</td>
                <td>{{ metrics['support'] }}</td>
            </tr>
            {% endfor %}
            <tr>
                <td colspan="4"><strong>Accuracy</strong></td>
                <td><strong>{{ '{0:.2f}'.format(report['accuracy']) }}</strong></td>
            </tr>
        </table>
        <p>Chosen prey threshold: {{ threshold }}</p>
    </body>
    </html>
    """
    template = Template(report_template)
    report_html = template.render(report=report_dict, classes=CLASSES, threshold=chosen_thr)
    with open(report_filename, 'w') as f:
        f.write(report_html)
    print(f"Classification report saved to {report_filename}")

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_pred_labels, labels=[0,1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 14})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(confusion_matrix_filename)
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_matrix_filename}")

    # Save class weights
    with open(class_weights_filename, 'w') as f:
        json.dump(class_weight_dict, f)
    print(f"Class weights saved to {class_weights_filename}")

    # Save summary
    with open(model_summary_filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print(f"Model summary saved to {model_summary_filename}")

    # -----------------------------
    # Export: SavedModel -> INT8 TFLite -> .cc
    # -----------------------------
    model_save_path = os.path.join(MODEL_DIR, 'my_model')
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

    quant_model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.tflite')
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized model saved to {quant_model_path}")

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

    cc_output_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.cc')
    convert_tflite_to_cc(quant_model_path, cc_output_path)
    print(f"C model source file saved to {cc_output_path}")
