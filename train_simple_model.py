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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from contextlib import redirect_stdout
import argparse
from collections import Counter
from collections import defaultdict
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Recommended starting parameters for a small custom CNN (not pretrained):
# - epochs: 30~50 (more epochs are usually needed for training from scratch)
# - fine_tune_epochs: 0 (no fine-tuning for a custom CNN)
# - learning_rate: 1e-3 (higher than for transfer learning)
# - fine_tune_at: (not used, but keep default for compatibility)

parser = argparse.ArgumentParser(description='Train the model with specified parameters.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for initial training.')
parser.add_argument('--fine_tune_epochs', type=int, default=0, help='Number of epochs for fine-tuning.')
parser.add_argument('--learning_rate', type=str, default='1e-3', help='Learning rate for training.')
parser.add_argument('--fine_tune_at', type=int, default=120, help='Layer number to start fine-tuning from.')
parser.add_argument('--seed', type=int, default=0, help='Seed number')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

args = parser.parse_args()


EPOCHS = args.epochs
FINE_TUNE_EPOCHS = args.fine_tune_epochs
LEARNING_RATE = float(args.learning_rate)
FINE_TUNE_AT = args.fine_tune_at
SEED = args.seed
BATCH_SIZE = args.batch_size  # Try reducing to 32 or even 16 if you see overfitting or want more updates per epoch

# If you want to always know the seed, you can:
# - Generate a random seed yourself when SEED == 0, print/store it, and use it for all libraries.
# Example:
if SEED == 0:
    SEED = random.randint(1, 2**32 - 1)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

# Directories
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_DIR = 'simple_model'
MODEL_NAME = 'my_simple_model_quant'
STATIC_DIR = 'static'
REPORTS_DIR = os.path.join(STATIC_DIR, 'reports/simple')
IMAGES_DIR = os.path.join(REPORTS_DIR, 'images')


# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Update file paths accordingly
report_filename = os.path.join(REPORTS_DIR, 'classification_report.html')
accuracy_plot_filename = os.path.join(IMAGES_DIR, 'accuracy_plot.png')
loss_plot_filename = os.path.join(IMAGES_DIR, 'loss_plot.png')
confusion_matrix_filename = os.path.join(IMAGES_DIR, 'confusion_matrix.png')
class_weights_filename = os.path.join(REPORTS_DIR, 'class_weights.json')
model_summary_filename = os.path.join(REPORTS_DIR, 'model_summary.txt')

# Classes
CLASSES = ['not_cat', 'not_prey', 'prey']
IMG_SIZE = (96, 96)


# To maximize recall for 'prey', increase the class weight for 'prey'
# This will penalize false negatives more during training

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, offset=0):
        super().__init__()
        self.total_epochs = total_epochs
        self.offset = offset  # Number of epochs completed before this phase

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        progress = int((current_epoch / self.total_epochs) * 100)
        print(f'\nPROGRESS:{progress}', flush=True)


def get_image_labels(image_path):
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        description = exif_dict['0th'].get(piexif.ImageIFD.ImageDescription, b'{}').decode('utf-8')
        labels = json.loads(description)
    except Exception as e:
        print(f"Error reading labels from {image_path}: {e}")
        labels = {
            "prey": False
        }
    return labels

def load_dataset(dataset_dir):
    image_paths = []
    labels_list = []

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            image_path = os.path.join(dataset_dir, filename)
            labels = get_image_labels(image_path)
            image_paths.append(image_path)
            labels_list.append(labels)

    return image_paths, labels_list

def convert_labels(labels_list):
    labels_encoded = []
    for labels in labels_list:
        label = 'not_cat'
        if labels['cat']:
            if labels['morris']:
                if labels['entering']:
                    if labels['prey']:
                        label = 'prey'
                    else:
                        label = 'not_prey'
                else:
                    label = 'not_cat'
            else:
                if labels['entering']:
                    label = 'not_cat'
                else:
                    label = 'not_cat'  # Adjust if you have data for unknown cat leaving
        else:
            label = 'not_cat'
        labels_encoded.append(CLASSES.index(label))
    return labels_encoded

# Data augmentation for training dataset
data_augmentation = tf.keras.Sequential([
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.1),
])

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Read as grayscale
    shorter_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, shorter_side, shorter_side)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

def preprocess_image_train(image_path, label):
    image, label = preprocess_image(image_path, label)
    #image = data_augmentation(image)
    return image, label

def preprocess_image_val(image_path, label):
    image, label = preprocess_image(image_path, label)
    return image, label

def representative_data_gen():
    # Group image paths by their class label
    class_to_images = defaultdict(list)
    for path, label in zip(image_paths, labels_encoded):
        class_to_images[label].append(path)

    # Sample a few images from each class
    sampled_paths = []
    for images in class_to_images.values():
        sampled_paths.extend(random.sample(images, min(len(images), 20)))  # Adjust per-class sample size as needed

    # Generate representative samples
    for image_path in sampled_paths[:100]:  # Limit to 100 total
        image = Image.open(image_path).convert("L").resize(IMG_SIZE)  # Ensure grayscale
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # Shape: (96, 96, 1)
        image = np.expand_dims(image, axis=0)   # Shape: (1, 96, 96, 1)
        yield [image]

# Notes on result variability and local minima:
# - Even with a fixed random seed, neural network training can be non-deterministic due to:
#   - Multi-threading, GPU parallelism, and non-deterministic operations in TensorFlow.
#   - Data pipeline shuffling and parallelism.
# - The model can get stuck in different local minima or saddle points, especially with small datasets or imbalanced classes.
# - This can lead to noticeably different results between runs, even with the same code and seed.

# To reduce variability:
# - Keep the random seed fixed (as you do).
# - Use a smaller learning rate for more stable convergence.
# - Train for more epochs with early stopping.
# - Try running the training multiple times and average the results (ensemble or cross-validation).
# - Consider using a larger or more diverse dataset if possible.

# For critical applications, it's common to train several models and select the best or average their predictions.

# To ensure you get the best possible model, consider these steps:

# 1. Use K-fold cross-validation to evaluate model robustness.
# 2. Train multiple models with different seeds and average their results (ensemble).
# 3. Perform a grid search over hyperparameters (class weights, learning rate, threshold, etc.).
# 4. Monitor both validation loss and accuracy, and save the best model based on your most important metric (e.g., recall for 'prey').
# 5. Optionally, use a validation split from the training set for early stopping, and keep a separate test set for final evaluation.

# Example: K-fold cross-validation (simplified, pseudocode)
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# for train_idx, val_idx in skf.split(image_paths, labels_encoded):
#     # Split data, train model, evaluate, and collect metrics

# Example: Save the best model based on recall for 'prey'
best_recall_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_recall_model.keras'),
    monitor='val_recall_prey',
    mode='max',
    save_best_only=True,
    save_weights_only=False
)

if __name__ == '__main__':
    # Load dataset
    # print("Loading dataset...")
    # print("Epochs:", EPOCHS)
    # print("Fine Tune Epochs:", FINE_TUNE_EPOCHS)
    # print("Learning Rate:", LEARNING_RATE)
    # print("Fine tune at layer:", FINE_TUNE_AT)
    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    labels_encoded = convert_labels(labels_list)


    # Count the occurrences of each class in the encoded labels
    class_counts = Counter(labels_encoded)
    print("Class distribution (encoded):", class_counts)

    # Optionally, map the counts to class names
    class_distribution = {CLASSES[label]: count for label, count in class_counts.items()}
    print("Class distribution (named):", class_distribution)

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    # Compute class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Increase the weight for 'prey' to prioritize recall (reduce missed detections)
    class_weight_dict[CLASSES.index('prey')] *= 4.0  # You can try 2.0, 3.0, or higher recall
    #class_weight_dict[CLASSES.index('not_prey')] *= 3.0

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(preprocess_image_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Define the model
    print("Defining the model...")

    # Small custom CNN for grayscale input and microcontroller deployment
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    precision_prey = tf.keras.metrics.Precision(class_id=CLASSES.index('prey'), name='precision_prey')
    recall_prey = tf.keras.metrics.Recall(class_id=CLASSES.index('prey'), name='recall_prey')

    # Compile the model with appropriate metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            precision_prey, 
            recall_prey
        ]
    )

    # Define a custom callback to save the best model based on validation accuracy
    checkpoint_filepath = os.path.join(MODEL_DIR, 'best_model.keras')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  # Save the full model
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,           # Allow more epochs before stopping
        restore_best_weights=True
    )

    # Calculate total epochs
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS

    # Initial training progress callback
    progress_callback_initial = ProgressCallback(total_epochs=total_epochs, offset=0)

    # Initial training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[model_checkpoint_callback, best_recall_checkpoint, progress_callback_initial, early_stopping],
        verbose=2
    )

    # Load the best model from training
    print("Loading the best model...")
    model = tf.keras.models.load_model(checkpoint_filepath)

    # Combine history from initial training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig(accuracy_plot_filename)
    plt.close()
    print(f"Accuracy plot saved to {accuracy_plot_filename}")

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(loss_plot_filename)
    plt.close()
    print(f"Loss plot saved to {loss_plot_filename}")


    # Evaluate the model on the validation set
    print("Evaluating the model...")
    val_images = []
    val_labels_list = []
    for image_path, label in zip(val_paths, val_labels):
        image = Image.open(image_path).convert("L").resize(IMG_SIZE)  # Ensure grayscale
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension: (96, 96, 1)
        val_images.append(image)
        val_labels_list.append(label)
    val_images = np.array(val_images)
    val_labels_list = np.array(val_labels_list)

    val_predictions = model.predict(val_images)
    # Lower threshold for 'prey' (e.g., 0.2 instead of default 0.5)
    prey_probs = val_predictions[:, CLASSES.index('prey')]
    threshold = 0.5  # Lower threshold increases recall

    # For 3 classes, you should assign 'prey' if above threshold, otherwise pick the highest of the other two
    not_cat_idx = CLASSES.index('not_cat')
    not_prey_idx = CLASSES.index('not_prey')
    prey_idx = CLASSES.index('prey')

    val_pred_labels = []
    for probs in val_predictions:
        if probs[prey_idx] > threshold:
            val_pred_labels.append(prey_idx)
        else:
            # Choose the higher probability between 'not_cat' and 'not_prey'
            if probs[not_cat_idx] > probs[not_prey_idx]:
                val_pred_labels.append(not_cat_idx)
            else:
                val_pred_labels.append(not_prey_idx)
    val_pred_labels = np.array(val_pred_labels)

    # Generate classification report
    report = classification_report(
        val_labels_list,
        val_pred_labels,
        target_names=CLASSES,
        zero_division=0
    )

    # Generate classification report as a dictionary
    report_dict = classification_report(
        val_labels_list,
        val_pred_labels,
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    # Calculate F1 score for the 'prey' class
    f1_prey = f1_score(val_labels_list, val_pred_labels, labels=[CLASSES.index('prey')], average='weighted')
    print(f"F1 Score for prey: {f1_prey}")

    # Output metrics in JSON format for subprocess
    output_metrics = {
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "f1_score": f1_prey
    }
    print(json.dumps(output_metrics))

    # Save the classification report as an HTML file
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
            {% for label, metrics in report.items() if label != 'accuracy' and label != 'macro avg' and label != 'weighted avg' %}
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
            <tr>
                <td colspan="4"><strong>Macro Avg</strong></td>
                <td></td>
            </tr>
            <tr>
                <td>Precision</td>
                <td colspan="4">{{ '{0:.2f}'.format(report['macro avg']['precision']) }}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td colspan="4">{{ '{0:.2f}'.format(report['macro avg']['recall']) }}</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td colspan="4">{{ '{0:.2f}'.format(report['macro avg']['f1-score']) }}</td>
            </tr>
        </table>
    </body>
    </html>
    """

    template = Template(report_template)
    report_html = template.render(report=report_dict)

    # Save the report as an HTML file
    with open(report_filename, 'w') as f:
        f.write(report_html)
    print(f"Classification report saved to {report_filename}")

    # Generate and save the confusion matrix plot
    cm = confusion_matrix(val_labels_list, val_pred_labels)
    # Optionally, abbreviate class labels for better fit
    abbreviated_classes = ['Not Cat', 'Not Prey', 'Prey']

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=abbreviated_classes, 
        yticklabels=abbreviated_classes,
        annot_kws={"size": 14}  # Increase the font size of annotations
        )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.savefig(confusion_matrix_filename)
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_matrix_filename}")


    # Save class weights to a JSON file
    with open(class_weights_filename, 'w') as f:
        json.dump(class_weight_dict, f)
    print(f"Class weights saved to {class_weights_filename}")

    with open(model_summary_filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print(f"Model summary saved to {model_summary_filename}")

    # Export the model
    model_save_path = os.path.join(MODEL_DIR, 'my_model')
    print(f"Exporting the model to {model_save_path}...")
    model.export(model_save_path)

    # Convert and quantize the model
    print("Converting and quantizing the model...")
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    quant_model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.tflite')
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized model saved to {quant_model_path}")

    # Export as C array for firmware embedding (model.cc/model.h)
    c_array_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.cc')
    h_array_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.h')
    array_name = "my_model_quant_tflite"

    def tflite_to_c_array(byte_data, var_name):
        hex_array = ','.join(str(b) for b in byte_data)
        c_str = f'unsigned char {var_name}[] = {{{hex_array}}};\n'
        c_str += f'unsigned int {var_name}_len = {len(byte_data)};\n'
        return c_str

    # Write .cc file
    with open(c_array_path, 'w') as f:
        f.write(tflite_to_c_array(tflite_quant_model, array_name))
    print(f"C array model saved to {c_array_path}")

    # Write .h file
    with open(h_array_path, 'w') as f:
        f.write(f'#ifndef MODEL_H\n#define MODEL_H\n\n')
        f.write(f'extern unsigned char {array_name}[];\n')
        f.write(f'extern unsigned int {array_name}_len;\n')
        f.write(f'#endif // MODEL_H\n')
    print(f"Header file saved to {h_array_path}")

    print("Model ready for ESP32. Use the .tflite file for SD card/FS, or .cc/.h for firmware embedding.")
    print(f"Randomly generated seed: {SEED}")
    print("Classification Report:")
    print(report)
    print("Weights:")
    print(class_weight_dict)

# Interpretation of your classification report:
# - 'not_cat': high precision (0.94), high recall (0.89)
# - 'not_prey': good precision (0.80), very high recall (0.96)
# - 'prey': high precision (0.94), moderate recall (0.65)
# - Overall accuracy: 0.87

# What this means:
# - Your model is now well balanced: it detects most 'prey' (recall=0.65) with few false positives (precision=0.94).
# - 'not_prey' is also well detected (recall=0.96).
# - The class weights (especially for 'prey') helped the model focus on the minority class.

# If you want even higher recall for 'prey', you can:
# - Slightly increase the class weight for 'prey' (currently 7.09).
# - Slightly lower the threshold for 'prey' in your prediction logic.
# - But note: this may reduce precision and overall accuracy.

# If you are satisfied with this balance, you can keep these settings.
# If you want to tune further, adjust class weights and threshold as needed.

