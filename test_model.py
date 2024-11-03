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
from sklearn.metrics import classification_report, f1_score
import argparse
from collections import Counter


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train the model with specified parameters.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for initial training.')
parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning.')
parser.add_argument('--learning_rate', type=str, default='1e-5', help='Learning rate for training.')
parser.add_argument('--fine_tune_at', type=int, default=120, help='Layer number to start fine-tuning from.')

args = parser.parse_args()

EPOCHS = args.epochs
FINE_TUNE_EPOCHS = args.fine_tune_epochs
LEARNING_RATE = float(args.learning_rate)
FINE_TUNE_AT = args.fine_tune_at


# Directories
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_DIR = 'model'
MODEL_NAME = 'my_model_quant'
STATIC_DIR = 'static'
REPORTS_DIR = os.path.join(STATIC_DIR, 'reports')
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
CLASSES = ['not_cat', 'unknown_cat_entering', 'cat_morris_leaving', 'cat_morris_entering', 'prey']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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
            "cat": False,
            "morris": False,
            "entering": False,
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
        # Default label
        label = 'not_cat'
        if labels['cat']:
            if labels['morris']:
                if labels['entering']:
                    if labels['prey']:
                        label = 'prey'
                    else:
                        label = 'cat_morris_entering'
                else:
                    label = 'cat_morris_leaving'
            else:
                if labels['entering']:
                    label = 'unknown_cat_entering'
                else:
                    label = 'not_cat'  # Adjust if you have data for unknown cat leaving
        else:
            label = 'not_cat'
        labels_encoded.append(CLASSES.index(label))
    return labels_encoded

# Data augmentation for training dataset
data_augmentation = tf.keras.Sequential([
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
    layers.GaussianNoise(0.05),
])

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    shorter_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, shorter_side, shorter_side)    
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

def preprocess_image_train(image_path, label):
    image, label = preprocess_image(image_path, label)

    # Use tf.py_function to get the probability for the given label
    augmentation_probability = tf.py_function(get_augmentation_probability, [label], tf.float32)

    # Apply augmentation based on the calculated probability
    if tf.random.uniform([]) < augmentation_probability:
        image = data_augmentation(image)

    return image, label


def preprocess_image_val(image_path, label):
    image, label = preprocess_image(image_path, label)
    return image, label

# TensorFlow-compatible function to retrieve probability
def get_augmentation_probability(label):
    # Convert label to int
    label = int(label)
    return tf.constant(augmentation_probabilities_dict.get(label, 0), dtype=tf.float32)


if __name__ == '__main__':
    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    labels_encoded = convert_labels(labels_list)

    # Count the occurrences of each class in the encoded labels
    class_counts = Counter(labels_encoded)
    print("Class distribution (encoded):", class_counts)

    # Optionally, map the counts to class names
    class_distribution = {CLASSES[label]: count for label, count in class_counts.items()}
    print("Class distribution (named):", class_distribution)

    max_count = max(class_counts.values())

    # Calculate augmentation probability for each class
    augmentation_probabilities = {
        label: max_count / count for label, count in class_counts.items()
    }
    print("Augmentation probabilities:", augmentation_probabilities)

    augmentation_probabilities_dict = {
        label: max_count / count for label, count in class_counts.items()
    }

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

    # Manually adjust the weight of 'unknown_cat_entering'
    #class_weight_dict[CLASSES.index('unknown_cat_entering')] *= 0.5

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

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model initially

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
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

    # Calculate total epochs
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS

    # Initial training progress callback
    progress_callback_initial = ProgressCallback(total_epochs=total_epochs, offset=0)

    # Initial training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        #class_weight=class_weight_dict,
        callbacks=[model_checkpoint_callback, progress_callback_initial],
        verbose=2
    )

    # Fine-tune the model
    
    progress_callback_fine_tune = ProgressCallback(total_epochs=total_epochs, offset=EPOCHS)
    
    fine_tune = True
    if fine_tune:
        # Unfreeze some layers of the base model
        base_model.trainable = True
        # Unfreeze the top N layers (adjust as needed)
        fine_tune_at = FINE_TUNE_AT  # Adjust this value based on your model
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            ]
        )

        # Define a new checkpoint callback for fine-tuning
        fine_tune_checkpoint_filepath = os.path.join(MODEL_DIR, 'best_model_fine_tuned.keras')
        fine_tune_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=fine_tune_checkpoint_filepath,
            save_weights_only=False,  # Save the full model
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            #class_weight=class_weight_dict,
            callbacks=[fine_tune_checkpoint_callback, early_stopping_callback, progress_callback_fine_tune],
            verbose=2
        )

    # Load the best model from fine-tuning
    model = tf.keras.models.load_model(fine_tune_checkpoint_filepath)

    # Combine history from initial training and fine-tuning
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(len(acc))

    # Evaluate the model on the validation set
    val_images = []
    val_labels_list = []
    for image_path, label in zip(val_paths, val_labels):
        image = Image.open(image_path).resize(IMG_SIZE)
        image = np.array(image).astype(np.float32) / 255.0
        val_images.append(image)
        val_labels_list.append(label)
    val_images = np.array(val_images)
    val_labels_list = np.array(val_labels_list)

    val_predictions = model.predict(val_images)
    val_pred_labels = np.argmax(val_predictions, axis=1)

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

    # Output metrics in JSON format for subprocess
    output_metrics = {
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "f1_score": f1_prey
    }
    print(json.dumps(output_metrics))
