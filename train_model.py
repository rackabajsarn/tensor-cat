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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from contextlib import redirect_stdout
import argparse

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
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.1),
])

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
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
    for image_path in image_paths[:100]:
        image = Image.open(image_path).resize(IMG_SIZE)
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        yield [image]

if __name__ == '__main__':
    # Load dataset
    print("Loading dataset...")
    image_paths, labels_list = load_dataset(DATASET_IMAGES_DIR)
    labels_encoded = convert_labels(labels_list)

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[model_checkpoint_callback]
    )

    # Fine-tune the model
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

        fine_tune_epochs = FINE_TUNE_EPOCHS
        total_epochs = EPOCHS + fine_tune_epochs

        print("Fine-tuning the model...")
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            class_weight=class_weight_dict,
            callbacks=[fine_tune_checkpoint_callback,early_stopping_callback]
        )

    # Load the best model from fine-tuning
    print("Loading the best fine-tuned model...")
    model = tf.keras.models.load_model(fine_tune_checkpoint_filepath)

    # ... [After the fine-tuning code and before the evaluation] ...

    # Plotting training & validation accuracy and loss

    # Combine history from initial training and fine-tuning
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

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
    abbreviated_classes = ['Not Cat', 'Unknown Enter', 'Morris Leave', 'Morris Enter', 'Prey']

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

    # Compile the model for the Edge TPU
    print("Compiling the model for the Edge TPU...")
    compile_command = f"edgetpu_compiler -o {MODEL_DIR} {quant_model_path}"
    os.system(compile_command)
    edgetpu_compiled_model = os.path.join(MODEL_DIR, f'{MODEL_NAME}_edgetpu.tflite')
    print(f"Edge TPU model saved to {edgetpu_compiled_model}")
    print("Classification Report:")
    print(report)
    print("Weights:")
    print(class_weight_dict)
    
