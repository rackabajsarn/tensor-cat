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
from sklearn.metrics import classification_report

# Directories
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_DIR = 'model'
MODEL_NAME = 'my_model_quant'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Classes
CLASSES = ['not_cat', 'unknown_cat_entering', 'cat_morris_leaving', 'cat_morris_entering', 'prey']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

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
                if labels['prey']:
                    label = 'prey'
                elif labels['entering']:
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
    image = data_augmentation(image)
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

    # Compile the model with appropriate metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
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
        fine_tune_at = 100  # Adjust this value based on your model
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
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

        fine_tune_epochs = 5
        total_epochs = EPOCHS + fine_tune_epochs

        print("Fine-tuning the model...")
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            class_weight=class_weight_dict,
            callbacks=[fine_tune_checkpoint_callback]
        )

    # Load the best model from fine-tuning
    print("Loading the best fine-tuned model...")
    model = tf.keras.models.load_model(fine_tune_checkpoint_filepath)

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
    print("Classification Report:")
    print(report)

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
