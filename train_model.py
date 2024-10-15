import os
# Disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import piexif
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

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
            label = 'not_cat'
        labels_encoded.append(CLASSES.index(label))
    return labels_encoded

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
    image = data_augmentation(image)
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
        image_paths, labels_encoded, test_size=0.2, random_state=42)

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Define the model
    print("Defining the model...")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Fine-tune the model (optional)
    fine_tune = True
    if fine_tune:
        # Unfreeze the top layers of the base model
        base_model.trainable = True

        # Freeze all layers except the top N layers
        fine_tune_at = 100  # Adjust this value based on your model
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        fine_tune_epochs = 5
        total_epochs = EPOCHS + fine_tune_epochs

        print("Fine-tuning the model...")
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1]
        )


    # Save the model
    model_save_path = os.path.join(MODEL_DIR, 'my_model')
    print(f"Saving the model to {model_save_path}...")
    model.export(model_save_path)

    # Convert and quantize the model
    print("Converting and quantizing the model...")
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

    # Compile the model for the Edge TPU
    print("Compiling the model for the Edge TPU...")
    edgetpu_compiled_model = os.path.join(MODEL_DIR, f'{MODEL_NAME}_edgetpu.tflite')
    compile_command = f"edgetpu_compiler -o {MODEL_DIR} {quant_model_path}"
    os.system(compile_command)
    print(f"Edge TPU model saved to {edgetpu_compiled_model}")
