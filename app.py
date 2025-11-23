import os
import threading
import base64
import datetime
import credentials
import logging
import json
import shutil
import time
import subprocess
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from paho.mqtt import client as mqtt_client
import piexif
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input
from pycoral.adapters.classify import get_classes
from PIL import ImageOps

app = Flask(__name__)
app.secret_key = credentials.SECRET_KEY

# MQTT Configuration
MQTT_BROKER = credentials.MQTT_SERVER
MQTT_PORT = 1883
MQTT_TOPIC = 'catflap/image'

# Directories
STATIC_IMAGES_DIR = 'static/images'
DATASET_IMAGES_DIR = 'dataset/images'
MODEL_DIR = 'model'
MODEL_NAME = 'my_model_quant_edgetpu.tflite'
MODEL_INFO_PATH = 'model_info.json'

# Shared state
retraining_status = {
    'retraining': False,
    'error': None,
    'last_trained': None,
    'images_used': 0,
    'output': "",
    'progress': 0,  # Add progress key
    'completed': False  # Add completed flag
}


# Ensure directories exist
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
os.makedirs(DATASET_IMAGES_DIR, exist_ok=True)

logging.basicConfig(
    filename='/home/tensor-cat/logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Suppress Werkzeug logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

logging.info('Application started.')

# Initialize global interpreter and a lock for thread safety
interpreter = None
model_lock = threading.Lock()

def load_model():
    global interpreter
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with model_lock:
        if interpreter:
            del interpreter  # Clean up the existing interpreter
        print("Loading the Edge TPU model...")
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully.")

# Initial model loading at startup
load_model()

# Classes mapping
CLASSES = ['not_cat', 'unknown_cat_entering', 'cat_morris_leaving', 'cat_morris_entering', 'prey']
IMG_SIZE = (224, 224)

# MQTT Client
initial_connection = True

def mqtt_on_connect(client, userdata, flags, rc):
    global initial_connection
    if rc == 0:
        print("Connected to MQTT Broker!")
        initial_connection = True
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect to MQTT Broker, return code {rc}")

def mqtt_on_message(client, userdata, msg):
    global initial_connection
    start = time.time()
    if initial_connection and msg.retain:
        # Ignore the retained message on initial connection
        print("Ignored retained message on initial connection.")
        initial_connection = False
        return
    else:
        # After the first message, process messages normally
        initial_connection = False

    try:
        image_data = msg.payload;
        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp}.jpg"
        image_path = os.path.join(STATIC_IMAGES_DIR, image_filename)

        # Save the image to disk
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved to {image_path}")

        # Classify the image
        predicted_class = classify_image(image_path)
        predicted_label = CLASSES[predicted_class]

        end = time.time()
        
        # Set initial EXIF tags based on prediction
        labels = {
            "cat": False,
            "morris": False,
            "entering": False,
            "prey": False
        }

        if predicted_label == 'not_cat':
            labels['cat'] = False
            logging.warning('Empty picture received?')
        elif predicted_label == 'cat_morris_entering':
            labels['cat'] = True
            labels['morris'] = True
            labels['entering'] = True
        elif predicted_label == 'cat_morris_leaving':
            labels['cat'] = True
            labels['morris'] = True
            labels['entering'] = False
        elif predicted_label == 'prey':
            labels['cat'] = True
            labels['morris'] = True
            labels['prey'] = True
            labels['entering'] = True
            client.publish('catflap/alert', json.dumps({"topic":"ALERT","message":"Mus!","title":"PREY ALERT!"}))
        elif predicted_label == 'unknown_cat_entering':
            labels['cat'] = True
            labels['morris'] = False
            labels['entering'] = True
            # client.publish('catflap/alert', json.dumps({"topic":"INFO","message":"Peekaboo!"}))

        # Publish server inference result (for comparison with ESP32)
        client.publish('catflap/inference', predicted_label)
        
        # Simplify server result for comparison (prey vs not_prey)
        server_simple = "prey" if predicted_label == "prey" else "not_prey"
        client.publish('catflap/server_inference', server_simple)
        
        # Write labels to EXIF
        write_labels(image_path, labels)
        
        # client.publish('catflap/debug', f"Inference ({predicted_label}) done in {int((end - start)*1000)} ms")
        new_images = count_current_classify_images()
        message = f"{new_images} New image to classify" if new_images < 2 else f"{new_images} New images to classify"
        message_json = {"topic":"INFO","message":message,"title":f"{predicted_label} ({int((end - start)*1000)} ms)"}
        message_json = json.dumps(message_json)
        client.publish('catflap/alert', message_json)
        logging.info(f"Image classified as {predicted_label} and labels updated.")

    except Exception as e:
        print(f"Error processing message: {e}")
        logging.error(f"Error processing message: {e}")

def mqtt_listen():
    client = mqtt_client.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# Start MQTT client in a separate thread
mqtt_thread = threading.Thread(target=mqtt_listen)
mqtt_thread.daemon = True
mqtt_thread.start()

# Helper Functions

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}

def get_image_list(directory):
    images = [f for f in os.listdir(directory) if allowed_file(f)]
    images.sort(reverse=True)  # Latest first
    return images

def read_labels(image_path):
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

def write_labels(image_path, labels):
    try:
        img = Image.open(image_path)
        # Attempt to retrieve existing EXIF data
        exif_bytes = img.info.get('exif', None)
        
        if exif_bytes:
            try:
                exif_dict = piexif.load(exif_bytes)
            except piexif.InvalidImageDataError:
                print(f"Invalid EXIF data for {image_path}, initializing new EXIF.")
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        else:
            # No EXIF data present
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        description = json.dumps(labels)
        exif_dict['0th'][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, "jpeg", exif=exif_bytes)
        return True
    except Exception as e:
        print(f"Error writing labels to {image_path}: {e}")
        return False

def classify_image(image_path):
    try:
        # Preprocess the image
        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Center crop to the smaller dimension to create a square
        shorter_side = min(image.size)  # Get the smaller of width or height
        image = ImageOps.fit(image, (shorter_side, shorter_side), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        
        # Resize to target dimensions
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.uint8)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        with model_lock:
            if not interpreter:
                return jsonify({'success': False, 'message': 'Model not loaded.'}), 500
            # Set the input tensor
            set_input(interpreter, image)

            # Run inference
            interpreter.invoke()

            # Get the results
            results = get_classes(interpreter, top_k=1)
            predicted_class = results[0].id  # Get the class index

        return predicted_class
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        logging.error(f"Error classifying image {image_path}: {e}")
        return 0  # Default to 'not_cat' in case of error

retrain_lock = threading.Lock()

retraining = False

def upload_model_to_esp32():
    """Upload the simple TFLite model to ESP32 via HTTP"""
    try:
        model_path = os.path.join('simple_model', 'my_simple_model_quant.tflite')
        if not os.path.exists(model_path):
            logging.error(f"Simple model file not found at {model_path}")
            return False
        
        # ESP32 IP address - should be configurable
        esp32_ip = credentials.ESP32_IP if hasattr(credentials, 'ESP32_IP') else '192.168.1.14'
        upload_url = f'http://{esp32_ip}/upload'
        
        logging.info(f"Uploading model to ESP32 at {upload_url}...")
        
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        files = {'file': ('model.tflite', model_data, 'application/octet-stream')}
        
        import requests
        response = requests.post(upload_url, files=files, timeout=30)
        
        if response.status_code == 200:
            logging.info("Model uploaded successfully to ESP32")
            retraining_status['output'] += "\nModel uploaded to ESP32 successfully!\n"
            return True
        else:
            error_msg = f"Failed to upload model to ESP32: {response.status_code} - {response.text}"
            logging.error(error_msg)
            retraining_status['output'] += f"\n{error_msg}\n"
            return False
            
    except Exception as e:
        error_msg = f"Error uploading model to ESP32: {e}"
        logging.error(error_msg)
        retraining_status['output'] += f"\n{error_msg}\n"
        return False

def run_retraining(epochs, fine_tune_epochs, learning_rate, fine_tune_at):
    global retraining_status
    with retrain_lock:
        logging.info("Starting retrain")
        if retraining_status['retraining']:
            logging.warning("Retraining is already in progress.")
            return
        retraining_status['retraining'] = True
        retraining_status['completed'] = False  # Reset completed flag
        retraining_status['error'] = None  # Reset previous errors
        retraining_status['output'] = ""    # Reset previous output
        retraining_status['progress'] = 0  # Reset progress
        update_model_info(retraining=True)
    
    try:
        logging.info("Starting model retraining...")
        
        # Validate input values
        if epochs < 1 or fine_tune_epochs < 0 or float(learning_rate) <= 0 or fine_tune_at < 0:
            flash('Invalid training parameters provided.', 'danger')
            return redirect(url_for('model'))

        # Path to the virtual environment's Python interpreter
        VENV_PATH = '/venv/coral'  # Adjust as per your virtual environment's path
        train_script_path = os.path.join(os.getcwd(), 'train_model.py')
        train_simple_script_path = os.path.join(os.getcwd(), 'train_simple_model.py')
        python_executable = os.path.join(VENV_PATH, 'bin', 'python')
        
        # Build the command for main Coral TPU model
        command = [
            python_executable,
            train_script_path,
            '--epochs', str(epochs),
            '--fine_tune_epochs', str(fine_tune_epochs),
            '--learning_rate', learning_rate,
            '--fine_tune_at', str(fine_tune_at)
        ]
        
        # Run the retraining script for Coral TPU model
        retraining_status['output'] += "=== Training Coral TPU Model ===\n"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # To capture output as string
        )

        logging.info("Coral TPU model retraining process started.")
        # Read the output in real-time
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()  # Remove leading/trailing whitespace
            if 'PROGRESS:' in line:
                # Extract progress value (scale to 50% for first phase)
                progress_value = int(line.split('PROGRESS:')[-1]) // 2
                retraining_status['progress'] = progress_value
                logging.info(f'Coral TPU retraining progress: {progress_value}%')
            else:
                # Regular output
                retraining_status['output'] += line + '\n'
                logging.info(line)
        process.stdout.close()
        return_code = process.wait()

        # Read any remaining stderr
        stderr = process.stderr.read()
        if stderr:
            logging.error(stderr.strip())
            retraining_status['output'] += stderr
        process.stderr.close()
        
        if return_code != 0:
            error_message = f"Coral TPU model training failed with return code: {return_code}"
            logging.error(error_message)
            retraining_status['error'] = error_message
            update_model_info(retraining=False)
            retraining_status['retraining'] = False
            return
        
        # Now train the simple ESP32 model
        retraining_status['output'] += "\n=== Training ESP32 Simple Model ===\n"
        simple_command = [
            python_executable,
            train_simple_script_path,
            '--epochs', str(40),  # More epochs for simple model
            '--learning_rate', '1e-3'  # Higher learning rate for custom CNN
        ]
        
        simple_process = subprocess.Popen(
            simple_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logging.info("ESP32 simple model training process started.")
        while True:
            line = simple_process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if 'PROGRESS:' in line:
                # Extract progress value (scale to 50-100% for second phase)
                progress_value = 50 + (int(line.split('PROGRESS:')[-1]) // 2)
                retraining_status['progress'] = progress_value
                logging.info(f'ESP32 model retraining progress: {progress_value}%')
            else:
                retraining_status['output'] += line + '\n'
                logging.info(line)
        simple_process.stdout.close()
        simple_return_code = simple_process.wait()
        
        # Read any remaining stderr
        simple_stderr = simple_process.stderr.read()
        if simple_stderr:
            logging.error(simple_stderr.strip())
            retraining_status['output'] += simple_stderr
        simple_process.stderr.close()
        
        if simple_return_code == 0:
            # Both models trained successfully
            retraining_status['last_trained'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retraining_status['images_used'] = count_current_dataset_images()
            retraining_status['completed'] = True
            update_model_info(
                last_trained=retraining_status['last_trained'],
                images_used=retraining_status['images_used'],
                retraining=False,
                epochs=epochs,
                fine_tune_epochs=fine_tune_epochs,
                learning_rate=learning_rate,
                fine_tune_at=fine_tune_at
            )
            logging.info("Both models retrained successfully.")
            load_model()  # Reload the Coral TPU model
            
            # Upload simple model to ESP32
            upload_model_to_esp32()
        else:
            error_message = f"ESP32 model training failed with return code: {simple_return_code}"
            logging.error(error_message)
            retraining_status['error'] = error_message
            update_model_info(retraining=False)
    
    except Exception as e:
        logging.error(f"An error occurred during retraining: {e}")
        retraining_status['error'] = str(e)
        update_model_info(retraining=False)
    
    finally:
        retraining_status['retraining'] = False


def count_current_dataset_images():
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add or remove as needed
    if not os.path.isdir(DATASET_IMAGES_DIR):
        app.logger.warning(f"Dataset directory '{DATASET_IMAGES_DIR}' does not exist.")
        return 0
    image_files = [f for f in os.listdir(DATASET_IMAGES_DIR) if f.lower().endswith(supported_extensions)]
    current_count = len(image_files)
    app.logger.info(f"Current number of images in dataset: {current_count}")
    return current_count

def count_current_classify_images():
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add or remove as needed
    if not os.path.isdir(STATIC_IMAGES_DIR):
        app.logger.warning(f"Classification directory '{STATIC_IMAGES_DIR}' does not exist.")
        return 0
    image_files = [f for f in os.listdir(STATIC_IMAGES_DIR) if f.lower().endswith(supported_extensions)]
    current_count = len(image_files)
    return current_count

def get_last_trained():
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            data = json.load(f)
            return data.get('last_trained', 'Never')
    except FileNotFoundError:
        return 'Never'
    except json.JSONDecodeError:
        return 'Never'

def update_last_trained():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"last_trained": now}
    with open(MODEL_INFO_PATH, 'w') as f:
        json.dump(data, f)

def get_model_info():
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            data = json.load(f)
            # Ensure 'training_params' exists
            if 'training_params' not in data:
                data['training_params'] = {
                    "epochs": 10,
                    "fine_tune_epochs": 5,
                    "learning_rate": '1e-5',
                    "fine_tune_at": 120
                }
            return data
    except FileNotFoundError:
        return {
            "last_trained": "Never",
            "images_used": 0,
            "training_params": {
                "epochs": 10,
                "fine_tune_epochs": 5,
                "learning_rate": '1e-5',
                "fine_tune_at": 120
            }
        }
    except json.JSONDecodeError:
        return {
            "last_trained": "Never",
            "images_used": 0,
            "training_params": {
                "epochs": 10,
                "fine_tune_epochs": 5,
                "learning_rate": '1e-5',
                "fine_tune_at": 120
            }
        }


def update_model_info(last_trained=None, images_used=None, retraining=None, epochs=None, fine_tune_epochs=None, learning_rate=None, fine_tune_at=None):
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {
            "last_trained": "Never",
            "images_used": 0,
            "retraining": False,
            "training_params": {}
        }

    if last_trained is not None:
        data['last_trained'] = last_trained
    if images_used is not None:
        data['images_used'] = images_used
    if retraining is not None:
        data['retraining'] = retraining
    if epochs is not None or fine_tune_epochs is not None or learning_rate is not None or fine_tune_at is not None:
        data['training_params'] = {
            "epochs": epochs if epochs is not None else data.get('training_params', {}).get('epochs', 10),
            "fine_tune_epochs": fine_tune_epochs if fine_tune_epochs is not None else data.get('training_params', {}).get('fine_tune_epochs', 10),
            "learning_rate": learning_rate if learning_rate is not None else data.get('training_params', {}).get('learning_rate', '1e-3'),
            "fine_tune_at": fine_tune_at if fine_tune_at is not None else data.get('training_params', {}).get('fine_tune_at', 150)
        }

    try:
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info("model_info.json updated successfully.")
    except IOError as e:
        logging.error(f"Failed to update model info: {e}")


# Flask Routes

@app.route('/')
def index():
    return redirect(url_for('classify'))

@app.route('/classify')
def classify():
    images = get_image_list(STATIC_IMAGES_DIR)
    return render_template('index.html', mode='classify', images=images, retraining_status=retraining_status)


@app.route('/update_label', methods=['POST'])
def update_label():
    data = request.get_json()
    filename = data.get('filename')
    label = data.get('label')
    action = data.get('action')  # 'toggle', 'save', 'back', or 'get_labels'
    mode = data.get('mode', 'classify')  # 'classify' or 'gallery'

    if not filename or not action:
        return jsonify({'success': False, 'message': 'Invalid data.'}), 400

    if mode == 'classify':
        image_dir = STATIC_IMAGES_DIR
    elif mode == 'gallery':
        image_dir = DATASET_IMAGES_DIR
    else:
        return jsonify({'success': False, 'message': 'Invalid mode.'}), 400

    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'message': 'Image not found.'}), 404

    if action == 'toggle':
        if label not in ['cat', 'morris', 'entering', 'prey']:
            return jsonify({'success': False, 'message': 'Invalid label.'}), 400

        labels = read_labels(image_path)
        current_state = labels.get(label, False)
        labels[label] = not current_state  # Toggle the label

        success = write_labels(image_path, labels)

        if success:
            return jsonify({'success': True, 'labels': labels})
        else:
            return jsonify({'success': False, 'message': 'Failed to update labels.'}), 500

    elif action == 'save':
        # Move the image to dataset directory
        dest_path = os.path.join(DATASET_IMAGES_DIR, filename)
        try:
            logging.debug("Attempting to move the image.")
            shutil.move(image_path, dest_path)
            logging.debug(f"Image moved successfully to {dest_path}.")

            return jsonify({'success': True, 'message': 'Image saved and moved.'})
        except Exception as e:
            print(f"Error moving image: {e}")
            logging.error(f"Failed to move image: {e}")
            return jsonify({'success': False, 'message': 'Failed to move image.'}), 500

    elif action == 'back':
        # No action needed on the server side
        return jsonify({'success': True, 'message': 'Back action received.'})

    elif action == 'get_labels':
        labels = read_labels(image_path)
        return jsonify({'success': True, 'labels': labels})

    else:
        return jsonify({'success': False, 'message': 'Invalid action.'}), 400

@app.route('/gallery')
def gallery():
    images = get_image_list(DATASET_IMAGES_DIR)
    return render_template('gallery.html', mode='gallery', images=images, retraining_status=retraining_status)

@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.get_json()
    filename = data.get('filename')
    mode = data.get('mode', 'classify')  # 'classify' or 'gallery'

    if not filename:
        return jsonify({'success': False, 'message': 'Filename not provided.'}), 400

    if mode == 'classify':
        image_dir = STATIC_IMAGES_DIR
    elif mode == 'gallery':
        image_dir = DATASET_IMAGES_DIR
    else:
        return jsonify({'success': False, 'message': 'Invalid mode.'}), 400

    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        return jsonify({'success': False, 'message': 'Image not found.'}), 404

    try:
        os.remove(image_path)
        logging.info(f"Deleted image: {image_path}")

        return jsonify({'success': True, 'message': 'Image deleted successfully.'})
    except Exception as e:
        logging.error(f"Error deleting image {image_path}: {e}")
        return jsonify({'success': False, 'message': 'Failed to delete image.'}), 500

@app.route('/delete_all_images', methods=['POST'])
def delete_all_images():
    try:
        image_files = [f for f in os.listdir(STATIC_IMAGES_DIR) if allowed_file(f)]
        for filename in image_files:
            image_path = os.path.join(STATIC_IMAGES_DIR, filename)
            os.remove(image_path)
        flash('All images have been deleted successfully.', 'success')
        logging.info("All images in the classification folder have been deleted.")
        return redirect(url_for('classify'))
    except Exception as e:
        logging.error(f"Error deleting all images: {e}")
        flash('An error occurred while deleting images.', 'danger')
        return redirect(url_for('classify'))


@app.route('/image/<mode>/<filename>')
def send_image(mode, filename):
    if mode == 'classify':
        return send_from_directory(STATIC_IMAGES_DIR, filename)
    elif mode == 'gallery':
        return send_from_directory(DATASET_IMAGES_DIR, filename)
    else:
        return "Invalid mode", 400

@app.route('/model')
def model():
    # Load class weights
    class_weights_filename = os.path.join('static', 'reports', 'class_weights.json')
    with open(class_weights_filename, 'r') as f:
        class_weights = json.load(f)
    model_info = get_model_info()
    last_trained = model_info.get('last_trained', 'Never')
    images_used = model_info.get('images_used', 0)
    current_dataset_images = count_current_dataset_images()
    # Class names
    class_names = ['not_cat', 'unknown_cat_entering', 'cat_morris_leaving', 'cat_morris_entering', 'prey']
    
    # Get training parameters
    training_params = model_info.get('training_params', {})
    epochs = training_params.get('epochs', 10)
    fine_tune_epochs = training_params.get('fine_tune_epochs', 5)
    learning_rate = training_params.get('learning_rate', '1e-5')
    fine_tune_at = training_params.get('fine_tune_at', 120)
    
    learning_rates = ['5e-6', '6e-6', '7e-6', '8e-6', '9e-6', '1e-5', '2e-5', '3e-5', '4e-5', '5e-5']


    return render_template('model.html',
                            mode='model', 
                            class_weights=class_weights, 
                            class_names=class_names,
                            last_trained=last_trained, 
                            images_used=images_used, 
                            current_dataset_images=current_dataset_images,
                            retraining_status=retraining_status,
                            epochs=epochs,
                            fine_tune_epochs=fine_tune_epochs,
                            learning_rate=learning_rate,
                            fine_tune_at=fine_tune_at,
                            learning_rates=learning_rates)



@app.route('/about')
def about():
    return render_template('about.html',
                           mode='about')

@app.route('/retrain', methods=['POST'])
def retrain_model():
    with retrain_lock:
        if retraining_status['retraining']:
            flash('Retraining is already in progress.', 'warning')
            return redirect(url_for('model'))
        
        # Retrieve form data
        epochs = request.form.get('epochs', default=10, type=int)
        fine_tune_epochs = request.form.get('fine_tune_epochs', default=5, type=int)
        learning_rate_str = request.form.get('learning_rate', default='8e-6')
        fine_tune_at = request.form.get('fine_tune_at', default=120, type=int)
        
        # Convert learning rate string to float
        try:
            learning_rate = float(eval(learning_rate_str))
        except Exception as e:
            flash('Invalid learning rate selected.', 'danger')
            return redirect(url_for('model'))
        
        # Start retraining in a separate thread and pass parameters
        retrain_thread = threading.Thread(target=run_retraining, args=(epochs, fine_tune_epochs, learning_rate_str, fine_tune_at))
        retrain_thread.start()
        
    flash('Retraining started successfully!', 'success')
    return redirect(url_for('model'))


@app.route('/status')
def status():
    global retraining_status
    status_copy = retraining_status.copy()
    if retraining_status['completed']:
        # Reset 'completed' flag after sending it to the client
        retraining_status['completed'] = False
    return jsonify(status_copy)


# Make read_labels available to templates
@app.context_processor
def utility_processor():
    def get_labels(image_filename):
        image_path = os.path.join(DATASET_IMAGES_DIR, image_filename)
        if not os.path.exists(image_path):
            image_path = os.path.join(STATIC_IMAGES_DIR, image_filename)
        return read_labels(image_path)
    return dict(read_labels=get_labels)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
