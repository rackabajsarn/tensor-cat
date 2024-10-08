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
    'output': ""  # To capture retraining output
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
        # Decode the Base64-encoded image
        base64_data = msg.payload.decode('utf-8')
        image_data = base64.b64decode(base64_data)

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
            client.publish('catflap/debug', 'Empty picture received')
            logging.warning('Empty picture received?')
        elif predicted_label == 'cat_morris_entering':
            labels['cat'] = True
            labels['morris'] = True
            labels['entering'] = True
            client.publish('catflap/cat_location/set', 'ON')
            client.publish('catflap/flap_state/set', 'ON')
        elif predicted_label == 'cat_morris_leaving':
            labels['cat'] = True
            labels['morris'] = True
            labels['entering'] = False
            client.publish('catflap/cat_location/set', 'OFF')
        elif predicted_label == 'prey':
            labels['cat'] = True
            labels['morris'] = True
            labels['prey'] = True
            client.publish('catflap/alert', 'Prey Detected')
            client.publish('catflap/flap_state/set', 'OFF')
        elif predicted_label == 'unknown_cat_entering':
            labels['cat'] = True
            labels['morris'] = False
            labels['entering'] = True
            client.publish('catflap/alert', 'Peekaboo!')

        # Write labels to EXIF
        write_labels(image_path, labels)
        
        client.publish('catflap/debug', f"Inference ({predicted_label}) done in {int((end - start)*1000)} ms")
        new_images = count_current_classify_images()
        message = f"{new_images} New image to classify" if new_images < 2 else f"{new_images} New images to classify"
        client.publish('catflap/alert', message)
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
        image = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
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

def run_retraining():
    global retraining_status
    with retrain_lock:
        logging.info("Starting retrain")
        if retraining_status['retraining']:
            logging.warning("Retraining is already in progress.")
            return
        retraining_status['retraining'] = True
        retraining_status['error'] = None  # Reset previous errors
        retraining_status['output'] = ""    # Reset previous output
        update_model_info(retraining=True)
    
    try:
        logging.info("Starting model retraining...")
        
        # Path to the virtual environment's Python interpreter
        VENV_PATH = '/venv/coral'  # Adjust as per your virtual environment's path
        train_script_path = os.path.join(os.getcwd(), 'train_model.py')  # Ensure correct path
        python_executable = os.path.join(VENV_PATH, 'bin', 'python')
        
        # Run the retraining script using the virtual environment's Python
        process = subprocess.Popen(
            [python_executable, train_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # To capture output as string
        )

        logging.info("Starting retraining process...")
        # Read the output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logging.info(line.strip())
                retraining_status['output'] += line
        process.stdout.close()
        return_code = process.wait()

        # Read any remaining stderr
        stderr = process.stderr.read()
        if stderr:
            logging.error(stderr.strip())
            retraining_status['output'] += stderr
        process.stderr.close()
        
        if return_code == 0:
            retraining_status['last_trained'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retraining_status['images_used'] = count_current_dataset_images()
            update_model_info(last_trained=retraining_status['last_trained'],
                              images_used=retraining_status['images_used'],
                              retraining=False)  # Update persistent state
            logging.info("Model retraining completed successfully.")
            load_model()
        else:
            # On failure
            error_message = f"Retraining failed with return code: {return_code}"
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
            return data
    except FileNotFoundError:
        return {"last_trained": "Never", "images_used": 0}
    except json.JSONDecodeError:
        return {"last_trained": "Never", "images_used": 0}

def update_model_info(last_trained=None, images_used=None, retraining=None):
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"last_trained": "Never", "images_used": 0, "retraining": False}

    if last_trained is not None:
        data['last_trained'] = last_trained
    if images_used is not None:
        data['images_used'] = images_used
    if retraining is not None:
        data['retraining'] = retraining

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

@app.route('/image/<mode>/<filename>')
def send_image(mode, filename):
    if mode == 'classify':
        return send_from_directory(STATIC_IMAGES_DIR, filename)
    elif mode == 'gallery':
        return send_from_directory(DATASET_IMAGES_DIR, filename)
    else:
        return "Invalid mode", 400

@app.route('/about')
def about():
    model_info = get_model_info()
    last_trained = model_info.get('last_trained', 'Never')
    images_used = model_info.get('images_used', 0)
    current_dataset_images = count_current_dataset_images()
    return render_template('about.html', 
                           last_trained=last_trained, 
                           images_used=images_used, 
                           current_dataset_images=current_dataset_images,
                           retraining_status=retraining_status)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    with retrain_lock:
        if retraining_status['retraining']:
            flash('Retraining is already in progress.', 'warning')
            return redirect(url_for('about'))
        
        # Start retraining in a separate thread
        retrain_thread = threading.Thread(target=run_retraining)
        retrain_thread.start()
        
    flash('Retraining started successfully!', 'success')
    return redirect(url_for('about'))

@app.route('/status')
def status():
    return jsonify(retraining_status)

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
