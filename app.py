import os
import threading
import base64
import datetime
import credentials
import logging
import json
import shutil
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

# Load the Edge TPU model
print("Loading the Edge TPU model...")
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Classes mapping
CLASSES = ['not_cat', 'cat_entering', 'cat_leaving', 'prey']
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

        # Set initial EXIF tags based on prediction
        labels = {
            "cat": False,
            "morris": False,
            "entering": False,
            "prey": False
        }

        if predicted_label == 'not_cat':
            labels['cat'] = False
        elif predicted_label == 'cat_entering':
            labels['cat'] = True
            labels['entering'] = True
        elif predicted_label == 'cat_leaving':
            labels['cat'] = True
            labels['entering'] = False
        elif predicted_label == 'prey':
            labels['cat'] = True
            labels['prey'] = True

        # Write labels to EXIF
        write_labels(image_path, labels)

        print(f"Image classified as {predicted_label} and labels updated.")
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

# Flask Routes

@app.route('/')
def index():
    return redirect(url_for('classify'))

@app.route('/classify')
def classify():
    images = get_image_list(STATIC_IMAGES_DIR)
    return render_template('index.html', mode='classify', images=images)

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
    return render_template('gallery.html', mode='gallery', images=images)

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
    return render_template('about.html', mode='about')

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
