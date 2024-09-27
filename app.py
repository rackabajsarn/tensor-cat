import os
import threading
import base64
import datetime
import random
import credentials
from flask import Flask, render_template, request, redirect, url_for, flash
from flask import send_from_directory
from paho.mqtt import client as mqtt_client

app = Flask(__name__)
app.secret_key = credentials.SECRET_KEY

# MQTT Configuration
MQTT_BROKER = credentials.MQTT_SERVER
MQTT_PORT = 1883
MQTT_TOPIC = 'catflap/image'

# Directories
STATIC_IMAGES_DIR = 'static/images'
DATASET_DIR = 'dataset'

# Ensure directories exist
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'train', 'prey'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'train', 'no_prey'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'validation', 'prey'), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, 'validation', 'no_prey'), exist_ok=True)

# MQTT Client
# Flag to indicate initial connection
initial_connection = True

def mqtt_on_connect(client, userdata, flags, rc):
    global initial_connection
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Reset the initial_connection flag on successful connection
        initial_connection = True
        client.subscribe(MQTT_TOPIC)
    else:
        print("Failed to connect to MQTT Broker, return code %d\n", rc)

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

    except Exception as e:
        print(f"Error processing message: {e}")

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

# Flask Routes
@app.route('/')
def index():
    return redirect(url_for('classify'))

@app.route('/classify')
def classify():
    # Get list of image filenames
    images = os.listdir(STATIC_IMAGES_DIR)
    images.sort(reverse=True)  # Show latest images first
    return render_template('index.html', mode='classify', images=images)

# New Route: Delete All Images
@app.route('/delete_all_images', methods=['POST'])
def delete_all_images():
    # Delete all images in the classification folder
    if os.path.exists(STATIC_IMAGES_DIR):
        for filename in os.listdir(STATIC_IMAGES_DIR):
            file_path = os.path.join(STATIC_IMAGES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All images deleted from classification folder.")
        flash("All images have been deleted.", 'success')
    else:
        flash("Classification folder does not exist.", 'error')
    return redirect(url_for('classify'))

@app.route('/classify/<action>/<filename>', methods=['POST'])
def classify_image(action, filename):
    # Determine destination directory
    train_or_val = random.choice(['train', 'validation'])
    prey_or_no_prey = 'prey' if action == 'prey' else 'no_prey'
    dest_dir = os.path.join(DATASET_DIR, train_or_val, prey_or_no_prey)

    # Securely handle the filename
    filename = os.path.basename(filename)
    src_path = os.path.join(STATIC_IMAGES_DIR, filename)
    dest_path = os.path.join(dest_dir, filename)

    # Move the image
    if os.path.exists(src_path):
        os.rename(src_path, dest_path)
        print(f"Moved {filename} to {dest_path}")
        flash(f"Image {filename} classified as {'Prey' if action == 'prey' else 'No Prey'}.")
    else:
        flash("Image not found.", 'error')

    # Redirect back to classify
    return redirect(url_for('classify'))

@app.route('/gallery')
def gallery():
    # Collect images from dataset directories
    categories = {
        'Prey': [],
        'No Prey': []
    }

    for subset in ['train', 'validation']:
        for label in ['prey', 'no_prey']:
            dir_path = os.path.join(DATASET_DIR, subset, label)
            files = os.listdir(dir_path)
            for file in files:
                categories['Prey' if label == 'prey' else 'No Prey'].append({
                    'filename': file,
                    'filepath': f'{subset}/{label}/{file}'
                })

    # Sort images by filename (assumed timestamp)
    for key in categories:
        categories[key].sort(key=lambda x: x['filename'], reverse=True)

    return render_template('gallery.html', mode='gallery', categories=categories)

@app.route('/gallery/delete/<path:filepath>', methods=['POST'])
def delete_image(filepath):
    # Securely handle the filepath
    filepath = os.path.normpath(filepath)
    allowed_dirs = [
        'train/prey', 'train/no_prey', 'validation/prey', 'validation/no_prey'
    ]
    if any(filepath.startswith(d) for d in allowed_dirs):
        image_path = os.path.join(DATASET_DIR, filepath)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_path}")
            flash(f"Image {os.path.basename(image_path)} deleted.")
        else:
            flash("Image not found.", 'error')
    else:
        flash("Invalid image path.", 'error')

    # Redirect back to gallery
    return redirect(url_for('gallery'))

@app.route('/image/<mode>/<path:filename>')
def send_image(mode, filename):
    if mode == 'classify':
        return send_from_directory(STATIC_IMAGES_DIR, filename)
    elif mode == 'gallery':
        # Extract subset and label from filename
        return send_from_directory(DATASET_DIR, filename)
    else:
        return "Invalid mode", 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
