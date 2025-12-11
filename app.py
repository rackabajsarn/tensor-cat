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
import copy
import sys
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from paho.mqtt import client as mqtt_client
import piexif
from PIL import Image
from PIL import ImageOps


def _consume_flag(flag_name):
    """Remove a CLI flag from sys.argv if present and report whether it was found."""
    if flag_name in sys.argv:
        sys.argv.remove(flag_name)
        return True
    return False


if _consume_flag('--offline'):
    OFFLINE_MODE = True
elif _consume_flag('--online'):
    OFFLINE_MODE = False
else:
    OFFLINE_MODE = False  # default behavior is online

try:
    if OFFLINE_MODE:
        raise ImportError("Offline mode active")
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters.common import set_input
    from pycoral.adapters.classify import get_classes
    PY_CORAL_AVAILABLE = True
except ImportError:
    make_interpreter = None
    set_input = None
    get_classes = None
    PY_CORAL_AVAILABLE = False

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
MODELS_DIR = 'models'
SERVER_MODELS_DIR = os.path.join(MODELS_DIR, 'server')
LOCAL_MODELS_DIR = os.path.join(MODELS_DIR, 'local')
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, 'active.json')
DB_PATH = os.path.join(os.getcwd(), 'tensor_cat.db')

# Ensure model directories exist
os.makedirs(SERVER_MODELS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

SERVER_PARAM_DEFAULTS = {
    "epochs": 10,
    "fine_tune_epochs": 5,
    "learning_rate": '1e-5',
    "fine_tune_at": 120
}

LOCAL_PARAM_DEFAULTS = {
    "epochs": 40,
    "learning_rate": '1e-3',
    "batch_size": 32,
    "seed": 0,
    "class_count": 2,
    "max_samples_per_class": 0
}

# Valid parameter ranges for local model training
LOCAL_PARAM_LIMITS = {
    "epochs": {"min": 10, "max": 120, "step": 5},
    "learning_rate": ['5e-4', '7.5e-4', '1e-3', '1.5e-3', '2e-3', '3e-3', '5e-3'],
    "batch_size": [16, 32, 48, 64],
    "seed": {"min": 0, "max": 999999},
    "class_count": [2, 3],
    "max_samples_per_class": {"min": 0, "max": 1000}
}


def default_section(params_defaults):
    return {
        "last_trained": "Never",
        "images_used": 0,
        "retraining": False,
        "training_params": copy.deepcopy(params_defaults)
    }


def get_active_models():
    """Get currently active model versions for server and local."""
    if os.path.exists(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"server": None, "local": None}


def get_active_local_class_count(default=LOCAL_PARAM_DEFAULTS["class_count"]):
    """Return class_count (2 or 3) for the active local model based on its metrics.json."""
    active = get_active_models()
    version = active.get('local') if isinstance(active, dict) else None
    if not version:
        return default

    metrics_path = os.path.join(LOCAL_MODELS_DIR, version, 'reports', 'metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        if isinstance(metrics, dict):
            training_params = metrics.get('training_params', {})
            class_count = training_params.get('class_count') if isinstance(training_params, dict) else None
            if class_count in (2, 3):
                return int(class_count)

            classes = metrics.get('classes')
            if isinstance(classes, list) and len(classes) in (2, 3):
                return len(classes)
    except Exception as e:
        logging.error(f"Failed to read class_count for active local model {version}: {e}")

    return default


def set_active_model(scope, version_name):
    """Set the active model version for a scope (server/local)."""
    active = get_active_models()
    active[scope] = version_name
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        json.dump(active, f)


def save_model_version(scope, version_name):
    """Save metadata for an existing model version folder.

    Trainers now write model artifacts and reports directly into
    models/<scope>/<version_name>/{model,reports}. This function should
    NOT create a new version directory; it only attaches metadata based
    on metrics.json when available.
    """
    if scope == 'server':
        models_dir = SERVER_MODELS_DIR
    else:
        models_dir = LOCAL_MODELS_DIR

    version_dir = os.path.join(models_dir, version_name)
    if not os.path.isdir(version_dir):
        logging.error(f"Version directory does not exist for {scope}: {version_name}")
        return None

    # Try to read metrics.json generated by the trainer
    reports_dir = os.path.join(version_dir, 'reports')
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logging.error(f"Error loading metrics for version {version_name}: {e}")

    # Save metadata with basic metrics if present
    summary_metrics = {}
    try:
        if isinstance(metrics, dict):
            report = metrics.get('report', {})
            if isinstance(report, dict) and 'accuracy' in report:
                summary_metrics['accuracy'] = float(report['accuracy'])
    except Exception:
        pass

    metadata = {
        "timestamp": version_name,
        "created": datetime.datetime.now().isoformat(),
        "metrics": summary_metrics
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return version_dir


def extract_model_metrics(report_path):
    """Extract key metrics (accuracy, macro avg) from classification report."""
    rows = parse_classification_report(report_path)
    if not rows:
        return {}
    
    metrics = {}
    for row in rows:
        if len(row) >= 2:
            first_cell = row[0].lower().strip()
            # Check for accuracy row (usually "Accuracy" in first cells, value in last)
            if 'accuracy' in first_cell:
                # Find accuracy value - usually the last non-empty cell
                for cell in reversed(row):
                    try:
                        val = float(cell)
                        metrics['accuracy'] = val
                        break
                    except (ValueError, TypeError):
                        continue
            # Check for macro avg F1
            elif 'f1' in first_cell and 'score' in first_cell:
                for cell in row[1:]:
                    try:
                        val = float(cell)
                        metrics['macro_f1'] = val
                        break
                    except (ValueError, TypeError):
                        continue
    
    return metrics


def list_model_versions(scope):
    """List all saved model versions for a scope."""
    models_dir = SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR
    versions = []
    
    if not os.path.exists(models_dir):
        return versions
    
    for name in sorted(os.listdir(models_dir), reverse=True):
        version_dir = os.path.join(models_dir, name)
        if not os.path.isdir(version_dir):
            continue
        
        metadata_path = os.path.join(version_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['name'] = name
                    metadata['path'] = version_dir
                    versions.append(metadata)
            except Exception as e:
                logging.error(f"Error loading metadata for {name}: {e}")
                versions.append({
                    'name': name,
                    'path': version_dir,
                    'timestamp': name,
                    'metrics': {}
                })
        else:
            versions.append({
                'name': name,
                'path': version_dir,
                'timestamp': name,
                'metrics': {}
            })
    
    return versions


def get_model_version_details(scope, version_name):
    """Get full details for a specific model version."""
    models_dir = SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR
    version_dir = os.path.join(models_dir, version_name)

    if not os.path.exists(version_dir):
        return None

    details = {
        'name': version_name,
        'path': version_dir
    }

    # Load metadata
    metadata_path = os.path.join(version_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            details.update(json.load(f))

    # Load metrics.json for detailed report
    reports_dir = os.path.join(version_dir, 'reports')
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                details['metrics'] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading metrics for {scope} version {version_name}: {e}")
            details['metrics'] = None
    else:
        details['metrics'] = None

    # Load model summary from metrics if present
    metrics_obj = details.get('metrics') or {}
    details['model_summary'] = metrics_obj.get('model_summary')

    return details


def parse_classification_report(html_path):
    """Parse classification_report.html and return structured data."""
    if not os.path.exists(html_path):
        return None
    try:
        from html.parser import HTMLParser
        
        class ReportParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.current_row = []
                self.current_cell = ""
                self.in_td = False
                self.in_th = False
                
            def handle_starttag(self, tag, attrs):
                if tag == 'tr':
                    self.current_row = []
                elif tag in ('td', 'th'):
                    self.in_td = tag == 'td'
                    self.in_th = tag == 'th'
                    self.current_cell = ""
                    
            def handle_endtag(self, tag):
                if tag == 'tr' and self.current_row:
                    self.rows.append(self.current_row)
                elif tag in ('td', 'th'):
                    self.current_row.append(self.current_cell.strip())
                    self.in_td = False
                    self.in_th = False
                    
            def handle_data(self, data):
                if self.in_td or self.in_th:
                    self.current_cell += data
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = ReportParser()
        parser.feed(content)
        return parser.rows
    except Exception as e:
        logging.error(f"Error parsing classification report: {e}")
        return None


def read_model_summary(txt_path):
    """Read model summary text file."""
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading model summary: {e}")
        return None


def default_model_info():
    return {
        "server": default_section(SERVER_PARAM_DEFAULTS),
        "local": default_section(LOCAL_PARAM_DEFAULTS)
    }

# Shared state
def make_status_dict():
    return {
        'retraining': False,
        'error': None,
        'last_trained': None,
        'images_used': 0,
        'output': "",
        'progress': 0,
        'completed': False
    }


retraining_status = make_status_dict()
local_retraining_status = make_status_dict()


# Ensure directories exist
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
os.makedirs(DATASET_IMAGES_DIR, exist_ok=True)

LOG_DIR = os.environ.get('TENSOR_CAT_LOG_DIR', os.path.join(os.getcwd(), 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

logging.basicConfig(
    filename=LOG_FILE,
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
db_lock = threading.Lock()


def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_log (
                    hash TEXT PRIMARY KEY,
                    timestamp TEXT,
                    esp32_model TEXT,
                    server_model TEXT,
                    esp32_inference TEXT,
                    server_inference TEXT,
                    server_simple_inference TEXT,
                    true_label TEXT,
                    esp32_confidence REAL,
                    server_confidence REAL
                );
                """
            )
            conn.commit()
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")


def fnv1a_32(data: bytes) -> int:
    FNV_OFFSET_BASIS = 2166136261
    FNV_PRIME = 16777619
    h = FNV_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF  # keep 32-bit
    return h


def hash_to_name(data: bytes) -> str:
    return f"{fnv1a_32(data):08X}"


def upsert_inference_record(hash_hex, *, timestamp, esp32_model=None, server_model=None,
                            esp32_inference=None, server_inference=None, server_simple_inference=None,
                            true_label=None, esp32_confidence=None, server_confidence=None):
    try:
        with db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO inference_log (
                    hash, timestamp, esp32_model, server_model,
                    esp32_inference, server_inference, server_simple_inference, true_label,
                    esp32_confidence, server_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    esp32_model=excluded.esp32_model,
                    server_model=excluded.server_model,
                    esp32_inference=excluded.esp32_inference,
                    server_inference=excluded.server_inference,
                    server_simple_inference=excluded.server_simple_inference,
                    true_label=excluded.true_label,
                    esp32_confidence=excluded.esp32_confidence,
                    server_confidence=excluded.server_confidence;
                """,
                [hash_hex, timestamp, esp32_model, server_model,
                 esp32_inference, server_inference, server_simple_inference, true_label,
                 esp32_confidence, server_confidence]
            )
            conn.commit()
    except Exception as e:
        logging.error(f"Failed to upsert inference_log for hash {hash_hex}: {e}")


def read_json_file(path, default=None):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def static_asset_exists(relative_path):
    normalized = relative_path.replace('/', os.sep)
    return os.path.exists(os.path.join(app.static_folder, normalized))

def load_model():
    global interpreter
    if OFFLINE_MODE or not PY_CORAL_AVAILABLE:
        logging.info("Offline mode or missing Coral libraries; skipping model load.")
        interpreter = None
        return
    # Load the active server version's Edge TPU model
    active = get_active_models()
    server_version = active.get('server')
    if server_version:
        model_path = os.path.join(SERVER_MODELS_DIR, server_version, 'model', MODEL_NAME)
    else:
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
if not OFFLINE_MODE and PY_CORAL_AVAILABLE:
    load_model()
else:
    logging.info("Application running in offline mode; Edge TPU inference disabled.")

# Initialize database on startup
init_db()

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
        image_data = msg.payload
        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp}.jpg"
        image_path = os.path.join(STATIC_IMAGES_DIR, image_filename)

        # Save the image to disk
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved to {image_path}")

        # Classify the image
        predicted_class, server_confidence = classify_image(image_path)
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
        
        # Hash the raw image bytes as primary key for DB logging
        img_hash = hash_to_name(image_data)
        active_models = get_active_models()
        local_class_count = get_active_local_class_count()
        if predicted_label == "prey":
            server_simple = "prey"
        elif local_class_count == 3 and predicted_label == "not_cat":
            server_simple = "not_cat"
        else:
            server_simple = "not_prey"
        server_model_name = active_models.get('server') if isinstance(active_models, dict) else None
        timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'
        upsert_inference_record(
            img_hash,
            timestamp=timestamp_iso,
            server_model=server_model_name,
            server_inference=predicted_label,
            server_simple_inference=server_simple,
            server_confidence=server_confidence,
        )

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
    if OFFLINE_MODE:
        logging.info("Offline mode - MQTT listener disabled.")
        return
    client = mqtt_client.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# Start MQTT client in a separate thread (only when online)
if not OFFLINE_MODE:
    mqtt_thread = threading.Thread(target=mqtt_listen)
    mqtt_thread.daemon = True
    mqtt_thread.start()
else:
    logging.info("MQTT communication disabled for offline testing.")

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
        
        if OFFLINE_MODE or not interpreter or not PY_CORAL_AVAILABLE:
            logging.debug("Offline mode - returning default classification")
            return 0, None

        with model_lock:
            # Set the input tensor
            set_input(interpreter, image)

            # Run inference
            interpreter.invoke()

            # Get the results
            results = get_classes(interpreter, top_k=1)
            predicted_class = results[0].id  # Get the class index
            predicted_score = getattr(results[0], 'score', None)

        return predicted_class, predicted_score
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        logging.error(f"Error classifying image {image_path}: {e}")
        return 0, None  # Default to 'not_cat' in case of error

retrain_lock = threading.Lock()

local_retrain_lock = threading.Lock()

retraining = False

def upload_model_to_esp32(version_name=None):
    """Upload the simple TFLite model to ESP32 via HTTP.

    Returns (success: bool, message: str) for clearer propagation to callers/UI.
    If version_name is provided, upload that local version; otherwise use the current active local model.
    """
    try:
        if OFFLINE_MODE:
            logging.info("Offline mode - skipping ESP32 model upload.")
            return False, "offline mode"

        # Prefer the provided version; otherwise use the active local version; otherwise legacy simple_model
        if version_name:
            model_path = os.path.join(LOCAL_MODELS_DIR, version_name, 'model', 'my_simple_model_quant.tflite')
        else:
            active = get_active_models()
            local_version = active.get('local')
            if local_version:
                model_path = os.path.join(LOCAL_MODELS_DIR, local_version, 'model', 'my_simple_model_quant.tflite')
            else:
                model_path = os.path.join('simple_model', 'my_simple_model_quant.tflite')
        if not os.path.exists(model_path):
            msg = f"Simple model file not found at {model_path}"
            logging.error(msg)
            return False, msg
        
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
            msg = "Model uploaded successfully to ESP32"
            logging.info(msg)
            retraining_status['output'] += "\nModel uploaded to ESP32 successfully!\n"
            return True, msg
        else:
            error_msg = f"Failed to upload model to ESP32: {response.status_code} - {response.text}"
            logging.error(error_msg)
            retraining_status['output'] += f"\n{error_msg}\n"
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error uploading model to ESP32: {e}"
        logging.error(error_msg)
        retraining_status['output'] += f"\n{error_msg}\n"
        return False, error_msg

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

        # Pre-generate a run_id so trainer and metadata share the same version name
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate input values
        if epochs < 1 or fine_tune_epochs < 0 or float(learning_rate) <= 0 or fine_tune_at < 0:
            flash('Invalid training parameters provided.', 'danger')
            return redirect(url_for('model'))

        # Path to the virtual environment's Python interpreter
        # VENV_PATH = '/venv/coral'  # Adjust as per your virtual environment's path
        train_script_path = os.path.join(os.getcwd(), 'train_model.py')
        #python_executable = os.path.join(VENV_PATH, 'bin', 'python')
        python_executable = sys.executable  # Use the current Python interpreter
        
        # Build the command for main Coral TPU model
        command = [
            python_executable,
            train_script_path,
            '--epochs', str(epochs),
            '--fine_tune_epochs', str(fine_tune_epochs),
            '--learning_rate', learning_rate,
            '--fine_tune_at', str(fine_tune_at),
            '--run_id', run_id
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
                try:
                    progress_value = int(line.split('PROGRESS:')[-1])
                    retraining_status['progress'] = max(0, min(100, progress_value))
                    logging.info(f'Retraining progress: {progress_value}%')
                except ValueError:
                    logging.debug(f"Unable to parse progress line: {line}")
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
        
        # Coral retraining completed successfully
        retraining_status['last_trained'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        retraining_status['images_used'] = count_current_dataset_images()
        retraining_status['completed'] = True
        retraining_status['progress'] = 100

        # Save this as a new model version (but don't activate - user must select)
        try:
            save_model_version('server', run_id)
            logging.info(f"Saved server model version: {run_id}")
        except Exception as e:
            logging.error(f"Failed to save model version: {e}")

        update_model_info(
            last_trained=retraining_status['last_trained'],
            images_used=retraining_status['images_used'],
            retraining=False,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            learning_rate=learning_rate,
            fine_tune_at=fine_tune_at
        )
        logging.info("Coral TPU model retrained successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred during retraining: {e}")
        retraining_status['error'] = str(e)
        update_model_info(retraining=False)
    
    finally:
        retraining_status['retraining'] = False


def run_local_retraining(epochs, learning_rate, batch_size, seed, class_count, max_samples_per_class):
    global local_retraining_status
    with local_retrain_lock:
        logging.info("Starting local retrain")
        if local_retraining_status['retraining']:
            logging.warning("Local retraining is already in progress.")
            return
        local_retraining_status['retraining'] = True
        local_retraining_status['completed'] = False
        local_retraining_status['error'] = None
        local_retraining_status['output'] = ""
        local_retraining_status['progress'] = 0
        update_model_info(section='local', retraining=True)

    try:
        python_executable = sys.executable
        train_simple_script_path = os.path.join(os.getcwd(), 'train_simple_model.py')
        logging.info("Launching local training script...")
        # Pre-generate a run_id so trainer and metadata share the same version name
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        command = [
            python_executable,
            train_simple_script_path,
            '--epochs', str(epochs),
            '--learning_rate', str(learning_rate),
            '--batch_size', str(batch_size),
            '--seed', str(seed),
            '--class_count', str(class_count),
            '--run_id', run_id
        ]

        if max_samples_per_class > 0:
            command.extend(['--max_samples_per_class', str(max_samples_per_class)])

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if 'PROGRESS:' in line:
                try:
                    progress_value = int(line.split('PROGRESS:')[-1])
                    local_retraining_status['progress'] = max(0, min(100, progress_value))
                except ValueError:
                    logging.debug(f"Unable to parse local progress line: {line}")
            else:
                local_retraining_status['output'] += line + '\n'
                logging.info(line)

        process.stdout.close()
        return_code = process.wait()

        stderr = process.stderr.read()
        if stderr:
            logging.error(stderr.strip())
            local_retraining_status['output'] += stderr
        process.stderr.close()

        if return_code != 0:
            error_message = f"Local model training failed with return code: {return_code}"
            logging.error(error_message)
            local_retraining_status['error'] = error_message
            update_model_info(section='local', retraining=False)
            return

        local_retraining_status['last_trained'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        local_retraining_status['images_used'] = count_current_dataset_images()
        local_retraining_status['completed'] = True
        local_retraining_status['progress'] = 100

        # Save this as a new model version (but don't activate - user must select)
        try:
            save_model_version('local', run_id)
            logging.info(f"Saved local model version: {run_id}")
        except Exception as e:
            logging.error(f"Failed to save model version: {e}")

        update_model_info(
            section='local',
            last_trained=local_retraining_status['last_trained'],
            images_used=local_retraining_status['images_used'],
            retraining=False,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            class_count=class_count,
            max_samples_per_class=max_samples_per_class
        )
        logging.info("Local model retrained successfully.")

    except Exception as e:
        logging.error(f"An error occurred during local retraining: {e}")
        local_retraining_status['error'] = str(e)
        update_model_info(section='local', retraining=False)

    finally:
        local_retraining_status['retraining'] = False


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

def _ensure_section(data, section, defaults):
    if section not in data or not isinstance(data[section], dict):
        data[section] = default_section(defaults)
    section_data = data[section]
    section_data.setdefault('last_trained', 'Never')
    section_data.setdefault('images_used', 0)
    section_data.setdefault('retraining', False)
    training_params = section_data.setdefault('training_params', {})
    for key, value in defaults.items():
        training_params.setdefault(key, value)
    return section_data


def get_model_info():
    data = default_model_info()
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            file_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        file_data = {}

    # Backwards compatibility for legacy structure
    if 'server' not in file_data and 'last_trained' in file_data:
        legacy = {
            'last_trained': file_data.get('last_trained', 'Never'),
            'images_used': file_data.get('images_used', 0),
            'retraining': file_data.get('retraining', False),
            'training_params': file_data.get('training_params', {})
        }
        file_data = {'server': legacy}

    data.update(file_data)
    _ensure_section(data, 'server', SERVER_PARAM_DEFAULTS)
    _ensure_section(data, 'local', LOCAL_PARAM_DEFAULTS)
    return data


def update_model_info(section='server', last_trained=None, images_used=None, retraining=None,
                      epochs=None, fine_tune_epochs=None, learning_rate=None, fine_tune_at=None,
                      batch_size=None, seed=None, class_count=None, max_samples_per_class=None):
    data = get_model_info()
    section_defaults = SERVER_PARAM_DEFAULTS if section == 'server' else LOCAL_PARAM_DEFAULTS
    section_data = _ensure_section(data, section, section_defaults)

    if last_trained is not None:
        section_data['last_trained'] = last_trained
    if images_used is not None:
        section_data['images_used'] = images_used
    if retraining is not None:
        section_data['retraining'] = retraining

    params = section_data.setdefault('training_params', {})

    if epochs is not None:
        params['epochs'] = epochs
    if section == 'server':
        if fine_tune_epochs is not None:
            params['fine_tune_epochs'] = fine_tune_epochs
        if learning_rate is not None:
            params['learning_rate'] = learning_rate
        if fine_tune_at is not None:
            params['fine_tune_at'] = fine_tune_at
    else:
        if learning_rate is not None:
            params['learning_rate'] = learning_rate
        if batch_size is not None:
            params['batch_size'] = batch_size
        if seed is not None:
            params['seed'] = seed
        if class_count is not None:
            params['class_count'] = class_count
        if max_samples_per_class is not None:
            params['max_samples_per_class'] = max_samples_per_class

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
    model_info = get_model_info()
    server_info = model_info.get('server', default_section(SERVER_PARAM_DEFAULTS))
    local_info = model_info.get('local', default_section(LOCAL_PARAM_DEFAULTS))

    current_dataset_images = count_current_dataset_images()
    class_names = ['not_cat', 'unknown_cat_entering', 'cat_morris_leaving', 'cat_morris_entering', 'prey']
    local_class_names = ['not_cat', 'not_prey', 'prey']

    learning_rates_server = ['5e-6', '6e-6', '7e-6', '8e-6', '9e-6', '1e-5', '2e-5', '3e-5', '4e-5', '5e-5']
    learning_rates_local = ['5e-4', '7.5e-4', '1e-3', '1.5e-3', '2e-3', '3e-3', '5e-3']
    batch_size_options = [16, 32, 48, 64]
    class_count_options = LOCAL_PARAM_LIMITS['class_count']
    max_samples_limits = LOCAL_PARAM_LIMITS['max_samples_per_class']

    active_models = get_active_models()

    # Load metrics for the currently active versions (if any)
    active_server_metrics = None
    if active_models.get('server'):
        details = get_model_version_details('server', active_models['server'])
        if details:
            active_server_metrics = details.get('metrics')

    active_local_metrics = None
    if active_models.get('local'):
        details = get_model_version_details('local', active_models['local'])
        if details:
            active_local_metrics = details.get('metrics')

    return render_template(
        'model.html',
        mode='model',
        current_dataset_images=current_dataset_images,
        retraining_status=retraining_status,
        local_retraining_status=local_retraining_status,
        server_info=server_info,
        local_info=local_info,
        class_names=class_names,
        local_class_names=local_class_names,
        learning_rates_server=learning_rates_server,
        learning_rates_local=learning_rates_local,
        batch_size_options=batch_size_options,
        class_count_options=class_count_options,
        local_class_default=LOCAL_PARAM_DEFAULTS['class_count'],
        max_samples_limits=max_samples_limits,
        max_samples_default=LOCAL_PARAM_DEFAULTS['max_samples_per_class'],
        active_server_metrics=active_server_metrics,
        active_local_metrics=active_local_metrics,
        server_versions=list_model_versions('server'),
        local_versions=list_model_versions('local'),
        active_models=active_models
    )


@app.route('/model/save/<scope>', methods=['POST'])
def save_current_model(scope):
    """API endpoint to manually save the current model as a version."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    # Check if model files exist
    if scope == 'server':
        model_check_path = os.path.join(MODEL_DIR, MODEL_NAME)
    else:
        model_check_path = os.path.join('simple_model', 'my_simple_model_quant.tflite')
    
    if not os.path.exists(model_check_path):
        return jsonify({'error': f'No {scope} model found to save'}), 404
    
    try:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = save_model_version(scope, timestamp_str)
        set_active_model(scope, timestamp_str)
        
        return jsonify({
            'success': True,
            'message': f'Saved version {timestamp_str}',
            'version': timestamp_str
        })
    except Exception as e:
        logging.error(f"Failed to save model version: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/versions/<scope>')
def get_model_versions(scope):
    """API endpoint to get list of model versions."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    versions = list_model_versions(scope)
    active = get_active_models()
    
    return jsonify({
        'versions': versions,
        'active': active.get(scope)
    })


@app.route('/model/version/<scope>/<version_name>')
def get_version_details(scope, version_name):
    """API endpoint to get full details of a specific version."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    details = get_model_version_details(scope, version_name)
    if not details:
        return jsonify({'error': 'Version not found'}), 404
    
    return jsonify(details)


@app.route('/model/activate/<scope>/<version_name>', methods=['POST'])
def activate_version(scope, version_name):
    """API endpoint to activate a specific model version."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    # Verify the version exists
    models_dir = SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR
    version_dir = os.path.join(models_dir, version_name)
    
    if not os.path.exists(version_dir):
        return jsonify({'error': 'Version not found'}), 404
    
    try:
        # Copy model files back to active location
        model_src_dir = os.path.join(version_dir, 'model')
        if scope == 'server':
            model_dest_dir = MODEL_DIR
        else:
            model_dest_dir = 'simple_model'
        
        if os.path.exists(model_src_dir):
            os.makedirs(model_dest_dir, exist_ok=True)
            for item in os.listdir(model_src_dir):
                src_item = os.path.join(model_src_dir, item)
                dest_item = os.path.join(model_dest_dir, item)
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dest_item)
        
        # For local scope, upload to ESP32 before marking active
        if scope == 'local':
            success, msg = upload_model_to_esp32(version_name)
            if not success:
                logging.error(f"Failed to upload local model {version_name} to ESP32; not activating. Reason: {msg}")
                return jsonify({'error': f'Upload to ESP32 failed; model not activated. Reason: {msg}'}), 500
        
        # Copy reports back to active location
        reports_src = os.path.join(version_dir, 'reports')
        if scope == 'server':
            reports_dest = os.path.join(app.static_folder, 'reports', 'server')
        else:
            reports_dest = os.path.join(app.static_folder, 'reports', 'local')
        
        if os.path.exists(reports_src):
            os.makedirs(reports_dest, exist_ok=True)
            # Clear destination and copy
            for item in os.listdir(reports_src):
                src_item = os.path.join(reports_src, item)
                dest_item = os.path.join(reports_dest, item)
                if os.path.isdir(src_item):
                    if os.path.exists(dest_item):
                        shutil.rmtree(dest_item)
                    shutil.copytree(src_item, dest_item)
                else:
                    shutil.copy2(src_item, dest_item)
        
        set_active_model(scope, version_name)
        logging.info(f"Activated {scope} model version: {version_name}")
        
        # Reload the model if it's the server model
        if scope == 'server':
            load_model()
        
        return jsonify({'success': True, 'message': f'Activated version {version_name}'})
    
    except Exception as e:
        logging.error(f"Failed to activate model version: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/delete/<scope>/<version_name>', methods=['POST'])
def delete_version(scope, version_name):
    """API endpoint to delete a model version."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    # Check if this is the active version
    active = get_active_models()
    if active.get(scope) == version_name:
        return jsonify({'error': 'Cannot delete the active model. Activate a different version first.'}), 400
    
    # Verify the version exists
    models_dir = SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR
    version_dir = os.path.join(models_dir, version_name)
    
    if not os.path.exists(version_dir):
        return jsonify({'error': 'Version not found'}), 404
    
    try:
        shutil.rmtree(version_dir)
        logging.info(f"Deleted {scope} model version: {version_name}")
        return jsonify({'success': True, 'message': f'Deleted version {version_name}'})
    except Exception as e:
        logging.error(f"Failed to delete model version: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/report/<scope>/<version_name>')
def view_model_report(scope, version_name):
    """Display full report page for a model version."""
    if scope not in ('server', 'local'):
        flash('Invalid scope.', 'danger')
        return redirect(url_for('model'))
    
    details = get_model_version_details(scope, version_name)
    if not details:
        flash('Version not found.', 'danger')
        return redirect(url_for('model'))
    
    # Check which report images exist
    version_dir = os.path.join(SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR, version_name)
    reports_dir = os.path.join(version_dir, 'reports')

    has_confusion = os.path.exists(os.path.join(reports_dir, 'images', 'confusion_matrix.png'))
    has_accuracy = os.path.exists(os.path.join(reports_dir, 'images', 'accuracy_plot.png'))
    has_loss = os.path.exists(os.path.join(reports_dir, 'images', 'loss_plot.png'))

    # Class weights are no longer stored as a separate file; use metrics/training_params if needed
    class_weights = None

    # Check if this is the active version
    active = get_active_models()
    is_active = active.get(scope) == version_name
    
    return render_template('report.html',
                           mode='model',
                           scope=scope,
                           version_name=version_name,
                           is_active=is_active,
                           metrics=details.get('metrics'),
                           model_summary=details.get('model_summary'),
                           has_confusion=has_confusion,
                           has_accuracy=has_accuracy,
                           has_loss=has_loss,
                           class_weights=class_weights)


@app.route('/models/<scope>/<version_name>/reports/images/<filename>')
def serve_version_image(scope, version_name, filename):
    """Serve report images from a model version directory."""
    if scope not in ('server', 'local'):
        return jsonify({'error': 'Invalid scope'}), 400
    
    # Only allow specific image files for security
    allowed_files = ['confusion_matrix.png', 'accuracy_plot.png', 'loss_plot.png']
    if filename not in allowed_files:
        return jsonify({'error': 'File not allowed'}), 403
    
    models_dir = SERVER_MODELS_DIR if scope == 'server' else LOCAL_MODELS_DIR
    images_dir = os.path.join(models_dir, version_name, 'reports', 'images')
    
    return send_from_directory(images_dir, filename)


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


@app.route('/local_retrain', methods=['POST'])
def retrain_local_model():
    with local_retrain_lock:
        if local_retraining_status['retraining']:
            flash('Local retraining is already in progress.', 'warning')
            return redirect(url_for('model'))

    epochs = request.form.get('local_epochs', default=LOCAL_PARAM_DEFAULTS['epochs'], type=int)
    learning_rate = request.form.get('local_learning_rate', default=LOCAL_PARAM_DEFAULTS['learning_rate'])
    batch_size = request.form.get('local_batch_size', default=LOCAL_PARAM_DEFAULTS['batch_size'], type=int)
    seed = request.form.get('local_seed', default=LOCAL_PARAM_DEFAULTS['seed'], type=int)
    class_count = request.form.get('local_class_count', default=LOCAL_PARAM_DEFAULTS['class_count'], type=int)
    max_samples_per_class = request.form.get(
        'local_max_samples_per_class',
        default=LOCAL_PARAM_DEFAULTS['max_samples_per_class'],
        type=int
    )

    # Validate epochs
    epoch_limits = LOCAL_PARAM_LIMITS['epochs']
    if epochs is None or epochs < epoch_limits['min'] or epochs > epoch_limits['max']:
        flash(f"Epochs must be between {epoch_limits['min']} and {epoch_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    # Validate learning rate
    if learning_rate not in LOCAL_PARAM_LIMITS['learning_rate']:
        flash('Invalid learning rate selected for local training.', 'danger')
        return redirect(url_for('model'))

    # Validate batch size
    if batch_size not in LOCAL_PARAM_LIMITS['batch_size']:
        flash('Invalid batch size selected for local training.', 'danger')
        return redirect(url_for('model'))

    # Validate seed
    seed_limits = LOCAL_PARAM_LIMITS['seed']
    if seed is None or seed < seed_limits['min'] or seed > seed_limits['max']:
        flash(f"Seed must be between {seed_limits['min']} and {seed_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    if class_count not in LOCAL_PARAM_LIMITS['class_count']:
        flash('Invalid class count selected for local training.', 'danger')
        return redirect(url_for('model'))

    sample_limits = LOCAL_PARAM_LIMITS['max_samples_per_class']
    if max_samples_per_class is None or max_samples_per_class < sample_limits['min'] or max_samples_per_class > sample_limits['max']:
        flash(f"Max samples per class must be between {sample_limits['min']} and {sample_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    retrain_thread = threading.Thread(
        target=run_local_retraining,
        args=(epochs, learning_rate, batch_size, seed, class_count, max_samples_per_class)
    )
    retrain_thread.start()

    flash('Local retraining started successfully!', 'success')
    return redirect(url_for('model'))


@app.route('/status')
def status():
    server_status = retraining_status.copy()
    local_status = local_retraining_status.copy()
    if retraining_status['completed']:
        retraining_status['completed'] = False
    if local_retraining_status['completed']:
        local_retraining_status['completed'] = False
    return jsonify({'server': server_status, 'local': local_status})


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
