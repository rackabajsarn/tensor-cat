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
from PIL import Image, ImageOps



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
    from pycoral.utils.edgetpu import make_interpreter  # type: ignore
    from pycoral.adapters.common import set_input  # type: ignore
    from pycoral.adapters.classify import get_classes  # type: ignore
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
ESP_INFERENCE_TOPIC = 'catflap/esp32_inference'

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
    "max_samples_per_class": 0,
    "width_mult": 0.75,
    "val_split": 0.2,
    "early_stop_patience": 5,
    "weight_decay": 1e-5,
    "dropout": 0.30,
    "augment": "light",
    "use_class_weights": True,
    "label_smoothing": 0.0,
    "lr_schedule": "cosine",
    "warmup_epochs": 0
}

# Valid parameter ranges for local model training
LOCAL_PARAM_LIMITS = {
    "epochs": {"min": 10, "max": 120, "step": 5},
    "learning_rate": ['5e-4', '7.5e-4', '1e-3', '1.5e-3', '2e-3', '3e-3', '5e-3'],
    "batch_size": [16, 32, 48, 64],
    "seed": {"min": 0, "max": 999999},
    "class_count": [2, 3],
    "max_samples_per_class": {"min": 0, "max": 1000},
    "width_mult": {"min": 0.4, "max": 1.0, "step": 0.05},
    "val_split": {"min": 0.05, "max": 0.4, "step": 0.05},
    "early_stop_patience": {"min": 1, "max": 20},
    "weight_decay": [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
    "dropout": {"min": 0.0, "max": 0.6, "step": 0.05},
    "augment": ["off", "light", "medium"],
    "use_class_weights": [True, False],
    "label_smoothing": {"min": 0.0, "max": 0.2, "step": 0.01},
    "lr_schedule": ["constant", "cosine", "step"],
    "warmup_epochs": {"min": 0, "max": 20}
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


def compute_simple_class(class_count, *, predicted_label=None, labels=None):
    """Return collapsed simple class ('prey', 'not_cat', or 'not_prey').

    Uses either a predicted_label string or label flags dict. class_count controls
    whether 'not_cat' is distinct (3-class) or merged into 'not_prey' (2-class).
    """
    if predicted_label is not None:
        if predicted_label == "prey":
            return "prey"
        if class_count == 3 and predicted_label == "not_cat":
            return "not_cat"
        return "not_prey"

    if isinstance(labels, dict):
        prey_flag = labels.get("prey")
        cat_flag = labels.get("cat")

        if prey_flag:
            return "prey"
        if class_count == 3 and cat_flag is False:
            return "not_cat"
        return "not_prey"

    return None


def _ordered_classes(labels_set):
    """Return stable class ordering for confusion matrices (supports 2 or 3 classes)."""
    preferred = ["prey", "not_cat", "not_prey"]
    ordered = [lbl for lbl in preferred if lbl in labels_set]
    for lbl in sorted(labels_set):
        if lbl not in ordered:
            ordered.append(lbl)
    return ordered


def compute_empirical_metrics(model_name=None, scope='server'):
    """Compute accumulated accuracy/confusion from inference_log.

    scope controls which model column to filter on: 'server' uses server_model with
    server_simple_inference; 'local' uses esp32_model with esp32_inference. Only rows
    that have both a true_label and the corresponding prediction are counted. Handles
    both 2-class and 3-class layouts by deriving the label set from the data.
    """
    model_column = 'server_model' if scope == 'server' else 'esp32_model'
    pred_column = 'server_simple_inference' if scope == 'server' else 'esp32_inference'
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            params = []
            where = f"true_label IS NOT NULL AND {pred_column} IS NOT NULL"
            if model_name:
                where += f" AND {model_column} = ?"
                params.append(model_name)
            cursor.execute(
                f"SELECT true_label, {pred_column} FROM inference_log WHERE {where}",
                params
            )
            rows = cursor.fetchall()
    except Exception as e:
        logging.error(f"Failed to compute empirical metrics for {model_name}: {e}")
        return None

    if not rows:
        return None

    labels_set = set()
    for true_lbl, pred_lbl in rows:
        if true_lbl:
            labels_set.add(true_lbl)
        if pred_lbl:
            labels_set.add(pred_lbl)

    classes = _ordered_classes(labels_set)
    idx = {lbl: i for i, lbl in enumerate(classes)}
    size = len(classes)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    total = 0
    correct = 0

    for true_lbl, pred_lbl in rows:
        if true_lbl is None or pred_lbl is None:
            continue
        i = idx.get(true_lbl)
        j = idx.get(pred_lbl)
        if i is None or j is None:
            continue
        matrix[i][j] += 1
        total += 1
        if true_lbl == pred_lbl:
            correct += 1

    if total == 0:
        return None

    accuracy = correct / total if total else None
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'classes': classes,
        'confusion_matrix': matrix,
    }


def get_inference_by_hash(hash_hex):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """
                SELECT esp32_inference, esp32_confidence, server_simple_inference, server_confidence
                FROM inference_log
                WHERE hash = ?
                """,
                [hash_hex]
            ).fetchone()
    except Exception as e:
        logging.error(f"Failed to fetch inference row for {hash_hex}: {e}")
        return None

    if not row:
        return None

    esp_inf, esp_conf, server_inf, server_conf = row
    return {
        'esp32_inference': esp_inf,
        'esp32_confidence': esp_conf,
        'server_inference': server_inf,
        'server_confidence': server_conf,
    }


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
        # Trainers write into models/<scope>/<version>/{model,reports}. Some flows
        # (notably local simple training) may not create metadata.json, so create it
        # opportunistically from reports/metrics.json to keep the UI consistent.
        if not os.path.exists(metadata_path):
            try:
                save_model_version(scope, name)
            except Exception as e:
                logging.error(f"Error generating metadata for {scope} version {name}: {e}")

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
                    timestamp_server TEXT,
                    timestamp_esp TEXT,
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


def upsert_inference_record(hash_hex, *, timestamp_server=None, timestamp_esp=None, esp32_model=None,
                            server_model=None, esp32_inference=None, server_inference=None,
                            server_simple_inference=None, true_label=None, esp32_confidence=None,
                            server_confidence=None):
    try:
        with db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO inference_log (
                    hash, timestamp_server, timestamp_esp, esp32_model, server_model,
                    esp32_inference, server_inference, server_simple_inference, true_label,
                    esp32_confidence, server_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    timestamp_server=COALESCE(excluded.timestamp_server, inference_log.timestamp_server),
                    timestamp_esp=COALESCE(excluded.timestamp_esp, inference_log.timestamp_esp),
                    esp32_model=COALESCE(excluded.esp32_model, inference_log.esp32_model),
                    server_model=COALESCE(excluded.server_model, inference_log.server_model),
                    esp32_inference=COALESCE(excluded.esp32_inference, inference_log.esp32_inference),
                    server_inference=COALESCE(excluded.server_inference, inference_log.server_inference),
                    server_simple_inference=COALESCE(excluded.server_simple_inference, inference_log.server_simple_inference),
                    true_label=COALESCE(excluded.true_label, inference_log.true_label),
                    esp32_confidence=COALESCE(excluded.esp32_confidence, inference_log.esp32_confidence),
                    server_confidence=COALESCE(excluded.server_confidence, inference_log.server_confidence);
                """,
                 [hash_hex, timestamp_server, timestamp_esp, esp32_model, server_model,
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
        client.subscribe(ESP_INFERENCE_TOPIC)
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
        # Handle ESP32 inference results published as JSON
        if msg.topic == ESP_INFERENCE_TOPIC:
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
            except Exception as json_err:
                logging.error(f"Failed to parse ESP32 inference payload: {json_err}")
                return

            hash_hex = payload.get('hash')
            esp_label = payload.get('label')
            esp_conf = payload.get('confidence')
            esp_model = payload.get('model')

            if not hash_hex or not esp_label:
                logging.error(f"ESP32 inference payload missing hash/label: {payload}")
                return

            timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'
            upsert_inference_record(
                hash_hex,
                timestamp_esp=timestamp_iso,
                esp32_model=esp_model,
                esp32_inference=esp_label,
                esp32_confidence=esp_conf,
            )
            return

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
        server_simple = compute_simple_class(local_class_count, predicted_label=predicted_label)
        server_model_name = active_models.get('server') if isinstance(active_models, dict) else None
        timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'
        upsert_inference_record(
            img_hash,
            timestamp_server=timestamp_iso,
            server_model=server_model_name,
            server_inference=predicted_label,
            server_simple_inference=server_simple,
            server_confidence=float(server_confidence) if server_confidence is not None else None,
        )
        logging.info(
            "SERVER upsert hash=%s label=%s simple=%s conf=%s model=%s ts=%s",
            img_hash,
            predicted_label,
            server_simple,
            server_confidence,
            server_model_name,
            timestamp_iso,
        )

        server_payload = json.dumps({
            "hash": img_hash,
            "label": server_simple,
            "confidence": float(server_confidence) if server_confidence is not None else None,
            "model": server_model_name or ""
        })
        client.publish('catflap/server_inference', server_payload)
        
        # Write labels to EXIF and ensure hash persists for later true-label logging
        write_labels(image_path, labels)
        write_imghash(image_path, img_hash)
        
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

def write_imghash(image_path, img_hash):
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

        # Ensure Exif block exists, then store the hash in UserComment (Exif IFD)
        exif_dict.setdefault('Exif', {})
        exiftag = img_hash
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = exiftag.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, "jpeg", exif=exif_bytes)
        return True
    except Exception as e:
        print(f"Error writing hash to {image_path}: {e}")
        return False


def read_imghash(image_path):
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        raw = exif_dict.get('Exif', {}).get(piexif.ExifIFD.UserComment)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            try:
                return raw.decode('utf-8')
            except Exception:
                return raw.decode(errors='ignore')
        return str(raw)
    except Exception as e:
        logging.error(f"Error reading hash from {image_path}: {e}")
        return None

def classify_image(image_path):
    try:
        # Preprocess the image
        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Resolution-dependent center crop to match training/data capture:
        # - legacy 640x480 -> 384x384 center crop
        # - new 320x240    -> 192x192 center crop
        w, h = image.size
        min_dim = min(w, h)
        if min_dim >= 480:
            crop_size = 384
        elif min_dim >= 240:
            crop_size = 192
        else:
            crop_size = min_dim

        crop_size = min(crop_size, w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        
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

        # Prefer the provided version; otherwise use the active local version
        active = get_active_models()
        local_version = version_name or (active.get('local') if isinstance(active, dict) else None)

        if not local_version:
            msg = "No local model version is active; cannot upload to ESP32"
            logging.error(msg)
            return False, msg

        model_path = os.path.join(LOCAL_MODELS_DIR, local_version, 'model', 'my_simple_model_quant.tflite')
        if not os.path.exists(model_path):
            msg = f"Simple model file not found at {model_path}"
            logging.error(msg)
            return False, msg
        
        # ESP32 IP address - should be configurable
        esp32_ip = credentials.ESP32_IP if hasattr(credentials, 'ESP32_IP') else '192.168.1.14'
        upload_url = f'http://{esp32_ip}/upload_model'
        metadata_url = f'http://{esp32_ip}/upload_metadata'
        
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

            # Build minimal metadata JSON expected by ESP32
            metadata_payload = {
                "model_name": str(local_version),
                "number_of_labels": int(get_active_local_class_count()),
                "threshold_value": 0.5,
            }

            metrics_path = os.path.join(LOCAL_MODELS_DIR, local_version, 'reports', 'metrics.json')
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as mf:
                        metrics_obj = json.load(mf)

                    training_params = metrics_obj.get('training_params', {}) if isinstance(metrics_obj, dict) else {}
                    classes_list = metrics_obj.get('classes') if isinstance(metrics_obj, dict) else None

                    class_count_val = training_params.get('class_count') if isinstance(training_params, dict) else None
                    if not isinstance(class_count_val, int) and isinstance(classes_list, list):
                        class_count_val = len(classes_list)
                    if isinstance(class_count_val, int) and class_count_val > 0:
                        metadata_payload['number_of_labels'] = class_count_val

                    # If the local model exports logits margin (single output), adapt ESP32 metadata.
                    export_output = training_params.get('export_output') if isinstance(training_params, dict) else None
                    if export_output == 'logits_margin':
                        metadata_payload['number_of_labels'] = 1
                        margin_thr = training_params.get('prey_logit_margin_threshold')
                        try:
                            if margin_thr is not None:
                                metadata_payload['threshold_value'] = float(margin_thr)
                        except (TypeError, ValueError):
                            pass

                    thr_val = training_params.get('prey_threshold') if isinstance(training_params, dict) else None
                    try:
                        if thr_val is not None:
                            # For probability-output models, threshold_value is the prey probability.
                            if metadata_payload.get('number_of_labels') != 1:
                                metadata_payload['threshold_value'] = float(thr_val)
                    except (TypeError, ValueError):
                        pass

            except Exception as meta_err:
                logging.error(f"Failed to build metadata for ESP32 upload: {meta_err}")

            try:
                metadata_bytes = json.dumps(metadata_payload).encode('utf-8')
                metadata_files = {'file': ('metadata.json', metadata_bytes, 'application/json')}
                logging.info(f"Uploading metadata to ESP32 at {metadata_url}...")
                meta_response = requests.post(metadata_url, files=metadata_files, timeout=15)
                if meta_response.status_code == 200:
                    meta_msg = "Metadata uploaded successfully to ESP32"
                    logging.info(meta_msg)
                    retraining_status['output'] += "Metadata uploaded to ESP32 successfully!\n"
                else:
                    error_msg = f"Failed to upload metadata to ESP32: {meta_response.status_code} - {meta_response.text}"
                    logging.error(error_msg)
                    retraining_status['output'] += f"\n{error_msg}\n"
                    return False, error_msg
            except Exception as meta_err:
                error_msg = f"Error uploading metadata to ESP32: {meta_err}"
                logging.error(error_msg)
                retraining_status['output'] += f"\n{error_msg}\n"
                return False, error_msg

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
            '-u',
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
            stderr=subprocess.STDOUT,
            text=True,  # To capture output as string
            bufsize=1
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


def run_local_retraining(
    epochs,
    learning_rate,
    batch_size,
    seed,
    class_count,
    max_samples_per_class,
    width_mult,
    val_split,
    early_stop_patience,
    weight_decay,
    dropout,
    augment,
    use_class_weights,
    label_smoothing,
    lr_schedule,
    warmup_epochs
):
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
            '-u',
            train_simple_script_path,
            '--epochs', str(epochs),
            '--learning_rate', str(learning_rate),
            '--batch_size', str(batch_size),
            '--seed', str(seed),
            '--class_count', str(class_count),
            '--run_id', run_id,
            '--width_mult', str(width_mult),
            '--val_split', str(val_split),
            '--early_stop_patience', str(early_stop_patience),
            '--weight_decay', str(weight_decay),
            '--dropout', str(dropout),
            '--augment', str(augment),
            '--use_class_weights', 'on' if use_class_weights else 'off',
            '--label_smoothing', str(label_smoothing),
            '--lr_schedule', str(lr_schedule),
            '--warmup_epochs', str(warmup_epochs)
        ]

        if max_samples_per_class > 0:
            command.extend(['--max_samples_per_class', str(max_samples_per_class)])

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
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
            max_samples_per_class=max_samples_per_class,
            width_mult=width_mult,
            val_split=val_split,
            early_stop_patience=early_stop_patience,
            weight_decay=weight_decay,
            dropout=dropout,
            augment=augment,
            use_class_weights=use_class_weights,
            label_smoothing=label_smoothing,
            lr_schedule=lr_schedule,
            warmup_epochs=warmup_epochs
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
                      batch_size=None, seed=None, class_count=None, max_samples_per_class=None,
                      width_mult=None, val_split=None, early_stop_patience=None, weight_decay=None, dropout=None,
                      augment=None, use_class_weights=None, label_smoothing=None, lr_schedule=None,
                      warmup_epochs=None):
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
        if width_mult is not None:
            params['width_mult'] = width_mult
        if val_split is not None:
            params['val_split'] = val_split
        if early_stop_patience is not None:
            params['early_stop_patience'] = early_stop_patience
        if weight_decay is not None:
            params['weight_decay'] = weight_decay
        if dropout is not None:
            params['dropout'] = dropout
        if augment is not None:
            params['augment'] = augment
        if use_class_weights is not None:
            params['use_class_weights'] = bool(use_class_weights)
        if label_smoothing is not None:
            params['label_smoothing'] = label_smoothing
        if lr_schedule is not None:
            params['lr_schedule'] = lr_schedule
        if warmup_epochs is not None:
            params['warmup_epochs'] = warmup_epochs

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

            # Before moving, persist true_label using current labels and hash
            labels = read_labels(image_path)
            img_hash = read_imghash(image_path)
            if img_hash:
                class_count = get_active_local_class_count()
                simple_true = compute_simple_class(class_count, labels=labels)
                try:
                    upsert_inference_record(
                        img_hash,
                        true_label=simple_true,
                    )
                except Exception as log_err:
                    logging.error(f"Failed to upsert true_label for {img_hash}: {log_err}")
            else:
                logging.warning(f"No EXIF hash found for {image_path}; skipping true_label upsert")

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
        inference = None
        img_hash = read_imghash(image_path)
        if img_hash:
            inference = get_inference_by_hash(img_hash)
        return jsonify({'success': True, 'labels': labels, 'inference': inference})

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
    val_split_options = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    weight_decay_options = LOCAL_PARAM_LIMITS['weight_decay']
    augment_options = LOCAL_PARAM_LIMITS['augment']
    lr_schedule_options = LOCAL_PARAM_LIMITS['lr_schedule']

    active_models = get_active_models()

    # Load metrics for the currently active versions (if any)
    active_server_metrics = None
    if active_models.get('server'):
        details = get_model_version_details('server', active_models['server'])
        if details:
            active_server_metrics = details.get('metrics')

    active_server_empirical = None
    if active_models.get('server'):
        active_server_empirical = compute_empirical_metrics(active_models['server'], scope='server')

    active_local_metrics = None
    if active_models.get('local'):
        details = get_model_version_details('local', active_models['local'])
        if details:
            active_local_metrics = details.get('metrics')

    active_local_empirical = None
    if active_models.get('local'):
        active_local_empirical = compute_empirical_metrics(active_models['local'], scope='local')

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
        val_split_options=val_split_options,
        weight_decay_options=weight_decay_options,
        augment_options=augment_options,
        lr_schedule_options=lr_schedule_options,
        active_server_metrics=active_server_metrics,
        active_local_metrics=active_local_metrics,
        active_server_empirical=active_server_empirical,
        active_local_empirical=active_local_empirical,
        server_versions=[
            {
                **version,
                'empirical': compute_empirical_metrics(version['name'], scope='server')
            } for version in list_model_versions('server')
        ],
        local_versions=[
            {
                **version,
                'empirical': compute_empirical_metrics(version['name'], scope='local')
            } for version in list_model_versions('local')
        ],
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
        if not os.path.exists(model_check_path):
            return jsonify({'error': f'No {scope} model found to save'}), 404
    else:
        active = get_active_models()
        active_local = active.get('local') if isinstance(active, dict) else None
        if not active_local:
            return jsonify({'error': 'No active local model to save'}), 404

        model_check_path = os.path.join(LOCAL_MODELS_DIR, active_local, 'model', 'my_simple_model_quant.tflite')
        if not os.path.exists(model_check_path):
            return jsonify({'error': f'Active local model files missing at {model_check_path}'}), 404
    
    try:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if scope == 'local':
            # Snapshot the currently active local version into a new version directory
            source_dir = os.path.join(LOCAL_MODELS_DIR, active_local)
            dest_dir = os.path.join(LOCAL_MODELS_DIR, timestamp_str)
            shutil.copytree(source_dir, dest_dir)

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
        # Ensure model artifacts exist
        model_src_dir = os.path.join(version_dir, 'model')
        if not os.path.exists(model_src_dir):
            return jsonify({'error': f'Model artifacts missing for {version_name}'}), 404

        # For server scope, copy model files back to active location for TPU inference
        if scope == 'server':
            os.makedirs(MODEL_DIR, exist_ok=True)
            for item in os.listdir(model_src_dir):
                src_item = os.path.join(model_src_dir, item)
                dest_item = os.path.join(MODEL_DIR, item)
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
    
    empirical_metrics = compute_empirical_metrics(version_name, scope=scope)

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
                           class_weights=class_weights,
                           empirical_metrics=empirical_metrics)

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

        # Persist chosen params immediately so the UI reflects the submitted values
        # even while retraining is running.
        update_model_info(
            section='server',
            retraining=True,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            learning_rate=learning_rate_str,
            fine_tune_at=fine_tune_at,
        )
        
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
    width_mult = request.form.get('local_width_mult', default=LOCAL_PARAM_DEFAULTS['width_mult'], type=float)
    val_split = request.form.get('local_val_split', default=LOCAL_PARAM_DEFAULTS['val_split'], type=float)
    early_stop_patience = request.form.get('local_early_stop_patience', default=LOCAL_PARAM_DEFAULTS['early_stop_patience'], type=int)
    weight_decay = request.form.get('local_weight_decay', default=LOCAL_PARAM_DEFAULTS['weight_decay'], type=float)
    dropout = request.form.get('local_dropout', default=LOCAL_PARAM_DEFAULTS['dropout'], type=float)
    augment = request.form.get('local_augment', default=LOCAL_PARAM_DEFAULTS['augment'])
    use_class_weights_raw = request.form.get('local_use_class_weights', default='on')
    use_class_weights = str(use_class_weights_raw).lower() != 'off'
    label_smoothing = request.form.get('local_label_smoothing', default=LOCAL_PARAM_DEFAULTS['label_smoothing'], type=float)
    lr_schedule = request.form.get('local_lr_schedule', default=LOCAL_PARAM_DEFAULTS['lr_schedule'])
    warmup_epochs = request.form.get('local_warmup_epochs', default=LOCAL_PARAM_DEFAULTS['warmup_epochs'], type=int)

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

    width_mult_limits = LOCAL_PARAM_LIMITS['width_mult']
    if width_mult is None or width_mult < width_mult_limits['min'] or width_mult > width_mult_limits['max']:
        flash(f"Width multiplier must be between {width_mult_limits['min']} and {width_mult_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    val_limits = LOCAL_PARAM_LIMITS['val_split']
    if val_split is None or val_split < val_limits['min'] or val_split > val_limits['max']:
        flash(f"Validation split must be between {val_limits['min']} and {val_limits['max']} (fraction).", 'danger')
        return redirect(url_for('model'))

    patience_limits = LOCAL_PARAM_LIMITS['early_stop_patience']
    if early_stop_patience is None or early_stop_patience < patience_limits['min'] or early_stop_patience > patience_limits['max']:
        flash(f"Early-stop patience must be between {patience_limits['min']} and {patience_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    if weight_decay not in LOCAL_PARAM_LIMITS['weight_decay']:
        flash('Invalid weight decay selected for local training.', 'danger')
        return redirect(url_for('model'))

    dropout_limits = LOCAL_PARAM_LIMITS['dropout']
    if dropout is None or dropout < dropout_limits['min'] or dropout > dropout_limits['max']:
        flash(f"Dropout must be between {dropout_limits['min']} and {dropout_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    if augment not in LOCAL_PARAM_LIMITS['augment']:
        flash('Invalid augmentation option selected.', 'danger')
        return redirect(url_for('model'))

    label_smoothing_limits = LOCAL_PARAM_LIMITS['label_smoothing']
    if label_smoothing is None or label_smoothing < label_smoothing_limits['min'] or label_smoothing > label_smoothing_limits['max']:
        flash(f"Label smoothing must be between {label_smoothing_limits['min']} and {label_smoothing_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    if lr_schedule not in LOCAL_PARAM_LIMITS['lr_schedule']:
        flash('Invalid learning rate schedule option.', 'danger')
        return redirect(url_for('model'))

    warmup_limits = LOCAL_PARAM_LIMITS['warmup_epochs']
    if warmup_epochs is None or warmup_epochs < warmup_limits['min'] or warmup_epochs > warmup_limits['max']:
        flash(f"Warmup epochs must be between {warmup_limits['min']} and {warmup_limits['max']}.", 'danger')
        return redirect(url_for('model'))

    retrain_thread = threading.Thread(
        target=run_local_retraining,
        args=(
            epochs,
            learning_rate,
            batch_size,
            seed,
            class_count,
            max_samples_per_class,
            width_mult,
            val_split,
            early_stop_patience,
            weight_decay,
            dropout,
            augment,
            use_class_weights,
            label_smoothing,
            lr_schedule,
            warmup_epochs
        )
    )

    # Persist chosen params immediately so the UI reflects the submitted values
    # even while retraining is running.
    update_model_info(
        section='local',
        retraining=True,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        class_count=class_count,
        max_samples_per_class=max_samples_per_class,
        width_mult=width_mult,
        val_split=val_split,
        early_stop_patience=early_stop_patience,
        weight_decay=weight_decay,
        dropout=dropout,
        augment=augment,
        use_class_weights=use_class_weights,
        label_smoothing=label_smoothing,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
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
