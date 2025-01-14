import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage

from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
import logging
from werkzeug.utils import secure_filename
import requests
import time
from celery import Celery
from celery.result import AsyncResult
import traceback
import redis
import io  # Import the io module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get Redis URL from environment
REDIS_URL = os.getenv('REDIS_PUBLIC_URL', 'redis://localhost:6379/0')

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL)

# Configure Celery with the Flask app
celery = Celery(
    'app',
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_connection_retry_on_startup=True
)

# Celery configuration
class CeleryConfig:
    broker_url = REDIS_URL
    result_backend = REDIS_URL
    task_track_started = True
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    broker_connection_retry_on_startup = True
    broker_connection_max_retries = 10
    worker_max_tasks_per_child = 1
    worker_prefetch_multiplier = 1
    task_acks_late = True
    task_reject_on_worker_lost = True
    task_time_limit = 600
    worker_concurrency = 1

celery.config_from_object(CeleryConfig)

# Initialize the Celery app with Flask context
celery.conf.update(app.config)

class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

# Google Drive download URL
MODEL_URL = "https://drive.google.com/uc?id=1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4"

# Ensure the model file is downloaded
MODEL_PATH = "Trained_model.h5"

_model = None

def get_model():
    """Cache and return the model"""
    global _model
    if _model is None:
        start_time = time.time()
        logger.info("Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        end_time = time.time()
        logger.info(f"Model loaded in {end_time - start_time:.2f} seconds")
    return _model

def download_model():
    # ... (Your existing download_model function)
    pass  # Placeholder, include your actual code here

# Predefined genres
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path, target_shape=(150, 150)):
    # ... (Your existing preprocess_audio function)
    pass  # Placeholder, include your actual code here

@celery.task(bind=True)
def process_audio_file(self, filename, file_data):
    """Process an audio file and return genre predictions."""
    try:
        # ... (Your existing process_audio_file function)
        pass  # Placeholder, include your actual code here

    except Exception as e:
        # ... (Your existing exception handling)
        pass  # Placeholder, include your actual code here

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ... (Your existing predict function)
        pass  # Placeholder, include your actual code here

    except Exception as e:
        # ... (Your existing exception handling)
        pass  # Placeholder, include your actual code here

@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """Check the status of a prediction task"""
    try:
        # ... (Your existing get_status function)
        pass  # Placeholder, include your actual code here

    except Exception as e:
        # ... (Your existing exception handling)
        pass  # Placeholder, include your actual code here

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Download and initialize model at startup
download_model()
get_model()  # Cache the model in memory

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
