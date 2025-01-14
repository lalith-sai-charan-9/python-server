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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get Redis URL from environment
REDIS_URL = os.getenv('REDIS_PUBLIC_URL', 'redis://localhost:6379/0')

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
    worker_prefetch_multiplier = 1
    task_acks_late = True

celery.config_from_object(CeleryConfig)

# Initialize the Celery app with Flask context
celery.conf.update(app.config)

class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

# Google Drive download URL (update this with the correct file ID if necessary)
MODEL_URL = "https://drive.google.com/uc?id=1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4"

# this is the original link : https://drive.google.com/file/d/1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4/view?usp=drive_link
# we need to remove the last part of the link to download the file, the new file link is : https://drive.google.com/uc?id=1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4

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
    """Download the model file from Google Drive if not already downloaded."""
    if not os.path.exists(MODEL_PATH):
        start_time = time.time()
        logger.info("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            end_time = time.time()
            logger.info(f"Model downloaded successfully in {end_time - start_time:.2f} seconds")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
    else:
        logger.info("Model already downloaded")

# Predefined genres (update according to your model's output mapping)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Set up allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}

def allowed_file(filename):
    """Check if the uploaded file is a valid audio file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            break

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = np.resize(mel_spectrogram, target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

@celery.task(bind=True, name='app.process_audio_file')
def process_audio_file(self, file_path):
    """Celery task for processing audio files"""
    try:
        self.update_state(state='PROCESSING')
        start_time = time.time()
        logger.info(f"Starting prediction task {self.request.id} for {file_path}")
        
        # Get preprocessed audio
        audio_features = preprocess_audio(file_path)
        
        # Get model prediction
        model = get_model()
        predictions = model.predict(audio_features)
        genre_index = np.argmax(np.sum(predictions, axis=0))
        
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        return {
            'status': 'completed',
            'genre': GENRES[genre_index],
            'processing_time': processing_time
        }
    except Exception as e:
        logger.error(f"Error in task {self.request.id}: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists(file_path):
            os.remove(file_path)
        self.update_state(state='FAILURE')
        raise

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    os.makedirs("uploads", exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join("uploads", filename)
                file.save(file_path)
                
                # Submit task to Celery
                task = process_audio_file.delay(file_path)
                logger.info(f"Created task {task.id} for file {filename}")
                
                results.append({
                    "file_name": filename,
                    "task_id": task.id,
                    "status": "processing"
                })
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                logger.error(traceback.format_exc())
                results.append({
                    "file_name": file.filename,
                    "error": str(e)
                })
        else:
            results.append({
                "file_name": file.filename,
                "error": "Invalid file type"
            })

    return jsonify(results), 202

@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """Check the status of a prediction task"""
    try:
        task = AsyncResult(task_id, app=celery)
        logger.info(f"Checking status for task {task_id}: {task.state}")
        
        if task.state == 'PENDING':
            response = {
                'status': 'processing',
                'current': 0,
                'total': 1,
            }
        elif task.state == 'FAILURE':
            response = {
                'status': 'error',
                'error': str(task.info),
            }
        elif task.state == 'SUCCESS':
            response = {
                'status': 'completed',
                'result': task.get()
            }
        else:
            response = {
                'status': 'processing',
                'state': task.state,
            }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error checking status for task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({"status": "healthy"}), 200

# Download and initialize model at startup
download_model()
get_model()  # Cache the model in memory

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)