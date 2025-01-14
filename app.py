import redis
import traceback
from celery.result import AsyncResult
from celery import Celery
import time
import requests
from werkzeug.utils import secure_filename
import logging
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage


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
            logger.info(
                f"Model downloaded successfully in {end_time - start_time:.2f} seconds")
        else:
            raise Exception(
                f"Failed to download model. Status code: {response.status_code}")
    else:
        logger.info("Model already downloaded")


# Predefined genres (update according to your model's output mapping)
GENRES = ['blues', 'classical', 'country', 'disco',
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Set up allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}


def allowed_file(filename):
    """Check if the uploaded file is a valid audio file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio(file_data, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(io.BytesIO(file_data), sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) /
                     (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            break

        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk, sr=sample_rate)
        mel_spectrogram = np.resize(mel_spectrogram, target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


@celery.task(bind=True)
def process_audio_file(self, filename, file_data):
    """Process an audio file and return genre predictions."""
    try:
        logger.info(f"Starting to process audio file: {filename}")
        logger.info(f"Task ID: {self.request.id}")

        # Save file data to temporary file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_data)

        try:
            # Load and preprocess the audio file
            features = preprocess_audio(temp_path)

            if features is None:
                return {"error": "Failed to process audio file"}

            # Get model predictions
            model = get_model()
            predictions = model.predict(np.expand_dims(features, axis=0))

            # Get the predicted genre
            predicted_index = np.argmax(predictions[0])
            predicted_genre = GENRES[predicted_index]
            confidence = float(predictions[0][predicted_index])

            # Create result dictionary
            result = {
                "genre": predicted_genre,
                "confidence": confidence,
                "predictions": {
                    genre: float(pred)
                    for genre, pred in zip(GENRES, predictions[0])
                }
            }

            return result

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")

        if 'files' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        files = request.files.getlist('files')
        results = []

        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")

                # Read file data
                file_data = file.read()

                # Create task
                task = process_audio_file.delay(filename, file_data)

                results.append({
                    "file_name": filename,
                    "status": "processing",
                    "task_id": task.id
                })
            else:
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "error": "Invalid file type"
                })

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


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
