from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)

# Set up logging for better monitoring
logging.basicConfig(level=logging.INFO)

# Google Drive download URL (update this with the correct file ID if necessary)
MODEL_URL = "https://drive.google.com/uc?id=1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4"

# this is the original link : https://drive.google.com/file/d/1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4/view?usp=drive_link
# we need to remove the last part of the link to download the file, the new file link is : https://drive.google.com/uc?id=1iBu-jwUSmNSZaWzQcqhOiCXlU6zNLJu4



# Ensure the model file is downloaded
MODEL_PATH = "Trained_model.h5"

def download_model():
    """Download the model file from Google Drive if not already downloaded."""
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

# Download the model and load it
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

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

def predict_genre(file_path):
    audio_features = preprocess_audio(file_path)
    predictions = model.predict(audio_features)
    genre_index = np.argmax(np.sum(predictions, axis=0))
    return GENRES[genre_index]

@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    result = []

    # Create a temporary directory to save uploaded files
    os.makedirs("uploads", exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            try:
                genre = predict_genre(file_path)
                result.append({"file_name": filename, "genre": genre})
                os.remove(file_path)  # Clean up the uploaded file after processing
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                result.append({"file_name": filename, "error": str(e)})
        else:
            result.append({"file_name": file.filename, "error": "Invalid file type"})

    # After processing all files, delete all files in the uploads directory
    for file in os.listdir("uploads"):
        file_path = os.path.join("uploads", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return jsonify(result), 200

if __name__ == "__main__":
    # Use host='0.0.0.0' for production and set debug=False
    app.run(debug=False, host="0.0.0.0", port=5000)
