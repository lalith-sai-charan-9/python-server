# Music Genre Prediction API

This API uses a pre-trained deep learning model to predict the genre of an audio file based on its content. It processes uploaded audio files and returns the predicted genre.

## How It Works

1. **Model Download**: Upon the first startup, the model is downloaded from Google Drive if not already present. The model is a deep learning model trained on audio data to classify music genres.
   
2. **Audio Preprocessing**: The uploaded audio file is processed into chunks, and each chunk is converted into a mel spectrogram, which is then fed into the model for prediction.

3. **Genre Prediction**: The model predicts the genre for each chunk of audio. The genre with the highest sum of predictions across chunks is returned as the overall genre.

4. **Allowed File Formats**: The API supports the following audio formats:
   - `.mp3`
   - `.wav`
   - `.flac`

## API Endpoint

### POST `/predict`

This endpoint processes one or more audio files and returns the predicted genre(s).

#### Request

- **Method**: `POST`
- **URL**: `http://<your-server-ip>:5000/predict`
- **Headers**: None
- **Body**: 
  - The request must be a `multipart/form-data` containing one or more audio files under the key `files`.

#### Example Request (Single File)

```bash
curl -X POST -F "files=@path/to/your/audio/file.mp3" http://<your-server-ip>:5000/predict
```

#### Example Request (Multiple Files)

```bash
curl -X POST -F "files=@path/to/your/audio/file1.mp3" -F "files=@path/to/your/audio/file2.mp3" http://<your-server-ip>:5000/predict
```

#### Response

The response will be a JSON array where each entry corresponds to a processed file. Each entry contains:
- `file_name`: The name of the uploaded file.
- `genre`: The predicted genre of the file (for valid files).
- `error`: If an error occurred during processing (e.g., invalid file type or processing failure).

##### Example Response (Single File)

```json
[
  {
    "file_name": "file.mp3",
    "genre": "rock"
  }
]
```

##### Example Response (Multiple Files)

```json
[
  {
    "file_name": "file1.mp3",
    "genre": "pop"
  },
  {
    "file_name": "file2.mp3",
    "genre": "classical"
  }
]
```

If there are any errors, such as invalid file type, the response will include an error message in the `error` field:

```json
[
  {
    "file_name": "file.mp3",
    "error": "Invalid file type"
  }
]
```

## How to Run the Server Locally

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.x installed. Then install the required Python libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask Application**:
   Run the Flask app with:
   ```bash
   python app.py
   ```
   The app will run on `http://127.0.0.1:5000/` by default.

## Notes

- Ensure that the model file (`Trained_model.h5`) is downloaded before starting the server. The app will attempt to download the model automatically if it doesn't already exist.
- The API is designed to process multiple files in one request. Each file will be processed independently, and the predicted genre for each file will be included in the response.
- Clean-up: After processing the files, the uploaded files are deleted from the server to save storage space.

## Troubleshooting

- **No files provided**: If no files are sent with the request, the API will return a `400` status code with the error message `"No files provided"`.
- **Invalid file type**: If an unsupported file type is uploaded, the API will return a `400` status code with the error message `"Invalid file type"`.
- **Model download failure**: If the model fails to download from Google Drive, the server will log the error and raise an exception.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Explanation:
1. **How It Works**: Explains the steps involved in downloading the model, processing the audio files, and predicting the genres.
2. **API Endpoint**: Provides details on how to use the `POST /predict` endpoint.
3. **Request and Response Examples**: Includes cURL examples for uploading one or multiple files, along with possible response formats.
4. **How to Run Locally**: Describes the steps to run the Flask app on your local machine, including dependency installation and app startup.
5. **Troubleshooting**: Addresses common issues and how they are handled by the API.

This documentation will help users understand how to interact with your API and what to expect from it.