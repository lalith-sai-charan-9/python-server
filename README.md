# Music Genre Classification API

This API service uses deep learning to classify music genres from audio files. It processes audio files asynchronously and returns the predicted genre.

## Supported Genres

- Blues
- Classical
- Country
- Disco
- Hip Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Requirements

- Python 3.10 or 3.11
- Redis Server
- Virtual Environment (recommended)

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and Start Redis Server**
   - Windows:
     1. Download Redis for Windows from [https://github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
     2. Install the MSI package
     3. Redis will run automatically as a Windows service
   
   - Linux:
     ```bash
     sudo apt-get update
     sudo apt-get install redis-server
     sudo systemctl start redis
     ```

## Deployment

1. **Start Redis Server** (if not running as a service)
   ```bash
   redis-server
   ```

2. **Start Celery Worker**
   ```bash
   # Activate virtual environment if not already activated
   celery -A celery_worker.celery worker --pool=solo -l info
   ```

3. **Start Flask Application**
   ```bash
   python app.py
   ```

## API Documentation

### 1. Predict Genre

**Endpoint:** `/predict`
**Method:** POST
**Content-Type:** multipart/form-data

**Request:**
- Form parameter: `files` (accepts multiple audio files)
- Supported formats: .mp3, .wav, .flac

```bash
curl -X POST -F "files=@path/to/your/audio.mp3" http://your-domain:5000/predict
```

**Response:**
```json
[
    {
        "file_name": "audio.mp3",
        "task_id": "task-uuid",
        "status": "processing"
    }
]
```

### 2. Check Prediction Status

**Endpoint:** `/status/<task_id>`
**Method:** GET

```bash
curl http://your-domain:5000/status/<task_id>
```

**Response:**
```json
{
    "status": "completed",
    "result": {
        "genre": "jazz",
        "processing_time": 2.5,
        "status": "completed"
    }
}
```

**Possible Status Values:**
- `processing`: Task is still running
- `completed`: Task finished successfully
- `error`: Task failed with an error

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful request
- 202: Request accepted (for async processing)
- 400: Bad request (invalid file type, no file provided)
- 500: Server error

## Production Deployment

For production deployment:

1. **Update Security Settings**
   - Set proper CORS headers
   - Use HTTPS
   - Set up proper authentication if needed

2. **Environment Variables**
   - Set `PORT` for custom port (default: 5000)
   - Set `REDIS_URL` for custom Redis configuration

3. **Supervisor Configuration (Linux)**
   Create `/etc/supervisor/conf.d/music-genre-api.conf`:
   ```ini
   [program:celery]
   command=/path/to/venv/bin/celery -A celery_worker.celery worker --pool=solo -l info
   directory=/path/to/project
   user=your_user
   autostart=true
   autorestart=true
   
   [program:flask]
   command=/path/to/venv/bin/python app.py
   directory=/path/to/project
   user=your_user
   autostart=true
   autorestart=true
   ```

4. **Nginx Configuration (Optional)**
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Monitoring

- Check Celery worker logs for task processing
- Check Flask application logs for API requests
- Monitor Redis server status
- Watch for disk space (temporary files)

## Common Issues

1. **Redis Connection Error**
   - Verify Redis is running: `redis-cli ping`
   - Check Redis service status
   
2. **Task Stuck in Processing**
   - Check Celery worker logs
   - Verify worker is running
   - Check Redis connection

3. **File Upload Issues**
   - Verify file format is supported
   - Check file size limits
   - Ensure proper permissions on upload directory

## Support

For issues and support, please create an issue in the repository or contact the maintainers.