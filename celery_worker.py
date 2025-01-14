import os
from app import celery, app

# Push the Flask application context
app.app_context().push()

# Ensure Celery worker uses the same broker URL
REDIS_URL = os.getenv('REDIS_PUBLIC_URL', 'redis://localhost:6379/0')
celery.conf.broker_url = REDIS_URL
celery.conf.result_backend = REDIS_URL

# Initialize the model within the worker
from app import download_model, get_model

download_model()
get_model()
