web: gunicorn app:app --bind 0.0.0.0:$PORT
worker: celery -A celery_worker.celery worker --pool=solo -l debug --concurrency=1