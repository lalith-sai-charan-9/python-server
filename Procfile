web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 1
worker: celery -A celery_worker.celery worker --pool=solo -l info --concurrency=1 --without-heartbeat --without-mingle
