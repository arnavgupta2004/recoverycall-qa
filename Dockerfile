# RecoveryCall QA web app for Google Cloud Run (also runs anywhere Docker does).
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the app needs at runtime: code, sample audio and cached sample results.
COPY app.py ./
COPY src/ ./src/
COPY static/ ./static/
COPY samples/ ./samples/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Cloud Run sets PORT (8080); keys arrive as environment variables from Secret Manager.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --no-server-header --timeout-keep-alive 75"]
