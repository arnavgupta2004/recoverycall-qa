#!/usr/bin/env bash
# Build and deploy RecoveryCall QA to Google Cloud Run (asia-south1).
#
#   PROJECT=<gcp-project-id> ./deploy_cloudrun.sh
#
# One-time setup this script expects (see README, "Deploy to Google Cloud Run"):
#   - billing linked; APIs run, artifactregistry, cloudbuild, secretmanager enabled
#   - Artifact Registry docker repo "rcqa" in asia-south1
#   - Secret Manager secrets GNANI_API_KEY, GEMINI_API_KEY, UPLOAD_PASSCODE
#   - service account rcqa-run@PROJECT with roles/secretmanager.secretAccessor on those secrets
# Secrets are never passed on the command line or baked into the image.
set -euo pipefail
: "${PROJECT:?set PROJECT to the GCP project id}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-recoverycall-qa}"
TAG="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/rcqa/app:$TAG"

gcloud builds submit --project "$PROJECT" --region "$REGION" --tag "$IMAGE" .

gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "rcqa-run@$PROJECT.iam.gserviceaccount.com" \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 1 --concurrency 20 \
  --cpu 1 --memory 1Gi --timeout 300 \
  --set-env-vars GRADIO_ANALYTICS_ENABLED=False,LIVE_RUNS_PER_HOUR=5 \
  --set-secrets GNANI_API_KEY=GNANI_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,UPLOAD_PASSCODE=UPLOAD_PASSCODE:latest

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format "value(status.url)"
