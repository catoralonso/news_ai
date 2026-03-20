#!/bin/bash
set -e

cd ~/news_ai

echo "Pulling latest changes..."
git pull

echo "Refreshing auth..."
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)

echo "Building and pushing image..."
IMAGE_TAG=$(date +%Y%m%d%H%M%S)
IMAGE_URL="us-central1-docker.pkg.dev/savia-490716/savia/savia:$IMAGE_TAG"
gcloud builds submit --tag $IMAGE_URL --project savia-490716 .

echo "Deploying..."
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)
cd infra && terraform apply \
  -auto-approve \
  -var="project_id=$(gcloud config get-value project)" \
  -var="image_tag=$IMAGE_TAG"

echo "Done! Testing health..."
API_URL=$(terraform output -raw api_url)
curl $API_URL/health
