#!/bin/bash
set -e

cd ~/news_ai

echo "Pulling latest changes..."
git pull

echo "Refreshing auth..."
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)

echo "Building image..."
IMAGE_URL=$(cd infra && terraform output -raw image_url)
docker build -t $IMAGE_URL .

echo "Pushing image..."
docker push $IMAGE_URL

echo "Deploying..."
cd infra && terraform apply \
  -auto-approve \
  -var="project_id=$(gcloud config get-value project)"

echo "Done! Testing health..."
API_URL=$(terraform output -raw api_url)
curl $API_URL/health
