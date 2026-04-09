#!/usr/bin/env bash
set -euo pipefail

# Deploy API + Streamlit frontend to Azure Container Apps using ACR images.
# Prerequisites:
# 1) az CLI installed
# 2) docker installed and running
# 3) az login already completed

# ---------- REQUIRED VARIABLES ----------
RESOURCE_GROUP="${RESOURCE_GROUP:-fraud-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-fraudacr$RANDOM}"
CONTAINER_ENV="${CONTAINER_ENV:-fraud-ca-env}"
API_APP_NAME="${API_APP_NAME:-fraud-api-app}"
FRONTEND_APP_NAME="${FRONTEND_APP_NAME:-fraud-frontend-app}"
MODEL_VERSION="${MODEL_VERSION:-v1}"

# Optional: set your subscription explicitly.
# AZ_SUBSCRIPTION_ID="<your-subscription-id>"

if ! command -v az >/dev/null; then
  echo "Azure CLI is not installed. Install it first."
  exit 1
fi

if ! command -v docker >/dev/null; then
  echo "Docker is not installed. Install it first."
  exit 1
fi

# Uncomment if using explicit subscription id.
# az account set --subscription "$AZ_SUBSCRIPTION_ID"

echo "[1/11] Ensure resource group"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

echo "[2/11] Ensure Azure Container Registry"
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Basic \
    --admin-enabled true >/dev/null
fi

ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)"
ACR_USERNAME="$(az acr credential show --name "$ACR_NAME" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show --name "$ACR_NAME" --query passwords[0].value -o tsv)"

echo "[3/11] Build and push API image"
az acr build --registry "$ACR_NAME" --image fraud-api:latest --file api/Dockerfile .

echo "[4/11] Build and push frontend image"
az acr build --registry "$ACR_NAME" --image fraud-frontend:latest --file frontend/Dockerfile .

echo "[5/11] Ensure Container Apps environment"
if ! az containerapp env show --name "$CONTAINER_ENV" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$CONTAINER_ENV" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" >/dev/null
fi

echo "[6/11] Deploy API Container App"
if ! az containerapp show --name "$API_APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp create \
    --name "$API_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_ENV" \
    --image "$ACR_LOGIN_SERVER/fraud-api:latest" \
    --target-port 8000 \
    --ingress external \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --env-vars MODEL_VERSION="$MODEL_VERSION" ARTIFACTS_DIR="/app/artifacts" MODEL_PATH="/app/artifacts/fraud_model_${MODEL_VERSION}.joblib" MODEL_METADATA_PATH="/app/artifacts/fraud_model_${MODEL_VERSION}.json" >/dev/null
else
  az containerapp update \
    --name "$API_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$ACR_LOGIN_SERVER/fraud-api:latest" \
    --set-env-vars MODEL_VERSION="$MODEL_VERSION" ARTIFACTS_DIR="/app/artifacts" MODEL_PATH="/app/artifacts/fraud_model_${MODEL_VERSION}.joblib" MODEL_METADATA_PATH="/app/artifacts/fraud_model_${MODEL_VERSION}.json" >/dev/null
fi

API_URL="$(az containerapp show --name "$API_APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"

echo "[7/11] Deploy frontend Container App"
if ! az containerapp show --name "$FRONTEND_APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp create \
    --name "$FRONTEND_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_ENV" \
    --image "$ACR_LOGIN_SERVER/fraud-frontend:latest" \
    --target-port 8501 \
    --ingress external \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --env-vars API_BASE_URL="https://${API_URL}" >/dev/null
else
  az containerapp update \
    --name "$FRONTEND_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$ACR_LOGIN_SERVER/fraud-frontend:latest" \
    --set-env-vars API_BASE_URL="https://${API_URL}" >/dev/null
fi

FRONTEND_URL="$(az containerapp show --name "$FRONTEND_APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"

echo "[8/11] Smoke test API health"
curl -s "https://${API_URL}/health" || true

echo "[9/11] Deployment outputs"
echo "API URL: https://${API_URL}"
echo "Frontend URL: https://${FRONTEND_URL}"

echo "[10/11] Useful follow-ups"
echo "az containerapp logs show --name $API_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo "az containerapp logs show --name $FRONTEND_APP_NAME --resource-group $RESOURCE_GROUP --follow"

echo "[11/11] Done"
