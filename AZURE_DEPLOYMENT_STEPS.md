# Azure Deployment Steps (Container Apps)

Use this for a fast public deployment of both services:
- API (FastAPI)
- Frontend (Streamlit)

## 0. Prerequisites
- Azure subscription
- Docker installed and running
- Project already has model artifacts:
  - `artifacts/fraud_model_v1.joblib`
  - `artifacts/fraud_model_v1.json`

## 1. Install Azure CLI (Linux)
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Verify:
```bash
az version
```

## 2. Login to Azure
```bash
az login
```

Optional (if you have multiple subscriptions):
```bash
az account list -o table
az account set --subscription "<SUBSCRIPTION_ID_OR_NAME>"
```

## 3. Run One-Command Deployment Script
From project root:
```bash
chmod +x deploy/azure_container_apps.sh
./deploy/azure_container_apps.sh
```

Optional custom names/region:
```bash
RESOURCE_GROUP=fraud-rg LOCATION=eastus ACR_NAME=fraudacr123 CONTAINER_ENV=fraud-env API_APP_NAME=fraud-api FRONTEND_APP_NAME=fraud-ui MODEL_VERSION=v1 ./deploy/azure_container_apps.sh
```

## 4. Validate Deployment
After script finishes, it prints two URLs:
- API URL
- Frontend URL

Check API:
```bash
curl -s https://<API_URL>/health
```

Open frontend in browser:
```text
https://<FRONTEND_URL>
```

## 5. Troubleshooting
Tail logs:
```bash
az containerapp logs show --name <API_APP_NAME> --resource-group <RESOURCE_GROUP> --follow
az containerapp logs show --name <FRONTEND_APP_NAME> --resource-group <RESOURCE_GROUP> --follow
```

Check app status:
```bash
az containerapp show --name <API_APP_NAME> --resource-group <RESOURCE_GROUP> --query properties.latestRevisionName -o tsv
az containerapp show --name <FRONTEND_APP_NAME> --resource-group <RESOURCE_GROUP> --query properties.latestRevisionName -o tsv
```

## 6. What To Share Back
Share these two URLs with me and I will mark the final roadmap checkbox complete:
- Public API URL
- Public Frontend URL
