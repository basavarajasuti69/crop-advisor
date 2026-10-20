#!/usr/bin/env bash
# Deploy the crop-advisor API to Azure App Service (Linux).
#
#   APP_NAME=crop-advisor-yourname-api RESOURCE_GROUP=crop-advisor-yourname-api \
#     ./scripts/deploy_azure.sh
#
# Requires: az login, and the app to exist (see scripts/deploy_azure.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${APP_NAME:-crop-advisor-yourname-api}"
RESOURCE_GROUP="${RESOURCE_GROUP:-$APP_NAME}"
STARTUP_COMMAND="${STARTUP_COMMAND:-/home/site/wwwroot/venv/bin/python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000}"

if ! az account show >/dev/null 2>&1; then
  echo "ERROR: not logged in. Run 'az login'." >&2
  exit 1
fi

echo "==> Building the deployment package"
"$ROOT/scripts/package_azure.sh"

echo "==> Configuring app settings"
az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --startup-command "$STARTUP_COMMAND" \
  --output none

az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --output none \
  --settings \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    ENABLE_ORYX_BUILD=true \
    PYTHONUNBUFFERED=1 \
    CA_REGION="${CA_REGION:-ap-south-1}" \
    AWS_REGION="${AWS_REGION:-ap-south-1}" \
    CA_CORS_ORIGINS="${CA_CORS_ORIGINS:-http://localhost:3000}"

echo "==> Deploying"
az webapp deploy \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --src-path "$ROOT/.deploy/crop-advisor-api.zip" \
  --type zip \
  --output none

echo "==> Restarting"
az webapp restart --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --output none

URL="https://${APP_NAME}.azurewebsites.net"
echo "==> Waiting for the app to answer"
for i in $(seq 1 30); do
  if curl -fsS --max-time 20 "$URL/health" >/dev/null 2>&1; then
    echo "UP: $URL/health -> $(curl -fsS --max-time 20 "$URL/health")"
    exit 0
  fi
  sleep 10
done

echo "The app did not answer in time. Check the logs:" >&2
echo "  az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP" >&2
exit 1
