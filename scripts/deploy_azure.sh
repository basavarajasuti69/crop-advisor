#!/usr/bin/env bash
# LEGACY: Azure App Service deployment.
#
# The deployment target is now Hugging Face Spaces (see the "Deploy
# (free)" section of README.md). This script is kept for reference only.
#
#   APP_NAME=crop-advisor-yourname-api RESOURCE_GROUP=crop-advisor-yourname-api \
#     ./scripts/deploy_azure.sh
#
# Requires: az login, and the app to exist (see scripts/deploy_azure.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${APP_NAME:-crop-advisor-yourname-api}"
RESOURCE_GROUP="${RESOURCE_GROUP:-$APP_NAME}"
STARTUP_FILE="${STARTUP_FILE:-application.py}"

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
  --startup-file "startup.sh" \
  --output none

# With a pre-built venv in the package the instance does not need to run pip.
if [ "${INCLUDE_VENV:-0}" = "1" ]; then
  BUILD_ON_INSTANCE=false
else
  BUILD_ON_INSTANCE=true
fi

az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --output none \
  --settings \
    SCM_DO_BUILD_DURING_DEPLOYMENT=$BUILD_ON_INSTANCE \
    ENABLE_ORYX_BUILD=$BUILD_ON_INSTANCE \
    PYTHONUNBUFFERED=1 \
    CA_REGION="${CA_REGION:-ap-south-1}" \
    AWS_REGION="${AWS_REGION:-ap-south-1}" \
    CA_CORS_ORIGINS="${CA_CORS_ORIGINS:-http://localhost:3000,https://zealous-smoke-014b04200.3.azurestaticapps.net}"

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
