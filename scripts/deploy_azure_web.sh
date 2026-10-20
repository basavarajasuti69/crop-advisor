#!/usr/bin/env bash
# Deploy the web app to Azure Static Web Apps.
#
# Needs a deployment token from the Azure portal:
#   Static Web Apps > crop-advisor-web > Overview > Manage deployment token
#
#   API_URL=https://... SWA_TOKEN=... ./scripts/deploy_azure_web.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$ROOT/apps/web"
SWA_NAME="${SWA_NAME:-crop-advisor-web}"
SWA_TOKEN="${SWA_TOKEN:-}"
API_URL="${API_URL:-}"

if [ -z "$SWA_TOKEN" ]; then
  echo "ERROR: SWA_TOKEN is not set." >&2
  echo "Get it from the portal: Static Web Apps > $SWA_NAME > Overview >" >&2
  echo "Manage deployment token, then run:" >&2
  echo "  API_URL=<the container app url> SWA_TOKEN=<token> $0" >&2
  exit 1
fi
if [ -z "$API_URL" ]; then
  echo "ERROR: API_URL is not set (the Container App URL)." >&2
  exit 1
fi

echo "==> Building the static export"
cd "$WEB_DIR"
NEXT_OUTPUT=export NEXT_PUBLIC_API_URL="$API_URL" npx next build

echo "==> Uploading to Static Web Apps"
npx --yes @azure/static-web-apps-cli@2.1.0 deploy \
  --app-location "$WEB_DIR/out" \
  --deployment-token "$SWA_TOKEN" \
  --env "API_URL=$API_URL" \
  --output-location ".swa-deploy"

echo
echo "Deployed. The URL is in .swa-deploy/ deployment.json and at:"
echo "  https://${SWA_NAME}.azurestaticapps.net"
