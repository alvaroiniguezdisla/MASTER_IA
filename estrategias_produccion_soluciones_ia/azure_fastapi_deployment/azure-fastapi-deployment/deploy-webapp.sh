#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# Configuración reutilizable
# Cambia estos valores o pásalos como variables:
# WEBAPP_NAME=miwebappunica ACR_NAME=miacr ./deploy-webapp.sh
# ==================================================
VERSION="${VERSION:-1.0.0}"
IMAGE_NAME="${IMAGE_NAME:-fastapi-app}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-fastapi-ai-demo}"
LOCATION="${LOCATION:-westeurope}"
ACR_NAME="${ACR_NAME:-acrfastapiia15873}"
APP_SERVICE_PLAN="${APP_SERVICE_PLAN:-asp-fastapi-ai-demo}"
WEBAPP_NAME="${WEBAPP_NAME:-webapp-fastapi-ai-15873}"
SKU="${SKU:-B1}"

echo "=========================================="
echo " Despliegue de FastAPI Docker en Azure Web App"
echo "=========================================="
echo "RESOURCE_GROUP:   ${RESOURCE_GROUP}"
echo "LOCATION:         ${LOCATION}"
echo "ACR_NAME:         ${ACR_NAME}"
echo "APP_SERVICE_PLAN: ${APP_SERVICE_PLAN}"
echo "WEBAPP_NAME:      ${WEBAPP_NAME}"
echo "IMAGE_NAME:       ${IMAGE_NAME}"
echo "VERSION:          ${VERSION}"
echo "SKU:              ${SKU}"
echo "------------------------------------------"

echo "1/10 Verificando login en Azure..."
az account show > /dev/null

echo "2/10 Creando Resource Group si no existe..."
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output table

echo "3/10 Verificando Azure Container Registry..."
if ! az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" > /dev/null 2>&1; then
  echo "ERROR: El ACR ${ACR_NAME} no existe."
  echo "Ejecuta primero: ./push-to-acr.sh"
  exit 1
fi

ACR_LOGIN_SERVER="$(az acr show \
  --name "${ACR_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query loginServer \
  --output tsv)"

FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${VERSION}"

echo "ACR_LOGIN_SERVER: ${ACR_LOGIN_SERVER}"
echo "FULL_IMAGE_NAME:  ${FULL_IMAGE_NAME}"

echo "4/10 Verificando que la imagen exista en ACR..."
if ! az acr repository show-tags \
  --name "${ACR_NAME}" \
  --repository "${IMAGE_NAME}" \
  --query "[?@=='${VERSION}']" \
  --output tsv | grep -q "${VERSION}"; then
  echo "ERROR: No se encuentra la imagen ${IMAGE_NAME}:${VERSION} en ACR."
  echo "Ejecuta primero:"
  echo "./build-image.sh"
  echo "./push-to-acr.sh"
  exit 1
fi

echo "5/10 Creando App Service Plan Linux si no existe..."
if az appservice plan show --name "${APP_SERVICE_PLAN}" --resource-group "${RESOURCE_GROUP}" > /dev/null 2>&1; then
  echo "App Service Plan ${APP_SERVICE_PLAN} ya existe."
else
  az appservice plan create \
    --name "${APP_SERVICE_PLAN}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --is-linux \
    --sku "${SKU}" \
    --output table
fi

echo "6/10 Obteniendo credenciales de ACR..."
az acr update \
  --name "${ACR_NAME}" \
  --admin-enabled true \
  --output none

ACR_USERNAME="$(az acr credential show \
  --name "${ACR_NAME}" \
  --query username \
  --output tsv)"

ACR_PASSWORD="$(az acr credential show \
  --name "${ACR_NAME}" \
  --query passwords[0].value \
  --output tsv)"

echo "7/10 Creando Web App si no existe..."
if az webapp show --name "${WEBAPP_NAME}" --resource-group "${RESOURCE_GROUP}" > /dev/null 2>&1; then
  echo "Web App ${WEBAPP_NAME} ya existe."
else
  az webapp create \
    --resource-group "${RESOURCE_GROUP}" \
    --plan "${APP_SERVICE_PLAN}" \
    --name "${WEBAPP_NAME}" \
    --deployment-container-image-name "${FULL_IMAGE_NAME}" \
    --output table
fi

echo "8/10 Configurando contenedor desde ACR..."
az webapp config container set \
  --name "${WEBAPP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --docker-custom-image-name "${FULL_IMAGE_NAME}" \
  --docker-registry-server-url "https://${ACR_LOGIN_SERVER}" \
  --docker-registry-server-user "${ACR_USERNAME}" \
  --docker-registry-server-password "${ACR_PASSWORD}" \
  --output table

echo "9/10 Configurando puerto 8000 y app settings..."
az webapp config appsettings set \
  --name "${WEBAPP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --settings \
    WEBSITES_PORT=8000 \
    DOCKER_ENABLE_CI=true \
  --output table

echo "Habilitando CORS para cualquier origen..."
az webapp cors add \
  --name "${WEBAPP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --allowed-origins "*" \
  --output table || true

echo "10/10 Reiniciando Web App..."
az webapp restart \
  --name "${WEBAPP_NAME}" \
  --resource-group "${RESOURCE_GROUP}"

DEFAULT_HOSTNAME="$(az webapp show \
  --name "${WEBAPP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query defaultHostName \
  --output tsv)"

echo "------------------------------------------"
echo "Despliegue completado correctamente."
echo "URL aplicación: https://${DEFAULT_HOSTNAME}"
echo "Health check:   https://${DEFAULT_HOSTNAME}/health"
echo "------------------------------------------"
echo "Si la app tarda unos minutos en responder, revisa logs con:"
echo "az webapp log tail --name ${WEBAPP_NAME} --resource-group ${RESOURCE_GROUP}"
