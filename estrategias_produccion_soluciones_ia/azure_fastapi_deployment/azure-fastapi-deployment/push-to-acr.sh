#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# Configuración reutilizable
# Cambia estos valores o pásalos como variables:
# ACR_NAME=miacr VERSION=1.0.1 ./push-to-acr.sh
# ==================================================
VERSION="${VERSION:-1.0.0}"
IMAGE_NAME="${IMAGE_NAME:-fastapi-app}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-fastapi-ai-demo}"
LOCATION="${LOCATION:-westeurope}"
ACR_NAME="${ACR_NAME:-acrfastapiia15873}"

echo "=========================================="
echo " Push de imagen Docker a Azure ACR"
echo "=========================================="
echo "RESOURCE_GROUP: ${RESOURCE_GROUP}"
echo "LOCATION:       ${LOCATION}"
echo "ACR_NAME:       ${ACR_NAME}"
echo "IMAGE_NAME:     ${IMAGE_NAME}"
echo "VERSION:        ${VERSION}"
echo "------------------------------------------"

echo "1/7 Verificando login en Azure..."
az account show > /dev/null

echo "2/7 Creando Resource Group si no existe..."
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output table

echo "3/7 Creando Azure Container Registry si no existe..."
if az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" > /dev/null 2>&1; then
  echo "ACR ${ACR_NAME} ya existe."
else
  az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ACR_NAME}" \
    --sku Basic \
    --admin-enabled true \
    --location "${LOCATION}" \
    --output table
fi

echo "4/7 Habilitando credenciales admin de ACR..."
az acr update \
  --name "${ACR_NAME}" \
  --admin-enabled true \
  --output none

echo "5/7 Login en ACR..."
az acr login --name "${ACR_NAME}"

ACR_LOGIN_SERVER="$(az acr show \
  --name "${ACR_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query loginServer \
  --output tsv)"

echo "ACR_LOGIN_SERVER: ${ACR_LOGIN_SERVER}"

echo "6/7 Re-etiquetando imagen local para ACR..."
docker tag "${IMAGE_NAME}:${VERSION}" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"

echo "Subiendo ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${VERSION}..."
docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${VERSION}"

echo "Subiendo ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest..."
docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"

echo "7/7 Verificando imágenes subidas a ACR..."
az acr repository list \
  --name "${ACR_NAME}" \
  --output table

echo "Tags disponibles para ${IMAGE_NAME}:"
az acr repository show-tags \
  --name "${ACR_NAME}" \
  --repository "${IMAGE_NAME}" \
  --output table

echo "------------------------------------------"
echo "Push a ACR completado correctamente."
echo "Imagen versionada: ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${VERSION}"
echo "Imagen latest:     ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"
echo "------------------------------------------"
