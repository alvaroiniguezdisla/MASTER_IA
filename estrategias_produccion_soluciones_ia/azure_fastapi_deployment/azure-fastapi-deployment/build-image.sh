#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuración reutilizable
# =========================
VERSION="${VERSION:-1.0.0}"
IMAGE_NAME="${IMAGE_NAME:-fastapi-app}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"

echo "=========================================="
echo " Construcción de imagen Docker"
echo "=========================================="
echo "IMAGE_NAME: ${IMAGE_NAME}"
echo "VERSION:    ${VERSION}"
echo "DOCKERFILE: ${DOCKERFILE}"
echo "CONTEXT:    ${BUILD_CONTEXT}"
echo "------------------------------------------"

echo "1/3 Construyendo imagen ${IMAGE_NAME}:${VERSION}..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}:${VERSION}" "${BUILD_CONTEXT}"

echo "2/3 Creando tag latest..."
docker tag "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest"

echo "3/3 Imágenes creadas:"
docker images "${IMAGE_NAME}"

echo "------------------------------------------"
echo "Build completado correctamente."
echo "Imagen versionada: ${IMAGE_NAME}:${VERSION}"
echo "Imagen latest:     ${IMAGE_NAME}:latest"
echo "------------------------------------------"
echo "Prueba local:"
echo "docker run --rm -p 8000:8000 ${IMAGE_NAME}:${VERSION}"
