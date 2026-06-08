# Azure FastAPI Deployment

Proyecto para desplegar una API REST de **FastAPI** en Azure usando:

- Docker
- Azure CLI
- Azure Container Registry (ACR)
- Azure Web App for Containers

El ZIP contiene una única carpeta raíz con código fuente y scripts Bash, tal y como pide el evaluador.

## Estructura

```text
azure-fastapi-deployment/
├── app.py
├── requirements.txt
├── Dockerfile
├── build-image.sh
├── push-to-acr.sh
├── deploy-webapp.sh
├── .dockerignore
└── README.md
```

## Requisitos previos

```bash
docker --version
az --version
az login
az account show
```

## 1. Construir imagen local

```bash
chmod +x build-image.sh push-to-acr.sh deploy-webapp.sh
./build-image.sh
```

Variables modificables:

```bash
VERSION=1.0.1 IMAGE_NAME=fastapi-app ./build-image.sh
```

## 2. Probar localmente

```bash
docker run --rm -p 8000:8000 fastapi-app:1.0.0
```

Abrir:

```text
http://localhost:8000
http://localhost:8000/health
```

## 3. Subir imagen a Azure Container Registry

Los nombres de ACR deben ser globalmente únicos en Azure. Si el nombre por defecto ya existe, cambia `ACR_NAME`.

```bash
ACR_NAME=tuacrglobalunico RESOURCE_GROUP=rg-fastapi-ai-demo LOCATION=westeurope ./push-to-acr.sh
```

El script realiza:

1. Verifica login en Azure.
2. Crea el Resource Group si no existe.
3. Crea ACR si no existe.
4. Hace `az acr login`.
5. Retaguea la imagen con el login server de ACR.
6. Hace push de:
   - `fastapi-app:1.0.0`
   - `fastapi-app:latest`
7. Lista repositorios y tags subidos.

## 4. Desplegar Web App en Azure

El nombre de la Web App también debe ser globalmente único. Si el nombre por defecto ya existe, cambia `WEBAPP_NAME`.

```bash
ACR_NAME=tuacrglobalunico WEBAPP_NAME=tuwebappglobalunica ./deploy-webapp.sh
```

El script realiza:

1. Crea Resource Group.
2. Verifica ACR e imagen.
3. Crea App Service Plan Linux.
4. Crea Web App basada en contenedor.
5. Configura imagen desde ACR.
6. Configura credenciales de ACR.
7. Configura `WEBSITES_PORT=8000`.
8. Habilita CORS para cualquier origen.
9. Reinicia la Web App.
10. Muestra la URL final.

## 5. Endpoints

```text
GET /
GET /health
```

Respuesta de `/`:

```json
{
  "message": "Hello World from Azure!",
  "status": "running"
}
```

Respuesta de `/health`:

```json
{
  "status": "healthy",
  "service": "FastAPI on Azure"
}
```

## 6. Orden recomendado de ejecución

```bash
az login
chmod +x *.sh

./build-image.sh

ACR_NAME=tuacrglobalunico ./push-to-acr.sh

ACR_NAME=tuacrglobalunico WEBAPP_NAME=tuwebappglobalunica ./deploy-webapp.sh
```

## 7. Notas importantes

- `VERSION` usa versionamiento semántico por defecto: `1.0.0`.
- La imagen recibe doble tag: `1.0.0` y `latest`.
- El puerto interno expuesto es `8000`.
- Uvicorn escucha en `0.0.0.0:8000`.
- Los scripts usan `set -euo pipefail` para manejo básico de errores.
- No se incluyen archivos compilados, carpetas `dist`, `build`, `node_modules`, `.pyc` ni similares.
