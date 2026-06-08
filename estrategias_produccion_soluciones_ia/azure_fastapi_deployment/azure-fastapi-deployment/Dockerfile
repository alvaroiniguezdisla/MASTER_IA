# Imagen base oficial, ligera y apropiada para producción
FROM python:3.12-slim

# Evita archivos .pyc y mejora logs en contenedores
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Primero copiamos requirements para aprovechar caché de Docker
COPY requirements.txt .

# Instalación de dependencias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiamos solo el código fuente necesario
COPY app.py .

# Puerto interno usado por FastAPI/Uvicorn
EXPOSE 8000

# Comando de inicio de la API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
