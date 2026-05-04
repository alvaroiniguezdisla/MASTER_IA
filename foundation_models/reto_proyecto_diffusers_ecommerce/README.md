# Reto proyecto: diffusers para generación de imágenes de productos e-commerce

Este paquete contiene un notebook listo para ejecutar en Google Colab o en un entorno local con Python.

## Archivo principal

- `reto_proyecto_diffusers_ecommerce.ipynb`

## Requisitos principales cubiertos

- Uso de `diffusers`, `torch` y `PIL`.
- Detección de GPU si está disponible.
- Modelo: `stabilityai/stable-diffusion-2-1-base`.
- Scheduler principal: `EulerAncestralDiscreteScheduler`.
- Generación de imágenes de producto para e-commerce.
- Uso de `negative_prompt`.
- Experimentación con prompts, parámetros y schedulers.
- Técnica image-to-image para crear variaciones de una imagen base.
- Guardado de imágenes y grids en carpeta `outputs`.

## Ejecución recomendada

1. Abrir el notebook en Google Colab.
2. Activar GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`.
3. Ejecutar las celdas en orden.
4. Subir el archivo `.ipynb` a la plataforma o a GitHub según indique el ejercicio.

## Nota

La primera ejecución descarga el modelo desde HuggingFace, por lo que requiere conexión a internet y puede tardar varios minutos.
