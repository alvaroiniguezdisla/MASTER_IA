# Reto proyecto: Genera imágenes con Stable Diffusion

Este proyecto está preparado para el ejercicio de generación de imágenes con Stable Diffusion.

## Contenido

- `reto_proyecto_genera_imagenes_stable_diffusion.ipynb`: notebook principal listo para entregar.
- `custom_dataset/`: carpeta con 10 imágenes base del concepto elegido.
- `requirements.txt`: dependencias necesarias.
- `outputs/`: carpeta donde se guardarán los resultados generados.

## Concepto elegido

Flores decorativas para ecommerce.

## Qué incluye el notebook

- Preparación y verificación del dataset `custom_dataset`.
- Comprobación de GPU con `torch.cuda.is_available()`.
- Uso de `stabilityai/stable-diffusion-2-1-base`.
- Scheduler principal `EulerAncestralDiscreteScheduler`.
- Generación text-to-image con dos prompts distintos.
- Uso de `negative_prompt`.
- Comparación de schedulers.
- Image-to-image con variaciones de una imagen base.
- Guardado automático de imágenes y grids comparativos.

## Ejecución recomendada

1. Abre el notebook en Google Colab.
2. Activa GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`.
3. Ejecuta todas las celdas.
4. Sube a GitHub únicamente el archivo `.ipynb`, como pide el enunciado.
