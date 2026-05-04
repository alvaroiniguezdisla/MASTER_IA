# Reto proyecto: Workflow con tres agentes usando LangChain

Proyecto para EduFlowTech, una empresa ficticia especializada en plataformas educativas online.

## Contenido

- `reto_langchain_tres_agentes_eduflowtech.ipynb`: notebook principal listo para entregar.
- `requirements.txt`: dependencias necesarias.

## Qué implementa

El notebook contiene un flujo de trabajo con tres agentes:

1. **Procesador de consultas**: interpreta y categoriza la consulta del usuario.
2. **Buscador de contenido**: consulta una base de datos simulada con cursos, lecciones y ejercicios.
3. **Generador de respuestas**: crea respuestas personalizadas para el usuario.

## Tecnologías

- Python
- pandas
- OpenAI con API key ficticia
- LangChain / LangChain Core

## Ejecución

En Google Colab o Jupyter Notebook:

```bash
pip install -q langchain langchain-core langchain-openai openai pandas
```

Después ejecuta el notebook de principio a fin.

## Nota de seguridad

El código usa una API key ficticia:

```python
openai.api_key = "sk-YOUR_FAKE_API_KEY"
```

El notebook funciona por defecto en modo demo sin llamar a OpenAI. Para producción, se puede activar `USE_OPENAI = True` y configurar una clave real.

## Entrega en GitHub

El enunciado indica subir únicamente el archivo `.ipynb`; por tanto, para la entrega sube solo:

```text
reto_langchain_tres_agentes_eduflowtech.ipynb
```