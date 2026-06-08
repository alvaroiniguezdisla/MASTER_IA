from app.models.faqs import FAQRequest, FAQResponse
from app.services.openai_client import AzureOpenAIService


class FAQService:
    def __init__(self, openai_service: AzureOpenAIService) -> None:
        self.openai_service = openai_service

    async def extract(self, request: FAQRequest) -> FAQResponse:
        system_prompt = """
        Eres un especialista SEO en rich snippets y datos estructurados Schema.org.
        Extrae FAQs naturales a partir del contenido del artículo.
        Devuelve únicamente JSON válido según el esquema indicado.
        """

        user_prompt = f"""
        Extrae hasta {request.max_questions} preguntas frecuentes del siguiente artículo.

        Artículo:
        {request.article_content}

        Requisitos:
        1. Genera preguntas naturales y relevantes.
        2. Cada respuesta debe ser concisa, útil y tener entre 50 y 150 palabras aproximadamente.
        3. Devuelve también un json_ld_schema válido de tipo FAQPage.
        4. El JSON-LD debe seguir Schema.org:
           {{
             "@context": "https://schema.org",
             "@type": "FAQPage",
             "mainEntity": [...]
           }}
        5. No incluyas campos extra fuera del esquema Pydantic.
        """

        return await self.openai_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=FAQResponse,
            temperature=0.35,
        )
