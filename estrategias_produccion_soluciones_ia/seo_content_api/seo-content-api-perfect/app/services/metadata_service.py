from app.models.metadata import MetadataRequest, MetadataResponse
from app.services.openai_client import AzureOpenAIService


class MetadataService:
    def __init__(self, openai_service: AzureOpenAIService) -> None:
        self.openai_service = openai_service

    async def generate(self, request: MetadataRequest) -> MetadataResponse:
        system_prompt = """
        Eres un experto SEO técnico especializado en mejorar CTR desde Google.
        Genera metadatos persuasivos, claros y optimizados.
        Devuelve únicamente JSON válido que cumpla estrictamente el esquema.
        """

        user_prompt = f"""
        Genera metadatos SEO para este artículo:
        - Título del artículo: {request.article_title}
        - Keyword principal: {request.main_keyword}
        - Extracto: {request.article_excerpt}

        Requisitos:
        1. Genera entre 3 y 5 meta titles.
        2. Cada meta title debe tener máximo 60 caracteres.
        3. Genera entre 3 y 5 meta descriptions.
        4. Cada meta description debe tener máximo 160 caracteres.
        5. Incluye la keyword principal de forma natural siempre que sea posible.
        6. Usa lenguaje persuasivo orientado a CTR.
        7. character_count debe coincidir con la longitud real del texto.
        """

        return await self.openai_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MetadataResponse,
            temperature=0.45,
        )
