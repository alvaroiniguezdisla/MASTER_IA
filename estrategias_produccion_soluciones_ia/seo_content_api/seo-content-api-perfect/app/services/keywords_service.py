from app.models.keywords import KeywordRequest, KeywordResponse
from app.services.openai_client import AzureOpenAIService


class KeywordService:
    def __init__(self, openai_service: AzureOpenAIService) -> None:
        self.openai_service = openai_service

    async def generate(self, request: KeywordRequest) -> KeywordResponse:
        system_prompt = """
        Eres un especialista senior en SEO, keyword research e intención de búsqueda.
        Devuelve únicamente JSON válido que cumpla exactamente el esquema indicado.
        Clasifica cada keyword como 'informacional' o 'transaccional'.
        No inventes campos adicionales.
        """

        user_prompt = f"""
        Genera una investigación SEO para:
        - Tema principal: {request.topic}
        - Industria: {request.industry}
        - Idioma: {request.language}

        Requisitos:
        1. Incluye 5-15 seed keywords.
        2. Incluye 5-15 long-tail keywords específicas y realistas.
        3. Incluye 5-15 preguntas naturales que un usuario buscaría en Google.
        4. Clasifica al menos 5 keywords por intención: informacional o transaccional.
        5. Añade una breve razón de la intención.
        """

        return await self.openai_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=KeywordResponse,
            temperature=0.35,
        )
