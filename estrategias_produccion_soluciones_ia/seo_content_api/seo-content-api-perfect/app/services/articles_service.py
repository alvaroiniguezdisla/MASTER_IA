from app.models.articles import ArticleRequest, ArticleResponse
from app.services.openai_client import AzureOpenAIService


class ArticleService:
    def __init__(self, openai_service: AzureOpenAIService) -> None:
        self.openai_service = openai_service

    async def generate(self, request: ArticleRequest) -> ArticleResponse:
        secondary = ", ".join(request.secondary_keywords) if request.secondary_keywords else "sin keywords secundarias"

        system_prompt = """
        Eres un redactor SEO experto.
        Genera artículos con estructura jerárquica H1/H2/H3, intención de búsqueda cubierta,
        keywords integradas de forma natural y sin keyword stuffing.
        Devuelve únicamente JSON válido según el esquema indicado.
        """

        user_prompt = f"""
        Crea un artículo SEO completo con estos datos:
        - Keyword principal: {request.main_keyword}
        - Keywords secundarias: {secondary}
        - Extensión aproximada: {request.word_count} palabras
        - Tono: {request.tone}

        Requisitos:
        1. El título debe funcionar como H1.
        2. Incluye una estructura con al menos 1 H1, varios H2 y algunos H3.
        3. Integra las keywords de forma natural.
        4. Calcula una densidad aproximada para la keyword principal y las secundarias.
        5. La densidad debe ser natural, sin sobreoptimización.
        6. Incluye 2-5 CTAs coherentes con el artículo.
        7. No uses campos extra fuera del esquema.
        """

        return await self.openai_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ArticleResponse,
            temperature=0.5,
        )
