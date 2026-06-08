from app.models.social import SocialRequest, SocialResponse
from app.services.openai_client import AzureOpenAIService


class SocialService:
    def __init__(self, openai_service: AzureOpenAIService) -> None:
        self.openai_service = openai_service

    async def generate(self, request: SocialRequest) -> SocialResponse:
        platforms = ", ".join(request.target_platforms)

        system_prompt = """
        Eres un social media manager experto en adaptar contenido SEO a redes sociales.
        Debes respetar los límites de caracteres y adaptar tono, formato, hashtags y CTA por plataforma.
        Devuelve únicamente JSON válido según el esquema indicado.
        """

        user_prompt = f"""
        Genera resúmenes sociales para las plataformas solicitadas.

        Plataformas objetivo: {platforms}
        Título del artículo: {request.article_title}

        Contenido del artículo:
        {request.article_content}

        Requisitos por plataforma:
        - Twitter/X: máximo 280 caracteres en el campo text, mensaje directo, 1-4 hashtags y CTA breve.
        - LinkedIn: tono profesional, más desarrollado, 3-8 hashtags y CTA orientado a conversación.
        - Instagram: caption visual/emocional, 5-15 hashtags y CTA de interacción.
        - Facebook: tono cercano, claro, 1-6 hashtags y CTA de lectura o comentario.

        Si una plataforma no está en target_platforms, su valor debe ser null.
        No añadas campos extra.
        """

        return await self.openai_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SocialResponse,
            temperature=0.55,
        )
