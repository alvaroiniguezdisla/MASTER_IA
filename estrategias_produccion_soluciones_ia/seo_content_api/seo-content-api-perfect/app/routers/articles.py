from fastapi import APIRouter, Depends

from app.dependencies import get_article_service
from app.models.articles import ArticleRequest, ArticleResponse
from app.services.articles_service import ArticleService

router = APIRouter(tags=["Articles"])


@router.post("/generate", response_model=ArticleResponse)
async def generate_article(
    request: ArticleRequest,
    service: ArticleService = Depends(get_article_service),
) -> ArticleResponse:
    """
    Genera un artículo SEO completo con H1/H2/H3, densidad natural de keywords y CTAs.
    """
    return await service.generate(request)
