from fastapi import APIRouter, Depends

from app.dependencies import get_keyword_service
from app.models.keywords import KeywordRequest, KeywordResponse
from app.services.keywords_service import KeywordService

router = APIRouter(tags=["Keywords"])


@router.post("/generate", response_model=KeywordResponse)
async def generate_keywords(
    request: KeywordRequest,
    service: KeywordService = Depends(get_keyword_service),
) -> KeywordResponse:
    """
    Genera keywords semilla, long-tail, preguntas e intención de búsqueda.
    """
    return await service.generate(request)
