from fastapi import APIRouter, Depends

from app.dependencies import get_social_service
from app.models.social import SocialRequest, SocialResponse
from app.services.social_service import SocialService

router = APIRouter(tags=["Social"])


@router.post("/summaries", response_model=SocialResponse)
async def generate_social_summaries(
    request: SocialRequest,
    service: SocialService = Depends(get_social_service),
) -> SocialResponse:
    """
    Genera contenido adaptado para Twitter/X, LinkedIn, Instagram y Facebook.
    """
    return await service.generate(request)
