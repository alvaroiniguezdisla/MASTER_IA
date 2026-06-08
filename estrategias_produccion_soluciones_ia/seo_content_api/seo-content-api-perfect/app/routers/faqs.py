from fastapi import APIRouter, Depends

from app.dependencies import get_faq_service
from app.models.faqs import FAQRequest, FAQResponse
from app.services.faqs_service import FAQService

router = APIRouter(tags=["FAQs"])


@router.post("/extract", response_model=FAQResponse)
async def extract_faqs(
    request: FAQRequest,
    service: FAQService = Depends(get_faq_service),
) -> FAQResponse:
    """
    Extrae FAQs relevantes y genera JSON-LD FAQPage válido.
    """
    return await service.extract(request)
