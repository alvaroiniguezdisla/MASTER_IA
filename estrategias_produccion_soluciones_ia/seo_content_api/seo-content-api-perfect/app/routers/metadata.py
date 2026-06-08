from fastapi import APIRouter, Depends

from app.dependencies import get_metadata_service
from app.models.metadata import MetadataRequest, MetadataResponse
from app.services.metadata_service import MetadataService

router = APIRouter(tags=["Metadata"])


@router.post("/generate", response_model=MetadataResponse)
async def generate_metadata(
    request: MetadataRequest,
    service: MetadataService = Depends(get_metadata_service),
) -> MetadataResponse:
    """
    Genera 3-5 meta titles y 3-5 meta descriptions optimizadas para CTR.
    """
    return await service.generate(request)
