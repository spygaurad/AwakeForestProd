from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_session, require_org_role
from app.models.user import User

from geoops.schemas import (
    AOIScanRequest,
    AOIScanResponse,
    EmbeddingCreateRequest,
    EmbeddingRead,
    EmbeddingSearchMatch,
    EmbeddingSearchRequest,
)
from geoops.service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("", response_model=EmbeddingRead, status_code=status.HTTP_201_CREATED)
async def create_embedding(
    payload: EmbeddingCreateRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Crop the selected bbox/mask, embed it, and store the vector.

    Synchronous — a single patch + one call to the embed model.
    """
    service = EmbeddingService(db)
    return await service.create_embedding(org_id, current_user.id, payload)


@router.get("/{embedding_id}", response_model=EmbeddingRead)
async def get_embedding(
    embedding_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    service = EmbeddingService(db)
    return await service.get_embedding(org_id, embedding_id)


@router.post("/search", response_model=list[EmbeddingSearchMatch])
async def search_embeddings(
    payload: EmbeddingSearchRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Cosine nearest-neighbours over the embedding bank."""
    service = EmbeddingService(db)
    return await service.search(org_id, payload)


@router.post("/aoi-scan", response_model=AOIScanResponse)
async def aoi_scan(
    payload: AOIScanRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Scan an AOI for patches similar to a reference embedding.

    View-only: tiles the AOI at the reference embedding's own scale, embeds
    each tile, and ranks by similarity. Nothing is written to the database —
    the response is the result.
    """
    service = EmbeddingService(db)
    return await service.aoi_scan(org_id, payload)
