from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.common import ORMModel


def _validate_bbox(bbox: list[float]) -> list[float]:
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox must be [minx, miny, maxx, maxy] with min < max")
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ValueError("bbox must be within EPSG:4326 bounds")
    return bbox


class EmbeddingCreateRequest(ORMModel):
    """Generate + store an embedding for one bbox/mask selection.

    Synchronous — a single crop + one call to the embed model, not a bulk job.
    """

    model_id: UUID
    dataset_item_id: UUID
    geometry: dict[str, Any] = Field(
        description="GeoJSON geometry (EPSG:4326) of the bbox or mask the user selected. "
        "Only its bounding box is used to crop the source patch."
    )
    annotation_id: UUID | None = Field(
        default=None,
        description="If the selection is already a saved Annotation, link the embedding to it "
        "and inherit its class_id (an explicit class_id is ignored when this is set).",
    )
    class_id: UUID | None = Field(
        default=None,
        description="Annotation class this embedding represents. Ignored if annotation_id is set.",
    )
    crop_size_px: int = Field(default=256, ge=32, le=1024)


class EmbeddingRead(ORMModel):
    id: UUID
    organization_id: UUID
    model_id: UUID | None
    model_name: str
    embedding_dim: int
    dataset_item_id: UUID
    class_id: UUID | None
    annotation_id: UUID | None
    source_geometry: dict[str, Any]
    created_at: datetime


class EmbeddingSearchRequest(ORMModel):
    """Nearest-neighbour search over the embedding bank.

    Always scoped to the reference embedding's ``model_name`` — vectors from
    different embedding models are never compared against each other.
    """

    embedding_id: UUID
    top_k: int = Field(default=10, ge=1, le=100)
    class_id: UUID | None = Field(default=None, description="Restrict candidates to this class.")
    dataset_id: UUID | None = Field(
        default=None, description="Restrict candidates to items belonging to this dataset."
    )
    exclude_self: bool = True


class EmbeddingSearchMatch(ORMModel):
    embedding_id: UUID
    dataset_item_id: UUID
    class_id: UUID | None
    annotation_id: UUID | None
    source_geometry: dict[str, Any]
    similarity: float = Field(description="1 - cosine distance. 1.0 = identical, 0.0 = orthogonal.")
    distance: float


class AOIScanRequest(ORMModel):
    """View-only similarity scan over an AOI. Nothing is persisted.

    Patches are tiled at the same geographic scale as the reference
    embedding's own source geometry, so matches are scale-comparable to the
    object being searched for.
    """

    reference_embedding_id: UUID
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5)
    aoi_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    scale_factor: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Multiplier on the reference geometry's bbox span used as the tile size.",
    )
    overlap: float = Field(default=0.5, ge=0.0, le=0.9, description="Fractional overlap between adjacent tiles.")
    crop_size_px: int = Field(default=256, ge=32, le=1024)
    top_k: int = Field(default=20, ge=1, le=200)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_patches: int = Field(default=200, ge=1, le=400)

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        return self


class AOIScanMatch(ORMModel):
    dataset_item_id: UUID
    bbox: list[float]
    similarity: float
    distance: float


class AOIScanResponse(ORMModel):
    reference_embedding_id: UUID
    model_name: str
    patches_scanned: int
    matches: list[AOIScanMatch]
