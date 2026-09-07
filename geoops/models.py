"""Embedding bank ORM model.

One row = one embedding vector for one bbox/mask selection on one dataset
item, produced by one embedding model. ``model_name`` + ``embedding_dim`` are
denormalized from the embed model's own response (not just looked up off
``ai_models.id``) so a bank entry stays self-describing even if the
``ai_models`` row is later edited or deleted, and so every similarity query
can scope its comparison set to "same model, same dimension" without a join.

The ``embedding`` column is an *unconstrained* pgvector ``vector`` (no fixed
dimension at the column level) — different embedding models legitimately
return different dimensions, and we never compare across models anyway.
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        Index("idx_embeddings_org_model", "organization_id", "model_name"),
        Index("idx_embeddings_dataset_item", "dataset_item_id"),
        Index("idx_embeddings_class", "class_id"),
        Index("idx_embeddings_annotation", "annotation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    # Denormalized from the embed endpoint's own response — see module docstring.
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_items.id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotation_classes.id", ondelete="SET NULL"), nullable=True
    )
    annotation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True
    )
    # GeoJSON geometry of the bbox/mask selection the embedding was computed
    # from, in EPSG:4326 — mirrors DatasetItem.geometry's plain-JSONB choice
    # rather than a PostGIS column, since nothing here needs spatial SQL on it.
    source_geometry: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    organization: Mapped["Organization"] = relationship("Organization")
    model: Mapped["AIModel | None"] = relationship("AIModel")
    dataset_item: Mapped["DatasetItem"] = relationship("DatasetItem")
    cls: Mapped["AnnotationClass | None"] = relationship("AnnotationClass")
    annotation: Mapped["Annotation | None"] = relationship("Annotation")
