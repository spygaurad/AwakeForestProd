"""Create embeddings table — the pgvector embedding bank.

The ``embedding`` column is an *unconstrained* pgvector ``vector`` (no fixed
dimension). Different embedding models can return different dimensions;
``model_name`` + ``embedding_dim`` are stored per row so every similarity
query can scope its comparison set to "same model, same dimension" and the
``<=>``/``cosine_distance`` operator never sees a mismatch. No ANN index
(HNSW/IVFFlat) yet — v1 scope, revisit once one model/dimension is standard.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-09-07
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "e1f2a3b4c5d6"
down_revision = "d0e1f2a3b4c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "organization_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_id", UUID(as_uuid=True), sa.ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column(
            "dataset_item_id",
            UUID(as_uuid=True),
            sa.ForeignKey("dataset_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "class_id",
            UUID(as_uuid=True),
            sa.ForeignKey("annotation_classes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "annotation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("annotations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_geometry", JSONB, nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("created_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_embeddings_org_model", "embeddings", ["organization_id", "model_name"])
    op.create_index("idx_embeddings_dataset_item", "embeddings", ["dataset_item_id"])
    op.create_index("idx_embeddings_class", "embeddings", ["class_id"])
    op.create_index("idx_embeddings_annotation", "embeddings", ["annotation_id"])

    op.execute("ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY embeddings_org_isolation ON embeddings
        USING (organization_id = NULLIF(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS embeddings_org_isolation ON embeddings")
    op.drop_index("idx_embeddings_annotation", table_name="embeddings")
    op.drop_index("idx_embeddings_class", table_name="embeddings")
    op.drop_index("idx_embeddings_dataset_item", table_name="embeddings")
    op.drop_index("idx_embeddings_org_model", table_name="embeddings")
    op.drop_table("embeddings")
