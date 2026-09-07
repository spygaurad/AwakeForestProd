"""Enable the pgvector extension.

Infra-only migration — no columns/indexes created here. The app-db image
(``infra/docker/app-db/Dockerfile``) now installs the ``postgresql-17-pgvector``
package on top of the existing PostGIS base; this just turns the extension on
inside the ``geoplat`` database.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-09-07
"""

from alembic import op

revision = "d0e1f2a3b4c5"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
