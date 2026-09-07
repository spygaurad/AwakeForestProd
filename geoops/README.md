# geoops

Embedding bank + similarity search for class objects (bboxes/masks selected on
a dataset item). Self-contained module — the rest of `app/` only touches it at
three points:

1. `app/api/v1/router.py` mounts `geoops.api.router`.
2. `app/db/_all_models.py` imports `geoops.models` so Alembic sees the
   `embeddings` table in `Base.metadata`.
3. Migrations `049`/`050` in `alembic/versions/` create the `vector` extension
   and the `embeddings` table.

Everything else — the ORM model, schemas, and service logic — lives in this
package and only reaches into `app/` for things it has no reason to
duplicate: the DB `Base`, the `AIModel`/`DatasetItem`/`Annotation`/
`AnnotationClass` models it references by FK, the RLS-aware session
dependency, and `app.services.titiler_service.get_item_bbox_preview` for
cropping a patch image out of a STAC item.

## Model

Each `ai_models` row can be an *embedding* model — same table used for
inference models, just pointed at an endpoint that returns
`{model_name, embedding_dim, embedding}` instead of predictions. Nothing new
to configure there.

## Flow

- `POST /api/v1/embeddings` — crop the selected bbox/mask out of a dataset
  item, POST it to the embedding model's endpoint, store the returned vector.
  Synchronous (single object, low latency) — not a bulk job.
- `POST /api/v1/embeddings/search` — cosine nearest-neighbours against the
  embedding bank, scoped to the reference's `model_name` (vectors from
  different models are never compared against each other) and org.
- `POST /api/v1/embeddings/aoi-scan` — tile an AOI at the *same geographic
  scale* as the reference embedding's own source geometry, embed each tile,
  rank by cosine similarity to the reference. View-only: nothing is written
  to the DB, the response is the result.

## v1 scope, on purpose

No ANN index (HNSW/IVFFlat), no async job for the AOI scan, no persistence of
scan results. The `embedding` column is unconstrained (`vector`, no fixed
dimension) because different embedding models can return different
dimensions — every query filters by `model_name` first so cosine distance
never compares mismatched dimensions. Revisit once one model/dimension is
the standard and volume justifies an index.
