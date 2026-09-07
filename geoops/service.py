"""Embedding generation + similarity search.

Reaches into ``app/`` only for what it has no reason to duplicate: the DB
models it references by FK (``AIModel``, ``DatasetItem``, ``Annotation``,
``AnnotationClass``, ``AnnotationSchema``), and
``app.services.titiler_service.get_item_bbox_preview`` to crop a patch image
out of a STAC item — the same TiTiler bbox-crop path
``app.services.model_manager`` uses for inference patches.
"""
from __future__ import annotations

import base64
import logging
from typing import Any
from uuid import UUID

import httpx
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import bad_request, not_found
from app.models.ai_model import AIModel
from app.models.annotation import Annotation
from app.models.annotation_class import AnnotationClass
from app.models.annotation_schema import AnnotationSchema
from app.models.annotation_set import AnnotationSet
from app.models.dataset_item import DatasetItem
from app.services.titiler_service import get_item_bbox_preview

from geoops.models import Embedding
from geoops.schemas import (
    AOIScanMatch,
    AOIScanRequest,
    AOIScanResponse,
    EmbeddingCreateRequest,
    EmbeddingSearchMatch,
    EmbeddingSearchRequest,
)

logger = logging.getLogger(__name__)


def _bbox_of(geometry: dict[str, Any]) -> list[float]:
    minx, miny, maxx, maxy = shape(geometry).bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _intersect(a: list[float], b: list[float]) -> list[float] | None:
    minx, miny = max(a[0], b[0]), max(a[1], b[1])
    maxx, maxy = min(a[2], b[2]), min(a[3], b[3])
    if minx >= maxx or miny >= maxy:
        return None
    return [minx, miny, maxx, maxy]


class EmbeddingService:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ── shared lookups ───────────────────────────────────────────────────

    async def _get_model(self, org_id: UUID, model_id: UUID) -> AIModel:
        model = await self.session.scalar(
            select(AIModel).where(
                AIModel.id == model_id,
                AIModel.organization_id == org_id,
                AIModel.deleted_at.is_(None),
            )
        )
        if model is None:
            raise not_found("Model")
        if not model.endpoint_url:
            raise bad_request("Model has no endpoint_url configured")
        return model

    async def _get_dataset_item(self, org_id: UUID, dataset_item_id: UUID) -> DatasetItem:
        item = await self.session.scalar(
            select(DatasetItem).where(
                DatasetItem.id == dataset_item_id,
                DatasetItem.organization_id == org_id,
                DatasetItem.is_active.is_(True),
            )
        )
        if item is None:
            raise not_found("Dataset item")
        return item

    async def get_embedding(self, org_id: UUID, embedding_id: UUID) -> Embedding:
        return await self._get_embedding(org_id, embedding_id)

    async def _get_embedding(self, org_id: UUID, embedding_id: UUID) -> Embedding:
        row = await self.session.scalar(
            select(Embedding).where(
                Embedding.id == embedding_id,
                Embedding.organization_id == org_id,
            )
        )
        if row is None:
            raise not_found("Embedding")
        return row

    async def _resolve_class_and_annotation(
        self, org_id: UUID, annotation_id: UUID | None, class_id: UUID | None
    ) -> tuple[UUID | None, UUID | None]:
        """Returns (class_id, annotation_id), preferring the annotation's own class."""
        if annotation_id is not None:
            annotation = await self.session.scalar(
                select(Annotation)
                .join(AnnotationSet, AnnotationSet.id == Annotation.annotation_set_id)
                .where(Annotation.id == annotation_id, AnnotationSet.organization_id == org_id)
            )
            if annotation is None:
                raise not_found("Annotation")
            return annotation.class_id, annotation.id

        if class_id is not None:
            cls = await self.session.scalar(
                select(AnnotationClass)
                .join(AnnotationSchema, AnnotationSchema.id == AnnotationClass.schema_id)
                .where(AnnotationClass.id == class_id, AnnotationSchema.organization_id == org_id)
            )
            if cls is None:
                raise not_found("Annotation class")
            return cls.id, None

        return None, None

    # ── embed model call ─────────────────────────────────────────────────

    async def _crop_png_b64(self, item: DatasetItem, bbox: list[float], size_px: int) -> str:
        image_bytes = await get_item_bbox_preview(
            item.stac_collection_id, item.stac_item_id, bbox=bbox, width=size_px, height=size_px
        )
        if not image_bytes:
            raise bad_request("Empty patch image from TiTiler")
        return base64.b64encode(image_bytes).decode("ascii")

    async def _call_embed_model(
        self, model: AIModel, item: DatasetItem, patch_image_b64: str, bbox: list[float]
    ) -> dict[str, Any]:
        body = {
            "dataset_item_id": str(item.id),
            "stac_item_id": item.stac_item_id,
            "bbox": bbox,
            "patch_image_format": "png",
            "patch_image_base64": patch_image_b64,
        }
        req_cfg = model.request_config or {}
        if isinstance(req_cfg.get("payload"), dict):
            body.update(req_cfg["payload"])

        headers = {"Content-Type": "application/json"}
        token = (model.auth_config or {}).get("bearer_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        timeout = float(req_cfg.get("timeout_seconds", 60))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                str(req_cfg.get("method", "POST")).upper(), model.endpoint_url, json=body, headers=headers
            )
        if resp.status_code != 200:
            logger.error("embed_model_request_failed model_id=%s status=%s body=%s", model.id, resp.status_code, resp.text[:500])
            raise bad_request(f"Embed model request failed: HTTP {resp.status_code}")

        data = resp.json()
        embedding = data.get("embedding")
        embedding_dim = data.get("embedding_dim")
        model_name = data.get("model_name")
        if not isinstance(embedding, list) or not embedding:
            raise bad_request("Embed model response missing a non-empty 'embedding' list")
        if not isinstance(model_name, str) or not model_name:
            raise bad_request("Embed model response missing 'model_name'")
        if not isinstance(embedding_dim, int) or embedding_dim != len(embedding):
            raise bad_request("Embed model response 'embedding_dim' does not match embedding length")
        return {"model_name": model_name, "embedding_dim": embedding_dim, "embedding": embedding}

    # ── create ────────────────────────────────────────────────────────────

    async def create_embedding(
        self, org_id: UUID, user_id: UUID | None, payload: EmbeddingCreateRequest
    ) -> Embedding:
        model = await self._get_model(org_id, payload.model_id)
        item = await self._get_dataset_item(org_id, payload.dataset_item_id)
        class_id, annotation_id = await self._resolve_class_and_annotation(
            org_id, payload.annotation_id, payload.class_id
        )

        bbox = _bbox_of(payload.geometry)
        patch_b64 = await self._crop_png_b64(item, bbox, payload.crop_size_px)
        embed_result = await self._call_embed_model(model, item, patch_b64, bbox)

        row = Embedding(
            organization_id=org_id,
            model_id=model.id,
            model_name=embed_result["model_name"],
            embedding_dim=embed_result["embedding_dim"],
            dataset_item_id=item.id,
            class_id=class_id,
            annotation_id=annotation_id,
            source_geometry=payload.geometry,
            embedding=embed_result["embedding"],
            created_by_user_id=user_id,
        )
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        return row

    # ── search over the bank ─────────────────────────────────────────────

    async def search(self, org_id: UUID, payload: EmbeddingSearchRequest) -> list[EmbeddingSearchMatch]:
        reference = await self._get_embedding(org_id, payload.embedding_id)

        distance_expr = Embedding.embedding.cosine_distance(reference.embedding)
        query = select(Embedding, distance_expr.label("distance")).where(
            Embedding.organization_id == org_id,
            Embedding.model_name == reference.model_name,
        )
        if payload.class_id is not None:
            query = query.where(Embedding.class_id == payload.class_id)
        if payload.dataset_id is not None:
            query = query.join(DatasetItem, DatasetItem.id == Embedding.dataset_item_id).where(
                DatasetItem.dataset_id == payload.dataset_id
            )
        if payload.exclude_self:
            query = query.where(Embedding.id != reference.id)

        query = query.order_by(distance_expr.asc()).limit(payload.top_k)
        rows = (await self.session.execute(query)).all()

        return [
            EmbeddingSearchMatch(
                embedding_id=row.Embedding.id,
                dataset_item_id=row.Embedding.dataset_item_id,
                class_id=row.Embedding.class_id,
                annotation_id=row.Embedding.annotation_id,
                source_geometry=row.Embedding.source_geometry,
                similarity=1.0 - float(row.distance),
                distance=float(row.distance),
            )
            for row in rows
        ]

    # ── AOI scan (view-only, nothing persisted) ──────────────────────────

    def _tile_grid(self, item_bbox: list[float], tile_w: float, tile_h: float, stride_w: float, stride_h: float) -> list[list[float]]:
        minx, miny, maxx, maxy = item_bbox
        tiles: list[list[float]] = []
        y = miny
        while y < maxy:
            x = minx
            top = min(y + tile_h, maxy)
            while x < maxx:
                right = min(x + tile_w, maxx)
                tiles.append([x, y, right, top])
                if right >= maxx:
                    break
                x += stride_w
            if top >= maxy:
                break
            y += stride_h
        return tiles

    async def aoi_scan(self, org_id: UUID, payload: AOIScanRequest) -> AOIScanResponse:
        reference = await self._get_embedding(org_id, payload.reference_embedding_id)
        model = await self._get_model(org_id, reference.model_id) if reference.model_id else None
        if model is None:
            raise bad_request("Reference embedding's model no longer exists")

        ref_minx, ref_miny, ref_maxx, ref_maxy = _bbox_of(reference.source_geometry)
        tile_w = max((ref_maxx - ref_minx) * payload.scale_factor, 1e-6)
        tile_h = max((ref_maxy - ref_miny) * payload.scale_factor, 1e-6)
        stride_w = max(tile_w * (1.0 - payload.overlap), tile_w * 0.1)
        stride_h = max(tile_h * (1.0 - payload.overlap), tile_h * 0.1)

        items = [await self._get_dataset_item(org_id, item_id) for item_id in payload.dataset_item_ids]

        all_patches: list[tuple[DatasetItem, list[float]]] = []
        for item in items:
            item_bbox = _bbox_of(item.geometry) if item.geometry else None
            if item_bbox is None:
                continue
            clip_bbox = _intersect(item_bbox, payload.aoi_bbox) if payload.aoi_bbox else item_bbox
            if clip_bbox is None:
                continue
            for tile_bbox in self._tile_grid(clip_bbox, tile_w, tile_h, stride_w, stride_h):
                all_patches.append((item, tile_bbox))

        if len(all_patches) > payload.max_patches:
            raise bad_request(
                f"AOI scan would produce {len(all_patches)} patches, over the max_patches "
                f"limit of {payload.max_patches}. Narrow the AOI, raise scale_factor, or "
                f"reduce overlap."
            )

        matches: list[AOIScanMatch] = []
        for item, tile_bbox in all_patches:
            try:
                patch_b64 = await self._crop_png_b64(item, tile_bbox, payload.crop_size_px)
                embed_result = await self._call_embed_model(model, item, patch_b64, tile_bbox)
            except Exception as exc:  # noqa: BLE001 — one bad patch shouldn't fail the whole scan
                logger.warning("aoi_scan_patch_failed dataset_item_id=%s bbox=%s error=%s", item.id, tile_bbox, exc)
                continue

            if embed_result["model_name"] != reference.model_name:
                continue

            distance = _cosine_distance(embed_result["embedding"], reference.embedding)
            similarity = 1.0 - distance
            if payload.min_similarity is not None and similarity < payload.min_similarity:
                continue
            matches.append(
                AOIScanMatch(dataset_item_id=item.id, bbox=tile_bbox, similarity=similarity, distance=distance)
            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return AOIScanResponse(
            reference_embedding_id=reference.id,
            model_name=reference.model_name,
            patches_scanned=len(all_patches),
            matches=matches[: payload.top_k],
        )


def _cosine_distance(a: list[float], b: Any) -> float:
    b_list = list(b)
    if len(a) != len(b_list):
        raise bad_request("Embedding dimension mismatch between patch and reference")
    dot = sum(x * y for x, y in zip(a, b_list))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b_list) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))
