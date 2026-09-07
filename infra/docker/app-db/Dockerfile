# =============================================================================
# app-db image — postgis/postgis base + pgvector
# =============================================================================
# Extends the existing PostGIS base (do not swap it) with the pgvector
# extension so the app DB can hold embedding columns alongside its existing
# PostGIS geometry columns. Everything else about the base image (roles,
# init scripts, data volume) is unchanged.
#
# The base image already ships a configured PGDG apt repo
# (/etc/apt/sources.list.d/pgdg.list) — that's how it installs
# postgresql-17-postgis-3 itself — so pgvector's package is just one more
# `apt-get install` away, no new repo/key setup needed.
# =============================================================================
FROM postgis/postgis:17-3.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-pgvector \
    && rm -rf /var/lib/apt/lists/*
