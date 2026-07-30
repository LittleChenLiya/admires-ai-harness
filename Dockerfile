# syntax=docker/dockerfile:1.7
FROM node:24-alpine AS web-builder
WORKDIR /build/web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="Admire's AI Harness" \
      org.opencontainers.image.description="Safe-by-default AI agent runtime and observability console" \
      org.opencontainers.image.source="https://github.com/LittleChenLiya/admires-ai-harness" \
      org.opencontainers.image.licenses="MIT"
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN groupadd --system --gid 10001 harness \
    && useradd --system --uid 10001 --gid harness --home-dir /nonexistent --shell /usr/sbin/nologin harness
RUN install --directory --owner=harness --group=harness --mode=0750 /data
COPY --from=python-builder --chown=harness:harness /app/.venv .venv
COPY --from=python-builder --chown=harness:harness /app/src src
COPY --from=web-builder --chown=harness:harness /build/web/dist web/dist
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["admire"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
