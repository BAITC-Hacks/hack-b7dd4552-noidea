FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy DATA_DIR=/app/data/workspace
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev
COPY samples/ ./samples/
COPY --from=frontend /build/dist ./frontend/dist
RUN useradd --create-home app && mkdir /app/data && chown app:app /app/data
USER app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "wind.app:app", "--host", "0.0.0.0", "--port", "8000"]
