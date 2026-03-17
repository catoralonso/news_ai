# ─────────────────────────────────────────────────────────────────────────────
# newspaper_ai — Dockerfile
# Builds a single Cloud Run image that exposes the FastAPI gateway.
# All agents run as Python modules within the same process.
#
# Build:
#   docker build -t newspaper-ai .
#
# Run locally:
#   docker run --env-file .env -p 8080:8080 newspaper-ai
#
# Deploy to Cloud Run:
#   gcloud run deploy newspaper-ai \
#     --image gcr.io/YOUR_PROJECT/newspaper-ai \
#     --platform managed \
#     --region europe-west1 \
#     --allow-unauthenticated \
#     --set-secrets GEMINI_API_KEY=gemini-api-key:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps needed to compile some Python packages (chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
        fastapi>=0.111.0 \
        uvicorn[standard]>=0.29.0 \
        google-cloud-logging>=3.10.0 \
        google-cloud-trace>=1.13.0 \
        opentelemetry-sdk>=1.24.0 \
        opentelemetry-exporter-gcp-trace>=1.6.0


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Create data directories that agents write to at runtime
# In Cloud Run these are ephemeral; for persistence use GCS or Firestore
RUN mkdir -p \
    data/embeddings \
    data/clickstream \
    data/trends \
    data/social_media_output \
    data/articles

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser /app
USER appuser

# Health check (Cloud Run will also do its own)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Entry point: FastAPI via uvicorn
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info"]
