"""
config.py
──────────
Central configuration for all newspaper_ai agents.
All agents import from here instead of reading env vars individually.

Usage:
    from config import NEWSPAPER_NAME, REGION, CHAT_MODEL

    # Activar observabilidad al arrancar (lo llama api/main.py):
    from config import setup_observability
    setup_observability()

To override at runtime, set the corresponding environment variable
before launching any agent or the orchestrator.
"""

import logging
import os

# ── Newspaper identity ────────────────────────────────────────────────────────
NEWSPAPER_NAME = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
REGION         = os.getenv("REGION_NEWS",    "ES")

# ── Model ─────────────────────────────────────────────────────────────────────
# Single place to change the model for ALL agents simultaneously.
# Options: "gemini-2.0-flash" (fast/cheap) | "gemini-2.5-flash" (balanced) | "gemini-2.5-pro" (best)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

# ── Google / Vertex AI ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY",       "")
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION  = os.getenv("GOOGLE_CLOUD_REGION",  "europe-west1")

# ── Storage ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR    = os.getenv("CHROMA_PERSIST_DIR",      "data/embeddings")
CLICKSTREAM_DIR   = os.getenv("CLICKSTREAM_DIR",         "data/clickstream")
ESCALATION_LOG    = os.getenv("ESCALATION_LOG",          "data/escalations.log")
SOCIAL_OUTPUT_DIR = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
ARTICLES_DIR      = os.getenv("ARTICLES_DIR",            "data/articles")

# ── Observabilidad ────────────────────────────────────────────────────────────
# Activar sólo cuando corremos en Google Cloud (GOOGLE_CLOUD_PROJECT detectado).
# En local usa logging estándar sin ningún cambio en los agentes.

_observability_initialized = False


def setup_observability() -> None:
    """
    Configura Cloud Logging y Cloud Trace si estamos en GCloud.
    Llamar UNA sola vez al arrancar (desde api/main.py o el orquestador).

    - Cloud Logging: redirige el módulo estándar `logging` a Cloud Logging.
      Cualquier `logging.info(...)` en cualquier agente aparece automáticamente
      en Google Cloud Console → Logging Explorer.

    - Cloud Trace: instrumenta las requests HTTP de FastAPI con spans.
      Cada llamada a /api/pipeline/run genera un trace completo visible en
      Google Cloud Console → Trace.

    En local (sin GOOGLE_CLOUD_PROJECT), configura solo un StreamHandler
    con formato legible — sin dependencias de GCloud.
    """
    global _observability_initialized
    if _observability_initialized:
        return
    _observability_initialized = True

    if VERTEX_PROJECT:
        _setup_gcloud_logging()
        _setup_gcloud_trace()
    else:
        _setup_local_logging()


def _setup_local_logging() -> None:
    """Logging local — colores y formato legible para desarrollo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("newspaper_ai").info(
        "Observabilidad local activa (Cloud Logging desactivado — GOOGLE_CLOUD_PROJECT no detectado)"
    )


def _setup_gcloud_logging() -> None:
    """
    Redirige todo el logging de Python a Cloud Logging.
    Una vez llamado, logging.info(...) en cualquier agente escribe
    directamente a Cloud Logging sin cambios en el código del agente.
    """
    try:
        import google.cloud.logging as gcl
        client = gcl.Client(project=VERTEX_PROJECT)
        client.setup_logging(log_level=logging.INFO)
        logging.getLogger("newspaper_ai").info(
            "Cloud Logging activado | project=%s", VERTEX_PROJECT
        )
    except ImportError:
        logging.warning(
            "google-cloud-logging no instalado. "
            "Añade 'google-cloud-logging>=3.10.0' a requirements.txt"
        )
    except Exception as e:
        logging.warning("No se pudo activar Cloud Logging: %s", e)


def _setup_gcloud_trace() -> None:
    """
    Instrumenta la aplicación con OpenTelemetry → Cloud Trace.
    Los spans se crean automáticamente para cada request HTTP en FastAPI.
    También puedes crear spans manuales en los agentes:

        from opentelemetry import trace
        tracer = trace.get_tracer("jose_news_research")
        with tracer.start_as_current_span("fetch_trends"):
            ...
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Exporter a Cloud Trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

        # Propagador de contexto para trazar entre servicios
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        provider = TracerProvider(
            sampler=TraceIdRatioBased(1.0),  # 100% de requests trazadas (reducir en alto tráfico)
        )
        provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter(project_id=VERTEX_PROJECT))
        )
        trace.set_tracer_provider(provider)

        # Instrumentación automática de FastAPI (si está instalada)
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument()
        except ImportError:
            pass  # Opcional — los traces manuales seguirán funcionando

        logging.getLogger("newspaper_ai").info(
            "Cloud Trace activado | project=%s | sampling=100%%", VERTEX_PROJECT
        )

    except ImportError:
        logging.warning(
            "opentelemetry-exporter-gcp-trace no instalado. "
            "Añade 'opentelemetry-exporter-gcp-trace>=1.6.0' a requirements.txt"
        )
    except Exception as e:
        logging.warning("No se pudo activar Cloud Trace: %s", e)
