"""
config.py
──────────
Central configuration for all newspaper_ai agents.
All agents import from here instead of reading env vars individually.

Usage:
    from config import NEWSPAPER_NAME, REGION, CHAT_MODEL

To override at runtime, set the corresponding environment variable
before launching any agent or the orchestrator.
"""

import logging
import os

# ── Newspaper identity ────────────────────────────────────────────────────────
NEWSPAPER_NAME = os.getenv("NEWSPAPER_NAME", "Savia")
REGION         = os.getenv("REGION_NEWS",    "ES")

# ── Model ─────────────────────────────────────────────────────────────────────
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# ── Google / Vertex AI ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY",       "")
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION  = os.getenv("GOOGLE_CLOUD_REGION",  "us-central1")

# ── Storage ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR    = os.getenv("CHROMA_PERSIST_DIR",      "data/embeddings")
CLICKSTREAM_DIR   = os.getenv("CLICKSTREAM_DIR",         "data/clickstream")
ESCALATION_LOG    = os.getenv("ESCALATION_LOG",          "data/escalations.log")
SOCIAL_OUTPUT_DIR = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
ARTICLES_DIR      = os.getenv("ARTICLES_DIR",            "data/articles")

# ── Model Armor ───────────────────────────────────────────────────────────────
MODEL_ARMOR_TEMPLATE = os.getenv(
    "MODEL_ARMOR_TEMPLATE",
    f"projects/{VERTEX_PROJECT}/locations/us-central1/templates/savia-template"
)

# ── Observability ─────────────────────────────────────────────────────────────
# Auto-activates on import — no manual call needed.
# Local: standard logging with readable format.
# GCloud: redirects all logging.* calls to Cloud Logging + activates Cloud Trace.


def _setup_local_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("newspaper_ai").info(
        "Local logging active (GOOGLE_CLOUD_PROJECT not detected)"
    )


def _setup_gcloud_logging() -> None:
    try:
        import google.cloud.logging as gcl
        client = gcl.Client(project=VERTEX_PROJECT)
        client.setup_logging(log_level=logging.INFO)
        logging.getLogger("newspaper_ai").info(
            "Cloud Logging active | project=%s", VERTEX_PROJECT
        )
    except ImportError:
        logging.warning(
            "google-cloud-logging not installed. "
            "Add 'google-cloud-logging>=3.10.0' to requirements.txt"
        )
    except Exception as e:
        logging.warning("Could not activate Cloud Logging: %s", e)


def _setup_gcloud_trace() -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        provider = TracerProvider(
            sampler=TraceIdRatioBased(1.0),
        )
        provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter(project_id=VERTEX_PROJECT))
        )
        trace.set_tracer_provider(provider)

        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument()
        except ImportError:
            pass

        logging.getLogger("newspaper_ai").info(
            "Cloud Trace active | project=%s | sampling=100%%", VERTEX_PROJECT
        )

    except ImportError:
        logging.warning(
            "opentelemetry-exporter-gcp-trace not installed. "
            "Add 'opentelemetry-exporter-gcp-trace>=1.6.0' to requirements.txt"
        )
    except Exception as e:
        logging.warning("Could not activate Cloud Trace: %s", e)


try:
    if VERTEX_PROJECT:
        _setup_gcloud_logging()
        _setup_gcloud_trace()
    else:
        _setup_local_logging()
except Exception:
    _setup_local_logging()
