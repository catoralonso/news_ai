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
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-haiku-4-5-20251001")

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Storage ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR    = os.getenv("CHROMA_PERSIST_DIR",      "data/embeddings")
CLICKSTREAM_DIR   = os.getenv("CLICKSTREAM_DIR",         "data/clickstream")
ESCALATION_LOG    = os.getenv("ESCALATION_LOG",          "data/escalations.log")
SOCIAL_OUTPUT_DIR = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
ARTICLES_DIR      = os.getenv("ARTICLES_DIR",            "data/articles")

# ── Observability ─────────────────────────────────────────────────────────────
# Auto-activates on import — no manual call needed.
# Local: standard logging with readable format.

def _setup_local_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("newspaper_ai").info("Local logging active")
try:
    _setup_local_logging()
except Exception:
    pass
