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
VERTEX_REGION  = os.getenv("GOOGLE_CLOUD_REGION",  "us-central1")

# ── Storage ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR   = os.getenv("CHROMA_PERSIST_DIR", "data/embeddings")
CLICKSTREAM_DIR  = os.getenv("CLICKSTREAM_DIR",    "data/clickstream")
ESCALATION_LOG   = os.getenv("ESCALATION_LOG",     "data/escalations.log")
SOCIAL_OUTPUT_DIR = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
