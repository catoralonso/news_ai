"""
tools/search_tools.py
─────────────────────
Herramientas que el News Research Agent puede invocar.

Cada tool es una función Python pura + su descriptor para ADK/LangChain.
Diseñadas para correr 100 % local; la búsqueda web real se activa solo
cuando hay API key de Google Custom Search o SerpAPI disponible.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import requests


# ─────────────────────────────────────────────────────────────────────────────
# 1. Búsqueda web (Google Custom Search API o fallback mock)
# ─────────────────────────────────────────────────────────────────────────────

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "")


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Busca noticias recientes sobre `query`.

    Devuelve lista de:
        {"title": str, "url": str, "snippet": str, "source": str}

    Si no hay API key configurada → devuelve resultados mock para desarrollo.
    """
    if GOOGLE_CSE_KEY and GOOGLE_CSE_CX:
        return _google_cse_search(query, num_results)
    else:
        return _mock_search(query, num_results)


def _google_cse_search(query: str, num: int) -> list[dict]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": min(num, 10),
        "dateRestrict": "d7",   # últimos 7 días
        "lr": "lang_es",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {
            "title":   i.get("title", ""),
            "url":     i.get("link", ""),
            "snippet": i.get("snippet", ""),
            "source":  i.get("displayLink", ""),
        }
        for i in items
    ]


def _mock_search(query: str, num: int) -> list[dict]:
    """Resultados ficticios para desarrollo offline."""
    today = datetime.now().strftime("%d/%m/%Y")
    return [
        {
            "title":   f"[MOCK] Tendencia: {query} — perspectiva local",
            "url":     "https://example.com/mock-1",
            "snippet": f"Análisis local sobre '{query}'. Expertos señalan impacto en comunidades cercanas.",
            "source":  "mock-news.local",
        },
        {
            "title":   f"[MOCK] Cobertura regional: {query} ({today})",
            "url":     "https://example.com/mock-2",
            "snippet": f"Residentes reaccionan ante los últimos desarrollos sobre '{query}'.",
            "source":  "mock-regional.local",
        },
    ][:num]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extractor de tendencias (Google Trends vía pytrends — local, sin API key)
# ─────────────────────────────────────────────────────────────────────────────

def get_trending_topics(region: str = "MX", category: str = "news") -> list[str]:
    """
    Obtiene temas en tendencia para la región dada.
    Usa pytrends (scraping libre de Google Trends).
    Fallback a lista mock si pytrends no está instalado.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
        pt = TrendReq(hl="es-MX", tz=360)
        df = pt.trending_searches(pn="mexico")
        return df[0].tolist()[:10]
    except Exception:
        return [
            "presupuesto municipal 2025",
            "transporte público reforma",
            "seguridad ciudadana estadísticas",
            "elecciones locales candidatos",
            "cultura festival próximo",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clasificador de relevancia local
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_KEYWORDS = [
    "municipio", "alcaldía", "ayuntamiento", "vecinos", "colonia",
    "barrio", "delegación", "presidente municipal", "cabildo",
    "parque", "mercado", "transporte local", "fiestas patronales",
]


def score_local_relevance(text: str) -> float:
    """
    Puntúa qué tan relevante es un texto para un periódico local.
    Retorna float en [0, 1].
    """
    text_lower = text.lower()
    hits = sum(1 for kw in LOCAL_KEYWORDS if kw in text_lower)
    return min(hits / 5.0, 1.0)   # normalizado: 5+ keywords → score 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Descriptores para ADK / LangChain function-calling
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Busca noticias recientes en la web sobre un tema. "
            "Úsala cuando necesites información actual que no esté en la base de conocimiento."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Términos de búsqueda"},
                "num_results": {"type": "integer", "description": "Número de resultados (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_trending_topics",
        "description": "Obtiene los temas más buscados/trending en la región para detectar oportunidades de cobertura.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Código de país ISO (MX, ES, AR…)", "default": "MX"},
            },
            "required": [],
        },
    },
    {
        "name": "score_local_relevance",
        "description": "Evalúa qué tan relevante es un texto para cobertura local (0-1).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Texto a evaluar"},
            },
            "required": ["text"],
        },
    },
]


# Dispatch map para el agente (nombre → función)
TOOL_DISPATCH: dict[str, Any] = {
    "web_search":            web_search,
    "get_trending_topics":   get_trending_topics,
    "score_local_relevance": score_local_relevance,
}
