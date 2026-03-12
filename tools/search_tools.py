"""
tools/search_tools.py
─────────────────────
Herramientas que el News Research Agent puede invocar.

Cada tool es una función Python pura + su descriptor para ADK/LangChain.
Diseñadas para correr 100 % local; la búsqueda web real se activa solo
cuando hay API key de Google Custom Search o SerpAPI disponible.

Clickstream:
    El sitio web del periódico (aún no construido) escribirá eventos en
    data/clickstream/events.jsonl con este formato por línea:
    {
      "article_id":   "nueva-ruta-transporte-2025-02-15",
      "title":        "Nueva ruta de transporte...",
      "category":     "comunidad",
      "event":        "read" | "click" | "scroll",
      "duration_sec": 142,      # solo en event="read"
      "scroll_pct":   87,       # solo en event="scroll" (0-100)
      "timestamp":    "2025-03-10T14:32:00"
    }
    Mientras el sitio no exista, get_clickstream_insights() genera datos
    mock realistas para que José pueda usarlo desde ahora.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
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
# 4. Clickstream — comportamiento real de lectores en el sitio
# ─────────────────────────────────────────────────────────────────────────────

CLICKSTREAM_DIR  = os.getenv("CLICKSTREAM_DIR", "data/clickstream")
CLICKSTREAM_FILE = os.path.join(CLICKSTREAM_DIR, "events.jsonl")


def log_event(
    article_id: str,
    title: str,
    category: str,
    event: str,                     # "read" | "click" | "scroll"
    duration_sec: int = 0,          # segundos leyendo (event="read")
    scroll_pct: int = 0,            # % de scroll completado (event="scroll")
) -> None:
    """
    Registra un evento de comportamiento del lector.
    Llamado desde el sitio web cuando se construya.

    El archivo events.jsonl crece con una línea JSON por evento.
    """
    Path(CLICKSTREAM_DIR).mkdir(parents=True, exist_ok=True)
    entry = {
        "article_id":   article_id,
        "title":        title,
        "category":     category,
        "event":        event,
        "duration_sec": duration_sec,
        "scroll_pct":   scroll_pct,
        "timestamp":    datetime.now().isoformat(),
    }
    with open(CLICKSTREAM_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_clickstream_insights(days: int = 7) -> dict:
    """
    Lee events.jsonl y devuelve métricas de engagement por categoría
    y por artículo para los últimos `days` días.

    Retorna:
    {
      "by_category": {
        "deportes":   {"clicks": 120, "avg_read_sec": 187, "avg_scroll_pct": 82},
        "comunidad":  {"clicks": 95,  "avg_read_sec": 143, "avg_scroll_pct": 71},
        ...
      },
      "top_articles": [
        {"title": "...", "category": "...", "clicks": 47, "avg_read_sec": 210},
        ...
      ],
      "total_events": 312,
      "period_days":  7,
      "source": "real" | "mock"
    }
    """
    if os.path.exists(CLICKSTREAM_FILE):
        return _parse_clickstream(days)
    else:
        return _mock_clickstream_insights()


def _parse_clickstream(days: int) -> dict:
    """Procesa el archivo real de eventos."""
    cutoff = datetime.now() - timedelta(days=days)

    by_category: dict[str, dict] = defaultdict(lambda: {
        "clicks": 0, "read_times": [], "scroll_pcts": []
    })
    by_article: dict[str, dict] = defaultdict(lambda: {
        "title": "", "category": "", "clicks": 0, "read_times": []
    })

    total = 0
    with open(CLICKSTREAM_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            ts = datetime.fromisoformat(e["timestamp"])
            if ts < cutoff:
                continue

            total += 1
            cat = e.get("category", "general")
            aid = e.get("article_id", "")
            evt = e.get("event", "")

            if evt == "click":
                by_category[cat]["clicks"] += 1
                by_article[aid]["clicks"] += 1
                by_article[aid]["title"] = e.get("title", "")
                by_article[aid]["category"] = cat

            elif evt == "read" and e.get("duration_sec", 0) > 0:
                by_category[cat]["read_times"].append(e["duration_sec"])
                by_article[aid]["read_times"].append(e["duration_sec"])

            elif evt == "scroll" and e.get("scroll_pct", 0) > 0:
                by_category[cat]["scroll_pcts"].append(e["scroll_pct"])

    # Calcular promedios
    def _avg(lst): return round(sum(lst) / len(lst)) if lst else 0

    summary_by_cat = {
        cat: {
            "clicks":        data["clicks"],
            "avg_read_sec":  _avg(data["read_times"]),
            "avg_scroll_pct": _avg(data["scroll_pcts"]),
        }
        for cat, data in by_category.items()
    }

    top_articles = sorted(
        [
            {
                "title":        data["title"],
                "category":     data["category"],
                "clicks":       data["clicks"],
                "avg_read_sec": _avg(data["read_times"]),
            }
            for aid, data in by_article.items()
        ],
        key=lambda x: (x["clicks"], x["avg_read_sec"]),
        reverse=True,
    )[:5]

    return {
        "by_category":  summary_by_cat,
        "top_articles": top_articles,
        "total_events": total,
        "period_days":  days,
        "source":       "real",
    }


def _mock_clickstream_insights() -> dict:
    """
    Datos mock realistas para cuando el sitio aún no existe.
    Simula una semana típica de un periódico local pequeño.
    """
    return {
        "by_category": {
            "deportes":  {"clicks": 312, "avg_read_sec": 198, "avg_scroll_pct": 84},
            "comunidad": {"clicks": 287, "avg_read_sec": 231, "avg_scroll_pct": 79},
            "política":  {"clicks": 201, "avg_read_sec": 176, "avg_scroll_pct": 65},
            "cultura":   {"clicks": 143, "avg_read_sec": 154, "avg_scroll_pct": 58},
            "economía":  {"clicks": 98,  "avg_read_sec": 142, "avg_scroll_pct": 51},
        },
        "top_articles": [
            {
                "title":        "Leones de San Cristóbal avanzan a semifinales",
                "category":     "deportes",
                "clicks":       147,
                "avg_read_sec": 214,
            },
            {
                "title":        "Nueva ruta de transporte conectará norte y sur",
                "category":     "comunidad",
                "clicks":       134,
                "avg_read_sec": 243,
            },
            {
                "title":        "Alcalde presenta presupuesto municipal 2025",
                "category":     "política",
                "clicks":       98,
                "avg_read_sec": 189,
            },
        ],
        "total_events": 1041,
        "period_days":  7,
        "source":       "mock",
    }


def format_insights_for_prompt(insights: dict) -> str:
    """
    Convierte el dict de insights en texto legible para inyectar
    en el prompt de José.
    """
    lines = [
        f"COMPORTAMIENTO DE LECTORES (últimos {insights['period_days']} días)"
        f" [{insights['source'].upper()}]:",
        "",
        "Engagement por categoría:",
    ]
    for cat, data in sorted(
        insights["by_category"].items(),
        key=lambda x: x[1]["clicks"],
        reverse=True,
    ):
        read_min = data["avg_read_sec"] // 60
        read_sec = data["avg_read_sec"] % 60
        lines.append(
            f"  • {cat}: {data['clicks']} clicks | "
            f"lectura promedio {read_min}m{read_sec:02d}s | "
            f"scroll {data['avg_scroll_pct']}%"
        )

    lines += ["", "Artículos más leídos esta semana:"]
    for i, art in enumerate(insights["top_articles"], 1):
        lines.append(f"  {i}. [{art['category']}] {art['title']} — {art['clicks']} clicks")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Descriptores para ADK / LangChain function-calling
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
    {
        "name": "get_clickstream_insights",
        "description": (
            "Analiza el comportamiento real de los lectores en el sitio del periódico. "
            "Devuelve qué categorías y artículos generan más clicks, tiempo de lectura "
            "y scroll. Úsala para priorizar temas con alta demanda probada de la audiencia."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Ventana de días a analizar (default: 7)", "default": 7},
            },
            "required": [],
        },
    },
]


# Dispatch map para el agente (nombre → función)
TOOL_DISPATCH: dict[str, Any] = {
    "web_search":               web_search,
    "get_trending_topics":      get_trending_topics,
    "score_local_relevance":    score_local_relevance,
    "get_clickstream_insights": get_clickstream_insights,
}
