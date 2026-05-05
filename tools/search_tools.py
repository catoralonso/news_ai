"""
tools/search_tools.py
─────────────────────
Herramientas que el News Research Agent puede invocar.

Cada tool es una función Python pura + su descriptor para ADK/LangChain.
Diseñadas para correr 100 % local; sin dependencias de GCloud.

Búsqueda web:
    Fuente principal → RSS de PubMed + Healthline
    Fallback         → mock para desarrollo offline

Tendencias:
    Fuente principal → CSV de Google Trends (manual y local)
    Fallback         → mock para desarrollo offline

Clickstream:
    El sitio web del periódico escribirá eventos en
    data/clickstream/events.jsonl con este formato por línea:
    {
      "article_id":   "dieta-mediterranea-2025-02-15",
      "title":        "Dieta mediterránea y diabetes tipo 2",
      "category":     "enfermedades y dieta",
      "event":        "read" | "click" | "scroll",
      "duration_sec": 142,      # solo en event="read"
      "scroll_pct":   87,       # solo en event="scroll" (0-100)
      "timestamp":    "2025-03-10T14:32:00"
    }
    Mientras el sitio no exista, get_clickstream_insights() genera datos
    mock realistas para que José pueda usarlo desde ahora.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser  # pip install feedparser
import requests


# ─────────────────────────────────────────────────────────────────────────────
# 1. Búsqueda web (RSS de PubMed + Healthline → fallback mock)
# ─────────────────────────────────────────────────────────────────────────────

RSS_SOURCES = {
    "pubmed": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term={query}&format=rss",
    "healthline": "https://www.healthline.com/rss/health-news",
}


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Busca artículos recientes de nutrición y salud.

    Fuente principal: RSS de PubMed + Healthline (gratis, sin API key).
    Fallback:         mock para desarrollo offline.

    Devuelve lista de:
        {"title": str, "url": str, "snippet": str, "source": str}
    """
    results = _rss_search(query, num_results)
    if results:
        return results
    return _mock_search(query, num_results)


def _rss_search(query: str, num: int) -> list[dict]:
    """
    Busca en RSS de PubMed (query-specific) y Healthline (feed general).
    Devuelve lista vacía si ambas fuentes fallan — web_search cae al mock.
    """
    results: list[dict] = []

    # PubMed — RSS específico por query
    try:
        pubmed_url = RSS_SOURCES["pubmed"].format(query=requests.utils.quote(query))
        feed = feedparser.parse(pubmed_url)
        for entry in feed.entries[:num]:
            results.append({
                "title":   entry.get("title", ""),
                "url":     entry.get("link", ""),
                "snippet": entry.get("summary", "")[:200],
                "source":  "PubMed",
            })
    except Exception:
        pass

    # Healthline — feed general, filtramos por query en título/resumen
    if len(results) < num:
        try:
            feed = feedparser.parse(RSS_SOURCES["healthline"])
            query_lower = query.lower()
            for entry in feed.entries:
                text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                if query_lower in text:
                    results.append({
                        "title":   entry.get("title", ""),
                        "url":     entry.get("link", ""),
                        "snippet": entry.get("summary", "")[:200],
                        "source":  "Healthline",
                    })
                if len(results) >= num:
                    break
        except Exception:
            pass

    return results[:num]


def _mock_search(query: str, num: int) -> list[dict]:
    """Resultados ficticios para desarrollo offline."""
    today = datetime.now().strftime("%d/%m/%Y")
    return [
        {
            "title":   f"[MOCK] Evidencia científica: {query}",
            "url":     "https://example.com/mock-1",
            "snippet": f"Estudio reciente sobre '{query}'. Expertos en nutrición señalan nuevos hallazgos.",
            "source":  "mock-nutrition.local",
        },
        {
            "title":   f"[MOCK] Guía práctica: {query} ({today})",
            "url":     "https://example.com/mock-2",
            "snippet": f"Recomendaciones dietéticas basadas en evidencia sobre '{query}'.",
            "source":  "mock-nutrition.local",
        },
    ][:num]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tendencias (CSV de Google Trends → fallback mock)
# ─────────────────────────────────────────────────────────────────────────────

TRENDS_CSV = os.getenv("TRENDS_CSV_PATH", "data/trends/trends.csv")

def get_trending_topics(region: str = "ES") -> list[str]:
    """
    Obtiene temas en tendencia de nutrición.

    Fuente principal: CSV exportado manualmente de Google Trends.
    Fallback:         mock para desarrollo offline.

    El parámetro `region` se mantiene para cuando el orchestrador lo pase,
    aunque el CSV ya viene filtrado por región desde Google Trends.
    """
    if os.path.exists(TRENDS_CSV):
        return _read_trends_csv(TRENDS_CSV)
    return _mock_trending_topics()


def _read_trends_csv(path: str) -> list[str]:
    """
    Lee el CSV exportado de Google Trends.
    TODO: ajustar el nombre de columna cuando el compañero suba el CSV real.

    Formato esperado (Google Trends export):
        Término de búsqueda, Valor
        proteínas y músculo,  100
        dieta cetogénica,      85
        ...
    """
    topics: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # TODO: confirmar nombre exacto de columna con el compañero
                term = row.get("Término de búsqueda") or row.get("term") or ""
                if term:
                    topics.append(term.strip())
    except Exception:
        pass
    return topics[:10]


def _mock_trending_topics() -> list[str]:
    return [
        "proteínas y pérdida de peso",
        "dieta cetogénica efectos",
        "suplementos omega-3 beneficios",
        "alimentación antiinflamatoria",
        "microbiota intestinal salud",
        "ayuno intermitente evidencia",
        "vitamina D deficiencia",
        "dieta mediterránea longevidad",
        "azúcar y enfermedades crónicas",
        "superalimentos mitos y realidad",
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Clickstream — comportamiento real de lectores en el sitio
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
    Llamado desde el sitio web del diario cuando se construya.

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
        "recetas":         {"clicks": 312, "avg_read_sec": 198, "avg_scroll_pct": 84},
        "pérdida de peso": {"clicks": 287, "avg_read_sec": 231, "avg_scroll_pct": 79},
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

    def _avg(lst): return round(sum(lst) / len(lst)) if lst else 0

    summary_by_cat = {
        cat: {
            "clicks":         data["clicks"],
            "avg_read_sec":   _avg(data["read_times"]),
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
    Simula una semana típica del periódico de nutrición.
    TODO: actualizar categorías cuando el Content Engineer defina las oficiales.
    """
    return {
        "by_category": {
            "recetas":              {"clicks": 312, "avg_read_sec": 198, "avg_scroll_pct": 84},
            "pérdida de peso":      {"clicks": 287, "avg_read_sec": 231, "avg_scroll_pct": 79},
            "suplementos":          {"clicks": 201, "avg_read_sec": 176, "avg_scroll_pct": 65},
            "ciencia y evidencia":  {"clicks": 143, "avg_read_sec": 154, "avg_scroll_pct": 58},
            "enfermedades y dieta": {"clicks": 98,  "avg_read_sec": 142, "avg_scroll_pct": 51},
        },
        "top_articles": [
            {
                "title":        "10 alimentos que aceleran el metabolismo",
                "category":     "pérdida de peso",
                "clicks":       147,
                "avg_read_sec": 214,
            },
            {
                "title":        "Omega-3: qué dice la evidencia científica",
                "category":     "suplementos",
                "clicks":       134,
                "avg_read_sec": 243,
            },
            {
                "title":        "Dieta mediterránea y diabetes tipo 2",
                "category":     "enfermedades y dieta",
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
# 4. Descriptores para ADK / LangChain function-calling
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Busca artículos recientes de nutrición y salud. "
            "Consulta PubMed y Healthline vía RSS. "
            "Úsala cuando necesites información actualizada que no esté en la base de conocimiento."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Términos de búsqueda en nutrición"},
                "num_results": {"type": "integer", "description": "Número de resultados (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_trending_topics",
        "description": (
            "Obtiene los temas de nutrición más buscados actualmente. "
            "Lee el CSV exportado de Google Trends por el equipo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Código de país ISO (ES, MX, AR…)", "default": "ES"},
            },
            "required": [],
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
    "get_clickstream_insights": get_clickstream_insights,
}
