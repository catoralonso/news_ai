"""
tools/search_tools.py
─────────────────────
Tools that the News Research Agent can invoke.
Each tool is a pure Python function + its descriptor for ADK/LangChain.
Designed to run 100% locally; no GCloud dependencies.
 
Web search:
    Primary source → RSS from PubMed + Healthline
    Fallback        → mock for offline development
 
Trends:
    Primary source → Google Trends CSV (manually exported, local)
    Fallback        → mock for offline development
 
Clickstream:
    The newspaper website will write events to
    data/clickstream/events.jsonl with this format per line:
    {
      "article_id":   "mediterranean-diet-2025-02-15",
      "title":        "Mediterranean diet and type 2 diabetes",
      "category":     "diseases and diet",
      "event":        "read" | "click" | "scroll",
      "duration_sec": 142,      # only for event="read"
      "scroll_pct":   87,       # only for event="scroll" (0-100)
      "timestamp":    "2025-03-10T14:32:00"
    }
    While the site does not exist, get_clickstream_insights() generates
    realistic mock data so José can use it from the start.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 1. Web search (PubMed + Healthline RSS → mock fallback)
# ─────────────────────────────────────────────────────────────────────────────

RSS_SOURCES = {
    "pubmed": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term={query}&format=rss",
    "healthline": "https://www.healthline.com/rss/health-news",
}

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Searches for recent nutrition and health articles.
    Primary source: PubMed + Healthline RSS (free, no API key required).
    Fallback:       mock for offline development.
    Returns a list of:
        {"title": str, "url": str, "snippet": str, "source": str}
    """
    results = _rss_search(query, num_results)
    if results:
        return results
    return _mock_search(query, num_results)

def _rss_search(query: str, num: int) -> list[dict]:
    """
    Searches PubMed RSS (query-specific) and Healthline (general feed).
    Returns an empty list if both sources fail — web_search falls back to mock.
    """
    results: list[dict] = []

    # PubMed — query-specific RSS
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

    # Healthline — general feed, filtered by query in title/summary
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
    return []

# ─────────────────────────────────────────────────────────────────────────────
# 2. Trends (Google Trends CSV → mock fallback)
# ─────────────────────────────────────────────────────────────────────────────

TRENDS_CSV = os.getenv("TRENDS_CSV_PATH", "data/trends/trends.csv")

def get_trending_topics(region: str = "ES") -> list[str]:
    """
    Retrieves trending nutrition topics.
    Primary source: CSV manually exported from Google Trends.
    Fallback:       mock for offline development.
    The `region` parameter is kept for when the orchestrator passes it,
    although the CSV is already filtered by region from Google Trends.
    """
    if os.path.exists(TRENDS_CSV):
        return _read_trends_csv(TRENDS_CSV)
    return _mock_trending_topics()

def _read_trends_csv(path: str) -> list[str]:
    """
    Reads the CSV exported from Google Trends.
    TODO: adjust column name once the teammate uploads the real CSV.
    Expected format (Google Trends export):
        Search term, Value
        proteins and muscle,  100
        ketogenic diet,        85
        ...
    """
    topics: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:            
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
# 3. Clickstream — real reader behavior on the site
# ─────────────────────────────────────────────────────────────────────────────

CLICKSTREAM_DIR  = os.getenv("CLICKSTREAM_DIR", "data/clickstream")
CLICKSTREAM_FILE = os.path.join(CLICKSTREAM_DIR, "events.jsonl")

def log_event(
    article_id: str,
    title: str,
    category: str,
    event: str,                     
    duration_sec: int = 0,          
    scroll_pct: int = 0,          
) -> None:
    """
    Logs a reader behavior event.
    Called from the newspaper website once it is built.
    The events.jsonl file grows with one JSON line per event.
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
    Reads events.jsonl and returns engagement metrics by category
    and by article for the last `days` days.
    Returns:
    {
      "by_category": {
        "recipes":           {"clicks": 312, "avg_read_sec": 198, "avg_scroll_pct": 84},
        "weight loss":       {"clicks": 287, "avg_read_sec": 231, "avg_scroll_pct": 79},
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
    """Processes the real events file."""
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
    Realistic mock data for when the site does not yet exist.
    Simulates a typical week for the nutrition newspaper.
    TODO: update categories once the Content Engineer defines the official ones.
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
    Converts the insights dict into readable text to inject
    into José's prompt.
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
# 4. Descriptors for ADK / LangChain function-calling
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

# Dispatch map for the agent (name → function)
TOOL_DISPATCH: dict[str, Any] = {
    "web_search":               web_search,
    "get_trending_topics":      get_trending_topics,
    "get_clickstream_insights": get_clickstream_insights,
}
