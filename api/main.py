"""
api/main.py
───────────
FastAPI gateway — reemplaza app.py (Streamlit).

Expone los endpoints que la web de Lovable consume.
Corre como el único proceso en Cloud Run.

Endpoints:
    GET  /health                    → Cloud Run health check
    GET  /api/trends                → Últimas tendencias (José, sin pipeline completo)
    POST /api/pipeline/run          → Pipeline completo: José→Camila→Manuel→Asti
    GET  /api/articles              → Lista artículos generados
    GET  /api/articles/{article_id} → Artículo concreto
    GET  /api/social/{article_id}   → SocialMediaPack de un artículo
    POST /api/chat                  → Chatbot Mauro (streaming SSE)

Uso local:
    uvicorn api.main:app --reload --port 8080

Variables de entorno requeridas (mismo .env que antes):
    GEMINI_API_KEY=...   (local)
    GOOGLE_CLOUD_PROJECT=...  (producción, activa Vertex AI)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Permite importar desde la raíz del proyecto sin instalar el paquete
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Observabilidad ────────────────────────────────────────────────────────────
from config import (
    NEWSPAPER_NAME, REGION,
    VERTEX_PROJECT,
)

logger = logging.getLogger("newspaper_ai.api")

# ── Agentes ───────────────────────────────────────────────────────────────────
# Importaciones lazy para no bloquear el arranque si un agente falla
def _import_agents():
    from agents.jose_news_research.agent import NewsResearchAgent, KnowledgeBase as JoseKB
    from agents.camila_fact_checking.agent import FactCheckingAgent, KnowledgeBase as CamilaKB
    from agents.manuel_article_generation.agent import ArticleGenerationAgent, KnowledgeBase as ManuelKB
    from agents.asti_social_media.agent import SocialMediaAgent, KnowledgeBase as AstiKB
    from agents.mauro_reader_interaction.agent import ReaderInteractionAgent, KnowledgeBase as MauroKB
    from agents.orchestrator.agent import Orchestrator
    from core.memory import Memory
    return (
        NewsResearchAgent, JoseKB,
        FactCheckingAgent, CamilaKB,
        ArticleGenerationAgent, ManuelKB,
        SocialMediaAgent, AstiKB,
        ReaderInteractionAgent, MauroKB,
        Orchestrator,
        Memory,
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Newspaper AI — API Gateway",
    description="Multi-agent system for a nutrition newspaper. Powers the Lovable frontend.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI en /docs
    redoc_url="/redoc",
)

# CORS — permite llamadas desde la web de Lovable (y localhost para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.lovable.app",
        "https://lovable.app",
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    """Parámetros opcionales para lanzar el pipeline completo."""
    topic_hint: Optional[str] = None   # e.g. "vitamina D en invierno"
    max_articles: int = 1              # cuántos artículos generar (1-3)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class PipelineStatus(BaseModel):
    status: str        # "running" | "done" | "error"
    message: str
    article_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store (simple; reemplazar con Redis/Firestore en producción)
# ─────────────────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}          # job_id → {status, result, error}
_chat_sessions: dict[str, object] = {}  # session_id → MauroAgent instance


def _get_articles_dir() -> Path:
    d = PROJECT_ROOT / "data" / "articles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_social_dir() -> Path:
    d = PROJECT_ROOT / "data" / "social_media_output"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Health check — Cloud Run lo llama cada 30s
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
async def health():
    return {
        "status": "ok",
        "newspaper": NEWSPAPER_NAME,
        "region": REGION,
        "vertex": bool(VERTEX_PROJECT),
        "timestamp": time.time(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/trends — José busca tendencias sin lanzar el pipeline completo
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/trends", tags=["content"])
async def get_trends(topic: str = "nutrición tendencias salud"):
    """
    Devuelve las últimas tendencias de nutrición detectadas por José.
    Llama directamente a José sin pasar por el pipeline completo.
    Útil para mostrar un widget "Temas del día" en la web de Lovable.
    """
    try:
        (
            NewsResearchAgent, JoseKB,
            *_rest
        ) = _import_agents()

        kb = JoseKB()
        agent = NewsResearchAgent(knowledge_base=kb)

        # José tiene un método ligero que sólo devuelve tendencias sin generar ideas completas
        # Si no existe, llamamos a run() y extraemos las tendencias del resultado
        if hasattr(agent, "get_trends"):
            trends = await asyncio.to_thread(agent.get_trends)
        else:
            result = await asyncio.to_thread(agent.run, topic)
            trends = [
                {"topic": t, "relevance": 1.0}
                for t in (result.trending_topics if hasattr(result, "trending_topics") else [str(result)])
            ]

        logger.info("Trends fetched: %d items", len(trends))
        return {"trends": trends, "fetched_at": time.time()}

    except Exception as e:
        logger.exception("Error fetching trends")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/pipeline/run — Pipeline completo asíncrono
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pipeline/run", tags=["content"])
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Lanza el pipeline completo en background:
      José (trends) → Camila (fact-check) en paralelo →
      Manuel (artículo) → Asti (social media)

    Devuelve un job_id inmediatamente. El cliente hace polling a
    GET /api/pipeline/status/{job_id} para saber cuándo ha terminado.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "result": None, "error": None}

    background_tasks.add_task(_run_pipeline_task, job_id, request)

    return {"job_id": job_id, "status": "running"}


async def _run_pipeline_task(job_id: str, request: PipelineRequest):
    """Ejecuta el pipeline en background."""
    try:
        (
            _NRA, _JKB, _FCA, _CKB, _AGA, _MKB,
            _SMA, _AKB, _RIA, _MauroKB,
            Orchestrator, Memory,
        ) = _import_agents()

        orchestrator = Orchestrator()

        # El orquestador ya maneja asyncio.gather internamente (José+Camila en paralelo)
        query = request.topic_hint or "nutrición tendencias salud"
        result = await orchestrator.run_pipeline_async(query)

        # Guardar artículo generado en disco para que GET /api/articles lo encuentre
        article = result.article if hasattr(result, "article") else result
        article_id = _save_article(article, job_id)

        _jobs[job_id] = {
            "status": "done",
            "result": {
                "article_id": article_id,
                "title": getattr(article, "title", "Sin título"),
                "social_saved": True,
            },
            "error": None,
        }
        logger.info("Pipeline job %s completed → article_id=%s", job_id, article_id)

    except Exception as e:
        logger.exception("Pipeline job %s failed", job_id)
        _jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


def _save_article(article, job_id: str) -> str:
    """Guarda el artículo como JSON y devuelve su ID."""
    article_id = f"art_{job_id}_{int(time.time())}"
    path = _get_articles_dir() / f"{article_id}.json"
    data = article.to_dict() if hasattr(article, "to_dict") else vars(article)
    data["_id"] = article_id
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return article_id


@app.get("/api/pipeline/status/{job_id}", tags=["content"])
async def pipeline_status(job_id: str):
    """Polling endpoint para saber si el pipeline ha terminado."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/articles — Lista artículos generados
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/articles", tags=["content"])
async def list_articles(limit: int = 10):
    """
    Lista los últimos artículos generados (ordenados por más reciente).
    Lovable los muestra en el feed principal de la web.
    """
    articles_dir = _get_articles_dir()
    files = sorted(articles_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    articles = []
    for f in files[:limit]:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            # Sólo devolvemos campos de resumen para el listado
            articles.append({
                "id":         data.get("_id", f.stem),
                "title":      data.get("title", "Sin título"),
                "category":   data.get("category", ""),
                "angle":      data.get("angle", ""),
                "keywords":   data.get("keywords", [])[:5],
                "relevance":  data.get("local_relevance_score", 0),
                "created_at": f.stat().st_mtime,
            })
        except Exception:
            continue

    return {"articles": articles, "total": len(articles)}


@app.get("/api/articles/{article_id}", tags=["content"])
async def get_article(article_id: str):
    """Devuelve el artículo completo (incluido article_content)."""
    path = _get_articles_dir() / f"{article_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/social/{article_id} — SocialMediaPack de un artículo
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/social/{article_id}", tags=["content"])
async def get_social_pack(article_id: str):
    """
    Devuelve el SocialMediaPack generado por Asti para un artículo concreto.
    Lovable puede mostrarlo en la vista de detalle del artículo.
    """
    social_dir = _get_social_dir()
    # Buscamos el JSON que tenga el article_id en el nombre
    matches = list(social_dir.glob(f"*{article_id}*.json"))
    if not matches:
        # Intentar por título aproximado
        raise HTTPException(status_code=404, detail=f"Social pack for '{article_id}' not found")

    with open(matches[0], encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat — Chatbot Mauro con streaming SSE
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["chat"])
async def chat(request: ChatRequest):
    """
    Chatbot Mauro — responde preguntas de los lectores.

    Devuelve Server-Sent Events (SSE) para que Lovable muestre
    la respuesta en streaming letra a letra.

    Ejemplo de consumo en React/Lovable:
        const res = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({message, session_id}) })
        const reader = res.body.getReader()
        // leer chunks y añadir al estado
    """
    session_id = request.session_id or "default"

    async def generate() -> AsyncIterator[str]:
        try:
            (
                *_agents,
                ReaderInteractionAgent, MauroKB,
                Orchestrator, Memory,
            ) = _import_agents()

            # Reutilizamos la instancia de Mauro por sesión para mantener memoria
            if session_id not in _chat_sessions:
                from agents.camila_fact_checking.agent import FactCheckingAgent, KnowledgeBase as CamilaKB
                # Camila
                camila_kb = CamilaKB()
                camila = FactCheckingAgent(knowledge_base=camila_kb)

                # Mauro con Camila inyectada
                kb = MauroKB()
                memory = Memory(max_turns=20)
                _chat_sessions[session_id] = ReaderInteractionAgent(
                    knowledge_base=kb,
                    camila=camila,
                    memory=memory,
                )

            mauro = _chat_sessions[session_id]

           # Mauro.chat() es síncrono — lo corremos en thread para no bloquear el event loop
            response = await asyncio.to_thread(mauro.chat, request.message)
            if hasattr(response, "answer"):
                text = response.answer
            elif hasattr(response, "text"):
                text = response.text
            else:
                text = str(response)
            words = text.split(" ")

            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.03)  # velocidad de streaming ~150 palabras/segundo

            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

        except Exception as e:
            logger.exception("Chat error for session %s", session_id)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Nginx no bufferee el SSE
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info(
        "newspaper_ai API started | newspaper=%s region=%s vertex=%s",
        NEWSPAPER_NAME, REGION, bool(VERTEX_PROJECT),
    )


@app.on_event("shutdown")
async def shutdown():
    _chat_sessions.clear()
    logger.info("newspaper_ai API shutdown")
