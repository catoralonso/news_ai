"""
api_main.py
───────────
FastAPI gateway for the Savia newspaper AI system.
Exposes the endpoints consumed by the Lovable frontend.
Runs as the single process in the HF Space container.
Endpoints:
    GET  /health                    → health check
    GET  /api/trends                → latest trends (José, no full pipeline)
    POST /api/pipeline/run          → full pipeline: José→Camila→Manuel→Asti
    GET  /api/pipeline/status/{id}  → polling for pipeline job status
    GET  /api/articles              → list generated articles
    GET  /api/articles/{article_id} → single article
    GET  /api/social/{article_id}   → SocialMediaPack for an article
    POST /api/chat                  → Mauro chatbot (SSE streaming)
Local usage:
    uvicorn api_main:app --reload --port 8080
Required environment variables:
    ANTHROPIC_API_KEY=...
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
# → "Allow imports from project root without installing as a package"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Observabilidad ────────────────────────────────────────────────────────────
from config import NEWSPAPER_NAME, REGION
logger = logging.getLogger("newspaper_ai.api")

# ── Agentes ───────────────────────────────────────────────────────────────────
# → "Lazy imports to avoid blocking startup if an agent fails to load"
def _import_agents():
    from jose_news_research.agent import NewsResearchAgent, KnowledgeBase as JoseKB
    from camila_fact_checking.agent import FactCheckingAgent, KnowledgeBase as CamilaKB
    from manuel_article_generation.agent import ArticleGenerationAgent, KnowledgeBase as ManuelKB
    from asti_social_media.agent import SocialMediaAgent, KnowledgeBase as AstiKB
    from mauro_reader_interaction.agent import ReaderInteractionAgent, KnowledgeBase as MauroKB
    from orchestrator.agent import Orchestrator
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

# "CORS — allow calls from the Lovable frontend and localhost for development"
#class CORSMiddlewareCustom(BaseHTTPMiddleware):
#    async def dispatch(self, request: Request, call_next):
#        origin = request.headers.get("origin", "")
        
        # Handle OPTIONS preflight
#        if request.method == "OPTIONS":
#            from starlette.responses import Response
#            response = Response()
#            response.headers["Access-Control-Allow-Origin"] = origin
#            response.headers["Access-Control-Allow-Credentials"] = "true"
#            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#            response.headers["Access-Control-Allow-Headers"] = "*"
#            return response
#            
#        response = await call_next(request)
#        if any([
#            "lovable.app" in origin,
#            "lovableproject.com" in origin,
#            "lovable.dev" in origin,
#            "localhost" in origin,
#        ]):
#            response.headers["Access-Control-Allow-Origin"] = origin
#            response.headers["Access-Control-Allow-Credentials"] = "true"
#            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#            response.headers["Access-Control-Allow-Headers"] = "*"
#        return response

#app.add_middleware(CORSMiddlewareCustom)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    """Parámetros opcionales para lanzar el pipeline completo."""
    topic_hint: Optional[str] = None   # e.g. "vitamina D en invierno"
    max_articles: int = 1              # how many articles it generates


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class PipelineStatus(BaseModel):
    status: str        # "running" | "done" | "error"
    message: str
    article_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store (simple; replace with Redis in production)
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
# Health check — called by the container health check every 30s
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
async def health():
    return {
        "status": "ok",
        "newspaper": NEWSPAPER_NAME,
        "region": REGION,
        "timestamp": time.time(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/trends — Jose search trends without launching the whole pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/trends", tags=["content"])
async def get_trends(topic: str = "nutrición tendencias salud"):
    """
    Returns last nutrition trends detected by Jose.
    Calls directly to Jose without using the whole pipeline.
    Useful for a future widget "Trends of the day" for the UI.
    """
    try:
        (
            NewsResearchAgent, JoseKB,
            *_rest
        ) = _import_agents()

        kb = JoseKB()
        agent = NewsResearchAgent(knowledge_base=kb)

        # Jose has a ligh mode that returns only trends withouth complex ideas
        # If not available, calls run() for extracting trends 
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

@app.get("/api/fact-check", tags=["content"])
async def fact_check_single(text: str):
    try:
        (_, _, _, CamilaKB, *_rest) = _import_agents()
        from camila_fact_checking.agent import FactCheckingAgent
        camila = FactCheckingAgent(knowledge_base=CamilaKB())
        result = await asyncio.to_thread(camila.verify_url, text)
        return {
            "verdict":    result.verdict,
            "confidence": result.confidence,
            "reason":     result.reason,
        }
    except Exception as e:
        logger.exception("Error in fact-check")
        raise HTTPException(status_code=500, detail=str(e))
    
# ─────────────────────────────────────────────────────────────────────────────
# POST /api/pipeline/run — Full async Pipeline      
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pipeline/run", tags=["content"])
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Launch full pipeline:
      José (trends) → Camila (fact-check) simultaneously →
      Manuel (article) → Asti (social media)
    Returns a job_id. Client mades polling to
    GET /api/pipeline/status/{job_id} to know when its finished.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "result": None, "error": None}

    background_tasks.add_task(_run_pipeline_task, job_id, request)

    return {"job_id": job_id, "status": "running"}


async def _run_pipeline_task(job_id: str, request: PipelineRequest):
    """Backgrund pipeline"""
    try:
        (
            _NRA, _JKB, _FCA, _CKB, _AGA, _MKB,
            _SMA, _AKB, _RIA, _MauroKB,
            Orchestrator, Memory,
        ) = _import_agents()

        orchestrator = Orchestrator()
        orchestrator.build_agents() 

        # Orchestrator uses asyncio.gather internarly (José+Camila simultaneously)
        query = request.topic_hint or "nutrición tendencias salud"
        result = await orchestrator.run_pipeline_async(query)

        # Saves the article as JSON and returns its ID.
        article = result.article if hasattr(result, "article") else result
        article_id = _save_article(article, job_id, result.fact_check_results)
        
        # Update to more recent social pack with article_id
        social_dir = _get_social_dir()
        packs = sorted(social_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if packs:
            with open(packs[0], encoding="utf-8") as f:
                pack_data = json.load(f)
            pack_data["_article_id"] = article_id
            with open(packs[0], "w", encoding="utf-8") as f:
                json.dump(pack_data, f, ensure_ascii=False, indent=2)

        _jobs[job_id] = {
            "status": "done",
            "result": {
                "article_id":    article_id,
                "title":         getattr(article, "title", "Sin título"),
                "social_saved":  True,
                "fact_check":    [r.to_dict() for r in result.fact_check_results],
            },
            "error": None,
        }
        logger.info("Pipeline job %s completed → article_id=%s", job_id, article_id)

    except Exception as e:
        logger.exception("Pipeline job %s failed", job_id)
        _jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


def _save_article(article, job_id: str, fact_results=None) -> str:
    """Saves the article as JSON and returns its ID."""
    article_id = f"art_{job_id}_{int(time.time())}"
    path = _get_articles_dir() / f"{article_id}.json"
    data = article.to_dict() if hasattr(article, "to_dict") else vars(article)
    data["_id"] = article_id
    if fact_results:
        data["_fact_check"] = [r.to_dict() for r in fact_results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return article_id


@app.get("/api/pipeline/status/{job_id}", tags=["content"])
async def pipeline_status(job_id: str):
    """Polling endpoint to knows if piepeline is over."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/articles — Article's list          
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/articles", tags=["content"])
async def list_articles(limit: int = 10):
    """
    Lists the latest generated articles (most recent first)
    Displayed in the main feed on the UI frontend.
    """
    articles_dir = _get_articles_dir()
    files = sorted(articles_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    articles = []
    for f in files[:limit]:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            # Return summary fields only for the listing
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
    """Returns the full article including article_content."""
    path = _get_articles_dir() / f"{article_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/social/{article_id} — SocialMediaPack               
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/social/{article_id}", tags=["content"])
async def get_social_pack(article_id: str):

    social_dir = _get_social_dir()
    # Find any JSON file in the directory
    matches = list(social_dir.glob("*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No social packs found")
    
    # Search by article_id or fall back to the most recent
    for f in matches:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get("_article_id") == article_id:
            return data
    
    # If no match by ID, return the most recent
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat — Chatbot Mauro with streaming SSE
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["chat"])
async def chat(request: ChatRequest):
    """
    Chatbot Mauro — answer user's inquiries.
    Returns Server-Sent Events (SSE) for word-by-word streaming in the frontend.
    """
    session_id = request.session_id or "default"

    async def generate() -> AsyncIterator[str]:
        try:
            (
                NewsResearchAgent, JoseKB,
                FactCheckingAgent, CamilaKB,
                ArticleGenerationAgent, ManuelKB,
                SocialMediaAgent, AstiKB,
                ReaderInteractionAgent, MauroKB,
                Orchestrator, Memory,
            ) = _import_agents()

            # Reuse Mauro instance per session to preserve conversation memory
            if session_id not in _chat_sessions:
                # Camila
                camila_kb = CamilaKB()
                camila = FactCheckingAgent(knowledge_base=camila_kb)

                # Mauro with Camila injected
                kb = MauroKB()
                memory = Memory(max_turns=20)
                _chat_sessions[session_id] = ReaderInteractionAgent(
                    knowledge_base=kb,
                    camila=camila,
                    memory=memory,
                )

            mauro = _chat_sessions[session_id]

           # Mauro.chat() is sync — run in thread to avoid blocking the event loop
            response = await asyncio.to_thread(mauro.chat, request.message)
            if hasattr(response, "message"):
                text = response.message
            else:
                text = str(response)
                
            words = text.split(" ")

            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.03)  # streaming speed ~150 words/second

            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

        except Exception as e:
            logger.exception("Chat error for session %s", session_id)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles
static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=static_path, html=True), name="frontend")

@app.on_event("startup")
async def startup():
    logger.info("newspaper_ai API started | newspaper=%s region=%s", NEWSPAPER_NAME, REGION,)

@app.on_event("shutdown")
async def shutdown():
    _chat_sessions.clear()
    logger.info("newspaper_ai API shutdown")
