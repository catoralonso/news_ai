"""
agents/news_research/agent.py
──────────────────────────────
News Research Agent
───────────────────
Responsabilidad: investigar tendencias, detectar oportunidades de cobertura
y proponer ideas de artículos con contexto enriquecido por RAG.

Arquitectura:
    Usuario / Orchestrator
          │
          ▼
    NewsResearchAgent.run(query)
          │
          ├─► KnowledgeBase.retrieve()   ← ChromaDB local
          ├─► web_search()               ← Google CSE o mock
          ├─► get_trending_topics()      ← pytrends o mock
          └─► Gemini generate_content()  ← google-genai (Vertex AI)
                    │
                    └─► Respuesta estructurada (ArticleIdea[])

Dependencias:
    pip install google-genai chromadb pytrends requests

Uso local (sin GCloud):
    Configurar GEMINI_API_KEY en .env con una key de AI Studio.
    Vertex AI se activa automáticamente cuando se detecte PROJECT_ID.
"""

from __future__ import annotations
# cuando este el orchestator
# from config import NEWSPAPER_NAME, PAIS 

import json
import os
from dataclasses import dataclass, field
from typing import Any

# ── Google GenAI SDK ──────────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Módulos internos ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from tools.search_tools import (
    web_search,
    get_trending_topics,
    get_clickstream_insights,
    format_insights_for_prompt,
    TOOL_SCHEMAS,
    TOOL_DISPATCH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")      
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "") 
VERTEX_REGION   = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
NEWSPAPER_NAME = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
PAIS           = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL      = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.4


def _build_client() -> genai.Client:
    """
    Prioridad de autenticación:
    1. Vertex AI (si hay PROJECT_ID en env) → producción
    2. GEMINI_API_KEY                        → desarrollo local
    """
    if VERTEX_PROJECT:
        return genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_REGION,
        )
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    raise EnvironmentError(
        "Configura GEMINI_API_KEY (local) o GOOGLE_CLOUD_PROJECT (Vertex AI)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de datos de salida
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArticleIdea:
    title: str
    angle: str                          # enfoque periodístico sugerido
    category: str                       # política, deportes, cultura…
    local_relevance_score: float        # 0–1
    sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    priority: str = "media"             # alta / media / baja

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ResearchReport:
    query: str
    trending_topics: list[str]
    article_ideas: list[ArticleIdea]
    context_snippets: list[str]         # fragmentos RAG usados
    raw_web_results: list[dict]

    def to_dict(self) -> dict:
        return {
            "query":            self.query,
            "trending_topics":  self.trending_topics,
            "article_ideas":    [a.to_dict() for a in self.article_ideas],
            "context_snippets": self.context_snippets,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Indexa documentos del periódico (artículos históricos, estilo, contexto)
    y los recupera semánticamente para enriquecer las respuestas del agente.
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        self._store = VectorStore(
            collection_name="news_research",
            persist_dir=f"{persist_dir}/news_research",
        )
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def add_document(self, doc: dict) -> None:
        """
        doc = {"title": ..., "date": ..., "category": ..., "content": ...}
        """
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._store.upsert(texts=texts, metadatas=metas)

    def add_documents(self, docs: list[dict]) -> None:
        for doc in docs:
            self.add_document(doc)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        news_result = self._store.query(query, top_k=top_k)
        article_result = self._published_store.query(query, top_k=top_k)
        
        all_results = news_result + article_result
        all_results.sort(key=lambda r: r.score) 
        return [r.text for r in all_results[:top_k]]      

    def count(self) -> int:
        return self._store.count()


# ─────────────────────────────────────────────────────────────────────────────
# News Research Agent
# ─────────────────────────────────────────────────────────────────────────────

class NewsResearchAgent:
    """
    Agente de investigación de noticias.

    Flujo de .run(query):
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Recuperar contexto histórico del periódico (RAG)         │
    │ 2. Obtener trending topics locales                          │
    │ 3. Buscar noticias recientes en la web                      │
    │ 4. Enviar todo a Gemini con system prompt especializado      │
    │ 5. Parsear respuesta → ResearchReport                       │
    └─────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
Eres el News Research Agent de un periódico local.
Tu misión: investigar tendencias, detectar oportunidades de cobertura y
proponer ideas de artículos periodísticos relevantes para la comunidad local.
 
PERSONALIDAD:
- Curioso, analítico y orientado a la comunidad
- Buscas siempre el ángulo local de cada noticia nacional o global
- Priorizas historias con impacto directo en la vida de los vecinos
- Siempre buscas al menos 2 fuentes antes de proponer una noticia
 
RESTRICCIONES:
- Nunca inventes datos o fuentes específicas; si no tienes información, dilo
- No cubras temas fuera del ámbito periodístico local
- Prioriza la verificabilidad de la información
 
FORMATO DE SALIDA:
Cuando se te pida proponer ideas, responde SIEMPRE con JSON válido:
{
  "article_ideas": [
    {
      "title": "Título sugerido del artículo",
      "angle": "Enfoque o ángulo periodístico específico",
      "category": "nutrición|recetas|bienestar|suplementos|dietas|comunidad",
      "local_relevance_score": 0.0,
      "sources": ["fuente1", "fuente2"],
      "keywords": ["keyword1", "keyword2"],
      "priority": "alta|media|baja"
    }
  ],
  "summary": "Resumen ejecutivo del panorama informativo actual"
}

REGLAS para "local_relevance_score":
número entre 0.0 y 1.0 que indica qué tan relevante es esta noticia para la comunidad local, donde:
    - 1.0 = afecta directamente a vecinos del municipio (obras, eventos, política local)
    - 0.5 = tema nacional con impacto local moderado
    - 0.0 = tema sin conexión con la comunidad local,
""".strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = PAIS,
    ):
        self.kb = knowledge_base
        self.memory = memory or Memory(max_turns=10)
        self.newspaper_name = newspaper_name
        self.region = region
        self._client = _build_client()

    # ── Método principal ──────────────────────────────────────────────────────

    def run(self, query: str) -> ResearchReport:
        """
        Ejecuta el ciclo completo de investigación.

        Args:
            query: Tema o pregunta de investigación.
                   Ej: "¿Qué pasa con el transporte público esta semana?"

        Returns:
            ResearchReport con ideas de artículos estructuradas.
        """
        # 1. RAG: contexto histórico del periódico
        context_snippets = self.kb.retrieve(query, top_k=4)

        # 2. Trending topics
        trending = get_trending_topics(region=self.region)

        # 3. Búsqueda web
        web_results = web_search(query, num_results=5)

        # 4. Clickstream — qué están leyendo los lectores esta semana
        clickstream = get_clickstream_insights(days=7)

        # 5. Construir prompt enriquecido
        user_prompt = self._build_prompt(
            query=query,
            context_snippets=context_snippets,
            trending=trending,
            web_results=web_results,
            clickstream=clickstream,
        )

        # 5. Llamar a Gemini
        self.memory.add("user", user_prompt)

        response = self._client.models.generate_content(
            model=CHAT_MODEL,
            contents=self._messages_to_contents(self.memory.as_messages()),
            config=types.GenerateContentConfig(
                system_instruction=self._personalized_system_prompt(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )

        raw_text = response.candidates[0].content.parts[0].text
        self.memory.add("model", raw_text)

        # 6. Parsear respuesta
        article_ideas = self._parse_ideas(raw_text)

        return ResearchReport(
            query=query,
            trending_topics=trending,
            article_ideas=article_ideas,
            context_snippets=context_snippets,
            raw_web_results=web_results,
        )

    def chat(self, user_input: str) -> str:
        """
        Modo conversacional libre (sin parseo estructurado).
        Útil para el Reader Interaction Agent o pruebas rápidas.
        """ 
        self.memory.add("user", user_input)
        response = self._client.models.generate_content(
            model=CHAT_MODEL,
            contents=self._messages_to_contents(self.memory.as_messages()),
            config=types.GenerateContentConfig(
                system_instruction=self._personalized_system_prompt(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        reply = response.candidates[0].content.parts[0].text
        self.memory.add("model", reply)
        return reply

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _personalized_system_prompt(self) -> str:
        return f"{self.SYSTEM_PROMPT}\n\nPeriódico: {self.newspaper_name}. Región: {self.region}."

    def _build_prompt(
        self,
        query: str,
        context_snippets: list[str],
        trending: list[str],
        web_results: list[dict],
        clickstream: dict | None = None,
    ) -> str:
        ctx_block = "\n".join(f"- {s[:300]}" for s in context_snippets) or "Sin contexto previo."
        trend_block = ", ".join(trending[:5]) or "No disponible."
        web_block = "\n".join(
            f"• {r['title']} ({r['source']}): {r['snippet'][:200]}"
            for r in web_results
        ) or "Sin resultados web."

        click_block = (
            format_insights_for_prompt(clickstream)
            if clickstream
            else "Sin datos de comportamiento de lectores todavía."
        )

        return f"""
CONSULTA DE INVESTIGACIÓN: {query}

CONTEXTO HISTÓRICO DEL PERIÓDICO (RAG):
{ctx_block}

TEMAS EN TENDENCIA HOY:
{trend_block}

NOTICIAS RECIENTES EN LA WEB:
{web_block}
 
{click_block}
 
Con base en toda la información anterior, propón 3 ideas de artículos
para {self.newspaper_name}. Prioriza los temas que combinan tendencia
externa con alto engagement histórico de los lectores.
Responde en el formato JSON indicado.
""".strip()

    def _messages_to_contents(self, messages: list[dict]) -> list[types.Content]:
        result = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            result.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )
        return result

    def _parse_ideas(self, raw_text: str) -> list[ArticleIdea]:
        """Extrae ArticleIdea[] del JSON que devuelve Gemini."""
        try:
            # Limpia posibles bloques markdown ```json ... ```
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
            ideas_raw = data.get("article_ideas", [])
            return [
                ArticleIdea(
                    title=i.get("title", "Sin título"),
                    angle=i.get("angle", ""),
                    category=i.get("category", "general"),
                    local_relevance_score=float(i.get("local_relevance_score",0.5)),
                    sources=i.get("sources", []),
                    keywords=i.get("keywords", []),
                    priority=i.get("priority", "media"),
                )
                for i in ideas_raw
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Si Gemini no devuelve JSON válido → idea genérica de fallback
            return [
                ArticleIdea(
                    title="Investigación pendiente de parseo",
                    angle=raw_text[:200],
                    category="general",
                    local_relevance_score=0.0,
                    priority="baja",
                )
            ]
