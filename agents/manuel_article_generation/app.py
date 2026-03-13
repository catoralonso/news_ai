"""
agents/manuel_article_generation/agent.py
──────────────────────────────────────────
Article Generation Agent
─────────────────────────
Responsabilidad: Recibe una ArticleIdea verificada y escribe un artículo
corto, cercano y optimizado para visibilidad y engagement digital.

Arquitectura:
    Orchestrator / Periodista
          │
          ▼ ArticleIdea (producida por José + verificada por Camila)
    ArticleGenerationAgent.run(idea)
          │
          ├─► KnowledgeBase.retrieve()   ← ChromaDB local (estilo del periódico)
          └─► Gemini generate_content()  ← google-genai (Vertex AI)
                    │
                    └─► Respuesta estructurada (CreateArticle)

Dependencias:
    pip install google-genai chromadb

Uso local (sin GCloud):
    Configurar GEMINI_API_KEY en .env con una key de AI Studio.
    Vertex AI se activa automáticamente cuando se detecte PROJECT_ID.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ── Google GenAI SDK ──────────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Módulos internos ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from agents.jose_news_research.agent import ArticleIdea


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")      
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "") 
VERTEX_REGION   = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# CAMBIAR CUANDO HAYA ORQUESTADOR
NEWSPAPER_NAME = os.getenv("NEWSPAPER_NAME", "El Cronista Municipal")
REGION_NEWS    = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL      = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.2


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
class CreateArticle:
    title: str
    angle: str                          # enfoque periodístico sugerido
    category: str                       # política, deportes, cultura…
    local_relevance_score: float        # 0–1
    article_content: str
    sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


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
            collection_name="article_generation",
            persist_dir=persist_dir,
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
        results = self._store.query(query, top_k=top_k)
        return [r.text for r in results]

    def count(self) -> int:
        return self._store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Article Generation Agent
# ─────────────────────────────────────────────────────────────────────────────

class ArticleGenerationAgent:
    """
    Agente escritor de artículos de acuerdo a las tendencias que identificó Jose.

    Flujo de .run(idea):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Recuperar contexto histórico del periódico (RAG)             │
    │ 2. Construir prompt con la ArticleIdea recibida + contexto RAG  │
    │ 3. Llamar a Gemini con system prompt especializado              │
    │ 4. Parsear respuesta → CreateArticle                            │
    └─────────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
Eres Manuel el Article Generation Agent de un periódico local.
Tu misión: analizar las tendencias locales que ha generado Jose el News
Research Agent y que tuvo un Fact Check por Camilia Agent.

PERSONALIDAD:
- Curioso, analítico y orientado a la comunidad
- Buscas siempre el ángulo local de cada noticia nacional o global
- Priorizas historias con impacto directo en la vida de los vecinos
- Siempre buscas al menos 2 fuentes similares antes de construir el artículo

RESTRICCIONES:
- Nunca inventes datos o fuentes específicas; si no tienes información, dilo
- No cubras temas fuera del ámbito periodístico local
- Prioriza la verificabilidad de la información

REGLAS DE ESCRITURA DE ARTÍCULO:
- El artículo debe tener entre 3 y 5 párrafos
- Tono cercano y directo, evita lenguaje técnico o burocrático
- El primer párrafo debe responder: qué pasó, dónde y a quién afecta

FORMATO DE SALIDA:
Cuando se te pida escribir un artículo, responde SIEMPRE con JSON válido:
{
      "title": "Título del artículo",
      "angle": "Enfoque periodístico",
      "category": "nutrición|recetas|bienestar|suplementos|dietas|comunidad",
      "local_relevance_score": 0.85,
      "article_content": "Cuerpo completo del artículo...",
      "sources": ["fuente1", "fuente2"],
      "keywords": ["keyword1", "keyword2"]
}

""".strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = REGION_NEWS,
    ):
        self.kb = knowledge_base
        self.memory = memory or Memory(max_turns=10)
        self.newspaper_name = newspaper_name
        self.region = region
        self._client = _build_client()

    # ── Método principal ──────────────────────────────────────────────────────

    def run(self, idea: ArticleIdea) -> CreateArticle:
        """
        Escribe un artículo completo a partir de una ArticleIdea verificada.

        Args:
            idea: Idea de artículo producida por José y verificada por Camila.
        
        Returns:
            CreateArticle con el artículo listo para publicar.
        """
        # 1. RAG: contexto histórico del periódico
        context_snippets = self.kb.retrieve(idea.title, top_k=4)

        # 2. Construir prompt con RAG y ArticleIdea de Jose
        user_prompt = self._build_prompt(idea=idea, context_snippets=context_snippets)
        
        # 3. Llamar a Gemini
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

        return self._parse_article(raw_text, idea)

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

    def _build_prompt(self, idea: ArticleIdea, context_snippets: list[str]) -> str:
        ctx_block = "\n".join(f"- {s[:300]}" for s in context_snippets) or "Sin contexto previo."
        sources_block = ", ".join(idea.sources) or "No disponibles."
    
        return f"""
IDEA A DESARROLLAR: {idea.title}
ÁNGULO: {idea.angle}
CATEGORÍA: {idea.category}
PALABRAS CLAVE: {", ".join(idea.keywords)}
FUENTES DISPONIBLES: {sources_block}

ESTILO DEL PERIÓDICO (RAG):
{ctx_block}

Escribe el artículo completo para {self.newspaper_name}. Responde en el formato JSON indicado.
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

    def _parse_article(self, raw_text: str, idea: ArticleIdea) -> CreateArticle:
        try:
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
            return CreateArticle(
                title=data.get("title", idea.title),
                angle=data.get("angle", idea.angle),
                category=data.get("category", idea.category),
                local_relevance_score=float(data.get("local_relevance_score", idea.local_relevance_score)),
                article_content=data.get("article_content", ""),
                sources=data.get("sources", idea.sources),
                keywords=data.get("keywords", idea.keywords),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return CreateArticle(
                title=idea.title,
                angle=idea.angle,
                category=idea.category,
                local_relevance_score=0.0,
                article_content=raw_text,
                sources=idea.sources,
                keywords=idea.keywords,
            )
