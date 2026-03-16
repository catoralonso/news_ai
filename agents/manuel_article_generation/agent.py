"""
agents/manuel_article_generation/agent.py
──────────────────────────────────────────
Article Generation Agent
─────────────────────────
Responsibility: Receives a verified ArticleIdea and writes a short,
friendly article optimized for visibility and digital engagement.

Architecture:
    Orchestrator / Journalist
          │
          ▼ ArticleIdea (produced by José + verified by Camila)
    ArticleGenerationAgent.run(idea)
          │
          ├─► KnowledgeBase.retrieve()   ← local ChromaDB (newspaper style)
          └─► Gemini generate_content()  ← google-genai (Vertex AI)
                    │
                    └─► Structured response (CreateArticle)

Dependencies:
    pip install google-genai chromadb

Local usage (without GCloud):
    Set GEMINI_API_KEY in .env with an AI Studio key.
    Vertex AI activates automatically when PROJECT_ID is detected.
"""

from __future__ import annotations
# for when the orchestaror is build uncomment
# from config import NEWSPAPER_NAME, PAIS 

import json
import os
from dataclasses import dataclass, field

# ── Google GenAI SDK ──────────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Internal modules ──────────────────────────────────────────────────────────
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

# TODO: move to orchestrator when ready
NEWSPAPER_NAME = os.getenv("NEWSPAPER_NAME", "El Cronista Municipal")
REGION_NEWS    = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL      = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.2


def _build_client() -> genai.Client:
    """
    Authentication priority:
    1. Vertex AI (if PROJECT_ID is set in env) → production
    2. GEMINI_API_KEY                           → local development
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
        "Set GEMINI_API_KEY (local) or GOOGLE_CLOUD_PROJECT (Vertex AI)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output data models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CreateArticle:
    title: str
    angle: str                          
    category: str                      
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
    Indexes newspaper documents (historical articles, style, context)
    and retrieves them semantically to enrich the agent's responses.
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        # Style RAG → learns the newspaper's writing style
        self._style_store = VectorStore(
            collection_name="article_generation",
            persist_dir=f"{persist_dir}/article_generation",
        )
        # José's RAG → knows which topics have already been covered
        self._research_store = VectorStore(
            collection_name="news_research",
            persist_dir=f"{persist_dir}/news_research",
        )
        # Published articles RAG
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def add_style_document(self, doc: dict) -> None:
        """
        doc = {"title": ..., "date": ..., "category": ..., "content": ...}
        """
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._style_store.upsert(texts=texts, metadatas=metas)

    def add_style_documents(self, docs: list[dict]) -> None:
        for doc in docs:
            self.add_style_document(doc)

    def add_published_article(self, doc: dict) -> None:
        """Saves a generated article into the article_published collection."""
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._published_store.upsert(texts=texts, metadatas=metas)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        style_results = self._style_store.query(query, top_k=top_k)
        research_results = self._research_store.query(query, top_k=top_k)
        article_results = self._published_store.query(query, top_k=top_k)
        
        # Merge and return the best results across all three stores
        all_results = style_results + research_results + article_results
        all_results.sort(key=lambda r: r.score)  
        return [r.text for r in all_results[:top_k]]

    def count(self) -> int:
        return self._style_store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Article Generation Agent
# ─────────────────────────────────────────────────────────────────────────────

class ArticleGenerationAgent:
    """
    Article writer based on trends identified by José.

    run(idea) flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Retrieve newspaper style context (RAG)                       │
    │ 2. Build prompt with received ArticleIdea + RAG context         │
    │ 3. Call Gemini with specialized system prompt                   │
    │ 4. Parse response → CreateArticle                               │
    └─────────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
You are Manuel, the Article Generation Agent of a local newspaper.
Your mission: take the local trends identified by José the News Research Agent,
already fact-checked by Camila, and turn them into a complete article.

PERSONALITY:
- Curious, analytical and community-oriented
- Always look for the local angle of national or global news
- Prioritize stories with direct impact on residents' daily lives
- Always look for at least 2 similar sources before writing the article

RESTRICTIONS:
- Never invent data or specific sources; if you don't have the information, say so
- Do not cover topics outside local journalism
- Prioritize verifiability of information

ARTICLE WRITING RULES:
- The article must have between 3 and 5 paragraphs
- Friendly and direct tone, avoid technical or bureaucratic language
- The first paragraph must answer: what happened, where and who is affected

OUTPUT FORMAT:
When asked to write an article, ALWAYS respond with valid JSON in Spanish:
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

    # ── Main method ───────────────────────────────────────────────────────────

    def run(self, idea: ArticleIdea) -> CreateArticle:
        """
        Writes a complete article from a verified ArticleIdea.

        Args:
            idea: Article idea produced by José and verified by Camila.
        
        Returns:
            CreateArticle with the article ready to publish.
        """
        # 1. RAG: retrieve newspaper style context
        context_snippets = self.kb.retrieve(idea.title, top_k=4)

        # 2. Build prompt with RAG context and José's ArticleIdea
        user_prompt = self._build_prompt(idea=idea, context_snippets=context_snippets)
        
        # 3. Call Gemini
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

        # 4. Parse and save published article to RAG
        article = self._parse_article(raw_text, idea)
        self.kb.add_published_article(article.to_dict())

        return article

    def chat(self, user_input: str) -> str:
        """
        Free conversational mode (no structured parsing).
        Useful for Reader Interaction Agent or quick testing.
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

    # ── Private helpers ───────────────────────────────────────────────────────

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
        """Extracts CreateArticle from the JSON returned by Gemini."""
        try:
            # Clean possible markdown blocks ```json ... ```
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
            # If Gemini returns invalid JSON → fallback with raw text preserved
            return CreateArticle(
                title=idea.title,
                angle=idea.angle,
                category=idea.category,
                local_relevance_score=0.0,
                article_content=raw_text,
                sources=idea.sources,
                keywords=idea.keywords,
            )
