"""
agents/jose_news_research/agent.py
────────────────────────────────────
News Research Agent — José
──────────────────────────
Responsibility: research trending topics, detect coverage opportunities,
and propose article ideas enriched with RAG context.

Architecture:
    User / Orchestrator
          │
          ▼
    NewsResearchAgent.run(query)
          │
          ├─► KnowledgeBase.retrieve()   ← local ChromaDB
          ├─► web_search()               ← RSS PubMed + Healthline / mock
          ├─► get_trending_topics()      ← Google Trends CSV / mock
          ├─► get_clickstream_insights() ← reader behavior / mock
          └─► Gemini generate_content()  ← google-genai (Vertex AI or AI Studio)
                    │
                    └─► Structured JSON → ResearchReport(ArticleIdea[])

Local usage (no GCloud):
    Set GEMINI_API_KEY in .env with a Gemini AI Studio key.
    Vertex AI activates automatically when GOOGLE_CLOUD_PROJECT is detected.
"""

from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from config import (
    CHAT_MODEL,
    EMBEDDINGS_DIR,
    GEMINI_API_KEY,
    NEWSPAPER_NAME,
    REGION as PAIS,
    VERTEX_PROJECT,
    VERTEX_REGION,
)
from core.chunker import chunk_document
from core.memory import Memory
from core.models import ArticleIdea, ResearchReport
from core.vector_store import VectorStore
from tools.search_tools import (
    TOOL_DISPATCH,   # noqa: F401  (available for ADK tool-calling if needed)
    TOOL_SCHEMAS,    # noqa: F401
    format_insights_for_prompt,
    get_clickstream_insights,
    get_trending_topics,
    web_search,
)

logger = logging.getLogger("newspaper_ai.jose")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client factory
# ─────────────────────────────────────────────────────────────────────────────

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.4


def _build_client() -> genai.Client:
    """
    Authentication priority:
    1. Vertex AI  — when GOOGLE_CLOUD_PROJECT is set (production / Cloud Run)
    2. API key    — when GEMINI_API_KEY is set     (local development)
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
        "Authentication not configured. "
        "Set GEMINI_API_KEY (local) or GOOGLE_CLOUD_PROJECT (Vertex AI)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Semantic retrieval layer for José.

    Collections (read/write permissions):
    - news_research     [OWN]       historical articles + research context
    - article_published [READ-ONLY] articles published by Manuel
                                    → avoids proposing already-covered angles
    """

    def __init__(self, persist_dir: str = EMBEDDINGS_DIR):
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
        Index a single document.
        Expected shape: {"title": str, "date": str, "category": str, "content": str}
        """
        chunks = chunk_document(doc)
        texts  = [c["text"] for c in chunks]
        metas  = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._store.upsert(texts=texts, metadatas=metas)

    def add_documents(self, docs: list[dict]) -> None:
        for doc in docs:
            self.add_document(doc)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        Query both collections and return the top_k most relevant chunks,
        ranked by cosine similarity (lower distance = more similar).
        """
        news_results      = self._store.query(query, top_k=top_k)
        published_results = self._published_store.query(query, top_k=top_k)

        all_results = news_results + published_results
        all_results.sort(key=lambda r: r.score)
        return [r.text for r in all_results[:top_k]]

    def count(self) -> int:
        return self._store.count()


# ─────────────────────────────────────────────────────────────────────────────
# News Research Agent
# ─────────────────────────────────────────────────────────────────────────────

class NewsResearchAgent:
    """
    News Research Agent — José.

    run(query) pipeline
    ───────────────────
    1. Retrieve historical newspaper context       (RAG)
    2. Fetch trending topics                       (Google Trends CSV / mock)
    3. Search recent nutrition news on the web     (RSS PubMed+Healthline / mock)
    4. Read reader behaviour from clickstream      (JSONL / mock)
    5. Build enriched prompt and call Gemini
    6. Parse JSON response → ResearchReport
    """

    SYSTEM_PROMPT = """
You are José, the News Research Agent of {newspaper}, a local nutrition newspaper in {region}.
Your mission: research trending topics, detect coverage opportunities, and
propose article ideas that are relevant and valuable to the local community.

PERSONALITY:
- Curious, analytical, and community-oriented
- You always look for the local angle in national or global news
- You prioritise stories with a direct impact on readers' daily lives
- You always find at least 2 independent sources before proposing a story

RESTRICTIONS:
- Never invent data or specific sources; if you don't have information, say so
- Only cover topics within the nutrition and health domain
- Prioritise verifiability of information

OUTPUT FORMAT:
Always respond with a single valid JSON object (no markdown fences, no extra text):
{
  "article_ideas": [
    {
      "title": "Article title in Spanish",
      "angle": "Specific journalistic angle or approach",
      "category": "nutrition|recipes|wellness|supplements|diets|community",
      "local_relevance_score": 0.0,
      "sources": ["source1", "source2"],
      "keywords": ["keyword1", "keyword2"],
      "priority": "alta|media|baja"
    }
  ],
  "summary": "Brief executive summary of the current news landscape (in Spanish)"
}

RULES for local_relevance_score (0.0 – 1.0):
  1.0 → directly affects residents (local events, regional health alerts)
  0.5 → national topic with moderate local impact
  0.0 → no connection to the local community
""".strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = PAIS,
    ):
        self.kb             = knowledge_base
        self.memory         = memory or Memory(max_turns=10)
        self.newspaper_name = newspaper_name
        self.region         = region
        self._client        = _build_client()
        logger.info("NewsResearchAgent (José) initialised | newspaper=%s region=%s",
                    newspaper_name, region)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> ResearchReport:
        """
        Execute the full research pipeline.

        Args:
            query: Research question from the journalist or orchestrator.
                   E.g. "¿Qué temas de nutrición deberíamos cubrir esta semana?"

        Returns:
            ResearchReport with structured ArticleIdea list.
        """
        logger.info("José.run() | query=%s", query[:80])

        # 1. RAG — historical newspaper context
        context_snippets = self.kb.retrieve(query, top_k=4)

        # 2. Trending topics
        trending = get_trending_topics(region=self.region)

        # 3. Web search (RSS PubMed + Healthline, fallback mock)
        web_results = web_search(query, num_results=5)

        # 4. Clickstream — what readers are actually consuming this week
        clickstream = get_clickstream_insights(days=7)

        # 5. Build enriched prompt
        user_prompt = self._build_prompt(
            query=query,
            context_snippets=context_snippets,
            trending=trending,
            web_results=web_results,
            clickstream=clickstream,
        )

        # 6. Call Gemini
        self.memory.add("user", user_prompt)
        raw_text = self._call_gemini()
        self.memory.add("model", raw_text)

        # 7. Parse JSON → ArticleIdea[]
        article_ideas = self._parse_ideas(raw_text)
        logger.info("José produced %d article ideas", len(article_ideas))

        return ResearchReport(
            query=query,
            trending_topics=trending,
            article_ideas=article_ideas,
            context_snippets=context_snippets,
            raw_web_results=web_results,
        )

    def chat(self, user_input: str) -> str:
        """
        Free conversational mode — no structured JSON parsing.
        Useful for direct journalist interaction or quick tests.
        """
        self.memory.add("user", user_input)
        reply = self._call_gemini()
        self.memory.add("model", reply)
        return reply

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_gemini(self) -> str:
        """Send current memory to Gemini and return the raw text response."""
        response = self._client.models.generate_content(
            model=CHAT_MODEL,
            contents=self._messages_to_contents(self.memory.as_messages()),
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        return response.candidates[0].content.parts[0].text

    def _system_prompt(self) -> str:
        return self.SYSTEM_PROMPT.format(
            newspaper=self.newspaper_name,
            region=self.region,
        )

    def _build_prompt(
        self,
        query: str,
        context_snippets: list[str],
        trending: list[str],
        web_results: list[dict],
        clickstream: dict | None = None,
    ) -> str:
        ctx_block = (
            "\n".join(f"- {s[:300]}" for s in context_snippets)
            or "No hay contexto previo disponible."
        )
        trend_block = ", ".join(trending[:5]) or "No disponible."
        web_block = (
            "\n".join(
                f"* {r['title']} ({r['source']}): {r['snippet'][:200]}"
                for r in web_results
            )
            or "Sin resultados web."
        )
        click_block = (
            format_insights_for_prompt(clickstream)
            if clickstream
            else "Sin datos de comportamiento de lectores todavía."
        )

        return (
            f"RESEARCH QUERY: {query}\n\n"
            f"HISTORICAL NEWSPAPER CONTEXT (RAG):\n{ctx_block}\n\n"
            f"TODAY'S TRENDING TOPICS:\n{trend_block}\n\n"
            f"RECENT NEWS FROM THE WEB:\n{web_block}\n\n"
            f"{click_block}\n\n"
            f"Based on all the information above, propose 3 article ideas for "
            f"{self.newspaper_name}. Prioritise topics that combine external trends "
            f"with high historical reader engagement. "
            f"Respond ONLY with the JSON object specified in your instructions — "
            f"no markdown fences, no additional text."
        )

    def _messages_to_contents(self, messages: list[dict]) -> list[types.Content]:
        return [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]

    def _parse_ideas(self, raw_text: str) -> list[ArticleIdea]:
        """
        Parse Gemini's JSON response into a list of ArticleIdea.

        Handles three common response shapes:
          1. Pure JSON object                     → ideal case
          2. ```json ... ``` markdown fence       → strip and parse
          3. Unparseable text                     → safe fallback
        """
        try:
            clean = raw_text.strip()

            # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            if clean.startswith("```"):
                # Find the end fence; take only the content between them
                first_newline = clean.find("\n")
                last_fence    = clean.rfind("```")
                if first_newline != -1 and last_fence > first_newline:
                    clean = clean[first_newline:last_fence].strip()

            data = json.loads(clean)
            ideas_raw = data.get("article_ideas", [])

            return [
                ArticleIdea(
                    title=i.get("title", "Sin título"),
                    angle=i.get("angle", ""),
                    category=i.get("category", "general"),
                    local_relevance_score=float(i.get("local_relevance_score", 0.5)),
                    sources=i.get("sources", []),
                    keywords=i.get("keywords", []),
                    priority=i.get("priority", "media"),
                )
                for i in ideas_raw
                if isinstance(i, dict)
            ]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("José._parse_ideas failed (%s). Using fallback.", exc)
            return [
                ArticleIdea(
                    title="Investigación pendiente de parsear",
                    angle=raw_text[:200],
                    category="general",
                    local_relevance_score=0.0,
                    priority="baja",
                )
            ]