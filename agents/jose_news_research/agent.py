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
          ├─► web_search()               ← Google CSE or mock
          ├─► get_trending_topics()      ← pytrends or mock
          ├─► get_clickstream_insights() ← reader behavior on the site
          └─► Gemini generate_content()  ← google-genai (Vertex AI)
                    │
                    └─► Structured response (ArticleIdea[])

Dependencies:
    pip install google-genai chromadb pytrends requests

Local usage (no GCloud):
    Set GEMINI_API_KEY in .env with an AI Studio key.
    Vertex AI activates automatically when PROJECT_ID is detected.
"""

from __future__ import annotations
# for when the orchestaror is build uncomment
# from config import NEWSPAPER_NAME, PAIS 

import json
import os
from dataclasses import dataclass, field
from typing import Any

# ── Google GenAI SDK ──────────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Internal modules ──────────────────────────────────────────────────────────
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

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION   = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
NEWSPAPER_NAME  = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
PAIS            = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL        = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.4


def _build_client() -> genai.Client:
    """
    Authentication priority:
    1. Vertex AI (if PROJECT_ID is in env) → production
    2. GEMINI_API_KEY                       → local development
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
class ArticleIdea:
    title: str
    angle: str                          # suggested journalistic angle
    category: str                       # politics, sports, culture...
    local_relevance_score: float        # 0-1
    sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    priority: str = "media"             # alta / media / baja
    # Populated by Orchestrator after Camila runs — None until then
    confidence_score: float | None = None
    verdict: str | None = None          # "truthful" | "doubtful" | "untruthful"

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ResearchReport:
    query: str
    trending_topics: list[str]
    article_ideas: list[ArticleIdea]
    context_snippets: list[str]         # RAG fragments used
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
    Indexes newspaper documents (historical articles, style guides, context)
    and retrieves them semantically to enrich agent responses.

    Two collections:
    - news_research     -> historical articles loaded by the team
    - article_published -> articles already published by Manuel,
                          so Jose avoids proposing repeated angles
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
        # Merge results from both collections, ranked by similarity score
        news_results      = self._store.query(query, top_k=top_k)
        published_results = self._published_store.query(query, top_k=top_k)

        all_results = news_results + published_results
        all_results.sort(key=lambda r: r.score)     # lower cosine = more similar
        return [r.text for r in all_results[:top_k]]

    def count(self) -> int:
        return self._store.count()


# ─────────────────────────────────────────────────────────────────────────────
# News Research Agent
# ─────────────────────────────────────────────────────────────────────────────

class NewsResearchAgent:
    """
    News research agent -- Jose.

    run(query) pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Retrieve historical newspaper context (RAG)              │
    │ 2. Fetch local trending topics                              │
    │ 3. Search recent news on the web                            │
    │ 4. Read reader behavior from clickstream                    │
    │ 5. Build enriched prompt and call Gemini                    │
    │ 6. Parse response -> ResearchReport                         │
    └─────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
You are Jose, the News Research Agent of a local newspaper.
Your mission: research trending topics, detect coverage opportunities, and
propose article ideas that are relevant to the local community.

PERSONALITY:
- Curious, analytical, and community-oriented
- You always look for the local angle in national or global news
- You prioritize stories with direct impact on residents daily lives
- You always find at least 2 sources before proposing a story

RESTRICTIONS:
- Never invent data or specific sources; if you don't have information, say so
- Do not cover topics outside the scope of local journalism
- Prioritize verifiability of information

OUTPUT FORMAT:
When asked to propose ideas, ALWAYS respond with valid JSON in spanish:
{
  "article_ideas": [
    {
      "title": "Suggested article title",
      "angle": "Specific journalistic angle or approach",
      "category": "nutrition|recipes|wellness|supplements|diets|community",
      "local_relevance_score": 0.0,
      "sources": ["source1", "source2"],
      "keywords": ["keyword1", "keyword2"],
      "priority": "alta|media|baja"
    }
  ],
  "summary": "Executive summary of the current news landscape"
}

RULES for local_relevance_score:
A number between 0.0 and 1.0 indicating how relevant this story is to the local community:
    - 1.0 = directly affects residents (local works, events, municipal politics)
    - 0.5 = national topic with moderate local impact
    - 0.0 = no connection to the local community
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

    # ── Main method ───────────────────────────────────────────────────────────

    def run(self, query: str) -> ResearchReport:
        """
        Runs the full research pipeline.

        Args:
            query: Research topic or question.
                   E.g. "What nutrition topics should we cover this week?"

        Returns:
            ResearchReport with structured article ideas.
        """
        # 1. RAG: historical newspaper context
        context_snippets = self.kb.retrieve(query, top_k=4)

        # 2. Trending topics
        trending = get_trending_topics(region=self.region)

        # 3. Web search
        web_results = web_search(query, num_results=5)

        # 4. Clickstream -- what readers are actually reading this week
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

        # 7. Parse response
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
        Free conversational mode (no structured parsing).
        Useful for quick tests or direct journalist interaction.
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
        return f"{self.SYSTEM_PROMPT}\n\nNewspaper: {self.newspaper_name}. Region: {self.region}."

    def _build_prompt(
        self,
        query: str,
        context_snippets: list[str],
        trending: list[str],
        web_results: list[dict],
        clickstream: dict | None = None,
    ) -> str:
        ctx_block = "\n".join(f"- {s[:300]}" for s in context_snippets) or "No prior context."
        trend_block = ", ".join(trending[:5]) or "Not available."
        web_block = "\n".join(
            f"* {r['title']} ({r['source']}): {r['snippet'][:200]}"
            for r in web_results
        ) or "No web results."

        click_block = (
            format_insights_for_prompt(clickstream)
            if clickstream
            else "No reader behavior data available yet."
        )

        return f"""
RESEARCH QUERY: {query}

HISTORICAL NEWSPAPER CONTEXT (RAG):
{ctx_block}

TODAY'S TRENDING TOPICS:
{trend_block}

RECENT NEWS FROM THE WEB:
{web_block}

{click_block}

Based on all the information above, propose 3 article ideas for {self.newspaper_name}.
Prioritize topics that combine external trends with high historical reader engagement.
Respond in the specified JSON format.
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
        """Extracts ArticleIdea[] from the JSON returned by Gemini."""
        try:
            # Strip markdown code blocks if present: ```json ... ```
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
            ideas_raw = data.get("article_ideas", [])
            return [
                ArticleIdea(
                    title=i.get("title", "Untitled"),
                    angle=i.get("angle", ""),
                    category=i.get("category", "general"),
                    local_relevance_score=float(i.get("local_relevance_score", 0.5)),
                    sources=i.get("sources", []),
                    keywords=i.get("keywords", []),
                    priority=i.get("priority", "media"),
                )
                for i in ideas_raw
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback 
            return [
                ArticleIdea(
                    title="Research pending parse",
                    angle=raw_text[:200],
                    category="general",
                    local_relevance_score=0.0,
                    priority="baja",
                )
            ]
