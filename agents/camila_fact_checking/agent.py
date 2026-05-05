"""
agents/camila_fact_checking/agent.py
─────────────────────────────────────
Fact Checking Agent — Camila
─────────────────────────────
Responsibility: Verifies ArticleIdeas from Jose and handles
fact-checking requests from Mauro's reader chatbot.
Architecture:
     Orchestrator / Mauro
          │
          ▼ ArticleIdea (from Jose) or str (reader claim)
    FactCheckingAgent.run(idea) / .verify_url(text)
          │
          ├─► KnowledgeBase.retrieve()   ← fact_checking/ + news_research/ + article_published/
          ├─► web_search()               ← RSS / mock fallback
          └─► LLM generate_content()     ← anthropic key
                    │
                    ├── FactCheckResult   (pipeline: idea + verdict + confidence + reason + sources)
                    └── VerificationResult (reader: input_text + verdict + confidence + reason + sources)
Dependencies:
    pip install google-genai chromadb requests feedparser
Local usage (without GCloud):
    Set ANTHROPIC_API_KEY in .env 
"""

from __future__ import annotations
from config import (NEWSPAPER_NAME, REGION as PAIS)

import json
import os
import logging
logger = logging.getLogger("newspaper_ai.camila")
from dataclasses import dataclass, field

from core.llm import call_llm
from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from core.models import ArticleIdea, FactCheckResult, VerificationResult
from tools.search_tools import web_search

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Three collections:
    - fact_checking/     (own) → known fake news patterns, grows over time
    - news_research/     (Jose, read-only) → research context
    - article_published/ (Manuel, read-only) → published articles
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        # Own RAG → fake news pattern repository
        self._fact_store = VectorStore(
            collection_name="fact_checking",
            persist_dir=f"{persist_dir}/fact_checking",
        )
        # Jose's RAG → read-only, research context
        self._research_store = VectorStore(
            collection_name="news_research",
            persist_dir=f"{persist_dir}/news_research",
        )
        # Manuel's RAG → read-only, published articles
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def add_fake_news_example(self, doc: dict) -> None:
        """
        Indexes a fake news example into the fact_checking collection.
        doc = {"title": ..., "category": ..., "content": ...}
        """
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._fact_store.upsert(texts=texts, metadatas=metas)

    def add_fake_news_examples(self, docs: list[dict]) -> None:
        for doc in docs:
            self.add_fake_news_example(doc)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        Queries all three collections and returns the top_k most
        relevant results sorted by similarity score.
        """
        fact_results     = self._fact_store.query(query, top_k=top_k)
        research_results = self._research_store.query(query, top_k=top_k)
        article_results  = self._published_store.query(query, top_k=top_k)

        all_results = fact_results + research_results + article_results
        all_results.sort(key=lambda r: r.score)  
        return [r.text for r in all_results[:top_k]]

    def count(self) -> int:
        return self._fact_store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Fact Checking Agent
# ─────────────────────────────────────────────────────────────────────────────

class FactCheckingAgent:
    """
    Verifies ArticleIdeas produced by Jose and claims submitted via Mauro.
    run(idea) flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Retrieve fake news patterns and context from RAG             │
    │ 2. Web search the main claim                                    │
    │ 3. Build enriched verification prompt                           │
    │ 4. Call Gemini with specialized system prompt                   │
    │ 5. Parse response → FactCheckResult                             │
    │ 6. If untruthful → persist to fact_checking RAG                 │
    └─────────────────────────────────────────────────────────────────┘
    verify_url(text) flow: same steps but input is free text from a reader,
    returns VerificationResult instead of FactCheckResult.
    """

    SYSTEM_PROMPT = """
You are Camila, the Fact-Checking Agent of a local nutrition newspaper.
Your mission: evaluate the credibility of article ideas produced by Jose
and claims submitted by readers through Mauro's chatbot.
PERSONALITY:
- Observant, analytical, and committed to the truth
- You always cross-reference information against reliable sources
- You pay close attention to detail — fake news often hides in plain sight
- You rely on verified sources: WHO, Spanish Ministry of Health, PubMed
RESTRICTIONS:
- Never invent data or specific sources; if you don't have information, say so
- Do not verify topics outside the nutrition and health domain
- Always prioritize verifiability over speed
VERIFICATION RULES:
- Cross-reference the claim against at least 2 independent sources
- Evaluate both the title and the angle of the article idea
- Consider whether the claim is supported by scientific consensus
- Assign a confidence score based on the evidence found
VERDICT SCALE:
- "truthful"   → confidence 0.7 - 1.0  (supported by reliable sources)
- "doubtful"   → confidence 0.4 - 0.69 (conflicting or insufficient evidence)
- "untruthful" → confidence 0.0 - 0.39 (contradicted by reliable sources)
OUTPUT FORMAT:
When asked to verify a claim, ALWAYS respond with valid JSON:
{
  "verdict": "truthful|doubtful|untruthful",
  "confidence": 0.85,
  "reason": "Detailed explanation of the verdict based on sources found...",
  "sources": ["source1", "source2"]
}
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

    # ── Pipeline method (Jose → Orchestrator → Camila) ───────────────────────

    def run(self, idea: ArticleIdea) -> FactCheckResult:
        """
        Verifies an ArticleIdea and returns a fact-check verdict.
        Args:
            idea: ArticleIdea produced by Jose's NewsResearchAgent.
        Returns:
            FactCheckResult with verdict, confidence, reason and sources.
        """
        # 1. RAG — retrieve similar fake news patterns and context
        context_snippets = self.kb.retrieve(idea.title, top_k=4)

        # 2. Web search the main claim
        web_results = web_search(idea.title, num_results=5)

        # 3. Build verification prompt
        user_prompt = self._build_prompt(
            idea=idea,
            context_snippets=context_snippets,
            web_results=web_results,
        )

        # 4. Call Gemini
        self.memory.add("user", user_prompt)

        raw_text = call_llm(
            prompt=self.memory.as_messages(),
            system=self._personalized_system_prompt(),
            model="fast",
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        self.memory.add("assistant", raw_text)

        # 5. Parse response → FactCheckResult
        result = self._parse_result(raw_text, idea)

        # 6. If untruthful → persist to RAG for future reference
        self._persist_if_fake(result.verdict, idea.title, result.reason, idea.category)

        return result

    def run_batch(self, ideas: list[ArticleIdea]) -> list[FactCheckResult]:
        """
        Verifies a list of ArticleIdeas and returns a verdict for each.
        Results are sorted by confidence descending so the journalist
        sees the most reliable ideas first.
        """
        results = [self.run(idea) for idea in ideas]
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # ── Reader path (Mauro → Camila) ─────────────────────────────────────────

    def verify_url(self, input_text: str) -> VerificationResult:
        """
        Verifies a raw claim or URL submitted by a reader via Mauro.
        Unlike run(), this method has no ArticleIdea — it works with free text.
        Args:
            input_text: Raw claim, headline, or URL from the reader.
        Returns:
            VerificationResult with verdict, confidence, reason and sources.
        """

        if input_text.startswith("http"):
             try:
                 import requests, re
                 r = requests.get(input_text, timeout=5)
                 clean = re.sub(r'<[^>]+>', ' ', r.text)
                 clean = re.sub(r'\s+', ' ', clean).strip()
                 input_text = input_text + "\n\nContent: " + r.text[:1000]
             except Exception:
                 pass
        # 1. RAG — retrieve fake news patterns similar to this claim
        context_snippets = self.kb.retrieve(input_text, top_k=4)

        # 2. Web search the claim
        web_results = web_search(input_text, num_results=5)

        # 3. Build prompt
        ctx_block = (
            "\n".join(f"- {s[:300]}" for s in context_snippets)
            or "No prior context available."
        )
        web_block = (
            "\n".join(
                f"• {r['title']} ({r['source']}): {r['snippet'][:200]}"
                for r in web_results
            )
            or "No web results found."
        )

        user_prompt = f"""
READER CLAIM TO VERIFY:
{input_text}
SIMILAR PATTERNS FROM FACT-CHECKING RAG:
{ctx_block}
WEB SEARCH RESULTS:
{web_block}
Evaluate this claim and respond with the JSON format specified in your instructions.
""".strip()

        # 4. Call Anthropic
        self.memory.add("user", user_prompt)

        raw_text = call_llm(
            prompt=self.memory.as_messages(),
            system=self._personalized_system_prompt(),
            model="fast",
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        self.memory.add("assistant", raw_text)

        # 5. Parse → VerificationResult
        result = self._parse_verification(raw_text, input_text)

        # 6. Persist if fake
        self._persist_if_fake(result.verdict, input_text, result.reason, "reader_submission")

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _personalized_system_prompt(self) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Newspaper: {self.newspaper_name}. Region: {self.region}."
        )

    def _build_prompt(
        self,
        idea: ArticleIdea,
        context_snippets: list[str],
        web_results: list[dict],
    ) -> str:
        ctx_block = (
            "\n".join(f"- {s[:300]}" for s in context_snippets)
            or "No prior context available."
        )
        sources_block = ", ".join(idea.sources) or "No sources provided by Jose."
        web_block = (
            "\n".join(
                f"• {r['title']} ({r['source']}): {r['snippet'][:200]}"
                for r in web_results
            )
            or "No web results found."
        )

        return f"""
CLAIM TO VERIFY: {idea.title}
ANGLE: {idea.angle}
CATEGORY: {idea.category}
SOURCES PROVIDED BY JOSE: {sources_block}
FAKE NEWS PATTERNS FROM RAG:
{ctx_block}
WEB SEARCH RESULTS:
{web_block}
Based on all the above, verify the credibility of this article idea
for {self.newspaper_name}. Respond in the JSON format specified.
""".strip()

    def _persist_if_fake(
        self,
        verdict: str,
        title: str,
        reason: str,
        category: str,
    ) -> None:
        """Persists untruthful content to the RAG so future checks benefit from it."""
        if verdict == "untruthful":
            self.kb.add_fake_news_example({
                "title":    title,
                "content":  reason,
                "category": category,
            })

    def _parse_result(self, raw_text: str, idea: ArticleIdea) -> FactCheckResult:
        """
        Parses Gemini's JSON response into a FactCheckResult.
        Falls back to a safe 'doubtful' verdict if parsing fails.
        """
        try:
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
            return FactCheckResult(
                idea=idea,
                verdict=data.get("verdict", "doubtful"),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                sources=data.get("sources", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Parse failed, raw: %s", raw_text[:200])
            return FactCheckResult(
                idea=idea,
                verdict="doubtful",
                reason="No se pudo verificar la información.",
                confidence=0.0,
                sources=[],
            )

    def _parse_verification(self, raw_text: str, input_text: str) -> VerificationResult:
        """
        Parses Gemini's JSON response into a VerificationResult.
        Same logic as _parse_result but returns VerificationResult.
        Falls back to 'doubtful' if parsing fails.
        """
        try:
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
            return VerificationResult(
                input_text=input_text,
                verdict=data.get("verdict", "doubtful"),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                sources=data.get("sources", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Parse failed, raw: %s", raw_text[:200])
            return VerificationResult(
                 input_text=input_text,
                 verdict="no_information",
                 reason="No se pudo verificar la información.",
                 confidence=0.0,
                 sources=[],
            )
