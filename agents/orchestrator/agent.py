"""
agents/orchestrator/agent.py
─────────────────────────────
Orchestrator — newspaper_ai
────────────────────────────
Coordinates all agents using LangChain tools. Exposes each agent
as a @tool so a LangChain AgentExecutor can call them in sequence.

Pipeline:
    1. José    → research trending topics → list[ArticleIdea]
    2. Camila  → fact-check ideas         → list[FactCheckResult]
    3. [PAUSE] → journalist selects ideas
    4. Manuel  → generate article         → CreateArticle
    5. Asti    → generate social pack     → SocialMediaPack  ┐ parallel
    6. Mauro   → setup context            → ready to chat    ┘

LangChain usage:
    The AgentExecutor drives the pipeline using the tools below.
    Each tool is a thin wrapper around the real agent's .run() method.

Local usage:
    Set ANTHROPIC_API_KEY in .env 
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain.tools import tool
#from langchain.agents import create_tool_calling_agent
#from langchain_core.agents import AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.memory import Memory
from jose_news_research.agent import (
    NewsResearchAgent,
    KnowledgeBase as JoseKnowledgeBase,
    ResearchReport,
    ArticleIdea,
)
from camila_fact_checking.agent import (
    FactCheckingAgent,
    KnowledgeBase as CamilaKnowledgeBase,
    FactCheckResult,
)
from manuel_article_generation.agent import (
    ArticleGenerationAgent,
    KnowledgeBase as ManuelKnowledgeBase,
    CreateArticle,
)
from asti_social_media.agent import (
    SocialMediaAgent,
    KnowledgeBase as AstiKnowledgeBase,
    SocialMediaPack,
)
from mauro_reader_interaction.agent import (
    ReaderInteractionAgent,
    KnowledgeBase as MauroKnowledgeBase,
    ReaderResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

from config import NEWSPAPER_NAME, REGION as PAIS, EMBEDDINGS_DIR, ANTHROPIC_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Output data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """
    Full pipeline output. Returned after the journalist approves ideas
    and Manuel + Asti finish. Mauro has already been set up by this point.
    """
    research_report: ResearchReport
    fact_check_results: list[FactCheckResult]
    approved_ideas: list[ArticleIdea]       # selected by journalist
    article: CreateArticle
    social_pack: SocialMediaPack
    generated_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.now().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "article_title":    self.article.title,
            "article_category": self.article.category,
            "social_pack":      self.social_pack.to_dict(),
            "approved_ideas":   [i.to_dict() for i in self.approved_ideas],
            "generated_at":     self.generated_at,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates all five agents using LangChain tools.

    Instantiation:
        orch = Orchestrator()
        orch.build_agents()   ← initializes all KnowledgeBases and agents

    Pipeline run:
        result = await orch.run_pipeline(query="nutrition trends this week")

    Reader chat:
        response = orch.chat_reader("Is intermittent fasting safe?")
    """

    def __init__(self):
        # Agents are None until build_agents() is called
        self._jose:   NewsResearchAgent      | None = None
        self._camila: FactCheckingAgent      | None = None
        self._manuel: ArticleGenerationAgent | None = None
        self._asti:   SocialMediaAgent       | None = None
        self._mauro:  ReaderInteractionAgent | None = None

        # LangChain executor — built after agents are ready
        #self._executor: AgentExecutor | None = None

        # Pipeline state — shared across tool calls within a run
        self._last_report:       ResearchReport      | None = None
        self._last_fact_results: list[FactCheckResult]      = []
        self._last_article:      CreateArticle       | None = None
        self._last_social_pack:  SocialMediaPack     | None = None

    # ── Initialization ────────────────────────────────────────────────────────

    def build_agents(self) -> None:
        """
        Instantiates all agents with their KnowledgeBases.
        Call this once before running the pipeline.
        """
        print("Building agents...")

        # José
        jose_kb = JoseKnowledgeBase(persist_dir=EMBEDDINGS_DIR)
        self._jose = NewsResearchAgent(
            knowledge_base=jose_kb,
            memory=Memory(max_turns=10),
            newspaper_name=NEWSPAPER_NAME,
            region=PAIS,
        )

        # Camila — instanced separately so Mauro can share the same instance
        camila_kb = CamilaKnowledgeBase(persist_dir=EMBEDDINGS_DIR)
        self._camila = FactCheckingAgent(
            knowledge_base=camila_kb,
            memory=Memory(max_turns=10),
            newspaper_name=NEWSPAPER_NAME,
            region=PAIS,
        )

        # Manuel
        manuel_kb = ManuelKnowledgeBase(persist_dir=EMBEDDINGS_DIR)
        self._manuel = ArticleGenerationAgent(
            knowledge_base=manuel_kb,
            memory=Memory(max_turns=10),
            newspaper_name=NEWSPAPER_NAME,
            region=PAIS,
        )

        # Asti
        asti_kb = AstiKnowledgeBase(persist_dir=EMBEDDINGS_DIR)
        self._asti = SocialMediaAgent(
            knowledge_base=asti_kb,
            memory=Memory(max_turns=10),
            newspaper_name=NEWSPAPER_NAME,
            region=PAIS,
        )

        # Mauro — receives the same Camila instance (shared, not duplicated)
        mauro_kb = MauroKnowledgeBase(persist_dir=EMBEDDINGS_DIR)
        self._mauro = ReaderInteractionAgent(
            knowledge_base=mauro_kb,
            camila=self._camila,            # ← injected, not created internally
            memory=Memory(max_turns=20),
            newspaper_name=NEWSPAPER_NAME,
            region=PAIS,
        )

        # Build LangChain executor with tools wired to this instance
        self._executor = self._build_executor()
        print("All agents ready.\n")

    def _build_executor(self) -> AgentExecutor:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=ANTHROPIC_API_KEY,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:        
        """
        Main entry point for the LangChain pipeline.
        Accepts free-form input from the journalist and routes accordingly.

        Args:
            user_input: e.g. "Research nutrition trends and write an article"

        Returns:
            Orchestrator's final response as a string.
        """
        raise NotImplementedError("LangChain mode disabled — use run_pipeline_async().")
        if not self._executor:
            raise RuntimeError("Call build_agents() before run().")

        result = self._executor.invoke({"input": user_input})
        return result["output"]

    def chat_reader(self, message: str) -> ReaderResponse:
        """
        Direct entry point for reader chat — bypasses LangChain executor.
        Used by the Streamlit UI's reader chatbot component.

        Args:
            message: Reader's message.

        Returns:
            ReaderResponse with Mauro's reply and metadata.
        """
        if not self._mauro:
            raise RuntimeError("Call build_agents() before chat_reader().")
        return self._mauro.chat(message)

    async def run_pipeline_async(self, query: str) -> OrchestratorResult:
        """
        Async version of the pipeline for production use.
        Runs José + Camila in parallel, then Manuel, then Asti + Mauro setup
        in parallel.

        Args:
            query: Research query for José.

        Returns:
            OrchestratorResult with all pipeline outputs.
        """
        if not self._jose:
            raise RuntimeError("Call build_agents() before run_pipeline_async().")
        
        t0 = time.time()

        # ── Step 1: José + Camila in parallel ─────────────────────────────────
        print("Step 1/3 — José researching trends + Camila pre-loading sources...")
        report, _ = await asyncio.gather(
            asyncio.to_thread(self._jose.run, query),
            asyncio.to_thread(self._camila.kb.retrieve, query, 4),
        )
        self._last_report = report

        t1=time.time()
        print(f"[TIMING] Stage 1 parallel (José + Camila warmup): {t1-t0:.2f}s")

        # ── Step 2: Camila scores José's ideas ────────────────────────────────
        print("Step 2/3 — Camila fact-checking ideas...")
        fact_results = await asyncio.to_thread(
            self._camila.run_batch, report.article_ideas
        )
        self._last_fact_results = fact_results
        t2 = time.time()
        print(f"[TIMING] Stage 2 (Camila fact-check batch): {t2-t1:.2f}s")

        # Attach verdict + confidence to each ArticleIdea
        for idea, result in zip(report.article_ideas, fact_results):
            idea.confidence_score = result.confidence
            idea.verdict = result.verdict

        # ── Step 3: Journalist selects (console fallback) ─────────────────────
        approved_ideas = self._select_ideas(report.article_ideas)

        # ── Step 4: Manuel generates article ─────────────────────────────────
        print("Step 3/3 — Manuel writing article...")
        article = await asyncio.to_thread(self._manuel.run, approved_ideas[0])
        self._last_article = article
        t3 = time.time()
        print(f"[TIMING] Stage 3 (Manuel article): {t3-t2:.2f}s")

        # ── Step 5: Asti + Mauro setup in parallel ────────────────────────────
        print("Publishing — Asti creating social pack, Mauro setting up...")
        social_pack, _ = await asyncio.gather(
            asyncio.to_thread(self._asti.run, article),
            asyncio.to_thread(lambda: None),  # placeholder — Mauro.setup is sync
        )
        self._last_social_pack = social_pack
        self._mauro.setup(article, social_pack)
        t4 = time.time()
        print(f"[TIMING] Stage 4 parallel (Asti + Mauro): {t4-t3:.2f}s")
        print(f"[TIMING] ──────────────────────────────────")
        print(f"[TIMING] Total pipeline: {t4-t0:.2f}s")
        print(f"[TIMING] Throughput: {3600/(t4-t0):.0f} articles/hour (theoretical)")
        
        return OrchestratorResult(
            research_report=report,
            fact_check_results=fact_results,
            approved_ideas=approved_ideas,
            article=article,
            social_pack=social_pack,
        )

    # ── Private helpers ───────────────────────────────────────────────────────
    def _select_ideas(self, ideas: list[ArticleIdea]) -> list[ArticleIdea]:
        if not os.isatty(0) or os.getenv("AUTO_SELECT_IDEAS"):
            best = sorted(ideas, key=lambda x: x.local_relevance_score, reverse=True)
            return [best[0]] if best else ideas[:1]        
        # Local dev mode
        print("\n" + "=" * 60)
        for i, idea in enumerate(ideas, 1):
            print(f"\n  [{i}] {idea.title}")
        print("\n" + "-" * 60)
        while True:
            raw = input("Enter idea numbers to proceed (comma-separated, e.g. 1,3): ").strip()
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                selected = [ideas[i - 1] for i in indices if 1 <= i <= len(ideas)]
                if selected:
                    return selected
            except (ValueError, IndexError):
                print("  Invalid input. Try again.")
