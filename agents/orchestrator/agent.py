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

Dependencies:
    pip install langchain langchain-google-genai google-genai chromadb

Local usage:
    Set GEMINI_API_KEY in .env — same as all other agents.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.memory import Memory

from agents.jose_news_research.agent import (
    NewsResearchAgent,
    KnowledgeBase as JoseKnowledgeBase,
    ResearchReport,
    ArticleIdea,
)
from agents.camila_fact_checking.agent import (
    FactCheckingAgent,
    KnowledgeBase as CamilaKnowledgeBase,
    FactCheckResult,
)
from agents.manuel_article_generation.agent import (
    ArticleGenerationAgent,
    KnowledgeBase as ManuelKnowledgeBase,
    CreateArticle,
)
from agents.asti_social_media.agent import (
    SocialMediaAgent,
    KnowledgeBase as AstiKnowledgeBase,
    SocialMediaPack,
)
from agents.mauro_reader_interaction.agent import (
    ReaderInteractionAgent,
    KnowledgeBase as MauroKnowledgeBase,
    ReaderResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
NEWSPAPER_NAME  = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
PAIS            = os.getenv("REGION_NEWS", "ES")
EMBEDDINGS_DIR  = os.getenv("CHROMA_PERSIST_DIR", "data/embeddings")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "gemini-2.0-flash")


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
        self._executor: AgentExecutor | None = None

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
        """
        Builds the LangChain AgentExecutor.
        Tools are closures over self so they access the real agent instances.
        """
        llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=GEMINI_API_KEY or None,
        )

        # ── Tools (closures over self) ────────────────────────────────────────

        @tool
        def research_trends(query: str) -> str:
            """
            José: researches trending nutrition topics and proposes article ideas.
            Input: research query (e.g. 'nutrition trends this week in Spain').
            Output: JSON with article ideas and confidence scores.
            """
            report = self._jose.run(query)
            self._last_report = report

            # Enrich ideas with Camila's confidence scores
            fact_results = self._camila.run_batch(report.article_ideas)
            self._last_fact_results = fact_results

            # Attach verdict + confidence to each idea
            for idea, result in zip(report.article_ideas, fact_results):
                idea.confidence_score = result.confidence
                idea.verdict = result.verdict

            # Format output for the journalist
            ideas_summary = []
            for i, idea in enumerate(report.article_ideas, 1):
                verdict_icon = {
                    "truthful":   "✓",
                    "doubtful":   "?",
                    "untruthful": "✗",
                }.get(idea.verdict or "", "·")

                ideas_summary.append({
                    "index":             i,
                    "title":             idea.title,
                    "angle":             idea.angle,
                    "category":          idea.category,
                    "priority":          idea.priority,
                    "local_relevance":   f"{idea.local_relevance_score:.0%}",
                    "verdict":           f"{verdict_icon} {idea.verdict}",
                    "confidence":        f"{idea.confidence_score:.0%}",
                })

            return json.dumps({
                "total_ideas": len(ideas_summary),
                "ideas":       ideas_summary,
                "summary":     report.summary,
            }, ensure_ascii=False, indent=2)

        @tool
        def generate_article(idea_index: int) -> str:
            """
            Manuel: generates a full article from an approved idea.
            Input: index (1-based) of the idea from research_trends output.
            Output: JSON with the generated article.
            """
            if not self._last_report:
                return "Error: run research_trends first."

            ideas = self._last_report.article_ideas
            if idea_index < 1 or idea_index > len(ideas):
                return f"Error: idea_index must be between 1 and {len(ideas)}."

            selected_idea = ideas[idea_index - 1]
            article = self._manuel.run(selected_idea)
            self._last_article = article

            return json.dumps({
                "title":                article.title,
                "category":             article.category,
                "angle":                article.angle,
                "local_relevance_score": article.local_relevance_score,
                "keywords":             article.keywords,
                "sources":              article.sources,
                "article_content":      article.article_content[:500] + "...",
            }, ensure_ascii=False, indent=2)

        @tool
        def publish_article(dummy: str = "") -> str:
            """
            Asti + Mauro: generates social media content and sets up the reader
            chatbot. Run this after generate_article.
            Input: any string (ignored — uses last generated article).
            Output: JSON summary of the social media pack.
            """
            if not self._last_article:
                return "Error: run generate_article first."

            # Run Asti and set up Mauro (simulated parallel — asyncio not needed
            # here because LangChain tools are sync; true async in run_pipeline)
            social_pack = self._asti.run(self._last_article)
            self._last_social_pack = social_pack

            # Set up Mauro with the new article and social pack
            self._mauro.setup(self._last_article, social_pack)

            return json.dumps({
                "status":        "published",
                "article_title": social_pack.article_title,
                "platforms":     [
                    p for p in ["twitter", "instagram", "carousel", "newsletter"]
                    if getattr(social_pack, p) is not None
                ],
                "mauro_ready":   True,
            }, ensure_ascii=False, indent=2)

        @tool
        def chat_with_reader(message: str) -> str:
            """
            Mauro: responds to a reader message.
            Input: reader's question or claim.
            Output: Mauro's response with intent and verdict metadata.
            """
            if not self._mauro:
                return "Error: Mauro is not initialized."

            response = self._mauro.chat(message)
            return json.dumps(response.to_dict(), ensure_ascii=False, indent=2)

        # ── Prompt ────────────────────────────────────────────────────────────

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are the Orchestrator of {NEWSPAPER_NAME}, a local nutrition newspaper.
You coordinate a team of AI agents to research, verify, write, and publish content.

Your agents:
- research_trends: José — finds trending topics, returns ideas with confidence scores
- generate_article: Manuel — writes a full article from a selected idea
- publish_article: Asti + Mauro — creates social media content and activates the reader chatbot
- chat_with_reader: Mauro — handles reader conversations

Standard pipeline:
1. Call research_trends to get ideas
2. Present ideas to the journalist and wait for selection (idea_index)
3. Call generate_article with the selected index
4. Call publish_article to distribute the content
5. Use chat_with_reader for any reader interactions

Always present results clearly to the journalist.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        tools = [research_trends, generate_article, publish_article, chat_with_reader]
        agent = create_tool_calling_agent(llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,               # shows tool calls in terminal
            max_iterations=10,
            handle_parsing_errors=True,
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

        # ── Step 1: José + Camila in parallel ─────────────────────────────────
        print("Step 1/3 — José researching trends + Camila pre-loading sources...")
        report, _ = await asyncio.gather(
            asyncio.to_thread(self._jose.run, query),
            asyncio.to_thread(self._camila.kb.retrieve, query, 4),
        )
        self._last_report = report

        # ── Step 2: Camila scores José's ideas ────────────────────────────────
        print("Step 2/3 — Camila fact-checking ideas...")
        fact_results = await asyncio.to_thread(
            self._camila.run_batch, report.article_ideas
        )
        self._last_fact_results = fact_results

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

        # ── Step 5: Asti + Mauro setup in parallel ────────────────────────────
        print("Publishing — Asti creating social pack, Mauro setting up...")
        social_pack, _ = await asyncio.gather(
            asyncio.to_thread(self._asti.run, article),
            asyncio.to_thread(lambda: None),  # placeholder — Mauro.setup is sync
        )
        self._last_social_pack = social_pack
        self._mauro.setup(article, social_pack)

        return OrchestratorResult(
            research_report=report,
            fact_check_results=fact_results,
            approved_ideas=approved_ideas,
            article=article,
            social_pack=social_pack,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_ideas(self, ideas: list[ArticleIdea]) -> list[ArticleIdea]:
        """
        In API mode: auto-selects the highest priority idea.
        In local dev: console-based selection.
        """
        # En Cloud Run no hay terminal — seleccionar automáticamente
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            best = sorted(ideas, key=lambda x: x.local_relevance_score, reverse=True)
            return [best[0]] if best else ideas[:1]
        
        # Modo local con terminal interactivo
        print("\n" + "=" * 60)
        print("  IDEAS FROM JOSÉ + CAMILA — Select which to proceed with")
        print("=" * 60)
        for i, idea in enumerate(ideas, 1):
            verdict_icon = {
                "truthful":   "✓",
                "doubtful":   "?",
                "untruthful": "✗",
            }.get(idea.verdict or "", "·")
            print(f"\n  [{i}] {idea.title}")
            print(f"      Angle:      {idea.angle}")
            print(f"      Priority:   {idea.priority} | "
                f"Relevance: {idea.local_relevance_score:.0%} | "
                f"Confidence: {verdict_icon} {(idea.confidence_score or 0):.0%}")
        print("\n" + "-" * 60)
        while True:
            raw = input("Enter idea numbers to proceed (comma-separated, e.g. 1,3): ").strip()
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                selected = [ideas[i - 1] for i in indices if 1 <= i <= len(ideas)]
                if selected:
                    return selected
                print("  No valid selection. Try again.")
            except (ValueError, IndexError):
                print("  Invalid input. Try again.")
