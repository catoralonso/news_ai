"""
agents/orchestrator/run.py
───────────────────────────
Demo / local development script for the full pipeline.

Usage:
    python agents/orchestrator/run.py

Modes:
    1. Pipeline mode  — journalist drives the full pipeline via LangChain
    2. Async mode     — runs the full pipeline programmatically (no LLM routing)
    3. Reader mode    — simulates a reader chatting with Mauro directly

Environment variables (.env or export):
    GEMINI_API_KEY=AIza...
    NEWSPAPER_NAME=Nutrición AI
    REGION_NEWS=ES
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.orchestrator.agent import Orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_divider(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(f"\n{line}")


def _print_result(result) -> None:
    """Pretty-prints an OrchestratorResult."""
    _print_divider("PIPELINE RESULT")
    print(f"\n  Article:   {result.article.title}")
    print(f"  Category:  {result.article.category}")
    print(f"  Relevance: {result.article.local_relevance_score:.0%}")

    if result.social_pack:
        platforms = [
            p for p in ["twitter", "instagram", "carousel", "newsletter"]
            if getattr(result.social_pack, p) is not None
        ]
        print(f"  Social:    {', '.join(platforms)}")

    print(f"\n  Ideas researched:    {len(result.research_report.article_ideas)}")
    print(f"  Ideas approved:      {len(result.approved_ideas)}")
    print(f"  Generated at:        {result.generated_at}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — LangChain pipeline (journalist drives via natural language)
# ─────────────────────────────────────────────────────────────────────────────

def run_langchain_mode(orch: Orchestrator) -> None:
    """
    Interactive mode where the journalist talks to the Orchestrator
    in natural language. LangChain decides which tools to call.
    """
    _print_divider("PIPELINE MODE — LangChain (journalist-driven)")
    print("""
  Examples of what you can type:
  - "Research nutrition trends in Spain and propose article ideas"
  - "Write an article about idea number 2"
  - "Publish the article and create social media content"
  - "A reader is asking: is intermittent fasting safe?"
  - "exit" to quit
""")

    while True:
        try:
            user_input = input("Journalist: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        print()
        response = orch.run(user_input)
        print(f"\nOrchestrator: {response}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Async pipeline (programmatic, no LLM routing)
# ─────────────────────────────────────────────────────────────────────────────

async def run_async_mode(orch: Orchestrator) -> None:
    """
    Runs the full pipeline programmatically.
    José + Camila in parallel → journalist selects → Manuel → Asti + Mauro.
    """
    _print_divider("ASYNC PIPELINE MODE")

    query = "Tendencias de nutrición y salud digestiva en España esta semana"
    print(f"\n  Query: {query}\n")

    result = await orch.run_pipeline_async(query)
    _print_result(result)

    # Drop into reader chat mode after pipeline completes
    _print_divider("READER CHAT (Mauro is ready)")
    print("  Try asking Mauro a nutrition question or submitting a claim.\n")

    while True:
        try:
            user_input = input("Reader: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        response = orch.chat_reader(user_input)
        print(f"\nMauro: {response.message}")
        if response.fact_check_verdict:
            print(f"  [verdict: {response.fact_check_verdict} "
                  f"| confidence: {response.fact_check_confidence:.0%}]")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Reader chat only (Mauro standalone)
# ─────────────────────────────────────────────────────────────────────────────

def run_reader_mode(orch: Orchestrator) -> None:
    """
    Skips the pipeline — talks directly to Mauro.
    Useful for testing reader interaction without running the full pipeline.
    """
    _print_divider("READER MODE — Mauro standalone")
    print("""
  Mauro is ready to chat without a fresh pipeline run.
  Note: no article has been generated in this session,
  so Mauro will rely on his existing RAG only.

  Type "exit" to quit.
""")

    while True:
        try:
            user_input = input("Reader: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        response = orch.chat_reader(user_input)
        print(f"\nMauro: {response.message}")
        if response.fact_check_verdict:
            print(f"  [verdict: {response.fact_check_verdict} "
                  f"| confidence: {response.fact_check_confidence:.0%}]")
        if response.was_escalated:
            print("  [⚠ escalated to journalist team]")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "1": ("LangChain pipeline (journalist-driven)", run_langchain_mode),
    "2": ("Async pipeline (programmatic)",          None),   # async — handled separately
    "3": ("Reader chat only (Mauro standalone)",    run_reader_mode),
}


def main() -> None:
    _print_divider("NEWSPAPER AI — Orchestrator Demo")

    # Select mode
    print("\n  Select mode:")
    for key, (label, _) in MODES.items():
        print(f"  [{key}] {label}")
    print()

    mode = input("Mode (1/2/3): ").strip()
    if mode not in MODES:
        print("Invalid mode. Defaulting to [1].")
        mode = "1"

    # Build all agents
    orch = Orchestrator()
    orch.build_agents()

    # Run selected mode
    if mode == "2":
        asyncio.run(run_async_mode(orch))
    else:
        _, fn = MODES[mode]
        fn(orch)

    _print_divider("Session ended")


if __name__ == "__main__":
    main()
