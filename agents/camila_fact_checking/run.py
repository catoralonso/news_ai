"""
agents/camila_fact_checking/run.py
────────────────────────────────────
Demo / local development script for the Fact Checking Agent.

Usage:
    python agents/camila_fact_checking/run.py

Environment variables (.env or export):
    GEMINI_API_KEY=AIza...     ← local development (AI Studio)
    GOOGLE_CLOUD_PROJECT=...   ← Vertex AI (production)
    GOOGLE_CLOUD_REGION=us-central1
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.camila_fact_checking.agent import FactCheckingAgent, KnowledgeBase
from agents.jose_news_research.agent import ArticleIdea
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Sample fake news examples to populate Camila's RAG
# ─────────────────────────────────────────────────────────────────────────────

FAKE_NEWS_EXAMPLES = [
    {
        "title": "Lemon water detox cures cancer",
        "date": "2024-01-01",
        "category": "dietas",
        "content": (
            "Claims that drinking lemon water can cure or prevent cancer "
            "are not supported by scientific evidence. The WHO and major "
            "oncology associations have repeatedly debunked this claim. "
            "While vitamin C has antioxidant properties, no food or drink "
            "has been proven to cure cancer. This type of content spreads "
            "through social media and can delay patients from seeking "
            "proper medical treatment."
        ),
    },
    {
        "title": "Intermittent fasting completely reverses type 2 diabetes",
        "date": "2024-02-01",
        "category": "enfermedades y dieta",
        "content": (
            "While intermittent fasting can improve insulin sensitivity and "
            "help manage blood sugar levels, claims that it completely reverses "
            "type 2 diabetes are misleading. Some studies show remission in "
            "early-stage cases combined with significant weight loss, but this "
            "is not universal. The Spanish Ministry of Health recommends "
            "consulting a specialist before making any dietary changes for "
            "diabetes management."
        ),
    },
    {
        "title": "Brown sugar does not cause weight gain unlike white sugar",
        "date": "2024-03-01",
        "category": "nutrición",
        "content": (
            "Brown sugar and white sugar have virtually identical caloric "
            "content and glycemic index. The minimal difference in mineral "
            "content does not make brown sugar a healthier alternative for "
            "weight management. This is a widespread misconception amplified "
            "by marketing. Both types should be consumed in moderation "
            "according to WHO guidelines."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Sample ArticleIdeas — simulates Jose's output
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_IDEAS = [
    ArticleIdea(
        title="Intermittent fasting reverses type 2 diabetes in 30 days",
        angle="New study confirms complete reversal with 16/8 protocol",
        category="enfermedades y dieta",
        local_relevance_score=0.85,
        sources=["diabetes-care.org", "pubmed.ncbi.nlm.nih.gov"],
        keywords=["ayuno intermitente", "diabetes tipo 2", "salud"],
        priority="alta",
    ),
    ArticleIdea(
        title="5 seasonal foods that strengthen your immune system this winter",
        angle="Practical guide with affordable options at local markets",
        category="nutrición",
        local_relevance_score=0.90,
        sources=["asociacion-nutricionistas.org", "mercado-municipal.es"],
        keywords=["nutrición", "alimentos temporada", "sistema inmune"],
        priority="alta",
    ),
    ArticleIdea(
        title="Omega-3 supplements vs oily fish: what does the evidence say?",
        angle="Comparing effectiveness of supplements vs natural sources",
        category="suplementos",
        local_relevance_score=0.70,
        sources=["examine.com", "pubmed.ncbi.nlm.nih.gov"],
        keywords=["omega-3", "suplementos", "pescado azul"],
        priority="media",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FACT CHECKING AGENT — Local Demo")
    print("=" * 60)

    # 1. Initialize components
    kb = KnowledgeBase(persist_dir="data/embeddings")
    memory = Memory(max_turns=10)

    # 2. Load fake news examples into RAG
    print(f"\nLoading {len(FAKE_NEWS_EXAMPLES)} fake news examples into ChromaDB...")
    kb.add_fake_news_examples(FAKE_NEWS_EXAMPLES)
    print(f"   → {kb.count()} chunks indexed\n")

    # 3. Create agent
    agent = FactCheckingAgent(
        knowledge_base=kb,
        memory=memory,
    )

    # 4. Run batch verification on Jose's ideas
    print(f"Verifying {len(SAMPLE_IDEAS)} article ideas from Jose...\n")
    results = agent.run_batch(SAMPLE_IDEAS)

    # 5. Display results sorted by confidence
    print("=" * 60)
    print("  FACT CHECK RESULTS (sorted by confidence)")
    print("=" * 60)

    verdict_icon = {
        "truthful":       "✓",
        "doubtful":       "?",
        "untruthful":     "✗",
        "no_information": "—",
    }

    for result in results:
        icon = verdict_icon.get(result.verdict, "?")
        print(f"\n  [{icon} {result.verdict.upper()}  {result.confidence:.0%}] {result.idea.title}")
        print(f"      Reason:  {result.reason[:200]}")
        if result.sources:
            print(f"      Sources: {', '.join(result.sources)}")

    print("\n" + "=" * 60)
    print("  Conversational mode (type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        # Quick fact check from free text — wrap in a minimal ArticleIdea
        idea = ArticleIdea(
            title=user_input,
            angle="",
            category="general",
            local_relevance_score=0.5,
            sources=[],
            keywords=[],
            priority="media",
        )
        result = agent.run(idea)
        icon = verdict_icon.get(result.verdict, "?")
        print(f"\nCamila [{icon} {result.verdict} {result.confidence:.0%}]: {result.reason}\n")


if __name__ == "__main__":
    main()
