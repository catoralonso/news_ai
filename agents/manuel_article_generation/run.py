"""
agents/manuel_article_generation/run.py
────────────────────────────────────────
Demo / local development script for the Article Generation Agent.

Usage:
    python agents/manuel_article_generation/run.py

Environment variables (.env or export):
    GEMINI_API_KEY=AIza...     ← for local development (AI Studio)
    GOOGLE_CLOUD_PROJECT=...   ← for Vertex AI (production)
    GOOGLE_CLOUD_REGION=us-central1
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.manuel_article_generation.agent import ArticleGenerationAgent, KnowledgeBase
from agents.jose_news_research.agent import ArticleIdea
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Sample articles to populate the style RAG
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    {
        "title": "Los alimentos de temporada que fortalecen tu sistema inmune este invierno",
        "date": "2025-01-10",
        "category": "nutrición",
        "content": (
            "Con la llegada del frío, los nutricionistas recomiendan incorporar "
            "alimentos ricos en vitamina C y zinc a la dieta diaria. "
            "Las naranjas, mandarinas y el brócoli son opciones accesibles y de "
            "temporada que ayudan a reforzar las defensas del organismo. "
            "La doctora Martínez, nutricionista del centro de salud municipal, "
            "señala que una dieta variada es más efectiva que cualquier suplemento. "
            "Los mercados locales ofrecen estas opciones a precios asequibles para todas las familias."
        ),
    },
    {
        "title": "Mitos y verdades sobre las dietas detox que arrasan en redes sociales",
        "date": "2025-02-03",
        "category": "nutrición",
        "content": (
            "Cada enero se repite el mismo fenómeno: las dietas detox inundan "
            "Instagram y TikTok prometiendo resultados milagrosos en pocos días. "
            "Pero los expertos advierten que el hígado y los riñones ya realizan "
            "esa función de forma natural y que estos regímenes pueden ser perjudiciales. "
            "El nutricionista Carlos Ruiz explica que la clave está en mantener "
            "hábitos sostenibles en el tiempo, no en soluciones rápidas. "
            "Consultar a un profesional antes de seguir cualquier dieta es siempre lo más recomendable."
        ),
    },
    {
        "title": "Guía práctica para leer etiquetas nutricionales en el supermercado",
        "date": "2025-02-20",
        "category": "bienestar",
        "content": (
            "Saber interpretar una etiqueta nutricional puede marcar la diferencia "
            "a la hora de elegir productos más saludables sin gastar más dinero. "
            "Los expertos recomiendan fijarse primero en el tamaño de la porción, "
            "luego en el contenido de azúcares añadidos y grasas saturadas. "
            "Un producto con menos de cinco ingredientes suele ser más natural. "
            "La Asociación de Consumidores local ofrece talleres gratuitos cada mes "
            "para aprender a comprar de forma más consciente e informada."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Mock ArticleIdea — simulates what José would produce
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_IDEA = ArticleIdea(
    title="5 alimentos de temporada que no deberían faltar en tu dieta este mes",
    angle="Guía práctica con opciones accesibles disponibles en mercados locales",
    category="nutrición",
    local_relevance_score=0.90,
    sources=["asociacion-nutricionistas.org", "mercado-municipal.es"],
    keywords=["nutrición", "alimentos temporada", "dieta", "salud", "mercado local"],
    priority="alta",
)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ARTICLE GENERATION AGENT — Local Demo")
    print("=" * 60)

    # 1. Initialize components
    kb = KnowledgeBase(persist_dir="data/embeddings")
    memory = Memory(max_turns=10)

    # 2. Load style articles into RAG
    print(f"\nLoading {len(SAMPLE_ARTICLES)} style articles into ChromaDB...")
    kb.add_style_documents(SAMPLE_ARTICLES)
    print(f"   → {kb.count()} chunks indexed\n")

    # 3. Create agent
    agent = ArticleGenerationAgent(
        knowledge_base=kb,
        memory=memory,
    )

    # 4. Display the idea to be developed
    print("RECEIVED IDEA (mock from José):")
    print(f"   Title:      {SAMPLE_IDEA.title}")
    print(f"   Angle:      {SAMPLE_IDEA.angle}")
    print(f"   Category:   {SAMPLE_IDEA.category}")
    print(f"   Relevance:  {SAMPLE_IDEA.local_relevance_score:.0%}")
    print(f"   Priority:   {SAMPLE_IDEA.priority}\n")

    # 5. Generate article
    print("Generating article...\n")
    article = agent.run(SAMPLE_IDEA)

    # 6. Display result
    print("=" * 60)
    print("  GENERATED ARTICLE")
    print("=" * 60)
    print(f"\nTitle:     {article.title}")
    print(f"Category:  {article.category}")
    print(f"Angle:     {article.angle}")
    print(f"Local relevance: {article.local_relevance_score:.0%}")
    if article.keywords:
        print(f"Keywords:  {', '.join(article.keywords)}")
    if article.sources:
        print(f"Sources:   {', '.join(article.sources)}")
    print(f"\n{article.article_content}")

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
        reply = agent.chat(user_input)
        print(f"\nManuel: {reply}\n")


if __name__ == "__main__":
    main()
