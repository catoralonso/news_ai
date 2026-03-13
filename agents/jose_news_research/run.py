"""
agents/jose_news_research/run.py
───────────────────────────
Script de demo / desarrollo local del News Research Agent.

Uso:
    python agents/jose_news_research/run.py

Variables de entorno (.env o export):
    GEMINI_API_KEY=AIza...     ← para desarrollo local (AI Studio)
    GOOGLE_CLOUD_PROJECT=...   ← para Vertex AI (producción)
    GOOGLE_CLOUD_REGION=us-central1
"""

import json
import os
import sys

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.jose_news_research.agent import NewsResearchAgent, KnowledgeBase
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Datos de ejemplo para poblar el RAG (simula artículos del periódico)
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
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NEWS RESEARCH AGENT — Demo Local")
    print("=" * 60)

    # 1. Inicializar componentes
    kb = KnowledgeBase(persist_dir="data/embeddings")
    memory = Memory(max_turns=10)

    # 2. Cargar artículos de ejemplo en el RAG
    print(f"\nCargando {len(SAMPLE_ARTICLES)} artículos en ChromaDB...")
    kb.add_documents(SAMPLE_ARTICLES)
    print(f"   → {kb.count()} chunks indexados\n")

    # 3. Crear agente
    agent = NewsResearchAgent(
        knowledge_base=kb,
        memory=memory,
    )

    # 4. Consulta de investigación
    query = "¿Qué temas de nutrición deberíamos cubrir esta semana?"
    print(f"Query: {query}\n")

    report = agent.run(query)

    # 5. Mostrar resultados
    print("TRENDING TOPICS:")
    for t in report.trending_topics[:5]:
        print(f"   • {t}")

    print(f"\nIDEAS DE ARTÍCULOS ({len(report.article_ideas)}):")
    for i, idea in enumerate(report.article_ideas, 1):
        print(f"\n  [{i}] {idea.title}")
        print(f"      Ángulo: {idea.angle}")
        print(f"      Categoría: {idea.category} | Prioridad: {idea.priority}")
        print(f"      Relevancia local: {idea.local_relevance_score:.0%}")
        if idea.keywords:
            print(f"      Keywords: {', '.join(idea.keywords)}")

    print("\n" + "=" * 60)
    print("  Modo conversacional (escribe 'salir' para terminar)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        reply = agent.chat(user_input)
        print(f"\nJosé: {reply}\n")


if __name__ == "__main__":
    main()
