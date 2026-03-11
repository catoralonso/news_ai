"""
agents/news_research/run.py
───────────────────────────
Script de demo / desarrollo local del News Research Agent.

Uso:
    python agents/news_research/run.py

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

from agents.news_research.agent import NewsResearchAgent, KnowledgeBase
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Datos de ejemplo para poblar el RAG (simula artículos del periódico)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    {
        "title": "Nueva ruta de transporte conectará el norte y sur del municipio",
        "date": "2025-02-15",
        "category": "comunidad",
        "content": (
            "El ayuntamiento anunció esta semana la creación de una nueva ruta de "
            "autobús que conectará las colonias del norte con el centro histórico. "
            "La medida beneficiará a más de 15 000 vecinos que actualmente tienen "
            "que realizar al menos dos trasbordos para llegar al trabajo. "
            "El presidente municipal señaló que el servicio iniciará en junio. "
            "Los vecinos del barrio de San Marcos celebraron la noticia aunque "
            "piden más frecuencia de paso en horas pico."
        ),
    },
    {
        "title": "Equipo local avanza a semifinales del torneo regional",
        "date": "2025-03-01",
        "category": "deportes",
        "content": (
            "Los Leones de San Cristóbal derrotaron 3-1 al Atlético del Valle en "
            "un emocionante partido disputado en el estadio municipal. "
            "Con este resultado, el equipo avanza a las semifinales del Torneo "
            "Regional de Fútbol Aficionado. El entrenador destacó el trabajo "
            "colectivo y pidió apoyo de la afición para el próximo encuentro. "
            "El partido semifinal se jugará el próximo 20 de marzo."
        ),
    },
    {
        "title": "Festival de cultura local reunirá a 30 artistas del municipio",
        "date": "2025-03-05",
        "category": "cultura",
        "content": (
            "La Casa de la Cultura anuncia el Primer Festival de Arte Local, "
            "que se celebrará del 22 al 24 de marzo en la plaza principal. "
            "Participarán pintores, músicos y artesanos de todas las colonias. "
            "La entrada es gratuita y habrá actividades para niños. "
            "El objetivo es visibilizar el talento local y fomentar el turismo cultural."
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
    print(f"\n📚 Cargando {len(SAMPLE_ARTICLES)} artículos en ChromaDB...")
    kb.add_documents(SAMPLE_ARTICLES)
    print(f"   → {kb.count()} chunks indexados\n")

    # 3. Crear agente
    agent = NewsResearchAgent(
        knowledge_base=kb,
        memory=memory,
        newspaper_name="El Cronista Municipal",
        region="MX",
    )

    # 4. Consulta de investigación
    query = "¿Qué temas de comunidad deberíamos cubrir esta semana?"
    print(f"🔍 Query: {query}\n")

    report = agent.run(query)

    # 5. Mostrar resultados
    print("📈 TRENDING TOPICS:")
    for t in report.trending_topics[:5]:
        print(f"   • {t}")

    print(f"\n💡 IDEAS DE ARTÍCULOS ({len(report.article_ideas)}):")
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
        print(f"\nAgente: {reply}\n")


if __name__ == "__main__":
    main()
