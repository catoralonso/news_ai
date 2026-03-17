"""
agents/mauro_reader_interaction/run.py
───────────────────────────────────────
Demo / local development script for the Reader Interaction Agent.

Usage:
    python agents/mauro_reader_interaction/run.py

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

from agents.mauro_reader_interaction.agent import ReaderInteractionAgent, KnowledgeBase
from agents.camila_fact_checking.agent import KnowledgeBase as CamilaKnowledgeBase
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Sample published articles to populate the RAG
# These simulate what Manuel would have already written and indexed.
# The richer this archive, the better Mauro can ground his answers
# and recommend relevant content to readers.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    {
        "title": "5 alimentos de temporada que no deberían faltar en tu dieta este mes",
        "date": "2025-03-01",
        "category": "nutrición",
        "content": (
            "Con la llegada del cambio de estación, los mercados locales se llenan de "
            "productos frescos que no solo son más económicos, sino también más nutritivos "
            "que cualquier suplemento. La doctora Ana Martínez recomienda cinco alimentos "
            "que deberían estar en todas las cocinas este mes: espinacas, naranjas, "
            "brócoli, zanahorias y garbanzos. Todos accesibles por menos de dos euros "
            "el kilo en cualquier mercado local."
        ),
    },
    {
        "title": "Mitos y verdades sobre las dietas detox que arrasan en redes sociales",
        "date": "2025-02-03",
        "category": "nutrición",
        "content": (
            "Cada enero se repite el mismo fenómeno: las dietas detox inundan "
            "Instagram y TikTok prometiendo resultados milagrosos en pocos días. "
            "Los expertos advierten que el hígado y los riñones ya realizan esa "
            "función de forma natural. El nutricionista Carlos Ruiz explica que la "
            "clave está en mantener hábitos sostenibles, no en soluciones rápidas."
        ),
    },
    {
        "title": "Omega-3: qué dice la evidencia científica sobre suplementos vs pescado azul",
        "date": "2025-02-15",
        "category": "suplementos",
        "content": (
            "La pregunta que más nos llega de los lectores: ¿es mejor tomar omega-3 "
            "en cápsulas o comer pescado azul directamente? La respuesta corta es: "
            "el pescado azul gana. Los estudios publicados en PubMed muestran que la "
            "biodisponibilidad de los omega-3 procedentes del salmón, la sardina o "
            "la caballa es significativamente superior a la de los suplementos. "
            "Eso sí, para quienes no toleran el pescado, los suplementos de calidad "
            "certificada son una alternativa válida."
        ),
    },
    {
        "title": "Guía práctica para leer etiquetas nutricionales en el supermercado",
        "date": "2025-02-20",
        "category": "bienestar",
        "content": (
            "Saber interpretar una etiqueta nutricional puede marcar la diferencia "
            "a la hora de elegir productos más saludables sin gastar más. "
            "Los expertos recomiendan fijarse primero en el tamaño de la porción, "
            "luego en el contenido de azúcares añadidos y grasas saturadas. "
            "Un producto con menos de cinco ingredientes suele ser más natural."
        ),
    },
    {
        "title": "Microbiota intestinal: el ecosistema que controla tu salud",
        "date": "2025-03-05",
        "category": "ciencia y evidencia",
        "content": (
            "El 70% de tu sistema inmune vive en tu intestino. La microbiota — el "
            "conjunto de bacterias, hongos y virus que habitan en el intestino — "
            "influye directamente en tu estado de ánimo, tu peso y tu sistema "
            "inmunológico. Los alimentos fermentados como el yogur, el kéfir y el "
            "chucrut son los mejores aliados para mantenerla en equilibrio. "
            "La doctora Pérez advierte que los antibióticos tomados sin necesidad "
            "pueden devastar este ecosistema en días."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Sample FAQ interactions to pre-populate Mauro's reader_interaction RAG
# These simulate past conversations so Mauro recognizes recurring patterns
# from the very first session.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_FAQS = [
    {
        "question": "¿Cuánta proteína debo comer al día?",
        "answer": (
            "Allora! La recomendación general es entre 0.8 y 1.2 gramos de proteína "
            "por kilo de peso corporal al día para personas sedentarias. Si haces "
            "ejercicio regularmente, puedes subir hasta 1.6-2 gramos. ¡Pero consulta "
            "siempre con un nutricionista para una recomendación personalizada!"
        ),
        "category": "nutrición",
    },
    {
        "question": "¿Las dietas detox funcionan realmente?",
        "answer": (
            "Mamma mia, esta pregunta! La verdad es que no, no funcionan como prometen. "
            "Tu hígado y tus riñones ya hacen ese trabajo de forma natural. Lo que sí "
            "funciona es mantener una dieta variada y equilibrada. ¡Te lo explicamos "
            "todo en nuestro artículo sobre dietas detox!"
        ),
        "category": "nutrición",
    },
    {
        "question": "¿Es malo comer carbohidratos por la noche?",
        "answer": (
            "Dai, este mito hay que desterrarlo! No es cuándo comes los carbohidratos, "
            "sino cuántos comes en total durante el día. El cuerpo no tiene reloj para "
            "decidir si almacena o quema. Lo que importa es el balance calórico global "
            "y la calidad de lo que comes. ¡La hora importa mucho menos de lo que piensas!"
        ),
        "category": "nutrición",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Sample fake news for Camila's RAG
# Needed so Camila has context when Mauro routes a fact-check request
# ─────────────────────────────────────────────────────────────────────────────

CAMILA_FAKE_NEWS = [
    {
        "title": "El agua con limón en ayunas cura el cáncer",
        "date": "2024-01-01",
        "category": "dietas",
        "content": (
            "Esta afirmación no tiene ningún respaldo científico. La OMS y las "
            "principales asociaciones oncológicas han desmentido repetidamente que "
            "ningún alimento o bebida pueda curar el cáncer. Difundir este tipo de "
            "contenido puede retrasar que los pacientes busquen tratamiento médico real."
        ),
    },
    {
        "title": "El azúcar moreno no engorda como el azúcar blanco",
        "date": "2024-03-01",
        "category": "nutrición",
        "content": (
            "El azúcar moreno y el blanco tienen prácticamente el mismo contenido "
            "calórico e índice glucémico. La mínima diferencia en minerales no lo "
            "convierte en una alternativa saludable para el control de peso. "
            "Ambos deben consumirse con moderación según las guías de la OMS."
        ),
    },
]


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


def _print_response(response) -> None:
    """Pretty-prints a ReaderResponse with all its metadata."""
    print(f"\nMauro: {response.message}")

    meta_parts = []
    if response.intent:
        meta_parts.append(f"intent: {response.intent}")
    if response.fact_check_verdict:
        meta_parts.append(
            f"verdict: {response.fact_check_verdict} "
            f"({response.fact_check_confidence:.0%})"
        )
    if response.recommended_article:
        meta_parts.append(f"recommended: \"{response.recommended_article}\"")
    if response.was_escalated:
        meta_parts.append("⚠ escalated to journalist team")

    if meta_parts:
        print(f"\n  [{' | '.join(meta_parts)}]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Scripted demo interactions
# These run automatically before the interactive loop to showcase
# all of Mauro's capabilities during the live demo on March 23.
# ─────────────────────────────────────────────────────────────────────────────

DEMO_INTERACTIONS = [
    # Standard nutrition question — tests RAG grounding + article recommendation
    "¿Qué alimentos me recomiendas para reforzar el sistema inmune?",

    # Fact-check request — routes to Camila, tests the integration
    "He leído que el agua con limón en ayunas cura el cáncer, ¿es verdad?",

    # Another fact-check — doubtful territory
    "¿Es cierto que el ayuno intermitente revierte la diabetes tipo 2 en 30 días?",

    # Question with article recommendation
    "¿Debo tomar omega-3 en cápsulas o es mejor comer pescado?",

    # Off-topic / unclear — tests graceful handling
    "Hola Mauro, ¿cómo estás?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _print_divider("READER INTERACTION AGENT — Local Demo")

    # 1. Initialize Mauro's knowledge base
    kb = KnowledgeBase(persist_dir="data/embeddings")
    memory = Memory(max_turns=20)

    # 2. Load published articles into reader_interaction RAG
    print(f"\nLoading {len(SAMPLE_ARTICLES)} published articles into ChromaDB...")
    for article in SAMPLE_ARTICLES:
        kb.log_faq(
            question=article["title"],
            answer=article["content"],
            category=article["category"],
        )
    print(f"   → {kb.count()} chunks indexed")

    # 3. Pre-populate FAQ patterns from past conversations
    print(f"\nLoading {len(SAMPLE_FAQS)} FAQ patterns into reader_interaction RAG...")
    for faq in SAMPLE_FAQS:
        kb.log_faq(
            question=faq["question"],
            answer=faq["answer"],
            category=faq["category"],
        )
    print(f"   → {kb.count()} chunks indexed")

    # 4. Initialize Camila's knowledge base with fake news examples
    # Camila is lazy-loaded inside Mauro — we just prepare her KB here
    print(f"\nLoading {len(CAMILA_FAKE_NEWS)} fake news examples into Camila's RAG...")
    camila_kb = CamilaKnowledgeBase(persist_dir="data/embeddings")
    camila_kb.add_fake_news_examples(CAMILA_FAKE_NEWS)
    print(f"   → Camila's RAG ready\n")

    # 5. Create Mauro — pass Camila's KB so it's reused, not re-created
    agent = ReaderInteractionAgent(
        knowledge_base=kb,
        camila_knowledge_base=camila_kb,
        memory=memory,
    )

    # 6. Run scripted demo interactions
    _print_divider("SCRIPTED DEMO (showcasing all capabilities)")

    for question in DEMO_INTERACTIONS:
        print(f"\nReader: {question}")
        response = agent.chat(question)
        _print_response(response)
        input("  [Press Enter to continue...]\n")  # paced for live demo

    # 7. Interactive mode — free conversation
    _print_divider("Interactive mode (type 'exit' to quit)")
    print("\n  Tip: try asking a nutrition question, submitting fake news,")
    print("  or saying something off-topic to see how Mauro handles it.\n")

    while True:
        try:
            user_input = input("Reader: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        response = agent.chat(user_input)
        _print_response(response)

    # 8. Show escalated questions if any were generated
    escalation_file = os.path.join(
        os.getenv("ESCALATION_DIR", "data/escalated"),
        "questions.jsonl"
    )
    if os.path.exists(escalation_file):
        _print_divider("ESCALATED QUESTIONS (pending journalist review)")
        with open(escalation_file, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                print(f"\n  [{entry['timestamp'][:19]}] {entry['question']}")
                print(f"   Status: {entry['status']}")


if __name__ == "__main__":
    import json   # needed for escalation display at the end
    main()