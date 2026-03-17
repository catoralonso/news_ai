"""
agents/asti_social_media/run.py
────────────────────────────────
Demo / local development script for the Social Media Agent.

Usage:
    python agents/asti_social_media/run.py

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

from agents.asti_social_media.agent import SocialMediaAgent, KnowledgeBase
from agents.manuel_article_generation.agent import CreateArticle
from core.memory import Memory


# ─────────────────────────────────────────────────────────────────────────────
# Sample post examples to populate Asti's RAG
# These teach Asti the newspaper's voice and what high-performing posts
# look like — the richer these examples, the better the output quality.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_POSTS = [
    {
        "platform": "instagram",
        "content": (
            "¿Sabías que comer mal no siempre significa comer poco saludable? 🥦\n\n"
            "Muchas veces el problema no está en lo que comemos, sino en cómo, "
            "cuándo y con qué combinamos los alimentos. La nutricionista María López "
            "nos explica que pequeños cambios en la rutina diaria pueden tener un "
            "impacto enorme en cómo nos sentimos. Desde añadir más fibra al desayuno "
            "hasta hidratarnos mejor durante el día. La clave está en la constancia, "
            "no en la perfección. ¿Qué pequeño cambio te comprometes a hacer esta semana?"
        ),
        "hashtags": "#nutricion #saludable #bienestardiary #alimentacionsana #vidasana",
        "engagement": "alto",
        "date": "2025-01-15",
        "category": "bienestar",
    },
    {
        "platform": "twitter",
        "content": (
            "El 70% de tu sistema inmune vive en tu intestino. "
            "Lo que comes hoy, lo sientes mañana. 🦠 #microbiota #nutricion"
        ),
        "hashtags": "#microbiota #nutricion",
        "engagement": "alto",
        "date": "2025-01-20",
        "category": "ciencia",
    },
    {
        "platform": "newsletter",
        "content": (
            "Esta semana exploramos la relación entre la dieta mediterránea y la "
            "prevención de enfermedades cardiovasculares. Un nuevo estudio publicado "
            "en The Lancet confirma lo que los nutricionistas llevan años diciendo: "
            "el aceite de oliva, las legumbres y el pescado azul son aliados "
            "insustituibles de tu corazón. Lee el artículo completo en nuestra web."
        ),
        "hashtags": "",
        "engagement": "medio",
        "date": "2025-02-01",
        "category": "ciencia y evidencia",
    },
    {
        "platform": "carousel",
        "content": (
            "5 señales de que necesitas más magnesio — muchas personas tienen déficit "
            "sin saberlo. Desliza para descubrirlas. ¿Te identificas con alguna?"
        ),
        "hashtags": "#magnesio #deficiencia #nutricion #suplementos #salud",
        "engagement": "muy alto",
        "date": "2025-02-10",
        "category": "suplementos",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Mock CreateArticle — simulates what Manuel would produce
# In production this object arrives from the Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLE = CreateArticle(
    title="5 alimentos de temporada que no deberían faltar en tu dieta este mes",
    angle="Guía práctica con opciones accesibles disponibles en mercados locales",
    category="nutrición",
    local_relevance_score=0.90,
    article_content=(
        "Con la llegada del cambio de estación, los mercados locales se llenan de "
        "productos frescos que no solo son más económicos, sino también más nutritivos "
        "que cualquier suplemento. La doctora Ana Martínez, nutricionista del centro "
        "de salud municipal, recomienda cinco alimentos que deberían estar en todas "
        "las cocinas este mes.\n\n"
        "Las espinacas encabezan la lista por su alto contenido en hierro y ácido "
        "fólico, especialmente importantes en los meses de transición cuando el "
        "cuerpo necesita más apoyo. Le siguen las naranjas de temporada, que aportan "
        "vitamina C natural muy superior a la de los suplementos embotellados.\n\n"
        "El brócoli, las zanahorias y los garbanzos completan el quinteto. Los tres "
        "son accesibles en cualquier mercado local por menos de dos euros el kilo. "
        "La clave, según la experta, está en la variedad y en cocinarlos de forma "
        "sencilla para preservar sus nutrientes."
    ),
    sources=["asociacion-nutricionistas.org", "mercado-municipal.es"],
    keywords=["nutrición", "alimentos temporada", "dieta", "salud", "mercado local"],
)


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
        print(line)


def _print_pack(pack) -> None:
    """Pretty-prints a SocialMediaPack to the terminal."""

    # ── Twitter ──────────────────────────────────────────────────────────────
    if pack.twitter:
        _print_divider("TWITTER / X")
        print(f"\n{pack.twitter.content}")
        char_count = len(pack.twitter.content)
        status = "✓" if char_count <= 280 else "⚠ OVER LIMIT"
        print(f"\n  [{status}] {char_count}/280 characters")
        if pack.twitter.hashtags:
            print(f"  Hashtags: {' '.join(pack.twitter.hashtags)}")

    # ── Instagram ────────────────────────────────────────────────────────────
    if pack.instagram:
        _print_divider("INSTAGRAM — Caption")
        print(f"\n{pack.instagram.content}")
        if pack.instagram.hashtags:
            print(f"\n{' '.join(pack.instagram.hashtags)}")

        print(f"\n  {'─' * 40}")
        print("  IMAGE PROMPTS")
        print(f"  {'─' * 40}")

        if pack.instagram.image_prompt_midjourney:
            print(f"\n  Midjourney:\n  {pack.instagram.image_prompt_midjourney}")

        if pack.instagram.image_prompt_vertex:
            print(f"\n  Vertex / Imagen API:\n  {pack.instagram.image_prompt_vertex}")

    # ── Carousel ─────────────────────────────────────────────────────────────
    if pack.carousel:
        _print_divider("CAROUSEL")
        print(f"\nIntro: {pack.carousel.content}")
        if pack.carousel.slides:
            print(f"\n  {len(pack.carousel.slides)} slides:\n")
            for slide in pack.carousel.slides:
                print(f"  [{slide.slide_number}] {slide.headline}")
                if slide.body:
                    print(f"       {slide.body}")
        if pack.carousel.hashtags:
            print(f"\n  Hashtags: {' '.join(pack.carousel.hashtags)}")

    # ── Newsletter ───────────────────────────────────────────────────────────
    if pack.newsletter:
        _print_divider("NEWSLETTER SNIPPET")
        print(f"\n{pack.newsletter.content}")
        word_count = len(pack.newsletter.content.split())
        print(f"\n  [{word_count} words]")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _print_divider("SOCIAL MEDIA AGENT — Local Demo")

    # 1. Initialize components
    kb = KnowledgeBase(persist_dir="data/embeddings")
    memory = Memory(max_turns=10)

    # 2. Load post examples into Asti's RAG
    print(f"\nLoading {len(SAMPLE_POSTS)} post examples into ChromaDB...")
    kb.add_post_examples(SAMPLE_POSTS)
    print(f"   → {kb.count()} chunks indexed\n")

    # 3. Create agent
    agent = SocialMediaAgent(
        knowledge_base=kb,
        memory=memory,
    )

    # 4. Display the article being adapted
    print("RECEIVED ARTICLE (mock from Manuel):")
    print(f"   Title:     {SAMPLE_ARTICLE.title}")
    print(f"   Category:  {SAMPLE_ARTICLE.category}")
    print(f"   Relevance: {SAMPLE_ARTICLE.local_relevance_score:.0%}")
    print(f"   Priority:  alta\n")

    # 5. Generate social media pack
    print("Generating SocialMediaPack...\n")
    pack = agent.run(SAMPLE_ARTICLE)

    # 6. Display results per platform
    _print_pack(pack)

    # 7. Confirm file was saved
    output_dir = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
    print(f"\n\n  Pack saved to: {output_dir}/")

    # 8. Conversational mode — journalist can ask Asti directly
    _print_divider("Conversational mode (type 'exit' to quit)")
    print()

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
        print(f"\nAsti: {reply}\n")


if __name__ == "__main__":
    main()
