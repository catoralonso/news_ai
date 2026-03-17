"""
app.py
───────
Streamlit UI for newspaper_ai.
Runs from the project root:

    streamlit run app.py

Requires:
    pip install streamlit
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Patch GOOGLE_CLOUD_PROJECT so _build_client() uses GEMINI_API_KEY
if not os.getenv("GOOGLE_CLOUD_PROJECT", "").strip():
    os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
else:
    # If running in GCloud Shell, force AI Studio path
    if "GEMINI_API_KEY" in os.environ:
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nutrición AI — Panel Editorial",
    page_icon="🗞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    /* Header */
    .newspaper-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 3px double #1a1a1a;
        margin-bottom: 1.5rem;
    }
    .newspaper-header h1 {
        font-size: 2.8rem;
        letter-spacing: -1px;
        margin: 0;
        color: #1a1a1a;
    }
    .newspaper-header p {
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #666;
        margin: 0.25rem 0 0 0;
    }

    /* Idea cards */
    .idea-card {
        border: 1px solid #e0e0e0;
        border-left: 4px solid #1a1a1a;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-radius: 0 4px 4px 0;
        background: #fafafa;
    }
    .idea-card.truthful  { border-left-color: #2d6a4f; }
    .idea-card.doubtful  { border-left-color: #e9a800; }
    .idea-card.untruthful{ border-left-color: #c1121f; }

    .verdict-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .badge-truthful  { background: #d8f3dc; color: #2d6a4f; }
    .badge-doubtful  { background: #fff3cd; color: #856404; }
    .badge-untruthful{ background: #fde8e9; color: #c1121f; }

    /* Article display */
    .article-box {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 2rem;
        line-height: 1.8;
    }
    .article-meta {
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
    }

    /* Social cards */
    .social-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .platform-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
    }

    /* Chat bubbles */
    .chat-reader {
        background: #e9ecef;
        border-radius: 16px 16px 4px 16px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-mauro {
        background: #1a1a1a;
        color: #fff;
        border-radius: 16px 16px 16px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 75%;
        font-size: 0.95rem;
    }
    .chat-meta {
        font-size: 0.72rem;
        color: #aaa;
        margin-top: 0.25rem;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: #333;
        color: white;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — cached so it only builds once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing agents...")
def get_orchestrator():
    from agents.orchestrator.agent import Orchestrator
    orch = Orchestrator()
    orch.build_agents()
    return orch


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="newspaper-header">
    <h1>🗞 Nutrición AI</h1>
    <p>Panel Editorial · Sistema Multi-Agente</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_pipeline, tab_social, tab_reader = st.tabs([
    "📰 Pipeline Editorial",
    "📱 Redes Sociales",
    "💬 Chat con Lectores",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline Editorial
# ═════════════════════════════════════════════════════════════════════════════

with tab_pipeline:

    # ── Step 1: Research ─────────────────────────────────────────────────────
    st.markdown("### 1 · Investigar tendencias")
    st.caption("José busca tendencias y Camila verifica cada idea en paralelo.")

    col_query, col_btn = st.columns([4, 1])
    with col_query:
        query = st.text_input(
            "Query de investigación",
            value="Tendencias de nutrición y salud digestiva en España",
            label_visibility="collapsed",
        )
    with col_btn:
        research_btn = st.button("🔍 Investigar", use_container_width=True)

    if research_btn and query:
        orch = get_orchestrator()
        with st.spinner("José investigando · Camila verificando..."):
            try:
                report = orch._jose.run(query)
                fact_results = orch._camila.run_batch(report.article_ideas)

                # Attach scores to ideas
                for idea, result in zip(report.article_ideas, fact_results):
                    idea.confidence_score = result.confidence
                    idea.verdict = result.verdict

                st.session_state["report"]       = report
                st.session_state["fact_results"] = fact_results
                st.session_state["article"]      = None
                st.session_state["social_pack"]  = None
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Display ideas ─────────────────────────────────────────────────────────
    if "report" in st.session_state and st.session_state["report"]:
        report = st.session_state["report"]

        st.markdown(f"**{len(report.article_ideas)} ideas encontradas**")
        if getattr(report, "summary", None):
            st.caption(report.summary)

        verdict_icon = {"truthful": "✓", "doubtful": "?", "untruthful": "✗"}
        selected_indices = []

        for i, idea in enumerate(report.article_ideas):
            verdict = idea.verdict or "doubtful"
            icon    = verdict_icon.get(verdict, "·")
            conf    = getattr(idea, "confidence_score", None)
            conf_str = f"{conf:.0%}" if conf is not None else "—"

            with st.container():
                st.markdown(f"""
<div class="idea-card {verdict}">
    <span class="verdict-badge badge-{verdict}">{icon} {verdict} · {conf_str}</span>
    &nbsp;&nbsp;
    <span style="font-size:0.75rem;color:#888;letter-spacing:1px;text-transform:uppercase">
        {idea.category} · {idea.priority} · relevancia {idea.local_relevance_score:.0%}
    </span>
    <br><strong style="font-size:1rem">{idea.title}</strong>
    <br><span style="font-size:0.88rem;color:#555">{idea.angle}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Step 2: Select & generate ─────────────────────────────────────────
        st.markdown("### 2 · Seleccionar y redactar")
        st.caption("El periodista elige qué ideas proceder. Manuel redacta el artículo.")

        idea_options = {
            f"[{i+1}] {idea.title}": i
            for i, idea in enumerate(report.article_ideas)
        }
        selected_label = st.selectbox(
            "Selecciona una idea",
            options=list(idea_options.keys()),
            label_visibility="collapsed",
        )
        generate_btn = st.button("✍️ Redactar artículo")

        if generate_btn and selected_label:
            orch = get_orchestrator()
            selected_idea = report.article_ideas[idea_options[selected_label]]

            with st.spinner("Manuel redactando..."):
                try:
                    article = orch._manuel.run(selected_idea)
                    st.session_state["article"] = article
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Display article ───────────────────────────────────────────────────────
    if st.session_state.get("article"):
        article = st.session_state["article"]
        st.markdown("---")
        st.markdown("### 3 · Artículo generado")

        st.markdown(f"""
<div class="article-box">
    <div class="article-meta">{article.category} · relevancia {article.local_relevance_score:.0%}</div>
    <h2 style="margin:0 0 0.5rem 0">{article.title}</h2>
    <p style="color:#666;font-style:italic;margin-bottom:1.5rem">{article.angle}</p>
    <div>{article.article_content.replace(chr(10), '<br>')}</div>
    <hr style="margin:1.5rem 0 0.75rem 0">
    <div style="font-size:0.8rem;color:#888">
        Fuentes: {', '.join(article.sources) if article.sources else '—'} &nbsp;|&nbsp;
        Keywords: {', '.join(article.keywords[:5]) if article.keywords else '—'}
    </div>
</div>
""", unsafe_allow_html=True)

        publish_btn = st.button("📤 Publicar en redes")

        if publish_btn:
            orch = get_orchestrator()
            with st.spinner("Asti creando contenido para redes · Mauro preparándose..."):
                try:
                    social_pack = orch._asti.run(article)
                    orch._mauro.setup(article, social_pack)
                    st.session_state["social_pack"] = social_pack
                    st.success("¡Publicado! Ve a la pestaña 📱 Redes Sociales.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Redes Sociales
# ═════════════════════════════════════════════════════════════════════════════

with tab_social:
    st.markdown("### Contenido para redes sociales")
    st.caption("Generado por Asti a partir del artículo publicado.")

    pack = st.session_state.get("social_pack")

    if not pack:
        st.info("Completa el Pipeline Editorial y pulsa **Publicar en redes** para ver el contenido aquí.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Twitter
            if pack.twitter:
                st.markdown('<div class="social-card">', unsafe_allow_html=True)
                st.markdown('<div class="platform-label">🐦 Twitter / X</div>', unsafe_allow_html=True)
                st.write(pack.twitter.content)
                char_count = len(pack.twitter.content)
                status = "✓" if char_count <= 280 else "⚠ Over limit"
                st.caption(f"{status} · {char_count}/280 · {' '.join(pack.twitter.hashtags)}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Newsletter
            if pack.newsletter:
                st.markdown('<div class="social-card">', unsafe_allow_html=True)
                st.markdown('<div class="platform-label">📧 Newsletter</div>', unsafe_allow_html=True)
                st.write(pack.newsletter.content)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Instagram
            if pack.instagram:
                st.markdown('<div class="social-card">', unsafe_allow_html=True)
                st.markdown('<div class="platform-label">📸 Instagram</div>', unsafe_allow_html=True)
                st.write(pack.instagram.content)
                st.caption(' '.join(pack.instagram.hashtags))
                if pack.instagram.image_prompt_vertex:
                    with st.expander("🖼 Image prompt (Vertex AI Imagen)"):
                        st.code(pack.instagram.image_prompt_vertex, language=None)
                st.markdown('</div>', unsafe_allow_html=True)

            # Carousel
            if pack.carousel and pack.carousel.slides:
                st.markdown('<div class="social-card">', unsafe_allow_html=True)
                st.markdown('<div class="platform-label">🎠 Carousel</div>', unsafe_allow_html=True)
                st.write(pack.carousel.content)
                for slide in pack.carousel.slides:
                    st.markdown(f"**[{slide.slide_number}]** {slide.headline}")
                    if slide.body:
                        st.caption(slide.body)
                st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Chat con Lectores (Mauro)
# ═════════════════════════════════════════════════════════════════════════════

with tab_reader:
    st.markdown("### Chat con lectores")
    st.caption("Mauro responde preguntas de nutrición y verifica noticias vía Camila.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "reader":
            st.markdown(
                f'<div class="chat-reader">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            verdict_line = ""
            if msg.get("verdict"):
                verdict_line = (
                    f'<div class="chat-meta">'
                    f'Camila: {msg["verdict"]} · {msg.get("confidence", 0):.0%} confianza'
                    f'</div>'
                )
            st.markdown(
                f'<div class="chat-mauro">{msg["content"]}{verdict_line}</div>',
                unsafe_allow_html=True,
            )

    # Input
    with st.form("reader_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            reader_msg = st.text_input(
                "Mensaje",
                placeholder="Pregunta algo sobre nutrición o pega una noticia a verificar...",
                label_visibility="collapsed",
            )
        with col_send:
            send_btn = st.form_submit_button("Enviar", use_container_width=True)

    if send_btn and reader_msg:
        orch = get_orchestrator()
        st.session_state["chat_history"].append({
            "role": "reader",
            "content": reader_msg,
        })

        with st.spinner("Mauro respondiendo..."):
            try:
                response = orch.chat_reader(reader_msg)
                st.session_state["chat_history"].append({
                    "role":       "mauro",
                    "content":    response.message,
                    "verdict":    response.fact_check_verdict or "",
                    "confidence": response.fact_check_confidence,
                })
            except Exception as e:
                st.session_state["chat_history"].append({
                    "role":    "mauro",
                    "content": f"Error: {e}",
                })

        st.rerun()
