"""
agents/camila_fact_checking/agent.py
─────────────────────────────────────
Fact Checking Agent
────────────────────
Responsabilidad: ...  # TODO: describe en tus palabras

Arquitectura:
    # TODO: dibuja el flujo como en José y Manuel
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from agents.jose_news_research.agent import ArticleIdea

# ─────────────────────────────────────────────────────────────────────────────
# Config — igual que José y Manuel, copia el patrón
# ─────────────────────────────────────────────────────────────────────────────

# TODO: copiar variables de entorno y _build_client()


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de datos de salida
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactCheckResult:
    idea: ArticleIdea
    verdict: str             # TODO: ¿qué valores acepta?
    reason: str              # explicación del veredicto
    confidence: float        # TODO: ¿qué rango?
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # TODO


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    def __init__(self, persist_dir: str = "data/embeddings"):
        # TODO: ¿cuántos stores necesita Camila?
        # Pista: una colección propia + ¿cuál más puede leer?
        pass

    def add_fake_news_example(self, doc: dict) -> None:
        # TODO: ¿en qué store escribe?
        pass

    def add_fake_news_examples(self, docs: list[dict]) -> None:
        # TODO: igual que add_style_documents en Manuel
        pass

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        # TODO: consulta ambos stores y combina
        # Pista: mismo patrón que Manuel
        pass

    def count(self) -> int:
        # TODO
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fact Checking Agent
# ─────────────────────────────────────────────────────────────────────────────

class FactCheckingAgent:

    SYSTEM_PROMPT = """
    # TODO: define la personalidad de Camila
    # TODO: restricciones — ¿qué nunca debe hacer?
    # TODO: formato de salida JSON con verdict, reason, confidence, sources
    # Pista: sigue el mismo patrón de REGLAS que José
    """.strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory: Memory | None = None,
        # TODO: ¿necesita newspaper_name y region como José y Manuel?
    ):
        # TODO
        pass

    def run(self, idea: ArticleIdea) -> FactCheckResult:
        """
        Verifica una ArticleIdea y devuelve un veredicto.
        """
        # 1. TODO: extraer el claim principal de la idea
        #    Pista: ¿qué campos de ArticleIdea contienen la información a verificar?

        # 2. TODO: RAG — buscar patrones de fake news similares

        # 3. TODO: web_search del claim
        #    Pista: ya tienes web_search en search_tools.py

        # 4. TODO: construir prompt

        # 5. TODO: llamar a Gemini — mismo patrón que José y Manuel

        # 6. TODO: parsear y devolver FactCheckResult
        pass

    def run_batch(self, ideas: list[ArticleIdea]) -> list[FactCheckResult]:
        # TODO: verifica una lista completa de ideas
        # Pista: es un loop sobre run()
        pass

    def _build_prompt(self, idea: ArticleIdea, context_snippets: list[str], web_results: list[dict]) -> str:
        # TODO: construir el prompt con:
        # - el claim extraído de la idea
        # - los snippets del RAG
        # - los resultados web
        pass

    def _parse_result(self, raw_text: str, idea: ArticleIdea) -> FactCheckResult:
        # TODO: mismo patrón que _parse_ideas en José y _parse_article en Manuel
        # ¿Qué devuelves en el fallback?
        pass
