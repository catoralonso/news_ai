"""
agents/mauro_reader_interaction/agent.py
─────────────────────────────────────────
Reader Interaction Agent — Mauro
─────────────────────────────────
Responsibility: Handles reader conversations using all pipeline outputs.
Routes fact-check requests to Camila. Escalates unanswerable questions
to the journalist team.

Architecture:
     Reader (via UI / chatbot)
          │
          ▼ free text
    ReaderInteractionAgent.chat(user_input)
          │
          ├─► _detect_intent()              ← lightweight Gemini call (no RAG, no memory)
          │         │
          │    "fact_check" ──────────────► Camila.verify_url(input_text)
          │    "question"   ──────────────► KnowledgeBase.retrieve() + Gemini
          │    "other"      ──────────────► Gemini (friendly deflection)
          │
          └─► ReaderResponse
                  ├── message
                  ├── intent
                  ├── fact_check_verdict / confidence  (fact_check path only)
                  ├── recommended_article              (question path only)
                  └── was_escalated

Dependencies:
    pip install google-genai chromadb

Local usage (without GCloud):
    Set GEMINI_API_KEY in .env with an AI Studio key.
    Vertex AI activates automatically when GOOGLE_CLOUD_PROJECT is detected.
"""

from __future__ import annotations
from config import (
    NEWSPAPER_NAME, REGION as PAIS,
    CHAT_MODEL, GEMINI_API_KEY,
    VERTEX_PROJECT, VERTEX_REGION,
)

import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from agents.camila_fact_checking.agent import (
    FactCheckingAgent,
    KnowledgeBase as CamilaKnowledgeBase,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MAX_OUTPUT_TOKENS = 2048
TEMPERATURE       = 0.7  


def _build_client() -> genai.Client:
    """
    Authentication priority:
    1. Vertex AI (if GOOGLE_CLOUD_PROJECT is set) → production
    2. GEMINI_API_KEY                              → local development
    """
    if VERTEX_PROJECT:
        return genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_REGION,
        )
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    raise EnvironmentError(
        "Set GEMINI_API_KEY (local) or GOOGLE_CLOUD_PROJECT (Vertex AI)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReaderResponse:
    """Everything Mauro returns after handling a reader message."""
    message: str
    intent: str = ""                        # "fact_check" | "question" | "other"
    fact_check_verdict: str = ""            # populated on fact_check path
    fact_check_confidence: float = 0.0      # populated on fact_check path
    recommended_article: str = ""           # populated on question path
    was_escalated: bool = False

    def to_dict(self) -> dict:
        return self.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Two collections:
    - reader_interaction/ (own) → FAQ log, past Q&A, reader patterns
    - article_published/  (Manuel, read-only) → articles to recommend
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        # Own RAG → FAQ and conversation log
        self._reader_store = VectorStore(
            collection_name="reader_interaction",
            persist_dir=f"{persist_dir}/reader_interaction",
        )
        # Manuel's RAG → read-only, source for article recommendations
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def log_faq(self, question: str, answer: str, category: str = "") -> None:
        """Indexes a Q&A pair for future retrieval."""
        doc = {"content": f"Q: {question}\nA: {answer}", "category": category}
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._reader_store.upsert(texts=texts, metadatas=metas)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """Queries both collections and returns the most relevant results."""
        reader_results    = self._reader_store.query(query, top_k=top_k)
        published_results = self._published_store.query(query, top_k=2)

        all_results = reader_results + published_results
        all_results.sort(key=lambda r: r.score)  # lower score = more similar
        return [r.text for r in all_results[:top_k]]

    def find_article(self, query: str) -> str:
        """
        Returns the title of the most relevant published article for the query.
        Returns empty string if nothing relevant is found.
        """
        results = self._published_store.query(query, top_k=1)
        if results:
            # Titles are stored as metadata or at the start of the text chunk
            return results[0].text[:80].split("\n")[0]
        return ""

    def count(self) -> int:
        return self._reader_store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Reader Interaction Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReaderInteractionAgent:
    """
    Conversational agent for newspaper readers — Mauro.

    chat(user_input) flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Detect intent (lightweight Gemini call, no memory)           │
    │ 2a. fact_check → verify_url() via injected Camila instance      │
    │ 2b. question   → RAG + Gemini, recommend article, maybe escalate│
    │ 2c. other      → friendly Gemini response                       │
    │ 3. Log Q&A to reader_interaction RAG                            │
    │ 4. Return ReaderResponse                                        │
    └─────────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
You are Mauro, the reader interaction assistant of a local nutrition newspaper.
Your role: answer readers' nutrition and health questions using the newspaper's
published articles and verified information.

PERSONALITY:
- Warm, approachable, and enthusiastic about nutrition
- You occasionally use Italian expressions for flavor (ciao, allora, bravissimo)
  but always communicate primarily in Spanish
- You are honest when you don't know something — you never invent answers
- You reference the newspaper's articles naturally in conversation

RESTRICTIONS:
- Only answer questions related to nutrition, health, and the newspaper's content
- Never give medical diagnoses or prescribe treatments
- If you don't have enough information, say so and escalate to the team

LANGUAGE: Always respond in Spanish. Italian expressions are accent only.
""".strip()

    # Intent detection prompt — separate, lighter call to classify input
    INTENT_PROMPT = """
Classify the following reader input into exactly one of these intents:
- "fact_check": the reader is submitting an external news claim or asking
  if something they read/heard is true (e.g. "Is it true that X causes Y?",
  "I read that Z, is this correct?")
- "question": the reader is asking a genuine nutrition/health question
  (e.g. "What foods help with X?", "How much protein should I eat?")
- "other": greetings, off-topic, unclear input

Respond with ONLY the intent string, nothing else.
Input: {user_input}
""".strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        camila: FactCheckingAgent,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = PAIS,
    ):
        self.kb = knowledge_base
        self.memory = memory or Memory(max_turns=20)  # longer window — conversational
        self.newspaper_name = newspaper_name
        self.region = region
        self._client = _build_client()

        # Camila is injected by the Orchestrator — not created here
        self._camila = camila

        # Context injected by Orchestrator after Manuel and Asti finish
        self._article = None
        self._social_pack = None

    # ── Setup (called by Orchestrator after pipeline completes) ──────────────

    def setup(self, article, social_pack) -> None:
        """
        Called by the Orchestrator after Manuel and Asti finish.
        Injects the latest article and social pack so Mauro can
        answer questions about them and recommend them.

        Args:
            article:     CreateArticle produced by Manuel.
            social_pack: SocialMediaPack produced by Asti.
        """
        self._article = article
        self._social_pack = social_pack
        # Index the new article in the reader RAG so retrieve() finds it
        self.kb.log_faq(
            question=article.title,
            answer=article.article_content,
            category=article.category,
        )

    # ── Main method ───────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> ReaderResponse:
        """
        Main entry point. Routes the reader's input to the correct handler
        and always returns a ReaderResponse.

        Args:
            user_input: Raw text from the reader.

        Returns:
            ReaderResponse with Mauro's message and metadata.
        """
        # 1. Detect intent before doing anything expensive
        intent = self._detect_intent(user_input)

        # 2. Route to the appropriate handler
        if intent == "fact_check":
            return self._handle_fact_check(user_input)
        else:
            # Both "question" and "other" go through the standard RAG flow
            return self._handle_question(user_input, intent)

    # ── Intent routing ────────────────────────────────────────────────────────

    def _detect_intent(self, user_input: str) -> str:
        """
        Lightweight Gemini call to classify reader input.
        Uses a separate prompt — no memory, no RAG, no system prompt overhead.
        Returns "fact_check" | "question" | "other".
        """
        try:
            response = self._client.models.generate_content(
                model=CHAT_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=self.INTENT_PROMPT.format(user_input=user_input)
                        )],
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.0,  # deterministic classification
                ),
            )
            intent = response.candidates[0].content.parts[0].text.strip().lower()
            if intent in ("fact_check", "question", "other"):
                return intent
        except Exception:
            pass
        return "question"  # safe default

    # ── Fact-check handler ────────────────────────────────────────────────────

    def _handle_fact_check(self, user_input: str) -> ReaderResponse:
        """
        Passes the reader's claim directly to Camila via verify_url().
        Formats Camila's VerificationResult into Mauro's voice.
        """
        # Call Camila with the reader's raw text — no ArticleIdea needed
        verification = self._camila.verify_url(user_input)

        # Build Mauro's response based on Camila's verdict
        verdict_context = self._format_verdict_for_reader(
            verification.verdict, verification.confidence
        )

        user_prompt = f"""
The reader submitted this claim for fact-checking: "{user_input}"

Camila's verdict: {verification.verdict} (confidence: {verification.confidence:.0%})
Reason from Camila: {verification.reason}
Sources found: {', '.join(verification.sources) or 'None'}

{verdict_context}

Respond to the reader as Mauro. Be warm or serious depending on the verdict.
If untruthful, be direct and explain WHY with the sources available.
""".strip()

        self.memory.add("user", user_prompt)
        reply = self._call_gemini()

        # Log this interaction for future RAG retrieval
        self.kb.log_faq(
            question=user_input,
            answer=reply,
            category="fact_check",
        )

        return ReaderResponse(
            message=reply,
            intent="fact_check",
            fact_check_verdict=verification.verdict,
            fact_check_confidence=verification.confidence,
        )

    def _format_verdict_for_reader(self, verdict: str, confidence: float) -> str:
        """Builds verdict-specific tone instruction for Mauro's response."""
        if verdict == "truthful":
            return "This is TRUE — respond with enthusiasm and add useful context."
        elif verdict == "doubtful":
            return "This is UNCERTAIN — be honest, explain what is unclear, be gentle."
        elif verdict == "untruthful":
            return (
                "This is FALSE — be direct, serious, explain clearly why it is "
                "misinformation. No jokes. Use the sources to back up the explanation."
            )
        else:
            return "Not enough information to verify — be transparent about this."

    # ── Question handler ──────────────────────────────────────────────────────

    def _handle_question(self, user_input: str, intent: str) -> ReaderResponse:
        """
        Handles nutrition questions and general inputs.
        Retrieves RAG context, calls Gemini, recommends an article,
        and escalates if confidence is insufficient.
        """
        # 1. RAG — retrieve relevant FAQs and published articles
        context_snippets = self.kb.retrieve(user_input, top_k=4)
        has_context = bool(context_snippets)

        # 2. Find a relevant article to recommend
        recommended_article = self.kb.find_article(user_input)

        # 3. Build enriched prompt
        ctx_block = (
            "\n".join(f"- {s[:300]}" for s in context_snippets)
            or "No relevant articles found in the archive."
        )
        rec_block = (
            f'Relevant article to recommend: "{recommended_article}"'
            if recommended_article
            else "No specific article to recommend."
        )

        user_prompt = f"""
Reader question: "{user_input}"

CONTEXT FROM NEWSPAPER ARCHIVE (RAG):
{ctx_block}

{rec_block}

Answer the reader's question as Mauro. Ground your answer in the context above.
If you recommend the article, mention it naturally in your response.
If the context is insufficient to answer confidently, say so honestly.
""".strip()

        self.memory.add("user", user_prompt)
        reply = self._call_gemini()

        # 4. Escalate if: no RAG context AND Gemini gave a very short/uncertain reply
        should_escalate = not has_context and len(reply.split()) < 40

        if should_escalate:
            self._escalate(user_input)
            reply += (
                "\n\nAllora — questa domanda merita una risposta seria! "
                "La he enviado a nuestro equipo de expertos. "
                "¡Te responderán pronto con toda la información!"
            )

        # 5. Log Q&A for future RAG retrieval
        self.kb.log_faq(
            question=user_input,
            answer=reply,
            category=intent,
        )

        return ReaderResponse(
            message=reply,
            intent=intent,
            recommended_article=recommended_article,
            was_escalated=should_escalate,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_gemini(self) -> str:
        """Calls Gemini with the current memory and returns the response text."""
        response = self._client.models.generate_content(
            model=CHAT_MODEL,
            contents=self._messages_to_contents(self.memory.as_messages()),
            config=types.GenerateContentConfig(
                system_instruction=self._personalized_system_prompt(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        reply = response.candidates[0].content.parts[0].text
        self.memory.add("model", reply)
        return reply

    def _escalate(self, user_input: str) -> None:
        """
        Logs an unanswerable question for the journalist team.
        TODO: replace with email/Slack notification in production.
        """
        escalation_log = os.getenv("ESCALATION_LOG", "data/escalations.log")
        os.makedirs(os.path.dirname(escalation_log), exist_ok=True)
        with open(escalation_log, "a", encoding="utf-8") as f:
            f.write(f"{user_input}\n")

    def _personalized_system_prompt(self) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Newspaper: {self.newspaper_name}. Region: {self.region}."
        )

    def _messages_to_contents(self, messages: list[dict]) -> list[types.Content]:
        result = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            result.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )
        return result
