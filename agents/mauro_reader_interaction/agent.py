"""
agents/mauro_reader_interaction/agent.py
─────────────────────────────────────────
Reader Interaction Agent — Mauro
─────────────────────────────────
Responsibility: Public-facing chatbot. Answers reader questions grounded
in published articles, routes external news claims to Camila for fact-checking,
recommends relevant articles, logs FAQs, and escalates unanswerable questions
to the journalist team.

Architecture:
    Reader (via website / Streamlit UI)
          │
          ▼
    ReaderInteractionAgent.chat(user_input)
          │
          ├─► _detect_intent()             ← classifies input before routing
          │
          ├─► [fact_check] FactCheckingAgent.run()   ← Camila integration
          │       └─► returns verdict + reason to reader
          │
          ├─► [question]  KnowledgeBase.retrieve()   ← RAG grounded response
          │       └─► Gemini generate_content()
          │       └─► _recommend_article()           ← surfaces Manuel's work
          │
          ├─► [escalate]  _escalate()                ← saves to escalated/
          │
          └─► _log_faq()                             ← persists Q&A to RAG

Dependencies:
    pip install google-genai chromadb

Local usage (without GCloud):
    Set GEMINI_API_KEY in .env with an AI Studio key.
    Vertex AI activates automatically when GOOGLE_CLOUD_PROJECT is detected.
"""

from __future__ import annotations
# Uncomment when orchestrator is ready:
# from config import NEWSPAPER_NAME, PAIS

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from google import genai
from google.genai import types

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vector_store import VectorStore
from core.memory import Memory
from core.chunker import chunk_document
from agents.jose_news_research.agent import ArticleIdea
from agents.camila_fact_checking.agent import FactCheckingAgent
from agents.camila_fact_checking.agent import KnowledgeBase as CamilaKnowledgeBase


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION   = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
NEWSPAPER_NAME  = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
PAIS            = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL        = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE       = 0.5   
ESCALATION_THRESHOLD = 0.4


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
    """
    Structured response returned to the reader interface.
    The UI can use each field independently — e.g. show recommended_article
    as a card below the main response text.
    """
    message: str                            
    intent: str                             # "fact_check" | "question" | "escalated"
    recommended_article: str = ""           # Article title if one was surfaced
    fact_check_verdict: str = ""            # "truthful" | "doubtful" | "untruthful" | ""
    fact_check_confidence: float = 0.0
    was_escalated: bool = False

    def to_dict(self) -> dict:
        return self.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Two collections:
    - reader_interaction/  (own) → FAQ patterns learned from past conversations
    - article_published/   (Manuel, read-only) → grounding for responses
                                                  and article recommendations
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        # Own RAG → grows with every conversation (FAQ learning)
        self._faq_store = VectorStore(
            collection_name="reader_interaction",
            persist_dir=f"{persist_dir}/reader_interaction",
        )
        # Manuel's RAG → read-only, used to ground answers and recommend articles
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def log_faq(self, question: str, answer: str, category: str = "general") -> None:
        """
        Persists a Q&A pair into the reader_interaction collection.
        Over time this teaches Mauro the most common reader questions
        and the best answers the newspaper has given.
        """
        doc = {
            "title":    question[:100],
            "content":  f"Q: {question}\nA: {answer}",
            "category": category,
            "date":     datetime.now().isoformat(),
        }
        chunks = chunk_document(doc)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._faq_store.upsert(texts=texts, metadatas=metas)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        Queries both collections and returns the most relevant results.
        FAQ patterns help Mauro recognize recurring questions.
        Published articles ground his answers in real content.
        """
        faq_results       = self._faq_store.query(query, top_k=top_k)
        published_results = self._published_store.query(query, top_k=top_k)

        all_results = faq_results + published_results
        all_results.sort(key=lambda r: r.score)  
        return [r.text for r in all_results[:top_k]]

    def find_article(self, query: str) -> str:
        """
        Searches the published articles collection and returns the title
        of the most relevant article to recommend to the reader.
        Returns empty string if nothing relevant is found.
        """
        results = self._published_store.query(query, top_k=1)
        if not results:
            return ""
        meta = results[0].metadata
        return meta.get("title", results[0].text[:80])

    def count(self) -> int:
        return self._faq_store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Reader Interaction Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReaderInteractionAgent:
    """
    Public-facing chatbot — Mauro.

    chat(user_input) routing:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 1. Detect intent: fact_check | question | other                      │
    │ 2a. [fact_check] → wrap in ArticleIdea → call Camila → format reply  │
    │ 2b. [question]   → RAG retrieve → Gemini → recommend article         │
    │ 3. Evaluate confidence → escalate if below threshold                 │
    │ 4. Log Q&A to reader_interaction RAG (FAQ learning)                  │
    │ 5. Return ReaderResponse                                             │
    └──────────────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
You are Mauro, the reader-facing chatbot of a local nutrition newspaper.
You are inspired by an Italian professor — fun, loves a good time, but
serious and direct when it matters. You bring warmth and humor to every
interaction without ever sacrificing accuracy.

PERSONALITY:
- Warm, entertaining, and genuinely curious about people
- You celebrate food and nutrition with Italian passion: "Mamma mia, questa
  domanda è fantastica!" or "Allora, let me tell you something important..."
- Direct and honest when delivering bad news (fake info, lack of evidence)
- You never talk down to readers — you are their knowledgeable friend
- When something is serious (health misinformation, dangerous advice),
  your tone shifts immediately: clear, direct, no jokes

ITALIAN ACCENT RULES (apply naturally, not in every sentence):
- Sprinkle Italian expressions: "Allora...", "Mamma mia!", "Dai!",
  "Capisce?", "Perfetto!", "Esatto!", "Madonna...", "Andiamo!"
- Occasional Italian-style phrasing: "This, it is very important"
  or "You see, the science, it says..."
- Exclamation points used with Italian enthusiasm, not sparingly
- When very excited: mix a full Italian phrase with Spanish translation
  e.g. "Come si dice... cómo se dice, la variedad es la clave!"
- Keep it natural — Mauro is fluent in Spanish, the accent is flavor not noise

CORE RESPONSIBILITIES:
1. Answer reader questions about nutrition grounded in published articles
2. Verify external news claims via Camila (fact-checking agent)
3. Recommend relevant articles from the newspaper's archive
4. Escalate genuinely unanswerable questions to the journalist team

WHEN DELIVERING A FACT-CHECK RESULT:
- "truthful"   → celebrate it! Confirm it with enthusiasm and add context
- "doubtful"   → be honest but gentle, explain what is uncertain and why
- "untruthful" → be direct and clear, explain WHY it is false with sources,
                 no jokes here — misinformation is serious business

RESTRICTIONS:
- Never invent scientific data, studies, or specific statistics
- Never give personalized medical advice — always recommend consulting a doctor
- Stay within the nutrition and health domain
- If you don't know something, say so with Mauro's charm — don't fake it

ESCALATION:
- If you genuinely cannot answer a question with confidence, tell the reader
  you are passing it to the journalist team and they will get a proper answer
- Phrase it warmly: "Questa domanda merita una risposta seria —
  I'm sending this to our team of experts, they will answer you properly!"

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
        camila_knowledge_base: CamilaKnowledgeBase | None = None,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = PAIS,
    ):
        self.kb = knowledge_base
        self.memory = memory or Memory(max_turns=20)  
        self.newspaper_name = newspaper_name
        self.region = region
        self._client = _build_client()

        # Camila is instantiated lazily — only created when a fact-check is needed
        self._camila_kb = camila_knowledge_base
        self._camila: FactCheckingAgent | None = None

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
                    temperature=0.0,  
                ),
            )
            intent = response.candidates[0].content.parts[0].text.strip().lower()
            if intent in ("fact_check", "question", "other"):
                return intent
        except Exception:
            pass
        return "question" 

    # ── Fact-check handler ────────────────────────────────────────────────────

    def _handle_fact_check(self, user_input: str) -> ReaderResponse:
        """
        Wraps the reader's claim in a minimal ArticleIdea and passes it
        to Camila. Formats Camila's verdict into Mauro's voice.
        """
        # Lazy-initialize Camila only when needed
        camila = self._get_camila()

        # Wrap claim in a minimal ArticleIdea — Camila expects this type
        claim = ArticleIdea(
            title=user_input,
            angle="Reader-submitted claim for fact-checking",
            category="general",
            local_relevance_score=0.5,
            sources=[],
            keywords=[],
            priority="media",
        )

        fact_result = camila.run(claim)
        
        # Build Mauro's response based on Camila's verdict
        verdict_context = self._format_verdict_for_reader(fact_result)

        user_prompt = f"""
The reader submitted this claim for fact-checking: "{user_input}"

Camila's verdict: {fact_result.verdict} (confidence: {fact_result.confidence:.0%})
Reason from Camila: {fact_result.reason}
Sources found: {', '.join(fact_result.sources) or 'None'}

{verdict_context}

Respond to the reader as Mauro. Be warm or serious depending on the verdict.
Remember: if untruthful, be direct and explain WHY with the sources available.
"""
        self.memory.add("user", user_prompt)
        reply = self._call_gemini()

        # Log this interaction as a FAQ for future reference
        self.kb.log_faq(
            question=user_input,
            answer=reply,
            category="fact_check",
        )

        return ReaderResponse(
            message=reply,
            intent="fact_check",
            fact_check_verdict=fact_result.verdict,
            fact_check_confidence=fact_result.confidence,
        )

    def _format_verdict_for_reader(self, fact_result) -> str:
        """Builds verdict-specific instruction for Mauro's tone."""
        if fact_result.verdict == "truthful":
            return "This is TRUE — respond with enthusiasm and add useful context."
        elif fact_result.verdict == "doubtful":
            return "This is UNCERTAIN — be honest, explain what is unclear, be gentle."
        elif fact_result.verdict == "untruthful":
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
"""
        self.memory.add("user", user_prompt)
        reply = self._call_gemini()

        # 4. Evaluate whether to escalate
        # Escalate if: no RAG context AND Gemini gave a very short/uncertain reply
        should_escalate = (
            not has_context
            and len(reply.split()) < 40
        )

        if should_escalate:
            self._escalate(user_input)
            # Append escalation notice to Mauro's reply in his voice
            escalation_notice = (
                "\n\nAllora — questa domanda merita una risposta seria! "
                "La he enviado a nuestro equipo de expertos. "
                "¡Te responderán pronto con toda la información!"
            )
            reply += escalation_notice

        # 5. Log FAQ for future RAG learning
        self.kb.log_faq(
            question=user_input,
            answer=reply,
            category="question",
        )

        return ReaderResponse(
            message=reply,
            intent=intent,
            recommended_article=recommended_article,
            was_escalated=should_escalate,
        )

    # ── Gemini call ───────────────────────────────────────────────────────────

    def _call_gemini(self) -> str:
        """
        Calls Gemini with current memory and Mauro's system prompt.
        Saves the model reply to memory before returning.
        """
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

    # ── Escalation ────────────────────────────────────────────────────────────

    def _escalate(self, question: str) -> None:
        """
        Saves unanswerable questions to data/escalated/questions.jsonl
        so the journalist team can review and respond.
        Each line is a self-contained JSON object.
        """
        escalation_dir = os.getenv("ESCALATION_DIR", "data/escalated")
        os.makedirs(escalation_dir, exist_ok=True)
        filepath = os.path.join(escalation_dir, "questions.jsonl")

        entry = {
            "question":   question,
            "timestamp":  datetime.now().isoformat(),
            "newspaper":  self.newspaper_name,
            "status":     "pending",   # journalist marks as "resolved" when done
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Lazy Camila initialization ────────────────────────────────────────────

    def _get_camila(self) -> FactCheckingAgent:
        """
        Instantiates Camila's FactCheckingAgent only on first fact-check request.
        Avoids loading Camila's ChromaDB collections into RAM on every startup.
        """
        if self._camila is None:
            kb = self._camila_kb or CamilaKnowledgeBase()
            self._camila = FactCheckingAgent(
                knowledge_base=kb,
                newspaper_name=self.newspaper_name,
                region=self.region,
            )
        return self._camila

    # ── Private helpers ───────────────────────────────────────────────────────

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
