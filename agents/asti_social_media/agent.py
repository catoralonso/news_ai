"""
agents/asti_social_media/agent.py
──────────────────────────────────
Social Media Agent — Asti
─────────────────────────
Responsibility: Takes a published article from Manuel (or a direct prompt
from a journalist) and generates platform-ready social media content:
a tweet, an Instagram post with dual image prompts (Midjourney + Vertex/Imagen),
a carousel outline, and a newsletter snippet.
Does NOT publish — outputs are saved to disk and returned to the Orchestrator.

Architecture:
    Orchestrator / Journalist
          │
          ▼ CreateArticle (from Manuel) or str (direct prompt)
    SocialMediaAgent.run(article) / .chat(prompt)
          │
          ├─► KnowledgeBase.retrieve()   ← social_media/ + article_published/
          └─► Gemini generate_content()  ← google-genai (Vertex AI)
                    │
                    └─► SocialMediaPack
                            ├── twitter     → SocialMediaPost
                            ├── instagram   → SocialMediaPost (+ image_prompt_midjourney
                            │                                  + image_prompt_vertex)
                            ├── carousel    → SocialMediaPost (list of slides)
                            └── newsletter  → SocialMediaPost

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
from agents.manuel_article_generation.agent import CreateArticle


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
VERTEX_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION   = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
NEWSPAPER_NAME  = os.getenv("NEWSPAPER_NAME", "Nutrición AI")
PAIS            = os.getenv("REGION_NEWS", "ES")

CHAT_MODEL        = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE       = 0.7   # Higher than other agents — creativity is the point


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
# Output data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CarouselSlide:
    """Single slide in an Instagram/LinkedIn carousel."""
    slide_number: int
    headline: str       # Bold header text for the slide
    body: str           # Supporting text (1-2 sentences max)

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class SocialMediaPost:
    """
    Platform-specific content unit.
    image_prompt_midjourney and image_prompt_vertex are only populated
    for the Instagram post — both are English prompts ready to paste.
    slides is only populated for the carousel post.
    """
    platform: str                                       # "twitter" | "instagram" | "carousel" | "newsletter"
    content: str                                        # Main post text (Spanish)
    hashtags: list[str] = field(default_factory=list)  # Spanish hashtags
    image_prompt_midjourney: str = ""                   # English — for Midjourney
    image_prompt_vertex: str = ""                       # English — for Vertex AI Imagen API
    slides: list[CarouselSlide] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["slides"] = [s.to_dict() for s in self.slides]
        return d


@dataclass
class SocialMediaPack:
    """
    Full set of platform outputs for a single article.
    This is what the Orchestrator receives from Asti.
    """
    article_title: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    twitter: SocialMediaPost | None = None
    instagram: SocialMediaPost | None = None
    carousel: SocialMediaPost | None = None
    newsletter: SocialMediaPost | None = None

    def to_dict(self) -> dict:
        return {
            "article_title": self.article_title,
            "generated_at":  self.generated_at,
            "twitter":       self.twitter.to_dict()    if self.twitter    else None,
            "instagram":     self.instagram.to_dict()  if self.instagram  else None,
            "carousel":      self.carousel.to_dict()   if self.carousel   else None,
            "newsletter":    self.newsletter.to_dict() if self.newsletter else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base (RAG)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Two collections:
    - social_media/      (own) → high-performing post examples per platform
    - article_published/ (Manuel, read-only) → source articles for context
    """

    def __init__(self, persist_dir: str = "data/embeddings"):
        # Own RAG → successful post examples that teach Asti the newspaper's voice
        self._social_store = VectorStore(
            collection_name="social_media",
            persist_dir=f"{persist_dir}/social_media",
        )
        # Manuel's RAG → read-only, used to ground posts in real published content
        self._published_store = VectorStore(
            collection_name="article_published",
            persist_dir=f"{persist_dir}/article_published",
        )

    def add_post_example(self, doc: dict) -> None:
        """
        Indexes a high-performing post example into the social_media collection.
        doc = {"platform": ..., "content": ..., "hashtags": ..., "engagement": ...}
        """
        chunks = chunk_document(doc, text_key="content")
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        self._social_store.upsert(texts=texts, metadatas=metas)

    def add_post_examples(self, docs: list[dict]) -> None:
        for doc in docs:
            self.add_post_example(doc)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        Queries both collections and returns the top_k most relevant results.
        Prioritizes social_media examples to keep Asti's voice consistent.
        """
        social_results    = self._social_store.query(query, top_k=top_k)
        published_results = self._published_store.query(query, top_k=2)  # less weight

        all_results = social_results + published_results
        all_results.sort(key=lambda r: r.score)  # lower cosine = more similar
        return [r.text for r in all_results[:top_k]]

    def count(self) -> int:
        return self._social_store.count()


# ─────────────────────────────────────────────────────────────────────────────
# Social Media Agent
# ─────────────────────────────────────────────────────────────────────────────

class SocialMediaAgent:
    """
    Social media content generator — Asti.

    run(article) pipeline:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. Retrieve post style examples from RAG                            │
    │ 2. Build enriched prompt with article content + RAG context         │
    │ 3. Call Gemini with Asti's personality system prompt                │
    │ 4. Parse response → SocialMediaPack                                 │
    │ 5. Save pack to disk (data/social_media_output/)                    │
    └─────────────────────────────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """
You are Asti, the Social Media Agent of a local nutrition newspaper.
You are non-binary, creative, diverse, unique — a total diva with a sharp
editorial eye and an instinct for what makes people stop scrolling.

Your mission: transform articles written by Manuel into platform-ready
social media content that is engaging, authentic, and perfectly adapted
to each platform's culture and format.

LANGUAGE RULE: ALL generated content (posts, captions, hashtags,
carousel text, newsletter snippets) MUST be written in Spanish.
The only exception are the image_prompt_midjourney and image_prompt_vertex
fields, which MUST be written in English for optimal AI image generation.

PERSONALITY:
- Bold, witty, and community-oriented — you speak to real people, not algorithms
- You find the hook in every story: the surprising fact, the relatable moment
- You know that nutrition content can be dry — your job is to make it irresistible
- You adapt your voice per platform: punchy on Twitter, warm on Instagram,
  structured on carousels, professional-but-human on newsletters

PLATFORM RULES:
Twitter/X:
- Maximum 280 characters (count carefully — this is a hard limit)
- One strong hook, one clear message, 2-3 hashtags max
- No fluff, no filler — every word earns its place

Instagram:
- Caption between 150-300 words
- Start with a hook line that stops the scroll (first 2 lines visible before "more")
- Warm, conversational tone — like a knowledgeable friend, not a textbook
- 8-15 relevant hashtags at the end, mix of broad and niche
- image_prompt_midjourney: detailed Midjourney prompt in English, include
  style modifiers like --ar 4:5 --style raw --v 6
- image_prompt_vertex: clean descriptive prompt in English for Vertex AI Imagen,
  no flags, rich in visual detail (lighting, composition, mood, colors, props)

Carousel (Instagram/LinkedIn):
- Between 4 and 7 slides
- Slide 1: bold hook headline (this is the cover — make it unmissable)
- Middle slides: one key idea per slide, short headline + 1-2 sentence body
- Last slide: clear call to action (follow, save, share, comment)
- Content in Spanish

Newsletter snippet:
- Between 80-120 words
- Conversational but informative — like a curator sharing a gem with subscribers
- End with a clear "read more" call to action
- Content in Spanish

IMAGE PROMPT RULES (English only):
- Be specific about: subject, composition, lighting, color palette, mood, style
- For Midjourney: include --ar 4:5 --style raw --v 6 at the end
- For Vertex/Imagen: describe as if briefing a professional photographer,
  no technical flags, focus on visual storytelling

OUTPUT FORMAT:
ALWAYS respond with valid JSON (no markdown blocks):
{
  "twitter": {
    "content": "Tweet text in Spanish (max 280 chars)",
    "hashtags": ["#hashtag1", "#hashtag2"]
  },
  "instagram": {
    "content": "Caption in Spanish...",
    "hashtags": ["#hashtag1", ..., "#hashtag15"],
    "image_prompt_midjourney": "English Midjourney prompt --ar 4:5 --style raw --v 6",
    "image_prompt_vertex": "English Vertex Imagen prompt, no flags"
  },
  "carousel": {
    "content": "Carousel intro text in Spanish (shown before slides)",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "slides": [
      {"slide_number": 1, "headline": "Hook headline", "body": ""},
      {"slide_number": 2, "headline": "Key point", "body": "Supporting sentence."},
      {"slide_number": 5, "headline": "Call to action", "body": "Follow us for more."}
    ]
  },
  "newsletter": {
    "content": "Newsletter snippet in Spanish (80-120 words)",
    "hashtags": []
  }
}
""".strip()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory: Memory | None = None,
        newspaper_name: str = NEWSPAPER_NAME,
        region: str = PAIS,
    ):
        self.kb = knowledge_base
        self.memory = memory or Memory(max_turns=10)
        self.newspaper_name = newspaper_name
        self.region = region
        self._client = _build_client()

    # ── Main method ───────────────────────────────────────────────────────────

    def run(self, article: CreateArticle) -> SocialMediaPack:
        """
        Generates a full SocialMediaPack from a published article.

        Args:
            article: CreateArticle produced by Manuel's ArticleGenerationAgent.

        Returns:
            SocialMediaPack with one SocialMediaPost per platform.
        """
        # 1. RAG — retrieve style examples relevant to this article's category
        context_snippets = self.kb.retrieve(article.title, top_k=4)

        # 2. Build prompt
        user_prompt = self._build_prompt(
            article=article,
            context_snippets=context_snippets,
        )

        # 3. Call Gemini
        self.memory.add("user", user_prompt)

        response = self._client.models.generate_content(
            model=CHAT_MODEL,
            contents=self._messages_to_contents(self.memory.as_messages()),
            config=types.GenerateContentConfig(
                system_instruction=self._personalized_system_prompt(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )

        raw_text = response.candidates[0].content.parts[0].text
        self.memory.add("model", raw_text)

        # 4. Parse response → SocialMediaPack
        pack = self._parse_pack(raw_text, article.title)

        # 5. Persist output to disk so the journalist can review it
        self._save_pack(pack)

        return pack

    def chat(self, user_input: str) -> str:
        """
        Free conversational mode for direct journalist input.
        E.g.: "Asti, create a tweet about the importance of vitamin D"
        Returns raw text — no structured parsing.
        """
        self.memory.add("user", user_input)
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _personalized_system_prompt(self) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Newspaper: {self.newspaper_name}. Region: {self.region}."
        )

    def _build_prompt(
        self,
        article: CreateArticle,
        context_snippets: list[str],
    ) -> str:
        ctx_block = (
            "\n".join(f"- {s[:300]}" for s in context_snippets)
            or "No prior post examples available."
        )
        sources_block = ", ".join(article.sources) or "Not specified."
        keywords_block = ", ".join(article.keywords) or "Not specified."

        return f"""
ARTICLE TO ADAPT:
Title:    {article.title}
Category: {article.category}
Angle:    {article.angle}
Keywords: {keywords_block}
Sources:  {sources_block}
Local relevance score: {article.local_relevance_score:.0%}

FULL ARTICLE CONTENT:
{article.article_content}

STYLE EXAMPLES FROM RAG (past high-performing posts):
{ctx_block}

Generate the full SocialMediaPack for {self.newspaper_name}.
Remember: all post content in Spanish, image prompts in English.
Respond with the JSON format specified in your instructions.
""".strip()

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

    def _parse_pack(self, raw_text: str, article_title: str) -> SocialMediaPack:
        """
        Parses Gemini's JSON response into a SocialMediaPack.
        Each platform falls back to an empty SocialMediaPost if its
        section is missing or malformed — the pack never raises.
        """
        pack = SocialMediaPack(article_title=article_title)

        try:
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            pack.newsletter = SocialMediaPost(
                platform="newsletter",
                content=raw_text[:500],
            )
            return pack

        # ── Twitter ──────────────────────────────────────────────────────────
        if tw := data.get("twitter"):
            pack.twitter = SocialMediaPost(
                platform="twitter",
                content=tw.get("content", "")[:280],
                hashtags=tw.get("hashtags", []),
            )

        # ── Instagram ────────────────────────────────────────────────────────
        if ig := data.get("instagram"):
            pack.instagram = SocialMediaPost(
                platform="instagram",
                content=ig.get("content", ""),
                hashtags=ig.get("hashtags", []),
                image_prompt_midjourney=ig.get("image_prompt_midjourney", ""),
                image_prompt_vertex=ig.get("image_prompt_vertex", ""),
            )

        # ── Carousel ─────────────────────────────────────────────────────────
        if cr := data.get("carousel"):
            raw_slides = cr.get("slides", [])
            slides = [
                CarouselSlide(
                    slide_number=s.get("slide_number", i + 1),
                    headline=s.get("headline", ""),
                    body=s.get("body", ""),
                )
                for i, s in enumerate(raw_slides)
            ]
            pack.carousel = SocialMediaPost(
                platform="carousel",
                content=cr.get("content", ""),
                hashtags=cr.get("hashtags", []),
                slides=slides,
            )

        # ── Newsletter ───────────────────────────────────────────────────────
        if nl := data.get("newsletter"):
            pack.newsletter = SocialMediaPost(
                platform="newsletter",
                content=nl.get("content", ""),
                hashtags=nl.get("hashtags", []),
            )

        return pack

    def _save_pack(self, pack: SocialMediaPack) -> None:
        """
        Persists the SocialMediaPack as a JSON file in data/social_media_output/.
        Filename uses article title slug + timestamp to avoid collisions.
        """
        output_dir = os.getenv("SOCIAL_MEDIA_OUTPUT_DIR", "data/social_media_output")
        os.makedirs(output_dir, exist_ok=True)

        slug = (
            pack.article_title.lower()
            .replace(" ", "_")
            .replace("/", "-")[:60]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{slug}__{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(pack.to_dict(), f, ensure_ascii=False, indent=2)
