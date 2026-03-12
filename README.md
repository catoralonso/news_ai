# Local AI Newspaper System

An autonomous multi-agent system that helps local newspapers gain visibility, track trends, and generate content efficiently — so journalists can focus on deep investigative work.

---

## Project Structure

```
newspaper_ai/
├── agents/
│   ├── jose_news_research/
│   │   ├── agent.py        ← NewsResearchAgent (main class)
│   │   └── run.py          ← Demo and conversational loop
│   ├── camila_fact_checking/
│   │   ├── agent.py
│   │   └── run.py
│   ├── manuel_article_generation/
│   │   ├── agent.py
│   │   └── run.py
│   ├── asti_social_media/
│   │   ├── agent.py
│   │   └── run.py
│   ├── mauro_reader_interaction/
│   │   ├── agent.py
│   │   └── run.py
│   └── orchestrator/
│       ├── agent.py
│       └── run.py
│
├── core/
│   ├── vector_store.py     ← ChromaDB wrapper (swappable to Vertex AI)
│   ├── memory.py           ← Conversation history
│   └── chunker.py          ← Text splitting for RAG
├── tools/
│   └── search_tools.py     ← web_search, trending_topics, local_relevance
├── data/
│   ├── raw_news/           ← .txt articles from Content Engineer
│   └── embeddings/         ← ChromaDB persists here
├── tests/
├── requirements.txt
└── .env.example
```

---

## Agents

### Jose — News Research
Finds trending topics and generates article ideas for the editorial team.

- **Inputs:** orchestrator prompt or weekly cron trigger
- **Tools:** pytrends, Google Custom Search (or mock), ChromaDB RAG
- **Output:** `ArticleIdeas[]` — structured ideas ready for the article agent

---

### Camila — Fact Checking
Verifies claims in two contexts:

1. **Internal flow:** checks facts from Jose's research before articles are written
2. **External flow:** verifies news that readers submit through Mauro's chatbot

- **Inputs:** article ideas from Jose, or user-submitted news via Mauro
- **Tools:** web search, trusted source lookup
- **Output:** verdict: truthful / doubtful / untruthful + reason + sources[]

---

### Manuel — Article Generation
Writes short, friendly news pieces optimized for visibility and engagement.

> Long-form investigative articles remain the journalists' responsibility. Manuel handles trend-driven, short-format content to keep the newspaper active and visible online.

- **Inputs:** verified ideas from Camila
- **Tools:** Gemini (`generate_content`)
- **Output:** publication-ready article (short format, friendly tone)

---

### Asti — Social Media Distribution
Publishes content across social platforms.

- **Platforms:** Twitter/X, LinkedIn, Instagram (caption + hashtags)
- **Phase 1:** text-only posts — journalists add images manually for Instagram
- **Phase 2 (future):** AI image generation via Imagen API
- **Inputs:** article from Manuel or direct prompt from journalist
- **Output:** published posts per platform

---

### Mauro — Reader Interaction
The public-facing chatbot. Readers interact only with Mauro.

- **Inputs:** reader questions, submitted news tips
- **Tools:** ChromaDB RAG (published articles) + Gemini
- **Camila integration:** if a reader submits an external news claim, Mauro invokes Camila to fact-check it before responding
- **Output:** informed, grounded responses to readers

---

### Orchestrator
Controls the workflow for the editorial team. Not visible to readers.

- **Automatic mode:** runs every Monday via cron — Gemini analyzes last week's context and decides which agents to activate
- **Manual mode:** journalist provides a prompt and the orchestrator interprets intent and routes to the right agents

**Examples:**
| Journalist input | Agents activated |
|---|---|
| "Show me this week's trends" | Jose + Camila (always paired)
| "Write an article about X" | Jose → Camila → Manuel |
| "Post yesterday's article on social media" | Asti only |
| *(no input, Monday cron)* | Gemini decides based on context |

---

## User Types

| User | Interface | Agents accessible |
|---|---|---|
| Reader | Mauro chatbot | Mauro (+ Camila indirectly) |
| Journalist / Editor | Orchestrator CLI or UI | All agents via orchestrator |

---

## System Flow

```
READER
    │
    ▼
mauro_reader_interaction
    ├── RAG (ChromaDB) ← published articles, knowledge base
    ├── Gemini         ← enriched responses
    └── camila         ← invoked if reader submits external news


JOURNALIST / CRON (Monday)
    │
    ▼
orchestrator
    ├── jose_news_research + camila_fact_checking  (always together)
    ├── manuel_article_generation (optional — if an idea is approved)
    └── asti_social_media        (optional — if article is ready to publish)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add your API keys
python agents/orchestrator/run.py
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | AI Studio key (local development) | Yes (local) |
| `GOOGLE_CLOUD_PROJECT` | GCloud Project ID (production) | Yes (prod) |
| `GOOGLE_CSE_KEY` | Google Custom Search API key | No (uses mock) |
| `CHROMA_PERSIST_DIR` | ChromaDB directory | No (default: `data/embeddings`) |
| `TWITTER_BEARER_TOKEN` | Twitter/X API key | Yes (Asti) |
| `LINKEDIN_ACCESS_TOKEN` | LinkedIn API token | Yes (Asti) |
| `INSTAGRAM_ACCESS_TOKEN` | Instagram Graph API token | Yes (Asti) |

---

## Roadmap

- [x] Agent skeleton (all 6 agents)
- [x] ChromaDB RAG core
- [ ] Orchestrator task routing with Gemini
- [ ] Camila external fact-check integration
- [ ] Asti Phase 1 — text posts (Twitter, LinkedIn, Instagram)
- [ ] Monday cron automation
- [ ] Asti Phase 2 — AI image generation for Instagram
- [ ] Deep investigative tools for journalists
