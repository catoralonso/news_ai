# Local AI Newspaper System

An autonomous multi-agent system that helps local newspapers gain visibility, track trends, and generate content efficiently — so journalists can focus on deep investigative work.

---

## Project Structure

```
newspaper_ai/
├── config.py                   ← Central config + Cloud Logging/Trace setup
├── requirements.txt
├── Dockerfile                  ← Cloud Run container build
├── app.py                      ← Streamlit UI (legacy, not used in production)
├── .env
├── .gitignore
│
├── api/                        ← FastAPI gateway (replaces Streamlit)
│   ├── __init__.py
│   └── main.py                 ← HTTP endpoints consumed by Lovable frontend
│
├── infra/
│   └── main.tf                 ← Terraform: Cloud Run, Scheduler, Secrets, IAM
│
├── agents/
│   ├── adk_app/
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── jose_news_research/
│   │   ├── agent.py
│   │   └── run.py
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
│   ├── vector_store.py         ← ChromaDB wrapper (swappable to Vertex AI)
│   ├── memory.py               ← Conversation history
│   └── chunker.py              ← Text splitting for RAG
│
├── tools/
│   └── search_tools.py         ← web_search, trending_topics, local_relevance
│
└── data/
    ├── raw_news/               ← .txt articles from Content Engineer
    ├── articles/               ← Generated articles (JSON)
    ├── social_media_output/    ← SocialMediaPack per article (JSON)
    ├── trends/                 ← Google Trends CSV (manual export)
    ├── clickstream/            ← Reader events (future: logged by web)
    └── embeddings/
        ├── global_nutrition/   ← shared knowledge base (studies, guides)
        ├── news_research/      ← José: topics already covered
        ├── article_style/      ← Manuel: writing style examples
        ├── article_published/  ← Manuel: published articles
        ├── fact_checking/      ← Camila: fake news patterns, trusted sources
        ├── reader_interaction/ ← Mauro: FAQs, reader context
        └── social_media/       ← Asti: successful post examples
```

---

## Architecture

The system runs as a single Cloud Run service. The FastAPI gateway (`api/main.py`) is the only entry point — it receives HTTP requests from the Lovable web frontend and routes them to the appropriate agents.

```
Lovable (React · lovable.app)
        │  HTTPS + CORS
        ▼
api/main.py — FastAPI Gateway  (Cloud Run)
        │
        ▼
orchestrator/agent.py
        │
        ├── asyncio.gather ──────────────────────┐
        │                                        │
        ▼                                        ▼
jose_news_research              camila_fact_checking
(trends · RSS · RAG)            (verify · sources)
        │                                        │
        └──────────────┬─────────────────────────┘
                       ▼
            manuel_article_generation
               (write · RAG · Gemini)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
asti_social_media          mauro_reader_interaction
(tweet · ig · newsletter)  (chatbot · SSE · memory)
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Cloud Run health check |
| `GET`  | `/api/trends` | Latest trends from José (no full pipeline) |
| `POST` | `/api/pipeline/run` | Run full pipeline — returns `job_id` immediately |
| `GET`  | `/api/pipeline/status/{job_id}` | Poll pipeline status |
| `GET`  | `/api/articles` | List generated articles |
| `GET`  | `/api/articles/{id}` | Full article content |
| `GET`  | `/api/social/{article_id}` | SocialMediaPack for an article |
| `POST` | `/api/chat` | Mauro chatbot — SSE streaming |
| `GET`  | `/docs` | Swagger UI |

---

## RAG Architecture

Each agent has its own ChromaDB collection scoped to its specific task.

### Collections

| Collection | Owner | Who can read | Purpose |
|---|---|---|---|
| `global_nutrition` | Content Engineer | All agents | Scientific studies, dietary guides, general nutrition knowledge |
| `news_research` | José | José, Manuel, Camila | Topics already covered — avoids repetition |
| `article_style` | Manuel | Manuel | Style examples and newspaper writing guide |
| `article_published` | Manuel | José, Mauro, Asti | Published articles |
| `fact_checking` | Camila | Camila | Fake news patterns, trusted source index |
| `reader_interaction` | Mauro | Mauro | Reader FAQs and recurring question patterns |
| `social_media` | Asti | Asti | High-performing post examples per platform |

### Rules

- **Write only to your own collection.** An agent never calls `upsert()` on another agent's collection.
- **Read across collections freely.** Any agent can query another agent's collection when it adds value.

### Cross-collection reads in practice

```
José.run()
    ├── reads news_research/      ← what has already been covered
    └── reads article_published/  ← what has already been written

Manuel.run(idea)
    ├── reads article_style/      ← how the newspaper writes
    ├── reads article_published/  ← what has already been published
    └── reads news_research/      ← context from José's research

Mauro.chat(question)
    ├── reads reader_interaction/ ← recurring question patterns
    └── reads article_published/  ← answers grounded in real articles

Asti.run(article)
    ├── reads social_media/       ← successful post formats
    └── reads article_published/  ← source article content
```

---

## Agents

### José — News Research
Finds trending topics and generates article ideas for the editorial team.

- **Inputs:** orchestrator prompt or daily cron trigger
- **Tools:** pytrends, RSS (PubMed + Healthline), ChromaDB RAG, clickstream
- **Output:** `ArticleIdeas[]` — structured ideas ready for the article agent

### Camila — Fact Checking
Verifies claims in two contexts:

1. **Internal flow:** checks facts from José's research before articles are written
2. **External flow:** verifies news submitted by readers through Mauro's chatbot

- **Inputs:** article ideas from José, or user-submitted news via Mauro
- **Tools:** web search, trusted source lookup
- **Output:** verdict: truthful / doubtful / untruthful + reason + sources[]

### Manuel — Article Generation
Writes short, friendly news pieces optimized for visibility and engagement.

> Long-form investigative articles remain the journalists' responsibility. Manuel handles trend-driven, short-format content to keep the newspaper active and visible online.

- **Inputs:** verified ideas from Camila
- **Tools:** Gemini `generate_content`, ChromaDB RAG
- **Output:** `CreateArticle` — publication-ready article

### Asti — Social Media Distribution
Generates platform-ready content from published articles.

- **Platforms:** Twitter/X, Instagram (caption + image prompts), carousel, newsletter snippet
- **Inputs:** `CreateArticle` from Manuel or direct journalist prompt
- **Output:** `SocialMediaPack` saved to `data/social_media_output/`

### Mauro — Reader Interaction
The public-facing chatbot. Readers interact only with Mauro.

- **Inputs:** reader questions, submitted news tips
- **Tools:** ChromaDB RAG (published articles) + Gemini
- **Camila integration:** if a reader submits an external news claim, Mauro invokes Camila to fact-check it before responding
- **Output:** streaming SSE responses to the Lovable frontend

### Orchestrator
Controls the editorial workflow. Not visible to readers.

- **Automatic mode:** runs every day at 07:00 CET via Cloud Scheduler
- **Manual mode:** journalist triggers pipeline via `POST /api/pipeline/run`

| Journalist input | Agents activated |
|---|---|
| "Show me this week's trends" | José + Camila |
| "Write an article about X" | José → Camila → Manuel |
| "Post yesterday's article on social media" | Asti only |
| *(daily cron)* | Full pipeline |

---

## Infrastructure

Managed entirely via Terraform (`infra/main.tf`).

| Service | Purpose |
|---|---|
| **Cloud Run** | Hosts the FastAPI container — scales to 0 when idle |
| **Artifact Registry** | Stores Docker images |
| **Secret Manager** | Stores `GEMINI_API_KEY` securely |
| **Cloud Scheduler** | Triggers pipeline daily at 07:00 CET |
| **Cloud Logging** | Receives all `logging.*` calls from all agents automatically |
| **Cloud Trace** | Records timeline of every API request (spans per agent step) |

### Deploy with Terraform

```bash
cd infra/
terraform init
terraform apply \
  -var="project_id=YOUR_PROJECT" \
  -var="gemini_api_key=YOUR_KEY"
```

Outputs the `api_url` to set as `VITE_API_URL` in Lovable.

### Deploy manually

```bash
# Build and push Docker image
docker build -t europe-west1-docker.pkg.dev/YOUR_PROJECT/newspaper-ai/newspaper-ai:latest .
docker push europe-west1-docker.pkg.dev/YOUR_PROJECT/newspaper-ai/newspaper-ai:latest

# Deploy to Cloud Run
gcloud run deploy newspaper-ai \
  --image europe-west1-docker.pkg.dev/YOUR_PROJECT/newspaper-ai/newspaper-ai:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

---

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add GEMINI_API_KEY

# Run the API gateway locally
uvicorn api.main:app --reload --port 8080

# Or run individual agents
python agents/jose_news_research/run.py
python agents/orchestrator/run.py
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | AI Studio key (local development) | Yes (local) |
| `GOOGLE_CLOUD_PROJECT` | GCloud Project ID — activates Vertex AI + observability | Yes (prod) |
| `GOOGLE_CLOUD_REGION` | GCloud region (default: `europe-west1`) | No |
| `NEWSPAPER_NAME` | Newspaper display name (default: `Nutrición AI`) | No |
| `CHAT_MODEL` | Gemini model (default: `gemini-2.0-flash`) | No |
| `CHROMA_PERSIST_DIR` | ChromaDB directory (default: `data/embeddings`) | No |
| `TWITTER_BEARER_TOKEN` | Twitter/X API key | Yes (Asti) |
| `INSTAGRAM_ACCESS_TOKEN` | Instagram Graph API token | Yes (Asti) |

---

## User Types

| User | Interface | Agents accessible |
|---|---|---|
| Reader | Mauro chatbot (Lovable web) | Mauro (+ Camila indirectly) |
| Journalist / Editor | Lovable web + `/docs` Swagger | All agents via orchestrator |

---

## Roadmap

- [x] Agent skeleton (all 6 agents)
- [x] ChromaDB RAG core
- [x] RAG collection architecture (per-agent + global)
- [x] FastAPI gateway (`api/main.py`) — replaces Streamlit
- [x] Dockerfile + Cloud Run deployment
- [x] Terraform infrastructure (`infra/main.tf`)
- [x] Cloud Logging + Cloud Trace observability
- [x] Cloud Scheduler — daily pipeline at 07:00 CET
- [x] Lovable frontend integration (CORS + SSE)
- [x] Orchestrator task routing with Gemini
- [x] Camila external fact-check integration
- [ ] Asti Phase 1 — text posts (Twitter, Instagram)
- [ ] Asti Phase 2 — AI image generation for Instagram
- [ ] Firestore / Cloud Storage for persistent article storage
- [ ] Vertex AI Vector Search (swap from ChromaDB)
