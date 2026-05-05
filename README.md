---
title: Savia
emoji: 🌿
colorFrom: pink
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🌿 Savia — AI-Powered Nutrition Newsroom

> An autonomous multi-agent system that runs a nutrition newspaper end-to-end: from trend research to published articles, social media, and reader interaction — fully automated on Google Cloud. For the Demo go to the tab 'Redaccion'

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Haiku-Anthropic-D97706?logo=anthropic&logoColor=white)
![Cloud Run](https://img.shields.io/badge/Cloud_Run-Deployed-34A853?logo=googlecloud&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Gateway-009688?logo=fastapi&logoColor=white)
![Terraform](https://img.shields.io/badge/Infra-Terraform-7B42BC?logo=terraform&logoColor=white)

🔗 **[Live Demo](https://savia-nutricion-inteligente.lovable.app/)**

---

## What does it do?

Every morning at 07:00 CET, the system wakes up and runs automatically:

1. **Researches** trending nutrition topics from RSS feeds, Google Trends, and reader clickstream
2. **Fact-checks** the findings before anything gets written
3. **Writes** a short, engaging article optimized for digital visibility
4. **Distributes** it across Twitter/X and Instagram with platform-ready copy
5. **Answers** reader questions via a streaming chatbot — grounded in published content

No manual intervention. Journalists focus on deep investigative work.

---

## Architecture

```
Lovable (React frontend)
        │  HTTPS + CORS
        ▼
FastAPI Gateway  ←  Cloud Run (single container)
        │
        ▼
   Orchestrator
        │
        ├── asyncio.gather ─────────────────┐
        ▼                                   ▼
José (research)                   Camila (fact-check)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
              Manuel (article generation)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
Asti (social media)        Mauro (reader chatbot)
```

Research and fact-checking run **in parallel**. Social media and chatbot spin up **in parallel** once the article is ready. Two `asyncio.gather` stages cut total pipeline time significantly.

---
## Performance (measured)

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 — José + Camila warmup (parallel) | ~13s | Both run via `asyncio.gather` |
| Stage 2 — Camila fact-checks 3 ideas | ~28s | ~9s per idea |
| Stage 3 — Manuel writes article | ~26s | Claude Sonnet, `temp=0.2` |
| Stage 4 — Asti + Mauro setup (parallel) | ~16s | Social pack + chatbot ready |
| **Total pipeline** | **~84s** | Research → published article + social pack |

**Throughput:** ~43 articles/hour (theoretical, no human selection step)  
**Sequential estimate:** Stage 4 alone would add ~17s without parallelization — Asti and Mauro setup run concurrently instead of waiting on each other.

**Fact-checking:** 3-class verdict system (`truthful / doubtful / untruthful`) evaluated against a curated set of known nutrition misinformation examples loaded into Camila's RAG at initialization. No formal accuracy benchmark — verdicts are cross-referenced against web search results and trusted sources (WHO, PubMed, Spanish Ministry of Health) at runtime.

**Chatbot:** Live demo at the March 2026 university exposition. Mauro answered reader questions in real-time, grounded in articles generated minutes earlier by the pipeline.

## The Agents

| Agent | Role | Key decisions |
|-------|------|---------------|
| **José** | Trend research · topic discovery | RSS (PubMed, Healthline) + Google Trends + clickstream; `temp=0.4` |
| **Camila** | Dual-mode fact-checking | Batch pipeline + live reader verification; 3-class verdict (`truthful / doubtful / untruthful`) |
| **Manuel** | Article generation | RAG-grounded writing with style examples; `temp=0.2` for consistency |
| **Asti** | Social media distribution | Twitter/X live; Instagram caption + Imagen prompt; `temp=0.7` |
| **Mauro** | Reader chatbot | SSE streaming; routes fact-check requests to Camila in real-time |
| **Orchestrator** | Pipeline coordination | Manual trigger via API or daily cron via Cloud Scheduler |

---

## RAG Design

Each agent owns one ChromaDB collection and **can only write to its own**. Cross-collection reads are unrestricted.

```
global_nutrition    ← shared knowledge base (studies, dietary guides)
news_research       ← José: topics already covered (avoids repetition)
article_style       ← Manuel: how this newspaper writes
article_published   ← Manuel → readable by José, Mauro, Asti
fact_checking       ← Camila: fake news patterns, trusted sources
reader_interaction  ← Mauro: FAQs, recurring question patterns
social_media        ← Asti: high-performing post examples
```

**Embeddings:** `gemini-embedding-001` via Vertex AI  
**Camila's RAG is selective:** only `untruthful` verdicts are persisted — keeps the collection signal-dense.

---

## Infrastructure

Everything provisioned with a single `terraform apply`. No manual GCP console steps.

| Service | Purpose |
|---------|---------|
| **Cloud Run** | Hosts the FastAPI container · scales to zero when idle |
| **Artifact Registry** | Docker image versioning |
| **Secret Manager** | API key injection at runtime |
| **Cloud Scheduler** | Daily pipeline trigger at 07:00 CET |
| **Cloud Logging** | Structured logs from all agents |
| **Cloud Trace (OTel)** | Per-request spans · auto-activates in GCP, no-op locally |

Observability uses module-level auto-detection: if `GOOGLE_CLOUD_PROJECT` is set → full tracing + logging. Otherwise → silent local fallback. Zero code branches between environments.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pipeline/run` | Launch pipeline · returns `job_id` immediately |
| `GET`  | `/api/pipeline/status/{job_id}` | Poll pipeline status |
| `POST` | `/api/chat` | Mauro chatbot · SSE streaming |
| `GET`  | `/api/trends` | Latest trends without running the full pipeline |
| `GET`  | `/api/articles` | List generated articles |
| `GET`  | `/api/social/{article_id}` | Social media pack for an article |
| `GET`  | `/docs` | Swagger UI |

---

## Stack

```
LLM          Claude Haiku 4.5 (José, Camila, Asti, Mauro) · Claude Sonnet 4.6 (Manuel)
Orchestration  LangChain · asyncio.gather
RAG          ChromaDB · gemini-embedding-001
API          FastAPI · SSE streaming
Frontend     Lovable (React) · CORS configured
Infra        Terraform · Cloud Run · Cloud Scheduler · Secret Manager
Observability  Cloud Logging · Cloud Trace · OpenTelemetry
Social       Twitter/X API · Vertex AI Imagen (Instagram)
```

---

## Engineering Decisions Worth Noting

- **ChromaDB over Vertex AI Vector Search** — local-first, no GCP dependency during dev; swappable via the `VectorStore` wrapper
- **FastAPI over Streamlit** — proper async support, background jobs, SSE streaming
- **Claude over Gemini** — migrated from Vertex AI due to API instability; Anthropic's API is significantly more reliable for production pipelines. Haiku for speed (José, Camila, Asti), Sonnet for quality (Manuel)
- **Temperature is intentional per agent** — `0.2` for Manuel (factual precision), `0.7` for Asti (creative copy)
- **Camila's fallback verdict is `no_information`** — avoids false positives polluting the fact-check collection

---

## Local Setup

```bash
git clone https://github.com/catoralonso/news_ai
cd news_ai

pip install -r requirements.txt

# .env
ANTHROPIC_API_KEY=sk-ant-...
NEWSPAPER_NAME=Savia

python orchestrator_run.py
```

For production deploy → see [Infrastructure](#infrastructure) section and `infra/main.tf`.

---

*Built as a course capstone project. Live exposition: March 2025.*
