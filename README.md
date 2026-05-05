# 🌿 Savia — AI-Powered Nutrition Newsroom

> An autonomous multi-agent system that runs a nutrition newspaper end-to-end: from trend research to published articles, social media, and reader interaction. For the Demo go to the link:
> 
🔗 **[Live Demo](https://huggingface.co/spaces/catoralonso/news_savia)**

**General**
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Gateway-009688?logo=fastapi&logoColor=white)

**Current — Anthropic + Hugging Face**
![HuggingFace](https://img.shields.io/badge/Hosted-HuggingFace_Spaces-FFD21E?logo=huggingface&logoColor=black)
![Claude](https://img.shields.io/badge/Claude_Haiku_4.5_·_Sonnet_4.6-Anthropic-D97706?logo=anthropic&logoColor=white)

**Legacy — Google Cloud Platform**
![Cloud Run](https://img.shields.io/badge/Cloud_Run-Deployed-34A853?logo=googlecloud&logoColor=white)
![Terraform](https://img.shields.io/badge/Infra-Terraform-7B42BC?logo=terraform&logoColor=white)

---

## What does it do?

1. **Researches** trending nutrition topics from RSS feeds, Google Trends, and reader clickstream
2. **Fact-checks** the findings before anything gets written
3. **Writes** a short, engaging article optimized for digital visibility
4. **Distributes** it across Twitter/X and Instagram with platform-ready copy
5. **Answers** reader questions via a streaming chatbot — grounded in published content

No manual intervention. Journalists focus on deep investigative work.

---

## Architecture

Frontedn for GCP was in Lovable in the tab "Redaccion" were the agents, they dont work due to migration to Anthropic (check live demo above)
🔗 **[Link](https://savia-nutricion-inteligente.lovable.app/)

```
Vanilla JS (static frontend · Hugging Face Spaces) 
        │  HTTPS
        ▼
FastAPI Gateway  ←  Hugging Face Spaces (Docker) | Cloud Run in GCP (legacy)
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

---

## The Agents

| Agent | Role | Key decisions |
|-------|------|---------------|
| **José** | Trend research · topic discovery | RSS (PubMed, Healthline) + Google Trends + clickstream; `temp=0.4` |
| **Camila** | Dual-mode fact-checking | Batch pipeline + live reader verification; 3-class verdict (`truthful / doubtful / untruthful`) |
| **Manuel** | Article generation | RAG-grounded writing with style examples; `temp=0.2` for consistency |
| **Asti** | Social media distribution | Twitter/X live; Instagram caption + image prompt; `temp=0.7` |
| **Mauro** | Reader chatbot | SSE streaming; routes fact-check requests to Camila in real-time |
| **Orchestrator** | Pipeline coordination | Manual trigger via API or daily cron |

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

**Embeddings:** `all-MiniLM-L6-v2` (ChromaDB default)  
**Camila's RAG is selective:** only `untruthful` verdicts are persisted — keeps the collection signal-dense.

---

## Infrastructure

### Current — Hugging Face Spaces

| Service | Purpose |
|---------|---------|
| **Hugging Face Spaces** | Hosts FastAPI container + static frontend (Docker) |
| **Anthropic API** | LLM inference · Haiku 4.5 + Sonnet 4.6 |

### Legacy — Google Cloud Platform

Full GCP infrastructure preserved in [`/gcp_infrastructure`](./gcp_infrastructure).

| Service | Purpose |
|---------|---------|
| **Cloud Run** | Hosts the FastAPI container · scales to zero when idle |
| **Artifact Registry** | Docker image versioning |
| **Secret Manager** | API key injection at runtime |
| **Cloud Scheduler** | Daily pipeline trigger at 07:00 CET |
| **Cloud Logging** | Structured logs from all agents |
| **Cloud Trace (OTel)** | Per-request spans · auto-activates in GCP, no-op locally |

GCP infra was provisioned with a single `terraform apply`. Migrated to HF Spaces to eliminate GCP billing dependency — Anthropic's free tier covers the full pipeline at this scale.

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
LLM            Claude Haiku 4.5 (José, Camila, Asti, Mauro) · Claude Sonnet 4.6 (Manuel) · Gemini 2.5 flash (GCP legacy)
Orchestration  LangChain · asyncio.gather
RAG            ChromaDB · all-MiniLM-L6-v2
API            FastAPI · SSE streaming
Frontend       Vanilla JS in Hugging Face Spaces · Lovable/React (GCP legacy)
Infra          Hugging Face Spaces (Docker) · GCP legacy in /gcp_infrastructure
Observability  Python logging
Social         Twitter/X · Instagram · Carrousel · Newsletter (mockups)
```

---

## Engineering Decisions Worth Noting

- **ChromaDB over Vertex AI Vector Search** — local-first, no GCP dependency during dev; swappable via the `VectorStore` wrapper. Only to avoid GCP billing.
- **FastAPI over Streamlit** — proper async support, background jobs, SSE streaming
- **Claude over Gemini** — migrated from Vertex AI to avoid GCP billing; Anthropic's free tier covers the full pipeline at this scale. Haiku for speed (José, Camila, Asti), Sonnet for quality (Manuel)
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

For legacy GCP deploy → see [`/gcp_infrastructure`](./gcp_infrastructure).

---

*Built as a course capstone project. Live exposition: March 2026.*
