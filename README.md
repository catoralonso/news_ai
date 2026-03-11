# News Research Agent — Periódico Local IA

## Arquitectura del agente

```
newspaper_ai/
├── agents/
│   └── news_research/
│       ├── agent.py        ← NewsResearchAgent (clase principal)
│       └── run.py          ← Demo y loop conversacional
├── core/
│   ├── vector_store.py     ← ChromaDB wrapper (swap-able a Vertex AI)
│   ├── memory.py           ← Historial de conversación
│   └── chunker.py          ← Divide textos para RAG
├── tools/
│   └── search_tools.py     ← web_search, trending_topics, local_relevance
├── data/
│   ├── raw_news/           ← Artículos .txt del Content Engineer
│   └── embeddings/         ← ChromaDB persiste aquí
├── tests/
│   └── test_news_research.py
├── requirements.txt
└── .env.example
```

## Flujo del News Research Agent

```
query del usuario / orchestrator
        │
        ▼
1. ChromaDB.retrieve()          ← contexto histórico del periódico (RAG)
2. get_trending_topics()        ← pytrends (local, sin API key)
3. web_search()                 ← Google CSE o mock si no hay key
        │
        ▼
4. Gemini generate_content()    ← prompt enriquecido con todo lo anterior
        │
        ▼
5. Parsear JSON → ArticleIdea[] ← ideas estructuradas listas para Article Agent
```

## Setup local

```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env: agregar GEMINI_API_KEY
python agents/news_research/run.py
```

## Tests

```bash
pytest tests/test_news_research.py -v
```

## Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `GEMINI_API_KEY` | Key de AI Studio (desarrollo local) | Sí (local) |
| `GOOGLE_CLOUD_PROJECT` | Project ID de GCloud (producción) | Sí (prod) |
| `GOOGLE_CSE_KEY` | Google Custom Search API key | No (usa mock) |
| `CHROMA_PERSIST_DIR` | Directorio de ChromaDB | No (default: data/embeddings) |
