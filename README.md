# Self-Learning AI Agent

An experience-driven AI agent combining a local LLM with a Jev-style structured decision layer, episodic/semantic/strategy memory, evaluation, and self-correction. **100% local — no API keys, no cloud.**

## Quickstart (fully local)

### 1. Install Ollama and pull a model

Download: https://ollama.com/download

```bash
ollama pull qwen2.5:3b     # ~2 GB, runs on 8 GB RAM
# or if you have 16 GB RAM:
# ollama pull qwen2.5:7b   # ~4.7 GB, better quality
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

### 2. Install Python deps

```bash
cp .env.example .env
pip install -r requirements.txt
```

Edit `.env` and set `OLLAMA_MODEL` to whatever you pulled. No other changes needed.

### 3. (optional) Ingest docs for RAG

```bash
mkdir -p data/knowledge
# drop .txt / .md files in
python -m scripts.ingest
```

### 4. Run

```bash
uvicorn app.api.main:app --reload            # terminal 1
streamlit run frontend/streamlit_app.py      # terminal 2
```

Open http://localhost:8501.

### Or with Docker

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:3b
docker compose up
```

## Learning loop

```
understand -> retrieve -> plan -> decide -> execute -> evaluate
                                              ^            |
                                              |            v
                                            correct <-- failure
                                                         |
                                                      success
                                                         v
                                                      finalize -> post_evaluate -> memory
```

## Design principles

- **AI makes the judgment; software controls the action.**
- Jev only produces Choice / Score / Noul decisions.
- Every Jev response is schema-validated. Mismatches raise `JevSchemaError`.
- Retries are bounded by `MAX_RETRIES`. Corrections choose one lever: rephrase / switch tool / give up.
- Thresholds in `.env` are prototypes; calibrate on the benchmark before trusting them.

## Leakage guarantees

Three independent guards prevent self-retrieval:

1. `exclude_task=task` — the current task is never returned by episodic search.
2. `isolate_run=True` — the current benchmark run's own episodes are excluded from retrieval.
3. `heldout_task_ids` — the held-out eval split is never visible to retrieval, even as a "past experience."

Production (`/run`) sets `isolate_run=False` so the agent *does* learn across calls.

## Benchmark protocol

```bash
python -m scripts.run_evaluation --mode cold --clear
python -m scripts.run_evaluation --mode warm
python -m scripts.compare_runs mlruns/cold_*.json mlruns/warm_*.json
```

Learning gain = warm success − cold success. If that number collapses when the guards are on, the previous gain was the leak.

## Hallucination guards

- Post-generation groundedness check via Jev (skeptic prompt).
- Deterministic grounding for calculator answers.
- Empty tool output is treated as failure.
- Refuse-or-hedge when evidence sufficiency is low.

## Cost & privacy

| Component | Where it runs | Cost |
|---|---|---|
| LLM | Your PC via Ollama | $0 |
| Jev (stub) | Same local LLM | $0 |
| Embeddings | `sentence-transformers` on CPU | $0 |
| Vector store | FAISS on disk | $0 |
| Web search | DuckDuckGo | $0 |
| API + UI | localhost | $0 |

Only outbound request is DuckDuckGo (only when the agent picks `web_search`). Everything else is `localhost`. Your prompts, documents, and memory never leave your machine.

To make it fully offline (zero outbound requests), remove `"web_search"` from the tool list in `app/tools/registry.py`.