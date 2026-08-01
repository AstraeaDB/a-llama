# a-llama

**A fully-embedded Ollama replacement with continual learning.**

a-llama is a single binary: a local LLM inference daemon that speaks an
**Ollama-compatible HTTP API**, with an embedded graph + vector database
([AstraeaDB](https://github.com/AstraeaDB/AstraeaDB-Official)) and a semantic
cache (Eunomia) compiled **in-process** as its memory subsystem. No separate
database server, no storage daemon, no cloud.

If you've used Ollama, that's the starting point: run a local model, talk to it
over HTTP. a-llama adds the thing Ollama doesn't have — **a memory that grows**.
It answers your question, *remembers* it, and gets more useful the more you use
it, **without retraining the model**. Turn on durable mode and that memory
survives a restart: a question answered in one process is recalled instantly by
a fresh process from the same data directory.

> **Honest framing.** a-llama is "Ollama *plus* an embedded learning memory."
> The quality of answers and extracted facts tracks whichever GGUF model you
> load. It runs one model at a time and targets the common Ollama verbs
> (`generate`, `chat`, `embeddings`, `tags`) — not yet the full model-management
> surface.

---

## Features

- **Drop-in local LLM server** — point any Ollama-compatible client, script, or
  SDK at it.
- **Semantic caching** — repeated or rephrased questions are served from cache
  (~15× faster in our tests) instead of regenerating.
- **Continual learning** — interactions and distilled facts are stored in an
  embedded knowledge graph and fed back into future prompts via GraphRAG.
- **Fact extraction** (opt-in) — distills `Fact`/`Entity` nodes and relations
  from conversations into an inspectable knowledge graph.
- **Durable persistence** (opt-in) — learned memory is written to disk and
  survives process restarts.
- **Fully embedded** — one binary; model, database, and vector search all run
  in-process. Good for offline, air-gapped, or privacy-sensitive use.
- **Perfect-recall retrieval** — the knowledge store uses an exact vector index
  (recall@k = 1.0), so it reliably surfaces what it learned.

## Quick start

```bash
# Default build: instant boot with a deterministic built-in mock engine
# (no model file, no network) — great for trying the API and the memory loop.
cargo run --release
# → a-llama listening on http://127.0.0.1:11434  (model: mock-gguf)
```

```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"q","prompt":"What is a knowledge graph?","stream":false}'
```

### Run a real model

```bash
cargo build --release --features mistralrs-engine

A_LLAMA_ENGINE=mistralrs \
A_LLAMA_GGUF=/abs/path/to/model.gguf \
A_LLAMA_EMBED_MODEL=google/embeddinggemma-300m \
  ./target/release/a-llama
```

The real backend uses [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
for in-process GGUF chat and `embeddinggemma` (768-dim) for embeddings. The
embedder is a gated Hugging Face model, so place a token at
`~/.cache/huggingface/token`.

### Durable mode (memory survives restarts)

```bash
cargo build --release --features durable          # or: mistralrs-engine,durable
A_LLAMA_DATA_DIR=/var/lib/a-llama ./target/release/a-llama
```

`Ctrl-C` / `SIGTERM` flushes the graph and cache to disk; the next start replays
them. Engine choice and storage choice are independent — any combination works.

## HTTP API

| Method & path | Purpose |
| --- | --- |
| `POST /api/generate` | Single-prompt generation (JSON or NDJSON stream) |
| `POST /api/chat` | Multi-message chat |
| `POST /api/embeddings`, `POST /api/embed` | 768-dim embeddings |
| `GET /api/tags` | List available models |
| `GET /api/version`, `GET /` | Version / health |

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `A_LLAMA_ADDR` | `127.0.0.1:11434` | Bind address (Ollama's port) |
| `A_LLAMA_ENGINE` | mock | `mistralrs` selects the real engine (needs the feature) |
| `A_LLAMA_GGUF` | — | Path to the chat GGUF |
| `A_LLAMA_EMBED_MODEL` | `google/embeddinggemma-300m` | 768-dim embedder (HF id, gated) |
| `A_LLAMA_EXTRACT_FACTS` | `0` | `1` enables detached fact extraction |
| `A_LLAMA_DATA_DIR` | — | Durable mode (needs `--features durable`) |

**Cargo features:** `mistralrs-engine` (real GGUF backend), `durable` (disk
persistence). The default build is in-memory with the mock engine.

## How it works

Every request runs **learn → store → retrieve → augment**: embed the prompt →
check the semantic cache → on a miss, run a GraphRAG query over the learned
knowledge graph and splice the context into the prompt → generate → store the
interaction → (optionally) tier evicted entries into the durable graph and
extract facts. See [`docs/index.html`](docs/index.html) for the full
architecture, component reference, and design notes.

## Building from source

a-llama depends on the AstraeaDB crates and the Eunomia cache engine:

- **AstraeaDB** (`astraea-core`/`-graph`/`-vector`/`-rag`/`-storage`) — from
  [crates.io](https://crates.io/crates/astraea-core) (`= "0.3.1"`).
- **`eunomia-core`** — from [crates.io](https://crates.io/crates/eunomia-core)
  (`= "0.1.0"`, the default `hnsw_rs` backend).

To develop against a local AstraeaDB checkout, replace the version entries in
`Cargo.toml` with `path = "../astraeadb/crates/<crate>"`.

## Project layout

```
src/
  main.rs            binary: engine/storage selection, serve + graceful persist
  api.rs             orchestrator (AppState) + Ollama-compatible router
  knowledge_store.rs embedded AstraeaDB graph + exact vector index
  exact_index.rs     ExactVectorIndex (perfect recall)
  inference.rs       InferenceEngine / Mock / MistralRs / provider adapters
  augment.rs         GraphRAG Augmenter
  cache.rs           Eunomia SemanticCache
  extract.rs         LLM fact/entity extraction
tests/               integration tests
examples/recall.rs   vector-index recall benchmark
docs/index.html      full documentation
```

## License

[MIT](LICENSE) © 2026 Jim Harris.
