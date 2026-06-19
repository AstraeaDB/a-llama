//! a-llama — a fully-embedded Ollama replacement with continual learning.
//!
//! AstraeaDB + Eunomia are compiled in-process as the memory subsystem; the
//! crate exposes an Ollama-compatible local HTTP daemon (built in later M1
//! tasks). See `DESIGN.md` for the architecture and the learn → store →
//! retrieve → augment loop.

/// Embedded knowledge store: an in-process AstraeaDB graph + vector index
/// holding the model's accumulated knowledge (Task 2, dev-astraea-graph).
pub mod knowledge_store;

/// Exact (brute-force) `VectorIndex` — perfect recall for the `KnowledgeStore`.
/// (Originally chosen over astraea-vector's HNSW, whose recall collapsed past ~3k
/// vectors — astradb #25, since fixed in 0.1.12; the exact index is retained.)
pub mod exact_index;

/// Local LLM inference: the swappable `InferenceEngine` trait and its backends,
/// plus the `astraea_rag` provider adapters (Task 3).
pub mod inference;

/// GraphRAG augment loop: retrieves learned knowledge from the `KnowledgeStore`
/// and assembles the augmented prompt at inference time (Task 4, dev-rag).
pub mod augment;

/// Eunomia semantic cache integration: the hot prompt→response cache and
/// recent working memory (Task 5).
pub mod cache;

/// Ollama-compatible HTTP daemon + the request orchestrator that runs the full
/// learn → store → retrieve → augment loop (Task 6).
pub mod api;

/// LLM-based fact/entity extraction: distills `Fact`/`Entity` nodes + edges from
/// interactions so learned knowledge is independently retrievable (Task 9, M2).
pub mod extract;
