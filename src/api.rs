//! Task 6 — Ollama-compatible HTTP daemon + request orchestrator.
//!
//! This module ties tasks 2–5 together into the running daemon:
//!
//! - [`AppState`] holds the shared, `Arc`-wrapped subsystems: the inference
//!   [`InferenceEngine`](crate::inference::InferenceEngine) (for direct
//!   `embed`/`complete`), the GraphRAG [`Augmenter`](crate::augment::Augmenter),
//!   and the Eunomia [`SemanticCache`](crate::cache::SemanticCache) behind a
//!   `tokio::sync::RwLock` (it needs `&mut self` to write).
//! - [`AppState::generate`] runs the **learn → store → retrieve → augment** loop
//!   (DESIGN.md §1.3): embed → cache lookup → (on miss) GraphRAG-augmented
//!   completion → store the interaction back into the cache.
//! - [`router`] exposes the Ollama-compatible HTTP surface (`/api/generate`,
//!   `/api/chat`, `/api/embeddings`, `/api/embed`, `/api/tags`, `/api/version`,
//!   `GET /`) over `axum` 0.8 with both non-streaming JSON and NDJSON streaming
//!   shapes.
//!
//! ## Runtime requirement
//!
//! The orchestrator drives GraphRAG, which bridges the async engine to
//! astraea-rag's synchronous provider traits via `block_in_place` (see
//! `src/inference.rs` and `src/augment.rs`). It therefore **must run on a
//! multi-threaded Tokio runtime** — `src/main.rs` uses the default
//! `#[tokio::main]` (multi-thread) and the tests use
//! `#[tokio::test(flavor = "multi_thread")]`. A `current_thread` runtime would
//! panic inside the nested `block_in_place`.
//!
//! ## Deferred to M2 (DESIGN.md §6 tasks 7–9)
//!
//! M1 stores interactions in the Eunomia hot cache only. Durable tiering of
//! evicted/promoted entries into the AstraeaDB graph as `CacheEntry` nodes (the
//! `eunomia-astraea` bridge), the real mistral.rs engine, and the full
//! Modelfile/`/api/pull` surface are all M2. `generate` documents the point at
//! which a Task-7 flush would hook in.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::{FromRequest, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::RwLock;

use astraea_rag::{GraphRagConfig, TextFormat};

use crate::augment::Augmenter;
use crate::cache::SemanticCache;
use crate::extract::Extractor;
use crate::inference::{EngineProvider, InferenceEngine, MockEngine};
use crate::knowledge_store::KnowledgeStore;

// ===========================================================================
// Orchestrator configuration + state
// ===========================================================================

/// Tunable behaviour of the request orchestrator.
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    /// When `true`, a cache hit above the [`SemanticCache`] threshold
    /// short-circuits the loop and serves the cached response without calling
    /// the inference engine (classic semantic-cache behaviour). Default `true`.
    pub serve_cache_hits: bool,
    /// TTL applied to every stored interaction. `None` = no expiry. Default 1h.
    pub ttl_secs: Option<u64>,
    /// When `true`, [`AppState::generate`] opportunistically flushes any
    /// TTL-evicted hot-cache interactions into the durable [`KnowledgeStore`]
    /// at the end of each call (Task 7 tiering trigger). The sweep is cheap when
    /// nothing has expired (the common case), and a tiering failure never fails
    /// the request. Default `true`.
    pub tier_on_generate: bool,
    /// Upper bound, in bytes, on the serialized JSON `value` of a `CacheEntry`
    /// node flushed during tiering (astraedb #26). A `CacheEntry` carries the
    /// 768-dim prompt embedding (~3 KB) **plus** this `value`; if the full
    /// `{prompt, response, …}` JSON is large the record can exceed astraedb's
    /// ~8 KB single-page limit and `upsert_cache_entry` fails with
    /// `record too large for a single page`. Before flushing, the orchestrator
    /// truncates the `response` (then, if still over budget, the `prompt`) so
    /// the serialized `value` fits within this bound, tagging it
    /// `"truncated": true`. The prompt embedding is always preserved so the node
    /// stays retrievable by GraphRAG. Default 4096.
    pub max_tiered_value_bytes: usize,
    /// When `true`, the miss path of [`AppState::generate`] runs an LLM
    /// fact-extraction pass ([`Extractor`](crate::extract::Extractor)) over the
    /// fresh `prompt → response` exchange and upserts the distilled
    /// `Fact`/`Entity` nodes + edges into the durable [`KnowledgeStore`]
    /// (DESIGN.md §4, task 9; Q6).
    ///
    /// **Cost/benefit (default `false`).** Extraction is an *extra* LLM
    /// completion on every cache miss — roughly doubling per-request inference
    /// cost — so it is opt-in. The benefit is a richer, connected knowledge
    /// graph: individual facts become independently GraphRAG-retrievable and
    /// related entities are linked, so later differently-phrased prompts can
    /// surface a single distilled fact rather than only whole-interaction
    /// `CacheEntry` nodes. Extraction failures never fail the request (logged
    /// and swallowed), and extraction quality tracks the chat model's ability to
    /// emit clean JSON.
    pub extract_facts: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            serve_cache_hits: true,
            ttl_secs: Some(3600),
            tier_on_generate: true,
            max_tiered_value_bytes: 4096,
            extract_facts: false,
        }
    }
}

/// Shared application state — the orchestrator that runs the learn → store →
/// retrieve → augment loop over the in-process subsystems.
///
/// Cloneable handle (`Arc` internally); pass `AppState` directly as the axum
/// router state.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    /// Direct handle to the inference engine for `embed` (cache key) and the
    /// empty-graph completion fallback. Shared with the augmenter's provider.
    engine: Arc<dyn InferenceEngine>,
    /// The orchestrator's long-term [`KnowledgeStore`]. This is the **same**
    /// `Arc` the `augmenter` retrieves over (see [`AppState::with_store`]), so
    /// seeding facts/cache-entries through [`AppState::knowledge_store`] is
    /// visible to GraphRAG retrieval — and it is the store Task 7 (durable
    /// tiering) will flush evicted/promoted Eunomia entries into.
    store: Arc<KnowledgeStore>,
    /// GraphRAG augment loop over the shared `KnowledgeStore`.
    augmenter: Augmenter,
    /// Eunomia hot semantic cache. `RwLock` because `store_interaction` /
    /// `sweep_expired` need `&mut self`.
    cache: RwLock<SemanticCache>,
    /// Configured model name advertised by `/api/tags` and used as the default
    /// when a request omits `model`.
    model: String,
    /// Orchestrator behaviour knobs.
    config: OrchestratorConfig,
}

/// Outcome of one orchestrator [`AppState::generate`] call.
#[derive(Clone, Debug)]
pub struct GenerateOutcome {
    /// The generated (or cached) response text.
    pub response: String,
    /// The model name attributed to the response.
    pub model: String,
    /// `true` if the response was served from the semantic cache.
    pub from_cache: bool,
    /// `true` if GraphRAG context was assembled and fed to the engine.
    pub context_used: bool,
}

impl AppState {
    /// Build an [`AppState`] over an arbitrary [`InferenceEngine`] trait object
    /// and a fresh in-process [`KnowledgeStore`] + [`SemanticCache`], both
    /// pinned to the engine's embedding dimension.
    pub fn from_engine(engine: Arc<dyn InferenceEngine>) -> anyhow::Result<Self> {
        Self::with_config(engine, OrchestratorConfig::default())
    }

    /// Like [`AppState::from_engine`] with an explicit [`OrchestratorConfig`].
    ///
    /// Builds a fresh in-process [`KnowledgeStore`] pinned to the engine's
    /// embedding dimension and delegates to [`AppState::with_store`]. Use
    /// [`with_store`](AppState::with_store) directly to share or pre-seed the
    /// store from outside.
    pub fn with_config(
        engine: Arc<dyn InferenceEngine>,
        config: OrchestratorConfig,
    ) -> anyhow::Result<Self> {
        let dim = engine.embedding_dim();
        let store = Arc::new(KnowledgeStore::with_dim(dim)?);
        Self::with_store(engine, store, config)
    }

    /// Build an [`AppState`] over an externally-built [`KnowledgeStore`],
    /// sharing that exact `Arc` between the orchestrator and the GraphRAG
    /// [`Augmenter`].
    ///
    /// This is the constructor that closes the continual-learning loop at the
    /// orchestrator level: facts seeded into `store` (directly, or — Task 7 —
    /// flushed in from evicted Eunomia entries) are retrievable by the same
    /// store the augmenter reads, so [`AppState::generate`] can return
    /// `context_used = true`. [`AppState::knowledge_store`] hands back a clone of
    /// this same `Arc`.
    ///
    /// The caller must ensure `store`'s dimension matches
    /// `engine.embedding_dim()`; [`with_config`](AppState::with_config) and
    /// [`from_engine`](AppState::from_engine) guarantee this for the store they
    /// build.
    pub fn with_store(
        engine: Arc<dyn InferenceEngine>,
        store: Arc<KnowledgeStore>,
        config: OrchestratorConfig,
    ) -> anyhow::Result<Self> {
        let dim = engine.embedding_dim();

        // One EngineProvider wraps the shared engine and serves as BOTH the
        // LlmProvider and EmbeddingProvider the augmenter needs.
        let provider = Arc::new(EngineProvider::new(Arc::clone(&engine)));
        // Mirror `Augmenter::default_config_for` (which requires a `Sized`
        // engine and so can't take our `&dyn InferenceEngine`): allocate 25 % of
        // the input window to GraphRAG context, capped at 8 000 tokens.
        let context_budget = (engine.input_context_tokens() / 4).min(8_000);
        let rag_config = GraphRagConfig {
            hops: 2,
            max_context_nodes: 20,
            text_format: TextFormat::Structured,
            token_budget: context_budget,
            system_prompt: Some(
                "You are a helpful assistant with access to a knowledge graph \
                 of prior interactions and learned facts. Use the provided \
                 context to inform your answer."
                    .to_string(),
            ),
        };
        // The augmenter retrieves over the SAME Arc the getter returns, so
        // seeding the store is visible to GraphRAG.
        let augmenter = Augmenter::with_provider(Arc::clone(&store), provider, rag_config);

        let model = engine.name().to_string();
        let cache = RwLock::new(SemanticCache::new(dim));

        Ok(Self {
            inner: Arc::new(AppStateInner {
                engine,
                store,
                augmenter,
                cache,
                model,
                config,
            }),
        })
    }

    /// The default M1 state: a deterministic [`MockEngine`] (768-dim).
    pub fn default_mock() -> anyhow::Result<Self> {
        Self::from_engine(Arc::new(MockEngine::new()))
    }

    // -----------------------------------------------------------------------
    // Durable backend (Task 8, `durable` feature)
    // -----------------------------------------------------------------------

    /// Build an [`AppState`] backed by durable storage at `data_dir`.
    ///
    /// Both subsystems are opened over `data_dir`:
    ///
    /// - `KnowledgeStore::open_durable(data_dir, dim)` — WAL-replay, HNSW
    ///   load-or-rebuild, dedup-index reconstruction.
    /// - `SemanticCache::open_durable(data_dir, dim)` — snapshot + WAL replay.
    ///
    /// The augmenter is wired over the **same** `Arc<KnowledgeStore>` as in
    /// [`with_store`](AppState::with_store), so seeding facts through
    /// [`knowledge_store`](AppState::knowledge_store) is visible to GraphRAG.
    #[cfg(feature = "durable")]
    pub fn open_durable(
        engine: Arc<dyn InferenceEngine>,
        data_dir: &std::path::Path,
        config: OrchestratorConfig,
    ) -> anyhow::Result<Self> {
        let dim = engine.embedding_dim();

        let store = Arc::new(
            KnowledgeStore::open_durable(data_dir, dim)
                .map_err(|e| anyhow::anyhow!("KnowledgeStore::open_durable failed: {e}"))?,
        );

        let durable_cache = SemanticCache::open_durable(data_dir, dim)
            .map_err(|e| anyhow::anyhow!("SemanticCache::open_durable failed: {e}"))?;

        // Mirror the augmenter setup from `with_store`.
        let provider = Arc::new(EngineProvider::new(Arc::clone(&engine)));
        let context_budget = (engine.input_context_tokens() / 4).min(8_000);
        let rag_config = astraea_rag::GraphRagConfig {
            hops: 2,
            max_context_nodes: 20,
            text_format: astraea_rag::TextFormat::Structured,
            token_budget: context_budget,
            system_prompt: Some(
                "You are a helpful assistant with access to a knowledge graph \
                 of prior interactions and learned facts. Use the provided \
                 context to inform your answer."
                    .to_string(),
            ),
        };
        let augmenter = crate::augment::Augmenter::with_provider(
            Arc::clone(&store),
            provider,
            rag_config,
        );
        let model = engine.name().to_string();

        Ok(Self {
            inner: Arc::new(AppStateInner {
                engine,
                store,
                augmenter,
                cache: tokio::sync::RwLock::new(durable_cache),
                model,
                config,
            }),
        })
    }

    /// Flush all durable state to disk.
    ///
    /// - [`KnowledgeStore::persist`] — atomically saves the HNSW index snapshot
    ///   and flushes the buffer pool (WAL checkpoint).
    /// - [`SemanticCache::persist`] — atomically snapshots the cache store and
    ///   syncs the WAL.
    ///
    /// Safe to call from a SIGTERM / SIGINT handler. Idempotent: calling twice
    /// is safe (just writes the snapshot again).
    ///
    /// Returns `Err` only on I/O failure; use `eprintln` and continue if called
    /// from a signal context.
    #[cfg(feature = "durable")]
    pub async fn persist(&self) -> anyhow::Result<()> {
        // Persist the knowledge store (HNSW index + buffer-pool flush).
        self.inner
            .store
            .persist()
            .map_err(|e| anyhow::anyhow!("KnowledgeStore::persist failed: {e}"))?;

        // Persist the cache (snapshot + WAL sync).  Acquire a read lock because
        // `SemanticCache::persist` uses a Mutex internally for WAL sync —
        // no exclusive access to the cache is needed.
        let cache = self.inner.cache.read().await;
        cache
            .persist()
            .map_err(|e| anyhow::anyhow!("SemanticCache::persist failed: {e}"))?;

        Ok(())
    }

    /// The configured/default model name.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Shared handle to the orchestrator's long-term [`KnowledgeStore`].
    ///
    /// Returns a clone of the **same** `Arc` the GraphRAG [`Augmenter`]
    /// retrieves over, so facts/cache-entries upserted through it are
    /// immediately visible to [`AppState::generate`]'s augment step (it can then
    /// return `context_used = true`). This is the seam for seeding knowledge and
    /// for Task 7's durable flush-in.
    pub fn knowledge_store(&self) -> Arc<KnowledgeStore> {
        Arc::clone(&self.inner.store)
    }

    /// The learn → store → retrieve → augment loop (DESIGN.md §1.3).
    ///
    /// 1. **Embed** the prompt (768-dim, via the engine directly — async, no
    ///    sync bridge needed here).
    /// 2. **Retrieve (hot)** — probe the Eunomia [`SemanticCache`]. On a hit
    ///    above threshold, if `allow_cache && config.serve_cache_hits`,
    ///    short-circuit and serve the cached response.
    /// 3. **Augment (warm)** — on a miss, run [`Augmenter::augment_question`]
    ///    (GraphRAG vector-search → subgraph → linearize → LLM). If the graph
    ///    is empty (no anchor), fall back to a direct
    ///    [`InferenceEngine::complete`] with no context.
    /// 4. **Learn / store** — write the new `prompt → response` exchange back
    ///    into the cache with the configured TTL.
    ///
    /// 5. **Tier (durable)** — if `config.tier_on_generate`, opportunistically
    ///    flush any TTL-evicted interactions into the durable `KnowledgeStore`
    ///    via [`tier_evicted`](AppState::tier_evicted) so GraphRAG can surface
    ///    them after they leave the hot cache (Task 7). This never fails the
    ///    request; a tiering error is logged and swallowed.
    ///
    /// > Task 7 (Decision 2145) tiers **natively** into a-llama's own
    /// > `KnowledgeStore` — it does **not** use the `eunomia-astraea` bridge
    /// > (that bridge owns a separate `Graph` GraphRAG could not read).
    pub async fn generate(
        &self,
        prompt: &str,
        model: &str,
        allow_cache: bool,
    ) -> anyhow::Result<GenerateOutcome> {
        let inner = &self.inner;

        // 1. Embed the prompt for the cache key.
        let embedding = inner.engine.embed(prompt).await?;

        // 2. Retrieve (hot): semantic cache probe.
        if allow_cache && inner.config.serve_cache_hits {
            let hit = {
                let cache = inner.cache.read().await;
                cache.lookup(&embedding)
            };
            if let Some(hit) = hit {
                return Ok(GenerateOutcome {
                    response: hit.response,
                    model: hit.model,
                    from_cache: true,
                    context_used: false,
                });
            }
        }

        // 3. Augment (warm): GraphRAG over the learned graph. The augmenter runs
        //    the LLM internally with the assembled context. If the store is
        //    empty (no anchor node) augment_question errors — fall back to a
        //    plain completion with no context.
        let (response, context_used) = match inner.augmenter.augment_question(prompt).await {
            Ok(result) => {
                let used = !result.context_text.is_empty();
                (result.answer, used)
            }
            Err(_) => {
                let answer = inner.engine.complete(prompt, None).await?;
                (answer, false)
            }
        };

        // 4. Learn / store: persist the exchange in the hot cache.
        {
            let mut cache = inner.cache.write().await;
            // A cache-store failure must not fail the request — log and move on.
            if let Err(e) =
                cache.store_interaction(prompt, &response, model, &embedding, inner.config.ttl_secs)
            {
                eprintln!("a-llama: warning: failed to store interaction in cache: {e}");
            }
        }

        // 4b. Extract (durable, opt-in): distill structured Fact/Entity nodes
        //     from this fresh exchange into the long-term graph so individual
        //     facts become independently GraphRAG-retrievable (Task 9, Q6). This
        //     is an EXTRA LLM call, so it is gated behind `config.extract_facts`
        //     (default off). It runs only on the miss path (a cache hit returned
        //     earlier). `derived_from` is `None`: in M1's flow the `CacheEntry`
        //     for this interaction is minted later, when Eunomia tiering runs —
        //     the facts are still retrievable on their own embeddings; the
        //     provenance edge is simply added once a cache-entry id is known. An
        //     extraction error never fails the request (logged and swallowed).
        if inner.config.extract_facts {
            // Run extraction DETACHED. It is an extra LLM completion that can be
            // slow (a weak model may generate up to its output cap before
            // emitting a stop), and awaiting it here would block the HTTP
            // response for the full extraction. Spawn it so the response returns
            // immediately; the facts land in the shared `KnowledgeStore`
            // asynchronously and errors are logged, never surfaced to the client.
            let source_id = interaction_source_id(model, prompt);
            let extractor = Extractor::new(Arc::clone(&inner.engine), self.knowledge_store());
            let (p, r) = (prompt.to_string(), response.clone());
            tokio::spawn(async move {
                match extractor.extract_and_store(&p, &r, &source_id, None).await {
                    Ok(stats) => {
                        if stats.facts > 0 || stats.entities > 0 {
                            eprintln!(
                                "a-llama: extracted {} fact(s), {} entit(y/ies), {} relation(s)",
                                stats.facts, stats.entities, stats.relations
                            );
                        }
                    }
                    Err(e) => eprintln!("a-llama: warning: fact extraction failed: {e}"),
                }
            });
        }

        // 5. Tier (durable): opportunistically flush evicted interactions into
        //    the long-term graph. Cheap when nothing has expired; never fails
        //    the request.
        if inner.config.tier_on_generate {
            if let Err(e) = self.tier_evicted().await {
                eprintln!("a-llama: warning: opportunistic tiering failed: {e}");
            }
        }

        Ok(GenerateOutcome {
            response,
            model: model.to_string(),
            from_cache: false,
            context_used,
        })
    }

    /// Flush TTL-evicted hot-cache interactions into the durable
    /// [`KnowledgeStore`] as `CacheEntry` nodes (Task 7, Decision 2145 — native
    /// tiering, **not** the `eunomia-astraea` bridge).
    ///
    /// Sweeps the Eunomia hot cache ([`SemanticCache::sweep_expired`], which
    /// needs `&mut self` and so runs under the cache write lock) and, for each
    /// evicted [`eunomia_core::Entry`], upserts it into the orchestrator's own
    /// `KnowledgeStore` via
    /// [`upsert_cache_entry`](crate::knowledge_store::KnowledgeStore::upsert_cache_entry):
    ///
    /// - the entry's `key.id` → `eunomia_id` (dedup identity),
    /// - the entry's `key.embedding` (the 768-dim prompt embedding) → the node
    ///   embedding, so the tiered node is retrievable by GraphRAG vector search,
    /// - the entry's `value` JSON → the node `value`, **bounded** to
    ///   `config.max_tiered_value_bytes` (astraedb #26, see [`bound_tiered_value`]),
    /// - the entry's `metadata.tags` → the node `tags`.
    ///
    /// Because `upsert_cache_entry` dedups by `(namespace, eunomia_id)`, tiering
    /// the same entry twice updates in place rather than duplicating, so this is
    /// safe to call repeatedly (e.g. opportunistically from [`generate`]). An
    /// individual node-store failure is logged and skipped rather than aborting
    /// the whole sweep — the remaining entries are still tiered. Returns the
    /// number of entries successfully tiered.
    ///
    /// [`generate`]: AppState::generate
    pub async fn tier_evicted(&self) -> anyhow::Result<usize> {
        // Sweep under the write lock; capture the namespace from the same guard
        // so the durable node shares the hot entry's `(namespace, id)` identity.
        let (evicted, namespace) = {
            let mut cache = self.inner.cache.write().await;
            let namespace = cache.namespace();
            (cache.sweep_expired(), namespace)
        };
        if evicted.is_empty() {
            return Ok(0);
        }

        let store = self.knowledge_store();
        let now = unix_secs(SystemTime::now());
        let max_value_bytes = self.inner.config.max_tiered_value_bytes;

        let mut tiered = 0usize;
        for entry in evicted {
            // The prompt embedding is mandatory: without it the node could not
            // be indexed for GraphRAG retrieval. Every entry written by
            // `store_interaction` carries one; defensively skip any that don't.
            let embedding = match entry.key.embedding {
                Some(e) => e,
                None => {
                    eprintln!(
                        "a-llama: warning: skipping tiering of entry {} (no embedding)",
                        entry.key.id
                    );
                    continue;
                }
            };

            // astraedb #26: keep the node + embedding under the page limit by
            // bounding the value JSON (truncating response, then prompt).
            let value = bound_tiered_value(&entry.value, max_value_bytes);
            // Prefer the interaction's own `created_at`; fall back to metadata.
            let created_at = entry
                .value
                .get("created_at")
                .and_then(Value::as_i64)
                .unwrap_or_else(|| unix_secs(entry.metadata.created_at));

            match store.upsert_cache_entry(
                namespace,
                &entry.key.id,
                value,
                &entry.metadata.tags,
                embedding,
                created_at,
                now,
            ) {
                Ok(_) => tiered += 1,
                Err(e) => eprintln!(
                    "a-llama: warning: failed to tier evicted entry {}: {e}",
                    entry.key.id
                ),
            }
        }

        Ok(tiered)
    }

    /// Embed one or more texts via the engine (used by `/api/embeddings`).
    pub async fn embed_all(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.inner.engine.embed(t).await?);
        }
        Ok(out)
    }

    /// Resolve the request's model name, falling back to the configured default.
    fn resolve_model(&self, requested: &Option<String>) -> String {
        match requested {
            Some(m) if !m.is_empty() => m.clone(),
            _ => self.inner.model.clone(),
        }
    }
}

// ===========================================================================
// HTTP router
// ===========================================================================

/// Build the Ollama-compatible axum router over the given [`AppState`].
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/embeddings", post(embeddings))
        .route("/api/embed", post(embed))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// HTTP error returned as Ollama-shaped `{"error": "..."}` JSON.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

/// Lenient JSON body extractor. Unlike axum's [`Json`], it parses the body as
/// JSON **regardless of the `Content-Type` header**, matching Ollama's
/// leniency so plain `curl -d '{...}'` (which sends
/// `application/x-www-form-urlencoded`) works as a drop-in.
struct JsonBody<T>(T);

impl<T, S> FromRequest<S> for JsonBody<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
        // An empty body deserializes as `{}` so all-optional request structs
        // (which every a-llama request is) succeed with their defaults.
        let slice: &[u8] = if bytes.is_empty() { b"{}" } else { &bytes };
        let value = serde_json::from_slice(slice)
            .map_err(|e| ApiError(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")))?;
        Ok(JsonBody(value))
    }
}

// ---------------------------------------------------------------------------
// GET / and /api/version
// ---------------------------------------------------------------------------

/// `GET /` — Ollama returns the literal string "Ollama is running".
async fn root() -> &'static str {
    "Ollama is running"
}

/// `GET /api/version` — a-llama version banner.
async fn version() -> Json<Value> {
    Json(json!({ "version": format!("{}-a-llama", env!("CARGO_PKG_VERSION")) }))
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

/// `GET /api/tags` — list available models (Ollama "list local models" shape).
async fn tags(State(state): State<AppState>) -> Json<Value> {
    let name = state.model();
    // Ollama presents names as "name:tag"; advertise ":latest" if untagged.
    let tagged = if name.contains(':') {
        name.to_string()
    } else {
        format!("{name}:latest")
    };
    let created = rfc3339_now();
    Json(json!({
        "models": [{
            "name": tagged,
            "model": tagged,
            "modified_at": created,
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "a-llama",
                "families": ["a-llama"],
                "parameter_size": "",
                "quantization_level": ""
            }
        }]
    }))
}

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    /// Optional extra system context prepended to the prompt.
    #[serde(default)]
    system: Option<String>,
    /// Ollama defaults `stream` to `true`.
    #[serde(default)]
    stream: Option<bool>,
}

async fn generate(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<GenerateRequest>,
) -> Result<Response, ApiError> {
    let model = state.resolve_model(&req.model);
    let prompt = req.prompt.unwrap_or_default();
    // A `system` field is folded into the prompt for M1 (no separate role here).
    let effective_prompt = match &req.system {
        Some(s) if !s.is_empty() => format!("{s}\n\n{prompt}"),
        _ => prompt.clone(),
    };
    let stream = req.stream.unwrap_or(true);

    let outcome = state
        .generate(&effective_prompt, &model, true)
        .await
        .map_err(ApiError::from)?;

    if stream {
        Ok(ndjson_response(generate_ndjson_lines(&outcome.model, &outcome.response)))
    } else {
        Ok(Json(json!({
            "model": outcome.model,
            "created_at": rfc3339_now(),
            "response": outcome.response,
            "done": true,
            "done_reason": "stop"
        }))
        .into_response())
    }
}

// ---------------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: Option<bool>,
}

async fn chat(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<ChatRequest>,
) -> Result<Response, ApiError> {
    let model = state.resolve_model(&req.model);
    let stream = req.stream.unwrap_or(true);

    // M1: flatten the message list into a single prompt (role-prefixed lines),
    // ending with an "Assistant:" cue. Multi-turn role handling is M2.
    let prompt = flatten_messages(&req.messages);

    let outcome = state
        .generate(&prompt, &model, true)
        .await
        .map_err(ApiError::from)?;

    if stream {
        Ok(ndjson_response(chat_ndjson_lines(&outcome.model, &outcome.response)))
    } else {
        Ok(Json(json!({
            "model": outcome.model,
            "created_at": rfc3339_now(),
            "message": { "role": "assistant", "content": outcome.response },
            "done": true,
            "done_reason": "stop"
        }))
        .into_response())
    }
}

/// Flatten an Ollama chat message list into a single prompt string for M1.
fn flatten_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let label = match m.role.as_str() {
            "system" => "System",
            "assistant" => "Assistant",
            _ => "User",
        };
        out.push_str(label);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out.push_str("Assistant:");
    out
}

// ---------------------------------------------------------------------------
// POST /api/embeddings (legacy) and POST /api/embed (new)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbeddingsRequest {
    // `model` is accepted but ignored (serde drops unknown fields); a-llama
    // serves a single in-process embedding model.
    /// Legacy single-text field.
    #[serde(default)]
    prompt: Option<String>,
}

/// `POST /api/embeddings` — legacy single-vector shape: `{ "embedding": [..] }`.
async fn embeddings(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<EmbeddingsRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = req.prompt.unwrap_or_default();
    let vectors = state.embed_all(&[text]).await.map_err(ApiError::from)?;
    let embedding = vectors.into_iter().next().unwrap_or_default();
    Ok(Json(json!({ "embedding": embedding })))
}

#[derive(Deserialize)]
struct EmbedRequest {
    #[serde(default)]
    model: Option<String>,
    /// New-style input: a single string or an array of strings.
    #[serde(default)]
    input: Option<OneOrMany>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

/// `POST /api/embed` — new batch shape: `{ "model", "embeddings": [[..], ..] }`.
async fn embed(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<EmbedRequest>,
) -> Result<Json<Value>, ApiError> {
    let model = state.resolve_model(&req.model);
    let inputs: Vec<String> = match req.input {
        Some(OneOrMany::One(s)) => vec![s],
        Some(OneOrMany::Many(v)) => v,
        None => Vec::new(),
    };
    let embeddings = state.embed_all(&inputs).await.map_err(ApiError::from)?;
    Ok(Json(json!({
        "model": model,
        "embeddings": embeddings
    })))
}

// ===========================================================================
// NDJSON streaming helpers
// ===========================================================================

/// Build an `application/x-ndjson` streaming [`Response`] from pre-rendered
/// JSON lines (each already terminated with `\n`).
fn ndjson_response(lines: Vec<String>) -> Response {
    let stream = futures::stream::iter(
        lines
            .into_iter()
            .map(|l| Ok::<Bytes, std::convert::Infallible>(Bytes::from(l))),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(stream))
        .expect("static ndjson response is always valid")
}

/// NDJSON lines for `/api/generate` streaming: one object per token-ish chunk
/// (`done:false`), then a terminal `done:true` object.
fn generate_ndjson_lines(model: &str, response: &str) -> Vec<String> {
    let created = rfc3339_now();
    let mut lines = Vec::new();
    for chunk in chunk_response(response) {
        lines.push(
            json!({
                "model": model,
                "created_at": created,
                "response": chunk,
                "done": false
            })
            .to_string()
                + "\n",
        );
    }
    lines.push(
        json!({
            "model": model,
            "created_at": created,
            "response": "",
            "done": true,
            "done_reason": "stop"
        })
        .to_string()
            + "\n",
    );
    lines
}

/// NDJSON lines for `/api/chat` streaming: one `message` delta per chunk, then a
/// terminal `done:true` object with an empty content delta.
fn chat_ndjson_lines(model: &str, response: &str) -> Vec<String> {
    let created = rfc3339_now();
    let mut lines = Vec::new();
    for chunk in chunk_response(response) {
        lines.push(
            json!({
                "model": model,
                "created_at": created,
                "message": { "role": "assistant", "content": chunk },
                "done": false
            })
            .to_string()
                + "\n",
        );
    }
    lines.push(
        json!({
            "model": model,
            "created_at": created,
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": "stop"
        })
        .to_string()
            + "\n",
    );
    lines
}

/// Split a response into word-ish chunks for simulated token streaming. The
/// mock engine returns the whole string at once; splitting on inclusive spaces
/// gives a realistic multi-chunk NDJSON stream while preserving the exact text
/// when concatenated.
fn chunk_response(response: &str) -> Vec<&str> {
    if response.is_empty() {
        return Vec::new();
    }
    response.split_inclusive(' ').collect()
}

// ===========================================================================
// Tiering helpers (Task 7, astraedb #26)
// ===========================================================================

/// Stable `source_id` for the facts extracted from an interaction.
///
/// Mirrors the Eunomia hot cache's `key.id` (DESIGN §4.1: `interaction:<model>:<hash>`,
/// FNV-1a over `model` + the *normalized* prompt) so an extracted `Fact`'s
/// `source_id` equals the `eunomia_id` of the `CacheEntry` the same interaction
/// is later tiered into. That alignment lets provenance be reconstructed even
/// though `generate` passes `derived_from = None` at extraction time (the
/// `CacheEntry` node does not exist yet). The hash function is duplicated here
/// (the cache's is private) and must stay in sync with `cache::stable_interaction_id`.
fn interaction_source_id(model: &str, prompt: &str) -> String {
    let normalized = prompt
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model
        .as_bytes()
        .iter()
        .chain(std::iter::once(&0u8))
        .chain(normalized.as_bytes())
    {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("interaction:{model}:{hash:016x}")
}

/// Unix epoch seconds for `t` (0 for a pre-epoch clock — pathological only).
fn unix_secs(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Bound a Eunomia interaction `value` JSON so the durable `CacheEntry` node
/// (value **+** its 768-dim embedding) stays under astraedb's ~8 KB single-page
/// limit (astraedb #26).
///
/// The interaction value is `{prompt, response, model, created_at}`. The
/// `model` and `created_at` fields are tiny and always kept verbatim. If the
/// serialized value already fits `max_bytes` it is returned unchanged.
/// Otherwise the `response` is truncated first (it is usually the largest and
/// least key-bearing field), then the `prompt` if the value is still over
/// budget, and a `"truncated": true` marker is added. The returned value is
/// guaranteed to serialize to `<= max_bytes` bytes (down to a minimal
/// `{prompt:"", response:"", …}` skeleton in the worst case), so the node can
/// always be stored together with its embedding.
fn bound_tiered_value(value: &Value, max_bytes: usize) -> Value {
    let prompt = value.get("prompt").and_then(Value::as_str).unwrap_or("");
    let response = value.get("response").and_then(Value::as_str).unwrap_or("");
    let model = value.get("model").and_then(Value::as_str).unwrap_or("");
    let created_at = value.get("created_at").and_then(Value::as_i64).unwrap_or(0);

    let fits = |v: &Value| serde_json::to_vec(v).map_or(false, |b| b.len() <= max_bytes);

    // Fast path: the untouched value already fits — keep it verbatim (no marker).
    if fits(value) {
        return value.clone();
    }

    // Shrink the response first, then the prompt, until the marked value fits.
    let mut prompt_len = prompt.len();
    let mut response_len = response.len();
    loop {
        let candidate = json!({
            "prompt": truncate_to_bytes(prompt, prompt_len),
            "response": truncate_to_bytes(response, response_len),
            "model": model,
            "created_at": created_at,
            "truncated": true,
        });
        if fits(&candidate) {
            return candidate;
        }
        // Trim ~12.5 % (min 64 bytes) off the response, then the prompt. The
        // floor on `model`/`created_at`/key overhead is small, so this always
        // converges to an empty-text skeleton that fits.
        if response_len > 0 {
            response_len = response_len.saturating_sub((response_len / 8).max(64));
        } else if prompt_len > 0 {
            prompt_len = prompt_len.saturating_sub((prompt_len / 8).max(64));
        } else {
            // Both texts empty and still over budget (absurdly small max_bytes):
            // return the minimal skeleton regardless.
            return json!({
                "prompt": "",
                "response": "",
                "model": model,
                "created_at": created_at,
                "truncated": true,
            });
        }
    }
}

/// Truncate `s` to at most `max` bytes, snapping back to the nearest UTF-8
/// character boundary so the result is always valid UTF-8.
fn truncate_to_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ===========================================================================
// Timestamp helper (RFC 3339, no chrono dependency)
// ===========================================================================

/// Current UTC time as an RFC-3339 string (e.g. `2026-06-16T12:34:56.000000000Z`).
fn rfc3339_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let nanos = now.subsec_nanos();
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}.{nanos:09}Z")
}

/// Convert days since the Unix epoch to a `(year, month, day)` civil date
/// (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::EMBEDDING_DIM;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn test_state() -> AppState {
        AppState::default_mock().expect("mock app state")
    }

    fn post_json(uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_json(resp: Response) -> Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn body_string(resp: Response) -> String {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    // --- Acceptance: /api/generate round-trips Ollama-shaped JSON ---

    #[tokio::test(flavor = "multi_thread")]
    async fn generate_non_stream_returns_ollama_shape() {
        let app = router(test_state());
        let resp = app
            .oneshot(post_json(
                "/api/generate",
                json!({ "model": "mock-gguf", "prompt": "What is Rust?", "stream": false }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let v = body_json(resp).await;
        assert_eq!(v["model"], "mock-gguf");
        assert_eq!(v["done"], true);
        assert!(v["response"].as_str().unwrap().contains("What is Rust?"));
        assert!(v["created_at"].as_str().unwrap().ends_with('Z'));
    }

    // --- Acceptance: /api/chat round-trips Ollama-shaped JSON ---

    #[tokio::test(flavor = "multi_thread")]
    async fn chat_non_stream_returns_message_shape() {
        let app = router(test_state());
        let resp = app
            .oneshot(post_json(
                "/api/chat",
                json!({
                    "model": "mock-gguf",
                    "messages": [
                        { "role": "system", "content": "Be brief." },
                        { "role": "user", "content": "Hello there" }
                    ],
                    "stream": false
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let v = body_json(resp).await;
        assert_eq!(v["done"], true);
        assert_eq!(v["message"]["role"], "assistant");
        let content = v["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty(), "assistant content must be non-empty");
        // The flattened prompt carried the user's text through to the mock.
        assert!(content.contains("Hello there"));
    }

    // --- Acceptance: /api/tags lists the model ---

    #[tokio::test(flavor = "multi_thread")]
    async fn tags_lists_configured_model() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let v = body_json(resp).await;
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "mock-gguf:latest");
    }

    // --- Embeddings: legacy + new shapes return 768-dim vectors ---

    #[tokio::test(flavor = "multi_thread")]
    async fn embeddings_legacy_returns_768_dim() {
        let app = router(test_state());
        let resp = app
            .oneshot(post_json(
                "/api/embeddings",
                json!({ "model": "mock-gguf", "prompt": "embed me" }),
            ))
            .await
            .unwrap();
        let v = body_json(resp).await;
        assert_eq!(v["embedding"].as_array().unwrap().len(), EMBEDDING_DIM);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_new_batch_shape() {
        let app = router(test_state());
        let resp = app
            .oneshot(post_json(
                "/api/embed",
                json!({ "model": "mock-gguf", "input": ["a", "b"] }),
            ))
            .await
            .unwrap();
        let v = body_json(resp).await;
        let embs = v["embeddings"].as_array().unwrap();
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].as_array().unwrap().len(), EMBEDDING_DIM);
    }

    // --- Streaming: NDJSON ends with done:true ---

    #[tokio::test(flavor = "multi_thread")]
    async fn generate_streaming_emits_ndjson() {
        let app = router(test_state());
        let resp = app
            .oneshot(post_json(
                "/api/generate",
                json!({ "model": "mock-gguf", "prompt": "stream this please", "stream": true }),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "application/x-ndjson"
        );

        let text = body_string(resp).await;
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert!(lines.len() >= 2, "streaming should emit multiple lines");

        // Reassembling the `response` deltas reproduces the full text.
        let mut reassembled = String::new();
        for (i, line) in lines.iter().enumerate() {
            let obj: Value = serde_json::from_str(line).unwrap();
            reassembled.push_str(obj["response"].as_str().unwrap_or(""));
            let is_last = i == lines.len() - 1;
            assert_eq!(obj["done"], is_last, "only the final line has done:true");
        }
        assert!(reassembled.contains("stream this please"));
    }

    // --- Bonus: the continual-learning loop hits the cache on repeat ---

    #[tokio::test(flavor = "multi_thread")]
    async fn repeated_prompt_hits_cache() {
        let state = test_state();
        let prompt = "What is the capital of France?";

        let first = state.generate(prompt, "mock-gguf", true).await.unwrap();
        assert!(!first.from_cache, "first request is a cache miss");

        // A repeated request with the same prompt embeds to the same vector
        // (MockEngine embeddings are deterministic per exact text), so it scores
        // ~1.0 against the stored interaction and is served from the cache —
        // the continual-learning loop closing on itself.
        let second = state.generate(prompt, "mock-gguf", true).await.unwrap();
        assert!(second.from_cache, "repeated prompt must hit the cache");
        assert_eq!(second.response, first.response);
    }

    // --- Issue 2048: seeding the shared store closes the augment loop ---

    #[tokio::test(flavor = "multi_thread")]
    async fn seeded_knowledge_store_makes_generate_use_context() {
        let state = test_state();

        // Before seeding, the store is empty: the augment step finds no anchor
        // and falls back to a plain completion with no context.
        let cold = state
            .generate("Tell me about Rust.", "mock-gguf", false)
            .await
            .unwrap();
        assert!(
            !cold.context_used,
            "with an empty store there is no GraphRAG context"
        );

        // Seed a fact through the orchestrator's OWN store handle. Because the
        // augmenter retrieves over the same Arc, this is immediately visible.
        let store = state.knowledge_store();
        let engine = MockEngine::new();
        let fact = "Rust is a memory-safe systems programming language.";
        let emb = engine.embed(fact).await.unwrap();
        store.upsert_fact(fact, "seed-0", 0.95, emb, 0).unwrap();

        // Now the augment step finds the seeded fact and assembles context.
        // `allow_cache=false` keeps this exercising the augment path (not a hit
        // on the interaction stored by the cold call).
        let warm = state
            .generate("What can you tell me about the Rust language?", "mock-gguf", false)
            .await
            .unwrap();
        assert!(
            warm.context_used,
            "a seeded fact must make GraphRAG context_used = true"
        );
        assert!(!warm.from_cache, "this path is the augment path, not a cache hit");
    }

    // --- Task 7: durable tiering / long-term memory ---

    /// Acceptance (DESIGN task 7): the full durable-memory loop. An interaction
    /// stored with a short TTL is expired, `tier_evicted` flushes it into the
    /// `KnowledgeStore`, and afterwards it is (a) gone from the hot cache but
    /// (b) present as a `CacheEntry` node AND retrievable via GraphRAG by its
    /// embedding (the augmenter reads the same store).
    #[tokio::test(flavor = "multi_thread")]
    async fn tier_evicted_persists_and_is_retrievable_via_graphrag() {
        use std::time::Duration;

        let state = test_state();
        let engine = MockEngine::new();
        let prompt = "What is the speed of light in a vacuum?";
        let response = "About 299,792 kilometres per second.";
        let emb = engine.embed(prompt).await.unwrap();

        // Store the interaction directly (deterministic timing) with a 1s TTL
        // created 60s in the past so it is already expired.
        {
            let mut cache = state.inner.cache.write().await;
            let created = SystemTime::now() - Duration::from_secs(60);
            cache
                .store_interaction_at(prompt, response, "mock-gguf", &emb, Some(1), created)
                .unwrap();
            assert_eq!(cache.len(), 1, "interaction is in the hot cache");
        }

        // Tier the evicted interaction into the durable graph.
        let n = state.tier_evicted().await.unwrap();
        assert_eq!(n, 1, "exactly one expired interaction is tiered");

        // (a) Gone from the hot cache.
        assert_eq!(
            state.inner.cache.read().await.len(),
            0,
            "the tiered interaction is evicted from the hot cache"
        );

        // (b) Present in the KnowledgeStore as a CacheEntry node.
        let store = state.knowledge_store();
        let nodes = store.graph_ops().find_by_label("CacheEntry").unwrap();
        assert_eq!(nodes.len(), 1, "one durable CacheEntry node exists");
        let node = store.graph_ops().get_node(nodes[0]).unwrap().unwrap();
        assert_eq!(node.properties["value"]["response"], response);
        assert!(node.embedding.is_some(), "the prompt embedding is preserved");

        // (b) Retrievable via the Augmenter/GraphRAG by its embedding:
        // re-asking the same question (cache bypassed) finds the tiered node and
        // assembles context from it.
        let warm = state.generate(prompt, "mock-gguf", false).await.unwrap();
        assert!(
            warm.context_used,
            "the tiered interaction must be retrievable via GraphRAG"
        );
        assert!(!warm.from_cache, "this is the augment path, not a hot-cache hit");
    }

    /// Acceptance (DESIGN task 7, astraedb #26): a large prompt+response is
    /// tiered successfully — the value is bounded under `max_tiered_value_bytes`
    /// and marked `truncated`, the embedding is kept, and the node stores with
    /// no `record too large` error.
    #[tokio::test(flavor = "multi_thread")]
    async fn tier_evicted_bounds_oversized_value() {
        use std::time::Duration;

        let state = test_state();
        let engine = MockEngine::new();
        // Far larger than the 4096-byte default value budget.
        let prompt = "Q ".repeat(8_000);
        let response = "A ".repeat(40_000);
        let emb = engine.embed(&prompt).await.unwrap();

        {
            let mut cache = state.inner.cache.write().await;
            let created = SystemTime::now() - Duration::from_secs(60);
            cache
                .store_interaction_at(&prompt, &response, "mock-gguf", &emb, Some(1), created)
                .unwrap();
        }

        // Tiering must succeed (no `record too large` error) despite the size.
        let n = state.tier_evicted().await.unwrap();
        assert_eq!(n, 1, "the oversized interaction is tiered successfully");

        let store = state.knowledge_store();
        let nodes = store.graph_ops().find_by_label("CacheEntry").unwrap();
        assert_eq!(nodes.len(), 1);
        let node = store.graph_ops().get_node(nodes[0]).unwrap().unwrap();

        // Value is bounded to the configured budget and marked truncated.
        let value = &node.properties["value"];
        assert_eq!(value["truncated"], true, "oversized value must be marked truncated");
        let serialized_len = serde_json::to_vec(value).unwrap().len();
        assert!(
            serialized_len <= state.inner.config.max_tiered_value_bytes,
            "bounded value ({serialized_len} B) must fit max_tiered_value_bytes ({} B)",
            state.inner.config.max_tiered_value_bytes
        );

        // The embedding is always preserved so the node stays GraphRAG-retrievable.
        assert!(node.embedding.is_some(), "embedding is kept even when value is truncated");
    }

    /// `tier_evicted` is idempotent: re-tiering the same entry updates in place
    /// (the `upsert_cache_entry` dedup) rather than duplicating the node.
    #[tokio::test(flavor = "multi_thread")]
    async fn tier_evicted_is_idempotent() {
        use std::time::Duration;

        let state = test_state();
        let engine = MockEngine::new();
        let prompt = "Idempotent tiering check";
        let emb = engine.embed(prompt).await.unwrap();

        // Store an already-expired interaction, then tier it.
        {
            let mut cache = state.inner.cache.write().await;
            let created = SystemTime::now() - Duration::from_secs(60);
            cache
                .store_interaction_at(prompt, "v1", "mock-gguf", &emb, Some(1), created)
                .unwrap();
        }
        assert_eq!(state.tier_evicted().await.unwrap(), 1);

        // Re-store the same (model, prompt) and tier again.
        {
            let mut cache = state.inner.cache.write().await;
            let created = SystemTime::now() - Duration::from_secs(60);
            cache
                .store_interaction_at(prompt, "v1", "mock-gguf", &emb, Some(1), created)
                .unwrap();
        }
        assert_eq!(state.tier_evicted().await.unwrap(), 1);

        let store = state.knowledge_store();
        let nodes = store.graph_ops().find_by_label("CacheEntry").unwrap();
        assert_eq!(nodes.len(), 1, "re-tiering must update in place, not duplicate");
    }

    /// An empty sweep tiers nothing and returns 0 (the cheap common path).
    #[tokio::test(flavor = "multi_thread")]
    async fn tier_evicted_noop_when_nothing_expired() {
        let state = test_state();
        assert_eq!(state.tier_evicted().await.unwrap(), 0);
    }

    // --- Task 9: fact-extraction wiring (opt-in) ---

    /// `extract_facts` defaults off, so a normal `generate` mints no `Fact` nodes.
    #[tokio::test(flavor = "multi_thread")]
    async fn extract_facts_off_by_default_creates_no_facts() {
        let state = test_state();
        assert!(!state.inner.config.extract_facts, "extraction is opt-in");
        let _ = state.generate("Teach me something.", "mock-gguf", false).await.unwrap();
        let store = state.knowledge_store();
        assert_eq!(store.graph_ops().find_by_label("Fact").unwrap().len(), 0);
    }

    /// With `extract_facts` enabled over the `MockEngine` (non-JSON completions),
    /// the miss path runs extraction as a graceful no-op: the request still
    /// succeeds and no malformed `Fact` nodes are minted. (A real chat model that
    /// emits JSON is what produces facts; the wiring is what is verified here.)
    #[tokio::test(flavor = "multi_thread")]
    async fn extract_facts_enabled_runs_without_faulting_request() {
        let mut config = OrchestratorConfig::default();
        config.extract_facts = true;
        let state =
            AppState::with_config(Arc::new(MockEngine::new()), config).expect("state");

        let outcome = state
            .generate("What is the boiling point of water?", "mock-gguf", false)
            .await
            .unwrap();
        assert!(!outcome.from_cache, "miss path");

        // MockEngine emits no JSON → parse yields nothing → no Fact nodes, but
        // the request itself succeeded.
        let store = state.knowledge_store();
        assert_eq!(
            store.graph_ops().find_by_label("Fact").unwrap().len(),
            0,
            "non-JSON extraction output stores no facts"
        );
    }

    /// The extraction `source_id` matches the eunomia entry id the same
    /// interaction would tier into (provenance alignment).
    #[test]
    fn interaction_source_id_is_stable_and_normalized() {
        let a = interaction_source_id("m", "Hello   World");
        let b = interaction_source_id("m", "hello world");
        assert_eq!(a, b, "normalized prompt yields a stable id");
        assert!(a.starts_with("interaction:m:"));
    }

    // --- root + version smoke ---

    #[tokio::test(flavor = "multi_thread")]
    async fn root_reports_running() {
        let app = router(test_state());
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(body_string(resp).await, "Ollama is running");
    }
}
