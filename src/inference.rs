//! Task 3 — local LLM inference engine + provider adapters.
//!
//! This module defines the swappable [`InferenceEngine`] trait (Decision 7659),
//! an always-compiled [`MockEngine`] for fast default builds and tests, a
//! feature-gated [`MistralRsEngine`] real backend (feature `mistralrs-engine`),
//! and the [`EngineProvider`] adapter that exposes any engine as
//! [`astraea_rag::LlmProvider`] + [`astraea_rag::EmbeddingProvider`] so it plugs
//! straight into the GraphRAG pipeline (`graph_rag_query`).
//!
//! ## The async → sync seam
//!
//! [`InferenceEngine`] is **async** (mistral.rs is async; so is any real GGUF
//! backend). astraea-rag's `LlmProvider`/`EmbeddingProvider` traits are
//! **synchronous** (see `astraea-rag/src/llm.rs`). [`EngineProvider`] therefore
//! bridges by blocking on the engine future. See [`block_on_engine`] for the
//! exact rules — in short, the orchestrator (Task 4+) must drive GraphRAG from a
//! **multi-threaded** Tokio runtime (or via `spawn_blocking`), because the
//! adapter uses `block_in_place` when already inside a runtime.

use std::sync::Arc;

use astraea_rag::embedding::EmbeddingProvider;
use astraea_rag::llm::LlmProvider;
use astraea_core::error::{AstraeaError, Result as AstraeaResult};
use async_trait::async_trait;

/// Embedding dimension pinned across a-llama (Decision 7660: `embeddinggemma`,
/// 768-dim). Eunomia's hot index, the graph's `HnswVectorIndex`, and every
/// `EmbeddingProvider` must agree on this value.
pub const EMBEDDING_DIM: usize = 768;

// ---------------------------------------------------------------------------
// InferenceEngine trait — the swappable seam (Decision 7659)
// ---------------------------------------------------------------------------

/// A local, in-process LLM inference backend.
///
/// This is the swappable seam: `MockEngine` for tests/default builds,
/// `MistralRsEngine` for the real GGUF backend, llama-cpp-2 as a future
/// feature-gated fallback. It is shaped to mirror `astraea_rag::LlmProvider`
/// (node 6789) and `astraea_rag::EmbeddingProvider` (node 6752), but **async**
/// because real engines (mistral.rs) are async. [`EngineProvider`] adapts it
/// back to those synchronous traits for GraphRAG.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate a completion for `prompt`, optionally prepended with `ctx`
    /// (the GraphRAG-assembled system context). Returns the generated text.
    async fn complete(&self, prompt: &str, ctx: Option<&str>) -> anyhow::Result<String>;

    /// Embed a single text into a fixed-dimension vector. The length must equal
    /// [`InferenceEngine::embedding_dim`] (768 for embeddinggemma).
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Dimension of vectors returned by [`embed`](InferenceEngine::embed).
    /// Defaults to [`EMBEDDING_DIM`].
    fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Maximum tokens the engine emits per completion (output cap). Mirrors
    /// `LlmProvider::max_output_tokens`.
    fn max_output_tokens(&self) -> usize {
        4096
    }

    /// Maximum tokens the model accepts in a single prompt (input window).
    /// Mirrors `LlmProvider::input_context_tokens`; used to size the retrieval
    /// budget in `GraphRagConfig`.
    fn input_context_tokens(&self) -> usize {
        32_000
    }

    /// Engine name for logging (e.g. `"mock"`, `"mistralrs"`).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// MockEngine — always compiled, deterministic, no heavy deps
// ---------------------------------------------------------------------------

/// A deterministic, dependency-free [`InferenceEngine`] for tests and the fast
/// default build. `complete` returns a canned string echoing the prompt;
/// `embed` returns a deterministic, L2-normalized 768-dim pseudo-embedding
/// seeded by a stable hash of the input text (so equal texts embed equally and
/// different texts get different — but reproducible — vectors).
#[derive(Debug, Clone)]
pub struct MockEngine {
    /// Reported model name.
    pub model: String,
    /// Dimension of the pseudo-embeddings produced.
    pub dim: usize,
}

impl Default for MockEngine {
    fn default() -> Self {
        Self {
            model: "mock-gguf".to_string(),
            dim: EMBEDDING_DIM,
        }
    }
}

impl MockEngine {
    /// A `MockEngine` reporting the standard 768-dim embedding size.
    pub fn new() -> Self {
        Self::default()
    }

    /// A `MockEngine` with a custom embedding dimension (for tests that need a
    /// non-768 size).
    pub fn with_dim(dim: usize) -> Self {
        Self {
            model: "mock-gguf".to_string(),
            dim,
        }
    }

    /// Deterministic pseudo-embedding: FNV-1a hash → xorshift PRNG → fill → L2
    /// normalize. Stable across runs and processes (no random seeds).
    fn pseudo_embedding(&self, text: &str) -> Vec<f32> {
        // FNV-1a 64-bit hash of the text — deterministic, no external deps.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        // xorshift64 PRNG seeded by the hash (avoid a zero seed).
        let mut state = hash | 1;
        let mut out = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1.0, 1.0).
            let v = (state >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            out.push((v * 2.0 - 1.0) as f32);
        }
        // L2-normalize so vectors behave under cosine/dot similarity.
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in out.iter_mut() {
                *x /= norm;
            }
        }
        out
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
    async fn complete(&self, prompt: &str, ctx: Option<&str>) -> anyhow::Result<String> {
        let ctx_note = match ctx {
            Some(c) if !c.is_empty() => format!(" (ctx: {} chars)", c.len()),
            _ => String::new(),
        };
        Ok(format!("[{}] mock response to: {prompt}{ctx_note}", self.model))
    }

    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.pseudo_embedding(text))
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// MistralRsEngine — real GGUF backend (feature `mistralrs-engine`, Decision 7659)
// ---------------------------------------------------------------------------
//
// Verified symbol-by-symbol against the mistral.rs 0.8.1 crate source
// (`mistralrs-0.8.1` + `mistralrs-core-0.8.1`). Entry points:
//   * Chat:      `GgufModelBuilder::new(model_id, files: Vec<impl ToString>)
//                 .with_chat_template(impl ToString).build().await -> anyhow::Result<Model>`,
//                 then `model.send_chat_request(impl RequestLike).await
//                 -> mistralrs::error::Result<ChatCompletionResponse>`, reading
//                 `response.choices: Vec<Choice>`, `Choice.message: ResponseMessage`,
//                 `ResponseMessage.content: Option<String>`. `TextMessages::new()
//                 .add_message(TextMessageRole, impl ToString)` implements `RequestLike`.
//   * Embedding: `EmbeddingModelBuilder::new(impl ToString).build().await
//                 -> anyhow::Result<Model>`, then `model.generate_embeddings(
//                 EmbeddingRequestBuilder).await -> mistralrs::error::Result<Vec<Vec<f32>>>`
//                 (one 1-D vector per input, already pooled + projected).
//                 `EmbeddingRequest::builder().add_prompt(impl Into<String>)`
//                 yields the `EmbeddingRequestBuilder` (passed UNBUILT — the SDK
//                 calls `.build()` internally).
//
// `mistralrs::error::Error` is a `thiserror`-derived enum (Send + Sync +
// std::error::Error), so `?` on these calls coerces cleanly into `anyhow::Error`.
//
// 768-dim embeddinggemma is a first-class embedder here: mistralrs-core ships
// `embedding_models/embedding_gemma.rs` and the embedding loader auto-detects it
// from the HF config `architectures: ["Gemma3TextModel"]` (also `FromStr`
// "embeddinggemma"). `google/embeddinggemma-300m` has `hidden_size = 768`, and
// `load()` probes the real output dim and hard-fails if it isn't 768.
//
// NOTE: this backend's full build is heavy and downloads model weights at
// runtime; it is NOT compiled by the default build or exercised by tests. The
// code matches the 0.8.1 API but a live end-to-end run is done by the top-level
// harness, not here.

/// Real in-process GGUF inference backend built on mistral.rs (feature
/// `mistralrs-engine`). Holds a chat `Model` and an embeddinggemma embedding
/// `Model`.
#[cfg(feature = "mistralrs-engine")]
pub struct MistralRsEngine {
    chat: mistralrs::Model,
    embed: mistralrs::Model,
    embedding_dim: usize,
    max_output_tokens: usize,
    input_context_tokens: usize,
}

#[cfg(feature = "mistralrs-engine")]
impl MistralRsEngine {
    /// Load a local GGUF chat model and the embeddinggemma embedding model.
    ///
    /// * `gguf_dir` — directory containing the GGUF file(s).
    /// * `gguf_files` — the GGUF filename(s) within `gguf_dir`.
    /// * `chat_template` — optional path to a chat-template JSON (mistral.rs
    ///   needs one for GGUF models that don't embed a template).
    /// * `embedding_model_id` — HF id of the embedding model; defaults to
    ///   `"google/embeddinggemma-300m"` (768-dim, Decision 7660) when `None`.
    pub async fn load(
        gguf_dir: impl Into<String>,
        gguf_files: Vec<String>,
        chat_template: Option<String>,
        embedding_model_id: Option<String>,
    ) -> anyhow::Result<Self> {
        use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, GgufModelBuilder, ModelDType};

        let mut chat_builder = GgufModelBuilder::new(gguf_dir.into(), gguf_files).with_logging();
        if let Some(tpl) = chat_template {
            chat_builder = chat_builder.with_chat_template(tpl);
        }
        let chat = chat_builder.build().await?;

        let embed_id =
            embedding_model_id.unwrap_or_else(|| "google/embeddinggemma-300m".to_string());
        // Force F32 for the embedder. embeddinggemma's forward pass (RMSNorm,
        // softcapping, pooling) produces a degenerate all-zero vector under the
        // default F16 on CPU; F32 yields correct, non-zero embeddings. The chat
        // model is unaffected and keeps its default dtype.
        let embed = EmbeddingModelBuilder::new(embed_id.clone())
            .with_dtype(ModelDType::F32)
            .with_logging()
            .build()
            .await?;

        // Probe the real embedding dimension and hard-fail if it isn't 768
        // (Decision 7660). embeddinggemma-300m is 768-dim, but this catches a
        // wrong/incompatible embedder at load time — before any mismatched
        // vector can corrupt the HNSW index, Eunomia's cache, or GraphRAG.
        let probe_vec = embed
            .generate_embeddings(EmbeddingRequest::builder().add_prompt("dimension probe"))
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                anyhow::anyhow!("embedding model `{embed_id}` returned no vector on probe")
            })?;
        let embedding_dim = probe_vec.len();
        if embedding_dim != EMBEDDING_DIM {
            anyhow::bail!(
                "embedding model `{embed_id}` produced {embedding_dim}-dim vectors, but a-llama \
                 is pinned to {EMBEDDING_DIM}-dim (Decision 7660). Set A_LLAMA_EMBED_MODEL to a \
                 768-dim embedder (e.g. the default google/embeddinggemma-300m)."
            );
        }
        // Reject a degenerate (all-zero / near-zero norm) embedding — a correct
        // dim with a zero vector still breaks cosine similarity (every prompt
        // looks identical, so the semantic cache false-hits and GraphRAG can't
        // discriminate). This catches dtype/precision regressions at load time.
        let probe_norm: f32 = probe_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !probe_norm.is_finite() || probe_norm < 1e-6 {
            anyhow::bail!(
                "embedding model `{embed_id}` produced a degenerate embedding (L2 norm \
                 {probe_norm:.3e}) — likely a precision/dtype issue in the embedding backend. \
                 Embeddings must be non-zero for the cache and GraphRAG to work."
            );
        }

        Ok(Self {
            chat,
            embed,
            embedding_dim,
            // Conservative defaults; TODO: derive from the loaded model's config.
            max_output_tokens: 4096,
            input_context_tokens: 32_000,
        })
    }
}

#[cfg(feature = "mistralrs-engine")]
#[async_trait]
impl InferenceEngine for MistralRsEngine {
    async fn complete(&self, prompt: &str, ctx: Option<&str>) -> anyhow::Result<String> {
        use mistralrs::{TextMessageRole, TextMessages};

        // GraphRAG context is injected as a system message when present.
        let mut messages = TextMessages::new();
        if let Some(c) = ctx {
            if !c.is_empty() {
                messages = messages.add_message(TextMessageRole::System, c);
            }
        }
        messages = messages.add_message(TextMessageRole::User, prompt);

        let response = self.chat.send_chat_request(messages).await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("mistralrs chat response had no content"))?;
        Ok(text)
    }

    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use mistralrs::EmbeddingRequest;

        let vectors = self
            .embed
            .generate_embeddings(EmbeddingRequest::builder().add_prompt(text))
            .await?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("mistralrs returned no embedding for input"))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    fn input_context_tokens(&self) -> usize {
        self.input_context_tokens
    }

    fn name(&self) -> &str {
        "mistralrs"
    }
}

// ---------------------------------------------------------------------------
// EngineProvider — adapts an InferenceEngine to astraea_rag's sync traits
// ---------------------------------------------------------------------------

/// Adapts any [`InferenceEngine`] to astraea-rag's synchronous
/// [`LlmProvider`] and [`EmbeddingProvider`] traits so it can be passed to
/// `graph_rag_query`. A single value implements **both** traits, so the same
/// `EngineProvider` can serve as the LLM and the embedder.
///
/// Generic over `?Sized` so it wraps both a concrete engine
/// (`EngineProvider<MockEngine>`) and a trait object
/// (`EngineProvider<dyn InferenceEngine>`).
#[derive(Clone)]
pub struct EngineProvider<E: ?Sized> {
    engine: Arc<E>,
}

impl<E: InferenceEngine + ?Sized> EngineProvider<E> {
    /// Wrap a shared engine.
    pub fn new(engine: Arc<E>) -> Self {
        Self { engine }
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &Arc<E> {
        &self.engine
    }
}

impl<E: InferenceEngine> From<E> for EngineProvider<E> {
    fn from(engine: E) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }
}

impl<E: InferenceEngine + ?Sized> LlmProvider for EngineProvider<E> {
    fn complete(&self, prompt: &str, context: &str) -> AstraeaResult<String> {
        let ctx = if context.is_empty() {
            None
        } else {
            Some(context)
        };
        block_on_engine(self.engine.complete(prompt, ctx))
            .map_err(|e| AstraeaError::QueryExecution(format!("inference engine error: {e}")))
    }

    fn max_output_tokens(&self) -> usize {
        self.engine.max_output_tokens()
    }

    fn input_context_tokens(&self) -> usize {
        self.engine.input_context_tokens()
    }

    fn name(&self) -> &str {
        self.engine.name()
    }
}

impl<E: InferenceEngine + ?Sized> EmbeddingProvider for EngineProvider<E> {
    fn embed(&self, texts: &[&str]) -> AstraeaResult<Vec<Vec<f32>>> {
        let fut = async {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.engine.embed(t).await?);
            }
            Ok::<Vec<Vec<f32>>, anyhow::Error>(out)
        };
        block_on_engine(fut)
            .map_err(|e| AstraeaError::QueryExecution(format!("embedding engine error: {e}")))
    }

    fn embedding_dim(&self) -> usize {
        self.engine.embedding_dim()
    }
}

/// Block on an engine future from synchronous code.
///
/// - **Inside a Tokio runtime** (the orchestrator path): uses
///   `block_in_place` + the current handle's `block_on`. This requires a
///   **multi-threaded** runtime — on a current-thread runtime `block_in_place`
///   panics, so the orchestrator (Task 4+) must either run on
///   `#[tokio::main(flavor = "multi_thread")]` or wrap synchronous GraphRAG
///   calls in `tokio::task::spawn_blocking`.
/// - **Outside any runtime** (plain `#[test]`, simple sync callers): spins up a
///   transient current-thread runtime to drive the future to completion.
fn block_on_engine<F: std::future::Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(move || handle.block_on(fut)),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build transient Tokio runtime for inference bridge")
            .block_on(fut),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_engine_complete_returns_text() {
        let engine = MockEngine::new();
        let out = engine.complete("What is Rust?", None).await.unwrap();
        assert!(out.contains("What is Rust?"));
        assert!(out.contains("mock"));

        let out_ctx = engine
            .complete("Q", Some("some retrieved context"))
            .await
            .unwrap();
        assert!(out_ctx.contains("ctx:"));
    }

    #[tokio::test]
    async fn mock_engine_embed_is_768_dim_and_deterministic() {
        let engine = MockEngine::new();
        let a = engine.embed("hello world").await.unwrap();
        let b = engine.embed("hello world").await.unwrap();
        let c = engine.embed("a different string").await.unwrap();

        assert_eq!(a.len(), 768, "embedding must be 768-dim (Decision 7660)");
        assert_eq!(engine.embedding_dim(), 768);
        assert_eq!(a, b, "same text must embed identically (deterministic)");
        assert_ne!(a, c, "different text must embed differently");

        // L2-normalized → unit length.
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "embedding should be L2-normalized");
    }

    #[test]
    fn llm_provider_adapter_compiles_and_completes() {
        // Plain (non-async) test: exercises the block_on_engine fallback path.
        let provider = EngineProvider::from(MockEngine::new());
        let llm: &dyn LlmProvider = &provider;
        let out = llm.complete("hi", "context text").unwrap();
        assert!(out.contains("hi"));
        assert_eq!(llm.name(), "mock-gguf");
        assert!(llm.input_context_tokens() > 0);
    }

    #[test]
    fn embedding_provider_adapter_returns_768_dim_batch() {
        let provider = EngineProvider::from(MockEngine::new());
        let embedder: &dyn EmbeddingProvider = &provider;
        let vecs = embedder.embed(&["one", "two", "three"]).unwrap();
        assert_eq!(vecs.len(), 3, "one vector per input text");
        assert!(vecs.iter().all(|v| v.len() == 768));
        assert_eq!(embedder.embedding_dim(), 768);
    }

    #[test]
    fn one_provider_serves_both_roles() {
        // A single EngineProvider implements both traits — what GraphRAG needs.
        let provider = EngineProvider::from(MockEngine::new());
        let _as_llm: &dyn LlmProvider = &provider;
        let _as_embedder: &dyn EmbeddingProvider = &provider;
    }

    #[test]
    fn adapter_works_over_trait_object_engine() {
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::new());
        let provider = EngineProvider::new(engine);
        let llm: &dyn LlmProvider = &provider;
        assert!(llm.complete("q", "").unwrap().contains("q"));
    }
}
