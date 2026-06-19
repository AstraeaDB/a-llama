//! Task 4 (dev-rag) — GraphRAG retrieval-augment loop.
//!
//! [`Augmenter`] drives `astraea_rag::graph_rag_query` and
//! `astraea_rag::graph_rag_query_anchored` over an a-llama [`KnowledgeStore`]
//! and an in-process LLM / embedding provider, assembling the augmented prompt
//! that the inference engine completes.
//!
//! ## Async / sync runtime contract
//!
//! `astraea_rag::graph_rag_query` and `graph_rag_query_anchored` are
//! **synchronous**.  They call `LlmProvider::complete` and optionally
//! `EmbeddingProvider::embed`, which are also synchronous trait methods.
//! [`EngineProvider`](crate::inference::EngineProvider) bridges the underlying
//! async engine to those sync traits via `tokio::task::block_in_place` (see
//! `src/inference.rs` — `block_on_engine`).  `block_in_place` requires a
//! **multi-threaded** Tokio runtime.
//!
//! [`Augmenter::augment_question`] and [`Augmenter::augment_anchored`]
//! therefore wrap the synchronous pipeline in
//! `tokio::task::spawn_blocking`, which:
//!
//! - is safe to *call* from any runtime flavor (current-thread or multi-thread),
//! - keeps the async caller's worker thread free while the blocking sync work
//!   runs,
//! - and, once inside the blocking thread on a multi-thread runtime, allows
//!   the nested `block_in_place` (inside `EngineProvider`) to drive the async
//!   engine forward.
//!
//! ### Task 6 (HTTP daemon) contract
//!
//! `axum` defaults to `#[tokio::main]` which is multi-threaded, so axum
//! handler tasks that `await` `augment_question` / `augment_anchored` are
//! automatically safe.
//!
//! **Do not** run the HTTP daemon on a `current_thread` runtime
//! (`#[tokio::main(flavor = "current_thread")]`).  The nested `block_in_place`
//! would panic.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use astraea_core::types::NodeId;
use astraea_rag::{
    EmbeddingProvider, GraphRagConfig, GraphRagResult, LlmProvider, TextFormat,
    graph_rag_query, graph_rag_query_anchored,
};

use crate::inference::InferenceEngine;
use crate::knowledge_store::KnowledgeStore;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The `(question, context)` pair ready to hand to
/// [`InferenceEngine::complete`](crate::inference::InferenceEngine::complete).
///
/// Produced by [`Augmenter::assembled_prompt`].  Use this helper when you
/// want to drive inference separately from retrieval (e.g. for streaming
/// responses or to log the context before calling the engine).
#[derive(Debug, Clone)]
pub struct AugmentedPrompt {
    /// The user question — pass as the `prompt` argument to
    /// `InferenceEngine::complete`.
    pub question: String,
    /// The GraphRAG-assembled system context — pass as the `ctx` argument to
    /// `InferenceEngine::complete`.
    ///
    /// `None` when the graph returned no relevant context (e.g. the
    /// `KnowledgeStore` is empty or the context text would be empty after
    /// linearization).
    pub context: Option<String>,
}

// ---------------------------------------------------------------------------
// Augmenter
// ---------------------------------------------------------------------------

/// GraphRAG augment loop for a-llama.
///
/// Wires `astraea_rag::graph_rag_query` / `graph_rag_query_anchored` over the
/// a-llama [`KnowledgeStore`] and an in-process LLM + embedding provider.
///
/// ## Thread-safety
///
/// `Augmenter` is `Send + Sync`; wrap it in an `Arc` and share it across
/// axum handler tasks.
///
/// ## Runtime requirement
///
/// [`augment_question`](Augmenter::augment_question) and
/// [`augment_anchored`](Augmenter::augment_anchored) **must be awaited on a
/// multi-threaded Tokio runtime**.  The internals use `spawn_blocking` for the
/// synchronous RAG pipeline and `block_in_place` for the async engine bridge;
/// both require the multi-thread scheduler.  See the module-level doc for the
/// full explanation and the Task 6 contract.
pub struct Augmenter {
    store: Arc<KnowledgeStore>,
    llm: Arc<dyn LlmProvider>,
    embedder: Arc<dyn EmbeddingProvider>,
    /// Retrieval and token-budget configuration applied to every `augment*`
    /// call.  Exposed publicly so the orchestrator can inspect or swap the
    /// config between requests.
    pub config: GraphRagConfig,
}

impl Augmenter {
    /// Create an `Augmenter` from explicit trait objects.
    ///
    /// Prefer [`Augmenter::with_provider`] when a single
    /// [`EngineProvider`](crate::inference::EngineProvider) serves as both
    /// the LLM and the embedder — the typical a-llama case.
    pub fn new(
        store: Arc<KnowledgeStore>,
        llm: Arc<dyn LlmProvider>,
        embedder: Arc<dyn EmbeddingProvider>,
        config: GraphRagConfig,
    ) -> Self {
        Self { store, llm, embedder, config }
    }

    /// Create an `Augmenter` from a single provider that implements both
    /// [`LlmProvider`] and [`EmbeddingProvider`].
    ///
    /// This is the standard a-llama construction path:
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use a_llama::inference::{EngineProvider, MockEngine};
    /// use a_llama::augment::Augmenter;
    /// use a_llama::knowledge_store::KnowledgeStore;
    ///
    /// let store  = Arc::new(KnowledgeStore::new().unwrap());
    /// let engine = MockEngine::new();
    /// let config = Augmenter::default_config_for(&engine);
    /// let provider = Arc::new(EngineProvider::from(engine));
    /// let augmenter = Augmenter::with_provider(store, provider, config);
    /// ```
    ///
    /// Both `Arc::clone(&provider)` coercions (to `Arc<dyn LlmProvider>` and
    /// `Arc<dyn EmbeddingProvider>`) share ownership of the same underlying
    /// provider so no copies of the engine are made.
    pub fn with_provider<P>(
        store: Arc<KnowledgeStore>,
        provider: Arc<P>,
        config: GraphRagConfig,
    ) -> Self
    where
        P: LlmProvider + EmbeddingProvider + 'static,
    {
        // P: LlmProvider (requires Send + Sync) and EmbeddingProvider (requires
        // Send + Sync), so both unsizing coercions are safe.
        //
        // Split into a concrete intermediate to break the backwards type-inference
        // that makes Arc::clone try to accept &Arc<dyn ...> instead of &Arc<P>.
        let llm_concrete: Arc<P> = Arc::clone(&provider);
        let llm: Arc<dyn LlmProvider> = llm_concrete;
        let embedder: Arc<dyn EmbeddingProvider> = provider;
        Self { store, llm, embedder, config }
    }

    /// Build a sensible default [`GraphRagConfig`] sized against an engine's
    /// input-context window.
    ///
    /// Allocates **25 % of `engine.input_context_tokens()`**, capped at
    /// **8 000 tokens**, to GraphRAG context.  This leaves the remaining 75 %
    /// for the question and the generated response.
    ///
    /// For [`MockEngine`](crate::inference::MockEngine) (32 000-token input
    /// window) this produces a 8 000-token context budget.
    pub fn default_config_for(engine: &impl InferenceEngine) -> GraphRagConfig {
        let context_budget = (engine.input_context_tokens() / 4).min(8_000);
        GraphRagConfig {
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
        }
    }

    /// Assemble the `(question, context)` pair from a [`GraphRagResult`] for
    /// separate use with [`InferenceEngine::complete`](crate::inference::InferenceEngine::complete).
    ///
    /// Call this when you want to inspect or stream the context before (or
    /// instead of) having `augment_question` drive the LLM itself.
    ///
    /// ```ignore
    /// let result = augmenter.augment_question("What is Rust?").await?;
    /// let ap     = Augmenter::assembled_prompt("What is Rust?", &result);
    /// // Optionally inspect ap.context here, then call the engine:
    /// let answer = engine.complete(&ap.question, ap.context.as_deref()).await?;
    /// ```
    pub fn assembled_prompt(question: &str, result: &GraphRagResult) -> AugmentedPrompt {
        AugmentedPrompt {
            question: question.to_string(),
            context: if result.context_text.is_empty() {
                None
            } else {
                Some(result.context_text.clone())
            },
        }
    }

    // -----------------------------------------------------------------------
    // Core augment methods
    // -----------------------------------------------------------------------

    /// Run the full GraphRAG augment loop.
    ///
    /// Pipeline:
    /// 1. **Embed** — call `EmbeddingProvider::embed(question)` to get the
    ///    query vector.
    /// 2. **Vector search** — find the most similar node in the
    ///    [`KnowledgeStore`]'s HNSW index.
    /// 3. **BFS extraction** — expand a subgraph of up to
    ///    `config.max_context_nodes` around the anchor, up to `config.hops`
    ///    hops.
    /// 4. **Linearize** — convert the subgraph to text under
    ///    `config.token_budget` (the pipeline binary-halves `max_context_nodes`
    ///    until the text fits).
    /// 5. **LLM call** — `LlmProvider::complete(prompt, context)` produces the
    ///    answer.
    ///
    /// Returns the full [`GraphRagResult`] which contains the LLM answer
    /// (`result.answer`), the linearized context (`result.context_text`), the
    /// anchor node, and token-budget diagnostics.
    ///
    /// Use [`Augmenter::assembled_prompt`] to split the result into
    /// `(question, context)` for a separate LLM call.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding fails, the vector index is empty
    /// (`AstraeaError::QueryExecution("vector search returned no results")`),
    /// the graph traversal fails, or the LLM call fails.
    ///
    /// # Runtime requirement
    ///
    /// Must be awaited on a **multi-threaded** Tokio runtime.  The call is
    /// wrapped in `tokio::task::spawn_blocking` so the async worker thread is
    /// never blocked, but the runtime must be multi-thread for the nested
    /// `block_in_place` (inside `EngineProvider`) to succeed.
    pub async fn augment_question(&self, question: &str) -> Result<GraphRagResult> {
        let store = Arc::clone(&self.store);
        let llm = Arc::clone(&self.llm);
        let embedder = Arc::clone(&self.embedder);
        let config = self.config.clone();
        let question = question.to_string();

        tokio::task::spawn_blocking(move || {
            // Step 1: Embed the question.
            //
            // `EmbeddingProvider::embed` is a sync trait method; the
            // EngineProvider implementation bridges to the async engine via
            // `block_on_engine` → `block_in_place`.  From a `spawn_blocking`
            // thread on a multi-thread runtime `block_in_place` runs the
            // closure directly (no task to migrate), so this is valid.
            let embeddings = embedder
                .embed(&[question.as_str()])
                .map_err(|e| anyhow!("embed failed: {e}"))?;
            let embedding = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("embed returned empty vector batch"))?;

            // Steps 2-5: GraphRAG pipeline (vector search → BFS → linearize → LLM).
            //
            // `store.graph_ops()` and `store.vector_index_ref()` borrow from
            // `store` which is owned by this closure, so the lifetimes are valid
            // for the entire `graph_rag_query` call.
            graph_rag_query(
                store.graph_ops(),
                store.vector_index_ref(),
                llm.as_ref(),
                &question,
                &embedding,
                &config,
            )
            .map_err(|e| anyhow!("graph_rag_query failed: {e}"))
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking join error: {e}"))?
    }

    /// Run the anchored GraphRAG augment loop, skipping the vector search.
    ///
    /// Use this when the caller already knows the best anchor [`NodeId`] —
    /// for example, when a Eunomia cache hit carries the `NodeId` of the
    /// flushed `CacheEntry` graph node (DESIGN.md §4.3).  Skipping the HNSW
    /// search makes this cheaper than [`augment_question`](Augmenter::augment_question)
    /// and avoids a redundant vector search. (The astraea-vector recall issue
    /// this once also sidestepped, astraeadb-issues #25, has since been fixed.)
    ///
    /// Pipeline: BFS from `anchor` → linearize → token budget → LLM call.
    ///
    /// # Runtime requirement
    ///
    /// Must be awaited on a **multi-threaded** Tokio runtime.  See the module
    /// doc and [`augment_question`](Augmenter::augment_question) for details.
    pub async fn augment_anchored(
        &self,
        question: &str,
        anchor: NodeId,
    ) -> Result<GraphRagResult> {
        let store = Arc::clone(&self.store);
        let llm = Arc::clone(&self.llm);
        let config = self.config.clone();
        let question = question.to_string();

        tokio::task::spawn_blocking(move || {
            graph_rag_query_anchored(
                store.graph_ops(),
                llm.as_ref(),
                &question,
                anchor,
                &config,
            )
            .map_err(|e| anyhow!("graph_rag_query_anchored failed: {e}"))
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking join error: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{EngineProvider, MockEngine};
    use crate::knowledge_store::KnowledgeStore;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a `KnowledgeStore` whose dimension matches `MockEngine` (768).
    fn make_store() -> Arc<KnowledgeStore> {
        Arc::new(KnowledgeStore::with_dim(MockEngine::new().dim).unwrap())
    }

    /// Build a tight config suitable for tests (small budget, 1 hop).
    fn test_config(budget: usize) -> GraphRagConfig {
        GraphRagConfig {
            hops: 1,
            max_context_nodes: 5,
            text_format: TextFormat::Structured,
            token_budget: budget,
            system_prompt: None,
        }
    }

    // -----------------------------------------------------------------------
    // Acceptance test (M1 Task 4)
    // -----------------------------------------------------------------------

    /// **Acceptance test** (DESIGN.md §6 task 4):
    ///
    /// Seed the `KnowledgeStore` with `Fact` nodes whose embeddings come from
    /// `MockEngine::embed`, ask a related question via the **anchored** path
    /// (deterministic — no HNSW ordering uncertainty), and assert that:
    ///
    /// - The assembled context contains the seeded fact text.
    /// - The estimated token count respects the configured budget.
    /// - The LLM was called and returned a non-empty answer.
    ///
    /// The anchored path is preferred here for determinism: `MockEngine` uses
    /// a pseudo-random FNV-1a hash for embeddings (not semantic similarity),
    /// so the vector-search path cannot guarantee which fact is retrieved.
    /// Task 6 integration will exercise the full vector-search path end-to-end
    /// with real (or high-quality mock) embeddings.
    #[tokio::test(flavor = "multi_thread")]
    async fn acceptance_anchored_context_contains_seeded_fact_within_budget() {
        let engine = MockEngine::new();
        let store = make_store();

        // Seed two Fact nodes, embedding each with MockEngine.
        let fact_text_a =
            "Rust is a systems programming language focused on memory safety.";
        let fact_text_b =
            "AstraeaDB is an in-process graph database built in Rust.";

        let emb_a = engine.embed(fact_text_a).await.unwrap();
        let emb_b = engine.embed(fact_text_b).await.unwrap();

        let fact_a = store
            .upsert_fact(fact_text_a, "src-0", 0.95, emb_a, 0)
            .unwrap();
        let _fact_b = store
            .upsert_fact(fact_text_b, "src-1", 0.90, emb_b, 0)
            .unwrap();

        // Wire up the augmenter.
        let provider = Arc::new(EngineProvider::from(MockEngine::new()));
        let augmenter = Augmenter::with_provider(
            Arc::clone(&store),
            provider,
            test_config(1_000),
        );

        // Anchor directly to fact_a — deterministic, no HNSW involved.
        let result = augmenter
            .augment_anchored("What do you know about Rust?", fact_a)
            .await
            .unwrap();

        // The Structured linearizer renders each node as:
        //   Node [Fact: <id>] (text: "...", confidence: ..., source_id: "...", created_at: 0)
        // so the fact text is present in the properties section.
        assert!(
            result.context_text.contains("memory safety"),
            "context should contain the seeded fact text; got:\n{}",
            result.context_text
        );

        // Token budget must be respected.
        assert!(
            result.estimated_tokens <= 1_000,
            "estimated_tokens {} exceeds budget 1000",
            result.estimated_tokens
        );

        // The LLM was invoked and produced a response.
        assert!(!result.answer.is_empty(), "LLM answer must not be empty");

        // The anchor we supplied must be echoed back.
        assert_eq!(
            result.anchor_node_id, fact_a,
            "anchor_node_id must match the NodeId we passed"
        );

        // Sanity: the anchor node must exist in the store with the Fact label.
        let node = store
            .graph_ops()
            .get_node(result.anchor_node_id)
            .unwrap()
            .expect("anchor node must exist in store");
        assert!(
            node.labels.contains(&"Fact".to_string()),
            "anchor node must have the Fact label"
        );
    }

    // -----------------------------------------------------------------------
    // Vector-search path (augment_question)
    // -----------------------------------------------------------------------

    /// Verify that `augment_question` (the full vector-search path) runs the
    /// pipeline end-to-end and returns a non-empty context within the token
    /// budget.
    ///
    /// We do not assert *which* fact is retrieved because `MockEngine` uses
    /// pseudo-random embeddings (not semantic).  We only verify that the
    /// pipeline succeeds and the budget is respected.
    #[tokio::test(flavor = "multi_thread")]
    async fn augment_question_runs_pipeline_and_respects_budget() {
        let engine = MockEngine::new();
        let store = make_store();

        // Seed three facts so HNSW has nodes to return.
        let facts = [
            "The capital of France is Paris.",
            "Tokio is an asynchronous runtime for Rust.",
            "GraphRAG combines vector search with graph traversal.",
        ];
        for (i, text) in facts.iter().enumerate() {
            let emb = engine.embed(text).await.unwrap();
            store
                .upsert_fact(text, &format!("src-{i}"), 0.9, emb, 0)
                .unwrap();
        }

        let provider = Arc::new(EngineProvider::from(MockEngine::new()));
        let augmenter = Augmenter::with_provider(
            Arc::clone(&store),
            provider,
            test_config(2_000),
        );

        let result = augmenter
            .augment_question("What is the capital of France?")
            .await
            .unwrap();

        // Whatever node HNSW returned, the context must be non-empty.
        assert!(
            !result.context_text.is_empty(),
            "context must be non-empty when the store has seeded facts"
        );

        // Budget is respected.
        assert!(
            result.estimated_tokens <= 2_000,
            "estimated_tokens {} exceeds budget 2000",
            result.estimated_tokens
        );

        // At least one node contributed to the context.
        assert!(result.nodes_in_context > 0, "nodes_in_context must be > 0");

        // The anchor node must exist in the store.
        assert!(
            store
                .graph_ops()
                .get_node(result.anchor_node_id)
                .unwrap()
                .is_some(),
            "anchor node must exist in the store"
        );
    }

    // -----------------------------------------------------------------------
    // Anchored path with CacheEntry nodes
    // -----------------------------------------------------------------------

    /// Verify the anchored path works with `CacheEntry` nodes (the other
    /// primary node kind in the long-term graph, written by the Eunomia bridge).
    #[tokio::test(flavor = "multi_thread")]
    async fn augment_anchored_works_with_cache_entry_node() {
        let engine = MockEngine::new();
        let store = make_store();

        let prompt = "What is GraphRAG?";
        let response = "GraphRAG retrieves graph context for LLM prompts.";
        let emb = engine.embed(prompt).await.unwrap();

        let entry_id = store
            .upsert_cache_entry(
                "default",
                "entry-001",
                serde_json::json!({"prompt": prompt, "response": response, "model": "mock"}),
                &["interaction".to_string()],
                emb,
                0,
                0,
            )
            .unwrap();

        let provider = Arc::new(EngineProvider::from(MockEngine::new()));
        let augmenter =
            Augmenter::with_provider(Arc::clone(&store), provider, test_config(1_000));

        let result = augmenter
            .augment_anchored("What is GraphRAG?", entry_id)
            .await
            .unwrap();

        // The JSON context (Structured format) should include the prompt text
        // since `format_properties` includes all non-name properties.
        assert!(
            result.context_text.contains("GraphRAG")
                || result.context_text.contains("CacheEntry"),
            "context should reference the CacheEntry; got:\n{}",
            result.context_text
        );
        assert!(
            result.estimated_tokens <= 1_000,
            "estimated_tokens {} exceeds budget 1000",
            result.estimated_tokens
        );
    }

    // -----------------------------------------------------------------------
    // default_config_for
    // -----------------------------------------------------------------------

    /// `default_config_for` allocates 25 % of the engine's input context
    /// window (capped at 8 000) and sets a non-empty system prompt.
    #[test]
    fn default_config_for_sizes_budget_to_quarter_of_input_window() {
        let engine = MockEngine::new();
        // MockEngine::input_context_tokens() returns 32_000.
        let config = Augmenter::default_config_for(&engine);

        assert_eq!(
            config.token_budget, 8_000,
            "25 % of 32 000 = 8 000, capped at 8 000"
        );
        assert!(config.hops > 0, "hops must be positive");
        assert!(
            config.max_context_nodes > 0,
            "max_context_nodes must be positive"
        );
        assert!(
            config.system_prompt.is_some(),
            "system_prompt should be set by default_config_for"
        );
    }

    // -----------------------------------------------------------------------
    // assembled_prompt helper
    // -----------------------------------------------------------------------

    /// `assembled_prompt` splits a `GraphRagResult` into `(question, context)`
    /// for a separate `InferenceEngine::complete` call.
    #[test]
    fn assembled_prompt_splits_result_correctly() {
        // Non-empty context_text → Some(context).
        let result = GraphRagResult {
            answer: "mock answer".to_string(),
            anchor_node_id: NodeId(1),
            context_text: "graph context here".to_string(),
            nodes_in_context: 1,
            estimated_tokens: 4,
        };
        let ap = Augmenter::assembled_prompt("What is X?", &result);
        assert_eq!(ap.question, "What is X?");
        assert_eq!(ap.context, Some("graph context here".to_string()));

        // Empty context_text → None.
        let empty = GraphRagResult {
            answer: String::new(),
            anchor_node_id: NodeId(1),
            context_text: String::new(),
            nodes_in_context: 0,
            estimated_tokens: 0,
        };
        let ap_empty = Augmenter::assembled_prompt("Q", &empty);
        assert_eq!(
            ap_empty.context, None,
            "empty context_text should yield None"
        );
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    /// Verify `Augmenter` is `Send + Sync` so it can be wrapped in `Arc` and
    /// shared across axum handler tasks.
    #[test]
    fn augmenter_is_send_and_sync() {
        let store = make_store();
        let provider = Arc::new(EngineProvider::from(MockEngine::new()));
        let augmenter = Augmenter::with_provider(
            store,
            provider,
            Augmenter::default_config_for(&MockEngine::new()),
        );
        // If Augmenter does not implement Send + Sync, this line won't compile.
        let _shared: Arc<Augmenter> = Arc::new(augmenter);
    }
}
