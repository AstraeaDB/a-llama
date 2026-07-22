//! Task 2 (dev-astraea-graph) — embedded knowledge store.
//!
//! [`KnowledgeStore`] wraps an in-process [`astraea_graph::Graph`] backed by
//! [`InMemoryStorage`] with an [`crate::exact_index::ExactVectorIndex`] attached.
//! Every node inserted with an embedding is automatically indexed in the exact
//! vector index so that `astraea_rag::graph_rag_query` can find it.
//!
//! The exact (brute-force) index gives recall@k = 1.0 at every corpus size.
//! `astraea_vector::HnswVectorIndex` was replaced because, at the time, its
//! recall@10 collapsed from ~0.98 to ~0.6 past ~3k vectors at 768 dims (astradb
//! #25, since fixed in 0.1.12; measured in `examples/recall.rs`). The exact index
//! is retained — it is correct and O(N·dim) sub-millisecond at every graph size
//! a-llama realistically reaches.
//!
//! ## Task 4 integration
//!
//! [`astraea_rag::graph_rag_query`] expects `(&dyn GraphOps, &dyn VectorIndex,
//! …)`. Obtain those references directly from the store:
//!
//! ```ignore
//! use astraea_rag::{graph_rag_query, GraphRagConfig};
//!
//! let result = graph_rag_query(
//!     store.graph_ops(),
//!     store.vector_index_ref(),
//!     &llm_provider,
//!     question,
//!     &question_embedding,
//!     &config,
//! )?;
//! ```
//!
//! ## Upsert semantics (Task 7 prerequisite)
//!
//! The three `upsert_*` methods are true upserts: on a repeated stable key they
//! update the existing node in-place (properties + embedding) and return the
//! same [`NodeId`]. A new key creates a new node.
//!
//! Dedup keys:
//! - `upsert_cache_entry`: `"{namespace}\x00{eunomia_id}"` — one graph node per
//!   Eunomia entry per namespace.
//! - `upsert_fact`: `"{source_id}\x00{text}"` — a single `(source, text)` pair
//!   is idempotent; distinct fact texts from the same source produce separate
//!   nodes.
//! - `upsert_entity`: `"{kind}\x00{name}"` — one graph node per named entity
//!   per kind.
//!
//! The dedup index is an in-process `Mutex<HashMap<String, NodeId>>` and does
//! not survive a process restart. On restart, Task 8 (durable tiering) will
//! reconstruct the index from the graph when it walks persisted nodes. For M1
//! (InMemoryStorage, single process lifetime) this is transparent.
//!
//! ## Durable mode (Task 8, `durable` feature)
//!
//! When the `durable` cargo feature is enabled, [`KnowledgeStore::open_durable`]
//! opens a [`DiskStorageEngine`]-backed graph at a caller-supplied directory:
//!
//! - `DiskStorageEngine::open(data_dir/"graph")` replays the WAL so all
//!   previously persisted nodes and edges are immediately visible.
//! - A fresh [`ExactVectorIndex`] is attached via `Graph::set_vector_index` and
//!   then repopulated by `Graph::rebuild_vector_index`, which walks **every**
//!   persisted node and reinserts any that carry an embedding — regardless of
//!   label. No snapshot file is written or loaded — the exact index is always
//!   rebuilt from the authoritative graph storage in O(nodes · dim), which is
//!   fast enough that the cost is negligible at every realistic corpus size.
//! - The three dedup indexes (`cache_entry_index`, `fact_index`, `entity_index`)
//!   are reconstructed by scanning persisted nodes labelled `"CacheEntry"`,
//!   `"Fact"`, and `"Entity"` respectively, so that re-`upsert` calls on the
//!   same stable key update in-place rather than duplicating. This dedup walk is
//!   independent of the vector-index rebuild above, so embedding-bearing nodes
//!   under other labels (e.g. a caller-created `"Product"`) are still indexed.
//!
//! Call [`KnowledgeStore::persist`] before dropping to flush the buffer pool
//! to disk. WAL durability is unconditional (astraeadb #27 fix); `persist`
//! only writes a checkpoint. The [`ExactVectorIndex`] needs no snapshot file.
//!
//! [`ExactVectorIndex`]: crate::exact_index::ExactVectorIndex
//!
//! ### astraeadb issue #26 — page size constraint
//!
//! A node record containing a large `value` / `text` property **plus** a
//! 768-dim embedding can exceed astraeadb's ~8 KB single-page limit. Callers
//! must keep the JSON `value` passed to `upsert_cache_entry` and the `text`
//! passed to `upsert_fact` bounded in size (recommended: ≤ 4 KB). The upsert
//! methods do not enforce this limit; a storage-layer error will propagate if
//! the record is too large.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use astraea_core::traits::{GraphOps, VectorIndex};
use astraea_core::types::{EdgeId, NodeId};
use astraea_graph::Graph;
use astraea_graph::test_utils::InMemoryStorage;

use crate::exact_index::ExactVectorIndex;

#[cfg(feature = "durable")]
use std::path::{Path, PathBuf};

#[cfg(feature = "durable")]
use astraea_storage::DiskStorageEngine;

/// Default embedding dimension (Decision 7660: `embeddinggemma`, 768-dim).
/// Every vector index, Eunomia hot store, and in-process `EmbeddingProvider`
/// must agree on this value.
pub const DEFAULT_DIM: usize = 768;

/// Embedded knowledge store for a-llama.
///
/// Holds an in-process [`Graph`] (InMemoryStorage backend) with an attached
/// [`ExactVectorIndex`] for exact nearest-neighbour lookups (recall@k = 1.0).
/// Two node kinds live here (§4.2 of DESIGN.md):
///
/// - **`CacheEntry`** — a prompt→response pair flushed from Eunomia into the
///   durable graph (labels `["CacheEntry", namespace]`).
/// - **`Fact`** — a distilled piece of knowledge extracted from an interaction
///   (label `["Fact"]`).
/// - **`Entity`** — a named entity referenced by facts (label `["Entity"]`).
///
/// Schema edges:
/// ```text
/// (Fact) -DERIVED_FROM-> (CacheEntry)
/// (Fact) -MENTIONS->     (Entity)
/// (Entity) -RELATES_TO-> (Entity)
/// ```
///
/// All insertions that include an embedding automatically index that vector in
/// the [`ExactVectorIndex`] (via [`GraphOps::create_node`] on the underlying
/// `Graph`), making nodes reachable by `graph_rag_query`'s vector-search step.
///
/// ### Thread safety
///
/// `KnowledgeStore` is `Send + Sync` and is intended to be shared behind an
/// `Arc`. The three dedup indexes use `Mutex<HashMap<…>>` for interior
/// mutability; the `Graph` and `ExactVectorIndex` already carry their own
/// internal synchronisation.
pub struct KnowledgeStore {
    graph: Graph,
    /// The same `Arc` that [`Graph`] holds internally; kept here so callers
    /// can obtain `Arc<dyn VectorIndex>` without going through the `Graph`'s
    /// private `Option`.
    vector_index: Arc<dyn VectorIndex>,
    /// Embedding dimension configured at construction time. All embeddings
    /// passed to upsert methods must match this.
    pub dim: usize,

    // ------------------------------------------------------------------
    // Dedup indexes — in-process identity maps for true upsert semantics.
    // Key separator `\x00` prevents collisions between components that
    // might otherwise share a prefix.
    // ------------------------------------------------------------------

    /// `"{namespace}\x00{eunomia_id}"` → NodeId for `CacheEntry` nodes.
    cache_entry_index: Mutex<HashMap<String, NodeId>>,

    /// `"{source_id}\x00{text}"` → NodeId for `Fact` nodes.
    ///
    /// Keying on (source_id, text) means distinct facts from the same
    /// Eunomia entry each get their own node, while re-flushing the same
    /// text is idempotent.
    fact_index: Mutex<HashMap<String, NodeId>>,

    /// `"{kind}\x00{name}"` → NodeId for `Entity` nodes.
    entity_index: Mutex<HashMap<String, NodeId>>,

    /// Data directory for the durable backend (set by [`open_durable`], `None`
    /// in in-memory mode). Used by [`persist`] to determine whether to issue
    /// a storage flush. The [`ExactVectorIndex`] needs no snapshot file and
    /// is rebuilt from graph storage on every [`open_durable`] call.
    /// Only compiled when the `durable` feature is enabled.
    ///
    /// [`open_durable`]: KnowledgeStore::open_durable
    /// [`persist`]: KnowledgeStore::persist
    #[cfg(feature = "durable")]
    data_dir: Option<PathBuf>,
}

impl KnowledgeStore {
    /// Build a `KnowledgeStore` with the default 768-dim HNSW index
    /// (Decision 7660: `embeddinggemma`).
    pub fn new() -> Result<Self> {
        Self::with_dim(DEFAULT_DIM)
    }

    /// Build a `KnowledgeStore` with a custom embedding dimension.
    ///
    /// Useful when the real embedding model is not yet loaded (e.g. in tests
    /// that use a small, cheap index instead of the 768-dim production one).
    pub fn with_dim(dim: usize) -> Result<Self> {
        // ExactVectorIndex gives recall@k = 1.0 at all corpus sizes.
        // It defaults to DistanceMetric::Cosine — correct for L2-normalised
        // text embeddings from embeddinggemma (Decision 7660).
        let vi: Arc<dyn VectorIndex> = Arc::new(ExactVectorIndex::new(dim));
        let storage = Box::new(InMemoryStorage::new());
        let graph = Graph::with_vector_index(storage, Arc::clone(&vi));
        Ok(Self {
            graph,
            vector_index: vi,
            dim,
            cache_entry_index: Mutex::new(HashMap::new()),
            fact_index: Mutex::new(HashMap::new()),
            entity_index: Mutex::new(HashMap::new()),
            #[cfg(feature = "durable")]
            data_dir: None,
        })
    }

    // -----------------------------------------------------------------------
    // Durable backend (Task 8, `durable` feature)
    // -----------------------------------------------------------------------

    /// Open a durable [`KnowledgeStore`] at `data_dir` using
    /// [`DiskStorageEngine`] as the graph backend.
    ///
    /// ## What happens on open
    ///
    /// 1. `DiskStorageEngine::open(data_dir/"graph")` replays the WAL, rebuilding
    ///    in-memory indexes and returning `(engine, max_node_id, max_edge_id)`.
    /// 2. `Graph::with_start_ids(engine, max_node_id+1, max_edge_id+1)` resumes
    ///    id allocation where the previous session left off.
    /// 3. A fresh [`ExactVectorIndex`] is attached via `Graph::set_vector_index`.
    ///    No snapshot file is loaded — the exact index is always rebuilt from the
    ///    authoritative graph storage during step 5.
    /// 4. The three dedup indexes are rebuilt by scanning `"CacheEntry"`,
    ///    `"Fact"`, and `"Entity"` nodes from storage, so re-`upsert` of the same
    ///    stable key updates in-place rather than duplicating.
    /// 5. `Graph::rebuild_vector_index` repopulates the vector index from **every**
    ///    persisted embedding-bearing node — independent of the dedup labels — so
    ///    the index is fully populated (including nodes under other labels such as
    ///    a caller-created `"Product"`) by the time `open_durable` returns.
    ///
    /// For a fresh (empty) `data_dir` this is equivalent to [`with_dim`] with a
    /// `DiskStorageEngine` backend.
    ///
    /// [`ExactVectorIndex`]: crate::exact_index::ExactVectorIndex
    /// [`with_dim`]: KnowledgeStore::with_dim
    #[cfg(feature = "durable")]
    pub fn open_durable(data_dir: &Path, dim: usize) -> Result<Self> {
        let graph_dir = data_dir.join("graph");

        // --- Phase 1: open the WAL-backed storage engine ---
        let (engine, max_node_id, max_edge_id) = DiskStorageEngine::open(&graph_dir)
            .map_err(|e| anyhow!("DiskStorageEngine::open({graph_dir:?}) failed: {e}"))?;

        // Resume id allocation from the last used ids (+1 so the next allocation
        // doesn't collide; start from 1 for a fresh database).
        let next_node_id = max_node_id.saturating_add(1).max(1);
        let next_edge_id = max_edge_id.saturating_add(1).max(1);

        // --- Phase 2: attach a fresh ExactVectorIndex ---
        //
        // The ExactVectorIndex needs no file persistence — it is cheaply rebuilt
        // from authoritative graph storage in the node walk below. No snapshot is
        // saved or loaded. This eliminates the hnsw.idx file and guarantees recall
        // (the HNSW recall bug that originally motivated this, astradb #25, has
        // since been fixed in 0.1.12; the exact index is retained for simplicity).
        let vi: Arc<dyn VectorIndex> = Arc::new(ExactVectorIndex::new(dim));
        let mut graph = Graph::with_start_ids(Box::new(engine), next_node_id, next_edge_id);
        graph.set_vector_index(Arc::clone(&vi));

        // --- Phase 3: rebuild the three in-memory dedup indexes. ---
        //
        // Scan every `CacheEntry`, `Fact`, and `Entity` node from storage and
        // reconstruct the same `(key → NodeId)` maps that the upsert methods
        // maintain during normal operation.  This guarantees that a re-`upsert`
        // of the same key after reopen updates in-place rather than creating a
        // duplicate node.  The vector index is repopulated separately in Phase 3b
        // below (a general walk over all embedding-bearing nodes), so this dedup
        // scan is deliberately *not* responsible for it.

        let mut cache_entry_index: HashMap<String, NodeId> = HashMap::new();
        let mut fact_index: HashMap<String, NodeId> = HashMap::new();
        let mut entity_index: HashMap<String, NodeId> = HashMap::new();

        // CacheEntry: labels = ["CacheEntry", <namespace>]; properties["eunomia_id"]
        let ce_ids = graph
            .find_by_label("CacheEntry")
            .map_err(|e| anyhow!("label scan (CacheEntry) failed: {e}"))?;
        for node_id in ce_ids {
            if let Some(node) = graph
                .get_node(node_id)
                .map_err(|e| anyhow!("get_node {node_id:?} failed: {e}"))?
            {
                if let Some(eid) = node.properties.get("eunomia_id").and_then(|v| v.as_str()) {
                    // Namespace is whichever label is NOT "CacheEntry".
                    let ns = node
                        .labels
                        .iter()
                        .find(|l| l.as_str() != "CacheEntry")
                        .map(String::as_str)
                        .unwrap_or("");
                    cache_entry_index.insert(format!("{ns}\x00{eid}"), node_id);
                }
            }
        }

        // Fact: labels = ["Fact"]; properties["source_id"] + ["text"]
        let fact_ids = graph
            .find_by_label("Fact")
            .map_err(|e| anyhow!("label scan (Fact) failed: {e}"))?;
        for node_id in fact_ids {
            if let Some(node) = graph
                .get_node(node_id)
                .map_err(|e| anyhow!("get_node {node_id:?} failed: {e}"))?
            {
                let src = node.properties.get("source_id").and_then(|v| v.as_str()).unwrap_or("");
                let txt = node.properties.get("text").and_then(|v| v.as_str()).unwrap_or("");
                fact_index.insert(format!("{src}\x00{txt}"), node_id);
            }
        }

        // Entity: labels = ["Entity"]; properties["kind"] + ["name"]
        let entity_ids = graph
            .find_by_label("Entity")
            .map_err(|e| anyhow!("label scan (Entity) failed: {e}"))?;
        for node_id in entity_ids {
            if let Some(node) = graph
                .get_node(node_id)
                .map_err(|e| anyhow!("get_node {node_id:?} failed: {e}"))?
            {
                let kind = node.properties.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                let name = node.properties.get("name").and_then(|v| v.as_str()).unwrap_or("");
                entity_index.insert(format!("{kind}\x00{name}"), node_id);
            }
        }

        // --- Phase 3b: repopulate the exact vector index from ALL persisted
        //     embedding-bearing nodes, independent of the dedup label scans. ---
        //
        // The dedup walk above is label-driven (CacheEntry / Fact / Entity); any
        // other embedding-bearing node — e.g. a caller-created `Product` — would
        // otherwise be persisted but invisible to vector search after reopen
        // (a-llama#1). `Graph::rebuild_vector_index` walks
        // `StorageEngine::list_all_nodes` and inserts every node whose embedding
        // is `Some(..)` into the attached index, returning the count. Inserts are
        // idempotent (ExactVectorIndex overwrites by node_id), so this pass is
        // authoritative regardless of what the dedup scans touched.
        let vectors_indexed = graph
            .rebuild_vector_index()
            .map_err(|e| anyhow!("rebuild_vector_index failed: {e}"))?;

        eprintln!(
            "a-llama: durable graph opened from {data_dir:?} \
             (nodes_in_dedup: ce={}, fact={}, entity={}; vectors_indexed={})",
            cache_entry_index.len(),
            fact_index.len(),
            entity_index.len(),
            vectors_indexed,
        );

        Ok(Self {
            graph,
            vector_index: vi,
            dim,
            cache_entry_index: Mutex::new(cache_entry_index),
            fact_index: Mutex::new(fact_index),
            entity_index: Mutex::new(entity_index),
            data_dir: Some(data_dir.to_path_buf()),
        })
    }

    /// Flush the buffer pool to disk.
    ///
    /// Calls `Graph::flush()` → `DiskStorageEngine::flush_all` which writes
    /// all dirty buffer-pool pages and appends a WAL checkpoint.
    ///
    /// The [`ExactVectorIndex`] needs no snapshot file — it is always rebuilt
    /// from graph storage on the next [`open_durable`] call. No index file is
    /// written.
    ///
    /// Calling `persist` is optional for crash safety (the WAL alone is
    /// sufficient for recovery on the next restart) but recommended on clean
    /// shutdown so the next startup avoids replaying a long WAL.
    ///
    /// Returns `Err` only if the page flush fails at the I/O level; a `None`
    /// `data_dir` (in-memory mode) is treated as a no-op.
    ///
    /// [`ExactVectorIndex`]: crate::exact_index::ExactVectorIndex
    /// [`open_durable`]: KnowledgeStore::open_durable
    #[cfg(feature = "durable")]
    pub fn persist(&self) -> Result<()> {
        // In-memory mode — no storage engine to flush.
        if self.data_dir.is_none() {
            return Ok(());
        }

        // Flush the buffer pool (dirty pages) and write a WAL checkpoint.
        // The ExactVectorIndex is rebuilt from storage on the next open; no
        // index snapshot is needed.
        self.graph
            .storage()
            .flush()
            .map_err(|e| anyhow!("graph flush failed: {e}"))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Node upsert (create-or-update)
    // -----------------------------------------------------------------------

    /// Upsert a `CacheEntry` node into the long-term knowledge graph.
    ///
    /// Called by the Eunomia bridge when it promotes/evicts an entry from the
    /// hot cache into the durable store (DESIGN.md §4.1, Task 7).
    ///
    /// **Dedup key**: `"{namespace}\x00{eunomia_id}"`. A second call with the
    /// same `(namespace, eunomia_id)` pair updates the existing node's
    /// properties and embedding in-place and returns the same [`NodeId`].
    ///
    /// Labels: `["CacheEntry", namespace]`
    /// Properties:
    /// ```json
    /// { "eunomia_id": "…", "value": {…}, "tags": […],
    ///   "created_at": 0, "updated_at": 0 }
    /// ```
    ///
    /// The `embedding` (the prompt embedding from Eunomia) is indexed in the
    /// exact vector index so that `graph_rag_query` can retrieve this node by
    /// vector similarity.
    ///
    /// **Note (astraeadb #26):** keep `value` ≤ ~4 KB; large `value` plus a
    /// 768-dim embedding can exceed the ~8 KB page limit and fail at the
    /// storage layer.
    ///
    /// Returns the allocated or existing [`NodeId`].
    pub fn upsert_cache_entry(
        &self,
        namespace: &str,
        eunomia_id: &str,
        value: Value,
        tags: &[String],
        embedding: Vec<f32>,
        created_at: i64,
        updated_at: i64,
    ) -> Result<NodeId> {
        self.check_dim(&embedding)?;

        let key = format!("{}\x00{}", namespace, eunomia_id);
        let mut idx = self.cache_entry_index.lock().unwrap();

        let props = json!({
            "eunomia_id": eunomia_id,
            "value": value,
            "tags": tags,
            "created_at": created_at,
            "updated_at": updated_at,
        });

        if let Some(&existing_id) = idx.get(&key) {
            // UPDATE path — fetch the existing node, replace properties and
            // embedding, write back through the StorageEngine directly so that
            // both the persisted Node struct and the HNSW index stay in sync.
            let mut node = self
                .graph
                .get_node(existing_id)
                .map_err(|e| anyhow!("upsert_cache_entry get_node failed: {e}"))?
                .ok_or_else(|| {
                    anyhow!(
                        "upsert_cache_entry: node {existing_id:?} absent from graph (index stale?)"
                    )
                })?;

            node.properties = props;
            node.embedding = Some(embedding.clone());

            self.graph
                .storage()
                .put_node(&node)
                .map_err(|e| anyhow!("upsert_cache_entry put_node failed: {e}"))?;

            // Update exact index: remove old vector entry (ignore "not found")
            // then re-insert under the same NodeId with the new embedding.
            let _ = self.vector_index.remove(existing_id);
            self.vector_index
                .insert(existing_id, &embedding)
                .map_err(|e| anyhow!("upsert_cache_entry vector insert failed: {e}"))?;

            Ok(existing_id)
        } else {
            // CREATE path.
            let labels = vec!["CacheEntry".to_string(), namespace.to_string()];
            let id = self
                .graph
                .create_node(labels, props, Some(embedding))
                .map_err(|e| anyhow!("upsert_cache_entry create_node failed: {e}"))?;
            idx.insert(key, id);
            Ok(id)
        }
    }

    /// Upsert a `Fact` node (distilled knowledge) into the graph.
    ///
    /// **Dedup key**: `"{source_id}\x00{text}"`. A second call with the same
    /// `(source_id, text)` pair updates the existing node's `confidence`,
    /// `created_at`, and embedding in-place. Different fact texts from the
    /// same `source_id` produce distinct nodes so a single Eunomia entry can
    /// contribute multiple independent facts.
    ///
    /// Labels: `["Fact"]`
    /// Properties: `{ text, source_id, confidence, created_at }`
    ///
    /// The `embedding` (of the fact text) is indexed in the exact vector index
    /// so it is retrievable by `graph_rag_query`.
    ///
    /// **Note (astraeadb #26):** keep `text` ≤ ~4 KB.
    ///
    /// Returns the allocated or existing [`NodeId`].
    pub fn upsert_fact(
        &self,
        text: &str,
        source_id: &str,
        confidence: f32,
        embedding: Vec<f32>,
        created_at: i64,
    ) -> Result<NodeId> {
        self.check_dim(&embedding)?;

        let key = format!("{}\x00{}", source_id, text);
        let mut idx = self.fact_index.lock().unwrap();

        let props = json!({
            "text": text,
            "source_id": source_id,
            "confidence": confidence,
            "created_at": created_at,
        });

        if let Some(&existing_id) = idx.get(&key) {
            let mut node = self
                .graph
                .get_node(existing_id)
                .map_err(|e| anyhow!("upsert_fact get_node failed: {e}"))?
                .ok_or_else(|| {
                    anyhow!("upsert_fact: node {existing_id:?} absent from graph (index stale?)")
                })?;

            node.properties = props;
            node.embedding = Some(embedding.clone());

            self.graph
                .storage()
                .put_node(&node)
                .map_err(|e| anyhow!("upsert_fact put_node failed: {e}"))?;

            let _ = self.vector_index.remove(existing_id);
            self.vector_index
                .insert(existing_id, &embedding)
                .map_err(|e| anyhow!("upsert_fact vector insert failed: {e}"))?;

            Ok(existing_id)
        } else {
            let id = self
                .graph
                .create_node(vec!["Fact".to_string()], props, Some(embedding))
                .map_err(|e| anyhow!("upsert_fact create_node failed: {e}"))?;
            idx.insert(key, id);
            Ok(id)
        }
    }

    /// Upsert an `Entity` node (a named thing referenced by facts).
    ///
    /// **Dedup key**: `"{kind}\x00{name}"`. A second call with the same
    /// `(kind, name)` pair updates the existing node's properties and — if
    /// `embedding` is `Some` — its embedding too. Passing `None` for
    /// `embedding` on an update leaves the stored embedding unchanged.
    ///
    /// Labels: `["Entity"]`
    /// Properties: `{ name, kind }`
    ///
    /// `embedding` is optional; supply it when the entity name has been
    /// embedded so that hybrid/semantic searches can reach it.
    ///
    /// Returns the allocated or existing [`NodeId`].
    pub fn upsert_entity(
        &self,
        name: &str,
        kind: &str,
        embedding: Option<Vec<f32>>,
    ) -> Result<NodeId> {
        if let Some(ref emb) = embedding {
            self.check_dim(emb)?;
        }

        let key = format!("{}\x00{}", kind, name);
        let mut idx = self.entity_index.lock().unwrap();

        let props = json!({ "name": name, "kind": kind });

        if let Some(&existing_id) = idx.get(&key) {
            let mut node = self
                .graph
                .get_node(existing_id)
                .map_err(|e| anyhow!("upsert_entity get_node failed: {e}"))?
                .ok_or_else(|| {
                    anyhow!(
                        "upsert_entity: node {existing_id:?} absent from graph (index stale?)"
                    )
                })?;

            node.properties = props;

            if let Some(emb) = embedding {
                // Update the embedding in storage and in the HNSW index.
                node.embedding = Some(emb.clone());
                self.graph
                    .storage()
                    .put_node(&node)
                    .map_err(|e| anyhow!("upsert_entity put_node failed: {e}"))?;

                let _ = self.vector_index.remove(existing_id);
                self.vector_index
                    .insert(existing_id, &emb)
                    .map_err(|e| anyhow!("upsert_entity vector insert failed: {e}"))?;
            } else {
                // No new embedding supplied — update only properties; keep
                // whatever embedding is already stored on the node.
                self.graph
                    .storage()
                    .put_node(&node)
                    .map_err(|e| anyhow!("upsert_entity put_node (props-only) failed: {e}"))?;
            }

            Ok(existing_id)
        } else {
            let id = self
                .graph
                .create_node(vec!["Entity".to_string()], props, embedding)
                .map_err(|e| anyhow!("upsert_entity create_node failed: {e}"))?;
            idx.insert(key, id);
            Ok(id)
        }
    }

    // -----------------------------------------------------------------------
    // Edge helpers (schema §4.2)
    // -----------------------------------------------------------------------

    /// `(Fact) -DERIVED_FROM-> (CacheEntry)`
    ///
    /// Records which interaction a fact was extracted from.
    pub fn link_derived_from(
        &self,
        fact_id: NodeId,
        cache_entry_id: NodeId,
    ) -> Result<EdgeId> {
        self.graph
            .create_edge(
                fact_id,
                cache_entry_id,
                "DERIVED_FROM".to_string(),
                json!({}),
                1.0,
                None,
                None,
            )
            .map_err(|e| anyhow!("link_derived_from failed: {e}"))
    }

    /// `(Fact) -MENTIONS-> (Entity)`
    ///
    /// Records entities mentioned in a fact.
    pub fn link_mentions(&self, fact_id: NodeId, entity_id: NodeId) -> Result<EdgeId> {
        self.graph
            .create_edge(
                fact_id,
                entity_id,
                "MENTIONS".to_string(),
                json!({}),
                1.0,
                None,
                None,
            )
            .map_err(|e| anyhow!("link_mentions failed: {e}"))
    }

    /// `(Entity) -RELATES_TO-> (Entity)`
    ///
    /// Links related entities together.
    pub fn link_relates_to(
        &self,
        from_entity: NodeId,
        to_entity: NodeId,
    ) -> Result<EdgeId> {
        self.graph
            .create_edge(
                from_entity,
                to_entity,
                "RELATES_TO".to_string(),
                json!({}),
                1.0,
                None,
                None,
            )
            .map_err(|e| anyhow!("link_relates_to failed: {e}"))
    }

    // -----------------------------------------------------------------------
    // GraphRAG accessors (Task 4 integration)
    // -----------------------------------------------------------------------

    /// Borrow the graph as `&dyn GraphOps`.
    ///
    /// Pass this directly to [`astraea_rag::graph_rag_query`]:
    /// ```ignore
    /// graph_rag_query(store.graph_ops(), store.vector_index_ref(), …)
    /// ```
    pub fn graph_ops(&self) -> &dyn GraphOps {
        &self.graph
    }

    /// Borrow the exact vector index as `&dyn VectorIndex`.
    ///
    /// Pass this directly to [`astraea_rag::graph_rag_query`].
    pub fn vector_index_ref(&self) -> &dyn VectorIndex {
        self.vector_index.as_ref()
    }

    /// Shared ownership of the exact vector index as `Arc<dyn VectorIndex>`.
    ///
    /// Use when another component (e.g. the Eunomia bridge) must share the
    /// same index without borrowing `KnowledgeStore`.
    pub fn vector_index_arc(&self) -> Arc<dyn VectorIndex> {
        Arc::clone(&self.vector_index)
    }

    /// Direct borrow of the underlying [`Graph`].
    ///
    /// Needed for `Graph`-specific methods not on the `GraphOps` trait (e.g.
    /// [`Graph::storage`]).
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn check_dim(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dim {
            Err(anyhow!(
                "embedding dim mismatch: store is {}-dim but got {}-dim vector",
                self.dim,
                embedding.len()
            ))
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use astraea_core::types::Direction;

    /// Build a small, deterministic L2-normalised embedding of `dim` dimensions.
    ///
    /// Index `i` produces a vector that is dominant in dimension `i % dim`
    /// and has a small positive value (0.01) in all other dimensions.
    /// Different `i` values produce genuinely cosine-distinct vectors so
    /// nearest-neighbour results are unambiguous even with only a few
    /// indexed vectors.
    fn make_embedding(i: usize, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|d| if d == i % dim { 1.0_f32 } else { 0.01_f32 })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    // --- Construction ---

    #[test]
    fn new_produces_768_dim_store() {
        let store = KnowledgeStore::new().unwrap();
        assert_eq!(store.dim, DEFAULT_DIM);
        assert_eq!(store.vector_index_ref().dimension(), DEFAULT_DIM);
    }

    #[test]
    fn with_dim_uses_requested_dimension() {
        let store = KnowledgeStore::with_dim(4).unwrap();
        assert_eq!(store.dim, 4);
        assert_eq!(store.vector_index_ref().dimension(), 4);
    }

    // --- upsert_cache_entry ---

    #[test]
    fn upsert_cache_entry_creates_labeled_node_with_embedding() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb = make_embedding(0, dim);

        let id = store
            .upsert_cache_entry(
                "default",
                "entry-001",
                json!({"prompt": "hello", "response": "world", "model": "mock"}),
                &["interaction".to_string(), "mock".to_string()],
                emb.clone(),
                1_700_000_000,
                1_700_000_001,
            )
            .unwrap();

        let node = store.graph_ops().get_node(id).unwrap().unwrap();
        assert!(
            node.labels.contains(&"CacheEntry".to_string()),
            "must have CacheEntry label"
        );
        assert!(
            node.labels.contains(&"default".to_string()),
            "must have namespace label"
        );
        assert_eq!(node.properties["eunomia_id"], "entry-001");
        assert_eq!(node.properties["created_at"], 1_700_000_000_i64);
        assert!(node.embedding.is_some(), "embedding must be stored on node");
    }

    #[test]
    fn upsert_cache_entry_deduplicates_by_namespace_and_eunomia_id() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb1 = make_embedding(0, dim);
        let emb2 = make_embedding(1, dim);

        // First upsert — creates the node.
        let id1 = store
            .upsert_cache_entry(
                "ns",
                "e-1",
                json!({"prompt": "hello", "response": "world"}),
                &[],
                emb1.clone(),
                100,
                100,
            )
            .unwrap();

        let count_before = store.graph_ops().find_by_label("CacheEntry").unwrap().len();

        // Second upsert — same (namespace, eunomia_id), different content.
        let id2 = store
            .upsert_cache_entry(
                "ns",
                "e-1",
                json!({"prompt": "hello updated", "response": "world updated"}),
                &["tag1".to_string()],
                emb2.clone(),
                200,
                200,
            )
            .unwrap();

        // Must return the SAME NodeId.
        assert_eq!(id1, id2, "upsert with same key must return existing NodeId");

        // Node count must not have increased.
        let count_after = store.graph_ops().find_by_label("CacheEntry").unwrap().len();
        assert_eq!(
            count_before, count_after,
            "upsert must not create a duplicate CacheEntry node"
        );

        // Properties must be updated.
        let node = store.graph_ops().get_node(id1).unwrap().unwrap();
        assert_eq!(node.properties["updated_at"], 200_i64, "updated_at must be updated");
        assert_eq!(
            node.properties["value"]["response"], "world updated",
            "value must be updated"
        );

        // Embedding in storage must reflect the new vector.
        assert_eq!(
            node.embedding.as_deref(),
            Some(emb2.as_slice()),
            "node.embedding in storage must be updated to emb2"
        );

        // HNSW index must also be updated: searching for emb2 returns id1.
        let results = store.vector_index_ref().search(&emb2, 1).unwrap();
        assert!(!results.is_empty(), "HNSW search must return a result");
        assert_eq!(
            results[0].node_id, id1,
            "HNSW index must have been updated to emb2 for id1"
        );
    }

    #[test]
    fn upsert_cache_entry_different_key_creates_new_node() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb = make_embedding(0, dim);

        let id1 = store
            .upsert_cache_entry("ns", "e-1", json!({}), &[], emb.clone(), 0, 0)
            .unwrap();
        // Different eunomia_id → new node.
        let id2 = store
            .upsert_cache_entry("ns", "e-2", json!({}), &[], emb.clone(), 0, 0)
            .unwrap();
        assert_ne!(id1, id2, "different eunomia_id must create a new node");

        // Different namespace → new node even with same eunomia_id.
        let id3 = store
            .upsert_cache_entry("ns2", "e-1", json!({}), &[], emb.clone(), 0, 0)
            .unwrap();
        assert_ne!(id1, id3, "different namespace must create a new node");
    }

    // --- upsert_fact ---

    #[test]
    fn upsert_fact_creates_fact_node() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb = make_embedding(1, dim);

        let id = store
            .upsert_fact(
                "Rust is a systems programming language.",
                "entry-001",
                0.95,
                emb,
                1_700_000_000,
            )
            .unwrap();

        let node = store.graph_ops().get_node(id).unwrap().unwrap();
        assert!(node.labels.contains(&"Fact".to_string()));
        assert_eq!(
            node.properties["text"],
            "Rust is a systems programming language."
        );
        assert!(
            (node.properties["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-4
        );
        assert_eq!(node.properties["source_id"], "entry-001");
    }

    #[test]
    fn upsert_fact_deduplicates_by_source_id_and_text() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb1 = make_embedding(0, dim);
        let emb2 = make_embedding(1, dim);

        // First upsert — creates the node.
        let id1 = store
            .upsert_fact("Rust is fast.", "src-1", 0.9, emb1.clone(), 100)
            .unwrap();

        let count_before = store.graph_ops().find_by_label("Fact").unwrap().len();

        // Second upsert — same (source_id, text), updated confidence + embedding.
        let id2 = store
            .upsert_fact("Rust is fast.", "src-1", 0.99, emb2.clone(), 200)
            .unwrap();

        assert_eq!(id1, id2, "upsert with same (source_id, text) must return existing NodeId");

        let count_after = store.graph_ops().find_by_label("Fact").unwrap().len();
        assert_eq!(
            count_before, count_after,
            "upsert must not create a duplicate Fact node"
        );

        // Properties must be updated.
        let node = store.graph_ops().get_node(id1).unwrap().unwrap();
        assert!(
            (node.properties["confidence"].as_f64().unwrap() - 0.99).abs() < 1e-4,
            "confidence must be updated to 0.99"
        );
        assert_eq!(node.properties["created_at"], 200_i64, "created_at must be updated");

        // Embedding in storage must be the new vector.
        assert_eq!(
            node.embedding.as_deref(),
            Some(emb2.as_slice()),
            "node.embedding in storage must be updated to emb2"
        );

        // HNSW index updated: emb2 is nearest to id1.
        let results = store.vector_index_ref().search(&emb2, 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, id1, "HNSW must map emb2 to id1 after update");
    }

    #[test]
    fn upsert_fact_different_text_same_source_creates_new_node() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb1 = make_embedding(0, dim);
        let emb2 = make_embedding(1, dim);

        // Same source_id, different text — two distinct facts.
        let id1 = store.upsert_fact("Rust is fast.", "src-1", 0.9, emb1, 0).unwrap();
        let id2 = store
            .upsert_fact("Rust is memory-safe.", "src-1", 0.9, emb2, 0)
            .unwrap();
        assert_ne!(id1, id2, "different text with same source must create a new node");
        assert_eq!(
            store.graph_ops().find_by_label("Fact").unwrap().len(),
            2,
            "two distinct facts must produce two nodes"
        );
    }

    // --- upsert_entity ---

    #[test]
    fn upsert_entity_with_and_without_embedding() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        // With embedding.
        let emb = make_embedding(2, dim);
        let id_with = store.upsert_entity("Rust", "language", Some(emb)).unwrap();
        let node = store.graph_ops().get_node(id_with).unwrap().unwrap();
        assert!(node.labels.contains(&"Entity".to_string()));
        assert_eq!(node.properties["name"], "Rust");
        assert_eq!(node.properties["kind"], "language");
        assert!(node.embedding.is_some());

        // Without embedding.
        let id_bare = store.upsert_entity("GitHub", "platform", None).unwrap();
        let node2 = store.graph_ops().get_node(id_bare).unwrap().unwrap();
        assert!(node2.embedding.is_none());
    }

    #[test]
    fn upsert_entity_deduplicates_by_kind_and_name() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb1 = make_embedding(0, dim);
        let emb2 = make_embedding(1, dim);

        // First upsert — creates the node.
        let id1 = store.upsert_entity("Rust", "language", Some(emb1.clone())).unwrap();

        let count_before = store.graph_ops().find_by_label("Entity").unwrap().len();

        // Second upsert — same (kind, name), new embedding.
        let id2 = store.upsert_entity("Rust", "language", Some(emb2.clone())).unwrap();

        assert_eq!(id1, id2, "upsert with same (kind, name) must return existing NodeId");

        let count_after = store.graph_ops().find_by_label("Entity").unwrap().len();
        assert_eq!(
            count_before, count_after,
            "upsert must not create a duplicate Entity node"
        );

        // Embedding in storage must be the new vector.
        let node = store.graph_ops().get_node(id1).unwrap().unwrap();
        assert_eq!(
            node.embedding.as_deref(),
            Some(emb2.as_slice()),
            "node.embedding in storage must be updated to emb2"
        );

        // HNSW index updated: emb2 is nearest to id1.
        let results = store.vector_index_ref().search(&emb2, 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, id1, "HNSW must map emb2 to id1 after entity update");
    }

    #[test]
    fn upsert_entity_update_without_embedding_preserves_existing() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();
        let emb = make_embedding(0, dim);

        // Create with an embedding.
        let id1 = store.upsert_entity("Rust", "language", Some(emb.clone())).unwrap();

        // Update with None — embedding must be preserved.
        let id2 = store.upsert_entity("Rust", "language", None).unwrap();
        assert_eq!(id1, id2, "None-embedding upsert must return existing NodeId");

        let node = store.graph_ops().get_node(id1).unwrap().unwrap();
        assert_eq!(
            node.embedding.as_deref(),
            Some(emb.as_slice()),
            "existing embedding must be preserved when upsert supplies None"
        );
    }

    #[test]
    fn upsert_entity_different_name_or_kind_creates_new_node() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        let id1 = store.upsert_entity("Rust", "language", None).unwrap();
        // Different name → new node.
        let id2 = store.upsert_entity("Python", "language", None).unwrap();
        assert_ne!(id1, id2, "different name must create a new node");

        // Different kind → new node (even if name matches).
        let id3 = store.upsert_entity("Rust", "tool", None).unwrap();
        assert_ne!(id1, id3, "different kind must create a new node");
    }

    // --- Vector search ---

    #[test]
    fn vector_search_returns_nearest_inserted_node() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        // Two facts with genuinely different (cosine-distant) embeddings.
        let emb_a = make_embedding(0, dim); // dominant in dim 0
        let emb_b = make_embedding(1, dim); // dominant in dim 1 — far from query

        let id_a = store
            .upsert_fact("Fact A", "src-a", 0.9, emb_a.clone(), 0)
            .unwrap();
        let _id_b = store.upsert_fact("Fact B", "src-b", 0.9, emb_b, 0).unwrap();

        // Query with emb_a — should come back as the nearest hit.
        let results = store.vector_index_ref().search(&emb_a, 1).unwrap();
        assert!(!results.is_empty(), "search must return at least one result");
        assert_eq!(
            results[0].node_id, id_a,
            "nearest node should be the one whose embedding matches the query"
        );
    }

    // --- BFS traversal ---

    #[test]
    fn bfs_traverses_mentions_edge() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        let fact_id = store
            .upsert_fact("Rust is fast.", "src-001", 0.9, make_embedding(0, dim), 0)
            .unwrap();
        let entity_id = store
            .upsert_entity("Rust", "language", Some(make_embedding(1, dim)))
            .unwrap();

        store.link_mentions(fact_id, entity_id).unwrap();

        // BFS from fact_id at depth 1 must reach entity_id.
        let reachable: Vec<NodeId> = store
            .graph_ops()
            .bfs(fact_id, 1)
            .unwrap()
            .into_iter()
            .map(|(id, _depth)| id)
            .collect();

        assert!(reachable.contains(&fact_id), "start node must appear in BFS");
        assert!(
            reachable.contains(&entity_id),
            "BFS should reach the entity via the MENTIONS edge"
        );
    }

    // --- Full schema (acceptance test) ---

    /// Acceptance test for Task 2: creates nodes with embeddings, vector-searches
    /// them, and BFS-traverses the DERIVED_FROM / MENTIONS / RELATES_TO edge set —
    /// all in-process, no server.
    #[test]
    fn full_schema_nodes_vector_search_and_bfs() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        // Build the schema graph.  Each embedding is dominant in a distinct
        // dimension so cosine distances are unambiguous for HNSW recall.
        let cache_id = store
            .upsert_cache_entry(
                "session-1",
                "e-abc",
                json!({"prompt": "What is AstraeaDB?", "response": "A graph DB."}),
                &["interaction".to_string()],
                make_embedding(0, dim), // dominant in dim 0
                0,
                0,
            )
            .unwrap();

        let fact_id = store
            .upsert_fact(
                "AstraeaDB is an in-process graph database.",
                "e-abc",
                0.85,
                make_embedding(1, dim), // dominant in dim 1
                0,
            )
            .unwrap();

        let entity_a = store
            .upsert_entity("AstraeaDB", "product", Some(make_embedding(2, dim))) // dim 2
            .unwrap();
        let entity_b = store
            .upsert_entity("graph database", "category", Some(make_embedding(3, dim))) // dim 3
            .unwrap();

        // Wire up edges: schema §4.2.
        store.link_derived_from(fact_id, cache_id).unwrap();
        store.link_mentions(fact_id, entity_a).unwrap();
        store.link_mentions(fact_id, entity_b).unwrap();
        store.link_relates_to(entity_a, entity_b).unwrap();

        // --- Vector search ---
        // Query dominant in dim 1 → should match the fact (make_embedding(1)).
        let query = make_embedding(1, dim);
        let results = store.vector_index_ref().search(&query, 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].node_id, fact_id,
            "vector search should return the fact as nearest node"
        );

        // --- BFS from fact_id at depth 2 ---
        // depth 1: cache_id, entity_a, entity_b (via DERIVED_FROM, MENTIONS x2)
        // depth 2: entity_b via entity_a's RELATES_TO (already at depth 1 too)
        let reachable: Vec<NodeId> = store
            .graph_ops()
            .bfs(fact_id, 2)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        assert!(reachable.contains(&fact_id));
        assert!(reachable.contains(&cache_id));
        assert!(reachable.contains(&entity_a));
        assert!(reachable.contains(&entity_b));

        // --- Neighbor check ---
        let outgoing = store
            .graph_ops()
            .neighbors(fact_id, Direction::Outgoing)
            .unwrap();
        assert_eq!(
            outgoing.len(),
            3,
            "fact_id must have 3 outgoing edges: DERIVED_FROM + MENTIONS x2"
        );
    }

    // --- Dimension mismatch ---

    #[test]
    fn dim_mismatch_is_rejected() {
        let store = KnowledgeStore::with_dim(4).unwrap();
        // Supply a 2-dim embedding to a 4-dim store.
        let bad_emb = vec![1.0f32, 0.0];
        assert!(
            store.upsert_fact("test", "src", 0.5, bad_emb, 0).is_err(),
            "dim mismatch must return Err"
        );
    }

    // --- ExactVectorIndex integration (test 3 from exact-index spec) ---

    /// Confirm that `KnowledgeStore` now uses the exact index and therefore
    /// returns the deterministically correct nearest neighbour — not an
    /// approximate one.  Three facts with maximally distinct embeddings
    /// (each dominant in a different dimension) are inserted; searching with
    /// each embedding must return its own node as rank-1.
    #[test]
    fn knowledge_store_exact_retrieval_is_deterministic() {
        let dim = 4;
        let store = KnowledgeStore::with_dim(dim).unwrap();

        // Three L2-normalised embeddings, each dominant in a different dimension.
        let emb_a = make_embedding(0, dim); // dominant in dim 0
        let emb_b = make_embedding(1, dim); // dominant in dim 1
        let emb_c = make_embedding(2, dim); // dominant in dim 2

        let id_a = store.upsert_fact("fact A", "src-a", 0.9, emb_a.clone(), 0).unwrap();
        let id_b = store.upsert_fact("fact B", "src-b", 0.9, emb_b.clone(), 0).unwrap();
        let id_c = store.upsert_fact("fact C", "src-c", 0.9, emb_c.clone(), 0).unwrap();

        // Each embedding must rank-1 its own node exactly.
        let r_a = store.vector_index_ref().search(&emb_a, 1).unwrap();
        assert_eq!(r_a[0].node_id, id_a, "emb_a must nearest-match fact A");
        assert_eq!(r_a[0].distance, 0.0_f32, "self-distance must be 0.0 (exact)");

        let r_b = store.vector_index_ref().search(&emb_b, 1).unwrap();
        assert_eq!(r_b[0].node_id, id_b, "emb_b must nearest-match fact B");

        let r_c = store.vector_index_ref().search(&emb_c, 1).unwrap();
        assert_eq!(r_c[0].node_id, id_c, "emb_c must nearest-match fact C");

        // Top-3 search must return all three nodes.
        let r_all = store.vector_index_ref().search(&emb_a, 3).unwrap();
        let ids: Vec<_> = r_all.iter().map(|r| r.node_id).collect();
        assert!(ids.contains(&id_a) && ids.contains(&id_b) && ids.contains(&id_c),
            "top-3 search must return all three facts; got {ids:?}");
    }

    // --- Accessor type checks ---

    #[test]
    fn graphrag_accessors_return_expected_types() {
        let store = KnowledgeStore::new().unwrap();
        // These lines are the exact call pattern Task 4 will use.
        let _graph_ops: &dyn GraphOps = store.graph_ops();
        let _vi_ref: &dyn VectorIndex = store.vector_index_ref();
        let _vi_arc: Arc<dyn VectorIndex> = store.vector_index_arc();
        let _graph: &Graph = store.graph();
    }
}

// ---------------------------------------------------------------------------
// Durable-mode acceptance tests (Task 8)
// ---------------------------------------------------------------------------
//
// These tests require the `durable` feature.  Run with:
//   cargo test --features durable
//
// Each test gets its own uniquely-named temp directory so they can run in
// parallel without racing on shared paths.

#[cfg(all(test, feature = "durable"))]
mod durable_tests {
    use super::*;

    /// Build a small L2-normalised embedding with a dominant spike at dimension
    /// `i % dim`. Mirrors the helper in the in-memory test module above.
    fn make_embedding(i: usize, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|d| if d == i % dim { 1.0_f32 } else { 0.01_f32 })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// Create a unique temporary directory for one test run.  The directory is
    /// automatically removed when the returned guard is dropped.
    fn tmp_dir(label: &str) -> TmpDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("a_llama_ks_durable_{label}_{nanos}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TmpDir(path)
    }

    struct TmpDir(std::path::PathBuf);
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ------------------------------------------------------------------
    // Acceptance test (a) – (d): nodes, embeddings, HNSW, dedup, edges
    // ------------------------------------------------------------------

    #[test]
    fn durable_store_survives_restart() {
        let dir = tmp_dir("survives");
        let dim = 4;

        // --- Session 1: write nodes + edges, persist, drop ---
        let (fact_id, entity_id) = {
            let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();

            let fid = store
                .upsert_fact("AstraeaDB persists.", "src-durable-1", 0.9, make_embedding(0, dim), 100)
                .unwrap();
            let eid = store
                .upsert_entity("AstraeaDB", "product", Some(make_embedding(1, dim)))
                .unwrap();
            store.link_mentions(fid, eid).unwrap();

            // Also test CacheEntry dedup reconstruction.
            store
                .upsert_cache_entry(
                    "test-ns",
                    "ce-001",
                    serde_json::json!({"prompt": "p", "response": "r"}),
                    &["interaction".to_string()],
                    make_embedding(2, dim),
                    1_000,
                    1_001,
                )
                .unwrap();

            store.persist().unwrap();
            (fid, eid)
        };

        // --- Session 2: reopen and verify all acceptance criteria ---
        {
            let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();

            // (a) Nodes + properties + embeddings survive.
            let fact = store.graph_ops().get_node(fact_id).unwrap().unwrap();
            assert_eq!(fact.properties["text"], "AstraeaDB persists.", "(a) text property");
            assert_eq!(fact.properties["source_id"], "src-durable-1", "(a) source_id");
            assert!(fact.embedding.is_some(), "(a) embedding must survive");
            assert_eq!(fact.embedding.as_deref(), Some(make_embedding(0, dim).as_slice()));

            let entity = store.graph_ops().get_node(entity_id).unwrap().unwrap();
            assert_eq!(entity.properties["name"], "AstraeaDB", "(a) entity name");
            assert!(entity.embedding.is_some(), "(a) entity embedding");

            // (b) Exact vector search returns the correct node after restart.
            let results = store.vector_index_ref().search(&make_embedding(0, dim), 1).unwrap();
            assert!(!results.is_empty(), "(b) exact index must return results after restart");
            assert_eq!(results[0].node_id, fact_id, "(b) fact must be nearest node");

            // (c) Dedup indexes rebuilt: re-upsert same key → same NodeId, no duplicate.
            let fact_id2 = store
                .upsert_fact("AstraeaDB persists.", "src-durable-1", 0.99, make_embedding(0, dim), 200)
                .unwrap();
            assert_eq!(fact_id2, fact_id, "(c) re-upsert Fact must return existing NodeId");
            assert_eq!(
                store.graph_ops().find_by_label("Fact").unwrap().len(),
                1,
                "(c) no duplicate Fact nodes after re-upsert"
            );

            let entity_id2 = store
                .upsert_entity("AstraeaDB", "product", Some(make_embedding(1, dim)))
                .unwrap();
            assert_eq!(entity_id2, entity_id, "(c) re-upsert Entity must return existing NodeId");

            let ce_id2 = store
                .upsert_cache_entry(
                    "test-ns",
                    "ce-001",
                    serde_json::json!({"prompt": "p2", "response": "r2"}),
                    &[],
                    make_embedding(2, dim),
                    1_002,
                    1_003,
                )
                .unwrap();
            let ce_nodes = store.graph_ops().find_by_label("CacheEntry").unwrap();
            assert_eq!(ce_nodes.len(), 1, "(c) no duplicate CacheEntry nodes");
            let ce_node = store.graph_ops().get_node(ce_id2).unwrap().unwrap();
            assert_eq!(ce_node.properties["value"]["response"], "r2", "(c) CacheEntry updated");

            // (d) Edges survive: entity reachable from fact via BFS.
            let reachable: Vec<NodeId> = store
                .graph_ops()
                .bfs(fact_id, 1)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert!(
                reachable.contains(&entity_id),
                "(d) MENTIONS edge must survive restart (entity BFS-reachable from fact)"
            );
        }
    }

    // ------------------------------------------------------------------
    // Multi-session test: three open→write→persist→drop cycles (WAL #27)
    // ------------------------------------------------------------------

    #[test]
    fn durable_multi_session_no_corruption() {
        let dir = tmp_dir("multi_session");
        let dim = 4;

        let mut session_ids: Vec<(NodeId, String)> = Vec::new();

        // Three sessions, each appending one unique fact.
        for i in 0..3usize {
            let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();
            let text = format!("Multi-session fact {i}");
            let src = format!("session-src-{i}");
            let id = store
                .upsert_fact(&text, &src, 0.9, make_embedding(i, dim), i as i64 * 100)
                .unwrap();
            session_ids.push((id, text));
            store.persist().unwrap();
            // `store` drops here, releasing the DiskStorageEngine.
        }

        // Final read-only session: all three facts must be intact, none corrupted.
        let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();
        let all_facts = store.graph_ops().find_by_label("Fact").unwrap();
        assert_eq!(
            all_facts.len(),
            3,
            "all three session facts must survive three open/persist/drop cycles"
        );

        for (id, expected_text) in &session_ids {
            let node = store.graph_ops().get_node(*id).unwrap().unwrap();
            assert_eq!(
                node.properties["text"].as_str().unwrap(),
                expected_text,
                "text from session {expected_text:?} must be intact"
            );
            assert!(node.embedding.is_some(), "embedding must persist across sessions");
        }

        // Exact index must still serve all three embeddings.
        for (i, (id, _)) in session_ids.iter().enumerate() {
            let results = store.vector_index_ref().search(&make_embedding(i, dim), 1).unwrap();
            assert!(!results.is_empty(), "exact index must find session {i} embedding");
            assert_eq!(
                results[0].node_id,
                *id,
                "session {i} must be the nearest node for its own embedding"
            );
        }
    }

    /// Regression test for a-llama#1: `open_durable` must repopulate the vector
    /// index from **all** embedding-bearing nodes, not just the CacheEntry /
    /// Fact / Entity dedup labels. A caller-created `Product` node's embedding
    /// must be searchable after a durable reopen.
    #[test]
    fn durable_open_reindexes_product_nodes() {
        let dir = tmp_dir("reindex_product");
        let dim = 4;
        let emb = make_embedding(0, dim);

        // --- Session 1: create a Product node directly via graph_ops, confirm
        //     it is searchable live, then persist + drop. ---
        let product_id = {
            let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();

            let id = store
                .graph_ops()
                .create_node(
                    vec!["Product".to_string(), "footwear_boots".to_string()],
                    serde_json::json!({ "sku": "BOOT-1", "name": "Trail Boot" }),
                    Some(emb.clone()),
                )
                .unwrap();

            // Live insert must index the Product embedding (in-memory path).
            let live = store.vector_index_ref().search(&emb, 1).unwrap();
            assert!(!live.is_empty(), "Product must be searchable before restart");
            assert_eq!(live[0].node_id, id, "Product must be nearest node pre-restart");

            store.persist().unwrap();
            id
        };

        // --- Session 2: reopen and confirm the embedding survived AND was
        //     reinserted into the vector index by open_durable. ---
        {
            let store = KnowledgeStore::open_durable(&dir.0, dim).unwrap();

            // Embedding + label survive on the node itself.
            let node = store.graph_ops().get_node(product_id).unwrap().unwrap();
            assert!(
                node.embedding.is_some(),
                "Product embedding must survive durable reopen on the node"
            );
            assert!(
                node.labels.contains(&"Product".to_string()),
                "reopened node must retain its Product label"
            );

            // The regression: vector search must find the Product after reopen.
            let results = store.vector_index_ref().search(&emb, 1).unwrap();
            assert!(
                !results.is_empty(),
                "open_durable must rebuild the vector index for Product nodes"
            );
            assert_eq!(
                results[0].node_id, product_id,
                "reopened Product must be nearest node for its own embedding"
            );
        }
    }
}
