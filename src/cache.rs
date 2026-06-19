//! Task 5 — Eunomia semantic cache integration (M1, Role 1).
//!
//! [`SemanticCache`] wraps [`eunomia_core::Store`] as a-llama's **hot semantic
//! prompt cache and recent working memory** (DESIGN.md §3, §4.1). On each
//! request the orchestrator embeds the prompt, calls [`SemanticCache::lookup`]
//! (`Store::nearest`), and serves/augments from a hit above a configurable
//! similarity threshold. On each response it calls
//! [`SemanticCache::store_interaction`] (`Store::put`) to persist the
//! `prompt → response` exchange with its prompt embedding and a TTL. A periodic
//! [`SemanticCache::sweep_expired`] (`Store::remove_expired`) evicts stale
//! entries.
//!
//! This is the **semantic-cache role only**. The `eunomia-astraea` bridge that
//! tiers evicted/promoted entries into the AstraeaDB graph as `CacheEntry`
//! nodes is **Task 7 (M2)** and is intentionally not pulled in here — but the
//! eviction path ([`sweep_expired`](SemanticCache::sweep_expired)) returns the
//! removed [`Entry`]s precisely so Task 7 can flush them downstream, and each
//! [`CacheHit`] carries its eunomia entry `id` so a hit can later anchor a
//! `graph_rag_query_anchored` call.
//!
//! Real `eunomia_core` API used (matched exactly, nothing invented):
//! [`Store::new`], [`Store::put`], [`Store::nearest`], [`Store::get`],
//! [`Store::remove_expired`], [`Store::len`]; types [`Namespace`], [`Key`],
//! [`Entry`], [`Metadata`], [`Hit`] (re-exported from the crate root).

use std::time::{SystemTime, UNIX_EPOCH};

use eunomia_core::{Entry, Key, Metadata, Namespace, Store, StoreError};
use serde_json::json;

#[cfg(feature = "durable")]
use std::path::{Path, PathBuf};

#[cfg(feature = "durable")]
use eunomia_core::{FsyncPolicy, Snapshot, Wal, WalRecord};

/// Default cosine-similarity threshold for a cache hit. `Store::nearest`
/// returns scores where `1.0` is identical and higher is closer; equal prompts
/// (equal `MockEngine` embeddings) score `~1.0`, while unrelated prompts score
/// near `0.0`, so this cleanly separates hits from misses.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.85;

/// The namespace a-llama's hot interaction cache lives in.
const CACHE_NAMESPACE: &str = "a-llama:interactions";

/// Tag applied to every interaction entry (alongside the model name), so a
/// future filtered/hybrid query can scope to interactions only (DESIGN §4.1).
const INTERACTION_TAG: &str = "interaction";

/// A semantic cache hit: the cached exchange plus the metadata Task 6 (HTTP
/// orchestrator) needs to serve it and Task 7 (tiering) needs to anchor it.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheHit {
    /// The eunomia entry id — `stable hash(model, normalized_prompt)`. Carried
    /// so a hit can later anchor `graph_rag_query_anchored` once the entry is
    /// flushed to the graph (Task 7).
    pub id: String,
    /// The original prompt that produced the cached response.
    pub prompt: String,
    /// The cached response to serve or inject as a few-shot exemplar.
    pub response: String,
    /// The model that generated the cached response.
    pub model: String,
    /// `created_at` of the cached interaction (unix epoch seconds).
    pub created_at: i64,
    /// Cosine similarity of the looked-up embedding to the cached prompt
    /// embedding (`1.0` = identical). Always `>= threshold` for a returned hit.
    pub score: f32,
}

/// a-llama's hot semantic prompt cache, wrapping [`eunomia_core::Store`].
///
/// In durable mode (built with `--features durable`), every `put` is also
/// appended to a WAL (`durable_wal`) so interactions survive a crash between
/// snapshot writes.  Call [`persist`](SemanticCache::persist) on clean shutdown
/// to atomically snapshot the full store state and sync the WAL.
pub struct SemanticCache {
    store: Store,
    threshold: f32,

    /// WAL writer (interior-mutable via `Mutex`) and snapshot path.
    /// `None` in in-memory mode; `Some(...)` when opened via
    /// [`open_durable`](SemanticCache::open_durable).
    #[cfg(feature = "durable")]
    durable_wal: std::sync::Mutex<Option<Wal>>,

    /// Snapshot path (`data_dir/"cache.snap"`) for [`persist`](SemanticCache::persist).
    #[cfg(feature = "durable")]
    durable_snapshot_path: Option<PathBuf>,
}

impl SemanticCache {
    /// Create a cache pinned to embedding dimension `dim` (768 for
    /// embeddinggemma, Decision 7660) using [`DEFAULT_SIMILARITY_THRESHOLD`].
    pub fn new(dim: usize) -> Self {
        Self::with_threshold(dim, DEFAULT_SIMILARITY_THRESHOLD)
    }

    /// Create a cache pinned to `dim` with a custom hit `threshold`.
    pub fn with_threshold(dim: usize, threshold: f32) -> Self {
        Self {
            store: Store::new(Namespace::new(CACHE_NAMESPACE), dim),
            threshold,
            #[cfg(feature = "durable")]
            durable_wal: std::sync::Mutex::new(None),
            #[cfg(feature = "durable")]
            durable_snapshot_path: None,
        }
    }

    // -----------------------------------------------------------------------
    // Durable backend (Task 8, `durable` feature)
    // -----------------------------------------------------------------------

    /// Open a durable [`SemanticCache`] backed by a WAL + snapshot at
    /// `data_dir/"cache.wal"` and `data_dir/"cache.snap"`.
    ///
    /// ## Recovery on open
    ///
    /// Calls [`eunomia_core::recover`] which:
    /// 1. Loads the snapshot at `cache.snap` if present (all entries at last
    ///    snapshot time, HNSW rebuilt from stored embeddings).
    /// 2. Replays the WAL tail at `cache.wal` (entries added/deleted since the
    ///    snapshot) on top of the snapshot.
    ///
    /// ## Durability during operation
    ///
    /// Every [`store_interaction_at`](Self::store_interaction_at) call appends a
    /// `WalRecord::Put` to the WAL **before** the in-memory `Store::put`.
    /// Every [`sweep_expired_at`](Self::sweep_expired_at) call appends
    /// `WalRecord::Delete` records for evicted entries.  WAL writes use
    /// `FsyncPolicy::Interval(500ms)` (group commit) for throughput.
    ///
    /// ## Snapshot on persist
    ///
    /// Call [`persist`](Self::persist) on clean shutdown.  It atomically writes
    /// the full in-memory store to `cache.snap` (capturing all entries regardless
    /// of WAL), then syncs the WAL to disk.  On the next `open_durable`, `recover`
    /// will load the snapshot and replay any WAL tail after it.
    #[cfg(feature = "durable")]
    pub fn open_durable(data_dir: &Path, dim: usize) -> std::io::Result<Self> {
        use std::time::Duration;

        let snapshot_path = data_dir.join("cache.snap");
        let wal_path = data_dir.join("cache.wal");

        // Recover: snapshot + WAL tail.
        let store = eunomia_core::recover(
            Namespace::new(CACHE_NAMESPACE),
            dim,
            &snapshot_path,
            &wal_path,
        )?;

        // Open the WAL for ongoing appends (create if absent).
        let wal = Wal::open(&wal_path, FsyncPolicy::Interval(Duration::from_millis(500)))?;

        eprintln!(
            "a-llama: durable cache opened from {data_dir:?} ({} entries recovered)",
            store.len()
        );

        Ok(Self {
            store,
            threshold: DEFAULT_SIMILARITY_THRESHOLD,
            durable_wal: std::sync::Mutex::new(Some(wal)),
            durable_snapshot_path: Some(snapshot_path),
        })
    }

    /// Atomically snapshot the in-memory store and sync the WAL to disk.
    ///
    /// The snapshot captures **all** current entries regardless of WAL state,
    /// so a clean-shutdown persist followed by a restart recovers the full
    /// in-memory state even if some WAL writes hadn't been fsynced.
    ///
    /// A `None` WAL (in-memory mode) is a no-op.
    #[cfg(feature = "durable")]
    pub fn persist(&self) -> std::io::Result<()> {
        let mut guard = self.durable_wal.lock().unwrap();
        if let (Some(snap_path), Some(wal)) =
            (&self.durable_snapshot_path, guard.as_mut())
        {
            Snapshot::capture(&self.store).write(snap_path)?;
            let _ = wal.sync(); // best-effort; snapshot already covers all state
        }
        Ok(())
    }

    /// The embedding dimension this cache is pinned to.
    pub fn dim(&self) -> usize {
        self.store.dim()
    }

    /// The current similarity threshold for a hit.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// The namespace this hot cache stores interactions under. Task 7 (durable
    /// tiering) reuses this exact string as the `CacheEntry` graph namespace so
    /// the durable node carries the same `(namespace, eunomia_id)` identity as
    /// the hot entry it was flushed from.
    pub fn namespace(&self) -> &'static str {
        CACHE_NAMESPACE
    }

    /// Adjust the hit threshold at runtime.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Number of cached interactions.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Probe the cache for a semantically equivalent prior prompt. Returns the
    /// single best match **only if** its cosine similarity is at or above the
    /// configured threshold **and** it has not outlived its TTL as of now;
    /// otherwise `None` (a miss). Calls [`lookup_at`](Self::lookup_at) with
    /// `SystemTime::now()`.
    ///
    /// A dimension mismatch (wrong-length `prompt_embedding`) is treated as a
    /// miss rather than an error — the orchestrator should never feed a
    /// wrong-dim vector, and a cache probe must not fail a request.
    pub fn lookup(&self, prompt_embedding: &[f32]) -> Option<CacheHit> {
        self.lookup_at(prompt_embedding, SystemTime::now())
    }

    /// Deterministic-testable [`lookup`](Self::lookup): evaluates TTL expiry
    /// against the caller-supplied `now` instead of the wall clock.
    ///
    /// `Store::nearest` ranks by vector similarity alone and **ignores TTL**, so
    /// an expired interaction would otherwise be served until an explicit
    /// [`sweep_expired`](Self::sweep_expired). This guards against that: the best
    /// match is returned only if it is *not* expired as of `now`, using the exact
    /// rule `Store::remove_expired` applies — `Metadata::is_expired(now)`
    /// (no `ttl_secs` ⇒ never expires; otherwise expired once
    /// `now - created_at >= ttl_secs`). The entry is left in the store; only the
    /// hit is suppressed.
    pub fn lookup_at(&self, prompt_embedding: &[f32], now: SystemTime) -> Option<CacheHit> {
        let hits = self.store.nearest(&prompt_embedding.to_vec(), 1).ok()?;
        let hit = hits.into_iter().next()?;
        if hit.score < self.threshold {
            return None;
        }
        // Mirror `Store::remove_expired` (`Metadata::is_expired`): never serve a
        // TTL-expired entry even though `Store::nearest` would still return it.
        if hit.entry.metadata.is_expired(now) {
            return None;
        }
        cache_hit_from(&hit.entry, hit.score)
    }

    /// Store a `prompt → response` interaction (DESIGN §4.1). Uses
    /// `SystemTime::now()` for `created_at`; see [`store_interaction_at`] for a
    /// deterministic-testable variant. Returns the eunomia entry id.
    ///
    /// [`store_interaction_at`]: SemanticCache::store_interaction_at
    pub fn store_interaction(
        &mut self,
        prompt: &str,
        response: &str,
        model: &str,
        prompt_embedding: &[f32],
        ttl_secs: Option<u64>,
    ) -> Result<String, StoreError> {
        self.store_interaction_at(
            prompt,
            response,
            model,
            prompt_embedding,
            ttl_secs,
            SystemTime::now(),
        )
    }

    /// Deterministic-testable [`store_interaction`](Self::store_interaction):
    /// `created_at` (both the JSON `value.created_at` and `Metadata.created_at`
    /// that drives TTL expiry) is taken from the caller-supplied `created_at`
    /// instead of the wall clock.
    pub fn store_interaction_at(
        &mut self,
        prompt: &str,
        response: &str,
        model: &str,
        prompt_embedding: &[f32],
        ttl_secs: Option<u64>,
        created_at: SystemTime,
    ) -> Result<String, StoreError> {
        let id = stable_interaction_id(model, prompt);
        let created_at_secs = unix_secs(created_at);

        let entry = Entry {
            key: Key::with_embedding(id.clone(), prompt_embedding.to_vec()),
            value: json!({
                "prompt": prompt,
                "response": response,
                "model": model,
                "created_at": created_at_secs,
            }),
            metadata: Metadata {
                tags: vec![INTERACTION_TAG.to_string(), model.to_string()],
                ttl_secs,
                created_at,
                last_accessed: created_at,
            },
        };

        // Durable: WAL-log the put BEFORE the in-memory write so a crash between
        // the two leaves the entry recoverable from the WAL on next open.
        // WAL write errors are swallowed (logged): the in-memory store is always
        // updated, and the snapshot taken by `persist()` covers any WAL gaps.
        #[cfg(feature = "durable")]
        {
            let mut guard = self.durable_wal.lock().unwrap();
            if let Some(wal) = guard.as_mut() {
                if let Err(e) = wal.append(&WalRecord::Put(entry.clone())) {
                    eprintln!("a-llama: warning: cache WAL put failed: {e}");
                }
            }
        }

        self.store.put(entry)?;
        Ok(id)
    }

    /// Evict every entry whose TTL has elapsed as of now, returning the removed
    /// [`Entry`]s. Delegates to `Store::remove_expired(SystemTime::now())`. The
    /// returned entries are the seam Task 7 (M2) uses to flush durable working
    /// memory into the AstraeaDB graph before it is dropped.
    pub fn sweep_expired(&mut self) -> Vec<Entry> {
        self.sweep_expired_at(SystemTime::now())
    }

    /// Deterministic-testable [`sweep_expired`](Self::sweep_expired): evicts
    /// relative to the caller-supplied `now`.
    pub fn sweep_expired_at(&mut self, now: SystemTime) -> Vec<Entry> {
        let evicted = self.store.remove_expired(now);

        // Durable: WAL-log each deletion so it is replayed on restart.
        #[cfg(feature = "durable")]
        if !evicted.is_empty() {
            let mut guard = self.durable_wal.lock().unwrap();
            if let Some(wal) = guard.as_mut() {
                for e in &evicted {
                    if let Err(err) = wal.append(&WalRecord::Delete(e.key.id.clone())) {
                        eprintln!("a-llama: warning: cache WAL delete failed: {err}");
                    }
                }
            }
        }

        evicted
    }
}

/// Stable id for an interaction: `interaction:<model>:<hash>` where the hash is
/// FNV-1a over the model and the *normalized* prompt (trimmed, lowercased,
/// internal whitespace collapsed). Equal `(model, prompt)` pairs map to the
/// same id, so re-storing an identical exchange replaces it in place rather
/// than duplicating (DESIGN §4.1: `key.id = stable hash(model, normalized_prompt)`).
fn stable_interaction_id(model: &str, prompt: &str) -> String {
    let normalized = normalize_prompt(prompt);
    // FNV-1a 64-bit over model + NUL separator + normalized prompt.
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

/// Normalize a prompt for stable hashing: trim, lowercase, collapse runs of
/// whitespace to a single space.
fn normalize_prompt(prompt: &str) -> String {
    prompt.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Unix epoch seconds for `t` (0 if before the epoch — only happens with a
/// pathological clock).
fn unix_secs(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Reconstruct a [`CacheHit`] from a stored [`Entry`] and its score. Returns
/// `None` if the entry's JSON value is missing the expected fields (defensive —
/// every entry written by this module has them).
fn cache_hit_from(entry: &Entry, score: f32) -> Option<CacheHit> {
    let v = &entry.value;
    Some(CacheHit {
        id: entry.key.id.clone(),
        prompt: v.get("prompt")?.as_str()?.to_string(),
        response: v.get("response")?.as_str()?.to_string(),
        model: v.get("model")?.as_str()?.to_string(),
        created_at: v.get("created_at")?.as_i64()?,
        score,
    })
}

// ---------------------------------------------------------------------------
// Durable-mode acceptance tests (Task 8)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "durable"))]
mod durable_tests {
    use super::*;
    use crate::inference::{InferenceEngine, MockEngine, EMBEDDING_DIM};

    fn embed_sync(text: &str) -> Vec<f32> {
        let engine = MockEngine::new();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(engine.embed(text))
            .unwrap()
    }

    fn tmp_dir(label: &str) -> TmpDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir()
            .join(format!("a_llama_cache_durable_{label}_{nanos}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TmpDir(path)
    }

    struct TmpDir(std::path::PathBuf);
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A stored interaction must be recalled after a restart (snapshot recovery).
    #[test]
    fn cache_interaction_survives_restart() {
        let dir = tmp_dir("cache_restart");
        let prompt = "What is AstraeaDB?";
        let response = "An embedded graph database.";
        let emb = embed_sync(prompt);

        // Session 1: store and persist.
        {
            let mut cache = SemanticCache::open_durable(&dir.0, EMBEDDING_DIM).unwrap();
            cache
                .store_interaction(prompt, response, "mock-gguf", &emb, None)
                .unwrap();
            assert_eq!(cache.len(), 1);
            cache.persist().unwrap();
        }

        // Session 2: reopen and recall.
        {
            let cache = SemanticCache::open_durable(&dir.0, EMBEDDING_DIM).unwrap();
            assert_eq!(cache.len(), 1, "entry must survive restart");
            let hit = cache.lookup(&emb).expect("stored interaction must hit after restart");
            assert_eq!(hit.response, response, "response must be intact");
            assert_eq!(hit.prompt, prompt, "prompt must be intact");
            assert_eq!(hit.model, "mock-gguf");
        }
    }

    /// TTL-evicted entries are WAL-logged as deletes; after restart they are gone.
    #[test]
    fn evicted_entries_not_recalled_after_restart() {
        use std::time::Duration;

        let dir = tmp_dir("cache_evict");
        let emb = embed_sync("ephemeral prompt");
        let live_emb = embed_sync("durable prompt");

        {
            let mut cache = SemanticCache::open_durable(&dir.0, EMBEDDING_DIM).unwrap();
            // Store one entry that has already expired.
            let past = std::time::SystemTime::now() - Duration::from_secs(120);
            cache
                .store_interaction_at("ephemeral prompt", "resp", "mock-gguf", &emb, Some(1), past)
                .unwrap();
            // Store one that won't expire.
            cache
                .store_interaction("durable prompt", "resp2", "mock-gguf", &live_emb, None)
                .unwrap();
            // Sweep (WAL-logs the delete) then persist (snapshot).
            let evicted = cache.sweep_expired();
            assert_eq!(evicted.len(), 1);
            cache.persist().unwrap();
        }

        // After restart, only the live entry must be present.
        {
            let cache = SemanticCache::open_durable(&dir.0, EMBEDDING_DIM).unwrap();
            assert_eq!(cache.len(), 1, "evicted entry must not reappear after restart");
            assert!(cache.lookup(&emb).is_none(), "expired entry is gone");
            assert!(cache.lookup(&live_emb).is_some(), "live entry remains");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceEngine, MockEngine, EMBEDDING_DIM};
    use std::time::Duration;

    /// Embed `text` with the deterministic `MockEngine` (768-dim, L2-normalized).
    fn embed(text: &str) -> Vec<f32> {
        let engine = MockEngine::new();
        // No runtime active in a plain #[test]; drive the async embed to done.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(engine.embed(text))
            .unwrap()
    }

    #[test]
    fn store_then_lookup_same_prompt_hits() {
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let prompt = "What is the capital of France?";
        let emb = embed(prompt);

        let id = cache
            .store_interaction(prompt, "Paris.", "mock-gguf", &emb, Some(3600))
            .unwrap();
        assert_eq!(cache.len(), 1);

        // Looking up the same prompt embedding returns a hit above threshold.
        let hit = cache.lookup(&emb).expect("same prompt must hit");
        assert_eq!(hit.id, id);
        assert_eq!(hit.response, "Paris.");
        assert_eq!(hit.prompt, prompt);
        assert_eq!(hit.model, "mock-gguf");
        assert!(
            hit.score >= cache.threshold(),
            "hit score {} must clear threshold {}",
            hit.score,
            cache.threshold()
        );
        assert!(hit.score > 0.99, "identical embedding => near-1.0 score");
    }

    #[test]
    fn near_identical_embedding_hits() {
        // MockEngine embeddings are deterministic per-text, so "paraphrase"
        // testing realistically means a slightly perturbed vector (a near-
        // duplicate prompt that embeds to an almost-identical point).
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let emb = embed("Tell me about the Rust borrow checker.");
        cache
            .store_interaction("Tell me about the Rust borrow checker.", "It enforces ownership.", "mock-gguf", &emb, None)
            .unwrap();

        // Perturb a few dimensions slightly and renormalize — cosine stays high.
        let mut near = emb.clone();
        for i in 0..8 {
            near[i] += 0.001;
        }
        let norm: f32 = near.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in near.iter_mut() {
            *x /= norm;
        }

        let hit = cache.lookup(&near).expect("near-identical vector must hit");
        assert_eq!(hit.response, "It enforces ownership.");
    }

    #[test]
    fn dissimilar_prompt_misses() {
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let stored = embed("How do I write a for loop in Python?");
        cache
            .store_interaction("How do I write a for loop in Python?", "Use `for x in ...`.", "mock-gguf", &stored, None)
            .unwrap();

        // A completely different prompt embeds to a near-orthogonal vector.
        let other = embed("Explain the theory of general relativity.");
        assert!(
            cache.lookup(&other).is_none(),
            "dissimilar embedding must not hit"
        );
    }

    #[test]
    fn ttl_expired_entry_is_swept() {
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let emb = embed("ephemeral fact");

        // Stored 60s in the past with a 1s TTL => already expired.
        let created = SystemTime::now() - Duration::from_secs(60);
        cache
            .store_interaction_at("ephemeral fact", "value", "mock-gguf", &emb, Some(1), created)
            .unwrap();
        // Live entry with no TTL must survive the sweep.
        let live = embed("durable fact");
        cache
            .store_interaction("durable fact", "value2", "mock-gguf", &live, None)
            .unwrap();
        assert_eq!(cache.len(), 2);

        let evicted = cache.sweep_expired();
        assert_eq!(evicted.len(), 1, "only the expired entry is swept");
        assert_eq!(cache.len(), 1);

        // The expired entry no longer hits; the live one still does.
        assert!(cache.lookup(&emb).is_none(), "swept entry is gone");
        assert!(cache.lookup(&live).is_some(), "live entry survives");
    }

    #[test]
    fn lookup_does_not_serve_ttl_expired_entry() {
        // Issue 2049: `Store::nearest` ranks by similarity alone and ignores
        // TTL, so without a guard an expired interaction is served indefinitely
        // (the M1 orchestrator never sweeps). `lookup_at` must suppress it.
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let emb = embed("ephemeral cached answer");

        // Stored at t0 with a 10s TTL.
        let t0 = SystemTime::now();
        cache
            .store_interaction_at(
                "ephemeral cached answer",
                "stale response",
                "mock-gguf",
                &emb,
                Some(10),
                t0,
            )
            .unwrap();

        // Within the TTL → still served.
        assert!(
            cache.lookup_at(&emb, t0 + Duration::from_secs(5)).is_some(),
            "within TTL the entry must still hit"
        );

        // Past the TTL → suppressed, even though `Store::nearest` still holds it.
        assert!(
            cache.lookup_at(&emb, t0 + Duration::from_secs(20)).is_none(),
            "a TTL-expired entry must not be served"
        );

        // `lookup_at` only suppresses the hit; it does not evict (that's
        // `sweep_expired`'s job) — the entry is still physically present.
        assert_eq!(cache.len(), 1, "lookup_at must not remove the expired entry");

        // An entry with no TTL is never suppressed by expiry.
        let live = embed("immortal cached answer");
        cache
            .store_interaction_at(
                "immortal cached answer",
                "durable response",
                "mock-gguf",
                &live,
                None,
                t0,
            )
            .unwrap();
        assert!(
            cache
                .lookup_at(&live, t0 + Duration::from_secs(100_000))
                .is_some(),
            "an entry without a TTL never expires"
        );
    }

    #[test]
    fn restoring_same_interaction_replaces_in_place() {
        let mut cache = SemanticCache::new(EMBEDDING_DIM);
        let emb = embed("dedupe me");
        let id1 = cache
            .store_interaction("dedupe me", "v1", "mock-gguf", &emb, None)
            .unwrap();
        // Same model + (whitespace/case-different) prompt => same stable id.
        let id2 = cache
            .store_interaction("  Dedupe   Me  ", "v2", "mock-gguf", &emb, None)
            .unwrap();
        assert_eq!(id1, id2, "normalized prompt yields a stable id");
        assert_eq!(cache.len(), 1, "re-store replaces rather than duplicates");
        assert_eq!(cache.lookup(&emb).unwrap().response, "v2");
    }
}
