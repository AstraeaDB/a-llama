//! The in-memory storage engine: exact K/V map + vector index.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{Embedding, Entry, Namespace, NamespaceConfig};
use crate::query::Query;
use crate::vector::{ActiveIndex, VectorIndex};

/// Over-fetch multiplier for hybrid search: pull `k * HYBRID_FANOUT` candidates
/// from the vector index before applying the filter, so post-filter results
/// still reach `k`.
const HYBRID_FANOUT: usize = 4;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("key not found: {0}")]
    NotFound(String),
    #[error("embedding dimension mismatch in namespace '{namespace}': expected {expected}, got {got}")]
    DimensionMismatch {
        namespace: String,
        expected: usize,
        got: usize,
    },
}

/// A scored search hit: the matching entry and its cosine similarity (1.0 =
/// identical; higher is closer).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub entry: Entry,
    pub score: f32,
}

/// Eunomia's per-namespace working-memory store: an exact O(1) key/value map
/// plus an HNSW vector index for approximate nearest-neighbor search.
///
/// The embedding dimension is pinned per namespace (Decision 7626) and
/// validated on insert; entries without an embedding are still allowed (pure
/// K/V). The vector backend sits behind [`VectorIndex`] so it can be swapped.
pub struct Store {
    namespace: Namespace,
    dim: usize,
    kv: HashMap<String, Entry>,
    index: ActiveIndex,
}

impl Store {
    /// Create an empty store for `namespace`, pinned to embedding dimension
    /// `dim`. Entries *with* an embedding must match `dim`; entries without one
    /// are stored as pure K/V.
    pub fn new(namespace: Namespace, dim: usize) -> Self {
        Self {
            namespace,
            dim,
            kv: HashMap::new(),
            index: ActiveIndex::new(dim),
        }
    }

    /// Create a store from a [`NamespaceConfig`] — the config-driven entry point
    /// used at namespace-creation time (Decision 7637). [`Store::new`] remains
    /// the lower-level primitive.
    pub fn with_config(config: NamespaceConfig) -> Self {
        Self::new(config.namespace, config.dim)
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    fn check_dim(&self, got: usize) -> Result<(), StoreError> {
        if got != self.dim {
            return Err(StoreError::DimensionMismatch {
                namespace: self.namespace.0.clone(),
                expected: self.dim,
                got,
            });
        }
        Ok(())
    }

    /// Insert or replace an entry. O(1) for the K/V map; also indexes the
    /// embedding if present. Returns [`StoreError::DimensionMismatch`] if the
    /// entry carries an embedding of the wrong dimension.
    pub fn put(&mut self, entry: Entry) -> Result<(), StoreError> {
        if let Some(emb) = &entry.key.embedding {
            self.check_dim(emb.len())?;
            self.index.insert(&entry.key.id, emb);
        }
        self.kv.insert(entry.key.id.clone(), entry);
        Ok(())
    }

    /// Exact key lookup. O(1).
    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.kv.get(id)
    }

    /// Remove an entry by id, returning it if present. Also drops it from the
    /// vector index.
    pub fn remove(&mut self, id: &str) -> Option<Entry> {
        self.index.remove(id);
        self.kv.remove(id)
    }

    /// Approximate nearest-neighbor search over embeddings. Returns up to `k`
    /// hits sorted by similarity (best first). The query dimension must match
    /// the namespace dimension.
    pub fn nearest(&self, query: &Embedding, k: usize) -> Result<Vec<Hit>, StoreError> {
        self.check_dim(query.len())?;
        let hits = self
            .index
            .search(query, k)
            .into_iter()
            .filter_map(|(id, score)| {
                self.kv.get(&id).map(|entry| Hit {
                    entry: entry.clone(),
                    score,
                })
            })
            .collect();
        Ok(hits)
    }

    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    /// Number of entries currently indexed for vector search (those carrying an
    /// embedding). May be less than [`Store::len`], which also counts pure
    /// key/value entries.
    pub fn vector_count(&self) -> usize {
        self.index.len()
    }

    /// Iterate over all stored entries (used for snapshotting).
    pub fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.kv.values()
    }

    /// Remove every entry that has expired as of `now` (per
    /// [`Metadata::is_expired`](crate::model::Metadata::is_expired)), returning
    /// the removed entries so a caller can flush them to a cold tier. Each is
    /// also dropped from the vector index. Entries without a TTL are never
    /// touched. This is the scan the server's background TTL sweeper drives.
    pub fn remove_expired(&mut self, now: SystemTime) -> Vec<Entry> {
        let expired: Vec<String> = self
            .kv
            .iter()
            .filter(|(_, entry)| entry.metadata.is_expired(now))
            .map(|(id, _)| id.clone())
            .collect();
        expired
            .into_iter()
            .filter_map(|id| self.remove(&id))
            .collect()
    }

    /// Update an entry's `last_accessed` timestamp. Returns `false` if the id is
    /// absent. Used when replaying a [`Touch`](crate::persistence::WalRecord::Touch)
    /// record.
    pub fn touch(&mut self, id: &str, at: SystemTime) -> bool {
        match self.kv.get_mut(id) {
            Some(entry) => {
                entry.metadata.last_accessed = at;
                true
            }
            None => false,
        }
    }

    /// Execute a planned [`Query`] against this store, returning hits best-first
    /// (capped at the query's `k` for vector/hybrid queries). An `Exact` lookup
    /// yields at most one hit with `score = 1.0`. Hybrid search over-fetches
    /// from the vector index, applies the [`Filter`](crate::query::Filter), then
    /// truncates to `k`.
    pub fn execute(&self, query: &Query) -> Result<Vec<Hit>, StoreError> {
        match query {
            Query::Exact { id } => Ok(self
                .get(id)
                .map(|entry| Hit {
                    entry: entry.clone(),
                    score: 1.0,
                })
                .into_iter()
                .collect()),
            Query::Semantic { embedding, k } => self.nearest(embedding, *k),
            Query::Hybrid {
                embedding,
                k,
                filter,
            } => {
                let now = SystemTime::now();
                let overfetch = k.saturating_mul(HYBRID_FANOUT).max(*k);
                let mut hits = self.nearest(embedding, overfetch)?;
                hits.retain(|h| filter.matches(&h.entry, now));
                hits.truncate(*k);
                Ok(hits)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Key;
    use serde_json::json;

    fn store3() -> Store {
        Store::new(Namespace::new("test"), 3)
    }

    fn entry(id: &str, emb: Option<Vec<f32>>, value: serde_json::Value) -> Entry {
        Entry {
            key: Key {
                id: id.into(),
                embedding: emb,
            },
            value,
            metadata: Default::default(),
        }
    }

    #[test]
    fn put_then_get_roundtrips() {
        let mut store = store3();
        store
            .put(entry("user:pref:lang", None, json!({"language": "rust"})))
            .unwrap();
        let got = store.get("user:pref:lang").expect("entry present");
        assert_eq!(got.value, json!({"language": "rust"}));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut store = store3();
        let err = store
            .put(entry("k", Some(vec![1.0, 2.0]), json!(null)))
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }
        ));
        // bad query dimension is rejected too.
        assert!(store.nearest(&vec![1.0, 2.0], 1).is_err());
    }

    #[test]
    fn config_driven_namespaces_have_independent_dims() {
        let mut a = Store::with_config(NamespaceConfig::new(Namespace::new("a"), 3));
        let mut b = Store::with_config(NamespaceConfig::new(Namespace::new("b"), 4));
        assert_eq!(a.dim(), 3);
        assert_eq!(b.dim(), 4);
        // A 4-dim vector is rejected by the dim-3 namespace, accepted by dim-4.
        assert!(a
            .put(entry("x", Some(vec![1.0, 2.0, 3.0, 4.0]), json!(null)))
            .is_err());
        assert!(b
            .put(entry("y", Some(vec![1.0, 2.0, 3.0, 4.0]), json!(null)))
            .is_ok());
    }

    #[test]
    fn nearest_recalls_closest() {
        let mut store = store3();
        store
            .put(entry("a", Some(vec![1.0, 0.0, 0.0]), json!("a")))
            .unwrap();
        store
            .put(entry("b", Some(vec![0.0, 1.0, 0.0]), json!("b")))
            .unwrap();
        store
            .put(entry("c", Some(vec![0.9, 0.1, 0.0]), json!("c")))
            .unwrap();

        let hits = store.nearest(&vec![1.0, 0.0, 0.0], 2).unwrap();
        assert!(hits.len() <= 2 && !hits.is_empty());
        assert_eq!(hits[0].entry.key.id, "a", "identical vector first");
        if hits.len() == 2 {
            assert!(hits[0].score >= hits[1].score, "sorted by similarity");
        }
    }

    #[test]
    fn remove_drops_from_kv_and_index() {
        let mut store = store3();
        store
            .put(entry("a", Some(vec![1.0, 0.0, 0.0]), json!("a")))
            .unwrap();
        assert!(store.remove("a").is_some());
        assert_eq!(store.len(), 0);
        let hits = store.nearest(&vec![1.0, 0.0, 0.0], 5).unwrap();
        assert!(hits.iter().all(|h| h.entry.key.id != "a"));
    }

    #[test]
    fn remove_expired_drops_only_stale_entries() {
        use std::time::Duration;

        let mut store = store3();

        // Live: no TTL.
        store
            .put(entry("live", Some(vec![1.0, 0.0, 0.0]), json!("live")))
            .unwrap();
        // Fresh + short TTL -> not yet expired.
        let mut fresh = entry("fresh", None, json!("fresh"));
        fresh.metadata.ttl_secs = Some(60);
        store.put(fresh).unwrap();
        // Stale: created 60s ago with a 1s TTL.
        let mut stale = entry("stale", Some(vec![0.0, 1.0, 0.0]), json!("stale"));
        stale.metadata.ttl_secs = Some(1);
        stale.metadata.created_at = SystemTime::now() - Duration::from_secs(60);
        store.put(stale).unwrap();

        let removed = store.remove_expired(SystemTime::now());
        assert_eq!(removed.len(), 1, "only the stale entry is removed");
        assert_eq!(removed[0].key.id, "stale");

        assert!(store.get("live").is_some());
        assert!(store.get("fresh").is_some());
        assert!(store.get("stale").is_none());
        // ...and the stale embedding is gone from the vector index too.
        let hits = store.nearest(&vec![0.0, 1.0, 0.0], 5).unwrap();
        assert!(hits.iter().all(|h| h.entry.key.id != "stale"));
    }

    #[test]
    fn execute_exact_returns_single_hit() {
        let mut store = store3();
        store.put(entry("k", None, json!(42))).unwrap();
        let hits = store.execute(&Query::Exact { id: "k".into() }).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.value, json!(42));
        assert_eq!(hits[0].score, 1.0);
        // missing key => no hits.
        assert!(store
            .execute(&Query::Exact { id: "nope".into() })
            .unwrap()
            .is_empty());
    }

    #[test]
    fn execute_hybrid_filters_tags_and_expiry() {
        use crate::query::Filter;
        use std::time::Duration;

        let mut store = store3();

        let mut good = entry("good", Some(vec![1.0, 0.0, 0.0]), json!("good"));
        good.metadata.tags = vec!["python".into()];
        store.put(good).unwrap();

        // matching tag but expired (created 60s ago, ttl 1s).
        let mut stale = entry("stale", Some(vec![0.99, 0.01, 0.0]), json!("stale"));
        stale.metadata.tags = vec!["python".into()];
        stale.metadata.ttl_secs = Some(1);
        stale.metadata.created_at = SystemTime::now() - Duration::from_secs(60);
        store.put(stale).unwrap();

        // fresh + close but wrong tag.
        let mut offtopic = entry("offtopic", Some(vec![1.0, 0.0, 0.0]), json!("offtopic"));
        offtopic.metadata.tags = vec!["rust".into()];
        store.put(offtopic).unwrap();

        let q = Query::Hybrid {
            embedding: vec![1.0, 0.0, 0.0],
            k: 5,
            filter: Filter {
                tags: vec!["python".into()],
                exclude_expired: true,
            },
        };
        let ids: Vec<String> = store
            .execute(&q)
            .unwrap()
            .into_iter()
            .map(|h| h.entry.key.id)
            .collect();
        assert!(ids.contains(&"good".to_string()), "matching live entry kept");
        assert!(!ids.contains(&"stale".to_string()), "expired excluded");
        assert!(!ids.contains(&"offtopic".to_string()), "wrong tag excluded");
    }
}
