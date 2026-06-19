//! Internal vector-index abstraction for [`Store`](crate::store::Store).
//!
//! `Store` answers nearest-neighbor queries through the [`VectorIndex`] trait,
//! so the ANN backend can be swapped without touching callers. Exactly one
//! backend is compiled, selected by a Cargo feature:
//!
//! - **default** — [`HnswIndex`], the pure-Rust `hnsw_rs` backend. No AstraeaDB
//!   dependency; deletes are tombstoned (rebuild on snapshot to compact).
//! - **`vector-astraea`** — `AstraeaVectorBackend`, the reused
//!   `astraea_vector::HnswVectorIndex` (Decision 7632). Real connection-repair
//!   deletes, tunable params, and serde persistence. Its only AstraeaDB
//!   transitive dep is the runtime-free `astraea-core` types crate — no graph or
//!   storage engine — so runtime independence holds.
//!
//! [`ActiveIndex`] is the compile-time-selected concrete backend `Store` holds
//! (no dynamic dispatch on the hot path). Scores are cosine **similarity**
//! (1.0 = identical; higher is closer) regardless of backend.

/// Maps Eunomia string ids to embeddings and answers k-NN queries.
pub(crate) trait VectorIndex: Send + Sync {
    /// Insert or replace the embedding for `id`.
    fn insert(&mut self, id: &str, embedding: &[f32]);
    /// Drop `id` from the index.
    fn remove(&mut self, id: &str);
    /// Return up to `k` nearest ids with their cosine similarity, best first.
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>;
    /// Number of live vectors.
    fn len(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Default backend: hnsw_rs (no AstraeaDB dependency).
// ---------------------------------------------------------------------------
#[cfg(not(feature = "vector-astraea"))]
mod hnsw_backend {
    use std::collections::{HashMap, HashSet};

    use hnsw_rs::prelude::*;

    use super::VectorIndex;

    // Conservative HNSW params; tunable later (DESIGN.md).
    const MAX_NB_CONNECTION: usize = 16;
    const EF_CONSTRUCTION: usize = 200;
    const MAX_LAYER: usize = 16;
    const CAPACITY_HINT: usize = 10_000;
    const MIN_EF_SEARCH: usize = 64;

    /// HNSW-backed [`VectorIndex`] over `hnsw_rs`.
    ///
    /// `hnsw_rs` indexes by `usize` point ids and has no real delete, so this
    /// owns the string-id ↔ point-id mapping and tombstones removed points,
    /// filtering them from search results (rebuild on snapshot to compact).
    pub(crate) struct HnswIndex {
        hnsw: Hnsw<'static, f32, DistCosine>,
        next_point: usize,
        point_of: HashMap<String, usize>,
        id_of: HashMap<usize, String>,
        tombstones: HashSet<usize>,
    }

    impl HnswIndex {
        /// `_dim` is accepted for a uniform backend constructor signature;
        /// `hnsw_rs` does not need it up front (`Store` validates dimension).
        pub(crate) fn new(_dim: usize) -> Self {
            let hnsw = Hnsw::<f32, DistCosine>::new(
                MAX_NB_CONNECTION,
                CAPACITY_HINT,
                MAX_LAYER,
                EF_CONSTRUCTION,
                DistCosine {},
            );
            Self {
                hnsw,
                next_point: 0,
                point_of: HashMap::new(),
                id_of: HashMap::new(),
                tombstones: HashSet::new(),
            }
        }
    }

    impl VectorIndex for HnswIndex {
        fn insert(&mut self, id: &str, embedding: &[f32]) {
            if let Some(old) = self.point_of.remove(id) {
                self.id_of.remove(&old);
                self.tombstones.insert(old);
            }
            let point = self.next_point;
            self.next_point += 1;
            self.hnsw.insert((embedding, point));
            self.point_of.insert(id.to_string(), point);
            self.id_of.insert(point, id.to_string());
        }

        fn remove(&mut self, id: &str) {
            if let Some(point) = self.point_of.remove(id) {
                self.id_of.remove(&point);
                self.tombstones.insert(point);
            }
        }

        fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
            if k == 0 || self.point_of.is_empty() {
                return Vec::new();
            }
            let want = k + self.tombstones.len();
            let ef_search = want.max(MIN_EF_SEARCH);
            let mut out = Vec::with_capacity(k);
            for n in self.hnsw.search(query, want, ef_search) {
                if self.tombstones.contains(&n.d_id) {
                    continue;
                }
                if let Some(id) = self.id_of.get(&n.d_id) {
                    out.push((id.clone(), 1.0 - n.distance));
                    if out.len() == k {
                        break;
                    }
                }
            }
            out
        }

        fn len(&self) -> usize {
            self.point_of.len()
        }
    }
}

#[cfg(not(feature = "vector-astraea"))]
pub(crate) use hnsw_backend::HnswIndex;

/// The vector backend `Store` uses, selected at compile time. Default:
/// `hnsw_rs`.
#[cfg(not(feature = "vector-astraea"))]
pub(crate) type ActiveIndex = HnswIndex;

// ---------------------------------------------------------------------------
// Reused backend: astraea-vector (Decision 7632), feature-gated.
// ---------------------------------------------------------------------------
#[cfg(feature = "vector-astraea")]
mod astraea_backend {
    use std::collections::HashMap;

    use astraea_core::traits::VectorIndex as AstraeaVectorIndex;
    use astraea_core::types::{DistanceMetric, NodeId};
    use astraea_vector::HnswVectorIndex;

    use super::VectorIndex;

    /// [`VectorIndex`] backed by `astraea_vector::HnswVectorIndex`. Removals are
    /// real connection-repair deletes (no tombstones). Owns the string-id ↔
    /// `NodeId(u64)` mapping; converts cosine distance to similarity.
    pub(crate) struct AstraeaVectorBackend {
        index: HnswVectorIndex,
        next_id: u64,
        node_of: HashMap<String, u64>,
        id_of: HashMap<u64, String>,
    }

    impl AstraeaVectorBackend {
        pub(crate) fn new(dim: usize) -> Self {
            Self {
                index: HnswVectorIndex::new(dim, DistanceMetric::Cosine),
                next_id: 0,
                node_of: HashMap::new(),
                id_of: HashMap::new(),
            }
        }
    }

    impl VectorIndex for AstraeaVectorBackend {
        fn insert(&mut self, id: &str, embedding: &[f32]) {
            if let Some(old) = self.node_of.remove(id) {
                let _ = self.index.remove(NodeId(old));
                self.id_of.remove(&old);
            }
            let nid = self.next_id;
            self.next_id += 1;
            let _ = self.index.insert(NodeId(nid), embedding);
            self.node_of.insert(id.to_string(), nid);
            self.id_of.insert(nid, id.to_string());
        }

        fn remove(&mut self, id: &str) {
            if let Some(nid) = self.node_of.remove(id) {
                let _ = self.index.remove(NodeId(nid));
                self.id_of.remove(&nid);
            }
        }

        fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
            if k == 0 {
                return Vec::new();
            }
            self.index
                .search(query, k)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|r| {
                    self.id_of
                        .get(&r.node_id.0)
                        .map(|id| (id.clone(), 1.0 - r.distance))
                })
                .collect()
        }

        fn len(&self) -> usize {
            self.node_of.len()
        }
    }
}

#[cfg(feature = "vector-astraea")]
pub(crate) use astraea_backend::AstraeaVectorBackend;

/// The vector backend `Store` uses, selected at compile time. With
/// `vector-astraea`: the reused `astraea-vector` HNSW.
#[cfg(feature = "vector-astraea")]
pub(crate) type ActiveIndex = AstraeaVectorBackend;

// ---------------------------------------------------------------------------
// Tests — the same exercise runs against whichever backend is compiled.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Backend-agnostic contract: recall ranks the identical vector first,
    /// results are sorted by similarity, and a removed id never reappears.
    fn exercise<I: VectorIndex>(idx: &mut I) {
        idx.insert("a", &[1.0, 0.0, 0.0]);
        idx.insert("b", &[0.0, 1.0, 0.0]);
        idx.insert("c", &[0.9, 0.1, 0.0]);
        assert_eq!(idx.len(), 3);

        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(hits[0].0, "a", "identical vector ranks first");
        assert!(hits[0].1 >= hits.last().unwrap().1, "sorted by similarity");

        idx.remove("a");
        assert_eq!(idx.len(), 2);
        let hits = idx.search(&[1.0, 0.0, 0.0], 3);
        assert!(hits.iter().all(|(id, _)| id != "a"), "removed id excluded");
    }

    #[cfg(not(feature = "vector-astraea"))]
    #[test]
    fn hnsw_backend_contract() {
        let mut idx = HnswIndex::new(3);
        exercise(&mut idx);
    }

    #[cfg(feature = "vector-astraea")]
    #[test]
    fn astraea_backend_contract() {
        let mut idx = AstraeaVectorBackend::new(3);
        exercise(&mut idx);
    }

    #[cfg(feature = "vector-astraea")]
    #[test]
    fn astraea_real_delete_no_tombstone() {
        // The astraea-vector backend does real deletes: after removal the id is
        // gone, len reflects it, and no tombstone bookkeeping is involved.
        let mut idx = AstraeaVectorBackend::new(3);
        idx.insert("x", &[1.0, 0.0, 0.0]);
        idx.insert("y", &[0.0, 1.0, 0.0]);
        idx.remove("x");
        assert_eq!(idx.len(), 1);
        let hits = idx.search(&[1.0, 0.0, 0.0], 5);
        assert!(hits.iter().all(|(id, _)| id != "x"));
    }
}
