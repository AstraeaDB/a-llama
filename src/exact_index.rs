//! Exact (brute-force) vector index — perfect recall for the `KnowledgeStore`.
//!
//! [`ExactVectorIndex`] implements [`astraea_core::traits::VectorIndex`] via a
//! linear scan over all stored vectors: every `search` call computes the
//! distance from the query to **every** stored vector and returns the `k`
//! closest. This gives recall@k = 1.0 by definition.
//!
//! ## Why exact (not HNSW)?
//!
//! Originally chosen because `astraea_vector::HnswVectorIndex` recall@10 at
//! a-llama's 768-dim config collapsed from ~0.98 below ~2k vectors to ~0.6 past
//! ~3k (astradb issue #25 — a construction bug). #25 has since been fixed
//! upstream (0.1.12, `SELECT-NEIGHBORS-HEURISTIC`); the exact index is retained
//! because it guarantees recall@k = 1.0 and is fast at realistic sizes. The
//! measurement lives in `examples/recall.rs`.
//!
//! ## Performance
//!
//! O(N · dim) per search — sub-millisecond at production graph sizes;
//! a-llama's knowledge graph realistically stays under ~50k nodes. Verified in
//! the latency guard test: < 250 ms for 10 000 × 768 in a debug build.
//!
//! ## Thread safety
//!
//! All methods take `&self`. Interior mutability is provided by
//! `std::sync::RwLock`: `insert`/`remove` hold a write lock; `search`,
//! `len`, and `node_ids` hold a read lock.

use std::collections::HashMap;
use std::sync::RwLock;

use astraea_core::error::AstraeaError;
use astraea_core::traits::VectorIndex;
use astraea_core::types::{DistanceMetric, NodeId, SimilarityResult};
use astraea_vector::distance::compute_distance;

/// Exact (brute-force) vector index with perfect recall.
///
/// Stores vectors in a `HashMap<u64, Vec<f32>>` protected by an `RwLock`.
/// On every `search`, computes the distance to all stored vectors and returns
/// the `k` smallest, sorted ascending by distance. Distance values are
/// identical to what HNSW would use: they are computed via
/// `astraea_vector::distance::compute_distance` so metric semantics match
/// exactly (cosine = `1.0 − cosine_sim`, lower = more similar).
///
/// Tie-breaking (when two stored vectors are equidistant from the query) is
/// done by `NodeId` ascending, which makes results deterministic regardless of
/// `HashMap` iteration order.
///
/// Use [`ExactVectorIndex::new`] for cosine (the default for a-llama's
/// `embeddinggemma` embeddings) or [`ExactVectorIndex::with_metric`] for other
/// metrics.
#[derive(Debug)]
pub struct ExactVectorIndex {
    dim: usize,
    metric: DistanceMetric,
    data: RwLock<HashMap<u64, Vec<f32>>>,
}

impl ExactVectorIndex {
    /// Create a new exact index for the given dimension, using
    /// `DistanceMetric::Cosine` (correct for L2-normalised text embeddings
    /// from `embeddinggemma`, Decision 7660).
    pub fn new(dim: usize) -> Self {
        Self::with_metric(dim, DistanceMetric::Cosine)
    }

    /// Create a new exact index with an explicit distance metric.
    pub fn with_metric(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl VectorIndex for ExactVectorIndex {
    /// Insert (or overwrite) the embedding for `node_id`.
    ///
    /// Overwriting an existing id is intentional: it is how
    /// `KnowledgeStore`'s upsert update path works
    /// (`vector_index.remove(id)` → `insert(id, new_emb)`). Both
    /// `remove`-then-`insert` and direct overwrite produce correct results.
    ///
    /// Returns `DimensionMismatch` if `embedding.len() != self.dim`.
    fn insert(&self, node_id: NodeId, embedding: &[f32]) -> astraea_core::error::Result<()> {
        if embedding.len() != self.dim {
            return Err(AstraeaError::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        self.data.write().unwrap().insert(node_id.0, embedding.to_vec());
        Ok(())
    }

    /// Remove the embedding for `node_id`. Returns `true` if it was present,
    /// `false` if it was already absent (idempotent).
    fn remove(&self, node_id: NodeId) -> astraea_core::error::Result<bool> {
        Ok(self.data.write().unwrap().remove(&node_id.0).is_some())
    }

    /// Find the `k` nearest neighbours of `query` by exhaustive scan.
    ///
    /// Returns at most `k` results (fewer if fewer vectors are stored).
    /// Results are sorted ascending by distance. Ties are broken by `NodeId`
    /// ascending for deterministic output.
    ///
    /// Returns `DimensionMismatch` if `query.len() != self.dim`.
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> astraea_core::error::Result<Vec<SimilarityResult>> {
        if query.len() != self.dim {
            return Err(AstraeaError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let guard = self.data.read().unwrap();
        if guard.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Compute distance to every stored vector.
        let mut scored: Vec<(u64, f32)> = guard
            .iter()
            .map(|(&id, vec)| {
                // compute_distance cannot fail here: query and vec both have
                // self.dim dimensions (enforced on insert and validated above).
                let dist = compute_distance(self.metric, query, vec)
                    .unwrap_or(f32::INFINITY);
                (id, dist)
            })
            .collect();

        // Sort ascending by distance; break ties by node_id (ascending) so
        // results are deterministic regardless of HashMap iteration order.
        scored.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let take = k.min(scored.len());
        Ok(scored[..take]
            .iter()
            .map(|&(id, distance)| SimilarityResult {
                node_id: NodeId(id),
                distance,
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    /// Returns all `NodeId`s currently held by the index.
    ///
    /// Overrides the default (which returns an empty `Vec`) to support
    /// `Graph`'s snapshot-reconciliation logic, which diffs `node_ids()`
    /// against storage to detect embedding drift.
    fn node_ids(&self) -> Vec<NodeId> {
        self.data
            .read()
            .unwrap()
            .keys()
            .map(|&id| NodeId(id))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Local random-number helpers (no external crates; deterministic from seed).
    // -------------------------------------------------------------------------

    /// One step of a 64-bit Knuth multiplicative LCG — deterministic,
    /// period 2^64, enough variety for test vectors.
    fn lcg_step(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map the high 53 bits into [−1.0, 1.0].
        (*state >> 11) as f32 / (1u64 << 52) as f32 * 2.0 - 1.0
    }

    /// Generate a random `dim`-dimensional float vector from `seed`.
    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim).map(|_| lcg_step(&mut state)).collect()
    }

    /// Independent brute-force cosine distance (zero dependency on
    /// astraea-vector). Matches the convention used by
    /// `astraea_vector::distance::compute_distance`: 1 − cosine_similarity,
    /// lower = more similar.
    fn cosine_dist_ref(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - dot / (na * nb)
    }

    // -------------------------------------------------------------------------
    // 1. Exactness — recall@k == 1.0 vs independent brute force
    // -------------------------------------------------------------------------

    /// Insert 1 000 random 768-dim vectors, then for five independent queries
    /// confirm that `ExactVectorIndex::search` returns **exactly** the same
    /// top-10 as an independently-written brute force, in the same order.
    #[test]
    fn exactness_vs_brute_force_random_1000_vectors() {
        const DIM: usize = 768;
        const N: usize = 1_000;
        const K: usize = 10;

        let idx = ExactVectorIndex::new(DIM);

        // Insert N random vectors, keeping a local copy for the reference check.
        let mut stored: Vec<(u64, Vec<f32>)> = Vec::with_capacity(N);
        for i in 0u64..N as u64 {
            let v = random_vec(DIM, i.wrapping_add(1));
            idx.insert(NodeId(i + 1), &v).unwrap();
            stored.push((i + 1, v));
        }

        // Five independent random queries.
        for q in 0u64..5 {
            let query = random_vec(DIM, q.wrapping_add(100_000));

            // Ground-truth top-K by independent brute force (cosine_dist_ref,
            // NOT astraea_vector::distance::compute_distance).
            let mut scored: Vec<(u64, f32)> = stored
                .iter()
                .map(|(id, v)| (*id, cosine_dist_ref(&query, v)))
                .collect();
            scored.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let true_ids: Vec<u64> = scored[..K].iter().map(|(id, _)| *id).collect();

            // ExactVectorIndex result.
            let results = idx.search(&query, K).unwrap();
            assert_eq!(results.len(), K, "must return exactly K results (query {q})");

            let result_ids: Vec<u64> = results.iter().map(|r| r.node_id.0).collect();
            assert_eq!(
                result_ids, true_ids,
                "ExactVectorIndex must return the same top-{K} as brute force (query {q})"
            );

            // Verify ascending-distance order.
            for w in results.windows(2) {
                assert!(
                    w[0].distance <= w[1].distance,
                    "results must be sorted ascending by distance (query {q})"
                );
            }
        }
    }

    /// Clustered data: two tight clusters. A cluster-A query must return only
    /// cluster-A members in the top-K, and vice versa. This is the hardest
    /// case for approximate algorithms and trivial for exact search.
    #[test]
    fn exactness_clustered_data() {
        const DIM: usize = 32;
        const K: usize = 5;

        let idx = ExactVectorIndex::new(DIM);

        // Cluster A: dominant in the first half of dimensions.
        let mut cluster_a: Vec<NodeId> = Vec::new();
        for i in 0u64..20 {
            let mut state = i.wrapping_mul(7).wrapping_add(3);
            let mut v = vec![0.01f32; DIM];
            for d in 0..DIM / 2 {
                v[d] = 1.0 + lcg_step(&mut state) * 0.05;
            }
            let id = NodeId(i + 1);
            idx.insert(id, &v).unwrap();
            cluster_a.push(id);
        }

        // Cluster B: dominant in the second half of dimensions.
        let mut cluster_b: Vec<NodeId> = Vec::new();
        for i in 0u64..20 {
            let mut state = i.wrapping_mul(11).wrapping_add(7);
            let mut v = vec![0.01f32; DIM];
            for d in DIM / 2..DIM {
                v[d] = 1.0 + lcg_step(&mut state) * 0.05;
            }
            let id = NodeId(i + 21);
            idx.insert(id, &v).unwrap();
            cluster_b.push(id);
        }

        // Cluster-A query: all top-K must be cluster-A members.
        let mut q_a = vec![0.01f32; DIM];
        for d in 0..DIM / 2 {
            q_a[d] = 1.0;
        }
        for r in idx.search(&q_a, K).unwrap() {
            assert!(
                cluster_a.contains(&r.node_id),
                "cluster-A query hit {:?} which is not in cluster A",
                r.node_id
            );
        }

        // Cluster-B query: all top-K must be cluster-B members.
        let mut q_b = vec![0.01f32; DIM];
        for d in DIM / 2..DIM {
            q_b[d] = 1.0;
        }
        for r in idx.search(&q_b, K).unwrap() {
            assert!(
                cluster_b.contains(&r.node_id),
                "cluster-B query hit {:?} which is not in cluster B",
                r.node_id
            );
        }
    }

    // -------------------------------------------------------------------------
    // 2. Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn k_greater_than_len_returns_all() {
        let idx = ExactVectorIndex::new(4);
        for i in 1u64..=3 {
            idx.insert(NodeId(i), &[0.1, 0.2, 0.3, 0.4]).unwrap();
        }
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(results.len(), 3, "k > len must return all stored vectors");
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = ExactVectorIndex::new(4);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty(), "search on empty index must return empty Vec");
    }

    #[test]
    fn dim_mismatch_on_insert_errors() {
        let idx = ExactVectorIndex::new(4);
        // 3-dim embedding into a 4-dim index.
        let err = idx.insert(NodeId(1), &[0.1f32, 0.2, 0.3]).unwrap_err();
        assert!(
            matches!(err, AstraeaError::DimensionMismatch { expected: 4, got: 3 }),
            "insert with wrong dim must return DimensionMismatch, got: {err:?}"
        );
    }

    #[test]
    fn dim_mismatch_on_search_errors() {
        let idx = ExactVectorIndex::new(4);
        idx.insert(NodeId(1), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        // 2-dim query into a 4-dim index.
        let err = idx.search(&[1.0f32, 0.0], 1).unwrap_err();
        assert!(
            matches!(err, AstraeaError::DimensionMismatch { expected: 4, got: 2 }),
            "search with wrong dim must return DimensionMismatch, got: {err:?}"
        );
    }

    #[test]
    fn remove_returns_true_then_false() {
        let idx = ExactVectorIndex::new(4);
        idx.insert(NodeId(42), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.remove(NodeId(42)).unwrap(), "first remove must return true");
        assert!(!idx.remove(NodeId(42)).unwrap(), "second remove of same id must return false");
        assert!(idx.is_empty(), "index must be empty after remove");
    }

    #[test]
    fn reinsert_overwrites_embedding() {
        let idx = ExactVectorIndex::new(4);
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0]; // dominant in dim 0
        let v2 = vec![0.0f32, 1.0, 0.0, 0.0]; // dominant in dim 1

        idx.insert(NodeId(1), &v1).unwrap();
        idx.insert(NodeId(2), &v2).unwrap();

        // Before overwrite: searching with v1 → id=1 is nearest (distance ≈ 0).
        let r_before = idx.search(&v1, 1).unwrap();
        assert_eq!(r_before[0].node_id, NodeId(1), "id=1 must be nearest to v1 before overwrite");

        // Overwrite NodeId(1) with v2's direction — len must stay 2.
        idx.insert(NodeId(1), &v2).unwrap();
        assert_eq!(idx.len(), 2, "overwrite must not increase len");

        // After overwrite: searching with v2 must find both id=1 (now v2) and id=2.
        let r_v2 = idx.search(&v2, 2).unwrap();
        let ids_v2: Vec<NodeId> = r_v2.iter().map(|r| r.node_id).collect();
        assert!(ids_v2.contains(&NodeId(1)), "id=1 must appear in v2 search after overwrite");
        assert!(ids_v2.contains(&NodeId(2)), "id=2 must also appear in v2 search");

        // id=1 now stores v2, so it must be far from v1 (orthogonal vectors,
        // cosine distance ≈ 1.0).
        let r_v1 = idx.search(&v1, 2).unwrap();
        let id1_dist = r_v1.iter().find(|r| r.node_id == NodeId(1)).map(|r| r.distance);
        assert!(
            id1_dist.map_or(true, |d| d > 0.5),
            "after overwrite with v2, id=1 must no longer be close to v1 (distance was {:?})",
            id1_dist
        );
    }

    #[test]
    fn node_ids_returns_full_id_set() {
        let idx = ExactVectorIndex::new(4);
        let expected: Vec<NodeId> = vec![NodeId(10), NodeId(20), NodeId(30)];
        for &id in &expected {
            idx.insert(id, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        }

        let mut got = idx.node_ids();
        got.sort_by_key(|n| n.0);
        let mut exp = expected.clone();
        exp.sort_by_key(|n| n.0);
        assert_eq!(got, exp, "node_ids must return every stored NodeId");
    }

    #[test]
    fn len_and_is_empty_are_correct() {
        let idx = ExactVectorIndex::new(4);
        assert!(idx.is_empty(), "fresh index must be empty");
        assert_eq!(idx.len(), 0);

        idx.insert(NodeId(1), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(!idx.is_empty());
        assert_eq!(idx.len(), 1);

        idx.insert(NodeId(2), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 2);

        idx.remove(NodeId(1)).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());

        idx.remove(NodeId(2)).unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    // -------------------------------------------------------------------------
    // 4. Latency guard — regression against accidental O(N²)
    // -------------------------------------------------------------------------

    /// Insert 10 000 random 768-dim vectors and assert that one `search` call
    /// completes in < 250 ms. This is an O(N·dim) regression guard, not a
    /// performance benchmark.
    #[test]
    fn latency_guard_10k_vectors_under_250ms() {
        const DIM: usize = 768;
        const N: usize = 10_000;

        let idx = ExactVectorIndex::new(DIM);
        for i in 0u64..N as u64 {
            let v = random_vec(DIM, i.wrapping_add(77_777));
            idx.insert(NodeId(i + 1), &v).unwrap();
        }

        let query = random_vec(DIM, 999_999);
        let start = std::time::Instant::now();
        let results = idx.search(&query, 10).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10, "latency guard must return 10 results");
        assert!(
            elapsed.as_millis() < 250,
            "10k × 768 brute-force search must complete in < 250 ms (debug build); took {elapsed:?}"
        );
    }
}
