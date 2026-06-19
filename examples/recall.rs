//! Recall@k benchmark for astraea-vector's HNSW vs brute-force ground truth.
//! (Originally built to quantify astradb #25, since fixed upstream in 0.1.12.)
//!
//! Measures the approximate-nearest-neighbour **accuracy** of
//! `astraea_vector::HnswVectorIndex` configured EXACTLY as a-llama's
//! `KnowledgeStore` configures it (Cosine, m=16, ef_construction=200), against
//! brute-force ground truth on clustered, L2-of-gaussian data. The data
//! generation (PRNG, clustering, noise, seed) is copied verbatim from Eunomia's
//! harness (`projects/eunomia/.../examples/recall.rs`, which found recall@10 ≈
//! 0.65 for astraea-vector at dim 128) so results are directly comparable — the
//! only differences are dim (a-llama uses 768), the params, and that this queries
//! `HnswVectorIndex` directly rather than through Eunomia's Store abstraction.
//!
//! ```bash
//! # a-llama's real config (dim 768, ef_search 50):
//! cargo run --release --example recall
//! # harness sanity-check — should reproduce Eunomia's ~0.65 at dim 128:
//! RECALL_DIM=128 cargo run --release --example recall
//! # sweep:
//! RECALL_DIM=768 RECALL_N=2000 RECALL_EF=100 cargo run --release --example recall
//! ```
//! Env knobs: RECALL_DIM, RECALL_N, RECALL_K, RECALL_QUERIES, RECALL_CLUSTERS,
//! RECALL_NOISE, RECALL_EF (ef_search), RECALL_SEED.

use std::collections::HashSet;
use std::time::Instant;

use astraea_core::traits::VectorIndex;
use astraea_core::types::{DistanceMetric, NodeId};
use astraea_vector::HnswVectorIndex;

// a-llama KnowledgeStore HNSW construction params (src/knowledge_store.rs).
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;

/// Minimal deterministic PRNG (xorshift64*) — verbatim from Eunomia's harness so
/// the generated dataset is identical for a given (dim, n, seed).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.unit() - 0.5;
        }
        s
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Exact top-`k` node ids by cosine similarity (ground truth).
fn exact_topk(points: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<u64> {
    let mut scored: Vec<(f32, u64)> = points
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine(query, v), i as u64))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn main() {
    let n = env_usize("RECALL_N", 10_000);
    let dim = env_usize("RECALL_DIM", 768); // a-llama: embeddinggemma 768 (Decision 7660)
    let k = env_usize("RECALL_K", 10);
    let queries = env_usize("RECALL_QUERIES", 500);
    let clusters = env_usize("RECALL_CLUSTERS", 100);
    let noise = env_f32("RECALL_NOISE", 0.55);
    let ef_search = env_usize("RECALL_EF", 50); // a-llama KnowledgeStore default
    let seed = env_usize("RECALL_SEED", 42) as u64;

    let mut rng = Rng::new(seed);

    // Cluster centers (verbatim with Eunomia's harness).
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gauss()).collect())
        .collect();

    // Dataset: cluster center + per-component gaussian noise. Keep a copy for
    // exact ground truth, and insert into the index under matching NodeIds.
    let index =
        HnswVectorIndex::with_params(dim, DistanceMetric::Cosine, HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
    let mut points: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let c = rng.below(clusters);
        let v: Vec<f32> = (0..dim).map(|d| centers[c][d] + noise * rng.gauss()).collect();
        index.insert(NodeId(i as u64), &v).expect("insert");
        points.push(v);
    }

    // Queries: independent held-out perturbations of random clusters.
    let mut recall_sum = 0.0f64;
    let mut returned_sum = 0usize;
    let mut elapsed = std::time::Duration::ZERO;
    for _ in 0..queries {
        let c = rng.below(clusters);
        let q: Vec<f32> = (0..dim).map(|d| centers[c][d] + noise * rng.gauss()).collect();

        let truth = exact_topk(&points, &q, k);

        let start = Instant::now();
        let hits = index.search(&q, k).expect("search");
        elapsed += start.elapsed();

        returned_sum += hits.len();
        let found = hits.iter().filter(|h| truth.contains(&h.node_id.0)).count();
        recall_sum += found as f64 / k as f64;
    }

    let recall = recall_sum / queries as f64;
    let avg_returned = returned_sum as f64 / queries as f64;
    let avg_us = elapsed.as_secs_f64() * 1e6 / queries as f64;

    println!("a-llama KnowledgeStore vector recall — astraea_vector::HnswVectorIndex");
    println!("  params:       m={HNSW_M} ef_construction={HNSW_EF_CONSTRUCTION} ef_search={ef_search} metric=Cosine");
    println!("  vectors:      {n}   dim: {dim}   clusters: {clusters}   noise: {noise}");
    println!("  queries:      {queries}   k: {k}   seed: {seed}");
    println!("  recall@{k}:    {recall:.4}");
    println!("  avg returned: {avg_returned:.2} / {k}");
    println!("  avg query:    {avg_us:.2} µs");
}
