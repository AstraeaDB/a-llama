//! Knowledge-graph HTTP endpoints — bulk ingest, raw vector search and
//! tariff-code classification over the daemon's [`KnowledgeStore`].
//!
//! These routes let an external loader push a **pre-embedded** graph (tariff
//! headings/subheadings, description chunks, rulings, …) into the same store
//! GraphRAG retrieves over, then query it:
//!
//! | Route                   | Purpose                                                  |
//! |-------------------------|----------------------------------------------------------|
//! | `POST /api/ingest`      | Bulk-load nodes (+ embeddings) and edges keyed by `ref`. |
//! | `POST /api/search`      | Embed a text, k-NN over the exact index, 1-hop expand.   |
//! | `POST /api/classify`    | Similarity voting over tariff codes at every prefix level.|
//! | `GET  /api/graph/stats` | Node counts per label, edge count, vector-index size.    |
//!
//! ## Similarity
//!
//! The store's [`ExactVectorIndex`](crate::exact_index::ExactVectorIndex)
//! reports **distances** (`Cosine` metric: `1 − cos_sim`, lower = closer).
//! Every `similarity` in these responses is the cosine similarity converted
//! back and clamped to `[0, 1]` — see [`similarity_from_distance`] — so
//! `1.0` is an identical direction and anti-correlated vectors floor at `0`.
//!
//! ## Blocking work
//!
//! Graph and index operations are synchronous. JSON parsing of a (up to
//! 256 MB) ingest body, node/edge creation, vector search and neighbour
//! expansion all run under `tokio::task::spawn_blocking` so the HTTP worker
//! threads stay responsive; only the engine calls (`embed`, `complete`) are
//! awaited on the async side.
//!
//! ## Refs
//!
//! A node's `ref` is a caller-chosen string key. It is stored on the node as
//! the `"ref"` property and mirrored in an in-process map; in durable mode the
//! map is rebuilt from the persisted property on
//! [`KnowledgeStore::open_durable`], so edges in a later process may reference
//! refs ingested by an earlier one. Re-ingesting a known ref is a no-op that
//! returns the existing id (properties are **not** updated).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use astraea_core::types::{Direction, DistanceMetric, Node, NodeId};

use crate::api::{ApiError, AppState, JsonBody};
use crate::knowledge_store::{code_property, normalize_code, KnowledgeStore, CODE_LEVELS};

/// Maximum accepted `/api/ingest` body (256 MB — ~50k nodes × 768 floats as
/// JSON text). Other routes keep axum's 2 MB default.
pub const INGEST_BODY_LIMIT: usize = 256 * 1024 * 1024;

/// Default / maximum `k` for `/api/search` and `/api/classify`.
pub const DEFAULT_SEARCH_K: usize = 10;
pub const DEFAULT_CLASSIFY_K: usize = 20;
pub const MAX_K: usize = 200;
/// Default `min_similarity` under which `/api/classify` reports
/// `"confident": false`.
pub const DEFAULT_MIN_SIMILARITY: f32 = 0.35;
/// Default per-level cap on the candidate lists returned by `/api/classify`.
pub const DEFAULT_MAX_CANDIDATES: usize = 20;
/// Number of evidence hits returned by `/api/classify`.
const EVIDENCE_HITS: usize = 5;
/// Distinct full codes offered to the LLM when `"llm": true`.
const LLM_CANDIDATES: usize = 8;
/// Snippet length (characters) for evidence / LLM prompts.
const SNIPPET_CHARS: usize = 300;
/// Over-fetch factor applied to `k` when a label filter is given.
const LABEL_OVERFETCH: usize = 4;

/// The graph routes, to be merged into the main router (same `AppState`).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/ingest", post(ingest))
        // Only the routes added above get the raised limit.
        .layer(DefaultBodyLimit::max(INGEST_BODY_LIMIT))
        .route("/api/search", post(search))
        .route("/api/classify", post(classify))
        .route("/api/complete", post(complete))
        .route("/api/graph/stats", get(graph_stats))
}

#[derive(Deserialize)]
struct CompleteRequest {
    #[serde(default)]
    prompt: Option<String>,
    /// Cap on generated tokens (default 256, max 4096).
    #[serde(default)]
    max_tokens: Option<usize>,
}

/// `POST /api/complete` — a raw, stateless completion by the in-process engine:
/// no semantic cache, no GraphRAG augmentation, nothing stored. Used by
/// classification pipelines that orchestrate their own retrieval and need the
/// LLM only as a judge.
async fn complete(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<CompleteRequest>,
) -> Result<Json<Value>, ApiError> {
    let prompt = req.prompt.unwrap_or_default();
    if prompt.trim().is_empty() {
        return Err(bad_request("`prompt` must be a non-empty string"));
    }
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, 4096);
    let response = state
        .engine()
        .complete_capped(&prompt, None, max_tokens)
        .await
        .map_err(internal)?;
    Ok(Json(json!({ "response": response })))
}

// ===========================================================================
// Shared helpers
// ===========================================================================

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn internal(msg: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

/// Convert an index distance to a similarity in `[0, 1]`.
///
/// - `Cosine`: distance is `1 − cos_sim` → `cos_sim = 1 − d`.
/// - `DotProduct`: distance is `−dot` → `dot` (equals `cos_sim` for unit vectors).
/// - `Euclidean`: for unit vectors `‖a−b‖² = 2 − 2·cos_sim` → `1 − d²/2`.
///
/// All are clamped to `[0, 1]`: anti-correlated vectors floor at `0`.
pub fn similarity_from_distance(metric: DistanceMetric, distance: f32) -> f32 {
    let sim = match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::DotProduct => -distance,
        DistanceMetric::Euclidean => 1.0 - distance * distance / 2.0,
    };
    if sim.is_nan() {
        0.0
    } else {
        sim.clamp(0.0, 1.0)
    }
}

/// Clamp a requested `k` into `1..=MAX_K`, applying `default` when absent.
fn clamp_k(requested: Option<usize>, default: usize) -> usize {
    requested.unwrap_or(default).clamp(1, MAX_K)
}

/// First `SNIPPET_CHARS` characters of the node's `text`, else `title`.
fn snippet_of(props: &Value) -> Option<String> {
    let s = props
        .get("text")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| props.get("title").and_then(Value::as_str))?;
    Some(s.chars().take(SNIPPET_CHARS).collect())
}

fn title_of(props: &Value) -> Option<String> {
    props
        .get("title")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// A vector-search hit with its node loaded.
struct Hit {
    node: Node,
    similarity: f32,
}

/// k-NN over the exact index with an optional label filter, returning at most
/// `k` loaded nodes. When filtering, it over-fetches `LABEL_OVERFETCH × k`
/// and keeps widening the fetch until `k` nodes carry a wanted label or the
/// index is exhausted, so a rare label is not crowded out by other nodes.
fn vector_hits(
    store: &KnowledgeStore,
    query: &[f32],
    k: usize,
    labels: Option<&[String]>,
) -> Result<Vec<Hit>, ApiError> {
    let index = store.vector_index_ref();
    let metric = index.metric();
    let graph = store.graph_ops();

    let mut fetch = if labels.is_some() { k.saturating_mul(LABEL_OVERFETCH) } else { k };
    let mut scanned = 0;
    let mut hits = Vec::with_capacity(k);
    loop {
        let results = index.search(query, fetch).map_err(internal)?;
        let exhausted = results.len() < fetch;
        for r in results.into_iter().skip(scanned) {
            scanned += 1;
            let Some(node) = graph.get_node(r.node_id).map_err(internal)? else {
                continue; // index entry without a node (should not happen)
            };
            if let Some(wanted) = labels {
                if !node.labels.iter().any(|l| wanted.iter().any(|w| w == l)) {
                    continue;
                }
            }
            hits.push(Hit {
                node,
                similarity: similarity_from_distance(metric, r.distance),
            });
            if hits.len() >= k {
                return Ok(hits);
            }
        }
        if labels.is_none() || exhausted {
            return Ok(hits);
        }
        fetch = fetch.saturating_mul(LABEL_OVERFETCH);
    }
}

/// All outgoing 1-hop neighbours of `id` as `(edge_type, node)`.
fn outgoing_neighbors(store: &KnowledgeStore, id: NodeId) -> Result<Vec<(String, Node)>, ApiError> {
    let graph = store.graph();
    let edges = graph
        .storage()
        .get_edges(id, Direction::Outgoing)
        .map_err(internal)?;
    let mut out = Vec::with_capacity(edges.len());
    for edge in edges {
        if let Some(node) = graph.storage().get_node(edge.target).map_err(internal)? {
            out.push((edge.edge_type, node));
        }
    }
    Ok(out)
}

fn node_json(node: &Node) -> (u64, Vec<String>, Value) {
    (node.id.0, node.labels.clone(), node.properties.clone())
}

// ===========================================================================
// POST /api/ingest
// ===========================================================================

#[derive(Deserialize)]
struct IngestRequest {
    #[serde(default)]
    nodes: Vec<IngestNode>,
    #[serde(default)]
    edges: Vec<IngestEdge>,
    /// When `true` (default), an edge whose `src` node already existed before
    /// this request is skipped if an identical `src -[type]-> dst` edge is
    /// already present, so re-running a loader does not duplicate edges.
    #[serde(default)]
    dedupe_edges: Option<bool>,
}

#[derive(Deserialize)]
struct IngestNode {
    #[serde(rename = "ref")]
    ref_key: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    properties: Value,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct IngestEdge {
    src: String,
    dst: String,
    #[serde(rename = "type")]
    edge_type: String,
    #[serde(default)]
    weight: Option<f64>,
    #[serde(default)]
    properties: Value,
}

/// `POST /api/ingest` — bulk-load pre-embedded nodes and edges.
///
/// The whole request is validated (refs, embedding dimension, JSON shapes,
/// edge endpoints) **before** anything is written, so a `400` never mutates
/// the store. Writes then go node by node through the AstraeaDB graph API
/// (`create_node` keeps the exact vector index in sync); a storage failure
/// mid-way returns `500` naming the ref and how many nodes were created —
/// re-sending the same request is safe because known refs are skipped.
/// AstraeaDB 0.3.1's disk buffer pool deadlocks when several threads read or
/// write the graph at once (they all block in `BufferPool::find_or_evict_frame`,
/// seen with concurrent `/api/search` + `/api/classify`, and with `stats` during
/// `ingest`), so every graph request runs under this one lock. Throughput is
/// unaffected in practice: the inference engine serialises requests anyway.
static GRAPH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn graph_guard() -> std::sync::MutexGuard<'static, ()> {
    GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

async fn ingest(State(state): State<AppState>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let dim = state.engine().embedding_dim();
    let store = state.knowledge_store();
    let out = tokio::task::spawn_blocking(move || {
        let _g = graph_guard();
        ingest_blocking(&store, dim, &body)
    })
        .await
        .map_err(|e| internal(format!("ingest task failed: {e}")))??;
    Ok(Json(out))
}

fn ingest_blocking(store: &KnowledgeStore, dim: usize, body: &[u8]) -> Result<Value, ApiError> {
    let slice: &[u8] = if body.is_empty() { b"{}" } else { body };
    let req: IngestRequest = serde_json::from_slice(slice)
        .map_err(|e| bad_request(format!("invalid JSON body: {e}")))?;

    // --- Validate everything first (no writes on a 400). ---
    let mut request_refs: HashSet<&str> = HashSet::with_capacity(req.nodes.len());
    for (i, n) in req.nodes.iter().enumerate() {
        if n.ref_key.is_empty() {
            return Err(bad_request(format!("nodes[{i}]: `ref` must be a non-empty string")));
        }
        if !(n.properties.is_object() || n.properties.is_null()) {
            return Err(bad_request(format!(
                "nodes[{i}] (ref `{}`): `properties` must be a JSON object",
                n.ref_key
            )));
        }
        if let Some(emb) = &n.embedding {
            if emb.len() != dim {
                return Err(bad_request(format!(
                    "nodes[{i}] (ref `{}`): embedding has {} dims, engine embedding dim is {dim}",
                    n.ref_key,
                    emb.len()
                )));
            }
        }
        request_refs.insert(n.ref_key.as_str());
    }
    for (i, e) in req.edges.iter().enumerate() {
        if e.edge_type.is_empty() {
            return Err(bad_request(format!("edges[{i}]: `type` must be a non-empty string")));
        }
        if !(e.properties.is_object() || e.properties.is_null()) {
            return Err(bad_request(format!("edges[{i}]: `properties` must be a JSON object")));
        }
        for (field, r) in [("src", &e.src), ("dst", &e.dst)] {
            if !request_refs.contains(r.as_str()) && store.resolve_ref(r).is_none() {
                return Err(bad_request(format!("edges[{i}]: unknown {field} ref `{r}`")));
            }
        }
    }

    // --- Nodes. ---
    let mut ids: HashMap<String, NodeId> = HashMap::with_capacity(req.nodes.len());
    let mut created_ids: HashSet<NodeId> = HashSet::new();
    let (mut nodes_created, mut nodes_existing) = (0usize, 0usize);
    for n in req.nodes {
        let outcome = store
            .ingest_node(&n.ref_key, n.labels, n.properties, n.embedding)
            .map_err(|e| {
                internal(format!(
                    "ingest failed at node ref `{}` ({nodes_created} node(s) created so far in this request): {e}",
                    n.ref_key
                ))
            })?;
        if outcome.created {
            nodes_created += 1;
            created_ids.insert(outcome.id);
        } else {
            nodes_existing += 1;
        }
        ids.insert(n.ref_key, outcome.id);
    }

    // --- Edges. ---
    let dedupe = req.dedupe_edges.unwrap_or(true);
    let (mut edges_created, mut edges_existing) = (0usize, 0usize);
    for (i, e) in req.edges.into_iter().enumerate() {
        let resolve = |r: &str| ids.get(r).copied().or_else(|| store.resolve_ref(r));
        let (Some(src), Some(dst)) = (resolve(&e.src), resolve(&e.dst)) else {
            // Validated above; only reachable if the store lost a ref mid-request.
            return Err(internal(format!("edges[{i}]: ref vanished during ingest")));
        };
        let props = if e.properties.is_null() { json!({}) } else { e.properties };
        // A node created in this request cannot have pre-existing edges, so
        // the (adjacency-read) dedupe check is only paid for old sources.
        let check = dedupe && !created_ids.contains(&src);
        match store.ingest_edge(src, dst, &e.edge_type, e.weight.unwrap_or(1.0), props, check) {
            Ok(Some(_)) => edges_created += 1,
            Ok(None) => edges_existing += 1,
            Err(err) => {
                return Err(internal(format!(
                    "ingest failed at edges[{i}] `{}` -[{}]-> `{}` ({edges_created} edge(s) created so far): {err}",
                    e.src, e.edge_type, e.dst
                )))
            }
        }
    }

    // Durable builds: checkpoint after a bulk load (WAL already made it
    // durable; this just avoids a long replay on the next start).
    #[cfg(feature = "durable")]
    if let Err(e) = store.persist() {
        eprintln!("a-llama: warning: persist after ingest failed: {e}");
    }

    let nodes: BTreeMap<String, u64> = ids.into_iter().map(|(r, id)| (r, id.0)).collect();
    Ok(json!({
        "nodes": nodes,
        "nodes_created": nodes_created,
        "nodes_existing": nodes_existing,
        "edges_created": edges_created,
        "edges_existing": edges_existing,
    }))
}

// ===========================================================================
// POST /api/search
// ===========================================================================

#[derive(Deserialize)]
struct SearchRequest {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    labels: Option<Vec<String>>,
}

/// `POST /api/search` — embed `text`, k-NN over the exact index (optionally
/// keeping only hits with at least one of `labels`, widening the fetch until
/// `k` such hits are found) and expand each hit with all its outgoing 1-hop
/// neighbours.
async fn search(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<SearchRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = req.text.unwrap_or_default();
    if text.trim().is_empty() {
        return Err(bad_request("`text` must be a non-empty string"));
    }
    let k = clamp_k(req.k, DEFAULT_SEARCH_K);
    let labels = req.labels.filter(|l| !l.is_empty());

    let query = state.engine().embed(&text).await.map_err(internal)?;
    let store = state.knowledge_store();
    let hits = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
        let _g = graph_guard();
        let mut out = Vec::new();
        for hit in vector_hits(&store, &query, k, labels.as_deref())? {
            let neighbors: Vec<Value> = outgoing_neighbors(&store, hit.node.id)?
                .into_iter()
                .map(|(edge_type, n)| {
                    let (id, labels, props) = node_json(&n);
                    json!({
                        "edge_type": edge_type,
                        "direction": "out",
                        "node_id": id,
                        "labels": labels,
                        "properties": props,
                    })
                })
                .collect();
            let (id, labels, props) = node_json(&hit.node);
            out.push(json!({
                "node_id": id,
                "labels": labels,
                "similarity": hit.similarity,
                "properties": props,
                "neighbors": neighbors,
            }));
        }
        Ok(out)
    })
    .await
    .map_err(|e| internal(format!("search task failed: {e}")))??;

    Ok(Json(json!({ "hits": hits })))
}

// ===========================================================================
// POST /api/classify
// ===========================================================================

#[derive(Deserialize)]
struct ClassifyRequest {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    material: Option<String>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    min_similarity: Option<f32>,
    #[serde(default)]
    llm: Option<bool>,
    #[serde(default)]
    max_candidates: Option<usize>,
    /// Let the LLM rewrite a terse retail description into a customs-style one
    /// (material, function, form) before embedding; the rewrite is appended to
    /// the query and returned as `expansion`.
    #[serde(default)]
    expand: Option<bool>,
}

/// Retail abbreviations that the embedding model and the LLM misread when left
/// as-is ("S/STEEL", "PYO", "4PK"). Applied deterministically to the description
/// before embedding and before the expansion prompt, so that no glossary text
/// can leak from the prompt into the LLM's answer.
const ABBREVIATIONS: &[(&str, &str)] = &[
    ("S/STEEL", "stainless steel"),
    ("S/S", "stainless steel"),
    ("PYO", "paint your own"),
    ("MYO", "make your own"),
    ("FYO", "fill your own"),
    ("XMAS", "Christmas"),
    ("RC", "radio-controlled"),
    ("PK", "pack"),
    ("PCS", "pieces"),
    ("ASSTD", "assorted"),
    ("ASST", "assorted"),
    ("LGE", "large"),
    ("SML", "small"),
];

/// Lower-cases a retail description and expands known abbreviations
/// ("FOOD STORAGE CLIPS S/STEEL 4PK" → "food storage clips stainless steel 4 pack").
pub(crate) fn normalize_description(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for tok in text.split_whitespace() {
        let up = tok.to_ascii_uppercase();
        let clean = up.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/');
        if let Some(num) = clean
            .strip_suffix("PK")
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
        {
            out.push(format!("{num} pack"));
        } else if let Some((_, rep)) = ABBREVIATIONS.iter().find(|(a, _)| *a == clean) {
            out.push((*rep).to_string());
        } else if let Some(rest) = clean.strip_prefix("W/").filter(|r| !r.is_empty()) {
            out.push(format!("with {}", rest.to_ascii_lowercase()));
        } else {
            out.push(tok.to_lowercase());
        }
    }
    out.join(" ")
}

/// Prompt asking the engine to expand a terse product description. The text
/// must already be normalised; the prompt deliberately contains no examples of
/// materials or features, because small models copy them into the answer.
pub(crate) fn build_expand_prompt(text: &str, material: Option<&str>) -> String {
    let mut p = String::from(
        "You are a customs tariff classification expert. Rewrite the retail product description \
         below as one plain sentence for tariff classification: say what kind of article it is and \
         what it is used for, and name its material only when the description or the seller states \
         it or it is obvious for that kind of article. Do not invent materials, brands, features, \
         sizes or occasions that are not implied by the description. Do not mention tariff codes. \
         Answer with the sentence only.\n\n",
    );
    p.push_str(&format!("Product: {text}\n"));
    if let Some(m) = material.filter(|m| !m.trim().is_empty()) {
        p.push_str(&format!("Material category given by the seller (approximate): {m}\n"));
    }
    p
}

/// First non-empty line of the model's answer, without wrapping quotes, capped.
pub(crate) fn clean_expansion(raw: &str) -> Option<String> {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("<think"))?;
    let line = line.trim_matches(|c| c == '"' || c == '\'' || c == '*');
    let cleaned: String = line.chars().take(400).collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// The parts of a classification computed on the blocking side.
struct ClassifyCore {
    candidates: BTreeMap<String, Vec<Value>>,
    evidence: Vec<Value>,
    best_similarity: Option<f32>,
    /// Distinct full codes (actual code nodes, best score first) for the LLM.
    llm_candidates: Vec<LlmCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LlmCandidate {
    pub code: String,
    pub title: Option<String>,
    pub snippet: Option<String>,
}

/// Every prefix level of `code` (`2/4/6/8`, shorter than the code) plus the
/// code itself. `"94055000"` → `["94", "9405", "940550", "94055000"]`.
pub fn code_levels(code: &str) -> Vec<String> {
    CODE_LEVELS
        .iter()
        .filter(|&&len| len <= code.len())
        .map(|&len| code[..len].to_string())
        .collect()
}

/// Build the query string: `text`, plus `", material: <material>"` when given.
fn classify_query(text: &str, material: Option<&str>) -> String {
    match material.map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => format!("{text}, material: {m}"),
        None => text.to_string(),
    }
}

/// `POST /api/classify` — similarity voting over tariff codes.
///
/// For each of the top-`k` hits, the *code nodes* are the hit itself if it has
/// a valid `code` property, otherwise its outgoing neighbours that do. Every
/// code node adds the hit's similarity to its code **and** to every shorter
/// prefix level (2/4/6/8), so a chunk describing `94055000` also votes for
/// `940550`, `9405` and `94`. Within one hit each code (or prefix) is credited
/// at most once, so a chunk linked to five subheadings of `9405` does not
/// inflate `9405` fivefold.
async fn classify(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<ClassifyRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = req.text.unwrap_or_default();
    if text.trim().is_empty() {
        return Err(bad_request("`text` must be a non-empty string"));
    }
    let text = normalize_description(&text);
    let query = classify_query(&text, req.material.as_deref());
    let k = clamp_k(req.k, DEFAULT_CLASSIFY_K);
    let min_similarity = req.min_similarity.unwrap_or(DEFAULT_MIN_SIMILARITY);
    let max_candidates = req.max_candidates.unwrap_or(DEFAULT_MAX_CANDIDATES).max(1);
    let use_llm = req.llm.unwrap_or(false);

    let engine = state.engine();
    // Optional LLM query expansion — never fails the request.
    let mut expansion = Value::Null;
    let mut query = query;
    if req.expand.unwrap_or(false) {
        match engine
            .complete_capped(&build_expand_prompt(&text, req.material.as_deref()), None, 120)
            .await
        {
            Ok(raw) => {
                if let Some(exp) = clean_expansion(&raw) {
                    query = format!("{query}. {exp}");
                    expansion = Value::String(exp);
                }
            }
            Err(e) => eprintln!("a-llama: warning: classify expansion failed: {e}"),
        }
    }
    let embedding = engine.embed(&query).await.map_err(internal)?;
    let store = state.knowledge_store();
    let core = tokio::task::spawn_blocking(move || {
        let _g = graph_guard();
        classify_blocking(&store, &embedding, k, max_candidates)
    })
    .await
    .map_err(|e| internal(format!("classify task failed: {e}")))??;

    let confident = core.best_similarity.map_or(false, |s| s >= min_similarity);

    // Optional LLM arbitration — never fails the request.
    let (mut llm_choice, mut llm_raw) = (Value::Null, Value::Null);
    if use_llm && !core.llm_candidates.is_empty() {
        let prompt = build_llm_prompt(&query, &core.llm_candidates);
        match engine.complete_capped(&prompt, None, 80).await {
            Ok(raw) => {
                let allowed: Vec<&str> = core.llm_candidates.iter().map(|c| c.code.as_str()).collect();
                if let Some((code, reason)) = parse_llm_choice(&raw, &allowed) {
                    llm_choice = json!({ "code": code, "reason": reason });
                }
                llm_raw = Value::String(raw.chars().take(1000).collect());
            }
            Err(e) => eprintln!("a-llama: warning: classify LLM call failed: {e}"),
        }
    }

    Ok(Json(json!({
        "query": query,
        "expansion": expansion,
        "confident": confident,
        "best_similarity": core.best_similarity,
        "candidates": core.candidates,
        "evidence": core.evidence,
        "llm_choice": llm_choice,
        "llm_raw": llm_raw,
    })))
}

fn classify_blocking(
    store: &KnowledgeStore,
    query: &[f32],
    k: usize,
    max_candidates: usize,
) -> Result<ClassifyCore, ApiError> {
    let hits = vector_hits(store, query, k, None)?;

    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut titles: HashMap<String, String> = HashMap::new();
    // Codes that came from an actual code node (not only a synthesised prefix).
    let mut direct: HashSet<String> = HashSet::new();
    // code → (best similarity, snippet of that hit) for the LLM prompt.
    let mut best_snippet: HashMap<String, (f32, Option<String>)> = HashMap::new();
    let mut evidence = Vec::with_capacity(EVIDENCE_HITS);

    for hit in &hits {
        // Code nodes for this hit: itself, or its outgoing neighbours.
        let mut code_nodes: Vec<(String, Option<String>)> = Vec::new();
        if let Some(code) = code_property(&hit.node.properties) {
            code_nodes.push((code, title_of(&hit.node.properties)));
        } else {
            for (edge_type, n) in outgoing_neighbors(store, hit.node.id)? {
                // A note that *cites* another heading (usually to exclude it) must
                // not vote for that heading; only DESCRIBES / RULING_FOR / CHILD_OF do.
                if edge_type == "REFERS_TO" {
                    continue;
                }
                if let Some(code) = code_property(&n.properties) {
                    code_nodes.push((code, title_of(&n.properties)));
                }
            }
        }

        // Deterministic order (adjacency order is unspecified): most specific
        // code first, then lexicographic — so `evidence.code` is the deepest.
        code_nodes.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        code_nodes.dedup_by(|a, b| a.0 == b.0);

        let snippet = snippet_of(&hit.node.properties);
        let mut voted: HashSet<String> = HashSet::new();
        for (code, title) in &code_nodes {
            if let Some(t) = title {
                titles.entry(code.clone()).or_insert_with(|| t.clone());
            }
            direct.insert(code.clone());
            voted.extend(code_levels(code));
            let entry = best_snippet
                .entry(code.clone())
                .or_insert((hit.similarity, snippet.clone()));
            if hit.similarity > entry.0 {
                *entry = (hit.similarity, snippet.clone());
            }
        }
        for code in voted {
            *scores.entry(code).or_insert(0.0) += f64::from(hit.similarity);
        }

        if evidence.len() < EVIDENCE_HITS {
            let codes: Vec<&String> = code_nodes.iter().map(|(c, _)| c).collect();
            evidence.push(json!({
                "node_id": hit.node.id.0,
                "similarity": hit.similarity,
                "code": codes.first().cloned(),
                "codes": codes,
                "snippet": snippet,
            }));
        }
    }

    // Rank: score desc, then code asc (deterministic).
    let mut ranked: Vec<(&String, &f64)> = scores.iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));

    let mut candidates: BTreeMap<String, Vec<Value>> =
        CODE_LEVELS.iter().map(|l| (l.to_string(), Vec::new())).collect();
    for (code, score) in &ranked {
        let level = candidates.get_mut(&code.len().to_string()).expect("valid level");
        if level.len() >= max_candidates {
            continue;
        }
        let title = titles
            .get(*code)
            .cloned()
            .or_else(|| lookup_title(store, code));
        level.push(json!({ "code": code, "score": score, "title": title }));
    }

    let llm_candidates = ranked
        .iter()
        .filter(|(code, _)| direct.contains(*code))
        .take(LLM_CANDIDATES)
        .map(|(code, _)| LlmCandidate {
            code: (*code).clone(),
            title: titles
                .get(*code)
                .cloned()
                .or_else(|| lookup_title(store, code)),
            snippet: best_snippet.get(*code).and_then(|(_, s)| s.clone()),
        })
        .collect();

    Ok(ClassifyCore {
        candidates,
        evidence,
        best_similarity: hits.first().map(|h| h.similarity),
        llm_candidates,
    })
}

/// Title of the ingested code node for `code`, if one exists (lets prefix
/// levels that were never a direct hit still carry their heading title).
fn lookup_title(store: &KnowledgeStore, code: &str) -> Option<String> {
    let id = store.node_by_code(code)?;
    let node = store.graph_ops().get_node(id).ok()??;
    title_of(&node.properties)
}

/// Prompt asking the engine to pick exactly one of the candidate codes.
pub(crate) fn build_llm_prompt(query: &str, candidates: &[LlmCandidate]) -> String {
    let mut p = String::new();
    p.push_str("You are a customs tariff classification assistant.\n");
    p.push_str(&format!("Product description: {query:?}\n\n"));
    p.push_str("Candidate tariff codes (most similar first):\n");
    for (i, c) in candidates.iter().enumerate() {
        p.push_str(&format!(
            "{}. {} — {}\n",
            i + 1,
            c.code,
            c.title.as_deref().unwrap_or("(no title)")
        ));
        if let Some(s) = &c.snippet {
            p.push_str(&format!("   evidence: {:?}\n", s));
        }
    }
    p.push_str(
        "\nChoose the single best tariff code for the product from the candidates above. \
         Reply with exactly one line in the form:\nCODE: <digits> | REASON: <one sentence>\n",
    );
    p
}

/// Byte offset of the first mention of `code` in `text`, tolerating a single
/// `.` or space between digit groups (`9405.50.00`, `9405 50 00`) and
/// requiring digit boundaries on both sides, so `9405` is **not** found inside
/// `94055000`.
fn find_code_mention(text: &str, code: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let code = code.as_bytes();
    if code.is_empty() {
        return None;
    }
    for start in 0..bytes.len() {
        if !bytes[start].is_ascii_digit() || (start > 0 && bytes[start - 1].is_ascii_digit()) {
            continue;
        }
        let (mut i, mut j) = (start, 0);
        while j < code.len() && i < bytes.len() {
            if bytes[i] == code[j] {
                i += 1;
                j += 1;
            } else if j > 0
                && matches!(bytes[i], b'.' | b' ')
                && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)
            {
                i += 1; // one separator between digit groups
            } else {
                break;
            }
        }
        if j == code.len() && !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            return Some(start);
        }
    }
    None
}

/// Parse the engine's reply into `(code, reason)`.
///
/// 1. Any `CODE:` marker whose value (digits only, up to `|` / newline) is a
///    valid code — accepted even if it is not in `allowed` (the model may
///    know better; the client can check it against `candidates`).
/// 2. Otherwise the earliest mention of one of the `allowed` codes (see
///    [`find_code_mention`]: dotted/spaced forms accepted, digit-bounded);
///    on a tie the longer code wins (`9405 50 00` is `94055000`, not `9405`).
///
/// The reason is the text after `REASON:` (to end of line), else the first
/// 300 characters of the reply. Returns `None` if no code can be found.
pub(crate) fn parse_llm_choice(raw: &str, allowed: &[&str]) -> Option<(String, String)> {
    let mut code: Option<String> = None;
    let mut rest = raw;
    while let Some(pos) = rest.find("CODE:") {
        let after = &rest[pos + "CODE:".len()..];
        let segment = after.split(['|', '\n']).next().unwrap_or("");
        if let Some(c) = normalize_code(segment) {
            code = Some(c);
            break;
        }
        rest = after;
    }
    if code.is_none() {
        code = allowed
            .iter()
            .filter_map(|c| find_code_mention(raw, c).map(|pos| (pos, *c)))
            .min_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.len().cmp(&a.1.len())))
            .map(|(_, c)| c.to_string());
    }
    let code = code?;

    let reason = raw
        .find("REASON:")
        .map(|pos| raw[pos + "REASON:".len()..].lines().next().unwrap_or("").trim().to_string())
        // An echoed template placeholder (`<one sentence>`) is not a reason.
        .filter(|r| !r.is_empty() && !(r.starts_with('<') && r.ends_with('>')))
        .unwrap_or_else(|| raw.trim().chars().take(SNIPPET_CHARS).collect());
    Some((code, reason))
}

// ===========================================================================
// GET /api/graph/stats
// ===========================================================================

/// `GET /api/graph/stats` — exact node counts per label, edge count, vector
/// index size and number of known refs (one full node walk).
async fn graph_stats(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let store: Arc<KnowledgeStore> = state.knowledge_store();
    let stats = tokio::task::spawn_blocking(move || {
        let _g = graph_guard();
        store.stats()
    })
        .await
        .map_err(|e| internal(format!("stats task failed: {e}")))?
        .map_err(internal)?;
    Ok(Json(json!({
        "nodes": stats.nodes,
        "nodes_by_label": stats.nodes_by_label,
        "edges": stats.edges,
        "vector_index_size": stats.vector_index_size,
        "refs": stats.refs,
    })))
}

// ===========================================================================
// Tests (pure helpers; the HTTP surface is covered in tests/graph_api.rs)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_conversion_per_metric() {
        let c = DistanceMetric::Cosine;
        assert_eq!(similarity_from_distance(c, 0.0), 1.0);
        assert!((similarity_from_distance(c, 0.17) - 0.83).abs() < 1e-6);
        assert_eq!(similarity_from_distance(c, 1.0), 0.0);
        assert_eq!(similarity_from_distance(c, 2.0), 0.0, "anti-correlated floors at 0");
        assert_eq!(similarity_from_distance(c, -0.5), 1.0, "rounding above 1 clamps");
        assert_eq!(similarity_from_distance(c, f32::NAN), 0.0);
        assert!((similarity_from_distance(DistanceMetric::DotProduct, -0.6) - 0.6).abs() < 1e-6);
        // Unit vectors at 60°: ‖a−b‖ = 1 → cos = 0.5.
        assert!((similarity_from_distance(DistanceMetric::Euclidean, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn code_levels_expand_to_every_prefix() {
        assert_eq!(code_levels("94055000"), ["94", "9405", "940550", "94055000"]);
        assert_eq!(code_levels("94"), ["94"]);
        assert_eq!(
            code_levels("9405500010"),
            ["94", "9405", "940550", "94055000", "9405500010"]
        );
    }

    #[test]
    fn classify_query_appends_material_only_when_present() {
        assert_eq!(classify_query("clips", Some("Metal/Steel")), "clips, material: Metal/Steel");
        assert_eq!(classify_query("clips", Some("  ")), "clips");
        assert_eq!(classify_query("clips", None), "clips");
    }

    #[test]
    fn clamp_k_bounds() {
        assert_eq!(clamp_k(None, 10), 10);
        assert_eq!(clamp_k(Some(0), 10), 1);
        assert_eq!(clamp_k(Some(5_000), 10), MAX_K);
        assert_eq!(clamp_k(Some(7), 10), 7);
    }

    #[test]
    fn parse_llm_choice_marker_line_and_fallbacks() {
        let allowed = ["94055000", "73239300"];
        // Canonical reply.
        let (c, r) = parse_llm_choice(
            "CODE: 94055000 | REASON: Non-electrical lamp made of brass.",
            &allowed,
        )
        .unwrap();
        assert_eq!(c, "94055000");
        assert_eq!(r, "Non-electrical lamp made of brass.");

        // Spaced code after the marker, reason on the next line.
        let (c, r) = parse_llm_choice("CODE: 9405 50 00\nREASON: because\n", &allowed).unwrap();
        assert_eq!(c, "94055000");
        assert_eq!(r, "because");

        // Marker value that is not a valid code is skipped; the earliest
        // mention of an allowed code (dotted, after a list number) wins —
        // the mock-engine echo shape.
        let (c, r) = parse_llm_choice(
            "CODE: <digits> | REASON: <one sentence>\n2. 7323.93.00 — kitchenware",
            &allowed,
        )
        .unwrap();
        assert_eq!(c, "73239300");
        assert!(r.starts_with("CODE: <digits>"), "placeholder reason falls back to the reply: {r}");

        // Mention fallback restricted to allowed codes (list numbers ignored);
        // reason falls back to the reply itself.
        let (c, r) = parse_llm_choice("1. maybe 10 or 94055000 fits", &allowed).unwrap();
        assert_eq!(c, "94055000");
        assert_eq!(r, "1. maybe 10 or 94055000 fits");

        // Earliest mention wins; a spaced full code beats its own prefix.
        let both = ["9405", "94055000", "73239300"];
        let (c, _) = parse_llm_choice("not 7323.93.00; rather 9405 50 00", &both).unwrap();
        assert_eq!(c, "73239300", "earliest mention");
        let (c, _) = parse_llm_choice("choose 9405 50 00 (lamps)", &both).unwrap();
        assert_eq!(c, "94055000", "longer code wins the tie at the same position");

        // A prefix is not found inside a longer digit run.
        assert_eq!(parse_llm_choice("it is 94055000", &["9405"]), None);
        assert_eq!(find_code_mention("x 194055000 y", "94055000"), None);
        assert_eq!(find_code_mention("x 9405.50.00, y", "94055000"), Some(2));

        // Nothing usable → None.
        assert_eq!(parse_llm_choice("I cannot decide.", &allowed), None);
        assert_eq!(parse_llm_choice("", &allowed), None);
    }

    #[test]
    fn llm_prompt_lists_candidates_and_format() {
        let prompt = build_llm_prompt(
            "clips",
            &[LlmCandidate {
                code: "73239300".into(),
                title: Some("Kitchenware of stainless steel".into()),
                snippet: Some("food storage clips".into()),
            }],
        );
        assert!(prompt.contains("1. 73239300 — Kitchenware of stainless steel"));
        assert!(prompt.contains("evidence: \"food storage clips\""));
        assert!(prompt.contains("CODE: <digits> | REASON:"));
    }

    #[test]
    fn snippet_prefers_text_and_truncates_on_char_boundary() {
        let long = "é".repeat(400);
        let s = snippet_of(&json!({ "text": long, "title": "t" })).unwrap();
        assert_eq!(s.chars().count(), SNIPPET_CHARS);
        assert_eq!(snippet_of(&json!({ "title": "only title" })).as_deref(), Some("only title"));
        assert_eq!(snippet_of(&json!({ "text": "", "title": "fallback" })).as_deref(), Some("fallback"));
        assert_eq!(snippet_of(&json!({ "code": "94" })), None);
    }
}
