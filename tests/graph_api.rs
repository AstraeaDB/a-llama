//! Integration tests for the knowledge-graph HTTP endpoints
//! (`/api/ingest`, `/api/search`, `/api/classify`, `/api/graph/stats`).
//!
//! Drives the real axum router via `tower::ServiceExt::oneshot` over an
//! in-memory `AppState` with the deterministic `MockEngine` (768-dim,
//! FNV-seeded pseudo-embeddings): embedding the exact same text twice gives
//! the same vector (similarity 1.0), while unrelated texts are near-orthogonal
//! (|cos| ≲ 0.1), so ranking assertions are deterministic.
//!
//! The durable round-trip (ref map rebuilt after reopen) runs only with
//! `--features durable`.

use std::sync::Arc;

use a_llama::api::{AppState, OrchestratorConfig};
use a_llama::inference::{InferenceEngine, MockEngine};
use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
    response::Response,
};
use serde_json::{json, Value};
use tower::ServiceExt;

// ===========================================================================
// Helpers
// ===========================================================================

fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

async fn body_json(resp: Response) -> Value {
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("non-JSON body: {e}"))
}

/// Send one request through a fresh router over `state` and return
/// `(status, json body)`.
async fn call(state: &AppState, req: Request<Body>) -> (StatusCode, Value) {
    let resp = a_llama::api::router(state.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    (status, body_json(resp).await)
}

async fn embed(text: &str) -> Vec<f32> {
    MockEngine::new().embed(text).await.unwrap()
}

// Lower-case and abbreviation-free so that classify's query normalisation leaves
// it unchanged and the mock embedding of the query equals the chunk's.
const CHUNK_LAMP: &str = "brass candle holder, a non-electrical lamp for tea lights";
const CHUNK_CLIPS: &str = "FOOD STORAGE CLIPS S/STEEL bag sealing clips for the kitchen";

/// The fixture graph: 3 code nodes + 2 chunk nodes.
///
/// ```text
/// c:lamp  -DESCRIBES->  s:94055000 (Subheading, "non-electrical luminaires")
/// c:lamp  -DESCRIBES->  h:9405     (Heading,    "Lamps and lighting fittings")
/// c:clips -DESCRIBES->  s:73239300 (Subheading, "Kitchenware of stainless steel")
/// s:94055000 -CHILD_OF-> h:9405
/// ```
/// `s:73239300` deliberately has no embedding (`null`).
async fn fixture() -> Value {
    json!({
        "nodes": [
            { "ref": "h:9405", "labels": ["Heading", "Tariff"],
              "properties": { "code": "9405", "title": "Lamps and lighting fittings" },
              "embedding": embed("Lamps and lighting fittings").await },
            { "ref": "s:94055000", "labels": ["Subheading", "Tariff"],
              "properties": { "code": "9405 50 00", "title": "non-electrical luminaires" },
              "embedding": embed("non-electrical luminaires").await },
            { "ref": "s:73239300", "labels": ["Subheading", "Tariff"],
              "properties": { "code": "73239300", "title": "Kitchenware of stainless steel" },
              "embedding": null },
            { "ref": "c:lamp", "labels": ["Chunk"],
              "properties": { "text": CHUNK_LAMP, "source": "explanatory-notes" },
              "embedding": embed(CHUNK_LAMP).await },
            { "ref": "c:clips", "labels": ["Chunk"],
              "properties": { "text": CHUNK_CLIPS },
              "embedding": embed(CHUNK_CLIPS).await }
        ],
        "edges": [
            { "src": "c:lamp", "dst": "s:94055000", "type": "DESCRIBES", "weight": 1.0, "properties": {} },
            { "src": "c:lamp", "dst": "h:9405", "type": "DESCRIBES" },
            { "src": "c:clips", "dst": "s:73239300", "type": "DESCRIBES", "weight": 0.9 },
            { "src": "s:94055000", "dst": "h:9405", "type": "CHILD_OF" }
        ]
    })
}

fn state() -> AppState {
    AppState::default_mock().unwrap()
}

// ===========================================================================
// /api/ingest
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn ingest_creates_nodes_and_edges_and_is_idempotent() {
    let state = state();

    let (status, v) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["nodes_created"], 5);
    assert_eq!(v["nodes_existing"], 0);
    assert_eq!(v["edges_created"], 4);
    assert_eq!(v["edges_existing"], 0);
    let ids = v["nodes"].as_object().unwrap();
    assert_eq!(ids.len(), 5);
    for r in ["h:9405", "s:94055000", "s:73239300", "c:lamp", "c:clips"] {
        assert!(ids[r].is_u64(), "ref {r} must map to a numeric node id");
    }

    // The nodes landed in the shared KnowledgeStore with `ref` stored on them.
    let store = state.knowledge_store();
    let lamp_id = store.resolve_ref("c:lamp").expect("ref map populated");
    assert_eq!(lamp_id.0, ids["c:lamp"].as_u64().unwrap());
    let node = store.graph_ops().get_node(lamp_id).unwrap().unwrap();
    assert_eq!(node.properties["ref"], "c:lamp");
    assert_eq!(node.properties["text"], CHUNK_LAMP);
    assert!(node.embedding.is_some());

    // Second identical ingest: nothing new, same ids, no duplicate edges.
    let (status, v2) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK, "{v2}");
    assert_eq!(v2["nodes_created"], 0);
    assert_eq!(v2["nodes_existing"], 5);
    assert_eq!(v2["edges_created"], 0);
    assert_eq!(v2["edges_existing"], 4);
    assert_eq!(v2["nodes"], v["nodes"], "existing refs return the same ids");

    // An edge referencing refs from an earlier request only.
    let (status, v3) = call(
        &state,
        post_json(
            "/api/ingest",
            json!({ "edges": [ { "src": "c:clips", "dst": "h:9405", "type": "RULING_FOR" } ] }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v3}");
    assert_eq!(v3["edges_created"], 1);
    assert_eq!(v3["nodes"], json!({}));
}

#[tokio::test(flavor = "multi_thread")]
async fn ingest_rejects_bad_input_without_writing() {
    let state = state();
    let dim = MockEngine::new().embedding_dim();

    // Wrong embedding length → 400 naming the ref.
    let (status, v) = call(
        &state,
        post_json(
            "/api/ingest",
            json!({ "nodes": [ { "ref": "bad:dim", "labels": [], "properties": {}, "embedding": [0.1, 0.2] } ] }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let msg = v["error"].as_str().unwrap();
    assert!(msg.contains("bad:dim") && msg.contains(&dim.to_string()), "{msg}");

    // Unknown edge ref → 400 naming the ref; the valid node in the same
    // request must NOT have been written (validation precedes writes).
    let (status, v) = call(
        &state,
        post_json(
            "/api/ingest",
            json!({
                "nodes": [ { "ref": "ok:1", "labels": ["X"], "properties": {} } ],
                "edges": [ { "src": "ok:1", "dst": "nope:2", "type": "LINK" } ]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("nope:2"), "{v}");
    assert!(state.knowledge_store().resolve_ref("ok:1").is_none());

    // Empty ref, non-object properties, empty edge type, malformed JSON.
    for body in [
        json!({ "nodes": [ { "ref": "", "properties": {} } ] }),
        json!({ "nodes": [ { "ref": "p", "properties": [1, 2] } ] }),
        json!({ "nodes": [ { "ref": "a" }, { "ref": "b" } ], "edges": [ { "src": "a", "dst": "b", "type": "" } ] }),
    ] {
        let (status, _) = call(&state, post_json("/api/ingest", body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    let resp = a_llama::api::router(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/ingest")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let (_, stats) = call(&state, get("/api/graph/stats")).await;
    assert_eq!(stats["nodes"], 0, "no rejected request may write");
}

#[tokio::test(flavor = "multi_thread")]
async fn ingest_accepts_body_larger_than_axum_default_limit() {
    // ~3.5 MB body: > axum's 2 MB default, well under the 256 MB route limit.
    let state = state();
    let emb = embed("bulk").await;
    let nodes: Vec<Value> = (0..300)
        .map(|i| json!({ "ref": format!("bulk:{i}"), "labels": ["Bulk"], "properties": {}, "embedding": emb }))
        .collect();
    let body = json!({ "nodes": nodes });
    assert!(body.to_string().len() > 2 * 1024 * 1024, "fixture must exceed 2 MB");

    let (status, v) = call(&state, post_json("/api/ingest", body)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["nodes_created"], 300);
}

// ===========================================================================
// /api/search
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn search_returns_chunk_with_neighbor_codes() {
    let state = state();
    let (status, ingested) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK);
    let ids = ingested["nodes"].clone();

    let (status, v) = call(
        &state,
        post_json("/api/search", json!({ "text": CHUNK_LAMP, "k": 3, "labels": ["Chunk"] })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty() && hits.len() <= 3);
    for h in hits {
        assert!(h["labels"].as_array().unwrap().contains(&json!("Chunk")), "label filter: {h}");
        let sim = h["similarity"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&sim), "similarity in [0,1]: {sim}");
        assert!(h["properties"].get("embedding").is_none(), "no embedding in properties");
    }

    // Top hit is the identical-text chunk with similarity 1.0.
    let top = &hits[0];
    assert_eq!(top["node_id"], ids["c:lamp"]);
    assert!(top["similarity"].as_f64().unwrap() > 0.999, "{top}");
    assert_eq!(top["properties"]["ref"], "c:lamp");
    assert_eq!(top["properties"]["text"], CHUNK_LAMP);

    // Its outgoing neighbours are the two code nodes, with their properties.
    let neighbors = top["neighbors"].as_array().unwrap();
    assert_eq!(neighbors.len(), 2, "{neighbors:?}");
    for n in neighbors {
        assert_eq!(n["edge_type"], "DESCRIBES");
        assert_eq!(n["direction"], "out");
        assert!(n["labels"].as_array().unwrap().contains(&json!("Tariff")));
        assert!(n["properties"]["code"].is_string());
    }
    let neighbor_ids: Vec<u64> = neighbors.iter().map(|n| n["node_id"].as_u64().unwrap()).collect();
    assert!(neighbor_ids.contains(&ids["s:94055000"].as_u64().unwrap()));
    assert!(neighbor_ids.contains(&ids["h:9405"].as_u64().unwrap()));

    // Without a label filter the same query still ranks the chunk first;
    // `k` defaults to 10 and is capped at 200 (no error on a huge k).
    let (status, v) = call(&state, post_json("/api/search", json!({ "text": CHUNK_LAMP, "k": 100000 })))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["hits"][0]["node_id"], ids["c:lamp"]);
    // 4 nodes carry embeddings (s:73239300 has none).
    assert_eq!(v["hits"].as_array().unwrap().len(), 4);

    // Label filter that matches nothing → empty hits, not an error.
    let (status, v) = call(&state, post_json("/api/search", json!({ "text": "x", "labels": ["Nope"] }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["hits"], json!([]));

    // Missing text → 400.
    let (status, _) = call(&state, post_json("/api/search", json!({ "k": 3 }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ===========================================================================
// /api/classify
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn classify_aggregates_prefix_levels_from_chunk_hit() {
    let state = state();
    let (status, ingested) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK);
    let ids = ingested["nodes"].clone();

    let (status, v) = call(
        &state,
        post_json("/api/classify", json!({ "text": CHUNK_LAMP, "k": 20, "llm": false })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["query"], CHUNK_LAMP, "no material → query is the normalised text");
    assert_eq!(v["confident"], true);
    assert!(v["best_similarity"].as_f64().unwrap() > 0.999);
    assert!(v["llm_choice"].is_null());

    let cands = &v["candidates"];
    for level in ["2", "4", "6", "8", "10"] {
        assert!(cands[level].is_array(), "level {level} must be present: {cands}");
    }
    // The chunk (sim 1.0) describes 94055000 and 9405 → credits 94, 9405,
    // 940550, 94055000 each exactly once (per-hit dedup: 9405 is linked
    // directly AND is a prefix of 94055000, yet scores ~1.0, not ~2.0).
    let top = |level: &str| cands[level][0].clone();
    assert_eq!(top("2")["code"], "94");
    assert_eq!(top("4")["code"], "9405");
    assert_eq!(top("6")["code"], "940550");
    assert_eq!(top("8")["code"], "94055000");
    assert!(cands["10"].as_array().unwrap().is_empty(), "no 10-digit codes ingested");
    for level in ["2", "4", "6", "8"] {
        let score = top(level)["score"].as_f64().unwrap();
        assert!((0.99..1.15).contains(&score), "level {level} score {score} must be ~1.0");
    }
    // Titles come from the ingested code nodes (9405 and 94055000 exist;
    // 94 / 940550 have no node → null).
    assert_eq!(top("4")["title"], "Lamps and lighting fittings");
    assert_eq!(top("8")["title"], "non-electrical luminaires");
    assert!(top("2")["title"].is_null());
    assert!(top("6")["title"].is_null());
    // The unrelated code family scores far lower (near-orthogonal chunk).
    let clips_score = cands["8"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["code"] == "73239300")
        .map(|c| c["score"].as_f64().unwrap())
        .unwrap_or(0.0);
    assert!(clips_score < 0.3, "unrelated code must not score near the top: {clips_score}");

    // Evidence: top hit first, with its code and a snippet of its text.
    let evidence = v["evidence"].as_array().unwrap();
    assert!(!evidence.is_empty() && evidence.len() <= 5);
    assert_eq!(evidence[0]["node_id"], ids["c:lamp"]);
    assert!(evidence[0]["similarity"].as_f64().unwrap() > 0.999);
    assert_eq!(evidence[0]["code"], "94055000", "most specific linked code of the chunk");
    assert_eq!(evidence[0]["codes"], json!(["94055000", "9405"]));
    assert_eq!(evidence[0]["snippet"], CHUNK_LAMP);
}

#[tokio::test(flavor = "multi_thread")]
async fn classify_material_confidence_and_llm_path() {
    let state = state();
    let (status, _) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK);

    // The description is normalised (lower-case, abbreviations expanded) and the
    // material is appended to the query string.
    let (status, v) = call(
        &state,
        post_json(
            "/api/classify",
            json!({ "text": "FOOD STORAGE CLIPS S/STEEL 4PK", "material": "Metal/Steel/Tin" }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["query"], "food storage clips stainless steel 4 pack, material: Metal/Steel/Tin");
    // Mock embeddings of different texts are near-orthogonal, so the best
    // similarity is well under the 0.35 default → not confident, but the
    // candidate structure is still returned.
    assert_eq!(v["confident"], false);
    assert!(v["candidates"]["2"].is_array());

    // A hit on a code node itself (not via a chunk) votes with its own code.
    let (status, v) = call(
        &state,
        post_json("/api/classify", json!({ "text": "non-electrical luminaires", "min_similarity": 0.9 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["confident"], true);
    assert_eq!(v["candidates"]["8"][0]["code"], "94055000");
    assert_eq!(v["candidates"]["4"][0]["code"], "9405");
    assert_eq!(v["evidence"][0]["code"], "94055000");
    assert_eq!(v["evidence"][0]["codes"], json!(["94055000"]));
    assert_eq!(v["evidence"][0]["snippet"], "non-electrical luminaires", "title used when no text");

    // min_similarity above the best hit → confident=false.
    let (_, v) = call(
        &state,
        post_json("/api/classify", json!({ "text": CHUNK_LAMP, "min_similarity": 1.5 })),
    )
    .await;
    assert_eq!(v["confident"], false);

    // llm=true: the mock engine echoes the prompt, which lists the candidate
    // codes, so the parser resolves to one of the offered codes. Must not crash.
    let (status, v) = call(&state, post_json("/api/classify", json!({ "text": CHUNK_LAMP, "llm": true }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let choice = &v["llm_choice"];
    assert!(choice.is_object(), "mock echo must yield a parsed choice: {v}");
    let code = choice["code"].as_str().unwrap();
    assert!(["94055000", "9405", "73239300"].contains(&code), "{code}");
    assert!(choice["reason"].is_string());
    assert!(v["llm_raw"].as_str().unwrap().starts_with("[mock-gguf]"));

    // Empty graph: classify still answers (no hits → not confident, empty lists).
    let empty = AppState::default_mock().unwrap();
    let (status, v) = call(&empty, post_json("/api/classify", json!({ "text": "anything", "llm": true }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["confident"], false);
    assert!(v["best_similarity"].is_null());
    assert_eq!(v["evidence"], json!([]));
    assert!(v["llm_choice"].is_null());

    // Missing text → 400.
    let (status, _) = call(&state, post_json("/api/classify", json!({ "material": "Steel" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ===========================================================================
// /api/graph/stats
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn graph_stats_counts_match_ingest() {
    let state = state();
    let (_, empty) = call(&state, get("/api/graph/stats")).await;
    assert_eq!(empty["nodes"], 0);
    assert_eq!(empty["edges"], 0);
    assert_eq!(empty["vector_index_size"], 0);
    assert_eq!(empty["nodes_by_label"], json!({}));

    let (status, _) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK);

    let (status, v) = call(&state, get("/api/graph/stats")).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["nodes"], 5);
    assert_eq!(v["nodes_by_label"]["Chunk"], 2);
    assert_eq!(v["nodes_by_label"]["Tariff"], 3);
    assert_eq!(v["nodes_by_label"]["Heading"], 1);
    assert_eq!(v["nodes_by_label"]["Subheading"], 2);
    assert_eq!(v["edges"], 4);
    assert_eq!(v["vector_index_size"], 4, "s:73239300 has no embedding");
    assert_eq!(v["refs"], 5);
}

// ===========================================================================
// Durable round-trip (ref map rebuilt on reopen)
// ===========================================================================

#[cfg(feature = "durable")]
#[tokio::test(flavor = "multi_thread")]
async fn durable_reopen_rebuilds_ref_map_and_index() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("a_llama_graph_api_durable_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());

    let open = || {
        AppState::open_durable(
            Arc::new(MockEngine::new()),
            &dir,
            OrchestratorConfig::default(),
        )
        .expect("open_durable")
    };

    // --- Session 1: ingest, persist, drop everything. ---
    let ids = {
        let state = open();
        let (status, v) = call(&state, post_json("/api/ingest", fixture().await)).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        state.persist().await.unwrap();
        v["nodes"].clone()
    };

    // --- Session 2: reopen; refs, edges, vectors and code titles survive. ---
    let state = open();
    let store = state.knowledge_store();
    assert_eq!(store.ref_count(), 5, "ref map rebuilt from persisted `ref` properties");
    assert_eq!(store.resolve_ref("c:lamp").unwrap().0, ids["c:lamp"].as_u64().unwrap());

    // An edge referencing refs ingested by the previous process.
    let (status, v) = call(
        &state,
        post_json(
            "/api/ingest",
            json!({ "edges": [ { "src": "c:clips", "dst": "h:9405", "type": "RULING_FOR" } ] }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["edges_created"], 1);

    // Re-ingesting the original payload is still idempotent after reopen.
    let (status, v) = call(&state, post_json("/api/ingest", fixture().await)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["nodes_created"], 0);
    assert_eq!(v["nodes_existing"], 5);
    assert_eq!(v["edges_existing"], 4);
    assert_eq!(v["nodes"], ids, "ids are stable across restarts");

    // Search still finds the node (vector index rebuilt) with its neighbours.
    let (status, v) = call(&state, post_json("/api/search", json!({ "text": CHUNK_LAMP, "k": 1 }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["hits"][0]["node_id"], ids["c:lamp"]);
    assert!(v["hits"][0]["similarity"].as_f64().unwrap() > 0.999);
    assert_eq!(v["hits"][0]["neighbors"].as_array().unwrap().len(), 2);

    // Classify titles resolve through the rebuilt code map.
    let (_, v) = call(&state, post_json("/api/classify", json!({ "text": CHUNK_LAMP }))).await;
    assert_eq!(v["candidates"]["4"][0]["code"], "9405");
    assert_eq!(v["candidates"]["4"][0]["title"], "Lamps and lighting fittings");

    let (_, stats) = call(&state, get("/api/graph/stats")).await;
    assert_eq!(stats["nodes"], 5);
    assert_eq!(stats["edges"], 5);
    assert_eq!(stats["vector_index_size"], 4);
    assert_eq!(stats["refs"], 5);
}
