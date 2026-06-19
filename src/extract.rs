//! Task 9 (M2) — LLM-based fact / entity extraction.
//!
//! An [`Extractor`] runs an LLM pass over a prompt→response interaction to
//! distill structured [`Fact`](crate::knowledge_store::KnowledgeStore::upsert_fact)
//! and `Entity` nodes plus `MENTIONS` / `RELATES_TO` / `DERIVED_FROM` edges into
//! the [`KnowledgeStore`], so learned knowledge becomes **independently**
//! GraphRAG-retrievable rather than only existing as whole-interaction
//! `CacheEntry` nodes (DESIGN.md §4, task 9; Q6).
//!
//! ## Shape of the pipeline
//!
//! 1. [`extraction_prompt`] builds an instruction asking the chat model to emit
//!    a single JSON object describing salient facts, named entities, and
//!    relations from the exchange.
//! 2. [`Extractor::extract_and_store`] runs the engine, then
//! 3. [`parse_extraction`] tolerantly recovers the JSON object from whatever
//!    prose/markdown the model wrapped it in, and finally
//! 4. [`Extractor::store_extraction`] embeds each fact and upserts the
//!    nodes + edges into the [`KnowledgeStore`].
//!
//! ## Tolerance is the whole game
//!
//! Real chat models — especially small ones — wrap JSON in prose, fence it in
//! ```` ```json ```` blocks, append commentary, or emit nothing parseable at
//! all. [`parse_extraction`] is a **pure, panic-free** function that extracts
//! the first balanced `{...}` object, parses it leniently (every field
//! defaults, unknown junk is ignored), and returns an empty [`Extraction`] when
//! it finds nothing usable. It never errors. This makes the LLM-facing edge of
//! the system unit-testable without a real model, and keeps a garbage
//! generation from ever failing a user request.
//!
//! ## Quality caveat
//!
//! Extraction quality is a function of the chat model: a 0.5B GGUF distills
//! facts crudely and may emit malformed JSON the parser discards. The
//! *mechanism* — extract → embed → upsert → link, made GraphRAG-retrievable —
//! is what this module delivers; swapping in a stronger model improves the
//! harvest without any code change.
//!
//! ## Cost
//!
//! Extraction is an **extra LLM completion per interaction**, so it is opt-in
//! (`OrchestratorConfig.extract_facts`, default `false`). Enable it when the
//! richer, connected knowledge graph is worth roughly doubling per-request
//! inference cost on the cache-miss path.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Deserialize;

use astraea_core::types::NodeId;

use crate::inference::InferenceEngine;
use crate::knowledge_store::KnowledgeStore;

/// Upper bound on the bytes stored in a `Fact` node's `text` property.
///
/// astraedb #26: a node record carrying `text` **plus** a 768-dim embedding
/// (~3 KB) must stay under the ~8 KB single-page limit. Facts are short by
/// nature; this 2 KB ceiling keeps even a pathological extraction well clear of
/// the limit. Over-long fact text is truncated at a UTF-8 char boundary.
pub const MAX_FACT_TEXT_BYTES: usize = 2048;

/// Default confidence assigned to a fact whose `confidence` field the model
/// omitted (or emitted as something unparseable).
const DEFAULT_CONFIDENCE: f32 = 0.5;

/// Entity `kind` used for an entity that a fact mentions (or a relation
/// references) but that the model never listed in the top-level `entities`
/// array with an explicit kind.
const UNKNOWN_KIND: &str = "unknown";

// ===========================================================================
// Structured extraction types
// ===========================================================================

/// One distilled fact: a short, self-contained statement plus the names of the
/// entities it mentions and a model-assigned confidence in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct ExtractedFact {
    /// The fact text (e.g. `"Rust is a systems programming language."`).
    #[serde(default)]
    pub text: String,
    /// Confidence in `[0, 1]`. Defaults to [`DEFAULT_CONFIDENCE`] when absent.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Names of entities this fact mentions; each links `(Fact)-MENTIONS->(Entity)`.
    #[serde(default)]
    pub entities: Vec<String>,
}

fn default_confidence() -> f32 {
    DEFAULT_CONFIDENCE
}

/// A named entity with a coarse type (`person`, `language`, `product`, …).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct ExtractedEntity {
    /// The entity name (the dedup identity, together with `kind`).
    #[serde(default)]
    pub name: String,
    /// The entity kind/type. Defaults to [`UNKNOWN_KIND`] when absent.
    #[serde(default = "default_kind")]
    pub kind: String,
}

fn default_kind() -> String {
    UNKNOWN_KIND.to_string()
}

/// A directed relation between two named entities:
/// `(from)-RELATES_TO->(to)` (the `kind` is carried for readability/future use).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct ExtractedRelation {
    /// Source entity name.
    #[serde(default)]
    pub from: String,
    /// Target entity name.
    #[serde(default)]
    pub to: String,
    /// Relationship label (e.g. `"created_by"`, `"part_of"`).
    #[serde(default)]
    pub kind: String,
}

/// The full structured result of one extraction pass.
///
/// This is exactly the JSON shape [`extraction_prompt`] asks the model to emit;
/// it derives [`Deserialize`] with every field defaulting so a partial or
/// junk-laden object still parses into a best-effort value.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct Extraction {
    /// Distilled facts.
    #[serde(default)]
    pub facts: Vec<ExtractedFact>,
    /// Named entities with explicit kinds.
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    /// Relations between entities.
    #[serde(default)]
    pub relations: Vec<ExtractedRelation>,
}

impl Extraction {
    /// `true` when nothing was extracted (no facts, entities, or relations).
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty() && self.entities.is_empty() && self.relations.is_empty()
    }
}

/// Counts of what [`Extractor::store_extraction`] / [`Extractor::extract_and_store`]
/// committed to the graph. Counts reflect items **successfully stored**, so a
/// per-item store failure (logged and skipped) is not counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtractionStats {
    /// `Fact` nodes upserted.
    pub facts: usize,
    /// Distinct `Entity` nodes upserted (top-level + fact-mentioned + relation ends).
    pub entities: usize,
    /// `RELATES_TO` edges created.
    pub relations: usize,
}

// ===========================================================================
// Prompt construction
// ===========================================================================

/// Build the extraction instruction for a prompt→response exchange.
///
/// The model is asked to return **only** a single JSON object of the documented
/// shape. We still parse defensively ([`parse_extraction`]) because models
/// routinely ignore "only JSON" instructions.
pub fn extraction_prompt(prompt: &str, response: &str) -> String {
    format!(
        "You are an information-extraction system. Read the following exchange \
between a User and an Assistant and extract the salient, durable facts, the \
named entities, and the relations between entities.\n\
\n\
Return ONLY a single JSON object (no prose, no markdown code fences) of exactly \
this shape:\n\
{{\n\
  \"facts\": [\n\
    {{ \"text\": \"<a short standalone factual statement>\", \"confidence\": <number 0.0-1.0>, \"entities\": [\"<entity name>\"] }}\n\
  ],\n\
  \"entities\": [\n\
    {{ \"name\": \"<entity name>\", \"kind\": \"<type, e.g. person, place, organization, language, product, concept>\" }}\n\
  ],\n\
  \"relations\": [\n\
    {{ \"from\": \"<entity name>\", \"to\": \"<entity name>\", \"kind\": \"<relationship>\" }}\n\
  ]\n\
}}\n\
\n\
Rules:\n\
- Extract only facts stated or clearly implied in the exchange.\n\
- Keep each fact text under 200 characters and self-contained.\n\
- Use an empty array [] for any section with nothing to extract.\n\
- Output the JSON object and nothing else.\n\
\n\
User: {prompt}\n\
Assistant: {response}\n",
    )
}

// ===========================================================================
// Tolerant parsing (the pure, unit-testable core)
// ===========================================================================

/// Parse an [`Extraction`] out of raw LLM output, tolerating prose, markdown
/// fences, and trailing commentary.
///
/// Strategy: scan for each `{`, extract the **balanced** `{...}` substring
/// starting there (respecting strings and escapes), and try to deserialize it.
/// The first candidate that parses wins. If none parse, returns an empty
/// [`Extraction`]. This function is **pure** and never panics or errors —
/// malformed/empty input simply yields `Extraction::default()`.
pub fn parse_extraction(llm_output: &str) -> Extraction {
    let bytes = llm_output.as_bytes();
    let mut search_from = 0;
    while let Some(rel) = find_byte(&bytes[search_from..], b'{') {
        let start = search_from + rel;
        if let Some(end) = balanced_object_end(llm_output, start) {
            let candidate = &llm_output[start..=end];
            if let Ok(extraction) = serde_json::from_str::<Extraction>(candidate) {
                return extraction;
            }
            // Parsed as a JSON object but not our shape (or invalid JSON);
            // advance past this `{` and keep looking for another candidate.
        }
        search_from = start + 1;
    }
    Extraction::default()
}

/// Index of the first occurrence of `needle` in `hay`, if any.
fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
}

/// Given `s` and a byte index `start` pointing at a `{`, return the byte index
/// (inclusive) of the matching `}`, honouring JSON string literals and escapes.
/// Returns `None` if the braces never balance (truncated output).
fn balanced_object_end(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    debug_assert_eq!(bytes[start], b'{');
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

// ===========================================================================
// The extractor
// ===========================================================================

/// Distils [`Fact`]/`Entity` nodes + edges from interactions and stores them in
/// the [`KnowledgeStore`].
///
/// Holds shared handles to the inference [`engine`](InferenceEngine) (for the
/// extraction completion and for embedding each fact) and the [`KnowledgeStore`]
/// it writes into. Cheap to clone (two `Arc`s).
#[derive(Clone)]
pub struct Extractor {
    engine: Arc<dyn InferenceEngine>,
    store: Arc<KnowledgeStore>,
}

impl Extractor {
    /// Build an extractor over a shared engine and knowledge store.
    pub fn new(engine: Arc<dyn InferenceEngine>, store: Arc<KnowledgeStore>) -> Self {
        Self { engine, store }
    }

    /// Run the full pipeline: extraction completion → parse → store.
    ///
    /// 1. Ask the engine to extract structured knowledge from `(prompt,
    ///    response)`.
    /// 2. [`parse_extraction`] the (messy) output into an [`Extraction`].
    /// 3. [`store_extraction`](Self::store_extraction) embeds + upserts the
    ///    facts/entities/relations, linking facts to `derived_from` when given.
    ///
    /// `source_id` is the dedup identity for the facts (e.g. the eunomia entry
    /// id); `derived_from` is the `CacheEntry` `NodeId` the facts came from, if
    /// known (`None` is fine — see [`store_extraction`](Self::store_extraction)).
    ///
    /// Returns counts. A model that emits no parseable JSON (e.g. `MockEngine`,
    /// whose `complete` returns a canned non-JSON echo) yields a zero-count
    /// [`ExtractionStats`] without erroring — a graceful no-op.
    pub async fn extract_and_store(
        &self,
        prompt: &str,
        response: &str,
        source_id: &str,
        derived_from: Option<NodeId>,
    ) -> Result<ExtractionStats> {
        let raw = self
            .engine
            .complete(&extraction_prompt(prompt, response), None)
            .await?;
        let extraction = parse_extraction(&raw);
        self.store_extraction(&extraction, source_id, derived_from)
            .await
    }

    /// Store an already-parsed [`Extraction`] into the [`KnowledgeStore`].
    ///
    /// Split out from [`extract_and_store`](Self::extract_and_store) so the
    /// store path is testable without depending on the LLM emitting JSON: a
    /// caller can construct an [`Extraction`] directly and assert the resulting
    /// `Fact`/`Entity` nodes exist and are GraphRAG-retrievable.
    ///
    /// For each fact: embed its (bounded, astraedb #26) text via the engine and
    /// `upsert_fact`; link `DERIVED_FROM` to `derived_from` when `Some`; resolve
    /// each mentioned entity and link `MENTIONS`. For each relation: resolve both
    /// ends and link `RELATES_TO`. Entities carry an embedding-less node (`None`)
    /// — they are reached structurally via edges from embedded facts, not by
    /// vector search.
    ///
    /// Resilience: an embed or store failure on **one** item is logged and
    /// skipped; it never aborts the whole call. Counts reflect only
    /// successfully-stored items.
    ///
    /// > `derived_from` note: in M1's flow the `CacheEntry` node may not exist
    /// > yet at extraction time (it is minted later, when Eunomia tiering runs),
    /// > so the orchestrator passes `None`. The `Fact` is still fully
    /// > retrievable on its own embedding; the `DERIVED_FROM` provenance edge is
    /// > simply absent until/unless a caller supplies the cache-entry id.
    pub async fn store_extraction(
        &self,
        extraction: &Extraction,
        source_id: &str,
        derived_from: Option<NodeId>,
    ) -> Result<ExtractionStats> {
        let mut stats = ExtractionStats::default();
        let now = unix_now();

        // Map an entity *name* to the kind the model gave it in the top-level
        // `entities` list, so facts that mention the same name resolve to the
        // same (kind, name) node rather than a separate "unknown"-kind sibling.
        let mut kind_of: HashMap<&str, &str> = HashMap::new();
        for e in &extraction.entities {
            if !e.name.is_empty() {
                kind_of.insert(e.name.as_str(), e.kind.as_str());
            }
        }

        // Local cache of resolved entity NodeIds, keyed by (kind, name), so we
        // upsert each entity once and can count distinct entities.
        let mut entity_ids: HashMap<(String, String), NodeId> = HashMap::new();

        // Pre-create the explicitly-listed entities (so even an entity no fact
        // mentions still lands in the graph).
        for e in &extraction.entities {
            if e.name.is_empty() {
                continue;
            }
            self.resolve_entity(&mut entity_ids, &e.name, &e.kind);
        }

        // Facts.
        for fact in &extraction.facts {
            let text = fact.text.trim();
            if text.is_empty() {
                continue;
            }
            let bounded = truncate_to_bytes(text, MAX_FACT_TEXT_BYTES);

            let embedding = match self.engine.embed(bounded).await {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("a-llama: extract: skipping fact (embed failed): {e}");
                    continue;
                }
            };

            let confidence = fact.confidence.clamp(0.0, 1.0);
            let fact_id =
                match self
                    .store
                    .upsert_fact(bounded, source_id, confidence, embedding, now)
                {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("a-llama: extract: skipping fact (upsert failed): {e}");
                        continue;
                    }
                };
            stats.facts += 1;

            // Provenance edge to the originating CacheEntry, when known.
            if let Some(src) = derived_from {
                if let Err(e) = self.store.link_derived_from(fact_id, src) {
                    eprintln!("a-llama: extract: DERIVED_FROM link failed: {e}");
                }
            }

            // Link mentioned entities.
            for name in &fact.entities {
                if name.is_empty() {
                    continue;
                }
                let kind = kind_of.get(name.as_str()).copied().unwrap_or(UNKNOWN_KIND);
                if let Some(entity_id) = self.resolve_entity(&mut entity_ids, name, kind) {
                    if let Err(e) = self.store.link_mentions(fact_id, entity_id) {
                        eprintln!("a-llama: extract: MENTIONS link failed: {e}");
                    }
                }
            }
        }

        // Relations.
        for rel in &extraction.relations {
            if rel.from.is_empty() || rel.to.is_empty() {
                continue;
            }
            let from_kind = kind_of.get(rel.from.as_str()).copied().unwrap_or(UNKNOWN_KIND);
            let to_kind = kind_of.get(rel.to.as_str()).copied().unwrap_or(UNKNOWN_KIND);
            let from_id = self.resolve_entity(&mut entity_ids, &rel.from, from_kind);
            let to_id = self.resolve_entity(&mut entity_ids, &rel.to, to_kind);
            if let (Some(from_id), Some(to_id)) = (from_id, to_id) {
                match self.store.link_relates_to(from_id, to_id) {
                    Ok(_) => stats.relations += 1,
                    Err(e) => eprintln!("a-llama: extract: RELATES_TO link failed: {e}"),
                }
            }
        }

        stats.entities = entity_ids.len();
        Ok(stats)
    }

    /// Upsert an `(name, kind)` entity once, caching its `NodeId`. Returns the
    /// id, or `None` if the upsert failed (logged).
    fn resolve_entity(
        &self,
        cache: &mut HashMap<(String, String), NodeId>,
        name: &str,
        kind: &str,
    ) -> Option<NodeId> {
        let key = (kind.to_string(), name.to_string());
        if let Some(&id) = cache.get(&key) {
            return Some(id);
        }
        match self.store.upsert_entity(name, kind, None) {
            Ok(id) => {
                cache.insert(key, id);
                Some(id)
            }
            Err(e) => {
                eprintln!("a-llama: extract: skipping entity `{name}` (upsert failed): {e}");
                None
            }
        }
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Current Unix time in seconds (0 on a pre-epoch clock — pathological only).
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Truncate `s` to at most `max` bytes at a UTF-8 char boundary.
fn truncate_to_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::MockEngine;

    // -------------------------------------------------------------------
    // parse_extraction — tolerance fixtures
    // -------------------------------------------------------------------

    #[test]
    fn parse_clean_json() {
        let raw = r#"{
            "facts": [
                {"text": "Rust is a systems language.", "confidence": 0.9, "entities": ["Rust"]}
            ],
            "entities": [{"name": "Rust", "kind": "language"}],
            "relations": [{"from": "Rust", "to": "Mozilla", "kind": "created_by"}]
        }"#;
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "Rust is a systems language.");
        assert!((e.facts[0].confidence - 0.9).abs() < 1e-6);
        assert_eq!(e.facts[0].entities, vec!["Rust".to_string()]);
        assert_eq!(e.entities.len(), 1);
        assert_eq!(e.entities[0].kind, "language");
        assert_eq!(e.relations.len(), 1);
        assert_eq!(e.relations[0].kind, "created_by");
    }

    #[test]
    fn parse_json_inside_markdown_fence() {
        let raw = "Sure, here is the extraction:\n\n```json\n{\n  \"facts\": [{\"text\": \"The sky is blue.\"}],\n  \"entities\": [],\n  \"relations\": []\n}\n```\nHope that helps!";
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "The sky is blue.");
        // confidence absent → default.
        assert!((e.facts[0].confidence - DEFAULT_CONFIDENCE).abs() < 1e-6);
    }

    #[test]
    fn parse_json_with_leading_and_trailing_prose() {
        let raw = "I analyzed the exchange. {\"facts\": [{\"text\": \"Water boils at 100C.\", \"confidence\": 1.0}], \"entities\": [{\"name\": \"Water\", \"kind\": \"substance\"}]} That is all.";
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "Water boils at 100C.");
        assert_eq!(e.entities.len(), 1);
        assert_eq!(e.entities[0].name, "Water");
        // relations absent → empty.
        assert!(e.relations.is_empty());
    }

    #[test]
    fn parse_skips_non_matching_brace_before_real_object() {
        // A stray `{...}` that isn't our shape precedes the real object.
        let raw = "note {oops not json} then {\"facts\": [{\"text\": \"A fact.\"}]}";
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "A fact.");
    }

    #[test]
    fn parse_partial_fields_default() {
        // Only facts present; an entity missing its kind defaults to "unknown".
        let raw = r#"{"facts": [{"text": "Partial."}], "entities": [{"name": "Foo"}]}"#;
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert!((e.facts[0].confidence - DEFAULT_CONFIDENCE).abs() < 1e-6);
        assert!(e.facts[0].entities.is_empty());
        assert_eq!(e.entities[0].kind, UNKNOWN_KIND);
    }

    #[test]
    fn parse_ignores_unknown_junk_fields() {
        let raw = r#"{"facts": [{"text": "Has junk.", "weird": 7}], "extra_top": true, "entities": []}"#;
        let e = parse_extraction(raw);
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "Has junk.");
    }

    #[test]
    fn parse_malformed_returns_empty() {
        for raw in [
            "",
            "no json here at all",
            "{ this is not valid json",
            "{\"facts\": [unclosed",
            "}{}{ broken",
        ] {
            let e = parse_extraction(raw);
            assert!(e.is_empty(), "expected empty Extraction for {raw:?}");
        }
    }

    #[test]
    fn parse_empty_object_is_empty_extraction() {
        let e = parse_extraction("{}");
        assert!(e.is_empty());
    }

    #[test]
    fn balanced_object_end_handles_braces_in_strings() {
        let s = r#"{"text": "a } brace { inside"}"#;
        let end = balanced_object_end(s, 0).unwrap();
        assert_eq!(end, s.len() - 1, "must close at the final brace, not one in a string");
    }

    // -------------------------------------------------------------------
    // store_extraction — store path without depending on the LLM
    // -------------------------------------------------------------------

    fn extractor_with_dim(dim: usize) -> Extractor {
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::with_dim(dim));
        let store = Arc::new(KnowledgeStore::with_dim(dim).unwrap());
        Extractor::new(engine, store)
    }

    #[tokio::test]
    async fn store_extraction_creates_nodes_and_edges() {
        let ex = extractor_with_dim(768);
        let extraction = Extraction {
            facts: vec![ExtractedFact {
                text: "Rust is a memory-safe systems language.".to_string(),
                confidence: 0.92,
                entities: vec!["Rust".to_string()],
            }],
            entities: vec![ExtractedEntity {
                name: "Rust".to_string(),
                kind: "language".to_string(),
            }],
            relations: vec![ExtractedRelation {
                from: "Rust".to_string(),
                to: "Mozilla".to_string(),
                kind: "created_by".to_string(),
            }],
        };

        let stats = ex
            .store_extraction(&extraction, "src-1", None)
            .await
            .unwrap();
        assert_eq!(stats.facts, 1);
        // Rust (language) + Mozilla (unknown, from the relation) = 2 entities.
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.relations, 1);

        let store = &ex.store;
        assert_eq!(store.graph_ops().find_by_label("Fact").unwrap().len(), 1);
        assert_eq!(store.graph_ops().find_by_label("Entity").unwrap().len(), 2);
    }

    #[tokio::test]
    async fn stored_fact_is_graphrag_retrievable_by_vector_search() {
        let ex = extractor_with_dim(768);
        let fact_text = "The speed of light is about 299,792 km/s.";
        let extraction = Extraction {
            facts: vec![ExtractedFact {
                text: fact_text.to_string(),
                confidence: 0.99,
                entities: vec![],
            }],
            ..Default::default()
        };
        ex.store_extraction(&extraction, "src-light", None)
            .await
            .unwrap();

        // Embed the fact text and vector-search the store — it must come back.
        let query = ex.engine.embed(fact_text).await.unwrap();
        let results = ex.store.vector_index_ref().search(&query, 1).unwrap();
        assert!(!results.is_empty(), "fact must be indexed in HNSW");
        let node = ex
            .store
            .graph_ops()
            .get_node(results[0].node_id)
            .unwrap()
            .unwrap();
        assert!(node.labels.contains(&"Fact".to_string()));
        assert_eq!(node.properties["text"], fact_text);
    }

    #[tokio::test]
    async fn store_extraction_mentions_and_derived_from_edges() {
        use astraea_core::types::Direction;

        let ex = extractor_with_dim(768);

        // A stand-in CacheEntry to point DERIVED_FROM at.
        let cache_id = ex
            .store
            .upsert_cache_entry(
                "ns",
                "e-1",
                serde_json::json!({"prompt": "p", "response": "r"}),
                &[],
                ex.engine.embed("p").await.unwrap(),
                0,
                0,
            )
            .unwrap();

        let extraction = Extraction {
            facts: vec![ExtractedFact {
                text: "AstraeaDB is an in-process graph database.".to_string(),
                confidence: 0.9,
                entities: vec!["AstraeaDB".to_string()],
            }],
            entities: vec![ExtractedEntity {
                name: "AstraeaDB".to_string(),
                kind: "product".to_string(),
            }],
            ..Default::default()
        };

        ex.store_extraction(&extraction, "e-1", Some(cache_id))
            .await
            .unwrap();

        let fact_id = ex.store.graph_ops().find_by_label("Fact").unwrap()[0];
        let outgoing: Vec<NodeId> = ex
            .store
            .graph_ops()
            .neighbors(fact_id, Direction::Outgoing)
            .unwrap()
            .into_iter()
            .map(|(_edge, node)| node)
            .collect();
        // The fact has DERIVED_FROM (cache) + MENTIONS (AstraeaDB) = 2 edges.
        assert_eq!(outgoing.len(), 2, "fact must link DERIVED_FROM and MENTIONS");
        assert!(outgoing.contains(&cache_id), "DERIVED_FROM must reach the cache entry");
    }

    #[tokio::test]
    async fn store_extraction_dedups_mentioned_entity_with_listed_kind() {
        // A fact mentions "Rust"; the entities list gives "Rust" kind "language".
        // The MENTIONS edge must resolve to the SAME (language, Rust) node, not a
        // separate "unknown"-kind sibling.
        let ex = extractor_with_dim(768);
        let extraction = Extraction {
            facts: vec![ExtractedFact {
                text: "Rust is fast.".to_string(),
                confidence: 0.8,
                entities: vec!["Rust".to_string()],
            }],
            entities: vec![ExtractedEntity {
                name: "Rust".to_string(),
                kind: "language".to_string(),
            }],
            ..Default::default()
        };
        let stats = ex.store_extraction(&extraction, "s", None).await.unwrap();
        assert_eq!(stats.entities, 1, "the mentioned entity must dedup with the listed one");
        assert_eq!(ex.store.graph_ops().find_by_label("Entity").unwrap().len(), 1);
    }

    #[tokio::test]
    async fn store_extraction_skips_empty_facts() {
        let ex = extractor_with_dim(768);
        let extraction = Extraction {
            facts: vec![
                ExtractedFact { text: "   ".to_string(), confidence: 0.5, entities: vec![] },
                ExtractedFact { text: "Real.".to_string(), confidence: 0.5, entities: vec![] },
            ],
            ..Default::default()
        };
        let stats = ex.store_extraction(&extraction, "s", None).await.unwrap();
        assert_eq!(stats.facts, 1, "blank-text fact is skipped");
    }

    #[tokio::test]
    async fn store_extraction_bounds_fact_text() {
        let ex = extractor_with_dim(768);
        let huge = "x".repeat(MAX_FACT_TEXT_BYTES * 4);
        let extraction = Extraction {
            facts: vec![ExtractedFact { text: huge, confidence: 0.5, entities: vec![] }],
            ..Default::default()
        };
        ex.store_extraction(&extraction, "s", None).await.unwrap();
        let fact_id = ex.store.graph_ops().find_by_label("Fact").unwrap()[0];
        let node = ex.store.graph_ops().get_node(fact_id).unwrap().unwrap();
        let stored = node.properties["text"].as_str().unwrap();
        assert!(stored.len() <= MAX_FACT_TEXT_BYTES, "stored fact text must be bounded (#26)");
    }

    // -------------------------------------------------------------------
    // extract_and_store over MockEngine — graceful no-op (non-JSON output)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn extract_and_store_over_mock_is_graceful_noop() {
        // MockEngine::complete returns a canned non-JSON echo, so parse yields
        // an empty Extraction and nothing is stored — but it must not error.
        let ex = extractor_with_dim(768);
        let stats = ex
            .extract_and_store("What is Rust?", "Rust is a language.", "src-x", None)
            .await
            .unwrap();
        assert_eq!(stats, ExtractionStats::default(), "non-JSON output stores nothing");
        assert_eq!(ex.store.graph_ops().find_by_label("Fact").unwrap().len(), 0);
    }

    #[test]
    fn extraction_prompt_mentions_schema_and_inputs() {
        let p = extraction_prompt("PROMPT_TEXT", "RESPONSE_TEXT");
        assert!(p.contains("PROMPT_TEXT"));
        assert!(p.contains("RESPONSE_TEXT"));
        assert!(p.contains("\"facts\""));
        assert!(p.contains("\"entities\""));
        assert!(p.contains("\"relations\""));
    }
}
