//! Eunomia's data model: composite keys, JSON values, and cache metadata.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// A high-dimensional semantic embedding used for nearest-neighbor recall.
pub type Embedding = Vec<f32>;

/// A logical isolation boundary for a set of cache entries — one agent
/// session, tenant, or user. Namespaces keep working memory from
/// cross-contaminating (PRD 6.2) and pin a single embedding dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace(pub String);

impl Namespace {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Configuration captured when a namespace is created (Decision 7637). The
/// embedding dimension is pinned here — set once, up front — rather than
/// inferred on first insert, so a namespace's vector space is fixed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceConfig {
    pub namespace: Namespace,
    /// Embedding dimension for every vector stored in this namespace.
    pub dim: usize,
}

impl NamespaceConfig {
    pub fn new(namespace: Namespace, dim: usize) -> Self {
        Self { namespace, dim }
    }
}

/// A composite cache key: a string identifier plus an optional semantic
/// embedding. Exact lookups use [`Key::id`]; vector search uses the embedding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Embedding>,
}

impl Key {
    /// A pure string key with no embedding (exact K/V only).
    pub fn id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            embedding: None,
        }
    }

    /// A composite key carrying a semantic embedding for vector recall.
    pub fn with_embedding(id: impl Into<String>, embedding: Embedding) -> Self {
        Self {
            id: id.into(),
            embedding: Some(embedding),
        }
    }
}

/// Tags and temporal data driving cache eviction and context decay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(default)]
    pub tags: Vec<String>,
    /// Time-to-live in seconds; `None` means the entry never expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_secs: Option<u64>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
}

impl Default for Metadata {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            tags: Vec::new(),
            ttl_secs: None,
            created_at: now,
            last_accessed: now,
        }
    }
}

impl Metadata {
    /// Whether this entry has outlived its TTL as of `now`. Entries with no
    /// `ttl_secs` never expire. Clock skew (a `created_at` in the future) is
    /// treated as not-expired.
    pub fn is_expired(&self, now: std::time::SystemTime) -> bool {
        match self.ttl_secs {
            None => false,
            Some(ttl) => now
                .duration_since(self.created_at)
                .map(|age| age.as_secs() >= ttl)
                .unwrap_or(false),
        }
    }
}

/// A stored cache entry: the composite key, an arbitrary JSON value (which may
/// embed AstraeaDB node IDs / URIs), and its metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub value: serde_json::Value,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn ttl_expiry() {
        let mut m = Metadata::default();
        // No TTL => never expires.
        assert!(!m.is_expired(SystemTime::now()));

        m.ttl_secs = Some(10);
        m.created_at = SystemTime::now() - Duration::from_secs(20);
        assert!(m.is_expired(SystemTime::now()), "20s-old, 10s ttl => expired");

        m.created_at = SystemTime::now();
        assert!(!m.is_expired(SystemTime::now()), "fresh entry => live");
    }

    #[test]
    fn namespace_display() {
        assert_eq!(Namespace::new("agent-7").to_string(), "agent-7");
    }
}
