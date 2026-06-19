//! The query planner: routes a request to K/V, vector, or hybrid execution.
//!
//! Planning (`plan`) is a pure routing function. Execution lives on
//! [`Store::execute`](crate::store::Store::execute), which runs the plan
//! against the data.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::model::{Embedding, Entry};

/// A declarative filter applied to candidate entries in a [`Query::Hybrid`]
/// search: an entry must carry **all** `tags`, and — if `exclude_expired` — it
/// must not be past its TTL as of evaluation time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filter {
    /// The entry must contain every tag listed here.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Drop entries whose TTL has elapsed (the "ttl > 0" constraint).
    #[serde(default)]
    pub exclude_expired: bool,
}

impl Filter {
    /// Whether `entry` passes this filter as of `now`.
    pub fn matches(&self, entry: &Entry, now: SystemTime) -> bool {
        if self.exclude_expired && entry.metadata.is_expired(now) {
            return false;
        }
        self.tags.iter().all(|t| entry.metadata.tags.contains(t))
    }
}

/// A request against the store, as received from an agent.
#[derive(Clone, Debug)]
pub enum Query {
    /// Exact key/value lookup by string id.
    Exact { id: String },
    /// Nearest-neighbor search by embedding.
    Semantic { embedding: Embedding, k: usize },
    /// Semantic search constrained by a [`Filter`] (tags + TTL).
    Hybrid {
        embedding: Embedding,
        k: usize,
        filter: Filter,
    },
}

/// The execution strategy the planner chose for a [`Query`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryPlan {
    KvLookup,
    VectorSearch,
    HybridScan,
}

/// Decide how to execute a query. Cost-based planning is tracked in DESIGN.md;
/// this is the routing skeleton.
pub fn plan(query: &Query) -> QueryPlan {
    match query {
        Query::Exact { .. } => QueryPlan::KvLookup,
        Query::Semantic { .. } => QueryPlan::VectorSearch,
        Query::Hybrid { .. } => QueryPlan::HybridScan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Key, Metadata};
    use std::time::Duration;

    #[test]
    fn routes_each_query_kind() {
        assert_eq!(plan(&Query::Exact { id: "k".into() }), QueryPlan::KvLookup);
        assert_eq!(
            plan(&Query::Semantic {
                embedding: vec![0.0],
                k: 5
            }),
            QueryPlan::VectorSearch
        );
        assert_eq!(
            plan(&Query::Hybrid {
                embedding: vec![0.0],
                k: 5,
                filter: Filter::default(),
            }),
            QueryPlan::HybridScan
        );
    }

    fn tagged(tags: &[&str], ttl: Option<u64>, age_secs: u64) -> Entry {
        let mut m = Metadata {
            tags: tags.iter().map(|s| s.to_string()).collect(),
            ttl_secs: ttl,
            ..Default::default()
        };
        m.created_at = SystemTime::now() - Duration::from_secs(age_secs);
        Entry {
            key: Key::id("x"),
            value: serde_json::Value::Null,
            metadata: m,
        }
    }

    #[test]
    fn filter_requires_all_tags_and_excludes_expired() {
        let now = SystemTime::now();
        let f = Filter {
            tags: vec!["python".into()],
            exclude_expired: true,
        };
        assert!(f.matches(&tagged(&["python", "perf"], None, 0), now));
        assert!(!f.matches(&tagged(&["rust"], None, 0), now), "missing tag");
        assert!(
            !f.matches(&tagged(&["python"], Some(1), 60), now),
            "expired entry excluded"
        );
    }
}
