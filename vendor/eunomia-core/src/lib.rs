//! `eunomia-core` — the independent heart of Eunomia.
//!
//! This crate implements Eunomia's working-memory engine with **no AstraeaDB
//! dependency**, so the cache compiles and runs fully standalone:
//!
//! - [`model`] — the data model: composite [`Key`]s (string id + optional
//!   semantic [`Embedding`]), arbitrary JSON values, and [`Metadata`] (tags,
//!   TTL, timestamps) for cache eviction / context decay.
//! - [`store`] — the in-memory storage engine: an exact O(1) key/value map
//!   plus a vector index for approximate nearest-neighbor search.
//! - [`query`] — the query planner that routes a request to a K/V lookup, a
//!   vector search, or a hybrid of both.
//! - [`persistence`] — write-ahead logging and snapshotting for crash recovery
//!   and fast cold-starts.
//!
//! AstraeaDB integration (long-term tiering) lives behind the optional
//! `eunomia-astraea` bridge crate and is never linked here.

pub mod model;
pub mod persistence;
pub mod query;
pub mod store;
pub mod tier;
mod vector;

pub use model::{Embedding, Entry, Key, Metadata, Namespace, NamespaceConfig};
pub use persistence::{recover, FsyncPolicy, Snapshot, Wal, WalRecord};
pub use query::{plan, Filter, Query, QueryPlan};
pub use store::{Hit, Store, StoreError};
pub use tier::Tier;
