//! The `Tier` seam: Eunomia's outward boundary to a long-term store.
//!
//! `eunomia-core` *defines* this trait but never implements it against any
//! specific backend. The optional `eunomia-astraea` crate implements it over an
//! AstraeaDB graph. Keeping the trait here — with only std / serde / anyhow
//! types in its signatures — is what lets `eunomia-server` hold an
//! `Option<Arc<dyn Tier>>` without ever linking AstraeaDB. **No `astraea-*`
//! type may appear in this module.**

use crate::model::Entry;

/// A long-term storage tier that cold or promoted entries can be flushed into,
/// and from which opaque references can be resolved back to values.
///
/// Implementors live outside `eunomia-core` (e.g. `eunomia-astraea`). The trait
/// is object-safe so the server can hold it as `dyn Tier`.
pub trait Tier: Send + Sync {
    /// Flush an entry into the long-term tier, returning an opaque reference
    /// (e.g. an `astraea://node/<id>` URI) that can later be [`resolve`]d.
    ///
    /// [`resolve`]: Tier::resolve
    fn flush(&self, entry: &Entry) -> anyhow::Result<String>;

    /// Resolve a reference previously returned by [`Tier::flush`] (or embedded
    /// in a cached JSON value) back to its stored value, if present. The
    /// reference is an opaque string to `eunomia-core`; the implementor owns
    /// its scheme.
    fn resolve(&self, reference: &str) -> anyhow::Result<Option<serde_json::Value>>;
}

#[cfg(test)]
mod tests {
    use super::Tier;

    // Compile-time proof that `Tier` is object-safe — `eunomia-server` relies on
    // holding it as `Option<Arc<dyn Tier>>`.
    #[allow(dead_code)]
    fn assert_object_safe(_t: &dyn Tier) {}
}
