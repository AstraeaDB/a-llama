//! Crash recovery and cold-start: write-ahead log + snapshots.
//!
//! Durability model (DESIGN.md §5):
//! - The [`Wal`] is an append-only file of length-prefixed, CRC-checked records
//!   written before a write is acknowledged. On startup the WAL is replayed; a
//!   torn or corrupt tail (a partial final write interrupted by a crash) is
//!   truncated at the last intact record.
//! - A [`Snapshot`] captures the full key/value set. The HNSW index is
//!   *derived* state — rebuilt from the entries' embeddings on load — so the
//!   snapshot need not persist the index itself (this also compacts tombstones).
//! - [`recover`] loads the latest snapshot, then replays the WAL tail.
//!
//! Record payloads are JSON (`serde_json`): entries hold arbitrary
//! `serde_json::Value`s, so a self-describing codec keeps round-tripping them
//! (and `SystemTime`) trivial. Each payload is still binary-framed with a
//! length prefix and CRC.

use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::model::{Entry, Namespace};
use crate::store::{Store, StoreError};

/// When the WAL forces data to disk.
#[derive(Clone, Copy, Debug)]
pub enum FsyncPolicy {
    /// fsync after every append — durable per write, slowest.
    Always,
    /// fsync at most once per interval — group commit; a crash may lose writes
    /// from the current window in exchange for throughput.
    Interval(Duration),
}

/// A single mutation recorded in the WAL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WalRecord {
    Put(Entry),
    Delete(String),
    Touch { id: String, at: SystemTime },
    /// Marks that a snapshot with this id was taken; records before it are
    /// covered by the snapshot and may be truncated.
    Checkpoint(u64),
}

fn other<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// Append-only, CRC-framed write-ahead log.
///
/// Frame layout: `[u32 len][u32 crc32][payload]`, all little-endian, where
/// `payload` is the MessagePack encoding of a [`WalRecord`] and `crc32` is the
/// CRC of `payload`.
pub struct Wal {
    file: File,
    policy: FsyncPolicy,
    last_fsync: Instant,
}

impl Wal {
    /// Open (creating if absent) the WAL at `path` for appending.
    pub fn open(path: impl AsRef<Path>, policy: FsyncPolicy) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file,
            policy,
            last_fsync: Instant::now(),
        })
    }

    /// Append a record, honoring the fsync policy. The record is fully written
    /// (and, under [`FsyncPolicy::Always`], durable) before this returns.
    pub fn append(&mut self, rec: &WalRecord) -> io::Result<()> {
        let payload = serde_json::to_vec(rec).map_err(other)?;
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&payload)?;
        match self.policy {
            FsyncPolicy::Always => self.sync()?,
            FsyncPolicy::Interval(d) => {
                if self.last_fsync.elapsed() >= d {
                    self.sync()?;
                }
            }
        }
        Ok(())
    }

    /// Force buffered data to disk and reset the group-commit timer.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_data()?;
        self.last_fsync = Instant::now();
        Ok(())
    }

    /// Read every intact record in order, repairing a torn/corrupt tail by
    /// truncating the file at the last valid record boundary. A missing file
    /// yields no records.
    pub fn replay(path: impl AsRef<Path>) -> io::Result<Vec<WalRecord>> {
        let mut file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut records = Vec::new();
        let mut offset = 0usize;
        let mut valid_len = 0usize;
        while offset + 8 <= buf.len() {
            let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
            let start = offset + 8;
            let end = match start.checked_add(len) {
                Some(e) if e <= buf.len() => e,
                _ => break, // torn tail: header promises more than is present
            };
            let payload = &buf[start..end];
            if crc32fast::hash(payload) != crc {
                break; // corrupt frame
            }
            match serde_json::from_slice::<WalRecord>(payload) {
                Ok(rec) => {
                    records.push(rec);
                    offset = end;
                    valid_len = end;
                }
                Err(_) => break,
            }
        }
        if valid_len < buf.len() {
            file.set_len(valid_len as u64)?;
            file.sync_data()?;
        }
        Ok(records)
    }
}

/// A point-in-time capture of a store's key/value set. The vector index is
/// rebuilt from these entries on load.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub namespace: String,
    pub dim: usize,
    pub entries: Vec<Entry>,
}

impl Snapshot {
    /// Capture `store`'s current state.
    pub fn capture(store: &Store) -> Self {
        Self {
            namespace: store.namespace().0.clone(),
            dim: store.dim(),
            entries: store.iter().cloned().collect(),
        }
    }

    /// Write atomically: serialize to `<path>.tmp`, fsync, then rename over
    /// `path` so a crash mid-write never leaves a partial snapshot.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let bytes = serde_json::to_vec(self).map_err(other)?;
        let tmp = path.with_extension("tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_data()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a snapshot if one exists at `path`.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Option<Self>> {
        match File::open(&path) {
            Ok(mut f) => {
                let mut bytes = Vec::new();
                f.read_to_end(&mut bytes)?;
                Ok(Some(serde_json::from_slice(&bytes).map_err(other)?))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Rebuild a [`Store`] from this snapshot (re-indexing all embeddings).
    pub fn into_store(self) -> Store {
        let mut store = Store::new(Namespace::new(self.namespace), self.dim);
        for entry in self.entries {
            // Entries came from a valid store, so their dimensions already match.
            let _ = store.put(entry);
        }
        store
    }
}

/// Apply a single WAL record to a store.
pub fn apply(store: &mut Store, rec: WalRecord) -> Result<(), StoreError> {
    match rec {
        WalRecord::Put(e) => store.put(e),
        WalRecord::Delete(id) => {
            store.remove(&id);
            Ok(())
        }
        WalRecord::Touch { id, at } => {
            store.touch(&id, at);
            Ok(())
        }
        WalRecord::Checkpoint(_) => Ok(()),
    }
}

/// Reconstruct a store: load the snapshot at `snapshot_path` (if any), then
/// replay the WAL tail at `wal_path`. With no snapshot, start empty using the
/// given `namespace` / `dim`.
pub fn recover(
    namespace: Namespace,
    dim: usize,
    snapshot_path: impl AsRef<Path>,
    wal_path: impl AsRef<Path>,
) -> io::Result<Store> {
    let mut store = match Snapshot::load(snapshot_path)? {
        Some(snap) => snap.into_store(),
        None => Store::new(namespace, dim),
    };
    for rec in Wal::replay(wal_path)? {
        let _ = apply(&mut store, rec);
    }
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Key;
    use serde_json::json;

    fn entry(id: &str, emb: Option<Vec<f32>>, v: serde_json::Value) -> Entry {
        Entry {
            key: Key {
                id: id.into(),
                embedding: emb,
            },
            value: v,
            metadata: Default::default(),
        }
    }

    #[test]
    fn wal_append_then_replay_reconstructs() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");
        {
            let mut wal = Wal::open(&wal_path, FsyncPolicy::Always).unwrap();
            wal.append(&WalRecord::Put(entry("a", Some(vec![1.0, 0.0, 0.0]), json!(1))))
                .unwrap();
            wal.append(&WalRecord::Put(entry("b", None, json!({"x": 2}))))
                .unwrap();
            wal.append(&WalRecord::Delete("a".into())).unwrap();
        }
        let mut store = Store::new(Namespace::new("ns"), 3);
        for rec in Wal::replay(&wal_path).unwrap() {
            apply(&mut store, rec).unwrap();
        }
        assert!(store.get("a").is_none(), "delete applied");
        assert_eq!(store.get("b").unwrap().value, json!({"x": 2}));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn torn_tail_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");
        {
            let mut wal = Wal::open(&wal_path, FsyncPolicy::Always).unwrap();
            wal.append(&WalRecord::Put(entry("a", None, json!(1)))).unwrap();
            wal.append(&WalRecord::Put(entry("b", None, json!(2)))).unwrap();
        }
        let good_len = fs::metadata(&wal_path).unwrap().len();
        // Simulate an interrupted final write: a frame header claiming a
        // 100-byte payload, but only 5 bytes actually written.
        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(b"short").unwrap();
        }
        let recs = Wal::replay(&wal_path).unwrap();
        assert_eq!(recs.len(), 2, "only intact records returned");
        assert_eq!(
            fs::metadata(&wal_path).unwrap().len(),
            good_len,
            "torn tail truncated to last valid record"
        );
    }

    #[test]
    fn cold_start_is_snapshot_plus_wal_tail() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("snap.mp");
        let wal_path = dir.path().join("wal.log");

        // Build a store and snapshot it.
        let mut store = Store::new(Namespace::new("ns"), 3);
        store
            .put(entry("a", Some(vec![1.0, 0.0, 0.0]), json!("a")))
            .unwrap();
        store
            .put(entry("b", Some(vec![0.0, 1.0, 0.0]), json!("b")))
            .unwrap();
        Snapshot::capture(&store).write(&snap_path).unwrap();

        // Post-snapshot mutations recorded only in the WAL.
        {
            let mut wal = Wal::open(&wal_path, FsyncPolicy::Always).unwrap();
            wal.append(&WalRecord::Checkpoint(1)).unwrap();
            wal.append(&WalRecord::Delete("a".into())).unwrap();
            wal.append(&WalRecord::Put(entry("c", Some(vec![0.0, 0.0, 1.0]), json!("c"))))
                .unwrap();
        }

        let recovered = recover(Namespace::new("ns"), 3, &snap_path, &wal_path).unwrap();
        assert!(recovered.get("a").is_none(), "deleted post-snapshot");
        assert_eq!(recovered.get("b").unwrap().value, json!("b"), "from snapshot");
        assert_eq!(recovered.get("c").unwrap().value, json!("c"), "from wal tail");
        // The vector index was rebuilt: the WAL-only entry is searchable.
        let hits = recovered.nearest(&vec![0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(hits[0].entry.key.id, "c");
    }
}
