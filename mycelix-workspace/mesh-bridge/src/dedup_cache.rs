// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent dedup cache — survives restarts on solar-powered nodes.
//!
//! Saves seen action hashes and content hashes to files on shutdown,
//! loads them on startup to prevent re-relaying already-seen entries.

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

/// LRU cache for binary content hashes.
///
/// Uses a `VecDeque` for insertion-order tracking and a `HashSet` for O(1) lookup.
/// When the cache exceeds capacity, the oldest entry (front of the deque) is evicted.
pub struct LruBinaryCache {
    capacity: usize,
    order: VecDeque<[u8; 32]>,
    set: HashSet<[u8; 32]>,
}

impl LruBinaryCache {
    /// Create a new LRU cache with the given capacity.
    /// A capacity of 0 means no entries will be stored.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity.min(16_384)),
            set: HashSet::with_capacity(capacity.min(16_384)),
        }
    }

    /// Check whether the cache contains the given hash.
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.set.contains(hash)
    }

    /// Insert a hash into the cache.
    /// Returns `true` if the hash was new (not already present).
    /// When inserting past capacity, the oldest entry is evicted.
    pub fn insert(&mut self, hash: [u8; 32]) -> bool {
        if self.capacity == 0 {
            return false;
        }
        if self.set.contains(&hash) {
            return false;
        }
        // Evict oldest if at capacity
        while self.order.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.set.remove(&oldest);
            }
        }
        self.set.insert(hash);
        self.order.push_back(hash);
        true
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Borrow the underlying `HashSet` for persistence compatibility.
    pub fn as_hash_set(&self) -> &HashSet<[u8; 32]> {
        &self.set
    }

    /// Construct an LRU cache from a pre-existing `HashSet`.
    ///
    /// If the set exceeds `capacity`, arbitrary entries are dropped to fit.
    /// Insertion order within the loaded set is arbitrary (HashSet iteration order).
    pub fn from_hash_set(set: HashSet<[u8; 32]>, capacity: usize) -> Self {
        let mut order = VecDeque::with_capacity(capacity.min(set.len()));
        let mut retained = HashSet::with_capacity(capacity.min(set.len()));
        for hash in &set {
            if retained.len() >= capacity {
                break;
            }
            retained.insert(*hash);
            order.push_back(*hash);
        }
        Self {
            capacity,
            order,
            set: retained,
        }
    }
}

/// Default dedup cache directory.
const DEFAULT_CACHE_DIR: &str = "/tmp/mycelix-mesh-bridge";

/// Get the cache directory from env or default.
pub fn cache_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("MESH_CACHE_DIR").unwrap_or_else(|_| DEFAULT_CACHE_DIR.to_string()),
    )
}

/// Save string-keyed dedup set (poller action hashes) to disk.
pub fn save_string_cache(hashes: &HashSet<String>, filename: &str) -> std::io::Result<()> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    let file = std::fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for hash in hashes {
        writeln!(writer, "{hash}")?;
    }
    writer.flush()?;
    tracing::info!("Saved {} dedup entries to {}", hashes.len(), path.display());
    Ok(())
}

/// Load string-keyed dedup set from disk.
pub fn load_string_cache(filename: &str) -> HashSet<String> {
    let path = cache_dir().join(filename);
    load_string_cache_from_path(&path)
}

fn load_string_cache_from_path(path: &Path) -> HashSet<String> {
    match std::fs::File::open(path) {
        Ok(file) => {
            let reader = std::io::BufReader::new(file);
            let mut set = HashSet::new();
            for line in reader.lines() {
                if let Ok(hash) = line {
                    let trimmed = hash.trim().to_string();
                    if !trimmed.is_empty() {
                        set.insert(trimmed);
                    }
                }
            }
            tracing::info!("Loaded {} dedup entries from {}", set.len(), path.display());
            set
        }
        Err(_) => {
            tracing::debug!("No dedup cache at {}, starting fresh", path.display());
            HashSet::new()
        }
    }
}

/// Save binary-keyed dedup set (relay content hashes) to disk.
///
/// Accepts a raw `HashSet` for backward compatibility. Prefer `save_lru_cache`
/// when working with `LruBinaryCache`.
pub fn save_binary_cache(hashes: &HashSet<[u8; 32]>, filename: &str) -> std::io::Result<()> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    let file = std::fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for hash in hashes {
        writer.write_all(hash)?;
    }
    writer.flush()?;
    tracing::info!(
        "Saved {} binary dedup entries to {}",
        hashes.len(),
        path.display()
    );
    Ok(())
}

/// Save an `LruBinaryCache` to disk (delegates to `save_binary_cache`).
pub fn save_lru_cache(cache: &LruBinaryCache, filename: &str) -> std::io::Result<()> {
    save_binary_cache(cache.as_hash_set(), filename)
}

/// Load binary-keyed dedup set from disk.
pub fn load_binary_cache(filename: &str) -> HashSet<[u8; 32]> {
    let path = cache_dir().join(filename);
    load_binary_cache_from_path(&path)
}

fn load_binary_cache_from_path(path: &Path) -> HashSet<[u8; 32]> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut set = HashSet::new();
            for chunk in bytes.chunks_exact(32) {
                let mut hash = [0u8; 32];
                hash.copy_from_slice(chunk);
                set.insert(hash);
            }
            tracing::info!(
                "Loaded {} binary dedup entries from {}",
                set.len(),
                path.display()
            );
            set
        }
        Err(_) => {
            tracing::debug!("No binary dedup cache at {}, starting fresh", path.display());
            HashSet::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use direct file paths to avoid env var races in parallel test execution.

    #[test]
    fn test_string_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let mut set = HashSet::new();
        set.insert("hash-001".to_string());
        set.insert("hash-002".to_string());
        set.insert("hash-003".to_string());

        let path = dir.path().join("test-string.cache");
        // Write directly
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for hash in &set {
            writeln!(writer, "{hash}").unwrap();
        }
        writer.flush().unwrap();

        let loaded = load_string_cache_from_path(&path);

        assert_eq!(loaded.len(), 3);
        assert!(loaded.contains("hash-001"));
        assert!(loaded.contains("hash-002"));
        assert!(loaded.contains("hash-003"));
    }

    #[test]
    fn test_binary_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let mut set = HashSet::new();
        set.insert(<[u8; 32]>::from(blake3::hash(b"payload-1")));
        set.insert(<[u8; 32]>::from(blake3::hash(b"payload-2")));

        let path = dir.path().join("test-binary.cache");
        // Write directly
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for hash in &set {
            writer.write_all(hash).unwrap();
        }
        writer.flush().unwrap();

        let loaded = load_binary_cache_from_path(&path);

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(&<[u8; 32]>::from(blake3::hash(b"payload-1"))));
        assert!(loaded.contains(&<[u8; 32]>::from(blake3::hash(b"payload-2"))));
    }

    #[test]
    fn test_load_nonexistent_cache() {
        let path = Path::new("/tmp/nonexistent-mesh-test-dir-12345/does-not-exist.cache");
        let set = load_string_cache_from_path(path);
        assert!(set.is_empty());

        let bset = load_binary_cache_from_path(path);
        assert!(bset.is_empty());
    }

    #[test]
    fn test_empty_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-empty.cache");
        std::fs::write(&path, "").unwrap();
        let loaded = load_string_cache_from_path(&path);
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_string_cache_ignores_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blanks.cache");
        std::fs::write(&path, "hash-1\n\n  \nhash-2\n").unwrap();
        let loaded = load_string_cache_from_path(&path);
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruBinaryCache::new(3);
        let h1: [u8; 32] = blake3::hash(b"a").into();
        let h2: [u8; 32] = blake3::hash(b"b").into();
        let h3: [u8; 32] = blake3::hash(b"c").into();
        let h4: [u8; 32] = blake3::hash(b"d").into();

        assert!(cache.insert(h1));
        assert!(cache.insert(h2));
        assert!(cache.insert(h3));
        assert_eq!(cache.len(), 3);

        // Inserting a 4th should evict the oldest (h1)
        assert!(cache.insert(h4));
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&h1), "h1 should have been evicted");
        assert!(cache.contains(&h2));
        assert!(cache.contains(&h3));
        assert!(cache.contains(&h4));
    }

    #[test]
    fn test_lru_contains() {
        let mut cache = LruBinaryCache::new(10);
        let h1: [u8; 32] = blake3::hash(b"x").into();
        let h2: [u8; 32] = blake3::hash(b"y").into();

        assert!(!cache.contains(&h1));
        assert!(cache.insert(h1));
        assert!(cache.contains(&h1));
        assert!(!cache.contains(&h2));

        // Duplicate insert returns false
        assert!(!cache.insert(h1));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_from_hash_set() {
        let mut set = HashSet::new();
        set.insert(<[u8; 32]>::from(blake3::hash(b"p1")));
        set.insert(<[u8; 32]>::from(blake3::hash(b"p2")));
        set.insert(<[u8; 32]>::from(blake3::hash(b"p3")));

        let cache = LruBinaryCache::from_hash_set(set.clone(), 100);
        assert_eq!(cache.len(), 3);
        for h in &set {
            assert!(cache.contains(h));
        }

        // from_hash_set with smaller capacity truncates
        let small = LruBinaryCache::from_hash_set(set, 2);
        assert_eq!(small.len(), 2);
    }

    #[test]
    fn test_lru_capacity_zero_safe() {
        let mut cache = LruBinaryCache::new(0);
        let h: [u8; 32] = blake3::hash(b"anything").into();

        // Insert returns false, nothing stored
        assert!(!cache.insert(h));
        assert!(!cache.contains(&h));
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
