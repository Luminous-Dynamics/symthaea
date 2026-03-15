//! Persistent dedup cache — survives restarts on solar-powered nodes.
//!
//! Saves seen action hashes and content hashes to files on shutdown,
//! loads them on startup to prevent re-relaying already-seen entries.

use std::collections::HashSet;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

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
}
