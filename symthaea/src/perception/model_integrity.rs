// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model Integrity Verification
//!
//! SHA-256 integrity checks for ML model weight files before unsafe
//! memory-mapped loading. Prevents loading corrupted or tampered
//! safetensors files.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::model_integrity::verify_model_integrity;
//!
//! // With a known hash (strict mode):
//! verify_model_integrity(&path, Some("abcdef1234..."))?;
//!
//! // Without a known hash (logs warning, allows loading):
//! verify_model_integrity(&path, None)?;
//!
//! // Compute hash for manifest generation:
//! let hash = compute_model_hash(&path)?;
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "neural-bridge")]
use anyhow::Context;
#[cfg(feature = "neural-bridge")]
use sha2::{Digest, Sha256};
#[cfg(feature = "neural-bridge")]
use std::io::Read;

/// A manifest mapping model file paths to their expected SHA-256 hashes.
///
/// Used for batch verification of model directories. Each entry maps
/// an absolute or relative model path to its hex-encoded SHA-256 digest.
#[derive(Debug, Clone, Default)]
pub struct ModelManifest {
    /// Map from model file path to expected SHA-256 hex digest.
    pub hashes: HashMap<PathBuf, String>,
}

impl ModelManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            hashes: HashMap::new(),
        }
    }

    /// Insert a path -> hash entry.
    pub fn insert(&mut self, path: impl Into<PathBuf>, hash: impl Into<String>) {
        self.hashes.insert(path.into(), hash.into());
    }

    /// Look up the expected hash for a given path.
    pub fn expected_hash(&self, path: &Path) -> Option<&str> {
        self.hashes.get(path).map(|s| s.as_str())
    }

    /// Generate a manifest by scanning a directory for safetensors files
    /// and computing their SHA-256 hashes.
    #[cfg(feature = "neural-bridge")]
    pub fn from_directory(dir: &Path) -> Result<Self> {
        let mut manifest = Self::new();
        if !dir.is_dir() {
            anyhow::bail!("model_integrity: not a directory: {}", dir.display());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "safetensors") {
                let hash = compute_model_hash(&path)?;
                manifest.insert(path, hash);
            }
        }
        Ok(manifest)
    }

    /// Serialize manifest to JSON for storage.
    pub fn to_json(&self) -> Result<String> {
        let map: std::collections::BTreeMap<String, &str> = self
            .hashes
            .iter()
            .map(|(p, h)| (p.display().to_string(), h.as_str()))
            .collect();
        serde_json::to_string_pretty(&map)
            .map_err(|e| anyhow::anyhow!("model_integrity: JSON serialization failed: {e}"))
    }

    /// Deserialize manifest from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        let map: std::collections::BTreeMap<String, String> = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("model_integrity: JSON parse failed: {e}"))?;
        let mut manifest = Self::new();
        for (path, hash) in map {
            manifest.insert(PathBuf::from(path), hash);
        }
        Ok(manifest)
    }
}

/// Compute the SHA-256 hash of a file and return it as a lowercase hex string.
///
/// Reads the file in 64 KiB chunks for memory efficiency with large model files.
#[cfg(feature = "neural-bridge")]
pub fn compute_model_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("model_integrity: failed to open {}", path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536]; // 64 KiB chunks

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .with_context(|| format!("model_integrity: failed to read {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

/// Verify the integrity of a model file before unsafe memory-mapped loading.
///
/// # Behavior
///
/// - If `expected_hash` is `Some(hash)`: computes SHA-256 and compares. Returns
///   an error on mismatch.
/// - If `expected_hash` is `None`: logs a warning about unverified loading but
///   succeeds (graceful degradation for models without known hashes).
/// - If the file does not exist: returns an error regardless.
///
/// # Errors
///
/// - File not found or unreadable.
/// - SHA-256 mismatch when an expected hash is provided.
#[cfg(feature = "neural-bridge")]
pub fn verify_model_integrity(path: &Path, expected_hash: Option<&str>) -> Result<()> {
    // Always verify the file exists and is readable before any unsafe loading
    if !path.exists() {
        anyhow::bail!(
            "model_integrity: model file does not exist: {}",
            path.display()
        );
    }

    match expected_hash {
        Some(expected) => {
            let actual = compute_model_hash(path)?;
            if actual != expected {
                anyhow::bail!(
                    "model_integrity: SHA-256 mismatch for {}\n  expected: {}\n  actual:   {}",
                    path.display(),
                    expected,
                    actual
                );
            }
            tracing::debug!(
                path = %path.display(),
                "model_integrity: SHA-256 verified"
            );
            Ok(())
        }
        None => {
            tracing::warn!(
                path = %path.display(),
                "model_integrity: loading model without integrity verification \
                 (no expected hash provided)"
            );
            Ok(())
        }
    }
}

/// Verify integrity and load safetensors via memory-mapped I/O.
///
/// Consolidates the `verify → unsafe mmap` pattern into a single safe
/// wrapper. The `unsafe` is justified because `verify_model_integrity`
/// has already confirmed file existence and (optionally) SHA-256 hash.
#[cfg(feature = "neural-bridge")]
pub fn verified_mmap_safetensors(
    paths: &[impl AsRef<std::path::Path>],
    dtype: candle_core::DType,
    device: &candle_core::Device,
    expected_hash: Option<&str>,
) -> Result<candle_nn::VarBuilder<'static>> {
    for p in paths {
        verify_model_integrity(p.as_ref(), expected_hash)?;
    }
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    // SAFETY: Files verified above (existence + optional SHA-256 integrity).
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&path_refs, dtype, device)? })
}

/// Stub for non-neural-bridge builds. Always succeeds with a no-op.
#[cfg(not(feature = "neural-bridge"))]
pub fn verify_model_integrity(_path: &Path, _expected_hash: Option<&str>) -> Result<()> {
    Ok(())
}

/// Stub for non-neural-bridge builds.
#[cfg(not(feature = "neural-bridge"))]
pub fn compute_model_hash(_path: &Path) -> Result<String> {
    anyhow::bail!("model_integrity: SHA-256 hashing requires the 'neural-bridge' feature")
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "neural-bridge")]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write content to a temp file and return the path.
    fn write_temp_file(name: &str, content: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join("symthaea_model_integrity_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }

    #[test]
    fn test_compute_hash_consistency() {
        let path = write_temp_file("consistency.bin", b"deterministic content for hashing");
        let hash1 = compute_model_hash(&path).unwrap();
        let hash2 = compute_model_hash(&path).unwrap();
        assert_eq!(hash1, hash2, "Same file must produce same hash");
        assert_eq!(hash1.len(), 64, "SHA-256 hex digest is 64 characters");
    }

    #[test]
    fn test_verify_integrity_correct_hash() {
        let content = b"model weights placeholder data";
        let path = write_temp_file("correct_hash.bin", content);
        let hash = compute_model_hash(&path).unwrap();
        // Verification with the correct hash should succeed
        verify_model_integrity(&path, Some(&hash)).unwrap();
    }

    #[test]
    fn test_verify_integrity_wrong_hash() {
        let content = b"model weights placeholder data";
        let path = write_temp_file("wrong_hash.bin", content);
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = verify_model_integrity(&path, Some(wrong_hash));
        assert!(result.is_err(), "Wrong hash must fail verification");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("SHA-256 mismatch"),
            "Error should mention SHA-256 mismatch, got: {err_msg}"
        );
    }

    #[test]
    fn test_verify_integrity_no_expected() {
        let content = b"unverified model data";
        let path = write_temp_file("no_expected.bin", content);
        // None hash should succeed (graceful degradation)
        verify_model_integrity(&path, None).unwrap();
    }

    #[test]
    fn test_verify_nonexistent_file() {
        let path = Path::new("/tmp/symthaea_model_integrity_tests/nonexistent_model_xyz.bin");
        let result = verify_model_integrity(path, None);
        assert!(result.is_err(), "Nonexistent file must fail");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("does not exist"),
            "Error should mention file not existing, got: {err_msg}"
        );
    }
}
