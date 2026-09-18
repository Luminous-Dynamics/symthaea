// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Activation boundary for Symthaea's existing aesthetic scientific-evidence modules.
//!
//! The source modules below already exist under `symthaea-aesthetic`, but are not
//! declared by that crate's current `lib.rs`. This bridge compiles those exact
//! source files without rewriting them, supplies the integrity helpers they were
//! authored against, and exposes a conservative projection into
//! `symthaea-science-research`.
//!
//! Important authority boundary: the historical FNV-1a digests in this domain are
//! deterministic compatibility identifiers, not cryptographic authenticity or
//! qualification capabilities. The adapter computes separate SHA-256 identities
//! for shared-science subjects and artifacts.

#![deny(unsafe_code)]

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum IntegrityError {
    Json(serde_json::Error),
}

impl std::fmt::Display for IntegrityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "JSON integrity encoding failed: {error}"),
        }
    }
}

impl std::error::Error for IntegrityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
        }
    }
}

/// Deterministic 64-bit FNV-1a compatibility digest used by the historical
/// aesthetic evidence formats. This is not a cryptographic digest.
pub fn digest_bytes(bytes: &[u8]) -> String {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

pub(crate) fn digest_json<T: Serialize>(value: &T) -> Result<String, IntegrityError> {
    let bytes = serde_json::to_vec(value).map_err(IntegrityError::Json)?;
    Ok(digest_bytes(&bytes))
}

pub(crate) fn is_stable_digest(value: &str) -> bool {
    value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub(crate) fn load_json<T, E>(path: &Path) -> Result<T, E>
where
    T: DeserializeOwned,
    E: From<std::io::Error> + From<serde_json::Error>,
{
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn save_json_atomic<T, E>(path: &Path, value: &T) -> Result<(), E>
where
    T: Serialize,
    E: From<std::io::Error> + From<serde_json::Error>,
{
    let bytes = serde_json::to_vec_pretty(value)?;
    let temporary = temporary_path(path);
    let result = (|| -> Result<(), E> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn temporary_path(path: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("aesthetic-evidence");
    path.with_file_name(format!(
        ".{name}.tmp-{}-{sequence}",
        std::process::id()
    ))
}

#[path = "../../symthaea-aesthetic/src/causal.rs"]
pub mod causal;
#[path = "../../symthaea-aesthetic/src/metric_lifecycle.rs"]
pub mod metric_lifecycle;
#[path = "../../symthaea-aesthetic/src/semantic_drift.rs"]
pub mod semantic_drift;
#[path = "../../symthaea-aesthetic/src/study_quality.rs"]
pub mod study_quality;
#[path = "../../symthaea-aesthetic/src/scientific_validation.rs"]
pub mod scientific_validation;

pub mod adapter;

pub use causal::*;
pub use metric_lifecycle::*;
pub use scientific_validation::*;
pub use semantic_drift::*;
pub use study_quality::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_digest_is_canonical_lowercase_hex() {
        let digest = digest_bytes(b"symthaea-aesthetic-science");
        assert!(is_stable_digest(&digest));
        assert_eq!(digest.len(), 16);
    }

    #[test]
    fn stable_digest_rejects_noncanonical_text() {
        assert!(!is_stable_digest("ABCDEF0123456789"));
        assert!(!is_stable_digest("abc"));
        assert!(!is_stable_digest("zzzzzzzzzzzzzzzz"));
    }
}
