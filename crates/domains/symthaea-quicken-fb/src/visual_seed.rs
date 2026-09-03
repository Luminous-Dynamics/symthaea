// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded presentation-only visual seed handling for `quicken-fb`.
//!
//! Visual seed material exists only to make procedural artwork stable across
//! boots. It is not a credential, recovery phrase, key-derivation input,
//! authentication secret, or authority-bearing machine identity. Prefer a
//! dedicated persistent file rather than argv so operators are not encouraged
//! to reuse meaningful secret material for presentation.

#![forbid(unsafe_code)]

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

/// Hard input bound before seed material reaches hashing/RNG initialization.
pub const MAX_VISUAL_SEED_BYTES: usize = 4_096;

#[derive(Debug)]
pub enum VisualSeedError {
    Io(io::Error),
    Empty,
    TooLarge { bytes: usize, max: usize },
    InvalidUtf8,
}

impl From<io::Error> for VisualSeedError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl fmt::Display for VisualSeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to read visual seed: {error}"),
            Self::Empty => write!(f, "visual seed is empty"),
            Self::TooLarge { bytes, max } => {
                write!(f, "visual seed is too large: {bytes} bytes (max {max})")
            }
            Self::InvalidUtf8 => write!(f, "visual seed file is not valid UTF-8"),
        }
    }
}

impl std::error::Error for VisualSeedError {}

/// Normalize presentation seed text while preserving all non-edge content.
///
/// The returned value is deterministic input material only. Callers must never
/// reinterpret it as authentication or recovery state.
pub fn normalize_visual_seed(input: &str) -> Result<String, VisualSeedError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(VisualSeedError::Empty);
    }
    if trimmed.len() > MAX_VISUAL_SEED_BYTES {
        return Err(VisualSeedError::TooLarge {
            bytes: trimmed.len(),
            max: MAX_VISUAL_SEED_BYTES,
        });
    }
    Ok(trimmed.to_owned())
}

/// Load a bounded UTF-8 seed from a dedicated presentation-only file.
///
/// Metadata is checked before allocation and the byte length is checked again
/// after reading so a concurrently replaced file cannot bypass the bound.
pub fn load_visual_seed_file(path: &Path) -> Result<String, VisualSeedError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_VISUAL_SEED_BYTES as u64 {
        return Err(VisualSeedError::TooLarge {
            bytes: usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            max: MAX_VISUAL_SEED_BYTES,
        });
    }

    let bytes = fs::read(path)?;
    if bytes.len() > MAX_VISUAL_SEED_BYTES {
        return Err(VisualSeedError::TooLarge {
            bytes: bytes.len(),
            max: MAX_VISUAL_SEED_BYTES,
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| VisualSeedError::InvalidUtf8)?;
    normalize_visual_seed(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_seed_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-quicken-visual-seed-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn seed_normalization_rejects_empty_and_oversized_input() {
        assert!(matches!(normalize_visual_seed("  \n"), Err(VisualSeedError::Empty)));
        let oversized = "x".repeat(MAX_VISUAL_SEED_BYTES + 1);
        assert!(matches!(
            normalize_visual_seed(&oversized),
            Err(VisualSeedError::TooLarge { .. })
        ));
    }

    #[test]
    fn seed_normalization_trims_only_edge_whitespace() {
        assert_eq!(
            normalize_visual_seed("  public visual seed 42\n").unwrap(),
            "public visual seed 42"
        );
    }

    #[test]
    fn file_loader_is_bounded_and_deterministic() {
        let path = temp_seed_path();
        fs::write(&path, b"stable-public-visual-seed\n").unwrap();
        assert_eq!(
            load_visual_seed_file(&path).unwrap(),
            "stable-public-visual-seed"
        );

        fs::write(&path, vec![b'x'; MAX_VISUAL_SEED_BYTES + 1]).unwrap();
        assert!(matches!(
            load_visual_seed_file(&path),
            Err(VisualSeedError::TooLarge { .. })
        ));
        let _ = fs::remove_file(path);
    }
}
