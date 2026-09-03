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
use std::fs::OpenOptions;
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Hard input bound before seed material reaches hashing/RNG initialization.
pub const MAX_VISUAL_SEED_BYTES: usize = 4_096;

#[derive(Debug)]
pub enum VisualSeedError {
    Io(io::Error),
    Empty,
    TooLarge { bytes: usize, max: usize },
    InvalidUtf8,
    NotRegularFile,
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
            Self::NotRegularFile => write!(f, "visual seed path is not a regular file"),
        }
    }
}

impl std::error::Error for VisualSeedError {}

/// Validate presentation seed text without rewriting literal bytes.
///
/// Preserving the exact string is intentional: the deprecated compatibility
/// input must render identically even when a historical seed contains leading
/// or trailing spaces. The returned value remains presentation input only and
/// must never be reinterpreted as authentication or recovery state.
pub fn normalize_visual_seed(input: &str) -> Result<String, VisualSeedError> {
    if input.is_empty() {
        return Err(VisualSeedError::Empty);
    }
    if input.len() > MAX_VISUAL_SEED_BYTES {
        return Err(VisualSeedError::TooLarge {
            bytes: input.len(),
            max: MAX_VISUAL_SEED_BYTES,
        });
    }
    Ok(input.to_owned())
}

/// Load a bounded UTF-8 seed from a dedicated presentation-only regular file.
///
/// `O_NOFOLLOW` prevents a configured path from becoming a root-readable
/// symlink oracle. Reading is capped at `MAX + 1` bytes from the opened file
/// descriptor, eliminating the metadata/read replacement race and bounding
/// allocation even if the file changes while it is being read. One conventional
/// terminal LF (or CRLF) is excluded from seed material so text seed files are
/// stable across ordinary POSIX file writers; all other bytes are preserved.
pub fn load_visual_seed_file(path: &Path) -> Result<String, VisualSeedError> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err(VisualSeedError::NotRegularFile);
    }

    let mut bytes = Vec::with_capacity(MAX_VISUAL_SEED_BYTES + 1);
    file.take((MAX_VISUAL_SEED_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_VISUAL_SEED_BYTES {
        return Err(VisualSeedError::TooLarge {
            bytes: bytes.len(),
            max: MAX_VISUAL_SEED_BYTES,
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| VisualSeedError::InvalidUtf8)?;
    let text = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    normalize_visual_seed(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;
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
    fn seed_validation_rejects_empty_and_oversized_input() {
        assert!(matches!(normalize_visual_seed(""), Err(VisualSeedError::Empty)));
        let oversized = "x".repeat(MAX_VISUAL_SEED_BYTES + 1);
        assert!(matches!(
            normalize_visual_seed(&oversized),
            Err(VisualSeedError::TooLarge { .. })
        ));
    }

    #[test]
    fn literal_seed_validation_preserves_legacy_bytes() {
        assert_eq!(
            normalize_visual_seed("  public visual seed 42  ").unwrap(),
            "  public visual seed 42  "
        );
    }

    #[test]
    fn file_loader_is_bounded_and_strips_only_terminal_newline() {
        let path = temp_seed_path();
        fs::write(&path, b" stable-public-visual-seed \n").unwrap();
        assert_eq!(
            load_visual_seed_file(&path).unwrap(),
            " stable-public-visual-seed "
        );

        fs::write(&path, vec![b'x'; MAX_VISUAL_SEED_BYTES + 1]).unwrap();
        assert!(matches!(
            load_visual_seed_file(&path),
            Err(VisualSeedError::TooLarge { .. })
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn file_loader_refuses_symlink_seed_paths() {
        let target = temp_seed_path();
        let link = target.with_extension("link");
        fs::write(&target, b"public-seed\n").unwrap();
        symlink(&target, &link).unwrap();

        assert!(load_visual_seed_file(&link).is_err());

        let _ = fs::remove_file(link);
        let _ = fs::remove_file(target);
    }
}
