// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Filesystem persistence for the gallery.
//!
//! SVG and WAV files stored on disk, poetry stored inline in the JSON index.
//! No database dependency — flat file storage sufficient for single-instance system.

use crate::GalleryIndex;
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const MAX_ARTIFACT_FILENAME_BYTES: usize = 255;

/// Validate an artifact filename before joining it to a gallery directory.
///
/// Gallery metadata stores basenames, not paths. Keeping that distinction at
/// the write boundary prevents a caller from escaping the configured root.
pub(crate) fn validate_artifact_filename(
    filename: &str,
    expected_extension: &str,
) -> std::io::Result<()> {
    let path = Path::new(filename);
    let is_single_component = path.file_name() == Some(OsStr::new(filename));
    let has_expected_extension = path.extension() == Some(OsStr::new(expected_extension));
    if filename.is_empty()
        || filename.len() > MAX_ARTIFACT_FILENAME_BYTES
        || filename.contains('\0')
        || filename.contains('/')
        || filename.contains('\\')
        || !is_single_component
        || !has_expected_extension
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid gallery artifact filename: {filename:?}"),
        ));
    }
    Ok(())
}

fn atomic_write(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no parent")
    })?;
    let filename = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no filename")
    })?;
    let temporary = parent.join(format!(".{filename}.{}.tmp", Uuid::new_v4()));

    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// Gallery storage manager.
pub struct GalleryStorage {
    /// Root directory for gallery files.
    root: PathBuf,
}

impl GalleryStorage {
    /// Create a new storage manager rooted at the given directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to the gallery index JSON file.
    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.json")
    }

    /// Directory for visual artwork SVG files.
    pub fn visual_dir(&self) -> PathBuf {
        self.root.join("visual")
    }

    /// Directory for music WAV files.
    pub fn music_dir(&self) -> PathBuf {
        self.root.join("music")
    }

    /// Ensure all gallery directories exist.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.visual_dir())?;
        std::fs::create_dir_all(self.music_dir())?;
        Ok(())
    }

    /// Save a visual artwork SVG to disk.
    pub fn save_visual(&self, filename: &str, svg_content: &str) -> std::io::Result<PathBuf> {
        validate_artifact_filename(filename, "svg")?;
        let path = self.visual_dir().join(filename);
        atomic_write(&path, svg_content.as_bytes())?;
        Ok(path)
    }

    /// Save a music WAV file to disk (raw PCM i16 mono at given sample rate).
    pub fn save_music(
        &self,
        filename: &str,
        samples: &[i16],
        sample_rate: u32,
    ) -> std::io::Result<PathBuf> {
        validate_artifact_filename(filename, "wav")?;
        let path = self.music_dir().join(filename);
        // Write minimal WAV header + data
        let data = encode_wav(samples, sample_rate)?;
        atomic_write(&path, &data)?;
        Ok(path)
    }

    /// Save the gallery index to JSON.
    pub fn save_index(&self, index: &GalleryIndex) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(index)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        atomic_write(&self.index_path(), json.as_bytes())
    }

    /// Load the gallery index from JSON.
    pub fn load_index(&self) -> std::io::Result<GalleryIndex> {
        let json = std::fs::read_to_string(self.index_path())?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Encode PCM i16 mono samples as a minimal WAV byte buffer.
fn encode_wav(samples: &[i16], sample_rate: u32) -> std::io::Result<Vec<u8>> {
    let byte_rate = sample_rate
        .checked_mul(2)
        .filter(|_| sample_rate > 0)
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid WAV sample rate")
        })?;
    let data_len = samples
        .len()
        .checked_mul(2)
        .and_then(|len| u32::try_from(len).ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAV data exceeds RIFF limit",
            )
        })?;
    let file_len = 36u32.checked_add(data_len).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAV file exceeds RIFF limit",
        )
    })?;

    let mut buf = Vec::with_capacity(44 + samples.len() * 2);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_len.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        buf.extend_from_slice(&sample.to_le_bytes());
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_valid() {
        let samples = vec![0i16, 1000, -1000, 500, -500];
        let wav = encode_wav(&samples, 44100).unwrap();

        // RIFF header
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        // fmt chunk
        assert_eq!(&wav[12..16], b"fmt ");

        // data chunk
        assert_eq!(&wav[36..40], b"data");

        // Total size: 44 header + 5 samples * 2 bytes = 54
        assert_eq!(wav.len(), 54);
    }

    #[test]
    fn storage_paths() {
        let storage = GalleryStorage::new("/tmp/test-gallery");
        assert_eq!(
            storage.index_path(),
            PathBuf::from("/tmp/test-gallery/index.json")
        );
        assert_eq!(
            storage.visual_dir(),
            PathBuf::from("/tmp/test-gallery/visual")
        );
        assert_eq!(
            storage.music_dir(),
            PathBuf::from("/tmp/test-gallery/music")
        );
    }

    #[test]
    fn round_trip_index() {
        let dir =
            std::env::temp_dir().join(format!("symthaea-gallery-test-{}", std::process::id()));
        let storage = GalleryStorage::new(&dir);
        storage.ensure_dirs().unwrap();

        let mut index = GalleryIndex::new(100);
        let mut score = symthaea_aesthetic::AestheticScore::uniform(0.7);
        score.compute_composite();
        index.add(crate::create_entry(
            crate::ArtModality::Poetry {
                text: "test poem".into(),
            },
            score,
            [0.5; 8],
            42,
        ));

        storage.save_index(&index).unwrap();
        let loaded = storage.load_index().unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries[0].created_at_cycle, 42);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_visual_file() {
        let dir =
            std::env::temp_dir().join(format!("symthaea-gallery-visual-{}", std::process::id()));
        let storage = GalleryStorage::new(&dir);
        storage.ensure_dirs().unwrap();

        let path = storage.save_visual("test.svg", "<svg></svg>").unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "<svg></svg>");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn artifact_writes_reject_paths_and_wrong_extensions() {
        let dir = std::env::temp_dir().join(format!("symthaea-gallery-paths-{}", Uuid::new_v4()));
        let storage = GalleryStorage::new(&dir);
        storage.ensure_dirs().unwrap();

        assert_eq!(
            storage
                .save_visual("../escape.svg", "<svg/>")
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            storage
                .save_music("nested/audio.wav", &[], 44_100)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            storage
                .save_visual("art.html", "<svg/>")
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(!dir.join("escape.svg").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wav_rejects_invalid_sample_rate() {
        assert_eq!(
            encode_wav(&[], 0).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            encode_wav(&[], u32::MAX).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
