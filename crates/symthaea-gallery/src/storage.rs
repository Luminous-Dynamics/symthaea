// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Filesystem persistence for the gallery.
//!
//! SVG and WAV files stored on disk, poetry stored inline in the JSON index.
//! No database dependency — flat file storage sufficient for single-instance system.

use crate::GalleryIndex;
use std::path::{Path, PathBuf};

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
        let path = self.visual_dir().join(filename);
        std::fs::write(&path, svg_content)?;
        Ok(path)
    }

    /// Save a music WAV file to disk (raw PCM i16 mono at given sample rate).
    pub fn save_music(
        &self,
        filename: &str,
        samples: &[i16],
        sample_rate: u32,
    ) -> std::io::Result<PathBuf> {
        let path = self.music_dir().join(filename);
        // Write minimal WAV header + data
        let data = encode_wav(samples, sample_rate);
        std::fs::write(&path, data)?;
        Ok(path)
    }

    /// Save the gallery index to JSON.
    pub fn save_index(&self, index: &GalleryIndex) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(index)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(self.index_path(), json)
    }

    /// Load the gallery index from JSON.
    pub fn load_index(&self) -> std::io::Result<GalleryIndex> {
        let json = std::fs::read_to_string(self.index_path())?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Encode PCM i16 mono samples as a minimal WAV byte buffer.
fn encode_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let file_len = 36 + data_len;

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
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        buf.extend_from_slice(&sample.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_valid() {
        let samples = vec![0i16, 1000, -1000, 500, -500];
        let wav = encode_wav(&samples, 44100);

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
}
