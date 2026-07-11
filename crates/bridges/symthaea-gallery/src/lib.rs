// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-gallery
//!
//! Self-curating art gallery for Symthaea. Persists generated artworks,
//! tracks aesthetic evolution over time, and auto-curates by pruning
//! low-scoring works while preserving surprising experiments.
//!
//! # Architecture
//!
//! ```text
//! Artwork → GalleryEntry → Gallery (in-memory index) → filesystem persistence
//!                                                     → evolution tracking
//!                                                     → self-curation
//! ```

#![deny(unsafe_code)]

pub mod curation;
pub mod evolution;
#[cfg(feature = "server")]
pub mod server;
pub mod storage;
pub mod style;

use serde::{Deserialize, Serialize};
use symthaea_aesthetic::AestheticScore;
use uuid::Uuid;

// ─── Core Types ──────────────────────────────────────────────────────────────

/// Art modality — what kind of artifact is stored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtModality {
    /// SVG visual artwork (stored as file).
    Visual { filename: String },
    /// PCM audio (stored as WAV file).
    Music {
        filename: String,
        duration_secs: u32,
    },
    /// Poetry/creative text (stored inline).
    Poetry { text: String },
    /// Synesthetic: simultaneous visual + audio, cross-linked via Scriabin mapping.
    Synesthetic {
        visual_filename: String,
        audio_filename: String,
        /// Which synesthetic mapping profile was used (e.g. "scriabin").
        mapping_profile: String,
    },
    /// Dance choreography (stored as keyframe JSON).
    Dance {
        filename: String,
        duration_secs: u32,
    },
    /// Musical score notation (MusicXML + optional SVG).
    Score {
        musicxml_filename: String,
        score_svg: Option<String>,
    },
}

impl ArtModality {
    pub fn name(&self) -> &str {
        match self {
            ArtModality::Visual { .. } => "visual",
            ArtModality::Music { .. } => "music",
            ArtModality::Poetry { .. } => "poetry",
            ArtModality::Synesthetic { .. } => "synesthetic",
            ArtModality::Dance { .. } => "dance",
            ArtModality::Score { .. } => "score",
        }
    }
}

/// A single entry in the gallery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryEntry {
    /// Unique identifier.
    pub id: Uuid,
    /// When this artwork was created (cycle count).
    pub created_at_cycle: u64,
    /// Wall-clock creation time.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// What kind of artwork.
    pub modality: ArtModality,
    /// Aesthetic quality score at time of creation.
    pub aesthetic_score: AestheticScore,
    /// Eight Harmony activations at time of creation.
    pub harmony_activations: [f32; 8],
    /// Auto-generated tags from dominant harmonies.
    pub tags: Vec<String>,
    /// Whether this entry is protected from curation pruning.
    pub protected: bool,
}

/// The gallery index — in-memory representation of all stored artworks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryIndex {
    /// All entries, ordered by creation time.
    pub entries: Vec<GalleryEntry>,
    /// Maximum number of entries before curation triggers.
    pub max_entries: usize,
}

impl GalleryIndex {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Add an entry to the gallery.
    pub fn add(&mut self, entry: GalleryEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the gallery is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the gallery is at capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_entries
    }

    /// Get entries sorted by aesthetic score (highest first).
    pub fn by_score(&self) -> Vec<&GalleryEntry> {
        let mut sorted: Vec<&GalleryEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.aesthetic_score
                .composite
                .partial_cmp(&a.aesthetic_score.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get entries filtered by modality.
    pub fn by_modality(&self, modality_name: &str) -> Vec<&GalleryEntry> {
        self.entries
            .iter()
            .filter(|e| e.modality.name() == modality_name)
            .collect()
    }

    /// Get the N best entries by aesthetic score.
    pub fn top_n(&self, n: usize) -> Vec<&GalleryEntry> {
        self.by_score().into_iter().take(n).collect()
    }

    /// Average aesthetic score across all entries.
    pub fn average_score(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .entries
            .iter()
            .map(|e| e.aesthetic_score.composite)
            .sum();
        sum / self.entries.len() as f32
    }
}

impl Default for GalleryIndex {
    fn default() -> Self {
        Self::new(10_000)
    }
}

/// Harmony names for tag generation.
const HARMONY_NAMES: [&str; 8] = [
    "resonant-coherence",
    "pan-sentient-flourishing",
    "integral-wisdom",
    "infinite-play",
    "universal-interconnectedness",
    "sacred-reciprocity",
    "evolutionary-progression",
    "sacred-stillness",
];

/// Create a gallery entry from artwork metadata.
pub fn create_entry(
    modality: ArtModality,
    aesthetic_score: AestheticScore,
    harmony_activations: [f32; 8],
    cycle_count: u64,
) -> GalleryEntry {
    // Auto-generate tags from dominant harmonies (top 3 above 0.4)
    let mut harmony_pairs: Vec<(usize, f32)> = harmony_activations
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    harmony_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let tags: Vec<String> = harmony_pairs
        .iter()
        .take(3)
        .filter(|(_, v)| *v > 0.4)
        .map(|(i, _)| HARMONY_NAMES[*i].to_string())
        .collect();

    // Protect high-surprise entries
    let protected = aesthetic_score.surprise > 0.8;

    GalleryEntry {
        id: Uuid::new_v4(),
        created_at_cycle: cycle_count,
        created_at: chrono::Utc::now(),
        modality,
        aesthetic_score,
        harmony_activations,
        tags,
        protected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_score() -> AestheticScore {
        let mut s = AestheticScore::uniform(0.6);
        s.compute_composite();
        s
    }

    #[test]
    fn create_entry_generates_tags() {
        let entry = create_entry(
            ArtModality::Visual {
                filename: "art_001.svg".into(),
            },
            test_score(),
            [0.9, 0.1, 0.2, 0.8, 0.1, 0.5, 0.7, 0.0],
            100,
        );
        assert!(!entry.tags.is_empty());
        assert!(entry.tags.contains(&"resonant-coherence".to_string()));
    }

    #[test]
    fn gallery_index_add_and_len() {
        let mut index = GalleryIndex::new(100);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        index.add(create_entry(
            ArtModality::Poetry {
                text: "test poem".into(),
            },
            test_score(),
            [0.5; 8],
            1,
        ));
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn gallery_by_score_sorted() {
        let mut index = GalleryIndex::new(100);

        let mut low = AestheticScore::uniform(0.2);
        low.compute_composite();
        let mut high = AestheticScore::uniform(0.9);
        high.compute_composite();

        index.add(create_entry(
            ArtModality::Poetry { text: "low".into() },
            low,
            [0.5; 8],
            1,
        ));
        index.add(create_entry(
            ArtModality::Poetry {
                text: "high".into(),
            },
            high,
            [0.5; 8],
            2,
        ));

        let sorted = index.by_score();
        assert!(sorted[0].aesthetic_score.composite > sorted[1].aesthetic_score.composite);
    }

    #[test]
    fn gallery_by_modality() {
        let mut index = GalleryIndex::new(100);
        index.add(create_entry(
            ArtModality::Visual {
                filename: "a.svg".into(),
            },
            test_score(),
            [0.5; 8],
            1,
        ));
        index.add(create_entry(
            ArtModality::Poetry {
                text: "poem".into(),
            },
            test_score(),
            [0.5; 8],
            2,
        ));
        index.add(create_entry(
            ArtModality::Visual {
                filename: "b.svg".into(),
            },
            test_score(),
            [0.5; 8],
            3,
        ));

        assert_eq!(index.by_modality("visual").len(), 2);
        assert_eq!(index.by_modality("poetry").len(), 1);
        assert_eq!(index.by_modality("music").len(), 0);
    }

    #[test]
    fn gallery_is_full() {
        let mut index = GalleryIndex::new(2);
        assert!(!index.is_full());

        index.add(create_entry(
            ArtModality::Poetry { text: "one".into() },
            test_score(),
            [0.5; 8],
            1,
        ));
        index.add(create_entry(
            ArtModality::Poetry { text: "two".into() },
            test_score(),
            [0.5; 8],
            2,
        ));
        assert!(index.is_full());
    }

    #[test]
    fn high_surprise_protected() {
        let mut score = AestheticScore::uniform(0.5);
        score.surprise = 0.9;
        score.compute_composite();
        let entry = create_entry(
            ArtModality::Poetry {
                text: "surprising".into(),
            },
            score,
            [0.5; 8],
            1,
        );
        assert!(entry.protected);
    }

    #[test]
    fn average_score_empty() {
        let index = GalleryIndex::new(100);
        assert_eq!(index.average_score(), 0.0);
    }

    #[test]
    fn top_n() {
        let mut index = GalleryIndex::new(100);
        for i in 0..5 {
            let mut score = AestheticScore::uniform(i as f32 / 5.0);
            score.compute_composite();
            index.add(create_entry(
                ArtModality::Poetry {
                    text: format!("poem {i}"),
                },
                score,
                [0.5; 8],
                i as u64,
            ));
        }
        let top = index.top_n(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].aesthetic_score.composite >= top[1].aesthetic_score.composite);
    }
}
