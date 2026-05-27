// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-curation: prune low-scoring works while preserving experiments.

use crate::GalleryIndex;

/// Curate the gallery: remove lowest-scoring entries when at capacity.
///
/// Rules:
/// - Never prune protected entries (high-surprise experiments).
/// - Remove entries with the lowest aesthetic composite scores.
/// - Keep at least `min_entries` entries.
pub fn curate(index: &mut GalleryIndex, min_entries: usize) {
    if index.entries.len() <= index.max_entries {
        return; // Not at capacity
    }

    let target = index.max_entries.max(min_entries);
    let to_remove = index.entries.len() - target;

    if to_remove == 0 {
        return;
    }

    // Sort by score (lowest first), but skip protected
    let mut scored: Vec<(usize, f32, bool)> = index
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e.aesthetic_score.composite, e.protected))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Collect indices of unprotected lowest-scoring entries
    let mut remove_indices: Vec<usize> = scored
        .iter()
        .filter(|(_, _, protected)| !protected)
        .take(to_remove)
        .map(|(i, _, _)| *i)
        .collect();

    // Remove in reverse order to preserve indices
    remove_indices.sort_unstable();
    remove_indices.reverse();

    for idx in remove_indices {
        if index.entries.len() > min_entries {
            index.entries.remove(idx);
        }
    }
}

/// Compute curation statistics.
pub fn curation_stats(index: &GalleryIndex) -> CurationStats {
    let protected_count = index.entries.iter().filter(|e| e.protected).count();
    let prunable_count = index.entries.len() - protected_count;

    CurationStats {
        total_entries: index.entries.len(),
        protected_count,
        prunable_count,
        at_capacity: index.is_full(),
        average_score: index.average_score(),
    }
}

/// Statistics about the gallery's curation state.
#[derive(Debug, Clone)]
pub struct CurationStats {
    pub total_entries: usize,
    pub protected_count: usize,
    pub prunable_count: usize,
    pub at_capacity: bool,
    pub average_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtModality, create_entry};
    use symthaea_aesthetic::AestheticScore;

    #[test]
    fn curate_removes_lowest() {
        let mut index = GalleryIndex::new(3);
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

        assert_eq!(index.len(), 5);
        curate(&mut index, 1);
        assert!(index.len() <= 3);

        // Remaining entries should be the highest scored
        for entry in &index.entries {
            assert!(
                entry.aesthetic_score.composite > 0.2,
                "low-scored entry survived: {}",
                entry.aesthetic_score.composite
            );
        }
    }

    #[test]
    fn curate_preserves_protected() {
        let mut index = GalleryIndex::new(2);

        // Add a low-scored but protected entry
        let mut low_score = AestheticScore::uniform(0.1);
        low_score.surprise = 0.9; // will be protected
        low_score.compute_composite();
        index.add(create_entry(
            ArtModality::Poetry {
                text: "surprising".into(),
            },
            low_score,
            [0.5; 8],
            1,
        ));

        // Add high-scored entries
        for i in 2..5 {
            let mut score = AestheticScore::uniform(0.8);
            score.compute_composite();
            index.add(create_entry(
                ArtModality::Poetry {
                    text: format!("good {i}"),
                },
                score,
                [0.5; 8],
                i as u64,
            ));
        }

        curate(&mut index, 1);

        // Protected entry should survive
        let protected = index.entries.iter().any(|e| e.protected);
        assert!(protected, "protected entry was pruned");
    }

    #[test]
    fn curate_noop_when_under_capacity() {
        let mut index = GalleryIndex::new(100);
        let mut score = AestheticScore::uniform(0.5);
        score.compute_composite();
        index.add(create_entry(
            ArtModality::Poetry {
                text: "poem".into(),
            },
            score,
            [0.5; 8],
            1,
        ));

        curate(&mut index, 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn curation_stats_correct() {
        let mut index = GalleryIndex::new(5);
        for i in 0..3 {
            let mut score = AestheticScore::uniform(0.5);
            if i == 0 {
                score.surprise = 0.9;
            }
            score.compute_composite();
            index.add(create_entry(
                ArtModality::Poetry {
                    text: format!("{i}"),
                },
                score,
                [0.5; 8],
                i as u64,
            ));
        }

        let stats = curation_stats(&index);
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.protected_count, 1);
        assert_eq!(stats.prunable_count, 2);
        assert!(!stats.at_capacity);
    }
}
