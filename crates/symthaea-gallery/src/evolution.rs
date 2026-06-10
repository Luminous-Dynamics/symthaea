// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolution tracking: detect style periods and aesthetic trend lines.
//!
//! Analyzes the gallery's history to identify dominant harmony shifts,
//! score trajectories, and style transitions.

use crate::GalleryIndex;
use serde::{Deserialize, Serialize};

/// A detected style period in the gallery's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePeriod {
    /// Start cycle of this period.
    pub start_cycle: u64,
    /// End cycle of this period.
    pub end_cycle: u64,
    /// Dominant harmony index (0-7) during this period.
    pub dominant_harmony: usize,
    /// Dominant harmony name.
    pub harmony_name: String,
    /// Average aesthetic score during this period.
    pub average_score: f32,
    /// Number of artworks in this period.
    pub artwork_count: usize,
}

/// Evolution summary for the gallery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSummary {
    /// Detected style periods (chronological).
    pub periods: Vec<StylePeriod>,
    /// Overall score trend: positive = improving, negative = declining.
    pub score_trend: f32,
    /// Total artworks analyzed.
    pub total_artworks: usize,
}

const HARMONY_NAMES: [&str; 8] = [
    "ResonantCoherence",
    "PanSentientFlourishing",
    "IntegralWisdom",
    "InfinitePlay",
    "UniversalInterconnectedness",
    "SacredReciprocity",
    "EvolutionaryProgression",
    "SacredStillness",
];

/// Analyze the gallery's evolution.
///
/// Scans entries chronologically, detecting shifts in dominant harmony
/// and computing score trend lines.
pub fn analyze_evolution(index: &GalleryIndex, window_size: usize) -> EvolutionSummary {
    if index.entries.is_empty() {
        return EvolutionSummary {
            periods: Vec::new(),
            score_trend: 0.0,
            total_artworks: 0,
        };
    }

    let _window = window_size.max(2);
    let mut periods = Vec::new();

    // Detect style periods via sliding window of dominant harmony
    let mut current_dominant = dominant_harmony(&index.entries[0].harmony_activations);
    let mut period_start = index.entries[0].created_at_cycle;
    let mut period_scores: Vec<f32> = vec![index.entries[0].aesthetic_score.composite];
    let mut period_count = 1usize;

    for entry in index.entries.iter().skip(1) {
        let dom = dominant_harmony(&entry.harmony_activations);

        if dom != current_dominant {
            // End current period, start new one
            let avg_score = period_scores.iter().sum::<f32>() / period_scores.len().max(1) as f32;
            periods.push(StylePeriod {
                start_cycle: period_start,
                end_cycle: entry.created_at_cycle,
                dominant_harmony: current_dominant,
                harmony_name: HARMONY_NAMES[current_dominant].to_string(),
                average_score: avg_score,
                artwork_count: period_count,
            });

            current_dominant = dom;
            period_start = entry.created_at_cycle;
            period_scores.clear();
            period_count = 0;
        }

        period_scores.push(entry.aesthetic_score.composite);
        period_count += 1;
    }

    // Close final period
    let last_cycle = index
        .entries
        .last()
        .map(|e| e.created_at_cycle)
        .unwrap_or(period_start);
    let avg_score = period_scores.iter().sum::<f32>() / period_scores.len().max(1) as f32;
    periods.push(StylePeriod {
        start_cycle: period_start,
        end_cycle: last_cycle,
        dominant_harmony: current_dominant,
        harmony_name: HARMONY_NAMES[current_dominant].to_string(),
        average_score: avg_score,
        artwork_count: period_count,
    });

    // Compute score trend (simple linear regression slope)
    let score_trend = compute_trend(index);

    EvolutionSummary {
        periods,
        score_trend,
        total_artworks: index.entries.len(),
    }
}

/// Find the index of the dominant (highest) harmony activation.
fn dominant_harmony(activations: &[f32; 8]) -> usize {
    activations
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute linear trend of aesthetic scores (positive = improving).
fn compute_trend(index: &GalleryIndex) -> f32 {
    let n = index.entries.len();
    if n < 2 {
        return 0.0;
    }

    let n_f = n as f32;
    let sum_x: f32 = (0..n).map(|i| i as f32).sum();
    let sum_y: f32 = index
        .entries
        .iter()
        .map(|e| e.aesthetic_score.composite)
        .sum();
    let sum_xy: f32 = index
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| i as f32 * e.aesthetic_score.composite)
        .sum();
    let sum_x2: f32 = (0..n).map(|i| (i as f32).powi(2)).sum();

    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    (n_f * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtModality, create_entry};
    use symthaea_aesthetic::AestheticScore;

    fn make_entry(score: f32, dominant: usize, cycle: u64) -> crate::GalleryEntry {
        let mut harmonies = [0.3f32; 8];
        harmonies[dominant] = 0.9;
        let mut s = AestheticScore::uniform(score);
        s.compute_composite();
        create_entry(
            ArtModality::Poetry {
                text: format!("cycle {cycle}"),
            },
            s,
            harmonies,
            cycle,
        )
    }

    #[test]
    fn empty_gallery_empty_evolution() {
        let index = GalleryIndex::new(100);
        let evo = analyze_evolution(&index, 5);
        assert!(evo.periods.is_empty());
        assert_eq!(evo.total_artworks, 0);
    }

    #[test]
    fn single_period_detected() {
        let mut index = GalleryIndex::new(100);
        for i in 0..5 {
            index.add(make_entry(0.5, 3, i)); // all InfinitePlay
        }

        let evo = analyze_evolution(&index, 5);
        assert_eq!(evo.periods.len(), 1);
        assert_eq!(evo.periods[0].dominant_harmony, 3);
        assert_eq!(evo.periods[0].harmony_name, "InfinitePlay");
        assert_eq!(evo.periods[0].artwork_count, 5);
    }

    #[test]
    fn multiple_periods_detected() {
        let mut index = GalleryIndex::new(100);
        // InfinitePlay period
        for i in 0..3 {
            index.add(make_entry(0.5, 3, i));
        }
        // SacredStillness period
        for i in 3..6 {
            index.add(make_entry(0.6, 7, i));
        }

        let evo = analyze_evolution(&index, 5);
        assert_eq!(evo.periods.len(), 2);
        assert_eq!(evo.periods[0].harmony_name, "InfinitePlay");
        assert_eq!(evo.periods[1].harmony_name, "SacredStillness");
    }

    #[test]
    fn improving_trend_positive() {
        let mut index = GalleryIndex::new(100);
        for i in 0..10 {
            index.add(make_entry(0.3 + i as f32 * 0.05, 0, i));
        }
        let evo = analyze_evolution(&index, 5);
        assert!(evo.score_trend > 0.0, "trend = {}", evo.score_trend);
    }

    #[test]
    fn declining_trend_negative() {
        let mut index = GalleryIndex::new(100);
        for i in 0..10 {
            index.add(make_entry(0.8 - i as f32 * 0.05, 0, i));
        }
        let evo = analyze_evolution(&index, 5);
        assert!(evo.score_trend < 0.0, "trend = {}", evo.score_trend);
    }
}
