// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measure inter-onset space in a repeating ensemble groove.
//!
//! This is deliberately named "inter-onset space", not acoustic silence:
//! GrooveEventV1 records attacks but not note durations, release envelopes,
//! sustain pedals, reverberation or sample tails. The report therefore
//! measures distances between attacks, not audible silence. This distinction
//! is important for sparse styles: a long inter-onset interval can still
//! contain a sustained bass or texture.
//!
//! The analysis is source-independent. It can measure whether a declared
//! rhythmic layer is sparse or dense without asserting that sparsity makes a
//! pattern Gqom, or that a particular gap is culturally canonical.

use serde::{Deserialize, Serialize};

use crate::ensemble_groove::{
    EnsembleGrooveV1, GrooveValidationIssue, MAX_SUPERCYCLE_TICKS, validate_ensemble_groove,
};

/// Version identifier for the inter-onset-space analysis.
pub const GROOVE_SPACE_ANALYSIS_SCHEMA_V1: &str = "groove-space-analysis-v1";

/// Per-layer inter-onset space measured over the shared joint supercycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveLayerSpaceAnalysisV1 {
    pub layer_id: String,
    pub role: String,
    pub unique_attack_count: u64,
    /// Sorted circular gaps between consecutive unique attacks.
    /// Empty when the layer has no attacks.
    pub inter_onset_gaps_ticks: Vec<u64>,
    pub shortest_inter_onset_gap_ticks: Option<u64>,
    pub longest_inter_onset_gap_ticks: Option<u64>,
    /// Exact mean represented as horizon / unique_attack_count.
    pub mean_gap_numerator_ticks: Option<u64>,
    pub mean_gap_denominator: Option<u64>,
    /// Number of circular gaps greater than or equal to the caller's threshold.
    pub long_gap_count: u64,
    /// Fraction of the cycle covered by gaps >= threshold, in basis points.
    /// This is an inter-onset-space descriptor, not an acoustic-silence claim.
    pub long_gap_span_basis_points: u16,
}

/// Source-independent inter-onset-space analysis for all layers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveSpaceAnalysisV1 {
    pub schema: String,
    pub supercycle_ticks: u64,
    pub long_gap_threshold_ticks: u32,
    pub layers: Vec<GrooveLayerSpaceAnalysisV1>,
}

/// Analyze cyclic inter-onset gaps for every role layer.
///
/// The same full LCM horizon used by ensemble-groove analysis is preserved.
/// A threshold of 24 ticks means every circular inter-onset gap >= 24 ticks
/// contributes its full span to long_gap_span_basis_points.
pub fn analyze_groove_space(
    groove: &EnsembleGrooveV1,
    long_gap_threshold_ticks: u32,
) -> Result<GrooveSpaceAnalysisV1, Vec<GrooveValidationIssue>> {
    let supercycle_ticks = validate_ensemble_groove(groove)?;
    if long_gap_threshold_ticks == 0
        || long_gap_threshold_ticks as u64 > supercycle_ticks
        || supercycle_ticks > MAX_SUPERCYCLE_TICKS
    {
        return Err(vec![GrooveValidationIssue {
            code: "invalid_long_gap_threshold".into(),
            path: "long_gap_threshold_ticks".into(),
            message: format!(
                "threshold must be in [1, {supercycle_ticks}] ticks"
            ),
        }]);
    }

    let mut layers = Vec::with_capacity(groove.layers.len());
    for layer in &groove.layers {
        let repetitions = supercycle_ticks / layer.cycle_ticks as u64;
        let mut positions = Vec::new();
        for repetition in 0..repetitions {
            let base = repetition * layer.cycle_ticks as u64;
            for event in &layer.events {
                positions.push(base + event.onset_tick as u64);
            }
        }
        positions.sort_unstable();
        positions.dedup();

        let (gaps, longest, shortest, mean_num, mean_den, long_count, span_bps) =
            circular_gap_summary(&positions, supercycle_ticks, long_gap_threshold_ticks as u64);

        layers.push(GrooveLayerSpaceAnalysisV1 {
            layer_id: layer.id.clone(),
            role: layer.role.clone(),
            unique_attack_count: positions.len() as u64,
            inter_onset_gaps_ticks: gaps,
            shortest_inter_onset_gap_ticks: shortest,
            longest_inter_onset_gap_ticks: longest,
            mean_gap_numerator_ticks: mean_num,
            mean_gap_denominator: mean_den,
            long_gap_count: long_count,
            long_gap_span_basis_points: span_bps,
        });
    }

    Ok(GrooveSpaceAnalysisV1 {
        schema: GROOVE_SPACE_ANALYSIS_SCHEMA_V1.into(),
        supercycle_ticks,
        long_gap_threshold_ticks,
        layers,
    })
}

fn circular_gap_summary(
    positions: &[u64],
    horizon: u64,
    threshold: u64,
) -> (Vec<u64>, Option<u64>, Option<u64>, Option<u64>, Option<u64>, u64, u16) {
    if positions.is_empty() {
        // An explicitly empty layer is a complete rest over this horizon.
        return (Vec::new(), None, None, None, None, 1, 10_000);
    }

    let mut gaps = Vec::with_capacity(positions.len());
    for index in 0..positions.len() {
        let current = positions[index];
        let next = positions[(index + 1) % positions.len()];
        let gap = if index + 1 == positions.len() {
            horizon - current + next
        } else {
            next - current
        };
        gaps.push(gap);
    }
    gaps.sort_unstable();

    let shortest = gaps.first().copied();
    let longest = gaps.last().copied();
    let long_gap_count = gaps.iter().filter(|gap| **gap >= threshold).count() as u64;
    let long_gap_ticks: u64 = gaps
        .iter()
        .filter(|gap| **gap >= threshold)
        .copied()
        .sum();
    let long_gap_span_basis_points =
        ((long_gap_ticks.saturating_mul(10_000) / horizon) as u16).min(10_000);

    (
        gaps,
        longest,
        shortest,
        Some(horizon),
        Some(positions.len() as u64),
        long_gap_count,
        long_gap_span_basis_points,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_groove::{
        ENSEMBLE_GROOVE_SCHEMA_V1, EnsembleGrooveV1, GrooveEventV1, GrooveLayerV1,
    };

    fn groove(events: Vec<u32>) -> EnsembleGrooveV1 {
        EnsembleGrooveV1 {
            schema: ENSEMBLE_GROOVE_SCHEMA_V1.into(),
            ticks_per_beat: 24,
            layers: vec![GrooveLayerV1 {
                id: "primary".into(),
                role: "primary_rhythm".into(),
                cycle_ticks: 96,
                events: events
                    .into_iter()
                    .map(|onset_tick| GrooveEventV1 {
                        onset_tick,
                        accent: 1,
                    })
                    .collect(),
            }],
            constraints: vec![],
        }
    }

    #[test]
    fn distinguishes_sparse_and_dense_inter_onset_space() {
        let sparse = analyze_groove_space(&groove(vec![0, 48]), 24).unwrap();
        let dense = analyze_groove_space(&groove(vec![0, 12, 24, 36, 48, 60, 72, 84]), 24)
            .unwrap();
        let sparse_layer = &sparse.layers[0];
        let dense_layer = &dense.layers[0];

        assert_eq!(sparse_layer.inter_onset_gaps_ticks, vec![48, 48]);
        assert_eq!(sparse_layer.longest_inter_onset_gap_ticks, Some(48));
        assert_eq!(sparse_layer.long_gap_span_basis_points, 10_000);
        assert_eq!(dense_layer.inter_onset_gaps_ticks, vec![12; 8]);
        assert_eq!(dense_layer.longest_inter_onset_gap_ticks, Some(12));
        assert_eq!(dense_layer.long_gap_span_basis_points, 0);
    }

    #[test]
    fn circular_gaps_conserve_the_entire_supercycle() {
        let report = analyze_groove_space(&groove(vec![5, 29, 61]), 20).unwrap();
        assert_eq!(report.layers[0].inter_onset_gaps_ticks.iter().sum::<u64>(), 96);
        assert_eq!(
            report.layers[0].mean_gap_numerator_ticks,
            Some(96)
        );
        assert_eq!(report.layers[0].mean_gap_denominator, Some(3));
    }

    #[test]
    fn a_single_attack_has_one_full_horizon_gap() {
        let report = analyze_groove_space(&groove(vec![17]), 48).unwrap();
        assert_eq!(report.layers[0].inter_onset_gaps_ticks, vec![96]);
        assert_eq!(report.layers[0].longest_inter_onset_gap_ticks, Some(96));
    }

    #[test]
    fn explicit_empty_layer_is_reported_as_full_unattacked_span() {
        let report = analyze_groove_space(&groove(vec![]), 24).unwrap();
        assert_eq!(report.layers[0].unique_attack_count, 0);
        assert_eq!(report.layers[0].longest_inter_onset_gap_ticks, None);
        assert_eq!(report.layers[0].long_gap_span_basis_points, 10_000);
    }

    #[test]
    fn common_phase_translation_preserves_circular_gap_profile() {
        let original = analyze_groove_space(&groove(vec![5, 29, 61]), 20).unwrap();
        let translated = analyze_groove_space(&groove(vec![22, 46, 78]), 20).unwrap();
        assert_eq!(
            original.layers[0].inter_onset_gaps_ticks,
            translated.layers[0].inter_onset_gaps_ticks
        );
        assert_eq!(
            original.layers[0].long_gap_span_basis_points,
            translated.layers[0].long_gap_span_basis_points
        );
        assert_eq!(
            original.layers[0].longest_inter_onset_gap_ticks,
            translated.layers[0].longest_inter_onset_gap_ticks
        );
    }

    #[test]
    fn rejects_zero_or_overlong_thresholds() {
        assert_eq!(
            analyze_groove_space(&groove(vec![0, 48]), 0)
                .unwrap_err()[0]
                .code,
            "invalid_long_gap_threshold"
        );
        assert_eq!(
            analyze_groove_space(&groove(vec![0, 48]), 97)
                .unwrap_err()[0]
                .code,
            "invalid_long_gap_threshold"
        );
    }
}
