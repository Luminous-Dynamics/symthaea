// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sensitivity analysis for preregistered multiscale causal sweeps.
//!
//! GEOM-002B asks whether a causal-scale result is robust to alternative,
//! explicitly declared coarse-grainings with the same state-count profile.
//! It reports ranges and peak-scale stability rather than collapsing variants
//! into a single preferred partition.

use super::multiscale_causal::{
    CausalAdvantageRule, CoarseGrainingSpec, MultiscaleCausalError, MultiscaleCausalSweep,
    analyze_multiscale_causality,
};
use std::collections::HashSet;
use std::fmt;

/// One preregistered coarse-graining family to compare with its peers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepVariant {
    pub label: String,
    pub coarse_grainings: Vec<CoarseGrainingSpec>,
}

impl SweepVariant {
    pub fn new(label: impl Into<String>, coarse_grainings: Vec<CoarseGrainingSpec>) -> Self {
        Self {
            label: label.into(),
            coarse_grainings,
        }
    }
}

/// Fail-closed errors for sensitivity analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum CausalSensitivityError {
    TooFewVariants { observed: usize },
    EmptyVariantLabel,
    DuplicateVariantLabel { label: String },
    VariantSweepFailed {
        variant: String,
        source: MultiscaleCausalError,
    },
    IncomparableScaleCount {
        variant: String,
        expected: usize,
        observed: usize,
    },
    IncomparableStateCount {
        variant: String,
        scale_index: usize,
        expected: usize,
        observed: usize,
    },
}

impl fmt::Display for CausalSensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewVariants { observed } => write!(
                f,
                "sensitivity analysis requires at least two variants; observed {observed}"
            ),
            Self::EmptyVariantLabel => write!(f, "variant labels must not be empty"),
            Self::DuplicateVariantLabel { label } => {
                write!(f, "variant label must be unique: {label}")
            }
            Self::VariantSweepFailed { variant, source } => {
                write!(f, "variant {variant} failed causal sweep validation: {source}")
            }
            Self::IncomparableScaleCount {
                variant,
                expected,
                observed,
            } => write!(
                f,
                "variant {variant} has {observed} scales; expected {expected} for comparison"
            ),
            Self::IncomparableStateCount {
                variant,
                scale_index,
                expected,
                observed,
            } => write!(
                f,
                "variant {variant} scale {scale_index} has {observed} states; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for CausalSensitivityError {}

/// Closed interval spanning one metric across preregistered variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricRange {
    pub min: f64,
    pub max: f64,
}

impl MetricRange {
    pub fn width(self) -> f64 {
        self.max - self.min
    }
}

/// Sensitivity envelope for one scale index.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleSensitivity {
    pub scale_index: usize,
    pub state_count: usize,
    pub effective_information_bits: MetricRange,
    pub determinism: MetricRange,
    pub degeneracy: MetricRange,
    pub advantage_vs_finest_bits: MetricRange,
}

/// Aggregate robustness report across preregistered coarse-graining variants.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalSensitivityEnvelope {
    pub variant_count: usize,
    pub scales: Vec<ScaleSensitivity>,
    /// Count of variants whose maximum-EI scale occurs at each scale index.
    pub peak_scale_counts: Vec<usize>,
    /// Largest peak-scale count divided by variant count, in [0, 1].
    pub peak_scale_stability: f64,
    pub criterion_met_count: usize,
    pub criterion_met_fraction: f64,
}

fn range(values: impl IntoIterator<Item = f64>) -> MetricRange {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    MetricRange { min, max }
}

fn validate_variant_labels(variants: &[SweepVariant]) -> Result<(), CausalSensitivityError> {
    let mut labels = HashSet::with_capacity(variants.len());
    for variant in variants {
        let label = variant.label.trim();
        if label.is_empty() {
            return Err(CausalSensitivityError::EmptyVariantLabel);
        }
        if !labels.insert(label.to_owned()) {
            return Err(CausalSensitivityError::DuplicateVariantLabel {
                label: label.to_owned(),
            });
        }
    }
    Ok(())
}

fn verify_comparable_profiles(
    label: &str,
    reference: &MultiscaleCausalSweep,
    candidate: &MultiscaleCausalSweep,
) -> Result<(), CausalSensitivityError> {
    if candidate.scales.len() != reference.scales.len() {
        return Err(CausalSensitivityError::IncomparableScaleCount {
            variant: label.to_owned(),
            expected: reference.scales.len(),
            observed: candidate.scales.len(),
        });
    }

    for (scale_index, (expected, observed)) in reference
        .scales
        .iter()
        .zip(candidate.scales.iter())
        .enumerate()
    {
        if expected.state_count != observed.state_count {
            return Err(CausalSensitivityError::IncomparableStateCount {
                variant: label.to_owned(),
                scale_index,
                expected: expected.state_count,
                observed: observed.state_count,
            });
        }
    }

    Ok(())
}

/// Compare causal-scale measurements across preregistered coarse-grainings.
///
/// All variants share the same finest TPM and decision rule. For per-scale
/// ranges to be interpretable, every variant must also share the same sequence
/// of state counts, though the actual state assignments may differ.
pub fn analyze_coarse_graining_sensitivity(
    finest_label: &str,
    finest_tpm: &[Vec<f64>],
    variants: &[SweepVariant],
    rule: CausalAdvantageRule,
) -> Result<CausalSensitivityEnvelope, CausalSensitivityError> {
    if variants.len() < 2 {
        return Err(CausalSensitivityError::TooFewVariants {
            observed: variants.len(),
        });
    }
    validate_variant_labels(variants)?;

    let mut sweeps = Vec::with_capacity(variants.len());
    for variant in variants {
        let sweep = analyze_multiscale_causality(
            finest_label,
            finest_tpm,
            &variant.coarse_grainings,
            rule,
        )
        .map_err(|source| CausalSensitivityError::VariantSweepFailed {
            variant: variant.label.clone(),
            source,
        })?;
        sweeps.push(sweep);
    }

    let reference = &sweeps[0];
    for (variant, sweep) in variants.iter().zip(sweeps.iter()).skip(1) {
        verify_comparable_profiles(&variant.label, reference, sweep)?;
    }

    let mut scales = Vec::with_capacity(reference.scales.len());
    for scale_index in 0..reference.scales.len() {
        scales.push(ScaleSensitivity {
            scale_index,
            state_count: reference.scales[scale_index].state_count,
            effective_information_bits: range(
                sweeps
                    .iter()
                    .map(|sweep| sweep.scales[scale_index].effective_information_bits),
            ),
            determinism: range(
                sweeps
                    .iter()
                    .map(|sweep| sweep.scales[scale_index].determinism),
            ),
            degeneracy: range(
                sweeps
                    .iter()
                    .map(|sweep| sweep.scales[scale_index].degeneracy),
            ),
            advantage_vs_finest_bits: range(
                sweeps
                    .iter()
                    .map(|sweep| sweep.scales[scale_index].advantage_vs_finest_bits),
            ),
        });
    }

    let mut peak_scale_counts = vec![0usize; reference.scales.len()];
    let mut criterion_met_count = 0usize;
    for sweep in &sweeps {
        peak_scale_counts[sweep.decision.peak_scale_index] += 1;
        if sweep.decision.criterion_met {
            criterion_met_count += 1;
        }
    }

    let max_peak_count = peak_scale_counts.iter().copied().max().unwrap_or(0);
    let variant_count = variants.len();

    Ok(CausalSensitivityEnvelope {
        variant_count,
        scales,
        peak_scale_counts,
        peak_scale_stability: max_peak_count as f64 / variant_count as f64,
        criterion_met_count,
        criterion_met_fraction: criterion_met_count as f64 / variant_count as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-10;

    fn asymmetric_fine_tpm() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        ]
    }

    #[test]
    fn identical_variants_have_zero_metric_ranges_and_stable_peak() {
        let fine = asymmetric_fine_tpm();
        let variants = [
            SweepVariant::new(
                "partition-a",
                vec![CoarseGrainingSpec::new("macro-a", vec![0, 0, 0, 1])],
            ),
            SweepVariant::new(
                "partition-a-replication",
                vec![CoarseGrainingSpec::new("macro-a-copy", vec![0, 0, 0, 1])],
            ),
        ];
        let rule = CausalAdvantageRule::new(0.1).expect("valid rule");
        let envelope = analyze_coarse_graining_sensitivity("fine", &fine, &variants, rule)
            .expect("comparable variants");

        assert_eq!(envelope.variant_count, 2);
        assert_eq!(envelope.peak_scale_counts, vec![0, 2]);
        assert!((envelope.peak_scale_stability - 1.0).abs() < TOLERANCE);
        assert_eq!(envelope.criterion_met_count, 2);
        assert!((envelope.criterion_met_fraction - 1.0).abs() < TOLERANCE);
        assert!(envelope.scales.iter().all(|scale| {
            scale.effective_information_bits.width().abs() < TOLERANCE
                && scale.advantage_vs_finest_bits.width().abs() < TOLERANCE
        }));
    }

    #[test]
    fn alternative_partition_exposes_partition_sensitive_causal_advantage() {
        let fine = asymmetric_fine_tpm();
        let variants = [
            SweepVariant::new(
                "degeneracy-aligned",
                vec![CoarseGrainingSpec::new("macro-aligned", vec![0, 0, 0, 1])],
            ),
            SweepVariant::new(
                "balanced-alternative",
                vec![CoarseGrainingSpec::new("macro-balanced", vec![0, 0, 1, 1])],
            ),
        ];
        let rule = CausalAdvantageRule::new(0.1).expect("valid rule");
        let envelope = analyze_coarse_graining_sensitivity("fine", &fine, &variants, rule)
            .expect("comparable variants");

        assert_eq!(envelope.peak_scale_counts, vec![1, 1]);
        assert!((envelope.peak_scale_stability - 0.5).abs() < TOLERANCE);
        assert_eq!(envelope.criterion_met_count, 1);
        assert!((envelope.criterion_met_fraction - 0.5).abs() < TOLERANCE);
        assert!(envelope.scales[1].effective_information_bits.width() > 0.5);
        assert!(envelope.scales[1].advantage_vs_finest_bits.min < 0.0);
        assert!(envelope.scales[1].advantage_vs_finest_bits.max > 0.1);
    }

    #[test]
    fn incomparable_state_count_profiles_fail_closed() {
        let fine = asymmetric_fine_tpm();
        let variants = [
            SweepVariant::new(
                "two-state",
                vec![CoarseGrainingSpec::new("macro-2", vec![0, 0, 1, 1])],
            ),
            SweepVariant::new(
                "three-state",
                vec![CoarseGrainingSpec::new("macro-3", vec![0, 1, 2, 2])],
            ),
        ];
        let rule = CausalAdvantageRule::new(0.0).expect("valid rule");
        assert!(matches!(
            analyze_coarse_graining_sensitivity("fine", &fine, &variants, rule),
            Err(CausalSensitivityError::IncomparableStateCount {
                scale_index: 1,
                expected: 2,
                observed: 3,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_or_missing_variant_labels_fail_closed() {
        let fine = asymmetric_fine_tpm();
        let rule = CausalAdvantageRule::new(0.0).expect("valid rule");
        let duplicate = [
            SweepVariant::new("same", vec![CoarseGrainingSpec::new("a", vec![0, 0, 1, 1])]),
            SweepVariant::new("same", vec![CoarseGrainingSpec::new("b", vec![0, 1, 0, 1])]),
        ];
        assert!(matches!(
            analyze_coarse_graining_sensitivity("fine", &fine, &duplicate, rule),
            Err(CausalSensitivityError::DuplicateVariantLabel { .. })
        ));

        let missing = [
            SweepVariant::new("a", vec![CoarseGrainingSpec::new("a", vec![0, 0, 1, 1])]),
            SweepVariant::new(" ", vec![CoarseGrainingSpec::new("b", vec![0, 1, 0, 1])]),
        ];
        assert!(matches!(
            analyze_coarse_graining_sensitivity("fine", &fine, &missing, rule),
            Err(CausalSensitivityError::EmptyVariantLabel)
        ));
    }

    #[test]
    fn fewer_than_two_variants_is_not_a_sensitivity_analysis() {
        let fine = asymmetric_fine_tpm();
        let one = [SweepVariant::new(
            "only",
            vec![CoarseGrainingSpec::new("macro", vec![0, 0, 1, 1])],
        )];
        let rule = CausalAdvantageRule::new(0.0).expect("valid rule");
        assert!(matches!(
            analyze_coarse_graining_sensitivity("fine", &fine, &one, rule),
            Err(CausalSensitivityError::TooFewVariants { observed: 1 })
        ));
    }
}
