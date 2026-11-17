// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative rank/template ambiguity under explicit operator-norm
//! perturbation bounds.
//!
//! If ||Delta V_l||_2 <= delta_l, Weyl's singular-value perturbation theorem
//! bounds every singular-value change by delta_l. The degree-zero scale in
//! SymFP's shared rank threshold is uncertain too, so this module propagates
//! an interval for tau before classifying every degree.
//!
//! The input bounds are assumptions supplied by the caller. This module does
//! not prove those bounds, infer them from the SVD convergence residual, or
//! turn spectral compatibility into a symmetry certificate.

use std::collections::BTreeSet;
use std::fmt;

use super::{PointGroupTemplateV1, SymFpErrorV1, SymFpReceiptV1, analyze_spectra};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericalRankIntervalV1 {
    pub degree: usize,
    pub minimum_rank: usize,
    pub maximum_rank: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateDistanceIntervalV1 {
    pub schoenflies: &'static str,
    pub minimum_l1_distance: usize,
    pub maximum_l1_distance: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustPointGroupMatchV1 {
    pub rank_threshold_lower: f64,
    pub rank_threshold_upper: f64,
    pub rank_intervals: Vec<NumericalRankIntervalV1>,
    pub template_distance_intervals: Vec<TemplateDistanceIntervalV1>,
    /// Present only when one candidate's worst-case distance is strictly
    /// below every competitor's best-case distance.
    pub guaranteed_nearest_template: Option<&'static str>,
    /// Candidates that cannot be ruled out by pairwise interval dominance.
    /// This is a conservative possibility set, not a posterior distribution.
    pub not_ruled_out_templates: Vec<&'static str>,
    pub authority: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustSymFpErrorV1 {
    InvalidReceipt,
    BoundCountMismatch,
    InvalidPerturbationBound,
    EmptyCandidateFamily,
    DuplicateCandidateName,
    ArithmeticOverflow,
    BaseAnalysis(SymFpErrorV1),
}

impl fmt::Display for RobustSymFpErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RobustSymFpErrorV1 {}

fn distance_interval(target_rank: usize, observed: NumericalRankIntervalV1) -> (usize, usize) {
    let minimum = observed
        .minimum_rank
        .saturating_sub(target_rank)
        .max(target_rank.saturating_sub(observed.maximum_rank));
    let maximum = target_rank
        .abs_diff(observed.minimum_rank)
        .max(target_rank.abs_diff(observed.maximum_rank));
    (minimum, maximum)
}

/// Classify rank templates under explicit per-degree spectral perturbation
/// bounds.
///
/// operator_norm_error_bounds[l] is a caller-supplied upper bound delta_l on
/// the spectral-norm perturbation of the degree-l representation block:
///
/// `||Delta V_l||_2 <= delta_l`.
///
/// It is propagated using Weyl's singular-value perturbation theorem. The
/// degree-zero singular value used in SymFP's common threshold is also given
/// an interval, so the threshold interval is propagated to every degree.
///
/// A template is a guaranteed nearest match only when its maximum possible
/// L1 distance is strictly smaller than every competitor's minimum possible
/// distance. Otherwise not_ruled_out_templates reports candidates that
/// cannot be excluded by interval dominance. The output remains spectral
/// compatibility evidence only.
pub fn classify_rank_templates_with_perturbation_bounds(
    receipt: &SymFpReceiptV1,
    candidates: &[PointGroupTemplateV1],
    operator_norm_error_bounds: &[f64],
) -> Result<RobustPointGroupMatchV1, RobustSymFpErrorV1> {
    let replayed = analyze_spectra(&receipt.spectra, receipt.profile)
        .map_err(RobustSymFpErrorV1::BaseAnalysis)?;
    if replayed.degrees != receipt.degrees
        || replayed.rank_sequence != receipt.rank_sequence
        || replayed.tau.to_bits() != receipt.tau.to_bits()
    {
        return Err(RobustSymFpErrorV1::InvalidReceipt);
    }
    if operator_norm_error_bounds.len() != receipt.spectra.len() {
        return Err(RobustSymFpErrorV1::BoundCountMismatch);
    }
    if operator_norm_error_bounds
        .iter()
        .any(|bound| !bound.is_finite() || *bound < 0.0)
    {
        return Err(RobustSymFpErrorV1::InvalidPerturbationBound);
    }
    if candidates.is_empty() {
        return Err(RobustSymFpErrorV1::EmptyCandidateFamily);
    }
    let mut candidate_names = BTreeSet::new();
    for candidate in candidates {
        if !candidate_names.insert(candidate.schoenflies) {
            return Err(RobustSymFpErrorV1::DuplicateCandidateName);
        }
    }

    let degree_zero_leading = receipt.spectra[0].singular_values[0];
    let degree_zero_error = operator_norm_error_bounds[0];
    let degree_zero_lower = (degree_zero_leading - degree_zero_error).max(0.0);
    let degree_zero_upper = degree_zero_leading + degree_zero_error;
    let rank_threshold_lower = receipt
        .profile
        .tau_abs
        .max(receipt.profile.tau_rel * degree_zero_lower);
    let rank_threshold_upper = receipt
        .profile
        .tau_abs
        .max(receipt.profile.tau_rel * degree_zero_upper);
    if !degree_zero_upper.is_finite()
        || !rank_threshold_lower.is_finite()
        || !rank_threshold_upper.is_finite()
    {
        return Err(RobustSymFpErrorV1::ArithmeticOverflow);
    }

    let mut rank_intervals = Vec::with_capacity(receipt.spectra.len());
    for (block, error_bound) in receipt
        .spectra
        .iter()
        .zip(operator_norm_error_bounds.iter().copied())
    {
        let mut minimum_rank = 0usize;
        let mut maximum_rank = 0usize;
        for singular_value in &block.singular_values {
            let lower = (*singular_value - error_bound).max(0.0);
            let upper = *singular_value + error_bound;
            if !lower.is_finite() || !upper.is_finite() {
                return Err(RobustSymFpErrorV1::ArithmeticOverflow);
            }
            if lower > rank_threshold_upper {
                minimum_rank += 1;
            }
            if upper > rank_threshold_lower {
                maximum_rank += 1;
            }
        }
        if minimum_rank > maximum_rank {
            return Err(RobustSymFpErrorV1::InvalidReceipt);
        }
        rank_intervals.push(NumericalRankIntervalV1 {
            degree: block.degree,
            minimum_rank,
            maximum_rank,
        });
    }

    let mut template_distance_intervals = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut minimum_l1_distance = 0usize;
        let mut maximum_l1_distance = 0usize;
        for rank_interval in &rank_intervals {
            let expected_rank = candidate.fixed_dimensions[rank_interval.degree] as usize;
            let (minimum, maximum) = distance_interval(expected_rank, *rank_interval);
            minimum_l1_distance = minimum_l1_distance
                .checked_add(minimum)
                .ok_or(RobustSymFpErrorV1::ArithmeticOverflow)?;
            maximum_l1_distance = maximum_l1_distance
                .checked_add(maximum)
                .ok_or(RobustSymFpErrorV1::ArithmeticOverflow)?;
        }
        template_distance_intervals.push(TemplateDistanceIntervalV1 {
            schoenflies: candidate.schoenflies,
            minimum_l1_distance,
            maximum_l1_distance,
        });
    }

    let mut guaranteed_nearest_template = None;
    for (index, candidate) in template_distance_intervals.iter().enumerate() {
        let beats_every_competitor = template_distance_intervals
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != index)
            .all(|(_, other)| candidate.maximum_l1_distance < other.minimum_l1_distance);
        if beats_every_competitor {
            guaranteed_nearest_template = Some(candidate.schoenflies);
            break;
        }
    }

    let mut not_ruled_out_templates = Vec::new();
    for (index, candidate) in template_distance_intervals.iter().enumerate() {
        let dominated_by_competitor =
            template_distance_intervals
                .iter()
                .enumerate()
                .any(|(other_index, other)| {
                    other_index != index
                        && other.maximum_l1_distance < candidate.minimum_l1_distance
                });
        if !dominated_by_competitor {
            not_ruled_out_templates.push(candidate.schoenflies);
        }
    }

    Ok(RobustPointGroupMatchV1 {
        rank_threshold_lower,
        rank_threshold_upper,
        rank_intervals,
        template_distance_intervals,
        guaranteed_nearest_template,
        not_ruled_out_templates,
        authority: "SPECTRAL_COMPATIBILITY_INTERVAL_ONLY_NOT_A_SYMMETRY_CERTIFICATE",
    })
}

#[cfg(test)]
mod tests {
    use super::super::{SpectrumBlockV1, SymFpProfileV1, analyze_spectra, template_by_name};
    use super::*;

    fn receipt_for_rank_template(name: &str, maximum_degree: usize) -> SymFpReceiptV1 {
        let template = template_by_name(name).unwrap();
        let spectra = (0..=maximum_degree)
            .map(|degree| SpectrumBlockV1 {
                degree,
                singular_values: (0..=2 * degree)
                    .map(|index| {
                        if index < template.fixed_dimensions[degree] as usize {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                svd_sweeps: None,
                svd_residual: None,
            })
            .collect::<Vec<_>>();
        analyze_spectra(&spectra, SymFpProfileV1::paper_degree_9()).unwrap()
    }

    #[test]
    fn zero_perturbation_bounds_recover_the_exact_unique_template() {
        let receipt = receipt_for_rank_template("C2", 9);
        let result = classify_rank_templates_with_perturbation_bounds(
            &receipt,
            &super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1,
            &[0.0; 10],
        )
        .unwrap();
        assert_eq!(result.guaranteed_nearest_template, Some("C2"));
        assert_eq!(result.not_ruled_out_templates, vec!["C2"]);
        assert!(
            result
                .rank_intervals
                .iter()
                .all(|rank| rank.minimum_rank == rank.maximum_rank)
        );
    }

    #[test]
    fn threshold_crossing_returns_a_rank_interval_and_abstains_on_label() {
        let profile = SymFpProfileV1 {
            tau_rel: 0.1,
            ..SymFpProfileV1::default()
        };
        let receipt = analyze_spectra(
            &[
                SpectrumBlockV1 {
                    degree: 0,
                    singular_values: vec![1.0],
                    svd_sweeps: None,
                    svd_residual: None,
                },
                SpectrumBlockV1 {
                    degree: 1,
                    singular_values: vec![1.0, 0.10005, 0.09995],
                    svd_sweeps: None,
                    svd_residual: None,
                },
            ],
            profile,
        )
        .unwrap();
        let result = classify_rank_templates_with_perturbation_bounds(
            &receipt,
            &[
                *template_by_name("C2").unwrap(),
                *template_by_name("Cs").unwrap(),
            ],
            &[0.0, 0.0001],
        )
        .unwrap();
        assert_eq!(result.rank_intervals[1].minimum_rank, 1);
        assert_eq!(result.rank_intervals[1].maximum_rank, 3);
        assert_eq!(result.guaranteed_nearest_template, None);
        assert_eq!(result.not_ruled_out_templates, vec!["C2", "Cs"]);
    }

    #[test]
    fn degree_zero_threshold_uncertainty_propagates_to_every_degree() {
        let profile = SymFpProfileV1 {
            tau_rel: 0.1,
            ..SymFpProfileV1::default()
        };
        let receipt = analyze_spectra(
            &[
                SpectrumBlockV1 {
                    degree: 0,
                    singular_values: vec![1.0],
                    svd_sweeps: None,
                    svd_residual: None,
                },
                SpectrumBlockV1 {
                    degree: 1,
                    singular_values: vec![0.5, 0.2, 0.05],
                    svd_sweeps: None,
                    svd_residual: None,
                },
            ],
            profile,
        )
        .unwrap();
        let result = classify_rank_templates_with_perturbation_bounds(
            &receipt,
            &[*template_by_name("C2").unwrap()],
            &[0.2, 0.0],
        )
        .unwrap();
        assert!((result.rank_threshold_lower - 0.08).abs() < 1.0e-14);
        assert!((result.rank_threshold_upper - 0.12).abs() < 1.0e-14);
        assert_eq!(result.rank_intervals[1].minimum_rank, 2);
        assert_eq!(result.rank_intervals[1].maximum_rank, 2);
        assert_eq!(result.guaranteed_nearest_template, Some("C2"));
    }

    #[test]
    fn invalid_bounds_and_mismatched_receipts_fail_closed() {
        let receipt = receipt_for_rank_template("C2", 9);
        assert_eq!(
            classify_rank_templates_with_perturbation_bounds(
                &receipt,
                &super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1,
                &[-1.0; 10],
            ),
            Err(RobustSymFpErrorV1::InvalidPerturbationBound)
        );
        assert_eq!(
            classify_rank_templates_with_perturbation_bounds(
                &receipt,
                &super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1,
                &[0.0; 9],
            ),
            Err(RobustSymFpErrorV1::BoundCountMismatch)
        );
        let mut forged = receipt;
        forged.rank_sequence[0] += 1;
        assert_eq!(
            classify_rank_templates_with_perturbation_bounds(
                &forged,
                &super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1,
                &[0.0; 10],
            ),
            Err(RobustSymFpErrorV1::InvalidReceipt)
        );
    }

    #[test]
    fn overflowing_perturbation_intervals_are_rejected() {
        let receipt = analyze_spectra(
            &[SpectrumBlockV1 {
                degree: 0,
                singular_values: vec![f64::MAX],
                svd_sweeps: None,
                svd_residual: None,
            }],
            SymFpProfileV1 {
                tau_rel: 0.0,
                ..SymFpProfileV1::default()
            },
        )
        .unwrap();
        assert_eq!(
            classify_rank_templates_with_perturbation_bounds(
                &receipt,
                &[*template_by_name("C1").unwrap()],
                &[f64::MAX],
            ),
            Err(RobustSymFpErrorV1::ArithmeticOverflow)
        );
    }
}
