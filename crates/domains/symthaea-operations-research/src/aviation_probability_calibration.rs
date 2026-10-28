// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative total-variation radius calibration from calibration-partition counts.
//!
//! The confidence bound assumes calibration episodes are independent draws from
//! one fixed finite-support distribution. It does not provide a guarantee under
//! arbitrary temporal dependence, arbitrary future drift, or omitted categories.
//! Callers should calibrate separately by regime/time block when pooling would hide
//! heterogeneity, and should take a documented envelope over regimes if required.

use std::collections::{BTreeMap, BTreeSet};

use crate::aviation_calibration_protocol::EvaluationPartition;

/// Probability scale, in parts per million.
pub const PROBABILITY_SCALE_PPM: u32 = 1_000_000;

/// One modeled scenario category with a frozen nominal probability and observed
/// count from the calibration partition only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationFrequency {
    pub scenario_id: String,
    pub nominal_probability_ppm: u32,
    pub observed_count: u64,
}

/// Errors in the calibration inputs or calculations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadiusCalibrationError {
    EmptyCalibrationSet,
    EmptyScenarioId,
    EmptyRecordId,
    EmptyEventClusterId,
    DuplicateScenarioId { scenario_id: String },
    InvalidNominalProbability { scenario_id: String, probability_ppm: u32 },
    NominalProbabilityMassMismatch { actual_ppm: u64 },
    CountTotalOverflow,
    ZeroCalibrationEpisodes,
    InvalidFailureProbability { failure_probability_ppm: u32 },
    EmptyRegimeSet,
    EmptyRegimeId,
    DuplicateRegimeId { regime_id: String },
    FailureProbabilityTooSmallForRegimeCount { failure_probability_ppm: u32, regime_count: usize },
    NonCalibrationPartition { record_id: String, partition: EvaluationPartition },
    DuplicateRecordId { record_id: String },
    DuplicateEventClusterId { event_cluster_id: String },
    UnknownRegime { regime_id: String },
    EmptyNominalDistribution { regime_id: String },
    DuplicateNominalRegime { regime_id: String },
    DuplicateNominalScenarioId { regime_id: String, scenario_id: String },
    InvalidNominalScenarioProbability { regime_id: String, scenario_id: String, probability_ppm: u32 },
    NominalRegimeProbabilityMassMismatch { regime_id: String, actual_ppm: u64 },
    ObservedScenarioOutsideNominalSupport { regime_id: String, scenario_id: String },
    RegimeHasNoCalibrationEpisodes { regime_id: String },
    ArithmeticOverflow,
}

/// Auditable parts of the conservative radius calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct TvRadiusCalibrationReport {
    pub scenario_count: usize,
    pub calibration_episode_count: u64,
    /// Requested failure probability, delta, in ppm. Confidence is 1-delta.
    pub failure_probability_ppm: u32,
    /// Exact empirical TV distance, rounded upward to ppm.
    pub empirical_tv_distance_ppm: u32,
    /// Union-Hoeffding allowance on TV error, rounded upward to ppm.
    pub sampling_allowance_ppm: u32,
    /// Conservative triangle-inequality bound, capped at TV distance 1.
    pub required_tv_radius_ppm: u32,
    pub capped_at_maximum_radius: bool,
    /// Claim is valid only under the documented independent fixed-distribution assumption.
    pub assumption: &'static str,
}

/// Calibrate a conservative TV ambiguity radius from a fixed nominal distribution
/// and observed calibration counts.
///
/// For K categories and n independent calibration observations, Hoeffding plus a
/// union bound gives each category-probability error at most
/// sqrt(ln(2K/delta)/(2n)) with probability at least 1-delta. The resulting
/// TV-error allowance is K/2 times that coordinate bound. By the triangle
/// inequality, TV(nominal, true) is bounded by TV(nominal, empirical) plus this
/// sampling allowance. The radius is rounded upward to integer ppm and capped at 1.
///
/// This is deliberately conservative and may saturate at 100% for small n or large
/// K. It is a baseline calibration method, not an efficient minimax-optimal bound.
pub fn calibrate_tv_radius(
    frequencies: &[CalibrationFrequency],
    failure_probability_ppm: u32,
) -> Result<TvRadiusCalibrationReport, RadiusCalibrationError> {
    if frequencies.is_empty() {
        return Err(RadiusCalibrationError::EmptyCalibrationSet);
    }
    if failure_probability_ppm == 0 || failure_probability_ppm >= PROBABILITY_SCALE_PPM {
        return Err(RadiusCalibrationError::InvalidFailureProbability {
            failure_probability_ppm,
        });
    }

    let mut ids = BTreeSet::new();
    let mut probability_sum = 0_u64;
    let mut count_total = 0_u64;
    let mut empirical_distance_numerator = 0_i128;

    for item in frequencies {
        if item.scenario_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyScenarioId);
        }
        if !ids.insert(item.scenario_id.as_str()) {
            return Err(RadiusCalibrationError::DuplicateScenarioId {
                scenario_id: item.scenario_id.clone(),
            });
        }
        if item.nominal_probability_ppm > PROBABILITY_SCALE_PPM {
            return Err(RadiusCalibrationError::InvalidNominalProbability {
                scenario_id: item.scenario_id.clone(),
                probability_ppm: item.nominal_probability_ppm,
            });
        }

        probability_sum += u64::from(item.nominal_probability_ppm);
        count_total = count_total
            .checked_add(item.observed_count)
            .ok_or(RadiusCalibrationError::CountTotalOverflow)?;
    }

    if probability_sum != u64::from(PROBABILITY_SCALE_PPM) {
        return Err(RadiusCalibrationError::NominalProbabilityMassMismatch {
            actual_ppm: probability_sum,
        });
    }
    if count_total == 0 {
        return Err(RadiusCalibrationError::ZeroCalibrationEpisodes);
    }

    let n = i128::from(count_total);
    for item in frequencies {
        // Difference in ppm*n between nominal mass and observed empirical mass.
        let nominal_mass = i128::from(item.nominal_probability_ppm)
            .checked_mul(n)
            .ok_or(RadiusCalibrationError::ArithmeticOverflow)?;
        let empirical_mass = i128::from(item.observed_count)
            .checked_mul(i128::from(PROBABILITY_SCALE_PPM))
            .ok_or(RadiusCalibrationError::ArithmeticOverflow)?;
        let difference = (nominal_mass - empirical_mass).abs();
        empirical_distance_numerator = empirical_distance_numerator
            .checked_add(difference)
            .ok_or(RadiusCalibrationError::ArithmeticOverflow)?;
    }

    // TV distance in ppm = sum(|p_i*n - count_i*1e6|) / (2*n).
    let empirical_denominator = n
        .checked_mul(2)
        .ok_or(RadiusCalibrationError::ArithmeticOverflow)?;
    let empirical_tv_ppm = div_ceil_nonnegative(
        empirical_distance_numerator,
        empirical_denominator,
    )?;
    let empirical_tv_ppm = empirical_tv_ppm.min(i128::from(PROBABILITY_SCALE_PPM)) as u32;

    let k = frequencies.len() as f64;
    let delta = f64::from(failure_probability_ppm) / f64::from(PROBABILITY_SCALE_PPM);
    let coordinate_bound = ((2.0 * k / delta).ln() / (2.0 * count_total as f64)).sqrt();
    let tv_allowance = (0.5 * k * coordinate_bound).min(1.0);
    let sampling_allowance_ppm =
        (tv_allowance * f64::from(PROBABILITY_SCALE_PPM)).ceil() as u32;
    let unsaturated = u64::from(empirical_tv_ppm) + u64::from(sampling_allowance_ppm);
    let required_tv_radius_ppm = unsaturated.min(u64::from(PROBABILITY_SCALE_PPM)) as u32;

    Ok(TvRadiusCalibrationReport {
        scenario_count: frequencies.len(),
        calibration_episode_count: count_total,
        failure_probability_ppm,
        empirical_tv_distance_ppm: empirical_tv_ppm,
        sampling_allowance_ppm,
        required_tv_radius_ppm,
        capped_at_maximum_radius: unsaturated > u64::from(PROBABILITY_SCALE_PPM),
        assumption: "independent calibration episodes from one fixed finite-support distribution",
    })
}

fn div_ceil_nonnegative(
    numerator: i128,
    denominator: i128,
) -> Result<i128, RadiusCalibrationError> {
    if numerator < 0 || denominator <= 0 {
        return Err(RadiusCalibrationError::ArithmeticOverflow);
    }
    let adjusted = numerator
        .checked_add(denominator - 1)
        .ok_or(RadiusCalibrationError::ArithmeticOverflow)?;
    Ok(adjusted / denominator)
}


/// One independently counted event episode from a declared data partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationEpisode {
    pub record_id: String,
    pub event_cluster_id: String,
    pub regime_id: String,
    pub observed_scenario_id: String,
    pub partition: EvaluationPartition,
}

/// One category's frozen nominal probability within a regime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NominalScenarioProbability {
    pub scenario_id: String,
    pub probability_ppm: u32,
}

/// Frozen nominal scenario distribution for one regime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegimeNominalDistribution {
    pub regime_id: String,
    pub probabilities: Vec<NominalScenarioProbability>,
}

/// Build regime-specific frequency counts from event-level Calibration records,
/// then calibrate the simultaneous maximum radius across regimes.
///
/// This is the preferred path over hand-entered aggregate counts when episode-level
/// provenance is available. Each event cluster must appear exactly once and every
/// row must belong to Calibration; train/validation/test rows cannot influence the
/// estimated radius. The source/event labels themselves are not authenticated.
pub fn calibrate_tv_radius_from_episodes(
    episodes: &[CalibrationEpisode],
    nominal_distributions: &[RegimeNominalDistribution],
    family_failure_probability_ppm: u32,
) -> Result<TvRadiusEnvelopeReport, RadiusCalibrationError> {
    if episodes.is_empty() {
        return Err(RadiusCalibrationError::ZeroCalibrationEpisodes);
    }
    if nominal_distributions.is_empty() {
        return Err(RadiusCalibrationError::EmptyRegimeSet);
    }

    let mut seen_records = BTreeSet::new();
    let mut seen_clusters = BTreeSet::new();
    let mut regime_episode_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut observed_counts: BTreeMap<(String, String), u64> = BTreeMap::new();

    let mut nominal_by_regime: BTreeMap<String, &RegimeNominalDistribution> = BTreeMap::new();
    for distribution in nominal_distributions {
        if distribution.regime_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyRegimeId);
        }
        if distribution.probabilities.is_empty() {
            return Err(RadiusCalibrationError::EmptyNominalDistribution {
                regime_id: distribution.regime_id.clone(),
            });
        }
        if nominal_by_regime
            .insert(distribution.regime_id.clone(), distribution)
            .is_some()
        {
            return Err(RadiusCalibrationError::DuplicateNominalRegime {
                regime_id: distribution.regime_id.clone(),
            });
        }

        let mut category_ids = BTreeSet::new();
        let mut sum = 0_u64;
        for probability in &distribution.probabilities {
            if probability.scenario_id.trim().is_empty() {
                return Err(RadiusCalibrationError::EmptyScenarioId);
            }
            if !category_ids.insert(probability.scenario_id.as_str()) {
                return Err(RadiusCalibrationError::DuplicateNominalScenarioId {
                    regime_id: distribution.regime_id.clone(),
                    scenario_id: probability.scenario_id.clone(),
                });
            }
            if probability.probability_ppm > PROBABILITY_SCALE_PPM {
                return Err(RadiusCalibrationError::InvalidNominalScenarioProbability {
                    regime_id: distribution.regime_id.clone(),
                    scenario_id: probability.scenario_id.clone(),
                    probability_ppm: probability.probability_ppm,
                });
            }
            sum += u64::from(probability.probability_ppm);
        }
        if sum != u64::from(PROBABILITY_SCALE_PPM) {
            return Err(RadiusCalibrationError::NominalRegimeProbabilityMassMismatch {
                regime_id: distribution.regime_id.clone(),
                actual_ppm: sum,
            });
        }
    }

    for episode in episodes {
        if episode.partition != EvaluationPartition::Calibration {
            return Err(RadiusCalibrationError::NonCalibrationPartition {
                record_id: episode.record_id.clone(),
                partition: episode.partition,
            });
        }
        if episode.record_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyRecordId);
        }
        if !seen_records.insert(episode.record_id.as_str()) {
            return Err(RadiusCalibrationError::DuplicateRecordId {
                record_id: episode.record_id.clone(),
            });
        }
        if episode.event_cluster_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyEventClusterId);
        }
        if !seen_clusters.insert(episode.event_cluster_id.as_str()) {
            return Err(RadiusCalibrationError::DuplicateEventClusterId {
                event_cluster_id: episode.event_cluster_id.clone(),
            });
        }
        if episode.regime_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyRegimeId);
        }
        if episode.observed_scenario_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyScenarioId);
        }

        let distribution = nominal_by_regime
            .get(&episode.regime_id)
            .ok_or_else(|| RadiusCalibrationError::UnknownRegime {
                regime_id: episode.regime_id.clone(),
            })?;
        if !distribution
            .probabilities
            .iter()
            .any(|p| p.scenario_id == episode.observed_scenario_id)
        {
            return Err(RadiusCalibrationError::ObservedScenarioOutsideNominalSupport {
                regime_id: episode.regime_id.clone(),
                scenario_id: episode.observed_scenario_id.clone(),
            });
        }

        let regime_count = regime_episode_counts.entry(episode.regime_id.clone()).or_default();
        *regime_count = regime_count
            .checked_add(1)
            .ok_or(RadiusCalibrationError::CountTotalOverflow)?;
        let scenario_count = observed_counts
            .entry((episode.regime_id.clone(), episode.observed_scenario_id.clone()))
            .or_default();
        *scenario_count = scenario_count
            .checked_add(1)
            .ok_or(RadiusCalibrationError::CountTotalOverflow)?;
    }

    let mut regimes = Vec::with_capacity(nominal_distributions.len());
    for (regime_id, distribution) in nominal_by_regime {
        if !regime_episode_counts.contains_key(&regime_id) {
            return Err(RadiusCalibrationError::RegimeHasNoCalibrationEpisodes {
                regime_id,
            });
        }

        let frequencies = distribution
            .probabilities
            .iter()
            .map(|probability| CalibrationFrequency {
                scenario_id: probability.scenario_id.clone(),
                nominal_probability_ppm: probability.probability_ppm,
                observed_count: observed_counts
                    .get(&(distribution.regime_id.clone(), probability.scenario_id.clone()))
                    .copied()
                    .unwrap_or(0),
            })
            .collect();
        regimes.push(RegimeCalibration { regime_id, frequencies });
    }

    calibrate_tv_radius_envelope(&regimes, family_failure_probability_ppm)
}

/// A regime-specific calibration set, e.g. a weather/airport cohort or season.
/// Frequencies must count complete event episodes, not duplicated flight rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegimeCalibration {
    pub regime_id: String,
    pub frequencies: Vec<CalibrationFrequency>,
}

/// Calibration result for one declared regime.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeRadiusCalibration {
    pub regime_id: String,
    pub report: TvRadiusCalibrationReport,
}

/// A simultaneous conservative radius envelope over several regimes.
#[derive(Debug, Clone, PartialEq)]
pub struct TvRadiusEnvelopeReport {
    pub regime_count: usize,
    pub requested_family_failure_probability_ppm: u32,
    /// Integer Bonferroni allocation; each regime receives this delta, so total
    /// allocated failure probability is no greater than the requested delta.
    pub per_regime_failure_probability_ppm: u32,
    /// Maximum regime-specific radius; use this as a single radius if one global
    /// ambiguity radius must cover every supplied regime.
    pub maximum_required_tv_radius_ppm: u32,
    pub any_regime_capped_at_maximum_radius: bool,
    /// Sorted by regime ID for deterministic manifests.
    pub regimes: Vec<RegimeRadiusCalibration>,
}

/// Calibrate a common TV radius over several heterogeneous calibration regimes.
///
/// The overall failure probability is divided by the regime count using integer
/// floor division (Bonferroni); any remainder is left unused to preserve the
/// family-wise bound. The returned common radius is the maximum of per-regime
/// radii. This does not cover arbitrary unseen regimes or omitted categories.
/// Each regime still requires independent samples from one fixed distribution.
pub fn calibrate_tv_radius_envelope(
    regimes: &[RegimeCalibration],
    family_failure_probability_ppm: u32,
) -> Result<TvRadiusEnvelopeReport, RadiusCalibrationError> {
    if regimes.is_empty() {
        return Err(RadiusCalibrationError::EmptyRegimeSet);
    }
    if family_failure_probability_ppm == 0
        || family_failure_probability_ppm >= PROBABILITY_SCALE_PPM
    {
        return Err(RadiusCalibrationError::InvalidFailureProbability {
            failure_probability_ppm: family_failure_probability_ppm,
        });
    }

    let regime_count = regimes.len();
    let per_regime_delta = family_failure_probability_ppm / regime_count as u32;
    if per_regime_delta == 0 {
        return Err(RadiusCalibrationError::FailureProbabilityTooSmallForRegimeCount {
            failure_probability_ppm: family_failure_probability_ppm,
            regime_count,
        });
    }

    let mut ids = BTreeSet::new();
    for regime in regimes {
        if regime.regime_id.trim().is_empty() {
            return Err(RadiusCalibrationError::EmptyRegimeId);
        }
        if !ids.insert(regime.regime_id.as_str()) {
            return Err(RadiusCalibrationError::DuplicateRegimeId {
                regime_id: regime.regime_id.clone(),
            });
        }
    }

    let mut ordered: Vec<&RegimeCalibration> = regimes.iter().collect();
    ordered.sort_by(|a, b| a.regime_id.cmp(&b.regime_id));

    let mut calibrated = Vec::with_capacity(regime_count);
    let mut maximum_required_tv_radius_ppm = 0;
    let mut any_regime_capped_at_maximum_radius = false;

    for regime in ordered {
        let report = calibrate_tv_radius(&regime.frequencies, per_regime_delta)?;
        maximum_required_tv_radius_ppm =
            maximum_required_tv_radius_ppm.max(report.required_tv_radius_ppm);
        any_regime_capped_at_maximum_radius |= report.capped_at_maximum_radius;
        calibrated.push(RegimeRadiusCalibration {
            regime_id: regime.regime_id.clone(),
            report,
        });
    }

    Ok(TvRadiusEnvelopeReport {
        regime_count,
        requested_family_failure_probability_ppm: family_failure_probability_ppm,
        per_regime_failure_probability_ppm: per_regime_delta,
        maximum_required_tv_radius_ppm,
        any_regime_capped_at_maximum_radius,
        regimes: calibrated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frequencies() -> Vec<CalibrationFrequency> {
        vec![
            CalibrationFrequency {
                scenario_id: "routine".into(),
                nominal_probability_ppm: 600_000,
                observed_count: 60,
            },
            CalibrationFrequency {
                scenario_id: "capacity".into(),
                nominal_probability_ppm: 300_000,
                observed_count: 30,
            },
            CalibrationFrequency {
                scenario_id: "weather".into(),
                nominal_probability_ppm: 100_000,
                observed_count: 10,
            },
        ]
    }

    #[test]
    fn exact_empirical_match_still_has_nonzero_sampling_allowance() {
        let report = calibrate_tv_radius(&frequencies(), 50_000).unwrap();
        assert_eq!(report.empirical_tv_distance_ppm, 0);
        assert!(report.sampling_allowance_ppm > 0);
        assert_eq!(report.required_tv_radius_ppm, report.sampling_allowance_ppm);
        assert_eq!(report.calibration_episode_count, 100);
    }

    #[test]
    fn observed_distribution_shift_increases_required_radius() {
        let nominal = calibrate_tv_radius(&frequencies(), 50_000).unwrap();
        let shifted = vec![
            CalibrationFrequency { scenario_id: "routine".into(), nominal_probability_ppm: 600_000, observed_count: 40 },
            CalibrationFrequency { scenario_id: "capacity".into(), nominal_probability_ppm: 300_000, observed_count: 30 },
            CalibrationFrequency { scenario_id: "weather".into(), nominal_probability_ppm: 100_000, observed_count: 30 },
        ];
        let shifted_report = calibrate_tv_radius(&shifted, 50_000).unwrap();
        assert!(shifted_report.empirical_tv_distance_ppm > nominal.empirical_tv_distance_ppm);
        assert!(shifted_report.required_tv_radius_ppm >= shifted_report.empirical_tv_distance_ppm);
    }

    #[test]
    fn lower_failure_probability_never_reduces_conservative_radius() {
        let high_confidence = calibrate_tv_radius(&frequencies(), 10_000).unwrap();
        let lower_confidence = calibrate_tv_radius(&frequencies(), 100_000).unwrap();
        assert!(high_confidence.required_tv_radius_ppm >= lower_confidence.required_tv_radius_ppm);
    }

    #[test]
    fn more_calibration_episodes_do_not_increase_sampling_allowance() {
        let small = frequencies();
        let mut large = frequencies();
        for item in &mut large {
            item.observed_count *= 100;
        }
        let a = calibrate_tv_radius(&small, 50_000).unwrap();
        let b = calibrate_tv_radius(&large, 50_000).unwrap();
        assert!(b.sampling_allowance_ppm <= a.sampling_allowance_ppm);
        assert_eq!(a.empirical_tv_distance_ppm, b.empirical_tv_distance_ppm);
    }

    #[test]
    fn zero_observations_in_a_category_are_supported() {
        let items = vec![
            CalibrationFrequency { scenario_id: "routine".into(), nominal_probability_ppm: 900_000, observed_count: 10 },
            CalibrationFrequency { scenario_id: "rare".into(), nominal_probability_ppm: 100_000, observed_count: 0 },
        ];
        let report = calibrate_tv_radius(&items, 50_000).unwrap();
        assert!(report.empirical_tv_distance_ppm > 0);
    }

    #[test]
    fn probability_mass_must_sum_exactly_to_one_million() {
        let mut bad = frequencies();
        bad[0].nominal_probability_ppm -= 1;
        assert_eq!(
            calibrate_tv_radius(&bad, 50_000),
            Err(RadiusCalibrationError::NominalProbabilityMassMismatch { actual_ppm: 999_999 })
        );
    }

    #[test]
    fn duplicate_and_empty_scenario_ids_are_rejected() {
        let mut bad = frequencies();
        bad[1].scenario_id = bad[0].scenario_id.clone();
        assert!(matches!(calibrate_tv_radius(&bad, 50_000), Err(RadiusCalibrationError::DuplicateScenarioId { .. })));
        bad[1].scenario_id.clear();
        assert_eq!(calibrate_tv_radius(&bad, 50_000), Err(RadiusCalibrationError::EmptyScenarioId));
    }

    #[test]
    fn zero_sample_and_empty_universe_are_rejected() {
        assert_eq!(calibrate_tv_radius(&[], 50_000), Err(RadiusCalibrationError::EmptyCalibrationSet));
        let no_observations = frequencies().into_iter().map(|mut x| { x.observed_count = 0; x }).collect::<Vec<_>>();
        assert_eq!(calibrate_tv_radius(&no_observations, 50_000), Err(RadiusCalibrationError::ZeroCalibrationEpisodes));
    }

    #[test]
    fn failure_probability_must_be_strictly_between_zero_and_one() {
        for delta in [0, 1_000_000] {
            assert!(matches!(calibrate_tv_radius(&frequencies(), delta), Err(RadiusCalibrationError::InvalidFailureProbability { .. })));
        }
    }

    #[test]
    fn radius_is_capped_at_one_for_small_sample_or_large_support() {
        let items = vec![
            CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 500_000, observed_count: 1 },
            CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 500_000, observed_count: 0 },
        ];
        let report = calibrate_tv_radius(&items, 50_000).unwrap();
        assert_eq!(report.required_tv_radius_ppm, PROBABILITY_SCALE_PPM);
        assert!(report.capped_at_maximum_radius);
    }

    #[test]
    fn nominal_scenario_universe_may_not_be_missing() {
        let bad = vec![
            CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 700_000, observed_count: 10 },
            CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 200_000, observed_count: 10 },
        ];
        assert!(matches!(calibrate_tv_radius(&bad, 50_000), Err(RadiusCalibrationError::NominalProbabilityMassMismatch { .. })));
    }

    #[test]
    fn row_order_does_not_change_calibration_result() {
        let base = frequencies();
        let mut reversed = base.clone();
        reversed.reverse();
        assert_eq!(
            calibrate_tv_radius(&base, 50_000).unwrap(),
            calibrate_tv_radius(&reversed, 50_000).unwrap()
        );
    }

    #[test]
    fn confidence_allowance_is_bounded_by_one() {
        let report = calibrate_tv_radius(&frequencies(), 1).unwrap();
        assert!(report.sampling_allowance_ppm <= PROBABILITY_SCALE_PPM);
    }

    #[test]
    fn empirical_tv_distance_is_rounded_up_conservatively() {
        let items = vec![
            CalibrationFrequency { scenario_id: "a".into(), nominal_probability_ppm: 666_667, observed_count: 2 },
            CalibrationFrequency { scenario_id: "b".into(), nominal_probability_ppm: 333_333, observed_count: 1 },
        ];
        let report = calibrate_tv_radius(&items, 50_000).unwrap();
        assert_eq!(report.empirical_tv_distance_ppm, 1);
    }

    fn regime_fixture(id: &str, counts: [u64; 3]) -> RegimeCalibration {
        RegimeCalibration {
            regime_id: id.into(),
            frequencies: vec![
                CalibrationFrequency { scenario_id: "routine".into(), nominal_probability_ppm: 600_000, observed_count: counts[0] },
                CalibrationFrequency { scenario_id: "capacity".into(), nominal_probability_ppm: 300_000, observed_count: counts[1] },
                CalibrationFrequency { scenario_id: "weather".into(), nominal_probability_ppm: 100_000, observed_count: counts[2] },
            ],
        }
    }

    #[test]
    fn regime_envelope_uses_maximum_radius_with_bonferroni_budget() {
        let regimes = [
            regime_fixture("calm", [60, 30, 10]),
            regime_fixture("disrupted", [40, 30, 30]),
        ];
        let report = calibrate_tv_radius_envelope(&regimes, 50_000).unwrap();
        assert_eq!(report.regime_count, 2);
        assert_eq!(report.per_regime_failure_probability_ppm, 25_000);
        assert_eq!(report.regimes[0].regime_id, "calm");
        assert_eq!(report.regimes[1].regime_id, "disrupted");
        assert_eq!(
            report.maximum_required_tv_radius_ppm,
            report.regimes.iter().map(|r| r.report.required_tv_radius_ppm).max().unwrap()
        );
        assert!(
            report.regimes[1].report.empirical_tv_distance_ppm
                > report.regimes[0].report.empirical_tv_distance_ppm
        );
    }

    #[test]
    fn regime_envelope_is_order_invariant() {
        let a = vec![regime_fixture("winter", [40, 30, 30]), regime_fixture("summer", [60, 30, 10])];
        let mut b = a.clone();
        b.reverse();
        assert_eq!(
            calibrate_tv_radius_envelope(&a, 50_000).unwrap(),
            calibrate_tv_radius_envelope(&b, 50_000).unwrap()
        );
    }

    #[test]
    fn regime_envelope_rejects_empty_duplicate_and_blank_regime_ids() {
        assert_eq!(
            calibrate_tv_radius_envelope(&[], 50_000),
            Err(RadiusCalibrationError::EmptyRegimeSet)
        );
        let duplicate = vec![regime_fixture("same", [60, 30, 10]), regime_fixture("same", [60, 30, 10])];
        assert!(matches!(
            calibrate_tv_radius_envelope(&duplicate, 50_000),
            Err(RadiusCalibrationError::DuplicateRegimeId { .. })
        ));
        let blank = vec![regime_fixture(" ", [60, 30, 10])];
        assert_eq!(
            calibrate_tv_radius_envelope(&blank, 50_000),
            Err(RadiusCalibrationError::EmptyRegimeId)
        );
    }

    #[test]
    fn regime_envelope_fails_if_family_delta_cannot_be_allocated() {
        let regimes = vec![
            regime_fixture("a", [60, 30, 10]),
            regime_fixture("b", [60, 30, 10]),
        ];
        assert!(matches!(
            calibrate_tv_radius_envelope(&regimes, 1),
            Err(RadiusCalibrationError::FailureProbabilityTooSmallForRegimeCount { .. })
        ));
    }

    #[test]
    fn envelope_propagates_invalid_regime_frequency_data() {
        let mut bad = regime_fixture("bad", [60, 30, 10]);
        bad.frequencies[0].nominal_probability_ppm = 1;
        assert!(matches!(
            calibrate_tv_radius_envelope(&[bad], 50_000),
            Err(RadiusCalibrationError::NominalProbabilityMassMismatch { .. })
        ));
    }



    fn nominal_distribution(regime_id: &str) -> RegimeNominalDistribution {
        RegimeNominalDistribution {
            regime_id: regime_id.into(),
            probabilities: vec![
                NominalScenarioProbability { scenario_id: "routine".into(), probability_ppm: 600_000 },
                NominalScenarioProbability { scenario_id: "capacity".into(), probability_ppm: 300_000 },
                NominalScenarioProbability { scenario_id: "weather".into(), probability_ppm: 100_000 },
            ],
        }
    }

    fn episode(id: &str, regime: &str, scenario: &str) -> CalibrationEpisode {
        CalibrationEpisode {
            record_id: format!("record-{id}"),
            event_cluster_id: format!("cluster-{id}"),
            regime_id: regime.into(),
            observed_scenario_id: scenario.into(),
            partition: EvaluationPartition::Calibration,
        }
    }

    #[test]
    fn event_level_calibration_counts_observations_and_builds_regime_envelope() {
        let mut events = Vec::new();
        for (index, scenario) in ["routine", "routine", "capacity", "weather"].into_iter().enumerate() {
            events.push(episode(&index.to_string(), "airport-a", scenario));
        }
        for (index, scenario) in ["routine", "capacity", "weather", "weather"].into_iter().enumerate() {
            events.push(episode(&format!("b-{index}"), "airport-b", scenario));
        }
        let report = calibrate_tv_radius_from_episodes(
            &events,
            &[nominal_distribution("airport-a"), nominal_distribution("airport-b")],
            50_000,
        ).unwrap();
        assert_eq!(report.regime_count, 2);
        assert_eq!(report.per_regime_failure_probability_ppm, 25_000);
        assert_eq!(report.regimes[0].report.calibration_episode_count, 4);
        assert_eq!(report.regimes[1].report.calibration_episode_count, 4);
        assert!(report.regimes[0].report.empirical_tv_distance_ppm > 0);
    }

    #[test]
    fn event_level_calibration_rejects_test_or_validation_rows() {
        for partition in [EvaluationPartition::Training, EvaluationPartition::Validation, EvaluationPartition::Test] {
            let mut event = episode("x", "airport-a", "routine");
            event.partition = partition;
            assert!(matches!(
                calibrate_tv_radius_from_episodes(&[event], &[nominal_distribution("airport-a")], 50_000),
                Err(RadiusCalibrationError::NonCalibrationPartition { .. })
            ));
        }
    }

    #[test]
    fn event_level_calibration_rejects_duplicate_record_and_cluster_ids() {
        let a = episode("a", "airport-a", "routine");
        let mut duplicate_record = episode("b", "airport-a", "capacity");
        duplicate_record.record_id = a.record_id.clone();
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[a.clone(), duplicate_record], &[nominal_distribution("airport-a")], 50_000),
            Err(RadiusCalibrationError::DuplicateRecordId { .. })
        ));

        let mut duplicate_cluster = episode("b", "airport-a", "capacity");
        duplicate_cluster.event_cluster_id = a.event_cluster_id.clone();
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[a, duplicate_cluster], &[nominal_distribution("airport-a")], 50_000),
            Err(RadiusCalibrationError::DuplicateEventClusterId { .. })
        ));
    }

    #[test]
    fn event_level_calibration_rejects_unknown_regime_or_scenario() {
        let unknown_regime = episode("a", "not-declared", "routine");
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[unknown_regime], &[nominal_distribution("airport-a")], 50_000),
            Err(RadiusCalibrationError::UnknownRegime { .. })
        ));
        let unknown_scenario = episode("b", "airport-a", "unexpected");
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[unknown_scenario], &[nominal_distribution("airport-a")], 50_000),
            Err(RadiusCalibrationError::ObservedScenarioOutsideNominalSupport { .. })
        ));
    }

    #[test]
    fn every_declared_regime_requires_calibration_evidence() {
        assert!(matches!(
            calibrate_tv_radius_from_episodes(
                &[episode("a", "airport-a", "routine")],
                &[nominal_distribution("airport-a"), nominal_distribution("airport-b")],
                50_000
            ),
            Err(RadiusCalibrationError::RegimeHasNoCalibrationEpisodes { .. })
        ));
    }

    #[test]
    fn nominal_regime_support_and_mass_are_validated() {
        let duplicate = RegimeNominalDistribution {
            regime_id: "airport-a".into(),
            probabilities: vec![
                NominalScenarioProbability { scenario_id: "x".into(), probability_ppm: 500_000 },
                NominalScenarioProbability { scenario_id: "x".into(), probability_ppm: 500_000 },
            ],
        };
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[episode("a", "airport-a", "x")], &[duplicate], 50_000),
            Err(RadiusCalibrationError::DuplicateNominalScenarioId { .. })
        ));
        let bad_mass = RegimeNominalDistribution {
            regime_id: "airport-a".into(),
            probabilities: vec![NominalScenarioProbability { scenario_id: "x".into(), probability_ppm: 900_000 }],
        };
        assert!(matches!(
            calibrate_tv_radius_from_episodes(&[episode("b", "airport-a", "x")], &[bad_mass], 50_000),
            Err(RadiusCalibrationError::NominalRegimeProbabilityMassMismatch { .. })
        ));
    }

}
