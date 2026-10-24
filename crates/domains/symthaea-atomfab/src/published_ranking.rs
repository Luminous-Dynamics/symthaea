//! Paper-compatible ranking metrics for matched atomistic candidate structures.
//!
//! This module implements extrapolated rank-biased overlap (RBO) and
//! Kendall's tau-b over the same canonical candidate universe. RBO requires a
//! total order; exact energy ties are broken by the frozen canonical structure
//! ID. Kendall's tau-b uses a preregistered energy tie tolerance and preserves
//! tie counts rather than turning them into arbitrary concordant/discordant
//! pairs.
//!
//! Discovery-set overlap belongs in a separate metric: these metrics compare
//! rankings only after both lists contain exactly the same structure IDs.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const PUBLISHED_RANKING_SCHEMA_VERSION: u16 = 1;

/// One candidate's energy under one evaluator, in the declared common unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankedEnergy {
    pub structure_id: String,
    pub energy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedRankingMetrics {
    pub schema_version: u16,
    pub candidate_count: usize,
    pub rbo_persistence: f64,
    /// Finite-list extrapolated RBO: (1-p) sum p^(d-1) A_d + p^k A_k.
    pub extrapolated_rbo: f64,
    /// Kendall's tau-b over paired energies, using energy_tie_tolerance.
    pub kendall_tau_b: f64,
    pub energy_tie_tolerance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishedRankingError {
    EmptyInput,
    LengthMismatch,
    EmptyStructureId { index: usize },
    DuplicateStructureId { side: &'static str, index: usize },
    MismatchedCandidateUniverse,
    NonFiniteEnergy { index: usize },
    InvalidPersistence,
    InvalidTieTolerance,
    NoComparablePairs,
    InvalidMetric,
    UnsupportedSchemaVersion(u16),
}

impl fmt::Display for PublishedRankingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "published ranking comparison needs candidates"),
            Self::LengthMismatch => write!(f, "prediction and reference ranking lengths differ"),
            Self::EmptyStructureId { index } => {
                write!(f, "candidate {index} has an empty canonical structure ID")
            }
            Self::DuplicateStructureId { side, index } => {
                write!(
                    f,
                    "{side} ranking has a duplicate structure ID at index {index}"
                )
            }
            Self::MismatchedCandidateUniverse => write!(
                f,
                "published ranking metrics require the exact same canonical candidate IDs"
            ),
            Self::NonFiniteEnergy { index } => {
                write!(f, "candidate {index} has a non-finite energy")
            }
            Self::InvalidPersistence => {
                write!(
                    f,
                    "RBO persistence must be finite and strictly between zero and one"
                )
            }
            Self::InvalidTieTolerance => {
                write!(f, "energy tie tolerance must be finite and non-negative")
            }
            Self::NoComparablePairs => {
                write!(
                    f,
                    "Kendall tau-b is undefined because no non-tied pairs are comparable"
                )
            }
            Self::InvalidMetric => {
                write!(f, "computed ranking metric is not finite or out of range")
            }
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported published-ranking schema version: {version}")
            }
        }
    }
}

impl Error for PublishedRankingError {}

impl PublishedRankingMetrics {
    /// Validate the version and metric ranges of an imported report.
    ///
    /// This checks internal plausibility only. Evidence provenance and the
    /// canonical prediction/reference inputs must be checked by the caller.
    pub fn validate(&self) -> Result<(), PublishedRankingError> {
        if self.schema_version != PUBLISHED_RANKING_SCHEMA_VERSION {
            return Err(PublishedRankingError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.candidate_count == 0
            || !self.rbo_persistence.is_finite()
            || self.rbo_persistence <= 0.0
            || self.rbo_persistence >= 1.0
            || !self.extrapolated_rbo.is_finite()
            || !(0.0..=1.0).contains(&self.extrapolated_rbo)
            || !self.kendall_tau_b.is_finite()
            || !(-1.0..=1.0).contains(&self.kendall_tau_b)
            || !self.energy_tie_tolerance.is_finite()
            || self.energy_tie_tolerance < 0.0
        {
            return Err(PublishedRankingError::InvalidMetric);
        }
        Ok(())
    }
}

/// Compute extrapolated RBO and Kendall tau-b over matched structure identities.
///
/// Energies are ordered low-to-high (minimum first). For RBO, exact energy ties
/// break by canonical ID to provide a deterministic strict ranking. Kendall
/// tau-b uses the frozen tolerance so scientifically indistinguishable energies
/// can remain ties without arbitrary tie-breaking influencing the correlation.
pub fn compare_published_rankings(
    predicted: &[RankedEnergy],
    reference: &[RankedEnergy],
    persistence: f64,
    tie_tolerance: f64,
) -> Result<PublishedRankingMetrics, PublishedRankingError> {
    validate_side(predicted, "prediction")?;
    validate_side(reference, "reference")?;
    if predicted.is_empty() {
        return Err(PublishedRankingError::EmptyInput);
    }
    if predicted.len() != reference.len() {
        return Err(PublishedRankingError::LengthMismatch);
    }
    if !persistence.is_finite() || persistence <= 0.0 || persistence >= 1.0 {
        return Err(PublishedRankingError::InvalidPersistence);
    }
    if !tie_tolerance.is_finite() || tie_tolerance < 0.0 {
        return Err(PublishedRankingError::InvalidTieTolerance);
    }

    let predicted_ids: BTreeSet<&str> = predicted
        .iter()
        .map(|item| item.structure_id.as_str())
        .collect();
    let reference_ids: BTreeSet<&str> = reference
        .iter()
        .map(|item| item.structure_id.as_str())
        .collect();
    if predicted_ids != reference_ids {
        return Err(PublishedRankingError::MismatchedCandidateUniverse);
    }

    let mut predicted_order: Vec<&RankedEnergy> = predicted.iter().collect();
    predicted_order.sort_by(|a, b| compare_energy_order(a, b));
    let mut reference_order: Vec<&RankedEnergy> = reference.iter().collect();
    reference_order.sort_by(|a, b| compare_energy_order(a, b));

    let raw_extrapolated_rbo =
        compute_extrapolated_rbo(&predicted_order, &reference_order, persistence);
    if !raw_extrapolated_rbo.is_finite() || !(-1e-12..=1.0 + 1e-12).contains(&raw_extrapolated_rbo)
    {
        return Err(PublishedRankingError::InvalidMetric);
    }
    // RBO is mathematically bounded by [0, 1]. Clamp only tiny floating-point
    // round-off around those bounds after rejecting materially invalid values.
    let extrapolated_rbo = raw_extrapolated_rbo.clamp(0.0, 1.0);
    let reference_energy: BTreeMap<&str, f64> = reference
        .iter()
        .map(|item| (item.structure_id.as_str(), item.energy))
        .collect();
    let kendall_tau_b = compute_kendall_tau_b(predicted, &reference_energy, tie_tolerance)?;

    let metrics = PublishedRankingMetrics {
        schema_version: PUBLISHED_RANKING_SCHEMA_VERSION,
        candidate_count: predicted.len(),
        rbo_persistence: persistence,
        extrapolated_rbo,
        kendall_tau_b,
        energy_tie_tolerance: tie_tolerance,
    };
    metrics.validate()?;
    Ok(metrics)
}

fn compare_energy_order(left: &RankedEnergy, right: &RankedEnergy) -> std::cmp::Ordering {
    if left.energy == right.energy {
        left.structure_id.cmp(&right.structure_id)
    } else {
        left.energy.total_cmp(&right.energy)
    }
}

fn validate_side(items: &[RankedEnergy], side: &'static str) -> Result<(), PublishedRankingError> {
    let mut ids = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        if item.structure_id.trim().is_empty() {
            return Err(PublishedRankingError::EmptyStructureId { index });
        }
        if !ids.insert(item.structure_id.as_str()) {
            return Err(PublishedRankingError::DuplicateStructureId { side, index });
        }
        if !item.energy.is_finite() {
            return Err(PublishedRankingError::NonFiniteEnergy { index });
        }
    }
    Ok(())
}

fn compute_extrapolated_rbo(
    predicted: &[&RankedEnergy],
    reference: &[&RankedEnergy],
    persistence: f64,
) -> f64 {
    let mut predicted_prefix = BTreeSet::new();
    let mut reference_prefix = BTreeSet::new();
    let mut overlap = 0usize;
    let mut weighted_prefix_sum = 0.0;
    let mut persistence_power = 1.0;

    for depth in 1..=predicted.len() {
        let predicted_id = predicted[depth - 1].structure_id.as_str();
        let reference_id = reference[depth - 1].structure_id.as_str();

        if predicted_id == reference_id {
            overlap += 1;
        } else {
            if reference_prefix.contains(predicted_id) {
                overlap += 1;
            }
            if predicted_prefix.contains(reference_id) {
                overlap += 1;
            }
        }
        predicted_prefix.insert(predicted_id);
        reference_prefix.insert(reference_id);

        let agreement = overlap as f64 / depth as f64;
        weighted_prefix_sum += (1.0 - persistence) * persistence_power * agreement;
        persistence_power *= persistence;
    }

    weighted_prefix_sum + (overlap as f64 / predicted.len() as f64) * persistence_power
}

fn compare_with_tolerance(left: f64, right: f64, tolerance: f64) -> std::cmp::Ordering {
    if (left - right).abs() <= tolerance {
        std::cmp::Ordering::Equal
    } else {
        left.total_cmp(&right)
    }
}

fn compute_kendall_tau_b(
    predicted: &[RankedEnergy],
    reference: &BTreeMap<&str, f64>,
    tie_tolerance: f64,
) -> Result<f64, PublishedRankingError> {
    let mut concordant = 0u64;
    let mut discordant = 0u64;
    let mut tied_only_predicted = 0u64;
    let mut tied_only_reference = 0u64;

    for (i, predicted_i) in predicted.iter().enumerate() {
        for predicted_j in predicted.iter().skip(i + 1) {
            let predicted_cmp =
                compare_with_tolerance(predicted_i.energy, predicted_j.energy, tie_tolerance);
            let reference_cmp = compare_with_tolerance(
                reference[predicted_i.structure_id.as_str()],
                reference[predicted_j.structure_id.as_str()],
                tie_tolerance,
            );
            match (predicted_cmp, reference_cmp) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {}
                (std::cmp::Ordering::Equal, _) => tied_only_predicted += 1,
                (_, std::cmp::Ordering::Equal) => tied_only_reference += 1,
                (left, right) if left == right => concordant += 1,
                _ => discordant += 1,
            }
        }
    }

    let concordant = concordant as f64;
    let discordant = discordant as f64;
    let tied_only_predicted = tied_only_predicted as f64;
    let tied_only_reference = tied_only_reference as f64;
    let numerator = concordant - discordant;
    let denominator = ((concordant + discordant + tied_only_predicted)
        * (concordant + discordant + tied_only_reference))
        .sqrt();
    if denominator == 0.0 {
        return Err(PublishedRankingError::NoComparablePairs);
    }
    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energies(ids_and_values: &[(&str, f64)]) -> Vec<RankedEnergy> {
        ids_and_values
            .iter()
            .map(|(id, energy)| RankedEnergy {
                structure_id: (*id).to_owned(),
                energy: *energy,
            })
            .collect()
    }

    #[test]
    fn serialized_metrics_match_the_versioned_report_schema() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/published-ranking-metrics-v1.schema.json"
        ))
        .expect("published ranking metric schema must parse");
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(
            schema["properties"]["rbo_persistence"]["exclusiveMinimum"],
            0
        );
        assert_eq!(schema["properties"]["extrapolated_rbo"]["maximum"], 1);
        assert_eq!(schema["properties"]["kendall_tau_b"]["minimum"], -1);

        let predicted = energies(&[("a", -2.0), ("b", -1.0)]);
        let report = compare_published_rankings(&predicted, &predicted, 0.94, 0.0).unwrap();
        let json = serde_json::to_value(&report).unwrap();
        let round_trip: PublishedRankingMetrics = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, report);
        round_trip.validate().unwrap();
    }

    #[test]
    fn identical_rankings_have_unit_rbo_and_tau() {
        let predicted = energies(&[("a", -3.0), ("b", -2.0), ("c", -1.0)]);
        let reference = predicted.clone();
        let metrics = compare_published_rankings(&predicted, &reference, 0.94, 1e-8).unwrap();
        assert!((metrics.extrapolated_rbo - 1.0).abs() < 1e-12);
        assert!((metrics.kendall_tau_b - 1.0).abs() < 1e-12);
        assert_eq!(metrics.candidate_count, 3);
    }

    #[test]
    fn rbo_persistence_must_be_strictly_between_zero_and_one() {
        let candidates = energies(&[("a", -3.0), ("b", -2.0)]);
        for persistence in [0.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                compare_published_rankings(&candidates, &candidates, persistence, 0.0),
                Err(PublishedRankingError::InvalidPersistence)
            );
        }
    }

    #[test]
    fn extrapolated_rbo_matches_hand_calculation_for_reversed_three_item_list() {
        let predicted = energies(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let reference = energies(&[("c", 1.0), ("b", 2.0), ("a", 3.0)]);
        let metrics = compare_published_rankings(&predicted, &reference, 0.5, 0.0).unwrap();
        assert!((metrics.extrapolated_rbo - 0.375).abs() < 1e-12);
        assert!((metrics.kendall_tau_b - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn rbo_tie_break_is_canonical_even_for_signed_zero() {
        let predicted = energies(&[("b", 0.0), ("a", -0.0), ("c", 1.0)]);
        let reference = energies(&[("a", 0.0), ("b", -0.0), ("c", 1.0)]);
        let first = compare_published_rankings(&predicted, &reference, 0.94, 0.0).unwrap();
        let second = compare_published_rankings(
            &predicted.iter().cloned().rev().collect::<Vec<_>>(),
            &reference.iter().cloned().rev().collect::<Vec<_>>(),
            0.94,
            0.0,
        )
        .unwrap();
        assert!((first.extrapolated_rbo - 1.0).abs() < 1e-12);
        assert_eq!(first, second);
    }

    #[test]
    fn kendall_tau_b_preserves_energy_ties() {
        let predicted = energies(&[("a", 0.0), ("b", 0.0), ("c", 1.0)]);
        let reference = energies(&[("a", 0.0), ("b", 1.0), ("c", 1.0)]);
        let metrics = compare_published_rankings(&predicted, &reference, 0.94, 0.0).unwrap();
        assert!((metrics.kendall_tau_b - 0.5).abs() < 1e-12);
        assert!(metrics.extrapolated_rbo.is_finite());
    }

    #[test]
    fn candidate_universe_and_inputs_must_be_valid() {
        let predicted = energies(&[("a", -3.0), ("b", -2.0)]);
        let different_ids = energies(&[("a", -3.0), ("c", -2.0)]);
        assert_eq!(
            compare_published_rankings(&predicted, &different_ids, 0.94, 0.0),
            Err(PublishedRankingError::MismatchedCandidateUniverse)
        );

        let duplicate = energies(&[("a", -3.0), ("a", -2.0)]);
        assert!(matches!(
            compare_published_rankings(&duplicate, &predicted, 0.94, 0.0),
            Err(PublishedRankingError::DuplicateStructureId { .. })
        ));

        let non_finite = energies(&[("a", f64::NAN), ("b", -2.0)]);
        assert_eq!(
            compare_published_rankings(&non_finite, &non_finite, 0.94, 0.0),
            Err(PublishedRankingError::NonFiniteEnergy { index: 0 })
        );
    }
}
