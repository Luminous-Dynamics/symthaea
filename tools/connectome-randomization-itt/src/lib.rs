//! Exact blocked Fisher randomization inference for optogenetic assignment effects.
//!
//! This crate tests the total effect of randomized stimulation assignment on a
//! predeclared outcome summary (the intention-to-treat / reduced-form effect).
//! It does not require an IV exclusion restriction: opsin-mediated effects,
//! direct optical effects, heating, imaging artifacts and all other pathways
//! caused by assignment are part of this estimand.
//!
//! The implemented assignment mechanism is uniform complete randomization
//! within each registered stratum, conditional on its observed treated count.
//! The test enumerates every such allocation and is finite-sample exact for a
//! sharp constant-additive assignment-effect null, provided the supplied
//! assignment mechanism and no-between-unit-interference assertions are true.
//!
//! Two design modes are supported:
//! - WithinClusterBlocks: assignment units are repeated within animals or
//!   sessions; each stratum belongs to one cluster and contains both arms.
//! - ClusterLevelBlocks: one assignment unit represents one independent
//!   cluster, with clusters randomized within strata.
//!
//! The primary statistic gives each independent cluster equal weight in the
//! within-cluster mode, after combining that cluster's block contrasts with
//! block-size weights. In the cluster-level mode it gives each assignment
//! unit (one per cluster) equal weight through block-size-weighted contrasts.
//!
//! Exact randomization inference is conditional on the sharp null. A rejected
//! null does not identify which biological path carried the assignment effect.
//! A non-rejected sharp null is not automatically a test of a weak/average
//! treatment-effect null. Constant-effect confidence sets can be formed by
//! inverting this function over candidate effects, but this crate does not
//! produce a continuous confidence-set inversion.

use core::fmt;
use std::collections::{BTreeMap, HashSet};

const TIE_RELATIVE_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentMode {
    WithinClusterBlocks,
    ClusterLevelBlocks,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomizedUnit {
    /// Unique randomized assignment-unit identifier (for example trial/session).
    pub unit_id: String,
    /// Independent cluster identifier (normally animal, or animal-session).
    pub cluster_id: String,
    /// Randomization block / stratum identifier from the actual design.
    /// In within-cluster mode, a stratum must be nested within one cluster.
    pub stratum_id: String,
    /// Assignment receipt for the binary stimulation schedule.
    pub assigned_treatment: bool,
    /// One predeclared scalar outcome summary per assignment unit.
    pub outcome: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomizationProvenance {
    pub design_id: String,
    /// Digest of the registered randomization protocol and block definition.
    pub protocol_sha256: String,
    /// Digest of the actual assignment receipt / allocation manifest.
    pub allocation_receipt_sha256: String,
    /// Digest of the predeclared outcome-summary artifact / extraction recipe.
    pub outcome_summary_sha256: String,
    /// The outcome and contrast definition were fixed before outcomes were seen.
    pub outcome_summary_prespecified: bool,
    /// The assignment was uniform complete randomization within each stratum.
    pub complete_randomization_within_strata: bool,
    /// Potential outcomes for one assignment unit do not depend on other units'
    /// assignments under this design (within-unit neural/network interactions
    /// remain part of that unit's outcome).
    pub no_between_unit_interference_declared: bool,
    /// Washout / sequence design addresses carryover between assignment units.
    pub carryover_controlled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactRandomizationResult {
    pub design_id: String,
    pub assignment_mode: AssignmentMode,
    pub independent_cluster_count: usize,
    pub assignment_unit_count: usize,
    pub stratum_count: usize,
    pub allocation_count: usize,
    /// Raw observed cluster-macro contrast (candidate effect set to zero).
    pub observed_assignment_effect: f64,
    /// Candidate constant additive effect defining the sharp null.
    pub candidate_effect: f64,
    /// Contrast after subtracting candidate_effect from treated outcomes.
    pub adjusted_observed_contrast: f64,
    pub absolute_test_statistic: f64,
    pub extreme_allocation_count: usize,
    /// Exact two-sided conditional randomization p-value, ties included.
    pub p_value_two_sided: f64,
    pub randomization_distribution_mean: f64,
    pub randomization_distribution_sd: f64,
    pub claim_ceiling: &'static str,
    pub provenance: RandomizationProvenance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RandomizationError {
    EmptyUnitSet,
    EmptyUnitId,
    EmptyClusterId { unit_id: String },
    EmptyStratumId { unit_id: String },
    DuplicateUnitId { unit_id: String },
    NonFiniteOutcome { unit_id: String },
    InvalidCandidateEffect { value: f64 },
    InvalidAllocationLimit { value: usize },
    InvalidDesignId,
    InvalidProtocolDigest,
    InvalidAllocationDigest,
    InvalidOutcomeDigest,
    OutcomeSummaryNotPrespecified,
    AssignmentMechanismNotSupported,
    InterferenceNotAddressed,
    CarryoverNotAddressed,
    TooFewIndependentClusters { minimum: usize, actual: usize },
    SingleArmStratum { stratum_id: String },
    StratumSpansClusters { stratum_id: String },
    RepeatedClusterInClusterLevelMode { cluster_id: String },
    AllocationSpaceTooLarge { limit: usize },
    AllocationCountOverflow,
    ObservedAllocationNotInEnumeratedSpace,
    EnumerationCountMismatch { expected: usize, actual: usize },
    InvalidRandomizationStatistic,
    NumericalOverflow { field: &'static str },
}

impl fmt::Display for RandomizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyUnitSet => write!(f, "randomization unit set must not be empty"),
            Self::EmptyUnitId => write!(f, "randomization unit id must not be empty"),
            Self::EmptyClusterId { unit_id } => write!(f, "unit {unit_id:?} has an empty cluster id"),
            Self::EmptyStratumId { unit_id } => write!(f, "unit {unit_id:?} has an empty stratum id"),
            Self::DuplicateUnitId { unit_id } => write!(f, "duplicate randomization unit id {unit_id:?}"),
            Self::NonFiniteOutcome { unit_id } => write!(f, "unit {unit_id:?} has a non-finite outcome"),
            Self::InvalidCandidateEffect { value } => write!(f, "candidate effect is not finite: {value}"),
            Self::InvalidAllocationLimit { value } => write!(f, "allocation enumeration limit must be positive: {value}"),
            Self::InvalidDesignId => write!(f, "design id must not be empty"),
            Self::InvalidProtocolDigest => write!(f, "protocol digest must be lowercase SHA-256"),
            Self::InvalidAllocationDigest => write!(f, "allocation receipt digest must be lowercase SHA-256"),
            Self::InvalidOutcomeDigest => write!(f, "outcome summary digest must be lowercase SHA-256"),
            Self::OutcomeSummaryNotPrespecified => write!(f, "outcome summary was not prespecified"),
            Self::AssignmentMechanismNotSupported => write!(f, "only uniform complete randomization within registered strata is implemented"),
            Self::InterferenceNotAddressed => write!(f, "between-assignment-unit interference is not addressed by the declared design"),
            Self::CarryoverNotAddressed => write!(f, "carryover between randomized assignment units is not controlled"),
            Self::TooFewIndependentClusters { minimum, actual } => write!(f, "at least {minimum} independent clusters are required; received {actual}"),
            Self::SingleArmStratum { stratum_id } => write!(f, "stratum {stratum_id:?} must contain both treatment and control assignments"),
            Self::StratumSpansClusters { stratum_id } => write!(f, "within-cluster stratum {stratum_id:?} spans more than one independent cluster"),
            Self::RepeatedClusterInClusterLevelMode { cluster_id } => write!(f, "cluster-level mode requires exactly one assignment unit per cluster; repeated cluster {cluster_id:?} found"),
            Self::AllocationSpaceTooLarge { limit } => write!(f, "exact allocation space exceeds the registered limit {limit}; no approximate fallback is performed"),
            Self::AllocationCountOverflow => write!(f, "allocation-space cardinality overflow"),
            Self::ObservedAllocationNotInEnumeratedSpace => write!(f, "observed assignment is not in the generated randomization space"),
            Self::EnumerationCountMismatch { expected, actual } => write!(f, "enumerated {actual} allocations but expected {expected}"),
            Self::InvalidRandomizationStatistic => write!(f, "randomization statistic is undefined for this allocation"),
            Self::NumericalOverflow { field } => write!(f, "numerical overflow while computing {field}"),
        }
    }
}

impl std::error::Error for RandomizationError {}

#[derive(Debug, Clone)]
struct Stratum {
    indices: Vec<usize>,
    cluster_id: Option<String>,
    combinations: Vec<Vec<usize>>,
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_provenance(
    provenance: &RandomizationProvenance,
) -> Result<(), RandomizationError> {
    if provenance.design_id.trim().is_empty() {
        return Err(RandomizationError::InvalidDesignId);
    }
    if !is_lower_hex_sha256(&provenance.protocol_sha256) {
        return Err(RandomizationError::InvalidProtocolDigest);
    }
    if !is_lower_hex_sha256(&provenance.allocation_receipt_sha256) {
        return Err(RandomizationError::InvalidAllocationDigest);
    }
    if !is_lower_hex_sha256(&provenance.outcome_summary_sha256) {
        return Err(RandomizationError::InvalidOutcomeDigest);
    }
    if !provenance.outcome_summary_prespecified {
        return Err(RandomizationError::OutcomeSummaryNotPrespecified);
    }
    if !provenance.complete_randomization_within_strata {
        return Err(RandomizationError::AssignmentMechanismNotSupported);
    }
    if !provenance.no_between_unit_interference_declared {
        return Err(RandomizationError::InterferenceNotAddressed);
    }
    if !provenance.carryover_controlled {
        return Err(RandomizationError::CarryoverNotAddressed);
    }
    Ok(())
}

fn binomial_capped(n: usize, k: usize, cap: usize) -> Result<usize, RandomizationError> {
    let k = k.min(n.saturating_sub(k));
    let mut result = 1_u128;
    for i in 1..=k {
        let numerator = (n - k + i) as u128;
        result = result
            .checked_mul(numerator)
            .ok_or(RandomizationError::AllocationCountOverflow)?
            / i as u128;
        if result > cap as u128 {
            return Err(RandomizationError::AllocationSpaceTooLarge { limit: cap });
        }
    }
    usize::try_from(result).map_err(|_| RandomizationError::AllocationCountOverflow)
}

fn build_combinations(
    indices: &[usize],
    choose: usize,
    start: usize,
    selected: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if selected.len() == choose {
        output.push(selected.clone());
        return;
    }
    let remaining = choose - selected.len();
    if indices.len() < remaining || start > indices.len() - remaining {
        return;
    }
    let last_start = indices.len() - remaining;
    for position in start..=last_start {
        selected.push(indices[position]);
        build_combinations(indices, choose, position + 1, selected, output);
        selected.pop();
    }
}

fn prepare_strata(
    units: &[RandomizedUnit],
    mode: AssignmentMode,
    max_allocations: usize,
) -> Result<(Vec<Stratum>, usize, usize), RandomizationError> {
    let mut unit_ids = HashSet::with_capacity(units.len());
    let mut clusters: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut strata: BTreeMap<&str, Vec<usize>> = BTreeMap::new();

    for (index, unit) in units.iter().enumerate() {
        if unit.unit_id.trim().is_empty() {
            return Err(RandomizationError::EmptyUnitId);
        }
        if !unit_ids.insert(unit.unit_id.as_str()) {
            return Err(RandomizationError::DuplicateUnitId {
                unit_id: unit.unit_id.clone(),
            });
        }
        if unit.cluster_id.trim().is_empty() {
            return Err(RandomizationError::EmptyClusterId {
                unit_id: unit.unit_id.clone(),
            });
        }
        if unit.stratum_id.trim().is_empty() {
            return Err(RandomizationError::EmptyStratumId {
                unit_id: unit.unit_id.clone(),
            });
        }
        if !unit.outcome.is_finite() {
            return Err(RandomizationError::NonFiniteOutcome {
                unit_id: unit.unit_id.clone(),
            });
        }
        clusters.entry(unit.cluster_id.as_str()).or_default().push(index);
        strata.entry(unit.stratum_id.as_str()).or_default().push(index);
    }

    let cluster_count = clusters.len();
    if cluster_count < 2 {
        return Err(RandomizationError::TooFewIndependentClusters {
            minimum: 2,
            actual: cluster_count,
        });
    }

    if mode == AssignmentMode::ClusterLevelBlocks {
        for (cluster_id, indices) in &clusters {
            if indices.len() != 1 {
                return Err(RandomizationError::RepeatedClusterInClusterLevelMode {
                    cluster_id: (*cluster_id).to_owned(),
                });
            }
        }
    }

    let mut prepared = Vec::with_capacity(strata.len());
    let mut allocation_count = 1usize;

    for (stratum_id, indices) in strata {
        let treated_count = indices
            .iter()
            .filter(|&&index| units[index].assigned_treatment)
            .count();
        if treated_count == 0 || treated_count == indices.len() {
            return Err(RandomizationError::SingleArmStratum {
                stratum_id: stratum_id.to_owned(),
            });
        }

        let cluster_id = match mode {
            AssignmentMode::WithinClusterBlocks => {
                let first = units[indices[0]].cluster_id.as_str();
                if indices
                    .iter()
                    .any(|&index| units[index].cluster_id.as_str() != first)
                {
                    return Err(RandomizationError::StratumSpansClusters {
                        stratum_id: stratum_id.to_owned(),
                    });
                }
                Some(first.to_owned())
            }
            AssignmentMode::ClusterLevelBlocks => None,
        };

        let combination_count = binomial_capped(indices.len(), treated_count, max_allocations)?;
        allocation_count = allocation_count
            .checked_mul(combination_count)
            .ok_or(RandomizationError::AllocationCountOverflow)?;
        if allocation_count > max_allocations {
            return Err(RandomizationError::AllocationSpaceTooLarge {
                limit: max_allocations,
            });
        }

        let mut combinations = Vec::with_capacity(combination_count);
        build_combinations(
            &indices,
            treated_count,
            0,
            &mut Vec::with_capacity(treated_count),
            &mut combinations,
        );
        if combinations.len() != combination_count {
            return Err(RandomizationError::EnumerationCountMismatch {
                expected: combination_count,
                actual: combinations.len(),
            });
        }

        prepared.push(Stratum {
            indices,
            cluster_id,
            combinations,
        });
    }

    Ok((prepared, cluster_count, allocation_count))
}

fn allocation_statistic(
    assignment: &[bool],
    adjusted_outcomes: &[f64],
    strata: &[Stratum],
    mode: AssignmentMode,
    cluster_count: usize,
) -> Option<f64> {
    let mut block_contrasts = Vec::with_capacity(strata.len());

    for stratum in strata {
        let mut treated_sum = 0.0;
        let mut control_sum = 0.0;
        let mut treated_n = 0usize;
        let mut control_n = 0usize;

        for &index in &stratum.indices {
            if assignment[index] {
                treated_sum += adjusted_outcomes[index];
                treated_n += 1;
            } else {
                control_sum += adjusted_outcomes[index];
                control_n += 1;
            }
        }

        if treated_n == 0 || control_n == 0 {
            return None;
        }
        let treated_mean = treated_sum / treated_n as f64;
        let control_mean = control_sum / control_n as f64;
        let difference = treated_mean - control_mean;
        if !difference.is_finite() {
            return None;
        }
        block_contrasts.push((stratum, difference));
    }

    match mode {
        AssignmentMode::WithinClusterBlocks => {
            let mut by_cluster: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
            for (stratum, difference) in block_contrasts {
                let cluster_id = stratum.cluster_id.as_deref()?;
                let entry = by_cluster.entry(cluster_id).or_default();
                entry.0 += difference * stratum.indices.len() as f64;
                entry.1 += stratum.indices.len();
            }
            if by_cluster.len() != cluster_count {
                return None;
            }
            let cluster_effects: Vec<f64> = by_cluster
                .values()
                .map(|(weighted_sum, unit_count)| weighted_sum / *unit_count as f64)
                .collect();
            let mean = cluster_effects.iter().sum::<f64>() / cluster_effects.len() as f64;
            if mean.is_finite() { Some(mean) } else { None }
        }
        AssignmentMode::ClusterLevelBlocks => {
            let total_units: usize = block_contrasts
                .iter()
                .map(|(stratum, _)| stratum.indices.len())
                .sum();
            let weighted_sum = block_contrasts
                .iter()
                .map(|(stratum, difference)| difference * stratum.indices.len() as f64)
                .sum::<f64>();
            let mean = weighted_sum / total_units as f64;
            if mean.is_finite() { Some(mean) } else { None }
        }
    }
}

fn enumerate_strata<F>(
    strata: &[Stratum],
    stratum_index: usize,
    assignment: &mut [bool],
    callback: &mut F,
) where
    F: FnMut(&[bool]),
{
    if stratum_index == strata.len() {
        callback(assignment);
        return;
    }

    let stratum = &strata[stratum_index];
    for combination in &stratum.combinations {
        for &index in &stratum.indices {
            assignment[index] = false;
        }
        for &index in combination {
            assignment[index] = true;
        }
        enumerate_strata(strata, stratum_index + 1, assignment, callback);
    }
}

/// Perform exact two-sided Fisher randomization inference for a blocked
/// stimulation-assignment experiment.
///
/// candidate_effect specifies the sharp constant-additive null:
/// Y_i(1) = Y_i(0) + candidate_effect. The procedure adjusts outcomes by
/// subtracting the candidate effect from treated observations, then enumerates
/// every uniformly assigned allocation preserving the observed treated count
/// in each stratum.
///
/// max_allocations is a hard safety cap. The function never silently changes
/// to Monte Carlo inference. If the allocation space exceeds the cap it fails
/// closed and returns AllocationSpaceTooLarge.
pub fn exact_blocked_randomization_test(
    units: &[RandomizedUnit],
    mode: AssignmentMode,
    candidate_effect: f64,
    provenance: RandomizationProvenance,
    max_allocations: usize,
) -> Result<ExactRandomizationResult, RandomizationError> {
    if units.is_empty() {
        return Err(RandomizationError::EmptyUnitSet);
    }
    if !candidate_effect.is_finite() {
        return Err(RandomizationError::InvalidCandidateEffect {
            value: candidate_effect,
        });
    }
    if max_allocations == 0 {
        return Err(RandomizationError::InvalidAllocationLimit { value: max_allocations });
    }
    validate_provenance(&provenance)?;

    let (strata, cluster_count, expected_allocations) =
        prepare_strata(units, mode, max_allocations)?;

    let observed_assignment: Vec<bool> = units
        .iter()
        .map(|unit| unit.assigned_treatment)
        .collect();
    let raw_outcomes: Vec<f64> = units.iter().map(|unit| unit.outcome).collect();
    let adjusted_outcomes: Vec<f64> = units
        .iter()
        .map(|unit| {
            unit.outcome
                - if unit.assigned_treatment {
                    candidate_effect
                } else {
                    0.0
                }
        })
        .collect();
    if adjusted_outcomes.iter().any(|value| !value.is_finite()) {
        return Err(RandomizationError::NumericalOverflow {
            field: "sharp-null outcome adjustment",
        });
    }

    let observed_assignment_effect = allocation_statistic(
        &observed_assignment,
        &raw_outcomes,
        &strata,
        mode,
        cluster_count,
    )
    .ok_or(RandomizationError::InvalidRandomizationStatistic)?;
    let adjusted_observed_contrast = allocation_statistic(
        &observed_assignment,
        &adjusted_outcomes,
        &strata,
        mode,
        cluster_count,
    )
    .ok_or(RandomizationError::InvalidRandomizationStatistic)?;
    let absolute_test_statistic = adjusted_observed_contrast.abs();
    let tie_tolerance = TIE_RELATIVE_TOLERANCE * absolute_test_statistic.abs().max(1.0);

    let mut enumerated = 0usize;
    let mut extreme = 0usize;
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut observed_seen = false;
    let mut statistic_invalid = false;
    let mut assignment = vec![false; units.len()];

    enumerate_strata(&strata, 0, &mut assignment, &mut |candidate_assignment| {
        enumerated += 1;
        if candidate_assignment == observed_assignment.as_slice() {
            observed_seen = true;
        }

        match allocation_statistic(
            candidate_assignment,
            &adjusted_outcomes,
            &strata,
            mode,
            cluster_count,
        ) {
            Some(value) => {
                if value.abs() + tie_tolerance >= absolute_test_statistic {
                    extreme += 1;
                }
                sum += value;
                sum_squares += value * value;
                if !sum.is_finite() || !sum_squares.is_finite() {
                    statistic_invalid = true;
                }
            }
            None => statistic_invalid = true,
        }
    });

    if statistic_invalid {
        return Err(RandomizationError::NumericalOverflow {
            field: "randomization distribution statistic",
        });
    }
    if !observed_seen {
        return Err(RandomizationError::ObservedAllocationNotInEnumeratedSpace);
    }
    if enumerated != expected_allocations {
        return Err(RandomizationError::EnumerationCountMismatch {
            expected: expected_allocations,
            actual: enumerated,
        });
    }
    if extreme == 0 {
        return Err(RandomizationError::ObservedAllocationNotInEnumeratedSpace);
    }

    let allocation_count = enumerated;
    let distribution_mean = sum / allocation_count as f64;
    let distribution_variance =
        (sum_squares / allocation_count as f64 - distribution_mean * distribution_mean)
            .max(0.0);
    let distribution_sd = distribution_variance.sqrt();
    let p_value = extreme as f64 / allocation_count as f64;

    for (field, value) in [
        ("randomization distribution mean", distribution_mean),
        ("randomization distribution standard deviation", distribution_sd),
        ("exact p-value", p_value),
    ] {
        if !value.is_finite() {
            return Err(RandomizationError::NumericalOverflow { field });
        }
    }

    Ok(ExactRandomizationResult {
        design_id: provenance.design_id.clone(),
        assignment_mode: mode,
        independent_cluster_count: cluster_count,
        assignment_unit_count: units.len(),
        stratum_count: strata.len(),
        allocation_count,
        observed_assignment_effect,
        candidate_effect,
        adjusted_observed_contrast,
        absolute_test_statistic,
        extreme_allocation_count: extreme,
        p_value_two_sided: p_value,
        randomization_distribution_mean: distribution_mean,
        randomization_distribution_sd: distribution_sd,
        claim_ceiling: "FINITE_SAMPLE_RANDOMIZATION_SHARP_NULL_ONLY",
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> RandomizationProvenance {
        RandomizationProvenance {
            design_id: "registered-optogenetic-blocked-design-v1".to_owned(),
            protocol_sha256: "a".repeat(64),
            allocation_receipt_sha256: "b".repeat(64),
            outcome_summary_sha256: "c".repeat(64),
            outcome_summary_prespecified: true,
            complete_randomization_within_strata: true,
            no_between_unit_interference_declared: true,
            carryover_controlled: true,
        }
    }

    fn unit(
        id: &str,
        cluster: &str,
        stratum: &str,
        treated: bool,
        outcome: f64,
    ) -> RandomizedUnit {
        RandomizedUnit {
            unit_id: id.to_owned(),
            cluster_id: cluster.to_owned(),
            stratum_id: stratum.to_owned(),
            assigned_treatment: treated,
            outcome,
        }
    }

    fn within_fixture() -> Vec<RandomizedUnit> {
        vec![
            unit("a1", "animal-a", "a-block-1", true, 1.0),
            unit("a2", "animal-a", "a-block-1", false, 2.0),
            unit("a3", "animal-a", "a-block-1", true, 3.0),
            unit("a4", "animal-a", "a-block-1", false, 4.0),
            unit("b1", "animal-b", "b-block-1", true, 2.0),
            unit("b2", "animal-b", "b-block-1", false, 3.0),
            unit("b3", "animal-b", "b-block-1", false, 4.0),
            unit("b4", "animal-b", "b-block-1", true, 5.0),
        ]
    }

    fn between_fixture() -> Vec<RandomizedUnit> {
        vec![
            unit("animal-a", "animal-a", "block-1", true, 9.0),
            unit("animal-b", "animal-b", "block-1", true, 8.0),
            unit("animal-c", "animal-c", "block-1", false, 1.0),
            unit("animal-d", "animal-d", "block-1", false, 0.0),
        ]
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual:.15}, expected {expected:.15}"
        );
    }

    #[test]
    fn within_cluster_enumeration_preserves_block_counts_and_clusters() {
        let result = exact_blocked_randomization_test(
            &within_fixture(),
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        assert_eq!(result.independent_cluster_count, 2);
        assert_eq!(result.assignment_unit_count, 8);
        assert_eq!(result.stratum_count, 2);
        assert_eq!(result.allocation_count, 36);
        assert_eq!(result.extreme_allocation_count, 28);
        close(result.p_value_two_sided, 28.0 / 36.0);
        close(result.observed_assignment_effect, -0.5);
        assert_eq!(result.claim_ceiling, "FINITE_SAMPLE_RANDOMIZATION_SHARP_NULL_ONLY");
    }

    #[test]
    fn between_cluster_exact_randomization_has_hand_calculated_p_value() {
        let result = exact_blocked_randomization_test(
            &between_fixture(),
            AssignmentMode::ClusterLevelBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        assert_eq!(result.independent_cluster_count, 4);
        assert_eq!(result.allocation_count, 6);
        assert_eq!(result.extreme_allocation_count, 2);
        close(result.observed_assignment_effect, 8.0);
        close(result.p_value_two_sided, 1.0 / 3.0);
    }

    #[test]
    fn candidate_constant_effect_is_subtracted_under_sharp_null() {
        let result = exact_blocked_randomization_test(
            &between_fixture(),
            AssignmentMode::ClusterLevelBlocks,
            3.0,
            provenance(),
            100,
        )
        .unwrap();
        close(result.observed_assignment_effect, 8.0);
        close(result.adjusted_observed_contrast, 5.0);
        close(result.p_value_two_sided, 1.0 / 3.0);
        close(result.candidate_effect, 3.0);
    }

    #[test]
    fn exact_p_value_includes_observed_allocation_and_never_uses_plus_one() {
        let result = exact_blocked_randomization_test(
            &between_fixture(),
            AssignmentMode::ClusterLevelBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        assert!(result.extreme_allocation_count >= 1);
        close(
            result.p_value_two_sided,
            result.extreme_allocation_count as f64 / result.allocation_count as f64,
        );
    }

    #[test]
    fn repeated_units_within_animal_do_not_increase_independent_cluster_count() {
        let result = exact_blocked_randomization_test(
            &within_fixture(),
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        assert_eq!(result.assignment_unit_count, 8);
        assert_eq!(result.independent_cluster_count, 2);
    }

    #[test]
    fn multiple_strata_within_animal_are_combined_before_equal_cluster_weighting() {
        let units = vec![
            unit("a1", "a", "a1", true, 1.0),
            unit("a2", "a", "a1", false, 2.0),
            unit("a3", "a", "a2", true, 3.0),
            unit("a4", "a", "a2", false, 4.0),
            unit("b1", "b", "b1", true, 4.0),
            unit("b2", "b", "b1", false, 3.0),
            unit("b3", "b", "b2", true, 2.0),
            unit("b4", "b", "b2", false, 1.0),
        ];
        let result = exact_blocked_randomization_test(
            &units,
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            1000,
        )
        .unwrap();
        assert_eq!(result.independent_cluster_count, 2);
        assert_eq!(result.stratum_count, 4);
        assert_eq!(result.allocation_count, 16);
        close(result.observed_assignment_effect, 0.0);
    }

    #[test]
    fn allocation_space_cardinality_is_product_of_block_binomials() {
        let units = within_fixture();
        let result = exact_blocked_randomization_test(
            &units,
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            36,
        )
        .unwrap();
        assert_eq!(result.allocation_count, 36);
    }

    #[test]
    fn too_small_allocation_cap_fails_closed_without_approximate_fallback() {
        assert_eq!(
            exact_blocked_randomization_test(
                &within_fixture(),
                AssignmentMode::WithinClusterBlocks,
                0.0,
                provenance(),
                35,
            ),
            Err(RandomizationError::AllocationSpaceTooLarge { limit: 35 })
        );
    }

    #[test]
    fn within_cluster_stratum_may_not_cross_animals() {
        let units = vec![
            unit("a1", "a", "shared", true, 1.0),
            unit("a2", "a", "shared", false, 2.0),
            unit("b1", "b", "shared", true, 3.0),
            unit("b2", "b", "shared", false, 4.0),
        ];
        assert!(matches!(
            exact_blocked_randomization_test(
                &units,
                AssignmentMode::WithinClusterBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::StratumSpansClusters { .. })
        ));
    }

    #[test]
    fn cluster_level_mode_rejects_repeated_rows_per_animal() {
        assert!(matches!(
            exact_blocked_randomization_test(
                &within_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::RepeatedClusterInClusterLevelMode { .. })
        ));
    }

    #[test]
    fn duplicate_unit_id_fails_closed() {
        let mut units = between_fixture();
        units[1].unit_id = units[0].unit_id.clone();
        assert!(matches!(
            exact_blocked_randomization_test(
                &units,
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::DuplicateUnitId { .. })
        ));
    }

    #[test]
    fn non_finite_outcome_and_candidate_effect_fail_closed() {
        let mut units = between_fixture();
        units[0].outcome = f64::NAN;
        assert!(matches!(
            exact_blocked_randomization_test(
                &units,
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::NonFiniteOutcome { .. })
        ));

        assert!(matches!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                f64::INFINITY,
                provenance(),
                100,
            ),
            Err(RandomizationError::InvalidCandidateEffect { .. })
        ));
    }

    #[test]
    fn malformed_provenance_and_unregistered_mechanism_fail_closed() {
        let mut meta = provenance();
        meta.protocol_sha256 = "bad".to_owned();
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                meta,
                100,
            ),
            Err(RandomizationError::InvalidProtocolDigest)
        );

        let mut meta = provenance();
        meta.complete_randomization_within_strata = false;
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                meta,
                100,
            ),
            Err(RandomizationError::AssignmentMechanismNotSupported)
        );
    }

    #[test]
    fn missing_interference_or_carryover_controls_fail_closed() {
        let mut meta = provenance();
        meta.no_between_unit_interference_declared = false;
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                meta,
                100,
            ),
            Err(RandomizationError::InterferenceNotAddressed)
        );

        let mut meta = provenance();
        meta.carryover_controlled = false;
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                meta,
                100,
            ),
            Err(RandomizationError::CarryoverNotAddressed)
        );
    }

    #[test]
    fn observed_outcome_summary_must_be_prespecified() {
        let mut meta = provenance();
        meta.outcome_summary_prespecified = false;
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                meta,
                100,
            ),
            Err(RandomizationError::OutcomeSummaryNotPrespecified)
        );
    }

    #[test]
    fn one_independent_cluster_fails_closed() {
        let units = vec![
            unit("a1", "a", "block", true, 1.0),
            unit("a2", "a", "block", false, 2.0),
        ];
        assert_eq!(
            exact_blocked_randomization_test(
                &units,
                AssignmentMode::WithinClusterBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::TooFewIndependentClusters { minimum: 2, actual: 1 })
        );
    }

    #[test]
    fn design_strata_are_counted_and_sorted_deterministically() {
        let units = vec![
            unit("b1", "b", "block-b", true, 4.0),
            unit("b2", "b", "block-b", false, 3.0),
            unit("a1", "a", "block-a", true, 1.0),
            unit("a2", "a", "block-a", false, 2.0),
        ];
        let first = exact_blocked_randomization_test(
            &units,
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        let second = exact_blocked_randomization_test(
            &units,
            AssignmentMode::WithinClusterBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.stratum_count, 2);
        assert_eq!(first.allocation_count, 4);
    }

    #[test]
    fn assigned_light_artifact_remains_part_of_the_itt_estimand() {
        // Outcomes here are stipulated to arise from a direct light/thermal
        // artifact, not from the intended opsin-mediated source population.
        // The randomization oracle correctly tests the assignment effect but
        // has no field that can relabel it as neural mediation.
        let units = vec![
            unit("animal-a", "animal-a", "block-1", true, 12.0),
            unit("animal-b", "animal-b", "block-1", true, 0.0),
            unit("animal-c", "animal-c", "block-1", false, 0.0),
            unit("animal-d", "animal-d", "block-1", false, 0.0),
        ];
        let result = exact_blocked_randomization_test(
            &units,
            AssignmentMode::ClusterLevelBlocks,
            0.0,
            provenance(),
            100,
        )
        .unwrap();
        close(result.observed_assignment_effect, 6.0);
        assert_eq!(result.claim_ceiling, "FINITE_SAMPLE_RANDOMIZATION_SHARP_NULL_ONLY");
    }

    #[test]
    fn single_arm_block_fails_closed() {
        let units = vec![
            unit("a1", "a", "a-block", true, 1.0),
            unit("a2", "a", "a-block", true, 2.0),
            unit("b1", "b", "b-block", true, 3.0),
            unit("b2", "b", "b-block", false, 4.0),
        ];
        assert_eq!(
            exact_blocked_randomization_test(
                &units,
                AssignmentMode::WithinClusterBlocks,
                0.0,
                provenance(),
                100,
            ),
            Err(RandomizationError::SingleArmStratum {
                stratum_id: "a-block".to_owned()
            })
        );
    }

    #[test]
    fn zero_allocation_cap_fails_closed() {
        assert_eq!(
            exact_blocked_randomization_test(
                &between_fixture(),
                AssignmentMode::ClusterLevelBlocks,
                0.0,
                provenance(),
                0,
            ),
            Err(RandomizationError::InvalidAllocationLimit { value: 0 })
        );
    }

}
