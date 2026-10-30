//! Cluster-aware paired inference for hierarchical benchmark suites.
//!
//! The input is one already-reduced quantum/classical pair per workload
//! instance. Instances are nested inside independently sampled workload
//! families. The API first reduces instance-level effects to one median
//! effect per family, then performs exact sign tests across families. It
//! therefore targets a family-balanced estimand, not an instance-weighted
//! estimand. Repeated shots, attempts, seeds and timing samples must be
//! reduced before constructing these observations.

use std::collections::{BTreeMap, HashSet};

use crate::paired_advantage::exact_one_sided_sign_test_p_value;

/// One paired comparison for one distinct workload instance within a family.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredBenchmarkObservation {
    /// Independent family or generator-lineage identifier.
    pub family_id: String,
    /// Unique block ID after preregistered within-instance reduction.
    pub independent_block_id: String,
    /// Globally unique workload-instance fingerprint.
    pub workload_fingerprint: String,
    /// Digest binding workload, objective, quality contract, verifier and resources.
    pub comparison_contract_fingerprint: String,
    /// Quantum end-to-end resource per verified acceptable answer.
    pub quantum_resource_per_accepted_answer: f64,
    /// Classical end-to-end resource per verified acceptable answer.
    pub classical_resource_per_accepted_answer: f64,
    /// Quantum output quality; larger values must be better.
    pub quantum_quality: f64,
    /// Classical output quality; larger values must be better.
    pub classical_quality: f64,
}

/// Preregistered margins and minimum number of independent families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusteredAdvantageConfig {
    /// Required minimum number of independent families for inference.
    pub minimum_independent_families: usize,
    /// Required minimum fractional reduction from classical resource use, in [0, 1).
    pub minimum_relative_resource_reduction: f64,
    /// Largest tolerated quantum quality decrease relative to the classical pair.
    pub quality_non_inferiority_margin: f64,
    /// Tie tolerance for log(classical resource / quantum resource).
    pub resource_log_ratio_tie_tolerance: f64,
    /// Tie tolerance for quantum minus classical quality.
    pub quality_difference_tie_tolerance: f64,
}

/// One cluster's equally weighted summary of its workload instances.
#[derive(Debug, Clone, PartialEq)]
pub struct FamilyEffect {
    /// Family / generator-lineage identifier.
    pub family_id: String,
    /// Number of distinct workload instances summarized in this family.
    pub instance_count: usize,
    /// Median across instances of quantum quality minus classical quality.
    pub median_quality_difference: f64,
    /// Median across instances of log(classical resource / quantum resource).
    pub median_log_classical_over_quantum_resource: f64,
}

/// Exact one-sided sign test across independent families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilySignTest {
    /// Families whose within-family median exceeds the declared margin.
    pub wins: usize,
    /// Families whose within-family median falls below the declared margin.
    pub losses: usize,
    /// Families whose within-family median falls within the tie tolerance.
    pub ties: usize,
    /// Number of independent families; ties remain in this denominator.
    pub independent_families: usize,
    /// Exact one-sided fair-binomial tail probability.
    pub p_value_one_sided: f64,
}

/// Family-balanced result with leave-one-family-out sensitivity.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredAdvantageReport {
    /// Number of independent workload families.
    pub independent_families: usize,
    /// Total distinct workload instances across those families.
    pub workload_instances: usize,
    /// Family-balanced median quality difference.
    pub median_family_quality_difference: f64,
    /// Family-balanced median log classical/quantum resource ratio.
    pub median_family_log_resource_ratio: f64,
    /// Exponentiated family-balanced median log ratio, absent on overflow.
    pub median_family_resource_ratio: Option<f64>,
    /// Quality non-inferiority test across independent families.
    pub quality_non_inferiority: FamilySignTest,
    /// Practical resource-superiority test across independent families.
    pub resource_superiority: FamilySignTest,
    /// Intersection-union p-value for quality AND resource claim.
    pub composite_claim_p_value: f64,
    /// Min/max leave-one-family-out median quality effect.
    pub leave_one_family_out_quality_range: (f64, f64),
    /// Min/max leave-one-family-out median log resource effect.
    pub leave_one_family_out_resource_range: (f64, f64),
    /// Per-family summaries, ordered by family identifier.
    pub family_effects: Vec<FamilyEffect>,
    /// Shared comparison-contract digest.
    pub comparison_contract_fingerprint: String,
}

/// Fail-closed errors for cluster-aware paired inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteredAdvantageError {
    /// No paired workload instances were supplied.
    EmptyObservations,
    /// A required identifier is empty.
    EmptyIdentifier,
    /// An independent block identifier is duplicated.
    DuplicateBlockId,
    /// A workload fingerprint is duplicated.
    DuplicateWorkloadFingerprint,
    /// Comparison contract fingerprints differ.
    ContractMismatch,
    /// Resource values must be finite and strictly positive.
    InvalidResource,
    /// Quality values or their paired difference must be finite.
    InvalidQuality,
    /// A margin, tolerance, or required family count is invalid.
    InvalidConfig,
    /// Too few distinct families for the preregistered minimum.
    InsufficientIndependentFamilies,
}

impl std::fmt::Display for ClusteredAdvantageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyObservations => "no paired workload instances",
            Self::EmptyIdentifier => "required family, block, workload or contract ID is empty",
            Self::DuplicateBlockId => "duplicate independent block identifier",
            Self::DuplicateWorkloadFingerprint => "duplicate workload fingerprint",
            Self::ContractMismatch => "comparison-contract fingerprints differ",
            Self::InvalidResource => "resource values must be finite and strictly positive",
            Self::InvalidQuality => "quality values or paired difference are non-finite",
            Self::InvalidConfig => "invalid margin, tolerance or minimum family count",
            Self::InsufficientIndependentFamilies => "too few independent families",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ClusteredAdvantageError {}

/// Aggregate instance effects within family, then infer across families.
///
/// Each family contributes exactly one sign to each test, regardless of its
/// instance count. The test therefore targets the median effect across
/// independent families. It does not estimate an instance-weighted population
/// effect. Family IDs must encode shared generators, source lineages or other
/// dependence that makes instances exchangeable only within a cluster.
pub fn analyze_clustered_advantage(
    observations: &[ClusteredBenchmarkObservation],
    config: ClusteredAdvantageConfig,
) -> Result<ClusteredAdvantageReport, ClusteredAdvantageError> {
    if observations.is_empty() {
        return Err(ClusteredAdvantageError::EmptyObservations);
    }
    if config.minimum_independent_families < 2
        || !config.minimum_relative_resource_reduction.is_finite()
        || !(0.0..1.0).contains(&config.minimum_relative_resource_reduction)
        || !config.quality_non_inferiority_margin.is_finite()
        || config.quality_non_inferiority_margin < 0.0
        || !config.resource_log_ratio_tie_tolerance.is_finite()
        || config.resource_log_ratio_tie_tolerance < 0.0
        || !config.quality_difference_tie_tolerance.is_finite()
        || config.quality_difference_tie_tolerance < 0.0
    {
        return Err(ClusteredAdvantageError::InvalidConfig);
    }

    let contract = observations[0].comparison_contract_fingerprint.trim();
    if contract.is_empty() {
        return Err(ClusteredAdvantageError::EmptyIdentifier);
    }

    let mut blocks = HashSet::with_capacity(observations.len());
    let mut workloads = HashSet::with_capacity(observations.len());
    let mut grouped: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();

    for observation in observations {
        if observation.family_id.trim().is_empty()
            || observation.independent_block_id.trim().is_empty()
            || observation.workload_fingerprint.trim().is_empty()
            || observation.comparison_contract_fingerprint.trim().is_empty()
        {
            return Err(ClusteredAdvantageError::EmptyIdentifier);
        }
        if !blocks.insert(observation.independent_block_id.as_str()) {
            return Err(ClusteredAdvantageError::DuplicateBlockId);
        }
        if !workloads.insert(observation.workload_fingerprint.as_str()) {
            return Err(ClusteredAdvantageError::DuplicateWorkloadFingerprint);
        }
        if observation.comparison_contract_fingerprint != contract {
            return Err(ClusteredAdvantageError::ContractMismatch);
        }

        let quantum_resource = observation.quantum_resource_per_accepted_answer;
        let classical_resource = observation.classical_resource_per_accepted_answer;
        if !quantum_resource.is_finite()
            || !classical_resource.is_finite()
            || quantum_resource <= 0.0
            || classical_resource <= 0.0
        {
            return Err(ClusteredAdvantageError::InvalidResource);
        }
        if !observation.quantum_quality.is_finite()
            || !observation.classical_quality.is_finite()
        {
            return Err(ClusteredAdvantageError::InvalidQuality);
        }

        let quality_difference = observation.quantum_quality - observation.classical_quality;
        if !quality_difference.is_finite() {
            return Err(ClusteredAdvantageError::InvalidQuality);
        }
        let log_resource_ratio = classical_resource.ln() - quantum_resource.ln();
        grouped
            .entry(observation.family_id.clone())
            .or_default()
            .push((quality_difference, log_resource_ratio));
    }

    if grouped.len() < config.minimum_independent_families {
        return Err(ClusteredAdvantageError::InsufficientIndependentFamilies);
    }

    let mut family_effects = Vec::with_capacity(grouped.len());
    for (family_id, pairs) in grouped {
        let mut quality: Vec<f64> = pairs.iter().map(|pair| pair.0).collect();
        let mut resource: Vec<f64> = pairs.iter().map(|pair| pair.1).collect();
        family_effects.push(FamilyEffect {
            family_id,
            instance_count: pairs.len(),
            median_quality_difference: median(&mut quality),
            median_log_classical_over_quantum_resource: median(&mut resource),
        });
    }

    let quality_threshold = -config.quality_non_inferiority_margin;
    let resource_threshold =
        -(-config.minimum_relative_resource_reduction).ln_1p();
    if !(quality_threshold - config.quality_difference_tie_tolerance).is_finite()
        || !(quality_threshold + config.quality_difference_tie_tolerance).is_finite()
        || !(resource_threshold - config.resource_log_ratio_tie_tolerance).is_finite()
        || !(resource_threshold + config.resource_log_ratio_tie_tolerance).is_finite()
    {
        return Err(ClusteredAdvantageError::InvalidConfig);
    }

    let quality_signs = family_sign_test(
        family_effects
            .iter()
            .map(|family| family.median_quality_difference),
        quality_threshold,
        config.quality_difference_tie_tolerance,
        family_effects.len(),
    );
    let resource_signs = family_sign_test(
        family_effects
            .iter()
            .map(|family| family.median_log_classical_over_quantum_resource),
        resource_threshold,
        config.resource_log_ratio_tie_tolerance,
        family_effects.len(),
    );
    let composite_claim_p_value = quality_signs
        .p_value_one_sided
        .max(resource_signs.p_value_one_sided);

    let mut family_quality: Vec<f64> = family_effects
        .iter()
        .map(|family| family.median_quality_difference)
        .collect();
    let mut family_resource: Vec<f64> = family_effects
        .iter()
        .map(|family| family.median_log_classical_over_quantum_resource)
        .collect();
    let median_family_quality_difference = median(&mut family_quality);
    let median_family_log_resource_ratio = median(&mut family_resource);
    let median_ratio = median_family_log_resource_ratio.exp();

    let quality_loo = leave_one_out_median_range(
        family_effects
            .iter()
            .map(|family| family.median_quality_difference)
            .collect(),
    );
    let resource_loo = leave_one_out_median_range(
        family_effects
            .iter()
            .map(|family| family.median_log_classical_over_quantum_resource)
            .collect(),
    );

    Ok(ClusteredAdvantageReport {
        independent_families: family_effects.len(),
        workload_instances: observations.len(),
        median_family_quality_difference,
        median_family_log_resource_ratio,
        median_family_resource_ratio: (median_ratio.is_finite() && median_ratio > 0.0)
            .then_some(median_ratio),
        quality_non_inferiority: quality_signs,
        resource_superiority: resource_signs,
        composite_claim_p_value,
        leave_one_family_out_quality_range: quality_loo,
        leave_one_family_out_resource_range: resource_loo,
        family_effects,
        comparison_contract_fingerprint: contract.to_owned(),
    })
}

fn family_sign_test(
    values: impl Iterator<Item = f64>,
    margin: f64,
    tolerance: f64,
    family_count: usize,
) -> FamilySignTest {
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    for value in values {
        if value > margin + tolerance {
            wins += 1;
        } else if value < margin - tolerance {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    FamilySignTest {
        wins,
        losses,
        ties,
        independent_families: family_count,
        p_value_one_sided: exact_one_sided_sign_test_p_value(wins, family_count),
    }
}

fn leave_one_out_median_range(values: Vec<f64>) -> (f64, f64) {
    let mut medians = Vec::with_capacity(values.len());
    for removed in 0..values.len() {
        let reduced: Vec<f64> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (index != removed).then_some(*value))
            .collect();
        let mut reduced = reduced;
        medians.push(median(&mut reduced));
    }
    medians.sort_by(f64::total_cmp);
    (medians[0], medians[medians.len() - 1])
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        let lower = values[middle - 1];
        let upper = values[middle];
        lower * 0.5 + upper * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(minimum_independent_families: usize) -> ClusteredAdvantageConfig {
        ClusteredAdvantageConfig {
            minimum_independent_families,
            minimum_relative_resource_reduction: 0.10,
            quality_non_inferiority_margin: 0.02,
            resource_log_ratio_tie_tolerance: 1e-12,
            quality_difference_tie_tolerance: 1e-12,
        }
    }

    fn observation(family: usize, instance: usize, resource_wins: bool) -> ClusteredBenchmarkObservation {
        ClusteredBenchmarkObservation {
            family_id: format!("family-{family:02}"),
            independent_block_id: format!("block-{family:02}-{instance:03}"),
            workload_fingerprint: format!("workload-{family:02}-{instance:03}"),
            comparison_contract_fingerprint: "sha256:cluster-contract".to_owned(),
            quantum_resource_per_accepted_answer: if resource_wins { 80.0 } else { 100.0 },
            classical_resource_per_accepted_answer: if resource_wins { 100.0 } else { 80.0 },
            quantum_quality: 0.995,
            classical_quality: 0.99,
        }
    }

    fn pseudo_replication_fixture() -> Vec<ClusteredBenchmarkObservation> {
        let mut rows = Vec::new();
        for instance in 0..80 {
            rows.push(observation(0, instance, true));
        }
        for family in 1..8 {
            rows.push(observation(family, 0, false));
        }
        rows
    }

    fn next_uniform(state: &mut u32) -> f64 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state as f64 / 4_294_967_296.0
    }

    fn simulate_null_cell(
        cluster_sizes: &[usize],
        icc: f64,
        seed: u32,
        replicates: usize,
    ) -> (usize, usize) {
        let copy_probability = icc.sqrt();
        let mut state = seed;
        let mut naive_false_positives = 0usize;
        let mut family_false_positives = 0usize;
        let instance_count = cluster_sizes.iter().sum::<usize>();

        for _ in 0..replicates {
            let mut instance_wins = 0usize;
            let mut family_wins = 0usize;

            for &cluster_size in cluster_sizes {
                let common = if next_uniform(&mut state) < 0.5 { 1 } else { -1 };
                let mut family_instance_wins = 0usize;
                for _ in 0..cluster_size {
                    let sign = if next_uniform(&mut state) < copy_probability {
                        common
                    } else if next_uniform(&mut state) < 0.5 {
                        1
                    } else {
                        -1
                    };
                    if sign > 0 {
                        instance_wins += 1;
                        family_instance_wins += 1;
                    }
                }
                if family_instance_wins * 2 > cluster_size {
                    family_wins += 1;
                }
            }

            if crate::paired_advantage::exact_one_sided_sign_test_p_value(
                instance_wins,
                instance_count,
            ) <= 0.05
            {
                naive_false_positives += 1;
            }
            if crate::paired_advantage::exact_one_sided_sign_test_p_value(
                family_wins,
                cluster_sizes.len(),
            ) <= 0.05
            {
                family_false_positives += 1;
            }
        }

        (naive_false_positives, family_false_positives)
    }

    fn compact_fixture() -> Vec<ClusteredBenchmarkObservation> {
        let sizes = [3usize, 2, 4, 1, 5, 2, 3, 1];
        let resource_winner_families = [true, true, true, true, true, true, true, false];
        let mut rows = Vec::new();
        for (family, size) in sizes.into_iter().enumerate() {
            for instance in 0..size {
                rows.push(observation(family, instance, resource_winner_families[family]));
            }
        }
        rows
    }

    #[test]
    fn frozen_cluster_null_simulation_reproduces_all_twelve_cells() {
        const MASTER_SEED: u32 = 1_369_948_382;
        const REPLICATES: usize = 10_000;
        let iccs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
        let designs: [&[usize]; 2] = [
            &[3, 3, 3, 3, 3, 3, 3, 3],
            &[21, 1, 1, 1, 1, 1, 1, 1],
        ];
        let expected = [
            (313, 384),
            (457, 332),
            (605, 321),
            (997, 344),
            (1_246, 367),
            (1_392, 360),
            (420, 311),
            (1_695, 353),
            (3_184, 349),
            (4_628, 378),
            (5_010, 387),
            (5_008, 353),
        ];

        let mut cell_index = 0u32;
        for design in designs {
            for icc in iccs {
                let seed = MASTER_SEED.wrapping_add(cell_index.wrapping_mul(65_537));
                let observed = simulate_null_cell(design, icc, seed, REPLICATES);
                assert_eq!(observed, expected[cell_index as usize]);
                cell_index += 1;
            }
        }
        assert_eq!(cell_index, 12);
    }

    #[test]
    fn one_family_contributes_one_vote_regardless_of_instance_count() {
        let report = analyze_clustered_advantage(
            &pseudo_replication_fixture(),
            config(8),
        ).unwrap();
        assert_eq!(report.workload_instances, 87);
        assert_eq!(report.independent_families, 8);
        assert_eq!(report.resource_superiority.wins, 1);
        assert_eq!(report.resource_superiority.losses, 7);
        assert!((report.resource_superiority.p_value_one_sided - 255.0 / 256.0).abs() < 1e-12);
    }

    #[test]
    fn unequal_family_sizes_do_not_change_family_weighting() {
        let report = analyze_clustered_advantage(&compact_fixture(), config(8)).unwrap();
        assert_eq!(report.workload_instances, 21);
        assert_eq!(report.independent_families, 8);
        assert_eq!(report.resource_superiority.wins, 7);
        assert_eq!(report.resource_superiority.losses, 1);
        assert!((report.resource_superiority.p_value_one_sided - 9.0 / 256.0).abs() < 1e-12);
    }

    #[test]
    fn family_effects_use_the_median_of_instance_effects() {
        let mut rows = vec![
            observation(0, 0, true),
            observation(0, 1, false),
            observation(0, 2, true),
            observation(1, 0, false),
            observation(2, 0, true),
        ];
        let report = analyze_clustered_advantage(&rows, config(3)).unwrap();
        assert_eq!(report.family_effects[0].instance_count, 3);
        assert!(report.family_effects[0].median_log_classical_over_quantum_resource > 0.0);
        rows.reverse();
        let reversed = analyze_clustered_advantage(&rows, config(3)).unwrap();
        assert_eq!(report, reversed);
    }

    #[test]
    fn paired_quality_and_resource_claim_uses_intersection_union_p_value() {
        let report = analyze_clustered_advantage(&compact_fixture(), config(8)).unwrap();
        assert_eq!(
            report.composite_claim_p_value,
            report.quality_non_inferiority.p_value_one_sided
                .max(report.resource_superiority.p_value_one_sided)
        );
    }

    #[test]
    fn threshold_ties_remain_in_family_denominator() {
        let mut rows = compact_fixture();
        for row in &mut rows {
            row.quantum_resource_per_accepted_answer = 90.0;
            row.classical_resource_per_accepted_answer = 100.0;
        }
        let report = analyze_clustered_advantage(&rows, config(8)).unwrap();
        assert_eq!(report.resource_superiority.wins, 0);
        assert_eq!(report.resource_superiority.losses, 0);
        assert_eq!(report.resource_superiority.ties, 8);
        assert_eq!(report.resource_superiority.independent_families, 8);
        assert_eq!(report.resource_superiority.p_value_one_sided, 1.0);
    }

    #[test]
    fn insufficient_number_of_families_is_rejected_even_with_many_instances() {
        let mut rows = Vec::new();
        for instance in 0..100 {
            rows.push(observation(0, instance, true));
        }
        rows.push(observation(1, 0, true));
        assert_eq!(
            analyze_clustered_advantage(&rows, config(5)).unwrap_err(),
            ClusteredAdvantageError::InsufficientIndependentFamilies
        );
    }

    #[test]
    fn duplicate_block_ids_are_rejected() {
        let mut rows = compact_fixture();
        rows[1].independent_block_id = rows[0].independent_block_id.clone();
        assert_eq!(
            analyze_clustered_advantage(&rows, config(8)).unwrap_err(),
            ClusteredAdvantageError::DuplicateBlockId
        );
    }

    #[test]
    fn duplicate_workload_fingerprints_are_rejected_across_families() {
        let mut rows = compact_fixture();
        rows[1].workload_fingerprint = rows[0].workload_fingerprint.clone();
        assert_eq!(
            analyze_clustered_advantage(&rows, config(8)).unwrap_err(),
            ClusteredAdvantageError::DuplicateWorkloadFingerprint
        );
    }

    #[test]
    fn mixed_comparison_contracts_are_rejected() {
        let mut rows = compact_fixture();
        rows[1].comparison_contract_fingerprint = "sha256:different".to_owned();
        assert_eq!(
            analyze_clustered_advantage(&rows, config(8)).unwrap_err(),
            ClusteredAdvantageError::ContractMismatch
        );
    }

    #[test]
    fn zero_negative_and_non_finite_resources_are_rejected() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut rows = compact_fixture();
            rows[0].quantum_resource_per_accepted_answer = value;
            assert_eq!(
                analyze_clustered_advantage(&rows, config(8)).unwrap_err(),
                ClusteredAdvantageError::InvalidResource
            );
        }
    }

    #[test]
    fn non_finite_quality_is_rejected() {
        let mut rows = compact_fixture();
        rows[0].quantum_quality = f64::NAN;
        assert_eq!(
            analyze_clustered_advantage(&rows, config(8)).unwrap_err(),
            ClusteredAdvantageError::InvalidQuality
        );
    }

    #[test]
    fn invalid_fractional_reduction_is_rejected() {
        for reduction in [-0.1, 1.0, f64::NAN] {
            let mut cfg = config(8);
            cfg.minimum_relative_resource_reduction = reduction;
            assert_eq!(
                analyze_clustered_advantage(&compact_fixture(), cfg).unwrap_err(),
                ClusteredAdvantageError::InvalidConfig
            );
        }
    }

    #[test]
    fn leave_one_family_out_range_exposes_dependency_on_a_single_family() {
        let mut rows = compact_fixture();
        for row in rows.iter_mut() {
            let family_index = row.family_id[7..].parse::<usize>().unwrap();
            let resource_wins = family_index < 4;
            row.quantum_resource_per_accepted_answer = if resource_wins { 80.0 } else { 100.0 };
            row.classical_resource_per_accepted_answer = if resource_wins { 100.0 } else { 80.0 };
        }
        let report = analyze_clustered_advantage(&rows, config(8)).unwrap();
        assert!(report.leave_one_family_out_resource_range.0 < 0.0);
        assert!(report.leave_one_family_out_resource_range.1 > 0.0);
    }

    #[test]
    fn observation_order_does_not_change_family_result() {
        let rows = compact_fixture();
        let mut reversed = rows.clone();
        reversed.reverse();
        let a = analyze_clustered_advantage(&rows, config(8)).unwrap();
        let b = analyze_clustered_advantage(&reversed, config(8)).unwrap();
        assert_eq!(a, b);
    }
}
