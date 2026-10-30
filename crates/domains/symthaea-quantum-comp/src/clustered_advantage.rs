//! Cluster-aware paired inference for hierarchical benchmark suites.
//!
//! Repeated runs are first reduced within a workload instance. Instances are
//! then reduced to one median effect per independent family/cluster. Exact
//! sign tests operate on clusters, never on instance rows. This deliberately
//! targets an equal-cluster estimand; population weights require a separate,
//! preregistered sampling design and are not inferred from cluster sizes.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::paired_advantage::{
    ExactSignTest, PairedAdvantageError, exact_one_sided_sign_test_p_value,
};

/// One paired quantum/classical workload instance nested in an independent cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredBenchmarkObservation {
    /// Independent family, generator lineage, or other top-level cluster ID.
    pub independent_cluster_id: String,
    /// Stable fingerprint for the cluster; shared by its instances only.
    pub cluster_fingerprint: String,
    /// Unique workload-instance ID.
    pub instance_id: String,
    /// Unique fingerprint for the exact workload instance.
    pub workload_fingerprint: String,
    /// Digest binding workload, objective, quality contract, verifier and resource scope.
    pub comparison_contract_fingerprint: String,
    /// Quantum end-to-end resource per independently verified acceptable answer.
    pub quantum_resource_per_accepted_answer: f64,
    /// Matched classical end-to-end resource per independently verified acceptable answer.
    pub classical_resource_per_accepted_answer: f64,
    /// Quantum output quality; larger values must be better.
    pub quantum_quality: f64,
    /// Classical output quality; larger values must be better.
    pub classical_quality: f64,
}

/// Preregistered margins and the minimum number of independent clusters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusteredAdvantageConfig {
    /// Minimum fractional reduction in quantum resource relative to classical, in [0, 1).
    pub min_relative_resource_reduction: f64,
    /// Largest tolerated quantum quality decrease relative to classical.
    pub quality_non_inferiority_margin: f64,
    /// Tie tolerance for cluster-median log(classical / quantum resource).
    pub resource_log_ratio_tie_tolerance: f64,
    /// Tie tolerance for cluster-median quantum-minus-classical quality.
    pub quality_difference_tie_tolerance: f64,
    /// Preregistered minimum number of independent clusters required.
    pub minimum_independent_clusters: usize,
}

/// One cluster-level effect after the preregistered within-cluster median reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterEffectSummary {
    /// Independent cluster ID.
    pub independent_cluster_id: String,
    /// Number of distinct workload instances represented by this cluster.
    pub instance_count: usize,
    /// Median quantum-minus-classical quality across instances in this cluster.
    pub median_quality_difference: f64,
    /// Median log(classical / quantum resource) across instances in this cluster.
    pub median_log_classical_over_quantum_resource: f64,
}

/// Cluster-level result. P-values are raw and must be Holm-adjusted across claims.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredAdvantageReport {
    /// Number of independent clusters used for inference.
    pub independent_clusters: usize,
    /// Total number of distinct workload instances.
    pub total_instances: usize,
    /// Smallest number of instances in any cluster.
    pub minimum_cluster_size: usize,
    /// Largest number of instances in any cluster.
    pub maximum_cluster_size: usize,
    /// Equal-cluster-weighted median of the cluster-level quality effects.
    pub cluster_balanced_median_quality_difference: f64,
    /// Equal-cluster-weighted median of the cluster-level resource log-ratios.
    pub cluster_balanced_median_log_resource_ratio: f64,
    /// Minimum leave-one-cluster-out median quality difference.
    pub leave_one_cluster_out_min_quality_median: f64,
    /// Maximum leave-one-cluster-out median quality difference.
    pub leave_one_cluster_out_max_quality_median: f64,
    /// Minimum leave-one-cluster-out median resource log-ratio.
    pub leave_one_cluster_out_min_resource_median: f64,
    /// Maximum leave-one-cluster-out median resource log-ratio.
    pub leave_one_cluster_out_max_resource_median: f64,
    /// Every leave-one-cluster-out median preserves the quality margin.
    pub quality_margin_survives_every_cluster_deletion: bool,
    /// Every leave-one-cluster-out median preserves the practical resource margin.
    pub resource_margin_survives_every_cluster_deletion: bool,
    /// Exact one-sided sign test across independent clusters for quality non-inferiority.
    pub quality_non_inferiority: ExactSignTest,
    /// Exact one-sided sign test across independent clusters for resource superiority.
    pub resource_superiority: ExactSignTest,
    /// Intersection-union p-value: max(quality p, resource p).
    pub composite_claim_p_value: f64,
    /// Shared comparison-contract digest.
    pub comparison_contract_fingerprint: String,
    /// Auditable cluster effects in deterministic cluster-ID order.
    pub cluster_effects: Vec<ClusterEffectSummary>,
}

/// Errors from malformed cluster hierarchy or underpowered input contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteredAdvantageError {
    /// No paired workload instances were supplied.
    EmptyObservations,
    /// Required identifier is blank.
    EmptyIdentifier,
    /// An instance ID or workload fingerprint appears more than once.
    DuplicateInstance,
    /// A cluster fingerprint is reused by a different cluster ID.
    DuplicateClusterFingerprint,
    /// Instances assigned to the same cluster carry different fingerprints.
    ClusterFingerprintMismatch,
    /// Workload, objective, verifier or resource comparison contract differs.
    ContractMismatch,
    /// Resource values must be finite and strictly positive.
    InvalidResource,
    /// Quality values and their paired difference must be finite.
    InvalidQuality,
    /// A margin, tolerance or minimum-cluster configuration is invalid.
    InvalidConfig,
    /// The synthetic null-stress configuration is invalid.
    InvalidStressConfig,
    /// Fewer independent clusters were supplied than preregistered.
    InsufficientIndependentClusters,
}

impl std::fmt::Display for ClusteredAdvantageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyObservations => "no clustered paired workload instances",
            Self::EmptyIdentifier => "required cluster, instance or contract identifier is blank",
            Self::DuplicateInstance => "duplicate workload instance ID or fingerprint",
            Self::DuplicateClusterFingerprint => "cluster fingerprint is reused by another cluster",
            Self::ClusterFingerprintMismatch => "one cluster ID maps to multiple fingerprints",
            Self::ContractMismatch => "clustered comparison contract fingerprints differ",
            Self::InvalidResource => "resource values must be finite and strictly positive",
            Self::InvalidQuality => "quality values and paired differences must be finite",
            Self::InvalidConfig => "invalid margin, tolerance or minimum-cluster configuration",
            Self::InvalidStressConfig => "invalid cluster-null stress configuration",
            Self::InsufficientIndependentClusters => "fewer independent clusters than preregistered",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ClusteredAdvantageError {}

/// Analyze paired observations using one median effect per independent cluster.
///
/// The cluster-level sign tests assume independent clusters and a defensible
/// sampling frame. Cluster sizes do not determine inferential weight: each
/// cluster contributes one sign. The within-cluster median is the preregistered
/// reduction used by this API; repeated shots/seeds must already be reduced
/// within each workload instance before observations are supplied.
pub fn analyze_clustered_advantage(
    observations: &[ClusteredBenchmarkObservation],
    config: ClusteredAdvantageConfig,
) -> Result<ClusteredAdvantageReport, ClusteredAdvantageError> {
    if observations.is_empty() {
        return Err(ClusteredAdvantageError::EmptyObservations);
    }
    if config.minimum_independent_clusters < 2
        || !config.min_relative_resource_reduction.is_finite()
        || config.min_relative_resource_reduction < 0.0
        || config.min_relative_resource_reduction >= 1.0
        || !config.quality_non_inferiority_margin.is_finite()
        || config.quality_non_inferiority_margin < 0.0
        || !config.resource_log_ratio_tie_tolerance.is_finite()
        || config.resource_log_ratio_tie_tolerance < 0.0
        || !config.quality_difference_tie_tolerance.is_finite()
        || config.quality_difference_tie_tolerance < 0.0
    {
        return Err(ClusteredAdvantageError::InvalidConfig);
    }

    let quality_threshold = -config.quality_non_inferiority_margin;
    let resource_threshold = -(-config.min_relative_resource_reduction).ln_1p();
    if !(quality_threshold - config.quality_difference_tie_tolerance).is_finite()
        || !(quality_threshold + config.quality_difference_tie_tolerance).is_finite()
        || !(resource_threshold - config.resource_log_ratio_tie_tolerance).is_finite()
        || !(resource_threshold + config.resource_log_ratio_tie_tolerance).is_finite()
    {
        return Err(ClusteredAdvantageError::InvalidConfig);
    }

    let contract = observations[0].comparison_contract_fingerprint.trim();
    if contract.is_empty() {
        return Err(ClusteredAdvantageError::EmptyIdentifier);
    }

    let mut instance_ids = HashSet::with_capacity(observations.len());
    let mut workload_fingerprints = HashSet::with_capacity(observations.len());
    let mut cluster_fingerprints: HashMap<&str, &str> = HashMap::new();
    let mut clusters: BTreeMap<String, (String, Vec<&ClusteredBenchmarkObservation>)> =
        BTreeMap::new();

    for observation in observations {
        if observation.independent_cluster_id.trim().is_empty()
            || observation.cluster_fingerprint.trim().is_empty()
            || observation.instance_id.trim().is_empty()
            || observation.workload_fingerprint.trim().is_empty()
            || observation.comparison_contract_fingerprint.trim().is_empty()
        {
            return Err(ClusteredAdvantageError::EmptyIdentifier);
        }
        if !instance_ids.insert(observation.instance_id.as_str())
            || !workload_fingerprints.insert(observation.workload_fingerprint.as_str())
        {
            return Err(ClusteredAdvantageError::DuplicateInstance);
        }
        if observation.comparison_contract_fingerprint != contract {
            return Err(ClusteredAdvantageError::ContractMismatch);
        }

        match cluster_fingerprints.get(observation.cluster_fingerprint.as_str()) {
            Some(existing_id) if *existing_id != observation.independent_cluster_id => {
                return Err(ClusteredAdvantageError::DuplicateClusterFingerprint);
            }
            Some(_) => {}
            None => {
                cluster_fingerprints.insert(
                    observation.cluster_fingerprint.as_str(),
                    observation.independent_cluster_id.as_str(),
                );
            }
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
            || !(observation.quantum_quality - observation.classical_quality).is_finite()
        {
            return Err(ClusteredAdvantageError::InvalidQuality);
        }

        let entry = clusters
            .entry(observation.independent_cluster_id.clone())
            .or_insert_with(|| (observation.cluster_fingerprint.clone(), Vec::new()));
        if entry.0 != observation.cluster_fingerprint {
            return Err(ClusteredAdvantageError::ClusterFingerprintMismatch);
        }
        entry.1.push(observation);
    }

    if clusters.len() < config.minimum_independent_clusters {
        return Err(ClusteredAdvantageError::InsufficientIndependentClusters);
    }

    let mut cluster_effects = Vec::with_capacity(clusters.len());
    let mut cluster_quality = Vec::with_capacity(clusters.len());
    let mut cluster_resource = Vec::with_capacity(clusters.len());
    let mut minimum_cluster_size = usize::MAX;
    let mut maximum_cluster_size = 0usize;

    for (cluster_id, (_, rows)) in &clusters {
        let mut quality_differences: Vec<f64> = rows
            .iter()
            .map(|row| row.quantum_quality - row.classical_quality)
            .collect();
        let mut resource_log_ratios: Vec<f64> = rows
            .iter()
            .map(|row| {
                row.classical_resource_per_accepted_answer.ln()
                    - row.quantum_resource_per_accepted_answer.ln()
            })
            .collect();

        let quality_median = median(&mut quality_differences);
        let resource_median = median(&mut resource_log_ratios);
        if !quality_median.is_finite() || !resource_median.is_finite() {
            return Err(ClusteredAdvantageError::InvalidQuality);
        }

        minimum_cluster_size = minimum_cluster_size.min(rows.len());
        maximum_cluster_size = maximum_cluster_size.max(rows.len());
        cluster_quality.push(quality_median);
        cluster_resource.push(resource_median);
        cluster_effects.push(ClusterEffectSummary {
            independent_cluster_id: cluster_id.clone(),
            instance_count: rows.len(),
            median_quality_difference: quality_median,
            median_log_classical_over_quantum_resource: resource_median,
        });
    }

    let quality_non_inferiority = exact_cluster_margin_sign_test(
        &cluster_quality,
        quality_threshold,
        config.quality_difference_tie_tolerance,
    );
    let resource_superiority = exact_cluster_margin_sign_test(
        &cluster_resource,
        resource_threshold,
        config.resource_log_ratio_tie_tolerance,
    );
    let composite_claim_p_value = quality_non_inferiority
        .p_value_one_sided
        .max(resource_superiority.p_value_one_sided);

    let cluster_balanced_median_quality_difference = median(&mut cluster_quality.clone());
    let cluster_balanced_median_log_resource_ratio = median(&mut cluster_resource.clone());

    let quality_leave_one_out = leave_one_cluster_out_range(&cluster_quality);
    let resource_leave_one_out = leave_one_cluster_out_range(&cluster_resource);

    Ok(ClusteredAdvantageReport {
        independent_clusters: clusters.len(),
        total_instances: observations.len(),
        minimum_cluster_size,
        maximum_cluster_size,
        cluster_balanced_median_quality_difference,
        cluster_balanced_median_log_resource_ratio,
        leave_one_cluster_out_min_quality_median: quality_leave_one_out.0,
        leave_one_cluster_out_max_quality_median: quality_leave_one_out.1,
        leave_one_cluster_out_min_resource_median: resource_leave_one_out.0,
        leave_one_cluster_out_max_resource_median: resource_leave_one_out.1,
        quality_margin_survives_every_cluster_deletion: quality_leave_one_out.0
            > quality_threshold + config.quality_difference_tie_tolerance,
        resource_margin_survives_every_cluster_deletion: resource_leave_one_out.0
            > resource_threshold + config.resource_log_ratio_tie_tolerance,
        quality_non_inferiority,
        resource_superiority,
        composite_claim_p_value,
        comparison_contract_fingerprint: contract.to_owned(),
        cluster_effects,
    })
}

fn exact_cluster_margin_sign_test(
    values: &[f64],
    margin: f64,
    tolerance: f64,
) -> ExactSignTest {
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    for &value in values {
        if value > margin + tolerance {
            wins += 1;
        } else if value < margin - tolerance {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    ExactSignTest {
        wins,
        losses,
        ties,
        effective_n: values.len(),
        p_value_one_sided: exact_one_sided_sign_test_p_value(wins, values.len()),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        values[middle - 1] * 0.5 + values[middle] * 0.5
    } else {
        values[middle]
    }
}

fn leave_one_cluster_out_range(values: &[f64]) -> (f64, f64) {
    let mut medians = Vec::with_capacity(values.len());
    for omitted in 0..values.len() {
        let mut remaining: Vec<f64> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (index != omitted).then_some(*value))
            .collect();
        medians.push(median(&mut remaining));
    }
    medians.sort_by(f64::total_cmp);
    (medians[0], medians[medians.len() - 1])
}

/// Configuration for a reproducible cluster-dependence null stress experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterNullStressConfig {
    /// Independent workload families generated per replicate.
    pub independent_clusters: usize,
    /// Paired workload instances nested within each family.
    pub instances_per_cluster: usize,
    /// Intraclass correlations to evaluate, each in [0, 1].
    pub intraclass_correlations: Vec<f64>,
    /// Independent null replicates per ICC cell.
    pub replicates: usize,
    /// Deterministic Mulberry32 seed.
    pub seed: u32,
    /// One-sided test rejection threshold, in (0, 1).
    pub significance_level: f64,
}

/// Wilson score interval for a Monte Carlo rejection probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonRateInterval {
    /// Lower 95% Wilson score bound.
    pub lower: f64,
    /// Upper 95% Wilson score bound.
    pub upper: f64,
}

/// Empirical null rejection rates for one intraclass-correlation cell.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterNullStressPoint {
    /// Intraclass correlation used to generate nested observations.
    pub intraclass_correlation: f64,
    /// Number of null simulation replicates.
    pub replicates: usize,
    /// Number of naive instance-level sign-test rejections.
    pub naive_instance_rejections: usize,
    /// Number of cluster-median sign-test rejections.
    pub cluster_level_rejections: usize,
    /// Naive instance-level empirical type-I error rate.
    pub naive_instance_type_i_rate: f64,
    /// Cluster-level empirical type-I error rate.
    pub cluster_level_type_i_rate: f64,
    /// Monte Carlo uncertainty interval for naive instance-level rate.
    pub naive_instance_wilson_95: WilsonRateInterval,
    /// Monte Carlo uncertainty interval for cluster-level rate.
    pub cluster_level_wilson_95: WilsonRateInterval,
}

/// Reproducible null-stress result. The model is synthetic, not an empirical ICC estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterNullStressReport {
    /// Seed used for deterministic pseudorandom draws.
    pub seed: u32,
    /// Independent clusters in each replicate.
    pub independent_clusters: usize,
    /// Instances nested within each cluster.
    pub instances_per_cluster: usize,
    /// One-sided rejection threshold.
    pub significance_level: f64,
    /// Results in the same order as the preregistered ICC input.
    pub points: Vec<ClusterNullStressPoint>,
}

/// Compare naive instance-level and cluster-median sign tests under a null.
///
/// Each cluster receives a latent success probability
/// 0.5 +/- 0.5 * sqrt(ICC) with equal probability. Conditional instance signs
/// are Bernoulli draws from that probability, giving marginal success
/// probability 0.5 and within-cluster Bernoulli ICC equal to the configured
/// value. Independent clusters receive independent latent probabilities.
/// Random continuous magnitudes avoid deterministic median ties.
///
/// The naive test treats every nested instance as independent. The cluster
/// test first takes one median signed effect per cluster. This stress harness
/// estimates type-I error for this particular synthetic exchangeable model; it
/// does not establish performance for every dependence structure.
pub fn run_cluster_null_stress(
    config: &ClusterNullStressConfig,
) -> Result<ClusterNullStressReport, ClusteredAdvantageError> {
    if config.independent_clusters < 2
        || config.instances_per_cluster < 2
        || config.replicates == 0
        || !config.significance_level.is_finite()
        || !(0.0..1.0).contains(&config.significance_level)
        || config.intraclass_correlations.is_empty()
        || config.intraclass_correlations.iter().any(|icc| {
            !icc.is_finite() || !(0.0..=1.0).contains(icc)
        })
    {
        return Err(ClusteredAdvantageError::InvalidStressConfig);
    }

    let mut random = Mulberry32::new(config.seed);
    let mut points = Vec::with_capacity(config.intraclass_correlations.len());
    for &icc in &config.intraclass_correlations {
        let mut naive_rejections = 0usize;
        let mut cluster_rejections = 0usize;
        let total_instances = config.independent_clusters * config.instances_per_cluster;

        for _ in 0..config.replicates {
            let mut total_instance_wins = 0usize;
            let mut cluster_median_wins = 0usize;
            for _ in 0..config.independent_clusters {
                let latent_direction = if random.next_f64() < 0.5 { 1.0 } else { -1.0 };
                let instance_win_probability = 0.5 + 0.5 * icc.sqrt() * latent_direction;
                let mut signed_effects = Vec::with_capacity(config.instances_per_cluster);
                for _ in 0..config.instances_per_cluster {
                    let is_win = random.next_f64() < instance_win_probability;
                    if is_win {
                        total_instance_wins += 1;
                    }
                    let magnitude = 0.5 + random.next_f64();
                    signed_effects.push(if is_win { magnitude } else { -magnitude });
                }
                signed_effects.sort_by(f64::total_cmp);
                let cluster_median = median_pair(
                    signed_effects[config.instances_per_cluster / 2 - 1],
                    signed_effects[config.instances_per_cluster / 2],
                    config.instances_per_cluster,
                );
                if cluster_median > 0.0 {
                    cluster_median_wins += 1;
                }
            }

            let naive_p = exact_one_sided_sign_test_p_value(
                total_instance_wins,
                total_instances,
            );
            let cluster_p = exact_one_sided_sign_test_p_value(
                cluster_median_wins,
                config.independent_clusters,
            );
            if naive_p <= config.significance_level {
                naive_rejections += 1;
            }
            if cluster_p <= config.significance_level {
                cluster_rejections += 1;
            }
        }

        points.push(ClusterNullStressPoint {
            intraclass_correlation: icc,
            replicates: config.replicates,
            naive_instance_rejections: naive_rejections,
            cluster_level_rejections: cluster_rejections,
            naive_instance_type_i_rate: naive_rejections as f64 / config.replicates as f64,
            cluster_level_type_i_rate: cluster_rejections as f64 / config.replicates as f64,
            naive_instance_wilson_95: wilson_interval(naive_rejections, config.replicates),
            cluster_level_wilson_95: wilson_interval(cluster_rejections, config.replicates),
        });
    }

    Ok(ClusterNullStressReport {
        seed: config.seed,
        independent_clusters: config.independent_clusters,
        instances_per_cluster: config.instances_per_cluster,
        significance_level: config.significance_level,
        points,
    })
}

fn median_pair(lower: f64, upper: f64, count: usize) -> f64 {
    if count % 2 == 0 {
        lower * 0.5 + upper * 0.5
    } else {
        upper
    }
}

fn wilson_interval(successes: usize, trials: usize) -> WilsonRateInterval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let half_width = z
        * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()
        / denominator;
    WilsonRateInterval {
        lower: (center - half_width).max(0.0),
        upper: (center + half_width).min(1.0),
    }
}

struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        let output = value ^ (value >> 14);
        output as f64 / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(minimum_independent_clusters: usize) -> ClusteredAdvantageConfig {
        ClusteredAdvantageConfig {
            min_relative_resource_reduction: 0.10,
            quality_non_inferiority_margin: 0.02,
            resource_log_ratio_tie_tolerance: 1e-12,
            quality_difference_tie_tolerance: 1e-12,
            minimum_independent_clusters,
        }
    }

    fn row(
        cluster: usize,
        instance: usize,
        q_resource: f64,
        c_resource: f64,
        q_quality: f64,
        c_quality: f64,
    ) -> ClusteredBenchmarkObservation {
        ClusteredBenchmarkObservation {
            independent_cluster_id: format!("family-{cluster:02}"),
            cluster_fingerprint: format!("sha256:family-{cluster:02}"),
            instance_id: format!("instance-{cluster:02}-{instance:03}"),
            workload_fingerprint: format!("sha256:workload-{cluster:02}-{instance:03}"),
            comparison_contract_fingerprint: "sha256:shared-contract".to_owned(),
            quantum_resource_per_accepted_answer: q_resource,
            classical_resource_per_accepted_answer: c_resource,
            quantum_quality: q_quality,
            classical_quality: c_quality,
        }
    }

    fn replicated_null_fixture(
        positive_families: usize,
        family_count: usize,
        instances_per_family: usize,
    ) -> Vec<ClusteredBenchmarkObservation> {
        let mut observations = Vec::new();
        for family in 0..family_count {
            let quantum_wins = family < positive_families;
            let (q_resource, c_resource) = if quantum_wins {
                (8.0, 10.0)
            } else {
                (11.0, 10.0)
            };
            for instance in 0..instances_per_family {
                observations.push(row(
                    family,
                    instance,
                    q_resource,
                    c_resource,
                    0.995,
                    0.99,
                ));
            }
        }
        observations
    }

    #[test]
    fn replicated_instances_count_as_one_independent_cluster_each() {
        let observations = replicated_null_fixture(6, 10, 10);
        let report = analyze_clustered_advantage(&observations, config(10)).unwrap();
        assert_eq!(report.total_instances, 100);
        assert_eq!(report.independent_clusters, 10);
        assert_eq!(report.minimum_cluster_size, 10);
        assert_eq!(report.maximum_cluster_size, 10);
        assert_eq!(report.resource_superiority.wins, 6);
        assert_eq!(report.resource_superiority.losses, 4);
        assert!((report.resource_superiority.p_value_one_sided - 386.0 / 1024.0).abs() < 1e-12);
        assert!(report.composite_claim_p_value > 0.05);
    }

    #[test]
    fn perfect_cluster_correlation_exposes_instance_level_false_discovery() {
        let observations = replicated_null_fixture(6, 10, 10);
        let report = analyze_clustered_advantage(&observations, config(10)).unwrap();
        let naive_instance_p = exact_one_sided_sign_test_p_value(60, 100);
        assert!(naive_instance_p < 0.05);
        assert!(report.resource_superiority.p_value_one_sided > 0.05);
    }

    #[test]
    fn target_family_imbalance_does_not_change_equal_cluster_weighting() {
        let mut observations = Vec::new();
        observations.push(row(0, 0, 8.0, 10.0, 0.995, 0.99));
        observations.push(row(1, 0, 8.0, 10.0, 0.995, 0.99));
        for instance in 0..20 {
            observations.push(row(2, instance, 11.0, 10.0, 0.995, 0.99));
        }
        let report = analyze_clustered_advantage(&observations, config(3)).unwrap();
        assert_eq!(report.total_instances, 22);
        assert_eq!(report.independent_clusters, 3);
        assert_eq!(report.minimum_cluster_size, 1);
        assert_eq!(report.maximum_cluster_size, 20);
        assert_eq!(report.resource_superiority.wins, 2);
        assert_eq!(report.resource_superiority.losses, 1);
        assert!((report.cluster_balanced_median_log_resource_ratio - (10.0f64 / 8.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn leave_one_cluster_out_diagnostic_detects_effect_instability() {
        let mut observations = Vec::new();
        for cluster in 0..3 {
            let (q, c) = if cluster < 2 { (8.0, 10.0) } else { (11.0, 10.0) };
            observations.push(row(cluster, 0, q, c, 0.995, 0.99));
        }
        let report = analyze_clustered_advantage(&observations, config(3)).unwrap();
        assert!(!report.resource_margin_survives_every_cluster_deletion);
        assert!(
            report.leave_one_cluster_out_min_resource_median
                < -(-config(3).min_relative_resource_reduction).ln_1p()
        );
    }

    #[test]
    fn duplicate_instance_id_is_rejected() {
        let mut observations = replicated_null_fixture(6, 3, 2);
        observations[1].instance_id = observations[0].instance_id.clone();
        assert_eq!(
            analyze_clustered_advantage(&observations, config(3)).unwrap_err(),
            ClusteredAdvantageError::DuplicateInstance
        );
    }

    #[test]
    fn duplicate_workload_fingerprint_is_rejected() {
        let mut observations = replicated_null_fixture(6, 3, 2);
        observations[1].workload_fingerprint = observations[0].workload_fingerprint.clone();
        assert_eq!(
            analyze_clustered_advantage(&observations, config(3)).unwrap_err(),
            ClusteredAdvantageError::DuplicateInstance
        );
    }

    #[test]
    fn duplicate_cluster_fingerprint_is_rejected() {
        let mut observations = replicated_null_fixture(2, 3, 1);
        observations[1].cluster_fingerprint = observations[0].cluster_fingerprint.clone();
        assert_eq!(
            analyze_clustered_advantage(&observations, config(3)).unwrap_err(),
            ClusteredAdvantageError::DuplicateClusterFingerprint
        );
    }

    #[test]
    fn cluster_fingerprint_mismatch_within_cluster_is_rejected() {
        let mut observations = replicated_null_fixture(2, 3, 1);
        observations[1].independent_cluster_id = observations[0].independent_cluster_id.clone();
        assert_eq!(
            analyze_clustered_advantage(&observations, config(2)).unwrap_err(),
            ClusteredAdvantageError::ClusterFingerprintMismatch
        );
    }

    #[test]
    fn comparison_contract_mismatch_is_rejected() {
        let mut observations = replicated_null_fixture(2, 3, 1);
        observations[1].comparison_contract_fingerprint = "sha256:other".to_owned();
        assert_eq!(
            analyze_clustered_advantage(&observations, config(3)).unwrap_err(),
            ClusteredAdvantageError::ContractMismatch
        );
    }

    #[test]
    fn too_few_clusters_fails_closed() {
        let observations = replicated_null_fixture(2, 3, 2);
        assert_eq!(
            analyze_clustered_advantage(&observations, config(4)).unwrap_err(),
            ClusteredAdvantageError::InsufficientIndependentClusters
        );
    }

    #[test]
    fn invalid_resource_and_margin_are_rejected() {
        let mut observations = replicated_null_fixture(2, 3, 2);
        observations[0].quantum_resource_per_accepted_answer = 0.0;
        assert_eq!(
            analyze_clustered_advantage(&observations, config(3)).unwrap_err(),
            ClusteredAdvantageError::InvalidResource
        );
        let observations = replicated_null_fixture(2, 3, 2);
        assert_eq!(
            analyze_clustered_advantage(
                &observations,
                ClusteredAdvantageConfig {
                    min_relative_resource_reduction: 1.0,
                    ..config(3)
                }
            )
            .unwrap_err(),
            ClusteredAdvantageError::InvalidConfig
        );
    }

    #[test]
    fn permutation_of_instances_and_clusters_does_not_change_report() {
        let forward = replicated_null_fixture(6, 10, 3);
        let mut reversed = forward.clone();
        reversed.reverse();
        let a = analyze_clustered_advantage(&forward, config(10)).unwrap();
        let b = analyze_clustered_advantage(&reversed, config(10)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cluster_null_stress_is_deterministic_and_reports_intervals() {
        let config = ClusterNullStressConfig {
            independent_clusters: 5,
            instances_per_cluster: 6,
            intraclass_correlations: vec![0.0, 0.6],
            replicates: 200,
            seed: 0x516016,
            significance_level: 0.05,
        };
        let first = run_cluster_null_stress(&config).unwrap();
        let second = run_cluster_null_stress(&config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.points.len(), 2);
        for point in &first.points {
            assert!((0.0..=1.0).contains(&point.naive_instance_type_i_rate));
            assert!((0.0..=1.0).contains(&point.cluster_level_type_i_rate));
            assert!(point.naive_instance_wilson_95.lower <= point.naive_instance_wilson_95.upper);
            assert!(point.cluster_level_wilson_95.lower <= point.cluster_level_wilson_95.upper);
        }
    }

    #[test]
    fn cluster_null_stress_rejects_invalid_icc() {
        let config = ClusterNullStressConfig {
            independent_clusters: 5,
            instances_per_cluster: 6,
            intraclass_correlations: vec![-0.1],
            replicates: 10,
            seed: 1,
            significance_level: 0.05,
        };
        assert_eq!(
            run_cluster_null_stress(&config).unwrap_err(),
            ClusteredAdvantageError::InvalidStressConfig
        );
    }

    #[test]
    fn exact_cluster_resource_boundary_is_not_a_win() {
        let observations = (0..5)
            .map(|cluster| row(cluster, 0, 9.0, 10.0, 0.995, 0.99))
            .collect::<Vec<_>>();
        let report = analyze_clustered_advantage(&observations, config(5)).unwrap();
        assert_eq!(report.resource_superiority.wins, 0);
        assert_eq!(report.resource_superiority.ties, 5);
    }
}
