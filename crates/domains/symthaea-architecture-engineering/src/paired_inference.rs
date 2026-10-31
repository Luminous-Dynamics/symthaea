//! Preregistered, landscape-clustered paired inference for search-policy comparisons.
//!
//! Seeds are paired within each landscape. The independent sampling unit is the landscape,
//! not an individual seed/run. Percentile cluster bootstrap intervals resample landscapes;
//! a paired sign-flip randomization test operates on landscape-level paired mean deltas.
//! This is a primary-endpoint decision primitive, not a substitute for a preregistered
//! multi-endpoint analysis plan or external scientific replication.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.architecture-engineering.paired-inference";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedInferenceError {
    pub path: String,
    pub message: String,
}

impl PairedInferenceError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for PairedInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for PairedInferenceError {}

/// Metric orientation is frozen before policy outcomes are observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// Frozen primary endpoint and inferential design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedInferencePlan {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    /// Digest of the sorted, frozen EventTape set used to derive the observations.
    pub campaign_event_tape_set_sha256: String,
    pub baseline_policy_id: String,
    pub challenger_policy_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    /// Smallest improvement considered practically meaningful, in metric units.
    pub minimum_practical_effect: f64,
    /// Two-sided significance level for the paired sign-flip test.
    pub alpha: f64,
    /// Number of cluster-bootstrap replicates; must be at least 1,000.
    pub bootstrap_replicates: u32,
    /// Monte Carlo sign-flip replicates when exact enumeration is too expensive.
    pub randomization_replicates: u32,
    /// Exact sign-flip enumeration is used up to this number of landscape clusters.
    pub exact_randomization_max_clusters: u32,
    pub minimum_landscape_clusters: u32,
    pub minimum_paired_seeds_per_landscape: u32,
    /// Frozen seed for deterministic bootstrap/permutation receipts.
    pub analysis_seed: u64,
}

/// One raw score for one policy on one landscape and seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedMetricObservation {
    pub campaign_id: String,
    pub policy_id: String,
    pub landscape_id: String,
    pub seed: u64,
    pub metric_id: String,
    /// Digest of the archived aggregate artifact that produced this landscape/seed metric.
    pub source_artifact_sha256: String,
    pub value: f64,
}

/// Paired delta within an independent landscape, after averaging its matched seeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapePairedDelta {
    pub landscape_id: String,
    pub paired_seed_count: usize,
    /// Positive is always in the challenger's favour, independent of raw metric direction.
    pub challenger_advantage: f64,
}

/// Result category deliberately avoids equating a non-significant result with equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceDecision {
    EvidenceOfSuperiority,
    EvidenceOfInferiority,
    NoDemonstratedDifference,
}

/// Reproducible statistical receipt for one preregistered primary endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedInferenceSummary {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub baseline_policy_id: String,
    pub challenger_policy_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub landscape_cluster_count: usize,
    pub paired_seed_count: usize,
    pub landscape_win_count: usize,
    pub landscape_tie_count: usize,
    pub landscape_loss_count: usize,
    pub paired_seed_win_count: usize,
    pub paired_seed_tie_count: usize,
    pub paired_seed_loss_count: usize,
    pub mean_challenger_advantage: f64,
    pub median_landscape_advantage: f64,
    pub confidence_level: f64,
    pub cluster_bootstrap_lower: f64,
    pub cluster_bootstrap_upper: f64,
    pub paired_randomization_p_value: f64,
    pub minimum_practical_effect: f64,
    pub decision: InferenceDecision,
    pub landscape_deltas: Vec<LandscapePairedDelta>,
    pub plan_sha256: String,
    pub observations_sha256: String,
    pub inference_receipt_sha256: String,
}

#[derive(Serialize)]
struct InferenceDigestPayload<'a> {
    plan: &'a PairedInferencePlan,
    observations_sha256: &'a str,
    landscape_deltas: &'a [LandscapePairedDelta],
    mean_challenger_advantage: f64,
    median_landscape_advantage: f64,
    cluster_bootstrap_lower: f64,
    cluster_bootstrap_upper: f64,
    paired_randomization_p_value: f64,
    decision: InferenceDecision,
}

/// Analyze one frozen primary endpoint with exact pairing and landscape-clustered uncertainty.
///
/// Requirements:
/// - every observation has the same campaign and metric identity;
/// - each landscape/seed pair has exactly one baseline and one challenger value;
/// - every landscape contributes at least the preregistered number of paired seeds;
/// - enough independent landscapes are present;
/// - all values are finite.
///
/// Repeated seeds inside one landscape do not inflate the independent sample size.
pub fn analyze_paired_metric(
    plan: &PairedInferencePlan,
    observations: &[PairedMetricObservation],
) -> Result<PairedInferenceSummary, PairedInferenceError> {
    validate_plan(plan)?;
    if observations.is_empty() {
        return Err(PairedInferenceError::new(
            "observations",
            "at least one paired observation is required",
        ));
    }

    // landscape -> seed -> policy -> value
    let mut cells = BTreeMap::<String, BTreeMap<u64, BTreeMap<String, f64>>>::new();
    for (index, observation) in observations.iter().enumerate() {
        let path = format!("observations[{index}]");
        if observation.campaign_id != plan.campaign_id {
            return Err(PairedInferenceError::new(
                format!("{path}.campaign_id"),
                "observation campaign differs from the frozen plan",
            ));
        }
        if observation.metric_id != plan.metric_id {
            return Err(PairedInferenceError::new(
                format!("{path}.metric_id"),
                "observation metric differs from the preregistered primary endpoint",
            ));
        }
        if observation.policy_id != plan.baseline_policy_id
            && observation.policy_id != plan.challenger_policy_id
        {
            return Err(PairedInferenceError::new(
                format!("{path}.policy_id"),
                "observation policy is neither the frozen baseline nor challenger",
            ));
        }
        validate_sha256(
            &observation.source_artifact_sha256,
            &format!("{path}.source_artifact_sha256"),
        )?;
        if observation.landscape_id.trim().is_empty() {
            return Err(PairedInferenceError::new(
                format!("{path}.landscape_id"),
                "landscape ID must be non-empty",
            ));
        }
        if !observation.value.is_finite() {
            return Err(PairedInferenceError::new(
                format!("{path}.value"),
                "metric value must be finite",
            ));
        }
        let policy_values = cells
            .entry(observation.landscape_id.clone())
            .or_default()
            .entry(observation.seed)
            .or_default();
        if policy_values
            .insert(observation.policy_id.clone(), observation.value)
            .is_some()
        {
            return Err(PairedInferenceError::new(
                &path,
                "duplicate policy/landscape/seed observation",
            ));
        }
    }

    if cells.len() < plan.minimum_landscape_clusters as usize {
        return Err(PairedInferenceError::new(
            "observations.landscapes",
            format!(
                "found {} independent landscapes; preregistered minimum is {}",
                cells.len(),
                plan.minimum_landscape_clusters
            ),
        ));
    }

    let mut landscape_deltas = Vec::with_capacity(cells.len());
    let mut paired_seed_win_count = 0usize;
    let mut paired_seed_tie_count = 0usize;
    let mut paired_seed_loss_count = 0usize;
    let mut paired_seed_count = 0usize;

    for (landscape_id, seed_cells) in &cells {
        if seed_cells.len() < plan.minimum_paired_seeds_per_landscape as usize {
            return Err(PairedInferenceError::new(
                format!("landscapes[{landscape_id}].seeds"),
                format!(
                    "found {} paired seeds; preregistered minimum is {}",
                    seed_cells.len(),
                    plan.minimum_paired_seeds_per_landscape
                ),
            ));
        }

        let mut landscape_mean = 0.0;
        for (seed_index, (seed, policy_values)) in seed_cells.iter().enumerate() {
            if policy_values.len() != 2
                || !policy_values.contains_key(&plan.baseline_policy_id)
                || !policy_values.contains_key(&plan.challenger_policy_id)
            {
                return Err(PairedInferenceError::new(
                    format!("landscapes[{landscape_id}].seeds[{seed}]"),
                    "every landscape/seed cell must contain exactly one baseline and one challenger observation",
                ));
            }
            let baseline = policy_values[&plan.baseline_policy_id];
            let challenger = policy_values[&plan.challenger_policy_id];
            let advantage = oriented_advantage(
                baseline,
                challenger,
                plan.metric_direction,
            );
            if !advantage.is_finite() {
                return Err(PairedInferenceError::new(
                    format!("landscapes[{landscape_id}].seeds[{seed}].delta"),
                    "paired metric difference overflowed or is non-finite",
                ));
            }
            let count = (seed_index + 1) as f64;
            landscape_mean = landscape_mean * ((count - 1.0) / count) + advantage / count;
            paired_seed_count += 1;
            match compare_with_tolerance(advantage) {
                std::cmp::Ordering::Greater => paired_seed_win_count += 1,
                std::cmp::Ordering::Equal => paired_seed_tie_count += 1,
                std::cmp::Ordering::Less => paired_seed_loss_count += 1,
            }
        }

        landscape_deltas.push(LandscapePairedDelta {
            landscape_id: landscape_id.clone(),
            paired_seed_count: seed_cells.len(),
            challenger_advantage: landscape_mean,
        });
    }

    // Sorting makes the analysis receipt invariant to caller observation order.
    landscape_deltas.sort_by(|left, right| left.landscape_id.cmp(&right.landscape_id));
    let cluster_values: Vec<f64> = landscape_deltas
        .iter()
        .map(|delta| delta.challenger_advantage)
        .collect();
    let mean = stable_mean(&cluster_values);
    let median = median(&mut cluster_values.clone());
    let (bootstrap_lower, bootstrap_upper) = cluster_bootstrap_interval(
        &cluster_values,
        plan.bootstrap_replicates as usize,
        plan.alpha,
        plan.analysis_seed,
    );
    let p_value = paired_sign_flip_p_value(
        &cluster_values,
        mean,
        plan.alpha,
        plan.exact_randomization_max_clusters as usize,
        plan.randomization_replicates as usize,
        plan.analysis_seed ^ 0xa5a5_5a5a_d3c1_b7e9,
    );

    let confidence_level = 1.0 - plan.alpha;
    let decision = if bootstrap_lower > plan.minimum_practical_effect && p_value <= plan.alpha {
        InferenceDecision::EvidenceOfSuperiority
    } else if bootstrap_upper < -plan.minimum_practical_effect && p_value <= plan.alpha {
        InferenceDecision::EvidenceOfInferiority
    } else {
        InferenceDecision::NoDemonstratedDifference
    };
    let landscape_win_count = landscape_deltas
        .iter()
        .filter(|delta| compare_with_tolerance(delta.challenger_advantage).is_gt())
        .count();
    let landscape_tie_count = landscape_deltas
        .iter()
        .filter(|delta| compare_with_tolerance(delta.challenger_advantage).is_eq())
        .count();
    let landscape_loss_count = landscape_deltas
        .iter()
        .filter(|delta| compare_with_tolerance(delta.challenger_advantage).is_lt())
        .count();

    let mut canonical_observations = observations.to_vec();
    canonical_observations.sort_by(|left, right| {
        (
            &left.landscape_id,
            left.seed,
            &left.policy_id,
            &left.metric_id,
        )
            .cmp(&(
                &right.landscape_id,
                right.seed,
                &right.policy_id,
                &right.metric_id,
            ))
    });
    let observations_sha256 = digest_serializable(&canonical_observations)?;
    let plan_sha256 = digest_serializable(plan)?;

    let payload = InferenceDigestPayload {
        plan,
        observations_sha256: &observations_sha256,
        landscape_deltas: &landscape_deltas,
        mean_challenger_advantage: mean,
        median_landscape_advantage: median,
        cluster_bootstrap_lower: bootstrap_lower,
        cluster_bootstrap_upper: bootstrap_upper,
        paired_randomization_p_value: p_value,
        decision,
    };
    let inference_receipt_sha256 = digest_serializable(&payload)?;

    Ok(PairedInferenceSummary {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        campaign_id: plan.campaign_id.clone(),
        baseline_policy_id: plan.baseline_policy_id.clone(),
        challenger_policy_id: plan.challenger_policy_id.clone(),
        metric_id: plan.metric_id.clone(),
        metric_direction: plan.metric_direction,
        landscape_cluster_count: landscape_deltas.len(),
        paired_seed_count,
        landscape_win_count,
        landscape_tie_count,
        landscape_loss_count,
        paired_seed_win_count,
        paired_seed_tie_count,
        paired_seed_loss_count,
        mean_challenger_advantage: mean,
        median_landscape_advantage: median,
        confidence_level,
        cluster_bootstrap_lower: bootstrap_lower,
        cluster_bootstrap_upper: bootstrap_upper,
        paired_randomization_p_value: p_value,
        minimum_practical_effect: plan.minimum_practical_effect,
        decision,
        landscape_deltas,
        plan_sha256,
        observations_sha256,
        inference_receipt_sha256,
    })
}

fn validate_plan(plan: &PairedInferencePlan) -> Result<(), PairedInferenceError> {
    if plan.schema != SCHEMA || plan.version != VERSION {
        return Err(PairedInferenceError::new(
            "plan.schema",
            "unsupported paired-inference schema/version",
        ));
    }
    validate_sha256(
        &plan.campaign_event_tape_set_sha256,
        "plan.campaign_event_tape_set_sha256",
    )?;
    for (path, value) in [
        ("campaign_id", plan.campaign_id.as_str()),
        ("baseline_policy_id", plan.baseline_policy_id.as_str()),
        ("challenger_policy_id", plan.challenger_policy_id.as_str()),
        ("metric_id", plan.metric_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(PairedInferenceError::new(
                format!("plan.{path}"),
                "value must be non-empty",
            ));
        }
    }
    if plan.baseline_policy_id == plan.challenger_policy_id {
        return Err(PairedInferenceError::new(
            "plan.challenger_policy_id",
            "baseline and challenger must be distinct policies",
        ));
    }
    if !plan.minimum_practical_effect.is_finite() || plan.minimum_practical_effect < 0.0 {
        return Err(PairedInferenceError::new(
            "plan.minimum_practical_effect",
            "must be finite and non-negative",
        ));
    }
    if !plan.alpha.is_finite() || !(0.0..0.5).contains(&plan.alpha) {
        return Err(PairedInferenceError::new(
            "plan.alpha",
            "must be finite and strictly between zero and 0.5",
        ));
    }
    if plan.bootstrap_replicates < 1_000 {
        return Err(PairedInferenceError::new(
            "plan.bootstrap_replicates",
            "must be at least 1,000",
        ));
    }
    if plan.randomization_replicates < 1_000 {
        return Err(PairedInferenceError::new(
            "plan.randomization_replicates",
            "must be at least 1,000",
        ));
    }
    if plan.exact_randomization_max_clusters == 0
        || plan.exact_randomization_max_clusters > 20
    {
        return Err(PairedInferenceError::new(
            "plan.exact_randomization_max_clusters",
            "must be between 1 and 20 to bound exact sign-flip enumeration",
        ));
    }
    if plan.minimum_landscape_clusters < 2 || plan.minimum_paired_seeds_per_landscape == 0 {
        return Err(PairedInferenceError::new(
            "plan.minimum_sample",
            "require at least two landscapes and one paired seed per landscape",
        ));
    }
    Ok(())
}

fn oriented_advantage(
    baseline: f64,
    challenger: f64,
    direction: MetricDirection,
) -> f64 {
    match direction {
        MetricDirection::HigherIsBetter => challenger - baseline,
        MetricDirection::LowerIsBetter => baseline - challenger,
    }
}

fn compare_with_tolerance(value: f64) -> std::cmp::Ordering {
    const TOLERANCE: f64 = 1e-12;
    if value > TOLERANCE {
        std::cmp::Ordering::Greater
    } else if value < -TOLERANCE {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }
}

fn stable_mean(values: &[f64]) -> f64 {
    let mut mean = 0.0;
    for (index, value) in values.iter().enumerate() {
        let count = (index + 1) as f64;
        mean = mean * ((count - 1.0) / count) + *value / count;
    }
    mean
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn cluster_bootstrap_interval(
    values: &[f64],
    replicates: usize,
    alpha: f64,
    seed: u64,
) -> (f64, f64) {
    let mut rng = DeterministicRng::new(seed);
    let mut samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut mean = 0.0;
        for sample_index in 0..values.len() {
            let count = (sample_index + 1) as f64;
            mean = mean * ((count - 1.0) / count)
                + values[rng.index(values.len())] / count;
        }
        samples.push(mean);
    }
    samples.sort_by(f64::total_cmp);
    (
        quantile(&samples, alpha / 2.0),
        quantile(&samples, 1.0 - alpha / 2.0),
    )
}

fn paired_sign_flip_p_value(
    values: &[f64],
    observed_mean: f64,
    _alpha: f64,
    exact_max_clusters: usize,
    monte_carlo_replicates: usize,
    seed: u64,
) -> f64 {
    let n = values.len();
    if n <= exact_max_clusters {
        let combinations = 1u64 << n;
        let extreme = (0..combinations)
            .filter(|mask| {
            let mut randomized_mean = 0.0;
            for (index, value) in values.iter().enumerate() {
                let count = (index + 1) as f64;
                let signed = if (*mask & (1u64 << index)) == 0 {
                    -*value
                } else {
                    *value
                };
                randomized_mean =
                    randomized_mean * ((count - 1.0) / count) + signed / count;
            }
                randomized_mean.abs() + 1e-15 >= observed_mean.abs()
            })
            .count();
        extreme as f64 / combinations as f64
    } else {
        let mut rng = DeterministicRng::new(seed);
        let mut extreme = 0usize;
        for _ in 0..monte_carlo_replicates {
            let mut randomized_mean = 0.0;
            for (index, value) in values.iter().enumerate() {
                let count = (index + 1) as f64;
                let signed = if rng.next_u64() & 1 == 0 { -*value } else { *value };
                randomized_mean =
                    randomized_mean * ((count - 1.0) / count) + signed / count;
            }
            if randomized_mean.abs() + 1e-15 >= observed_mean.abs() {
                extreme += 1;
            }
        }
        (extreme + 1) as f64 / (monte_carlo_replicates + 1) as f64
    }
}

/// Small deterministic PRNG for resampling/sign flips. It is not cryptographic; it only
/// makes the statistical resampling sequence reproducible from the preregistered seed.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn index(&mut self, length: usize) -> usize {
        let bound = length as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

fn validate_sha256(value: &str, path: &str) -> Result<(), PairedInferenceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PairedInferenceError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, PairedInferenceError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        PairedInferenceError::new(
            "serialization",
            format!("deterministic inference encoding failed: {error}"),
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(direction: MetricDirection) -> PairedInferencePlan {
        PairedInferencePlan {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: "ar5b4-paired-test".to_owned(),
            campaign_event_tape_set_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            baseline_policy_id: "baseline".to_owned(),
            challenger_policy_id: "challenger".to_owned(),
            metric_id: "verified_improvement_rate".to_owned(),
            metric_direction: direction,
            minimum_practical_effect: 0.01,
            alpha: 0.05,
            bootstrap_replicates: 2_000,
            randomization_replicates: 2_000,
            exact_randomization_max_clusters: 16,
            minimum_landscape_clusters: 6,
            minimum_paired_seeds_per_landscape: 4,
            analysis_seed: 0x5eed,
        }
    }

    fn observations(
        direction: MetricDirection,
        landscape_count: usize,
        seed_count: usize,
        delta: f64,
    ) -> Vec<PairedMetricObservation> {
        let mut result = Vec::new();
        for landscape in 0..landscape_count {
            for seed in 0..seed_count {
                let base = 0.5 + landscape as f64 * 0.001 + seed as f64 * 0.0001;
                let challenger = match direction {
                    MetricDirection::HigherIsBetter => base + delta,
                    MetricDirection::LowerIsBetter => base - delta,
                };
                for (policy_id, value) in [("baseline", base), ("challenger", challenger)] {
                    result.push(PairedMetricObservation {
                        campaign_id: "ar5b4-paired-test".to_owned(),
                        policy_id: policy_id.to_owned(),
                        landscape_id: format!("landscape-{landscape:03}"),
                        seed: seed as u64,
                        metric_id: "verified_improvement_rate".to_owned(),
                        source_artifact_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
                        value,
                    });
                }
            }
        }
        result
    }

    #[test]
    fn paired_clustered_analysis_detects_consistent_practical_superiority() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let rows = observations(MetricDirection::HigherIsBetter, 12, 8, 0.05);
        let summary = analyze_paired_metric(&plan, &rows).unwrap();
        assert_eq!(summary.landscape_cluster_count, 12);
        assert_eq!(summary.paired_seed_count, 96);
        assert_eq!(summary.landscape_win_count, 12);
        assert_eq!(summary.decision, InferenceDecision::EvidenceOfSuperiority);
        assert!(summary.cluster_bootstrap_lower > plan.minimum_practical_effect);
    }

    #[test]
    fn lower_is_better_metrics_are_oriented_before_inference() {
        let plan = plan(MetricDirection::LowerIsBetter);
        let rows = observations(MetricDirection::LowerIsBetter, 12, 8, 0.05);
        let summary = analyze_paired_metric(&plan, &rows).unwrap();
        assert_eq!(summary.decision, InferenceDecision::EvidenceOfSuperiority);
        assert!(summary.mean_challenger_advantage > 0.0);
    }

    #[test]
    fn missing_policy_pair_is_rejected() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let mut rows = observations(MetricDirection::HigherIsBetter, 6, 4, 0.05);
        rows.pop();
        assert!(analyze_paired_metric(&plan, &rows).is_err());
    }

    #[test]
    fn duplicate_policy_landscape_seed_cell_is_rejected() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let mut rows = observations(MetricDirection::HigherIsBetter, 6, 4, 0.05);
        rows.push(rows[0].clone());
        assert!(analyze_paired_metric(&plan, &rows).is_err());
    }

    #[test]
    fn low_landscape_count_cannot_be_replaced_by_many_seeds() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let rows = observations(MetricDirection::HigherIsBetter, 5, 10_000, 0.05);
        assert!(analyze_paired_metric(&plan, &rows).is_err());
    }

    #[test]
    fn input_order_does_not_change_inference_receipt() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let rows = observations(MetricDirection::HigherIsBetter, 12, 8, 0.02);
        let mut permuted = rows.clone();
        permuted.reverse();
        let left = analyze_paired_metric(&plan, &rows).unwrap();
        let right = analyze_paired_metric(&plan, &permuted).unwrap();
        assert_eq!(left.inference_receipt_sha256, right.inference_receipt_sha256);
    }

    #[test]
    fn underpowered_or_noisy_results_are_not_promoted_to_wins() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let mut rows = observations(MetricDirection::HigherIsBetter, 12, 8, 0.0);
        for row in &mut rows {
            if row.policy_id == "challenger" {
                row.value += if row.landscape_id.ends_with("000") { 0.2 } else { -0.02 };
            }
        }
        let summary = analyze_paired_metric(&plan, &rows).unwrap();
        assert_ne!(summary.decision, InferenceDecision::EvidenceOfSuperiority);
    }
    #[test]
    fn seeded_monte_carlo_sign_flip_is_reproducible() {
        let mut plan = plan(MetricDirection::HigherIsBetter);
        plan.exact_randomization_max_clusters = 6;
        let rows = observations(MetricDirection::HigherIsBetter, 12, 8, 0.03);
        let left = analyze_paired_metric(&plan, &rows).unwrap();
        let right = analyze_paired_metric(&plan, &rows).unwrap();
        assert_eq!(
            left.paired_randomization_p_value,
            right.paired_randomization_p_value
        );
        assert_eq!(left.inference_receipt_sha256, right.inference_receipt_sha256);
    }

    #[test]
    fn metric_observation_requires_a_verifiable_source_artifact_digest() {
        let plan = plan(MetricDirection::HigherIsBetter);
        let mut rows = observations(MetricDirection::HigherIsBetter, 6, 4, 0.05);
        rows[0].source_artifact_sha256 = "not-a-digest".to_owned();
        assert!(analyze_paired_metric(&plan, &rows).is_err());
    }

    #[test]
    fn inference_plan_requires_an_event_tape_set_digest() {
        let mut plan = plan(MetricDirection::HigherIsBetter);
        plan.campaign_event_tape_set_sha256 = "missing".to_owned();
        let rows = observations(MetricDirection::HigherIsBetter, 6, 4, 0.05);
        assert!(analyze_paired_metric(&plan, &rows).is_err());
    }

}
