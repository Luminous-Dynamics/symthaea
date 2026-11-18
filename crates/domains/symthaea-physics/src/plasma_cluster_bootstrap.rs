//! Deterministic shot-cluster bootstrap plans for window-level plasma metrics.
//!
//! The resampling unit is the discharge, not the individual time window.
//! For each sampled shot index, benchmark runners must include every window
//! from that shot together. Repeated indices mean repeated whole-shot clusters.
//! This prevents within-shot temporal correlation from being mistaken for
//! independent evidence when estimating uncertainty on window-level metrics.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const SHOT_CLUSTER_BOOTSTRAP_SCHEMA: &str =
    "luminous.plasma-shot-cluster-bootstrap.v0.1";
pub const SHOT_CLUSTER_RESAMPLING_UNIT: &str = "shot_id";
pub const SHOT_CLUSTER_INCLUSION_RULE: &str =
    "include_every_window_for_each_sampled_shot_as_one_indivisible_cluster";
pub const SHOT_CLUSTER_BOOTSTRAP_RNG_ALGORITHM: &str =
    "splitmix64-v1-uniform-rejection";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShotClusterBootstrapError {
    pub message: String,
}

impl ShotClusterBootstrapError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for ShotClusterBootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ShotClusterBootstrapError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShotClusterBootstrapPlanBody {
    pub schema_version: String,
    pub dataset_revision_id: String,
    /// SHA-256 over the exact ordered shot-ID sequence, with length prefixes.
    pub shot_id_sequence_sha256: String,
    pub shot_count: usize,
    pub seed: u64,
    pub replicate_count: usize,
    pub draws_per_replicate: usize,
    pub resampling_unit: String,
    pub inclusion_rule: String,
    pub rng_algorithm: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShotClusterBootstrapPlan {
    pub body: ShotClusterBootstrapPlanBody,
    pub plan_sha256: String,
}

fn bootstrap_error(message: impl Into<String>) -> ShotClusterBootstrapError {
    ShotClusterBootstrapError::new(message)
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, ShotClusterBootstrapError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| bootstrap_error(format!("bootstrap serialization failed: {error}")))?;
    Ok(sha256(&bytes))
}

fn shot_id_sequence_digest(
    shot_ids: &[String],
) -> Result<String, ShotClusterBootstrapError> {
    if shot_ids.is_empty() {
        return Err(bootstrap_error("shot_ids must not be empty"));
    }
    let mut seen = BTreeSet::new();
    let mut canonical = Vec::new();
    for shot_id in shot_ids {
        if shot_id.trim().is_empty() {
            return Err(bootstrap_error("shot_ids must be non-empty"));
        }
        if !seen.insert(shot_id.as_str()) {
            return Err(bootstrap_error("shot_ids must be unique"));
        }
        let bytes = shot_id.as_bytes();
        let length = u64::try_from(bytes.len())
            .map_err(|_| bootstrap_error("shot ID length exceeds supported range"))?;
        canonical.extend_from_slice(&length.to_le_bytes());
        canonical.extend_from_slice(bytes);
    }
    Ok(sha256(&canonical))
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sample_bounded(state: &mut u64, upper_exclusive: u64) -> u64 {
    // Rejection sampling avoids modulo bias for arbitrary shot counts.
    let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
    loop {
        let value = splitmix64_next(state);
        if value >= threshold {
            return value % upper_exclusive;
        }
    }
}

/// Paired confidence interval for a metric difference between two models.
///
/// The direction is `model_b_minus_model_a_delta`. Both model metrics are
/// evaluated on the exact same sampled shot clusters for every replicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedMetricBootstrapInterval {
    pub model_a_metric: f64,
    pub model_b_metric: f64,
    pub model_b_minus_model_a_delta: f64,
    pub delta_lower_95: f64,
    pub delta_upper_95: f64,
    pub confidence_level: f64,
    pub requested_replicates: usize,
    pub valid_replicates: usize,
    pub plan_sha256: String,
    pub rng_algorithm: String,
}

/// Computes a paired 95% percentile bootstrap interval for the difference
/// between two model metrics. The evaluator receives shot indices with
/// replacement; it must expand each selected index to all of that shot's
/// windows, preserving repeated cluster multiplicity.
///
/// The same resampled index vector is passed to both models inside one
/// evaluator invocation. Undefined or non-finite metrics fail closed rather
/// than being silently dropped from the interval.
pub fn paired_shot_cluster_bootstrap<F, E>(
    plan: &ShotClusterBootstrapPlan,
    shot_ids: &[String],
    evaluate_pair: F,
) -> Result<PairedMetricBootstrapInterval, ShotClusterBootstrapError>
where
    F: Fn(&[usize]) -> Result<(f64, f64), E>,
    E: std::fmt::Display,
{
    verify_shot_cluster_bootstrap_plan(plan, &plan.body.dataset_revision_id, shot_ids)?;

    let full_cohort = (0..shot_ids.len()).collect::<Vec<_>>();
    let (model_a_metric, model_b_metric) = evaluate_pair(&full_cohort).map_err(|error| {
        bootstrap_error(format!("full-cohort paired metric evaluation failed: {error}"))
    })?;
    if !model_a_metric.is_finite() || !model_b_metric.is_finite() {
        return Err(bootstrap_error(
            "full-cohort paired metrics must be finite",
        ));
    }

    let mut deltas = Vec::with_capacity(plan.body.replicate_count);
    for replicate_index in 0..plan.body.replicate_count {
        let indices = shot_cluster_bootstrap_indices(plan, shot_ids, replicate_index)?;
        let (metric_a, metric_b) = evaluate_pair(&indices).map_err(|error| {
            bootstrap_error(format!(
                "paired metric evaluation failed for replicate {replicate_index}: {error}"
            ))
        })?;
        if !metric_a.is_finite() || !metric_b.is_finite() {
            return Err(bootstrap_error(format!(
                "paired metrics must be finite for replicate {replicate_index}",
            )));
        }
        deltas.push(metric_b - metric_a);
    }

    if deltas.iter().any(|delta| !delta.is_finite()) {
        return Err(bootstrap_error("paired metric delta overflowed"));
    }
    deltas.sort_by(f64::total_cmp);
    let lower_index = ((deltas.len() as f64 * 0.025).ceil() as usize)
        .saturating_sub(1)
        .min(deltas.len() - 1);
    let upper_index = ((deltas.len() as f64 * 0.975).ceil() as usize)
        .saturating_sub(1)
        .min(deltas.len() - 1);

    Ok(PairedMetricBootstrapInterval {
        model_a_metric,
        model_b_metric,
        model_b_minus_model_a_delta: model_b_metric - model_a_metric,
        delta_lower_95: deltas[lower_index],
        delta_upper_95: deltas[upper_index],
        confidence_level: 0.95,
        requested_replicates: plan.body.replicate_count,
        valid_replicates: deltas.len(),
        plan_sha256: plan.plan_sha256.clone(),
        rng_algorithm: SHOT_CLUSTER_BOOTSTRAP_RNG_ALGORITHM.to_owned(),
    })
}

/// Builds a reproducible plan. Each replicate draws exactly N shot clusters
/// with replacement from N unique shot IDs. The supplied order of shot IDs is
/// part of the plan identity and must be preserved by the benchmark runner.
pub fn build_shot_cluster_bootstrap_plan(
    dataset_revision_id: &str,
    shot_ids: &[String],
    seed: u64,
    replicate_count: usize,
) -> Result<ShotClusterBootstrapPlan, ShotClusterBootstrapError> {
    if dataset_revision_id.trim().is_empty() {
        return Err(bootstrap_error("dataset_revision_id must be non-empty"));
    }
    if replicate_count == 0 {
        return Err(bootstrap_error("replicate_count must be positive"));
    }
    let shot_id_sequence_sha256 = shot_id_sequence_digest(shot_ids)?;
    let body = ShotClusterBootstrapPlanBody {
        schema_version: SHOT_CLUSTER_BOOTSTRAP_SCHEMA.to_owned(),
        dataset_revision_id: dataset_revision_id.to_owned(),
        shot_id_sequence_sha256,
        shot_count: shot_ids.len(),
        seed,
        replicate_count,
        draws_per_replicate: shot_ids.len(),
        resampling_unit: SHOT_CLUSTER_RESAMPLING_UNIT.to_owned(),
        inclusion_rule: SHOT_CLUSTER_INCLUSION_RULE.to_owned(),
        rng_algorithm: SHOT_CLUSTER_BOOTSTRAP_RNG_ALGORITHM.to_owned(),
    };
    let plan_sha256 = digest_serialized(&body)?;
    Ok(ShotClusterBootstrapPlan { body, plan_sha256 })
}

/// Recomputes and validates a plan against the exact dataset revision and
/// ordered shot-ID list supplied by the evaluation runner.
pub fn verify_shot_cluster_bootstrap_plan(
    plan: &ShotClusterBootstrapPlan,
    dataset_revision_id: &str,
    shot_ids: &[String],
) -> Result<(), ShotClusterBootstrapError> {
    let body = &plan.body;
    if body.schema_version != SHOT_CLUSTER_BOOTSTRAP_SCHEMA
        || body.dataset_revision_id != dataset_revision_id
        || dataset_revision_id.trim().is_empty()
        || body.shot_count != shot_ids.len()
        || body.draws_per_replicate != body.shot_count
        || body.replicate_count == 0
        || body.resampling_unit != SHOT_CLUSTER_RESAMPLING_UNIT
        || body.inclusion_rule != SHOT_CLUSTER_INCLUSION_RULE
        || body.rng_algorithm != SHOT_CLUSTER_BOOTSTRAP_RNG_ALGORITHM
    {
        return Err(bootstrap_error(
            "shot-cluster bootstrap plan schema or population metadata mismatch",
        ));
    }
    if body.shot_id_sequence_sha256 != shot_id_sequence_digest(shot_ids)? {
        return Err(bootstrap_error(
            "shot-cluster bootstrap plan does not match the ordered shot-ID cohort",
        ));
    }
    if plan.plan_sha256 != digest_serialized(body)? {
        return Err(bootstrap_error("shot-cluster bootstrap plan digest mismatch"));
    }
    Ok(())
}

/// Produces one replicate as indices into the exact ordered shot-ID list.
///
/// A runner must gather all windows for each selected index. It must not
/// resample windows within a shot. Duplicate shot indices preserve ordinary
/// bootstrap multiplicity and therefore duplicate the entire shot cluster.
pub fn shot_cluster_bootstrap_indices(
    plan: &ShotClusterBootstrapPlan,
    shot_ids: &[String],
    replicate_index: usize,
) -> Result<Vec<usize>, ShotClusterBootstrapError> {
    verify_shot_cluster_bootstrap_plan(plan, &plan.body.dataset_revision_id, shot_ids)?;
    if replicate_index >= plan.body.replicate_count {
        return Err(bootstrap_error(
            "replicate_index is outside the frozen bootstrap plan",
        ));
    }
    let upper = u64::try_from(plan.body.shot_count)
        .map_err(|_| bootstrap_error("shot_count exceeds supported bootstrap range"))?;
    if upper == 0 {
        return Err(bootstrap_error("shot_count must be positive"));
    }
    let replicate_number = u64::try_from(replicate_index)
        .map_err(|_| bootstrap_error("replicate index exceeds supported range"))?;
    let mut state = plan.body.seed
        ^ replicate_number.wrapping_mul(0xd134_2543_de82_ef95);
    let mut indices = Vec::with_capacity(plan.body.draws_per_replicate);
    for _ in 0..plan.body.draws_per_replicate {
        let sampled = sample_bounded(&mut state, upper);
        indices.push(
            usize::try_from(sampled)
                .map_err(|_| bootstrap_error("sampled shot index conversion failed"))?,
        );
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot_ids() -> Vec<String> {
        vec![
            "shot-001".to_owned(),
            "shot-002".to_owned(),
            "shot-003".to_owned(),
            "shot-004".to_owned(),
            "shot-005".to_owned(),
        ]
    }

    #[test]
    fn plan_and_replicate_are_deterministic_for_frozen_order_and_seed() {
        let ids = shot_ids();
        let first = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 42, 100)
            .expect("valid plan");
        let second = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 42, 100)
            .expect("same plan");
        assert_eq!(first, second);
        assert_eq!(
            shot_cluster_bootstrap_indices(&first, &ids, 7).expect("replicate"),
            shot_cluster_bootstrap_indices(&second, &ids, 7).expect("same replicate")
        );
    }

    #[test]
    fn shot_order_dataset_revision_and_seed_are_bound_to_plan_identity() {
        let ids = shot_ids();
        let plan = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 42, 100)
            .expect("valid plan");
        let mut reversed = ids.clone();
        reversed.reverse();
        assert!(verify_shot_cluster_bootstrap_plan(&plan, "tokamark-v1", &reversed).is_err());
        assert!(verify_shot_cluster_bootstrap_plan(&plan, "tokamark-v2", &ids).is_err());

        let different_seed = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 43, 100)
            .expect("different seed");
        assert_ne!(plan.plan_sha256, different_seed.plan_sha256);
    }

    #[test]
    fn each_replicate_draws_n_whole_shot_clusters_with_replacement() {
        let ids = shot_ids();
        let plan = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 9, 100)
            .expect("valid plan");
        for replicate in 0..plan.body.replicate_count {
            let indices =
                shot_cluster_bootstrap_indices(&plan, &ids, replicate).expect("valid replicate");
            assert_eq!(indices.len(), ids.len());
            assert!(indices.iter().all(|index| *index < ids.len()));
        }
    }

    #[test]
    fn paired_metric_bootstrap_uses_the_same_shot_draws_for_both_models() {
        let ids = shot_ids();
        let plan = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 42, 500)
            .expect("paired plan");
        let values_a = [0.1, 0.3, 0.5, 0.7, 0.9];
        let values_b = [0.2, 0.4, 0.6, 0.8, 1.0];

        let interval = paired_shot_cluster_bootstrap(&plan, &ids, |indices| {
            let sum_a = indices.iter().map(|index| values_a[*index]).sum::<f64>();
            let sum_b = indices.iter().map(|index| values_b[*index]).sum::<f64>();
            let denominator = indices.len() as f64;
            Ok::<(f64, f64), String>((sum_a / denominator, sum_b / denominator))
        }).expect("paired bootstrap");
        
        assert!((interval.model_a_metric - 0.5).abs() < 1e-12);
        assert!((interval.model_b_metric - 0.6).abs() < 1e-12);
        assert!((interval.model_b_minus_model_a_delta - 0.1).abs() < 1e-12);
        assert!((interval.delta_lower_95 - 0.1).abs() < 1e-12);
        assert!((interval.delta_upper_95 - 0.1).abs() < 1e-12);
        assert_eq!(interval.requested_replicates, 500);
        assert_eq!(interval.valid_replicates, 500);
        assert_eq!(interval.plan_sha256, plan.plan_sha256);
    }

    #[test]
    fn paired_metric_bootstrap_rejects_non_finite_or_undefined_replicates() {
        let ids = shot_ids();
        let plan = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 7, 100)
            .expect("plan");
        let result = paired_shot_cluster_bootstrap(&plan, &ids, |_indices| {
            Ok::<(f64, f64), String>((0.5, f64::NAN))
        });
        assert!(result.is_err());

        let result = paired_shot_cluster_bootstrap(&plan, &ids, |_indices| {
            Err::<(f64, f64), _>("metric undefined for this sampled cohort")
        });
        assert!(result.is_err());
    }

    #[test]
    fn paired_metric_bootstrap_binds_exact_ordered_population() {
        let ids = shot_ids();
        let plan = build_shot_cluster_bootstrap_plan("tokamark-v1", &ids, 42, 50)
            .expect("plan");
        let mut reversed = ids.clone();
        reversed.reverse();
        let result = paired_shot_cluster_bootstrap(&plan, &reversed, |_indices| {
            Ok::<(f64, f64), String>((0.4, 0.5))
        });
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_and_empty_shot_ids_are_rejected() {
        let duplicate = vec!["shot-1".to_owned(), "shot-1".to_owned()];
        assert!(build_shot_cluster_bootstrap_plan("dataset", &duplicate, 1, 100).is_err());
        let empty = vec!["shot-1".to_owned(), " ".to_owned()];
        assert!(build_shot_cluster_bootstrap_plan("dataset", &empty, 1, 100).is_err());
        assert!(build_shot_cluster_bootstrap_plan("dataset", &[], 1, 100).is_err());
    }

    #[test]
    fn invalid_replicate_configuration_and_indices_are_rejected() {
        let ids = shot_ids();
        assert!(build_shot_cluster_bootstrap_plan("dataset", &ids, 1, 0).is_err());
        let plan =
            build_shot_cluster_bootstrap_plan("dataset", &ids, 1, 5).expect("valid plan");
        assert!(shot_cluster_bootstrap_indices(&plan, &ids, 5).is_err());
    }

    #[test]
    fn protocol_manifest_pins_shot_cluster_bootstrap_contract() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/plasma/event-eval-v0.json"
        ))
        .expect("event evaluation manifest should parse");
        let contract = &manifest["uncertainty_protocols"]["window_level_metrics"];
        assert_eq!(
            contract["resampling_unit"],
            "shot_id"
        );
        assert_eq!(
            contract["bootstrap_schema"],
            SHOT_CLUSTER_BOOTSTRAP_SCHEMA
        );
        assert_eq!(contract["replicate_count"], 2000);
        assert_eq!(
            contract["rng_algorithm"],
            SHOT_CLUSTER_BOOTSTRAP_RNG_ALGORITHM
        );
        assert_eq!(
            contract["paired_model_comparison"],
            "use the exact same bootstrap shot-index plan for every model and report paired metric deltas"
        );
        let paired = &contract["paired_metric_delta_bootstrap"];
        assert_eq!(paired["same_replicate_draws_for_both_models"], true);
        assert_eq!(
            paired["delta_direction"],
            "model_b_metric - model_a_metric"
        );
        assert_eq!(
            paired["schema"],
            "PairedMetricBootstrapInterval"
        );
        assert_eq!(
            paired["undefined_or_non_finite_replicate"],
            "fail closed; do not silently drop undefined metrics"
        );
        assert!(
            contract["forbidden"]
                .as_str()
                .expect("forbidden rule")
                .contains("resampling individual windows independently")
        );
    }

    #[test]
    fn plan_digest_tampering_is_rejected() {
        let ids = shot_ids();
        let mut plan =
            build_shot_cluster_bootstrap_plan("dataset", &ids, 1, 5).expect("valid plan");
        plan.body.seed += 1;
        assert!(verify_shot_cluster_bootstrap_plan(&plan, "dataset", &ids).is_err());
    }
}
