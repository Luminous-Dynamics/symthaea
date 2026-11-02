//! Claim-scoped Measurement-as-Distribution protocol and robustness reporting.
//!
//! Implements P = (C, G, Phi, E, R): construct, generation, validated
//! perturbations, repeated evaluation, and distributional reporting.
//! The kernel consumes evaluator-produced observations; it does not run models,
//! generate tasks, or certify that a declared perturbation preserves a construct.

use crate::{
    causal_analysis::{MeanIntervalV1, mean_interval},
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MEASUREMENT_PROTOCOL_SCHEMA: &str = "symthaea.measurement_distribution_protocol.v1";
pub const MEASUREMENT_RUN_BUNDLE_SCHEMA: &str = "symthaea.measurement_distribution_run_bundle.v1";
pub const MEASUREMENT_REPORT_SCHEMA: &str = "symthaea.measurement_distribution_report.v1";
pub const MAX_AXES: usize = 32;
pub const MAX_FAMILIES_PER_AXIS: usize = 64;
pub const MAX_MODELS: usize = 256;
pub const MAX_RUNS_PER_MODEL: usize = 1_000;
pub const MAX_INSTANCES: usize = 500_000;
pub const MAX_OBSERVATIONS: usize = 2_000_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimScopeV1 {
    FixedConfiguration,
    TaskFamily,
    CrossDomain,
    ModelFamily,
    DeploymentDistribution,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PerturbationAxisKindV1 {
    SurfaceForm,
    Semantic,
    SolutionRecipe,
    Difficulty,
    Language,
    Context,
    Prompt,
    ToolEnvironment,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidatorKindV1 {
    Formal,
    Verifier,
    Human,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndependentUnitKindV1 {
    ModelPairReplicate,
    SourceCluster,
    BaseItem,
    EvaluationRun,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationDecisionV1 {
    Baseline,
    Validated,
    Rejected,
    Indeterminate,
    FallbackApplied,
    ExecutionFailed,
    Missing,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConstructSpecV1 {
    pub construct_id: String,
    pub construct_version: String,
    pub target_description: String,
    pub operational_definition: String,
    pub invariants: Vec<String>,
    pub permitted_nuisance_variation: Vec<String>,
    pub prohibited_claims: Vec<String>,
    /// When true, prompt variation can be a primary construct axis rather than
    /// a separately reported auxiliary analysis.
    pub instruction_following_is_target: bool,
    pub claim_scope: ClaimScopeV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorSpecV1 {
    pub generator_id: String,
    pub generator_version: String,
    pub generator_config_sha256: String,
    pub source_dataset_sha256: String,
    pub generated_manifest_sha256: String,
    pub seed_commitment_sha256: String,
    pub checkpoint_freeze_receipt_sha256: Option<String>,
    pub fresh_instances_after_checkpoint_freeze: bool,
    pub generated_after_checkpoint_freeze: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PerturbationFamilySpecV1 {
    pub family_id: String,
    pub condition_id: String,
    pub transformer_sha256: String,
    pub invariance_claim: String,
    pub acceptance_rule: String,
    pub validator_kind: ValidatorKindV1,
    pub validation_protocol_sha256: String,
    pub minimum_acceptance_permille: u16,
    pub fallback_condition_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PerturbationAxisSpecV1 {
    pub axis_id: String,
    pub kind: PerturbationAxisKindV1,
    pub description: String,
    pub baseline_condition_id: String,
    pub auxiliary_only: bool,
    pub families: Vec<PerturbationFamilySpecV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSpecV1 {
    pub harness_sha256: String,
    pub prompt_template_sha256: String,
    pub scorer_sha256: String,
    pub decoding_config_sha256: String,
    pub evaluator_sha256: String,
    pub metric_id: String,
    pub higher_is_better: bool,
    pub independent_unit_kind: IndependentUnitKindV1,
    pub minimum_model_count: u16,
    pub minimum_runs_per_model: u16,
    pub minimum_independent_units_for_delta: u16,
    pub resample_instances_across_runs: bool,
    pub resample_perturbations_across_runs: bool,
    pub resample_model_inference_across_runs: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReportingSpecV1 {
    pub minimum_coverage_permille: u16,
    pub minimum_acceptance_permille: u16,
    /// Quantiles in basis points, e.g. 50, 5000, 9950.
    pub quantiles_basis_points: [u16; 5],
    pub require_rank_stability: bool,
    pub minimum_models_for_rank_stability: u16,
    pub multiple_comparison_policy: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementProtocolV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub construct: ConstructSpecV1,
    pub generator: GeneratorSpecV1,
    pub axes: Vec<PerturbationAxisSpecV1>,
    pub evaluation: EvaluationSpecV1,
    pub reporting: ReportingSpecV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementInstanceV1 {
    pub item_id: String,
    pub task_family: String,
    pub source_domain_id: String,
    pub source_cluster_id: String,
    pub source_lineage_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementInstanceManifestV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub instances: Vec<MeasurementInstanceV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementModelV1 {
    pub model_id: String,
    pub model_family_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementRunV1 {
    pub model_id: String,
    pub run_id: String,
    /// E.g. an independently trained model-pair replicate, source cluster,
    /// base item, or evaluation run, as declared by the protocol.
    pub independent_unit_id: String,
    pub seed_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementObservationV1 {
    pub model_id: String,
    pub run_id: String,
    pub independent_unit_id: String,
    pub axis_id: String,
    pub condition_id: String,
    pub item_id: String,
    pub task_family: String,
    pub source_cluster_id: String,
    pub decision: ObservationDecisionV1,
    pub validation_receipt_sha256: Option<String>,
    pub perturbation_seed_commitment_sha256: Option<String>,
    pub external_egress_receipt_sha256: Option<String>,
    pub score: Option<f64>,
    pub runtime_milliseconds: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub external_calls: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementRunBundleV1 {
    pub schema: String,
    pub protocol_sha256: String,
    pub instance_manifest: MeasurementInstanceManifestV1,
    pub instance_manifest_sha256: String,
    pub models: Vec<MeasurementModelV1>,
    pub runs: Vec<MeasurementRunV1>,
    pub observations: Vec<MeasurementObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionProfileCellV1 {
    pub model_id: String,
    pub model_family_id: String,
    pub task_family: String,
    pub axis_id: String,
    pub condition_id: String,
    pub expected_count: u64,
    pub observation_count: u64,
    pub scored_count: u64,
    pub rejected_count: u64,
    pub indeterminate_count: u64,
    pub fallback_count: u64,
    pub execution_failed_count: u64,
    pub missing_count: u64,
    pub coverage_permille: f64,
    pub validation_acceptance_permille: Option<f64>,
    pub mean_score: Option<f64>,
    pub sample_standard_deviation: Option<f64>,
    pub quantiles: [Option<f64>; 5],
    pub paired_observation_count: u64,
    pub paired_independent_unit_count: u64,
    pub paired_delta_vs_baseline: Option<MeanIntervalV1>,
    pub paired_delta_status: String,
    pub mean_runtime_milliseconds: Option<f64>,
    pub mean_input_tokens: Option<f64>,
    pub mean_output_tokens: Option<f64>,
    pub external_call_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankStabilityV1 {
    pub task_family: String,
    pub axis_id: String,
    pub condition_id: String,
    pub model_count: u64,
    pub kendall_tau_b: Option<f64>,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRobustnessSummaryV1 {
    pub model_id: String,
    pub model_family_id: String,
    pub worst_axis_mean_delta: Option<f64>,
    pub minimum_axis_coverage_permille: f64,
    pub minimum_axis_acceptance_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDistributionReportV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub protocol_sha256: String,
    pub instance_manifest_sha256: String,
    pub run_bundle_sha256: String,
    pub claim_scope: ClaimScopeV1,
    pub measurement_paradigm: String,
    pub model_count: u64,
    pub model_family_count: u64,
    pub task_family_count: u64,
    pub source_domain_count: u64,
    pub run_count: u64,
    pub independent_unit_kind: IndependentUnitKindV1,
    pub independent_unit_count: u64,
    pub instance_count: u64,
    pub observation_count: u64,
    pub distributional_evidence_eligible: bool,
    pub claim_scope_eligible: bool,
    /// Paired deltas are multiplied by metric direction so positive always means better.
    pub paired_delta_positive_is_better: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub rank_stability_status: String,
    pub profile_cells: Vec<DistributionProfileCellV1>,
    pub rank_stability: Vec<RankStabilityV1>,
    pub model_summaries: Vec<ModelRobustnessSummaryV1>,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ObservationKey {
    model_id: String,
    run_id: String,
    axis_id: String,
    condition_id: String,
    item_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ProfileKey {
    model_id: String,
    task_family: String,
    axis_id: String,
    condition_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PairKey {
    model_id: String,
    run_id: String,
    axis_id: String,
    item_id: String,
}

#[derive(Debug, Clone)]
struct Counts {
    expected: u64,
    observation: u64,
    scored: u64,
    rejected: u64,
    indeterminate: u64,
    fallback: u64,
    failed: u64,
    missing: u64,
}

impl Counts {
    fn new(expected: u64) -> Self {
        Self {
            expected,
            observation: 0,
            scored: 0,
            rejected: 0,
            indeterminate: 0,
            fallback: 0,
            failed: 0,
            missing: 0,
        }
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

pub fn compute_measurement_protocol_sha256(
    protocol: &MeasurementProtocolV1,
) -> Result<String, String> {
    if protocol.schema != MEASUREMENT_PROTOCOL_SCHEMA {
        return Err(format!(
            "protocol schema must be {MEASUREMENT_PROTOCOL_SCHEMA}"
        ));
    }
    let bytes = serde_json::to_vec(protocol)
        .map_err(|error| format!("cannot serialize measurement protocol: {error}"))?;
    Ok(hash_domain(b"measurement-distribution-protocol.v1", &bytes))
}

pub fn compute_instance_manifest_sha256(
    manifest: &MeasurementInstanceManifestV1,
) -> Result<String, String> {
    if manifest.schema != "symthaea.measurement_distribution_instance_manifest.v1" {
        return Err("instance manifest schema is invalid".to_owned());
    }
    let bytes = serde_json::to_vec(manifest)
        .map_err(|error| format!("cannot serialize measurement instance manifest: {error}"))?;
    Ok(hash_domain(
        b"measurement-distribution-instance-manifest.v1",
        &bytes,
    ))
}

pub fn validate_measurement_protocol(protocol: &MeasurementProtocolV1) -> Result<(), String> {
    if protocol.schema != MEASUREMENT_PROTOCOL_SCHEMA {
        return Err(format!(
            "protocol schema must be {MEASUREMENT_PROTOCOL_SCHEMA}"
        ));
    }
    if !valid_id(&protocol.benchmark_id, 128)
        || !valid_id(&protocol.benchmark_version, 64)
        || !valid_id(&protocol.construct.construct_id, 128)
        || !valid_id(&protocol.construct.construct_version, 64)
        || protocol.construct.target_description.trim().is_empty()
        || protocol.construct.operational_definition.trim().is_empty()
    {
        return Err("benchmark and construct identity/definition fields are required".to_owned());
    }
    if protocol.construct.invariants.is_empty() || protocol.construct.prohibited_claims.is_empty() {
        return Err("construct invariants and prohibited claims must be explicit".to_owned());
    }
    let generator = &protocol.generator;
    if !valid_id(&generator.generator_id, 128) || !valid_id(&generator.generator_version, 64) {
        return Err("generator identity is invalid".to_owned());
    }
    for (digest, label) in [
        (
            &generator.generator_config_sha256,
            "generator_config_sha256",
        ),
        (&generator.source_dataset_sha256, "source_dataset_sha256"),
        (
            &generator.generated_manifest_sha256,
            "generated_manifest_sha256",
        ),
        (&generator.seed_commitment_sha256, "seed_commitment_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    if generator.fresh_instances_after_checkpoint_freeze {
        if !generator.generated_after_checkpoint_freeze {
            return Err(
                "fresh instances require explicit post-freeze generation evidence".to_owned(),
            );
        }
        let receipt = generator
            .checkpoint_freeze_receipt_sha256
            .as_deref()
            .ok_or_else(|| "fresh instances require a checkpoint-freeze receipt".to_owned())?;
        require_digest(receipt, "checkpoint_freeze_receipt_sha256")?;
    } else if generator.checkpoint_freeze_receipt_sha256.is_some() {
        let receipt = generator
            .checkpoint_freeze_receipt_sha256
            .as_deref()
            .unwrap();
        require_digest(receipt, "checkpoint_freeze_receipt_sha256")?;
    }
    if protocol.axes.is_empty() || protocol.axes.len() > MAX_AXES {
        return Err(format!("axis count must be in 1..={MAX_AXES}"));
    }
    let mut axis_ids = BTreeSet::new();
    let mut condition_ids = BTreeSet::new();
    for axis in &protocol.axes {
        if !valid_id(&axis.axis_id, 64)
            || axis.description.trim().is_empty()
            || !valid_id(&axis.baseline_condition_id, 64)
            || !axis_ids.insert(axis.axis_id.as_str())
        {
            return Err("axis identifiers/descriptions must be valid and unique".to_owned());
        }
        if !condition_ids.insert((axis.axis_id.as_str(), axis.baseline_condition_id.as_str())) {
            return Err("axis baseline condition is duplicated".to_owned());
        }
        if axis.kind == PerturbationAxisKindV1::Prompt
            && !protocol.construct.instruction_following_is_target
            && !axis.auxiliary_only
        {
            return Err(
                "prompt sensitivity must be auxiliary unless instruction following is the target"
                    .to_owned(),
            );
        }
        if axis.families.len() > MAX_FAMILIES_PER_AXIS
            || (axis.families.is_empty()
                && protocol.construct.claim_scope != ClaimScopeV1::FixedConfiguration)
        {
            return Err(format!(
                "axis {} must declare perturbation families unless this is a fixed-configuration protocol",
                axis.axis_id
            ));
        }
        let mut family_ids = BTreeSet::new();
        for family in &axis.families {
            if !valid_id(&family.family_id, 64)
                || !valid_id(&family.condition_id, 64)
                || !family_ids.insert(family.family_id.as_str())
                || family.condition_id == axis.baseline_condition_id
                || !condition_ids.insert((axis.axis_id.as_str(), family.condition_id.as_str()))
                || family.invariance_claim.trim().is_empty()
                || family.acceptance_rule.trim().is_empty()
                || !(500..=1_000).contains(&family.minimum_acceptance_permille)
            {
                return Err(format!(
                    "axis {} has an invalid or duplicate perturbation family",
                    axis.axis_id
                ));
            }
            for (digest, label) in [
                (&family.transformer_sha256, "transformer_sha256"),
                (
                    &family.validation_protocol_sha256,
                    "validation_protocol_sha256",
                ),
            ] {
                require_digest(digest, label)?;
            }
            if let Some(fallback) = &family.fallback_condition_id {
                if !valid_id(fallback, 64)
                    || fallback == &family.condition_id
                    || fallback == &axis.baseline_condition_id
                {
                    return Err("fallback condition ID must be distinct from baseline and primary condition".to_owned());
                }
            }
        }
        for family in &axis.families {
            if let Some(fallback) = &family.fallback_condition_id
                && !condition_ids.contains(&(axis.axis_id.as_str(), fallback.as_str()))
            {
                return Err(format!(
                    "fallback condition {fallback} is not declared in axis {}",
                    axis.axis_id
                ));
            }
        }
    }
    let evaluation = &protocol.evaluation;
    for (digest, label) in [
        (&evaluation.harness_sha256, "harness_sha256"),
        (&evaluation.prompt_template_sha256, "prompt_template_sha256"),
        (&evaluation.scorer_sha256, "scorer_sha256"),
        (&evaluation.decoding_config_sha256, "decoding_config_sha256"),
        (&evaluation.evaluator_sha256, "evaluator_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    if !valid_id(&evaluation.metric_id, 128)
        || evaluation.minimum_model_count == 0
        || evaluation.minimum_runs_per_model < 1
        || evaluation.minimum_independent_units_for_delta < 2
    {
        return Err(
            "evaluation metric and minimum model/run/independent-unit counts are invalid"
                .to_owned(),
        );
    }
    if protocol.reporting.minimum_coverage_permille > 1_000
        || protocol.reporting.minimum_acceptance_permille > 1_000
        || protocol.reporting.minimum_models_for_rank_stability < 2
        || protocol
            .reporting
            .quantiles_basis_points
            .iter()
            .any(|value| *value > 10_000)
        || protocol
            .reporting
            .quantiles_basis_points
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || protocol.reporting.multiple_comparison_policy != "holm_when_tests_are_reported"
    {
        return Err(
            "reporting thresholds, quantiles or multiplicity policy are invalid".to_owned(),
        );
    }
    if protocol.reporting.require_rank_stability
        && protocol.reporting.minimum_models_for_rank_stability < 3
    {
        return Err("rank stability requires at least three models".to_owned());
    }
    Ok(())
}

fn quantile(sorted: &[f64], basis_points: u16) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let position = basis_points as f64 / 10_000.0 * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

fn sample_sd(values: &[f64], mean: f64) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some(variance.sqrt())
}

fn kendall_tau_b(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let mut concordant = 0_u64;
    let mut discordant = 0_u64;
    let mut tied_left = 0_u64;
    let mut tied_right = 0_u64;
    for i in 0..left.len() {
        for j in (i + 1)..left.len() {
            let left_order = left[i].total_cmp(&left[j]);
            let right_order = right[i].total_cmp(&right[j]);
            match (left_order, right_order) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {}
                (std::cmp::Ordering::Equal, _) => tied_left += 1,
                (_, std::cmp::Ordering::Equal) => tied_right += 1,
                (a, b) if a == b => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    let comparable_left = concordant + discordant + tied_left;
    let comparable_right = concordant + discordant + tied_right;
    let denominator = (comparable_left as f64 * comparable_right as f64).sqrt();
    if denominator == 0.0 {
        None
    } else {
        Some((concordant as f64 - discordant as f64) / denominator)
    }
}

fn score_summary(
    values: &[f64],
    quantiles: [u16; 5],
) -> (Option<f64>, Option<f64>, [Option<f64>; 5]) {
    if values.is_empty() {
        return (None, None, [None; 5]);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let qs = [
        quantile(&sorted, quantiles[0]),
        quantile(&sorted, quantiles[1]),
        quantile(&sorted, quantiles[2]),
        quantile(&sorted, quantiles[3]),
        quantile(&sorted, quantiles[4]),
    ];
    (Some(mean), sample_sd(values, mean), qs)
}

/// Validate the complete protocol/run evidence bundle and emit distributional
/// profiles. The score CI uses equally weighted independent-unit means; item
/// and perturbation copies do not become independent replicates.
pub fn analyze_measurement_distribution(
    protocol: &MeasurementProtocolV1,
    bundle: &MeasurementRunBundleV1,
) -> Result<MeasurementDistributionReportV1, String> {
    validate_measurement_protocol(protocol)?;
    if bundle.schema != MEASUREMENT_RUN_BUNDLE_SCHEMA {
        return Err(format!(
            "run bundle schema must be {MEASUREMENT_RUN_BUNDLE_SCHEMA}"
        ));
    }
    let protocol_bytes = serde_json::to_vec(protocol)
        .map_err(|error| format!("cannot serialize measurement protocol: {error}"))?;
    let protocol_sha = hash_domain(b"measurement-distribution-protocol.v1", &protocol_bytes);
    if bundle.protocol_sha256 != protocol_sha {
        return Err("run bundle protocol digest does not match the supplied protocol".to_owned());
    }
    if bundle.instance_manifest.schema != "symthaea.measurement_distribution_instance_manifest.v1"
        || bundle.instance_manifest.benchmark_id != protocol.benchmark_id
        || bundle.instance_manifest.benchmark_version != protocol.benchmark_version
        || bundle.instance_manifest.instances.is_empty()
        || bundle.instance_manifest.instances.len() > MAX_INSTANCES
    {
        return Err("instance manifest identity or size is invalid".to_owned());
    }
    let manifest_bytes = serde_json::to_vec(&bundle.instance_manifest)
        .map_err(|error| format!("cannot serialize instance manifest: {error}"))?;
    let manifest_sha = hash_domain(
        b"measurement-distribution-instance-manifest.v1",
        &manifest_bytes,
    );
    if bundle.instance_manifest_sha256 != manifest_sha {
        return Err("instance manifest digest mismatch".to_owned());
    }
    if protocol.generator.generated_manifest_sha256 != manifest_sha {
        return Err(
            "generator's committed output manifest does not match the evaluator instance manifest"
                .to_owned(),
        );
    }
    if bundle.models.is_empty() || bundle.models.len() > MAX_MODELS {
        return Err(format!("model count must be in 1..={MAX_MODELS}"));
    }
    if bundle.observations.is_empty() || bundle.observations.len() > MAX_OBSERVATIONS {
        return Err(format!(
            "observation count must be in 1..={MAX_OBSERVATIONS}"
        ));
    }

    let mut models = BTreeMap::<String, &MeasurementModelV1>::new();
    for model in &bundle.models {
        if !valid_id(&model.model_id, 128)
            || !valid_id(&model.model_family_id, 128)
            || models.insert(model.model_id.clone(), model).is_some()
        {
            return Err("model IDs/family IDs must be valid and model IDs unique".to_owned());
        }
        require_digest(&model.model_checkpoint_sha256, "model_checkpoint_sha256")?;
        require_digest(&model.tokenizer_sha256, "tokenizer_sha256")?;
    }

    let mut runs = BTreeMap::<(String, String), &MeasurementRunV1>::new();
    let mut runs_by_model = BTreeMap::<String, u64>::new();
    for run in &bundle.runs {
        if !models.contains_key(&run.model_id)
            || !valid_id(&run.run_id, 128)
            || !valid_id(&run.independent_unit_id, 128)
            || runs
                .insert((run.model_id.clone(), run.run_id.clone()), run)
                .is_some()
        {
            return Err(
                "run identity is invalid, duplicated or references an unknown model".to_owned(),
            );
        }
        require_digest(&run.seed_commitment_sha256, "run seed_commitment_sha256")?;
        *runs_by_model.entry(run.model_id.clone()).or_default() += 1;
    }
    if bundle.runs.is_empty() || bundle.runs.len() > MAX_RUNS_PER_MODEL * MAX_MODELS {
        return Err("run count is empty or exceeds the supported bound".to_owned());
    }
    for model_id in models.keys() {
        if runs_by_model.get(model_id).copied().unwrap_or(0)
            < protocol.evaluation.minimum_runs_per_model as u64
        {
            return Err(format!(
                "model {model_id} does not meet minimum evaluation-run count"
            ));
        }
    }

    let mut instances = BTreeMap::<String, &MeasurementInstanceV1>::new();
    for item in &bundle.instance_manifest.instances {
        if !valid_id(&item.item_id, 128)
            || !valid_id(&item.task_family, 64)
            || !valid_id(&item.source_domain_id, 128)
            || !valid_id(&item.source_cluster_id, 128)
            || instances.insert(item.item_id.clone(), item).is_some()
        {
            return Err(
                "instance IDs/task-family/source-cluster fields are invalid or duplicated"
                    .to_owned(),
            );
        }
        require_digest(&item.source_lineage_sha256, "source_lineage_sha256")?;
    }

    let axes: BTreeMap<&str, &PerturbationAxisSpecV1> = protocol
        .axes
        .iter()
        .map(|axis| (axis.axis_id.as_str(), axis))
        .collect();
    let mut conditions_by_axis = BTreeMap::<String, BTreeSet<String>>::new();
    for axis in &protocol.axes {
        let mut conditions = BTreeSet::new();
        conditions.insert(axis.baseline_condition_id.clone());
        for family in &axis.families {
            conditions.insert(family.condition_id.clone());
        }
        conditions_by_axis.insert(axis.axis_id.clone(), conditions);
    }
    let expected_observation_count = bundle
        .runs
        .len()
        .checked_mul(bundle.instance_manifest.instances.len())
        .and_then(|value| value.checked_mul(conditions_by_axis.values().map(BTreeSet::len).sum()))
        .ok_or_else(|| "expected observation count overflows".to_owned())?;
    if expected_observation_count != bundle.observations.len() {
        return Err(format!(
            "run bundle must include exactly one explicit observation for every model/run/item/axis/condition combination: expected {expected_observation_count}, received {}",
            bundle.observations.len()
        ));
    }

    let mut observation_map = BTreeMap::<ObservationKey, &MeasurementObservationV1>::new();
    let mut profile_observations = BTreeMap::<ProfileKey, Vec<&MeasurementObservationV1>>::new();
    let mut independent_units = BTreeSet::<(String, String)>::new();

    for observation in &bundle.observations {
        if !models.contains_key(&observation.model_id) {
            return Err("observation references an unknown model".to_owned());
        }
        let run = runs
            .get(&(observation.model_id.clone(), observation.run_id.clone()))
            .ok_or_else(|| "observation references an unknown model run".to_owned())?;
        let item = instances
            .get(&observation.item_id)
            .ok_or_else(|| "observation references an unknown manifest item".to_owned())?;
        let axis = axes
            .get(observation.axis_id.as_str())
            .ok_or_else(|| "observation references an unknown perturbation axis".to_owned())?;
        if observation.task_family != item.task_family
            || observation.source_cluster_id != item.source_cluster_id
        {
            return Err(
                "observation task family/source cluster differs from its instance manifest"
                    .to_owned(),
            );
        }
        let independent_unit_matches = match protocol.evaluation.independent_unit_kind {
            IndependentUnitKindV1::ModelPairReplicate => {
                observation.independent_unit_id == run.independent_unit_id
            }
            IndependentUnitKindV1::SourceCluster => {
                observation.independent_unit_id == item.source_cluster_id
            }
            IndependentUnitKindV1::BaseItem => observation.independent_unit_id == item.item_id,
            IndependentUnitKindV1::EvaluationRun => observation.independent_unit_id == run.run_id,
        };
        if !independent_unit_matches {
            return Err(
                "observation independent unit does not match the protocol's declared unit kind"
                    .to_owned(),
            );
        }
        let is_baseline = observation.condition_id == axis.baseline_condition_id;
        let family = axis
            .families
            .iter()
            .find(|candidate| candidate.condition_id == observation.condition_id);
        if !is_baseline && family.is_none() {
            return Err("observation condition is not declared on its axis".to_owned());
        }
        match (is_baseline, observation.decision, observation.score) {
            (true, ObservationDecisionV1::Baseline, Some(score)) if score.is_finite() => {
                if observation.validation_receipt_sha256.is_some()
                    || observation.perturbation_seed_commitment_sha256.is_some()
                {
                    return Err("baseline observations must not claim perturbation validation or seed receipts".to_owned());
                }
            }
            (true, _, _) => {
                return Err(
                    "baseline condition requires Baseline decision and a finite score".to_owned(),
                );
            }
            (false, ObservationDecisionV1::Validated, Some(score)) if score.is_finite() => {
                let seed = observation
                    .perturbation_seed_commitment_sha256
                    .as_deref()
                    .ok_or_else(|| {
                        "perturbation observation requires a seed commitment".to_owned()
                    })?;
                require_digest(seed, "perturbation_seed_commitment_sha256")?;
                let receipt = observation
                    .validation_receipt_sha256
                    .as_deref()
                    .ok_or_else(|| {
                        "validated perturbation requires a validation receipt".to_owned()
                    })?;
                require_digest(receipt, "validation_receipt_sha256")?;
            }
            (false, ObservationDecisionV1::FallbackApplied, Some(score)) if score.is_finite() => {
                let seed = observation
                    .perturbation_seed_commitment_sha256
                    .as_deref()
                    .ok_or_else(|| {
                        "fallback observation requires a perturbation seed commitment".to_owned()
                    })?;
                require_digest(seed, "perturbation_seed_commitment_sha256")?;
                let family = family.unwrap();
                if family.fallback_condition_id.is_none() {
                    return Err(
                        "fallback observation has no declared fallback condition".to_owned()
                    );
                }
                let receipt = observation
                    .validation_receipt_sha256
                    .as_deref()
                    .ok_or_else(|| {
                        "fallback application requires a validation receipt".to_owned()
                    })?;
                require_digest(receipt, "validation_receipt_sha256")?;
            }
            (false, ObservationDecisionV1::Rejected, None)
            | (false, ObservationDecisionV1::Indeterminate, None)
            | (false, ObservationDecisionV1::ExecutionFailed, None)
            | (false, ObservationDecisionV1::Missing, None) => {}
            _ => return Err("perturbation decision and score fields are inconsistent".to_owned()),
        }
        if let Some(score) = observation.score {
            if !score.is_finite() {
                return Err("observation score must be finite".to_owned());
            }
        }
        if let Some(receipt) = &observation.validation_receipt_sha256 {
            require_digest(receipt, "validation_receipt_sha256")?;
        }
        match (
            observation.external_calls,
            observation.external_egress_receipt_sha256.as_deref(),
        ) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(
                    "zero external calls must not carry an external-egress receipt".to_owned(),
                );
            }
            (_, Some(receipt)) => require_digest(receipt, "external_egress_receipt_sha256")?,
            (_, None) => {
                return Err(
                    "external calls require a destination/egress evidence receipt".to_owned(),
                );
            }
        }
        independent_units.insert((
            observation.model_id.clone(),
            observation.independent_unit_id.clone(),
        ));
        let key = ObservationKey {
            model_id: observation.model_id.clone(),
            run_id: observation.run_id.clone(),
            axis_id: observation.axis_id.clone(),
            condition_id: observation.condition_id.clone(),
            item_id: observation.item_id.clone(),
        };
        if observation_map.insert(key, observation).is_some() {
            return Err(
                "duplicate observation for a model/run/axis/condition/item combination".to_owned(),
            );
        }
        let profile_key = ProfileKey {
            model_id: observation.model_id.clone(),
            task_family: observation.task_family.clone(),
            axis_id: observation.axis_id.clone(),
            condition_id: observation.condition_id.clone(),
        };
        profile_observations
            .entry(profile_key)
            .or_default()
            .push(observation);
    }

    for model_id in models.keys() {
        for run_id in runs.keys().filter(|(id, _)| id == model_id) {
            for axis in &protocol.axes {
                for condition in conditions_by_axis
                    .get(&axis.axis_id)
                    .ok_or_else(|| "condition matrix is missing an axis".to_owned())?
                {
                    for item_id in instances.keys() {
                        let key = ObservationKey {
                            model_id: model_id.clone(),
                            run_id: run_id.1.clone(),
                            axis_id: axis.axis_id.clone(),
                            condition_id: condition.clone(),
                            item_id: item_id.clone(),
                        };
                        if !observation_map.contains_key(&key) {
                            return Err(
                                "explicit observation matrix contains a missing key".to_owned()
                            );
                        }
                    }
                }
            }
        }
    }

    let mut profile_cells = Vec::new();
    let mut cell_lookup = BTreeMap::<ProfileKey, DistributionProfileCellV1>::new();
    for (key, observations) in &profile_observations {
        let axis = axes
            .get(key.axis_id.as_str())
            .ok_or_else(|| "profile axis disappeared".to_owned())?;
        let family_instance_count = instances
            .values()
            .filter(|item| item.task_family == key.task_family)
            .count() as u64;
        let expected =
            runs_by_model.get(&key.model_id).copied().unwrap_or(0) * family_instance_count;
        let scored: Vec<&MeasurementObservationV1> = observations
            .iter()
            .copied()
            .filter(|row| row.score.is_some())
            .collect();
        let scores: Vec<f64> = scored.iter().filter_map(|row| row.score).collect();
        let (mean, sd, quantiles) =
            score_summary(&scores, protocol.reporting.quantiles_basis_points);
        let rejected = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::Rejected)
            .count() as u64;
        let indeterminate = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::Indeterminate)
            .count() as u64;
        let fallback = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::FallbackApplied)
            .count() as u64;
        let failed = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::ExecutionFailed)
            .count() as u64;
        let missing = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::Missing)
            .count() as u64;
        let perturbation_attempted = observations
            .iter()
            .filter(|row| row.decision != ObservationDecisionV1::Baseline)
            .count() as u64;
        let validated = observations
            .iter()
            .filter(|row| row.decision == ObservationDecisionV1::Validated)
            .count() as u64;
        let coverage = if expected == 0 {
            0.0
        } else {
            scores.len() as f64 * 1_000.0 / expected as f64
        };
        let acceptance = if perturbation_attempted == 0 {
            None
        } else {
            Some(validated as f64 * 1_000.0 / perturbation_attempted as f64)
        };
        let mut paired_count = 0_u64;
        let mut deltas_by_unit = BTreeMap::<String, Vec<f64>>::new();
        let paired_delta = if key.condition_id == axis.baseline_condition_id {
            None
        } else {
            for observation in observations {
                if let Some(score) = observation.score {
                    let baseline_key = ObservationKey {
                        model_id: observation.model_id.clone(),
                        run_id: observation.run_id.clone(),
                        axis_id: observation.axis_id.clone(),
                        condition_id: axis.baseline_condition_id.clone(),
                        item_id: observation.item_id.clone(),
                    };
                    let baseline = observation_map
                        .get(&baseline_key)
                        .and_then(|row| row.score)
                        .ok_or_else(|| {
                            "scored perturbation has no paired scored baseline observation"
                                .to_owned()
                        })?;
                    let direction = if protocol.evaluation.higher_is_better {
                        1.0
                    } else {
                        -1.0
                    };
                    deltas_by_unit
                        .entry(observation.independent_unit_id.clone())
                        .or_default()
                        .push((score - baseline) * direction);
                    paired_count += 1;
                }
            }
            let unit_means: Vec<f64> = deltas_by_unit
                .values()
                .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                .collect();
            if unit_means.len() >= protocol.evaluation.minimum_independent_units_for_delta as usize
            {
                Some(mean_interval(&unit_means)?)
            } else {
                None
            }
        };
        let unit_count = deltas_by_unit.len() as u64;
        let delta_status = if key.condition_id == axis.baseline_condition_id {
            "baseline_reference".to_owned()
        } else if paired_delta.is_some() {
            "estimable_across_declared_independent_units".to_owned()
        } else {
            "insufficient_independent_units".to_owned()
        };
        let runtime: Vec<f64> = scored
            .iter()
            .map(|row| row.runtime_milliseconds as f64)
            .collect();
        let input_tokens: Vec<f64> = scored.iter().map(|row| row.input_tokens as f64).collect();
        let output_tokens: Vec<f64> = scored.iter().map(|row| row.output_tokens as f64).collect();
        let external_call_count = scored.iter().map(|row| row.external_calls as u64).sum();
        let cell = DistributionProfileCellV1 {
            model_id: key.model_id.clone(),
            model_family_id: models.get(&key.model_id).unwrap().model_family_id.clone(),
            task_family: key.task_family.clone(),
            axis_id: key.axis_id.clone(),
            condition_id: key.condition_id.clone(),
            expected_count: expected,
            observation_count: observations.len() as u64,
            scored_count: scores.len() as u64,
            rejected_count: rejected,
            indeterminate_count: indeterminate,
            fallback_count: fallback,
            execution_failed_count: failed,
            missing_count: missing,
            coverage_permille: coverage,
            validation_acceptance_permille: acceptance,
            mean_score: mean,
            sample_standard_deviation: sd,
            quantiles,
            paired_observation_count: paired_count,
            paired_independent_unit_count: unit_count,
            paired_delta_vs_baseline: paired_delta,
            paired_delta_status: delta_status,
            mean_runtime_milliseconds: if runtime.is_empty() {
                None
            } else {
                Some(runtime.iter().sum::<f64>() / runtime.len() as f64)
            },
            mean_input_tokens: if input_tokens.is_empty() {
                None
            } else {
                Some(input_tokens.iter().sum::<f64>() / input_tokens.len() as f64)
            },
            mean_output_tokens: if output_tokens.is_empty() {
                None
            } else {
                Some(output_tokens.iter().sum::<f64>() / output_tokens.len() as f64)
            },
            external_call_count,
        };
        cell_lookup.insert(key.clone(), cell.clone());
        profile_cells.push(cell);
    }

    let mut rank_stability = Vec::new();
    let mut any_rank_not_estimable = false;
    let task_families: BTreeSet<String> = bundle
        .instance_manifest
        .instances
        .iter()
        .map(|item| item.task_family.clone())
        .collect();
    for task_family in &task_families {
        for axis in &protocol.axes {
            if axis.auxiliary_only {
                continue;
            }
            let conditions = conditions_by_axis
                .get(&axis.axis_id)
                .ok_or_else(|| "rank-stability condition set missing".to_owned())?;
            for condition_id in conditions {
                if condition_id == &axis.baseline_condition_id {
                    continue;
                }
                let mut baseline_scores = Vec::new();
                let mut condition_scores = Vec::new();
                for model in &bundle.models {
                    let baseline_key = ProfileKey {
                        model_id: model.model_id.clone(),
                        task_family: task_family.clone(),
                        axis_id: axis.axis_id.clone(),
                        condition_id: axis.baseline_condition_id.clone(),
                    };
                    let condition_key = ProfileKey {
                        model_id: model.model_id.clone(),
                        task_family: task_family.clone(),
                        axis_id: axis.axis_id.clone(),
                        condition_id: condition_id.clone(),
                    };
                    if let (Some(base), Some(condition)) = (
                        cell_lookup.get(&baseline_key),
                        cell_lookup.get(&condition_key),
                    ) && let (Some(base_score), Some(condition_score)) =
                        (base.mean_score, condition.mean_score)
                    {
                        let direction = if protocol.evaluation.higher_is_better {
                            1.0
                        } else {
                            -1.0
                        };
                        baseline_scores.push(base_score * direction);
                        condition_scores.push(condition_score * direction);
                    }
                }
                let tau = if baseline_scores.len()
                    >= protocol.reporting.minimum_models_for_rank_stability as usize
                {
                    kendall_tau_b(&baseline_scores, &condition_scores)
                } else {
                    None
                };
                let status = if tau.is_some() {
                    "estimated"
                } else {
                    any_rank_not_estimable = true;
                    "non_estimable_insufficient_models_or_rank_variation"
                };
                rank_stability.push(RankStabilityV1 {
                    task_family: task_family.clone(),
                    axis_id: axis.axis_id.clone(),
                    condition_id: condition_id.clone(),
                    model_count: baseline_scores.len() as u64,
                    kendall_tau_b: tau,
                    status: status.to_owned(),
                });
            }
        }
    }

    let model_families: BTreeSet<String> = bundle
        .models
        .iter()
        .map(|model| model.model_family_id.clone())
        .collect();
    let source_domains: BTreeSet<String> = bundle
        .instance_manifest
        .instances
        .iter()
        .map(|item| item.source_domain_id.clone())
        .collect();
    let minimum_model_count_met = models.len() >= protocol.evaluation.minimum_model_count as usize;
    let minimum_runs_met = bundle.models.iter().all(|model| {
        runs_by_model.get(&model.model_id).copied().unwrap_or(0)
            >= protocol.evaluation.minimum_runs_per_model as u64
    });
    let primary_cells: Vec<DistributionProfileCellV1> = profile_cells
        .iter()
        .filter(|cell| !axes[cell.axis_id.as_str()].auxiliary_only)
        .cloned()
        .collect();
    let coverage_met = primary_cells
        .iter()
        .all(|cell| cell.coverage_permille >= protocol.reporting.minimum_coverage_permille as f64);
    let acceptance_met = primary_cells.iter().all(|cell| {
        cell.validation_acceptance_permille
            .is_none_or(|value| value >= protocol.reporting.minimum_acceptance_permille as f64)
    });
    let independent_units_met = primary_cells
        .iter()
        .filter(|cell| cell.condition_id != axes[cell.axis_id.as_str()].baseline_condition_id)
        .all(|cell| {
            cell.paired_independent_unit_count
                >= protocol.evaluation.minimum_independent_units_for_delta as u64
                && cell.paired_delta_vs_baseline.is_some()
        });
    let scope_population_met = match protocol.construct.claim_scope {
        ClaimScopeV1::FixedConfiguration => true,
        ClaimScopeV1::TaskFamily => !task_families.is_empty(),
        ClaimScopeV1::CrossDomain => task_families.len() >= 2 && source_domains.len() >= 2,
        ClaimScopeV1::ModelFamily => model_families.len() >= 2,
        ClaimScopeV1::DeploymentDistribution => {
            task_families.len() >= 2 && source_domains.len() >= 2 && model_families.len() >= 2
        }
    };
    let rank_met = !protocol.reporting.require_rank_stability
        || (!rank_stability.is_empty()
            && !any_rank_not_estimable
            && models.len() >= protocol.reporting.minimum_models_for_rank_stability as usize);
    let core_evidence_met = minimum_model_count_met
        && minimum_runs_met
        && coverage_met
        && acceptance_met
        && independent_units_met;
    let primary_perturbation_count = protocol
        .axes
        .iter()
        .filter(|axis| !axis.auxiliary_only)
        .map(|axis| axis.families.len())
        .sum::<usize>();
    let distributional_evidence_eligible = primary_perturbation_count > 0 && core_evidence_met;
    let fixed_configuration_evidence_eligible = protocol.construct.claim_scope
        == ClaimScopeV1::FixedConfiguration
        && primary_perturbation_count == 0
        && core_evidence_met;
    let claim_scope_eligible = (distributional_evidence_eligible
        || fixed_configuration_evidence_eligible)
        && scope_population_met
        && rank_met;
    let measurement_paradigm = if primary_perturbation_count > 0 {
        "measurement_as_distribution"
    } else {
        "measurement_as_instance_restricted"
    };

    let mut issues = Vec::new();
    if !minimum_model_count_met {
        issues.push("model count is below the protocol minimum".to_owned());
    }
    if !minimum_runs_met {
        issues.push("one or more models are below the minimum run count".to_owned());
    }
    if !coverage_met {
        issues.push("one or more profile cells are below minimum coverage".to_owned());
    }
    if !acceptance_met {
        issues.push(
            "one or more perturbation axes are below minimum validation acceptance".to_owned(),
        );
    }
    if !independent_units_met {
        issues.push("one or more paired deltas lack the minimum independent units".to_owned());
    }
    if !scope_population_met {
        issues.push(
            "observed model/task population is too narrow for the declared claim scope".to_owned(),
        );
    }
    if !rank_met {
        issues.push("rank stability is required but is not estimable for every declared primary axis/condition".to_owned());
    }
    if primary_perturbation_count == 0 {
        issues.push("fixed-configuration Measurement-as-Instance evidence does not support a distributional robustness claim".to_owned());
    }
    if protocol
        .axes
        .iter()
        .any(|axis| axis.auxiliary_only && axis.kind == PerturbationAxisKindV1::Prompt)
    {
        issues.push("prompt sensitivity is reported as a separate auxiliary axis and is not included in the primary construct claim".to_owned());
    }

    let run_bundle_bytes = serde_json::to_vec(bundle)
        .map_err(|error| format!("cannot serialize measurement run bundle: {error}"))?;
    let run_bundle_sha = hash_domain(b"measurement-distribution-run-bundle.v1", &run_bundle_bytes);
    let independent_unit_count = independent_units.len() as u64;
    Ok(MeasurementDistributionReportV1 {
        schema: MEASUREMENT_REPORT_SCHEMA.to_owned(),
        benchmark_id: protocol.benchmark_id.clone(),
        benchmark_version: protocol.benchmark_version.clone(),
        protocol_sha256: protocol_sha,
        instance_manifest_sha256: manifest_sha,
        run_bundle_sha256: run_bundle_sha,
        claim_scope: protocol.construct.claim_scope,
        measurement_paradigm: measurement_paradigm.to_owned(),
        model_count: models.len() as u64,
        model_family_count: model_families.len() as u64,
        task_family_count: task_families.len() as u64,
        source_domain_count: source_domains.len() as u64,
        run_count: bundle.runs.len() as u64,
        independent_unit_kind: protocol.evaluation.independent_unit_kind,
        independent_unit_count,
        instance_count: instances.len() as u64,
        observation_count: bundle.observations.len() as u64,
        distributional_evidence_eligible,
        claim_scope_eligible,
        paired_delta_positive_is_better: true,
        independently_authenticated: false,
        confirmatory_eligible: false,
        rank_stability_status: if rank_met { "qualified_or_not_required" } else { "non_estimable" }.to_owned(),
        profile_cells,
        rank_stability,
        model_summaries: build_model_summaries(&models, &primary_cells),
        issues,
        claim_boundary: "This report describes score distributions and paired robustness deltas only over the supplied, digest-bound measurement protocol, instance manifest, models, runs and observations. It does not prove that generated items or perturbations preserve the intended construct, that the declared model/evaluator identities are authenticated, or that the registered distribution equals deployment. Confidence intervals are computed over the declared independent-unit means; items, perturbation variants and repeated calls do not create independent training replicates. This analysis performs no hypothesis tests; the declared Holm policy applies if a separate registered test family reports p-values.".to_owned(),
    })
}

fn build_model_summaries(
    models: &BTreeMap<String, &MeasurementModelV1>,
    cells: &[DistributionProfileCellV1],
) -> Vec<ModelRobustnessSummaryV1> {
    models
        .iter()
        .map(|(model_id, model)| {
            let model_cells: Vec<&DistributionProfileCellV1> = cells
                .iter()
                .filter(|cell| &cell.model_id == model_id)
                .collect();
            let deltas: Vec<f64> = model_cells
                .iter()
                .filter_map(|cell| {
                    cell.paired_delta_vs_baseline
                        .as_ref()
                        .map(|interval| interval.mean)
                })
                .collect();
            let coverages: Vec<f64> = model_cells
                .iter()
                .map(|cell| cell.coverage_permille)
                .collect();
            let acceptances: Vec<f64> = model_cells
                .iter()
                .filter_map(|cell| cell.validation_acceptance_permille)
                .collect();
            ModelRobustnessSummaryV1 {
                model_id: model_id.clone(),
                model_family_id: model.model_family_id.clone(),
                worst_axis_mean_delta: deltas.iter().copied().reduce(f64::min),
                minimum_axis_coverage_permille: coverages
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .unwrap_or(0.0),
                minimum_axis_acceptance_permille: acceptances
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .unwrap_or(1_000.0),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn fixture_manifest() -> MeasurementInstanceManifestV1 {
        MeasurementInstanceManifestV1 {
            schema: "symthaea.measurement_distribution_instance_manifest.v1".to_owned(),
            benchmark_id: "mad-fixture".to_owned(),
            benchmark_version: "v1".to_owned(),
            instances: vec![
                MeasurementInstanceV1 {
                    item_id: "item-1".to_owned(),
                    task_family: "math".to_owned(),
                    source_domain_id: "source-a".to_owned(),
                    source_cluster_id: "cluster-1".to_owned(),
                    source_lineage_sha256: digest('6'),
                },
                MeasurementInstanceV1 {
                    item_id: "item-2".to_owned(),
                    task_family: "math".to_owned(),
                    source_domain_id: "source-b".to_owned(),
                    source_cluster_id: "cluster-2".to_owned(),
                    source_lineage_sha256: digest('7'),
                },
                MeasurementInstanceV1 {
                    item_id: "item-3".to_owned(),
                    task_family: "code".to_owned(),
                    source_domain_id: "source-c".to_owned(),
                    source_cluster_id: "cluster-3".to_owned(),
                    source_lineage_sha256: digest('8'),
                },
            ],
        }
    }

    fn protocol() -> MeasurementProtocolV1 {
        MeasurementProtocolV1 {
            schema: MEASUREMENT_PROTOCOL_SCHEMA.to_owned(),
            benchmark_id: "mad-fixture".to_owned(),
            benchmark_version: "v1".to_owned(),
            construct: ConstructSpecV1 {
                construct_id: "reasoning".to_owned(),
                construct_version: "v1".to_owned(),
                target_description: "reasoning correctness".to_owned(),
                operational_definition: "binary verifier-backed success".to_owned(),
                invariants: vec!["same underlying problem".to_owned()],
                permitted_nuisance_variation: vec!["surface form".to_owned()],
                prohibited_claims: vec!["universal capability".to_owned()],
                instruction_following_is_target: false,
                claim_scope: ClaimScopeV1::TaskFamily,
            },
            generator: GeneratorSpecV1 {
                generator_id: "fixture-generator".to_owned(),
                generator_version: "1".to_owned(),
                generator_config_sha256: digest('a'),
                source_dataset_sha256: digest('b'),
                generated_manifest_sha256: compute_instance_manifest_sha256(&fixture_manifest())
                    .unwrap(),
                seed_commitment_sha256: digest('d'),
                checkpoint_freeze_receipt_sha256: None,
                fresh_instances_after_checkpoint_freeze: false,
                generated_after_checkpoint_freeze: false,
            },
            axes: vec![PerturbationAxisSpecV1 {
                axis_id: "surface".to_owned(),
                kind: PerturbationAxisKindV1::SurfaceForm,
                description: "surface-preserving rewrite".to_owned(),
                baseline_condition_id: "original".to_owned(),
                auxiliary_only: false,
                families: vec![PerturbationFamilySpecV1 {
                    family_id: "paraphrase".to_owned(),
                    condition_id: "paraphrased".to_owned(),
                    transformer_sha256: digest('e'),
                    invariance_claim: "same answer and skill".to_owned(),
                    acceptance_rule: "verifier and blinded review".to_owned(),
                    validator_kind: ValidatorKindV1::Hybrid,
                    validation_protocol_sha256: digest('f'),
                    minimum_acceptance_permille: 800,
                    fallback_condition_id: None,
                }],
            }],
            evaluation: EvaluationSpecV1 {
                harness_sha256: digest('1'),
                prompt_template_sha256: digest('2'),
                scorer_sha256: digest('3'),
                decoding_config_sha256: digest('4'),
                evaluator_sha256: digest('5'),
                metric_id: "accuracy".to_owned(),
                higher_is_better: true,
                independent_unit_kind: IndependentUnitKindV1::SourceCluster,
                minimum_model_count: 1,
                minimum_runs_per_model: 2,
                minimum_independent_units_for_delta: 2,
                resample_instances_across_runs: false,
                resample_perturbations_across_runs: true,
                resample_model_inference_across_runs: true,
            },
            reporting: ReportingSpecV1 {
                minimum_coverage_permille: 900,
                minimum_acceptance_permille: 800,
                quantiles_basis_points: [50, 2_500, 5_000, 7_500, 9_950],
                require_rank_stability: false,
                minimum_models_for_rank_stability: 3,
                multiple_comparison_policy: "holm_when_tests_are_reported".to_owned(),
            },
        }
    }

    fn bundle(protocol: &MeasurementProtocolV1) -> MeasurementRunBundleV1 {
        let manifest = fixture_manifest();
        assert_eq!(manifest.benchmark_id, protocol.benchmark_id);
        assert_eq!(manifest.benchmark_version, protocol.benchmark_version);
        let manifest_sha = compute_instance_manifest_sha256(&manifest).unwrap();
        let protocol_bytes = serde_json::to_vec(protocol).unwrap();
        let protocol_sha = hash_domain(b"measurement-distribution-protocol.v1", &protocol_bytes);
        let mut runs = Vec::new();
        let mut observations = Vec::new();
        for run_idx in 0..2 {
            let run_id = format!("run-{run_idx}");
            runs.push(MeasurementRunV1 {
                model_id: "model-a".to_owned(),
                run_id: run_id.clone(),
                independent_unit_id: format!("unit-{run_idx}"),
                seed_commitment_sha256: digest(if run_idx == 0 { '8' } else { '9' }),
            });
            for item in &manifest.instances {
                for (condition_id, score, decision, receipt) in [
                    ("original", 1.0, ObservationDecisionV1::Baseline, None),
                    (
                        "paraphrased",
                        0.8,
                        ObservationDecisionV1::Validated,
                        Some(digest('a')),
                    ),
                ] {
                    observations.push(MeasurementObservationV1 {
                        model_id: "model-a".to_owned(),
                        run_id: run_id.clone(),
                        independent_unit_id: item.source_cluster_id.clone(),
                        axis_id: "surface".to_owned(),
                        condition_id: condition_id.to_owned(),
                        item_id: item.item_id.clone(),
                        task_family: item.task_family.clone(),
                        source_cluster_id: item.source_cluster_id.clone(),
                        decision,
                        validation_receipt_sha256: receipt,
                        perturbation_seed_commitment_sha256: if decision
                            == ObservationDecisionV1::Baseline
                        {
                            None
                        } else {
                            Some(digest('f'))
                        },
                        external_egress_receipt_sha256: None,
                        score: Some(score),
                        runtime_milliseconds: 10,
                        input_tokens: 20,
                        output_tokens: 30,
                        external_calls: 0,
                    });
                }
            }
        }
        MeasurementRunBundleV1 {
            schema: MEASUREMENT_RUN_BUNDLE_SCHEMA.to_owned(),
            protocol_sha256: protocol_sha,
            instance_manifest: manifest,
            instance_manifest_sha256: manifest_sha,
            models: vec![MeasurementModelV1 {
                model_id: "model-a".to_owned(),
                model_family_id: "family-a".to_owned(),
                model_checkpoint_sha256: digest('b'),
                tokenizer_sha256: digest('c'),
            }],
            runs,
            observations,
        }
    }

    #[test]
    fn protocol_hash_is_stable_for_identical_structs() {
        let p = protocol();
        assert_eq!(
            compute_measurement_protocol_sha256(&p).unwrap(),
            compute_measurement_protocol_sha256(&p).unwrap()
        );
    }

    #[test]
    fn complete_bundle_reports_paired_delta_and_distribution() {
        let p = protocol();
        let b = bundle(&p);
        let report = analyze_measurement_distribution(&p, &b).unwrap();
        let cell = report
            .profile_cells
            .iter()
            .find(|cell| cell.condition_id == "paraphrased")
            .unwrap();
        assert_eq!(cell.scored_count, 6);
        assert_eq!(cell.paired_observation_count, 6);
        assert_eq!(cell.paired_independent_unit_count, 3);
        assert_eq!(report.profile_cells.len(), 4);
        assert!(
            report
                .profile_cells
                .iter()
                .any(|profile| profile.task_family == "math")
        );
        assert!(
            report
                .profile_cells
                .iter()
                .any(|profile| profile.task_family == "code")
        );
        assert_eq!(report.task_family_count, 2);
        assert_eq!(report.source_domain_count, 3);
        assert!(report.paired_delta_positive_is_better);
        assert!((cell.paired_delta_vs_baseline.as_ref().unwrap().mean - (-0.2)).abs() < 1.0e-12);
        assert!(report.distributional_evidence_eligible);
    }

    #[test]
    fn missing_condition_observation_fails_exact_coverage() {
        let p = protocol();
        let mut b = bundle(&p);
        b.observations.pop();
        assert!(analyze_measurement_distribution(&p, &b).is_err());
    }

    #[test]
    fn fixed_configuration_is_representable_without_perturbations() {
        let mut p = protocol();
        p.construct.claim_scope = ClaimScopeV1::FixedConfiguration;
        p.axes[0].families.clear();
        let mut b = bundle(&p);
        b.observations.retain(|row| row.condition_id == "original");
        b.protocol_sha256 = compute_measurement_protocol_sha256(&p).unwrap();
        let report = analyze_measurement_distribution(&p, &b).unwrap();
        assert_eq!(
            report.measurement_paradigm,
            "measurement_as_instance_restricted"
        );
        assert!(!report.distributional_evidence_eligible);
        assert!(report.claim_scope_eligible);
    }

    #[test]
    fn cross_domain_scope_requires_multiple_task_families_and_source_domains() {
        let mut p = protocol();
        p.construct.claim_scope = ClaimScopeV1::CrossDomain;
        let b = bundle(&p);
        let report = analyze_measurement_distribution(&p, &b).unwrap();
        assert_eq!(report.task_family_count, 2);
        assert_eq!(report.source_domain_count, 3);
        assert!(report.claim_scope_eligible);
    }

    #[test]
    fn invalid_perturbation_is_counted_not_scored() {
        let p = protocol();
        let mut b = bundle(&p);
        let row = b
            .observations
            .iter_mut()
            .find(|row| row.condition_id == "paraphrased")
            .unwrap();
        row.decision = ObservationDecisionV1::Rejected;
        row.score = None;
        row.validation_receipt_sha256 = Some(digest('f'));
        b.instance_manifest_sha256 =
            compute_instance_manifest_sha256(&b.instance_manifest).unwrap();
        assert!(analyze_measurement_distribution(&p, &b).is_ok());
    }

    #[test]
    fn prompt_sensitivity_must_be_auxiliary_for_non_instruction_constructs() {
        let mut p = protocol();
        p.axes[0].kind = PerturbationAxisKindV1::Prompt;
        p.axes[0].auxiliary_only = false;
        assert!(validate_measurement_protocol(&p).is_err());
    }

    #[test]
    fn fresh_instances_require_checkpoint_freeze_receipt_and_ordering_claim() {
        let mut p = protocol();
        p.generator.fresh_instances_after_checkpoint_freeze = true;
        assert!(validate_measurement_protocol(&p).is_err());
        p.generator.generated_after_checkpoint_freeze = true;
        p.generator.checkpoint_freeze_receipt_sha256 = Some(digest('f'));
        assert!(validate_measurement_protocol(&p).is_ok());
    }

    #[test]
    fn kendall_tau_b_handles_ties_and_perfect_rank_reversal() {
        assert!((kendall_tau_b(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap() + 1.0).abs() < 1.0e-12);
        assert!(
            (kendall_tau_b(&[1.0, 1.0, 3.0], &[1.0, 2.0, 3.0]).unwrap() - 0.816496580927726).abs()
                < 1.0e-12
        );
    }
}
