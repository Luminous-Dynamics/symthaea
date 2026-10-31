//! Empirical qualification gate for decision-focused NAS value-of-computation models.
//!
//! The gate scores frozen posterior-predictive replay episodes on task-, architecture-family-,
//! and seed-disjoint evaluation partitions. It checks joint top-1/top-K predictions, marginal
//! predictive intervals and PIT ranks, action-outcome probabilities, realized regret reduction,
//! resource-cost agreement, particle ESS, and conditional calibration strata.
//!
//! This is an empirical qualification receipt for the tested benchmark/task distribution. It is
//! not a proof of arbitrary-domain transfer, global-optimum retention, or frequentist VOI bounds.
//! Split-manifest integrity and artifact provenance must be independently audited by the runner.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.nas.voc-qualification";
const VERSION: &str = "0.1.0";
const SHA256_HEX_LEN: usize = 64;
const DIGEST_DOMAIN: &[u8] = b"symthaea.nas.voc-qualification.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationActionKind {
    FidelityQuery,
    DormantReactivation,
    SentinelAudit,
}

impl QualificationActionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::FidelityQuery => "fidelity_query",
            Self::DormantReactivation => "dormant_reactivation",
            Self::SentinelAudit => "sentinel_audit",
        }
    }

    fn all() -> [Self; 3] {
        [
            Self::FidelityQuery,
            Self::DormantReactivation,
            Self::SentinelAudit,
        ]
    }
}

/// Frozen split and model identity. IDs are group identities, not individual observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoCQualificationManifest {
    pub benchmark_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub source_artifact_sha256: String,
    pub posterior_model_sha256: String,
    pub action_likelihood_model_sha256: String,
    pub fidelity_ladder_sha256: String,
    pub fidelity_cost_model_sha256: String,
    /// Exact candidate-column order used by every particle matrix.
    pub candidate_ids: Vec<String>,
    pub fidelity_ids: Vec<String>,
    pub target_k: usize,
    /// Frozen posterior weights shared by each evaluation episode.
    pub posterior_particle_weights: Vec<f64>,
    pub fit_task_ids: Vec<String>,
    pub calibration_task_ids: Vec<String>,
    pub evaluation_task_ids: Vec<String>,
    pub fit_architecture_families: Vec<String>,
    pub calibration_architecture_families: Vec<String>,
    pub evaluation_architecture_families: Vec<String>,
    /// Expected evaluation strata are declared before outcome inspection.
    pub evaluation_rank_regimes: Vec<String>,
    pub evaluation_curve_shapes: Vec<String>,
    pub evaluation_noise_regimes: Vec<String>,
    pub fit_seed_ids: Vec<String>,
    pub calibration_seed_ids: Vec<String>,
    pub evaluation_seed_ids: Vec<String>,
    /// Recomputed from the exact split IDs and model/artifact digests.
    pub split_manifest_sha256: String,
}

/// One held-out task episode. Each row of particle_values is one joint posterior particle;
/// columns follow manifest.candidate_ids. Scores are normalized to [0,1] before direction
/// conversion, so utility is always maximize-is-better.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoCReplayEpisode {
    pub episode_id: String,
    pub task_id: String,
    pub architecture_family: String,
    pub seed_id: String,
    pub fidelity_id: String,
    pub rank_regime: String,
    pub curve_shape: String,
    pub noise_regime: String,
    pub particle_values: Vec<Vec<f64>>,
    pub realized_final_values: Vec<f64>,
    pub action_forecasts: Vec<ActionOutcomeForecast>,
}

/// Predictive probabilities and the observed outcome for a replayed action. Costs are kept in
/// the same frozen resource unit; cross-unit scalarization is deliberately not performed here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcomeForecast {
    pub action_id: String,
    pub kind: QualificationActionKind,
    pub outcome_ids: Vec<String>,
    pub predicted_outcome_probabilities: Vec<f64>,
    pub realized_outcome_id: String,
    pub predicted_expected_regret_reduction: f64,
    /// May be negative: an experiment can make the final recommendation worse in hindsight.
    pub realized_final_regret_reduction: f64,
    pub planned_cost_units: u64,
    pub measured_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoCQualificationThresholds {
    pub min_fit_tasks: usize,
    pub min_calibration_tasks: usize,
    pub min_evaluation_tasks: usize,
    pub min_evaluation_episodes: usize,
    pub min_episodes_per_stratum: usize,
    pub min_action_forecasts_per_kind: usize,
    pub min_effective_sample_size_fraction: f64,
    pub max_top1_log_loss: f64,
    pub max_top1_brier: f64,
    pub max_top1_ece: f64,
    pub min_topk_membership_brier_skill: f64,
    pub min_topk_credible_set_coverage: f64,
    pub topk_inclusion_probability_floor: f64,
    pub max_mean_credible_set_size_fraction: f64,
    pub nominal_interval_mass: f64,
    pub interval_coverage_tolerance: f64,
    pub max_mean_interval_width: f64,
    pub rank_histogram_bins: usize,
    pub ece_bins: usize,
    pub max_action_outcome_log_loss: f64,
    pub max_action_outcome_brier: f64,
    pub max_action_outcome_ece: f64,
    pub max_voi_mean_absolute_error: f64,
    pub max_cost_relative_error: f64,
}

impl Default for VoCQualificationThresholds {
    fn default() -> Self {
        Self {
            min_fit_tasks: 3,
            min_calibration_tasks: 3,
            min_evaluation_tasks: 5,
            min_evaluation_episodes: 50,
            min_episodes_per_stratum: 10,
            min_action_forecasts_per_kind: 20,
            min_effective_sample_size_fraction: 0.10,
            max_top1_log_loss: 1.00,
            max_top1_brier: 0.35,
            max_top1_ece: 0.08,
            min_topk_membership_brier_skill: 0.10,
            min_topk_credible_set_coverage: 0.90,
            topk_inclusion_probability_floor: 0.05,
            max_mean_credible_set_size_fraction: 0.60,
            nominal_interval_mass: 0.90,
            interval_coverage_tolerance: 0.05,
            max_mean_interval_width: 0.50,
            rank_histogram_bins: 10,
            ece_bins: 10,
            max_action_outcome_log_loss: 1.00,
            max_action_outcome_brier: 0.35,
            max_action_outcome_ece: 0.10,
            max_voi_mean_absolute_error: 0.08,
            max_cost_relative_error: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalForecastMetrics {
    pub observations: usize,
    pub log_loss: f64,
    /// Multiclass Brier score: sum of squared category errors, averaged over observations.
    pub brier_score: f64,
    /// Top-label expected calibration error using fixed-width confidence bins.
    pub expected_calibration_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratumMetrics {
    pub stratum_id: String,
    pub episodes: usize,
    pub top1_ece: f64,
    pub topk_membership_brier: f64,
    pub topk_membership_brier_skill: f64,
    pub topk_credible_set_coverage: f64,
    pub interval_coverage: f64,
    pub mean_interval_width: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoCQualificationMetrics {
    pub evaluation_tasks: usize,
    pub evaluation_episodes: usize,
    pub particle_count: usize,
    pub effective_sample_size: f64,
    pub effective_sample_size_fraction: f64,
    pub top1_accuracy: f64,
    pub top1_log_loss: f64,
    pub top1_brier: f64,
    pub top1_ece: f64,
    pub topk_membership_brier: f64,
    pub topk_membership_brier_skill: f64,
    pub topk_credible_set_coverage: f64,
    pub mean_credible_set_size_fraction: f64,
    pub predictive_interval_coverage: f64,
    pub mean_predictive_interval_width: f64,
    pub pit_rank_histogram: Vec<usize>,
    pub action_outcome_metrics: BTreeMap<String, CategoricalForecastMetrics>,
    pub action_forecast_counts: BTreeMap<String, usize>,
    pub voi_mean_absolute_error: f64,
    pub voi_mean_signed_error: f64,
    pub cost_mean_relative_error: f64,
    pub strata: Vec<StratumMetrics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationDecision {
    Qualified,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoCQualificationReceipt {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub metric_id: String,
    pub decision: QualificationDecision,
    pub failure_reasons: Vec<String>,
    pub source_artifact_sha256: String,
    pub posterior_model_sha256: String,
    pub action_likelihood_model_sha256: String,
    pub fidelity_ladder_sha256: String,
    pub fidelity_cost_model_sha256: String,
    pub split_manifest_sha256: String,
    pub replay_evidence_sha256: String,
    pub thresholds_sha256: String,
    pub metrics: VoCQualificationMetrics,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationError {
    pub field: String,
    pub message: String,
}

impl QualificationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for QualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for QualificationError {}

/// Hashes the sorted split IDs and all model/evaluator bindings, not just the task names.
pub fn compute_split_manifest_sha256(
    manifest: &VoCQualificationManifest,
) -> Result<String, QualificationError> {
    let mut canonical = manifest.clone();
    canonical.candidate_ids.sort();
        canonical.fit_task_ids.sort();
    canonical.calibration_task_ids.sort();
    canonical.evaluation_task_ids.sort();
    canonical.fit_architecture_families.sort();
    canonical.calibration_architecture_families.sort();
    canonical.evaluation_architecture_families.sort();
    canonical.fit_seed_ids.sort();
    canonical.calibration_seed_ids.sort();
    canonical.evaluation_seed_ids.sort();
    canonical.split_manifest_sha256.clear();
    digest_serializable(b"split-manifest", &canonical)
}

/// Score held-out replay evidence and issue a digest-bound qualification or abstention receipt.
/// The evidence contains joint posterior particles, so candidate correlations are preserved.
pub fn qualify_voc_model(
    manifest: &VoCQualificationManifest,
    episodes: &[VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
) -> Result<VoCQualificationReceipt, QualificationError> {
    validate_manifest(manifest, thresholds)?;
    validate_thresholds(thresholds)?;

    if compute_split_manifest_sha256(manifest)? != manifest.split_manifest_sha256 {
        return Err(QualificationError::new(
            "split_manifest_sha256",
            "split manifest digest does not match the exact split and model identities",
        ));
    }

    validate_episodes(manifest, episodes, thresholds)?;
    let metrics = compute_metrics(manifest, episodes, thresholds)?;
    let threshold_bytes = serde_json::to_vec(thresholds)
        .map_err(|error| QualificationError::new("thresholds", error.to_string()))?;
    let thresholds_sha256 = digest_bytes(b"thresholds", &threshold_bytes);
    let replay_evidence_sha256 = compute_replay_evidence_sha256(episodes)?;
    let mut failure_reasons = Vec::new();

    check_at_most(
        "top1_log_loss",
        metrics.top1_log_loss,
        thresholds.max_top1_log_loss,
        &mut failure_reasons,
    );
    check_at_most(
        "top1_brier",
        metrics.top1_brier,
        thresholds.max_top1_brier,
        &mut failure_reasons,
    );
    check_at_most(
        "top1_ece",
        metrics.top1_ece,
        thresholds.max_top1_ece,
        &mut failure_reasons,
    );
    check_at_least(
        "topk_membership_brier_skill",
        metrics.topk_membership_brier_skill,
        thresholds.min_topk_membership_brier_skill,
        &mut failure_reasons,
    );
    check_at_least(
        "topk_credible_set_coverage",
        metrics.topk_credible_set_coverage,
        thresholds.min_topk_credible_set_coverage,
        &mut failure_reasons,
    );
    check_at_most(
        "mean_credible_set_size_fraction",
        metrics.mean_credible_set_size_fraction,
        thresholds.max_mean_credible_set_size_fraction,
        &mut failure_reasons,
    );
    let coverage_error =
        (metrics.predictive_interval_coverage - thresholds.nominal_interval_mass).abs();
    check_at_most(
        "predictive_interval_coverage_absolute_error",
        coverage_error,
        thresholds.interval_coverage_tolerance,
        &mut failure_reasons,
    );
    check_at_most(
        "mean_predictive_interval_width",
        metrics.mean_predictive_interval_width,
        thresholds.max_mean_interval_width,
        &mut failure_reasons,
    );
    check_at_least(
        "effective_sample_size_fraction",
        metrics.effective_sample_size_fraction,
        thresholds.min_effective_sample_size_fraction,
        &mut failure_reasons,
    );
    check_at_most(
        "voi_mean_absolute_error",
        metrics.voi_mean_absolute_error,
        thresholds.max_voi_mean_absolute_error,
        &mut failure_reasons,
    );
    check_at_most(
        "cost_mean_relative_error",
        metrics.cost_mean_relative_error,
        thresholds.max_cost_relative_error,
        &mut failure_reasons,
    );

    for kind in QualificationActionKind::all() {
        let key = kind.as_str().to_owned();
        match metrics.action_outcome_metrics.get(&key) {
            Some(action_metrics) => {
                check_at_most(
                    &format!("{key}.log_loss"),
                    action_metrics.log_loss,
                    thresholds.max_action_outcome_log_loss,
                    &mut failure_reasons,
                );
                check_at_most(
                    &format!("{key}.brier_score"),
                    action_metrics.brier_score,
                    thresholds.max_action_outcome_brier,
                    &mut failure_reasons,
                );
                check_at_most(
                    &format!("{key}.ece"),
                    action_metrics.expected_calibration_error,
                    thresholds.max_action_outcome_ece,
                    &mut failure_reasons,
                );
            }
            None => failure_reasons.push(format!("{key}: no action outcome forecasts")),
        }
    }

    for stratum in &metrics.strata {
        if stratum.episodes < thresholds.min_episodes_per_stratum {
            failure_reasons.push(format!(
                "{}: only {} episodes; minimum is {}",
                stratum.stratum_id,
                stratum.episodes,
                thresholds.min_episodes_per_stratum
            ));
            continue;
        }
        check_at_most(
            &format!("{}.top1_ece", stratum.stratum_id),
            stratum.top1_ece,
            thresholds.max_top1_ece,
            &mut failure_reasons,
        );
        check_at_least(
            &format!("{}.topk_membership_brier_skill", stratum.stratum_id),
            stratum.topk_membership_brier_skill,
            thresholds.min_topk_membership_brier_skill,
            &mut failure_reasons,
        );
        check_at_least(
            &format!("{}.topk_credible_set_coverage", stratum.stratum_id),
            stratum.topk_credible_set_coverage,
            thresholds.min_topk_credible_set_coverage,
            &mut failure_reasons,
        );
        let stratum_coverage_error =
            (stratum.interval_coverage - thresholds.nominal_interval_mass).abs();
        check_at_most(
            &format!("{}.interval_coverage_error", stratum.stratum_id),
            stratum_coverage_error,
            thresholds.interval_coverage_tolerance,
            &mut failure_reasons,
        );
        check_at_most(
            &format!("{}.mean_interval_width", stratum.stratum_id),
            stratum.mean_interval_width,
            thresholds.max_mean_interval_width,
            &mut failure_reasons,
        );
    }

    failure_reasons.sort();
    failure_reasons.dedup();
    let mut receipt = VoCQualificationReceipt {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        benchmark_id: manifest.benchmark_id.clone(),
        metric_id: manifest.metric_id.clone(),
        decision: if failure_reasons.is_empty() {
            QualificationDecision::Qualified
        } else {
            QualificationDecision::Abstain
        },
        failure_reasons,
        source_artifact_sha256: manifest.source_artifact_sha256.clone(),
        posterior_model_sha256: manifest.posterior_model_sha256.clone(),
        action_likelihood_model_sha256: manifest.action_likelihood_model_sha256.clone(),
        fidelity_ladder_sha256: manifest.fidelity_ladder_sha256.clone(),
        fidelity_cost_model_sha256: manifest.fidelity_cost_model_sha256.clone(),
        split_manifest_sha256: manifest.split_manifest_sha256.clone(),
        replay_evidence_sha256,
        thresholds_sha256,
        metrics,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Deterministically replays qualification from exact inputs and rejects any altered receipt.
pub fn verify_voc_qualification_receipt(
    manifest: &VoCQualificationManifest,
    episodes: &[VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
    receipt: &VoCQualificationReceipt,
) -> Result<(), QualificationError> {
    let recomputed = qualify_voc_model(manifest, episodes, thresholds)?;
    if &recomputed != receipt {
        return Err(QualificationError::new(
            "receipt",
            "receipt differs from the deterministic qualification replay",
        ));
    }
    if receipt_digest(receipt)? != receipt.receipt_sha256 {
        return Err(QualificationError::new(
            "receipt_sha256",
            "receipt digest does not match the canonical receipt content",
        ));
    }
    Ok(())
}

fn validate_manifest(
    manifest: &VoCQualificationManifest,
    thresholds: &VoCQualificationThresholds,
) -> Result<(), QualificationError> {
    for (field, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("metric_id", manifest.metric_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(QualificationError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("source_artifact_sha256", manifest.source_artifact_sha256.as_str()),
        ("posterior_model_sha256", manifest.posterior_model_sha256.as_str()),
        (
            "action_likelihood_model_sha256",
            manifest.action_likelihood_model_sha256.as_str(),
        ),
        ("fidelity_ladder_sha256", manifest.fidelity_ladder_sha256.as_str()),
        (
            "fidelity_cost_model_sha256",
            manifest.fidelity_cost_model_sha256.as_str(),
        ),
        ("split_manifest_sha256", manifest.split_manifest_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(QualificationError::new(
                field,
                "must be a 64-character lowercase hexadecimal SHA-256 digest",
            ));
        }
    }
    if manifest.target_k == 0 || manifest.candidate_ids.len() <= manifest.target_k {
        return Err(QualificationError::new(
            "candidate_ids",
            "candidate count must be at least target_k and target_k must be positive",
        ));
    }
    validate_unique_nonempty("candidate_ids", &manifest.candidate_ids)?;
    if manifest.candidate_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(QualificationError::new(
            "candidate_ids",
            "candidate IDs must be strictly sorted to bind particle-matrix columns",
        ));
    }
    validate_unique_nonempty("fidelity_ids", &manifest.fidelity_ids)?;
    if manifest.fidelity_ids.is_empty() {
        return Err(QualificationError::new("fidelity_ids", "must not be empty"));
    }
    for (field, ids) in [
        ("fit_task_ids", &manifest.fit_task_ids),
        ("calibration_task_ids", &manifest.calibration_task_ids),
        ("evaluation_task_ids", &manifest.evaluation_task_ids),
        (
            "fit_architecture_families",
            &manifest.fit_architecture_families,
        ),
        (
            "calibration_architecture_families",
            &manifest.calibration_architecture_families,
        ),
        (
            "evaluation_architecture_families",
            &manifest.evaluation_architecture_families,
        ),
        ("evaluation_rank_regimes", &manifest.evaluation_rank_regimes),
        ("evaluation_curve_shapes", &manifest.evaluation_curve_shapes),
        ("evaluation_noise_regimes", &manifest.evaluation_noise_regimes),
        ("fit_seed_ids", &manifest.fit_seed_ids),
        ("calibration_seed_ids", &manifest.calibration_seed_ids),
        ("evaluation_seed_ids", &manifest.evaluation_seed_ids),
    ] {
        validate_unique_nonempty(field, ids)?;
    }
    validate_disjoint(
        "task_ids",
        &manifest.fit_task_ids,
        &manifest.calibration_task_ids,
        &manifest.evaluation_task_ids,
    )?;
    validate_disjoint(
        "architecture_families",
        &manifest.fit_architecture_families,
        &manifest.calibration_architecture_families,
        &manifest.evaluation_architecture_families,
    )?;
    validate_disjoint(
        "seed_ids",
        &manifest.fit_seed_ids,
        &manifest.calibration_seed_ids,
        &manifest.evaluation_seed_ids,
    )?;

    if manifest.fit_task_ids.len() < thresholds.min_fit_tasks {
        return Err(QualificationError::new(
            "fit_task_ids",
            "fit partition is smaller than the pre-registered minimum",
        ));
    }
    if manifest.calibration_task_ids.len() < thresholds.min_calibration_tasks {
        return Err(QualificationError::new(
            "calibration_task_ids",
            "calibration partition is smaller than the pre-registered minimum",
        ));
    }
    if manifest.evaluation_task_ids.len() < thresholds.min_evaluation_tasks {
        return Err(QualificationError::new(
            "evaluation_task_ids",
            "evaluation partition is smaller than the pre-registered minimum",
        ));
    }
    if manifest.posterior_particle_weights.len() < 2 {
        return Err(QualificationError::new(
            "posterior_particle_weights",
            "at least two posterior particles are required for qualification",
        ));
    }
    let mut weight_sum = 0.0;
    for (index, weight) in manifest.posterior_particle_weights.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(QualificationError::new(
                format!("posterior_particle_weights[{index}]"),
                "weights must be finite and non-negative",
            ));
        }
        weight_sum += *weight;
    }
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1e-9 {
        return Err(QualificationError::new(
            "posterior_particle_weights",
            "posterior weights must sum to one",
        ));
    }
    Ok(())
}

fn validate_thresholds(
    thresholds: &VoCQualificationThresholds,
) -> Result<(), QualificationError> {
    if thresholds.min_fit_tasks == 0
        || thresholds.min_calibration_tasks == 0
        || thresholds.min_evaluation_tasks == 0
        || thresholds.min_evaluation_episodes == 0
        || thresholds.min_episodes_per_stratum == 0
        || thresholds.min_action_forecasts_per_kind == 0
        || thresholds.rank_histogram_bins < 2
        || thresholds.ece_bins == 0
    {
        return Err(QualificationError::new(
            "thresholds",
            "minimum counts and histogram/bin counts must be positive (rank bins >= 2)",
        ));
    }
    for (field, value) in [
        (
            "min_effective_sample_size_fraction",
            thresholds.min_effective_sample_size_fraction,
        ),
        ("max_top1_ece", thresholds.max_top1_ece),
        (
            "min_topk_membership_brier_skill",
            thresholds.min_topk_membership_brier_skill,
        ),
        (
            "min_topk_credible_set_coverage",
            thresholds.min_topk_credible_set_coverage,
        ),
        (
            "topk_inclusion_probability_floor",
            thresholds.topk_inclusion_probability_floor,
        ),
        ("nominal_interval_mass", thresholds.nominal_interval_mass),
        (
            "interval_coverage_tolerance",
            thresholds.interval_coverage_tolerance,
        ),
        ("max_action_outcome_ece", thresholds.max_action_outcome_ece),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(QualificationError::new(
                field,
                "must be finite and in [0, 1]",
            ));
        }
    }
    for (field, value) in [
        ("max_top1_log_loss", thresholds.max_top1_log_loss),
        ("max_top1_brier", thresholds.max_top1_brier),
        ("max_mean_credible_set_size_fraction", thresholds.max_mean_credible_set_size_fraction),
        ("max_mean_interval_width", thresholds.max_mean_interval_width),
        ("max_action_outcome_log_loss", thresholds.max_action_outcome_log_loss),
        ("max_action_outcome_brier", thresholds.max_action_outcome_brier),
        ("max_voi_mean_absolute_error", thresholds.max_voi_mean_absolute_error),
        ("max_cost_relative_error", thresholds.max_cost_relative_error),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(QualificationError::new(
                field,
                "must be finite and non-negative",
            ));
        }
    }
    if thresholds.min_effective_sample_size_fraction == 0.0
        || thresholds.nominal_interval_mass == 0.0
        || thresholds.topk_inclusion_probability_floor == 0.0
        || thresholds.max_mean_credible_set_size_fraction == 0.0
        || thresholds.max_mean_credible_set_size_fraction > 1.0
        || thresholds.max_mean_interval_width > 1.0
    {
        return Err(QualificationError::new(
            "thresholds",
            "probability and normalized-width limits must be positive and no greater than one",
        ));
    }
    if thresholds.nominal_interval_mass >= 1.0 {
        return Err(QualificationError::new(
            "nominal_interval_mass",
            "must be less than one",
        ));
    }
    Ok(())
}

fn validate_episodes(
    manifest: &VoCQualificationManifest,
    episodes: &[VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
) -> Result<(), QualificationError> {
    if episodes.len() < thresholds.min_evaluation_episodes {
        return Err(QualificationError::new(
            "episodes",
            "replay evidence is smaller than the pre-registered minimum",
        ));
    }
    let task_ids: HashSet<&str> = manifest.evaluation_task_ids.iter().map(String::as_str).collect();
    let family_ids: HashSet<&str> = manifest
        .evaluation_architecture_families
        .iter()
        .map(String::as_str)
        .collect();
    let seed_ids: HashSet<&str> = manifest.evaluation_seed_ids.iter().map(String::as_str).collect();
    let fidelity_ids: HashSet<&str> = manifest.fidelity_ids.iter().map(String::as_str).collect();
    let rank_regimes: HashSet<&str> = manifest.evaluation_rank_regimes.iter().map(String::as_str).collect();
    let curve_shapes: HashSet<&str> = manifest.evaluation_curve_shapes.iter().map(String::as_str).collect();
    let noise_regimes: HashSet<&str> = manifest.evaluation_noise_regimes.iter().map(String::as_str).collect();
    let candidate_count = manifest.candidate_ids.len();
    let particle_count = manifest.posterior_particle_weights.len();
    let mut episode_ids = HashSet::with_capacity(episodes.len());
    let mut covered_tasks = HashSet::new();
    let mut covered_families = HashSet::new();
    let mut covered_fidelities = HashSet::new();
    let mut covered_rank_regimes = HashSet::new();
    let mut covered_curve_shapes = HashSet::new();
    let mut covered_noise_regimes = HashSet::new();

    for (index, episode) in episodes.iter().enumerate() {
        let field = format!("episodes[{index}]");
        if episode.episode_id.trim().is_empty()
            || !episode_ids.insert(episode.episode_id.as_str())
        {
            return Err(QualificationError::new(
                format!("{field}.episode_id"),
                "episode IDs must be non-empty and unique",
            ));
        }
        if !task_ids.contains(episode.task_id.as_str()) {
            return Err(QualificationError::new(
                format!("{field}.task_id"),
                "episode task is not in the frozen evaluation partition",
            ));
        }
        if !family_ids.contains(episode.architecture_family.as_str()) {
            return Err(QualificationError::new(
                format!("{field}.architecture_family"),
                "architecture family is not in the frozen evaluation partition",
            ));
        }
        if !seed_ids.contains(episode.seed_id.as_str()) {
            return Err(QualificationError::new(
                format!("{field}.seed_id"),
                "seed is not in the frozen evaluation partition",
            ));
        }
        if !fidelity_ids.contains(episode.fidelity_id.as_str()) {
            return Err(QualificationError::new(
                format!("{field}.fidelity_id"),
                "fidelity is not in the frozen ordered ladder",
            ));
        }
        for (name, value, declared) in [
            ("rank_regime", episode.rank_regime.as_str(), &rank_regimes),
            ("curve_shape", episode.curve_shape.as_str(), &curve_shapes),
            ("noise_regime", episode.noise_regime.as_str(), &noise_regimes),
        ] {
            if value.trim().is_empty() || !declared.contains(value) {
                return Err(QualificationError::new(
                    format!("{field}.{name}"),
                    "stratum identity must be non-empty and present in the frozen evaluation taxonomy",
                ));
            }
        }
        if episode.particle_values.len() != particle_count {
            return Err(QualificationError::new(
                format!("{field}.particle_values"),
                "particle row count does not match the frozen posterior weights",
            ));
        }
        for (particle_index, row) in episode.particle_values.iter().enumerate() {
            if row.len() != candidate_count {
                return Err(QualificationError::new(
                    format!("{field}.particle_values[{particle_index}]"),
                    "candidate column count does not match the manifest",
                ));
            }
            for (candidate_index, value) in row.iter().enumerate() {
                if !value.is_finite() || !(0.0..=1.0).contains(value) {
                    return Err(QualificationError::new(
                        format!(
                            "{field}.particle_values[{particle_index}][{candidate_index}]"
                        ),
                        "scores must be finite normalized values in [0, 1]",
                    ));
                }
            }
        }
        if episode.realized_final_values.len() != candidate_count {
            return Err(QualificationError::new(
                format!("{field}.realized_final_values"),
                "realized candidate count does not match the manifest",
            ));
        }
        for (candidate_index, value) in episode.realized_final_values.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(QualificationError::new(
                    format!("{field}.realized_final_values[{candidate_index}]"),
                    "scores must be finite normalized values in [0, 1]",
                ));
            }
        }
        if episode.action_forecasts.is_empty() {
            return Err(QualificationError::new(
                format!("{field}.action_forecasts"),
                "every replay episode must include action-outcome and realized-VOI evidence",
            ));
        }
        let mut action_ids = HashSet::new();
        for (action_index, action) in episode.action_forecasts.iter().enumerate() {
            validate_action_forecast(action, &format!("{field}.action_forecasts[{action_index}]"))?;
            if !action_ids.insert(action.action_id.as_str()) {
                return Err(QualificationError::new(
                    format!("{field}.action_forecasts[{action_index}].action_id"),
                    "action IDs must be unique within an episode",
                ));
            }
        }
        covered_tasks.insert(episode.task_id.as_str());
        covered_families.insert(episode.architecture_family.as_str());
        covered_fidelities.insert(episode.fidelity_id.as_str());
        covered_rank_regimes.insert(episode.rank_regime.as_str());
        covered_curve_shapes.insert(episode.curve_shape.as_str());
        covered_noise_regimes.insert(episode.noise_regime.as_str());
    }

    for (field, declared_count, covered_count) in [
        ("evaluation_task_ids", manifest.evaluation_task_ids.len(), covered_tasks.len()),
        ("evaluation_architecture_families", manifest.evaluation_architecture_families.len(), covered_families.len()),
        ("fidelity_ids", manifest.fidelity_ids.len(), covered_fidelities.len()),
        ("evaluation_rank_regimes", manifest.evaluation_rank_regimes.len(), covered_rank_regimes.len()),
        ("evaluation_curve_shapes", manifest.evaluation_curve_shapes.len(), covered_curve_shapes.len()),
        ("evaluation_noise_regimes", manifest.evaluation_noise_regimes.len(), covered_noise_regimes.len()),
    ] {
        if declared_count != covered_count {
            return Err(QualificationError::new(
                field,
                "one or more declared evaluation strata have no replay episodes",
            ));
        }
    }
    for kind in QualificationActionKind::all() {
        let count = episodes
            .iter()
            .flat_map(|episode| episode.action_forecasts.iter())
            .filter(|action| action.kind == kind)
            .count();
        if count < thresholds.min_action_forecasts_per_kind {
            return Err(QualificationError::new(
                format!("action_forecasts.{}", kind.as_str()),
                "action forecast count is below the pre-registered minimum",
            ));
        }
    }
    Ok(())
}

fn validate_action_forecast(
    action: &ActionOutcomeForecast,
    field: &str,
) -> Result<(), QualificationError> {
    if action.action_id.trim().is_empty() || action.realized_outcome_id.trim().is_empty() {
        return Err(QualificationError::new(
            field,
            "action and realized outcome IDs must be non-empty",
        ));
    }
    if action.outcome_ids.len() < 2
        || action.outcome_ids.len() != action.predicted_outcome_probabilities.len()
    {
        return Err(QualificationError::new(
            format!("{field}.outcome_ids"),
            "at least two outcome IDs and one probability per ID are required",
        ));
    }
    validate_unique_nonempty(&format!("{field}.outcome_ids"), &action.outcome_ids)?;
    if !action.outcome_ids.contains(&action.realized_outcome_id) {
        return Err(QualificationError::new(
            format!("{field}.realized_outcome_id"),
            "observed outcome is not one of the frozen forecast outcomes",
        ));
    }
    let mut sum = 0.0;
    for (index, probability) in action.predicted_outcome_probabilities.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
            return Err(QualificationError::new(
                format!("{field}.predicted_outcome_probabilities[{index}]"),
                "probabilities must be finite and in [0, 1]",
            ));
        }
        sum += probability;
    }
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-9 {
        return Err(QualificationError::new(
            format!("{field}.predicted_outcome_probabilities"),
            "outcome probabilities must sum to one",
        ));
    }
    if !action.predicted_expected_regret_reduction.is_finite()
        || action.predicted_expected_regret_reduction < 0.0
        || !action.realized_final_regret_reduction.is_finite()
    {
        return Err(QualificationError::new(
            field,
            "predicted regret reduction must be finite/non-negative and realized regret finite",
        ));
    }
    if action.planned_cost_units == 0 || action.measured_cost_units == 0 {
        return Err(QualificationError::new(
            field,
            "planned and measured resource costs must be positive",
        ));
    }
    Ok(())
}

fn compute_metrics(
    manifest: &VoCQualificationManifest,
    episodes: &[VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
) -> Result<VoCQualificationMetrics, QualificationError> {
    let weights = &manifest.posterior_particle_weights;
    let ess = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let episode_refs: Vec<&VoCReplayEpisode> = episodes.iter().collect();
    let global = summarize_episodes(manifest, &episode_refs, thresholds)?;
    let mut action_forecasts: BTreeMap<String, Vec<(Vec<f64>, usize)>> = BTreeMap::new();
    let mut action_outcome_taxonomy: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut action_forecast_counts = BTreeMap::new();
    let mut voi_absolute_error = 0.0;
    let mut voi_signed_error = 0.0;
    let mut cost_relative_error = 0.0;
    let mut action_count = 0usize;

    for episode in episodes {
        for action in &episode.action_forecasts {
            let key = action.kind.as_str().to_owned();
            // Outcome labels have to mean the same thing across forecasts of one action class.
            // Canonicalize label order, but reject a changed category set instead of silently
            // comparing unrelated categories by vector position.
            let mut labeled_probabilities: Vec<(String, f64)> = action
                .outcome_ids
                .iter()
                .cloned()
                .zip(action.predicted_outcome_probabilities.iter().copied())
                .collect();
            labeled_probabilities.sort_by(|left, right| left.0.cmp(&right.0));
            let canonical_outcomes: Vec<String> = labeled_probabilities
                .iter()
                .map(|(outcome, _)| outcome.clone())
                .collect();
            if let Some(existing) = action_outcome_taxonomy.get(&key) {
                if existing != &canonical_outcomes {
                    return Err(QualificationError::new(
                        "action_outcome_taxonomy",
                        format!("inconsistent outcome categories for action class {key}"),
                    ));
                }
            } else {
                action_outcome_taxonomy.insert(key.clone(), canonical_outcomes.clone());
            }
            let actual_index = canonical_outcomes
                .iter()
                .position(|outcome| outcome == &action.realized_outcome_id)
                .ok_or_else(|| {
                    QualificationError::new("action_forecasts", "realized outcome disappeared")
                })?;
            let canonical_probabilities: Vec<f64> = labeled_probabilities
                .iter()
                .map(|(_, probability)| *probability)
                .collect();
            action_forecasts
                .entry(key.clone())
                .or_default()
                .push((canonical_probabilities, actual_index));
            *action_forecast_counts.entry(key).or_insert(0usize) += 1;
            let signed_error = action.predicted_expected_regret_reduction
                - action.realized_final_regret_reduction;
            voi_absolute_error += signed_error.abs();
            voi_signed_error += signed_error;
            cost_relative_error +=
                (action.measured_cost_units as f64 - action.planned_cost_units as f64).abs()
                    / action.planned_cost_units as f64;
            action_count += 1;
        }
    }

    let mut action_outcome_metrics = BTreeMap::new();
    for (kind, forecasts) in action_forecasts {
        action_outcome_metrics.insert(
            kind,
            categorical_metrics(&forecasts, thresholds.ece_bins)?,
        );
    }
    let strata = compute_strata(manifest, episodes, thresholds)?;
    Ok(VoCQualificationMetrics {
        evaluation_tasks: manifest.evaluation_task_ids.len(),
        evaluation_episodes: episodes.len(),
        particle_count: weights.len(),
        effective_sample_size: ess,
        effective_sample_size_fraction: ess / weights.len() as f64,
        top1_accuracy: global.top1_accuracy,
        top1_log_loss: global.top1.log_loss,
        top1_brier: global.top1.brier_score,
        top1_ece: global.top1.expected_calibration_error,
        topk_membership_brier: global.topk_membership_brier,
        topk_membership_brier_skill: global.topk_membership_brier_skill,
        topk_credible_set_coverage: global.topk_credible_set_coverage,
        mean_credible_set_size_fraction: global.mean_credible_set_size_fraction,
        predictive_interval_coverage: global.interval_coverage,
        mean_predictive_interval_width: global.mean_interval_width,
        pit_rank_histogram: global.pit_rank_histogram,
        action_outcome_metrics,
        action_forecast_counts,
        voi_mean_absolute_error: voi_absolute_error / action_count as f64,
        voi_mean_signed_error: voi_signed_error / action_count as f64,
        cost_mean_relative_error: cost_relative_error / action_count as f64,
        strata,
    })
}

#[derive(Debug)]
struct EpisodeSummary {
    top1: CategoricalForecastMetrics,
    top1_accuracy: f64,
    topk_membership_brier: f64,
    topk_membership_brier_skill: f64,
    topk_credible_set_coverage: f64,
    mean_credible_set_size_fraction: f64,
    interval_coverage: f64,
    mean_interval_width: f64,
    pit_rank_histogram: Vec<usize>,
}

fn summarize_episodes(
    manifest: &VoCQualificationManifest,
    episodes: &[&VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
) -> Result<EpisodeSummary, QualificationError> {
    let weights = &manifest.posterior_particle_weights;
    let candidate_count = manifest.candidate_ids.len();
    let mut top1_forecasts = Vec::with_capacity(episodes.len());
    let mut correct_count = 0usize;
    let mut membership_squared_error = 0.0;
    let mut membership_count = 0usize;
    let mut credible_set_coverage = 0usize;
    let mut credible_set_size = 0.0;
    let mut interval_covered = 0usize;
    let mut interval_width = 0.0;
    let mut interval_count = 0usize;
    let mut pit_histogram = vec![0usize; thresholds.rank_histogram_bins];

    for episode in episodes {
        let mut best_probabilities = vec![0.0; candidate_count];
        let mut topk_probabilities = vec![0.0; candidate_count];
        let mut particle_utility_rows = Vec::with_capacity(episode.particle_values.len());
        for (particle_index, row) in episode.particle_values.iter().enumerate() {
            let utility_row = to_utility(row, manifest.metric_direction);
            let order = ranked_candidate_indices(&utility_row, &manifest.candidate_ids);
            best_probabilities[order[0]] += weights[particle_index];
            for candidate_index in order.iter().take(manifest.target_k) {
                topk_probabilities[*candidate_index] += weights[particle_index];
            }
            particle_utility_rows.push(utility_row);
        }

        let actual_utilities = to_utility(&episode.realized_final_values, manifest.metric_direction);
        let actual_order = ranked_candidate_indices(&actual_utilities, &manifest.candidate_ids);
        let actual_best = actual_order[0];
        let predicted_best = argmax_with_tie_break(&best_probabilities, &manifest.candidate_ids);
        if actual_best == predicted_best {
            correct_count += 1;
        }
        top1_forecasts.push((best_probabilities, actual_best));

        let actual_topk: HashSet<usize> = actual_order.iter().take(manifest.target_k).copied().collect();
        let credible_set: Vec<usize> = topk_probabilities
            .iter()
            .enumerate()
            .filter_map(|(index, probability)| {
                (*probability >= thresholds.topk_inclusion_probability_floor).then_some(index)
            })
            .collect();
        if actual_topk.iter().all(|candidate| credible_set.contains(candidate)) {
            credible_set_coverage += 1;
        }
        credible_set_size += credible_set.len() as f64;
        for candidate_index in 0..candidate_count {
            let actual_member = if actual_topk.contains(&candidate_index) {
                1.0
            } else {
                0.0
            };
            membership_squared_error += (topk_probabilities[candidate_index] - actual_member).powi(2);
            membership_count += 1;

            let mut predictive_values = Vec::with_capacity(weights.len());
            for (particle_index, row) in particle_utility_rows.iter().enumerate() {
                predictive_values.push((row[candidate_index], weights[particle_index]));
            }
            let tail_probability = (1.0 - thresholds.nominal_interval_mass) / 2.0;
            let lower = weighted_quantile(&predictive_values, tail_probability);
            let upper = weighted_quantile(&predictive_values, 1.0 - tail_probability);
            let realized = actual_utilities[candidate_index];
            if (lower..=upper).contains(&realized) {
                interval_covered += 1;
            }
            interval_width += (upper - lower).max(0.0);
            interval_count += 1;

            let pit = weighted_midrank(&predictive_values, realized).clamp(0.0, 1.0);
            let bin = ((pit * thresholds.rank_histogram_bins as f64).floor() as usize)
                .min(thresholds.rank_histogram_bins - 1);
            pit_histogram[bin] += 1;
        }
    }

    let top1 = categorical_metrics(&top1_forecasts, thresholds.ece_bins)?;
    let topk_membership_brier = membership_squared_error / membership_count as f64;
    let topk_prevalence = manifest.target_k as f64 / candidate_count as f64;
    let prevalence_brier = topk_prevalence * (1.0 - topk_prevalence);
    let topk_membership_brier_skill = 1.0 - topk_membership_brier / prevalence_brier;
    Ok(EpisodeSummary {
        top1,
        top1_accuracy: correct_count as f64 / episodes.len() as f64,
        topk_membership_brier,
        topk_membership_brier_skill,
        topk_credible_set_coverage: credible_set_coverage as f64 / episodes.len() as f64,
        mean_credible_set_size_fraction: credible_set_size / episodes.len() as f64 / candidate_count as f64,
        interval_coverage: interval_covered as f64 / interval_count as f64,
        mean_interval_width: interval_width / interval_count as f64,
        pit_rank_histogram: pit_histogram,
    })
}

fn compute_strata(
    manifest: &VoCQualificationManifest,
    episodes: &[VoCReplayEpisode],
    thresholds: &VoCQualificationThresholds,
) -> Result<Vec<StratumMetrics>, QualificationError> {
    let mut groups: BTreeMap<String, Vec<&VoCReplayEpisode>> = BTreeMap::new();
    for episode in episodes {
        let keys = [
            format!("family={}", episode.architecture_family),
            format!("fidelity={}", episode.fidelity_id),
            format!("rank={}", episode.rank_regime),
            format!("curve={}", episode.curve_shape),
            format!("noise={}", episode.noise_regime),
        ];
        for key in keys {
            groups.entry(key).or_default().push(episode);
        }
    }
    let mut output = Vec::with_capacity(groups.len());
    for (stratum_id, group) in groups {
        let summary = summarize_episodes(manifest, &group, thresholds)?;
        output.push(StratumMetrics {
            stratum_id,
            episodes: group.len(),
            top1_ece: summary.top1.expected_calibration_error,
            topk_membership_brier: summary.topk_membership_brier,
            topk_membership_brier_skill: summary.topk_membership_brier_skill,
            topk_credible_set_coverage: summary.topk_credible_set_coverage,
            interval_coverage: summary.interval_coverage,
            mean_interval_width: summary.mean_interval_width,
        });
    }
    Ok(output)
}

fn categorical_metrics(
    forecasts: &[(Vec<f64>, usize)],
    ece_bins: usize,
) -> Result<CategoricalForecastMetrics, QualificationError> {
    if forecasts.is_empty() || ece_bins == 0 {
        return Err(QualificationError::new(
            "categorical_forecasts",
            "non-empty forecasts and positive ECE bins are required",
        ));
    }
    let mut log_loss = 0.0;
    let mut brier = 0.0;
    let mut bin_confidence = vec![0.0; ece_bins];
    let mut bin_accuracy = vec![0.0; ece_bins];
    let mut bin_count = vec![0usize; ece_bins];
    for (probabilities, actual_index) in forecasts {
        if probabilities.is_empty() || *actual_index >= probabilities.len() {
            return Err(QualificationError::new(
                "categorical_forecasts",
                "probability vector and realized category disagree",
            ));
        }
        let mut probability_sum = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(QualificationError::new(
                    "categorical_forecasts",
                    "probabilities must be finite and in [0, 1]",
                ));
            }
            probability_sum += probability;
            let target = if index == *actual_index { 1.0 } else { 0.0 };
            brier += (probability - target).powi(2) / forecasts.len() as f64;
        }
        if (probability_sum - 1.0).abs() > 1e-9 {
            return Err(QualificationError::new(
                "categorical_forecasts",
                "probabilities must sum to one",
            ));
        }
        log_loss += -probabilities[*actual_index].max(1e-15).ln() / forecasts.len() as f64;
        let mut predicted_index = 0usize;
        for index in 1..probabilities.len() {
            if probabilities[index] > probabilities[predicted_index] {
                predicted_index = index;
            }
        }
        let confidence = probabilities[predicted_index];
        let bin = ((confidence * ece_bins as f64).floor() as usize).min(ece_bins - 1);
        bin_confidence[bin] += confidence;
        bin_accuracy[bin] += if predicted_index == *actual_index { 1.0 } else { 0.0 };
        bin_count[bin] += 1;
    }
    let mut ece = 0.0;
    for bin in 0..ece_bins {
        if bin_count[bin] > 0 {
            let count = bin_count[bin] as f64;
            ece += count / forecasts.len() as f64
                * (bin_confidence[bin] / count - bin_accuracy[bin] / count).abs();
        }
    }
    Ok(CategoricalForecastMetrics {
        observations: forecasts.len(),
        log_loss,
        brier_score: brier,
        expected_calibration_error: ece,
    })
}

fn to_utility(values: &[f64], direction: MetricDirection) -> Vec<f64> {
    values
        .iter()
        .map(|value| match direction {
            MetricDirection::Maximize => *value,
            MetricDirection::Minimize => 1.0 - *value,
        })
        .collect()
}

fn ranked_candidate_indices(values: &[f64], candidate_ids: &[String]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| candidate_ids[*left].cmp(&candidate_ids[*right]))
    });
    indices
}

fn argmax_with_tie_break(values: &[f64], candidate_ids: &[String]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best]
            || (values[index] == values[best] && candidate_ids[index] < candidate_ids[best])
        {
            best = index;
        }
    }
    best
}

fn weighted_quantile(values: &[(f64, f64)], quantile: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight: f64 = ordered.iter().map(|(_, weight)| *weight).sum();
    let threshold = quantile * total_weight;
    let mut cumulative = 0.0;
    for (value, weight) in &ordered {
        cumulative += *weight;
        if cumulative + 1e-15 >= threshold {
            return *value;
        }
    }
    ordered.last().map(|(value, _)| *value).unwrap_or(0.0)
}

fn weighted_midrank(values: &[(f64, f64)], observed: f64) -> f64 {
    let total_weight: f64 = values.iter().map(|(_, weight)| *weight).sum();
    let below: f64 = values
        .iter()
        .filter(|(value, _)| *value < observed)
        .map(|(_, weight)| *weight)
        .sum();
    let tied: f64 = values
        .iter()
        .filter(|(value, _)| *value == observed)
        .map(|(_, weight)| *weight)
        .sum();
    if total_weight <= 0.0 {
        0.5
    } else {
        (below + 0.5 * tied) / total_weight
    }
}

fn validate_disjoint(
    field: &str,
    fit: &[String],
    calibration: &[String],
    evaluation: &[String],
) -> Result<(), QualificationError> {
    let fit_set: HashSet<&str> = fit.iter().map(String::as_str).collect();
    let calibration_set: HashSet<&str> = calibration.iter().map(String::as_str).collect();
    let evaluation_set: HashSet<&str> = evaluation.iter().map(String::as_str).collect();
    if fit_set.iter().any(|id| calibration_set.contains(id) || evaluation_set.contains(id))
        || calibration_set.iter().any(|id| evaluation_set.contains(id))
    {
        return Err(QualificationError::new(
            field,
            "fit, calibration, and evaluation partitions must be pairwise disjoint",
        ));
    }
    Ok(())
}

fn validate_unique_nonempty(field: &str, ids: &[String]) -> Result<(), QualificationError> {
    let mut seen = HashSet::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        if id.trim().is_empty() || !seen.insert(id.as_str()) {
            return Err(QualificationError::new(
                format!("{field}[{index}]"),
                "IDs must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn check_at_most(name: &str, value: f64, limit: f64, failures: &mut Vec<String>) {
    if !value.is_finite() || value > limit {
        failures.push(format!("{name}: {value} exceeds maximum {limit}"));
    }
}

fn check_at_least(name: &str, value: f64, limit: f64, failures: &mut Vec<String>) {
    if !value.is_finite() || value < limit {
        failures.push(format!("{name}: {value} is below minimum {limit}"));
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_serializable<T: Serialize>(
    suffix: &[u8],
    value: &T,
) -> Result<String, QualificationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| QualificationError::new("serialization", error.to_string()))?;
    let mut domain = DIGEST_DOMAIN.to_vec();
    domain.extend_from_slice(suffix);
    Ok(digest_bytes(&domain, &bytes))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn compute_replay_evidence_sha256(
    episodes: &[VoCReplayEpisode],
) -> Result<String, QualificationError> {
    let mut canonical = episodes.to_vec();
    canonical.sort_by(|left, right| left.episode_id.cmp(&right.episode_id));
    for episode in &mut canonical {
        episode
            .action_forecasts
            .sort_by(|left, right| left.action_id.cmp(&right.action_id));
    }
    digest_serializable(b"replay-evidence", &canonical)
}

fn receipt_digest(receipt: &VoCQualificationReceipt) -> Result<String, QualificationError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    digest_serializable(b"qualification-receipt", &canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn manifest() -> VoCQualificationManifest {
        let mut value = VoCQualificationManifest {
            benchmark_id: "nb201-cifar10".into(),
            metric_id: "validation_accuracy".into(),
            metric_direction: MetricDirection::Maximize,
            source_artifact_sha256: sha('a'),
            posterior_model_sha256: sha('b'),
            action_likelihood_model_sha256: sha('c'),
            fidelity_ladder_sha256: sha('d'),
            fidelity_cost_model_sha256: sha('e'),
            candidate_ids: vec!["A".into(), "B".into()],
            fidelity_ids: vec!["epoch-12".into(), "epoch-36".into(), "full".into()],
            target_k: 1,
            posterior_particle_weights: vec![0.5, 0.5],
            fit_task_ids: vec!["fit-1".into()],
            calibration_task_ids: vec!["cal-1".into()],
            evaluation_task_ids: vec!["eval-1".into()],
            fit_architecture_families: vec!["fit-family".into()],
            calibration_architecture_families: vec!["cal-family".into()],
            evaluation_architecture_families: vec!["eval-family".into()],
            evaluation_rank_regimes: vec!["clear-winner".into()],
            evaluation_curve_shapes: vec!["early-separating".into()],
            evaluation_noise_regimes: vec!["low-noise".into()],
            fit_seed_ids: vec!["fit-seed".into()],
            calibration_seed_ids: vec!["cal-seed".into()],
            evaluation_seed_ids: vec!["eval-seed".into()],
            split_manifest_sha256: String::new(),
        };
        value.split_manifest_sha256 = compute_split_manifest_sha256(&value).unwrap();
        value
    }

    fn action(kind: QualificationActionKind) -> ActionOutcomeForecast {
        ActionOutcomeForecast {
            action_id: format!("action-{}", kind.as_str()),
            kind,
            outcome_ids: vec!["success".into(), "failure".into()],
            predicted_outcome_probabilities: vec![0.9, 0.1],
            realized_outcome_id: "success".into(),
            predicted_expected_regret_reduction: 0.20,
            realized_final_regret_reduction: 0.22,
            planned_cost_units: 10,
            measured_cost_units: 10,
        }
    }

    fn episode(id: &str) -> VoCReplayEpisode {
        VoCReplayEpisode {
            episode_id: id.into(),
            task_id: "eval-1".into(),
            architecture_family: "eval-family".into(),
            seed_id: "eval-seed".into(),
            fidelity_id: "epoch-12".into(),
            rank_regime: "clear-winner".into(),
            curve_shape: "early-separating".into(),
            noise_regime: "low-noise".into(),
            particle_values: vec![vec![0.9, 0.2], vec![0.8, 0.1]],
            realized_final_values: vec![0.85, 0.15],
            action_forecasts: QualificationActionKind::all()
                .into_iter()
                .map(action)
                .collect(),
        }
    }

    fn thresholds() -> VoCQualificationThresholds {
        VoCQualificationThresholds {
            min_fit_tasks: 1,
            min_calibration_tasks: 1,
            min_evaluation_tasks: 1,
            min_evaluation_episodes: 1,
            min_episodes_per_stratum: 1,
            min_action_forecasts_per_kind: 1,
            min_effective_sample_size_fraction: 0.8,
            max_top1_log_loss: 0.1,
            max_top1_brier: 0.1,
            max_top1_ece: 0.1,
            min_topk_membership_brier_skill: 0.1,
            min_topk_credible_set_coverage: 1.0,
            topk_inclusion_probability_floor: 0.05,
            max_mean_credible_set_size_fraction: 1.0,
            nominal_interval_mass: 0.8,
            interval_coverage_tolerance: 0.2,
            max_mean_interval_width: 0.5,
            rank_histogram_bins: 5,
            ece_bins: 5,
            max_action_outcome_log_loss: 0.2,
            max_action_outcome_brier: 0.1,
            max_action_outcome_ece: 0.1,
            max_voi_mean_absolute_error: 0.03,
            max_cost_relative_error: 0.0,
        }
    }

    #[test]
    fn qualified_receipt_replays_and_binds_all_inputs() {
        let manifest = manifest();
        let episodes = vec![episode("ep-1")];
        let thresholds = thresholds();
        let receipt = qualify_voc_model(&manifest, &episodes, &thresholds).unwrap();
        assert_eq!(receipt.decision, QualificationDecision::Qualified);
        assert_eq!(receipt.metrics.top1_accuracy, 1.0);
        assert_eq!(receipt.metrics.topk_credible_set_coverage, 1.0);
        assert_eq!(receipt.metrics.effective_sample_size_fraction, 1.0);
        verify_voc_qualification_receipt(&manifest, &episodes, &thresholds, &receipt).unwrap();
    }

    #[test]
    fn receipt_tampering_is_detected() {
        let manifest = manifest();
        let episodes = vec![episode("ep-1")];
        let thresholds = thresholds();
        let mut receipt = qualify_voc_model(&manifest, &episodes, &thresholds).unwrap();
        receipt.metrics.top1_accuracy = 0.0;
        assert!(verify_voc_qualification_receipt(&manifest, &episodes, &thresholds, &receipt).is_err());
    }

    #[test]
    fn rejects_task_architecture_and_seed_leakage() {
        let mut value = manifest();
        value.evaluation_task_ids = vec!["fit-1".into()];
        value.split_manifest_sha256 = compute_split_manifest_sha256(&value).unwrap();
        assert!(qualify_voc_model(&value, &[episode("ep-1")], &thresholds()).is_err());
    }

    #[test]
    fn refuses_non_normalized_action_likelihood_forecasts() {
        let manifest = manifest();
        let mut replay = episode("ep-1");
        replay.action_forecasts[0].predicted_outcome_probabilities = vec![0.8, 0.8];
        assert!(qualify_voc_model(&manifest, &[replay], &thresholds()).is_err());
    }

    #[test]
    fn poorly_calibrated_action_forecasts_force_abstention() {
        let manifest = manifest();
        let mut replay = episode("ep-1");
        for forecast in &mut replay.action_forecasts {
            forecast.predicted_outcome_probabilities = vec![0.01, 0.99];
        }
        let receipt = qualify_voc_model(&manifest, &[replay], &thresholds()).unwrap();
        assert_eq!(receipt.decision, QualificationDecision::Abstain);
        assert!(!receipt.failure_reasons.is_empty());
    }

    #[test]
    fn low_effective_sample_size_forces_abstention() {
        let mut value = manifest();
        value.posterior_particle_weights = vec![0.999, 0.001];
        value.split_manifest_sha256 = compute_split_manifest_sha256(&value).unwrap();
        let receipt = qualify_voc_model(&value, &[episode("ep-1")], &thresholds()).unwrap();
        assert_eq!(receipt.decision, QualificationDecision::Abstain);
        assert!(receipt.metrics.effective_sample_size_fraction < 0.8);
    }

    #[test]
    fn minimize_metric_direction_is_normalized_to_utility() {
        let mut value = manifest();
        value.metric_direction = MetricDirection::Minimize;
        value.split_manifest_sha256 = compute_split_manifest_sha256(&value).unwrap();
        let mut replay = episode("ep-1");
        replay.particle_values = vec![vec![0.1, 0.8], vec![0.2, 0.9]];
        replay.realized_final_values = vec![0.15, 0.85];
        let receipt = qualify_voc_model(&value, &[replay], &thresholds()).unwrap();
        assert_eq!(receipt.metrics.top1_accuracy, 1.0);
    }

    #[test]
    fn rejects_inconsistent_outcome_categories_within_an_action_class() {
        let manifest = manifest();
        let mut first = episode("ep-1");
        let mut second = episode("ep-2");
        second.action_forecasts[0].outcome_ids = vec!["other".into(), "success".into()];
        let error = qualify_voc_model(&manifest, &[first, second], &thresholds())
            .expect_err("different category meanings cannot share one Brier score");
        assert!(error.to_string().contains("inconsistent outcome categories"));
    }

    #[test]
    fn accepts_reordered_but_semantically_identical_outcome_categories() {
        let manifest = manifest();
        let mut first = episode("ep-1");
        let mut second = episode("ep-2");
        second.action_forecasts[0].outcome_ids.reverse();
        second.action_forecasts[0].predicted_outcome_probabilities.reverse();
        let receipt = qualify_voc_model(&manifest, &[first, second], &thresholds()).unwrap();
        assert_eq!(receipt.decision, QualificationDecision::Qualified);
    }

    #[test]
    fn fidelity_order_is_part_of_the_split_manifest_digest() {
        let first = manifest();
        let mut second = first.clone();
        second.fidelity_ids.reverse();
        second.split_manifest_sha256 = compute_split_manifest_sha256(&second).unwrap();
        assert_ne!(first.split_manifest_sha256, second.split_manifest_sha256);
    }

    #[test]
    fn membership_brier_skill_is_zero_for_the_constant_prevalence_baseline() {
        // With two candidates and K=1, the constant 0.5 forecast has Brier 0.25,
        // exactly matching the K/N prevalence baseline.
        let manifest = manifest();
        let mut replay = episode("ep-1");
        replay.particle_values = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        replay.realized_final_values = vec![0.9, 0.1];
        let receipt = qualify_voc_model(&manifest, &[replay], &thresholds()).unwrap();
        assert!((receipt.metrics.topk_membership_brier_skill - 0.0).abs() < 1e-12);
        assert_eq!(receipt.decision, QualificationDecision::Abstain);
    }

    #[test]
    fn negative_topk_membership_brier_skill_fails_the_gate() {
        let manifest = manifest();
        let mut replay = episode("ep-1");
        replay.particle_values = vec![vec![0.1, 0.9], vec![0.1, 0.9]];
        replay.realized_final_values = vec![0.9, 0.1];
        let receipt = qualify_voc_model(&manifest, &[replay], &thresholds()).unwrap();
        assert!(receipt.metrics.topk_membership_brier_skill < 0.0);
        assert_eq!(receipt.decision, QualificationDecision::Abstain);
    }

    #[test]
    fn rejects_an_episode_outside_the_declared_evaluation_strata() {
        let manifest = manifest();
        let mut replay = episode("ep-1");
        replay.noise_regime = "unlisted-noise".into();
        assert!(qualify_voc_model(&manifest, &[replay], &thresholds()).is_err());
    }

    #[test]
    fn replay_evidence_digest_is_order_invariant() {
        let manifest = manifest();
        let mut first = episode("ep-1");
        let mut second = episode("ep-2");
        first.action_forecasts.reverse();
        second.action_forecasts.reverse();
        let a = compute_replay_evidence_sha256(&[first.clone(), second.clone()]).unwrap();
        let b = compute_replay_evidence_sha256(&[second, first]).unwrap();
        assert_eq!(a, b);
    }
}
