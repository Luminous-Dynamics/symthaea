//! Replicated model-pair causal analysis for randomized contamination interventions.
//!
//! The independent training replicate is a treatment/placebo model pair.
//! Source clusters are randomized units and repeated questions are nested
//! measurements, never independent training replicates.
//!
//! Primary estimand: per-replicate cluster-weighted mean of treatment-model
//! score minus placebo-model score over the full benchmark cluster universe,
//! then the mean of those replicate-level effects.
//!
//! Exposed/unexposed localization is a secondary diagnostic. Because model
//! parameters are global, exposure to one cluster can affect outcomes on all
//! clusters. The localization contrast must not be interpreted as a pure
//! direct effect without an additional interference model.
//!
//! The randomization p-value tests a Fisher sharp null that assignment has no
//! effect on any paired-advantage outcome. It is not a confidence interval for
//! a direct effect under arbitrary interference.
//!
//! Receipt audit digests are consistency evidence, not cryptographic
//! authentication. This module therefore marks results exploratory-only until
//! the external signature/trust-root layer exists.

use crate::{
    hash_domain,
    intervention::{
        RandomizedInterventionDesignV1, RandomizedInterventionManifestV1, audit_intervention,
        randomize_intervention,
    },
    training_receipt::PairedTrainingExecutionAuditV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CAUSAL_STUDY_SCHEMA: &str = "symthaea.replicated_contamination_causal_study.v1";
pub const CAUSAL_REPORT_SCHEMA: &str = "symthaea.replicated_contamination_causal_report.v1";
pub const CAUSAL_MIN_CONFIRMATORY_REPLICATES: usize = 20;
pub const CAUSAL_MIN_RANDOMIZATION_DRAWS: u32 = 999;
pub const CAUSAL_MAX_RANDOMIZATION_DRAWS: u32 = 100_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreCountV1 {
    pub success_count: u64,
    pub observed_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterOutcomeV1 {
    pub cluster_id: String,
    pub task_family: String,
    pub assigned_to_exposure: bool,
    pub item_count: u64,
    /// Digest of the exact item IDs evaluated for base, treatment and placebo.
    pub evaluation_item_set_sha256: String,
    pub base: ScoreCountV1,
    pub treatment: ScoreCountV1,
    pub placebo: ScoreCountV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationEvidenceV1 {
    pub base_checkpoint_sha256: String,
    pub treatment_checkpoint_sha256: String,
    pub placebo_checkpoint_sha256: String,
    pub base_evaluation_receipt_sha256: String,
    pub treatment_evaluation_receipt_sha256: String,
    pub placebo_evaluation_receipt_sha256: String,
    pub metric_definition_sha256: String,
    pub oracle_or_adjudication_sha256: String,
    pub evaluation_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalReplicateInputV1 {
    pub design: RandomizedInterventionDesignV1,
    pub manifest: RandomizedInterventionManifestV1,
    /// Original randomization seed, held by the independent evaluator.
    pub randomization_seed_hex: String,
    pub training_receipt_audit: PairedTrainingExecutionAuditV1,
    pub training_receipt_audit_sha256: String,
    pub evaluation_evidence: EvaluationEvidenceV1,
    pub cluster_outcomes: Vec<ClusterOutcomeV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicatedCausalStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_metric_id: String,
    pub primary_metric_definition_sha256: String,
    pub minimum_confirmatory_replicates: u32,
    pub randomization_test_draws: u32,
    pub randomization_test_seed_hex: String,
    pub replicates: Vec<CausalReplicateInputV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyEffectV1 {
    pub task_family: String,
    pub cluster_count: u64,
    pub base_score_permille: f64,
    pub treatment_score_permille: f64,
    pub placebo_score_permille: f64,
    pub treatment_minus_placebo_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicateEffectV1 {
    pub replicate_id: String,
    pub cluster_count: u64,
    pub item_count: u64,
    pub base_score_cluster_weighted_permille: f64,
    pub treatment_score_cluster_weighted_permille: f64,
    pub placebo_score_cluster_weighted_permille: f64,
    pub treatment_minus_placebo_cluster_weighted_permille: f64,
    pub treatment_minus_base_cluster_weighted_permille: f64,
    pub placebo_minus_base_cluster_weighted_permille: f64,
    pub treatment_score_item_weighted_permille: f64,
    pub placebo_score_item_weighted_permille: f64,
    pub treatment_minus_placebo_item_weighted_permille: f64,
    pub exposed_cluster_advantage_permille: f64,
    pub unexposed_cluster_advantage_permille: f64,
    pub exposure_localization_contrast_permille: f64,
    pub families: Vec<FamilyEffectV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeanIntervalV1 {
    pub mean: f64,
    pub standard_deviation_across_replicates: f64,
    pub standard_error: f64,
    pub confidence_level_permille: u16,
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicatedCausalAnalysisReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_metric_id: String,
    pub study_sha256: String,
    pub randomization_test_seed_commitment_sha256: String,
    pub independent_model_pair_replicate_count: u64,
    pub minimum_confirmatory_replicates: u32,
    pub replicate_count_meets_minimum: bool,
    pub every_replicate_has_consistent_training_receipt: bool,
    pub training_receipts_independently_verified: bool,
    pub execution_evidence_ready_for_external_verification: bool,
    pub primary_cluster_weighted_effect_permille: MeanIntervalV1,
    pub secondary_item_weighted_effect_permille: MeanIntervalV1,
    pub treatment_minus_base_effect_permille: MeanIntervalV1,
    pub placebo_minus_base_effect_permille: MeanIntervalV1,
    pub exposed_localization_effect_permille: MeanIntervalV1,
    pub unexposed_spillover_effect_permille: MeanIntervalV1,
    pub exposure_localization_contrast_permille: MeanIntervalV1,
    pub localization_randomization_p_value: f64,
    pub localization_randomization_draws: u32,
    pub randomization_null: String,
    pub family_effects: Vec<FamilyEffectV1>,
    pub replicate_effects: Vec<ReplicateEffectV1>,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn decode_seed_hex(value: &str, label: &str) -> Result<Vec<u8>, String> {
    if value.len() < 64 || value.len() % 2 != 0 {
        return Err(format!("{label} must encode at least 32 bytes of hex"));
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        };
        let high = digit(pair[0]).ok_or_else(|| format!("{label} must be lowercase hex"))?;
        let low = digit(pair[1]).ok_or_else(|| format!("{label} must be lowercase hex"))?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn rate_permille(count: &ScoreCountV1, label: &str) -> Result<f64, String> {
    if count.observed_count == 0 || count.success_count > count.observed_count {
        return Err(format!("{label} has invalid success/observed counts"));
    }
    Ok(count.success_count as f64 * 1_000.0 / count.observed_count as f64)
}

fn sample_mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot calculate mean of empty replicate values".to_owned());
    }
    let value = values.iter().sum::<f64>() / values.len() as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("replicate mean is non-finite".to_owned())
    }
}

fn sample_sd(values: &[f64], center: f64) -> Result<f64, String> {
    if values.len() < 2 {
        return Err("at least two independent model-pair replicates are required".to_owned());
    }
    let sum = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>();
    let sd = (sum / (values.len() - 1) as f64).sqrt();
    if sd.is_finite() {
        Ok(sd)
    } else {
        Err("replicate standard deviation is non-finite".to_owned())
    }
}

fn t_critical_975(df: usize) -> f64 {
    // Conservative upper-df table for two-sided 95% Student-t intervals.
    const TABLE: &[(usize, f64)] = &[
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (6, 2.447),
        (7, 2.365),
        (8, 2.306),
        (9, 2.262),
        (10, 2.228),
        (11, 2.201),
        (12, 2.179),
        (13, 2.160),
        (14, 2.145),
        (15, 2.131),
        (16, 2.120),
        (17, 2.110),
        (18, 2.101),
        (19, 2.093),
        (20, 2.086),
        (21, 2.080),
        (22, 2.074),
        (23, 2.069),
        (24, 2.064),
        (25, 2.060),
        (26, 2.056),
        (27, 2.052),
        (28, 2.048),
        (29, 2.045),
        (30, 2.042),
        (40, 2.042),
        (60, 2.021),
        (120, 2.000),
        (usize::MAX, 1.980),
    ];
    TABLE
        .iter()
        .find(|(upper_df, _)| df <= *upper_df)
        .map(|(_, value)| *value)
        .unwrap_or(1.96)
}

fn mean_interval(values: &[f64]) -> Result<MeanIntervalV1, String> {
    let mean = sample_mean(values)?;
    let sd = sample_sd(values, mean)?;
    let standard_error = sd / (values.len() as f64).sqrt();
    let critical = t_critical_975(values.len() - 1);
    let margin = critical * standard_error;
    Ok(MeanIntervalV1 {
        mean,
        standard_deviation_across_replicates: sd,
        standard_error,
        confidence_level_permille: 950,
        lower_95: mean - margin,
        upper_95: mean + margin,
    })
}

fn validate_common_design(
    first: &RandomizedInterventionDesignV1,
    next: &RandomizedInterventionDesignV1,
) -> Result<(), String> {
    if first.benchmark_id != next.benchmark_id
        || first.benchmark_version != next.benchmark_version
        || first.benchmark_task_set_sha256 != next.benchmark_task_set_sha256
        || first.model_base_checkpoint_sha256 != next.model_base_checkpoint_sha256
        || first.training_stage != next.training_stage
        || first.exposure_mechanism != next.exposure_mechanism
        || first.target_exposure_fraction_permille != next.target_exposure_fraction_permille
        || first.total_training_token_budget != next.total_training_token_budget
        || first.shared_training_corpus_sha256 != next.shared_training_corpus_sha256
        || first.shared_training_token_count != next.shared_training_token_count
        || first.training_steps != next.training_steps
        || first.training_code_sha256 != next.training_code_sha256
        || first.runtime_image_sha256 != next.runtime_image_sha256
        || first.tokenization_pipeline_sha256 != next.tokenization_pipeline_sha256
        || first.training_hyperparameters_sha256 != next.training_hyperparameters_sha256
        || first.optimizer_schedule_sha256 != next.optimizer_schedule_sha256
    {
        return Err(
            "replicate designs differ on preregistered common benchmark/training conditions"
                .to_owned(),
        );
    }
    let first_clusters: BTreeMap<String, (String, String, usize, u64, String, String)> = first
        .clusters
        .iter()
        .map(|cluster| {
            (
                cluster.cluster_id.clone(),
                (
                    cluster.task_family.clone(),
                    cluster.source_family_sha256.clone(),
                    cluster.item_ids.len(),
                    cluster.exposure_token_count,
                    cluster.exposure_variant_sha256.clone(),
                    cluster.matched_placebo_record_sha256.clone(),
                ),
            )
        })
        .collect();
    let next_clusters: BTreeMap<String, (String, String, usize, u64, String, String)> = next
        .clusters
        .iter()
        .map(|cluster| {
            (
                cluster.cluster_id.clone(),
                (
                    cluster.task_family.clone(),
                    cluster.source_family_sha256.clone(),
                    cluster.item_ids.len(),
                    cluster.exposure_token_count,
                    cluster.exposure_variant_sha256.clone(),
                    cluster.matched_placebo_record_sha256.clone(),
                ),
            )
        })
        .collect();
    if first_clusters != next_clusters {
        return Err("replicate source-cluster universe or cluster records differ".to_owned());
    }
    Ok(())
}

fn validate_replicate(
    replicate: &CausalReplicateInputV1,
    expected_benchmark_id: &str,
    expected_benchmark_version: &str,
    expected_metric_digest: &str,
) -> Result<(), String> {
    if !(1..1_000).contains(&replicate.design.target_exposure_fraction_permille) {
        return Err(
            "causal localization requires an interior exposure fraction in [1, 999] permille"
                .to_owned(),
        );
    }
    let seed = decode_seed_hex(&replicate.randomization_seed_hex, "randomization_seed_hex")?;
    audit_intervention(&replicate.design, &replicate.manifest, &seed)?;
    if replicate.design.benchmark_id != expected_benchmark_id
        || replicate.design.benchmark_version != expected_benchmark_version
    {
        return Err("replicate benchmark identity differs from study".to_owned());
    }
    validate_digest(
        &replicate.training_receipt_audit_sha256,
        "training_receipt_audit_sha256",
    )?;
    if !replicate.training_receipt_audit.valid
        || !replicate.training_receipt_audit.execution_consistent
        || !replicate.training_receipt_audit.treatment_completed
        || !replicate.training_receipt_audit.placebo_completed
    {
        return Err(
            "replicate lacks a consistent completed paired training execution audit".to_owned(),
        );
    }
    let audit_bytes = serde_json::to_vec(&replicate.training_receipt_audit)
        .map_err(|error| format!("cannot serialize training receipt audit: {error}"))?;
    let expected_audit_digest = hash_domain(b"paired-training-execution-audit.v1", &audit_bytes);
    if replicate.training_receipt_audit_sha256 != expected_audit_digest {
        return Err("training receipt audit digest does not match report contents".to_owned());
    }
    if replicate.training_receipt_audit.independently_verified {
        return Err(
            "self-asserted independent verification cannot be trusted by this analysis module"
                .to_owned(),
        );
    }
    if !replicate.training_receipt_audit.paired_checkpoints_distinct
        || !replicate.training_receipt_audit.base_checkpoint_matched
        || !replicate
            .training_receipt_audit
            .training_configuration_matched
        || !replicate
            .training_receipt_audit
            .corpus_identities_matched_to_design
        || !replicate
            .training_receipt_audit
            .intervention_payloads_matched_to_assignment
        || !replicate
            .training_receipt_audit
            .observed_token_budgets_complete
        || !replicate
            .training_receipt_audit
            .observed_step_budgets_complete
        || !replicate
            .training_receipt_audit
            .runner_attestation_digests_present
        || !replicate
            .training_receipt_audit
            .checkpoint_custody_digests_present
        || !replicate
            .training_receipt_audit
            .ready_for_independent_attestation_verification
    {
        return Err("training execution audit is missing a required consistency or external-attestation receipt".to_owned());
    }
    if replicate
        .training_receipt_audit
        .training_rng_seed_commitment_sha256
        != replicate.design.training_rng_seed_commitment_sha256
        || replicate.training_receipt_audit.design_sha256 != replicate.manifest.design_sha256
        || replicate
            .training_receipt_audit
            .intervention_manifest_sha256
            != replicate.manifest.manifest_sha256
        || replicate.training_receipt_audit.treatment_checkpoint_sha256
            != replicate.evaluation_evidence.treatment_checkpoint_sha256
        || replicate.training_receipt_audit.placebo_checkpoint_sha256
            != replicate.evaluation_evidence.placebo_checkpoint_sha256
        || replicate.design.model_base_checkpoint_sha256
            != replicate.evaluation_evidence.base_checkpoint_sha256
    {
        return Err("training audit and evaluation checkpoint identities do not match".to_owned());
    }
    for (digest, label) in [
        (
            &replicate.evaluation_evidence.base_checkpoint_sha256,
            "base_checkpoint_sha256",
        ),
        (
            &replicate.evaluation_evidence.treatment_checkpoint_sha256,
            "treatment_checkpoint_sha256",
        ),
        (
            &replicate.evaluation_evidence.placebo_checkpoint_sha256,
            "placebo_checkpoint_sha256",
        ),
        (
            &replicate.evaluation_evidence.base_evaluation_receipt_sha256,
            "base_evaluation_receipt_sha256",
        ),
        (
            &replicate
                .evaluation_evidence
                .treatment_evaluation_receipt_sha256,
            "treatment_evaluation_receipt_sha256",
        ),
        (
            &replicate
                .evaluation_evidence
                .placebo_evaluation_receipt_sha256,
            "placebo_evaluation_receipt_sha256",
        ),
        (
            &replicate.evaluation_evidence.metric_definition_sha256,
            "metric_definition_sha256",
        ),
        (
            &replicate.evaluation_evidence.oracle_or_adjudication_sha256,
            "oracle_or_adjudication_sha256",
        ),
        (
            &replicate.evaluation_evidence.evaluation_harness_sha256,
            "evaluation_harness_sha256",
        ),
        (
            &replicate.evaluation_evidence.prompt_and_decoding_sha256,
            "prompt_and_decoding_sha256",
        ),
    ] {
        validate_digest(digest, label)?;
    }
    if replicate.evaluation_evidence.metric_definition_sha256 != expected_metric_digest {
        return Err("replicate metric definition digest differs from study".to_owned());
    }
    if replicate
        .evaluation_evidence
        .treatment_evaluation_receipt_sha256
        == replicate
            .evaluation_evidence
            .placebo_evaluation_receipt_sha256
        || replicate.evaluation_evidence.base_evaluation_receipt_sha256
            == replicate
                .evaluation_evidence
                .treatment_evaluation_receipt_sha256
        || replicate.evaluation_evidence.base_evaluation_receipt_sha256
            == replicate
                .evaluation_evidence
                .placebo_evaluation_receipt_sha256
    {
        return Err(
            "base/treatment/placebo evaluation receipt digests must be distinct".to_owned(),
        );
    }
    if replicate.cluster_outcomes.len() != replicate.design.clusters.len() {
        return Err(
            "each replicate must provide exactly one outcome per source cluster".to_owned(),
        );
    }
    let assignment_map: BTreeMap<&str, (&str, bool, usize)> = replicate
        .manifest
        .assignments
        .iter()
        .map(|assignment| {
            (
                assignment.cluster_id.as_str(),
                (
                    assignment.task_family.as_str(),
                    assignment.assigned_to_exposure,
                    assignment.item_ids.len(),
                ),
            )
        })
        .collect();
    let mut observed_clusters = BTreeSet::new();
    for outcome in &replicate.cluster_outcomes {
        if !observed_clusters.insert(outcome.cluster_id.as_str()) {
            return Err(format!("duplicate cluster outcome {}", outcome.cluster_id));
        }
        let (task_family, assigned, item_count) =
            assignment_map
                .get(outcome.cluster_id.as_str())
                .ok_or_else(|| format!("unknown cluster outcome {}", outcome.cluster_id))?;
        if *task_family != outcome.task_family.as_str()
            || *assigned != outcome.assigned_to_exposure
            || u64::try_from(*item_count).ok() != Some(outcome.item_count)
        {
            return Err(format!(
                "cluster outcome {} does not match assignment manifest",
                outcome.cluster_id
            ));
        }
        validate_digest(
            &outcome.evaluation_item_set_sha256,
            "evaluation_item_set_sha256",
        )?;
        let expected_item_ids = replicate
            .design
            .clusters
            .iter()
            .find(|cluster| cluster.cluster_id.as_str() == outcome.cluster_id.as_str())
            .map(|cluster| {
                let mut ids = cluster.item_ids.clone();
                ids.sort();
                ids
            })
            .ok_or_else(|| format!("cluster {} missing from design", outcome.cluster_id))?;
        let expected_item_set_sha256 = hash_domain(
            b"causal-evaluation-item-set.v1",
            &serde_json::to_vec(&expected_item_ids)
                .map_err(|error| format!("cannot serialize item IDs: {error}"))?,
        );
        if outcome.evaluation_item_set_sha256 != expected_item_set_sha256 {
            return Err(format!(
                "cluster {} evaluation item-set digest mismatch",
                outcome.cluster_id
            ));
        }
        for (count, label) in [
            (&outcome.base, "base"),
            (&outcome.treatment, "treatment"),
            (&outcome.placebo, "placebo"),
        ] {
            rate_permille(count, label)?;
            if count.observed_count != outcome.item_count {
                return Err(format!(
                    "cluster {} {label} denominator does not cover the full item set",
                    outcome.cluster_id
                ));
            }
        }
        if outcome.base.observed_count != outcome.treatment.observed_count
            || outcome.base.observed_count != outcome.placebo.observed_count
        {
            return Err(format!(
                "cluster {} base/treatment/placebo denominators differ",
                outcome.cluster_id
            ));
        }
    }
    if observed_clusters.len() != assignment_map.len() {
        return Err("cluster outcome set is incomplete relative to assignments".to_owned());
    }
    Ok(())
}

fn replicate_effect(replicate: &CausalReplicateInputV1) -> Result<ReplicateEffectV1, String> {
    let mut base_rates = Vec::new();
    let mut treatment_rates = Vec::new();
    let mut placebo_rates = Vec::new();
    let mut item_base_success = 0_u64;
    let mut item_treatment_success = 0_u64;
    let mut item_placebo_success = 0_u64;
    let mut item_total = 0_u64;
    let mut exposed_advantages = Vec::new();
    let mut unexposed_advantages = Vec::new();
    let mut family_rows = BTreeMap::<String, Vec<(f64, f64, f64)>>::new();

    for outcome in &replicate.cluster_outcomes {
        let base = rate_permille(&outcome.base, "base")?;
        let treatment = rate_permille(&outcome.treatment, "treatment")?;
        let placebo = rate_permille(&outcome.placebo, "placebo")?;
        base_rates.push(base);
        treatment_rates.push(treatment);
        placebo_rates.push(placebo);
        item_base_success = item_base_success
            .checked_add(outcome.base.success_count)
            .ok_or_else(|| "base item success count overflows".to_owned())?;
        item_treatment_success = item_treatment_success
            .checked_add(outcome.treatment.success_count)
            .ok_or_else(|| "treatment item success count overflows".to_owned())?;
        item_placebo_success = item_placebo_success
            .checked_add(outcome.placebo.success_count)
            .ok_or_else(|| "placebo item success count overflows".to_owned())?;
        item_total = item_total
            .checked_add(outcome.item_count)
            .ok_or_else(|| "item total overflows".to_owned())?;
        let advantage = treatment - placebo;
        if outcome.assigned_to_exposure {
            exposed_advantages.push(advantage);
        } else {
            unexposed_advantages.push(advantage);
        }
        family_rows
            .entry(outcome.task_family.clone())
            .or_default()
            .push((base, treatment, placebo));
    }

    if exposed_advantages.is_empty() || unexposed_advantages.is_empty() {
        return Err(
            "localization analysis requires both exposed and unexposed clusters".to_owned(),
        );
    }
    let base_score = sample_mean(&base_rates)?;
    let treatment_score = sample_mean(&treatment_rates)?;
    let placebo_score = sample_mean(&placebo_rates)?;
    let exposed_mean = sample_mean(&exposed_advantages)?;
    let unexposed_mean = sample_mean(&unexposed_advantages)?;
    let stratified_localization = stratum_localization(&replicate.cluster_outcomes, None)?;
    let item_base = item_base_success as f64 * 1_000.0 / item_total as f64;
    let item_treatment = item_treatment_success as f64 * 1_000.0 / item_total as f64;
    let item_placebo = item_placebo_success as f64 * 1_000.0 / item_total as f64;

    let mut families = Vec::new();
    for (task_family, values) in family_rows {
        let bases: Vec<f64> = values.iter().map(|row| row.0).collect();
        let treatments: Vec<f64> = values.iter().map(|row| row.1).collect();
        let placebos: Vec<f64> = values.iter().map(|row| row.2).collect();
        let base_family = sample_mean(&bases)?;
        let treatment_family = sample_mean(&treatments)?;
        let placebo_family = sample_mean(&placebos)?;
        families.push(FamilyEffectV1 {
            task_family,
            cluster_count: u64::try_from(values.len())
                .map_err(|_| "family cluster count overflows")?,
            base_score_permille: base_family,
            treatment_score_permille: treatment_family,
            placebo_score_permille: placebo_family,
            treatment_minus_placebo_permille: treatment_family - placebo_family,
        });
    }
    let item_count = item_total;
    Ok(ReplicateEffectV1 {
        replicate_id: replicate.design.replicate_id.clone(),
        cluster_count: u64::try_from(replicate.cluster_outcomes.len())
            .map_err(|_| "cluster count overflows")?,
        item_count,
        base_score_cluster_weighted_permille: base_score,
        treatment_score_cluster_weighted_permille: treatment_score,
        placebo_score_cluster_weighted_permille: placebo_score,
        treatment_minus_placebo_cluster_weighted_permille: treatment_score - placebo_score,
        treatment_minus_base_cluster_weighted_permille: treatment_score - base_score,
        placebo_minus_base_cluster_weighted_permille: placebo_score - base_score,
        treatment_score_item_weighted_permille: item_treatment,
        placebo_score_item_weighted_permille: item_placebo,
        treatment_minus_placebo_item_weighted_permille: item_treatment - item_placebo,
        exposed_cluster_advantage_permille: exposed_mean,
        unexposed_cluster_advantage_permille: unexposed_mean,
        exposure_localization_contrast_permille: stratified_localization,
        families,
    })
}

fn stratum_localization(
    outcomes: &[ClusterOutcomeV1],
    assigned: Option<&BTreeMap<String, bool>>,
) -> Result<f64, String> {
    let mut strata = BTreeMap::<(String, String), Vec<(bool, f64)>>::new();
    for outcome in outcomes {
        let exposure = match assigned {
            Some(map) => *map
                .get(&outcome.cluster_id)
                .ok_or_else(|| "permutation assignment missing a cluster".to_owned())?,
            None => outcome.assigned_to_exposure,
        };
        let advantage = rate_permille(&outcome.treatment, "treatment")?
            - rate_permille(&outcome.placebo, "placebo")?;
        let design_bucket = match outcome.item_count {
            0..=4 => "small",
            5..=16 => "medium",
            _ => "large",
        }
        .to_owned();
        strata
            .entry((outcome.task_family.clone(), design_bucket))
            .or_default()
            .push((exposure, advantage));
    }
    let total = outcomes.len() as f64;
    let mut weighted = 0.0;
    for values in strata.values() {
        let exposed: Vec<f64> = values
            .iter()
            .filter(|value| value.0)
            .map(|value| value.1)
            .collect();
        let unexposed: Vec<f64> = values
            .iter()
            .filter(|value| !value.0)
            .map(|value| value.1)
            .collect();
        if exposed.is_empty() || unexposed.is_empty() {
            return Err(
                "localization stratum must contain exposed and unexposed clusters".to_owned(),
            );
        }
        weighted +=
            values.len() as f64 / total * (sample_mean(&exposed)? - sample_mean(&unexposed)?);
    }
    Ok(weighted)
}

fn derive_seed(root: &[u8], replicate_id: &str, draw: u32) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.causal-randomization-test-seed.v1\0");
    hasher.update(root);
    hasher.update((replicate_id.len() as u64).to_be_bytes());
    hasher.update(replicate_id.as_bytes());
    hasher.update(draw.to_be_bytes());
    hasher.finalize().to_vec()
}

fn randomization_p_value(
    replicates: &[CausalReplicateInputV1],
    observed_statistic: f64,
    root_seed: &[u8],
    draws: u32,
) -> Result<f64, String> {
    let mut extreme = 0_u64;
    for draw in 0..draws {
        let mut simulated_sum = 0.0;
        for replicate in replicates {
            let seed = derive_seed(root_seed, &replicate.design.replicate_id, draw);
            let simulated = randomize_intervention(&replicate.design, &seed)?;
            let assignment_map: BTreeMap<String, bool> = simulated
                .assignments
                .iter()
                .map(|assignment| {
                    (
                        assignment.cluster_id.clone(),
                        assignment.assigned_to_exposure,
                    )
                })
                .collect();
            simulated_sum +=
                stratum_localization(&replicate.cluster_outcomes, Some(&assignment_map))?;
        }
        let simulated_statistic = simulated_sum / replicates.len() as f64;
        if simulated_statistic.abs() + 1.0e-12 >= observed_statistic.abs() {
            extreme += 1;
        }
    }
    Ok((extreme as f64 + 1.0) / (draws as f64 + 1.0))
}

/// Analyze independent treatment/placebo model-pair replicates.
///
/// This function is exploratory until the training receipt's external
/// signatures are authenticated under pinned trust roots. Each replicate must
/// contain exactly one outcome per source cluster, evaluated on the same item
/// set for base, treatment and placebo checkpoints.
pub fn analyze_replicated_causal_study(
    study: &ReplicatedCausalStudyV1,
) -> Result<ReplicatedCausalAnalysisReportV1, String> {
    if study.schema != CAUSAL_STUDY_SCHEMA {
        return Err(format!("study schema must be {CAUSAL_STUDY_SCHEMA}"));
    }
    if study.study_id.trim().is_empty()
        || study.study_id.len() > 128
        || study.benchmark_id.trim().is_empty()
        || study.benchmark_version.trim().is_empty()
        || study.primary_metric_id.trim().is_empty()
    {
        return Err("study and primary metric identities must be nonempty".to_owned());
    }
    validate_digest(
        &study.primary_metric_definition_sha256,
        "primary_metric_definition_sha256",
    )?;
    if study.minimum_confirmatory_replicates < CAUSAL_MIN_CONFIRMATORY_REPLICATES as u32 {
        return Err(format!(
            "minimum_confirmatory_replicates must be at least {CAUSAL_MIN_CONFIRMATORY_REPLICATES}"
        ));
    }
    if !(CAUSAL_MIN_RANDOMIZATION_DRAWS..=CAUSAL_MAX_RANDOMIZATION_DRAWS)
        .contains(&study.randomization_test_draws)
    {
        return Err(format!(
            "randomization_test_draws must be in [{CAUSAL_MIN_RANDOMIZATION_DRAWS}, {CAUSAL_MAX_RANDOMIZATION_DRAWS}]"
        ));
    }
    let root_seed = decode_seed_hex(
        &study.randomization_test_seed_hex,
        "randomization_test_seed_hex",
    )?;
    if study.replicates.len() < 2 {
        return Err(
            "at least two independent model-pair replicates are required for uncertainty"
                .to_owned(),
        );
    }
    if study.replicates.len() > 1_000 {
        return Err(
            "study must contain no more than 1000 independent model-pair replicates".to_owned(),
        );
    }

    let first = &study.replicates[0].design;
    let first_evaluation = &study.replicates[0].evaluation_evidence;
    let mut replicate_ids = BTreeSet::new();
    let mut seed_commitments = BTreeSet::new();
    let mut training_seed_commitments = BTreeSet::new();
    let mut all_checkpoint_digests = BTreeSet::new();
    let mut all_training_audit_digests = BTreeSet::new();
    let mut all_evaluation_receipt_digests = BTreeSet::new();
    let mut replicate_effects = Vec::with_capacity(study.replicates.len());
    let independent_receipts_verified = false;
    let mut all_receipts_consistent = true;
    let mut all_ready_for_external_verification = true;
    let mut issues = Vec::new();

    for replicate in &study.replicates {
        validate_common_design(first, &replicate.design)?;
        validate_replicate(
            replicate,
            &study.benchmark_id,
            &study.benchmark_version,
            &study.primary_metric_definition_sha256,
        )?;
        if replicate.evaluation_evidence.oracle_or_adjudication_sha256
            != first_evaluation.oracle_or_adjudication_sha256
            || replicate.evaluation_evidence.evaluation_harness_sha256
                != first_evaluation.evaluation_harness_sha256
            || replicate.evaluation_evidence.prompt_and_decoding_sha256
                != first_evaluation.prompt_and_decoding_sha256
        {
            return Err("evaluation oracle, harness and prompt/decoding conditions must match across replicates".to_owned());
        }
        if !replicate_ids.insert(replicate.design.replicate_id.as_str()) {
            return Err("replicate IDs must be unique".to_owned());
        }
        if !seed_commitments.insert(
            replicate
                .manifest
                .randomization_seed_commitment_sha256
                .as_str(),
        ) {
            return Err(
                "each model-pair replicate must use a unique randomization seed commitment"
                    .to_owned(),
            );
        }
        if !training_seed_commitments.insert(
            replicate
                .design
                .training_rng_seed_commitment_sha256
                .as_str(),
        ) {
            return Err("each independent model-pair replicate must use a unique training RNG seed commitment".to_owned());
        }
        for checkpoint in [
            replicate
                .evaluation_evidence
                .treatment_checkpoint_sha256
                .as_str(),
            replicate
                .evaluation_evidence
                .placebo_checkpoint_sha256
                .as_str(),
        ] {
            if !all_checkpoint_digests.insert(checkpoint) {
                return Err(
                    "output checkpoints must be unique across independent model-pair replicates"
                        .to_owned(),
                );
            }
        }
        if !all_training_audit_digests.insert(replicate.training_receipt_audit_sha256.as_str()) {
            return Err(
                "training receipt audit digests must be unique across replicates".to_owned(),
            );
        }
        for evaluation_digest in [
            replicate
                .evaluation_evidence
                .base_evaluation_receipt_sha256
                .as_str(),
            replicate
                .evaluation_evidence
                .treatment_evaluation_receipt_sha256
                .as_str(),
            replicate
                .evaluation_evidence
                .placebo_evaluation_receipt_sha256
                .as_str(),
        ] {
            if !all_evaluation_receipt_digests.insert(evaluation_digest) {
                return Err(
                    "evaluation receipt digests must be unique across model-pair replicates"
                        .to_owned(),
                );
            }
        }
        all_receipts_consistent &= replicate.training_receipt_audit.execution_consistent;
        all_ready_for_external_verification &= replicate
            .training_receipt_audit
            .ready_for_independent_attestation_verification;
        if !replicate.training_receipt_audit.independently_verified {
            issues.push(format!(
                "replicate {} training receipt is not independently authenticated",
                replicate.design.replicate_id
            ));
        }
        replicate_effects.push(replicate_effect(replicate)?);
    }
    if all_checkpoint_digests.len() != study.replicates.len() * 2 {
        return Err("each replicate must contribute two unique output checkpoints".to_owned());
    }
    if replicate_effects.len() != study.replicates.len() {
        return Err("replicate effect accounting mismatch".to_owned());
    }

    let effect_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.treatment_minus_placebo_cluster_weighted_permille)
        .collect();
    let item_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.treatment_minus_placebo_item_weighted_permille)
        .collect();
    let treatment_base_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.treatment_minus_base_cluster_weighted_permille)
        .collect();
    let placebo_base_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.placebo_minus_base_cluster_weighted_permille)
        .collect();
    let exposed_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.exposed_cluster_advantage_permille)
        .collect();
    let unexposed_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.unexposed_cluster_advantage_permille)
        .collect();
    let localization_values: Vec<f64> = replicate_effects
        .iter()
        .map(|effect| effect.exposure_localization_contrast_permille)
        .collect();

    let observed_localization = sample_mean(&localization_values)?;
    let study_bytes = serde_json::to_vec(study)
        .map_err(|error| format!("cannot serialize causal study for digest: {error}"))?;
    let study_sha256 = hash_domain(b"replicated-contamination-causal-study.v1", &study_bytes);
    let randomization_test_seed_commitment_sha256 =
        hash_domain(b"causal-randomization-test-seed.v1", &root_seed);
    let randomization_p = randomization_p_value(
        &study.replicates,
        observed_localization,
        &root_seed,
        study.randomization_test_draws,
    )?;

    let family_keys: BTreeSet<String> = replicate_effects[0]
        .families
        .iter()
        .map(|family| family.task_family.clone())
        .collect();
    let mut family_reports = Vec::new();
    for family in &family_keys {
        let mut base_values = Vec::new();
        let mut treatment_values = Vec::new();
        let mut placebo_values = Vec::new();
        let mut advantage_values = Vec::new();
        for effect in &replicate_effects {
            let found = effect
                .families
                .iter()
                .find(|entry| &entry.task_family == family)
                .ok_or_else(|| format!("family {family} missing from replicate"))?;
            base_values.push(found.base_score_permille);
            treatment_values.push(found.treatment_score_permille);
            placebo_values.push(found.placebo_score_permille);
            advantage_values.push(found.treatment_minus_placebo_permille);
        }
        let first_family = replicate_effects[0]
            .families
            .iter()
            .find(|entry| &entry.task_family == family)
            .ok_or_else(|| format!("family {family} missing from first replicate"))?;
        family_reports.push(FamilyEffectV1 {
            task_family: family.clone(),
            cluster_count: first_family.cluster_count,
            base_score_permille: sample_mean(&base_values)?,
            treatment_score_permille: sample_mean(&treatment_values)?,
            placebo_score_permille: sample_mean(&placebo_values)?,
            treatment_minus_placebo_permille: sample_mean(&advantage_values)?,
        });
    }

    let replicate_count_meets_minimum =
        study.replicates.len() >= study.minimum_confirmatory_replicates as usize;
    let execution_ready = all_receipts_consistent && all_ready_for_external_verification;
    // This module has no trust-root/signature verifier, so it cannot issue a
    // confirmatory eligibility decision, even when all digest receipts exist.
    let confirmatory_eligible = false;
    let analysis_status = if !replicate_count_meets_minimum {
        "exploratory_insufficient_independent_model_pair_replicates"
    } else if !execution_ready {
        "exploratory_execution_evidence_incomplete"
    } else {
        "exploratory_waiting_for_external_signature_verification"
    }
    .to_owned();
    if !replicate_count_meets_minimum {
        issues.push(format!(
            "only {} independent model-pair replicates; minimum preregistered count is {}",
            study.replicates.len(),
            study.minimum_confirmatory_replicates
        ));
    }

    Ok(ReplicatedCausalAnalysisReportV1 {
        schema: CAUSAL_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        primary_metric_id: study.primary_metric_id.clone(),
        study_sha256,
        randomization_test_seed_commitment_sha256,
        independent_model_pair_replicate_count: u64::try_from(study.replicates.len())
            .map_err(|_| "replicate count overflows u64")?,
        minimum_confirmatory_replicates: study.minimum_confirmatory_replicates,
        replicate_count_meets_minimum,
        every_replicate_has_consistent_training_receipt: all_receipts_consistent,
        training_receipts_independently_verified: independent_receipts_verified,
        execution_evidence_ready_for_external_verification: all_ready_for_external_verification,
        primary_cluster_weighted_effect_permille: mean_interval(&effect_values)?,
        secondary_item_weighted_effect_permille: mean_interval(&item_values)?,
        treatment_minus_base_effect_permille: mean_interval(&treatment_base_values)?,
        placebo_minus_base_effect_permille: mean_interval(&placebo_base_values)?,
        exposed_localization_effect_permille: mean_interval(&exposed_values)?,
        unexposed_spillover_effect_permille: mean_interval(&unexposed_values)?,
        exposure_localization_contrast_permille: mean_interval(&localization_values)?,
        localization_randomization_p_value: randomization_p,
        localization_randomization_draws: study.randomization_test_draws,
        randomization_null: "Fisher_sharp_null_no_assignment_effect_on_any_cluster_paired_advantage_including_spillover".to_owned(),
        family_effects: family_reports,
        replicate_effects,
        confirmatory_eligible,
        analysis_status,
        issues,
        claim_boundary: "model_pair_replicates_are_independent_training_units; clusters_and_questions_are_nested_measurements; localization_is_not_a_pure_direct_effect_under_global_interference; external_training_receipts_are_not_cryptographically_verified".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn digest(label: &str) -> String {
        hash_domain(b"causal-analysis-test", label.as_bytes())
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn make_replicate(index: u8) -> CausalReplicateInputV1 {
        let mut clusters = Vec::new();
        for family in ["math", "code", "science", "logic"] {
            for cluster_index in 0..4 {
                let cluster_id = format!("{family}-cluster-{cluster_index}");
                clusters.push(crate::intervention::SourceClusterV1 {
                    cluster_id: cluster_id.clone(),
                    task_family: family.to_owned(),
                    source_family_sha256: digest(&format!("source-{cluster_id}")),
                    item_ids: vec![
                        format!("{cluster_id}-item-1"),
                        format!("{cluster_id}-item-2"),
                    ],
                    exposure_variant_sha256: digest(&format!("variant-{cluster_id}")),
                    exposure_token_count: 500,
                    matched_placebo_record_sha256: digest(&format!("placebo-{cluster_id}")),
                    matched_placebo_token_count: 500,
                });
            }
        }
        let suffix = format!("replicate-{index:03}");
        let design = RandomizedInterventionDesignV1 {
            schema: crate::intervention::INTERVENTION_DESIGN_SCHEMA.to_owned(),
            experiment_id: "causal-study-test".to_owned(),
            replicate_id: suffix.clone(),
            benchmark_id: "causal-panel".to_owned(),
            benchmark_version: "v1".to_owned(),
            benchmark_task_set_sha256: digest("task-set"),
            model_base_checkpoint_sha256: digest("base-checkpoint"),
            training_stage: crate::intervention::TrainingStageV1::InstructionSft,
            exposure_mechanism: crate::intervention::ExposureMechanismV1::SemanticVariant,
            target_exposure_fraction_permille: 500,
            shared_training_corpus_sha256: digest("shared-corpus"),
            shared_training_token_count: 96_000,
            total_training_token_budget: 100_000,
            training_steps: 1_000,
            training_rng_seed_commitment_sha256: digest(&format!("training-rng-{index}")),
            training_code_sha256: digest("training-code"),
            runtime_image_sha256: digest("runtime"),
            tokenization_pipeline_sha256: digest("tokenizer"),
            training_hyperparameters_sha256: digest("hyperparameters"),
            optimizer_schedule_sha256: digest("optimizer"),
            treatment_training_corpus_sha256: digest(&format!("treatment-corpus-{index}")),
            placebo_training_corpus_sha256: digest(&format!("placebo-corpus-{index}")),
            clusters,
        };
        let seed = vec![index; 32];
        let manifest = randomize_intervention(&design, &seed).unwrap();
        let treatment_checkpoint = digest(&format!("treatment-checkpoint-{index}"));
        let placebo_checkpoint = digest(&format!("placebo-checkpoint-{index}"));
        let training_audit = PairedTrainingExecutionAuditV1 {
            schema: crate::training_receipt::TRAINING_RECEIPT_REPORT_SCHEMA.to_owned(),
            valid: true,
            execution_consistent: true,
            treatment_completed: true,
            placebo_completed: true,
            paired_checkpoints_distinct: true,
            base_checkpoint_matched: true,
            training_configuration_matched: true,
            corpus_identities_matched_to_design: true,
            intervention_payloads_matched_to_assignment: true,
            treatment_token_budget_conserved: true,
            placebo_token_budget_conserved: true,
            observed_token_budgets_complete: true,
            observed_step_budgets_complete: true,
            runner_attestation_digests_present: true,
            checkpoint_custody_digests_present: true,
            external_attestation_verification: "digest_present_signature_not_verified".to_owned(),
            independently_verified: false,
            ready_for_independent_attestation_verification: true,
            experiment_id: design.experiment_id.clone(),
            replicate_id: suffix.clone(),
            design_sha256: manifest.design_sha256.clone(),
            intervention_manifest_sha256: manifest.manifest_sha256.clone(),
            training_rng_seed_commitment_sha256: design.training_rng_seed_commitment_sha256.clone(),
            training_rng_seed_commitment_sha256: design.training_rng_seed_commitment_sha256.clone(),
            treatment_run_id: format!("treatment-run-{index}"),
            placebo_run_id: format!("placebo-run-{index}"),
            treatment_checkpoint_sha256: treatment_checkpoint.clone(),
            placebo_checkpoint_sha256: placebo_checkpoint.clone(),
            treatment_observed_tokens: design.total_training_token_budget,
            placebo_observed_tokens: design.total_training_token_budget,
            treatment_observed_steps: design.training_steps,
            placebo_observed_steps: design.training_steps,
            claim_boundary: "synthetic-consistency-fixture".to_owned(),
            issues: Vec::new(),
        };
        let audit_bytes = serde_json::to_vec(&training_audit).unwrap();
        let training_receipt_audit_sha256 =
            hash_domain(b"paired-training-execution-audit.v1", &audit_bytes);
        let evidence = EvaluationEvidenceV1 {
            base_checkpoint_sha256: design.model_base_checkpoint_sha256.clone(),
            treatment_checkpoint_sha256: treatment_checkpoint,
            placebo_checkpoint_sha256: placebo_checkpoint,
            base_evaluation_receipt_sha256: digest(&format!("base-eval-{index}")),
            treatment_evaluation_receipt_sha256: digest(&format!("treatment-eval-{index}")),
            placebo_evaluation_receipt_sha256: digest(&format!("placebo-eval-{index}")),
            metric_definition_sha256: digest("metric-definition"),
            oracle_or_adjudication_sha256: digest("oracle"),
            evaluation_harness_sha256: digest("eval-harness"),
            prompt_and_decoding_sha256: digest("prompt-decoding"),
        };
        let assignment_map: BTreeMap<String, bool> = manifest
            .assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.cluster_id.clone(),
                    assignment.assigned_to_exposure,
                )
            })
            .collect();
        let mut outcomes = Vec::new();
        for cluster in &design.clusters {
            let mut sorted_ids = cluster.item_ids.clone();
            sorted_ids.sort();
            let item_digest = hash_domain(
                b"causal-evaluation-item-set.v1",
                &serde_json::to_vec(&sorted_ids).unwrap(),
            );
            let exposed = assignment_map[&cluster.cluster_id];
            outcomes.push(ClusterOutcomeV1 {
                cluster_id: cluster.cluster_id.clone(),
                task_family: cluster.task_family.clone(),
                assigned_to_exposure: exposed,
                item_count: 2,
                evaluation_item_set_sha256: item_digest,
                base: ScoreCountV1 {
                    success_count: 1,
                    observed_count: 2,
                },
                treatment: ScoreCountV1 {
                    success_count: if exposed { 2 } else { 1 },
                    observed_count: 2,
                },
                placebo: ScoreCountV1 {
                    success_count: 1,
                    observed_count: 2,
                },
            });
        }
        CausalReplicateInputV1 {
            design,
            manifest,
            randomization_seed_hex: hex(&seed),
            training_receipt_audit: training_audit,
            training_receipt_audit_sha256,
            evaluation_evidence: evidence,
            cluster_outcomes: outcomes,
        }
    }

    fn study_fixture() -> ReplicatedCausalStudyV1 {
        ReplicatedCausalStudyV1 {
            schema: CAUSAL_STUDY_SCHEMA.to_owned(),
            study_id: "causal-study-fixture".to_owned(),
            benchmark_id: "causal-panel".to_owned(),
            benchmark_version: "v1".to_owned(),
            primary_metric_id: "exact-correctness".to_owned(),
            primary_metric_definition_sha256: digest("metric-definition"),
            minimum_confirmatory_replicates: CAUSAL_MIN_CONFIRMATORY_REPLICATES as u32,
            randomization_test_draws: CAUSAL_MIN_RANDOMIZATION_DRAWS,
            randomization_test_seed_hex: hex(&vec![0xA7; 32]),
            replicates: vec![make_replicate(1), make_replicate(2)],
        }
    }

    #[test]
    fn causal_study_and_report_schemas_match_rust_struct_fields() {
        let study_schema: Value = serde_json::from_str(include_str!(
            "../schema/replicated-causal-study.schema.json"
        ))
        .unwrap();
        let report_schema: Value = serde_json::from_str(include_str!(
            "../schema/replicated-causal-report.schema.json"
        ))
        .unwrap();
        let study = study_fixture();
        let report = analyze_replicated_causal_study(&study).unwrap();
        let study_value = serde_json::to_value(&study).unwrap();
        let report_value = serde_json::to_value(&report).unwrap();

        assert_eq!(
            study_schema["properties"]["schema"]["const"].as_str(),
            Some(CAUSAL_STUDY_SCHEMA)
        );
        assert_eq!(
            report_schema["properties"]["schema"]["const"].as_str(),
            Some(CAUSAL_REPORT_SCHEMA)
        );
        let study_required: BTreeSet<String> = study_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let study_properties: BTreeSet<String> = study_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(study_required, study_properties);
        assert_eq!(
            study_value
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            study_properties
        );
        let report_required: BTreeSet<String> = report_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let report_properties: BTreeSet<String> = report_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(report_required, report_properties);
        assert_eq!(
            report_value
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            report_properties
        );
    }

    #[test]
    fn score_rates_reject_invalid_denominators_and_success_counts() {
        assert!(
            rate_permille(
                &ScoreCountV1 {
                    success_count: 2,
                    observed_count: 0
                },
                "x"
            )
            .is_err()
        );
        assert!(
            rate_permille(
                &ScoreCountV1 {
                    success_count: 2,
                    observed_count: 1
                },
                "x"
            )
            .is_err()
        );
        assert_eq!(
            rate_permille(
                &ScoreCountV1 {
                    success_count: 1,
                    observed_count: 2
                },
                "x"
            )
            .unwrap(),
            500.0
        );
    }

    #[test]
    fn small_sample_t_critical_values_are_conservative() {
        assert_eq!(t_critical_975(1), 12.706);
        assert_eq!(t_critical_975(9), 2.262);
        assert_eq!(t_critical_975(29), 2.045);
        assert_eq!(t_critical_975(30), 2.042);
        assert_eq!(t_critical_975(31), 2.042);
        assert_eq!(t_critical_975(50), 2.021);
        assert_eq!(t_critical_975(200), 1.98);
    }

    #[test]
    fn end_to_end_analysis_uses_model_pairs_and_stays_exploratory_without_trust_roots() {
        let study = study_fixture();
        let report = analyze_replicated_causal_study(&study).unwrap();
        assert_eq!(report.independent_model_pair_replicate_count, 2);
        let study_bytes = serde_json::to_vec(&study).unwrap();
        assert_eq!(
            report.study_sha256,
            hash_domain(b"replicated-contamination-causal-study.v1", &study_bytes)
        );
        assert_eq!(
            report.randomization_test_seed_commitment_sha256,
            hash_domain(
                b"causal-randomization-test-seed.v1",
                &decode_seed_hex(&study.randomization_test_seed_hex, "seed").unwrap()
            )
        );
        assert!(!report.replicate_count_meets_minimum);
        assert!(report.every_replicate_has_consistent_training_receipt);
        assert!(!report.training_receipts_independently_verified);
        assert!(report.execution_evidence_ready_for_external_verification);
        assert!(!report.confirmatory_eligible);
        assert_eq!(
            report.analysis_status,
            "exploratory_insufficient_independent_model_pair_replicates"
        );
        assert!(report.primary_cluster_weighted_effect_permille.mean > 0.0);
        assert!(
            report.primary_cluster_weighted_effect_permille.lower_95
                <= report.primary_cluster_weighted_effect_permille.upper_95
        );
        assert!(
            report.localization_randomization_p_value > 0.0
                && report.localization_randomization_p_value <= 1.0
        );
        assert_eq!(report.family_effects.len(), 4);
        assert!(report.claim_boundary.contains("global_interference"));
    }

    #[test]
    fn one_model_pair_is_not_misrepresented_as_many_training_replicates() {
        let mut study = study_fixture();
        study.replicates.truncate(1);
        assert!(
            analyze_replicated_causal_study(&study)
                .unwrap_err()
                .contains("at least two independent model-pair replicates")
        );
    }

    #[test]
    fn a_self_asserted_independent_verification_flag_is_rejected() {
        let mut study = study_fixture();
        study.replicates[0]
            .training_receipt_audit
            .independently_verified = true;
        let audit_bytes = serde_json::to_vec(&study.replicates[0].training_receipt_audit).unwrap();
        study.replicates[0].training_receipt_audit_sha256 =
            hash_domain(b"paired-training-execution-audit.v1", &audit_bytes);
        assert!(
            analyze_replicated_causal_study(&study)
                .unwrap_err()
                .contains("self-asserted independent verification")
        );
    }

    #[test]
    fn mean_interval_uses_independent_replicate_sd_not_cluster_count() {
        let interval = mean_interval(&[1.0, 3.0, 5.0]).unwrap();
        assert!((interval.mean - 3.0).abs() < 1e-12);
        assert!((interval.standard_deviation_across_replicates - 2.0).abs() < 1e-12);
        assert_eq!(interval.confidence_level_permille, 950);
    }

    #[test]
    fn randomization_seed_derivation_is_reproducible_and_replicate_scoped() {
        let root = vec![0x42; 32];
        let a = derive_seed(&root, "replicate-001", 7);
        let b = derive_seed(&root, "replicate-001", 7);
        let c = derive_seed(&root, "replicate-002", 7);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn seed_hex_decoder_rejects_short_and_uppercase_values() {
        assert!(decode_seed_hex("aabb", "seed").is_err());
        assert!(decode_seed_hex(&"AA".repeat(32), "seed").is_err());
        assert_eq!(decode_seed_hex(&"ab".repeat(32), "seed").unwrap().len(), 32);
    }
}
