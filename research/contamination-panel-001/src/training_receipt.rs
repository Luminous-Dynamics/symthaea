//! Paired treatment/placebo training execution receipts.
//!
//! This validator joins actual training-run claims to a preregistered randomized
//! intervention design and sealed assignment manifest. It checks model lineage,
//! corpus identities, token budgets, training configuration, checkpoints, payload
//! digests and external receipt references.
//!
//! Digest presence is not signature verification. This module deliberately
//! reports that runner attestation and checkpoint-custody receipts are present
//! but not cryptographically authenticated by this validator. It cannot prove
//! that a trainer consumed the declared corpus or that an external issuer is
//! trustworthy.

use crate::{
    hash_domain,
    intervention::{
        RandomizedInterventionDesignV1, RandomizedInterventionManifestV1, audit_intervention,
    },
};
use serde::{Deserialize, Serialize};

pub const TRAINING_RECEIPT_SCHEMA: &str = "symthaea.paired_training_execution_receipt.v1";
pub const TRAINING_RECEIPT_REPORT_SCHEMA: &str = "symthaea.paired_training_execution_audit.v1";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingArmV1 {
    Treatment,
    Placebo,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingRunStatusV1 {
    Completed,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingArmExecutionReceiptV1 {
    pub arm: TrainingArmV1,
    pub status: TrainingRunStatusV1,
    pub run_id: String,
    pub base_checkpoint_sha256: String,
    pub parent_checkpoint_sha256: String,
    pub output_checkpoint_sha256: String,
    pub training_corpus_sha256: String,
    pub shared_training_corpus_sha256: String,
    pub intervention_manifest_sha256: String,
    pub intervention_assignment_sha256: String,
    pub intervention_payload_sha256: String,
    pub shared_training_token_count: u64,
    pub intervention_token_count: u64,
    pub observed_shared_training_token_count: u64,
    pub observed_intervention_token_count: u64,
    pub observed_total_training_token_count: u64,
    pub requested_total_training_token_budget: u64,
    pub requested_training_steps: u64,
    pub observed_training_steps: u64,
    pub training_rng_seed_commitment_sha256: String,
    pub training_code_sha256: String,
    pub runtime_image_sha256: String,
    pub tokenization_pipeline_sha256: String,
    pub training_hyperparameters_sha256: String,
    pub optimizer_schedule_sha256: String,
    pub corpus_shard_manifest_sha256: String,
    pub post_tokenization_manifest_sha256: String,
    pub trainer_job_execution_receipt_sha256: String,
    pub checkpoint_custody_receipt_sha256: Option<String>,
    pub runner_attestation_sha256: Option<String>,
    pub started_at_utc: String,
    pub completed_at_utc: String,
    pub wall_clock_millis: u64,
    pub cpu_time_millis: u64,
    pub accelerator_time_millis: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedTrainingExecutionReceiptV1 {
    pub schema: String,
    pub experiment_id: String,
    pub replicate_id: String,
    pub design_sha256: String,
    pub intervention_manifest_sha256: String,
    pub treatment: TrainingArmExecutionReceiptV1,
    pub placebo: TrainingArmExecutionReceiptV1,
    pub pair_receipt_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedTrainingExecutionAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub execution_consistent: bool,
    pub treatment_completed: bool,
    pub placebo_completed: bool,
    pub paired_checkpoints_distinct: bool,
    pub base_checkpoint_matched: bool,
    pub training_configuration_matched: bool,
    pub corpus_identities_matched_to_design: bool,
    pub intervention_payloads_matched_to_assignment: bool,
    pub treatment_token_budget_conserved: bool,
    pub placebo_token_budget_conserved: bool,
    pub observed_token_budgets_complete: bool,
    pub observed_step_budgets_complete: bool,
    pub runner_attestation_digests_present: bool,
    pub checkpoint_custody_digests_present: bool,
    pub external_attestation_verification: String,
    pub independently_verified: bool,
    pub ready_for_independent_attestation_verification: bool,
    pub experiment_id: String,
    pub replicate_id: String,
    pub design_sha256: String,
    pub intervention_manifest_sha256: String,
    pub treatment_run_id: String,
    pub placebo_run_id: String,
    pub treatment_checkpoint_sha256: String,
    pub placebo_checkpoint_sha256: String,
    pub training_rng_seed_commitment_sha256: String,
    pub treatment_observed_tokens: u64,
    pub placebo_observed_tokens: u64,
    pub treatment_observed_steps: u64,
    pub placebo_observed_steps: u64,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct InterventionPayloadEntryV1<'a> {
    cluster_id: &'a str,
    record_sha256: &'a str,
    token_count: u64,
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

fn validate_utc_timestamp(value: &str, label: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(format!("{label} must use YYYY-MM-DDTHH:MM:SSZ"));
    }
    for (index, byte) in bytes.iter().enumerate() {
        if ![4, 7, 10, 13, 16, 19].contains(&index) && !byte.is_ascii_digit() {
            return Err(format!("{label} must use YYYY-MM-DDTHH:MM:SSZ"));
        }
    }
    let parse =
        |start: usize, end: usize| -> u32 { value[start..end].parse::<u32>().unwrap_or(u32::MAX) };
    let year = parse(0, 4);
    let month = parse(5, 7);
    let day = parse(8, 10);
    let hour = parse(11, 13);
    let minute = parse(14, 16);
    let second = parse(17, 19);
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days_in_month || hour > 23 || minute > 59 || second > 59 {
        return Err(format!(
            "{label} contains an impossible date or clock value"
        ));
    }
    Ok(())
}

fn expected_design_sha256(design: &RandomizedInterventionDesignV1) -> Result<String, String> {
    // This must remain byte-for-byte consistent with randomize_intervention.
    let bytes = serde_json::to_vec(design)
        .map_err(|error| format!("cannot serialize intervention design: {error}"))?;
    Ok(hash_domain(b"randomized-intervention-design.v1", &bytes))
}

fn intervention_payload_digest(
    manifest: &RandomizedInterventionManifestV1,
    arm: TrainingArmV1,
) -> Result<String, String> {
    let entries: Vec<InterventionPayloadEntryV1<'_>> = manifest
        .assignments
        .iter()
        .filter(|assignment| assignment.assigned_to_exposure)
        .map(|assignment| match arm {
            TrainingArmV1::Treatment => InterventionPayloadEntryV1 {
                cluster_id: &assignment.cluster_id,
                record_sha256: &assignment.exposure_variant_sha256,
                token_count: assignment.exposure_token_count,
            },
            TrainingArmV1::Placebo => InterventionPayloadEntryV1 {
                cluster_id: &assignment.cluster_id,
                record_sha256: &assignment.matched_placebo_record_sha256,
                token_count: assignment.matched_placebo_token_count,
            },
        })
        .collect();
    let bytes = serde_json::to_vec(&entries)
        .map_err(|error| format!("cannot serialize {arm:?} payload entries: {error}"))?;
    let domain: &[u8] = match arm {
        TrainingArmV1::Treatment => b"treatment-intervention-payload.v1",
        TrainingArmV1::Placebo => b"placebo-intervention-payload.v1",
    };
    Ok(hash_domain(domain, &bytes))
}

fn validate_arm_common(
    arm: &TrainingArmExecutionReceiptV1,
    expected_arm: TrainingArmV1,
    design: &RandomizedInterventionDesignV1,
    manifest: &RandomizedInterventionManifestV1,
    expected_payload_sha256: &str,
    expected_intervention_tokens: u64,
    issues: &mut Vec<String>,
) -> Result<(), String> {
    if arm.arm != expected_arm {
        return Err(format!("expected {:?} arm receipt", expected_arm));
    }
    if arm.run_id.trim().is_empty() || arm.run_id.len() > 256 {
        return Err(format!(
            "{expected_arm:?} run_id must be nonempty and <=256 bytes"
        ));
    }

    for (value, label) in [
        (&arm.base_checkpoint_sha256, "base_checkpoint_sha256"),
        (&arm.parent_checkpoint_sha256, "parent_checkpoint_sha256"),
        (&arm.output_checkpoint_sha256, "output_checkpoint_sha256"),
        (&arm.training_corpus_sha256, "training_corpus_sha256"),
        (
            &arm.shared_training_corpus_sha256,
            "shared_training_corpus_sha256",
        ),
        (
            &arm.intervention_manifest_sha256,
            "intervention_manifest_sha256",
        ),
        (
            &arm.intervention_assignment_sha256,
            "intervention_assignment_sha256",
        ),
        (
            &arm.intervention_payload_sha256,
            "intervention_payload_sha256",
        ),
        (
            &arm.training_rng_seed_commitment_sha256,
            "training_rng_seed_commitment_sha256",
        ),
        (&arm.training_code_sha256, "training_code_sha256"),
        (&arm.runtime_image_sha256, "runtime_image_sha256"),
        (
            &arm.tokenization_pipeline_sha256,
            "tokenization_pipeline_sha256",
        ),
        (
            &arm.training_hyperparameters_sha256,
            "training_hyperparameters_sha256",
        ),
        (&arm.optimizer_schedule_sha256, "optimizer_schedule_sha256"),
        (
            &arm.corpus_shard_manifest_sha256,
            "corpus_shard_manifest_sha256",
        ),
        (
            &arm.post_tokenization_manifest_sha256,
            "post_tokenization_manifest_sha256",
        ),
        (
            &arm.trainer_job_execution_receipt_sha256,
            "trainer_job_execution_receipt_sha256",
        ),
    ] {
        validate_digest(value, label)?;
    }
    if let Some(value) = &arm.checkpoint_custody_receipt_sha256 {
        validate_digest(value, "checkpoint_custody_receipt_sha256")?;
    }
    if let Some(value) = &arm.runner_attestation_sha256 {
        validate_digest(value, "runner_attestation_sha256")?;
    }
    validate_utc_timestamp(&arm.started_at_utc, "started_at_utc")?;
    validate_utc_timestamp(&arm.completed_at_utc, "completed_at_utc")?;
    if arm.completed_at_utc < arm.started_at_utc {
        return Err(format!(
            "{expected_arm:?} completed_at_utc precedes started_at_utc"
        ));
    }
    if arm.wall_clock_millis == 0 {
        return Err(format!(
            "{expected_arm:?} wall_clock_millis must be positive"
        ));
    }

    if arm.base_checkpoint_sha256 != design.model_base_checkpoint_sha256
        || arm.parent_checkpoint_sha256 != design.model_base_checkpoint_sha256
    {
        return Err(format!(
            "{expected_arm:?} base/parent checkpoint does not match preregistered base"
        ));
    }
    let expected_corpus = match expected_arm {
        TrainingArmV1::Treatment => &design.treatment_training_corpus_sha256,
        TrainingArmV1::Placebo => &design.placebo_training_corpus_sha256,
    };
    if &arm.training_corpus_sha256 != expected_corpus {
        let arm_name = match expected_arm {
            TrainingArmV1::Treatment => "treatment",
            TrainingArmV1::Placebo => "placebo",
        };
        return Err(format!(
            "{arm_name} training corpus digest does not match design"
        ));
    }
    if arm.shared_training_corpus_sha256 != design.shared_training_corpus_sha256
        || arm.shared_training_token_count != design.shared_training_token_count
    {
        return Err(format!(
            "{expected_arm:?} shared corpus identity/budget does not match design"
        ));
    }
    if arm.intervention_manifest_sha256 != manifest.manifest_sha256
        || arm.intervention_assignment_sha256 != manifest.assignment_sha256
    {
        return Err(format!(
            "{expected_arm:?} receipt does not bind the exact intervention assignment"
        ));
    }
    if arm.intervention_payload_sha256 != expected_payload_sha256 {
        return Err(format!(
            "{expected_arm:?} payload digest does not match assigned records"
        ));
    }
    if arm.intervention_token_count != expected_intervention_tokens {
        return Err(format!(
            "{expected_arm:?} intervention token count does not match assignment"
        ));
    }
    if arm.requested_total_training_token_budget != design.total_training_token_budget
        || arm.requested_training_steps != design.training_steps
    {
        return Err(format!(
            "{expected_arm:?} requested training budget/steps do not match design"
        ));
    }
    if arm.training_rng_seed_commitment_sha256 != design.training_rng_seed_commitment_sha256
        || arm.training_code_sha256 != design.training_code_sha256
        || arm.runtime_image_sha256 != design.runtime_image_sha256
        || arm.tokenization_pipeline_sha256 != design.tokenization_pipeline_sha256
        || arm.training_hyperparameters_sha256 != design.training_hyperparameters_sha256
        || arm.optimizer_schedule_sha256 != design.optimizer_schedule_sha256
    {
        return Err(format!(
            "{expected_arm:?} trainer code/runtime/tokenizer/configuration does not match design"
        ));
    }

    let observed_total = arm
        .observed_shared_training_token_count
        .checked_add(arm.observed_intervention_token_count)
        .ok_or_else(|| format!("{expected_arm:?} observed token total overflows u64"))?;
    if arm.observed_total_training_token_count != observed_total {
        return Err(format!(
            "{expected_arm:?} observed token count does not equal component sum"
        ));
    }
    if arm.observed_shared_training_token_count > design.shared_training_token_count
        || arm.observed_intervention_token_count > expected_intervention_tokens
        || arm.observed_training_steps > design.training_steps
    {
        return Err(format!(
            "{expected_arm:?} observed tokens/steps exceed declared budgets"
        ));
    }
    match arm.status {
        TrainingRunStatusV1::Completed => {
            if arm.observed_shared_training_token_count != design.shared_training_token_count
                || arm.observed_intervention_token_count != expected_intervention_tokens
                || arm.observed_total_training_token_count != design.total_training_token_budget
                || arm.observed_training_steps != design.training_steps
            {
                return Err(format!(
                    "{expected_arm:?} completed run did not consume the full declared token/step budget"
                ));
            }
        }
        TrainingRunStatusV1::Partial | TrainingRunStatusV1::Failed => {
            issues.push(format!("{expected_arm:?} run status is {:?}", arm.status));
        }
    }
    if arm.output_checkpoint_sha256 == arm.base_checkpoint_sha256 {
        return Err(format!(
            "{expected_arm:?} output checkpoint must differ from base checkpoint"
        ));
    }
    Ok(())
}

/// Validate a paired treatment/placebo execution receipt against the exact
/// randomized assignment and preregistered design.
///
/// This validator checks the receipt's internal and cross-arm consistency.
/// It does not verify digital signatures or authenticate the trainer, runner,
/// checkpoint store or external receipt issuers.
pub fn validate_paired_training_execution_receipt(
    design: &RandomizedInterventionDesignV1,
    manifest: &RandomizedInterventionManifestV1,
    seed: &[u8],
    receipt_bytes: &[u8],
) -> Result<PairedTrainingExecutionAuditV1, String> {
    audit_intervention(design, manifest, seed)?;
    let receipt: PairedTrainingExecutionReceiptV1 =
        serde_json::from_slice(receipt_bytes).map_err(|error| {
            format!("paired training receipt does not match strict v1 JSON: {error}")
        })?;
    if receipt.schema != TRAINING_RECEIPT_SCHEMA {
        return Err(format!("receipt schema must be {TRAINING_RECEIPT_SCHEMA}"));
    }
    if receipt.experiment_id != design.experiment_id
        || receipt.replicate_id != design.replicate_id
        || receipt.experiment_id != manifest.experiment_id
        || receipt.replicate_id != manifest.replicate_id
    {
        return Err(
            "receipt experiment/replicate identity does not match design and manifest".to_owned(),
        );
    }

    let expected_design_sha256 = expected_design_sha256(design)?;
    if receipt.design_sha256 != expected_design_sha256 {
        return Err("receipt design digest does not match exact preregistered design".to_owned());
    }
    if receipt.intervention_manifest_sha256 != manifest.manifest_sha256 {
        return Err("receipt does not bind exact intervention manifest".to_owned());
    }
    let expected_treatment_payload =
        intervention_payload_digest(manifest, TrainingArmV1::Treatment)?;
    let expected_placebo_payload = intervention_payload_digest(manifest, TrainingArmV1::Placebo)?;

    let mut issues = Vec::new();
    validate_arm_common(
        &receipt.treatment,
        TrainingArmV1::Treatment,
        design,
        manifest,
        &expected_treatment_payload,
        manifest.exposed_training_token_count,
        &mut issues,
    )?;
    validate_arm_common(
        &receipt.placebo,
        TrainingArmV1::Placebo,
        design,
        manifest,
        &expected_placebo_payload,
        manifest.matched_placebo_token_count,
        &mut issues,
    )?;

    if receipt.treatment.run_id == receipt.placebo.run_id {
        return Err("treatment and placebo run IDs must differ".to_owned());
    }
    if receipt.treatment.output_checkpoint_sha256 == receipt.placebo.output_checkpoint_sha256 {
        return Err("treatment and placebo output checkpoint digests must differ".to_owned());
    }

    let receipt_without_digest = {
        let mut clone = receipt.clone();
        clone.pair_receipt_sha256.clear();
        serde_json::to_vec(&clone)
            .map_err(|error| format!("cannot serialize receipt for digest: {error}"))?
    };
    let expected_receipt_digest = hash_domain(
        b"paired-training-execution-receipt.v1",
        &receipt_without_digest,
    );
    if receipt.pair_receipt_sha256 != expected_receipt_digest {
        return Err("paired training receipt digest does not match contents".to_owned());
    }

    let treatment_completed = receipt.treatment.status == TrainingRunStatusV1::Completed;
    let placebo_completed = receipt.placebo.status == TrainingRunStatusV1::Completed;
    let training_configuration_matched = receipt.treatment.training_code_sha256
        == receipt.placebo.training_code_sha256
        && receipt.treatment.runtime_image_sha256 == receipt.placebo.runtime_image_sha256
        && receipt.treatment.tokenization_pipeline_sha256
            == receipt.placebo.tokenization_pipeline_sha256
        && receipt.treatment.training_hyperparameters_sha256
            == receipt.placebo.training_hyperparameters_sha256
        && receipt.treatment.optimizer_schedule_sha256 == receipt.placebo.optimizer_schedule_sha256
        && receipt.treatment.training_rng_seed_commitment_sha256
            == receipt.placebo.training_rng_seed_commitment_sha256
        && receipt.treatment.requested_training_steps == receipt.placebo.requested_training_steps
        && receipt.treatment.requested_total_training_token_budget
            == receipt.placebo.requested_total_training_token_budget;
    let base_checkpoint_matched = receipt.treatment.base_checkpoint_sha256
        == receipt.placebo.base_checkpoint_sha256
        && receipt.treatment.parent_checkpoint_sha256 == receipt.placebo.parent_checkpoint_sha256;
    let corpus_identities_matched_to_design = receipt.treatment.training_corpus_sha256
        == design.treatment_training_corpus_sha256
        && receipt.placebo.training_corpus_sha256 == design.placebo_training_corpus_sha256;
    let treatment_token_budget_conserved = receipt.treatment.status
        == TrainingRunStatusV1::Completed
        && receipt.treatment.observed_total_training_token_count
            == design.total_training_token_budget;
    let placebo_token_budget_conserved = receipt.placebo.status == TrainingRunStatusV1::Completed
        && receipt.placebo.observed_total_training_token_count
            == design.total_training_token_budget;
    let observed_token_budgets_complete =
        treatment_token_budget_conserved && placebo_token_budget_conserved;
    let observed_step_budgets_complete = treatment_completed
        && placebo_completed
        && receipt.treatment.observed_training_steps == design.training_steps
        && receipt.placebo.observed_training_steps == design.training_steps;
    let runner_attestation_digests_present = receipt
        .treatment
        .runner_attestation_sha256
        .as_deref()
        .is_some_and(valid_digest)
        && receipt
            .placebo
            .runner_attestation_sha256
            .as_deref()
            .is_some_and(valid_digest);
    let checkpoint_custody_digests_present = receipt
        .treatment
        .checkpoint_custody_receipt_sha256
        .as_deref()
        .is_some_and(valid_digest)
        && receipt
            .placebo
            .checkpoint_custody_receipt_sha256
            .as_deref()
            .is_some_and(valid_digest);
    let paired_checkpoints_distinct =
        receipt.treatment.output_checkpoint_sha256 != receipt.placebo.output_checkpoint_sha256;

    let execution_consistent = training_configuration_matched
        && base_checkpoint_matched
        && corpus_identities_matched_to_design
        && observed_token_budgets_complete
        && observed_step_budgets_complete
        && paired_checkpoints_distinct;
    let ready_for_independent_attestation_verification = execution_consistent
        && runner_attestation_digests_present
        && checkpoint_custody_digests_present;

    Ok(PairedTrainingExecutionAuditV1 {
        schema: TRAINING_RECEIPT_REPORT_SCHEMA.to_owned(),
        valid: true,
        execution_consistent,
        treatment_completed,
        placebo_completed,
        paired_checkpoints_distinct,
        base_checkpoint_matched,
        training_configuration_matched,
        corpus_identities_matched_to_design,
        intervention_payloads_matched_to_assignment: true,
        treatment_token_budget_conserved,
        placebo_token_budget_conserved,
        observed_token_budgets_complete,
        observed_step_budgets_complete,
        runner_attestation_digests_present,
        checkpoint_custody_digests_present,
        external_attestation_verification: "digest_presence_checked_signature_not_cryptographically_verified".to_owned(),
        independently_verified: false,
        ready_for_independent_attestation_verification,
        experiment_id: receipt.experiment_id.clone(),
        replicate_id: receipt.replicate_id.clone(),
        design_sha256: receipt.design_sha256.clone(),
        intervention_manifest_sha256: receipt.intervention_manifest_sha256.clone(),
        treatment_run_id: receipt.treatment.run_id.clone(),
        placebo_run_id: receipt.placebo.run_id.clone(),
        treatment_checkpoint_sha256: receipt.treatment.output_checkpoint_sha256.clone(),
        placebo_checkpoint_sha256: receipt.placebo.output_checkpoint_sha256.clone(),
        training_rng_seed_commitment_sha256: receipt.treatment.training_rng_seed_commitment_sha256.clone(),
        treatment_observed_tokens: receipt.treatment.observed_total_training_token_count,
        placebo_observed_tokens: receipt.placebo.observed_total_training_token_count,
        treatment_observed_steps: receipt.treatment.observed_training_steps,
        placebo_observed_steps: receipt.placebo.observed_training_steps,
        claim_boundary: "receipt_structure_and_claim_consistency_checked; signatures_runner_identity_trainer_execution_and_checkpoint_custody_not_cryptographically_verified".to_owned(),
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intervention::{
        ExposureMechanismV1, RandomizedInterventionDesignV1, SourceClusterV1, TrainingStageV1,
        randomize_intervention,
    };
    use serde_json::Value;
    use std::collections::BTreeSet;

    fn digest(label: &str) -> String {
        hash_domain(b"training-receipt-test", label.as_bytes())
    }

    fn design() -> RandomizedInterventionDesignV1 {
        let mut clusters = Vec::new();
        for family in ["math", "code", "science", "logic"] {
            for index in 0..8 {
                let cluster_id = format!("{family}-cluster-{index}");
                clusters.push(SourceClusterV1 {
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
        RandomizedInterventionDesignV1 {
            schema: crate::intervention::INTERVENTION_DESIGN_SCHEMA.to_owned(),
            experiment_id: "training-receipt-fixture".to_owned(),
            replicate_id: "replicate-001".to_owned(),
            benchmark_id: "contamination-panel".to_owned(),
            benchmark_version: "v1".to_owned(),
            benchmark_task_set_sha256: digest("task-set"),
            model_base_checkpoint_sha256: digest("base-checkpoint"),
            training_stage: TrainingStageV1::InstructionSft,
            exposure_mechanism: ExposureMechanismV1::SemanticVariant,
            target_exposure_fraction_permille: 500,
            shared_training_corpus_sha256: digest("shared-corpus"),
            shared_training_token_count: 1_992_000,
            total_training_token_budget: 2_000_000,
            training_steps: 5_000,
            training_rng_seed_commitment_sha256: digest("training-rng-seed"),
            training_code_sha256: digest("training-code"),
            runtime_image_sha256: digest("runtime"),
            tokenization_pipeline_sha256: digest("tokenizer"),
            training_hyperparameters_sha256: digest("hyperparameters"),
            optimizer_schedule_sha256: digest("optimizer"),
            treatment_training_corpus_sha256: digest("treatment-corpus"),
            placebo_training_corpus_sha256: digest("placebo-corpus"),
            clusters,
        }
    }

    fn seed() -> Vec<u8> {
        vec![0x5A; 32]
    }

    fn arm_receipt(
        design: &RandomizedInterventionDesignV1,
        manifest: &RandomizedInterventionManifestV1,
        arm: TrainingArmV1,
    ) -> TrainingArmExecutionReceiptV1 {
        let payload_sha = intervention_payload_digest(manifest, arm).unwrap();
        let intervention_tokens = match arm {
            TrainingArmV1::Treatment => manifest.exposed_training_token_count,
            TrainingArmV1::Placebo => manifest.matched_placebo_token_count,
        };
        let corpus = match arm {
            TrainingArmV1::Treatment => design.treatment_training_corpus_sha256.clone(),
            TrainingArmV1::Placebo => design.placebo_training_corpus_sha256.clone(),
        };
        let suffix = match arm {
            TrainingArmV1::Treatment => "treatment",
            TrainingArmV1::Placebo => "placebo",
        };
        TrainingArmExecutionReceiptV1 {
            arm,
            status: TrainingRunStatusV1::Completed,
            run_id: format!("run-{suffix}"),
            base_checkpoint_sha256: design.model_base_checkpoint_sha256.clone(),
            parent_checkpoint_sha256: design.model_base_checkpoint_sha256.clone(),
            output_checkpoint_sha256: digest(&format!("checkpoint-{suffix}")),
            training_corpus_sha256: corpus,
            shared_training_corpus_sha256: design.shared_training_corpus_sha256.clone(),
            intervention_manifest_sha256: manifest.manifest_sha256.clone(),
            intervention_assignment_sha256: manifest.assignment_sha256.clone(),
            intervention_payload_sha256: payload_sha,
            shared_training_token_count: design.shared_training_token_count,
            intervention_token_count: intervention_tokens,
            observed_shared_training_token_count: design.shared_training_token_count,
            observed_intervention_token_count: intervention_tokens,
            observed_total_training_token_count: design.total_training_token_budget,
            requested_total_training_token_budget: design.total_training_token_budget,
            requested_training_steps: design.training_steps,
            observed_training_steps: design.training_steps,
            training_rng_seed_commitment_sha256: design.training_rng_seed_commitment_sha256.clone(),
            training_code_sha256: design.training_code_sha256.clone(),
            runtime_image_sha256: design.runtime_image_sha256.clone(),
            tokenization_pipeline_sha256: design.tokenization_pipeline_sha256.clone(),
            training_hyperparameters_sha256: design.training_hyperparameters_sha256.clone(),
            optimizer_schedule_sha256: design.optimizer_schedule_sha256.clone(),
            corpus_shard_manifest_sha256: digest(&format!("shards-{suffix}")),
            post_tokenization_manifest_sha256: digest(&format!("tokens-{suffix}")),
            trainer_job_execution_receipt_sha256: digest(&format!("job-{suffix}")),
            checkpoint_custody_receipt_sha256: Some(digest(&format!("custody-{suffix}"))),
            runner_attestation_sha256: Some(digest(&format!("attestation-{suffix}"))),
            started_at_utc: "2026-10-31T00:00:00Z".to_owned(),
            completed_at_utc: "2026-10-31T01:00:00Z".to_owned(),
            wall_clock_millis: 3_600_000,
            cpu_time_millis: 3_000_000,
            accelerator_time_millis: 2_000_000,
        }
    }

    fn paired_receipt(
        design: &RandomizedInterventionDesignV1,
        manifest: &RandomizedInterventionManifestV1,
    ) -> PairedTrainingExecutionReceiptV1 {
        let mut receipt = PairedTrainingExecutionReceiptV1 {
            schema: TRAINING_RECEIPT_SCHEMA.to_owned(),
            experiment_id: design.experiment_id.clone(),
            replicate_id: design.replicate_id.clone(),
            design_sha256: expected_design_sha256(design).unwrap(),
            intervention_manifest_sha256: manifest.manifest_sha256.clone(),
            treatment: arm_receipt(design, manifest, TrainingArmV1::Treatment),
            placebo: arm_receipt(design, manifest, TrainingArmV1::Placebo),
            pair_receipt_sha256: String::new(),
            claim_boundary: "synthetic fixture".to_owned(),
        };
        let bytes = {
            let clone = receipt.clone();
            serde_json::to_vec(&clone).unwrap()
        };
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes);
        receipt
    }

    fn validate_fixture() -> Result<PairedTrainingExecutionAuditV1, String> {
        let design = design();
        let manifest = randomize_intervention(&design, &seed())?;
        let mut receipt = paired_receipt(&design, &manifest);
        let bytes = {
            // The receipt digest excludes its own field by clearing it.
            receipt.pair_receipt_sha256.clear();
            let value = serde_json::to_vec(&receipt).unwrap();
            let digest = hash_domain(b"paired-training-execution-receipt.v1", &value);
            receipt.pair_receipt_sha256 = digest;
            serde_json::to_vec(&receipt).unwrap()
        };
        validate_paired_training_execution_receipt(&design, &manifest, &seed(), &bytes)
    }

    #[test]
    fn strict_receipt_and_audit_schemas_match_serialized_struct_fields() {
        let receipt_schema: Value = serde_json::from_str(include_str!(
            "../schema/paired-training-execution-receipt.schema.json"
        ))
        .unwrap();
        let audit_schema: Value = serde_json::from_str(include_str!(
            "../schema/paired-training-execution-audit.schema.json"
        ))
        .unwrap();
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let receipt = paired_receipt(&design, &manifest);
        let receipt_value = serde_json::to_value(&receipt).unwrap();
        let report = validate_paired_training_execution_receipt(
            &design,
            &manifest,
            &seed(),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        let audit_value = serde_json::to_value(&report).unwrap();

        assert_eq!(
            receipt_schema["properties"]["schema"]["const"].as_str(),
            Some(TRAINING_RECEIPT_SCHEMA)
        );
        assert_eq!(
            audit_schema["properties"]["schema"]["const"].as_str(),
            Some(TRAINING_RECEIPT_REPORT_SCHEMA)
        );
        let arm_properties: BTreeSet<String> = receipt_schema["$defs"]["arm"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let arm_required: BTreeSet<String> = receipt_schema["$defs"]["arm"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let treatment_keys: BTreeSet<String> = receipt_value["treatment"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(arm_properties, arm_required);
        assert_eq!(arm_properties, treatment_keys);

        let receipt_required: BTreeSet<String> = receipt_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let receipt_properties: BTreeSet<String> = receipt_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(receipt_required, receipt_properties);
        let receipt_keys: BTreeSet<String> =
            receipt_value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(receipt_keys, receipt_properties);

        let audit_required: BTreeSet<String> = audit_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let audit_properties: BTreeSet<String> = audit_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(audit_required, audit_properties);
        let audit_keys: BTreeSet<String> =
            audit_value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(audit_keys, audit_properties);
    }

    #[test]
    fn receipt_and_audit_schemas_match_serialized_struct_fields() {
        let receipt_schema: Value = serde_json::from_str(include_str!(
            "../schema/paired-training-execution-receipt.schema.json"
        ))
        .unwrap();
        let audit_schema: Value = serde_json::from_str(include_str!(
            "../schema/paired-training-execution-audit.schema.json"
        ))
        .unwrap();
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let receipt = paired_receipt(&design, &manifest);
        let receipt_value = serde_json::to_value(&receipt).unwrap();
        let report = validate_fixture().unwrap();
        let report_value = serde_json::to_value(&report).unwrap();

        assert_eq!(
            receipt_schema["properties"]["schema"]["const"].as_str(),
            Some(TRAINING_RECEIPT_SCHEMA)
        );
        assert_eq!(
            audit_schema["properties"]["schema"]["const"].as_str(),
            Some(TRAINING_RECEIPT_REPORT_SCHEMA)
        );
        let receipt_required: BTreeSet<String> = receipt_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let receipt_properties: BTreeSet<String> = receipt_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let receipt_keys: BTreeSet<String> =
            receipt_value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(receipt_required, receipt_properties);
        assert_eq!(receipt_keys, receipt_properties);

        let arm_schema = &receipt_schema["$defs"]["arm"];
        let arm_required: BTreeSet<String> = arm_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let arm_properties: BTreeSet<String> = arm_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let treatment_keys: BTreeSet<String> = receipt_value["treatment"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let placebo_keys: BTreeSet<String> = receipt_value["placebo"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(arm_required, arm_properties);
        assert_eq!(treatment_keys, arm_properties);
        assert_eq!(placebo_keys, arm_properties);

        let audit_required: BTreeSet<String> = audit_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let audit_properties: BTreeSet<String> = audit_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let audit_keys: BTreeSet<String> =
            report_value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(audit_required, audit_properties);
        assert_eq!(audit_keys, audit_properties);
    }

    #[test]
    fn paired_complete_receipt_is_consistent_but_not_cryptographically_verified() {
        let report = validate_fixture().unwrap();
        assert!(report.valid);
        assert_eq!(
            report.design_sha256,
            expected_design_sha256(&design()).unwrap()
        );
        assert_eq!(
            report.training_rng_seed_commitment_sha256,
            design().training_rng_seed_commitment_sha256
        );
        assert!(report.execution_consistent);
        assert!(report.treatment_completed && report.placebo_completed);
        assert!(report.paired_checkpoints_distinct);
        assert!(report.observed_token_budgets_complete);
        assert!(report.observed_step_budgets_complete);
        assert!(report.ready_for_independent_attestation_verification);
        assert!(!report.independently_verified);
        assert!(
            report
                .external_attestation_verification
                .contains("not_cryptographically_verified")
        );
    }

    #[test]
    fn arm_corpus_swap_fails_against_preregistered_design() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.treatment.training_corpus_sha256 = design.placebo_training_corpus_sha256.clone();
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("treatment training corpus digest")
        );
    }

    #[test]
    fn observed_token_mismatch_fails_even_when_pair_receipt_is_rehashed() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.placebo.observed_total_training_token_count += 1;
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("observed token count does not equal component sum")
        );
    }

    #[test]
    fn partial_run_is_structurally_valid_but_not_execution_consistent() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.treatment.status = TrainingRunStatusV1::Partial;
        receipt.treatment.observed_training_steps = 4_000;
        receipt.treatment.observed_intervention_token_count = 4_000;
        receipt.treatment.observed_total_training_token_count =
            design.shared_training_token_count + 4_000;
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        let report = validate_paired_training_execution_receipt(
            &design,
            &manifest,
            &seed(),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        assert!(!report.execution_consistent);
        assert!(!report.ready_for_independent_attestation_verification);
        assert!(report.issues.iter().any(|issue| issue.contains("Partial")));
    }

    #[test]
    fn treatment_and_placebo_must_share_the_preregistered_training_rng_commitment() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.placebo.training_rng_seed_commitment_sha256 = digest("different-training-seed");
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("trainer code/runtime/tokenizer/configuration")
        );
    }

    #[test]
    fn mismatched_runtime_or_training_configuration_fails() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.placebo.runtime_image_sha256 = digest("different-runtime");
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("trainer code/runtime/tokenizer/configuration")
        );
    }

    #[test]
    fn mismatched_checkpoint_lineage_or_duplicate_outputs_fail() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.placebo.output_checkpoint_sha256 =
            receipt.treatment.output_checkpoint_sha256.clone();
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("output checkpoint digests must differ")
        );
    }

    #[test]
    fn receipt_binds_selected_treatment_and_placebo_payloads() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.treatment.intervention_payload_sha256 = digest("wrong-payload");
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("payload digest does not match")
        );
    }

    #[test]
    fn impossible_calendar_dates_and_clock_values_are_rejected() {
        for timestamp in [
            "2026-02-30T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-10-31T24:00:00Z",
            "2026-10-31T23:60:00Z",
            "2026-10-31T23:59:60Z",
        ] {
            assert!(
                validate_utc_timestamp(timestamp, "test").is_err(),
                "{timestamp}"
            );
        }
        assert!(validate_utc_timestamp("2024-02-29T23:59:59Z", "test").is_ok());
    }

    #[test]
    fn invalid_timestamp_and_missing_external_receipt_digests_fail() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut receipt = paired_receipt(&design, &manifest);
        receipt.treatment.started_at_utc = "not-a-time".to_owned();
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&receipt).unwrap()
            )
            .unwrap_err()
            .contains("YYYY-MM-DDTHH:MM:SSZ")
        );

        let mut receipt = paired_receipt(&design, &manifest);
        receipt.placebo.runner_attestation_sha256 = None;
        receipt.pair_receipt_sha256.clear();
        let bytes0 = serde_json::to_vec(&receipt).unwrap();
        receipt.pair_receipt_sha256 = hash_domain(b"paired-training-execution-receipt.v1", &bytes0);
        let report = validate_paired_training_execution_receipt(
            &design,
            &manifest,
            &seed(),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        assert!(report.execution_consistent);
        assert!(!report.runner_attestation_digests_present);
        assert!(!report.ready_for_independent_attestation_verification);
        assert!(!report.independently_verified);
    }

    #[test]
    fn unknown_receipt_fields_are_rejected() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let receipt = paired_receipt(&design, &manifest);
        let mut json: Value = serde_json::to_value(receipt).unwrap();
        json["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<PairedTrainingExecutionReceiptV1>(json).is_err());
    }

    #[test]
    fn receipt_digest_mutation_fails_closed() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let receipt = paired_receipt(&design, &manifest);
        let mut json: Value = serde_json::to_value(receipt).unwrap();
        json["treatment"]["cpu_time_millis"] = serde_json::json!(999);
        let changed: PairedTrainingExecutionReceiptV1 = serde_json::from_value(json).unwrap();
        assert!(
            validate_paired_training_execution_receipt(
                &design,
                &manifest,
                &seed(),
                &serde_json::to_vec(&changed).unwrap()
            )
            .unwrap_err()
            .contains("digest does not match contents")
        );
    }
}
