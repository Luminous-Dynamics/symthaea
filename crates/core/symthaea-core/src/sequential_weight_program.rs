// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Candidate-only sequential execution for frozen WeightProgram templates.
//!
//! A template is compiled once. Each evidence batch is rebound to the current
//! native candidate fingerprint and materialized as a detached candidate.
//! Session advancement requires an independent verifier receipt. This module
//! never mutates or promotes a live model.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::weight_space_program::{
    ValidatedWeightCandidate, WeightModelSnapshot, WeightProgram, WeightProgramError,
    WeightProgramOperation,
};

/// Reusable edit program and its cumulative session budgets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightProgramTemplate {
    pub format_version: u32,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub target_specification_digest: String,
    pub fault_hypothesis_ids: Vec<String>,
    pub operations: Vec<WeightProgramOperation>,
    pub max_changed_parameters_per_batch: usize,
    pub max_total_l2_delta_per_batch: f32,
    pub max_abs_delta_per_batch: f32,
    /// Sum of per-batch L2 edit norms, not net displacement.
    pub max_cumulative_l2_path: f64,
    /// Counts parameter-change events across batches, not unique coordinates.
    pub max_cumulative_parameter_change_events: usize,
    pub max_batches: usize,
}

/// One sealed evidence batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequentialEvidenceBatch {
    pub session_id: String,
    pub batch_index: usize,
    pub parent_candidate_model_digest: String,
    pub evidence_batch_digest: String,
    pub observation_manifest_digest: String,
}

/// Persistent, serializable candidate-session state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialEditSession {
    pub format_version: u32,
    pub session_id: String,
    pub template_digest: String,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub target_specification_digest: String,
    pub initial_model_digest: String,
    pub latest_candidate_model_digest: String,
    pub next_batch_index: usize,
    pub cumulative_l2_path: f64,
    pub cumulative_parameter_change_events: usize,
    pub processed_evidence_batch_digests: Vec<String>,
    pub previous_step_receipt_digest: Option<String>,
}

/// Candidate prepared on a detached generic snapshot; not yet accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedSequentialCandidate {
    pub session_digest: String,
    pub template_digest: String,
    pub batch: SequentialEvidenceBatch,
    pub program: WeightProgram,
    pub candidate: ValidatedWeightCandidate,
    pub prepared_digest: String,
}

/// Receipt linking one accepted evidence batch to its parent and verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialStepReceipt {
    pub format_version: u32,
    pub session_id: String,
    pub batch_index: usize,
    pub template_digest: String,
    pub parent_candidate_model_digest: String,
    pub evidence_batch_digest: String,
    pub observation_manifest_digest: String,
    pub program_digest: String,
    pub candidate_snapshot_digest: String,
    pub candidate_native_model_digest: String,
    pub verifier_receipt_digest: String,
    pub previous_step_receipt_digest: Option<String>,
    pub cumulative_l2_path_after: f64,
    pub cumulative_parameter_change_events_after: usize,
    pub step_receipt_digest: String,
}

/// Fail-closed sequential session errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SequentialWeightProgramError {
    #[error(transparent)]
    Program(#[from] WeightProgramError),
    #[error("unsupported sequential template version {0}")]
    UnsupportedTemplateVersion(u32),
    #[error("invalid digest: {0}")]
    InvalidDigest(&'static str),
    #[error("template does not match the model snapshot: {0}")]
    TemplateMismatch(&'static str),
    #[error("template has missing or duplicate fault hypotheses")]
    InvalidFaultHypotheses,
    #[error("template requires non-empty operations and positive budgets")]
    InvalidTemplateBudget,
    #[error("session has reached its maximum batch count")]
    BatchLimitExceeded,
    #[error("evidence batch belongs to a different session")]
    SessionMismatch,
    #[error("batch {actual} does not equal expected batch {expected}")]
    OutOfOrderBatch { expected: usize, actual: usize },
    #[error("evidence batch parent fingerprint is stale")]
    StaleParentCandidate,
    #[error("evidence batch has already been processed")]
    EvidenceBatchReplay,
    #[error("prepared candidate is stale or was modified")]
    StalePreparedCandidate,
    #[error("candidate native digest is invalid or unchanged")]
    InvalidCandidateNativeDigest,
    #[error("independent verifier rejected the candidate")]
    VerifierRejectedCandidate,
    #[error("cumulative L2 edit path exceeds the session budget")]
    CumulativeL2BudgetExceeded,
    #[error("cumulative parameter-change events exceed the session budget")]
    CumulativeParameterBudgetExceeded,
}

/// Begin a session anchored to one exact native model snapshot.
pub fn begin_sequential_edit_session(
    session_id: impl Into<String>,
    template: &SequentialWeightProgramTemplate,
    snapshot: &WeightModelSnapshot,
) -> Result<SequentialEditSession, SequentialWeightProgramError> {
    validate_template(template)?;
    let session_id = session_id.into();
    validate_digest(&session_id)?;
    validate_snapshot_binding(template, snapshot)?;
    Ok(SequentialEditSession {
        format_version: 1,
        session_id,
        template_digest: template_digest(template)?,
        architecture_digest: snapshot.architecture_digest.clone(),
        manifest_digest: snapshot.manifest_digest.clone(),
        target_specification_digest: snapshot.target_specification_digest.clone(),
        initial_model_digest: snapshot.model_digest.clone(),
        latest_candidate_model_digest: snapshot.model_digest.clone(),
        next_batch_index: 0,
        cumulative_l2_path: 0.0,
        cumulative_parameter_change_events: 0,
        processed_evidence_batch_digests: Vec::new(),
        previous_step_receipt_digest: None,
    })
}

/// Prepare a detached candidate for one strictly ordered evidence batch.
///
/// Preparation does not advance the session. The caller must materialize the
/// candidate through the native SWI adapter and independently verify it before
/// calling accept_verified_sequential_candidate.
pub fn prepare_sequential_candidate(
    session: &SequentialEditSession,
    template: &SequentialWeightProgramTemplate,
    snapshot: &WeightModelSnapshot,
    batch: &SequentialEvidenceBatch,
) -> Result<PreparedSequentialCandidate, SequentialWeightProgramError> {
    validate_template(template)?;
    validate_session_binding(session, template, snapshot)?;
    if session.next_batch_index >= template.max_batches {
        return Err(SequentialWeightProgramError::BatchLimitExceeded);
    }
    if batch.session_id != session.session_id {
        return Err(SequentialWeightProgramError::SessionMismatch);
    }
    if batch.batch_index != session.next_batch_index {
        return Err(SequentialWeightProgramError::OutOfOrderBatch {
            expected: session.next_batch_index,
            actual: batch.batch_index,
        });
    }
    if batch.parent_candidate_model_digest != session.latest_candidate_model_digest
        || snapshot.model_digest != session.latest_candidate_model_digest
    {
        return Err(SequentialWeightProgramError::StaleParentCandidate);
    }
    validate_digest(&batch.evidence_batch_digest)?;
    validate_digest(&batch.observation_manifest_digest)?;
    if session
        .processed_evidence_batch_digests
        .contains(&batch.evidence_batch_digest)
    {
        return Err(SequentialWeightProgramError::EvidenceBatchReplay);
    }

    let program = WeightProgram {
        format_version: 1,
        architecture_digest: template.architecture_digest.clone(),
        manifest_digest: template.manifest_digest.clone(),
        base_model_digest: snapshot.model_digest.clone(),
        target_specification_digest: template.target_specification_digest.clone(),
        fault_hypothesis_ids: template.fault_hypothesis_ids.clone(),
        evidence_receipt_ids: vec![
            batch.evidence_batch_digest.clone(),
            batch.observation_manifest_digest.clone(),
        ],
        max_changed_parameters: template.max_changed_parameters_per_batch,
        max_total_l2_delta: template.max_total_l2_delta_per_batch,
        max_abs_delta: template.max_abs_delta_per_batch,
        operations: template.operations.clone(),
    };
    let candidate = program.validate_and_materialize(snapshot)?;
    let session_digest = session_digest(session)?;
    let template_digest = session.template_digest.clone();
    let prepared_digest = prepared_digest(
        &session_digest,
        &template_digest,
        batch,
        &program,
        &candidate,
    )?;
    Ok(PreparedSequentialCandidate {
        session_digest,
        template_digest,
        batch: batch.clone(),
        program,
        candidate,
        prepared_digest,
    })
}

/// Advance session state only after an independent verifier accepts the candidate.
///
/// The verifier receipt must bind the candidate native fingerprint and all
/// held-out/retention/negative-control results. This function records the
/// attestation but does not promote or mutate a live model.
pub fn accept_verified_sequential_candidate(
    session: &SequentialEditSession,
    template: &SequentialWeightProgramTemplate,
    prepared: &PreparedSequentialCandidate,
    candidate_native_model_digest: &str,
    verifier_receipt_digest: &str,
    verified: bool,
) -> Result<(SequentialEditSession, SequentialStepReceipt), SequentialWeightProgramError> {
    validate_template(template)?;
    if session.format_version != 1
        || session.template_digest != template_digest(template)?
    {
        return Err(SequentialWeightProgramError::StalePreparedCandidate);
    }
    if prepared.session_digest != session_digest(session)?
        || prepared.template_digest != session.template_digest
        || prepared.batch.session_id != session.session_id
        || prepared.batch.batch_index != session.next_batch_index
        || prepared.batch.parent_candidate_model_digest != session.latest_candidate_model_digest
        || prepared.program.base_model_digest != session.latest_candidate_model_digest
        || prepared.candidate.receipt.base_model_digest != session.latest_candidate_model_digest
        || prepared.program.architecture_digest != session.architecture_digest
        || prepared.program.manifest_digest != session.manifest_digest
        || prepared.program.target_specification_digest != session.target_specification_digest
    {
        return Err(SequentialWeightProgramError::StalePreparedCandidate);
    }
    let expected_program_digest = hash_value(
        b"symthaea-weight-program-v1\0",
        &prepared.program,
    )?;
    if expected_program_digest != prepared.candidate.receipt.program_digest {
        return Err(SequentialWeightProgramError::StalePreparedCandidate);
    }
    let expected_prepared_digest = prepared_digest(
        &prepared.session_digest,
        &prepared.template_digest,
        &prepared.batch,
        &prepared.program,
        &prepared.candidate,
    )?;
    if expected_prepared_digest != prepared.prepared_digest {
        return Err(SequentialWeightProgramError::StalePreparedCandidate);
    }
    if !verified {
        return Err(SequentialWeightProgramError::VerifierRejectedCandidate);
    }
    validate_digest(candidate_native_model_digest)?;
    validate_digest(verifier_receipt_digest)?;
    if candidate_native_model_digest == session.latest_candidate_model_digest {
        return Err(SequentialWeightProgramError::InvalidCandidateNativeDigest);
    }
    if session
        .processed_evidence_batch_digests
        .contains(&prepared.batch.evidence_batch_digest)
    {
        return Err(SequentialWeightProgramError::EvidenceBatchReplay);
    }

    let cumulative_l2_path =
        session.cumulative_l2_path + f64::from(prepared.candidate.receipt.total_l2_delta);
    if !cumulative_l2_path.is_finite()
        || cumulative_l2_path > template.max_cumulative_l2_path
    {
        return Err(SequentialWeightProgramError::CumulativeL2BudgetExceeded);
    }
    let cumulative_parameter_change_events = session
        .cumulative_parameter_change_events
        .checked_add(prepared.candidate.receipt.changed_parameter_count)
        .ok_or(SequentialWeightProgramError::CumulativeParameterBudgetExceeded)?;
    if cumulative_parameter_change_events > template.max_cumulative_parameter_change_events {
        return Err(SequentialWeightProgramError::CumulativeParameterBudgetExceeded);
    }

    let mut next = session.clone();
    next.latest_candidate_model_digest = candidate_native_model_digest.to_owned();
    next.next_batch_index += 1;
    next.cumulative_l2_path = cumulative_l2_path;
    next.cumulative_parameter_change_events = cumulative_parameter_change_events;
    next.processed_evidence_batch_digests
        .push(prepared.batch.evidence_batch_digest.clone());

    let mut receipt = SequentialStepReceipt {
        format_version: 1,
        session_id: session.session_id.clone(),
        batch_index: prepared.batch.batch_index,
        template_digest: session.template_digest.clone(),
        parent_candidate_model_digest: session.latest_candidate_model_digest.clone(),
        evidence_batch_digest: prepared.batch.evidence_batch_digest.clone(),
        observation_manifest_digest: prepared.batch.observation_manifest_digest.clone(),
        program_digest: prepared.candidate.receipt.program_digest.clone(),
        candidate_snapshot_digest: prepared.candidate.receipt.candidate_snapshot_digest.clone(),
        candidate_native_model_digest: candidate_native_model_digest.to_owned(),
        verifier_receipt_digest: verifier_receipt_digest.to_owned(),
        previous_step_receipt_digest: session.previous_step_receipt_digest.clone(),
        cumulative_l2_path_after: cumulative_l2_path,
        cumulative_parameter_change_events_after: cumulative_parameter_change_events,
        step_receipt_digest: String::new(),
    };
    receipt.step_receipt_digest = step_receipt_digest(&receipt)?;
    next.previous_step_receipt_digest = Some(receipt.step_receipt_digest.clone());
    Ok((next, receipt))
}

fn validate_template(
    template: &SequentialWeightProgramTemplate,
) -> Result<(), SequentialWeightProgramError> {
    if template.format_version != 1 {
        return Err(SequentialWeightProgramError::UnsupportedTemplateVersion(
            template.format_version,
        ));
    }
    validate_digest(&template.architecture_digest)?;
    validate_digest(&template.manifest_digest)?;
    validate_digest(&template.target_specification_digest)?;
    let mut hypotheses = template.fault_hypothesis_ids.clone();
    hypotheses.sort();
    hypotheses.dedup();
    if template.fault_hypothesis_ids.is_empty()
        || hypotheses.len() != template.fault_hypothesis_ids.len()
        || template.fault_hypothesis_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(SequentialWeightProgramError::InvalidFaultHypotheses);
    }
    if template.operations.is_empty()
        || template.max_changed_parameters_per_batch == 0
        || !template.max_total_l2_delta_per_batch.is_finite()
        || template.max_total_l2_delta_per_batch <= 0.0
        || !template.max_abs_delta_per_batch.is_finite()
        || template.max_abs_delta_per_batch <= 0.0
        || !template.max_cumulative_l2_path.is_finite()
        || template.max_cumulative_l2_path <= 0.0
        || template.max_cumulative_parameter_change_events == 0
        || template.max_batches == 0
    {
        return Err(SequentialWeightProgramError::InvalidTemplateBudget);
    }
    Ok(())
}

fn validate_snapshot_binding(
    template: &SequentialWeightProgramTemplate,
    snapshot: &WeightModelSnapshot,
) -> Result<(), SequentialWeightProgramError> {
    if template.architecture_digest != snapshot.architecture_digest {
        return Err(SequentialWeightProgramError::TemplateMismatch("architecture"));
    }
    if template.manifest_digest != snapshot.manifest_digest {
        return Err(SequentialWeightProgramError::TemplateMismatch("manifest"));
    }
    if template.target_specification_digest != snapshot.target_specification_digest {
        return Err(SequentialWeightProgramError::TemplateMismatch("task"));
    }
    Ok(())
}

fn validate_session_binding(
    session: &SequentialEditSession,
    template: &SequentialWeightProgramTemplate,
    snapshot: &WeightModelSnapshot,
) -> Result<(), SequentialWeightProgramError> {
    if session.format_version != 1 {
        return Err(SequentialWeightProgramError::StalePreparedCandidate);
    }
    validate_snapshot_binding(template, snapshot)?;
    if session.template_digest != template_digest(template)?
        || session.architecture_digest != snapshot.architecture_digest
        || session.manifest_digest != snapshot.manifest_digest
        || session.target_specification_digest != snapshot.target_specification_digest
        || session.latest_candidate_model_digest != snapshot.model_digest
    {
        return Err(SequentialWeightProgramError::TemplateMismatch("session/model binding"));
    }
    Ok(())
}

fn template_digest(
    template: &SequentialWeightProgramTemplate,
) -> Result<String, SequentialWeightProgramError> {
    hash_value(b"symthaea-sequential-weight-template-v1\0", template)
}

fn session_digest(
    session: &SequentialEditSession,
) -> Result<String, SequentialWeightProgramError> {
    hash_value(b"symthaea-sequential-edit-session-v1\0", session)
}

fn prepared_digest(
    session: &str,
    template: &str,
    batch: &SequentialEvidenceBatch,
    program: &WeightProgram,
    candidate: &ValidatedWeightCandidate,
) -> Result<String, SequentialWeightProgramError> {
    hash_value(
        b"symthaea-sequential-prepared-candidate-v1\0",
        &(session, template, batch, program, candidate),
    )
}

fn step_receipt_digest(
    receipt: &SequentialStepReceipt,
) -> Result<String, SequentialWeightProgramError> {
    let mut unsigned = receipt.clone();
    unsigned.step_receipt_digest.clear();
    hash_value(b"symthaea-sequential-step-receipt-v1\0", &unsigned)
}

fn hash_value<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, SequentialWeightProgramError> {
    let encoded = serde_json::to_vec(value)
        .map_err(|_| SequentialWeightProgramError::InvalidDigest("serialized value"))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_digest(value: &str) -> Result<(), SequentialWeightProgramError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SequentialWeightProgramError::InvalidDigest("sha256/blake3 digest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_space_program::{
        CfCWeightGroupId, ParameterGroupConstraints, ParameterGroupRole,
        ParameterGroupSnapshot, WeightProgramTarget,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn snapshot() -> WeightModelSnapshot {
        WeightModelSnapshot {
            format_version: 1,
            architecture_digest: digest('a'),
            manifest_digest: digest('b'),
            model_digest: digest('c'),
            target_specification_digest: digest('d'),
            groups: vec![ParameterGroupSnapshot {
                target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
                role: ParameterGroupRole::TimeConstant,
                shape: vec![2],
                values: vec![1.0, 1.5],
                constraints: ParameterGroupConstraints {
                    min_value: Some(0.01),
                    max_value: Some(10.0),
                    ..Default::default()
                },
            }],
        }
    }

    fn template() -> SequentialWeightProgramTemplate {
        SequentialWeightProgramTemplate {
            format_version: 1,
            architecture_digest: digest('a'),
            manifest_digest: digest('b'),
            target_specification_digest: digest('d'),
            fault_hypothesis_ids: vec!["fault:tau-regime".into()],
            operations: vec![WeightProgramOperation::ScaleRange {
                target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
                start: 0,
                end: 2,
                factor: 1.01,
            }],
            max_changed_parameters_per_batch: 2,
            max_total_l2_delta_per_batch: 1.0,
            max_abs_delta_per_batch: 0.1,
            max_cumulative_l2_path: 0.03,
            max_cumulative_parameter_change_events: 8,
            max_batches: 3,
        }
    }

    fn batch(session_id: &str, index: usize, parent: &str, evidence: char) -> SequentialEvidenceBatch {
        SequentialEvidenceBatch {
            session_id: session_id.to_owned(),
            batch_index: index,
            parent_candidate_model_digest: parent.to_owned(),
            evidence_batch_digest: digest(evidence),
            observation_manifest_digest: digest('e'),
        }
    }

    #[test]
    fn fixed_template_advances_only_after_verifier_receipt() {
        let base = snapshot();
        let template = template();
        let session_id = digest('f');
        let session = begin_sequential_edit_session(session_id.clone(), &template, &base)
            .expect("begin");
        let prepared = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare");
        assert_eq!(session.next_batch_index, 0, "preparation is not advancement");
        assert_eq!(prepared.candidate.receipt.changed_parameter_count, 2);
        assert!(prepared.candidate.receipt.total_l2_delta > 0.0);
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &digest('8'),
                false,
            ),
            Err(SequentialWeightProgramError::VerifierRejectedCandidate)
        ));
        let (advanced, receipt) = accept_verified_sequential_candidate(
            &session,
            &template,
            &prepared,
            &digest('9'),
            &digest('8'),
            true,
        )
        .expect("verified batch advances");
        assert_eq!(advanced.next_batch_index, 1);
        assert_eq!(advanced.latest_candidate_model_digest, digest('9'));
        assert_eq!(receipt.parent_candidate_model_digest, base.model_digest);
        assert_eq!(receipt.candidate_native_model_digest, digest('9'));
        assert_eq!(
            advanced.previous_step_receipt_digest,
            Some(receipt.step_receipt_digest)
        );
    }

    #[test]
    fn replay_out_of_order_stale_parent_and_modified_candidate_reject() {
        let base = snapshot();
        let template = template();
        let session_id = digest('f');
        let session = begin_sequential_edit_session(session_id.clone(), &template, &base)
            .expect("begin");
        assert!(matches!(
            prepare_sequential_candidate(
                &session,
                &template,
                &base,
                &batch(&session_id, 1, &base.model_digest, '1')
            ),
            Err(SequentialWeightProgramError::OutOfOrderBatch { .. })
        ));
        let prepared = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare");
        assert!(matches!(
            prepare_sequential_candidate(
                &session,
                &template,
                &base,
                &batch(&session_id, 0, &digest('7'), '2')
            ),
            Err(SequentialWeightProgramError::StaleParentCandidate)
        ));
        let mut modified = prepared.clone();
        modified.batch.evidence_batch_digest = digest('2');
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &modified,
                &digest('9'),
                &digest('8'),
                true,
            ),
            Err(SequentialWeightProgramError::StalePreparedCandidate)
        ));
    }

    #[test]
    fn cumulative_l2_path_budget_is_enforced_across_accepted_batches() {
        let base = snapshot();
        let template = template();
        let session_id = digest('f');
        let session = begin_sequential_edit_session(session_id.clone(), &template, &base)
            .expect("begin");
        let prepared0 = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare batch zero");
        let (advanced, _) = accept_verified_sequential_candidate(
            &session,
            &template,
            &prepared0,
            &digest('9'),
            &digest('8'),
            true,
        )
        .expect("accept batch zero");

        let mut next_snapshot = base;
        next_snapshot.model_digest = digest('9');
        next_snapshot.groups = prepared0.candidate.groups.clone();
        let prepared1 = prepare_sequential_candidate(
            &advanced,
            &template,
            &next_snapshot,
            &batch(&session_id, 1, &digest('9'), '2'),
        )
        .expect("prepare batch one");
        assert!(matches!(
            accept_verified_sequential_candidate(
                &advanced,
                &template,
                &prepared1,
                &digest('7'),
                &digest('6'),
                true,
            ),
            Err(SequentialWeightProgramError::CumulativeL2BudgetExceeded)
        ));
    }

    #[test]
    fn template_digest_changes_when_operation_or_budget_changes() {
        let original = template();
        let mut changed = original.clone();
        changed.max_cumulative_l2_path += 0.1;
        assert_ne!(
            template_digest(&original).expect("original"),
            template_digest(&changed).expect("changed")
        );
    }
}
