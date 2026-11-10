// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Candidate-only sequential execution for frozen WeightProgram templates.
//!
//! A template is compiled once. Each evidence batch is rebound to the current
//! native candidate fingerprint and materialized as a detached candidate.
//! Session advancement requires an independent verifier receipt. This module
//! never mutates or promotes a live model.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::family_guardrail::{FamilyGuardrailStatus, FamilyGuardrailVerifier};
use crate::sequential_weight_verification::{
    ConfirmatoryEpisodeManifest, SequentialWeightVerificationError,
    SequentialWeightVerificationReport, validate_qualified_report,
};
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
    /// Fingerprint of the pinned verifier public key that authenticated this step.
    pub verifier_key_id: String,
    /// Versioned protocol used by the independent verifier.
    pub verification_protocol_digest: String,
    /// Digest of held-out, retention, negative-control, and constraint results.
    pub result_manifest_digest: String,
    /// Digest of the complete signed verifier-attestation envelope.
    pub verifier_receipt_digest: String,
    /// Open-world adequacy precondition bound into the signed verifier claims.
    pub precondition_gate_digest: String,
    /// Anytime-valid paired-loss report bound into the signed verifier claims.
    pub statistical_verification_report_digest: String,
    /// All-preregistered-families non-inferiority report bound into the signed claims.
    pub family_guardrail_report_digest: String,
    /// Combined aggregate-plus-family alpha budget, stored as exact IEEE-754 bits.
    pub combined_campaign_alpha_budget_bits: u64,
    pub previous_step_receipt_digest: Option<String>,
    pub cumulative_l2_path_after: f64,
    pub cumulative_parameter_change_events_after: usize,
    pub step_receipt_digest: String,
}

/// Claims an independent verifier must sign after checking a candidate.
///
/// The verifier protocol must independently recompute the native model digest,
/// run held-out and retention probes, evaluate negative controls, and record
/// its results in the manifest identified by `result_manifest_digest`. The
/// caller must resolve the verifying key from a pinned trust store; a key
/// supplied by the candidate proposer is not a trust anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequentialVerifierAttestationClaims {
    pub format_version: u32,
    pub verifier_id: String,
    pub verification_protocol_digest: String,
    pub result_manifest_digest: String,
    pub session_id: String,
    pub batch_index: usize,
    pub template_digest: String,
    pub parent_candidate_model_digest: String,
    pub candidate_native_model_digest: String,
    pub target_specification_digest: String,
    pub evidence_batch_digest: String,
    pub observation_manifest_digest: String,
    pub program_digest: String,
    pub candidate_snapshot_digest: String,
    /// Open-world/model-adequacy report digest that permitted detached evaluation.
    pub precondition_gate_digest: String,
    /// Qualified anytime-valid paired-loss report digest.
    pub statistical_verification_report_digest: String,
    /// Qualified all-families non-inferiority report digest.
    pub family_guardrail_report_digest: String,
    /// Preregistered combined alpha budget, stored as exact IEEE-754 bits.
    pub combined_campaign_alpha_budget_bits: u64,
    /// False is a signed rejection. A verifier must never sign true before all
    /// required tests and negative controls have completed.
    pub accepted: bool,
}

/// Digest binding obtained only after independent aggregate and family-level
/// statistical reports have been recomputed and matched to one frozen candidate.
///
/// Fields are private so callers cannot manufacture a binding from arbitrary
/// digest strings; use [`bind_sequential_verification_evidence`] instead.
#[derive(Debug, PartialEq, Eq)]
pub struct SequentialVerifierEvidenceBinding {
    precondition_gate_digest: String,
    statistical_verification_report_digest: String,
    family_guardrail_report_digest: String,
    combined_campaign_alpha_budget_bits: u64,
}

impl SequentialVerifierEvidenceBinding {
    /// Open-world gate digest that preceded detached candidate evaluation.
    pub fn precondition_gate_digest(&self) -> &str {
        &self.precondition_gate_digest
    }

    /// Digest of the independently qualified aggregate sequential report.
    pub fn statistical_verification_report_digest(&self) -> &str {
        &self.statistical_verification_report_digest
    }

    /// Digest of the independently qualified all-family non-inferiority report.
    pub fn family_guardrail_report_digest(&self) -> &str {
        &self.family_guardrail_report_digest
    }

    /// Exact preregistered total alpha budget included in the verifier's signed claims.
    pub fn combined_campaign_alpha_budget_bits(&self) -> u64 {
        self.combined_campaign_alpha_budget_bits
    }
}

/// Validate and bind aggregate benefit plus all-family retention evidence to
/// the exact sequential edit candidate.
///
/// Both reports must be recomputable and terminally qualified. Their candidate
/// ordinal, baseline/candidate identities, evaluation protocol and global
/// confirmatory manifest must match. The two alpha allocations must fit inside
/// the caller's preregistered combined campaign budget.
pub fn bind_sequential_verification_evidence(
    session: &SequentialEditSession,
    candidate_native_model_digest: &str,
    report: &SequentialWeightVerificationReport,
    family_verifier: &FamilyGuardrailVerifier,
    combined_alpha_budget: f64,
) -> Result<SequentialVerifierEvidenceBinding, SequentialWeightProgramError> {
    validate_qualified_report(report)?;
    validate_digest(candidate_native_model_digest)?;
    if report.binding.baseline_model_digest != session.latest_candidate_model_digest {
        return Err(SequentialWeightProgramError::StatisticalEvidenceMismatch(
            "baseline_model_digest",
        ));
    }
    if report.binding.candidate_model_digest != candidate_native_model_digest {
        return Err(SequentialWeightProgramError::StatisticalEvidenceMismatch(
            "candidate_model_digest",
        ));
    }
    validate_digest(&report.binding.precondition_gate_digest)?;

    let family_report = family_verifier
        .report()
        .map_err(|_| SequentialWeightProgramError::FamilyGuardrailReportInvalid)?;
    let family_manifest = family_verifier.manifest();
    if family_report.status != FamilyGuardrailStatus::AllFamiliesNonInferior {
        return Err(SequentialWeightProgramError::FamilyGuardrailNotQualified);
    }
    let family_episode_ids = family_manifest
        .families
        .iter()
        .flat_map(|family| family.episode_ids.iter().cloned())
        .collect::<Vec<_>>();
    let reconstructed_confirmatory_manifest =
        ConfirmatoryEpisodeManifest::new(family_episode_ids)?;
    if reconstructed_confirmatory_manifest.manifest_digest
        != report.binding.confirmatory_episode_manifest_digest
        || reconstructed_confirmatory_manifest.manifest_digest
            != family_manifest.confirmatory_manifest_digest
    {
        return Err(SequentialWeightProgramError::FamilyGuardrailEvidenceMismatch(
            "confirmatory_manifest_membership",
        ));
    }

    for (field, matches) in [
        (
            "candidate_index",
            family_report.candidate_index == report.binding.candidate_index
                && family_manifest.candidate_index == report.binding.candidate_index,
        ),
        (
            "baseline_model_digest",
            family_report.baseline_model_digest == session.latest_candidate_model_digest
                && family_manifest.baseline_model_digest == session.latest_candidate_model_digest,
        ),
        (
            "candidate_model_digest",
            family_report.candidate_model_digest == candidate_native_model_digest
                && family_manifest.candidate_model_digest == candidate_native_model_digest,
        ),
        (
            "evaluation_protocol_digest",
            family_report.evaluation_protocol_digest
                == report.binding.evaluation_protocol_digest
                && family_manifest.evaluation_protocol_digest
                    == report.binding.evaluation_protocol_digest,
        ),
        (
            "confirmatory_manifest_digest",
            family_manifest.confirmatory_manifest_digest
                == report.binding.confirmatory_episode_manifest_digest,
        ),
        (
            "manifest_digest",
            family_report.manifest_digest == family_manifest.manifest_digest,
        ),
    ] {
        if !matches {
            return Err(SequentialWeightProgramError::FamilyGuardrailEvidenceMismatch(field));
        }
    }
    validate_digest(&family_report.report_digest)?;

    let allocated_alpha =
        report.config.campaign_alpha + family_manifest.campaign_alpha;
    if !combined_alpha_budget.is_finite()
        || !(0.0..1.0).contains(&combined_alpha_budget)
        || !allocated_alpha.is_finite()
        || allocated_alpha > combined_alpha_budget
    {
        return Err(SequentialWeightProgramError::CombinedAlphaBudgetExceeded);
    }

    Ok(SequentialVerifierEvidenceBinding {
        precondition_gate_digest: report.binding.precondition_gate_digest.clone(),
        statistical_verification_report_digest: report.report_digest.clone(),
        family_guardrail_report_digest: family_report.report_digest,
        combined_campaign_alpha_budget_bits: combined_alpha_budget.to_bits(),
    })
}

/// Ed25519-signed verifier claims. The private signing key remains outside this crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedSequentialVerifierAttestation {
    pub claims: SequentialVerifierAttestationClaims,
    pub verifier_key_id: String,
    pub signature: Vec<u8>,
}

/// Stable, domain-separated message digest signed by the verifier.
pub fn sequential_verifier_attestation_message(
    claims: &SequentialVerifierAttestationClaims,
) -> Result<[u8; 32], SequentialWeightProgramError> {
    let encoded = serde_json::to_vec(claims).map_err(|_| {
        SequentialWeightProgramError::InvalidDigest("verifier claims serialization")
    })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-sequential-verifier-attestation-v3\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

/// Stable identifier for a verifier public key. Trust still comes from the
/// caller's pinned key registry, not from this identifier alone.
pub fn sequential_verifier_key_id(verifying_key: &VerifyingKey) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-sequential-verifier-key-id-v1\0");
    hasher.update(&verifying_key.to_bytes());
    hasher.finalize().to_hex().to_string()
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
    #[error("verifier key ID does not match the pinned trust key")]
    VerifierKeyIdMismatch,
    #[error("verifier signature is malformed or invalid")]
    InvalidVerifierSignature,
    #[error("verifier attestation field does not match the prepared candidate: {0}")]
    VerifierAttestationMismatch(&'static str),
    #[error("statistical verification evidence is not bound to this candidate: {0}")]
    StatisticalEvidenceMismatch(&'static str),
    #[error("family-level non-inferiority evidence has not qualified every registered family")]
    FamilyGuardrailNotQualified,
    #[error("family guardrail report could not be independently recomputed")]
    FamilyGuardrailReportInvalid,
    #[error("family guardrail evidence does not match the aggregate candidate: {0}")]
    FamilyGuardrailEvidenceMismatch(&'static str),
    #[error("aggregate and family alpha allocations exceed the combined campaign budget")]
    CombinedAlphaBudgetExceeded,
    #[error(transparent)]
    StatisticalVerification(#[from] SequentialWeightVerificationError),
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
    evidence_binding: &SequentialVerifierEvidenceBinding,
    attestation: &SignedSequentialVerifierAttestation,
    pinned_verifier_key: &VerifyingKey,
) -> Result<(SequentialEditSession, SequentialStepReceipt), SequentialWeightProgramError> {
    validate_template(template)?;
    if session.format_version != 1 || session.template_digest != template_digest(template)? {
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
    let expected_program_digest = hash_value(b"symthaea-weight-program-v1\0", &prepared.program)?;
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
    validate_digest(candidate_native_model_digest)?;
    let claims = &attestation.claims;
    if claims.format_version != 3 {
        return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
            "format_version",
        ));
    }
    for (field, value) in [
        (
            "verification_protocol_digest",
            claims.verification_protocol_digest.as_str(),
        ),
        (
            "result_manifest_digest",
            claims.result_manifest_digest.as_str(),
        ),
        ("template_digest", claims.template_digest.as_str()),
        (
            "parent_candidate_model_digest",
            claims.parent_candidate_model_digest.as_str(),
        ),
        (
            "candidate_native_model_digest",
            claims.candidate_native_model_digest.as_str(),
        ),
        (
            "target_specification_digest",
            claims.target_specification_digest.as_str(),
        ),
        (
            "evidence_batch_digest",
            claims.evidence_batch_digest.as_str(),
        ),
        (
            "observation_manifest_digest",
            claims.observation_manifest_digest.as_str(),
        ),
        ("program_digest", claims.program_digest.as_str()),
        (
            "candidate_snapshot_digest",
            claims.candidate_snapshot_digest.as_str(),
        ),
        (
            "precondition_gate_digest",
            claims.precondition_gate_digest.as_str(),
        ),
        (
            "statistical_verification_report_digest",
            claims.statistical_verification_report_digest.as_str(),
        ),
        (
            "family_guardrail_report_digest",
            claims.family_guardrail_report_digest.as_str(),
        ),
    ] {
        validate_digest(value)?;
        if value.is_empty() {
            return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                field,
            ));
        }
    }
    let claimed_alpha_budget = f64::from_bits(claims.combined_campaign_alpha_budget_bits);
    if !claimed_alpha_budget.is_finite() || !(0.0..1.0).contains(&claimed_alpha_budget) {
        return Err(SequentialWeightProgramError::CombinedAlphaBudgetExceeded);
    }
    if claims.verifier_id.trim().is_empty() {
        return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
            "verifier_id",
        ));
    }
    if attestation.verifier_key_id != sequential_verifier_key_id(pinned_verifier_key) {
        return Err(SequentialWeightProgramError::VerifierKeyIdMismatch);
    }
    if claims.session_id != session.session_id {
        return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
            "session_id",
        ));
    }
    if claims.batch_index != prepared.batch.batch_index {
        return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
            "batch_index",
        ));
    }
    for (field, actual, expected) in [
        (
            "template_digest",
            claims.template_digest.as_str(),
            prepared.template_digest.as_str(),
        ),
        (
            "parent_candidate_model_digest",
            claims.parent_candidate_model_digest.as_str(),
            session.latest_candidate_model_digest.as_str(),
        ),
        (
            "candidate_native_model_digest",
            claims.candidate_native_model_digest.as_str(),
            candidate_native_model_digest,
        ),
        (
            "target_specification_digest",
            claims.target_specification_digest.as_str(),
            session.target_specification_digest.as_str(),
        ),
        (
            "evidence_batch_digest",
            claims.evidence_batch_digest.as_str(),
            prepared.batch.evidence_batch_digest.as_str(),
        ),
        (
            "observation_manifest_digest",
            claims.observation_manifest_digest.as_str(),
            prepared.batch.observation_manifest_digest.as_str(),
        ),
        (
            "program_digest",
            claims.program_digest.as_str(),
            prepared.candidate.receipt.program_digest.as_str(),
        ),
        (
            "candidate_snapshot_digest",
            claims.candidate_snapshot_digest.as_str(),
            prepared
                .candidate
                .receipt
                .candidate_snapshot_digest
                .as_str(),
        ),
        (
            "precondition_gate_digest",
            claims.precondition_gate_digest.as_str(),
            evidence_binding.precondition_gate_digest(),
        ),
        (
            "statistical_verification_report_digest",
            claims.statistical_verification_report_digest.as_str(),
            evidence_binding.statistical_verification_report_digest(),
        ),
        (
            "family_guardrail_report_digest",
            claims.family_guardrail_report_digest.as_str(),
            evidence_binding.family_guardrail_report_digest(),
        ),
    ] {
        if actual != expected {
            return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                field,
            ));
        }
    }
    if claims.combined_campaign_alpha_budget_bits
        != evidence_binding.combined_campaign_alpha_budget_bits()
    {
        return Err(SequentialWeightProgramError::VerifierAttestationMismatch(
            "combined_campaign_alpha_budget_bits",
        ));
    }
    let message = sequential_verifier_attestation_message(claims)?;
    let signature = Signature::from_slice(&attestation.signature)
        .map_err(|_| SequentialWeightProgramError::InvalidVerifierSignature)?;
    pinned_verifier_key
        .verify(&message, &signature)
        .map_err(|_| SequentialWeightProgramError::InvalidVerifierSignature)?;
    if !claims.accepted {
        return Err(SequentialWeightProgramError::VerifierRejectedCandidate);
    }
    let verifier_receipt_digest =
        hash_value(b"symthaea-sequential-verifier-envelope-v3\0", attestation)?;
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
    if !cumulative_l2_path.is_finite() || cumulative_l2_path > template.max_cumulative_l2_path {
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
        format_version: 3,
        session_id: session.session_id.clone(),
        batch_index: prepared.batch.batch_index,
        template_digest: session.template_digest.clone(),
        parent_candidate_model_digest: session.latest_candidate_model_digest.clone(),
        evidence_batch_digest: prepared.batch.evidence_batch_digest.clone(),
        observation_manifest_digest: prepared.batch.observation_manifest_digest.clone(),
        program_digest: prepared.candidate.receipt.program_digest.clone(),
        candidate_snapshot_digest: prepared.candidate.receipt.candidate_snapshot_digest.clone(),
        candidate_native_model_digest: candidate_native_model_digest.to_owned(),
        verifier_key_id: attestation.verifier_key_id.clone(),
        verification_protocol_digest: claims.verification_protocol_digest.clone(),
        result_manifest_digest: claims.result_manifest_digest.clone(),
        verifier_receipt_digest,
        precondition_gate_digest: claims.precondition_gate_digest.clone(),
        statistical_verification_report_digest: claims
            .statistical_verification_report_digest
            .clone(),
        family_guardrail_report_digest: claims.family_guardrail_report_digest.clone(),
        combined_campaign_alpha_budget_bits: claims.combined_campaign_alpha_budget_bits,
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
        || template
            .fault_hypothesis_ids
            .iter()
            .any(|id| id.trim().is_empty())
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
        return Err(SequentialWeightProgramError::TemplateMismatch(
            "architecture",
        ));
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
        return Err(SequentialWeightProgramError::TemplateMismatch(
            "session/model binding",
        ));
    }
    Ok(())
}

fn template_digest(
    template: &SequentialWeightProgramTemplate,
) -> Result<String, SequentialWeightProgramError> {
    hash_value(b"symthaea-sequential-weight-template-v1\0", template)
}

fn session_digest(session: &SequentialEditSession) -> Result<String, SequentialWeightProgramError> {
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
    hash_value(b"symthaea-sequential-step-receipt-v3\0", &unsigned)
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
        return Err(SequentialWeightProgramError::InvalidDigest(
            "sha256/blake3 digest",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_space_program::{
        CfCWeightGroupId, ParameterGroupConstraints, ParameterGroupRole, ParameterGroupSnapshot,
        WeightProgramTarget,
    };
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn statistical_evidence_binding(
        session: &SequentialEditSession,
        candidate_native_model_digest: &str,
    ) -> SequentialVerifierEvidenceBinding {
        use crate::sequential_weight_verification::{
            ConfirmatoryEpisodeManifest, PairedLossObservation,
            SequentialWeightVerificationCampaign, SequentialWeightVerificationConfig,
        };
        use crate::family_guardrail::{
            FamilyGuardrailManifest, FamilyGuardrailSpec, FamilyPairedLossObservation,
        };
        let episode_ids = (0..256)
            .map(|index| format!("bearing-{index:04}"))
            .chain((0..256).map(|index| format!("gearbox-{index:04}")))
            .collect::<Vec<_>>();
        let confirmatory_episode_manifest =
            ConfirmatoryEpisodeManifest::new(episode_ids.clone())
                .expect("valid confirmatory episode manifest");
        let confirmatory_manifest_digest = confirmatory_episode_manifest.manifest_digest.clone();

        let mut campaign = SequentialWeightVerificationCampaign::new(
            SequentialWeightVerificationConfig {
                campaign_alpha: 0.025,
                minimum_improvement: 0.05,
                minimum_observations: 2,
                maximum_observations: 512,
            },
            digest('8'),
        )
        .expect("valid statistical campaign");
        let mut verifier = campaign
            .start_candidate(
                session.latest_candidate_model_digest.clone(),
                candidate_native_model_digest.to_owned(),
                digest('6'),
                confirmatory_episode_manifest,
                digest('5'),
            )
            .expect("fresh statistical candidate");
        let mut report = verifier.report().expect("initial statistical report");
        for episode_id in &episode_ids {
            report = verifier
                .observe(PairedLossObservation {
                    episode_id: episode_id.clone(),
                    baseline_loss: 0.9,
                    candidate_loss: 0.5,
                })
                .expect("paired confirmatory observation");
            if report.status == crate::sequential_weight_verification::SequentialWeightVerificationStatus::BenefitEstablished {
                break;
            }
        }

        let family_manifest = FamilyGuardrailManifest::new(
            report.binding.candidate_index,
            session.latest_candidate_model_digest.clone(),
            candidate_native_model_digest.to_owned(),
            report.binding.evaluation_protocol_digest.clone(),
            confirmatory_manifest_digest,
            0.025,
            vec![
                FamilyGuardrailSpec {
                    family_id: "bearing".into(),
                    noninferiority_margin: 0.2,
                    minimum_observations: 256,
                    maximum_observations: 256,
                    episode_ids: (0..256)
                        .map(|index| format!("bearing-{index:04}"))
                        .collect(),
                },
                FamilyGuardrailSpec {
                    family_id: "gearbox".into(),
                    noninferiority_margin: 0.2,
                    minimum_observations: 256,
                    maximum_observations: 256,
                    episode_ids: (0..256)
                        .map(|index| format!("gearbox-{index:04}"))
                        .collect(),
                },
            ],
        )
        .expect("valid family guardrail manifest");
        let mut family_verifier =
            FamilyGuardrailVerifier::new(family_manifest).expect("family verifier");
        for (family_id, prefix) in [("bearing", "bearing"), ("gearbox", "gearbox")] {
            for index in 0..256 {
                family_verifier
                    .observe(FamilyPairedLossObservation {
                        family_id: family_id.to_owned(),
                        episode_id: format!("{prefix}-{index:04}"),
                        baseline_loss: 0.9,
                        candidate_loss: 0.5,
                    })
                    .expect("family paired confirmatory observation");
            }
        }

        bind_sequential_verification_evidence(
            session,
            candidate_native_model_digest,
            &report,
            &family_verifier,
            0.05,
        )
        .expect("aggregate and family reports bound to sequential edit candidate")
    }

    fn signed_attestation(
        session: &SequentialEditSession,
        prepared: &PreparedSequentialCandidate,
        candidate_native_model_digest: &str,
        evidence_binding: &SequentialVerifierEvidenceBinding,
        accepted: bool,
        signing_key: &SigningKey,
    ) -> SignedSequentialVerifierAttestation {
        let claims = SequentialVerifierAttestationClaims {
            format_version: 3,
            verifier_id: "independent-test-verifier".into(),
            verification_protocol_digest: digest('8'),
            result_manifest_digest: digest('7'),
            session_id: session.session_id.clone(),
            batch_index: prepared.batch.batch_index,
            template_digest: prepared.template_digest.clone(),
            parent_candidate_model_digest: session.latest_candidate_model_digest.clone(),
            candidate_native_model_digest: candidate_native_model_digest.to_owned(),
            target_specification_digest: session.target_specification_digest.clone(),
            evidence_batch_digest: prepared.batch.evidence_batch_digest.clone(),
            observation_manifest_digest: prepared.batch.observation_manifest_digest.clone(),
            program_digest: prepared.candidate.receipt.program_digest.clone(),
            candidate_snapshot_digest: prepared.candidate.receipt.candidate_snapshot_digest.clone(),
            precondition_gate_digest: evidence_binding.precondition_gate_digest().to_owned(),
            statistical_verification_report_digest: evidence_binding
                .statistical_verification_report_digest()
                .to_owned(),
            family_guardrail_report_digest: evidence_binding
                .family_guardrail_report_digest()
                .to_owned(),
            combined_campaign_alpha_budget_bits: evidence_binding
                .combined_campaign_alpha_budget_bits(),
            accepted,
        };
        let message =
            sequential_verifier_attestation_message(&claims).expect("attestation message");
        let signature = signing_key.sign(&message).to_bytes().to_vec();
        SignedSequentialVerifierAttestation {
            claims,
            verifier_key_id: sequential_verifier_key_id(&signing_key.verifying_key()),
            signature,
        }
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

    fn batch(
        session_id: &str,
        index: usize,
        parent: &str,
        evidence: char,
    ) -> SequentialEvidenceBatch {
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
        let session =
            begin_sequential_edit_session(session_id.clone(), &template, &base).expect("begin");
        let prepared = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare");
        assert_eq!(
            session.next_batch_index, 0,
            "preparation is not advancement"
        );
        assert_eq!(prepared.candidate.receipt.changed_parameter_count, 2);
        assert!(prepared.candidate.receipt.total_l2_delta > 0.0);
        let signing_key = SigningKey::from_bytes(&[42_u8; 32]);
        let pinned_key = signing_key.verifying_key();
        let evidence_binding = statistical_evidence_binding(&session, &digest('9'));
        let rejected = signed_attestation(
            &session,
            &prepared,
            &digest('9'),
            &evidence_binding,
            false,
            &signing_key,
        );
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &rejected,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::VerifierRejectedCandidate)
        ));
        let attestation = signed_attestation(
            &session,
            &prepared,
            &digest('9'),
            &evidence_binding,
            true,
            &signing_key,
        );
        let (advanced, receipt) = accept_verified_sequential_candidate(
            &session,
            &template,
            &prepared,
            &digest('9'),
            &evidence_binding,
            &attestation,
            &pinned_key,
        )
        .expect("authenticated verifier attestation advances");
        assert_eq!(
            receipt.verifier_key_id,
            sequential_verifier_key_id(&pinned_key)
        );
        assert_eq!(receipt.verification_protocol_digest, digest('8'));
        assert_eq!(receipt.result_manifest_digest, digest('7'));
        assert_eq!(receipt.precondition_gate_digest, digest('5'));
        assert!(validate_digest(&receipt.statistical_verification_report_digest).is_ok());
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
        let session =
            begin_sequential_edit_session(session_id.clone(), &template, &base).expect("begin");
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
        let signing_key = SigningKey::from_bytes(&[43_u8; 32]);
        let pinned_key = signing_key.verifying_key();
        let evidence_binding = statistical_evidence_binding(&session, &digest('9'));
        let attestation = signed_attestation(
            &session,
            &modified,
            &digest('9'),
            &evidence_binding,
            true,
            &signing_key,
        );
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &modified,
                &digest('9'),
                &evidence_binding,
                &attestation,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::StalePreparedCandidate)
        ));
    }

    #[test]
    fn cumulative_l2_path_budget_is_enforced_across_accepted_batches() {
        let base = snapshot();
        let template = template();
        let session_id = digest('f');
        let session =
            begin_sequential_edit_session(session_id.clone(), &template, &base).expect("begin");
        let prepared0 = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare batch zero");
        let signing_key = SigningKey::from_bytes(&[44_u8; 32]);
        let pinned_key = signing_key.verifying_key();
        let evidence_binding0 = statistical_evidence_binding(&session, &digest('9'));
        let attestation0 = signed_attestation(
            &session,
            &prepared0,
            &digest('9'),
            &evidence_binding0,
            true,
            &signing_key,
        );
        let (advanced, _) = accept_verified_sequential_candidate(
            &session,
            &template,
            &prepared0,
            &digest('9'),
            &evidence_binding0,
            &attestation0,
            &pinned_key,
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
        let evidence_binding1 = statistical_evidence_binding(&advanced, &digest('7'));
        let attestation1 = signed_attestation(
            &advanced,
            &prepared1,
            &digest('7'),
            &evidence_binding1,
            true,
            &signing_key,
        );
        assert!(matches!(
            accept_verified_sequential_candidate(
                &advanced,
                &template,
                &prepared1,
                &digest('7'),
                &evidence_binding1,
                &attestation1,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::CumulativeL2BudgetExceeded)
        ));
    }

    #[test]
    fn verifier_attestation_is_bound_to_exact_candidate_and_trusted_key() {
        let base = snapshot();
        let template = template();
        let session_id = digest('f');
        let session =
            begin_sequential_edit_session(session_id.clone(), &template, &base).expect("begin");
        let prepared = prepare_sequential_candidate(
            &session,
            &template,
            &base,
            &batch(&session_id, 0, &base.model_digest, '1'),
        )
        .expect("prepare");
        let signing_key = SigningKey::from_bytes(&[45_u8; 32]);
        let pinned_key = signing_key.verifying_key();
        let evidence_binding = statistical_evidence_binding(&session, &digest('9'));
        let wrong_candidate_binding = statistical_evidence_binding(&session, &digest('7'));
        let correctly_bound = signed_attestation(
            &session,
            &prepared,
            &digest('9'),
            &evidence_binding,
            true,
            &signing_key,
        );
        let wrong_candidate = signed_attestation(
            &session,
            &prepared,
            &digest('7'),
            &wrong_candidate_binding,
            true,
            &signing_key,
        );
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &wrong_candidate,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                "candidate_native_model_digest"
            ))
        ));

        let mut tampered = correctly_bound.clone();
        tampered.claims.result_manifest_digest = digest('6');
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &tampered,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::InvalidVerifierSignature)
        ));

        let mut wrong_statistical_report = correctly_bound.clone();
        wrong_statistical_report
            .claims
            .statistical_verification_report_digest = digest('4');
        let message = sequential_verifier_attestation_message(&wrong_statistical_report.claims)
            .expect("message for re-signed mismatched statistical report");
        wrong_statistical_report.signature = signing_key.sign(&message).to_bytes().to_vec();
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &wrong_statistical_report,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                "statistical_verification_report_digest"
            ))
        ));

        let mut wrong_family_report = correctly_bound.clone();
        wrong_family_report.claims.family_guardrail_report_digest = digest('2');
        let message = sequential_verifier_attestation_message(&wrong_family_report.claims)
            .expect("message for re-signed mismatched family guardrail report");
        wrong_family_report.signature = signing_key.sign(&message).to_bytes().to_vec();
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &wrong_family_report,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                "family_guardrail_report_digest"
            ))
        ));

        let mut wrong_precondition = correctly_bound.clone();
        wrong_precondition.claims.precondition_gate_digest = digest('3');
        let message = sequential_verifier_attestation_message(&wrong_precondition.claims)
            .expect("message for re-signed mismatched precondition");
        wrong_precondition.signature = signing_key.sign(&message).to_bytes().to_vec();
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &wrong_precondition,
                &pinned_key,
            ),
            Err(SequentialWeightProgramError::VerifierAttestationMismatch(
                "precondition_gate_digest"
            ))
        ));

        let untrusted_key = SigningKey::from_bytes(&[46_u8; 32]).verifying_key();
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &prepared,
                &digest('9'),
                &evidence_binding,
                &correctly_bound,
                &untrusted_key,
            ),
            Err(SequentialWeightProgramError::VerifierKeyIdMismatch)
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
