// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Predictive, budgeted memory surfacing.
//!
//! A future-use prediction only chooses which already-admitted evidence to
//! surface in a context window. It cannot promote evidence, alter trust,
//! authorize an action, delete memory, or override lifecycle controls.

use broca_mace_ingress::{AsKnownQuery, ConsolidationEvidenceRefV1, SourceTrustStatus};
use broca_mace_policy::Scope;
use sha2::{Digest, Sha256};

pub const MAX_CONTEXT_BUDGET_TOKENS: u32 = 16_384;
pub const MAX_PREDICTED_VALUE_MICROS: u32 = 1_000_000;
pub const REUSE_PROBABILITY_SCALE: u32 = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemorySurfacingModeV1 {
    FailureOnly,
    RecencyFirst,
    FrequencyFirst,
    PredictedFutureValue,
}

impl MemorySurfacingModeV1 {
    fn code(self) -> &'static str {
        match self {
            Self::FailureOnly => "failure_only",
            Self::RecencyFirst => "recency_first",
            Self::FrequencyFirst => "frequency_first",
            Self::PredictedFutureValue => "predicted_future_value",
        }
    }
}

/// Model-proposed prediction that an evidence-linked memory will be useful
/// within a future task horizon. All fields are proposals, not trust claims.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FutureUsePredictionV1 {
    pub event_id: u32,
    /// Identity of the agent model whose future memory utility is predicted.
    pub target_model_sha256: String,
    pub target_scope: Scope,
    pub predicted_at_ms: i64,
    pub horizon_end_ms: i64,
    pub reuse_probability_basis_points: u16,
    pub expected_value_gain_micros: u32,
    pub uncertainty_penalty_micros: u32,
    pub model_id: String,
    pub model_version: String,
    pub model_config_sha256: String,
    pub rationale_sha256: String,
    pub evidence_sha256: String,
    pub prediction_sha256: String,
}

pub const MIN_SALIENCE_TRAIN_CLUSTERS: u32 = 32;
pub const MIN_SALIENCE_CALIBRATION_CLUSTERS: u32 = 16;
pub const MIN_SALIENCE_HELDOUT_CLUSTERS: u32 = 16;
pub const MAX_SALIENCE_ECE_MICROS: u32 = 50_000;
pub const MAX_SALIENCE_GROUP_ECE_MICROS: u32 = 100_000;

/// Independent calibration evidence for a future-use prediction. The receipt
/// must be issued by a verifier that checks disjoint task-family/source-root
/// splits and the frozen calibration protocol; a self-hash is not a signature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SalienceCalibrationReceiptV1 {
    pub prediction_sha256: String,
    pub evidence_sha256: String,
    pub target_model_sha256: String,
    pub target_scope: Scope,
    pub as_known: AsKnownQuery,
    pub model_id: String,
    pub model_version: String,
    pub model_config_sha256: String,
    pub training_dataset_sha256: String,
    pub calibration_dataset_sha256: String,
    pub heldout_evaluation_sha256: String,
    pub task_family_split_sha256: String,
    pub source_root_split_sha256: String,
    pub training_cluster_count: u32,
    pub calibration_cluster_count: u32,
    pub heldout_cluster_count: u32,
    pub brier_score_micros: u32,
    pub base_rate_brier_score_micros: u32,
    pub expected_calibration_error_micros: u32,
    pub worst_group_calibration_error_micros: u32,
    pub task_families_disjoint: bool,
    pub source_roots_disjoint: bool,
    pub verified_at_ms: i64,
    pub valid_until_ms: i64,
    pub verifier_id: String,
    pub receipt_sha256: String,
}

impl SalienceCalibrationReceiptV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.prediction_sha256.clone(),
            self.evidence_sha256.clone(),
            self.target_model_sha256.clone(),
            scope_code(self.target_scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.model_id.clone(),
            self.model_version.clone(),
            self.model_config_sha256.clone(),
            self.training_dataset_sha256.clone(),
            self.calibration_dataset_sha256.clone(),
            self.heldout_evaluation_sha256.clone(),
            self.task_family_split_sha256.clone(),
            self.source_root_split_sha256.clone(),
            self.training_cluster_count.to_string(),
            self.calibration_cluster_count.to_string(),
            self.heldout_cluster_count.to_string(),
            self.brier_score_micros.to_string(),
            self.base_rate_brier_score_micros.to_string(),
            self.expected_calibration_error_micros.to_string(),
            self.worst_group_calibration_error_micros.to_string(),
            self.task_families_disjoint.to_string(),
            self.source_roots_disjoint.to_string(),
            self.verified_at_ms.to_string(),
            self.valid_until_ms.to_string(),
            self.verifier_id.clone(),
        ];
        digest_fields(b"mycelix:mace:salience-calibration-receipt:v1\\0", &fields)
    }
}

/// Production implementations must verify the issuer/signature, frozen
/// evaluation protocol, held-out split lineage and metric calculations using
/// Mycelix/SAM. Test implementations are synthetic fixtures only.
pub trait SaliencePredictionVerifierV1 {
    fn verify_prediction(
        &self,
        prediction: &FutureUsePredictionV1,
        target_scope: Scope,
        as_known: AsKnownQuery,
    ) -> Option<SalienceCalibrationReceiptV1>;
}

/// Opaque prediction that has passed the independent calibration gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedFutureUsePredictionV1 {
    prediction: FutureUsePredictionV1,
    calibration: SalienceCalibrationReceiptV1,
}

impl VerifiedFutureUsePredictionV1 {
    pub fn prediction(&self) -> &FutureUsePredictionV1 {
        &self.prediction
    }

    pub fn calibration_receipt(&self) -> &SalienceCalibrationReceiptV1 {
        &self.calibration
    }

    pub fn target_model_sha256(&self) -> &str {
        &self.prediction.target_model_sha256
    }
}

fn validate_calibration_receipt(
    prediction: &FutureUsePredictionV1,
    receipt: &SalienceCalibrationReceiptV1,
    target_scope: Scope,
    as_known: AsKnownQuery,
) -> Result<(), SalienceReject> {
    if receipt.target_model_sha256 != prediction.target_model_sha256 {
        return Err(SalienceReject::PredictionTargetModelMismatch);
    }
    if receipt.prediction_sha256 != prediction.prediction_sha256
        || receipt.evidence_sha256 != prediction.evidence_sha256
        || receipt.model_id != prediction.model_id
        || receipt.model_version != prediction.model_version
        || receipt.model_config_sha256 != prediction.model_config_sha256
    {
        return Err(SalienceReject::PredictionCalibrationMismatch);
    }
    if receipt.target_scope != target_scope
        || prediction.target_scope != target_scope
    {
        return Err(SalienceReject::PredictionScopeMismatch);
    }
    if receipt.as_known != as_known {
        return Err(SalienceReject::PredictionCalibrationSnapshotMismatch);
    }

    let digests = [
        &receipt.prediction_sha256,
        &receipt.evidence_sha256,
        &receipt.model_config_sha256,
        &receipt.training_dataset_sha256,
        &receipt.calibration_dataset_sha256,
        &receipt.heldout_evaluation_sha256,
        &receipt.task_family_split_sha256,
        &receipt.source_root_split_sha256,
        &receipt.receipt_sha256,
    ];
    if digests.iter().any(|digest| !is_lower_sha256(digest)) {
        return Err(SalienceReject::InvalidCalibrationDigest);
    }
    if receipt.training_dataset_sha256 == receipt.calibration_dataset_sha256
        || receipt.training_dataset_sha256 == receipt.heldout_evaluation_sha256
        || receipt.calibration_dataset_sha256 == receipt.heldout_evaluation_sha256
    {
        return Err(SalienceReject::CalibrationDatasetsOverlap);
    }
    if !receipt.task_families_disjoint || !receipt.source_roots_disjoint {
        return Err(SalienceReject::CalibrationSplitNotIndependent);
    }
    if receipt.training_cluster_count < MIN_SALIENCE_TRAIN_CLUSTERS
        || receipt.calibration_cluster_count < MIN_SALIENCE_CALIBRATION_CLUSTERS
        || receipt.heldout_cluster_count < MIN_SALIENCE_HELDOUT_CLUSTERS
    {
        return Err(SalienceReject::CalibrationEvidenceInsufficient);
    }
    if receipt.brier_score_micros >= receipt.base_rate_brier_score_micros
        || receipt.expected_calibration_error_micros > MAX_SALIENCE_ECE_MICROS
        || receipt.worst_group_calibration_error_micros
            > MAX_SALIENCE_GROUP_ECE_MICROS
    {
        return Err(SalienceReject::CalibrationQualityInsufficient);
    }
    if receipt.verifier_id.trim().is_empty()
        || receipt.receipt_sha256 != receipt.canonical_digest()
    {
        return Err(SalienceReject::InvalidCalibrationReceipt);
    }
    if receipt.verified_at_ms > as_known.known_at_ms
        || receipt.valid_until_ms <= as_known.known_at_ms
    {
        return Err(SalienceReject::CalibrationNotValidAtSnapshot);
    }
    Ok(())
}

/// Convert a raw model forecast into an eligible prediction only after an
/// independent calibration verifier attests its model, split and metrics.
pub fn verify_future_use_prediction(
    prediction: FutureUsePredictionV1,
    target_scope: Scope,
    as_known: AsKnownQuery,
    verifier: &impl SaliencePredictionVerifierV1,
) -> Result<VerifiedFutureUsePredictionV1, SalienceReject> {
    if prediction.event_id == 0 {
        return Err(SalienceReject::InvalidEventId);
    }
    if prediction.target_scope != target_scope {
        return Err(SalienceReject::PredictionScopeMismatch);
    }
    if prediction.predicted_at_ms > as_known.known_at_ms {
        return Err(SalienceReject::PredictionFutureLeakage);
    }
    if prediction.horizon_end_ms <= as_known.known_at_ms
        || prediction.predicted_at_ms >= prediction.horizon_end_ms
    {
        return Err(SalienceReject::InvalidPredictionHorizon);
    }
    if u32::from(prediction.reuse_probability_basis_points) > REUSE_PROBABILITY_SCALE {
        return Err(SalienceReject::InvalidReuseProbability);
    }
    if prediction.expected_value_gain_micros > MAX_PREDICTED_VALUE_MICROS
        || prediction.uncertainty_penalty_micros > MAX_PREDICTED_VALUE_MICROS
    {
        return Err(SalienceReject::InvalidPredictedValue);
    }
    if prediction.model_id.trim().is_empty() || prediction.model_version.trim().is_empty() {
        return Err(SalienceReject::MissingModelIdentity);
    }
    if !is_lower_sha256(&prediction.evidence_sha256)
        || !is_lower_sha256(&prediction.target_model_sha256)
        || !is_lower_sha256(&prediction.model_config_sha256)
        || !is_lower_sha256(&prediction.rationale_sha256)
        || !is_lower_sha256(&prediction.prediction_sha256)
        || prediction.prediction_sha256 != prediction.canonical_digest()
    {
        return Err(SalienceReject::InvalidPredictionDigest);
    }
    let calibration = verifier
        .verify_prediction(&prediction, target_scope, as_known)
        .ok_or(SalienceReject::PredictionCalibrationMissing)?;
    validate_calibration_receipt(&prediction, &calibration, target_scope, as_known)?;
    Ok(VerifiedFutureUsePredictionV1 {
        prediction,
        calibration,
    })
}

impl FutureUsePredictionV1 {
    pub fn new(
        event_id: u32,
        target_model_sha256: impl Into<String>,
        target_scope: Scope,
        predicted_at_ms: i64,
        horizon_end_ms: i64,
        reuse_probability_basis_points: u16,
        expected_value_gain_micros: u32,
        uncertainty_penalty_micros: u32,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        model_config_sha256: impl Into<String>,
        rationale_sha256: impl Into<String>,
        evidence_sha256: impl Into<String>,
    ) -> Self {
        let mut prediction = Self {
            event_id,
            target_model_sha256: target_model_sha256.into(),
            target_scope,
            predicted_at_ms,
            horizon_end_ms,
            reuse_probability_basis_points,
            expected_value_gain_micros,
            uncertainty_penalty_micros,
            model_id: model_id.into(),
            model_version: model_version.into(),
            model_config_sha256: model_config_sha256.into(),
            rationale_sha256: rationale_sha256.into(),
            evidence_sha256: evidence_sha256.into(),
            prediction_sha256: String::new(),
        };
        prediction.prediction_sha256 = prediction.canonical_digest();
        prediction
    }

    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.event_id.to_string(),
            self.target_model_sha256.clone(),
            scope_code(self.target_scope).to_owned(),
            self.predicted_at_ms.to_string(),
            self.horizon_end_ms.to_string(),
            self.reuse_probability_basis_points.to_string(),
            self.expected_value_gain_micros.to_string(),
            self.uncertainty_penalty_micros.to_string(),
            self.model_id.clone(),
            self.model_version.clone(),
            self.model_config_sha256.clone(),
            self.rationale_sha256.clone(),
            self.evidence_sha256.clone(),
        ];
        digest_fields(b"mycelix:mace:future-use-prediction:v1\0", &fields)
    }

    /// Expected value is deliberately integer-only and deterministic.
    pub fn expected_net_value_micros(&self) -> i64 {
        let gross = u64::from(self.reuse_probability_basis_points)
            * u64::from(self.expected_value_gain_micros)
            / u64::from(REUSE_PROBABILITY_SCALE);
        i64::try_from(gross).unwrap_or(i64::MAX)
            - i64::from(self.uncertainty_penalty_micros)
    }
}

/// Candidate must already be present in a verified evidence pack. The source
/// digest is retained in the surfacing plan; salience never re-verifies it.
/// Access telemetry is a retrieval hint, never an evidence or trust claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryAccessTelemetryV1 {
    pub context_token_cost: u32,
    pub last_accessed_at_ms: i64,
    pub historical_access_count: u32,
}

/// Opaque candidate constructed from an admitted, receipt-preserving evidence
/// reference. Callers cannot manufacture an eligible memory by filling fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySurfacingCandidateV1 {
    event_id: u32,
    scope: Scope,
    as_known: AsKnownQuery,
    context_token_cost: u32,
    last_accessed_at_ms: i64,
    historical_access_count: u32,
    observed_failure: bool,
    evidence_sha256: String,
    prediction: Option<VerifiedFutureUsePredictionV1>,
}

impl MemorySurfacingCandidateV1 {
    pub fn from_verified_evidence(
        evidence: &ConsolidationEvidenceRefV1,
        access: MemoryAccessTelemetryV1,
        prediction: Option<VerifiedFutureUsePredictionV1>,
        as_known: AsKnownQuery,
    ) -> Result<Self, SalienceReject> {
        if evidence.event_id() == 0 {
            return Err(SalienceReject::InvalidEventId);
        }
        if evidence.source_trust_status() != SourceTrustStatus::TrustedForScope {
            return Err(SalienceReject::EvidenceNotTrusted);
        }
        if !evidence.outcome_reconciled() {
            return Err(SalienceReject::EvidenceOutcomeUnreconciled);
        }
        if evidence.source_family() == 0
            || !evidence.source_root_digest().is_known()
        {
            return Err(SalienceReject::EvidenceRootUnknown);
        }
        let error_bound = evidence.measurement_error_bound();
        if !error_bound.is_finite() || !(0.0..=1.0).contains(&error_bound) {
            return Err(SalienceReject::InvalidEvidenceMeasurement);
        }
        if access.context_token_cost == 0 {
            return Err(SalienceReject::InvalidContextCost);
        }
        if evidence.as_known() != as_known {
            return Err(SalienceReject::EvidenceSnapshotMismatch);
        }
        if access.last_accessed_at_ms > as_known.known_at_ms {
            return Err(SalienceReject::FutureAccessTimestamp);
        }

        let evidence_sha256 = receipt_bound_evidence_digest(evidence);
        if let Some(value) = &prediction {
            let raw = value.prediction();
            if raw.event_id != evidence.event_id() {
                return Err(SalienceReject::PredictionEventMismatch);
            }
            if raw.evidence_sha256 != evidence_sha256 {
                return Err(SalienceReject::PredictionEvidenceMismatch);
            }
            validate_calibration_receipt(
                raw,
                value.calibration_receipt(),
                evidence.scope(),
                as_known,
            )?;
        }

        Ok(Self {
            event_id: evidence.event_id(),
            scope: evidence.scope(),
            as_known,
            context_token_cost: access.context_token_cost,
            last_accessed_at_ms: access.last_accessed_at_ms,
            historical_access_count: access.historical_access_count,
            observed_failure: !evidence.outcome_success(),
            evidence_sha256,
            prediction,
        })
    }

    pub fn event_id(&self) -> u32 {
        self.event_id
    }

    pub fn target_model_sha256(&self) -> Option<&str> {
        self.prediction
            .as_ref()
            .map(VerifiedFutureUsePredictionV1::target_model_sha256)
    }

    pub fn evidence_sha256(&self) -> &str {
        &self.evidence_sha256
    }
}

/// Return the receipt-bound evidence digest required by future-use
/// predictions. This cannot admit or manufacture an evidence reference.
pub fn future_use_prediction_evidence_digest(
    evidence: &ConsolidationEvidenceRefV1,
) -> String {
    receipt_bound_evidence_digest(evidence)
}

fn receipt_bound_evidence_digest(evidence: &ConsolidationEvidenceRefV1) -> String {
    let receipt = evidence.receipts();
    let fields = vec![
        evidence.as_known().valid_at_ms.to_string(),
        evidence.as_known().known_at_ms.to_string(),
        evidence.event_id().to_string(),
        scope_code(evidence.scope()).to_owned(),
        evidence.session_id().to_owned(),
        evidence.task_id().to_owned(),
        evidence.action().name().to_owned(),
        evidence.observed_utility().to_bits().to_string(),
        evidence.measurement_error_bound().to_bits().to_string(),
        evidence.source_family().to_string(),
        evidence.source_root_id().to_string(),
        evidence.source_root_digest().to_hex(),
        receipt.raw_event_digest.clone(),
        receipt.raw_event_receipt_digest.clone(),
        receipt.canonical_event_hash.clone(),
        receipt.authority_receipt_digest.clone(),
        receipt.event_occurred_at_ms.to_string(),
        receipt.authority_received_at_ms.to_string(),
        receipt.lineage_digest.clone(),
        receipt.source_acquisition_mode.clone(),
        receipt.upstream_source_id.clone(),
        receipt.source_artifact_digest.clone(),
        receipt.transformation_digest.clone(),
        receipt.source_issuer_id.clone(),
        receipt.source_trust_policy_digest.clone(),
        receipt.calibration_digest.clone(),
        receipt.measurement_unit.clone(),
        receipt.measurement_instrument_id.clone(),
        receipt.calibration_method_id.clone(),
        receipt.calibration_revision.clone(),
        receipt.causal_assessment_digest.clone(),
        receipt.outcome_success.to_string(),
        receipt.outcome_observed_utility_bits.to_string(),
        receipt.outcome_utility_unit.clone(),
        receipt.outcome_digest.clone(),
    ];
    digest_fields(b"mycelix:mace:verified-memory-evidence:v1\\0", &fields)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SalienceReject {
    InvalidBudget,
    EmptyCandidateSet,
    DuplicateEventId,
    InvalidEventId,
    InvalidContextCost,
    InvalidEvidenceDigest,
    FutureAccessTimestamp,
    PredictionEventMismatch,
    PredictionEvidenceMismatch,
    PredictionScopeMismatch,
    PredictionFutureLeakage,
    InvalidPredictionHorizon,
    InvalidReuseProbability,
    InvalidPredictedValue,
    MissingModelIdentity,
    InvalidModelDigest,
    InvalidRationaleDigest,
    InvalidPredictionDigest,
    PredictionCalibrationMissing,
    PredictionCalibrationMismatch,
    PredictionTargetModelMismatch,
    PredictionCalibrationSnapshotMismatch,
    InvalidCalibrationDigest,
    CalibrationDatasetsOverlap,
    CalibrationSplitNotIndependent,
    CalibrationEvidenceInsufficient,
    CalibrationQualityInsufficient,
    InvalidCalibrationReceipt,
    CalibrationNotValidAtSnapshot,
    EvidenceNotTrusted,
    EvidenceOutcomeUnreconciled,
    EvidenceScopeMismatch,
    EvidenceSnapshotMismatch,
    EvidenceRootUnknown,
    InvalidEvidenceMeasurement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfacedMemoryV1 {
    pub event_id: u32,
    pub context_token_cost: u32,
    pub evidence_sha256: String,
    pub prediction_sha256: Option<String>,
    pub calibration_receipt_sha256: Option<String>,
    pub expected_net_value_micros: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySurfacingPlanV1 {
    pub schema_version: u16,
    pub mode: MemorySurfacingModeV1,
    pub target_scope: Scope,
    pub as_known: AsKnownQuery,
    pub context_budget_tokens: u32,
    pub context_tokens_used: u32,
    pub predicted_net_value_micros: i64,
    pub surfaced: Vec<SurfacedMemoryV1>,
    pub plan_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn scope_code(scope: Scope) -> &'static str {
    match (scope.regime, scope.objective) {
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Quality,
        ) => "baseline:quality",
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Cost,
        ) => "baseline:cost",
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Speed,
        ) => "baseline:speed",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Quality,
        ) => "shifted:quality",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Cost,
        ) => "shifted:cost",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Speed,
        ) => "shifted:speed",
    }
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_candidate(
    candidate: &MemorySurfacingCandidateV1,
    target_scope: Scope,
    as_known: AsKnownQuery,
) -> Result<(), SalienceReject> {
    if candidate.event_id == 0 {
        return Err(SalienceReject::InvalidEventId);
    }
    if candidate.scope != target_scope {
        return Err(SalienceReject::EvidenceScopeMismatch);
    }
    if candidate.as_known != as_known {
        return Err(SalienceReject::EvidenceSnapshotMismatch);
    }
    if candidate.context_token_cost == 0 {
        return Err(SalienceReject::InvalidContextCost);
    }
    if !is_lower_sha256(&candidate.evidence_sha256) {
        return Err(SalienceReject::InvalidEvidenceDigest);
    }
    if candidate.last_accessed_at_ms > as_known.known_at_ms {
        return Err(SalienceReject::FutureAccessTimestamp);
    }
    if let Some(verified) = &candidate.prediction {
        let prediction = verified.prediction();
        if prediction.event_id != candidate.event_id {
            return Err(SalienceReject::PredictionEventMismatch);
        }
        if prediction.evidence_sha256 != candidate.evidence_sha256 {
            return Err(SalienceReject::PredictionEvidenceMismatch);
        }
        validate_calibration_receipt(
            prediction,
            verified.calibration_receipt(),
            target_scope,
            as_known,
        )?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct KnapsackState {
    value_micros: i64,
    selected_indices: Vec<usize>,
    used_tokens: u32,
}

fn state_is_better(candidate: &KnapsackState, current: &KnapsackState) -> bool {
    if candidate.value_micros != current.value_micros {
        return candidate.value_micros > current.value_micros;
    }
    if candidate.used_tokens != current.used_tokens {
        return candidate.used_tokens < current.used_tokens;
    }
    candidate.selected_indices < current.selected_indices
}

/// Select a bounded set of evidence references for context. This function
/// returns IDs/digests only: it does not mutate memory or grant authority.
pub fn select_for_context(
    candidates: &[MemorySurfacingCandidateV1],
    mode: MemorySurfacingModeV1,
    target_scope: Scope,
    as_known: AsKnownQuery,
    context_budget_tokens: u32,
) -> Result<MemorySurfacingPlanV1, SalienceReject> {
    if context_budget_tokens == 0 || context_budget_tokens > MAX_CONTEXT_BUDGET_TOKENS {
        return Err(SalienceReject::InvalidBudget);
    }
    if candidates.is_empty() {
        return Err(SalienceReject::EmptyCandidateSet);
    }

    let mut seen = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.event_id) {
            return Err(SalienceReject::DuplicateEventId);
        }
        validate_candidate(candidate, target_scope, as_known)?;
    }

    let mut selected_indices = Vec::<usize>::new();
    if mode == MemorySurfacingModeV1::PredictedFutureValue {
        let width = usize::try_from(context_budget_tokens)
            .map_err(|_| SalienceReject::InvalidBudget)?;
        let mut states: Vec<Option<KnapsackState>> = vec![None; width + 1];
        states[0] = Some(KnapsackState {
            value_micros: 0,
            selected_indices: Vec::new(),
            used_tokens: 0,
        });

        for (index, candidate) in candidates.iter().enumerate() {
            let Some(verified) = &candidate.prediction else {
                continue;
            };
            let value = verified.prediction().expected_net_value_micros();
            if value <= 0 || candidate.context_token_cost > context_budget_tokens {
                continue;
            }
            let cost = usize::try_from(candidate.context_token_cost)
                .map_err(|_| SalienceReject::InvalidContextCost)?;
            for capacity in (cost..=width).rev() {
                let Some(previous) = states[capacity - cost].clone() else {
                    continue;
                };
                let mut next = previous;
                next.value_micros = next.value_micros.saturating_add(value);
                next.used_tokens = next
                    .used_tokens
                    .saturating_add(candidate.context_token_cost);
                next.selected_indices.push(index);
                match &states[capacity] {
                    Some(current) if !state_is_better(&next, current) => {}
                    _ => states[capacity] = Some(next),
                }
            }
        }

        let best = states
            .into_iter()
            .flatten()
            .reduce(|best, current| {
                if state_is_better(&current, &best) {
                    current
                } else {
                    best
                }
            })
            .unwrap_or(KnapsackState {
                value_micros: 0,
                selected_indices: Vec::new(),
                used_tokens: 0,
            });
        selected_indices = best.selected_indices;
    } else {
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        match mode {
            MemorySurfacingModeV1::FailureOnly => {
                order.retain(|index| candidates[*index].observed_failure);
                order.sort_by(|left, right| {
                    candidates[*right]
                        .last_accessed_at_ms
                        .cmp(&candidates[*left].last_accessed_at_ms)
                        .then_with(|| candidates[*left].event_id.cmp(&candidates[*right].event_id))
                });
            }
            MemorySurfacingModeV1::RecencyFirst => {
                order.sort_by(|left, right| {
                    candidates[*right]
                        .last_accessed_at_ms
                        .cmp(&candidates[*left].last_accessed_at_ms)
                        .then_with(|| candidates[*left].event_id.cmp(&candidates[*right].event_id))
                });
            }
            MemorySurfacingModeV1::FrequencyFirst => {
                order.sort_by(|left, right| {
                    candidates[*right]
                        .historical_access_count
                        .cmp(&candidates[*left].historical_access_count)
                        .then_with(|| {
                            candidates[*right]
                                .last_accessed_at_ms
                                .cmp(&candidates[*left].last_accessed_at_ms)
                        })
                        .then_with(|| candidates[*left].event_id.cmp(&candidates[*right].event_id))
                });
            }
            MemorySurfacingModeV1::PredictedFutureValue => unreachable!(),
        }
        let mut used = 0_u32;
        for index in order {
            let cost = candidates[index].context_token_cost;
            if used.saturating_add(cost) <= context_budget_tokens {
                used += cost;
                selected_indices.push(index);
            }
        }
    }

    let mut surfaced = Vec::with_capacity(selected_indices.len());
    let mut context_tokens_used = 0_u32;
    let mut predicted_net_value_micros = 0_i64;
    for index in selected_indices {
        let candidate = &candidates[index];
        let (expected, prediction_sha256, calibration_receipt_sha256) =
            candidate.prediction.as_ref().map_or(
                (0, None, None),
                |verified| {
                    (
                        verified.prediction().expected_net_value_micros(),
                        Some(verified.prediction().prediction_sha256.clone()),
                        Some(
                            verified
                                .calibration_receipt()
                                .receipt_sha256
                                .clone(),
                        ),
                    )
                },
            );
        context_tokens_used = context_tokens_used.saturating_add(candidate.context_token_cost);
        predicted_net_value_micros =
            predicted_net_value_micros.saturating_add(expected);
        surfaced.push(SurfacedMemoryV1 {
            event_id: candidate.event_id,
            context_token_cost: candidate.context_token_cost,
            evidence_sha256: candidate.evidence_sha256.clone(),
            prediction_sha256,
            calibration_receipt_sha256,
            expected_net_value_micros: expected,
        });
    }

    let mut plan = MemorySurfacingPlanV1 {
        schema_version: 1,
        mode,
        target_scope,
        as_known,
        context_budget_tokens,
        context_tokens_used,
        predicted_net_value_micros,
        surfaced,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = plan.canonical_digest();
    Ok(plan)
}

impl MemorySurfacingPlanV1 {
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            self.schema_version.to_string(),
            self.mode.code().to_owned(),
            scope_code(self.target_scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.context_budget_tokens.to_string(),
            self.context_tokens_used.to_string(),
            self.predicted_net_value_micros.to_string(),
            self.surfaced.len().to_string(),
        ];
        for item in &self.surfaced {
            fields.push(item.event_id.to_string());
            fields.push(item.context_token_cost.to_string());
            fields.push(item.evidence_sha256.clone());
            fields.push(item.prediction_sha256.clone().unwrap_or_default());
            fields.push(
                item.calibration_receipt_sha256
                    .clone()
                    .unwrap_or_default(),
            );
            fields.push(item.expected_net_value_micros.to_string());
        }
        digest_fields(b"mycelix:mace:memory-surfacing-plan:v1\0", &fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{Objective, Regime};

    fn scope() -> Scope {
        Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        }
    }

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn candidate(
        event_id: u32,
        tokens: u32,
        failure: bool,
        last_access: i64,
        frequency: u32,
        prediction: Option<VerifiedFutureUsePredictionV1>,
    ) -> MemorySurfacingCandidateV1 {
        MemorySurfacingCandidateV1 {
            event_id,
            scope: scope(),
            as_known: AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            context_token_cost: tokens,
            last_accessed_at_ms: last_access,
            historical_access_count: frequency,
            observed_failure: failure,
            evidence_sha256: digest(&format!("evidence-{event_id}")),
            prediction,
        }
    }

    fn raw_prediction(
        event_id: u32,
        probability: u16,
        gain: u32,
        penalty: u32,
        evidence_sha256: String,
    ) -> FutureUsePredictionV1 {
        FutureUsePredictionV1::new(
            event_id,
            digest("target-agent-model"),
            scope(),
            90,
            1_000,
            probability,
            gain,
            penalty,
            "symthaea-salience",
            "v1",
            digest("model-config"),
            digest(&format!("rationale-{event_id}")),
            evidence_sha256,
        )
    }

    fn calibration_receipt(
        prediction: &FutureUsePredictionV1,
        as_known: AsKnownQuery,
    ) -> SalienceCalibrationReceiptV1 {
        let mut receipt = SalienceCalibrationReceiptV1 {
            prediction_sha256: prediction.prediction_sha256.clone(),
            evidence_sha256: prediction.evidence_sha256.clone(),
            target_model_sha256: prediction.target_model_sha256.clone(),
            target_scope: prediction.target_scope,
            as_known,
            model_id: prediction.model_id.clone(),
            model_version: prediction.model_version.clone(),
            model_config_sha256: prediction.model_config_sha256.clone(),
            training_dataset_sha256: digest("training-dataset-v1"),
            calibration_dataset_sha256: digest("calibration-dataset-v1"),
            heldout_evaluation_sha256: digest("heldout-evaluation-v1"),
            task_family_split_sha256: digest("task-family-split-v1"),
            source_root_split_sha256: digest("source-root-split-v1"),
            training_cluster_count: 128,
            calibration_cluster_count: 64,
            heldout_cluster_count: 64,
            brier_score_micros: 80_000,
            base_rate_brier_score_micros: 140_000,
            expected_calibration_error_micros: 20_000,
            worst_group_calibration_error_micros: 40_000,
            task_families_disjoint: true,
            source_roots_disjoint: true,
            verified_at_ms: 95,
            valid_until_ms: 10_000,
            verifier_id: "fixture-independent-salience-verifier".into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.canonical_digest();
        receipt
    }

    struct FixtureSaliencePredictionVerifier {
        receipt: Option<SalienceCalibrationReceiptV1>,
    }

    impl SaliencePredictionVerifierV1 for FixtureSaliencePredictionVerifier {
        fn verify_prediction(
            &self,
            _prediction: &FutureUsePredictionV1,
            _target_scope: Scope,
            _as_known: AsKnownQuery,
        ) -> Option<SalienceCalibrationReceiptV1> {
            self.receipt.clone()
        }
    }

    fn verify_fixture_prediction(
        prediction: FutureUsePredictionV1,
        as_known: AsKnownQuery,
    ) -> Result<VerifiedFutureUsePredictionV1, SalienceReject> {
        let receipt = calibration_receipt(&prediction, as_known);
        verify_future_use_prediction(
            prediction,
            scope(),
            as_known,
            &FixtureSaliencePredictionVerifier {
                receipt: Some(receipt),
            },
        )
    }

    fn prediction(
        event_id: u32,
        probability: u16,
        gain: u32,
        penalty: u32,
        evidence_sha256: String,
    ) -> VerifiedFutureUsePredictionV1 {
        verify_fixture_prediction(
            raw_prediction(event_id, probability, gain, penalty, evidence_sha256),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
        )
        .expect("fixture calibration receipt is valid")
    }

    #[derive(Clone, Copy, Debug)]
    struct SalienceRng {
        state: u32,
    }

    impl SalienceRng {
        fn new(seed: u32) -> Self {
            Self { state: seed.max(1) }
        }

        fn unit(&mut self) -> f64 {
            let mut value = self.state;
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            self.state = value;
            f64::from(value) / (f64::from(u32::MAX) + 1.0)
        }
    }

    #[derive(Clone, Debug)]
    struct GeneratedFutureUse {
        candidate: MemorySurfacingCandidateV1,
        true_reuse_probability: f64,
        true_value_gain_micros: f64,
        realised_reuse: bool,
    }

    #[derive(Clone, Debug, Default)]
    struct OracleKnapsackState {
        value_micros: u64,
        selected_indices: Vec<usize>,
    }

    fn oracle_knapsack(
        items: &[GeneratedFutureUse],
        budget_tokens: u32,
    ) -> Vec<usize> {
        let width = usize::try_from(budget_tokens).unwrap();
        let mut states: Vec<Option<OracleKnapsackState>> = vec![None; width + 1];
        states[0] = Some(OracleKnapsackState::default());

        for (index, item) in items.iter().enumerate() {
            let cost = usize::try_from(item.candidate.context_token_cost).unwrap();
            if cost > width {
                continue;
            }
            let value = (item.true_reuse_probability
                * item.true_value_gain_micros
                * 1_000.0)
                .round() as u64;
            for capacity in (cost..=width).rev() {
                let Some(previous) = states[capacity - cost].clone() else {
                    continue;
                };
                let mut next = previous;
                next.value_micros = next.value_micros.saturating_add(value);
                next.selected_indices.push(index);
                let replace = match &states[capacity] {
                    Some(current) => {
                        next.value_micros > current.value_micros
                            || (next.value_micros == current.value_micros
                                && next.selected_indices < current.selected_indices)
                    }
                    None => true,
                };
                if replace {
                    states[capacity] = Some(next);
                }
            }
        }

        states
            .into_iter()
            .flatten()
            .reduce(|best, current| {
                if current.value_micros > best.value_micros
                    || (current.value_micros == best.value_micros
                        && current.selected_indices < best.selected_indices)
                {
                    current
                } else {
                    best
                }
            })
            .unwrap_or_default()
            .selected_indices
    }

    #[test]
    fn calibration_receipt_cannot_cross_target_agent_models() {
        let raw = raw_prediction(
            77,
            9_500,
            1_000,
            0,
            digest("evidence-77"),
        );
        let mut receipt = calibration_receipt(
            &raw,
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
        );
        receipt.target_model_sha256 = digest("different-target-model");
        receipt.receipt_sha256 = receipt.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                raw,
                scope(),
                AsKnownQuery {
                    valid_at_ms: 100,
                    known_at_ms: 100,
                },
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(receipt),
                },
            ),
            Err(SalienceReject::PredictionTargetModelMismatch)
        );
    }

    #[test]
    fn model_proposed_future_value_can_surface_rare_successes() {
        let low_value_failure = candidate(
            1,
            8,
            true,
            99,
            1,
            Some(prediction(1, 5_000, 1_000, 50, digest("evidence-1"))),
        );
        let rare_success = candidate(
            2,
            8,
            false,
            40,
            0,
            Some(prediction(2, 9_500, 8_000, 100, digest("evidence-2"))),
        );
        let candidates = vec![low_value_failure, rare_success];
        let failure_only = select_for_context(
            &candidates,
            MemorySurfacingModeV1::FailureOnly,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        let predicted = select_for_context(
            &candidates,
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();

        assert_eq!(
            failure_only
                .surfaced
                .iter()
                .map(|item| item.event_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            predicted
                .surfaced
                .iter()
                .map(|item| item.event_id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(predicted.context_tokens_used, 8);
    }

    #[test]
    fn frequency_first_can_beats_value_and_is_not_the_oracle() {
        let often_used = candidate(
            1,
            8,
            false,
            99,
            100,
            Some(prediction(1, 3_000, 500, 100, digest("evidence-1"))),
        );
        let high_value = candidate(
            2,
            8,
            false,
            40,
            1,
            Some(prediction(2, 9_000, 7_000, 100, digest("evidence-2"))),
        );
        let candidates = vec![often_used, high_value];
        let frequency = select_for_context(
            &candidates,
            MemorySurfacingModeV1::FrequencyFirst,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        let predicted = select_for_context(
            &candidates,
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();

        assert_eq!(frequency.surfaced[0].event_id, 1);
        assert_eq!(predicted.surfaced[0].event_id, 2);
    }

    #[test]
    fn negative_or_zero_expected_value_is_not_preloaded() {
        let candidate = candidate(
            1,
            8,
            false,
            99,
            1,
            Some(prediction(1, 1_000, 100, 100, digest("evidence-1"))),
        );
        let plan = select_for_context(
            &[candidate],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        assert!(plan.surfaced.is_empty());
        assert_eq!(plan.context_tokens_used, 0);
    }

    #[test]
    fn future_prediction_and_wrong_scope_cannot_be_calibration_verified() {
        let as_known = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 100,
        };
        let evidence = digest("evidence-1");
        let mut future = raw_prediction(1, 9_000, 8_000, 100, evidence.clone());
        future.predicted_at_ms = 101;
        future.prediction_sha256 = future.canonical_digest();
        assert_eq!(
            verify_fixture_prediction(future, as_known),
            Err(SalienceReject::PredictionFutureLeakage)
        );

        let mut other_scope = raw_prediction(1, 9_000, 8_000, 100, evidence);
        other_scope.target_scope.objective = Objective::Cost;
        other_scope.prediction_sha256 = other_scope.canonical_digest();
        assert_eq!(
            verify_fixture_prediction(other_scope, as_known),
            Err(SalienceReject::PredictionScopeMismatch)
        );
    }

    #[test]
    fn stale_calibration_receipt_cannot_authorize_a_changed_prediction_or_evidence() {
        let as_known = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 100,
        };
        let original = raw_prediction(
            1,
            9_000,
            8_000,
            100,
            digest("evidence-1"),
        );
        let stale_receipt = calibration_receipt(&original, as_known);

        let mut changed_prediction = original.clone();
        changed_prediction.reuse_probability_basis_points = 100;
        changed_prediction.prediction_sha256 = changed_prediction.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                changed_prediction,
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(stale_receipt.clone()),
                },
            ),
            Err(SalienceReject::PredictionCalibrationMismatch)
        );

        let mut changed_evidence = original;
        changed_evidence.evidence_sha256 = digest("different-evidence");
        changed_evidence.prediction_sha256 = changed_evidence.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                changed_evidence,
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(stale_receipt),
                },
            ),
            Err(SalienceReject::PredictionCalibrationMismatch)
        );
    }

    #[test]
    fn missing_or_poorly_calibrated_receipt_fails_closed() {
        let as_known = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 100,
        };
        let raw = raw_prediction(2, 9_000, 8_000, 100, digest("evidence-2"));
        assert_eq!(
            verify_future_use_prediction(
                raw.clone(),
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier { receipt: None },
            ),
            Err(SalienceReject::PredictionCalibrationMissing)
        );

        let mut poor = calibration_receipt(&raw, as_known);
        poor.expected_calibration_error_micros = MAX_SALIENCE_ECE_MICROS + 1;
        poor.receipt_sha256 = poor.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                raw,
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(poor),
                },
            ),
            Err(SalienceReject::CalibrationQualityInsufficient)
        );
    }

    #[test]
    fn exact_knapsack_fills_context_budget_by_total_expected_value() {
        let candidates = vec![
            candidate(
                1,
                6,
                false,
                99,
                0,
                Some(prediction(1, 9_000, 4_000, 100, digest("evidence-1"))),
            ),
            candidate(
                2,
                6,
                false,
                90,
                0,
                Some(prediction(2, 9_000, 4_000, 100, digest("evidence-2"))),
            ),
            candidate(
                3,
                12,
                false,
                80,
                0,
                Some(prediction(3, 9_000, 9_000, 100, digest("evidence-3"))),
            ),
        ];
        let plan = select_for_context(
            &candidates,
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            12,
        )
        .unwrap();
        assert_eq!(
            plan.surfaced
                .iter()
                .map(|item| item.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(plan.context_tokens_used, 12);
        assert_eq!(plan.predicted_net_value_micros, 7_000);
    }

    #[test]
    fn selection_is_deterministic_and_the_plan_binds_source_evidence() {
        let candidate = candidate(
            2,
            8,
            false,
            99,
            1,
            Some(prediction(2, 9_000, 8_000, 100, digest("evidence-2"))),
        );
        let first = select_for_context(
            &[candidate.clone()],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        let second = select_for_context(
            &[candidate],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.plan_sha256, first.canonical_digest());
    }

    #[test]
    fn cross_scope_evidence_is_rejected_by_every_surfacing_policy() {
        let mut value = candidate(9, 4, false, 90, 1, None);
        value.scope.objective = Objective::Cost;
        for mode in [
            MemorySurfacingModeV1::FailureOnly,
            MemorySurfacingModeV1::RecencyFirst,
            MemorySurfacingModeV1::FrequencyFirst,
            MemorySurfacingModeV1::PredictedFutureValue,
        ] {
            assert_eq!(
                select_for_context(
                    &[value.clone()],
                    mode,
                    scope(),
                    AsKnownQuery {
                        valid_at_ms: 100,
                        known_at_ms: 100,
                    },
                    8,
                ),
                Err(SalienceReject::EvidenceScopeMismatch)
            );
        }
    }

    #[test]
    fn salience_rejects_stale_and_future_bitemporal_evidence_snapshots() {
        let value = candidate(10, 4, false, 90, 1, None);
        for requested in [
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 99,
            },
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 101,
            },
            AsKnownQuery {
                valid_at_ms: 99,
                known_at_ms: 100,
            },
            AsKnownQuery {
                valid_at_ms: 101,
                known_at_ms: 100,
            },
        ] {
            assert_eq!(
                select_for_context(
                    &[value.clone()],
                    MemorySurfacingModeV1::RecencyFirst,
                    scope(),
                    requested,
                    8,
                ),
                Err(SalienceReject::EvidenceSnapshotMismatch)
            );
        }
    }

    #[test]
    fn generated_future_use_benchmark_calibrates_salience_under_equal_budgets() {
        const CLUSTERS: usize = 256;
        const ITEMS_PER_CLUSTER: usize = 16;
        const CONTEXT_BUDGET: u32 = 64;
        const AS_KNOWN: AsKnownQuery = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 100,
        };

        let modes = [
            MemorySurfacingModeV1::FailureOnly,
            MemorySurfacingModeV1::RecencyFirst,
            MemorySurfacingModeV1::FrequencyFirst,
            MemorySurfacingModeV1::PredictedFutureValue,
        ];
        let mut rng = SalienceRng::new(0x5eed);
        let mut forecast_rows = Vec::<(f64, bool)>::new();
        let mut expected_value = [0.0_f64; 4];
        let mut realised_value = [0.0_f64; 4];
        let mut token_use = [0_u64; 4];
        let mut oracle_value = 0.0_f64;

        for cluster in 0..CLUSTERS {
            let mut items = Vec::with_capacity(ITEMS_PER_CLUSTER);
            let mut id_to_index = std::collections::BTreeMap::new();

            for offset in 0..ITEMS_PER_CLUSTER {
                let event_id = (cluster * ITEMS_PER_CLUSTER + offset + 1) as u32;
                let latent_value = rng.unit().powf(0.82);
                let true_reuse_probability = 0.04 + 0.92 * latent_value;
                let forecast_noise = (rng.unit() * 2.0 - 1.0) * 0.08;
                let forecast_probability =
                    (true_reuse_probability + forecast_noise).clamp(0.02, 0.98);
                let true_value_gain_micros = 1_200.0 + 8_000.0 * latent_value;
                let predicted_value_gain = (1_200.0
                    + 8_000.0 * latent_value
                    + (rng.unit() * 2.0 - 1.0) * 800.0)
                    .clamp(500.0, 10_000.0)
                    .round() as u32;
                let uncertainty_penalty =
                    120 + ((1.0 - latent_value) * 200.0).round() as u32;
                let context_tokens = 6 + (rng.unit() * 15.0).floor() as u32;
                let observed_failure = rng.unit() < 0.25;
                let last_accessed_at_ms = (rng.unit() * 100.0).floor() as i64;
                let access_frequency = (rng.unit() * 20.0).floor() as u32;
                let realised_reuse = rng.unit() < true_reuse_probability;
                forecast_rows.push((forecast_probability, realised_reuse));

                let evidence_digest = digest(&format!("salience-evidence-{event_id}"));
                let forecast = prediction(
                    event_id,
                    (forecast_probability * 10_000.0).round() as u16,
                    predicted_value_gain,
                    uncertainty_penalty,
                    evidence_digest,
                );
                let value = candidate(
                    event_id,
                    context_tokens,
                    observed_failure,
                    last_accessed_at_ms,
                    access_frequency,
                    Some(forecast),
                );
                id_to_index.insert(event_id, items.len());
                items.push(GeneratedFutureUse {
                    candidate: value,
                    true_reuse_probability,
                    true_value_gain_micros,
                    realised_reuse,
                });
            }

            for (mode_index, mode) in modes.iter().copied().enumerate() {
                let candidates: Vec<_> =
                    items.iter().map(|item| item.candidate.clone()).collect();
                let plan = select_for_context(
                    &candidates,
                    mode,
                    scope(),
                    AS_KNOWN,
                    CONTEXT_BUDGET,
                )
                .unwrap();
                for surfaced in &plan.surfaced {
                    let index = *id_to_index.get(&surfaced.event_id).unwrap();
                    let item = &items[index];
                    expected_value[mode_index] +=
                        item.true_reuse_probability * item.true_value_gain_micros;
                    if item.realised_reuse {
                        realised_value[mode_index] += item.true_value_gain_micros;
                    }
                    token_use[mode_index] +=
                        u64::from(item.candidate.context_token_cost);
                }
                assert!(plan.context_tokens_used <= CONTEXT_BUDGET);
            }

            let oracle_indices = oracle_knapsack(&items, CONTEXT_BUDGET);
            for index in oracle_indices {
                let item = &items[index];
                oracle_value +=
                    item.true_reuse_probability * item.true_value_gain_micros;
            }
        }

        let total = forecast_rows.len() as f64;
        let observed_rate = forecast_rows
            .iter()
            .filter(|(_, outcome)| *outcome)
            .count() as f64
            / total;
        let mut brier = 0.0_f64;
        let mut base_rate_brier = 0.0_f64;
        let mut bin_count = [0_u32; 10];
        let mut bin_probability = [0.0_f64; 10];
        let mut bin_outcome = [0.0_f64; 10];
        for (probability, outcome) in &forecast_rows {
            let observed = if *outcome { 1.0 } else { 0.0 };
            brier += (probability - observed).powi(2);
            base_rate_brier += (observed_rate - observed).powi(2);
            let bin = ((probability * 10.0).floor() as usize).min(9);
            bin_count[bin] += 1;
            bin_probability[bin] += probability;
            bin_outcome[bin] += observed;
        }
        brier /= total;
        base_rate_brier /= total;
        let mut expected_calibration_error = 0.0_f64;
        for bin in 0..10 {
            if bin_count[bin] == 0 {
                continue;
            }
            let count = f64::from(bin_count[bin]);
            let mean_probability = bin_probability[bin] / count;
            let observed_frequency = bin_outcome[bin] / count;
            expected_calibration_error +=
                count / total * (mean_probability - observed_frequency).abs();
        }

        let predicted_index = 3;
        assert_eq!(forecast_rows.len(), CLUSTERS * ITEMS_PER_CLUSTER);
        assert!(brier < base_rate_brier);
        assert!(expected_calibration_error < 0.05);
        assert!(expected_value[predicted_index] > expected_value[0] * 1.5);
        assert!(expected_value[predicted_index] > expected_value[1] * 1.5);
        assert!(expected_value[predicted_index] > expected_value[2] * 1.5);
        assert!(
            expected_value[predicted_index] / oracle_value > 0.98,
            "predicted policy regret exceeded the generated oracle tolerance"
        );
        assert!(realised_value[predicted_index] > realised_value[0]);
        assert!(realised_value[predicted_index] > realised_value[1]);
        assert!(realised_value[predicted_index] > realised_value[2]);
        let predicted_value_per_token =
            expected_value[predicted_index] / token_use[predicted_index] as f64;
        for mode_index in 0..3 {
            let baseline_value_per_token =
                expected_value[mode_index] / token_use[mode_index] as f64;
            assert!(predicted_value_per_token > baseline_value_per_token * 1.4);
        }

        println!(
            concat!(
                "salience_generated clusters={} forecasts={} brier={:.4} ",
                "base_rate_brier={:.4} ece={:.4} predicted_vs_oracle={:.4} ",
                "predicted_expected_utility_per_cluster={:.1}"
            ),
            CLUSTERS,
            forecast_rows.len(),
            brier,
            base_rate_brier,
            expected_calibration_error,
            expected_value[predicted_index] / oracle_value,
            expected_value[predicted_index] / CLUSTERS as f64
        );
    }

    #[test]
    fn calibration_receipt_must_attest_disjoint_task_families_and_source_roots() {
        let as_known = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 100,
        };
        let raw = raw_prediction(3, 9_000, 8_000, 100, digest("evidence-3"));

        let mut overlapping_tasks = calibration_receipt(&raw, as_known);
        overlapping_tasks.task_families_disjoint = false;
        overlapping_tasks.receipt_sha256 = overlapping_tasks.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                raw.clone(),
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(overlapping_tasks),
                },
            ),
            Err(SalienceReject::CalibrationSplitNotIndependent)
        );

        let mut overlapping_roots = calibration_receipt(&raw, as_known);
        overlapping_roots.source_roots_disjoint = false;
        overlapping_roots.receipt_sha256 = overlapping_roots.canonical_digest();
        assert_eq!(
            verify_future_use_prediction(
                raw,
                scope(),
                as_known,
                &FixtureSaliencePredictionVerifier {
                    receipt: Some(overlapping_roots),
                },
            ),
            Err(SalienceReject::CalibrationSplitNotIndependent)
        );
    }

    #[test]
    fn surfaced_plan_binds_the_prediction_and_independent_calibration_receipt() {
        let verified = prediction(
            5,
            9_000,
            8_000,
            100,
            digest("evidence-5"),
        );
        let prediction_sha = verified.prediction().prediction_sha256.clone();
        let receipt_sha = verified.calibration_receipt().receipt_sha256.clone();
        let candidate = candidate(5, 8, false, 99, 1, Some(verified));
        let plan = select_for_context(
            &[candidate],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope(),
            AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            8,
        )
        .unwrap();
        assert_eq!(
            plan.surfaced[0].prediction_sha256.as_deref(),
            Some(prediction_sha.as_str())
        );
        assert_eq!(
            plan.surfaced[0].calibration_receipt_sha256.as_deref(),
            Some(receipt_sha.as_str())
        );
        assert_eq!(plan.plan_sha256, plan.canonical_digest());
    }

    #[test]
    fn budget_and_duplicate_event_guards_fail_closed() {
        let candidate = candidate(1, 8, false, 99, 1, None);
        assert_eq!(
            select_for_context(
                &[candidate.clone()],
                MemorySurfacingModeV1::RecencyFirst,
                scope(),
                AsKnownQuery {
                    valid_at_ms: 100,
                    known_at_ms: 100,
                },
                MAX_CONTEXT_BUDGET_TOKENS + 1,
            ),
            Err(SalienceReject::InvalidBudget)
        );
        assert_eq!(
            select_for_context(
                &[candidate.clone(), candidate],
                MemorySurfacingModeV1::RecencyFirst,
                scope(),
                AsKnownQuery {
                    valid_at_ms: 100,
                    known_at_ms: 100,
                },
                8,
            ),
            Err(SalienceReject::DuplicateEventId)
        );
    }
}
