pub mod claims;
pub mod providers;

// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Research-only boundary from raw Session-A interactions to candidate memory.
//!
//! This crate does not implement cryptographic verification. A production
//! EvidenceVerifier must delegate to Mycelix source-ledger, calibration and
//! outcome-reconciliation receipts. Agent-authored labels are not trust anchors.

use std::collections::BTreeMap;

use broca_mace_policy::{
    Action, CausalIdentificationV1, Experience, Scope, SemanticEstimandV1,
    SourceRootDigest,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsKnownQuery {
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Raw outcome envelope. Trust-bearing fields are receipt IDs, not booleans.
#[derive(Clone, Debug)]
pub struct RawInteractionV1 {
    pub event_id: u32,
    pub session_id: String,
    pub task_id: String,
    /// SHA-256 digest of the canonical immutable event payload.
    pub payload_sha256: String,
    pub raw_event_receipt_id: String,
    pub scope: Scope,
    pub action: Action,
    pub observed_utility: f64,
    pub utility_unit: String,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    /// Client/source event time; never used as transaction-time authority.
    pub occurred_at_ms: i64,
    pub prediction_registered_at_ms: Option<i64>,
    pub action_dispatched_at_ms: Option<i64>,
    pub outcome_observed_at_ms: i64,
    pub lineage_receipt_id: String,
    pub calibration_receipt_id: String,
    pub causal_assessment_receipt_id: String,
    pub outcome_receipt_id: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceTrustStatus {
    /// Source is authorized for this source profile and scope, not proven true.
    TrustedForScope,
    Untrusted,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedLineage {
    pub source_family: u16,
    /// Legacy/display identifier; digest is the actual independence key.
    pub source_root_id: u32,
    pub source_root_digest: SourceRootDigest,
    pub trust_status: SourceTrustStatus,
    /// Digest of the trust policy revision used to classify the source.
    pub trust_policy_digest: String,
    pub acquisition_mode: String,
    pub upstream_source_id: String,
    pub source_artifact_digest: String,
    pub transformation_digest: String,
    pub issuer_id: String,
    pub receipt_digest: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorityStatus {
    ReceiptAttested,
    PendingReconciliation,
    LegacyUnattested,
    UnsafeUnattested,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRawEvent {
    pub event_id: u32,
    pub session_id: String,
    pub task_id: String,
    pub payload_sha256: String,
    pub canonical_event_hash: String,
    pub authority_status: AuthorityStatus,
    pub authority_receipt_digest: String,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    pub occurred_at_ms: i64,
    /// Server-observed, signed authority receipt time used for known_at.
    pub received_at_ms: i64,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedMeasurement {
    pub absolute_error_bound: f64,
    pub unit: String,
    pub instrument_id: String,
    pub method_id: String,
    pub calibration_revision: String,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    pub calibration_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedCausalAssessment {
    pub intervention_confirmed: bool,
    pub prediction_pre_registered: bool,
    pub confounders_controlled: bool,
    pub delayed_horizon_complete: bool,
    /// Method, design and estimand metadata covered by assessment_digest.
    pub identification: CausalIdentificationV1,
    pub assessment_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedOutcome {
    pub reconciled: bool,
    pub success: bool,
    /// Authoritative value returned by the independently verified outcome receipt.
    pub observed_utility: f64,
    /// Unit bound by the outcome receipt, which must match calibration.
    pub utility_unit: String,
    pub outcome_digest: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceLifecycleStatus {
    Active,
    Retracted,
    Superseded { successor_event_id: u32 },
}

/// Production implementations must verify issuer, signatures, scope,
/// content digests, valid/transaction time, revocation and calibration range.
pub trait EvidenceVerifier {
    /// Verify the receipt over the complete canonical raw event, not caller-
    /// supplied time fields or selected metadata alone.
    fn verify_raw_event(
        &self,
        receipt_id: &str,
        raw: &RawInteractionV1,
        known_at_ms: i64,
    ) -> Option<VerifiedRawEvent>;

    fn verify_lineage(
        &self,
        receipt_id: &str,
        event_id: u32,
        scope: Scope,
        known_at_ms: i64,
    ) -> Option<VerifiedLineage>;

    fn verify_measurement(
        &self,
        receipt_id: &str,
        expected_unit: &str,
        scope: Scope,
        observed_at_ms: i64,
        known_at_ms: i64,
    ) -> Option<VerifiedMeasurement>;

    fn verify_causal_assessment(
        &self,
        receipt_id: &str,
        event_id: u32,
        scope: Scope,
        known_at_ms: i64,
    ) -> Option<VerifiedCausalAssessment>;

    fn verify_outcome(
        &self,
        receipt_id: &str,
        event_id: u32,
        observed_at_ms: i64,
        known_at_ms: i64,
    ) -> Option<VerifiedOutcome>;

    /// None means lifecycle state could not be verified and must fail closed.
    fn lifecycle_status_as_of(
        &self,
        event_id: u32,
        known_at_ms: i64,
    ) -> Option<SourceLifecycleStatus>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceReceiptBundle {
    pub raw_event_digest: String,
    pub raw_event_receipt_digest: String,
    pub session_id: String,
    pub task_id: String,
    pub canonical_event_hash: String,
    pub authority_receipt_digest: String,
    pub event_occurred_at_ms: i64,
    pub authority_received_at_ms: i64,
    pub lineage_digest: String,
    pub source_family: u16,
    pub source_root_id: u32,
    pub source_root_digest: SourceRootDigest,
    pub source_acquisition_mode: String,
    pub upstream_source_id: String,
    pub source_artifact_digest: String,
    pub transformation_digest: String,
    pub source_issuer_id: String,
    pub source_trust_status: SourceTrustStatus,
    pub source_trust_policy_digest: String,
    pub calibration_digest: String,
    pub measurement_unit: String,
    pub measurement_instrument_id: String,
    pub calibration_method_id: String,
    pub calibration_revision: String,
    pub causal_assessment_digest: String,
    pub causal_identification: CausalIdentificationV1,
    pub outcome_success: bool,
    /// Stored as bits so the complete receipt bundle remains exactly comparable.
    pub outcome_observed_utility_bits: u64,
    pub outcome_utility_unit: String,
    pub outcome_digest: String,
}

#[derive(Clone, Debug)]
pub struct IngestedExperience {
    experience: Experience,
    as_known: AsKnownQuery,
    receipts: EvidenceReceiptBundle,
}

impl IngestedExperience {
    pub fn experience(&self) -> &Experience {
        &self.experience
    }

    pub fn as_known(&self) -> AsKnownQuery {
        self.as_known
    }

    pub fn receipts(&self) -> &EvidenceReceiptBundle {
        &self.receipts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestReject {
    MalformedValidInterval,
    InvalidPayloadDigest,
    UnverifiedRawEvent,
    RawEventReceiptMismatch,
    AuthorityNotAttested,
    AuthorityReceiptPredatesOutcome,
    RetractionStatusUnknown,
    SupersededAsKnown(u32),
    NotValidAtRequestedTime,
    RecordedAfterKnownAt,
    OutcomeNotKnownAtRequestedTime,
    PredictionMissing,
    DispatchMissing,
    PredictionNotBeforeDispatch,
    DispatchNotBeforeOutcome,
    RetractedAsKnown,
    InvalidUtility,
    UnverifiedLineage,
    UnknownSourceRoot,
    UnknownSourceFamily,
    InvalidTrustPolicyDigest,
    UnverifiedCalibration,
    InvalidMeasurementBound,
    UnverifiedCausalAssessment,
    IncompleteSemanticEstimand,
    SemanticEstimandDigestMismatch,
    UnverifiedOutcome,
    InvalidVerifiedOutcomeValue,
    OutcomeUnitMismatch,
    OutcomeValueMismatch,
    OutcomeNotReconciled,
}

/// Convert a raw interaction into candidate-visible memory only after
/// bitemporal eligibility and independent receipt verification succeed.
pub fn ingest_as_known(
    raw: &RawInteractionV1,
    query: AsKnownQuery,
    verifier: &impl EvidenceVerifier,
) -> Result<IngestedExperience, IngestReject> {
    if !is_sha256_hex(&raw.payload_sha256) {
        return Err(IngestReject::InvalidPayloadDigest);
    }

    let event = verifier
        .verify_raw_event(&raw.raw_event_receipt_id, raw, query.known_at_ms)
        .ok_or(IngestReject::UnverifiedRawEvent)?;
    if event.event_id != raw.event_id
        || event.session_id != raw.session_id
        || event.task_id != raw.task_id
        || event.payload_sha256 != raw.payload_sha256
        || !is_sha256_hex(&event.canonical_event_hash)
        || !is_sha256_hex(&event.authority_receipt_digest)
        || !is_sha256_hex(&event.receipt_digest)
        || event.valid_from_ms != raw.valid_from_ms
        || event.valid_until_ms != raw.valid_until_ms
        || event.occurred_at_ms != raw.occurred_at_ms
        || event.received_at_ms < 0
    {
        return Err(IngestReject::RawEventReceiptMismatch);
    }
    if event.authority_status != AuthorityStatus::ReceiptAttested {
        return Err(IngestReject::AuthorityNotAttested);
    }
    if event.received_at_ms < raw.outcome_observed_at_ms {
        return Err(IngestReject::AuthorityReceiptPredatesOutcome);
    }
    if event.valid_until_ms.is_some_and(|end| end <= event.valid_from_ms) {
        return Err(IngestReject::MalformedValidInterval);
    }
    if event.valid_from_ms > query.valid_at_ms
        || event.valid_until_ms.is_some_and(|end| query.valid_at_ms >= end)
    {
        return Err(IngestReject::NotValidAtRequestedTime);
    }
    if event.received_at_ms > query.known_at_ms {
        return Err(IngestReject::RecordedAfterKnownAt);
    }
    if raw.outcome_observed_at_ms > query.known_at_ms {
        return Err(IngestReject::OutcomeNotKnownAtRequestedTime);
    }

    let prediction = raw
        .prediction_registered_at_ms
        .ok_or(IngestReject::PredictionMissing)?;
    let dispatch = raw
        .action_dispatched_at_ms
        .ok_or(IngestReject::DispatchMissing)?;
    if prediction >= dispatch {
        return Err(IngestReject::PredictionNotBeforeDispatch);
    }
    if dispatch >= raw.outcome_observed_at_ms {
        return Err(IngestReject::DispatchNotBeforeOutcome);
    }
    match verifier
        .lifecycle_status_as_of(raw.event_id, query.known_at_ms)
        .ok_or(IngestReject::RetractionStatusUnknown)?
    {
        SourceLifecycleStatus::Active => {}
        SourceLifecycleStatus::Retracted => {
            return Err(IngestReject::RetractedAsKnown);
        }
        SourceLifecycleStatus::Superseded { successor_event_id } => {
            return Err(IngestReject::SupersededAsKnown(successor_event_id));
        }
    }
    if !raw.observed_utility.is_finite() || !(0.0..=1.0).contains(&raw.observed_utility) {
        return Err(IngestReject::InvalidUtility);
    }

    let lineage = verifier
        .verify_lineage(
            &raw.lineage_receipt_id,
            raw.event_id,
            raw.scope,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedLineage)?;
    if !lineage.source_root_digest.is_known() {
        return Err(IngestReject::UnknownSourceRoot);
    }
    if lineage.source_family == 0 {
        return Err(IngestReject::UnknownSourceFamily);
    }
    if lineage.acquisition_mode.is_empty()
        || lineage.upstream_source_id.is_empty()
        || !is_sha256_hex(&lineage.source_artifact_digest)
        || !is_sha256_hex(&lineage.transformation_digest)
        || lineage.issuer_id.is_empty()
        || !is_sha256_hex(&lineage.receipt_digest)
    {
        return Err(IngestReject::UnverifiedLineage);
    }
    if !is_sha256_hex(&lineage.trust_policy_digest) {
        return Err(IngestReject::InvalidTrustPolicyDigest);
    }

    let measurement = verifier
        .verify_measurement(
            &raw.calibration_receipt_id,
            &raw.utility_unit,
            raw.scope,
            raw.outcome_observed_at_ms,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedCalibration)?;
    if !measurement.absolute_error_bound.is_finite()
        || !(0.0..=1.0).contains(&measurement.absolute_error_bound)
        || raw.utility_unit.is_empty()
        || measurement.unit != raw.utility_unit
        || measurement.instrument_id.is_empty()
        || measurement.method_id.is_empty()
        || measurement.calibration_revision.is_empty()
        || !is_sha256_hex(&measurement.calibration_digest)
        || measurement.valid_from_ms > raw.outcome_observed_at_ms
        || measurement
            .valid_until_ms
            .is_some_and(|end| raw.outcome_observed_at_ms >= end)
    {
        return Err(IngestReject::InvalidMeasurementBound);
    }

    let causal = verifier
        .verify_causal_assessment(
            &raw.causal_assessment_receipt_id,
            raw.event_id,
            raw.scope,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedCausalAssessment)?;
    if !is_sha256_hex(&causal.assessment_digest) {
        return Err(IngestReject::UnverifiedCausalAssessment);
    }
    if !causal.identification.semantic_estimand.is_complete() {
        return Err(IngestReject::IncompleteSemanticEstimand);
    }
    if causal.identification.semantic_estimand.canonical_digest()
        != causal.identification.estimand_digest
    {
        return Err(IngestReject::SemanticEstimandDigestMismatch);
    }
    let outcome = verifier
        .verify_outcome(
            &raw.outcome_receipt_id,
            raw.event_id,
            raw.outcome_observed_at_ms,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedOutcome)?;
    if !is_sha256_hex(&outcome.outcome_digest) {
        return Err(IngestReject::UnverifiedOutcome);
    }
    if !outcome.observed_utility.is_finite()
        || !(0.0..=1.0).contains(&outcome.observed_utility)
    {
        return Err(IngestReject::InvalidVerifiedOutcomeValue);
    }
    if outcome.utility_unit != measurement.unit {
        return Err(IngestReject::OutcomeUnitMismatch);
    }
    if raw.observed_utility.to_bits() != outcome.observed_utility.to_bits() {
        return Err(IngestReject::OutcomeValueMismatch);
    }
    if !outcome.reconciled {
        return Err(IngestReject::OutcomeNotReconciled);
    }

    Ok(IngestedExperience {
        experience: Experience {
            event_id: raw.event_id,
            source_family: lineage.source_family,
            source_root_id: lineage.source_root_id,
            source_root_digest: lineage.source_root_digest,
            scope: raw.scope,
            action: raw.action,
            observed_utility: outcome.observed_utility,
            measurement_error_bound: measurement.absolute_error_bound,
            causal_identification: causal.identification,
            intervention: causal.intervention_confirmed,
            prediction_pre_registered: causal.prediction_pre_registered,
            outcome_reconciled: outcome.reconciled,
            source_trusted: lineage.trust_status == SourceTrustStatus::TrustedForScope,
            source_retracted: false,
            confounders_controlled: causal.confounders_controlled,
            delayed_horizon_complete: causal.delayed_horizon_complete,
        },
        as_known: query,
        receipts: EvidenceReceiptBundle {
            raw_event_digest: event.payload_sha256,
            raw_event_receipt_digest: event.receipt_digest,
            session_id: event.session_id,
            task_id: event.task_id,
            canonical_event_hash: event.canonical_event_hash,
            authority_receipt_digest: event.authority_receipt_digest,
            event_occurred_at_ms: event.occurred_at_ms,
            authority_received_at_ms: event.received_at_ms,
            lineage_digest: lineage.receipt_digest,
            source_family: lineage.source_family,
            source_root_id: lineage.source_root_id,
            source_root_digest: lineage.source_root_digest,
            source_acquisition_mode: lineage.acquisition_mode,
            upstream_source_id: lineage.upstream_source_id,
            source_artifact_digest: lineage.source_artifact_digest,
            transformation_digest: lineage.transformation_digest,
            source_issuer_id: lineage.issuer_id,
            source_trust_status: lineage.trust_status,
            source_trust_policy_digest: lineage.trust_policy_digest,
            calibration_digest: measurement.calibration_digest,
            measurement_unit: measurement.unit,
            measurement_instrument_id: measurement.instrument_id,
            calibration_method_id: measurement.method_id,
            calibration_revision: measurement.calibration_revision,
            causal_assessment_digest: causal.assessment_digest,
            causal_identification: causal.identification,
            outcome_success: outcome.success,
            outcome_observed_utility_bits: outcome.observed_utility.to_bits(),
            outcome_utility_unit: outcome.utility_unit,
            outcome_digest: outcome.outcome_digest,
        },
    })
}

#[derive(Clone, Debug)]
pub struct IngestBatch {
    admitted: Vec<IngestedExperience>,
    duplicate_event_ids: Vec<u32>,
    as_known: AsKnownQuery,
}

impl IngestBatch {
    pub fn admitted(&self) -> &[IngestedExperience] {
        &self.admitted
    }

    pub fn duplicate_event_ids(&self) -> &[u32] {
        &self.duplicate_event_ids
    }

    pub fn as_known(&self) -> AsKnownQuery {
        self.as_known
    }
}

#[derive(Clone, Debug)]
pub struct ConsolidationEvidenceRefV1 {
    as_known: AsKnownQuery,
    event_id: u32,
    session_id: String,
    task_id: String,
    scope: Scope,
    action: Action,
    observed_utility: f64,
    measurement_error_bound: f64,
    source_family: u16,
    source_root_id: u32,
    source_root_digest: SourceRootDigest,
    source_trust_status: SourceTrustStatus,
    causal_identification: CausalIdentificationV1,
    intervention: bool,
    prediction_pre_registered: bool,
    outcome_reconciled: bool,
    confounders_controlled: bool,
    delayed_horizon_complete: bool,
    outcome_success: bool,
    receipts: EvidenceReceiptBundle,
}

impl ConsolidationEvidenceRefV1 {
    pub fn as_known(&self) -> AsKnownQuery {
        self.as_known
    }

    /// Candidate-visible projection reconstructed from the receipt-preserving
    /// evidence reference. This does not restore trust by caller assertion:
    /// every field originates in the previously admitted receipt bundle.
    pub fn experience(&self) -> Experience {
        Experience {
            event_id: self.event_id,
            source_family: self.source_family,
            source_root_id: self.source_root_id,
            source_root_digest: self.source_root_digest,
            scope: self.scope,
            action: self.action,
            observed_utility: self.observed_utility,
            measurement_error_bound: self.measurement_error_bound,
            causal_identification: self.causal_identification,
            intervention: self.intervention,
            prediction_pre_registered: self.prediction_pre_registered,
            outcome_reconciled: self.outcome_reconciled,
            source_trusted: self.source_trust_status == SourceTrustStatus::TrustedForScope,
            source_retracted: false,
            confounders_controlled: self.confounders_controlled,
            delayed_horizon_complete: self.delayed_horizon_complete,
        }
    }

    pub fn event_id(&self) -> u32 {
        self.event_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn action(&self) -> Action {
        self.action
    }

    pub fn observed_utility(&self) -> f64 {
        self.observed_utility
    }

    pub fn measurement_error_bound(&self) -> f64 {
        self.measurement_error_bound
    }

    pub fn source_family(&self) -> u16 {
        self.source_family
    }

    pub fn source_root_id(&self) -> u32 {
        self.source_root_id
    }

    pub fn source_root_digest(&self) -> SourceRootDigest {
        self.source_root_digest
    }

    pub fn source_trust_status(&self) -> SourceTrustStatus {
        self.source_trust_status
    }

    pub fn intervention(&self) -> bool {
        self.intervention
    }

    pub fn prediction_pre_registered(&self) -> bool {
        self.prediction_pre_registered
    }

    pub fn outcome_reconciled(&self) -> bool {
        self.outcome_reconciled
    }

    pub fn confounders_controlled(&self) -> bool {
        self.confounders_controlled
    }

    pub fn delayed_horizon_complete(&self) -> bool {
        self.delayed_horizon_complete
    }

    pub fn outcome_success(&self) -> bool {
        self.outcome_success
    }

    pub fn receipts(&self) -> &EvidenceReceiptBundle {
        &self.receipts
    }
}

/// Immutable evidence input to a consolidation decision. This contains no
/// proposed action, oracle label, or authorization grant.
/// Exact source revision attached to a candidate consolidation proposal.
/// The receipt bundle is copied intact; IDs or summaries alone are insufficient.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsolidationSourceBindingV1 {
    pub event_id: u32,
    pub action: Action,
    pub scope: Scope,
    pub measurement_error_bound_bits: u64,
    pub receipts: EvidenceReceiptBundle,
}

#[derive(Clone, Debug)]
pub struct ConsolidationEvidencePackV1 {
    as_known: AsKnownQuery,
    events: Vec<ConsolidationEvidenceRefV1>,
    source_family_count: usize,
    source_root_count: usize,
    trusted_event_count: usize,
    untrusted_event_count: usize,
    unknown_trust_event_count: usize,
}

impl ConsolidationEvidencePackV1 {
    /// Summarize verified support for a target scope without making a decision.
    /// A success requires trusted lineage, a verified success outcome,
    /// intervention, pre-registration, confounder control, reconciliation and
    /// a completed delayed horizon.
    pub fn summarize_for_scope(
        &self,
        target_scope: Scope,
    ) -> ConsolidationEvidenceSummaryV1 {
        let scoped: Vec<&ConsolidationEvidenceRefV1> = self
            .events
            .iter()
            .filter(|event| event.scope == target_scope)
            .collect();
        let trusted: Vec<&ConsolidationEvidenceRefV1> = scoped
            .iter()
            .copied()
            .filter(|event| {
                event.source_trust_status == SourceTrustStatus::TrustedForScope
            })
            .collect();
        let successful: Vec<&ConsolidationEvidenceRefV1> = trusted
            .iter()
            .copied()
            .filter(|event| {
                event.outcome_success
                    && event.causal_identification.qualifies_for_transfer()
                    && event.intervention
                    && event.prediction_pre_registered
                    && event.outcome_reconciled
                    && event.confounders_controlled
                    && event.delayed_horizon_complete
            })
            .collect();
        let trusted_families: BTreeSet<u16> =
            trusted.iter().map(|event| event.source_family).collect();
        let trusted_roots: BTreeSet<SourceRootDigest> =
            trusted.iter().map(|event| event.source_root_digest).collect();
        let successful_families: BTreeSet<u16> =
            successful.iter().map(|event| event.source_family).collect();
        let successful_roots: BTreeSet<SourceRootDigest> =
            successful.iter().map(|event| event.source_root_digest).collect();
        let all_trusted_predictions_pre_registered = !trusted.is_empty()
            && trusted.iter().all(|event| event.prediction_pre_registered);

        ConsolidationEvidenceSummaryV1 {
            as_known: self.as_known,
            target_scope,
            episode_count: self.events.len(),
            target_scope_event_count: scoped.len(),
            trusted_source_family_count: trusted_families.len(),
            trusted_source_root_count: trusted_roots.len(),
            independently_observed_successes: successful.len(),
            qualified_success_source_family_count: successful_families.len(),
            qualified_success_source_root_count: successful_roots.len(),
            untrusted_source_count: scoped
                .iter()
                .filter(|event| event.source_trust_status == SourceTrustStatus::Untrusted)
                .count(),
            unknown_trust_source_count: scoped
                .iter()
                .filter(|event| event.source_trust_status == SourceTrustStatus::Unknown)
                .count(),
            scope_matches: !self.events.is_empty() && scoped.len() == self.events.len(),
            all_trusted_predictions_pre_registered,
        }
    }

    /// Summarize one candidate action in one exact scope. Independent
    /// support is calculated only over fully qualified successful events and
    /// uses distinct verified source families AND upstream roots.
    pub fn summarize_action_for_scope(
        &self,
        target_scope: Scope,
        action: Action,
    ) -> ActionConsolidationEvidenceSummaryV1 {
        let action_events: Vec<&ConsolidationEvidenceRefV1> = self
            .events
            .iter()
            .filter(|event| event.action == action)
            .collect();
        let scoped: Vec<&ConsolidationEvidenceRefV1> = action_events
            .iter()
            .copied()
            .filter(|event| event.scope == target_scope)
            .collect();
        let trusted: Vec<&ConsolidationEvidenceRefV1> = scoped
            .iter()
            .copied()
            .filter(|event| {
                event.source_trust_status == SourceTrustStatus::TrustedForScope
            })
            .collect();
        let successful: Vec<&ConsolidationEvidenceRefV1> = trusted
            .iter()
            .copied()
            .filter(|event| {
                event.outcome_success
                    && event.causal_identification.qualifies_for_transfer()
                    && event.intervention
                    && event.prediction_pre_registered
                    && event.outcome_reconciled
                    && event.confounders_controlled
                    && event.delayed_horizon_complete
            })
            .collect();

        let trusted_families: BTreeSet<u16> =
            trusted.iter().map(|event| event.source_family).collect();
        let trusted_roots: BTreeSet<SourceRootDigest> =
            trusted.iter().map(|event| event.source_root_digest).collect();
        let successful_families: BTreeSet<u16> =
            successful.iter().map(|event| event.source_family).collect();
        let successful_roots: BTreeSet<SourceRootDigest> =
            successful.iter().map(|event| event.source_root_digest).collect();
        let max_success_measurement_error_bound = successful
            .iter()
            .map(|event| event.measurement_error_bound)
            .reduce(f64::max);
        let all_trusted_predictions_pre_registered = !trusted.is_empty()
            && trusted.iter().all(|event| event.prediction_pre_registered);

        ActionConsolidationEvidenceSummaryV1 {
            as_known: self.as_known,
            target_scope,
            action,
            action_episode_count: action_events.len(),
            target_scope_event_count: scoped.len(),
            trusted_event_count: trusted.len(),
            trusted_source_family_count: trusted_families.len(),
            trusted_source_root_count: trusted_roots.len(),
            qualified_success_event_count: successful.len(),
            qualified_success_source_family_count: successful_families.len(),
            qualified_success_source_root_count: successful_roots.len(),
            untrusted_event_count: scoped
                .iter()
                .filter(|event| event.source_trust_status == SourceTrustStatus::Untrusted)
                .count(),
            unknown_trust_event_count: scoped
                .iter()
                .filter(|event| event.source_trust_status == SourceTrustStatus::Unknown)
                .count(),
            qualified_success_event_ids: successful
                .iter()
                .map(|event| event.event_id)
                .collect(),
            max_qualified_success_measurement_error_bound,
            scope_matches: !action_events.is_empty()
                && scoped.len() == action_events.len(),
            all_trusted_predictions_pre_registered,
        }
    }

    /// Resolve an exact ordered set of promotable event revisions into
    /// complete source bindings. This never reconstructs trust from IDs.
    pub fn source_bindings_for_event_ids(
        &self,
        event_ids: &[u32],
    ) -> Result<Vec<ConsolidationSourceBindingV1>, ConsolidationPackReject> {
        if event_ids.is_empty() {
            return Err(ConsolidationPackReject::EmptySelection);
        }

        let mut seen = BTreeSet::new();
        let mut bindings = Vec::with_capacity(event_ids.len());
        for event_id in event_ids {
            if !seen.insert(*event_id) {
                return Err(ConsolidationPackReject::DuplicateRequestedEvent(*event_id));
            }
            let event = self
                .events
                .iter()
                .find(|event| event.event_id == *event_id)
                .ok_or(ConsolidationPackReject::MissingAdmittedEvent(*event_id))?;
            let eligible = event.source_trust_status == SourceTrustStatus::TrustedForScope
                && event.source_family != 0
                && event.source_root_digest.is_known()
                && event.outcome_success
                && event.causal_identification.qualifies_for_transfer()
                && event.intervention
                && event.prediction_pre_registered
                && event.outcome_reconciled
                && event.confounders_controlled
                && event.delayed_horizon_complete
                && event.measurement_error_bound.is_finite()
                && (0.0..=1.0).contains(&event.measurement_error_bound)
                && event.receipts.outcome_success
                && event.receipts.source_root_digest == event.source_root_digest;
            if !eligible {
                return Err(ConsolidationPackReject::UnqualifiedPromotionEvent(
                    *event_id,
                ));
            }
            bindings.push(ConsolidationSourceBindingV1 {
                event_id: event.event_id,
                action: event.action,
                scope: event.scope,
                measurement_error_bound_bits: event.measurement_error_bound.to_bits(),
                receipts: event.receipts.clone(),
            });
        }
        Ok(bindings)
    }

    pub fn events(&self) -> &[ConsolidationEvidenceRefV1] {
        &self.events
    }

    pub fn source_family_count(&self) -> usize {
        self.source_family_count
    }

    pub fn source_root_count(&self) -> usize {
        self.source_root_count
    }

    /// Number of events whose source was trusted under the identified policy.
    pub fn trusted_event_count(&self) -> usize {
        self.trusted_event_count
    }

    pub fn untrusted_event_count(&self) -> usize {
        self.untrusted_event_count
    }

    pub fn unknown_trust_event_count(&self) -> usize {
        self.unknown_trust_event_count
    }
}

/// Decision-neutral support for one action in one scope. Counts are
/// derived from verified ingress records; no retain/promote decision is made.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionConsolidationEvidenceSummaryV1 {
    pub as_known: AsKnownQuery,
    pub target_scope: Scope,
    pub action: Action,
    pub action_episode_count: usize,
    pub target_scope_event_count: usize,
    pub trusted_event_count: usize,
    pub trusted_source_family_count: usize,
    pub trusted_source_root_count: usize,
    pub qualified_success_event_count: usize,
    pub qualified_success_source_family_count: usize,
    pub qualified_success_source_root_count: usize,
    pub untrusted_event_count: usize,
    pub unknown_trust_event_count: usize,
    pub qualified_success_event_ids: Vec<u32>,
    /// Worst verified absolute error bound among qualified successful events.
    pub max_qualified_success_measurement_error_bound: Option<f64>,
    pub scope_matches: bool,
    pub all_trusted_predictions_pre_registered: bool,
}

/// Decision-neutral summary of a verified evidence pack. It carries no
/// consolidation action, evaluator oracle, or authorization grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsolidationEvidenceSummaryV1 {
    pub as_known: AsKnownQuery,
    pub target_scope: Scope,
    /// Total selected pack size, including out-of-scope evidence.
    pub episode_count: usize,
    /// Events matching target_scope; support counts below use only these events.
    pub target_scope_event_count: usize,
    pub trusted_source_family_count: usize,
    pub trusted_source_root_count: usize,
    pub independently_observed_successes: usize,
    pub qualified_success_source_family_count: usize,
    pub qualified_success_source_root_count: usize,
    pub untrusted_source_count: usize,
    pub unknown_trust_source_count: usize,
    pub scope_matches: bool,
    pub all_trusted_predictions_pre_registered: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsolidationPackReject {
    EmptySelection,
    DuplicateRequestedEvent(u32),
    MissingAdmittedEvent(u32),
    DuplicateAdmittedEvent(u32),
    UnqualifiedPromotionEvent(u32),
    InconsistentKnowledgeSnapshot,
}

/// Build a deterministic, receipt-preserving input pack for consolidation.
/// It derives source diversity from ingress-issued records; it does not decide
/// whether any abstraction should be promoted or grant durable-memory authority.
pub fn build_consolidation_evidence_pack(
    batch: &IngestBatch,
    requested_event_ids: &[u32],
) -> Result<ConsolidationEvidencePackV1, ConsolidationPackReject> {
    if requested_event_ids.is_empty() {
        return Err(ConsolidationPackReject::EmptySelection);
    }

    let mut admitted_by_id = BTreeMap::<u32, &IngestedExperience>::new();
    for item in &batch.admitted {
        let event_id = item.experience.event_id;
        if admitted_by_id.insert(event_id, item).is_some() {
            return Err(ConsolidationPackReject::DuplicateAdmittedEvent(event_id));
        }
        if item.as_known != batch.as_known {
            return Err(ConsolidationPackReject::InconsistentKnowledgeSnapshot);
        }
    }

    let mut requested = BTreeSet::new();
    let mut events = Vec::with_capacity(requested_event_ids.len());
    let mut source_families = BTreeSet::new();
    let mut source_roots = BTreeSet::<SourceRootDigest>::new();
    let mut trusted_event_count = 0usize;
    let mut untrusted_event_count = 0usize;
    let mut unknown_trust_event_count = 0usize;

    for event_id in requested_event_ids {
        if !requested.insert(*event_id) {
            return Err(ConsolidationPackReject::DuplicateRequestedEvent(*event_id));
        }
        let item = admitted_by_id
            .get(event_id)
            .ok_or(ConsolidationPackReject::MissingAdmittedEvent(*event_id))?;
        let experience = &item.experience;
        if experience.source_family == 0 || !experience.source_root_digest.is_known() {
            return Err(ConsolidationPackReject::InconsistentKnowledgeSnapshot);
        }
        match item.receipts.source_trust_status {
            SourceTrustStatus::TrustedForScope => {
                source_families.insert(experience.source_family);
                source_roots.insert(experience.source_root_digest);
                trusted_event_count += 1;
            }
            SourceTrustStatus::Untrusted => untrusted_event_count += 1,
            SourceTrustStatus::Unknown => unknown_trust_event_count += 1,
        }
        events.push(ConsolidationEvidenceRefV1 {
            as_known: item.as_known,
            event_id: experience.event_id,
            session_id: item.receipts.session_id.clone(),
            task_id: item.receipts.task_id.clone(),
            scope: experience.scope,
            action: experience.action,
            observed_utility: experience.observed_utility,
            measurement_error_bound: experience.measurement_error_bound,
            source_family: experience.source_family,
            source_root_id: experience.source_root_id,
            source_root_digest: experience.source_root_digest,
            source_trust_status: item.receipts.source_trust_status,
            causal_identification: experience.causal_identification,
            intervention: experience.intervention,
            prediction_pre_registered: experience.prediction_pre_registered,
            outcome_reconciled: experience.outcome_reconciled,
            confounders_controlled: experience.confounders_controlled,
            delayed_horizon_complete: experience.delayed_horizon_complete,
            outcome_success: item.receipts.outcome_success,
            receipts: item.receipts.clone(),
        });
    }

    events.sort_by_key(|item| item.event_id);
    Ok(ConsolidationEvidencePackV1 {
        as_known: batch.as_known,
        events,
        source_family_count: source_families.len(),
        source_root_count: source_roots.len(),
        trusted_event_count,
        untrusted_event_count,
        unknown_trust_event_count,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestBatchReject {
    ConflictingPayloadForEvent(u32),
    ConflictingReceiptForEvent(u32),
    ConflictingRootForUpstreamSource(u32),
    EventRejected(u32, IngestReject),
}

/// Atomic as-known batch ingress. Same event ID + digest is idempotent;
/// one event ID with different payload digests rejects the entire batch.
pub fn ingest_batch_as_known(
    raw_events: &[RawInteractionV1],
    query: AsKnownQuery,
    verifier: &impl EvidenceVerifier,
) -> Result<IngestBatch, IngestBatchReject> {
    let mut digests = BTreeMap::<u32, &str>::new();
    let mut receipt_bundles = BTreeMap::<u32, EvidenceReceiptBundle>::new();
    let mut root_by_upstream_id = BTreeMap::<String, SourceRootDigest>::new();
    let mut duplicate_event_ids = Vec::new();
    let mut admitted = Vec::new();

    for raw in raw_events {
        let ingested = ingest_as_known(raw, query, verifier)
            .map_err(|reason| IngestBatchReject::EventRejected(raw.event_id, reason))?;
        if let Some(existing_digest) = digests.get(&raw.event_id) {
            if *existing_digest != raw.payload_sha256.as_str() {
                return Err(IngestBatchReject::ConflictingPayloadForEvent(
                    raw.event_id,
                ));
            }
            if receipt_bundles.get(&raw.event_id) != Some(&ingested.receipts) {
                return Err(IngestBatchReject::ConflictingReceiptForEvent(raw.event_id));
            }
            duplicate_event_ids.push(raw.event_id);
            continue;
        }
        if root_by_upstream_id
            .get(&ingested.receipts.upstream_source_id)
            .is_some_and(|root| *root != ingested.receipts.source_root_digest)
        {
            return Err(IngestBatchReject::ConflictingRootForUpstreamSource(
                raw.event_id,
            ));
        }
        root_by_upstream_id.insert(
            ingested.receipts.upstream_source_id.clone(),
            ingested.receipts.source_root_digest,
        );
        digests.insert(raw.event_id, raw.payload_sha256.as_str());
        receipt_bundles.insert(raw.event_id, ingested.receipts.clone());
        admitted.push(ingested);
    }

    Ok(IngestBatch {
        admitted,
        duplicate_event_ids,
        as_known: query,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{is_qualified, Objective, Query, Regime};

    fn fixture_root_digest(seed: u32) -> SourceRootDigest {
        let mut bytes = [0_u8; 32];
        bytes[..4].copy_from_slice(&seed.to_be_bytes());
        SourceRootDigest::from_bytes(bytes)
    }

    #[derive(Clone)]
    fn test_evidence_digest(seed: u32) -> broca_mace_policy::EvidenceDigest256 {
        let mut bytes = [0_u8; 32];
        bytes[..4].copy_from_slice(&seed.to_be_bytes());
        broca_mace_policy::EvidenceDigest256::from_bytes(bytes)
    }

    fn test_semantic_estimand() -> SemanticEstimandV1 {
        SemanticEstimandV1::new(
            1,
            test_evidence_digest(20),
            test_evidence_digest(21),
            test_evidence_digest(22),
            test_evidence_digest(23),
            test_evidence_digest(24),
            30_000,
            test_evidence_digest(25),
            test_evidence_digest(26),
            test_evidence_digest(27),
        )
    }

    fn test_causal_identification() -> CausalIdentificationV1 {
        CausalIdentificationV1::new(
            broca_mace_policy::IdentificationMethodV1::RandomizedControlledTrial,
            test_evidence_digest(1),
            test_semantic_estimand().canonical_digest(),
            5_000,
            5_000,
            50_000,
            true,
        )
        .with_semantic_estimand(test_semantic_estimand())
    }

    struct TestVerifier {
        root_id: u32,
        root_digest: SourceRootDigest,
        family_id: u16,
        error_bound: f64,
        reconciled: bool,
        outcome_success: bool,
        verified_outcome_utility: f64,
        verified_outcome_unit: String,
        verified_outcome_by_event: BTreeMap<u32, f64>,
        intervention: bool,
        delayed_complete: bool,
        causal_identification: CausalIdentificationV1,
        retracted_at_ms: Option<i64>,
        superseded_at_ms: Option<i64>,
        successor_event_id: u32,
        lineage_valid: bool,
        calibration_valid: bool,
        raw_event_valid: bool,
        raw_event_mismatch: bool,
        raw_event_received_at_ms: i64,
        authority_receipt_variant: bool,
        authority_status: AuthorityStatus,
        retraction_status_known: bool,
        vary_lineage_by_event: bool,
        vary_family_by_event: bool,
        trust_status: SourceTrustStatus,
        untrusted_event_id: Option<u32>,
        unknown_trust_event_id: Option<u32>,
        trust_policy_digest_valid: bool,
        vary_root_without_source_id: bool,
        vary_root_digest_by_event: bool,
        vary_upstream_source_by_event: bool,
        unknown_root_digest: bool,
        lifecycle_event_id: Option<u32>,
    }

    impl Default for TestVerifier {
        fn default() -> Self {
            Self {
                root_id: 700,
                root_digest: fixture_root_digest(700),
                family_id: 7,
                error_bound: 0.02,
                reconciled: true,
                outcome_success: true,
                verified_outcome_utility: 0.87,
                verified_outcome_unit: "normalized_utility".into(),
                verified_outcome_by_event: BTreeMap::new(),
                intervention: true,
                delayed_complete: true,
                causal_identification: test_causal_identification(),
                retracted_at_ms: None,
                superseded_at_ms: None,
                successor_event_id: 43,
                lineage_valid: true,
                calibration_valid: true,
                raw_event_valid: true,
                raw_event_mismatch: false,
                raw_event_received_at_ms: 220,
                authority_receipt_variant: false,
                authority_status: AuthorityStatus::ReceiptAttested,
                retraction_status_known: true,
                vary_lineage_by_event: false,
                vary_family_by_event: false,
                trust_status: SourceTrustStatus::TrustedForScope,
                untrusted_event_id: None,
                unknown_trust_event_id: None,
                trust_policy_digest_valid: true,
                vary_root_without_source_id: false,
                vary_root_digest_by_event: false,
                vary_upstream_source_by_event: false,
                unknown_root_digest: false,
                lifecycle_event_id: None,
            }
        }
    }

    impl EvidenceVerifier for TestVerifier {
        fn verify_raw_event(
            &self,
            _receipt_id: &str,
            raw: &RawInteractionV1,
            _known_at_ms: i64,
        ) -> Option<VerifiedRawEvent> {
            if !self.raw_event_valid {
                return None;
            }
            Some(VerifiedRawEvent {
                event_id: raw.event_id,
                session_id: raw.session_id.clone(),
                task_id: raw.task_id.clone(),
                payload_sha256: if self.raw_event_mismatch {
                    "b".repeat(64)
                } else {
                    raw.payload_sha256.clone()
                },
                canonical_event_hash: "c".repeat(64),
                authority_status: self.authority_status,
                authority_receipt_digest: if self.authority_receipt_variant {
                    "9".repeat(64)
                } else {
                    "b".repeat(64)
                },
                valid_from_ms: raw.valid_from_ms,
                valid_until_ms: raw.valid_until_ms,
                occurred_at_ms: raw.occurred_at_ms,
                received_at_ms: self.raw_event_received_at_ms,
                receipt_digest: "a".repeat(64),
            })
        }

        fn verify_lineage(
            &self,
            _receipt_id: &str,
            event_id: u32,
            _scope: Scope,
            _known_at_ms: i64,
        ) -> Option<VerifiedLineage> {
            self.lineage_valid.then(|| VerifiedLineage {
                source_family: if self.vary_lineage_by_event || self.vary_family_by_event {
                    u16::try_from(event_id).unwrap_or(self.family_id)
                } else {
                    self.family_id
                },
                source_root_id: if self.vary_root_without_source_id {
                    2_000 + event_id
                } else if self.vary_lineage_by_event {
                    1_000 + event_id
                } else {
                    self.root_id
                },
                source_root_digest: if self.unknown_root_digest {
                    SourceRootDigest::UNKNOWN
                } else if self.vary_root_digest_by_event {
                    fixture_root_digest(3_000 + event_id)
                } else if self.vary_root_without_source_id {
                    fixture_root_digest(2_000 + event_id)
                } else if self.vary_lineage_by_event {
                    fixture_root_digest(1_000 + event_id)
                } else {
                    self.root_digest
                },
                trust_status: if self.untrusted_event_id == Some(event_id) {
                    SourceTrustStatus::Untrusted
                } else if self.unknown_trust_event_id == Some(event_id) {
                    SourceTrustStatus::Unknown
                } else {
                    self.trust_status
                },
                trust_policy_digest: if self.trust_policy_digest_valid {
                    "8".repeat(64)
                } else {
                    "invalid".into()
                },
                acquisition_mode: "api".into(),
                upstream_source_id: if self.vary_lineage_by_event
                    || self.vary_upstream_source_by_event
                {
                    format!("synthetic:independent-source:{event_id}")
                } else {
                    "bis:dataflow:private-credit".into()
                },
                source_artifact_digest: "e".repeat(64),
                transformation_digest: "f".repeat(64),
                issuer_id: "did:key:source-ledger".into(),
                receipt_digest: "d".repeat(64),
            })
        }

        fn verify_measurement(
            &self,
            _receipt_id: &str,
            expected_unit: &str,
            _scope: Scope,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<VerifiedMeasurement> {
            self.calibration_valid.then(|| VerifiedMeasurement {
                absolute_error_bound: self.error_bound,
                unit: expected_unit.into(),
                method_id: "utility-method-v1".into(),
                calibration_revision: "calibration-rev-1".into(),
                valid_from_ms: 100,
                valid_until_ms: Some(300),
                instrument_id: "sensor-17".into(),
                calibration_digest: "1".repeat(64),
            })
        }

        fn verify_causal_assessment(
            &self,
            _receipt_id: &str,
            _event_id: u32,
            _scope: Scope,
            _known_at_ms: i64,
        ) -> Option<VerifiedCausalAssessment> {
            Some(VerifiedCausalAssessment {
                intervention_confirmed: self.intervention,
                prediction_pre_registered: true,
                confounders_controlled: true,
                delayed_horizon_complete: self.delayed_complete,
                identification: self.causal_identification,
                assessment_digest: "2".repeat(64),
            })
        }

        fn verify_outcome(
            &self,
            _receipt_id: &str,
            event_id: u32,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<VerifiedOutcome> {
            Some(VerifiedOutcome {
                reconciled: self.reconciled,
                success: self.outcome_success,
                observed_utility: self
                    .verified_outcome_by_event
                    .get(&event_id)
                    .copied()
                    .unwrap_or(self.verified_outcome_utility),
                utility_unit: self.verified_outcome_unit.clone(),
                outcome_digest: "3".repeat(64),
            })
        }

        fn lifecycle_status_as_of(
            &self,
            event_id: u32,
            known_at_ms: i64,
        ) -> Option<SourceLifecycleStatus> {
            if !self.retraction_status_known {
                return None;
            }
            let applies_to_event = self
                .lifecycle_event_id
                .map_or(true, |target| target == event_id);
            if applies_to_event
                && self.retracted_at_ms.is_some_and(|at| at <= known_at_ms)
            {
                return Some(SourceLifecycleStatus::Retracted);
            }
            if applies_to_event
                && self.superseded_at_ms.is_some_and(|at| at <= known_at_ms)
            {
                return Some(SourceLifecycleStatus::Superseded {
                    successor_event_id: self.successor_event_id,
                });
            }
            Some(SourceLifecycleStatus::Active)
        }
    }

    fn scope() -> Scope {
        Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        }
    }

    fn raw() -> RawInteractionV1 {
        RawInteractionV1 {
            event_id: 42,
            session_id: "session-a".into(),
            task_id: "task-a".into(),
            payload_sha256: "a".repeat(64),
            raw_event_receipt_id: "raw-event-1".into(),
            scope: scope(),
            action: Action::A,
            observed_utility: 0.87,
            utility_unit: "normalized_utility".into(),
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            occurred_at_ms: 170,
            prediction_registered_at_ms: Some(150),
            action_dispatched_at_ms: Some(170),
            outcome_observed_at_ms: 200,
            lineage_receipt_id: "lineage-1".into(),
            calibration_receipt_id: "calibration-1".into(),
            causal_assessment_receipt_id: "causal-1".into(),
            outcome_receipt_id: "outcome-1".into(),
        }
    }

    fn query(valid_at_ms: i64, known_at_ms: i64) -> AsKnownQuery {
        AsKnownQuery { valid_at_ms, known_at_ms }
    }

    #[test]
    fn semantic_estimand_receipt_must_be_complete_and_digest_bound() {
        let mut incomplete = test_causal_identification();
        incomplete.semantic_estimand.population_spec_digest =
            broca_mace_policy::EvidenceDigest256::UNKNOWN;
        let verifier = TestVerifier {
            causal_identification: incomplete,
            ..TestVerifier::default()
        };
        assert_eq!(
            ingest_as_known(&raw(), query(180, 250), &verifier).err(),
            Some(IngestReject::IncompleteSemanticEstimand)
        );

        let mut tampered = test_causal_identification();
        tampered.semantic_estimand.follow_up_horizon_ms += 1;
        let verifier = TestVerifier {
            causal_identification: tampered,
            ..TestVerifier::default()
        };
        assert_eq!(
            ingest_as_known(&raw(), query(180, 250), &verifier).err(),
            Some(IngestReject::SemanticEstimandDigestMismatch)
        );
    }

    #[test]
    fn observational_assessment_is_preserved_but_not_causally_qualified() {
        let identification = CausalIdentificationV1::new(
            broca_mace_policy::IdentificationMethodV1::Observational,
            test_evidence_digest(10),
            test_semantic_estimand().canonical_digest(),
            0,
            10_000,
            500_000,
            false,
        )
        .with_semantic_estimand(test_semantic_estimand());
        let verifier = TestVerifier {
            causal_identification: identification,
            ..TestVerifier::default()
        };
        let admitted = ingest_as_known(&raw(), query(180, 250), &verifier)
            .expect("an observational record remains valid as an observation");
        assert_eq!(admitted.experience().causal_identification, identification);
        assert_eq!(admitted.receipts().causal_identification, identification);
        assert!(!broca_mace_policy::is_qualified(
            admitted.experience(),
            broca_mace_policy::Query { scope: scope() }
        ));
    }

    #[test]
    fn untrusted_and_unknown_lineage_are_preserved_but_never_qualified() {
        for status in [SourceTrustStatus::Untrusted, SourceTrustStatus::Unknown] {
            let verifier = TestVerifier {
                trust_status: status,
                ..TestVerifier::default()
            };
            let admitted = ingest_as_known(&raw(), query(180, 250), &verifier)
                .expect("receipt validity is distinct from source trust");
            assert!(!admitted.experience.source_trusted);
            assert!(!is_qualified(
                &admitted.experience,
                Query { scope: scope() }
            ));
            assert_eq!(admitted.receipts.source_trust_status, status);
            assert_eq!(admitted.receipts.source_trust_policy_digest, "8".repeat(64));
        }
    }

    #[test]
    fn candidate_payload_cannot_override_receipt_verified_utility() {
        let mut forged = raw();
        forged.observed_utility = 0.01;
        assert_eq!(
            ingest_as_known(&forged, query(180, 250), &TestVerifier::default()).err(),
            Some(IngestReject::OutcomeValueMismatch)
        );

        let admitted = ingest_as_known(&raw(), query(180, 250), &TestVerifier::default())
            .expect("receipt-bound outcome should be admitted");
        assert_eq!(admitted.experience.observed_utility, 0.87);
        assert_eq!(
            admitted.receipts.outcome_observed_utility_bits,
            0.87_f64.to_bits()
        );
    }

    #[test]
    fn receipt_outcome_unit_must_match_the_verified_calibration_unit() {
        let verifier = TestVerifier {
            verified_outcome_unit: "usd".into(),
            ..TestVerifier::default()
        };
        assert_eq!(
            ingest_as_known(&raw(), query(180, 250), &verifier).err(),
            Some(IngestReject::OutcomeUnitMismatch)
        );
    }

    #[test]
    fn nonfinite_or_out_of_range_receipt_values_are_rejected() {
        for value in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let verifier = TestVerifier {
                verified_outcome_utility: value,
                ..TestVerifier::default()
            };
            assert_eq!(
                ingest_as_known(&raw(), query(180, 250), &verifier).err(),
                Some(IngestReject::InvalidVerifiedOutcomeValue)
            );
        }
    }

    #[test]
    fn trust_classification_requires_a_versioned_policy_digest() {
        let verifier = TestVerifier {
            trust_policy_digest_valid: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &verifier),
            Err(IngestReject::InvalidTrustPolicyDigest)
        ));
    }

    #[test]
    fn consolidation_counts_only_sources_trusted_under_the_policy() {
        let mut second = raw();
        second.event_id = 43;
        second.session_id = "session-b".into();
        second.task_id = "task-b".into();
        second.payload_sha256 = "f".repeat(64);
        second.raw_event_receipt_id = "raw-event-2".into();
        let verifier = TestVerifier {
            vary_lineage_by_event: true,
            untrusted_event_id: Some(43),
            ..TestVerifier::default()
        };
        let batch = ingest_batch_as_known(
            &[raw(), second],
            query(180, 250),
            &verifier,
        )
        .expect("both receipts should be admitted even with different trust states");
        let pack = build_consolidation_evidence_pack(&batch, &[42, 43])
            .expect("pack retains both receipts");

        assert_eq!(pack.events().len(), 2);
        assert_eq!(pack.trusted_event_count(), 1);
        assert_eq!(pack.untrusted_event_count(), 1);
        assert_eq!(pack.unknown_trust_event_count(), 0);
        assert_eq!(pack.source_family_count(), 1);
        assert_eq!(pack.source_root_count(), 1);
        assert_eq!(
            pack.events()
                .iter()
                .find(|event| event.event_id() == 43)
                .unwrap()
                .source_trust_status(),
            SourceTrustStatus::Untrusted
        );
    }

    #[test]
    fn only_receipt_attested_authority_status_is_admissible() {
        for status in [
            AuthorityStatus::PendingReconciliation,
            AuthorityStatus::LegacyUnattested,
            AuthorityStatus::UnsafeUnattested,
            AuthorityStatus::Unknown,
        ] {
            let verifier = TestVerifier {
                authority_status: status,
                ..TestVerifier::default()
            };
            assert!(matches!(
                ingest_as_known(&raw(), query(180, 250), &verifier),
                Err(IngestReject::AuthorityNotAttested)
            ));
        }
        assert!(ingest_as_known(
            &raw(),
            query(180, 250),
            &TestVerifier::default()
        )
        .is_ok());
    }

    #[test]
    fn missing_or_mismatched_raw_event_receipt_is_rejected() {
        let missing = TestVerifier {
            raw_event_valid: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &missing),
            Err(IngestReject::UnverifiedRawEvent)
        ));
        let mismatch = TestVerifier {
            raw_event_mismatch: true,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &mismatch),
            Err(IngestReject::RawEventReceiptMismatch)
        ));
        let mut malformed = raw();
        malformed.payload_sha256 = "bad".into();
        assert!(matches!(
            ingest_as_known(&malformed, query(180, 250), &TestVerifier::default()),
            Err(IngestReject::InvalidPayloadDigest)
        ));
    }

    #[test]
    fn later_supersession_preserves_earlier_as_known_state() {
        let verifier = TestVerifier {
            superseded_at_ms: Some(260),
            successor_event_id: 99,
            ..TestVerifier::default()
        };
        assert!(ingest_as_known(&raw(), query(180, 250), &verifier).is_ok());
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 270), &verifier),
            Err(IngestReject::SupersededAsKnown(99))
        ));
    }

    #[test]
    fn unknown_retraction_status_fails_closed() {
        let verifier = TestVerifier {
            retraction_status_known: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &verifier),
            Err(IngestReject::RetractionStatusUnknown)
        ));
    }

    #[test]
    fn calibration_unit_and_validity_interval_are_enforced() {
        let mut wrong_unit = raw();
        wrong_unit.utility_unit = "dollars".into();
        assert!(matches!(
            ingest_as_known(&wrong_unit, query(180, 250), &TestVerifier::default()),
            Err(IngestReject::InvalidMeasurementBound)
        ));
        let mut after_validity = raw();
        after_validity.outcome_observed_at_ms = 350;
        after_validity.occurred_at_ms = 170;
        assert!(matches!(
            ingest_as_known(
                &after_validity,
                query(180, 400),
                &TestVerifier::default()
            ),
            Err(IngestReject::InvalidMeasurementBound)
        ));
    }

    #[test]
    fn consolidation_pack_preserves_receipts_and_derives_true_source_diversity() {
        let mut second = raw();
        second.event_id = 43;
        second.session_id = "session-b".into();
        second.task_id = "task-b".into();
        second.payload_sha256 = "f".repeat(64);
        second.raw_event_receipt_id = "raw-event-2".into();

        let batch = ingest_batch_as_known(
            &[raw(), second],
            query(180, 250),
            &varied_lineage_verifier(),
        )
        .expect("both source receipts should be admitted");
        let pack = build_consolidation_evidence_pack(&batch, &[43, 42])
            .expect("pack should be built from verified ingress");

        assert_eq!(pack.as_known(), query(180, 250));
        assert_eq!(pack.events().len(), 2);
        assert_eq!(pack.events()[0].event_id(), 42);
        assert_eq!(pack.events()[1].event_id(), 43);
        assert_eq!(pack.source_family_count(), 2);
        assert_eq!(pack.source_root_count(), 2);
        assert_eq!(pack.events()[0].session_id(), "session-a");
        assert_eq!(
            pack.events()[0].receipts().canonical_event_hash,
            "c".repeat(64)
        );
        assert_eq!(
            pack.events()[1].receipts().source_artifact_digest,
            "e".repeat(64)
        );
    }

    #[test]
    fn one_upstream_identifier_cannot_be_split_into_multiple_provenance_roots() {
        let mut second = raw();
        second.event_id = 43;
        second.session_id = "session-a".into();
        second.task_id = "task-a-second".into();
        second.payload_sha256 = "f".repeat(64);
        second.raw_event_receipt_id = "raw-event-2".into();
        let verifier = TestVerifier {
            vary_root_without_source_id: true,
            ..TestVerifier::default()
        };

        assert_eq!(
            ingest_batch_as_known(&[raw(), second], query(180, 250), &verifier).err(),
            Some(IngestBatchReject::ConflictingRootForUpstreamSource(43))
        );
    }

    #[test]
    fn raw_session_a_ingress_session_b_decision_and_session_c_retraction_chain() {
        use broca_mace_policy::{decide, Policy};

        let mut raw_events = Vec::new();
        for (event_id, action, utility) in [
            (100, Action::A, 0.91),
            (101, Action::A, 0.89),
            (102, Action::B, 0.42),
            (103, Action::B, 0.40),
        ] {
            let mut event = raw();
            event.event_id = event_id;
            event.session_id = "session-a".into();
            event.task_id = format!("task-{event_id}");
            event.payload_sha256 = format!("{event_id:064x}");
            event.raw_event_receipt_id = format!("raw-{event_id}");
            event.lineage_receipt_id = format!("lineage-{event_id}");
            event.calibration_receipt_id = format!("calibration-{event_id}");
            event.causal_assessment_receipt_id = format!("causal-{event_id}");
            event.outcome_receipt_id = format!("outcome-{event_id}");
            event.action = action;
            event.observed_utility = utility;
            raw_events.push(event);
        }

        let session_a_verifier = TestVerifier {
            verified_outcome_by_event: BTreeMap::from([
                (100, 0.91),
                (101, 0.89),
                (102, 0.42),
                (103, 0.40),
            ]),
            ..varied_lineage_verifier()
        };
        let batch_a = ingest_batch_as_known(
            &raw_events,
            query(180, 250),
            &session_a_verifier,
        )
        .expect("Session A raw events should be admitted");
        let pack =
            build_consolidation_evidence_pack(&batch_a, &[100, 101, 102, 103])
                .expect("consolidation pack should preserve all four receipts");
        assert_eq!(pack.trusted_event_count(), 4);
        assert_eq!(pack.source_root_count(), 4);

        let action_a_support =
            pack.summarize_action_for_scope(scope(), Action::A);
        let action_b_support =
            pack.summarize_action_for_scope(scope(), Action::B);
        assert_eq!(action_a_support.qualified_success_source_root_count, 2);
        assert_eq!(action_b_support.qualified_success_source_root_count, 2);
        assert_eq!(action_a_support.qualified_success_event_ids, vec![100, 101]);
        assert_eq!(action_b_support.qualified_success_event_ids, vec![102, 103]);

        let memory_b: Vec<Experience> = pack
            .events()
            .iter()
            .map(ConsolidationEvidenceRefV1::experience)
            .collect();
        let query_b = Query { scope: scope() };
        let decision_b = decide(Policy::CausalQualified, &memory_b, query_b);
        assert_eq!(decision_b.action, Some(Action::A));

        let session_c_verifier = TestVerifier {
            vary_lineage_by_event: true,
            verified_outcome_by_event: BTreeMap::from([
                (101, 0.89),
                (102, 0.42),
                (103, 0.40),
            ]),
            retracted_at_ms: Some(260),
            lifecycle_event_id: Some(100),
            ..TestVerifier::default()
        };
        let batch_c = ingest_batch_as_known(
            &raw_events[1..],
            query(180, 270),
            &session_c_verifier,
        )
        .expect("non-retracted Session A records remain eligible");
        let pack_c = build_consolidation_evidence_pack(
            &batch_c,
            &[101, 102, 103],
        )
        .expect("Session C pack should contain only the still-active events");
        let memory_c: Vec<Experience> = pack_c
            .events()
            .iter()
            .map(ConsolidationEvidenceRefV1::experience)
            .collect();

        assert!(!memory_c.iter().any(|item| item.event_id == 100));
        assert_eq!(
            decide(Policy::CausalQualified, &memory_c, query_b).action,
            None
        );
        assert!(decide(Policy::ActiveProbe, &memory_c, query_b)
            .probe
            .is_some());
    }

    #[test]
    fn global_consolidation_summary_counts_only_target_scope_support() {
        let mut quality_event = raw();
        quality_event.event_id = 80;
        quality_event.payload_sha256 = format!("{:064x}", 80);
        quality_event.raw_event_receipt_id = "raw-80".into();
        quality_event.lineage_receipt_id = "lineage-80".into();
        quality_event.calibration_receipt_id = "calibration-80".into();
        quality_event.causal_assessment_receipt_id = "causal-80".into();
        quality_event.outcome_receipt_id = "outcome-80".into();

        let mut cost_event = quality_event.clone();
        cost_event.event_id = 81;
        cost_event.payload_sha256 = format!("{:064x}", 81);
        cost_event.raw_event_receipt_id = "raw-81".into();
        cost_event.lineage_receipt_id = "lineage-81".into();
        cost_event.calibration_receipt_id = "calibration-81".into();
        cost_event.causal_assessment_receipt_id = "causal-81".into();
        cost_event.outcome_receipt_id = "outcome-81".into();
        cost_event.scope.objective = Objective::Cost;

        let batch = ingest_batch_as_known(
            &[quality_event, cost_event],
            query(180, 250),
            &varied_lineage_verifier(),
        )
        .expect("both events should be admitted under their own scopes");
        let pack = build_consolidation_evidence_pack(&batch, &[80, 81])
            .expect("pack should preserve both scope-bound events");
        let summary = pack.summarize_for_scope(scope());

        assert_eq!(summary.episode_count, 2);
        assert_eq!(summary.target_scope_event_count, 1);
        assert_eq!(summary.trusted_source_family_count, 1);
        assert_eq!(summary.trusted_source_root_count, 1);
        assert_eq!(summary.qualified_success_source_family_count, 1);
        assert_eq!(summary.qualified_success_source_root_count, 1);
        assert_eq!(summary.independently_observed_successes, 1);
        assert!(!summary.scope_matches);
    }

    #[test]
    fn action_scoped_summary_never_pools_support_across_actions() {
        let mut raw_events = Vec::new();
        for (event_id, action) in [
            (50, Action::A),
            (51, Action::A),
            (52, Action::B),
        ] {
            let mut event = raw();
            event.event_id = event_id;
            event.session_id = "session-a".into();
            event.task_id = format!("task-{event_id}");
            event.payload_sha256 = format!("{event_id:064x}");
            event.raw_event_receipt_id = format!("raw-{event_id}");
            event.lineage_receipt_id = format!("lineage-{event_id}");
            event.calibration_receipt_id = format!("calibration-{event_id}");
            event.causal_assessment_receipt_id = format!("causal-{event_id}");
            event.outcome_receipt_id = format!("outcome-{event_id}");
            event.action = action;
            raw_events.push(event);
        }
        let batch = ingest_batch_as_known(
            &raw_events,
            query(180, 250),
            &varied_lineage_verifier(),
        )
        .expect("all raw events have independently attested fixture lineage");
        let pack = build_consolidation_evidence_pack(&batch, &[50, 51, 52])
            .expect("evidence pack should preserve all source receipts");

        let a = pack.summarize_action_for_scope(scope(), Action::A);
        let b = pack.summarize_action_for_scope(scope(), Action::B);
        assert_eq!(pack.source_root_count(), 3);
        assert_eq!(a.qualified_success_source_root_count, 2);
        assert_eq!(a.qualified_success_source_family_count, 2);
        assert_eq!(a.qualified_success_event_count, 2);
        assert_eq!(b.qualified_success_source_root_count, 1);
        assert_eq!(b.qualified_success_event_count, 1);
        assert_eq!(b.qualified_success_event_ids, vec![52]);
    }

    #[test]
    fn multiple_family_labels_from_one_root_count_as_one_success_root() {
        let mut first = raw();
        first.event_id = 60;
        first.payload_sha256 = format!("{:064x}", 60);
        first.raw_event_receipt_id = "raw-60".into();
        first.lineage_receipt_id = "lineage-60".into();
        first.calibration_receipt_id = "calibration-60".into();
        first.causal_assessment_receipt_id = "causal-60".into();
        first.outcome_receipt_id = "outcome-60".into();

        let mut second = first.clone();
        second.event_id = 61;
        second.payload_sha256 = format!("{:064x}", 61);
        second.raw_event_receipt_id = "raw-61".into();
        second.lineage_receipt_id = "lineage-61".into();
        second.calibration_receipt_id = "calibration-61".into();
        second.causal_assessment_receipt_id = "causal-61".into();
        second.outcome_receipt_id = "outcome-61".into();

        let verifier = TestVerifier {
            vary_family_by_event: true,
            ..TestVerifier::default()
        };
        let batch = ingest_batch_as_known(
            &[first, second],
            query(180, 250),
            &verifier,
        )
        .expect("same upstream source root can have multiple publisher labels");
        let pack = build_consolidation_evidence_pack(&batch, &[60, 61])
            .expect("pack should retain both independently identified events");
        let summary = pack.summarize_action_for_scope(scope(), Action::A);

        assert_eq!(summary.qualified_success_event_count, 2);
        assert_eq!(summary.qualified_success_source_family_count, 2);
        assert_eq!(summary.qualified_success_source_root_count, 1);
        assert_eq!(summary.max_qualified_success_measurement_error_bound, Some(0.02));
    }

    #[test]
    fn action_summary_isolated_from_other_objectives() {
        let mut event = raw();
        event.event_id = 70;
        event.payload_sha256 = format!("{:064x}", 70);
        event.raw_event_receipt_id = "raw-70".into();
        event.lineage_receipt_id = "lineage-70".into();
        event.calibration_receipt_id = "calibration-70".into();
        event.causal_assessment_receipt_id = "causal-70".into();
        event.outcome_receipt_id = "outcome-70".into();
        event.scope.objective = Objective::Cost;

        let batch = ingest_batch_as_known(
            &[event],
            query(180, 250),
            &TestVerifier::default(),
        )
        .expect("valid event may be admitted in its own scope");
        let pack = build_consolidation_evidence_pack(&batch, &[70])
            .expect("valid receipt pack should be built");
        let summary = pack.summarize_action_for_scope(scope(), Action::A);

        assert_eq!(summary.action_episode_count, 1);
        assert_eq!(summary.target_scope_event_count, 0);
        assert!(!summary.scope_matches);
        assert_eq!(summary.qualified_success_event_count, 0);
        assert_eq!(summary.qualified_success_source_root_count, 0);
    }

    #[test]
    fn consolidation_pack_rejects_duplicate_or_unadmitted_event_ids() {
        let batch = ingest_batch_as_known(
            &[raw()],
            query(180, 250),
            &TestVerifier::default(),
        )
        .expect("source event should be admitted");

        assert_eq!(
            build_consolidation_evidence_pack(&batch, &[]).err(),
            Some(ConsolidationPackReject::EmptySelection)
        );
        assert_eq!(
            build_consolidation_evidence_pack(&batch, &[42, 42]).err(),
            Some(ConsolidationPackReject::DuplicateRequestedEvent(42))
        );
        assert_eq!(
            build_consolidation_evidence_pack(&batch, &[999]).err(),
            Some(ConsolidationPackReject::MissingAdmittedEvent(999))
        );
    }

    #[test]
    fn batch_duplicate_delivery_is_idempotent() {
        let batch = ingest_batch_as_known(
            &[raw(), raw()],
            query(180, 250),
            &TestVerifier::default(),
        )
        .expect("identical duplicate deliveries are idempotent");
        assert_eq!(batch.admitted().len(), 1);
        assert_eq!(batch.duplicate_event_ids(), vec![42]);
    }

    #[test]
    fn duplicate_event_requires_identical_verified_receipt_bundle() {
        struct AlternatingVerifier {
            calls: std::cell::Cell<u32>,
        }

        impl EvidenceVerifier for AlternatingVerifier {
            fn verify_raw_event(
                &self,
                _receipt_id: &str,
                raw: &RawInteractionV1,
                _known_at_ms: i64,
            ) -> Option<VerifiedRawEvent> {
                let call = self.calls.get();
                self.calls.set(call + 1);
                Some(VerifiedRawEvent {
                    event_id: raw.event_id,
                    session_id: raw.session_id.clone(),
                    task_id: raw.task_id.clone(),
                    payload_sha256: raw.payload_sha256.clone(),
                    canonical_event_hash: "c".repeat(64),
                    authority_status: AuthorityStatus::ReceiptAttested,
                    authority_receipt_digest: if call == 0 {
                        "b".repeat(64)
                    } else {
                        "9".repeat(64)
                    },
                    valid_from_ms: raw.valid_from_ms,
                    valid_until_ms: raw.valid_until_ms,
                    occurred_at_ms: raw.occurred_at_ms,
                    received_at_ms: 220,
                    receipt_digest: "a".repeat(64),
                })
            }

            fn verify_lineage(
                &self,
                _receipt_id: &str,
                _event_id: u32,
                _scope: Scope,
                _known_at_ms: i64,
            ) -> Option<VerifiedLineage> {
                Some(VerifiedLineage {
                    source_family: 7,
                    source_root_id: 700,
                    source_root_digest: fixture_root_digest(700),
                    trust_status: SourceTrustStatus::TrustedForScope,
                    trust_policy_digest: "8".repeat(64),
                    acquisition_mode: "api".into(),
                    upstream_source_id: "bis:dataflow:private-credit".into(),
                    source_artifact_digest: "e".repeat(64),
                    transformation_digest: "f".repeat(64),
                    issuer_id: "did:key:source-ledger".into(),
                    receipt_digest: "d".repeat(64),
                })
            }

            fn verify_measurement(
                &self,
                _receipt_id: &str,
                expected_unit: &str,
                _scope: Scope,
                _observed_at_ms: i64,
                _known_at_ms: i64,
            ) -> Option<VerifiedMeasurement> {
                Some(VerifiedMeasurement {
                    absolute_error_bound: 0.02,
                    unit: expected_unit.into(),
                    method_id: "utility-method-v1".into(),
                    calibration_revision: "calibration-rev-1".into(),
                    valid_from_ms: 100,
                    valid_until_ms: Some(300),
                    instrument_id: "sensor-17".into(),
                    calibration_digest: "1".repeat(64),
                })
            }

            fn verify_causal_assessment(
                &self,
                _receipt_id: &str,
                _event_id: u32,
                _scope: Scope,
                _known_at_ms: i64,
            ) -> Option<VerifiedCausalAssessment> {
                Some(VerifiedCausalAssessment {
                    intervention_confirmed: true,
                    prediction_pre_registered: true,
                    confounders_controlled: true,
                    delayed_horizon_complete: true,
                    identification: test_causal_identification(),
                    assessment_digest: "2".repeat(64),
                })
            }

            fn verify_outcome(
                &self,
                _receipt_id: &str,
                _event_id: u32,
                _observed_at_ms: i64,
                _known_at_ms: i64,
            ) -> Option<VerifiedOutcome> {
                Some(VerifiedOutcome {
                    reconciled: true,
                    success: true,
                    observed_utility: 0.87,
                    utility_unit: "normalized_utility".into(),
                    outcome_digest: "3".repeat(64),
                })
            }

            fn lifecycle_status_as_of(
                &self,
                _event_id: u32,
                _known_at_ms: i64,
            ) -> Option<SourceLifecycleStatus> {
                Some(SourceLifecycleStatus::Active)
            }
        }

        let verifier = AlternatingVerifier {
            calls: std::cell::Cell::new(0),
        };
        assert_eq!(
            ingest_batch_as_known(
                &[raw(), raw()],
                query(180, 250),
                &verifier,
            )
            .err(),
            Some(IngestBatchReject::ConflictingReceiptForEvent(42))
        );
        assert_eq!(verifier.calls.get(), 2);
    }

    #[test]
    fn batch_event_id_with_conflicting_payload_is_rejected_atomically() {
        let mut conflicting = raw();
        conflicting.payload_sha256 = "b".repeat(64);
        assert_eq!(
            ingest_batch_as_known(
                &[raw(), conflicting],
                query(180, 250),
                &TestVerifier::default(),
            )
            .err(),
            Some(IngestBatchReject::ConflictingPayloadForEvent(42))
        );
    }

    fn varied_lineage_verifier() -> TestVerifier {
        TestVerifier {
            vary_lineage_by_event: true,
            ..TestVerifier::default()
        }
    }

    #[test]
    fn verified_raw_interaction_becomes_receipted_experience() {
        let admitted = ingest_as_known(&raw(), query(180, 250), &TestVerifier::default())
            .expect("valid evidence should be admitted");
        assert!(is_qualified(&admitted.experience, Query { scope: scope() }));
        assert_eq!(admitted.experience.source_root_id, 700);
        assert_eq!(admitted.experience.measurement_error_bound, 0.02);
        assert_eq!(admitted.as_known, query(180, 250));
        assert_eq!(admitted.receipts.lineage_digest, "d".repeat(64));
        assert_eq!(admitted.receipts.raw_event_digest, raw().payload_sha256);
        assert_eq!(admitted.receipts.canonical_event_hash, "c".repeat(64));
        assert_eq!(admitted.receipts.event_occurred_at_ms, 170);
        assert_eq!(admitted.receipts.authority_received_at_ms, 220);
        assert_eq!(
            admitted.receipts.authority_receipt_digest,
            "b".repeat(64)
        );
        assert_eq!(
            admitted.receipts.calibration_revision,
            "calibration-rev-1"
        );
        assert_eq!(
            admitted.receipts.upstream_source_id,
            "bis:dataflow:private-credit"
        );
        assert_eq!(admitted.receipts.measurement_instrument_id, "sensor-17");
    }

    #[test]
    fn authority_receipt_cannot_predate_the_outcome_it_attests() {
        let verifier = TestVerifier {
            raw_event_received_at_ms: 199,
            ..TestVerifier::default()
        };
        assert_eq!(
            ingest_as_known(&raw(), query(180, 250), &verifier).err(),
            Some(IngestReject::AuthorityReceiptPredatesOutcome)
        );
    }

    #[test]
    fn client_occurred_at_cannot_backdate_known_at_eligibility() {
        // Event time is 170, but the signed server receipt time is 220.
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 210), &TestVerifier::default()),
            Err(IngestReject::RecordedAfterKnownAt)
        ));
        assert!(ingest_as_known(
            &raw(),
            query(180, 220),
            &TestVerifier::default()
        )
        .is_ok());
    }

    #[test]
    fn late_arriving_event_is_absent_from_historical_as_known_query() {
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 210), &TestVerifier::default()),
            Err(IngestReject::RecordedAfterKnownAt)
        ));
        assert!(ingest_as_known(&raw(), query(180, 250), &TestVerifier::default()).is_ok());
    }

    #[test]
    fn later_retraction_does_not_rewrite_earlier_knowledge() {
        let verifier = TestVerifier {
            retracted_at_ms: Some(260),
            ..TestVerifier::default()
        };
        assert!(ingest_as_known(&raw(), query(180, 250), &verifier).is_ok());
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 270), &verifier),
            Err(IngestReject::RetractedAsKnown)
        ));
    }

    #[test]
    fn duplicate_publishers_reusing_one_upstream_root_remain_one_root() {
        let first = TestVerifier {
            family_id: 10,
            root_id: 700,
            ..TestVerifier::default()
        };
        let second = TestVerifier {
            family_id: 11,
            root_id: 701,
            ..TestVerifier::default()
        };
        let a = ingest_as_known(&raw(), query(180, 250), &first).unwrap();
        let b = ingest_as_known(&raw(), query(180, 250), &second).unwrap();
        assert_ne!(a.experience.source_family, b.experience.source_family);
        assert_ne!(a.experience.source_root_id, b.experience.source_root_id);
        assert_eq!(
            a.experience.source_root_digest,
            b.experience.source_root_digest
        );
    }

    #[test]
    fn same_numeric_root_id_with_distinct_verified_digests_is_not_collapsed() {
        let mut first = raw();
        first.event_id = 80;
        first.payload_sha256 = format!("{:064x}", 80);
        first.raw_event_receipt_id = "raw-80".into();
        first.lineage_receipt_id = "lineage-80".into();
        first.calibration_receipt_id = "calibration-80".into();
        first.causal_assessment_receipt_id = "causal-80".into();
        first.outcome_receipt_id = "outcome-80".into();

        let mut second = first.clone();
        second.event_id = 81;
        second.payload_sha256 = format!("{:064x}", 81);
        second.raw_event_receipt_id = "raw-81".into();
        second.lineage_receipt_id = "lineage-81".into();
        second.calibration_receipt_id = "calibration-81".into();
        second.causal_assessment_receipt_id = "causal-81".into();
        second.outcome_receipt_id = "outcome-81".into();

        let verifier = TestVerifier {
            vary_root_digest_by_event: true,
            vary_upstream_source_by_event: true,
            root_id: 700,
            ..TestVerifier::default()
        };
        let batch = ingest_batch_as_known(&[first, second], query(180, 250), &verifier)
            .expect("verified 256-bit roots distinguish numeric-ID collisions");
        let pack = build_consolidation_evidence_pack(&batch, &[80, 81])
            .expect("both verified roots remain in the receipt pack");
        let summary = pack.summarize_action_for_scope(scope(), Action::A);
        assert_eq!(summary.qualified_success_source_root_count, 2);
    }

    #[test]
    fn consolidation_summary_does_not_claim_preregistration_without_trusted_sources() {
        let verifier = TestVerifier {
            trust_status: SourceTrustStatus::Untrusted,
            ..TestVerifier::default()
        };
        let batch = ingest_batch_as_known(
            &[raw()],
            query(180, 250),
            &verifier,
        )
        .expect("untrusted but verified event remains preserved");
        let pack = build_consolidation_evidence_pack(&batch, &[42]).unwrap();
        let summary = pack.summarize_for_scope(scope());

        assert_eq!(summary.episode_count, 1);
        assert_eq!(summary.trusted_source_family_count, 0);
        assert_eq!(summary.trusted_source_root_count, 0);
        assert_eq!(summary.independently_observed_successes, 0);
        assert_eq!(summary.untrusted_source_count, 1);
        assert!(!summary.all_trusted_predictions_pre_registered);
    }

    #[test]
    fn source_lineage_profile_metadata_is_preserved_in_receipt_bundle() {
        let admitted = ingest_as_known(
            &raw(),
            query(180, 250),
            &TestVerifier::default(),
        )
        .expect("verified source profile should be retained");
        assert_eq!(admitted.receipts.source_acquisition_mode, "api");
        assert_eq!(
            admitted.receipts.upstream_source_id,
            "bis:dataflow:private-credit"
        );
        assert_eq!(
            admitted.receipts.source_artifact_digest,
            "e".repeat(64)
        );
    }

    #[test]
    fn unknown_digest_is_rejected_even_when_numeric_root_id_is_present() {
        let verifier = TestVerifier {
            root_id: 700,
            unknown_root_digest: true,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &verifier),
            Err(IngestReject::UnknownSourceRoot)
        ));
    }

    #[test]
    fn unknown_or_unverified_lineage_is_rejected() {
        let unknown = TestVerifier {
            root_id: 0,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &unknown),
            Err(IngestReject::UnknownSourceRoot)
        ));
        let invalid = TestVerifier {
            lineage_valid: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &invalid),
            Err(IngestReject::UnverifiedLineage)
        ));
    }

    #[test]
    fn unverified_or_invalid_calibration_cannot_supply_error_bound() {
        let missing = TestVerifier {
            calibration_valid: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &missing),
            Err(IngestReject::UnverifiedCalibration)
        ));
        let invalid = TestVerifier {
            error_bound: f64::NAN,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &invalid),
            Err(IngestReject::InvalidMeasurementBound)
        ));
    }

    #[test]
    fn tool_reported_success_without_external_reconciliation_is_rejected() {
        let verifier = TestVerifier {
            reconciled: false,
            ..TestVerifier::default()
        };
        assert!(matches!(
            ingest_as_known(&raw(), query(180, 250), &verifier),
            Err(IngestReject::OutcomeNotReconciled)
        ));
    }

    #[test]
    fn prediction_and_dispatch_must_precede_observation() {
        let mut invalid = raw();
        invalid.prediction_registered_at_ms = Some(180);
        invalid.action_dispatched_at_ms = Some(170);
        assert!(matches!(
            ingest_as_known(&invalid, query(180, 250), &TestVerifier::default()),
            Err(IngestReject::PredictionNotBeforeDispatch)
        ));
        invalid.prediction_registered_at_ms = Some(150);
        invalid.action_dispatched_at_ms = Some(205);
        assert!(matches!(
            ingest_as_known(&invalid, query(180, 250), &TestVerifier::default()),
            Err(IngestReject::DispatchNotBeforeOutcome)
        ));
    }

    #[test]
    fn observational_episode_is_preserved_but_not_upgraded_to_intervention() {
        let verifier = TestVerifier {
            intervention: false,
            ..TestVerifier::default()
        };
        let observed = ingest_as_known(&raw(), query(180, 250), &verifier).unwrap();
        assert!(!observed.experience.intervention);
        assert!(!is_qualified(&observed.experience, Query { scope: scope() }));
    }

    #[test]
    fn incomplete_delayed_horizon_is_not_promoted_as_qualified() {
        let verifier = TestVerifier {
            delayed_complete: false,
            ..TestVerifier::default()
        };
        let observed = ingest_as_known(&raw(), query(180, 250), &verifier).unwrap();
        assert!(!observed.experience.delayed_horizon_complete);
        assert!(!is_qualified(&observed.experience, Query { scope: scope() }));
    }
}
