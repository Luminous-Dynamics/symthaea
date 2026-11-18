// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Research-only boundary from raw Session-A interactions to candidate memory.
//!
//! This crate does not implement cryptographic verification. A production
//! EvidenceVerifier must delegate to Mycelix source-ledger, calibration and
//! outcome-reconciliation receipts. Agent-authored labels are not trust anchors.

use std::collections::BTreeMap;

use broca_mace_policy::{Action, Experience, Scope};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedLineage {
    pub source_family: u16,
    pub source_root_id: u32,
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
    pub assessment_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedOutcome {
    pub reconciled: bool,
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
    pub canonical_event_hash: String,
    pub authority_receipt_digest: String,
    pub event_occurred_at_ms: i64,
    pub authority_received_at_ms: i64,
    pub lineage_digest: String,
    pub source_acquisition_mode: String,
    pub upstream_source_id: String,
    pub source_artifact_digest: String,
    pub transformation_digest: String,
    pub source_issuer_id: String,
    pub calibration_digest: String,
    pub measurement_unit: String,
    pub measurement_instrument_id: String,
    pub calibration_method_id: String,
    pub calibration_revision: String,
    pub causal_assessment_digest: String,
    pub outcome_digest: String,
}

#[derive(Clone, Debug)]
pub struct IngestedExperience {
    pub experience: Experience,
    pub as_known: AsKnownQuery,
    pub receipts: EvidenceReceiptBundle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestReject {
    MalformedValidInterval,
    InvalidPayloadDigest,
    UnverifiedRawEvent,
    RawEventReceiptMismatch,
    AuthorityNotAttested,
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
    UnverifiedCalibration,
    InvalidMeasurementBound,
    UnverifiedCausalAssessment,
    UnverifiedOutcome,
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
    if lineage.source_root_id == 0 {
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
    if !outcome.reconciled {
        return Err(IngestReject::OutcomeNotReconciled);
    }

    Ok(IngestedExperience {
        experience: Experience {
            event_id: raw.event_id,
            source_family: lineage.source_family,
            source_root_id: lineage.source_root_id,
            scope: raw.scope,
            action: raw.action,
            observed_utility: raw.observed_utility,
            measurement_error_bound: measurement.absolute_error_bound,
            intervention: causal.intervention_confirmed,
            prediction_pre_registered: causal.prediction_pre_registered,
            outcome_reconciled: outcome.reconciled,
            source_trusted: true,
            source_retracted: false,
            confounders_controlled: causal.confounders_controlled,
            delayed_horizon_complete: causal.delayed_horizon_complete,
        },
        as_known: query,
        receipts: EvidenceReceiptBundle {
            raw_event_digest: event.payload_sha256,
            raw_event_receipt_digest: event.receipt_digest,
            canonical_event_hash: event.canonical_event_hash,
            authority_receipt_digest: event.authority_receipt_digest,
            event_occurred_at_ms: event.occurred_at_ms,
            authority_received_at_ms: event.received_at_ms,
            lineage_digest: lineage.receipt_digest,
            source_acquisition_mode: lineage.acquisition_mode,
            upstream_source_id: lineage.upstream_source_id,
            source_artifact_digest: lineage.source_artifact_digest,
            transformation_digest: lineage.transformation_digest,
            source_issuer_id: lineage.issuer_id,
            calibration_digest: measurement.calibration_digest,
            measurement_unit: measurement.unit,
            measurement_instrument_id: measurement.instrument_id,
            calibration_method_id: measurement.method_id,
            calibration_revision: measurement.calibration_revision,
            causal_assessment_digest: causal.assessment_digest,
            outcome_digest: outcome.outcome_digest,
        },
    })
}

#[derive(Clone, Debug)]
pub struct IngestBatch {
    pub admitted: Vec<IngestedExperience>,
    pub duplicate_event_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestBatchReject {
    ConflictingPayloadForEvent(u32),
    ConflictingReceiptForEvent(u32),
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
        digests.insert(raw.event_id, raw.payload_sha256.as_str());
        receipt_bundles.insert(raw.event_id, ingested.receipts.clone());
        admitted.push(ingested);
    }

    Ok(IngestBatch {
        admitted,
        duplicate_event_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{is_qualified, Objective, Query, Regime};

    #[derive(Clone)]
    struct TestVerifier {
        root_id: u32,
        family_id: u16,
        error_bound: f64,
        reconciled: bool,
        intervention: bool,
        delayed_complete: bool,
        retracted_at_ms: Option<i64>,
        superseded_at_ms: Option<i64>,
        successor_event_id: u32,
        lineage_valid: bool,
        calibration_valid: bool,
        raw_event_valid: bool,
        raw_event_mismatch: bool,
        authority_receipt_variant: bool,
        authority_status: AuthorityStatus,
        retraction_status_known: bool,
    }

    impl Default for TestVerifier {
        fn default() -> Self {
            Self {
                root_id: 700,
                family_id: 7,
                error_bound: 0.02,
                reconciled: true,
                intervention: true,
                delayed_complete: true,
                retracted_at_ms: None,
                superseded_at_ms: None,
                successor_event_id: 43,
                lineage_valid: true,
                calibration_valid: true,
                raw_event_valid: true,
                raw_event_mismatch: false,
                authority_receipt_variant: false,
                authority_status: AuthorityStatus::ReceiptAttested,
                retraction_status_known: true,
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
            self.lineage_valid.then(|| VerifiedLineage {
                source_family: self.family_id,
                source_root_id: self.root_id,
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
                reconciled: self.reconciled,
                outcome_digest: "3".repeat(64),
            })
        }

        fn lifecycle_status_as_of(
            &self,
            _event_id: u32,
            known_at_ms: i64,
        ) -> Option<SourceLifecycleStatus> {
            if !self.retraction_status_known {
                return None;
            }
            if self.retracted_at_ms.is_some_and(|at| at <= known_at_ms) {
                return Some(SourceLifecycleStatus::Retracted);
            }
            if self.superseded_at_ms.is_some_and(|at| at <= known_at_ms) {
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
    fn batch_duplicate_delivery_is_idempotent() {
        let batch = ingest_batch_as_known(
            &[raw(), raw()],
            query(180, 250),
            &TestVerifier::default(),
        )
        .expect("identical duplicate deliveries are idempotent");
        assert_eq!(batch.admitted.len(), 1);
        assert_eq!(batch.duplicate_event_ids, vec![42]);
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
            root_id: 700,
            ..TestVerifier::default()
        };
        let a = ingest_as_known(&raw(), query(180, 250), &first).unwrap();
        let b = ingest_as_known(&raw(), query(180, 250), &second).unwrap();
        assert_ne!(a.experience.source_family, b.experience.source_family);
        assert_eq!(a.experience.source_root_id, b.experience.source_root_id);
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
