// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Research-only boundary from raw Session-A interactions to candidate memory.
//!
//! This crate does not implement cryptographic verification. A production
//! EvidenceVerifier must delegate to Mycelix source-ledger, calibration and
//! outcome-reconciliation receipts. Agent-authored labels are not trust anchors.

use broca_mace_policy::{Action, Experience, Scope};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsKnownQuery {
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
}

/// Raw outcome envelope. Trust-bearing fields are receipt IDs, not booleans.
#[derive(Clone, Debug)]
pub struct RawInteractionV1 {
    pub event_id: u32,
    pub scope: Scope,
    pub action: Action,
    pub observed_utility: f64,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    pub recorded_at_ms: i64,
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
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedMeasurement {
    pub absolute_error_bound: f64,
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

/// Production implementations must verify issuer, signatures, scope,
/// content digests, valid/transaction time, revocation and calibration range.
pub trait EvidenceVerifier {
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

    fn is_retracted_as_of(&self, event_id: u32, known_at_ms: i64) -> bool;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceReceiptBundle {
    pub lineage_digest: String,
    pub calibration_digest: String,
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
    if raw.valid_until_ms.is_some_and(|end| end <= raw.valid_from_ms) {
        return Err(IngestReject::MalformedValidInterval);
    }
    if raw.valid_from_ms > query.valid_at_ms
        || raw.valid_until_ms.is_some_and(|end| query.valid_at_ms >= end)
    {
        return Err(IngestReject::NotValidAtRequestedTime);
    }
    if raw.recorded_at_ms > query.known_at_ms {
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
    if verifier.is_retracted_as_of(raw.event_id, query.known_at_ms) {
        return Err(IngestReject::RetractedAsKnown);
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

    let measurement = verifier
        .verify_measurement(
            &raw.calibration_receipt_id,
            raw.scope,
            raw.outcome_observed_at_ms,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedCalibration)?;
    if !measurement.absolute_error_bound.is_finite()
        || !(0.0..=1.0).contains(&measurement.absolute_error_bound)
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
    let outcome = verifier
        .verify_outcome(
            &raw.outcome_receipt_id,
            raw.event_id,
            raw.outcome_observed_at_ms,
            query.known_at_ms,
        )
        .ok_or(IngestReject::UnverifiedOutcome)?;
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
            lineage_digest: lineage.receipt_digest,
            calibration_digest: measurement.calibration_digest,
            causal_assessment_digest: causal.assessment_digest,
            outcome_digest: outcome.outcome_digest,
        },
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
        lineage_valid: bool,
        calibration_valid: bool,
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
                lineage_valid: true,
                calibration_valid: true,
            }
        }
    }

    impl EvidenceVerifier for TestVerifier {
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
                receipt_digest: "lineage-digest".into(),
            })
        }

        fn verify_measurement(
            &self,
            _receipt_id: &str,
            _scope: Scope,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<VerifiedMeasurement> {
            self.calibration_valid.then(|| VerifiedMeasurement {
                absolute_error_bound: self.error_bound,
                calibration_digest: "calibration-digest".into(),
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
                assessment_digest: "causal-digest".into(),
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
                outcome_digest: "outcome-digest".into(),
            })
        }

        fn is_retracted_as_of(&self, _event_id: u32, known_at_ms: i64) -> bool {
            self.retracted_at_ms
                .is_some_and(|at| at <= known_at_ms)
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
            scope: scope(),
            action: Action::A,
            observed_utility: 0.87,
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            recorded_at_ms: 220,
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
    fn verified_raw_interaction_becomes_receipted_experience() {
        let admitted = ingest_as_known(&raw(), query(180, 250), &TestVerifier::default())
            .expect("valid evidence should be admitted");
        assert!(is_qualified(&admitted.experience, Query { scope: scope() }));
        assert_eq!(admitted.experience.source_root_id, 700);
        assert_eq!(admitted.experience.measurement_error_bound, 0.02);
        assert_eq!(admitted.as_known, query(180, 250));
        assert_eq!(admitted.receipts.lineage_digest, "lineage-digest");
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
