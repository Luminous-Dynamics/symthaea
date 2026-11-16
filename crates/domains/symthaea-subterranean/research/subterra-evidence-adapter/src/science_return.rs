//! Evidence-aware science-return lifecycle.
//!
//! Distinguishes a proposed measurement, execution, acquisition, integrity
//! verification, local persistence, store-and-forward, delivery, and admission
//! to a scientific belief update.
//!
//! The record validator checks event order and required references only.
//! Evidence references are opaque: this module does not authenticate them,
//! verify signatures, prove durable storage, or establish custody authority.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{SiteEvidenceRecord, SiteIssue};

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScienceReturnStage {
    Proposed,
    Executed,
    Acquired,
    IntegrityVerified,
    Persisted,
    StoreAndForwarded,
    Delivered,
    Admitted,
    Failed,
    Cancelled,
}

impl ScienceReturnStage {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Admitted | Self::Failed | Self::Cancelled)
    }

    const fn requires_custody(self) -> bool {
        matches!(
            self,
            Self::Persisted | Self::StoreAndForwarded | Self::Delivered | Self::Admitted
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScienceReturnEvent {
    /// Monotonic, one-based sequence within this observation record.
    pub sequence: u64,
    pub event_id: String,
    pub stage: ScienceReturnStage,
    pub recorded_at: String,
    pub actor_id: String,
    /// Opaque source/evidence reference. Not an authenticity proof.
    pub evidence_ref: String,
    #[serde(default)]
    pub calibration_ref: Option<String>,
    #[serde(default)]
    pub custody_ref: Option<String>,
    #[serde(default)]
    pub delivery_receipt_ref: Option<String>,
    #[serde(default)]
    pub admission_ref: Option<String>,
    #[serde(default)]
    pub disposition_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScienceReturnRecord {
    pub site_id: String,
    pub observation_id: String,
    pub events: Vec<ScienceReturnEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScienceReturnIssue {
    EmptySiteId,
    EmptyObservationId,
    EmptyEventLog,
    InvalidSequence { expected: u64, found: u64 },
    EmptyEventId,
    DuplicateEventId(String),
    EmptyRecordedAt(String),
    EmptyActorId(String),
    EmptyEvidenceRef(String),
    InvalidTransition {
        from: ScienceReturnStage,
        to: ScienceReturnStage,
    },
    TerminalEventNotLast(ScienceReturnStage),
    IntegrityVerificationMissingCalibration(String),
    CustodyReferenceMissing(String),
    DeliveryReceiptMissing(String),
    AdmissionReferenceMissing(String),
    TerminalDispositionMissing(ScienceReturnStage),
    SiteIdMismatch { record: String, source: String },
    ObservationNotInSite(String),
    InvalidSiteEvidence(Vec<SiteIssue>),
}

fn transition_allowed(
    previous: ScienceReturnStage,
    next: ScienceReturnStage,
) -> bool {
    use ScienceReturnStage::{
        Acquired, Admitted, Cancelled, Delivered, Executed, Failed, IntegrityVerified,
        Persisted, Proposed, StoreAndForwarded,
    };

    match previous {
        Proposed => matches!(next, Executed | Failed | Cancelled),
        Executed => matches!(next, Acquired | Failed | Cancelled),
        Acquired => matches!(next, IntegrityVerified | Failed | Cancelled),
        IntegrityVerified => matches!(next, Persisted | Failed | Cancelled),
        Persisted => matches!(next, Delivered | StoreAndForwarded | Failed | Cancelled),
        StoreAndForwarded => matches!(next, Delivered | Failed | Cancelled),
        Delivered => matches!(next, Admitted | Failed | Cancelled),
        Admitted | Failed | Cancelled => false,
    }
}

impl ScienceReturnRecord {
    /// Validate event ordering and stage-specific references.
    ///
    /// The only successful terminal state is Admitted. Store-and-forward is
    /// explicitly pending delivery, not a delivered observation.
    pub fn validate(&self) -> Result<(), Vec<ScienceReturnIssue>> {
        let mut issues = Vec::new();

        if !nonempty(&self.site_id) {
            issues.push(ScienceReturnIssue::EmptySiteId);
        }
        if !nonempty(&self.observation_id) {
            issues.push(ScienceReturnIssue::EmptyObservationId);
        }
        if self.events.is_empty() {
            issues.push(ScienceReturnIssue::EmptyEventLog);
            return Err(issues);
        }

        let mut event_ids = BTreeSet::new();
        let mut previous_stage: Option<ScienceReturnStage> = None;

        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = index as u64 + 1;
            if event.sequence != expected_sequence {
                issues.push(ScienceReturnIssue::InvalidSequence {
                    expected: expected_sequence,
                    found: event.sequence,
                });
            }

            if !nonempty(&event.event_id) {
                issues.push(ScienceReturnIssue::EmptyEventId);
            }
            if !event_ids.insert(event.event_id.as_str()) {
                issues.push(ScienceReturnIssue::DuplicateEventId(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.recorded_at) {
                issues.push(ScienceReturnIssue::EmptyRecordedAt(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.actor_id) {
                issues.push(ScienceReturnIssue::EmptyActorId(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.evidence_ref) {
                issues.push(ScienceReturnIssue::EmptyEvidenceRef(
                    event.event_id.clone(),
                ));
            }

            if let Some(previous) = previous_stage {
                if previous.is_terminal() {
                    issues.push(ScienceReturnIssue::TerminalEventNotLast(previous));
                } else if !transition_allowed(previous, event.stage) {
                    issues.push(ScienceReturnIssue::InvalidTransition {
                        from: previous,
                        to: event.stage,
                    });
                }
            } else if event.stage != ScienceReturnStage::Proposed
                && !matches!(event.stage, ScienceReturnStage::Failed | ScienceReturnStage::Cancelled)
            {
                issues.push(ScienceReturnIssue::InvalidTransition {
                    from: ScienceReturnStage::Proposed,
                    to: event.stage,
                });
            }

            if event.stage == ScienceReturnStage::IntegrityVerified
                && !event.calibration_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(
                    ScienceReturnIssue::IntegrityVerificationMissingCalibration(
                        event.event_id.clone(),
                    ),
                );
            }
            if event.stage.requires_custody()
                && !event.custody_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::CustodyReferenceMissing(
                    event.event_id.clone(),
                ));
            }
            if matches!(
                event.stage,
                ScienceReturnStage::Delivered | ScienceReturnStage::Admitted
            ) && !event.delivery_receipt_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::DeliveryReceiptMissing(
                    event.event_id.clone(),
                ));
            }
            if event.stage == ScienceReturnStage::Admitted
                && !event.admission_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::AdmissionReferenceMissing(
                    event.event_id.clone(),
                ));
            }
            if matches!(event.stage, ScienceReturnStage::Failed | ScienceReturnStage::Cancelled)
                && !event.disposition_reason.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::TerminalDispositionMissing(
                    event.stage,
                ));
            }

            previous_stage = Some(event.stage);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Check that this record refers to a known observation in the source site
    /// record. This checks identifier linkage only; it does not authenticate
    /// the observation or any event evidence.
    pub fn validate_for_site(
        &self,
        site: &SiteEvidenceRecord,
    ) -> Result<(), Vec<ScienceReturnIssue>> {
        let mut issues = self.validate().err().unwrap_or_default();

        if self.site_id != site.site_id {
            issues.push(ScienceReturnIssue::SiteIdMismatch {
                record: self.site_id.clone(),
                source: site.site_id.clone(),
            });
        }
        if !site
            .observations
            .iter()
            .any(|observation| observation.observation_id == self.observation_id)
        {
            issues.push(ScienceReturnIssue::ObservationNotInSite(
                self.observation_id.clone(),
            ));
        }
        if let Err(site_issues) = site.validate() {
            issues.push(ScienceReturnIssue::InvalidSiteEvidence(site_issues));
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    pub fn last_stage(&self) -> Option<ScienceReturnStage> {
        self.events.last().map(|event| event.stage)
    }

    /// True only after a delivery event exists. Store-and-forward is not
    /// classified as delivery.
    pub fn evidence_delivered(&self) -> bool {
        self.events.iter().any(|event| {
            matches!(
                event.stage,
                ScienceReturnStage::Delivered | ScienceReturnStage::Admitted
            )
        })
    }

    /// True only when an external scientific-admission reference is recorded.
    pub fn scientifically_admitted(&self) -> bool {
        self.last_stage() == Some(ScienceReturnStage::Admitted)
    }

    pub fn awaiting_store_and_forward_retrieval(&self) -> bool {
        self.last_stage() == Some(ScienceReturnStage::StoreAndForwarded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        sequence: u64,
        stage: ScienceReturnStage,
        event_id: &str,
    ) -> ScienceReturnEvent {
        ScienceReturnEvent {
            sequence,
            event_id: event_id.into(),
            stage,
            recorded_at: format!("2026-11-16T12:{:02}:00Z", sequence),
            actor_id: "synthetic-robot-01".into(),
            evidence_ref: format!("synthetic-evidence-{event_id}"),
            calibration_ref: None,
            custody_ref: None,
            delivery_receipt_ref: None,
            admission_ref: None,
            disposition_reason: None,
        }
    }

    fn complete_record() -> ScienceReturnRecord {
        let mut events = vec![
            event(1, ScienceReturnStage::Proposed, "proposed"),
            event(2, ScienceReturnStage::Executed, "executed"),
            event(3, ScienceReturnStage::Acquired, "acquired"),
            event(4, ScienceReturnStage::IntegrityVerified, "verified"),
            event(5, ScienceReturnStage::Persisted, "persisted"),
            event(6, ScienceReturnStage::Delivered, "delivered"),
            event(7, ScienceReturnStage::Admitted, "admitted"),
        ];
        events[3].calibration_ref = Some("synthetic-calibration-record".into());
        events[4].custody_ref = Some("synthetic-storage-receipt".into());
        events[5].custody_ref = Some("synthetic-delivery-custodian".into());
        events[5].delivery_receipt_ref = Some("synthetic-delivery-receipt".into());
        events[6].custody_ref = Some("synthetic-science-custody".into());
        events[6].delivery_receipt_ref = Some("synthetic-delivery-receipt".into());
        events[6].admission_ref = Some("synthetic-science-admission".into());

        ScienceReturnRecord {
            site_id: "synthetic-mars-subterranean-candidate-01".into(),
            observation_id: "synthetic-optical-001".into(),
            events,
        }
    }

    #[test]
    fn complete_chain_requires_delivery_before_scientific_admission() {
        let record = complete_record();
        assert!(record.validate().is_ok());
        assert!(record.evidence_delivered());
        assert!(record.scientifically_admitted());
        assert!(!record.awaiting_store_and_forward_retrieval());
    }

    #[test]
    fn local_persistence_is_not_delivery() {
        let mut record = complete_record();
        record.events.truncate(5);
        assert!(record.validate().is_ok());
        assert!(!record.evidence_delivered());
        assert!(!record.scientifically_admitted());
    }

    #[test]
    fn store_and_forward_is_pending_not_delivered() {
        let mut record = complete_record();
        record.events.truncate(5);
        let mut event = event(6, ScienceReturnStage::StoreAndForwarded, "store-forward");
        event.custody_ref = Some("local-buffer-custody".into());
        record.events.push(event);

        assert!(record.validate().is_ok());
        assert!(!record.evidence_delivered());
        assert!(record.awaiting_store_and_forward_retrieval());
    }

    #[test]
    fn skipped_integrity_stage_is_rejected() {
        let mut record = complete_record();
        record.events.remove(3);
        for (index, event) in record.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }
        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::InvalidTransition {
                from: ScienceReturnStage::Acquired,
                to: ScienceReturnStage::Persisted
            }
        )));
    }

    #[test]
    fn delivered_stage_requires_custody_and_delivery_receipt() {
        let mut record = complete_record();
        record.events[5].custody_ref = None;
        record.events[5].delivery_receipt_ref = None;
        let issues = record.validate().unwrap_err();
        assert!(issues.contains(&ScienceReturnIssue::CustodyReferenceMissing(
            "delivered".into()
        )));
        assert!(issues.contains(&ScienceReturnIssue::DeliveryReceiptMissing(
            "delivered".into()
        )));
    }

    #[test]
    fn integrity_verification_requires_calibration_reference() {
        let mut record = complete_record();
        record.events[3].calibration_ref = None;
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::IntegrityVerificationMissingCalibration(
                "verified".into()
            )
        ));
    }

    #[test]
    fn admission_cannot_precede_delivery() {
        let mut record = complete_record();
        record.events.remove(5);
        for (index, event) in record.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }
        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::InvalidTransition {
                from: ScienceReturnStage::Persisted,
                to: ScienceReturnStage::Admitted
            }
        )));
    }

    #[test]
    fn failure_requires_reason_and_is_terminal() {
        let mut record = complete_record();
        record.events.truncate(3);
        let failed = event(4, ScienceReturnStage::Failed, "failed");
        record.events.push(failed);
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::TerminalDispositionMissing(ScienceReturnStage::Failed)
        ));
        record.events[3].disposition_reason = Some("robot immobilized".into());
        assert!(record.validate().is_ok());
    }

    #[test]
    fn terminal_failure_cannot_be_followed_by_success() {
        let mut record = complete_record();
        record.events.truncate(3);
        let mut failed = event(4, ScienceReturnStage::Failed, "failed");
        failed.disposition_reason = Some("mission aborted".into());
        record.events.push(failed);
        record.events.push(event(5, ScienceReturnStage::Acquired, "late-acquired"));

        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::TerminalEventNotLast(ScienceReturnStage::Failed)
        )));
    }

    #[test]
    fn site_and_observation_linkage_are_checked_without_authenticating_refs() {
        let mut site: SiteEvidenceRecord = serde_json::from_str(include_str!(
            "../fixtures/site-evidence-synthetic.json"
        ))
        .unwrap();
        let record = complete_record();
        assert!(record.validate_for_site(&site).is_ok());
        site.site_id = "different-site".into();
        assert!(record.validate_for_site(&site).unwrap_err().iter().any(
            |issue| matches!(issue, ScienceReturnIssue::SiteIdMismatch { .. })
        ));
    }
}
