//! As-of evidence replay and filtration-admission receipts for connectomics.
//!
//! The operational filtration is ordered by global availability sequence,
//! never by biological event time. The verifier audits chronology and
//! receipt-binding semantics. It does not authenticate artifacts or prove
//! conditional supermartingale validity / an adjuster theorem.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditError(pub String);

impl fmt::Display for AuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for AuditError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AvailabilityState {
    Approved,
    Provisional,
    Disputed,
    Failed,
    Missing,
    OutOfSupport,
    NotAssessed,
}

impl AvailabilityState {
    fn consumable_as_typed_state(self) -> bool {
        matches!(
            self,
            Self::Approved | Self::Failed | Self::Missing | Self::OutOfSupport | Self::NotAssessed
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceSplit {
    Exploration,
    Fit,
    Calibration,
    Confirmation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceEvent {
    pub event_id: String,
    pub evidence_family_id: String,
    pub source_stream: String,
    pub source_artifact_sha256: String,
    pub revision: u32,
    pub supersedes_event_id: Option<String>,
    /// Biological event time. It never determines operational availability.
    pub event_time_ms: Option<i64>,
    pub acquired_at_ms: Option<i64>,
    pub annotated_at_ms: Option<i64>,
    pub qc_at_ms: Option<i64>,
    pub availability_state: AvailabilityState,
    pub split: EvidenceSplit,
    /// Required for confirmation-split data; binds it to the frozen family.
    pub confirmatory_family_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyFreeze {
    pub family_id: String,
    pub family_commitment_sha256: String,
    /// The lockbox is still closed at the instant of this freeze event.
    pub confirmation_lockbox_closed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionKind {
    Discovery,
    Confirmation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionReceipt {
    pub decision_id: String,
    pub campaign_id: String,
    pub kind: DecisionKind,
    pub policy_version_sha256: String,
    pub input_prefix_sha256: String,
    pub driver_receipt_sha256: String,
    pub history_prefix_sha256: String,
    pub action_sha256: String,
    pub feasibility_receipt_sha256: String,
    /// Exact set of evidence made available to the policy before this decision.
    pub available_event_ids: Vec<String>,
    /// Exact evidence inputs reported by the driver as consumed.
    pub consumed_event_ids: Vec<String>,
    pub confirmatory_family_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerPayload {
    Evidence(EvidenceEvent),
    FamilyFreeze(FamilyFreeze),
    Decision(DecisionReceipt),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEntry {
    /// One-based global availability/order sequence; all entries are unique.
    pub sequence: u64,
    pub entry_commitment_sha256: String,
    pub payload: LedgerPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalLedger {
    pub campaign_id: String,
    pub global_filtration_id: String,
    pub global_filtration_sha256: String,
    pub entries: Vec<LedgerEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionAuditStatus {
    Accepted,
    LookaheadDetected,
    PrefixMismatch,
    LockboxLeakage,
    InvalidReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionAudit {
    pub decision_id: String,
    pub sequence: u64,
    pub status: DecisionAuditStatus,
    pub available_prefix_ids: Vec<String>,
    pub consumed_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerAudit {
    pub campaign_id: String,
    pub final_sequence: u64,
    pub global_filtration_sha256: String,
    pub decisions: Vec<DecisionAudit>,
    pub evidence_ids: Vec<String>,
}

/// Replay the immutable ledger in availability order. The event_time fields
/// are intentionally ignored for ordering and lookahead decisions.
pub fn audit_ledger(ledger: &GlobalLedger) -> Result<LedgerAudit, AuditError> {
    if ledger.campaign_id.is_empty() || ledger.global_filtration_id.is_empty() {
        return Err(AuditError("campaign and global filtration IDs are required".into()));
    }
    if !valid_sha256(&ledger.global_filtration_sha256) {
        return Err(AuditError("malformed global filtration digest".into()));
    }

    let mut evidence: BTreeMap<String, (u64, EvidenceEvent)> = BTreeMap::new();
    let mut families: BTreeMap<String, (u64, FamilyFreeze)> = BTreeMap::new();
    let mut decisions = Vec::new();
    let mut ordered_evidence_ids = Vec::new();

    for (index, entry) in ledger.entries.iter().enumerate() {
        let expected_sequence = index as u64 + 1;
        if entry.sequence != expected_sequence {
            return Err(AuditError(format!(
                "global ledger sequence gap: expected {expected_sequence}, found {}",
                entry.sequence
            )));
        }
        if !valid_sha256(&entry.entry_commitment_sha256) {
            return Err(AuditError(format!(
                "malformed entry commitment at sequence {}",
                entry.sequence
            )));
        }

        match &entry.payload {
            LedgerPayload::Evidence(event) => {
                if event.event_id.is_empty()
                    || event.evidence_family_id.is_empty()
                    || event.source_stream.is_empty()
                    || !valid_sha256(&event.source_artifact_sha256)
                {
                    return Err(AuditError(format!(
                        "invalid evidence identity or source digest at sequence {}",
                        entry.sequence
                    )));
                }
                if evidence.contains_key(&event.event_id) {
                    return Err(AuditError(format!(
                        "duplicate or rewritten event ID {}",
                        event.event_id
                    )));
                }
                match event.split {
                    EvidenceSplit::Confirmation => {
                        let Some(family_id) = event.confirmatory_family_id.as_ref() else {
                            return Err(AuditError(format!(
                                "confirmation event {} lacks a family binding",
                                event.event_id
                            )));
                        };
                        let Some((freeze_sequence, _)) = families.get(family_id) else {
                            return Err(AuditError(format!(
                                "confirmation event {} arrived before its family freeze",
                                event.event_id
                            )));
                        };
                        if *freeze_sequence >= entry.sequence {
                            return Err(AuditError(format!(
                                "confirmation event {} is not after its family freeze",
                                event.event_id
                            )));
                        }
                    }
                    _ if event.confirmatory_family_id.is_some() => {
                        return Err(AuditError(format!(
                            "non-confirmation event {} has a confirmatory family binding",
                            event.event_id
                        )));
                    }
                    _ => {}
                }
                match (&event.supersedes_event_id, event.revision) {
                    (None, 0) => {}
                    (Some(previous_id), revision) if revision > 0 => {
                        let (previous_sequence, previous) = evidence
                            .get(previous_id)
                            .ok_or_else(|| AuditError(format!(
                                "{} supersedes an unknown or future event {previous_id}",
                                event.event_id
                            )))?;
                        if *previous_sequence >= entry.sequence
                            || previous.evidence_family_id != event.evidence_family_id
                            || previous.revision.checked_add(1) != Some(revision)
                        {
                            return Err(AuditError(format!(
                                "invalid revision/supersession chain for {}",
                                event.event_id
                            )));
                        }
                    }
                    _ => {
                        return Err(AuditError(format!(
                            "revision zero must have no supersession; later revisions must name a parent ({})",
                            event.event_id
                        )));
                    }
                }
                evidence.insert(event.event_id.clone(), (entry.sequence, event.clone()));
                ordered_evidence_ids.push(event.event_id.clone());
            }
            LedgerPayload::FamilyFreeze(freeze) => {
                if freeze.family_id.is_empty()
                    || !valid_sha256(&freeze.family_commitment_sha256)
                    || !freeze.confirmation_lockbox_closed
                    || families.contains_key(&freeze.family_id)
                {
                    return Err(AuditError(format!(
                        "invalid or duplicate family freeze at sequence {}",
                        entry.sequence
                    )));
                }
                families.insert(freeze.family_id.clone(), (entry.sequence, freeze.clone()));
            }
            LedgerPayload::Decision(decision) => {
                let mut status = DecisionAuditStatus::Accepted;
                if decision.campaign_id != ledger.campaign_id
                    || decision.decision_id.is_empty()
                    || !valid_sha256(&decision.policy_version_sha256)
                    || !valid_sha256(&decision.input_prefix_sha256)
                    || !valid_sha256(&decision.driver_receipt_sha256)
                    || !valid_sha256(&decision.history_prefix_sha256)
                    || !valid_sha256(&decision.action_sha256)
                    || !valid_sha256(&decision.feasibility_receipt_sha256)
                {
                    status = DecisionAuditStatus::InvalidReceipt;
                }

                // The prefix is every evidence event actually delivered before
                // this decision, in canonical global availability order.
                if decision.available_event_ids != ordered_evidence_ids {
                    status = DecisionAuditStatus::PrefixMismatch;
                }

                let mut consumed = BTreeSet::new();
                for event_id in &decision.consumed_event_ids {
                    if !consumed.insert(event_id.as_str()) {
                        status = DecisionAuditStatus::InvalidReceipt;
                        continue;
                    }
                    let Some((available_sequence, event)) = evidence.get(event_id) else {
                        status = DecisionAuditStatus::LookaheadDetected;
                        continue;
                    };
                    if *available_sequence >= entry.sequence {
                        status = DecisionAuditStatus::LookaheadDetected;
                    }
                    if !event.availability_state.consumable_as_typed_state() {
                        status = DecisionAuditStatus::InvalidReceipt;
                    }
                    match decision.kind {
                        DecisionKind::Discovery => {
                            if event.split == EvidenceSplit::Confirmation {
                                status = DecisionAuditStatus::LockboxLeakage;
                            }
                        }
                        DecisionKind::Confirmation => {
                            if event.split == EvidenceSplit::Confirmation {
                                let Some(family_id) = decision.confirmatory_family_id.as_ref() else {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                    continue;
                                };
                                let Some((freeze_sequence, _)) = families.get(family_id) else {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                    continue;
                                };
                                if *freeze_sequence >= entry.sequence
                                    || *available_sequence <= *freeze_sequence
                                    || event.confirmatory_family_id.as_deref()
                                        != Some(family_id.as_str())
                                {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                }
                            }
                        }
                    }
                }
                if decision.kind == DecisionKind::Confirmation {
                    let valid_freeze = decision
                        .confirmatory_family_id
                        .as_ref()
                        .and_then(|id| families.get(id))
                        .is_some_and(|(freeze_sequence, _)| *freeze_sequence < entry.sequence);
                    if !valid_freeze {
                        status = DecisionAuditStatus::LockboxLeakage;
                    }
                }
                decisions.push(DecisionAudit {
                    decision_id: decision.decision_id.clone(),
                    sequence: entry.sequence,
                    status,
                    available_prefix_ids: decision.available_event_ids.clone(),
                    consumed_ids: decision.consumed_event_ids.clone(),
                });
            }
        }
    }

    Ok(LedgerAudit {
        campaign_id: ledger.campaign_id.clone(),
        final_sequence: ledger.entries.len() as u64,
        global_filtration_sha256: ledger.global_filtration_sha256.clone(),
        decisions,
        evidence_ids: ordered_evidence_ids,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionMode {
    DirectGlobalSupermartingale,
    AdjusterLifted,
    LocalOnly,
    IncomparableOrUnknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdjusterReceipt {
    pub adjuster_id: String,
    pub adjuster_version: String,
    pub parameter_sha256: String,
    pub theorem_assumptions_sha256: String,
    pub lifted_output_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiltrationAdmission {
    pub campaign_id: String,
    pub process_id: String,
    pub null_spec_sha256: String,
    pub input_process_sha256: String,
    pub output_process_sha256: String,
    pub source_filtration_id: String,
    pub source_filtration_sha256: String,
    pub target_filtration_id: String,
    pub target_filtration_sha256: String,
    pub observed_event_ids: Vec<String>,
    /// The process output is committed after this availability prefix.
    pub process_output_sequence: u64,
    pub mode: AdmissionMode,
    /// Independent proof-review receipt; this verifier does not check its theorem.
    pub proof_receipt_sha256: Option<String>,
    pub assumption_sha256: Option<String>,
    pub adjuster: Option<AdjusterReceipt>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionStatus {
    AdmittedDirectGlobal,
    AdmittedAfterLift,
    NotCombinableLocalOnly,
    NotCombinableIncomparable,
    InvalidReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionAudit {
    pub process_id: String,
    pub status: AdmissionStatus,
    pub proof_receipt_sha256: Option<String>,
    pub source_filtration_sha256: String,
    pub target_filtration_sha256: String,
}

/// Structural gate only. Admission means that a matching external proof
/// receipt is present and bound to these exact artifacts; it is not a claim
/// that this crate proved the supermartingale/adjuster theorem.
pub fn audit_admission(
    ledger: &GlobalLedger,
    admission: &FiltrationAdmission,
) -> AdmissionAudit {
    let invalid = |reason_status| AdmissionAudit {
        process_id: admission.process_id.clone(),
        status: reason_status,
        proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
        source_filtration_sha256: admission.source_filtration_sha256.clone(),
        target_filtration_sha256: admission.target_filtration_sha256.clone(),
    };
    if admission.campaign_id != ledger.campaign_id
        || admission.process_id.is_empty()
        || admission.source_filtration_id.is_empty()
        || admission.target_filtration_id != ledger.global_filtration_id
        || admission.target_filtration_sha256 != ledger.global_filtration_sha256
        || !valid_sha256(&admission.null_spec_sha256)
        || !valid_sha256(&admission.input_process_sha256)
        || !valid_sha256(&admission.output_process_sha256)
        || !valid_sha256(&admission.source_filtration_sha256)
        || !valid_sha256(&admission.target_filtration_sha256)
        || admission.observed_event_ids.is_empty()
        || admission.process_output_sequence == 0
        || admission.process_output_sequence > ledger.entries.len() as u64
    {
        return invalid(AdmissionStatus::InvalidReceipt);
    }

    let evidence_map: BTreeMap<_, _> = ledger
        .entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            LedgerPayload::Evidence(event) => Some((event.event_id.as_str(), event)),
            _ => None,
        })
        .collect();
    let mut observed = BTreeSet::new();
    for event_id in &admission.observed_event_ids {
        let Some((available_sequence, event)) = evidence_map.get(event_id.as_str()) else {
            return invalid(AdmissionStatus::InvalidReceipt);
        };
        if !observed.insert(event_id.as_str())
            || *available_sequence >= admission.process_output_sequence
            || !event.availability_state.consumable_as_typed_state()
        {
            return invalid(AdmissionStatus::InvalidReceipt);
        }
    }

    match admission.mode {
        AdmissionMode::DirectGlobalSupermartingale => {
            if admission.source_filtration_id != ledger.global_filtration_id
                || admission.source_filtration_sha256 != ledger.global_filtration_sha256
                || admission.proof_receipt_sha256.as_deref().is_none_or(|d| !valid_sha256(d))
                || admission.assumption_sha256.as_deref().is_none_or(|d| !valid_sha256(d))
                || admission.adjuster.is_some()
            {
                return invalid(AdmissionStatus::InvalidReceipt);
            }
            AdmissionAudit {
                process_id: admission.process_id.clone(),
                status: AdmissionStatus::AdmittedDirectGlobal,
                proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
                source_filtration_sha256: admission.source_filtration_sha256.clone(),
                target_filtration_sha256: admission.target_filtration_sha256.clone(),
            }
        }
        AdmissionMode::AdjusterLifted => {
            let Some(adjuster) = admission.adjuster.as_ref() else {
                return invalid(AdmissionStatus::InvalidReceipt);
            };
            if admission.source_filtration_sha256 == ledger.global_filtration_sha256
                || admission.proof_receipt_sha256.as_deref().is_none_or(|d| !valid_sha256(d))
                || admission.assumption_sha256.as_deref().is_none_or(|d| !valid_sha256(d))
                || adjuster.adjuster_id.is_empty()
                || adjuster.adjuster_version.is_empty()
                || !valid_sha256(&adjuster.parameter_sha256)
                || !valid_sha256(&adjuster.theorem_assumptions_sha256)
                || !valid_sha256(&adjuster.lifted_output_sha256)
                || adjuster.lifted_output_sha256 != admission.output_process_sha256
            {
                return invalid(AdmissionStatus::InvalidReceipt);
            }
            AdmissionAudit {
                process_id: admission.process_id.clone(),
                status: AdmissionStatus::AdmittedAfterLift,
                proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
                source_filtration_sha256: admission.source_filtration_sha256.clone(),
                target_filtration_sha256: admission.target_filtration_sha256.clone(),
            }
        }
        AdmissionMode::LocalOnly => invalid(AdmissionStatus::NotCombinableLocalOnly),
        AdmissionMode::IncomparableOrUnknown => {
            invalid(AdmissionStatus::NotCombinableIncomparable)
        }
    }
}

/// Replayable as-of snapshots for prefix-invariance metamorphic tests.
pub fn decision_prefixes(ledger: &GlobalLedger) -> Result<Vec<(String, Vec<String>, String)>, AuditError> {
    let audit = audit_ledger(ledger)?;
    if audit
        .decisions
        .iter()
        .any(|decision| decision.status != DecisionAuditStatus::Accepted)
    {
        return Err(AuditError("cannot produce prefixes from an invalid decision ledger".into()));
    }
    Ok(audit
        .decisions
        .iter()
        .map(|decision| {
            let action_sha = ledger
                .entries
                .iter()
                .find_map(|entry| match &entry.payload {
                    LedgerPayload::Decision(receipt)
                        if receipt.decision_id == decision.decision_id =>
                    {
                        Some(receipt.action_sha256.clone())
                    }
                    _ => None,
                })
                .expect("audited decision exists");
            (
                decision.decision_id.clone(),
                decision.consumed_ids.clone(),
                action_sha,
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn event(id: &str, family: &str, event_time: i64, state: AvailabilityState, split: EvidenceSplit) -> LedgerPayload {
        LedgerPayload::Evidence(EvidenceEvent {
            event_id: id.into(),
            evidence_family_id: family.into(),
            source_stream: "synthetic-stream".into(),
            source_artifact_sha256: d('a'),
            revision: 0,
            supersedes_event_id: None,
            event_time_ms: Some(event_time),
            acquired_at_ms: Some(event_time + 1),
            annotated_at_ms: Some(event_time + 2),
            qc_at_ms: Some(event_time + 3),
            availability_state: state,
            split,
            confirmatory_family_id: None,
        })
    }

    fn decision(
        id: &str,
        available: &[&str],
        consumed: &[&str],
        kind: DecisionKind,
        family: Option<&str>,
    ) -> LedgerPayload {
        LedgerPayload::Decision(DecisionReceipt {
            decision_id: id.into(),
            campaign_id: "campaign-1".into(),
            kind,
            policy_version_sha256: d('b'),
            input_prefix_sha256: d('c'),
            driver_receipt_sha256: d('d'),
            history_prefix_sha256: d('e'),
            action_sha256: d('f'),
            feasibility_receipt_sha256: d('1'),
            available_event_ids: available.iter().map(|x| (*x).into()).collect(),
            consumed_event_ids: consumed.iter().map(|x| (*x).into()).collect(),
            confirmatory_family_id: family.map(str::to_owned),
        })
    }

    fn entry(sequence: u64, payload: LedgerPayload) -> LedgerEntry {
        LedgerEntry {
            sequence,
            entry_commitment_sha256: d('9'),
            payload,
        }
    }

    fn fixture() -> GlobalLedger {
        GlobalLedger {
            campaign_id: "campaign-1".into(),
            global_filtration_id: "connectome-global-v1".into(),
            global_filtration_sha256: d('8'),
            entries: vec![
                entry(1, event("S1", "structural", 100, AvailabilityState::Approved, EvidenceSplit::Fit)),
                entry(2, decision("D1", &["S1"], &["S1"], DecisionKind::Discovery, None)),
                entry(3, event("T7", "behaviour-trial-7", 90, AvailabilityState::Approved, EvidenceSplit::Exploration)),
                entry(4, decision("D2", &["S1", "T7"], &["S1", "T7"], DecisionKind::Discovery, None)),
                entry(5, event("T8-failed", "behaviour-trial-8", 105, AvailabilityState::Failed, EvidenceSplit::Exploration)),
            ],
        }
    }

    fn confirmation_event() -> LedgerPayload {
        let mut payload = event(
            "T7",
            "behaviour-trial-7",
            90,
            AvailabilityState::Approved,
            EvidenceSplit::Confirmation,
        );
        if let LedgerPayload::Evidence(ref mut value) = payload {
            value.confirmatory_family_id = Some("family-1".into());
        }
        payload
    }

    fn admission(mode: AdmissionMode) -> FiltrationAdmission {
        FiltrationAdmission {
            campaign_id: "campaign-1".into(),
            process_id: "P-S".into(),
            null_spec_sha256: d('a'),
            input_process_sha256: d('b'),
            output_process_sha256: d('c'),
            source_filtration_id: "connectome-global-v1".into(),
            source_filtration_sha256: d('8'),
            target_filtration_id: "connectome-global-v1".into(),
            target_filtration_sha256: d('8'),
            observed_event_ids: vec!["S1".into(), "T7".into()],
            process_output_sequence: 4,
            mode,
            proof_receipt_sha256: Some(d('d')),
            assumption_sha256: Some(d('e')),
            adjuster: None,
        }
    }

    #[test]
    fn known_answer_uses_availability_not_event_time() {
        let report = audit_ledger(&fixture()).unwrap();
        assert_eq!(report.final_sequence, 5);
        assert_eq!(report.decisions.len(), 2);
        assert_eq!(report.decisions[0].status, DecisionAuditStatus::Accepted);
        assert_eq!(report.decisions[1].status, DecisionAuditStatus::Accepted);
        assert_eq!(report.decisions[0].available_prefix_ids, vec!["S1"]);
        assert_eq!(report.decisions[1].available_prefix_ids, vec!["S1", "T7"]);
        // T7's biological event time (90) predates D1, but its availability
        // sequence (3) follows D1 (2), so D1 cannot consume it.
    }

    #[test]
    fn detects_lookahead_even_when_biological_event_time_is_earlier() {
        let mut ledger = fixture();
        ledger.entries[1] = entry(
            2,
            decision("D1", &["S1", "T7"], &["S1", "T7"], DecisionKind::Discovery, None),
        );
        // T7 has not yet been delivered at sequence 2.
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[0].status, DecisionAuditStatus::LookaheadDetected);
    }

    #[test]
    fn rejects_provisional_and_disputed_inputs() {
        let mut ledger = fixture();
        ledger.entries[0] = entry(1, event("S1", "structural", 100, AvailabilityState::Provisional, EvidenceSplit::Fit));
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[0].status, DecisionAuditStatus::InvalidReceipt);
    }

    #[test]
    fn rejects_noncontiguous_global_order() {
        let mut ledger = fixture();
        ledger.entries[2].sequence = 7;
        assert!(audit_ledger(&ledger).is_err());
    }

    #[test]
    fn rejects_duplicate_event_ids_and_in_place_rewrites() {
        let mut ledger = fixture();
        ledger.entries.push(entry(6, event("S1", "structural", 101, AvailabilityState::Approved, EvidenceSplit::Fit)));
        assert!(audit_ledger(&ledger).is_err());
    }

    #[test]
    fn accepts_immutable_revision_chain_as_new_events() {
        let mut ledger = fixture();
        let revised = EvidenceEvent {
            event_id: "T7-v2".into(),
            evidence_family_id: "behaviour-trial-7".into(),
            source_stream: "synthetic-stream".into(),
            source_artifact_sha256: d('2'),
            revision: 1,
            supersedes_event_id: Some("T7".into()),
            event_time_ms: Some(90),
            acquired_at_ms: Some(91),
            annotated_at_ms: Some(110),
            qc_at_ms: Some(111),
            availability_state: AvailabilityState::Approved,
            split: EvidenceSplit::Exploration,
        };
        ledger.entries.push(entry(6, LedgerPayload::Evidence(revised)));
        assert!(audit_ledger(&ledger).is_ok());
    }

    #[test]
    fn rejects_discovery_consuming_confirmation_lockbox_data() {
        let mut ledger = fixture();
        ledger.entries[2] = entry(3, event("T7", "behaviour-trial-7", 90, AvailabilityState::Approved, EvidenceSplit::Confirmation));
        ledger.entries[3] = entry(4, decision("D2", &["S1", "T7"], &["S1", "T7"], DecisionKind::Discovery, None));
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[1].status, DecisionAuditStatus::LockboxLeakage);
    }

    #[test]
    fn confirmation_requires_a_prior_freeze_and_postfreeze_lockbox_data() {
        let mut ledger = fixture();
        ledger.entries.insert(1, entry(2, LedgerPayload::FamilyFreeze(FamilyFreeze {
            family_id: "family-1".into(),
            family_commitment_sha256: d('7'),
            confirmation_lockbox_closed: true,
        })));
        ledger.entries[2] = entry(3, decision("D1", &["S1"], &["S1"], DecisionKind::Discovery, None));
        ledger.entries[3] = entry(4, event("T7", "behaviour-trial-7", 90, AvailabilityState::Approved, EvidenceSplit::Confirmation));
        ledger.entries[4] = entry(5, decision("D2", &["S1", "T7"], &["S1", "T7"], DecisionKind::Confirmation, Some("family-1")));
        for (idx, e) in ledger.entries.iter_mut().enumerate() {
            e.sequence = idx as u64 + 1;
        }
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[1].status, DecisionAuditStatus::Accepted);
    }

    #[test]
    fn rejects_confirmation_event_that_predates_family_freeze() {
        let mut ledger = fixture();
        ledger.entries.insert(1, entry(2, LedgerPayload::FamilyFreeze(FamilyFreeze {
            family_id: "family-1".into(),
            family_commitment_sha256: d('7'),
            confirmation_lockbox_closed: true,
        })));
        ledger.entries[2] = entry(3, event("T7", "behaviour-trial-7", 90, AvailabilityState::Approved, EvidenceSplit::Confirmation));
        ledger.entries[3] = entry(4, decision("D2", &["S1", "T7"], &["S1", "T7"], DecisionKind::Confirmation, Some("family-1")));
        for (idx, e) in ledger.entries.iter_mut().enumerate() {
            e.sequence = idx as u64 + 1;
        }
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[0].status, DecisionAuditStatus::LockboxLeakage);
    }

    #[test]
    fn direct_global_admission_requires_exact_global_filtration_and_review_receipt() {
        let ledger = fixture();
        let report = audit_admission(&ledger, &admission(AdmissionMode::DirectGlobalSupermartingale));
        assert_eq!(report.status, AdmissionStatus::AdmittedDirectGlobal);
    }

    #[test]
    fn local_only_stream_is_not_combinable_by_digest_relabelling() {
        let ledger = fixture();
        let mut local = admission(AdmissionMode::LocalOnly);
        local.source_filtration_id = "neural-local".into();
        local.source_filtration_sha256 = d('4');
        let report = audit_admission(&ledger, &local);
        assert_eq!(report.status, AdmissionStatus::NotCombinableLocalOnly);

        let mut forged = admission(AdmissionMode::DirectGlobalSupermartingale);
        forged.source_filtration_id = "neural-local".into();
        forged.source_filtration_sha256 = d('4');
        assert_eq!(
            audit_admission(&ledger, &forged).status,
            AdmissionStatus::InvalidReceipt
        );
    }

    #[test]
    fn lifted_stream_must_bind_adjuster_and_exact_output() {
        let ledger = fixture();
        let mut lifted = admission(AdmissionMode::AdjusterLifted);
        lifted.source_filtration_id = "neural-local".into();
        lifted.source_filtration_sha256 = d('4');
        lifted.adjuster = Some(AdjusterReceipt {
            adjuster_id: "reviewed-adjuster".into(),
            adjuster_version: "1".into(),
            parameter_sha256: d('5'),
            theorem_assumptions_sha256: d('6'),
            lifted_output_sha256: d('c'),
        });
        assert_eq!(
            audit_admission(&ledger, &lifted).status,
            AdmissionStatus::AdmittedAfterLift
        );
        lifted.adjuster.as_mut().unwrap().lifted_output_sha256 = d('7');
        assert_eq!(
            audit_admission(&ledger, &lifted).status,
            AdmissionStatus::InvalidReceipt
        );
    }

    #[test]
    fn incomparable_filtration_is_explicitly_not_combinable() {
        let ledger = fixture();
        assert_eq!(
            audit_admission(&ledger, &admission(AdmissionMode::IncomparableOrUnknown)).status,
            AdmissionStatus::NotCombinableIncomparable
        );
    }

    #[test]
    fn future_only_extension_preserves_prior_decision_prefixes() {
        let ledger = fixture();
        let before = decision_prefixes(&ledger).unwrap();
        let mut extended = fixture();
        extended.entries.push(entry(
            6,
            event("future-track", "later-trial", -999, AvailabilityState::Approved, EvidenceSplit::Exploration),
        ));
        let after = decision_prefixes(&extended).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn typed_failed_and_out_of_support_events_are_not_zero_filled() {
        let mut ledger = fixture();
        ledger.entries[4] = entry(5, event("T8-failed", "behaviour-trial-8", 105, AvailabilityState::OutOfSupport, EvidenceSplit::Exploration));
        let report = audit_ledger(&ledger).unwrap();
        assert_eq!(report.evidence_ids.last().unwrap(), "T8-failed");
    }
}
