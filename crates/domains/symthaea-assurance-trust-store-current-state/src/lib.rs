// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resolve the currently active trust-store segment across reviewed recoveries.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_trust_store::{TrustStoreCheckpoint, TrustStoreProfile};
use symthaea_assurance_trust_store_recovery_ledger::{
    RecoveryAcceptanceRecord, RecoveryLedgerStatus, assess_recovery_ledger,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentTrustStoreStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentTrustStoreIssue {
    InvalidProfile,
    EmptyCurrentSegment,
    InvalidRecoveryLedger,
    InvalidCheckpoint(u64),
    CurrentProfileDoesNotMatchAcceptedRecovery,
    FirstCheckpointDoesNotMatchAcceptedRecovery,
    InitialSegmentMustStartAtRevisionOne,
    InitialSegmentMustUseInitialEpoch,
    SegmentRevisionGap { previous: u64, next: u64 },
    SegmentCounterEpochChanged { revision: u64 },
    SegmentCounterDidNotIncrease { revision: u64 },
    SegmentPredecessorMismatch { revision: u64 },
    SegmentAnchorRevisionDecreased { revision: u64 },
    SegmentPolicyRevisionDecreased { revision: u64 },
    SegmentTimeRegressed { revision: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentTrustStoreReport {
    pub status: CurrentTrustStoreStatus,
    pub recovered: bool,
    pub segment_start_revision: Option<u64>,
    pub tip_store_revision: Option<u64>,
    pub tip_checkpoint_digest: Option<String>,
    pub tip_anchor_revision: Option<u64>,
    pub tip_anchor_digest: Option<String>,
    pub tip_policy_revision: Option<u64>,
    pub issues: Vec<CurrentTrustStoreIssue>,
}

impl CurrentTrustStoreReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_current_trust_store_state(
    profile: &TrustStoreProfile,
    current_segment: &[TrustStoreCheckpoint],
    recovery_ledger: &[RecoveryAcceptanceRecord],
) -> CurrentTrustStoreReport {
    let mut issues = Vec::new();
    if !profile.validate() {
        issues.push(CurrentTrustStoreIssue::InvalidProfile);
    }
    if current_segment.is_empty() {
        issues.push(CurrentTrustStoreIssue::EmptyCurrentSegment);
        return CurrentTrustStoreReport {
            status: CurrentTrustStoreStatus::Invalid,
            recovered: !recovery_ledger.is_empty(),
            segment_start_revision: None,
            tip_store_revision: None,
            tip_checkpoint_digest: None,
            tip_anchor_revision: None,
            tip_anchor_digest: None,
            tip_policy_revision: None,
            issues,
        };
    }

    let mut ordered = current_segment.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|checkpoint| checkpoint.store_revision);
    for checkpoint in &ordered {
        if !checkpoint.validate(profile) {
            issues.push(CurrentTrustStoreIssue::InvalidCheckpoint(
                checkpoint.store_revision,
            ));
        }
    }

    let recovered = !recovery_ledger.is_empty();
    if recovered {
        let ledger_report = assess_recovery_ledger(recovery_ledger);
        if ledger_report.status != RecoveryLedgerStatus::Valid {
            issues.push(CurrentTrustStoreIssue::InvalidRecoveryLedger);
        }
        let latest = recovery_ledger.iter().max_by_key(|record| record.ledger_revision);
        if let Some(latest) = latest {
            if latest.logical_store_id != profile.store_id
                || latest.replacement_trust_store_ref != profile.trust_store_ref
                || latest.replacement_counter_epoch != profile.initial_counter_epoch
            {
                issues.push(CurrentTrustStoreIssue::CurrentProfileDoesNotMatchAcceptedRecovery);
            }
            if ordered[0].checkpoint_digest() != latest.first_replacement_checkpoint_digest {
                issues.push(CurrentTrustStoreIssue::FirstCheckpointDoesNotMatchAcceptedRecovery);
            }
        }
    } else {
        if ordered[0].store_revision != 1 || ordered[0].predecessor_checkpoint_digest.is_some() {
            issues.push(CurrentTrustStoreIssue::InitialSegmentMustStartAtRevisionOne);
        }
        if ordered[0].counter_epoch != profile.initial_counter_epoch {
            issues.push(CurrentTrustStoreIssue::InitialSegmentMustUseInitialEpoch);
        }
    }

    let segment_epoch = ordered[0].counter_epoch.clone();
    for checkpoint in &ordered {
        if checkpoint.counter_epoch != segment_epoch {
            issues.push(CurrentTrustStoreIssue::SegmentCounterEpochChanged {
                revision: checkpoint.store_revision,
            });
        }
    }

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.store_revision != previous.store_revision.saturating_add(1) {
            issues.push(CurrentTrustStoreIssue::SegmentRevisionGap {
                previous: previous.store_revision,
                next: current.store_revision,
            });
        }
        if current.counter_value <= previous.counter_value {
            issues.push(CurrentTrustStoreIssue::SegmentCounterDidNotIncrease {
                revision: current.store_revision,
            });
        }
        let previous_digest = previous.checkpoint_digest();
        if current.predecessor_checkpoint_digest.as_deref() != Some(previous_digest.as_str()) {
            issues.push(CurrentTrustStoreIssue::SegmentPredecessorMismatch {
                revision: current.store_revision,
            });
        }
        if current.anchor_revision < previous.anchor_revision {
            issues.push(CurrentTrustStoreIssue::SegmentAnchorRevisionDecreased {
                revision: current.store_revision,
            });
        }
        if current.policy_tip_revision < previous.policy_tip_revision {
            issues.push(CurrentTrustStoreIssue::SegmentPolicyRevisionDecreased {
                revision: current.store_revision,
            });
        }
        if current.recorded_at_ms < previous.recorded_at_ms {
            issues.push(CurrentTrustStoreIssue::SegmentTimeRegressed {
                revision: current.store_revision,
            });
        }
    }

    let tip = ordered.last().expect("non-empty current segment");
    CurrentTrustStoreReport {
        status: if issues.is_empty() {
            CurrentTrustStoreStatus::Valid
        } else {
            CurrentTrustStoreStatus::Invalid
        },
        recovered,
        segment_start_revision: Some(ordered[0].store_revision),
        tip_store_revision: Some(tip.store_revision),
        tip_checkpoint_digest: Some(tip.checkpoint_digest()),
        tip_anchor_revision: Some(tip.anchor_revision),
        tip_anchor_digest: Some(tip.anchor_digest.clone()),
        tip_policy_revision: Some(tip.policy_tip_revision),
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_trust_store::TrustStoreBackendKind;

    fn profile(store_ref: &str, epoch: &str) -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: store_ref.into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some(format!("hardware:{store_ref}")),
            monotonic_counter_ref: format!("counter:{store_ref}"),
            initial_counter_epoch: epoch.into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:profile".into()],
        }
    }

    fn checkpoint(
        revision: u64,
        epoch: &str,
        counter: u64,
        previous: Option<String>,
    ) -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: format!("checkpoint:{revision}"),
            store_id: "policy-root".into(),
            store_revision: revision,
            counter_epoch: epoch.into(),
            counter_value: counter,
            anchor_revision: revision,
            anchor_digest: format!("blake3:anchor-{revision}"),
            policy_tip_revision: revision,
            predecessor_checkpoint_digest: previous,
            recorded_at_ms: revision * 1_000,
            attestation_ref: format!("attestation:{revision}"),
            independent_verification_ref: format!("verification:{revision}"),
            evidence_refs: vec![format!("audit:{revision}")],
        }
    }

    #[test]
    fn initial_segment_must_start_at_revision_one() {
        let profile = profile("trust-store:a", "epoch:a");
        let c1 = checkpoint(1, "epoch:a", 1, None);
        let c2 = checkpoint(2, "epoch:a", 2, Some(c1.checkpoint_digest()));
        let report = assess_current_trust_store_state(&profile, &[c1, c2], &[]);
        assert_eq!(report.status, CurrentTrustStoreStatus::Valid);
        assert!(!report.recovered);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn post_recovery_segment_must_start_at_accepted_checkpoint() {
        let profile = profile("trust-store:b", "epoch:b");
        let first = checkpoint(4, "epoch:b", 1, Some("blake3:old-tip".into()));
        let second = checkpoint(5, "epoch:b", 2, Some(first.checkpoint_digest()));
        let accepted = RecoveryAcceptanceRecord {
            schema_version: "1".into(),
            ledger_id: "recovery-ledger:policy-root".into(),
            ledger_revision: 1,
            logical_store_id: "policy-root".into(),
            authorization_id: "authorization:1".into(),
            recovery_commit_id: "commit:1".into(),
            previous_checkpoint_digest: "blake3:old-tip".into(),
            replacement_trust_store_ref: "trust-store:b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            first_replacement_checkpoint_digest: first.checkpoint_digest(),
            continuity_evidence_ref: "qualification:continuity".into(),
            continuity_evidence_digest: "blake3:continuity".into(),
            accepted_at_ms: 4_100,
            predecessor_acceptance_digest: None,
            evidence_refs: vec!["audit:acceptance".into()],
        };
        let report = assess_current_trust_store_state(&profile, &[first, second], &[accepted]);
        assert_eq!(report.status, CurrentTrustStoreStatus::Valid);
        assert!(report.recovered);
    }

    #[test]
    fn recovered_segment_cannot_start_from_unaccepted_checkpoint() {
        let profile = profile("trust-store:b", "epoch:b");
        let first = checkpoint(4, "epoch:b", 1, Some("blake3:old-tip".into()));
        let accepted = RecoveryAcceptanceRecord {
            schema_version: "1".into(),
            ledger_id: "recovery-ledger:policy-root".into(),
            ledger_revision: 1,
            logical_store_id: "policy-root".into(),
            authorization_id: "authorization:1".into(),
            recovery_commit_id: "commit:1".into(),
            previous_checkpoint_digest: "blake3:old-tip".into(),
            replacement_trust_store_ref: "trust-store:b".into(),
            replacement_counter_epoch: "epoch:b".into(),
            first_replacement_checkpoint_digest: "blake3:different".into(),
            continuity_evidence_ref: "qualification:continuity".into(),
            continuity_evidence_digest: "blake3:continuity".into(),
            accepted_at_ms: 4_100,
            predecessor_acceptance_digest: None,
            evidence_refs: vec!["audit:acceptance".into()],
        };
        let report = assess_current_trust_store_state(&profile, &[first], &[accepted]);
        assert_eq!(report.status, CurrentTrustStoreStatus::Invalid);
        assert!(report
            .issues
            .contains(&CurrentTrustStoreIssue::FirstCheckpointDoesNotMatchAcceptedRecovery));
    }

    #[test]
    fn current_segment_counter_must_remain_monotonic() {
        let profile = profile("trust-store:a", "epoch:a");
        let c1 = checkpoint(1, "epoch:a", 2, None);
        let c2 = checkpoint(2, "epoch:a", 2, Some(c1.checkpoint_digest()));
        let report = assess_current_trust_store_state(&profile, &[c1, c2], &[]);
        assert_eq!(report.status, CurrentTrustStoreStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CurrentTrustStoreIssue::SegmentCounterDidNotIncrease { revision: 2 }
        )));
    }
}
