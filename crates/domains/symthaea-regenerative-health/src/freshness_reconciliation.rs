// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Receiver-side freshness sequencing and offline-replica reconciliation.
//!
//! This module does not authenticate markers, establish authority, or infer
//! physical truth. It only maintains deterministic acceptance state for a
//! freshness handle supplied by a configured trust/transport layer.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const STATE_DOMAIN: &[u8] = b"symthaea:freshness-acceptance-state:v1\n";
const POLICY_DOMAIN: &[u8] = b"symthaea:freshness-acceptance-policy:v1\n";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessMarker {
    pub scheme: String,
    pub source_id: String,
    pub epoch: u64,
    pub marker_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAcceptancePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_scheme: String,
    pub expected_source_id: String,
    pub minimum_accepted_epoch: u64,
    pub acceptance_window: u64,
}

impl FreshnessAcceptancePolicy {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.expected_scheme.trim().is_empty()
            || self.expected_source_id.trim().is_empty()
            || self.minimum_accepted_epoch == 0
        {
            return Err("invalid freshness acceptance policy");
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("freshness policy is serializable");
        let mut input = Vec::with_capacity(POLICY_DOMAIN.len() + bytes.len());
        input.extend_from_slice(POLICY_DOMAIN);
        input.extend_from_slice(&bytes);
        blake3::hash(&input).to_hex().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessAcceptanceOutcome {
    Advanced,
    AcceptedWithinWindow,
    Duplicate,
    Conflicted,
    Rollback,
    Quarantined,
    BlockedByConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessAcceptanceIssue {
    EmptyMarkerIdentity,
    WrongScheme,
    WrongSource,
    InvalidEpoch,
    BelowMinimumEpoch,
    EpochOutsideAcceptanceWindow,
    ConflictingMarker,
    ExistingConflict,
    PolicyMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAcceptanceState {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_fingerprint: String,
    pub scheme: String,
    pub source_id: String,
    pub minimum_accepted_epoch: u64,
    pub acceptance_window: u64,
    pub highest_accepted_epoch: Option<u64>,
    pub marker_digests: BTreeMap<u64, BTreeSet<String>>,
    pub conflicted: bool,
}

impl FreshnessAcceptanceState {
    pub fn new(policy: &FreshnessAcceptancePolicy) -> Result<Self, &'static str> {
        policy.validate()?;
        Ok(Self {
            schema_version: policy.schema_version.clone(),
            policy_id: policy.policy_id.clone(),
            policy_fingerprint: policy.fingerprint(),
            scheme: policy.expected_scheme.clone(),
            source_id: policy.expected_source_id.clone(),
            minimum_accepted_epoch: policy.minimum_accepted_epoch,
            acceptance_window: policy.acceptance_window,
            highest_accepted_epoch: None,
            marker_digests: BTreeMap::new(),
            conflicted: false,
        })
    }

    pub fn status_conflicted(&self) -> bool {
        self.conflicted || self.marker_digests.values().any(|set| set.len() > 1)
    }

    pub fn fingerprint(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("freshness state is serializable");
        let mut input = Vec::with_capacity(STATE_DOMAIN.len() + bytes.len());
        input.extend_from_slice(STATE_DOMAIN);
        input.extend_from_slice(&bytes);
        blake3::hash(&input).to_hex().to_string()
    }

    fn matches_policy(&self, policy: &FreshnessAcceptancePolicy) -> bool {
        self.schema_version == policy.schema_version
            && self.policy_id == policy.policy_id
            && self.policy_fingerprint == policy.fingerprint()
            && self.scheme == policy.expected_scheme
            && self.source_id == policy.expected_source_id
            && self.minimum_accepted_epoch == policy.minimum_accepted_epoch
            && self.acceptance_window == policy.acceptance_window
    }

    fn decision(
        &self,
        outcome: FreshnessAcceptanceOutcome,
        issues: Vec<FreshnessAcceptanceIssue>,
    ) -> FreshnessAcceptanceDecision {
        FreshnessAcceptanceDecision {
            state: self.clone(),
            outcome,
            issues,
        }
    }

    fn record_marker(&mut self, marker: &FreshnessMarker) {
        let set = self.marker_digests.entry(marker.epoch).or_default();
        set.insert(marker.marker_digest.clone());
        if set.len() > 1 {
            self.conflicted = true;
        }
    }

    fn prune_to_window(&mut self) {
        let Some(highest) = self.highest_accepted_epoch else {
            return;
        };
        let floor = highest.saturating_sub(self.acceptance_window);
        self.marker_digests.retain(|epoch, _| *epoch >= floor);
    }

    pub fn apply(
        &self,
        policy: &FreshnessAcceptancePolicy,
        marker: &FreshnessMarker,
    ) -> FreshnessAcceptanceDecision {
        if !self.matches_policy(policy) {
            return self.decision(
                FreshnessAcceptanceOutcome::Quarantined,
                vec![FreshnessAcceptanceIssue::PolicyMismatch],
            );
        }

        let mut issues = Vec::new();
        if marker.scheme.trim().is_empty()
            || marker.source_id.trim().is_empty()
            || marker.marker_digest.trim().is_empty()
        {
            issues.push(FreshnessAcceptanceIssue::EmptyMarkerIdentity);
        }
        if marker.epoch == 0 {
            issues.push(FreshnessAcceptanceIssue::InvalidEpoch);
        }
        if marker.scheme != policy.expected_scheme {
            issues.push(FreshnessAcceptanceIssue::WrongScheme);
        }
        if marker.source_id != policy.expected_source_id {
            issues.push(FreshnessAcceptanceIssue::WrongSource);
        }
        if marker.epoch < policy.minimum_accepted_epoch {
            issues.push(FreshnessAcceptanceIssue::BelowMinimumEpoch);
        }
        if !issues.is_empty() {
            return self.decision(FreshnessAcceptanceOutcome::Quarantined, issues);
        }

        if self.status_conflicted() {
            return self.decision(
                FreshnessAcceptanceOutcome::BlockedByConflict,
                vec![FreshnessAcceptanceIssue::ExistingConflict],
            );
        }

        let Some(highest) = self.highest_accepted_epoch else {
            let mut next = self.clone();
            next.highest_accepted_epoch = Some(marker.epoch);
            next.record_marker(marker);
            next.prune_to_window();
            return FreshnessAcceptanceDecision {
                state: next,
                outcome: FreshnessAcceptanceOutcome::Advanced,
                issues: vec![],
            };
        };

        if marker.epoch > highest {
            let mut next = self.clone();
            next.highest_accepted_epoch = Some(marker.epoch);
            next.record_marker(marker);
            next.prune_to_window();
            return FreshnessAcceptanceDecision {
                state: next,
                outcome: FreshnessAcceptanceOutcome::Advanced,
                issues: vec![],
            };
        }

        let already_seen = self
            .marker_digests
            .get(&marker.epoch)
            .is_some_and(|digests| digests.contains(&marker.marker_digest));

        if marker.epoch == highest {
            if already_seen {
                return self.decision(FreshnessAcceptanceOutcome::Duplicate, vec![]);
            }
            let mut next = self.clone();
            next.record_marker(marker);
            return FreshnessAcceptanceDecision {
                state: next,
                outcome: FreshnessAcceptanceOutcome::Conflicted,
                issues: vec![FreshnessAcceptanceIssue::ConflictingMarker],
            };
        }

        if highest.saturating_sub(marker.epoch) > policy.acceptance_window {
            return self.decision(
                FreshnessAcceptanceOutcome::Rollback,
                vec![FreshnessAcceptanceIssue::EpochOutsideAcceptanceWindow],
            );
        }

        if already_seen {
            return self.decision(FreshnessAcceptanceOutcome::Duplicate, vec![]);
        }

        let mut next = self.clone();
        next.record_marker(marker);
        if next.status_conflicted() {
            FreshnessAcceptanceDecision {
                state: next,
                outcome: FreshnessAcceptanceOutcome::Conflicted,
                issues: vec![FreshnessAcceptanceIssue::ConflictingMarker],
            }
        } else {
            FreshnessAcceptanceDecision {
                state: next,
                outcome: FreshnessAcceptanceOutcome::AcceptedWithinWindow,
                issues: vec![],
            }
        }
    }

    pub fn reconcile(
        &self,
        policy: &FreshnessAcceptancePolicy,
        other: &FreshnessAcceptanceState,
    ) -> FreshnessReconciliationDecision {
        if !self.matches_policy(policy) || !other.matches_policy(policy) {
            return FreshnessReconciliationDecision {
                state: None,
                outcome: FreshnessReconciliationOutcome::Quarantined,
                issues: vec![FreshnessAcceptanceIssue::PolicyMismatch],
            };
        }

        let highest = match (self.highest_accepted_epoch, other.highest_accepted_epoch) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        let mut merged = self.clone();
        merged.highest_accepted_epoch = highest;
        merged.conflicted |= other.conflicted;

        for (epoch, digests) in &other.marker_digests {
            merged
                .marker_digests
                .entry(*epoch)
                .or_default()
                .extend(digests.iter().cloned());
        }

        merged.prune_to_window();
        if merged.marker_digests.values().any(|set| set.len() > 1) {
            merged.conflicted = true;
        }

        if merged.status_conflicted() {
            FreshnessReconciliationDecision {
                state: Some(merged),
                outcome: FreshnessReconciliationOutcome::Conflicted,
                issues: vec![FreshnessAcceptanceIssue::ConflictingMarker],
            }
        } else {
            FreshnessReconciliationDecision {
                state: Some(merged),
                outcome: FreshnessReconciliationOutcome::Merged,
                issues: vec![],
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAcceptanceDecision {
    pub state: FreshnessAcceptanceState,
    pub outcome: FreshnessAcceptanceOutcome,
    pub issues: Vec<FreshnessAcceptanceIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessReconciliationOutcome {
    Merged,
    Conflicted,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessReconciliationDecision {
    pub state: Option<FreshnessAcceptanceState>,
    pub outcome: FreshnessReconciliationOutcome,
    pub issues: Vec<FreshnessAcceptanceIssue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> FreshnessAcceptancePolicy {
        FreshnessAcceptancePolicy {
            schema_version: "0.1".into(),
            policy_id: "freshness-window-v1".into(),
            expected_scheme: "epoch-marker-v1".into(),
            expected_source_id: "epoch-bell-1".into(),
            minimum_accepted_epoch: 7,
            acceptance_window: 2,
        }
    }

    fn marker(epoch: u64, digest: &str) -> FreshnessMarker {
        FreshnessMarker {
            scheme: "epoch-marker-v1".into(),
            source_id: "epoch-bell-1".into(),
            epoch,
            marker_digest: digest.into(),
        }
    }

    fn state() -> FreshnessAcceptanceState {
        FreshnessAcceptanceState::new(&policy()).unwrap()
    }

    fn apply(
        state: &FreshnessAcceptanceState,
        epoch: u64,
        digest: &str,
    ) -> FreshnessAcceptanceDecision {
        state.apply(&policy(), &marker(epoch, digest))
    }

    #[test]
    fn first_marker_initializes_cursor() {
        let d = apply(&state(), 7, "m7");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::Advanced);
        assert_eq!(d.state.highest_accepted_epoch, Some(7));
    }

    #[test]
    fn reordered_marker_inside_window_does_not_advance() {
        let s7 = apply(&state(), 7, "m7").state;
        let s9 = apply(&s7, 9, "m9").state;
        let d = apply(&s9, 8, "m8");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::AcceptedWithinWindow);
        assert_eq!(d.state.highest_accepted_epoch, Some(9));
    }

    #[test]
    fn exact_replay_is_idempotent() {
        let s7 = apply(&state(), 7, "m7").state;
        let d = apply(&s7, 7, "m7");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::Duplicate);
        assert_eq!(d.state, s7);
    }

    #[test]
    fn old_marker_is_rollback_evidence() {
        let s7 = apply(&state(), 7, "m7").state;
        let s10 = apply(&s7, 10, "m10").state;
        let d = apply(&s10, 7, "m7");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::Rollback);
    }

    #[test]
    fn same_epoch_different_digest_latches_conflict() {
        let s7 = apply(&state(), 7, "m7").state;
        let d = apply(&s7, 7, "different");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::Conflicted);
        assert!(d.state.conflicted);
        assert_eq!(d.state.marker_digests[&7].len(), 2);
    }

    #[test]
    fn conflict_blocks_future_advancement() {
        let s7 = apply(&state(), 7, "m7").state;
        let conflict = apply(&s7, 7, "different").state;
        let d = apply(&conflict, 8, "m8");
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::BlockedByConflict);
    }

    #[test]
    fn offline_reconciliation_is_commutative() {
        let a = apply(&state(), 8, "m8").state;
        let b = apply(&state(), 9, "m9").state;
        let ab = a.reconcile(&policy(), &b);
        let ba = b.reconcile(&policy(), &a);
        assert_eq!(ab.outcome, FreshnessReconciliationOutcome::Merged);
        assert_eq!(ab.state, ba.state);
    }

    #[test]
    fn offline_reconciliation_preserves_conflicting_digests() {
        let a = apply(&state(), 7, "m7-a").state;
        let b = apply(&state(), 7, "m7-b").state;
        let merged = a.reconcile(&policy(), &b);
        assert_eq!(merged.outcome, FreshnessReconciliationOutcome::Conflicted);
        let state = merged.state.unwrap();
        assert_eq!(
            state.marker_digests[&7],
            BTreeSet::from(["m7-a".to_string(), "m7-b".to_string()])
        );
    }

    #[test]
    fn policy_change_is_quarantined() {
        let mut changed = policy();
        changed.acceptance_window = 3;
        let d = state().apply(&changed, &marker(7, "m7"));
        assert_eq!(d.outcome, FreshnessAcceptanceOutcome::Quarantined);
        assert!(d.issues.contains(&FreshnessAcceptanceIssue::PolicyMismatch));
    }
}
