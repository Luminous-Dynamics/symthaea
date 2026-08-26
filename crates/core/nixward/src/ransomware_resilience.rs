// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Nixward evidence contract for host reconstructibility and post-recovery verification.
//!
//! This module deliberately contains no recovery executor and no backup implementation.
//! It gives Nixward a stable, serializable vocabulary for proving what host state was
//! declared, what was observed, what recovery action was attempted, and whether the
//! resulting state actually matched the intended target.
//!
//! Schema v3 makes every decision-relevant observation claim-bound, exercise-bound,
//! subject-bound, time-bound, revision-bound, and digest-bearing. A bare
//! `passed = true` value or successful command exit is never reconstruction proof.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const HOST_RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION: u16 = 3;

/// Identity of a NixOS host state relevant to reconstruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStateIdentity {
    /// NixOS generation number when known.
    pub generation: Option<u64>,
    /// Resolved system profile/store path, for example `/nix/store/...-nixos-system-host-...`.
    pub system_profile: String,
    /// Source revision or other declarative configuration identity when available.
    pub configuration_revision: Option<String>,
    /// Optional cryptographic digest over a caller-defined canonical closure manifest.
    /// Nixward does not invent the digest formula here; the producer must document it.
    pub closure_digest: Option<String>,
}

impl HostStateIdentity {
    /// A state is minimally addressable when it names a concrete system profile.
    pub fn is_addressable(&self) -> bool {
        !self.system_profile.trim().is_empty()
    }
}

/// Result of comparing declared/target state with observed state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DriftAssessment {
    InSync,
    Drifted { differences: Vec<String> },
    Unproven { missing_evidence: Vec<String> },
}

/// Compare a desired/declared state with an observed state without overstating certainty.
///
/// The system profile is required on both sides. Optional revision/digest fields are compared
/// when supplied by either side; if one side supplies one and the other does not, the result is
/// UNPROVEN rather than silently ignoring the missing identity evidence.
pub fn assess_drift(expected: &HostStateIdentity, observed: &HostStateIdentity) -> DriftAssessment {
    let mut missing = Vec::new();

    if !expected.is_addressable() {
        missing.push("expected.system_profile".to_string());
    }
    if !observed.is_addressable() {
        missing.push("observed.system_profile".to_string());
    }

    match (&expected.configuration_revision, &observed.configuration_revision) {
        (Some(_), None) => missing.push("observed.configuration_revision".to_string()),
        (None, Some(_)) => missing.push("expected.configuration_revision".to_string()),
        _ => {}
    }

    match (&expected.closure_digest, &observed.closure_digest) {
        (Some(_), None) => missing.push("observed.closure_digest".to_string()),
        (None, Some(_)) => missing.push("expected.closure_digest".to_string()),
        _ => {}
    }

    if !missing.is_empty() {
        return DriftAssessment::Unproven {
            missing_evidence: missing,
        };
    }

    let mut differences = Vec::new();
    if expected.system_profile != observed.system_profile {
        differences.push("system_profile".to_string());
    }
    if expected.configuration_revision != observed.configuration_revision {
        differences.push("configuration_revision".to_string());
    }
    if expected.closure_digest != observed.closure_digest {
        differences.push("closure_digest".to_string());
    }

    if differences.is_empty() {
        DriftAssessment::InSync
    } else {
        DriftAssessment::Drifted { differences }
    }
}

/// Recovery transition Nixward observed or orchestrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryActionKind {
    RebuildSwitch,
    BootGeneration,
    Rollback,
    Reinstall,
}

/// Decision-relevant claim made by a reconstruction evidence artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReconstructionEvidenceClaim {
    RecoveryAction { action: RecoveryActionKind },
    ObservedPostState,
    PostRecoveryCheck { name: String },
}

impl ReconstructionEvidenceClaim {
    fn is_complete(&self) -> bool {
        match self {
            Self::PostRecoveryCheck { name } => !name.trim().is_empty(),
            _ => true,
        }
    }
}

/// Inspectable evidence emitted by the action/observer/check layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionEvidenceRef {
    pub claim: ReconstructionEvidenceClaim,
    pub locator: String,
    pub digest: String,
    pub producer_revision: String,
    pub exercise_id: String,
    /// Stable exercise-local host/target identifier. It should not be a raw
    /// hardware serial or other unnecessary persistent identifier.
    pub subject_id: String,
    pub captured_at_unix_ms: u64,
}

impl ReconstructionEvidenceRef {
    fn is_complete(&self) -> bool {
        self.claim.is_complete()
            && !self.locator.trim().is_empty()
            && !self.digest.trim().is_empty()
            && !self.producer_revision.trim().is_empty()
            && !self.exercise_id.trim().is_empty()
            && !self.subject_id.trim().is_empty()
    }

    fn proves(
        &self,
        exercise_id: &str,
        subject_id: &str,
        expected_claim: &ReconstructionEvidenceClaim,
        started_at_unix_ms: u64,
        finished_at_unix_ms: u64,
    ) -> bool {
        self.is_complete()
            && self.exercise_id == exercise_id
            && self.subject_id == subject_id
            && &self.claim == expected_claim
            && started_at_unix_ms <= self.captured_at_unix_ms
            && self.captured_at_unix_ms <= finished_at_unix_ms
    }
}

/// Whether the recovery action actually mutated a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoveryExecutionMode {
    /// The action was executed against the intended lab/host target.
    Live,
    /// The action was simulated or rendered without changing the target.
    DryRun,
    /// The producer did not establish whether mutation occurred.
    Unknown,
}

/// Stable action-layer receipt for one recovery attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryActionReceipt {
    pub execution_mode: RecoveryExecutionMode,
    /// Runner-supplied timestamps. They are useful for measured RTO but are not
    /// trusted wall-clock attestations by themselves.
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    /// Claim-bound inspectable evidence that the recovery action actually ran.
    pub evidence: Option<ReconstructionEvidenceRef>,
}

impl RecoveryActionReceipt {
    fn proven_live_window(
        &self,
        exercise_id: &str,
        subject_id: &str,
        action: RecoveryActionKind,
    ) -> Option<(u64, u64)> {
        if self.execution_mode != RecoveryExecutionMode::Live {
            return None;
        }
        let (Some(start), Some(finish)) = (self.started_at_unix_ms, self.finished_at_unix_ms) else {
            return None;
        };
        if start > finish {
            return None;
        }
        let evidence = self.evidence.as_ref()?;
        if !evidence.proves(
            exercise_id,
            subject_id,
            &ReconstructionEvidenceClaim::RecoveryAction { action },
            start,
            finish,
        ) {
            return None;
        }
        Some((start, finish))
    }
}

/// One explicit post-recovery assertion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostRecoveryCheck {
    pub name: String,
    pub passed: bool,
    #[serde(default)]
    pub detail: String,
    /// A check outcome is not a fact without independently inspectable evidence.
    pub evidence: Option<ReconstructionEvidenceRef>,
}

impl PostRecoveryCheck {
    fn evidence_is_applicable(
        &self,
        exercise_id: &str,
        subject_id: &str,
        started_at_unix_ms: u64,
        finished_at_unix_ms: u64,
    ) -> bool {
        if self.name.trim().is_empty() {
            return false;
        }
        self.evidence.as_ref().is_some_and(|evidence| {
            evidence.proves(
                exercise_id,
                subject_id,
                &ReconstructionEvidenceClaim::PostRecoveryCheck {
                    name: self.name.clone(),
                },
                started_at_unix_ms,
                finished_at_unix_ms,
            )
        })
    }
}

/// High-level decision about whether the host reconstruction was demonstrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReconstructionOutcome {
    Verified,
    Failed,
    Unproven,
}

/// Evidence for one attempt to return a host to a known declared state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostReconstructionEvidence {
    pub schema_version: u16,
    pub exercise_id: String,
    pub subject_id: String,
    pub action: RecoveryActionKind,
    pub before: HostStateIdentity,
    pub target: HostStateIdentity,
    pub after: Option<HostStateIdentity>,
    /// Evidence that the `after` identity was actually observed during this run.
    pub post_state_evidence: Option<ReconstructionEvidenceRef>,
    /// Explicit checks such as service health, filesystem mount state, or application probes.
    #[serde(default)]
    pub post_checks: Vec<PostRecoveryCheck>,
    pub action_receipt: Option<RecoveryActionReceipt>,
}

impl HostReconstructionEvidence {
    /// Evaluate reconstruction conservatively.
    ///
    /// VERIFIED requires:
    /// - schema v3 plus non-empty exercise and subject ids;
    /// - claim-bound evidence proving the exact LIVE recovery action and run window;
    /// - an observed post-recovery state with claim-bound evidence for the same subject/run;
    /// - exact in-sync identity under [`assess_drift`];
    /// - at least one uniquely named post-recovery check;
    /// - every post-recovery check carrying matching claim evidence and passing.
    ///
    /// A dry-run, wrong-run artifact, wrong-subject artifact, wrong-action/claim
    /// artifact, missing digest/revision, duplicate check, or unevidenced result
    /// is UNPROVEN. Demonstrated live post-state drift or an evidenced failed
    /// post-check yields FAILED.
    pub fn outcome(&self) -> ReconstructionOutcome {
        if self.schema_version != HOST_RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION
            || self.exercise_id.trim().is_empty()
            || self.subject_id.trim().is_empty()
            || !self.before.is_addressable()
            || !self.target.is_addressable()
        {
            return ReconstructionOutcome::Unproven;
        }

        let Some(receipt) = &self.action_receipt else {
            return ReconstructionOutcome::Unproven;
        };
        let Some((started_at_unix_ms, finished_at_unix_ms)) = receipt.proven_live_window(
            &self.exercise_id,
            &self.subject_id,
            self.action,
        ) else {
            return ReconstructionOutcome::Unproven;
        };

        let Some(after) = &self.after else {
            return ReconstructionOutcome::Unproven;
        };
        let post_state_is_proven = self.post_state_evidence.as_ref().is_some_and(|evidence| {
            evidence.proves(
                &self.exercise_id,
                &self.subject_id,
                &ReconstructionEvidenceClaim::ObservedPostState,
                started_at_unix_ms,
                finished_at_unix_ms,
            )
        });
        if !post_state_is_proven {
            return ReconstructionOutcome::Unproven;
        }

        match assess_drift(&self.target, after) {
            DriftAssessment::Drifted { .. } => return ReconstructionOutcome::Failed,
            DriftAssessment::Unproven { .. } => return ReconstructionOutcome::Unproven,
            DriftAssessment::InSync => {}
        }

        if self.post_checks.is_empty() {
            return ReconstructionOutcome::Unproven;
        }

        let mut names = BTreeSet::new();
        for check in &self.post_checks {
            if !names.insert(check.name.as_str())
                || !check.evidence_is_applicable(
                    &self.exercise_id,
                    &self.subject_id,
                    started_at_unix_ms,
                    finished_at_unix_ms,
                )
            {
                return ReconstructionOutcome::Unproven;
            }
        }

        if self.post_checks.iter().any(|check| !check.passed) {
            return ReconstructionOutcome::Failed;
        }

        ReconstructionOutcome::Verified
    }
}

#[cfg(feature = "native")]
/// Observe the currently activated NixOS system profile without fabricating
/// configuration-revision or closure-digest evidence.
///
/// The `/run/current-system` link is the concrete activated system identity.
/// Generation discovery is best-effort because the store path is the stronger
/// reconstruction anchor; if generation enumeration fails, the generation field
/// remains `None` rather than making up a value.
pub fn observe_current_host_state() -> std::io::Result<HostStateIdentity> {
    use crate::observe::generations::GenerationObserver;

    let system_profile = std::fs::read_link("/run/current-system")?
        .to_string_lossy()
        .into_owned();
    let generation = GenerationObserver::current_generation().ok();

    Ok(HostStateIdentity {
        generation,
        system_profile,
        configuration_revision: None,
        closure_digest: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXERCISE_ID: &str = "exercise-001";
    const SUBJECT_ID: &str = "host:web-01";

    fn state(profile: &str) -> HostStateIdentity {
        HostStateIdentity {
            generation: Some(42),
            system_profile: profile.to_string(),
            configuration_revision: Some("git:abc123".to_string()),
            closure_digest: Some("sha256:deadbeef".to_string()),
        }
    }

    fn evidence(
        claim: ReconstructionEvidenceClaim,
        label: &str,
        captured: u64,
    ) -> ReconstructionEvidenceRef {
        ReconstructionEvidenceRef {
            claim,
            locator: format!("receipt:{label}"),
            digest: format!("sha256:{label}"),
            producer_revision: "git:producer-abc123".to_string(),
            exercise_id: EXERCISE_ID.to_string(),
            subject_id: SUBJECT_ID.to_string(),
            captured_at_unix_ms: captured,
        }
    }

    fn live_receipt() -> RecoveryActionReceipt {
        RecoveryActionReceipt {
            execution_mode: RecoveryExecutionMode::Live,
            started_at_unix_ms: Some(1_000),
            finished_at_unix_ms: Some(2_000),
            evidence: Some(evidence(
                ReconstructionEvidenceClaim::RecoveryAction {
                    action: RecoveryActionKind::RebuildSwitch,
                },
                "recovery-action",
                1_100,
            )),
        }
    }

    fn check(name: &str, passed: bool) -> PostRecoveryCheck {
        PostRecoveryCheck {
            name: name.to_string(),
            passed,
            detail: String::new(),
            evidence: Some(evidence(
                ReconstructionEvidenceClaim::PostRecoveryCheck {
                    name: name.to_string(),
                },
                name,
                1_900,
            )),
        }
    }

    fn reconstruction() -> HostReconstructionEvidence {
        let target = state("/nix/store/system-a");
        HostReconstructionEvidence {
            schema_version: HOST_RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
            exercise_id: EXERCISE_ID.to_string(),
            subject_id: SUBJECT_ID.to_string(),
            action: RecoveryActionKind::RebuildSwitch,
            before: state("/nix/store/system-compromised"),
            target: target.clone(),
            after: Some(target),
            post_state_evidence: Some(evidence(
                ReconstructionEvidenceClaim::ObservedPostState,
                "post-state",
                1_800,
            )),
            post_checks: vec![check("critical-service-health", true)],
            action_receipt: Some(live_receipt()),
        }
    }

    #[test]
    fn identical_addressable_states_are_in_sync() {
        let expected = state("/nix/store/system-a");
        assert_eq!(assess_drift(&expected, &expected), DriftAssessment::InSync);
    }

    #[test]
    fn changed_profile_is_explicit_drift() {
        let expected = state("/nix/store/system-a");
        let observed = state("/nix/store/system-b");
        assert_eq!(
            assess_drift(&expected, &observed),
            DriftAssessment::Drifted {
                differences: vec!["system_profile".to_string()]
            }
        );
    }

    #[test]
    fn missing_optional_identity_on_one_side_is_unproven() {
        let expected = state("/nix/store/system-a");
        let mut observed = expected.clone();
        observed.closure_digest = None;

        assert_eq!(
            assess_drift(&expected, &observed),
            DriftAssessment::Unproven {
                missing_evidence: vec!["observed.closure_digest".to_string()]
            }
        );
    }

    #[test]
    fn successful_live_rebuild_requires_claim_bound_evidence() {
        assert_eq!(reconstruction().outcome(), ReconstructionOutcome::Verified);
    }

    #[test]
    fn dry_run_cannot_verify_reconstruction() {
        let mut reconstruction = reconstruction();
        reconstruction.action_receipt.as_mut().unwrap().execution_mode =
            RecoveryExecutionMode::DryRun;
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn failed_live_post_check_with_evidence_fails_reconstruction() {
        let mut reconstruction = reconstruction();
        reconstruction.post_checks[0].passed = false;
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Failed);
    }

    #[test]
    fn failed_check_without_evidence_is_unproven_not_failed() {
        let mut reconstruction = reconstruction();
        reconstruction.post_checks[0].passed = false;
        reconstruction.post_checks[0].evidence = None;
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn wrong_claim_cannot_satisfy_post_check() {
        let mut reconstruction = reconstruction();
        reconstruction.post_checks[0].evidence.as_mut().unwrap().claim =
            ReconstructionEvidenceClaim::ObservedPostState;
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn wrong_action_claim_cannot_prove_recovery_action() {
        let mut reconstruction = reconstruction();
        reconstruction
            .action_receipt
            .as_mut()
            .unwrap()
            .evidence
            .as_mut()
            .unwrap()
            .claim = ReconstructionEvidenceClaim::RecoveryAction {
            action: RecoveryActionKind::Rollback,
        };
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn foreign_exercise_action_receipt_is_unproven() {
        let mut reconstruction = reconstruction();
        reconstruction
            .action_receipt
            .as_mut()
            .unwrap()
            .evidence
            .as_mut()
            .unwrap()
            .exercise_id = "exercise-other".to_string();
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn foreign_subject_action_receipt_is_unproven() {
        let mut reconstruction = reconstruction();
        reconstruction
            .action_receipt
            .as_mut()
            .unwrap()
            .evidence
            .as_mut()
            .unwrap()
            .subject_id = "host:db-01".to_string();
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn matching_state_without_post_state_evidence_is_unproven() {
        let mut reconstruction = reconstruction();
        reconstruction.post_state_evidence = None;
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn reversed_receipt_timestamps_are_unproven() {
        let mut reconstruction = reconstruction();
        let receipt = reconstruction.action_receipt.as_mut().unwrap();
        receipt.started_at_unix_ms = Some(2_000);
        receipt.finished_at_unix_ms = Some(1_000);
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn duplicated_post_check_name_is_unproven() {
        let mut reconstruction = reconstruction();
        reconstruction
            .post_checks
            .push(check("critical-service-health", true));
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn evidenced_post_state_drift_is_failed() {
        let mut reconstruction = reconstruction();
        reconstruction.after = Some(state("/nix/store/system-b"));
        assert_eq!(reconstruction.outcome(), ReconstructionOutcome::Failed);
    }
}
