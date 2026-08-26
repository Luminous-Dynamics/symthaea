// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Nixward evidence contract for host reconstructibility and post-recovery verification.
//!
//! This module deliberately contains no recovery executor and no backup implementation.
//! It gives Nixward a stable, serializable vocabulary for proving what host state was
//! declared, what was observed, what recovery action was attempted, and whether the
//! resulting state actually matched the intended target.

use serde::{Deserialize, Serialize};

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
    /// Caller-supplied stable locator for logs/receipts produced by the action layer.
    pub locator: String,
    pub execution_mode: RecoveryExecutionMode,
    /// Runner-supplied timestamps. They are useful for measured RTO but are not
    /// trusted wall-clock attestations by themselves.
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
}

impl RecoveryActionReceipt {
    pub fn is_addressable(&self) -> bool {
        !self.locator.trim().is_empty()
    }

    pub fn has_valid_time_order(&self) -> bool {
        match (self.started_at_unix_ms, self.finished_at_unix_ms) {
            (Some(start), Some(finish)) => start <= finish,
            _ => true,
        }
    }
}

/// One explicit post-recovery assertion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostRecoveryCheck {
    pub name: String,
    pub passed: bool,
    #[serde(default)]
    pub detail: String,
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
    pub action: RecoveryActionKind,
    pub before: HostStateIdentity,
    pub target: HostStateIdentity,
    pub after: Option<HostStateIdentity>,
    /// Explicit checks such as service health, filesystem mount state, or application probes.
    #[serde(default)]
    pub post_checks: Vec<PostRecoveryCheck>,
    pub action_receipt: Option<RecoveryActionReceipt>,
}

impl HostReconstructionEvidence {
    /// Evaluate reconstruction conservatively.
    ///
    /// VERIFIED requires:
    /// - an addressable receipt proving a LIVE (not dry-run) action;
    /// - an observed post-recovery state;
    /// - exact in-sync identity under [`assess_drift`];
    /// - at least one explicit post-recovery check;
    /// - every post-recovery check passing.
    ///
    /// A dry-run or unknown execution mode is never promoted to either VERIFIED
    /// or FAILED because it did not establish a real host transition. Demonstrated
    /// live post-state drift or a failed live post-check yields FAILED. Missing
    /// evidence remains UNPROVEN.
    pub fn outcome(&self) -> ReconstructionOutcome {
        let Some(receipt) = &self.action_receipt else {
            return ReconstructionOutcome::Unproven;
        };

        if !receipt.is_addressable()
            || !receipt.has_valid_time_order()
            || receipt.execution_mode != RecoveryExecutionMode::Live
        {
            return ReconstructionOutcome::Unproven;
        }

        if self.post_checks.iter().any(|check| !check.passed) {
            return ReconstructionOutcome::Failed;
        }

        let Some(after) = &self.after else {
            return ReconstructionOutcome::Unproven;
        };

        match assess_drift(&self.target, after) {
            DriftAssessment::Drifted { .. } => return ReconstructionOutcome::Failed,
            DriftAssessment::Unproven { .. } => return ReconstructionOutcome::Unproven,
            DriftAssessment::InSync => {}
        }

        if self.post_checks.is_empty() {
            return ReconstructionOutcome::Unproven;
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

    fn state(profile: &str) -> HostStateIdentity {
        HostStateIdentity {
            generation: Some(42),
            system_profile: profile.to_string(),
            configuration_revision: Some("git:abc123".to_string()),
            closure_digest: Some("sha256:deadbeef".to_string()),
        }
    }

    fn live_receipt(locator: &str) -> RecoveryActionReceipt {
        RecoveryActionReceipt {
            locator: locator.to_string(),
            execution_mode: RecoveryExecutionMode::Live,
            started_at_unix_ms: Some(1_000),
            finished_at_unix_ms: Some(2_000),
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
    fn successful_live_rebuild_requires_post_checks_and_receipt() {
        let target = state("/nix/store/system-a");
        let evidence = HostReconstructionEvidence {
            action: RecoveryActionKind::RebuildSwitch,
            before: state("/nix/store/system-compromised"),
            target: target.clone(),
            after: Some(target),
            post_checks: vec![PostRecoveryCheck {
                name: "critical-service-health".to_string(),
                passed: true,
                detail: "HTTP probe returned expected response".to_string(),
            }],
            action_receipt: Some(live_receipt("nixward-action:run-001")),
        };

        assert_eq!(evidence.outcome(), ReconstructionOutcome::Verified);
    }

    #[test]
    fn dry_run_cannot_verify_reconstruction() {
        let target = state("/nix/store/system-a");
        let mut receipt = live_receipt("nixward-action:dry-run-001");
        receipt.execution_mode = RecoveryExecutionMode::DryRun;
        let evidence = HostReconstructionEvidence {
            action: RecoveryActionKind::RebuildSwitch,
            before: state("/nix/store/system-b"),
            target: target.clone(),
            after: Some(target),
            post_checks: vec![PostRecoveryCheck {
                name: "synthetic-health".to_string(),
                passed: true,
                detail: String::new(),
            }],
            action_receipt: Some(receipt),
        };

        assert_eq!(evidence.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn failed_live_post_check_fails_reconstruction() {
        let target = state("/nix/store/system-a");
        let evidence = HostReconstructionEvidence {
            action: RecoveryActionKind::Rollback,
            before: state("/nix/store/system-b"),
            target: target.clone(),
            after: Some(target),
            post_checks: vec![PostRecoveryCheck {
                name: "database-integrity".to_string(),
                passed: false,
                detail: "integrity probe failed".to_string(),
            }],
            action_receipt: Some(live_receipt("nixward-action:run-002")),
        };

        assert_eq!(evidence.outcome(), ReconstructionOutcome::Failed);
    }

    #[test]
    fn matching_state_without_checks_is_unproven() {
        let target = state("/nix/store/system-a");
        let evidence = HostReconstructionEvidence {
            action: RecoveryActionKind::Reinstall,
            before: state("/nix/store/system-b"),
            target: target.clone(),
            after: Some(target),
            post_checks: Vec::new(),
            action_receipt: Some(live_receipt("spore:install-001")),
        };

        assert_eq!(evidence.outcome(), ReconstructionOutcome::Unproven);
    }

    #[test]
    fn reversed_receipt_timestamps_are_unproven() {
        let target = state("/nix/store/system-a");
        let mut receipt = live_receipt("nixward-action:run-003");
        receipt.started_at_unix_ms = Some(2_000);
        receipt.finished_at_unix_ms = Some(1_000);
        let evidence = HostReconstructionEvidence {
            action: RecoveryActionKind::Rollback,
            before: state("/nix/store/system-b"),
            target: target.clone(),
            after: Some(target),
            post_checks: vec![PostRecoveryCheck {
                name: "critical-service-health".to_string(),
                passed: true,
                detail: String::new(),
            }],
            action_receipt: Some(receipt),
        };

        assert_eq!(evidence.outcome(), ReconstructionOutcome::Unproven);
    }
}
