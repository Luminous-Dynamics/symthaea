// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Clean-host rehydration contracts for ransomware-recovery exercises.
//!
//! This module is intentionally separate from [`crate::persistence`]. A
//! `SporeCheckpoint` preserves consciousness-kernel state; a
//! [`HostRehydrationManifest`] describes rebuilding an operating-system host
//! from externally trusted installation media and a pinned declaration. Mixing
//! those concepts would let cognitive persistence accidentally become evidence
//! that a compromised machine was cleanly reconstructed.
//!
//! The types here are evidence contracts, not an installer implementation.
//! Spore's existing installer/relay remains responsible for disk preparation,
//! NixOS installation, Secure Boot guidance, and first-boot orchestration.
//! Nixward remains responsible for proving the resulting Nix generation and
//! configuration identity. A backup/recovery provider remains responsible for
//! protected generations and restore authorization.
//!
//! In particular, [`ProtectedRestoreRef`] contains only opaque recovery
//! references. It has no credentials, delete operation, retention mutation, or
//! overwrite authority by construction.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema version for both host rehydration manifests and receipts.
pub const HOST_REHYDRATION_SCHEMA_VERSION: u16 = 1;

/// Immutable artifact or declaration identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableArtifactRef {
    /// Human- or machine-resolvable locator. May be a flake URI, artifact id,
    /// content-addressed object id, or other non-secret reference.
    pub locator: String,
    /// Content digest over the referenced artifact/declaration.
    pub digest: String,
    /// Immutable source/build revision that produced the artifact.
    pub revision: String,
}

impl ImmutableArtifactRef {
    /// Whether all provenance-bearing fields are present.
    pub fn is_complete(&self) -> bool {
        !self.locator.trim().is_empty()
            && !self.digest.trim().is_empty()
            && !self.revision.trim().is_empty()
    }
}

/// Evidence locator emitted by an existing subsystem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RehydrationEvidenceRef {
    pub locator: String,
    pub digest: String,
    pub producer_revision: String,
}

impl RehydrationEvidenceRef {
    pub fn is_complete(&self) -> bool {
        !self.locator.trim().is_empty()
            && !self.digest.trim().is_empty()
            && !self.producer_revision.trim().is_empty()
    }
}

/// Privacy-preserving identity for the physical/virtual recovery target.
///
/// Raw serial numbers are deliberately absent. The installer may hash stable
/// hardware/disk observations locally and place only those digests here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryTargetIdentity {
    /// Digest over the stable hardware identity selected by the installer.
    pub hardware_identity_digest: String,
    /// Architecture expected by the pinned system declaration (`x86_64`,
    /// `aarch64`, ...). Kept as data rather than an enum so new architectures
    /// do not require a schema break.
    pub architecture: String,
    /// Digest over a stable disk identity (serial/WWN/NVMe identity or a
    /// virtual-disk identity), never merely `/dev/sdX`.
    pub disk_identity_digest: String,
    /// Optional lower bound used to detect accidentally selecting a different
    /// smaller target even if an upstream identity source regresses.
    pub expected_min_disk_bytes: Option<u64>,
}

impl RecoveryTargetIdentity {
    pub fn is_complete(&self) -> bool {
        !self.hardware_identity_digest.trim().is_empty()
            && !self.architecture.trim().is_empty()
            && !self.disk_identity_digest.trim().is_empty()
            && self.expected_min_disk_bytes != Some(0)
    }

    fn matches_observation(&self, observed: &ObservedRecoveryTarget) -> bool {
        self.hardware_identity_digest == observed.hardware_identity_digest
            && self.architecture == observed.architecture
            && self.disk_identity_digest == observed.disk_identity_digest
            && self
                .expected_min_disk_bytes
                .is_none_or(|minimum| observed.disk_bytes >= minimum)
    }
}

/// Target identity observed by the installer immediately before destructive
/// preparation. The receipt binds the actual operation to this observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedRecoveryTarget {
    pub hardware_identity_digest: String,
    pub architecture: String,
    pub disk_identity_digest: String,
    pub disk_bytes: u64,
}

impl ObservedRecoveryTarget {
    fn is_complete(&self) -> bool {
        !self.hardware_identity_digest.trim().is_empty()
            && !self.architecture.trim().is_empty()
            && !self.disk_identity_digest.trim().is_empty()
            && self.disk_bytes > 0
    }
}

/// Opaque reference to protected data that a recovery provider may restore.
///
/// There is intentionally no credential, bearer token, deletion API, retention
/// mutation, or overwrite operation in this type. The recovery provider owns
/// those capabilities and should expose only a scoped restore path to the
/// exercise runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedRestoreRef {
    /// Stable recovery-provider identity (`restic-vault`, `zfs-recovery`, ...).
    pub provider: String,
    /// Opaque immutable generation/snapshot reference.
    pub generation_ref: String,
    /// Digest or signed manifest commitment describing the protected contents.
    pub manifest_digest: String,
}

impl ProtectedRestoreRef {
    pub fn is_complete(&self) -> bool {
        !self.provider.trim().is_empty()
            && !self.generation_ref.trim().is_empty()
            && !self.manifest_digest.trim().is_empty()
    }
}

/// V1 deliberately has no production-target mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoveryScope {
    /// Disposable VM/bare-metal lab target created for the exercise.
    DisposableLab,
    /// Dedicated recovery machine or recovery disk that is not serving live
    /// production traffic while the exercise runs.
    DedicatedRecoveryTarget,
}

/// A post-install assertion the exercise runner must obtain evidence for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PostInstallCheckKind {
    /// Nixward proves the live generation/configuration matches the pinned
    /// declaration in this manifest.
    SystemGenerationMatchesDeclaration,
    /// One required service is healthy after reconstruction.
    RequiredServiceHealthy { service: String },
    /// Secure Boot posture matches the exercise expectation.
    SecureBootPosture,
    /// Restored protected data matches the provider's manifest/commitment.
    ProtectedDataIntegrity,
    /// Network isolation/egress assertion proven by the Network Twin or an
    /// authorized enforcement/probe adapter.
    NetworkIsolation,
    /// Extension point; the id remains part of the evidence contract.
    Custom { kind: String },
}

impl PostInstallCheckKind {
    fn is_complete(&self) -> bool {
        match self {
            Self::RequiredServiceHealthy { service } => !service.trim().is_empty(),
            Self::Custom { kind } => !kind.trim().is_empty(),
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostInstallCheckSpec {
    pub check_id: String,
    pub kind: PostInstallCheckKind,
    /// Optional digest/commitment for a check-specific expectation. For
    /// example, a service-set manifest or network-intent revision.
    pub expectation_digest: Option<String>,
}

impl PostInstallCheckSpec {
    fn is_complete(&self) -> bool {
        !self.check_id.trim().is_empty()
            && self.kind.is_complete()
            && !self
                .expectation_digest
                .as_deref()
                .is_some_and(|digest| digest.trim().is_empty())
    }
}

/// Declarative input to one clean-host recovery exercise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRehydrationManifest {
    pub schema_version: u16,
    pub manifest_id: String,
    pub exercise_id: String,
    pub scope: RecoveryScope,
    /// Exact installer/recovery-environment artifact used to perform the
    /// rebuild. This prevents a mutable "latest ISO" from satisfying evidence.
    pub installer: ImmutableArtifactRef,
    /// Exact NixOS/flake declaration the rebuilt host must converge to.
    pub system_declaration: ImmutableArtifactRef,
    /// Target selected before destructive preparation.
    pub target: RecoveryTargetIdentity,
    /// Optional protected data restore. Opaque and restore-only by contract.
    pub protected_restore: Option<ProtectedRestoreRef>,
    #[serde(default)]
    pub post_install_checks: Vec<PostInstallCheckSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HostRehydrationManifestError {
    UnsupportedSchemaVersion { found: u16 },
    EmptyManifestId,
    EmptyExerciseId,
    IncompleteInstallerArtifact,
    IncompleteSystemDeclaration,
    IncompleteTargetIdentity,
    IncompleteProtectedRestore,
    MissingPostInstallCheck,
    IncompletePostInstallCheck { check_id: String },
    DuplicatePostInstallCheck { check_id: String },
    MissingGenerationVerification,
    MissingProtectedDataIntegrityCheck,
}

impl HostRehydrationManifest {
    /// Structural validation of a recovery manifest.
    pub fn validation_errors(&self) -> Vec<HostRehydrationManifestError> {
        let mut errors = Vec::new();
        if self.schema_version != HOST_REHYDRATION_SCHEMA_VERSION {
            errors.push(HostRehydrationManifestError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self.manifest_id.trim().is_empty() {
            errors.push(HostRehydrationManifestError::EmptyManifestId);
        }
        if self.exercise_id.trim().is_empty() {
            errors.push(HostRehydrationManifestError::EmptyExerciseId);
        }
        if !self.installer.is_complete() {
            errors.push(HostRehydrationManifestError::IncompleteInstallerArtifact);
        }
        if !self.system_declaration.is_complete() {
            errors.push(HostRehydrationManifestError::IncompleteSystemDeclaration);
        }
        if !self.target.is_complete() {
            errors.push(HostRehydrationManifestError::IncompleteTargetIdentity);
        }
        if self
            .protected_restore
            .as_ref()
            .is_some_and(|restore| !restore.is_complete())
        {
            errors.push(HostRehydrationManifestError::IncompleteProtectedRestore);
        }
        if self.post_install_checks.is_empty() {
            errors.push(HostRehydrationManifestError::MissingPostInstallCheck);
        }

        let mut check_ids = BTreeSet::new();
        let mut has_generation_check = false;
        let mut has_protected_data_integrity = false;
        for check in &self.post_install_checks {
            if !check.is_complete() {
                errors.push(HostRehydrationManifestError::IncompletePostInstallCheck {
                    check_id: check.check_id.clone(),
                });
            }
            if !check_ids.insert(check.check_id.clone()) {
                errors.push(HostRehydrationManifestError::DuplicatePostInstallCheck {
                    check_id: check.check_id.clone(),
                });
            }
            has_generation_check |= matches!(
                check.kind,
                PostInstallCheckKind::SystemGenerationMatchesDeclaration
            );
            has_protected_data_integrity |=
                matches!(check.kind, PostInstallCheckKind::ProtectedDataIntegrity);
        }

        if !has_generation_check {
            errors.push(HostRehydrationManifestError::MissingGenerationVerification);
        }
        if self.protected_restore.is_some() && !has_protected_data_integrity {
            errors.push(HostRehydrationManifestError::MissingProtectedDataIntegrityCheck);
        }
        errors
    }

    pub fn is_valid(&self) -> bool {
        self.validation_errors().is_empty()
    }
}

/// How the recovery operation was actually executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RehydrationExecutionMode {
    LiveDisposableLab,
    LiveDedicatedRecoveryTarget,
    DryRun,
    Unknown,
}

impl RehydrationExecutionMode {
    fn matches_scope(self, scope: RecoveryScope) -> bool {
        matches!(
            (self, scope),
            (Self::LiveDisposableLab, RecoveryScope::DisposableLab)
                | (
                    Self::LiveDedicatedRecoveryTarget,
                    RecoveryScope::DedicatedRecoveryTarget
                )
        )
    }
}

/// Where the rebuild environment booted from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoveryBootOrigin {
    /// Booted from the immutable installer artifact named in the manifest.
    ExternalInstaller,
    /// Booted from another dedicated recovery environment whose identity is
    /// independently evidenced.
    DedicatedRecoveryEnvironment,
    /// The previous runtime filesystem remained the trust root.
    PriorRuntime,
    Unknown,
}

impl RecoveryBootOrigin {
    fn proves_external_trust_root(self) -> bool {
        matches!(
            self,
            Self::ExternalInstaller | Self::DedicatedRecoveryEnvironment
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PreparationResult {
    Prepared,
    Failed,
    NotAttempted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetPreparationReceipt {
    pub result: PreparationResult,
    pub evidence: Option<RehydrationEvidenceRef>,
}

impl TargetPreparationReceipt {
    fn has_complete_evidence(&self) -> bool {
        self.evidence
            .as_ref()
            .is_some_and(RehydrationEvidenceRef::is_complete)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RestoreResult {
    NotRequested,
    Restored,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedRestoreReceipt {
    pub result: RestoreResult,
    /// Must remain opaque; the backup provider owns authorization and secrets.
    pub generation_ref: Option<String>,
    pub evidence: Option<RehydrationEvidenceRef>,
}

impl ProtectedRestoreReceipt {
    fn has_complete_evidence(&self) -> bool {
        self.evidence
            .as_ref()
            .is_some_and(RehydrationEvidenceRef::is_complete)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostInstallCheckResult {
    pub check_id: String,
    pub passed: bool,
    pub evidence: Option<RehydrationEvidenceRef>,
}

impl PostInstallCheckResult {
    fn has_complete_evidence(&self) -> bool {
        !self.check_id.trim().is_empty()
            && self
                .evidence
                .as_ref()
                .is_some_and(RehydrationEvidenceRef::is_complete)
    }
}

/// Evidence emitted after one rehydration attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRehydrationReceipt {
    pub schema_version: u16,
    pub manifest_id: String,
    pub exercise_id: String,
    pub execution_mode: RehydrationExecutionMode,
    pub boot_origin: RecoveryBootOrigin,
    /// Installer artifact actually observed by the runner.
    pub installer: ImmutableArtifactRef,
    /// Declaration actually supplied to the install/rebuild path.
    pub system_declaration: ImmutableArtifactRef,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    pub observed_target: ObservedRecoveryTarget,
    /// Evidence that destructive preparation happened from the recovery trust
    /// root rather than relying on the compromised runtime filesystem.
    pub preparation: TargetPreparationReceipt,
    /// Nixward-owned proof that resulting host state matches the declaration.
    pub nixward_reconstruction_evidence: Option<RehydrationEvidenceRef>,
    pub protected_restore: ProtectedRestoreReceipt,
    #[serde(default)]
    pub post_install_checks: Vec<PostInstallCheckResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RehydrationOutcome {
    /// Complete evidence proves the clean-host exercise succeeded.
    Verified,
    /// Complete evidence positively establishes a required operation/check
    /// failed during a live exercise.
    Failed { reasons: Vec<String> },
    /// Evidence is missing, ambiguous, mismatched, dry-run, or otherwise
    /// insufficient to establish success/failure.
    Unproven { reasons: Vec<String> },
}

impl HostRehydrationReceipt {
    /// Evaluate one receipt against the manifest it claims to satisfy.
    ///
    /// The evaluator is intentionally asymmetric. A complete live failure is
    /// `FAILED`; absence or ambiguity is `UNPROVEN`; `VERIFIED` requires every
    /// positive proof gate to pass.
    pub fn evaluate(&self, manifest: &HostRehydrationManifest) -> RehydrationOutcome {
        let mut unproven = Vec::new();
        let mut failed = Vec::new();

        if !manifest.is_valid() {
            unproven.push("manifest is invalid".to_string());
        }
        if self.schema_version != HOST_REHYDRATION_SCHEMA_VERSION {
            unproven.push("receipt schema version is unsupported".to_string());
        }
        if self.manifest_id != manifest.manifest_id || self.manifest_id.trim().is_empty() {
            unproven.push("receipt is not bound to the expected manifest".to_string());
        }
        if self.exercise_id != manifest.exercise_id || self.exercise_id.trim().is_empty() {
            unproven.push("receipt is not bound to the expected exercise".to_string());
        }
        if !self.execution_mode.matches_scope(manifest.scope) {
            unproven.push("receipt does not prove a live execution in the manifest scope".to_string());
        }
        if !self.boot_origin.proves_external_trust_root() {
            unproven.push("recovery did not prove an external recovery trust root".to_string());
        }
        if self.installer != manifest.installer || !self.installer.is_complete() {
            unproven.push("observed installer does not match the pinned installer".to_string());
        }
        if self.system_declaration != manifest.system_declaration
            || !self.system_declaration.is_complete()
        {
            unproven.push("applied system declaration does not match the pinned declaration".to_string());
        }
        if self.started_at_unix_ms > self.finished_at_unix_ms {
            unproven.push("receipt time window is invalid".to_string());
        }
        if !self.observed_target.is_complete()
            || !manifest.target.matches_observation(&self.observed_target)
        {
            unproven.push("observed recovery target does not match manifest target identity".to_string());
        }

        match self.preparation.result {
            PreparationResult::Prepared if self.preparation.has_complete_evidence() => {}
            PreparationResult::Failed if self.preparation.has_complete_evidence() => {
                failed.push("target preparation failed".to_string());
            }
            PreparationResult::Prepared | PreparationResult::Failed => {
                unproven.push("target preparation result lacks complete evidence".to_string());
            }
            PreparationResult::NotAttempted | PreparationResult::Unknown => {
                unproven.push("target was not proven cleanly prepared".to_string());
            }
        }

        if !self
            .nixward_reconstruction_evidence
            .as_ref()
            .is_some_and(RehydrationEvidenceRef::is_complete)
        {
            unproven.push("missing complete Nixward reconstruction evidence".to_string());
        }

        match (&manifest.protected_restore, &self.protected_restore.result) {
            (None, RestoreResult::NotRequested) => {}
            (None, _) => unproven.push(
                "receipt reports a protected restore not requested by the manifest".to_string(),
            ),
            (Some(expected), RestoreResult::Restored) => {
                if self.protected_restore.generation_ref.as_deref()
                    != Some(expected.generation_ref.as_str())
                    || !self.protected_restore.has_complete_evidence()
                {
                    unproven.push(
                        "protected restore is missing generation binding or evidence".to_string(),
                    );
                }
            }
            (Some(_), RestoreResult::Failed) if self.protected_restore.has_complete_evidence() => {
                failed.push("protected data restore failed".to_string());
            }
            (Some(_), RestoreResult::Failed) => unproven.push(
                "protected restore failure lacks complete evidence".to_string(),
            ),
            (Some(_), RestoreResult::NotRequested | RestoreResult::Unknown) => {
                unproven.push("required protected data restore was not proven".to_string());
            }
        }

        let mut expected: BTreeMap<&str, &PostInstallCheckSpec> = BTreeMap::new();
        for check in &manifest.post_install_checks {
            expected.insert(check.check_id.as_str(), check);
        }
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for result in &self.post_install_checks {
            *counts.entry(result.check_id.as_str()).or_default() += 1;
        }

        for result in &self.post_install_checks {
            if !expected.contains_key(result.check_id.as_str()) {
                unproven.push(format!(
                    "unexpected post-install check result: {}",
                    result.check_id
                ));
                continue;
            }
            if counts.get(result.check_id.as_str()).copied().unwrap_or_default() != 1 {
                continue;
            }
            if !result.has_complete_evidence() {
                unproven.push(format!(
                    "post-install check lacks complete evidence: {}",
                    result.check_id
                ));
            } else if !result.passed {
                failed.push(format!("post-install check failed: {}", result.check_id));
            }
        }

        for check_id in expected.keys() {
            match counts.get(check_id).copied().unwrap_or_default() {
                0 => unproven.push(format!("missing post-install check result: {check_id}")),
                1 => {}
                _ => unproven.push(format!("duplicate post-install check result: {check_id}")),
            }
        }

        // Do not turn a positively observed failure into UNPROVEN merely because
        // a separate field is incomplete. A complete required-operation/check
        // failure is itself enough to establish failure of this exercise.
        if !failed.is_empty() {
            return RehydrationOutcome::Failed { reasons: failed };
        }
        if !unproven.is_empty() {
            return RehydrationOutcome::Unproven { reasons: unproven };
        }
        RehydrationOutcome::Verified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(name: &str) -> ImmutableArtifactRef {
        ImmutableArtifactRef {
            locator: format!("artifact:{name}"),
            digest: format!("sha256:{name}-digest"),
            revision: format!("git:{name}-revision"),
        }
    }

    fn evidence(name: &str) -> RehydrationEvidenceRef {
        RehydrationEvidenceRef {
            locator: format!("receipt:{name}"),
            digest: format!("sha256:{name}-digest"),
            producer_revision: format!("git:{name}-producer"),
        }
    }

    fn manifest(with_restore: bool) -> HostRehydrationManifest {
        let mut post_install_checks = vec![
            PostInstallCheckSpec {
                check_id: "generation".to_string(),
                kind: PostInstallCheckKind::SystemGenerationMatchesDeclaration,
                expectation_digest: None,
            },
            PostInstallCheckSpec {
                check_id: "service".to_string(),
                kind: PostInstallCheckKind::RequiredServiceHealthy {
                    service: "case-management.service".to_string(),
                },
                expectation_digest: None,
            },
        ];
        if with_restore {
            post_install_checks.push(PostInstallCheckSpec {
                check_id: "restore-integrity".to_string(),
                kind: PostInstallCheckKind::ProtectedDataIntegrity,
                expectation_digest: Some("sha256:expected-data".to_string()),
            });
        }

        HostRehydrationManifest {
            schema_version: HOST_REHYDRATION_SCHEMA_VERSION,
            manifest_id: "rehydration-001".to_string(),
            exercise_id: "exercise-001".to_string(),
            scope: RecoveryScope::DisposableLab,
            installer: artifact("installer"),
            system_declaration: artifact("nixos-declaration"),
            target: RecoveryTargetIdentity {
                hardware_identity_digest: "sha256:hardware".to_string(),
                architecture: "x86_64-linux".to_string(),
                disk_identity_digest: "sha256:disk".to_string(),
                expected_min_disk_bytes: Some(64 * 1024 * 1024 * 1024),
            },
            protected_restore: with_restore.then(|| ProtectedRestoreRef {
                provider: "recovery-vault".to_string(),
                generation_ref: "generation:2026-08-26T10:00Z".to_string(),
                manifest_digest: "sha256:protected-manifest".to_string(),
            }),
            post_install_checks,
        }
    }

    fn receipt(manifest: &HostRehydrationManifest) -> HostRehydrationReceipt {
        HostRehydrationReceipt {
            schema_version: HOST_REHYDRATION_SCHEMA_VERSION,
            manifest_id: manifest.manifest_id.clone(),
            exercise_id: manifest.exercise_id.clone(),
            execution_mode: RehydrationExecutionMode::LiveDisposableLab,
            boot_origin: RecoveryBootOrigin::ExternalInstaller,
            installer: manifest.installer.clone(),
            system_declaration: manifest.system_declaration.clone(),
            started_at_unix_ms: 1_000,
            finished_at_unix_ms: 2_000,
            observed_target: ObservedRecoveryTarget {
                hardware_identity_digest: manifest.target.hardware_identity_digest.clone(),
                architecture: manifest.target.architecture.clone(),
                disk_identity_digest: manifest.target.disk_identity_digest.clone(),
                disk_bytes: 128 * 1024 * 1024 * 1024,
            },
            preparation: TargetPreparationReceipt {
                result: PreparationResult::Prepared,
                evidence: Some(evidence("disk-preparation")),
            },
            nixward_reconstruction_evidence: Some(evidence("nixward-reconstruction")),
            protected_restore: match &manifest.protected_restore {
                Some(restore) => ProtectedRestoreReceipt {
                    result: RestoreResult::Restored,
                    generation_ref: Some(restore.generation_ref.clone()),
                    evidence: Some(evidence("protected-restore")),
                },
                None => ProtectedRestoreReceipt {
                    result: RestoreResult::NotRequested,
                    generation_ref: None,
                    evidence: None,
                },
            },
            post_install_checks: manifest
                .post_install_checks
                .iter()
                .map(|check| PostInstallCheckResult {
                    check_id: check.check_id.clone(),
                    passed: true,
                    evidence: Some(evidence(&check.check_id)),
                })
                .collect(),
        }
    }

    #[test]
    fn manifest_requires_generation_check() {
        let mut manifest = manifest(false);
        manifest.post_install_checks.retain(|check| {
            !matches!(
                check.kind,
                PostInstallCheckKind::SystemGenerationMatchesDeclaration
            )
        });
        assert!(manifest
            .validation_errors()
            .contains(&HostRehydrationManifestError::MissingGenerationVerification));
    }

    #[test]
    fn restore_manifest_requires_data_integrity_check() {
        let mut manifest = manifest(true);
        manifest.post_install_checks.retain(|check| {
            !matches!(check.kind, PostInstallCheckKind::ProtectedDataIntegrity)
        });
        assert!(manifest.validation_errors().contains(
            &HostRehydrationManifestError::MissingProtectedDataIntegrityCheck
        ));
    }

    #[test]
    fn complete_live_clean_rehydration_is_verified() {
        let manifest = manifest(true);
        assert!(manifest.is_valid());
        assert_eq!(receipt(&manifest).evaluate(&manifest), RehydrationOutcome::Verified);
    }

    #[test]
    fn dry_run_can_never_verify_rehydration() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.execution_mode = RehydrationExecutionMode::DryRun;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn trusting_prior_runtime_can_never_verify_clean_rehydration() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.boot_origin = RecoveryBootOrigin::PriorRuntime;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn wrong_disk_identity_is_unproven_not_failure() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.observed_target.disk_identity_digest = "sha256:other-disk".to_string();
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn evidenced_destructive_preparation_failure_is_failed() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.preparation.result = PreparationResult::Failed;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Failed { .. }
        ));
    }

    #[test]
    fn failed_restore_with_complete_evidence_is_failed() {
        let manifest = manifest(true);
        let mut receipt = receipt(&manifest);
        receipt.protected_restore.result = RestoreResult::Failed;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Failed { .. }
        ));
    }

    #[test]
    fn failed_restore_without_evidence_is_unproven() {
        let manifest = manifest(true);
        let mut receipt = receipt(&manifest);
        receipt.protected_restore.result = RestoreResult::Failed;
        receipt.protected_restore.evidence = None;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn failed_post_install_check_with_evidence_is_failed() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.post_install_checks[0].passed = false;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Failed { .. }
        ));
    }

    #[test]
    fn missing_post_install_evidence_is_unproven() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.post_install_checks[0].evidence = None;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn missing_nixward_reconstruction_evidence_is_unproven() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.nixward_reconstruction_evidence = None;
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn unexpected_restore_is_unproven() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.protected_restore = ProtectedRestoreReceipt {
            result: RestoreResult::Restored,
            generation_ref: Some("unexpected".to_string()),
            evidence: Some(evidence("unexpected-restore")),
        };
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }

    #[test]
    fn duplicate_check_result_is_unproven() {
        let manifest = manifest(false);
        let mut receipt = receipt(&manifest);
        receipt.post_install_checks.push(receipt.post_install_checks[0].clone());
        assert!(matches!(
            receipt.evaluate(&manifest),
            RehydrationOutcome::Unproven { .. }
        ));
    }
}
