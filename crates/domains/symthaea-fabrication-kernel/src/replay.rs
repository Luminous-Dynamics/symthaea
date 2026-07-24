// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit deterministic replay contracts for fabrication evidence.

use crate::audit::{AuditJournal, digest_audit_journal};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::provenance::{FabricationManifest, digest_fabrication_manifest};
use crate::trust::{TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEnvironment {
    pub kernel_version: String,
    pub source_revision: String,
    pub target_triple: String,
    pub rustc_version: String,
    pub cargo_lock_digest: Option<Sha256Digest>,
    pub feature_flags: Vec<String>,
}

impl ReplayEnvironment {
    pub fn canonicalize(&mut self) {
        self.feature_flags.sort();
        self.feature_flags.dedup();
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        for (name, value) in [
            ("kernel_version", self.kernel_version.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("target_triple", self.target_triple.as_str()),
            ("rustc_version", self.rustc_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(name);
            }
        }
        if self
            .feature_flags
            .iter()
            .any(|feature| feature.trim().is_empty())
        {
            return Err("feature_flags");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlgorithmVersion {
    pub component: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FabricationReplayContract {
    pub schema_version: String,
    pub manifest_digest: Sha256Digest,
    pub environment: ReplayEnvironment,
    pub deterministic_seed: u64,
    pub algorithms: Vec<AlgorithmVersion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayContractError {
    InvalidEnvironment(&'static str),
    ManifestEncoding(String),
    ContractEncoding(String),
    TrustSnapshot(String),
    AuditJournal(String),
    EmptyAuditJournal,
}

pub fn build_replay_contract(
    manifest: &FabricationManifest,
    mut environment: ReplayEnvironment,
    deterministic_seed: u64,
) -> Result<FabricationReplayContract, ReplayContractError> {
    environment.canonicalize();
    environment
        .validate()
        .map_err(ReplayContractError::InvalidEnvironment)?;
    let manifest_digest = digest_fabrication_manifest(manifest)
        .map_err(|error| ReplayContractError::ManifestEncoding(error.to_string()))?;
    Ok(FabricationReplayContract {
        schema_version: "symthaea.fabrication.replay-contract.v1".into(),
        manifest_digest,
        environment,
        deterministic_seed,
        algorithms: algorithm_inventory(),
    })
}

pub fn digest_replay_contract(
    contract: &FabricationReplayContract,
) -> Result<Sha256Digest, ReplayContractError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| ReplayContractError::ContractEncoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.replay-contract-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMismatch {
    SchemaVersion,
    ManifestDigest,
    Environment,
    AlgorithmInventory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayVerificationReport {
    pub mismatches: Vec<ReplayMismatch>,
}

impl ReplayVerificationReport {
    pub fn reproducible(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn verify_replay_contract(
    contract: &FabricationReplayContract,
    manifest: &FabricationManifest,
    mut environment: ReplayEnvironment,
) -> Result<ReplayVerificationReport, ReplayContractError> {
    environment.canonicalize();
    environment
        .validate()
        .map_err(ReplayContractError::InvalidEnvironment)?;
    let manifest_digest = digest_fabrication_manifest(manifest)
        .map_err(|error| ReplayContractError::ManifestEncoding(error.to_string()))?;
    let mut mismatches = Vec::new();
    if contract.schema_version != "symthaea.fabrication.replay-contract.v1" {
        mismatches.push(ReplayMismatch::SchemaVersion);
    }
    if contract.manifest_digest != manifest_digest {
        mismatches.push(ReplayMismatch::ManifestDigest);
    }
    if contract.environment != environment {
        mismatches.push(ReplayMismatch::Environment);
    }
    if contract.algorithms != algorithm_inventory() {
        mismatches.push(ReplayMismatch::AlgorithmInventory);
    }
    Ok(ReplayVerificationReport { mismatches })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernedFabricationReplayContract {
    pub schema_version: String,
    pub base: FabricationReplayContract,
    pub trust_snapshot_digest: Sha256Digest,
    pub audit_journal_digest: Sha256Digest,
    pub audit_head: Sha256Digest,
}

pub fn build_governed_replay_contract(
    manifest: &FabricationManifest,
    environment: ReplayEnvironment,
    deterministic_seed: u64,
    trust_snapshot: &TrustSnapshot,
    audit_journal: &AuditJournal,
) -> Result<GovernedFabricationReplayContract, ReplayContractError> {
    let base = build_replay_contract(manifest, environment, deterministic_seed)?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| ReplayContractError::TrustSnapshot(format!("{error:?}")))?;
    let audit_journal_digest = digest_audit_journal(audit_journal)
        .map_err(|error| ReplayContractError::AuditJournal(format!("{error:?}")))?;
    let audit_head = audit_journal
        .head()
        .ok_or(ReplayContractError::EmptyAuditJournal)?;
    Ok(GovernedFabricationReplayContract {
        schema_version: "symthaea.fabrication.governed-replay-contract.v1".into(),
        base,
        trust_snapshot_digest,
        audit_journal_digest,
        audit_head,
    })
}

pub fn digest_governed_replay_contract(
    contract: &GovernedFabricationReplayContract,
) -> Result<Sha256Digest, ReplayContractError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| ReplayContractError::ContractEncoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.governed-replay-contract-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GovernedReplayMismatch {
    SchemaVersion,
    Base(ReplayMismatch),
    TrustSnapshotDigest,
    AuditJournalDigest,
    AuditHead,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GovernedReplayVerificationReport {
    pub mismatches: Vec<GovernedReplayMismatch>,
}

impl GovernedReplayVerificationReport {
    pub fn reproducible(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn verify_governed_replay_contract(
    contract: &GovernedFabricationReplayContract,
    manifest: &FabricationManifest,
    environment: ReplayEnvironment,
    trust_snapshot: &TrustSnapshot,
    audit_journal: &AuditJournal,
) -> Result<GovernedReplayVerificationReport, ReplayContractError> {
    let mut mismatches = Vec::new();
    if contract.schema_version != "symthaea.fabrication.governed-replay-contract.v1" {
        mismatches.push(GovernedReplayMismatch::SchemaVersion);
    }
    let base = verify_replay_contract(&contract.base, manifest, environment)?;
    mismatches.extend(
        base.mismatches
            .into_iter()
            .map(GovernedReplayMismatch::Base),
    );
    let trust_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| ReplayContractError::TrustSnapshot(format!("{error:?}")))?;
    if contract.trust_snapshot_digest != trust_digest {
        mismatches.push(GovernedReplayMismatch::TrustSnapshotDigest);
    }
    let audit_digest = digest_audit_journal(audit_journal)
        .map_err(|error| ReplayContractError::AuditJournal(format!("{error:?}")))?;
    if contract.audit_journal_digest != audit_digest {
        mismatches.push(GovernedReplayMismatch::AuditJournalDigest);
    }
    let audit_head = audit_journal
        .head()
        .ok_or(ReplayContractError::EmptyAuditJournal)?;
    if contract.audit_head != audit_head {
        mismatches.push(GovernedReplayMismatch::AuditHead);
    }
    Ok(GovernedReplayVerificationReport { mismatches })
}

fn algorithm_inventory() -> Vec<AlgorithmVersion> {
    vec![
        AlgorithmVersion {
            component: "fabrication-manifest".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "geometry-fingerprint".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "machine-profile-fingerprint".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "sha256".into(),
            version: "fips-180-4".into(),
        },
        AlgorithmVersion {
            component: "stored-3mf-opc".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "trust-snapshot".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "audit-journal".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "execution-guard-checkpoint".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "governed-release-package".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "release-policy".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "manifest-scoped-delegation".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "timed-machine-session".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "machine-session-tracker".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "audit-segment".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "signed-audit-anchor".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "containment-fault-matrix".into(),
            version: "v1".into(),
        },
        AlgorithmVersion {
            component: "operational-replay-contract".into(),
            version: "v1".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{FabricationManifest, StableFingerprint};

    fn manifest() -> FabricationManifest {
        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        }
    }

    fn environment() -> ReplayEnvironment {
        ReplayEnvironment {
            kernel_version: "0.11.0".into(),
            source_revision: "abc123".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            rustc_version: "rustc-test".into(),
            cargo_lock_digest: None,
            feature_flags: vec!["analytical".into(), "analytical".into()],
        }
    }

    #[test]
    fn feature_order_and_duplicates_are_canonicalized() {
        let contract = build_replay_contract(&manifest(), environment(), 7).unwrap();
        assert_eq!(contract.environment.feature_flags, vec!["analytical"]);
    }

    #[test]
    fn environment_drift_is_reported() {
        let contract = build_replay_contract(&manifest(), environment(), 7).unwrap();
        let mut changed = environment();
        changed.rustc_version = "different".into();
        let report = verify_replay_contract(&contract, &manifest(), changed).unwrap();
        assert_eq!(report.mismatches, vec![ReplayMismatch::Environment]);
    }

    #[test]
    fn contract_digest_is_deterministic() {
        let contract = build_replay_contract(&manifest(), environment(), 7).unwrap();
        assert_eq!(
            digest_replay_contract(&contract).unwrap(),
            digest_replay_contract(&contract).unwrap()
        );
    }

    #[test]
    fn governed_contract_detects_trust_and_audit_drift() {
        use crate::attestation::SignatureAlgorithm;
        use crate::audit::{AuditAction, AuditJournal};
        use crate::crypto_digest::sha256;
        use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
        use std::collections::BTreeSet;

        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "release".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        let mut audit = AuditJournal::default();
        audit
            .append(
                500,
                "operator",
                AuditAction::JobAuthorized,
                sha256(b"job"),
                None,
            )
            .unwrap();
        let contract =
            build_governed_replay_contract(&manifest(), environment(), 7, &trust, &audit).unwrap();
        assert!(
            verify_governed_replay_contract(&contract, &manifest(), environment(), &trust, &audit,)
                .unwrap()
                .reproducible()
        );

        let mut changed_audit = audit.clone();
        changed_audit
            .append(
                501,
                "operator",
                AuditAction::JobSubmitted,
                sha256(b"job"),
                None,
            )
            .unwrap();
        let report = verify_governed_replay_contract(
            &contract,
            &manifest(),
            environment(),
            &trust,
            &changed_audit,
        )
        .unwrap();
        assert!(
            report
                .mismatches
                .contains(&GovernedReplayMismatch::AuditJournalDigest)
        );
        assert!(
            report
                .mismatches
                .contains(&GovernedReplayMismatch::AuditHead)
        );
    }
}
