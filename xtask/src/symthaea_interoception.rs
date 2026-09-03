//! Frozen schema mirror for Native Interoception v0.1 qualification artifacts.
//!
//! The promotion verifier is intentionally external to the native runtime crate.  It
//! mirrors only the immutable v0.1 evidence schema needed to independently parse and
//! validate a promotion bundle.  Any future native schema change requires an explicit
//! verifier-policy/schema update instead of silently changing verifier behavior through
//! a path dependency.

use std::{collections::{BTreeMap, BTreeSet}, fmt::Write as _};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const INTEROCEPTIVE_MODEL_SEMANTICS_VERSION: u16 = 1;
pub const INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION: u16 = 2;
pub const QUALIFICATION_RECEIPT_SCHEMA_VERSION: u16 = 2;
pub const EVIDENCE_CAPSULE_SCHEMA_VERSION: u16 = 2;
pub const QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION: u16 = 2;

pub const REQUIRED_QUALIFICATION_GATES: [&str; 5] = [
    "local_fmt",
    "local_test",
    "local_clippy",
    "workspace_ci",
    "showroom_integrity",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    Passed,
    Failed,
    Skipped,
    Pending,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationGateEvidence {
    LocalCommand {
        subject_commit: String,
        command: String,
        environment_sha256: String,
        transcript_sha256: String,
    },
    GitHubActions {
        subject_commit: String,
        workflow: String,
        run_id: u64,
        run_attempt: u32,
    },
}

impl QualificationGateEvidence {
    pub fn local_command(
        subject_commit: impl Into<String>,
        command: impl Into<String>,
        environment_sha256: impl Into<String>,
        transcript_sha256: impl Into<String>,
    ) -> Self {
        Self::LocalCommand {
            subject_commit: subject_commit.into(),
            command: command.into(),
            environment_sha256: environment_sha256.into(),
            transcript_sha256: transcript_sha256.into(),
        }
    }

    pub fn github_actions(
        subject_commit: impl Into<String>,
        workflow: impl Into<String>,
        run_id: u64,
        run_attempt: u32,
    ) -> Self {
        Self::GitHubActions {
            subject_commit: subject_commit.into(),
            workflow: workflow.into(),
            run_id,
            run_attempt,
        }
    }

    fn subject_commit(&self) -> &str {
        match self {
            Self::LocalCommand { subject_commit, .. }
            | Self::GitHubActions { subject_commit, .. } => subject_commit,
        }
    }

    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !is_lower_hex(self.subject_commit(), 40) {
            errors.push(
                "gate evidence subject_commit must be a 40-character lowercase Git SHA-1".into(),
            );
        }
        match self {
            Self::LocalCommand {
                command,
                environment_sha256,
                transcript_sha256,
                ..
            } => {
                if command.trim().is_empty() {
                    errors.push("local-command evidence must include the executed command".into());
                }
                validate_sha256("local-command environment_sha256", environment_sha256, &mut errors);
                validate_sha256("local-command transcript_sha256", transcript_sha256, &mut errors);
            }
            Self::GitHubActions {
                workflow,
                run_id,
                run_attempt,
                ..
            } => {
                if workflow.trim().is_empty() {
                    errors.push("GitHub Actions evidence must include a workflow identity".into());
                }
                if *run_id == 0 {
                    errors.push("GitHub Actions run_id must be positive".into());
                }
                if *run_attempt == 0 {
                    errors.push("GitHub Actions run_attempt must be positive".into());
                }
            }
        }
        errors
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationGateReceipt {
    pub gate_id: String,
    pub status: GateStatus,
    pub evidence: Option<QualificationGateEvidence>,
}

impl QualificationGateReceipt {
    pub fn with_evidence(
        gate_id: impl Into<String>,
        status: GateStatus,
        evidence: QualificationGateEvidence,
    ) -> Self {
        Self {
            gate_id: gate_id.into(),
            status,
            evidence: Some(evidence),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationReceipt {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub source_commit: String,
    pub gates: Vec<QualificationGateReceipt>,
}

impl QualificationReceipt {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != QUALIFICATION_RECEIPT_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported qualification receipt schema version: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if !is_lower_hex(&self.source_commit, 40) {
            errors.push("source_commit must be a 40-character lowercase Git SHA-1".into());
        }

        let mut seen = BTreeSet::new();
        for gate in &self.gates {
            if gate.gate_id.trim().is_empty() {
                errors.push("gate_id must not be empty".into());
                continue;
            }
            if !seen.insert(gate.gate_id.as_str()) {
                errors.push(format!("duplicate qualification gate: {}", gate.gate_id));
            }
            if gate.status != GateStatus::Pending && gate.evidence.is_none() {
                errors.push(format!(
                    "non-pending gate {} must include typed evidence identity",
                    gate.gate_id
                ));
            }
            if let Some(evidence) = &gate.evidence {
                errors.extend(
                    evidence
                        .validation_errors()
                        .into_iter()
                        .map(|error| format!("gate {} evidence: {error}", gate.gate_id)),
                );
                if evidence.subject_commit() != self.source_commit {
                    errors.push(format!(
                        "gate {} evidence subject commit does not match qualification source_commit",
                        gate.gate_id
                    ));
                }
                if !evidence_matches_gate_contract(&gate.gate_id, evidence) {
                    errors.push(format!(
                        "gate {} uses an incompatible evidence kind or identity",
                        gate.gate_id
                    ));
                }
            }
        }

        for required in REQUIRED_QUALIFICATION_GATES {
            if !seen.contains(required) {
                errors.push(format!("missing required qualification gate: {required}"));
            }
        }
        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    pub fn is_qualified(&self) -> bool {
        if self.validate().is_err() {
            return false;
        }
        let statuses: BTreeMap<&str, GateStatus> = self
            .gates
            .iter()
            .map(|gate| (gate.gate_id.as_str(), gate.status))
            .collect();
        REQUIRED_QUALIFICATION_GATES
            .iter()
            .all(|gate| statuses.get(gate).copied() == Some(GateStatus::Passed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastBasisId {
    Kinematic,
    DynamicsAwareConstantDrive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDigest {
    pub name: String,
    pub sha256: String,
}

impl ArtifactDigest {
    pub fn new(name: impl Into<String>, sha256: impl Into<String>) -> Self {
        Self { name: name.into(), sha256: sha256.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCapsuleManifest {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub source_commit: String,
    pub cargo_lock_sha256: String,
    pub flake_lock_sha256: Option<String>,
    pub rust_toolchain_sha256: Option<String>,
    pub rustc_vv: String,
    pub cargo_vv: String,
    pub target_triple: String,
    pub architecture: String,
    pub experiment_id: String,
    pub preregistration_sha256: String,
    pub forecast_basis: ForecastBasisId,
    pub experiment_config_sha256: String,
    pub input_sequence_sha256: String,
    pub snapshot_schema_version: u16,
    pub evidence_plane_sha256: String,
    pub artifacts: Vec<ArtifactDigest>,
}

impl EvidenceCapsuleManifest {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != EVIDENCE_CAPSULE_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported evidence capsule schema version: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if self.snapshot_schema_version != INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "snapshot schema version mismatch: {}",
                self.snapshot_schema_version
            ));
        }
        if !is_lower_hex(&self.source_commit, 40) {
            errors.push("source_commit must be a 40-character lowercase Git SHA-1".into());
        }
        validate_sha256("cargo_lock_sha256", &self.cargo_lock_sha256, &mut errors);
        if let Some(value) = &self.flake_lock_sha256 {
            validate_sha256("flake_lock_sha256", value, &mut errors);
        }
        if let Some(value) = &self.rust_toolchain_sha256 {
            validate_sha256("rust_toolchain_sha256", value, &mut errors);
        }
        validate_sha256("preregistration_sha256", &self.preregistration_sha256, &mut errors);
        validate_sha256("experiment_config_sha256", &self.experiment_config_sha256, &mut errors);
        validate_sha256("input_sequence_sha256", &self.input_sequence_sha256, &mut errors);
        validate_sha256("evidence_plane_sha256", &self.evidence_plane_sha256, &mut errors);

        for (name, value) in [
            ("rustc_vv", self.rustc_vv.as_str()),
            ("cargo_vv", self.cargo_vv.as_str()),
            ("target_triple", self.target_triple.as_str()),
            ("architecture", self.architecture.as_str()),
            ("experiment_id", self.experiment_id.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("{name} must not be empty"));
            }
        }

        if self.artifacts.is_empty() {
            errors.push("at least one raw result artifact digest is required".into());
        }
        let mut names = BTreeSet::new();
        for artifact in &self.artifacts {
            if artifact.name.trim().is_empty() {
                errors.push("artifact names must not be empty".into());
            } else if !names.insert(artifact.name.as_str()) {
                errors.push(format!("duplicate artifact name: {}", artifact.name));
            }
            validate_sha256(
                &format!("artifact[{}].sha256", artifact.name),
                &artifact.sha256,
                &mut errors,
            );
        }
        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationEvidenceBundle {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub source_commit: String,
    pub qualification: QualificationReceipt,
    pub evidence: EvidenceCapsuleManifest,
}

impl QualificationEvidenceBundle {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported qualification evidence bundle schema version: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "qualification evidence bundle model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if let Err(inner) = self.qualification.validate() {
            errors.extend(inner.into_iter().map(|e| format!("qualification receipt: {e}")));
        }
        if let Err(inner) = self.evidence.validate() {
            errors.extend(inner.into_iter().map(|e| format!("evidence capsule: {e}")));
        }
        if self.source_commit != self.qualification.source_commit {
            errors.push("bundle source_commit does not match qualification receipt".into());
        }
        if self.source_commit != self.evidence.source_commit {
            errors.push("bundle source_commit does not match evidence capsule".into());
        }
        if self.qualification.source_commit != self.evidence.source_commit {
            errors.push("qualification receipt and evidence capsule source commits differ".into());
        }
        if self.model_semantics_version != self.qualification.model_semantics_version {
            errors.push("bundle model semantics version does not match qualification receipt".into());
        }
        if self.model_semantics_version != self.evidence.model_semantics_version {
            errors.push("bundle model semantics version does not match evidence capsule".into());
        }
        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    pub fn is_qualified(&self) -> bool {
        self.validate().is_ok() && self.qualification.is_qualified()
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, Vec<String>> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| vec![format!("failed to serialize qualification evidence bundle: {error}")])
    }

    pub fn sha256(&self) -> Result<String, Vec<String>> {
        let bytes = self.canonical_json()?;
        let digest = Sha256::digest(&bytes);
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(encoded)
    }
}

fn evidence_matches_gate_contract(gate_id: &str, evidence: &QualificationGateEvidence) -> bool {
    match (gate_id, evidence) {
        ("local_fmt", QualificationGateEvidence::LocalCommand { command, .. }) => {
            command == "cargo fmt --all --check"
        }
        ("local_test", QualificationGateEvidence::LocalCommand { command, .. }) => {
            command == "cargo test -p symthaea-interoception"
        }
        ("local_clippy", QualificationGateEvidence::LocalCommand { command, .. }) => {
            command == "cargo clippy -p symthaea-interoception --all-targets -- -D warnings"
        }
        ("workspace_ci", QualificationGateEvidence::GitHubActions { workflow, .. }) => workflow == "CI",
        ("showroom_integrity", QualificationGateEvidence::GitHubActions { workflow, .. }) => {
            workflow == "Showroom Integrity"
        }
        ("benchmark_suite", QualificationGateEvidence::GitHubActions { workflow, .. }) => {
            workflow == "Symthaea Benchmark Suite"
        }
        ("local_fmt" | "local_test" | "local_clippy", _) => false,
        ("workspace_ci" | "showroom_integrity" | "benchmark_suite", _) => false,
        _ => true,
    }
}

fn validate_sha256(name: &str, value: &str, errors: &mut Vec<String>) {
    if !is_lower_hex(value, 64) {
        errors.push(format!("{name} must be a 64-character lowercase SHA-256 digest"));
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
