use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::snapshot::INTEROCEPTIVE_MODEL_SEMANTICS_VERSION;

pub const QUALIFICATION_RECEIPT_SCHEMA_VERSION: u16 = 2;
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

/// Typed identity for the evidence underlying one qualification-gate status.
///
/// This establishes provenance consistency inside the receipt. It does not by
/// itself authenticate an external service or prove that a reported status is
/// truthful; the qualification harness must verify the referenced transcript or
/// workflow run before constructing evidence-bearing `Passed`/`Failed` receipts.
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

    pub fn subject_commit(&self) -> &str {
        match self {
            Self::LocalCommand { subject_commit, .. }
            | Self::GitHubActions { subject_commit, .. } => subject_commit,
        }
    }

    pub fn validation_errors(&self) -> Vec<String> {
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
                if !is_lower_hex(environment_sha256, 64) {
                    errors.push(
                        "local-command environment_sha256 must be a 64-character lowercase SHA-256 digest"
                            .into(),
                    );
                }
                if !is_lower_hex(transcript_sha256, 64) {
                    errors.push(
                        "local-command transcript_sha256 must be a 64-character lowercase SHA-256 digest"
                            .into(),
                    );
                }
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

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
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

    pub fn pending(gate_id: impl Into<String>) -> Self {
        Self {
            gate_id: gate_id.into(),
            status: GateStatus::Pending,
            evidence: None,
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
                if let Err(evidence_errors) = evidence.validate() {
                    errors.extend(
                        evidence_errors
                            .into_iter()
                            .map(|error| format!("gate {} evidence: {error}", gate.gate_id)),
                    );
                }
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
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// True only when the receipt is structurally valid and every fixed v0.1
    /// required gate explicitly passed. `Skipped` never counts as `Passed`.
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

    pub fn blocking_required_gates(&self) -> Vec<&QualificationGateReceipt> {
        self.gates
            .iter()
            .filter(|gate| {
                REQUIRED_QUALIFICATION_GATES.contains(&gate.gate_id.as_str())
                    && gate.status != GateStatus::Passed
            })
            .collect()
    }
}

fn evidence_matches_gate_contract(gate_id: &str, evidence: &QualificationGateEvidence) -> bool {
    match (gate_id, evidence) {
        (
            "local_fmt",
            QualificationGateEvidence::LocalCommand { command, .. },
        ) => command == "cargo fmt --all --check",
        (
            "local_test",
            QualificationGateEvidence::LocalCommand { command, .. },
        ) => command == "cargo test -p symthaea-interoception",
        (
            "local_clippy",
            QualificationGateEvidence::LocalCommand { command, .. },
        ) => command
            == "cargo clippy -p symthaea-interoception --all-targets -- -D warnings",
        (
            "workspace_ci",
            QualificationGateEvidence::GitHubActions { workflow, .. },
        ) => workflow == "CI",
        (
            "showroom_integrity",
            QualificationGateEvidence::GitHubActions { workflow, .. },
        ) => workflow == "Showroom Integrity",
        (
            "benchmark_suite",
            QualificationGateEvidence::GitHubActions { workflow, .. },
        ) => workflow == "Symthaea Benchmark Suite",
        ("local_fmt" | "local_test" | "local_clippy", _) => false,
        ("workspace_ci" | "showroom_integrity" | "benchmark_suite", _) => false,
        _ => true,
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
