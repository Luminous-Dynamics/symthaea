use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const QUALIFICATION_RECEIPT_SCHEMA_VERSION: u16 = 1;
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
pub struct QualificationGateReceipt {
    pub gate_id: String,
    pub status: GateStatus,
    /// Human- or machine-readable evidence locator, run id, or command transcript identity.
    pub evidence: String,
}

impl QualificationGateReceipt {
    pub fn new(
        gate_id: impl Into<String>,
        status: GateStatus,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            gate_id: gate_id.into(),
            status,
            evidence: evidence.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationReceipt {
    pub schema_version: u16,
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
            if gate.evidence.trim().is_empty() && gate.status != GateStatus::Pending {
                errors.push(format!(
                    "non-pending gate {} must include evidence identity",
                    gate.gate_id
                ));
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

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
