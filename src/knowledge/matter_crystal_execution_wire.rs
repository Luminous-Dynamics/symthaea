// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned fail-closed wire envelope for external crystal-solver adapters.
//!
//! External workflow systems should not deserialize arbitrary JSON directly into
//! a trusted execution receipt and rely on downstream callers to remember a
//! separate validation step. This module makes the validated boundary itself
//! serializable: unknown fields/schema versions and invalid receipts fail during
//! deserialization.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use super::matter_crystal_execution::{
    CrystalExecutionError, CrystalSolverExecutionReceipt, CrystalSolverTask,
};

pub const CRYSTAL_EXECUTION_WIRE_SCHEMA_V1: &str =
    "symthaea.matter.crystal-solver-execution/v1";
const MAX_ID_LEN: usize = 256;
const MAX_TEXT_LEN: usize = 512;
const MAX_OUTPUT_ARTIFACTS: usize = 128;
const MAX_DEPENDENCY_SNAPSHOTS: usize = 128;

/// Stable wire spelling for a crystal execution task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrystalSolverTaskWireV1 {
    PeriodicElectronicStructure,
    StructuralRelaxation,
    ThermodynamicPhaseCompetition,
    LatticeDynamics,
}

impl From<CrystalSolverTask> for CrystalSolverTaskWireV1 {
    fn from(value: CrystalSolverTask) -> Self {
        match value {
            CrystalSolverTask::PeriodicElectronicStructure => Self::PeriodicElectronicStructure,
            CrystalSolverTask::StructuralRelaxation => Self::StructuralRelaxation,
            CrystalSolverTask::ThermodynamicPhaseCompetition => {
                Self::ThermodynamicPhaseCompetition
            }
            CrystalSolverTask::LatticeDynamics => Self::LatticeDynamics,
        }
    }
}

impl From<CrystalSolverTaskWireV1> for CrystalSolverTask {
    fn from(value: CrystalSolverTaskWireV1) -> Self {
        match value {
            CrystalSolverTaskWireV1::PeriodicElectronicStructure => {
                Self::PeriodicElectronicStructure
            }
            CrystalSolverTaskWireV1::StructuralRelaxation => Self::StructuralRelaxation,
            CrystalSolverTaskWireV1::ThermodynamicPhaseCompetition => {
                Self::ThermodynamicPhaseCompetition
            }
            CrystalSolverTaskWireV1::LatticeDynamics => Self::LatticeDynamics,
        }
    }
}

/// JSON-safe v1 execution envelope.
///
/// Fields are private so Rust callers cannot hand-construct an unchecked wire
/// object. Use `try_from_receipt`, or deserialize it: deserialization reconstructs
/// and validates the canonical domain receipt before returning success.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CrystalSolverExecutionWireV1 {
    schema_version: String,
    claim_id: String,
    task: CrystalSolverTaskWireV1,
    execution_evidence_id: String,
    declared_execution_yyyymmdd: u32,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    exit_code: i32,
    solver_name: String,
    solver_version: String,
    adapter_name: String,
    adapter_version: String,
    input_artifact_id: String,
    configuration_artifact_id: String,
    environment_snapshot_id: String,
    parser_snapshot_id: String,
    output_artifact_ids: Vec<String>,
    dependency_snapshot_ids: Vec<String>,
}

impl CrystalSolverExecutionWireV1 {
    pub fn try_from_receipt(
        receipt: CrystalSolverExecutionReceipt,
    ) -> Result<Self, CrystalExecutionWireError> {
        receipt
            .validate()
            .map_err(CrystalExecutionWireError::Receipt)?;
        let value = Self {
            schema_version: CRYSTAL_EXECUTION_WIRE_SCHEMA_V1.to_string(),
            claim_id: receipt.claim_id,
            task: receipt.task.into(),
            execution_evidence_id: receipt.execution_evidence_id,
            declared_execution_yyyymmdd: receipt.declared_execution_yyyymmdd,
            started_unix_ms: receipt.started_unix_ms,
            finished_unix_ms: receipt.finished_unix_ms,
            exit_code: receipt.exit_code,
            solver_name: receipt.solver_name,
            solver_version: receipt.solver_version,
            adapter_name: receipt.adapter_name,
            adapter_version: receipt.adapter_version,
            input_artifact_id: receipt.input_artifact_id,
            configuration_artifact_id: receipt.configuration_artifact_id,
            environment_snapshot_id: receipt.environment_snapshot_id,
            parser_snapshot_id: receipt.parser_snapshot_id,
            output_artifact_ids: receipt.output_artifact_ids,
            dependency_snapshot_ids: receipt.dependency_snapshot_ids,
        };
        value.validate_wire_shape()?;
        Ok(value)
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn task(&self) -> CrystalSolverTask {
        self.task.into()
    }

    pub fn receipt(&self) -> Result<CrystalSolverExecutionReceipt, CrystalExecutionWireError> {
        let receipt = CrystalSolverExecutionReceipt {
            claim_id: self.claim_id.clone(),
            task: self.task.into(),
            execution_evidence_id: self.execution_evidence_id.clone(),
            declared_execution_yyyymmdd: self.declared_execution_yyyymmdd,
            started_unix_ms: self.started_unix_ms,
            finished_unix_ms: self.finished_unix_ms,
            exit_code: self.exit_code,
            solver_name: self.solver_name.clone(),
            solver_version: self.solver_version.clone(),
            adapter_name: self.adapter_name.clone(),
            adapter_version: self.adapter_version.clone(),
            input_artifact_id: self.input_artifact_id.clone(),
            configuration_artifact_id: self.configuration_artifact_id.clone(),
            environment_snapshot_id: self.environment_snapshot_id.clone(),
            parser_snapshot_id: self.parser_snapshot_id.clone(),
            output_artifact_ids: self.output_artifact_ids.clone(),
            dependency_snapshot_ids: self.dependency_snapshot_ids.clone(),
        };
        receipt
            .validate()
            .map_err(CrystalExecutionWireError::Receipt)?;
        Ok(receipt)
    }

    pub fn into_receipt(self) -> Result<CrystalSolverExecutionReceipt, CrystalExecutionWireError> {
        self.receipt()
    }

    fn validate_wire_shape(&self) -> Result<(), CrystalExecutionWireError> {
        if self.schema_version != CRYSTAL_EXECUTION_WIRE_SCHEMA_V1 {
            return Err(CrystalExecutionWireError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        for (field, value, max_len) in [
            ("claim_id", self.claim_id.as_str(), MAX_ID_LEN),
            (
                "execution_evidence_id",
                self.execution_evidence_id.as_str(),
                MAX_ID_LEN,
            ),
            ("solver_name", self.solver_name.as_str(), MAX_TEXT_LEN),
            ("solver_version", self.solver_version.as_str(), MAX_TEXT_LEN),
            ("adapter_name", self.adapter_name.as_str(), MAX_TEXT_LEN),
            ("adapter_version", self.adapter_version.as_str(), MAX_TEXT_LEN),
            ("input_artifact_id", self.input_artifact_id.as_str(), MAX_ID_LEN),
            (
                "configuration_artifact_id",
                self.configuration_artifact_id.as_str(),
                MAX_ID_LEN,
            ),
            (
                "environment_snapshot_id",
                self.environment_snapshot_id.as_str(),
                MAX_ID_LEN,
            ),
            ("parser_snapshot_id", self.parser_snapshot_id.as_str(), MAX_ID_LEN),
        ] {
            validate_text(field, value, max_len)?;
        }
        if self.output_artifact_ids.len() > MAX_OUTPUT_ARTIFACTS {
            return Err(CrystalExecutionWireError::TooManyItems {
                field: "output_artifact_ids",
                max: MAX_OUTPUT_ARTIFACTS,
            });
        }
        if self.dependency_snapshot_ids.len() > MAX_DEPENDENCY_SNAPSHOTS {
            return Err(CrystalExecutionWireError::TooManyItems {
                field: "dependency_snapshot_ids",
                max: MAX_DEPENDENCY_SNAPSHOTS,
            });
        }
        for id in &self.output_artifact_ids {
            validate_text("output_artifact_ids[]", id, MAX_ID_LEN)?;
        }
        for id in &self.dependency_snapshot_ids {
            validate_text("dependency_snapshot_ids[]", id, MAX_ID_LEN)?;
        }
        self.receipt()?;
        Ok(())
    }
}

fn validate_text(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), CrystalExecutionWireError> {
    if value.trim().is_empty() {
        return Err(CrystalExecutionWireError::EmptyField(field));
    }
    if value.len() > max_len {
        return Err(CrystalExecutionWireError::FieldTooLong { field, max_len });
    }
    if value.chars().any(char::is_control) {
        return Err(CrystalExecutionWireError::ControlCharacter(field));
    }
    Ok(())
}

impl<'de> Deserialize<'de> for CrystalSolverExecutionWireV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawWire {
            schema_version: String,
            claim_id: String,
            task: CrystalSolverTaskWireV1,
            execution_evidence_id: String,
            declared_execution_yyyymmdd: u32,
            started_unix_ms: u64,
            finished_unix_ms: u64,
            exit_code: i32,
            solver_name: String,
            solver_version: String,
            adapter_name: String,
            adapter_version: String,
            input_artifact_id: String,
            configuration_artifact_id: String,
            environment_snapshot_id: String,
            parser_snapshot_id: String,
            output_artifact_ids: Vec<String>,
            dependency_snapshot_ids: Vec<String>,
        }

        let raw = RawWire::deserialize(deserializer)?;
        let wire = Self {
            schema_version: raw.schema_version,
            claim_id: raw.claim_id,
            task: raw.task,
            execution_evidence_id: raw.execution_evidence_id,
            declared_execution_yyyymmdd: raw.declared_execution_yyyymmdd,
            started_unix_ms: raw.started_unix_ms,
            finished_unix_ms: raw.finished_unix_ms,
            exit_code: raw.exit_code,
            solver_name: raw.solver_name,
            solver_version: raw.solver_version,
            adapter_name: raw.adapter_name,
            adapter_version: raw.adapter_version,
            input_artifact_id: raw.input_artifact_id,
            configuration_artifact_id: raw.configuration_artifact_id,
            environment_snapshot_id: raw.environment_snapshot_id,
            parser_snapshot_id: raw.parser_snapshot_id,
            output_artifact_ids: raw.output_artifact_ids,
            dependency_snapshot_ids: raw.dependency_snapshot_ids,
        };
        wire.validate_wire_shape()
            .map_err(serde::de::Error::custom)?;
        Ok(wire)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrystalExecutionWireError {
    Receipt(CrystalExecutionError),
    UnsupportedSchema(String),
    EmptyField(&'static str),
    FieldTooLong { field: &'static str, max_len: usize },
    ControlCharacter(&'static str),
    TooManyItems { field: &'static str, max: usize },
}

impl fmt::Display for CrystalExecutionWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Receipt(error) => write!(f, "crystal execution receipt rejected: {error}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported crystal execution wire schema `{schema}`")
            }
            Self::EmptyField(field) => write!(f, "wire field `{field}` is empty"),
            Self::FieldTooLong { field, max_len } => {
                write!(f, "wire field `{field}` exceeds {max_len} bytes")
            }
            Self::ControlCharacter(field) => {
                write!(f, "wire field `{field}` contains a control character")
            }
            Self::TooManyItems { field, max } => {
                write!(f, "wire field `{field}` exceeds {max} entries")
            }
        }
    }
}

impl std::error::Error for CrystalExecutionWireError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt() -> CrystalSolverExecutionReceipt {
        CrystalSolverExecutionReceipt {
            claim_id: "claim:relax".into(),
            task: CrystalSolverTask::StructuralRelaxation,
            execution_evidence_id: "execution:relax".into(),
            declared_execution_yyyymmdd: 20260907,
            started_unix_ms: 1_000,
            finished_unix_ms: 2_000,
            exit_code: 0,
            solver_name: "external-solver".into(),
            solver_version: "1.2.3".into(),
            adapter_name: "external-adapter".into(),
            adapter_version: "0.1".into(),
            input_artifact_id: "input".into(),
            configuration_artifact_id: "config".into(),
            environment_snapshot_id: "environment".into(),
            parser_snapshot_id: "parser".into(),
            output_artifact_ids: vec!["output".into()],
            dependency_snapshot_ids: vec!["dependency".into()],
        }
    }

    #[test]
    fn valid_wire_round_trip_revalidates_domain_receipt() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let json = serde_json::to_string(&wire).unwrap();
        let decoded: CrystalSolverExecutionWireV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, wire);
        assert_eq!(decoded.receipt().unwrap(), receipt());
        assert_eq!(decoded.schema_version(), CRYSTAL_EXECUTION_WIRE_SCHEMA_V1);
    }

    #[test]
    fn unknown_schema_fails_during_deserialization() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let mut json = serde_json::to_value(wire).unwrap();
        json["schema_version"] = serde_json::Value::String("future/v99".into());
        assert!(serde_json::from_value::<CrystalSolverExecutionWireV1>(json).is_err());
    }

    #[test]
    fn unknown_field_fails_during_deserialization() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let mut json = serde_json::to_value(wire).unwrap();
        json["surprise"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CrystalSolverExecutionWireV1>(json).is_err());
    }

    #[test]
    fn invalid_domain_receipt_fails_during_deserialization() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let mut json = serde_json::to_value(wire).unwrap();
        json["exit_code"] = serde_json::json!(3);
        assert!(serde_json::from_value::<CrystalSolverExecutionWireV1>(json).is_err());
    }

    #[test]
    fn control_characters_fail_closed() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let mut json = serde_json::to_value(wire).unwrap();
        json["adapter_name"] = serde_json::Value::String("bad\nadapter".into());
        assert!(serde_json::from_value::<CrystalSolverExecutionWireV1>(json).is_err());
    }

    #[test]
    fn oversized_artifact_lists_fail_closed() {
        let wire = CrystalSolverExecutionWireV1::try_from_receipt(receipt()).unwrap();
        let mut json = serde_json::to_value(wire).unwrap();
        json["output_artifact_ids"] = serde_json::Value::Array(
            (0..=MAX_OUTPUT_ARTIFACTS)
                .map(|index| serde_json::Value::String(format!("output-{index}")))
                .collect(),
        );
        assert!(serde_json::from_value::<CrystalSolverExecutionWireV1>(json).is_err());
    }
}
