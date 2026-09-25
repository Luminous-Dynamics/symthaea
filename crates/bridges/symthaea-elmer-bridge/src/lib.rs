// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Closure-bound Elmer FEM/multiphysics adapter boundary.
//!
//! ENG-FEM-001A deliberately stops before real process execution. It defines a
//! case/output contract whose inputs are bound by `symthaea-solver-closure` and
//! provides a strict parser for an adapter-owned Elmer `SaveScalars` output.
//!
//! Real Elmer execution remains fail-closed until ENG-EXEC-001 gives the shared
//! `CommandSolver` an explicit working-directory/environment policy. This crate
//! must not grow a private subprocess runner merely to bypass that dependency.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Component, Path};
use symthaea_sim_bridge::{
    EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest, SimulationResult,
    SolverKind,
};
use symthaea_solver_closure::{
    ClosureError, ContentDigest, SolverInputClosure, SolverInputRole,
};
use thiserror::Error;

pub const ELMER_CASE_PROFILE_ID: &str = "symthaea-elmer-case-v1";
pub const ELMER_SCALAR_SCHEMA_ID: &str = "symthaea-elmer-save-scalars-schema-v1";

/// One scalar column emitted by the adapter-owned `SaveScalars` result file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElmerScalarColumn {
    pub name: String,
    pub unit: String,
}

/// Exact parser contract for one Elmer scalar result file.
///
/// The schema itself is an engineering input: changing metric order, units, or
/// output filename changes its content digest and therefore must change the
/// bound solver closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElmerScalarSchema {
    pub schema_id: String,
    pub output_filename: String,
    pub columns: Vec<ElmerScalarColumn>,
}

impl ElmerScalarSchema {
    pub fn new(output_filename: impl Into<String>, columns: Vec<ElmerScalarColumn>) -> Self {
        Self {
            schema_id: ELMER_SCALAR_SCHEMA_ID.into(),
            output_filename: output_filename.into(),
            columns,
        }
    }

    pub fn validate(&self) -> Result<(), ElmerBridgeError> {
        if self.schema_id != ELMER_SCALAR_SCHEMA_ID {
            return Err(ElmerBridgeError::UnsupportedScalarSchema(
                self.schema_id.clone(),
            ));
        }
        validate_local_filename(&self.output_filename)?;
        if self.columns.is_empty() {
            return Err(ElmerBridgeError::NoScalarColumns);
        }
        let mut names = BTreeSet::new();
        for column in &self.columns {
            if column.name.trim().is_empty() || column.unit.trim().is_empty() {
                return Err(ElmerBridgeError::InvalidScalarColumn);
            }
            if column.name.trim() != column.name || column.unit.trim() != column.unit {
                return Err(ElmerBridgeError::NonCanonicalScalarColumn {
                    name: column.name.clone(),
                    unit: column.unit.clone(),
                });
            }
            if !names.insert(column.name.as_str()) {
                return Err(ElmerBridgeError::DuplicateScalarColumn(
                    column.name.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Canonical bytes used to bind this parser/output contract into the solver
    /// input closure as a `Configuration` artifact.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ElmerBridgeError> {
        self.validate()?;
        let mut bytes = Vec::new();
        push_field(&mut bytes, "schema", &self.schema_id);
        push_field(&mut bytes, "output_filename", &self.output_filename);
        for column in &self.columns {
            push_field(&mut bytes, "column_name", &column.name);
            push_field(&mut bytes, "column_unit", &column.unit);
        }
        Ok(bytes)
    }

    pub fn content_digest(&self) -> Result<ContentDigest, ElmerBridgeError> {
        Ok(ContentDigest::blake3(&self.canonical_bytes()?))
    }
}

/// Closure-bound, non-executing Elmer case contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElmerCaseManifest {
    pub profile_id: String,
    pub case_id: String,
    /// Closure artifact containing the top-level SIF bytes.
    pub sif_artifact_id: String,
    /// Closure `Configuration` artifact whose digest must equal the canonical
    /// scalar schema digest below.
    pub scalar_schema_artifact_id: String,
    pub scalar_schema: ElmerScalarSchema,
    pub input_closure: SolverInputClosure,
}

impl ElmerCaseManifest {
    pub fn validate(&self) -> Result<(), ElmerBridgeError> {
        if self.profile_id != ELMER_CASE_PROFILE_ID {
            return Err(ElmerBridgeError::UnsupportedCaseProfile(
                self.profile_id.clone(),
            ));
        }
        if self.case_id.trim().is_empty() || self.case_id.trim() != self.case_id {
            return Err(ElmerBridgeError::InvalidCaseId(self.case_id.clone()));
        }
        if self.sif_artifact_id.trim().is_empty()
            || self.scalar_schema_artifact_id.trim().is_empty()
        {
            return Err(ElmerBridgeError::MissingArtifactBinding);
        }
        self.scalar_schema.validate()?;
        self.input_closure.validate()?;

        let sif = self
            .input_closure
            .artifacts
            .iter()
            .find(|artifact| artifact.id == self.sif_artifact_id)
            .ok_or_else(|| ElmerBridgeError::MissingSifArtifact(self.sif_artifact_id.clone()))?;
        if sif.role != SolverInputRole::Primary {
            return Err(ElmerBridgeError::SifArtifactMustBePrimary {
                artifact: sif.id.clone(),
                actual: sif.role,
            });
        }

        let schema_artifact = self
            .input_closure
            .artifacts
            .iter()
            .find(|artifact| artifact.id == self.scalar_schema_artifact_id)
            .ok_or_else(|| {
                ElmerBridgeError::MissingScalarSchemaArtifact(
                    self.scalar_schema_artifact_id.clone(),
                )
            })?;
        if schema_artifact.role != SolverInputRole::Configuration {
            return Err(ElmerBridgeError::ScalarSchemaArtifactMustBeConfiguration {
                artifact: schema_artifact.id.clone(),
                actual: schema_artifact.role,
            });
        }
        let expected = self.scalar_schema.content_digest()?;
        if schema_artifact.digest != expected {
            return Err(ElmerBridgeError::ScalarSchemaDigestMismatch {
                expected: expected.canonical_string(),
                actual: schema_artifact.digest.canonical_string(),
            });
        }

        // The generic closure theorem requires exactly one solver executable;
        // verify that this case is specifically bound to Elmer rather than some
        // arbitrary FEM executable while still leaving exact version policy to
        // qualification profiles.
        let solver = self
            .input_closure
            .artifacts
            .iter()
            .find(|artifact| artifact.role == SolverInputRole::SolverExecutable)
            .ok_or(ElmerBridgeError::MissingSolverExecutable)?;
        let identity = solver
            .reported_identity
            .as_deref()
            .ok_or(ElmerBridgeError::MissingSolverExecutable)?;
        if !identity.to_ascii_lowercase().contains("elmer") {
            return Err(ElmerBridgeError::SolverIdentityNotElmer(identity.into()));
        }

        Ok(())
    }

    pub fn closure_id(&self) -> Result<String, ElmerBridgeError> {
        self.validate()?;
        Ok(self.input_closure.closure_id()?)
    }

    /// Strictly parse the adapter-owned SaveScalars file.
    ///
    /// Blank lines and lines beginning with `#` or `!` are ignored. Every data
    /// row must contain exactly the declared number of whitespace-separated
    /// finite scalar values. Headers, missing cells, extra cells, NaN/Inf and
    /// arbitrary log text fail closed.
    pub fn parse_save_scalars(&self, text: &str) -> Result<Vec<ElmerScalarRow>, ElmerBridgeError> {
        self.validate()?;
        let mut rows = Vec::new();
        for (line_index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                continue;
            }
            let tokens: Vec<_> = line.split_whitespace().collect();
            if tokens.len() != self.scalar_schema.columns.len() {
                return Err(ElmerBridgeError::ScalarColumnCount {
                    line: line_index + 1,
                    expected: self.scalar_schema.columns.len(),
                    actual: tokens.len(),
                });
            }
            let mut values = Vec::with_capacity(tokens.len());
            for token in tokens {
                let value = token.parse::<f64>().map_err(|_| {
                    ElmerBridgeError::InvalidScalarToken {
                        line: line_index + 1,
                        token: token.into(),
                    }
                })?;
                if !value.is_finite() {
                    return Err(ElmerBridgeError::NonFiniteScalar {
                        line: line_index + 1,
                        value,
                    });
                }
                values.push(value);
            }
            rows.push(ElmerScalarRow { values });
        }
        if rows.is_empty() {
            return Err(ElmerBridgeError::NoScalarRows);
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElmerScalarRow {
    pub values: Vec<f64>,
}

/// First Elmer adapter boundary. Real execution is intentionally disabled until
/// the shared execution-context theorem in ENG-EXEC-001 lands in CommandSolver.
#[derive(Debug, Clone)]
pub struct ElmerBridge {
    pub case: ElmerCaseManifest,
    pub dry_run: bool,
}

impl ElmerBridge {
    pub fn new(case: ElmerCaseManifest) -> Self {
        Self {
            case,
            dry_run: false,
        }
    }

    pub fn dry_run(case: ElmerCaseManifest) -> Self {
        Self { case, dry_run: true }
    }
}

impl SimulationBackend for ElmerBridge {
    fn name(&self) -> &'static str {
        "elmer"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::FiniteElement, SolverKind::MultiPhysics]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        request.validate()?;
        if !self.supported_solvers().contains(&request.solver) {
            return Err(SimulationError::InvalidRequest(format!(
                "Elmer bridge cannot satisfy {:?}",
                request.solver
            )));
        }
        if !matches!(
            request.domain,
            EngineeringDomain::Mechanical
                | EngineeringDomain::Electrical
                | EngineeringDomain::Aerospace
                | EngineeringDomain::Robotics
                | EngineeringDomain::Nuclear
                | EngineeringDomain::Materials
                | EngineeringDomain::Environmental
                | EngineeringDomain::Systems
                | EngineeringDomain::Civil
        ) {
            return Err(SimulationError::InvalidRequest(format!(
                "Elmer bridge does not admit {:?} under ENG-FEM-001A",
                request.domain
            )));
        }
        self.case
            .validate()
            .map_err(|error| SimulationError::Adapter(error.to_string()))?;

        if self.dry_run {
            let mut result = SimulationResult::dry_run(&request.id, self.name(), 0.0);
            result.warnings.push(format!(
                "Elmer case validated with solver input closure {}; no external solver was executed",
                self.case
                    .closure_id()
                    .map_err(|error| SimulationError::Adapter(error.to_string()))?
            ));
            return Ok(result);
        }

        Err(SimulationError::Adapter(
            "real Elmer execution is intentionally unavailable in ENG-FEM-001A until ENG-EXEC-001 binds working directory and ambient environment in the shared CommandSolver"
                .into(),
        ))
    }
}

fn validate_local_filename(value: &str) -> Result<(), ElmerBridgeError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(ElmerBridgeError::InvalidOutputFilename(value.into()));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(ElmerBridgeError::InvalidOutputFilename(value.into()));
    }
    let components: Vec<_> = path.components().collect();
    if components.len() != 1 || !matches!(components[0], Component::Normal(_)) {
        return Err(ElmerBridgeError::InvalidOutputFilename(value.into()));
    }
    Ok(())
}

fn push_field(bytes: &mut Vec<u8>, name: &str, value: &str) {
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Error)]
pub enum ElmerBridgeError {
    #[error(transparent)]
    Closure(#[from] ClosureError),
    #[error("unsupported Elmer case profile {0:?}")]
    UnsupportedCaseProfile(String),
    #[error("unsupported Elmer scalar schema {0:?}")]
    UnsupportedScalarSchema(String),
    #[error("invalid Elmer case id {0:?}")]
    InvalidCaseId(String),
    #[error("case is missing required closure artifact binding")]
    MissingArtifactBinding,
    #[error("SaveScalars output filename must be one local filename, got {0:?}")]
    InvalidOutputFilename(String),
    #[error("SaveScalars schema requires at least one scalar column")]
    NoScalarColumns,
    #[error("SaveScalars columns require non-empty name and unit")]
    InvalidScalarColumn,
    #[error("SaveScalars column is not canonical: name={name:?}, unit={unit:?}")]
    NonCanonicalScalarColumn { name: String, unit: String },
    #[error("duplicate SaveScalars column {0:?}")]
    DuplicateScalarColumn(String),
    #[error("closure is missing SIF artifact {0:?}")]
    MissingSifArtifact(String),
    #[error("SIF artifact {artifact:?} must be Primary, got {actual:?}")]
    SifArtifactMustBePrimary {
        artifact: String,
        actual: SolverInputRole,
    },
    #[error("closure is missing scalar-schema artifact {0:?}")]
    MissingScalarSchemaArtifact(String),
    #[error("scalar-schema artifact {artifact:?} must be Configuration, got {actual:?}")]
    ScalarSchemaArtifactMustBeConfiguration {
        artifact: String,
        actual: SolverInputRole,
    },
    #[error("scalar-schema digest mismatch: expected {expected}, got {actual}")]
    ScalarSchemaDigestMismatch { expected: String, actual: String },
    #[error("closure is missing its solver executable")]
    MissingSolverExecutable,
    #[error("solver executable identity does not identify Elmer: {0:?}")]
    SolverIdentityNotElmer(String),
    #[error("SaveScalars row {line} has {actual} columns; expected {expected}")]
    ScalarColumnCount {
        line: usize,
        expected: usize,
        actual: usize,
    },
    #[error("SaveScalars row {line} contains non-numeric token {token:?}")]
    InvalidScalarToken { line: usize, token: String },
    #[error("SaveScalars row {line} contains non-finite value {value}")]
    NonFiniteScalar { line: usize, value: f64 },
    #[error("SaveScalars output contains no scalar rows")]
    NoScalarRows,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_solver_closure::{
        AmbientDiscoveryPolicy, SolverInputArtifact, SolverInputClosure,
    };

    fn schema() -> ElmerScalarSchema {
        ElmerScalarSchema::new(
            "symthaea-scalars.dat",
            vec![
                ElmerScalarColumn {
                    name: "max_temperature".into(),
                    unit: "K".into(),
                },
                ElmerScalarColumn {
                    name: "max_displacement".into(),
                    unit: "m".into(),
                },
            ],
        )
    }

    fn closure(schema: &ElmerScalarSchema) -> SolverInputClosure {
        let primary = SolverInputArtifact::new(
            "case-sif",
            SolverInputRole::Primary,
            ContentDigest::blake3(b"Simulation\nEnd\n"),
        )
        .with_locator("case.sif");
        let solver = SolverInputArtifact::new(
            "elmer-solver",
            SolverInputRole::SolverExecutable,
            ContentDigest::blake3(b"fake-elmer-binary"),
        )
        .with_reported_identity("ElmerSolver 9.0");
        let config = SolverInputArtifact::new(
            "scalar-schema",
            SolverInputRole::Configuration,
            schema.content_digest().unwrap(),
        )
        .with_parent("case-sif");
        SolverInputClosure::new(
            "elmer-fixture-v1",
            AmbientDiscoveryPolicy::Prohibited,
            vec![primary, solver, config],
            vec!["ambient case discovery disabled".into()],
        )
        .unwrap()
    }

    fn case() -> ElmerCaseManifest {
        let schema = schema();
        ElmerCaseManifest {
            profile_id: ELMER_CASE_PROFILE_ID.into(),
            case_id: "fixture-case".into(),
            sif_artifact_id: "case-sif".into(),
            scalar_schema_artifact_id: "scalar-schema".into(),
            input_closure: closure(&schema),
            scalar_schema: schema,
        }
    }

    #[test]
    fn valid_case_binds_sif_schema_and_elmer_solver() {
        let case = case();
        assert!(case.validate().is_ok());
        assert!(case.closure_id().unwrap().starts_with("blake3:"));
    }

    #[test]
    fn scalar_schema_change_breaks_closure_binding() {
        let mut case = case();
        case.scalar_schema.columns[0].unit = "degC".into();
        assert!(matches!(
            case.validate(),
            Err(ElmerBridgeError::ScalarSchemaDigestMismatch { .. })
        ));
    }

    #[test]
    fn strict_scalar_parser_accepts_finite_rows_and_comments() {
        let rows = case()
            .parse_save_scalars("# values\n300.0 1.0e-4\n! second row\n310.5 2.5e-4\n")
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values, vec![300.0, 1.0e-4]);
    }

    #[test]
    fn scalar_parser_rejects_headers_extra_columns_and_non_finite_values() {
        assert!(matches!(
            case().parse_save_scalars("temperature displacement\n"),
            Err(ElmerBridgeError::InvalidScalarToken { .. })
        ));
        assert!(matches!(
            case().parse_save_scalars("300 1e-4 9\n"),
            Err(ElmerBridgeError::ScalarColumnCount { .. })
        ));
        assert!(matches!(
            case().parse_save_scalars("NaN 1e-4\n"),
            Err(ElmerBridgeError::NonFiniteScalar { .. })
        ));
    }

    #[test]
    fn output_path_cannot_escape_case_directory() {
        let mut schema = schema();
        schema.output_filename = "../escape.dat".into();
        assert!(matches!(
            schema.validate(),
            Err(ElmerBridgeError::InvalidOutputFilename(_))
        ));
    }

    #[test]
    fn real_execution_remains_blocked_until_generic_context_binding_lands() {
        let backend = ElmerBridge::new(case());
        let request = SimulationRequest::new(
            "elmer-real-blocked",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
            "qualification boundary test",
        );
        assert!(matches!(
            backend.run(&request),
            Err(SimulationError::Adapter(message)) if message.contains("ENG-EXEC-001")
        ));
    }

    #[test]
    fn dry_run_is_not_external_engineering_evidence() {
        let backend = ElmerBridge::dry_run(case());
        let request = SimulationRequest::new(
            "elmer-dry",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
            "orchestration test",
        );
        let result = backend.run(&request).unwrap();
        assert!(!result.is_engineering_evidence());
        assert!(result.metrics.is_empty());
    }
}
