// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducibility-bound qualification for external simulation results.
//!
//! `symthaea-sim-bridge` deliberately provides a broad adapter API. Its legacy
//! `SimulationEvidence` distinguishes dry runs from parsed external-solver output,
//! but solver version strings are not enough to reproduce scientific computation.
//! This crate adds a stricter qualification layer without breaking existing
//! adapters: exact adapter, executable, execution environment, rendered command/input,
//! raw outputs, parser artifact, exit state, and convergence evidence must all be
//! bound and must agree with the normalized `SimulationResult` provenance.
//!
//! Qualification here means reproducible solver evidence. It does not by itself
//! establish that the numerical model is scientifically valid for a particular
//! domain or that a downstream safety/materials claim is experimentally true.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_sim_bridge::{ExecutionMode, SimulationResult};
use thiserror::Error;

/// Exact immutable artifact reference used in a solver evidence capsule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverArtifactRef {
    /// Stable artifact identifier.
    pub artifact_id: String,
    /// SHA-256 of exact artifact bytes.
    pub sha256: String,
}

impl SolverArtifactRef {
    fn validate(&self) -> Result<(), SimulationEvidenceError> {
        nonempty("artifact_id", &self.artifact_id)?;
        validate_sha256(&self.sha256)
    }
}

/// Exact solver executable/package identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverExecutableRef {
    /// Stable solver identifier such as `OpenFOAM`, `OpenSees`, `QE-pw`, or `MuMax3`.
    pub solver_id: String,
    /// Human-readable/package version.
    pub version: String,
    /// SHA-256 of the exact executable/package artifact.
    pub executable_sha256: String,
}

impl SolverExecutableRef {
    fn validate(&self) -> Result<(), SimulationEvidenceError> {
        nonempty("solver_id", &self.solver_id)?;
        nonempty("solver version", &self.version)?;
        validate_sha256(&self.executable_sha256)
    }
}

/// Exact execution environment / Nix closure / container identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionEnvironmentRef {
    /// Stable environment identifier.
    pub environment_id: String,
    /// SHA-256 of exact closure/container/environment manifest.
    pub manifest_sha256: String,
}

impl ExecutionEnvironmentRef {
    fn validate(&self) -> Result<(), SimulationEvidenceError> {
        nonempty("environment_id", &self.environment_id)?;
        validate_sha256(&self.manifest_sha256)
    }
}

/// Exact parser/normalizer used to turn raw outputs into normalized metrics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverParserRef {
    /// Stable parser identifier.
    pub parser_id: String,
    /// Parser version label.
    pub version: String,
    /// SHA-256 of exact parser source/binary artifact.
    pub artifact_sha256: String,
}

impl SolverParserRef {
    fn validate(&self) -> Result<(), SimulationEvidenceError> {
        nonempty("parser_id", &self.parser_id)?;
        nonempty("parser version", &self.version)?;
        validate_sha256(&self.artifact_sha256)
    }
}

/// Strong external-solver evidence capsule layered over a normalized simulation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedSimulationEvidence {
    /// Stable adapter/backend name. Must match `SimulationResult.evidence.backend`.
    pub backend: String,
    /// Exact adapter/translation implementation that rendered and normalized the run.
    pub adapter_artifact: SolverArtifactRef,
    /// Exact solver executable/package.
    pub solver: SolverExecutableRef,
    /// Exact execution environment.
    pub environment: ExecutionEnvironmentRef,
    /// Exact rendered command/workflow manifest with secrets removed before hashing.
    pub command_manifest: SolverArtifactRef,
    /// Exact complete solver input bundle.
    pub input_bundle: SolverArtifactRef,
    /// Exact raw output bundle consumed by the parser.
    pub raw_output_bundle: SolverArtifactRef,
    /// Exact parser/normalizer.
    pub parser: SolverParserRef,
    /// Artifact demonstrating domain-specific convergence, not merely process exit.
    pub convergence_evidence: SolverArtifactRef,
    /// Captured stdout artifact when retained.
    pub stdout: Option<SolverArtifactRef>,
    /// Captured stderr artifact when retained.
    pub stderr: Option<SolverArtifactRef>,
    /// External process exit code.
    pub exit_code: i32,
    /// Normalized result emitted by the ordinary simulation adapter.
    pub result: SimulationResult,
}

impl QualifiedSimulationEvidence {
    /// Validate exact provenance and agreement with the legacy normalized result.
    pub fn validate(&self) -> Result<(), SimulationEvidenceError> {
        nonempty("backend", &self.backend)?;
        self.adapter_artifact.validate()?;
        self.solver.validate()?;
        self.environment.validate()?;
        self.command_manifest.validate()?;
        self.input_bundle.validate()?;
        self.raw_output_bundle.validate()?;
        self.parser.validate()?;
        self.convergence_evidence.validate()?;
        if let Some(stdout) = &self.stdout {
            stdout.validate()?;
        }
        if let Some(stderr) = &self.stderr {
            stderr.validate()?;
        }

        if self.exit_code != 0 {
            return Err(SimulationEvidenceError::NonZeroExitCode(self.exit_code));
        }
        self.result
            .validate()
            .map_err(|error| SimulationEvidenceError::NormalizedResult(error.to_string()))?;
        if !self.result.converged {
            return Err(SimulationEvidenceError::ResultNotConverged);
        }
        if self.result.metrics.is_empty() {
            return Err(SimulationEvidenceError::ResultHasNoMetrics);
        }
        if self.result.evidence.mode != ExecutionMode::ExternalSolver {
            return Err(SimulationEvidenceError::NotExternalSolverEvidence);
        }
        if !self.result.is_engineering_evidence() {
            return Err(SimulationEvidenceError::LegacyProvenanceIncomplete);
        }

        let legacy = &self.result.evidence;
        if legacy.backend.as_deref() != Some(self.backend.as_str()) {
            return Err(SimulationEvidenceError::BackendMismatch);
        }
        if legacy.solver_version.as_deref() != Some(self.solver.version.as_str()) {
            return Err(SimulationEvidenceError::SolverVersionMismatch);
        }
        if !legacy
            .input_digest
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(&self.input_bundle.sha256))
        {
            return Err(SimulationEvidenceError::InputDigestMismatch);
        }
        if !legacy
            .output_digest
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(&self.raw_output_bundle.sha256))
        {
            return Err(SimulationEvidenceError::OutputDigestMismatch);
        }
        if legacy.parser_version.as_deref() != Some(self.parser.version.as_str()) {
            return Err(SimulationEvidenceError::ParserVersionMismatch);
        }
        Ok(())
    }

    /// True only for a fully reproducibility-bound, converged external-solver result.
    pub fn is_qualified(&self) -> bool {
        self.validate().is_ok()
    }

    /// SHA-256 of the exact serialized qualification capsule.
    ///
    /// This binds the normalized metrics as well as all raw/execution provenance.
    pub fn qualification_sha256(&self) -> Result<String, SimulationEvidenceError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), SimulationEvidenceError> {
    if value.trim().is_empty() {
        Err(SimulationEvidenceError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_sha256(value: &str) -> Result<(), SimulationEvidenceError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(SimulationEvidenceError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Strong solver-evidence qualification failure.
#[derive(Debug, Error)]
pub enum SimulationEvidenceError {
    /// Required text field empty.
    #[error("required solver evidence field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid solver evidence SHA-256")]
    InvalidSha256,
    /// External process did not exit successfully.
    #[error("external solver exited with non-zero code: {0}")]
    NonZeroExitCode(i32),
    /// Normalized result failed ordinary adapter validation.
    #[error("normalized simulation result failed validation: {0}")]
    NormalizedResult(String),
    /// Adapter did not establish domain convergence.
    #[error("normalized simulation result is not converged")]
    ResultNotConverged,
    /// No normalized metrics were parsed.
    #[error("normalized simulation result contains no metrics")]
    ResultHasNoMetrics,
    /// Result was a dry run/unknown rather than parsed external-solver output.
    #[error("simulation result is not external-solver evidence")]
    NotExternalSolverEvidence,
    /// Legacy adapter provenance was incomplete.
    #[error("legacy simulation provenance is incomplete")]
    LegacyProvenanceIncomplete,
    /// Strong backend identity disagreed with normalized result.
    #[error("solver evidence backend mismatch")]
    BackendMismatch,
    /// Strong solver version disagreed with normalized result.
    #[error("solver evidence version mismatch")]
    SolverVersionMismatch,
    /// Exact rendered input bundle disagreed with normalized result provenance.
    #[error("solver input digest mismatch")]
    InputDigestMismatch,
    /// Exact raw output bundle disagreed with normalized result provenance.
    #[error("solver output digest mismatch")]
    OutputDigestMismatch,
    /// Parser version disagreed with normalized result provenance.
    #[error("solver parser version mismatch")]
    ParserVersionMismatch,
    /// Qualification serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{SimulationEvidence, SimulationResult};

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E64: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F64: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn artifact(id: &str, sha: &str) -> SolverArtifactRef {
        SolverArtifactRef {
            artifact_id: id.to_string(),
            sha256: sha.to_string(),
        }
    }

    fn normalized() -> SimulationResult {
        SimulationResult::converged("fixture-request", 0.95)
            .with_metric("max_stress_mpa", 123.0, "MPa")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("fixture-solver".to_string()),
                solver_version: Some("1.2.3".to_string()),
                input_digest: Some(C64.to_string()),
                output_digest: Some(D64.to_string()),
                parser_version: Some("7".to_string()),
            })
    }

    fn qualified() -> QualifiedSimulationEvidence {
        QualifiedSimulationEvidence {
            backend: "fixture-solver".to_string(),
            adapter_artifact: artifact("adapter", F64),
            solver: SolverExecutableRef {
                solver_id: "fixture".to_string(),
                version: "1.2.3".to_string(),
                executable_sha256: A64.to_string(),
            },
            environment: ExecutionEnvironmentRef {
                environment_id: "nix-fixture".to_string(),
                manifest_sha256: B64.to_string(),
            },
            command_manifest: artifact("command", E64),
            input_bundle: artifact("input", C64),
            raw_output_bundle: artifact("raw-output", D64),
            parser: SolverParserRef {
                parser_id: "fixture-parser".to_string(),
                version: "7".to_string(),
                artifact_sha256: F64.to_string(),
            },
            convergence_evidence: artifact("convergence", D64),
            stdout: Some(artifact("stdout", E64)),
            stderr: None,
            exit_code: 0,
            result: normalized(),
        }
    }

    #[test]
    fn fully_bound_external_solver_result_qualifies() {
        let evidence = qualified();
        evidence.validate().unwrap();
        assert!(evidence.is_qualified());
        assert_eq!(evidence.qualification_sha256().unwrap().len(), 64);
    }

    #[test]
    fn dry_run_can_never_be_qualified_solver_evidence() {
        let mut evidence = qualified();
        evidence.result = SimulationResult::dry_run("fixture-request", "fixture-solver", 0.5)
            .with_metric("max_stress_mpa", 123.0, "MPa");
        assert!(matches!(
            evidence.validate(),
            Err(SimulationEvidenceError::NotExternalSolverEvidence)
        ));
    }

    #[test]
    fn successful_process_exit_is_not_domain_convergence() {
        let mut evidence = qualified();
        evidence.result.converged = false;
        assert!(matches!(
            evidence.validate(),
            Err(SimulationEvidenceError::ResultNotConverged)
        ));
    }

    #[test]
    fn input_and_raw_output_must_match_adapter_provenance() {
        let mut evidence = qualified();
        evidence.input_bundle.sha256 = E64.to_string();
        assert!(matches!(
            evidence.validate(),
            Err(SimulationEvidenceError::InputDigestMismatch)
        ));

        let mut evidence = qualified();
        evidence.raw_output_bundle.sha256 = E64.to_string();
        assert!(matches!(
            evidence.validate(),
            Err(SimulationEvidenceError::OutputDigestMismatch)
        ));
    }

    #[test]
    fn implementation_or_environment_change_alters_qualification_identity() {
        let a = qualified();

        let mut adapter = a.clone();
        adapter.adapter_artifact.sha256 = A64.to_string();
        assert_ne!(
            a.qualification_sha256().unwrap(),
            adapter.qualification_sha256().unwrap()
        );

        let mut solver = a.clone();
        solver.solver.executable_sha256 = B64.to_string();
        assert_ne!(
            a.qualification_sha256().unwrap(),
            solver.qualification_sha256().unwrap()
        );

        let mut environment = a.clone();
        environment.environment.manifest_sha256 = C64.to_string();
        assert_ne!(
            a.qualification_sha256().unwrap(),
            environment.qualification_sha256().unwrap()
        );
    }

    #[test]
    fn parser_artifact_changes_qualification_identity_without_faking_new_metrics() {
        let a = qualified();
        let mut b = a.clone();
        b.parser.artifact_sha256 = A64.to_string();
        assert_ne!(
            a.qualification_sha256().unwrap(),
            b.qualification_sha256().unwrap()
        );
        assert_eq!(a.result.metrics, b.result.metrics);
    }

    #[test]
    fn nonzero_exit_code_is_rejected_even_if_result_claims_convergence() {
        let mut evidence = qualified();
        evidence.exit_code = 2;
        assert!(matches!(
            evidence.validate(),
            Err(SimulationEvidenceError::NonZeroExitCode(2))
        ));
    }
}
