// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducibility capsules and capability guards for magnetic solver adapters.
//!
//! This crate does not execute external programs. It defines the exact scientific
//! request/result boundary that future DFT, spin-model, and micromagnetic adapters
//! must satisfy before their outputs can be normalized into `symthaea-magnetics`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use symthaea_magnetics::{MagneticEvidencePlane, MagneticPropertyKind};
use thiserror::Error;

const SHA256_HEX_LEN: usize = 64;

/// Immutable artifact reference used by solver reproducibility capsules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverArtifactRef {
    /// Stable artifact identifier.
    pub artifact_id: String,
    /// SHA-256 of exact artifact bytes.
    pub sha256: String,
}

impl SolverArtifactRef {
    fn validate(&self) -> Result<(), MagneticSolverError> {
        nonempty("artifact_id", &self.artifact_id)?;
        sha256(&self.sha256)
    }
}

/// Broad magnetic numerical method family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagneticSolverFamily {
    /// Collinear/non-collinear spin-polarized electronic-structure calculation.
    SpinPolarizedDft,
    /// Relativistic/spin-orbit DFT suitable for magnetocrystalline anisotropy.
    RelativisticDft,
    /// Exchange-parameter/stiffness extraction.
    ExchangeParameter,
    /// Finite-temperature spin/Heisenberg/Monte-Carlo-type calculation.
    FiniteTemperatureSpin,
    /// Micromagnetic hysteresis/domain calculation.
    Micromagnetics,
}

impl MagneticSolverFamily {
    fn supports(self, property: MagneticPropertyKind) -> bool {
        match self {
            Self::SpinPolarizedDft => matches!(
                property,
                MagneticPropertyKind::SaturationPolarization
                    | MagneticPropertyKind::SaturationMagnetization
            ),
            Self::RelativisticDft => matches!(
                property,
                MagneticPropertyKind::SaturationPolarization
                    | MagneticPropertyKind::SaturationMagnetization
                    | MagneticPropertyKind::MagnetocrystallineAnisotropyK1
            ),
            Self::ExchangeParameter => property == MagneticPropertyKind::ExchangeStiffness,
            Self::FiniteTemperatureSpin => property == MagneticPropertyKind::CurieTemperature,
            Self::Micromagnetics => property.plane() == MagneticEvidencePlane::Extrinsic,
        }
    }
}

/// Exact solver executable identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverExecutableRef {
    /// Stable solver/backend ID.
    pub solver_id: String,
    /// Reported/package version.
    pub version: String,
    /// SHA-256 of the exact executable/package artifact used for this run.
    pub executable_sha256: String,
}

/// Exact execution environment/container/Nix closure identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverEnvironmentRef {
    /// Stable environment ID.
    pub environment_id: String,
    /// SHA-256 of the exact environment/closure manifest.
    pub artifact_sha256: String,
}

/// Parser/normalizer version that turns raw solver output into scientific observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverParserRef {
    /// Stable parser ID.
    pub parser_id: String,
    /// Parser version.
    pub version: String,
    /// SHA-256 of exact parser artifact/source capsule.
    pub artifact_sha256: String,
}

/// Fully bound magnetic solver request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticSolverRequest {
    /// Stable request ID.
    pub request_id: String,
    /// Exact MAT-007 material subject identity.
    pub subject_identity: String,
    /// Numerical method family.
    pub family: MagneticSolverFamily,
    /// Exact solver executable/package.
    pub solver: SolverExecutableRef,
    /// Exact execution environment.
    pub environment: SolverEnvironmentRef,
    /// Exact rendered input deck/workflow input bundle.
    pub input_deck: SolverArtifactRef,
    /// Exact parser expected to normalize the raw output.
    pub parser: SolverParserRef,
    /// Magnetic quantities requested from this method.
    pub requested_properties: Vec<MagneticPropertyKind>,
    /// Exact MAG-002 microstructure comparison identity when the method needs one.
    pub microstructure_context_identity: Option<String>,
    /// Whether spin-orbit coupling is explicitly enabled in the bound input/method.
    pub spin_orbit_coupling: bool,
}

impl MagneticSolverRequest {
    /// Validate solver capabilities and provenance completeness.
    pub fn validate(&self) -> Result<(), MagneticSolverError> {
        nonempty("request_id", &self.request_id)?;
        nonempty("subject_identity", &self.subject_identity)?;
        validate_solver(&self.solver)?;
        validate_environment(&self.environment)?;
        self.input_deck.validate()?;
        validate_parser(&self.parser)?;
        if self.requested_properties.is_empty() {
            return Err(MagneticSolverError::NoRequestedProperties);
        }
        let mut properties = HashSet::new();
        for property in &self.requested_properties {
            if !properties.insert(format!("{property:?}")) {
                return Err(MagneticSolverError::DuplicateRequestedProperty(*property));
            }
            if !self.family.supports(*property) {
                return Err(MagneticSolverError::UnsupportedProperty {
                    family: self.family,
                    property: *property,
                });
            }
            if *property == MagneticPropertyKind::MagnetocrystallineAnisotropyK1
                && !self.spin_orbit_coupling
            {
                return Err(MagneticSolverError::SpinOrbitRequiredForAnisotropy);
            }
            if property.plane() == MagneticEvidencePlane::Extrinsic
                && self
                    .microstructure_context_identity
                    .as_deref()
                    .is_none_or(|value| value.trim().is_empty())
            {
                return Err(MagneticSolverError::MicrostructureRequiredForExtrinsicSolver);
            }
        }
        if self.family != MagneticSolverFamily::Micromagnetics
            && self.requested_properties.iter().any(|property| {
                property.plane() == MagneticEvidencePlane::Extrinsic
            })
        {
            return Err(MagneticSolverError::ExtrinsicPropertyRequiresMicromagnetics);
        }
        if let Some(context) = &self.microstructure_context_identity {
            nonempty("microstructure_context_identity", context)?;
        }
        Ok(())
    }

    /// Deterministic exact request identity. Any solver/environment/input/parser mutation changes it.
    pub fn canonical_identity(&self) -> Result<String, MagneticSolverError> {
        self.validate()?;
        let mut properties = self
            .requested_properties
            .iter()
            .map(|property| format!("{property:?}"))
            .collect::<Vec<_>>();
        properties.sort();
        Ok(format!(
            "magnetic-solver-request:v1|id={}|subject={}|family={:?}|solver={}:{}:{}|env={}:{}|input={}:{}|parser={}:{}:{}|properties=[{}]|microstructure={}|soc={}",
            token(&self.request_id),
            token(&self.subject_identity),
            self.family,
            token(&self.solver.solver_id),
            token(&self.solver.version),
            self.solver.executable_sha256.to_ascii_lowercase(),
            token(&self.environment.environment_id),
            self.environment.artifact_sha256.to_ascii_lowercase(),
            token(&self.input_deck.artifact_id),
            self.input_deck.sha256.to_ascii_lowercase(),
            token(&self.parser.parser_id),
            token(&self.parser.version),
            self.parser.artifact_sha256.to_ascii_lowercase(),
            properties.join(","),
            self.microstructure_context_identity
                .as_deref()
                .map(token)
                .unwrap_or_else(|| "none".to_string()),
            self.spin_orbit_coupling,
        ))
    }
}

/// Raw-output/result capsule returned by an adapter after an external solver run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticSolverResultCapsule {
    /// Exact canonical request identity satisfied by this result.
    pub request_identity: String,
    /// Process exit code when available.
    pub exit_code: Option<i32>,
    /// Whether the domain-specific solver/parser established convergence.
    pub converged: bool,
    /// Raw stdout artifact when retained.
    pub stdout: Option<SolverArtifactRef>,
    /// Raw stderr artifact when retained.
    pub stderr: Option<SolverArtifactRef>,
    /// Other raw output/checkpoint/result files consumed by the parser.
    pub raw_outputs: Vec<SolverArtifactRef>,
    /// Exact parser that produced normalized evidence IDs.
    pub parser: SolverParserRef,
    /// MAG-002/MAT-008 evidence record IDs emitted by normalization.
    pub produced_evidence_ids: Vec<String>,
    /// Adapter/parser warnings; never silently discarded.
    pub warnings: Vec<String>,
}

impl MagneticSolverResultCapsule {
    /// Validate raw-output and evidence lineage semantics.
    pub fn validate_for(&self, request: &MagneticSolverRequest) -> Result<(), MagneticSolverError> {
        request.validate()?;
        if self.request_identity != request.canonical_identity()? {
            return Err(MagneticSolverError::ResultRequestMismatch);
        }
        validate_parser(&self.parser)?;
        if self.parser != request.parser {
            return Err(MagneticSolverError::ParserMismatch);
        }
        if let Some(stdout) = &self.stdout {
            stdout.validate()?;
        }
        if let Some(stderr) = &self.stderr {
            stderr.validate()?;
        }
        let mut artifacts = HashSet::new();
        for output in &self.raw_outputs {
            output.validate()?;
            if !artifacts.insert((output.artifact_id.as_str(), output.sha256.as_str())) {
                return Err(MagneticSolverError::DuplicateRawOutput);
            }
        }
        let mut evidence = HashSet::new();
        for id in &self.produced_evidence_ids {
            nonempty("produced_evidence_id", id)?;
            if !evidence.insert(id.as_str()) {
                return Err(MagneticSolverError::DuplicateProducedEvidence(id.clone()));
            }
        }

        if self.converged {
            if self.exit_code != Some(0) {
                return Err(MagneticSolverError::ConvergedRunRequiresZeroExit);
            }
            if self.raw_outputs.is_empty() {
                return Err(MagneticSolverError::ConvergedRunRequiresRawOutput);
            }
            if self.produced_evidence_ids.is_empty() {
                return Err(MagneticSolverError::ConvergedRunRequiresEvidence);
            }
        } else if !self.produced_evidence_ids.is_empty() {
            return Err(MagneticSolverError::NonConvergedRunCannotEmitPropertyEvidence);
        }
        Ok(())
    }

    /// True only when a converged, zero-exit, provenance-complete run produced evidence.
    pub fn is_qualifiable_for(&self, request: &MagneticSolverRequest) -> bool {
        self.validate_for(request).is_ok()
            && self.converged
            && self.exit_code == Some(0)
            && !self.raw_outputs.is_empty()
            && !self.produced_evidence_ids.is_empty()
    }
}

fn validate_solver(value: &SolverExecutableRef) -> Result<(), MagneticSolverError> {
    nonempty("solver_id", &value.solver_id)?;
    nonempty("solver version", &value.version)?;
    sha256(&value.executable_sha256)
}

fn validate_environment(value: &SolverEnvironmentRef) -> Result<(), MagneticSolverError> {
    nonempty("environment_id", &value.environment_id)?;
    sha256(&value.artifact_sha256)
}

fn validate_parser(value: &SolverParserRef) -> Result<(), MagneticSolverError> {
    nonempty("parser_id", &value.parser_id)?;
    nonempty("parser version", &value.version)?;
    sha256(&value.artifact_sha256)
}

fn nonempty(field: &'static str, value: &str) -> Result<(), MagneticSolverError> {
    if value.trim().is_empty() {
        Err(MagneticSolverError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256(value: &str) -> Result<(), MagneticSolverError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(MagneticSolverError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Magnetic solver capsule validation failure.
#[derive(Debug, Error, PartialEq)]
pub enum MagneticSolverError {
    /// Required text was empty.
    #[error("required solver field is empty: {0}")]
    EmptyField(&'static str),
    /// Malformed SHA-256.
    #[error("invalid SHA-256")]
    InvalidSha256,
    /// No magnetic outputs requested.
    #[error("magnetic solver request contains no requested properties")]
    NoRequestedProperties,
    /// Same property requested twice.
    #[error("duplicate requested magnetic property: {0:?}")]
    DuplicateRequestedProperty(MagneticPropertyKind),
    /// Solver family cannot establish the requested quantity.
    #[error("solver family {family:?} cannot establish property {property:?}")]
    UnsupportedProperty {
        /// Solver family.
        family: MagneticSolverFamily,
        /// Requested property.
        property: MagneticPropertyKind,
    },
    /// MAE/K1 calculation omitted SOC.
    #[error("spin-orbit coupling is required for magnetocrystalline anisotropy")]
    SpinOrbitRequiredForAnisotropy,
    /// Extrinsic solver request omitted microstructure context.
    #[error("extrinsic magnetic solver request requires microstructure context")]
    MicrostructureRequiredForExtrinsicSolver,
    /// An extrinsic property was routed to a non-micromagnetic family.
    #[error("extrinsic permanent-magnet performance requires a micromagnetic solver family")]
    ExtrinsicPropertyRequiresMicromagnetics,
    /// Result did not bind the exact request.
    #[error("solver result request identity mismatch")]
    ResultRequestMismatch,
    /// Result parser did not match requested parser.
    #[error("solver result parser mismatch")]
    ParserMismatch,
    /// Raw output artifact repeated.
    #[error("duplicate raw solver output artifact")]
    DuplicateRawOutput,
    /// Evidence ID repeated.
    #[error("duplicate produced evidence ID: {0}")]
    DuplicateProducedEvidence(String),
    /// Converged result did not have exit code zero.
    #[error("converged solver run requires exit code 0")]
    ConvergedRunRequiresZeroExit,
    /// Converged result omitted raw output artifacts.
    #[error("converged solver run requires bound raw outputs")]
    ConvergedRunRequiresRawOutput,
    /// Converged result emitted no normalized evidence.
    #[error("converged solver run requires normalized evidence IDs")]
    ConvergedRunRequiresEvidence,
    /// Failed/non-converged run attempted to emit property evidence.
    #[error("non-converged run cannot emit magnetic property evidence")]
    NonConvergedRunCannotEmitPropertyEvidence,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn artifact(id: &str, hash: &str) -> SolverArtifactRef {
        SolverArtifactRef {
            artifact_id: id.to_string(),
            sha256: hash.to_string(),
        }
    }

    fn relativistic_dft() -> MagneticSolverRequest {
        MagneticSolverRequest {
            request_id: "mag-dft-001".to_string(),
            subject_identity: "material-subject:v1|Fe5Co18Zr6".to_string(),
            family: MagneticSolverFamily::RelativisticDft,
            solver: SolverExecutableRef {
                solver_id: "qe-pw".to_string(),
                version: "fixture".to_string(),
                executable_sha256: A64.to_string(),
            },
            environment: SolverEnvironmentRef {
                environment_id: "nix-closure-fixture".to_string(),
                artifact_sha256: B64.to_string(),
            },
            input_deck: artifact("input-deck", C64),
            parser: SolverParserRef {
                parser_id: "magnetic-dft-parser".to_string(),
                version: "1".to_string(),
                artifact_sha256: D64.to_string(),
            },
            requested_properties: vec![
                MagneticPropertyKind::SaturationPolarization,
                MagneticPropertyKind::MagnetocrystallineAnisotropyK1,
            ],
            microstructure_context_identity: None,
            spin_orbit_coupling: true,
        }
    }

    #[test]
    fn dft_cannot_request_coercivity() {
        let mut request = relativistic_dft();
        request.requested_properties = vec![MagneticPropertyKind::CoerciveField];
        request.microstructure_context_identity = Some("microstructure:v1|fixture".to_string());
        assert!(matches!(
            request.validate(),
            Err(MagneticSolverError::UnsupportedProperty { .. })
        ));
    }

    #[test]
    fn anisotropy_requires_spin_orbit_coupling() {
        let mut request = relativistic_dft();
        request.spin_orbit_coupling = false;
        assert_eq!(
            request.validate(),
            Err(MagneticSolverError::SpinOrbitRequiredForAnisotropy)
        );
    }

    #[test]
    fn micromagnetics_requires_microstructure_context() {
        let mut request = relativistic_dft();
        request.family = MagneticSolverFamily::Micromagnetics;
        request.requested_properties = vec![MagneticPropertyKind::CoerciveField];
        request.spin_orbit_coupling = false;
        request.microstructure_context_identity = None;
        assert_eq!(
            request.validate(),
            Err(MagneticSolverError::MicrostructureRequiredForExtrinsicSolver)
        );
    }

    #[test]
    fn environment_or_solver_mutation_changes_request_identity() {
        let a = relativistic_dft();
        let mut b = a.clone();
        b.environment.artifact_sha256 = C64.to_string();
        assert_ne!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());

        let mut c = a.clone();
        c.solver.executable_sha256 = B64.to_string();
        assert_ne!(a.canonical_identity().unwrap(), c.canonical_identity().unwrap());
    }

    #[test]
    fn converged_result_requires_raw_output_and_evidence() {
        let request = relativistic_dft();
        let result = MagneticSolverResultCapsule {
            request_identity: request.canonical_identity().unwrap(),
            exit_code: Some(0),
            converged: true,
            stdout: None,
            stderr: None,
            raw_outputs: vec![],
            parser: request.parser.clone(),
            produced_evidence_ids: vec![],
            warnings: vec![],
        };
        assert_eq!(
            result.validate_for(&request),
            Err(MagneticSolverError::ConvergedRunRequiresRawOutput)
        );
    }

    #[test]
    fn nonconverged_run_cannot_emit_property_evidence() {
        let request = relativistic_dft();
        let result = MagneticSolverResultCapsule {
            request_identity: request.canonical_identity().unwrap(),
            exit_code: Some(1),
            converged: false,
            stdout: Some(artifact("stdout", A64)),
            stderr: Some(artifact("stderr", B64)),
            raw_outputs: vec![artifact("raw-output", C64)],
            parser: request.parser.clone(),
            produced_evidence_ids: vec!["should-not-exist".to_string()],
            warnings: vec!["solver did not converge".to_string()],
        };
        assert_eq!(
            result.validate_for(&request),
            Err(MagneticSolverError::NonConvergedRunCannotEmitPropertyEvidence)
        );
    }

    #[test]
    fn complete_converged_capsule_is_qualifiable_input_to_normalization() {
        let request = relativistic_dft();
        let result = MagneticSolverResultCapsule {
            request_identity: request.canonical_identity().unwrap(),
            exit_code: Some(0),
            converged: true,
            stdout: Some(artifact("stdout", A64)),
            stderr: None,
            raw_outputs: vec![artifact("raw-output", C64)],
            parser: request.parser.clone(),
            produced_evidence_ids: vec!["mag-evidence-k1".to_string()],
            warnings: vec![],
        };
        result.validate_for(&request).unwrap();
        assert!(result.is_qualifiable_for(&request));
    }
}
