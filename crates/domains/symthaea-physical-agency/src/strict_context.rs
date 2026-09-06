// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict typed-context simulation lineage for Physical Agency.
//!
//! This module deliberately wraps the generic `symthaea-sim-bridge` rather than
//! changing legacy simulation requests in place. Only adapters that explicitly
//! implement [`ContextAwareSimulationBackend`] can enter this path.
//!
//! The first version uses an exact canonical request transcript as the lineage
//! identity. Byte-for-byte comparison has no hash-collision ambiguity and adds
//! no new dependency. A compact cryptographic digest may later be derived from
//! the transcript, but must never replace it as the source of truth.
//!
//! A validated receipt is structural simulation evidence only. It is neither an
//! authentication of backend honesty nor physical execution authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, ModelParameter, SimulationError, SimulationRequest,
    SimulationResult, SolverKind, UncertaintyEstimate,
};
use thiserror::Error;

pub const CONTEXT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationContextKind {
    WorldSnapshot,
    GeometrySnapshot,
    MaterialDataset,
    BoundaryConditions,
    CalibrationSnapshot,
    EnvironmentSnapshot,
    InitialState,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextDigestAlgorithm {
    Blake3,
    Sha256,
}

impl ContextDigestAlgorithm {
    fn tag(self) -> &'static str {
        match self {
            Self::Blake3 => "blake3-256",
            Self::Sha256 => "sha256",
        }
    }
}

/// Immutable identity for solver-relevant context.
///
/// A context digest establishes equality under the producer's canonicalization
/// contract. It does not establish that the referenced state is true, current,
/// safe, or faithfully consumed by a malicious backend.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SimulationContextRef {
    pub schema_version: u16,
    pub kind: SimulationContextKind,
    pub context_id: String,
    pub digest_algorithm: ContextDigestAlgorithm,
    pub digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_ref: Option<String>,
}

impl SimulationContextRef {
    pub fn world_snapshot(
        context_id: impl Into<String>,
        digest_algorithm: ContextDigestAlgorithm,
        digest: impl Into<String>,
        frame_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: CONTEXT_SCHEMA_VERSION,
            kind: SimulationContextKind::WorldSnapshot,
            context_id: context_id.into(),
            digest_algorithm,
            digest: digest.into(),
            frame_id: Some(frame_id.into()),
            provenance_ref: None,
        }
    }

    pub fn validate(&self) -> Result<(), StrictSimulationError> {
        if self.schema_version != CONTEXT_SCHEMA_VERSION {
            return Err(StrictSimulationError::UnsupportedContextSchema(
                self.schema_version,
            ));
        }
        if self.context_id.trim().is_empty() {
            return Err(StrictSimulationError::InvalidContext(
                "context_id cannot be empty".into(),
            ));
        }
        if self.digest.len() != 64 || !self.digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(StrictSimulationError::InvalidContext(format!(
                "context {:?} requires a 32-byte hexadecimal digest",
                self.context_id
            )));
        }
        if self
            .frame_id
            .as_deref()
            .is_some_and(|frame| frame.trim().is_empty())
        {
            return Err(StrictSimulationError::InvalidContext(
                "frame_id cannot be empty when present".into(),
            ));
        }
        if self
            .provenance_ref
            .as_deref()
            .is_some_and(|reference| reference.trim().is_empty())
        {
            return Err(StrictSimulationError::InvalidContext(
                "provenance_ref cannot be empty when present".into(),
            ));
        }
        if matches!(&self.kind, SimulationContextKind::WorldSnapshot)
            && self.frame_id.as_deref().is_none_or(|frame| frame.is_empty())
        {
            return Err(StrictSimulationError::InvalidContext(
                "world snapshots require a frame_id".into(),
            ));
        }
        if let SimulationContextKind::Custom(namespace) = &self.kind {
            if namespace.trim().is_empty() || !namespace.contains(':') {
                return Err(StrictSimulationError::InvalidContext(
                    "custom context kind must use a non-empty namespaced identifier".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Exact machine-relevant normalized request identity.
///
/// The bytes are canonical and deterministic across context/parameter/metric
/// vector ordering. The human-readable request objective is deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalRequestTranscript {
    bytes: Vec<u8>,
}

impl CanonicalRequestTranscript {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Legacy normalized request plus the immutable contexts a strict adapter must
/// claim to consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextBoundSimulationRequest {
    pub request: SimulationRequest,
    pub contexts: Vec<SimulationContextRef>,
}

impl ContextBoundSimulationRequest {
    pub fn new(request: SimulationRequest, contexts: Vec<SimulationContextRef>) -> Self {
        Self { request, contexts }
    }

    pub fn validate(&self) -> Result<(), StrictSimulationError> {
        self.request
            .validate()
            .map_err(|error| StrictSimulationError::Bridge(error.to_string()))?;
        validate_context_set(&self.contexts)
    }

    pub fn canonical_transcript(&self) -> Result<CanonicalRequestTranscript, StrictSimulationError> {
        self.validate()?;
        let mut bytes = Vec::new();
        push_str(&mut bytes, "symthaea.physical-agency.sim-context.v1");
        push_str(&mut bytes, &self.request.id);
        push_str(&mut bytes, domain_tag(self.request.domain));
        push_str(&mut bytes, solver_tag(self.request.solver));

        let mut parameters = self
            .request
            .parameters
            .iter()
            .map(parameter_bytes)
            .collect::<Vec<_>>();
        parameters.sort();
        push_u64(&mut bytes, parameters.len() as u64);
        for parameter in parameters {
            push_bytes(&mut bytes, &parameter);
        }

        let mut metrics = self.request.requested_metrics.clone();
        metrics.sort();
        push_u64(&mut bytes, metrics.len() as u64);
        for metric in metrics {
            push_str(&mut bytes, &metric);
        }

        let contexts = canonical_contexts(&self.contexts)?;
        push_u64(&mut bytes, contexts.len() as u64);
        for context in contexts {
            push_context(&mut bytes, &context);
        }

        Ok(CanonicalRequestTranscript { bytes })
    }
}

/// Backend-reported context consumption. This remains caller-constructible
/// evidence; the strict registry must compare it to its own request transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextConsumptionEvidence {
    pub request_transcript: CanonicalRequestTranscript,
    pub consumed_contexts: Vec<SimulationContextRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextBoundSimulationResult {
    pub result: SimulationResult,
    pub consumption: ContextConsumptionEvidence,
}

/// Explicit opt-in adapter surface. Implementing ordinary `SimulationBackend`
/// does not make an adapter context-aware.
pub trait ContextAwareSimulationBackend: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_solvers(&self) -> &[SolverKind];
    fn run_context_bound(
        &self,
        request: &ContextBoundSimulationRequest,
    ) -> Result<ContextBoundSimulationResult, SimulationError>;
}

/// Non-serializable runtime receipt proving only that the strict registry saw a
/// structurally valid external-solver result whose reported context/transcript
/// exactly matched the request it dispatched.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryValidatedContextSimulation {
    result: SimulationResult,
    request_transcript: CanonicalRequestTranscript,
    contexts: Vec<SimulationContextRef>,
    backend: String,
}

impl RegistryValidatedContextSimulation {
    pub fn result(&self) -> &SimulationResult {
        &self.result
    }

    pub fn request_transcript(&self) -> &CanonicalRequestTranscript {
        &self.request_transcript
    }

    pub fn contexts(&self) -> &[SimulationContextRef] {
        &self.contexts
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }
}

#[derive(Default, Debug)]
pub struct StrictSimulationRegistry {
    backends: Vec<Box<dyn ContextAwareSimulationBackend>>,
}

impl StrictSimulationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, backend: impl ContextAwareSimulationBackend + 'static) {
        self.backends.push(Box::new(backend));
    }

    pub fn find_backend(&self, solver: SolverKind) -> Option<&dyn ContextAwareSimulationBackend> {
        self.backends
            .iter()
            .find(|backend| backend.supported_solvers().contains(&solver))
            .map(|backend| backend.as_ref())
    }

    pub fn run(
        &self,
        request: &ContextBoundSimulationRequest,
    ) -> Result<RegistryValidatedContextSimulation, StrictSimulationError> {
        request.validate()?;
        let expected_transcript = request.canonical_transcript()?;
        let expected_contexts = canonical_contexts(&request.contexts)?;

        let backend = self
            .find_backend(request.request.solver)
            .ok_or(StrictSimulationError::SolverUnavailable(
                request.request.solver,
            ))?;
        let returned = backend
            .run_context_bound(request)
            .map_err(|error| StrictSimulationError::Bridge(error.to_string()))?;
        returned
            .result
            .validate()
            .map_err(|error| StrictSimulationError::Bridge(error.to_string()))?;

        if returned.result.request_id != request.request.id {
            return Err(StrictSimulationError::RequestIdMismatch {
                expected: request.request.id.clone(),
                actual: returned.result.request_id.clone(),
            });
        }
        if returned.result.evidence.mode != ExecutionMode::ExternalSolver
            || !returned.result.is_engineering_evidence()
        {
            return Err(StrictSimulationError::NotEngineeringEvidence);
        }
        if returned.result.evidence.backend.as_deref() != Some(backend.name()) {
            return Err(StrictSimulationError::BackendMismatch {
                expected: backend.name().to_string(),
                actual: returned.result.evidence.backend.clone(),
            });
        }
        if returned.consumption.request_transcript != expected_transcript {
            return Err(StrictSimulationError::RequestTranscriptMismatch);
        }

        let consumed = canonical_contexts(&returned.consumption.consumed_contexts)?;
        if consumed != expected_contexts {
            return Err(StrictSimulationError::ContextSetMismatch {
                expected: expected_contexts,
                actual: consumed,
            });
        }

        Ok(RegistryValidatedContextSimulation {
            result: returned.result,
            request_transcript: expected_transcript,
            contexts: expected_contexts,
            backend: backend.name().to_string(),
        })
    }
}

fn validate_context_set(contexts: &[SimulationContextRef]) -> Result<(), StrictSimulationError> {
    if contexts.is_empty() {
        return Err(StrictSimulationError::MissingContext);
    }
    let mut identities = BTreeSet::new();
    for context in contexts {
        context.validate()?;
        let identity = (context.kind.clone(), context.context_id.clone());
        if !identities.insert(identity.clone()) {
            return Err(StrictSimulationError::DuplicateContextIdentity {
                kind: identity.0,
                context_id: identity.1,
            });
        }
    }
    Ok(())
}

fn canonical_contexts(
    contexts: &[SimulationContextRef],
) -> Result<Vec<SimulationContextRef>, StrictSimulationError> {
    validate_context_set(contexts)?;
    let mut canonical = contexts.to_vec();
    for context in &mut canonical {
        context.digest.make_ascii_lowercase();
    }
    canonical.sort();
    Ok(canonical)
}

fn push_context(bytes: &mut Vec<u8>, context: &SimulationContextRef) {
    push_u64(bytes, context.schema_version as u64);
    match &context.kind {
        SimulationContextKind::WorldSnapshot => push_str(bytes, "world_snapshot"),
        SimulationContextKind::GeometrySnapshot => push_str(bytes, "geometry_snapshot"),
        SimulationContextKind::MaterialDataset => push_str(bytes, "material_dataset"),
        SimulationContextKind::BoundaryConditions => push_str(bytes, "boundary_conditions"),
        SimulationContextKind::CalibrationSnapshot => push_str(bytes, "calibration_snapshot"),
        SimulationContextKind::EnvironmentSnapshot => push_str(bytes, "environment_snapshot"),
        SimulationContextKind::InitialState => push_str(bytes, "initial_state"),
        SimulationContextKind::Custom(namespace) => {
            push_str(bytes, "custom");
            push_str(bytes, namespace);
        }
    }
    push_str(bytes, &context.context_id);
    push_str(bytes, context.digest_algorithm.tag());
    push_str(bytes, &context.digest.to_ascii_lowercase());
    push_opt_str(bytes, context.frame_id.as_deref());
    push_opt_str(bytes, context.provenance_ref.as_deref());
}

fn parameter_bytes(parameter: &ModelParameter) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_str(&mut bytes, &parameter.name);
    push_u64(&mut bytes, parameter.value.to_bits());
    push_str(&mut bytes, &parameter.unit);
    push_str(&mut bytes, &parameter.provenance);
    match parameter.uncertainty {
        Some(uncertainty) => {
            bytes.push(1);
            push_uncertainty(&mut bytes, uncertainty);
        }
        None => bytes.push(0),
    }
    bytes
}

fn push_uncertainty(bytes: &mut Vec<u8>, uncertainty: UncertaintyEstimate) {
    push_u64(bytes, uncertainty.epistemic.to_bits());
    push_u64(bytes, uncertainty.aleatoric.to_bits());
    match uncertainty.interval {
        Some(interval) => {
            bytes.push(1);
            push_u64(bytes, interval.lower.to_bits());
            push_u64(bytes, interval.upper.to_bits());
        }
        None => bytes.push(0),
    }
}

fn domain_tag(domain: EngineeringDomain) -> &'static str {
    match domain {
        EngineeringDomain::Civil => "civil",
        EngineeringDomain::Mechanical => "mechanical",
        EngineeringDomain::Electrical => "electrical",
        EngineeringDomain::Aerospace => "aerospace",
        EngineeringDomain::ChemicalProcess => "chemical_process",
        EngineeringDomain::Robotics => "robotics",
        EngineeringDomain::Nuclear => "nuclear",
        EngineeringDomain::Materials => "materials",
        EngineeringDomain::Environmental => "environmental",
        EngineeringDomain::Systems => "systems",
    }
}

fn solver_tag(solver: SolverKind) -> &'static str {
    match solver {
        SolverKind::FiniteElement => "finite_element",
        SolverKind::ComputationalFluidDynamics => "computational_fluid_dynamics",
        SolverKind::MultibodyDynamics => "multibody_dynamics",
        SolverKind::Circuit => "circuit",
        SolverKind::Process => "process",
        SolverKind::CadGeometry => "cad_geometry",
        SolverKind::MultiPhysics => "multi_physics",
        SolverKind::Custom => "custom",
    }
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(target: &mut Vec<u8>, value: &[u8]) {
    push_u64(target, value.len() as u64);
    target.extend_from_slice(value);
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_bytes(bytes, value.as_bytes());
}

fn push_opt_str(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            push_str(bytes, value);
        }
        None => bytes.push(0),
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StrictSimulationError {
    #[error("typed simulation context set cannot be empty")]
    MissingContext,
    #[error("unsupported simulation-context schema version {0}")]
    UnsupportedContextSchema(u16),
    #[error("invalid simulation context: {0}")]
    InvalidContext(String),
    #[error("duplicate context identity {kind:?}/{context_id:?}")]
    DuplicateContextIdentity {
        kind: SimulationContextKind,
        context_id: String,
    },
    #[error("context-aware solver unavailable: {0:?}")]
    SolverUnavailable(SolverKind),
    #[error("simulation bridge error: {0}")]
    Bridge(String),
    #[error("backend returned request id {actual:?}, expected {expected:?}")]
    RequestIdMismatch { expected: String, actual: String },
    #[error("context-aware run was not external-solver engineering evidence")]
    NotEngineeringEvidence,
    #[error("external evidence backend mismatch: expected {expected:?}, got {actual:?}")]
    BackendMismatch {
        expected: String,
        actual: Option<String>,
    },
    #[error("backend-reported canonical request transcript did not match dispatched request")]
    RequestTranscriptMismatch,
    #[error("backend-reported consumed context set did not exactly match requested contexts")]
    ContextSetMismatch {
        expected: Vec<SimulationContextRef>,
        actual: Vec<SimulationContextRef>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{SimulationEvidence, UncertaintyEstimate};

    #[derive(Debug, Clone, Copy)]
    enum Behavior {
        Good,
        Missing,
        Substitute,
        Extra,
        Reordered,
        StaleTranscript,
        DryRun,
    }

    #[derive(Debug)]
    struct FixtureBackend(Behavior);

    impl ContextAwareSimulationBackend for FixtureBackend {
        fn name(&self) -> &'static str {
            "context-fixture"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::Custom]
        }

        fn run_context_bound(
            &self,
            request: &ContextBoundSimulationRequest,
        ) -> Result<ContextBoundSimulationResult, SimulationError> {
            let mut contexts = request.contexts.clone();
            let mut transcript = request
                .canonical_transcript()
                .map_err(|error| SimulationError::Adapter(error.to_string()))?;

            match self.0 {
                Behavior::Missing => contexts.clear(),
                Behavior::Substitute => contexts[0].digest = "b".repeat(64),
                Behavior::Extra => contexts.push(geometry()),
                Behavior::Reordered => contexts.reverse(),
                Behavior::StaleTranscript => transcript.bytes.push(0),
                Behavior::Good | Behavior::DryRun => {}
            }

            let result = if matches!(self.0, Behavior::DryRun) {
                SimulationResult::dry_run(&request.request.id, self.name(), 0.95)
                    .with_metric("diagnostic_quality", 0.9, "1")
            } else {
                SimulationResult::converged(&request.request.id, 0.95)
                    .with_uncertainty(UncertaintyEstimate::new(0.05, 0.02))
                    .with_metric("diagnostic_quality", 0.9, "1")
                    .with_external_evidence(SimulationEvidence {
                        mode: ExecutionMode::ExternalSolver,
                        backend: Some(self.name().into()),
                        solver_version: Some("fixture-1".into()),
                        input_digest: Some("input-digest".into()),
                        output_digest: Some("output-digest".into()),
                        parser_version: Some("parser-1".into()),
                    })
            };

            Ok(ContextBoundSimulationResult {
                result,
                consumption: ContextConsumptionEvidence {
                    request_transcript: transcript,
                    consumed_contexts: contexts,
                },
            })
        }
    }

    fn world() -> SimulationContextRef {
        SimulationContextRef::world_snapshot(
            "world-1",
            ContextDigestAlgorithm::Blake3,
            "a".repeat(64),
            "world",
        )
    }

    fn geometry() -> SimulationContextRef {
        SimulationContextRef {
            schema_version: CONTEXT_SCHEMA_VERSION,
            kind: SimulationContextKind::GeometrySnapshot,
            context_id: "geometry-1".into(),
            digest_algorithm: ContextDigestAlgorithm::Sha256,
            digest: "c".repeat(64),
            frame_id: Some("world".into()),
            provenance_ref: Some("fixture:geometry".into()),
        }
    }

    fn request(contexts: Vec<SimulationContextRef>) -> ContextBoundSimulationRequest {
        ContextBoundSimulationRequest::new(
            SimulationRequest::new(
                "ctx-run-1",
                EngineeringDomain::Systems,
                SolverKind::Custom,
                "strict context fixture",
            )
            .with_parameter("scale", 1.0, "1", "fixture"),
            contexts,
        )
    }

    fn registry(behavior: Behavior) -> StrictSimulationRegistry {
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend(behavior));
        registry
    }

    #[test]
    fn exact_context_lineage_qualifies_structurally() {
        let run = registry(Behavior::Good).run(&request(vec![world()])).unwrap();
        assert_eq!(run.backend(), "context-fixture");
        assert_eq!(run.contexts(), &[world()]);
        assert!(!run.request_transcript().as_bytes().is_empty());
        assert!(run.result().is_engineering_evidence());
    }

    #[test]
    fn missing_substituted_extra_and_stale_evidence_fail_closed() {
        for behavior in [
            Behavior::Missing,
            Behavior::Substitute,
            Behavior::Extra,
            Behavior::StaleTranscript,
        ] {
            assert!(registry(behavior).run(&request(vec![world()])).is_err());
        }
    }

    #[test]
    fn context_order_is_canonical_not_semantic() {
        let contexts = vec![world(), geometry()];
        let run = registry(Behavior::Reordered)
            .run(&request(contexts.clone()))
            .unwrap();
        assert_eq!(run.contexts().len(), 2);

        let a = request(contexts);
        let b = request(vec![geometry(), world()]);
        assert_eq!(
            a.canonical_transcript().unwrap(),
            b.canonical_transcript().unwrap()
        );
    }

    #[test]
    fn duplicate_context_identity_is_rejected_before_dispatch() {
        let mut duplicate = world();
        duplicate.digest = "d".repeat(64);
        let error = registry(Behavior::Good)
            .run(&request(vec![world(), duplicate]))
            .unwrap_err();
        assert!(matches!(
            error,
            StrictSimulationError::DuplicateContextIdentity { .. }
        ));
    }

    #[test]
    fn dry_run_cannot_masquerade_as_context_validated_engineering_evidence() {
        assert_eq!(
            registry(Behavior::DryRun)
                .run(&request(vec![world()]))
                .unwrap_err(),
            StrictSimulationError::NotEngineeringEvidence
        );
    }

    #[test]
    fn machine_lineage_ignores_human_objective_but_binds_parameters() {
        let base = request(vec![world()]);
        let mut wording = base.clone();
        wording.request.objective = "different human wording".into();
        assert_eq!(
            base.canonical_transcript().unwrap(),
            wording.canonical_transcript().unwrap()
        );

        let mut changed = base.clone();
        changed.request.parameters[0].value = 2.0;
        assert_ne!(
            base.canonical_transcript().unwrap(),
            changed.canonical_transcript().unwrap()
        );
    }
}
