// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selection-bound strict simulation preparation and evidence.
//!
//! PA-12 removes caller-written world-snapshot identity from the strict run
//! path. The frame, digest bytes, and digest scheme are derived directly from
//! the non-serializable [`SelectedCandidate`] minted by deliberation. The
//! prepared request and returned evidence receipt are also non-serializable.
//!
//! This is still simulation-only structural evidence. It grants no physical
//! execution authority and does not authenticate a malicious solver backend.

use crate::deliberation::{SelectedCandidate, SnapshotDigestAlgorithm};
use crate::strict_context::{
    ContextBoundSimulationRequest, ContextDigestAlgorithm, RegistryValidatedContextSimulation,
    SimulationContextKind, SimulationContextRef, StrictSimulationError, StrictSimulationRegistry,
};
use symthaea_sim_bridge::SimulationRequest;
use thiserror::Error;

/// Non-serializable binding between one deliberated selection and the exact
/// strict simulation request prepared from its world snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSelectedSimulation {
    selected: SelectedCandidate,
    request: ContextBoundSimulationRequest,
}

impl PreparedSelectedSimulation {
    pub fn selected(&self) -> &SelectedCandidate {
        &self.selected
    }

    pub fn request(&self) -> &ContextBoundSimulationRequest {
        &self.request
    }
}

/// Build a strict context-bound request directly from the selected world state.
///
/// The caller supplies no world frame, digest, or digest algorithm. All three
/// are inherited from `SelectedCandidate::world_snapshot`. Historical
/// `LegacyOpaque` snapshots remain valid for deliberation but fail closed here.
pub fn prepare_selected_simulation(
    selected: &SelectedCandidate,
    request: SimulationRequest,
) -> Result<PreparedSelectedSimulation, SelectionBoundSimulationError> {
    let snapshot = selected.world_snapshot();
    let digest_algorithm = strict_digest_algorithm(snapshot.digest_algorithm())?;
    let world_context = SimulationContextRef::world_snapshot(
        format!("world-snapshot:{}", snapshot.frame_id()),
        digest_algorithm,
        snapshot.snapshot_digest(),
        snapshot.frame_id(),
    );
    let request = ContextBoundSimulationRequest::new(request, vec![world_context]);
    request
        .validate()
        .map_err(SelectionBoundSimulationError::Strict)?;

    Ok(PreparedSelectedSimulation {
        selected: selected.clone(),
        request,
    })
}

fn strict_digest_algorithm(
    algorithm: SnapshotDigestAlgorithm,
) -> Result<ContextDigestAlgorithm, SelectionBoundSimulationError> {
    match algorithm {
        SnapshotDigestAlgorithm::LegacyOpaque => {
            Err(SelectionBoundSimulationError::LegacySnapshotDigest)
        }
        SnapshotDigestAlgorithm::Blake3 => Ok(ContextDigestAlgorithm::Blake3),
        SnapshotDigestAlgorithm::Sha256 => Ok(ContextDigestAlgorithm::Sha256),
    }
}

/// Non-serializable simulation receipt preserving both the deliberative
/// selection and registry-validated strict context lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionBoundSimulationEvidence {
    selected: SelectedCandidate,
    validated: RegistryValidatedContextSimulation,
}

impl SelectionBoundSimulationEvidence {
    pub fn selected(&self) -> &SelectedCandidate {
        &self.selected
    }

    pub fn validated(&self) -> &RegistryValidatedContextSimulation {
        &self.validated
    }
}

/// Execute a previously prepared selection-bound request.
///
/// Defense-in-depth rechecks the validated world context against the selected
/// snapshot even though `PreparedSelectedSimulation` has private fields and can
/// only be constructed by [`prepare_selected_simulation`].
pub fn run_prepared_selected_simulation(
    registry: &StrictSimulationRegistry,
    prepared: &PreparedSelectedSimulation,
) -> Result<SelectionBoundSimulationEvidence, SelectionBoundSimulationError> {
    let validated = registry
        .run(prepared.request())
        .map_err(SelectionBoundSimulationError::Strict)?;
    validate_selected_world_context(prepared.selected(), &validated)?;

    Ok(SelectionBoundSimulationEvidence {
        selected: prepared.selected().clone(),
        validated,
    })
}

fn validate_selected_world_context(
    selected: &SelectedCandidate,
    validated: &RegistryValidatedContextSimulation,
) -> Result<(), SelectionBoundSimulationError> {
    let worlds = validated
        .contexts()
        .iter()
        .filter(|context| matches!(&context.kind, SimulationContextKind::WorldSnapshot))
        .collect::<Vec<_>>();
    if worlds.len() != 1 {
        return Err(SelectionBoundSimulationError::WorldContextCount(worlds.len()));
    }

    let context = worlds[0];
    let snapshot = selected.world_snapshot();
    let expected_algorithm = strict_digest_algorithm(snapshot.digest_algorithm())?;
    if context.digest_algorithm != expected_algorithm {
        return Err(SelectionBoundSimulationError::SelectedSnapshotAlgorithmMismatch {
            selected: snapshot.digest_algorithm(),
            context: context.digest_algorithm,
        });
    }
    if context.frame_id.as_deref() != Some(snapshot.frame_id())
        || !context.digest.eq_ignore_ascii_case(snapshot.snapshot_digest())
    {
        return Err(SelectionBoundSimulationError::SelectedSnapshotMismatch {
            selected_frame: snapshot.frame_id().to_string(),
            selected_digest: snapshot.snapshot_digest().to_string(),
            context_frame: context.frame_id.clone(),
            context_digest: context.digest.clone(),
        });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SelectionBoundSimulationError {
    #[error("strict simulation context failure: {0}")]
    Strict(StrictSimulationError),
    #[error("legacy opaque world snapshots cannot enter the strict simulation-evidence path")]
    LegacySnapshotDigest,
    #[error("strict simulation evidence contains {0} world contexts; exactly one is required")]
    WorldContextCount(usize),
    #[error(
        "selected snapshot digest algorithm {selected:?} does not match validated context {context:?}"
    )]
    SelectedSnapshotAlgorithmMismatch {
        selected: SnapshotDigestAlgorithm,
        context: ContextDigestAlgorithm,
    },
    #[error(
        "selected snapshot {selected_frame:?}/{selected_digest:?} does not match validated context {context_frame:?}/{context_digest:?}"
    )]
    SelectedSnapshotMismatch {
        selected_frame: String,
        selected_digest: String,
        context_frame: Option<String>,
        context_digest: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deliberation::{DeliberationOutcome, WorldSnapshotRef, deliberate};
    use crate::portfolio::{
        CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
    };
    use crate::strict_context::{
        CanonicalRequestTranscript, ContextAwareSimulationBackend, ContextBoundSimulationResult,
        ContextConsumptionEvidence,
    };
    use symthaea_physical_effects::{
        AuthorityClass, DesiredTransition, EffectKind, MechanismRef, PhysicalModality,
        PredictedOutcome, ProposedIntervention, TargetRegion,
    };
    use symthaea_sim_bridge::{
        EngineeringDomain, ExecutionMode, SimulationEvidence, SimulationError, SimulationResult,
        SolverKind, UncertaintyEstimate,
    };

    #[derive(Debug)]
    struct FixtureBackend;

    impl ContextAwareSimulationBackend for FixtureBackend {
        fn name(&self) -> &'static str {
            "selection-context-fixture"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::Custom]
        }

        fn run_context_bound(
            &self,
            request: &ContextBoundSimulationRequest,
        ) -> Result<ContextBoundSimulationResult, SimulationError> {
            let transcript: CanonicalRequestTranscript = request
                .canonical_transcript()
                .map_err(|error| SimulationError::Adapter(error.to_string()))?;
            let result = SimulationResult::converged(&request.request.id, 0.96)
                .with_uncertainty(UncertaintyEstimate::new(0.03, 0.01))
                .with_metric("diagnostic_quality", 0.92, "1")
                .with_external_evidence(SimulationEvidence {
                    mode: ExecutionMode::ExternalSolver,
                    backend: Some(self.name().into()),
                    solver_version: Some("fixture-1".into()),
                    input_digest: Some("input-digest".into()),
                    output_digest: Some("output-digest".into()),
                    parser_version: Some("parser-1".into()),
                });
            Ok(ContextBoundSimulationResult {
                result,
                consumption: ContextConsumptionEvidence {
                    request_transcript: transcript,
                    consumed_contexts: request.contexts.clone(),
                },
            })
        }
    }

    fn selected_from_snapshot(snapshot: WorldSnapshotRef) -> SelectedCandidate {
        let transition = DesiredTransition::simulation_only(
            "strict-selection-t0",
            "selection-bound diagnostic simulation",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let candidate = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "strict-selection-p0".into(),
                transition_id: "strict-selection-t0".into(),
                mechanism: MechanismRef {
                    backend: "fixture-model".into(),
                    mechanism: "diagnostic".into(),
                    modality: PhysicalModality::Acoustic,
                },
                required_authority: AuthorityClass::SimulationOnly,
                predicted_outcome: PredictedOutcome {
                    success_probability: 0.9,
                    epistemic_uncertainty: 0.08,
                    aleatoric_uncertainty: 0.03,
                },
            },
            model_predictions: vec![
                ModelPrediction {
                    model_id: "model-a".into(),
                    success_probability: 0.9,
                },
                ModelPrediction {
                    model_id: "model-b".into(),
                    success_probability: 0.88,
                },
            ],
            expected_energy_j: 1.0,
            expected_power_w: None,
            expected_duration_ms: 100,
            information_gain: 0.8,
            reversibility_score: 1.0,
            safety_margin: 0.95,
        };
        let portfolio = CandidatePortfolio {
            transition,
            candidates: vec![candidate],
        };
        let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };
        frontier.select("strict-selection-p0").unwrap()
    }

    fn cryptographic_selected(algorithm: SnapshotDigestAlgorithm) -> SelectedCandidate {
        selected_from_snapshot(WorldSnapshotRef::cryptographic(
            "world",
            algorithm,
            "a".repeat(64),
        ))
    }

    fn request() -> SimulationRequest {
        SimulationRequest::new(
            "strict-selection-run-0",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "verify selected diagnostic simulation",
        )
    }

    #[test]
    fn world_context_is_derived_from_non_serializable_selection() {
        let selected = cryptographic_selected(SnapshotDigestAlgorithm::Blake3);
        let prepared = prepare_selected_simulation(&selected, request()).unwrap();

        assert_eq!(prepared.selected(), &selected);
        assert_eq!(prepared.request().contexts.len(), 1);
        let context = &prepared.request().contexts[0];
        assert_eq!(context.frame_id.as_deref(), Some("world"));
        assert_eq!(context.digest, "a".repeat(64));
        assert_eq!(context.context_id, "world-snapshot:world");
        assert_eq!(context.digest_algorithm, ContextDigestAlgorithm::Blake3);
    }

    #[test]
    fn digest_algorithm_cannot_be_relabelled_by_strict_request_caller() {
        let blake = cryptographic_selected(SnapshotDigestAlgorithm::Blake3);
        let sha = cryptographic_selected(SnapshotDigestAlgorithm::Sha256);
        let blake_prepared = prepare_selected_simulation(&blake, request()).unwrap();
        let sha_prepared = prepare_selected_simulation(&sha, request()).unwrap();

        assert_eq!(
            blake_prepared.request().contexts[0].digest_algorithm,
            ContextDigestAlgorithm::Blake3
        );
        assert_eq!(
            sha_prepared.request().contexts[0].digest_algorithm,
            ContextDigestAlgorithm::Sha256
        );
        assert_ne!(
            blake_prepared.request().canonical_transcript().unwrap(),
            sha_prepared.request().canonical_transcript().unwrap()
        );
    }

    #[test]
    fn legacy_non_cryptographic_snapshot_identifier_cannot_enter_strict_path() {
        let selected = selected_from_snapshot(WorldSnapshotRef::new(
            "world",
            "legacy-snapshot-name",
        ));
        assert_eq!(
            prepare_selected_simulation(&selected, request()).unwrap_err(),
            SelectionBoundSimulationError::LegacySnapshotDigest
        );
    }

    #[test]
    fn strict_run_retains_exact_selected_lineage() {
        let selected = cryptographic_selected(SnapshotDigestAlgorithm::Blake3);
        let prepared = prepare_selected_simulation(&selected, request()).unwrap();
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend);

        let evidence = run_prepared_selected_simulation(&registry, &prepared).unwrap();
        assert_eq!(
            evidence.selected().assessment().proposal.id,
            "strict-selection-p0"
        );
        assert_eq!(evidence.validated().backend(), "selection-context-fixture");
        assert_eq!(evidence.validated().contexts()[0].digest, "a".repeat(64));
        assert_eq!(
            evidence.validated().contexts()[0].digest_algorithm,
            ContextDigestAlgorithm::Blake3
        );
    }
}
