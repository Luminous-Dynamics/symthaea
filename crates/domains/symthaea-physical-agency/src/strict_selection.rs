// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selection-bound strict simulation preparation and evidence.
//!
//! PA-12 removes the caller-written world-snapshot digest from the strict run
//! path. The world context is derived directly from the non-serializable
//! [`SelectedCandidate`] minted by deliberation. The prepared request and the
//! returned evidence receipt are also non-serializable.
//!
//! This is still simulation-only structural evidence. It grants no physical
//! execution authority and does not authenticate a malicious solver backend.

use crate::deliberation::SelectedCandidate;
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
/// The caller chooses only the declared digest algorithm. The frame and digest
/// bytes themselves come exclusively from `SelectedCandidate::world_snapshot`.
/// A legacy/non-cryptographic snapshot identifier therefore fails closed when it
/// attempts to enter the strict context path.
pub fn prepare_selected_simulation(
    selected: &SelectedCandidate,
    request: SimulationRequest,
    digest_algorithm: ContextDigestAlgorithm,
) -> Result<PreparedSelectedSimulation, SelectionBoundSimulationError> {
    let snapshot = selected.world_snapshot();
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
    #[error("strict simulation evidence contains {0} world contexts; exactly one is required")]
    WorldContextCount(usize),
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

    fn selected(snapshot_digest: String) -> SelectedCandidate {
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
        let snapshot = WorldSnapshotRef::new("world", snapshot_digest);
        let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };
        frontier.select("strict-selection-p0").unwrap()
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
        let selected = selected("a".repeat(64));
        let prepared = prepare_selected_simulation(
            &selected,
            request(),
            ContextDigestAlgorithm::Blake3,
        )
        .unwrap();

        assert_eq!(prepared.selected(), &selected);
        assert_eq!(prepared.request().contexts.len(), 1);
        let context = &prepared.request().contexts[0];
        assert_eq!(context.frame_id.as_deref(), Some("world"));
        assert_eq!(context.digest, "a".repeat(64));
        assert_eq!(context.context_id, "world-snapshot:world");
    }

    #[test]
    fn legacy_non_cryptographic_snapshot_identifier_cannot_enter_strict_path() {
        let selected = selected("legacy-snapshot-name".into());
        assert!(matches!(
            prepare_selected_simulation(&selected, request(), ContextDigestAlgorithm::Blake3),
            Err(SelectionBoundSimulationError::Strict(
                StrictSimulationError::InvalidContext(_)
            ))
        ));
    }

    #[test]
    fn strict_run_retains_exact_selected_lineage() {
        let selected = selected("a".repeat(64));
        let prepared = prepare_selected_simulation(
            &selected,
            request(),
            ContextDigestAlgorithm::Blake3,
        )
        .unwrap();
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend);

        let evidence = run_prepared_selected_simulation(&registry, &prepared).unwrap();
        assert_eq!(
            evidence.selected().assessment().proposal.id,
            "strict-selection-p0"
        );
        assert_eq!(evidence.validated().backend(), "selection-context-fixture");
        assert_eq!(evidence.validated().contexts()[0].digest, "a".repeat(64));
    }
}
