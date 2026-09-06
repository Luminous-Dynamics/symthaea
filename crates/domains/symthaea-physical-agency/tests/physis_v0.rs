// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHYSIS v0 — simulation-only physical-intelligence architecture benchmark.
//!
//! These tests exercise causal/evidence architecture, not artistic or physical
//! performance claims. No hardware or actuator path is present.

use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_physical_agency::deliberation::{DeliberationOutcome, deliberate};
use symthaea_physical_agency::portfolio::{
    CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
};
use symthaea_physical_agency::{
    BackendCapabilities, BackendCapability, BackendCapabilityManifest, CapabilityCatalog,
    CapabilityRequirement, DeliberationQualificationError, SimulationEvidenceBinding,
    execute_registry_validated_simulation, qualify_selected_simulation_candidate,
};
use symthaea_physical_effects::{
    AbstentionReason, AuthorityClass, DesiredTransition, EffectKind, MechanismRef,
    PhysicalModality, PredictedOutcome, ProposedIntervention, TargetRegion,
};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, SimulationBackend, SimulationError, SimulationEvidence,
    SimulationRegistry, SimulationRequest, SimulationResult, SolverKind, UncertaintyEstimate,
};

#[derive(Debug)]
struct PhysisExternalBackend;

impl SimulationBackend for PhysisExternalBackend {
    fn name(&self) -> &'static str {
        "physis-solver"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Custom]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        Ok(SimulationResult::converged(&request.id, 0.95)
            .with_uncertainty(UncertaintyEstimate::new(0.05, 0.05))
            .with_metric("diagnostic_information", 0.88, "1")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some(self.name().into()),
                solver_version: Some("physis-v0-fixture".into()),
                input_digest: Some("physis-input-digest".into()),
                output_digest: Some("physis-output-digest".into()),
                parser_version: Some("physis-parser-v0".into()),
            }))
    }
}

fn transition() -> DesiredTransition {
    let mut transition = DesiredTransition::simulation_only(
        "physis-t0",
        "compare non-contact diagnostic mechanisms in simulation",
        TargetRegion::new("physis-world", "diagnostic-fixture"),
        EffectKind::Characterize,
        vec![PhysicalModality::Acoustic, PhysicalModality::Photonic],
    );
    transition.uncertainty.min_confidence = 0.8;
    transition.uncertainty.max_epistemic = 0.2;
    transition.uncertainty.max_aleatoric = 0.2;
    transition.resources.max_energy_j = Some(10.0);
    transition.resources.max_duration_ms = Some(500);
    transition
}

fn proposal(
    id: &str,
    modality: PhysicalModality,
    success: f64,
    epistemic: f64,
) -> ProposedIntervention {
    ProposedIntervention {
        id: id.into(),
        transition_id: "physis-t0".into(),
        mechanism: MechanismRef {
            backend: "physis-reference-model".into(),
            mechanism: format!("{id}-diagnostic"),
            modality,
        },
        required_authority: AuthorityClass::SimulationOnly,
        predicted_outcome: PredictedOutcome {
            success_probability: success,
            epistemic_uncertainty: epistemic,
            aleatoric_uncertainty: 0.05,
        },
    }
}

fn assessment(
    proposal: ProposedIntervention,
    model_predictions: Vec<ModelPrediction>,
    energy_j: f64,
    information_gain: f64,
    safety_margin: f64,
) -> CandidateAssessment {
    CandidateAssessment {
        proposal,
        model_predictions,
        expected_energy_j: energy_j,
        expected_power_w: None,
        expected_duration_ms: 100,
        information_gain,
        reversibility_score: 1.0,
        safety_margin,
    }
}

#[test]
fn physis_v0_preserves_deliberation_lineage_through_simulation_qualification() {
    let backend = PhysisExternalBackend;

    let mut capabilities = CapabilityCatalog::new();
    capabilities
        .register(BackendCapabilityManifest {
            backend_name: backend.name().into(),
            supported_solvers: vec![SolverKind::Custom],
            capabilities: BackendCapabilities {
                uncertainty_quantification: true,
                batched_counterfactuals: true,
                ..BackendCapabilities::default()
            },
            declaration_provenance: "PHYSIS v0 fixture declaration".into(),
        })
        .unwrap();

    let requirement = CapabilityRequirement::new(SolverKind::Custom)
        .requiring(BackendCapability::UncertaintyQuantification)
        .requiring(BackendCapability::BatchedCounterfactuals);
    assert!(capabilities
        .negotiate(&backend, &requirement)
        .unwrap()
        .is_accepted());

    let acoustic = assessment(
        proposal("acoustic-p", PhysicalModality::Acoustic, 0.89, 0.10),
        vec![
            ModelPrediction {
                model_id: "analytical-acoustics".into(),
                success_probability: 0.88,
            },
            ModelPrediction {
                model_id: "numerical-acoustics".into(),
                success_probability: 0.91,
            },
        ],
        2.0,
        0.75,
        0.95,
    );
    let photonic = assessment(
        proposal("photonic-p", PhysicalModality::Photonic, 0.93, 0.08),
        vec![
            ModelPrediction {
                model_id: "geometric-optics".into(),
                success_probability: 0.94,
            },
            ModelPrediction {
                model_id: "numerical-optics".into(),
                success_probability: 0.90,
            },
        ],
        5.0,
        0.90,
        0.85,
    );

    let portfolio = CandidatePortfolio {
        transition: transition(),
        candidates: vec![acoustic, photonic],
    };
    let policy = PortfolioPolicy {
        min_success_probability: 0.8,
        max_epistemic_uncertainty: 0.2,
        max_aleatoric_uncertainty: 0.2,
        max_model_disagreement: 0.2,
        min_safety_margin: 0.8,
    };
    let frontier = match deliberate(&portfolio, policy).unwrap() {
        DeliberationOutcome::ParetoFrontier(frontier) => frontier,
        other => panic!("PHYSIS v0 expected a Pareto frontier, got {other:?}"),
    };
    assert_eq!(
        frontier.candidates().len(),
        2,
        "a real tradeoff must not be scalarized away"
    );

    // Selection mints a non-serializable receipt only for a candidate that
    // actually survived the evaluated frontier.
    let selected = frontier.select("acoustic-p").unwrap();
    assert!(frontier.select("not-on-frontier").is_none());
    assert!(selected.assessment().model_disagreement().unwrap() <= 0.2);
    assert!(
        selected
            .assessment()
            .effective_epistemic_uncertainty()
            .unwrap()
            <= 0.2
    );

    let request = SimulationRequest::new(
        "physis-sim-0",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "validate selected simulation-only diagnostic candidate",
    );
    let mut registry = SimulationRegistry::new();
    registry.register(backend);
    let validated = execute_registry_validated_simulation(&registry, &request).unwrap();

    let binding = SimulationEvidenceBinding {
        proposal_id: selected.assessment().proposal.id.clone(),
        simulation_request_id: request.id.clone(),
        expected_backend: "physis-solver".into(),
    };
    let mut safety_case = SafetyCase::new(&selected.assessment().proposal.id);
    safety_case.add_obligation(
        ProofObligation::new(
            "exact PHYSIS simulation evidence is attached",
            EvidenceKind::Simulation,
        )
        .discharge(validated.safety_evidence_ref()),
    );

    let qualified = qualify_selected_simulation_candidate(
        &selected,
        &binding,
        &validated,
        &safety_case,
    )
    .unwrap();

    assert_eq!(qualified.proposal_id(), "acoustic-p");
    assert_eq!(qualified.backend(), "physis-solver");
    assert_eq!(qualified.output_digest(), "physis-output-digest");
    assert_eq!(qualified.assessment().model_predictions.len(), 2);
    assert_eq!(qualified.selection_policy(), policy);
}

#[test]
fn physis_v0_model_disagreement_can_force_abstention() {
    let disputed = assessment(
        proposal("disputed-p", PhysicalModality::Acoustic, 0.9, 0.1),
        vec![
            ModelPrediction {
                model_id: "disputed-model-a".into(),
                success_probability: 0.9,
            },
            ModelPrediction {
                model_id: "disputed-model-b".into(),
                success_probability: 0.25,
            },
        ],
        2.0,
        0.8,
        0.9,
    );
    let portfolio = CandidatePortfolio {
        transition: transition(),
        candidates: vec![disputed],
    };
    let policy = PortfolioPolicy {
        max_model_disagreement: 0.2,
        ..PortfolioPolicy::default()
    };

    assert_eq!(
        deliberate(&portfolio, policy).unwrap(),
        DeliberationOutcome::Abstain(AbstentionReason::NoQualifiedAction)
    );
}

#[test]
fn physis_v0_unknown_simulator_capability_fails_closed() {
    let backend = PhysisExternalBackend;
    let catalog = CapabilityCatalog::new();
    let requirement = CapabilityRequirement::new(SolverKind::Custom)
        .requiring(BackendCapability::UncertaintyQuantification);

    assert!(catalog.negotiate(&backend, &requirement).is_err());
}

#[test]
fn physis_v0_transition_contract_rejects_out_of_envelope_candidate() {
    let over_budget = assessment(
        proposal("over-budget", PhysicalModality::Acoustic, 0.9, 0.1),
        vec![
            ModelPrediction {
                model_id: "budget-model-a".into(),
                success_probability: 0.9,
            },
            ModelPrediction {
                model_id: "budget-model-b".into(),
                success_probability: 0.88,
            },
        ],
        20.0,
        0.7,
        0.9,
    );

    let portfolio = CandidatePortfolio {
        transition: transition(),
        candidates: vec![over_budget],
    };
    assert!(deliberate(&portfolio, PortfolioPolicy::default()).is_err());
}

#[test]
fn physis_v0_transition_uncertainty_gate_overrides_permissive_selection_policy() {
    let mut strict_transition = transition();
    strict_transition.uncertainty.min_confidence = 0.7;
    strict_transition.uncertainty.max_epistemic = 0.2;

    let disputed = assessment(
        proposal("policy-permitted-dispute", PhysicalModality::Acoustic, 0.85, 0.1),
        vec![
            ModelPrediction {
                model_id: "policy-model-a".into(),
                success_probability: 0.95,
            },
            ModelPrediction {
                model_id: "policy-model-b".into(),
                success_probability: 0.70,
            },
        ],
        2.0,
        0.7,
        0.9,
    );
    let portfolio = CandidatePortfolio {
        transition: strict_transition,
        candidates: vec![disputed],
    };

    // Deliberately permissive policy: selection is allowed to proceed so the
    // transition-level gate itself is exercised at qualification.
    let frontier = match deliberate(
        &portfolio,
        PortfolioPolicy {
            min_success_probability: 0.0,
            max_epistemic_uncertainty: 1.0,
            max_aleatoric_uncertainty: 1.0,
            max_model_disagreement: 1.0,
            min_safety_margin: 0.0,
        },
    )
    .unwrap()
    {
        DeliberationOutcome::ParetoFrontier(frontier) => frontier,
        other => panic!("expected deliberately permissive frontier, got {other:?}"),
    };
    let selected = frontier.select("policy-permitted-dispute").unwrap();
    assert!(selected.assessment().model_disagreement().unwrap() > 0.2);

    let request = SimulationRequest::new(
        "physis-sim-policy-gate",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "exercise transition uncertainty gate",
    );
    let mut registry = SimulationRegistry::new();
    registry.register(PhysisExternalBackend);
    let validated = execute_registry_validated_simulation(&registry, &request).unwrap();
    let binding = SimulationEvidenceBinding {
        proposal_id: selected.assessment().proposal.id.clone(),
        simulation_request_id: request.id.clone(),
        expected_backend: "physis-solver".into(),
    };
    let mut safety_case = SafetyCase::new(&selected.assessment().proposal.id);
    safety_case.add_obligation(
        ProofObligation::new("exact evidence", EvidenceKind::Simulation)
            .discharge(validated.safety_evidence_ref()),
    );

    assert!(matches!(
        qualify_selected_simulation_candidate(&selected, &binding, &validated, &safety_case),
        Err(DeliberationQualificationError::EnsembleUncertaintyOutsideTransitionBudget { .. })
    ));
}
