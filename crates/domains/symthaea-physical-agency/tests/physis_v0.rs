// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHYSIS v0 — simulation-only physical-intelligence architecture benchmark.
//!
//! These tests exercise causal/evidence architecture, not artistic or physical
//! performance claims. No hardware or actuator path is present.

use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_physical_agency::portfolio::{
    CandidateAssessment, CandidatePortfolio, ModelEnsembleSummary, ModelPrediction,
    PortfolioOutcome, PortfolioPolicy,
};
use symthaea_physical_agency::qualification::{
    SimulationEvidenceBinding, execute_verified_simulation, qualify_simulation_candidate,
};
use symthaea_physical_agency::{
    BackendCapabilities, BackendCapability, BackendCapabilityManifest, CapabilityCatalog,
    CapabilityRequirement,
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
    energy_j: f64,
    information_gain: f64,
    safety_margin: f64,
    disagreement: f64,
) -> CandidateAssessment {
    CandidateAssessment {
        proposal,
        expected_energy_j: energy_j,
        expected_duration_ms: 100,
        information_gain,
        reversibility_score: 1.0,
        safety_margin,
        model_disagreement: disagreement,
    }
}

#[test]
fn physis_v0_preserves_cross_modal_tradeoffs_and_qualifies_exact_simulation_lineage() {
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

    let acoustic_models = ModelEnsembleSummary::from_predictions(&[
        ModelPrediction {
            model_id: "analytical-acoustics".into(),
            success_probability: 0.88,
        },
        ModelPrediction {
            model_id: "numerical-acoustics".into(),
            success_probability: 0.91,
        },
    ])
    .unwrap();
    let photonic_models = ModelEnsembleSummary::from_predictions(&[
        ModelPrediction {
            model_id: "geometric-optics".into(),
            success_probability: 0.94,
        },
        ModelPrediction {
            model_id: "numerical-optics".into(),
            success_probability: 0.90,
        },
    ])
    .unwrap();

    let acoustic = assessment(
        proposal("acoustic-p", PhysicalModality::Acoustic, 0.89, 0.10),
        2.0,
        0.75,
        0.95,
        acoustic_models.disagreement,
    );
    let photonic = assessment(
        proposal("photonic-p", PhysicalModality::Photonic, 0.93, 0.08),
        5.0,
        0.90,
        0.85,
        photonic_models.disagreement,
    );

    let portfolio = CandidatePortfolio {
        transition_id: "physis-t0".into(),
        candidates: vec![acoustic.clone(), photonic],
    };
    let frontier = match portfolio.evaluate(PortfolioPolicy::default()).unwrap() {
        PortfolioOutcome::ParetoFrontier(frontier) => frontier,
        other => panic!("PHYSIS v0 expected a Pareto frontier, got {other:?}"),
    };
    assert_eq!(frontier.len(), 2, "a real tradeoff must not be scalarized away");

    // A higher deliberative layer may select a frontier member for more
    // evidence. That selection still grants no physical execution authority.
    let selected = frontier
        .iter()
        .find(|candidate| candidate.proposal.id == "acoustic-p")
        .unwrap();

    let request = SimulationRequest::new(
        "physis-sim-0",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "verify selected simulation-only diagnostic candidate",
    );
    let mut registry = SimulationRegistry::new();
    registry.register(backend);
    let verified = execute_verified_simulation(&registry, &request).unwrap();

    let binding = SimulationEvidenceBinding {
        proposal_id: selected.proposal.id.clone(),
        simulation_request_id: request.id.clone(),
        expected_backend: "physis-solver".into(),
    };
    let mut safety_case = SafetyCase::new(&selected.proposal.id);
    safety_case.add_obligation(
        ProofObligation::new(
            "exact PHYSIS simulation evidence is attached",
            EvidenceKind::Simulation,
        )
        .discharge(verified.safety_evidence_ref()),
    );

    let qualified = qualify_simulation_candidate(
        &transition(),
        &selected.proposal,
        &binding,
        &verified,
        &safety_case,
    )
    .unwrap();

    assert_eq!(qualified.proposal_id(), "acoustic-p");
    assert_eq!(qualified.backend(), "physis-solver");
    assert_eq!(qualified.output_digest(), "physis-output-digest");
}

#[test]
fn physis_v0_model_disagreement_can_force_abstention() {
    let disputed = assessment(
        proposal("disputed-p", PhysicalModality::Acoustic, 0.9, 0.1),
        2.0,
        0.8,
        0.9,
        0.65,
    );
    let portfolio = CandidatePortfolio {
        transition_id: "physis-t0".into(),
        candidates: vec![disputed],
    };
    let policy = PortfolioPolicy {
        max_model_disagreement: 0.2,
        ..PortfolioPolicy::default()
    };

    assert_eq!(
        portfolio.evaluate(policy).unwrap(),
        PortfolioOutcome::Abstain(AbstentionReason::NoQualifiedAction)
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
