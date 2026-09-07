// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHYSIS v1 — strict simulation-evidence architecture benchmark.
//!
//! This suite proves evidence-flow properties only. It does not claim physical
//! performance, authenticate arbitrary evidence references, or grant hardware
//! execution authority.

use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation, SafetyCase};
use symthaea_physical_agency::deliberation::{
    DeliberationOutcome, SnapshotDigestAlgorithm, WorldSnapshotRef, deliberate,
};
use symthaea_physical_agency::outcome_claim::{
    ConfirmatoryClaimOutcome, MetricCriterion, MetricPredicate, MetricUncertaintyPolicy,
    SimulationOutcomeClaim, evaluate_confirmatory_claim, prepare_confirmatory_simulation,
};
use symthaea_physical_agency::portfolio::{
    CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
};
use symthaea_physical_agency::safety_preregistration::{
    SafetyPreregistrationError, preregister_confirmatory_safety,
    qualify_preregistered_safety_confirmatory_simulation,
    required_preregistered_safety_evidence_ref,
    run_preregistered_safety_confirmatory_simulation,
};
use symthaea_physical_agency::strict_context::{
    ContextAwareSimulationBackend, ContextBoundSimulationRequest, ContextBoundSimulationResult,
    ContextConsumptionEvidence, StrictSimulationRegistry,
};
use symthaea_physical_effects::{
    AuthorityClass, DesiredTransition, EffectKind, MechanismRef, PhysicalModality,
    PredictedOutcome, ProposedIntervention, TargetRegion,
};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, Interval, SimulationEvidence, SimulationError,
    SimulationMetric, SimulationRequest, SimulationResult, SolverKind, UncertaintyEstimate,
};

#[derive(Debug)]
struct PhysisV1Backend {
    interval: Interval,
}

impl ContextAwareSimulationBackend for PhysisV1Backend {
    fn name(&self) -> &'static str {
        "physis-v1-context-solver"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Custom]
    }

    fn run_context_bound(
        &self,
        request: &ContextBoundSimulationRequest,
    ) -> Result<ContextBoundSimulationResult, SimulationError> {
        let mut result = SimulationResult::converged(&request.request.id, 0.98)
            .with_uncertainty(UncertaintyEstimate::new(0.03, 0.02))
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some(self.name().into()),
                solver_version: Some("physis-v1-fixture".into()),
                input_digest: Some(format!("physis-v1-input:{}", request.request.id)),
                output_digest: Some(format!("physis-v1-output:{}", request.request.id)),
                parser_version: Some("physis-v1-parser".into()),
            });

        result.metrics = vec![SimulationMetric {
            name: "diagnostic_quality".into(),
            value: self.interval.midpoint(),
            unit: "1".into(),
            uncertainty: Some(
                UncertaintyEstimate::new(0.03, 0.02).with_interval(self.interval),
            ),
        }];

        Ok(ContextBoundSimulationResult {
            result,
            consumption: ContextConsumptionEvidence {
                request_transcript: request
                    .canonical_transcript()
                    .map_err(|error| SimulationError::Adapter(error.to_string()))?,
                consumed_contexts: request.contexts.clone(),
            },
        })
    }
}

fn transition() -> DesiredTransition {
    DesiredTransition::simulation_only(
        "physis-v1-transition",
        "characterize an uncertain target through strict simulation evidence",
        TargetRegion::new("physis-v1-world", "diagnostic-target"),
        EffectKind::Characterize,
        vec![PhysicalModality::Acoustic],
    )
}

fn assessment() -> CandidateAssessment {
    CandidateAssessment {
        proposal: ProposedIntervention {
            id: "physis-v1-acoustic".into(),
            transition_id: "physis-v1-transition".into(),
            mechanism: MechanismRef {
                backend: "physis-v1-model".into(),
                mechanism: "acoustic-diagnostic-simulation".into(),
                modality: PhysicalModality::Acoustic,
            },
            required_authority: AuthorityClass::SimulationOnly,
            predicted_outcome: PredictedOutcome {
                success_probability: 0.91,
                epistemic_uncertainty: 0.06,
                aleatoric_uncertainty: 0.03,
            },
        },
        model_predictions: vec![
            ModelPrediction {
                model_id: "analytical-acoustic-model".into(),
                success_probability: 0.91,
            },
            ModelPrediction {
                model_id: "numerical-acoustic-model".into(),
                success_probability: 0.88,
            },
        ],
        expected_energy_j: 0.5,
        expected_power_w: None,
        expected_duration_ms: 50,
        information_gain: 0.85,
        reversibility_score: 1.0,
        safety_margin: 0.97,
    }
}

fn selected(snapshot: WorldSnapshotRef) -> symthaea_physical_agency::deliberation::SelectedCandidate {
    let portfolio = CandidatePortfolio {
        transition: transition(),
        candidates: vec![assessment()],
    };
    let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
        DeliberationOutcome::ParetoFrontier(frontier) => frontier,
        other => panic!("expected Pareto frontier, got {other:?}"),
    };
    frontier.select("physis-v1-acoustic").unwrap()
}

fn cryptographic_selected() -> symthaea_physical_agency::deliberation::SelectedCandidate {
    selected(WorldSnapshotRef::cryptographic(
        "physis-v1-world",
        SnapshotDigestAlgorithm::Blake3,
        "c".repeat(64),
    ))
}

fn claim() -> SimulationOutcomeClaim {
    SimulationOutcomeClaim::all_criteria(
        "physis-v1-claim",
        "physis-v1-transition",
        "physis-v1-acoustic",
        vec![MetricCriterion {
            metric_name: "diagnostic_quality".into(),
            unit: "1".into(),
            predicate: MetricPredicate::AtLeast(0.8),
            uncertainty_policy: MetricUncertaintyPolicy::RequireInterval,
        }],
    )
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "PHYSIS v1 strict diagnostic benchmark",
    )
}

fn open_safety_case() -> SafetyCase {
    let mut safety = SafetyCase::new("physis-v1-acoustic");
    safety.add_obligation(ProofObligation::new(
        "exact confirmatory simulation satisfies the preregistered diagnostic claim",
        EvidenceKind::Simulation,
    ));
    safety.add_obligation(ProofObligation::new(
        "independent invariant review confirms the benchmark safety assumptions",
        EvidenceKind::FormalProof,
    ));
    safety
}

#[test]
fn physis_v1_exercises_the_complete_strict_simulation_chain() {
    let selected = cryptographic_selected();
    let prepared = prepare_confirmatory_simulation(
        &selected,
        request("physis-v1-run"),
        claim(),
    )
    .unwrap();

    let safety = open_safety_case();
    let mut completed = safety.clone();
    let prepared = preregister_confirmatory_safety(prepared, &safety).unwrap();

    let mut registry = StrictSimulationRegistry::new();
    registry.register(PhysisV1Backend {
        interval: Interval::new(0.86, 0.94),
    });
    let evidence = run_preregistered_safety_confirmatory_simulation(&registry, &prepared).unwrap();

    let satisfied = match evaluate_confirmatory_claim(evidence.confirmatory()).unwrap() {
        ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
        other => panic!("expected interval-backed satisfied claim, got {other:?}"),
    };

    let exact = required_preregistered_safety_evidence_ref(&evidence, &satisfied).unwrap();
    for obligation in &mut completed.obligations {
        obligation.status = ObligationStatus::Discharged;
        obligation.evidence_refs.push(if obligation.expected_evidence == EvidenceKind::Simulation {
            exact.clone()
        } else {
            "formal-proof:physis-v1-independent-invariant-review".into()
        });
    }

    let qualified = qualify_preregistered_safety_confirmatory_simulation(
        &evidence,
        &satisfied,
        &completed,
    )
    .unwrap();

    assert_eq!(qualified.assessment().proposal.id, "physis-v1-acoustic");
    assert_eq!(qualified.backend(), "physis-v1-context-solver");
    assert_eq!(
        qualified.world_snapshot().digest_algorithm(),
        SnapshotDigestAlgorithm::Blake3
    );
    assert_eq!(qualified.world_snapshot().snapshot_digest(), "c".repeat(64));
    assert_eq!(qualified.contexts().len(), 1);
}

#[test]
fn legacy_snapshot_cannot_enter_the_strict_confirmatory_path() {
    let selected = selected(WorldSnapshotRef::new(
        "physis-v1-world",
        "legacy-opaque-snapshot-id",
    ));

    assert!(prepare_confirmatory_simulation(
        &selected,
        request("physis-v1-legacy-run"),
        claim(),
    )
    .is_err());
}

#[test]
fn simulation_only_safety_argument_is_rejected_before_solver_execution() {
    let selected = cryptographic_selected();
    let prepared = prepare_confirmatory_simulation(
        &selected,
        request("physis-v1-sim-only-safety"),
        claim(),
    )
    .unwrap();

    let mut safety = SafetyCase::new("physis-v1-acoustic");
    safety.add_obligation(ProofObligation::new(
        "simulation alone is the entire safety argument",
        EvidenceKind::Simulation,
    ));

    assert!(matches!(
        preregister_confirmatory_safety(prepared, &safety),
        Err(SafetyPreregistrationError::MissingIndependentObligation)
    ));
}

#[test]
fn threshold_straddling_interval_is_indeterminate() {
    let selected = cryptographic_selected();
    let prepared = prepare_confirmatory_simulation(
        &selected,
        request("physis-v1-straddle"),
        claim(),
    )
    .unwrap();
    let safety = open_safety_case();
    let prepared = preregister_confirmatory_safety(prepared, &safety).unwrap();

    let mut registry = StrictSimulationRegistry::new();
    registry.register(PhysisV1Backend {
        interval: Interval::new(0.75, 0.9),
    });
    let evidence = run_preregistered_safety_confirmatory_simulation(&registry, &prepared).unwrap();

    assert!(matches!(
        evaluate_confirmatory_claim(evidence.confirmatory()).unwrap(),
        ConfirmatoryClaimOutcome::Indeterminate(_)
    ));
}

#[test]
fn posthoc_replacement_safety_case_cannot_qualify_the_run() {
    let selected = cryptographic_selected();
    let prepared = prepare_confirmatory_simulation(
        &selected,
        request("physis-v1-posthoc"),
        claim(),
    )
    .unwrap();
    let safety = open_safety_case();
    let prepared = preregister_confirmatory_safety(prepared, &safety).unwrap();

    let mut registry = StrictSimulationRegistry::new();
    registry.register(PhysisV1Backend {
        interval: Interval::new(0.86, 0.94),
    });
    let evidence = run_preregistered_safety_confirmatory_simulation(&registry, &prepared).unwrap();
    let satisfied = match evaluate_confirmatory_claim(evidence.confirmatory()).unwrap() {
        ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
        other => panic!("expected satisfied claim, got {other:?}"),
    };
    let exact = required_preregistered_safety_evidence_ref(&evidence, &satisfied).unwrap();

    let mut posthoc = SafetyCase::new("physis-v1-acoustic");
    posthoc.add_obligation(
        ProofObligation::new("invented simulation obligation", EvidenceKind::Simulation)
            .discharge(exact),
    );
    posthoc.add_obligation(
        ProofObligation::new("invented independent obligation", EvidenceKind::FormalProof)
            .discharge("formal-proof:invented-after-run"),
    );

    assert!(matches!(
        qualify_preregistered_safety_confirmatory_simulation(
            &evidence,
            &satisfied,
            &posthoc,
        ),
        Err(SafetyPreregistrationError::SafetyCaseIdMismatch { .. })
    ));
}
