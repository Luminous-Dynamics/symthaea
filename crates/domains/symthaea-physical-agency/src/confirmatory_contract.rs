// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public strict confirmatory-simulation contract.
//!
//! PA-16 hardens the PA-13/14/15 primitives without widening physical authority:
//! every metric judged by a preregistered outcome claim must already be present
//! exactly once in the machine `SimulationRequest::requested_metrics`, and the
//! structural safety-evidence identity binds the complete canonical claim
//! definition rather than only `claim_id`.
//!
//! Lower-level PA-13/14/15 modules remain crate-private for regression coverage.
//! External callers therefore cannot bypass this facade to mint the weaker v2
//! strict qualification path.

use crate::deliberation::SelectedCandidate;
use crate::outcome_claim::{
    ConfirmatoryClaimOutcome, ConfirmatorySimulationEvidence, MetricPredicate, OutcomeClaimError,
    PreparedConfirmatorySimulation, SatisfiedSimulationClaim, SimulationOutcomeClaim,
};
use crate::safety_preregistration::{
    PreparedSafetyConfirmatorySimulation, SafetyPreregisteredConfirmatoryEvidence,
    SafetyPreregistrationError,
};
use crate::strict_context::StrictSimulationRegistry;
use crate::strict_qualification::StrictConfirmatorySimulationQualification;
use symthaea_formal_safety::{EvidenceKind, SafetyCase};
use symthaea_sim_bridge::SimulationRequest;
use thiserror::Error;

const CLAIM_TRANSCRIPT_DOMAIN: &[u8] = b"symthaea.physical-agency.outcome-claim.v1";

/// Exact deterministic machine representation of one complete outcome claim.
///
/// This is intentionally a runtime value rather than a serialized authority
/// token. A compact digest may be derived later for indexing, but the canonical
/// bytes remain the source of truth for equality/binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalClaimTranscript {
    bytes: Vec<u8>,
}

impl CanonicalClaimTranscript {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Strict simulation qualification additionally bound to the complete canonical
/// preregistered claim definition.
///
/// Still simulation-only. This type grants no HAL access, actuator capability,
/// execution permit, or physical authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimBoundConfirmatorySimulationQualification {
    inner: StrictConfirmatorySimulationQualification,
    claim_transcript: CanonicalClaimTranscript,
}

impl ClaimBoundConfirmatorySimulationQualification {
    pub fn selected(&self) -> &SelectedCandidate {
        self.inner.selected()
    }

    pub fn assessment(&self) -> &crate::portfolio::CandidateAssessment {
        self.inner.assessment()
    }

    pub fn world_snapshot(&self) -> &crate::deliberation::WorldSnapshotRef {
        self.inner.world_snapshot()
    }

    pub fn satisfied_claim(&self) -> &SatisfiedSimulationClaim {
        self.inner.satisfied_claim()
    }

    pub fn safety_case_id(&self) -> &str {
        self.inner.safety_case_id()
    }

    pub fn backend(&self) -> &str {
        self.inner.backend()
    }

    pub fn solver_version(&self) -> &str {
        self.inner.solver_version()
    }

    pub fn parser_version(&self) -> &str {
        self.inner.parser_version()
    }

    pub fn input_digest(&self) -> &str {
        self.inner.input_digest()
    }

    pub fn output_digest(&self) -> &str {
        self.inner.output_digest()
    }

    pub fn request_transcript(&self) -> &crate::strict_context::CanonicalRequestTranscript {
        self.inner.request_transcript()
    }

    pub fn contexts(&self) -> &[crate::strict_context::SimulationContextRef] {
        self.inner.contexts()
    }

    pub fn claim_transcript(&self) -> &CanonicalClaimTranscript {
        &self.claim_transcript
    }
}

/// Prepare a confirmatory run only when every claimed metric was explicitly
/// requested before execution.
///
/// `requested_metrics` is currently name-only in `symthaea-sim-bridge`, so the
/// exact unit/predicate/uncertainty semantics remain bound by the canonical claim
/// transcript while the metric name is also part of the canonical request
/// transcript.
pub fn prepare_confirmatory_simulation(
    selected: &SelectedCandidate,
    request: SimulationRequest,
    claim: SimulationOutcomeClaim,
) -> Result<PreparedConfirmatorySimulation, ConfirmatoryContractError> {
    validate_claim_request_binding(&request, &claim)?;
    let _ = canonical_claim_transcript(&claim)?;
    Ok(crate::outcome_claim::prepare_confirmatory_simulation(
        selected, request, claim,
    )?)
}

/// Freeze safety obligations after the claim/request binding has already been
/// established but before solver execution.
pub fn preregister_confirmatory_safety(
    prepared: PreparedConfirmatorySimulation,
    safety_case: &SafetyCase,
) -> Result<PreparedSafetyConfirmatorySimulation, ConfirmatoryContractError> {
    Ok(crate::safety_preregistration::preregister_confirmatory_safety(
        prepared,
        safety_case,
    )?)
}

/// Execute only a request that already passed strict claim/request and safety
/// preregistration.
pub fn run_preregistered_safety_confirmatory_simulation(
    registry: &StrictSimulationRegistry,
    prepared: &PreparedSafetyConfirmatorySimulation,
) -> Result<SafetyPreregisteredConfirmatoryEvidence, ConfirmatoryContractError> {
    Ok(
        crate::safety_preregistration::run_preregistered_safety_confirmatory_simulation(
            registry, prepared,
        )?,
    )
}

pub fn evaluate_confirmatory_claim(
    evidence: &ConfirmatorySimulationEvidence,
) -> Result<ConfirmatoryClaimOutcome, ConfirmatoryContractError> {
    Ok(crate::outcome_claim::evaluate_confirmatory_claim(evidence)?)
}

/// Claim-bound v3 evidence reference for the preregistered Simulation
/// obligation.
///
/// The PA-14 v2 reference already binds proposal/request/world/context and exact
/// canonical request bytes. v3 nests that exact lineage and additionally binds
/// the deterministic bytes of the *complete* claim definition. Reusing a
/// neighboring claim with the same `claim_id` therefore changes the reference.
pub fn required_confirmatory_safety_evidence_ref(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
) -> Result<String, ConfirmatoryContractError> {
    let v2 = crate::safety_preregistration::required_preregistered_safety_evidence_ref(
        evidence, satisfied,
    )?;
    let claim = canonical_claim_transcript(satisfied.claim())?;

    let mut reference = String::from("physical-agency-confirmatory:v3");
    push_field(&mut reference, &v2);
    let claim_hex = bytes_to_hex(claim.as_bytes());
    push_field(&mut reference, &claim_hex);
    Ok(reference)
}

/// Strictly qualify only when the completed preregistered safety case cites the
/// full-claim-bound v3 lineage.
///
/// Internally, PA-15's structural qualifier still checks its v2 lineage. The
/// facade derives that v2 reference itself after v3 has been established and
/// adds it only to an ephemeral clone of the completed case. Callers therefore
/// cannot use the older v2 label as the externally sufficient qualification
/// token.
pub fn qualify_confirmatory_simulation(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
    completed_safety_case: &SafetyCase,
) -> Result<ClaimBoundConfirmatorySimulationQualification, ConfirmatoryContractError> {
    let required_v3 = required_confirmatory_safety_evidence_ref(evidence, satisfied)?;
    let mut simulation_obligation_index = None;
    for (index, obligation) in completed_safety_case.obligations.iter().enumerate() {
        if obligation.expected_evidence == EvidenceKind::Simulation
            && obligation
                .evidence_refs
                .iter()
                .any(|reference| reference == &required_v3)
        {
            simulation_obligation_index = Some(index);
            break;
        }
    }
    let Some(index) = simulation_obligation_index else {
        return Err(ConfirmatoryContractError::MissingClaimBoundSafetyEvidence);
    };

    let required_v2 =
        crate::safety_preregistration::required_preregistered_safety_evidence_ref(
            evidence, satisfied,
        )?;
    let mut structural_case = completed_safety_case.clone();
    if !structural_case.obligations[index]
        .evidence_refs
        .iter()
        .any(|reference| reference == &required_v2)
    {
        structural_case.obligations[index]
            .evidence_refs
            .push(required_v2);
    }

    let inner =
        crate::safety_preregistration::qualify_preregistered_safety_confirmatory_simulation(
            evidence,
            satisfied,
            &structural_case,
        )?;
    let claim_transcript = canonical_claim_transcript(satisfied.claim())?;

    Ok(ClaimBoundConfirmatorySimulationQualification {
        inner,
        claim_transcript,
    })
}

pub fn canonical_claim_transcript(
    claim: &SimulationOutcomeClaim,
) -> Result<CanonicalClaimTranscript, ConfirmatoryContractError> {
    claim.validate()?;

    let mut bytes = Vec::new();
    push_bytes(&mut bytes, CLAIM_TRANSCRIPT_DOMAIN);
    bytes.extend_from_slice(&claim.schema_version.to_le_bytes());
    push_bytes(&mut bytes, claim.claim_id.as_bytes());
    push_bytes(&mut bytes, claim.transition_id.as_bytes());
    push_bytes(&mut bytes, claim.proposal_id.as_bytes());

    match claim.aggregation {
        crate::outcome_claim::ClaimAggregation::AllCriteria => bytes.push(0),
    }

    bytes.extend_from_slice(&(claim.criteria.len() as u64).to_le_bytes());
    for criterion in &claim.criteria {
        push_bytes(&mut bytes, criterion.metric_name.as_bytes());
        push_bytes(&mut bytes, criterion.unit.as_bytes());
        match &criterion.predicate {
            MetricPredicate::AtLeast(value) => {
                bytes.push(0);
                push_f64(&mut bytes, *value);
            }
            MetricPredicate::AtMost(value) => {
                bytes.push(1);
                push_f64(&mut bytes, *value);
            }
            MetricPredicate::InsideClosedInterval { lower, upper } => {
                bytes.push(2);
                push_f64(&mut bytes, *lower);
                push_f64(&mut bytes, *upper);
            }
            MetricPredicate::OutsideOpenInterval { lower, upper } => {
                bytes.push(3);
                push_f64(&mut bytes, *lower);
                push_f64(&mut bytes, *upper);
            }
        }
        bytes.push(match criterion.uncertainty_policy {
            crate::outcome_claim::MetricUncertaintyPolicy::RequireInterval => 0,
            crate::outcome_claim::MetricUncertaintyPolicy::AllowPointEstimate => 1,
        });
    }

    Ok(CanonicalClaimTranscript { bytes })
}

fn validate_claim_request_binding(
    request: &SimulationRequest,
    claim: &SimulationOutcomeClaim,
) -> Result<(), ConfirmatoryContractError> {
    claim.validate()?;
    for criterion in &claim.criteria {
        let count = request
            .requested_metrics
            .iter()
            .filter(|metric| *metric == &criterion.metric_name)
            .count();
        match count {
            1 => {}
            0 => {
                return Err(ConfirmatoryContractError::ClaimMetricNotRequested(
                    criterion.metric_name.clone(),
                ));
            }
            _ => {
                return Err(ConfirmatoryContractError::ClaimMetricRequestedMultipleTimes(
                    criterion.metric_name.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn push_field(reference: &mut String, field: &str) {
    reference.push('|');
    reference.push_str(&field.len().to_string());
    reference.push(':');
    reference.push_str(field);
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}

fn push_f64(output: &mut Vec<u8>, value: f64) {
    output.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Debug, Error)]
pub enum ConfirmatoryContractError {
    #[error("outcome-claim contract failed: {0}")]
    Outcome(#[from] OutcomeClaimError),
    #[error("safety preregistration/qualification failed: {0}")]
    Safety(#[from] SafetyPreregistrationError),
    #[error("claim metric {0:?} was not explicitly requested by the simulation request")]
    ClaimMetricNotRequested(String),
    #[error("claim metric {0:?} appears more than once in simulation requested_metrics")]
    ClaimMetricRequestedMultipleTimes(String),
    #[error("completed safety case does not cite the full-claim-bound v3 simulation lineage")]
    MissingClaimBoundSafetyEvidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deliberation::{
        DeliberationOutcome, SnapshotDigestAlgorithm, WorldSnapshotRef, deliberate,
    };
    use crate::outcome_claim::{
        MetricCriterion, MetricPredicate, MetricUncertaintyPolicy, SimulationOutcomeClaim,
    };
    use crate::portfolio::{
        CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
    };
    use symthaea_physical_effects::{
        AuthorityClass, DesiredTransition, EffectKind, MechanismRef, PhysicalModality,
        PredictedOutcome, ProposedIntervention, TargetRegion,
    };
    use symthaea_sim_bridge::{EngineeringDomain, SimulationRequest, SolverKind};

    fn selected() -> SelectedCandidate {
        let transition = DesiredTransition::simulation_only(
            "contract-t0",
            "claim/request binding fixture",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let assessment = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "contract-p0".into(),
                transition_id: "contract-t0".into(),
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
            candidates: vec![assessment],
        };
        let snapshot = WorldSnapshotRef::cryptographic(
            "world",
            SnapshotDigestAlgorithm::Blake3,
            "d".repeat(64),
        );
        let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };
        frontier.select("contract-p0").unwrap()
    }

    fn claim(threshold: f64) -> SimulationOutcomeClaim {
        SimulationOutcomeClaim::all_criteria(
            "same-claim-id",
            "contract-t0",
            "contract-p0",
            vec![MetricCriterion {
                metric_name: "diagnostic_quality".into(),
                unit: "1".into(),
                predicate: MetricPredicate::AtLeast(threshold),
                uncertainty_policy: MetricUncertaintyPolicy::RequireInterval,
            }],
        )
    }

    fn request(metrics: Vec<&str>) -> SimulationRequest {
        let mut request = SimulationRequest::new(
            "contract-run",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "claim/request binding fixture",
        );
        request.requested_metrics = metrics.into_iter().map(str::to_string).collect();
        request
    }

    #[test]
    fn claimed_metric_must_be_explicitly_requested() {
        assert!(matches!(
            prepare_confirmatory_simulation(&selected(), request(vec![]), claim(0.8)),
            Err(ConfirmatoryContractError::ClaimMetricNotRequested(metric))
                if metric == "diagnostic_quality"
        ));
    }

    #[test]
    fn duplicate_requested_claim_metric_is_rejected() {
        assert!(matches!(
            prepare_confirmatory_simulation(
                &selected(),
                request(vec!["diagnostic_quality", "diagnostic_quality"]),
                claim(0.8),
            ),
            Err(ConfirmatoryContractError::ClaimMetricRequestedMultipleTimes(metric))
                if metric == "diagnostic_quality"
        ));
    }

    #[test]
    fn same_claim_id_with_different_definition_has_different_canonical_identity() {
        let a = canonical_claim_transcript(&claim(0.8)).unwrap();
        let b = canonical_claim_transcript(&claim(0.9)).unwrap();
        assert_ne!(a, b);
    }
}
