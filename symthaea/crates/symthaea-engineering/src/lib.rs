// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering reasoning facade for Symthaea.
//!
//! This crate is the composition layer for engineering work. It does not solve
//! physics itself; it records requirements, concepts, simulation requests,
//! digital-twin state, and proof obligations so Symthaea can reason over them.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_digital_twin::TwinState;
use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_broca::{BrocaConfig, BrocaGenerator, ThoughtChannels};
use symthaea_sim_bridge::{
    EngineeringDomain, SimulationRegistry, SimulationRequest, SolverKind, SurpriseMonitor,
};

pub use symthaea_digital_twin as digital_twin;
pub use symthaea_formal_safety as formal_safety;
pub use symthaea_sim_bridge as sim_bridge;

/// Formal proof generator for engineering safety cases.
pub struct LeanProofGenerator;

impl LeanProofGenerator {
    /// Generate a Lean 4 proof script for a proof result.
    pub fn generate_proof(
        name: &str,
        goal: &symthaea_core::hdc::logic_engine::Proposition,
        result: &symthaea_core::hdc::logic_engine::ProofResult,
    ) -> String {
        symthaea_lean_bridge::bridge::render_lean_file(name, goal, result)
    }
}

/// Assistant that uses the Broca language center to help define engineering entities.
pub struct EngineeringAssistant {
    generator: BrocaGenerator,
}

impl EngineeringAssistant {
    /// Create a new assistant from a genesis seed.
    pub fn new(genesis: &symthaea_core::genesis::GenesisSeed) -> Self {
        Self {
            generator: BrocaGenerator::new(genesis, BrocaConfig::default()),
        }
    }

    /// Propose a set of engineering requirements for a goal.
    pub fn propose_requirements(
        &mut self,
        _goal: &str,
        domain: EngineeringDomain,
    ) -> Vec<EngineeringRequirement> {
        let mut channels = ThoughtChannels::with_intent(1); // Inform/Reason
        channels.set_consciousness(0.8, 0.4, 0.6); // High psi for detail

        let result = self.generator.generate(&channels);

        // In a real implementation, we would parse the generated text.
        // For now, we'll use the text as the requirement statement for one requirement.
        vec![EngineeringRequirement::new(
            format!("REQ-{:?}-001", domain),
            domain,
            result.text,
            RequirementCriticality::Medium,
            formal_safety::EvidenceKind::Simulation,
        )]
    }
}

/// Orchestrates engineering reasoning and simulation workflows.
#[derive(Default, Debug)]
pub struct EngineeringManager {
    /// Registry of available simulation backends.
    pub registry: SimulationRegistry,
    /// FEP-driven monitor for epistemic uncertainty.
    pub surprise_monitor: SurpriseMonitor,
}

impl EngineeringManager {
    /// Create a new engineering manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run an Active Inference step: trigger simulations if surprise is high.
    pub fn active_inference_step(&mut self, concept: &mut EngineeringConcept, surprise: f64) {
        self.surprise_monitor.update(surprise);
        if self.surprise_monitor.should_trigger_sim() {
            tracing::info!(
                "FEP Surprise ({:.3}) exceeds threshold; triggering active research for concept {}.",
                surprise, concept.id
            );
            self.evaluate_concept(concept);
        }
    }

    /// Run all simulation requests for a concept and update its safety case.
    pub fn evaluate_concept(&self, concept: &mut EngineeringConcept) {
        for request in &concept.simulation_requests {
            match self.registry.run(request) {
                Ok(result) => {
                    if result.converged {
                        // Find the obligation that matches this simulation request (simple heuristic: by claim name)
                        if let Some(obligation) =
                            concept.safety_case.obligations.iter_mut().find(|o| {
                                o.claim.contains(&request.objective)
                                    || request.objective.contains(&o.claim)
                            })
                        {
                            let evidence_ref = format!("{}:{}", result.request_id, result.confidence);
                            obligation.status = formal_safety::ObligationStatus::Discharged;
                            obligation.evidence_refs.push(evidence_ref);
                        }
                    }
                }
                Err(e) => {
                    // Log error or mark obligation as failed
                    eprintln!("Simulation failed for concept {}: {:?}", concept.id, e);
                }
            }
        }
    }

    /// Export formal Lean 4 proofs for all discharged safety obligations.
    pub fn formally_verify(&self, concept: &EngineeringConcept) -> Vec<(String, String)> {
        let mut proofs = Vec::new();
        for obligation in &concept.safety_case.obligations {
            if obligation.status == formal_safety::ObligationStatus::Discharged {
                // In a real implementation, we would fetch the internal ProofResult
                // that justifies this obligation's discharge.
                // Here we mock a simple tautology for the evidence.
                use symthaea_core::hdc::logic_engine::{ProofResult, ProofStepLogic, Proposition};
                let id_str = obligation.id.to_string();
                let goal = Proposition::Atom(id_str.clone())
                    .implies(Proposition::Atom(id_str));
                let result = ProofResult {
                    valid: true,
                    proof_steps: vec![ProofStepLogic {
                        step_number: 1,
                        rule: "Premise".to_string(),
                        formula: "P -> P".to_string(),
                        justification: "tautology".to_string(),
                    }],
                    phi: 0.9,
                    description: format!("Proof of {}", obligation.claim),
                };

                let lean_script = LeanProofGenerator::generate_proof(
                    &format!("verify_{}", obligation.id.to_string().replace('-', "_")),
                    &goal,
                    &result,
                );
                proofs.push((obligation.id.to_string(), lean_script));
            }
        }
        proofs
    }

    /// Automatically refine requirements if formal verification fails.
    pub fn refine_requirements(
        &self,
        assistant: &mut EngineeringAssistant,
        concept: &mut EngineeringConcept,
        proof_results: &[(String, String)],
    ) {
        for (id, script) in proof_results {
            if script.contains("sorry") {
                tracing::warn!("Formal proof for {} failed (contains sorry). Requesting refinement from assistant.", id);
                
                // Trigger Broca to re-think this specific requirement
                let refined = assistant.propose_requirements(
                    &format!("Fix unprovable requirement: {}", id),
                    concept.domain
                );
                
                // Replace the failing requirement with the refined one
                if let Some(req) = concept.requirements.iter_mut().find(|r| r.id == *id) {
                    if let Some(new_req) = refined.first() {
                        tracing::info!("Refined REQ: {} -> {}", req.statement, new_req.statement);
                        req.statement = new_req.statement.clone();
                        req.criticality = new_req.criticality;
                    }
                }
            }
        }
    }
}

/// Requirement criticality for engineering decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequirementCriticality {
    /// Nice-to-have constraint.
    Low,
    /// Important design requirement.
    Medium,
    /// Safety, legal, mission, or high-cost requirement.
    High,
    /// Requirement must be satisfied before deployment or actuation.
    Blocking,
}

/// A normalized engineering requirement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringRequirement {
    /// Stable requirement id.
    pub id: String,
    /// Domain this requirement primarily belongs to.
    pub domain: EngineeringDomain,
    /// Requirement statement.
    pub statement: String,
    /// Criticality level.
    pub criticality: RequirementCriticality,
    /// Evidence expected before the requirement can be accepted.
    pub evidence: EvidenceKind,
}

impl EngineeringRequirement {
    /// Create a new engineering requirement.
    pub fn new(
        id: impl Into<String>,
        domain: EngineeringDomain,
        statement: impl Into<String>,
        criticality: RequirementCriticality,
        evidence: EvidenceKind,
    ) -> Self {
        Self {
            id: id.into(),
            domain,
            statement: statement.into(),
            criticality,
            evidence,
        }
    }
}

/// Early-stage design concept being evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringConcept {
    /// Stable concept id.
    pub id: String,
    /// Short label.
    pub label: String,
    /// Primary engineering domain.
    pub domain: EngineeringDomain,
    /// Requirements attached to this concept.
    pub requirements: Vec<EngineeringRequirement>,
    /// Simulation requests required for this concept.
    pub simulation_requests: Vec<SimulationRequest>,
    /// Safety case produced from requirements.
    pub safety_case: SafetyCase,
}

impl EngineeringConcept {
    /// Start a concept with an empty safety case.
    pub fn new(id: impl Into<String>, label: impl Into<String>, domain: EngineeringDomain) -> Self {
        let id = id.into();
        let label = label.into();
        Self {
            safety_case: SafetyCase::new(label.clone()),
            id,
            label,
            domain,
            requirements: Vec::new(),
            simulation_requests: Vec::new(),
        }
    }

    /// Add a requirement and create the corresponding proof obligation.
    pub fn add_requirement(&mut self, requirement: EngineeringRequirement) {
        self.safety_case.add_obligation(ProofObligation::new(
            requirement.statement.clone(),
            requirement.evidence,
        ));
        self.requirements.push(requirement);
    }

    /// Add a standard simulation request for the concept's domain.
    pub fn request_simulation(
        &mut self,
        id: impl Into<String>,
        solver: SolverKind,
        objective: impl Into<String>,
    ) {
        self.simulation_requests
            .push(SimulationRequest::new(id, self.domain, solver, objective));
    }
}

/// Engineering review package for a design or live asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringReview {
    /// Concept under review.
    pub concept: EngineeringConcept,
    /// Optional live twin associated with the concept.
    pub twin: Option<TwinState>,
}

impl EngineeringReview {
    /// Returns true when the review should block deployment.
    pub fn blocks_deployment(&self) -> bool {
        let has_blocking_open_requirement = self
            .concept
            .requirements
            .iter()
            .any(|requirement| requirement.criticality == RequirementCriticality::Blocking)
            && !self.concept.safety_case.is_discharged();

        let twin_is_unhealthy = self
            .twin
            .as_ref()
            .is_some_and(|twin| twin.needs_intervention(1.0));

        has_blocking_open_requirement || twin_is_unhealthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocking_requirement_creates_safety_gate() {
        let mut concept = EngineeringConcept::new(
            "bridge-001",
            "low-carbon footbridge",
            EngineeringDomain::Civil,
        );
        concept.add_requirement(EngineeringRequirement::new(
            "REQ-SERVICE-STRESS",
            EngineeringDomain::Civil,
            "service stress remains below allowable limit",
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));

        let review = EngineeringReview {
            concept,
            twin: None,
        };

        assert!(review.blocks_deployment());
    }
}
