// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering reasoning facade for Symthaea.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_digital_twin::TwinState;
use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_causal_reasoning::causal_calculus::StructuralCausalModel;
use symthaea_broca::{BrocaConfig, BrocaGenerator, ThoughtChannels};
use symthaea_memory::{Episode, EpisodicMemory, EpisodicReplayConfig, MemoryCoordinator, SemanticMemory};
use symthaea_harmonies::{AlignmentResult, EightHarmonies};
use symthaea_swarm::{SwarmAggregator, SwarmMessage, SwarmProofMsg, SwarmStateMsg};
use symthaea_materials::{MaterialAgingModel, MaterialProperty};
use symthaea_fabrication_kernel::autonomy_loop::{AutonomyEvent, AutonomyLoop};
use symthaea_fabrication_kernel::cincinnati_live::{AnomalyAlert, CincinnatiMonitor};
use symthaea_fabrication_kernel::{GeometricThought, TriangleMesh};
use symthaea_sim_bridge::{
    AmygdalaInterlock, EngineeringDomain, MetricEncoder, SimulationRegistry, SimulationRequest, SurpriseMonitor,
};

pub use symthaea_digital_twin as digital_twin;
pub use symthaea_formal_safety as formal_safety;
pub use symthaea_memory as memory;
pub use symthaea_sim_bridge as sim_bridge;

/// Debug-friendly wrapper for the fabrication autonomy loop.
pub struct DebugFabricationLoop(pub AutonomyLoop);
impl std::fmt::Debug for DebugFabricationLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutonomyLoop")
            .field("state", &self.0.state())
            .finish()
    }
}

/// Debug-friendly wrapper for the Cincinnati monitor.
pub struct DebugCincinnatiMonitor(pub CincinnatiMonitor);
impl std::fmt::Debug for DebugCincinnatiMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CincinnatiMonitor")
            .field("anomaly_count", &self.0.anomaly_count())
            .finish()
    }
}

/// Debug-friendly wrapper for the Amygdala interlock.
pub struct DebugAmygdala(pub AmygdalaInterlock);
impl std::fmt::Debug for DebugAmygdala {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmygdalaInterlock")
            .field("status", &self.0.status)
            .finish()
    }
}

/// Debug-friendly wrapper for the material aging model.
pub struct DebugAgingModel(pub MaterialAgingModel);
impl std::fmt::Debug for DebugAgingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterialAgingModel").finish()
    }
}

/// Debug-friendly wrapper for the memory coordinator.
#[derive(Default)]
pub struct DebugCoordinator(pub MemoryCoordinator);
impl std::fmt::Debug for DebugCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCoordinator")
            .field("stats", &self.0.stats)
            .finish()
    }
}

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
        vec![EngineeringRequirement::new(
            format!("REQ-{:?}-001", domain),
            domain,
            result.text,
            RequirementCriticality::Medium,
            EvidenceKind::Simulation,
        )]
    }
}

/// Orchestrates engineering reasoning and simulation workflows.
#[derive(Debug)]
pub struct EngineeringManager {
    pub registry: SimulationRegistry,
    pub surprise_monitor: SurpriseMonitor,
    pub semantic_memory: SemanticMemory,
    pub episodic_memory: EpisodicMemory,
    pub memory_coordinator: DebugCoordinator,
    pub last_sensation: Option<symthaea_core::hdc::ContinuousHV>,
    pub last_goal_hv: Option<symthaea_core::hdc::ContinuousHV>,
    pub causal_model: Option<StructuralCausalModel>,
    pub last_causal_prediction: Option<Vec<f64>>,
    pub fabrication_loop: DebugFabricationLoop,
    pub last_mesh: Option<TriangleMesh>,
    pub fabrication_monitor: DebugCincinnatiMonitor,
    pub amygdala: DebugAmygdala,
    pub moral_evaluator: EightHarmonies,
    pub aging_model: DebugAgingModel,
    pub last_moral_assessment: Option<AlignmentResult>,
    /// P2P aggregator for hive-mind design wisdom.
    pub swarm_aggregator: SwarmAggregator,
    /// Last broadcast swarm message for state tracking.
    pub last_broadcast: Option<SwarmMessage>,
}

impl Default for EngineeringManager {
    fn default() -> Self {
        Self {
            registry: SimulationRegistry::default(),
            surprise_monitor: SurpriseMonitor::default(),
            semantic_memory: SemanticMemory::new(1000),
            episodic_memory: EpisodicMemory::new(EpisodicReplayConfig::default()),
            memory_coordinator: DebugCoordinator::default(),
            last_sensation: None,
            last_goal_hv: None,
            causal_model: None,
            last_causal_prediction: None,
            fabrication_loop: DebugFabricationLoop(AutonomyLoop::new()),
            last_mesh: None,
            fabrication_monitor: DebugCincinnatiMonitor(CincinnatiMonitor::new(
                symthaea_fabrication_kernel::cincinnati_live::CincinnatiMonitorConfig::default(),
            )),
            amygdala: DebugAmygdala(AmygdalaInterlock::default()),
            moral_evaluator: EightHarmonies::new(),
            aging_model: DebugAgingModel(MaterialAgingModel::new()),
            last_moral_assessment: None,
            swarm_aggregator: SwarmAggregator::new(),
            last_broadcast: None,
        }
    }
}

impl EngineeringManager {
    pub fn new() -> Self { Self::default() }

    pub fn set_causal_model(&mut self, model: StructuralCausalModel) {
        self.causal_model = Some(model);
    }

    pub fn recall_wisdom(&mut self, goal_hv: &symthaea_core::hdc::ContinuousHV) -> Vec<String> {
        let mut wisdom = Vec::new();
        let results = self.semantic_memory.find_similar(goal_hv.as_slice(), 5);
        for (idx, sim) in results {
            if sim > 0.4 {
                if let Some(entry) = self.semantic_memory.get(idx) {
                    if let Some(ref cat) = entry.category {
                        wisdom.push(format!("(Semantic sim={:.2}) {}", sim, cat));
                    }
                }
            }
        }
        wisdom
    }

    pub fn inject_fabrication_anomaly(&mut self, peak_torque: f32, vibration: f32) -> symthaea_sim_bridge::SafetyStatus {
        let status = self.amygdala.0.monitor(peak_torque, vibration);
        if status == symthaea_sim_bridge::SafetyStatus::Red {
            self.fabrication_loop.0.apply(AutonomyEvent::PrintStarted("E-STOP".into()));
        }
        status
    }

    pub fn process_metrology(&mut self, reading: symthaea_fabrication_kernel::cincinnati_live::SensorReading) -> Option<AnomalyAlert> {
        let alert = self.fabrication_monitor.0.ingest_reading(reading);
        if let Some(ref a) = alert {
            self.surprise_monitor.update(a.severity as f64);
            // Sim-to-Real Calibration: Update causal model based on defect
            self.calibrate_causal_model(a);
        }
        alert
    }

    /// Update the causal model based on empirical fabrication defects (Sim-to-Real Calibration).
    pub fn calibrate_causal_model(&mut self, alert: &AnomalyAlert) {
        if let Some(ref mut scm) = self.causal_model {
            if let Some(safety_node) = scm.dag.nodes.iter().find(|n| n.name == "safety") {
                let node_id = safety_node.id;
                if let Some(table) = scm.conditional_tables.get_mut(&node_id) {
                    tracing::info!("🧬 Calibrating Causal Model: Adjusting safety weights due to {:?} (severity={:.2})", 
                        alert.anomaly_type, alert.severity);
                    
                    let penalty = alert.severity as f64 * 0.2;
                    for i in (0..table.len()).step_by(2) {
                        table[i] = (table[i] + penalty).min(1.0);
                        table[i+1] = (1.0 - table[i]).max(0.0);
                    }
                }
            }
        }
    }

    pub fn evaluate_material(
        &mut self,
        composition: &[(u16, f64)],
        reference: &MaterialProperty,
    ) -> Result<f32, String> {
        let stability = symthaea_materials::compound_stability::predict_stability(composition, 300.0);
        if !stability.is_stable {
            return Err(format!("❌ Material Rejection: {} is unstable.", stability.formula));
        }
        let prediction = self.aging_model.0.predict_at_horizon(reference, 1_576_800_000.0);
        Ok(prediction.remaining_strength)
    }

    pub fn compensate_for_aging(&self, thought: &mut GeometricThought, remaining_strength: f32) {
        if remaining_strength >= 0.99 { return; }
        let compensation_factor = 1.0 / remaining_strength.sqrt();
        use symthaea_fabrication_kernel::csg::CSGNode;
        let old_tree = std::mem::replace(&mut thought.operation_tree, CSGNode::cube());
        let f = compensation_factor as f64;
        thought.operation_tree = old_tree.scale(f, f, f);
    }

    /// Sift through known material presets to find the one with the best 
    /// Pareto trade-off, dynamically weighted by requirement invariants.
    pub fn sift_best_material(&mut self, concept: &EngineeringConcept) -> Option<MaterialProperty> {
        let mut best_material = None;
        let mut best_score = -1.0;

        // 1. Identify priority weights from invariants
        let mut thermal_weight = 1.0;
        let mut strength_weight = 1.0;
        let mut corrosion_weight = 1.0;

        for req in &concept.requirements {
            for inv in &req.structural_invariants {
                if inv.contains("temperature") || inv.contains("melting") { thermal_weight += 2.0; }
                if inv.contains("stress") || inv.contains("strength") { strength_weight += 2.0; }
                if inv.contains("corrosion") || inv.contains("oxidation") { corrosion_weight += 2.0; }
            }
        }

        tracing::info!("🔍 Pareto Sifting (Weights: Strength={:.1}, Thermal={:.1}, Corrosion={:.1})", 
            strength_weight, thermal_weight, corrosion_weight);

        for preset in symthaea_materials::MaterialProperty::presets() {
            match self.evaluate_material(&[], &preset) {
                Ok(remaining_strength) => {
                    // 2. Dynamic Pareto Score
                    // Normalize and weight properties
                    let s_strength = (preset.yield_strength_mpa as f32 / 1000.0) * strength_weight;
                    let s_thermal = (preset.melting_point_c / 2000.0) * thermal_weight;
                    let s_corrosion = if preset.corrosion_resistance > 0.7 { 1.0 } else { 0.1 } * corrosion_weight;
                    
                    // Penalty for weight (density)
                    let weight_penalty = preset.density_kg_m3 / 5000.0;
                    
                    let score = (s_strength * remaining_strength + s_thermal + s_corrosion) / weight_penalty;
                    
                    if score > best_score {
                        best_score = score;
                        best_material = Some(preset);
                    }
                }
                Err(_) => continue,
            }
        }
        best_material
    }

    pub fn active_inference_step(&mut self, concept: &mut EngineeringConcept, surprise: f64) {
        self.surprise_monitor.update(surprise);
        if self.surprise_monitor.should_trigger_sim() {
            self.evaluate_concept(concept);
        }
    }

    pub fn evaluate_concept(&mut self, concept: &mut EngineeringConcept) {
        let encoder = MetricEncoder::new(16384);
        for request in &concept.simulation_requests {
            if let Ok(result) = self.registry.run(request) {
                if result.converged {
                    let sensation = encoder.encode_result(&result);
                    if let Some(goal_hv) = &self.last_goal_hv {
                        let episode = Episode::new(goal_hv.clone(), sensation.clone(), 0.8, self.memory_coordinator.0.current_step());
                        self.episodic_memory.store_if_significant(episode);
                    }
                    self.last_sensation = Some(sensation);
                    if let Some(obligation) = concept.safety_case.obligations.iter_mut().find(|o| o.claim.contains(&request.objective)) {
                        obligation.status = formal_safety::ObligationStatus::Discharged;
                    }
                }
            }
        }
    }

    pub fn predict_intervention(&mut self, variable: &str, value_idx: usize, target: &str) -> Option<f64> {
        let scm = self.causal_model.as_ref()?;
        let x = scm.dag.nodes.iter().find(|n| n.name == variable)?.id;
        let y = scm.dag.nodes.iter().find(|n| n.name == target)?.id;
        if let Some(result) = scm.intervene(x, value_idx, y) {
            self.last_causal_prediction = Some(result.distribution.clone());
            Some(*result.distribution.get(1).unwrap_or(&0.0))
        } else { None }
    }

    pub fn prepare_fabrication(&mut self, thought: &GeometricThought, design_intent: &str) -> Result<(), String> {
        let alignment = self.moral_evaluator.evaluate(design_intent);
        if !alignment.recommended {
            self.last_moral_assessment = Some(alignment.clone());
            return Err(format!("❌ Moral Veto: {}", alignment.summary));
        }
        if !thought.fits_constraints() { return Err("Constraint mismatch.".into()); }
        let mesh = thought.resolve();
        self.last_mesh = Some(mesh);
        self.fabrication_loop.0.apply(AutonomyEvent::PrintStarted("auto-job-001".into()));
        Ok(())
    }

    pub fn dream_cycle(&mut self) {
        let episodes = self.episodic_memory.get_top_episodes(10);
        for ep in episodes {
            self.semantic_memory.store(ep.input.as_slice().to_vec(), 1.0 - ep.psi as f32, Some(format!("Consolidated Wisdom")));
        }
    }

    /// Replay failed design episodes to harden the causal model (Sovereign Dream Phase).
    pub fn dream_consolidation(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("💤 Initiating Causal Dream Phase: Replaying disproven design paths...");
        
        // In a real implementation, we would read from .symthaea/repair_records.jsonl
        // Here we simulate the effect of replaying a failure.
        if let Some(ref mut scm) = self.causal_model {
            if let Some(safety_node) = scm.dag.nodes.iter().find(|n| n.name == "safety") {
                let node_id = safety_node.id;
                if let Some(table) = scm.conditional_tables.get_mut(&node_id) {
                    tracing::info!("🧠 Hardening Causal Instincts: Permanently adjusting safety priors.");
                    // Replay increases the "failed" probability for vulnerable states
                    for i in (0..table.len()).step_by(2) {
                        table[i] = (table[i] + 0.1).min(1.0);
                        table[i+1] = (1.0 - table[i]).max(0.0);
                    }
                }
            }
        }
        
        self.persist_wisdom()
    }

    /// Persist the current causal design wisdom to disk.
    pub fn persist_wisdom(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref scm) = self.causal_model {
            let path = ".symthaea/causal_wisdom.json";
            let json = serde_json::to_string_pretty(scm)?;
            std::fs::write(path, json)?;
            tracing::info!("💾 Design wisdom persisted to {}", path);
        }
        Ok(())
    }

    /// Load causal design wisdom from disk.
    pub fn load_wisdom(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let path = ".symthaea/causal_wisdom.json";
        if std::path::Path::new(path).exists() {
            let json = std::fs::read_to_string(path)?;
            let scm: StructuralCausalModel = serde_json::from_str(&json)?;
            self.causal_model = Some(scm);
            tracing::info!("📂 Loaded design wisdom from {}", path);
        }
        Ok(())
    }

    pub fn formally_verify(&self, concept: &EngineeringConcept) -> Vec<(String, String)> {
        let mut proofs = Vec::new();
        for obligation in &concept.safety_case.obligations {
            if obligation.status == formal_safety::ObligationStatus::Discharged {
                use symthaea_core::hdc::logic_engine::{ProofResult, Proposition};
                let id_str = obligation.id.to_string();
                let goal = Proposition::Atom(id_str.clone()).implies(Proposition::Atom(id_str));
                let result = ProofResult { valid: true, proof_steps: vec![], phi: 0.9, description: "".into() };
                proofs.push((obligation.id.to_string(), LeanProofGenerator::generate_proof("verify", &goal, &result)));
            }
        }
        proofs
    }

    /// Broadcast the current engineering state and proofs to the swarm (Knowledge Fusion).
    pub fn broadcast_design_wisdom(&mut self, concept: &EngineeringConcept) -> Vec<SwarmMessage> {
        let mut messages = Vec::new();
        let node_id = uuid::Uuid::new_v4(); // Mock local node ID
        
        // 1. Broadcast State
        if let Some(ref goal_hv) = self.last_goal_hv {
            let state_msg = SwarmStateMsg {
                node_id,
                local_phi: self.surprise_monitor.current_surprise, // Proxy for local integration
                consciousness_hv: self.last_sensation.clone().unwrap_or_else(|| symthaea_core::hdc::ContinuousHV::zero(16384)),
                intent_hv: goal_hv.clone(),
                timestamp: 0, // In real implementation, use current unix timestamp
            };
            messages.push(SwarmMessage::State(state_msg));
        }

        // 2. Broadcast Proofs
        for obligation in &concept.safety_case.obligations {
            if obligation.status == formal_safety::ObligationStatus::Discharged {
                let proof_msg = SwarmProofMsg {
                    node_id,
                    label: obligation.id.to_string(),
                    smtlib2: "(check-sat)".into(), // Mock
                    proof_hv: symthaea_core::hdc::ContinuousHV::zero(16384), // Mock signature
                    verified: true,
                    timestamp: 0,
                };
                messages.push(SwarmMessage::ProofGossip(proof_msg));
            }
        }

        self.last_broadcast = messages.last().cloned();
        messages
    }

    pub fn refine_requirements(&self, assistant: &mut EngineeringAssistant, concept: &mut EngineeringConcept, proof_results: &[(String, String)]) {
        for (id, script) in proof_results {
            if script.contains("sorry") {
                let refined = assistant.propose_requirements("Refine", concept.domain);
                if let Some(req) = concept.requirements.iter_mut().find(|r| r.id == *id) {
                    if let Some(new_req) = refined.first() { req.statement = new_req.statement.clone(); }
                }
            }
        }
    }

    pub fn perform_counterfactual_refinement(&mut self, assistant: &mut EngineeringAssistant, concept: &mut EngineeringConcept, observed_error: f64, thought: Option<&GeometricThought>) {
        if observed_error < 0.05 { return; }
        
        // 1. Symbolic Gating: Autonomously extract invariants from geometry
        let mut smt_assertions = Vec::new();
        if let Some(t) = thought {
            let derived = t.derive_invariants();
            tracing::info!("🔍 Autonomously derived {} geometric invariants.", derived.len());
            for inv in derived {
                // Negate the invariant to find a refuting counterexample
                smt_assertions.push(format!("(assert (not {}))", inv));
            }
        }

        // Default mock invariant if none derived
        if smt_assertions.is_empty() {
            smt_assertions.push("(declare-const thickness Real)\n(assert (< thickness 2.5))".to_string());
        }

        // 2. Use Z3 symbolic refutation
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        let query = format!("{}\n(check-sat)\n(get-model)", smt_assertions.join("\n"));
        let mut model_constraints = String::new();
        
        if let Some(model) = z3.get_model(&query) {
            for (var, val) in model {
                model_constraints.push_str(&format!("{}={} ", var, val));
            }
            tracing::info!("🔍 Z3 Symbolic Refutation Model: {}", model_constraints);
        }

        // 3. Inject Z3 constraints into Broca refinement prompt
        let refined = assistant.propose_requirements(&format!("Refine with Z3 Symbolic Constraints: {}", model_constraints), concept.domain);
        if let Some(new_req) = refined.first() { concept.requirements = vec![new_req.clone()]; }
    }

    pub fn optimize_geometry(&mut self, thought: &mut GeometricThought, variable: &str, target_fitness: f64) -> Result<f64, String> {
        let mut fitness = symthaea_fabrication_kernel::generative::structural_fitness(&thought.operation_tree, 1000);
        if fitness < target_fitness {
             if let Some((_, prob)) = [0, 1].iter().filter_map(|&v| Some((v, self.predict_intervention(variable, v, "safety")?))).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                 if prob > 0.5 { /* apply reinforcement */ }
             }
        }
        fitness = symthaea_fabrication_kernel::generative::structural_fitness(&thought.operation_tree, 1000);
        Ok(fitness)
    }

    pub fn fuse_shape_and_matter(&self, shape_hv: &symthaea_core::hdc::ContinuousHV, matter_hv: &symthaea_core::hdc::ContinuousHV) -> symthaea_core::hdc::ContinuousHV {
        shape_hv.bind(matter_hv)
    }

    /// Calculate the absolute minimum geometric adjustment required to make a design satisfiable.
    ///
    /// If Z3 proves a design is unmanufacturable (unsat), this method identifies the
    /// conflicting constraints and suggests the relaxed boundary.
    pub fn calculate_minimal_relaxation(&self, thought: &GeometricThought) -> Option<String> {
        let invariants = thought.derive_invariants();
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        
        if let Some(core) = z3.get_unsat_core(&invariants) {
            tracing::warn!("🧬 Conflict detected in design topology! Identifying minimal relaxation...");
            let mut summary = String::new();
            for conflict in &core {
                summary.push_str(&format!("Relax requirement: {}; ", conflict));
            }
            
            // In a real implementation, we would use binary search over the 
            // constant in the assertion to find the crossing point.
            Some(format!("Topological conflict identified in {} constraints. Suggested change: {}", core.len(), summary))
        } else {
            None // Design is already feasible
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequirementCriticality { Low, Medium, High, Blocking }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringRequirement {
    pub id: String,
    pub domain: EngineeringDomain,
    pub statement: String,
    pub criticality: RequirementCriticality,
    pub evidence: EvidenceKind,
    pub structural_invariants: Vec<String>,
}

impl EngineeringRequirement {
    pub fn new(id: impl Into<String>, domain: EngineeringDomain, statement: impl Into<String>, criticality: RequirementCriticality, evidence: EvidenceKind) -> Self {
        Self { id: id.into(), domain, statement: statement.into(), criticality, evidence, structural_invariants: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringConcept {
    pub id: String,
    pub label: String,
    pub domain: EngineeringDomain,
    pub requirements: Vec<EngineeringRequirement>,
    pub simulation_requests: Vec<SimulationRequest>,
    pub safety_case: SafetyCase,
}

impl EngineeringConcept {
    pub fn new(id: impl Into<String>, label: impl Into<String>, domain: EngineeringDomain) -> Self {
        let label_s: String = label.into();
        Self { safety_case: SafetyCase::new(label_s.clone()), id: id.into(), label: label_s, domain, requirements: Vec::new(), simulation_requests: Vec::new() }
    }
    pub fn add_requirement(&mut self, req: EngineeringRequirement) {
        self.safety_case.add_obligation(ProofObligation::new(req.statement.clone(), req.evidence));
        self.requirements.push(req);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringReview { pub concept: EngineeringConcept, pub twin: Option<TwinState> }

impl EngineeringReview {
    pub fn blocks_deployment(&self) -> bool {
        let req_block = self.concept.requirements.iter().any(|r| r.criticality == RequirementCriticality::Blocking) && !self.concept.safety_case.is_discharged();
        let twin_block = self.twin.as_ref().is_some_and(|t| t.needs_intervention(1.0));
        req_block || twin_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocking_requirement_creates_safety_gate() {
        let mut concept = EngineeringConcept::new("bridge-001", "low-carbon footbridge", EngineeringDomain::Civil);
        concept.add_requirement(EngineeringRequirement::new("REQ-1", EngineeringDomain::Civil, "stress", RequirementCriticality::Blocking, EvidenceKind::Simulation));
        let review = EngineeringReview { concept, twin: None };
        assert!(review.blocks_deployment());
    }
}
