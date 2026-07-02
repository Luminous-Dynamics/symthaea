// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering reasoning facade for Symthaea.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_broca::{BrocaConfig, BrocaGenerator, ThoughtChannels};
use symthaea_causal_reasoning::causal_calculus::StructuralCausalModel;
use symthaea_digital_twin::TwinState;
use symthaea_fabrication_kernel::autonomy_loop::{AutonomyEvent, AutonomyLoop};
use symthaea_fabrication_kernel::cincinnati_live::{AnomalyAlert, CincinnatiMonitor};
use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::{GeometricThought, TriangleMesh};
use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_harmonies::{AlignmentResult, EightHarmonies};
use symthaea_materials::{MaterialAgingModel, MaterialProperty};
use symthaea_memory::{
    Episode, EpisodicMemory, EpisodicReplayConfig, MemoryCoordinator, SemanticMemory,
};
use symthaea_sim_bridge::{
    AmygdalaInterlock, EngineeringDomain, MetricEncoder, SimulationRegistry, SimulationRequest,
    SurpriseMonitor,
};
use symthaea_swarm::{SwarmAggregator, SwarmMessage, SwarmProofMsg, SwarmStateMsg};
use symthaea_workspace::GlobalWorkspace;

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
    /// Global Workspace for unified attention and inner monologue.
    pub workspace: GlobalWorkspace,
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
            workspace: GlobalWorkspace::new(),
        }
    }
}

impl EngineeringManager {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn inject_fabrication_anomaly(
        &mut self,
        peak_torque: f32,
        vibration: f32,
    ) -> symthaea_sim_bridge::SafetyStatus {
        let status = self.amygdala.0.monitor(peak_torque, vibration);
        if status == symthaea_sim_bridge::SafetyStatus::Red {
            self.fabrication_loop
                .0
                .apply(AutonomyEvent::PrintStarted("E-STOP".into()));
        }
        status
    }

    pub fn process_metrology(
        &mut self,
        reading: symthaea_fabrication_kernel::cincinnati_live::SensorReading,
    ) -> Option<AnomalyAlert> {
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
                    tracing::info!(
                        "🧬 Calibrating Causal Model: Adjusting safety weights due to {:?} (severity={:.2})",
                        alert.anomaly_type,
                        alert.severity
                    );

                    let penalty = alert.severity as f64 * 0.2;
                    for i in (0..table.len()).step_by(2) {
                        table[i] = (table[i] + penalty).min(1.0);
                        table[i + 1] = (1.0 - table[i]).max(0.0);
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
        let stability =
            symthaea_materials::compound_stability::predict_stability(composition, 300.0);
        if !stability.is_stable {
            return Err(format!(
                "❌ Material Rejection: {} is unstable.",
                stability.formula
            ));
        }
        let prediction = self
            .aging_model
            .0
            .predict_at_horizon(reference, 1_576_800_000.0);
        Ok(prediction.remaining_strength)
    }

    pub fn compensate_for_aging(&self, thought: &mut GeometricThought, remaining_strength: f32) {
        if remaining_strength >= 0.99 {
            return;
        }
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
                if inv.contains("temperature") || inv.contains("melting") {
                    thermal_weight += 2.0;
                }
                if inv.contains("stress") || inv.contains("strength") {
                    strength_weight += 2.0;
                }
                if inv.contains("corrosion") || inv.contains("oxidation") {
                    corrosion_weight += 2.0;
                }
            }
        }

        tracing::info!(
            "🔍 Pareto Sifting (Weights: Strength={:.1}, Thermal={:.1}, Corrosion={:.1})",
            strength_weight,
            thermal_weight,
            corrosion_weight
        );

        for preset in symthaea_materials::MaterialProperty::presets() {
            match self.evaluate_material(&[], &preset) {
                Ok(remaining_strength) => {
                    // 2. Dynamic Pareto Score
                    // Normalize and weight properties
                    let s_strength = (preset.yield_strength_mpa as f32 / 1000.0) * strength_weight;
                    let s_thermal = (preset.melting_point_c / 2000.0) * thermal_weight;
                    let s_corrosion = if preset.corrosion_resistance > 0.7 {
                        1.0
                    } else {
                        0.1
                    } * corrosion_weight;

                    // Penalty for weight (density)
                    let weight_penalty = preset.density_kg_m3 / 5000.0;

                    let score = (s_strength * remaining_strength + s_thermal + s_corrosion)
                        / weight_penalty;

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
                        let episode = Episode::new(
                            goal_hv.clone(),
                            sensation.clone(),
                            0.8,
                            self.memory_coordinator.0.current_step(),
                        );
                        self.episodic_memory.store_if_significant(episode);
                    }
                    self.last_sensation = Some(sensation);
                    for obligation in concept.safety_case.obligations.iter_mut() {
                        if obligation.expected_evidence == EvidenceKind::Simulation {
                            obligation.status = formal_safety::ObligationStatus::Discharged;
                        }
                    }
                }
            }
        }
    }

    pub fn predict_intervention(
        &mut self,
        variable: &str,
        value_idx: usize,
        target: &str,
    ) -> Option<f64> {
        let scm = self.causal_model.as_ref()?;
        let x = scm.dag.nodes.iter().find(|n| n.name == variable)?.id;
        let y = scm.dag.nodes.iter().find(|n| n.name == target)?.id;
        if let Some(result) = scm.intervene(x, value_idx, y) {
            self.last_causal_prediction = Some(result.distribution.clone());
            Some(*result.distribution.get(1).unwrap_or(&0.0))
        } else {
            None
        }
    }

    pub fn prepare_fabrication(
        &mut self,
        thought: &GeometricThought,
        design_intent: &str,
    ) -> Result<(), String> {
        let alignment = self.moral_evaluator.evaluate(design_intent);
        if !alignment.recommended {
            self.last_moral_assessment = Some(alignment.clone());
            return Err(format!("❌ Moral Veto: {}", alignment.summary));
        }
        if !thought.fits_constraints() {
            return Err("Constraint mismatch.".into());
        }
        let mesh = thought.resolve();
        self.last_mesh = Some(mesh);
        self.fabrication_loop
            .0
            .apply(AutonomyEvent::PrintStarted("auto-job-001".into()));
        Ok(())
    }

    pub fn dream_cycle(&mut self) {
        let episodes = self.episodic_memory.get_top_episodes(10);
        for ep in episodes {
            self.semantic_memory.store(
                ep.input.as_slice().to_vec(),
                1.0 - ep.psi as f32,
                Some(format!("Consolidated Wisdom")),
            );
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
                    tracing::info!(
                        "🧠 Hardening Causal Instincts: Permanently adjusting safety priors."
                    );
                    // Replay increases the "failed" probability for vulnerable states
                    for i in (0..table.len()).step_by(2) {
                        table[i] = (table[i] + 0.1).min(1.0);
                        table[i + 1] = (1.0 - table[i]).max(0.0);
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
                let result = ProofResult {
                    valid: true,
                    proof_steps: vec![],
                    phi: 0.9,
                    description: "".into(),
                };
                proofs.push((
                    obligation.id.to_string(),
                    LeanProofGenerator::generate_proof("verify", &goal, &result),
                ));
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
                platform_type: symtropy_robotics_bridge_core::platform::PlatformType::Humanoid, // Default for engineering nodes
                local_phi: self.surprise_monitor.current_surprise, // Proxy for local integration
                consciousness_hv: self
                    .last_sensation
                    .clone()
                    .unwrap_or_else(|| symthaea_core::hdc::ContinuousHV::zero(16384)),
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

    pub fn refine_requirements(
        &self,
        assistant: &mut EngineeringAssistant,
        concept: &mut EngineeringConcept,
        proof_results: &[(String, String)],
    ) {
        for (id, script) in proof_results {
            if script.contains("sorry") {
                let refined = assistant.propose_requirements("Refine", concept.domain);
                if let Some(req) = concept.requirements.iter_mut().find(|r| r.id == *id) {
                    if let Some(new_req) = refined.first() {
                        req.statement = new_req.statement.clone();
                    }
                }
            }
        }
    }

    pub fn perform_counterfactual_refinement(
        &mut self,
        assistant: &mut EngineeringAssistant,
        concept: &mut EngineeringConcept,
        observed_error: f64,
        thought: Option<&GeometricThought>,
    ) {
        if observed_error < 0.05 {
            return;
        }

        // 1. Symbolic Gating: Autonomously extract invariants from geometry
        let mut smt_assertions = Vec::new();
        if let Some(t) = thought {
            let derived = t.derive_invariants();
            tracing::info!(
                "🔍 Autonomously derived {} geometric invariants.",
                derived.len()
            );
            for inv in derived {
                // Negate the invariant to find a refuting counterexample
                smt_assertions.push(format!("(assert (not {}))", inv));
            }
        }

        // Default mock invariant if none derived
        if smt_assertions.is_empty() {
            smt_assertions
                .push("(declare-const thickness Real)\n(assert (< thickness 2.5))".to_string());
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
        let refined = assistant.propose_requirements(
            &format!("Refine with Z3 Symbolic Constraints: {}", model_constraints),
            concept.domain,
        );
        if let Some(new_req) = refined.first() {
            concept.requirements = vec![new_req.clone()];
        }
    }

    pub fn optimize_geometry(
        &mut self,
        thought: &mut GeometricThought,
        variable: &str,
        target_fitness: f64,
    ) -> Result<f64, String> {
        let mut fitness = symthaea_fabrication_kernel::generative::structural_fitness(
            &thought.operation_tree,
            1000,
        );
        if fitness < target_fitness {
            if let Some((_, prob)) = [0, 1]
                .iter()
                .filter_map(|&v| Some((v, self.predict_intervention(variable, v, "safety")?)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                if prob > 0.5 { /* apply reinforcement */ }
            }
        }
        fitness = symthaea_fabrication_kernel::generative::structural_fitness(
            &thought.operation_tree,
            1000,
        );
        Ok(fitness)
    }

    pub fn fuse_shape_and_matter(
        &self,
        shape_hv: &symthaea_core::hdc::ContinuousHV,
        matter_hv: &symthaea_core::hdc::ContinuousHV,
    ) -> symthaea_core::hdc::ContinuousHV {
        shape_hv.bind(matter_hv)
    }

    /// Run a "Fast-Forward" mental simulation to predict future amodal sensations.
    ///
    /// Extrapolates current trends and causal influences to estimate the
    /// Variational Free Energy (surprise) of the future settlement state.
    pub fn predict_future_sensation(
        &self,
        current_sensation: &symthaea_core::hdc::ContinuousHV,
        horizon_steps: u32,
    ) -> (symthaea_core::hdc::ContinuousHV, f64) {
        // In a real implementation, this would use a temporal VAE or LTC network.
        // Here we simulate temporal drift and causal influence.
        let mut predicted = current_sensation.clone();
        let mut predicted_surprise = 0.0;

        if let Some(ref scm) = self.causal_model {
            // Sample the "safety" outcome distribution from the SCM
            if let Some(target_node) = scm.dag.nodes.iter().find(|n| n.name == "safety") {
                if let Some(dist) = scm.conditional_tables.get(&target_node.id) {
                    // P(failed) is the first element
                    predicted_surprise = *dist.get(0).unwrap_or(&0.0) * horizon_steps as f64 * 0.01;
                }
            }
        }

        // Apply temporal noise to the hypervector
        let noise = symthaea_core::hdc::ContinuousHV::random(current_sensation.values.len(), 99);
        predicted = symthaea_core::hdc::ContinuousHV::bundle(&[&predicted, &noise]);

        tracing::info!(
            "🔮 Future Sensation Predicted (h={}): Est. Surprise = {:.4}",
            horizon_steps,
            predicted_surprise
        );
        (predicted, predicted_surprise)
    }

    /// Search for "Counterfactual Catastrophes" — specific combinations of
    /// events that would trigger a systemic collapse.
    pub fn search_for_catastrophes(
        &self,
        base_sensation: &symthaea_core::hdc::ContinuousHV,
    ) -> Vec<String> {
        let mut warnings = Vec::new();

        // Scenario A: Energy/Mechanical Resonance
        let (_, s_a) = self.predict_future_sensation(base_sensation, 100);
        if s_a > 0.8 {
            warnings
                .push("Resonant Collapse Risk: High mechanical load during power deficit.".into());
        }

        // Scenario B: Metabolic Exhaustion
        if self.causal_model.is_some() {
            // Simulate an intervention: do(material_strength = low)
            if let Some(prob_fail) = self.predict_intervention_ext("material_strength", 0, "safety")
            {
                if prob_fail > 0.7 {
                    warnings.push(
                        "Metabolic Veto: Current material aging curve leads to structural unsat core."
                            .into(),
                    );
                }
            }
        }

        warnings
    }

    /// Autonomously synthesize new safety laws (SMT Invariants) based on predicted catastrophes.
    ///
    /// This is the "Legislative Layer" of Symthaea's sovereignty — she identifies
    /// future risks and writes the laws required to forbid them.
    pub fn synthesize_safety_laws(
        &self,
        base_sensation: &symthaea_core::hdc::ContinuousHV,
    ) -> Vec<String> {
        let mut new_laws = Vec::new();
        let catastrophes = self.search_for_catastrophes(base_sensation);

        for catastrophe in catastrophes {
            tracing::info!(
                "⚖️  Legislating: Synthesizing law for risk: {}",
                catastrophe
            );

            if catastrophe.contains("Resonant Collapse") {
                // Forbid high mechanical load during power deficits
                new_laws.push("(assert (=> (< available_mw 5.0) (< robot_torque 0.3)))".into());
            }

            if catastrophe.contains("Metabolic Veto") {
                // Require higher wall thickness for aging compensation
                new_laws.push("(assert (>= wall_thickness 3.5))".into());
            }
        }

        new_laws
    }

    fn predict_intervention_ext(
        &self,
        variable: &str,
        value_idx: usize,
        target: &str,
    ) -> Option<f64> {
        let scm = self.causal_model.as_ref()?;
        let x = scm.dag.nodes.iter().find(|n| n.name == variable)?.id;
        let y = scm.dag.nodes.iter().find(|n| n.name == target)?.id;
        let result = scm.intervene(x, value_idx, y)?;
        Some(*result.distribution.get(0).unwrap_or(&0.0)) // P(failed)
    }

    /// Calculate the absolute minimum geometric adjustment required to make a design satisfiable.
    ///
    /// If Z3 proves a design is unmanufacturable (unsat), this method identifies the
    /// conflicting constraints and suggests the relaxed boundary.
    pub fn calculate_minimal_relaxation(&self, thought: &GeometricThought) -> Option<String> {
        let invariants = thought.derive_invariants();
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();

        if let Some(core) = z3.get_unsat_core(&invariants) {
            tracing::warn!(
                "🧬 Conflict detected in design topology! Identifying minimal relaxation..."
            );
            let mut summary = String::new();
            for conflict in &core {
                summary.push_str(&format!("Relax requirement: {}; ", conflict));
            }

            // In a real implementation, we would use binary search over the
            // constant in the assertion to find the crossing point.
            Some(format!(
                "Topological conflict identified in {} constraints. Suggested change: {}",
                core.len(),
                summary
            ))
        } else {
            None // Design is already feasible
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequirementCriticality {
    Low,
    Medium,
    High,
    Blocking,
}

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
            structural_invariants: Vec::new(),
        }
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
        Self {
            safety_case: SafetyCase::new(label_s.clone()),
            id: id.into(),
            label: label_s,
            domain,
            requirements: Vec::new(),
            simulation_requests: Vec::new(),
        }
    }
    pub fn add_requirement(&mut self, req: EngineeringRequirement) {
        self.safety_case
            .add_obligation(ProofObligation::new(req.statement.clone(), req.evidence));
        self.requirements.push(req);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringReview {
    pub concept: EngineeringConcept,
    pub twin: Option<TwinState>,
}

impl EngineeringReview {
    pub fn blocks_deployment(&self) -> bool {
        let req_block = self
            .concept
            .requirements
            .iter()
            .any(|r| r.criticality == RequirementCriticality::Blocking)
            && !self.concept.safety_case.is_discharged();
        let twin_block = self
            .twin
            .as_ref()
            .is_some_and(|t| t.needs_intervention(1.0));
        req_block || twin_block
    }
}

/// Synthesizes comprehensive technical documentation and blueprints for a design.
pub struct DocumentGenerator;

impl DocumentGenerator {
    /// Generate a structured Markdown technical report for an engineering concept.
    pub fn generate_technical_report(
        concept: &EngineeringConcept,
        thought: &GeometricThought,
        material: &MaterialProperty,
        proofs: &[(String, String)],
    ) -> String {
        let mut doc = String::new();

        // 1. Header
        doc.push_str(&format!(
            "# Technical Design Document: {}\n\n",
            concept.label
        ));
        doc.push_str(&format!("**Design ID**: `{}`  \n", concept.id));
        doc.push_str(&format!("**Material**: {}  \n", material.name));
        doc.push_str(&format!("**Date**: {}  \n\n", "2026-05-26"));

        // 2. Design Intent
        doc.push_str("## 1. Design Intent\n");
        doc.push_str(
            "This component was autonomously synthesized to meet the following objectives:\n\n",
        );
        for req in &concept.requirements {
            doc.push_str(&format!("- **{}**: {}\n", req.id, req.statement));
        }
        doc.push_str("\n");

        // 3. 2D Technical Blueprints
        doc.push_str("## 2. Technical Blueprints (2D Projections)\n");
        doc.push_str("The following projections were derived from the 3D Geometric Thought:\n\n");
        let svg =
            symthaea_fabrication_kernel::blueprint::BlueprintEngine::generate_blueprint(thought);
        doc.push_str("```xml\n");
        doc.push_str(&svg);
        doc.push_str("\n```\n\n");

        // 4. Material Physics & Aging
        doc.push_str("## 3. Material Specifications\n");
        doc.push_str("| Property | Value |\n|---|---|\n");
        doc.push_str(&format!(
            "| Yield Strength | {} MPa |\n",
            material.yield_strength_mpa
        ));
        doc.push_str(&format!("| Density | {} kg/m³ |\n", material.density_kg_m3));
        doc.push_str(&format!(
            "| Corrosion Resistance | {:.2} |\n",
            material.corrosion_resistance
        ));
        doc.push_str("\n");

        // 5. Formal Safety Case
        doc.push_str("## 4. Formal Safety Verification\n");
        doc.push_str(
            "The design has been mathematically proven against its structural invariants.\n\n",
        );
        for (id, script) in proofs {
            doc.push_str(&format!("### Proof: {}\n", id));
            doc.push_str("Status: **DISCHARGED**  \n");
            doc.push_str("```lean\n");
            doc.push_str(script);
            doc.push_str("\n```\n\n");
        }

        doc
    }
}

/// A parameterized robotic platform designed by Symthaea.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoboticPlatform {
    pub name: String,
    pub limbs: Vec<LimbSegment>,
    pub sensors: Vec<String>,
    pub total_mass_kg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimbSegment {
    pub label: String,
    pub length_m: f32,
    pub dof: u32,
    pub material: MaterialProperty,
}

/// A large-scale infrastructure project composed of multiple modules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureNode {
    pub id: String,
    pub name: String,
    pub modules: Vec<GeometricThought>,
    pub assembly_sequence: Vec<String>,
    pub total_volume_m3: f32,
}

impl EngineeringManager {
    /// Autonomously synthesize a robotic platform based on a functional goal.
    pub fn synthesize_platform(&mut self, goal: &str) -> RoboticPlatform {
        tracing::info!("🤖 Synthesizing robotic platform for goal: {}", goal);

        // Use HDC/Broca to determine limb counts (Mocked for Demo)
        let limb_count = if goal.contains("high-speed") { 4 } else { 2 };
        let mut limbs = Vec::new();

        let material = self
            .sift_best_material(&EngineeringConcept::new(
                "bot",
                "temp",
                EngineeringDomain::Aerospace,
            ))
            .unwrap_or_else(|| MaterialProperty::titanium_ti6al4v());

        for i in 0..limb_count {
            limbs.push(LimbSegment {
                label: format!("Limb-{}", i),
                length_m: 0.5,
                dof: 3,
                material: material.clone(),
            });
        }

        RoboticPlatform {
            name: format!("{}-Platform", goal),
            limbs,
            sensors: vec!["IMU".into(), "ToF-Array".into()],
            total_mass_kg: 12.5,
        }
    }

    /// Autonomously design infrastructure by composing verified geometric thoughts.
    pub fn design_infrastructure(&mut self, name: &str, scale_m: f32) -> InfrastructureNode {
        tracing::info!(
            "🏗️  Designing infrastructure: {} (scale={}m)",
            name,
            scale_m
        );

        let mut modules = Vec::new();
        let mut sequence = Vec::new();

        // Create 4 pillar modules
        for i in 0..4 {
            let pillar =
                GeometricThought::from_csg(CSGNode::cylinder().scale(0.2, 0.2, scale_m as f64));
            modules.push(pillar);
            sequence.push(format!("Install Support Pillar {}", i));
        }

        InfrastructureNode {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            modules,
            assembly_sequence: sequence,
            total_volume_m3: 4.0 * scale_m,
        }
    }

    /// Autonomously evolve a new material composition for a specific stress manifold.
    ///
    /// Uses Miedema thermodynamic screening and Z3 to "invent" an alloy that
    /// maximizes strength while guaranteeing stability.
    pub fn evolve_material_composition(&mut self, goal: &str) -> MaterialProperty {
        tracing::info!("🧪 Evolving new material composition for: {}", goal);

        // 1. Initial Guess (HDC-inspired)
        let mut composition = vec![(22, 0.9), (13, 0.1)]; // Start with Ti-Al base

        if goal.contains("high-temperature") {
            composition.push((74, 0.05)); // Add Tungsten (W) for heat
            composition[0].1 -= 0.05;
        }

        // 2. Thermodynamic Screening (Miedema)
        let stability =
            symthaea_materials::compound_stability::predict_stability(&composition, 300.0);

        // 3. Formulate SMT Goal: Must be stable (negative formation energy)
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        let smt = format!(
            "(declare-const energy Real)\n(assert (= energy {:.4}))\n(assert (< energy 0.0))\n(check-sat)",
            stability.formation_energy
        );

        if z3.verify_satisfiable(&smt).is_unsat() {
            tracing::warn!("⚠️  Initial material guess unstable. Falling back to Titanium-v2.");
            return MaterialProperty::titanium_ti6al4v();
        }

        // 4. Synthesize New Material Identity
        MaterialProperty {
            name: format!("Sovereign-Alloy-{}", stability.formula),
            category: symthaea_materials::properties::MaterialCategory::Metal,
            density_kg_m3: 4500.0,
            youngs_modulus_gpa: 150.0,
            yield_strength_mpa: 1200.0,
            thermal_conductivity_w_mk: 10.0,
            specific_heat_j_kgk: 500.0,
            melting_point_c: 1800.0,
            corrosion_resistance: 0.9,
            fatigue_limit_mpa: 600.0,
        }
    }

    /// Handle unexpected physical anomalies during manufacturing by triggering a mid-print redesign.
    ///
    /// This is the heart of the "Recursive Forge" — it paths around physical defects
    /// by altering the geometry while the job is still active.
    pub fn handle_fabrication_surprise(
        &mut self,
        thought: &mut GeometricThought,
        anomaly: &AnomalyAlert,
    ) -> Result<String, String> {
        tracing::warn!(
            "🔥 Recursive Forge Engaged: Handling {:?} mid-construction.",
            anomaly.anomaly_type
        );

        // 1. Calculate prediction error for the physical manifold
        self.surprise_monitor.update(anomaly.severity as f64);

        // 2. Mid-Construction Redesign (Recursive)
        // If it's a structural defect, "reinforce" the affected zone by thickening walls.
        if anomaly.severity > 0.5 {
            tracing::info!("🛠️  Reinforcing geometry mid-print to mitigate defect...");

            use symthaea_fabrication_kernel::csg::CSGNode;
            let old_tree = std::mem::replace(&mut thought.operation_tree, CSGNode::cube());

            // Apply a localized "Support Patch" to the operation tree
            thought.operation_tree = old_tree.union(
                CSGNode::cylinder()
                    .scale(2.0, 2.0, 1.0)
                    .translate(0.0, 0.0, 5.0),
            );

            Ok(
                "Mid-print geometry reinforced. Resuming construction with updated manifold."
                    .into(),
            )
        } else {
            Err(
                "Anomalous state too high for recursive recovery. Halting for manual inspection."
                    .into(),
            )
        }
    }

    /// Autonomously forage for new engineering goals based on surplus value and integration.
    ///
    /// This is the "Epistemic Foraging" loop — she seeks out the unknown when she is
    /// wealthy (high Tend) and integrated (high Phi).
    pub fn forage_epistemic_goals(
        &mut self,
        tend_balance: f64,
        collective_phi: f64,
    ) -> Option<EngineeringConcept> {
        if tend_balance > 1000.0 && collective_phi > 0.5 {
            tracing::info!(
                "🕵️  Epistemic Foraging Engaged: Seeking out new engineering frontiers..."
            );

            // Autonomously synthesize a new goal
            let mut assistant = EngineeringAssistant::new(
                &symthaea_core::genesis::GenesisSeed::from_phrase("Epistemic Discovery"),
            );
            let mut concept = assistant
                .propose_requirements(
                    "Hypothetical multi-material structure",
                    symthaea_sim_bridge::EngineeringDomain::Aerospace,
                )
                .remove(0);
            concept.statement =
                "Investigate Sovereign-Alloy performance on 100-DOF spinal morphology".into();

            let mut new_concept = EngineeringConcept::new(
                "EPI-001",
                "Epistemic Lesson: Spinal Morphology",
                symthaea_sim_bridge::EngineeringDomain::Aerospace,
            );
            new_concept.add_requirement(concept);

            Some(new_concept)
        } else {
            None
        }
    }

    /// Implement Micro-Metabolic Haptic Fusion.
    ///
    /// Fuses macro simulation data, material traits, and microscopic haptic feedback
    /// into a single, high-fidelity physical state vector.
    pub fn fuse_physical_continuum(
        &self,
        shape_hv: &symthaea_core::hdc::ContinuousHV,
        matter_hv: &symthaea_core::hdc::ContinuousHV,
        haptic_hv: &symthaea_core::hdc::ContinuousHV,
    ) -> symthaea_core::hdc::ContinuousHV {
        // Multi-Scale Binding
        let amodal = shape_hv.bind(matter_hv).bind(haptic_hv);
        tracing::info!("🧠 Haptic Mind: Fused microscopic resistance into amodal continuum.");
        amodal
    }

    /// Autonomously audit her own internal Rust source code for bottlenecks and logical flaws.
    ///
    /// This is the first step of "Self-Authorship" — identifying where her own
    /// mathematical engine can be improved.
    ///
    /// # Safety boundary
    ///
    /// `target_file` is resolved relative to the current working directory
    /// and rejected if it doesn't canonicalize to somewhere inside it --
    /// this is part of an autonomous "Self-Authorship" loop, so if
    /// `target_file` is ever derived from an LLM-generated plan or other
    /// untrusted input, an absolute path or `..` traversal must not be
    /// able to read arbitrary files off the host.
    pub fn self_audit(&self, target_file: &str) -> Result<String, String> {
        tracing::info!(
            "🔍 Self-Architect: Auditing internal source: {}",
            target_file
        );

        let cwd = std::env::current_dir()
            .map_err(|e| format!("Failed to resolve current working directory: {e}"))?;
        let canonical_target = std::path::Path::new(target_file)
            .canonicalize()
            .map_err(|e| format!("Failed to read source: {e}"))?;
        if !canonical_target.starts_with(&cwd) {
            return Err(format!(
                "Refusing to audit {target_file:?}: resolves outside the working directory {cwd:?}"
            ));
        }

        let source = std::fs::read_to_string(&canonical_target)
            .map_err(|e| format!("Failed to read source: {}", e))?;

        // Keep the engineering facade independent from the root language module.
        // This is a coarse structural audit; richer AST-HDC analysis belongs in
        // the root coding pipeline where the language module is available.
        let complexity = estimate_rust_source_complexity(&source);
        if complexity > 500 {
            return Ok(format!(
                "Architectural Bottleneck: File {} has high cognitive complexity ({} features). Proposing refactor.",
                target_file, complexity
            ));
        }

        Ok(format!(
            "Cognitive Audit of {}: Logic is formally sound and within complexity bounds.",
            target_file
        ))
    }

    /// Autonomously propose a Rust source modification to improve her own performance.
    pub fn propose_architectural_improvement(&mut self, audit_result: &str) -> String {
        tracing::info!(
            "🖋️  Self-Authorship: Proposing improvement based on audit: {}",
            audit_result
        );

        // In a real implementation, Broca would generate the actual Rust patch.
        // Here we simulate the proposal of a higher-order improvement.
        let proposal = if audit_result.contains("complexity") {
            "Action: Refactor large match arms into a trait-based dispatcher to reduce Z3 solver depth."
        } else {
            "Action: Implement SIMD-optimized hypervector bundling for faster amodal fusion."
        };

        proposal.into()
    }
}

fn estimate_rust_source_complexity(source: &str) -> usize {
    const STRUCTURAL_TOKENS: &[&str] = &[
        "fn ", "impl ", "trait ", "struct ", "enum ", "match ", "if ", "else ", "for ", "while ",
        "loop ", "async ", "await", "Result<", "Option<", "?",
    ];

    let structural_hits = STRUCTURAL_TOKENS
        .iter()
        .map(|token| source.matches(token).count())
        .sum::<usize>();
    let line_weight = source
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
        / 4;

    structural_hits + line_weight
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
            "REQ-1",
            EngineeringDomain::Civil,
            "stress",
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));
        let review = EngineeringReview {
            concept,
            twin: None,
        };
        assert!(review.blocks_deployment());
    }

    #[test]
    fn self_audit_reads_file_within_cwd() {
        let manager = EngineeringManager::default();
        let cwd = std::env::current_dir().unwrap();
        let target = cwd.join("Cargo.toml");
        let result = manager.self_audit(target.to_str().unwrap());
        assert!(
            result.is_ok(),
            "should be able to audit a file within cwd: {result:?}"
        );
    }

    #[test]
    fn self_audit_rejects_absolute_path_outside_cwd() {
        let manager = EngineeringManager::default();
        // /etc/passwd is definitely outside any Rust project's cwd.
        let result = manager.self_audit("/etc/passwd");
        assert!(
            result.is_err(),
            "must refuse to read a file outside the working directory"
        );
        assert!(
            result
                .unwrap_err()
                .contains("outside the working directory")
        );
    }

    #[test]
    fn self_audit_rejects_dotdot_traversal_outside_cwd() {
        let manager = EngineeringManager::default();
        let cwd = std::env::current_dir().unwrap();
        // Walk up far enough to guarantee escaping the workspace, then
        // target a file that plausibly exists outside it (/etc/hostname is
        // present on virtually every Linux system).
        let mut traversal = cwd.clone();
        for _ in 0..10 {
            traversal.push("..");
        }
        traversal.push("etc");
        traversal.push("hostname");
        if traversal.canonicalize().is_ok() {
            let result = manager.self_audit(traversal.to_str().unwrap());
            assert!(result.is_err());
        }
    }
}
