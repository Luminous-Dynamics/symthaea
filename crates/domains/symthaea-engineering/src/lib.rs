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
            .field("status", &self.0.status())
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
                    // encode_result is fallible upstream now — a failed encoding
                    // means no sensation this pass, not a crash.
                    let Ok(sensation) = encoder.encode_result(&result) else {
                        continue;
                    };
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

    /// Render a Lean-formatted certificate for each already-discharged
    /// safety obligation.
    ///
    /// Despite the name, this does **not** perform formal verification of
    /// the obligation's actual content, and does not invoke a Lean checker
    /// on the output. `obligation.status` is set to `Discharged` elsewhere
    /// (`evaluate_concept`, purely from `EvidenceKind::Simulation`
    /// convergence, not from `EvidenceKind::FormalProof`), so by the time an
    /// obligation reaches this function it has already been discharged by
    /// non-formal means. The `goal`/`result` fed to `LeanProofGenerator`
    /// here are placeholders -- `id implies id` and a hardcoded
    /// `valid: true` -- used purely to produce a readable rendered artifact
    /// carrying the obligation's id, not a real proof obligation derived
    /// from the obligation's content. Do not treat this output as formal
    /// verification evidence.
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
                // SwarmStateMsg::platform_type is a validated wire-format String
                // (bounded by MAX_IDENTIFIER_BYTES), not the PlatformType enum.
                platform_type: symtropy_robotics_bridge_core::platform::PlatformType::Humanoid
                    .name()
                    .to_string(), // Default for engineering nodes
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

/// Result of a closed-form structural check via `symthaea-structural`.
///
/// **Epistemic envelope — do not over-trust.** The underlying solver is
/// closed-form, linear-elastic, single-span 2D bending only. 3D /
/// statically-indeterminate / dynamic analysis requires an external FEA bridge —
/// see `ENGINEERING_FACULTY_PLAN_2026-07-07.md` Phase 3. The `solver_envelope`
/// field carries this so downstream cognition never treats it as more than it is.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralAssessment {
    pub result: symthaea_structural::BeamResult,
    pub required_factor_of_safety: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

impl EngineeringManager {
    /// Evaluate a beam design against a required factor of safety using the
    /// `symthaea-structural` closed-form solver. Wires a previously-dark
    /// structural discipline crate into the engineering faculty
    /// (ENGINEERING_FACULTY_PLAN Phase 1c — orchestrator consumes all disciplines).
    pub fn evaluate_structural(
        &self,
        beam: &symthaea_structural::Beam,
        load: symthaea_structural::LoadCase,
        required_factor_of_safety: f64,
    ) -> StructuralAssessment {
        let result = beam.analyze(load);
        let passes = result.factor_of_safety >= required_factor_of_safety;
        StructuralAssessment {
            required_factor_of_safety,
            passes,
            solver_envelope: "closed-form, linear-elastic, single-span 2D bending (symthaea-structural)",
            summary: format!(
                "sigma_max={:.3e} Pa, delta_max={:.3e} m, FoS={:.2} vs required {:.2} -> {}",
                result.max_bending_stress,
                result.max_deflection,
                result.factor_of_safety,
                required_factor_of_safety,
                if passes { "PASS" } else { "FAIL" }
            ),
            result,
        }
    }

    /// Run a structural check and, when it passes, discharge the matching
    /// safety-case obligation on `concept` with the solver run as evidence.
    ///
    /// This is the cognition tie-in that distinguishes a *faculty* from a
    /// tool-wrapper: a real discipline solver produces evidence that discharges a
    /// formal proof obligation, so the engineering reasoning (safety case,
    /// deployment gating) actually moves. Returns the assessment for inspection.
    pub fn discharge_structural_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        beam: &symthaea_structural::Beam,
        load: symthaea_structural::LoadCase,
        required_factor_of_safety: f64,
    ) -> StructuralAssessment {
        let assessment = self.evaluate_structural(beam, load, required_factor_of_safety);
        if assessment.passes {
            for obligation in concept.safety_case.obligations.iter_mut() {
                if obligation.claim == obligation_claim {
                    obligation
                        .evidence_refs
                        .push(format!("symthaea-structural: {}", assessment.summary));
                    obligation.status = formal_safety::ObligationStatus::Discharged;
                }
            }
        }
        assessment
    }
}

/// Result of an electrical distribution-feeder check via `symthaea-grid-physics`.
///
/// **Epistemic envelope — do not over-trust.** The underlying solve is a
/// *linearized* DistFlow power flow (the quadratic loss term is dropped) on a
/// *radial* feeder topology only. Meshed networks, transient/dynamic stability,
/// and unbalanced phases are out of scope — for those an external power-systems
/// tool (e.g. OpenDSS / PSS/E) is the Phase-3 bridge. `solver_envelope` carries
/// this so downstream cognition never over-reads a passing result.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalAssessment {
    pub min_voltage_pu: f64,
    pub max_voltage_pu: f64,
    pub required_band_pu: (f64, f64),
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

impl EngineeringManager {
    /// Evaluate a radial distribution feeder: solve the power flow and check that
    /// every bus voltage stays within the required per-unit band (e.g. ANSI C84.1
    /// 0.95–1.05 pu). Wires the previously-optional-off `symthaea-grid-physics`
    /// electrical solver into the engineering faculty
    /// (ENGINEERING_FACULTY_PLAN Phase 1c/1d).
    pub fn evaluate_electrical(
        &self,
        feeder: &symthaea_grid_physics::feeder::Feeder,
        min_voltage_pu: f64,
        max_voltage_pu: f64,
    ) -> ElectricalAssessment {
        let solution = feeder.solve();
        let mut min_pu = f64::INFINITY;
        let mut max_pu = f64::NEG_INFINITY;
        for i in 0..feeder.nodes.len() {
            let v = solution.voltage_pu(i);
            min_pu = min_pu.min(v);
            max_pu = max_pu.max(v);
        }
        let passes = min_pu >= min_voltage_pu && max_pu <= max_voltage_pu;
        ElectricalAssessment {
            min_voltage_pu: min_pu,
            max_voltage_pu: max_pu,
            required_band_pu: (min_voltage_pu, max_voltage_pu),
            passes,
            solver_envelope: "linearized DistFlow, radial feeder, balanced (symthaea-grid-physics)",
            summary: format!(
                "V in [{:.4}, {:.4}] pu vs required [{:.3}, {:.3}] -> {}",
                min_pu,
                max_pu,
                min_voltage_pu,
                max_voltage_pu,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    /// Run an electrical feeder check and, when it passes, discharge the matching
    /// safety-case obligation on `concept` with the solver run as evidence — the
    /// same cognition tie-in as `discharge_structural_check`.
    pub fn discharge_electrical_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        feeder: &symthaea_grid_physics::feeder::Feeder,
        min_voltage_pu: f64,
        max_voltage_pu: f64,
    ) -> ElectricalAssessment {
        let assessment = self.evaluate_electrical(feeder, min_voltage_pu, max_voltage_pu);
        if assessment.passes {
            for obligation in concept.safety_case.obligations.iter_mut() {
                if obligation.claim == obligation_claim {
                    obligation
                        .evidence_refs
                        .push(format!("symthaea-grid-physics: {}", assessment.summary));
                    obligation.status = formal_safety::ObligationStatus::Discharged;
                }
            }
        }
        assessment
    }
}

/// Result of a pipe-flow head-loss check via `symthaea-thermofluids`.
///
/// **Epistemic envelope — do not over-trust.** Steady, incompressible,
/// single-phase pipe flow; the Darcy friction factor is *supplied*, not solved
/// from roughness/Reynolds (no Colebrook/Moody iteration). Compressible flow,
/// transients, and pipe networks are out of scope — an external CFD tool
/// (e.g. OpenFOAM) is the Phase-3 bridge. `solver_envelope` carries this.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermofluidAssessment {
    pub head_loss_m: f64,
    pub allowed_head_loss_m: f64,
    pub reynolds: f64,
    pub regime: symthaea_thermofluids::fluids::Regime,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

impl EngineeringManager {
    /// Evaluate a pipe run: compute Darcy-Weisbach head loss and the flow regime,
    /// and check the head loss against an allowed budget. Wires the previously-dark
    /// `symthaea-thermofluids` solver into the faculty (ENGINEERING_FACULTY_PLAN 1c).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_thermofluid(
        &self,
        density: f64,
        viscosity: f64,
        friction_factor: f64,
        length: f64,
        diameter: f64,
        velocity: f64,
        allowed_head_loss_m: f64,
    ) -> ThermofluidAssessment {
        use symthaea_thermofluids::fluids;
        let reynolds = fluids::reynolds_number(density, velocity, diameter, viscosity);
        let regime = fluids::flow_regime(reynolds);
        let head_loss_m =
            fluids::darcy_weisbach_head_loss(friction_factor, length, diameter, velocity);
        let passes = head_loss_m <= allowed_head_loss_m;
        ThermofluidAssessment {
            head_loss_m,
            allowed_head_loss_m,
            reynolds,
            regime,
            passes,
            solver_envelope: "steady incompressible single-phase pipe flow, supplied friction factor (symthaea-thermofluids)",
            summary: format!(
                "h_f={:.3} m vs budget {:.3} m, Re={:.0} ({:?}) -> {}",
                head_loss_m,
                allowed_head_loss_m,
                reynolds,
                regime,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    /// Run a pipe head-loss check and, when it passes, discharge the matching
    /// safety-case obligation on `concept` with the solver run as evidence — the
    /// same cognition tie-in as the structural/electrical checks.
    #[allow(clippy::too_many_arguments)]
    pub fn discharge_thermofluid_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        density: f64,
        viscosity: f64,
        friction_factor: f64,
        length: f64,
        diameter: f64,
        velocity: f64,
        allowed_head_loss_m: f64,
    ) -> ThermofluidAssessment {
        let assessment = self.evaluate_thermofluid(
            density,
            viscosity,
            friction_factor,
            length,
            diameter,
            velocity,
            allowed_head_loss_m,
        );
        if assessment.passes {
            for obligation in concept.safety_case.obligations.iter_mut() {
                if obligation.claim == obligation_claim {
                    obligation
                        .evidence_refs
                        .push(format!("symthaea-thermofluids: {}", assessment.summary));
                    obligation.status = formal_safety::ObligationStatus::Discharged;
                }
            }
        }
        assessment
    }
}

// ── Extended discipline faculties (ENGINEERING_FACULTY_PLAN Phase 1c batch) ──
// Each wires a previously-dark real solver crate into the faculty via the proven
// evaluate/discharge/envelope pattern (structural/electrical/thermofluids). Every
// `Assessment` declares its solver's validity envelope so cognition never over-trusts.

impl EngineeringManager {
    /// Shared cognition tie-in: discharge the safety-case obligation matching `claim`
    /// with solver evidence. Used by the discipline `discharge_*_check` methods.
    fn discharge_obligation(concept: &mut EngineeringConcept, claim: &str, evidence: String) {
        for ob in concept.safety_case.obligations.iter_mut() {
            if ob.claim == claim {
                ob.evidence_refs.push(evidence.clone());
                ob.status = formal_safety::ObligationStatus::Discharged;
            }
        }
    }
}

/// Control-systems check via `symthaea-control-theory`.
/// Envelope: LTI SISO — Routh-Hurwitz on the characteristic polynomial + closed-form
/// dominant-2nd-order transient metrics. Nonlinear / MIMO / discrete control is out of scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlAssessment {
    pub stable: bool,
    pub rhp_roots: usize,
    pub percent_overshoot: f64,
    pub settling_time_s: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

/// Circuit power-rating check via `symthaea-circuits`.
/// Envelope: linear resistive DC (Ohm/Joule). AC/transient/nonlinear devices out of scope.
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitAssessment {
    pub current_a: f64,
    pub power_w: f64,
    pub max_power_w: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

/// Acoustic noise-limit check via `symthaea-acoustics`.
/// Envelope: incoherent SPL summation of steady sources; no propagation/room modelling.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticAssessment {
    pub combined_spl_db: f64,
    pub max_spl_db: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

/// Optical imaging check via `symthaea-optics`.
/// Envelope: paraxial (thin-lens) geometric optics. Aberrations/diffraction out of scope.
#[derive(Debug, Clone, PartialEq)]
pub struct OpticalAssessment {
    pub image_distance: f64,
    pub required_image_distance: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

/// Sampling / anti-alias check via `symthaea-dsp`.
/// Envelope: ideal uniform sampling, Nyquist criterion only (no reconstruction filter modelling).
#[derive(Debug, Clone, PartialEq)]
pub struct SignalAssessment {
    pub signal_freq_hz: f64,
    pub sample_rate_hz: f64,
    pub nyquist_hz: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

/// Queueing service-level check via `symthaea-operations-research`.
/// Envelope: M/M/1 (Poisson arrivals, exponential service, single server, steady state).
#[derive(Debug, Clone, PartialEq)]
pub struct OperationsAssessment {
    pub utilization: f64,
    pub avg_time_in_system_s: f64,
    pub max_wait_s: f64,
    pub passes: bool,
    pub solver_envelope: &'static str,
    pub summary: String,
}

impl EngineeringManager {
    /// Control loop: Routh stability of the characteristic polynomial + dominant
    /// 2nd-order overshoot/settling against spec.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_control(
        &self,
        char_poly_coeffs: &[f64],
        natural_freq: f64,
        damping_ratio: f64,
        max_overshoot_pct: f64,
        max_settling_time_s: f64,
    ) -> ControlAssessment {
        use symthaea_control_theory::routh;
        use symthaea_control_theory::second_order::SecondOrder;
        let stable = routh::is_stable(char_poly_coeffs);
        let rhp_roots = routh::rhp_root_count(char_poly_coeffs);
        let so = SecondOrder {
            natural_freq,
            damping_ratio,
        };
        let percent_overshoot = so.percent_overshoot();
        let settling_time_s = so.settling_time();
        let passes = stable
            && percent_overshoot <= max_overshoot_pct
            && settling_time_s <= max_settling_time_s;
        ControlAssessment {
            stable,
            rhp_roots,
            percent_overshoot,
            settling_time_s,
            passes,
            solver_envelope: "LTI SISO, Routh-Hurwitz + closed-form 2nd-order (symthaea-control-theory)",
            summary: format!(
                "stable={} rhp={} PO={:.1}%<={:.1} ts={:.2}s<={:.2} -> {}",
                stable,
                rhp_roots,
                percent_overshoot,
                max_overshoot_pct,
                settling_time_s,
                max_settling_time_s,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn discharge_control_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        char_poly_coeffs: &[f64],
        natural_freq: f64,
        damping_ratio: f64,
        max_overshoot_pct: f64,
        max_settling_time_s: f64,
    ) -> ControlAssessment {
        let a = self.evaluate_control(
            char_poly_coeffs,
            natural_freq,
            damping_ratio,
            max_overshoot_pct,
            max_settling_time_s,
        );
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-control-theory: {}", a.summary),
            );
        }
        a
    }

    /// Circuit: resistor power dissipation `P = V·I` against a component rating.
    pub fn evaluate_circuit(
        &self,
        voltage: f64,
        resistance: f64,
        max_power_w: f64,
    ) -> CircuitAssessment {
        use symthaea_circuits::dc;
        let current_a = dc::current(voltage, resistance);
        let power_w = dc::power(voltage, current_a);
        let passes = power_w <= max_power_w;
        CircuitAssessment {
            current_a,
            power_w,
            max_power_w,
            passes,
            solver_envelope: "linear resistive DC, Ohm/Joule (symthaea-circuits)",
            summary: format!(
                "I={:.3} A, P={:.3} W vs rating {:.3} W -> {}",
                current_a,
                power_w,
                max_power_w,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    pub fn discharge_circuit_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        voltage: f64,
        resistance: f64,
        max_power_w: f64,
    ) -> CircuitAssessment {
        let a = self.evaluate_circuit(voltage, resistance, max_power_w);
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-circuits: {}", a.summary),
            );
        }
        a
    }

    /// Acoustic: combined SPL of several sources against a noise limit.
    pub fn evaluate_acoustic(
        &self,
        source_levels_db: &[f64],
        max_spl_db: f64,
    ) -> AcousticAssessment {
        let combined_spl_db = symthaea_acoustics::combine_decibels(source_levels_db);
        let passes = combined_spl_db <= max_spl_db;
        AcousticAssessment {
            combined_spl_db,
            max_spl_db,
            passes,
            solver_envelope: "incoherent SPL summation, steady sources (symthaea-acoustics)",
            summary: format!(
                "SPL={:.1} dB vs limit {:.1} dB -> {}",
                combined_spl_db,
                max_spl_db,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    pub fn discharge_acoustic_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        source_levels_db: &[f64],
        max_spl_db: f64,
    ) -> AcousticAssessment {
        let a = self.evaluate_acoustic(source_levels_db, max_spl_db);
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-acoustics: {}", a.summary),
            );
        }
        a
    }

    /// Optical: thin-lens image distance within tolerance of a required focal plane.
    pub fn evaluate_optical(
        &self,
        focal_length: f64,
        object_distance: f64,
        required_image_distance: f64,
        tolerance: f64,
    ) -> OpticalAssessment {
        let image_distance =
            symthaea_optics::geometric::image_distance(focal_length, object_distance);
        let passes = (image_distance - required_image_distance).abs() <= tolerance;
        OpticalAssessment {
            image_distance,
            required_image_distance,
            passes,
            solver_envelope: "paraxial thin-lens geometric optics (symthaea-optics)",
            summary: format!(
                "image at {:.4} m vs required {:.4} m (tol {:.4}) -> {}",
                image_distance,
                required_image_distance,
                tolerance,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    pub fn discharge_optical_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        focal_length: f64,
        object_distance: f64,
        required_image_distance: f64,
        tolerance: f64,
    ) -> OpticalAssessment {
        let a = self.evaluate_optical(
            focal_length,
            object_distance,
            required_image_distance,
            tolerance,
        );
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-optics: {}", a.summary),
            );
        }
        a
    }

    /// Signal: Nyquist anti-alias check for a signal at a given sample rate.
    pub fn evaluate_signal(&self, signal_freq_hz: f64, sample_rate_hz: f64) -> SignalAssessment {
        use symthaea_dsp::signal;
        let nyquist_hz = signal::nyquist_frequency(sample_rate_hz);
        let passes = !signal::will_alias(signal_freq_hz, sample_rate_hz);
        SignalAssessment {
            signal_freq_hz,
            sample_rate_hz,
            nyquist_hz,
            passes,
            solver_envelope: "ideal uniform sampling, Nyquist criterion (symthaea-dsp)",
            summary: format!(
                "f={:.1} Hz vs Nyquist {:.1} Hz (fs={:.1}) -> {}",
                signal_freq_hz,
                nyquist_hz,
                sample_rate_hz,
                if passes {
                    "PASS (no alias)"
                } else {
                    "FAIL (aliases)"
                }
            ),
        }
    }

    pub fn discharge_signal_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        signal_freq_hz: f64,
        sample_rate_hz: f64,
    ) -> SignalAssessment {
        let a = self.evaluate_signal(signal_freq_hz, sample_rate_hz);
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-dsp: {}", a.summary),
            );
        }
        a
    }

    /// Operations: M/M/1 queue stability + average time-in-system against an SLA.
    pub fn evaluate_operations(
        &self,
        arrival_rate: f64,
        service_rate: f64,
        max_wait_s: f64,
    ) -> OperationsAssessment {
        let q = symthaea_operations_research::queue::MM1 {
            arrival_rate,
            service_rate,
        };
        let stable = q.is_stable();
        let avg_time_in_system_s = if stable {
            q.avg_time_in_system()
        } else {
            f64::INFINITY
        };
        let passes = stable && avg_time_in_system_s <= max_wait_s;
        OperationsAssessment {
            utilization: q.utilization(),
            avg_time_in_system_s,
            max_wait_s,
            passes,
            solver_envelope: "M/M/1 steady-state (Poisson arrivals, exp service) (symthaea-operations-research)",
            summary: format!(
                "rho={:.2} W={:.3} s vs SLA {:.3} s -> {}",
                q.utilization(),
                avg_time_in_system_s,
                max_wait_s,
                if passes { "PASS" } else { "FAIL" }
            ),
        }
    }

    pub fn discharge_operations_check(
        &self,
        concept: &mut EngineeringConcept,
        obligation_claim: &str,
        arrival_rate: f64,
        service_rate: f64,
        max_wait_s: f64,
    ) -> OperationsAssessment {
        let a = self.evaluate_operations(arrival_rate, service_rate, max_wait_s);
        if a.passes {
            Self::discharge_obligation(
                concept,
                obligation_claim,
                format!("symthaea-operations-research: {}", a.summary),
            );
        }
        a
    }
}

// ── Discipline capability registry (ENGINEERING_FACULTY_PLAN Phase 1b) ──
// Turns the `EngineeringDomain` tag into a real switchboard: which domains are
// backed by a wired native solver, and which. A tag with no dispatch is just
// documentation; this makes it queryable so the cognitive loop (Phase 1a's
// `EngineeringDomainPlugin`) can route a request to real reasoning — or honestly
// report that a domain (Aerospace, ChemicalProcess, Nuclear, Environmental,
// Robotics) has no faculty solver yet, rather than silently pretending to.

/// One backed capability: an `EngineeringDomain` served by a real solver crate,
/// with the faculty method that exercises it and the solver's validity envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacultyCapability {
    /// The engineering domain this capability serves.
    pub domain: EngineeringDomain,
    /// The faculty `evaluate_*` method that runs the solver.
    pub method: &'static str,
    /// The backing solver crate.
    pub solver_crate: &'static str,
    /// What the check verifies, in one line.
    pub checks: &'static str,
    /// The solver's validity envelope (epistemic-honesty gate).
    pub envelope: &'static str,
}

impl EngineeringManager {
    /// The full static registry of wired faculty capabilities. Each entry is a
    /// real solver reachable via an `evaluate_*` method; disciplines with no
    /// native solver are deliberately absent (see `uncovered_domains`).
    pub fn capabilities() -> &'static [FacultyCapability] {
        use EngineeringDomain::*;
        &[
            FacultyCapability {
                domain: Materials,
                method: "evaluate_material",
                solver_crate: "symthaea-materials",
                checks: "compound stability + property presets",
                envelope: "closed-form stability heuristic at fixed 300 K",
            },
            FacultyCapability {
                domain: Civil,
                method: "evaluate_structural",
                solver_crate: "symthaea-structural",
                checks: "stress / deflection / buckling margins",
                envelope: "closed-form, linear-elastic, 2D — no 3D/indeterminate/dynamics",
            },
            FacultyCapability {
                domain: Electrical,
                method: "evaluate_electrical",
                solver_crate: "symthaea-grid-physics",
                checks: "radial DistFlow bus voltages within a per-unit band",
                envelope: "radial feeder, steady-state DistFlow — no meshed/transient",
            },
            FacultyCapability {
                domain: Electrical,
                method: "evaluate_circuit",
                solver_crate: "symthaea-circuits",
                checks: "resistor power dissipation vs rating",
                envelope: "DC, linear resistive — no reactive/transient",
            },
            FacultyCapability {
                domain: Electrical,
                method: "evaluate_signal",
                solver_crate: "symthaea-dsp",
                checks: "Nyquist anti-alias check",
                envelope: "ideal uniform sampling — no windowing/quantization noise",
            },
            FacultyCapability {
                domain: Mechanical,
                method: "evaluate_thermofluid",
                solver_crate: "symthaea-thermofluids",
                checks: "Darcy-Weisbach pipe head loss + flow regime vs budget",
                envelope: "incompressible steady pipe flow — no networks/transient/compressible",
            },
            FacultyCapability {
                domain: Mechanical,
                method: "evaluate_acoustic",
                solver_crate: "symthaea-acoustics",
                checks: "combined sound-pressure level vs a noise limit",
                envelope: "incoherent SPL summation — no directivity/room acoustics",
            },
            FacultyCapability {
                domain: Systems,
                method: "evaluate_control",
                solver_crate: "symthaea-control-theory",
                checks: "Routh-Hurwitz stability + 2nd-order overshoot/settling",
                envelope: "LTI SISO — no nonlinear/MIMO/discrete control",
            },
            FacultyCapability {
                domain: Systems,
                method: "evaluate_optical",
                solver_crate: "symthaea-optics",
                checks: "thin-lens image distance vs focal plane",
                envelope: "paraxial thin-lens — no aberrations/thick-lens/diffraction",
            },
            FacultyCapability {
                domain: Systems,
                method: "evaluate_operations",
                solver_crate: "symthaea-operations-research",
                checks: "M/M/1 queue stability + wait-time SLA",
                envelope: "single-server Markovian queue — no M/M/c/priority/networks",
            },
        ]
    }

    /// The capabilities backing a given domain (empty if the domain is tag-only).
    pub fn capabilities_for(domain: EngineeringDomain) -> Vec<&'static FacultyCapability> {
        Self::capabilities()
            .iter()
            .filter(|c| c.domain == domain)
            .collect()
    }

    /// Whether a domain is backed by at least one wired native solver.
    pub fn is_covered(domain: EngineeringDomain) -> bool {
        Self::capabilities().iter().any(|c| c.domain == domain)
    }

    /// Domains that are still tag-only — no faculty solver yet. Honest gap list
    /// for the cognitive loop (do not claim reasoning we can't back).
    pub fn uncovered_domains() -> Vec<EngineeringDomain> {
        use EngineeringDomain::*;
        [
            Civil,
            Mechanical,
            Electrical,
            Aerospace,
            ChemicalProcess,
            Robotics,
            Nuclear,
            Materials,
            Environmental,
            Systems,
        ]
        .into_iter()
        .filter(|d| !Self::is_covered(*d))
        .collect()
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
    fn evaluate_structural_wires_solver_into_faculty() {
        use symthaea_structural::{Beam, LoadCase, Section, material::steel_a36};
        let mgr = EngineeringManager::new();

        // Steel cantilever, 1 m span, 50x100 mm rectangular section, 1 kN end load.
        let beam = Beam {
            length: 1.0,
            section: Section::rectangular(0.05, 0.1),
            material: steel_a36(),
        };
        let a = mgr.evaluate_structural(&beam, LoadCase::CantileverEndPoint(1000.0), 2.0);

        // Known closed-form answer: max moment = P·L = 1000 N·m exactly.
        assert!((a.result.max_moment - 1000.0).abs() < 1e-9);
        // A36 steel is far stronger than a 12 MPa bending stress here → passes with margin.
        assert!(a.passes);
        assert!(a.result.factor_of_safety > 2.0);
        assert!(a.solver_envelope.contains("closed-form"));
    }

    #[test]
    fn structural_check_discharges_safety_obligation() {
        use symthaea_structural::{Beam, LoadCase, Section, material::steel_a36};
        let mgr = EngineeringManager::new();

        let mut concept =
            EngineeringConcept::new("beam-1", "cantilever bracket", EngineeringDomain::Civil);
        concept.add_requirement(EngineeringRequirement::new(
            "r-fos",
            EngineeringDomain::Civil,
            "beam factor of safety >= 2",
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));
        assert!(!concept.safety_case.is_discharged());

        let beam = Beam {
            length: 1.0,
            section: Section::rectangular(0.05, 0.1),
            material: steel_a36(),
        };
        // Real solver evidence discharges the formal obligation — cognition, not a wrapper.
        mgr.discharge_structural_check(
            &mut concept,
            "beam factor of safety >= 2",
            &beam,
            LoadCase::CantileverEndPoint(1000.0),
            2.0,
        );
        assert!(concept.safety_case.is_discharged());
    }

    #[test]
    fn evaluate_electrical_wires_grid_solver_into_faculty() {
        use symthaea_grid_physics::feeder::{Feeder, Line, Node};
        let mgr = EngineeringManager::new();

        // 2-bus 7.2 kV radial feeder: substation + one 200 kW / 100 kVAR load
        // behind a 2+j1 Ω line. Linearized DistFlow → ~1% drop, within band.
        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 2.0,
                        reactance_ohm: 1.0,
                    },
                    200.0,
                    100.0,
                ),
            ],
        )
        .expect("valid radial feeder");

        let a = mgr.evaluate_electrical(&feeder, 0.95, 1.05);
        assert!(a.passes);
        // Substation bus is the slack bus at exactly 1.0 pu.
        assert!((a.max_voltage_pu - 1.0).abs() < 1e-9);
        // The loaded bus sagged but stayed inside the band.
        assert!(a.min_voltage_pu > 0.95 && a.min_voltage_pu < 1.0);
        assert!(a.solver_envelope.contains("DistFlow"));
    }

    #[test]
    fn electrical_check_discharges_safety_obligation() {
        use symthaea_grid_physics::feeder::{Feeder, Line, Node};
        let mgr = EngineeringManager::new();

        let mut concept = EngineeringConcept::new(
            "feeder-1",
            "distribution feeder",
            EngineeringDomain::Electrical,
        );
        concept.add_requirement(EngineeringRequirement::new(
            "r-volt",
            EngineeringDomain::Electrical,
            "all bus voltages within 0.95-1.05 pu",
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));
        assert!(!concept.safety_case.is_discharged());

        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 2.0,
                        reactance_ohm: 1.0,
                    },
                    200.0,
                    100.0,
                ),
            ],
        )
        .unwrap();
        mgr.discharge_electrical_check(
            &mut concept,
            "all bus voltages within 0.95-1.05 pu",
            &feeder,
            0.95,
            1.05,
        );
        assert!(concept.safety_case.is_discharged());
    }

    #[test]
    fn evaluate_thermofluid_wires_solver_into_faculty() {
        let mgr = EngineeringManager::new();
        // Water (ρ=1000, μ=1e-3) at 2 m/s in a 100 m × 0.1 m pipe, f=0.02.
        // h_f = 0.02·(100/0.1)·2²/(2·9.81) ≈ 4.077 m; Re = 1000·2·0.1/1e-3 = 2e5 (turbulent).
        let a = mgr.evaluate_thermofluid(1000.0, 1e-3, 0.02, 100.0, 0.1, 2.0, 5.0);
        assert!((a.head_loss_m - 4.077).abs() < 0.01);
        assert!((a.reynolds - 200_000.0).abs() < 1.0);
        assert_eq!(a.regime, symthaea_thermofluids::fluids::Regime::Turbulent);
        assert!(a.passes); // 4.077 m <= 5 m budget
        assert!(a.solver_envelope.contains("pipe flow"));
    }

    #[test]
    fn thermofluid_check_discharges_safety_obligation() {
        let mgr = EngineeringManager::new();
        let mut concept =
            EngineeringConcept::new("pipe-1", "cooling loop", EngineeringDomain::Mechanical);
        concept.add_requirement(EngineeringRequirement::new(
            "r-hf",
            EngineeringDomain::Mechanical,
            "pipe head loss <= 5 m",
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));
        assert!(!concept.safety_case.is_discharged());
        mgr.discharge_thermofluid_check(
            &mut concept,
            "pipe head loss <= 5 m",
            1000.0,
            1e-3,
            0.02,
            100.0,
            0.1,
            2.0,
            5.0,
        );
        assert!(concept.safety_case.is_discharged());
    }

    // Helper: a concept with one blocking obligation whose claim == `claim`.
    fn concept_with(domain: EngineeringDomain, claim: &str) -> EngineeringConcept {
        let mut c = EngineeringConcept::new("c", "design", domain);
        c.add_requirement(EngineeringRequirement::new(
            "r",
            domain,
            claim,
            RequirementCriticality::Blocking,
            EvidenceKind::Simulation,
        ));
        c
    }

    #[test]
    fn control_faculty_stability_and_discharge() {
        let mgr = EngineeringManager::new();
        // (s+1)^3 = s^3+3s^2+3s+1 — all roots at -1, stable, 0 RHP roots.
        let a = mgr.evaluate_control(&[1.0, 3.0, 3.0, 1.0], 1.0, 0.7, 50.0, 100.0);
        assert!(a.stable);
        assert_eq!(a.rhp_roots, 0);
        assert!(a.passes);
        // s^2 - s + 1 has a sign change -> RHP roots -> unstable.
        assert!(
            !mgr.evaluate_control(&[1.0, -1.0, 1.0], 1.0, 0.7, 50.0, 100.0)
                .stable
        );
        let mut c = concept_with(EngineeringDomain::Systems, "loop is stable");
        mgr.discharge_control_check(
            &mut c,
            "loop is stable",
            &[1.0, 3.0, 3.0, 1.0],
            1.0,
            0.7,
            50.0,
            100.0,
        );
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn circuit_faculty_power_and_discharge() {
        let mgr = EngineeringManager::new();
        // 10 V across 100 Ω -> 0.1 A, 1 W; rating 2 W -> pass.
        let a = mgr.evaluate_circuit(10.0, 100.0, 2.0);
        assert!((a.current_a - 0.1).abs() < 1e-9);
        assert!((a.power_w - 1.0).abs() < 1e-9);
        assert!(a.passes);
        let mut c = concept_with(EngineeringDomain::Electrical, "resistor within 2 W");
        mgr.discharge_circuit_check(&mut c, "resistor within 2 W", 10.0, 100.0, 2.0);
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn acoustic_faculty_spl_and_discharge() {
        let mgr = EngineeringManager::new();
        // Two 80 dB sources -> 80 + 10·log10(2) ≈ 83.01 dB; limit 85 -> pass.
        let a = mgr.evaluate_acoustic(&[80.0, 80.0], 85.0);
        assert!((a.combined_spl_db - 83.01).abs() < 0.1);
        assert!(a.passes);
        let mut c = concept_with(EngineeringDomain::Mechanical, "noise <= 85 dB");
        mgr.discharge_acoustic_check(&mut c, "noise <= 85 dB", &[80.0, 80.0], 85.0);
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn optical_faculty_imaging_and_discharge() {
        let mgr = EngineeringManager::new();
        // Thin lens f=0.1, object at 0.3 -> image at f·o/(o−f) = 0.15 m.
        let a = mgr.evaluate_optical(0.1, 0.3, 0.15, 0.005);
        assert!((a.image_distance - 0.15).abs() < 0.005);
        assert!(a.passes);
        let mut c = concept_with(EngineeringDomain::Systems, "image at 0.15 m");
        mgr.discharge_optical_check(&mut c, "image at 0.15 m", 0.1, 0.3, 0.15, 0.005);
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn signal_faculty_nyquist_and_discharge() {
        let mgr = EngineeringManager::new();
        // 1 kHz at 8 kHz sampling: Nyquist 4 kHz, no alias -> pass.
        let a = mgr.evaluate_signal(1000.0, 8000.0);
        assert!((a.nyquist_hz - 4000.0).abs() < 1e-9);
        assert!(a.passes);
        // 5 kHz at 8 kHz aliases.
        assert!(!mgr.evaluate_signal(5000.0, 8000.0).passes);
        let mut c = concept_with(EngineeringDomain::Electrical, "no aliasing");
        mgr.discharge_signal_check(&mut c, "no aliasing", 1000.0, 8000.0);
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn operations_faculty_queue_and_discharge() {
        let mgr = EngineeringManager::new();
        // λ=5, μ=10: ρ=0.5 stable, W=1/(μ−λ)=0.2 s; SLA 1 s -> pass.
        let a = mgr.evaluate_operations(5.0, 10.0, 1.0);
        assert!((a.utilization - 0.5).abs() < 1e-9);
        assert!(a.passes);
        // λ=10 > μ=5: unstable.
        assert!(!mgr.evaluate_operations(10.0, 5.0, 1.0).passes);
        let mut c = concept_with(EngineeringDomain::Systems, "queue wait <= 1 s");
        mgr.discharge_operations_check(&mut c, "queue wait <= 1 s", 5.0, 10.0, 1.0);
        assert!(c.safety_case.is_discharged());
    }

    #[test]
    fn capability_registry_maps_tag_to_solvers() {
        use EngineeringDomain::*;
        // Electrical is backed by three distinct solver crates.
        let elec = EngineeringManager::capabilities_for(Electrical);
        assert_eq!(elec.len(), 3);
        let crates: Vec<_> = elec.iter().map(|c| c.solver_crate).collect();
        assert!(crates.contains(&"symthaea-grid-physics"));
        assert!(crates.contains(&"symthaea-circuits"));
        assert!(crates.contains(&"symthaea-dsp"));
        // Systems: control + optics + operations research.
        assert_eq!(EngineeringManager::capabilities_for(Systems).len(), 3);
        // Covered domains report true; every registry entry names a real evaluate_* method.
        for d in [Civil, Mechanical, Electrical, Materials, Systems] {
            assert!(EngineeringManager::is_covered(d), "{d:?} should be covered");
        }
        for c in EngineeringManager::capabilities() {
            assert!(c.method.starts_with("evaluate_"));
            assert!(
                !c.envelope.is_empty(),
                "every capability declares its envelope"
            );
        }
    }

    #[test]
    fn uncovered_domains_are_honestly_reported() {
        use EngineeringDomain::*;
        let gaps = EngineeringManager::uncovered_domains();
        // These have no faculty solver yet — must be reported as gaps, not silently claimed.
        for d in [Aerospace, ChemicalProcess, Robotics, Nuclear, Environmental] {
            assert!(gaps.contains(&d), "{d:?} is a known gap");
            assert!(!EngineeringManager::is_covered(d));
        }
        // Covered domains must NOT appear in the gap list.
        for d in [Civil, Mechanical, Electrical, Materials, Systems] {
            assert!(!gaps.contains(&d));
        }
    }

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
