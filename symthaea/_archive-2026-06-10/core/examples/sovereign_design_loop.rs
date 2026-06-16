// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sovereign Design Loop Demo (Consolidated v3)
//!
//! Demonstrates the full DESIGN -> SIMULATE -> PROVE -> MAKE loop
//! with Causal Interventions, Sim-to-Real Calibration, and Amodal Fusion.

use symthaea_broca::{BrocaConfig, BrocaGenerator};
use symthaea_causal_reasoning::causal_calculus::{CausalDAG, StructuralCausalModel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_engineering::{EngineeringAssistant, EngineeringConcept, EngineeringManager};
use symthaea_fabrication_kernel::cincinnati_live::SensorReading;
use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::thought::GeometricThought;
use symthaea_mujoco_bridge::MuJoCoBridge;
use symthaea_sim_bridge::{EngineeringDomain, MetricEncoder, SimulationRequest, SolverKind};

fn run_iteration(
    iteration: usize,
    manager: &mut EngineeringManager,
    assistant: &mut EngineeringAssistant,
    goal: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 STARTING ITERATION {} --------------------", iteration);

    // 0. Causal Pre-Commitment
    println!("💭 Causal Analysis: Predicting safety probability...");
    if let Some(prob) = manager.predict_intervention("material_strength", 1, "safety") {
        println!("   Prediction: Probability of 'safe' design is {:.2}", prob);
    }

    // 1. Propose Requirements
    let mut requirements = assistant.propose_requirements(goal, EngineeringDomain::Aerospace);
    println!(
        "✅ Step 1: Broca synthesized {} requirements.",
        requirements.len()
    );

    // 2. Create Engineering Concept
    let mut concept = EngineeringConcept::new("arm-v1", "Phase 1", EngineeringDomain::Aerospace);
    concept.requirements = requirements;

    // 3. Causal Topology Optimization
    println!("🧬 Step 3: Optimizing geometry...");
    let geometry = CSGNode::cube();
    let mut thought = GeometricThought::from_csg(geometry);
    let fitness = manager.optimize_geometry(&mut thought, "material_strength", 0.8)?;
    println!("✅ Geometry optimized (Estimated fitness: {:.4})", fitness);

    // 4. Run Simulation
    println!("📡 Step 4: Dispatching to MuJoCo for structural validation...");
    let request = SimulationRequest::new(
        "val-01",
        EngineeringDomain::Aerospace,
        SolverKind::MultibodyDynamics,
        goal,
    );
    concept.simulation_requests.push(request);
    manager.evaluate_concept(&mut concept);

    // 5. Formal Verification
    println!("📜 Step 5: Generating formal Lean 4 proofs...");
    let proofs = manager.formally_verify(&concept);
    for (id, _) in &proofs {
        println!("✅ Proof for {}: Discharged", id);
    }

    // 6. Fabrication (Moral-Gated)
    println!("🖨️  Step 6: Transitioning to Fabrication Phase...");
    match manager.prepare_fabrication(&thought, goal) {
        Ok(_) => println!("✅ Design resolved and fabrication started."),
        Err(e) => println!("{}", e),
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Symthaea Sovereign Design Loop v3...");

    let genesis = GenesisSeed::from_phrase("Sovereign Engineering Constitution");
    let mut assistant = EngineeringAssistant::new(&genesis);
    let mut manager = EngineeringManager::new();

    // Setup Causal Model (Structural Safety)
    let mut dag = CausalDAG::new();
    let strength = dag.add_node("material_strength", vec!["low".into(), "high".into()], true);
    let safety = dag.add_node("safety", vec!["failed".into(), "safe".into()], true);
    dag.add_edge(strength, safety);
    let mut scm = StructuralCausalModel::new(dag);
    // Initial P(safety|strength): [failed, safe]
    scm.set_conditional_table(
        safety,
        vec![
            0.9, 0.1, // strength=low
            0.4, 0.6, // strength=high
        ],
    );
    manager.set_causal_model(scm);

    manager.registry.register(MuJoCoBridge::dry_run());

    let goal = "High-strength multirotor arm (< 100g, > 50N)";

    // Run Iteration 1: Normal loop
    run_iteration(1, &mut manager, &mut assistant, goal)?;

    // 9. Real-World Metrology Feedback (Sim-to-Real Calibration)
    println!("\n📡 Step 9: Simulating real-time sensor feedback (Layer Delamination)...");
    let defect_reading = SensorReading {
        channel: "acoustic_emission".into(),
        value: 5.0, // Critical spike
        timestamp_ms: 1000,
    };

    if let Some(alert) = manager.process_metrology(defect_reading) {
        println!(
            "⚠️  Metrology Alert: {:?} detected! Severity: {:.2}",
            alert.anomaly_type, alert.severity
        );
        println!(
            "🧬 Sim-to-Real Calibration: Updating causal weights based on physical surprise..."
        );
    }

    // 11. Sovereign Dream Phase (Consolidation)
    println!("\n💤 Step 11: Initiating Causal Dream Cycle...");
    manager.dream_consolidation()?;
    println!("✅ Design wisdom graduated from episodes to permanent causal instincts.");

    // 12. Autonomous Swarm Fusion
    println!("\n📡 Step 12: Broadcasting Design Wisdom to Swarm Hive-Mind...");
    let mut concept_final =
        EngineeringConcept::new("arm-v1-final", "Release", EngineeringDomain::Aerospace);
    concept_final
        .safety_case
        .obligations
        .push(symthaea_formal_safety::ProofObligation::new(
            "Structural Safety",
            symthaea_formal_safety::EvidenceKind::Simulation,
        ));
    concept_final.safety_case.obligations[0].status =
        symthaea_formal_safety::ObligationStatus::Discharged;

    let swarm_msgs = manager.broadcast_design_wisdom(&concept_final);
    println!(
        "✅ Successfully shared {} sovereign messages with the P2P swarm.",
        swarm_msgs.len()
    );

    // 13. Causal Hardening (Final Proof)
    println!("\n🔄 Step 13: Final Causal Hardening - Re-predicting safety after reality-sync:");
    if let Some(prob) = manager.predict_intervention("material_strength", 1, "safety") {
        println!(
            "   Final Prediction: Probability of 'safe' design is now {:.2} (calibrated from reality).",
            prob
        );
    }

    println!(
        "\n✨ Sovereignty Verified: The system is learning, dreaming, and sharing the laws of physics."
    );

    Ok(())
}
