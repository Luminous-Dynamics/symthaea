// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference Full Loop Example
//!
//! Demonstrates the complete perception→action→learning cycle integrating:
//! - FEP Active Inference Agent (perception, action selection, TD learning)
//! - Hierarchical Free Energy (multi-level belief updating)
//! - Master Consciousness Equation (dynamic MaxEnt weighting)
//! - Phi computation (integrated information across topologies)
//!
//! This is the primary end-to-end demonstration of the mathematical
//! architecture described in the paper.

use symthaea::consciousness::hierarchical_free_energy::{
    HierarchicalFEConfig, HierarchicalFreeEnergy,
};
use symthaea::consciousness::master_consciousness_equation::{
    ConsciousnessInputs, MasterConsciousnessEquation,
};
use symthaea::consciousness::phi_architecture_search::{
    ArchitectureGenome, DecodedArchitecture, TopologyGene,
};
use symthaea::phi_engine::PhiEngine;
use symthaea_core::hdc::ContinuousHV;
use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, FreeEnergyComponents, Observation,
};

fn main() {
    println!("=== Active Inference Full Loop ===\n");

    // ─── 1. Initialize FEP Agent ───
    let agent_config = ActiveInferenceAgentConfig {
        state_dim: 8,
        obs_dim: 6,
        num_actions: 9,
        inference_iterations: 5,
        belief_learning_rate: 0.1,
        planning_horizon: 3,
        action_temperature: 0.5,
        enable_model_learning: true,
        enable_td_learning: true,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(agent_config);

    // ─── 2. Initialize Hierarchical Free Energy ───
    let hfe_config = HierarchicalFEConfig {
        num_levels: 4,
        state_dim: 8,
        learning_rate: 0.05,
        precision_decay: 0.8,
    };
    let mut hfe = HierarchicalFreeEnergy::new(hfe_config);

    // ─── 3. Initialize Master Consciousness Equation (dynamic weights) ───
    let mut mce = MasterConsciousnessEquation::default();

    // ─── 4. Initialize PhiEngine ───
    let phi_engine = PhiEngine::auto();

    // ─── 5. Build a topology for Phi evaluation ───
    let genome = ArchitectureGenome {
        num_nodes: 6,
        topology_type: TopologyGene::SmallWorld,
        hdc_dim: 256,
        seed: 42,
        ..Default::default()
    };
    let arch = DecodedArchitecture::from_genome(&genome);
    let nodes: Vec<ContinuousHV> = arch.nodes.clone();

    // Compute initial Phi
    let phi_result = phi_engine.compute(&nodes);
    println!(
        "Initial Phi ({:?} topology): {:.4}",
        TopologyGene::SmallWorld,
        phi_result.phi
    );

    // ─── 6. Run perception-action loop ───
    println!("\n--- Perception-Action Loop (20 cycles) ---\n");
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>6} {:>8} {:>10}",
        "Cycle", "F_total", "HFE_F", "C(t)", "Action", "TD_err", "Explore?"
    );
    println!("{}", "-".repeat(65));

    let mut last_fe: Option<FreeEnergyComponents> = None;

    for cycle in 0..20 {
        // Generate observation (simulated environment signal)
        let phase = (cycle as f64) * 0.3;
        let obs_values: Vec<f64> = (0..6)
            .map(|i| {
                let base = (phase + i as f64 * 0.5).sin() * 0.3 + 0.5;
                base.clamp(0.0, 1.0)
            })
            .collect();

        let observation = Observation::new(obs_values.clone(), 1.0, "environment");

        // a) Perceive: FEP belief update
        let perception = agent.perceive(&observation);

        // b) Hierarchical belief update
        let hfe_obs: Vec<f64> = perception.updated_belief.mean.to_vec();
        hfe.update_beliefs(&hfe_obs);
        let hfe_decomp = hfe.compute_free_energy();

        // c) Compute consciousness level via MCE
        let consciousness_inputs = ConsciousnessInputs {
            phi: phi_result.phi,
            broadcast: perception.precision.min(1.0),
            working_memory: 0.7,
            attention: perception.precision.min(1.0),
            recurrence: 0.5,
            embodiment: 0.6,
            knowledge: 0.4,
            synchrony: 0.5,
        };
        let consciousness_result = mce.compute(&consciousness_inputs);

        // d) Select action
        let action_result = agent.select_action();

        // e) Execute and learn
        agent.act(action_result.action);

        // Generate next observation (shifted by action)
        let next_obs_values: Vec<f64> = (0..6)
            .map(|i| {
                let shift = action_result.action as f64 * 0.05;
                let base = (phase + shift + i as f64 * 0.5).sin() * 0.3 + 0.5;
                base.clamp(0.0, 1.0)
            })
            .collect();
        let next_obs = Observation::new(next_obs_values, 1.0, "environment");
        agent.learn_from_outcome(action_result.action, &next_obs);

        // Get TD error
        let td_err = agent.td_stats().map(|s| s.avg_td_error).unwrap_or(0.0);

        println!(
            "{:>5} {:>8.4} {:>8.4} {:>8.4} {:>6} {:>8.4} {:>10}",
            cycle,
            perception.free_energy.total,
            hfe_decomp.total,
            consciousness_result.consciousness_level,
            action_result.action,
            td_err,
            if action_result.is_exploratory {
                "yes"
            } else {
                "no"
            }
        );

        last_fe = Some(perception.free_energy);
    }

    // ─── 7. Summary ───
    println!("\n--- Summary ---\n");
    let summary = agent.summary();
    println!("Belief confidence:   {:.4}", summary.belief_confidence);
    println!("Final free energy:   {:.4}", summary.free_energy);
    println!("Precision:           {:.4}", summary.precision);
    println!(
        "Exploration rate:    {:.2}%",
        summary.exploration_rate * 100.0
    );
    println!("Total cycles:        {}", summary.total_cycles);

    if let Some(fe) = last_fe {
        println!("Final F components:");
        println!("  Accuracy:    {:.4}", fe.accuracy);
        println!("  Complexity:  {:.4}", fe.complexity);
        println!("  Surprise:    {:.4}", fe.surprise);
        println!("  Pred. Error: {:.4}", fe.prediction_error);
    }

    // HFE summary
    let hfe_f = hfe.total_free_energy();
    println!("\nHierarchical Free Energy: {:.4}", hfe_f);
    let hfe_errors = hfe.precision_weighted_errors();
    println!(
        "Per-level errors: {:?}",
        hfe_errors
            .iter()
            .map(|e| format!("{:.4}", e))
            .collect::<Vec<_>>()
    );

    // MCE summary
    println!("\nConsciousness trend: {:.4}", mce.consciousness_trend());

    // TD learning summary
    if let Some(td) = agent.td_stats() {
        println!("\nTD Learning:");
        println!("  Avg TD error:         {:.4}", td.avg_td_error);
        println!("  Prediction accuracy:  {:.4}", td.avg_prediction_accuracy);
        println!("  Total updates:        {}", td.total_updates);
    }

    println!("\n=== Done ===");
}