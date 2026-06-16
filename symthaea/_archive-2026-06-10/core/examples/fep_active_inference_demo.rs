// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Full Active Inference Demo
//!
//! Demonstrates the complete Free Energy Principle (FEP) integration with
//! active inference for consciousness-aware processing.
//!
//! ## What this demonstrates:
//!
//! 1. **Generative Model**: How the system predicts observations from beliefs
//! 2. **Free Energy Computation**: Variational free energy calculation
//! 3. **Perception**: Belief updating via free energy minimization
//! 4. **Action Selection**: Choosing actions via expected free energy
//! 5. **Precision Weighting**: Confidence-modulated prediction errors
//! 6. **Learning**: Model adaptation based on prediction errors
//!
//! ## Run:
//! ```bash
//! cargo run --example fep_active_inference_demo
//! ```

use symthaea::consciousness::fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, CognitiveLoopFEPBridge, FreeEnergyCalculator,
    GenerativeModel, HiddenState, Observation, PrecisionEstimator,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     FREE ENERGY PRINCIPLE (FEP) DEMO                         ║");
    println!("║                     Active Inference Implementation                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // -------------------------------------------------------------------------
    // Part 1: Basic Components Demo
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("PART 1: Basic FEP Components");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    demo_generative_model();
    demo_free_energy_computation();
    demo_precision_weighting();

    // -------------------------------------------------------------------------
    // Part 2: Full Active Inference Loop
    // -------------------------------------------------------------------------
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("PART 2: Full Active Inference Loop");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    demo_perception_action_loop();

    // -------------------------------------------------------------------------
    // Part 3: Cognitive Loop Integration
    // -------------------------------------------------------------------------
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("PART 3: Cognitive Loop Integration");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    demo_cognitive_loop_integration();

    // -------------------------------------------------------------------------
    // Part 4: Goal-Directed Behavior
    // -------------------------------------------------------------------------
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("PART 4: Goal-Directed Behavior (Pragmatic Value)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    demo_goal_directed_behavior();

    // -------------------------------------------------------------------------
    // Part 5: Exploration vs Exploitation
    // -------------------------------------------------------------------------
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("PART 5: Exploration vs Exploitation (Epistemic Value)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    demo_exploration_exploitation();

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         DEMO COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

/// Demonstrate the generative model (state -> observation prediction)
fn demo_generative_model() {
    println!("--- Generative Model Demo ---");
    println!();

    let model = GenerativeModel::new(4, 4, 4);
    println!("Created generative model:");
    println!("  - State dimension: {}", model.state_dim);
    println!("  - Observation dimension: {}", model.obs_dim);
    println!("  - Number of actions: {}", model.num_actions);
    println!();

    // Create a belief state
    let mut state = HiddenState::new(4);
    state.mean = vec![0.7, 0.6, 0.8, 0.5];
    println!("Current belief state: {:?}", state.mean);

    // Predict observation
    let predicted_obs = model.predict_observation(&state);
    println!(
        "Predicted observation: {:?}",
        predicted_obs
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // Predict next state under different actions
    println!();
    println!("State transitions under different actions:");
    for action in 0..model.num_actions {
        let next_state = model.predict_next_state(&state, action);
        println!(
            "  Action {}: state -> {:?}",
            action,
            next_state
                .mean
                .iter()
                .map(|x| format!("{:.3}", x))
                .collect::<Vec<_>>()
        );
    }
    println!();
}

/// Demonstrate free energy computation
fn demo_free_energy_computation() {
    println!("--- Free Energy Computation Demo ---");
    println!();
    println!("Free Energy F = Complexity - Accuracy");
    println!("  Accuracy  = E_q[log p(o|s)]  (how well predictions match observations)");
    println!("  Complexity = D_KL[q(s)||p(s)] (divergence from prior beliefs)");
    println!();

    let model = GenerativeModel::new(4, 4, 4);
    let mut calc = FreeEnergyCalculator::new(100);

    let state = HiddenState::new(4);

    // Test with matching observation (low surprise)
    let good_obs = Observation::from_consciousness_state(0.5, 0.5, 0.5, 0.5);
    let fe_good = calc.compute(&state, &good_obs, &model);
    println!("Observation matching prior (phi=0.5, int=0.5, coh=0.5, att=0.5):");
    println!("  Free Energy:     {:.4}", fe_good.total);
    println!("  Accuracy:        {:.4}", fe_good.accuracy);
    println!("  Complexity:      {:.4}", fe_good.complexity);
    println!("  Prediction Error: {:.4}", fe_good.prediction_error);
    println!();

    // Test with surprising observation
    let surprising_obs = Observation::from_consciousness_state(0.9, 0.1, 0.9, 0.1);
    let fe_surprising = calc.compute(&state, &surprising_obs, &model);
    println!("Surprising observation (phi=0.9, int=0.1, coh=0.9, att=0.1):");
    println!("  Free Energy:     {:.4}", fe_surprising.total);
    println!("  Accuracy:        {:.4}", fe_surprising.accuracy);
    println!("  Complexity:      {:.4}", fe_surprising.complexity);
    println!("  Prediction Error: {:.4}", fe_surprising.prediction_error);
    println!();

    println!("Higher free energy = more surprise = worse model fit");
    println!();
}

/// Demonstrate precision weighting
fn demo_precision_weighting() {
    println!("--- Precision Weighting Demo ---");
    println!();
    println!("Precision modulates the influence of prediction errors.");
    println!("High precision = trust this signal, Low precision = discount it.");
    println!();

    let mut precision = PrecisionEstimator::new();
    println!("Initial precisions:");
    println!("  Sensory:  {:.3}", precision.sensory_precision);
    println!("  Prior:    {:.3}", precision.prior_precision);
    println!("  Action:   {:.3}", precision.action_precision);
    println!();

    // Simulate high prediction errors (should trust observations more)
    println!("After high prediction errors (0.8):");
    for i in 0..5 {
        precision.update_from_error(0.8, i);
    }
    println!(
        "  Sensory:  {:.3} (increased - trust observations)",
        precision.sensory_precision
    );
    println!(
        "  Prior:    {:.3} (decreased - distrust predictions)",
        precision.prior_precision
    );
    println!();

    // Simulate low prediction errors (should trust predictions more)
    println!("After low prediction errors (0.1):");
    for i in 5..15 {
        precision.update_from_error(0.1, i);
    }
    println!("  Sensory:  {:.3}", precision.sensory_precision);
    println!(
        "  Prior:    {:.3} (increased - trust predictions)",
        precision.prior_precision
    );
    println!("  Stability: {:.3}", precision.stability());
    println!();
}

/// Demonstrate the full perception-action loop
fn demo_perception_action_loop() {
    println!("--- Perception-Action Loop Demo ---");
    println!();
    println!("Active Inference Loop:");
    println!("  1. Observe -> 2. Perceive (update beliefs) -> 3. Select action -> 4. Act");
    println!();

    let config = ActiveInferenceAgentConfig {
        state_dim: 4,
        obs_dim: 4,
        num_actions: 4,
        inference_iterations: 5,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(config);

    // Simulate 10 cycles of perception and action
    println!("Running 10 perception-action cycles:");
    println!();
    println!(
        "{:^6} {:^10} {:^10} {:^10} {:^8} {:^12}",
        "Cycle", "Phi", "FreeEnergy", "PredError", "Action", "Exploratory"
    );
    println!(
        "{:-<6} {:-<10} {:-<10} {:-<10} {:-<8} {:-<12}",
        "", "", "", "", "", ""
    );

    for cycle in 0..10 {
        // Generate observation (simulating consciousness state)
        let phi = 0.3 + 0.4 * (cycle as f64 * 0.5).sin().abs();
        let integration = 0.4 + 0.3 * (cycle as f64 * 0.3).cos().abs();
        let coherence = 0.5 + 0.3 * (cycle as f64 * 0.4).sin().abs();
        let attention = 0.6 + 0.2 * (cycle as f64 * 0.2).cos().abs();

        let observation =
            Observation::from_consciousness_state(phi, integration, coherence, attention);

        // Perception step
        let perception = agent.perceive(&observation);

        // Action selection step
        let action = agent.select_action();

        println!(
            "{:^6} {:^10.3} {:^10.4} {:^10.4} {:^8} {:^12}",
            cycle,
            phi,
            perception.free_energy.total,
            perception.free_energy.prediction_error,
            action.action,
            if action.is_exploratory { "yes" } else { "no" }
        );
    }

    let summary = agent.summary();
    println!();
    println!("Agent Summary:");
    println!("  Total cycles:     {}", summary.total_cycles);
    println!("  Belief confidence: {:.3}", summary.belief_confidence);
    println!("  Exploration rate:  {:.3}", summary.exploration_rate);
    println!("  Average precision: {:.3}", summary.precision);
    println!();
}

/// Demonstrate cognitive loop integration
fn demo_cognitive_loop_integration() {
    println!("--- Cognitive Loop Integration Demo ---");
    println!();
    println!("The CognitiveLoopFEPBridge provides:");
    println!("  - Precision-weighted prediction errors for the cognitive loop");
    println!("  - Learning rate modulation based on free energy");
    println!("  - Recommended actions for consciousness optimization");
    println!();

    let config = ActiveInferenceAgentConfig::default();
    let mut bridge = CognitiveLoopFEPBridge::new(config);

    // Set goals for high consciousness state
    bridge.set_goals(0.8, 0.7, 0.8, 0.7);
    println!("Goals set: phi=0.8, integration=0.7, coherence=0.8, attention=0.7");
    println!();

    println!("Processing different consciousness states:");
    println!();
    println!(
        "{:^12} {:^12} {:^12} {:^14} {:^10} {:^10}",
        "State", "FreeEnergy", "PrecWtdErr", "LearningMod", "ShouldLrn", "Surprised"
    );
    println!(
        "{:-<12} {:-<12} {:-<12} {:-<14} {:-<10} {:-<10}",
        "", "", "", "", "", ""
    );

    let states = [
        (0.3, 0.3, 0.3, 0.3, "Low"),
        (0.5, 0.5, 0.5, 0.5, "Medium"),
        (0.7, 0.7, 0.7, 0.7, "High"),
        (0.9, 0.9, 0.9, 0.9, "Very High"),
    ];

    for (phi, int, coh, att, label) in states {
        let result = bridge.process(phi, int, coh, att);
        println!(
            "{:^12} {:^12.4} {:^12.4} {:^14.3} {:^10} {:^10}",
            label,
            result.free_energy,
            result.precision_weighted_error,
            result.learning_rate_modulation,
            if result.should_learn { "yes" } else { "no" },
            if result.is_surprised { "yes" } else { "no" }
        );
    }
    println!();
}

/// Demonstrate goal-directed behavior
fn demo_goal_directed_behavior() {
    println!("--- Goal-Directed Behavior Demo ---");
    println!();
    println!("The agent selects actions to minimize expected free energy,");
    println!("which includes pragmatic value (reaching goals) and epistemic value.");
    println!();

    let config = ActiveInferenceAgentConfig::default();
    let mut bridge = CognitiveLoopFEPBridge::new(config);

    // Test different goal states
    println!("Scenario 1: Goal = High Phi (0.9)");
    bridge.set_goals(0.9, 0.5, 0.5, 0.5);

    // Start from low phi
    let result1 = bridge.process(0.3, 0.5, 0.5, 0.5);
    println!("  Current state: phi=0.3 (far from goal)");
    println!(
        "  Pragmatic value: {:.4} (high motivation to act)",
        result1.pragmatic_value
    );
    println!("  Recommended action: {}", result1.recommended_action);
    println!();

    // Start from high phi
    let result2 = bridge.process(0.85, 0.5, 0.5, 0.5);
    println!("  Current state: phi=0.85 (close to goal)");
    println!(
        "  Pragmatic value: {:.4} (low motivation - near goal)",
        result2.pragmatic_value
    );
    println!("  Recommended action: {}", result2.recommended_action);
    println!();

    println!("Scenario 2: Goal = High Coherence (0.9)");
    bridge.set_goals(0.5, 0.5, 0.9, 0.5);
    bridge.agent.reset();

    let result3 = bridge.process(0.5, 0.5, 0.3, 0.5);
    println!("  Current state: coherence=0.3 (far from goal)");
    println!("  Pragmatic value: {:.4}", result3.pragmatic_value);
    println!("  Recommended action: {}", result3.recommended_action);
    println!();
}

/// Demonstrate exploration vs exploitation
fn demo_exploration_exploitation() {
    println!("--- Exploration vs Exploitation Demo ---");
    println!();
    println!("Active inference balances:");
    println!("  - Exploitation: Actions that achieve goals (pragmatic)");
    println!("  - Exploration: Actions that reduce uncertainty (epistemic)");
    println!();

    let config = ActiveInferenceAgentConfig {
        state_dim: 4,
        obs_dim: 4,
        num_actions: 6,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(config);

    // High epistemic weight encourages exploration
    agent.efe_computer.epistemic_weight = 2.0;
    agent.efe_computer.pragmatic_weight = 0.5;
    println!("Configuration: High epistemic weight (exploration encouraged)");

    // Run multiple cycles
    let mut exploration_count = 0;
    let total_cycles = 20;

    for cycle in 0..total_cycles {
        let phi = 0.5 + 0.3 * (cycle as f64 * 0.3).sin();
        let obs = Observation::from_consciousness_state(phi, 0.5, 0.5, 0.5);
        agent.perceive(&obs);
        let action = agent.select_action();
        if action.is_exploratory {
            exploration_count += 1;
        }
    }

    println!(
        "  Exploration rate: {}/{} = {:.1}%",
        exploration_count,
        total_cycles,
        100.0 * exploration_count as f64 / total_cycles as f64
    );
    println!();

    // Reset and use low epistemic weight
    agent.reset();
    agent.efe_computer.epistemic_weight = 0.1;
    agent.efe_computer.pragmatic_weight = 2.0;
    println!("Configuration: High pragmatic weight (exploitation encouraged)");

    exploration_count = 0;
    for cycle in 0..total_cycles {
        let phi = 0.5 + 0.3 * (cycle as f64 * 0.3).sin();
        let obs = Observation::from_consciousness_state(phi, 0.5, 0.5, 0.5);
        agent.perceive(&obs);
        let action = agent.select_action();
        if action.is_exploratory {
            exploration_count += 1;
        }
    }

    println!(
        "  Exploration rate: {}/{} = {:.1}%",
        exploration_count,
        total_cycles,
        100.0 * exploration_count as f64 / total_cycles as f64
    );
    println!();

    println!("The epistemic/pragmatic balance enables adaptive behavior:");
    println!("  - Novel situations -> more exploration");
    println!("  - Familiar situations -> more exploitation");
    println!();
}