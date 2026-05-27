// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sensorimotor Contingency Demo
//!
//! Demonstrates the enactivist sensorimotor contingency system:
//! - Learning contingencies from action-outcome pairs
//! - Predicting sensory changes from actions
//! - Detecting surprising outcomes (contingency violations)
//! - Gibson-style affordance detection
//!
//! ## Running
//!
//! ```bash
//! cargo run --example sensorimotor_contingency_demo
//! ```

use symthaea::hdc::sensorimotor_contingencies::{
    ActionDescriptor, ActionType, AffordanceDetector, ContextDescriptor,
    ContingencyConsciousnessContribution, ContingencyLearner, EnactivistPerception, SensoryChange,
    SensoryModality,
};

fn main() {
    println!("============================================================");
    println!("     ENACTIVIST SENSORIMOTOR CONTINGENCY DEMONSTRATION      ");
    println!("============================================================\n");

    println!("This demo illustrates O'Regan & Noe's Sensorimotor Contingency");
    println!("Theory: perception is NOT passive reception of information.");
    println!("Perception IS implicit knowledge of how actions affect sensation.\n");

    // Run all demonstrations
    demo_basic_learning();
    demo_prediction_and_surprise();
    demo_enactivist_perception();
    demo_affordance_detection();
    demo_consciousness_contribution();

    println!("\n============================================================");
    println!("                    DEMO COMPLETE                            ");
    println!("============================================================");
}

/// Demonstrate basic contingency learning
fn demo_basic_learning() {
    println!("\n--- DEMO 1: Basic Contingency Learning ---\n");

    let mut learner = ContingencyLearner::new();

    // Scenario: Learning that reaching toward an object produces tactile feedback
    println!("Scenario: Learning reach-to-touch contingencies\n");

    // Define the action: reaching
    let reach_action = ActionDescriptor::new(ActionType::Reach)
        .with_parameter("distance", 0.5)
        .with_parameter("direction", 0.0)
        .with_effector("right_arm");

    // Define the context: object present in front
    let context = ContextDescriptor::new("object_in_reach", SensoryModality::Visual)
        .with_feature("object_present", 1.0)
        .with_feature("distance", 0.5)
        .with_object("cup");

    // Define the typical outcome: tactile contact
    let contact_outcome = SensoryChange::new(SensoryModality::Tactile)
        .with_vector(vec![0.8, 0.5, 0.2, 0.0]) // pressure, temperature, texture, pain
        .with_description("contact with smooth surface");

    // Train the system with repeated experiences
    println!("Training: Performing reach action 10 times...\n");

    for i in 1..=10 {
        let result = learner.learn(
            reach_action.clone(),
            context.clone(),
            contact_outcome.clone(),
        );

        if i == 1 || i == 5 || i == 10 {
            println!(
                "  Experience {}: surprise = {:.3}, contingency: {}",
                i,
                result.surprise,
                result.contingency_id.as_deref().unwrap_or("none")
            );
        }
    }

    // Check what was learned
    println!("\nLearning Results:");
    println!("  Total experiences: {}", learner.stats().total_experiences);
    println!("  Contingencies learned: {}", learner.contingency_count());
    println!(
        "  Prediction accuracy: {:.1}%",
        learner.prediction_accuracy() * 100.0
    );

    // Make a prediction
    if let Some(prediction) = learner.predict(&reach_action, &context) {
        println!("\nPrediction for reach in this context:");
        println!("  Modality: {:?}", prediction.expected.modality);
        println!("  Confidence: {:.3}", prediction.confidence);
        println!("  Description: {}", prediction.expected.description);
    }
}

/// Demonstrate prediction and surprise detection
fn demo_prediction_and_surprise() {
    println!("\n\n--- DEMO 2: Prediction and Surprise Detection ---\n");

    let mut learner = ContingencyLearner::new();

    // Learn a reliable contingency
    let push_action = ActionDescriptor::new(ActionType::Push).with_parameter("force", 0.5);

    let context = ContextDescriptor::new("ball_on_surface", SensoryModality::Visual)
        .with_feature("object_mass", 0.3)
        .with_object("ball");

    // Normal outcome: ball moves forward
    let normal_outcome = SensoryChange::new(SensoryModality::Visual)
        .with_vector(vec![0.5, 0.0, 0.0, 0.0]) // motion in x direction
        .with_description("ball rolls forward");

    println!("Phase 1: Learning normal contingency (push -> ball rolls)\n");

    for _ in 0..10 {
        learner.learn(push_action.clone(), context.clone(), normal_outcome.clone());
    }

    println!(
        "  After 10 experiences, prediction accuracy: {:.1}%\n",
        learner.prediction_accuracy() * 100.0
    );

    // Now introduce a surprising outcome!
    println!("Phase 2: Encountering surprising outcome\n");

    let surprising_outcome = SensoryChange::new(SensoryModality::Visual)
        .with_vector(vec![-0.3, 0.8, 0.0, 0.0]) // ball moves backwards and up!
        .with_description("ball bounces back unexpectedly");

    let result = learner.learn(push_action.clone(), context.clone(), surprising_outcome);

    println!("  Surprising event occurred!");
    println!("  Surprise level: {:.3}", result.surprise);
    println!("  Is violation: {}", result.is_violation);
    println!(
        "  Prediction error: {:.3}",
        result.prediction_error.unwrap_or(0.0)
    );

    println!("\n  This surprise signal would feed into:");
    println!("    - Free energy minimization");
    println!("    - Attention allocation");
    println!("    - Learning rate modulation");
    println!("    - Memory encoding prioritization");
}

/// Demonstrate enactivist perception
fn demo_enactivist_perception() {
    println!("\n\n--- DEMO 3: Enactivist Perception ---\n");

    println!("In enactivism, 'seeing red' is not having a red quale.");
    println!("It is KNOWING how red things behave under action:\n");
    println!("  - How they darken in shadow");
    println!("  - How they shift toward orange in warm light");
    println!("  - How they reflect certain wavelengths");
    println!("  - What actions they afford\n");

    let mut perception = EnactivistPerception::new();

    // Simulate visual exploration of a scene
    let saccade_left =
        ActionDescriptor::new(ActionType::SaccadeLeft).with_parameter("magnitude", 10.0);
    let saccade_right =
        ActionDescriptor::new(ActionType::SaccadeRight).with_parameter("magnitude", 10.0);

    let visual_scene = ContextDescriptor::new("office_scene", SensoryModality::Visual)
        .with_feature("complexity", 0.7)
        .with_object("desk")
        .with_object("monitor")
        .with_object("chair");

    // Visual outcome: scene shifts as expected
    let scene_shift_right = SensoryChange::new(SensoryModality::Visual)
        .with_vector(vec![0.3, 0.0, 0.0, 0.0]) // retinal image shifts right
        .with_description("scene shifts rightward");

    let scene_shift_left = SensoryChange::new(SensoryModality::Visual)
        .with_vector(vec![-0.3, 0.0, 0.0, 0.0])
        .with_description("scene shifts leftward");

    println!("Simulating visual exploration (saccadic eye movements)...\n");

    // Explore the scene
    for i in 0..20 {
        let (action, outcome) = if i % 2 == 0 {
            (saccade_left.clone(), scene_shift_right.clone())
        } else {
            (saccade_right.clone(), scene_shift_left.clone())
        };

        let result = perception.perceive_through_action(action, visual_scene.clone(), outcome);

        if i == 0 || i == 9 || i == 19 {
            println!(
                "  Cycle {}: clarity = {:.3}, surprise = {:.3}",
                i + 1,
                result.clarity,
                result.surprise
            );
        }
    }

    println!("\nPerception Results:");
    println!("  Perceptual mastery: {:.1}%", perception.mastery() * 100.0);
    println!(
        "  Active contingencies: {}",
        perception.stats().avg_active_contingencies as usize
    );

    let ready_actions = perception.get_ready_actions();
    println!("\n  Motor readiness (actions primed for execution):");
    for (action, readiness) in ready_actions.iter().take(3) {
        println!("    {:?}: {:.2}", action, readiness);
    }

    // Imagination: predict what would happen
    println!("\n  Imagination (predicting unexecuted action):");
    let imagined = ActionDescriptor::new(ActionType::HeadTurnLeft).with_parameter("angle", 30.0);

    if let Some(prediction) = perception.imagine_action(&imagined, &visual_scene) {
        println!("    If I turn my head left, I predict:");
        println!("    Confidence: {:.3}", prediction.confidence);
    } else {
        println!("    No prediction available (need more experience with head turns)");
    }
}

/// Demonstrate Gibson-style affordance detection
fn demo_affordance_detection() {
    println!("\n\n--- DEMO 4: Gibson-Style Affordance Detection ---\n");

    println!("Affordances are action possibilities the environment offers");
    println!("to an embodied agent. They are NOT object properties alone,");
    println!("but RELATIONS between object + agent + context.\n");

    let mut detector = AffordanceDetector::new();

    // Train on grasping actions
    let grasp_action = ActionDescriptor::new(ActionType::Grasp)
        .with_parameter("aperture", 0.1)
        .with_effector("hand");

    let graspable_context = ContextDescriptor::new("small_object", SensoryModality::Visual)
        .with_feature("size", 0.1) // small
        .with_feature("graspable", 1.0)
        .with_object("apple");

    let grasp_outcome = SensoryChange::new(SensoryModality::Tactile)
        .with_vector(vec![0.9, 0.3, 0.5, 0.0]) // firm grip
        .with_description("successful grasp");

    println!("Phase 1: Learning grasp contingency...\n");

    for _ in 0..15 {
        detector.process_action(
            grasp_action.clone(),
            graspable_context.clone(),
            grasp_outcome.clone(),
        );
    }

    // Train on pushing actions
    let push_action = ActionDescriptor::new(ActionType::Push).with_parameter("force", 0.3);

    let push_outcome = SensoryChange::new(SensoryModality::Visual)
        .with_vector(vec![0.4, 0.0, 0.0, 0.0])
        .with_description("object moves away");

    for _ in 0..10 {
        detector.process_action(
            push_action.clone(),
            graspable_context.clone(),
            push_outcome.clone(),
        );
    }

    // Detect affordances
    println!("Phase 2: Detecting affordances in current context...\n");

    let affordances = detector.detect(&graspable_context);

    println!("  Detected {} affordances:\n", affordances.len());

    for (i, affordance) in affordances.iter().enumerate() {
        println!("  {}. {:?}", i + 1, affordance.action);
        println!("     Source: {}", affordance.source);
        println!("     Salience: {:.3}", affordance.salience);
        println!("     Confidence: {:.3}", affordance.confidence);
        println!("     Attractiveness: {:.3}", affordance.attractiveness());
        // Note: In symthaea-core implementation, affordances don't store predicted_outcome directly
        // The prediction can be obtained via detector.perception().imagine_action()
        println!();
    }

    if let Some(best) = detector.most_attractive() {
        println!(
            "  Most attractive action: {:?} (attractiveness: {:.3})",
            best.action,
            best.attractiveness()
        );
    }
}

/// Demonstrate consciousness contribution calculation
fn demo_consciousness_contribution() {
    println!("\n\n--- DEMO 5: Consciousness Contribution ---\n");

    println!("Enactivist insight: Consciousness requires MASTERY of contingencies.");
    println!("A being with perfect sensorimotor predictions has higher awareness.\n");

    let mut detector = AffordanceDetector::new();

    // Train extensively
    let actions = vec![
        (ActionType::Touch, "tactile exploration"),
        (ActionType::Reach, "reaching movement"),
        (ActionType::SaccadeLeft, "visual scanning"),
        (ActionType::Grasp, "grasping objects"),
    ];

    let context = ContextDescriptor::new("rich_environment", SensoryModality::MultiModal)
        .with_feature("complexity", 0.8)
        .with_object("various_objects");

    println!("Training on multiple action types...\n");

    for (action_type, description) in &actions {
        let action = ActionDescriptor::new(*action_type);
        let outcome = SensoryChange::new(SensoryModality::Tactile)
            .with_vector(vec![0.5, 0.3, 0.2, 0.0])
            .with_description(description);

        for _ in 0..20 {
            detector.process_action(action.clone(), context.clone(), outcome.clone());
        }
    }

    // Calculate consciousness contribution
    let contribution = ContingencyConsciousnessContribution::from_detector(&detector, &context);

    println!("Consciousness Contribution Analysis:");
    println!("  ----------------------------------------");
    println!(
        "  Sensorimotor Mastery: {:.1}%",
        contribution.mastery * 100.0
    );
    println!(
        "  Prediction Accuracy:  {:.1}%",
        contribution.prediction_accuracy * 100.0
    );
    println!("  Current Surprise:     {:.3}", contribution.surprise);
    println!(
        "  Contingencies Learned: {}",
        contribution.contingency_count
    );
    println!(
        "  Motor Readiness:      {:.3}",
        contribution.motor_readiness
    );
    println!(
        "  Perceptual Clarity:   {:.3}",
        contribution.perceptual_clarity
    );
    println!("  ----------------------------------------");
    println!(
        "  CONSCIOUSNESS CONTRIBUTION (M_smc): {:.3}",
        contribution.consciousness_contribution()
    );

    println!("\n  This value feeds into the Master Consciousness Equation:");
    println!("  C(t) = sigma(softmin(...)) x [weighted_sum] x S x rho(t)");
    println!("\n  Where M_smc contributes to the embodiment factor.");

    println!(
        "\n  Prediction Error ({:.3}) feeds into:",
        contribution.prediction_error()
    );
    println!("    - Free Energy Principle minimization");
    println!("    - Active Inference action selection");
    println!("    - Attention reallocation");
}