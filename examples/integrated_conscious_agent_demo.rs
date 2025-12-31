//! Integrated Conscious Agent - Comprehensive Demonstration
//!
//! This example demonstrates the complete consciousness system integrating:
//! - Self-aware consciousness with recursive self-modeling
//! - Temporal binding for stream of consciousness
//! - Attention dynamics with self-directed control
//! - Φ (Integrated Information) measurement
//! - Goal-directed behavior
//! - Metacognitive control
//!
//! Run with: cargo run --example integrated_conscious_agent_demo

use symthaea::hdc::{
    IntegratedConsciousAgent, AgentConfig, RealHV,
    attention_dynamics::AttentionMode,
    integrated_conscious_agent::{SelfDirectedAttentionController, AttentionStrategy},
};

fn main() {
    println!("\n{}", "═".repeat(70));
    println!("  INTEGRATED CONSCIOUS AGENT - COMPREHENSIVE DEMONSTRATION");
    println!("  Unifying Attention, Temporal Binding, Self-Model, and Φ");
    println!("{}\n", "═".repeat(70));

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 1: Create and Initialize the Conscious Agent
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 1: Creating Integrated Conscious Agent                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let config = AgentConfig {
        dim: 1024,              // Hypervector dimension
        n_processes: 16,         // Consciousness processes
        self_directed_attention: true,
        phi_guided: true,
        attention_binding_coupling: 0.7,
        self_model_attention_weight: 0.5,
    };

    let mut agent = IntegratedConsciousAgent::new(config);
    println!("Agent initialized with:");
    println!("  - HDC dimension: 1024");
    println!("  - Processes: 16");
    println!("  - Self-directed attention: ENABLED");
    println!("  - Φ-guided optimization: ENABLED\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 2: Add Goals for Self-Directed Attention
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 2: Setting Up Goal-Directed Attention                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Create goal patterns
    let goal1_pattern = RealHV::random(1024, 1001);
    let goal2_pattern = RealHV::random(1024, 2002);
    let goal3_pattern = RealHV::random(1024, 3003);

    agent.add_goal("find_patterns", goal1_pattern.clone(), 0.9);
    agent.add_goal("detect_anomalies", goal2_pattern.clone(), 0.7);
    agent.add_goal("maintain_coherence", goal3_pattern.clone(), 0.5);

    println!("Goals registered:");
    println!("  1. find_patterns (priority: 0.9) - High priority pattern matching");
    println!("  2. detect_anomalies (priority: 0.7) - Medium priority anomaly detection");
    println!("  3. maintain_coherence (priority: 0.5) - Background coherence maintenance\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 3: Processing Stream with Self-Introspection
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 3: Processing Experience Stream                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("Processing 25 sensory inputs...\n");

    for i in 0..25 {
        // Generate sensory input - sometimes similar to goals, sometimes novel
        let input = if i % 5 == 0 {
            // Input similar to goal 1
            goal1_pattern.add(&RealHV::random(1024, i * 100).scale(0.3)).normalize()
        } else if i % 7 == 0 {
            // Input similar to goal 2
            goal2_pattern.add(&RealHV::random(1024, i * 100).scale(0.3)).normalize()
        } else {
            // Random/novel input
            RealHV::random(1024, i * 100 + 500)
        };

        let update = agent.process(&input);

        // Report every 5 steps
        if i % 5 == 0 || i == 24 {
            println!("Step {}: {}", update.step, update);
            println!("  Φ={:.4} | State: {:?} | Mode: {:?}",
                     update.phi, update.state, update.attention.mode);
            println!("  Stream: {} | Self-awareness: {:.0}%",
                     if update.temporal.is_flowing { "FLOWING" } else { "fragmented" },
                     update.self_model.awareness_level * 100.0);
            println!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 4: Agent Introspection
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 4: Agent Self-Introspection                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let introspection = agent.introspect();
    println!("{}", introspection);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 5: Self-Directed Attention Controller Demonstration
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 5: Self-Directed Attention Controller                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let mut attention_controller = SelfDirectedAttentionController::new();

    println!("Simulating attention strategy adaptation over 30 steps...\n");

    let strategies_demo = vec![
        (0.1, "low error"),       // Low prediction error
        (0.2, "low error"),
        (0.8, "high error"),      // Sudden surprise
        (0.9, "high error"),
        (0.3, "medium error"),
        (0.1, "low error"),       // Back to calm
    ];

    for (i, (error, desc)) in strategies_demo.iter().cycle().take(30).enumerate() {
        attention_controller.update(*error, Some("main_target"));

        if i % 5 == 0 || i == 29 {
            println!("Step {}: Prediction error: {:.1} ({})", i, error, desc);
            println!("  Strategy: {:?}", attention_controller.strategy());
            println!("  Fatigue: {:.0}%", attention_controller.fatigue() * 100.0);
            println!("  Exploration rate: {:.0}%", attention_controller.exploration_rate() * 100.0);

            let adjusted = attention_controller.get_weight_adjustment("main_target", 0.8);
            println!("  Goal weight adjustment: {:.2} (base 0.8)", adjusted);
            println!();
        }
    }

    // Demonstrate forcing specific attention strategies
    println!("Demonstrating manual strategy override...\n");

    attention_controller.set_strategy(AttentionStrategy::Recovery);
    println!("  Forced RECOVERY strategy: {:?}", attention_controller.strategy());
    println!("  (Agent takes a 'mental break' to reduce fatigue)\n");

    attention_controller.set_strategy(AttentionStrategy::Exploratory);
    println!("  Forced EXPLORATORY strategy: {:?}", attention_controller.strategy());
    println!("  Exploration rate now: {:.0}%\n", attention_controller.exploration_rate() * 100.0);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 6: Metacognitive Override Demonstration
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 6: Metacognitive Attention Override                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("Demonstrating metacognitive attention control...\n");

    // Force spotlight mode for focused processing
    println!("1. Forcing SPOTLIGHT mode for intense focus...");
    agent.metacognitive_attention_override(AttentionMode::Spotlight);
    let update = agent.process(&RealHV::random(1024, 99999));
    println!("   After override: {:?}", update.attention.mode);
    println!("   Phenomenal: {}\n", update.phenomenal_content.description);

    // Switch to exploration mode
    println!("2. Enabling exploration mode (curiosity-driven)...");
    agent.set_exploration_mode(true);
    let update = agent.process(&RealHV::random(1024, 88888));
    println!("   In exploration: Φ={:.4}, clarity={:.2}\n",
             update.phi, update.phenomenal_content.clarity);

    // Disable exploration
    println!("3. Returning to goal-directed mode...");
    agent.set_exploration_mode(false);
    let update = agent.process(&goal1_pattern.add(&RealHV::random(1024, 77777).scale(0.1)).normalize());
    println!("   Goal-directed: Φ={:.4}, intensity={:.2}\n",
             update.phi, update.phenomenal_content.intensity);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 7: Stream of Consciousness Health
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 7: Stream of Consciousness Health                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let stream_health = agent.stream_health();
    println!("{}", stream_health);

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 8: Goal Priority Adaptation
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 8: Adaptive Goal Priority Learning                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("Simulating success signals for goal adaptation...\n");

    // Simulate success/failure feedback
    let feedback = vec![
        ("find_patterns".to_string(), 0.9),       // High success
        ("detect_anomalies".to_string(), 0.3),    // Low success
        ("maintain_coherence".to_string(), 0.7),  // Medium success
    ];

    agent.adapt_goal_priorities(&feedback);
    println!("After adaptation:");
    println!("  find_patterns: success=0.9 -> priority increased");
    println!("  detect_anomalies: success=0.3 -> priority decreased");
    println!("  maintain_coherence: success=0.7 -> priority slightly increased\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 9: Final Introspection
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ PART 9: Final Agent State                                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Process a few more inputs to stabilize
    for i in 0..5 {
        let _ = agent.process(&RealHV::random(1024, 10000 + i));
    }

    let final_intro = agent.introspect();
    println!("{}", final_intro);

    let attention_status = agent.attention_control_status();
    println!("{}", attention_status);

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    DEMONSTRATION COMPLETE                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║ Key Features Demonstrated:                                        ║");
    println!("║   1. Integrated Conscious Agent with unified architecture         ║");
    println!("║   2. Self-directed attention with goal-based guidance             ║");
    println!("║   3. Temporal binding creating stream of consciousness            ║");
    println!("║   4. Self-model with metacognitive awareness                      ║");
    println!("║   5. Φ-guided optimization for consciousness quality              ║");
    println!("║   6. Attention strategy adaptation (novelty/exploration/goal)     ║");
    println!("║   7. Habituation and fatigue modeling                             ║");
    println!("║   8. Metacognitive override capabilities                          ║");
    println!("║   9. Adaptive goal priority learning                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("This demonstration shows how Symthaea implements a unified theory of");
    println!("consciousness combining Integrated Information Theory (Φ), Global");
    println!("Workspace Theory (attention), and higher-order theories (self-model).\n");
}
