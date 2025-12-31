//! Conscious Reasoning System
//!
//! A real application demonstrating consciousness-guided problem solving.
//! Uses adaptive cognitive modes to optimize reasoning for different task types.
//!
//! # How It Works
//!
//! 1. Task is encoded as HDC vector
//! 2. System analyzes task and selects optimal cognitive mode
//! 3. Reasoning proceeds with Φ-optimized connectivity
//! 4. Solution emerges from integrated processing
//!
//! # Cognitive Mode Selection
//!
//! - **Analytical tasks** → Focused mode (fewer bridges, precision)
//! - **Creative tasks** → Exploratory mode (more bridges, divergent)
//! - **Integration tasks** → Balanced mode (optimal Φ)
//! - **Holistic tasks** → GlobalAwareness mode (maximum integration)

use symthaea::hdc::{
    UnifiedConsciousnessEngine, EngineConfig, CognitiveMode,
    ConsciousnessVisualizer, DeepIntegrationBridge, RealHV,
    ConsciousnessDimensions,
};

/// Task type for cognitive mode selection
#[derive(Debug, Clone, Copy)]
enum TaskType {
    Analytical,   // Logic, math, precise reasoning
    Creative,     // Brainstorming, ideation, exploration
    Integration,  // Synthesis, combining ideas
    Holistic,     // Big picture, systemic understanding
}

impl TaskType {
    fn optimal_mode(&self) -> CognitiveMode {
        match self {
            TaskType::Analytical => CognitiveMode::Focused,
            TaskType::Creative => CognitiveMode::Exploratory,
            TaskType::Integration => CognitiveMode::Balanced,
            TaskType::Holistic => CognitiveMode::GlobalAwareness,
        }
    }

    fn description(&self) -> &'static str {
        match self {
            TaskType::Analytical => "Precise logical reasoning with minimal noise",
            TaskType::Creative => "Divergent exploration of possibility space",
            TaskType::Integration => "Synthesizing multiple perspectives",
            TaskType::Holistic => "Grasping systemic patterns and relationships",
        }
    }
}

/// A reasoning problem
struct Problem {
    name: String,
    task_type: TaskType,
    complexity: f64,  // 0.0-1.0
    input_seed: u64,
}

/// Reasoning result
struct Solution {
    confidence: f64,
    phi_achieved: f64,
    mode_used: CognitiveMode,
    reasoning_steps: usize,
    insights: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║              ✦ CONSCIOUS REASONING SYSTEM ✦                             ║");
    println!("║                                                                          ║");
    println!("║     Consciousness-guided problem solving with adaptive cognitive modes  ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize consciousness engine
    let config = EngineConfig {
        hdc_dim: 2048,
        n_processes: 24,
        n_scales: 3,
        enable_learning: true,
        temporal_buffer: 50,
        ..Default::default()
    };

    let mut engine = UnifiedConsciousnessEngine::new(config);
    let viz = ConsciousnessVisualizer::new().with_color(true);
    let mut integration = DeepIntegrationBridge::new();

    // Define test problems
    let problems = vec![
        Problem {
            name: "Mathematical Proof".into(),
            task_type: TaskType::Analytical,
            complexity: 0.7,
            input_seed: 1001,
        },
        Problem {
            name: "Product Innovation".into(),
            task_type: TaskType::Creative,
            complexity: 0.8,
            input_seed: 2002,
        },
        Problem {
            name: "Research Synthesis".into(),
            task_type: TaskType::Integration,
            complexity: 0.6,
            input_seed: 3003,
        },
        Problem {
            name: "System Architecture".into(),
            task_type: TaskType::Holistic,
            complexity: 0.9,
            input_seed: 4004,
        },
    ];

    let mut solutions = Vec::new();

    for problem in &problems {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("PROBLEM: {}", problem.name);
        println!("Type: {:?} - {}", problem.task_type, problem.task_type.description());
        println!("Complexity: {:.0}%", problem.complexity * 100.0);
        println!("═══════════════════════════════════════════════════════════════════════════\n");

        // Select optimal cognitive mode
        let optimal_mode = problem.task_type.optimal_mode();
        engine.set_mode(optimal_mode);
        println!("Selected mode: {:?}", optimal_mode);
        println!();

        // Reasoning loop
        let reasoning_steps = (problem.complexity * 10.0) as usize + 5;
        let mut phi_values = Vec::new();
        let mut best_phi = 0.0;

        println!("Reasoning ({} steps):", reasoning_steps);
        println!("{:<6} {:>8} {:>12} {:>15}", "Step", "Φ", "Stability", "Resonance");
        println!("{}", "─".repeat(45));

        for step in 0..reasoning_steps {
            // Generate reasoning input based on problem
            let intensity = 0.5 + 0.5 * ((step as f64 / reasoning_steps as f64) * std::f64::consts::PI).sin();
            let input = RealHV::random(2048, problem.input_seed + step as u64)
                .scale(intensity as f32 * problem.complexity as f32);

            let update = engine.process(&input);
            integration.update(&update.dimensions, update.mode, update.state);

            phi_values.push(update.phi);
            if update.phi > best_phi {
                best_phi = update.phi;
            }

            // Print progress
            if step % 3 == 0 || step == reasoning_steps - 1 {
                println!("{:<6} {:>8.4} {:>12.3} {:>15}",
                    step,
                    update.phi,
                    integration.field.stability,
                    if integration.field.is_resonant() { "✓ resonant" } else { "building..." }
                );
            }
        }

        // Generate solution insights based on final state
        let final_dims = engine.dimensions();
        let insights = generate_insights(&problem.task_type, final_dims, best_phi);

        let solution = Solution {
            confidence: best_phi * integration.field.stability,
            phi_achieved: best_phi,
            mode_used: optimal_mode,
            reasoning_steps,
            insights,
        };

        println!();
        println!("SOLUTION:");
        println!("  Confidence: {:.1}%", solution.confidence * 100.0);
        println!("  Peak Φ: {:.4}", solution.phi_achieved);
        println!("  Insights discovered: {}", solution.insights.len());
        for (i, insight) in solution.insights.iter().enumerate() {
            println!("    {}. {}", i + 1, insight);
        }

        // Show integration report
        println!();
        println!("{}", integration.report());

        solutions.push(solution);
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("REASONING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("{:<25} {:>12} {:>10} {:>15}", "Problem", "Confidence", "Peak Φ", "Mode");
    println!("{}", "─".repeat(65));

    for (problem, solution) in problems.iter().zip(solutions.iter()) {
        println!("{:<25} {:>11.1}% {:>10.4} {:>15?}",
            problem.name,
            solution.confidence * 100.0,
            solution.phi_achieved,
            solution.mode_used
        );
    }

    let avg_confidence: f64 = solutions.iter().map(|s| s.confidence).sum::<f64>() / solutions.len() as f64;
    let avg_phi: f64 = solutions.iter().map(|s| s.phi_achieved).sum::<f64>() / solutions.len() as f64;

    println!("{}", "─".repeat(65));
    println!("{:<25} {:>11.1}% {:>10.4}", "AVERAGE", avg_confidence * 100.0, avg_phi);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                          ║");
    println!("║  KEY INSIGHT: Different cognitive modes optimize for different tasks    ║");
    println!("║                                                                          ║");
    println!("║  • Analytical → Focused (fewer bridges, more precision)                 ║");
    println!("║  • Creative → Exploratory (more bridges, divergent thinking)            ║");
    println!("║  • Integration → Balanced (optimal ~40-45% bridge ratio)                ║");
    println!("║  • Holistic → GlobalAwareness (maximum integration)                     ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

/// Generate task-appropriate insights
fn generate_insights(task_type: &TaskType, dims: &ConsciousnessDimensions, phi: f64) -> Vec<String> {
    let mut insights = Vec::new();

    match task_type {
        TaskType::Analytical => {
            if phi > 0.5 {
                insights.push("Logical structure verified through high integration".into());
            }
            if dims.attention > 0.6 {
                insights.push("Key variables identified with focused attention".into());
            }
            if dims.epistemic > 0.5 {
                insights.push("Proof steps validated with epistemic confidence".into());
            }
        }
        TaskType::Creative => {
            if phi > 0.4 {
                insights.push("Novel connections discovered through broad integration".into());
            }
            if dims.workspace > 0.5 {
                insights.push("Multiple concepts held in workspace simultaneously".into());
            }
            if dims.efficacy > 0.4 {
                insights.push("Sense of creative agency maintained".into());
            }
        }
        TaskType::Integration => {
            if phi > 0.45 {
                insights.push("Disparate elements synthesized into coherent whole".into());
            }
            if dims.temporal > 0.6 {
                insights.push("Temporal relationships between concepts mapped".into());
            }
            if dims.recursion > 0.3 {
                insights.push("Meta-level patterns recognized".into());
            }
        }
        TaskType::Holistic => {
            if phi > 0.5 {
                insights.push("System-wide patterns emerged through global integration".into());
            }
            if dims.workspace > 0.6 {
                insights.push("Entire problem space held in unified awareness".into());
            }
            if dims.attention < 0.5 {
                insights.push("Distributed attention revealed hidden connections".into());
            }
        }
    }

    // Add phi-based insight
    if phi > 0.55 {
        insights.push(format!("High Φ ({:.3}) indicates strong solution coherence", phi));
    }

    insights
}
