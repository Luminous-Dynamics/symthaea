//! Prediction-Gate Experiment: Testing if Symthaea Can Learn
//!
//! ## Background
//!
//! Our Φ-Gate experiment (Phase 2) showed no correlation (r ≈ 0) between
//! consciousness integration (Φ) and reasoning accuracy. This is expected:
//! Φ measures integration, not understanding.
//!
//! ## New Approach: Prediction-Gate
//!
//! Understanding = accurate prediction + generalization
//!
//! Instead of asking "does high Φ mean correct answers?", we ask:
//! "Can the system learn to predict better with experience?"
//!
//! ## Metrics
//!
//! 1. **Prediction Error Reduction (PER)**: How much does error decrease? (>50% to pass)
//! 2. **Generalization Gap (GG)**: Test vs train performance gap (<30% to pass)
//! 3. **Learning Rate (LR)**: Consistent improvement over episodes (>0 to pass)
//! 4. **Retention (R)**: Memory after consolidation (>80% to pass)
//!
//! ## Tasks
//!
//! 1. Sequence Prediction - Continue arithmetic patterns
//! 2. Pattern Classification - Identify sinusoid frequencies
//! 3. Transformation - Learn input→output mappings
//! 4. Dimension Detection - Find dominant dimensions
//! 5. Periodicity Detection - Extract period/amplitude/phase
//!
//! ## Why This Matters
//!
//! If Symthaea passes the Prediction-Gate, it demonstrates genuine learning
//! capability - a prerequisite for any higher cognitive function. This is
//! more fundamental than reasoning benchmarks like MMLU which require
//! pre-existing world knowledge.

use symthaea::benchmarks::prediction_gate::{
    PredictionGate, PredictionGateConfig, PatternTask, TaskType,
    print_prediction_gate_summary,
};
use symthaea::learnable_ltc::LearnableLTCConfig;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      PREDICTION-GATE EXPERIMENT - Phase 3 Validation         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Hypothesis: The system can LEARN to predict better           ║");
    println!("║ Criteria: PER>50%, GG<30%, LR>0, Retention>80%              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configure the experiment
    let config = PredictionGateConfig {
        num_episodes: 15,           // 15 training episodes
        examples_per_episode: 100,   // 100 examples per episode
        test_size: 50,              // 50 held-out test examples
        consolidation_cycles: 5,     // 5 memory consolidation cycles
        pattern_dim: 64,            // 64-dimensional patterns
        output_dim: 16,             // 16-dimensional output
        ltc_config: LearnableLTCConfig {
            num_neurons: 256,       // 256 LTC neurons
            input_dim: 64,          // Match pattern_dim
            output_dim: 16,         // Match output_dim
            num_steps: 30,          // 30 integration steps
            lr_weights: 0.003,      // Weight learning rate
            lr_tau: 0.0003,         // Time constant learning rate (slower)
            lr_bias: 0.003,         // Bias learning rate
            sparsity: 0.15,         // 15% sparse connectivity
            l2_reg: 0.0001,         // L2 regularization
            grad_clip: 1.0,         // Gradient clipping
            ..Default::default()
        },
    };

    println!("Configuration:");
    println!("  Episodes: {}", config.num_episodes);
    println!("  Examples/episode: {}", config.examples_per_episode);
    println!("  Test size: {}", config.test_size);
    println!("  LTC neurons: {}", config.ltc_config.num_neurons);
    println!("  Pattern dimension: {}", config.pattern_dim);
    println!();

    // Create the experiment
    let mut gate = PredictionGate::new(config)?;

    println!("Running all 5 prediction tasks...\n");

    // Run all tasks
    let results = gate.run_all_tasks()?;

    // Print detailed summary
    print_prediction_gate_summary(&results);

    // Print learning curves for each task
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  LEARNING CURVES                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for result in &results {
        println!("Task: {}", result.task_name);
        println!("Episode | Train Loss | Test Loss  | Test Accuracy | Tau Mean");
        println!("--------|------------|------------|---------------|----------");

        for ep in &result.episodes {
            println!(
                "   {:2}   |   {:.4}   |   {:.4}   |    {:.1}%      |  {:.2}",
                ep.episode,
                ep.train_loss,
                ep.test_loss,
                ep.test_accuracy * 100.0,
                ep.tau_mean
            );
        }
        println!();
    }

    // Final verdict
    let passed_count = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let pass_rate = passed_count as f32 / total as f32;

    println!("╔══════════════════════════════════════════════════════════════╗");
    if pass_rate >= 0.6 {
        println!("║  🎉 PREDICTION-GATE: PASSED ({}/{} tasks)                     ║", passed_count, total);
        println!("║                                                              ║");
        println!("║  The system demonstrates genuine learning capability!        ║");
        println!("║  This validates the LearnableLTC + BPTT + Adam architecture  ║");
        println!("║                                                              ║");
        println!("║  Next Steps:                                                 ║");
        println!("║  1. Connect learning to consciousness-guided tasks           ║");
        println!("║  2. Test transfer learning between domains                   ║");
        println!("║  3. Integrate with primitive system for concept learning     ║");
    } else {
        println!("║  ⚠️  PREDICTION-GATE: NOT YET PASSED ({}/{} tasks)            ║", passed_count, total);
        println!("║                                                              ║");
        println!("║  Learning needs improvement. Suggestions:                    ║");
        println!("║  1. Increase training episodes                               ║");
        println!("║  2. Tune learning rates (especially lr_tau)                  ║");
        println!("║  3. Adjust network architecture (neurons, sparsity)          ║");
        println!("║  4. Check for numerical instability in gradients             ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Calculate average metrics
    let avg_per = results.iter().map(|r| r.per).sum::<f32>() / total as f32;
    let avg_gg = results.iter().map(|r| r.generalization_gap).sum::<f32>() / total as f32;
    let avg_lr = results.iter().map(|r| r.learning_rate).sum::<f32>() / total as f32;
    let avg_ret = results.iter().map(|r| r.retention).sum::<f32>() / total as f32;

    println!("Average Metrics Across All Tasks:");
    println!("  Prediction Error Reduction: {:.1}%", avg_per);
    println!("  Generalization Gap: {:.1}%", avg_gg);
    println!("  Learning Rate: {:.4}", avg_lr);
    println!("  Retention: {:.1}%", avg_ret);

    Ok(())
}
