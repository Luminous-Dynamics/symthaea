//! Primitive Learning with Cognitive Loop - Comparison Benchmark
//!
//! Tests whether the bidirectional HDC↔LTC cognitive loop improves accuracy
//! over the original one-way pipeline that achieved 0% accuracy.
//!
//! Run with:
//!   cargo run --example primitive_learning_cognitive --release

use symthaea::benchmarks::{
    CognitivePrimitiveLearning,
    CognitivePrimitiveLearningConfig,
    CognitivePrimitiveTask,
    CognitivePrimitiveLearningResults,
    print_comparison_results,
};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   COGNITIVE LOOP vs ORIGINAL PIPELINE - Accuracy Test        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("This benchmark compares:");
    println!("  1. Original Pipeline: Input → HDC → LTC → Output (one-way)");
    println!("  2. Cognitive Loop: HDC ↔ LTC bidirectional with prediction error\n");

    println!("The original Primitive Learning Gate showed:");
    println!("  • Loss: 0.0667 → 0.0037 (18x reduction) - LTC WAS learning!");
    println!("  • Accuracy: 0% - But predictions were wrong\n");

    println!("The hypothesis: Bidirectional coupling creates emergent learning\n");

    // Configure with tuned parameters (Step 1: Parameter Tuning)
    // - Increased learning rate: 0.01 → 0.03 (3x for faster convergence)
    // - Increased attention lr: 0.05 → 0.1 (2x for stronger attention emergence)
    // Note: Random Projection preserves HDC orthogonality (Step 2)
    let config = CognitivePrimitiveLearningConfig {
        num_episodes: 15,          // Increased from 10
        cycles_per_episode: 100,   // Increased from 50
        test_size: 20,
        learning_rate: 0.03,       // 3x base
        attention_lr: 0.1,         // 2x base
    };

    println!("Configuration:");
    println!("  Episodes: {}", config.num_episodes);
    println!("  Cycles per episode: {}", config.cycles_per_episode);
    println!("  Test patterns: {}", config.test_size);
    println!("  LTC learning rate: {}", config.learning_rate);
    println!("  Attention learning rate: {}\n", config.attention_lr);

    let benchmark = CognitivePrimitiveLearning::new(config);

    // Run all 4 tasks
    let tasks = [
        CognitivePrimitiveTask::CausalChain,
        CognitivePrimitiveTask::LogicalInference,
        CognitivePrimitiveTask::ActionSequence,
        CognitivePrimitiveTask::TemporalOrder,
    ];

    let mut all_cognitive: Vec<CognitivePrimitiveLearningResults> = Vec::new();
    let mut all_original: Vec<CognitivePrimitiveLearningResults> = Vec::new();

    println!("Running {} reasoning tasks with both approaches...\n", tasks.len());

    for task in &tasks {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Task: {}", task.name());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let (cognitive_result, original_result) = benchmark.run_comparison(*task)?;

        // Print individual comparison
        print_comparison_results(&cognitive_result, &original_result);

        all_cognitive.push(cognitive_result);
        all_original.push(original_result);
    }

    // Overall summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    OVERALL SUMMARY                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("┌──────────────────────────┬────────────────┬────────────────┬──────────────┐");
    println!("│ Task                     │ Original Acc   │ Cognitive Acc  │ Improvement  │");
    println!("├──────────────────────────┼────────────────┼────────────────┼──────────────┤");

    let mut total_original_acc = 0.0_f32;
    let mut total_cognitive_acc = 0.0_f32;

    for (cog, orig) in all_cognitive.iter().zip(all_original.iter()) {
        let improvement = cog.final_accuracy - orig.final_accuracy;
        let improvement_str = if improvement > 0.0 {
            format!("+{:.1}%", improvement * 100.0)
        } else if improvement < 0.0 {
            format!("{:.1}%", improvement * 100.0)
        } else {
            "0.0%".to_string()
        };

        println!("│ {:24} │ {:>13.1}% │ {:>13.1}% │ {:>12} │",
                 cog.task_name,
                 orig.final_accuracy * 100.0,
                 cog.final_accuracy * 100.0,
                 improvement_str);

        total_original_acc += orig.final_accuracy;
        total_cognitive_acc += cog.final_accuracy;
    }

    println!("├──────────────────────────┼────────────────┼────────────────┼──────────────┤");

    let avg_original = total_original_acc / tasks.len() as f32;
    let avg_cognitive = total_cognitive_acc / tasks.len() as f32;
    let avg_improvement = avg_cognitive - avg_original;

    let improvement_str = if avg_improvement > 0.0 {
        format!("+{:.1}%", avg_improvement * 100.0)
    } else if avg_improvement < 0.0 {
        format!("{:.1}%", avg_improvement * 100.0)
    } else {
        "0.0%".to_string()
    };

    println!("│ {:24} │ {:>13.1}% │ {:>13.1}% │ {:>12} │",
             "AVERAGE",
             avg_original * 100.0,
             avg_cognitive * 100.0,
             improvement_str);
    println!("└──────────────────────────┴────────────────┴────────────────┴──────────────┘");

    // Summary verdict
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      FINAL VERDICT                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if avg_cognitive > avg_original + 0.05 {
        println!("✅ HYPOTHESIS CONFIRMED!");
        println!();
        println!("The bidirectional HDC↔LTC cognitive loop significantly");
        println!("improves accuracy over the one-way pipeline.");
        println!();
        println!("  Original average:  {:.1}%", avg_original * 100.0);
        println!("  Cognitive average: {:.1}%", avg_cognitive * 100.0);
        println!("  Improvement:       {:.1}%", avg_improvement * 100.0);
        println!();
        println!("This validates the core insight: prediction error creates");
        println!("the missing feedback signal that allows HDC and LTC to");
        println!("learn together rather than in isolation.");
    } else if avg_cognitive > avg_original {
        println!("🔶 PARTIAL IMPROVEMENT");
        println!();
        println!("The cognitive loop shows some improvement, but not as");
        println!("dramatic as expected.");
        println!();
        println!("  Original average:  {:.1}%", avg_original * 100.0);
        println!("  Cognitive average: {:.1}%", avg_cognitive * 100.0);
        println!("  Improvement:       {:.1}%", avg_improvement * 100.0);
        println!();
        println!("Consider:");
        println!("  • Increasing cycles_per_episode");
        println!("  • Tuning learning rates");
        println!("  • Examining specific failing tasks");
    } else {
        println!("❌ HYPOTHESIS NOT SUPPORTED");
        println!();
        println!("The cognitive loop did not improve accuracy.");
        println!();
        println!("  Original average:  {:.1}%", avg_original * 100.0);
        println!("  Cognitive average: {:.1}%", avg_cognitive * 100.0);
        println!();
        println!("This suggests:");
        println!("  • The feedback mechanism needs refinement");
        println!("  • The tasks may not benefit from temporal prediction");
        println!("  • The compression/expansion may be losing information");
    }

    // Attention emergence summary
    let attention_emerged_count = all_cognitive.iter()
        .filter(|r| r.attention_emerged)
        .count();

    println!("\n📊 Attention Emergence: {}/{} tasks", attention_emerged_count, tasks.len());

    // Total time
    let total_time: u64 = all_cognitive.iter().map(|r| r.total_time_ms).sum::<u64>()
        + all_original.iter().map(|r| r.total_time_ms).sum::<u64>();
    println!("⏱️  Total time: {}ms", total_time);

    Ok(())
}
