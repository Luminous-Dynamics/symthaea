// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train the CfC code sequencer on synthetic + augmented training data.
//!
//! Run: `cargo run --example train_code_sequencer --features code_generation`
//!
//! This generates training data from the 100-function benchmark + structural
//! examples + synonym augmentation, trains the CfC sequencer with BPTT + Adam
//! (early stopping), and saves the trained weights.

use symthaea::dynamics::cfc_code_sequencer::CfCCodeSequencer;
use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::sequencer_training::{
    evaluate_sequencer, generate_training_data, train_sequencer,
};

fn main() {
    println!("=== CfC Code Sequencer Training ===\n");

    // 1. Generate training data
    let encoder = CodeHDEncoder::new(512);
    let (train_data, eval_data) = generate_training_data(&encoder);
    println!(
        "Training examples: {} (train: {}, eval: {})",
        train_data.len() + eval_data.len(),
        train_data.len(),
        eval_data.len()
    );

    // 2. Create sequencer (starts with random weights)
    let sequencer = CfCCodeSequencer::default();
    println!(
        "Network parameters: {}",
        sequencer.export_weights().len()
    );

    // 3. Evaluate BEFORE training (baseline)
    println!("\n--- Pre-Training Evaluation ---");
    let pre_eval = evaluate_sequencer(&sequencer, &eval_data);
    for (cat, accuracy) in &pre_eval {
        println!("  {:20} {:.1}%", cat.as_str(), accuracy * 100.0);
    }
    let pre_avg: f32 = if pre_eval.is_empty() {
        0.0
    } else {
        pre_eval.iter().map(|(_, a)| a).sum::<f32>() / pre_eval.len() as f32
    };
    println!("  Average accuracy:  {:.1}%", pre_avg * 100.0);

    // 4. Train
    println!("\n--- Training (50 epochs, lr=0.01, patience=5) ---");
    let result = train_sequencer(&sequencer, &train_data, &eval_data, 50, 0.01, 5);

    println!("  Epochs completed:  {}", result.epochs);
    println!("  Best epoch:        {}", result.best_epoch);
    println!("  Final train loss:  {:.4}", result.final_train_loss);
    println!("  Best eval loss:    {:.4}", result.best_eval_loss);

    // 5. Evaluate AFTER training
    println!("\n--- Post-Training Evaluation ---");
    let post_eval = evaluate_sequencer(&sequencer, &eval_data);
    for (cat, accuracy) in &post_eval {
        println!("  {:20} {:.1}%", cat.as_str(), accuracy * 100.0);
    }
    let post_avg: f32 = if post_eval.is_empty() {
        0.0
    } else {
        post_eval.iter().map(|(_, a)| a).sum::<f32>() / post_eval.len() as f32
    };
    println!("  Average accuracy:  {:.1}%", post_avg * 100.0);

    // 6. Improvement
    println!("\n--- Improvement ---");
    println!(
        "  Accuracy: {:.1}% → {:.1}% ({:+.1}%)",
        pre_avg * 100.0,
        post_avg * 100.0,
        (post_avg - pre_avg) * 100.0
    );

    // 7. Save weights
    match sequencer.persist_weights() {
        Ok(()) => {
            let path = CfCCodeSequencer::default_weights_path();
            println!("\n  Weights saved to: {}", path.display());
        }
        Err(e) => {
            eprintln!("\n  Failed to save weights: {}", e);
        }
    }

    println!("\n=== Training Complete ===");
}
