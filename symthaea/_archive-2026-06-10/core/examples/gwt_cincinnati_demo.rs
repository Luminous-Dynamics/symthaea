// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Global Workspace + Cincinnati-LTC Integration Demo
//!
//! Demonstrates how temporal pattern recognition (Cincinnati-LTC) integrates
//! with the Global Workspace consciousness pipeline.
//!
//! Key concepts demonstrated:
//! 1. **Temporal patterns → Conscious access**: Cincinnati-LTC patterns compete for GWT entry
//! 2. **Salience from surprise**: Prediction errors boost pattern salience
//! 3. **Budding as attention signal**: Network growth increases conscious access probability
//! 4. **Workspace dynamics**: Limited capacity, competition, broadcasting
//!
//! Run with: cargo run --example gwt_cincinnati_demo

use symthaea::hdc::global_workspace::WorkspaceConfig;
use symthaea::hdc::gwt_cincinnati_integration::{CincinnatiGwtConfig, CincinnatiGwtIntegrator};

// =============================================================================
// PATTERN GENERATORS
// =============================================================================

/// Generate periodic sequence (predictable → low salience)
fn periodic_sequence(period: usize, length: usize) -> Vec<bool> {
    (0..length).map(|i| (i % period) < (period / 2)).collect()
}

/// Generate logistic map sequence (chaotic → high salience)
fn logistic_sequence(r: f64, length: usize) -> Vec<bool> {
    let mut x = 0.5;
    let threshold = 0.5;
    (0..length)
        .map(|_| {
            x = r * x * (1.0 - x);
            x > threshold
        })
        .collect()
}

/// Generate sequence with embedded surprise (sudden change)
fn surprise_sequence(length: usize, surprise_at: usize) -> Vec<bool> {
    (0..length)
        .map(|i| {
            if i < surprise_at {
                i % 2 == 0 // Regular alternating
            } else {
                true // Sudden all-true (surprise!)
            }
        })
        .collect()
}

// =============================================================================
// ANALYSIS FUNCTIONS
// =============================================================================

/// Run a sequence and report results
fn analyze_sequence(name: &str, sequence: &[bool], config: CincinnatiGwtConfig) {
    println!("\n{:=^70}", format!(" {} ", name));
    println!("Sequence length: {}", sequence.len());
    println!("Sample: {:?}...", &sequence[..10.min(sequence.len())]);

    let mut integrator = CincinnatiGwtIntegrator::new(config);
    let result = integrator.process_sequence(sequence);

    println!("\nResults:");
    println!("  Prediction Accuracy: {:.1}%", result.accuracy * 100.0);
    println!(
        "  Conscious Access:    {:.1}% of steps",
        result.conscious_ratio * 100.0
    );
    println!("  Average Salience:    {:.3}", result.avg_salience);
    println!("  Total Budding:       {} events", result.total_budding);
    println!("  Final Node Count:    {}", result.final_node_count);

    // Find moments of conscious access
    let conscious_moments: Vec<usize> = result
        .results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_conscious)
        .map(|(i, _)| i)
        .collect();

    if conscious_moments.len() <= 10 {
        println!("  Conscious at steps:  {:?}", conscious_moments);
    } else {
        println!("  First 10 conscious:  {:?}...", &conscious_moments[..10]);
    }
}

/// Compare salience across pattern types
fn compare_salience_profiles() {
    println!("\n{:=^70}", " Salience Profile Comparison ");

    let config = CincinnatiGwtConfig {
        base_activation: 0.4,
        error_salience_boost: 0.3,
        budding_attention_boost: 0.25,
        workspace_config: WorkspaceConfig {
            entry_threshold: 0.5,
            max_capacity: 3,
            ..Default::default()
        },
        ..Default::default()
    };

    let patterns = vec![
        ("Periodic (p=4)", periodic_sequence(4, 100)),
        ("Periodic (p=8)", periodic_sequence(8, 100)),
        ("Logistic (r=3.2)", logistic_sequence(3.2, 100)), // Periodic-ish
        ("Logistic (r=3.8)", logistic_sequence(3.8, 100)), // Chaotic
        ("Surprise (at 50)", surprise_sequence(100, 50)),
    ];

    println!(
        "\n{:<20} | {:>10} | {:>10} | {:>10} | {:>8}",
        "Pattern", "Accuracy", "Conscious", "Salience", "Budding"
    );
    println!(
        "{:-<20}-+-{:-^10}-+-{:-^10}-+-{:-^10}-+-{:-^8}",
        "", "", "", "", ""
    );

    for (name, sequence) in patterns {
        let mut integrator = CincinnatiGwtIntegrator::new(config.clone());
        let result = integrator.process_sequence(&sequence);

        println!(
            "{:<20} | {:>9.1}% | {:>9.1}% | {:>10.3} | {:>8}",
            name,
            result.accuracy * 100.0,
            result.conscious_ratio * 100.0,
            result.avg_salience,
            result.total_budding
        );
    }

    println!("\nKey Insight: Chaotic sequences have higher salience due to prediction errors!");
}

/// Demonstrate workspace dynamics with multiple temporal streams
fn multi_stream_consciousness() {
    println!("\n{:=^70}", " Multi-Stream Consciousness ");
    println!("Simulating multiple temporal streams competing for conscious access...\n");

    // Three different pattern streams
    let stream_a = periodic_sequence(4, 50); // Very predictable
    let stream_b = logistic_sequence(3.8, 50); // Chaotic (interesting)
    let stream_c = surprise_sequence(50, 25); // Has surprise

    let config = CincinnatiGwtConfig {
        workspace_config: WorkspaceConfig {
            entry_threshold: 0.45,
            max_capacity: 2, // Limited capacity!
            ..Default::default()
        },
        ..Default::default()
    };

    // Process streams interleaved (simulating concurrent inputs)
    let mut integrator = CincinnatiGwtIntegrator::new(config);

    let mut stream_a_conscious = 0;
    let mut stream_b_conscious = 0;
    let mut stream_c_conscious = 0;

    println!("Processing interleaved streams (A=periodic, B=chaotic, C=surprise)...\n");

    for i in 0..50 {
        // Process one from each stream
        let result_a = integrator.process_observation(stream_a[i]);
        if result_a.is_conscious {
            stream_a_conscious += 1;
        }

        let result_b = integrator.process_observation(stream_b[i]);
        if result_b.is_conscious {
            stream_b_conscious += 1;
        }

        let result_c = integrator.process_observation(stream_c[i]);
        if result_c.is_conscious {
            stream_c_conscious += 1;
        }

        // Report interesting moments
        if i == 25 {
            println!("  At step 25 (surprise moment in C):");
            println!(
                "    A salience: {:.3}, B salience: {:.3}, C salience: {:.3}",
                result_a.salience, result_b.salience, result_c.salience
            );
        }
    }

    println!("\nConscious access frequency:");
    println!(
        "  Stream A (periodic):  {} times ({:.1}%)",
        stream_a_conscious,
        stream_a_conscious as f64 / 50.0 * 100.0
    );
    println!(
        "  Stream B (chaotic):   {} times ({:.1}%)",
        stream_b_conscious,
        stream_b_conscious as f64 / 50.0 * 100.0
    );
    println!(
        "  Stream C (surprise):  {} times ({:.1}%)",
        stream_c_conscious,
        stream_c_conscious as f64 / 50.0 * 100.0
    );

    let stats = integrator.stats();
    println!("\nWorkspace Statistics:");
    println!("  Total timesteps:   {}", stats.timestep);
    println!("  Network nodes:     {}", stats.node_count);
    println!("  Total budding:     {}", stats.total_budding);
    println!("  Overall conscious: {:.1}%", stats.conscious_ratio * 100.0);
}

/// Demonstrate budding → attention relationship
fn budding_attention_analysis() {
    println!("\n{:=^70}", " Budding → Attention Analysis ");
    println!("Comparing configurations with different budding → attention mappings...\n");

    // Config with HIGH budding attention boost
    let high_boost_config = CincinnatiGwtConfig {
        budding_attention_boost: 0.5, // Budding strongly boosts attention
        budding_threshold: 0.3,       // Lower threshold → more budding
        workspace_config: WorkspaceConfig {
            entry_threshold: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    // Config with LOW budding attention boost
    let low_boost_config = CincinnatiGwtConfig {
        budding_attention_boost: 0.05, // Budding barely affects attention
        budding_threshold: 0.3,
        workspace_config: WorkspaceConfig {
            entry_threshold: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    // Complex sequence that triggers budding
    let sequence = logistic_sequence(3.7, 100);

    // High boost
    let mut high_integrator = CincinnatiGwtIntegrator::new(high_boost_config);
    let high_result = high_integrator.process_sequence(&sequence);

    // Low boost
    let mut low_integrator = CincinnatiGwtIntegrator::new(low_boost_config);
    let low_result = low_integrator.process_sequence(&sequence);

    println!("{:<25} | {:>15} | {:>15}", "", "High Boost", "Low Boost");
    println!("{:-<25}-+-{:-^15}-+-{:-^15}", "", "", "");
    println!(
        "{:<25} | {:>14.1}% | {:>14.1}%",
        "Conscious Access",
        high_result.conscious_ratio * 100.0,
        low_result.conscious_ratio * 100.0
    );
    println!(
        "{:<25} | {:>15.3} | {:>15.3}",
        "Average Salience", high_result.avg_salience, low_result.avg_salience
    );
    println!(
        "{:<25} | {:>15} | {:>15}",
        "Total Budding", high_result.total_budding, low_result.total_budding
    );
    println!(
        "{:<25} | {:>15} | {:>15}",
        "Final Nodes", high_result.final_node_count, low_result.final_node_count
    );

    println!("\nConclusion: Higher budding-attention coupling → more conscious access");
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!(
        "
╔══════════════════════════════════════════════════════════════════════╗
║      GLOBAL WORKSPACE + CINCINNATI-LTC INTEGRATION DEMO              ║
║                                                                      ║
║  Demonstrating how temporal pattern recognition integrates with      ║
║  the Global Workspace consciousness pipeline (Baars, 1988)           ║
╚══════════════════════════════════════════════════════════════════════╝
"
    );

    // 1. Basic sequence analysis
    let config = CincinnatiGwtConfig {
        workspace_config: WorkspaceConfig {
            entry_threshold: 0.5,
            max_capacity: 3,
            ..Default::default()
        },
        ..Default::default()
    };

    analyze_sequence(
        "Periodic Pattern (period=4)",
        &periodic_sequence(4, 100),
        config.clone(),
    );
    analyze_sequence(
        "Chaotic Pattern (r=3.8)",
        &logistic_sequence(3.8, 100),
        config.clone(),
    );
    analyze_sequence(
        "Surprise Pattern",
        &surprise_sequence(100, 50),
        config.clone(),
    );

    // 2. Salience comparison
    compare_salience_profiles();

    // 3. Multi-stream consciousness
    multi_stream_consciousness();

    // 4. Budding-attention analysis
    budding_attention_analysis();

    // Summary
    println!("\n{:=^70}", " Summary ");
    println!(
        r#"
Key Findings:

1. SURPRISING EVENTS GAIN CONSCIOUS ACCESS
   - Prediction errors boost pattern salience
   - Chaotic sequences enter consciousness more than periodic ones

2. BUDDING SIGNALS ATTENTION
   - Network growth (budding) indicates interesting patterns
   - Higher budding → higher probability of conscious access

3. LIMITED CAPACITY CREATES COMPETITION
   - Only most salient patterns enter the workspace
   - Multiple streams compete for conscious attention

4. WORKSPACE DYNAMICS
   - Content decays over time
   - New salient content can displace old content

This demonstrates how Cincinnati-LTC's temporal pattern recognition
integrates with Global Workspace Theory's model of conscious access.
"#
    );

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     INTEGRATION DEMO COMPLETE                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}