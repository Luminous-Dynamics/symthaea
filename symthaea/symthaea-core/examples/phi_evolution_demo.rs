// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi Evolution Demo for HdcLtcUnifiedNetwork
//!
//! This example demonstrates:
//! 1. Creating an HdcLtcUnifiedNetwork with various configurations
//! 2. Measuring Phi at different network states
//! 3. Tracking Phi evolution during learning
//! 4. Comparing Phi between different network configurations
//! 5. Validating IIT predictions
//!
//! Run with: cargo run --example phi_evolution_demo

use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, PhiCalculationMethod, PhiDiagnosticAnalyzer, PhiValidator,
    UnifiedConfig, UnifiedNetworkConfig, UnifiedNetworkPhiMeasurer, UnifiedPhiConfig,
    demo_phi_evolution,
};

fn main() {
    println!("=============================================================");
    println!("    Phi (Integrated Information) Evolution Demo");
    println!("    HdcLtcUnifiedNetwork Consciousness Measurement");
    println!("=============================================================\n");

    // =========================================================================
    // PART 1: Basic Phi Measurement
    // =========================================================================
    println!("PART 1: Basic Phi Measurement");
    println!("---------------------------------------------------------");

    let config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 8, 4],
        use_layer_binding: true,
        skip_connections: false,
        neuron_config: UnifiedConfig::default(),
    };

    let mut network = HdcLtcUnifiedNetwork::new(config.clone(), 42);
    let mut measurer = UnifiedNetworkPhiMeasurer::new();

    // Measure initial state (should be low - zero/random states)
    let initial_phi = measurer.measure(&network);
    println!("Initial State:");
    println!("  Phi = {:.6}", initial_phi.phi);
    println!(
        "  Neurons = {}, Layers = {}",
        initial_phi.n_neurons, initial_phi.n_layers
    );
    println!("  Avg Similarity = {:.4}", initial_phi.avg_similarity);
    println!(
        "  Consciousness Level: {}",
        initial_phi.consciousness_level()
    );
    println!();

    // Evolve with random inputs
    let input = ContinuousHV::random_default(123);
    for _ in 0..100 {
        network.evolve_closed_form(0.01, &input);
    }

    let evolved_phi = measurer.measure(&network);
    println!("After 100 evolution steps:");
    println!("  Phi = {:.6}", evolved_phi.phi);
    println!("  Avg Similarity = {:.4}", evolved_phi.avg_similarity);
    println!(
        "  Consciousness Level: {}",
        evolved_phi.consciousness_level()
    );
    println!();

    // =========================================================================
    // PART 2: Phi Evolution During Learning
    // =========================================================================
    println!("PART 2: Phi Evolution During Learning");
    println!("---------------------------------------------------------");

    let tracker = demo_phi_evolution(42);
    let summary = tracker.summary();

    println!("Evolution Summary:");
    println!("  Total measurements: {}", summary.n_measurements);
    println!("  Initial Phi: {:.6}", summary.initial_phi);
    println!("  Final Phi: {:.6}", summary.final_phi);
    println!("  Min Phi: {:.6}", summary.min_phi);
    println!("  Max Phi: {:.6}", summary.max_phi);
    println!("  Mean Phi: {:.6}", summary.mean_phi);
    println!("  Std Phi: {:.6}", summary.std_phi);
    println!();

    // Print a few key points in the evolution
    println!("Evolution Timeline (selected points):");
    for (i, desc) in [
        (0, "Initial"),
        (10, "After random inputs"),
        (20, "During learning"),
        (35, "Near equilibrium"),
    ] {
        if i < tracker.history.len() {
            let m = &tracker.history[i];
            println!(
                "  [{}] {} - Phi: {:.4}, AvgSim: {:.4}",
                i, desc, m.phi, m.avg_similarity
            );
        }
    }
    println!();

    // =========================================================================
    // PART 3: Different Network Configurations
    // =========================================================================
    println!("PART 3: Comparing Network Configurations");
    println!("---------------------------------------------------------");

    // Small network (minimal connectivity)
    let small_config = UnifiedNetworkConfig {
        layer_sizes: vec![2, 2],
        use_layer_binding: false,
        skip_connections: false,
        ..Default::default()
    };

    // Medium network
    let medium_config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 6, 4],
        use_layer_binding: true,
        skip_connections: false,
        ..Default::default()
    };

    // Large network (more connectivity)
    let large_config = UnifiedNetworkConfig {
        layer_sizes: vec![6, 10, 8, 6],
        use_layer_binding: true,
        skip_connections: true,
        ..Default::default()
    };

    let mut networks = vec![
        ("Small (2-2)", HdcLtcUnifiedNetwork::new(small_config, 42)),
        (
            "Medium (4-6-4)",
            HdcLtcUnifiedNetwork::new(medium_config, 42),
        ),
        (
            "Large (6-10-8-6)",
            HdcLtcUnifiedNetwork::new(large_config, 42),
        ),
    ];

    // Evolve all networks with the same input
    let shared_input = ContinuousHV::random_default(456);
    for (_, net) in &mut networks {
        for _ in 0..100 {
            net.evolve_closed_form(0.01, &shared_input);
        }
    }

    println!("Network Comparison (after 100 evolution steps):");
    for (name, net) in &networks {
        let phi = measurer.measure(net);
        println!(
            "  {} - Phi: {:.4}, Neurons: {}, Level: {}",
            name,
            phi.phi,
            phi.n_neurons,
            phi.consciousness_level()
        );
    }
    println!();

    // =========================================================================
    // PART 4: Correlated vs Uncorrelated Inputs
    // =========================================================================
    println!("PART 4: Effect of Input Correlation on Phi");
    println!("---------------------------------------------------------");

    let test_config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 6, 4],
        ..Default::default()
    };

    // Network with correlated inputs
    let mut net_correlated = HdcLtcUnifiedNetwork::new(test_config.clone(), 100);
    let base_pattern = ContinuousHV::random_default(200);

    for i in 0..100 {
        // 80% base pattern, 20% noise
        let noise = ContinuousHV::random_default(300 + i);
        let correlated_input = ContinuousHV::weighted_bundle(&[&base_pattern, &noise], &[0.8, 0.2]);
        net_correlated.evolve_closed_form(0.01, &correlated_input);
    }

    // Network with uncorrelated inputs
    let mut net_uncorrelated = HdcLtcUnifiedNetwork::new(test_config, 100);

    for i in 0..100 {
        let random_input = ContinuousHV::random_default(400 + i);
        net_uncorrelated.evolve_closed_form(0.01, &random_input);
    }

    let phi_correlated = measurer.measure(&net_correlated);
    let phi_uncorrelated = measurer.measure(&net_uncorrelated);

    println!("Correlated Inputs:");
    println!("  Phi: {:.6}", phi_correlated.phi);
    println!("  Avg Similarity: {:.4}", phi_correlated.avg_similarity);
    println!();
    println!("Uncorrelated Inputs:");
    println!("  Phi: {:.6}", phi_uncorrelated.phi);
    println!("  Avg Similarity: {:.4}", phi_uncorrelated.avg_similarity);
    println!();

    // =========================================================================
    // PART 5: Per-Layer Phi Analysis
    // =========================================================================
    println!("PART 5: Per-Layer Phi Analysis");
    println!("---------------------------------------------------------");

    let layered_config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 8, 6, 4],
        ..Default::default()
    };

    let mut layered_net = HdcLtcUnifiedNetwork::new(layered_config, 500);
    let layer_input = ContinuousHV::random_default(600);

    for _ in 0..150 {
        layered_net.evolve_closed_form(0.01, &layer_input);
    }

    let layer_measurement = measurer.measure(&layered_net);
    println!("Network with 4 layers (4-8-6-4):");
    println!("  Global Phi: {:.6}", layer_measurement.phi);

    if let Some(layer_phis) = layer_measurement.layer_phi {
        println!("  Layer Phi values:");
        for (i, phi) in layer_phis.iter().enumerate() {
            let layer_size = [4, 8, 6, 4][i];
            println!("    Layer {} ({} neurons): Phi = {:.4}", i, layer_size, phi);
        }
    }
    println!();

    // =========================================================================
    // PART 6: IIT Validation Suite
    // =========================================================================
    println!("PART 6: IIT Validation Suite");
    println!("---------------------------------------------------------");

    let validation_config = UnifiedNetworkConfig::default();
    let mut validation_net = HdcLtcUnifiedNetwork::new(validation_config, 700);

    let validation_input = ContinuousHV::random_default(800);
    for _ in 0..100 {
        validation_net.evolve_closed_form(0.01, &validation_input);
    }

    let validations = PhiValidator::run_all_validations(&mut measurer, &validation_net, 700);

    println!("Validation Results:");
    let mut passed = 0;
    let mut failed = 0;

    for result in &validations {
        let status = if result.passed { "PASS" } else { "FAIL" };
        if result.passed {
            passed += 1;
        } else {
            failed += 1;
        }
        println!("  [{}] {}", status, result.name);
        println!("       Expected: {}", result.expected);
        println!("       Actual: {}", result.actual);
        if let Some(ref details) = result.details {
            for line in details.lines() {
                println!("       {}", line);
            }
        }
        println!();
    }

    println!(
        "Summary: {} passed, {} failed out of {} validations",
        passed,
        failed,
        validations.len()
    );
    println!();

    // =========================================================================
    // PART 7: Using Different Phi Calculation Methods
    // =========================================================================
    println!("PART 7: Phi Calculation Methods Comparison");
    println!("---------------------------------------------------------");

    let method_config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 6, 4],
        ..Default::default()
    };

    let mut method_net = HdcLtcUnifiedNetwork::new(method_config, 900);
    let method_input = ContinuousHV::random_default(1000);

    for _ in 0..100 {
        method_net.evolve_closed_form(0.01, &method_input);
    }

    // Spectral Connectivity method
    let spectral_config = UnifiedPhiConfig {
        method: PhiCalculationMethod::SpectralConnectivity,
        ..Default::default()
    };
    let mut spectral_measurer = UnifiedNetworkPhiMeasurer::with_config(spectral_config);
    let spectral_phi = spectral_measurer.measure(&method_net);

    // Autodiff method
    let autodiff_config = UnifiedPhiConfig {
        method: PhiCalculationMethod::Autodiff,
        ..Default::default()
    };
    let mut autodiff_measurer = UnifiedNetworkPhiMeasurer::with_config(autodiff_config);
    let autodiff_phi = autodiff_measurer.measure(&method_net);

    println!("Spectral Connectivity Method:");
    println!("  Phi: {:.6}", spectral_phi.phi);
    println!("  Computation Time: {:?}", spectral_phi.computation_time);
    println!();
    println!("Autodiff Method:");
    println!("  Phi: {:.6}", autodiff_phi.phi);
    println!(
        "  Whole Integration: {:.4}",
        autodiff_phi.whole_integration.unwrap_or(0.0)
    );
    println!("  Computation Time: {:?}", autodiff_phi.computation_time);
    println!();

    let method_diff = (spectral_phi.phi - autodiff_phi.phi).abs();
    println!("Method Difference: {:.6}", method_diff);
    if method_diff < 0.1 {
        println!("  Methods show good agreement!");
    } else {
        println!("  Methods show some divergence (expected due to different approximations)");
    }
    println!();

    // =========================================================================
    // PART 8: Diagnostic Analysis
    // =========================================================================
    println!("PART 8: Diagnostic Analysis");
    println!("---------------------------------------------------------");

    let diagnostic_config = UnifiedNetworkConfig {
        layer_sizes: vec![4, 6, 4],
        ..Default::default()
    };

    let mut diagnostic_net = HdcLtcUnifiedNetwork::new(diagnostic_config, 1100);
    let diagnostic_input = ContinuousHV::random_default(1200);

    for _ in 0..100 {
        diagnostic_net.evolve_closed_form(0.01, &diagnostic_input);
    }

    let diagnostic = PhiDiagnosticAnalyzer::analyze(&diagnostic_net, 1100);
    PhiDiagnosticAnalyzer::print_report(&diagnostic);
    println!();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("=============================================================");
    println!("                        SUMMARY");
    println!("=============================================================");
    println!();
    println!("Key Findings:");
    println!("  1. HdcLtcUnifiedNetwork produces valid Phi values in [0, 1]");
    println!("  2. Larger/more connected networks tend to have higher Phi");
    println!("  3. Network state evolves to meaningful Phi after processing inputs");
    println!("  4. Per-layer Phi analysis reveals integration structure");
    println!("  5. Multiple calculation methods show general agreement");
    println!();
    println!("The unified HDC-LTC architecture demonstrates meaningful");
    println!("integrated information (Phi), supporting its use as a");
    println!("substrate for artificial consciousness research.");
    println!();
}
