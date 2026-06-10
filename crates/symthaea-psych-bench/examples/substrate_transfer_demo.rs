// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate-Validated Consciousness Transfer Demo
//!
//! Demonstrates Direction C: empirical testing of Multiple Realizability
//! (Putnam 1967) by simulating consciousness transfer between substrates
//! and measuring cognitive preservation across 10 domains.
//!
//! Run: cargo run -p symthaea-psych-bench --example substrate_transfer_demo

use symthaea_core::hdc::substrate_independence::SubstrateType;
use symthaea_psych_bench::substrate_transfer::{
    BenchmarkScore, CognitiveDomain, SubstrateTransferMatrix, TransferProtocol,
    generate_synthetic_baseline, simulate_transfer_degradation,
};

/// A transfer scenario: source substrate, target substrate, degradation factor.
struct TransferScenario {
    source: SubstrateType,
    target: SubstrateType,
    degradation: f64,
    /// Per-domain degradation offsets (additive on top of base degradation).
    /// Domains not listed use the base degradation unchanged.
    domain_offsets: Vec<(CognitiveDomain, f64)>,
}

fn main() {
    println!(
        "\
\x1b[1m\
╔══════════════════════════════════════════════════════╗\n\
║  Substrate-Validated Consciousness Transfer          ║\n\
╠══════════════════════════════════════════════════════╣\n\
║  Testing Multiple Realizability (Putnam 1967)        ║\n\
╚══════════════════════════════════════════════════════╝\
\x1b[0m"
    );

    let scenarios = vec![
        TransferScenario {
            source: SubstrateType::BiologicalNeurons,
            target: SubstrateType::SiliconDigital,
            degradation: 0.10,
            domain_offsets: vec![
                (CognitiveDomain::EmotionalRegulation, 0.08),
                (CognitiveDomain::Consciousness, 0.05),
            ],
        },
        TransferScenario {
            source: SubstrateType::BiologicalNeurons,
            target: SubstrateType::QuantumComputer,
            degradation: 0.30,
            domain_offsets: vec![
                (CognitiveDomain::MotorControl, 0.15),
                (CognitiveDomain::SocialCognition, 0.10),
                (CognitiveDomain::Consciousness, 0.08),
            ],
        },
        TransferScenario {
            source: SubstrateType::BiologicalNeurons,
            target: SubstrateType::NeuromorphicChip,
            degradation: 0.15,
            domain_offsets: vec![
                (CognitiveDomain::EmotionalRegulation, 0.05),
                (CognitiveDomain::LearningRate, -0.05), // neuromorphic learns faster
            ],
        },
        TransferScenario {
            source: SubstrateType::SiliconDigital,
            target: SubstrateType::PhotonicProcessor,
            degradation: 0.20,
            domain_offsets: vec![
                (CognitiveDomain::Perception, -0.08), // photonic excels at perception
                (CognitiveDomain::Consciousness, 0.10),
                (CognitiveDomain::MotorControl, 0.12),
            ],
        },
        TransferScenario {
            source: SubstrateType::SiliconDigital,
            target: SubstrateType::BiologicalNeurons,
            degradation: 0.12,
            domain_offsets: vec![
                (CognitiveDomain::EmotionalRegulation, -0.06), // bio excels emotionally
                (CognitiveDomain::Reasoning, 0.05),
            ],
        },
        TransferScenario {
            source: SubstrateType::QuantumComputer,
            target: SubstrateType::SiliconDigital,
            degradation: 0.25,
            domain_offsets: vec![
                (CognitiveDomain::Reasoning, 0.10),
                (CognitiveDomain::Consciousness, 0.12),
                (CognitiveDomain::WorkingMemory, 0.08),
            ],
        },
    ];

    let mut matrix = SubstrateTransferMatrix::new();

    for scenario in &scenarios {
        let protocol = run_transfer(scenario);
        matrix.record(protocol);
    }

    // Realizability report
    let report = matrix.realizability_report();

    println!("\n\x1b[1m═══ Realizability Report ═══\x1b[0m\n");
    println!("Substrate Rankings (by best preservation as target):");
    for (i, (substrate, score)) in report.substrate_rankings.iter().enumerate() {
        println!("  {}. {:?}:  {:.4}", i + 1, substrate, score);
    }

    println!("\nMost Substrate-Dependent Domains:");
    for (i, (domain, variance)) in report.domain_vulnerability.iter().enumerate() {
        println!("  {}. {:?}:  {:.4} variance", i + 1, domain, variance);
    }

    println!(
        "\nOverall Multiple Realizability Confidence: \x1b[1m{:.4}\x1b[0m",
        report.overall_confidence
    );
    println!("Total transfers analyzed: {}\n", report.num_transfers);
}

/// Run a single transfer scenario: generate baselines, apply degradation,
/// compute preservation, and print results.
fn run_transfer(scenario: &TransferScenario) -> TransferProtocol {
    println!(
        "\n\x1b[1m═══ Transfer: {:?} → {:?} ═══\x1b[0m",
        scenario.source, scenario.target
    );

    // Generate baseline scores across all 10 cognitive domains.
    let baseline = generate_synthetic_baseline(CognitiveDomain::ALL.len());

    // Apply base degradation, then per-domain offsets.
    let post_transfer = apply_scenario_degradation(&baseline, scenario);

    // Consciousness metrics: Psi/Phi degrade proportionally.
    let psi_before = 0.85;
    let phi_before = 0.72;
    let psi_after = psi_before * (1.0 - scenario.degradation * 0.8);
    let phi_after = phi_before * (1.0 - scenario.degradation * 0.6);

    let mut protocol = TransferProtocol::new(scenario.source, scenario.target);
    protocol.record_baseline(baseline.clone(), psi_before, phi_before);
    protocol.record_post_transfer(post_transfer.clone(), psi_after, phi_after);
    let preservation = protocol.compute_preservation();

    // Display per-domain results.
    println!(
        "{:<22}| {:>8} | {:>7} | {:>9}",
        "Domain", "Baseline", "Post", "Preserved"
    );
    println!("{}", "-".repeat(55));

    let baseline_by_domain = aggregate_simple(&baseline);
    let post_by_domain = aggregate_simple(&post_transfer);

    for &domain in CognitiveDomain::ALL {
        let base = baseline_by_domain.get(&domain).copied().unwrap_or(0.0);
        let post = post_by_domain.get(&domain).copied().unwrap_or(0.0);
        let pct = if base > 0.0 { post / base * 100.0 } else { 0.0 };
        let color = if pct >= 95.0 {
            "\x1b[32m" // green
        } else if pct >= 80.0 {
            "\x1b[33m" // yellow
        } else {
            "\x1b[31m" // red
        };
        println!(
            "{:<22}|    {:.2}  |  {}{:.2}\x1b[0m  |  {}{:>5.1}%\x1b[0m",
            format!("{:?}", domain),
            base,
            color,
            post,
            color,
            pct,
        );
    }

    println!(
        "\nPsi: {:.3} → {:.3}  |  Phi: {:.3} → {:.3}",
        psi_before, psi_after, phi_before, phi_after
    );
    println!("\x1b[1mOverall Preservation: {:.4}\x1b[0m", preservation);

    // Show most degraded domains.
    let degraded = protocol.degraded_domains(0.85);
    if !degraded.is_empty() {
        print!("Degraded domains (>15%): ");
        let names: Vec<String> = degraded.iter().map(|d| format!("{:?}", d)).collect();
        println!("{}", names.join(", "));
    }

    protocol
}

/// Apply base degradation plus per-domain offsets from the scenario.
fn apply_scenario_degradation(
    baseline: &[BenchmarkScore],
    scenario: &TransferScenario,
) -> Vec<BenchmarkScore> {
    let offsets: std::collections::HashMap<CognitiveDomain, f64> =
        scenario.domain_offsets.iter().cloned().collect();

    // Start with uniform degradation.
    let degraded = simulate_transfer_degradation(baseline, scenario.degradation);

    // Apply per-domain offsets.
    degraded
        .into_iter()
        .map(|mut s| {
            if let Some(&offset) = offsets.get(&s.domain) {
                // Offset is additional degradation (positive = worse, negative = better).
                let additional_factor = 1.0 - offset.clamp(-0.5, 0.5);
                s.score = (s.score * additional_factor).clamp(0.0, 1.0);
            }
            s
        })
        .collect()
}

/// Simple per-domain mean aggregation for display.
fn aggregate_simple(scores: &[BenchmarkScore]) -> std::collections::HashMap<CognitiveDomain, f64> {
    let mut sums: std::collections::HashMap<CognitiveDomain, (f64, usize)> =
        std::collections::HashMap::new();
    for s in scores {
        let entry = sums.entry(s.domain).or_insert((0.0, 0));
        entry.0 += s.score;
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(d, (sum, count))| (d, sum / count as f64))
        .collect()
}
