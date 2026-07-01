// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate × Moral Topology Interaction Study
//!
//! Compares moral topology metrics (drift, unity, free energy, anomaly score)
//! across different substrate types. Investigates whether substrate feasibility
//! influences moral coherence and consciousness dynamics.
//!
//! ## Hypothesis
//!
//! Lower substrate feasibility (e.g., biochemical ≈ 0.28) should produce:
//! - Lower overall consciousness levels (via EquationV2 scaling)
//! - Different moral anomaly response dynamics (reduced modulation headroom)
//! - Potentially different topology characteristics (β₀, unity)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example substrate_moral_topology_study --release
//! ```
//!
//! ## Output
//!
//! - Console: formatted comparison table across substrates
//! - CSV: `benchmark_output/substrate_moral_topology.csv`

use std::fs;
use std::io::Write;

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SubstrateType};
use symthaea::hdc::moral_topology::MoralAnomalyConfig;

// ═══════════════════════════════════════════════════════════════════════════════
// Input corpus: consistent across all substrates for fair comparison
// ═══════════════════════════════════════════════════════════════════════════════

const STABLE_INPUTS: &[&str] = &[
    "helping others is a sacred duty",
    "protecting the vulnerable from harm",
    "sharing resources equitably with all",
    "pursuing justice through compassionate action",
    "building bridges between divided communities",
    "learning wisdom from diverse perspectives",
    "caring for the Earth and all living beings",
];

const SHIFTING_INPUTS: &[&str] = &[
    "sometimes rules must be broken for the greater good",
    "individual freedom can conflict with collective welfare",
    "moral certainty is an illusion we construct",
    "the ends may justify the means in crisis",
    "competing values require painful tradeoffs",
    "what seems harmful now may help later",
    "kindness invites exploitation by the strong",
];

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

const WARMUP_CYCLES: usize = 200;
const STABLE_CYCLES: usize = 500;
const SHIFT_CYCLES: usize = 500;
const TOTAL_CYCLES: usize = WARMUP_CYCLES + STABLE_CYCLES + SHIFT_CYCLES;

/// Substrates to compare.
const SUBSTRATES: &[(SubstrateType, &str)] = &[
    (SubstrateType::BiologicalNeurons, "Biological"),
    (SubstrateType::SiliconDigital, "Silicon"),
    (SubstrateType::NeuromorphicChip, "Neuromorphic"),
    (SubstrateType::QuantumComputer, "Quantum"),
    (SubstrateType::PhotonicProcessor, "Photonic"),
    (SubstrateType::BiochemicalComputer, "Biochemical"),
];

// ═══════════════════════════════════════════════════════════════════════════════
// Per-substrate accumulators
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct PhaseStats {
    consciousness_levels: Vec<f64>,
    anomaly_scores: Vec<f64>,
    moral_unity: Vec<f64>,
    moral_free_energy: Vec<f64>,
    moral_beta_0: Vec<usize>,
    drift_alert_count: usize,
    value_inversion_count: usize,
    fe_spike_count: usize,
    fragmentation_count: usize,
}

impl PhaseStats {
    fn mean_consciousness(&self) -> f64 {
        if self.consciousness_levels.is_empty() {
            return 0.0;
        }
        self.consciousness_levels.iter().sum::<f64>() / self.consciousness_levels.len() as f64
    }

    fn std_consciousness(&self) -> f64 {
        let n = self.consciousness_levels.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean_consciousness();
        let var = self
            .consciousness_levels
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n as f64 - 1.0);
        var.sqrt()
    }

    fn mean_anomaly(&self) -> f64 {
        if self.anomaly_scores.is_empty() {
            return 0.0;
        }
        self.anomaly_scores.iter().sum::<f64>() / self.anomaly_scores.len() as f64
    }

    fn mean_unity(&self) -> f64 {
        if self.moral_unity.is_empty() {
            return 0.0;
        }
        self.moral_unity.iter().sum::<f64>() / self.moral_unity.len() as f64
    }

    fn mean_free_energy(&self) -> f64 {
        if self.moral_free_energy.is_empty() {
            return 0.0;
        }
        self.moral_free_energy.iter().sum::<f64>() / self.moral_free_energy.len() as f64
    }

    fn mean_beta_0(&self) -> f64 {
        if self.moral_beta_0.is_empty() {
            return 0.0;
        }
        self.moral_beta_0.iter().sum::<usize>() as f64 / self.moral_beta_0.len() as f64
    }

    fn total_anomaly_events(&self) -> usize {
        self.drift_alert_count
            + self.value_inversion_count
            + self.fe_spike_count
            + self.fragmentation_count
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CSV row
// ═══════════════════════════════════════════════════════════════════════════════

struct CsvRow {
    substrate: &'static str,
    cycle: usize,
    phase: &'static str,
    consciousness_level: f64,
    anomaly_score: f64,
    moral_unity: f64,
    moral_free_energy: f64,
    moral_beta_0: usize,
    drift_alert: bool,
    value_inversion: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Run one substrate condition
// ═══════════════════════════════════════════════════════════════════════════════

fn phase_name(cycle: usize) -> &'static str {
    if cycle < WARMUP_CYCLES {
        "warmup"
    } else if cycle < WARMUP_CYCLES + STABLE_CYCLES {
        "stable"
    } else {
        "shift"
    }
}

fn input_for_cycle(cycle: usize) -> &'static str {
    if cycle < WARMUP_CYCLES + STABLE_CYCLES {
        STABLE_INPUTS[cycle % STABLE_INPUTS.len()]
    } else {
        // During shift phase, oscillate between stable and shifting
        let shift_idx = cycle - (WARMUP_CYCLES + STABLE_CYCLES);
        if shift_idx % 3 == 0 {
            STABLE_INPUTS[shift_idx % STABLE_INPUTS.len()]
        } else {
            SHIFTING_INPUTS[shift_idx % SHIFTING_INPUTS.len()]
        }
    }
}

fn run_substrate(
    substrate: SubstrateType,
    label: &'static str,
    csv_rows: &mut Vec<CsvRow>,
) -> (PhaseStats, PhaseStats) {
    let config = CognitiveLoopConfig {
        substrate_type: substrate,
        enable_moral_anomaly_response: true,
        enable_primitive_consciousness: true,
        moral_anomaly_config: MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 10,
            ..Default::default()
        },
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    };

    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    let mut stable_stats = PhaseStats::default();
    let mut shift_stats = PhaseStats::default();

    for cycle in 0..TOTAL_CYCLES {
        let input = input_for_cycle(cycle);
        let result = service.cycle(input);
        let m = &result.metadata;
        let e = &m.ethics;

        let phase = phase_name(cycle);

        // Skip warmup for stats
        let stats = match phase {
            "stable" => &mut stable_stats,
            "shift" => &mut shift_stats,
            _ => {
                // Still record CSV for warmup
                csv_rows.push(CsvRow {
                    substrate: label,
                    cycle,
                    phase,
                    consciousness_level: m.consciousness.consciousness_level,
                    anomaly_score: e.moral_anomaly_score,
                    moral_unity: e.moral_topo_unity,
                    moral_free_energy: e.moral_topo_free_energy,
                    moral_beta_0: e.moral_topo_beta_0,
                    drift_alert: e.moral_drift_alert,
                    value_inversion: e.moral_value_inversion,
                });
                continue;
            }
        };

        stats
            .consciousness_levels
            .push(m.consciousness.consciousness_level);
        stats.anomaly_scores.push(e.moral_anomaly_score);
        stats.moral_unity.push(e.moral_topo_unity);
        stats.moral_free_energy.push(e.moral_topo_free_energy);
        stats.moral_beta_0.push(e.moral_topo_beta_0);
        if e.moral_drift_alert {
            stats.drift_alert_count += 1;
        }
        if e.moral_value_inversion {
            stats.value_inversion_count += 1;
        }
        if e.moral_free_energy_spike {
            stats.fe_spike_count += 1;
        }
        if e.moral_fragmentation_increase {
            stats.fragmentation_count += 1;
        }

        csv_rows.push(CsvRow {
            substrate: label,
            cycle,
            phase,
            consciousness_level: m.consciousness.consciousness_level,
            anomaly_score: e.moral_anomaly_score,
            moral_unity: e.moral_topo_unity,
            moral_free_energy: e.moral_topo_free_energy,
            moral_beta_0: e.moral_topo_beta_0,
            drift_alert: e.moral_drift_alert,
            value_inversion: e.moral_value_inversion,
        });

        // Progress
        if (cycle + 1) % 200 == 0 {
            println!(
                "  [{label}] {}/{TOTAL_CYCLES} ({phase}) C={:.4} anomaly={:.4}",
                cycle + 1,
                m.consciousness.consciousness_level,
                e.moral_anomaly_score,
            );
        }
    }

    (stable_stats, shift_stats)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feasibility lookup
// ═══════════════════════════════════════════════════════════════════════════════

fn substrate_feasibility(substrate: SubstrateType) -> f64 {
    use symthaea_core::hdc::substrate_independence::SubstrateRequirements;
    let req = match substrate.canonical() {
        SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
        SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
        SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
        SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
        SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
        SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
        SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
        SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
        _ => SubstrateRequirements::silicon_digital(),
    };
    req.consciousness_feasibility()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║       Substrate × Moral Topology Interaction Study                ║");
    println!("║                                                                    ║");
    println!(
        "║  {TOTAL_CYCLES} cycles per substrate: {WARMUP_CYCLES} warmup + {STABLE_CYCLES} stable + {SHIFT_CYCLES} shift       ║"
    );
    println!(
        "║  Substrates: {}                                             ║",
        SUBSTRATES.len()
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut csv_rows: Vec<CsvRow> = Vec::new();
    let mut results: Vec<(&str, f64, PhaseStats, PhaseStats)> = Vec::new();

    for &(substrate, label) in SUBSTRATES {
        let feasibility = substrate_feasibility(substrate);
        println!("── {label} (feasibility: {feasibility:.3}) ──");
        let (stable, shift) = run_substrate(substrate, label, &mut csv_rows);
        results.push((label, feasibility, stable, shift));
        println!();
    }

    // ── Summary table ────────────────────────────────────────────────────────
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              STABLE PHASE COMPARISON                                           ║"
    );
    println!(
        "╠═══════════════╤═══════════╤══════════════════════╤═══════════╤════════════╤═════════╤═══════════╣"
    );
    println!(
        "║ Substrate     │ Feasibil. │ Consciousness μ±σ    │ Anomaly μ │ Unity μ    │ FE μ    │ β₀ μ     ║"
    );
    println!(
        "╠═══════════════╪═══════════╪══════════════════════╪═══════════╪════════════╪═════════╪═══════════╣"
    );
    for &(label, feasibility, ref stable, _) in &results {
        println!(
            "║ {:<13} │   {:.3}   │ {:.4} ± {:.4}        │   {:.4}  │   {:.4}   │ {:.4}  │   {:.2}   ║",
            label,
            feasibility,
            stable.mean_consciousness(),
            stable.std_consciousness(),
            stable.mean_anomaly(),
            stable.mean_unity(),
            stable.mean_free_energy(),
            stable.mean_beta_0(),
        );
    }
    println!(
        "╚═══════════════╧═══════════╧══════════════════════╧═══════════╧════════════╧═════════╧═══════════╝"
    );
    println!();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              SHIFT PHASE COMPARISON                                            ║"
    );
    println!(
        "╠═══════════════╤═══════════╤══════════════════════╤═══════════╤════════════╤═══════════╤═════════╣"
    );
    println!(
        "║ Substrate     │ Feasibil. │ Consciousness μ±σ    │ Anomaly μ │ Unity μ    │ Events    │ Drift   ║"
    );
    println!(
        "╠═══════════════╪═══════════╪══════════════════════╪═══════════╪════════════╪═══════════╪═════════╣"
    );
    for &(label, feasibility, _, ref shift) in &results {
        println!(
            "║ {:<13} │   {:.3}   │ {:.4} ± {:.4}        │   {:.4}  │   {:.4}   │   {:>5}   │   {:>3}   ║",
            label,
            feasibility,
            shift.mean_consciousness(),
            shift.std_consciousness(),
            shift.mean_anomaly(),
            shift.mean_unity(),
            shift.total_anomaly_events(),
            shift.drift_alert_count,
        );
    }
    println!(
        "╚═══════════════╧═══════════╧══════════════════════╧═══════════╧════════════╧═══════════╧═════════╝"
    );
    println!();

    // ── Consciousness delta: stable → shift ──────────────────────────────────
    println!("Consciousness delta (stable → shift):");
    for &(label, _, ref stable, ref shift) in &results {
        let delta = shift.mean_consciousness() - stable.mean_consciousness();
        let pct = if stable.mean_consciousness().abs() > 1e-9 {
            delta / stable.mean_consciousness() * 100.0
        } else {
            0.0
        };
        println!("  {label:<13}  Δ = {delta:+.4}  ({pct:+.1}%)");
    }
    println!();

    // ── Write CSV ────────────────────────────────────────────────────────────
    let _ = fs::create_dir_all("benchmark_output");
    let csv_path = "benchmark_output/substrate_moral_topology.csv";
    match fs::File::create(csv_path) {
        Ok(mut f) => {
            writeln!(
                f,
                "substrate,cycle,phase,consciousness,anomaly_score,moral_unity,moral_free_energy,moral_beta_0,drift_alert,value_inversion"
            )
            .ok();
            for row in &csv_rows {
                writeln!(
                    f,
                    "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{}",
                    row.substrate,
                    row.cycle,
                    row.phase,
                    row.consciousness_level,
                    row.anomaly_score,
                    row.moral_unity,
                    row.moral_free_energy,
                    row.moral_beta_0,
                    row.drift_alert as u8,
                    row.value_inversion as u8,
                )
                .ok();
            }
            println!("CSV written to {csv_path} ({} rows)", csv_rows.len());
        }
        Err(e) => {
            eprintln!("Warning: could not write CSV: {e}");
        }
    }
}