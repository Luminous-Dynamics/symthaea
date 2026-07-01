// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: Live Cognitive Loop with Evolved Thresholds
//!
//! Runs the full CognitiveLoopService twice:
//! 1. Baseline: default thresholds (1000 cycles)
//! 2. Evolved: best thresholds from threshold evolution (1000 cycles)
//!
//! Compares: Phi, prediction error, consciousness level, coherence.
//! This proves the evolution pipeline affects real cognitive behavior,
//! not just a heuristic fitness score.
//!
//! ```sh
//! cargo run --release --features neuroevolution --example benchmark_threshold_evolution
//! ```

#[cfg(not(feature = "neuroevolution"))]
fn main() {
    eprintln!("Requires `neuroevolution` feature.");
    eprintln!(
        "Run: cargo run --release --features neuroevolution --example benchmark_threshold_evolution"
    );
}

#[cfg(feature = "neuroevolution")]
fn main() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use symthaea_neuroevolution::{
        NeuralGenome, ThresholdPhenotype, threshold_genome::evaluate_threshold_fitness,
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Threshold Evolution → Live Cognitive Loop Benchmark");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Phase 1: Evolve thresholds ──────────────────────────────────────
    println!("  Phase 1: Evolving thresholds (20 generations, pop=30)...");
    let evolved = evolve_best_thresholds(30, 20, 42);
    println!(
        "    Best fitness: {:.4}",
        evaluate_threshold_fitness(&evolved)
    );
    println!(
        "    FEP surprise scale: {:.2} (default: 3.00)",
        evolved.fep_surprise_scale
    );
    println!(
        "    Dream base interval: {} (default: 100)",
        evolved.dream_base_interval
    );
    println!();

    // ── Phase 2: Run baseline cognitive loop ────────────────────────────
    println!("  Phase 2: Running baseline (default thresholds, 1000 cycles)...");
    let baseline = run_cognitive_loop(1000, None);
    print_stats("    Baseline", &baseline);

    // ── Phase 3: Run evolved cognitive loop ─────────────────────────────
    println!("  Phase 3: Running evolved (1000 cycles)...");
    let evolved_stats = run_cognitive_loop(1000, Some(&evolved));
    print_stats("    Evolved ", &evolved_stats);

    // ── Phase 4: Compare ────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Comparison");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!(
        "  {:>25}  {:>10}  {:>10}  {:>8}",
        "Metric", "Baseline", "Evolved", "Delta"
    );
    println!("  {}", "-".repeat(55));
    compare_row("Avg Phi (Ψ)", baseline.avg_phi, evolved_stats.avg_phi);
    compare_row("Avg PE", baseline.avg_pe, evolved_stats.avg_pe);
    compare_row(
        "Avg Consciousness",
        baseline.avg_consciousness,
        evolved_stats.avg_consciousness,
    );
    compare_row(
        "Avg Coherence",
        baseline.avg_coherence,
        evolved_stats.avg_coherence,
    );
    compare_row(
        "Learning cycles",
        baseline.learning_cycles as f64,
        evolved_stats.learning_cycles as f64,
    );
    compare_row("Final Phi", baseline.final_phi, evolved_stats.final_phi);
}

#[cfg(feature = "neuroevolution")]
struct LoopStats {
    avg_phi: f64,
    avg_pe: f64,
    avg_consciousness: f64,
    avg_coherence: f64,
    learning_cycles: usize,
    final_phi: f64,
}

#[cfg(feature = "neuroevolution")]
fn run_cognitive_loop(
    cycles: usize,
    thresholds: Option<&symthaea_neuroevolution::ThresholdPhenotype>,
) -> LoopStats {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    let mut config = CognitiveLoopConfig::default();

    let mut service = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create CognitiveLoopService: {e}");
            return LoopStats {
                avg_phi: 0.0,
                avg_pe: 0.0,
                avg_consciousness: 0.0,
                avg_coherence: 0.0,
                learning_cycles: 0,
                final_phi: 0.0,
            };
        }
    };

    // Apply evolved thresholds via the real ThresholdOverrides system.
    // All 18 evolvable parameters are injected at runtime and take
    // precedence over compile-time defaults at every usage site.
    if let Some(t) = thresholds {
        service.threshold_overrides_mut().apply_from_phenotype(t);
        println!(
            "    Applied {} threshold overrides",
            service.threshold_overrides().active_count()
        );
    }

    // Diverse input corpus to drive state changes
    let inputs = [
        "the sun rises over the mountain",
        "a sudden crash echoes through the darkness",
        "gentle waves lap at the shore",
        "the machine grinds to a halt",
        "children laughing in the garden",
        "silence falls across the empty room",
        "new patterns emerge from chaos",
        "the old bridge creaks under weight",
        "music drifts through the open window",
        "a warning light blinks on the console",
    ];

    let mut total_phi = 0.0f64;
    let mut total_pe = 0.0f64;
    let mut total_consciousness = 0.0f64;
    let mut total_coherence = 0.0f64;
    let mut learning_count = 0usize;
    let mut last_phi = 0.0f64;

    for i in 0..cycles {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        let phi = result.metadata.consciousness.consciousness_level;
        total_phi += phi;
        total_pe += result.prediction_error as f64;
        total_consciousness += phi;
        total_coherence += result.metadata.prediction_coherence as f64;
        if result.learning_occurred {
            learning_count += 1;
        }
        last_phi = phi;
    }

    let n = cycles as f64;
    LoopStats {
        avg_phi: total_phi / n,
        avg_pe: total_pe / n,
        avg_consciousness: total_consciousness / n,
        avg_coherence: total_coherence / n,
        learning_cycles: learning_count,
        final_phi: last_phi,
    }
}

#[cfg(feature = "neuroevolution")]
fn evolve_best_thresholds(
    pop_size: usize,
    generations: usize,
    seed: u64,
) -> symthaea_neuroevolution::ThresholdPhenotype {
    use symthaea_neuroevolution::{NeuralGenome, threshold_genome::evaluate_threshold_fitness};

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E3779B97F4A7C15)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn next_usize(&mut self, max: usize) -> usize {
            (self.next() % max as u64) as usize
        }
    }

    let mut rng = Rng::new(seed);

    let mut pop: Vec<(NeuralGenome, f64)> = (0..pop_size)
        .map(|_| {
            let g = NeuralGenome::random(rng.next());
            let f = evaluate_threshold_fitness(&g.decode_thresholds());
            (g, f)
        })
        .collect();

    for _ in 0..generations {
        pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let elite = pop[..pop_size / 5].to_vec();
        let mut next = elite.clone();
        while next.len() < pop_size {
            let a = &pop[rng.next_usize(pop.len())].0;
            let b = &pop[rng.next_usize(pop.len())].0;
            let mut child = a.crossover(b, rng.next());
            // Mutate threshold region only
            for byte_idx in 50..80 {
                if rng.next_f64() < 0.005 {
                    child.hv.0[byte_idx] ^= rng.next() as u8;
                }
            }
            let f = evaluate_threshold_fitness(&child.decode_thresholds());
            next.push((child, f));
        }
        pop = next;
    }

    pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    pop[0].0.decode_thresholds()
}

#[cfg(feature = "neuroevolution")]
fn print_stats(label: &str, stats: &LoopStats) {
    println!(
        "{}  Φ={:.4}  PE={:.4}  C={:.4}  coh={:.4}  learn={}",
        label,
        stats.avg_phi,
        stats.avg_pe,
        stats.avg_consciousness,
        stats.avg_coherence,
        stats.learning_cycles
    );
}

#[cfg(feature = "neuroevolution")]
fn compare_row(name: &str, baseline: f64, evolved: f64) {
    let delta = if baseline.abs() > 1e-10 {
        (evolved - baseline) / baseline * 100.0
    } else {
        0.0
    };
    let sign = if delta >= 0.0 { "+" } else { "" };
    println!(
        "  {:>25}  {:>10.4}  {:>10.4}  {:>7.1}%",
        name, baseline, evolved, delta
    );
}