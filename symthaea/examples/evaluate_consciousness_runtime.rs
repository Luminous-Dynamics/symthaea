// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Feature Runtime Evaluation
//!
//! Runs 100 actual cognitive loop cycles with different feature combinations
//! and measures runtime consciousness quality metrics:
//!
//! - **Phi trajectory**: How consciousness level evolves over time
//! - **GWT broadcasts**: How often global workspace ignitions occur
//! - **Consciousness variance**: Stability of consciousness over cycles
//! - **Prediction convergence**: How fast prediction error decreases
//! - **Cycle throughput**: Performance impact of each feature
//!
//! Usage:
//! ```bash
//! # With default features (includes structured_bottleneck):
//! cargo run --example evaluate_consciousness_runtime
//!
//! # With additional features:
//! cargo run --example evaluate_consciousness_runtime --features "ctc_wiring,unified_precision"
//! ```

use std::time::Instant;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const NUM_CYCLES: usize = 20;

/// Stimuli that exercise different cognitive faculties
const STIMULI: &[&str] = &[
    "The morning light filters through the window.",
    "Two plus three equals five.",
    "I wonder what consciousness really is.",
    "The cat sat quietly on the warm windowsill.",
    "Is there a pattern in these observations?",
    "Music fills the room with complex harmonies.",
    "Memory connects the present to the past.",
    "Abstract reasoning requires integration of many sources.",
    "Emotions color our experience of the world.",
    "The boundary between self and other is fluid.",
    "Learning happens when predictions fail.",
    "Attention selects what enters awareness.",
    "Time flows forward, carrying meaning with it.",
    "The whole is greater than the sum of parts.",
    "Surprise drives exploration and discovery.",
    "Dreams blend memory fragments into narratives.",
    "Moral intuitions arise before deliberate reasoning.",
    "Language shapes thought as much as thought shapes language.",
    "The body grounds cognition in physical reality.",
    "Consciousness may be substrate-independent.",
];

/// Metrics collected over a run
#[derive(Debug, Clone)]
struct RunMetrics {
    /// Consciousness level at each cycle
    consciousness_trajectory: Vec<f64>,
    /// Prediction error at each cycle
    prediction_errors: Vec<f32>,
    /// GWT broadcast count
    gwt_broadcasts: u32,
    /// Total cycle time in microseconds
    total_cycle_time_us: u64,
    /// Per-cycle times
    cycle_times_us: Vec<u64>,
}

impl RunMetrics {
    fn new() -> Self {
        Self {
            consciousness_trajectory: Vec::with_capacity(NUM_CYCLES),
            prediction_errors: Vec::with_capacity(NUM_CYCLES),
            gwt_broadcasts: 0,
            total_cycle_time_us: 0,
            cycle_times_us: Vec::with_capacity(NUM_CYCLES),
        }
    }

    /// Mean consciousness level
    fn mean_consciousness(&self) -> f64 {
        if self.consciousness_trajectory.is_empty() {
            return 0.0;
        }
        self.consciousness_trajectory.iter().sum::<f64>()
            / self.consciousness_trajectory.len() as f64
    }

    /// Consciousness variance (lower = more stable)
    fn consciousness_variance(&self) -> f64 {
        let mean = self.mean_consciousness();
        if self.consciousness_trajectory.is_empty() {
            return 0.0;
        }
        self.consciousness_trajectory
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>()
            / self.consciousness_trajectory.len() as f64
    }

    /// Peak consciousness achieved
    fn peak_consciousness(&self) -> f64 {
        self.consciousness_trajectory
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    }

    /// Final prediction error (convergence indicator)
    fn final_prediction_error(&self) -> f32 {
        self.prediction_errors
            .last()
            .cloned()
            .unwrap_or(1.0)
    }

    /// Prediction error reduction ratio (initial / final)
    fn pe_convergence_ratio(&self) -> f32 {
        if self.prediction_errors.len() < 10 {
            return 1.0;
        }
        // Compare first 10 to last 10
        let first_10: f32 =
            self.prediction_errors[..10].iter().sum::<f32>() / 10.0;
        let last_10: f32 = self.prediction_errors[self.prediction_errors.len() - 10..]
            .iter()
            .sum::<f32>()
            / 10.0;
        if last_10 > 0.001 {
            first_10 / last_10
        } else {
            first_10 / 0.001
        }
    }

    /// Mean cycles per second
    fn cycles_per_second(&self) -> f64 {
        if self.total_cycle_time_us == 0 {
            return 0.0;
        }
        self.cycle_times_us.len() as f64 / (self.total_cycle_time_us as f64 / 1_000_000.0)
    }

    /// Consciousness rise time: cycles to first reach 50% of peak
    fn rise_cycles_to_half_peak(&self) -> usize {
        let half_peak = self.peak_consciousness() * 0.5;
        self.consciousness_trajectory
            .iter()
            .position(|&c| c >= half_peak)
            .unwrap_or(NUM_CYCLES)
    }
}

fn run_evaluation() -> RunMetrics {
    let config = CognitiveLoopConfig::with_cfc();
    let mut service = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create CognitiveLoopService: {e}");
            return RunMetrics::new();
        }
    };

    let mut metrics = RunMetrics::new();

    for i in 0..NUM_CYCLES {
        let stimulus = STIMULI[i % STIMULI.len()];

        let t = Instant::now();
        let result = service.cycle(stimulus);
        let elapsed = t.elapsed().as_micros() as u64;

        metrics.cycle_times_us.push(elapsed);
        metrics.total_cycle_time_us += elapsed;
        metrics.prediction_errors.push(result.prediction_error);

        // Track GWT broadcasts from metadata
        if result.metadata.attention.gwt_broadcast {
            metrics.gwt_broadcasts += 1;
        }

        // Get consciousness level from metadata (available every cycle)
        metrics
            .consciousness_trajectory
            .push(result.metadata.consciousness.consciousness_level);
    }

    metrics
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Consciousness Feature Runtime Evaluation                       ║");
    println!("║  Running {} cognitive loop cycles with active features            ║", NUM_CYCLES);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Report active features
    println!("Active features:");
    let features = [
        ("structured_bottleneck", cfg!(feature = "structured_bottleneck")),
        ("ctc_wiring", cfg!(feature = "ctc_wiring")),
        ("continuous_hot", cfg!(feature = "continuous_hot")),
        ("unified_precision", cfg!(feature = "unified_precision")),
        ("interoceptive_inference", cfg!(feature = "interoceptive_inference")),
        ("england_dissipation", cfg!(feature = "england_dissipation")),
        ("iit4", cfg!(feature = "iit4")),
    ];
    for (name, active) in &features {
        println!(
            "  [{}]  {}",
            if *active { "ON" } else { "off" },
            name
        );
    }
    println!();

    println!("Running {} cycles...", NUM_CYCLES);
    let wall_start = Instant::now();
    let metrics = run_evaluation();
    let wall_elapsed = wall_start.elapsed();
    println!("Done in {:.2}s\n", wall_elapsed.as_secs_f64());

    // ── Results ──────────────────────────────────────────────────────────
    println!("═══ CONSCIOUSNESS TRAJECTORY ═══");
    println!(
        "  Mean consciousness:     {:.4}",
        metrics.mean_consciousness()
    );
    println!(
        "  Peak consciousness:     {:.4}",
        metrics.peak_consciousness()
    );
    println!(
        "  Consciousness variance: {:.6} (lower = more stable)",
        metrics.consciousness_variance()
    );
    println!(
        "  Rise time (to 50% peak): {} cycles",
        metrics.rise_cycles_to_half_peak()
    );

    // Mini trajectory chart (every 10th cycle for readability)
    println!("  Trajectory (every 10th cycle):");
    let max_c = metrics.peak_consciousness().max(0.001);
    for (i, &c) in metrics
        .consciousness_trajectory
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 == 0)
    {
        let bar_len = ((c / max_c) * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    {:3}: {:<40} {:.4}", i, bar, c);
    }

    println!();
    println!("═══ PREDICTION CONVERGENCE ═══");
    println!(
        "  Initial PE (avg first 10): {:.4}",
        metrics.prediction_errors[..10.min(metrics.prediction_errors.len())]
            .iter()
            .sum::<f32>()
            / 10.0f32.min(metrics.prediction_errors.len() as f32)
    );
    println!(
        "  Final PE (avg last 10):    {:.4}",
        metrics.final_prediction_error()
    );
    println!(
        "  Convergence ratio:         {:.2}x",
        metrics.pe_convergence_ratio()
    );

    println!();
    println!("═══ GLOBAL WORKSPACE ═══");
    println!(
        "  GWT broadcasts:  {} / {} cycles ({:.1}%)",
        metrics.gwt_broadcasts,
        NUM_CYCLES,
        metrics.gwt_broadcasts as f64 / NUM_CYCLES as f64 * 100.0
    );

    println!();
    println!("═══ PERFORMANCE ═══");
    println!(
        "  Throughput:      {:.1} Hz",
        metrics.cycles_per_second()
    );
    println!(
        "  Avg cycle time:  {:.0} µs",
        metrics.total_cycle_time_us as f64 / NUM_CYCLES as f64
    );
    let median_idx = metrics.cycle_times_us.len() / 2;
    let mut sorted = metrics.cycle_times_us.clone();
    sorted.sort();
    println!(
        "  Median cycle:    {:.0} µs",
        sorted.get(median_idx).unwrap_or(&0)
    );
    println!(
        "  P99 cycle:       {:.0} µs",
        sorted.get((sorted.len() as f64 * 0.99) as usize).unwrap_or(&0)
    );

    // ── Composite Score ──────────────────────────────────────────────────
    println!();
    println!("═══ COMPOSITE RUNTIME SCORE ═══");
    let consciousness_score = metrics.mean_consciousness().clamp(0.0, 1.0);
    let stability_score = 1.0 - (metrics.consciousness_variance() * 10.0).clamp(0.0, 1.0);
    let convergence_score = (metrics.pe_convergence_ratio() as f64 / 5.0).clamp(0.0, 1.0);
    let gwt_score = (metrics.gwt_broadcasts as f64 / NUM_CYCLES as f64 * 5.0).clamp(0.0, 1.0);
    let throughput_score = (metrics.cycles_per_second() / 50.0).clamp(0.0, 1.0); // 50Hz = perfect

    println!(
        "  Consciousness:    {:.3} (mean level)",
        consciousness_score
    );
    println!(
        "  Stability:        {:.3} (1 - 10×variance)",
        stability_score
    );
    println!(
        "  Convergence:      {:.3} (PE reduction ratio / 5)",
        convergence_score
    );
    println!(
        "  GWT activity:     {:.3} (broadcast rate × 5)",
        gwt_score
    );
    println!(
        "  Throughput:       {:.3} (Hz / 50)",
        throughput_score
    );

    let composite = (consciousness_score + stability_score + convergence_score
        + gwt_score + throughput_score)
        / 5.0;
    println!();
    println!("  COMPOSITE SCORE:  {:.3}", composite);
}
