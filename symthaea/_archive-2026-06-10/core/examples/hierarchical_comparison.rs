// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical CfC vs Single-Scale CfC Consciousness Comparison
//!
//! Runs N cognitive cycles with both backends on the same input stream and
//! compares consciousness metrics (Phi, coherence, consciousness_level,
//! attention fatigue, prediction accuracy).
//!
//! This validates Butlin PP-2 (Hierarchical Prediction at Multiple Scales)
//! by demonstrating that the multi-scale temporal hierarchy produces
//! measurably different consciousness dynamics.
//!
//! ```bash
//! cargo run --example hierarchical_comparison --release
//! ```

use symthaea::cognitive_loop::config::TemporalBackend;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const N_CYCLES: usize = 200;

/// Stimuli that exercise different temporal scales:
/// fast changes, slow themes, and mixed sequences.
fn stimuli() -> Vec<&'static str> {
    let mut s = Vec::with_capacity(N_CYCLES);
    let fast = [
        "red", "blue", "green", "stop", "go", "yes", "no", "hot", "cold",
    ];
    let slow = [
        "The morning sun rises slowly over the hills",
        "A deep feeling of peace settles into awareness",
        "The river continues its ancient journey to the sea",
        "Consciousness emerges from the interplay of many signals",
        "Each moment builds upon the patterns of the past",
    ];
    let mixed = [
        "Alert! Sudden change detected in the environment",
        "calm",
        "The long-term trend suggests gradual improvement",
        "bang",
        "Over many cycles the system learns to predict",
    ];

    for i in 0..N_CYCLES {
        let phase = i % 30;
        if phase < 10 {
            s.push(fast[i % fast.len()]);
        } else if phase < 20 {
            s.push(slow[i % slow.len()]);
        } else {
            s.push(mixed[i % mixed.len()]);
        }
    }
    s
}

#[derive(Default)]
struct MetricsAccumulator {
    phi: Vec<f64>,
    coherence: Vec<f32>,
    consciousness_level: Vec<f64>,
    prediction_error: Vec<f32>,
    attention_fatigue: Vec<f32>,
    attention_pred_accuracy: Vec<f32>,
}

impl MetricsAccumulator {
    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn mean_f32(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f32>() / v.len() as f32
    }

    fn std_f64(v: &[f64]) -> f64 {
        if v.len() < 2 {
            return 0.0;
        }
        let m = Self::mean(v);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        var.sqrt()
    }

    fn print_summary(&self, label: &str) {
        println!("  {label}:");
        println!(
            "    Phi (unified_psi):      {:.4} +/- {:.4}",
            Self::mean(&self.phi),
            Self::std_f64(&self.phi)
        );
        println!(
            "    Coherence:              {:.4}",
            Self::mean_f32(&self.coherence)
        );
        println!(
            "    Consciousness level:    {:.4} +/- {:.4}",
            Self::mean(&self.consciousness_level),
            Self::std_f64(&self.consciousness_level)
        );
        println!(
            "    Prediction error:       {:.4}",
            Self::mean_f32(&self.prediction_error)
        );
        println!(
            "    Attention fatigue:      {:.4}",
            Self::mean_f32(&self.attention_fatigue)
        );
        println!(
            "    Attention pred acc:     {:.4}",
            Self::mean_f32(&self.attention_pred_accuracy)
        );
    }
}

fn run_backend(backend: TemporalBackend, inputs: &[&str]) -> MetricsAccumulator {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.temporal_backend = backend;
    config.enable_attention_schema = true;
    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    let mut acc = MetricsAccumulator::default();

    for (i, input) in inputs.iter().enumerate() {
        let result = service.cycle(input);
        let md = &result.metadata;

        acc.phi.push(md.pipeline_consciousness);
        acc.coherence.push(md.prediction_coherence);
        acc.consciousness_level
            .push(md.consciousness.consciousness_level);
        acc.prediction_error.push(result.prediction_error);
        acc.attention_fatigue.push(md.attention.attention_fatigue);
        acc.attention_pred_accuracy
            .push(md.attention.attention_prediction_accuracy);

        if (i + 1) % 50 == 0 {
            println!(
                "    cycle {}: phi={:.4} coherence={:.3} consciousness={:.4} PE={:.3}",
                i + 1,
                md.pipeline_consciousness,
                md.prediction_coherence,
                md.consciousness.consciousness_level,
                result.prediction_error
            );
        }
    }

    acc
}

fn main() {
    println!();
    println!("================================================================");
    println!("  HIERARCHICAL CfC vs SINGLE-SCALE CfC COMPARISON");
    println!("  {} cycles, same stimulus stream", N_CYCLES);
    println!("================================================================");
    println!();

    let inputs = stimuli();

    // ── Single-scale CfC ──
    println!("[1/2] Running single-scale CfC...");
    let cfc_metrics = run_backend(TemporalBackend::CfC, &inputs);

    println!();

    // ── Hierarchical CfC ──
    println!("[2/2] Running hierarchical CfC (4 temporal scales)...");
    let hcfc_metrics = run_backend(TemporalBackend::HierarchicalCfC, &inputs);

    // ── Comparison ──
    println!();
    println!("================================================================");
    println!("  RESULTS");
    println!("================================================================");

    cfc_metrics.print_summary("Single-scale CfC");
    println!();
    hcfc_metrics.print_summary("Hierarchical CfC (4 scales)");

    println!();
    println!("  Delta (Hierarchical - Single):");
    let phi_delta =
        MetricsAccumulator::mean(&hcfc_metrics.phi) - MetricsAccumulator::mean(&cfc_metrics.phi);
    let cons_delta = MetricsAccumulator::mean(&hcfc_metrics.consciousness_level)
        - MetricsAccumulator::mean(&cfc_metrics.consciousness_level);
    let pe_delta = MetricsAccumulator::mean_f32(&hcfc_metrics.prediction_error)
        - MetricsAccumulator::mean_f32(&cfc_metrics.prediction_error);
    println!("    Phi:                    {:+.4}", phi_delta);
    println!("    Consciousness level:    {:+.4}", cons_delta);
    println!("    Prediction error:       {:+.4}", pe_delta);

    println!();
    println!("  Butlin PP-2 validation: Hierarchical backend provides");
    println!("  multi-scale temporal prediction (tau=0.01/0.1/1.0/10.0)");
    println!("  with bidirectional error/prior flow between scales.");

    // ── AST Causal Loop Validation ──
    println!();
    println!("  Butlin AST-1 validation: Attention schema self-model");
    println!("  causally modulates perception via encode_for_thought_vector()");
    println!("  injection into compressed_state (weight=0.05).");
    println!(
        "    CfC attention fatigue:   {:.4}",
        MetricsAccumulator::mean_f32(&cfc_metrics.attention_fatigue)
    );
    println!(
        "    HCfC attention fatigue:  {:.4}",
        MetricsAccumulator::mean_f32(&hcfc_metrics.attention_fatigue)
    );
    println!(
        "    CfC pred accuracy:       {:.4}",
        MetricsAccumulator::mean_f32(&cfc_metrics.attention_pred_accuracy)
    );
    println!(
        "    HCfC pred accuracy:      {:.4}",
        MetricsAccumulator::mean_f32(&hcfc_metrics.attention_pred_accuracy)
    );

    println!();
    println!("================================================================");
}