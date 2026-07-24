// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observer-ΔΨ A/B experiment: does Symthaea's consciousness respond to the
//! *composition* of her artwork, or just to any bright frame?
//!
//! Runs a live cognitive loop in A/B mode (visual-art plan "Option 2"
//! follow-up): every other artwork-viewing window shows a pixel-scrambled
//! version of the render — identical color/luminance histogram, zero
//! composition. Collects Δψ verdicts per arm and reports the comparison.
//!
//! Run:
//! ```bash
//! cargo run --example art_observer_ab --no-default-features \
//!   --features art-observer,reasoning_engine --release
//! ```
//!
//! Honesty notes: this is a research probe, not a benchmark. ψ is a live,
//! confounded signal; N is small; the report prints raw per-verdict values
//! and means with spread, and deliberately computes no p-value (with N this
//! small it would be theater). A consistent sign difference across arms over
//! repeated runs is the meaningful observation.

fn main() {
    #[cfg(feature = "art-observer")]
    experiment::run();
    #[cfg(not(feature = "art-observer"))]
    eprintln!("build with --features art-observer");
}

#[cfg(feature = "art-observer")]
mod experiment {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    const INPUTS: [&str; 5] = [
        "consciousness emerges from integration",
        "the garden grows in silence",
        "hello world, what do you see",
        "a memory of light and water",
        "we are learning to trust each other",
    ];

    /// Target verdicts per arm; the loop alternates arms, so 2× this many
    /// windows total.
    const VERDICTS_PER_ARM: usize = 6;
    /// Hard cycle budget so the probe always terminates.
    const MAX_CYCLES: usize = 6_000;

    pub fn run() {
        let mut config = CognitiveLoopConfig::default();
        config.art_observer_ab_mode = true;
        // Ultra auto-dilation OOMs camera-less loops; see config docs.
        config.enable_vision_auto_dilation = false;
        let mut service = CognitiveLoopService::new(config).expect("cognitive loop construction");

        let mut art: Vec<f32> = Vec::new();
        let mut control: Vec<f32> = Vec::new();
        let mut seen_verdicts = 0u64;

        println!("observer-ΔΨ A/B: alternating real artwork vs pixel-scrambled control");
        println!("collecting up to {VERDICTS_PER_ARM} verdicts per arm (≤{MAX_CYCLES} cycles)\n");

        for i in 0..MAX_CYCLES {
            let _ = service.cycle(INPUTS[i % INPUTS.len()]);
            let telemetry = service
                .creative_telemetry()
                .expect("creative manager present");

            if telemetry.observer_verdicts > seen_verdicts {
                seen_verdicts = telemetry.observer_verdicts;
                let arm = if telemetry.observer_was_control {
                    control.push(telemetry.observer_delta_psi);
                    "control"
                } else {
                    art.push(telemetry.observer_delta_psi);
                    "art    "
                };
                println!(
                    "  verdict {:2} [{}] Δψ = {:+.5}  (viewing surprise {:.4}, cycle {})",
                    seen_verdicts,
                    arm,
                    telemetry.observer_delta_psi,
                    telemetry.observer_viewing_surprise,
                    i
                );
            }

            if art.len() >= VERDICTS_PER_ARM && control.len() >= VERDICTS_PER_ARM {
                break;
            }
        }

        println!();
        report("art     ", &art);
        report("control ", &control);

        if art.is_empty() || control.is_empty() {
            println!("\nInsufficient verdicts in one arm — no comparison possible.");
            return;
        }
        let mean_art = mean(&art);
        let mean_control = mean(&control);
        println!("\nΔψ(art) − Δψ(control) = {:+.5}", mean_art - mean_control);
        println!(
            "N = {}/{} — treat as one observation; repeat runs before believing a direction.",
            art.len(),
            control.len()
        );
    }

    fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    fn report(label: &str, values: &[f32]) {
        if values.is_empty() {
            println!("{label}: no verdicts collected");
            return;
        }
        let m = mean(values);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let positive = values.iter().filter(|v| **v > 0.0).count();
        println!(
            "{label}: N={:2}  mean Δψ {:+.5}  range [{:+.5}, {:+.5}]  positive {}/{}",
            values.len(),
            m,
            min,
            max,
            positive,
            values.len()
        );
    }
}
