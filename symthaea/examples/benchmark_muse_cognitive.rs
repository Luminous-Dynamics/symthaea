// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: Muse audio from a live CognitiveLoopService.
//!
//! Runs the full cognitive loop for 300 cycles (~10s at 30Hz),
//! captures muse audio + consciousness state each cycle, then
//! computes correlations between internal state and audio features.
//!
//! This is the *real* benchmark — audio driven by live consciousness
//! dynamics, not static scenarios.
//!
//! ```sh
//! cargo run --release --features muse --example benchmark_muse_cognitive
//! ```

#[cfg(not(feature = "muse"))]
fn main() {
    eprintln!("This example requires the `muse` feature.");
    eprintln!("Run: cargo run --release --features muse --example benchmark_muse_cognitive");
}

#[cfg(feature = "muse")]
fn main() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea: Cognitive Loop → Muse Audio Benchmark           ║");
    println!("║  Live consciousness dynamics → measured audio features      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create cognitive loop with default config
    let config = CognitiveLoopConfig::default();
    let mut service = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create CognitiveLoopService: {e}");
            return;
        }
    };

    println!("  CognitiveLoopService created (CfC backend, muse enabled)");
    println!("  Running 300 cycles...\n");

    // Diverse inputs to drive state changes
    let inputs = [
        "the sun rises over the mountain, warming the valley below",
        "a sudden crash echoes through the darkness",
        "gentle waves lap at the shore as the moon rises",
        "the machine grinds to a halt, alarms blaring",
        "children laugh and play in the garden",
        "silence fills the empty room, dust motes floating",
        "the equation reveals a beautiful symmetry hidden in chaos",
        "pain radiates through the body, each breath a struggle",
        "we gather around the fire, sharing stories of the ancestors",
        "the experiment failed again, months of work destroyed",
    ];

    let mut log: Vec<CycleRecord> = Vec::with_capacity(300);

    for cycle in 0..300 {
        let input = inputs[cycle % inputs.len()];
        let _result = service.cycle(input);

        // Extract consciousness state
        let snap = service.consciousness_snapshot();
        let ms = service.muse_musical_state().clone();

        // Extract audio features from last muse chunk
        let chunk = service.muse_audio_chunk();
        let rms = if !chunk.is_empty() {
            let mono: Vec<f32> = chunk.iter().map(|p| (p[0] + p[1]) * 0.5).collect();
            (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt()
        } else {
            0.0
        };

        let zcr = if chunk.len() > 1 {
            let mono: Vec<f32> = chunk.iter().map(|p| (p[0] + p[1]) * 0.5).collect();
            mono.windows(2)
                .filter(|w| w[0].signum() != w[1].signum())
                .count() as f32
                / mono.len() as f32
        } else {
            0.0
        };

        log.push(CycleRecord {
            cycle: cycle as u32,
            psi: snap.consciousness_level as f32,
            arousal: snap.emotional_arousal,
            valence: snap.emotional_valence,
            prediction_error: snap.prediction_confidence, // proxy
            dopamine: ms.dopamine,
            serotonin: ms.serotonin,
            noradrenaline: ms.noradrenaline,
            rms,
            zcr,
        });

        if cycle % 50 == 0 {
            println!(
                "  cycle={:3}  Ψ={:.3}  V={:+.3}  A={:.3}  RMS={:.4}  ZCR={:.3}",
                cycle,
                snap.consciousness_level,
                snap.emotional_valence,
                snap.emotional_arousal,
                rms,
                zcr
            );
        }
    }

    // ── Correlations ─────────────────────────────────────────────────────
    println!(
        "\n═══ Live Cognitive Loop Correlations (n={}) ═══\n",
        log.len()
    );

    let psi: Vec<f32> = log.iter().map(|r| r.psi).collect();
    let arousal: Vec<f32> = log.iter().map(|r| r.arousal).collect();
    let valence: Vec<f32> = log.iter().map(|r| r.valence).collect();
    let ne: Vec<f32> = log.iter().map(|r| r.noradrenaline).collect();
    let rms_vec: Vec<f32> = log.iter().map(|r| r.rms).collect();
    let zcr_vec: Vec<f32> = log.iter().map(|r| r.zcr).collect();

    print_corr("Arousal ↔ RMS Energy", &arousal, &rms_vec);
    print_corr("Ψ (consciousness) ↔ RMS", &psi, &rms_vec);
    print_corr("Valence ↔ RMS Energy", &valence, &rms_vec);
    print_corr("NE ↔ Zero-Crossing Rate", &ne, &zcr_vec);
    print_corr("Valence ↔ ZCR", &valence, &zcr_vec);

    // Save CSV
    let csv_path = "data/muse_cognitive_loop.csv";
    if let Ok(mut f) = std::fs::File::create(csv_path) {
        use std::io::Write;
        let _ = writeln!(f, "cycle,psi,arousal,valence,pe,da,5ht,ne,rms,zcr");
        for r in &log {
            let _ = writeln!(
                f,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4}",
                r.cycle,
                r.psi,
                r.arousal,
                r.valence,
                r.prediction_error,
                r.dopamine,
                r.serotonin,
                r.noradrenaline,
                r.rms,
                r.zcr
            );
        }
        println!("\n  Saved {csv_path} ({} rows)", log.len());
    }

    println!("\n  Done.");
}

#[cfg(feature = "muse")]
struct CycleRecord {
    cycle: u32,
    psi: f32,
    arousal: f32,
    valence: f32,
    prediction_error: f32,
    dopamine: f32,
    serotonin: f32,
    noradrenaline: f32,
    rms: f32,
    zcr: f32,
}

#[cfg(feature = "muse")]
fn print_corr(name: &str, x: &[f32], y: &[f32]) {
    let n = x.len() as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let mut sxy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let r = if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        0.0
    };
    let r2 = r * r;
    let rating = if r2 > 0.5 {
        "STRONG"
    } else if r2 > 0.3 {
        "MODERATE"
    } else if r2 > 0.1 {
        "WEAK"
    } else {
        "NONE"
    };
    println!("  {:<35} r={:+.3}  R²={:.4}  [{}]", name, r, r2, rating);
}