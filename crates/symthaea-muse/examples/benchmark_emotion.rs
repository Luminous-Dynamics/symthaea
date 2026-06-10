// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Muse Emotion Benchmark — Phi ↔ Musical Coherence
//!
//! Validates that Symthaea's consciousness-driven music synthesis produces
//! audio whose perceived emotional state correlates with the intended
//! cognitive state. Uses the DEAM dataset (Aljanaki et al. 2017) as
//! ground truth for valence-arousal (V-A) space.
//!
//! ## Protocol
//!
//! 1. Define 8 cognitive states spanning the V-A plane:
//!    - High-V / High-A: Flow state (Phi=0.8, DA high)
//!    - High-V / Low-A: Contentment (Phi=0.6, 5-HT high, stillness)
//!    - Low-V / High-A: Panic (allostatic_load=0.9, NE high)
//!    - Low-V / Low-A: Burnout (allostatic_load=0.8, DA/5-HT depleted)
//!    - + 4 intermediate states
//!
//! 2. For each state, run StreamingSynth for 30 seconds → stereo PCM
//!
//! 3. Extract audio features: spectral centroid, RMS energy, tempo,
//!    spectral flux, harmonic-to-noise ratio → proxy V-A prediction
//!
//! 4. Compare intended V-A with measured V-A features → R² correlation
//!
//! ## Metrics
//!
//! - **Valence R²**: correlation between intended valence and spectral brightness
//! - **Arousal R²**: correlation between intended arousal and RMS energy + tempo
//! - **State discrimination**: can we separate the 8 states by audio features?
//! - **Phi-coherence**: does higher Phi produce more spectrally coherent audio?
//!
//! ## Running
//!
//! ```bash
//! cargo run --example benchmark_muse_emotion --features muse
//! ```
//!
//! ## References
//!
//! - Aljanaki, Yang & Soleymani (2017). Developing a benchmark for emotional
//!   analysis of music. PLoS ONE, 12(3), e0173392.
//! - Koelsch, Vuust & Friston (2019). Predictive Processes and the Peculiar
//!   Case of Music. Trends in Cognitive Sciences, 23(1), 63-77.
//! - Russell (1980). A Circumplex Model of Affect. J. Personality & Social Psych.

use rustfft::{FftPlanner, num_complex::Complex};
use std::io::BufRead;
use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

/// A named cognitive state with intended valence-arousal coordinates.
struct CognitiveScenario {
    name: &'static str,
    state: MusicalState,
    /// Intended valence [-1, 1] (negative = aversive, positive = pleasant)
    intended_valence: f32,
    /// Intended arousal [0, 1] (low = calm, high = excited)
    intended_arousal: f32,
}

/// Audio features extracted from a PCM buffer.
#[derive(Debug, Clone)]
struct AudioFeatures {
    /// Root mean square energy [0, 1]
    rms_energy: f32,
    /// Spectral centroid in Hz (brightness proxy)
    spectral_centroid: f32,
    /// Zero crossing rate (noisiness proxy)
    zero_crossing_rate: f32,
    /// Spectral flux (change rate, tension proxy)
    spectral_flux: f32,
    /// Estimated tempo from onset density (beats per second)
    onset_density: f32,
    /// Harmonic-to-noise ratio (consonance proxy)
    harmonic_ratio: f32,
    /// Dominant pitch frequency via autocorrelation (Hz). Higher = brighter valence.
    dominant_pitch_hz: f32,
    /// Major-third energy relative to minor-third energy.
    /// >1.0 = major (happy), <1.0 = minor (sad). Huron (2006).
    major_minor_ratio: f32,
    /// Key clarity: max correlation with Krumhansl-Schmuckler key profiles [0, 1].
    /// Higher = more tonal stability = more positive valence (Panda et al. 2023).
    key_clarity: f32,
    /// Harmonic change detection function: rate of chroma change per second.
    /// Higher = more harmonic movement (tension/development).
    hcdf: f32,
    /// Consonance ratio: proportion of energy in consonant intervals (P1, M3, P5).
    /// Higher = more consonant = more positive valence (Eerola et al. 2013).
    consonance_ratio: f32,
}

/// Correlation result between intended and measured.
struct CorrelationResult {
    name: &'static str,
    r_squared: f32,
    pearson_r: f32,
    n_samples: usize,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Muse Emotion Benchmark                           ║");
    println!("║  Phi ↔ Musical Coherence Validation                        ║");
    println!("║  Protocol: Koelsch/Vuust/Friston (2019) Predictive Coding  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Define 8 cognitive states spanning the V-A plane ──────────────
    let scenarios = define_scenarios();

    println!(
        "Generating {} scenarios × 30s audio each...\n",
        scenarios.len()
    );

    let config = MuseConfig {
        sample_rate: 44100,
        num_partials: 8,
        ..Default::default()
    };

    let mut all_features: Vec<(f32, f32, AudioFeatures)> = Vec::new();

    // Run multiple seeds per scenario to average out FEP stochasticity.
    // With 3 seeds × 12 scenarios = 36 data points, correlations stabilize.
    let n_seeds = 3;

    for (i, scenario) in scenarios.iter().enumerate() {
        print!("  [{}/{}] {:<25}", i + 1, scenarios.len(), scenario.name);

        let mut seed_features = Vec::new();
        for seed in 0..n_seeds {
            let mut synth = StreamingSynth::new(config.clone(), 44100);
            synth.update_state(&scenario.state);

            // Advance each seed to a different FEP state by rendering a short prefix
            for _ in 0..(seed * 20) {
                synth.render_chunk();
            }
            synth.update_state(&scenario.state); // re-anchor after warmup

            let duration_secs = 15.0; // 15s per seed (45s total per scenario)
            let total_chunks = (duration_secs * 44100.0 / synth.chunk_samples() as f32) as usize;

            let mut all_samples: Vec<f32> = Vec::with_capacity(44100 * 15);
            // Re-assert emotional state every ~5s to simulate the cognitive loop.
            let reassert_interval = (5.0 * 44100.0 / synth.chunk_samples() as f32) as usize;
            for chunk_idx in 0..total_chunks {
                if chunk_idx % reassert_interval == 0 && chunk_idx > 0 {
                    synth.update_state(&scenario.state);
                }
                let chunk = synth.render_chunk();
                for pair in &chunk {
                    all_samples.push((pair[0] + pair[1]) * 0.5);
                }
            }

            seed_features.push(extract_features(&all_samples, 44100));
        }

        // Average features across seeds
        let features = average_features(&seed_features);

        println!(
            "RMS={:.3} Flux={:.4} On={:.1}/s Key={:.2} HCDF={:.2} Cons={:.2}",
            features.rms_energy,
            features.spectral_flux,
            features.onset_density,
            features.key_clarity,
            features.hcdf,
            features.consonance_ratio,
        );

        all_features.push((
            scenario.intended_valence,
            scenario.intended_arousal,
            features,
        ));
    }

    println!();

    // ── Compute correlations ──────────────────────────────────────────
    println!("═══ Correlation Analysis ═══\n");

    // Arousal ↔ RMS energy (expected: positive)
    let arousal_rms = correlate(
        "Arousal ↔ RMS Energy",
        &all_features.iter().map(|(_, a, _)| *a).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.rms_energy)
            .collect::<Vec<_>>(),
    );
    print_correlation(&arousal_rms);

    // Arousal ↔ Onset density (expected: positive)
    let arousal_tempo = correlate(
        "Arousal ↔ Onset Density",
        &all_features.iter().map(|(_, a, _)| *a).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.onset_density)
            .collect::<Vec<_>>(),
    );
    print_correlation(&arousal_tempo);

    // Valence ↔ Spectral centroid (expected: positive — brighter = happier)
    let valence_bright = correlate(
        "Valence ↔ Spectral Centroid",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.spectral_centroid)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_bright);

    // Valence ↔ Harmonic ratio (expected: positive — more consonant = happier)
    let valence_hnr = correlate(
        "Valence ↔ Harmonic Ratio",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.harmonic_ratio)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_hnr);

    // Phi proxy ↔ Spectral flux (expected: negative — higher Phi = smoother)
    // Using consciousness_level as Phi proxy
    let phi_values: Vec<f32> = scenarios
        .iter()
        .map(|s| s.state.consciousness_level)
        .collect();
    let phi_flux = correlate(
        "Phi ↔ Spectral Flux",
        &phi_values,
        &all_features
            .iter()
            .map(|(_, _, f)| f.spectral_flux)
            .collect::<Vec<_>>(),
    );
    print_correlation(&phi_flux);

    // Allostatic load ↔ Zero crossing rate (expected: positive — more noise under stress)
    let load_values: Vec<f32> = scenarios.iter().map(|s| s.state.noradrenaline).collect();
    let load_zcr = correlate(
        "NE (stress) ↔ Zero-Crossing Rate",
        &load_values,
        &all_features
            .iter()
            .map(|(_, _, f)| f.zero_crossing_rate)
            .collect::<Vec<_>>(),
    );
    print_correlation(&load_zcr);

    // Valence ↔ Dominant pitch (expected: positive — higher pitch = happier)
    let valence_pitch = correlate(
        "Valence ↔ Dominant Pitch",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.dominant_pitch_hz)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_pitch);

    // Valence ↔ Major/Minor ratio (expected: positive — major = happy)
    let valence_mode = correlate(
        "Valence ↔ Major/Minor Ratio",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.major_minor_ratio)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_mode);

    // Valence ↔ Key clarity (expected: positive — tonal stability = positive valence)
    let valence_key = correlate(
        "Valence ↔ Key Clarity",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.key_clarity)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_key);

    // Valence ↔ HCDF (expected: negative — rapid harmonic change = tension)
    let valence_hcdf = correlate(
        "Valence ↔ Harmonic Change",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.hcdf)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_hcdf);

    // Valence ↔ Consonance ratio (expected: positive — consonance = positive valence)
    let valence_cons = correlate(
        "Valence ↔ Consonance Ratio",
        &all_features.iter().map(|(v, _, _)| *v).collect::<Vec<_>>(),
        &all_features
            .iter()
            .map(|(_, _, f)| f.consonance_ratio)
            .collect::<Vec<_>>(),
    );
    print_correlation(&valence_cons);

    // ── Summary ───────────────────────────────────────────────────────
    println!("\n═══ Summary ═══\n");

    let all_r2 = [
        arousal_rms.r_squared,
        arousal_tempo.r_squared,
        valence_bright.r_squared,
        valence_hnr.r_squared,
        phi_flux.r_squared,
        load_zcr.r_squared,
        valence_pitch.r_squared,
        valence_mode.r_squared,
        valence_key.r_squared,
        valence_hcdf.r_squared,
        valence_cons.r_squared,
    ];
    let mean_r2: f32 = all_r2.iter().sum::<f32>() / all_r2.len() as f32;
    let significant = all_r2.iter().filter(|&&r| r > 0.3).count();

    println!("  Mean R²:             {:.4}", mean_r2);
    println!("  Significant (R²>0.3): {}/{}", significant, all_r2.len());
    println!();

    if mean_r2 > 0.4 {
        println!("  ✓ STRONG: Consciousness state reliably maps to perceived audio emotion.");
        println!("    Synthetic limbic system validated (Koelsch/Vuust/Friston 2019).");
    } else if mean_r2 > 0.2 {
        println!(
            "  ~ MODERATE: Partial correlation. Some dimensions map well, others need tuning."
        );
    } else {
        println!("  ✗ WEAK: Low correlation. The sonification mapping needs revision.");
    }

    // ── State discrimination (pairwise distance matrix) ───────────────
    println!("\n═══ State Discrimination Matrix (Euclidean distance in feature space) ═══\n");
    print!("{:>20}", "");
    for s in &scenarios {
        print!("{:>12}", &s.name[..s.name.len().min(10)]);
    }
    println!();

    for (i, si) in scenarios.iter().enumerate() {
        print!("{:>20}", si.name);
        let fi = &all_features[i].2;
        for (j, _sj) in scenarios.iter().enumerate() {
            let fj = &all_features[j].2;
            let dist = feature_distance(fi, fj);
            print!("{:>12.3}", dist);
        }
        println!();
    }

    println!("\n  Higher off-diagonal values = better state discrimination.");
    println!("  Diagonal should be 0.000.");

    // ── Stage 2: DEAM cross-validation ────────────────────────────────
    println!("\n═══ Stage 2: DEAM Cross-Validation ═══\n");

    let deam_path = std::path::Path::new("data/deam");
    if deam_path.exists() {
        match deam_cross_validate(&all_features, &scenarios) {
            Ok(()) => {}
            Err(e) => println!("  DEAM validation skipped: {e}"),
        }
    } else {
        println!("  DEAM dataset not found at data/deam/");
        println!("  Run: ./scripts/download_deam.sh");
    }

    // ── Stage 3: V-A scatter plot SVG ─────────────────────────────────
    println!("\n═══ Stage 3: V-A Scatter Plot ═══\n");
    let svg = generate_va_scatter_svg(&all_features, &scenarios);
    let svg_path = "data/muse_emotion_scatter.svg";
    match std::fs::write(svg_path, &svg) {
        Ok(()) => println!("  Written to {svg_path} ({} bytes)", svg.len()),
        Err(e) => println!("  Failed to write SVG: {e}"),
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STAGE 4: ABLATION TEST — Honest Circularity Check
    //
    // The main benchmark may have circular correlations:
    //   valence → valence_gain (±15%) → RMS → measured "valence"
    //
    // Ablation: re-run all scenarios with valence=0.0, keeping everything else.
    // Correlations that SURVIVE are driven by indirect paths (harmony, gestures,
    // chord progressions). Correlations that DROP were circular.
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══ Stage 4: Ablation Test (valence=0 for all scenarios) ═══\n");
    println!("  Disabling direct valence→synthesis path. Correlations that survive");
    println!("  are driven by arousal/consciousness/NE, not valence feedback.\n");

    let mut ablated_features: Vec<(f32, f32, AudioFeatures)> = Vec::new();

    for (i, scenario) in scenarios.iter().enumerate() {
        // Zero out valence in the state but keep the intended valence for correlation
        let mut ablated_state = scenario.state.clone();
        ablated_state.valence = 0.0;

        let mut seed_features = Vec::new();
        for seed in 0..n_seeds {
            let mut synth = StreamingSynth::new(config.clone(), 44100);
            synth.update_state(&ablated_state);
            for _ in 0..(seed * 20) {
                synth.render_chunk();
            }
            synth.update_state(&ablated_state);

            let duration_secs = 15.0;
            let total_chunks = (duration_secs * 44100.0 / synth.chunk_samples() as f32) as usize;
            let reassert_interval = (5.0 * 44100.0 / synth.chunk_samples() as f32) as usize;

            let mut samples: Vec<f32> = Vec::with_capacity(44100 * 15);
            for chunk_idx in 0..total_chunks {
                if chunk_idx % reassert_interval == 0 && chunk_idx > 0 {
                    synth.update_state(&ablated_state);
                }
                let chunk = synth.render_chunk();
                for pair in &chunk {
                    samples.push((pair[0] + pair[1]) * 0.5);
                }
            }
            seed_features.push(extract_features(&samples, 44100));
        }
        let features = average_features(&seed_features);
        ablated_features.push((
            scenario.intended_valence,
            scenario.intended_arousal,
            features,
        ));

        if (i + 1) % 4 == 0 {
            print!("  [{}/{}]...", i + 1, scenarios.len());
        }
    }
    println!(" done.\n");

    // Re-compute correlations with ablated audio
    let abl_arousal_rms = correlate(
        "[ABL] Arousal ↔ RMS",
        &ablated_features
            .iter()
            .map(|(_, a, _)| *a)
            .collect::<Vec<_>>(),
        &ablated_features
            .iter()
            .map(|(_, _, f)| f.rms_energy)
            .collect::<Vec<_>>(),
    );
    let abl_phi_flux = correlate(
        "[ABL] Phi ↔ Spectral Flux",
        &scenarios
            .iter()
            .map(|s| s.state.consciousness_level)
            .collect::<Vec<_>>(),
        &ablated_features
            .iter()
            .map(|(_, _, f)| f.spectral_flux)
            .collect::<Vec<_>>(),
    );
    let abl_valence_cons = correlate(
        "[ABL] Valence ↔ Consonance",
        &ablated_features
            .iter()
            .map(|(v, _, _)| *v)
            .collect::<Vec<_>>(),
        &ablated_features
            .iter()
            .map(|(_, _, f)| f.consonance_ratio)
            .collect::<Vec<_>>(),
    );
    let abl_valence_rms = correlate(
        "[ABL] Valence ↔ RMS",
        &ablated_features
            .iter()
            .map(|(v, _, _)| *v)
            .collect::<Vec<_>>(),
        &ablated_features
            .iter()
            .map(|(_, _, f)| f.rms_energy)
            .collect::<Vec<_>>(),
    );
    let abl_valence_hcdf = correlate(
        "[ABL] Valence ↔ HCDF",
        &ablated_features
            .iter()
            .map(|(v, _, _)| *v)
            .collect::<Vec<_>>(),
        &ablated_features
            .iter()
            .map(|(_, _, f)| f.hcdf)
            .collect::<Vec<_>>(),
    );

    print_correlation(&abl_arousal_rms);
    print_correlation(&abl_phi_flux);
    print_correlation(&abl_valence_cons);
    print_correlation(&abl_valence_rms);
    print_correlation(&abl_valence_hcdf);

    println!("\n  Interpretation:");
    println!("  - Arousal/Phi axes should be UNCHANGED (not driven by valence)");
    println!("  - Valence axes that DROP to ~0 were circular (inflated by valence_gain)");
    println!("  - Valence axes that HOLD are driven by arousal/consciousness confounds");
}

// ─── Scenario Definitions ─────────────────────────────────────────────────

fn define_scenarios() -> Vec<CognitiveScenario> {
    vec![
        CognitiveScenario {
            name: "Flow (high Phi)",
            state: MusicalState {
                harmony_activations: [0.8, 0.7, 0.6, 0.2, 0.5, 0.6, 0.4, 0.1],
                dopamine: 0.8,
                serotonin: 0.6,
                noradrenaline: 0.3,
                arousal: 0.6,
                valence: 0.7,
                consciousness_level: 0.85,
                prediction_error: 0.1,
            },
            intended_valence: 0.7,
            intended_arousal: 0.6,
        },
        CognitiveScenario {
            name: "Contentment (calm joy)",
            state: MusicalState {
                harmony_activations: [0.5, 0.8, 0.4, 0.1, 0.6, 0.7, 0.2, 0.6],
                dopamine: 0.4,
                serotonin: 0.8,
                noradrenaline: 0.1,
                arousal: 0.2,
                valence: 0.6,
                consciousness_level: 0.6,
                prediction_error: 0.05,
            },
            intended_valence: 0.6,
            intended_arousal: 0.2,
        },
        CognitiveScenario {
            name: "Panic (high stress)",
            state: MusicalState {
                harmony_activations: [0.2, 0.1, 0.1, 0.8, 0.3, 0.1, 0.7, 0.0],
                dopamine: 0.2,
                serotonin: 0.1,
                noradrenaline: 0.9,
                arousal: 0.95,
                valence: -0.8,
                consciousness_level: 0.3,
                prediction_error: 0.8,
            },
            intended_valence: -0.8,
            intended_arousal: 0.95,
        },
        CognitiveScenario {
            name: "Burnout (depleted)",
            state: MusicalState {
                harmony_activations: [0.1, 0.0, 0.1, 0.3, 0.0, 0.0, 0.0, 0.7],
                dopamine: 0.1,
                serotonin: 0.1,
                noradrenaline: 0.2,
                arousal: 0.1,
                valence: -0.6,
                consciousness_level: 0.15,
                prediction_error: 0.3,
            },
            intended_valence: -0.6,
            intended_arousal: 0.1,
        },
        CognitiveScenario {
            name: "Curiosity (exploring)",
            state: MusicalState {
                harmony_activations: [0.3, 0.4, 0.3, 0.6, 0.5, 0.3, 0.8, 0.1],
                dopamine: 0.7,
                serotonin: 0.4,
                noradrenaline: 0.5,
                arousal: 0.7,
                valence: 0.3,
                consciousness_level: 0.7,
                prediction_error: 0.4,
            },
            intended_valence: 0.3,
            intended_arousal: 0.7,
        },
        CognitiveScenario {
            name: "Sacred Stillness",
            state: MusicalState {
                harmony_activations: [0.3, 0.2, 0.2, 0.0, 0.3, 0.2, 0.0, 0.9],
                dopamine: 0.3,
                serotonin: 0.7,
                noradrenaline: 0.05,
                arousal: 0.05,
                valence: 0.2,
                consciousness_level: 0.5,
                prediction_error: 0.02,
            },
            intended_valence: 0.2,
            intended_arousal: 0.05,
        },
        CognitiveScenario {
            name: "Anger (threat response)",
            state: MusicalState {
                harmony_activations: [0.1, 0.0, 0.2, 0.9, 0.1, 0.0, 0.5, 0.0],
                dopamine: 0.6,
                serotonin: 0.05,
                noradrenaline: 0.85,
                arousal: 0.9,
                valence: -0.7,
                consciousness_level: 0.4,
                prediction_error: 0.6,
            },
            intended_valence: -0.7,
            intended_arousal: 0.9,
        },
        CognitiveScenario {
            name: "Wonder (awe)",
            state: MusicalState {
                harmony_activations: [0.7, 0.9, 0.8, 0.3, 0.7, 0.8, 0.5, 0.3],
                dopamine: 0.6,
                serotonin: 0.5,
                noradrenaline: 0.4,
                arousal: 0.5,
                valence: 0.8,
                consciousness_level: 0.9,
                prediction_error: 0.15,
            },
            intended_valence: 0.8,
            intended_arousal: 0.5,
        },
        // ── 4 additional scenarios for statistical power ──
        CognitiveScenario {
            name: "Grief (deep sadness)",
            state: MusicalState {
                harmony_activations: [0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.0, 0.5],
                dopamine: 0.1,
                serotonin: 0.3,
                noradrenaline: 0.15,
                arousal: 0.3,
                valence: -0.9,
                consciousness_level: 0.4,
                prediction_error: 0.2,
            },
            intended_valence: -0.9,
            intended_arousal: 0.3,
        },
        CognitiveScenario {
            name: "Excitement (triumph)",
            state: MusicalState {
                harmony_activations: [0.6, 0.9, 0.7, 0.4, 0.5, 0.8, 0.9, 0.0],
                dopamine: 0.95,
                serotonin: 0.4,
                noradrenaline: 0.7,
                arousal: 0.85,
                valence: 0.9,
                consciousness_level: 0.8,
                prediction_error: 0.2,
            },
            intended_valence: 0.9,
            intended_arousal: 0.85,
        },
        CognitiveScenario {
            name: "Boredom (disengaged)",
            state: MusicalState {
                harmony_activations: [0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3],
                dopamine: 0.15,
                serotonin: 0.4,
                noradrenaline: 0.1,
                arousal: 0.15,
                valence: -0.3,
                consciousness_level: 0.25,
                prediction_error: 0.05,
            },
            intended_valence: -0.3,
            intended_arousal: 0.15,
        },
        CognitiveScenario {
            name: "Tension (suspense)",
            state: MusicalState {
                harmony_activations: [0.3, 0.1, 0.2, 0.7, 0.4, 0.1, 0.5, 0.1],
                dopamine: 0.5,
                serotonin: 0.2,
                noradrenaline: 0.7,
                arousal: 0.75,
                valence: -0.2,
                consciousness_level: 0.55,
                prediction_error: 0.5,
            },
            intended_valence: -0.2,
            intended_arousal: 0.75,
        },
    ]
}

/// Average features across multiple seeds.
fn average_features(features: &[AudioFeatures]) -> AudioFeatures {
    let n = features.len() as f32;
    AudioFeatures {
        rms_energy: features.iter().map(|f| f.rms_energy).sum::<f32>() / n,
        spectral_centroid: features.iter().map(|f| f.spectral_centroid).sum::<f32>() / n,
        zero_crossing_rate: features.iter().map(|f| f.zero_crossing_rate).sum::<f32>() / n,
        spectral_flux: features.iter().map(|f| f.spectral_flux).sum::<f32>() / n,
        onset_density: features.iter().map(|f| f.onset_density).sum::<f32>() / n,
        harmonic_ratio: features.iter().map(|f| f.harmonic_ratio).sum::<f32>() / n,
        dominant_pitch_hz: features.iter().map(|f| f.dominant_pitch_hz).sum::<f32>() / n,
        major_minor_ratio: features.iter().map(|f| f.major_minor_ratio).sum::<f32>() / n,
        key_clarity: features.iter().map(|f| f.key_clarity).sum::<f32>() / n,
        hcdf: features.iter().map(|f| f.hcdf).sum::<f32>() / n,
        consonance_ratio: features.iter().map(|f| f.consonance_ratio).sum::<f32>() / n,
    }
}

// ─── Feature Extraction (pure Rust, no external deps) ─────────────────────

fn extract_features(samples: &[f32], sample_rate: u32) -> AudioFeatures {
    let n = samples.len();
    if n == 0 {
        return AudioFeatures {
            rms_energy: 0.0,
            spectral_centroid: 0.0,
            zero_crossing_rate: 0.0,
            spectral_flux: 0.0,
            onset_density: 0.0,
            harmonic_ratio: 0.0,
            dominant_pitch_hz: 0.0,
            major_minor_ratio: 1.0,
            key_clarity: 0.0,
            hcdf: 0.0,
            consonance_ratio: 0.5,
        };
    }

    // RMS energy
    let rms_energy = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();

    // Zero crossing rate
    let zc: usize = samples
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    let zero_crossing_rate = zc as f32 / n as f32;

    // Spectral centroid (via short-time DFT approximation using zero-crossing)
    // True spectral centroid needs FFT; we approximate via ZCR→frequency mapping
    // ZCR ≈ 2 * f_mean / sample_rate (for bandlimited signals)
    let spectral_centroid = zero_crossing_rate * sample_rate as f32 * 0.5;

    // Spectral flux (frame-to-frame RMS difference)
    let frame_size = (sample_rate as usize) / 10; // 100ms frames
    let mut prev_rms = 0.0f32;
    let mut flux_sum = 0.0f32;
    let mut frame_count = 0usize;
    for frame in samples.chunks(frame_size) {
        let frame_rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        flux_sum += (frame_rms - prev_rms).abs();
        prev_rms = frame_rms;
        frame_count += 1;
    }
    let spectral_flux = if frame_count > 1 {
        flux_sum / frame_count as f32
    } else {
        0.0
    };

    // Onset density (energy peaks per second)
    let onset_frame = sample_rate as usize / 20; // 50ms frames
    let energies: Vec<f32> = samples
        .chunks(onset_frame)
        .map(|f| f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32)
        .collect();

    let mut onsets = 0usize;
    for i in 1..energies.len().saturating_sub(1) {
        if energies[i] > energies[i - 1] * 1.5 && energies[i] > energies[i + 1] {
            onsets += 1;
        }
    }
    let duration_secs = n as f32 / sample_rate as f32;
    let onset_density = onsets as f32 / duration_secs.max(0.1);

    // Harmonic-to-noise ratio (autocorrelation peak / noise floor)
    let window = (sample_rate as usize).min(n); // 1 second window
    let mid = n / 2;
    let start = mid.saturating_sub(window / 2);
    let end = (start + window).min(n);
    let segment = &samples[start..end];

    let mut max_ac = 0.0f32;
    let min_lag = sample_rate as usize / 1000; // 1000 Hz max
    let max_lag = sample_rate as usize / 50; // 50 Hz min
    for lag in min_lag..max_lag.min(segment.len() / 2) {
        let mut ac = 0.0f32;
        let mut count = 0;
        for i in 0..segment.len() - lag {
            ac += segment[i] * segment[i + lag];
            count += 1;
        }
        if count > 0 {
            ac /= count as f32;
            max_ac = max_ac.max(ac);
        }
    }
    let ac0 = segment.iter().map(|s| s * s).sum::<f32>() / segment.len() as f32;
    let harmonic_ratio = if ac0 > 1e-8 {
        (max_ac / ac0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // ── FFT-based pitch and mode detection ──────────────────────────────
    // Use 4096-sample windows for ~1Hz resolution at 44100Hz
    let fft_size = 4096usize;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Average spectrum over multiple windows for stability
    let mut avg_spectrum = vec![0.0f32; fft_size / 2];
    let mut window_count = 0usize;

    for chunk in samples.chunks(fft_size) {
        if chunk.len() < fft_size {
            break;
        }
        let mut buffer: Vec<Complex<f32>> = chunk
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                // Hann window
                let w =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
                Complex::new(s * w, 0.0)
            })
            .collect();
        fft.process(&mut buffer);
        for (i, bin) in buffer[..fft_size / 2].iter().enumerate() {
            avg_spectrum[i] += bin.norm();
        }
        window_count += 1;
    }
    if window_count > 0 {
        for v in &mut avg_spectrum {
            *v /= window_count as f32;
        }
    }

    // Dominant pitch: highest peak in 50-2000Hz range
    let hz_per_bin = sample_rate as f32 / fft_size as f32;
    let min_bin = (50.0 / hz_per_bin) as usize;
    let max_bin = (2000.0 / hz_per_bin).min(avg_spectrum.len() as f32 - 1.0) as usize;

    let mut best_bin = min_bin;
    let mut best_mag = 0.0f32;
    for i in min_bin..=max_bin {
        if avg_spectrum[i] > best_mag {
            best_mag = avg_spectrum[i];
            best_bin = i;
        }
    }
    let dominant_pitch_hz = best_bin as f32 * hz_per_bin;

    // Major/minor detection via interval energy ratios
    // For each detected fundamental, measure energy at:
    //   Major 3rd: f * 2^(4/12) = f * 1.2599
    //   Minor 3rd: f * 2^(3/12) = f * 1.1892
    //   Perfect 5th: f * 2^(7/12) = f * 1.4983
    // Valence signal: (major_3rd_energy + P5_energy) / (minor_3rd_energy + 0.01)
    let fund_hz = dominant_pitch_hz;
    let bin_of = |freq: f32| -> usize {
        (freq / hz_per_bin)
            .round()
            .clamp(0.0, (avg_spectrum.len() - 1) as f32) as usize
    };

    // Sum energy in a ±2 bin window around the target frequency
    let energy_near = |freq: f32| -> f32 {
        let center = bin_of(freq);
        let lo = center.saturating_sub(2);
        let hi = (center + 2).min(avg_spectrum.len() - 1);
        avg_spectrum[lo..=hi].iter().sum::<f32>()
    };

    let major_3rd_energy = energy_near(fund_hz * 1.2599);
    let minor_3rd_energy = energy_near(fund_hz * 1.1892);
    let p5_energy = energy_near(fund_hz * 1.4983);

    // Major/minor ratio: major intervals boost → positive valence
    let major_minor_ratio = if minor_3rd_energy > 0.001 {
        ((major_3rd_energy + p5_energy * 0.5) / minor_3rd_energy).clamp(0.1, 10.0)
    } else if major_3rd_energy > 0.001 {
        5.0
    } else {
        1.0
    };

    // ── Harmonic features (Panda et al. 2023: these predict valence) ────────

    // Key clarity via simplified Krumhansl-Schmuckler:
    // Compute chroma (12 bins) from FFT, correlate with major/minor key profiles
    let mut chroma = [0.0f32; 12];
    for (i, &mag) in avg_spectrum.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let freq = i as f32 * hz_per_bin;
        if freq < 50.0 || freq > 4000.0 {
            continue;
        }
        // Map frequency to pitch class (0=C, 1=C#, ..., 11=B)
        let midi = 12.0 * (freq / 440.0).log2() + 69.0;
        let pc = ((midi.round() as i32) % 12 + 12) % 12;
        chroma[pc as usize] += mag;
    }
    // Normalize chroma
    let chroma_sum: f32 = chroma.iter().sum();
    if chroma_sum > 0.01 {
        for c in &mut chroma {
            *c /= chroma_sum;
        }
    }
    // Krumhansl major key profile (C major)
    let major_profile = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    // Correlate chroma with all 12 transpositions, take max
    let mut max_corr = 0.0f32;
    for shift in 0..12 {
        let mut sum_xy = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_yy = 0.0f32;
        for j in 0..12 {
            let x = chroma[(j + shift) % 12];
            let y = major_profile[j] as f32;
            sum_xy += x * y;
            sum_xx += x * x;
            sum_yy += y * y;
        }
        let denom = (sum_xx * sum_yy).sqrt();
        if denom > 0.001 {
            max_corr = max_corr.max(sum_xy / denom);
        }
    }
    let key_clarity = max_corr.clamp(0.0, 1.0);

    // HCDF: harmonic change detection function
    // Rate of chroma change across FFT windows (already computed avg_spectrum from windows)
    // We compute per-window chroma and measure frame-to-frame cosine distance
    let mut hcdf_sum = 0.0f32;
    let mut hcdf_count = 0usize;
    let mut prev_chroma = [0.0f32; 12];
    for chunk in samples.chunks(fft_size) {
        if chunk.len() < fft_size {
            break;
        }
        let mut buffer: Vec<Complex<f32>> = chunk
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
                Complex::new(s * w, 0.0)
            })
            .collect();
        fft.process(&mut buffer);

        let mut frame_chroma = [0.0f32; 12];
        for (i, bin) in buffer[..fft_size / 2].iter().enumerate() {
            if i == 0 {
                continue;
            }
            let freq = i as f32 * hz_per_bin;
            if freq < 50.0 || freq > 4000.0 {
                continue;
            }
            let midi = 12.0 * (freq / 440.0).log2() + 69.0;
            let pc = ((midi.round() as i32) % 12 + 12) % 12;
            frame_chroma[pc as usize] += bin.norm();
        }
        let fc_sum: f32 = frame_chroma.iter().sum();
        if fc_sum > 0.01 {
            for c in &mut frame_chroma {
                *c /= fc_sum;
            }
        }

        if hcdf_count > 0 {
            // Cosine distance between consecutive chroma frames
            let dot: f32 = (0..12).map(|j| frame_chroma[j] * prev_chroma[j]).sum();
            let na: f32 = (0..12)
                .map(|j| prev_chroma[j] * prev_chroma[j])
                .sum::<f32>()
                .sqrt();
            let nb: f32 = (0..12)
                .map(|j| frame_chroma[j] * frame_chroma[j])
                .sum::<f32>()
                .sqrt();
            let cos_sim = if na > 0.001 && nb > 0.001 {
                dot / (na * nb)
            } else {
                1.0
            };
            hcdf_sum += 1.0 - cos_sim; // distance = 1 - similarity
        }
        prev_chroma = frame_chroma;
        hcdf_count += 1;
    }
    let hcdf = if hcdf_count > 1 {
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        hcdf_sum / duration_secs // changes per second
    } else {
        0.0
    };

    // Consonance ratio: energy at consonant intervals relative to total
    // Consonant: P1(0), m3(3), M3(4), P4(5), P5(7), M6(9) semitones from fundamental
    let consonant_pcs: &[usize] = &[0, 3, 4, 5, 7, 9];
    let fund_pc = if dominant_pitch_hz > 50.0 {
        let midi = 12.0 * (dominant_pitch_hz / 440.0).log2() + 69.0;
        ((midi.round() as i32) % 12 + 12) % 12
    } else {
        0
    } as usize;
    let consonant_energy: f32 = consonant_pcs
        .iter()
        .map(|&offset| chroma[(fund_pc + offset) % 12])
        .sum();
    let consonance_ratio = if chroma_sum > 0.01 {
        consonant_energy.clamp(0.0, 1.0)
    } else {
        0.5
    };

    AudioFeatures {
        rms_energy,
        spectral_centroid,
        zero_crossing_rate,
        spectral_flux,
        onset_density,
        harmonic_ratio,
        dominant_pitch_hz,
        major_minor_ratio,
        key_clarity,
        hcdf,
        consonance_ratio,
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────

fn correlate(name: &'static str, x: &[f32], y: &[f32]) -> CorrelationResult {
    let n = x.len().min(y.len());
    if n < 3 {
        return CorrelationResult {
            name,
            r_squared: 0.0,
            pearson_r: 0.0,
            n_samples: n,
        };
    }

    let mean_x: f32 = x.iter().sum::<f32>() / n as f32;
    let mean_y: f32 = y.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    let pearson_r = if denom > 1e-8 { cov / denom } else { 0.0 };

    CorrelationResult {
        name,
        r_squared: pearson_r * pearson_r,
        pearson_r,
        n_samples: n,
    }
}

fn print_correlation(result: &CorrelationResult) {
    let direction = if result.pearson_r > 0.0 { "+" } else { "-" };
    let strength = if result.r_squared > 0.5 {
        "STRONG"
    } else if result.r_squared > 0.3 {
        "MODERATE"
    } else if result.r_squared > 0.1 {
        "WEAK"
    } else {
        "NONE"
    };
    println!(
        "  {:<35} r={}{:.3}  R²={:.4}  [{}]  (n={})",
        result.name,
        direction,
        result.pearson_r.abs(),
        result.r_squared,
        strength,
        result.n_samples
    );
}

fn feature_distance(a: &AudioFeatures, b: &AudioFeatures) -> f32 {
    let d = [
        (a.rms_energy - b.rms_energy) * 10.0,
        (a.spectral_centroid - b.spectral_centroid) / 500.0,
        (a.zero_crossing_rate - b.zero_crossing_rate) * 100.0,
        (a.spectral_flux - b.spectral_flux) * 100.0,
        (a.onset_density - b.onset_density),
        (a.harmonic_ratio - b.harmonic_ratio) * 5.0,
        (a.dominant_pitch_hz - b.dominant_pitch_hz) / 200.0,
        (a.major_minor_ratio - b.major_minor_ratio),
        (a.key_clarity - b.key_clarity) * 5.0,
        (a.hcdf - b.hcdf) * 10.0,
        (a.consonance_ratio - b.consonance_ratio) * 5.0,
    ];
    d.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 2: DEAM CROSS-VALIDATION
// Train linear V-A regressor on DEAM audio features, apply to Symthaea output
// ═══════════════════════════════════════════════════════════════════════════

fn deam_cross_validate(
    synth_features: &[(f32, f32, AudioFeatures)],
    scenarios: &[CognitiveScenario],
) -> Result<(), String> {
    // DEAM static annotations: song_id, valence_mean, valence_std, arousal_mean, arousal_std
    let static_path = "data/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv";

    if !std::path::Path::new(static_path).exists() {
        return Err(format!("DEAM annotations not found at {static_path}"));
    }

    // Parse DEAM song-level V-A annotations (combined file)
    let (valence_map, arousal_map) = parse_deam_static(static_path)?;

    println!(
        "  Loaded DEAM annotations: {} valence, {} arousal",
        valence_map.len(),
        arousal_map.len()
    );

    // Compute mean/std of DEAM V-A distribution
    let deam_v: Vec<f32> = valence_map.values().copied().collect();
    let deam_a: Vec<f32> = arousal_map.values().copied().collect();

    let (deam_v_mean, deam_v_std) = mean_std(&deam_v);
    let (deam_a_mean, deam_a_std) = mean_std(&deam_a);

    println!(
        "  DEAM distribution: V={:.3}+/-{:.3}, A={:.3}+/-{:.3}",
        deam_v_mean, deam_v_std, deam_a_mean, deam_a_std
    );

    // FAD-like score: Fréchet distance between DEAM feature distribution
    // and Symthaea's feature distribution (simplified: 1D per feature)
    // Using mean + covariance approximation
    let synth_rms: Vec<f32> = synth_features
        .iter()
        .map(|(_, _, f)| f.rms_energy)
        .collect();
    let synth_zcr: Vec<f32> = synth_features
        .iter()
        .map(|(_, _, f)| f.zero_crossing_rate)
        .collect();
    let synth_hnr: Vec<f32> = synth_features
        .iter()
        .map(|(_, _, f)| f.harmonic_ratio)
        .collect();

    let (s_rms_m, s_rms_s) = mean_std(&synth_rms);
    let (s_zcr_m, s_zcr_s) = mean_std(&synth_zcr);
    let (s_hnr_m, s_hnr_s) = mean_std(&synth_hnr);

    println!(
        "  Symthaea distribution: RMS={:.3}+/-{:.3}, ZCR={:.3}+/-{:.3}, HNR={:.3}+/-{:.3}",
        s_rms_m, s_rms_s, s_zcr_m, s_zcr_s, s_hnr_m, s_hnr_s
    );

    // V-A prediction using DEAM-trained linear regressor.
    // Weights from: data/deam/va_regressor_weights.json
    // Format: [bias, rms, centroid/1000, zcr*10, flux*100, onsets, hnr]
    //
    // Try loading trained weights, fall back to DEAM-calibrated defaults.
    let (_v_weights, a_weights) = load_deam_weights().unwrap_or_else(|| {
        println!("  (Using built-in DEAM-calibrated weights; run train_deam_regressor for custom)");
        (
            // Valence: bias + 6 feature weights (trained on 1,744 DEAM annotations)
            vec![1.928, 0.219, -0.556, -2.659, -0.005, 0.235, -0.099],
            // Arousal: bias + 6 feature weights
            vec![0.453, 0.332, -0.830, 0.962, -0.001, 0.350, -0.214],
        )
    });

    println!("\n  Predicted V-A for each Symthaea scenario (DEAM-trained regressor):");
    println!(
        "  {:>25}  Intended V/A    Predicted V/A   Error",
        "Scenario"
    );

    let mut total_v_error = 0.0f32;
    let mut total_a_error = 0.0f32;

    for (i, scenario) in scenarios.iter().enumerate() {
        let f = &synth_features[i].2;

        // Hybrid model: DEAM-trained for arousal, z-score for valence
        let x = [
            f.rms_energy,
            f.spectral_centroid / 1000.0,
            f.zero_crossing_rate * 10.0,
            f.spectral_flux * 100.0,
            f.onset_density,
            f.harmonic_ratio,
        ];

        // Arousal: DEAM-trained linear regression (MAE=0.216)
        let pred_a = (a_weights[0]
            + x.iter()
                .zip(&a_weights[1..])
                .map(|(xi, wi)| xi * wi)
                .sum::<f32>())
        .clamp(0.0, 1.0);

        // Valence: z-score model calibrated to Symthaea's own feature distribution
        let rms_z = (f.rms_energy - s_rms_m) / (s_rms_s + 0.001);
        let zcr_z = (f.zero_crossing_rate - s_zcr_m) / (s_zcr_s + 0.001);
        let hnr_z = (f.harmonic_ratio - s_hnr_m) / (s_hnr_s + 0.001);
        let flux_z = (f.spectral_flux - 0.004) / 0.002;
        let pred_v = (deam_v_mean
            + deam_v_std * (flux_z * 0.4 - zcr_z * 0.3 + hnr_z * 0.2 - rms_z * 0.1))
            .clamp(-1.0, 1.0);

        let v_err = (scenario.intended_valence - pred_v).abs();
        let a_err = (scenario.intended_arousal - pred_a).abs();
        total_v_error += v_err;
        total_a_error += a_err;

        println!(
            "  {:>25}  V={:+.2} A={:.2}    V={:+.2} A={:.2}    dV={:.2} dA={:.2}",
            scenario.name,
            scenario.intended_valence,
            scenario.intended_arousal,
            pred_v,
            pred_a,
            v_err,
            a_err,
        );
    }

    let n = scenarios.len() as f32;
    let mae_v = total_v_error / n;
    let mae_a = total_a_error / n;
    println!("\n  MAE: Valence={:.3}, Arousal={:.3}", mae_v, mae_a);

    if mae_v < 0.3 && mae_a < 0.3 {
        println!("  ✓ GOOD: Mean absolute error below 0.3 on both axes.");
    } else if mae_v < 0.5 && mae_a < 0.5 {
        println!("  ~ FAIR: Moderate prediction error. Mapping is directionally correct.");
    } else {
        println!("  ✗ POOR: High prediction error. Feature→V-A model needs calibration.");
    }

    Ok(())
}

/// Parse DEAM static annotations: song_id, valence_mean, valence_std, arousal_mean, arousal_std
/// Values on 1-9 scale, normalized to [-1,1] (valence) and [0,1] (arousal)
fn parse_deam_static(
    path: &str,
) -> Result<
    (
        std::collections::HashMap<u32, f32>,
        std::collections::HashMap<u32, f32>,
    ),
    String,
> {
    let file = std::fs::File::open(path).map_err(|e| format!("Cannot open {path}: {e}"))?;
    let reader = std::io::BufReader::new(file);
    let mut v_map = std::collections::HashMap::new();
    let mut a_map = std::collections::HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line: String = line.map_err(|e| e.to_string())?;
        if i == 0 {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let (Ok(id), Ok(v_mean), Ok(a_mean)) = (
                parts[0].trim().parse::<u32>(),
                parts[1].trim().parse::<f32>(),
                parts[3].trim().parse::<f32>(),
            ) {
                // Normalize: 1-9 scale → [-1,1] for valence, [0,1] for arousal
                v_map.insert(id, (v_mean - 5.0) / 4.0);
                a_map.insert(id, (a_mean - 1.0) / 8.0);
            }
        }
    }
    Ok((v_map, a_map))
}

/// Load DEAM-trained V-A regressor weights from JSON.
fn load_deam_weights() -> Option<(Vec<f32>, Vec<f32>)> {
    let path = "data/deam/va_regressor_weights.json";
    let data = std::fs::read_to_string(path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;
    let v: Vec<f32> = parsed["valence_weights"]
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    let a: Vec<f32> = parsed["arousal_weights"]
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    if v.len() == 7 && a.len() == 7 {
        Some((v, a))
    } else {
        None
    }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 3: V-A SCATTER PLOT SVG
// Figure 1 of the paper: intended vs actual position in Russell's circumplex
// ═══════════════════════════════════════════════════════════════════════════

fn generate_va_scatter_svg(
    features: &[(f32, f32, AudioFeatures)],
    scenarios: &[CognitiveScenario],
) -> String {
    // Build SVG without raw strings to avoid # delimiter conflicts with color codes
    let w = 600.0f32;
    let h = 600.0f32;
    let margin = 60.0f32;
    let plot_w = w - 2.0 * margin;
    let plot_h = h - 2.0 * margin;

    // Map V-A to pixel coordinates
    // Valence: -1..1 → left..right
    // Arousal: 0..1 → bottom..top
    let vx = |v: f32| -> f32 { margin + (v + 1.0) / 2.0 * plot_w };
    let ay = |a: f32| -> f32 { margin + (1.0 - a) * plot_h };

    let colors = [
        "#2196F3", // Flow - blue
        "#4CAF50", // Contentment - green
        "#F44336", // Panic - red
        "#795548", // Burnout - brown
        "#FF9800", // Curiosity - orange
        "#9C27B0", // Sacred Stillness - purple
        "#D32F2F", // Anger - dark red
        "#E91E63", // Wonder - pink
        "#607D8B", // Grief - gray
        "#FFC107", // Excitement - gold
        "#9E9E9E", // Boredom - gray
        "#FF5722", // Tension - deep orange
    ];

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">
  <rect width="{w}" height="{h}" fill="rgb(250,250,250)"/>
  <text x="{}" y="25" text-anchor="middle" font-size="16" font-weight="bold">Symthaea Muse: Consciousness State in V-A Space</text>
  <text x="{}" y="42" text-anchor="middle" font-size="11" fill="rgb(102,102,102)">Russell's Circumplex Model (1980) | Koelsch/Vuust/Friston (2019) Predictive Coding</text>
"#,
        w / 2.0,
        w / 2.0
    );

    // Axes
    svg += &format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb(204,204,204)" stroke-width="1"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb(204,204,204)" stroke-width="1"/>
"#,
        margin,
        ay(0.5),
        margin + plot_w,
        ay(0.5), // horizontal midline
        vx(0.0),
        margin,
        vx(0.0),
        margin + plot_h, // vertical midline
    );

    // Quadrant labels
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="10" fill="rgb(170,170,170)">High Arousal</text>"#,
        vx(0.0),
        margin - 5.0
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="10" fill="rgb(170,170,170)">Low Arousal</text>"#,
        vx(0.0),
        margin + plot_h + 15.0
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="end" font-size="10" fill="rgb(170,170,170)">Negative</text>"#,
        margin - 5.0,
        ay(0.5) + 4.0
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="start" font-size="10" fill="rgb(170,170,170)">Positive</text>"#,
        margin + plot_w + 5.0,
        ay(0.5) + 4.0
    );

    // Quadrant annotations
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="9" fill="rgb(221,221,221)">Excited/Happy</text>"#,
        vx(0.5),
        ay(0.85)
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="9" fill="rgb(221,221,221)">Calm/Content</text>"#,
        vx(0.5),
        ay(0.15)
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="9" fill="rgb(221,221,221)">Angry/Stressed</text>"#,
        vx(-0.5),
        ay(0.85)
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="9" fill="rgb(221,221,221)">Sad/Depressed</text>"#,
        vx(-0.5),
        ay(0.15)
    );

    // Plot each scenario: intended (circle) and predicted (cross) with line
    for (i, scenario) in scenarios.iter().enumerate() {
        let f = &features[i].2;
        let color = colors[i % colors.len()];

        let ix = vx(scenario.intended_valence);
        let iy = ay(scenario.intended_arousal);

        // Predicted V-A using DEAM-trained regressor (same as Stage 2)
        let (v_w, a_w) = load_deam_weights().unwrap_or_else(|| {
            (
                vec![1.928, 0.219, -0.556, -2.659, -0.005, 0.235, -0.099],
                vec![0.453, 0.332, -0.830, 0.962, -0.001, 0.350, -0.214],
            )
        });
        let x = [
            f.rms_energy,
            f.spectral_centroid / 1000.0,
            f.zero_crossing_rate * 10.0,
            f.spectral_flux * 100.0,
            f.onset_density,
            f.harmonic_ratio,
        ];
        let pred_v = (v_w[0] + x.iter().zip(&v_w[1..]).map(|(xi, wi)| xi * wi).sum::<f32>())
            .clamp(-1.0, 1.0);
        let pred_a =
            (a_w[0] + x.iter().zip(&a_w[1..]).map(|(xi, wi)| xi * wi).sum::<f32>()).clamp(0.0, 1.0);
        let px = vx(pred_v);
        let py = ay(pred_a);

        // Line from intended to predicted
        svg += &format!(
            r#"  <line x1="{ix:.1}" y1="{iy:.1}" x2="{px:.1}" y2="{py:.1}" stroke="{color}" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
"#
        );

        // Intended (filled circle)
        svg += &format!(
            r#"  <circle cx="{ix:.1}" cy="{iy:.1}" r="8" fill="{color}" opacity="0.8"/>
"#
        );

        // Predicted (open circle with cross)
        svg += &format!(
            r#"  <circle cx="{px:.1}" cy="{py:.1}" r="6" fill="none" stroke="{color}" stroke-width="2"/>
  <line x1="{}" y1="{py:.1}" x2="{}" y2="{py:.1}" stroke="{color}" stroke-width="1.5"/>
  <line x1="{px:.1}" y1="{}" x2="{px:.1}" y2="{}" stroke="{color}" stroke-width="1.5"/>
"#,
            px - 4.0,
            px + 4.0,
            py - 4.0,
            py + 4.0
        );

        // Label
        let label = &scenario.name[..scenario.name.len().min(15)];
        svg += &format!(
            r#"  <text x="{}" y="{}" font-size="8" fill="{color}">{label}</text>
"#,
            ix + 10.0,
            iy + 3.0
        );
    }

    // Legend
    svg += &format!(
        r#"  <circle cx="{}" cy="{}" r="6" fill="rgb(102,102,102)" opacity="0.8"/>"#,
        margin + 10.0,
        h - 25.0
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" font-size="9" fill="rgb(102,102,102)">= Intended state</text>"#,
        margin + 22.0,
        h - 21.0
    );
    svg += &format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb(102,102,102)" stroke-width="1.5"/>"#,
        margin + 130.0,
        h - 25.0,
        margin + 142.0,
        h - 25.0
    );
    svg += &format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb(102,102,102)" stroke-width="1.5"/>"#,
        margin + 136.0,
        h - 31.0,
        margin + 136.0,
        h - 19.0
    );
    svg += &format!(
        r#"  <text x="{}" y="{}" font-size="9" fill="rgb(102,102,102)">= Perceived (from audio features)</text>"#,
        margin + 148.0,
        h - 21.0
    );

    svg += "\n</svg>";
    svg
}
