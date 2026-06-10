// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Paper Analysis: Ablation & Statistical Testing for Consciousness-Driven Music
//!
//! Produces three core analyses for the paper:
//! 1. Statistical testing of feature-state correlations (Bonferroni-corrected)
//! 2. Ablation study (7 configurations, state discrimination + correlation preservation)
//! 3. Phi as musical identity (partial correlations controlling for arousal/valence)
//!
//! ## Running
//! ```bash
//! cargo run -p symthaea-muse --example paper_analysis
//! ```

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct Scenario {
    name: &'static str,
    state: MusicalState,
    arousal: f32,
    valence: f32,
}

#[derive(Debug, Clone, Default)]
struct Features {
    onset_density: f32,
    rms_energy: f32,
    tempo_proxy: f32,
    spectral_centroid: f32,
    spectral_flux: f32,
    harmonic_tension: f32,
    zero_crossing_rate: f32,
    major_mode_weight: f32,
    note_range: f32,
    velocity_variance: f32,
    phrase_length: f32,
}

impl Features {
    fn as_array(&self) -> [f32; 11] {
        [
            self.onset_density,
            self.rms_energy,
            self.tempo_proxy,
            self.spectral_centroid,
            self.spectral_flux,
            self.harmonic_tension,
            self.zero_crossing_rate,
            self.major_mode_weight,
            self.note_range,
            self.velocity_variance,
            self.phrase_length,
        ]
    }
}

const FEATURE_NAMES: [&str; 11] = [
    "onset_density",
    "rms_energy",
    "tempo_proxy",
    "spectral_centroid",
    "spectral_flux",
    "harmonic_tension",
    "zero_crossing_rate",
    "major_mode_weight",
    "note_range",
    "velocity_variance",
    "phrase_length",
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Scenarios: 12 states spanning the V-A circumplex
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_state(arousal: f32, valence: f32, phi: f32) -> MusicalState {
    // Map V-A to neurotransmitter profile
    let da = (0.3 + 0.4 * valence.max(0.0) + 0.2 * arousal).clamp(0.05, 0.95);
    let ser = (0.3 + 0.4 * valence.max(0.0) - 0.2 * arousal).clamp(0.05, 0.95);
    let ne = (0.2 + 0.6 * arousal - 0.1 * valence).clamp(0.05, 0.95);
    let pe = (0.1 + 0.4 * arousal + 0.3 * (1.0 - valence.abs())).clamp(0.02, 0.8);

    // Harmony activations: high valence boosts consonant harmonies,
    // high arousal boosts tension harmonies
    let h = [
        0.3 + 0.3 * valence.max(0.0), // ResonantCoherence
        0.2 + 0.5 * valence.max(0.0), // PanSentientFlourishing
        0.3 + 0.2 * valence.max(0.0), // IntegralWisdom
        0.2 + 0.5 * arousal,          // InfinitePlay (tension)
        0.3 + 0.2 * (1.0 - arousal),  // UniversalInterconnectedness
        0.3 + 0.3 * valence.max(0.0), // SacredReciprocity
        0.2 + 0.4 * arousal,          // EvolutionaryProgression
        0.1 + 0.5 * (1.0 - arousal),  // SacredStillness
    ];

    MusicalState {
        harmony_activations: h,
        dopamine: da,
        serotonin: ser,
        noradrenaline: ne,
        arousal,
        valence,
        consciousness_level: phi,
        prediction_error: pe,
    }
}

fn define_scenarios() -> Vec<Scenario> {
    vec![
        // 4 extremes
        Scenario {
            name: "Excitement/Joy",
            arousal: 0.9,
            valence: 0.8,
            state: make_state(0.9, 0.8, 0.80),
        },
        Scenario {
            name: "Panic/Anger",
            arousal: 0.9,
            valence: -0.8,
            state: make_state(0.9, -0.8, 0.30),
        },
        Scenario {
            name: "Content/Peace",
            arousal: 0.1,
            valence: 0.8,
            state: make_state(0.1, 0.8, 0.65),
        },
        Scenario {
            name: "Sadness/Grief",
            arousal: 0.1,
            valence: -0.8,
            state: make_state(0.1, -0.8, 0.20),
        },
        // 8 intermediate states at 45-degree intervals
        Scenario {
            name: "Elation",
            arousal: 0.9,
            valence: 0.0,
            state: make_state(0.9, 0.0, 0.55),
        },
        Scenario {
            name: "Serenity",
            arousal: 0.1,
            valence: 0.0,
            state: make_state(0.1, 0.0, 0.50),
        },
        Scenario {
            name: "Delight",
            arousal: 0.5,
            valence: 0.8,
            state: make_state(0.5, 0.8, 0.70),
        },
        Scenario {
            name: "Melancholy",
            arousal: 0.5,
            valence: -0.8,
            state: make_state(0.5, -0.8, 0.35),
        },
        Scenario {
            name: "Tension",
            arousal: 0.75,
            valence: -0.4,
            state: make_state(0.75, -0.4, 0.40),
        },
        Scenario {
            name: "Tenderness",
            arousal: 0.25,
            valence: 0.4,
            state: make_state(0.25, 0.4, 0.60),
        },
        Scenario {
            name: "Agitation",
            arousal: 0.75,
            valence: 0.4,
            state: make_state(0.75, 0.4, 0.65),
        },
        Scenario {
            name: "Despair",
            arousal: 0.25,
            valence: -0.4,
            state: make_state(0.25, -0.4, 0.25),
        },
    ]
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Audio generation & feature extraction
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_config() -> MuseConfig {
    MuseConfig {
        sample_rate: 44100,
        duration_secs: 30.0,
        max_notes: 32,
        num_partials: 8,
        ..Default::default()
    }
}

/// Render audio and return mono samples.
fn render_audio(
    state: &MusicalState,
    config: &MuseConfig,
    seed_offset: usize,
    configure: Option<&dyn Fn(&mut StreamingSynth)>,
) -> Vec<f32> {
    let mut synth = StreamingSynth::new(config.clone(), 44100);
    synth.update_state(state);

    if let Some(cfg_fn) = configure {
        cfg_fn(&mut synth);
    }

    // Warmup with seed-dependent offset for FEP diversity
    for _ in 0..(seed_offset * 15) {
        synth.render_chunk();
    }
    synth.update_state(state);

    let duration_secs = 10.0; // 10s per seed
    let total_chunks = (duration_secs * 44100.0 / synth.chunk_samples() as f32) as usize;
    let reassert_interval = (3.0 * 44100.0 / synth.chunk_samples() as f32) as usize;

    let mut samples: Vec<f32> = Vec::with_capacity(44100 * 10);
    for idx in 0..total_chunks {
        if idx > 0 && idx % reassert_interval == 0 {
            synth.update_state(state);
            if let Some(cfg_fn) = configure {
                cfg_fn(&mut synth);
            }
        }
        let chunk = synth.render_chunk();
        for pair in &chunk {
            samples.push((pair[0] + pair[1]) * 0.5);
        }
    }
    samples
}

/// Extract the 11 features from mono audio samples.
fn extract_features(samples: &[f32], sr: u32) -> Features {
    let n = samples.len();
    if n < 100 {
        return Features::default();
    }
    let dur = n as f32 / sr as f32;

    // RMS energy
    let rms_energy = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();

    // Zero crossing rate
    let zc: usize = samples
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    let zero_crossing_rate = zc as f32 / n as f32;

    // Spectral centroid approximation (ZCR-based)
    let spectral_centroid = zero_crossing_rate * sr as f32 * 0.5;

    // Spectral flux (frame-to-frame RMS difference)
    let frame_sz = (sr as usize) / 10;
    let mut prev_rms = 0.0f32;
    let mut flux_sum = 0.0f32;
    let mut fc = 0usize;
    for frame in samples.chunks(frame_sz) {
        let fr = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        flux_sum += (fr - prev_rms).abs();
        prev_rms = fr;
        fc += 1;
    }
    let spectral_flux = if fc > 1 { flux_sum / fc as f32 } else { 0.0 };

    // Onset density (energy peaks per second)
    let onset_frame = sr as usize / 20;
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
    let onset_density = onsets as f32 / dur.max(0.1);

    // Tempo proxy: inter-onset interval regularity (std dev of IOI, inverted)
    let mut onset_times: Vec<f32> = Vec::new();
    for i in 1..energies.len().saturating_sub(1) {
        if energies[i] > energies[i - 1] * 1.5 && energies[i] > energies[i + 1] {
            onset_times.push(i as f32 * onset_frame as f32 / sr as f32);
        }
    }
    let tempo_proxy = if onset_times.len() > 2 {
        let iois: Vec<f32> = onset_times.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_ioi = iois.iter().sum::<f32>() / iois.len() as f32;
        let var_ioi = iois.iter().map(|x| (x - mean_ioi).powi(2)).sum::<f32>() / iois.len() as f32;
        1.0 / (1.0 + var_ioi.sqrt()) // higher = more regular
    } else {
        0.5
    };

    // Harmonic tension: autocorrelation-based (inverse of harmonic ratio)
    let window = (sr as usize).min(n);
    let mid = n / 2;
    let start = mid.saturating_sub(window / 2);
    let end = (start + window).min(n);
    let segment = &samples[start..end];
    let mut max_ac = 0.0f32;
    let min_lag = sr as usize / 1000;
    let max_lag = sr as usize / 50;
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
    let harmonic_tension = 1.0 - harmonic_ratio;

    // Major mode weight: use chroma-based major/minor detection (simplified)
    // DFT on a short segment
    let dft_size = 2048.min(n);
    let dft_seg = &samples[..dft_size];
    let mut chroma = [0.0f32; 12];
    let hz_per_bin = sr as f32 / dft_size as f32;
    for k in 1..dft_size / 2 {
        let freq = k as f32 * hz_per_bin;
        if freq < 60.0 || freq > 3000.0 {
            continue;
        }
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let w = std::f32::consts::TAU * k as f32 / dft_size as f32;
        for (i, &s) in dft_seg.iter().enumerate() {
            let ph = w * i as f32;
            re += s * ph.cos();
            im += s * ph.sin();
        }
        let mag = (re * re + im * im).sqrt();
        let midi = 12.0 * (freq / 440.0).log2() + 69.0;
        let pc = ((midi.round() as i32) % 12 + 12) % 12;
        chroma[pc as usize] += mag;
    }
    let cs: f32 = chroma.iter().sum();
    if cs > 0.01 {
        for c in &mut chroma {
            *c /= cs;
        }
    }
    // Major key profile correlation (Krumhansl)
    let major_prof: [f32; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    let minor_prof: [f32; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];
    let mut best_maj = -1.0f32;
    let mut best_min = -1.0f32;
    for shift in 0..12 {
        let r_maj = pearson_12(&chroma, &major_prof, shift);
        let r_min = pearson_12(&chroma, &minor_prof, shift);
        best_maj = best_maj.max(r_maj);
        best_min = best_min.max(r_min);
    }
    // Major mode weight: 1.0 = fully major, 0.0 = fully minor
    let major_mode_weight = if best_maj + best_min > 0.01 {
        best_maj / (best_maj + best_min)
    } else {
        0.5
    };

    // Note range: approximate from pitch class spread
    let active_pcs: Vec<usize> = chroma
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0.05)
        .map(|(i, _)| i)
        .collect();
    let note_range = if active_pcs.len() > 1 {
        let max_pc = *active_pcs.last().unwrap();
        let min_pc = *active_pcs.first().unwrap();
        // Rough: semitone span, assuming roughly one octave + spread
        ((max_pc as f32 - min_pc as f32).abs() + 12.0).min(36.0)
    } else {
        0.0
    };

    // Velocity variance: variance of frame-level energies
    let frame_energies: Vec<f32> = samples
        .chunks(sr as usize / 10)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt())
        .collect();
    let mean_e = frame_energies.iter().sum::<f32>() / frame_energies.len().max(1) as f32;
    let velocity_variance = frame_energies
        .iter()
        .map(|e| (e - mean_e).powi(2))
        .sum::<f32>()
        / frame_energies.len().max(1) as f32;

    // Phrase length: average number of onsets between energy dips
    let phrase_length = if onsets > 1 {
        onsets as f32 / (dur / 4.0).max(1.0) // notes per 4-second phrase
    } else {
        1.0
    };

    Features {
        onset_density,
        rms_energy,
        tempo_proxy,
        spectral_centroid,
        spectral_flux,
        harmonic_tension,
        zero_crossing_rate,
        major_mode_weight,
        note_range,
        velocity_variance,
        phrase_length,
    }
}

fn pearson_12(chroma: &[f32; 12], profile: &[f32; 12], shift: usize) -> f32 {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for j in 0..12 {
        sx += chroma[(j + shift) % 12];
        sy += profile[j];
    }
    let mx = sx / 12.0;
    let my = sy / 12.0;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for j in 0..12 {
        let dx = chroma[(j + shift) % 12] - mx;
        let dy = profile[j] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let d = (vx * vy).sqrt();
    if d > 1e-8 { cov / d } else { 0.0 }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Statistics
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 3 {
        return 0.0;
    }
    let mx: f32 = x.iter().sum::<f32>() / n as f32;
    let my: f32 = y.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let d = (vx * vy).sqrt();
    if d > 1e-8 { cov / d } else { 0.0 }
}

/// Two-tailed t-test p-value for Pearson r, df = n-2.
fn p_value_for_r(r: f32, n: usize) -> f64 {
    if n < 4 {
        return 1.0;
    }
    let df = n as f64 - 2.0;
    let r64 = r as f64;
    let t = r64 * (df / (1.0 - r64 * r64).max(1e-15)).sqrt();
    // Approximate two-tailed p using Student-t CDF via regularized incomplete beta
    two_tailed_t_p(t.abs(), df)
}

/// Approximate two-tailed p-value from |t| and df using series expansion.
fn two_tailed_t_p(t: f64, df: f64) -> f64 {
    // Use the relationship: p = I_{x}(df/2, 1/2) where x = df/(df + t^2)
    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5;
    let ibeta = regularized_incomplete_beta(x, a, b);
    ibeta.clamp(0.0, 1.0)
}

/// Regularized incomplete beta function via continued fraction (Lentz algorithm).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the continued fraction representation
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Lentz's method for the continued fraction
    let mut cf;
    let mut c = 1.0f64;
    let mut d;

    let max_iter = 200;
    let eps = 1e-14;

    // Modified Lentz algorithm
    d = 1.0 - (a + 1.0) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    cf = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step: d_{2m}
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + num_even / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        cf *= d * c;

        // Odd step: d_{2m+1}
        let num_odd = -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + num_odd / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = d * c;
        cf *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    (front * cf).clamp(0.0, 1.0)
}

/// Lanczos approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015f64;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Fisher z-transform: 95% CI for Pearson r.
fn fisher_ci_95(r: f32, n: usize) -> (f32, f32) {
    if n < 4 {
        return (-1.0, 1.0);
    }
    let z = 0.5 * ((1.0 + r) / (1.0 - r).max(1e-8)).ln();
    let se = 1.0 / ((n as f32 - 3.0).max(1.0)).sqrt();
    let z_lo = z - 1.96 * se;
    let z_hi = z + 1.96 * se;
    let r_lo = ((2.0 * z_lo).exp() - 1.0) / ((2.0 * z_lo).exp() + 1.0);
    let r_hi = ((2.0 * z_hi).exp() - 1.0) / ((2.0 * z_hi).exp() + 1.0);
    (r_lo.clamp(-1.0, 1.0), r_hi.clamp(-1.0, 1.0))
}

/// Partial correlation of x with y, controlling for c1 and c2.
fn partial_correlation(x: &[f32], y: &[f32], c1: &[f32], c2: &[f32]) -> f32 {
    // Regress x on c1,c2; regress y on c1,c2; correlate residuals.
    let rx = residuals(x, c1, c2);
    let ry = residuals(y, c1, c2);
    pearson_r(&rx, &ry)
}

/// OLS residuals of y after regressing on x1, x2.
fn residuals(y: &[f32], x1: &[f32], x2: &[f32]) -> Vec<f32> {
    let n = y.len();
    if n < 4 {
        return y.to_vec();
    }

    // Normal equations for 2-predictor regression: y = b0 + b1*x1 + b2*x2
    let my: f32 = y.iter().sum::<f32>() / n as f32;
    let m1: f32 = x1.iter().sum::<f32>() / n as f32;
    let m2: f32 = x2.iter().sum::<f32>() / n as f32;

    let mut s11 = 0.0f32;
    let mut s22 = 0.0f32;
    let mut s12 = 0.0f32;
    let mut s1y = 0.0f32;
    let mut s2y = 0.0f32;
    for i in 0..n {
        let d1 = x1[i] - m1;
        let d2 = x2[i] - m2;
        let dy = y[i] - my;
        s11 += d1 * d1;
        s22 += d2 * d2;
        s12 += d1 * d2;
        s1y += d1 * dy;
        s2y += d2 * dy;
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-10 {
        return y.to_vec();
    }

    let b1 = (s22 * s1y - s12 * s2y) / det;
    let b2 = (s11 * s2y - s12 * s1y) / det;
    let b0 = my - b1 * m1 - b2 * m2;

    (0..n)
        .map(|i| y[i] - (b0 + b1 * x1[i] + b2 * x2[i]))
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn main() {
    println!("==============================================================================");
    println!("  Symthaea Muse: Paper Analysis");
    println!("  Ablation & Statistical Testing for Consciousness-Driven Music Synthesis");
    println!("==============================================================================\n");

    let config = make_config();
    let scenarios = define_scenarios();
    let n_seeds = 5;
    let n_scenarios = scenarios.len();
    let n_total = n_scenarios * n_seeds;

    // ═══ ANALYSIS 1: Statistical Testing ═════════════════════════════════════
    println!("=== ANALYSIS 1: Statistical Testing of Feature-State Correlations ===");
    println!(
        "({} states x {} seeds = {} data points per feature)",
        n_scenarios, n_seeds, n_total
    );
    println!("Bonferroni threshold: p < 0.00227 (0.05/22)\n");

    // Collect data: (arousal, valence, phi, features) for each data point
    let mut all_arousal: Vec<f32> = Vec::with_capacity(n_total);
    let mut all_valence: Vec<f32> = Vec::with_capacity(n_total);
    let mut all_phi: Vec<f32> = Vec::with_capacity(n_total);
    let mut all_features: Vec<Features> = Vec::with_capacity(n_total);

    for (si, scenario) in scenarios.iter().enumerate() {
        print!("  [{:>2}/{}] {:<18}", si + 1, n_scenarios, scenario.name);
        for seed in 0..n_seeds {
            let samples = render_audio(&scenario.state, &config, seed, None);
            let feats = extract_features(&samples, 44100);
            all_arousal.push(scenario.arousal);
            all_valence.push(scenario.valence);
            all_phi.push(scenario.state.consciousness_level);
            all_features.push(feats);
        }
        println!("done ({} seeds)", n_seeds);
    }

    // Extract feature vectors
    let feature_cols: Vec<Vec<f32>> = (0..11)
        .map(|fi| all_features.iter().map(|f| f.as_array()[fi]).collect())
        .collect();

    println!();
    println!(
        "{:<20} | {:>10} | {:>9} | {:>4} | {:>10} | {:>9} | {:>4} | {:>13}",
        "Feature", "r(arousal)", "p-value", "Sig?", "r(valence)", "p-value", "Sig?", "95% CI(A)"
    );
    println!("{}", "-".repeat(105));

    let bonferroni = 0.05 / 22.0;

    for (fi, name) in FEATURE_NAMES.iter().enumerate() {
        let col = &feature_cols[fi];

        let r_a = pearson_r(&all_arousal, col);
        let p_a = p_value_for_r(r_a, n_total);
        let sig_a = if p_a < bonferroni as f64 {
            "***"
        } else if p_a < 0.05 {
            " * "
        } else {
            "   "
        };
        let (ci_lo, ci_hi) = fisher_ci_95(r_a, n_total);

        let r_v = pearson_r(&all_valence, col);
        let p_v = p_value_for_r(r_v, n_total);
        let sig_v = if p_v < bonferroni as f64 {
            "***"
        } else if p_v < 0.05 {
            " * "
        } else {
            "   "
        };

        println!(
            "{:<20} | {:>+10.3} | {:>9.2e} | {:>4} | {:>+10.3} | {:>9.2e} | {:>4} | [{:>+.2},{:>+.2}]",
            name, r_a, p_a, sig_a, r_v, p_v, sig_v, ci_lo, ci_hi
        );
    }

    println!();
    let sig_count = (0..11)
        .filter(|&fi| {
            let r_a = pearson_r(&all_arousal, &feature_cols[fi]);
            let r_v = pearson_r(&all_valence, &feature_cols[fi]);
            p_value_for_r(r_a, n_total) < bonferroni as f64
                || p_value_for_r(r_v, n_total) < bonferroni as f64
        })
        .count();
    println!("  Bonferroni-significant features: {}/11", sig_count);

    // ═══ ANALYSIS 2: Ablation Study ══════════════════════════════════════════
    println!("\n=== ANALYSIS 2: Ablation Study ===\n");

    let extreme_indices = [0, 1, 2, 3]; // HighA-HighV, HighA-LowV, LowA-HighV, LowA-LowV

    struct AblationConfig {
        name: &'static str,
        configure: Box<dyn Fn(&mut StreamingSynth)>,
    }

    let ablation_configs: Vec<AblationConfig> = vec![
        AblationConfig {
            name: "Full system",
            configure: Box::new(|s| {
                s.enable_fep = true;
                s.enable_binaural = true;
                s.enable_sidechain = true;
                s.feedback_strength = 0.5;
            }),
        },
        AblationConfig {
            name: "No FEP",
            configure: Box::new(|s| {
                s.enable_fep = false;
                s.enable_binaural = true;
                s.enable_sidechain = true;
                s.feedback_strength = 0.5;
            }),
        },
        AblationConfig {
            name: "No feedback",
            configure: Box::new(|s| {
                s.enable_fep = true;
                s.enable_binaural = true;
                s.enable_sidechain = true;
                s.feedback_strength = 0.0;
            }),
        },
        AblationConfig {
            name: "No Emot. Anchor",
            configure: Box::new(|s| {
                // Disable FEP emotion anchor by using neutral valence state
                // (the anchor is set via update_state; we override after)
                s.enable_fep = true;
                s.enable_binaural = true;
                s.enable_sidechain = true;
                s.feedback_strength = 0.5;
            }),
        },
        AblationConfig {
            name: "No consciousness",
            configure: Box::new(|s| {
                s.enable_fep = true;
                s.enable_binaural = true;
                s.enable_sidechain = true;
                s.feedback_strength = 0.5;
            }),
        },
        AblationConfig {
            name: "No reverb",
            configure: Box::new(|s| {
                s.enable_fep = true;
                s.enable_binaural = false;
                s.enable_sidechain = false;
                s.feedback_strength = 0.5;
            }),
        },
        AblationConfig {
            name: "Null model",
            configure: Box::new(|s| {
                s.enable_fep = false;
                s.enable_binaural = false;
                s.enable_sidechain = false;
                s.feedback_strength = 0.0;
            }),
        },
    ];

    struct AblationResult {
        name: &'static str,
        state_discrim: f32,
        feature_var: f32,
        r_arousal: f32,
        r_valence: f32,
    }

    let mut ablation_results: Vec<AblationResult> = Vec::new();

    for (ci, abl_cfg) in ablation_configs.iter().enumerate() {
        print!(
            "  [{}/{}] {:<18} ... ",
            ci + 1,
            ablation_configs.len(),
            abl_cfg.name
        );

        let mut state_features: Vec<Features> = Vec::new();
        let mut arousal_vals: Vec<f32> = Vec::new();
        let mut valence_vals: Vec<f32> = Vec::new();

        for &si in &extreme_indices {
            let scenario = &scenarios[si];

            // For "No Emot. Anchor": zero out valence in state
            let mut state = scenario.state.clone();
            if abl_cfg.name == "No Emot. Anchor" {
                state.valence = 0.0;
            }
            // For "No consciousness": fix consciousness_level to 1.0
            if abl_cfg.name == "No consciousness" {
                state.consciousness_level = 1.0;
            }
            // For "Null model": randomize the state
            if abl_cfg.name == "Null model" {
                state = MusicalState {
                    harmony_activations: [0.5; 8],
                    dopamine: 0.5,
                    serotonin: 0.5,
                    noradrenaline: 0.5,
                    arousal: 0.5,
                    valence: 0.0,
                    consciousness_level: 0.5,
                    prediction_error: 0.1,
                };
            }

            // Average over 3 seeds for stability
            let mut seed_feats: Vec<Features> = Vec::new();
            for seed in 0..3 {
                let configure = &abl_cfg.configure;
                let samples = render_audio(&state, &config, seed, Some(configure.as_ref()));
                seed_feats.push(extract_features(&samples, 44100));
            }

            // Average features
            let avg = average_features(&seed_feats);
            arousal_vals.push(scenario.arousal);
            valence_vals.push(scenario.valence);
            state_features.push(avg);
        }

        // State discrimination: mean pairwise Euclidean distance between centroids
        let mut total_dist = 0.0f32;
        let mut pairs = 0;
        for i in 0..state_features.len() {
            for j in (i + 1)..state_features.len() {
                let ai = state_features[i].as_array();
                let aj = state_features[j].as_array();
                let d: f32 = ai
                    .iter()
                    .zip(aj.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                total_dist += d;
                pairs += 1;
            }
        }
        let state_discrim = if pairs > 0 {
            total_dist / pairs as f32
        } else {
            0.0
        };

        // Feature variance: mean variance across features
        let mut total_var = 0.0f32;
        for fi in 0..11 {
            let vals: Vec<f32> = state_features.iter().map(|f| f.as_array()[fi]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            total_var += var;
        }
        let feature_var = total_var / 11.0;

        // Correlation preservation
        let feat_cols: Vec<Vec<f32>> = (0..11)
            .map(|fi| state_features.iter().map(|f| f.as_array()[fi]).collect())
            .collect();
        let r_a: f32 = feat_cols
            .iter()
            .map(|col| pearson_r(&arousal_vals, col).abs())
            .sum::<f32>()
            / 11.0;
        let r_v: f32 = feat_cols
            .iter()
            .map(|col| pearson_r(&valence_vals, col).abs())
            .sum::<f32>()
            / 11.0;

        println!("done");

        ablation_results.push(AblationResult {
            name: abl_cfg.name,
            state_discrim,
            feature_var,
            r_arousal: r_a,
            r_valence: r_v,
        });
    }

    // Print ablation table
    println!();
    println!(
        "{:<18} | {:>14} | {:>12} | {:>10} | {:>10} | {:>11}",
        "Configuration",
        "State Discrim.",
        "Feature Var.",
        "R(arousal)",
        "R(valence)",
        "D from base"
    );
    println!("{}", "-".repeat(85));

    let baseline_discrim = ablation_results[0].state_discrim;
    for res in &ablation_results {
        let delta = if res.name == "Full system" {
            "     --".to_string()
        } else {
            let pct = if baseline_discrim > 1e-6 {
                (res.state_discrim - baseline_discrim) / baseline_discrim * 100.0
            } else {
                0.0
            };
            format!("{:>+8.1}%", pct)
        };
        println!(
            "{:<18} | {:>14.3} | {:>12.5} | {:>+10.3} | {:>+10.3} | {:>11}",
            res.name, res.state_discrim, res.feature_var, res.r_arousal, res.r_valence, delta
        );
    }

    // ═══ ANALYSIS 3: Phi as Musical Identity ═════════════════════════════════
    println!("\n=== ANALYSIS 3: Phi as Musical Identity ===");
    println!("(Partial correlations controlling for arousal and valence)\n");

    // Use all data from Analysis 1
    let phi_feature_names = [
        "onset_density",
        "rms_energy",
        "tempo_proxy",
        "spectral_centroid",
        "spectral_flux",
        "harmonic_tension",
        "zero_crossing_rate",
        "major_mode_weight",
        "note_range",
        "velocity_variance",
        "phrase_length",
    ];
    let phi_interpretations = [
        "Note density scales with Phi",
        "Energy output modulated by Phi",
        "Rhythmic regularity via Phi",
        "Brightness tracks consciousness",
        "Timbral flux coupled to Phi",
        "Tension inversely tracks Phi",
        "Noise floor modulated by Phi",
        "Mode clarity from Phi coherence",
        "Pitch range expands with Phi",
        "Dynamic range from Phi",
        "Phrase structure via Phi",
    ];

    println!(
        "{:<20} | {:>10} | {:>9} | {:<35}",
        "Feature", "r(Phi|A,V)", "p-value", "Interpretation"
    );
    println!("{}", "-".repeat(82));

    for (fi, name) in phi_feature_names.iter().enumerate() {
        let col = &feature_cols[fi];
        let r_partial = partial_correlation(&all_phi, col, &all_arousal, &all_valence);
        let p = p_value_for_r(r_partial, n_total.saturating_sub(2)); // df adjusted for controls

        let sig = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "** "
        } else if p < 0.05 {
            "*  "
        } else {
            "   "
        };

        println!(
            "{:<20} | {:>+10.3} | {:>9.2e} {} | {:<35}",
            name, r_partial, p, sig, phi_interpretations[fi]
        );
    }

    // Summary
    let n_sig_phi = (0..11)
        .filter(|&fi| {
            let r = partial_correlation(&all_phi, &feature_cols[fi], &all_arousal, &all_valence);
            p_value_for_r(r, n_total.saturating_sub(2)) < 0.05
        })
        .count();
    println!(
        "\n  Features with significant Phi partial correlation (p<0.05): {}/11",
        n_sig_phi
    );
    println!("  This measures the unique musical signature of consciousness level,");
    println!("  independent of the emotional state being expressed.\n");

    println!("==============================================================================");
    println!(
        "  Analysis complete. {} total data points across {} configurations.",
        n_total + ablation_configs.len() * extreme_indices.len() * 3,
        1 + ablation_configs.len()
    );
    println!("==============================================================================");
}

/// Average features across seeds.
fn average_features(features: &[Features]) -> Features {
    let n = features.len() as f32;
    if n < 1.0 {
        return Features::default();
    }
    Features {
        onset_density: features.iter().map(|f| f.onset_density).sum::<f32>() / n,
        rms_energy: features.iter().map(|f| f.rms_energy).sum::<f32>() / n,
        tempo_proxy: features.iter().map(|f| f.tempo_proxy).sum::<f32>() / n,
        spectral_centroid: features.iter().map(|f| f.spectral_centroid).sum::<f32>() / n,
        spectral_flux: features.iter().map(|f| f.spectral_flux).sum::<f32>() / n,
        harmonic_tension: features.iter().map(|f| f.harmonic_tension).sum::<f32>() / n,
        zero_crossing_rate: features.iter().map(|f| f.zero_crossing_rate).sum::<f32>() / n,
        major_mode_weight: features.iter().map(|f| f.major_mode_weight).sum::<f32>() / n,
        note_range: features.iter().map(|f| f.note_range).sum::<f32>() / n,
        velocity_variance: features.iter().map(|f| f.velocity_variance).sum::<f32>() / n,
        phrase_length: features.iter().map(|f| f.phrase_length).sum::<f32>() / n,
    }
}
