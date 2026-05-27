// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Biosignal Pattern Recognition with Cincinnati-LTC
//!
//! Demonstrates applying Cincinnati-LTC to real-world-like biosignal patterns:
//! - EEG-like alpha/beta/gamma rhythms
//! - Heart rate variability (HRV) patterns
//! - Respiratory cycles
//! - Audio waveform patterns (speech/music)
//!
//! Uses synthetic signals that mimic real biosignal characteristics.

use std::f64::consts::PI;
use symthaea::hdc::cincinnati_ltc::CincinnatiLtcEngine;
use symthaea::hdc::unified_hv::ContinuousHV;

// Use smaller dimension for faster execution
const SIGNAL_DIM: usize = 1024;

/// Biosignal generator trait
trait BiosignalGenerator {
    fn sample(&mut self) -> f64;
    #[allow(dead_code)]
    fn name(&self) -> &str;
    #[allow(dead_code)]
    fn description(&self) -> &str;
    fn expected_frequency(&self) -> f64;
}

// =============================================================================
// EEG-LIKE SIGNALS
// =============================================================================

/// Alpha rhythm (8-12 Hz) - relaxed, closed eyes
struct AlphaRhythm {
    t: f64,
    dt: f64,
    freq: f64,
    noise_level: f64,
}

impl AlphaRhythm {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            freq: 10.0, // 10 Hz center frequency
            noise_level: 0.2,
        }
    }
}

impl BiosignalGenerator for AlphaRhythm {
    fn sample(&mut self) -> f64 {
        let base = (2.0 * PI * self.freq * self.t).sin();
        // Add harmonics and noise
        let harmonic = 0.3 * (2.0 * PI * self.freq * 2.0 * self.t).sin();
        let noise = self.noise_level * (self.t * 12345.67).sin() * (self.t * 7654.32).cos();
        self.t += self.dt;
        base + harmonic + noise
    }
    fn name(&self) -> &str {
        "EEG Alpha"
    }
    fn description(&self) -> &str {
        "8-12 Hz alpha rhythm (relaxed state)"
    }
    fn expected_frequency(&self) -> f64 {
        self.freq
    }
}

/// Beta rhythm (12-30 Hz) - active, focused
struct BetaRhythm {
    t: f64,
    dt: f64,
    freq: f64,
}

impl BetaRhythm {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            freq: 20.0,
        }
    }
}

impl BiosignalGenerator for BetaRhythm {
    fn sample(&mut self) -> f64 {
        let base = (2.0 * PI * self.freq * self.t).sin();
        let modulation = 1.0 + 0.3 * (2.0 * PI * 0.5 * self.t).sin(); // Slow modulation
        self.t += self.dt;
        base * modulation
    }
    fn name(&self) -> &str {
        "EEG Beta"
    }
    fn description(&self) -> &str {
        "12-30 Hz beta rhythm (focused state)"
    }
    fn expected_frequency(&self) -> f64 {
        self.freq
    }
}

/// Gamma rhythm (30-100 Hz) - cognitive processing
struct GammaRhythm {
    t: f64,
    dt: f64,
    freq: f64,
}

impl GammaRhythm {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            freq: 40.0,
        }
    }
}

impl BiosignalGenerator for GammaRhythm {
    fn sample(&mut self) -> f64 {
        // Gamma bursts with theta modulation
        let gamma = (2.0 * PI * self.freq * self.t).sin();
        let theta_envelope = (2.0 * PI * 6.0 * self.t).sin().max(0.0);
        self.t += self.dt;
        gamma * (0.3 + 0.7 * theta_envelope)
    }
    fn name(&self) -> &str {
        "EEG Gamma"
    }
    fn description(&self) -> &str {
        "30-100 Hz gamma bursts (cognitive processing)"
    }
    fn expected_frequency(&self) -> f64 {
        self.freq
    }
}

// =============================================================================
// CARDIAC SIGNALS
// =============================================================================

/// Heart Rate Variability - irregular but patterned
struct HeartRateVariability {
    t: f64,
    dt: f64,
    base_rate: f64, // ~1 Hz = 60 bpm
    respiratory_sinus_arrhythmia: f64,
}

impl HeartRateVariability {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            base_rate: 1.2,                     // 72 bpm
            respiratory_sinus_arrhythmia: 0.15, // 15% variation with breathing
        }
    }
}

impl BiosignalGenerator for HeartRateVariability {
    fn sample(&mut self) -> f64 {
        // Heart rate modulated by respiration (~0.25 Hz = 15 breaths/min)
        let respiratory_mod = self.respiratory_sinus_arrhythmia * (2.0 * PI * 0.25 * self.t).sin();
        let instantaneous_rate = self.base_rate * (1.0 + respiratory_mod);

        // Simulate R-wave peaks
        let phase = (2.0 * PI * instantaneous_rate * self.t) % (2.0 * PI);
        let r_wave = if phase < 0.1 * PI {
            (phase / (0.1 * PI) * PI).sin()
        } else {
            0.0
        };

        self.t += self.dt;
        r_wave
    }
    fn name(&self) -> &str {
        "HRV"
    }
    fn description(&self) -> &str {
        "Heart rate variability with respiratory modulation"
    }
    fn expected_frequency(&self) -> f64 {
        self.base_rate
    }
}

// =============================================================================
// RESPIRATORY SIGNALS
// =============================================================================

/// Respiratory pattern with natural irregularity
struct RespiratoryPattern {
    t: f64,
    dt: f64,
    breath_rate: f64,
}

impl RespiratoryPattern {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            breath_rate: 0.25, // 15 breaths/min
        }
    }
}

impl BiosignalGenerator for RespiratoryPattern {
    fn sample(&mut self) -> f64 {
        // Asymmetric breathing: faster exhale than inhale
        let phase = (2.0 * PI * self.breath_rate * self.t) % (2.0 * PI);
        let breath = if phase < PI * 1.2 {
            // Inhale (slower)
            (phase / 1.2).sin()
        } else {
            // Exhale (faster)
            -((phase - PI * 1.2) / 0.8 * PI).sin()
        };

        // Add slight irregularity
        let irregularity = 0.05 * (self.t * 0.1).sin();

        self.t += self.dt;
        breath + irregularity
    }
    fn name(&self) -> &str {
        "Respiratory"
    }
    fn description(&self) -> &str {
        "Breathing pattern with natural asymmetry"
    }
    fn expected_frequency(&self) -> f64 {
        self.breath_rate
    }
}

// =============================================================================
// AUDIO PATTERNS
// =============================================================================

/// Speech-like formant pattern
struct SpeechFormant {
    t: f64,
    dt: f64,
    f1: f64, // First formant
    f2: f64, // Second formant
    pitch: f64,
}

impl SpeechFormant {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            f1: 500.0,    // Vowel "a" first formant
            f2: 1500.0,   // Second formant
            pitch: 150.0, // Fundamental frequency
        }
    }
}

impl BiosignalGenerator for SpeechFormant {
    fn sample(&mut self) -> f64 {
        // Glottal pulse train
        let glottal = (2.0 * PI * self.pitch * self.t).sin();

        // Formant filters (simplified)
        let formant1 = 0.5 * (2.0 * PI * self.f1 * self.t).sin();
        let formant2 = 0.3 * (2.0 * PI * self.f2 * self.t).sin();

        self.t += self.dt;
        glottal * (0.5 + formant1 + formant2)
    }
    fn name(&self) -> &str {
        "Speech"
    }
    fn description(&self) -> &str {
        "Speech-like formant pattern"
    }
    fn expected_frequency(&self) -> f64 {
        self.pitch
    }
}

/// Music-like harmonic pattern
struct MusicHarmonic {
    t: f64,
    dt: f64,
    fundamental: f64,
}

impl MusicHarmonic {
    fn new(sample_rate: f64) -> Self {
        Self {
            t: 0.0,
            dt: 1.0 / sample_rate,
            fundamental: 440.0, // A4 note
        }
    }
}

impl BiosignalGenerator for MusicHarmonic {
    fn sample(&mut self) -> f64 {
        // Rich harmonic series (like a string instrument)
        let f = self.fundamental;
        let mut sum = 0.0;
        for n in 1..=8 {
            let amplitude = 1.0 / (n as f64);
            sum += amplitude * (2.0 * PI * f * n as f64 * self.t).sin();
        }

        // Envelope (attack-decay-sustain)
        let envelope = (-(self.t % 0.5) * 4.0).exp() * 0.8 + 0.2;

        self.t += self.dt;
        sum * envelope / 2.0
    }
    fn name(&self) -> &str {
        "Music"
    }
    fn description(&self) -> &str {
        "Harmonic series (musical tone)"
    }
    fn expected_frequency(&self) -> f64 {
        self.fundamental
    }
}

// =============================================================================
// ANALYSIS ENGINE
// =============================================================================

/// Run pattern recognition on a biosignal
fn analyze_biosignal(
    signal: &mut dyn BiosignalGenerator,
    sample_rate: f64,
    duration_sec: f64,
    threshold: f64,
) -> BiosignalResults {
    let num_samples = (sample_rate * duration_sec) as usize;

    let mut engine = CincinnatiLtcEngine::new(5);
    engine.set_budding_threshold(0.5);
    engine.set_sustain_steps(3);

    let mut correct = 0;
    let mut total = 0;
    let mut prev_above = false;
    let mut crossings = 0;

    // Track node states for budding
    let mut node_states: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(SIGNAL_DIM, i as u64 * 1000))
        .collect();

    for i in 0..num_samples {
        let sample = signal.sample();
        let above_threshold = sample > threshold;

        // Count zero crossings for frequency estimation
        if above_threshold != prev_above {
            crossings += 1;
        }
        prev_above = above_threshold;

        // Make prediction
        let (prediction, _) = engine.predict();

        // Track accuracy (skip warmup)
        if i >= 50 {
            if prediction == above_threshold {
                correct += 1;
            }
            total += 1;
        }

        // Step engine
        let input_hv = ContinuousHV::random(SIGNAL_DIM, i as u64);
        engine.step(above_threshold, &input_hv);

        // Update prediction error for budding
        let node_count = engine.node_count();
        while node_states.len() < node_count {
            node_states.push(ContinuousHV::random(
                SIGNAL_DIM,
                (node_states.len() * 1000 + i) as u64,
            ));
        }

        for node_id in 0..node_count {
            let expected =
                ContinuousHV::random(SIGNAL_DIM, if prediction { 111111 } else { 222222 });
            let actual =
                ContinuousHV::random(SIGNAL_DIM, if above_threshold { 111111 } else { 222222 });
            engine.update_prediction_error(node_id, &expected, &actual);
        }

        // Process budding
        let _ = engine.process_budding(&node_states[..node_count], i as f64 / sample_rate);
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.5
    };
    let estimated_freq = (crossings as f64 / 2.0) / duration_sec;

    BiosignalResults {
        accuracy,
        final_nodes: engine.node_count(),
        estimated_frequency: estimated_freq,
        expected_frequency: signal.expected_frequency(),
    }
}

#[derive(Debug)]
struct BiosignalResults {
    accuracy: f64,
    final_nodes: usize,
    estimated_frequency: f64,
    expected_frequency: f64,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          BIOSIGNAL PATTERN RECOGNITION with Cincinnati-LTC          ║");
    println!("║     Demonstrating real-world-like signal processing capabilities     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let sample_rate = 250.0; // Hz (common for EEG)
    let duration = 4.0; // seconds
    let threshold = 0.0; // Zero-crossing threshold

    println!("Configuration:");
    println!("  Sample Rate:  {} Hz", sample_rate);
    println!("  Duration:     {} seconds", duration);
    println!("  Samples:      {}", (sample_rate * duration) as usize);
    println!();

    // Create all biosignal generators
    let mut signals: Vec<(&str, Box<dyn BiosignalGenerator>)> = vec![
        ("EEG Alpha (10 Hz)", Box::new(AlphaRhythm::new(sample_rate))),
        ("EEG Beta (20 Hz)", Box::new(BetaRhythm::new(sample_rate))),
        ("EEG Gamma (40 Hz)", Box::new(GammaRhythm::new(sample_rate))),
        (
            "Heart Rate Variability",
            Box::new(HeartRateVariability::new(sample_rate)),
        ),
        (
            "Respiratory Pattern",
            Box::new(RespiratoryPattern::new(sample_rate)),
        ),
        ("Speech Formant", Box::new(SpeechFormant::new(sample_rate))),
        ("Musical Tone", Box::new(MusicHarmonic::new(sample_rate))),
    ];

    println!("┌────────────────────────┬──────────┬───────┬──────────┬──────────┐");
    println!("│ Signal Type            │ Accuracy │ Nodes │ Est. Hz  │ True Hz  │");
    println!("├────────────────────────┼──────────┼───────┼──────────┼──────────┤");

    for (name, ref mut signal) in signals.iter_mut() {
        let results = analyze_biosignal(signal.as_mut(), sample_rate, duration, threshold);

        let name_fmt = if name.len() > 22 {
            format!("{}...", &name[..19])
        } else {
            format!("{:<22}", name)
        };

        println!(
            "│ {} │ {:5.1}%   │ {:>5} │ {:7.1}  │ {:7.1}  │",
            name_fmt,
            results.accuracy * 100.0,
            results.final_nodes,
            results.estimated_frequency,
            results.expected_frequency
        );
    }

    println!("└────────────────────────┴──────────┴───────┴──────────┴──────────┘");
    println!();

    println!("Key Insights:");
    println!("  - Higher frequency signals (Beta, Gamma) may show lower accuracy due to");
    println!("    rapid threshold crossings exceeding the network's temporal resolution");
    println!("  - Low frequency signals (Respiratory, HRV) should show higher accuracy");
    println!("  - Budding correlates with signal complexity and prediction difficulty");
    println!("  - Network adapts to signal characteristics through autopoiesis");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                       ANALYSIS COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}