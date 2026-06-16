// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Sentinel - Standalone Demo
//!
//! Zero-shot temporal rhythm recognition using LTC dynamics.
//! This is a self-contained version that proves the core thesis:
//! "Transformers spatialize time. Symthaea LIVES in time."

use anyhow::Result;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// =============================================================================
// HDC (Hyperdimensional Computing)
// =============================================================================

const HDC_DIM: usize = 1024; // Reduced for speed (full system uses 16,384)

#[derive(Clone)]
struct HV {
    values: Vec<f32>,
}

impl HV {
    fn zero() -> Self {
        Self {
            values: vec![0.0; HDC_DIM],
        }
    }

    fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            values: (0..HDC_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        }
    }

    fn similarity(&self, other: &HV) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a > 1e-6 && mag_b > 1e-6 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }

    fn add(&self, other: &HV) -> HV {
        HV {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    fn scale(&self, s: f32) -> HV {
        HV {
            values: self.values.iter().map(|v| v * s).collect(),
        }
    }
}

// =============================================================================
// LTC (Liquid Time-Constant) Network
// =============================================================================

struct LtcNode {
    state: HV,
    tau: f32,   // Time constant in ms
    weight: HV, // Input weight vector
    bias: HV,   // Bias vector
}

impl LtcNode {
    fn new(tau: f32) -> Self {
        Self {
            state: HV::random().scale(0.1),
            tau,
            weight: HV::random().scale(0.5),
            bias: HV::random().scale(0.1),
        }
    }

    fn step(&mut self, dt: f32, input: &HV) {
        // LTC dynamics: dx/dt = (-x + tanh(Wx + b)) / τ
        let mut activation = HV::zero();
        for i in 0..HDC_DIM {
            let wx = self.weight.values[i] * input.values[i];
            activation.values[i] = (wx + self.bias.values[i]).tanh();
        }

        let decay = (-dt / self.tau).exp();
        for i in 0..HDC_DIM {
            self.state.values[i] =
                self.state.values[i] * decay + activation.values[i] * (1.0 - decay);
        }
    }
}

struct HierarchicalLtc {
    nodes: Vec<LtcNode>, // 5 levels with different τ
    phi: f64,            // Integrated information estimate
}

impl HierarchicalLtc {
    fn new() -> Self {
        // 5 levels with τ from 500ms (slow) to 30ms (fast)
        let taus = [500.0, 200.0, 80.0, 40.0, 30.0];
        Self {
            nodes: taus.iter().map(|&tau| LtcNode::new(tau)).collect(),
            phi: 0.0,
        }
    }

    fn step(&mut self, dt: f32, input: &HV) {
        // Feed input to all levels (different τ respond to different frequencies)
        for node in &mut self.nodes {
            node.step(dt, input);
        }

        // Compute simple Φ estimate (state variance across levels)
        let mean: Vec<f32> = (0..HDC_DIM)
            .map(|i| {
                self.nodes.iter().map(|n| n.state.values[i]).sum::<f32>() / self.nodes.len() as f32
            })
            .collect();

        let var: f32 = self
            .nodes
            .iter()
            .map(|n| {
                n.state
                    .values
                    .iter()
                    .zip(&mean)
                    .map(|(s, m)| (s - m).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (self.nodes.len() * HDC_DIM) as f32;

        self.phi = var as f64;
    }

    fn get_state(&self) -> HV {
        // Combine states from all levels
        let mut combined = HV::zero();
        for node in &self.nodes {
            combined = combined.add(&node.state);
        }
        combined.scale(1.0 / self.nodes.len() as f32)
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

// =============================================================================
// Pattern Detection
// =============================================================================

#[derive(Clone)]
struct RhythmPattern {
    _name: String,
    trajectory: Vec<HV>,
    mean_state: HV,
    freq_signature: [f32; 4], // Power at 0.5, 1, 2, 4 Hz
}

impl RhythmPattern {
    fn new(name: &str) -> Self {
        Self {
            _name: name.to_string(),
            trajectory: Vec::new(),
            mean_state: HV::zero(),
            freq_signature: [0.0; 4],
        }
    }

    fn finalize(&mut self) {
        if self.trajectory.is_empty() {
            return;
        }

        // Compute mean
        let mut sum = HV::zero();
        for state in &self.trajectory {
            sum = sum.add(state);
        }
        self.mean_state = sum.scale(1.0 / self.trajectory.len() as f32);

        // Compute frequency signature (simple DFT at target frequencies)
        let fps = 30.0;
        let freqs = [0.5, 1.0, 2.0, 4.0];
        let signal: Vec<f32> = self
            .trajectory
            .iter()
            .map(|s| s.values.iter().take(100).sum())
            .collect();

        let n = signal.len();
        let mean: f32 = signal.iter().sum::<f32>() / n as f32;

        for (fi, &freq) in freqs.iter().enumerate() {
            let omega = 2.0 * std::f32::consts::PI * freq / fps;
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &x) in signal.iter().enumerate() {
                let angle = omega * i as f32;
                real += (x - mean) * angle.cos();
                imag += (x - mean) * angle.sin();
            }

            self.freq_signature[fi] = (real * real + imag * imag).sqrt() / n as f32;
        }
    }

    fn similarity(&self, other: &RhythmPattern) -> f32 {
        if self.trajectory.is_empty() || other.trajectory.is_empty() {
            return 0.0;
        }

        // 1. Mean state similarity
        let mean_sim = self.mean_state.similarity(&other.mean_state);

        // 2. Frequency signature similarity
        let self_sum: f32 = self.freq_signature.iter().sum::<f32>().max(1e-6);
        let other_sum: f32 = other.freq_signature.iter().sum::<f32>().max(1e-6);

        let freq_sim: f32 = self
            .freq_signature
            .iter()
            .zip(&other.freq_signature)
            .map(|(a, b)| (a / self_sum) * (b / other_sum))
            .sum::<f32>()
            .sqrt();

        // Weighted combination
        0.4 * mean_sim + 0.6 * freq_sim
    }
}

struct RhythmDetector {
    ltc: HierarchicalLtc,
    patterns: HashMap<char, RhythmPattern>,
    current_trajectory: Vec<HV>,
    learning_mode: Option<char>,
    dt_ms: f32,
}

impl RhythmDetector {
    fn new() -> Self {
        Self {
            ltc: HierarchicalLtc::new(),
            patterns: HashMap::new(),
            current_trajectory: Vec::new(),
            learning_mode: None,
            dt_ms: 33.3, // ~30 fps
        }
    }

    fn process(&mut self, brightness: f32) -> (String, f32, f64, HashMap<char, f32>) {
        // Create input HV from brightness
        let mut input = HV::zero();
        for i in 0..HDC_DIM {
            input.values[i] = brightness * ((i as f32 * 0.1).sin() + 1.0);
        }

        // Step LTC
        self.ltc.step(self.dt_ms, &input);

        // Record trajectory
        let state = self.ltc.get_state();
        self.current_trajectory.push(state.clone());
        if self.current_trajectory.len() > 90 {
            self.current_trajectory.remove(0);
        }

        // If learning, record to pattern
        if let Some(key) = self.learning_mode {
            if let Some(p) = self.patterns.get_mut(&key) {
                p.trajectory.push(state);
            }
        }

        // Detect
        let mut sims = HashMap::new();
        let mut current = RhythmPattern::new("current");
        current.trajectory = self.current_trajectory.clone();
        current.finalize();

        for (key, pattern) in &self.patterns {
            let sim = pattern.similarity(&current);
            sims.insert(*key, sim);
        }

        let (best_pattern, best_sim) = sims
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.to_string(), *v))
            .unwrap_or(("None".to_string(), 0.0));

        let pattern_str = if best_sim > 0.5 {
            best_pattern
        } else if best_sim > 0.3 {
            "Uncertain".to_string()
        } else {
            "Unknown".to_string()
        };

        (pattern_str, best_sim, self.ltc.phi, sims)
    }

    fn start_learning(&mut self, key: char) {
        self.learning_mode = Some(key);
        self.patterns
            .insert(key, RhythmPattern::new(&key.to_string()));
        self.ltc.reset();
        self.current_trajectory.clear();
    }

    fn stop_learning(&mut self) -> Option<char> {
        if let Some(key) = self.learning_mode.take() {
            if let Some(p) = self.patterns.get_mut(&key) {
                p.finalize();
            }
            self.current_trajectory.clear();
            Some(key)
        } else {
            None
        }
    }

    fn learned_count(&self) -> usize {
        self.patterns.len()
    }
}

// =============================================================================
// Mock Video Patterns
// =============================================================================

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum MockPattern {
    Blink { bpm: u32 },
    Heartbeat { bpm: u32 },
    Polyrhythm { bpm_a: u32, bpm_b: u32 },
    Triplet { bpm: u32 },
}

fn generate_brightness(pattern: MockPattern, frame: u64, fps: f32) -> f32 {
    match pattern {
        MockPattern::Blink { bpm } => {
            let frames_per_beat = (fps * 60.0) / bpm as f32;
            let pos = (frame as f32 % frames_per_beat) / frames_per_beat;
            if pos < 0.5 { 1.0 } else { 0.2 }
        }
        MockPattern::Heartbeat { bpm } => {
            let frames_per_beat = (fps * 60.0) / bpm as f32;
            let pos = (frame as f32 % frames_per_beat) / frames_per_beat;
            if pos < 0.08 {
                1.0
            } else if pos < 0.15 {
                0.3
            } else if pos < 0.22 {
                0.8
            } else {
                0.2
            }
        }
        MockPattern::Polyrhythm { bpm_a, bpm_b } => {
            let fpb_a = (fps * 60.0) / bpm_a as f32;
            let fpb_b = (fps * 60.0) / bpm_b as f32;
            let pos_a = (frame as f32 % fpb_a) / fpb_a;
            let pos_b = (frame as f32 % fpb_b) / fpb_b;
            let pulse_a = if pos_a < 0.3 { 0.5 } else { 0.0 };
            let pulse_b = if pos_b < 0.3 { 0.5 } else { 0.0 };
            0.2 + pulse_a + pulse_b
        }
        MockPattern::Triplet { bpm } => {
            let frames_per_beat = (fps * 60.0) / bpm as f32;
            let pos = (frame as f32 % frames_per_beat) / frames_per_beat;
            if pos < 0.11 || (pos > 0.33 && pos < 0.44) || (pos > 0.67 && pos < 0.78) {
                1.0
            } else {
                0.2
            }
        }
    }
}

// =============================================================================
// Demo
// =============================================================================

struct PatternDef {
    key: char,
    name: &'static str,
    desc: &'static str,
    pattern: MockPattern,
}

fn main() -> Result<()> {
    let patterns = vec![
        PatternDef {
            key: 'A',
            name: "60 BPM",
            desc: "1 beat/sec",
            pattern: MockPattern::Blink { bpm: 60 },
        },
        PatternDef {
            key: 'B',
            name: "120 BPM",
            desc: "2 beats/sec",
            pattern: MockPattern::Blink { bpm: 120 },
        },
        PatternDef {
            key: 'C',
            name: "Heartbeat",
            desc: "LUB-dub",
            pattern: MockPattern::Heartbeat { bpm: 72 },
        },
        PatternDef {
            key: 'D',
            name: "Polyrhythm",
            desc: "3:2 cross",
            pattern: MockPattern::Polyrhythm {
                bpm_a: 90,
                bpm_b: 60,
            },
        },
    ];

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::Relaxed);
    })
    .ok();

    let mut detector = RhythmDetector::new();
    let fps = 30.0;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║      SEMANTIC SENTINEL - LTC Temporal Rhythm Recognition      ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  \"Transformers spatialize time. Symthaea LIVES in time.\"      ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    for p in &patterns {
        println!(
            "║  [{}] {:12} - {:20}                   ║",
            p.key, p.name, p.desc
        );
    }
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  [X] Detect  [S] Summary  [R] Reset  [Q] Quit                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    loop {
        print!("Command> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let cmd = input.trim().to_uppercase();

        match cmd.chars().next() {
            Some('A'..='D') => {
                let key = cmd.chars().next().unwrap();
                let pdef = patterns.iter().find(|p| p.key == key).unwrap();

                println!("\n  Learning {} ({})...", pdef.name, pdef.desc);
                detector.start_learning(key);

                let start = Instant::now();
                let mut frame = 0u64;

                while start.elapsed() < Duration::from_secs(3) && running.load(Ordering::Relaxed) {
                    let brightness = generate_brightness(pdef.pattern, frame, fps);
                    let (_, _, phi, _) = detector.process(brightness);

                    let progress = start.elapsed().as_secs_f32() / 3.0;
                    let bar = (progress * 30.0) as usize;
                    print!(
                        "\r  [{}{}] {:.0}%  Φ={:.4}",
                        "█".repeat(bar),
                        "░".repeat(30 - bar),
                        progress * 100.0,
                        phi
                    );
                    io::stdout().flush()?;

                    frame += 1;
                    std::thread::sleep(Duration::from_millis((1000.0 / fps) as u64));
                }

                detector.stop_learning();
                println!("\n  ✓ Pattern {} learned!\n", key);
            }
            Some('X') => {
                if detector.learned_count() < 2 {
                    println!("\n  Learn at least 2 patterns first!\n");
                    continue;
                }

                println!("\n  Detection mode (Ctrl+C to stop)...\n");
                running.store(true, Ordering::Relaxed);

                let learned: Vec<_> = patterns
                    .iter()
                    .filter(|p| detector.patterns.contains_key(&p.key))
                    .collect();

                let mut pattern_idx = 0;
                let mut pattern_start = Instant::now();
                let mut frame = 0u64;

                while running.load(Ordering::Relaxed) {
                    if pattern_start.elapsed() > Duration::from_secs(5) {
                        pattern_idx = (pattern_idx + 1) % learned.len();
                        pattern_start = Instant::now();
                        println!("\n  [Switching to {}]", learned[pattern_idx].name);
                    }

                    let pdef = learned[pattern_idx];
                    let brightness = generate_brightness(pdef.pattern, frame, fps);
                    let (detected, conf, phi, sims) = detector.process(brightness);

                    let mut sim_str = String::new();
                    let mut keys: Vec<_> = sims.keys().collect();
                    keys.sort();
                    for k in keys {
                        let s = sims[k];
                        let bar = (s * 10.0) as usize;
                        let color = if detected == k.to_string() {
                            "\x1b[32m"
                        } else {
                            "\x1b[90m"
                        };
                        sim_str.push_str(&format!(
                            "{}: {}[{}{}]\x1b[0m {:.2}  ",
                            k,
                            color,
                            "█".repeat(bar.min(10)),
                            "░".repeat(10 - bar.min(10)),
                            s
                        ));
                    }

                    print!(
                        "\r  {} │ Detected: {} ({:.0}%)  Φ={:.4}   ",
                        sim_str,
                        detected,
                        conf * 100.0,
                        phi
                    );
                    io::stdout().flush()?;

                    frame += 1;
                    std::thread::sleep(Duration::from_millis((1000.0 / fps) as u64));
                }
                println!("\n");
            }
            Some('S') => {
                println!("\n  HierarchicalLTC: 5 levels (τ = 500ms → 30ms)");
                println!("  HDC dimension: {}", HDC_DIM);
                println!(
                    "  Learned patterns: {:?}",
                    detector.patterns.keys().collect::<Vec<_>>()
                );
                for (k, p) in &detector.patterns {
                    println!(
                        "    {} - freq sig: [{:.3}, {:.3}, {:.3}, {:.3}]",
                        k,
                        p.freq_signature[0],
                        p.freq_signature[1],
                        p.freq_signature[2],
                        p.freq_signature[3]
                    );
                }
                println!();
            }
            Some('R') => {
                detector = RhythmDetector::new();
                println!("\n  Reset complete.\n");
            }
            Some('Q') => {
                println!("\nGoodbye!");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
