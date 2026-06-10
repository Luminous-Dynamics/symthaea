// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wavetable synthesis with consciousness-driven morphing.
//!
//! Provides a bank of single-cycle wavetables that consciousness level
//! continuously morphs between: noise → saw → square → glass → choir.
//! Cubic interpolation for reading, linear crossfade for morphing.

use std::f32::consts::{PI, TAU};

/// Single-cycle wavetable (power-of-2 samples).
pub struct Wavetable {
    pub samples: Vec<f32>,
    pub name: &'static str,
}

/// Ordered bank of wavetables for consciousness-driven morphing.
pub struct WavetableBank {
    pub tables: Vec<Wavetable>,
}

impl WavetableBank {
    /// Create the default 5-table bank: sine → triangle → saw → square → organ.
    pub fn default_bank() -> Self {
        let size = 2048;
        Self {
            tables: vec![
                generate_sine(size),
                generate_triangle(size),
                generate_saw(size),
                generate_square(size),
                generate_organ(size),
            ],
        }
    }

    /// Number of tables in the bank.
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Read a sample at fractional position with cubic interpolation,
    /// morphing between two adjacent tables.
    ///
    /// `morph` [0, 1]: 0 = first table, 1 = last table.
    /// `phase` [0, 1): fractional position within the cycle.
    pub fn read(&self, phase: f32, morph: f32) -> f32 {
        if self.tables.is_empty() {
            return 0.0;
        }

        let morph = morph.clamp(0.0, 1.0);
        let max_idx = (self.tables.len() - 1) as f32;
        let table_pos = morph * max_idx;
        let t0 = (table_pos as usize).min(self.tables.len() - 1);
        let t1 = (t0 + 1).min(self.tables.len() - 1);
        let frac = table_pos - t0 as f32;

        let s0 = cubic_read(&self.tables[t0].samples, phase);
        let s1 = cubic_read(&self.tables[t1].samples, phase);

        // Linear crossfade between adjacent tables
        s0 * (1.0 - frac) + s1 * frac
    }
}

/// Wavetable oscillator with phase accumulation and smooth morph targeting.
pub struct WavetableOscillator {
    pub phase: f32,
    pub morph_position: f32,
    pub target_morph: f32,
    pub morph_rate: f32,
}

impl WavetableOscillator {
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            morph_position: 0.0,
            target_morph: 0.0,
            morph_rate: 0.005, // smooth morph over ~200 samples
        }
    }

    /// Generate one sample, advancing phase.
    pub fn tick(&mut self, bank: &WavetableBank, freq: f32, sample_rate: f32) -> f32 {
        // Smooth morph toward target
        self.morph_position += (self.target_morph - self.morph_position) * self.morph_rate;

        let sample = bank.read(self.phase, self.morph_position);

        // Advance phase
        self.phase += freq / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        sample
    }

    /// Set consciousness-driven morph target.
    /// Low consciousness → simple waveforms (sine/triangle).
    /// High consciousness → complex waveforms (square/organ).
    pub fn set_consciousness_morph(&mut self, consciousness: f32) {
        self.target_morph = consciousness.clamp(0.0, 1.0);
    }
}

// ─── Wavetable Generation ───────────────────────────────────────────────────

fn generate_sine(size: usize) -> Wavetable {
    Wavetable {
        samples: (0..size)
            .map(|i| (TAU * i as f32 / size as f32).sin())
            .collect(),
        name: "sine",
    }
}

fn generate_triangle(size: usize) -> Wavetable {
    Wavetable {
        samples: (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                if t < 0.25 {
                    4.0 * t
                } else if t < 0.75 {
                    2.0 - 4.0 * t
                } else {
                    4.0 * t - 4.0
                }
            })
            .collect(),
        name: "triangle",
    }
}

fn generate_saw(size: usize) -> Wavetable {
    // Band-limited saw via additive synthesis (16 harmonics)
    Wavetable {
        samples: (0..size)
            .map(|i| {
                let phase = TAU * i as f32 / size as f32;
                let mut s = 0.0f32;
                for h in 1..=16 {
                    s += (h as f32 * phase).sin() / h as f32;
                }
                s * 2.0 / PI
            })
            .collect(),
        name: "saw",
    }
}

fn generate_square(size: usize) -> Wavetable {
    // Band-limited square via odd harmonics
    Wavetable {
        samples: (0..size)
            .map(|i| {
                let phase = TAU * i as f32 / size as f32;
                let mut s = 0.0f32;
                for h in (1..=15).step_by(2) {
                    s += (h as f32 * phase).sin() / h as f32;
                }
                s * 4.0 / PI
            })
            .collect(),
        name: "square",
    }
}

fn generate_organ(size: usize) -> Wavetable {
    // Organ-like: fundamental + 2nd + 3rd + 4th + 6th + 8th harmonics
    Wavetable {
        samples: (0..size)
            .map(|i| {
                let phase = TAU * i as f32 / size as f32;
                let s = phase.sin() * 1.0
                    + (2.0 * phase).sin() * 0.5
                    + (3.0 * phase).sin() * 0.25
                    + (4.0 * phase).sin() * 0.125
                    + (6.0 * phase).sin() * 0.06
                    + (8.0 * phase).sin() * 0.03;
                s / 1.965 // normalize
            })
            .collect(),
        name: "organ",
    }
}

/// Cubic Hermite interpolation for smooth wavetable reading.
fn cubic_read(table: &[f32], phase: f32) -> f32 {
    let len = table.len();
    if len == 0 {
        return 0.0;
    }
    let pos = phase * len as f32;
    let idx = pos as usize;
    let frac = pos - idx as f32;

    let i0 = if idx == 0 { len - 1 } else { idx - 1 };
    let i1 = idx % len;
    let i2 = (idx + 1) % len;
    let i3 = (idx + 2) % len;

    let (y0, y1, y2, y3) = (table[i0], table[i1], table[i2], table[i3]);

    // Cubic Hermite
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

    ((c3 * frac + c2) * frac + c1) * frac + c0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_has_5_tables() {
        let bank = WavetableBank::default_bank();
        assert_eq!(bank.len(), 5);
    }

    #[test]
    fn sine_table_correct() {
        let bank = WavetableBank::default_bank();
        // At morph=0 (sine), phase=0.25 should be ~1.0 (peak of sine)
        let val = bank.read(0.25, 0.0);
        assert!((val - 1.0).abs() < 0.05, "sine peak: {val}");
    }

    #[test]
    fn morph_interpolates() {
        let bank = WavetableBank::default_bank();
        let sine_val = bank.read(0.25, 0.0); // pure sine
        let tri_val = bank.read(0.25, 0.25); // 50% sine + 50% triangle
        let saw_val = bank.read(0.25, 0.5); // pure triangle (table index 1→2 midpoint)
        // Values should differ across morph positions
        assert!(
            (sine_val - saw_val).abs() > 0.01,
            "morph should change output"
        );
    }

    #[test]
    fn oscillator_advances_phase() {
        let bank = WavetableBank::default_bank();
        let mut osc = WavetableOscillator::new();
        let s1 = osc.tick(&bank, 440.0, 44100.0);
        let s2 = osc.tick(&bank, 440.0, 44100.0);
        assert!(osc.phase > 0.0, "phase should advance");
        // At 440Hz/44100, phase advances ~0.01 per sample
        assert!((osc.phase - 440.0 * 2.0 / 44100.0).abs() < 0.001);
    }

    #[test]
    fn consciousness_morph_targets() {
        let bank = WavetableBank::default_bank();
        let mut osc = WavetableOscillator::new();
        osc.set_consciousness_morph(0.0); // sine
        for _ in 0..500 {
            osc.tick(&bank, 440.0, 44100.0);
        }
        assert!(osc.morph_position < 0.1, "low consciousness → sine region");

        osc.set_consciousness_morph(1.0); // organ
        for _ in 0..500 {
            osc.tick(&bank, 440.0, 44100.0);
        }
        assert!(
            osc.morph_position > 0.8,
            "high consciousness → organ region"
        );
    }

    #[test]
    fn cubic_interpolation_smooth() {
        let table: Vec<f32> = (0..2048).map(|i| (TAU * i as f32 / 2048.0).sin()).collect();
        // Read at exact positions and between — should be smooth
        let v1 = cubic_read(&table, 0.1);
        let v2 = cubic_read(&table, 0.1001);
        assert!(
            (v1 - v2).abs() < 0.01,
            "cubic should be smooth between samples"
        );
    }
}
