// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Acoustic wave propagation and cochlear mechanics.
//!
//! Solves the wave equation in 1D/2D elastic media and models the
//! basilar membrane frequency-position map for consciousness research.
//!
//! ∂²p/∂t² = c² ∇²p  (linear wave equation)
//!
//! References:
//! - Kinsler et al. (2000). *Fundamentals of Acoustics*. Wiley.
//! - de Boer, E. (1996). Mechanics of the Cochlea. Springer Handbook.

use std::f64::consts::PI;

/// Speed of sound in various media (m/s).
pub const C_AIR: f64 = 343.0;
pub const C_WATER: f64 = 1480.0;
pub const C_STEEL: f64 = 5960.0;
pub const C_BONE: f64 = 4080.0;

/// 1D acoustic wave equation solver (leapfrog method).
#[derive(Debug, Clone)]
pub struct AcousticWave1D {
    /// Pressure field at current time
    pub p: Vec<f64>,
    /// Pressure field at previous time
    p_prev: Vec<f64>,
    /// Sound speed at each cell (m/s)
    pub c: Vec<f64>,
    /// Grid spacing (m)
    pub dx: f64,
    /// Time step (s)
    pub dt: f64,
    pub n: usize,
    pub step: usize,
}

impl AcousticWave1D {
    pub fn new(n: usize, dx: f64, c_sound: f64) -> Self {
        let dt = 0.5 * dx / c_sound; // CFL
        Self {
            p: vec![0.0; n],
            p_prev: vec![0.0; n],
            c: vec![c_sound; n],
            dx, dt, n,
            step: 0,
        }
    }

    /// Advance one step: p^{n+1} = 2p^n - p^{n-1} + (c dt/dx)² (p_{i+1} - 2p_i + p_{i-1})
    pub fn step_once(&mut self) {
        let mut p_next = vec![0.0; self.n];
        for i in 1..self.n - 1 {
            let courant = self.c[i] * self.dt / self.dx;
            let courant_sq = courant * courant;
            p_next[i] = 2.0 * self.p[i] - self.p_prev[i]
                + courant_sq * (self.p[i + 1] - 2.0 * self.p[i] + self.p[i - 1]);
        }
        self.p_prev = std::mem::replace(&mut self.p, p_next);
        self.step += 1;
    }

    /// Inject a pressure pulse at position `idx`.
    pub fn inject_pulse(&mut self, idx: usize, amplitude: f64) {
        if idx < self.n {
            self.p[idx] += amplitude;
        }
    }

    /// Total acoustic energy: E = Σ ½ p²/(ρc²) dx
    pub fn total_energy(&self) -> f64 {
        let rho = 1.225; // Air density kg/m³
        self.p.iter().enumerate()
            .map(|(i, &pi)| 0.5 * pi * pi / (rho * self.c[i] * self.c[i]) * self.dx)
            .sum()
    }
}

/// Sound pressure level in decibels: SPL = 20 log₁₀(p_rms / p_ref)
/// p_ref = 20 μPa (threshold of hearing)
pub fn sound_pressure_level(p_rms: f64) -> f64 {
    let p_ref = 20e-6; // Pa
    20.0 * (p_rms / p_ref).log10()
}

/// Sound intensity from pressure: I = p²/(ρc) (W/m²)
pub fn sound_intensity(p_rms: f64, rho: f64, c: f64) -> f64 {
    p_rms * p_rms / (rho * c)
}

/// Acoustic impedance: Z = ρc (Pa·s/m = Rayl)
pub fn acoustic_impedance(density: f64, speed: f64) -> f64 {
    density * speed
}

/// Reflection coefficient at impedance boundary.
/// R = (Z₂ - Z₁)/(Z₂ + Z₁)
pub fn impedance_reflection(z1: f64, z2: f64) -> f64 {
    (z2 - z1) / (z2 + z1)
}

// ── Cochlear Mechanics ──────────────────────────────────────────────────────

/// Greenwood frequency-position map for the human cochlea.
///
/// f(x) = A × (10^(ax) - k)
///
/// where x is the fractional distance from apex (0 = apex, 1 = base),
/// A = 165.4 Hz, a = 2.1, k = 0.88.
///
/// This maps position along the basilar membrane to characteristic frequency.
pub fn greenwood_frequency(x_fractional: f64) -> f64 {
    165.4 * (10.0_f64.powf(2.1 * x_fractional) - 0.88)
}

/// Inverse Greenwood: frequency → position on basilar membrane.
pub fn greenwood_position(frequency: f64) -> f64 {
    ((frequency / 165.4 + 0.88).log10()) / 2.1
}

/// Critical bandwidth (ERB scale): ERB(f) = 24.7 × (4.37f/1000 + 1)
/// in Hz, for center frequency f in Hz.
pub fn equivalent_rectangular_bandwidth(frequency: f64) -> f64 {
    24.7 * (4.37 * frequency / 1000.0 + 1.0)
}

/// Number of ERBs between two frequencies (perceptual distance).
pub fn erb_distance(f1: f64, f2: f64) -> f64 {
    let erb_rate = |f: f64| 21.4 * (4.37 * f / 1000.0 + 1.0).log10();
    (erb_rate(f2) - erb_rate(f1)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_propagation() {
        let mut wave = AcousticWave1D::new(200, 0.01, C_AIR);
        wave.inject_pulse(100, 1.0);

        for _ in 0..50 {
            wave.step_once();
        }

        // Energy should have spread from source
        let right_energy: f64 = wave.p[110..150].iter().map(|p| p * p).sum();
        assert!(right_energy > 0.0, "Wave should propagate");
    }

    #[test]
    fn test_spl_threshold() {
        // 20 μPa → 0 dB SPL (threshold of hearing)
        let spl = sound_pressure_level(20e-6);
        assert!((spl - 0.0).abs() < 0.01, "SPL at threshold = {:.2} dB", spl);
    }

    #[test]
    fn test_spl_conversation() {
        // Normal conversation: ~60 dB SPL → p_rms ≈ 0.02 Pa
        let spl = sound_pressure_level(0.02);
        assert!((spl - 60.0).abs() < 2.0, "Conversation SPL = {:.1} dB", spl);
    }

    #[test]
    fn test_greenwood_range() {
        // Base (x=1): ~20kHz, Apex (x=0): ~20Hz
        let f_base = greenwood_frequency(1.0);
        let f_apex = greenwood_frequency(0.0);
        assert!(f_base > 15000.0 && f_base < 25000.0, "Base f = {:.0} Hz", f_base);
        assert!(f_apex > 10.0 && f_apex < 50.0, "Apex f = {:.0} Hz", f_apex);
    }

    #[test]
    fn test_greenwood_roundtrip() {
        let f = 1000.0; // Hz
        let x = greenwood_position(f);
        let f_back = greenwood_frequency(x);
        assert!((f - f_back).abs() < 1.0, "Roundtrip: {:.1} → {:.4} → {:.1}", f, x, f_back);
    }

    #[test]
    fn test_impedance_mismatch() {
        // Air → water: large reflection (Z_air ≈ 415, Z_water ≈ 1.5M)
        let z_air = acoustic_impedance(1.225, C_AIR);
        let z_water = acoustic_impedance(1000.0, C_WATER);
        let r = impedance_reflection(z_air, z_water);
        assert!(r > 0.99, "Air-water reflection = {:.4} (nearly total)", r);
    }
}
