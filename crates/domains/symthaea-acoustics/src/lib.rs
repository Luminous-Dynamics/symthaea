// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-acoustics
//!
//! Acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
//! combination, and the Doppler effect. Complements `symthaea-dsp` (digital
//! signals) with the physical-sound layer.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs textbook.
//!
//! ## Example
//!
//! ```
//! use symthaea_acoustics::{speed_of_sound_air, wavelength};
//! let c = speed_of_sound_air(20.0);            // ≈ 343.2 m/s at 20 °C
//! assert!((wavelength(440.0, c) - 0.78).abs() < 0.01); // A4 ≈ 0.78 m
//! ```

/// Speed of sound in dry air at `temp_celsius`: `c = 331.3·√(1 + T/273.15)` (m/s).
pub fn speed_of_sound_air(temp_celsius: f64) -> f64 {
    331.3 * (1.0 + temp_celsius / 273.15).sqrt()
}

/// Wavelength `λ = c/f` (m).
pub fn wavelength(frequency: f64, speed: f64) -> f64 {
    speed / frequency
}

/// Frequency `f = c/λ` (Hz).
pub fn frequency(wavelength: f64, speed: f64) -> f64 {
    speed / wavelength
}

/// Combine incoherent sound-pressure levels (dB): `L = 10·log₁₀(Σ 10^(Lᵢ/10))`.
pub fn combine_decibels(levels: &[f64]) -> f64 {
    let sum: f64 = levels.iter().map(|l| 10f64.powf(l / 10.0)).sum();
    if sum <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * sum.log10()
}

/// Observed frequency under the Doppler effect. Velocities are positive toward
/// the other party: `f' = f·(c + v_observer)/(c − v_source)`.
pub fn doppler_frequency(
    source_freq: f64,
    source_velocity: f64,
    observer_velocity: f64,
    speed: f64,
) -> f64 {
    source_freq * (speed + observer_velocity) / (speed - source_velocity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_at_20c() {
        assert!((speed_of_sound_air(20.0) - 343.2).abs() < 0.2);
        // Colder air → slower sound.
        assert!(speed_of_sound_air(0.0) < speed_of_sound_air(20.0));
    }

    #[test]
    fn wavelength_frequency_roundtrip() {
        let c = 343.0;
        let w = wavelength(440.0, c);
        assert!((frequency(w, c) - 440.0).abs() < 1e-9);
    }

    #[test]
    fn two_equal_sources_add_three_db() {
        // Two incoherent 60 dB sources → ~63.01 dB.
        assert!((combine_decibels(&[60.0, 60.0]) - 63.0103).abs() < 1e-3);
    }

    #[test]
    fn approaching_source_raises_pitch() {
        // 440 Hz source approaching at 34.3 m/s (0.1 Mach), c=343 → ~488.9 Hz.
        let f = doppler_frequency(440.0, 34.3, 0.0, 343.0);
        assert!((f - 488.89).abs() < 0.1, "f={f}");
        assert!(f > 440.0);
    }
}
