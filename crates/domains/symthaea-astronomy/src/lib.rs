// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-astronomy
//!
//! Observational astronomy for Symthaea — stellar/blackbody, orbital, distance,
//! and relativistic relations. Complements `mycelix-space` (orbital-mechanics
//! propagation) and `symthaea-orbital` with the observational-astronomy layer.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs textbook.
//!
//! ## Example
//!
//! ```
//! use symthaea_astronomy::{wien_peak_wavelength_nm, orbital_period_years};
//! assert!((wien_peak_wavelength_nm(5778.0) - 501.6).abs() < 1.0);  // the Sun
//! assert!((orbital_period_years(1.0) - 1.0).abs() < 1e-9);          // Earth
//! ```

/// Wien's displacement law: blackbody peak wavelength `λ = b/T`, returned in nm.
/// Wien constant `b = 2.897771955e-3 m·K`.
pub fn wien_peak_wavelength_nm(temperature_k: f64) -> f64 {
    2.897_771_955e-3 / temperature_k * 1.0e9
}

/// Kepler's third law in solar units: `T = a^{3/2}` (years, `a` in AU).
pub fn orbital_period_years(semi_major_axis_au: f64) -> f64 {
    semi_major_axis_au.powf(1.5)
}

/// Semi-major axis (AU) from an orbital period (years): inverse of Kepler's third.
pub fn semi_major_axis_au(period_years: f64) -> f64 {
    period_years.powf(2.0 / 3.0)
}

/// Distance modulus `μ = m − M = 5·log₁₀(d) − 5` for `d` in parsecs.
pub fn distance_modulus(distance_parsecs: f64) -> f64 {
    5.0 * distance_parsecs.log10() - 5.0
}

/// Absolute magnitude from apparent magnitude and distance (parsecs):
/// `M = m − 5·log₁₀(d/10)`.
pub fn absolute_magnitude(apparent_magnitude: f64, distance_parsecs: f64) -> f64 {
    apparent_magnitude - 5.0 * (distance_parsecs / 10.0).log10()
}

/// Schwarzschild radius `r_s = 2GM/c²` (m).
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    const G: f64 = 6.674_30e-11;
    const C: f64 = 2.997_924_58e8;
    2.0 * G * mass_kg / (C * C)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wien_for_the_sun() {
        // The Sun (~5778 K) peaks at ~502 nm (green-ish).
        assert!((wien_peak_wavelength_nm(5778.0) - 501.6).abs() < 1.0);
        // Hotter stars peak bluer (shorter wavelength).
        assert!(wien_peak_wavelength_nm(10000.0) < wien_peak_wavelength_nm(5778.0));
    }

    #[test]
    fn kepler_known_planets() {
        assert!((orbital_period_years(1.0) - 1.0).abs() < 1e-9); // Earth
        // Mars: a=1.524 AU → ~1.88 yr.
        assert!((orbital_period_years(1.524) - 1.881).abs() < 1e-2);
        // Inverse round-trips.
        assert!((semi_major_axis_au(orbital_period_years(2.5)) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn distance_modulus_known() {
        // 100 pc → μ = 5·log10(100) − 5 = 5.
        assert!((distance_modulus(100.0) - 5.0).abs() < 1e-9);
        // A star of apparent mag 10 at 100 pc has absolute mag 5.
        assert!((absolute_magnitude(10.0, 100.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn schwarzschild_radius_of_the_sun() {
        // Sun (~1.989e30 kg) → ~2953 m.
        let rs = schwarzschild_radius(1.989e30);
        assert!((rs - 2953.0).abs() < 5.0, "rs={rs}");
    }
}
