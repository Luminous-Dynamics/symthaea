// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Geometric optics: thin lenses/mirrors, refraction, diffraction. Angles in
//! radians.

/// Thin-lens/mirror image distance from `1/f = 1/dₒ + 1/dᵢ`.
/// Returns `f64::INFINITY` when the object sits at the focal point.
pub fn image_distance(focal_length: f64, object_distance: f64) -> f64 {
    let inv = 1.0 / focal_length - 1.0 / object_distance;
    if inv.abs() < 1e-15 {
        f64::INFINITY
    } else {
        1.0 / inv
    }
}

/// Lateral magnification `m = −dᵢ/dₒ` (negative = inverted, |m|>1 = enlarged).
pub fn magnification(image_distance: f64, object_distance: f64) -> f64 {
    -image_distance / object_distance
}

/// Snell's law refraction angle: `n₁·sinθ₁ = n₂·sinθ₂`. Returns `None` on total
/// internal reflection (no real solution).
pub fn refraction_angle(n1: f64, theta1: f64, n2: f64) -> Option<f64> {
    let s = n1 * theta1.sin() / n2;
    if s.abs() > 1.0 { None } else { Some(s.asin()) }
}

/// Critical angle for total internal reflection going from `n1` into `n2`
/// (requires `n1 > n2`): `θc = asin(n₂/n₁)`.
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 > n2 {
        Some((n2 / n1).asin())
    } else {
        None
    }
}

/// Diffraction-grating angle for order `m`: `d·sinθ = m·λ`. `None` if the order
/// is evanescent (`|mλ/d| > 1`).
pub fn grating_angle(spacing: f64, order: i32, wavelength: f64) -> Option<f64> {
    let s = order as f64 * wavelength / spacing;
    if s.abs() > 1.0 { None } else { Some(s.asin()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn converging_lens_forms_real_image() {
        // f=10, dₒ=30 → dᵢ=15, m=−0.5 (real, inverted, reduced).
        let di = image_distance(10.0, 30.0);
        assert!((di - 15.0).abs() < 1e-9, "di={di}");
        assert!((magnification(di, 30.0) + 0.5).abs() < 1e-9);
    }

    #[test]
    fn object_at_focus_images_at_infinity() {
        assert!(image_distance(10.0, 10.0).is_infinite());
    }

    #[test]
    fn snell_air_to_water() {
        // n1=1, θ1=30°, n2=1.33 → θ2 ≈ 22.08°.
        let t2 = refraction_angle(1.0, 30.0_f64.to_radians(), 1.33).unwrap();
        assert!(
            (t2.to_degrees() - 22.08).abs() < 0.02,
            "θ2={}",
            t2.to_degrees()
        );
    }

    #[test]
    fn critical_angle_water_to_air() {
        // n1=1.33, n2=1 → θc ≈ 48.77°.
        let tc = critical_angle(1.33, 1.0).unwrap();
        assert!(
            (tc.to_degrees() - 48.77).abs() < 0.05,
            "θc={}",
            tc.to_degrees()
        );
        // Going the other way (into a denser medium) has no critical angle.
        assert!(critical_angle(1.0, 1.33).is_none());
    }

    #[test]
    fn grating_first_order() {
        // d=1 µm, λ=500 nm, m=1 → sinθ=0.5 → θ=30°.
        let t = grating_angle(1e-6, 1, 500e-9).unwrap();
        assert!((t - PI / 6.0).abs() < 1e-9);
        // A wavelength wider than the spacing has no first order.
        assert!(grating_angle(400e-9, 1, 500e-9).is_none());
    }
}
