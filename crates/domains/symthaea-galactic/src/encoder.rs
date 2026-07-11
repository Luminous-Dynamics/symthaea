// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC encoder for rotation-curve points, following `symthaea-nuclear`'s
//! `NuclearStateEncoder` pattern: one deterministic random basis vector per
//! physical channel, scaled by a normalized value, then bundled.
//!
//! Radius, luminosity, and surface brightness span multiple decades across
//! the SPARC sample (dwarf irregulars to giant spirals), so those channels
//! are log-normalized before the [0,1] clamp — a plain linear normalization
//! would collapse nearly the whole sample into a thin sliver near 0.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// Deterministic seeds for basis vectors — distinct high bits from
// symthaea-nuclear's 0xA0C1_* range to avoid accidental collisions if both
// crates' encoders are ever used side by side.
const SEED_R: u64 = 0xCA1A_0001_0001; // channel: radius
const SEED_VGAS: u64 = 0xCA1A_0002_0002;
const SEED_VDISK: u64 = 0xCA1A_0003_0003;
const SEED_VBUL: u64 = 0xCA1A_0004_0004;
const SEED_SBDISK: u64 = 0xCA1A_0005_0005;
const SEED_SBBUL: u64 = 0xCA1A_0006_0006;
const SEED_LUM: u64 = 0xCA1A_0007_0007;
const SEED_DIST: u64 = 0xCA1A_0008_0008;
const SEED_INC: u64 = 0xCA1A_0009_0009;
const SEED_GASFRAC: u64 = 0xCA1A_000A_000A;

/// One rotation-curve point plus its parent galaxy's global properties,
/// ready for HDC encoding.
#[derive(Debug, Clone)]
pub struct GalaxyPointState {
    /// Radius [kpc]
    pub r_kpc: f64,
    pub v_gas: f64,
    pub v_disk: f64,
    pub v_bul: f64,
    pub sb_disk: f64,
    pub sb_bul: f64,
    /// Galaxy total [3.6] luminosity [1e9 L☉]
    pub luminosity_3p6: f64,
    /// Galaxy distance [Mpc]
    pub distance_mpc: f64,
    /// Galaxy inclination [deg]
    pub inclination_deg: f64,
    /// Galaxy gas fraction: M_HI / (M_HI + Υd·L)
    pub gas_fraction: f64,
}

/// log-normalize `x` (must be > 0) against a plausible max, into [0, 1].
fn log_norm(x: f64, min: f64, max: f64) -> f32 {
    if x <= 0.0 || min <= 0.0 || max <= min {
        return 0.0;
    }
    let x = x.clamp(min, max);
    ((x.ln() - min.ln()) / (max.ln() - min.ln())) as f32
}

/// linear-normalize `x` into [0, 1] given a max magnitude, preserving sign
/// information is NOT needed here — only channel magnitude is encoded, since
/// direction (e.g. negative Vgas) is a physics-model concern, not an HDC one.
fn lin_norm(x: f64, max: f64) -> f32 {
    (x.abs() / max).clamp(0.0, 1.0) as f32
}

/// Encodes galaxy rotation-curve points as 16,384-dimensional hypervectors.
pub struct GalaxyStateEncoder {
    basis_r: ContinuousHV,
    basis_vgas: ContinuousHV,
    basis_vdisk: ContinuousHV,
    basis_vbul: ContinuousHV,
    basis_sbdisk: ContinuousHV,
    basis_sbbul: ContinuousHV,
    basis_lum: ContinuousHV,
    basis_dist: ContinuousHV,
    basis_inc: ContinuousHV,
    basis_gasfrac: ContinuousHV,
}

impl GalaxyStateEncoder {
    pub fn new() -> Self {
        Self {
            basis_r: ContinuousHV::random(HDC_DIMENSION, SEED_R),
            basis_vgas: ContinuousHV::random(HDC_DIMENSION, SEED_VGAS),
            basis_vdisk: ContinuousHV::random(HDC_DIMENSION, SEED_VDISK),
            basis_vbul: ContinuousHV::random(HDC_DIMENSION, SEED_VBUL),
            basis_sbdisk: ContinuousHV::random(HDC_DIMENSION, SEED_SBDISK),
            basis_sbbul: ContinuousHV::random(HDC_DIMENSION, SEED_SBBUL),
            basis_lum: ContinuousHV::random(HDC_DIMENSION, SEED_LUM),
            basis_dist: ContinuousHV::random(HDC_DIMENSION, SEED_DIST),
            basis_inc: ContinuousHV::random(HDC_DIMENSION, SEED_INC),
            basis_gasfrac: ContinuousHV::random(HDC_DIMENSION, SEED_GASFRAC),
        }
    }

    /// Encode a galaxy point state as a ContinuousHV.
    pub fn encode(&self, s: &GalaxyPointState) -> ContinuousHV {
        // r spans ~0.01–200 kpc across the SPARC sample → log-normalize
        let r_hv = self.basis_r.scale(log_norm(s.r_kpc, 0.01, 200.0));
        // Velocities: 0–400 km/s, linear (already a modest dynamic range)
        let vgas_hv = self.basis_vgas.scale(lin_norm(s.v_gas, 400.0));
        let vdisk_hv = self.basis_vdisk.scale(lin_norm(s.v_disk, 400.0));
        let vbul_hv = self.basis_vbul.scale(lin_norm(s.v_bul, 400.0));
        // Surface brightness: decades of range → log-normalize
        let sbdisk_hv = self.basis_sbdisk.scale(log_norm(s.sb_disk, 1.0, 100_000.0));
        let sbbul_hv = self.basis_sbbul.scale(log_norm(s.sb_bul, 1.0, 100_000.0));
        // Luminosity [1e9 Lsun]: dwarfs ~0.001 to giants ~1000 → log-normalize
        let lum_hv = self
            .basis_lum
            .scale(log_norm(s.luminosity_3p6, 0.001, 2000.0));
        // Distance [Mpc]: nearby dwarfs ~0.5 to distant spirals ~300 → log-normalize
        let dist_hv = self.basis_dist.scale(log_norm(s.distance_mpc, 0.5, 300.0));
        // Inclination [deg]: 0–90, linear
        let inc_hv = self.basis_inc.scale(lin_norm(s.inclination_deg, 90.0));
        // Gas fraction: already in [0,1]
        let gasfrac_hv = self
            .basis_gasfrac
            .scale(s.gas_fraction.clamp(0.0, 1.0) as f32);

        ContinuousHV::bundle(&[
            &r_hv,
            &vgas_hv,
            &vdisk_hv,
            &vbul_hv,
            &sbdisk_hv,
            &sbbul_hv,
            &lum_hv,
            &dist_hv,
            &inc_hv,
            &gasfrac_hv,
        ])
    }
}

impl Default for GalaxyStateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Gas fraction M_HI / (M_HI + Υd·L), from the galaxy's HI mass and Υ_d
/// stellar-mass proxy. Clamped to [0, 1] against pathological metadata.
pub fn gas_fraction(mhi_e9msun: f64, luminosity_3p6: f64) -> f64 {
    use crate::constants::UPSILON_DISK;
    let m_star = UPSILON_DISK * luminosity_3p6;
    let denom = mhi_e9msun + m_star;
    if denom <= 0.0 {
        0.0
    } else {
        (mhi_e9msun / denom).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state(r: f64) -> GalaxyPointState {
        GalaxyPointState {
            r_kpc: r,
            v_gas: 20.0,
            v_disk: 80.0,
            v_bul: 0.0,
            sb_disk: 500.0,
            sb_bul: 0.0,
            luminosity_3p6: 5.0,
            distance_mpc: 10.0,
            inclination_deg: 60.0,
            gas_fraction: 0.3,
        }
    }

    #[test]
    fn encoder_produces_correct_dimension() {
        let enc = GalaxyStateEncoder::new();
        let hv = enc.encode(&sample_state(2.0));
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn encoder_is_deterministic() {
        let enc = GalaxyStateEncoder::new();
        let s = sample_state(2.0);
        let hv1 = enc.encode(&s);
        let hv2 = enc.encode(&s);
        assert!(hv1.similarity(&hv2) > 0.999);
    }

    #[test]
    fn neighboring_radii_cluster_more_than_distant_ones() {
        let enc = GalaxyStateEncoder::new();
        let near_a = enc.encode(&sample_state(2.0));
        let near_b = enc.encode(&sample_state(2.2));
        let far = enc.encode(&sample_state(50.0));

        let sim_near = near_a.similarity(&near_b);
        let sim_far = near_a.similarity(&far);
        assert!(
            sim_near > sim_far,
            "r=2.0/r=2.2 similarity ({sim_near}) should exceed r=2.0/r=50 ({sim_far})"
        );
    }

    #[test]
    fn log_norm_handles_full_sparc_radius_range() {
        // Should not saturate to exactly 1.0 for typical outer-disk radii,
        // and must NOT collapse sub-1-kpc inner radii to zero (the bug this
        // test caught: the original single-bound formula returned 0 for any
        // x < 1, silently discarding all inner-disk structure).
        assert!(log_norm(30.0, 0.01, 200.0) < 1.0);
        assert!(log_norm(0.1, 0.01, 200.0) > 0.0);
        assert!(log_norm(0.02, 0.01, 200.0) < log_norm(0.1, 0.01, 200.0));
        assert_eq!(log_norm(0.0, 0.01, 200.0), 0.0);
        assert_eq!(log_norm(-5.0, 0.01, 200.0), 0.0);
    }

    #[test]
    fn gas_fraction_is_bounded() {
        assert!((0.0..=1.0).contains(&gas_fraction(1.0, 5.0)));
        assert_eq!(gas_fraction(0.0, 0.0), 0.0);
        assert_eq!(gas_fraction(-1.0, -1.0), 0.0);
    }
}
