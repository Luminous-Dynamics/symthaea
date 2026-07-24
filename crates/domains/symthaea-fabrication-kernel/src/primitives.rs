// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geometric primitives as hypervectors
//!
//! Each primitive shape, transform operator, and boolean operator has a
//! deterministic prototype HV in 16,384 dimensions.

use symthaea_core::hdc::unified_hv::ContinuousHV;

pub const FAB_KERNEL_DIM: usize = 16_384;

// Deterministic seeds for primitive HVs
const CUBE_SEED: u64 = 0x4355_4245_0001;
const CYLINDER_SEED: u64 = 0x4359_4C49_0002;
const SPHERE_SEED: u64 = 0x5350_4845_0003;
const CONE_SEED: u64 = 0x434F_4E45_0004;
const TORUS_SEED: u64 = 0x544F_5255_0005;
const SCALE_SEED: u64 = 0x5343_414C_0010;
const ROTATE_SEED: u64 = 0x524F_5441_0020;
const TRANSLATE_SEED: u64 = 0x5452_414E_0030;
const UNION_SEED: u64 = 0x554E_494F_0040;
const SUBTRACT_SEED: u64 = 0x5355_4254_0050;
const INTERSECT_SEED: u64 = 0x494E_5445_0060;

// Shape primitives
pub fn cube_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, CUBE_SEED)
}
pub fn cylinder_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, CYLINDER_SEED)
}
pub fn sphere_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, SPHERE_SEED)
}
pub fn cone_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, CONE_SEED)
}
pub fn torus_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, TORUS_SEED)
}

// Transform operators
pub fn scale_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, SCALE_SEED)
}
pub fn rotate_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, ROTATE_SEED)
}
pub fn translate_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, TRANSLATE_SEED)
}

// Boolean operators
pub fn union_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, UNION_SEED)
}
pub fn subtract_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, SUBTRACT_SEED)
}
pub fn intersect_hv() -> ContinuousHV {
    ContinuousHV::random(FAB_KERNEL_DIM, INTERSECT_SEED)
}

/// Encode an exact scalar parameter as a deterministic role-specific HV.
///
/// The previous `base.scale(value).normalize()` construction erased every
/// positive magnitude: `0.1`, `1.0`, and `100.0` normalized to the same vector.
/// Mixing the canonical IEEE-754 bits into the seed preserves parameter identity
/// without relying on vector magnitude surviving later normalization.
pub fn param_hv(param_seed: u64, value: f32) -> ContinuousHV {
    let canonical_bits = if value == 0.0 {
        0
    } else if value.is_nan() {
        f32::NAN.to_bits()
    } else {
        value.to_bits()
    } as u64;
    ContinuousHV::random(FAB_KERNEL_DIM, mix_seed(param_seed ^ canonical_bits))
}

fn mix_seed(mut value: u64) -> u64 {
    // SplitMix64 finalizer: deterministic avalanche without mutable global state.
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Identify which primitive an HV is closest to
pub fn identify_primitive(hv: &ContinuousHV) -> (&'static str, f32) {
    let primitives = [
        ("cube", cube_hv()),
        ("cylinder", cylinder_hv()),
        ("sphere", sphere_hv()),
        ("cone", cone_hv()),
        ("torus", torus_hv()),
    ];
    let mut best = ("unknown", -1.0f32);
    for (name, prim_hv) in &primitives {
        let sim = hv.similarity(prim_hv);
        if sim > best.1 {
            best = (name, sim);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        assert_eq!(cube_hv().values, cube_hv().values);
    }

    #[test]
    fn test_orthogonal() {
        let c = cube_hv();
        let s = sphere_hv();
        assert!(
            c.similarity(&s).abs() < 0.1,
            "Primitives should be near-orthogonal"
        );
    }

    #[test]
    fn test_self_similarity() {
        let c = cube_hv();
        assert!((c.similarity(&c) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_identify_primitive() {
        let (name, sim) = identify_primitive(&cube_hv());
        assert_eq!(name, "cube");
        assert!(sim > 0.99);
    }

    #[test]
    fn scalar_parameter_magnitude_is_not_erased() {
        let one = param_hv(0xABCD, 1.0);
        let two = param_hv(0xABCD, 2.0);
        assert!(one.similarity(&two).abs() < 0.15);
        assert_eq!(one.values, param_hv(0xABCD, 1.0).values);
    }

    #[test]
    fn test_bind_produces_dissimilar() {
        let c = cube_hv();
        let s = scale_hv();
        let bound = c.bind(&s);
        assert!(bound.similarity(&c).abs() < 0.15);
    }
}
