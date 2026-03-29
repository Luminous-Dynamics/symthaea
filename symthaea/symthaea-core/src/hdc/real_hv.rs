// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Backward-Compatible Alias for ContinuousHV
//!
//! `RealHV` is a type alias for [`ContinuousHV`](super::unified_hv::ContinuousHV).
//! New code should use `ContinuousHV` directly.

/// Backward-compatible alias: `RealHV` is now `ContinuousHV`.
///
/// All methods previously on `RealHV` are available on `ContinuousHV`.
/// New code should use `ContinuousHV` directly.
pub type RealHV = super::unified_hv::ContinuousHV;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_hv_is_continuous_hv() {
        // RealHV is now a type alias for ContinuousHV
        let a = RealHV::random(2048, 42);
        let b = RealHV::random(2048, 43);

        let sim = a.similarity(&b);
        assert!(
            sim.abs() < 0.2,
            "Random real-valued vectors should be nearly orthogonal, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_real_hv_bind_preserves_similarity_gradient() {
        let dim = 2048;
        let a = RealHV::random(dim, 42);

        let noise_0_1 = RealHV::random(dim, 100).scale(0.1_f32);
        let noise_1_0 = RealHV::random(dim, 103);

        let ones = RealHV::ones(dim);
        let a_with_0_1 = a.bind(&ones.add(&noise_0_1));
        let a_with_1_0 = a.bind(&noise_1_0);

        let sim_0_1 = a.similarity(&a_with_0_1);
        let sim_1_0 = a.similarity(&a_with_1_0);

        assert!(
            sim_0_1 > 0.7,
            "Small noise should preserve high similarity, got {:.4}",
            sim_0_1
        );
        assert!(
            sim_1_0.abs() < 0.3,
            "Large noise should create low similarity, got {:.4}",
            sim_1_0
        );
    }

    #[test]
    fn test_real_hv_bundle_preserves_components() {
        let a = RealHV::random(2048, 1);
        let b = RealHV::random(2048, 2);
        let c = RealHV::random(2048, 3);

        let bundled = RealHV::bundle(&[&a, &b, &c]);

        assert!(
            bundled.similarity(&a) > 0.4,
            "Bundle should be similar to component A"
        );
        assert!(
            bundled.similarity(&b) > 0.4,
            "Bundle should be similar to component B"
        );
        assert!(
            bundled.similarity(&c) > 0.4,
            "Bundle should be similar to component C"
        );
    }

    #[test]
    fn test_similarity_convention_comparison_with_binary() {
        use crate::hdc::binary_hv::BinaryHV;

        let real_a = RealHV::random(2048, 42);
        let real_b = RealHV::random(2048, 43);
        let bin_a = BinaryHV::random(42);
        let bin_b = BinaryHV::random(43);

        let real_sim = real_a.similarity(&real_b);
        let bin_sim = bin_a.similarity(&bin_b);

        assert!(
            real_sim.abs() < 0.15,
            "RealHV random vectors should be near-orthogonal, got {:.4}",
            real_sim
        );
        assert!(
            (bin_sim - 0.5).abs() < 0.05,
            "BinaryHV random vectors should be ~0.5, got {:.4}",
            bin_sim
        );
    }
}
