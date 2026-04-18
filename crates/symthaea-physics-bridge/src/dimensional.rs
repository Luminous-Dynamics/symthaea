// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SI dimensional analysis encoding.
//!
//! Encodes 7-dimensional SI signatures as ContinuousHV using orthogonal
//! basis vectors, one per SI base dimension. The resulting hypervector
//! captures dimensional structure: Energy ≈ Torque (same dims), but
//! Energy ≠ Momentum (different dims).

use crate::types::DimensionalSignature;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Seed for the orthogonal basis set (deterministic).
const BASIS_SEED: u64 = 0xD1A5_1070_5EED_0001;

/// Number of SI base dimensions.
const NUM_SI_DIMS: usize = 7;

/// Total basis vectors: 7 SI dimensions + 1 dedicated DIMENSIONLESS marker.
/// The marker vector is orthogonal to all SI axes, so dimensionless quantities
/// get a self-matching identity without bleeding into any physical signature.
const NUM_BASIS: usize = NUM_SI_DIMS + 1;
const DIMENSIONLESS_BASIS_INDEX: usize = 7;

/// Encodes `DimensionalSignature` to `ContinuousHV`.
///
/// Uses 8 orthogonal basis vectors: one per SI dimension [M, L, T, I, Θ, N, J]
/// plus a dedicated "DIMENSIONLESS" axis. Each dimension's exponent scales its
/// basis vector, then all are bundled:
///
/// ```text
/// dim_hv = bundle(M_basis·m + L_basis·l + T_basis·t + ... + D_basis·[is_dimensionless])
/// ```
///
/// ## Dimensionless handling
///
/// Earlier versions encoded dimensionless quantities as the zero vector, which
/// meant `cosine(0, 0) = 0` — two dimensionless entries scored **zero** on the
/// dimensional axis even when they should have scored a perfect 1.0. This
/// capped natural-units invariants (harmonic oscillator, Lotka-Volterra,
/// Hénon-Heiles) at ~0.706 recognition score because their 0.20-weight
/// dimensional axis contributed nothing.
///
/// The fix maps dimensionless quantities to the dedicated 8th basis vector
/// (`DIMENSIONLESS_BASIS_INDEX`). Self-match now scores 1.0, and the vector
/// is orthogonal to all physical signatures so the dimensional axis still
/// correctly rejects dimensionless-vs-physical matches.
pub struct DimensionalEncoder {
    /// 8 orthogonal basis vectors: [M, L, T, I, Θ, N, J, DIMENSIONLESS].
    basis: Vec<ContinuousHV>,
}

impl DimensionalEncoder {
    /// Create a new encoder with deterministic orthogonal basis.
    pub fn new() -> Self {
        let basis = ContinuousHV::orthogonal_set(HDC_DIMENSION, NUM_BASIS, BASIS_SEED);
        Self { basis }
    }

    /// Encode a dimensional signature as a ContinuousHV.
    ///
    /// Each SI dimension gets its orthogonal basis vector scaled by the exponent,
    /// then all are summed and normalized. This preserves the sign information
    /// (T⁻² and T² point in opposite directions along the T axis).
    ///
    /// Dimensionless quantities return the dedicated DIMENSIONLESS basis
    /// vector — nonzero, orthogonal to all physical signatures, so
    /// `sim(dimless, dimless) = 1.0` and `sim(dimless, physical) ≈ 0`.
    pub fn encode(&self, sig: &DimensionalSignature) -> ContinuousHV {
        if sig.is_dimensionless() {
            return self.basis[DIMENSIONLESS_BASIS_INDEX].clone();
        }

        let exponents = sig.as_array();

        // Sum basis vectors scaled by exponents (manual, not weighted_bundle,
        // because weighted_bundle divides by weight_sum which can be near-zero
        // with negative exponents like T⁻²).
        let mut result = ContinuousHV::zero(HDC_DIMENSION);
        let mut any_nonzero = false;

        for (i, &exp) in exponents.iter().enumerate() {
            if exp != 0 {
                let scaled = self.basis[i].scale(exp as f32);
                result = result.add(&scaled);
                any_nonzero = true;
            }
        }

        if !any_nonzero {
            // Unreachable in practice (is_dimensionless() already handled
            // above), but fall back to the dimensionless marker for safety.
            return self.basis[DIMENSIONLESS_BASIS_INDEX].clone();
        }

        result.normalize()
    }

    /// Similarity between two dimensional signatures.
    pub fn similarity(&self, a: &DimensionalSignature, b: &DimensionalSignature) -> f32 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b)
    }
}

impl Default for DimensionalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensionless_self_matches() {
        // Regression: dimensionless MUST encode to a nonzero self-matching
        // vector, not the zero vector. Two dimensionless queries should score
        // 1.0 on the dimensional similarity axis so natural-units invariants
        // like x² + v² can actually receive the 0.20 dimensional-axis bonus
        // against their dimensionless catalog cousins.
        let enc = DimensionalEncoder::new();
        let a = enc.encode(&DimensionalSignature::DIMENSIONLESS);
        let b = enc.encode(&DimensionalSignature::DIMENSIONLESS);
        assert_eq!(a.dim(), HDC_DIMENSION);
        assert!(a.values.iter().any(|v| *v != 0.0), "should be nonzero");
        let sim = a.similarity(&b);
        assert!(sim > 0.999, "dimensionless ↔ dimensionless = 1.0, got {sim}");
    }

    #[test]
    fn energy_equals_torque() {
        let enc = DimensionalEncoder::new();
        let energy = enc.encode(&DimensionalSignature::ENERGY);
        let torque = enc.encode(&DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        });
        // Identical dimensions → identical encoding
        let sim = energy.similarity(&torque);
        assert!(sim > 0.999, "Energy ≈ Torque (same dims), got sim = {sim}");
    }

    #[test]
    fn energy_differs_from_momentum() {
        let enc = DimensionalEncoder::new();
        let energy = enc.encode(&DimensionalSignature::ENERGY);
        let momentum = enc.encode(&DimensionalSignature::MOMENTUM);
        let sim = energy.similarity(&momentum);
        // Energy (M¹L²T⁻²) and Momentum (M¹L¹T⁻¹) share M and differ in L/T exponents.
        // High similarity is expected since they share 2 of 3 active dimensions.
        // But they should NOT be identical (sim < 1.0).
        assert!(
            sim < 0.999,
            "Energy ≠ Momentum (should differ), got sim = {sim}"
        );
        assert!(
            sim > 0.5,
            "Energy and Momentum are related (share M, L, T dims), got sim = {sim}"
        );
    }

    #[test]
    fn force_differs_from_pressure() {
        let enc = DimensionalEncoder::new();
        let force = enc.encode(&DimensionalSignature::FORCE);
        let pressure = enc.encode(&DimensionalSignature::PRESSURE);
        let sim = force.similarity(&pressure);
        // Force (MLT⁻²) vs Pressure (ML⁻¹T⁻²) — differ in L exponent
        assert!(sim < 0.95, "Force ≠ Pressure, got sim = {sim}");
    }

    #[test]
    fn same_quantity_is_identical() {
        let enc = DimensionalEncoder::new();
        let e1 = enc.encode(&DimensionalSignature::ENERGY);
        let e2 = enc.encode(&DimensionalSignature::ENERGY);
        let sim = e1.similarity(&e2);
        assert!(sim > 0.999, "Same quantity, got sim = {sim}");
    }

    #[test]
    fn dimensionless_orthogonal_to_physical() {
        // Dimensionless now encodes to its dedicated 8th basis vector, which
        // IS nonzero but is orthogonal to all physical signatures by
        // construction (it's part of the orthogonal_set). So
        // sim(dimless, energy) ≈ 0 still — just without the zero-vs-zero
        // pathology that capped natural-units invariants at 0.706.
        let enc = DimensionalEncoder::new();
        let dimless = enc.encode(&DimensionalSignature::DIMENSIONLESS);
        let energy = enc.encode(&DimensionalSignature::ENERGY);
        let norm = dimless.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Dimensionless should be nonzero now");
        let sim = dimless.similarity(&energy);
        assert!(sim.abs() < 0.05, "Dimensionless ⊥ Energy, got sim = {sim}");
    }

    #[test]
    fn related_quantities_more_similar() {
        let enc = DimensionalEncoder::new();
        // Velocity (LT⁻¹) and Acceleration (LT⁻²) share L
        let vel = enc.encode(&DimensionalSignature::VELOCITY);
        let acc = enc.encode(&DimensionalSignature::ACCELERATION);
        let vel_acc_sim = vel.similarity(&acc);

        // Velocity vs Charge (IT) — completely different
        let charge = enc.encode(&DimensionalSignature::CHARGE);
        let vel_charge_sim = vel.similarity(&charge);

        assert!(
            vel_acc_sim > vel_charge_sim,
            "Velocity should be more similar to Acceleration ({vel_acc_sim}) than Charge ({vel_charge_sim})"
        );
    }

    #[test]
    fn nonzero_norm_for_physical_quantities() {
        let enc = DimensionalEncoder::new();
        for sig in &[
            DimensionalSignature::ENERGY,
            DimensionalSignature::FORCE,
            DimensionalSignature::VELOCITY,
            DimensionalSignature::MOMENTUM,
            DimensionalSignature::PRESSURE,
            DimensionalSignature::CHARGE,
            DimensionalSignature::ACTION,
        ] {
            let hv = enc.encode(sig);
            let norm = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(norm > 0.0, "Physical quantity should have non-zero norm");
            assert!(hv.values.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn deterministic_encoding() {
        let enc1 = DimensionalEncoder::new();
        let enc2 = DimensionalEncoder::new();
        let hv1 = enc1.encode(&DimensionalSignature::ENERGY);
        let hv2 = enc2.encode(&DimensionalSignature::ENERGY);
        assert_eq!(hv1.values, hv2.values, "Encoding should be deterministic");
    }
}
