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

/// Encodes `DimensionalSignature` to `ContinuousHV`.
///
/// Uses 7 orthogonal basis vectors (one per SI dimension: M, L, T, I, Θ, N, J).
/// Each dimension's exponent scales its basis vector, then all are bundled:
///
/// ```text
/// dim_hv = weighted_bundle([M_basis, L_basis, T_basis, ...], [m_exp, l_exp, t_exp, ...])
/// ```
///
/// Dimensionless quantities encode as the zero vector.
pub struct DimensionalEncoder {
    /// 7 orthogonal basis vectors [M, L, T, I, Θ, N, J].
    basis: Vec<ContinuousHV>,
}

impl DimensionalEncoder {
    /// Create a new encoder with deterministic orthogonal basis.
    pub fn new() -> Self {
        let basis = ContinuousHV::orthogonal_set(HDC_DIMENSION, NUM_SI_DIMS, BASIS_SEED);
        Self { basis }
    }

    /// Encode a dimensional signature as a ContinuousHV.
    ///
    /// Each SI dimension gets its orthogonal basis vector scaled by the exponent,
    /// then all are summed and normalized. This preserves the sign information
    /// (T⁻² and T² point in opposite directions along the T axis).
    ///
    /// Returns the zero vector for dimensionless quantities.
    pub fn encode(&self, sig: &DimensionalSignature) -> ContinuousHV {
        if sig.is_dimensionless() {
            return ContinuousHV::zero(HDC_DIMENSION);
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
            return ContinuousHV::zero(HDC_DIMENSION);
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
    fn dimensionless_encodes_to_zero() {
        let enc = DimensionalEncoder::new();
        let hv = enc.encode(&DimensionalSignature::DIMENSIONLESS);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        // Zero vector has zero norm
        assert!(hv.values.iter().all(|v| *v == 0.0));
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
    fn dimensionless_orthogonal_to_everything() {
        let enc = DimensionalEncoder::new();
        let dimless = enc.encode(&DimensionalSignature::DIMENSIONLESS);
        let energy = enc.encode(&DimensionalSignature::ENERGY);
        // Zero vector has similarity 0 with everything (or NaN if both zero)
        let norm = dimless.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 1e-10, "Dimensionless should be zero vector");
        // Similarity with non-zero should be 0 (cosine with zero vec)
        let sim = dimless.similarity(&energy);
        assert!(sim.abs() < 0.01, "Dimensionless ⊥ Energy, got sim = {sim}");
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
