// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::deterministic_seeds::seed_from_name;
use crate::hdc::integrated_information::IntegratedInformation;
use crate::hdc::primitive_system::PrimitiveSystem;
use serde::{Deserialize, Serialize};

/// A number represented in Hyperdimensional space via Peano construction
///
/// Numbers are built compositionally:
/// - 0 = ZERO primitive
/// - 1 = SUCCESSOR ⊗ ZERO
/// - 2 = SUCCESSOR ⊗ (SUCCESSOR ⊗ ZERO)
/// - n = S(S(S(...S(0)...))) with n applications of successor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcNumber {
    /// The hypervector encoding of this number
    pub encoding: BinaryHV,

    /// The numeric value (for verification and display)
    pub value: u64,

    /// The Peano construction trace (for proofs)
    pub construction: Vec<String>,

    /// Φ accumulated during construction
    pub construction_phi: f64,
}

impl HdcNumber {
    /// Create zero - the base case
    pub fn zero(primitives: &PrimitiveSystem) -> Self {
        let zero_prim = primitives.get("ZERO").expect("ZERO primitive must exist");

        Self {
            encoding: zero_prim.encoding,
            value: 0,
            construction: vec!["ZERO".to_string()],
            construction_phi: 0.0,
        }
    }

    /// Create a number from its value using binary decomposition.
    ///
    /// Represents n by bundling deterministic bit-position basis vectors
    /// for each set bit. O(log n) — at most 64 steps for any u64.
    /// Falls back to Peano construction for small values (n <= 16) to
    /// preserve detailed proof traces.
    pub fn from_value(n: u64, primitives: &PrimitiveSystem) -> Self {
        let zero_prim = primitives.get("ZERO").expect("ZERO primitive must exist");

        if n == 0 {
            return Self {
                encoding: zero_prim.encoding,
                value: 0,
                construction: vec!["ZERO".to_string()],
                construction_phi: 0.0,
            };
        }

        // For small values, use Peano construction for detailed proof traces
        if n <= 16 {
            let succ_prim = primitives
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist");

            let mut encoding = zero_prim.encoding;
            let mut construction = vec!["ZERO".to_string()];
            let mut total_phi = 0.0;

            for i in 0..n {
                let new_encoding = succ_prim.encoding.bind(&encoding);
                let step_phi = Self::measure_step_phi(&encoding, &new_encoding);
                total_phi += step_phi;
                encoding = new_encoding;
                construction.push(format!("S({i})"));
            }

            return Self {
                encoding,
                value: n,
                construction,
                construction_phi: total_phi,
            };
        }

        // Binary decomposition for larger values: O(log n)
        let mut components = vec![zero_prim.encoding];
        let mut construction = vec![format!("BINARY_DECOMP({})", n)];
        let mut total_phi = 0.0;

        for bit_pos in 0..64 {
            if n & (1u64 << bit_pos) != 0 {
                let bit_basis = BinaryHV::random(seed_from_name(&format!("NUM_BIT_{bit_pos}")));
                if let Some(last) = components.last() {
                    let step_phi = Self::measure_step_phi(last, &bit_basis);
                    total_phi += step_phi;
                }
                components.push(bit_basis);
                construction.push(format!("bit_{bit_pos}"));
            }
        }

        let encoding = BinaryHV::bundle(&components);

        Self {
            encoding,
            value: n,
            construction,
            construction_phi: total_phi,
        }
    }

    /// Measure Φ contribution of a construction step
    pub(crate) fn measure_step_phi(before: &BinaryHV, after: &BinaryHV) -> f64 {
        let mut phi_calc = IntegratedInformation::new();
        let components = vec![*before, *after];
        phi_calc.compute_phi(&components)
    }

    /// Apply successor to get next number: S(n) = n + 1
    pub fn successor(&self, primitives: &PrimitiveSystem) -> Self {
        let succ_prim = primitives
            .get("SUCCESSOR")
            .expect("SUCCESSOR primitive must exist");

        let new_encoding = succ_prim.encoding.bind(&self.encoding);
        let step_phi = Self::measure_step_phi(&self.encoding, &new_encoding);

        let mut construction = self.construction.clone();
        construction.push(format!("S({})", self.value));

        Self {
            encoding: new_encoding,
            value: self.value + 1,
            construction,
            construction_phi: self.construction_phi + step_phi,
        }
    }

    /// Get similarity to another HdcNumber (for verification)
    pub fn similarity(&self, other: &HdcNumber) -> f32 {
        self.encoding.similarity(&other.encoding)
    }
}
