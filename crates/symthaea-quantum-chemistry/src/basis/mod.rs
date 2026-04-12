// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gaussian basis set infrastructure.
//!
//! Defines primitive and contracted Gaussian-type orbitals (GTOs),
//! basis sets, and the provider trait for building a basis from a molecule.

pub mod basis_631g;
pub mod sto3g;

use crate::constants::{double_factorial, PI_CONST};
use crate::molecule::Molecule;
use serde::{Deserialize, Serialize};

/// Angular momentum shell type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellType {
    S,
    P,
    D,
}

impl ShellType {
    /// Total angular momentum quantum number L.
    pub fn angular_momentum(self) -> u8 {
        match self {
            ShellType::S => 0,
            ShellType::P => 1,
            ShellType::D => 2,
        }
    }

    /// Number of Cartesian components (1 for S, 3 for P, 6 for D).
    pub fn n_cartesian(self) -> usize {
        match self {
            ShellType::S => 1,
            ShellType::P => 3,
            ShellType::D => 6,
        }
    }

    /// Cartesian angular momentum triples (l, m, n) where l+m+n = L.
    pub fn cartesian_components(self) -> Vec<(u8, u8, u8)> {
        match self {
            ShellType::S => vec![(0, 0, 0)],
            ShellType::P => vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            ShellType::D => vec![
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2),
            ],
        }
    }
}

/// A single primitive Gaussian function.
///
/// g(r) = N * x^l * y^m * z^n * exp(-α|r - R|²)
///
/// where N is the normalization constant, α is the exponent,
/// and (l,m,n) are Cartesian angular momentum quantum numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveGaussian {
    /// Orbital exponent α
    pub alpha: f64,
    /// Contraction coefficient (within a contracted shell)
    pub coeff: f64,
    /// Center position in Bohr
    pub center: [f64; 3],
    /// Cartesian angular momentum: x^l
    pub l: u8,
    /// Cartesian angular momentum: y^m
    pub m: u8,
    /// Cartesian angular momentum: z^n
    pub n: u8,
}

impl PrimitiveGaussian {
    /// Normalization constant for this primitive.
    ///
    /// N = (2α/π)^(3/4) * (4α)^((l+m+n)/2) / sqrt((2l-1)!! * (2m-1)!! * (2n-1)!!)
    pub fn normalization(&self) -> f64 {
        let l = self.l as i32;
        let m = self.m as i32;
        let n = self.n as i32;
        let ltot = l + m + n;

        let prefactor = (2.0 * self.alpha / PI_CONST).powf(0.75);
        let angular = (4.0 * self.alpha).powi(ltot) as f64;
        let denom =
            double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1);

        prefactor * (angular / denom).sqrt()
    }

    /// Total angular momentum L = l + m + n.
    pub fn total_angular_momentum(&self) -> u8 {
        self.l + self.m + self.n
    }
}

/// A contracted Gaussian-type orbital: linear combination of primitives.
///
/// χ(r) = Σ_i c_i * N_i * g_i(r)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractedGaussian {
    /// Primitive Gaussians with their contraction coefficients.
    pub primitives: Vec<PrimitiveGaussian>,
    /// Shell type (S, P, D)
    pub shell_type: ShellType,
}

impl ContractedGaussian {
    /// Compute the self-overlap of this contracted function: ⟨χ|χ⟩
    pub fn self_overlap(&self) -> f64 {
        let mut result = 0.0;
        for pa in &self.primitives {
            let na = pa.normalization();
            for pb in &self.primitives {
                let nb = pb.normalization();
                let p = pa.alpha + pb.alpha;
                // Overlap of two s-type primitives at the same center
                // For general angular momentum, use the full overlap formula
                let ex = crate::integrals::hermite::hermite_coefficient(
                    pa.l as i32,
                    pb.l as i32,
                    0,
                    pa.alpha,
                    pb.alpha,
                    pa.center[0],
                    pb.center[0],
                );
                let ey = crate::integrals::hermite::hermite_coefficient(
                    pa.m as i32,
                    pb.m as i32,
                    0,
                    pa.alpha,
                    pb.alpha,
                    pa.center[1],
                    pb.center[1],
                );
                let ez = crate::integrals::hermite::hermite_coefficient(
                    pa.n as i32,
                    pb.n as i32,
                    0,
                    pa.alpha,
                    pb.alpha,
                    pa.center[2],
                    pb.center[2],
                );
                let s = ex * ey * ez * (PI_CONST / p).powf(1.5);
                result += pa.coeff * pb.coeff * na * nb * s;
            }
        }
        result
    }

    /// Normalize this contracted function so ⟨χ|χ⟩ = 1.
    /// Modifies the contraction coefficients in place.
    pub fn normalize(&mut self) {
        let s = self.self_overlap();
        if s > 1e-15 {
            let factor = 1.0 / s.sqrt();
            for p in &mut self.primitives {
                p.coeff *= factor;
            }
        }
    }
}

/// A complete basis set for a molecule.
#[derive(Debug, Clone)]
pub struct BasisSet {
    /// Name of the basis set (e.g., "STO-3G")
    pub name: String,
    /// All basis functions (one per Cartesian component).
    /// For an sp shell, this includes 1 s-function + 3 p-functions.
    pub functions: Vec<ContractedGaussian>,
}

impl BasisSet {
    /// Number of basis functions.
    pub fn n_basis(&self) -> usize {
        self.functions.len()
    }
}

/// Trait for basis set providers.
pub trait BasisSetProvider {
    /// Build a basis set for the given molecule.
    fn build(molecule: &Molecule) -> BasisSet;

    /// Name of this basis set.
    fn name() -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_type_components() {
        assert_eq!(ShellType::S.n_cartesian(), 1);
        assert_eq!(ShellType::P.n_cartesian(), 3);
        assert_eq!(ShellType::D.n_cartesian(), 6);
    }

    #[test]
    fn test_s_normalization() {
        // For s-type (l=m=n=0): N = (2α/π)^(3/4)
        let g = PrimitiveGaussian {
            alpha: 1.0,
            coeff: 1.0,
            center: [0.0; 3],
            l: 0,
            m: 0,
            n: 0,
        };
        let n = g.normalization();
        let expected = (2.0 / PI_CONST).powf(0.75);
        assert!(
            (n - expected).abs() < 1e-12,
            "N={}, expected={}",
            n,
            expected
        );
    }
}
