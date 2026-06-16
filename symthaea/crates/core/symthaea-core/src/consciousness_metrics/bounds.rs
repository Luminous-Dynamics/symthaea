// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mathematical Bounds Verification
//!
//! Runtime assertions for information-theoretic invariants.

use crate::hdc::unified_hv::ContinuousHV;

use super::{ContinuousEntropyEstimator, TruePhiResult};

/// Mathematical bounds for information-theoretic quantities
///
/// These bounds are fundamental to information theory and must hold
/// for any valid implementation.
#[derive(Debug, Clone)]
pub struct InformationBounds {
    /// Maximum entropy for n bins: H_max = log(n)
    pub max_entropy_bits: f64,
    /// Maximum entropy in nats: H_max = ln(n)
    pub max_entropy_nats: f64,
    /// Number of bins used
    pub num_bins: usize,
}

impl InformationBounds {
    /// Create bounds for a given number of bins
    pub fn for_bins(num_bins: usize) -> Self {
        Self {
            max_entropy_bits: (num_bins as f64).log2(),
            max_entropy_nats: (num_bins as f64).ln(),
            num_bins,
        }
    }

    /// Default bounds for 16 bins
    pub fn default_16() -> Self {
        Self::for_bins(16)
    }

    /// Check if entropy value is valid
    pub fn is_valid_entropy(&self, h: f64, use_bits: bool) -> bool {
        let max_h = if use_bits {
            self.max_entropy_bits
        } else {
            self.max_entropy_nats
        };
        h >= 0.0 && h <= max_h + 1e-10 // Small tolerance for numerical precision
    }

    /// Check if mutual information value is valid
    pub fn is_valid_mi(&self, mi: f64, h1: f64, h2: f64) -> bool {
        // MI must be non-negative and ≤ min(H(X), H(Y))
        mi >= -1e-10 && mi <= h1.min(h2) + 1e-10
    }

    /// Check if Φ value is valid given system EI
    pub fn is_valid_phi(&self, phi: f64, system_ei: f64) -> bool {
        // Φ must be non-negative and ≤ system EI
        phi >= -1e-10 && phi <= system_ei + 1e-10
    }
}

/// Bounds-checking entropy calculator
///
/// Wraps an estimator and verifies all returned values satisfy
/// information-theoretic bounds.
#[derive(Debug, Clone)]
pub struct BoundsCheckingCalculator {
    estimator: ContinuousEntropyEstimator,
    bounds: InformationBounds,
    /// Whether to panic on bound violation (true) or just warn (false)
    strict: bool,
}

impl BoundsCheckingCalculator {
    /// Create a new bounds-checking calculator
    pub fn new(strict: bool) -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
            bounds: InformationBounds::default_16(),
            strict,
        }
    }

    /// Compute entropy with bounds checking
    pub fn entropy(&self, hv: &ContinuousHV) -> f64 {
        let h = self.estimator.entropy(hv);

        if !self.bounds.is_valid_entropy(h, self.estimator.use_bits) {
            let msg = format!(
                "Entropy bound violation: H = {:.6}, max = {:.6}",
                h,
                if self.estimator.use_bits {
                    self.bounds.max_entropy_bits
                } else {
                    self.bounds.max_entropy_nats
                }
            );
            if self.strict {
                panic!("{}", msg);
            }
        }

        h.clamp(0.0, self.bounds.max_entropy_bits)
    }

    /// Compute mutual information with bounds checking
    pub fn mutual_information(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let h1 = self.estimator.entropy(hv1);
        let h2 = self.estimator.entropy(hv2);
        let mi = self.estimator.mutual_information_fast(hv1, hv2);

        if !self.bounds.is_valid_mi(mi, h1, h2) {
            let msg = format!(
                "MI bound violation: I = {:.6}, H1 = {:.6}, H2 = {:.6}, max = {:.6}",
                mi,
                h1,
                h2,
                h1.min(h2)
            );
            if self.strict {
                panic!("{}", msg);
            }
        }

        mi.clamp(0.0, h1.min(h2))
    }

    /// Verify all bounds for a Φ computation result
    pub fn verify_phi_result(&self, result: &TruePhiResult) -> Vec<String> {
        let mut violations = Vec::new();

        // Check component entropies
        for (i, &h) in result.component_entropies.iter().enumerate() {
            if !self.bounds.is_valid_entropy(h, true) {
                violations.push(format!("Component {i} entropy out of bounds: {h:.6}"));
            }
        }

        // Check MI matrix symmetry
        let n = result.mutual_information_matrix.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let mi_ij = result.mutual_information_matrix[i][j];
                let mi_ji = result.mutual_information_matrix[j][i];
                if (mi_ij - mi_ji).abs() > 1e-10 {
                    violations.push(format!(
                        "MI matrix not symmetric: [{i},{j}]={mi_ij:.6}, [{j},{i}]={mi_ji:.6}"
                    ));
                }
            }
        }

        // Check Φ ≤ system EI
        if !self.bounds.is_valid_phi(result.phi, result.system_ei) {
            violations.push(format!(
                "Φ exceeds system EI: Φ={:.6}, EI={:.6}",
                result.phi, result.system_ei
            ));
        }

        // Check MIP EI ≤ system EI
        if result.mip_ei > result.system_ei + 1e-10 {
            violations.push(format!(
                "MIP EI exceeds system EI: MIP={:.6}, system={:.6}",
                result.mip_ei, result.system_ei
            ));
        }

        violations
    }
}
