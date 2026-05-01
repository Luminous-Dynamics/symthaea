// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tensor Networks — efficient representation of quantum many-body states.
//!
//! Matrix Product States (MPS) represent 1D quantum states as a chain of
//! tensors, enabling efficient computation of properties without exponential
//! cost. The bond dimension χ controls the amount of entanglement captured.
//!
//! References:
//! - Perez-Garcia et al. (2007). Quant. Inf. Comput. 7, 401.
//! - Orús, R. (2014). Ann. Phys. 349, 117.

/// A Matrix Product State (MPS) for a 1D quantum system.
///
/// |ψ⟩ = Σ Tr(A₁^{s₁} A₂^{s₂} ... A_N^{s_N}) |s₁ s₂ ... s_N⟩
///
/// Each A_i^{s_i} is a χ×χ matrix (or χ×χ' at boundaries).
#[derive(Debug, Clone)]
pub struct MatrixProductState {
    /// Number of sites
    pub n_sites: usize,
    /// Local Hilbert space dimension (e.g., 2 for spin-1/2)
    pub d: usize,
    /// Bond dimension (max entanglement)
    pub chi: usize,
    /// Tensors: tensors[site][s * chi * chi + i * chi + j] = A_site^s[i,j]
    pub tensors: Vec<Vec<f64>>,
}

impl MatrixProductState {
    /// Create a product state (no entanglement, χ=1).
    /// Each site is in state |0⟩.
    pub fn product_state(n_sites: usize, d: usize) -> Self {
        let chi = 1;
        let tensors = (0..n_sites)
            .map(|_| {
                let mut t = vec![0.0; d * chi * chi];
                t[0] = 1.0; // |0⟩ component
                t
            })
            .collect();
        Self {
            n_sites,
            d,
            chi,
            tensors,
        }
    }

    /// Create a GHZ-like entangled state: (|00...0⟩ + |11...1⟩)/√2
    /// Requires χ = 2.
    pub fn ghz_state(n_sites: usize) -> Self {
        let d = 2;
        let chi = 2;
        let s2 = 1.0 / 2.0_f64.sqrt();

        let mut tensors = Vec::with_capacity(n_sites);

        // First site: A^0 = [s2, 0; 0, 0], A^1 = [0, 0; 0, s2]
        let mut t_first = vec![0.0; d * chi * chi];
        t_first[0 * chi * chi + 0 * chi + 0] = s2; // A^0[0,0]
        t_first[1 * chi * chi + 1 * chi + 1] = s2; // A^1[1,1]
        tensors.push(t_first);

        // Middle sites: A^0 = [[1,0],[0,0]], A^1 = [[0,0],[0,1]]
        for _ in 1..n_sites - 1 {
            let mut t = vec![0.0; d * chi * chi];
            t[0 * chi * chi + 0 * chi + 0] = 1.0; // A^0[0,0]
            t[1 * chi * chi + 1 * chi + 1] = 1.0; // A^1[1,1]
            tensors.push(t);
        }

        // Last site: A^0 = [1, 0; 0, 0], A^1 = [0, 0; 0, 1]
        let mut t_last = vec![0.0; d * chi * chi];
        t_last[0 * chi * chi + 0 * chi + 0] = 1.0;
        t_last[1 * chi * chi + 1 * chi + 1] = 1.0;
        tensors.push(t_last);

        Self {
            n_sites,
            d,
            chi,
            tensors,
        }
    }

    /// Compute the norm ⟨ψ|ψ⟩ via transfer matrix contraction.
    pub fn norm_squared(&self) -> f64 {
        let chi = self.chi;
        let d = self.d;

        // Transfer matrix: T[i,j;i',j'] = Σ_s A^s[i,j] × (A^s[i',j'])
        // Start with identity and contract left to right
        let mut transfer = vec![0.0; chi * chi]; // T[i,i']
        for i in 0..chi {
            transfer[i * chi + i] = 1.0; // Identity
        }

        for site in 0..self.n_sites {
            let t = &self.tensors[site];
            let mut new_transfer = vec![0.0; chi * chi];

            for j in 0..chi {
                for jp in 0..chi {
                    let mut val = 0.0;
                    for i in 0..chi {
                        for ip in 0..chi {
                            let t_ii: f64 = transfer[i * chi + ip];
                            if t_ii.abs() < 1e-15 {
                                continue;
                            }
                            for s in 0..d {
                                val += t_ii
                                    * t[s * chi * chi + i * chi + j]
                                    * t[s * chi * chi + ip * chi + jp];
                            }
                        }
                    }
                    new_transfer[j * chi + jp] = val;
                }
            }
            transfer = new_transfer;
        }

        // Final trace
        (0..chi).map(|i| transfer[i * chi + i]).sum()
    }

    /// Entanglement entropy at bond l (between sites l and l+1).
    /// Computed via the singular values of the bipartition.
    /// For an MPS with bond dimension χ, S ≤ ln(χ).
    pub fn bond_entropy_upper_bound(&self) -> f64 {
        (self.chi as f64).ln()
    }

    /// Memory cost: total number of parameters.
    pub fn parameter_count(&self) -> usize {
        self.n_sites * self.d * self.chi * self.chi
    }

    /// Compression ratio vs full state vector.
    pub fn compression_ratio(&self) -> f64 {
        let full_dim = (self.d as f64).powi(self.n_sites as i32);
        let mps_params = self.parameter_count() as f64;
        full_dim / mps_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_state_norm() {
        let mps = MatrixProductState::product_state(5, 2);
        let norm = mps.norm_squared();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Product state norm = {:.6}",
            norm
        );
    }

    #[test]
    fn test_ghz_state_norm() {
        let mps = MatrixProductState::ghz_state(4);
        let norm = mps.norm_squared();
        assert!((norm - 1.0).abs() < 1e-10, "GHZ state norm = {:.6}", norm);
    }

    #[test]
    fn test_compression_exponential() {
        // 20-site spin chain: 2^20 = 1M states, but MPS with χ=4 has 20*2*16 = 640 params
        let mps = MatrixProductState::product_state(20, 2);
        let ratio = mps.compression_ratio();
        assert!(
            ratio > 1000.0,
            "Compression ratio should be large: {:.0}",
            ratio
        );
    }

    #[test]
    fn test_entropy_bound() {
        let mps = MatrixProductState::ghz_state(4);
        let s_max = mps.bond_entropy_upper_bound();
        assert!(
            (s_max - 2.0_f64.ln()).abs() < 1e-14,
            "S_max = ln(2) for χ=2"
        );
    }

    #[test]
    fn test_parameter_count() {
        let mps = MatrixProductState::product_state(10, 2);
        assert_eq!(mps.parameter_count(), 10 * 2 * 1 * 1);
    }
}
