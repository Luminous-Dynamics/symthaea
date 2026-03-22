// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tensor structure encoding.
//!
//! Encodes tensor descriptors (rank, index positions, symmetry, metric)
//! as ContinuousHV via role-binding composition:
//!
//! ```text
//! tensor_hv = bind(ROLE_RANK, rank_hv)
//!           ⊕ bundle(position-bound indices)
//!           ⊕ bind(ROLE_SYMMETRY, sym_hv)
//!           ⊕ bind(ROLE_METRIC, metric_hv)
//!           ⊕ bind(ROLE_SOLUTION, solution_hv)  [if present]
//! ```

use crate::types::{
    IndexPosition, MetricSignature, MetricSolution, TensorDescriptor, TensorSymmetry,
};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// Deterministic seeds for role vectors.
const SEED_ROLE_RANK: u64 = 0x7E45_BA01_0001;
const SEED_ROLE_SYMMETRY: u64 = 0x7E45_BA02_0002;
const SEED_ROLE_METRIC: u64 = 0x7E45_BA03_0003;
const SEED_ROLE_INDEX: u64 = 0x7E45_BA04_0004;
const SEED_ROLE_SOLUTION: u64 = 0x7E45_BA05_0005;

// Seeds for enum variant vectors.
const SEED_CONTRA: u64 = 0x1D0C_0001_0001;
const SEED_COVAR: u64 = 0x1D0C_0002_0002;
const SEED_SYM_NONE: u64 = 0x5A00_0001_0001;
const SEED_SYM_SYMMETRIC: u64 = 0x5A00_0002_0002;
const SEED_SYM_ANTISYMMETRIC: u64 = 0x5A00_0003_0003;
const SEED_SYM_MIXED: u64 = 0x5A00_0004_0004;
const SEED_EUCLIDEAN: u64 = 0xAE70_EC01_0001;
const SEED_LORENTZIAN: u64 = 0xAE70_EC02_0002;
const SEED_GENERAL: u64 = 0xAE70_EC03_0003;

// Seeds for position vectors (index ordering).
const SEED_POSITION_BASE: u64 = 0xF050_BA5E_0001;

// Seeds for MetricSolution boolean properties.
const SEED_SINGULARITY: u64 = 0x501A_0001_0001;
const SEED_HORIZON: u64 = 0x501A_0002_0002;
const SEED_VACUUM: u64 = 0x501A_0003_0003;
const SEED_STATIONARY: u64 = 0x501A_0004_0004;
const SEED_AXISYMMETRIC: u64 = 0x501A_0005_0005;
const SEED_SPHERICAL: u64 = 0x501A_0006_0006;
const SEED_COSMOLOGICAL: u64 = 0x501A_0007_0007;
const SEED_LAMBDA: u64 = 0x501A_0008_0008;
const SEED_CHARGED: u64 = 0x501A_0009_0009;

/// Encodes `TensorDescriptor` to `ContinuousHV`.
pub struct TensorEncoder {
    role_rank: ContinuousHV,
    role_symmetry: ContinuousHV,
    role_metric: ContinuousHV,
    role_index: ContinuousHV,
    role_solution: ContinuousHV,
    // Enum variant vectors
    contra_hv: ContinuousHV,
    covar_hv: ContinuousHV,
    sym_none_hv: ContinuousHV,
    sym_symmetric_hv: ContinuousHV,
    sym_antisymmetric_hv: ContinuousHV,
    sym_mixed_hv: ContinuousHV,
    euclidean_hv: ContinuousHV,
    lorentzian_hv: ContinuousHV,
    general_hv: ContinuousHV,
    // Position vectors for index ordering
    position_hvs: Vec<ContinuousHV>,
    // Solution property vectors
    singularity_hv: ContinuousHV,
    horizon_hv: ContinuousHV,
    vacuum_hv: ContinuousHV,
    stationary_hv: ContinuousHV,
    axisymmetric_hv: ContinuousHV,
    spherical_hv: ContinuousHV,
    cosmological_hv: ContinuousHV,
    lambda_hv: ContinuousHV,
    charged_hv: ContinuousHV,
}

impl TensorEncoder {
    /// Create a new tensor encoder with deterministic role vectors.
    pub fn new() -> Self {
        let mut position_hvs = Vec::with_capacity(8);
        for i in 0..8u64 {
            position_hvs.push(ContinuousHV::random(
                HDC_DIMENSION,
                SEED_POSITION_BASE.wrapping_add(i * 7919),
            ));
        }

        Self {
            role_rank: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_RANK),
            role_symmetry: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_SYMMETRY),
            role_metric: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_METRIC),
            role_index: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_INDEX),
            role_solution: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_SOLUTION),
            contra_hv: ContinuousHV::random(HDC_DIMENSION, SEED_CONTRA),
            covar_hv: ContinuousHV::random(HDC_DIMENSION, SEED_COVAR),
            sym_none_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SYM_NONE),
            sym_symmetric_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SYM_SYMMETRIC),
            sym_antisymmetric_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SYM_ANTISYMMETRIC),
            sym_mixed_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SYM_MIXED),
            euclidean_hv: ContinuousHV::random(HDC_DIMENSION, SEED_EUCLIDEAN),
            lorentzian_hv: ContinuousHV::random(HDC_DIMENSION, SEED_LORENTZIAN),
            general_hv: ContinuousHV::random(HDC_DIMENSION, SEED_GENERAL),
            position_hvs,
            singularity_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SINGULARITY),
            horizon_hv: ContinuousHV::random(HDC_DIMENSION, SEED_HORIZON),
            vacuum_hv: ContinuousHV::random(HDC_DIMENSION, SEED_VACUUM),
            stationary_hv: ContinuousHV::random(HDC_DIMENSION, SEED_STATIONARY),
            axisymmetric_hv: ContinuousHV::random(HDC_DIMENSION, SEED_AXISYMMETRIC),
            spherical_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SPHERICAL),
            cosmological_hv: ContinuousHV::random(HDC_DIMENSION, SEED_COSMOLOGICAL),
            lambda_hv: ContinuousHV::random(HDC_DIMENSION, SEED_LAMBDA),
            charged_hv: ContinuousHV::random(HDC_DIMENSION, SEED_CHARGED),
        }
    }

    /// Encode a tensor descriptor to ContinuousHV.
    pub fn encode(&self, desc: &TensorDescriptor) -> ContinuousHV {
        let mut components: Vec<ContinuousHV> = Vec::new();

        // 1. Rank encoding: bind(ROLE_RANK, scale(rank))
        let rank_hv = self.role_rank.scale(desc.rank as f32 / 4.0); // Normalize by typical max rank
        components.push(rank_hv);

        // 2. Index positions: bundle of position-bound co/contravariant markers
        if !desc.index_positions.is_empty() {
            let index_hvs: Vec<ContinuousHV> = desc
                .index_positions
                .iter()
                .enumerate()
                .map(|(i, pos)| {
                    let pos_hv = match pos {
                        IndexPosition::Contravariant => &self.contra_hv,
                        IndexPosition::Covariant => &self.covar_hv,
                    };
                    let position = &self.position_hvs[i % self.position_hvs.len()];
                    self.role_index.bind(&position.bind(pos_hv))
                })
                .collect();
            let refs: Vec<&ContinuousHV> = index_hvs.iter().collect();
            components.push(ContinuousHV::bundle(&refs));
        }

        // 3. Symmetry encoding
        let sym_hv = match desc.symmetry {
            TensorSymmetry::None => &self.sym_none_hv,
            TensorSymmetry::Symmetric => &self.sym_symmetric_hv,
            TensorSymmetry::Antisymmetric => &self.sym_antisymmetric_hv,
            TensorSymmetry::Mixed => &self.sym_mixed_hv,
        };
        components.push(self.role_symmetry.bind(sym_hv));

        // 4. Metric signature encoding
        let metric_hv = match desc.metric {
            MetricSignature::Euclidean(n) => self.euclidean_hv.scale(n as f32 / 4.0),
            MetricSignature::Lorentzian(n) => self.lorentzian_hv.scale(n as f32 / 4.0),
            MetricSignature::General(p, q) => {
                let p_hv = self.euclidean_hv.scale(p as f32 / 4.0);
                let q_hv = self.general_hv.scale(q as f32 / 4.0);
                ContinuousHV::bundle(&[&p_hv, &q_hv])
            }
        };
        components.push(self.role_metric.bind(&metric_hv));

        // 5. Optional metric solution properties
        if let Some(sol) = &desc.solution {
            components.push(self.encode_solution(sol));
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode metric solution properties as a bundle of boolean property vectors.
    fn encode_solution(&self, sol: &MetricSolution) -> ContinuousHV {
        let mut active = Vec::new();

        if sol.singularity {
            active.push(&self.singularity_hv);
        }
        if sol.horizon {
            active.push(&self.horizon_hv);
        }
        if sol.vacuum {
            active.push(&self.vacuum_hv);
        }
        if sol.stationary {
            active.push(&self.stationary_hv);
        }
        if sol.axisymmetric {
            active.push(&self.axisymmetric_hv);
        }
        if sol.spherically_symmetric {
            active.push(&self.spherical_hv);
        }
        if sol.cosmological {
            active.push(&self.cosmological_hv);
        }
        if sol.has_cosmological_constant {
            active.push(&self.lambda_hv);
        }
        if sol.has_charge {
            active.push(&self.charged_hv);
        }

        if active.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let bundled = ContinuousHV::bundle(&active);
        self.role_solution.bind(&bundled)
    }

    /// Compute similarity between two tensor descriptors.
    pub fn similarity(&self, a: &TensorDescriptor, b: &TensorDescriptor) -> f32 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b)
    }
}

impl Default for TensorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_encodes() {
        let enc = TensorEncoder::new();
        let hv = enc.encode(&TensorDescriptor::scalar(MetricSignature::Euclidean(3)));
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn same_tensor_identical() {
        let enc = TensorEncoder::new();
        let t1 = TensorDescriptor::antisymmetric_2(MetricSignature::Lorentzian(4));
        let t2 = TensorDescriptor::antisymmetric_2(MetricSignature::Lorentzian(4));
        let sim = enc.similarity(&t1, &t2);
        assert!(sim > 0.999, "Same tensor, got sim = {sim}");
    }

    #[test]
    fn symmetric_vs_antisymmetric() {
        let enc = TensorEncoder::new();
        let sym = TensorDescriptor::symmetric_2(MetricSignature::Lorentzian(4));
        let anti = TensorDescriptor::antisymmetric_2(MetricSignature::Lorentzian(4));
        let sim = enc.similarity(&sym, &anti);
        // Same rank, same metric, different symmetry — moderately similar
        assert!(
            sim < 0.95 && sim > 0.2,
            "Sym vs Antisym: expected moderate sim, got {sim}"
        );
    }

    #[test]
    fn different_rank_different() {
        let enc = TensorEncoder::new();
        let scalar = TensorDescriptor::scalar(MetricSignature::Lorentzian(4));
        let riemann = TensorDescriptor::riemann(MetricSignature::Lorentzian(4));
        let sim = enc.similarity(&scalar, &riemann);
        assert!(sim < 0.7, "Scalar vs Riemann: expected low sim, got {sim}");
    }

    #[test]
    fn euclidean_vs_lorentzian_metric() {
        let enc = TensorEncoder::new();
        let euc = TensorDescriptor::vector(MetricSignature::Euclidean(3));
        let lor = TensorDescriptor::vector(MetricSignature::Lorentzian(4));
        let sim = enc.similarity(&euc, &lor);
        // Same rank but different metric — moderately different
        assert!(
            sim < 0.95,
            "Euclidean vs Lorentzian vector, got sim = {sim}"
        );
    }

    #[test]
    fn metric_solution_encoding() {
        let enc = TensorEncoder::new();
        let schwarzschild = MetricSolution {
            singularity: true,
            horizon: true,
            vacuum: true,
            stationary: true,
            axisymmetric: true,
            spherically_symmetric: true,
            cosmological: false,
            has_cosmological_constant: false,
            has_charge: false,
        };
        let kerr = MetricSolution {
            singularity: true,
            horizon: true,
            vacuum: true,
            stationary: true,
            axisymmetric: true,
            spherically_symmetric: false, // Kerr breaks spherical symmetry
            cosmological: false,
            has_cosmological_constant: false,
            has_charge: false,
        };
        let flrw = MetricSolution {
            singularity: false,
            horizon: false,
            vacuum: false,
            stationary: false,
            axisymmetric: false,
            spherically_symmetric: false,
            cosmological: true,
            has_cosmological_constant: true,
            has_charge: false,
        };

        let mut t_schw = TensorDescriptor::symmetric_2(MetricSignature::Lorentzian(4));
        t_schw.solution = Some(schwarzschild);
        let mut t_kerr = TensorDescriptor::symmetric_2(MetricSignature::Lorentzian(4));
        t_kerr.solution = Some(kerr);
        let mut t_flrw = TensorDescriptor::symmetric_2(MetricSignature::Lorentzian(4));
        t_flrw.solution = Some(flrw);

        let schw_kerr = enc.similarity(&t_schw, &t_kerr);
        let schw_flrw = enc.similarity(&t_schw, &t_flrw);

        assert!(
            schw_kerr > schw_flrw,
            "Schwarzschild ≈ Kerr ({schw_kerr}) > Schwarzschild ≈ FLRW ({schw_flrw})"
        );
    }

    #[test]
    fn deterministic_tensor_encoding() {
        let enc1 = TensorEncoder::new();
        let enc2 = TensorEncoder::new();
        let t = TensorDescriptor::riemann(MetricSignature::Lorentzian(4));
        let hv1 = enc1.encode(&t);
        let hv2 = enc2.encode(&t);
        assert_eq!(hv1.values, hv2.values);
    }

    #[test]
    fn vector_vs_covector() {
        let enc = TensorEncoder::new();
        let vec = TensorDescriptor::vector(MetricSignature::Lorentzian(4));
        let cov = TensorDescriptor::covector(MetricSignature::Lorentzian(4));
        let sim = enc.similarity(&vec, &cov);
        // Same rank, same metric, different index position — high but not identical
        assert!(
            sim > 0.3 && sim < 0.999,
            "Vector vs Covector: expected moderate-high sim, got {sim}"
        );
    }
}
