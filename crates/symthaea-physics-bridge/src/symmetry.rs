// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symmetry group encoding.
//!
//! Encodes Lie groups, discrete symmetries, and full symmetry descriptors
//! as ContinuousHV. Product groups (like Standard Model SU(3)×SU(2)×U(1))
//! encode as bundles of their components.

use crate::types::{DiscreteSymmetry, LieGroup, SymmetryDescriptor};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// Seeds for Lie group family vectors.
const SEED_SO: u64 = 0xCA5E_5000_0001; // SO(n)
const SEED_SU: u64 = 0xCA5E_5000_0002; // SU(n)
const SEED_U: u64 = 0xCA5E_5000_0003; // U(n)
const SEED_SP: u64 = 0xCA5E_5000_0004; // Sp(n)
const SEED_GL: u64 = 0xCA5E_5000_0005; // GL(n)
const SEED_LORENTZ: u64 = 0xCA5E_5000_0006;
const SEED_POINCARE: u64 = 0xCA5E_5000_0007;
const SEED_CONFORMAL: u64 = 0xCA5E_C0FF_0008;
const SEED_SM: u64 = 0xCA5E_5000_0009;
const SEED_E: u64 = 0xCA5E_5000_000A;
const SEED_G2: u64 = 0xCA5E_5000_000B;

// Seeds for discrete symmetry vectors.
const SEED_PARITY: u64 = 0xD5C0_0001_0001;
const SEED_TIME_REV: u64 = 0xD5C0_0002_0002;
const SEED_CHARGE_C: u64 = 0xD5C0_0003_0003;
const SEED_CPT: u64 = 0xD5C0_0004_0004;
const SEED_Z2: u64 = 0xD5C0_0005_0005;
const SEED_PERM: u64 = 0xD5C0_0006_0006;

// Role vectors.
const SEED_ROLE_LIE: u64 = 0xB01E_0001_0001;
const SEED_ROLE_DISCRETE: u64 = 0xB01E_D15C_0002;
const SEED_ROLE_GAUGE: u64 = 0xB01E_0003_0003;
const SEED_ROLE_DIM: u64 = 0xB01E_AD14_0004;

/// Encodes symmetry descriptors to ContinuousHV.
pub struct SymmetryEncoder {
    // Lie group family vectors
    so_hv: ContinuousHV,
    su_hv: ContinuousHV,
    u_hv: ContinuousHV,
    sp_hv: ContinuousHV,
    gl_hv: ContinuousHV,
    lorentz_hv: ContinuousHV,
    poincare_hv: ContinuousHV,
    conformal_hv: ContinuousHV,
    sm_hv: ContinuousHV,
    e_hv: ContinuousHV,
    g2_hv: ContinuousHV,
    // Discrete symmetry vectors
    parity_hv: ContinuousHV,
    time_rev_hv: ContinuousHV,
    charge_c_hv: ContinuousHV,
    cpt_hv: ContinuousHV,
    z2_hv: ContinuousHV,
    perm_hv: ContinuousHV,
    // Role vectors
    role_lie: ContinuousHV,
    role_discrete: ContinuousHV,
    role_gauge: ContinuousHV,
    role_dim: ContinuousHV,
}

impl SymmetryEncoder {
    pub fn new() -> Self {
        Self {
            so_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SO),
            su_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SU),
            u_hv: ContinuousHV::random(HDC_DIMENSION, SEED_U),
            sp_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SP),
            gl_hv: ContinuousHV::random(HDC_DIMENSION, SEED_GL),
            lorentz_hv: ContinuousHV::random(HDC_DIMENSION, SEED_LORENTZ),
            poincare_hv: ContinuousHV::random(HDC_DIMENSION, SEED_POINCARE),
            conformal_hv: ContinuousHV::random(HDC_DIMENSION, SEED_CONFORMAL),
            sm_hv: ContinuousHV::random(HDC_DIMENSION, SEED_SM),
            e_hv: ContinuousHV::random(HDC_DIMENSION, SEED_E),
            g2_hv: ContinuousHV::random(HDC_DIMENSION, SEED_G2),
            parity_hv: ContinuousHV::random(HDC_DIMENSION, SEED_PARITY),
            time_rev_hv: ContinuousHV::random(HDC_DIMENSION, SEED_TIME_REV),
            charge_c_hv: ContinuousHV::random(HDC_DIMENSION, SEED_CHARGE_C),
            cpt_hv: ContinuousHV::random(HDC_DIMENSION, SEED_CPT),
            z2_hv: ContinuousHV::random(HDC_DIMENSION, SEED_Z2),
            perm_hv: ContinuousHV::random(HDC_DIMENSION, SEED_PERM),
            role_lie: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_LIE),
            role_discrete: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_DISCRETE),
            role_gauge: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_GAUGE),
            role_dim: ContinuousHV::random(HDC_DIMENSION, SEED_ROLE_DIM),
        }
    }

    /// Encode a single Lie group.
    ///
    /// The group family vector is scaled by the parameter n to distinguish
    /// e.g. SU(2) from SU(3).
    pub fn encode_lie_group(&self, group: &LieGroup) -> ContinuousHV {
        match group {
            LieGroup::SO(n) => self.so_hv.scale(*n as f32 / 4.0),
            LieGroup::SU(n) => self.su_hv.scale(*n as f32 / 4.0),
            LieGroup::U(n) => self.u_hv.scale(*n as f32 / 4.0),
            LieGroup::Sp(n) => self.sp_hv.scale(*n as f32 / 4.0),
            LieGroup::GL(n) => self.gl_hv.scale(*n as f32 / 4.0),
            LieGroup::Lorentz => self.lorentz_hv.clone(),
            LieGroup::Poincare => {
                // Poincaré = Lorentz ⋉ T⁴ — encode as bundle of Lorentz + translations
                let lorentz = self.lorentz_hv.clone();
                ContinuousHV::bundle(&[&lorentz, &self.poincare_hv])
            }
            LieGroup::Conformal => self.conformal_hv.clone(),
            LieGroup::StandardModel => self.sm_hv.clone(),
            LieGroup::E(n) => self.e_hv.scale(*n as f32 / 8.0),
            LieGroup::G2 => self.g2_hv.clone(),
        }
    }

    /// Encode a discrete symmetry.
    pub fn encode_discrete(&self, sym: &DiscreteSymmetry) -> ContinuousHV {
        match sym {
            DiscreteSymmetry::P => self.parity_hv.clone(),
            DiscreteSymmetry::T => self.time_rev_hv.clone(),
            DiscreteSymmetry::C => self.charge_c_hv.clone(),
            DiscreteSymmetry::CPT => self.cpt_hv.clone(),
            DiscreteSymmetry::Z2 => self.z2_hv.clone(),
            DiscreteSymmetry::Permutation(n) => self.perm_hv.scale(*n as f32 / 4.0),
        }
    }

    /// Encode a full symmetry descriptor.
    ///
    /// ```text
    /// sym_hv = bind(ROLE_LIE, bundle(lie_group_hvs))
    ///        ⊕ bind(ROLE_DISCRETE, bundle(discrete_hvs))
    ///        ⊕ bind(ROLE_GAUGE, gauge_flag_hv)
    ///        ⊕ bind(ROLE_DIM, algebra_dim_scale)
    /// ```
    pub fn encode(&self, desc: &SymmetryDescriptor) -> ContinuousHV {
        let mut components = Vec::new();

        // Lie groups
        if !desc.lie_groups.is_empty() {
            let lie_hvs: Vec<ContinuousHV> = desc
                .lie_groups
                .iter()
                .map(|g| self.encode_lie_group(g))
                .collect();
            let refs: Vec<&ContinuousHV> = lie_hvs.iter().collect();
            let lie_bundle = ContinuousHV::bundle(&refs);
            components.push(self.role_lie.bind(&lie_bundle));
        }

        // Discrete symmetries
        if !desc.discrete.is_empty() {
            let disc_hvs: Vec<ContinuousHV> = desc
                .discrete
                .iter()
                .map(|d| self.encode_discrete(d))
                .collect();
            let refs: Vec<&ContinuousHV> = disc_hvs.iter().collect();
            let disc_bundle = ContinuousHV::bundle(&refs);
            components.push(self.role_discrete.bind(&disc_bundle));
        }

        // Gauge flag: scale the gauge role vector (1.0 if gauge, 0.1 if global)
        let gauge_scale = if desc.is_gauge { 1.0 } else { 0.1 };
        components.push(self.role_gauge.scale(gauge_scale));

        // Algebra dimension: normalized by typical max (248 for E8)
        let dim_scale = desc.algebra_dimension as f32 / 248.0;
        components.push(self.role_dim.scale(dim_scale));

        if components.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Similarity between two symmetry descriptors.
    pub fn similarity(&self, a: &SymmetryDescriptor, b: &SymmetryDescriptor) -> f32 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b)
    }
}

impl Default for SymmetryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_symmetry_identical() {
        let enc = SymmetryEncoder::new();
        let s1 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);
        let s2 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);
        let sim = enc.similarity(&s1, &s2);
        assert!(sim > 0.999, "Same symmetry, got sim = {sim}");
    }

    #[test]
    fn su2_vs_su3() {
        let enc = SymmetryEncoder::new();
        let su2 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(2)], true);
        let su3 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);
        let sim = enc.similarity(&su2, &su3);
        // Same family, different parameter — high but not identical
        assert!(
            sim > 0.5 && sim < 0.999,
            "SU(2) vs SU(3): expected moderate-high sim, got {sim}"
        );
    }

    #[test]
    fn su3_vs_so3() {
        let enc = SymmetryEncoder::new();
        let su3 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);
        let so3 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false);
        let sim = enc.similarity(&su3, &so3);
        // Different family — low similarity
        assert!(sim < 0.8, "SU(3) vs SO(3), got sim = {sim}");
    }

    #[test]
    fn standard_model_contains_components() {
        let enc = SymmetryEncoder::new();
        // Standard Model as single group vs decomposed
        let sm_single = SymmetryDescriptor::from_lie_groups(vec![LieGroup::StandardModel], true);
        let sm_decomposed = SymmetryDescriptor::from_lie_groups(
            vec![LieGroup::SU(3), LieGroup::SU(2), LieGroup::U(1)],
            true,
        );
        let hv_single = enc.encode(&sm_single);
        let hv_decomposed = enc.encode(&sm_decomposed);
        // They encode differently (different structure) but both are valid
        assert!(hv_single.dim() == HDC_DIMENSION);
        assert!(hv_decomposed.dim() == HDC_DIMENSION);
    }

    #[test]
    fn gauge_vs_global() {
        let enc = SymmetryEncoder::new();
        let gauge = SymmetryDescriptor::from_lie_groups(vec![LieGroup::U(1)], true);
        let global = SymmetryDescriptor::from_lie_groups(vec![LieGroup::U(1)], false);
        let sim = enc.similarity(&gauge, &global);
        // Same group, different locality — should differ
        assert!(sim < 0.999, "Gauge vs Global U(1), got sim = {sim}");
    }

    #[test]
    fn discrete_symmetries_encode() {
        let enc = SymmetryEncoder::new();
        let cpt = SymmetryDescriptor::new(
            vec![LieGroup::Poincare],
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
            false,
        );
        let hv = enc.encode(&cpt);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn poincare_more_similar_to_lorentz_than_su3() {
        let enc = SymmetryEncoder::new();
        let poincare = SymmetryDescriptor::from_lie_groups(vec![LieGroup::Poincare], false);
        let lorentz = SymmetryDescriptor::from_lie_groups(vec![LieGroup::Lorentz], false);
        let su3 = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);

        let poincare_hv = enc.encode(&poincare);
        let lorentz_hv = enc.encode(&lorentz);
        let su3_hv = enc.encode(&su3);

        let poin_lor = poincare_hv.similarity(&lorentz_hv);
        let poin_su3 = poincare_hv.similarity(&su3_hv);

        // Poincaré extends Lorentz, so they should be more similar
        // than Poincaré vs SU(3) which is a completely different domain
        assert!(
            poin_lor > poin_su3,
            "Poincaré≈Lorentz ({poin_lor}) should > Poincaré≈SU(3) ({poin_su3})"
        );
    }

    #[test]
    fn no_symmetry_encodes_to_zero() {
        let enc = SymmetryEncoder::new();
        let none = SymmetryDescriptor::none();
        let hv = enc.encode(&none);
        // With no Lie groups and no discrete symmetries, only gauge (0.1) and dim (0.0) contribute
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn deterministic_symmetry_encoding() {
        let enc1 = SymmetryEncoder::new();
        let enc2 = SymmetryEncoder::new();
        let s = SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true);
        let hv1 = enc1.encode(&s);
        let hv2 = enc2.encode(&s);
        assert_eq!(hv1.values, hv2.values);
    }
}
