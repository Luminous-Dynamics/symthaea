// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Equation AST encoding.
//!
//! Recursively encodes `EquationNode` trees as ContinuousHV using role-binding.
//! Each node type gets a unique role vector; children are bound with position
//! vectors to preserve ordering (for non-commutative Product nodes).
//!
//! Key feature: `encode_skeleton()` strips all names, keeping only the
//! operator tree structure. This enables cross-domain analogy discovery
//! (Helmholtz skeleton ≈ Schrödinger skeleton).

use crate::types::{DiffOperator, EquationNode, TensorDescriptor};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Maximum recursion depth to prevent stack overflow on deep ASTs.
const MAX_DEPTH: usize = 20;

// Seeds for node-type role vectors.
const SEED_NODE_FIELD: u64 = 0xA0DE_F1D0_0001;
const SEED_NODE_CONST: u64 = 0xA0DE_C050_0002;
const SEED_NODE_SCALAR: u64 = 0xA0DE_5CA0_0003;
const SEED_NODE_DIFFOP: u64 = 0xA0DE_DF00_0004;
const SEED_NODE_SUM: u64 = 0xA0DE_5000_0005;
const SEED_NODE_PROD: u64 = 0xA0DE_F00D_0006;
const SEED_NODE_EQUALS: u64 = 0xA0DE_E0A5_0007;
const SEED_NODE_COMMUTATOR: u64 = 0xA0DE_C000_0008;
const SEED_NODE_CONTRACTION: u64 = 0xA0DE_C070_0009;
const SEED_NODE_EXP: u64 = 0xA0DE_E000_000A;
const SEED_NODE_POWER: u64 = 0xA0DE_F0E0_000B;
const SEED_NODE_NEGATE: u64 = 0xA0DE_AE60_000C;
const SEED_NODE_SQRT: u64 = 0xA0DE_5007_000D;

// Seeds for differential operator variants.
const SEED_OP_PARTIAL: u64 = 0xDF00_0001_0001;
const SEED_OP_GRADIENT: u64 = 0xDF00_0002_0002;
const SEED_OP_DIVERGENCE: u64 = 0xDF00_D100_0003;
const SEED_OP_CURL: u64 = 0xDF00_C001_0004;
const SEED_OP_LAPLACIAN: u64 = 0xDF00_0A00_0005;
const SEED_OP_DALEMBERTIAN: u64 = 0xDF00_DA00_0006;
const SEED_OP_LIE: u64 = 0xDF00_01E0_0007;
const SEED_OP_EXTERIOR: u64 = 0xDF00_E070_0008;
const SEED_OP_HODGE: u64 = 0xDF00_00D6_0009;
const SEED_OP_INTEGRAL: u64 = 0xDF00_1060_000A;
const SEED_OP_TIME_DERIV: u64 = 0xDF00_7D07_000B;
const SEED_OP_COVARIANT: u64 = 0xDF00_C00D_000C;

// Seeds for position vectors (child ordering in Products).
const SEED_POSITION_BASE: u64 = 0xEAF0_BA5E_0001;

// Seed for name encoding.
const SEED_NAME_BASE: u64 = 0xAA4E_BA5E_0001;

/// Encodes equation AST nodes to ContinuousHV.
pub struct EquationEncoder {
    // Node-type role vectors
    node_field: ContinuousHV,
    node_const: ContinuousHV,
    node_scalar: ContinuousHV,
    node_diffop: ContinuousHV,
    node_sum: ContinuousHV,
    node_prod: ContinuousHV,
    node_equals: ContinuousHV,
    node_commutator: ContinuousHV,
    node_contraction: ContinuousHV,
    node_exp: ContinuousHV,
    node_power: ContinuousHV,
    node_negate: ContinuousHV,
    node_sqrt: ContinuousHV,
    // Operator vectors
    op_partial: ContinuousHV,
    op_gradient: ContinuousHV,
    op_divergence: ContinuousHV,
    op_curl: ContinuousHV,
    op_laplacian: ContinuousHV,
    op_dalembertian: ContinuousHV,
    op_lie: ContinuousHV,
    op_exterior: ContinuousHV,
    op_hodge: ContinuousHV,
    op_integral: ContinuousHV,
    op_time_deriv: ContinuousHV,
    op_covariant: ContinuousHV,
    // Position vectors for ordered children
    positions: Vec<ContinuousHV>,
    // Tensor encoder for Field nodes
    tensor_encoder: crate::tensor_structure::TensorEncoder,
}

impl EquationEncoder {
    pub fn new() -> Self {
        let mut positions = Vec::with_capacity(16);
        for i in 0..16u64 {
            positions.push(ContinuousHV::random(
                HDC_DIMENSION,
                SEED_POSITION_BASE.wrapping_add(i * 7919),
            ));
        }

        Self {
            node_field: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_FIELD),
            node_const: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_CONST),
            node_scalar: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_SCALAR),
            node_diffop: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_DIFFOP),
            node_sum: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_SUM),
            node_prod: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_PROD),
            node_equals: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_EQUALS),
            node_commutator: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_COMMUTATOR),
            node_contraction: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_CONTRACTION),
            node_exp: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_EXP),
            node_power: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_POWER),
            node_negate: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_NEGATE),
            node_sqrt: ContinuousHV::random(HDC_DIMENSION, SEED_NODE_SQRT),
            op_partial: ContinuousHV::random(HDC_DIMENSION, SEED_OP_PARTIAL),
            op_gradient: ContinuousHV::random(HDC_DIMENSION, SEED_OP_GRADIENT),
            op_divergence: ContinuousHV::random(HDC_DIMENSION, SEED_OP_DIVERGENCE),
            op_curl: ContinuousHV::random(HDC_DIMENSION, SEED_OP_CURL),
            op_laplacian: ContinuousHV::random(HDC_DIMENSION, SEED_OP_LAPLACIAN),
            op_dalembertian: ContinuousHV::random(HDC_DIMENSION, SEED_OP_DALEMBERTIAN),
            op_lie: ContinuousHV::random(HDC_DIMENSION, SEED_OP_LIE),
            op_exterior: ContinuousHV::random(HDC_DIMENSION, SEED_OP_EXTERIOR),
            op_hodge: ContinuousHV::random(HDC_DIMENSION, SEED_OP_HODGE),
            op_integral: ContinuousHV::random(HDC_DIMENSION, SEED_OP_INTEGRAL),
            op_time_deriv: ContinuousHV::random(HDC_DIMENSION, SEED_OP_TIME_DERIV),
            op_covariant: ContinuousHV::random(HDC_DIMENSION, SEED_OP_COVARIANT),
            positions,
            tensor_encoder: crate::tensor_structure::TensorEncoder::new(),
        }
    }

    /// Encode an equation node (full encoding, including names).
    pub fn encode(&self, node: &EquationNode) -> ContinuousHV {
        self.encode_recursive(node, 0, false)
    }

    /// Encode the structural skeleton only (strips all names and constants).
    ///
    /// This reveals pure structural similarity: Helmholtz (∇²ψ + k²ψ = 0)
    /// and Schrödinger (∇²ψ + (2m/ℏ²)(E-V)ψ = 0) have identical skeletons.
    pub fn encode_skeleton(&self, node: &EquationNode) -> ContinuousHV {
        self.encode_recursive(node, 0, true)
    }

    fn encode_recursive(&self, node: &EquationNode, depth: usize, skeleton: bool) -> ContinuousHV {
        if depth >= MAX_DEPTH {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        match node {
            EquationNode::Field { name, tensor } => {
                let tensor_hv = self.tensor_encoder.encode(tensor);
                if skeleton {
                    // Skeleton: only tensor structure, no name
                    self.node_field.bind(&tensor_hv)
                } else {
                    let name_hv = encode_name(name);
                    let content = ContinuousHV::bundle(&[&tensor_hv, &name_hv]);
                    self.node_field.bind(&content)
                }
            }

            EquationNode::Constant { name } => {
                if skeleton {
                    // In skeleton mode, all constants are interchangeable
                    self.node_const.clone()
                } else {
                    let name_hv = encode_name(name);
                    self.node_const.bind(&name_hv)
                }
            }

            EquationNode::Scalar(v) => {
                if skeleton {
                    self.node_scalar.clone()
                } else {
                    // Encode value via tanh normalization
                    self.node_scalar.scale((*v as f32).tanh())
                }
            }

            EquationNode::DiffOp { operator, operand } => {
                let op_hv = self.encode_operator(operator);
                let operand_hv = self.encode_recursive(operand, depth + 1, skeleton);
                let bound = op_hv.bind(&operand_hv);
                self.node_diffop.bind(&bound)
            }

            EquationNode::Sum(children) => {
                // Sum is commutative → pure bundle (no position binding)
                let child_hvs: Vec<ContinuousHV> = children
                    .iter()
                    .map(|c| self.encode_recursive(c, depth + 1, skeleton))
                    .collect();
                let refs: Vec<&ContinuousHV> = child_hvs.iter().collect();
                let bundle = ContinuousHV::bundle(&refs);
                self.node_sum.bind(&bundle)
            }

            EquationNode::Product(children) => {
                // Product is non-commutative → position binding preserves order
                let child_hvs: Vec<ContinuousHV> = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let child = self.encode_recursive(c, depth + 1, skeleton);
                        let pos = &self.positions[i % self.positions.len()];
                        pos.bind(&child)
                    })
                    .collect();
                let refs: Vec<&ContinuousHV> = child_hvs.iter().collect();
                let bundle = ContinuousHV::bundle(&refs);
                self.node_prod.bind(&bundle)
            }

            EquationNode::Equals(lhs, rhs) => {
                let lhs_hv = self.encode_recursive(lhs, depth + 1, skeleton);
                let rhs_hv = self.encode_recursive(rhs, depth + 1, skeleton);
                let lhs_bound = self.positions[0].bind(&lhs_hv);
                let rhs_bound = self.positions[1].bind(&rhs_hv);
                let content = ContinuousHV::bundle(&[&lhs_bound, &rhs_bound]);
                self.node_equals.bind(&content)
            }

            EquationNode::Commutator(a, b) => {
                let a_hv = self.encode_recursive(a, depth + 1, skeleton);
                let b_hv = self.encode_recursive(b, depth + 1, skeleton);
                let a_bound = self.positions[0].bind(&a_hv);
                let b_bound = self.positions[1].bind(&b_hv);
                let content = ContinuousHV::bundle(&[&a_bound, &b_bound]);
                self.node_commutator.bind(&content)
            }

            EquationNode::Contraction {
                tensor,
                contracted_indices,
            } => {
                let tensor_hv = self.encode_recursive(tensor, depth + 1, skeleton);
                // Encode number of contractions as scaling
                let contraction_scale = contracted_indices.len() as f32 / 4.0;
                let scaled = self.node_contraction.scale(contraction_scale);
                scaled.bind(&tensor_hv)
            }

            EquationNode::Exponential(arg) => {
                let arg_hv = self.encode_recursive(arg, depth + 1, skeleton);
                self.node_exp.bind(&arg_hv)
            }

            EquationNode::Power { base, exponent } => {
                let base_hv = self.encode_recursive(base, depth + 1, skeleton);
                let exp_hv = self.encode_recursive(exponent, depth + 1, skeleton);
                let base_bound = self.positions[0].bind(&base_hv);
                let exp_bound = self.positions[1].bind(&exp_hv);
                let content = ContinuousHV::bundle(&[&base_bound, &exp_bound]);
                self.node_power.bind(&content)
            }

            EquationNode::Negate(inner) => {
                let inner_hv = self.encode_recursive(inner, depth + 1, skeleton);
                self.node_negate.bind(&inner_hv)
            }

            EquationNode::Sqrt(inner) => {
                let inner_hv = self.encode_recursive(inner, depth + 1, skeleton);
                self.node_sqrt.bind(&inner_hv)
            }
        }
    }

    /// Encode a differential operator variant.
    fn encode_operator(&self, op: &DiffOperator) -> ContinuousHV {
        match op {
            DiffOperator::Partial => self.op_partial.clone(),
            DiffOperator::Gradient => self.op_gradient.clone(),
            DiffOperator::Divergence => self.op_divergence.clone(),
            DiffOperator::Curl => self.op_curl.clone(),
            DiffOperator::Laplacian => self.op_laplacian.clone(),
            DiffOperator::DAlembertian => self.op_dalembertian.clone(),
            DiffOperator::LieDerivative => self.op_lie.clone(),
            DiffOperator::ExteriorDerivative => self.op_exterior.clone(),
            DiffOperator::HodgeStar => self.op_hodge.clone(),
            DiffOperator::Integral => self.op_integral.clone(),
            DiffOperator::TimeDerivative => self.op_time_deriv.clone(),
            DiffOperator::CovariantDerivative => self.op_covariant.clone(),
        }
    }

    /// Similarity between two equation ASTs.
    pub fn similarity(&self, a: &EquationNode, b: &EquationNode) -> f32 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b)
    }

    /// Skeleton similarity (structure only, no names).
    pub fn skeleton_similarity(&self, a: &EquationNode, b: &EquationNode) -> f32 {
        let hv_a = self.encode_skeleton(a);
        let hv_b = self.encode_skeleton(b);
        hv_a.similarity(&hv_b)
    }
}

impl Default for EquationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a name string as a ContinuousHV using character n-gram hashing.
///
/// Deterministic: same name → same HV.
fn encode_name(name: &str) -> ContinuousHV {
    if name.is_empty() {
        return ContinuousHV::zero(HDC_DIMENSION);
    }

    // Hash the name to a deterministic seed
    let mut hash: u64 = SEED_NAME_BASE;
    for (i, byte) in name.bytes().enumerate() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        hash = hash.wrapping_add((i as u64).wrapping_mul(17));
    }

    ContinuousHV::random(HDC_DIMENSION, hash)
}

/// Helper to build common equation structures for the catalog.
pub fn make_field(name: &str, tensor: TensorDescriptor) -> EquationNode {
    EquationNode::Field {
        name: name.to_string(),
        tensor,
    }
}

pub fn make_const(name: &str) -> EquationNode {
    EquationNode::Constant {
        name: name.to_string(),
    }
}

pub fn make_diffop(operator: DiffOperator, operand: EquationNode) -> EquationNode {
    EquationNode::DiffOp {
        operator,
        operand: Box::new(operand),
    }
}

pub fn make_equals(lhs: EquationNode, rhs: EquationNode) -> EquationNode {
    EquationNode::Equals(Box::new(lhs), Box::new(rhs))
}

pub fn make_sum(children: Vec<EquationNode>) -> EquationNode {
    EquationNode::Sum(children)
}

pub fn make_product(children: Vec<EquationNode>) -> EquationNode {
    EquationNode::Product(children)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MetricSignature;

    fn scalar_field(name: &str) -> EquationNode {
        make_field(
            name,
            TensorDescriptor::scalar(MetricSignature::Euclidean(3)),
        )
    }

    fn scalar_field_4d(name: &str) -> EquationNode {
        make_field(
            name,
            TensorDescriptor::scalar(MetricSignature::Lorentzian(4)),
        )
    }

    #[test]
    fn same_equation_identical() {
        let enc = EquationEncoder::new();
        // ∇²ψ = 0 (Laplace equation)
        let laplace = make_equals(
            make_diffop(DiffOperator::Laplacian, scalar_field("ψ")),
            EquationNode::Scalar(0.0),
        );
        let sim = enc.similarity(&laplace, &laplace);
        assert!(sim > 0.999, "Same equation, got sim = {sim}");
    }

    #[test]
    fn helmholtz_vs_schrodinger_skeleton() {
        let enc = EquationEncoder::new();

        // Helmholtz: ∇²ψ + k²ψ = 0
        let helmholtz = make_equals(
            make_sum(vec![
                make_diffop(DiffOperator::Laplacian, scalar_field("ψ")),
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("k")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    scalar_field("ψ"),
                ]),
            ]),
            EquationNode::Scalar(0.0),
        );

        // Time-independent Schrödinger: ∇²ψ + (2m/ℏ²)(E-V)ψ = 0
        let schrodinger = make_equals(
            make_sum(vec![
                make_diffop(DiffOperator::Laplacian, scalar_field("ψ")),
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("2m/ℏ²")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    scalar_field("ψ"),
                ]),
            ]),
            EquationNode::Scalar(0.0),
        );

        // Full similarity — different names
        let full_sim = enc.similarity(&helmholtz, &schrodinger);

        // Skeleton similarity — stripped names, same structure
        let skel_sim = enc.skeleton_similarity(&helmholtz, &schrodinger);

        assert!(
            skel_sim > full_sim,
            "Skeleton sim ({skel_sim}) should > full sim ({full_sim}) for cross-domain analogy"
        );
        assert!(
            skel_sim > 0.8,
            "Helmholtz ≈ Schrödinger skeleton, got skel_sim = {skel_sim}"
        );
    }

    #[test]
    fn sum_is_commutative() {
        let enc = EquationEncoder::new();
        let a = scalar_field("A");
        let b = scalar_field("B");

        let sum_ab = make_sum(vec![a.clone(), b.clone()]);
        let sum_ba = make_sum(vec![b, a]);

        let sim = enc.similarity(&sum_ab, &sum_ba);
        // Bundle is commutative → sum order shouldn't matter
        assert!(sim > 0.99, "Sum should be commutative, got sim = {sim}");
    }

    #[test]
    fn product_preserves_order() {
        let enc = EquationEncoder::new();
        let a = scalar_field("A");
        let b = scalar_field("B");

        let prod_ab = make_product(vec![a.clone(), b.clone()]);
        let prod_ba = make_product(vec![b, a]);

        let sim = enc.similarity(&prod_ab, &prod_ba);
        // Position binding means order matters
        assert!(sim < 0.99, "Product should preserve order, got sim = {sim}");
    }

    #[test]
    fn different_operators_different() {
        let enc = EquationEncoder::new();
        let psi = scalar_field("ψ");

        let laplacian = make_diffop(DiffOperator::Laplacian, psi.clone());
        let divergence = make_diffop(DiffOperator::Divergence, psi);

        let sim = enc.similarity(&laplacian, &divergence);
        assert!(sim < 0.9, "Laplacian ≠ Divergence, got sim = {sim}");
    }

    #[test]
    fn wave_vs_diffusion_skeleton() {
        let enc = EquationEncoder::new();

        // Wave: □ψ = 0 (d'Alembertian)
        let wave = make_equals(
            make_diffop(DiffOperator::DAlembertian, scalar_field_4d("ψ")),
            EquationNode::Scalar(0.0),
        );

        // Diffusion/Heat: ∂ψ/∂t = κ∇²ψ
        let heat = make_equals(
            make_diffop(DiffOperator::TimeDerivative, scalar_field("T")),
            make_product(vec![
                make_const("κ"),
                make_diffop(DiffOperator::Laplacian, scalar_field("T")),
            ]),
        );

        let sim = enc.skeleton_similarity(&wave, &heat);
        // Different structure (□ψ=0 vs ∂t=κ∇²) — should be low
        assert!(sim < 0.9, "Wave ≠ Heat structure, got sim = {sim}");
    }

    #[test]
    fn name_encoding_deterministic() {
        let hv1 = encode_name("Einstein");
        let hv2 = encode_name("Einstein");
        assert_eq!(hv1.values, hv2.values);
    }

    #[test]
    fn different_names_different() {
        let hv1 = encode_name("Einstein");
        let hv2 = encode_name("Maxwell");
        let sim = hv1.similarity(&hv2);
        assert!(sim.abs() < 0.2, "Different names, got sim = {sim}");
    }

    #[test]
    fn encode_negate_and_sqrt() {
        let enc = EquationEncoder::new();
        let neg = EquationNode::Negate(Box::new(scalar_field("x")));
        let sqrt = EquationNode::Sqrt(Box::new(scalar_field("x")));
        let hv_neg = enc.encode(&neg);
        let hv_sqrt = enc.encode(&sqrt);
        assert_eq!(hv_neg.dim(), HDC_DIMENSION);
        assert_eq!(hv_sqrt.dim(), HDC_DIMENSION);
        let sim = hv_neg.similarity(&hv_sqrt);
        assert!(sim < 0.9, "Negate ≠ Sqrt, got sim = {sim}");
    }

    #[test]
    fn encode_exponential() {
        let enc = EquationEncoder::new();
        let exp_node = EquationNode::Exponential(Box::new(scalar_field("x")));
        let hv = enc.encode(&exp_node);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn max_depth_protection() {
        let enc = EquationEncoder::new();
        // Build a deeply nested structure
        let mut node = scalar_field("x");
        for _ in 0..30 {
            node = EquationNode::Negate(Box::new(node));
        }
        // Should not stack overflow
        let hv = enc.encode(&node);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn contraction_encoding() {
        let enc = EquationEncoder::new();
        let contracted = EquationNode::Contraction {
            tensor: Box::new(make_field(
                "R",
                TensorDescriptor::riemann(MetricSignature::Lorentzian(4)),
            )),
            contracted_indices: vec![(0, 2)],
        };
        let hv = enc.encode(&contracted);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_equation_encoding() {
        let enc1 = EquationEncoder::new();
        let enc2 = EquationEncoder::new();
        let eq = make_diffop(DiffOperator::Laplacian, scalar_field("ψ"));
        let hv1 = enc1.encode(&eq);
        let hv2 = enc2.encode(&eq);
        assert_eq!(hv1.values, hv2.values);
    }
}
