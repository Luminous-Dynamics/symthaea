// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;

/// An element in an algebraic structure with a label and HDC encoding
#[derive(Clone)]
pub struct AlgebraicElement {
    pub label: String,
    pub encoding: BinaryHV,
}

/// Trait for binary operations on algebraic elements
pub trait BinaryOperation {
    fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement;
    fn name(&self) -> &str;
}

/// Result of checking a single axiom
#[derive(Debug, Clone)]
pub struct AxiomCheck {
    pub axiom: String,
    pub holds: bool,
    pub counterexample: Option<String>,
    pub phi: f64,
}

/// Result of verifying an algebraic structure
#[derive(Debug)]
pub struct StructureVerification {
    pub structure_name: String,
    pub axioms_checked: Vec<AxiomCheck>,
    pub is_valid: bool,
    pub total_phi: f64,
}

impl StructureVerification {
    fn new(structure_name: String) -> Self {
        Self {
            structure_name,
            axioms_checked: Vec::new(),
            is_valid: true,
            total_phi: 0.0,
        }
    }

    fn add_check(&mut self, check: AxiomCheck) {
        if !check.holds {
            self.is_valid = false;
        }
        self.axioms_checked.push(check);
    }

    fn finalize(&mut self) {
        if !self.axioms_checked.is_empty() {
            self.total_phi = self.axioms_checked.iter().map(|c| c.phi).sum::<f64>()
                / self.axioms_checked.len() as f64;
        }
    }
}

/// Verifier for group structures
pub struct GroupVerifier;

impl GroupVerifier {
    const SIMILARITY_THRESHOLD: f32 = 0.7;

    pub fn verify(
        elements: &[AlgebraicElement],
        op: &dyn BinaryOperation,
    ) -> StructureVerification {
        let mut result = StructureVerification::new(format!("Group under {}", op.name()));

        // Check closure
        let closure_check = Self::check_closure(elements, op);
        result.add_check(closure_check);

        // Check associativity
        let assoc_check = Self::check_associativity(elements, op);
        result.add_check(assoc_check);

        // Check identity
        let (identity_check, identity_elem) = Self::check_identity(elements, op);
        result.add_check(identity_check);

        // Check inverse (if identity exists)
        if let Some(identity) = identity_elem {
            let inverse_check = Self::check_inverse(elements, op, &identity);
            result.add_check(inverse_check);
        }

        result.finalize();
        result
    }

    fn find_closest_element(target: &BinaryHV, elements: &[AlgebraicElement]) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_sim = 0.0f32;

        for (i, elem) in elements.iter().enumerate() {
            let sim = target.similarity(&elem.encoding);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (best_idx, best_sim)
    }

    fn check_closure(elements: &[AlgebraicElement], op: &dyn BinaryOperation) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        for a in elements {
            for b in elements {
                let result = op.apply(a, b);
                let (_closest_idx, sim) = Self::find_closest_element(&result.encoding, elements);

                total_sim += sim;
                count += 1;

                if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                    holds = false;
                    counterexample = Some(format!(
                        "{} {} {} not closed (similarity: {:.3})",
                        a.label,
                        op.name(),
                        b.label,
                        sim
                    ));
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: "Closure".to_string(),
            holds,
            counterexample,
            phi,
        }
    }

    fn check_associativity(elements: &[AlgebraicElement], op: &dyn BinaryOperation) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        // Sample a subset for efficiency (full check is O(n³))
        let sample_size = elements.len().min(5);
        for a in elements.iter().take(sample_size) {
            for b in elements.iter().take(sample_size) {
                for c in elements.iter().take(sample_size) {
                    let ab = op.apply(a, b);
                    let ab_c = op.apply(&ab, c);

                    let bc = op.apply(b, c);
                    let a_bc = op.apply(a, &bc);

                    let sim = ab_c.encoding.similarity(&a_bc.encoding);
                    total_sim += sim;
                    count += 1;

                    if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                        holds = false;
                        counterexample = Some(format!(
                            "({} {} {}) {} {} != {} {} ({} {} {}) (similarity: {:.3})",
                            a.label,
                            op.name(),
                            b.label,
                            op.name(),
                            c.label,
                            a.label,
                            op.name(),
                            b.label,
                            op.name(),
                            c.label,
                            sim
                        ));
                    }
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: "Associativity".to_string(),
            holds,
            counterexample,
            phi,
        }
    }

    fn check_identity(
        elements: &[AlgebraicElement],
        op: &dyn BinaryOperation,
    ) -> (AxiomCheck, Option<AlgebraicElement>) {
        let mut best_identity = None;
        let mut best_avg_sim = 0.0f32;

        for candidate in elements {
            let mut total_sim = 0.0f32;
            let mut valid = true;

            for a in elements {
                let ea = op.apply(candidate, a);
                let ae = op.apply(a, candidate);

                let sim_ea = ea.encoding.similarity(&a.encoding);
                let sim_ae = ae.encoding.similarity(&a.encoding);

                total_sim += sim_ea + sim_ae;

                if sim_ea < Self::SIMILARITY_THRESHOLD || sim_ae < Self::SIMILARITY_THRESHOLD {
                    valid = false;
                }
            }

            let avg_sim = total_sim / (2.0 * elements.len() as f32);
            if valid && avg_sim > best_avg_sim {
                best_avg_sim = avg_sim;
                best_identity = Some(candidate.clone());
            }
        }

        let check = if let Some(ref identity) = best_identity {
            AxiomCheck {
                axiom: "Identity".to_string(),
                holds: true,
                counterexample: None,
                phi: best_avg_sim as f64,
            }
        } else {
            AxiomCheck {
                axiom: "Identity".to_string(),
                holds: false,
                counterexample: Some("No identity element found".to_string()),
                phi: best_avg_sim as f64,
            }
        };

        (check, best_identity)
    }

    fn check_inverse(
        elements: &[AlgebraicElement],
        op: &dyn BinaryOperation,
        identity: &AlgebraicElement,
    ) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        for a in elements {
            let mut found_inverse = false;
            let mut best_sim = 0.0f32;

            for b in elements {
                let ab = op.apply(a, b);
                let ba = op.apply(b, a);

                let sim_ab = ab.encoding.similarity(&identity.encoding);
                let sim_ba = ba.encoding.similarity(&identity.encoding);

                let sim = (sim_ab + sim_ba) / 2.0;
                best_sim = best_sim.max(sim);

                if sim >= Self::SIMILARITY_THRESHOLD {
                    found_inverse = true;
                    total_sim += sim;
                    count += 1;
                    break;
                }
            }

            if !found_inverse && counterexample.is_none() {
                holds = false;
                counterexample = Some(format!(
                    "No inverse found for {} (best similarity: {:.3})",
                    a.label, best_sim
                ));
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: "Inverse".to_string(),
            holds,
            counterexample,
            phi,
        }
    }
}

/// Verifier for ring structures
pub struct RingVerifier;

impl RingVerifier {
    const SIMILARITY_THRESHOLD: f32 = 0.7;

    pub fn verify(
        elements: &[AlgebraicElement],
        add_op: &dyn BinaryOperation,
        mul_op: &dyn BinaryOperation,
    ) -> StructureVerification {
        let mut result = StructureVerification::new(format!(
            "Ring under ({}, {})",
            add_op.name(),
            mul_op.name()
        ));

        // Check (S, +) is an abelian group
        let group_result = GroupVerifier::verify(elements, add_op);
        for check in group_result.axioms_checked {
            result.add_check(check);
        }

        // Check commutativity of addition
        let comm_check = Self::check_commutativity(elements, add_op);
        result.add_check(comm_check);

        // Check (S, *) is a monoid (closure, associativity, identity)
        let mul_closure = Self::check_closure(elements, mul_op);
        result.add_check(mul_closure);

        let mul_assoc = Self::check_associativity_sample(elements, mul_op);
        result.add_check(mul_assoc);

        let mul_identity = Self::check_mul_identity(elements, mul_op);
        result.add_check(mul_identity);

        // Check distributivity
        let dist_check = Self::check_distributivity(elements, add_op, mul_op);
        result.add_check(dist_check);

        result.finalize();
        result
    }

    fn check_commutativity(elements: &[AlgebraicElement], op: &dyn BinaryOperation) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        for a in elements {
            for b in elements {
                let ab = op.apply(a, b);
                let ba = op.apply(b, a);

                let sim = ab.encoding.similarity(&ba.encoding);
                total_sim += sim;
                count += 1;

                if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                    holds = false;
                    counterexample = Some(format!(
                        "{} {} {} != {} {} {} (similarity: {:.3})",
                        a.label,
                        op.name(),
                        b.label,
                        b.label,
                        op.name(),
                        a.label,
                        sim
                    ));
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: format!("Commutativity ({})", op.name()),
            holds,
            counterexample,
            phi,
        }
    }

    fn check_closure(elements: &[AlgebraicElement], op: &dyn BinaryOperation) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        for a in elements {
            for b in elements {
                let result = op.apply(a, b);
                let (_closest_idx, sim) =
                    GroupVerifier::find_closest_element(&result.encoding, elements);

                total_sim += sim;
                count += 1;

                if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                    holds = false;
                    counterexample = Some(format!(
                        "{} {} {} not closed (similarity: {:.3})",
                        a.label,
                        op.name(),
                        b.label,
                        sim
                    ));
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: format!("Closure ({})", op.name()),
            holds,
            counterexample,
            phi,
        }
    }

    fn check_associativity_sample(
        elements: &[AlgebraicElement],
        op: &dyn BinaryOperation,
    ) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        let sample_size = elements.len().min(5);
        for a in elements.iter().take(sample_size) {
            for b in elements.iter().take(sample_size) {
                for c in elements.iter().take(sample_size) {
                    let ab = op.apply(a, b);
                    let ab_c = op.apply(&ab, c);

                    let bc = op.apply(b, c);
                    let a_bc = op.apply(a, &bc);

                    let sim = ab_c.encoding.similarity(&a_bc.encoding);
                    total_sim += sim;
                    count += 1;

                    if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                        holds = false;
                        counterexample = Some(format!(
                            "({} {} {}) {} {} != {} {} ({} {} {}) (similarity: {:.3})",
                            a.label,
                            op.name(),
                            b.label,
                            op.name(),
                            c.label,
                            a.label,
                            op.name(),
                            b.label,
                            op.name(),
                            c.label,
                            sim
                        ));
                    }
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: format!("Associativity ({})", op.name()),
            holds,
            counterexample,
            phi,
        }
    }

    fn check_mul_identity(elements: &[AlgebraicElement], op: &dyn BinaryOperation) -> AxiomCheck {
        let mut best_avg_sim = 0.0f32;
        let mut found_identity = false;

        for candidate in elements {
            let mut total_sim = 0.0f32;
            let mut valid = true;

            for a in elements {
                let ea = op.apply(candidate, a);
                let ae = op.apply(a, candidate);

                let sim_ea = ea.encoding.similarity(&a.encoding);
                let sim_ae = ae.encoding.similarity(&a.encoding);

                total_sim += sim_ea + sim_ae;

                if sim_ea < Self::SIMILARITY_THRESHOLD || sim_ae < Self::SIMILARITY_THRESHOLD {
                    valid = false;
                }
            }

            let avg_sim = total_sim / (2.0 * elements.len() as f32);
            if valid {
                found_identity = true;
                best_avg_sim = best_avg_sim.max(avg_sim);
            }
        }

        AxiomCheck {
            axiom: format!("Identity ({})", op.name()),
            holds: found_identity,
            counterexample: if found_identity {
                None
            } else {
                Some("No multiplicative identity found".to_string())
            },
            phi: best_avg_sim as f64,
        }
    }

    fn check_distributivity(
        elements: &[AlgebraicElement],
        add_op: &dyn BinaryOperation,
        mul_op: &dyn BinaryOperation,
    ) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        let sample_size = elements.len().min(4);
        for a in elements.iter().take(sample_size) {
            for b in elements.iter().take(sample_size) {
                for c in elements.iter().take(sample_size) {
                    // a * (b + c) = a * b + a * c
                    let b_plus_c = add_op.apply(b, c);
                    let left = mul_op.apply(a, &b_plus_c);

                    let ab = mul_op.apply(a, b);
                    let ac = mul_op.apply(a, c);
                    let right = add_op.apply(&ab, &ac);

                    let sim = left.encoding.similarity(&right.encoding);
                    total_sim += sim;
                    count += 1;

                    if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                        holds = false;
                        counterexample = Some(format!(
                            "{} * ({} + {}) != {} * {} + {} * {} (similarity: {:.3})",
                            a.label, b.label, c.label, a.label, b.label, a.label, c.label, sim
                        ));
                    }
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: "Distributivity".to_string(),
            holds,
            counterexample,
            phi,
        }
    }
}

/// Verifier for field structures
pub struct FieldVerifier;

impl FieldVerifier {
    const SIMILARITY_THRESHOLD: f32 = 0.7;

    pub fn verify(
        elements: &[AlgebraicElement],
        add_op: &dyn BinaryOperation,
        mul_op: &dyn BinaryOperation,
    ) -> StructureVerification {
        let mut result = StructureVerification::new(format!(
            "Field under ({}, {})",
            add_op.name(),
            mul_op.name()
        ));

        // Check ring axioms
        let ring_result = RingVerifier::verify(elements, add_op, mul_op);
        for check in ring_result.axioms_checked {
            result.add_check(check);
        }

        // Find additive identity (zero)
        let zero = Self::find_identity(elements, add_op);

        // Check multiplicative inverses for nonzero elements
        if let Some(zero_elem) = zero {
            let mul_identity = Self::find_identity(elements, mul_op);
            if let Some(one_elem) = mul_identity {
                let inverse_check =
                    Self::check_multiplicative_inverses(elements, mul_op, &zero_elem, &one_elem);
                result.add_check(inverse_check);
            }
        }

        result.finalize();
        result
    }

    fn find_identity(
        elements: &[AlgebraicElement],
        op: &dyn BinaryOperation,
    ) -> Option<AlgebraicElement> {
        for candidate in elements {
            let mut is_identity = true;

            for a in elements {
                let ea = op.apply(candidate, a);
                let ae = op.apply(a, candidate);

                let sim_ea = ea.encoding.similarity(&a.encoding);
                let sim_ae = ae.encoding.similarity(&a.encoding);

                if sim_ea < Self::SIMILARITY_THRESHOLD || sim_ae < Self::SIMILARITY_THRESHOLD {
                    is_identity = false;
                    break;
                }
            }

            if is_identity {
                return Some(candidate.clone());
            }
        }

        None
    }

    fn check_multiplicative_inverses(
        elements: &[AlgebraicElement],
        mul_op: &dyn BinaryOperation,
        zero: &AlgebraicElement,
        one: &AlgebraicElement,
    ) -> AxiomCheck {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        for a in elements {
            // Skip if a is the zero element
            let sim_to_zero = a.encoding.similarity(&zero.encoding);
            if sim_to_zero >= Self::SIMILARITY_THRESHOLD {
                continue;
            }

            let mut found_inverse = false;
            let mut best_sim = 0.0f32;

            for b in elements {
                let ab = mul_op.apply(a, b);
                let ba = mul_op.apply(b, a);

                let sim_ab = ab.encoding.similarity(&one.encoding);
                let sim_ba = ba.encoding.similarity(&one.encoding);

                let sim = (sim_ab + sim_ba) / 2.0;
                best_sim = best_sim.max(sim);

                if sim >= Self::SIMILARITY_THRESHOLD {
                    found_inverse = true;
                    total_sim += sim;
                    count += 1;
                    break;
                }
            }

            if !found_inverse && counterexample.is_none() {
                holds = false;
                counterexample = Some(format!(
                    "No multiplicative inverse found for {} (best similarity: {:.3})",
                    a.label, best_sim
                ));
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        AxiomCheck {
            axiom: "Multiplicative Inverse".to_string(),
            holds,
            counterexample,
            phi,
        }
    }
}

/// Verifier for homomorphisms between algebraic structures
pub struct HomomorphismVerifier;

impl HomomorphismVerifier {
    const SIMILARITY_THRESHOLD: f32 = 0.7;

    pub fn verify(
        domain_elements: &[AlgebraicElement],
        codomain_elements: &[AlgebraicElement],
        map: &dyn Fn(&AlgebraicElement) -> AlgebraicElement,
        domain_op: &dyn BinaryOperation,
        codomain_op: &dyn BinaryOperation,
    ) -> StructureVerification {
        let mut result = StructureVerification::new(format!(
            "Homomorphism from {} to {}",
            domain_op.name(),
            codomain_op.name()
        ));

        let mut total_sim = 0.0f32;
        let mut count = 0;
        let mut holds = true;
        let mut counterexample = None;

        let sample_size = domain_elements.len().min(5);
        for a in domain_elements.iter().take(sample_size) {
            for b in domain_elements.iter().take(sample_size) {
                // f(a op b)
                let a_op_b = domain_op.apply(a, b);
                let f_a_op_b = map(&a_op_b);

                // f(a) op' f(b)
                let fa = map(a);
                let fb = map(b);
                let fa_op_fb = codomain_op.apply(&fa, &fb);

                let sim = f_a_op_b.encoding.similarity(&fa_op_fb.encoding);
                total_sim += sim;
                count += 1;

                if sim < Self::SIMILARITY_THRESHOLD && counterexample.is_none() {
                    holds = false;
                    counterexample = Some(format!(
                        "f({} {} {}) != f({}) {} f({}) (similarity: {:.3})",
                        a.label,
                        domain_op.name(),
                        b.label,
                        a.label,
                        codomain_op.name(),
                        b.label,
                        sim
                    ));
                }
            }
        }

        let phi = if count > 0 {
            total_sim as f64 / count as f64
        } else {
            0.0
        };

        let check = AxiomCheck {
            axiom: "Homomorphism property".to_string(),
            holds,
            counterexample,
            phi,
        };

        result.add_check(check);
        result.finalize();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple XOR group: Z/2Z = {0, 1} under XOR
    struct XorOp;
    impl BinaryOperation for XorOp {
        fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
            AlgebraicElement {
                label: format!("{}^{}", a.label, b.label),
                encoding: a.encoding.bind(&b.encoding),
            }
        }
        fn name(&self) -> &str {
            "xor"
        }
    }

    #[test]
    fn test_z2_is_group() {
        // Z/2Z under XOR: identity is zero vector, every element is its own inverse
        let zero = AlgebraicElement {
            label: "0".into(),
            encoding: BinaryHV([0u8; 2048]),
        };
        let one = AlgebraicElement {
            label: "1".into(),
            encoding: BinaryHV::random(1),
        };
        let elements = vec![zero, one];
        let op = XorOp;
        let result = GroupVerifier::verify(&elements, &op);
        // Should check all 4 axioms: closure, associativity, identity, inverse
        assert!(result.axioms_checked.len() >= 4);
        // Closure and identity should hold; associativity holds for XOR
        for check in &result.axioms_checked {
            if check.axiom == "Identity" {
                assert!(check.holds, "Identity axiom should hold with zero vector");
            }
        }
    }

    #[test]
    fn test_homomorphism_check() {
        let a = AlgebraicElement {
            label: "a".into(),
            encoding: BinaryHV::random(10),
        };
        let b = AlgebraicElement {
            label: "b".into(),
            encoding: BinaryHV::random(20),
        };
        let domain = vec![a, b];

        let fa = AlgebraicElement {
            label: "f(a)".into(),
            encoding: BinaryHV::random(10),
        };
        let fb = AlgebraicElement {
            label: "f(b)".into(),
            encoding: BinaryHV::random(20),
        };
        let codomain = vec![fa, fb];

        let identity_map = |e: &AlgebraicElement| e.clone();
        let op = XorOp;

        let result = HomomorphismVerifier::verify(&domain, &codomain, &identity_map, &op, &op);
        assert!(!result.axioms_checked.is_empty());
    }

    #[test]
    fn test_group_axiom_count() {
        let zero = AlgebraicElement {
            label: "0".into(),
            encoding: BinaryHV([0u8; 2048]),
        };
        let one = AlgebraicElement {
            label: "1".into(),
            encoding: BinaryHV::random(1),
        };
        let result = GroupVerifier::verify(&[zero, one], &XorOp);
        // Must check at least 4 axioms: closure, associativity, identity, inverse
        assert!(
            result.axioms_checked.len() >= 4,
            "Group verification should check ≥4 axioms, got {}",
            result.axioms_checked.len()
        );
        // Total phi should be non-negative
        assert!(result.total_phi >= 0.0);
    }

    #[test]
    fn test_single_element_group() {
        // Trivial group: {e} under any operation
        let e = AlgebraicElement {
            label: "e".into(),
            encoding: BinaryHV([0u8; 2048]),
        };
        let result = GroupVerifier::verify(&[e], &XorOp);
        // Single element: e*e should be ~e (closure), identity is e, inverse is e
        assert!(!result.axioms_checked.is_empty());
    }

    // Define addition-like and multiplication-like operations for ring tests
    struct BundleAddOp;
    impl BinaryOperation for BundleAddOp {
        fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
            AlgebraicElement {
                label: format!("({}+{})", a.label, b.label),
                encoding: BinaryHV::bundle(&[a.encoding.clone(), b.encoding.clone()]),
            }
        }
        fn name(&self) -> &str {
            "bundle_add"
        }
    }

    struct BindMulOp;
    impl BinaryOperation for BindMulOp {
        fn apply(&self, a: &AlgebraicElement, b: &AlgebraicElement) -> AlgebraicElement {
            AlgebraicElement {
                label: format!("({}*{})", a.label, b.label),
                encoding: a.encoding.bind(&b.encoding),
            }
        }
        fn name(&self) -> &str {
            "bind_mul"
        }
    }

    #[test]
    fn test_ring_verifier_checks_distributivity() {
        let elements: Vec<AlgebraicElement> = (0..3)
            .map(|i| AlgebraicElement {
                label: format!("{}", i),
                encoding: BinaryHV::random(i as u64 + 100),
            })
            .collect();

        let result = RingVerifier::verify(&elements, &BundleAddOp, &BindMulOp);
        // Should have axioms from group + ring checks including distributivity
        let axiom_names: Vec<&str> = result
            .axioms_checked
            .iter()
            .map(|a| a.axiom.as_str())
            .collect();
        assert!(
            axiom_names.iter().any(|n| n.contains("Distribut")),
            "Ring verification should check distributivity, got: {:?}",
            axiom_names
        );
    }

    #[test]
    fn test_field_verifier_runs() {
        let elements: Vec<AlgebraicElement> = (0..3)
            .map(|i| AlgebraicElement {
                label: format!("{}", i),
                encoding: BinaryHV::random(i as u64 + 200),
            })
            .collect();

        let result = FieldVerifier::verify(&elements, &BundleAddOp, &BindMulOp);
        // Should check more axioms than a ring (multiplicative inverse)
        assert!(
            result.axioms_checked.len() >= 6,
            "Field should check many axioms, got {}",
            result.axioms_checked.len()
        );
    }

    #[test]
    fn test_structure_verification_phi_nonnegative() {
        let elements: Vec<AlgebraicElement> = (0..4)
            .map(|i| AlgebraicElement {
                label: format!("e{}", i),
                encoding: BinaryHV::random(i as u64 + 300),
            })
            .collect();

        let result = GroupVerifier::verify(&elements, &XorOp);
        assert!(
            result.total_phi >= 0.0,
            "Total Phi should be non-negative: {}",
            result.total_phi
        );
        for check in &result.axioms_checked {
            assert!(
                check.phi >= 0.0,
                "Axiom Phi should be non-negative: {} = {}",
                check.axiom,
                check.phi
            );
        }
    }

    #[test]
    fn test_axiom_check_counterexample_on_failure() {
        // When an axiom fails, counterexample should be populated
        let elements: Vec<AlgebraicElement> = (0..3)
            .map(|i| AlgebraicElement {
                label: format!("{}", i),
                encoding: BinaryHV::random(i as u64 + 400),
            })
            .collect();

        let result = GroupVerifier::verify(&elements, &XorOp);
        // For random elements, some axioms may fail
        for check in &result.axioms_checked {
            if !check.holds {
                // Failed axioms should have counterexamples
                assert!(
                    check.counterexample.is_some(),
                    "Failed axiom '{}' should have a counterexample",
                    check.axiom
                );
            }
        }
    }
}
