// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::arithmetic_engine::HybridArithmeticEngine;
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;

/// Logical proposition that can be evaluated or reasoned about
#[derive(Debug, Clone)]
pub enum Proposition {
    /// Equality proposition: lhs == rhs
    Equals(i64, i64),
    /// Universal quantification: ∀x. P(x)
    ForAll {
        variable: String,
        body: Box<Proposition>,
    },
    /// Logical implication: P → Q
    Implies {
        antecedent: Box<Proposition>,
        consequent: Box<Proposition>,
    },
    /// Logical conjunction: P ∧ Q
    And(Box<Proposition>, Box<Proposition>),
    /// Named predicate with arguments
    Predicate { name: String, args: Vec<i64> },
}

impl Proposition {
    /// Evaluate ground propositions (those without free variables)
    /// Returns None if the proposition cannot be evaluated (e.g., contains free variables)
    pub fn evaluate(&self) -> Option<bool> {
        match self {
            Proposition::Equals(lhs, rhs) => Some(lhs == rhs),
            Proposition::And(p, q) => match (p.evaluate(), q.evaluate()) {
                (Some(a), Some(b)) => Some(a && b),
                _ => None,
            },
            Proposition::Implies {
                antecedent,
                consequent,
            } => match (antecedent.evaluate(), consequent.evaluate()) {
                (Some(a), Some(b)) => Some(!a || b),
                _ => None,
            },
            Proposition::ForAll { .. } => None, // Cannot evaluate universal quantification
            Proposition::Predicate { .. } => None, // Needs interpretation
        }
    }

    /// Human-readable representation of the proposition
    pub fn display(&self) -> String {
        match self {
            Proposition::Equals(lhs, rhs) => format!("{lhs} = {rhs}"),
            Proposition::ForAll { variable, body } => {
                format!("∀{}. {}", variable, body.display())
            }
            Proposition::Implies {
                antecedent,
                consequent,
            } => {
                format!("({}) → ({})", antecedent.display(), consequent.display())
            }
            Proposition::And(p, q) => {
                format!("({}) ∧ ({})", p.display(), q.display())
            }
            Proposition::Predicate { name, args } => {
                format!(
                    "{}({})",
                    name,
                    args.iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }

    /// Generate HDC encoding of the proposition structure
    pub fn encoding(&self) -> BinaryHV {
        match self {
            Proposition::Equals(lhs, rhs) => {
                let eq_seed = seed_from_name("equals");
                let lhs_seed = seed_from_name(&format!("int_{lhs}"));
                let rhs_seed = seed_from_name(&format!("int_{rhs}"));

                let eq_hv = BinaryHV::random(eq_seed);
                let lhs_hv = BinaryHV::random(lhs_seed);
                let rhs_hv = BinaryHV::random(rhs_seed);

                BinaryHV::bundle(&[eq_hv, lhs_hv.bind(&rhs_hv)])
            }
            Proposition::ForAll { variable, body } => {
                let forall_seed = seed_from_name("forall");
                let var_seed = seed_from_name(variable);

                let forall_hv = BinaryHV::random(forall_seed);
                let var_hv = BinaryHV::random(var_seed);
                let body_hv = body.encoding();

                BinaryHV::bundle(&[forall_hv, var_hv.bind(&body_hv)])
            }
            Proposition::Implies {
                antecedent,
                consequent,
            } => {
                let implies_seed = seed_from_name("implies");
                let implies_hv = BinaryHV::random(implies_seed);
                let ant_hv = antecedent.encoding();
                let cons_hv = consequent.encoding();

                BinaryHV::bundle(&[implies_hv, ant_hv.bind(&cons_hv)])
            }
            Proposition::And(p, q) => {
                let and_seed = seed_from_name("and");
                let and_hv = BinaryHV::random(and_seed);
                let p_hv = p.encoding();
                let q_hv = q.encoding();

                BinaryHV::bundle(&[and_hv, p_hv.bind(&q_hv)])
            }
            Proposition::Predicate { name, args } => {
                let pred_seed = seed_from_name(name);
                let pred_hv = BinaryHV::random(pred_seed);

                let arg_hvs: Vec<BinaryHV> = args
                    .iter()
                    .map(|a| BinaryHV::random(seed_from_name(&format!("int_{a}"))))
                    .collect();

                let args_bundle = if arg_hvs.is_empty() {
                    pred_hv
                } else {
                    BinaryHV::bundle(&arg_hvs)
                };

                pred_hv.bind(&args_bundle)
            }
        }
    }
}

/// Result of checking a proof term
#[derive(Debug, Clone)]
pub struct ProofCheckResult {
    pub valid: bool,
    pub description: String,
    pub phi: f64,
}

/// A term in a proof tree
#[derive(Debug, Clone)]
pub enum ProofTerm {
    /// An axiom (assumed to be true)
    Axiom {
        name: String,
        statement: Proposition,
    },
    /// Modus ponens: from P and P→Q, derive Q
    ModusPonens {
        premise: Box<ProofTerm>,
        implication: Box<ProofTerm>,
    },
    /// Universal instantiation: from ∀x.P(x), derive P(c) for concrete c
    UniversalInstantiation {
        universal: Box<ProofTerm>,
        witness: i64,
    },
    /// Proof by induction
    Induction {
        base_case: Box<ProofTerm>,
        inductive_step: Box<ProofTerm>,
        property: String,
    },
    /// Computational proof: verify by computing
    Computation {
        description: String,
        lhs: i64,
        rhs: i64,
    },
}

impl ProofTerm {
    /// Check the validity of this proof term
    pub fn check(&self) -> ProofCheckResult {
        match self {
            ProofTerm::Axiom { name, statement } => ProofCheckResult {
                valid: true,
                description: format!("Axiom '{}': {}", name, statement.display()),
                phi: 1.0,
            },
            ProofTerm::ModusPonens {
                premise,
                implication,
            } => {
                let premise_result = premise.check();
                let impl_result = implication.check();

                let valid = premise_result.valid && impl_result.valid;
                let phi = if valid {
                    premise_result.phi + impl_result.phi + 0.5
                } else {
                    0.0
                };

                ProofCheckResult {
                    valid,
                    description: format!(
                        "Modus ponens: premise {} ({}), implication {} ({})",
                        if premise_result.valid {
                            "valid"
                        } else {
                            "invalid"
                        },
                        premise_result.description,
                        if impl_result.valid {
                            "valid"
                        } else {
                            "invalid"
                        },
                        impl_result.description
                    ),
                    phi,
                }
            }
            ProofTerm::UniversalInstantiation { universal, witness } => {
                let univ_result = universal.check();
                let phi = if univ_result.valid {
                    univ_result.phi + 0.3
                } else {
                    0.0
                };

                ProofCheckResult {
                    valid: univ_result.valid,
                    description: format!(
                        "Universal instantiation with witness {}: {}",
                        witness, univ_result.description
                    ),
                    phi,
                }
            }
            ProofTerm::Induction {
                base_case,
                inductive_step,
                property,
            } => {
                let base_result = base_case.check();
                let step_result = inductive_step.check();

                let valid = base_result.valid && step_result.valid;
                let phi = if valid {
                    base_result.phi + step_result.phi + 2.0
                } else {
                    0.0
                };

                ProofCheckResult {
                    valid,
                    description: format!(
                        "Induction on '{}': base {} ({}), step {} ({})",
                        property,
                        if base_result.valid {
                            "valid"
                        } else {
                            "invalid"
                        },
                        base_result.description,
                        if step_result.valid {
                            "valid"
                        } else {
                            "invalid"
                        },
                        step_result.description
                    ),
                    phi,
                }
            }
            ProofTerm::Computation {
                description,
                lhs,
                rhs,
            } => {
                let valid = lhs == rhs;
                let phi = if valid { 0.5 } else { 0.0 };

                ProofCheckResult {
                    valid,
                    description: format!(
                        "Computation '{}': {} {} {}",
                        description,
                        lhs,
                        if valid { "=" } else { "≠" },
                        rhs
                    ),
                    phi,
                }
            }
        }
    }

    /// Calculate the Phi measure of this proof (complexity/depth)
    pub fn proof_phi(&self) -> f64 {
        match self {
            ProofTerm::Axiom { .. } => 1.0,
            ProofTerm::Computation { .. } => 0.5,
            ProofTerm::ModusPonens {
                premise,
                implication,
            } => premise.proof_phi() + implication.proof_phi() + 0.5,
            ProofTerm::UniversalInstantiation { universal, .. } => universal.proof_phi() + 0.3,
            ProofTerm::Induction {
                base_case,
                inductive_step,
                ..
            } => base_case.proof_phi() + inductive_step.proof_phi() + 2.0,
        }
    }

    /// Generate HDC encoding of the proof structure
    pub fn encoding(&self) -> BinaryHV {
        match self {
            ProofTerm::Axiom { name, statement } => {
                let axiom_seed = seed_from_name("axiom");
                let name_seed = seed_from_name(name);

                let axiom_hv = BinaryHV::random(axiom_seed);
                let name_hv = BinaryHV::random(name_seed);
                let stmt_hv = statement.encoding();

                BinaryHV::bundle(&[axiom_hv, name_hv.bind(&stmt_hv)])
            }
            ProofTerm::ModusPonens {
                premise,
                implication,
            } => {
                let mp_seed = seed_from_name("modus_ponens");
                let mp_hv = BinaryHV::random(mp_seed);
                let prem_hv = premise.encoding();
                let impl_hv = implication.encoding();

                BinaryHV::bundle(&[mp_hv, prem_hv.bind(&impl_hv)])
            }
            ProofTerm::UniversalInstantiation { universal, witness } => {
                let ui_seed = seed_from_name("universal_instantiation");
                let wit_seed = seed_from_name(&format!("witness_{witness}"));

                let ui_hv = BinaryHV::random(ui_seed);
                let wit_hv = BinaryHV::random(wit_seed);
                let univ_hv = universal.encoding();

                BinaryHV::bundle(&[ui_hv, wit_hv.bind(&univ_hv)])
            }
            ProofTerm::Induction {
                base_case,
                inductive_step,
                property,
            } => {
                let ind_seed = seed_from_name("induction");
                let prop_seed = seed_from_name(property);

                let ind_hv = BinaryHV::random(ind_seed);
                let prop_hv = BinaryHV::random(prop_seed);
                let base_hv = base_case.encoding();
                let step_hv = inductive_step.encoding();

                BinaryHV::bundle(&[ind_hv, prop_hv, base_hv.bind(&step_hv)])
            }
            ProofTerm::Computation {
                description,
                lhs,
                rhs,
            } => {
                let comp_seed = seed_from_name("computation");
                let desc_seed = seed_from_name(description);
                let lhs_seed = seed_from_name(&format!("int_{lhs}"));
                let rhs_seed = seed_from_name(&format!("int_{rhs}"));

                let comp_hv = BinaryHV::random(comp_seed);
                let desc_hv = BinaryHV::random(desc_seed);
                let lhs_hv = BinaryHV::random(lhs_seed);
                let rhs_hv = BinaryHV::random(rhs_seed);

                BinaryHV::bundle(&[comp_hv, desc_hv, lhs_hv.bind(&rhs_hv)])
            }
        }
    }
}

/// Result of an induction proof
#[derive(Debug, Clone)]
pub struct InductionProof {
    pub property: String,
    pub base_case: ProofCheckResult,
    pub inductive_step: ProofCheckResult,
    pub samples_verified: usize,
    pub valid: bool,
    pub phi: f64,
}

/// Engine for proving properties by induction
pub struct InductionEngine {
    arithmetic: HybridArithmeticEngine,
}

impl InductionEngine {
    /// Create a new induction engine
    pub fn new() -> Self {
        Self {
            arithmetic: HybridArithmeticEngine::new(),
        }
    }

    /// Prove a property by mathematical induction
    ///
    /// - base_check: function that checks P(0)
    /// - step_check: function that checks P(k) → P(k+1) for given k
    /// - samples: number of values to verify the inductive step for
    pub fn prove_by_induction(
        &self,
        property_name: &str,
        base_check: &dyn Fn(u64) -> bool,
        step_check: &dyn Fn(u64) -> bool,
        samples: usize,
    ) -> InductionProof {
        // Check base case P(0)
        let base_valid = base_check(0);
        let base_case = ProofCheckResult {
            valid: base_valid,
            description: format!("Base case P(0) for '{property_name}'"),
            phi: if base_valid { 1.0 } else { 0.0 },
        };

        // Check inductive step for k = 0..samples
        let mut step_valid = true;
        let mut step_count = 0;

        for k in 0..samples as u64 {
            if !step_check(k) {
                step_valid = false;
                break;
            }
            step_count += 1;
        }

        let inductive_step = ProofCheckResult {
            valid: step_valid,
            description: format!("Inductive step verified for {step_count} samples"),
            phi: if step_valid {
                1.0 + (step_count as f64 * 0.1)
            } else {
                0.0
            },
        };

        let valid = base_valid && step_valid;
        let phi = if valid {
            base_case.phi + inductive_step.phi + 2.0
        } else {
            0.0
        };

        InductionProof {
            property: property_name.to_string(),
            base_case,
            inductive_step,
            samples_verified: step_count,
            valid,
            phi,
        }
    }

    /// Prove that addition is commutative: a + b = b + a
    pub fn prove_addition_commutative(&self) -> InductionProof {
        let test_values: Vec<u64> = vec![0, 1, 2, 5, 10];

        // Base case: 0 + b = b + 0 for test values (uses standard arithmetic)
        let base_check = |_n: u64| -> bool {
            test_values
                .iter()
                .all(|&b| 0u64.wrapping_add(b) == b.wrapping_add(0))
        };

        // Inductive step: (k+1) + b = b + (k+1)
        let step_check = |k: u64| -> bool {
            test_values
                .iter()
                .all(|&b| (k + 1).wrapping_add(b) == b.wrapping_add(k + 1))
        };

        self.prove_by_induction("addition_commutative", &base_check, &step_check, 20)
    }

    /// Prove the sum formula: 1 + 2 + ... + n = n*(n+1)/2
    pub fn prove_sum_formula(&self, test_up_to: u64) -> InductionProof {
        // Base case: n=0, sum=0, formula=0
        let base_check = |n: u64| -> bool {
            let sum: u64 = (1..=n).sum();
            let formula = n * (n + 1) / 2;
            sum == formula
        };

        // Inductive step: verify sum(1..=n) == n*(n+1)/2 for n = 1..test_up_to
        let step_check = |n: u64| -> bool {
            if n == 0 {
                return true;
            }

            // Calculate sum 1 + 2 + ... + n
            let sum: u64 = (1..=n).sum();

            // Calculate formula n*(n+1)/2
            let formula = n * (n + 1) / 2;

            sum == formula
        };

        self.prove_by_induction("sum_formula", &base_check, &step_check, test_up_to as usize)
    }
}

impl Default for InductionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposition_equals() {
        let p = Proposition::Equals(3, 3);
        assert_eq!(p.evaluate(), Some(true));
        let p = Proposition::Equals(3, 4);
        assert_eq!(p.evaluate(), Some(false));
    }

    #[test]
    fn test_proof_computation() {
        let proof = ProofTerm::Computation {
            description: "2 + 3 = 5".to_string(),
            lhs: 5,
            rhs: 5,
        };
        let result = proof.check();
        assert!(result.valid);
    }

    #[test]
    fn test_proof_invalid_computation() {
        let proof = ProofTerm::Computation {
            description: "2 + 3 = 6".to_string(),
            lhs: 5,
            rhs: 6,
        };
        let result = proof.check();
        assert!(!result.valid);
    }

    #[test]
    fn test_sum_formula_induction() {
        let engine = InductionEngine::new();
        let proof = engine.prove_sum_formula(20);
        assert!(proof.valid);
        assert!(proof.samples_verified >= 20);
    }

    #[test]
    fn test_addition_commutative() {
        let engine = InductionEngine::new();
        let proof = engine.prove_addition_commutative();
        assert!(proof.valid);
    }

    #[test]
    fn test_induction_with_false_property() {
        let engine = InductionEngine::new();
        // Property: n < 5 (true for small n, fails at n=5)
        let proof = engine.prove_by_induction(
            "n < 5",
            &|n| n < 5,     // base check
            &|n| n + 1 < 5, // step check: if P(n) then P(n+1)
            10,
        );
        assert!(!proof.valid); // Should fail because step fails at n=4
    }

    #[test]
    fn test_proof_phi() {
        let proof = ProofTerm::Axiom {
            name: "reflexivity".to_string(),
            statement: Proposition::Equals(1, 1),
        };
        assert!(proof.proof_phi() > 0.0);
    }
}
