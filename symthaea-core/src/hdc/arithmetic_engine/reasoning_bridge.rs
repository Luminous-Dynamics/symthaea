// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::HashMap;

use super::discovery::MathDiscovery;
use super::hybrid::{AbstractProof, HybridArithmeticEngine};
use super::theorems::TheoremProver;

/// Mathematical concept types for reasoning
#[derive(Debug, Clone, PartialEq)]
pub enum MathConceptType {
    /// Natural number (0, 1, 2, ...)
    Number,
    /// Arithmetic operation (add, multiply, etc.)
    Operation,
    /// Mathematical property (prime, even, etc.)
    Property,
    /// Mathematical theorem
    Theorem,
    /// Proof step
    ProofStep,
    /// Abstract structure (group, ring, etc.)
    Structure,
}

/// Mathematical relations for reasoning
#[derive(Debug, Clone, PartialEq)]
pub enum MathRelation {
    /// a equals b
    Equals,
    /// a is less than b
    LessThan,
    /// a divides b evenly
    Divides,
    /// a is coprime to b (gcd = 1)
    Coprime,
    /// a proves b (logical entailment)
    Proves,
    /// a implies b
    Implies,
    /// a is instance of b (5 is instance of Prime)
    InstanceOf,
    /// a has property b
    HasProperty,
    /// Proof step a follows from b
    FollowsFrom,
    /// a composes with b (operations)
    ComposesWith,
}

/// A mathematical assertion that can be used in reasoning
#[derive(Debug, Clone)]
pub struct MathAssertion {
    /// Subject of the assertion
    pub subject: String,
    /// Relation type
    pub relation: MathRelation,
    /// Object of the assertion
    pub object: String,
    /// Confidence (0.0 - 1.0), based on proof strength
    pub confidence: f64,
    /// Φ from the proof (higher = more integrated reasoning)
    pub phi: f64,
    /// Source proof if available
    pub proof_source: Option<AbstractProof>,
}

/// Bridge connecting arithmetic engine to reasoning system
pub struct MathReasoningBridge {
    /// Arithmetic computation engine
    engine: HybridArithmeticEngine,
    /// Mathematical discovery system
    discovery: MathDiscovery,
    /// Accumulated assertions
    assertions: Vec<MathAssertion>,
    /// Proven theorems (cached for reuse)
    proven_theorems: HashMap<String, AbstractProof>,
}

impl MathReasoningBridge {
    /// Create new bridge
    pub fn new() -> Self {
        Self {
            engine: HybridArithmeticEngine::new(),
            discovery: MathDiscovery::new(),
            assertions: Vec::new(),
            proven_theorems: HashMap::new(),
        }
    }

    // ========================================================================
    // ASSERTION GENERATION
    // ========================================================================

    /// Generate assertion from arithmetic result
    pub fn assert_equality(&mut self, a: u64, b: u64, op: &str) -> MathAssertion {
        let (value, phi, proof) = match op {
            "add" | "+" => {
                let r = self.engine.add(a, b);
                (r.value, r.phi, r.abstract_proof)
            }
            "multiply" | "*" | "×" => {
                let r = self.engine.multiply(a, b);
                (r.value, r.phi, r.abstract_proof)
            }
            "subtract" | "-" => {
                if let Some(r) = self.engine.subtract(a, b) {
                    (r.value, r.phi, r.abstract_proof)
                } else {
                    return MathAssertion {
                        subject: format!("{a} - {b}"),
                        relation: MathRelation::Equals,
                        object: "undefined (negative in naturals)".to_string(),
                        confidence: 1.0,
                        phi: 0.0,
                        proof_source: None,
                    };
                }
            }
            "divide" | "/" | "÷" => {
                if let Some(r) = self.engine.divide(a, b) {
                    (r.value, r.phi, r.abstract_proof)
                } else {
                    return MathAssertion {
                        subject: format!("{a} / {b}"),
                        relation: MathRelation::Equals,
                        object: "undefined (division by zero)".to_string(),
                        confidence: 1.0,
                        phi: 0.0,
                        proof_source: None,
                    };
                }
            }
            "mod" | "%" => {
                if let Some(r) = self.engine.modulo(a, b) {
                    (r.value, r.phi, r.abstract_proof)
                } else {
                    return MathAssertion {
                        subject: format!("{a} % {b}"),
                        relation: MathRelation::Equals,
                        object: "undefined (modulo by zero)".to_string(),
                        confidence: 1.0,
                        phi: 0.0,
                        proof_source: None,
                    };
                }
            }
            "gcd" => {
                let r = self.engine.gcd(a, b);
                (r.value, r.phi, r.abstract_proof)
            }
            "power" | "^" => {
                let r = self.engine.power(a, b);
                (r.value, r.phi, r.abstract_proof)
            }
            _ => {
                return MathAssertion {
                    subject: format!("{a} {op} {b}"),
                    relation: MathRelation::Equals,
                    object: "unknown operation".to_string(),
                    confidence: 0.0,
                    phi: 0.0,
                    proof_source: None,
                };
            }
        };

        let assertion = MathAssertion {
            subject: format!("{a} {op} {b}"),
            relation: MathRelation::Equals,
            object: value.to_string(),
            confidence: 1.0, // Mathematical facts are certain
            phi,
            proof_source: proof,
        };

        self.assertions.push(assertion.clone());
        assertion
    }

    /// Assert divisibility relation
    pub fn assert_divides(&mut self, a: u64, b: u64) -> MathAssertion {
        if b == 0 {
            return MathAssertion {
                subject: a.to_string(),
                relation: MathRelation::Divides,
                object: "0".to_string(),
                confidence: 1.0, // Everything divides 0
                phi: 0.0,
                proof_source: None,
            };
        }

        if let Some(mod_result) = self.engine.modulo(b, a) {
            let divides = mod_result.value == 0;

            let assertion = MathAssertion {
                subject: a.to_string(),
                relation: MathRelation::Divides,
                object: b.to_string(),
                confidence: if divides { 1.0 } else { 0.0 },
                phi: mod_result.phi,
                proof_source: mod_result.abstract_proof,
            };

            self.assertions.push(assertion.clone());
            assertion
        } else {
            // a == 0, cannot check divisibility
            MathAssertion {
                subject: a.to_string(),
                relation: MathRelation::Divides,
                object: b.to_string(),
                confidence: 0.0,
                phi: 0.0,
                proof_source: None,
            }
        }
    }

    /// Assert primality
    pub fn assert_prime(&mut self, n: u64) -> MathAssertion {
        let result = self.engine.is_prime(n);
        let is_prime = result.value == 1;

        let assertion = MathAssertion {
            subject: n.to_string(),
            relation: MathRelation::InstanceOf,
            object: if is_prime { "Prime" } else { "Composite" }.to_string(),
            confidence: 1.0,
            phi: result.phi,
            proof_source: result.abstract_proof,
        };

        self.assertions.push(assertion.clone());
        assertion
    }

    /// Assert coprimality (gcd = 1)
    pub fn assert_coprime(&mut self, a: u64, b: u64) -> MathAssertion {
        let gcd_result = self.engine.gcd(a, b);
        let coprime = gcd_result.value == 1;

        let assertion = MathAssertion {
            subject: a.to_string(),
            relation: MathRelation::Coprime,
            object: b.to_string(),
            confidence: if coprime { 1.0 } else { 0.0 },
            phi: gcd_result.phi,
            proof_source: gcd_result.abstract_proof,
        };

        self.assertions.push(assertion.clone());
        assertion
    }

    // ========================================================================
    // THEOREM PROVING FOR REASONING
    // ========================================================================

    /// Prove a theorem and add to reasoning base
    pub fn prove_theorem(&mut self, theorem: &str, params: &[u64]) -> Option<MathAssertion> {
        let mut prover = TheoremProver::new();

        match theorem {
            "commutativity_add" if params.len() >= 2 => {
                let result = prover.prove_addition_commutative(params[0], params[1]);
                if result.verified {
                    let proof_strings: Vec<String> = result
                        .proof_steps
                        .iter()
                        .map(|s| format!("{:?}", s.result.value))
                        .collect();
                    let assertion = MathAssertion {
                        subject: format!("{} + {}", params[0], params[1]),
                        relation: MathRelation::Equals,
                        object: format!("{} + {}", params[1], params[0]),
                        confidence: 1.0,
                        phi: result.total_phi,
                        proof_source: None,
                    };
                    self.proven_theorems.insert(
                        theorem.to_string(),
                        AbstractProof {
                            theorem: format!(
                                "Commutativity: {} + {} = {} + {}",
                                params[0], params[1], params[1], params[0]
                            ),
                            base_cases: proof_strings.clone(),
                            inductive_step: "Addition is commutative by construction".to_string(),
                            justification: proof_strings,
                            is_sound: true,
                        },
                    );
                    self.assertions.push(assertion.clone());
                    return Some(assertion);
                }
            }
            "commutativity_mul" if params.len() >= 2 => {
                // Use the engine directly for multiplication commutativity
                let result1 = self.engine.multiply(params[0], params[1]);
                let result2 = self.engine.multiply(params[1], params[0]);
                if result1.value == result2.value {
                    let total_phi = result1.phi + result2.phi;
                    let proof_strings = vec![
                        format!("{} × {} = {}", params[0], params[1], result1.value),
                        format!("{} × {} = {}", params[1], params[0], result2.value),
                    ];
                    let assertion = MathAssertion {
                        subject: format!("{} × {}", params[0], params[1]),
                        relation: MathRelation::Equals,
                        object: format!("{} × {}", params[1], params[0]),
                        confidence: 1.0,
                        phi: total_phi,
                        proof_source: None,
                    };
                    self.proven_theorems.insert(
                        theorem.to_string(),
                        AbstractProof {
                            theorem: format!(
                                "Commutativity: {} × {} = {} × {}",
                                params[0], params[1], params[1], params[0]
                            ),
                            base_cases: proof_strings.clone(),
                            inductive_step: "Multiplication is commutative by construction"
                                .to_string(),
                            justification: proof_strings,
                            is_sound: true,
                        },
                    );
                    self.assertions.push(assertion.clone());
                    return Some(assertion);
                }
            }
            "associativity" if params.len() >= 3 => {
                let result = prover.prove_addition_associative(params[0], params[1], params[2]);
                if result.verified {
                    let proof_strings: Vec<String> = result
                        .proof_steps
                        .iter()
                        .map(|s| format!("{:?}", s.result.value))
                        .collect();
                    let assertion = MathAssertion {
                        subject: format!("({} + {}) + {}", params[0], params[1], params[2]),
                        relation: MathRelation::Equals,
                        object: format!("{} + ({} + {})", params[0], params[1], params[2]),
                        confidence: 1.0,
                        phi: result.total_phi,
                        proof_source: None,
                    };
                    self.proven_theorems.insert(
                        theorem.to_string(),
                        AbstractProof {
                            theorem: format!(
                                "Associativity: ({} + {}) + {} = {} + ({} + {})",
                                params[0], params[1], params[2], params[0], params[1], params[2]
                            ),
                            base_cases: proof_strings.clone(),
                            inductive_step: "Addition is associative by construction".to_string(),
                            justification: proof_strings,
                            is_sound: true,
                        },
                    );
                    self.assertions.push(assertion.clone());
                    return Some(assertion);
                }
            }
            "distributive" if params.len() >= 3 => {
                let result = prover.prove_distributive(params[0], params[1], params[2]);
                if result.verified {
                    let proof_strings: Vec<String> = result
                        .proof_steps
                        .iter()
                        .map(|s| format!("{:?}", s.result.value))
                        .collect();
                    let assertion = MathAssertion {
                        subject: format!("{} × ({} + {})", params[0], params[1], params[2]),
                        relation: MathRelation::Equals,
                        object: format!(
                            "({} × {}) + ({} × {})",
                            params[0], params[1], params[0], params[2]
                        ),
                        confidence: 1.0,
                        phi: result.total_phi,
                        proof_source: None,
                    };
                    self.proven_theorems.insert(
                        theorem.to_string(),
                        AbstractProof {
                            theorem: format!(
                                "Distributive: {} × ({} + {}) = ({} × {}) + ({} × {})",
                                params[0],
                                params[1],
                                params[2],
                                params[0],
                                params[1],
                                params[0],
                                params[2]
                            ),
                            base_cases: proof_strings.clone(),
                            inductive_step: "Multiplication distributes over addition".to_string(),
                            justification: proof_strings,
                            is_sound: true,
                        },
                    );
                    self.assertions.push(assertion.clone());
                    return Some(assertion);
                }
            }
            _ => {}
        }

        None
    }

    // ========================================================================
    // REASONING CHAIN SUPPORT
    // ========================================================================

    /// Chain reasoning: If a | b and b | c, then a | c
    pub fn reason_transitive_divisibility(
        &mut self,
        a: u64,
        b: u64,
        c: u64,
    ) -> Option<MathAssertion> {
        let a_divides_b = self.assert_divides(a, b);
        if a_divides_b.confidence < 1.0 {
            return None;
        }

        let b_divides_c = self.assert_divides(b, c);
        if b_divides_c.confidence < 1.0 {
            return None;
        }

        // By transitivity, a | c
        let a_divides_c = self.assert_divides(a, c);

        let total_phi = a_divides_b.phi + b_divides_c.phi + a_divides_c.phi;

        Some(MathAssertion {
            subject: a.to_string(),
            relation: MathRelation::Divides,
            object: c.to_string(),
            confidence: 1.0,
            phi: total_phi,
            proof_source: Some(AbstractProof {
                theorem: format!("{a} divides {c} by transitivity"),
                base_cases: vec![format!("{} | {}", a, b), format!("{} | {}", b, c)],
                inductive_step: "Divisibility is transitive: a|b ∧ b|c → a|c".to_string(),
                justification: vec![
                    format!("∃k: b = {}k", a),
                    format!("∃m: c = {}m", b),
                    format!("Therefore c = {}km", a),
                ],
                is_sound: true,
            }),
        })
    }

    /// Reason about GCD properties
    pub fn reason_gcd_properties(&mut self, a: u64, b: u64) -> Vec<MathAssertion> {
        let mut results = Vec::new();

        let gcd = self.engine.gcd(a, b);
        let g = gcd.value;

        // Property 1: gcd(a, b) divides a
        results.push(MathAssertion {
            subject: g.to_string(),
            relation: MathRelation::Divides,
            object: a.to_string(),
            confidence: 1.0,
            phi: gcd.phi * 0.3,
            proof_source: None,
        });

        // Property 2: gcd(a, b) divides b
        results.push(MathAssertion {
            subject: g.to_string(),
            relation: MathRelation::Divides,
            object: b.to_string(),
            confidence: 1.0,
            phi: gcd.phi * 0.3,
            proof_source: None,
        });

        // Property 3: If coprime, gcd(a, b) = 1
        if g == 1 {
            results.push(MathAssertion {
                subject: a.to_string(),
                relation: MathRelation::Coprime,
                object: b.to_string(),
                confidence: 1.0,
                phi: gcd.phi,
                proof_source: gcd.abstract_proof.clone(),
            });
        }

        // Property 4: gcd(a, b) = gcd(b, a) (commutativity)
        results.push(MathAssertion {
            subject: format!("gcd({a}, {b})"),
            relation: MathRelation::Equals,
            object: format!("gcd({b}, {a})"),
            confidence: 1.0,
            phi: gcd.phi * 0.2,
            proof_source: None,
        });

        results
    }

    /// Multi-step proof with Φ tracking
    pub fn multi_step_proof(&mut self, steps: Vec<(&str, &[u64])>) -> (Vec<MathAssertion>, f64) {
        let mut assertions = Vec::new();
        let mut total_phi = 0.0;

        for (theorem, params) in steps {
            if let Some(assertion) = self.prove_theorem(theorem, params) {
                total_phi += assertion.phi;
                assertions.push(assertion);
            }
        }

        (assertions, total_phi)
    }

    // ========================================================================
    // QUERY INTERFACE
    // ========================================================================

    /// Get all assertions of a specific relation type
    pub fn query_by_relation(&self, relation: &MathRelation) -> Vec<&MathAssertion> {
        self.assertions
            .iter()
            .filter(|a| &a.relation == relation)
            .collect()
    }

    /// Get assertions involving a specific number
    pub fn query_involving(&self, n: u64) -> Vec<&MathAssertion> {
        let n_str = n.to_string();
        self.assertions
            .iter()
            .filter(|a| a.subject.contains(&n_str) || a.object.contains(&n_str))
            .collect()
    }

    /// Get highest Φ assertion
    pub fn highest_phi_assertion(&self) -> Option<&MathAssertion> {
        self.assertions.iter().max_by(|a, b| {
            a.phi
                .partial_cmp(&b.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get total accumulated Φ from all reasoning
    pub fn total_phi(&self) -> f64 {
        self.assertions.iter().map(|a| a.phi).sum()
    }

    /// Get all proven theorems
    pub fn proven_theorems(&self) -> &HashMap<String, AbstractProof> {
        &self.proven_theorems
    }

    /// Get all assertions
    pub fn assertions(&self) -> &[MathAssertion] {
        &self.assertions
    }

    /// Access the discovery engine
    pub fn discovery(&mut self) -> &mut MathDiscovery {
        &mut self.discovery
    }

    /// Access the arithmetic engine directly
    pub fn engine(&mut self) -> &mut HybridArithmeticEngine {
        &mut self.engine
    }
}

impl Default for MathReasoningBridge {
    fn default() -> Self {
        Self::new()
    }
}
