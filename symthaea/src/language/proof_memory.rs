// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC-indexed memory for narrow SMT proof results.

use super::rust_ast_hdc::{ast_feature_cosine_similarity, encode_rust_ast_hdc};
use super::smt_serializer;
use crate::z3_bridge::{VerificationResult, Z3Bridge};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use syn::Expr;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofVerdict {
    Proven,
    Refuted,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub label: String,
    pub verdict: ProofVerdict,
    pub ast_features: BTreeMap<String, usize>,
    pub smtlib2: String,
    pub summary: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofMemory {
    pub records: Vec<ProofRecord>,
}

impl ProofMemory {
    pub fn observe(&mut self, record: ProofRecord) {
        self.records.push(record);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn nearest(
        &self,
        ast_features: &BTreeMap<String, usize>,
        min_similarity: f32,
    ) -> Option<(&ProofRecord, f32)> {
        self.records
            .iter()
            .filter_map(|record| {
                ast_feature_cosine_similarity(ast_features, &record.ast_features)
                    .map(|score| (record, score))
            })
            .filter(|(_, score)| *score >= min_similarity)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

pub fn proof_record_for_rust_source(
    label: impl Into<String>,
    source: &str,
    smtlib2: impl Into<String>,
    verdict: ProofVerdict,
    summary: impl Into<String>,
) -> Result<ProofRecord, syn::Error> {
    let encoded = encode_rust_ast_hdc(source, symthaea_core::hdc::unified_hv::HDC_DIMENSION)?;
    Ok(ProofRecord {
        label: label.into(),
        verdict,
        ast_features: encoded.features,
        smtlib2: smtlib2.into(),
        summary: summary.into(),
    })
}

pub struct CachedProofEngine {
    pub bridge: Z3Bridge,
    pub memory: ProofMemory,
    pub cache_hits: usize,
    pub solver_calls: usize,
}

impl CachedProofEngine {
    pub fn new(bridge: Z3Bridge, memory: ProofMemory) -> Self {
        Self {
            bridge,
            memory,
            cache_hits: 0,
            solver_calls: 0,
        }
    }

    pub fn verify_with_cache(
        &mut self,
        label: &str,
        source: &str,
        smtlib2: &str,
        min_similarity: f32,
        #[cfg(feature = "swarm")] swarm_proofs: &[symthaea_swarm::SwarmProofMsg],
        #[cfg(not(feature = "swarm"))] _swarm_proofs: &[()],
    ) -> (ProofVerdict, String) {
        // 1. Distributed SMT Short-Circuit (Read Path Interception)
        #[cfg(feature = "swarm")]
        if let Some(peer_proof) = swarm_proofs.iter().find(|p| p.smtlib2 == smtlib2) {
            self.cache_hits += 1;
            let verdict = if peer_proof.verified {
                ProofVerdict::Proven
            } else {
                ProofVerdict::Refuted
            };
            return (
                verdict,
                format!(
                    "SWARM SHORT-CIRCUIT HIT: Found exact match in peer mesh registry for `{}`.",
                    peer_proof.label
                ),
            );
        }

        if let Ok(encoded) =
            encode_rust_ast_hdc(source, symthaea_core::hdc::unified_hv::HDC_DIMENSION)
        {
            // Check local cache tracking
            if let Some((cached_record, score)) =
                self.memory.nearest(&encoded.features, min_similarity)
            {
                self.cache_hits += 1;
                return (
                    cached_record.verdict.clone(),
                    format!(
                        "LOCAL CACHE HIT [similarity: {:.3}]: Matches formal prototype `{}`. Prover summary: {}",
                        score, cached_record.label, cached_record.summary
                    ),
                );
            }

            // 2. Fuzzy Lemma Chaining via High-Dimensional Computing (HDC)
            #[cfg(feature = "swarm")]
            {
                let mut best_swarm_match: Option<(&symthaea_swarm::SwarmProofMsg, f32)> = None;
                for peer_proof in swarm_proofs {
                    let score = encoded.hv.similarity(&peer_proof.proof_hv);
                    if score >= min_similarity {
                        if best_swarm_match.is_none() || score > best_swarm_match.unwrap().1 {
                            best_swarm_match = Some((peer_proof, score));
                        }
                    }
                }

                if let Some((peer_proof, score)) = best_swarm_match {
                    self.cache_hits += 1;
                    let verdict = if peer_proof.verified {
                        ProofVerdict::Proven
                    } else {
                        ProofVerdict::Refuted
                    };
                    return (
                        verdict,
                        format!(
                            "SWARM HDC HIT [similarity: {:.3}]: Matches peer formal lemma `{}`.",
                            score, peer_proof.label
                        ),
                    );
                }
            }
        }

        self.solver_calls += 1;
        let bridge_result = self.bridge.verify_satisfiable(smtlib2);

        let verdict = match bridge_result {
            VerificationResult::Unsat { .. } | VerificationResult::Valid => ProofVerdict::Proven,
            VerificationResult::Sat { .. } | VerificationResult::Invalid => ProofVerdict::Refuted,
            _ => ProofVerdict::Unknown,
        };

        let summary = format!("Evaluated via structural Z3 SMT runtime engine execution.");

        if let Ok(record) =
            proof_record_for_rust_source(label, source, smtlib2, verdict.clone(), &summary)
        {
            self.memory.observe(record);
        }

        (verdict, summary)
    }

    /// Export refuted logical structures as negative prototypes for MCTS penalties.
    pub fn export_negative_prototypes(
        &self,
        dim: usize,
    ) -> crate::consciousness::temporal_planning::mcts::NegativePrototypeBank {
        let mut bank =
            crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default();
        for record in &self.memory.records {
            if record.verdict == ProofVerdict::Refuted {
                // Convert AST features to a dense embedding vector
                let mut embedding = vec![0.0f32; dim];
                for (feature, count) in &record.ast_features {
                    let idx = (feature
                        .bytes()
                        .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize)))
                        % dim;
                    embedding[idx] += *count as f32;
                }
                // Normalize
                let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if mag > 0.0 {
                    for x in &mut embedding {
                        *x /= mag;
                    }
                }
                bank.prototypes.push((embedding, 1.0)); // Default penalty weight 1.0
            }
        }
        bank
    }

    /// Prove that two arithmetic expressions are logically equivalent.
    ///
    /// This automatically serializes the Rust expressions to SMT-LIB2 and
    /// queries Z3 (with caching) to verify if ∀vars: expr_a == expr_b.
    pub fn verify_arithmetic_equivalence(
        &mut self,
        expr_a_str: &str,
        expr_b_str: &str,
    ) -> (ProofVerdict, String) {
        let expr_a: Expr = match syn::parse_str(expr_a_str) {
            Ok(e) => e,
            Err(e) => return (ProofVerdict::Unknown, format!("Parse error (A): {}", e)),
        };
        let expr_b: Expr = match syn::parse_str(expr_b_str) {
            Ok(e) => e,
            Err(e) => return (ProofVerdict::Unknown, format!("Parse error (B): {}", e)),
        };

        let smt_a = match smt_serializer::expr_to_smtlib2(&expr_a) {
            Some(s) => s,
            None => {
                return (
                    ProofVerdict::Unknown,
                    "Expression A is not loop-free arithmetic".to_string(),
                );
            }
        };
        let smt_b = match smt_serializer::expr_to_smtlib2(&expr_b) {
            Some(s) => s,
            None => {
                return (
                    ProofVerdict::Unknown,
                    "Expression B is not loop-free arithmetic".to_string(),
                );
            }
        };

        let decls_a = smt_serializer::get_smt_declarations(&expr_a, "Int");
        let decls_b = smt_serializer::get_smt_declarations(&expr_b, "Int");

        // Merge declarations
        let mut decls_set: std::collections::HashSet<String> =
            decls_a.lines().map(|s| s.to_string()).collect();
        decls_set.extend(decls_b.lines().map(|s| s.to_string()));
        let mut decls_vec: Vec<_> = decls_set.into_iter().collect();
        decls_vec.sort();
        let decls_str = decls_vec.join("\n");

        // We want to prove ∀x: smt_a == smt_b, so we check if ∃x: smt_a != smt_b is UNSAT.
        let smt_query = format!(
            "(set-logic QF_LIA)\n{}\n(assert (not (= {} {})))\n(check-sat)",
            decls_str, smt_a, smt_b
        );

        let label = format!("equiv({} , {})", expr_a_str, expr_b_str);
        self.verify_with_cache(&label, expr_a_str, &smt_query, 0.95, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_arithmetic_equivalence_identity() {
        let bridge = Z3Bridge::new();
        let memory = ProofMemory::default();
        let mut engine = CachedProofEngine::new(bridge, memory);

        // a + b + c == c + b + a
        let (verdict, details) = engine.verify_arithmetic_equivalence("a + b + c", "c + b + a");

        // If Z3 is available, it should be Proven. If not, it depends on fallback.
        if engine.bridge.z3_available {
            assert_eq!(verdict, ProofVerdict::Proven);
        } else {
            assert!(matches!(
                verdict,
                ProofVerdict::Proven | ProofVerdict::Unknown
            ));
        }
        assert!(
            details.contains("Evaluated via structural Z3 SMT")
                || details.contains("LOCAL CACHE HIT")
        );
    }

    #[test]
    fn test_verify_arithmetic_equivalence_refutation() {
        let bridge = Z3Bridge::new();
        let memory = ProofMemory::default();
        let mut engine = CachedProofEngine::new(bridge, memory);

        // a + 1 == a + 2 (False)
        let (verdict, _) = engine.verify_arithmetic_equivalence("a + 1", "a + 2");

        if engine.bridge.z3_available {
            assert_eq!(verdict, ProofVerdict::Refuted);
        }
    }

    #[test]
    fn test_cached_proof_engine_avoids_solver_on_hit() {
        let mut memory = ProofMemory::default();
        let record = proof_record_for_rust_source(
            "polynomial_identity",
            "pub fn inc(x: i32) -> i32 { x + 1 }",
            "(assert (= y (+ x 1)))",
            ProofVerdict::Proven,
            "algebraic base proven",
        )
        .unwrap();
        memory.observe(record);

        let bridge = Z3Bridge {
            z3_available: false,
            z3_path: None,
            timeout_secs: 5,
        };

        let mut engine = CachedProofEngine::new(bridge, memory);

        let query_source = "pub fn bump(x: i32) -> i32 { x + 1 }";
        let smt = "(assert (= y (+ x 1)))";

        #[cfg(feature = "swarm")]
        let (verdict, details) =
            engine.verify_with_cache("bump_check", query_source, smt, 0.85, &[]);
        #[cfg(not(feature = "swarm"))]
        let (verdict, details) =
            engine.verify_with_cache("bump_check", query_source, smt, 0.85, &[]);
        assert_eq!(verdict, ProofVerdict::Proven);
        assert_eq!(engine.cache_hits, 1);
        assert_eq!(engine.solver_calls, 0);
        assert!(details.contains("LOCAL CACHE HIT"));
    }

    #[test]
    #[cfg(feature = "swarm")]
    fn test_swarm_short_circuit_hit() {
        let memory = ProofMemory::default();
        let bridge = Z3Bridge {
            z3_available: false,
            z3_path: None,
            timeout_secs: 5,
        };
        let mut engine = CachedProofEngine::new(bridge, memory);

        let peer_proof = symthaea_swarm::SwarmProofMsg {
            node_id: uuid::Uuid::new_v4(),
            label: "peer_lemma_alpha".to_string(),
            smtlib2: "(assert (= a b))".to_string(),
            proof_hv: symthaea_core::hdc::ContinuousHV::zero(16384),
            verified: true,
            timestamp: 0,
        };

        let (verdict, details) = engine.verify_with_cache(
            "local_check",
            "pub fn test() {}",
            "(assert (= a b))",
            0.85,
            &[peer_proof],
        );

        assert_eq!(verdict, ProofVerdict::Proven);
        assert_eq!(engine.cache_hits, 1);
        assert!(details.contains("SWARM SHORT-CIRCUIT HIT"));
    }
}

#[cfg(test)]
mod demo_tests {
    use super::*;
    use crate::z3_bridge::Z3Bridge;

    #[test]
    fn run_neuro_symbolic_proof_cache_demo() {
        let bridge = Z3Bridge::new();
        let mut memory = ProofMemory::default();

        let record1 = proof_record_for_rust_source(
            "polynomial_increment",
            "pub fn add_one(n: i32) -> i32 { n + 1 }",
            "(assert (not (= (+ n 1) (+ n 1))))",
            ProofVerdict::Proven,
            "Identity verified via Z3 integer math constraints.",
        )
        .unwrap();
        memory.observe(record1);

        let mut engine = CachedProofEngine::new(bridge, memory);
        let candidate_code = "pub fn execute_accumulation_step(n: i32) -> i32 { n + 1 }";
        let smt_query = "(assert (not (= (+ n 1) (+ n 1))))";

        let (verdict, log) =
            engine.verify_with_cache("accumulate_task", candidate_code, smt_query, 0.90, &[]);
        assert_eq!(verdict, ProofVerdict::Proven);
        assert_eq!(engine.cache_hits, 1);
    }
}
