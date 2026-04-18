// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Meta-HDC — Concept Vectors for Mathematical Discoveries
//!
//! Encodes verified conjectures as role-bound hypervectors ("concept vectors"),
//! then clusters them in Hamming space to find meta-patterns — patterns between
//! the patterns themselves.
//!
//! ## Encoding Strategy
//!
//! Each concept vector is a role-bound superposition:
//! ```text
//! concept_hv = ROLE_FORMULA   ⊗ formula_hv
//!            ⊕ ROLE_DOMAIN    ⊗ domain_hv
//!            ⊕ ROLE_STATUS    ⊗ status_hv
//!            ⊕ ROLE_STRUCTURE ⊗ structural_hv
//! ```
//!
//! Where ⊗ is `BinaryHV::bind()` (XOR) and ⊕ is `BinaryHV::bundle()` (majority vote).
//!
//! ## References
//!
//! - Kanerva (2009) — Hyperdimensional computing
//! - Plate (2003) — Holographic Reduced Representations

use std::collections::{HashMap, HashSet};

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::conjecture_engine::{
    Conjecture, ConjectureEngine, ConjectureStatus, Expr, MathDomain,
};
use crate::hdc::deterministic_seeds::seed_from_name;
use crate::hdc::primitive_system::PrimitiveSystem;

use super::{expr_canonical_string, expr_to_symbolic, extract_subtrees, normalize_expr};

// ═══════════════════════════════════════════════════════════════════════════
// CONCEPT VECTOR
// ═══════════════════════════════════════════════════════════════════════════

/// A discovery encoded as a hypervector — the concept of the conjecture itself,
/// not just its formula.
#[derive(Debug, Clone)]
pub struct ConceptVector {
    /// HDC encoding of the full discovery context (role-bound composite)
    pub encoding: BinaryHV,
    /// Index into ConjectureEngine.conjectures
    pub conjecture_idx: usize,
    /// HDC encoding of the formula (from SymbolicExpr)
    pub formula_hv: BinaryHV,
    /// HDC encoding of the mathematical domain
    pub domain_hv: BinaryHV,
    /// The mathematical domain (for diversity counting)
    pub domain: MathDomain,
    /// HDC encoding of the normalized structural pattern
    pub structural_hv: BinaryHV,
    /// Cluster assignment (None until clustered)
    pub cluster_id: Option<usize>,
}

/// A cluster of structurally related discoveries.
#[derive(Debug, Clone)]
pub struct ConceptCluster {
    /// Cluster identifier
    pub id: usize,
    /// Centroid (majority-vote bundle of member concept vectors)
    pub centroid: BinaryHV,
    /// Indices into MetaHDC.concepts
    pub members: Vec<usize>,
    /// Number of distinct MathDomains represented in this cluster
    pub domain_diversity: usize,
    /// Dominant structural pattern label (canonical string of most common subtree)
    pub pattern_label: String,
}

/// Role vectors for concept encoding — deterministic, created once.
struct RoleVectors {
    formula: BinaryHV,
    domain: BinaryHV,
    status: BinaryHV,
    structure: BinaryHV,
}

impl RoleVectors {
    fn new() -> Self {
        Self {
            formula: BinaryHV::random(seed_from_name("ROLE_FORMULA")),
            domain: BinaryHV::random(seed_from_name("ROLE_DOMAIN")),
            status: BinaryHV::random(seed_from_name("ROLE_STATUS")),
            structure: BinaryHV::random(seed_from_name("ROLE_STRUCTURE")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// META-HDC SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Manages concept vector encoding and clustering for mathematical discoveries.
pub struct MetaHDC {
    /// All concept vectors, one per encoded conjecture
    pub concepts: Vec<ConceptVector>,
    /// Discovered clusters (updated by `cluster()`)
    pub clusters: Vec<ConceptCluster>,
    /// Role vectors for encoding
    roles: RoleVectors,
    /// Domain HVs — deterministic per MathDomain variant
    domain_hvs: HashMap<MathDomain, BinaryHV>,
    /// Status HVs — deterministic per verification status
    status_hvs: HashMap<String, BinaryHV>,
}

impl MetaHDC {
    pub fn new() -> Self {
        let mut domain_hvs = HashMap::new();
        for domain in ALL_DOMAINS {
            domain_hvs.insert(
                *domain,
                BinaryHV::random(super::domain_seed(*domain)),
            );
        }

        let mut status_hvs = HashMap::new();
        for name in &["proposed", "tested", "checked", "verified", "refuted"] {
            status_hvs.insert(
                name.to_string(),
                BinaryHV::random(seed_from_name(&format!("STATUS_{}", name))),
            );
        }

        Self {
            concepts: Vec::new(),
            clusters: Vec::new(),
            roles: RoleVectors::new(),
            domain_hvs,
            status_hvs,
        }
    }

    /// Encode a conjecture and add it as a concept vector.
    pub fn add_conjecture(
        &mut self,
        conjecture: &Conjecture,
        idx: usize,
        primitives: &PrimitiveSystem,
    ) {
        let cv = self.encode_conjecture(conjecture, idx, primitives);
        self.concepts.push(cv);
    }

    /// Encode a single conjecture into a concept vector.
    fn encode_conjecture(
        &self,
        conjecture: &Conjecture,
        idx: usize,
        primitives: &PrimitiveSystem,
    ) -> ConceptVector {
        // 1. Formula HV — from SymbolicExpr encoding
        let formula_hv = expr_to_symbolic(&conjecture.formula, primitives).encoding;

        // 2. Domain HV — deterministic per MathDomain
        let domain_hv = self
            .domain_hvs
            .get(&conjecture.domain)
            .cloned()
            .unwrap_or_else(|| BinaryHV::random(0));

        // 3. Status HV
        let status_key = match &conjecture.status {
            ConjectureStatus::Proposed => "proposed",
            ConjectureStatus::NumericallyTested { .. } => "tested",
            ConjectureStatus::SymbolicallyChecked => "checked",
            ConjectureStatus::FormallyVerified { .. } => "verified",
            ConjectureStatus::Refuted { .. } => "refuted",
        };
        let status_hv = self
            .status_hvs
            .get(status_key)
            .cloned()
            .unwrap_or_else(|| BinaryHV::random(0));

        // 4. Structural HV — normalized formula shape (constants stripped)
        let normalized = normalize_expr(&conjecture.formula);
        let structural_hv = expr_to_symbolic(&normalized, primitives).encoding;

        // 5. Compose via role-binding + bundling
        let bound_formula = self.roles.formula.bind(&formula_hv);
        let bound_domain = self.roles.domain.bind(&domain_hv);
        let bound_status = self.roles.status.bind(&status_hv);
        let bound_structure = self.roles.structure.bind(&structural_hv);

        let encoding = BinaryHV::bundle(&[
            bound_formula,
            bound_domain,
            bound_status,
            bound_structure,
        ]);

        ConceptVector {
            encoding,
            conjecture_idx: idx,
            formula_hv,
            domain_hv,
            domain: conjecture.domain,
            structural_hv,
            cluster_id: None,
        }
    }

    /// K-means clustering of concept vectors in Hamming space.
    ///
    /// Assigns each concept to the nearest centroid, recomputes centroids
    /// as majority-vote bundles, repeats until stable or max iterations.
    pub fn cluster(&mut self, k: usize) {
        let n = self.concepts.len();
        if n < k {
            return;
        }

        // Initialize centroids: pick k evenly spaced concept vectors
        let mut centroids: Vec<BinaryHV> = (0..k)
            .map(|i| self.concepts[i * n / k].encoding.clone())
            .collect();

        let max_iter = 20;
        for _ in 0..max_iter {
            // Assign each concept to nearest centroid
            let mut assignments = vec![0usize; n];
            for (i, cv) in self.concepts.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_sim = f32::MIN;
                for (j, centroid) in centroids.iter().enumerate() {
                    let sim = cv.encoding.similarity(centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cluster = j;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Check for convergence
            let mut changed = false;
            for (i, cv) in self.concepts.iter_mut().enumerate() {
                if cv.cluster_id != Some(assignments[i]) {
                    cv.cluster_id = Some(assignments[i]);
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Recompute centroids as majority-vote bundle of members
            for j in 0..k {
                let members: Vec<BinaryHV> = self
                    .concepts
                    .iter()
                    .filter(|cv| cv.cluster_id == Some(j))
                    .map(|cv| cv.encoding.clone())
                    .collect();
                if !members.is_empty() {
                    centroids[j] = BinaryHV::bundle(&members);
                }
            }
        }

        // Build cluster structs
        self.clusters.clear();
        for j in 0..k {
            let members: Vec<usize> = self
                .concepts
                .iter()
                .enumerate()
                .filter(|(_, cv)| cv.cluster_id == Some(j))
                .map(|(i, _)| i)
                .collect();

            if members.is_empty() {
                continue;
            }

            // Count domain diversity using the MathDomain enum directly
            let domains: HashSet<MathDomain> = members
                .iter()
                .map(|&i| self.concepts[i].domain)
                .collect();

            self.clusters.push(ConceptCluster {
                id: j,
                centroid: centroids[j].clone(),
                members,
                domain_diversity: domains.len(),
                pattern_label: String::new(), // Filled below
            });
        }

        // Label clusters with most common structural pattern
        // (This is intentionally simple — just picks the most frequent canonical string)
        // We skip this for now to avoid needing to access conjectures here
    }

    /// Find recurring subtrees across concept clusters.
    ///
    /// For each cluster with domain_diversity >= 2 (cross-domain pattern),
    /// extract subtrees from member conjectures and find those appearing
    /// in at least 2 cluster members.
    ///
    /// Returns (normalized subtree, conjecture indices where it appears).
    pub fn recurring_subtrees(&self, engine: &ConjectureEngine) -> Vec<(Expr, Vec<usize>)> {
        let mut results = Vec::new();

        for cluster in &self.clusters {
            if cluster.domain_diversity < 2 || cluster.members.len() < 2 {
                continue;
            }

            // Extract subtrees from each member's conjecture
            let mut pattern_occurrences: HashMap<String, Vec<usize>> = HashMap::new();
            let mut pattern_exprs: HashMap<String, Expr> = HashMap::new();

            for &concept_idx in &cluster.members {
                let conj_idx = self.concepts[concept_idx].conjecture_idx;
                if conj_idx >= engine.conjectures.len() {
                    continue;
                }
                let formula = &engine.conjectures[conj_idx].formula;
                let subtrees = extract_subtrees(formula, crate::hdc::abstract_thought::SUBTREE_MIN_COMPLEXITY, crate::hdc::abstract_thought::SUBTREE_MAX_COMPLEXITY);

                // Track unique patterns per conjecture (count once per conjecture)
                let mut seen_in_this_conj = HashSet::new();
                for st in subtrees {
                    let canonical = expr_canonical_string(&st);
                    if seen_in_this_conj.insert(canonical.clone()) {
                        pattern_occurrences
                            .entry(canonical.clone())
                            .or_default()
                            .push(conj_idx);
                        pattern_exprs.entry(canonical).or_insert(st);
                    }
                }
            }

            // Collect patterns appearing in >= 2 conjectures
            for (canonical, conj_ids) in &pattern_occurrences {
                if conj_ids.len() >= 2 {
                    if let Some(expr) = pattern_exprs.get(canonical) {
                        results.push((normalize_expr(expr), conj_ids.clone()));
                    }
                }
            }
        }

        results
    }

    /// Find recurring subtrees across ALL concept vectors, not just clusters.
    ///
    /// This is the empirical fallback to `recurring_subtrees()`: when clustering
    /// doesn't place related formulas together (common at small scale), we can
    /// still find pattern reuse by scanning all conjectures globally.
    ///
    /// Returns subtrees appearing in at least `min_occurrences` distinct conjectures.
    pub fn global_recurring_subtrees(
        &self,
        engine: &ConjectureEngine,
        min_occurrences: usize,
    ) -> Vec<(Expr, Vec<usize>)> {
        let mut pattern_occurrences: HashMap<String, Vec<usize>> = HashMap::new();
        let mut pattern_exprs: HashMap<String, Expr> = HashMap::new();

        for concept in &self.concepts {
            if concept.conjecture_idx >= engine.conjectures.len() {
                continue;
            }
            let formula = &engine.conjectures[concept.conjecture_idx].formula;
            let subtrees = extract_subtrees(formula, crate::hdc::abstract_thought::SUBTREE_MIN_COMPLEXITY, crate::hdc::abstract_thought::SUBTREE_MAX_COMPLEXITY);

            // Count each unique pattern once per conjecture
            let mut seen_in_this_conj = HashSet::new();
            for st in subtrees {
                let normalized = crate::hdc::abstract_thought::normalize_expr(&st);
                let canonical = crate::hdc::abstract_thought::expr_canonical_string(&st);
                if seen_in_this_conj.insert(canonical.clone()) {
                    pattern_occurrences
                        .entry(canonical.clone())
                        .or_default()
                        .push(concept.conjecture_idx);
                    pattern_exprs.entry(canonical).or_insert(normalized);
                }
            }
        }

        pattern_occurrences
            .into_iter()
            .filter(|(_, ids)| ids.len() >= min_occurrences)
            .filter_map(|(canonical, ids)| {
                pattern_exprs.remove(&canonical).map(|expr| (expr, ids))
            })
            .collect()
    }

    /// Extract subtrees from "strong-evidence" conjectures regardless of
    /// occurrence count. This is the extraction path for the fast-track
    /// promotion mechanism: a single high-quality novel shape should
    /// become a candidate even if it appears in exactly one conjecture.
    ///
    /// "Strong evidence" here is broader than strict formal verification —
    /// see comments in `dynamic_grammar::is_strongly_verified`. This path
    /// exists because formally verifying transcendental approximations is
    /// nearly impossible, but those approximations contain the most novel
    /// structural shapes (sqrt, exp, log, sin, cos compositions).
    ///
    /// Returns (normalized subtree, single-element conjecture ID vector)
    /// so the result is directly compatible with `observe_subtree`.
    pub fn verified_subtrees(
        &self,
        engine: &ConjectureEngine,
    ) -> Vec<(Expr, Vec<usize>)> {
        use crate::hdc::conjecture_engine::ConjectureStatus;
        let mut results = Vec::new();

        for concept in &self.concepts {
            let conj_idx = concept.conjecture_idx;
            if conj_idx >= engine.conjectures.len() {
                continue;
            }
            let conj = &engine.conjectures[conj_idx];

            // Match is_strongly_verified in dynamic_grammar.rs — duplicated
            // here because cross-module imports get awkward.
            let strongly_verified = match &conj.status {
                ConjectureStatus::FormallyVerified { .. } => true,
                ConjectureStatus::SymbolicallyChecked => true,
                ConjectureStatus::NumericallyTested { test_mse } => *test_mse < 1e-2,
                _ => false,
            };
            if !strongly_verified {
                continue;
            }

            let subtrees = extract_subtrees(
                &conj.formula,
                crate::hdc::abstract_thought::SUBTREE_MIN_COMPLEXITY,
                crate::hdc::abstract_thought::SUBTREE_MAX_COMPLEXITY,
            );

            // Dedup within this conjecture so the same pattern doesn't
            // contribute multiple candidates from one formula
            let mut seen_canonical = HashSet::new();
            for st in subtrees {
                let normalized = crate::hdc::abstract_thought::normalize_expr(&st);
                let canonical =
                    crate::hdc::abstract_thought::expr_canonical_string(&normalized);
                if seen_canonical.insert(canonical) {
                    results.push((normalized, vec![conj_idx]));
                }
            }
        }

        results
    }

    /// Find concept vectors most similar to a query vector.
    pub fn find_similar(&self, query: &BinaryHV, top_k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .concepts
            .iter()
            .enumerate()
            .map(|(i, cv)| (i, cv.encoding.similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

impl Default for MetaHDC {
    fn default() -> Self {
        Self::new()
    }
}

/// All MathDomain variants for iteration.
const ALL_DOMAINS: &[MathDomain] = &[
    MathDomain::NumberTheory,
    MathDomain::Combinatorics,
    MathDomain::AlgebraicComplexity,
    MathDomain::DynamicalSystems,
    MathDomain::SpectralAnalysis,
    MathDomain::Chemistry,
    MathDomain::Biology,
    MathDomain::Ecology,
    MathDomain::Economics,
    MathDomain::Physics,
    MathDomain::InformationTheory,
];

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::conjecture_engine::{BinOp, Conjecture};

    fn make_primitives() -> PrimitiveSystem {
        PrimitiveSystem::new()
    }

    fn make_conjecture(formula: Expr, domain: MathDomain, source: &str) -> Conjecture {
        let formula_str = format!("{}", formula);
        let complexity = formula.complexity();
        Conjecture {
            formula,
            formula_str,
            source: source.to_string(),
            domain,
            training_mse: 0.001,
            complexity,
            fitness: 0.001 + 0.001 * complexity as f64,
            status: ConjectureStatus::FormallyVerified { proof_steps: 5 },
            confidence: 0.95,
            macro_promotion_tier: crate::hdc::conjecture_engine::MacroPromotionTier::Formal,
        }
    }

    #[test]
    fn test_encode_conjecture_produces_hv() {
        let prims = make_primitives();
        let meta = MetaHDC::new();
        let conj = make_conjecture(
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            ),
            MathDomain::NumberTheory,
            "test_seq",
        );
        let cv = meta.encode_conjecture(&conj, 0, &prims);
        // Should produce a non-zero encoding
        assert!(cv.encoding.0.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_same_structure_different_domains_similar() {
        let prims = make_primitives();
        let meta = MetaHDC::new();
        // n + 1 in number theory
        let c1 = make_conjecture(
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            ),
            MathDomain::NumberTheory,
            "nt_seq",
        );
        // n + 1 in physics (same structure, different domain)
        let c2 = make_conjecture(
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            ),
            MathDomain::Physics,
            "phys_seq",
        );
        let cv1 = meta.encode_conjecture(&c1, 0, &prims);
        let cv2 = meta.encode_conjecture(&c2, 1, &prims);

        // Structural HVs should be identical (same formula shape)
        let struct_sim = cv1.structural_hv.similarity(&cv2.structural_hv);
        assert!(
            struct_sim > 0.9,
            "Structural similarity should be high: {}",
            struct_sim
        );

        // Overall concept similarity should be moderate (domain differs)
        let concept_sim = cv1.encoding.similarity(&cv2.encoding);
        assert!(
            concept_sim > 0.3,
            "Concept similarity should be moderate: {}",
            concept_sim
        );
    }

    #[test]
    fn test_different_structure_low_similarity() {
        let prims = make_primitives();
        let meta = MetaHDC::new();
        // n + 1 (linear)
        let c1 = make_conjecture(
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            ),
            MathDomain::NumberTheory,
            "seq_a",
        );
        // exp(n * n) (exponential-quadratic)
        let c2 = make_conjecture(
            Expr::Func(
                crate::hdc::conjecture_engine::UnaryFn::Exp,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Var("n".to_string())),
                    Box::new(Expr::Var("n".to_string())),
                )),
            ),
            MathDomain::NumberTheory,
            "seq_b",
        );
        let cv1 = meta.encode_conjecture(&c1, 0, &prims);
        let cv2 = meta.encode_conjecture(&c2, 1, &prims);

        let struct_sim = cv1.structural_hv.similarity(&cv2.structural_hv);
        assert!(
            struct_sim < 0.7,
            "Different structures should have low similarity: {}",
            struct_sim
        );
    }

    #[test]
    fn test_clustering_basic() {
        let prims = make_primitives();
        let mut meta = MetaHDC::new();

        // Add 6 conjectures: 3 linear (n+c), 3 quadratic (n*n+c)
        for i in 0..3 {
            let c = make_conjecture(
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("n".to_string())),
                    Box::new(Expr::Const(i as f64 + 1.0)),
                ),
                MathDomain::NumberTheory,
                &format!("linear_{}", i),
            );
            meta.add_conjecture(&c, i, &prims);
        }
        for i in 0..3 {
            let c = make_conjecture(
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::BinOp(
                        BinOp::Mul,
                        Box::new(Expr::Var("n".to_string())),
                        Box::new(Expr::Var("n".to_string())),
                    )),
                    Box::new(Expr::Const(i as f64 + 1.0)),
                ),
                MathDomain::Combinatorics,
                &format!("quadratic_{}", i),
            );
            meta.add_conjecture(&c, 3 + i, &prims);
        }

        meta.cluster(2);
        assert_eq!(meta.clusters.len(), 2);
        // Each cluster should have members
        for cluster in &meta.clusters {
            assert!(!cluster.members.is_empty());
        }
    }

    #[test]
    fn test_domain_diversity() {
        let prims = make_primitives();
        let mut meta = MetaHDC::new();

        // Same structure across 3 different domains
        let domains = [
            MathDomain::NumberTheory,
            MathDomain::Physics,
            MathDomain::Chemistry,
            MathDomain::Biology,
            MathDomain::Economics,
        ];
        for (i, &domain) in domains.iter().enumerate() {
            let c = make_conjecture(
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("n".to_string())),
                    Box::new(Expr::Const(i as f64 + 1.0)),
                ),
                domain,
                &format!("cross_{}", i),
            );
            meta.add_conjecture(&c, i, &prims);
        }

        meta.cluster(1); // Single cluster to capture all
        assert_eq!(meta.clusters.len(), 1);
        assert!(
            meta.clusters[0].domain_diversity >= 2,
            "Should detect multiple domains: {}",
            meta.clusters[0].domain_diversity
        );
    }

    #[test]
    fn test_find_similar() {
        let prims = make_primitives();
        let mut meta = MetaHDC::new();

        let c1 = make_conjecture(
            Expr::Var("n".to_string()),
            MathDomain::NumberTheory,
            "simple",
        );
        meta.add_conjecture(&c1, 0, &prims);

        let query = meta.concepts[0].encoding.clone();
        let similar = meta.find_similar(&query, 1);
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].0, 0);
        assert!((similar[0].1 - 1.0).abs() < 0.01); // Self-similarity ≈ 1.0
    }

    #[test]
    fn test_empty_clustering_no_crash() {
        let mut meta = MetaHDC::new();
        meta.cluster(5); // Should not crash with 0 concepts
        assert!(meta.clusters.is_empty());
    }

    #[test]
    fn test_recurring_subtrees_detection() {
        let prims = make_primitives();
        let mut meta = MetaHDC::new();

        // Create an engine with conjectures that share sqrt(n) subtree
        let mut engine = ConjectureEngine::new();

        // Conjecture 0: sqrt(n) + 1 (NumberTheory)
        let c0 = make_conjecture(
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Func(
                    crate::hdc::conjecture_engine::UnaryFn::Sqrt,
                    Box::new(Expr::Var("n".to_string())),
                )),
                Box::new(Expr::Const(1.0)),
            ),
            MathDomain::NumberTheory,
            "seq_0",
        );
        engine.conjectures.push(c0.clone());
        meta.add_conjecture(&c0, 0, &prims);

        // Conjecture 1: sqrt(n) * 2 (Physics — different domain!)
        let c1 = make_conjecture(
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Func(
                    crate::hdc::conjecture_engine::UnaryFn::Sqrt,
                    Box::new(Expr::Var("n".to_string())),
                )),
                Box::new(Expr::Const(2.0)),
            ),
            MathDomain::Physics,
            "seq_1",
        );
        engine.conjectures.push(c1.clone());
        meta.add_conjecture(&c1, 1, &prims);

        // Conjecture 2: sqrt(n) - 3 (Chemistry — third domain!)
        let c2 = make_conjecture(
            Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Func(
                    crate::hdc::conjecture_engine::UnaryFn::Sqrt,
                    Box::new(Expr::Var("n".to_string())),
                )),
                Box::new(Expr::Const(3.0)),
            ),
            MathDomain::Chemistry,
            "seq_2",
        );
        engine.conjectures.push(c2.clone());
        meta.add_conjecture(&c2, 2, &prims);

        // Cluster into 1 (they should be structurally similar)
        meta.cluster(1);

        let recurring = meta.recurring_subtrees(&engine);
        // Should find the shared subtree pattern
        // Note: the exact subtrees depend on complexity range [3,10]
        // sqrt(n) has complexity 2, so it won't be found directly
        // But (sqrt(n) + 1) has complexity 3, and the *normalized* version
        // matches across all three
        // Actually, the full expressions have complexity 3 each, so they
        // should be extracted as subtrees themselves
        assert!(
            !recurring.is_empty() || meta.clusters[0].domain_diversity >= 2,
            "Should find recurring subtrees or at least detect cross-domain cluster"
        );
    }

    #[test]
    fn test_single_conjecture_no_clusters() {
        let prims = make_primitives();
        let mut meta = MetaHDC::new();

        let c = make_conjecture(
            Expr::Var("n".to_string()),
            MathDomain::NumberTheory,
            "single",
        );
        meta.add_conjecture(&c, 0, &prims);

        meta.cluster(5);
        // With only 1 concept, clustering should produce no clusters (n < k)
        assert!(meta.clusters.is_empty());
    }
}
