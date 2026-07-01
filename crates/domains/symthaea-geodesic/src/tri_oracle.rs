// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Tri-Oracle — Three Discriminative Scoring Layers
//!
//! Three oracles, each capturing a different aspect of program correctness:
//!
//! 1. **Structural** (O(1)): HDC encoding similarity — catches wrong operators,
//!    swapped arguments, wrong nesting. The encoding preserves structure by design.
//!
//! 2. **Topological** (O(n)): PDG Betti number comparison — catches missing loops,
//!    extra branches, dead code. Programs with different shapes score differently.
//!
//! 3. **Trajectory** (O(k)): CfC evolution divergence — compares how TWO programs'
//!    states evolve over time. Similar programs evolve similarly; different diverge.
//!    (This is the reimagined execution oracle — comparing trajectories, not endpoints.)
//!
//! Consensus: `composite = w₁×structural + w₂×topological + w₃×trajectory`
//!
//! ## Why Three Oracles Beat One
//!
//! A correct program scores high on ALL THREE. An incorrect program typically
//! fails at least one:
//! - Wrong operator → structural oracle catches it
//! - Missing loop → topological oracle catches it
//! - Same structure, different behavior → trajectory oracle catches it

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;

use crate::execution_oracle::{ExecutionOracle, OperationType};
use crate::pdg::ProgramDependenceGraph;
use crate::program_emitter::emit_expression;
use crate::topology::TopologicalFingerprint;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Scores from each of the three oracles.
#[derive(Debug, Clone)]
pub struct TriOracleScore {
    /// Oracle 1: HDC encoding similarity [0.0, 1.0]
    pub structural: f32,
    /// Oracle 2: Betti number match [0.0, 1.0]
    pub topological: f32,
    /// Oracle 3: CfC trajectory similarity [0.0, 1.0]
    pub trajectory: f32,
    /// Weighted consensus score
    pub composite: f32,
}

/// Configuration for the tri-oracle.
#[derive(Debug, Clone)]
pub struct TriOracleConfig {
    /// Weight for structural oracle (default 0.5)
    pub structural_weight: f32,
    /// Weight for topological oracle (default 0.2)
    pub topological_weight: f32,
    /// Weight for trajectory oracle (default 0.3)
    pub trajectory_weight: f32,
    /// Number of CfC evolution steps for trajectory comparison (default 10)
    pub trajectory_steps: usize,
}

impl Default for TriOracleConfig {
    fn default() -> Self {
        Self {
            structural_weight: 0.5,
            topological_weight: 0.2,
            trajectory_weight: 0.3,
            trajectory_steps: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRI-ORACLE
// ═══════════════════════════════════════════════════════════════════════════════

/// Three discriminative oracles with consensus scoring.
pub struct TriOracle {
    config: TriOracleConfig,
}

impl TriOracle {
    pub fn new(config: TriOracleConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(TriOracleConfig::default())
    }

    /// Score a candidate ProgramNode against an expected ProgramNode.
    /// Uses all three oracles and returns consensus.
    pub fn score(&self, candidate: &ProgramNode, expected: &ProgramNode) -> TriOracleScore {
        // Blend property-based structural encoding (name-independent, for composition)
        // with name-based encoding (operator-discriminative, for scoring).
        // Structural encoding makes ADD(a,b) ≈ ADD(x,y) but also ADD(a,b) ≈ MUL(a,b).
        // Name encoding makes ADD(a,b) ≈ ADD(x,y)=0.5 but ADD(a,b) ≠ MUL(a,b).
        // Blend: 50% structural + 50% name → best of both.
        let structural_enc_c = crate::periodic_table::encode_structural(candidate);
        let structural_enc_e = crate::periodic_table::encode_structural(expected);
        let name_enc_c = candidate.encode();
        let name_enc_e = expected.encode();

        let structural_sim = structural_enc_c.similarity(&structural_enc_e);
        let name_sim = name_enc_c.similarity(&name_enc_e);
        let structural = 0.5 * structural_sim + 0.5 * name_sim;

        // Use name-based encoding for trajectory (it needs operator distinction)
        let candidate_enc = name_enc_c;
        let expected_enc = name_enc_e;

        // Oracle 2: Topological (PDG Betti comparison)
        let topological = Self::topological_score(candidate, expected);

        // Oracle 3: Trajectory (CfC evolution divergence)
        let trajectory =
            Self::trajectory_score(&candidate_enc, &expected_enc, self.config.trajectory_steps);

        // Consensus
        let composite = self.config.structural_weight * structural
            + self.config.topological_weight * topological
            + self.config.trajectory_weight * trajectory;

        TriOracleScore {
            structural,
            topological,
            trajectory,
            composite,
        }
    }

    /// Score encodings directly (fastest path — structural + trajectory only).
    pub fn score_encodings(&self, candidate: &BinaryHV, expected: &BinaryHV) -> TriOracleScore {
        let structural = Self::structural_score(candidate, expected);
        let topological = 0.5; // neutral — no ProgramNode available for PDG
        let trajectory = Self::trajectory_score(candidate, expected, self.config.trajectory_steps);

        let composite = self.config.structural_weight * structural
            + self.config.topological_weight * topological
            + self.config.trajectory_weight * trajectory;

        TriOracleScore {
            structural,
            topological,
            trajectory,
            composite,
        }
    }

    // ── Oracle 1: Structural ────────────────────────────────────────────

    /// Direct HDC encoding similarity.
    ///
    /// Since ProgramNode encoding preserves operator identity through role-binding,
    /// different operators produce orthogonal HVs → low similarity.
    /// Same program → similarity ≈ 1.0. Wrong operator → similarity ≈ 0.5.
    fn structural_score(candidate: &BinaryHV, expected: &BinaryHV) -> f32 {
        candidate.similarity(expected)
    }

    /// Sub-tree decomposition for deeper structural matching.
    ///
    /// Decomposes both ProgramNodes into their immediate children,
    /// encodes each child, and averages the per-child similarities.
    /// This catches partial matches (right operator + wrong arg).
    #[allow(dead_code)]
    fn structural_subtree_score(candidate: &ProgramNode, expected: &ProgramNode) -> f32 {
        let c_children = extract_children(candidate);
        let e_children = extract_children(expected);

        if c_children.is_empty() && e_children.is_empty() {
            // Both are leaves — compare encodings directly
            return candidate.encode().similarity(&expected.encode());
        }

        if c_children.is_empty() || e_children.is_empty() {
            // Different structure (one is leaf, other is composite)
            return candidate.encode().similarity(&expected.encode()) * 0.5;
        }

        // Compare matched children by position
        let pairs = c_children.len().min(e_children.len());
        let mut total_sim = 0.0_f32;
        for i in 0..pairs {
            total_sim += c_children[i].encode().similarity(&e_children[i].encode());
        }

        // Penalize for different number of children
        let length_penalty = pairs as f32 / c_children.len().max(e_children.len()) as f32;

        (total_sim / pairs as f32) * length_penalty
    }

    // ── Oracle 2: Topological ───────────────────────────────────────────

    /// PDG Betti number comparison.
    ///
    /// Emits both ProgramNodes as Rust expressions, builds PDGs, computes
    /// topological fingerprints, and compares Betti numbers.
    fn topological_score(candidate: &ProgramNode, expected: &ProgramNode) -> f32 {
        let c_source = emit_expression(candidate);
        let e_source = emit_expression(expected);

        let c_pdg = ProgramDependenceGraph::from_rust_source(&c_source, "candidate");
        let e_pdg = ProgramDependenceGraph::from_rust_source(&e_source, "expected");

        let c_fp = TopologicalFingerprint::from_complex(&c_pdg.to_simplicial_complex());
        let e_fp = TopologicalFingerprint::from_complex(&e_pdg.to_simplicial_complex());

        // Score: 1.0 if all Betti numbers match, penalize per mismatch
        let mut score = 1.0_f32;
        if c_fp.betti.beta_0 != e_fp.betti.beta_0 {
            score -= 0.2;
        }
        if c_fp.betti.beta_1 != e_fp.betti.beta_1 {
            score -= 0.4; // Loops matter most
        }
        if c_fp.betti.beta_2 != e_fp.betti.beta_2 {
            score -= 0.2;
        }

        // Also compare structural size (node/edge counts)
        let size_ratio = if e_fp.node_count > 0 {
            1.0 - ((c_fp.node_count as f32 - e_fp.node_count as f32).abs() / e_fp.node_count as f32)
                .min(1.0)
                * 0.2
        } else {
            1.0
        };

        (score * size_ratio).max(0.0)
    }

    // ── Oracle 3: Trajectory ────────────────────────────────────────────

    /// CfC trajectory divergence.
    ///
    /// Runs TWO execution oracles in parallel — one with the candidate
    /// encoding, one with the expected encoding. At each step, measures
    /// how similar their internal states are. Similar programs produce
    /// similar trajectories; different programs diverge.
    ///
    /// **This is the key fix**: the old oracle compared one program's state
    /// to an external "expected" HV. This oracle compares two programs'
    /// states against EACH OTHER as they evolve.
    fn trajectory_score(candidate: &BinaryHV, expected: &BinaryHV, steps: usize) -> f32 {
        if steps == 0 {
            return candidate.similarity(expected);
        }

        let mut oracle_c = ExecutionOracle::new();
        let mut oracle_e = ExecutionOracle::new();

        let mut total_sim = 0.0_f32;

        for step in 0..steps {
            // Evolve both with their respective encodings
            // Use different operation types across steps for richer dynamics
            let op = match step % 4 {
                0 => OperationType::Arithmetic,
                1 => OperationType::Assignment,
                2 => OperationType::Comparison,
                _ => OperationType::Arithmetic,
            };

            oracle_c.execute_step(candidate, op);
            oracle_e.execute_step(expected, op);

            // Compare states at each step
            let step_sim = oracle_c.state().similarity(oracle_e.state());
            total_sim += step_sim;
        }

        total_sim / steps as f32
    }
}

/// Extract immediate children of a ProgramNode for sub-tree comparison.
fn extract_children(node: &ProgramNode) -> Vec<&ProgramNode> {
    match node {
        ProgramNode::Atom(_) | ProgramNode::Typed(_, _) => vec![],
        ProgramNode::Apply { func, args } => {
            let mut children = vec![func.as_ref()];
            children.extend(args.iter());
            children
        }
        ProgramNode::Sequence(steps) => steps.iter().collect(),
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => vec![
            condition.as_ref(),
            then_branch.as_ref(),
            else_branch.as_ref(),
        ],
        ProgramNode::Map { func, collection } => vec![func.as_ref(), collection.as_ref()],
        ProgramNode::Filter {
            predicate,
            collection,
        } => vec![predicate.as_ref(), collection.as_ref()],
        ProgramNode::Reduce {
            func,
            initial,
            collection,
        } => vec![func.as_ref(), initial.as_ref(), collection.as_ref()],
        ProgramNode::Compose(f, g) => vec![f.as_ref(), g.as_ref()],
        ProgramNode::Recurse {
            base_case,
            recursive_step,
        } => vec![base_case.as_ref(), recursive_step.as_ref()],
        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => vec![init.as_ref(), step.as_ref(), condition.as_ref()],
        ProgramNode::Collect(source) => vec![source.as_ref()],
        ProgramNode::Abstract(examples) => examples.iter().collect(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn add_ab() -> ProgramNode {
        ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        )
    }

    fn mul_ab() -> ProgramNode {
        ProgramNode::apply(
            ProgramNode::op("MUL"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        )
    }

    #[test]
    fn test_structural_same_program() {
        let a = add_ab();
        let score = TriOracle::structural_score(&a.encode(), &a.encode());
        assert!(
            (score - 1.0).abs() < 0.01,
            "same program should have structural similarity ~1.0, got {score}"
        );
    }

    #[test]
    fn test_structural_different_operator() {
        let add = add_ab();
        let mul = mul_ab();
        let score = TriOracle::structural_score(&add.encode(), &mul.encode());
        // Different operators produce orthogonal HVs → sim ≈ 0.5
        assert!(
            score < 0.7,
            "different operators should have low structural similarity, got {score}"
        );
    }

    #[test]
    fn test_topological_same_structure() {
        let a = add_ab();
        let b = mul_ab();
        // Both are Apply(op, [atom, atom]) → same topology
        let score = TriOracle::topological_score(&a, &b);
        assert!(
            score > 0.8,
            "same-structure programs should have high topological score, got {score}"
        );
    }

    #[test]
    fn test_topological_different_structure() {
        let simple = add_ab();
        let looped = ProgramNode::iterate(
            ProgramNode::atom("init"),
            ProgramNode::atom("step"),
            ProgramNode::atom("cond"),
        );
        let score = TriOracle::topological_score(&simple, &looped);
        // Different topology (β₁=0 vs β₁≥1) → lower score
        assert!(
            score < 1.0,
            "different-topology programs should score lower, got {score}"
        );
    }

    #[test]
    fn test_trajectory_same_encoding() {
        let a = add_ab().encode();
        let score = TriOracle::trajectory_score(&a, &a, 10);
        assert!(
            score > 0.95,
            "same encoding should produce near-identical trajectories, got {score}"
        );
    }

    #[test]
    fn test_trajectory_different_encoding() {
        let add = add_ab().encode();
        let mul = mul_ab().encode();
        let score = TriOracle::trajectory_score(&add, &mul, 10);
        // Different encodings should produce divergent trajectories
        // (lower than same-encoding but may still be moderate)
        assert!(
            score < 0.95,
            "different encodings should produce somewhat divergent trajectories, got {score}"
        );
    }

    #[test]
    fn test_consensus_correct_higher_than_wrong() {
        let oracle = TriOracle::with_defaults();
        let correct = add_ab();
        let wrong = mul_ab();
        let expected = add_ab();

        let correct_score = oracle.score(&correct, &expected);
        let wrong_score = oracle.score(&wrong, &expected);

        println!(
            "Correct: structural={:.4} topo={:.4} traj={:.4} composite={:.4}",
            correct_score.structural,
            correct_score.topological,
            correct_score.trajectory,
            correct_score.composite,
        );
        println!(
            "Wrong:   structural={:.4} topo={:.4} traj={:.4} composite={:.4}",
            wrong_score.structural,
            wrong_score.topological,
            wrong_score.trajectory,
            wrong_score.composite,
        );

        assert!(
            correct_score.composite > wrong_score.composite,
            "correct program ({:.4}) should score higher than wrong ({:.4})",
            correct_score.composite,
            wrong_score.composite,
        );
    }

    #[test]
    fn test_config_weights_sum_to_one() {
        let config = TriOracleConfig::default();
        let sum = config.structural_weight + config.topological_weight + config.trajectory_weight;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_score_encodings_fast_path() {
        let oracle = TriOracle::with_defaults();
        let a = add_ab().encode();
        let b = mul_ab().encode();
        let score = oracle.score_encodings(&a, &b);
        // Should produce a score (not panic) even without ProgramNode
        assert!(score.composite >= 0.0 && score.composite <= 1.0);
    }
}
