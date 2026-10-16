// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded minimal-witness evaluation for alternative and joint evidence proofs.
//!
//! The derivation graph records where evidence came from, but a flat root set
//! loses logical shape: is one source sufficient, are two sources jointly needed,
//! or are there alternative ways to establish a result? This module models those
//! combinations explicitly and computes minimal evidence-witness sets.
//!
//! This is inspired by database provenance semirings and why-provenance. It does
//! not calculate truth, probabilities, source independence, or calibrated belief.
//! An Evidence leaf must be a stable, claim-specific evidence-witness ID; it is
//! not merely a URL or publisher identity. Origin lineage can be resolved later
//! for each witness using the provenance graph.
//!
//! The evaluator is deliberately bounded. Combinatorial proof expansion can grow
//! exponentially, so exceeding any budget returns Abstain with no partial witness
//! set. It never truncates a proof result and presents that subset as complete.

use std::collections::BTreeSet;

/// Logical structure of one evidence derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceProofFormula {
    /// One claim-specific evidence witness.
    Evidence(String),
    /// Joint derivation: every child is required for this proof.
    All(Vec<EvidenceProofFormula>),
    /// Alternative derivation: any child is a sufficient proof path.
    Any(Vec<EvidenceProofFormula>),
}

/// Limits on input complexity and minimal-witness expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProofPolicy {
    pub max_expression_nodes: usize,
    pub max_depth: usize,
    /// Maximum number of incomparable minimal witness sets returned.
    pub max_witness_sets: usize,
    /// Maximum cross-product candidate combinations during conjunction expansion.
    pub max_intermediate_combinations: usize,
}

impl EvidenceProofPolicy {
    pub fn new(
        max_expression_nodes: usize,
        max_depth: usize,
        max_witness_sets: usize,
        max_intermediate_combinations: usize,
    ) -> Result<Self, String> {
        if max_expression_nodes == 0
            || max_depth == 0
            || max_witness_sets == 0
            || max_intermediate_combinations == 0
        {
            return Err("all evidence proof budgets must be greater than zero".to_owned());
        }
        Ok(Self {
            max_expression_nodes,
            max_depth,
            max_witness_sets,
            max_intermediate_combinations,
        })
    }

    /// Conservative operational defaults. These are resource limits, not
    /// scientifically calibrated confidence thresholds.
    pub fn bounded_default() -> Self {
        Self {
            max_expression_nodes: 200_000,
            max_depth: 128,
            max_witness_sets: 1_024,
            max_intermediate_combinations: 200_000,
        }
    }
}

impl Default for EvidenceProofPolicy {
    fn default() -> Self {
        Self::bounded_default()
    }
}

/// Reasons a proof formula cannot be completely evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceProofAbstentionReason {
    EmptyEvidenceId,
    EmptyConjunction,
    EmptyDisjunction,
    ExpressionNodeBudgetExceeded,
    ExpressionDepthExceeded,
    WitnessSetBudgetExceeded,
    IntermediateCombinationBudgetExceeded,
}

/// Complete or fail-closed result. On abstention, minimal_witness_sets is None:
/// a partial enumeration must not be mistaken for a complete proof explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProofAssessment {
    pub complete: bool,
    pub minimal_witness_sets: Option<Vec<Vec<String>>>,
    pub reasons: Vec<EvidenceProofAbstentionReason>,
    pub expression_node_count: usize,
    /// Candidate combinations examined, useful for resource diagnostics.
    pub intermediate_combination_count: usize,
}

impl EvidenceProofAssessment {
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    fn abstain(
        reasons: Vec<EvidenceProofAbstentionReason>,
        expression_node_count: usize,
        intermediate_combination_count: usize,
    ) -> Self {
        Self {
            complete: false,
            minimal_witness_sets: None,
            reasons,
            expression_node_count,
            intermediate_combination_count,
        }
    }
}

fn add_reason_once(
    reasons: &mut Vec<EvidenceProofAbstentionReason>,
    reason: EvidenceProofAbstentionReason,
) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn validate_formula(
    formula: &EvidenceProofFormula,
    policy: &EvidenceProofPolicy,
) -> Result<usize, EvidenceProofAbstentionReason> {
    let mut stack = vec![(formula, 1usize)];
    let mut node_count = 0usize;

    while let Some((node, depth)) = stack.pop() {
        node_count += 1;
        if node_count > policy.max_expression_nodes {
            return Err(EvidenceProofAbstentionReason::ExpressionNodeBudgetExceeded);
        }
        if depth > policy.max_depth {
            return Err(EvidenceProofAbstentionReason::ExpressionDepthExceeded);
        }

        match node {
            EvidenceProofFormula::Evidence(id) if id.trim().is_empty() => {
                return Err(EvidenceProofAbstentionReason::EmptyEvidenceId);
            }
            EvidenceProofFormula::Evidence(_) => {}
            EvidenceProofFormula::All(children) if children.is_empty() => {
                return Err(EvidenceProofAbstentionReason::EmptyConjunction);
            }
            EvidenceProofFormula::Any(children) if children.is_empty() => {
                return Err(EvidenceProofAbstentionReason::EmptyDisjunction);
            }
            EvidenceProofFormula::All(children) | EvidenceProofFormula::Any(children) => {
                stack.extend(children.iter().map(|child| (child, depth + 1)));
            }
        }
    }

    Ok(node_count)
}

fn insert_minimal(
    frontier: &mut Vec<BTreeSet<String>>,
    candidate: BTreeSet<String>,
    policy: &EvidenceProofPolicy,
) -> Result<(), EvidenceProofAbstentionReason> {
    // Existing subset means the new candidate is not a minimal explanation.
    if frontier
        .iter()
        .any(|existing| existing.is_subset(&candidate))
    {
        return Ok(());
    }

    // Remove explanations made redundant by this strictly smaller candidate.
    frontier.retain(|existing| !candidate.is_subset(existing));
    frontier.push(candidate);
    frontier.sort_by(|left, right| {
        left.len()
            .cmp(&right.len())
            .then_with(|| left.iter().cmp(right.iter()))
    });

    if frontier.len() > policy.max_witness_sets {
        return Err(EvidenceProofAbstentionReason::WitnessSetBudgetExceeded);
    }
    Ok(())
}

fn evaluate_node(
    formula: &EvidenceProofFormula,
    policy: &EvidenceProofPolicy,
    combinations: &mut usize,
) -> Result<Vec<BTreeSet<String>>, EvidenceProofAbstentionReason> {
    match formula {
        EvidenceProofFormula::Evidence(id) => {
            Ok(vec![BTreeSet::from([id.trim().to_owned()])])
        }
        EvidenceProofFormula::Any(children) => {
            let mut frontier = Vec::new();
            for child in children {
                for candidate in evaluate_node(child, policy, combinations)? {
                    *combinations = combinations.saturating_add(1);
                    if *combinations > policy.max_intermediate_combinations {
                        return Err(
                            EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded,
                        );
                    }
                    insert_minimal(&mut frontier, candidate, policy)?;
                }
            }
            Ok(frontier)
        }
        EvidenceProofFormula::All(children) => {
            let mut current: Option<Vec<BTreeSet<String>>> = None;

            for child in children {
                let child_sets = evaluate_node(child, policy, combinations)?;
                let Some(mut accumulated) = current.take() else {
                    current = Some(child_sets);
                    continue;
                };

                // Common large conjunction: when both sides have exactly one
                // proof, union them in-place rather than cloning an ever-growing
                // set at every step.
                if accumulated.len() == 1 && child_sets.len() == 1 {
                    *combinations = combinations.saturating_add(1);
                    if *combinations > policy.max_intermediate_combinations {
                        return Err(
                            EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded,
                        );
                    }
                    accumulated[0].extend(child_sets.into_iter().next().unwrap());
                    current = Some(accumulated);
                    continue;
                }

                let product = accumulated
                    .len()
                    .checked_mul(child_sets.len())
                    .ok_or(EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded)?;
                let total = combinations
                    .checked_add(product)
                    .ok_or(EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded)?;
                if total > policy.max_intermediate_combinations {
                    return Err(
                        EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded,
                    );
                }

                let mut next_frontier = Vec::new();
                for left in &accumulated {
                    for right in &child_sets {
                        *combinations += 1;
                        let mut combined = left.clone();
                        combined.extend(right.iter().cloned());
                        insert_minimal(&mut next_frontier, combined, policy)?;
                    }
                }
                current = Some(next_frontier);
            }

            current.ok_or(EvidenceProofAbstentionReason::EmptyConjunction)
        }
    }
}

/// Evaluate a proof formula into a canonical antichain of minimal evidence sets.
impl EvidenceProofFormula {
    pub fn assess(&self, policy: &EvidenceProofPolicy) -> EvidenceProofAssessment {
        let node_count = match validate_formula(self, policy) {
            Ok(count) => count,
            Err(reason) => return EvidenceProofAssessment::abstain(vec![reason], 0, 0),
        };

        let mut combinations = 0usize;
        match evaluate_node(self, policy, &mut combinations) {
            Ok(frontier) => EvidenceProofAssessment {
                complete: true,
                minimal_witness_sets: Some(
                    frontier
                        .into_iter()
                        .map(|set| set.into_iter().collect())
                        .collect(),
                ),
                reasons: Vec::new(),
                expression_node_count: node_count,
                intermediate_combination_count: combinations,
            },
            Err(reason) => EvidenceProofAssessment::abstain(
                vec![reason],
                node_count,
                combinations,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str) -> EvidenceProofFormula {
        EvidenceProofFormula::Evidence(id.to_owned())
    }

    fn all(children: Vec<EvidenceProofFormula>) -> EvidenceProofFormula {
        EvidenceProofFormula::All(children)
    }

    fn any(children: Vec<EvidenceProofFormula>) -> EvidenceProofFormula {
        EvidenceProofFormula::Any(children)
    }

    fn default_policy() -> EvidenceProofPolicy {
        EvidenceProofPolicy::default()
    }

    #[test]
    fn conjunction_requires_every_evidence_atom() {
        let formula = all(vec![evidence("lab-result"), evidence("calibration-record")]);
        let result = formula.assess(&default_policy());

        assert!(result.complete);
        assert_eq!(
            result.minimal_witness_sets,
            Some(vec![vec![
                "calibration-record".to_owned(),
                "lab-result".to_owned()
            ]])
        );
    }

    #[test]
    fn disjunction_preserves_alternative_proof_paths() {
        let formula = any(vec![evidence("primary-study"), evidence("replication")]);
        let result = formula.assess(&default_policy());

        assert_eq!(
            result.minimal_witness_sets,
            Some(vec![
                vec!["primary-study".to_owned()],
                vec!["replication".to_owned()]
            ])
        );
    }

    #[test]
    fn alternatives_and_joint_inputs_are_not_interchanged() {
        let joint = all(vec![evidence("A"), evidence("B")]).assess(&default_policy());
        let alternate = any(vec![evidence("A"), evidence("B")]).assess(&default_policy());

        assert_eq!(joint.minimal_witness_sets.unwrap(), vec![vec!["A", "B"]]);
        assert_eq!(
            alternate.minimal_witness_sets.unwrap(),
            vec![vec!["A"], vec!["B"]]
        );
    }

    #[test]
    fn minimality_removes_superset_alternatives() {
        let formula = any(vec![
            evidence("A"),
            all(vec![evidence("A"), evidence("B")]),
            all(vec![evidence("C"), evidence("D")]),
        ]);
        let result = formula.assess(&default_policy());

        assert_eq!(
            result.minimal_witness_sets,
            Some(vec![vec!["A".to_owned()], vec!["C".to_owned(), "D".to_owned()]])
        );
    }

    #[test]
    fn repeated_evidence_atom_is_not_double_counted_in_one_joint_proof() {
        let formula = all(vec![evidence("A"), evidence("A")]);
        assert_eq!(
            formula.assess(&default_policy()).minimal_witness_sets,
            Some(vec![vec!["A".to_owned()]])
        );
    }

    #[test]
    fn nested_multi_parent_and_alternative_proofs_are_canonical() {
        let formula = any(vec![
            all(vec![evidence("A"), evidence("B")]),
            all(vec![evidence("A"), evidence("C")]),
        ]);
        let result = formula.assess(&default_policy());

        assert_eq!(
            result.minimal_witness_sets,
            Some(vec![
                vec!["A".to_owned(), "B".to_owned()],
                vec!["A".to_owned(), "C".to_owned()]
            ])
        );
    }

    #[test]
    fn empty_nodes_and_invalid_budgets_fail_closed() {
        assert!(EvidenceProofPolicy::new(0, 1, 1, 1).is_err());
        let empty_all = all(vec![]).assess(&default_policy());
        assert_eq!(
            empty_all.reasons,
            vec![EvidenceProofAbstentionReason::EmptyConjunction]
        );
        let empty_any = any(vec![]).assess(&default_policy());
        assert_eq!(
            empty_any.reasons,
            vec![EvidenceProofAbstentionReason::EmptyDisjunction]
        );
        let empty_id = evidence(" ").assess(&default_policy());
        assert_eq!(
            empty_id.reasons,
            vec![EvidenceProofAbstentionReason::EmptyEvidenceId]
        );
    }

    #[test]
    fn output_budget_abstains_without_returning_partial_witnesses() {
        let formula = any(vec![evidence("A"), evidence("B"), evidence("C")]);
        let policy = EvidenceProofPolicy::new(100, 10, 2, 100).unwrap();
        let result = formula.assess(&policy);

        assert!(!result.complete);
        assert_eq!(result.minimal_witness_sets, None);
        assert_eq!(
            result.reasons,
            vec![EvidenceProofAbstentionReason::WitnessSetBudgetExceeded]
        );
    }

    #[test]
    fn cartesian_explosion_abstains_instead_of_truncating() {
        let left = any(vec![evidence("A"), evidence("B"), evidence("C")]);
        let right = any(vec![evidence("D"), evidence("E"), evidence("F")]);
        let formula = all(vec![left, right]);
        let policy = EvidenceProofPolicy::new(100, 10, 100, 5).unwrap();
        let result = formula.assess(&policy);

        assert!(!result.complete);
        assert_eq!(result.minimal_witness_sets, None);
        assert!(result.reasons.contains(
            &EvidenceProofAbstentionReason::IntermediateCombinationBudgetExceeded
        ));
    }

    #[test]
    fn node_and_depth_budgets_are_checked_before_evaluation() {
        let formula = all(vec![evidence("A"), evidence("B")]);
        let node_limited = EvidenceProofPolicy::new(2, 10, 10, 10).unwrap();
        assert_eq!(
            formula.assess(&node_limited).reasons,
            vec![EvidenceProofAbstentionReason::ExpressionNodeBudgetExceeded]
        );

        let depth_limited = EvidenceProofPolicy::new(100, 2, 10, 10).unwrap();
        let deep = any(vec![all(vec![evidence("A")])]);
        assert_eq!(
            deep.assess(&depth_limited).reasons,
            vec![EvidenceProofAbstentionReason::ExpressionDepthExceeded]
        );
    }

    #[test]
    fn large_conjunction_is_linear_in_number_of_atoms_and_has_one_witness() {
        let formula = all(
            (0..5_000)
                .map(|index| evidence(&format!("evidence-{index}")))
                .collect(),
        );
        let result = formula.assess(&default_policy());
        assert!(result.complete);
        assert_eq!(result.minimal_witness_sets.as_ref().unwrap().len(), 1);
        assert_eq!(result.minimal_witness_sets.unwrap()[0].len(), 5_000);
    }

    #[test]
    fn benchmark_minimal_witness_evaluation_scaling() {
        let formula = any(vec![
            all(vec![evidence("sensor-A"), evidence("calibration-A")]),
            all(vec![evidence("sensor-B"), evidence("calibration-B")]),
        ]);
        let result = formula.assess(&default_policy());
        assert_eq!(result.minimal_witness_sets.as_ref().unwrap().len(), 2);
    }

    /// Opt-in release-mode microbenchmark. The formula is built outside timing;
    /// each sample measures validation and complete evaluation of the same DAG-shaped input.
    #[test]
    #[ignore = "manual release-mode scalability benchmark; not a CI correctness gate"]
    fn benchmark_minimal_witness_frontier_scaling() {
        use std::time::Instant;

        const REPEATS: usize = 9;
        let policy = default_policy();

        for size in [1_000usize, 10_000, 100_000] {
            let formula = all(
                (0..size)
                    .map(|index| evidence(&format!("evidence-{index}")))
                    .collect(),
            );
            let expected = formula.assess(&policy);
            assert!(expected.complete);
            assert_eq!(expected.minimal_witness_sets.as_ref().unwrap().len(), 1);
            assert_eq!(expected.minimal_witness_sets.as_ref().unwrap()[0].len(), size);

            let mut samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                let assessment = formula.assess(&policy);
                std::hint::black_box((
                    assessment.complete,
                    assessment.expression_node_count,
                    assessment.minimal_witness_sets.as_ref().map(Vec::len),
                ));
                samples.push(start.elapsed().as_nanos());
            }
            samples.sort_unstable();
            let p50 = samples[REPEATS / 2];
            let p95 = samples[(REPEATS * 95 + 99) / 100 - 1];
            println!(
                "atoms={} reps={} p50_ns={} p95_ns={} median_ns_per_atom={}",
                size,
                REPEATS,
                p50,
                p95,
                p50 / size as u128
            );
        }
    }
}
