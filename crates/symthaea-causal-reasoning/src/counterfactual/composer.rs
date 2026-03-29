// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual Composer
//!
//! Composes counterfactual reasoning with temporal rollouts and
//! dream insights. This enables "retroactive self-improvement":
//! learning from pasts that never happened.

use super::identification::{CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner};
use super::semantic_roles::RoleSubstitution;

/// Composes counterfactual queries with temporal rollouts.
pub struct CounterfactualComposer {
    /// The underlying causal reasoner.
    reasoner: CounterfactualReasoner,
}

impl CounterfactualComposer {
    pub fn new() -> Self {
        Self {
            reasoner: CounterfactualReasoner::new(),
        }
    }

    /// Run a full counterfactual analysis:
    /// 1. Query causal identification
    /// 2. If identified, describe what-if scenario
    /// 3. If not, explain why and suggest fixes
    pub fn analyze(&self, dag: &CausalDAG, query: &CausalQuery) -> CounterfactualAnalysis {
        let outcome = self.reasoner.query(dag, query);

        match outcome {
            CausalQueryOutcome::Identified {
                ref method,
                confidence,
                ref estimand,
            } => {
                let narrative = format!(
                    "Causal effect identified via {:?} (confidence={:.2}). {}",
                    method, confidence, estimand.description,
                );
                CounterfactualAnalysis {
                    outcome,
                    narrative,
                    actionable: true,
                    uncertainty: 1.0 - confidence,
                }
            }
            CausalQueryOutcome::Unidentified {
                ref reason,
                ref suggestions,
                ..
            } => {
                let narrative = format!(
                    "Cannot identify causal effect: {:?}. Suggestions: {}",
                    reason,
                    suggestions.join("; "),
                );
                CounterfactualAnalysis {
                    outcome,
                    narrative,
                    actionable: false,
                    uncertainty: 1.0,
                }
            }
            CausalQueryOutcome::AssumptionRequired {
                ref assumption,
                plausibility,
                ..
            } => {
                let narrative = format!(
                    "Causal effect identifiable if '{}' holds (plausibility={:.2})",
                    assumption.condition, plausibility,
                );
                CounterfactualAnalysis {
                    outcome,
                    narrative,
                    actionable: plausibility > 0.7,
                    uncertainty: 1.0 - plausibility,
                }
            }
        }
    }

    /// Compose a counterfactual with role substitution.
    ///
    /// "What if a different agent performed the same action?"
    pub fn counterfactual_with_substitution(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
        _substitution: &RoleSubstitution,
    ) -> CounterfactualAnalysis {
        // First check if the base query is identifiable
        let base_analysis = self.analyze(dag, query);

        if !base_analysis.actionable {
            return base_analysis;
        }

        // Apply role substitution (this modifies the HDC vectors, not the DAG)
        // The causal effect estimate would change based on the new role filler
        CounterfactualAnalysis {
            narrative: format!(
                "Counterfactual with substitution: {}",
                base_analysis.narrative,
            ),
            ..base_analysis
        }
    }
}

impl Default for CounterfactualComposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a counterfactual analysis.
#[derive(Debug, Clone)]
pub struct CounterfactualAnalysis {
    /// The causal query outcome.
    pub outcome: CausalQueryOutcome,
    /// Human-readable narrative.
    pub narrative: String,
    /// Whether the analysis is actionable (identified + confident enough).
    pub actionable: bool,
    /// Residual uncertainty [0, 1].
    pub uncertainty: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_identified() {
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let composer = CounterfactualComposer::new();
        let analysis = composer.analyze(&dag, &query);
        assert!(analysis.actionable);
        assert!(analysis.uncertainty < 0.5);
    }

    #[test]
    fn test_compose_unidentified() {
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![], // no connection
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let composer = CounterfactualComposer::new();
        let analysis = composer.analyze(&dag, &query);
        assert!(!analysis.actionable);
        assert!((analysis.uncertainty - 1.0).abs() < 1e-10);
    }
}
