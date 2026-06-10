// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal model construction from coding history for counterfactual reasoning.

use super::CodingAgent;
use crate::consciousness::counterfactual::CausalDAG;

#[derive(Debug, Clone)]
pub struct CodingAttempt {
    pub strategy: String,
    pub compiled: bool,
    pub tests_passed: Option<bool>,
    pub error_pattern: String,
    pub iteration: usize,
}

impl CodingAgent {
    pub(super) fn build_causal_dag(&self) -> Option<CausalDAG> {
        if self.coding_attempts.len() < 2 {
            return None;
        }
        Some(CausalDAG::new(
            vec![
                "task_complexity".into(),
                "strategy_choice".into(),
                "error_pattern".into(),
                "compilation_result".into(),
            ],
            vec![(0, 1), (1, 2), (1, 3), (2, 3)],
        ))
    }
    pub(super) fn record_coding_attempt(&mut self, attempt: CodingAttempt) {
        if self.coding_attempts.len() >= 20 {
            self.coding_attempts.remove(0);
        }
        self.coding_attempts.push(attempt);
    }
    pub(super) fn should_run_counterfactual(&self) -> bool {
        if self.coding_attempts.len() < 2 {
            return false;
        }
        let failed: std::collections::HashSet<&str> = self
            .coding_attempts
            .iter()
            .filter(|a| !a.compiled)
            .map(|a| a.strategy.as_str())
            .collect();
        failed.len() >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dag_structure() {
        let d = CausalDAG::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![(0, 1), (1, 2), (1, 3), (2, 3)],
        );
        assert_eq!(d.nodes.len(), 4);
        assert_eq!(d.parents(1), vec![0]);
        let p = d.parents(3);
        assert!(p.contains(&1) && p.contains(&2));
    }
    #[test]
    fn dag_children() {
        let d = CausalDAG::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![(0, 1), (1, 2)],
        );
        assert_eq!(d.children(0), vec![1]);
        assert_eq!(d.children(1), vec![2]);
    }
    #[test]
    fn empty_dag() {
        let d = CausalDAG::new(vec!["x".into()], vec![]);
        assert!(d.parents(0).is_empty());
    }
    #[test]
    fn attempt_record() {
        let a = CodingAttempt {
            strategy: "fix".into(),
            compiled: false,
            tests_passed: None,
            error_pattern: "E0382".into(),
            iteration: 3,
        };
        assert!(!a.compiled);
    }
}
