// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof State: Persistent Working Memory for Mathematical Proof Search.
//!
//! Provides a proof tree with backtracking, goal management,
//! and integration with the tactic engine.
//!
//! A `ProofTree` is a directed tree where:
//! - Leaves are either open goals or closed (proven) goals
//! - Internal nodes represent tactic applications
//! - The root represents the original conjecture

#![allow(dead_code)]

use std::collections::HashMap;

// ─── Goal ID ─────────────────────────────────────────────────────────────────

/// A unique identifier for a proof node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GoalId(pub usize);

// ─── Node Status ─────────────────────────────────────────────────────────────

/// The current status of a proof node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Open,
    Closed,
    Failed,
}

// ─── ProofNode ───────────────────────────────────────────────────────────────

/// A single node in the proof tree.
#[derive(Debug, Clone)]
pub struct ProofNode {
    pub id: GoalId,
    pub goal_statement: String,
    pub hypotheses: Vec<String>,
    pub status: NodeStatus,
    pub tactic_applied: Option<String>,
    pub children: Vec<GoalId>,
    pub parent: Option<GoalId>,
    pub depth: usize,
}

// ─── ProofTree ───────────────────────────────────────────────────────────────

/// A proof tree with backtracking and goal focus management.
#[derive(Debug, Clone)]
pub struct ProofTree {
    pub nodes: HashMap<GoalId, ProofNode>,
    pub root: GoalId,
    pub next_id: usize,
    pub current_goal: GoalId,
    pub closed_count: usize,
    pub open_count: usize,
}

impl ProofTree {
    /// Create a new proof tree with a single open root goal.
    pub fn new(root_goal: &str) -> Self {
        let root_id = GoalId(0);
        let root_node = ProofNode {
            id: root_id.clone(),
            goal_statement: root_goal.to_string(),
            hypotheses: Vec::new(),
            status: NodeStatus::Open,
            tactic_applied: None,
            children: Vec::new(),
            parent: None,
            depth: 0,
        };
        let mut nodes = HashMap::new();
        nodes.insert(root_id.clone(), root_node);
        Self {
            nodes,
            root: root_id.clone(),
            next_id: 1,
            current_goal: root_id,
            closed_count: 0,
            open_count: 1,
        }
    }

    /// Apply a tactic to a goal: mark it with the tactic, add subgoals as children.
    ///
    /// Returns the new subgoal IDs.
    pub fn apply_tactic(
        &mut self,
        goal_id: &GoalId,
        tactic: &str,
        subgoals: Vec<String>,
    ) -> Vec<GoalId> {
        let mut new_ids = Vec::new();
        let depth = self.nodes.get(goal_id).map(|n| n.depth + 1).unwrap_or(1);

        for subgoal_stmt in subgoals {
            let id = GoalId(self.next_id);
            self.next_id += 1;
            let node = ProofNode {
                id: id.clone(),
                goal_statement: subgoal_stmt,
                hypotheses: Vec::new(),
                status: NodeStatus::Open,
                tactic_applied: None,
                children: Vec::new(),
                parent: Some(goal_id.clone()),
                depth,
            };
            self.nodes.insert(id.clone(), node);
            new_ids.push(id);
            self.open_count += 1;
        }

        if let Some(node) = self.nodes.get_mut(goal_id) {
            node.tactic_applied = Some(tactic.to_string());
            node.children = new_ids.clone();
            if new_ids.is_empty() {
                // Tactic closed the goal directly
                if node.status == NodeStatus::Open {
                    node.status = NodeStatus::Closed;
                    if self.open_count > 0 {
                        self.open_count -= 1;
                    }
                    self.closed_count += 1;
                }
            } else {
                // Node becomes an internal node; status reflects it has children
                // Keep open_count for the goal itself since we'll close via children
                if node.status == NodeStatus::Open {
                    node.status = NodeStatus::Open; // still tracking via children
                }
            }
        }

        // Update current_goal to first new subgoal if available
        if let Some(first) = new_ids.first() {
            self.current_goal = first.clone();
        }

        new_ids
    }

    /// Mark a goal as Closed (proven).
    pub fn close_goal(&mut self, goal_id: &GoalId) {
        if let Some(node) = self.nodes.get_mut(goal_id) {
            if node.status == NodeStatus::Open {
                node.status = NodeStatus::Closed;
                if self.open_count > 0 {
                    self.open_count -= 1;
                }
                self.closed_count += 1;
            }
        }
        // Move focus to next open goal
        let _ = self.focus_next_open();
    }

    /// Remove all descendants of goal, reset it to Open.
    pub fn backtrack(&mut self, to_goal: &GoalId) {
        if let Some(node) = self.nodes.get(to_goal).cloned() {
            // Collect all descendants
            let descendants = self.collect_descendants(&node.children);

            // Count closed descendants being removed
            let closed_removed = descendants
                .iter()
                .filter(|id| {
                    self.nodes
                        .get(id)
                        .map(|n| n.status == NodeStatus::Closed)
                        .unwrap_or(false)
                })
                .count();
            let open_removed = descendants
                .iter()
                .filter(|id| {
                    self.nodes
                        .get(id)
                        .map(|n| n.status == NodeStatus::Open)
                        .unwrap_or(false)
                })
                .count();

            for id in &descendants {
                self.nodes.remove(id);
            }

            self.closed_count = self.closed_count.saturating_sub(closed_removed);
            self.open_count = self.open_count.saturating_sub(open_removed);

            // Reset the target goal
            if let Some(node) = self.nodes.get_mut(to_goal) {
                if node.status == NodeStatus::Closed {
                    self.closed_count = self.closed_count.saturating_sub(1);
                    self.open_count += 1;
                } else if node.status != NodeStatus::Open {
                    self.open_count += 1;
                }
                node.status = NodeStatus::Open;
                node.tactic_applied = None;
                node.children.clear();
            }

            self.current_goal = to_goal.clone();
        }
    }

    fn collect_descendants(&self, children: &[GoalId]) -> Vec<GoalId> {
        let mut result = Vec::new();
        for child in children {
            result.push(child.clone());
            if let Some(node) = self.nodes.get(child) {
                let mut sub = self.collect_descendants(&node.children);
                result.append(&mut sub);
            }
        }
        result
    }

    /// True if all leaf nodes are Closed.
    pub fn is_complete(&self) -> bool {
        self.open_goals().is_empty()
    }

    /// List all open leaf goal IDs.
    pub fn open_goals(&self) -> Vec<GoalId> {
        self.nodes
            .values()
            .filter(|n| n.status == NodeStatus::Open && n.children.is_empty())
            .map(|n| n.id.clone())
            .collect()
    }

    /// Set `current_goal` to the next open goal (depth-first, left-to-right).
    ///
    /// Returns the selected GoalId, or None if no open goals remain.
    pub fn focus_next_open(&mut self) -> Option<GoalId> {
        let next = self.dfs_first_open(&self.root.clone());
        if let Some(ref id) = next {
            self.current_goal = id.clone();
        }
        next
    }

    fn dfs_first_open(&self, id: &GoalId) -> Option<GoalId> {
        if let Some(node) = self.nodes.get(id) {
            if node.status == NodeStatus::Open && node.children.is_empty() {
                return Some(id.clone());
            }
            for child in &node.children {
                if let Some(found) = self.dfs_first_open(child) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Produce ordered list of "goal_id: tactic" steps for the closed proof.
    pub fn proof_script(&self) -> Vec<String> {
        let mut steps = Vec::new();
        self.collect_proof_steps(&self.root.clone(), &mut steps);
        steps
    }

    fn collect_proof_steps(&self, id: &GoalId, steps: &mut Vec<String>) {
        if let Some(node) = self.nodes.get(id) {
            if let Some(tactic) = &node.tactic_applied {
                steps.push(format!("goal {}: {}", id.0, tactic));
            }
            for child in &node.children {
                self.collect_proof_steps(child, steps);
            }
        }
    }

    /// Depth of a node in the tree.
    pub fn depth_of(&self, id: &GoalId) -> usize {
        self.nodes.get(id).map(|n| n.depth).unwrap_or(0)
    }

    /// All ancestor IDs from root to id (exclusive of id).
    pub fn ancestors(&self, id: &GoalId) -> Vec<GoalId> {
        let mut result = Vec::new();
        let mut current = id.clone();
        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent) = &node.parent {
                result.push(parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }
        result.reverse();
        result
    }

    /// Produce a summary of the current proof state.
    pub fn summary(&self) -> ProofSummary {
        let open_goals = self.open_goals().len();
        let max_depth = self.nodes.values().map(|n| n.depth).max().unwrap_or(0);
        let tactics_used: Vec<String> = self
            .nodes
            .values()
            .filter_map(|n| n.tactic_applied.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ProofSummary {
            total_nodes: self.nodes.len(),
            open_goals,
            closed_goals: self.closed_count,
            max_depth,
            tactics_used,
            is_complete: open_goals == 0,
        }
    }
}

// ─── ProofSummary ────────────────────────────────────────────────────────────

/// Summary statistics for a proof tree.
#[derive(Debug, Clone)]
pub struct ProofSummary {
    pub total_nodes: usize,
    pub open_goals: usize,
    pub closed_goals: usize,
    pub max_depth: usize,
    pub tactics_used: Vec<String>,
    pub is_complete: bool,
}

// ─── ProofSession ────────────────────────────────────────────────────────────

/// An interactive proof session with undo support.
pub struct ProofSession {
    pub tree: ProofTree,
    pub history: Vec<ProofTree>,
    pub conjecture: String,
}

impl ProofSession {
    /// Start a new proof session for a conjecture.
    pub fn new(conjecture: &str) -> Self {
        Self {
            tree: ProofTree::new(conjecture),
            history: Vec::new(),
            conjecture: conjecture.to_string(),
        }
    }

    /// Apply a tactic to the current goal, saving state for undo.
    pub fn apply(&mut self, tactic: &str, subgoals: Vec<String>) -> Result<Vec<GoalId>, String> {
        self.history.push(self.tree.clone());
        let goal_id = self.tree.current_goal.clone();
        let new_ids = self.tree.apply_tactic(&goal_id, tactic, subgoals);
        Ok(new_ids)
    }

    /// Undo the last tactic application.
    pub fn undo(&mut self) -> Result<(), String> {
        if let Some(prev) = self.history.pop() {
            self.tree = prev;
            Ok(())
        } else {
            Err("undo: no history to undo".to_string())
        }
    }

    /// Admit the current goal without proof (for incomplete proofs / testing).
    pub fn admit(&mut self) {
        let goal_id = self.tree.current_goal.clone();
        self.tree.close_goal(&goal_id);
    }

    /// Human-readable proof status.
    pub fn status(&self) -> String {
        let summary = self.tree.summary();
        if summary.is_complete {
            format!(
                "Proof complete. {} goals closed, {} tactics used.",
                summary.closed_goals,
                summary.tactics_used.len()
            )
        } else {
            format!(
                "Proof in progress: {} open goals, {} closed goals.",
                summary.open_goals, summary.closed_goals
            )
        }
    }

    /// Export proof tree as a LaTeX proof skeleton.
    pub fn export_latex(&self) -> String {
        let mut lines = vec![
            r"\begin{proof}".to_string(),
            format!("  % Conjecture: {}", self.conjecture),
        ];
        let script = self.tree.proof_script();
        for step in &script {
            lines.push(format!("  % {}", step));
        }
        let summary = self.tree.summary();
        if summary.is_complete {
            lines.push("  % Proof complete.".to_string());
        } else {
            lines.push(format!(
                "  % Proof incomplete: {} open goals.",
                summary.open_goals
            ));
        }
        lines.push(r"\end{proof}".to_string());
        lines.join("\n")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree_has_one_open_goal() {
        let tree = ProofTree::new("P → P");
        assert_eq!(tree.open_count, 1);
        assert_eq!(tree.closed_count, 0);
        assert_eq!(tree.open_goals().len(), 1);
        assert!(!tree.is_complete());
    }

    #[test]
    fn test_apply_tactic_adds_children() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        assert_eq!(children.len(), 2);
        assert_eq!(tree.open_goals().len(), 2);
        // Root has children
        assert_eq!(tree.nodes[&root].children.len(), 2);
    }

    #[test]
    fn test_apply_tactic_with_no_subgoals_closes_goal() {
        let mut tree = ProofTree::new("P");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "assumption", vec![]);
        assert!(children.is_empty());
        assert!(tree.is_complete(), "No subgoals means goal closed");
        assert_eq!(tree.closed_count, 1);
    }

    #[test]
    fn test_close_goal_marks_closed() {
        let mut tree = ProofTree::new("P");
        let root = tree.root.clone();
        // Apply split to get two subgoals
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        tree.close_goal(&children[0]);
        tree.close_goal(&children[1]);
        assert_eq!(tree.closed_count, 2);
        assert_eq!(tree.open_goals().len(), 0);
    }

    #[test]
    fn test_is_complete_only_when_all_closed() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        assert!(!tree.is_complete());
        tree.close_goal(&children[0]);
        assert!(!tree.is_complete());
        tree.close_goal(&children[1]);
        assert!(tree.is_complete());
    }

    #[test]
    fn test_backtrack_restores_to_open() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        // Close first child
        tree.close_goal(&children[0]);
        assert_eq!(tree.closed_count, 1);
        // Backtrack to root
        tree.backtrack(&root);
        assert_eq!(tree.nodes[&root].status, NodeStatus::Open);
        assert!(tree.nodes[&root].children.is_empty());
        // Children removed
        assert!(!tree.nodes.contains_key(&children[0]));
        assert!(!tree.nodes.contains_key(&children[1]));
    }

    #[test]
    fn test_proof_script_in_order() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        tree.apply_tactic(&children[0], "assumption", vec![]);
        tree.apply_tactic(&children[1], "assumption", vec![]);

        let script = tree.proof_script();
        // Should contain split and two assumption steps
        assert!(script.iter().any(|s| s.contains("split")));
        assert_eq!(
            script.iter().filter(|s| s.contains("assumption")).count(),
            2
        );
    }

    #[test]
    fn test_proof_session_undo_works() {
        let mut session = ProofSession::new("A ∧ B");
        session
            .apply("split", vec!["A".to_string(), "B".to_string()])
            .unwrap();
        assert_eq!(session.tree.open_goals().len(), 2);

        session.undo().unwrap();
        // Back to one open goal (root)
        assert_eq!(session.tree.open_goals().len(), 1);
    }

    #[test]
    fn test_proof_session_status_messages() {
        let mut session = ProofSession::new("P");
        assert!(session.status().contains("open"));
        session.admit();
        assert!(session.status().contains("complete"));
    }

    #[test]
    fn test_export_latex_structure() {
        let session = ProofSession::new("P → P");
        let latex = session.export_latex();
        assert!(latex.contains(r"\begin{proof}"));
        assert!(latex.contains(r"\end{proof}"));
        assert!(latex.contains("P → P"));
    }

    #[test]
    fn test_ancestors() {
        let mut tree = ProofTree::new("root");
        let root = tree.root.clone();
        let children = tree.apply_tactic(&root, "split", vec!["A".to_string(), "B".to_string()]);
        let grandchildren = tree.apply_tactic(
            &children[0],
            "split",
            vec!["A1".to_string(), "A2".to_string()],
        );
        let ancestors = tree.ancestors(&grandchildren[0]);
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0], root);
        assert_eq!(ancestors[1], children[0]);
    }
}
