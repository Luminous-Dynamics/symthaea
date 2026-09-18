// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof State: Persistent Working Memory for Mathematical Proof Search.
//!
//! This module records local proof-search state only. It never establishes
//! mathematical truth or formal authority by itself. A locally complete tree is
//! a proof *candidate* that still requires an authenticated external verifier
//! receipt (for example, the Lean execution path in `symthaea-lean-bridge`).
//!
//! Authority boundary:
//!
//! ```text
//! local tactic trace -> proof candidate -> authenticated verifier -> authority
//! ```

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, HashSet};

const CANDIDATE_ARTIFACT_DOMAIN: &str = "symthaea.proof-candidate.v1";

/// A unique identifier for a proof node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GoalId(pub usize);

/// Local proof-search state for one node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    /// Unresolved leaf goal.
    Open,
    /// A local tactic was applied and produced child goals.
    Expanded,
    /// A local tactic was applied and produced no child goals.
    ///
    /// This is a proof-candidate closure only, not formal authority.
    CandidateClosed,
    /// Explicitly assumed without a proof-producing tactic.
    Admitted,
    /// Search terminated unsuccessfully for this leaf.
    Failed,
}

/// A single node in the local proof-search tree.
#[derive(Debug, Clone)]
pub struct ProofNode {
    id: GoalId,
    goal_statement: String,
    hypotheses: Vec<String>,
    status: NodeStatus,
    tactic_applied: Option<String>,
    children: Vec<GoalId>,
    parent: Option<GoalId>,
    depth: usize,
}

impl ProofNode {
    pub fn id(&self) -> &GoalId {
        &self.id
    }

    pub fn goal_statement(&self) -> &str {
        &self.goal_statement
    }

    pub fn hypotheses(&self) -> &[String] {
        &self.hypotheses
    }

    pub fn status(&self) -> &NodeStatus {
        &self.status
    }

    pub fn tactic_applied(&self) -> Option<&str> {
        self.tactic_applied.as_deref()
    }

    pub fn children(&self) -> &[GoalId] {
        &self.children
    }

    pub fn parent(&self) -> Option<&GoalId> {
        self.parent.as_ref()
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStateError {
    UnknownGoal(GoalId),
    GoalNotOpen {
        goal_id: GoalId,
        status: NodeStatus,
    },
    GoalAlreadyExpanded(GoalId),
    EmptyTactic,
    EmptySubgoal {
        index: usize,
    },
    NoHistory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStateIssue {
    MissingRoot,
    MissingCurrentGoal,
    MissingChild {
        parent: GoalId,
        child: GoalId,
    },
    ParentMismatch {
        child: GoalId,
        expected_parent: GoalId,
        found_parent: Option<GoalId>,
    },
    DuplicateChild {
        parent: GoalId,
        child: GoalId,
    },
    Cycle {
        goal_id: GoalId,
    },
    UnreachableNode {
        goal_id: GoalId,
    },
    OpenNodeHasTrace {
        goal_id: GoalId,
    },
    ExpandedNodeMissingChildren {
        goal_id: GoalId,
    },
    ExpandedNodeMissingTactic {
        goal_id: GoalId,
    },
    CandidateClosureMissingTactic {
        goal_id: GoalId,
    },
    TerminalNodeHasChildren {
        goal_id: GoalId,
        status: NodeStatus,
    },
}

/// Local proof-search tree with backtracking and goal focus management.
///
/// Internal storage is private so callers cannot manufacture a completed
/// candidate by directly mutating node status or counters.
#[derive(Debug, Clone)]
pub struct ProofTree {
    nodes: HashMap<GoalId, ProofNode>,
    root: GoalId,
    next_id: usize,
    current_goal: GoalId,
    candidate_closed_leaf_count: usize,
    open_leaf_count: usize,
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
            candidate_closed_leaf_count: 0,
            open_leaf_count: 1,
        }
    }

    pub fn root(&self) -> &GoalId {
        &self.root
    }

    pub fn current_goal(&self) -> &GoalId {
        &self.current_goal
    }

    pub fn node(&self, goal_id: &GoalId) -> Option<&ProofNode> {
        self.nodes.get(goal_id)
    }

    pub fn open_leaf_count(&self) -> usize {
        self.open_leaf_count
    }

    pub fn candidate_closed_leaf_count(&self) -> usize {
        self.candidate_closed_leaf_count
    }

    fn ensure_open_leaf(&self, goal_id: &GoalId) -> Result<(), ProofStateError> {
        let node = self
            .nodes
            .get(goal_id)
            .ok_or_else(|| ProofStateError::UnknownGoal(goal_id.clone()))?;
        if node.status != NodeStatus::Open {
            return Err(ProofStateError::GoalNotOpen {
                goal_id: goal_id.clone(),
                status: node.status.clone(),
            });
        }
        if !node.children.is_empty() {
            return Err(ProofStateError::GoalAlreadyExpanded(goal_id.clone()));
        }
        Ok(())
    }

    /// Apply a local tactic to an unresolved leaf.
    ///
    /// A tactic with no returned subgoals records `CandidateClosed`. This is
    /// still only local search evidence; external verification remains required.
    pub fn apply_tactic(
        &mut self,
        goal_id: &GoalId,
        tactic: &str,
        subgoals: Vec<String>,
    ) -> Result<Vec<GoalId>, ProofStateError> {
        self.ensure_open_leaf(goal_id)?;
        if tactic.trim().is_empty() {
            return Err(ProofStateError::EmptyTactic);
        }
        for (index, subgoal) in subgoals.iter().enumerate() {
            if subgoal.trim().is_empty() {
                return Err(ProofStateError::EmptySubgoal { index });
            }
        }

        let depth = self
            .nodes
            .get(goal_id)
            .expect("goal validated above")
            .depth
            + 1;

        if subgoals.is_empty() {
            let node = self.nodes.get_mut(goal_id).expect("goal validated above");
            node.tactic_applied = Some(tactic.trim().to_string());
            node.status = NodeStatus::CandidateClosed;
            self.open_leaf_count = self.open_leaf_count.saturating_sub(1);
            self.candidate_closed_leaf_count += 1;
            let _ = self.focus_next_open();
            return Ok(Vec::new());
        }

        let mut new_ids = Vec::with_capacity(subgoals.len());
        for subgoal_statement in subgoals {
            let id = GoalId(self.next_id);
            self.next_id += 1;
            self.nodes.insert(
                id.clone(),
                ProofNode {
                    id: id.clone(),
                    goal_statement: subgoal_statement,
                    hypotheses: Vec::new(),
                    status: NodeStatus::Open,
                    tactic_applied: None,
                    children: Vec::new(),
                    parent: Some(goal_id.clone()),
                    depth,
                },
            );
            new_ids.push(id);
        }

        let node = self.nodes.get_mut(goal_id).expect("goal validated above");
        node.tactic_applied = Some(tactic.trim().to_string());
        node.children = new_ids.clone();
        node.status = NodeStatus::Expanded;

        self.open_leaf_count = self
            .open_leaf_count
            .saturating_sub(1)
            .saturating_add(new_ids.len());
        self.current_goal = new_ids[0].clone();
        Ok(new_ids)
    }

    /// Resolve an open leaf by explicit admission.
    ///
    /// Admission is terminal for exploration but can never satisfy proof-candidate
    /// completion.
    pub fn admit_goal(&mut self, goal_id: &GoalId) -> Result<(), ProofStateError> {
        self.ensure_open_leaf(goal_id)?;
        let node = self.nodes.get_mut(goal_id).expect("goal validated above");
        node.status = NodeStatus::Admitted;
        self.open_leaf_count = self.open_leaf_count.saturating_sub(1);
        let _ = self.focus_next_open();
        Ok(())
    }

    /// Resolve an open leaf as a failed search branch.
    pub fn fail_goal(&mut self, goal_id: &GoalId) -> Result<(), ProofStateError> {
        self.ensure_open_leaf(goal_id)?;
        let node = self.nodes.get_mut(goal_id).expect("goal validated above");
        node.status = NodeStatus::Failed;
        self.open_leaf_count = self.open_leaf_count.saturating_sub(1);
        let _ = self.focus_next_open();
        Ok(())
    }

    /// Remove all descendants of a goal and reset that goal to an open leaf.
    pub fn backtrack(&mut self, to_goal: &GoalId) -> Result<(), ProofStateError> {
        let target = self
            .nodes
            .get(to_goal)
            .cloned()
            .ok_or_else(|| ProofStateError::UnknownGoal(to_goal.clone()))?;
        let descendants = self.collect_descendants(&target.children);

        let removed_open = descendants
            .iter()
            .filter(|id| {
                self.nodes
                    .get(id)
                    .is_some_and(|node| node.status == NodeStatus::Open && node.children.is_empty())
            })
            .count();
        let removed_candidate_closed = descendants
            .iter()
            .filter(|id| {
                self.nodes
                    .get(id)
                    .is_some_and(|node| node.status == NodeStatus::CandidateClosed)
            })
            .count();

        for id in &descendants {
            self.nodes.remove(id);
        }
        self.open_leaf_count = self.open_leaf_count.saturating_sub(removed_open);
        self.candidate_closed_leaf_count = self
            .candidate_closed_leaf_count
            .saturating_sub(removed_candidate_closed);

        match target.status {
            NodeStatus::Open if target.children.is_empty() => {}
            NodeStatus::CandidateClosed => {
                self.candidate_closed_leaf_count =
                    self.candidate_closed_leaf_count.saturating_sub(1);
                self.open_leaf_count += 1;
            }
            NodeStatus::Expanded | NodeStatus::Admitted | NodeStatus::Failed | NodeStatus::Open => {
                self.open_leaf_count += 1;
            }
        }

        let node = self.nodes.get_mut(to_goal).expect("target existed above");
        node.status = NodeStatus::Open;
        node.tactic_applied = None;
        node.children.clear();
        self.current_goal = to_goal.clone();
        Ok(())
    }

    fn collect_descendants(&self, children: &[GoalId]) -> Vec<GoalId> {
        let mut result = Vec::new();
        for child in children {
            result.push(child.clone());
            if let Some(node) = self.nodes.get(child) {
                result.extend(self.collect_descendants(&node.children));
            }
        }
        result
    }

    /// Validate structural invariants before a candidate is exported.
    pub fn validate(&self) -> Vec<ProofStateIssue> {
        let mut issues = Vec::new();
        if !self.nodes.contains_key(&self.root) {
            issues.push(ProofStateIssue::MissingRoot);
            return issues;
        }
        if !self.nodes.contains_key(&self.current_goal) {
            issues.push(ProofStateIssue::MissingCurrentGoal);
        }

        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();
        self.validate_node(
            &self.root,
            None,
            &mut visiting,
            &mut visited,
            &mut issues,
        );

        let mut all_ids: Vec<_> = self.nodes.keys().cloned().collect();
        all_ids.sort();
        for id in all_ids {
            if !visited.contains(&id) {
                issues.push(ProofStateIssue::UnreachableNode { goal_id: id });
            }
        }
        issues
    }

    fn validate_node(
        &self,
        goal_id: &GoalId,
        expected_parent: Option<&GoalId>,
        visiting: &mut HashSet<GoalId>,
        visited: &mut HashSet<GoalId>,
        issues: &mut Vec<ProofStateIssue>,
    ) {
        if visited.contains(goal_id) {
            return;
        }
        if !visiting.insert(goal_id.clone()) {
            issues.push(ProofStateIssue::Cycle {
                goal_id: goal_id.clone(),
            });
            return;
        }

        let Some(node) = self.nodes.get(goal_id) else {
            visiting.remove(goal_id);
            return;
        };

        if node.parent.as_ref() != expected_parent {
            if let Some(expected_parent) = expected_parent {
                issues.push(ProofStateIssue::ParentMismatch {
                    child: goal_id.clone(),
                    expected_parent: expected_parent.clone(),
                    found_parent: node.parent.clone(),
                });
            }
        }

        match node.status {
            NodeStatus::Open => {
                if node.tactic_applied.is_some() || !node.children.is_empty() {
                    issues.push(ProofStateIssue::OpenNodeHasTrace {
                        goal_id: goal_id.clone(),
                    });
                }
            }
            NodeStatus::Expanded => {
                if node.children.is_empty() {
                    issues.push(ProofStateIssue::ExpandedNodeMissingChildren {
                        goal_id: goal_id.clone(),
                    });
                }
                if node.tactic_applied.as_deref().is_none_or(str::is_empty) {
                    issues.push(ProofStateIssue::ExpandedNodeMissingTactic {
                        goal_id: goal_id.clone(),
                    });
                }
            }
            NodeStatus::CandidateClosed => {
                if node.tactic_applied.as_deref().is_none_or(str::is_empty) {
                    issues.push(ProofStateIssue::CandidateClosureMissingTactic {
                        goal_id: goal_id.clone(),
                    });
                }
                if !node.children.is_empty() {
                    issues.push(ProofStateIssue::TerminalNodeHasChildren {
                        goal_id: goal_id.clone(),
                        status: node.status.clone(),
                    });
                }
            }
            NodeStatus::Admitted | NodeStatus::Failed => {
                if !node.children.is_empty() {
                    issues.push(ProofStateIssue::TerminalNodeHasChildren {
                        goal_id: goal_id.clone(),
                        status: node.status.clone(),
                    });
                }
            }
        }

        let mut local_children = HashSet::new();
        for child in &node.children {
            if !local_children.insert(child.clone()) {
                issues.push(ProofStateIssue::DuplicateChild {
                    parent: goal_id.clone(),
                    child: child.clone(),
                });
                continue;
            }
            if !self.nodes.contains_key(child) {
                issues.push(ProofStateIssue::MissingChild {
                    parent: goal_id.clone(),
                    child: child.clone(),
                });
                continue;
            }
            self.validate_node(child, Some(goal_id), visiting, visited, issues);
        }

        visiting.remove(goal_id);
        visited.insert(goal_id.clone());
    }

    /// True only when the tree is structurally valid and every leaf is locally
    /// closed by a recorded tactic.
    ///
    /// This is a proof-candidate predicate, not a theorem-verification predicate.
    pub fn is_candidate_complete(&self) -> bool {
        if !self.validate().is_empty() {
            return false;
        }
        let mut saw_leaf = false;
        for node in self.nodes.values().filter(|node| node.children.is_empty()) {
            saw_leaf = true;
            if node.status != NodeStatus::CandidateClosed {
                return false;
            }
        }
        saw_leaf
    }

    /// True when no unresolved leaf remains. Admitted/failed leaves count as
    /// resolved for search control but never make a candidate complete.
    pub fn is_resolved(&self) -> bool {
        if !self.validate().is_empty() {
            return false;
        }
        !self
            .nodes
            .values()
            .any(|node| node.children.is_empty() && node.status == NodeStatus::Open)
    }

    /// List open leaf goals in deterministic goal-id order.
    pub fn open_goals(&self) -> Vec<GoalId> {
        let mut goals: Vec<_> = self
            .nodes
            .values()
            .filter(|node| node.status == NodeStatus::Open && node.children.is_empty())
            .map(|node| node.id.clone())
            .collect();
        goals.sort();
        goals
    }

    /// Set `current_goal` to the next open goal in depth-first child order.
    pub fn focus_next_open(&mut self) -> Option<GoalId> {
        let next = self.dfs_first_open(&self.root.clone());
        if let Some(id) = &next {
            self.current_goal = id.clone();
        }
        next
    }

    fn dfs_first_open(&self, goal_id: &GoalId) -> Option<GoalId> {
        let node = self.nodes.get(goal_id)?;
        if node.status == NodeStatus::Open && node.children.is_empty() {
            return Some(goal_id.clone());
        }
        for child in &node.children {
            if let Some(found) = self.dfs_first_open(child) {
                return Some(found);
            }
        }
        None
    }

    /// Ordered local tactic trace. This is not an externally verified proof script.
    pub fn proof_candidate_trace(&self) -> Vec<String> {
        let mut steps = Vec::new();
        self.collect_candidate_steps(&self.root, &mut steps);
        steps
    }

    fn collect_candidate_steps(&self, goal_id: &GoalId, steps: &mut Vec<String>) {
        if let Some(node) = self.nodes.get(goal_id) {
            if let Some(tactic) = &node.tactic_applied {
                steps.push(format!("goal {}: {}", goal_id.0, tactic));
            }
            for child in &node.children {
                self.collect_candidate_steps(child, steps);
            }
        }
    }

    /// Deterministic local proof-candidate artifact.
    ///
    /// Returns `Ok(None)` until every leaf is locally candidate-closed. The bytes
    /// intentionally contain no formal-authority marker; downstream verifier code
    /// may hash/bind them as provenance for the candidate that produced a Lean proof.
    pub fn candidate_artifact_bytes(
        &self,
    ) -> Result<Option<Vec<u8>>, Vec<ProofStateIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        if !self.is_candidate_complete() {
            return Ok(None);
        }

        let mut output = Vec::new();
        put_text(&mut output, CANDIDATE_ARTIFACT_DOMAIN);
        self.encode_candidate_node(&self.root, &mut output);
        Ok(Some(output))
    }

    fn encode_candidate_node(&self, goal_id: &GoalId, output: &mut Vec<u8>) {
        let node = self
            .nodes
            .get(goal_id)
            .expect("candidate artifact is generated only after validation");
        put_u64(output, goal_id.0 as u64);
        put_text(output, &node.goal_statement);
        put_u64(output, node.hypotheses.len() as u64);
        for hypothesis in &node.hypotheses {
            put_text(output, hypothesis);
        }
        put_text(output, node_status_tag(&node.status));
        match &node.tactic_applied {
            Some(tactic) => {
                put_text(output, "tactic-present");
                put_text(output, tactic);
            }
            None => put_text(output, "tactic-absent"),
        }
        match &node.parent {
            Some(parent) => {
                put_text(output, "parent-present");
                put_u64(output, parent.0 as u64);
            }
            None => put_text(output, "parent-absent"),
        }
        put_u64(output, node.depth as u64);
        put_u64(output, node.children.len() as u64);
        for child in &node.children {
            put_u64(output, child.0 as u64);
        }
        for child in &node.children {
            self.encode_candidate_node(child, output);
        }
    }

    pub fn depth_of(&self, goal_id: &GoalId) -> Option<usize> {
        self.nodes.get(goal_id).map(|node| node.depth)
    }

    pub fn ancestors(&self, goal_id: &GoalId) -> Result<Vec<GoalId>, ProofStateError> {
        if !self.nodes.contains_key(goal_id) {
            return Err(ProofStateError::UnknownGoal(goal_id.clone()));
        }
        let mut result = Vec::new();
        let mut current = goal_id.clone();
        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent) = &node.parent {
                result.push(parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }
        result.reverse();
        Ok(result)
    }

    pub fn summary(&self) -> ProofSummary {
        let open_goals = self.open_goals().len();
        let max_depth = self.nodes.values().map(|node| node.depth).max().unwrap_or(0);
        let tactics_used: Vec<String> = self
            .nodes
            .values()
            .filter_map(|node| node.tactic_applied.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let admitted_goals = self
            .nodes
            .values()
            .filter(|node| node.children.is_empty() && node.status == NodeStatus::Admitted)
            .count();
        let failed_goals = self
            .nodes
            .values()
            .filter(|node| node.children.is_empty() && node.status == NodeStatus::Failed)
            .count();
        ProofSummary {
            total_nodes: self.nodes.len(),
            open_goals,
            candidate_closed_goals: self.candidate_closed_leaf_count,
            admitted_goals,
            failed_goals,
            max_depth,
            tactics_used,
            is_candidate_complete: self.is_candidate_complete(),
            is_resolved: self.is_resolved(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProofSummary {
    pub total_nodes: usize,
    pub open_goals: usize,
    pub candidate_closed_goals: usize,
    pub admitted_goals: usize,
    pub failed_goals: usize,
    pub max_depth: usize,
    pub tactics_used: Vec<String>,
    /// Local proof-search completion only; external verification is still required.
    pub is_candidate_complete: bool,
    pub is_resolved: bool,
}

/// Interactive local proof-search session with undo support.
pub struct ProofSession {
    tree: ProofTree,
    history: Vec<ProofTree>,
    conjecture: String,
}

impl ProofSession {
    pub fn new(conjecture: &str) -> Self {
        Self {
            tree: ProofTree::new(conjecture),
            history: Vec::new(),
            conjecture: conjecture.to_string(),
        }
    }

    pub fn tree(&self) -> &ProofTree {
        &self.tree
    }

    pub fn conjecture(&self) -> &str {
        &self.conjecture
    }

    pub fn apply(
        &mut self,
        tactic: &str,
        subgoals: Vec<String>,
    ) -> Result<Vec<GoalId>, ProofStateError> {
        let previous = self.tree.clone();
        let goal_id = self.tree.current_goal.clone();
        let result = self.tree.apply_tactic(&goal_id, tactic, subgoals)?;
        self.history.push(previous);
        Ok(result)
    }

    pub fn admit(&mut self) -> Result<(), ProofStateError> {
        let previous = self.tree.clone();
        let goal_id = self.tree.current_goal.clone();
        self.tree.admit_goal(&goal_id)?;
        self.history.push(previous);
        Ok(())
    }

    pub fn fail(&mut self) -> Result<(), ProofStateError> {
        let previous = self.tree.clone();
        let goal_id = self.tree.current_goal.clone();
        self.tree.fail_goal(&goal_id)?;
        self.history.push(previous);
        Ok(())
    }

    pub fn undo(&mut self) -> Result<(), ProofStateError> {
        let previous = self.history.pop().ok_or(ProofStateError::NoHistory)?;
        self.tree = previous;
        Ok(())
    }

    pub fn status(&self) -> String {
        let summary = self.tree.summary();
        if summary.is_candidate_complete {
            format!(
                "Proof candidate locally complete: {} candidate-closed goals, {} tactics used. External formal verification required.",
                summary.candidate_closed_goals,
                summary.tactics_used.len()
            )
        } else if summary.is_resolved {
            format!(
                "Proof candidate incomplete: {} admitted goals, {} failed goals, {} candidate-closed goals.",
                summary.admitted_goals,
                summary.failed_goals,
                summary.candidate_closed_goals
            )
        } else {
            format!(
                "Proof search in progress: {} open goals, {} candidate-closed goals, {} admitted goals.",
                summary.open_goals,
                summary.candidate_closed_goals,
                summary.admitted_goals
            )
        }
    }

    /// Export a human-readable LaTeX skeleton of the local candidate trace.
    ///
    /// The export deliberately states that external verification is required.
    pub fn export_latex(&self) -> String {
        let mut lines = vec![
            r"\begin{proof}".to_string(),
            format!("  % Conjecture: {}", self.conjecture),
            "  % Local Symthaea proof candidate; not formal authority.".to_string(),
        ];
        for step in self.tree.proof_candidate_trace() {
            lines.push(format!("  % {step}"));
        }
        let summary = self.tree.summary();
        if summary.is_candidate_complete {
            lines.push(
                "  % Candidate locally complete; authenticated external verification required."
                    .to_string(),
            );
        } else if summary.is_resolved {
            lines.push(format!(
                "  % Candidate incomplete: {} admitted goals, {} failed goals.",
                summary.admitted_goals, summary.failed_goals
            ));
        } else {
            lines.push(format!(
                "  % Candidate incomplete: {} open goals, {} admitted goals.",
                summary.open_goals, summary.admitted_goals
            ));
        }
        lines.push(r"\end{proof}".to_string());
        lines.join("\n")
    }
}

fn node_status_tag(status: &NodeStatus) -> &'static str {
    match status {
        NodeStatus::Open => "open",
        NodeStatus::Expanded => "expanded",
        NodeStatus::CandidateClosed => "candidate-closed",
        NodeStatus::Admitted => "admitted",
        NodeStatus::Failed => "failed",
    }
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    put_u64(output, value.len() as u64);
    output.extend_from_slice(value.as_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tree_has_one_open_leaf() {
        let tree = ProofTree::new("P → P");
        assert_eq!(tree.open_leaf_count(), 1);
        assert_eq!(tree.candidate_closed_leaf_count(), 0);
        assert_eq!(tree.open_goals(), vec![GoalId(0)]);
        assert!(!tree.is_candidate_complete());
        assert!(tree.validate().is_empty());
    }

    #[test]
    fn apply_tactic_expands_goal_without_counting_parent_as_open_leaf() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root().clone();
        let children = tree
            .apply_tactic(&root, "split", vec!["A".into(), "B".into()])
            .unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(tree.open_leaf_count(), 2);
        assert_eq!(tree.node(&root).unwrap().status(), &NodeStatus::Expanded);
        assert!(tree.validate().is_empty());
    }

    #[test]
    fn zero_subgoal_tactic_creates_candidate_closure_only() {
        let mut tree = ProofTree::new("P");
        let root = tree.root().clone();
        tree.apply_tactic(&root, "assumption", vec![]).unwrap();
        assert_eq!(
            tree.node(&root).unwrap().status(),
            &NodeStatus::CandidateClosed
        );
        assert_eq!(tree.candidate_closed_leaf_count(), 1);
        assert!(tree.is_candidate_complete());
    }

    #[test]
    fn arbitrary_close_primitive_is_replaced_by_recorded_tactic_path() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root().clone();
        let children = tree
            .apply_tactic(&root, "split", vec!["A".into(), "B".into()])
            .unwrap();
        tree.apply_tactic(&children[0], "assumption", vec![])
            .unwrap();
        assert!(!tree.is_candidate_complete());
        tree.apply_tactic(&children[1], "assumption", vec![])
            .unwrap();
        assert!(tree.is_candidate_complete());
        assert_eq!(tree.candidate_closed_leaf_count(), 2);
    }

    #[test]
    fn invalid_goal_does_not_mutate_tree() {
        let mut tree = ProofTree::new("P");
        let before = tree.summary();
        let error = tree
            .apply_tactic(&GoalId(999), "magic", vec![])
            .unwrap_err();
        assert_eq!(error, ProofStateError::UnknownGoal(GoalId(999)));
        let after = tree.summary();
        assert_eq!(before.total_nodes, after.total_nodes);
        assert_eq!(before.open_goals, after.open_goals);
    }

    #[test]
    fn empty_tactic_and_subgoal_fail_before_mutation() {
        let mut tree = ProofTree::new("P");
        let root = tree.root().clone();
        assert_eq!(
            tree.apply_tactic(&root, " ", vec![]).unwrap_err(),
            ProofStateError::EmptyTactic
        );
        assert_eq!(
            tree.apply_tactic(&root, "split", vec![" ".into()])
                .unwrap_err(),
            ProofStateError::EmptySubgoal { index: 0 }
        );
        assert_eq!(tree.open_leaf_count(), 1);
        assert!(tree.validate().is_empty());
    }

    #[test]
    fn admitted_leaf_is_resolved_but_never_candidate_complete() {
        let mut tree = ProofTree::new("P");
        let root = tree.root().clone();
        tree.admit_goal(&root).unwrap();
        let summary = tree.summary();
        assert_eq!(summary.open_goals, 0);
        assert_eq!(summary.admitted_goals, 1);
        assert!(summary.is_resolved);
        assert!(!summary.is_candidate_complete);
        assert_eq!(tree.node(&root).unwrap().status(), &NodeStatus::Admitted);
        assert_eq!(tree.candidate_artifact_bytes().unwrap(), None);
    }

    #[test]
    fn backtrack_restores_expanded_goal_to_open_leaf() {
        let mut tree = ProofTree::new("A ∧ B");
        let root = tree.root().clone();
        let children = tree
            .apply_tactic(&root, "split", vec!["A".into(), "B".into()])
            .unwrap();
        tree.apply_tactic(&children[0], "assumption", vec![])
            .unwrap();
        tree.backtrack(&root).unwrap();
        assert_eq!(tree.node(&root).unwrap().status(), &NodeStatus::Open);
        assert!(tree.node(&root).unwrap().children().is_empty());
        assert_eq!(tree.open_leaf_count(), 1);
        assert!(tree.node(&children[0]).is_none());
        assert!(tree.node(&children[1]).is_none());
        assert!(tree.validate().is_empty());
    }

    #[test]
    fn candidate_artifact_is_deterministic_and_trace_sensitive() {
        let mut left = ProofTree::new("P");
        let root = left.root().clone();
        left.apply_tactic(&root, "assumption", vec![]).unwrap();
        let left_bytes = left.candidate_artifact_bytes().unwrap().unwrap();
        let cloned_bytes = left.clone().candidate_artifact_bytes().unwrap().unwrap();
        assert_eq!(left_bytes, cloned_bytes);

        let mut right = ProofTree::new("P");
        let root = right.root().clone();
        right.apply_tactic(&root, "exact hp", vec![]).unwrap();
        let right_bytes = right.candidate_artifact_bytes().unwrap().unwrap();
        assert_ne!(left_bytes, right_bytes);
    }

    #[test]
    fn proof_session_admission_is_undoable_and_non_authorizing() {
        let mut session = ProofSession::new("P");
        assert!(session.status().contains("in progress"));
        session.admit().unwrap();
        assert!(session.status().contains("incomplete"));
        assert!(session.status().contains("admitted"));
        assert!(!session.tree().is_candidate_complete());
        session.undo().unwrap();
        assert_eq!(session.tree().open_leaf_count(), 1);
    }

    #[test]
    fn latex_export_never_calls_local_completion_formal_authority() {
        let mut session = ProofSession::new("P → P");
        session.apply("assumption", vec![]).unwrap();
        let latex = session.export_latex();
        assert!(latex.contains(r"\begin{proof}"));
        assert!(latex.contains("proof candidate"));
        assert!(latex.contains("external verification required"));
        assert!(!latex.contains("Proof complete."));
    }

    #[test]
    fn ancestors_are_root_to_parent() {
        let mut tree = ProofTree::new("root");
        let root = tree.root().clone();
        let children = tree
            .apply_tactic(&root, "split", vec!["A".into(), "B".into()])
            .unwrap();
        let grandchildren = tree
            .apply_tactic(
                &children[0],
                "split",
                vec!["A1".into(), "A2".into()],
            )
            .unwrap();
        let ancestors = tree.ancestors(&grandchildren[0]).unwrap();
        assert_eq!(ancestors, vec![root, children[0].clone()]);
    }
}
