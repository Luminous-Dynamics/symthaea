// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Program Dependence Graph (PDG)
//!
//! Layer 1 of the Geodesic Code Synthesis architecture. The PDG merges
//! control flow and data dependency information from parsed code into a
//! single graph that all higher layers build upon.
//!
//! ## Design
//!
//! A PDG node represents a statement, branch, loop header, call, or
//! entry/exit point. Edges carry a [`EdgeKind`] discriminant that
//! separates control-flow edges (Sequential, BranchTrue/False, LoopBack,
//! etc.) from data-dependency edges (DefUse, UseDef, OutputDep).
//!
//! The graph converts losslessly to:
//! - [`symthaea_core::hdc::graph_theory::Graph`] for BFS/DFS/topological sort
//! - [`symthaea_hodge::SimplicialComplex`] for Hodge-theoretic analysis

use std::collections::{HashMap, HashSet};
use symthaea_core::hdc::graph_theory::Graph;
use symthaea_hodge::SimplicialComplex;

// ============================================================================
// NODE TYPES
// ============================================================================

/// A node in the Program Dependence Graph.
#[derive(Debug, Clone)]
pub struct PdgNode {
    pub id: usize,
    pub kind: NodeKind,
    /// Source text or description.
    pub label: String,
    /// Source line number (1-based), if known.
    pub line: Option<usize>,
}

/// Classification of a PDG node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// Function entry point.
    Entry,
    /// Function exit point.
    Exit,
    /// `let` binding, assignment, or expression statement.
    Statement,
    /// `if` / `match` condition.
    Branch,
    /// `for` / `while` / `loop` head.
    LoopHead,
    /// `return` statement.
    Return,
    /// Function or method call.
    Call,
}

// ============================================================================
// EDGE TYPES
// ============================================================================

/// An edge in the PDG.
#[derive(Debug, Clone)]
pub struct PdgEdge {
    pub from: usize,
    pub to: usize,
    pub kind: EdgeKind,
}

/// Classification of a PDG edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    // ── Control flow ──────────────────────────────────────────────────
    /// Straight-line flow between consecutive statements.
    Sequential,
    /// True branch of a conditional.
    BranchTrue,
    /// False branch of a conditional.
    BranchFalse,
    /// Back-edge to loop head.
    LoopBack,
    /// Exit from loop body.
    LoopExit,
    /// Return to caller.
    ReturnEdge,

    // ── Data dependency ───────────────────────────────────────────────
    /// Definition flows to a use (write-then-read).
    DefUse,
    /// Use depends on definition (read-after-write, reverse query direction).
    UseDef,
    /// Write-after-write on the same variable.
    OutputDep,
}

impl EdgeKind {
    /// Returns `true` for control-flow edge kinds.
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            EdgeKind::Sequential
                | EdgeKind::BranchTrue
                | EdgeKind::BranchFalse
                | EdgeKind::LoopBack
                | EdgeKind::LoopExit
                | EdgeKind::ReturnEdge
        )
    }

    /// Returns `true` for data-dependency edge kinds.
    pub fn is_data_dependency(&self) -> bool {
        matches!(
            self,
            EdgeKind::DefUse | EdgeKind::UseDef | EdgeKind::OutputDep
        )
    }
}

// ============================================================================
// PDG
// ============================================================================

/// The Program Dependence Graph.
///
/// Combines control flow and data dependency into one structure so that
/// downstream layers (curvature, sheaf, manifold) can reason about both
/// simultaneously.
#[derive(Debug, Clone)]
pub struct ProgramDependenceGraph {
    pub nodes: Vec<PdgNode>,
    pub edges: Vec<PdgEdge>,
    pub function_name: String,
}

impl ProgramDependenceGraph {
    /// Create an empty PDG for the given function.
    pub fn new(function_name: &str) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            function_name: function_name.to_string(),
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self, kind: NodeKind, label: &str, line: Option<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(PdgNode {
            id,
            kind,
            label: label.to_string(),
            line,
        });
        id
    }

    /// Add an edge between two existing nodes.
    pub fn add_edge(&mut self, from: usize, to: usize, kind: EdgeKind) {
        self.edges.push(PdgEdge { from, to, kind });
    }

    // ── Query helpers ─────────────────────────────────────────────────

    /// All successors of `node_id` with their edge kind.
    pub fn successors(&self, node_id: usize) -> Vec<(usize, EdgeKind)> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id)
            .map(|e| (e.to, e.kind))
            .collect()
    }

    /// Control-flow successors only.
    pub fn cf_successors(&self, node_id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id && e.kind.is_control_flow())
            .map(|e| e.to)
            .collect()
    }

    /// Data-dependency successors only.
    pub fn dd_successors(&self, node_id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id && e.kind.is_data_dependency())
            .map(|e| e.to)
            .collect()
    }

    /// Number of loops (counted as LoopBack edges).
    pub fn loop_count(&self) -> usize {
        // Count both explicit LoopBack edges AND LoopHead nodes
        // (LoopHead is more reliable — always created for while/for/loop)
        let loop_heads = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::LoopHead)
            .count();
        let loop_backs = self
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::LoopBack)
            .count();
        // Use the higher count — LoopHead is always created but LoopBack
        // may be missed if the closing brace heuristic fails
        loop_heads.max(loop_backs)
    }

    /// Number of branch nodes.
    pub fn branch_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Branch)
            .count()
    }

    // ── Conversions ───────────────────────────────────────────────────

    /// Convert to [`Graph`] for standard graph algorithms (BFS, DFS,
    /// topological sort, cycle detection, etc.).
    ///
    /// The resulting graph is directed with unit-weight edges.
    pub fn to_graph(&self) -> Graph {
        let n = self.nodes.len();
        let mut g = Graph::new(n, true);
        for e in &self.edges {
            g.add_edge(e.from, e.to);
        }
        g
    }

    /// Convert to [`SimplicialComplex`] for Hodge-theoretic analysis.
    ///
    /// **Uses control-flow edges ONLY** — data dependency edges (DefUse, UseDef,
    /// OutputDep) are excluded to prevent spurious cycles in the topology.
    /// This gives meaningful Betti numbers:
    /// - β₀ = connected components in control flow
    /// - β₁ = loops (while/for/loop back-edges create actual cycles)
    /// - β₂ = unreachable enclosed regions
    ///
    /// - 0-simplices: all node IDs (vertices)
    /// - 1-simplices: control-flow edges as sorted pairs `[min, max]`
    /// - 2-simplices: triangles from control-flow edges only
    pub fn to_simplicial_complex(&self) -> SimplicialComplex {
        let n = self.nodes.len();

        // 0-simplices
        let vertices: Vec<usize> = (0..n).collect();
        let zero_simplices: Vec<Vec<usize>> = vertices.iter().map(|&v| vec![v]).collect();

        // Build undirected adjacency set for clique detection
        // ONLY from control-flow edges — data dependencies create spurious topology
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

        for e in &self.edges {
            if !e.kind.is_control_flow() {
                continue; // Skip DefUse, UseDef, OutputDep
            }
            let lo = e.from.min(e.to);
            let hi = e.from.max(e.to);
            edge_set.insert((lo, hi));
            adj.entry(lo).or_default().insert(hi);
            adj.entry(hi).or_default().insert(lo);
        }

        // 1-simplices (deduplicated, sorted)
        let one_simplices: Vec<Vec<usize>> = edge_set.iter().map(|&(a, b)| vec![a, b]).collect();

        // 2-simplices: find triangles via adjacency intersection
        let mut two_simplices: Vec<Vec<usize>> = Vec::new();
        let mut seen_triangles: HashSet<(usize, usize, usize)> = HashSet::new();

        for &(u, v) in &edge_set {
            let neighbors_u = adj.get(&u).cloned().unwrap_or_default();
            let neighbors_v = adj.get(&v).cloned().unwrap_or_default();
            for &w in neighbors_u.intersection(&neighbors_v) {
                let mut tri = [u, v, w];
                tri.sort();
                let key = (tri[0], tri[1], tri[2]);
                if seen_triangles.insert(key) {
                    two_simplices.push(vec![tri[0], tri[1], tri[2]]);
                }
            }
        }

        SimplicialComplex {
            vertices,
            simplices: vec![zero_simplices, one_simplices, two_simplices],
            max_dim: 2,
        }
    }

    // ── Simplified Rust source parser ─────────────────────────────────

    /// Build a PDG from Rust source code via simplified line-level analysis.
    ///
    /// Recognises `let` bindings, `if`/`else`, `for`/`while`/`loop`,
    /// `return`, `match`, and bare function calls. Tracks def-use chains
    /// for variables introduced by `let`.
    pub fn from_rust_source(source: &str, function_name: &str) -> Self {
        let mut pdg = Self::new(function_name);

        let entry = pdg.add_node(NodeKind::Entry, &format!("entry: {function_name}"), None);
        let exit = pdg.add_node(NodeKind::Exit, &format!("exit: {function_name}"), None);

        // Variable tracking: name → (defining node id, line)
        let mut defs: HashMap<String, Vec<usize>> = HashMap::new();
        // Previous node for sequential chaining
        let mut prev_node: usize = entry;
        // Stack of (branch/loop node, kind) for nesting
        let mut branch_stack: Vec<(usize, NodeKind)> = Vec::new();
        // Track "else" to connect BranchFalse
        let mut pending_branch_false: Option<usize> = None;
        // Nodes inside current loop body (for LoopBack)
        let mut loop_body_nodes: Vec<Vec<usize>> = Vec::new();

        let lines: Vec<&str> = source.lines().collect();

        for (line_idx, raw_line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = raw_line.trim();

            // Skip empty lines, braces-only, fn signature, comments
            // NOTE: "} else" lines are NOT skipped — they flow through to else-arm detection below
            if trimmed.is_empty()
                || trimmed == "{"
                || trimmed == "}"
                || trimmed.starts_with("//")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
            {
                // Closing brace may end a branch or loop.
                // Use brace-depth tracking for reliable loop close detection.
                if trimmed == "}" {
                    if pending_branch_false.take().is_some() {
                        // End of if-body; the next statement becomes the join point.
                    }

                    // Check if this closes a loop by tracking indentation depth.
                    // A loop closes when its closing brace is at the same indent
                    // as the loop header. We approximate: count leading spaces.
                    if let Some((head_id, kind)) = branch_stack.last().copied() {
                        if kind == NodeKind::LoopHead {
                            // Find the loop header's indentation
                            let head_line = pdg
                                .nodes
                                .iter()
                                .find(|n| n.id == head_id)
                                .and_then(|n| n.line)
                                .and_then(|l| lines.get(l - 1))
                                .map(|l| l.len() - l.trim_start().len())
                                .unwrap_or(0);
                            let this_indent = raw_line.len() - raw_line.trim_start().len();

                            // Close the loop if this brace is at or shallower than
                            // the loop header's indent, and it's a plain "}"
                            // (not "} else" which would be part of an inner branch)
                            if trimmed == "}" && this_indent <= head_line {
                                // LoopBack from last body node to loop head
                                if let Some(body) = loop_body_nodes.last() {
                                    if let Some(&last) = body.last() {
                                        pdg.add_edge(last, head_id, EdgeKind::LoopBack);
                                    }
                                }
                                // LoopExit
                                let exit_node = pdg.add_node(
                                    NodeKind::Statement,
                                    "loop-exit-join",
                                    Some(line_num),
                                );
                                pdg.add_edge(head_id, exit_node, EdgeKind::LoopExit);
                                prev_node = exit_node;
                                branch_stack.pop();
                                loop_body_nodes.pop();
                            }
                        } else if kind == NodeKind::Branch && trimmed == "}" {
                            // Might close a branch — pop if indent matches
                            branch_stack.pop();
                        }
                    }
                }
                continue;
            }

            // ── Detect node kind ──────────────────────────────────────────

            let (kind, label) = if trimmed.starts_with("if ") || trimmed.starts_with("if(") {
                (NodeKind::Branch, trimmed.to_string())
            } else if trimmed == "else {"
                || trimmed.starts_with("else ")
                || trimmed == "} else {"
                || trimmed.starts_with("} else ")
            {
                // Else arm — connect BranchFalse from the pending branch
                if let Some(branch_id) = pending_branch_false.take() {
                    let else_node = pdg.add_node(
                        NodeKind::Statement,
                        &format!("else-body (line {line_num})"),
                        Some(line_num),
                    );
                    pdg.add_edge(branch_id, else_node, EdgeKind::BranchFalse);
                    prev_node = else_node;
                }
                continue;
            } else if trimmed.starts_with("match ") {
                (NodeKind::Branch, trimmed.to_string())
            } else if trimmed.starts_with("for ")
                || trimmed.starts_with("while ")
                || trimmed == "loop {"
                || trimmed == "loop"
            {
                (NodeKind::LoopHead, trimmed.to_string())
            } else if trimmed.contains(".iter()")
                || trimmed.contains(".into_iter()")
                || trimmed.contains(".iter_mut()")
                || trimmed.contains(".map(")
                || trimmed.contains(".filter(")
                || trimmed.contains(".fold(")
                || trimmed.contains(".for_each(")
                || trimmed.contains(".flat_map(")
                || trimmed.contains(".filter_map(")
                || trimmed.contains(".find(")
                || trimmed.contains(".any(")
                || trimmed.contains(".all(")
                || trimmed.contains(".position(")
                || trimmed.contains(".enumerate(")
                || trimmed.contains(".zip(")
                || trimmed.contains(".scan(")
            {
                // Iterator chains are implicit loops — each consumes the
                // collection element-by-element, equivalent to β₁=1.
                (NodeKind::LoopHead, trimmed.to_string())
            } else if trimmed.starts_with("return") {
                (NodeKind::Return, trimmed.to_string())
            } else if trimmed.starts_with("let ") {
                (NodeKind::Statement, trimmed.to_string())
            } else if trimmed.contains('(') && !trimmed.starts_with("let ") {
                // Heuristic: bare function call
                (NodeKind::Call, trimmed.to_string())
            } else {
                (NodeKind::Statement, trimmed.to_string())
            };

            let node = pdg.add_node(kind, &label, Some(line_num));

            // ── Sequential edge from previous ─────────────────────────────
            match kind {
                NodeKind::Branch => {
                    pdg.add_edge(prev_node, node, EdgeKind::Sequential);
                    // Next statement is BranchTrue
                    branch_stack.push((node, NodeKind::Branch));
                    pending_branch_false = Some(node);
                    // Create implicit true-branch target (the next node)
                    prev_node = node;
                }
                NodeKind::LoopHead => {
                    pdg.add_edge(prev_node, node, EdgeKind::Sequential);
                    branch_stack.push((node, NodeKind::LoopHead));
                    loop_body_nodes.push(Vec::new());
                    prev_node = node;
                }
                NodeKind::Return => {
                    pdg.add_edge(prev_node, node, EdgeKind::Sequential);
                    pdg.add_edge(node, exit, EdgeKind::ReturnEdge);
                    prev_node = node;
                }
                _ => {
                    // Determine edge kind based on context
                    if let Some((branch_id, NodeKind::Branch)) = branch_stack.last() {
                        // First statement after a branch is BranchTrue
                        if prev_node == *branch_id {
                            pdg.add_edge(*branch_id, node, EdgeKind::BranchTrue);
                        } else {
                            pdg.add_edge(prev_node, node, EdgeKind::Sequential);
                        }
                    } else {
                        pdg.add_edge(prev_node, node, EdgeKind::Sequential);
                    }
                    prev_node = node;
                }
            }

            // Track loop body nodes
            if let Some(body) = loop_body_nodes.last_mut() {
                body.push(node);
            }

            // ── Def-use tracking ──────────────────────────────────────────
            if trimmed.starts_with("let ") {
                // Extract variable name: "let x = ...", "let mut x = ...", "let x: T = ..."
                let rest = &trimmed[4..];
                let rest = rest.strip_prefix("mut ").unwrap_or(rest);
                // Take identifier (alphanumeric + underscore)
                let var_name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();

                // Check RHS for uses of previously defined variables.
                // The RHS is everything after '=' (if present).
                if let Some(eq_pos) = trimmed.find('=') {
                    let rhs = &trimmed[eq_pos + 1..];
                    for (var, def_nodes) in &defs {
                        if rhs.contains(var.as_str()) {
                            if let Some(&def_node) = def_nodes.last() {
                                if def_node != node {
                                    pdg.add_edge(def_node, node, EdgeKind::DefUse);
                                }
                            }
                        }
                    }
                }

                // Register this definition after checking uses (so self-ref
                // like `let x = x + 1` sees the *previous* x definition).
                if !var_name.is_empty() {
                    defs.entry(var_name).or_default().push(node);
                }
            } else {
                // Check for uses of previously defined variables
                for (var, def_nodes) in &defs {
                    if trimmed.contains(var.as_str()) {
                        // Add DefUse from most recent definition
                        if let Some(&def_node) = def_nodes.last() {
                            if def_node != node {
                                pdg.add_edge(def_node, node, EdgeKind::DefUse);
                            }
                        }
                    }
                }
            }

            // OutputDep: if this is a let-binding for an already-defined var
            if trimmed.starts_with("let ") {
                let rest = &trimmed[4..];
                let rest = rest.strip_prefix("mut ").unwrap_or(rest);
                let var_name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if let Some(prev_defs) = defs.get(&var_name) {
                    if prev_defs.len() > 1 {
                        // Write-after-write: previous def → this def
                        let prev_def = prev_defs[prev_defs.len() - 2];
                        let cur_def = prev_defs[prev_defs.len() - 1];
                        pdg.add_edge(prev_def, cur_def, EdgeKind::OutputDep);
                    }
                }
            }
        }

        // ── Final edges ───────────────────────────────────────────────────
        // If the last node isn't already connected to exit, connect it.
        if prev_node != exit {
            let has_exit_edge = pdg
                .edges
                .iter()
                .any(|e| e.from == prev_node && e.to == exit);
            if !has_exit_edge {
                pdg.add_edge(prev_node, exit, EdgeKind::Sequential);
            }
        }

        pdg
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdg_simple_function() {
        let source = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let pdg = ProgramDependenceGraph::from_rust_source(source, "add");

        assert_eq!(pdg.function_name, "add");

        // Entry + statement("a + b") + Exit = 3 nodes
        assert_eq!(
            pdg.nodes.len(),
            3,
            "expected 3 nodes, got: {:#?}",
            pdg.nodes
        );

        // Entry→Statement, Statement→Exit = 2 sequential edges
        let seq_edges: Vec<_> = pdg
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Sequential)
            .collect();
        assert_eq!(
            seq_edges.len(),
            2,
            "expected 2 sequential edges, got: {:#?}",
            seq_edges
        );

        assert_eq!(pdg.nodes[0].kind, NodeKind::Entry);
        assert_eq!(pdg.nodes[1].kind, NodeKind::Exit);
        assert_eq!(pdg.nodes[2].kind, NodeKind::Statement);
    }

    #[test]
    fn test_pdg_if_else() {
        let source = r#"
fn check(x: i32) -> i32 {
    if x > 0 {
        let pos = x;
    } else {
        let neg = -x;
    }
}
"#;
        let pdg = ProgramDependenceGraph::from_rust_source(source, "check");

        // There should be a Branch node
        assert_eq!(pdg.branch_count(), 1);

        // There should be BranchTrue and BranchFalse edges
        let bt = pdg.edges.iter().any(|e| e.kind == EdgeKind::BranchTrue);
        let bf = pdg.edges.iter().any(|e| e.kind == EdgeKind::BranchFalse);
        assert!(bt, "expected BranchTrue edge");
        assert!(bf, "expected BranchFalse edge");
    }

    #[test]
    fn test_pdg_for_loop() {
        let source = r#"
fn sum_to(n: i32) -> i32 {
    let mut total = 0;
    for i in 0..n {
        let step = i;
    }
    return total;
}
"#;
        let pdg = ProgramDependenceGraph::from_rust_source(source, "sum_to");

        assert_eq!(pdg.loop_count(), 1, "expected 1 loop back-edge");

        // There should be a LoopHead node
        let loop_heads: Vec<_> = pdg
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::LoopHead)
            .collect();
        assert_eq!(loop_heads.len(), 1, "expected 1 LoopHead node");

        // LoopExit edge should exist
        let has_loop_exit = pdg.edges.iter().any(|e| e.kind == EdgeKind::LoopExit);
        assert!(has_loop_exit, "expected LoopExit edge");
    }

    #[test]
    fn test_pdg_def_use_chain() {
        let source = r#"
fn compute() -> i32 {
    let x = 1;
    let y = x + 1;
}
"#;
        let pdg = ProgramDependenceGraph::from_rust_source(source, "compute");

        // There should be a DefUse edge from the "let x" node to the "let y = x + 1" node
        let def_use_edges: Vec<_> = pdg
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::DefUse)
            .collect();
        assert!(
            !def_use_edges.is_empty(),
            "expected at least one DefUse edge for x → y"
        );

        // Verify it goes from the x-definition to the y-definition
        let x_node = pdg
            .nodes
            .iter()
            .find(|n| n.label.contains("let x"))
            .expect("should have let x node");
        let y_node = pdg
            .nodes
            .iter()
            .find(|n| n.label.contains("let y"))
            .expect("should have let y node");

        let found = def_use_edges
            .iter()
            .any(|e| e.from == x_node.id && e.to == y_node.id);
        assert!(found, "DefUse edge should go from x-def to y-use");
    }

    #[test]
    fn test_pdg_to_simplicial() {
        let mut pdg = ProgramDependenceGraph::new("test");
        let a = pdg.add_node(NodeKind::Entry, "a", None);
        let b = pdg.add_node(NodeKind::Statement, "b", None);
        let c = pdg.add_node(NodeKind::Exit, "c", None);

        // Create a triangle using CONTROL-FLOW edges only
        // (data dependency edges are excluded from simplicial complex)
        pdg.add_edge(a, b, EdgeKind::Sequential);
        pdg.add_edge(b, c, EdgeKind::Sequential);
        pdg.add_edge(a, c, EdgeKind::BranchTrue); // control-flow, not DefUse

        let sc = pdg.to_simplicial_complex();

        // 3 vertices
        assert_eq!(sc.vertices.len(), 3);
        // 3 unique edges (undirected, control-flow only)
        assert_eq!(
            sc.simplices[1].len(),
            3,
            "expected 3 one-simplices (CF edges)"
        );
        // 1 triangle (all 3 pairs connected)
        assert_eq!(
            sc.simplices[2].len(),
            1,
            "expected 1 two-simplex (triangle)"
        );

        // Verify DefUse edges are excluded from the complex
        let mut pdg2 = ProgramDependenceGraph::new("test2");
        let x = pdg2.add_node(NodeKind::Entry, "x", None);
        let y = pdg2.add_node(NodeKind::Statement, "y", None);
        pdg2.add_edge(x, y, EdgeKind::Sequential);
        pdg2.add_edge(x, y, EdgeKind::DefUse); // should be ignored
        let sc2 = pdg2.to_simplicial_complex();
        assert_eq!(sc2.simplices[1].len(), 1, "DefUse edge should be excluded");
    }

    #[test]
    fn test_pdg_to_graph() {
        let mut pdg = ProgramDependenceGraph::new("test");
        let a = pdg.add_node(NodeKind::Entry, "a", None);
        let b = pdg.add_node(NodeKind::Statement, "b", None);
        let c = pdg.add_node(NodeKind::Exit, "c", None);
        pdg.add_edge(a, b, EdgeKind::Sequential);
        pdg.add_edge(b, c, EdgeKind::Sequential);

        let g = pdg.to_graph();

        assert_eq!(g.n, 3);
        assert_eq!(g.num_edges(), 2);
        assert!(g.directed);
    }

    #[test]
    fn test_edge_kind_classification() {
        assert!(EdgeKind::Sequential.is_control_flow());
        assert!(EdgeKind::BranchTrue.is_control_flow());
        assert!(EdgeKind::BranchFalse.is_control_flow());
        assert!(EdgeKind::LoopBack.is_control_flow());
        assert!(EdgeKind::LoopExit.is_control_flow());
        assert!(EdgeKind::ReturnEdge.is_control_flow());

        assert!(!EdgeKind::Sequential.is_data_dependency());
        assert!(!EdgeKind::BranchTrue.is_data_dependency());

        assert!(EdgeKind::DefUse.is_data_dependency());
        assert!(EdgeKind::UseDef.is_data_dependency());
        assert!(EdgeKind::OutputDep.is_data_dependency());

        assert!(!EdgeKind::DefUse.is_control_flow());
    }

    #[test]
    fn test_pdg_return_edge() {
        let source = r#"
fn early(x: i32) -> i32 {
    if x > 0 {
        return x;
    }
    return -x;
}
"#;
        let pdg = ProgramDependenceGraph::from_rust_source(source, "early");

        let return_edges: Vec<_> = pdg
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::ReturnEdge)
            .collect();
        assert!(
            !return_edges.is_empty(),
            "expected ReturnEdge(s) connecting return statements to exit"
        );

        // All ReturnEdge targets should be the exit node (id=1)
        for re in &return_edges {
            assert_eq!(re.to, 1, "ReturnEdge should target the exit node");
        }
    }

    #[test]
    fn test_pdg_successors() {
        let mut pdg = ProgramDependenceGraph::new("test");
        let a = pdg.add_node(NodeKind::Entry, "a", None);
        let b = pdg.add_node(NodeKind::Statement, "b", None);
        let c = pdg.add_node(NodeKind::Statement, "c", None);

        pdg.add_edge(a, b, EdgeKind::Sequential);
        pdg.add_edge(a, c, EdgeKind::DefUse);

        let all = pdg.successors(a);
        assert_eq!(all.len(), 2);

        let cf = pdg.cf_successors(a);
        assert_eq!(cf.len(), 1);
        assert_eq!(cf[0], b);

        let dd = pdg.dd_successors(a);
        assert_eq!(dd.len(), 1);
        assert_eq!(dd[0], c);
    }
}
