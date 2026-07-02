// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Constraint Satisfaction Problem (CSP) Solver
//!
//! Framework for solving constraint satisfaction problems with arc consistency,
//! backtracking with forward checking, and HDC encoding.
//!
//! ## Capabilities
//!
//! - CSP framework: variables, domains, constraints
//! - Arc consistency (AC-3)
//! - Backtracking with forward checking
//! - Built-in constraints: AllDifferent, Sum, LessThan, Equal
//! - HDC encoding for solution states

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use std::collections::{HashMap, HashSet, VecDeque};

// ─── Types ───────────────────────────────────────────────────────────────────

/// A constraint between variables
pub enum Constraint {
    /// All variables must have different values
    AllDifferent(Vec<String>),
    /// Variable must be less than another
    LessThan(String, String),
    /// Variable must equal another
    Equal(String, String),
    /// Variables must sum to a target
    Sum(Vec<String>, i64),
    /// Variable must not equal a specific value
    NotEqual(String, i64),
    /// Custom binary constraint
    Binary(String, String, Box<dyn Fn(i64, i64) -> bool + Send + Sync>),
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllDifferent(v) => f.debug_tuple("AllDifferent").field(v).finish(),
            Self::LessThan(a, b) => f.debug_tuple("LessThan").field(a).field(b).finish(),
            Self::Equal(a, b) => f.debug_tuple("Equal").field(a).field(b).finish(),
            Self::Sum(v, t) => f.debug_tuple("Sum").field(v).field(t).finish(),
            Self::NotEqual(v, val) => f.debug_tuple("NotEqual").field(v).field(val).finish(),
            Self::Binary(a, b, _) => f
                .debug_tuple("Binary")
                .field(a)
                .field(b)
                .field(&"<fn>")
                .finish(),
        }
    }
}

impl Constraint {
    /// Get all variables involved in this constraint
    pub fn variables(&self) -> Vec<String> {
        match self {
            Constraint::AllDifferent(vars) => vars.clone(),
            Constraint::LessThan(a, b) | Constraint::Equal(a, b) => vec![a.clone(), b.clone()],
            Constraint::Sum(vars, _) => vars.clone(),
            Constraint::NotEqual(v, _) => vec![v.clone()],
            Constraint::Binary(a, b, _) => vec![a.clone(), b.clone()],
        }
    }

    /// Check if the constraint is satisfied by the given assignment
    pub fn is_satisfied(&self, assignment: &HashMap<String, i64>) -> bool {
        match self {
            Constraint::AllDifferent(vars) => {
                let values: Vec<i64> = vars
                    .iter()
                    .filter_map(|v| assignment.get(v))
                    .copied()
                    .collect();
                let unique: HashSet<i64> = values.iter().copied().collect();
                values.len() == unique.len()
            }
            Constraint::LessThan(a, b) => {
                match (assignment.get(a), assignment.get(b)) {
                    (Some(va), Some(vb)) => va < vb,
                    _ => true, // Not yet assigned
                }
            }
            Constraint::Equal(a, b) => match (assignment.get(a), assignment.get(b)) {
                (Some(va), Some(vb)) => va == vb,
                _ => true,
            },
            Constraint::Sum(vars, target) => {
                if vars.iter().all(|v| assignment.contains_key(v)) {
                    let sum: i64 = vars.iter().map(|v| assignment[v]).sum();
                    sum == *target
                } else {
                    true // Not all assigned yet
                }
            }
            Constraint::NotEqual(v, val) => match assignment.get(v) {
                Some(va) => va != val,
                None => true,
            },
            Constraint::Binary(a, b, check) => match (assignment.get(a), assignment.get(b)) {
                (Some(va), Some(vb)) => check(*va, *vb),
                _ => true,
            },
        }
    }
}

/// A constraint satisfaction problem
pub struct CSP {
    /// Variable names to their domains (possible values)
    pub domains: HashMap<String, Vec<i64>>,
    /// Constraints
    pub constraints: Vec<Constraint>,
    /// Variable ordering (for search)
    pub variables: Vec<String>,
}

/// Result of solving a CSP
#[derive(Debug, Clone)]
pub struct CSPResult {
    /// Solution assignment (if found)
    pub solution: Option<HashMap<String, i64>>,
    /// All solutions found (if requested)
    pub all_solutions: Vec<HashMap<String, i64>>,
    /// Number of backtracks
    pub backtracks: usize,
    /// Number of nodes explored
    pub nodes_explored: usize,
    /// Whether a solution was found
    pub solved: bool,
    /// Phi measurement
    pub phi: f64,
    /// HDC encoding
    pub encoding: BinaryHV,
}

// ─── CSP Solver ──────────────────────────────────────────────────────────────

/// The Hyperdimensional CSP Solver
pub struct CSPSolver;

impl CSPSolver {
    /// Solve a CSP using backtracking with forward checking.
    pub fn solve(csp: &CSP) -> CSPResult {
        let mut assignment = HashMap::new();
        let mut domains = csp.domains.clone();
        let mut backtracks = 0;
        let mut nodes = 0;

        // Apply AC-3 first
        Self::ac3(&mut domains, &csp.constraints);

        let solved = Self::backtrack(
            &csp.variables,
            0,
            &mut assignment,
            &mut domains,
            &csp.constraints,
            &mut backtracks,
            &mut nodes,
        );

        let phi = if solved {
            0.3 + 0.2 / (1.0 + backtracks as f64 / 10.0)
        } else {
            0.1
        };

        let encoding = Self::encode_solution(&assignment);

        CSPResult {
            solution: if solved { Some(assignment) } else { None },
            all_solutions: Vec::new(),
            backtracks,
            nodes_explored: nodes,
            solved,
            phi,
            encoding,
        }
    }

    /// Find all solutions
    pub fn solve_all(csp: &CSP) -> CSPResult {
        let mut domains = csp.domains.clone();
        Self::ac3(&mut domains, &csp.constraints);

        let mut all_solutions = Vec::new();
        let mut assignment = HashMap::new();
        let mut backtracks = 0;
        let mut nodes = 0;

        Self::backtrack_all(
            &csp.variables,
            0,
            &mut assignment,
            &mut domains,
            &csp.constraints,
            &mut all_solutions,
            &mut backtracks,
            &mut nodes,
        );

        let solved = !all_solutions.is_empty();
        let phi = if solved { 0.4 } else { 0.1 };

        CSPResult {
            solution: all_solutions.first().cloned(),
            all_solutions,
            backtracks,
            nodes_explored: nodes,
            solved,
            phi,
            encoding: Self::encode_solution(&HashMap::new()),
        }
    }

    fn backtrack(
        variables: &[String],
        idx: usize,
        assignment: &mut HashMap<String, i64>,
        domains: &mut HashMap<String, Vec<i64>>,
        constraints: &[Constraint],
        backtracks: &mut usize,
        nodes: &mut usize,
    ) -> bool {
        *nodes += 1;

        if idx == variables.len() {
            return true; // All variables assigned
        }

        let var = &variables[idx];
        let domain = domains.get(var).cloned().unwrap_or_default();

        for &value in &domain {
            assignment.insert(var.clone(), value);

            // Check constraints
            let consistent = constraints.iter().all(|c| c.is_satisfied(assignment));

            if consistent {
                // Forward checking: prune future domains
                let mut saved_domains = HashMap::new();
                let mut valid = true;

                for future_var in &variables[idx + 1..] {
                    if let Some(future_domain) = domains.get(future_var) {
                        let pruned: Vec<i64> = future_domain
                            .iter()
                            .filter(|&&val| {
                                assignment.insert(future_var.clone(), val);
                                let ok = constraints.iter().all(|c| c.is_satisfied(assignment));
                                assignment.remove(future_var);
                                ok
                            })
                            .copied()
                            .collect();

                        saved_domains.insert(future_var.clone(), future_domain.clone());
                        if pruned.is_empty() {
                            valid = false;
                            break;
                        }
                        domains.insert(future_var.clone(), pruned);
                    }
                }

                if valid
                    && Self::backtrack(
                        variables,
                        idx + 1,
                        assignment,
                        domains,
                        constraints,
                        backtracks,
                        nodes,
                    )
                {
                    // Restore domains before returning
                    for (var, domain) in saved_domains {
                        domains.insert(var, domain);
                    }
                    return true;
                }

                // Restore domains
                for (var, domain) in saved_domains {
                    domains.insert(var, domain);
                }
            }

            *backtracks += 1;
        }

        assignment.remove(var);
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn backtrack_all(
        variables: &[String],
        idx: usize,
        assignment: &mut HashMap<String, i64>,
        domains: &mut HashMap<String, Vec<i64>>,
        constraints: &[Constraint],
        solutions: &mut Vec<HashMap<String, i64>>,
        backtracks: &mut usize,
        nodes: &mut usize,
    ) {
        *nodes += 1;

        if idx == variables.len() {
            solutions.push(assignment.clone());
            return;
        }

        let var = &variables[idx];
        let domain = domains.get(var).cloned().unwrap_or_default();

        for &value in &domain {
            assignment.insert(var.clone(), value);

            let consistent = constraints.iter().all(|c| c.is_satisfied(assignment));

            if consistent {
                Self::backtrack_all(
                    variables,
                    idx + 1,
                    assignment,
                    domains,
                    constraints,
                    solutions,
                    backtracks,
                    nodes,
                );
            } else {
                *backtracks += 1;
            }
        }

        assignment.remove(var);
    }

    // ─── AC-3 ────────────────────────────────────────────────────────────

    /// Arc Consistency (AC-3) preprocessing.
    ///
    /// Removes values from domains that can never participate in a solution
    /// by propagating binary constraints.
    fn ac3(domains: &mut HashMap<String, Vec<i64>>, constraints: &[Constraint]) {
        // Build arc queue from binary constraints
        let mut queue: VecDeque<(String, String)> = VecDeque::new();

        for constraint in constraints {
            match constraint {
                Constraint::LessThan(a, b) | Constraint::Equal(a, b) => {
                    queue.push_back((a.clone(), b.clone()));
                    queue.push_back((b.clone(), a.clone()));
                }
                Constraint::AllDifferent(vars) => {
                    for i in 0..vars.len() {
                        for j in 0..vars.len() {
                            if i != j {
                                queue.push_back((vars[i].clone(), vars[j].clone()));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        while let Some((xi, xj)) = queue.pop_front() {
            if Self::revise(domains, &xi, &xj, constraints) {
                if domains.get(&xi).is_none_or(|d| d.is_empty()) {
                    return; // No solution exists
                }
                // Add all arcs (xk, xi) back to queue
                for constraint in constraints {
                    for var in constraint.variables() {
                        if var != xi && var != xj {
                            queue.push_back((var, xi.clone()));
                        }
                    }
                }
            }
        }
    }

    fn revise(
        domains: &mut HashMap<String, Vec<i64>>,
        xi: &str,
        xj: &str,
        constraints: &[Constraint],
    ) -> bool {
        let di = match domains.get(xi) {
            Some(d) => d.clone(),
            None => return false,
        };
        let dj = match domains.get(xj) {
            Some(d) => d.clone(),
            None => return false,
        };

        let mut revised = false;
        let mut new_domain = Vec::new();

        for &vi in &di {
            // Check if there exists a value in dj consistent with vi
            let has_support = dj.iter().any(|&vj| {
                let mut assignment = HashMap::new();
                assignment.insert(xi.to_string(), vi);
                assignment.insert(xj.to_string(), vj);
                constraints.iter().all(|c| {
                    let vars = c.variables();
                    if vars.contains(&xi.to_string()) && vars.contains(&xj.to_string()) {
                        c.is_satisfied(&assignment)
                    } else {
                        true
                    }
                })
            });

            if has_support {
                new_domain.push(vi);
            } else {
                revised = true;
            }
        }

        if revised {
            domains.insert(xi.to_string(), new_domain);
        }
        revised
    }

    // ─── Problem Builders ────────────────────────────────────────────────

    /// Create an N-Queens problem
    pub fn n_queens(n: usize) -> CSP {
        let variables: Vec<String> = (0..n).map(|i| format!("Q{}", i)).collect();
        let domain: Vec<i64> = (0..n as i64).collect();
        let mut domains = HashMap::new();
        for var in &variables {
            domains.insert(var.clone(), domain.clone());
        }

        let mut constraints = Vec::new();

        // All queens in different columns
        constraints.push(Constraint::AllDifferent(variables.clone()));

        // No two queens on same diagonal
        for i in 0..n {
            for j in (i + 1)..n {
                let qi = variables[i].clone();
                let qj = variables[j].clone();
                let diff = (j - i) as i64;
                constraints.push(Constraint::Binary(
                    qi,
                    qj,
                    Box::new(move |vi, vj| (vi - vj).abs() != diff),
                ));
            }
        }

        CSP {
            domains,
            constraints,
            variables,
        }
    }

    /// Create a graph coloring problem
    pub fn graph_coloring(edges: &[(usize, usize)], n_vertices: usize, n_colors: usize) -> CSP {
        let variables: Vec<String> = (0..n_vertices).map(|i| format!("V{}", i)).collect();
        let domain: Vec<i64> = (0..n_colors as i64).collect();
        let mut domains = HashMap::new();
        for var in &variables {
            domains.insert(var.clone(), domain.clone());
        }

        let mut constraints = Vec::new();
        for &(u, v) in edges {
            constraints.push(Constraint::Binary(
                variables[u].clone(),
                variables[v].clone(),
                Box::new(|a, b| a != b),
            ));
        }

        CSP {
            domains,
            constraints,
            variables,
        }
    }

    // ─── Convenience Solvers ────────────────────────────────────────────

    /// Solve N-Queens and return the solution as a vector of column positions.
    ///
    /// `result[row]` = column of the queen in that row.
    /// Returns `None` if no solution exists (only for n=2 or n=3).
    pub fn solve_n_queens(n: usize) -> Option<Vec<i64>> {
        let csp = Self::n_queens(n);
        let result = Self::solve(&csp);
        result
            .solution
            .map(|sol| (0..n).map(|i| sol[&format!("Q{}", i)]).collect())
    }

    /// Create a Sudoku CSP from a 9x9 grid.
    ///
    /// `grid[row][col]` = 0 for empty, 1-9 for given digits.
    pub fn sudoku(grid: &[[i32; 9]; 9]) -> CSP {
        let mut variables = Vec::new();
        let mut domains = HashMap::new();

        for row in 0..9 {
            for col in 0..9 {
                let var = format!("R{}C{}", row, col);
                variables.push(var.clone());
                if grid[row][col] != 0 {
                    // Fixed cell — domain is just the given value
                    domains.insert(var, vec![grid[row][col] as i64]);
                } else {
                    domains.insert(var, (1..=9).map(|v| v as i64).collect());
                }
            }
        }

        let mut constraints = Vec::new();

        // Row constraints: all different in each row
        for row in 0..9 {
            let row_vars: Vec<String> = (0..9).map(|col| format!("R{}C{}", row, col)).collect();
            constraints.push(Constraint::AllDifferent(row_vars));
        }

        // Column constraints: all different in each column
        for col in 0..9 {
            let col_vars: Vec<String> = (0..9).map(|row| format!("R{}C{}", row, col)).collect();
            constraints.push(Constraint::AllDifferent(col_vars));
        }

        // Box constraints: all different in each 3x3 box
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut box_vars = Vec::new();
                for dr in 0..3 {
                    for dc in 0..3 {
                        let r = box_row * 3 + dr;
                        let c = box_col * 3 + dc;
                        box_vars.push(format!("R{}C{}", r, c));
                    }
                }
                constraints.push(Constraint::AllDifferent(box_vars));
            }
        }

        CSP {
            domains,
            constraints,
            variables,
        }
    }

    /// Solve a Sudoku puzzle.
    ///
    /// `grid[row][col]` = 0 for empty, 1-9 for given digits.
    /// Returns the completed 9x9 grid, or `None` if unsolvable.
    pub fn solve_sudoku(grid: &[[i32; 9]; 9]) -> Option<[[i32; 9]; 9]> {
        let csp = Self::sudoku(grid);
        let result = Self::solve(&csp);
        result.solution.map(|sol| {
            let mut out = [[0i32; 9]; 9];
            for row in 0..9 {
                for col in 0..9 {
                    out[row][col] = sol[&format!("R{}C{}", row, col)] as i32;
                }
            }
            out
        })
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn encode_solution(assignment: &HashMap<String, i64>) -> BinaryHV {
        let csp_prim = BinaryHV::random(seed_from_name("CONSTRAINT"));
        let mut encoding = csp_prim;
        let mut sorted_keys: Vec<&String> = assignment.keys().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            let val = assignment[key];
            let var_hv = BinaryHV::random(seed_from_name(&format!("CSP_{}", key)));
            let val_hv = BinaryHV::random(seed_from_name(&format!("CSP_VAL_{}", val)));
            encoding = encoding.bind(&var_hv.bind(&val_hv));
        }
        encoding
    }
}

// ─── MRV and LCV Heuristics ──────────────────────────────────────────────────

/// MRV (Minimum Remaining Values) heuristic: select the unassigned variable
/// with the smallest domain size.
///
/// Breaks ties by degree (most constraints with unassigned variables).
pub fn mrv_select_variable(domains: &[Vec<usize>], assigned: &[Option<usize>]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_size = usize::MAX;

    for (i, domain) in domains.iter().enumerate() {
        if assigned[i].is_some() {
            continue;
        }
        if domain.len() < best_size {
            best_size = domain.len();
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Degree heuristic: count the number of constraints between `var_idx` and
/// unassigned variables.
pub fn degree_heuristic(
    var_idx: usize,
    constraints: &[&dyn Fn(usize, usize, usize, usize) -> bool],
    n_vars: usize,
    assigned: &[Option<usize>],
) -> usize {
    // We approximate degree as the number of other unassigned variables that
    // share at least one constraint with var_idx.
    // Since the constraint fn takes (var1, val1, var2, val2), we check by probing.
    let mut degree = 0;
    for other in 0..n_vars {
        if other == var_idx || assigned[other].is_some() {
            continue;
        }
        // Check if any constraint involves both var_idx and other
        // We can't enumerate constraints by variable easily with the fn interface,
        // so we probe with dummy values (0,0) and count `other` once if any
        // constraint fn is triggered between them.
        for c in constraints {
            if c(var_idx, 0, other, 0) {
                degree += 1;
                break; // count this variable as constrained, don't double-count
            }
        }
    }
    degree
}

/// LCV (Least Constraining Value) heuristic: order domain values by how few
/// values they eliminate from neighbor domains.
///
/// Values that eliminate fewer choices from neighbors should be tried first.
pub fn lcv_order_values(
    var_idx: usize,
    domain: &[usize],
    domains: &[Vec<usize>],
    assigned: &[Option<usize>],
    constraints: &[Box<dyn Fn(usize, usize, usize, usize) -> bool>],
) -> Vec<usize> {
    let mut scored: Vec<(usize, usize)> = domain
        .iter()
        .map(|&val| {
            let eliminated = count_eliminations(var_idx, val, domains, assigned, constraints);
            (val, eliminated)
        })
        .collect();
    // Sort ascending: fewer eliminations → try first
    scored.sort_by_key(|(_, elim)| *elim);
    scored.into_iter().map(|(v, _)| v).collect()
}

/// Count how many values would be eliminated from neighbor domains if
/// variable `var_idx` is assigned value `val`.
fn count_eliminations(
    var_idx: usize,
    val: usize,
    domains: &[Vec<usize>],
    assigned: &[Option<usize>],
    constraints: &[Box<dyn Fn(usize, usize, usize, usize) -> bool>],
) -> usize {
    let mut eliminated = 0;
    for (other, domain) in domains.iter().enumerate() {
        if other == var_idx || assigned[other].is_some() {
            continue;
        }
        for &other_val in domain {
            // Check if assigning var_idx=val would eliminate other_val from domain[other]
            let violates = constraints
                .iter()
                .any(|c| !c(var_idx, val, other, other_val));
            if violates {
                eliminated += 1;
            }
        }
    }
    eliminated
}

/// Backtracking CSP solver with MRV + LCV heuristics + forward checking.
///
/// Returns true if a solution was found (assignment will be fully populated).
/// `backtracks` counts the number of backtrack operations performed.
pub fn backtrack_mrv_lcv(
    domains: &mut Vec<Vec<usize>>,
    assigned: &mut Vec<Option<usize>>,
    constraints: &[Box<dyn Fn(usize, usize, usize, usize) -> bool>],
    backtracks: &mut usize,
) -> bool {
    // Check if all variables are assigned
    if assigned.iter().all(|a| a.is_some()) {
        return true;
    }

    // MRV: pick variable with smallest domain
    let var = match mrv_select_variable(domains, assigned) {
        Some(v) => v,
        None => return false,
    };

    // LCV: order values to try
    let ordered_values =
        lcv_order_values(var, &domains[var].clone(), domains, assigned, constraints);

    for val in ordered_values {
        // Check consistency: does assigning var=val violate any constraint
        // with already-assigned variables?
        let consistent = assigned.iter().enumerate().all(|(other, other_val)| {
            if let Some(ov) = other_val {
                constraints
                    .iter()
                    .all(|c| c(var, val, other, *ov) && c(other, *ov, var, val))
            } else {
                true
            }
        });

        if !consistent {
            continue;
        }

        // Forward checking: ensure no neighbor domain becomes empty
        assigned[var] = Some(val);
        let mut pruned: Vec<(usize, usize)> = Vec::new(); // (var_idx, value) that were pruned
        let mut fc_ok = true;

        for other in 0..domains.len() {
            if assigned[other].is_some() {
                continue;
            }
            let original_len = domains[other].len();
            let surviving: Vec<usize> = domains[other]
                .iter()
                .copied()
                .filter(|&ov| {
                    constraints
                        .iter()
                        .all(|c| c(var, val, other, ov) && c(other, ov, var, val))
                })
                .collect();

            if surviving.is_empty() {
                fc_ok = false;
                break;
            }

            // Record what was pruned
            for v in &domains[other] {
                if !surviving.contains(v) {
                    pruned.push((other, *v));
                }
            }

            let _ = original_len;
            domains[other] = surviving;
        }

        if fc_ok && backtrack_mrv_lcv(domains, assigned, constraints, backtracks) {
            return true;
        }

        // Undo: restore pruned values
        assigned[var] = None;
        *backtracks += 1;
        for (other, pruned_val) in pruned {
            if !domains[other].contains(&pruned_val) {
                domains[other].push(pruned_val);
                domains[other].sort();
            }
        }
    }

    false
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── N-Queens ─────────────────────────────────────────────────────────

    #[test]
    fn test_4_queens() {
        let csp = CSPSolver::n_queens(4);
        let result = CSPSolver::solve(&csp);
        assert!(result.solved, "4-Queens should have a solution");

        let solution = result.solution.unwrap();
        // Verify all columns different
        let columns: Vec<i64> = (0..4).map(|i| solution[&format!("Q{}", i)]).collect();
        let unique: HashSet<i64> = columns.iter().copied().collect();
        assert_eq!(unique.len(), 4, "All queens must be in different columns");

        // Verify no diagonal conflicts
        for i in 0..4 {
            for j in (i + 1)..4 {
                let row_diff = (j - i) as i64;
                let col_diff = (columns[i] - columns[j]).abs();
                assert_ne!(
                    row_diff, col_diff,
                    "Queens {} and {} on same diagonal",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_8_queens() {
        let csp = CSPSolver::n_queens(8);
        let result = CSPSolver::solve(&csp);
        assert!(result.solved, "8-Queens should have a solution");
    }

    #[test]
    fn test_4_queens_all_solutions() {
        let csp = CSPSolver::n_queens(4);
        let result = CSPSolver::solve_all(&csp);
        assert_eq!(
            result.all_solutions.len(),
            2,
            "4-Queens has exactly 2 solutions"
        );
    }

    // ── Graph Coloring ───────────────────────────────────────────────────

    #[test]
    fn test_graph_coloring_triangle() {
        // Triangle graph needs 3 colors
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let csp = CSPSolver::graph_coloring(&edges, 3, 3);
        let result = CSPSolver::solve(&csp);
        assert!(result.solved);

        let solution = result.solution.unwrap();
        for &(u, v) in &edges {
            assert_ne!(
                solution[&format!("V{}", u)],
                solution[&format!("V{}", v)],
                "Adjacent vertices must have different colors"
            );
        }
    }

    #[test]
    fn test_graph_coloring_insufficient() {
        // Triangle with only 2 colors is unsolvable
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let csp = CSPSolver::graph_coloring(&edges, 3, 2);
        let result = CSPSolver::solve(&csp);
        assert!(!result.solved);
    }

    #[test]
    fn test_graph_coloring_bipartite() {
        // Bipartite graph (path 0-1-2-3) needs only 2 colors
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let csp = CSPSolver::graph_coloring(&edges, 4, 2);
        let result = CSPSolver::solve(&csp);
        assert!(result.solved);
    }

    // ── Custom CSP ───────────────────────────────────────────────────────

    #[test]
    fn test_sum_constraint() {
        // X + Y = 10, X in {1..9}, Y in {1..9}
        let mut domains = HashMap::new();
        let domain: Vec<i64> = (1..=9).collect();
        domains.insert("X".to_string(), domain.clone());
        domains.insert("Y".to_string(), domain);

        let csp = CSP {
            domains,
            constraints: vec![Constraint::Sum(vec!["X".to_string(), "Y".to_string()], 10)],
            variables: vec!["X".to_string(), "Y".to_string()],
        };

        let result = CSPSolver::solve(&csp);
        assert!(result.solved);
        let sol = result.solution.unwrap();
        assert_eq!(sol["X"] + sol["Y"], 10);
    }

    #[test]
    fn test_less_than_constraint() {
        let mut domains = HashMap::new();
        domains.insert("X".to_string(), vec![1, 2, 3, 4, 5]);
        domains.insert("Y".to_string(), vec![1, 2, 3, 4, 5]);

        let csp = CSP {
            domains,
            constraints: vec![Constraint::LessThan("X".to_string(), "Y".to_string())],
            variables: vec!["X".to_string(), "Y".to_string()],
        };

        let result = CSPSolver::solve(&csp);
        assert!(result.solved);
        let sol = result.solution.unwrap();
        assert!(sol["X"] < sol["Y"]);
    }

    #[test]
    fn test_all_different() {
        let mut domains = HashMap::new();
        domains.insert("A".to_string(), vec![1, 2, 3]);
        domains.insert("B".to_string(), vec![1, 2, 3]);
        domains.insert("C".to_string(), vec![1, 2, 3]);

        let csp = CSP {
            domains,
            constraints: vec![Constraint::AllDifferent(vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ])],
            variables: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };

        let result = CSPSolver::solve_all(&csp);
        assert_eq!(result.all_solutions.len(), 6); // 3! = 6 permutations
    }

    #[test]
    fn test_not_equal_constraint() {
        let mut domains = HashMap::new();
        domains.insert("X".to_string(), vec![1, 2, 3]);

        let csp = CSP {
            domains,
            constraints: vec![
                Constraint::NotEqual("X".to_string(), 1),
                Constraint::NotEqual("X".to_string(), 3),
            ],
            variables: vec!["X".to_string()],
        };

        let result = CSPSolver::solve(&csp);
        assert!(result.solved);
        assert_eq!(result.solution.unwrap()["X"], 2);
    }

    // ── Convenience solvers ────────────────────────────────────────────

    #[test]
    fn test_solve_n_queens_4() {
        let sol = CSPSolver::solve_n_queens(4);
        assert!(sol.is_some(), "4-Queens should have a solution");
        let queens = sol.unwrap();
        assert_eq!(queens.len(), 4);
        // Verify all columns different
        let unique: HashSet<i64> = queens.iter().copied().collect();
        assert_eq!(unique.len(), 4);
        // Verify no diagonal conflicts
        for i in 0..4 {
            for j in (i + 1)..4 {
                let row_diff = (j - i) as i64;
                let col_diff = (queens[i] - queens[j]).abs();
                assert_ne!(row_diff, col_diff);
            }
        }
    }

    #[test]
    fn test_solve_n_queens_8() {
        let sol = CSPSolver::solve_n_queens(8);
        assert!(sol.is_some(), "8-Queens should have a solution");
        assert_eq!(sol.unwrap().len(), 8);
    }

    // ── Sudoku ──────────────────────────────────────────────────────────

    #[test]
    fn test_sudoku_simple() {
        // A puzzle with many givens (easy)
        #[rustfmt::skip]
        let grid: [[i32; 9]; 9] = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ];

        let result = CSPSolver::solve_sudoku(&grid);
        assert!(result.is_some(), "Simple Sudoku should be solvable");

        let sol = result.unwrap();

        // Verify all givens preserved
        for row in 0..9 {
            for col in 0..9 {
                if grid[row][col] != 0 {
                    assert_eq!(
                        sol[row][col], grid[row][col],
                        "Given at ({},{}) changed",
                        row, col
                    );
                }
                assert!(
                    sol[row][col] >= 1 && sol[row][col] <= 9,
                    "Value out of range at ({},{}): {}",
                    row,
                    col,
                    sol[row][col]
                );
            }
        }

        // Verify row uniqueness
        for row in 0..9 {
            let vals: HashSet<i32> = sol[row].iter().copied().collect();
            assert_eq!(vals.len(), 9, "Row {} has duplicates", row);
        }

        // Verify column uniqueness
        for col in 0..9 {
            let vals: HashSet<i32> = (0..9).map(|row| sol[row][col]).collect();
            assert_eq!(vals.len(), 9, "Column {} has duplicates", col);
        }

        // Verify box uniqueness
        for br in 0..3 {
            for bc in 0..3 {
                let mut vals = HashSet::new();
                for dr in 0..3 {
                    for dc in 0..3 {
                        vals.insert(sol[br * 3 + dr][bc * 3 + dc]);
                    }
                }
                assert_eq!(vals.len(), 9, "Box ({},{}) has duplicates", br, bc);
            }
        }
    }

    #[test]
    fn test_sudoku_csp_creation() {
        let grid = [[0i32; 9]; 9];
        let csp = CSPSolver::sudoku(&grid);
        // 81 variables, each with domain 1..9
        assert_eq!(csp.variables.len(), 81);
        assert_eq!(csp.domains["R0C0"].len(), 9);
        // 9 row + 9 col + 9 box = 27 AllDifferent constraints
        assert_eq!(csp.constraints.len(), 27);
    }

    // ── Encoding ─────────────────────────────────────────────────────────

    #[test]
    fn test_solution_encoding() {
        let mut assignment = HashMap::new();
        assignment.insert("X".to_string(), 1);
        assignment.insert("Y".to_string(), 2);
        let enc = CSPSolver::encode_solution(&assignment);

        let mut assignment2 = HashMap::new();
        assignment2.insert("X".to_string(), 3);
        assignment2.insert("Y".to_string(), 4);
        let enc2 = CSPSolver::encode_solution(&assignment2);

        let sim = enc.similarity(&enc2);
        assert!(
            sim < 0.6,
            "Different solutions should have different encodings: {}",
            sim
        );
    }

    // ── MRV heuristic ────────────────────────────────────────────────────

    #[test]
    fn test_mrv_selects_most_constrained_variable() {
        // Domains: var0=[1,2,3], var1=[1], var2=[1,2]
        // MRV should pick var1 (smallest domain)
        let domains = vec![vec![1, 2, 3], vec![1], vec![1, 2]];
        let assigned = vec![None, None, None];
        let selected = mrv_select_variable(&domains, &assigned);
        assert_eq!(
            selected,
            Some(1),
            "MRV should select var1 with domain size 1"
        );
    }

    #[test]
    fn test_mrv_skips_assigned_variables() {
        let domains = vec![vec![1], vec![1, 2, 3], vec![1, 2]];
        let assigned = vec![Some(1), None, None]; // var0 already assigned
        let selected = mrv_select_variable(&domains, &assigned);
        // Should pick var2 (size 2) not var0 (assigned)
        assert_eq!(selected, Some(2), "MRV should skip assigned var0");
    }

    #[test]
    fn test_mrv_returns_none_when_all_assigned() {
        let domains = vec![vec![1], vec![2]];
        let assigned = vec![Some(1), Some(2)];
        assert_eq!(mrv_select_variable(&domains, &assigned), None);
    }

    // ── LCV heuristic ────────────────────────────────────────────────────

    #[test]
    fn test_lcv_orders_values() {
        // Simple CSP: var0 ∈ {1,2,3}, var1 ∈ {1,2,3}
        // Constraint: var0 ≠ var1 (AllDifferent)
        let domains = vec![vec![1, 2, 3], vec![1, 2, 3]];
        let assigned = vec![None, None];
        let constraints: Vec<Box<dyn Fn(usize, usize, usize, usize) -> bool>> = vec![Box::new(
            |v1, val1, v2, val2| {
                if v1 == v2 { true } else { val1 != val2 }
            },
        )];
        let ordered = lcv_order_values(0, &domains[0], &domains, &assigned, &constraints);
        // All values should be present
        assert_eq!(ordered.len(), 3);
    }

    // ── Backtracking with MRV + LCV ──────────────────────────────────────

    #[test]
    fn test_backtrack_mrv_lcv_simple_csp() {
        // 2-coloring: 3 vars, each ∈ {0, 1}, all must differ from neighbors
        // Graph: 0-1, 1-2 (chain)
        let mut domains = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
        let mut assigned = vec![None, None, None];
        let mut backtracks = 0;
        // Constraints: adjacent vars in the chain must differ
        let constraints: Vec<Box<dyn Fn(usize, usize, usize, usize) -> bool>> = vec![
            Box::new(|v1, val1, v2, val2| {
                // Edge 0-1
                if (v1 == 0 && v2 == 1) || (v1 == 1 && v2 == 0) {
                    val1 != val2
                } else {
                    true
                }
            }),
            Box::new(|v1, val1, v2, val2| {
                // Edge 1-2
                if (v1 == 1 && v2 == 2) || (v1 == 2 && v2 == 1) {
                    val1 != val2
                } else {
                    true
                }
            }),
        ];
        let solved = backtrack_mrv_lcv(&mut domains, &mut assigned, &constraints, &mut backtracks);
        assert!(solved, "Simple chain 2-coloring should be solvable");
        // Verify solution
        let a0 = assigned[0].unwrap();
        let a1 = assigned[1].unwrap();
        let a2 = assigned[2].unwrap();
        assert_ne!(a0, a1, "Adjacent vars 0-1 must differ");
        assert_ne!(a1, a2, "Adjacent vars 1-2 must differ");
    }

    #[test]
    fn test_4x4_sudoku_with_mrv_lcv() {
        // 4×4 Sudoku: digits 1-4, rows/cols/2x2 boxes all different
        // Initial partial grid:
        // [1, _, _, _]
        // [_, _, 1, _]
        // [_, 1, _, _]
        // [_, _, _, 1]
        let n = 4;
        // 16 variables, each initially domain {1,2,3,4}
        let mut domains: Vec<Vec<usize>> = vec![(1..=n).collect(); n * n];
        let mut assigned: Vec<Option<usize>> = vec![None; n * n];

        // Pre-assign initial clues
        let clues = vec![(0, 0, 1), (1, 2, 1), (2, 1, 1), (3, 3, 1)];
        for (row, col, val) in &clues {
            let idx = row * n + col;
            assigned[idx] = Some(*val);
            domains[idx] = vec![*val];
        }

        // Build AllDifferent-style constraints for rows, columns, and 2x2 boxes
        let constraints: Vec<Box<dyn Fn(usize, usize, usize, usize) -> bool>> =
            vec![Box::new(move |v1, val1, v2, val2| {
                if v1 == v2 {
                    return true;
                }
                let (r1, c1) = (v1 / n, v1 % n);
                let (r2, c2) = (v2 / n, v2 % n);
                // Same row or same column or same 2x2 box: values must differ
                let same_row = r1 == r2;
                let same_col = c1 == c2;
                let same_box = (r1 / 2 == r2 / 2) && (c1 / 2 == c2 / 2);
                if same_row || same_col || same_box {
                    val1 != val2
                } else {
                    true
                }
            })];

        let mut backtracks = 0;
        let solved = backtrack_mrv_lcv(&mut domains, &mut assigned, &constraints, &mut backtracks);
        assert!(solved, "4x4 Sudoku should be solvable");

        // Verify row uniqueness
        for row in 0..n {
            let vals: Vec<usize> = (0..n).map(|col| assigned[row * n + col].unwrap()).collect();
            let unique: std::collections::HashSet<_> = vals.iter().collect();
            assert_eq!(unique.len(), n, "Row {} has duplicates", row);
        }
    }
}
