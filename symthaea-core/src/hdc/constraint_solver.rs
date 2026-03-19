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
}
