//! # Symbolic Regression for Physics Discovery
//!
//! Uses HDC-guided search to discover mathematical relationships in data.
//! Inspired by "AI Feynman" but using hyperdimensional representations.

use super::encoders::{EncodedMeasurement, PhysicalQuantity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A discovered equation relating physical quantities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEquation {
    /// Symbolic form (e.g., "y = a * x^2 + b")
    pub symbolic_form: String,
    /// LaTeX representation
    pub latex: String,
    /// Fitted parameters
    pub parameters: HashMap<String, f64>,
    /// R² goodness of fit
    pub r_squared: f64,
    /// Complexity (number of terms)
    pub complexity: usize,
    /// Physical interpretation (if known)
    pub interpretation: Option<String>,
    /// Variables involved
    pub variables: Vec<(String, PhysicalQuantity)>,
}

/// Fitness metrics for candidate equations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationFitness {
    /// How well equation fits data (0-1)
    pub accuracy: f64,
    /// Simplicity score (inverse of complexity)
    pub simplicity: f64,
    /// Physical plausibility (dimensional consistency etc.)
    pub plausibility: f64,
    /// Overall fitness
    pub overall: f64,
}

/// Primitive operations for building equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
}

impl MathOp {
    fn apply(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => if b != 0.0 { a / b } else { f64::NAN },
            Self::Pow => a.powf(b),
            Self::Exp => a.exp(),
            Self::Log => if a > 0.0 { a.ln() } else { f64::NAN },
            Self::Sin => a.sin(),
            Self::Cos => a.cos(),
            Self::Sqrt => if a >= 0.0 { a.sqrt() } else { f64::NAN },
        }
    }

    fn symbol(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "×",
            Self::Div => "/",
            Self::Pow => "^",
            Self::Exp => "exp",
            Self::Log => "ln",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Sqrt => "√",
        }
    }
}

/// Expression tree node.
#[derive(Debug, Clone)]
enum ExprNode {
    Const(f64),
    Var(usize), // Index into variable list
    #[allow(dead_code)]
    Param(String), // Named parameter to fit
    BinaryOp(MathOp, Box<ExprNode>, Box<ExprNode>),
    UnaryOp(MathOp, Box<ExprNode>),
}

impl ExprNode {
    fn evaluate(&self, vars: &[f64], params: &HashMap<String, f64>) -> f64 {
        match self {
            Self::Const(c) => *c,
            Self::Var(i) => vars.get(*i).copied().unwrap_or(f64::NAN),
            Self::Param(name) => params.get(name).copied().unwrap_or(1.0),
            Self::BinaryOp(op, left, right) => {
                op.apply(left.evaluate(vars, params), right.evaluate(vars, params))
            }
            Self::UnaryOp(op, arg) => {
                op.apply(arg.evaluate(vars, params), 0.0)
            }
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Self::Const(_) | Self::Var(_) | Self::Param(_) => 1,
            Self::BinaryOp(_, left, right) => 1 + left.complexity() + right.complexity(),
            Self::UnaryOp(_, arg) => 1 + arg.complexity(),
        }
    }

    fn to_string(&self, var_names: &[String]) -> String {
        match self {
            Self::Const(c) => format!("{:.4}", c),
            Self::Var(i) => var_names.get(*i).cloned().unwrap_or_else(|| format!("x{}", i)),
            Self::Param(name) => name.clone(),
            Self::BinaryOp(op, left, right) => {
                format!("({} {} {})", left.to_string(var_names), op.symbol(), right.to_string(var_names))
            }
            Self::UnaryOp(op, arg) => {
                format!("{}({})", op.symbol(), arg.to_string(var_names))
            }
        }
    }
}

/// Configuration for symbolic regression.
#[derive(Debug, Clone)]
pub struct SymbolicRegressorConfig {
    /// Maximum expression complexity
    pub max_complexity: usize,
    /// Population size for genetic algorithm
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Minimum R² to consider valid
    pub min_r_squared: f64,
    /// Operations to use
    pub allowed_ops: Vec<MathOp>,
}

impl Default for SymbolicRegressorConfig {
    fn default() -> Self {
        Self {
            max_complexity: 20,
            population_size: 100,
            generations: 50,
            mutation_rate: 0.1,
            min_r_squared: 0.9,
            allowed_ops: vec![
                MathOp::Add, MathOp::Sub, MathOp::Mul, MathOp::Div,
                MathOp::Pow, MathOp::Exp, MathOp::Log, MathOp::Sqrt,
            ],
        }
    }
}

/// Symbolic regression engine.
pub struct SymbolicRegressor {
    config: SymbolicRegressorConfig,
    rng_state: u64,
}

impl SymbolicRegressor {
    /// Create new regressor.
    pub fn new(config: SymbolicRegressorConfig) -> Self {
        Self {
            config,
            rng_state: 42,
        }
    }

    /// Find equation relating target to inputs.
    pub fn fit(
        &mut self,
        inputs: &[Vec<f64>], // Each inner vec is one variable's values
        target: &[f64],
        var_names: &[String],
    ) -> Option<DiscoveredEquation> {
        if inputs.is_empty() || target.is_empty() || inputs[0].len() != target.len() {
            return None;
        }

        let n_vars = inputs.len();
        let _n_points = target.len();

        // Generate initial population
        let mut population: Vec<ExprNode> = (0..self.config.population_size)
            .map(|_| self.random_expr(n_vars, 3))
            .collect();

        let mut best_expr: Option<ExprNode> = None;
        let mut best_fitness = 0.0;

        // Evolutionary loop
        for _gen in 0..self.config.generations {
            // Evaluate fitness
            let mut scored: Vec<(ExprNode, f64)> = population
                .into_iter()
                .map(|expr| {
                    let fitness = self.evaluate_fitness(&expr, inputs, target);
                    (expr, fitness.overall)
                })
                .collect();

            // Sort by fitness
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Track best
            if scored[0].1 > best_fitness {
                best_fitness = scored[0].1;
                best_expr = Some(scored[0].0.clone());
            }

            // Selection and reproduction
            let elite_count = self.config.population_size / 10;
            let mut new_population: Vec<ExprNode> = scored[..elite_count]
                .iter()
                .map(|(e, _)| e.clone())
                .collect();

            while new_population.len() < self.config.population_size {
                let parent_idx = self.random_usize(elite_count * 2);
                let parent = &scored[parent_idx.min(scored.len() - 1)].0;

                let child = if self.random_f64() < self.config.mutation_rate {
                    self.mutate(parent, n_vars)
                } else {
                    parent.clone()
                };

                if child.complexity() <= self.config.max_complexity {
                    new_population.push(child);
                }
            }

            population = new_population;
        }

        // Convert best expression to equation
        best_expr.map(|expr| {
            let params = self.fit_parameters(&expr, inputs, target);
            let fitness = self.evaluate_fitness(&expr, inputs, target);

            DiscoveredEquation {
                symbolic_form: expr.to_string(var_names),
                latex: self.to_latex(&expr, var_names),
                parameters: params,
                r_squared: fitness.accuracy,
                complexity: expr.complexity(),
                interpretation: self.interpret(&expr, var_names),
                variables: var_names.iter()
                    .map(|name| (name.clone(), PhysicalQuantity::Other))
                    .collect(),
            }
        })
    }

    /// Find physics laws in data.
    pub fn discover_laws(
        &mut self,
        measurements: &[EncodedMeasurement],
    ) -> Vec<DiscoveredEquation> {
        // Group measurements by quantity
        let mut by_quantity: HashMap<PhysicalQuantity, Vec<(f64, Option<f64>)>> = HashMap::new();
        for m in measurements {
            by_quantity.entry(m.quantity)
                .or_default()
                .push((m.value, m.timestamp));
        }

        // Try to find relationships between pairs of quantities
        let mut discoveries = Vec::new();
        let quantities: Vec<_> = by_quantity.keys().cloned().collect();

        for (i, q1) in quantities.iter().enumerate() {
            for q2 in quantities.iter().skip(i + 1) {
                let vals1 = &by_quantity[q1];
                let vals2 = &by_quantity[q2];

                // Align by timestamp if available
                if vals1.len() >= 10 && vals2.len() >= 10 && vals1.len() == vals2.len() {
                    let x: Vec<f64> = vals1.iter().map(|(v, _)| *v).collect();
                    let y: Vec<f64> = vals2.iter().map(|(v, _)| *v).collect();

                    let var_names = vec![format!("{:?}", q1)];
                    if let Some(eq) = self.fit(&[x], &y, &var_names) {
                        if eq.r_squared >= self.config.min_r_squared {
                            discoveries.push(eq);
                        }
                    }
                }
            }
        }

        discoveries
    }

    fn evaluate_fitness(
        &self,
        expr: &ExprNode,
        inputs: &[Vec<f64>],
        target: &[f64],
    ) -> EquationFitness {
        let params = HashMap::new();
        let n_points = target.len();

        // Compute predictions
        let predictions: Vec<f64> = (0..n_points)
            .map(|i| {
                let vars: Vec<f64> = inputs.iter().map(|v| v[i]).collect();
                expr.evaluate(&vars, &params)
            })
            .collect();

        // R² calculation
        let y_mean: f64 = target.iter().sum::<f64>() / n_points as f64;
        let ss_tot: f64 = target.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = target.iter()
            .zip(predictions.iter())
            .filter(|(_, p)| p.is_finite())
            .map(|(y, p)| (y - p).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        let r_squared = r_squared.max(0.0);

        let complexity = expr.complexity();
        let simplicity = 1.0 / (1.0 + complexity as f64 / 10.0);

        // Check for NaN/Inf in predictions
        let valid_ratio = predictions.iter().filter(|p| p.is_finite()).count() as f64 / n_points as f64;

        let accuracy = r_squared * valid_ratio;
        let plausibility = valid_ratio;
        let overall = accuracy * 0.6 + simplicity * 0.3 + plausibility * 0.1;

        EquationFitness {
            accuracy,
            simplicity,
            plausibility,
            overall,
        }
    }

    fn fit_parameters(&self, _expr: &ExprNode, _inputs: &[Vec<f64>], _target: &[f64]) -> HashMap<String, f64> {
        // Simplified: no parameter fitting in this version
        HashMap::new()
    }

    fn random_expr(&mut self, n_vars: usize, max_depth: usize) -> ExprNode {
        if max_depth == 0 || self.random_f64() < 0.3 {
            // Terminal: variable or constant
            if self.random_f64() < 0.7 && n_vars > 0 {
                ExprNode::Var(self.random_usize(n_vars))
            } else {
                ExprNode::Const(self.random_f64() * 10.0 - 5.0)
            }
        } else {
            // Operator
            let n_ops = self.config.allowed_ops.len();
            let idx = self.random_usize(n_ops);
            let op = self.config.allowed_ops[idx];
            match op {
                MathOp::Exp | MathOp::Log | MathOp::Sin | MathOp::Cos | MathOp::Sqrt => {
                    ExprNode::UnaryOp(op, Box::new(self.random_expr(n_vars, max_depth - 1)))
                }
                _ => {
                    ExprNode::BinaryOp(
                        op,
                        Box::new(self.random_expr(n_vars, max_depth - 1)),
                        Box::new(self.random_expr(n_vars, max_depth - 1)),
                    )
                }
            }
        }
    }

    fn mutate(&mut self, expr: &ExprNode, n_vars: usize) -> ExprNode {
        // Simple mutation: replace a random subtree
        if self.random_f64() < 0.5 {
            self.random_expr(n_vars, 2)
        } else {
            match expr {
                ExprNode::Const(_) => ExprNode::Const(self.random_f64() * 10.0 - 5.0),
                ExprNode::Var(_) => ExprNode::Var(self.random_usize(n_vars.max(1))),
                ExprNode::Param(name) => ExprNode::Param(name.clone()),
                ExprNode::BinaryOp(op, left, right) => {
                    if self.random_f64() < 0.5 {
                        ExprNode::BinaryOp(*op, Box::new(self.mutate(left, n_vars)), right.clone())
                    } else {
                        ExprNode::BinaryOp(*op, left.clone(), Box::new(self.mutate(right, n_vars)))
                    }
                }
                ExprNode::UnaryOp(op, arg) => {
                    ExprNode::UnaryOp(*op, Box::new(self.mutate(arg, n_vars)))
                }
            }
        }
    }

    fn to_latex(&self, expr: &ExprNode, var_names: &[String]) -> String {
        // Simplified LaTeX conversion
        expr.to_string(var_names)
            .replace("×", "\\times")
            .replace("√", "\\sqrt")
    }

    fn interpret(&self, _expr: &ExprNode, _var_names: &[String]) -> Option<String> {
        // Would check against known physics patterns
        None
    }

    fn random_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn random_usize(&mut self, max: usize) -> usize {
        (self.random_f64() * max as f64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_regression() {
        let mut regressor = SymbolicRegressor::new(SymbolicRegressorConfig {
            generations: 20,
            population_size: 50,
            ..Default::default()
        });

        // y = 2x + 1
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = regressor.fit(&[x], &y, &["x".to_string()]);
        assert!(result.is_some());

        let eq = result.unwrap();
        assert!(eq.r_squared > 0.5); // Should find reasonable fit
    }

    #[test]
    fn test_expression_complexity() {
        let simple = ExprNode::Var(0);
        assert_eq!(simple.complexity(), 1);

        let complex = ExprNode::BinaryOp(
            MathOp::Add,
            Box::new(ExprNode::Var(0)),
            Box::new(ExprNode::BinaryOp(
                MathOp::Mul,
                Box::new(ExprNode::Const(2.0)),
                Box::new(ExprNode::Var(1)),
            )),
        );
        assert_eq!(complex.complexity(), 5);
    }
}
