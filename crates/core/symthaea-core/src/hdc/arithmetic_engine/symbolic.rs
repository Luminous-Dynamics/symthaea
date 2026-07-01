// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::deterministic_seeds::seed_from_name;
use crate::hdc::integrated_information::IntegratedInformation;
use crate::hdc::primitive_system::PrimitiveSystem;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core_number::HdcNumber;
use super::hybrid::{HybridArithmeticEngine, HybridResult};

/// Symbolic operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
}

impl std::fmt::Display for SymbolicOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicOp::Add => write!(f, "+"),
            SymbolicOp::Sub => write!(f, "-"),
            SymbolicOp::Mul => write!(f, "*"),
            SymbolicOp::Div => write!(f, "/"),
            SymbolicOp::Pow => write!(f, "^"),
            SymbolicOp::Neg => write!(f, "-"),
        }
    }
}

/// Term types in symbolic expressions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TermType {
    /// Numeric constant
    Constant(i64),
    /// Named variable
    Variable(String),
    /// Binary operation
    BinaryOp {
        op: SymbolicOp,
        left: Box<TermType>,
        right: Box<TermType>,
    },
    /// Unary operation (negation)
    UnaryOp {
        op: SymbolicOp,
        operand: Box<TermType>,
    },
    /// Named function application (sin, cos, exp, ln, tan, etc.)
    /// Enables transcendental function differentiation with chain rule.
    Function { name: String, arg: Box<TermType> },
}

/// A symbolic algebraic expression with HDC encoding
///
/// Expressions are represented both symbolically AND as HDC vectors.
/// This enables:
/// - Semantic similarity between equivalent expressions
/// - Consciousness-integrated simplification
/// - Pattern recognition across algebraic forms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicExpr {
    /// The symbolic structure of this expression
    pub term_type: TermType,
    /// HDC encoding capturing semantic meaning
    pub encoding: BinaryHV,
    /// Φ accumulated during construction
    pub phi: f64,
    /// Human-readable form
    pub display: String,
}

impl SymbolicExpr {
    /// Create a constant expression
    pub fn constant(value: i64, primitives: &PrimitiveSystem) -> Self {
        let constant_prim = primitives
            .get("CONSTANT")
            .or_else(|| primitives.get("ZERO"))
            .expect("Need CONSTANT or ZERO primitive");

        // Encode the constant value into HDC space
        let value_hv = if value >= 0 {
            HdcNumber::from_value(value as u64, primitives).encoding
        } else {
            // For negative, bind with negation primitive
            let abs_hv = HdcNumber::from_value((-value) as u64, primitives).encoding;
            if let Some(neg) = primitives.get("NEGATION") {
                neg.encoding.bind(&abs_hv)
            } else {
                abs_hv
            }
        };

        let encoding = constant_prim.encoding.bind(&value_hv);

        Self {
            term_type: TermType::Constant(value),
            encoding,
            phi: 0.1, // Small Φ for simple constants
            display: value.to_string(),
        }
    }

    /// Create a variable expression
    pub fn variable(name: &str, primitives: &PrimitiveSystem) -> Self {
        let var_prim = primitives
            .get("VARIABLE")
            .or_else(|| primitives.get("UNKNOWN"))
            .unwrap_or_else(|| primitives.get("ZERO").expect("Need ZERO primitive"));

        // Create unique encoding for this variable based on name
        let name_seed: u64 = name
            .bytes()
            .enumerate()
            .map(|(i, b)| (b as u64) << (i % 8))
            .fold(0, |acc, x| acc.wrapping_add(x));

        let var_hv = BinaryHV::random(name_seed);
        let encoding = var_prim.encoding.bind(&var_hv);

        Self {
            term_type: TermType::Variable(name.to_string()),
            encoding,
            phi: 0.2, // Variables have slightly more "meaning"
            display: name.to_string(),
        }
    }

    /// Add two expressions: self + other
    pub fn add(&self, other: &SymbolicExpr, primitives: &PrimitiveSystem) -> Self {
        let add_prim = primitives
            .get("ADDITION")
            .or_else(|| primitives.get("ADD"))
            .expect("ADDITION primitive required");

        // Combine encodings: ADD ⊗ (left ⊕ right)
        let combined = BinaryHV::bundle(&[self.encoding, other.encoding]);
        let encoding = add_prim.encoding.bind(&combined);

        let phi =
            self.phi + other.phi + Self::measure_composition_phi(&self.encoding, &other.encoding);

        Self {
            term_type: TermType::BinaryOp {
                op: SymbolicOp::Add,
                left: Box::new(self.term_type.clone()),
                right: Box::new(other.term_type.clone()),
            },
            encoding,
            phi,
            display: format!("({} + {})", self.display, other.display),
        }
    }

    /// Subtract: self - other
    pub fn sub(&self, other: &SymbolicExpr, primitives: &PrimitiveSystem) -> Self {
        let sub_prim = primitives
            .get("SUBTRACT")
            .or_else(|| primitives.get("ADD"))
            .expect("SUBTRACT or ADD primitive required");

        let combined = BinaryHV::bundle(&[self.encoding, other.encoding]);
        let encoding = sub_prim.encoding.bind(&combined);

        let phi =
            self.phi + other.phi + Self::measure_composition_phi(&self.encoding, &other.encoding);

        Self {
            term_type: TermType::BinaryOp {
                op: SymbolicOp::Sub,
                left: Box::new(self.term_type.clone()),
                right: Box::new(other.term_type.clone()),
            },
            encoding,
            phi,
            display: format!("({} - {})", self.display, other.display),
        }
    }

    /// Multiply: self * other
    pub fn mul(&self, other: &SymbolicExpr, primitives: &PrimitiveSystem) -> Self {
        let mul_prim = primitives
            .get("MULTIPLICATION")
            .or_else(|| primitives.get("MULTIPLY"))
            .expect("MULTIPLICATION primitive required");

        // For multiplication, use binding (not bundling) to capture relationship
        let encoding = mul_prim.encoding.bind(&self.encoding).bind(&other.encoding);

        let phi = self.phi
            + other.phi
            + Self::measure_composition_phi(&self.encoding, &other.encoding) * 1.5;

        Self {
            term_type: TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left: Box::new(self.term_type.clone()),
                right: Box::new(other.term_type.clone()),
            },
            encoding,
            phi,
            display: format!("({} * {})", self.display, other.display),
        }
    }

    /// Divide: self / other
    pub fn div(&self, other: &SymbolicExpr, primitives: &PrimitiveSystem) -> Self {
        let div_prim = primitives
            .get("DIVIDE")
            .or_else(|| primitives.get("MULTIPLY"))
            .expect("DIVIDE or MULTIPLY primitive required");

        let encoding = div_prim.encoding.bind(&self.encoding).bind(&other.encoding);

        let phi = self.phi
            + other.phi
            + Self::measure_composition_phi(&self.encoding, &other.encoding) * 1.3;

        Self {
            term_type: TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(self.term_type.clone()),
                right: Box::new(other.term_type.clone()),
            },
            encoding,
            phi,
            display: format!("({} / {})", self.display, other.display),
        }
    }

    /// Power: self ^ other
    pub fn pow(&self, other: &SymbolicExpr, primitives: &PrimitiveSystem) -> Self {
        let pow_prim = primitives
            .get("POWER")
            .or_else(|| primitives.get("MULTIPLY"))
            .or_else(|| primitives.get("MULTIPLICATION"))
            .expect("POWER, MULTIPLY, or MULTIPLICATION primitive required");

        let encoding = pow_prim.encoding.bind(&self.encoding).bind(&other.encoding);

        let phi = self.phi
            + other.phi
            + Self::measure_composition_phi(&self.encoding, &other.encoding) * 2.0;

        Self {
            term_type: TermType::BinaryOp {
                op: SymbolicOp::Pow,
                left: Box::new(self.term_type.clone()),
                right: Box::new(other.term_type.clone()),
            },
            encoding,
            phi,
            display: format!("({} ^ {})", self.display, other.display),
        }
    }

    /// Negate: -self
    pub fn neg(&self, primitives: &PrimitiveSystem) -> Self {
        let neg_prim = primitives
            .get("NEGATION")
            .or_else(|| primitives.get("SUBTRACT"))
            .expect("NEGATION primitive required");

        let encoding = neg_prim.encoding.bind(&self.encoding);

        Self {
            term_type: TermType::UnaryOp {
                op: SymbolicOp::Neg,
                operand: Box::new(self.term_type.clone()),
            },
            encoding,
            phi: self.phi + 0.1,
            display: format!("(-{})", self.display),
        }
    }

    /// Measure Φ contribution of composing two expressions
    fn measure_composition_phi(a: &BinaryHV, b: &BinaryHV) -> f64 {
        let mut phi_calc = IntegratedInformation::new();
        let components = vec![*a, *b];
        phi_calc.compute_phi(&components)
    }

    /// Evaluate expression with variable bindings
    pub fn evaluate(
        &self,
        bindings: &HashMap<String, i64>,
        engine: &mut HybridArithmeticEngine,
    ) -> Option<HybridResult> {
        match &self.term_type {
            TermType::Constant(v) => {
                if *v >= 0 {
                    Some(engine.add(*v as u64, 0))
                } else {
                    // Negative constant - represent as 0 - |v|
                    let abs_result = engine.add((-*v) as u64, 0);
                    Some(HybridResult {
                        value: *v as u64, // Will wrap for negative, but track in semantics
                        ..abs_result
                    })
                }
            }
            TermType::Variable(name) => bindings.get(name).map(|&v| {
                if v >= 0 {
                    engine.add(v as u64, 0)
                } else {
                    engine.add((-v) as u64, 0)
                }
            }),
            TermType::BinaryOp { op, left, right } => {
                let left_expr = SymbolicExpr::from_term_type(*left.clone(), &self.encoding);
                let right_expr = SymbolicExpr::from_term_type(*right.clone(), &self.encoding);

                let l_val = left_expr.evaluate(bindings, engine)?;
                let r_val = right_expr.evaluate(bindings, engine)?;

                match op {
                    SymbolicOp::Add => Some(engine.add(l_val.value, r_val.value)),
                    SymbolicOp::Sub => engine.subtract(l_val.value, r_val.value),
                    SymbolicOp::Mul => Some(engine.multiply(l_val.value, r_val.value)),
                    SymbolicOp::Div => engine.divide(l_val.value, r_val.value),
                    SymbolicOp::Pow => Some(engine.power(l_val.value, r_val.value)),
                    SymbolicOp::Neg => engine.subtract(0, l_val.value),
                }
            }
            TermType::UnaryOp { op, operand } => {
                let operand_expr = SymbolicExpr::from_term_type(*operand.clone(), &self.encoding);
                let val = operand_expr.evaluate(bindings, engine)?;

                match op {
                    SymbolicOp::Neg => engine.subtract(0, val.value),
                    _ => Some(val),
                }
            }
            TermType::Function { name: _, arg } => {
                // Functions evaluate to approximate integer values
                let arg_expr = SymbolicExpr::from_term_type(*arg.clone(), &self.encoding);
                let _val = arg_expr.evaluate(bindings, engine)?;
                // Cannot exactly evaluate transcendental functions in integer domain
                None
            }
        }
    }

    /// Create SymbolicExpr from TermType (helper for evaluation)
    fn from_term_type(term: TermType, encoding: &BinaryHV) -> Self {
        let display = match &term {
            TermType::Constant(v) => v.to_string(),
            TermType::Variable(n) => n.clone(),
            TermType::BinaryOp { op, left, right } => {
                let l = Self::from_term_type(*left.clone(), encoding);
                let r = Self::from_term_type(*right.clone(), encoding);
                format!("({} {} {})", l.display, op, r.display)
            }
            TermType::UnaryOp { op, operand } => {
                let o = Self::from_term_type(*operand.clone(), encoding);
                format!("({}{})", op, o.display)
            }
            TermType::Function { name, arg } => {
                let a = Self::from_term_type(*arg.clone(), encoding);
                format!("{}({})", name, a.display)
            }
        };

        Self {
            term_type: term,
            encoding: *encoding,
            phi: 0.0,
            display,
        }
    }

    /// Get variables used in this expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&self.term_type, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, term: &TermType, vars: &mut Vec<String>) {
        match term {
            TermType::Constant(_) => {}
            TermType::Variable(name) => vars.push(name.clone()),
            TermType::BinaryOp { left, right, .. } => {
                self.collect_variables(left, vars);
                self.collect_variables(right, vars);
            }
            TermType::UnaryOp { operand, .. } => {
                self.collect_variables(operand, vars);
            }
            TermType::Function { arg, .. } => {
                self.collect_variables(arg, vars);
            }
        }
    }

    /// Check if this expression is semantically similar to another
    pub fn similar_to(&self, other: &SymbolicExpr) -> f32 {
        self.encoding.similarity(&other.encoding)
    }
}

impl std::fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// POLYNOMIAL REPRESENTATION: Single-variable polynomials: evaluation, derivatives, display
// ═════════════════════════════════════════════════════════════════════════════

/// A polynomial in a single variable
///
/// Coefficients are stored in ascending order of degree:
/// [a₀, a₁, a₂, ...] represents a₀ + a₁x + a₂x² + ...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients in ascending order
    coefficients: Vec<i64>,
    /// Variable name
    variable: String,
    /// HDC encoding of the polynomial
    encoding: BinaryHV,
    /// Accumulated Φ
    phi: f64,
}

impl Polynomial {
    /// Create a polynomial from coefficients
    pub fn new(coefficients: Vec<i64>, variable: &str, primitives: &PrimitiveSystem) -> Self {
        // Remove trailing zeros
        let mut coeffs = coefficients;
        while coeffs.len() > 1 && coeffs.last() == Some(&0) {
            coeffs.pop();
        }

        // Build HDC encoding
        let poly_prim = primitives
            .get("POLYNOMIAL")
            .or_else(|| primitives.get("ADDITION"))
            .or_else(|| primitives.get("ADD"))
            .expect("Need POLYNOMIAL or ADDITION primitive");

        let var_encoding = SymbolicExpr::variable(variable, primitives).encoding;

        // Encode each term and bundle
        let term_encodings: Vec<BinaryHV> = coeffs
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c != 0)
            .map(|(degree, &coeff)| {
                let coeff_enc = HdcNumber::from_value(coeff.unsigned_abs(), primitives).encoding;
                let degree_enc = HdcNumber::from_value(degree as u64, primitives).encoding;
                // term = coeff * x^degree
                coeff_enc.bind(&var_encoding).bind(&degree_enc)
            })
            .collect();

        let encoding = if term_encodings.is_empty() {
            poly_prim.encoding
        } else {
            poly_prim.encoding.bind(&BinaryHV::bundle(&term_encodings))
        };

        let phi = coeffs.len() as f64 * 0.1;

        Self {
            coefficients: coeffs,
            variable: variable.to_string(),
            encoding,
            phi,
        }
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            return 0;
        }
        // Find the highest non-zero coefficient
        for i in (0..self.coefficients.len()).rev() {
            if self.coefficients[i] != 0 {
                return i;
            }
        }
        0 // All-zero polynomial has degree 0
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[i64] {
        &self.coefficients
    }

    /// Evaluate polynomial at a point
    pub fn evaluate(&self, x: i64, engine: &mut HybridArithmeticEngine) -> HybridResult {
        // Horner's method: a₀ + x(a₁ + x(a₂ + ...))
        let mut result = 0_u64;
        let x_abs = x.unsigned_abs();

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            // term = coeff * x^i
            let x_pow = engine.power(x_abs, i as u64);
            let coeff_abs = coeff.unsigned_abs();
            let term = engine.multiply(coeff_abs, x_pow.value);

            if coeff >= 0 {
                result = engine.add(result, term.value).value;
            } else if let Some(sub) = engine.subtract(result, term.value) {
                result = sub.value;
            }
        }

        engine.add(result, 0)
    }

    /// Get the leading coefficient
    pub fn leading_coefficient(&self) -> i64 {
        *self.coefficients.last().unwrap_or(&0)
    }

    /// Check if polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|&c| c == 0)
    }

    /// Compute the formal derivative d/dx of this polynomial.
    ///
    /// Coefficients `[a₀, a₁, a₂, a₃, ...]` become `[a₁, 2a₂, 3a₃, ...]`.
    pub fn derivative(&self, primitives: &PrimitiveSystem) -> Self {
        if self.coefficients.len() <= 1 {
            return Self::new(vec![0], &self.variable, primitives);
        }
        let deriv_coeffs: Vec<i64> = self
            .coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * i as i64)
            .collect();
        Self::new(deriv_coeffs, &self.variable, primitives)
    }

    /// Convert to symbolic expression
    pub fn to_expr(&self, primitives: &PrimitiveSystem) -> SymbolicExpr {
        if self.coefficients.is_empty() {
            return SymbolicExpr::constant(0, primitives);
        }

        let x = SymbolicExpr::variable(&self.variable, primitives);
        let mut result: Option<SymbolicExpr> = None;

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            if coeff == 0 {
                continue;
            }

            let coeff_expr = SymbolicExpr::constant(coeff, primitives);
            let term = if i == 0 {
                coeff_expr
            } else if i == 1 {
                coeff_expr.mul(&x, primitives)
            } else {
                let power = SymbolicExpr::constant(i as i64, primitives);
                let x_pow = x.pow(&power, primitives);
                coeff_expr.mul(&x_pow, primitives)
            };

            result = Some(match result {
                None => term,
                Some(r) => r.add(&term, primitives),
            });
        }

        result.unwrap_or_else(|| SymbolicExpr::constant(0, primitives))
    }
}

impl std::fmt::Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms = Vec::new();
        for (i, &coeff) in self.coefficients.iter().enumerate().rev() {
            if coeff == 0 {
                continue;
            }
            let term = if i == 0 {
                format!("{coeff}")
            } else if i == 1 {
                if coeff == 1 {
                    self.variable.clone()
                } else if coeff == -1 {
                    format!("-{}", self.variable)
                } else {
                    format!("{}{}", coeff, self.variable)
                }
            } else if coeff == 1 {
                format!("{}^{}", self.variable, i)
            } else if coeff == -1 {
                format!("-{}^{}", self.variable, i)
            } else {
                format!("{}{}^{}", coeff, self.variable, i)
            };
            terms.push(term);
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + ").replace(" + -", " - "))
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SYMBOLIC ALGEBRA ENGINE: Simplification, expansion, polynomial arithmetic, equation solving
// ═════════════════════════════════════════════════════════════════════════════

/// Statistics for symbolic algebra operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolicAlgebraStats {
    pub simplifications: usize,
    pub expansions: usize,
    pub factorizations: usize,
    pub equations_solved: usize,
    pub total_phi: f64,
}

/// Symbolic algebra engine with Φ-guided operations
pub struct SymbolicAlgebra {
    /// Cache of simplified expressions
    simplify_cache: HashMap<String, SymbolicExpr>,
    /// Statistics
    stats: SymbolicAlgebraStats,
}

impl SymbolicAlgebra {
    /// Create new algebra engine
    pub fn new() -> Self {
        Self {
            simplify_cache: HashMap::new(),
            stats: SymbolicAlgebraStats::default(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SymbolicAlgebraStats {
        &self.stats
    }

    /// Simplify an expression
    ///
    /// Applies algebraic rules:
    /// - Constant folding (3 + 4 → 7)
    /// - Identity elimination (x + 0 → x, x * 1 → x)
    /// - Zero multiplication (x * 0 → 0)
    /// - Like term combination (2x + 3x → 5x)
    pub fn simplify(&mut self, expr: &SymbolicExpr, primitives: &PrimitiveSystem) -> SymbolicExpr {
        // Check cache
        if let Some(cached) = self.simplify_cache.get(&expr.display) {
            return cached.clone();
        }

        self.stats.simplifications += 1;

        let result = self.simplify_recursive(&expr.term_type, primitives);
        self.stats.total_phi += result.phi;

        // Cache result
        self.simplify_cache
            .insert(expr.display.clone(), result.clone());

        result
    }

    fn simplify_recursive(&self, term: &TermType, primitives: &PrimitiveSystem) -> SymbolicExpr {
        match term {
            TermType::Constant(v) => SymbolicExpr::constant(*v, primitives),
            TermType::Variable(n) => SymbolicExpr::variable(n, primitives),

            TermType::BinaryOp { op, left, right } => {
                // First simplify children
                let l = self.simplify_recursive(left, primitives);
                let r = self.simplify_recursive(right, primitives);

                // Then apply rules
                match op {
                    SymbolicOp::Add => self.simplify_add(&l, &r, primitives),
                    SymbolicOp::Sub => self.simplify_sub(&l, &r, primitives),
                    SymbolicOp::Mul => self.simplify_mul(&l, &r, primitives),
                    SymbolicOp::Div => self.simplify_div(&l, &r, primitives),
                    SymbolicOp::Pow => self.simplify_pow(&l, &r, primitives),
                    SymbolicOp::Neg => l.neg(primitives),
                }
            }

            TermType::UnaryOp { op, operand } => {
                let inner = self.simplify_recursive(operand, primitives);
                match op {
                    SymbolicOp::Neg => {
                        // Simplify double negation
                        if let TermType::UnaryOp {
                            op: SymbolicOp::Neg,
                            operand: inner_operand,
                        } = &inner.term_type
                        {
                            return self.simplify_recursive(inner_operand, primitives);
                        }
                        // Simplify negation of constant
                        if let TermType::Constant(v) = inner.term_type {
                            return SymbolicExpr::constant(-v, primitives);
                        }
                        inner.neg(primitives)
                    }
                    _ => inner,
                }
            }

            TermType::Function { name, arg } => {
                let simplified_arg = self.simplify_recursive(arg, primitives);
                // Build a SymbolicExpr for the function with simplified argument
                let encoding = BinaryHV::random(seed_from_name(&format!("FN_{name}")));
                let display = format!("{}({})", name, simplified_arg.display);
                SymbolicExpr {
                    term_type: TermType::Function {
                        name: name.clone(),
                        arg: Box::new(simplified_arg.term_type),
                    },
                    encoding,
                    phi: simplified_arg.phi + 0.1,
                    display,
                }
            }
        }
    }

    fn simplify_add(
        &self,
        left: &SymbolicExpr,
        right: &SymbolicExpr,
        primitives: &PrimitiveSystem,
    ) -> SymbolicExpr {
        // Constant folding
        if let (TermType::Constant(l), TermType::Constant(r)) = (&left.term_type, &right.term_type)
        {
            return SymbolicExpr::constant(l + r, primitives);
        }

        // x + 0 = x
        if let TermType::Constant(0) = right.term_type {
            return left.clone();
        }
        if let TermType::Constant(0) = left.term_type {
            return right.clone();
        }

        // x + x = 2x (like terms)
        if left.term_type == right.term_type {
            let two = SymbolicExpr::constant(2, primitives);
            return two.mul(left, primitives);
        }

        left.add(right, primitives)
    }

    fn simplify_sub(
        &self,
        left: &SymbolicExpr,
        right: &SymbolicExpr,
        primitives: &PrimitiveSystem,
    ) -> SymbolicExpr {
        // Constant folding
        if let (TermType::Constant(l), TermType::Constant(r)) = (&left.term_type, &right.term_type)
        {
            return SymbolicExpr::constant(l - r, primitives);
        }

        // x - 0 = x
        if let TermType::Constant(0) = right.term_type {
            return left.clone();
        }

        // x - x = 0
        if left.term_type == right.term_type {
            return SymbolicExpr::constant(0, primitives);
        }

        left.sub(right, primitives)
    }

    fn simplify_mul(
        &self,
        left: &SymbolicExpr,
        right: &SymbolicExpr,
        primitives: &PrimitiveSystem,
    ) -> SymbolicExpr {
        // Constant folding
        if let (TermType::Constant(l), TermType::Constant(r)) = (&left.term_type, &right.term_type)
        {
            return SymbolicExpr::constant(l * r, primitives);
        }

        // x * 0 = 0
        if let TermType::Constant(0) = right.term_type {
            return SymbolicExpr::constant(0, primitives);
        }
        if let TermType::Constant(0) = left.term_type {
            return SymbolicExpr::constant(0, primitives);
        }

        // x * 1 = x
        if let TermType::Constant(1) = right.term_type {
            return left.clone();
        }
        if let TermType::Constant(1) = left.term_type {
            return right.clone();
        }

        // x * -1 = -x
        if let TermType::Constant(-1) = right.term_type {
            return left.neg(primitives);
        }
        if let TermType::Constant(-1) = left.term_type {
            return right.neg(primitives);
        }

        left.mul(right, primitives)
    }

    fn simplify_div(
        &self,
        left: &SymbolicExpr,
        right: &SymbolicExpr,
        primitives: &PrimitiveSystem,
    ) -> SymbolicExpr {
        // x / 1 = x
        if let TermType::Constant(1) = right.term_type {
            return left.clone();
        }

        // 0 / x = 0 (assuming x ≠ 0)
        if let TermType::Constant(0) = left.term_type {
            return SymbolicExpr::constant(0, primitives);
        }

        // x / x = 1 (assuming x ≠ 0)
        if left.term_type == right.term_type {
            return SymbolicExpr::constant(1, primitives);
        }

        // Constant folding for exact division
        if let (TermType::Constant(l), TermType::Constant(r)) = (&left.term_type, &right.term_type)
        {
            if *r != 0 && l % r == 0 {
                return SymbolicExpr::constant(l / r, primitives);
            }
        }

        left.div(right, primitives)
    }

    fn simplify_pow(
        &self,
        base: &SymbolicExpr,
        exp: &SymbolicExpr,
        primitives: &PrimitiveSystem,
    ) -> SymbolicExpr {
        // x^0 = 1
        if let TermType::Constant(0) = exp.term_type {
            return SymbolicExpr::constant(1, primitives);
        }

        // x^1 = x
        if let TermType::Constant(1) = exp.term_type {
            return base.clone();
        }

        // 0^n = 0 (for n > 0)
        if let (TermType::Constant(0), TermType::Constant(n)) = (&base.term_type, &exp.term_type) {
            if *n > 0 {
                return SymbolicExpr::constant(0, primitives);
            }
        }

        // 1^n = 1
        if let TermType::Constant(1) = base.term_type {
            return SymbolicExpr::constant(1, primitives);
        }

        // Constant folding
        if let (TermType::Constant(b), TermType::Constant(e)) = (&base.term_type, &exp.term_type) {
            if *e >= 0 && *e < 20 {
                // Reasonable exponent
                let result = (*b).pow(*e as u32);
                return SymbolicExpr::constant(result, primitives);
            }
        }

        base.pow(exp, primitives)
    }

    /// Expand an expression (distribute multiplication)
    ///
    /// a * (b + c) → a*b + a*c
    pub fn expand(&mut self, expr: &SymbolicExpr, primitives: &PrimitiveSystem) -> SymbolicExpr {
        self.stats.expansions += 1;
        let result = self.expand_recursive(&expr.term_type, primitives);
        self.stats.total_phi += result.phi;
        result
    }

    fn expand_recursive(&self, term: &TermType, primitives: &PrimitiveSystem) -> SymbolicExpr {
        match term {
            TermType::Constant(v) => SymbolicExpr::constant(*v, primitives),
            TermType::Variable(n) => SymbolicExpr::variable(n, primitives),

            TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left,
                right,
            } => {
                let l = self.expand_recursive(left, primitives);
                let r = self.expand_recursive(right, primitives);

                // Distribute: a * (b + c) = a*b + a*c
                if let TermType::BinaryOp {
                    op: SymbolicOp::Add,
                    left: b,
                    right: c,
                } = &r.term_type
                {
                    let b_expr = self.expand_recursive(b, primitives);
                    let c_expr = self.expand_recursive(c, primitives);
                    let ab = l.mul(&b_expr, primitives);
                    let ac = l.mul(&c_expr, primitives);
                    return ab.add(&ac, primitives);
                }

                // Distribute: (a + b) * c = a*c + b*c
                if let TermType::BinaryOp {
                    op: SymbolicOp::Add,
                    left: a,
                    right: b,
                } = &l.term_type
                {
                    let a_expr = self.expand_recursive(a, primitives);
                    let b_expr = self.expand_recursive(b, primitives);
                    let ac = a_expr.mul(&r, primitives);
                    let bc = b_expr.mul(&r, primitives);
                    return ac.add(&bc, primitives);
                }

                l.mul(&r, primitives)
            }

            TermType::BinaryOp { op, left, right } => {
                let l = self.expand_recursive(left, primitives);
                let r = self.expand_recursive(right, primitives);

                match op {
                    SymbolicOp::Add => l.add(&r, primitives),
                    SymbolicOp::Sub => l.sub(&r, primitives),
                    SymbolicOp::Div => l.div(&r, primitives),
                    SymbolicOp::Pow => l.pow(&r, primitives),
                    _ => l.mul(&r, primitives),
                }
            }

            TermType::UnaryOp {
                op: SymbolicOp::Neg,
                operand,
            } => {
                let inner = self.expand_recursive(operand, primitives);
                inner.neg(primitives)
            }

            TermType::UnaryOp { operand, .. } => self.expand_recursive(operand, primitives),

            TermType::Function { name, arg } => {
                let expanded_arg = self.expand_recursive(arg, primitives);
                let encoding = BinaryHV::random(seed_from_name(&format!("FN_{name}")));
                let display = format!("{}({})", name, expanded_arg.display);
                SymbolicExpr {
                    term_type: TermType::Function {
                        name: name.clone(),
                        arg: Box::new(expanded_arg.term_type),
                    },
                    encoding,
                    display,
                    phi: expanded_arg.phi,
                }
            }
        }
    }

    // ========================================================================
    // POLYNOMIAL OPERATIONS
    // ========================================================================

    /// Add two polynomials
    pub fn poly_add(
        &self,
        p1: &Polynomial,
        p2: &Polynomial,
        primitives: &PrimitiveSystem,
    ) -> Polynomial {
        let max_len = p1.coefficients.len().max(p2.coefficients.len());
        let mut coeffs = vec![0; max_len];

        for (i, &c) in p1.coefficients.iter().enumerate() {
            coeffs[i] += c;
        }
        for (i, &c) in p2.coefficients.iter().enumerate() {
            coeffs[i] += c;
        }

        Polynomial::new(coeffs, &p1.variable, primitives)
    }

    /// Subtract polynomials: p1 - p2
    pub fn poly_subtract(
        &self,
        p1: &Polynomial,
        p2: &Polynomial,
        primitives: &PrimitiveSystem,
    ) -> Polynomial {
        let max_len = p1.coefficients.len().max(p2.coefficients.len());
        let mut coeffs = vec![0; max_len];

        for (i, &c) in p1.coefficients.iter().enumerate() {
            coeffs[i] += c;
        }
        for (i, &c) in p2.coefficients.iter().enumerate() {
            coeffs[i] -= c;
        }

        Polynomial::new(coeffs, &p1.variable, primitives)
    }

    /// Multiply polynomials
    pub fn poly_multiply(
        &self,
        p1: &Polynomial,
        p2: &Polynomial,
        primitives: &PrimitiveSystem,
    ) -> Polynomial {
        if p1.is_zero() || p2.is_zero() {
            return Polynomial::new(vec![0], &p1.variable, primitives);
        }

        let result_len = p1.coefficients.len() + p2.coefficients.len() - 1;
        let mut coeffs = vec![0; result_len];

        for (i, &c1) in p1.coefficients.iter().enumerate() {
            for (j, &c2) in p2.coefficients.iter().enumerate() {
                coeffs[i + j] += c1 * c2;
            }
        }

        Polynomial::new(coeffs, &p1.variable, primitives)
    }

    /// Scale polynomial by constant
    pub fn poly_scale(&self, p: &Polynomial, c: i64, primitives: &PrimitiveSystem) -> Polynomial {
        let coeffs: Vec<i64> = p.coefficients.iter().map(|&x| x * c).collect();
        Polynomial::new(coeffs, &p.variable, primitives)
    }

    // ========================================================================
    // EQUATION SOLVING
    // ========================================================================

    /// Solve linear equation ax + b = 0
    /// Returns x = -b/a if a ≠ 0
    pub fn solve_linear(&mut self, p: &Polynomial) -> Option<i64> {
        if p.degree() != 1 {
            return None;
        }

        self.stats.equations_solved += 1;

        let a = p.coefficients.get(1).copied().unwrap_or(0);
        let b = p.coefficients.first().copied().unwrap_or(0);

        if a == 0 {
            return None;
        }

        // x = -b/a
        if b % a == 0 {
            Some(-b / a)
        } else {
            None // No integer solution
        }
    }

    /// Solve quadratic equation ax² + bx + c = 0
    /// Returns real solutions using quadratic formula
    pub fn solve_quadratic(&mut self, p: &Polynomial) -> Vec<f64> {
        if p.degree() != 2 {
            return Vec::new();
        }

        self.stats.equations_solved += 1;

        let a = p.coefficients.get(2).copied().unwrap_or(0) as f64;
        let b = p.coefficients.get(1).copied().unwrap_or(0) as f64;
        let c = p.coefficients.first().copied().unwrap_or(0) as f64;

        if a.abs() < 1e-10 {
            return Vec::new();
        }

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            Vec::new() // No real solutions
        } else if discriminant.abs() < 1e-10 {
            vec![-b / (2.0 * a)] // One solution
        } else {
            let sqrt_d = discriminant.sqrt();
            vec![(-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)]
        }
    }

    /// Find integer roots of a polynomial using rational root theorem
    pub fn find_integer_roots(
        &mut self,
        p: &Polynomial,
        engine: &mut HybridArithmeticEngine,
    ) -> Vec<i64> {
        if p.is_zero() {
            return Vec::new();
        }

        self.stats.equations_solved += 1;

        let constant = p.coefficients.first().copied().unwrap_or(0);
        let leading = p.leading_coefficient();

        if constant == 0 {
            // x = 0 is a root
            let roots = vec![0];
            // Factor out x and find remaining roots
            // (simplified - just return 0 for now)
            return roots;
        }

        let mut roots = Vec::new();

        // Test all divisors of constant term
        let abs_const = constant.unsigned_abs();
        let _abs_lead = leading.unsigned_abs();

        for d in 1..=abs_const {
            if abs_const % d == 0 {
                for &sign in &[1_i64, -1_i64] {
                    let candidate = sign * (d as i64);
                    let result = p.evaluate(candidate, engine);
                    if result.value == 0 {
                        roots.push(candidate);
                    }
                }
            }
        }

        roots.sort();
        roots.dedup();
        roots
    }
}

impl Default for SymbolicAlgebra {
    fn default() -> Self {
        Self::new()
    }
}
