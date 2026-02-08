#![allow(dead_code)]

//! Symbolic calculus: differentiation, integration, and fundamental theorem verification.

use crate::hdc::arithmetic_engine::{HybridArithmeticEngine, Polynomial, SymbolicExpr, SymbolicOp, TermType};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};

/// Result of a calculus operation with HDC encoding
pub struct CalculusResult {
    pub expr: SymbolicExpr,
    pub description: String,
    pub encoding: BinaryHV,
    pub phi: f64,
}

/// Symbolic differentiation engine
pub struct SymbolicDifferentiator;

impl SymbolicDifferentiator {
    /// Differentiate an expression with respect to a variable
    pub fn differentiate(expr: &SymbolicExpr, variable: &str, primitives: &PrimitiveSystem) -> CalculusResult {
        let diff_expr = Self::diff_recursive(&expr.term_type, variable, primitives);

        let diff_prim = primitives.get("DIFFERENTIATION")
            .expect("DIFFERENTIATION primitive not found");
        let var_encoding = BinaryHV::random(seed_from_name(variable));
        let encoding = diff_prim.encoding.bind(&expr.encoding).bind(&var_encoding);

        let result_expr = Self::term_to_expr(&diff_expr, primitives);
        let description = format!("d/d{} ({})", variable, expr.display);

        CalculusResult {
            expr: result_expr,
            description,
            encoding,
            phi: 0.0,
        }
    }

    /// Apply differentiation n times
    pub fn differentiate_n(expr: &SymbolicExpr, variable: &str, n: usize, primitives: &PrimitiveSystem) -> CalculusResult {
        let mut current_term = expr.term_type.clone();
        for _ in 0..n {
            current_term = Self::diff_recursive(&current_term, variable, primitives);
        }

        let diff_prim = primitives.get("DIFFERENTIATION")
            .expect("DIFFERENTIATION primitive not found");
        let var_encoding = BinaryHV::random(seed_from_name(variable));
        let mut encoding = expr.encoding;
        for _ in 0..n {
            encoding = diff_prim.encoding.bind(&encoding).bind(&var_encoding);
        }

        let result_expr = Self::term_to_expr(&current_term, primitives);
        let description = format!("d^{}/d{}^{} ({})", n, variable, n, expr.display);

        CalculusResult {
            expr: result_expr,
            description,
            encoding,
            phi: 0.0,
        }
    }

    /// Recursive differentiation on TermType trees
    fn diff_recursive(term: &TermType, variable: &str, _primitives: &PrimitiveSystem) -> TermType {
        match term {
            TermType::Constant(_) => TermType::Constant(0),

            TermType::Variable(v) => {
                if v == variable {
                    TermType::Constant(1)
                } else {
                    TermType::Constant(0)
                }
            }

            TermType::BinaryOp { op, left, right } => {
                match op {
                    SymbolicOp::Add => {
                        let dl = Self::diff_recursive(left, variable, _primitives);
                        let dr = Self::diff_recursive(right, variable, _primitives);
                        TermType::BinaryOp {
                            op: SymbolicOp::Add,
                            left: Box::new(dl),
                            right: Box::new(dr),
                        }
                    }
                    SymbolicOp::Sub => {
                        let dl = Self::diff_recursive(left, variable, _primitives);
                        let dr = Self::diff_recursive(right, variable, _primitives);
                        TermType::BinaryOp {
                            op: SymbolicOp::Sub,
                            left: Box::new(dl),
                            right: Box::new(dr),
                        }
                    }
                    SymbolicOp::Mul => {
                        // Product rule: d(f*g) = f'*g + f*g'
                        let df = Self::diff_recursive(left, variable, _primitives);
                        let dg = Self::diff_recursive(right, variable, _primitives);
                        TermType::BinaryOp {
                            op: SymbolicOp::Add,
                            left: Box::new(TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: Box::new(df),
                                right: right.clone(),
                            }),
                            right: Box::new(TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: left.clone(),
                                right: Box::new(dg),
                            }),
                        }
                    }
                    SymbolicOp::Div => {
                        // Quotient rule: d(f/g) = (f'g - fg') / g^2
                        let df = Self::diff_recursive(left, variable, _primitives);
                        let dg = Self::diff_recursive(right, variable, _primitives);
                        let numerator = TermType::BinaryOp {
                            op: SymbolicOp::Sub,
                            left: Box::new(TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: Box::new(df),
                                right: right.clone(),
                            }),
                            right: Box::new(TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: left.clone(),
                                right: Box::new(dg),
                            }),
                        };
                        let denominator = TermType::BinaryOp {
                            op: SymbolicOp::Mul,
                            left: right.clone(),
                            right: right.clone(),
                        };
                        TermType::BinaryOp {
                            op: SymbolicOp::Div,
                            left: Box::new(numerator),
                            right: Box::new(denominator),
                        }
                    }
                    SymbolicOp::Pow => {
                        // Power rule for constant exponent: d(f^n) = n*f^(n-1)*f'
                        if let TermType::Constant(n) = right.as_ref() {
                            let n_val = *n;
                            let df = Self::diff_recursive(left, variable, _primitives);
                            // n * f^(n-1) * f'
                            TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: Box::new(TermType::BinaryOp {
                                    op: SymbolicOp::Mul,
                                    left: Box::new(TermType::Constant(n_val)),
                                    right: Box::new(TermType::BinaryOp {
                                        op: SymbolicOp::Pow,
                                        left: left.clone(),
                                        right: Box::new(TermType::Constant(n_val - 1)),
                                    }),
                                }),
                                right: Box::new(df),
                            }
                        } else {
                            // General case not supported
                            TermType::Constant(0)
                        }
                    }
                    SymbolicOp::Neg => {
                        // Shouldn't appear as binary op
                        TermType::Constant(0)
                    }
                }
            }

            TermType::UnaryOp { op, operand } => {
                match op {
                    SymbolicOp::Neg => {
                        let d = Self::diff_recursive(operand, variable, _primitives);
                        TermType::UnaryOp {
                            op: SymbolicOp::Neg,
                            operand: Box::new(d),
                        }
                    }
                    _ => TermType::Constant(0),
                }
            }
        }
    }

    /// Convert a TermType back to a SymbolicExpr with proper HDC encoding
    fn term_to_expr(term: &TermType, primitives: &PrimitiveSystem) -> SymbolicExpr {
        match term {
            TermType::Constant(c) => SymbolicExpr::constant(*c, primitives),
            TermType::Variable(v) => SymbolicExpr::variable(v, primitives),
            TermType::BinaryOp { op, left, right } => {
                let l = Self::term_to_expr(left, primitives);
                let r = Self::term_to_expr(right, primitives);
                match op {
                    SymbolicOp::Add => l.add(&r, primitives),
                    SymbolicOp::Sub => l.sub(&r, primitives),
                    SymbolicOp::Mul => l.mul(&r, primitives),
                    SymbolicOp::Div => l.div(&r, primitives),
                    SymbolicOp::Pow => l.pow(&r, primitives),
                    SymbolicOp::Neg => l, // shouldn't happen for binary
                }
            }
            TermType::UnaryOp { op: _, operand } => {
                let inner = Self::term_to_expr(operand, primitives);
                inner.neg(primitives)
            }
        }
    }
}

/// Symbolic integration engine
pub struct SymbolicIntegrator;

impl SymbolicIntegrator {
    /// Integrate a polynomial using the power rule.
    /// Returns a new polynomial with coefficients shifted up by one degree.
    pub fn integrate_polynomial(poly: &Polynomial, primitives: &PrimitiveSystem) -> Polynomial {
        let coeffs = poly.coefficients();
        let mut new_coeffs = vec![0i64]; // Constant of integration = 0

        for (power, &coeff) in coeffs.iter().enumerate() {
            // integral of a_n * x^n = (a_n / (n+1)) * x^(n+1)
            let new_power = power + 1;
            let new_coeff = coeff / (new_power as i64); // Integer division
            new_coeffs.push(new_coeff);
        }

        Polynomial::new(new_coeffs, "x", primitives)
    }

    /// Compute definite integral of a polynomial using the antiderivative.
    pub fn definite_integral(
        poly: &Polynomial,
        lower: i64,
        upper: i64,
        primitives: &PrimitiveSystem,
        engine: &mut HybridArithmeticEngine,
    ) -> Option<i64> {
        let antiderivative = Self::integrate_polynomial(poly, primitives);
        let upper_val = antiderivative.evaluate(upper, engine).value;
        let lower_val = antiderivative.evaluate(lower, engine).value;
        // Note: HybridResult.value is u64, so this only works for non-negative results
        Some(upper_val as i64 - lower_val as i64)
    }
}

/// Verifies the fundamental theorem of calculus for polynomials
pub struct FundamentalTheoremVerifier;

impl FundamentalTheoremVerifier {
    /// Verify that d/dx integral f(t)dt = f(x) for a polynomial.
    pub fn verify(poly: &Polynomial, primitives: &PrimitiveSystem) -> bool {
        let integral = SymbolicIntegrator::integrate_polynomial(poly, primitives);
        let derivative_of_integral = integral.derivative(primitives);

        let original_coeffs = poly.coefficients();
        let result_coeffs = derivative_of_integral.coefficients();

        if original_coeffs.len() != result_coeffs.len() {
            return false;
        }

        original_coeffs.iter().zip(result_coeffs.iter()).all(|(a, b)| a == b)
    }
}

/// GCD for i64 values (used for rational normalization)
fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// A polynomial with rational (exact) coefficients.
///
/// Allows lossless integration: ∫x dx = (1/2)x² instead of truncating to 0.
/// Coefficients are stored as (numerator, denominator) pairs in lowest terms.
pub struct RationalPolynomial {
    /// Coefficients as (numerator, denominator) pairs, index = power of x
    coefficients: Vec<(i64, i64)>,
    /// Variable name
    variable: String,
}

impl RationalPolynomial {
    /// Create from integer polynomial coefficients
    pub fn from_integer_coeffs(coeffs: &[i64], variable: &str) -> Self {
        Self {
            coefficients: coeffs.iter().map(|&c| (c, 1)).collect(),
            variable: variable.to_string(),
        }
    }

    /// Create from rational coefficient pairs
    pub fn new(coeffs: Vec<(i64, i64)>, variable: &str) -> Self {
        // Normalize each coefficient
        let normalized: Vec<(i64, i64)> = coeffs.iter().map(|&(n, d)| {
            if n == 0 { return (0, 1); }
            let g = gcd_i64(n.abs(), d.abs());
            let (n2, d2) = (n / g, d / g);
            if d2 < 0 { (-n2, -d2) } else { (n2, d2) }
        }).collect();
        Self {
            coefficients: normalized,
            variable: variable.to_string(),
        }
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[(i64, i64)] {
        &self.coefficients
    }

    /// Evaluate at an integer point, returning a rational result (num, den)
    pub fn evaluate_rational(&self, x: i64) -> (i64, i64) {
        let mut num = 0i64;
        let mut den = 1i64;

        for (power, &(cn, cd)) in self.coefficients.iter().enumerate() {
            if cn == 0 { continue; }
            // term = (cn/cd) * x^power
            let x_pow = x.pow(power as u32);
            let term_num = cn * x_pow;
            let term_den = cd;

            // Add: num/den + term_num/term_den = (num*term_den + term_num*den) / (den*term_den)
            num = num * term_den + term_num * den;
            den *= term_den;

            // Reduce to prevent overflow
            let g = gcd_i64(num.abs(), den.abs());
            if g > 1 {
                num /= g;
                den /= g;
            }
        }

        (num, den)
    }

    /// Evaluate at an integer point, returning f64
    pub fn evaluate_f64(&self, x: i64) -> f64 {
        let (num, den) = self.evaluate_rational(x);
        num as f64 / den as f64
    }

    /// Integrate: exact power rule with rational coefficients
    ///
    /// ∫ (a_n/b_n) x^n dx = (a_n / (b_n * (n+1))) x^(n+1)
    pub fn integrate(&self) -> RationalPolynomial {
        let mut new_coeffs = vec![(0i64, 1i64)]; // constant of integration = 0

        for (power, &(cn, cd)) in self.coefficients.iter().enumerate() {
            let new_power = power as i64 + 1;
            // (cn/cd) / new_power = cn / (cd * new_power)
            let new_num = cn;
            let new_den = cd * new_power;

            // Normalize
            if new_num == 0 {
                new_coeffs.push((0, 1));
            } else {
                let g = gcd_i64(new_num.abs(), new_den.abs());
                let (n, d) = (new_num / g, new_den / g);
                if d < 0 { new_coeffs.push((-n, -d)); } else { new_coeffs.push((n, d)); }
            }
        }

        RationalPolynomial {
            coefficients: new_coeffs,
            variable: self.variable.clone(),
        }
    }

    /// Differentiate: power rule with rational coefficients
    ///
    /// d/dx (a_n/b_n) x^n = (n * a_n / b_n) x^(n-1)
    pub fn derivative(&self) -> RationalPolynomial {
        if self.coefficients.len() <= 1 {
            return RationalPolynomial {
                coefficients: vec![(0, 1)],
                variable: self.variable.clone(),
            };
        }

        let mut new_coeffs = Vec::new();
        for (power, &(cn, cd)) in self.coefficients.iter().enumerate().skip(1) {
            let new_num = cn * power as i64;
            let new_den = cd;
            if new_num == 0 {
                new_coeffs.push((0, 1));
            } else {
                let g = gcd_i64(new_num.abs(), new_den.abs());
                new_coeffs.push((new_num / g, new_den / g));
            }
        }

        if new_coeffs.is_empty() {
            new_coeffs.push((0, 1));
        }

        RationalPolynomial {
            coefficients: new_coeffs,
            variable: self.variable.clone(),
        }
    }

    /// Compute definite integral from lower to upper
    pub fn definite_integral(&self, lower: i64, upper: i64) -> (i64, i64) {
        let antideriv = self.integrate();
        let (upper_num, upper_den) = antideriv.evaluate_rational(upper);
        let (lower_num, lower_den) = antideriv.evaluate_rational(lower);

        // upper - lower
        let num = upper_num * lower_den - lower_num * upper_den;
        let den = upper_den * lower_den;

        let g = gcd_i64(num.abs(), den.abs());
        if g > 1 { (num / g, den / g) } else { (num, den) }
    }

    /// Verify fundamental theorem: d/dx ∫f(x) dx == f(x)
    pub fn verify_fundamental_theorem(&self) -> bool {
        let integral = self.integrate();
        let derivative_of_integral = integral.derivative();

        if self.coefficients.len() != derivative_of_integral.coefficients.len() {
            return false;
        }

        self.coefficients.iter().zip(derivative_of_integral.coefficients.iter())
            .all(|(&(an, ad), &(bn, bd))| {
                // Compare: an/ad == bn/bd => an*bd == bn*ad
                an * bd == bn * ad
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_differentiate_power() {
        let prims = PrimitiveSystem::global();
        // d/dx(x^3) = 3x^2
        let x = SymbolicExpr::variable("x", prims);
        let three = SymbolicExpr::constant(3, prims);
        let x_cubed = x.pow(&three, prims);
        let result = SymbolicDifferentiator::differentiate(&x_cubed, "x", prims);
        // Evaluate at x=2: should be 3*4 = 12
        let mut engine = HybridArithmeticEngine::new();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2);
        let val = result.expr.evaluate(&vars, &mut engine).unwrap();
        assert_eq!(val.value, 12);
    }

    #[test]
    fn test_differentiate_constant() {
        let prims = PrimitiveSystem::global();
        let c = SymbolicExpr::constant(5, prims);
        let result = SymbolicDifferentiator::differentiate(&c, "x", prims);
        let mut engine = HybridArithmeticEngine::new();
        let vars = HashMap::new();
        let val = result.expr.evaluate(&vars, &mut engine).unwrap();
        assert_eq!(val.value, 0);
    }

    #[test]
    fn test_differentiate_sum() {
        let prims = PrimitiveSystem::global();
        // d/dx(x^2 + 3x) = 2x + 3
        let x = SymbolicExpr::variable("x", prims);
        let two = SymbolicExpr::constant(2, prims);
        let x2 = x.pow(&two, prims);
        let three = SymbolicExpr::constant(3, prims);
        let three_x = three.mul(&x, prims);
        let expr = x2.add(&three_x, prims);
        let result = SymbolicDifferentiator::differentiate(&expr, "x", prims);
        let mut engine = HybridArithmeticEngine::new();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1);
        let val = result.expr.evaluate(&vars, &mut engine).unwrap();
        assert_eq!(val.value, 5); // 2*1 + 3 = 5
    }

    #[test]
    fn test_polynomial_integration() {
        let prims = PrimitiveSystem::global();
        // integral of (3x^2 + 2x + 1) = x^3 + x^2 + x
        let poly = Polynomial::new(vec![1, 2, 3], "x", prims); // 1 + 2x + 3x^2
        let integral = SymbolicIntegrator::integrate_polynomial(&poly, prims);
        assert_eq!(integral.coefficients()[0], 0);
        assert_eq!(integral.coefficients()[1], 1); // 1/1
        assert_eq!(integral.coefficients()[2], 1); // 2/2
        assert_eq!(integral.coefficients()[3], 1); // 3/3
    }

    #[test]
    fn test_fundamental_theorem() {
        let prims = PrimitiveSystem::global();
        let poly = Polynomial::new(vec![1, 2, 3], "x", prims);
        assert!(FundamentalTheoremVerifier::verify(&poly, prims));
    }

    #[test]
    fn test_definite_integral() {
        let prims = PrimitiveSystem::global();
        let mut engine = HybridArithmeticEngine::new();
        // integral from 0 to 1 of 2x dx = x^2 |_0^1 = 1
        let poly = Polynomial::new(vec![0, 2], "x", prims);
        let result = SymbolicIntegrator::definite_integral(&poly, 0, 1, prims, &mut engine).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_higher_order_derivative() {
        let prims = PrimitiveSystem::global();
        // d^2/dx^2(x^3) = 6x
        let x = SymbolicExpr::variable("x", prims);
        let three = SymbolicExpr::constant(3, prims);
        let x_cubed = x.pow(&three, prims);
        let result = SymbolicDifferentiator::differentiate_n(&x_cubed, "x", 2, prims);
        let mut engine = HybridArithmeticEngine::new();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2);
        let val = result.expr.evaluate(&vars, &mut engine).unwrap();
        assert_eq!(val.value, 12); // 6*2 = 12
    }

    #[test]
    fn test_rational_polynomial_integrate_x() {
        // ∫x dx = (1/2)x² — the case that integer division breaks
        let poly = RationalPolynomial::from_integer_coeffs(&[0, 1], "x"); // f(x) = x
        let integral = poly.integrate();
        // integral should be [0, 0, (1,2)] → (1/2)x²
        assert_eq!(integral.coefficients()[2], (1, 2));
    }

    #[test]
    fn test_rational_polynomial_ftc() {
        // Fundamental theorem should hold for any integer-coefficient polynomial
        for coeffs in &[
            vec![1],                    // f(x) = 1
            vec![0, 1],                 // f(x) = x
            vec![0, 0, 1],             // f(x) = x²
            vec![1, 2, 3],             // f(x) = 1 + 2x + 3x²
            vec![5, -3, 0, 7],         // f(x) = 5 - 3x + 7x³
        ] {
            let poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
            assert!(poly.verify_fundamental_theorem(),
                "FTC should hold for {:?}", coeffs);
        }
    }

    #[test]
    fn test_rational_definite_integral() {
        // ∫₀¹ x dx = 1/2
        let poly = RationalPolynomial::from_integer_coeffs(&[0, 1], "x");
        let (num, den) = poly.definite_integral(0, 1);
        assert_eq!((num, den), (1, 2));

        // ∫₀² (3x²) dx = [x³]₀² = 8
        let poly2 = RationalPolynomial::from_integer_coeffs(&[0, 0, 3], "x");
        let (num2, den2) = poly2.definite_integral(0, 2);
        assert_eq!(num2 as f64 / den2 as f64, 8.0);
    }

    #[test]
    fn test_rational_polynomial_evaluate() {
        // f(x) = 1/2 + (3/4)x, evaluate at x=2: 1/2 + 3/2 = 2
        let poly = RationalPolynomial::new(vec![(1, 2), (3, 4)], "x");
        let (num, den) = poly.evaluate_rational(2);
        assert_eq!(num as f64 / den as f64, 2.0);
    }
}
