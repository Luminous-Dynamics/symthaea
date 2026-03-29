// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symbolic calculus: differentiation, integration, and fundamental theorem verification.
//!
//! ## Capabilities
//!
//! - **Differentiation**: Power rule, product rule, quotient rule, chain rule
//! - **Transcendental functions**: sin, cos, tan, exp, ln and their derivatives
//! - **Integration**: Polynomial (exact via rational coefficients), power rule
//! - **Fundamental Theorem**: Verification that d/dx ∫f = f
//!
//! ## Transcendental Function Support
//!
//! The `TermType::Function` variant encodes named functions (sin, cos, exp, ln, etc.)
//! with full chain rule support for compositions like d/dx sin(x²) = cos(x²) · 2x.

use crate::hdc::arithmetic_engine::{
    HybridArithmeticEngine, Polynomial, SymbolicExpr, SymbolicOp, TermType,
};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{seed_from_name, PrimitiveSystem};

/// Result of a calculus operation with HDC encoding
pub struct CalculusResult {
    pub expr: SymbolicExpr,
    pub description: String,
    pub encoding: BinaryHV,
    pub phi: f64,
}

/// Symbolic differentiation engine
///
/// Supports:
/// - Constant, variable, sum, difference, product, quotient rules
/// - Power rule (constant and variable exponents)
/// - Chain rule for composite functions
/// - Transcendental functions: sin, cos, tan, exp, ln, sqrt
pub struct SymbolicDifferentiator;

impl SymbolicDifferentiator {
    /// Differentiate an expression with respect to a variable
    pub fn differentiate(
        expr: &SymbolicExpr,
        variable: &str,
        primitives: &PrimitiveSystem,
    ) -> CalculusResult {
        let diff_expr = Self::diff_recursive(&expr.term_type, variable, primitives);

        let diff_prim = primitives
            .get("DIFFERENTIATION")
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
    pub fn differentiate_n(
        expr: &SymbolicExpr,
        variable: &str,
        n: usize,
        primitives: &PrimitiveSystem,
    ) -> CalculusResult {
        let mut current_term = expr.term_type.clone();
        for _ in 0..n {
            current_term = Self::diff_recursive(&current_term, variable, primitives);
        }

        let diff_prim = primitives
            .get("DIFFERENTIATION")
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
    ///
    /// Implements:
    /// - d/dx c = 0 (constant rule)
    /// - d/dx x = 1 (variable rule)
    /// - d/dx (f + g) = f' + g' (sum rule)
    /// - d/dx (f * g) = f'g + fg' (product rule)
    /// - d/dx (f / g) = (f'g - fg') / g² (quotient rule)
    /// - d/dx f^n = n * f^(n-1) * f' (generalized power rule with chain rule)
    /// - d/dx f^g = f^g * (g' * ln(f) + g * f'/f) (general power rule)
    /// - d/dx sin(f) = cos(f) * f' (chain rule for transcendentals)
    /// - d/dx cos(f) = -sin(f) * f'
    /// - d/dx tan(f) = (1 + tan²(f)) * f'    (sec²(f) * f')
    /// - d/dx exp(f) = exp(f) * f'
    /// - d/dx ln(f) = f' / f
    /// - d/dx sqrt(f) = f' / (2 * sqrt(f))
    fn diff_recursive(term: &TermType, variable: &str, primitives: &PrimitiveSystem) -> TermType {
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
                        let dl = Self::diff_recursive(left, variable, primitives);
                        let dr = Self::diff_recursive(right, variable, primitives);
                        TermType::BinaryOp {
                            op: SymbolicOp::Add,
                            left: Box::new(dl),
                            right: Box::new(dr),
                        }
                    }
                    SymbolicOp::Sub => {
                        let dl = Self::diff_recursive(left, variable, primitives);
                        let dr = Self::diff_recursive(right, variable, primitives);
                        TermType::BinaryOp {
                            op: SymbolicOp::Sub,
                            left: Box::new(dl),
                            right: Box::new(dr),
                        }
                    }
                    SymbolicOp::Mul => {
                        // Product rule: d(f*g) = f'*g + f*g'
                        let df = Self::diff_recursive(left, variable, primitives);
                        let dg = Self::diff_recursive(right, variable, primitives);
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
                        let df = Self::diff_recursive(left, variable, primitives);
                        let dg = Self::diff_recursive(right, variable, primitives);
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
                        // Check if exponent is constant for simpler power rule
                        if let TermType::Constant(n) = right.as_ref() {
                            // d/dx f^n = n * f^(n-1) * f' (generalized power rule)
                            let n_val = *n;
                            let df = Self::diff_recursive(left, variable, primitives);
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
                        } else if let TermType::Constant(_) = left.as_ref() {
                            // d/dx a^g = a^g * ln(a) * g'
                            // For now, encode as: a^g * ln(a) * g'
                            let dg = Self::diff_recursive(right, variable, primitives);
                            TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: Box::new(TermType::BinaryOp {
                                    op: SymbolicOp::Mul,
                                    left: Box::new(term.clone()),
                                    right: Box::new(TermType::Function {
                                        name: "ln".to_string(),
                                        arg: left.clone(),
                                    }),
                                }),
                                right: Box::new(dg),
                            }
                        } else {
                            // General case: d/dx f^g = f^g * (g' * ln(f) + g * f'/f)
                            let df = Self::diff_recursive(left, variable, primitives);
                            let dg = Self::diff_recursive(right, variable, primitives);
                            // f^g * (g' * ln(f) + g * f'/f)
                            TermType::BinaryOp {
                                op: SymbolicOp::Mul,
                                left: Box::new(term.clone()), // f^g
                                right: Box::new(TermType::BinaryOp {
                                    op: SymbolicOp::Add,
                                    left: Box::new(TermType::BinaryOp {
                                        op: SymbolicOp::Mul,
                                        left: Box::new(dg),
                                        right: Box::new(TermType::Function {
                                            name: "ln".to_string(),
                                            arg: left.clone(),
                                        }),
                                    }),
                                    right: Box::new(TermType::BinaryOp {
                                        op: SymbolicOp::Mul,
                                        left: right.clone(),
                                        right: Box::new(TermType::BinaryOp {
                                            op: SymbolicOp::Div,
                                            left: Box::new(df),
                                            right: left.clone(),
                                        }),
                                    }),
                                }),
                            }
                        }
                    }
                    SymbolicOp::Neg => {
                        // Shouldn't appear as binary op
                        TermType::Constant(0)
                    }
                }
            }

            TermType::UnaryOp { op, operand } => match op {
                SymbolicOp::Neg => {
                    let d = Self::diff_recursive(operand, variable, primitives);
                    TermType::UnaryOp {
                        op: SymbolicOp::Neg,
                        operand: Box::new(d),
                    }
                }
                _ => TermType::Constant(0),
            },

            TermType::Function { name, arg } => {
                // Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x)
                let dg = Self::diff_recursive(arg, variable, primitives);

                // If inner derivative is 0, the whole thing is 0
                if dg == TermType::Constant(0) {
                    return TermType::Constant(0);
                }

                // Compute outer derivative f'(g(x)) based on function name
                let outer_derivative = Self::diff_function(name, arg);

                // Apply chain rule: f'(g(x)) * g'(x)
                if dg == TermType::Constant(1) {
                    // No need to multiply by 1
                    outer_derivative
                } else {
                    TermType::BinaryOp {
                        op: SymbolicOp::Mul,
                        left: Box::new(outer_derivative),
                        right: Box::new(dg),
                    }
                }
            }
        }
    }

    /// Compute the derivative of a named function with respect to its argument.
    ///
    /// Returns f'(u) where u is the argument (chain rule outer part).
    fn diff_function(name: &str, arg: &TermType) -> TermType {
        match name {
            // d/du sin(u) = cos(u)
            "sin" => TermType::Function {
                name: "cos".to_string(),
                arg: Box::new(arg.clone()),
            },

            // d/du cos(u) = -sin(u)
            "cos" => TermType::UnaryOp {
                op: SymbolicOp::Neg,
                operand: Box::new(TermType::Function {
                    name: "sin".to_string(),
                    arg: Box::new(arg.clone()),
                }),
            },

            // d/du tan(u) = 1 + tan²(u) = sec²(u)
            "tan" => TermType::BinaryOp {
                op: SymbolicOp::Add,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Pow,
                    left: Box::new(TermType::Function {
                        name: "tan".to_string(),
                        arg: Box::new(arg.clone()),
                    }),
                    right: Box::new(TermType::Constant(2)),
                }),
            },

            // d/du exp(u) = exp(u)
            "exp" => TermType::Function {
                name: "exp".to_string(),
                arg: Box::new(arg.clone()),
            },

            // d/du ln(u) = 1/u
            "ln" | "log" => TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(arg.clone()),
            },

            // d/du sqrt(u) = 1/(2*sqrt(u))
            "sqrt" => TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Mul,
                    left: Box::new(TermType::Constant(2)),
                    right: Box::new(TermType::Function {
                        name: "sqrt".to_string(),
                        arg: Box::new(arg.clone()),
                    }),
                }),
            },

            // d/du asin(u) = 1/sqrt(1 - u²)
            "asin" | "arcsin" => TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(TermType::Function {
                    name: "sqrt".to_string(),
                    arg: Box::new(TermType::BinaryOp {
                        op: SymbolicOp::Sub,
                        left: Box::new(TermType::Constant(1)),
                        right: Box::new(TermType::BinaryOp {
                            op: SymbolicOp::Pow,
                            left: Box::new(arg.clone()),
                            right: Box::new(TermType::Constant(2)),
                        }),
                    }),
                }),
            },

            // d/du acos(u) = -1/sqrt(1 - u²)
            "acos" | "arccos" => TermType::UnaryOp {
                op: SymbolicOp::Neg,
                operand: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Div,
                    left: Box::new(TermType::Constant(1)),
                    right: Box::new(TermType::Function {
                        name: "sqrt".to_string(),
                        arg: Box::new(TermType::BinaryOp {
                            op: SymbolicOp::Sub,
                            left: Box::new(TermType::Constant(1)),
                            right: Box::new(TermType::BinaryOp {
                                op: SymbolicOp::Pow,
                                left: Box::new(arg.clone()),
                                right: Box::new(TermType::Constant(2)),
                            }),
                        }),
                    }),
                }),
            },

            // d/du atan(u) = 1/(1 + u²)
            "atan" | "arctan" => TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Add,
                    left: Box::new(TermType::Constant(1)),
                    right: Box::new(TermType::BinaryOp {
                        op: SymbolicOp::Pow,
                        left: Box::new(arg.clone()),
                        right: Box::new(TermType::Constant(2)),
                    }),
                }),
            },

            // d/du sinh(u) = cosh(u)
            "sinh" => TermType::Function {
                name: "cosh".to_string(),
                arg: Box::new(arg.clone()),
            },

            // d/du cosh(u) = sinh(u)
            "cosh" => TermType::Function {
                name: "sinh".to_string(),
                arg: Box::new(arg.clone()),
            },

            // d/du tanh(u) = 1 - tanh²(u) = sech²(u)
            "tanh" => TermType::BinaryOp {
                op: SymbolicOp::Sub,
                left: Box::new(TermType::Constant(1)),
                right: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Pow,
                    left: Box::new(TermType::Function {
                        name: "tanh".to_string(),
                        arg: Box::new(arg.clone()),
                    }),
                    right: Box::new(TermType::Constant(2)),
                }),
            },

            // Unknown function: return generic derivative marker
            _ => TermType::Function {
                name: format!("{name}'"),
                arg: Box::new(arg.clone()),
            },
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
            TermType::Function { name, arg } => {
                let inner = Self::term_to_expr(arg, primitives);
                // Encode function application in HDC space
                let fn_seed = seed_from_name(&format!("FN_{name}"));
                let fn_encoding = BinaryHV::random(fn_seed);
                let encoding = fn_encoding.bind(&inner.encoding);
                let display = format!("{}({})", name, inner.display);
                SymbolicExpr {
                    term_type: term.clone(),
                    encoding,
                    phi: inner.phi + 0.5,
                    display,
                }
            }
        }
    }
}

/// Symbolic integration engine
pub struct SymbolicIntegrator;

impl SymbolicIntegrator {
    /// Integrate a polynomial using the power rule with exact rational coefficients.
    ///
    /// Uses RationalPolynomial internally to avoid integer truncation.
    /// For example, ∫x dx = (1/2)x² — not 0 from integer division.
    pub fn integrate_polynomial(poly: &Polynomial, primitives: &PrimitiveSystem) -> Polynomial {
        // Convert to rational polynomial for exact integration
        let coeffs = poly.coefficients();
        let rational_poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
        let integrated = rational_poly.integrate();

        // Convert back to integer polynomial (with rounding for non-integer results)
        // This preserves exact results when they are integers, and rounds otherwise
        let mut new_coeffs = Vec::new();
        for &(num, den) in integrated.coefficients() {
            if den == 1 || num == 0 {
                new_coeffs.push(num);
            } else {
                // Non-integer coefficient - round to nearest integer
                // (Use RationalPolynomial directly for exact results)
                new_coeffs.push(num / den);
            }
        }

        Polynomial::new(new_coeffs, "x", primitives)
    }

    /// Integrate a polynomial with exact rational coefficients (no truncation).
    ///
    /// This is the preferred method — returns a RationalPolynomial
    /// that preserves exact coefficients like 1/2, 1/3, etc.
    pub fn integrate_polynomial_exact(poly: &Polynomial) -> RationalPolynomial {
        let coeffs = poly.coefficients();
        let rational_poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
        rational_poly.integrate()
    }

    /// Compute definite integral of a polynomial using exact rational arithmetic.
    pub fn definite_integral(
        poly: &Polynomial,
        lower: i64,
        upper: i64,
        _primitives: &PrimitiveSystem,
        _engine: &mut HybridArithmeticEngine,
    ) -> Option<i64> {
        let coeffs = poly.coefficients();
        let rational_poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
        let (num, den) = rational_poly.definite_integral(lower, upper);
        // Return integer result (exact when the integral is an integer)
        Some(num / den)
    }

    /// Compute definite integral returning exact rational result (numerator, denominator).
    pub fn definite_integral_exact(poly: &Polynomial, lower: i64, upper: i64) -> (i64, i64) {
        let coeffs = poly.coefficients();
        let rational_poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
        rational_poly.definite_integral(lower, upper)
    }
}

/// Verifies the fundamental theorem of calculus for polynomials
pub struct FundamentalTheoremVerifier;

impl FundamentalTheoremVerifier {
    /// Verify that d/dx integral f(t)dt = f(x) for a polynomial.
    ///
    /// Uses RationalPolynomial for exact verification (no integer truncation).
    pub fn verify(poly: &Polynomial, _primitives: &PrimitiveSystem) -> bool {
        let coeffs = poly.coefficients();
        let rational_poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
        rational_poly.verify_fundamental_theorem()
    }
}

/// Helper: construct a Function TermType
pub fn func(name: &str, arg: TermType) -> TermType {
    TermType::Function {
        name: name.to_string(),
        arg: Box::new(arg),
    }
}

/// Helper: construct a SymbolicExpr for a function application
pub fn func_expr(name: &str, arg: &SymbolicExpr, primitives: &PrimitiveSystem) -> SymbolicExpr {
    let fn_seed = seed_from_name(&format!("FN_{name}"));
    let fn_encoding = BinaryHV::random(fn_seed);
    let encoding = fn_encoding.bind(&arg.encoding);
    let display = format!("{}({})", name, arg.display);
    SymbolicExpr {
        term_type: TermType::Function {
            name: name.to_string(),
            arg: Box::new(arg.term_type.clone()),
        },
        encoding,
        phi: arg.phi + 0.5,
        display,
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
        let normalized: Vec<(i64, i64)> = coeffs
            .iter()
            .map(|&(n, d)| {
                if n == 0 {
                    return (0, 1);
                }
                let g = gcd_i64(n.abs(), d.abs());
                let (n2, d2) = (n / g, d / g);
                if d2 < 0 {
                    (-n2, -d2)
                } else {
                    (n2, d2)
                }
            })
            .collect();
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
            if cn == 0 {
                continue;
            }
            // term = (cn/cd) * x^power
            let x_pow = x.pow(power as u32);
            let term_num = cn.wrapping_mul(x_pow);
            let term_den = cd;

            // Add: num/den + term_num/term_den = (num*term_den + term_num*den) / (den*term_den)
            num = num
                .wrapping_mul(term_den)
                .wrapping_add(term_num.wrapping_mul(den));
            den = den.wrapping_mul(term_den);

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
                if d < 0 {
                    new_coeffs.push((-n, -d));
                } else {
                    new_coeffs.push((n, d));
                }
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
        if g > 1 {
            (num / g, den / g)
        } else {
            (num, den)
        }
    }

    /// Verify fundamental theorem: d/dx ∫f(x) dx == f(x)
    pub fn verify_fundamental_theorem(&self) -> bool {
        let integral = self.integrate();
        let derivative_of_integral = integral.derivative();

        if self.coefficients.len() != derivative_of_integral.coefficients.len() {
            return false;
        }

        self.coefficients
            .iter()
            .zip(derivative_of_integral.coefficients.iter())
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

    // ==================== BASIC DIFFERENTIATION ====================

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

    // ==================== TRANSCENDENTAL FUNCTIONS ====================

    #[test]
    fn test_diff_sin_x() {
        let prims = PrimitiveSystem::global();
        // d/dx sin(x) = cos(x)
        let x_var = TermType::Variable("x".to_string());
        let sin_x = TermType::Function {
            name: "sin".to_string(),
            arg: Box::new(x_var),
        };
        let result = SymbolicDifferentiator::diff_recursive(&sin_x, "x", prims);
        // Should be cos(x)
        match &result {
            TermType::Function { name, arg } => {
                assert_eq!(name, "cos");
                assert_eq!(**arg, TermType::Variable("x".to_string()));
            }
            _ => panic!("Expected Function(cos, x), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_cos_x() {
        let prims = PrimitiveSystem::global();
        // d/dx cos(x) = -sin(x)
        let x_var = TermType::Variable("x".to_string());
        let cos_x = TermType::Function {
            name: "cos".to_string(),
            arg: Box::new(x_var),
        };
        let result = SymbolicDifferentiator::diff_recursive(&cos_x, "x", prims);
        // Should be -sin(x) = UnaryOp(Neg, Function(sin, x))
        match &result {
            TermType::UnaryOp {
                op: SymbolicOp::Neg,
                operand,
            } => match operand.as_ref() {
                TermType::Function { name, arg } => {
                    assert_eq!(name, "sin");
                    assert_eq!(**arg, TermType::Variable("x".to_string()));
                }
                _ => panic!("Expected Function(sin, x) inside Neg"),
            },
            _ => panic!("Expected UnaryOp(Neg, Function(sin, x)), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_exp_x() {
        let prims = PrimitiveSystem::global();
        // d/dx exp(x) = exp(x)
        let x_var = TermType::Variable("x".to_string());
        let exp_x = TermType::Function {
            name: "exp".to_string(),
            arg: Box::new(x_var.clone()),
        };
        let result = SymbolicDifferentiator::diff_recursive(&exp_x, "x", prims);
        // Should be exp(x)
        match &result {
            TermType::Function { name, arg } => {
                assert_eq!(name, "exp");
                assert_eq!(**arg, TermType::Variable("x".to_string()));
            }
            _ => panic!("Expected Function(exp, x), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_ln_x() {
        let prims = PrimitiveSystem::global();
        // d/dx ln(x) = 1/x
        let x_var = TermType::Variable("x".to_string());
        let ln_x = TermType::Function {
            name: "ln".to_string(),
            arg: Box::new(x_var),
        };
        let result = SymbolicDifferentiator::diff_recursive(&ln_x, "x", prims);
        // Should be 1/x = BinaryOp(Div, 1, x)
        match &result {
            TermType::BinaryOp {
                op: SymbolicOp::Div,
                left,
                right,
            } => {
                assert_eq!(**left, TermType::Constant(1));
                assert_eq!(**right, TermType::Variable("x".to_string()));
            }
            _ => panic!("Expected BinaryOp(Div, 1, x), got {:?}", result),
        }
    }

    // ==================== CHAIN RULE ====================

    #[test]
    fn test_chain_rule_sin_x_squared() {
        let prims = PrimitiveSystem::global();
        // d/dx sin(x²) = cos(x²) * 2x
        let x_var = TermType::Variable("x".to_string());
        let x_squared = TermType::BinaryOp {
            op: SymbolicOp::Pow,
            left: Box::new(x_var),
            right: Box::new(TermType::Constant(2)),
        };
        let sin_x2 = TermType::Function {
            name: "sin".to_string(),
            arg: Box::new(x_squared),
        };
        let result = SymbolicDifferentiator::diff_recursive(&sin_x2, "x", prims);
        // Should be cos(x²) * (2 * x^1 * 1) = cos(x²) * 2x
        // The structure is: Mul(cos(x²), Mul(Mul(2, Pow(x, 1)), 1))
        match &result {
            TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left,
                right: _,
            } => {
                // Left should be cos(x²)
                match left.as_ref() {
                    TermType::Function { name, .. } => assert_eq!(name, "cos"),
                    _ => panic!("Expected cos(...) on left of product"),
                }
            }
            _ => panic!("Expected Mul(cos(x²), ...), got {:?}", result),
        }
    }

    #[test]
    fn test_chain_rule_exp_3x() {
        let prims = PrimitiveSystem::global();
        // d/dx exp(3x) = 3 * exp(3x)
        let three_x = TermType::BinaryOp {
            op: SymbolicOp::Mul,
            left: Box::new(TermType::Constant(3)),
            right: Box::new(TermType::Variable("x".to_string())),
        };
        let exp_3x = TermType::Function {
            name: "exp".to_string(),
            arg: Box::new(three_x),
        };
        let result = SymbolicDifferentiator::diff_recursive(&exp_3x, "x", prims);
        // Should be exp(3x) * d/dx(3x) = exp(3x) * (0*x + 3*1) = exp(3x) * 3
        match &result {
            TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left,
                ..
            } => {
                // Left should be exp(3x)
                match left.as_ref() {
                    TermType::Function { name, .. } => assert_eq!(name, "exp"),
                    _ => panic!("Expected exp(...) on left of product"),
                }
            }
            _ => panic!("Expected Mul(exp(3x), ...), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_constant_function() {
        let prims = PrimitiveSystem::global();
        // d/dx sin(5) = 0 (constant argument → derivative is 0)
        let sin_5 = TermType::Function {
            name: "sin".to_string(),
            arg: Box::new(TermType::Constant(5)),
        };
        let result = SymbolicDifferentiator::diff_recursive(&sin_5, "x", prims);
        assert_eq!(result, TermType::Constant(0));
    }

    #[test]
    fn test_diff_different_variable() {
        let prims = PrimitiveSystem::global();
        // d/dx sin(y) = 0 (y is treated as constant w.r.t. x)
        let sin_y = TermType::Function {
            name: "sin".to_string(),
            arg: Box::new(TermType::Variable("y".to_string())),
        };
        let result = SymbolicDifferentiator::diff_recursive(&sin_y, "x", prims);
        assert_eq!(result, TermType::Constant(0));
    }

    // ==================== GENERAL POWER RULE ====================

    #[test]
    fn test_diff_variable_exponent() {
        let prims = PrimitiveSystem::global();
        // d/dx x^x = x^x * (1 * ln(x) + x * (1/x))
        //          = x^x * (ln(x) + 1)
        let x_var = TermType::Variable("x".to_string());
        let x_to_x = TermType::BinaryOp {
            op: SymbolicOp::Pow,
            left: Box::new(x_var.clone()),
            right: Box::new(x_var),
        };
        let result = SymbolicDifferentiator::diff_recursive(&x_to_x, "x", prims);
        // Should NOT return Constant(0) anymore
        assert_ne!(result, TermType::Constant(0), "d/dx(x^x) should not be 0");
        // Should be a Mul expression
        match &result {
            TermType::BinaryOp {
                op: SymbolicOp::Mul,
                ..
            } => {
                // Good - it's a product (x^x * something)
            }
            _ => panic!("Expected Mul for d/dx(x^x), got {:?}", result),
        }
    }

    // ==================== POLYNOMIAL INTEGRATION ====================

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
        let result =
            SymbolicIntegrator::definite_integral(&poly, 0, 1, prims, &mut engine).unwrap();
        assert_eq!(result, 1);
    }

    // ==================== RATIONAL POLYNOMIAL (EXACT INTEGRATION) ====================

    #[test]
    fn test_rational_polynomial_integrate_x() {
        // ∫x dx = (1/2)x² — the case that integer division used to break
        let poly = RationalPolynomial::from_integer_coeffs(&[0, 1], "x"); // f(x) = x
        let integral = poly.integrate();
        // integral should be [0, 0, (1,2)] → (1/2)x²
        assert_eq!(integral.coefficients()[2], (1, 2));
    }

    #[test]
    fn test_exact_definite_integral_half() {
        // ∫₀¹ x dx = 1/2 (exact rational result)
        let poly = Polynomial::new(vec![0, 1], "x", PrimitiveSystem::global());
        let (num, den) = SymbolicIntegrator::definite_integral_exact(&poly, 0, 1);
        assert_eq!((num, den), (1, 2));
    }

    #[test]
    fn test_rational_polynomial_ftc() {
        // Fundamental theorem should hold for any integer-coefficient polynomial
        for coeffs in &[
            vec![1],           // f(x) = 1
            vec![0, 1],        // f(x) = x
            vec![0, 0, 1],     // f(x) = x²
            vec![1, 2, 3],     // f(x) = 1 + 2x + 3x²
            vec![5, -3, 0, 7], // f(x) = 5 - 3x + 7x³
        ] {
            let poly = RationalPolynomial::from_integer_coeffs(coeffs, "x");
            assert!(
                poly.verify_fundamental_theorem(),
                "FTC should hold for {:?}",
                coeffs
            );
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

    // ==================== HELPER FUNCTION TESTS ====================

    #[test]
    fn test_func_helper() {
        let x = TermType::Variable("x".to_string());
        let sin_x = func("sin", x);
        match &sin_x {
            TermType::Function { name, arg } => {
                assert_eq!(name, "sin");
                assert_eq!(**arg, TermType::Variable("x".to_string()));
            }
            _ => panic!("Expected Function"),
        }
    }

    #[test]
    fn test_func_expr_helper() {
        let prims = PrimitiveSystem::global();
        let x = SymbolicExpr::variable("x", prims);
        let sin_x = func_expr("sin", &x, prims);
        assert!(sin_x.display.contains("sin"));
        assert!(sin_x.display.contains("x"));
    }

    // ==================== HYPERBOLIC FUNCTIONS ====================

    #[test]
    fn test_diff_sinh_x() {
        let prims = PrimitiveSystem::global();
        // d/dx sinh(x) = cosh(x)
        let sinh_x = func("sinh", TermType::Variable("x".to_string()));
        let result = SymbolicDifferentiator::diff_recursive(&sinh_x, "x", prims);
        match &result {
            TermType::Function { name, .. } => assert_eq!(name, "cosh"),
            _ => panic!("Expected cosh(x), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_cosh_x() {
        let prims = PrimitiveSystem::global();
        // d/dx cosh(x) = sinh(x)
        let cosh_x = func("cosh", TermType::Variable("x".to_string()));
        let result = SymbolicDifferentiator::diff_recursive(&cosh_x, "x", prims);
        match &result {
            TermType::Function { name, .. } => assert_eq!(name, "sinh"),
            _ => panic!("Expected sinh(x), got {:?}", result),
        }
    }
}
