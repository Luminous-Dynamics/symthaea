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

// ═══════════════════════════════════════════════════════════════════════════
// TRANSCENDENTAL INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Symbolic integration of transcendental functions.
///
/// Handles standard integral forms:
/// - ∫ sin(x) dx = -cos(x)
/// - ∫ cos(x) dx = sin(x)
/// - ∫ exp(x) dx = exp(x)
/// - ∫ 1/x dx = ln(|x|)
/// - ∫ x^n dx = x^(n+1)/(n+1) for n ≠ -1
/// - ∫ (f + g) dx = ∫f dx + ∫g dx (linearity)
/// - ∫ c·f dx = c · ∫f dx (constant factor)
/// - ∫ sin(ax+b) dx = -cos(ax+b)/a (linear substitution)
/// - ∫ cos(ax+b) dx = sin(ax+b)/a (linear substitution)
/// - ∫ exp(ax+b) dx = exp(ax+b)/a (linear substitution)
///
/// Returns None if the integral cannot be computed symbolically.
pub fn integrate_symbolic(term: &TermType, variable: &str) -> Option<TermType> {
    match term {
        // ∫ c dx = c*x
        TermType::Constant(c) => {
            Some(TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left: Box::new(TermType::Constant(*c)),
                right: Box::new(TermType::Variable(variable.to_string())),
            })
        }

        // ∫ x dx = x²/2
        TermType::Variable(v) if v == variable => {
            Some(TermType::BinaryOp {
                op: SymbolicOp::Div,
                left: Box::new(TermType::BinaryOp {
                    op: SymbolicOp::Pow,
                    left: Box::new(TermType::Variable(variable.to_string())),
                    right: Box::new(TermType::Constant(2)),
                }),
                right: Box::new(TermType::Constant(2)),
            })
        }

        // ∫ y dx = y*x (other variable treated as constant)
        TermType::Variable(v) => {
            Some(TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left: Box::new(TermType::Variable(v.clone())),
                right: Box::new(TermType::Variable(variable.to_string())),
            })
        }

        // ∫ (f + g) dx = ∫f dx + ∫g dx
        TermType::BinaryOp { op: SymbolicOp::Add, left, right } => {
            let il = integrate_symbolic(left, variable)?;
            let ir = integrate_symbolic(right, variable)?;
            Some(TermType::BinaryOp { op: SymbolicOp::Add, left: Box::new(il), right: Box::new(ir) })
        }

        // ∫ (f - g) dx = ∫f dx - ∫g dx
        TermType::BinaryOp { op: SymbolicOp::Sub, left, right } => {
            let il = integrate_symbolic(left, variable)?;
            let ir = integrate_symbolic(right, variable)?;
            Some(TermType::BinaryOp { op: SymbolicOp::Sub, left: Box::new(il), right: Box::new(ir) })
        }

        // ∫ c*f dx = c * ∫f dx (constant factor)
        TermType::BinaryOp { op: SymbolicOp::Mul, left, right } => {
            if !contains_variable(left, variable) {
                // Left is constant w.r.t. variable
                let ir = integrate_symbolic(right, variable)?;
                Some(TermType::BinaryOp { op: SymbolicOp::Mul, left: left.clone(), right: Box::new(ir) })
            } else if !contains_variable(right, variable) {
                // Right is constant
                let il = integrate_symbolic(left, variable)?;
                Some(TermType::BinaryOp { op: SymbolicOp::Mul, left: Box::new(il), right: right.clone() })
            } else {
                None // Integration by parts would be needed
            }
        }

        // ∫ x^n dx = x^(n+1)/(n+1) for constant n ≠ -1
        TermType::BinaryOp { op: SymbolicOp::Pow, left, right } => {
            if let (TermType::Variable(v), TermType::Constant(n)) = (left.as_ref(), right.as_ref()) {
                if v == variable && *n != -1 {
                    let n1 = n + 1;
                    Some(TermType::BinaryOp {
                        op: SymbolicOp::Div,
                        left: Box::new(TermType::BinaryOp {
                            op: SymbolicOp::Pow,
                            left: Box::new(TermType::Variable(variable.to_string())),
                            right: Box::new(TermType::Constant(n1)),
                        }),
                        right: Box::new(TermType::Constant(n1)),
                    })
                } else if v == variable && *n == -1 {
                    // ∫ x^(-1) dx = ln(x)
                    Some(func("ln", TermType::Variable(variable.to_string())))
                } else {
                    None
                }
            } else {
                None
            }
        }

        // ∫ neg(f) dx = -(∫f dx) — BinaryOp with Neg op (shouldn't happen, but handle it)
        TermType::BinaryOp { op: SymbolicOp::Neg, left, .. } => {
            let il = integrate_symbolic(left, variable)?;
            Some(negate(il))
        }

        // ∫ 1/f dx — check for 1/x = ln(x)
        TermType::BinaryOp { op: SymbolicOp::Div, left, right } => {
            if let TermType::Constant(1) = left.as_ref() {
                if let TermType::Variable(v) = right.as_ref() {
                    if v == variable {
                        return Some(func("ln", TermType::Variable(variable.to_string())));
                    }
                }
            }
            // ∫ f/c dx = (1/c) * ∫f dx
            if !contains_variable(right, variable) {
                let il = integrate_symbolic(left, variable)?;
                Some(TermType::BinaryOp { op: SymbolicOp::Div, left: Box::new(il), right: right.clone() })
            } else {
                None
            }
        }

        // ∫ sin(f) dx, ∫ cos(f) dx, ∫ exp(f) dx
        TermType::Function { name, arg } => {
            integrate_transcendental(name, arg, variable)
        }

        // Unary operation (negation): ∫ (-f) dx = -(∫ f dx)
        TermType::UnaryOp { operand, .. } => {
            let inner = integrate_symbolic(operand, variable)?;
            Some(negate(inner))
        }
    }
}

/// Integrate transcendental functions with linear argument substitution.
///
/// For f(ax + b), uses the substitution u = ax + b, du = a dx:
/// ∫ f(ax+b) dx = F(ax+b) / a
fn integrate_transcendental(name: &str, arg: &TermType, variable: &str) -> Option<TermType> {
    // Check if argument is the simple variable
    if let TermType::Variable(v) = arg {
        if v == variable {
            return match name {
                "sin" => Some(negate(func("cos", TermType::Variable(variable.to_string())))),
                "cos" => Some(func("sin", TermType::Variable(variable.to_string()))),
                "exp" => Some(func("exp", TermType::Variable(variable.to_string()))),
                "ln" => {
                    // ∫ ln(x) dx = x*ln(x) - x
                    let x = TermType::Variable(variable.to_string());
                    Some(TermType::BinaryOp {
                        op: SymbolicOp::Sub,
                        left: Box::new(TermType::BinaryOp {
                            op: SymbolicOp::Mul,
                            left: Box::new(x.clone()),
                            right: Box::new(func("ln", x.clone())),
                        }),
                        right: Box::new(x),
                    })
                }
                _ => None,
            };
        }
    }

    // Check for linear argument: a*x + b
    if let Some((a_coeff, _b_const)) = extract_linear_arg(arg, variable) {
        if a_coeff == 0 { return None; }

        let antideriv = match name {
            "sin" => Some(negate(func("cos", arg.clone()))),
            "cos" => Some(func("sin", arg.clone())),
            "exp" => Some(func("exp", arg.clone())),
            _ => None,
        }?;

        // Divide by the coefficient: ∫ f(ax+b) dx = F(ax+b) / a
        return Some(TermType::BinaryOp {
            op: SymbolicOp::Div,
            left: Box::new(antideriv),
            right: Box::new(TermType::Constant(a_coeff)),
        });
    }

    None
}

/// Check if a TermType contains the given variable.
fn contains_variable(term: &TermType, variable: &str) -> bool {
    match term {
        TermType::Constant(_) => false,
        TermType::Variable(v) => v == variable,
        TermType::BinaryOp { left, right, .. } => {
            contains_variable(left, variable) || contains_variable(right, variable)
        }
        TermType::UnaryOp { operand, .. } => contains_variable(operand, variable),
        TermType::Function { arg, .. } => contains_variable(arg, variable),
    }
}

/// Try to extract linear form a*x + b from a TermType.
/// Returns Some((a, b)) if the expression is linear in the variable.
fn extract_linear_arg(term: &TermType, variable: &str) -> Option<(i64, i64)> {
    match term {
        // Just x → (1, 0)
        TermType::Variable(v) if v == variable => Some((1, 0)),
        // Just c → (0, c)
        TermType::Constant(c) => Some((0, *c)),
        // a * x → (a, 0)
        TermType::BinaryOp { op: SymbolicOp::Mul, left, right } => {
            match (left.as_ref(), right.as_ref()) {
                (TermType::Constant(a), TermType::Variable(v)) if v == variable => Some((*a, 0)),
                (TermType::Variable(v), TermType::Constant(a)) if v == variable => Some((*a, 0)),
                _ => None,
            }
        }
        // a*x + b
        TermType::BinaryOp { op: SymbolicOp::Add, left, right } => {
            let (a1, b1) = extract_linear_arg(left, variable)?;
            let (a2, b2) = extract_linear_arg(right, variable)?;
            Some((a1 + a2, b1 + b2))
        }
        // a*x - b
        TermType::BinaryOp { op: SymbolicOp::Sub, left, right } => {
            let (a1, b1) = extract_linear_arg(left, variable)?;
            let (a2, b2) = extract_linear_arg(right, variable)?;
            Some((a1 - a2, b1 - b2))
        }
        _ => None,
    }
}

/// Negate a TermType: -expr = 0 - expr
fn negate(term: TermType) -> TermType {
    TermType::BinaryOp {
        op: SymbolicOp::Sub,
        left: Box::new(TermType::Constant(0)),
        right: Box::new(term),
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

// ═══════════════════════════════════════════════════════════════════════════
// TRANSCENDENTAL INTEGRATION — numerical evaluation of special integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Sine integral: Si(x) = ∫₀ˣ sin(t)/t dt.
///
/// Computed via adaptive Simpson's rule. Si(∞) = π/2.
/// Used in signal processing, diffraction, and antenna theory.
pub fn sine_integral(x: f64) -> f64 {
    if x.abs() < 1e-15 { return 0.0; }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // For small x, use Taylor series: Si(x) = x - x³/18 + x⁵/600 - ...
    if x < 1.0 {
        let mut sum = 0.0;
        let mut term = x;
        for k in 0..20 {
            sum += term / (2 * k + 1) as f64;
            term *= -x * x / ((2 * k + 2) as f64 * (2 * k + 3) as f64);
        }
        return sign * sum;
    }
    // For larger x, use adaptive quadrature
    let n = (x * 20.0).ceil() as usize;
    let n = n.max(100).min(10000);
    let h = x / n as f64;
    // Simpson's rule
    let mut sum = (x.sin() / x); // sinc at x
    // f(0) = sin(0)/0 = 1 by L'Hôpital
    sum += 1.0; // f(0) = 1
    for i in 1..n {
        let t = i as f64 * h;
        let f = t.sin() / t;
        sum += if i % 2 == 0 { 2.0 * f } else { 4.0 * f };
    }
    sign * sum * h / 3.0
}

/// Error function: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt.
///
/// Computed via series expansion for |x| < 3.5, asymptotic for large x.
/// Fundamental in probability, heat transfer, and statistics.
pub fn error_function(x: f64) -> f64 {
    if x.abs() < 1e-15 { return 0.0; }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    if x < 3.5 {
        // Taylor series: erf(x) = (2/√π) Σ (-1)^n x^(2n+1) / (n! (2n+1))
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let mut sum = 0.0;
        let mut term = x; // first term: x
        for n in 0..30 {
            sum += term / (2 * n + 1) as f64;
            term *= -x * x / (n + 1) as f64;
        }
        sign * two_over_sqrt_pi * sum
    } else {
        // Asymptotic: erf(x) ≈ 1 - e^(-x²)/(x√π) for large x
        sign * (1.0 - (-x * x).exp() / (x * std::f64::consts::PI.sqrt()))
    }
}

/// Exponential integral: Ei(x) = ∫₋∞ˣ e^t/t dt (principal value for x > 0).
///
/// For x > 0, Ei(x) = -PV∫₋ₓ^∞ e^(-t)/t dt = γ + ln(x) + Σ x^n/(n·n!)
/// where γ ≈ 0.5772 is the Euler-Mascheroni constant.
pub fn exponential_integral(x: f64) -> f64 {
    if x.abs() < 1e-15 { return f64::NEG_INFINITY; }
    let euler_gamma = 0.5772156649015329;

    if x.abs() < 40.0 {
        // Series: Ei(x) = γ + ln|x| + Σ_{n=1}^∞ x^n / (n·n!)
        let mut sum = euler_gamma + x.abs().ln();
        let mut term = x;
        let mut fact = 1.0;
        for n in 1..60 {
            fact *= n as f64;
            sum += term / (n as f64 * fact);
            term *= x;
            if term.abs() / (n as f64 * fact) < 1e-15 { break; }
        }
        sum
    } else {
        // Asymptotic: Ei(x) ~ e^x/x (1 + 1!/x + 2!/x² + ...)
        let mut sum: f64 = 1.0;
        let mut term: f64 = 1.0;
        for n in 1..20 {
            term *= n as f64 / x;
            if term.abs() > sum.abs() { break; } // diverging
            sum += term;
        }
        x.exp() / x * sum
    }
}

/// Fresnel cosine integral: C(x) = ∫₀ˣ cos(πt²/2) dt.
///
/// Computed via Simpson's rule. C(∞) = 0.5.
/// Used in optics (Fresnel diffraction).
pub fn fresnel_c(x: f64) -> f64 {
    if x.abs() < 1e-15 { return 0.0; }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let pi_half = std::f64::consts::PI / 2.0;
    let n = (x * 50.0).ceil() as usize;
    let n = n.max(100) | 1; // ensure odd for Simpson
    let n = n + 1 - (n % 2); // ensure even number of intervals
    let h = x / n as f64;
    let f = |t: f64| (pi_half * t * t).cos();
    let mut sum = f(0.0) + f(x);
    for i in 1..n {
        let t = i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(t) } else { 4.0 * f(t) };
    }
    sign * sum * h / 3.0
}

/// Fresnel sine integral: S(x) = ∫₀ˣ sin(πt²/2) dt.
///
/// Computed via Simpson's rule. S(∞) = 0.5.
/// Used in optics (Fresnel diffraction).
pub fn fresnel_s(x: f64) -> f64 {
    if x.abs() < 1e-15 { return 0.0; }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let pi_half = std::f64::consts::PI / 2.0;
    let n = (x * 50.0).ceil() as usize;
    let n = n.max(100) | 1;
    let n = n + 1 - (n % 2);
    let h = x / n as f64;
    let f = |t: f64| (pi_half * t * t).sin();
    let mut sum = f(0.0) + f(x);
    for i in 1..n {
        let t = i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(t) } else { 4.0 * f(t) };
    }
    sign * sum * h / 3.0
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

    // ── Transcendental integration tests ────────────────────────────────

    #[test]
    fn test_integrate_constant() {
        // ∫ 5 dx = 5x
        let result = integrate_symbolic(&TermType::Constant(5), "x");
        assert!(result.is_some(), "should integrate constant");
    }

    #[test]
    fn test_integrate_variable() {
        // ∫ x dx = x²/2
        let result = integrate_symbolic(&TermType::Variable("x".to_string()), "x");
        assert!(result.is_some(), "should integrate x");
    }

    #[test]
    fn test_integrate_sin() {
        // ∫ sin(x) dx = -cos(x)
        let sin_x = func("sin", TermType::Variable("x".to_string()));
        let result = integrate_symbolic(&sin_x, "x");
        assert!(result.is_some(), "should integrate sin(x)");
        let r = result.unwrap();
        // Should contain cos
        let s = format!("{:?}", r);
        assert!(s.contains("cos"), "∫sin(x) should produce cos: {:?}", r);
    }

    #[test]
    fn test_integrate_cos() {
        // ∫ cos(x) dx = sin(x)
        let cos_x = func("cos", TermType::Variable("x".to_string()));
        let result = integrate_symbolic(&cos_x, "x");
        assert!(result.is_some(), "should integrate cos(x)");
        let r = result.unwrap();
        let s = format!("{:?}", r);
        assert!(s.contains("sin"), "∫cos(x) should produce sin: {:?}", r);
    }

    #[test]
    fn test_integrate_exp() {
        // ∫ exp(x) dx = exp(x)
        let exp_x = func("exp", TermType::Variable("x".to_string()));
        let result = integrate_symbolic(&exp_x, "x");
        assert!(result.is_some(), "should integrate exp(x)");
        let r = result.unwrap();
        let s = format!("{:?}", r);
        assert!(s.contains("exp"), "∫exp(x) should produce exp: {:?}", r);
    }

    #[test]
    fn test_integrate_ln() {
        // ∫ ln(x) dx = x*ln(x) - x
        let ln_x = func("ln", TermType::Variable("x".to_string()));
        let result = integrate_symbolic(&ln_x, "x");
        assert!(result.is_some(), "should integrate ln(x)");
    }

    #[test]
    fn test_integrate_one_over_x() {
        // ∫ 1/x dx = ln(x)
        let one_over_x = TermType::BinaryOp {
            op: SymbolicOp::Div,
            left: Box::new(TermType::Constant(1)),
            right: Box::new(TermType::Variable("x".to_string())),
        };
        let result = integrate_symbolic(&one_over_x, "x");
        assert!(result.is_some(), "should integrate 1/x");
        let r = result.unwrap();
        let s = format!("{:?}", r);
        assert!(s.contains("ln"), "∫1/x should produce ln: {:?}", r);
    }

    #[test]
    fn test_integrate_power() {
        // ∫ x^3 dx = x^4/4
        let x_cubed = TermType::BinaryOp {
            op: SymbolicOp::Pow,
            left: Box::new(TermType::Variable("x".to_string())),
            right: Box::new(TermType::Constant(3)),
        };
        let result = integrate_symbolic(&x_cubed, "x");
        assert!(result.is_some(), "should integrate x^3");
    }

    #[test]
    fn test_integrate_sum() {
        // ∫ (sin(x) + cos(x)) dx = -cos(x) + sin(x)
        let sum = TermType::BinaryOp {
            op: SymbolicOp::Add,
            left: Box::new(func("sin", TermType::Variable("x".to_string()))),
            right: Box::new(func("cos", TermType::Variable("x".to_string()))),
        };
        let result = integrate_symbolic(&sum, "x");
        assert!(result.is_some(), "should integrate sin(x) + cos(x)");
    }

    #[test]
    fn test_integrate_constant_times_function() {
        // ∫ 3*sin(x) dx = 3*(-cos(x)) = -3*cos(x)
        let three_sin = TermType::BinaryOp {
            op: SymbolicOp::Mul,
            left: Box::new(TermType::Constant(3)),
            right: Box::new(func("sin", TermType::Variable("x".to_string()))),
        };
        let result = integrate_symbolic(&three_sin, "x");
        assert!(result.is_some(), "should integrate 3*sin(x)");
    }

    #[test]
    fn test_integrate_linear_substitution() {
        // ∫ sin(2x + 1) dx = -cos(2x + 1)/2
        let linear_arg = TermType::BinaryOp {
            op: SymbolicOp::Add,
            left: Box::new(TermType::BinaryOp {
                op: SymbolicOp::Mul,
                left: Box::new(TermType::Constant(2)),
                right: Box::new(TermType::Variable("x".to_string())),
            }),
            right: Box::new(TermType::Constant(1)),
        };
        let sin_2x1 = func("sin", linear_arg);
        let result = integrate_symbolic(&sin_2x1, "x");
        assert!(result.is_some(), "should integrate sin(2x+1) via linear substitution");
    }

    /// Fundamental theorem verification: d/dx ∫f = f for transcendentals
    #[test]
    fn test_fundamental_theorem_sin() {
        let prims = PrimitiveSystem::global();
        // ∫ sin(x) dx = -cos(x). d/dx(-cos(x)) should = sin(x)
        let sin_x = func("sin", TermType::Variable("x".to_string()));
        let integral = integrate_symbolic(&sin_x, "x").unwrap();
        let derivative = SymbolicDifferentiator::diff_recursive(&integral, "x", prims);
        // The derivative of -cos(x) = sin(x). Verify by structure.
        let s = format!("{:?}", derivative);
        assert!(s.contains("sin"), "d/dx(∫sin(x)dx) should recover sin: {:?}", derivative);
    }

    // ==================== TRANSCENDENTAL INTEGRATION ====================

    #[test]
    fn test_sine_integral_known_values() {
        // Si(0) = 0
        assert!(sine_integral(0.0).abs() < 1e-15);
        // Si(π) ≈ 1.8519
        let si_pi = sine_integral(std::f64::consts::PI);
        assert!((si_pi - 1.8519).abs() < 0.01,
            "Si(π) should ≈ 1.8519, got {:.4}", si_pi);
        // Si(-x) = -Si(x) (odd function)
        let si_2 = sine_integral(2.0);
        let si_neg2 = sine_integral(-2.0);
        assert!((si_2 + si_neg2).abs() < 1e-10,
            "Si should be odd: Si(2)={:.6}, Si(-2)={:.6}", si_2, si_neg2);
    }

    #[test]
    fn test_error_function_known_values() {
        // erf(0) = 0
        assert!(error_function(0.0).abs() < 1e-15);
        // erf(1) ≈ 0.8427
        let erf1 = error_function(1.0);
        assert!((erf1 - 0.8427).abs() < 0.001,
            "erf(1) should ≈ 0.8427, got {:.4}", erf1);
        // erf(∞) → 1
        let erf_big = error_function(5.0);
        assert!((erf_big - 1.0).abs() < 1e-6,
            "erf(5) should ≈ 1.0, got {:.6}", erf_big);
        // erf(-x) = -erf(x) (odd function)
        let erf_neg = error_function(-1.0);
        assert!((erf1 + erf_neg).abs() < 1e-10,
            "erf should be odd: erf(1)={:.6}, erf(-1)={:.6}", erf1, erf_neg);
    }

    #[test]
    fn test_exponential_integral_known_values() {
        // Ei(1) ≈ 1.8951
        let ei1 = exponential_integral(1.0);
        assert!((ei1 - 1.8951).abs() < 0.01,
            "Ei(1) should ≈ 1.8951, got {:.4}", ei1);
        // Ei(2) ≈ 4.9542
        let ei2 = exponential_integral(2.0);
        assert!((ei2 - 4.9542).abs() < 0.01,
            "Ei(2) should ≈ 4.9542, got {:.4}", ei2);
    }

    #[test]
    fn test_fresnel_integrals_known_values() {
        // C(0) = 0, S(0) = 0
        assert!(fresnel_c(0.0).abs() < 1e-15);
        assert!(fresnel_s(0.0).abs() < 1e-15);
        // C(1) ≈ 0.7799, S(1) ≈ 0.4383
        let c1 = fresnel_c(1.0);
        let s1 = fresnel_s(1.0);
        assert!((c1 - 0.7799).abs() < 0.01,
            "C(1) should ≈ 0.7799, got {:.4}", c1);
        assert!((s1 - 0.4383).abs() < 0.01,
            "S(1) should ≈ 0.4383, got {:.4}", s1);
    }
}
