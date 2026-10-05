use super::*;

#[derive(Debug, Clone)]
pub enum SymExpr {
    Var(String),
    Const(f64),
    Add(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Div(Box<SymExpr>, Box<SymExpr>),
    Neg(Box<SymExpr>),
    Pow(Box<SymExpr>, f64),
    Log(Box<SymExpr>),
    Sin(Box<SymExpr>),
    Cos(Box<SymExpr>),
}

impl SymExpr {
    pub fn eval(&self, vars: &[(&str, f64)]) -> f64 {
        match self {
            SymExpr::Var(name) => vars
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .map(|(_, v)| *v)
                .unwrap_or(0.0),
            SymExpr::Const(c) => *c,
            SymExpr::Add(a, b) => a.eval(vars) + b.eval(vars),
            SymExpr::Mul(a, b) => a.eval(vars) * b.eval(vars),
            SymExpr::Div(a, b) => {
                let bv = b.eval(vars);
                if bv.abs() > 1e-15 {
                    a.eval(vars) / bv
                } else {
                    f64::NAN
                }
            }
            SymExpr::Neg(a) => -a.eval(vars),
            SymExpr::Pow(base, exp) => base.eval(vars).powf(*exp),
            SymExpr::Log(a) => {
                let v = a.eval(vars);
                if v > 0.0 { v.ln() } else { f64::NAN }
            }
            SymExpr::Sin(a) => a.eval(vars).sin(),
            SymExpr::Cos(a) => a.eval(vars).cos(),
        }
    }

    pub fn diff(&self, var: &str) -> SymExpr {
        match self {
            SymExpr::Var(name) => {
                if name == var {
                    SymExpr::Const(1.0)
                } else {
                    SymExpr::Const(0.0)
                }
            }
            SymExpr::Const(_) => SymExpr::Const(0.0),
            SymExpr::Add(a, b) => SymExpr::Add(Box::new(a.diff(var)), Box::new(b.diff(var))),
            SymExpr::Mul(a, b) => SymExpr::Add(
                Box::new(SymExpr::Mul(Box::new(a.diff(var)), b.clone())),
                Box::new(SymExpr::Mul(a.clone(), Box::new(b.diff(var)))),
            ),
            SymExpr::Div(a, b) => SymExpr::Div(
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Mul(Box::new(a.diff(var)), b.clone())),
                    Box::new(SymExpr::Neg(Box::new(SymExpr::Mul(
                        a.clone(),
                        Box::new(b.diff(var)),
                    )))),
                )),
                Box::new(SymExpr::Pow(b.clone(), 2.0)),
            ),
            SymExpr::Neg(a) => SymExpr::Neg(Box::new(a.diff(var))),
            SymExpr::Pow(base, exp) => SymExpr::Mul(
                Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Const(*exp)),
                    Box::new(SymExpr::Pow(base.clone(), *exp - 1.0)),
                )),
                Box::new(base.diff(var)),
            ),
            SymExpr::Log(a) => SymExpr::Div(Box::new(a.diff(var)), a.clone()),
            SymExpr::Sin(a) => {
                SymExpr::Mul(Box::new(SymExpr::Cos(a.clone())), Box::new(a.diff(var)))
            }
            SymExpr::Cos(a) => SymExpr::Mul(
                Box::new(SymExpr::Neg(Box::new(SymExpr::Sin(a.clone())))),
                Box::new(a.diff(var)),
            ),
        }
    }

    pub fn simplify(&self) -> SymExpr {
        match self {
            SymExpr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymExpr::Const(x), _) if x.abs() < 1e-15 => b,
                    (_, SymExpr::Const(x)) if x.abs() < 1e-15 => a,
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymExpr::Const(x + y),
                    _ => SymExpr::Add(Box::new(a), Box::new(b)),
                }
            }
            SymExpr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymExpr::Const(x), _) if x.abs() < 1e-15 => SymExpr::Const(0.0),
                    (_, SymExpr::Const(x)) if x.abs() < 1e-15 => SymExpr::Const(0.0),
                    (SymExpr::Const(x), _) if (*x - 1.0).abs() < 1e-15 => b,
                    (_, SymExpr::Const(x)) if (*x - 1.0).abs() < 1e-15 => a,
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymExpr::Const(x * y),
                    _ => SymExpr::Mul(Box::new(a), Box::new(b)),
                }
            }
            SymExpr::Neg(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(x) => SymExpr::Const(-x),
                    SymExpr::Neg(inner) => *inner.clone(),
                    _ => SymExpr::Neg(Box::new(a)),
                }
            }
            SymExpr::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymExpr::Const(x), _) if x.abs() < 1e-15 => SymExpr::Const(0.0),
                    (_, SymExpr::Const(x)) if (*x - 1.0).abs() < 1e-15 => a,
                    (SymExpr::Const(x), SymExpr::Const(y)) if y.abs() > 1e-15 => {
                        SymExpr::Const(x / y)
                    }
                    _ => SymExpr::Div(Box::new(a), Box::new(b)),
                }
            }
            SymExpr::Pow(base, exp) => {
                let base = base.simplify();
                if (*exp - 1.0).abs() < 1e-15 {
                    return base;
                }
                if exp.abs() < 1e-15 {
                    return SymExpr::Const(1.0);
                }
                match &base {
                    SymExpr::Const(c) => SymExpr::Const(c.powf(*exp)),
                    _ => SymExpr::Pow(Box::new(base), *exp),
                }
            }
            SymExpr::Log(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) if *c > 0.0 => SymExpr::Const(c.ln()),
                    _ => SymExpr::Log(Box::new(a)),
                }
            }
            SymExpr::Sin(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) => SymExpr::Const(c.sin()),
                    _ => SymExpr::Sin(Box::new(a)),
                }
            }
            SymExpr::Cos(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) => SymExpr::Const(c.cos()),
                    _ => SymExpr::Cos(Box::new(a)),
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymExpr::Var(name) => write!(f, "{}", name),
            SymExpr::Const(c) => {
                if (*c - c.round()).abs() < 1e-10 {
                    write!(f, "{}", *c as i64)
                } else {
                    write!(f, "{:.4}", c)
                }
            }
            SymExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            SymExpr::Mul(a, b) => write!(f, "({} · {})", a, b),
            SymExpr::Div(a, b) => write!(f, "({}/{})", a, b),
            SymExpr::Neg(a) => write!(f, "(-{})", a),
            SymExpr::Pow(base, exp) => write!(f, "{}^{}", base, exp),
            SymExpr::Log(a) => write!(f, "ln({})", a),
            SymExpr::Sin(a) => write!(f, "sin({})", a),
            SymExpr::Cos(a) => write!(f, "cos({})", a),
        }
    }
}

/// Evidence produced by differentiating a candidate conservation law against
/// symbolic dynamics and then evaluating the resulting derivative at six fixed
/// numeric points.
///
/// This is deliberately an *assessment*, not a proof: finite-point sampling
/// cannot establish a universal conservation law. Formal proof status belongs
/// to a separate proof backend (for example, the Z3 path for supported
/// polynomial statements).
#[derive(Debug)]
pub struct ConservationCheck {
    pub quantity: String,
    pub total_derivative: String,
    /// True only when the current simplifier reduces the full derivative to a
    /// constant numerical zero. This is structural evidence about the current
    /// simplifier, not a universal proof.
    pub symbolic_derivative_simplified_to_zero: bool,
    /// True when the derived symbolic derivative has absolute residual below
    /// 1e-10 at all six fixed numeric test points.
    pub sampled_residual_passed: bool,
    pub max_numerical_residual: f64,
}

impl fmt::Display for ConservationCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Conservation assessment for {}:", self.quantity)?;
        writeln!(f, "  dE/dt = {}", self.total_derivative)?;
        writeln!(
            f,
            "  Simplified to zero: {}",
            if self.symbolic_derivative_simplified_to_zero { "YES" } else { "NO" }
        )?;
        writeln!(
            f,
            "  Six-point residual check: {} (max residual: {:.2e})",
            if self.sampled_residual_passed { "PASS" } else { "FAIL" },
            self.max_numerical_residual
        )?;
        Ok(())
    }
}

/// Differentiate a candidate conservation law against the supplied symbolic
/// dynamics and assess the result using both the local simplifier and six
/// fixed numeric sample points. The sample points are evidence only; they do
/// not establish a universal identity.
pub fn assess_conservation_symbolic(
    energy: &SymExpr,
    dynamics: &[(&str, SymExpr)],
) -> ConservationCheck {
    let mut total_deriv = SymExpr::Const(0.0);
    for (var, dvar_dt) in dynamics {
        let partial = energy.diff(var).simplify();
        let term = SymExpr::Mul(Box::new(partial), Box::new(dvar_dt.clone()));
        total_deriv = SymExpr::Add(Box::new(total_deriv), Box::new(term));
    }
    let total_deriv = total_deriv.simplify();
    let symbolic_derivative_simplified_to_zero =
        matches!(total_deriv, SymExpr::Const(c) if c.abs() < 1e-15);
    let test_points: Vec<Vec<(&str, f64)>> = vec![
        vec![("x", 1.0), ("v", 0.0)],
        vec![("x", 0.0), ("v", 1.0)],
        vec![
            ("x", std::f64::consts::FRAC_1_SQRT_2),
            ("v", std::f64::consts::FRAC_1_SQRT_2),
        ],
        vec![("x", -1.0), ("v", 0.5)],
        vec![("x", 0.3), ("v", -0.9)],
        vec![("x", 2.0), ("v", -1.5)],
    ];
    let max_residual = test_points
        .iter()
        .map(|pt| total_deriv.eval(pt).abs())
        .fold(0.0f64, f64::max);

    ConservationCheck {
        quantity: format!("{}", energy),
        total_derivative: format!("{}", total_deriv),
        symbolic_derivative_simplified_to_zero,
        sampled_residual_passed: max_residual < 1e-10,
        max_numerical_residual: max_residual,
    }
}

/// Backward-compatible name for assess_conservation_symbolic.
///
/// Deprecated because the former name implied a proof while the implementation
/// provides finite-point numerical evidence plus a local symbolic simplifier.
#[deprecated(note = "use assess_conservation_symbolic; this check is evidence, not a proof")]
pub fn verify_conservation_symbolic(
    energy: &SymExpr,
    dynamics: &[(&str, SymExpr)],
) -> ConservationCheck {
    assess_conservation_symbolic(energy, dynamics)
}

/// Backward-compatible type name for callers that referred to the former
/// proof-shaped result. The fields now expose assessment evidence explicitly.
#[deprecated(note = "use ConservationCheck; the result is evidence, not a proof")]
pub type ConservationProof = ConservationCheck;

pub fn expr_to_sym(expr: &Expr) -> Option<SymExpr> {
    match expr {
        Expr::Var(name) => Some(SymExpr::Var(name.clone())),
        Expr::Const(c) => Some(SymExpr::Const(*c)),
        Expr::BinOp(op, left, right) => {
            let l = expr_to_sym(left)?;
            let r = expr_to_sym(right)?;
            match op {
                BinOp::Add => Some(SymExpr::Add(Box::new(l), Box::new(r))),
                BinOp::Sub => Some(SymExpr::Add(
                    Box::new(l),
                    Box::new(SymExpr::Neg(Box::new(r))),
                )),
                BinOp::Mul => Some(SymExpr::Mul(Box::new(l), Box::new(r))),
                BinOp::Pow => {
                    if let Expr::Const(exp) = right.as_ref() {
                        Some(SymExpr::Pow(Box::new(l), *exp))
                    } else {
                        None
                    }
                }
                BinOp::Div => Some(SymExpr::Mul(
                    Box::new(l),
                    Box::new(SymExpr::Pow(Box::new(r), -1.0)),
                )),
            }
        }
        Expr::Func(f, arg) => {
            let a = expr_to_sym(arg)?;
            match f {
                UnaryFn::Sin => Some(SymExpr::Sin(Box::new(a))),
                UnaryFn::Cos => Some(SymExpr::Cos(Box::new(a))),
                UnaryFn::Log => Some(SymExpr::Log(Box::new(a))),
                UnaryFn::Sqrt => Some(SymExpr::Pow(Box::new(a), 0.5)),
                UnaryFn::Exp | UnaryFn::Abs | UnaryFn::Floor => None,
            }
        }
        Expr::Sum(_, _) => None,
    }
}

pub fn verify_formula_derivative(
    expr: &Expr,
    data: &[(f64, f64)],
    var: &str,
) -> Option<DerivativeVerification> {
    let sym = expr_to_sym(expr)?;
    let deriv = sym.diff(var).simplify();

    let mut max_rel_error = 0.0f64;
    let mut checked = 0;
    for w in data.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let dx = x1 - x0;
        if dx.abs() < 1e-15 {
            continue;
        }
        let finite_diff = (y1 - y0) / dx;
        let midpoint = (x0 + x1) / 2.0;
        let symbolic_val = deriv.eval(&[(var, midpoint)]);
        if symbolic_val.is_finite() && finite_diff.abs() > 1e-10 {
            let rel_err = (symbolic_val - finite_diff).abs() / finite_diff.abs();
            max_rel_error = max_rel_error.max(rel_err);
            checked += 1;
        }
    }

    if checked == 0 {
        return None;
    }

    Some(DerivativeVerification {
        derivative_str: format!("{}", deriv),
        max_relative_error: max_rel_error,
        is_consistent: max_rel_error < 0.2,
    })
}

#[derive(Debug)]
pub struct DerivativeVerification {
    pub derivative_str: String,
    pub max_relative_error: f64,
    pub is_consistent: bool,
}

#[cfg(test)]
mod conservation_evidence_tests {
    use super::*;

    fn harmonic_dynamics() -> Vec<(&'static str, SymExpr)> {
        vec![
            ("x", SymExpr::Var("v".into())),
            ("v", SymExpr::Neg(Box::new(SymExpr::Var("x".into())))),
        ]
    }

    #[test]
    fn assessment_does_not_report_sampling_as_symbolic_proof() {
        let energy = SymExpr::Add(
            Box::new(SymExpr::Pow(Box::new(SymExpr::Var("x".into())), 2.0)),
            Box::new(SymExpr::Pow(Box::new(SymExpr::Var("v".into())), 2.0)),
        );
        let check = assess_conservation_symbolic(&energy, &harmonic_dynamics());

        assert!(check.sampled_residual_passed);
        assert!(!check.symbolic_derivative_simplified_to_zero);
        assert!(check.max_numerical_residual < 1e-10);
    }

    #[test]
    fn non_conserved_candidate_fails_sampled_residual_check() {
        let energy = SymExpr::Var("x".into());
        let check = assess_conservation_symbolic(&energy, &harmonic_dynamics());

        assert!(!check.sampled_residual_passed);
        assert!(check.max_numerical_residual > 0.0);
    }
}
