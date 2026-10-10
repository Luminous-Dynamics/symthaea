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

/// Failure modes for strict symbolic-expression evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymExprEvalError {
    MissingVariable(String),
    DuplicateVariableBinding(String),
    NonFiniteVariableBinding(String),
    DivisionByZero,
    LogDomain,
    IndeterminatePower,
    NonFiniteResult,
}

impl std::fmt::Display for SymExprEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVariable(name) => write!(f, "missing variable binding: {name}"),
            Self::DuplicateVariableBinding(name) => {
                write!(f, "duplicate variable binding: {name}")
            }
            Self::NonFiniteVariableBinding(name) => {
                write!(f, "non-finite variable binding: {name}")
            }
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::LogDomain => write!(f, "logarithm requires a positive argument"),
            Self::IndeterminatePower => {
                write!(f, "zero raised to the zero power is indeterminate")
            }
            Self::NonFiniteResult => write!(f, "expression produced a non-finite result"),
        }
    }
}

impl std::error::Error for SymExprEvalError {}

impl SymExpr {
    /// Evaluate without silently replacing missing variables with zero.
    ///
    /// This checked API rejects missing/duplicate/non-finite bindings,
    /// division by zero, invalid logarithm arguments, and non-finite results.
    /// It is the required path for evidence-producing mathematical checks.
    pub fn eval_checked(&self, vars: &[(&str, f64)]) -> Result<f64, SymExprEvalError> {
        // Validate the complete environment, including bindings unused by this
        // expression. Strict evaluation must not accept malformed hidden inputs.
        for (index, (name, value)) in vars.iter().enumerate() {
            if !value.is_finite() {
                return Err(SymExprEvalError::NonFiniteVariableBinding(
                    (*name).to_string(),
                ));
            }
            if vars[index + 1..]
                .iter()
                .any(|(other, _)| *other == *name)
            {
                return Err(SymExprEvalError::DuplicateVariableBinding(
                    (*name).to_string(),
                ));
            }
        }

        let value = match self {
            SymExpr::Var(name) => {
                let mut matches = vars
                    .iter()
                    .filter(|(candidate, _)| *candidate == name.as_str());
                let (_, value) = matches
                    .next()
                    .ok_or_else(|| SymExprEvalError::MissingVariable(name.clone()))?;
                if matches.next().is_some() {
                    return Err(SymExprEvalError::DuplicateVariableBinding(name.clone()));
                }
                if !value.is_finite() {
                    return Err(SymExprEvalError::NonFiniteVariableBinding(name.clone()));
                }
                *value
            }
            SymExpr::Const(value) => *value,
            SymExpr::Add(left, right) => {
                left.eval_checked(vars)? + right.eval_checked(vars)?
            }
            SymExpr::Mul(left, right) => {
                left.eval_checked(vars)? * right.eval_checked(vars)?
            }
            SymExpr::Div(left, right) => {
                let numerator = left.eval_checked(vars)?;
                let denominator = right.eval_checked(vars)?;
                if denominator == 0.0 {
                    return Err(SymExprEvalError::DivisionByZero);
                }
                numerator / denominator
            }
            SymExpr::Neg(inner) => -inner.eval_checked(vars)?,
            SymExpr::Pow(base, exponent) => {
                if !exponent.is_finite() {
                    return Err(SymExprEvalError::NonFiniteResult);
                }
                let base_value = base.eval_checked(vars)?;
                if *exponent == 0.0 && base_value == 0.0 {
                    return Err(SymExprEvalError::IndeterminatePower);
                }
                base_value.powf(*exponent)
            },
            SymExpr::Log(inner) => {
                let argument = inner.eval_checked(vars)?;
                if argument <= 0.0 {
                    return Err(SymExprEvalError::LogDomain);
                }
                argument.ln()
            }
            SymExpr::Sin(inner) => inner.eval_checked(vars)?.sin(),
            SymExpr::Cos(inner) => inner.eval_checked(vars)?.cos(),
        };

        if value.is_finite() {
            Ok(value)
        } else {
            Err(SymExprEvalError::NonFiniteResult)
        }
    }

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
                if bv != 0.0 {
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

    /// Whether the expression is defined over all real-valued variable bindings.
    ///
    /// This is deliberately conservative: division by a variable, logarithms
    /// of variables, negative powers, and non-integer powers may be undefined
    /// for some real inputs. Zero-product rewrites must not erase those domain
    /// restrictions.
    fn is_total_over_reals(&self) -> bool {
        match self {
            SymExpr::Var(_) => true,
            SymExpr::Const(value) => value.is_finite(),
            SymExpr::Add(a, b) | SymExpr::Mul(a, b) => {
                a.is_total_over_reals() && b.is_total_over_reals()
            }
            SymExpr::Div(numerator, denominator) => {
                numerator.is_total_over_reals()
                    && matches!(
                        denominator.as_ref(),
                        SymExpr::Const(value) if value.is_finite() && *value != 0.0
                    )
            }
            SymExpr::Neg(inner) => inner.is_total_over_reals(),
            SymExpr::Pow(base, exponent) => {
                exponent.is_finite()
                    && *exponent >= 1.0
                    && exponent.fract() == 0.0
                    && base.is_total_over_reals()
            }
            SymExpr::Log(inner) => matches!(
                inner.as_ref(),
                SymExpr::Const(value) if value.is_finite() && *value > 0.0
            ),
            SymExpr::Sin(inner) | SymExpr::Cos(inner) => inner.is_total_over_reals(),
        }
    }

    /// Whether the expression is structurally a finite, nonzero constant.
    fn is_provably_nonzero(&self) -> bool {
        matches!(self, SymExpr::Const(value) if value.is_finite() && *value != 0.0)
    }

    /// Apply exact algebraic identities without treating small floating-point
    /// values as zero or one. Rewrites that discard a subtree preserve the
    /// domain of partial expressions (for example, 0 * ln(x) is not erased).
    pub fn simplify(&self) -> SymExpr {
        match self {
            SymExpr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymExpr::Const(x), _) if *x == 0.0 => b,
                    (_, SymExpr::Const(x)) if *x == 0.0 => a,
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymExpr::Const(x + y),
                    _ => SymExpr::Add(Box::new(a), Box::new(b)),
                }
            }
            SymExpr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymExpr::Const(x), _) if *x == 0.0 && b.is_total_over_reals() => {
                        SymExpr::Const(0.0)
                    }
                    (_, SymExpr::Const(x)) if *x == 0.0 && a.is_total_over_reals() => {
                        SymExpr::Const(0.0)
                    }
                    (SymExpr::Const(x), _) if *x == 1.0 => b,
                    (_, SymExpr::Const(x)) if *x == 1.0 => a,
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
                    (SymExpr::Const(x), SymExpr::Const(y))
                        if y.is_finite() && *y != 0.0 =>
                    {
                        SymExpr::Const(x / y)
                    }
                    (SymExpr::Const(x), _) if *x == 0.0 && b.is_provably_nonzero() => {
                        SymExpr::Const(0.0)
                    }
                    (_, SymExpr::Const(x)) if *x == 1.0 => a,
                    _ => SymExpr::Div(Box::new(a), Box::new(b)),
                }
            }
            SymExpr::Pow(base, exp) => {
                let base = base.simplify();
                if *exp == 1.0 {
                    return base;
                }
                if *exp == 0.0 && base.is_provably_nonzero() {
                    return SymExpr::Const(1.0);
                }
                match &base {
                    SymExpr::Const(c) if *exp == 0.0 && *c == 0.0 => {
                        SymExpr::Pow(Box::new(base), *exp)
                    }
                    SymExpr::Const(c)
                        if c.is_finite()
                            && exp.is_finite()
                            && c.powf(*exp).is_finite() =>
                    {
                        SymExpr::Const(c.powf(*exp))
                    }
                    _ => SymExpr::Pow(Box::new(base), *exp),
                }
            }
            SymExpr::Log(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) if c.is_finite() && *c > 0.0 => {
                        SymExpr::Const(c.ln())
                    }
                    _ => SymExpr::Log(Box::new(a)),
                }
            }
            SymExpr::Sin(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) if c.is_finite() => SymExpr::Const(c.sin()),
                    _ => SymExpr::Sin(Box::new(a)),
                }
            }
            SymExpr::Cos(a) => {
                let a = a.simplify();
                match &a {
                    SymExpr::Const(c) if c.is_finite() => SymExpr::Const(c.cos()),
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
    /// True only when every sample has complete bindings, respects expression
    /// domains, and evaluates to a finite value. Any evaluation error fails closed.
    pub sampled_evaluations_valid: bool,
    /// True when every sample evaluation is valid and its absolute residual is
    /// below 1e-10 at all six fixed numeric test points.
    pub sampled_residual_passed: bool,
    /// Infinity denotes invalid sampling (for example, a missing binding or
    /// a non-finite/domain-invalid residual), not a successful zero residual.
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
            "  Sample evaluations valid: {}",
            if self.sampled_evaluations_valid { "YES" } else { "NO" }
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
        matches!(&total_deriv, SymExpr::Const(c) if *c == 0.0);
    // Bind every declared state variable at every sample point. Positive,
    // nonzero base values support common restricted domains (logs and negative
    // powers); each variable receives a deterministic rotation to avoid
    // restricting multivariate checks to the x=v diagonal. This is not full
    // domain inference: invalid or unbound evaluations still fail closed.
    const BASE_SAMPLE_VALUES: [f64; 6] = [
        1.0,
        0.5,
        std::f64::consts::FRAC_1_SQRT_2,
        2.0,
        0.3,
        3.0,
    ];
    let test_points: Vec<Vec<(&str, f64)>> = (0..BASE_SAMPLE_VALUES.len())
        .map(|sample_index| {
            dynamics
                .iter()
                .enumerate()
                .map(|(variable_index, (name, _))| {
                    (
                        *name,
                            BASE_SAMPLE_VALUES
                                [(sample_index + variable_index) % BASE_SAMPLE_VALUES.len()],
                    )
                })
                .collect()
        })
        .collect();
    let sampled_residuals: Result<Vec<f64>, SymExprEvalError> = test_points
        .iter()
        .map(|point| total_deriv.eval_checked(point))
        .collect();
    let (sampled_evaluations_valid, max_residual) = match sampled_residuals {
        Ok(values) if !dynamics.is_empty() => (
            true,
            values.iter().map(|value| value.abs()).fold(0.0f64, f64::max),
        ),
        _ => (false, f64::INFINITY),
    };

    ConservationCheck {
        quantity: format!("{}", energy),
        total_derivative: format!("{}", total_deriv),
        symbolic_derivative_simplified_to_zero,
        sampled_evaluations_valid,
        sampled_residual_passed: sampled_evaluations_valid && max_residual < 1e-10,
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
    fn checked_eval_rejects_missing_variable_binding() {
        let expr = SymExpr::Var("y".into());
        assert_eq!(
            expr.eval_checked(&[("x", 1.0)]),
            Err(SymExprEvalError::MissingVariable("y".into()))
        );
    }

    #[test]
    fn checked_eval_rejects_duplicate_and_non_finite_bindings() {
        let expr = SymExpr::Var("x".into());
        assert_eq!(
            expr.eval_checked(&[("x", 1.0), ("x", 2.0)]),
            Err(SymExprEvalError::DuplicateVariableBinding("x".into()))
        );
        assert_eq!(
            expr.eval_checked(&[("x", f64::NAN)]),
            Err(SymExprEvalError::NonFiniteVariableBinding("x".into()))
        );
    }

    #[test]
    fn checked_eval_rejects_unused_duplicate_or_non_finite_bindings() {
        let constant = SymExpr::Const(2.0);
        assert_eq!(
            constant.eval_checked(&[("unused", 1.0), ("unused", 2.0)]),
            Err(SymExprEvalError::DuplicateVariableBinding("unused".into()))
        );
        assert_eq!(
            constant.eval_checked(&[("unused", f64::NAN)]),
            Err(SymExprEvalError::NonFiniteVariableBinding("unused".into()))
        );
    }

    #[test]
    fn conservation_assessment_fails_when_a_state_variable_is_unbound() {
        let energy = SymExpr::Mul(
            Box::new(SymExpr::Var("x".into())),
            Box::new(SymExpr::Var("y".into())),
        );
        let dynamics = [("x", SymExpr::Const(1.0))];

        let check = assess_conservation_symbolic(&energy, &dynamics);

        assert!(!check.sampled_evaluations_valid);
        assert!(!check.sampled_residual_passed);
        assert!(check.max_numerical_residual.is_infinite());
    }

    #[test]
    fn fixed_sample_points_can_miss_a_nonzero_derivative() {
        // Construct a nonzero polynomial whose roots are exactly the assessor's
        // six fixed sample x-coordinates. This demonstrates why sampled success
        // cannot be promoted to a universal conservation claim.
        let roots = [
            1.0,
            0.5,
            std::f64::consts::FRAC_1_SQRT_2,
            2.0,
            0.3,
            3.0,
        ];
        let mut rhs = SymExpr::Const(1.0);
        for root in roots {
            let factor = SymExpr::Add(
                Box::new(SymExpr::Var("x".into())),
                Box::new(SymExpr::Neg(Box::new(SymExpr::Const(root)))),
            );
            rhs = SymExpr::Mul(Box::new(rhs), Box::new(factor));
        }

        let check = assess_conservation_symbolic(
            &SymExpr::Var("x".into()),
            &[("x", rhs.clone())],
        );

        assert!(check.sampled_residual_passed);
        assert!(!check.symbolic_derivative_simplified_to_zero);
        assert!(check.max_numerical_residual == 0.0);
        assert!(rhs.eval(&[("x", 4.0)]).abs() > 1e-10);
    }

    #[test]
    fn tiny_nonzero_constant_is_not_marked_as_structural_zero() {
        let check = assess_conservation_symbolic(
            &SymExpr::Var("x".into()),
            &[("x", SymExpr::Const(1e-16))],
        );

        // The finite residual threshold accepts this value, but exact structural
        // zero must not be inferred from an epsilon comparison.
        assert!(check.sampled_residual_passed);
        assert!(!check.symbolic_derivative_simplified_to_zero);
    }

    #[test]
    fn non_finite_sample_evaluation_fails_closed() {
        // x - x is zero at every input, so this derivative is undefined at
        // every sample. Checked evaluation must fail closed on the domain error.
        let denominator = SymExpr::Add(
            Box::new(SymExpr::Var("x".into())),
            Box::new(SymExpr::Neg(Box::new(SymExpr::Var("x".into())))),
        );
        let rhs = SymExpr::Div(
            Box::new(SymExpr::Var("x".into())),
            Box::new(denominator),
        );
        let check = assess_conservation_symbolic(
            &SymExpr::Var("x".into()),
            &[("x", rhs)],
        );

        assert!(!check.sampled_evaluations_valid);
        assert!(!check.sampled_residual_passed);
        assert!(check.max_numerical_residual.is_infinite());
    }

    #[test]
    fn simplify_preserves_small_nonzero_additive_constant() {
        let expr = SymExpr::Add(
            Box::new(SymExpr::Var("x".into())),
            Box::new(SymExpr::Const(1e-16)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymExpr::Add(_, _)));
        assert_eq!(simplified.eval_checked(&[("x", 0.0)]), Ok(1e-16));
    }

    #[test]
    fn simplify_preserves_near_one_multiplicative_constant() {
        let near_one = 1.0 + f64::EPSILON;
        let expr = SymExpr::Mul(
            Box::new(SymExpr::Const(near_one)),
            Box::new(SymExpr::Var("x".into())),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymExpr::Mul(_, _)));
        assert_eq!(simplified.eval_checked(&[("x", 1.0)]), Ok(near_one));
    }

    #[test]
    fn simplify_does_not_erase_denominator_domain_of_zero_over_x() {
        let expr = SymExpr::Div(
            Box::new(SymExpr::Const(0.0)),
            Box::new(SymExpr::Var("x".into())),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymExpr::Div(_, _)));
        assert_eq!(
            simplified.eval_checked(&[("x", 0.0)]),
            Err(SymExprEvalError::DivisionByZero)
        );
        assert_eq!(simplified.eval_checked(&[("x", 2.0)]), Ok(0.0));
    }

    #[test]
    fn simplify_does_not_erase_log_domain_in_zero_product() {
        let expr = SymExpr::Mul(
            Box::new(SymExpr::Const(0.0)),
            Box::new(SymExpr::Log(Box::new(SymExpr::Var("x".into())))),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymExpr::Mul(_, _)));
        assert_eq!(
            simplified.eval_checked(&[("x", -1.0)]),
            Err(SymExprEvalError::LogDomain)
        );
        assert_eq!(simplified.eval_checked(&[("x", 2.0)]), Ok(0.0));
    }

    #[test]
    fn simplify_preserves_undefined_constant_power() {
        let expr = SymExpr::Pow(Box::new(SymExpr::Const(-1.0)), 0.5);
        let simplified = expr.simplify();

        assert!(matches!(simplified, SymExpr::Pow(_, exponent) if exponent == 0.5));
        assert_eq!(
            simplified.eval_checked(&[]),
            Err(SymExprEvalError::NonFiniteResult)
        );
    }

    #[test]
    fn checked_eval_rejects_zero_to_the_zero_power() {
        let expr = SymExpr::Pow(Box::new(SymExpr::Const(0.0)), 0.0);
        assert_eq!(
            expr.eval_checked(&[]),
            Err(SymExprEvalError::IndeterminatePower)
        );
        assert!(matches!(expr.simplify(), SymExpr::Pow(_, _)));
    }

    #[test]
    fn simplify_preserves_small_nonzero_power_exponent() {
        let expr = SymExpr::Pow(Box::new(SymExpr::Var("x".into())), 1e-16);
        assert!(matches!(expr.simplify(), SymExpr::Pow(_, exponent) if exponent == 1e-16));
    }

    #[test]
    fn simplify_preserves_near_one_power_exponent() {
        let exponent = 1.0 + f64::EPSILON;
        let expr = SymExpr::Pow(Box::new(SymExpr::Var("x".into())), exponent);
        assert!(matches!(expr.simplify(), SymExpr::Pow(_, actual) if actual == exponent));
    }

    #[test]
    fn simplify_preserves_near_one_divisor() {
        let denominator = 1.0 + f64::EPSILON;
        let expr = SymExpr::Div(
            Box::new(SymExpr::Var("x".into())),
            Box::new(SymExpr::Const(denominator)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, SymExpr::Div(_, _)));
        assert_eq!(
            simplified.eval_checked(&[("x", 1.0)]),
            Ok(1.0 / denominator)
        );
    }

    #[test]
    fn legacy_eval_divides_by_small_nonzero_denominator() {
        let expr = SymExpr::Div(
            Box::new(SymExpr::Const(1.0)),
            Box::new(SymExpr::Const(1e-16)),
        );
        assert_eq!(expr.eval(&[]), 1e16);
    }

    #[test]
    fn small_nonzero_divisor_is_not_treated_as_zero() {
        let expr = SymExpr::Div(
            Box::new(SymExpr::Const(1.0)),
            Box::new(SymExpr::Const(1e-16)),
        );
        let simplified = expr.simplify();
        assert_eq!(simplified.eval_checked(&[]), Ok(1e16));
    }

    #[test]
    fn non_conserved_candidate_fails_sampled_residual_check() {
        let energy = SymExpr::Var("x".into());
        let check = assess_conservation_symbolic(&energy, &harmonic_dynamics());

        assert!(!check.sampled_residual_passed);
        assert!(check.max_numerical_residual > 0.0);
    }
}
