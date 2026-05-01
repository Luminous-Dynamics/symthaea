// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::fmt;

/// A symbolic expression tree for regression.
///
/// Kept simple and self-contained (no PrimitiveSystem dependency) for fast
/// genetic programming. Convert to `SymbolicExpr` for HDC encoding when needed.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Named variable (typically "n" for sequences)
    Var(String),
    /// Floating-point constant
    Const(f64),
    /// Binary operation
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    /// Unary function
    Func(UnaryFn, Box<Expr>),
    /// Summation: Σ_{k=1}^{n} body(k)
    Sum(Box<Expr>, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    Sqrt,
    Log,
    Exp,
    Sin,
    Cos,
    Abs,
    Floor,
}

impl Expr {
    /// Evaluate the expression with variable bindings.
    pub fn eval(&self, vars: &[(&str, f64)]) -> f64 {
        match self {
            Expr::Var(name) => vars
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .map(|(_, v)| *v)
                .unwrap_or(f64::NAN),
            Expr::Const(c) => *c,
            Expr::BinOp(op, left, right) => {
                let l = left.eval(vars);
                let r = right.eval(vars);
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => {
                        if r.abs() > 1e-15 {
                            l / r
                        } else {
                            f64::NAN
                        }
                    }
                    BinOp::Pow => l.powf(r),
                }
            }
            Expr::Func(f, arg) => {
                let x = arg.eval(vars);
                match f {
                    UnaryFn::Sqrt => {
                        if x >= 0.0 {
                            x.sqrt()
                        } else {
                            f64::NAN
                        }
                    }
                    UnaryFn::Log => {
                        if x > 0.0 {
                            x.ln()
                        } else {
                            f64::NAN
                        }
                    }
                    UnaryFn::Exp => x.exp(),
                    UnaryFn::Sin => x.sin(),
                    UnaryFn::Cos => x.cos(),
                    UnaryFn::Abs => x.abs(),
                    UnaryFn::Floor => x.floor(),
                }
            }
            Expr::Sum(body, var_name) => {
                let n = vars
                    .iter()
                    .find(|(name, _)| *name == "n")
                    .map(|(_, v)| *v as usize)
                    .unwrap_or(0)
                    .min(100);
                let mut sum = 0.0f64;
                for k in 0..=n {
                    let mut inner_vars: Vec<(&str, f64)> = vars.to_vec();
                    inner_vars.push((var_name.as_str(), k as f64));
                    sum += body.eval(&inner_vars);
                    if !sum.is_finite() {
                        return f64::NAN;
                    }
                }
                sum
            }
        }
    }

    /// Count AST nodes (complexity metric for Occam penalty).
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.complexity() + r.complexity(),
            Expr::Func(_, arg) => 1 + arg.complexity(),
            Expr::Sum(body, _) => 2 + body.complexity(),
        }
    }

    /// Deep clone with a random subtree replaced (mutation).
    pub fn mutate(&self, rng: &mut u64, depth: usize) -> Expr {
        *rng = lcg_step(*rng);
        let p = 1.0 / (1.0 + depth as f64);
        if (*rng as f64 / u64::MAX as f64) < p {
            return random_expr(rng, 2);
        }
        match self {
            Expr::Var(_) | Expr::Const(_) => random_expr(rng, 1),
            Expr::BinOp(op, l, r) => {
                *rng = lcg_step(*rng);
                if (*rng & 1) == 0 {
                    Expr::BinOp(*op, Box::new(l.mutate(rng, depth + 1)), r.clone())
                } else {
                    Expr::BinOp(*op, l.clone(), Box::new(r.mutate(rng, depth + 1)))
                }
            }
            Expr::Func(f, arg) => Expr::Func(*f, Box::new(arg.mutate(rng, depth + 1))),
            Expr::Sum(body, var) => Expr::Sum(Box::new(body.mutate(rng, depth + 1)), var.clone()),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(c) => {
                let pi = std::f64::consts::PI;
                let e = std::f64::consts::E;
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                if (*c - pi).abs() < 1e-10 {
                    write!(f, "π")
                } else if (*c - e).abs() < 1e-10 {
                    write!(f, "e")
                } else if (*c - phi).abs() < 1e-10 {
                    write!(f, "φ")
                } else if (*c - c.round()).abs() < 1e-10 && c.abs() < 1e12 {
                    write!(f, "{}", *c as i64)
                } else {
                    write!(f, "{:.6}", c)
                }
            }
            Expr::BinOp(op, l, r) => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Pow => "^",
                };
                write!(f, "({} {} {})", l, sym, r)
            }
            Expr::Sum(body, var) => write!(f, "Σ_{}({})", var, body),
            Expr::Func(func, arg) => {
                let name = match func {
                    UnaryFn::Sqrt => "sqrt",
                    UnaryFn::Log => "ln",
                    UnaryFn::Exp => "exp",
                    UnaryFn::Sin => "sin",
                    UnaryFn::Cos => "cos",
                    UnaryFn::Abs => "abs",
                    UnaryFn::Floor => "floor",
                };
                write!(f, "{}({})", name, arg)
            }
        }
    }
}

/// Convert an expression tree to paper-ready LaTeX.
pub fn expr_to_latex(expr: &Expr) -> String {
    fn precedence(e: &Expr) -> u8 {
        match e {
            Expr::Var(_) | Expr::Const(_) | Expr::Func(_, _) | Expr::Sum(_, _) => 10,
            Expr::BinOp(BinOp::Pow, _, _) => 8,
            Expr::BinOp(BinOp::Mul, _, _) | Expr::BinOp(BinOp::Div, _, _) => 6,
            Expr::BinOp(BinOp::Add, _, _) | Expr::BinOp(BinOp::Sub, _, _) => 4,
        }
    }

    fn wrap_if_lower(child: &Expr, parent_prec: u8) -> String {
        let child_latex = render(child);
        if precedence(child) < parent_prec {
            format!("\\left({}\\right)", child_latex)
        } else {
            child_latex
        }
    }

    fn const_to_latex(c: f64) -> String {
        let pi = std::f64::consts::PI;
        let e_const = std::f64::consts::E;
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

        if (c - pi).abs() < 1e-10 {
            return "\\pi".to_string();
        }
        if (c + pi).abs() < 1e-10 {
            return "-\\pi".to_string();
        }
        if (c - e_const).abs() < 1e-10 {
            return "e".to_string();
        }
        if (c - phi).abs() < 1e-10 {
            return "\\varphi".to_string();
        }
        if (c - 1.0 / e_const).abs() < 1e-10 {
            return "e^{-1}".to_string();
        }
        if (c - std::f64::consts::SQRT_2).abs() < 1e-10 {
            return "\\sqrt{2}".to_string();
        }
        if (c - 1.0 / std::f64::consts::PI.sqrt()).abs() < 1e-10 {
            return "\\frac{1}{\\sqrt{\\pi}}".to_string();
        }

        if (c - c.round()).abs() < 1e-10 && c.abs() < 1e12 {
            return format!("{}", c as i64);
        }
        for d in 2..=12 {
            for n in 1..=(d * 2) {
                let frac = n as f64 / d as f64;
                if (c - frac).abs() < 1e-10 {
                    return format!("\\frac{{{}}}{{{}}}", n, d);
                }
                if (c + frac).abs() < 1e-10 {
                    return format!("-\\frac{{{}}}{{{}}}", n, d);
                }
            }
        }
        format!("{:.4}", c)
    }

    fn render(expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => match name.as_str() {
                "phi" | "φ" => "\\varphi".to_string(),
                "theta" | "θ" => "\\theta".to_string(),
                "pi" | "π" => "\\pi".to_string(),
                "omega" | "ω" => "\\omega".to_string(),
                "alpha" | "α" => "\\alpha".to_string(),
                "beta" | "β" => "\\beta".to_string(),
                "gamma" | "γ" => "\\gamma".to_string(),
                "delta" | "δ" => "\\delta".to_string(),
                "epsilon" | "ε" => "\\epsilon".to_string(),
                "lambda" | "λ" => "\\lambda".to_string(),
                "mu" | "μ" => "\\mu".to_string(),
                "sigma" | "σ" => "\\sigma".to_string(),
                "tau" | "τ" => "\\tau".to_string(),
                s if s.len() > 1
                    && !s
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false) =>
                {
                    format!("{}", s)
                }
                s => s.to_string(),
            },
            Expr::Const(c) => const_to_latex(*c),
            Expr::BinOp(BinOp::Div, num, den) => {
                format!("\\frac{{{}}}{{{}}}", render(num), render(den))
            }
            Expr::BinOp(BinOp::Pow, base, exp) => {
                let base_str = wrap_if_lower(base, 10);
                let exp_str = render(exp);
                format!("{}^{{{}}}", base_str, exp_str)
            }
            Expr::BinOp(BinOp::Mul, l, r) => {
                let l_str = wrap_if_lower(l, 6);
                let r_str = wrap_if_lower(r, 6);
                match (l.as_ref(), r.as_ref()) {
                    (Expr::Const(_), Expr::Const(_)) => format!("{} \\cdot {}", l_str, r_str),
                    _ => format!("{} {}", l_str, r_str),
                }
            }
            Expr::BinOp(BinOp::Add, l, r) => {
                if let Expr::Const(c) = r.as_ref() {
                    if *c < 0.0 {
                        return format!("{} - {}", wrap_if_lower(l, 4), const_to_latex(-c));
                    }
                }
                format!("{} + {}", wrap_if_lower(l, 4), wrap_if_lower(r, 4))
            }
            Expr::BinOp(BinOp::Sub, l, r) => {
                format!("{} - {}", wrap_if_lower(l, 4), wrap_if_lower(r, 4))
            }
            Expr::Func(UnaryFn::Sqrt, arg) => format!("\\sqrt{{{}}}", render(arg)),
            Expr::Func(UnaryFn::Log, arg) => format!("\\ln\\left({}\\right)", render(arg)),
            Expr::Func(UnaryFn::Exp, arg) => format!("e^{{{}}}", render(arg)),
            Expr::Func(UnaryFn::Sin, arg) => format!("\\sin\\left({}\\right)", render(arg)),
            Expr::Func(UnaryFn::Cos, arg) => format!("\\cos\\left({}\\right)", render(arg)),
            Expr::Func(UnaryFn::Abs, arg) => format!("\\left|{}\\right|", render(arg)),
            Expr::Func(UnaryFn::Floor, arg) => format!("\\lfloor {} \\rfloor", render(arg)),
            Expr::Sum(body, var) => format!("\\sum_{{{}=0}}^{{n}} {}", var, render(body)),
        }
    }

    render(expr)
}

/// Escape LaTeX special characters for conjecture tables.
pub fn latex_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\textbackslash{}"),
            '&' => out.push_str("\\&"),
            '%' => out.push_str("\\%"),
            '$' => out.push_str("\\$"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => out.push(ch),
        }
    }
    out
}

/// Generate a random expression tree of bounded depth.
pub fn random_expr(rng: &mut u64, max_depth: usize) -> Expr {
    *rng = lcg_step(*rng);
    if max_depth == 0 || (*rng % 3 == 0 && max_depth < 3) {
        *rng = lcg_step(*rng);
        if *rng % 3 == 0 {
            Expr::Var("n".into())
        } else {
            let constants = [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                0.5,
                -1.0,
                -2.0,
                -0.5,
                std::f64::consts::PI,
                std::f64::consts::E,
                (1.0 + 5.0_f64.sqrt()) / 2.0,
                std::f64::consts::FRAC_1_SQRT_2,
                2.0 / 3.0,
                1.0 / std::f64::consts::E,
            ];
            *rng = lcg_step(*rng);
            Expr::Const(constants[*rng as usize % constants.len()])
        }
    } else {
        *rng = lcg_step(*rng);
        if *rng % 4 == 0 {
            let fns = [UnaryFn::Sqrt, UnaryFn::Log, UnaryFn::Exp, UnaryFn::Sin];
            *rng = lcg_step(*rng);
            Expr::Func(
                fns[*rng as usize % fns.len()],
                Box::new(random_expr(rng, max_depth - 1)),
            )
        } else {
            let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
            *rng = lcg_step(*rng);
            let op = ops[*rng as usize % ops.len()];
            Expr::BinOp(
                op,
                Box::new(random_expr(rng, max_depth - 1)),
                Box::new(random_expr(rng, max_depth - 1)),
            )
        }
    }
}

/// Simplify an expression tree by applying algebraic rewriting rules.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::BinOp(op, l, r) => {
            let sl = simplify(l);
            let sr = simplify(r);
            match (op, &sl, &sr) {
                (_, Expr::Const(_), Expr::Const(_)) => {
                    let result = Expr::BinOp(*op, Box::new(sl.clone()), Box::new(sr.clone()));
                    let val = result.eval(&[]);
                    if val.is_finite() {
                        Expr::Const(val)
                    } else {
                        result
                    }
                }
                (BinOp::Add, _, Expr::Const(c)) if *c == 0.0 => sl,
                (BinOp::Add, Expr::Const(c), _) if *c == 0.0 => sr,
                (BinOp::Sub, _, Expr::Const(c)) if *c == 0.0 => sl,
                (BinOp::Mul, _, Expr::Const(c)) if *c == 1.0 => sl,
                (BinOp::Mul, Expr::Const(c), _) if *c == 1.0 => sr,
                (BinOp::Mul, _, Expr::Const(c)) if *c == 0.0 => Expr::Const(0.0),
                (BinOp::Mul, Expr::Const(c), _) if *c == 0.0 => Expr::Const(0.0),
                (BinOp::Div, _, Expr::Const(c)) if *c == 1.0 => sl,
                (BinOp::Pow, _, Expr::Const(c)) if *c == 1.0 => sl,
                (BinOp::Pow, _, Expr::Const(c)) if *c == 0.0 => Expr::Const(1.0),
                (BinOp::Div, _, Expr::BinOp(BinOp::Div, b, c)) => simplify(&Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::BinOp(BinOp::Mul, Box::new(sl), c.clone())),
                    b.clone(),
                )),
                _ => Expr::BinOp(*op, Box::new(sl), Box::new(sr)),
            }
        }
        Expr::Func(f, arg) => {
            let sa = simplify(arg);
            if let Expr::Const(_) = &sa {
                let result = Expr::Func(*f, Box::new(sa.clone()));
                let val = result.eval(&[]);
                if val.is_finite() {
                    return Expr::Const(val);
                }
            }
            Expr::Func(*f, Box::new(sa))
        }
        Expr::Sum(body, var) => Expr::Sum(Box::new(simplify(body)), var.clone()),
        other => other.clone(),
    }
}

/// A detected recurrence relation.
#[derive(Debug, Clone)]
pub struct RecurrenceRelation {
    pub formula: String,
    pub order: usize,
    pub coefficients: Vec<f64>,
    pub max_residual: f64,
}

/// Detect if a sequence satisfies a simple recurrence relation.
pub fn detect_recurrence(data: &[(f64, f64)]) -> Option<RecurrenceRelation> {
    if data.len() < 4 {
        return None;
    }

    let values: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

    if values.len() >= 3 && values[0].abs() > 1e-15 {
        let n = values.len() - 1;
        let mut sum_yy = 0.0;
        let mut sum_y = 0.0;
        let mut sum_y1 = 0.0;
        let mut sum_yy1 = 0.0;
        let mut sum_1 = 0.0;
        for i in 1..=n {
            let y = values[i];
            let y1 = values[i - 1];
            sum_yy1 += y * y1;
            sum_y += y;
            sum_y1 += y1;
            sum_yy += y1 * y1;
            sum_1 += 1.0;
        }
        let det = sum_yy * sum_1 - sum_y1 * sum_y1;
        if det.abs() > 1e-15 {
            let a = (sum_yy1 * sum_1 - sum_y * sum_y1) / det;
            let b = (sum_yy * sum_y - sum_yy1 * sum_y1) / det;
            let max_residual = (1..=n)
                .map(|i| (values[i] - (a * values[i - 1] + b)).abs())
                .fold(0.0f64, f64::max);
            if max_residual < values.iter().map(|v| v.abs()).sum::<f64>() * 1e-10 / n as f64 {
                return Some(RecurrenceRelation {
                    formula: format!("f(n) = {:.6}*f(n-1) + {:.6}", a, b),
                    order: 1,
                    coefficients: vec![a, b],
                    max_residual,
                });
            }
        }
    }

    if values.len() >= 4 {
        let max_residual = (2..values.len())
            .map(|i| (values[i] - values[i - 1] - values[i - 2]).abs())
            .fold(0.0f64, f64::max);
        let scale = values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64;
        if max_residual < scale * 1e-10 {
            return Some(RecurrenceRelation {
                formula: "f(n) = f(n-1) + f(n-2)".to_string(),
                order: 2,
                coefficients: vec![1.0, 1.0],
                max_residual,
            });
        }
    }

    if values.len() >= 3 {
        let max_residual = (1..values.len())
            .map(|i| {
                let n_val = data[i].0;
                (values[i] - values[i - 1] - n_val).abs()
            })
            .fold(0.0f64, f64::max);
        if max_residual < 1e-10 {
            return Some(RecurrenceRelation {
                formula: "f(n) = f(n-1) + n".to_string(),
                order: 1,
                coefficients: vec![1.0],
                max_residual,
            });
        }
    }

    None
}

/// Solve a detected recurrence to obtain a closed-form expression.
pub fn solve_recurrence(rec: &RecurrenceRelation, data: &[(f64, f64)]) -> Option<Expr> {
    if data.is_empty() {
        return None;
    }

    match rec.order {
        1 => {
            let a = rec.coefficients.first().copied().unwrap_or(1.0);
            let b = rec.coefficients.get(1).copied().unwrap_or(0.0);
            let n0 = data[0].0;
            let v0 = data[0].1;

            if (a - 1.0).abs() < 1e-10 {
                if rec.formula.contains("+ n") || rec.formula.contains("+ 1.00*n") {
                    let tri_n = Expr::BinOp(
                        BinOp::Div,
                        Box::new(Expr::BinOp(
                            BinOp::Mul,
                            Box::new(Expr::Var("n".into())),
                            Box::new(Expr::BinOp(
                                BinOp::Add,
                                Box::new(Expr::Var("n".into())),
                                Box::new(Expr::Const(1.0)),
                            )),
                        )),
                        Box::new(Expr::Const(2.0)),
                    );
                    let offset = v0 - n0 * (n0 + 1.0) / 2.0;
                    if offset.abs() < 1e-10 {
                        Some(tri_n)
                    } else {
                        Some(Expr::BinOp(
                            BinOp::Add,
                            Box::new(tri_n),
                            Box::new(Expr::Const(offset)),
                        ))
                    }
                } else {
                    let intercept = v0 - b * n0;
                    let bn = Expr::BinOp(
                        BinOp::Mul,
                        Box::new(Expr::Const(b)),
                        Box::new(Expr::Var("n".into())),
                    );
                    if intercept.abs() < 1e-10 {
                        Some(bn)
                    } else {
                        Some(Expr::BinOp(
                            BinOp::Add,
                            Box::new(Expr::Const(intercept)),
                            Box::new(bn),
                        ))
                    }
                }
            } else if b.abs() < 1e-10 {
                let exp_node = if n0.abs() < 1e-10 {
                    Expr::Var("n".into())
                } else {
                    Expr::BinOp(
                        BinOp::Sub,
                        Box::new(Expr::Var("n".into())),
                        Box::new(Expr::Const(n0)),
                    )
                };
                Some(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(v0)),
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Const(a)),
                        Box::new(exp_node),
                    )),
                ))
            } else {
                None
            }
        }
        2 => {
            let a = rec.coefficients.first().copied().unwrap_or(1.0);
            let b = rec.coefficients.get(1).copied().unwrap_or(1.0);
            let discriminant = a * a + 4.0 * b;
            if discriminant < 0.0 {
                return None;
            }
            let root_disc = discriminant.sqrt();
            let r1 = (a + root_disc) / 2.0;
            let r2 = (a - root_disc) / 2.0;
            let y0 = data[0].1;
            let y1 = data.get(1).map(|(_, y)| *y).unwrap_or(y0);
            if (r1 - r2).abs() < 1e-10 {
                return None;
            }
            let c1 = (y1 - y0 * r2) / (r1 - r2);
            let c2 = y0 - c1;
            Some(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(c1)),
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Const(r1)),
                        Box::new(Expr::Var("n".into())),
                    )),
                )),
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(c2)),
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Const(r2)),
                        Box::new(Expr::Var("n".into())),
                    )),
                )),
            ))
        }
        _ => None,
    }
}

fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
