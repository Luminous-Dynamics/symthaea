use super::*;

/// Analyze growth class of a sequence (#5, #8).
/// Returns (growth_type, estimated_rate) to guide GP grammar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrowthClass {
    Constant,         // f(n) -> c
    Logarithmic,      // f(n) ~ ln(n)
    Polynomial(f64),  // f(n) ~ n^p (returns p)
    Exponential,      // f(n) ~ a^n
    SuperExponential, // f(n) ~ n! or faster
}

pub fn analyze_growth(data: &[(f64, f64)]) -> GrowthClass {
    if data.len() < 4 {
        return GrowthClass::Constant;
    }

    let values: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    if var < mean * mean * 0.01 {
        return GrowthClass::Constant;
    }

    let half = values.len() / 2;
    if half >= 3 {
        let mean1 = values[..half].iter().sum::<f64>() / half as f64;
        let var1 = values[..half]
            .iter()
            .map(|v| (v - mean1).powi(2))
            .sum::<f64>()
            / half as f64;
        let mean2 = values[half..].iter().sum::<f64>() / (values.len() - half) as f64;
        let var2 = values[half..]
            .iter()
            .map(|v| (v - mean2).powi(2))
            .sum::<f64>()
            / (values.len() - half) as f64;
        if var2 < var1 * 0.1 && var2 < mean2 * mean2 * 0.01 {
            return GrowthClass::Constant;
        }
    }

    let positive: Vec<(f64, f64)> = data
        .iter()
        .filter(|(x, y)| *x > 0.0 && *y > 0.0)
        .map(|(x, y)| (x.ln(), y.ln()))
        .collect();

    if positive.len() >= 3 {
        let n = positive.len() as f64;
        let sx: f64 = positive.iter().map(|(x, _)| x).sum();
        let sy: f64 = positive.iter().map(|(_, y)| y).sum();
        let sxy: f64 = positive.iter().map(|(x, y)| x * y).sum();
        let sx2: f64 = positive.iter().map(|(x, _)| x * x).sum();
        let denom = n * sx2 - sx * sx;
        if denom.abs() > 1e-10 {
            let p = (n * sxy - sx * sy) / denom;
            let r2 = {
                let ss_res: f64 = positive
                    .iter()
                    .map(|(x, y)| {
                        let pred = p * x + (sy - p * sx) / n;
                        (y - pred).powi(2)
                    })
                    .sum();
                let ss_tot: f64 = positive.iter().map(|(_, y)| (y - sy / n).powi(2)).sum();
                if ss_tot > 1e-10 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                }
            };

            if r2 > 0.95 && p > 0.0 && p < 10.0 {
                return GrowthClass::Polynomial(p);
            }
        }
    }

    let log_linear: Vec<(f64, f64)> = data
        .iter()
        .filter(|(_, y)| *y > 0.0)
        .map(|(x, y)| (*x, y.ln()))
        .collect();

    if log_linear.len() >= 3 {
        let n = log_linear.len() as f64;
        let sx: f64 = log_linear.iter().map(|(x, _)| x).sum();
        let sy: f64 = log_linear.iter().map(|(_, y)| y).sum();
        let sxy: f64 = log_linear.iter().map(|(x, y)| x * y).sum();
        let sx2: f64 = log_linear.iter().map(|(x, _)| x * x).sum();
        let denom = n * sx2 - sx * sx;
        if denom.abs() > 1e-10 {
            let slope = (n * sxy - sx * sy) / denom;
            let r2 = {
                let ss_res: f64 = log_linear
                    .iter()
                    .map(|(x, y)| {
                        let pred = slope * x + (sy - slope * sx) / n;
                        (y - pred).powi(2)
                    })
                    .sum();
                let ss_tot: f64 = log_linear.iter().map(|(_, y)| (y - sy / n).powi(2)).sum();
                if ss_tot > 1e-10 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                }
            };
            if r2 > 0.95 && slope > 0.1 {
                return GrowthClass::Exponential;
            }
        }
    }

    let ratios: Vec<f64> = values
        .windows(2)
        .filter_map(|w| {
            if w[0].abs() > 1e-10 {
                Some(w[1] / w[0])
            } else {
                None
            }
        })
        .collect();
    if ratios.len() >= 3 {
        let ratio_growing = ratios.windows(2).filter(|w| w[1] > w[0] * 1.05).count();
        if ratio_growing > ratios.len() / 2 {
            return GrowthClass::SuperExponential;
        }
    }

    GrowthClass::Polynomial(1.0)
}

/// Compute difference sequence Df(n) = f(n) - f(n-1) (#7).
pub fn difference_sequence(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    data.windows(2).map(|w| (w[1].0, w[1].1 - w[0].1)).collect()
}

/// Compute ratio sequence f(n)/f(n-1) (#8).
pub fn ratio_sequence(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    data.windows(2)
        .filter_map(|w| {
            if w[0].1.abs() > 1e-10 {
                Some((w[1].0, w[1].1 / w[0].1))
            } else {
                None
            }
        })
        .collect()
}

/// Build growth-class-specific template library for GP initialization (#4).
pub(crate) fn build_template_library(growth: &GrowthClass) -> Vec<Expr> {
    let n = || Box::new(Expr::Var("n".into()));
    let c = |v: f64| Box::new(Expr::Const(v));

    let mut templates = vec![
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(BinOp::Mul, c(1.0), n())),
            c(0.0),
        ),
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                c(1.0),
                Box::new(Expr::BinOp(BinOp::Pow, n(), c(2.0))),
            )),
            Box::new(Expr::BinOp(BinOp::Mul, c(1.0), n())),
        ),
        Expr::BinOp(
            BinOp::Mul,
            c(1.0),
            Box::new(Expr::BinOp(BinOp::Pow, n(), c(1.5))),
        ),
    ];

    match growth {
        GrowthClass::Constant => {
            templates.push(Expr::Const(1.0));
            templates.push(Expr::Const(std::f64::consts::PI));
            templates.push(Expr::Const(1.0 / std::f64::consts::E));
            templates.push(Expr::BinOp(
                BinOp::Sub,
                c(1.0),
                Box::new(Expr::BinOp(
                    BinOp::Div,
                    c(1.0),
                    Box::new(Expr::BinOp(BinOp::Pow, n(), c(0.5))),
                )),
            ));
            templates.push(Expr::BinOp(
                BinOp::Sub,
                c(1.0),
                Box::new(Expr::BinOp(BinOp::Div, c(1.0), n())),
            ));
            templates.push(Expr::BinOp(
                BinOp::Add,
                c(1.0),
                Box::new(Expr::BinOp(
                    BinOp::Div,
                    c(1.0),
                    Box::new(Expr::Func(UnaryFn::Sqrt, n())),
                )),
            ));
        }
        GrowthClass::Logarithmic => {
            templates.push(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    c(1.0),
                    Box::new(Expr::Func(UnaryFn::Log, n())),
                )),
                c(0.0),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(BinOp::Mul, c(1.0), n())),
                Box::new(Expr::Func(UnaryFn::Log, n())),
            ));
        }
        GrowthClass::Polynomial(p) => {
            templates.push(Expr::BinOp(
                BinOp::Mul,
                c(1.0),
                Box::new(Expr::BinOp(BinOp::Pow, n(), c(*p))),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    c(1.0),
                    Box::new(Expr::BinOp(BinOp::Pow, n(), c(*p))),
                )),
                Box::new(Expr::BinOp(BinOp::Add, c(1.0), n())),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    n(),
                    Box::new(Expr::BinOp(BinOp::Add, n(), c(1.0))),
                )),
                c(2.0),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                c(-1.0),
                Box::new(Expr::BinOp(BinOp::Pow, n(), c(2.0))),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                c(1.0),
                Box::new(Expr::BinOp(BinOp::Pow, n(), c(1.5))),
            ));
            templates.push(Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::BinOp(BinOp::Pow, n(), c(2.0))),
                    c(1.0),
                )),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                c(1.0),
                Box::new(Expr::Func(
                    UnaryFn::Sqrt,
                    Box::new(Expr::BinOp(
                        BinOp::Add,
                        Box::new(Expr::BinOp(BinOp::Pow, n(), c(2.0))),
                        c(1.0),
                    )),
                )),
            ));
        }
        GrowthClass::Exponential => {
            templates.push(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    c(0.15),
                    Box::new(Expr::Func(
                        UnaryFn::Exp,
                        Box::new(Expr::BinOp(
                            BinOp::Mul,
                            c(2.5),
                            Box::new(Expr::Func(UnaryFn::Sqrt, n())),
                        )),
                    )),
                )),
                Box::new(Expr::BinOp(BinOp::Mul, c(1.0), n())),
            ));
            templates.push(Expr::BinOp(
                BinOp::Mul,
                c(1.0),
                Box::new(Expr::BinOp(BinOp::Pow, c(2.0), n())),
            ));
            templates.push(Expr::BinOp(
                BinOp::Mul,
                c(1.0),
                Box::new(Expr::Func(
                    UnaryFn::Exp,
                    Box::new(Expr::BinOp(BinOp::Mul, c(0.5), n())),
                )),
            ));
            templates.push(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(BinOp::Pow, c(4.0), n())),
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Func(
                        UnaryFn::Sqrt,
                        Box::new(Expr::BinOp(BinOp::Mul, c(std::f64::consts::PI), n())),
                    )),
                    Box::new(Expr::BinOp(BinOp::Add, n(), c(1.0))),
                )),
            ));
        }
        GrowthClass::SuperExponential => {
            templates.push(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Func(
                    UnaryFn::Sqrt,
                    Box::new(Expr::BinOp(BinOp::Mul, c(std::f64::consts::TAU), n())),
                )),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::BinOp(BinOp::Div, n(), c(std::f64::consts::E))),
                    n(),
                )),
            ));
            templates.push(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::BinOp(BinOp::Pow, c(2.0), n())),
                Box::new(Expr::BinOp(BinOp::Pow, n(), c(1.0))),
            ));
        }
    }

    templates
}

/// Type of relationship between two sequences.
#[derive(Debug, Clone)]
pub enum RelationType {
    Proportional { constant: f64 },
    ConstantDifference { offset: f64 },
    Linear { slope: f64, intercept: f64 },
}

/// A discovered relationship between two sequences.
#[derive(Debug, Clone)]
pub struct CrossSequenceRelation {
    pub source_a: String,
    pub source_b: String,
    pub relation_type: RelationType,
    pub r_squared: f64,
}

impl fmt::Display for CrossSequenceRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.relation_type {
            RelationType::Proportional { constant } => write!(
                f,
                "{} ≈ {:.4} · {} (R²={:.4})",
                self.source_a, constant, self.source_b, self.r_squared
            ),
            RelationType::ConstantDifference { offset } => write!(
                f,
                "{} ≈ {} + {:.4} (R²={:.4})",
                self.source_a, self.source_b, offset, self.r_squared
            ),
            RelationType::Linear { slope, intercept } => write!(
                f,
                "{} ≈ {:.4}·{} + {:.4} (R²={:.4})",
                self.source_a, slope, self.source_b, intercept, self.r_squared
            ),
        }
    }
}

/// Discover relationships between two sequences by testing proportional,
/// constant-difference, and linear fits.
pub fn discover_cross_sequence_relations(
    a: &ObservedSequence,
    b: &ObservedSequence,
) -> Vec<CrossSequenceRelation> {
    let mut relations = Vec::new();
    let pairs: Vec<(f64, f64)> = a
        .data
        .iter()
        .filter_map(|(ax, ay)| {
            b.data
                .iter()
                .find(|(bx, _)| (*bx - *ax).abs() < 1e-10)
                .map(|(_, by)| (*ay, *by))
        })
        .collect();

    if pairs.len() < 3 {
        return relations;
    }

    let valid_ratios: Vec<f64> = pairs
        .iter()
        .filter(|(_, by)| by.abs() > 1e-10)
        .map(|(ay, by)| ay / by)
        .collect();
    if valid_ratios.len() >= 3 {
        let mean_ratio = valid_ratios.iter().sum::<f64>() / valid_ratios.len() as f64;
        let var = valid_ratios
            .iter()
            .map(|r| (r - mean_ratio).powi(2))
            .sum::<f64>()
            / valid_ratios.len() as f64;
        let cv = var.sqrt() / mean_ratio.abs().max(1e-10);
        if cv < 0.1 {
            let ss_res: f64 = pairs
                .iter()
                .map(|(ay, by)| (ay - mean_ratio * by).powi(2))
                .sum();
            let mean_a = pairs.iter().map(|(ay, _)| ay).sum::<f64>() / pairs.len() as f64;
            let ss_tot: f64 = pairs.iter().map(|(ay, _)| (ay - mean_a).powi(2)).sum();
            let r2 = if ss_tot > 1e-10 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };
            relations.push(CrossSequenceRelation {
                source_a: a.name.clone(),
                source_b: b.name.clone(),
                relation_type: RelationType::Proportional {
                    constant: mean_ratio,
                },
                r_squared: r2,
            });
        }
    }

    let diffs: Vec<f64> = pairs.iter().map(|(ay, by)| ay - by).collect();
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let diff_var = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;
    let mean_a = pairs.iter().map(|(ay, _)| ay).sum::<f64>() / pairs.len() as f64;
    let a_var = pairs
        .iter()
        .map(|(ay, _)| (ay - mean_a).powi(2))
        .sum::<f64>()
        / pairs.len() as f64;
    if diff_var < a_var * 0.01 && a_var > 1e-10 {
        let ss_res: f64 = pairs
            .iter()
            .map(|(ay, by)| (ay - by - mean_diff).powi(2))
            .sum();
        let ss_tot: f64 = pairs.iter().map(|(ay, _)| (ay - mean_a).powi(2)).sum();
        let r2 = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        relations.push(CrossSequenceRelation {
            source_a: a.name.clone(),
            source_b: b.name.clone(),
            relation_type: RelationType::ConstantDifference { offset: mean_diff },
            r_squared: r2,
        });
    }

    relations
}
