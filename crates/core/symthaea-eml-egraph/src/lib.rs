use egg::{rewrite as rw, *};
use symthaea_core::hdc::abstract_thought::expr_canonical_string;
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use thiserror::Error;

define_language! {
    pub enum SymbolLang {
        "1" = One,
        Symbol(Symbol),
        "add" = Add([Id; 2]),
        "div" = Div([Id; 2]),
        "eml" = Eml([Id; 2]),
        "exp" = Exp(Id),
        "ln" = Ln(Id),
        "mul" = Mul([Id; 2]),
        "pow" = Pow([Id; 2]),
        "sin" = Sin(Id),
        "sub" = Sub([Id; 2]),
        "sqrt" = Sqrt(Id),
    }
}

#[derive(Debug, Error)]
pub enum EgraphSpikeError {
    #[error("failed to parse expression `{expr}`: {detail}")]
    Parse { expr: String, detail: String },
    #[error("unsupported Symthaea expression for egg bridge: {0}")]
    UnsupportedExpr(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprEquivalenceClass {
    pub representative: usize,
    pub members: Vec<usize>,
    pub egg_canonical: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprBatchCollapseReport {
    pub classes: Vec<ExprEquivalenceClass>,
    pub unsupported: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalBucket {
    pub canonical: String,
    pub members: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapseComparisonClass {
    pub egg_class: ExprEquivalenceClass,
    pub current_canonical_buckets: Vec<CanonicalBucket>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapseComparisonReport {
    pub classes: Vec<CollapseComparisonClass>,
    pub unsupported: Vec<usize>,
}

pub fn rewrite_rules() -> Vec<Rewrite<SymbolLang, ()>> {
    vec![
        rw!("exp-to-eml"; "(exp ?x)" => "(eml ?x 1)"),
        rw!("ln-to-eml"; "(ln ?x)" => "(eml 1 (eml (eml 1 ?x) 1))"),
        rw!("pow2-to-mul"; "(pow ?x 2)" => "(mul ?x ?x)"),
        rw!("pow1"; "(pow ?x 1)" => "?x"),
        rw!("pow0"; "(pow ?x 0)" => "1"),
        rw!("mul-one-left"; "(mul 1 ?x)" => "?x"),
        rw!("mul-one-right"; "(mul ?x 1)" => "?x"),
        rw!("mul-comm"; "(mul ?x ?y)" => "(mul ?y ?x)"),
        rw!("mul-assoc-right"; "(mul ?x (mul ?y ?z))" => "(mul (mul ?x ?y) ?z)"),
        rw!("mul-assoc-left"; "(mul (mul ?x ?y) ?z)" => "(mul ?x (mul ?y ?z))"),
        rw!("add-comm"; "(add ?x ?y)" => "(add ?y ?x)"),
        rw!("add-assoc-right"; "(add ?x (add ?y ?z))" => "(add (add ?x ?y) ?z)"),
        rw!("add-assoc-left"; "(add (add ?x ?y) ?z)" => "(add ?x (add ?y ?z))"),
    ]
}

fn parse(expr: &str) -> Result<RecExpr<SymbolLang>, EgraphSpikeError> {
    expr.parse::<RecExpr<SymbolLang>>()
        .map_err(
            |source: RecExprParseError<FromOpError>| EgraphSpikeError::Parse {
                expr: expr.to_string(),
                detail: source.to_string(),
            },
        )
}

fn constant_atom(c: f64) -> Result<String, EgraphSpikeError> {
    if !c.is_finite() {
        return Err(EgraphSpikeError::UnsupportedExpr(format!(
            "non-finite constant: {c}"
        )));
    }

    if (c - 1.0).abs() < 1e-12 {
        return Ok("1".to_string());
    }

    if (c - std::f64::consts::E).abs() < 1e-12 {
        return Ok("const_e".to_string());
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    if (c - phi).abs() < 1e-12 {
        return Ok("const_phi".to_string());
    }

    let repr = if c.fract().abs() < 1e-12 {
        format!("{}", c as i64)
    } else {
        format!("{c:.6}")
    };

    let mut atom = String::from("const_");
    for ch in repr.chars() {
        match ch {
            '-' => atom.push_str("neg_"),
            '.' => atom.push('_'),
            _ => atom.push(ch),
        }
    }
    Ok(atom)
}

pub fn equivalent(lhs: &str, rhs: &str) -> Result<bool, EgraphSpikeError> {
    let lhs = parse(lhs)?;
    let rhs = parse(rhs)?;
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let lhs_id = egraph.add_expr(&lhs);
    let rhs_id = egraph.add_expr(&rhs);
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(12)
        .run(&rewrite_rules());
    let lhs_id = runner.egraph.find(lhs_id);
    let rhs_id = runner.egraph.find(rhs_id);
    Ok(lhs_id == rhs_id)
}

pub fn canonicalize(expr: &str) -> Result<String, EgraphSpikeError> {
    let expr = parse(expr)?;
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let root = egraph.add_expr(&expr);
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(12)
        .run(&rewrite_rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, best) = extractor.find_best(root);
    Ok(best.to_string())
}

pub fn expr_to_egg_sexp(expr: &Expr) -> Result<String, EgraphSpikeError> {
    match expr {
        Expr::Var(name) => Ok(name.clone()),
        Expr::Const(c) => constant_atom(*c),
        Expr::BinOp(BinOp::Mul, l, r) => Ok(format!(
            "(mul {} {})",
            expr_to_egg_sexp(l)?,
            expr_to_egg_sexp(r)?
        )),
        Expr::BinOp(BinOp::Add, l, r) => Ok(format!(
            "(add {} {})",
            expr_to_egg_sexp(l)?,
            expr_to_egg_sexp(r)?
        )),
        Expr::BinOp(BinOp::Div, l, r) => Ok(format!(
            "(div {} {})",
            expr_to_egg_sexp(l)?,
            expr_to_egg_sexp(r)?
        )),
        Expr::BinOp(BinOp::Sub, l, r) => Ok(format!(
            "(sub {} {})",
            expr_to_egg_sexp(l)?,
            expr_to_egg_sexp(r)?
        )),
        Expr::BinOp(BinOp::Pow, l, r) => Ok(format!(
            "(pow {} {})",
            expr_to_egg_sexp(l)?,
            expr_to_egg_sexp(r)?
        )),
        Expr::Func(UnaryFn::Exp, arg) => Ok(format!("(exp {})", expr_to_egg_sexp(arg)?)),
        Expr::Func(UnaryFn::Log, arg) => Ok(format!("(ln {})", expr_to_egg_sexp(arg)?)),
        Expr::Func(UnaryFn::Sin, arg) => Ok(format!("(sin {})", expr_to_egg_sexp(arg)?)),
        Expr::Func(UnaryFn::Sqrt, arg) => Ok(format!("(sqrt {})", expr_to_egg_sexp(arg)?)),
        other => Err(EgraphSpikeError::UnsupportedExpr(format!("{other}"))),
    }
}

pub fn equivalent_expr(lhs: &Expr, rhs: &Expr) -> Result<bool, EgraphSpikeError> {
    equivalent(&expr_to_egg_sexp(lhs)?, &expr_to_egg_sexp(rhs)?)
}

pub fn collapse_expr_candidates(exprs: &[Expr]) -> ExprBatchCollapseReport {
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let mut inserted: Vec<(usize, Id)> = Vec::new();
    let mut unsupported = Vec::new();

    for (idx, expr) in exprs.iter().enumerate() {
        match expr_to_egg_sexp(expr)
            .and_then(|sexp| parse(&sexp))
            .map(|parsed| egraph.add_expr(&parsed))
        {
            Ok(id) => inserted.push((idx, id)),
            Err(EgraphSpikeError::UnsupportedExpr(_)) => unsupported.push(idx),
            Err(EgraphSpikeError::Parse { .. }) => unsupported.push(idx),
        }
    }

    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(12)
        .run(&rewrite_rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let mut by_root: std::collections::BTreeMap<Id, Vec<usize>> = std::collections::BTreeMap::new();

    for (idx, id) in inserted {
        by_root.entry(runner.egraph.find(id)).or_default().push(idx);
    }

    let classes = by_root
        .into_iter()
        .map(|(root, mut members)| {
            members.sort_unstable();
            let (_cost, best) = extractor.find_best(root);
            ExprEquivalenceClass {
                representative: members[0],
                members,
                egg_canonical: best.to_string(),
            }
        })
        .collect();

    ExprBatchCollapseReport {
        classes,
        unsupported,
    }
}

pub fn compare_current_vs_egg_collapse(exprs: &[Expr]) -> CollapseComparisonReport {
    let collapsed = collapse_expr_candidates(exprs);
    let classes = collapsed
        .classes
        .iter()
        .cloned()
        .map(|egg_class| {
            let mut buckets: std::collections::BTreeMap<String, Vec<usize>> =
                std::collections::BTreeMap::new();
            for &idx in &egg_class.members {
                buckets
                    .entry(expr_canonical_string(&exprs[idx]))
                    .or_default()
                    .push(idx);
            }
            let current_canonical_buckets = buckets
                .into_iter()
                .map(|(canonical, mut members)| {
                    members.sort_unstable();
                    CanonicalBucket { canonical, members }
                })
                .collect();
            CollapseComparisonClass {
                egg_class,
                current_canonical_buckets,
            }
        })
        .collect();

    CollapseComparisonReport {
        classes,
        unsupported: collapsed.unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn exp_matches_strict_eml_basis() {
        assert!(equivalent("(exp x)", "(eml x 1)").unwrap());
    }

    #[test]
    fn ln_matches_eml_basis() {
        assert!(equivalent("(ln x)", "(eml 1 (eml (eml 1 x) 1))").unwrap());
    }

    #[test]
    fn pow2_collapses_to_mul() {
        assert!(equivalent("(pow x 2)", "(mul x x)").unwrap());
    }

    #[test]
    fn canonicalization_prefers_smaller_form() {
        let best = canonicalize("(pow x 1)").unwrap();
        assert_eq!(best, "x");
    }

    #[test]
    fn egg_proves_associative_mul_equivalence_that_current_canonical_string_misses() {
        let left_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
            Box::new(Expr::Var("z".into())),
        );
        let right_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("y".into())),
                Box::new(Expr::Var("z".into())),
            )),
        );

        assert_ne!(
            expr_canonical_string(&left_assoc),
            expr_canonical_string(&right_assoc)
        );
        assert!(equivalent("(mul (mul x y) z)", "(mul x (mul y z))").unwrap());
    }

    #[test]
    fn collapse_expr_candidates_groups_associative_multiplication_forms() {
        let left_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
            Box::new(Expr::Var("z".into())),
        );
        let right_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("y".into())),
                Box::new(Expr::Var("z".into())),
            )),
        );
        let exp_expr = Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into())));
        // Genuinely unsupported by expr_to_egg_sexp: Add/Sub/Mul/Div/Pow and
        // Exp/Log/Sin/Sqrt are all handled (its `other` arm only rejects
        // Sum and the Cos/Abs/Floor unary functions). This used to be
        // `x + y`, but BinOp::Add is fully supported (SymbolLang has an
        // "add" op with its own rewrite rules), so it silently became its
        // own real equivalence class instead of landing in
        // `report.unsupported` -- breaking both assertions below.
        let unsupported = Expr::Func(UnaryFn::Cos, Box::new(Expr::Var("x".into())));

        let report = collapse_expr_candidates(&[
            left_assoc.clone(),
            right_assoc.clone(),
            exp_expr.clone(),
            unsupported.clone(),
        ]);

        assert_eq!(report.classes.len(), 2);
        let mut member_sets: Vec<Vec<usize>> = report
            .classes
            .iter()
            .map(|class| class.members.clone())
            .collect();
        member_sets.sort();
        assert_eq!(member_sets, vec![vec![0, 1], vec![2]]);
        assert_eq!(report.unsupported, vec![3]);
    }

    #[test]
    fn compare_current_vs_egg_collapse_highlights_current_canonical_split() {
        let left_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
            Box::new(Expr::Var("z".into())),
        );
        let right_assoc = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("y".into())),
                Box::new(Expr::Var("z".into())),
            )),
        );

        let report = compare_current_vs_egg_collapse(&[left_assoc, right_assoc]);
        assert_eq!(report.unsupported, Vec::<usize>::new());
        assert_eq!(report.classes.len(), 1);
        assert_eq!(report.classes[0].egg_class.members, vec![0, 1]);
        assert_eq!(report.classes[0].current_canonical_buckets.len(), 2);
        let mut bucket_members: Vec<Vec<usize>> = report.classes[0]
            .current_canonical_buckets
            .iter()
            .map(|bucket| bucket.members.clone())
            .collect();
        bucket_members.sort();
        assert_eq!(bucket_members, vec![vec![0], vec![1]]);
    }

    #[test]
    fn bridge_supports_real_discovery_operator_subset() {
        let exprs = [
            Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::BinOp(
                        BinOp::Add,
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Var("n".into())),
                    )),
                    Box::new(Expr::Func(
                        UnaryFn::Sqrt,
                        Box::new(Expr::BinOp(
                            BinOp::Mul,
                            Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Var("n".into())),
                        )),
                    )),
                )),
            ),
            Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(1.0)),
                    Box::new(Expr::Var("n".into())),
                )),
            ),
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Const(1.0)),
                    Box::new(Expr::Var("n".into())),
                )),
                Box::new(Expr::Func(
                    UnaryFn::Sqrt,
                    Box::new(Expr::BinOp(
                        BinOp::Mul,
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Var("n".into())),
                    )),
                )),
            ),
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var("n".into())),
            ),
        ];

        let report = collapse_expr_candidates(&exprs);
        assert!(report.unsupported.is_empty());
        assert_eq!(report.classes.len(), 4);
    }

    #[test]
    fn bridge_supports_sub_and_non_unit_real_constants() {
        let exprs = [
            Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(1.0)),
            ),
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(0.5)),
            ),
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(std::f64::consts::E)),
            ),
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const((1.0 + 5.0_f64.sqrt()) / 2.0)),
            ),
            Expr::Func(
                UnaryFn::Sin,
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var("n".into())),
                    Box::new(Expr::Const(-2.0)),
                )),
            ),
        ];

        let report = collapse_expr_candidates(&exprs);
        assert!(report.unsupported.is_empty());
        assert_eq!(report.classes.len(), 5);
    }
}
