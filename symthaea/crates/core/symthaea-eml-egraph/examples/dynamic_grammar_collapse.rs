use symthaea_core::hdc::abstract_thought::dynamic_grammar::DynamicGrammar;
use symthaea_core::hdc::abstract_thought::expr_canonical_string;
use symthaea_core::hdc::abstract_thought::normalize_expr;
use symthaea_core::hdc::conjecture_engine::{
    BinOp, Conjecture, ConjectureEngine, ConjectureStatus, Expr, MacroPromotionTier, MathDomain,
    UnaryFn,
};
use symthaea_eml_egraph::compare_current_vs_egg_collapse;

fn make_conjecture(formula: Expr, source: &str) -> Conjecture {
    let complexity = formula.complexity();
    Conjecture {
        formula: formula.clone(),
        formula_str: format!("{formula}"),
        source: source.to_string(),
        domain: MathDomain::NumberTheory,
        training_mse: 0.0,
        complexity,
        fitness: 0.0,
        status: ConjectureStatus::FormallyVerified { proof_steps: 1 },
        confidence: 0.99,
        macro_promotion_tier: MacroPromotionTier::Formal,
        eml_compiled: None,
        eml_metrics: None,
        eml_verified_real: None,
        eml_real_domain: None,
        eml_verified_complex: None,
        eml_constructive_compiled: None,
        eml_constructive_metrics: None,
        eml_verified_constructive_real: None,
    }
}

fn main() {
    let mut engine = ConjectureEngine::new();
    let mut grammar = DynamicGrammar::new();

    let formulas = vec![
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
            Box::new(Expr::Var("z".into())),
        ),
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("y".into())),
                Box::new(Expr::Var("z".into())),
            )),
        ),
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("z".into())),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
        ),
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("x".into())),
        ),
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        ),
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("y".into())),
            Box::new(Expr::Var("y".into())),
        ),
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("y".into())),
            Box::new(Expr::Const(2.0)),
        ),
        Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into()))),
        Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("y".into()))),
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        ),
    ];

    for (idx, formula) in formulas.into_iter().enumerate() {
        engine
            .conjectures
            .push(make_conjecture(formula, &format!("candidate_{idx}")));
    }

    for idx in 0..engine.conjectures.len() {
        grammar.observe_subtree(
            normalize_expr(&engine.conjectures[idx].formula),
            &[idx],
            &engine,
        );
    }

    let patterns: Vec<Expr> = grammar
        .candidates
        .iter()
        .map(|candidate| candidate.pattern.clone())
        .collect();
    let report = compare_current_vs_egg_collapse(&patterns);
    let current_bucket_count = {
        let mut buckets = std::collections::BTreeSet::new();
        for pattern in &patterns {
            buckets.insert(expr_canonical_string(pattern));
        }
        buckets.len()
    };
    let egg_class_count = report.classes.len();
    let egg_only_merges = report
        .classes
        .iter()
        .filter(|class| class.current_canonical_buckets.len() > 1)
        .count();

    println!("dynamic grammar offline egg collapse");
    println!("candidate count: {}", grammar.candidates.len());
    println!("current canonical buckets: {}", current_bucket_count);
    println!("egg equivalence classes: {}", egg_class_count);
    println!("egg-only merged classes: {}", egg_only_merges);
    println!("unsupported candidate indices: {:?}", report.unsupported);
    println!();

    for (idx, candidate) in grammar.candidates.iter().enumerate() {
        println!("candidate {idx}: {}", candidate.canonical);
    }
    println!();

    for (class_idx, class) in report.classes.iter().enumerate() {
        println!("egg class {class_idx}");
        println!("  candidate members: {:?}", class.egg_class.members);
        println!("  egg canonical: {}", class.egg_class.egg_canonical);
        println!("  current canonical buckets:");
        for bucket in &class.current_canonical_buckets {
            println!("    {:?} -> {}", bucket.members, bucket.canonical);
        }
        println!();
    }
}
