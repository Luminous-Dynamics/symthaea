use symthaea_core::hdc::conjecture_engine::{
    BinOp, Expr, GrowthClass, MathDomain, ObservedSequence, RelationType, analyze_growth,
    discover_cross_sequence_relations, observe_gr_correction, verify_formula_derivative,
};

#[test]
fn split_surface_reexports_stay_stable() {
    let gr = observe_gr_correction(1.0, 2.0, 10.0, 12);
    assert_eq!(gr.domain, MathDomain::Physics);
    assert!(!gr.data.is_empty());
    assert!(gr.data.iter().all(|(_, value)| value.is_finite()));

    let quadratic: Vec<(f64, f64)> = (1..=6)
        .map(|n| {
            let n = n as f64;
            (n, n * n)
        })
        .collect();
    match analyze_growth(&quadratic) {
        GrowthClass::Polynomial(power) => assert!((power - 2.0).abs() < 0.2),
        other => panic!("expected polynomial growth, got {other:?}"),
    }

    let a = ObservedSequence::new(
        "double",
        MathDomain::NumberTheory,
        vec![(1.0, 4.0), (2.0, 8.0), (3.0, 12.0), (4.0, 16.0)],
    );
    let b = ObservedSequence::new(
        "base",
        MathDomain::NumberTheory,
        vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
    );
    let relations = discover_cross_sequence_relations(&a, &b);
    assert!(relations.iter().any(|relation| {
        matches!(
            relation.relation_type,
            RelationType::Proportional { constant } if (constant - 2.0).abs() < 1e-9
        )
    }));

    let expr = Expr::BinOp(
        BinOp::Mul,
        Box::new(Expr::Const(3.0)),
        Box::new(Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        )),
    );
    let samples: Vec<(f64, f64)> = (0..=5)
        .map(|x| {
            let x = x as f64;
            (x, 3.0 * x * x)
        })
        .collect();
    let derivative = verify_formula_derivative(&expr, &samples, "x")
        .expect("quadratic derivative should be representable symbolically");
    assert!(derivative.is_consistent);
    assert!(derivative.max_relative_error < 1e-9);
}
