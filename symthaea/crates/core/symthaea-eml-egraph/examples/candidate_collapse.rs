use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_eml_egraph::compare_current_vs_egg_collapse;

fn main() {
    let candidates = vec![
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
        Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into()))),
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        ),
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("x".into())),
        ),
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        ),
    ];

    let report = compare_current_vs_egg_collapse(&candidates);

    println!("offline egg collapse comparison");
    println!("unsupported candidate indices: {:?}", report.unsupported);
    println!();

    for (class_idx, class) in report.classes.iter().enumerate() {
        println!("egg class {class_idx}");
        println!("  members: {:?}", class.egg_class.members);
        println!("  egg canonical: {}", class.egg_class.egg_canonical);
        println!("  current canonical buckets:");
        for bucket in &class.current_canonical_buckets {
            println!("    {:?} -> {}", bucket.members, bucket.canonical);
        }
        println!();
    }
}
