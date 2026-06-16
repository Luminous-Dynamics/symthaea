// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::conjecture_engine::{
    BinOp, Conjecture, ConjectureStatus, Expr, MacroPromotionTier, MathDomain, attach_eml_metadata,
};
use symthaea_core::hdc::eml::{
    EmlEvalMode, compile_expr, compile_expr_constructive, verify_expr_compilation,
};

fn benchmark_exprs() -> Vec<(&'static str, Expr)> {
    vec![
        (
            "strict_exp",
            Expr::Func(
                symthaea_core::hdc::conjecture_engine::UnaryFn::Exp,
                Box::new(Expr::Var("x".into())),
            ),
        ),
        (
            "strict_div",
            Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            ),
        ),
        (
            "constructive_add",
            Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            ),
        ),
        (
            "constructive_square",
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Const(2.0)),
            ),
        ),
    ]
}

fn make_conjecture(name: &str, formula: Expr) -> Conjecture {
    Conjecture {
        formula,
        formula_str: name.to_string(),
        source: format!("bench::{name}"),
        domain: MathDomain::Physics,
        training_mse: 0.0,
        complexity: 1,
        fitness: 0.0,
        status: ConjectureStatus::Proposed,
        confidence: 1.0,
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

fn bench_eml_compile(c: &mut Criterion) {
    let exprs = benchmark_exprs();
    let mut group = c.benchmark_group("eml_compile");

    for (name, expr) in &exprs {
        group.bench_function(format!("{name}_strict"), |b| {
            b.iter(|| {
                let _ = compile_expr(black_box(expr));
            })
        });
        group.bench_function(format!("{name}_constructive"), |b| {
            b.iter(|| {
                let _ = compile_expr_constructive(black_box(expr));
            })
        });
    }

    group.finish();
}

fn bench_eml_verify(c: &mut Criterion) {
    let exprs = benchmark_exprs();
    let mut group = c.benchmark_group("eml_verify");

    for (name, expr) in &exprs {
        if let Ok(compiled) = compile_expr_constructive(expr) {
            group.bench_function(format!("{name}_constructive_real"), |b| {
                b.iter(|| {
                    let _ = verify_expr_compilation(
                        black_box(expr),
                        black_box(&compiled),
                        EmlEvalMode::RealConstructive,
                    );
                })
            });
        }
        if let Ok(compiled) = compile_expr(expr) {
            group.bench_function(format!("{name}_strict_real"), |b| {
                b.iter(|| {
                    let _ = verify_expr_compilation(
                        black_box(expr),
                        black_box(&compiled),
                        EmlEvalMode::RealIeee,
                    );
                })
            });
            group.bench_function(format!("{name}_strict_complex"), |b| {
                b.iter(|| {
                    let _ = verify_expr_compilation(
                        black_box(expr),
                        black_box(&compiled),
                        EmlEvalMode::ComplexPrincipal,
                    );
                })
            });
        }
    }

    group.finish();
}

fn bench_attach_eml_metadata(c: &mut Criterion) {
    let conjectures: Vec<Conjecture> = benchmark_exprs()
        .into_iter()
        .map(|(name, expr)| make_conjecture(name, expr))
        .collect();

    c.bench_function("attach_eml_metadata_batch4", |b| {
        b.iter(|| {
            let mut local = black_box(conjectures.clone());
            for conjecture in &mut local {
                attach_eml_metadata(conjecture);
            }
        })
    });
}

criterion_group!(
    benches,
    bench_eml_compile,
    bench_eml_verify,
    bench_attach_eml_metadata
);
criterion_main!(benches);
