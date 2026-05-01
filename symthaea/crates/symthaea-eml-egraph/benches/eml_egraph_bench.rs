// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Criterion benchmarks for the isolated EML/e-graph spike.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symthaea_core::hdc::abstract_thought::{canonicalize_expr, expr_canonical_string};
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_core::hdc::eml::compile_expr;
use symthaea_eml_egraph::{canonicalize, equivalent};

fn benchmark_cases() -> Vec<(&'static str, Expr)> {
    vec![
        (
            "(pow x 1)",
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Const(1.0)),
            ),
        ),
        (
            "(pow x 2)",
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Const(2.0)),
            ),
        ),
        (
            "(exp x)",
            Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into()))),
        ),
        (
            "(ln x)",
            Expr::Func(UnaryFn::Log, Box::new(Expr::Var("x".into()))),
        ),
        (
            "(mul (pow x 2) (pow y 2))",
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var("x".into())),
                    Box::new(Expr::Const(2.0)),
                )),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(2.0)),
                )),
            ),
        ),
    ]
}

fn canonicalize_small_forms(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonicalize_small_forms");

    for (expr, _) in benchmark_cases() {
        group.bench_function(expr, |b| b.iter(|| canonicalize(black_box(expr)).unwrap()));
    }

    group.finish();
}

fn equivalence_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("equivalence_checks");

    let cases = [
        ("(exp x)", "(eml x 1)"),
        ("(ln x)", "(eml 1 (eml (eml 1 x) 1))"),
        ("(pow x 2)", "(mul x x)"),
        ("(pow x 1)", "x"),
    ];

    for (lhs, rhs) in cases {
        group.bench_function(format!("{lhs} == {rhs}"), |b| {
            b.iter(|| equivalent(black_box(lhs), black_box(rhs)).unwrap())
        });
    }

    group.finish();
}

fn compare_with_current_symthaea_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_with_current_symthaea_path");

    for (label, expr) in benchmark_cases() {
        group.bench_function(format!("egg canonicalize {label}"), |b| {
            b.iter(|| canonicalize(black_box(label)).unwrap())
        });

        group.bench_function(format!("symthaea abstract canonical {label}"), |b| {
            b.iter(|| {
                let canonical = canonicalize_expr(black_box(&expr));
                expr_canonical_string(black_box(&canonical))
            })
        });

        group.bench_function(format!("symthaea strict eml compile {label}"), |b| {
            b.iter(|| {
                compile_expr(black_box(&expr))
                    .ok()
                    .map(|compiled| compiled.to_string())
            })
        });
    }

    group.finish();
}

fn stress_equivalence_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_equivalence_cases");

    let assoc_left = Expr::BinOp(
        BinOp::Mul,
        Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        )),
        Box::new(Expr::Var("z".into())),
    );
    let assoc_right = Expr::BinOp(
        BinOp::Mul,
        Box::new(Expr::Var("x".into())),
        Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("y".into())),
            Box::new(Expr::Var("z".into())),
        )),
    );

    group.bench_function("egg mul associativity equivalence", |b| {
        b.iter(|| {
            equivalent(
                black_box("(mul (mul x y) z)"),
                black_box("(mul x (mul y z))"),
            )
            .unwrap()
        })
    });

    group.bench_function("symthaea canonical string left-associated mul", |b| {
        b.iter(|| {
            let canonical = canonicalize_expr(black_box(&assoc_left));
            expr_canonical_string(black_box(&canonical))
        })
    });

    group.bench_function("symthaea canonical string right-associated mul", |b| {
        b.iter(|| {
            let canonical = canonicalize_expr(black_box(&assoc_right));
            expr_canonical_string(black_box(&canonical))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    canonicalize_small_forms,
    equivalence_checks,
    compare_with_current_symthaea_path,
    stress_equivalence_cases
);
criterion_main!(benches);
