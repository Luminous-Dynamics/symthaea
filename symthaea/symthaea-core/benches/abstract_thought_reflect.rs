// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symthaea_core::hdc::abstract_thought::AbstractThought;
use symthaea_core::hdc::conjecture_engine::{
    attach_eml_metadata, BinOp, Conjecture, ConjectureEngine, ConjectureStatus, Expr,
    MacroPromotionTier, MathDomain, UnaryFn,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

fn make_verified_conjecture(
    name: &str,
    formula: Expr,
    domain: MathDomain,
    source: &str,
) -> Conjecture {
    let complexity = formula.complexity();
    Conjecture {
        formula,
        formula_str: name.to_string(),
        source: source.to_string(),
        domain,
        training_mse: 0.001,
        complexity,
        fitness: 0.001 + 0.001 * complexity as f64,
        status: ConjectureStatus::FormallyVerified { proof_steps: 5 },
        confidence: 0.95,
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

fn strict_eml_fast_track_expr() -> Expr {
    Expr::BinOp(
        BinOp::Div,
        Box::new(Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Var("y".to_string())),
    )
}

fn constructive_eml_expr() -> Expr {
    Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Var("y".to_string())),
    )
}

fn simple_unary_strict_expr() -> Expr {
    Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".to_string())))
}

fn mixed_eml_conjectures() -> Vec<Conjecture> {
    let mut conjectures = Vec::new();

    let strict_domains = [
        (MathDomain::NumberTheory, "strict_nt"),
        (MathDomain::Physics, "strict_phys"),
        (MathDomain::Chemistry, "strict_chem"),
    ];
    for (domain, source) in strict_domains {
        let mut conjecture = make_verified_conjecture(
            "strict_fast_track",
            strict_eml_fast_track_expr(),
            domain,
            source,
        );
        attach_eml_metadata(&mut conjecture);
        conjectures.push(conjecture);
    }

    let constructive_domains = [
        (MathDomain::Biology, "constructive_bio"),
        (MathDomain::Economics, "constructive_econ"),
        (MathDomain::Combinatorics, "constructive_comb"),
    ];
    for (domain, source) in constructive_domains {
        let mut conjecture =
            make_verified_conjecture("constructive_pair", constructive_eml_expr(), domain, source);
        attach_eml_metadata(&mut conjecture);
        conjectures.push(conjecture);
    }

    let mut deferred = make_verified_conjecture(
        "simple_unary_strict",
        simple_unary_strict_expr(),
        MathDomain::InformationTheory,
        "simple_unary_strict",
    );
    attach_eml_metadata(&mut deferred);
    conjectures.push(deferred);

    conjectures
}

fn run_reflect(
    cap: usize,
    conjectures: &[Conjecture],
    primitives: &PrimitiveSystem,
) -> ConjectureEngine {
    let mut engine = ConjectureEngine::new();
    engine.enable_abstract_thought();
    engine
        .abstract_thought
        .as_mut()
        .expect("abstract thought enabled")
        .dynamic_grammar
        .max_operators = cap;
    engine.conjectures = conjectures.to_vec();
    engine.reflect(primitives);
    engine
}

fn make_engine_with_conjectures(conjectures: &[Conjecture]) -> ConjectureEngine {
    let mut engine = ConjectureEngine::new();
    engine.conjectures = conjectures.to_vec();
    engine
}

fn setup_encoded_and_clustered(
    conjectures: &[Conjecture],
    primitives: &PrimitiveSystem,
) -> (ConjectureEngine, AbstractThought) {
    let engine = make_engine_with_conjectures(conjectures);
    let mut at = AbstractThought::new();

    for (i, conjecture) in engine.conjectures.iter().enumerate() {
        if conjecture.confidence >= 0.3 {
            at.meta_hdc.add_conjecture(conjecture, i, primitives);
        }
    }

    let n = at.meta_hdc.concepts.len();
    if n >= 3 {
        let k = (n / 3).max(2).min(10);
        at.meta_hdc.cluster(k);
    }

    (engine, at)
}

fn populate_candidates(at: &mut AbstractThought, engine: &ConjectureEngine) {
    let cluster_recurring = at.meta_hdc.recurring_subtrees(engine);
    for (subtree, ids) in cluster_recurring {
        at.dynamic_grammar.observe_subtree(subtree, &ids, engine);
    }

    let global_recurring = at.meta_hdc.global_recurring_subtrees(engine, 2);
    for (subtree, ids) in global_recurring {
        at.dynamic_grammar.observe_subtree(subtree, &ids, engine);
    }

    let verified_only = at.meta_hdc.verified_subtrees(engine);
    for (subtree, ids) in verified_only {
        at.dynamic_grammar.observe_subtree(subtree, &ids, engine);
    }
}

fn bench_reflect(c: &mut Criterion) {
    let primitives = PrimitiveSystem::new();
    let conjectures = mixed_eml_conjectures();
    let mut group = c.benchmark_group("abstract_thought_reflect");

    group.bench_function("mixed_eml_cap1", |b| {
        b.iter(|| {
            let engine = run_reflect(1, black_box(&conjectures), black_box(&primitives));
            black_box(engine.macro_operators().len());
        })
    });

    group.bench_function("mixed_eml_cap2", |b| {
        b.iter(|| {
            let engine = run_reflect(2, black_box(&conjectures), black_box(&primitives));
            black_box(engine.macro_operators().len());
        })
    });

    group.finish();
}

fn bench_reflect_phases(c: &mut Criterion) {
    let primitives = PrimitiveSystem::new();
    let conjectures = mixed_eml_conjectures();
    let mut group = c.benchmark_group("abstract_thought_reflect_phases");

    group.bench_function("encode_and_cluster", |b| {
        b.iter(|| {
            let (_engine, at) =
                setup_encoded_and_clustered(black_box(&conjectures), black_box(&primitives));
            black_box(at.meta_hdc.concepts.len());
        })
    });

    group.bench_function("extract_and_observe", |b| {
        b.iter(|| {
            let (engine, mut at) =
                setup_encoded_and_clustered(black_box(&conjectures), black_box(&primitives));
            populate_candidates(&mut at, &engine);
            black_box(at.dynamic_grammar.candidates.len());
        })
    });

    group.bench_function("promote_eligible", |b| {
        b.iter(|| {
            let (engine, mut at) =
                setup_encoded_and_clustered(black_box(&conjectures), black_box(&primitives));
            at.dynamic_grammar.max_operators = 2;
            populate_candidates(&mut at, &engine);
            at.dynamic_grammar.promote_eligible(&engine);
            black_box(at.dynamic_grammar.operators.len());
        })
    });

    group.bench_function("prune_unused", |b| {
        b.iter(|| {
            let (engine, mut at) =
                setup_encoded_and_clustered(black_box(&conjectures), black_box(&primitives));
            at.dynamic_grammar.max_operators = 2;
            populate_candidates(&mut at, &engine);
            at.dynamic_grammar.promote_eligible(&engine);
            at.dynamic_grammar.cycle = 20;
            at.dynamic_grammar.prune_unused();
            black_box(at.dynamic_grammar.operators.len());
        })
    });

    group.finish();
}

fn bench_macro_pool_metrics(c: &mut Criterion) {
    let primitives = PrimitiveSystem::new();
    let conjectures = mixed_eml_conjectures();
    let engine = run_reflect(2, &conjectures, &primitives);

    c.bench_function("abstract_thought_macro_pool_metrics_after_reflect", |b| {
        b.iter(|| {
            black_box(engine.macro_pool_metrics());
        })
    });
}

criterion_group!(
    benches,
    bench_reflect,
    bench_reflect_phases,
    bench_macro_pool_metrics
);
criterion_main!(benches);
