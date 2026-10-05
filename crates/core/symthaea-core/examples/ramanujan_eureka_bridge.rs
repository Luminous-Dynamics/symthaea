// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic Ramanujan × EUREKA integration showcase.
//!
//! This example demonstrates the architecture only:
//! Ramanujan produces typed hypotheses, EUREKA-style inquiry selects a fresh
//! discriminating experiment, and later model revision becomes a new lineage.
//! None of the generated receipts constitute scientific confirmation.

use std::collections::HashMap;

use symthaea_core::hdc::conjecture_engine::{
    BinOp, Conjecture, ConjectureStatus, Expr, MathDomain, MacroPromotionTier,
    select_most_discriminative_experiment_with_receipt, scientific_hypothesis_set_digest,
};
use symthaea_types::{
    ModelMaturity, PhysicalDimension, PhysicalType, QuantityKind,
    ScientificHypothesisHandoff, ScientificHypothesisRevisionReceipt,
};

fn candidate(formula: Expr, source: &str) -> Conjecture {
    let complexity = formula.complexity();
    Conjecture {
        formula: formula.clone(),
        formula_str: formula.to_string(),
        source: source.into(),
        domain: MathDomain::Physics,
        training_mse: 0.0,
        complexity,
        fitness: complexity as f64,
        status: ConjectureStatus::Proposed,
        confidence: 0.5,
        macro_promotion_tier: MacroPromotionTier::RecurrentNumerical,
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

fn handoff(conjecture: &Conjecture) -> ScientificHypothesisHandoff {
    let energy = PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
    conjecture.scientific_hypothesis_handoff(
        &energy,
        ModelMaturity::ResearchPrototype,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        42,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        vec![
            "not empirical confirmation".into(),
            "not formal proof of the physical world".into(),
            "not engineering qualification".into(),
        ],
    )
}

fn main() {
    let x = Expr::Var("x".into());
    let v = Expr::Var("v".into());
    let h1 = Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::BinOp(
            BinOp::Pow,
            Box::new(x.clone()),
            Box::new(Expr::Const(2.0)),
        )),
        Box::new(Expr::BinOp(
            BinOp::Pow,
            Box::new(v.clone()),
            Box::new(Expr::Const(2.0)),
        )),
    );
    let h2 = Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::BinOp(
            BinOp::Pow,
            Box::new(x),
            Box::new(Expr::Const(2.0)),
        )),
        Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(v),
                Box::new(Expr::Const(2.0)),
            )),
        )),
    );

    let c1 = candidate(h1, "harmonic_candidate_h1");
    let c2 = candidate(h2, "harmonic_candidate_h2");
    let h1_ref = handoff(&c1);
    let h2_ref = handoff(&c2);

    assert!(h1_ref.validate().is_ok());
    assert!(h2_ref.validate().is_ok());

    let handoffs = vec![h1_ref.clone(), h2_ref.clone()];
    let hypothesis_set_digest = scientific_hypothesis_set_digest(&handoffs);

    // Candidate experiments are fresh initial conditions. Finite prediction
    // disagreement is the selector's criterion; it is not experimental evidence.
    let experiments = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 3.0)];
    let hypotheses = vec![c1.formula.clone(), c2.formula.clone()];
    let predict = |formula: &Expr, initial: &(f64, f64)| {
        let value = formula.eval(&[("x", initial.0), ("v", initial.1)]);
        value.is_finite().then_some(value)
    };

    let (selected, selection) = select_most_discriminative_experiment_with_receipt(
        &experiments,
        &hypotheses,
        predict,
        &h1_ref.digest_hex(),
        &hypothesis_set_digest,
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "ramanujan-eureka-selector-v2",
        7,
        |initial| format!("harmonic:{:.6}:{:.6}", initial.0, initial.1),
    )
    .expect("selection receipt should be structurally valid")
    .expect("candidate experiment pool is non-empty");

    assert_eq!(*selected, (2.0, 3.0));
    assert!(selection.validate().is_ok());

    // A later revision is a new lineage, never a mutation of the original handoff.
    let revised = ScientificHypothesisHandoff::new(
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        &PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY),
        ModelMaturity::ValidatedNumerical,
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "1111111111111111111111111111111111111111111111111111111111111111",
        43,
        "2222222222222222222222222222222222222222222222222222222222222222",
        c1.complexity as u64,
        "ramanujan",
        vec![
            "not a universal physical law".into(),
            "not engineering certification".into(),
        ],
    );

    let revision = ScientificHypothesisRevisionReceipt::new(
        h1_ref.digest_hex(),
        revised.digest_hex(),
        selection.digest_hex(),
        revised.campaign_digest_hex(),
        "candidate-refinement-after-independent-challenge",
    )
    .expect("revision receipt should be structurally valid");

    assert!(revision.validate().is_ok());
    assert!(revision
        .validate_against(&h1_ref, &revised)
        .is_ok());
    assert_ne!(revision.prior_handoff_digest, revision.new_handoff_digest);
    assert_ne!(
        h1_ref.campaign_digest_hex(),
        revised.campaign_digest_hex()
    );

    let mut summary = HashMap::new();
    summary.insert("hypothesis_set_digest", hypothesis_set_digest);
    summary.insert("selected_challenge_digest", selection.selected_challenge_digest.clone());

    println!("Ramanujan -> EUREKA bridge");
    println!("  H1 candidate: {}", h1_ref.candidate_digest);
    println!("  H2 candidate: {}", h2_ref.candidate_digest);
    println!("  hypothesis set: {}", summary["hypothesis_set_digest"]);
    println!("  selected challenge: {}", summary["selected_challenge_digest"]);
    println!(
        "  predicted disagreement score: {:.6}",
        selection.predicted_disagreement_score()
    );
    println!("  selection receipt: {}", selection.digest_hex());
    println!("  revision receipt: {}", revision.digest_hex());
    println!("  status: receipts are identity/selection artifacts, not scientific confirmation");
}
