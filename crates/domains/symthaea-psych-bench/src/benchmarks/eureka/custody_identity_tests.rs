// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent regression tests for EUREKA prospective commitment identity.
//!
//! These tests deliberately use fresh custodians so logical sequence remains
//! identical. That isolates prediction semantics from chronology when checking
//! commitment identity.

use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::custody::PredictionCustodian;
use super::hidden_world::{
    CorpusPartition, EvaluatorWorld, FixtureFamily, PublicAction, PublicValue, WorldBuildProfile,
};

fn evaluator(seed: u64) -> EvaluatorWorld {
    EvaluatorWorld::build(WorldBuildProfile {
        family: FixtureFamily::CausalBits,
        seed,
        mechanism_variant: 0,
        partition: CorpusPartition::HeldOutEvaluation,
    })
}

fn prediction(action: PublicAction, fields: Vec<PublicValue>) -> ConsequencePrediction {
    ConsequencePrediction {
        action,
        outcome: PredictionOutcome::Predicted { fields },
    }
}

#[test]
fn same_prediction_same_world_same_sequence_has_same_commitment() {
    let mut world = evaluator(3);
    let world_digest = world.world_digest();
    let pre = world.runtime().observe();
    let legal = world.runtime().legal_actions();
    let candidate = prediction(PublicAction::NoOp, pre.fields.clone());

    let mut custodian_a = PredictionCustodian::new(world_digest);
    let mut custodian_b = PredictionCustodian::new(world_digest);
    let frozen_a = custodian_a.freeze(&pre, &legal, &candidate).unwrap();
    let frozen_b = custodian_b.freeze(&pre, &legal, &candidate).unwrap();

    assert_eq!(frozen_a.commit_sequence(), 1);
    assert_eq!(frozen_b.commit_sequence(), 1);
    assert_eq!(frozen_a.commitment_digest(), frozen_b.commitment_digest());
}

#[test]
fn changing_only_prediction_content_changes_commitment() {
    let mut world = evaluator(3);
    let world_digest = world.world_digest();
    let pre = world.runtime().observe();
    let legal = world.runtime().legal_actions();

    let mut fields_a = pre.fields.clone();
    let mut fields_b = pre.fields.clone();
    fields_a[0] = PublicValue::Bit(false);
    fields_b[0] = PublicValue::Bit(true);

    let mut custodian_a = PredictionCustodian::new(world_digest);
    let mut custodian_b = PredictionCustodian::new(world_digest);
    let frozen_a = custodian_a
        .freeze(&pre, &legal, &prediction(PublicAction::NoOp, fields_a))
        .unwrap();
    let frozen_b = custodian_b
        .freeze(&pre, &legal, &prediction(PublicAction::NoOp, fields_b))
        .unwrap();

    assert_eq!(frozen_a.commit_sequence(), 1);
    assert_eq!(frozen_b.commit_sequence(), 1);
    assert_ne!(frozen_a.commitment_digest(), frozen_b.commitment_digest());
}
