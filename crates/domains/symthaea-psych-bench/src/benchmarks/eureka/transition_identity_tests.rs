// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-layer transition-identity regressions.

use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::custody::PredictionCustodian;
use super::hidden_world::{
    CorpusPartition, EvaluatorWorld, FixtureFamily, PublicAction, WorldBuildProfile,
};

fn world(seed: u64) -> EvaluatorWorld {
    EvaluatorWorld::build(WorldBuildProfile {
        family: FixtureFamily::CausalBits,
        seed,
        mechanism_variant: 0,
        partition: CorpusPartition::HeldOutEvaluation,
    })
}

#[test]
fn custody_and_realized_action_use_one_public_transition_identity() {
    let mut custody_world = world(11);
    let mut receipt_world = world(11);
    let pre = custody_world.runtime().observe();
    let legal = custody_world.runtime().legal_actions();
    let action = PublicAction::NoOp;
    let candidate = ConsequencePrediction {
        action,
        outcome: PredictionOutcome::Predicted {
            fields: pre.fields.clone(),
        },
    };

    let mut custodian = PredictionCustodian::new(custody_world.world_digest());
    let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
    let fresh = custody_world.runtime().step(action);
    let trial = custodian.score_after_reveal(frozen, &fresh).unwrap();

    let receipt = receipt_world.execute_qualified_action(action);
    assert_eq!(receipt.world_digest, trial.world_digest);
    assert_eq!(receipt.realized, Some(action));
    assert_eq!(receipt.pre_state, pre);
    assert_eq!(receipt.post_state, Some(fresh.observation));
    assert_eq!(receipt.transition_digest, Some(trial.transition_digest));
}
