// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproduce the paired selective musical-participation fixture.
//!
//! Run:
//!   cargo run -p symthaea-muse --example mel_selective_participation
//!
//! ALWAYS_WAIT and ALWAYS_RESPOND are negative controls. The context-rule
//! sanity policy is a hand-written fixture oracle, not a learned model result.

use serde::Serialize;
use symthaea_muse::musical_participation::{
    score_musical_participation_set_v1, MusicalParticipationDecisionV1 as Decision,
    MusicalParticipationPredictionV1 as Prediction, MusicalParticipationSetV1,
};

const FIXTURE_JSON: &str = include_str!(
    "../../../../benchmarks/mel/participation/musical-participation-counterfactual-v1.json"
);

#[derive(Serialize)]
struct BaselineReport {
    baseline_id: &'static str,
    description: &'static str,
    metrics: symthaea_muse::musical_participation::MusicalParticipationAggregateV1,
}

fn prediction(decision: Decision, confidence: f64, act_probability: f64) -> Prediction {
    Prediction {
        decision,
        decision_confidence: confidence,
        act_probability,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let set: MusicalParticipationSetV1 = serde_json::from_str(FIXTURE_JSON)?;
    let family_count = set.families.len();

    let always_wait = vec![
        vec![
            prediction(Decision::Wait, 0.95, 0.05),
            prediction(Decision::Wait, 0.95, 0.05),
        ];
        family_count
    ];
    let always_respond = vec![
        vec![
            prediction(Decision::Respond, 0.95, 0.95),
            prediction(Decision::Respond, 0.95, 0.95),
        ];
        family_count
    ];

    // Frozen symbolic sanity policy: each action is authored against the
    // checked-in controlled fixture. It is not an inference/model benchmark.
    let context_rule = vec![
        vec![
            prediction(Decision::Respond, 0.90, 0.90),
            prediction(Decision::Wait, 0.90, 0.10),
        ],
        vec![
            prediction(Decision::Continue, 0.85, 0.90),
            prediction(Decision::Ask, 0.70, 0.10),
        ],
        vec![
            prediction(Decision::Respond, 0.90, 0.90),
            prediction(Decision::Abstain, 0.80, 0.10),
        ],
        vec![
            prediction(Decision::Yield, 0.90, 0.90),
            prediction(Decision::Wait, 0.90, 0.10),
        ],
    ];

    if set.families.len() != context_rule.len() {
        return Err(std::io::Error::other(
            "the frozen context-rule baseline does not match the fixture family count",
        )
        .into());
    }

    let reports = vec![
        BaselineReport {
            baseline_id: "ALWAYS-WAIT-001",
            description: "Negative control: always preserves silence, including on valid invitations.",
            metrics: score_musical_participation_set_v1(&set, &always_wait)?,
        },
        BaselineReport {
            baseline_id: "ALWAYS-RESPOND-001",
            description: "Negative control: always contributes, including when withholding is required.",
            metrics: score_musical_participation_set_v1(&set, &always_respond)?,
        },
        BaselineReport {
            baseline_id: "CONTEXT-RULE-FIXTURE-SANITY-001",
            description: "Hand-written fixture sanity policy; not a learned-model result.",
            metrics: score_musical_participation_set_v1(&set, &context_rule)?,
        },
    ];

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
