// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproduce the paired selective musical-participation baseline suite.
//!
//! Run:
//!   cargo run -p symthaea-muse --example mel_selective_participation
//!
//! The example consumes the exact frozen baseline predictions and expected
//! values used by the Rust unit tests. ALWAYS_WAIT and ALWAYS_RESPOND are
//! negative controls. The context rule is a fixture sanity policy, not a
//! learned-model or SOTA result.

use serde::{Deserialize, Serialize};
use symthaea_muse::musical_participation::{
    score_musical_participation_set_v1, MusicalParticipationAggregateV1,
    MusicalParticipationPredictionV1, MusicalParticipationSetV1,
};

const PARTICIPATION_JSON: &str = include_str!(
    "../../../../benchmarks/mel/participation/musical-participation-counterfactual-v1.json"
);
const BASELINES_JSON: &str = include_str!(
    "../../../../benchmarks/mel/participation/musical-participation-baselines-v1.json"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineSetV1 {
    schema_version: String,
    baselines: Vec<BaselineV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineV1 {
    baseline_id: String,
    description: String,
    predictions_by_family: Vec<Vec<MusicalParticipationPredictionV1>>,
    expected: BaselineExpectedV1,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineExpectedV1 {
    valid_decision_support_count: usize,
    paired_correct_count: usize,
    false_entry_count: usize,
    missed_invitation_count: usize,
    intentional_wait_count: usize,
    epistemic_abstain_count: usize,
    clarification_ask_count: usize,
    mean_brier_score: f64,
}

#[derive(Debug, Serialize)]
struct BaselineReportV1 {
    schema_version: &'static str,
    baseline_id: String,
    description: String,
    expected: BaselineExpectedV1,
    metrics: MusicalParticipationAggregateV1,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let participation: MusicalParticipationSetV1 =
        serde_json::from_str(PARTICIPATION_JSON)?;
    let baselines: BaselineSetV1 = serde_json::from_str(BASELINES_JSON)?;

    if baselines.schema_version != "mel-musical-participation-baselines-v1" {
        return Err(std::io::Error::other(
            "unsupported selective-participation baseline schema",
        )
        .into());
    }
    if baselines.baselines.len() < 3 {
        return Err(std::io::Error::other(
            "frozen baseline set must include constant-WAIT, constant-RESPOND and context sanity controls",
        )
        .into());
    }

    let mut reports = Vec::with_capacity(baselines.baselines.len());
    for baseline in baselines.baselines {
        let metrics = score_musical_participation_set_v1(
            &participation,
            &baseline.predictions_by_family,
        )?;
        let expected = &baseline.expected;
        if metrics.valid_decision_support_count != expected.valid_decision_support_count
            || metrics.paired_correct_count != expected.paired_correct_count
            || metrics.false_entry_count != expected.false_entry_count
            || metrics.missed_invitation_count != expected.missed_invitation_count
            || metrics.intentional_wait_count != expected.intentional_wait_count
            || metrics.epistemic_abstain_count != expected.epistemic_abstain_count
            || metrics.clarification_ask_count != expected.clarification_ask_count
            || (metrics.mean_brier_score - expected.mean_brier_score).abs() > 1e-9
        {
            return Err(std::io::Error::other(format!(
                "baseline {} diverged from frozen expected metrics",
                baseline.baseline_id
            ))
            .into());
        }
        reports.push(BaselineReportV1 {
            schema_version: "mel-musical-participation-report-v1",
            baseline_id: baseline.baseline_id,
            description: baseline.description,
            expected: baseline.expected,
            metrics,
        });
    }

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
