// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproduce the Melothaea musical-context counterfactual baseline suite.
//!
//! Run:
//!   cargo run -p symthaea-muse --example mel_context_counterfactual
//!
//! The report keeps member-level action support, intent sensitivity, surface
//! invariance, and ambiguous-action coverage separate. The hand-written
//! context rule is a fixture sanity check, not a learned-model benchmark
//! result.

use serde::{Deserialize, Serialize};
use symthaea_muse::musical_context::{
    MusicalActionV1, MusicalContextCounterfactualAggregateV1,
    MusicalContextCounterfactualSetV1, score_musical_context_counterfactual_set_v1,
    validate_musical_context_counterfactual_set_v1,
};

const COUNTERFACTUALS_JSON: &str = include_str!(
    "../../../../benchmarks/mel/context/musical-invitation-counterfactual-v1.json"
);
const BASELINES_JSON: &str =
    include_str!("../../../../benchmarks/mel/context/mctx-baselines-v1.json");

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
    predictions_by_family: Vec<Vec<MusicalActionV1>>,
    expected: BaselineExpectedV1,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineExpectedV1 {
    member_valid_action_support_rate: f64,
    context_sensitivity_rate: f64,
    surface_invariance_rate: f64,
    ambiguous_valid_action_rate: f64,
}

#[derive(Debug, Serialize)]
struct BaselineReportV1 {
    schema_version: &'static str,
    baseline_id: String,
    description: String,
    expected: BaselineExpectedV1,
    metrics: MusicalContextCounterfactualAggregateV1,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus: MusicalContextCounterfactualSetV1 =
        serde_json::from_str(COUNTERFACTUALS_JSON)?;
    let baselines: BaselineSetV1 = serde_json::from_str(BASELINES_JSON)?;

    if baselines.schema_version != "mel-musical-context-baselines-v1" {
        return Err(std::io::Error::other(
            "unsupported musical-context baseline schema",
        )
        .into());
    }
    let validation = validate_musical_context_counterfactual_set_v1(&corpus);
    if !validation.is_empty() {
        return Err(std::io::Error::other(format!(
            "invalid counterfactual corpus: {validation:?}"
        ))
        .into());
    }

    let mut reports = Vec::with_capacity(baselines.baselines.len());
    for baseline in baselines.baselines {
        let metrics = score_musical_context_counterfactual_set_v1(
            &corpus,
            &baseline.predictions_by_family,
        )?;
        let actual = [
            metrics.member_valid_action_support_rate,
            metrics.context_sensitivity_rate.unwrap_or(f64::NAN),
            metrics.surface_invariance_rate.unwrap_or(f64::NAN),
            metrics.ambiguous_valid_action_rate.unwrap_or(f64::NAN),
        ];
        let expected = [
            baseline.expected.member_valid_action_support_rate,
            baseline.expected.context_sensitivity_rate,
            baseline.expected.surface_invariance_rate,
            baseline.expected.ambiguous_valid_action_rate,
        ];
        if actual
            .iter()
            .zip(expected.iter())
            .any(|(actual, expected)| !actual.is_finite() || (actual - expected).abs() > 1e-9)
        {
            return Err(std::io::Error::other(format!(
                "baseline {} diverged from its frozen expected metrics: actual={actual:?}, expected={expected:?}",
                baseline.baseline_id
            ))
            .into());
        }
        reports.push(BaselineReportV1 {
            schema_version: "mel-musical-context-counterfactual-report-v1",
            baseline_id: baseline.baseline_id,
            description: baseline.description,
            expected: baseline.expected,
            metrics,
        });
    }

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
