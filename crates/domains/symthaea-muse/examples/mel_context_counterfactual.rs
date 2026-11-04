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
}

#[derive(Debug, Serialize)]
struct BaselineReportV1 {
    schema_version: &'static str,
    baseline_id: String,
    description: String,
    metrics: MusicalContextCounterfactualAggregateV1,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus: MusicalContextCounterfactualSetV1 =
        serde_json::from_str(COUNTERFACTUALS_JSON)?;
    let baselines: BaselineSetV1 = serde_json::from_str(BASELINES_JSON)?;

    if baselines.schema_version != "mel-musical-context-baselines-v1" {
        return Err("unsupported musical-context baseline schema".into());
    }
    let validation = validate_musical_context_counterfactual_set_v1(&corpus);
    if !validation.is_empty() {
        return Err(format!("invalid counterfactual corpus: {validation:?}").into());
    }

    let mut reports = Vec::with_capacity(baselines.baselines.len());
    for baseline in baselines.baselines {
        let metrics = score_musical_context_counterfactual_set_v1(
            &corpus,
            &baseline.predictions_by_family,
        )?;
        reports.push(BaselineReportV1 {
            schema_version: "mel-musical-context-counterfactual-report-v1",
            baseline_id: baseline.baseline_id,
            description: baseline.description,
            metrics,
        });
    }

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
