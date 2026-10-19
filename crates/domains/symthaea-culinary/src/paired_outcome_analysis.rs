// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-gated descriptive analysis of preregistered paired culinary runs.
//!
//! For each declared outcome, this module reports the variant-minus-control
//! contrast within each block, the mean of those paired contrasts, and a
//! conservative interval propagated from the declared measurement bounds.
//! Measurement-bound intervals are not confidence intervals and do not quantify
//! between-batch sampling uncertainty.
//!
//! Analysis fails closed unless the experiment contract reports all outcomes as
//! observed and all outcome evidence labels as qualified. Those labels are
//! declarations rather than authenticated facts. This module computes no
//! p-values, confidence intervals, causal effects, or universal recipe score.

use serde_json::Value;

use crate::culinary_experiment::validate_culinary_experiment_v1;

/// One block's paired variant-minus-control difference for an outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcomeDifferenceV1 {
    pub block_id: String,
    pub control_run_id: String,
    pub variant_run_id: String,
    /// Point difference: variant.value - control.value.
    pub difference: f64,
    /// Conservative measurement-bound difference:
    /// variant.lower - control.upper.
    pub lower_difference: f64,
    /// Conservative measurement-bound difference:
    /// variant.upper - control.lower.
    pub upper_difference: f64,
}

/// Descriptive summary for one preregistered outcome, kept separate from others.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcomeSummaryV1 {
    pub outcome_id: String,
    pub unit: String,
    pub direction: String,
    pub target_value: Option<f64>,
    pub block_count: usize,
    pub pairs: Vec<PairedOutcomeDifferenceV1>,
    pub mean_paired_difference: f64,
    pub mean_measurement_bound_lower: f64,
    pub mean_measurement_bound_upper: f64,
    pub minimum_paired_difference: f64,
    pub maximum_paired_difference: f64,
    /// Descriptive point-estimate comparisons only; not hypothesis tests.
    /// None means the preregistered direction is "observe".
    pub variant_directional_wins: Option<usize>,
    pub control_directional_wins: Option<usize>,
    pub directional_ties: Option<usize>,
}

/// Report of all declared outcome contrasts. No composite score is calculated.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcomeAnalysisV1 {
    pub experiment_id: String,
    pub protocol_id: String,
    pub protocol_version: String,
    pub protocol_sha256: String,
    pub primary_outcome_id: String,
    pub control_condition_id: String,
    pub variant_condition_id: String,
    pub block_count: usize,
    pub summaries: Vec<PairedOutcomeSummaryV1>,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct BoundedOutcome {
    value: f64,
    lower: f64,
    upper: f64,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|number| number.is_finite())
}

fn bounded_outcome(run: &Value, outcome_id: &str) -> Option<BoundedOutcome> {
    let records = run.get("outcomes")?.as_array()?;
    let mut matching = records.iter().filter(|record| {
        non_empty(record, "outcome_id") == Some(outcome_id)
    });
    let record = matching.next()?;
    if matching.next().is_some() || non_empty(record, "status") != Some("observed") {
        return None;
    }
    let value = finite_number(record, "value")?;
    let lower = finite_number(record, "lower")?;
    let upper = finite_number(record, "upper")?;
    (lower <= value && value <= upper).then_some(BoundedOutcome { value, lower, upper })
}

fn directional_tally(
    direction: &str,
    target: Option<f64>,
    difference: &PairedOutcomeDifferenceV1,
    control_value: f64,
    variant_value: f64,
) -> Option<OrderingTally> {
    let comparison = match direction {
        "maximize" => variant_value.total_cmp(&control_value),
        "minimize" => control_value.total_cmp(&variant_value),
        "target" => {
            let target = target?;
            (target - control_value).abs().total_cmp(&(target - variant_value).abs())
        }
        "observe" => return None,
        _ => return None,
    };
    let _ = difference;
    Some(match comparison {
        std::cmp::Ordering::Less => OrderingTally::ControlWin,
        std::cmp::Ordering::Equal => OrderingTally::Tie,
        std::cmp::Ordering::Greater => OrderingTally::VariantWin,
    })
}

#[derive(Debug, Clone, Copy)]
enum OrderingTally {
    VariantWin,
    ControlWin,
    Tie,
}

/// Analyze all declared outcomes, refusing any estimated, unknown, or
/// non-qualified evidence-bearing outcome. All numeric outputs are descriptive.
///
/// The report's measurement-bound interval captures only the intervals declared
/// on the paired measurements. It is not a confidence interval for an underlying
/// population effect and does not correct for confounding, drift, or bias.
pub fn analyze_paired_outcomes_v1(
    experiment: &Value,
) -> Result<PairedOutcomeAnalysisV1, Vec<String>> {
    let validation = match validate_culinary_experiment_v1(experiment) {
        Ok(report) => report,
        Err(errors) => return Err(errors),
    };
    if !validation.paired_analysis_ready {
        let mut blockers = validation.blockers;
        blockers.push(
            "paired outcome analysis requires every declared outcome to be observed with evidence status qualified"
                .to_owned(),
        );
        blockers.sort();
        blockers.dedup();
        return Err(blockers);
    }

    let control_condition_id = non_empty(experiment, "control_condition_id")
        .ok_or_else(|| vec!["control_condition_id missing after validation".to_owned()])?
        .to_owned();
    let variant_condition_id = non_empty(experiment, "variant_condition_id")
        .ok_or_else(|| vec!["variant_condition_id missing after validation".to_owned()])?
        .to_owned();
    let plan = experiment.get("analysis_plan")
        .ok_or_else(|| vec!["analysis_plan missing after validation".to_owned()])?;
    let primary_outcome_id = non_empty(plan, "primary_outcome_id")
        .ok_or_else(|| vec!["primary_outcome_id missing after validation".to_owned()])?
        .to_owned();
    let definitions = experiment.get("outcome_definitions")
        .and_then(Value::as_array)
        .ok_or_else(|| vec!["outcome_definitions missing after validation".to_owned()])?;
    let blocks = experiment.get("blocks")
        .and_then(Value::as_array)
        .ok_or_else(|| vec!["blocks missing after validation".to_owned()])?;

    let mut summaries = Vec::with_capacity(definitions.len());
    for definition in definitions {
        let outcome_id = non_empty(definition, "outcome_id")
            .ok_or_else(|| vec!["outcome_id missing after validation".to_owned()])?;
        let unit = non_empty(definition, "unit")
            .ok_or_else(|| vec![format!("unit missing for {outcome_id:?}")])?;
        let direction = non_empty(definition, "direction")
            .ok_or_else(|| vec![format!("direction missing for {outcome_id:?}")])?;
        let target_value = finite_number(definition, "target_value");
        let mut pairs = Vec::with_capacity(blocks.len());
        let mut variant_wins = 0;
        let mut control_wins = 0;
        let mut ties = 0;

        for block in blocks {
            let block_id = non_empty(block, "block_id")
                .ok_or_else(|| vec!["block_id missing after validation".to_owned()])?;
            let runs = block.get("runs")
                .and_then(Value::as_array)
                .ok_or_else(|| vec![format!("runs missing for block {block_id:?}")])?;
            let mut control_run = None;
            let mut variant_run = None;
            for run in runs {
                match non_empty(run, "condition_id") {
                    Some(condition) if condition == control_condition_id => control_run = Some(run),
                    Some(condition) if condition == variant_condition_id => variant_run = Some(run),
                    _ => {}
                }
            }
            let control_run = control_run
                .ok_or_else(|| vec![format!("control run missing for block {block_id:?}")])?;
            let variant_run = variant_run
                .ok_or_else(|| vec![format!("variant run missing for block {block_id:?}")])?;
            let control = bounded_outcome(control_run, outcome_id)
                .ok_or_else(|| vec![format!("qualified observed control outcome {outcome_id:?} missing in block {block_id:?}")])?;
            let variant = bounded_outcome(variant_run, outcome_id)
                .ok_or_else(|| vec![format!("qualified observed variant outcome {outcome_id:?} missing in block {block_id:?}")])?;

            let pair = PairedOutcomeDifferenceV1 {
                block_id: block_id.to_owned(),
                control_run_id: non_empty(control_run, "run_id").unwrap_or_default().to_owned(),
                variant_run_id: non_empty(variant_run, "run_id").unwrap_or_default().to_owned(),
                difference: variant.value - control.value,
                lower_difference: variant.lower - control.upper,
                upper_difference: variant.upper - control.lower,
            };
            if !pair.difference.is_finite()
                || !pair.lower_difference.is_finite()
                || !pair.upper_difference.is_finite()
            {
                return Err(vec![format!("non-finite paired difference for outcome {outcome_id:?}")]);
            }
            if let Some(tally) = directional_tally(
                direction,
                target_value,
                &pair,
                control.value,
                variant.value,
            ) {
                match tally {
                    OrderingTally::VariantWin => variant_wins += 1,
                    OrderingTally::ControlWin => control_wins += 1,
                    OrderingTally::Tie => ties += 1,
                }
            }
            pairs.push(pair);
        }

        pairs.sort_by(|left, right| left.block_id.cmp(&right.block_id));
        if pairs.is_empty() {
            return Err(vec![format!("no paired blocks for outcome {outcome_id:?}")]);
        }
        let count = pairs.len() as f64;
        let mean_paired_difference = pairs.iter().map(|pair| pair.difference).sum::<f64>() / count;
        let mean_measurement_bound_lower =
            pairs.iter().map(|pair| pair.lower_difference).sum::<f64>() / count;
        let mean_measurement_bound_upper =
            pairs.iter().map(|pair| pair.upper_difference).sum::<f64>() / count;
        let minimum_paired_difference = pairs.iter()
            .map(|pair| pair.difference)
            .fold(f64::INFINITY, f64::min);
        let maximum_paired_difference = pairs.iter()
            .map(|pair| pair.difference)
            .fold(f64::NEG_INFINITY, f64::max);
        let directional = (direction != "observe").then_some(());
        summaries.push(PairedOutcomeSummaryV1 {
            outcome_id: outcome_id.to_owned(),
            unit: unit.to_owned(),
            direction: direction.to_owned(),
            target_value,
            block_count: pairs.len(),
            pairs,
            mean_paired_difference,
            mean_measurement_bound_lower,
            mean_measurement_bound_upper,
            minimum_paired_difference,
            maximum_paired_difference,
            variant_directional_wins: directional.map(|_| variant_wins),
            control_directional_wins: directional.map(|_| control_wins),
            directional_ties: directional.map(|_| ties),
        });
    }

    Ok(PairedOutcomeAnalysisV1 {
        experiment_id: non_empty(experiment, "experiment_id").unwrap_or_default().to_owned(),
        protocol_id: non_empty(experiment, "protocol_id").unwrap_or_default().to_owned(),
        protocol_version: non_empty(experiment, "protocol_version").unwrap_or_default().to_owned(),
        protocol_sha256: non_empty(experiment, "protocol_sha256").unwrap_or_default().to_owned(),
        primary_outcome_id,
        control_condition_id,
        variant_condition_id,
        block_count: blocks.len(),
        summaries,
        assumptions: vec![
            "All summaries are descriptive paired contrasts; no p-values, statistical confidence intervals or causal claims are calculated.".to_owned(),
            "Mean measurement bounds are propagated from declared per-run intervals. They are not sampling intervals and depend on those bounds being meaningful.".to_owned(),
            "Evidence status labels are declarations and are not authenticated by this module.".to_owned(),
            "The analysis plan freezes variant-minus-control as estimand, mean paired difference as summary method, fail-closed missing-data handling, and no population inference.".to_owned(),
            "All declared outcomes are reported independently; no composite winner is produced.".to_owned(),
        ],
    })
}
