// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pre-registered paired-batch experiment contract for reproducible culinary R&D.
//!
//! This module validates study structure and the evidence labels attached to
//! outcome measurements. It does not compute statistical significance, infer
//! causality, authenticate sources, or determine food safety.
//!
//! The intended unit of comparison is a block containing one control run and
//! one variant run. Assignment order is recorded before the runs and checked
//! against the observed execution order. Outcome values remain distinct by
//! unit, direction, method and evidence lineage; no universal composite score
//! is produced.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

const DIRECTIONS: &[&str] = &["minimize", "maximize", "target", "observe"];
const VALUE_STATUSES: &[&str] = &["observed", "estimated", "unknown"];
const EVIDENCE_STATUSES: &[&str] = &["synthetic", "unreviewed", "reviewed", "qualified"];

/// Summary of structural validity and analysis-readiness blockers.
///
/// `paired_analysis_ready` means only that this declared dataset passed the
/// contract's completeness/evidence-label gates. It is not proof of source
/// authenticity, a successful power analysis, causality, safety, or benefit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CulinaryExperimentResultV1 {
    pub block_count: usize,
    pub run_count: usize,
    pub measurement_count: usize,
    pub observed_count: usize,
    pub estimated_count: usize,
    pub unknown_count: usize,
    pub all_evidence_records_marked_qualified: bool,
    pub paired_analysis_ready: bool,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UtcTimestamp {
    whole_seconds: i64,
    fractional_nanos: u32,
}

impl UtcTimestamp {
    fn compare(&self, other: &Self) -> Ordering {
        self.whole_seconds
            .cmp(&other.whole_seconds)
            .then_with(|| self.fractional_nanos.cmp(&other.fractional_nanos))
    }
}

#[derive(Debug, Clone)]
struct OutcomeDefinition {
    unit: String,
    direction: String,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|number| number.is_finite())
}

fn parse_digits(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: u32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

fn days_from_civil(year: u32, month: u32, day: u32) -> i64 {
    let mut y = i64::from(year);
    if month <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let adjusted_month = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era
}

/// Strict UTC RFC3339 subset parser. Fractional seconds are kept separately
/// from whole seconds so tiny gaps cannot vanish in epoch-sized float arithmetic.
fn parse_utc_timestamp(value: &str) -> Option<UtcTimestamp> {
    if !value.is_ascii() || !value.ends_with('Z') {
        return None;
    }
    let without_z = value.strip_suffix('Z')?;
    let (clock, fraction) = match without_z.split_once('.') {
        Some((clock, fraction)) => {
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            (clock, Some(fraction))
        }
        None => (without_z, None),
    };
    let bytes = clock.as_bytes();
    if bytes.len() != 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let year = parse_digits(&clock[0..4])?;
    let month = parse_digits(&clock[5..7])?;
    let day = parse_digits(&clock[8..10])?;
    let hour = parse_digits(&clock[11..13])?;
    let minute = parse_digits(&clock[14..16])?;
    let second = parse_digits(&clock[17..19])?;
    if year == 0
        || month == 0
        || day == 0
        || day > days_in_month(year, month)?
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let fraction_digits = fraction.unwrap_or("");
    let first_nine = &fraction_digits[..fraction_digits.len().min(9)];
    let mut nano_text = first_nine.to_owned();
    while nano_text.len() < 9 {
        nano_text.push('0');
    }
    let fractional_nanos = if nano_text.is_empty() { 0 } else { nano_text.parse().ok()? };
    let whole_seconds = days_from_civil(year, month, day) * 86_400
        + i64::from(hour) * 3_600
        + i64::from(minute) * 60
        + i64::from(second);
    Some(UtcTimestamp { whole_seconds, fractional_nanos })
}

fn validate_evidence(
    evidence: &Value,
    label: &str,
    errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) -> bool {
    let source = non_empty(evidence, "source_ref");
    let version = non_empty(evidence, "source_version");
    let known_at = non_empty(evidence, "known_at");
    let status = non_empty(evidence, "evidence_status");

    if source.is_none() {
        errors.push(format!("{label}.source_ref is required"));
    }
    if version.is_none() {
        errors.push(format!("{label}.source_version is required"));
    }
    if !known_at.is_some_and(|timestamp| parse_utc_timestamp(timestamp).is_some()) {
        errors.push(format!("{label}.known_at must be a valid UTC timestamp ending in Z"));
    }
    if !status.is_some_and(|value| EVIDENCE_STATUSES.contains(&value)) {
        errors.push(format!("{label}.evidence_status is invalid"));
    }
    if status == Some("qualified") && source.is_some_and(|value| value.starts_with("urn:synthetic:")) {
        errors.push(format!("{label} cannot mark a synthetic source qualified"));
        return false;
    }

    let qualified = status == Some("qualified")
        && source.is_some()
        && version.is_some()
        && known_at.is_some_and(|timestamp| parse_utc_timestamp(timestamp).is_some());
    if !qualified {
        blockers.push(format!(
            "{label} is not evidence-qualified (declared status: {:?})",
            status.unwrap_or("missing")
        ));
    }
    qualified
}

fn validate_outcome_value(
    value: &Value,
    label: &str,
    definition: &OutcomeDefinition,
    errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) -> Option<&'static str> {
    if non_empty(value, "unit") != Some(definition.unit.as_str()) {
        errors.push(format!("{label}.unit must match declared unit {:?}", definition.unit));
    }
    if non_empty(value, "method").is_none() {
        errors.push(format!("{label}.method is required"));
    }
    let status = non_empty(value, "status");
    if !status.is_some_and(|item| VALUE_STATUSES.contains(&item)) {
        errors.push(format!("{label}.status must be observed, estimated, or unknown"));
        return None;
    }

    match status {
        Some("observed" | "estimated") => {
            let reading = finite_number(value, "value");
            let lower = finite_number(value, "lower");
            let upper = finite_number(value, "upper");
            match (reading, lower, upper) {
                (Some(point), Some(lo), Some(hi)) if lo <= point && point <= hi => {}
                _ => errors.push(format!(
                    "{label} must have finite lower <= value <= upper bounds"
                )),
            }
            if non_empty(value, "unknown_reason").is_some() {
                errors.push(format!("{label} must not carry unknown_reason when numeric values exist"));
            }
            if status == Some("estimated") {
                blockers.push(format!("{label} is estimated, not an empirical observation"));
            }
        }
        Some("unknown") => {
            if !["value", "lower", "upper"]
                .iter()
                .all(|key| value.get(key).is_some_and(Value::is_null))
            {
                errors.push(format!("{label} unknown status requires value/lower/upper null"));
            }
            if non_empty(value, "unknown_reason").is_none() {
                errors.push(format!("{label} unknown status requires unknown_reason"));
            }
            blockers.push(format!("{label} is explicitly unknown"));
        }
        _ => {}
    }

    Some(status.unwrap_or("invalid"))
}

/// Validate pre-registration, paired control/variant blocks, execution order,
/// outcome completeness, units, uncertainty intervals, and evidence lineage.
///
/// A structurally valid study with synthetic/estimated/missing outcomes returns
/// a result with blockers and `paired_analysis_ready == false`; invalid design
/// or malformed records return errors. No statistical significance is computed.
pub fn validate_culinary_experiment_v1(
    experiment: &Value,
) -> Result<CulinaryExperimentResultV1, Vec<String>> {
    let mut errors = Vec::new();
    let mut blockers = Vec::new();
    let mut all_qualified = true;

    if non_empty(experiment, "schema_version") != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    for key in ["experiment_id", "protocol_id", "protocol_version"] {
        if non_empty(experiment, key).is_none() {
            errors.push(format!("{key} is required"));
        }
    }
    let protocol_digest = non_empty(experiment, "protocol_sha256");
    if !protocol_digest.is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        errors.push("protocol_sha256 must be 64 hexadecimal characters".to_owned());
    }

    let preregistered_at = non_empty(experiment, "preregistered_at")
        .and_then(parse_utc_timestamp);
    if preregistered_at.is_none() {
        errors.push("preregistered_at must be a valid UTC timestamp ending in Z".to_owned());
    }

    let control = non_empty(experiment, "control_condition_id");
    let variant = non_empty(experiment, "variant_condition_id");
    if control.is_none() || variant.is_none() || control == variant {
        errors.push("control_condition_id and variant_condition_id must be distinct non-empty IDs".to_owned());
    }

    let outcome_defs: &[Value] = match experiment.get("outcome_definitions").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("outcome_definitions must be a non-empty array".to_owned());
            &[]
        }
    };
    let mut outcome_definitions = BTreeMap::<String, OutcomeDefinition>::new();
    for (index, value) in outcome_defs.iter().enumerate() {
        let label = format!("outcome_definitions[{index}]");
        let Some(id) = non_empty(value, "outcome_id") else {
            errors.push(format!("{label}.outcome_id is required"));
            continue;
        };
        let Some(unit) = non_empty(value, "unit") else {
            errors.push(format!("{label}.unit is required"));
            continue;
        };
        let Some(direction) = non_empty(value, "direction") else {
            errors.push(format!("{label}.direction is required"));
            continue;
        };
        if !DIRECTIONS.contains(&direction) {
            errors.push(format!("{label}.direction is invalid"));
        }
        if direction == "target" && finite_number(value, "target_value").is_none() {
            errors.push(format!("{label} with target direction requires finite target_value"));
        }
        if direction != "target" && value.get("target_value").is_some_and(|field| !field.is_null()) {
            errors.push(format!("{label} target_value must be null unless direction is target"));
        }
        if outcome_definitions.insert(id.to_owned(), OutcomeDefinition {
            unit: unit.to_owned(),
            direction: direction.to_owned(),
        }).is_some() {
            errors.push(format!("duplicate outcome_id definition {id:?}"));
        }
    }

    let blocks: &[Value] = match experiment.get("blocks").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("blocks must be a non-empty array".to_owned());
            &[]
        }
    };
    let mut block_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut run_count = 0;
    let mut measurement_count = 0;
    let mut observed_count = 0;
    let mut estimated_count = 0;
    let mut unknown_count = 0;

    for (block_index, block) in blocks.iter().enumerate() {
        let block_label = format!("blocks[{block_index}]");
        let Some(block_id) = non_empty(block, "block_id") else {
            errors.push(format!("{block_label}.block_id is required"));
            continue;
        };
        if !block_ids.insert(block_id.to_owned()) {
            errors.push(format!("duplicate block_id {block_id:?}"));
        }
        if non_empty(block, "randomization_seed").is_none() {
            errors.push(format!("{block_label}.randomization_seed is required"));
        }
        let random_order: Vec<&str> = block
            .get("randomized_order")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if random_order.len() != 2
            || !control.is_some_and(|value| random_order.contains(&value))
            || !variant.is_some_and(|value| random_order.contains(&value))
        {
            errors.push(format!("{block_label}.randomized_order must contain control and variant exactly once"));
        }

        let runs: &[Value] = match block.get("runs").and_then(Value::as_array) {
            Some(items) if items.len() == 2 => items.as_slice(),
            _ => {
                errors.push(format!("{block_label}.runs must contain exactly two paired runs"));
                continue;
            }
        };
        let mut condition_seen = BTreeSet::new();
        let mut order_seen = BTreeSet::new();
        let mut parsed_runs = Vec::<(u64, UtcTimestamp, UtcTimestamp, String)>::new();

        for (run_index, run) in runs.iter().enumerate() {
            run_count += 1;
            let run_label = format!("{block_label}.runs[{run_index}]");
            let Some(run_id) = non_empty(run, "run_id") else {
                errors.push(format!("{run_label}.run_id is required"));
                continue;
            };
            if !run_ids.insert(run_id.to_owned()) {
                errors.push(format!("duplicate run_id {run_id:?}"));
            }
            if non_empty(run, "process_trace_id").is_none() {
                errors.push(format!("{run_label}.process_trace_id is required"));
            }
            let Some(condition) = non_empty(run, "condition_id") else {
                errors.push(format!("{run_label}.condition_id is required"));
                continue;
            };
            if !control.is_some_and(|value| value == condition)
                && !variant.is_some_and(|value| value == condition)
            {
                errors.push(format!("{run_label}.condition_id is neither declared control nor variant"));
            }
            if !condition_seen.insert(condition.to_owned()) {
                errors.push(format!("{block_label} repeats condition {condition:?}"));
            }
            let Some(order) = run.get("execution_order").and_then(Value::as_u64) else {
                errors.push(format!("{run_label}.execution_order must be 0 or 1"));
                continue;
            };
            if order > 1 || !order_seen.insert(order) {
                errors.push(format!("{run_label}.execution_order must be unique and be 0 or 1"));
            }
            if random_order.get(order as usize).copied() != Some(condition) {
                errors.push(format!("{run_label} condition does not match preregistered randomized_order"));
            }

            let start = non_empty(run, "started_at").and_then(parse_utc_timestamp);
            let end = non_empty(run, "ended_at").and_then(parse_utc_timestamp);
            let (Some(start), Some(end)) = (start, end) else {
                errors.push(format!("{run_label} requires valid UTC started_at and ended_at"));
                continue;
            };
            if start.compare(&end) != Ordering::Less {
                errors.push(format!("{run_label}.ended_at must be after started_at"));
            }
            if preregistered_at.is_some_and(|registered| registered.compare(&start) != Ordering::Less) {
                errors.push(format!("{run_label} starts before protocol preregistration"));
            }
            parsed_runs.push((order, start, end, run_id.to_owned()));

            let run_outcomes: &[Value] = match run.get("outcomes").and_then(Value::as_array) {
                Some(items) if !items.is_empty() => items.as_slice(),
                _ => {
                    errors.push(format!("{run_label}.outcomes must be a non-empty array"));
                    continue;
                }
            };
            let mut local_outcome_ids = BTreeSet::new();
            for (outcome_index, outcome) in run_outcomes.iter().enumerate() {
                measurement_count += 1;
                let outcome_label = format!("{run_label}.outcomes[{outcome_index}]");
                let Some(outcome_id) = non_empty(outcome, "outcome_id") else {
                    errors.push(format!("{outcome_label}.outcome_id is required"));
                    continue;
                };
                if !local_outcome_ids.insert(outcome_id.to_owned()) {
                    errors.push(format!("{outcome_label} duplicates outcome_id {outcome_id:?}"));
                    continue;
                }
                let Some(definition) = outcome_definitions.get(outcome_id) else {
                    errors.push(format!("{outcome_label} references undeclared outcome {outcome_id:?}"));
                    continue;
                };
                match validate_outcome_value(
                    outcome,
                    &outcome_label,
                    definition,
                    &mut errors,
                    &mut blockers,
                ) {
                    Some("observed") => observed_count += 1,
                    Some("estimated") => estimated_count += 1,
                    Some("unknown") => unknown_count += 1,
                    _ => {}
                }
                all_qualified &= validate_evidence(
                    outcome.get("evidence").unwrap_or(&Value::Null),
                    &format!("{outcome_label}.evidence"),
                    &mut errors,
                    &mut blockers,
                );
            }
            for outcome_id in outcome_definitions.keys() {
                if !local_outcome_ids.contains(outcome_id) {
                    errors.push(format!("{run_label} is missing declared outcome {outcome_id:?}"));
                }
            }
        }

        if condition_seen.len() != 2
            || !control.is_some_and(|value| condition_seen.contains(value))
            || !variant.is_some_and(|value| condition_seen.contains(value))
        {
            errors.push(format!("{block_label} must contain exactly one control and one variant run"));
        }
        parsed_runs.sort_by_key(|(order, _, _, _)| *order);
        if parsed_runs.len() == 2 && parsed_runs[0].2.compare(&parsed_runs[1].1) == Ordering::Greater {
            errors.push(format!("{block_label} runs overlap or contradict declared execution order"));
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }
    if estimated_count > 0 || unknown_count > 0 {
        blockers.push("paired empirical analysis requires observed values for every declared outcome".to_owned());
    }
    if observed_count != measurement_count {
        blockers.push("not every outcome record is an observed value".to_owned());
    }
    blockers.sort();
    blockers.dedup();
    let paired_analysis_ready = blockers.is_empty() && all_qualified;

    Ok(CulinaryExperimentResultV1 {
        block_count: blocks.len(),
        run_count,
        measurement_count,
        observed_count,
        estimated_count,
        unknown_count,
        all_evidence_records_marked_qualified: all_qualified,
        paired_analysis_ready,
        blockers,
    })
}
