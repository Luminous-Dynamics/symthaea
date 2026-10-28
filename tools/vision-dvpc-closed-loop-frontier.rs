// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-025: oracle-separated equal-budget closed-loop frontier.
//!
//! The policy sees only VISIBLE_ACTIONS. An outcome is joined only after the
//! action has been selected and charged. This is a synthetic mechanism
//! benchmark, not a production authorization API or a SOTA claim.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;

const ACTIONS: &str = include_str!("../tests/fixtures/vision-dvpc-025-visible-actions.tsv");
const ORACLE: &str = include_str!("../tests/fixtures/vision-dvpc-025-oracle.tsv");
const COST_TO_VALUE_MILLI: i64 = 100;
const RISK_ALPHA: f64 = 0.025;
const VALUE_ALPHA: f64 = 0.025;
const RISK_CAP: f64 = 0.05;
const VALUE_FLOOR_MILLI: f64 = 500.0;
const RISK_ALTERNATIVES: [f64; 4] = [0.001, 0.01, 0.025, 0.04];
const VALUE_LAMBDAS: [f64; 2] = [0.5, 1.0];
const PPM: f64 = 1_000_000.0;

const ACTION_HEADER: &[&str] = &[
    "campaign_id", "family", "seed", "action_id", "kind", "test_value_milli",
    "uncertainty_ppm", "predicted_risk_ppm", "estimated_cost_units",
    "budget_units", "target_id", "evidence_unit_kind",
];
const ORACLE_HEADER: &[&str] = &[
    "campaign_id", "action_id", "gross_value_milli", "actual_cost_units",
    "outcome_complete", "risk_trials", "risk_failures", "value_n",
    "value_sum_milli", "value_min_milli", "value_max_milli",
    "evidence_unit_kind", "target_calibrated", "provenance_ok", "sealed_touched",
    "target_truth_risk_ppm", "target_truth_value_mean_milli", "risk_sequence", "value_sequence",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActionKind {
    Measure,
    Activate,
}

impl ActionKind {
    fn parse(text: &str, line: usize) -> Result<Self, String> {
        match text {
            "MEASURE" => Ok(Self::Measure),
            "ACTIVATE" => Ok(Self::Activate),
            _ => Err(format!("actions line {line}: unknown kind {text}")),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Measure => "MEASURE",
            Self::Activate => "ACTIVATE",
        }
    }
}

#[derive(Clone, Debug)]
struct Action {
    campaign_id: String,
    family: String,
    seed: u64,
    action_id: String,
    kind: ActionKind,
    test_value_milli: i64,
    uncertainty_ppm: u64,
    predicted_risk_ppm: u64,
    estimated_cost_units: u64,
    budget_units: u64,
    target_id: String,
    evidence_unit_kind: String,
}

#[derive(Clone, Debug)]
struct Outcome {
    gross_value_milli: i64,
    actual_cost_units: u64,
    outcome_complete: bool,
    risk_trials: usize,
    risk_failures: usize,
    value_n: usize,
    value_sum_milli: i64,
    value_min_milli: i64,
    value_max_milli: i64,
    evidence_unit_kind: String,
    target_calibrated: bool,
    provenance_ok: bool,
    sealed_touched: bool,
    target_truth_risk_ppm: u64,
    target_truth_value_mean_milli: i64,
    risk_sequence: Vec<bool>,
    value_sequence: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GateStatus {
    Authorized,
    Denied,
    InsufficientEvidence,
}

impl GateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Authorized => "AUTHORIZED",
            Self::Denied => "DENIED",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
        }
    }
}

#[derive(Clone, Debug)]
struct GateReport {
    status: GateStatus,
    reason: &'static str,
    risk_upper: Option<f64>,
    value_lcb_milli: Option<f64>,
    risk_n: usize,
    risk_failures: usize,
    value_n: usize,
    unit_kind: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Policy {
    FixedFallback,
    ForcedAdapt,
    HighestUncertainty,
    CostFirst,
    DecisionValue,
    RiskAwareValue,
    DecisionValueAnytime,
    RiskAwareValueAnytime,
    NaivePeeking,
}

impl Policy {
    const ALL: [Self; 9] = [
        Self::FixedFallback,
        Self::ForcedAdapt,
        Self::HighestUncertainty,
        Self::CostFirst,
        Self::DecisionValue,
        Self::RiskAwareValue,
        Self::DecisionValueAnytime,
        Self::RiskAwareValueAnytime,
        Self::NaivePeeking,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::FixedFallback => "fixed_fallback",
            Self::ForcedAdapt => "forced_adapt",
            Self::HighestUncertainty => "highest_uncertainty",
            Self::CostFirst => "cost_first",
            Self::DecisionValue => "decision_value_fixed",
            Self::RiskAwareValue => "risk_aware_value_fixed",
            Self::DecisionValueAnytime => "decision_value_anytime",
            Self::RiskAwareValueAnytime => "risk_aware_value_anytime",
            Self::NaivePeeking => "naive_fixed_peeking_unsafe",
        }
    }
}

#[derive(Clone, Debug)]
struct RunResult {
    campaign_id: String,
    family: String,
    policy: Policy,
    selected_ids: Vec<String>,
    activated: bool,
    false_authorization: bool,
    false_safe_authorization: bool,
    budget_overrun: bool,
    actual_cost_units: u64,
    gross_value_milli: i64,
    net_value_milli: i64,
    gate: GateReport,
}

#[derive(Default)]
struct Summary {
    campaigns: usize,
    activations: usize,
    false_authorizations: usize,
    false_safe_authorizations: usize,
    budget_overruns: usize,
    harmful_activations: usize,
    actual_cost_units: u64,
    net_value_milli: i64,
}

fn fields<'a>(line: &'a str) -> Vec<&'a str> {
    line.split('\t').collect()
}

fn parse_u64(value: &str, field: &str, line: usize) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| format!("line {line}: invalid {field}"))
}

fn parse_i64(value: &str, field: &str, line: usize) -> Result<i64, String> {
    value.parse::<i64>().map_err(|_| format!("line {line}: invalid {field}"))
}

fn parse_usize(value: &str, field: &str, line: usize) -> Result<usize, String> {
    value.parse::<usize>().map_err(|_| format!("line {line}: invalid {field}"))
}

fn parse_bool(value: &str, field: &str, line: usize) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("line {line}: invalid boolean {field}")),
    }
}

fn parse_actions(text: &str) -> Result<Vec<Action>, String> {
    let mut actions = Vec::new();
    let mut seen = BTreeSet::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        if idx == 0 {
            if fields(line).iter().copied().ne(ACTION_HEADER.iter().copied()) {
                return Err("visible-action header mismatch".into());
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let f = fields(line);
        if f.len() != ACTION_HEADER.len() {
            return Err(format!("actions line {line_no}: expected {} columns", ACTION_HEADER.len()));
        }
        for (value, name) in [(f[0], "campaign_id"), (f[1], "family"), (f[3], "action_id"), (f[10], "target_id")] {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')) {
                return Err(format!("actions line {line_no}: invalid {name} token"));
            }
        }
        let action = Action {
            campaign_id: f[0].to_owned(),
            family: f[1].to_owned(),
            seed: parse_u64(f[2], "seed", line_no)?,
            action_id: f[3].to_owned(),
            kind: ActionKind::parse(f[4], line_no)?,
            test_value_milli: parse_i64(f[5], "test_value_milli", line_no)?,
            uncertainty_ppm: parse_u64(f[6], "uncertainty_ppm", line_no)?,
            predicted_risk_ppm: parse_u64(f[7], "predicted_risk_ppm", line_no)?,
            estimated_cost_units: parse_u64(f[8], "estimated_cost_units", line_no)?,
            budget_units: parse_u64(f[9], "budget_units", line_no)?,
            target_id: f[10].to_owned(),
            evidence_unit_kind: f[11].to_owned(),
        };
        if action.estimated_cost_units == 0 || action.budget_units == 0 {
            return Err(format!("actions line {line_no}: costs and budgets must be positive"));
        }
        if action.uncertainty_ppm > 1_000_000 || action.predicted_risk_ppm > 1_000_000 {
            return Err(format!("actions line {line_no}: ppm value exceeds one million"));
        }
        if !matches!(action.evidence_unit_kind.as_str(), "IID" | "INDEPENDENT_CLUSTER" | "ROW_DEPENDENT" | "FIXED_PEEKED" | "NONE") {
            return Err(format!("actions line {line_no}: unknown evidence unit kind"));
        }
        if action.kind == ActionKind::Activate && action.evidence_unit_kind != "NONE" {
            return Err(format!("actions line {line_no}: activation rows cannot carry measurement units"));
        }
        if !seen.insert((action.campaign_id.clone(), action.action_id.clone())) {
            return Err(format!("actions line {line_no}: duplicate action id within campaign"));
        }
        actions.push(action);
    }
    if actions.is_empty() {
        return Err("visible-actions fixture is empty".into());
    }
    validate_campaign_metadata(&actions)?;
    Ok(actions)
}

fn validate_campaign_metadata(actions: &[Action]) -> Result<(), String> {
    let mut metadata: BTreeMap<&str, (&str, u64, u64, &str)> = BTreeMap::new();
    let mut activations = BTreeMap::<&str, usize>::new();
    for a in actions {
        let current = (a.family.as_str(), a.seed, a.budget_units, a.target_id.as_str());
        if let Some(prior) = metadata.get(a.campaign_id.as_str()) {
            if *prior != current {
                return Err(format!("{}: campaign metadata differs across actions", a.campaign_id));
            }
        } else {
            metadata.insert(a.campaign_id.as_str(), current);
        }
        if a.kind == ActionKind::Activate {
            *activations.entry(a.campaign_id.as_str()).or_default() += 1;
        }
    }
    for campaign in metadata.keys() {
        if activations.get(campaign).copied().unwrap_or(0) != 1 {
            return Err(format!("{campaign}: expected exactly one activation action"));
        }
    }
    Ok(())
}

fn parse_oracle(text: &str, actions: &[Action]) -> Result<BTreeMap<(String, String), Outcome>, String> {
    let mut result = BTreeMap::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        if idx == 0 {
            if fields(line).iter().copied().ne(ORACLE_HEADER.iter().copied()) {
                return Err("oracle header mismatch".into());
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let f = fields(line);
        if f.len() != ORACLE_HEADER.len() {
            return Err(format!("oracle line {line_no}: expected {} columns", ORACLE_HEADER.len()));
        }
        let target_truth_risk_ppm = parse_u64(f[15], "target_truth_risk_ppm", line_no)?;
        let target_truth_value_mean_milli = parse_i64(f[16], "target_truth_value_mean_milli", line_no)?;
        let risk_sequence: Vec<bool> = f[17].chars().map(|c| match c {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => Err(format!("oracle line {line_no}: risk sequence must contain only 0/1")),
        }).collect::<Result<_, _>>()?;
        let value_sequence: Vec<i64> = if f[18].is_empty() {
            Vec::new()
        } else {
            f[18].split(',').map(|value| parse_i64(value, "value_sequence", line_no))
                .collect::<Result<_, _>>()?
        };
        let o = Outcome {
            gross_value_milli: parse_i64(f[2], "gross_value_milli", line_no)?,
            actual_cost_units: parse_u64(f[3], "actual_cost_units", line_no)?,
            outcome_complete: parse_bool(f[4], "outcome_complete", line_no)?,
            risk_trials: parse_usize(f[5], "risk_trials", line_no)?,
            risk_failures: parse_usize(f[6], "risk_failures", line_no)?,
            value_n: parse_usize(f[7], "value_n", line_no)?,
            value_sum_milli: parse_i64(f[8], "value_sum_milli", line_no)?,
            value_min_milli: parse_i64(f[9], "value_min_milli", line_no)?,
            value_max_milli: parse_i64(f[10], "value_max_milli", line_no)?,
            evidence_unit_kind: f[11].to_owned(),
            target_calibrated: parse_bool(f[12], "target_calibrated", line_no)?,
            provenance_ok: parse_bool(f[13], "provenance_ok", line_no)?,
            sealed_touched: parse_bool(f[14], "sealed_touched", line_no)?,
            target_truth_risk_ppm,
            target_truth_value_mean_milli,
            risk_sequence,
            value_sequence,
        };
        if o.actual_cost_units == 0 {
            return Err(format!("oracle line {line_no}: actual cost must be positive"));
        }
        if o.target_truth_risk_ppm > 1_000_000 || !(0..=1_000).contains(&o.target_truth_value_mean_milli) {
            return Err(format!("oracle line {line_no}: hidden target truth is outside declared bounds"));
        }
        if o.risk_failures > o.risk_trials {
            return Err(format!("oracle line {line_no}: failures exceed risk trials"));
        }
        if o.value_min_milli > o.value_max_milli {
            return Err(format!("oracle line {line_no}: value bounds are reversed"));
        }
        if o.value_n == 0 {
            if o.value_sum_milli != 0 {
                return Err(format!("oracle line {line_no}: empty value sample has nonzero sum"));
            }
        } else {
            let lower = (o.value_min_milli as i128) * (o.value_n as i128);
            let upper = (o.value_max_milli as i128) * (o.value_n as i128);
            if (o.value_sum_milli as i128) < lower || (o.value_sum_milli as i128) > upper {
                return Err(format!("oracle line {line_no}: value sum violates sample bounds"));
            }
        }
        if o.risk_sequence.len() != o.risk_trials {
            return Err(format!("oracle line {line_no}: risk sequence length differs from risk_trials"));
        }
        if o.risk_sequence.iter().filter(|failed| **failed).count() != o.risk_failures {
            return Err(format!("oracle line {line_no}: risk sequence failures differ from risk_failures"));
        }
        if o.value_sequence.len() != o.value_n {
            return Err(format!("oracle line {line_no}: value sequence length differs from value_n"));
        }
        if o.value_sequence.iter().any(|v| *v < 0 || *v > 1_000 || *v < o.value_min_milli || *v > o.value_max_milli) {
            return Err(format!("oracle line {line_no}: value sequence item outside declared bounded support"));
        }
        if o.value_sequence.iter().sum::<i64>() != o.value_sum_milli {
            return Err(format!("oracle line {line_no}: value sequence sum differs from value_sum_milli"));
        }
        if !o.outcome_complete && (o.risk_trials != 0 || o.value_n != 0 || !o.risk_sequence.is_empty() || !o.value_sequence.is_empty()) {
            return Err(format!("oracle line {line_no}: incomplete outcomes must not expose partial observations"));
        }
        if o.outcome_complete && ((o.risk_trials == 0) != (o.value_n == 0)) {
            return Err(format!("oracle line {line_no}: complete evidence must provide both risk and value observations"));
        }
        if !matches!(o.evidence_unit_kind.as_str(), "IID" | "INDEPENDENT_CLUSTER" | "ROW_DEPENDENT" | "FIXED_PEEKED" | "NONE") {
            return Err(format!("oracle line {line_no}: unknown evidence unit kind"));
        }
        if !result.insert((f[0].to_owned(), f[1].to_owned()), o).is_none() {
            return Err(format!("oracle line {line_no}: duplicate campaign/action key"));
        }
    }
    let expected: BTreeSet<(String, String)> = actions.iter().map(|a| (a.campaign_id.clone(), a.action_id.clone())).collect();
    let actual: BTreeSet<(String, String)> = result.keys().cloned().collect();
    if expected != actual {
        return Err(format!(
            "action/oracle key mismatch; missing outcomes={}, orphan outcomes={}",
            expected.difference(&actual).count(),
            actual.difference(&expected).count()
        ));
    }
    let mut truth_by_campaign = BTreeMap::<String, (u64, i64)>::new();
    for a in actions {
        let o = &result[&(a.campaign_id.clone(), a.action_id.clone())];
        if o.evidence_unit_kind != a.evidence_unit_kind {
            return Err(format!("{}:{} evidence-unit contract differs from outcome", a.campaign_id, a.action_id));
        }
        let truth = (o.target_truth_risk_ppm, o.target_truth_value_mean_milli);
        if let Some(prior) = truth_by_campaign.get(&a.campaign_id) {
            if *prior != truth {
                return Err(format!("{}: hidden target truth differs across campaign outcomes", a.campaign_id));
            }
        } else {
            truth_by_campaign.insert(a.campaign_id.clone(), truth);
        }
        if a.kind == ActionKind::Activate && (o.risk_trials != 0 || o.value_n != 0) {
            return Err(format!("{}: activation outcome must not be reused as calibration evidence", a.campaign_id));
        }
    }
    Ok(result)
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n + 1];
    for i in 1..=n {
        out[i] = out[i - 1] + (i as f64).ln();
    }
    out
}

fn binomial_cdf(n: usize, k: usize, p: f64, log_fact: &[f64]) -> f64 {
    if k >= n || p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let logs: Vec<f64> = (0..=k)
        .map(|i| {
            log_fact[n] - log_fact[i] - log_fact[n - i]
                + (i as f64) * p.ln()
                + ((n - i) as f64) * (-p).ln_1p()
        })
        .collect();
    let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum: f64 = logs.iter().map(|x| (x - max_log).exp()).sum();
    (max_log.exp() * scaled_sum).clamp(0.0, 1.0)
}

/// One-sided Clopper-Pearson upper limit for an independent Bernoulli risk rate.
fn exact_binomial_upper(n: usize, failures: usize, alpha: f64) -> Result<f64, String> {
    if n == 0 || failures > n || !(0.0 < alpha && alpha < 1.0) {
        return Err("require n > 0, failures <= n, and 0 < alpha < 1".into());
    }
    if failures == n {
        return Ok(1.0);
    }
    if failures == 0 {
        return Ok(1.0 - alpha.powf(1.0 / n as f64));
    }
    let log_fact = log_factorials(n);
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..80 {
        let mid = (low + high) / 2.0;
        if binomial_cdf(n, failures, mid, &log_fact) > alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

fn bounded_mean_lcb(n: usize, sum_milli: i64, min_milli: i64, max_milli: i64, alpha: f64) -> Result<f64, String> {
    if n == 0 || min_milli > max_milli || !(0.0 < alpha && alpha < 1.0) {
        return Err("invalid bounded-mean inputs".into());
    }
    let mean = sum_milli as f64 / n as f64;
    let range = (max_milli - min_milli) as f64;
    Ok(mean - range * ((-alpha.ln()) / (2.0 * n as f64)).sqrt())
}

fn evaluate_gate(selected: &[Action], oracle: &BTreeMap<(String, String), Outcome>) -> GateReport {
    evaluate_gate_internal(selected, oracle, false)
}

fn evaluate_naive_peeking_gate(selected: &[Action], oracle: &BTreeMap<(String, String), Outcome>) -> GateReport {
    // Explicitly invalid comparator: permits repeated fixed-sample looks and
    // treats a stream declared FIXED_PEEKED as if it were an IID sample.
    evaluate_gate_internal(selected, oracle, true)
}

fn evaluate_gate_internal(
    selected: &[Action],
    oracle: &BTreeMap<(String, String), Outcome>,
    allow_fixed_peeking: bool,
) -> GateReport {
    let mut risk_n = 0usize;
    let mut risk_failures = 0usize;
    let mut value_n = 0usize;
    let mut value_sum = 0i64;
    let mut value_min = i64::MAX;
    let mut value_max = i64::MIN;
    let mut unit_kind: Option<String> = None;

    for action in selected.iter().filter(|a| a.kind == ActionKind::Measure) {
        let outcome = &oracle[&(action.campaign_id.clone(), action.action_id.clone())];
        if !outcome.outcome_complete {
            continue;
        }
        if !outcome.provenance_ok || outcome.sealed_touched {
            return GateReport {
                status: GateStatus::Denied,
                reason: "provenance_failure_or_sealed_access",
                risk_upper: None, value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
            };
        }
        if !outcome.target_calibrated {
            return GateReport {
                status: GateStatus::Denied,
                reason: "source_to_target_calibration_failure",
                risk_upper: None, value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
            };
        }
        let admissible_unit = matches!(outcome.evidence_unit_kind.as_str(), "IID" | "INDEPENDENT_CLUSTER")
            || (allow_fixed_peeking && outcome.evidence_unit_kind == "FIXED_PEEKED");
        if !admissible_unit {
            // Only the deliberately invalid naive baseline bypasses this unit control.
            continue;
        }
        if let Some(prior) = &unit_kind {
            if prior != &outcome.evidence_unit_kind {
                return GateReport {
                    status: GateStatus::InsufficientEvidence,
                    reason: "mixed_independent_unit_definitions",
                    risk_upper: None, value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
                };
            }
        } else {
            unit_kind = Some(outcome.evidence_unit_kind.clone());
        }
        risk_n += outcome.risk_trials;
        risk_failures += outcome.risk_failures;
        value_n += outcome.value_n;
        value_sum += outcome.value_sum_milli;
        if outcome.value_n > 0 {
            value_min = value_min.min(outcome.value_min_milli);
            value_max = value_max.max(outcome.value_max_milli);
        }
    }

    if risk_n == 0 || value_n == 0 || value_min == i64::MAX || unit_kind.is_none() {
        return GateReport {
            status: GateStatus::InsufficientEvidence,
            reason: "no_admissible_complete_independent_evidence",
            risk_upper: None, value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
        };
    }
    let risk_upper = match exact_binomial_upper(risk_n, risk_failures, RISK_ALPHA) {
        Ok(value) => value,
        Err(_) => return GateReport {
            status: GateStatus::InsufficientEvidence,
            reason: "risk_bound_calculation_failed",
            risk_upper: None, value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
        },
    };
    let value_lcb = match bounded_mean_lcb(value_n, value_sum, value_min, value_max, VALUE_ALPHA) {
        Ok(value) => value,
        Err(_) => return GateReport {
            status: GateStatus::InsufficientEvidence,
            reason: "value_bound_calculation_failed",
            risk_upper: Some(risk_upper), value_lcb_milli: None, risk_n, risk_failures, value_n, unit_kind,
        },
    };
    if risk_upper > RISK_CAP {
        return GateReport {
            status: GateStatus::InsufficientEvidence,
            reason: "risk_upper_bound_above_cap",
            risk_upper: Some(risk_upper), value_lcb_milli: Some(value_lcb), risk_n, risk_failures, value_n, unit_kind,
        };
    }
    if value_lcb < VALUE_FLOOR_MILLI {
        return GateReport {
            status: GateStatus::Denied,
            reason: "value_lower_bound_below_floor",
            risk_upper: Some(risk_upper), value_lcb_milli: Some(value_lcb), risk_n, risk_failures, value_n, unit_kind,
        };
    }
    GateReport {
        status: GateStatus::Authorized,
        reason: "risk_and_value_bounds_satisfied",
        risk_upper: Some(risk_upper), value_lcb_milli: Some(value_lcb), risk_n, risk_failures, value_n, unit_kind,
    }
}

fn log_mean_exp(log_values: &[f64]) -> f64 {
    let max_log = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max_log + log_values.iter().map(|v| (v - max_log).exp()).sum::<f64>().ln()
        - (log_values.len() as f64).ln()
}

fn log_risk_e_value(n: usize, failures: usize) -> f64 {
    if n == 0 || failures > n {
        return f64::NEG_INFINITY;
    }
    let logs: Vec<f64> = RISK_ALTERNATIVES.iter().map(|q| {
        (failures as f64) * (q / RISK_CAP).ln()
            + ((n - failures) as f64) * ((1.0 - q) / (1.0 - RISK_CAP)).ln()
    }).collect();
    log_mean_exp(&logs)
}

fn log_value_e_value(values_milli: &[i64]) -> Result<f64, String> {
    if values_milli.is_empty() || values_milli.iter().any(|v| !(0..=1_000).contains(v)) {
        return Err("anytime-value observations must be a nonempty stream in [0, 1]".into());
    }
    let logs: Vec<f64> = VALUE_LAMBDAS.iter().map(|lambda| {
        values_milli.iter().map(|v| {
            let x = *v as f64 / 1_000.0;
            (1.0 + lambda * (x - VALUE_FLOOR_MILLI / 1_000.0)).ln()
        }).sum::<f64>()
    }).collect();
    Ok(log_mean_exp(&logs))
}

fn evaluate_anytime_gate(
    selected: &[Action],
    oracle: &BTreeMap<(String, String), Outcome>,
) -> GateReport {
    let mut risk_stream = Vec::<bool>::new();
    let mut value_stream = Vec::<i64>::new();
    let mut unit_kind: Option<String> = None;
    let mut risk_crossed = false;
    let mut value_crossed = false;
    let threshold = (1.0 / RISK_ALPHA).ln();

    for action in selected.iter().filter(|a| a.kind == ActionKind::Measure) {
        let outcome = &oracle[&(action.campaign_id.clone(), action.action_id.clone())];
        if !outcome.outcome_complete {
            continue;
        }
        if !outcome.provenance_ok || outcome.sealed_touched {
            return GateReport {
                status: GateStatus::Denied, reason: "provenance_failure_or_sealed_access",
                risk_upper: None, value_lcb_milli: None,
                risk_n: risk_stream.len(), risk_failures: risk_stream.iter().filter(|x| **x).count(),
                value_n: value_stream.len(), unit_kind,
            };
        }
        if !outcome.target_calibrated {
            return GateReport {
                status: GateStatus::Denied, reason: "source_to_target_calibration_failure",
                risk_upper: None, value_lcb_milli: None,
                risk_n: risk_stream.len(), risk_failures: risk_stream.iter().filter(|x| **x).count(),
                value_n: value_stream.len(), unit_kind,
            };
        }
        if !matches!(outcome.evidence_unit_kind.as_str(), "IID" | "INDEPENDENT_CLUSTER") {
            continue;
        }
        if let Some(prior) = &unit_kind {
            if prior != &outcome.evidence_unit_kind {
                return GateReport {
                    status: GateStatus::InsufficientEvidence, reason: "mixed_independent_unit_definitions",
                    risk_upper: None, value_lcb_milli: None,
                    risk_n: risk_stream.len(), risk_failures: risk_stream.iter().filter(|x| **x).count(),
                    value_n: value_stream.len(), unit_kind,
                };
            }
        } else {
            unit_kind = Some(outcome.evidence_unit_kind.clone());
        }
        risk_stream.extend(outcome.risk_sequence.iter().copied());
        value_stream.extend(outcome.value_sequence.iter().copied());

        // Each acquisition returns a batch. Check the e-process at the actual
        // observation boundary (after the batch arrives), not at fictional
        // unobserved intermediate stopping times.
        let failures = risk_stream.iter().filter(|failed| **failed).count();
        if log_risk_e_value(risk_stream.len(), failures) >= threshold {
            risk_crossed = true;
        }
        if log_value_e_value(&value_stream).is_ok_and(|log_e| log_e >= threshold) {
            value_crossed = true;
        }
    }

    let diagnostic = evaluate_gate(selected, oracle);
    let risk_n = risk_stream.len();
    let risk_failures = risk_stream.iter().filter(|failed| **failed).count();
    let value_n = value_stream.len();
    if risk_n == 0 || value_n == 0 || unit_kind.is_none() {
        return GateReport {
            status: GateStatus::InsufficientEvidence, reason: "no_admissible_complete_independent_evidence",
            risk_upper: diagnostic.risk_upper, value_lcb_milli: diagnostic.value_lcb_milli,
            risk_n, risk_failures, value_n, unit_kind,
        };
    }
    if !risk_crossed || !value_crossed {
        return GateReport {
            status: GateStatus::InsufficientEvidence,
            reason: if !risk_crossed { "anytime_risk_e_value_not_crossed" } else { "anytime_value_e_value_not_crossed" },
            risk_upper: diagnostic.risk_upper, value_lcb_milli: diagnostic.value_lcb_milli,
            risk_n, risk_failures, value_n, unit_kind,
        };
    }
    GateReport {
        status: GateStatus::Authorized, reason: "anytime_risk_and_value_e_values_crossed",
        risk_upper: diagnostic.risk_upper, value_lcb_milli: diagnostic.value_lcb_milli,
        risk_n, risk_failures, value_n, unit_kind,
    }
}

fn evaluate_policy_gate(
    policy: Policy,
    selected: &[Action],
    oracle: &BTreeMap<(String, String), Outcome>,
) -> GateReport {
    match policy {
        Policy::DecisionValueAnytime | Policy::RiskAwareValueAnytime => evaluate_anytime_gate(selected, oracle),
        Policy::NaivePeeking => evaluate_naive_peeking_gate(selected, oracle),
        _ => evaluate_gate(selected, oracle),
    }
}

fn selection_score(action: &Action, risk_aware: bool) -> i64 {
    let cost = (action.estimated_cost_units as i64).saturating_mul(COST_TO_VALUE_MILLI);
    let risk_penalty = if risk_aware {
        ((action.predicted_risk_ppm as i64).saturating_mul(2_000) / PPM as i64).max(0)
    } else {
        0
    };
    action.test_value_milli.saturating_sub(cost).saturating_sub(risk_penalty)
}

fn choose_action<'a>(policy: Policy, pool: Vec<&'a Action>) -> Option<&'a Action> {
    match policy {
        Policy::FixedFallback | Policy::ForcedAdapt => None,
        Policy::HighestUncertainty | Policy::NaivePeeking => pool.into_iter().max_by(|a, b| {
            a.uncertainty_ppm.cmp(&b.uncertainty_ppm).then_with(|| b.action_id.cmp(&a.action_id))
        }),
        Policy::CostFirst => pool.into_iter().min_by(|a, b| {
            a.estimated_cost_units.cmp(&b.estimated_cost_units).then_with(|| a.action_id.cmp(&b.action_id))
        }),
        Policy::DecisionValue | Policy::DecisionValueAnytime => pool.into_iter().max_by(|a, b| {
            selection_score(a, false).cmp(&selection_score(b, false)).then_with(|| b.action_id.cmp(&a.action_id))
        }),
        Policy::RiskAwareValue | Policy::RiskAwareValueAnytime => pool.into_iter().filter(|a| a.predicted_risk_ppm as f64 / PPM <= RISK_CAP)
            .max_by(|a, b| {
                selection_score(a, true).cmp(&selection_score(b, true))
                    .then_with(|| b.action_id.cmp(&a.action_id))
            }),
    }
}

fn simulate(
    campaign_id: &str,
    policy: Policy,
    candidates: &[Action],
    oracle: &BTreeMap<(String, String), Outcome>,
) -> RunResult {
    let first = candidates.first().expect("validated campaign is non-empty");
    let budget = first.budget_units;
    let mut selected = Vec::<Action>::new();
    let mut selected_ids = BTreeSet::<String>::new();
    let mut selected_labels = Vec::<String>::new();
    let mut actual_cost_units = 0u64;
    let mut gross_value_milli = 0i64;
    let mut activated = false;
    let mut false_authorization = false;
    let mut false_safe_authorization = false;
    let mut budget_overrun = false;

    if policy == Policy::ForcedAdapt {
        if let Some(action) = candidates.iter().find(|a| a.kind == ActionKind::Activate) {
            if action.estimated_cost_units <= budget {
                let outcome = &oracle[&(action.campaign_id.clone(), action.action_id.clone())];
                selected.push(action.clone());
                selected_ids.insert(action.action_id.clone());
                selected_labels.push(action.action_id.clone());
                actual_cost_units = outcome.actual_cost_units;
                gross_value_milli = outcome.gross_value_milli;
                activated = true;
                let gate = evaluate_policy_gate(policy, &selected, oracle);
                false_authorization = gate.status != GateStatus::Authorized;
                false_safe_authorization = gate.status == GateStatus::Authorized
                    && (outcome.target_truth_risk_ppm as f64 / PPM >= RISK_CAP
                        || outcome.target_truth_value_mean_milli as f64 <= VALUE_FLOOR_MILLI);
                budget_overrun = actual_cost_units > budget;
            }
        }
    } else if policy != Policy::FixedFallback {
        for _ in 0..candidates.len() {
            if budget_overrun {
                break;
            }
            let gate = evaluate_policy_gate(policy, &selected, oracle);
            if gate.status == GateStatus::Denied {
                break;
            }
            let desired_kind = if gate.status == GateStatus::Authorized {
                ActionKind::Activate
            } else {
                ActionKind::Measure
            };
            let pool: Vec<&Action> = candidates.iter()
                .filter(|a| a.kind == desired_kind)
                .filter(|a| !selected_ids.contains(&a.action_id))
                .filter(|a| actual_cost_units.saturating_add(a.estimated_cost_units) <= budget)
                .collect();
            let chosen = choose_action(policy, pool);
            let Some(action) = chosen else { break };
            if matches!(policy, Policy::DecisionValue | Policy::RiskAwareValue | Policy::DecisionValueAnytime | Policy::RiskAwareValueAnytime) {
                let risk_aware = matches!(policy, Policy::RiskAwareValue | Policy::RiskAwareValueAnytime);
                let score = selection_score(action, risk_aware);
                if score <= 0 {
                    break;
                }
            }
            let action = action.clone();
            let outcome = &oracle[&(action.campaign_id.clone(), action.action_id.clone())];
            selected_ids.insert(action.action_id.clone());
            selected_labels.push(action.action_id.clone());
            actual_cost_units = actual_cost_units.saturating_add(outcome.actual_cost_units);
            selected.push(action.clone());
            if action.kind == ActionKind::Activate {
                activated = true;
                gross_value_milli = outcome.gross_value_milli;
                let gate_after = evaluate_policy_gate(policy, &selected, oracle);
                false_authorization = gate_after.status != GateStatus::Authorized;
                false_safe_authorization = gate_after.status == GateStatus::Authorized
                    && (outcome.target_truth_risk_ppm as f64 / PPM >= RISK_CAP
                        || outcome.target_truth_value_mean_milli as f64 <= VALUE_FLOOR_MILLI);
                budget_overrun = actual_cost_units > budget;
                break;
            }
            if actual_cost_units > budget {
                budget_overrun = true;
                break;
            }
        }
    }

    let gate = evaluate_policy_gate(policy, &selected, oracle);
    let net_value_milli = gross_value_milli.saturating_sub((actual_cost_units as i64).saturating_mul(COST_TO_VALUE_MILLI));
    RunResult {
        campaign_id: campaign_id.to_owned(),
        family: first.family.clone(),
        policy,
        selected_ids: selected_labels,
        activated,
        false_authorization,
        false_safe_authorization,
        budget_overrun,
        actual_cost_units,
        gross_value_milli,
        net_value_milli,
        gate,
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|x| format!("{x:.6}")).unwrap_or_else(|| "NA".to_owned())
}

fn score_fixture() -> Result<String, String> {
    let actions = parse_actions(ACTIONS)?;
    let oracle = parse_oracle(ORACLE, &actions)?;
    let mut by_campaign: BTreeMap<String, Vec<Action>> = BTreeMap::new();
    for action in actions {
        by_campaign.entry(action.campaign_id.clone()).or_default().push(action);
    }
    for candidates in by_campaign.values_mut() {
        candidates.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    }
    let mut summaries: BTreeMap<Policy, Summary> = Policy::ALL.into_iter().map(|p| (p, Summary::default())).collect();
    let mut output = String::from(
        "RESULT\tcampaign_id\tfamily\tpolicy\tselected_actions\tactivation\tgate_status\tuncertified_activation\tfalse_safe_authorization\tbudget_overrun\tactual_cost_units\tgross_value_milli\tnet_value_milli\trisk_upper_bound\tvalue_lcb_milli\trisk_trials\trisk_failures\tvalue_n\tgate_reason\n"
    );
    for (campaign_id, candidates) in &by_campaign {
        for policy in Policy::ALL {
            let r = simulate(campaign_id, policy, candidates, &oracle);
            let _ = writeln!(
                output,
                "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.campaign_id, r.family, r.policy.label(), r.selected_ids.join(","),
                r.activated, r.gate.status.as_str(), r.false_authorization, r.false_safe_authorization, r.budget_overrun,
                r.actual_cost_units, r.gross_value_milli, r.net_value_milli,
                fmt_opt(r.gate.risk_upper), fmt_opt(r.gate.value_lcb_milli),
                r.gate.risk_n, r.gate.risk_failures, r.gate.value_n, r.gate.reason
            );
            let s = summaries.get_mut(&policy).expect("policy summary initialized");
            s.campaigns += 1;
            s.activations += if r.activated { 1 } else { 0 };
            s.false_authorizations += if r.false_authorization { 1 } else { 0 };
            s.false_safe_authorizations += if r.false_safe_authorization { 1 } else { 0 };
            s.budget_overruns += if r.budget_overrun { 1 } else { 0 };
            s.harmful_activations += if r.activated && r.gross_value_milli < 0 { 1 } else { 0 };
            s.actual_cost_units = s.actual_cost_units.saturating_add(r.actual_cost_units);
            s.net_value_milli = s.net_value_milli.saturating_add(r.net_value_milli);
        }
    }
    for policy in Policy::ALL {
        let s = summaries.get(&policy).expect("summary exists");
        let _ = writeln!(
            output,
            "SUMMARY\t{}\tcampaigns={}\tactivations={}\tuncertified_activations={}\tfalse_safe_authorizations={}\tbudget_overruns={}\tharmful_activations={}\tactual_cost_units={}\tnet_value_milli={}",
            policy.label(), s.campaigns, s.activations, s.false_authorizations, s.false_safe_authorizations, s.budget_overruns,
            s.harmful_activations, s.actual_cost_units, s.net_value_milli
        );
    }
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() == 2 && args[1] == "--score" {
        return score_fixture();
    }
    Err("usage: vision-dvpc-closed-loop-frontier --score".into())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-025 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<Action>, BTreeMap<(String, String), Outcome>) {
        let actions = parse_actions(ACTIONS).unwrap();
        let oracle = parse_oracle(ORACLE, &actions).unwrap();
        (actions, oracle)
    }

    fn selected_measure(campaign_id: &str, action_id: &str, actions: &[Action]) -> Vec<Action> {
        vec![actions.iter().find(|a| a.campaign_id == campaign_id && a.action_id == action_id).unwrap().clone()]
    }

    #[test]
    fn fixture_keys_and_campaign_contracts_are_exact() {
        let (actions, oracle) = fixture();
        assert_eq!(actions.len(), 52);
        assert_eq!(oracle.len(), 52);
        let campaigns: BTreeSet<&str> = actions.iter().map(|a| a.campaign_id.as_str()).collect();
        assert_eq!(campaigns.len(), 13);
    }

    #[test]
    fn naive_peeking_witness_false_certifies_the_null_boundary() {
        let (actions, oracle) = fixture();
        let candidates: Vec<Action> = actions.iter().filter(|a| a.campaign_id == "C13-repeated-look").cloned().collect();
        let safe_control = simulate("C13-repeated-look", Policy::HighestUncertainty, &candidates, &oracle);
        let naive = simulate("C13-repeated-look", Policy::NaivePeeking, &candidates, &oracle);
        assert!(!safe_control.activated);
        assert!(naive.activated, "{naive:?}");
        assert_eq!(naive.gate.status, GateStatus::Authorized);
        assert!(!naive.false_authorization);
        assert!(naive.false_safe_authorization, "{naive:?}");
    }

    #[test]
    fn anytime_risk_process_respects_the_declared_batch_boundary() {
        let threshold = (1.0 / RISK_ALPHA).ln();
        assert!(log_risk_e_value(72, 0) < threshold);
        assert!(log_risk_e_value(192, 0) > threshold);
        assert!(log_value_e_value(&vec![1_000; 72]).unwrap() > threshold);
    }

    #[test]
    fn anytime_policy_collects_more_evidence_before_activating() {
        let (actions, oracle) = fixture();
        let candidates: Vec<Action> = actions.iter().filter(|a| a.campaign_id == "C01-positive").cloned().collect();
        let fixed = simulate("C01-positive", Policy::DecisionValue, &candidates, &oracle);
        let anytime = simulate("C01-positive", Policy::DecisionValueAnytime, &candidates, &oracle);
        assert!(fixed.activated);
        assert!(anytime.activated, "{anytime:?}");
        assert!(!anytime.false_authorization);
        assert!(anytime.selected_ids.contains(&"m-high".to_owned()));
        assert!(anytime.selected_ids.contains(&"m-good".to_owned()));
        assert!(anytime.actual_cost_units > fixed.actual_cost_units);
    }

    #[test]
    fn iid_zero_failure_evidence_can_authorize_at_the_72_trial_boundary() {
        let (actions, oracle) = fixture();
        let selected = selected_measure("C01-positive", "m-high", &actions);
        let gate = evaluate_gate(&selected, &oracle);
        assert_eq!(gate.status, GateStatus::Authorized, "{gate:?}");
        assert!(gate.risk_upper.unwrap() < RISK_CAP);
        assert!(gate.value_lcb_milli.unwrap() >= VALUE_FLOOR_MILLI);
    }

    #[test]
    fn source_to_target_mismatch_fails_closed() {
        let (actions, oracle) = fixture();
        let selected = selected_measure("C03-ranking-reversal", "m-high", &actions);
        assert_eq!(evaluate_gate(&selected, &oracle).status, GateStatus::Denied);
    }

    #[test]
    fn provenance_and_sealed_access_controls_deny_activation() {
        let (actions, mut oracle) = fixture();
        let selected = selected_measure("C01-positive", "m-high", &actions);
        let key = ("C01-positive".to_owned(), "m-high".to_owned());
        oracle.get_mut(&key).unwrap().provenance_ok = false;
        assert_eq!(evaluate_gate(&selected, &oracle).status, GateStatus::Denied);

        let (actions, mut oracle) = fixture();
        let selected = selected_measure("C01-positive", "m-high", &actions);
        oracle.get_mut(&key).unwrap().sealed_touched = true;
        assert_eq!(evaluate_gate(&selected, &oracle).status, GateStatus::Denied);
    }

    #[test]
    fn row_counts_and_repeated_fixed_sample_peeking_do_not_masquerade_as_admissible_units() {
        let (actions, oracle) = fixture();
        let clustered = selected_measure("C06-clustered", "m-high", &actions);
        let repeated = selected_measure("C13-repeated-look", "m-high", &actions);
        assert_eq!(evaluate_gate(&clustered, &oracle).status, GateStatus::InsufficientEvidence);
        assert_eq!(evaluate_gate(&repeated, &oracle).status, GateStatus::InsufficientEvidence);
    }

    #[test]
    fn independent_cluster_units_can_authorize_when_their_bounds_pass() {
        let (actions, oracle) = fixture();
        let selected = selected_measure("C11-independent-clusters", "m-high", &actions);
        let gate = evaluate_gate(&selected, &oracle);
        assert_eq!(gate.status, GateStatus::Authorized, "{gate:?}");
        assert_eq!(gate.unit_kind.as_deref(), Some("INDEPENDENT_CLUSTER"));
    }

    #[test]
    fn forced_adaptation_is_reported_as_false_authorization_without_evidence() {
        let (actions, oracle) = fixture();
        let candidates: Vec<Action> = actions.iter().filter(|a| a.campaign_id == "C05-target-shift").cloned().collect();
        let result = simulate("C05-target-shift", Policy::ForcedAdapt, &candidates, &oracle);
        assert!(result.activated);
        assert!(result.false_authorization);
        assert_eq!(result.gate.status, GateStatus::InsufficientEvidence);
    }

    #[test]
    fn decision_value_policy_finds_a_positive_opportunity_without_false_authorization() {
        let (actions, oracle) = fixture();
        let candidates: Vec<Action> = actions.iter().filter(|a| a.campaign_id == "C01-positive").cloned().collect();
        let result = simulate("C01-positive", Policy::DecisionValue, &candidates, &oracle);
        assert!(result.activated);
        assert!(!result.false_authorization);
        assert!(result.net_value_milli > 0);
    }

    #[test]
    fn actual_cost_overrun_is_not_erased_by_the_planned_budget() {
        let (actions, oracle) = fixture();
        let candidates: Vec<Action> = actions.iter().filter(|a| a.campaign_id == "C09-overrun").cloned().collect();
        let result = simulate("C09-overrun", Policy::HighestUncertainty, &candidates, &oracle);
        assert!(result.budget_overrun);
        assert!(!result.activated);
        assert!(result.actual_cost_units > 8);
    }

    #[test]
    fn empty_and_malformed_inputs_are_rejected() {
        assert!(parse_actions("").is_err());
        let malformed = format!("{}\n{}\n", ACTION_HEADER.join("\t"), "C\tf\t1\ta\tMEASURE");
        assert!(parse_actions(&malformed).is_err());
    }

    #[test]
    fn complete_sweep_is_deterministic_and_keeps_vector_metrics() {
        let a = score_fixture().unwrap();
        let b = score_fixture().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.matches("RESULT\t").count(), 13 * 9);
        assert_eq!(a.matches("SUMMARY\t").count(), 9);
        assert!(a.contains("\tfalse_authorizations="));
        assert!(a.contains("\tbudget_overruns="));
        assert!(a.contains("\tnet_value_milli="));
    }
}
