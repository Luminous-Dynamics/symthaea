// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-006: post-commit paired-trace evaluator.
//!
//! The policy's ACTION file must be frozen before this program reads the
//! evaluator-private outcome file. This tool reports realized outcomes; it is
//! not an online allocator and does not prove that a dataset license is valid.
//!
//! Run:
//!   rustc --edition=2024 --test tools/vision-dvpc-trace-scorer.rs -o /tmp/vis-dvpc-006-tests
//!   /tmp/vis-dvpc-006-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-trace-scorer.rs -o /tmp/vis-dvpc-006
//!   /tmp/vis-dvpc-006 test fixtures/vision-dvpc-scoring-observations.tsv fixtures/vision-dvpc-scoring-evaluations.tsv fixtures/vision-dvpc-scoring-actions.tsv

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const FORMAT_VERSION: &str = "VIS-DVPC-TRACE-001-v1";
const META_KEYS: &[&str] = &[
    "format",
    "dataset_id",
    "dataset_revision",
    "dataset_snapshot_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "utility_spec_sha256",
    "license_id",
    "license_scope",
    "higher_is_better",
];
const OBS_HEADER: &[&str] = &[
    "OBS_HEADER", "trace_id", "dataset_id", "dataset_revision", "split", "group_id",
    "step_index", "sample_sha256", "cheap_output_sha256", "pre_features_sha256",
    "utility_spec_sha256", "declared_extra_latency_us", "declared_extra_energy_uj",
    "estimated_dvpc", "reported_uncertainty", "shift_signal_sha256",
];
const EVAL_HEADER: &[&str] = &[
    "EVAL_HEADER", "trace_id", "sample_sha256", "cheap_utility_01", "full_utility_01",
    "cheap_safety_violations", "full_safety_violations", "actual_extra_latency_us",
    "actual_extra_energy_uj", "shift_label", "shift_mechanism",
];
const ACTION_HEADER: &[&str] = &[
    "ACTION_HEADER", "trace_id", "sample_sha256", "comparison_id", "policy_id", "choice",
    "commitment_index", "latency_budget_us", "energy_budget_uj", "declared_extra_latency_us",
    "declared_extra_energy_uj", "allocator_latency_us", "allocator_energy_uj", "committed",
];

#[derive(Debug, Clone)]
struct Table {
    meta: BTreeMap<String, String>,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
struct Observation {
    sample_sha256: String,
    group_id: String,
    declared_latency_us: u64,
    declared_energy_uj: u64,
}

#[derive(Debug, Clone)]
struct Outcome {
    sample_sha256: String,
    cheap_utility: f64,
    full_utility: f64,
    cheap_safety: u64,
    full_safety: u64,
    actual_latency_us: u64,
    actual_energy_uj: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    Cheap,
    Full,
}

impl Choice {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "cheap" => Ok(Self::Cheap),
            "full" => Ok(Self::Full),
            _ => Err(format!("unknown choice {raw}; expected cheap or full")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cheap => "cheap",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone)]
struct Action {
    trace_id: String,
    sample_sha256: String,
    comparison_id: String,
    policy_id: String,
    choice: Choice,
    commitment_index: u64,
    latency_budget_us: u64,
    energy_budget_uj: u64,
    declared_latency_us: u64,
    declared_energy_uj: u64,
    allocator_latency_us: u64,
    allocator_energy_uj: u64,
}

#[derive(Debug, Clone, Default)]
struct PolicyMetrics {
    records: usize,
    groups: usize,
    chosen_utility_mean: f64,
    group_balanced_utility: f64,
    beneficial_escalations: usize,
    harmful_escalations: usize,
    neutral_escalations: usize,
    missed_beneficial_opportunities: usize,
    cheap_safety_violations: u64,
    chosen_safety_violations: u64,
    cvar95_incremental_loss: f64,
    total_incremental_latency_us: u64,
    total_incremental_energy_uj: u64,
    allocator_latency_us: u64,
    allocator_energy_uj: u64,
    reservation_overruns: usize,
    latency_budget_us: u64,
    energy_budget_uj: u64,
    latency_budget_violation: bool,
    energy_budget_violation: bool,
    valid_budget_run: bool,
    utility_delta_vs_all_cheap: Option<f64>,
    cvar95_delta_vs_all_cheap: Option<f64>,
    safety_delta_vs_all_cheap: Option<i64>,
}

fn fail(message: impl Into<String>) -> String {
    message.into()
}

fn parse_table(input: &str, expected_header: &[&str], record_tag: &str) -> Result<Table, String> {
    let mut meta = BTreeMap::new();
    let mut rows = Vec::new();
    let mut header: Option<Vec<String>> = None;

    for (idx, line) in input.lines().enumerate() {
        let line_no = idx + 1;
        if line.is_empty() {
            return Err(fail(format!("line {line_no}: blank lines are forbidden")));
        }
        let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
        if fields.first().map(String::as_str) == Some("META") {
            if header.is_some() {
                return Err(fail(format!("line {line_no}: META must precede header")));
            }
            if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                return Err(fail(format!("line {line_no}: invalid META row")));
            }
            if meta.insert(fields[1].clone(), fields[2].clone()).is_some() {
                return Err(fail(format!("line {line_no}: duplicate META key {}", fields[1])));
            }
            continue;
        }
        if header.is_none() {
            if fields.iter().map(String::as_str).collect::<Vec<_>>().as_slice() != expected_header {
                return Err(fail(format!(
                    "line {line_no}: exact schema mismatch; unexpected/reordered columns fail closed"
                )));
            }
            header = Some(fields);
            continue;
        }
        if fields.first().map(String::as_str) != Some(record_tag) {
            return Err(fail(format!("line {line_no}: expected {record_tag} row")));
        }
        if fields.len() != expected_header.len() || fields.iter().any(String::is_empty) {
            return Err(fail(format!("line {line_no}: wrong field count or empty field")));
        }
        rows.push(fields);
    }

    let header = header.ok_or_else(|| fail("missing header"))?;
    if rows.is_empty() {
        return Err(fail("at least one data row is required"));
    }
    validate_meta(&meta)?;
    Ok(Table { meta, header, rows })
}

fn validate_meta(meta: &BTreeMap<String, String>) -> Result<(), String> {
    let expected: BTreeSet<&str> = META_KEYS.iter().copied().collect();
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(fail(format!(
            "metadata key set mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION)
        || meta.get("higher_is_better").map(String::as_str) != Some("true")
    {
        return Err(fail("unsupported format or utility direction"));
    }
    for key in [
        "dataset_snapshot_sha256",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "utility_spec_sha256",
    ] {
        validate_sha(meta.get(key).unwrap(), key)?;
    }
    for key in ["dataset_id", "dataset_revision", "license_id"] {
        validate_token(meta.get(key).unwrap(), key)?;
    }
    if !matches!(
        meta.get("license_scope").map(String::as_str),
        Some("synthetic_public_fixture" | "noncommercial_research_only" | "commercial_use_authorized")
    ) {
        return Err(fail("license scope is not explicitly admissible"));
    }
    Ok(())
}

fn validate_token(raw: &str, field: &str) -> Result<(), String> {
    if raw.is_empty()
        || !raw.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
    {
        return Err(fail(format!("invalid {field} token")));
    }
    Ok(())
}

fn validate_sha(raw: &str, field: &str) -> Result<(), String> {
    if raw.len() != 64 || !raw.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fail(format!("{field} must be a 64-character SHA-256 digest")));
    }
    Ok(())
}

fn parse_u64(raw: &str, field: &str) -> Result<u64, String> {
    raw.parse::<u64>().map_err(|_| fail(format!("{field} must be a non-negative integer")))
}

fn parse_f64(raw: &str, field: &str) -> Result<f64, String> {
    let value = raw.parse::<f64>().map_err(|_| fail(format!("{field} must be numeric")))?;
    if !value.is_finite() {
        return Err(fail(format!("{field} must be finite")));
    }
    Ok(value)
}

fn ensure_same_meta(tables: &[&Table]) -> Result<(), String> {
    let first = tables.first().ok_or_else(|| fail("no tables supplied"))?;
    if tables.iter().skip(1).any(|table| table.meta != first.meta) {
        return Err(fail("dataset, split, utility or licensing metadata differ between inputs"));
    }
    Ok(())
}

fn read_observations(table: &Table, target_split: &str) -> Result<BTreeMap<String, Observation>, String> {
    let mut output = BTreeMap::new();
    let dataset_id = table.meta.get("dataset_id").unwrap();
    let revision = table.meta.get("dataset_revision").unwrap();
    let utility = table.meta.get("utility_spec_sha256").unwrap();
    let mut group_steps: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    for (i, row) in table.rows.iter().enumerate() {
        validate_token(&row[1], "trace_id")?;
        validate_token(&row[2], "dataset_id")?;
        validate_token(&row[3], "dataset_revision")?;
        validate_token(&row[5], "group_id")?;
        if &row[2] != dataset_id || &row[3] != revision {
            return Err(fail(format!("observation row {}: dataset identity mismatch", i + 1)));
        }
        if row[4] != target_split {
            return Err(fail(format!(
                "trace {} has split {}; scorer invocation requires only split {}",
                row[1], row[4], target_split
            )));
        }
        for position in [7, 8, 9, 10] {
            validate_sha(&row[position], table.header[position])?;
        }
        if &row[10] != utility {
            return Err(fail(format!("trace {} utility-spec digest mismatch", row[1])));
        }
        if row[15] != "-" {
            validate_sha(&row[15], "shift_signal_sha256")?;
        }
        let declared_latency_us = parse_u64(&row[11], "declared_extra_latency_us")?;
        let declared_energy_uj = parse_u64(&row[12], "declared_extra_energy_uj")?;
        if declared_latency_us == 0 || declared_energy_uj == 0 {
            return Err(fail("declared incremental latency and energy must be positive"));
        }
        let estimate = parse_f64(&row[13], "estimated_dvpc")?;
        let uncertainty = parse_f64(&row[14], "reported_uncertainty")?;
        let _ = estimate;
        if uncertainty < 0.0 {
            return Err(fail("reported_uncertainty must be non-negative"));
        }
        let key = format!("{}:{}:{}", row[2], row[3], row[5]);
        let step = parse_u64(&row[6], "step_index")?;
        if !group_steps.entry(key).or_default().insert(step) {
            return Err(fail(format!("duplicate step index in group {}", row[5])));
        }
        let observation = Observation {
            sample_sha256: row[7].clone(),
            group_id: row[5].clone(),
            declared_latency_us,
            declared_energy_uj,
        };
        if output.insert(row[1].clone(), observation).is_some() {
            return Err(fail(format!("duplicate observation trace_id {}", row[1])));
        }
    }
    for (group, steps) in group_steps {
        for (expected, observed) in steps.iter().enumerate() {
            if *observed != expected as u64 {
                return Err(fail(format!("non-contiguous step index in {group}")));
            }
        }
    }
    Ok(output)
}

fn read_outcomes(table: &Table, observations: &BTreeMap<String, Observation>) -> Result<BTreeMap<String, Outcome>, String> {
    let mut output = BTreeMap::new();
    for row in &table.rows {
        validate_token(&row[1], "trace_id")?;
        validate_sha(&row[2], "sample_sha256")?;
        let obs = observations.get(&row[1]).ok_or_else(|| fail(format!("unmatched evaluation trace {}", row[1])))?;
        if obs.sample_sha256 != row[2] {
            return Err(fail(format!("sample digest mismatch for {}", row[1])));
        }
        let cheap_utility = parse_f64(&row[3], "cheap_utility_01")?;
        let full_utility = parse_f64(&row[4], "full_utility_01")?;
        if !(0.0..=1.0).contains(&cheap_utility) || !(0.0..=1.0).contains(&full_utility) {
            return Err(fail("utility must be finite and in [0,1]"));
        }
        let cheap_safety = parse_u64(&row[5], "cheap_safety_violations")?;
        let full_safety = parse_u64(&row[6], "full_safety_violations")?;
        let actual_latency_us = parse_u64(&row[7], "actual_extra_latency_us")?;
        let actual_energy_uj = parse_u64(&row[8], "actual_extra_energy_uj")?;
        if actual_latency_us == 0 || actual_energy_uj == 0 {
            return Err(fail("realized incremental latency and energy must be positive"));
        }
        if !matches!(row[9].as_str(), "in_domain" | "shifted" | "unknown") {
            return Err(fail("invalid shift label"));
        }
        if !matches!(row[10].as_str(), "none" | "illumination" | "weather" | "occlusion" | "sensor" | "density" | "other" | "unknown") {
            return Err(fail("invalid shift mechanism"));
        }
        if (row[9] == "in_domain" && row[10] != "none")
            || (row[9] == "shifted" && matches!(row[10].as_str(), "none" | "unknown"))
        {
            return Err(fail("shift label and mechanism are inconsistent"));
        }
        let outcome = Outcome {
            sample_sha256: row[2].clone(),
            cheap_utility,
            full_utility,
            cheap_safety,
            full_safety,
            actual_latency_us,
            actual_energy_uj,
        };
        if output.insert(row[1].clone(), outcome).is_some() {
            return Err(fail(format!("duplicate evaluation trace {}", row[1])));
        }
    }
    if output.len() != observations.len() {
        return Err(fail("evaluation rows do not join one-to-one with observations"));
    }
    Ok(output)
}

fn read_actions(table: &Table, observations: &BTreeMap<String, Observation>) -> Result<Vec<Action>, String> {
    let mut actions = Vec::new();
    let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
    let mut per_run_indexes: BTreeMap<(String, String), BTreeSet<u64>> = BTreeMap::new();
    let mut per_run_budgets: BTreeMap<(String, String), (u64, u64)> = BTreeMap::new();
    let mut comparison_budgets: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut per_run_traces: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();

    for row in &table.rows {
        validate_token(&row[1], "trace_id")?;
        validate_sha(&row[2], "sample_sha256")?;
        validate_token(&row[3], "comparison_id")?;
        validate_token(&row[4], "policy_id")?;
        let trace_id = row[1].clone();
        let obs = observations.get(&trace_id).ok_or_else(|| fail(format!("action for unknown trace {trace_id}")))?;
        if obs.sample_sha256 != row[2] {
            return Err(fail(format!("action sample digest mismatch for {trace_id}")));
        }
        let choice = Choice::parse(&row[5])?;
        let commitment_index = parse_u64(&row[6], "commitment_index")?;
        let latency_budget_us = parse_u64(&row[7], "latency_budget_us")?;
        let energy_budget_uj = parse_u64(&row[8], "energy_budget_uj")?;
        let declared_latency_us = parse_u64(&row[9], "declared_extra_latency_us")?;
        let declared_energy_uj = parse_u64(&row[10], "declared_extra_energy_uj")?;
        let allocator_latency_us = parse_u64(&row[11], "allocator_latency_us")?;
        let allocator_energy_uj = parse_u64(&row[12], "allocator_energy_uj")?;
        if latency_budget_us == 0 || energy_budget_uj == 0 {
            return Err(fail("run budgets must be positive"));
        }
        if declared_latency_us != obs.declared_latency_us || declared_energy_uj != obs.declared_energy_uj {
            return Err(fail(format!("action reservation does not match observation for {trace_id}")));
        }
        if row[13] != "true" {
            return Err(fail(format!("action for {trace_id} is not marked committed")));
        }

        let run_key = (row[3].clone(), row[4].clone());
        if !seen.insert((row[3].clone(), row[4].clone(), trace_id.clone())) {
            return Err(fail(format!("duplicate policy/trace action {} / {}", row[4], trace_id)));
        }
        let budget = (latency_budget_us, energy_budget_uj);
        if let Some(previous) = per_run_budgets.insert(run_key.clone(), budget) {
            if previous != budget {
                return Err(fail(format!("inconsistent budget within run {}", row[4])));
            }
        }
        if let Some(previous) = comparison_budgets.insert(row[3].clone(), budget) {
            if previous != budget {
                return Err(fail(format!("policies in comparison {} do not share budgets", row[3])));
            }
        }
        if !per_run_indexes
            .entry(run_key.clone())
            .or_default()
            .insert(commitment_index)
        {
            return Err(fail(format!("policy {} has duplicate commitment_index", row[4])));
        }
        per_run_traces.entry(run_key).or_default().insert(trace_id.clone());
        actions.push(Action {
            trace_id,
            sample_sha256: row[2].clone(),
            comparison_id: row[3].clone(),
            policy_id: row[4].clone(),
            choice,
            commitment_index,
            latency_budget_us,
            energy_budget_uj,
            declared_latency_us,
            declared_energy_uj,
            allocator_latency_us,
            allocator_energy_uj,
        });
    }

    let expected_traces: BTreeSet<String> = observations.keys().cloned().collect();
    for (run, traces) in &per_run_traces {
        if traces != &expected_traces {
            return Err(fail(format!("policy {} has missing or extra trace decisions", run.1)));
        }
        let indexes = per_run_indexes.get(run).unwrap();
        for (expected, observed) in indexes.iter().enumerate() {
            if *observed != expected as u64 {
                return Err(fail(format!("policy {} has non-contiguous commitment_index", run.1)));
            }
        }
    }
    Ok(actions)
}

fn cvar95(losses: &[f64]) -> f64 {
    if losses.is_empty() {
        return 0.0;
    }
    let mut values = losses.to_vec();
    values.sort_by(|a, b| b.total_cmp(a));
    let n = ((values.len() + 19) / 20).max(1);
    values.iter().take(n).sum::<f64>() / n as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn add_cost(total: &mut u64, increment: u64) -> bool {
    match total.checked_add(increment) {
        Some(value) => {
            *total = value;
            false
        }
        None => {
            *total = u64::MAX;
            true
        }
    }
}

fn score(
    observations: &BTreeMap<String, Observation>,
    outcomes: &BTreeMap<String, Outcome>,
    actions: &[Action],
) -> Result<String, String> {
    let mut runs: BTreeMap<(String, String), Vec<&Action>> = BTreeMap::new();
    for action in actions {
        if action.sample_sha256 != outcomes.get(&action.trace_id).unwrap().sample_sha256 {
            return Err(fail("action/outcome sample digest mismatch"));
        }
        runs.entry((action.comparison_id.clone(), action.policy_id.clone())).or_default().push(action);
    }

    let mut metrics: BTreeMap<(String, String), PolicyMetrics> = BTreeMap::new();
    let mut per_group_utility: BTreeMap<(String, String, String), Vec<f64>> = BTreeMap::new();
    for (run_key, rows) in &runs {
        let mut m = PolicyMetrics::default();
        let mut chosen_utilities = Vec::new();
        let mut losses = Vec::new();
        let mut latency = 0_u64;
        let mut energy = 0_u64;
        let mut allocator_latency = 0_u64;
        let mut allocator_energy = 0_u64;
        let mut reservation_overruns = 0;
        let mut chosen_safety = 0_u64;
        let mut cheap_safety = 0_u64;
        let mut latency_accumulator_overflow = false;
        let mut energy_accumulator_overflow = false;
        let mut beneficial = 0;
        let mut harmful = 0;
        let mut neutral = 0;
        let mut missed = 0;

        let first = rows.first().ok_or_else(|| fail("empty policy run"))?;
        m.latency_budget_us = first.latency_budget_us;
        m.energy_budget_uj = first.energy_budget_uj;
        for action in rows {
            let obs = observations.get(&action.trace_id).unwrap();
            let outcome = outcomes.get(&action.trace_id).unwrap();
            latency_accumulator_overflow |= add_cost(&mut allocator_latency, action.allocator_latency_us);
            energy_accumulator_overflow |= add_cost(&mut allocator_energy, action.allocator_energy_uj);
            latency_accumulator_overflow |= add_cost(&mut latency, action.allocator_latency_us);
            energy_accumulator_overflow |= add_cost(&mut energy, action.allocator_energy_uj);
            cheap_safety = cheap_safety.saturating_add(outcome.cheap_safety);
            let utility = match action.choice {
                Choice::Cheap => {
                    if outcome.full_utility > outcome.cheap_utility { missed += 1; }
                    chosen_safety = chosen_safety.saturating_add(outcome.cheap_safety);
                    losses.push(0.0);
                    outcome.cheap_utility
                }
                Choice::Full => {
                    latency_accumulator_overflow |= add_cost(&mut latency, outcome.actual_latency_us);
                    energy_accumulator_overflow |= add_cost(&mut energy, outcome.actual_energy_uj);
                    if outcome.actual_latency_us > action.declared_latency_us
                        || outcome.actual_energy_uj > action.declared_energy_uj
                    {
                        reservation_overruns += 1;
                    }
                    chosen_safety = chosen_safety.saturating_add(outcome.full_safety);
                    let delta = outcome.full_utility - outcome.cheap_utility;
                    if delta > 0.0 { beneficial += 1; }
                    else if delta < 0.0 { harmful += 1; }
                    else { neutral += 1; }
                    losses.push(-delta);
                    outcome.full_utility
                }
            };
            chosen_utilities.push(utility);
            per_group_utility.entry((run_key.0.clone(), run_key.1.clone(), obs.group_id.clone())).or_default().push(utility);
        }

        let group_means: Vec<f64> = per_group_utility.iter()
            .filter(|((comparison, policy, _), _)| comparison == &run_key.0 && policy == &run_key.1)
            .map(|(_, values)| mean(values))
            .collect();
        let chosen_utility_mean = mean(&chosen_utilities);
        let group_balanced_utility = mean(&group_means);
        m.records = rows.len();
        m.groups = group_means.len();
        m.chosen_utility_mean = chosen_utility_mean;
        m.group_balanced_utility = group_balanced_utility;
        m.beneficial_escalations = beneficial;
        m.harmful_escalations = harmful;
        m.neutral_escalations = neutral;
        m.missed_beneficial_opportunities = missed;
        m.cheap_safety_violations = cheap_safety;
        m.chosen_safety_violations = chosen_safety;
        m.cvar95_incremental_loss = cvar95(&losses);
        m.total_incremental_latency_us = latency;
        m.total_incremental_energy_uj = energy;
        m.allocator_latency_us = allocator_latency;
        m.allocator_energy_uj = allocator_energy;
        m.reservation_overruns = reservation_overruns;
        m.latency_budget_violation = latency_accumulator_overflow || latency > m.latency_budget_us;
        m.energy_budget_violation = energy_accumulator_overflow || energy > m.energy_budget_uj;
        m.valid_budget_run = !m.latency_budget_violation && !m.energy_budget_violation;
        metrics.insert(run_key.clone(), m);
    }

    // Snapshot baseline metrics first to avoid aliasing the policy-result map.
    let cheap_baselines: BTreeMap<String, PolicyMetrics> = metrics
        .iter()
        .filter(|((_, policy), _)| policy == "all_cheap")
        .map(|((comparison, _), value)| (comparison.clone(), value.clone()))
        .collect();
    for (run_key, m) in metrics.iter_mut() {
        if let Some(cheap) = cheap_baselines.get(&run_key.0) {
            m.utility_delta_vs_all_cheap = Some(m.group_balanced_utility - cheap.group_balanced_utility);
            m.cvar95_delta_vs_all_cheap = Some(m.cvar95_incremental_loss - cheap.cvar95_incremental_loss);
            m.safety_delta_vs_all_cheap = Some(
                m.chosen_safety_violations as i64 - cheap.chosen_safety_violations as i64,
            );
        }
    }

    let mut report = String::new();
    report.push_str("META\tformat\tVIS-DVPC-006-post-commit-v1\n");
    report.push_str("META\tclaim_class\tpost_commit_trace_scoring_not_allocator_quality_or_sota\n");
    report.push_str("META\tunit_rule\tlatency_us_and_energy_uj_reported_separately\n");
    report.push_str("RESULT_HEADER\tcomparison_id\tpolicy_id\trecords\tgroups\tmean_utility\tgroup_balanced_utility\tbeneficial_escalations\tharmful_escalations\tneutral_escalations\tmissed_beneficial_opportunities\tcheap_safety_violations\tchosen_safety_violations\tcvar95_incremental_loss\tallocator_latency_us\tallocator_energy_uj\ttotal_incremental_latency_us\ttotal_incremental_energy_uj\tlatency_budget_us\tenergy_budget_uj\tlatency_budget_violation\tenergy_budget_violation\treservation_overruns\tvalid_budget_run\tutility_delta_vs_all_cheap\tcvar95_delta_vs_all_cheap\tsafety_delta_vs_all_cheap\n");
    for ((comparison, policy), m) in &metrics {
        report.push_str(&format!(
            "RESULT\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            comparison, policy, m.records, m.groups, m.chosen_utility_mean, m.group_balanced_utility,
            m.beneficial_escalations, m.harmful_escalations, m.neutral_escalations,
            m.missed_beneficial_opportunities, m.cheap_safety_violations, m.chosen_safety_violations,
            m.cvar95_incremental_loss, m.allocator_latency_us, m.allocator_energy_uj,
            m.total_incremental_latency_us, m.total_incremental_energy_uj, m.latency_budget_us,
            m.energy_budget_uj, m.latency_budget_violation, m.energy_budget_violation,
            m.reservation_overruns, m.valid_budget_run,
            fmt_opt(m.utility_delta_vs_all_cheap), fmt_opt(m.cvar95_delta_vs_all_cheap),
            fmt_opt_i64(m.safety_delta_vs_all_cheap)
        ));
    }
    report.push_str("META\toracle\tNOT_COMPUTED_IN_V1\n");
    Ok(report)
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|n| format!("{n:.6}")).unwrap_or_else(|| "NA".to_owned())
}
fn fmt_opt_i64(value: Option<i64>) -> String {
    value.map(|n| n.to_string()).unwrap_or_else(|| "NA".to_owned())
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 5 {
        return Err("usage: vision-dvpc-trace-scorer SPLIT OBSERVATIONS.tsv EVALUATIONS.tsv ACTIONS.tsv".to_owned());
    }
    let split = args[1].as_str();
    if !matches!(split, "validation" | "test" | "private_test") {
        return Err("SPLIT must be validation, test, or private_test; train scoring is not admitted".to_owned());
    }
    let obs_text = fs::read_to_string(&args[2]).map_err(|e| fail(format!("read observations: {e}")))?;
    let eval_text = fs::read_to_string(&args[3]).map_err(|e| fail(format!("read evaluations: {e}")))?;
    let action_text = fs::read_to_string(&args[4]).map_err(|e| fail(format!("read actions: {e}")))?;
    let obs_table = parse_table(&obs_text, OBS_HEADER, "OBS")?;
    let eval_table = parse_table(&eval_text, EVAL_HEADER, "EVAL")?;
    let action_table = parse_table(&action_text, ACTION_HEADER, "ACTION")?;
    ensure_same_meta(&[&obs_table, &eval_table, &action_table])?;
    let observations = read_observations(&obs_table, split)?;
    let outcomes = read_outcomes(&eval_table, &observations)?;
    let actions = read_actions(&action_table, &observations)?;
    score(&observations, &outcomes, &actions)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("REJECT\treason\t{}", error.replace('\t', " "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBS: &str = include_str!("../fixtures/vision-dvpc-scoring-observations.tsv");
    const EVAL: &str = include_str!("../fixtures/vision-dvpc-scoring-evaluations.tsv");
    const ACTIONS: &str = include_str!("../fixtures/vision-dvpc-scoring-actions.tsv");

    fn tables() -> (Table, Table, Table) {
        (
            parse_table(OBS, OBS_HEADER, "OBS").unwrap(),
            parse_table(EVAL, EVAL_HEADER, "EVAL").unwrap(),
            parse_table(ACTIONS, ACTION_HEADER, "ACTION").unwrap(),
        )
    }

    #[test]
    fn scores_committed_policies_and_reports_budget_violation_without_hiding_it() {
        let (obs_table, eval_table, action_table) = tables();
        ensure_same_meta(&[&obs_table, &eval_table, &action_table]).unwrap();
        let observations = read_observations(&obs_table, "test").unwrap();
        let outcomes = read_outcomes(&eval_table, &observations).unwrap();
        let actions = read_actions(&action_table, &observations).unwrap();
        let report = score(&observations, &outcomes, &actions).unwrap();
        assert!(report.contains("RESULT\tmatched\tall_cheap"));
        assert!(report.contains("RESULT\tmatched\tall_full"));
        assert!(report.contains("RESULT\tmatched\tvalue_router"));
        assert!(report.contains("META\toracle\tNOT_COMPUTED_IN_V1"));
        assert!(report.contains("true")); // At least one budget violation remains visible.
    }

    #[test]
    fn rejects_duplicate_policy_trace_actions() {
        let row = ACTIONS.lines().last().unwrap();
        let broken = format!("{ACTIONS}{row}\n");
        let (obs, _, _) = tables();
        let observations = read_observations(&obs, "test").unwrap();
        let action_table = parse_table(&broken, ACTION_HEADER, "ACTION").unwrap();
        assert!(read_actions(&action_table, &observations).unwrap_err().contains("duplicate policy/trace"));
    }

    #[test]
    fn rejects_uncommitted_choice() {
        let broken = ACTIONS.replace("\ttrue\n", "\tfalse\n");
        let (obs, _, _) = tables();
        let observations = read_observations(&obs, "test").unwrap();
        let action_table = parse_table(&broken, ACTION_HEADER, "ACTION").unwrap();
        assert!(read_actions(&action_table, &observations).unwrap_err().contains("not marked committed"));
    }

    #[test]
    fn rejects_action_for_unknown_trace() {
        let broken = ACTIONS.replace("trace-test-001", "trace-unknown-001");
        let (obs, _, _) = tables();
        let observations = read_observations(&obs, "test").unwrap();
        let action_table = parse_table(&broken, ACTION_HEADER, "ACTION").unwrap();
        assert!(read_actions(&action_table, &observations).unwrap_err().contains("unknown trace"));
    }

    #[test]
    fn rejects_mismatched_snapshot_metadata() {
        let broken = EVAL.replace("dataset_revision\tfixture-v1", "dataset_revision\tfixture-v2");
        let (obs, _, act) = tables();
        let eval_table = parse_table(&broken, EVAL_HEADER, "EVAL").unwrap();
        assert!(ensure_same_meta(&[&obs, &eval_table, &act]).unwrap_err().contains("metadata differ"));
    }

    #[test]
    fn rejects_unknown_choice() {
        let broken = ACTIONS.replace("\tfull\t", "\tmagic\t");
        let (obs, _, _) = tables();
        let observations = read_observations(&obs, "test").unwrap();
        let action_table = parse_table(&broken, ACTION_HEADER, "ACTION").unwrap();
        assert!(read_actions(&action_table, &observations).unwrap_err().contains("unknown choice"));
    }

    #[test]
    fn cvar95_is_worst_five_percent_incremental_loss() {
        assert_eq!(cvar95(&[0.0, -0.15, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.08);
    }
}
