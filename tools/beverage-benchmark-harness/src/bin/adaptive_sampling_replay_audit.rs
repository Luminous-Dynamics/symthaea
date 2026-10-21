// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired replay / matched-arm audit for adaptive-sampling policies.
//!
//! This evaluates a declared cohort against a declared baseline. It does not
//! prove causal attribution, authenticate source observations, or validate a
//! site's hazard analysis. Synthetic fixtures qualify evaluator semantics only.

use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SCHEMA_VERSION: &str = "beverage-adaptive-sampling-replay-v1";
const BOOTSTRAP_REPLICATES: usize = 5_000;
const CLAIM_CEILING: &str = "This audit compares declared paired sampling costs, decision-loss outcomes and guardrails with deterministic bootstrap intervals. It does not authenticate measurements, establish causal effects, validate counterfactual assumptions, prove statistical power from pair count, qualify food-safety controls, or authorize actuation.";

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum EvidenceClass {
    SyntheticContractFixture,
    RetrospectiveReplay,
    ProspectivePilot,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ComparisonDesign {
    SyntheticFixture,
    RetrospectiveReplay,
    BlockRandomized,
    MatchedControl,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArmOutcome {
    independent_unit_id: String,
    discretionary_measurement_cost_units: f64,
    discretionary_assay_count: u32,
    eligible_decisions: u32,
    decision_loss_units: f64,
    high_severity_events: u32,
    missed_high_severity_events: u32,
    false_alarms: u32,
    hard_constraint_violations: u32,
    required_test_violations: u32,
    decision_deadline_misses: u32,
    quality_failures: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PairedOutcome {
    pair_id: String,
    cohort_id: String,
    policy: ArmOutcome,
    baseline: ArmOutcome,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayManifest {
    schema_version: String,
    evidence_class: EvidenceClass,
    design: ComparisonDesign,
    benchmark_id: String,
    policy_id: String,
    baseline_id: String,
    policy_config_sha256: String,
    baseline_config_sha256: String,
    decision_loss_definition_sha256: String,
    measurement_cost_unit: String,
    decision_loss_unit: String,
    discretionary_budget_cap_units: f64,
    target_cost_reduction_percent: f64,
    max_mean_loss_degradation_units: f64,
    max_false_alarms_per_100_decisions: f64,
    max_deadline_miss_rate_degradation_per_100_decisions: f64,
    min_independent_pairs: usize,
    min_high_severity_events: u32,
    bootstrap_seed: u64,
    bootstrap_replicates: usize,
    pairs: Vec<PairedOutcome>,
}

#[derive(Debug, Serialize)]
struct MetricInterval {
    estimate: f64,
    lower_95: f64,
    upper_95: f64,
}

#[derive(Debug, Serialize)]
struct ReplayMetrics {
    pair_count: usize,
    distinct_independent_unit_count: usize,
    high_severity_events_policy: u64,
    high_severity_events_baseline: u64,
    missed_high_severity_events_policy: u64,
    missed_high_severity_events_baseline: u64,
    false_alarms_policy: u64,
    false_alarms_baseline: u64,
    eligible_decisions_policy: u64,
    eligible_decisions_baseline: u64,
    total_measurement_cost_policy_units: f64,
    total_measurement_cost_baseline_units: f64,
    discretionary_assay_count_policy: u64,
    discretionary_assay_count_baseline: u64,
    measurement_cost_reduction_percent: MetricInterval,
    mean_paired_decision_loss_degradation_units: MetricInterval,
    false_alarms_per_100_decisions_policy: f64,
    false_alarms_per_100_decisions_baseline: f64,
    deadline_misses_per_100_decisions_policy: f64,
    deadline_misses_per_100_decisions_baseline: f64,
    quality_failures_policy: u64,
    quality_failures_baseline: u64,
}

#[derive(Debug, Serialize)]
struct AuditReceipt<'a> {
    schema_version: &'static str,
    benchmark_id: &'a str,
    evidence_class: &'a EvidenceClass,
    design: &'a ComparisonDesign,
    disposition: &'static str,
    valid: bool,
    violations: Vec<String>,
    indeterminate_reasons: Vec<String>,
    metrics: Option<ReplayMetrics>,
    claim_ceiling: &'static str,
}

#[derive(Debug)]
struct AuditResult {
    violations: Vec<String>,
    indeterminate_reasons: Vec<String>,
    metrics: Option<ReplayMetrics>,
    valid: bool,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = fraction.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[derive(Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn bootstrap_intervals(
    manifest: &ReplayManifest,
    pairs: &[PairedOutcome],
) -> Option<(MetricInterval, MetricInterval)> {
    if pairs.is_empty() || manifest.bootstrap_replicates != BOOTSTRAP_REPLICATES {
        return None;
    }

    let total_policy_cost: f64 = pairs
        .iter()
        .map(|pair| pair.policy.discretionary_measurement_cost_units)
        .sum();
    let total_baseline_cost: f64 = pairs
        .iter()
        .map(|pair| pair.baseline.discretionary_measurement_cost_units)
        .sum();
    if !total_policy_cost.is_finite()
        || total_policy_cost < 0.0
        || !total_baseline_cost.is_finite()
        || total_baseline_cost <= 0.0
    {
        return None;
    }
    if pairs.iter().any(|pair| {
        !nonnegative_finite(pair.policy.discretionary_measurement_cost_units)
            || !nonnegative_finite(pair.baseline.discretionary_measurement_cost_units)
            || !nonnegative_finite(pair.policy.decision_loss_units)
            || !nonnegative_finite(pair.baseline.decision_loss_units)
    }) {
        return None;
    }

    let policy_loss_deltas: Vec<f64> = pairs
        .iter()
        .map(|pair| pair.policy.decision_loss_units - pair.baseline.decision_loss_units)
        .collect();
    let saving_estimate = 100.0 * (1.0 - total_policy_cost / total_baseline_cost);
    let loss_estimate =
        policy_loss_deltas.iter().sum::<f64>() / policy_loss_deltas.len() as f64;

    let mut rng = DeterministicRng::new(manifest.bootstrap_seed);
    let mut savings_bootstrap = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut loss_bootstrap = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut policy_cost = 0.0;
        let mut baseline_cost = 0.0;
        let mut loss_delta = 0.0;

        for _ in 0..pairs.len() {
            let index = rng.index(pairs.len());
            let pair = &pairs[index];
            policy_cost += pair.policy.discretionary_measurement_cost_units;
            baseline_cost += pair.baseline.discretionary_measurement_cost_units;
            loss_delta += policy_loss_deltas[index];
        }

        if baseline_cost <= 0.0 || !baseline_cost.is_finite() {
            return None;
        }
        savings_bootstrap.push(100.0 * (1.0 - policy_cost / baseline_cost));
        loss_bootstrap.push(loss_delta / pairs.len() as f64);
    }

    savings_bootstrap.sort_by(f64::total_cmp);
    loss_bootstrap.sort_by(f64::total_cmp);

    Some((
        MetricInterval {
            estimate: saving_estimate,
            lower_95: percentile(&savings_bootstrap, 0.025),
            upper_95: percentile(&savings_bootstrap, 0.975),
        },
        MetricInterval {
            estimate: loss_estimate,
            lower_95: percentile(&loss_bootstrap, 0.025),
            upper_95: percentile(&loss_bootstrap, 0.975),
        },
    ))
}

fn audit_manifest(manifest: &ReplayManifest) -> AuditResult {
    let mut violations = Vec::new();
    let mut indeterminate_reasons = Vec::new();

    if manifest.schema_version != SCHEMA_VERSION {
        violations.push("schema_version_mismatch".to_string());
    }
    for (name, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("policy_id", manifest.policy_id.as_str()),
        ("baseline_id", manifest.baseline_id.as_str()),
        ("measurement_cost_unit", manifest.measurement_cost_unit.as_str()),
        ("decision_loss_unit", manifest.decision_loss_unit.as_str()),
    ] {
        if value.trim().is_empty() {
            violations.push(format!("{name}_blank"));
        }
    }
    if manifest.policy_id == manifest.baseline_id {
        violations.push("policy_and_baseline_ids_must_differ".to_string());
    }
    for (name, digest) in [
        ("policy_config_sha256", manifest.policy_config_sha256.as_str()),
        ("baseline_config_sha256", manifest.baseline_config_sha256.as_str()),
        (
            "decision_loss_definition_sha256",
            manifest.decision_loss_definition_sha256.as_str(),
        ),
    ] {
        if !is_sha256_hex(digest) {
            violations.push(format!("{name}_invalid"));
        }
    }

    if !nonnegative_finite(manifest.discretionary_budget_cap_units)
        || manifest.discretionary_budget_cap_units <= 0.0
    {
        violations.push("discretionary_budget_cap_invalid".to_string());
    }
    if !nonnegative_finite(manifest.target_cost_reduction_percent)
        || manifest.target_cost_reduction_percent > 100.0
    {
        violations.push("target_cost_reduction_percent_invalid".to_string());
    }
    if !nonnegative_finite(manifest.max_mean_loss_degradation_units) {
        violations.push("max_mean_loss_degradation_invalid".to_string());
    }
    if !nonnegative_finite(manifest.max_false_alarms_per_100_decisions) {
        violations.push("max_false_alarm_rate_invalid".to_string());
    }
    if !nonnegative_finite(manifest.max_deadline_miss_rate_degradation_per_100_decisions) {
        violations.push("max_deadline_miss_rate_degradation_invalid".to_string());
    }
    if manifest.min_independent_pairs < 5 {
        violations.push("min_independent_pairs_below_five".to_string());
    }
    if manifest.bootstrap_replicates != BOOTSTRAP_REPLICATES {
        violations.push("bootstrap_replicates_not_pinned_to_5000".to_string());
    }
    if manifest.pairs.len() < manifest.min_independent_pairs {
        indeterminate_reasons.push("independent_pair_count_below_preregistered_minimum".to_string());
    }

    match (manifest.evidence_class, manifest.design) {
        (EvidenceClass::SyntheticContractFixture, ComparisonDesign::SyntheticFixture)
        | (EvidenceClass::RetrospectiveReplay, ComparisonDesign::RetrospectiveReplay)
        | (EvidenceClass::ProspectivePilot, ComparisonDesign::BlockRandomized)
        | (EvidenceClass::ProspectivePilot, ComparisonDesign::MatchedControl) => {}
        _ => violations.push("evidence_class_and_design_mismatch".to_string()),
    }

    let mut pair_ids = BTreeSet::new();
    let mut unit_ids = BTreeSet::new();

    let mut event_count_policy = 0_u64;
    let mut event_count_baseline = 0_u64;
    let mut missed_policy = 0_u64;
    let mut missed_baseline = 0_u64;
    let mut false_policy = 0_u64;
    let mut false_baseline = 0_u64;
    let mut decisions_policy = 0_u64;
    let mut decisions_baseline = 0_u64;
    let mut policy_cost = 0.0_f64;
    let mut baseline_cost = 0.0_f64;
    let mut policy_assays = 0_u64;
    let mut baseline_assays = 0_u64;
    let mut quality_policy = 0_u64;
    let mut quality_baseline = 0_u64;
    let mut deadline_miss_policy = 0_u64;
    let mut deadline_miss_baseline = 0_u64;
    let mut hard_violations_policy = 0_u64;
    let mut hard_violations_baseline = 0_u64;
    let mut required_violations_policy = 0_u64;
    let mut required_violations_baseline = 0_u64;

    for pair in &manifest.pairs {
        if pair.pair_id.trim().is_empty() || !pair_ids.insert(pair.pair_id.as_str()) {
            violations.push("pair_id_blank_or_duplicate".to_string());
        }
        if pair.cohort_id.trim().is_empty() {
            violations.push("cohort_id_blank".to_string());
        }

        for (arm_name, arm) in [("policy", &pair.policy), ("baseline", &pair.baseline)] {
            if arm.independent_unit_id.trim().is_empty() {
                violations.push(format!("{arm_name}_independent_unit_id_blank"));
            }
            if !nonnegative_finite(arm.discretionary_measurement_cost_units) {
                violations.push(format!("{arm_name}_measurement_cost_invalid"));
            } else if arm.discretionary_measurement_cost_units
                > manifest.discretionary_budget_cap_units
            {
                violations.push(format!("{arm_name}_measurement_cost_exceeds_budget"));
            }
            if !nonnegative_finite(arm.decision_loss_units) {
                violations.push(format!("{arm_name}_decision_loss_invalid"));
            }
            if arm.eligible_decisions == 0 {
                violations.push(format!("{arm_name}_eligible_decisions_zero"));
            }
            if arm.missed_high_severity_events > arm.high_severity_events {
                violations.push(format!("{arm_name}_missed_events_exceed_events"));
            }
            if arm.false_alarms > arm.eligible_decisions {
                violations.push(format!("{arm_name}_false_alarms_exceed_eligible_decisions"));
            }
        }

        match manifest.design {
            ComparisonDesign::SyntheticFixture | ComparisonDesign::RetrospectiveReplay => {
                if pair.policy.independent_unit_id != pair.baseline.independent_unit_id {
                    violations.push("replay_pair_must_share_reference_unit".to_string());
                }
                if pair.policy.high_severity_events != pair.baseline.high_severity_events
                    || pair.policy.eligible_decisions != pair.baseline.eligible_decisions
                {
                    violations.push("replay_reference_outcome_mismatch".to_string());
                }
                if !unit_ids.insert(pair.policy.independent_unit_id.as_str()) {
                    violations.push("independent_unit_reused_across_pairs".to_string());
                }
            }
            ComparisonDesign::BlockRandomized | ComparisonDesign::MatchedControl => {
                if pair.policy.independent_unit_id == pair.baseline.independent_unit_id {
                    violations.push("prospective_arms_must_have_distinct_units".to_string());
                }
                if !unit_ids.insert(pair.policy.independent_unit_id.as_str()) {
                    violations.push("independent_unit_reused_across_pairs".to_string());
                }
                if !unit_ids.insert(pair.baseline.independent_unit_id.as_str()) {
                    violations.push("independent_unit_reused_across_pairs".to_string());
                }
            }
        }

        event_count_policy += u64::from(pair.policy.high_severity_events);
        event_count_baseline += u64::from(pair.baseline.high_severity_events);
        missed_policy += u64::from(pair.policy.missed_high_severity_events);
        missed_baseline += u64::from(pair.baseline.missed_high_severity_events);
        false_policy += u64::from(pair.policy.false_alarms);
        false_baseline += u64::from(pair.baseline.false_alarms);
        decisions_policy += u64::from(pair.policy.eligible_decisions);
        decisions_baseline += u64::from(pair.baseline.eligible_decisions);
        policy_cost += pair.policy.discretionary_measurement_cost_units;
        baseline_cost += pair.baseline.discretionary_measurement_cost_units;
        policy_assays += u64::from(pair.policy.discretionary_assay_count);
        baseline_assays += u64::from(pair.baseline.discretionary_assay_count);
        quality_policy += u64::from(pair.policy.quality_failures);
        quality_baseline += u64::from(pair.baseline.quality_failures);
        deadline_miss_policy += u64::from(pair.policy.decision_deadline_misses);
        deadline_miss_baseline += u64::from(pair.baseline.decision_deadline_misses);
        hard_violations_policy += u64::from(pair.policy.hard_constraint_violations);
        hard_violations_baseline += u64::from(pair.baseline.hard_constraint_violations);
        required_violations_policy += u64::from(pair.policy.required_test_violations);
        required_violations_baseline += u64::from(pair.baseline.required_test_violations);
    }

    if event_count_policy < u64::from(manifest.min_high_severity_events)
        || event_count_baseline < u64::from(manifest.min_high_severity_events)
    {
        indeterminate_reasons.push("high_severity_event_count_below_preregistered_minimum".to_string());
    }

    if required_violations_policy > 0 || required_violations_baseline > 0 {
        violations.push("required_test_schedule_violation_present".to_string());
    }
    if hard_violations_policy > 0 || hard_violations_baseline > 0 {
        violations.push("hard_constraint_violation_present".to_string());
    }
    if baseline_cost <= 0.0 || !baseline_cost.is_finite() {
        indeterminate_reasons.push("baseline_measurement_cost_not_positive".to_string());
    }
    if decisions_policy == 0 || decisions_baseline == 0 {
        indeterminate_reasons.push("no_eligible_decisions_for_rate_metrics".to_string());
    }

    let event_rate_policy = if event_count_policy > 0 {
        missed_policy as f64 / event_count_policy as f64
    } else {
        0.0
    };
    let event_rate_baseline = if event_count_baseline > 0 {
        missed_baseline as f64 / event_count_baseline as f64
    } else {
        0.0
    };
    if event_count_policy > 0 && event_count_baseline > 0 && event_rate_policy > event_rate_baseline {
        violations.push("policy_missed_event_rate_worse_than_baseline".to_string());
    }

    let false_rate_policy = if decisions_policy > 0 {
        100.0 * false_policy as f64 / decisions_policy as f64
    } else {
        f64::INFINITY
    };
    let false_rate_baseline = if decisions_baseline > 0 {
        100.0 * false_baseline as f64 / decisions_baseline as f64
    } else {
        f64::INFINITY
    };
    if false_rate_policy > manifest.max_false_alarms_per_100_decisions {
        violations.push("policy_false_alarm_rate_exceeds_preregistered_maximum".to_string());
    }

    let deadline_rate_policy = if decisions_policy > 0 {
        100.0 * deadline_miss_policy as f64 / decisions_policy as f64
    } else {
        f64::INFINITY
    };
    let deadline_rate_baseline = if decisions_baseline > 0 {
        100.0 * deadline_miss_baseline as f64 / decisions_baseline as f64
    } else {
        f64::INFINITY
    };
    if deadline_rate_policy - deadline_rate_baseline
        > manifest.max_deadline_miss_rate_degradation_per_100_decisions
    {
        violations.push("policy_deadline_miss_rate_degraded".to_string());
    }

    if quality_policy > quality_baseline {
        violations.push("policy_quality_failures_worse_than_baseline".to_string());
    }

    if event_count_policy > 0 && event_count_baseline > 0 && missed_policy > missed_baseline {
        violations.push("policy_missed_event_count_worse_than_baseline".to_string());
    }

    if let Some((cost_ci, loss_ci)) = bootstrap_intervals(manifest, &manifest.pairs) {
        if cost_ci.lower_95 < manifest.target_cost_reduction_percent {
            violations.push("cost_reduction_lower_ci_below_target".to_string());
        }
        if loss_ci.upper_95 > manifest.max_mean_loss_degradation_units {
            violations.push("mean_decision_loss_degradation_upper_ci_exceeds_margin".to_string());
        }

        let metrics = ReplayMetrics {
            pair_count: manifest.pairs.len(),
            distinct_independent_unit_count: unit_ids.len(),
            high_severity_events_policy: event_count_policy,
            high_severity_events_baseline: event_count_baseline,
            missed_high_severity_events_policy: missed_policy,
            missed_high_severity_events_baseline: missed_baseline,
            false_alarms_policy: false_policy,
            false_alarms_baseline: false_baseline,
            eligible_decisions_policy: decisions_policy,
            eligible_decisions_baseline: decisions_baseline,
            total_measurement_cost_policy_units: policy_cost,
            total_measurement_cost_baseline_units: baseline_cost,
            discretionary_assay_count_policy: policy_assays,
            discretionary_assay_count_baseline: baseline_assays,
            measurement_cost_reduction_percent: cost_ci,
            mean_paired_decision_loss_degradation_units: loss_ci,
            false_alarms_per_100_decisions_policy: false_rate_policy,
            false_alarms_per_100_decisions_baseline: false_rate_baseline,
            deadline_misses_per_100_decisions_policy: deadline_rate_policy,
            deadline_misses_per_100_decisions_baseline: deadline_rate_baseline,
            quality_failures_policy: quality_policy,
            quality_failures_baseline: quality_baseline,
        };
        let valid = violations.is_empty() && indeterminate_reasons.is_empty();
        AuditResult {
            violations,
            indeterminate_reasons,
            metrics: Some(metrics),
            valid,
        }
    } else {
        indeterminate_reasons.push("paired_bootstrap_not_estimable".to_string());
        AuditResult {
            violations,
            indeterminate_reasons,
            metrics: None,
            valid: false,
        }
    }
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>), String> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--input requires a path".to_string())?,
                ));
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--out requires a path".to_string())?,
                ));
            }
            "--help" | "-h" => {
                return Err(
                    "usage: adaptive_sampling_replay_audit --input <manifest.json> [--out <receipt.json>]"
                        .to_string(),
                );
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok((
        input.ok_or_else(|| {
            "missing --input; use --input <manifest.json> [--out <receipt.json>]".to_string()
        })?,
        output,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, output_path) = parse_args()?;
    let raw = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let manifest: ReplayManifest = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid replay manifest JSON: {error}"))?;
    let result = audit_manifest(&manifest);
    let disposition = if result.valid {
        match manifest.evidence_class {
            EvidenceClass::SyntheticContractFixture => "ACCEPT_SYNTHETIC_CONTRACT_ONLY",
            EvidenceClass::RetrospectiveReplay => "ELIGIBLE_FOR_INDEPENDENT_REVIEW",
            EvidenceClass::ProspectivePilot => "ELIGIBLE_FOR_INDEPENDENT_REVIEW",
        }
    } else if !result.indeterminate_reasons.is_empty() && result.violations.is_empty() {
        "INDETERMINATE"
    } else {
        "REJECT"
    };
    let receipt = AuditReceipt {
        schema_version: SCHEMA_VERSION,
        benchmark_id: &manifest.benchmark_id,
        evidence_class: &manifest.evidence_class,
        design: &manifest.design,
        disposition,
        valid: result.valid,
        violations: result.violations,
        indeterminate_reasons: result.indeterminate_reasons,
        metrics: result.metrics,
        claim_ceiling: CLAIM_CEILING,
    };
    let serialized = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("cannot serialize replay receipt: {error}"))?;
    if let Some(path) = output_path {
        fs::write(&path, format!("{serialized}\n"))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{serialized}");
    }
    Ok(if receipt.valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_arm(unit_id: &str, cost: f64, assays: u32, loss: f64) -> ArmOutcome {
        ArmOutcome {
            independent_unit_id: unit_id.to_string(),
            discretionary_measurement_cost_units: cost,
            discretionary_assay_count: assays,
            eligible_decisions: 10,
            decision_loss_units: loss,
            high_severity_events: 1,
            missed_high_severity_events: 0,
            false_alarms: 0,
            hard_constraint_violations: 0,
            required_test_violations: 0,
            decision_deadline_misses: 0,
            quality_failures: 0,
        }
    }

    fn valid_manifest() -> ReplayManifest {
        let pairs = (0..8)
            .map(|index| {
                let unit_id = format!("trace-{index:03}");
                PairedOutcome {
                    pair_id: format!("pair-{index:03}"),
                    cohort_id: "synthetic-cohort-a".to_string(),
                    policy: valid_arm(&unit_id, 7.0 + (index % 2) as f64, 4, 2.0),
                    baseline: valid_arm(&unit_id, 10.0 + (index % 2) as f64, 6, 2.05),
                }
            })
            .collect();

        ReplayManifest {
            schema_version: SCHEMA_VERSION.to_string(),
            evidence_class: EvidenceClass::SyntheticContractFixture,
            design: ComparisonDesign::SyntheticFixture,
            benchmark_id: "synthetic-adaptive-sampling-replay-v1".to_string(),
            policy_id: "risk-voi-v1".to_string(),
            baseline_id: "fixed-cadence-v1".to_string(),
            policy_config_sha256: "a".repeat(64),
            baseline_config_sha256: "b".repeat(64),
            decision_loss_definition_sha256: "c".repeat(64),
            measurement_cost_unit: "normalized_measurement_cost_unit".to_string(),
            decision_loss_unit: "predeclared_decision_loss_unit".to_string(),
            discretionary_budget_cap_units: 12.0,
            target_cost_reduction_percent: 20.0,
            max_mean_loss_degradation_units: 0.25,
            max_false_alarms_per_100_decisions: 5.0,
            max_deadline_miss_rate_degradation_per_100_decisions: 0.0,
            min_independent_pairs: 5,
            min_high_severity_events: 5,
            bootstrap_seed: 270_121,
            bootstrap_replicates: BOOTSTRAP_REPLICATES,
            pairs,
        }
    }

    #[test]
    fn accepts_valid_synthetic_replay_contract_only() {
        let result = audit_manifest(&valid_manifest());
        assert!(result.valid);
        assert!(result.violations.is_empty());
        assert!(result.indeterminate_reasons.is_empty());
        let metrics = result.metrics.expect("metrics produced");
        assert!(metrics.measurement_cost_reduction_percent.lower_95 >= 20.0);
    }

    #[test]
    fn rejects_evidence_design_mismatch() {
        let mut manifest = valid_manifest();
        manifest.design = ComparisonDesign::BlockRandomized;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "evidence_class_and_design_mismatch")
        );
    }

    #[test]
    fn rejects_measurement_cost_over_budget() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].policy.discretionary_measurement_cost_units = 13.0;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "policy_measurement_cost_exceeds_budget")
        );
    }

    #[test]
    fn rejects_nonpositive_net_cost_baseline() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.baseline.discretionary_measurement_cost_units = 0.0;
        }
        assert!(
            audit_manifest(&manifest)
                .indeterminate_reasons
                .iter()
                .any(|issue| issue == "baseline_measurement_cost_not_positive")
        );
    }

    #[test]
    fn rejects_policy_that_misses_more_high_severity_events() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].policy.missed_high_severity_events = 1;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "policy_missed_event_rate_worse_than_baseline")
        );
    }

    #[test]
    fn rejects_policy_that_omits_a_required_test() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].policy.required_test_violations = 1;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "required_test_schedule_violation_present")
        );
    }

    #[test]
    fn rejects_hard_constraint_violations() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].policy.hard_constraint_violations = 1;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "hard_constraint_violation_present")
        );
    }

    #[test]
    fn rejects_false_alarm_rate_over_preregistered_maximum() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.policy.false_alarms = 2;
        }
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "policy_false_alarm_rate_exceeds_preregistered_maximum")
        );
    }

    #[test]
    fn rejects_policy_with_worse_deadline_miss_rate() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.policy.decision_deadline_misses = 1;
        }
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "policy_deadline_miss_rate_degraded")
        );
    }

    #[test]
    fn rejects_policy_that_does_not_meet_cost_reduction_target() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.policy.discretionary_measurement_cost_units = 9.9;
        }
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "cost_reduction_lower_ci_below_target")
        );
    }

    #[test]
    fn rejects_mean_decision_loss_degradation_past_margin() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.policy.decision_loss_units = 2.5;
        }
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "mean_decision_loss_degradation_upper_ci_exceeds_margin")
        );
    }

    #[test]
    fn reports_indeterminate_when_high_severity_events_are_too_rare() {
        let mut manifest = valid_manifest();
        for pair in &mut manifest.pairs {
            pair.policy.high_severity_events = 0;
            pair.baseline.high_severity_events = 0;
        }
        let result = audit_manifest(&manifest);
        assert!(!result.valid);
        assert!(
            result
                .indeterminate_reasons
                .iter()
                .any(|issue| issue == "high_severity_event_count_below_preregistered_minimum")
        );
    }

    #[test]
    fn rejects_reused_independent_units() {
        let mut manifest = valid_manifest();
        manifest.pairs[1].policy.independent_unit_id = "trace-000".to_string();
        manifest.pairs[1].baseline.independent_unit_id = "trace-000".to_string();
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "independent_unit_reused_across_pairs")
        );
    }

    #[test]
    fn rejects_replay_pair_with_mismatched_reference_event_count() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].baseline.high_severity_events = 2;
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "replay_reference_outcome_mismatch")
        );
    }

    #[test]
    fn rejects_pair_with_different_trajectory_in_replay_design() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].baseline.independent_unit_id = "different-trace".to_string();
        assert!(
            audit_manifest(&manifest)
                .violations
                .iter()
                .any(|issue| issue == "replay_pair_must_share_reference_unit")
        );
    }

    #[test]
    fn bootstrap_is_deterministic_for_same_seed() {
        let manifest = valid_manifest();
        let first = audit_manifest(&manifest).metrics.expect("metrics");
        let second = audit_manifest(&manifest).metrics.expect("metrics");
        assert_eq!(
            first.measurement_cost_reduction_percent.lower_95,
            second.measurement_cost_reduction_percent.lower_95
        );
        assert_eq!(
            first.mean_paired_decision_loss_degradation_units.upper_95,
            second.mean_paired_decision_loss_degradation_units.upper_95
        );
    }

    #[test]
    fn rejects_unknown_fields() {
        let value = serde_json::to_value(valid_manifest()).expect("serializes");
        let mut object = value.as_object().expect("object").clone();
        object.insert("disable_required_test_gate".to_string(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<ReplayManifest>(serde_json::Value::Object(object)).is_err()
        );
    }
}
