//! Deterministic, evidence-grounded plain-language explanations for connectivity reports.
//!
//! This crate is intentionally not a generative model. It maps structured
//! outcomes into reviewed templates, preserves the declared evidence class,
//! refuses missing denominators, and discloses which questions the data cannot
//! answer. Future HDC retrieval may select reviewed explanation patterns; it
//! must not invent evidence or upgrade the status of a claim.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use serde_json::Value;

pub const EXPLAINER_VERSION: &str = "symthaea.connectivity-explainer.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    SyntheticSimulation,
    BenchObservedSnapshot,
    LabMeasurement,
    FieldPilot,
    ReliabilityQualification,
    Unknown,
}

impl EvidenceClass {
    fn from_label(value: Option<&str>) -> Self {
        match value {
            Some("synthetic_simulation") => Self::SyntheticSimulation,
            Some("bench_observed_snapshot") => Self::BenchObservedSnapshot,
            Some("lab_measurement" | "lab_screening") => Self::LabMeasurement,
            Some("field_pilot") => Self::FieldPilot,
            Some("reliability_qualification") => Self::ReliabilityQualification,
            _ => Self::Unknown,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::SyntheticSimulation => "synthetic simulation",
            Self::BenchObservedSnapshot => "observed benchmark snapshot",
            Self::LabMeasurement => "lab measurement",
            Self::FieldPilot => "field pilot",
            Self::ReliabilityQualification => "reliability qualification",
            Self::Unknown => "unknown evidence class",
        }
    }

    fn caveat(self) -> &'static str {
        match self {
            Self::SyntheticSimulation => {
                "This is a model result, not a physical-network measurement."
            }
            Self::BenchObservedSnapshot => {
                "This is a diagnostic snapshot; by itself it does not prove application task success or instrument calibration."
            }
            Self::LabMeasurement => {
                "The record is labelled as lab evidence; this explanation does not independently authenticate its instruments or measurements."
            }
            Self::FieldPilot => {
                "The record is labelled as field-pilot evidence; conclusions apply only to the recorded sites, population, hardware and trial conditions."
            }
            Self::ReliabilityQualification => {
                "The record is labelled as reliability-qualification evidence; inspect its protocol, raw receipts and statistical assumptions before generalizing."
            }
            Self::Unknown => {
                "The evidence class is missing or unrecognized; do not infer measured performance from this report."
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PriorityOutcome {
    pub priority: String,
    pub tasks: u64,
    pub baseline_completed: u64,
    pub candidate_completed: u64,
    pub baseline_deadline_misses: Option<u64>,
    pub candidate_deadline_misses: Option<u64>,
    pub plain_explanation: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModeExplanation {
    pub mode: String,
    pub plain_label: String,
    pub task_count: u64,
    pub baseline_completed: u64,
    pub candidate_completed: u64,
    pub completion_difference: i64,
    pub plain_explanation: String,
    pub by_priority: Vec<PriorityOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskComparison {
    pub task_id: String,
    pub mode: String,
    pub priority: String,
    pub service_bytes: u64,
    pub arrival_window: u64,
    pub deadline_window: u64,
    pub expires_after_window: u64,
    pub baseline_delivered_bytes: u64,
    pub candidate_delivered_bytes: u64,
    pub baseline_outcome: String,
    pub candidate_outcome: String,
    pub baseline_completion_window: Option<u64>,
    pub candidate_completion_window: Option<u64>,
    pub baseline_failure_reason: Option<String>,
    pub candidate_failure_reason: Option<String>,
    pub explanation: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    pub schema_version: String,
    pub evidence_class: EvidenceClass,
    pub evidence_label: String,
    pub headline: String,
    pub simple_explanation: String,
    pub findings: Vec<String>,
    pub modes: Vec<ModeExplanation>,
    pub task_comparisons: Vec<TaskComparison>,
    pub task_comparisons_total: u64,
    pub task_comparisons_truncated: bool,
    pub limitations: Vec<String>,
    pub recommended_next_step: String,
    pub scenario_sha256: Option<String>,
    pub profile_sha256: Option<String>,
}

fn required_u64(value: &Value, key: &str, path: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{path}/{key}: required non-negative integer is missing or invalid"))
}

fn optional_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn valid_sha(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(str::to_owned)
}

fn mode_label(mode: &str) -> &'static str {
    match mode {
        "online" => "normal connectivity",
        "wan_partition" => "Internet outage while the local network remains available",
        "route_recovery" => "Internet recovery and queued-work delivery",
        _ => "declared scenario",
    }
}

fn task_noun(count: u64) -> &'static str {
    if count == 1 { "task" } else { "tasks" }
}

fn completion_delta(candidate: u64, baseline: u64) -> Result<i64, String> {
    i64::try_from(i128::from(candidate) - i128::from(baseline))
        .map_err(|_| "completion difference outside supported range".to_owned())
}

fn parse_priority_rows(row: &Value) -> Result<Option<Vec<PriorityOutcome>>, String> {
    let Some(baseline_rows) = row.get("baseline_by_priority").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(candidate_rows) = row.get("candidate_by_priority").and_then(Value::as_array) else {
        return Ok(None);
    };

    let mut outcomes = Vec::new();
    for baseline in baseline_rows {
        let priority = baseline
            .get("priority")
            .and_then(Value::as_str)
            .ok_or_else(|| "baseline priority row is missing its label".to_owned())?;
        let candidate = candidate_rows
            .iter()
            .find(|item| item.get("priority").and_then(Value::as_str) == Some(priority))
            .ok_or_else(|| format!("candidate priority row is missing {priority}"))?;
        let tasks = required_u64(baseline, "task_count", "baseline_by_priority")?;
        let candidate_tasks = required_u64(candidate, "task_count", "candidate_by_priority")?;
        if tasks != candidate_tasks {
            return Err(format!(
                "priority {priority} has mismatched trial denominators"
            ));
        }
        let baseline_completed = required_u64(baseline, "completed_tasks", "baseline_by_priority")?;
        let candidate_completed =
            required_u64(candidate, "completed_tasks", "candidate_by_priority")?;
        if baseline_completed > tasks || candidate_completed > tasks {
            return Err(format!(
                "priority {priority} completion exceeds its denominator"
            ));
        }
        let baseline_misses = optional_u64(baseline, "deadline_misses");
        let candidate_misses = optional_u64(candidate, "deadline_misses");
        let delta = completion_delta(candidate_completed, baseline_completed)?;
        let plain_explanation = if delta > 0 {
            format!(
                "Fair-share completes {} more {priority} {} than FIFO in this scenario ({candidate_completed}/{tasks} versus {baseline_completed}/{tasks}).",
                delta,
                task_noun(delta as u64)
            )
        } else if delta < 0 {
            format!(
                "Fair-share completes {} fewer {priority} {} than FIFO in this scenario; treat this as a regression.",
                delta.unsigned_abs(),
                task_noun(delta.unsigned_abs())
            )
        } else {
            format!(
                "Both policies complete {baseline_completed}/{tasks} {priority} tasks in this scenario."
            )
        };
        outcomes.push(PriorityOutcome {
            priority: priority.to_owned(),
            tasks,
            baseline_completed,
            candidate_completed,
            baseline_deadline_misses: baseline_misses,
            candidate_deadline_misses: candidate_misses,
            plain_explanation,
        });
    }
    outcomes.sort_by(|a, b| a.priority.cmp(&b.priority));
    Ok(Some(outcomes))
}

fn optional_task_u64(value: &Value, key: &str, task_id: &str) -> Result<Option<u64>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(item) => item
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("task_results/{task_id}/{key}: expected a non-negative integer or null")),
    }
}

fn task_string(value: &Value, key: &str, task_id: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("task_results/{task_id}/{key}: required string is missing"))
}

fn task_completion_explanation(
    task_id: &str,
    priority: &str,
    service_bytes: u64,
    deadline_window: u64,
    baseline_delivered: u64,
    candidate_delivered: u64,
    baseline_outcome: &str,
    candidate_outcome: &str,
    baseline_completion: Option<u64>,
    candidate_completion: Option<u64>,
    baseline_failure: Option<&str>,
    candidate_failure: Option<&str>,
) -> String {
    match (baseline_outcome, candidate_outcome) {
        ("failed", "succeeded") => {
            let failure = match baseline_failure {
                Some("deadline") => format!("missed its deadline at window {deadline_window}"),
                Some("expired") => "expired before completion".to_owned(),
                Some(reason) => format!("failed ({reason})"),
                None => "failed before completion".to_owned(),
            };
            format!(
                "Task {task_id} ({priority}) needed {service_bytes} bytes by window {deadline_window}. FIFO delivered {baseline_delivered}/{service_bytes} bytes and {failure}; fair-share delivered {candidate_delivered}/{service_bytes} bytes and completed{}.",
                candidate_completion.map_or_else(String::new, |window| format!(" in window {window}"))
            )
        }
        ("succeeded", "failed") => {
            let failure = match candidate_failure {
                Some("deadline") => format!("missed its deadline at window {deadline_window}"),
                Some("expired") => "expired before completion".to_owned(),
                Some(reason) => format!("failed ({reason})"),
                None => "failed before completion".to_owned(),
            };
            format!(
                "REGRESSION: task {task_id} ({priority}) needed {service_bytes} bytes by window {deadline_window}. FIFO completed it{} after delivering {baseline_delivered} bytes, but fair-share delivered {candidate_delivered}/{service_bytes} bytes and {failure}.",
                baseline_completion.map_or_else(String::new, |window| format!(" in window {window}"))
            )
        }
        ("failed", "failed") if candidate_delivered > baseline_delivered => {
            format!(
                "Task {task_id} ({priority}) did not finish by its deadline in either policy. Fair-share delivered more partial progress ({candidate_delivered}/{service_bytes} bytes) than FIFO ({baseline_delivered}/{service_bytes}); this is not a successful task outcome."
            )
        }
        ("failed", "failed") => {
            let baseline_reason = baseline_failure.unwrap_or("unknown reason");
            let candidate_reason = candidate_failure.unwrap_or("unknown reason");
            format!(
                "Task {task_id} ({priority}) failed under both policies. FIFO delivered {baseline_delivered}/{service_bytes} bytes ({baseline_reason}); fair-share delivered {candidate_delivered}/{service_bytes} bytes ({candidate_reason})."
            )
        }
        ("succeeded", "succeeded") => {
            format!(
                "Task {task_id} ({priority}) completed under both policies: FIFO in window {}, fair-share in window {}.",
                baseline_completion.map_or_else(|| "not recorded".to_owned(), |window| window.to_string()),
                candidate_completion.map_or_else(|| "not recorded".to_owned(), |window| window.to_string())
            )
        }
        _ => format!(
            "Task {task_id} ({priority}) has an unrecognized outcome label; inspect its raw task receipt."
        ),
    }
}

fn parse_task_comparisons(root: &Value) -> Result<Option<(Vec<TaskComparison>, u64, bool)>, String> {
    const MAX_EXPLANATIONS: usize = 50;
    let Some(task_results) = root.get("task_results") else {
        return Ok(None);
    };
    let groups = task_results
        .as_array()
        .filter(|groups| groups.len() == 2)
        .ok_or_else(|| "multiwindow_recovery/task_results: expected exactly two task-result arrays".to_owned())?;

    let mut baseline = BTreeMap::<String, &Value>::new();
    let mut candidate = BTreeMap::<String, &Value>::new();
    for group in groups {
        let rows = group
            .as_array()
            .ok_or_else(|| "multiwindow_recovery/task_results/*: expected an array".to_owned())?;
        for row in rows {
            let task_id = row
                .get("task_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "task_results/*/task_id: required task token is missing".to_owned())?;
            let arm = row
                .get("arm")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("task_results/{task_id}/arm: required arm is missing"))?;
            let target = match arm {
                "baseline" => &mut baseline,
                "candidate" => &mut candidate,
                other => return Err(format!("task_results/{task_id}/arm: unsupported arm '{other}'")),
            };
            if target.insert(task_id.to_owned(), row).is_some() {
                return Err(format!("task_results/{task_id}: duplicate task ID within arm"));
            }
        }
    }

    if baseline.len() != candidate.len() {
        return Err("task_results: baseline and candidate task counts differ; paired explanation refused".to_owned());
    }

    let mut comparisons = Vec::new();
    for (task_id, base) in &baseline {
        let cand = candidate
            .get(task_id)
            .ok_or_else(|| format!("task_results/{task_id}: candidate task result is missing"))?;

        let mode = task_string(base, "mode", task_id)?;
        let candidate_mode = task_string(cand, "mode", task_id)?;
        let priority = task_string(base, "priority", task_id)?;
        let candidate_priority = task_string(cand, "priority", task_id)?;
        let service_bytes = required_u64(base, "service_bytes", &format!("task_results/{task_id}"))?;
        let candidate_service = required_u64(cand, "service_bytes", &format!("task_results/{task_id}"))?;
        let arrival_window = required_u64(base, "arrival_window", &format!("task_results/{task_id}"))?;
        let candidate_arrival = required_u64(cand, "arrival_window", &format!("task_results/{task_id}"))?;
        let deadline_window = required_u64(base, "deadline_window", &format!("task_results/{task_id}"))?;
        let candidate_deadline = required_u64(cand, "deadline_window", &format!("task_results/{task_id}"))?;
        let expires_after_window = required_u64(base, "expires_after_window", &format!("task_results/{task_id}"))?;
        let candidate_expiry = required_u64(cand, "expires_after_window", &format!("task_results/{task_id}"))?;

        if mode != candidate_mode
            || priority != candidate_priority
            || service_bytes != candidate_service
            || arrival_window != candidate_arrival
            || deadline_window != candidate_deadline
            || expires_after_window != candidate_expiry
        {
            return Err(format!(
                "task_results/{task_id}: scenario, priority, demand or deadline metadata differs between arms"
            ));
        }
        if arrival_window > deadline_window || arrival_window > expires_after_window {
            return Err(format!("task_results/{task_id}: task arrives after its deadline or expiry"));
        }

        let baseline_delivered = required_u64(base, "delivered_bytes", &format!("task_results/{task_id}"))?;
        let candidate_delivered = required_u64(cand, "delivered_bytes", &format!("task_results/{task_id}"))?;
        if baseline_delivered > service_bytes || candidate_delivered > service_bytes {
            return Err(format!("task_results/{task_id}: delivered bytes exceed requested service bytes"));
        }
        let baseline_outcome = task_string(base, "outcome", task_id)?;
        let candidate_outcome = task_string(cand, "outcome", task_id)?;
        if !matches!(baseline_outcome.as_str(), "succeeded" | "failed")
            || !matches!(candidate_outcome.as_str(), "succeeded" | "failed")
        {
            return Err(format!("task_results/{task_id}: unrecognized outcome state"));
        }
        if (baseline_outcome == "succeeded") != (baseline_delivered == service_bytes)
            || (candidate_outcome == "succeeded") != (candidate_delivered == service_bytes)
        {
            return Err(format!("task_results/{task_id}: outcome state conflicts with delivered-byte total"));
        }

        let baseline_completion = optional_task_u64(base, "completion_window", task_id)?;
        let candidate_completion = optional_task_u64(cand, "completion_window", task_id)?;
        let baseline_failure = base
            .get("failure_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let candidate_failure = cand
            .get("failure_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if (baseline_outcome == "succeeded") != baseline_completion.is_some()
            || (candidate_outcome == "succeeded") != candidate_completion.is_some()
        {
            return Err(format!("task_results/{task_id}: completion window conflicts with task outcome"));
        }

        let changed = baseline_outcome != candidate_outcome
            || baseline_delivered != candidate_delivered
            || baseline_completion != candidate_completion
            || baseline_failure != candidate_failure;
        if changed {
            let explanation = task_completion_explanation(
                task_id,
                &priority,
                service_bytes,
                deadline_window,
                baseline_delivered,
                candidate_delivered,
                &baseline_outcome,
                &candidate_outcome,
                baseline_completion,
                candidate_completion,
                baseline_failure.as_deref(),
                candidate_failure.as_deref(),
            );
            comparisons.push(TaskComparison {
                task_id: task_id.clone(),
                mode,
                priority,
                service_bytes,
                arrival_window,
                deadline_window,
                expires_after_window,
                baseline_delivered_bytes: baseline_delivered,
                candidate_delivered_bytes: candidate_delivered,
                baseline_outcome,
                candidate_outcome,
                baseline_completion_window: baseline_completion,
                candidate_completion_window: candidate_completion,
                baseline_failure_reason: baseline_failure,
                candidate_failure_reason: candidate_failure,
                explanation,
            });
        }
    }

    comparisons.sort_by(|left, right| {
        let rank = |priority: &str| match priority {
            "essential" => 0_u8,
            "interactive" => 1,
            "bulk" => 2,
            _ => 3,
        };
        rank(&left.priority)
            .cmp(&rank(&right.priority))
            .then_with(|| left.mode.cmp(&right.mode))
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
    let total = comparisons.len() as u64;
    let truncated = comparisons.len() > MAX_EXPLANATIONS;
    comparisons.truncate(MAX_EXPLANATIONS);
    Ok(Some((comparisons, total, truncated)))
}

/// Explain a recovery simulation report at household, operator and review level.
///
/// Expects a report containing multiwindow_recovery.evidence_class and
/// multiwindow_recovery.per_mode. Each mode remains separate: those episodes
/// do not represent one simultaneous shared radio budget.
pub fn explain_report(report: &Value) -> Result<Explanation, String> {
    let root = report
        .get("multiwindow_recovery")
        .filter(|value| value.is_object())
        .ok_or_else(|| "multiwindow_recovery: required report object is missing".to_owned())?;
    let evidence_class =
        EvidenceClass::from_label(root.get("evidence_class").and_then(Value::as_str));
    let rows = root
        .get("per_mode")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| "multiwindow_recovery/per_mode: non-empty array is required".to_owned())?;

    let mut modes = Vec::new();
    let mut findings = Vec::new();
    let mut missing_priority_data = false;
    for row in rows {
        let mode = row
            .get("mode")
            .and_then(Value::as_str)
            .ok_or_else(|| "per_mode/*/mode: scenario label is missing".to_owned())?;
        let baseline = row
            .get("baseline")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("per_mode/{mode}/baseline: required object is missing"))?;
        let candidate = row
            .get("candidate_fair_share")
            .filter(|value| value.is_object())
            .ok_or_else(|| {
                format!("per_mode/{mode}/candidate_fair_share: required object is missing")
            })?;
        let baseline_tasks = required_u64(baseline, "task_count", mode)?;
        let candidate_tasks = required_u64(candidate, "task_count", mode)?;
        if baseline_tasks != candidate_tasks {
            return Err(format!(
                "per_mode/{mode}: baseline and candidate denominators differ; paired explanation refused"
            ));
        }
        let baseline_completed = required_u64(baseline, "completed_tasks", mode)?;
        let candidate_completed = required_u64(candidate, "completed_tasks", mode)?;
        if baseline_completed > baseline_tasks || candidate_completed > candidate_tasks {
            return Err(format!(
                "per_mode/{mode}: completed count exceeds task denominator"
            ));
        }
        let delta = completion_delta(candidate_completed, baseline_completed)?;
        let plain_explanation = if baseline_tasks == 0 {
            format!("No tasks were declared for {}.", mode_label(mode))
        } else if delta > 0 {
            format!(
                "In {label}, fair-share completes {delta} more {} than FIFO ({candidate_completed}/{candidate_tasks} versus {baseline_completed}/{baseline_tasks}).",
                task_noun(delta as u64),
                label = mode_label(mode)
            )
        } else if delta < 0 {
            format!(
                "In {label}, fair-share completes {} fewer {} than FIFO. Treat it as a regression, not an improvement.",
                delta.unsigned_abs(),
                task_noun(delta.unsigned_abs()),
                label = mode_label(mode)
            )
        } else {
            format!(
                "In {label}, both policies complete {baseline_completed}/{baseline_tasks} tasks.",
                label = mode_label(mode)
            )
        };
        if delta < 0 {
            findings.push(format!(
                "Task completion regresses in {label}.",
                label = mode_label(mode)
            ));
        }
        if let (Some(b), Some(c)) = (
            optional_u64(baseline, "completed_task_jain_fairness_ppm"),
            optional_u64(candidate, "completed_task_jain_fairness_ppm"),
        ) && c < b
        {
            findings.push(format!(
                "Completed-task fairness falls in {label} (Jain index {b} to {c} parts per million). Inspect per-peer outcomes.",
                label = mode_label(mode)
            ));
        }
        let by_priority = match parse_priority_rows(row)? {
            Some(rows) => {
                for priority in &rows {
                    if priority.baseline_deadline_misses.unwrap_or(0) > 0 {
                        findings.push(format!(
                            "FIFO misses {} {} deadline(s) in {label}.",
                            priority.baseline_deadline_misses.unwrap_or(0),
                            priority.priority,
                            label = mode_label(mode)
                        ));
                    }
                    if priority.candidate_deadline_misses.unwrap_or(0) > 0 {
                        findings.push(format!(
                            "Fair-share misses {} {} deadline(s) in {label}.",
                            priority.candidate_deadline_misses.unwrap_or(0),
                            priority.priority,
                            label = mode_label(mode)
                        ));
                    }
                    if priority.candidate_completed < priority.baseline_completed {
                        findings.push(format!(
                            "Fair-share completes fewer {} tasks in {label}.",
                            priority.priority,
                            label = mode_label(mode)
                        ));
                    }
                }
                rows
            }
            None => {
                missing_priority_data = true;
                Vec::new()
            }
        };

        modes.push(ModeExplanation {
            mode: mode.to_owned(),
            plain_label: mode_label(mode).to_owned(),
            task_count: baseline_tasks,
            baseline_completed,
            candidate_completed,
            completion_difference: delta,
            plain_explanation,
            by_priority,
        });
    }
    modes.sort_by(|a, b| a.mode.cmp(&b.mode));
    findings.sort();
    findings.dedup();

    let headline = match evidence_class {
        EvidenceClass::SyntheticSimulation => "The modeled workload may favor fair-share in the cases shown; live Wi-Fi performance has not been tested.",
        EvidenceClass::BenchObservedSnapshot => "A diagnostic snapshot was recorded; a snapshot alone cannot show whether user tasks succeeded.",
        EvidenceClass::LabMeasurement | EvidenceClass::FieldPilot | EvidenceClass::ReliabilityQualification => "This report declares measured evidence; review the raw paired trials and protocol before generalizing.",
        EvidenceClass::Unknown => "The evidence class is unknown; treat performance conclusions as unverified.",
    }.to_owned();

    let mut limitations = vec![
        evidence_class.caveat().to_owned(),
        "Each mode is an independent simulation episode; do not add its counts together as if all modes shared one simultaneous radio budget.".to_owned(),
        "A byte allocation is not measured 802.11 airtime fairness.".to_owned(),
        "This report alone does not establish real application success, TCP/QUIC behavior, radio retries, measured latency, calibrated energy, tariff savings, consent, or provider/regulatory permission.".to_owned(),
    ];
    if missing_priority_data {
        limitations.push("Per-priority outcome rows were absent, so essential or interactive deadline regressions cannot be ruled out from this explanation.".to_owned());
    }
    let (task_comparisons, task_comparisons_total, task_comparisons_truncated) =
        match parse_task_comparisons(root)? {
            Some((items, total, truncated)) => (items, total, truncated),
            None => {
                limitations.push(
                    "Task-level paired receipts were absent, so this explanation cannot attribute a specific task outcome to a deadline, partial-byte total, or terminal window.".to_owned(),
                );
                (Vec::new(), 0, false)
            }
        };
    if task_comparisons_truncated {
        limitations.push(format!(
            "Only the first {} changed task outcomes are shown; {} changed task outcomes exist in the paired receipt.",
            task_comparisons.len(),
            task_comparisons_total
        ));
    }
    let recommended_next_step = match evidence_class {
        EvidenceClass::SyntheticSimulation | EvidenceClass::Unknown => "Run a matched experiment on an owner-authorized isolated AP testbed. Bind the AP firmware/radio configuration, workload profile and harness revision; collect available counters plus application-verified paired task outcomes; review each priority/mode denominator.".to_owned(),
        _ => "Inspect the exact testbed manifest, calibration record and raw paired trials; repeat across devices, days and sites before claiming general performance or savings.".to_owned(),
    };

    Ok(Explanation {
        schema_version: EXPLAINER_VERSION.to_owned(),
        evidence_class,
        evidence_label: evidence_class.label().to_owned(),
        headline,
        simple_explanation: "Imagine a fixed byte budget in each abstract time window. FIFO serves older work first; fair-share distributes bytes between peers while reserving capacity for essential requests. Short urgent tasks can finish sooner in some workloads, but the simulation does not model the Wi-Fi radio itself.".to_owned(),
        findings,
        modes,
        task_comparisons,
        task_comparisons_total,
        task_comparisons_truncated,
        limitations,
        recommended_next_step,
        scenario_sha256: valid_sha(root.get("scenario_sha256")),
        profile_sha256: valid_sha(report.pointer("/source/profile_sha256"))
            .or_else(|| valid_sha(report.get("profile_sha256"))),
    })
}

fn display_optional_count(value: Option<u64>) -> String {
    value.map_or_else(|| "not recorded".to_owned(), |count| count.to_string())
}

/// Render an explanation for a human reader, without hiding its evidence class,
/// denominators, per-priority misses, limitations, or next step. This is a
/// deterministic template renderer over the same structured explanation used
/// for JSON output.
pub fn render_human(explanation: &Explanation) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "CONNECTIVITY REPORT — {}",
        explanation.evidence_label.to_uppercase()
    );
    let _ = writeln!(out, "{}", explanation.headline);
    out.push('\n');
    out.push_str("IN PLAIN LANGUAGE\n");
    let _ = writeln!(out, "{}", explanation.simple_explanation);
    out.push('\n');

    out.push_str("WHAT THE SCENARIOS SHOW\n");
    for mode in &explanation.modes {
        let _ = writeln!(out, "\n{}", mode.plain_label);
        let _ = writeln!(
            out,
            "  FIFO baseline: {}/{} tasks completed",
            mode.baseline_completed, mode.task_count
        );
        let _ = writeln!(
            out,
            "  Fair-share:    {}/{} tasks completed",
            mode.candidate_completed, mode.task_count
        );
        let comparison = if mode.completion_difference > 0 {
            format!("+{} completed", mode.completion_difference)
        } else if mode.completion_difference < 0 {
            format!("{} completed (regression)", mode.completion_difference)
        } else {
            "no completion-count difference".to_owned()
        };
        let _ = writeln!(out, "  Difference:    {comparison}");
        let _ = writeln!(out, "  Explanation:   {}", mode.plain_explanation);

        if !mode.by_priority.is_empty() {
            out.push_str("  By priority:\n");
            for priority in &mode.by_priority {
                let _ = writeln!(
                    out,
                    "    - {}: FIFO {}/{}, fair-share {}/{}; deadline misses FIFO {}, fair-share {}",
                    priority.priority,
                    priority.baseline_completed,
                    priority.tasks,
                    priority.candidate_completed,
                    priority.tasks,
                    display_optional_count(priority.baseline_deadline_misses),
                    display_optional_count(priority.candidate_deadline_misses),
                );
            }
        }
    }
    out.push('\n');

    out.push_str("TASKS THAT CHANGED OUTCOME OR PROGRESS\n");
    if explanation.task_comparisons.is_empty() {
        out.push_str("  No changed task-level outcomes were available in the paired receipt.\n");
    } else {
        for task in &explanation.task_comparisons {
            let _ = writeln!(out, "  - {}", task.explanation);
        }
        if explanation.task_comparisons_truncated {
            let _ = writeln!(
                out,
                "  Showing {} of {} changed task outcomes; inspect the full JSON for the remaining records.",
                explanation.task_comparisons.len(),
                explanation.task_comparisons_total
            );
        }
    }
    out.push('\n');

    out.push_str("KEY FINDINGS\n");
    if explanation.findings.is_empty() {
        out.push_str("  No additional rule-based findings were emitted from this report.\n");
    } else {
        for finding in &explanation.findings {
            let _ = writeln!(out, "  - {finding}");
        }
    }
    out.push('\n');

    out.push_str("WHAT THIS DOES NOT PROVE\n");
    for limitation in &explanation.limitations {
        let _ = writeln!(out, "  - {limitation}");
    }
    out.push('\n');

    out.push_str("RECOMMENDED NEXT STEP\n");
    let _ = writeln!(out, "{}", explanation.recommended_next_step);
    out.push('\n');

    out.push_str("PROVENANCE\n");
    let _ = writeln!(
        out,
        "  Scenario SHA-256: {}",
        explanation
            .scenario_sha256
            .as_deref()
            .unwrap_or("not supplied")
    );
    let _ = writeln!(
        out,
        "  Profile SHA-256:  {}",
        explanation
            .profile_sha256
            .as_deref()
            .unwrap_or("not supplied")
    );
    let _ = writeln!(out, "  Explainer schema: {}", explanation.schema_version);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn report() -> Value {
        json!({
            "multiwindow_recovery": {
                "evidence_class":"synthetic_simulation",
                "scenario_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "per_mode":[
                    {
                        "mode":"online",
                        "baseline":{"task_count":3,"completed_tasks":1,"completed_task_jain_fairness_ppm":333333},
                        "candidate_fair_share":{"task_count":3,"completed_tasks":3,"completed_task_jain_fairness_ppm":1000000},
                        "baseline_by_priority":[
                            {"priority":"essential","task_count":1,"completed_tasks":0,"deadline_misses":1,"expiry_misses":0},
                            {"priority":"interactive","task_count":1,"completed_tasks":0,"deadline_misses":1,"expiry_misses":0},
                            {"priority":"bulk","task_count":1,"completed_tasks":1,"deadline_misses":0,"expiry_misses":0}
                        ],
                        "candidate_by_priority":[
                            {"priority":"essential","task_count":1,"completed_tasks":1,"deadline_misses":0,"expiry_misses":0},
                            {"priority":"interactive","task_count":1,"completed_tasks":1,"deadline_misses":0,"expiry_misses":0},
                            {"priority":"bulk","task_count":1,"completed_tasks":1,"deadline_misses":0,"expiry_misses":0}
                        ]
                    },
                    {
                        "mode":"wan_partition",
                        "baseline":{"task_count":3,"completed_tasks":1},
                        "candidate_fair_share":{"task_count":3,"completed_tasks":3}
                    }
                ],
                "task_results": [
                    [
                        {"task_id":"online-essential-small-b","mode":"online","priority":"essential","arm":"baseline","arrival_window":0,"deadline_window":0,"expires_after_window":4,"service_bytes":3000,"delivered_bytes":0,"outcome":"failed","completion_window":null,"terminal_window":0,"failure_reason":"deadline"},
                        {"task_id":"online-interactive-small-c","mode":"online","priority":"interactive","arm":"baseline","arrival_window":0,"deadline_window":1,"expires_after_window":4,"service_bytes":4000,"delivered_bytes":3000,"outcome":"failed","completion_window":null,"terminal_window":1,"failure_reason":"deadline"},
                        {"task_id":"online-bulk-large-a","mode":"online","priority":"bulk","arm":"baseline","arrival_window":0,"deadline_window":4,"expires_after_window":4,"service_bytes":20000,"delivered_bytes":20000,"outcome":"succeeded","completion_window":2,"terminal_window":2,"failure_reason":null},
                        {"task_id":"partition-essential-small-b","mode":"wan_partition","priority":"essential","arm":"baseline","arrival_window":0,"deadline_window":0,"expires_after_window":4,"service_bytes":3000,"delivered_bytes":0,"outcome":"failed","completion_window":null,"terminal_window":0,"failure_reason":"deadline"},
                        {"task_id":"partition-interactive-small-c","mode":"wan_partition","priority":"interactive","arm":"baseline","arrival_window":0,"deadline_window":1,"expires_after_window":4,"service_bytes":4000,"delivered_bytes":3000,"outcome":"failed","completion_window":null,"terminal_window":1,"failure_reason":"deadline"},
                        {"task_id":"partition-bulk-large-a","mode":"wan_partition","priority":"bulk","arm":"baseline","arrival_window":0,"deadline_window":4,"expires_after_window":4,"service_bytes":20000,"delivered_bytes":20000,"outcome":"succeeded","completion_window":2,"terminal_window":2,"failure_reason":null}
                    ],
                    [
                        {"task_id":"online-essential-small-b","mode":"online","priority":"essential","arm":"candidate","arrival_window":0,"deadline_window":0,"expires_after_window":4,"service_bytes":3000,"delivered_bytes":3000,"outcome":"succeeded","completion_window":0,"terminal_window":0,"failure_reason":null},
                        {"task_id":"online-interactive-small-c","mode":"online","priority":"interactive","arm":"candidate","arrival_window":0,"deadline_window":1,"expires_after_window":4,"service_bytes":4000,"delivered_bytes":4000,"outcome":"succeeded","completion_window":1,"terminal_window":1,"failure_reason":null},
                        {"task_id":"online-bulk-large-a","mode":"online","priority":"bulk","arm":"candidate","arrival_window":0,"deadline_window":4,"expires_after_window":4,"service_bytes":20000,"delivered_bytes":20000,"outcome":"succeeded","completion_window":2,"terminal_window":2,"failure_reason":null},
                        {"task_id":"partition-essential-small-b","mode":"wan_partition","priority":"essential","arm":"candidate","arrival_window":0,"deadline_window":0,"expires_after_window":4,"service_bytes":3000,"delivered_bytes":3000,"outcome":"succeeded","completion_window":0,"terminal_window":0,"failure_reason":null},
                        {"task_id":"partition-interactive-small-c","mode":"wan_partition","priority":"interactive","arm":"candidate","arrival_window":0,"deadline_window":1,"expires_after_window":4,"service_bytes":4000,"delivered_bytes":4000,"outcome":"succeeded","completion_window":1,"terminal_window":1,"failure_reason":null},
                        {"task_id":"partition-bulk-large-a","mode":"wan_partition","priority":"bulk","arm":"candidate","arrival_window":0,"deadline_window":4,"expires_after_window":4,"service_bytes":20000,"delivered_bytes":20000,"outcome":"succeeded","completion_window":2,"terminal_window":2,"failure_reason":null}
                    ]
                ]
            }
        })
    }

    #[test]
    fn explicit_simulation_boundary_is_preserved() {
        let explanation = explain_report(&report()).unwrap();
        assert_eq!(
            explanation.evidence_class,
            EvidenceClass::SyntheticSimulation
        );
        assert!(
            explanation
                .headline
                .contains("live Wi-Fi performance has not been tested")
        );
        assert!(
            explanation
                .limitations
                .iter()
                .any(|item| item.contains("airtime"))
        );
    }

    #[test]
    fn mode_episodes_are_not_summed() {
        let explanation = explain_report(&report()).unwrap();
        assert_eq!(explanation.modes.len(), 2);
        assert!(
            explanation
                .limitations
                .iter()
                .any(|item| item.contains("independent simulation episode"))
        );
    }

    #[test]
    fn essential_deadline_misses_are_visible() {
        let explanation = explain_report(&report()).unwrap();
        assert!(
            explanation
                .findings
                .iter()
                .any(|item| item.contains("FIFO misses 1 essential deadline"))
        );
        let online = explanation
            .modes
            .iter()
            .find(|item| item.mode == "online")
            .unwrap();
        let essential = online
            .by_priority
            .iter()
            .find(|item| item.priority == "essential")
            .unwrap();
        assert_eq!(essential.baseline_deadline_misses, Some(1));
        assert_eq!(essential.candidate_completed, 1);
    }

    #[test]
    fn unknown_evidence_class_is_not_promoted() {
        let mut value = report();
        value["multiwindow_recovery"]["evidence_class"] = json!("unrecognized");
        let explanation = explain_report(&value).unwrap();
        assert_eq!(explanation.evidence_class, EvidenceClass::Unknown);
        assert!(explanation.headline.contains("unverified"));
    }

    #[test]
    fn missing_numbers_and_denominator_mismatches_fail_closed() {
        let mut value = report();
        value["multiwindow_recovery"]["per_mode"][0]["baseline"]["completed_tasks"] = Value::Null;
        assert!(
            explain_report(&value)
                .unwrap_err()
                .contains("missing or invalid")
        );

        let mut value = report();
        value["multiwindow_recovery"]["per_mode"][0]["candidate_fair_share"]["task_count"] =
            json!(2);
        assert!(
            explain_report(&value)
                .unwrap_err()
                .contains("denominators differ")
        );
    }

    #[test]
    fn absent_priority_data_is_disclosed_not_filled_in() {
        let mut value = report();
        value["multiwindow_recovery"]["per_mode"][0]
            .as_object_mut()
            .unwrap()
            .remove("baseline_by_priority");
        let explanation = explain_report(&value).unwrap();
        assert!(
            explanation
                .limitations
                .iter()
                .any(|item| item.contains("Per-priority outcome rows were absent"))
        );
    }

    #[test]
    fn human_rendering_keeps_evidence_limits_and_priority_misses_visible() {
        let explanation = explain_report(&report()).unwrap();
        let rendered = render_human(&explanation);
        assert!(rendered.contains("synthetic simulation"));
        assert!(rendered.contains("live Wi-Fi performance has not been tested"));
        assert!(rendered.contains("WHAT THIS DOES NOT PROVE"));
        assert!(rendered.contains("FIFO misses 1 essential deadline"));
        assert!(rendered.contains("Scenario SHA-256: aaaaaaaaaa"));
        assert!(!rendered.contains("Recommended NEXT STEP"));
        assert!(rendered.contains("RECOMMENDED NEXT STEP"));
    }

    #[test]
    fn task_level_explanation_binds_partial_bytes_to_deadline_context() {
        let explanation = explain_report(&report()).unwrap();
        assert_eq!(explanation.task_comparisons_total, 4);
        assert!(!explanation.task_comparisons_truncated);
        let essential = explanation
            .task_comparisons
            .iter()
            .find(|task| task.task_id == "online-essential-small-b")
            .unwrap();
        assert_eq!(essential.deadline_window, 0);
        assert_eq!(essential.baseline_delivered_bytes, 0);
        assert_eq!(essential.candidate_delivered_bytes, 3000);
        assert_eq!(essential.baseline_failure_reason.as_deref(), Some("deadline"));
        assert!(essential.explanation.contains("needed 3000 bytes by window 0"));
        assert!(essential.explanation.contains("FIFO delivered 0/3000 bytes and missed its deadline"));
        assert!(essential.explanation.contains("completed in window 0"));
        let rendered = render_human(&explanation);
        assert!(rendered.contains("TASKS THAT CHANGED OUTCOME OR PROGRESS"));
        assert!(rendered.contains("online-essential-small-b"));
    }

    #[test]
    fn task_pair_metadata_mismatch_fails_closed() {
        let mut value = report();
        value["multiwindow_recovery"]["task_results"][1][0]["deadline_window"] = json!(2);
        let error = explain_report(&value).unwrap_err();
        assert!(error.contains("metadata differs"));
    }

    #[test]
    fn missing_task_rows_are_disclosed_not_assumed_clean() {
        let mut value = report();
        value["multiwindow_recovery"]
            .as_object_mut()
            .unwrap()
            .remove("task_results");
        let explanation = explain_report(&value).unwrap();
        assert!(explanation.task_comparisons.is_empty());
        assert!(explanation.limitations.iter().any(|item| item.contains("Task-level paired receipts were absent")));
    }

    #[test]
    fn serialized_explanation_is_deterministic() {
        let value = report();
        assert_eq!(
            serde_json::to_vec(&explain_report(&value).unwrap()).unwrap(),
            serde_json::to_vec(&explain_report(&value).unwrap()).unwrap()
        );
    }
}
