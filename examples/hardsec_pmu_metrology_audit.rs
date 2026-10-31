//! HARDSEC-018 PMU metrology and observer-effect auditor.
//!
//! Qualifies declared counter acquisition configurations using matched trial
//! blocks, singleton reads, co-scheduled groups, overcommitted groups and a
//! no-PMU workload baseline. Multiplex-scaled counts remain estimates and are
//! never promoted to directly observed values.
//!
//! Usage:
//!   hardsec_pmu_metrology_audit --plan plan.json \
//!     --observations observations.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-pmu-metrology-plan-v1";
const OBS_SCHEMA: &str = "hardsec-pmu-observations-v1";
const RECEIPT_SCHEMA: &str = "hardsec-pmu-metrology-receipt-v1";
const CONDITIONS: &[&str] = &[
    "baseline_no_pmu",
    "singleton_per_event",
    "grouped_co_scheduled",
    "grouped_overcommitted",
];

#[derive(Debug, Clone)]
struct EventSpec {
    id: String,
    encoding: String,
    semantic: String,
    unit: String,
}

#[derive(Debug, Clone)]
struct EventObservation {
    id: String,
    encoding: String,
    raw_count: u64,
    enabled_ns: u64,
    running_ns: u64,
    sample_count: u64,
    lost_samples: u64,
}

#[derive(Debug, Clone)]
struct Trial {
    id: String,
    block_id: String,
    order_index: usize,
    seed: String,
    condition: String,
    work_units: u64,
    elapsed_ns: u64,
    pmu_read_overhead_ns: u64,
    events: Vec<EventObservation>,
}

#[derive(Debug, Clone)]
struct Plan {
    protocol_id: String,
    evidence_domain: String,
    platform_id: String,
    workload_id: String,
    fixed_work_units: u64,
    counter_capacity: usize,
    events: Vec<EventSpec>,
    baseline_condition: String,
    singleton_condition: String,
    co_condition: String,
    over_condition: String,
    co_event_ids: Vec<String>,
    over_event_ids: Vec<String>,
    randomization_seed: String,
    minimum_blocks: usize,
    minimum_singleton_running_fraction: f64,
    minimum_co_running_fraction: f64,
    maximum_co_deviation: f64,
    maximum_throughput_degradation: f64,
}

#[derive(Debug, Clone)]
struct Args {
    plan: PathBuf,
    observations: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn nonempty(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(format!("{context} must be non-empty and contain no control characters"));
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    let result = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))?;
    nonempty(result, &format!("{context}.{key}"))?;
    Ok(result)
}

fn u64_field(value: &Value, key: &str, context: &str) -> Result<u64, String> {
    value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))
}

fn usize_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    usize::try_from(u64_field(value, key, context)?)
        .map_err(|_| format!("{context}.{key} is outside the supported range"))
}

fn f64_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}.{key} must be numeric"))?;
    if !result.is_finite() {
        return Err(format!("{context}.{key} must be finite"));
    }
    Ok(result)
}

fn bool_true(value: &Value, key: &str, context: &str) -> Result<(), String> {
    if value.get(key).and_then(Value::as_bool) != Some(true) {
        return Err(format!("{context}.{key} must be true"));
    }
    Ok(())
}

fn string_array(value: &Value, key: &str, context: &str) -> Result<Vec<String>, String> {
    let records = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    let mut seen = BTreeSet::new();
    let mut output = Vec::with_capacity(records.len());
    for (index, item) in records.iter().enumerate() {
        let item = item.as_str().ok_or_else(|| format!("{context}.{key}[{index}] must be a string"))?;
        nonempty(item, &format!("{context}.{key}[{index}]"))?;
        if !seen.insert(item.to_owned()) {
            return Err(format!("{context}.{key} contains duplicate entry {item}"));
        }
        output.push(item.to_owned());
    }
    Ok(output)
}

fn parse_plan(value: &Value) -> Result<Plan, String> {
    if string_field(value, "schema_version", "plan")? != PLAN_SCHEMA {
        return Err("unsupported PMU metrology plan schema_version".to_owned());
    }
    let protocol_id = string_field(value, "protocol_id", "plan")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "plan")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err(format!("unsupported evidence_domain {evidence_domain}"));
    }

    let platform = value.get("platform").ok_or("plan.platform is required")?;
    let platform_id = string_field(platform, "platform_id", "plan.platform")?.to_owned();
    for key in ["architecture", "cpu_model", "pmu_model", "firmware_id", "kernel_id", "perf_id"] {
        let _ = string_field(platform, key, "plan.platform")?;
    }
    let counter_capacity = usize_field(platform, "programmable_counter_capacity", "plan.platform")?;
    if counter_capacity == 0 {
        return Err("platform.programmable_counter_capacity must be positive".to_owned());
    }

    let workload = value.get("workload").ok_or("plan.workload is required")?;
    let workload_id = string_field(workload, "workload_id", "plan.workload")?.to_owned();
    let binary_sha = string_field(workload, "binary_sha256", "plan.workload")?;
    if binary_sha.len() != 64 || !binary_sha.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err("plan.workload.binary_sha256 must be a lowercase SHA-256 digest".to_owned());
    }
    let _ = string_field(workload, "work_unit", "plan.workload")?;
    let fixed_work_units = u64_field(workload, "fixed_work_units", "plan.workload")?;
    if fixed_work_units == 0 {
        return Err("workload.fixed_work_units must be positive".to_owned());
    }

    let event_values = value.get("events").and_then(Value::as_array)
        .ok_or("plan.events must be an array")?;
    if event_values.len() < 2 {
        return Err("plan.events must name at least two distinct PMU events".to_owned());
    }
    let mut events = Vec::with_capacity(event_values.len());
    let mut event_ids = BTreeSet::new();
    let mut encodings = BTreeSet::new();
    for (index, item) in event_values.iter().enumerate() {
        let context = format!("plan.events[{index}]");
        let event = EventSpec {
            id: string_field(item, "event_id", &context)?.to_owned(),
            encoding: string_field(item, "encoding", &context)?.to_owned(),
            semantic: string_field(item, "semantic", &context)?.to_owned(),
            unit: string_field(item, "unit", &context)?.to_owned(),
        };
        if !event_ids.insert(event.id.clone()) {
            return Err(format!("duplicate event_id {}", event.id));
        }
        if !encodings.insert(event.encoding.clone()) {
            return Err(format!("duplicate event encoding {}; aliases require explicit review and cannot be silently co-reported", event.encoding));
        }
        events.push(event);
    }

    let conditions = value.get("acquisition_conditions").ok_or("plan.acquisition_conditions is required")?;
    let baseline_condition = string_field(conditions, "baseline_condition", "plan.acquisition_conditions")?.to_owned();
    let singleton_condition = string_field(conditions, "singleton_condition", "plan.acquisition_conditions")?.to_owned();
    let co_condition = string_field(conditions, "co_scheduled_condition", "plan.acquisition_conditions")?.to_owned();
    let over_condition = string_field(conditions, "overcommitted_condition", "plan.acquisition_conditions")?.to_owned();
    if [baseline_condition.as_str(), singleton_condition.as_str(), co_condition.as_str(), over_condition.as_str()] != [
        CONDITIONS[0], CONDITIONS[1], CONDITIONS[2], CONDITIONS[3]
    ] {
        return Err("acquisition_conditions must declare the four v1 conditions in their canonical roles".to_owned());
    }
    let co_event_ids = string_array(conditions, "co_scheduled_event_ids", "plan.acquisition_conditions")?;
    let over_event_ids = string_array(conditions, "overcommitted_event_ids", "plan.acquisition_conditions")?;
    let all_event_ids: BTreeSet<_> = events.iter().map(|e| e.id.as_str()).collect();
    if co_event_ids.len() < 2 || co_event_ids.len() > counter_capacity {
        return Err("co_scheduled_event_ids count must be at least two and no greater than programmable counter capacity".to_owned());
    }
    if over_event_ids.len() <= counter_capacity {
        return Err("overcommitted_event_ids count must exceed programmable counter capacity".to_owned());
    }
    if co_event_ids.iter().any(|id| !all_event_ids.contains(id.as_str()))
        || over_event_ids.iter().any(|id| !all_event_ids.contains(id.as_str())) {
        return Err("acquisition condition references an event absent from plan.events".to_owned());
    }

    let protocol = value.get("protocol").ok_or("plan.protocol is required")?;
    bool_true(protocol, "randomized_order_required", "plan.protocol")?;
    bool_true(protocol, "matched_blocks_required", "plan.protocol")?;
    let randomization_seed = string_field(protocol, "randomization_seed", "plan.protocol")?.to_owned();
    let minimum_blocks = usize_field(protocol, "minimum_blocks_for_descriptive_quantiles", "plan.protocol")?;
    if minimum_blocks < 2 {
        return Err("minimum_blocks_for_descriptive_quantiles must be at least 2".to_owned());
    }
    let minimum_singleton_running_fraction = f64_field(protocol, "minimum_singleton_running_fraction", "plan.protocol")?;
    let minimum_co_running_fraction = f64_field(protocol, "minimum_co_scheduled_running_fraction", "plan.protocol")?;
    let maximum_co_deviation = f64_field(protocol, "maximum_co_scheduled_relative_deviation", "plan.protocol")?;
    let maximum_throughput_degradation = f64_field(protocol, "maximum_instrumentation_throughput_degradation", "plan.protocol")?;
    for (label, value) in [
        ("minimum_singleton_running_fraction", minimum_singleton_running_fraction),
        ("minimum_co_scheduled_running_fraction", minimum_co_running_fraction),
        ("maximum_co_scheduled_relative_deviation", maximum_co_deviation),
        ("maximum_instrumentation_throughput_degradation", maximum_throughput_degradation),
    ] {
        if !(0.0..=1.0).contains(&value) || (label.starts_with("minimum_") && value == 0.0) {
            return Err(format!("protocol.{label} must be within the supported (0,1] or [0,1] range"));
        }
    }
    if string_field(value, "claim_scope", "plan")? != "measurement_configuration_and_declared_trial_record_qualification_only" {
        return Err("unsupported plan.claim_scope".to_owned());
    }

    Ok(Plan {
        protocol_id, evidence_domain, platform_id, workload_id, fixed_work_units, counter_capacity,
        events, baseline_condition, singleton_condition, co_condition, over_condition,
        co_event_ids, over_event_ids, randomization_seed, minimum_blocks,
        minimum_singleton_running_fraction, minimum_co_running_fraction,
        maximum_co_deviation, maximum_throughput_degradation,
    })
}

fn parse_event_observation(value: &Value, context: &str) -> Result<EventObservation, String> {
    let id = string_field(value, "event_id", context)?.to_owned();
    let encoding = string_field(value, "encoding", context)?.to_owned();
    let raw_count = u64_field(value, "raw_count", context)?;
    let enabled_ns = u64_field(value, "time_enabled_ns", context)?;
    let running_ns = u64_field(value, "time_running_ns", context)?;
    let sample_count = u64_field(value, "sample_count", context)?;
    let lost_samples = u64_field(value, "lost_samples", context)?;
    if enabled_ns == 0 || running_ns == 0 {
        return Err(format!("{context}: zero time_enabled/time_running is not a usable counter observation"));
    }
    if running_ns > enabled_ns {
        return Err(format!("{context}: time_running_ns exceeds time_enabled_ns"));
    }
    if lost_samples > sample_count {
        return Err(format!("{context}: lost_samples exceeds sample_count"));
    }
    Ok(EventObservation { id, encoding, raw_count, enabled_ns, running_ns, sample_count, lost_samples })
}

fn parse_trials(value: &Value, plan: &Plan) -> Result<Vec<Trial>, String> {
    if string_field(value, "schema_version", "observations")? != OBS_SCHEMA {
        return Err("unsupported PMU observations schema_version".to_owned());
    }
    if string_field(value, "protocol_id", "observations")? != plan.protocol_id {
        return Err("observations.protocol_id does not match plan".to_owned());
    }
    if string_field(value, "evidence_domain", "observations")? != plan.evidence_domain {
        return Err("observations evidence_domain must match plan evidence_domain".to_owned());
    }
    let records = value.get("trials").and_then(Value::as_array)
        .ok_or("observations.trials must be an array")?;
    if records.is_empty() {
        return Err("observations.trials must not be empty".to_owned());
    }
    let event_specs: BTreeMap<_, _> = plan.events.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut trial_ids = BTreeSet::new();
    let mut trials = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let context = format!("observations.trials[{index}]");
        let id = string_field(record, "trial_id", &context)?.to_owned();
        if !trial_ids.insert(id.clone()) {
            return Err(format!("duplicate trial_id {id}"));
        }
        if string_field(record, "protocol_id", &context)? != plan.protocol_id
            || string_field(record, "platform_id", &context)? != plan.platform_id
            || string_field(record, "workload_id", &context)? != plan.workload_id {
            return Err(format!("{context}: platform/workload/protocol identity mismatch"));
        }
        let condition = string_field(record, "condition", &context)?.to_owned();
        if !CONDITIONS.contains(&condition.as_str()) {
            return Err(format!("{context}: unsupported acquisition condition {condition}"));
        }
        let work_units = u64_field(record, "work_units", &context)?;
        if work_units != plan.fixed_work_units {
            return Err(format!("{context}: work_units must equal plan.workload.fixed_work_units"));
        }
        let events_value = record.get("events").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.events must be an array"))?;
        let mut events = Vec::with_capacity(events_value.len());
        let mut seen_event_ids = BTreeSet::new();
        for (event_index, event_value) in events_value.iter().enumerate() {
            let event_context = format!("{context}.events[{event_index}]");
            let event = parse_event_observation(event_value, &event_context)?;
            let spec = event_specs.get(event.id.as_str())
                .ok_or_else(|| format!("{event_context}: event_id is not present in plan"))?;
            if event.encoding != spec.encoding {
                return Err(format!("{event_context}: event encoding does not match the plan's semantic event identity"));
            }
            if !seen_event_ids.insert(event.id.clone()) {
                return Err(format!("{event_context}: duplicate event_id in trial"));
            }
            events.push(event);
        }
        let trial = Trial {
            id, block_id: string_field(record, "block_id", &context)?.to_owned(),
            order_index: usize_field(record, "order_index", &context)?,
            seed: string_field(record, "randomization_seed", &context)?.to_owned(),
            condition,
            work_units,
            elapsed_ns: u64_field(record, "elapsed_ns", &context)?,
            pmu_read_overhead_ns: u64_field(record, "pmu_read_overhead_ns", &context)?,
            events,
        };
        if trial.elapsed_ns == 0 {
            return Err(format!("{context}.elapsed_ns must be positive"));
        }
        trials.push(trial);
    }
    Ok(trials)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() { return None; }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() { return None; }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((quantile * sorted.len() as f64).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[index])
}

fn value_or_null(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |v| json!(v))
}

fn validate_blocks(trials: &[Trial], plan: &Plan) -> Result<BTreeMap<String, BTreeMap<String, Trial>>, String> {
    let mut blocks: BTreeMap<String, Vec<&Trial>> = BTreeMap::new();
    for trial in trials {
        if trial.seed != plan.randomization_seed {
            return Err(format!("trial {} randomization seed differs from the preregistered plan", trial.id));
        }
        blocks.entry(trial.block_id.clone()).or_default().push(trial);
    }
    if blocks.len() < plan.minimum_blocks {
        return Err(format!("{} matched blocks are present; at least {} are required for descriptive quantiles", blocks.len(), plan.minimum_blocks));
    }
    let mut output = BTreeMap::new();
    for (block_id, block_trials) in blocks {
        let work_units: BTreeSet<_> = block_trials.iter().map(|t| t.work_units).collect();
        if work_units.len() != 1 {
            return Err(format!("block {block_id} does not use a matched fixed-work unit count"));
        }
        let mut order = BTreeSet::new();
        for trial in &block_trials {
            if !order.insert(trial.order_index) {
                return Err(format!("block {block_id} contains duplicate randomized order_index"));
            }
        }
        if order.len() != block_trials.len()
            || order.iter().enumerate().any(|(idx, value)| *value != idx + 1) {
            return Err(format!("block {block_id} randomized order_index values must be contiguous from 1"));
        }

        let mut by_condition: BTreeMap<String, Vec<&Trial>> = BTreeMap::new();
        for trial in &block_trials {
            by_condition.entry(trial.condition.clone()).or_default().push(trial);
        }
        for condition in CONDITIONS {
            let expected = if *condition == plan.singleton_condition { plan.events.len() } else { 1 };
            let actual = by_condition.get(*condition).map_or(0, Vec::len);
            if actual != expected {
                return Err(format!("block {block_id} condition {condition} has {actual} trials; expected {expected}"));
            }
        }
        if by_condition.len() != CONDITIONS.len() {
            return Err(format!("block {block_id} has an unexpected or missing acquisition condition"));
        }

        let baseline = by_condition[plan.baseline_condition.as_str()][0];
        if !baseline.events.is_empty() || baseline.pmu_read_overhead_ns != 0 {
            return Err(format!("block {block_id} baseline_no_pmu must have no events and zero PMU read overhead"));
        }

        let mut singleton_by_event: BTreeMap<String, Trial> = BTreeMap::new();
        for trial in &by_condition[plan.singleton_condition.as_str()] {
            if trial.events.len() != 1 {
                return Err(format!("block {block_id} singleton trial {} must contain exactly one event", trial.id));
            }
            let event_id = trial.events[0].id.clone();
            if singleton_by_event.insert(event_id.clone(), (**trial).clone()).is_some() {
                return Err(format!("block {block_id} has duplicate singleton observation for event {event_id}"));
            }
            if (trial.events[0].running_ns as f64 / trial.events[0].enabled_ns as f64) < plan.minimum_singleton_running_fraction {
                return Err(format!("block {block_id} singleton event {event_id} falls below minimum running fraction"));
            }
        }
        if singleton_by_event.len() != plan.events.len()
            || plan.events.iter().any(|event| !singleton_by_event.contains_key(&event.id)) {
            return Err(format!("block {block_id} does not have exactly one singleton trial per planned event"));
        }

        let co_trial = by_condition[plan.co_condition.as_str()][0];
        let co_ids: BTreeSet<_> = co_trial.events.iter().map(|e| e.id.as_str()).collect();
        let expected_co: BTreeSet<_> = plan.co_event_ids.iter().map(String::as_str).collect();
        if co_ids != expected_co {
            return Err(format!("block {block_id} co-scheduled event set differs from plan"));
        }
        if co_trial.events.iter().any(|event| (event.running_ns as f64 / event.enabled_ns as f64) < plan.minimum_co_running_fraction) {
            return Err(format!("block {block_id} co-scheduled event falls below minimum running fraction; it is not a simultaneous observation"));
        }

        let over_trial = by_condition[plan.over_condition.as_str()][0];
        let over_ids: BTreeSet<_> = over_trial.events.iter().map(|e| e.id.as_str()).collect();
        let expected_over: BTreeSet<_> = plan.over_event_ids.iter().map(String::as_str).collect();
        if over_ids != expected_over {
            return Err(format!("block {block_id} overcommitted event set differs from plan"));
        }
        if !over_trial.events.iter().any(|event| event.running_ns < event.enabled_ns) {
            return Err(format!("block {block_id} overcommitted group records no multiplexing despite its declared overcommit"));
        }
        output.insert(block_id, singleton_by_event);
    }
    Ok(output)
}

fn audit_values(plan_value: &Value, observation_value: &Value, plan_bytes: &[u8], observation_bytes: &[u8]) -> Result<Value, String> {
    let plan = parse_plan(plan_value)?;
    let trials = parse_trials(observation_value, &plan)?;
    let blocks = validate_blocks(&trials, &plan)?;

    let mut grouped_by_block: BTreeMap<String, BTreeMap<String, &Trial>> = BTreeMap::new();
    for trial in &trials {
        grouped_by_block.entry(trial.block_id.clone()).or_default().insert(trial.condition.clone(), trial);
    }

    let mut event_reports = Vec::new();
    let mut all_event_gates_pass = true;
    for spec in &plan.events {
        let mut co_deviation = Vec::new();
        let mut over_deviation = Vec::new();
        let mut singleton_running = Vec::new();
        let mut co_running = Vec::new();
        let mut over_running = Vec::new();
        for (block_id, singleton_map) in &blocks {
            let singleton = singleton_map.get(&spec.id).ok_or("validated singleton invariant failed")?;
            let singleton_obs = singleton.events.first().ok_or("singleton event invariant failed")?;
            let grouped = grouped_by_block.get(block_id).ok_or("grouped block invariant failed")?;
            let co_trial = grouped.get(&plan.co_condition).ok_or("co condition invariant failed")?;
            if let Some(co_obs) = co_trial.events.iter().find(|e| e.id == spec.id) {
                let ratio = co_obs.running_ns as f64 / co_obs.enabled_ns as f64;
                co_running.push(ratio);
                if singleton_obs.raw_count > 0 {
                    let relative = (co_obs.raw_count as f64 / singleton_obs.raw_count as f64) - 1.0;
                    co_deviation.push(relative.abs());
                }
            }
            let over_trial = grouped.get(&plan.over_condition).ok_or("over condition invariant failed")?;
            if let Some(over_obs) = over_trial.events.iter().find(|e| e.id == spec.id) {
                let ratio = over_obs.running_ns as f64 / over_obs.enabled_ns as f64;
                over_running.push(ratio);
                if singleton_obs.raw_count > 0 {
                    let scaled_estimate = over_obs.raw_count as f64 * over_obs.enabled_ns as f64 / over_obs.running_ns as f64;
                    let relative = (scaled_estimate / singleton_obs.raw_count as f64) - 1.0;
                    over_deviation.push(relative.abs());
                }
            }
            singleton_running.push(singleton_obs.running_ns as f64 / singleton_obs.enabled_ns as f64);
        }
        let co_p95 = percentile(&co_deviation, 0.95);
        let is_co_scheduled = plan.co_event_ids.contains(&spec.id);
        let co_pass = !is_co_scheduled || (
            co_p95.is_some_and(|v| v <= plan.maximum_co_deviation)
                && co_running.iter().all(|v| *v >= plan.minimum_co_running_fraction)
                && singleton_running.iter().all(|v| *v >= plan.minimum_singleton_running_fraction)
        );
        all_event_gates_pass &= co_pass;
        let co_status = if !is_co_scheduled {
            "not_in_co_scheduled_group_not_tested"
        } else if co_pass {
            "confirmatory_eligible_under_declared_thresholds"
        } else {
            "not_qualified_by_declared_thresholds"
        };
        let over_p95 = percentile(&over_deviation, 0.95);
        let total_samples: u64 = trials.iter().flat_map(|trial| trial.events.iter())
            .filter(|event| event.id == spec.id)
            .map(|event| event.sample_count).sum();
        let lost_samples: u64 = trials.iter().flat_map(|trial| trial.events.iter())
            .filter(|event| event.id == spec.id)
            .map(|event| event.lost_samples).sum();
        event_reports.push(json!({
            "event_id":spec.id,
            "encoding":spec.encoding,
            "semantic":spec.semantic,
            "unit":spec.unit,
            "singleton_running_fraction":{
                "minimum":value_or_null(singleton_running.iter().copied().reduce(f64::min)),
                "median":value_or_null(median(&singleton_running))
            },
            "co_scheduled":{
                "paired_blocks":co_deviation.len(),
                "absolute_relative_deviation_fraction_median":value_or_null(median(&co_deviation)),
                "absolute_relative_deviation_fraction_p95":value_or_null(co_p95),
                "minimum_running_fraction":value_or_null(co_running.iter().copied().reduce(f64::min)),
                "status":co_status
            },
            "overcommitted":{
                "paired_blocks":over_deviation.len(),
                "scaled_count_relative_deviation_fraction_median":value_or_null(median(&over_deviation)),
                "scaled_count_relative_deviation_fraction_p95":value_or_null(over_p95),
                "minimum_running_fraction":value_or_null(over_running.iter().copied().reduce(f64::min)),
                "value_semantics":"multiplex_scaled_estimate_not_observed_count",
                "confirmatory_eligible":false
            },
            "sampling":{
                "sample_count":total_samples,
                "lost_samples":lost_samples,
                "loss_fraction":if total_samples == 0 {Value::Null} else {json!(lost_samples as f64 / total_samples as f64)}
            }
        }));
    }

    let mut throughput_by_condition: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut observer_degradation_by_condition: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut read_overhead_by_condition: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut block_map: BTreeMap<String, BTreeMap<String, &Trial>> = BTreeMap::new();
    for trial in &trials {
        block_map.entry(trial.block_id.clone()).or_default().insert(trial.condition.clone(), trial);
    }
    for (block_id, conditions) in &block_map {
        let baseline = conditions.get(&plan.baseline_condition).ok_or("missing baseline after block validation")?;
        let baseline_rate = baseline.work_units as f64 / baseline.elapsed_ns as f64;
        for condition in CONDITIONS.iter().filter(|c| **c != plan.baseline_condition.as_str()) {
            let trial = conditions.get(*condition).ok_or("missing condition after block validation")?;
            let rate = trial.work_units as f64 / trial.elapsed_ns as f64;
            let degradation = 1.0 - (rate / baseline_rate);
            throughput_by_condition.entry((*condition).to_owned()).or_default().push(rate);
            observer_degradation_by_condition.entry((*condition).to_owned()).or_default().push(degradation);
            read_overhead_by_condition.entry((*condition).to_owned()).or_default()
                .push(trial.pmu_read_overhead_ns as f64 / baseline.elapsed_ns as f64);
        }
        let _ = block_id;
    }

    let mut throughput_reports = Vec::new();
    let mut throughput_gate_pass = true;
    for condition in CONDITIONS.iter().filter(|c| **c != plan.baseline_condition) {
        let degradations = observer_degradation_by_condition.get(*condition).ok_or("throughput condition invariant failed")?;
        let p95 = percentile(degradations, 0.95).unwrap_or(f64::INFINITY);
        let pass = p95 <= plan.maximum_throughput_degradation;
        throughput_gate_pass &= pass;
        throughput_reports.push(json!({
            "condition":condition,
            "relative_throughput_degradation_fraction_median":value_or_null(median(degradations)),
            "relative_throughput_degradation_fraction_p95":if p95.is_finite() {json!(p95)} else {Value::Null},
            "pmu_read_overhead_fraction_median":value_or_null(read_overhead_by_condition.get(*condition).and_then(|v| median(v))),
            "status":if pass {"within_declared_throughput_threshold"} else {"exceeds_declared_throughput_threshold"}
        }));
    }

    let total_trials = trials.len();
    let block_count = blocks.len();
    let overall_pass = all_event_gates_pass && throughput_gate_pass;
    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":if overall_pass {"qualified_for_named_acquisition_conditions"} else {"measurement_quality_gate_failed"},
        "claim_scope":"measurement_configuration_and_declared_trial_record_qualification_only",
        "protocol_id":plan.protocol_id,
        "evidence_domain":plan.evidence_domain,
        "platform_id":plan.platform_id,
        "workload_id":plan.workload_id,
        "inputs":{"plan_sha256":sha256(plan_bytes),"observations_sha256":sha256(observation_bytes)},
        "support":{
            "matched_block_count":block_count,
            "trial_count":total_trials,
            "trials_per_block":total_trials / block_count,
            "independence_status":"not_established",
            "effective_sample_size_estimate":null
        },
        "event_qualification":event_reports,
        "observer_effect":{
            "baseline_condition":plan.baseline_condition,
            "threshold_fraction":plan.maximum_throughput_degradation,
            "condition_reports":throughput_reports,
            "overall_status":if throughput_gate_pass {"within_declared_thresholds"} else {"exceeds_declared_thresholds"}
        },
        "measurement_semantics":{
            "raw_counts_are_preserved":true,
            "multiplex_scaled_values_are_estimates":true,
            "multiplex_scaled_values_confirmatory":false,
            "scaling_does_not_correct_systematic_interference":true
        },
        "protocol_attestations":{
            "preregistered_randomization_seed":plan.randomization_seed,
            "randomized_order_observed_as_declared_trial_order":true,
            "randomization_authenticity":"caller_declared_not_independently_verified",
            "pmu_counter_capacity":plan.counter_capacity,
            "co_scheduled_event_count":plan.co_event_ids.len(),
            "overcommitted_event_count":plan.over_event_ids.len()
        },
        "limitations":[
            "This receipt qualifies the exact plan and supplied trial records only; it does not authenticate raw hardware, firmware, kernel or perf identities.",
            "Recorded randomization seed/order metadata are caller-declared; the auditor cannot prove the randomization procedure occurred.",
            "time_enabled/time_running scaling is an estimate and does not correct systematic event interference, workload phase drift or event semantic mismatch.",
            "Matched trial blocks do not by themselves establish independent statistical units; no confidence interval or effective sample size is claimed.",
            "A qualified acquisition condition is not a detector performance result, causal attack finding or hardware security certification."
        ]
    }))
}

fn read_input(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat {label} {}: {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|e| format!("cannot read {label}: {e}"))?;
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid {label} JSON: {e}"))?;
    Ok((bytes, value))
}

fn parse_args() -> Result<Args, String> {
    let mut plan = None;
    let mut observations = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let target = match flag.as_str() {
            "--plan" => &mut plan,
            "--observations" => &mut observations,
            "--receipt" => &mut receipt,
            _ => return Err(format!("unknown argument {flag}")),
        };
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    Ok(Args {
        plan: plan.ok_or("missing required argument --plan")?,
        observations: observations.ok_or("missing required argument --observations")?,
        receipt: receipt.ok_or("missing required argument --receipt")?,
    })
}

fn validate_receipt_location(receipt: &Path, inputs: &[&Path]) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent).map_err(|e| format!("receipt parent must exist: {e}"))?;
    for input in inputs {
        let canonical_input = fs::canonicalize(input)
            .map_err(|e| format!("cannot canonicalize input {}: {e}", input.display()))?;
        let input_parent = canonical_input.parent().ok_or("input has no parent directory")?;
        if canonical_parent.starts_with(input_parent) {
            return Err("receipt must be outside every input directory".to_owned());
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (plan_bytes, plan) = read_input(&args.plan, "plan")?;
    let (observation_bytes, observations) = read_input(&args.observations, "observations")?;
    validate_receipt_location(&args.receipt, &[args.plan.as_path(), args.observations.as_path()])?;
    let receipt = audit_values(&plan, &observations, &plan_bytes, &observation_bytes)?;
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|e| format!("cannot encode receipt: {e}"))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(&args.receipt)
        .map_err(|e| format!("cannot create receipt without overwrite: {e}"))?;
    file.write_all(&encoded).map_err(|e| format!("cannot write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e| format!("cannot finish receipt: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-018 PMU metrology audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Vec<u8>, Vec<u8>) {
        let plan_bytes = include_bytes!("../tests/fixtures/hardsec/pmu-metrology-v1/plan.json").to_vec();
        let observations_bytes = include_bytes!("../tests/fixtures/hardsec/pmu-metrology-v1/observations.json").to_vec();
        let plan = serde_json::from_slice(&plan_bytes).unwrap();
        let observations = serde_json::from_slice(&observations_bytes).unwrap();
        (plan, observations, plan_bytes, observations_bytes)
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn audit(plan: &Value, obs: &Value) -> Result<Value, String> {
        let plan_bytes = encode(plan);
        let obs_bytes = encode(obs);
        audit_values(plan, obs, &plan_bytes, &obs_bytes)
    }

    #[test]
    fn known_answer_qualifies_declared_simultaneous_events_and_preserves_estimate_semantics() {
        let (plan, obs, pb, ob) = fixture();
        let receipt = audit_values(&plan, &obs, &pb, &ob).unwrap();
        assert_eq!(receipt["result"], "qualified_for_named_acquisition_conditions");
        assert_eq!(receipt["support"]["matched_block_count"], 2);
        assert_eq!(receipt["support"]["trial_count"], 14);
        assert_eq!(receipt["measurement_semantics"]["multiplex_scaled_values_confirmatory"], false);
        assert_eq!(receipt["support"]["effective_sample_size_estimate"], Value::Null);
        assert_eq!(receipt["event_qualification"][0]["co_scheduled"]["status"], "confirmatory_eligible_under_declared_thresholds");
        assert_eq!(receipt["event_qualification"][0]["overcommitted"]["confirmatory_eligible"], false);
    }

    #[test]
    fn rejects_running_time_greater_than_enabled_time() {
        let (mut p, mut o, _, _) = fixture();
        o["trials"][0]["events"] = json!([{
            "event_id":"instret","encoding":"0x00000002","raw_count":10,
            "time_enabled_ns":10,"time_running_ns":11,"sample_count":0,"lost_samples":0
        }]);
        let error = audit(&p, &o).unwrap_err();
        assert!(error.contains("time_running_ns exceeds time_enabled_ns"));
        let _ = &mut p;
    }

    #[test]
    fn rejects_zero_running_time() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["events"] = json!([{
            "event_id":"instret","encoding":"0x00000002","raw_count":10,
            "time_enabled_ns":10,"time_running_ns":0,"sample_count":0,"lost_samples":0
        }]);
        assert!(audit(&p, &o).unwrap_err().contains("zero time_enabled/time_running"));
    }

    #[test]
    fn rejects_unknown_or_aliased_event_encoding() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["events"] = json!([{
            "event_id":"instret","encoding":"0xdeadbeef","raw_count":10,
            "time_enabled_ns":10,"time_running_ns":10,"sample_count":0,"lost_samples":0
        }]);
        assert!(audit(&p, &o).unwrap_err().contains("encoding does not match"));
    }

    #[test]
    fn rejects_inconsistent_fixed_work_units() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["work_units"] = json!(999);
        assert!(audit(&p, &o).unwrap_err().contains("work_units must equal"));
    }

    #[test]
    fn rejects_duplicate_trial_identity() {
        let (p, mut o, _, _) = fixture();
        o["trials"][1]["trial_id"] = o["trials"][0]["trial_id"].clone();
        assert!(audit(&p, &o).unwrap_err().contains("duplicate trial_id"));
    }

    #[test]
    fn rejects_unbalanced_condition_count() {
        let (p, mut o, _, _) = fixture();
        o["trials"].as_array_mut().unwrap().pop();
        assert!(audit(&p, &o).unwrap_err().contains("condition"));
    }

    #[test]
    fn rejects_event_aliases_in_plan() {
        let (mut p, o, _, _) = fixture();
        p["events"][1]["encoding"] = p["events"][0]["encoding"].clone();
        assert!(audit(&p, &o).unwrap_err().contains("duplicate event encoding"));
    }

    #[test]
    fn rejects_overcommitted_group_that_does_not_multiplex() {
        let (p, mut o, _, _) = fixture();
        for trial in o["trials"].as_array_mut().unwrap().iter_mut() {
            if trial["condition"] == "grouped_overcommitted" {
                for event in trial["events"].as_array_mut().unwrap() {
                    event["time_running_ns"] = event["time_enabled_ns"].clone();
                }
            }
        }
        assert!(audit(&p, &o).unwrap_err().contains("records no multiplexing"));
    }

    #[test]
    fn rejects_wrong_randomization_seed() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["randomization_seed"] = json!("other-seed");
        assert!(audit(&p, &o).unwrap_err().contains("randomization seed differs"));
    }

    #[test]
    fn rejects_mismatched_paired_block_ids_or_workload_identity() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["block_id"] = json!("orphan-block");
        assert!(audit(&p, &o).unwrap_err().contains("block"));
    }

    #[test]
    fn rejects_sample_loss_exceeding_samples() {
        let (p, mut o, _, _) = fixture();
        o["trials"][0]["events"] = json!([{
            "event_id":"instret","encoding":"0x00000002","raw_count":10,
            "time_enabled_ns":10,"time_running_ns":10,"sample_count":2,"lost_samples":3
        }]);
        assert!(audit(&p, &o).unwrap_err().contains("lost_samples exceeds sample_count"));
    }

    #[test]
    fn rejects_plan_and_observations_evidence_domain_mismatch() {
        let (p, mut o, _, _) = fixture();
        o["evidence_domain"] = json!("physical_hardware");
        assert!(audit(&p, &o).unwrap_err().contains("evidence_domain must match"));
    }
}
