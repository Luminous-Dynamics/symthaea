//! HARDSEC-014 telemetry trace and window lineage auditor.
//!
//! Audits caller-declared trace/window derivation identities, replay ancestry,
//! split-group disjointness, offsets/stride where supported, and repeated
//! window-content support. It reports raw window counts beside independent
//! program/execution/family group counts. It does not estimate ESS, prove
//! independence, authenticate source bytes, or evaluate detector performance.
//!
//! Usage:
//!   hardsec_trace_lineage_audit --manifest lineage.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "declared_trace_window_lineage_and_group_consistency_only";

#[derive(Debug, Clone)]
struct Window {
    window_id: String,
    sample_index: u64,
    start_offset: Option<u64>,
    end_offset: Option<u64>,
    window_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Original,
    Replayed,
    RepeatedIteration,
    Unknown,
}

impl Origin {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "original" => Ok(Self::Original),
            "replayed" => Ok(Self::Replayed),
            "repeated_iteration" => Ok(Self::RepeatedIteration),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("unsupported trace origin {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Replayed => "replayed",
            Self::RepeatedIteration => "repeated_iteration",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
struct Trace {
    trace_id: String,
    source_trace_sha256: String,
    program_id: String,
    binary_sha256: String,
    execution_id: String,
    device_config_id: String,
    attack_family_id: String,
    adaptation_family_id: String,
    split: String,
    origin: Origin,
    parent_trace_id: Option<String>,
    iteration_id: u64,
    windows: Vec<Window>,
}

#[derive(Debug, Clone)]
struct Policy {
    offset_unit: String,
    window_size: Option<u64>,
    stride: Option<u64>,
}

#[derive(Debug, Clone)]
struct Args {
    manifest: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
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

fn optional_string(value: &Value, key: &str, context: &str) -> Result<Option<String>, String> {
    let field = value.get(key).ok_or_else(|| format!("{context}.{key} is required"))?;
    if field.is_null() {
        return Ok(None);
    }
    let text = field.as_str().ok_or_else(|| format!("{context}.{key} must be string or null"))?;
    nonempty(text, &format!("{context}.{key}"))?;
    Ok(Some(text.to_owned()))
}

fn optional_u64(value: &Value, key: &str, context: &str) -> Result<Option<u64>, String> {
    let field = value.get(key).ok_or_else(|| format!("{context}.{key} is required"))?;
    if field.is_null() {
        return Ok(None);
    }
    field.as_u64().map(Some)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer or null"))
}

fn u64_field(value: &Value, key: &str, context: &str) -> Result<u64, String> {
    value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn parse_policy(value: &Value) -> Result<Policy, String> {
    let context = "sampling_policy";
    let offset_unit = string_field(value, "offset_unit", context)?.to_owned();
    if !matches!(offset_unit.as_str(), "committed_instructions" | "record_index" | "byte_offset" | "unknown") {
        return Err(format!("unsupported sampling_policy.offset_unit {offset_unit}"));
    }
    let window_size = optional_u64(value, "window_size", context)?;
    let stride = optional_u64(value, "stride", context)?;
    if offset_unit == "unknown" {
        if window_size.is_some() || stride.is_some() {
            return Err("unknown offset_unit requires null window_size and stride".to_owned());
        }
    } else if window_size.unwrap_or(0) == 0 || stride.unwrap_or(0) == 0 {
        return Err("known offset_unit requires positive window_size and stride".to_owned());
    }
    Ok(Policy { offset_unit, window_size, stride })
}

fn parse_windows(value: &Value, context: &str) -> Result<Vec<Window>, String> {
    let values = value.get("windows").and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.windows must be an array"))?;
    if values.is_empty() {
        return Err(format!("{context}.windows must not be empty"));
    }
    let mut seen = BTreeSet::new();
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let ctx = format!("{context}.windows[{index}]");
        let window_id = string_field(item, "window_id", &ctx)?.to_owned();
        if !seen.insert(window_id.clone()) {
            return Err(format!("duplicate window_id {window_id} within {context}"));
        }
        let sample_index = u64_field(item, "sample_index", &ctx)?;
        let start_offset = optional_u64(item, "start_offset", &ctx)?;
        let end_offset = optional_u64(item, "end_offset", &ctx)?;
        let window_sha256 = string_field(item, "window_sha256", &ctx)?.to_owned();
        valid_sha(&window_sha256, &format!("{ctx}.window_sha256"))?;
        if start_offset.is_some() != end_offset.is_some() {
            return Err(format!("{ctx}: start_offset and end_offset must both be known or both be null"));
        }
        if let (Some(start), Some(end)) = (start_offset, end_offset) {
            if end <= start {
                return Err(format!("{ctx}: end_offset must be greater than start_offset"));
            }
        }
        output.push(Window { window_id, sample_index, start_offset, end_offset, window_sha256 });
    }
    Ok(output)
}

fn parse_traces(value: &Value) -> Result<Vec<Trace>, String> {
    let values = value.get("traces").and_then(Value::as_array)
        .ok_or_else(|| "traces must be an array".to_owned())?;
    if values.is_empty() {
        return Err("traces must not be empty".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut window_ids = BTreeSet::new();
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let ctx = format!("traces[{index}]");
        let trace_id = string_field(item, "trace_id", &ctx)?.to_owned();
        if !trace_ids.insert(trace_id.clone()) {
            return Err(format!("duplicate trace_id {trace_id}"));
        }
        let source_trace_sha256 = string_field(item, "source_trace_sha256", &ctx)?.to_owned();
        valid_sha(&source_trace_sha256, &format!("{ctx}.source_trace_sha256"))?;
        let program_id = string_field(item, "program_id", &ctx)?.to_owned();
        let binary_sha256 = string_field(item, "binary_sha256", &ctx)?.to_owned();
        valid_sha(&binary_sha256, &format!("{ctx}.binary_sha256"))?;
        let execution_id = string_field(item, "execution_id", &ctx)?.to_owned();
        let device_config_id = string_field(item, "device_config_id", &ctx)?.to_owned();
        let attack_family_id = string_field(item, "attack_family_id", &ctx)?.to_owned();
        let adaptation_family_id = string_field(item, "adaptation_family_id", &ctx)?.to_owned();
        let split = string_field(item, "split", &ctx)?.to_owned();
        if !matches!(split.as_str(), "train" | "validation" | "test") {
            return Err(format!("{ctx}.split must be train, validation or test"));
        }
        let origin = Origin::parse(string_field(item, "origin", &ctx)?)?;
        let parent_trace_id = optional_string(item, "parent_trace_id", &ctx)?;
        let iteration_id = u64_field(item, "iteration_id", &ctx)?;
        let windows = parse_windows(item, &ctx)?;
        if windows.windows(2).any(|pair| pair[1].sample_index <= pair[0].sample_index) {
            return Err(format!("{ctx}. sample_index must strictly increase"));
        }
        for window in &windows {
            if !window_ids.insert(window.window_id.clone()) {
                return Err(format!("duplicate global window_id {}", window.window_id));
            }
        }
        output.push(Trace {
            trace_id, source_trace_sha256, program_id, binary_sha256, execution_id,
            device_config_id, attack_family_id, adaptation_family_id, split,
            origin, parent_trace_id, iteration_id, windows,
        });
    }
    Ok(output)
}

fn validate_offsets(traces: &[Trace], policy: &Policy) -> Result<Value, String> {
    let mut timed_trace_count = 0usize;
    let mut untimed_trace_count = 0usize;
    let mut total_timed_windows = 0usize;
    let mut overlap_pair_count = 0usize;
    let mut validated_stride_windows = 0usize;

    for trace in traces {
        let any_offsets = trace.windows.iter().any(|w| w.start_offset.is_some());
        let all_offsets = trace.windows.iter().all(|w| w.start_offset.is_some() && w.end_offset.is_some());
        if policy.offset_unit == "unknown" {
            if any_offsets {
                return Err(format!("trace {} supplies offsets while policy.offset_unit is unknown", trace.trace_id));
            }
            untimed_trace_count += 1;
            continue;
        }
        if !any_offsets {
            if policy.offset_unit != "unknown" {
                return Err(format!("trace {} has no offsets but policy declares a known offset_unit; use unknown when offsets are unsupported", trace.trace_id));
            }
            untimed_trace_count += 1;
            continue;
        }
        if !all_offsets {
            return Err(format!("trace {} mixes known and unknown window offsets", trace.trace_id));
        }
        timed_trace_count += 1;
        total_timed_windows += trace.windows.len();

        for pair in trace.windows.windows(2) {
            if pair[1].sample_index <= pair[0].sample_index {
                return Err(format!("trace {} sample_index must strictly increase", trace.trace_id));
            }
            let previous_start = pair[0].start_offset.ok_or("offset check invariant failed")?;
            let next_start = pair[1].start_offset.ok_or("offset check invariant failed")?;
            let previous_end = pair[0].end_offset.ok_or("offset check invariant failed")?;
            let next_end = pair[1].end_offset.ok_or("offset check invariant failed")?;
            let window_size = policy.window_size.ok_or("known window_size missing")?;
            let stride = policy.stride.ok_or("known stride missing")?;
            if previous_end - previous_start != window_size || next_end - next_start != window_size {
                return Err(format!("trace {} window length does not match declared window_size", trace.trace_id));
            }
            if next_start <= previous_start {
                return Err(format!("trace {} start offsets must strictly increase", trace.trace_id));
            }
            if next_start - previous_start != stride {
                return Err(format!("trace {} offset difference does not match declared stride", trace.trace_id));
            }
            if next_start < previous_end {
                overlap_pair_count += 1;
            }
            validated_stride_windows += 1;
        }
        if trace.windows.len() == 1 {
            let start = trace.windows[0].start_offset.ok_or("offset check invariant failed")?;
            let end = trace.windows[0].end_offset.ok_or("offset check invariant failed")?;
            if end - start != policy.window_size.ok_or("known window_size missing")? {
                return Err(format!("trace {} window length does not match declared window_size", trace.trace_id));
            }
        }
    }
    Ok(json!({
        "offset_unit": policy.offset_unit,
        "timed_trace_count": timed_trace_count,
        "untimed_trace_count": untimed_trace_count,
        "timed_window_count": total_timed_windows,
        "validated_adjacent_stride_pairs": validated_stride_windows,
        "overlapping_adjacent_window_pairs": overlap_pair_count,
        "ess_estimate": null,
        "ess_unavailable_reason": "this lineage auditor does not estimate effective sample size"
    }))
}

fn validate_parentage(traces: &[Trace]) -> Result<(usize, usize, usize), String> {
    let by_id: BTreeMap<&str, &Trace> = traces.iter().map(|t| (t.trace_id.as_str(), t)).collect();
    let mut repeated_iterations = 0usize;
    let mut replayed_traces = 0usize;
    let mut unknown_origin_traces = 0usize;
    let mut seen_parent_iterations = BTreeSet::new();

    for trace in traces {
        match trace.origin {
            Origin::Original => {
                if trace.parent_trace_id.is_some() || trace.iteration_id != 0 {
                    return Err(format!("original trace {} must have null parent and iteration_id=0", trace.trace_id));
                }
            }
            Origin::RepeatedIteration => {
                repeated_iterations += 1;
                if trace.iteration_id == 0 {
                    return Err(format!("repeated iteration {} must have iteration_id > 0", trace.trace_id));
                }
                let parent_id = trace.parent_trace_id.as_deref()
                    .ok_or_else(|| format!("repeated iteration {} requires parent_trace_id", trace.trace_id))?;
                let parent = by_id.get(parent_id)
                    .ok_or_else(|| format!("repeated iteration {} references missing parent {}", trace.trace_id, parent_id))?;
                if parent.origin != Origin::Original {
                    return Err(format!("repeated iteration {} parent must be original", trace.trace_id));
                }
                if parent.program_id != trace.program_id
                    || parent.binary_sha256 != trace.binary_sha256
                    || parent.execution_id != trace.execution_id
                    || parent.device_config_id != trace.device_config_id
                    || parent.attack_family_id != trace.attack_family_id
                    || parent.adaptation_family_id != trace.adaptation_family_id
                    || parent.split != trace.split
                {
                    return Err(format!("repeated iteration {} does not match its parent lineage", trace.trace_id));
                }
                if !seen_parent_iterations.insert((parent_id.to_owned(), trace.iteration_id)) {
                    return Err(format!("duplicate iteration_id {} for parent {}", trace.iteration_id, parent_id));
                }
            }
            Origin::Replayed => {
                replayed_traces += 1;
                if trace.iteration_id == 0 {
                    return Err(format!("replayed trace {} must have iteration_id > 0", trace.trace_id));
                }
                let parent_id = trace.parent_trace_id.as_deref()
                    .ok_or_else(|| format!("replayed trace {} requires parent_trace_id", trace.trace_id))?;
                let parent = by_id.get(parent_id)
                    .ok_or_else(|| format!("replayed trace {} references missing parent {}", trace.trace_id, parent_id))?;
                if parent.origin != Origin::Original {
                    return Err(format!("replayed trace {} parent must be original", trace.trace_id));
                }
                if parent.program_id != trace.program_id
                    || parent.binary_sha256 != trace.binary_sha256
                    || parent.device_config_id != trace.device_config_id
                    || parent.attack_family_id != trace.attack_family_id
                    || parent.adaptation_family_id != trace.adaptation_family_id
                    || parent.split != trace.split
                {
                    return Err(format!("replayed trace {} does not match its parent workload lineage", trace.trace_id));
                }
                if parent.execution_id == trace.execution_id {
                    return Err(format!("replayed trace {} must have a distinct execution_id", trace.trace_id));
                }
                if !seen_parent_iterations.insert((parent_id.to_owned(), trace.iteration_id)) {
                    return Err(format!("duplicate iteration_id {} for parent {}", trace.iteration_id, parent_id));
                }
            }
            Origin::Unknown => {
                unknown_origin_traces += 1;
                if trace.parent_trace_id.is_some() || trace.iteration_id != 0 {
                    return Err(format!("unknown-origin trace {} must not claim parent or iteration", trace.trace_id));
                }
            }
        }
    }
    Ok((repeated_iterations, replayed_traces, unknown_origin_traces))
}

fn validate_holdouts(traces: &[Trace], axes: &[String]) -> Result<Vec<Value>, String> {
    let mut reports = Vec::new();
    for axis in axes {
        let mut groups: BTreeMap<String, String> = BTreeMap::new();
        for trace in traces {
            let group = match axis.as_str() {
                "program" => trace.program_id.as_str(),
                "execution" => trace.execution_id.as_str(),
                "device_config" => trace.device_config_id.as_str(),
                "attack_family" => trace.attack_family_id.as_str(),
                "adaptation_family" => trace.adaptation_family_id.as_str(),
                _ => return Err(format!("unsupported holdout axis {axis}")),
            };
            if matches!(axis.as_str(), "attack_family" | "adaptation_family") && group == "none" {
                continue;
            }
            if let Some(previous) = groups.insert(group.to_owned(), trace.split.clone()) {
                if previous != trace.split {
                    return Err(format!("{} holdout group {} crosses {} and {}", axis, group, previous, trace.split));
                }
            }
        }
        reports.push(json!({
            "axis": axis,
            "unique_non_sentinel_group_count": groups.len(),
            "disjoint_across_splits": true
        }));
    }
    Ok(reports)
}

fn audit_values(manifest: &Value, manifest_bytes: &[u8]) -> Result<Value, String> {
    if string_field(manifest, "schema_version", "manifest")? != "hardsec-telemetry-trace-lineage-v1" {
        return Err("unsupported manifest schema_version".to_owned());
    }
    let dataset_id = string_field(manifest, "dataset_id", "manifest")?;
    let corpus_sha = string_field(manifest, "source_corpus_sha256", "manifest")?;
    valid_sha(corpus_sha, "manifest.source_corpus_sha256")?;
    let policy_value = manifest.get("sampling_policy").ok_or("sampling_policy is required")?;
    let policy = parse_policy(policy_value)?;
    let axis_values = manifest.get("holdout_axes").and_then(Value::as_array)
        .ok_or_else(|| "holdout_axes must be an array".to_owned())?;
    let allowed = ["program", "execution", "device_config", "attack_family", "adaptation_family"];
    let mut axes = BTreeSet::new();
    for item in axis_values {
        let axis = item.as_str().ok_or("holdout_axes items must be strings")?;
        if !allowed.contains(&axis) {
            return Err(format!("unsupported holdout axis {axis}"));
        }
        if !axes.insert(axis.to_owned()) {
            return Err(format!("duplicate holdout axis {axis}"));
        }
    }
    for required in ["program", "execution", "attack_family", "adaptation_family"] {
        if !axes.contains(required) {
            return Err(format!("holdout_axes must include {required}"));
        }
    }
    let traces = parse_traces(manifest)?;
    let (repeated_iterations, replayed_traces, unknown_origin_traces) = validate_parentage(&traces)?;
    let holdout_reports = validate_holdouts(&traces, &axes.into_iter().collect::<Vec<_>>())?;
    let offset_report = validate_offsets(&traces, &policy)?;

    let mut window_hashes: BTreeMap<String, usize> = BTreeMap::new();
    let mut window_count = 0usize;
    let mut source_hashes = BTreeSet::new();
    let mut program_ids = BTreeSet::new();
    let mut execution_ids = BTreeSet::new();
    let mut device_ids = BTreeSet::new();
    let mut attack_families = BTreeSet::new();
    let mut adaptation_families = BTreeSet::new();
    let mut trace_origin_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut split_window_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut original_window_count = 0usize;
    let mut repeated_window_count = 0usize;
    let mut replayed_window_count = 0usize;
    let mut unknown_window_count = 0usize;
    let mut trace_program_split: BTreeSet<(String, String)> = BTreeSet::new();

    for trace in &traces {
        source_hashes.insert(trace.source_trace_sha256.clone());
        program_ids.insert(trace.program_id.clone());
        execution_ids.insert(trace.execution_id.clone());
        device_ids.insert(trace.device_config_id.clone());
        if trace.attack_family_id != "none" {
            attack_families.insert(trace.attack_family_id.clone());
        }
        if trace.adaptation_family_id != "none" {
            adaptation_families.insert(trace.adaptation_family_id.clone());
        }
        *trace_origin_counts.entry(trace.origin.as_str().to_owned()).or_default() += 1;
        *split_window_counts.entry(trace.split.clone()).or_default() += trace.windows.len();
        trace_program_split.insert((trace.program_id.clone(), trace.split.clone()));

        match trace.origin {
            Origin::Original => original_window_count += trace.windows.len(),
            Origin::RepeatedIteration => repeated_window_count += trace.windows.len(),
            Origin::Replayed => replayed_window_count += trace.windows.len(),
            Origin::Unknown => unknown_window_count += trace.windows.len(),
        }
        for window in &trace.windows {
            window_count += 1;
            *window_hashes.entry(window.window_sha256.clone()).or_default() += 1;
        }
    }
    let duplicate_content_groups: Vec<Value> = window_hashes.iter()
        .filter(|(_, count)| **count > 1)
        .map(|(sha, count)| json!({"window_sha256": sha, "occurrences": count}))
        .collect();
    let duplicate_content_occurrences = window_hashes.values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();
    let duplicate_content_hash_group_count = duplicate_content_groups.len();
    let original_trace_count = traces.iter().filter(|t| t.origin == Origin::Original).count();

    Ok(json!({
        "schema_version":"hardsec-telemetry-trace-lineage-receipt-v1",
        "result":"audited",
        "claim_scope":CLAIM_SCOPE,
        "dataset_id":dataset_id,
        "inputs":{"manifest_sha256":sha256(manifest_bytes),"source_corpus_sha256":corpus_sha},
        "trace_lineage":{
            "trace_record_count":traces.len(),
            "original_trace_count":original_trace_count,
            "repeated_iteration_trace_count":repeated_iterations,
            "replayed_trace_count":replayed_traces,
            "unknown_origin_trace_count":unknown_origin_traces,
            "unique_source_trace_sha256_count":source_hashes.len(),
            "trace_origin_counts":trace_origin_counts
        },
        "window_support":{
            "raw_window_count":window_count,
            "original_window_count":original_window_count,
            "repeated_iteration_window_count":repeated_window_count,
            "replayed_window_count":replayed_window_count,
            "unknown_origin_window_count":unknown_window_count,
            "duplicate_content_hash_groups":duplicate_content_groups,
            "duplicate_content_hash_group_count":duplicate_content_hash_group_count,
            "duplicate_content_extra_occurrences":duplicate_content_occurrences,
            "split_window_counts":split_window_counts
        },
        "independent_group_support":{
            "unique_program_count":program_ids.len(),
            "unique_execution_id_count":execution_ids.len(),
            "unique_device_config_count":device_ids.len(),
            "unique_attack_family_count_excluding_none":attack_families.len(),
            "unique_adaptation_family_count_excluding_none":adaptation_families.len(),
            "program_split_pairs":trace_program_split.len(),
            "effective_sample_size_estimate":null,
            "effective_sample_size_status":"not_estimated",
            "reason":"lineage counts do not establish temporal or cluster independence"
        },
        "sampling_policy":{
            "offset_unit":policy.offset_unit,
            "window_size":policy.window_size,
            "stride":policy.stride,
            "offset_validation":offset_report
        },
        "holdout_audit":holdout_reports,
        "limitations":[
            "This receipt validates caller-declared trace/window lineage consistency only; it does not authenticate raw files or identities.",
            "Window SHA-256 duplicates are reported but not automatically treated as independent evidence or as errors.",
            "Program/execution group counts are caller-declared; repeated iterations do not create new program IDs.",
            "No effective sample size is estimated; dependence diagnostics require ordered raw traces and a separately validated statistical method.",
            "The auditor does not validate detector predictions, attack success, ground truth, or any physical/simulator result."
        ]
    }))
}

fn read_input(path: &Path) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat manifest {}: {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("manifest must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(path).map_err(|e| format!("cannot read manifest: {e}"))?;
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid manifest JSON: {e}"))?;
    Ok((bytes, value))
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !matches!(flag.as_str(), "--manifest" | "--receipt") {
            return Err(format!("unknown argument {flag}"));
        }
        let value = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if values.insert(flag.clone(), PathBuf::from(value)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let take = |key: &str| values.get(key).cloned().ok_or_else(|| format!("missing required argument {key}"));
    Ok(Args { manifest:take("--manifest")?, receipt:take("--receipt")? })
}

fn validate_receipt_location(receipt: &Path, manifest: &Path) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent).map_err(|e| format!("receipt parent must exist: {e}"))?;
    let canonical_manifest = fs::canonicalize(manifest).map_err(|e| format!("cannot canonicalize manifest: {e}"))?;
    let manifest_parent = canonical_manifest.parent().ok_or("manifest has no parent directory")?;
    if canonical_parent.starts_with(manifest_parent) {
        return Err("receipt must be outside the manifest directory".to_owned());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (bytes, manifest) = read_input(&args.manifest)?;
    validate_receipt_location(&args.receipt, &args.manifest)?;
    let receipt = audit_values(&manifest, &bytes)?;
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|e| format!("cannot encode receipt: {e}"))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(&args.receipt)
        .map_err(|e| format!("cannot create no-overwrite receipt: {e}"))?;
    file.write_all(&encoded).map_err(|e| format!("cannot write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e| format!("cannot finish receipt: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-014 trace lineage audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Vec<u8>) {
        let manifest = json!({
            "schema_version":"hardsec-telemetry-trace-lineage-v1",
            "dataset_id":"hs014-synthetic-known-answer-v1",
            "source_corpus_sha256":"a".repeat(64),
            "sampling_policy":{"offset_unit":"committed_instructions","window_size":10000,"stride":10000},
            "holdout_axes":["program","execution","attack_family","adaptation_family"],
            "traces":[
                {"trace_id":"trace-train-original","source_trace_sha256":"b".repeat(64),"program_id":"program-train","binary_sha256":"c".repeat(64),"execution_id":"execution-train","device_config_id":"device-sim","attack_family_id":"family-train","adaptation_family_id":"none","split":"train","origin":"original","parent_trace_id":null,"iteration_id":0,
                 "windows":[{"window_id":"w-train-0","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"1".repeat(64)},{"window_id":"w-train-1","sample_index":1,"start_offset":10000,"end_offset":20000,"window_sha256":"2".repeat(64)}]},
                {"trace_id":"trace-train-repeat","source_trace_sha256":"b".repeat(64),"program_id":"program-train","binary_sha256":"c".repeat(64),"execution_id":"execution-train","device_config_id":"device-sim","attack_family_id":"family-train","adaptation_family_id":"none","split":"train","origin":"repeated_iteration","parent_trace_id":"trace-train-original","iteration_id":1,
                 "windows":[{"window_id":"w-repeat-0","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"1".repeat(64)},{"window_id":"w-repeat-1","sample_index":1,"start_offset":10000,"end_offset":20000,"window_sha256":"2".repeat(64)}]},
                {"trace_id":"trace-validation","source_trace_sha256":"d".repeat(64),"program_id":"program-validation","binary_sha256":"e".repeat(64),"execution_id":"execution-validation","device_config_id":"device-sim","attack_family_id":"family-validation","adaptation_family_id":"none","split":"validation","origin":"original","parent_trace_id":null,"iteration_id":0,
                 "windows":[{"window_id":"w-validation-0","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"3".repeat(64)}]},
                {"trace_id":"trace-test-attack","source_trace_sha256":"f".repeat(64),"program_id":"program-test","binary_sha256":"0".repeat(64),"execution_id":"execution-test","device_config_id":"device-sim","attack_family_id":"family-test","adaptation_family_id":"timing","split":"test","origin":"original","parent_trace_id":null,"iteration_id":0,
                 "windows":[{"window_id":"w-test-attack-0","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"4".repeat(64)}]},
                {"trace_id":"trace-test-benign","source_trace_sha256":"9".repeat(64),"program_id":"program-benign","binary_sha256":"8".repeat(64),"execution_id":"execution-benign","device_config_id":"device-sim","attack_family_id":"none","adaptation_family_id":"none","split":"test","origin":"original","parent_trace_id":null,"iteration_id":0,
                 "windows":[{"window_id":"w-test-benign-0","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"5".repeat(64)}]}
            ]
        });
        let bytes = serde_json::to_vec(&manifest).unwrap();
        (manifest, bytes)
    }

    #[test]
    fn known_answer_counts_windows_and_independent_groups_separately() {
        let (m,b) = fixture();
        let r = audit_values(&m,&b).unwrap();
        assert_eq!(r["trace_lineage"]["trace_record_count"], 5);
        assert_eq!(r["trace_lineage"]["original_trace_count"], 4);
        assert_eq!(r["trace_lineage"]["repeated_iteration_trace_count"], 1);
        assert_eq!(r["window_support"]["raw_window_count"], 7);
        assert_eq!(r["window_support"]["repeated_iteration_window_count"], 2);
        assert_eq!(r["independent_group_support"]["unique_program_count"], 4);
        assert_eq!(r["independent_group_support"]["unique_execution_id_count"], 4);
        assert_eq!(r["window_support"]["duplicate_content_hash_group_count"], 2);
        assert_eq!(r["window_support"]["duplicate_content_extra_occurrences"], 2);
        assert_eq!(r["independent_group_support"]["effective_sample_size_estimate"], Value::Null);
    }

    #[test]
    fn replay_requires_distinct_execution_id_from_parent() {
        let (mut m,_) = fixture();
        m["traces"].as_array_mut().unwrap().push(json!({
            "trace_id":"trace-replay","source_trace_sha256":"6".repeat(64),
            "program_id":"program-train","binary_sha256":"c".repeat(64),
            "execution_id":"execution-train","device_config_id":"device-sim",
            "attack_family_id":"family-train","adaptation_family_id":"none",
            "split":"train","origin":"replayed","parent_trace_id":"trace-train-original","iteration_id":2,
            "windows":[{"window_id":"w-replay","sample_index":0,"start_offset":0,"end_offset":10000,"window_sha256":"6".repeat(64)}]
        }));
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("distinct execution_id"));
    }

    #[test]
    fn known_offset_policy_rejects_missing_offsets() {
        let (mut m,_) = fixture();
        m["traces"][2]["windows"][0]["start_offset"] = Value::Null;
        m["traces"][2]["windows"][0]["end_offset"] = Value::Null;
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("no offsets but policy declares a known offset_unit"));
    }

    #[test]
    fn rejects_program_holdout_leakage() {
        let (mut m,_) = fixture();
        m["traces"][2]["program_id"] = json!("program-train");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("program holdout"));
    }

    #[test]
    fn rejects_execution_holdout_leakage() {
        let (mut m,_) = fixture();
        m["traces"][2]["execution_id"] = json!("execution-train");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("execution holdout"));
    }

    #[test]
    fn rejects_attack_family_holdout_leakage() {
        let (mut m,_) = fixture();
        m["traces"][2]["attack_family_id"] = json!("family-train");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("attack_family holdout"));
    }

    #[test]
    fn rejects_adaptation_family_holdout_leakage() {
        let (mut m,_) = fixture();
        m["traces"][2]["adaptation_family_id"] = json!("timing");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("adaptation_family holdout"));
    }

    #[test]
    fn rejects_repeated_iteration_without_parent() {
        let (mut m,_) = fixture();
        m["traces"][1]["parent_trace_id"] = Value::Null;
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("requires parent_trace_id"));
    }

    #[test]
    fn rejects_repeated_iteration_with_different_program() {
        let (mut m,_) = fixture();
        m["traces"][1]["program_id"] = json!("other-program");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("does not match its parent lineage"));
    }

    #[test]
    fn rejects_nonmatching_stride() {
        let (mut m,_) = fixture();
        m["traces"][0]["windows"][1]["start_offset"] = json!(15000);
        m["traces"][0]["windows"][1]["end_offset"] = json!(25000);
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("does not match declared stride"));
    }

    #[test]
    fn rejects_wrong_window_size() {
        let (mut m,_) = fixture();
        m["traces"][0]["windows"][0]["end_offset"] = json!(9999);
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("window length does not match"));
    }

    #[test]
    fn rejects_mixed_known_and_unknown_offsets() {
        let (mut m,_) = fixture();
        m["traces"][0]["windows"][1]["start_offset"] = Value::Null;
        m["traces"][0]["windows"][1]["end_offset"] = Value::Null;
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("mixes known and unknown"));
    }

    #[test]
    fn rejects_duplicate_global_window_id() {
        let (mut m,_) = fixture();
        m["traces"][1]["windows"][0]["window_id"] = json!("w-train-0");
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("duplicate global window_id"));
    }

    #[test]
    fn unknown_offset_policy_does_not_allow_fabricated_offsets() {
        let (mut m,_) = fixture();
        m["sampling_policy"]["offset_unit"] = json!("unknown");
        m["sampling_policy"]["window_size"] = Value::Null;
        m["sampling_policy"]["stride"] = Value::Null;
        let b = serde_json::to_vec(&m).unwrap();
        assert!(audit_values(&m,&b).unwrap_err().contains("supplies offsets while policy.offset_unit is unknown"));
    }
}
