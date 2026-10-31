//! HARDSEC-016 case-to-trace/window lineage linker.
//!
//! Binds each adaptive-evasion case to explicit trace and window identities,
//! reconciles the supplied HARDSEC-014 receipt with exact trace-manifest bytes,
//! and checks that all linked metadata agree. This is not source authentication
//! and does not estimate effective sample size or establish independence.
//!
//! Usage:
//!   hardsec_case_trace_link_audit --adaptive-manifest cases.json \
//!     --trace-manifest lineage.json --trace-receipt trace-receipt.json \
//!     --case-links links.json --receipt link-receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const ADAPTIVE_SCHEMA: &str = "hardsec-adaptive-case-manifest-v1";
const TRACE_SCHEMA: &str = "hardsec-telemetry-trace-lineage-v1";
const TRACE_RECEIPT_SCHEMA: &str = "hardsec-telemetry-trace-lineage-receipt-v1";
const LINKS_SCHEMA: &str = "hardsec-adaptive-case-trace-links-v1";
const TRACE_CLAIM_SCOPE: &str = "declared_trace_window_lineage_and_group_consistency_only";

#[derive(Debug, Clone)]
struct CaseMeta {
    split: String,
    program: String,
    execution: String,
    device: String,
    attack_family: String,
    adaptation_family: String,
}

#[derive(Debug, Clone)]
struct WindowMeta {
    sha256: String,
}

#[derive(Debug, Clone)]
struct TraceMeta {
    program: String,
    execution: String,
    device: String,
    attack_family: String,
    adaptation_family: String,
    split: String,
    origin: String,
    windows: BTreeMap<String, WindowMeta>,
}

#[derive(Debug, Clone, Default)]
struct TraceSummary {
    trace_count: usize,
    source_trace_count: usize,
    origin_trace_counts: BTreeMap<String, usize>,
    raw_window_count: usize,
    origin_window_counts: BTreeMap<String, usize>,
    duplicate_hash_groups: Vec<Value>,
    duplicate_hash_group_count: usize,
    duplicate_extra_occurrences: usize,
    split_window_counts: BTreeMap<String, usize>,
    program_count: usize,
    execution_count: usize,
    device_count: usize,
    attack_family_count: usize,
    adaptation_family_count: usize,
    program_split_pairs: usize,
}

#[derive(Debug, Clone)]
struct Args {
    adaptive_manifest: PathBuf,
    trace_manifest: PathBuf,
    trace_receipt: PathBuf,
    case_links: PathBuf,
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
    let field = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))?;
    nonempty(field, &format!("{context}.{key}"))?;
    Ok(field)
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn count_at(value: &Value, path: &[&str], expected: usize, context: &str) -> Result<(), String> {
    let mut current = value;
    for key in path {
        current = current.get(*key).ok_or_else(|| format!("{context}.{key} is missing"))?;
    }
    let actual = current.as_u64().ok_or_else(|| format!("{context} must be a non-negative integer"))?;
    if actual != expected as u64 {
        return Err(format!("{context} is {actual}; trace manifest requires {expected}"));
    }
    Ok(())
}

fn required_axes(value: &Value, context: &str) -> Result<BTreeSet<String>, String> {
    let values = value.get("holdout_axes").and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.holdout_axes must be an array"))?;
    let allowed = ["program", "execution", "device_config", "attack_family", "adaptation_family"];
    let mut axes = BTreeSet::new();
    for item in values {
        let axis = item.as_str().ok_or_else(|| format!("{context}.holdout_axes entries must be strings"))?;
        if !allowed.contains(&axis) {
            return Err(format!("{context}.holdout_axes contains unsupported axis {axis}"));
        }
        if !axes.insert(axis.to_owned()) {
            return Err(format!("{context}.holdout_axes contains duplicate axis {axis}"));
        }
    }
    for required in ["program", "execution", "attack_family", "adaptation_family"] {
        if !axes.contains(required) {
            return Err(format!("{context}.holdout_axes must include {required}"));
        }
    }
    Ok(axes)
}

fn parse_adaptive_cases(value: &Value) -> Result<(String, BTreeSet<String>, BTreeMap<String, CaseMeta>), String> {
    if string_field(value, "schema_version", "adaptive_manifest")? != ADAPTIVE_SCHEMA {
        return Err("unsupported adaptive manifest schema_version".to_owned());
    }
    let campaign_id = string_field(value, "campaign_id", "adaptive_manifest")?.to_owned();
    let axes = required_axes(value, "adaptive_manifest")?;
    let records = value.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "adaptive_manifest.cases must be an array".to_owned())?;
    if records.is_empty() {
        return Err("adaptive_manifest.cases must not be empty".to_owned());
    }
    let mut cases = BTreeMap::new();
    for (index, item) in records.iter().enumerate() {
        let context = format!("adaptive_manifest.cases[{index}]");
        let case_id = string_field(item, "case_id", &context)?.to_owned();
        let split = string_field(item, "split", &context)?.to_owned();
        if !matches!(split.as_str(), "train" | "validation" | "test") {
            return Err(format!("{context}.split is unsupported"));
        }
        let meta = CaseMeta {
            split,
            program: string_field(item, "program_group_id", &context)?.to_owned(),
            execution: string_field(item, "execution_group_id", &context)?.to_owned(),
            device: string_field(item, "device_config_id", &context)?.to_owned(),
            attack_family: string_field(item, "attack_family_id", &context)?.to_owned(),
            adaptation_family: string_field(item, "adaptation_family_id", &context)?.to_owned(),
        };
        if cases.insert(case_id.clone(), meta).is_some() {
            return Err(format!("duplicate adaptive case_id {case_id}"));
        }
    }
    Ok((campaign_id, axes, cases))
}

fn parse_traces(value: &Value) -> Result<(String, String, BTreeMap<String, TraceMeta>, TraceSummary), String> {
    if string_field(value, "schema_version", "trace_manifest")? != TRACE_SCHEMA {
        return Err("unsupported trace manifest schema_version".to_owned());
    }
    let dataset_id = string_field(value, "dataset_id", "trace_manifest")?.to_owned();
    let corpus_sha = string_field(value, "source_corpus_sha256", "trace_manifest")?.to_owned();
    valid_sha(&corpus_sha, "trace_manifest.source_corpus_sha256")?;
    let axes = required_axes(value, "trace_manifest")?;
    let _policy = value.get("sampling_policy").and_then(Value::as_object)
        .ok_or_else(|| "trace_manifest.sampling_policy must be an object".to_owned())?;
    let records = value.get("traces").and_then(Value::as_array)
        .ok_or_else(|| "trace_manifest.traces must be an array".to_owned())?;
    if records.is_empty() {
        return Err("trace_manifest.traces must not be empty".to_owned());
    }

    let mut traces = BTreeMap::new();
    let mut summary = TraceSummary::default();
    let mut sources = BTreeSet::new();
    let mut programs = BTreeSet::new();
    let mut executions = BTreeSet::new();
    let mut devices = BTreeSet::new();
    let mut attacks = BTreeSet::new();
    let mut adaptations = BTreeSet::new();
    let mut program_splits = BTreeSet::new();
    let mut all_window_hashes: BTreeMap<String, usize> = BTreeMap::new();
    let mut global_window_ids = BTreeSet::new();

    for (index, item) in records.iter().enumerate() {
        let context = format!("trace_manifest.traces[{index}]");
        let trace_id = string_field(item, "trace_id", &context)?.to_owned();
        let source_sha = string_field(item, "source_trace_sha256", &context)?;
        valid_sha(source_sha, &format!("{context}.source_trace_sha256"))?;
        let binary_sha = string_field(item, "binary_sha256", &context)?;
        valid_sha(binary_sha, &format!("{context}.binary_sha256"))?;
        let program = string_field(item, "program_id", &context)?.to_owned();
        let execution = string_field(item, "execution_id", &context)?.to_owned();
        let device = string_field(item, "device_config_id", &context)?.to_owned();
        let attack_family = string_field(item, "attack_family_id", &context)?.to_owned();
        let adaptation_family = string_field(item, "adaptation_family_id", &context)?.to_owned();
        let split = string_field(item, "split", &context)?.to_owned();
        if !matches!(split.as_str(), "train" | "validation" | "test") {
            return Err(format!("{context}.split is unsupported"));
        }
        let origin = string_field(item, "origin", &context)?.to_owned();
        if !matches!(origin.as_str(), "original" | "replayed" | "repeated_iteration" | "unknown") {
            return Err(format!("{context}.origin is unsupported"));
        }
        let windows = item.get("windows").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.windows must be an array"))?;
        if windows.is_empty() {
            return Err(format!("{context}.windows must not be empty"));
        }
        let mut trace_windows = BTreeMap::new();
        for (window_index, window) in windows.iter().enumerate() {
            let window_context = format!("{context}.windows[{window_index}]");
            let window_id = string_field(window, "window_id", &window_context)?.to_owned();
            let window_sha = string_field(window, "window_sha256", &window_context)?.to_owned();
            valid_sha(&window_sha, &format!("{window_context}.window_sha256"))?;
            if window.get("sample_index").and_then(Value::as_u64).is_none() {
                return Err(format!("{window_context}.sample_index must be a non-negative integer"));
            }
            let start = window.get("start_offset").ok_or_else(|| format!("{window_context}.start_offset is required"))?;
            let end = window.get("end_offset").ok_or_else(|| format!("{window_context}.end_offset is required"))?;
            match (start.as_u64(), end.as_u64(), start.is_null(), end.is_null()) {
                (Some(s), Some(e), _, _) if e > s => {},
                (None, None, true, true) => {},
                _ => return Err(format!("{window_context} offsets must both be null or an increasing non-negative range")),
            }
            if !global_window_ids.insert(window_id.clone()) {
                return Err(format!("duplicate global window_id {window_id}"));
            }
            if trace_windows.insert(window_id.clone(), WindowMeta { sha256: window_sha.clone() }).is_some() {
                return Err(format!("duplicate window_id {window_id} within trace"));
            }
            *all_window_hashes.entry(window_sha).or_default() += 1;
            summary.raw_window_count += 1;
            *summary.origin_window_counts.entry(origin.clone()).or_default() += 1;
            *summary.split_window_counts.entry(split.clone()).or_default() += 1;
        }
        if traces.insert(trace_id.clone(), TraceMeta {
            program: program.clone(),
            execution: execution.clone(),
            device: device.clone(),
            attack_family: attack_family.clone(),
            adaptation_family: adaptation_family.clone(),
            split: split.clone(),
            origin: origin.clone(),
            windows: trace_windows,
        }).is_some() {
            return Err(format!("duplicate trace_id {trace_id}"));
        }
        sources.insert(source_sha.to_owned());
        programs.insert(program.clone());
        executions.insert(execution);
        devices.insert(device);
        if attack_family != "none" { attacks.insert(attack_family); }
        if adaptation_family != "none" { adaptations.insert(adaptation_family); }
        program_splits.insert((program, split));
        *summary.origin_trace_counts.entry(origin).or_default() += 1;
    }

    for (hash, count) in &all_window_hashes {
        if *count > 1 {
            summary.duplicate_hash_groups.push(json!({"window_sha256":hash,"occurrences":count}));
            summary.duplicate_extra_occurrences += count - 1;
        }
    }
    summary.trace_count = records.len();
    summary.source_trace_count = sources.len();
    summary.duplicate_hash_group_count = summary.duplicate_hash_groups.len();
    summary.program_count = programs.len();
    summary.execution_count = executions.len();
    summary.device_count = devices.len();
    summary.attack_family_count = attacks.len();
    summary.adaptation_family_count = adaptations.len();
    summary.program_split_pairs = program_splits.len();

    let _ = axes;
    Ok((dataset_id, corpus_sha, traces, summary))
}

fn reconcile_trace_receipt(
    receipt: &Value,
    trace_bytes: &[u8],
    dataset_id: &str,
    corpus_sha: &str,
    summary: &TraceSummary,
) -> Result<(), String> {
    if string_field(receipt, "schema_version", "trace_receipt")? != TRACE_RECEIPT_SCHEMA {
        return Err("unsupported HARDSEC-014 trace receipt schema_version".to_owned());
    }
    if string_field(receipt, "result", "trace_receipt")? != "audited" {
        return Err("HARDSEC-014 receipt result must be audited".to_owned());
    }
    if string_field(receipt, "claim_scope", "trace_receipt")? != TRACE_CLAIM_SCOPE {
        return Err("HARDSEC-014 receipt has an unexpected claim_scope".to_owned());
    }
    if string_field(receipt, "dataset_id", "trace_receipt")? != dataset_id {
        return Err("trace receipt dataset_id does not match trace manifest".to_owned());
    }
    let inputs = receipt.get("inputs").ok_or("trace_receipt.inputs is missing")?;
    let manifest_sha = string_field(inputs, "manifest_sha256", "trace_receipt.inputs")?;
    if manifest_sha != sha256(trace_bytes) {
        return Err("trace receipt manifest_sha256 does not match exact trace-manifest bytes".to_owned());
    }
    if string_field(inputs, "source_corpus_sha256", "trace_receipt.inputs")? != corpus_sha {
        return Err("trace receipt source_corpus_sha256 does not match trace manifest".to_owned());
    }

    for (path, expected) in [
        (&["trace_lineage", "trace_record_count"][..], summary.trace_count),
        (&["trace_lineage", "original_trace_count"][..], *summary.origin_trace_counts.get("original").unwrap_or(&0)),
        (&["trace_lineage", "repeated_iteration_trace_count"][..], *summary.origin_trace_counts.get("repeated_iteration").unwrap_or(&0)),
        (&["trace_lineage", "replayed_trace_count"][..], *summary.origin_trace_counts.get("replayed").unwrap_or(&0)),
        (&["trace_lineage", "unknown_origin_trace_count"][..], *summary.origin_trace_counts.get("unknown").unwrap_or(&0)),
        (&["trace_lineage", "unique_source_trace_sha256_count"][..], summary.source_trace_count),
        (&["window_support", "raw_window_count"][..], summary.raw_window_count),
        (&["window_support", "original_window_count"][..], *summary.origin_window_counts.get("original").unwrap_or(&0)),
        (&["window_support", "repeated_iteration_window_count"][..], *summary.origin_window_counts.get("repeated_iteration").unwrap_or(&0)),
        (&["window_support", "replayed_window_count"][..], *summary.origin_window_counts.get("replayed").unwrap_or(&0)),
        (&["window_support", "unknown_origin_window_count"][..], *summary.origin_window_counts.get("unknown").unwrap_or(&0)),
        (&["window_support", "duplicate_content_hash_group_count"][..], summary.duplicate_hash_group_count),
        (&["window_support", "duplicate_content_extra_occurrences"][..], summary.duplicate_extra_occurrences),
        (&["independent_group_support", "unique_program_count"][..], summary.program_count),
        (&["independent_group_support", "unique_execution_id_count"][..], summary.execution_count),
        (&["independent_group_support", "unique_device_config_count"][..], summary.device_count),
        (&["independent_group_support", "unique_attack_family_count_excluding_none"][..], summary.attack_family_count),
        (&["independent_group_support", "unique_adaptation_family_count_excluding_none"][..], summary.adaptation_family_count),
        (&["independent_group_support", "program_split_pairs"][..], summary.program_split_pairs),
    ] {
        count_at(receipt, path, expected, "trace_receipt")?;
    }

    let origin_counts = receipt.pointer("/trace_lineage/trace_origin_counts")
        .ok_or("trace_receipt.trace_lineage.trace_origin_counts is missing")?;
    if origin_counts != &serde_json::to_value(&summary.origin_trace_counts).map_err(|e| e.to_string())? {
        return Err("trace receipt origin counts do not match trace manifest".to_owned());
    }
    let duplicate_groups = receipt.pointer("/window_support/duplicate_content_hash_groups")
        .ok_or("trace_receipt.window_support.duplicate_content_hash_groups is missing")?;
    if duplicate_groups != &Value::Array(summary.duplicate_hash_groups.clone()) {
        return Err("trace receipt duplicate-content groups do not match trace manifest".to_owned());
    }
    let split_counts = receipt.pointer("/window_support/split_window_counts")
        .ok_or("trace_receipt.window_support.split_window_counts is missing")?;
    if split_counts != &serde_json::to_value(&summary.split_window_counts).map_err(|e| e.to_string())? {
        return Err("trace receipt split-window counts do not match trace manifest".to_owned());
    }
    let group_support = receipt.get("independent_group_support")
        .ok_or("trace_receipt.independent_group_support is missing")?;
    if !group_support.get("effective_sample_size_estimate").is_some_and(Value::is_null) {
        return Err("HARDSEC-014 effective_sample_size_estimate must remain null".to_owned());
    }
    if string_field(group_support, "effective_sample_size_status", "trace_receipt.independent_group_support")? != "not_estimated" {
        return Err("HARDSEC-014 effective_sample_size_status must be not_estimated".to_owned());
    }
    Ok(())
}

fn audit_values(
    adaptive: &Value,
    trace: &Value,
    trace_receipt: &Value,
    links: &Value,
    adaptive_bytes: &[u8],
    trace_bytes: &[u8],
    trace_receipt_bytes: &[u8],
    links_bytes: &[u8],
) -> Result<Value, String> {
    let (campaign_id, adaptive_axes, cases) = parse_adaptive_cases(adaptive)?;
    let (dataset_id, corpus_sha, traces, summary) = parse_traces(trace)?;
    if required_axes(trace, "trace_manifest")? != adaptive_axes {
        return Err("adaptive and trace manifests declare different holdout axes".to_owned());
    }
    reconcile_trace_receipt(trace_receipt, trace_bytes, &dataset_id, &corpus_sha, &summary)?;

    if string_field(links, "schema_version", "case_links")? != LINKS_SCHEMA {
        return Err("unsupported case-links schema_version".to_owned());
    }
    if string_field(links, "campaign_id", "case_links")? != campaign_id {
        return Err("case-links campaign_id does not match adaptive manifest".to_owned());
    }
    if string_field(links, "trace_dataset_id", "case_links")? != dataset_id {
        return Err("case-links trace_dataset_id does not match trace manifest".to_owned());
    }
    let link_records = links.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "case_links.cases must be an array".to_owned())?;
    if link_records.len() != cases.len() {
        return Err(format!("case-link count {} does not equal adaptive case count {}", link_records.len(), cases.len()));
    }

    let mut seen_cases = BTreeSet::new();
    let mut used_window_ids: BTreeMap<String, String> = BTreeMap::new();
    let mut content_case_owner: BTreeMap<String, String> = BTreeMap::new();
    let mut mapped_trace_ids = BTreeSet::new();
    let mut mapped_programs = BTreeSet::new();
    let mut mapped_executions = BTreeSet::new();
    let mut mapped_devices = BTreeSet::new();
    let mut mapped_attacks = BTreeSet::new();
    let mut mapped_adaptations = BTreeSet::new();
    let mut mapped_origin_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut mapped_window_count = 0usize;

    for (index, link) in link_records.iter().enumerate() {
        let context = format!("case_links.cases[{index}]");
        let case_id = string_field(link, "case_id", &context)?.to_owned();
        if !seen_cases.insert(case_id.clone()) {
            return Err(format!("duplicate case link for case_id {case_id}"));
        }
        let case = cases.get(&case_id).ok_or_else(|| format!("case link references unknown adaptive case_id {case_id}"))?;
        let trace_id = string_field(link, "trace_id", &context)?;
        let trace_meta = traces.get(trace_id).ok_or_else(|| format!("case {case_id} references unknown trace_id {trace_id}"))?;
        if trace_meta.origin == "unknown" {
            return Err(format!("case {case_id} links to a trace with unknown origin"));
        }
        for (field, expected, actual) in [
            ("split", &case.split, &trace_meta.split),
            ("program_group_id/program_id", &case.program, &trace_meta.program),
            ("execution_group_id/execution_id", &case.execution, &trace_meta.execution),
            ("device_config_id", &case.device, &trace_meta.device),
            ("attack_family_id", &case.attack_family, &trace_meta.attack_family),
            ("adaptation_family_id", &case.adaptation_family, &trace_meta.adaptation_family),
        ] {
            if expected != actual {
                return Err(format!("case {case_id} metadata mismatch for {field}: adaptive={expected}, trace={actual}"));
            }
        }
        let window_ids = link.get("window_ids").and_then(Value::as_array)
            .ok_or_else(|| format!("{context}.window_ids must be an array"))?;
        if window_ids.is_empty() {
            return Err(format!("{context}.window_ids must not be empty"));
        }
        let mut case_window_ids = BTreeSet::new();
        for value in window_ids {
            let window_id = value.as_str().ok_or_else(|| format!("{context}.window_ids entries must be strings"))?;
            nonempty(window_id, &format!("{context}.window_ids entry"))?;
            if !case_window_ids.insert(window_id.to_owned()) {
                return Err(format!("case {case_id} repeats window_id {window_id}"));
            }
            if let Some(owner) = used_window_ids.insert(window_id.to_owned(), case_id.clone()) {
                return Err(format!("window_id {window_id} is linked by both case {owner} and {case_id}"));
            }
            let window = trace_meta.windows.get(window_id)
                .ok_or_else(|| format!("case {case_id} window {window_id} is not present in trace {trace_id}"))?;
            if let Some(owner) = content_case_owner.insert(window.sha256.clone(), case_id.clone()) {
                if owner != case_id {
                    return Err(format!("window content {0} is reused across adaptive cases {owner} and {case_id}", window.sha256));
                }
            }
            mapped_window_count += 1;
        }
        *mapped_origin_counts.entry(trace_meta.origin.clone()).or_default() += window_ids.len();
        mapped_trace_ids.insert(trace_id.to_owned());
        mapped_programs.insert(trace_meta.program.clone());
        mapped_executions.insert(trace_meta.execution.clone());
        mapped_devices.insert(trace_meta.device.clone());
        if trace_meta.attack_family != "none" { mapped_attacks.insert(trace_meta.attack_family.clone()); }
        if trace_meta.adaptation_family != "none" { mapped_adaptations.insert(trace_meta.adaptation_family.clone()); }
    }
    if seen_cases.len() != cases.len() {
        let missing: Vec<_> = cases.keys().filter(|id| !seen_cases.contains(*id)).cloned().collect();
        return Err(format!("case links omit adaptive case IDs: {}", missing.join(", ")));
    }

    let no_ess = json!({
        "effective_sample_size_estimate": null,
        "effective_sample_size_status": "not_estimated",
        "independence_status": "not_established",
        "reason": "case-to-trace linkage and group counts do not establish temporal or cluster independence"
    });
    Ok(json!({
        "schema_version":"hardsec-adaptive-case-trace-link-receipt-v1",
        "result":"linked",
        "claim_scope":"case_trace_window_identity_and_metadata_consistency_only",
        "campaign_id":campaign_id,
        "trace_dataset_id":dataset_id,
        "inputs":{
            "adaptive_manifest_sha256":sha256(adaptive_bytes),
            "trace_manifest_sha256":sha256(trace_bytes),
            "trace_audit_receipt_sha256":sha256(trace_receipt_bytes),
            "case_links_sha256":sha256(links_bytes)
        },
        "trace_audit":{
            "result":"audited",
            "claim_scope":TRACE_CLAIM_SCOPE,
            "trace_record_count":summary.trace_count,
            "raw_window_count":summary.raw_window_count,
            "duplicate_content_hash_group_count":summary.duplicate_hash_group_count,
            "duplicate_content_extra_occurrences":summary.duplicate_extra_occurrences,
            "repeated_iteration_trace_count":summary.origin_trace_counts.get("repeated_iteration").copied().unwrap_or(0),
            "replayed_trace_count":summary.origin_trace_counts.get("replayed").copied().unwrap_or(0),
            "unknown_origin_trace_count":summary.origin_trace_counts.get("unknown").copied().unwrap_or(0)
        },
        "linkage":{
            "adaptive_case_count":cases.len(),
            "linked_case_count":seen_cases.len(),
            "mapped_trace_count":mapped_trace_ids.len(),
            "mapped_window_count":mapped_window_count,
            "mapped_window_origin_counts":mapped_origin_counts,
            "mapped_program_group_count":mapped_programs.len(),
            "mapped_execution_group_count":mapped_executions.len(),
            "mapped_device_config_count":mapped_devices.len(),
            "mapped_attack_family_count_excluding_none":mapped_attacks.len(),
            "mapped_adaptation_family_count_excluding_none":mapped_adaptations.len(),
            "metadata_match_status":"exact",
            "case_id_coverage_status":"exact",
            "cross_case_window_content_reuse_count":0
        },
        "statistical_support":no_ess,
        "limitations":[
            "The HARDSEC-014 receipt is unsigned; matching its digests and summary does not authenticate its author or prove the auditor executed.",
            "Trace and case identities remain caller-declared; this receipt does not authenticate raw source files, binaries, devices or labels.",
            "Case-to-trace/window linkage does not establish statistical independence, effective sample size, causal detection or cross-platform transfer.",
            "Repeated and replayed traces remain part of their declared source lineage; they do not create new independent programs or executions.",
            "This linker validates associations only; run the HARDSEC-014 auditor against the same trace manifest and preserve its receipt."
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
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !matches!(flag.as_str(), "--adaptive-manifest" | "--trace-manifest" | "--trace-receipt" | "--case-links" | "--receipt") {
            return Err(format!("unknown argument {flag}"));
        }
        let path = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if values.insert(flag.clone(), PathBuf::from(path)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let take = |key: &str| values.get(key).cloned().ok_or_else(|| format!("missing required argument {key}"));
    Ok(Args {
        adaptive_manifest:take("--adaptive-manifest")?,
        trace_manifest:take("--trace-manifest")?,
        trace_receipt:take("--trace-receipt")?,
        case_links:take("--case-links")?,
        receipt:take("--receipt")?,
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
    let (adaptive_bytes, adaptive) = read_input(&args.adaptive_manifest, "adaptive manifest")?;
    let (trace_bytes, trace) = read_input(&args.trace_manifest, "trace manifest")?;
    let (trace_receipt_bytes, trace_receipt) = read_input(&args.trace_receipt, "trace receipt")?;
    let (links_bytes, links) = read_input(&args.case_links, "case links")?;
    let inputs = [
        args.adaptive_manifest.as_path(), args.trace_manifest.as_path(),
        args.trace_receipt.as_path(), args.case_links.as_path()
    ];
    validate_receipt_location(&args.receipt, &inputs)?;
    let receipt = audit_values(
        &adaptive, &trace, &trace_receipt, &links,
        &adaptive_bytes, &trace_bytes, &trace_receipt_bytes, &links_bytes
    )?;
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
            eprintln!("HARDSEC-016 case-trace link audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value, Value, Value, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let adaptive_bytes = include_bytes!("../tests/fixtures/hardsec/case-trace-links-v1/adaptive-case-manifest.json").to_vec();
        let trace_bytes = include_bytes!("../tests/fixtures/hardsec/case-trace-links-v1/trace-lineage-manifest.json").to_vec();
        let receipt_bytes = include_bytes!("../tests/fixtures/hardsec/case-trace-links-v1/trace-lineage-receipt.json").to_vec();
        let links_bytes = include_bytes!("../tests/fixtures/hardsec/case-trace-links-v1/case-trace-links.json").to_vec();
        let adaptive = serde_json::from_slice(&adaptive_bytes).unwrap();
        let trace = serde_json::from_slice(&trace_bytes).unwrap();
        let receipt = serde_json::from_slice(&receipt_bytes).unwrap();
        let links = serde_json::from_slice(&links_bytes).unwrap();
        (adaptive, trace, receipt, links, adaptive_bytes, trace_bytes, receipt_bytes, links_bytes)
    }

    fn json_bytes(value: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn refresh_trace_receipt(trace: &Value, trace_bytes: &[u8], receipt: &mut Value) {
        let (_, _, traces, summary) = parse_traces(trace).unwrap();
        let _ = traces;
        receipt["inputs"]["manifest_sha256"] = json!(sha256(trace_bytes));
        receipt["inputs"]["source_corpus_sha256"] = trace["source_corpus_sha256"].clone();
        receipt["dataset_id"] = trace["dataset_id"].clone();
        receipt["trace_lineage"]["trace_record_count"] = json!(summary.trace_count);
        receipt["trace_lineage"]["original_trace_count"] = json!(summary.origin_trace_counts.get("original").copied().unwrap_or(0));
        receipt["trace_lineage"]["repeated_iteration_trace_count"] = json!(summary.origin_trace_counts.get("repeated_iteration").copied().unwrap_or(0));
        receipt["trace_lineage"]["replayed_trace_count"] = json!(summary.origin_trace_counts.get("replayed").copied().unwrap_or(0));
        receipt["trace_lineage"]["unknown_origin_trace_count"] = json!(summary.origin_trace_counts.get("unknown").copied().unwrap_or(0));
        receipt["trace_lineage"]["unique_source_trace_sha256_count"] = json!(summary.source_trace_count);
        receipt["trace_lineage"]["trace_origin_counts"] = serde_json::to_value(summary.origin_trace_counts).unwrap();
        receipt["window_support"]["raw_window_count"] = json!(summary.raw_window_count);
        receipt["window_support"]["original_window_count"] = json!(summary.origin_window_counts.get("original").copied().unwrap_or(0));
        receipt["window_support"]["repeated_iteration_window_count"] = json!(summary.origin_window_counts.get("repeated_iteration").copied().unwrap_or(0));
        receipt["window_support"]["replayed_window_count"] = json!(summary.origin_window_counts.get("replayed").copied().unwrap_or(0));
        receipt["window_support"]["unknown_origin_window_count"] = json!(summary.origin_window_counts.get("unknown").copied().unwrap_or(0));
        receipt["window_support"]["duplicate_content_hash_groups"] = json!(summary.duplicate_hash_groups);
        receipt["window_support"]["duplicate_content_hash_group_count"] = json!(summary.duplicate_hash_group_count);
        receipt["window_support"]["duplicate_content_extra_occurrences"] = json!(summary.duplicate_extra_occurrences);
        receipt["window_support"]["split_window_counts"] = serde_json::to_value(summary.split_window_counts).unwrap();
        receipt["independent_group_support"]["unique_program_count"] = json!(summary.program_count);
        receipt["independent_group_support"]["unique_execution_id_count"] = json!(summary.execution_count);
        receipt["independent_group_support"]["unique_device_config_count"] = json!(summary.device_count);
        receipt["independent_group_support"]["unique_attack_family_count_excluding_none"] = json!(summary.attack_family_count);
        receipt["independent_group_support"]["unique_adaptation_family_count_excluding_none"] = json!(summary.adaptation_family_count);
        receipt["independent_group_support"]["program_split_pairs"] = json!(summary.program_split_pairs);
        receipt["independent_group_support"]["effective_sample_size_estimate"] = Value::Null;
        receipt["independent_group_support"]["effective_sample_size_status"] = json!("not_estimated");
    }

    fn run_fixture(
        adaptive: &Value, trace: &Value, trace_receipt: &Value, links: &Value,
        adaptive_bytes: &[u8], trace_bytes: &[u8], trace_receipt_bytes: &[u8], links_bytes: &[u8]
    ) -> Result<Value, String> {
        audit_values(adaptive, trace, trace_receipt, links,
            adaptive_bytes, trace_bytes, trace_receipt_bytes, links_bytes)
    }

    #[test]
    fn known_answer_binds_every_case_and_preserves_non_independence_claim_ceiling() {
        let (a,t,r,l,ab,tb,rb,lb)=fixture();
        let out=run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap();
        assert_eq!(out["result"],"linked");
        assert_eq!(out["linkage"]["adaptive_case_count"],11);
        assert_eq!(out["linkage"]["linked_case_count"],11);
        assert_eq!(out["linkage"]["mapped_trace_count"],11);
        assert_eq!(out["linkage"]["mapped_window_count"],11);
        assert_eq!(out["trace_audit"]["trace_record_count"],12);
        assert_eq!(out["trace_audit"]["duplicate_content_hash_group_count"],1);
        assert_eq!(out["trace_audit"]["repeated_iteration_trace_count"],1);
        assert_eq!(out["statistical_support"]["effective_sample_size_estimate"],Value::Null);
        assert_eq!(out["statistical_support"]["independence_status"],"not_established");
    }

    #[test]
    fn rejects_missing_case_link() {
        let (a,t,r,mut l,ab,tb,rb,mut lb)=fixture();
        l["cases"].as_array_mut().unwrap().pop();
        lb=json_bytes(&l);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("case-link count"));
    }

    #[test]
    fn rejects_case_to_trace_program_mismatch() {
        let (a,t,r,mut l,ab,tb,rb,mut lb)=fixture();
        l["cases"][2]["trace_id"]=json!("trace-c2");
        lb=json_bytes(&l);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("program_group_id/program_id"));
    }

    #[test]
    fn rejects_same_window_content_linked_to_two_cases() {
        let (a,mut t,mut r,l,ab,mut tb,mut rb,lb)=fixture();
        let c1_hash=t["traces"][2]["windows"][0]["window_sha256"].clone();
        t["traces"][3]["windows"][0]["window_sha256"]=c1_hash;
        tb=json_bytes(&t);
        refresh_trace_receipt(&t,&tb,&mut r);
        rb=json_bytes(&r);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("reused across adaptive cases"));
    }

    #[test]
    fn rejects_receipt_for_different_trace_bytes() {
        let (a,t,mut r,l,ab,tb,mut rb,lb)=fixture();
        r["inputs"]["manifest_sha256"]=json!("0".repeat(64));
        rb=json_bytes(&r);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("exact trace-manifest bytes"));
    }

    #[test]
    fn rejects_non_null_effective_sample_size_claim() {
        let (a,t,mut r,l,ab,tb,mut rb,lb)=fixture();
        r["independent_group_support"]["effective_sample_size_estimate"]=json!(999);
        rb=json_bytes(&r);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("must remain null"));
    }

    #[test]
    fn rejects_unknown_origin_for_linked_trace() {
        let (a,mut t,mut r,l,ab,mut tb,mut rb,lb)=fixture();
        t["traces"][2]["origin"]=json!("unknown");
        tb=json_bytes(&t);
        refresh_trace_receipt(&t,&tb,&mut r);
        rb=json_bytes(&r);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("unknown origin"));
    }

    #[test]
    fn rejects_duplicate_global_window_identity() {
        let (a,mut t,mut r,l,ab,mut tb,mut rb,lb)=fixture();
        t["traces"][3]["windows"][0]["window_id"]=t["traces"][2]["windows"][0]["window_id"].clone();
        tb=json_bytes(&t);
        refresh_trace_receipt(&t,&tb,&mut r);
        rb=json_bytes(&r);
        assert!(run_fixture(&a,&t,&r,&l,&ab,&tb,&rb,&lb).unwrap_err().contains("duplicate global window_id"));
    }
}
