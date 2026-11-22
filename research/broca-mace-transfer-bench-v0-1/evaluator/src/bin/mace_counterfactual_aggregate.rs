// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline, cluster-level statistical reduction for MACE counterfactual audits.
//!
//! Repeated event/root/family interventions are first averaged within
//! task/world blocks, then within independent world clusters. Bootstrap
//! resampling is performed over world clusters, never raw intervention rows.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const INPUT_SCHEMA: &str = "mace_counterfactual_evidence_v0_1";
const OUTPUT_SCHEMA: &str = "mace_counterfactual_summary_v0_1";
const BOOTSTRAP_REPLICATES: usize = 2_000;
const EXACT_METHOD: &str = "exact_scope_scan";

type MethodKind = (String, String);
type TaskKey = (u64, String);

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    utility: f64,
    regret: f64,
    regret_gap_vs_full_evidence: f64,
    regret_gap_vs_full_method: f64,
    decision_agreement: f64,
    decision_changed: f64,
    decision_optimal: f64,
    abstained: f64,
}

impl Metrics {
    fn divide(self, denominator: f64) -> Self {
        Self {
            utility: self.utility / denominator,
            regret: self.regret / denominator,
            regret_gap_vs_full_evidence: self.regret_gap_vs_full_evidence / denominator,
            regret_gap_vs_full_method: self.regret_gap_vs_full_method / denominator,
            decision_agreement: self.decision_agreement / denominator,
            decision_changed: self.decision_changed / denominator,
            decision_optimal: self.decision_optimal / denominator,
            abstained: self.abstained / denominator,
        }
    }

    fn plus(self, other: Self) -> Self {
        Self {
            utility: self.utility + other.utility,
            regret: self.regret + other.regret,
            regret_gap_vs_full_evidence: self.regret_gap_vs_full_evidence
                + other.regret_gap_vs_full_evidence,
            regret_gap_vs_full_method: self.regret_gap_vs_full_method
                + other.regret_gap_vs_full_method,
            decision_agreement: self.decision_agreement + other.decision_agreement,
            decision_changed: self.decision_changed + other.decision_changed,
            decision_optimal: self.decision_optimal + other.decision_optimal,
            abstained: self.abstained + other.abstained,
        }
    }

    fn metric(self, name: &str) -> Option<f64> {
        match name {
            "utility" => Some(self.utility),
            "regret" => Some(self.regret),
            "regret_gap_vs_full_evidence" => Some(self.regret_gap_vs_full_evidence),
            "regret_gap_vs_full_method" => Some(self.regret_gap_vs_full_method),
            "decision_agreement" => Some(self.decision_agreement),
            "decision_changed" => Some(self.decision_changed),
            "decision_optimal" => Some(self.decision_optimal),
            "abstained" => Some(self.abstained),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Accumulator {
    sums: Metrics,
    count: u64,
}

impl Accumulator {
    fn push(&mut self, value: Metrics) {
        self.sums = self.sums.plus(value);
        self.count = self.count.saturating_add(1);
    }

    fn mean(self) -> Option<Metrics> {
        (self.count > 0).then(|| self.sums.divide(self.count as f64))
    }
}

#[derive(Clone, Debug)]
struct SummaryRows {
    task_blocks: usize,
    intervention_rows: usize,
    cluster_means: BTreeMap<u64, Metrics>,
}

fn required_string<'a>(record: &'a Value, field: &str) -> Result<&'a str, String> {
    record
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing or invalid string field {field}"))
}

fn required_number(record: &Value, field: &str) -> Result<f64, String> {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or invalid numeric field {field}"))?;
    if !value.is_finite() {
        return Err(format!("non-finite numeric field {field}"));
    }
    Ok(value)
}

fn required_bool(record: &Value, field: &str) -> Result<bool, String> {
    record
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or invalid boolean field {field}"))
}

fn parse_cluster(pair_block_id: &str) -> Result<u64, String> {
    let suffix = pair_block_id
        .strip_prefix("world-")
        .ok_or_else(|| "pair_block_id must start with world-".to_string())?;
    let (cluster, task) = suffix
        .split_once(":task-")
        .ok_or_else(|| "pair_block_id is missing the opaque task component".to_string())?;
    if task.is_empty() {
        return Err("pair_block_id task component must not be empty".to_string());
    }
    cluster
        .parse::<u64>()
        .map_err(|error| format!("invalid world cluster ID: {error}"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InputProvenance {
    run_id: String,
    source_code_sha256: String,
    fixture_sha256: String,
    schema_sha256: String,
    source_revision: Option<String>,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_manifest(record: &Value) -> Result<(InputProvenance, String), String> {
    if required_string(record, "schema_version")? != INPUT_SCHEMA {
        return Err("unsupported intervention manifest schema version".to_string());
    }
    let provenance = InputProvenance {
        run_id: required_string(record, "run_id")?.to_string(),
        source_code_sha256: required_string(record, "source_code_sha256")?.to_string(),
        fixture_sha256: required_string(record, "fixture_sha256")?.to_string(),
        schema_sha256: required_string(record, "schema_sha256")?.to_string(),
        source_revision: record
            .get("source_revision")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    for (field, digest) in [
        ("source_code_sha256", provenance.source_code_sha256.as_str()),
        ("fixture_sha256", provenance.fixture_sha256.as_str()),
        ("schema_sha256", provenance.schema_sha256.as_str()),
    ] {
        if !valid_sha256(digest) {
            return Err(format!("invalid {field} in intervention manifest"));
        }
    }
    if provenance.run_id.is_empty() {
        return Err("intervention manifest run_id must not be empty".to_string());
    }
    if provenance
        .source_revision
        .as_deref()
        .is_some_and(|revision| {
            !matches!(revision.len(), 40 | 64)
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err("intervention manifest has invalid source_revision".to_string());
    }
    let manifest_sha256 = required_string(record, "manifest_sha256")?.to_string();
    if !valid_sha256(&manifest_sha256) {
        return Err("invalid manifest_sha256".to_string());
    }
    Ok((provenance, manifest_sha256))
}

fn parse_score(record: &Value) -> Result<(TaskKey, MethodKind, Metrics), String> {
    if required_string(record, "schema_version")? != INPUT_SCHEMA {
        return Err("unsupported input schema version".to_string());
    }
    if required_string(record, "record_type")? != "evaluator_score" {
        return Err("expected evaluator_score record".to_string());
    }

    let pair_block_id = required_string(record, "pair_block_id")?;
    let cluster = parse_cluster(pair_block_id)?;
    let method = required_string(record, "method")?.to_string();
    let intervention_kind = required_string(record, "intervention_kind")?.to_string();
    let metrics = Metrics {
        utility: required_number(record, "utility")?,
        regret: required_number(record, "regret")?,
        regret_gap_vs_full_evidence: required_number(record, "regret_gap_vs_full_evidence")?,
        regret_gap_vs_full_method: required_number(record, "regret_gap_vs_full_method")?,
        decision_agreement: if required_bool(record, "decision_agrees_with_full")? { 1.0 } else { 0.0 },
        decision_changed: if required_bool(record, "decision_changed")? { 1.0 } else { 0.0 },
        decision_optimal: if required_bool(record, "decision_is_optimal")? { 1.0 } else { 0.0 },
        abstained: if required_bool(record, "abstained")? { 1.0 } else { 0.0 },
    };

    Ok((
        (cluster, pair_block_id.to_string()),
        (method, intervention_kind),
        metrics,
    ))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn stable_seed(seed: u64, key: &str, metric: &str) -> u64 {
    let digest = Sha256::digest(format!("{seed}:{key}:{metric}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is eight bytes")).max(1)
}

fn bootstrap_ci(values: &[f64], seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    if values.len() == 1 {
        return (values[0], values[0]);
    }

    let mut state = seed.max(1);
    let mut means = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut total = 0.0;
        for _ in 0..values.len() {
            let index = (next_random(&mut state) % values.len() as u64) as usize;
            total += values[index];
        }
        means.push(total / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    (percentile(&means, 0.025), percentile(&means, 0.975))
}

fn cluster_means(
    block_rows: &BTreeMap<TaskKey, BTreeMap<MethodKind, Accumulator>>,
) -> BTreeMap<MethodKind, BTreeMap<u64, Vec<Metrics>>> {
    let mut grouped: BTreeMap<MethodKind, BTreeMap<u64, Vec<Metrics>>> = BTreeMap::new();
    for ((cluster, _pair_block), methods) in block_rows {
        for (key, accumulator) in methods {
            if let Some(mean) = accumulator.mean() {
                grouped
                    .entry(key.clone())
                    .or_default()
                    .entry(*cluster)
                    .or_default()
                    .push(mean);
            }
        }
    }
    grouped
}

fn reduce_cluster_blocks(blocks: &BTreeMap<u64, Vec<Metrics>>) -> BTreeMap<u64, Metrics> {
    blocks
        .iter()
        .filter_map(|(cluster, rows)| {
            if rows.is_empty() {
                return None;
            }
            let total = rows.iter().copied().fold(Metrics::default(), Metrics::plus);
            Some((*cluster, total.divide(rows.len() as f64)))
        })
        .collect()
}

fn summary_record(
    method: &str,
    intervention_kind: &str,
    cluster_values: &BTreeMap<u64, Metrics>,
    task_block_count: usize,
    intervention_row_count: usize,
    seed: u64,
) -> Value {
    let cluster_metrics: Vec<Metrics> = cluster_values.values().copied().collect();
    let means = cluster_metrics
        .iter()
        .copied()
        .fold(Metrics::default(), Metrics::plus)
        .divide(cluster_metrics.len().max(1) as f64);

    let cluster_values_for = |metric: &str| -> Vec<f64> {
        cluster_values
            .values()
            .filter_map(|value| value.metric(metric))
            .collect()
    };

    let gap_method_values = cluster_values_for("regret_gap_vs_full_method");
    let gap_evidence_values = cluster_values_for("regret_gap_vs_full_evidence");
    let (gap_method_lower, gap_method_upper) = bootstrap_ci(
        &gap_method_values,
        stable_seed(seed, method, intervention_kind),
    );
    let (gap_evidence_lower, gap_evidence_upper) = bootstrap_ci(
        &gap_evidence_values,
        stable_seed(seed ^ 0x9E37_79B9, method, intervention_kind),
    );

    json!({
        "schema_version": OUTPUT_SCHEMA,
        "record_type": "counterfactual_method_summary",
        "method": method,
        "intervention_kind": intervention_kind,
        "independent_world_cluster_count": cluster_values.len(),
        "task_block_count": task_block_count,
        "intervention_row_count": intervention_row_count,
        "mean_utility": means.utility,
        "mean_regret": means.regret,
        "mean_regret_gap_vs_full_evidence": means.regret_gap_vs_full_evidence,
        "mean_regret_gap_vs_full_method": means.regret_gap_vs_full_method,
        "regret_gap_vs_full_evidence_cluster_bootstrap_95pct_ci": [gap_evidence_lower, gap_evidence_upper],
        "regret_gap_vs_full_method_cluster_bootstrap_95pct_ci": [gap_method_lower, gap_method_upper],
        "mean_decision_agreement_with_full": means.decision_agreement,
        "mean_decision_changed": means.decision_changed,
        "mean_decision_optimal": means.decision_optimal,
        "mean_abstained": means.abstained,
        "resampling_unit": "independent_world_cluster",
        "task_block_aggregation": "mean_within_cluster_before_bootstrap",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "evaluation_validity": "diagnostic_only"
    })
}

fn paired_record(
    method: &str,
    intervention_kind: &str,
    cluster_values: &BTreeMap<u64, Metrics>,
    task_block_count: usize,
    seed: u64,
) -> Value {
    let cluster_metrics: Vec<Metrics> = cluster_values.values().copied().collect();
    let means = cluster_metrics
        .iter()
        .copied()
        .fold(Metrics::default(), Metrics::plus)
        .divide(cluster_metrics.len().max(1) as f64);
    let utility_values: Vec<f64> = cluster_values.values().map(|value| value.utility).collect();
    let regret_values: Vec<f64> = cluster_values.values().map(|value| value.regret).collect();
    let (utility_lower, utility_upper) = bootstrap_ci(
        &utility_values,
        stable_seed(seed, method, intervention_kind),
    );
    let (regret_lower, regret_upper) = bootstrap_ci(
        &regret_values,
        stable_seed(seed ^ 0xD1B5_4A32, method, intervention_kind),
    );

    json!({
        "schema_version": OUTPUT_SCHEMA,
        "record_type": "counterfactual_paired_method_contrast",
        "baseline_method": EXACT_METHOD,
        "method": method,
        "intervention_kind": intervention_kind,
        "independent_world_cluster_count": cluster_values.len(),
        "paired_task_block_count": task_block_count,
        "mean_utility_delta_vs_exact_scope": means.utility,
        "mean_regret_delta_vs_exact_scope": means.regret,
        "utility_delta_cluster_bootstrap_95pct_ci": [utility_lower, utility_upper],
        "regret_delta_cluster_bootstrap_95pct_ci": [regret_lower, regret_upper],
        "resampling_unit": "independent_world_cluster",
        "paired_task_block_aggregation": "mean_within_cluster_before_bootstrap",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "evaluation_validity": "diagnostic_only"
    })
}

fn reduce(
    input_path: &str,
    output_path: &str,
    seed: u64,
) -> Result<(usize, usize), String> {
    let input = File::open(input_path)
        .map_err(|error| format!("open evaluator input {input_path}: {error}"))?;
    let reader = BufReader::new(input);

    let mut block_rows: BTreeMap<TaskKey, BTreeMap<MethodKind, Accumulator>> = BTreeMap::new();
    let mut intervention_row_count = 0_usize;
    let mut manifest_row_count = 0_usize;
    let mut manifest_shas = BTreeSet::new();
    let mut input_provenance: Option<InputProvenance> = None;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| format!("read line {}: {error}", line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(&line)
            .map_err(|error| format!("parse JSONL line {}: {error}", line_index + 1))?;
        match record.get("record_type").and_then(Value::as_str) {
            Some("intervention_manifest") => {
                let (provenance, manifest_sha) = parse_manifest(&record)
                    .map_err(|error| format!("line {}: {error}", line_index + 1))?;
                if let Some(existing) = &input_provenance {
                    if existing != &provenance {
                        return Err("evaluator stream mixes run IDs or source/schema revisions".to_string());
                    }
                } else {
                    input_provenance = Some(provenance);
                }
                manifest_shas.insert(manifest_sha);
                manifest_row_count += 1;
                continue;
            }
            Some("evaluator_score") => {}
            Some(other) => return Err(format!("unsupported evaluator record_type {other}")),
            None => return Err(format!("line {} has no record_type", line_index + 1)),
        }

        let expected_run_id = input_provenance
            .as_ref()
            .ok_or_else(|| "evaluator_score appeared before its run manifest".to_string())?
            .run_id
            .as_str();
        if required_string(&record, "run_id")? != expected_run_id {
            return Err("evaluator_score run_id does not match manifest provenance".to_string());
        }
        let (task_key, method_kind, metrics) = parse_score(&record)
            .map_err(|error| format!("line {}: {error}", line_index + 1))?;
        block_rows
            .entry(task_key)
            .or_default()
            .entry(method_kind)
            .or_default()
            .push(metrics);
        intervention_row_count += 1;
    }

    if intervention_row_count == 0 {
        return Err("no evaluator_score records found".to_string());
    }
    if manifest_row_count == 0 || manifest_shas.is_empty() {
        return Err("evaluator stream contains no intervention manifest receipts".to_string());
    }
    let provenance = input_provenance
        .ok_or_else(|| "evaluator stream has no run provenance".to_string())?;
    let manifest_set_payload = manifest_shas.iter().cloned().collect::<Vec<_>>().join("\\n");
    let input_manifest_set_sha256 =
        hex_digest(Sha256::digest(manifest_set_payload.as_bytes()).as_slice());
    let aggregation_code_sha256 = hex_digest(
        Sha256::digest(include_bytes!("mace_counterfactual_aggregate.rs")).as_slice(),
    );

    let grouped = cluster_means(&block_rows);
    let mut output_records = Vec::new();

    for ((method, intervention_kind), cluster_blocks) in &grouped {
        let cluster_values = reduce_cluster_blocks(cluster_blocks);
        let task_block_count = block_rows
            .iter()
            .filter(|((cluster, _pair_block), rows)| {
                cluster_values.contains_key(cluster)
                    && rows.contains_key(&(method.clone(), intervention_kind.clone()))
            })
            .count();
        let row_count = block_rows
            .values()
            .filter_map(|rows| rows.get(&(method.clone(), intervention_kind.clone())))
            .map(|accumulator| accumulator.count as usize)
            .sum::<usize>();
        output_records.push(summary_record(
            method,
            intervention_kind,
            &cluster_values,
            task_block_count,
            row_count,
            seed,
        ));
    }

    // Paired method-minus-exact contrasts use only task blocks with both
    // methods present, and are aggregated to world clusters before bootstrap.
    let method_kinds: BTreeSet<String> = grouped
        .keys()
        .map(|(method, _)| method.clone())
        .filter(|method| method != EXACT_METHOD)
        .collect();
    let intervention_kinds: BTreeSet<String> = grouped
        .keys()
        .map(|(_, kind)| kind.clone())
        .collect();

    for method in method_kinds {
        for kind in &intervention_kinds {
            let key = (method.clone(), kind.clone());
            if !grouped.contains_key(&key) {
                continue;
            }
            let exact_key = (EXACT_METHOD.to_string(), kind.clone());
            if !grouped.contains_key(&exact_key) {
                continue;
            }

            let mut paired_by_cluster: BTreeMap<u64, Vec<Metrics>> = BTreeMap::new();
            let mut paired_task_block_count = 0_usize;
            for ((cluster, pair_block), methods) in &block_rows {
                let Some(treatment) = methods.get(&key).and_then(|value| value.mean()) else {
                    continue;
                };
                let Some(baseline) = methods.get(&exact_key).and_then(|value| value.mean()) else {
                    continue;
                };
                let delta = Metrics {
                    utility: treatment.utility - baseline.utility,
                    regret: treatment.regret - baseline.regret,
                    ..Metrics::default()
                };
                paired_by_cluster.entry(*cluster).or_default().push(delta);
                let _ = pair_block;
                paired_task_block_count += 1;
            }
            let paired_cluster_means = reduce_cluster_blocks(&paired_by_cluster);
            if !paired_cluster_means.is_empty() {
                output_records.push(paired_record(
                    &method,
                    kind,
                    &paired_cluster_means,
                    paired_task_block_count,
                    seed,
                ));
            }
        }
    }

    for record in &mut output_records {
        record["input_run_id"] = json!(provenance.run_id);
        record["input_manifest_set_sha256"] = json!(input_manifest_set_sha256);
        record["input_source_code_sha256"] = json!(provenance.source_code_sha256);
        record["input_fixture_sha256"] = json!(provenance.fixture_sha256);
        record["input_schema_sha256"] = json!(provenance.schema_sha256);
        record["input_source_revision"] = json!(provenance.source_revision);
        record["aggregation_code_sha256"] = json!(aggregation_code_sha256);
        record["bootstrap_seed"] = json!(seed);
        record["input_manifest_count"] = json!(manifest_row_count);
    }

    let output = File::create(output_path)
        .map_err(|error| format!("create summary output {output_path}: {error}"))?;
    let mut writer = BufWriter::new(output);
    for record in &output_records {
        serde_json::to_writer(&mut writer, record)
            .map_err(|error| format!("serialize summary record: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("write summary record: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush summary output: {error}"))?;

    eprintln!(
        "diagnostic-only MACE cluster aggregation: evaluator_rows={}, manifest_rows={}, summary_records={}",
        intervention_row_count,
        manifest_row_count,
        output_records.len()
    );
    Ok((intervention_row_count, output_records.len()))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(input_path) = args.first() else {
        eprintln!("usage: mace_counterfactual_aggregate <evaluator.jsonl> <summary.jsonl> [seed]");
        std::process::exit(2);
    };
    let output_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("mace_counterfactual_summary.jsonl");
    let seed = args
        .get(2)
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("invalid bootstrap seed: {error}");
            std::process::exit(2);
        })
        .unwrap_or(0x4d41_4345_5354_4154);

    if let Err(error) = reduce(input_path, output_path, seed) {
        eprintln!("mace_counterfactual_aggregate: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_record(
        pair_block_id: &str,
        method: &str,
        kind: &str,
        utility: f64,
        regret: f64,
    ) -> Value {
        json!({
            "schema_version": INPUT_SCHEMA,
            "record_type": "evaluator_score",
            "pair_block_id": pair_block_id,
            "method": method,
            "intervention_kind": kind,
            "utility": utility,
            "regret": regret,
            "regret_gap_vs_full_evidence": regret,
            "regret_gap_vs_full_method": regret,
            "decision_agrees_with_full": true,
            "decision_changed": false,
            "decision_is_optimal": true,
            "abstained": false
        })
    }

    #[test]
    fn parses_opaque_pair_identifier_to_world_cluster() {
        assert_eq!(
            parse_cluster("world-123:task-abcdef123456"),
            Ok(123)
        );
    }

    #[test]
    fn aggregates_intervention_rows_before_cluster_resampling() {
        let mut blocks: BTreeMap<TaskKey, BTreeMap<MethodKind, Accumulator>> = BTreeMap::new();
        for (task, utility) in [("world-1:task-a", 0.0), ("world-1:task-b", 1.0), ("world-2:task-c", 0.5)] {
            let (task_key, method_kind, metrics) = parse_score(&score_record(
                task, "typed_content_hdc", "source_root", utility, 0.0
            )).unwrap();
            blocks.entry(task_key).or_default().entry(method_kind).or_default().push(metrics);
        }

        let grouped = cluster_means(&blocks);
        let task_clusters = grouped
            .get(&("typed_content_hdc".to_string(), "source_root".to_string()))
            .unwrap();
        let cluster_means = reduce_cluster_blocks(task_clusters);
        assert_eq!(cluster_means.len(), 2);
        assert_eq!(cluster_means[&1].utility, 0.5);
        assert_eq!(cluster_means[&2].utility, 0.5);
    }

    #[test]
    fn cluster_bootstrap_is_deterministic_and_bounded_by_sample_extrema() {
        let values = vec![-2.0, 0.0, 3.0, 4.0];
        let first = bootstrap_ci(&values, 1234);
        let second = bootstrap_ci(&values, 1234);
        assert_eq!(first, second);
        assert!(first.0 >= -2.0);
        assert!(first.1 <= 4.0);
        assert!(first.0 <= first.1);
    }

    #[test]
    fn paired_contrast_requires_same_task_block_for_each_method() {
        let exact = score_record("world-1:task-a", EXACT_METHOD, "event", 0.5, 0.5);
        let typed_same = score_record("world-1:task-a", "typed_content_hdc", "event", 0.7, 0.3);
        let typed_unmatched = score_record("world-1:task-b", "typed_content_hdc", "event", 0.9, 0.1);
        assert_eq!(
            parse_score(&exact).unwrap().0,
            parse_score(&typed_same).unwrap().0
        );
        assert_ne!(
            parse_score(&exact).unwrap().0,
            parse_score(&typed_unmatched).unwrap().0
        );
    }
}
