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

fn required_u64(record: &Value, field: &str) -> Result<u64, String> {
    record
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid unsigned integer field {field}"))
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

    let pair_block_id = required_string(record, "pair_block_id")?;
    parse_cluster(pair_block_id)
        .map_err(|error| format!("invalid intervention manifest pair_block_id: {error}"))?;

    let manifest_payload_sha256 =
        required_string(record, "manifest_payload_sha256")?.to_string();
    if !valid_sha256(&manifest_payload_sha256) {
        return Err("invalid manifest_payload_sha256".to_string());
    }
    let manifest_sha256 = required_string(record, "manifest_sha256")?.to_string();
    if !valid_sha256(&manifest_sha256) {
        return Err("invalid manifest_sha256".to_string());
    }

    let public_seed = required_u64(record, "public_seed")?;
    let baseline_evidence_sha256 =
        required_string(record, "baseline_evidence_sha256")?.to_string();
    if !valid_sha256(&baseline_evidence_sha256) {
        return Err("invalid baseline_evidence_sha256 in intervention manifest".to_string());
    }
    let max_root_pairs = required_u64(record, "max_root_pairs")?;
    let declared_intervention_count = required_u64(record, "intervention_count")?;
    let interventions = record
        .get("interventions")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid interventions array".to_string())?;

    if usize::try_from(declared_intervention_count).ok() != Some(interventions.len()) {
        return Err("intervention_count does not match interventions array length".to_string());
    }

    let mut unique_intervention_ids = BTreeSet::new();
    for intervention in interventions {
        let id = required_string(intervention, "intervention_id")?;
        if !unique_intervention_ids.insert(id.to_string()) {
            return Err("intervention manifest contains duplicate intervention_id values".to_string());
        }
        let kind = required_string(intervention, "kind")?;
        if !matches!(
            kind,
            "event"
                | "source_root"
                | "source_family"
                | "unknown_root"
                | "unknown_family"
                | "source_root_pair"
                | "matched_random"
                | "no_memory"
        ) {
            return Err(format!("unsupported intervention kind {kind}"));
        }
        let _target_key = required_string(intervention, "target_key")?;
        let removed = required_event_ids(intervention, "removed_event_ids")?;
        let unique_removed: BTreeSet<u32> = removed.iter().copied().collect();
        if removed.is_empty() || unique_removed.len() != removed.len() {
            return Err("intervention removed_event_ids must be non-empty and unique".to_string());
        }
        if intervention.get("matched_to").is_none() {
            return Err("intervention is missing matched_to field".to_string());
        }
        if intervention
            .get("matched_to")
            .and_then(Value::as_str)
            .is_some_and(str::is_empty)
        {
            return Err("intervention matched_to must not be empty".to_string());
        }
    }

    let count_kind = |kind: &str| -> Result<u64, String> {
        let mut count = 0_u64;
        for intervention in interventions {
            let actual_kind = required_string(intervention, "kind")?;
            if actual_kind == kind {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| format!("intervention count overflow for {kind}"))?;
            }
        }
        Ok(count)
    };

    let actual_root_pair_count = count_kind("source_root_pair")?;
    let actual_matched_random_count = count_kind("matched_random")?;
    let root_pair_count = required_u64(record, "root_pair_count")?;
    let matched_random_count = required_u64(record, "matched_random_count")?;
    let unmatched_random_target_count =
        required_u64(record, "unmatched_random_target_count")?;

    if root_pair_count != actual_root_pair_count {
        return Err("root_pair_count does not match interventions array".to_string());
    }
    if matched_random_count != actual_matched_random_count {
        return Err("matched_random_count does not match interventions array".to_string());
    }

    // First verify the content-addressed intervention payload.
    let manifest_payload = serde_json::to_vec(&json!({
        "schema_version": INPUT_SCHEMA,
        "seed": public_seed,
        "baseline_evidence_sha256": baseline_evidence_sha256,
        "max_root_pairs": max_root_pairs,
        "root_pair_count": root_pair_count,
        "matched_random_count": matched_random_count,
        "unmatched_random_target_count": unmatched_random_target_count,
        "interventions": interventions
    }))
    .map_err(|error| format!("serialize intervention manifest payload: {error}"))?;
    let computed_payload_sha256 =
        hex_digest(Sha256::digest(manifest_payload).as_slice());
    if computed_payload_sha256 != manifest_payload_sha256 {
        return Err("intervention manifest payload digest does not match canonical payload".to_string());
    }

    // Then bind that payload to its exact run, task block and source provenance.
    let receipt_payload = serde_json::to_vec(&json!({
        "schema_version": INPUT_SCHEMA,
        "run_id": provenance.run_id.as_str(),
        "pair_block_id": pair_block_id,
        "manifest_payload_sha256": manifest_payload_sha256,
        "source_code_sha256": provenance.source_code_sha256.as_str(),
        "fixture_sha256": provenance.fixture_sha256.as_str(),
        "schema_sha256": provenance.schema_sha256.as_str(),
        "source_revision": provenance.source_revision.as_deref()
    }))
    .map_err(|error| format!("serialize intervention manifest receipt: {error}"))?;
    let computed_receipt_sha256 =
        hex_digest(Sha256::digest(receipt_payload).as_slice());
    if computed_receipt_sha256 != manifest_sha256 {
        return Err("intervention manifest receipt digest does not match run/pair-block provenance".to_string());
    }

    Ok((provenance, manifest_sha256))
}
fn required_event_ids(record: &Value, field: &str) -> Result<Vec<u32>, String> {
    let values = record
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid event ID array {field}"))?;
    values
        .iter()
        .map(|value| {
            let id = value
                .as_u64()
                .ok_or_else(|| format!("event ID array {field} contains a non-unsigned integer"))?;
            u32::try_from(id)
                .map_err(|_| format!("event ID array {field} contains an out-of-range ID"))
        })
        .collect()
}

fn validate_score_against_manifest(score: &Value, manifest: &Value) -> Result<(), String> {
    let candidate_observation_id = required_string(score, "candidate_observation_id")?;
    if candidate_observation_id.len() > 512 {
        return Err("candidate_observation_id exceeds 512 characters".to_string());
    }
    let candidate_input_sha256 = required_string(score, "candidate_input_sha256")?;
    if !valid_sha256(candidate_input_sha256) {
        return Err("invalid candidate_input_sha256 in evaluator score".to_string());
    }

    let score_intervention_id = required_string(score, "intervention_id")?;
    let score_kind = required_string(score, "intervention_kind")?;
    let score_method = required_string(score, "method")?;
    let score_target = required_string(score, "target_key")?;
    let score_removed = required_event_ids(score, "removed_event_ids")?;
    let declared_removed_count = required_u64(score, "removed_event_count")?;
    if usize::try_from(declared_removed_count).ok() != Some(score_removed.len()) {
        return Err("removed_event_count does not match removed_event_ids".to_string());
    }
    let unique_score_removed: BTreeSet<u32> = score_removed.iter().copied().collect();
    if unique_score_removed.len() != score_removed.len() {
        return Err("score removed_event_ids contains duplicate IDs".to_string());
    }
    let score_matched_to = score
        .get("matched_to")
        .cloned()
        .unwrap_or(Value::Null);

    // Full-evidence reference rows have no knockout entry in the manifest.
    // Keep this exception narrow and deterministic.
    if score_kind == "reference" {
        let expected_id = format!("full_evidence_{score_method}");
        if score_intervention_id != expected_id
            || score_target != score_method
            || !score_removed.is_empty()
            || !score_matched_to.is_null()
        {
            return Err("invalid full-evidence reference row contract".to_string());
        }
        return Ok(());
    }

    let interventions = manifest
        .get("interventions")
        .and_then(Value::as_array)
        .ok_or_else(|| "validated manifest has no interventions array".to_string())?;

    let declared = interventions
        .iter()
        .find(|item| {
            item.get("intervention_id")
                .and_then(Value::as_str)
                == Some(score_intervention_id)
        })
        .ok_or_else(|| "evaluator score intervention_id is absent from pair-block manifest".to_string())?;

    let declared_kind = required_string(declared, "kind")?;
    let declared_target = required_string(declared, "target_key")?;
    let declared_removed = required_event_ids(declared, "removed_event_ids")?;
    let declared_matched_to = declared
        .get("matched_to")
        .cloned()
        .unwrap_or(Value::Null);

    if score_kind != declared_kind {
        return Err("evaluator score intervention_kind differs from manifest".to_string());
    }
    if score_target != declared_target {
        return Err("evaluator score target_key differs from manifest".to_string());
    }
    if score_removed != declared_removed {
        return Err("evaluator score removed_event_ids differ from manifest".to_string());
    }
    if score_matched_to != declared_matched_to {
        return Err("evaluator score matched_to differs from manifest".to_string());
    }

    Ok(())
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
        "world_cluster_count": cluster_values.len(),
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
        "world_cluster_count": cluster_values.len(),
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
    let mut manifest_pair_blocks = BTreeSet::new();
    let mut manifest_records: BTreeMap<String, Value> = BTreeMap::new();
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
                let manifest_pair_block =
                    required_string(&record, "pair_block_id")?.to_string();
                if !manifest_pair_blocks.insert(manifest_pair_block.clone()) {
                    return Err("duplicate intervention manifest pair_block_id".to_string());
                }
                manifest_records.insert(manifest_pair_block, record.clone());
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
        let manifest = manifest_records
            .get(&task_key.1)
            .ok_or_else(|| "evaluator_score has no matching intervention manifest".to_string())?;
        validate_score_against_manifest(&record, manifest)
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
    let manifest_set_payload = manifest_shas.iter().cloned().collect::<Vec<_>>().join("\n");
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
            for ((cluster, _pair_block), methods) in &block_rows {
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
        record["input_run_id"] = json!(provenance.run_id.as_str());
        record["input_manifest_set_sha256"] = json!(input_manifest_set_sha256.as_str());
        record["input_source_code_sha256"] = json!(provenance.source_code_sha256.as_str());
        record["input_fixture_sha256"] = json!(provenance.fixture_sha256.as_str());
        record["input_schema_sha256"] = json!(provenance.schema_sha256.as_str());
        record["input_source_revision"] = json!(provenance.source_revision.as_deref());
        record["aggregation_code_sha256"] = json!(aggregation_code_sha256.as_str());
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

    fn refresh_manifest_digests(record: &mut Value) {
        let baseline_evidence_sha256 = record["baseline_evidence_sha256"]
            .as_str()
            .unwrap()
            .to_string();
        let public_seed = record["public_seed"].as_u64().unwrap();
        let max_root_pairs = record["max_root_pairs"].as_u64().unwrap();
        let root_pair_count = record["root_pair_count"].as_u64().unwrap();
        let matched_random_count = record["matched_random_count"].as_u64().unwrap();
        let unmatched_random_target_count =
            record["unmatched_random_target_count"].as_u64().unwrap();

        let payload = serde_json::to_vec(&json!({
            "schema_version": INPUT_SCHEMA,
            "seed": public_seed,
            "baseline_evidence_sha256": baseline_evidence_sha256,
            "max_root_pairs": max_root_pairs,
            "root_pair_count": root_pair_count,
            "matched_random_count": matched_random_count,
            "unmatched_random_target_count": unmatched_random_target_count,
            "interventions": record["interventions"]
        }))
        .unwrap();
        let payload_sha256 = hex_digest(Sha256::digest(payload).as_slice());
        record["manifest_payload_sha256"] = json!(payload_sha256);

        let receipt_payload = serde_json::to_vec(&json!({
            "schema_version": INPUT_SCHEMA,
            "run_id": record["run_id"],
            "pair_block_id": record["pair_block_id"],
            "manifest_payload_sha256": payload_sha256,
            "source_code_sha256": record["source_code_sha256"],
            "fixture_sha256": record["fixture_sha256"],
            "schema_sha256": record["schema_sha256"],
            "source_revision": record["source_revision"]
        }))
        .unwrap();
        record["manifest_sha256"] =
            json!(hex_digest(Sha256::digest(receipt_payload).as_slice()));
    }

    fn manifest_record(run_id: &str, source_sha: &str) -> Value {
        let baseline_evidence_sha256 = "a".repeat(64);
        let public_seed = 17_u64;
        let max_root_pairs = 8_u64;
        let root_pair_count = 0_u64;
        let matched_random_count = 0_u64;
        let unmatched_random_target_count = 0_u64;
        let interventions = Vec::<Value>::new();
        let mut record = json!({
            "schema_version": INPUT_SCHEMA,
            "record_type": "intervention_manifest",
            "run_id": run_id,
            "pair_block_id": "world-1:task-manifest-test",
            "manifest_payload_sha256": "0".repeat(64),
            "manifest_sha256": "0".repeat(64),
            "baseline_evidence_sha256": baseline_evidence_sha256,
            "public_seed": public_seed,
            "intervention_count": 0,
            "interventions": interventions,
            "max_root_pairs": max_root_pairs,
            "root_pair_count": root_pair_count,
            "matched_random_count": matched_random_count,
            "unmatched_random_target_count": unmatched_random_target_count,
            "source_code_sha256": source_sha,
            "fixture_sha256": "b".repeat(64),
            "schema_sha256": "c".repeat(64),
            "source_revision": "d".repeat(40)
        });
        refresh_manifest_digests(&mut record);
        record
    }

    fn append_intervention(record: &mut Value, intervention: Value) {
        let mut interventions = record["interventions"]
            .as_array()
            .unwrap()
            .clone();
        interventions.push(intervention);

        let root_pair_count = interventions
            .iter()
            .filter(|item| item["kind"] == "source_root_pair")
            .count() as u64;
        let matched_random_count = interventions
            .iter()
            .filter(|item| item["kind"] == "matched_random")
            .count() as u64;

        record["intervention_count"] = json!(interventions.len());
        record["interventions"] = json!(interventions);
        record["root_pair_count"] = json!(root_pair_count);
        record["matched_random_count"] = json!(matched_random_count);
        refresh_manifest_digests(record);
    }

    fn linked_score(intervention_id: &str, kind: &str, target: &str, removed: Value) -> Value {
        let removed_event_count = removed.as_array().unwrap().len();
        json!({
            "candidate_observation_id": "observation-1",
            "candidate_input_sha256": "f".repeat(64),
            "intervention_id": intervention_id,
            "intervention_kind": kind,
            "method": "typed_content_hdc",
            "target_key": target,
            "removed_event_ids": removed,
            "removed_event_count": removed_event_count,
            "matched_to": null
        })
    }

    #[test]
    fn score_row_must_match_the_digest_verified_manifest_intervention() {
        let mut manifest = manifest_record("run-1", &"a".repeat(64));
        append_intervention(&mut manifest, json!({
            "intervention_id": "event:event:1:1",
            "kind": "event",
            "target_key": "event:1",
            "removed_event_ids": [1],
            "matched_to": null
        }));
        parse_manifest(&manifest).unwrap();

        let score = linked_score("event:event:1:1", "event", "event:1", json!([1]));
        assert!(validate_score_against_manifest(&score, &manifest).is_ok());
    }

    #[test]
    fn score_row_rejects_unknown_intervention_id() {
        let manifest = manifest_record("run-1", &"a".repeat(64));
        let score = linked_score("event:missing", "event", "event:missing", json!([9]));
        let error = validate_score_against_manifest(&score, &manifest).unwrap_err();
        assert!(error.contains("absent from pair-block manifest"));
    }

    #[test]
    fn score_row_rejects_mutated_removed_event_set() {
        let mut manifest = manifest_record("run-1", &"a".repeat(64));
        append_intervention(&mut manifest, json!({
            "intervention_id": "event:event:1:1",
            "kind": "event",
            "target_key": "event:1",
            "removed_event_ids": [1],
            "matched_to": null
        }));
        let score = linked_score("event:event:1:1", "event", "event:1", json!([2]));
        let error = validate_score_against_manifest(&score, &manifest).unwrap_err();
        assert!(error.contains("removed_event_ids differ"));
    }

    #[test]
    fn score_row_rejects_kind_and_target_mismatch() {
        let mut manifest = manifest_record("run-1", &"a".repeat(64));
        append_intervention(&mut manifest, json!({
            "intervention_id": "event:event:1:1",
            "kind": "event",
            "target_key": "event:1",
            "removed_event_ids": [1],
            "matched_to": null
        }));
        let wrong_kind = linked_score("event:event:1:1", "source_root", "event:1", json!([1]));
        assert!(validate_score_against_manifest(&wrong_kind, &manifest).unwrap_err().contains("intervention_kind"));
        let wrong_target = linked_score("event:event:1:1", "event", "event:2", json!([1]));
        assert!(validate_score_against_manifest(&wrong_target, &manifest).unwrap_err().contains("target_key"));
    }

    #[test]
    fn reference_score_rows_are_limited_to_matching_full_evidence_contract() {
        let manifest = manifest_record("run-1", &"a".repeat(64));
        let valid = linked_score(
            "full_evidence_typed_content_hdc",
            "reference",
            "typed_content_hdc",
            json!([])
        );
        assert!(validate_score_against_manifest(&valid, &manifest).is_ok());

        let invalid = linked_score(
            "full_evidence_event_id_key_value_topk",
            "reference",
            "typed_content_hdc",
            json!([])
        );
        assert!(validate_score_against_manifest(&invalid, &manifest).is_err());
    }

    #[test]
    fn manifest_provenance_and_canonical_payload_digest_are_verified() {
        let valid = manifest_record("run-1", &"a".repeat(64));
        let (provenance, manifest_sha) = parse_manifest(&valid).unwrap();
        assert_eq!(provenance.run_id, "run-1");
        assert_eq!(provenance.source_revision.as_deref().map(str::len), Some(40));
        assert!(valid_sha256(&manifest_sha));

        let invalid = manifest_record("run-1", "not-a-sha");
        assert!(parse_manifest(&invalid).is_err());
    }

    #[test]
    fn manifest_rejects_intervention_mutation_with_stale_digest() {
        let mut record = manifest_record("run-1", &"a".repeat(64));
        record["intervention_count"] = json!(1);
        record["interventions"] = json!([{
            "intervention_id": "event:1",
            "kind": "event",
            "target_key": "event:1",
            "removed_event_ids": [1],
            "matched_to": null
        }]);
        let error = parse_manifest(&record).unwrap_err();
        assert!(error.contains("digest does not match"));
    }

    #[test]
    fn manifest_rejects_intervention_count_mismatch() {
        let mut record = manifest_record("run-1", &"a".repeat(64));
        record["intervention_count"] = json!(1);
        let error = parse_manifest(&record).unwrap_err();
        assert!(error.contains("intervention_count"));
    }

    #[test]
    fn manifest_rejects_root_pair_count_mismatch() {
        let mut record = manifest_record("run-1", &"a".repeat(64));
        record["root_pair_count"] = json!(1);
        let error = parse_manifest(&record).unwrap_err();
        assert!(error.contains("root_pair_count"));
    }

    #[test]
    fn mixed_run_provenance_is_not_equal() {
        let first = parse_manifest(&manifest_record("run-1", &"a".repeat(64)))
            .unwrap()
            .0;
        let second = parse_manifest(&manifest_record("run-2", &"a".repeat(64)))
            .unwrap()
            .0;
        assert_ne!(first, second);
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
