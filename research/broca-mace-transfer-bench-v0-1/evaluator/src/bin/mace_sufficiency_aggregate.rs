// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cluster-paired aggregation for the MACE decision-sufficiency frontier.
//!
//! Scenario families are averaged within each independent world cluster before
//! paired contrasts or bootstrap resampling. The number of scenario rows is not
//! treated as the inferential sample size. Outputs remain diagnostic-only:
//! candidate/evaluator custody is not independent and evidence lineage is absent.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const INPUT_SCHEMA: &str = "mace_sufficiency_frontier_v0_1";
const OUTPUT_SCHEMA: &str = "mace_sufficiency_aggregate_v0_1";
const BOOTSTRAP_REPLICATES: usize = 2_000;
const NO_MEMORY: &str = "no_memory";
const ALLOWED_METHODS: [&str; 6] = [
    "no_memory",
    "full_scope_reference",
    "event_id_ordered_control",
    "random_control",
    "static_hdc",
    "typed_content_hdc",
];

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ArmBudget {
    method: String,
    per_action_top_k: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RunProvenance {
    run_id: String,
    public_seed: u64,
    source_revision: Option<String>,
    source_code_sha256: String,
    fixture_sha256: String,
    schema_sha256: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct PointMetrics {
    utility: f64,
    absolute_regret: f64,
    regret_gap_vs_full_scope: f64,
    decision_sufficient_by_regret_only: f64,
    oracle_optimal: f64,
    agrees_with_full_scope: f64,
    selected_bytes_lower_bound: f64,
    index_bytes_lower_bound: f64,
    measured_latency_ns: f64,
}

impl PointMetrics {
    fn plus(self, other: Self) -> Self {
        Self {
            utility: self.utility + other.utility,
            absolute_regret: self.absolute_regret + other.absolute_regret,
            regret_gap_vs_full_scope: self.regret_gap_vs_full_scope
                + other.regret_gap_vs_full_scope,
            decision_sufficient_by_regret_only: self.decision_sufficient_by_regret_only
                + other.decision_sufficient_by_regret_only,
            oracle_optimal: self.oracle_optimal + other.oracle_optimal,
            agrees_with_full_scope: self.agrees_with_full_scope + other.agrees_with_full_scope,
            selected_bytes_lower_bound: self.selected_bytes_lower_bound
                + other.selected_bytes_lower_bound,
            index_bytes_lower_bound: self.index_bytes_lower_bound
                + other.index_bytes_lower_bound,
            measured_latency_ns: self.measured_latency_ns + other.measured_latency_ns,
        }
    }

    fn divide(self, denominator: f64) -> Self {
        Self {
            utility: self.utility / denominator,
            absolute_regret: self.absolute_regret / denominator,
            regret_gap_vs_full_scope: self.regret_gap_vs_full_scope / denominator,
            decision_sufficient_by_regret_only: self.decision_sufficient_by_regret_only
                / denominator,
            oracle_optimal: self.oracle_optimal / denominator,
            agrees_with_full_scope: self.agrees_with_full_scope / denominator,
            selected_bytes_lower_bound: self.selected_bytes_lower_bound / denominator,
            index_bytes_lower_bound: self.index_bytes_lower_bound / denominator,
            measured_latency_ns: self.measured_latency_ns / denominator,
        }
    }

}

#[derive(Clone, Copy, Debug)]
struct ParetoCost {
    selected_bytes: f64,
    index_bytes: f64,
    latency_ns: f64,
    regret: f64,
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

fn required_u64(record: &Value, field: &str) -> Result<u64, String> {
    record
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid unsigned integer field {field}"))
}

fn required_bool(record: &Value, field: &str) -> Result<bool, String> {
    record
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or invalid boolean field {field}"))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_provenance(record: &Value) -> Result<RunProvenance, String> {
    if required_string(record, "schema_version")? != INPUT_SCHEMA {
        return Err("unsupported frontier evaluator schema version".into());
    }
    let source_revision = match record.get("source_revision") {
        Some(Value::Null) | None => None,
        Some(value) => Some(
            value
                .as_str()
                .ok_or("source_revision must be string or null")?
                .to_owned(),
        ),
    };
    if source_revision.as_deref().is_some_and(|revision| {
        !matches!(revision.len(), 40 | 64)
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err("source_revision is not a lowercase Git SHA".into());
    }
    let provenance = RunProvenance {
        run_id: required_string(record, "run_id")?.to_owned(),
        public_seed: required_u64(record, "public_seed")?,
        source_revision,
        source_code_sha256: required_string(record, "source_code_sha256")?.to_owned(),
        fixture_sha256: required_string(record, "fixture_sha256")?.to_owned(),
        schema_sha256: required_string(record, "schema_sha256")?.to_owned(),
    };
    for (field, value) in [
        ("source_code_sha256", provenance.source_code_sha256.as_str()),
        ("fixture_sha256", provenance.fixture_sha256.as_str()),
        ("schema_sha256", provenance.schema_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(format!("{field} must be canonical lowercase SHA-256"));
        }
    }
    Ok(provenance)
}

fn parse_arm(record: &Value) -> Result<ArmBudget, String> {
    let method = required_string(record, "method")?.to_owned();
    if !ALLOWED_METHODS.contains(&method.as_str()) {
        return Err(format!("unsupported retrieval method {method}"));
    }
    let top_k = match record.get("per_action_top_k") {
        Some(Value::Null) | None => None,
        Some(value) => {
            let raw = value
                .as_u64()
                .ok_or("per_action_top_k must be unsigned integer or null")?;
            let value = usize::try_from(raw).map_err(|_| "per_action_top_k overflows usize")?;
            if ![1, 2, 4, 8].contains(&value) {
                return Err(format!("unsupported per-action budget {value}"));
            }
            Some(value)
        }
    };
    match method.as_str() {
        NO_MEMORY | "full_scope_reference" if top_k.is_some() => {
            return Err(format!("{method} must not have a top-k budget"));
        }
        "event_id_ordered_control" | "random_control" | "static_hdc" | "typed_content_hdc"
            if top_k.is_none() =>
        {
            return Err(format!("{method} requires a top-k budget"));
        }
        _ => {}
    }
    Ok(ArmBudget {
        method,
        per_action_top_k: top_k,
    })
}

fn parse_point(record: &Value) -> Result<(u64, String, ArmBudget, PointMetrics), String> {
    if required_string(record, "record_type")? != "evaluator_score" {
        return Err("expected evaluator_score record".into());
    }
    let cluster = required_u64(record, "world_cluster")?;
    if cluster == 0 {
        return Err("world_cluster must be positive".into());
    }
    let family = required_string(record, "scenario_family")?.to_owned();
    let arm = parse_arm(record)?;

    if required_string(record, "evidence_qualification")?
        != "diagnostic_only_missing_authenticated_lineage"
        || required_bool(record, "evidence_qualified_sufficient")?
    {
        return Err("frontier aggregate accepts only diagnostic, non-evidence-qualified rows".into());
    }

    let selected_bytes = required_u64(record, "selected_experience_struct_bytes_lower_bound")?;
    let index_bytes = required_u64(record, "index_payload_bytes_lower_bound")?;
    let latency = required_u64(record, "measured_index_and_selection_ns")?;
    let sufficient = required_bool(record, "decision_sufficient_by_regret_only")?;
    let oracle_optimal = required_bool(record, "oracle_optimal")?;
    let agrees = required_bool(record, "decision_agrees_with_full_scope_reference")?;

    let metrics = PointMetrics {
        utility: required_number(record, "utility")?,
        absolute_regret: required_number(record, "absolute_regret")?,
        regret_gap_vs_full_scope: required_number(record, "regret_gap_vs_full_scope_reference")?,
        decision_sufficient_by_regret_only: f64::from(u8::from(sufficient)),
        oracle_optimal: f64::from(u8::from(oracle_optimal)),
        agrees_with_full_scope: f64::from(u8::from(agrees)),
        selected_bytes_lower_bound: selected_bytes as f64,
        index_bytes_lower_bound: index_bytes as f64,
        measured_latency_ns: latency as f64,
    };
    if metrics.absolute_regret < 0.0 || metrics.utility < 0.0 || metrics.utility > 1.0 {
        return Err(format!("invalid utility/regret on cluster {cluster}, family {family}"));
    }
    Ok((cluster, family, arm, metrics))
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

fn stable_seed(seed: u64, method: &str, budget: Option<usize>, metric: &str) -> u64 {
    let key = format!("{method}:{budget:?}:{metric}");
    let digest = Sha256::digest(format!("{seed}:{key}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is eight bytes")).max(1)
}

fn bootstrap_paired_ci(values: &[f64], seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    if values.len() == 1 {
        return (values[0], values[0]);
    }
    let mut state = seed.max(1);
    let mut means = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut sum = 0.0;
        for _ in 0..values.len() {
            let index = (next_random(&mut state) % values.len() as u64) as usize;
            sum += values[index];
        }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    (percentile(&means, 0.025), percentile(&means, 0.975))
}

fn cluster_means(
    values: &BTreeMap<ArmBudget, BTreeMap<u64, BTreeMap<String, PointMetrics>>>,
    arm: &ArmBudget,
) -> Result<BTreeMap<u64, PointMetrics>, String> {
    let clusters = values
        .get(arm)
        .ok_or_else(|| format!("missing arm {} {:?}", arm.method, arm.per_action_top_k))?;
    let mut output = BTreeMap::new();
    for (cluster, families) in clusters {
        if families.is_empty() {
            return Err(format!("cluster {cluster} has no scenario families"));
        }
        let total = families.values().copied().fold(PointMetrics::default(), PointMetrics::plus);
        let mean = total.divide(families.len() as f64);
        if !mean.is_finite() {
            return Err(format!("cluster {cluster} has non-finite aggregated metrics"));
        }
        output.insert(*cluster, mean);
    }
    Ok(output)
}

fn paired_cluster_deltas(
    treatment: &BTreeMap<u64, PointMetrics>,
    baseline: &BTreeMap<u64, PointMetrics>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if treatment.keys().copied().collect::<BTreeSet<_>>()
        != baseline.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err("paired contrast requires identical world-cluster sets".into());
    }
    let mut utility_deltas = Vec::with_capacity(treatment.len());
    let mut regret_deltas = Vec::with_capacity(treatment.len());
    for (cluster, value) in treatment {
        let reference = baseline
            .get(cluster)
            .ok_or_else(|| format!("no_memory baseline missing cluster {cluster}"))?;
        utility_deltas.push(value.utility - reference.utility);
        regret_deltas.push(value.absolute_regret - reference.absolute_regret);
    }
    Ok((utility_deltas, regret_deltas))
}

fn dominates(left: ParetoCost, right: ParetoCost) -> bool {
    let no_worse = left.selected_bytes <= right.selected_bytes
        && left.index_bytes <= right.index_bytes
        && left.latency_ns <= right.latency_ns
        && left.regret <= right.regret;
    let strictly_better = left.selected_bytes < right.selected_bytes
        || left.index_bytes < right.index_bytes
        || left.latency_ns < right.latency_ns
        || left.regret < right.regret;
    no_worse && strictly_better
}

fn write_jsonl(path: &str, records: &[Value]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create {path}: {error}"))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)
            .map_err(|error| format!("serialize {path}: {error}"))?;
        writer.write_all(b"\n").map_err(|error| format!("write {path}: {error}"))?;
    }
    writer.flush().map_err(|error| format!("flush {path}: {error}"))
}

fn reduce(input_path: &str, output_path: &str, seed: u64) -> Result<(), String> {
    let input_contents = std::fs::read_to_string(input_path)
        .map_err(|error| format!("read {input_path}: {error}"))?;
    let input_evaluator_jsonl_sha256 =
        hex_digest(Sha256::digest(input_contents.as_bytes()).as_slice());
    let mut provenance: Option<RunProvenance> = None;
    let mut rows: BTreeMap<ArmBudget, BTreeMap<u64, BTreeMap<String, PointMetrics>>> =
        BTreeMap::new();

    for (index, line) in input_contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(line)
            .map_err(|error| format!("parse line {}: {error}", index + 1))?;
        let current_provenance = parse_provenance(&record)?;
        if let Some(expected) = &provenance {
            if expected != &current_provenance {
                return Err("evaluator stream mixes run IDs or source/schema revisions".into());
            }
        } else {
            provenance = Some(current_provenance);
        }
        let (cluster, family, arm, metrics) = parse_point(&record)?;
        let family_map = rows.entry(arm).or_default().entry(cluster).or_default();
        if family_map.insert(family.clone(), metrics).is_some() {
            return Err(format!(
                "duplicate evaluator row for cluster {cluster}, family {family}"
            ));
        }
    }

    if rows.is_empty() {
        return Err("no evaluator score records found".into());
    }
    let provenance = provenance.ok_or("evaluator stream had no provenance")?;
    let baseline_arm = ArmBudget {
        method: NO_MEMORY.into(),
        per_action_top_k: None,
    };
    let baseline_clusters = rows
        .get(&baseline_arm)
        .ok_or("no_memory baseline missing")?;
    if baseline_clusters.is_empty() {
        return Err("no_memory baseline has no clusters".into());
    }
    let clusters: BTreeSet<u64> = baseline_clusters.keys().copied().collect();
    let expected_families: BTreeMap<u64, BTreeSet<String>> = baseline_clusters
        .iter()
        .map(|(cluster, families)| (*cluster, families.keys().cloned().collect()))
        .collect();
    if expected_families.values().any(BTreeSet::is_empty) {
        return Err("no_memory baseline has an empty scenario-family block".into());
    }
    let canonical_family_set = expected_families
        .values()
        .next()
        .cloned()
        .ok_or("no_memory baseline has no scenario-family set")?;
    if expected_families.values().any(|families| families != &canonical_family_set) {
        return Err("world clusters do not share the same scenario-family set".into());
    }

    for (arm, cluster_map) in &rows {
        if cluster_map.keys().copied().collect::<BTreeSet<_>>() != clusters {
            return Err(format!(
                "arm {} {:?} does not cover the same world clusters as no_memory",
                arm.method, arm.per_action_top_k
            ));
        }
        for cluster in &clusters {
            let family_set: BTreeSet<String> = cluster_map
                .get(cluster)
                .ok_or("required cluster missing")?
                .keys()
                .cloned()
                .collect();
            if family_set != expected_families[cluster] {
                return Err(format!(
                    "arm {} {:?} has missing/extra scenario families in cluster {cluster}",
                    arm.method, arm.per_action_top_k
                ));
            }
        }
    }

    let baseline_means = cluster_means(&rows, &baseline_arm)?;
    let mut summaries = Vec::<(
        ArmBudget,
        PointMetrics,
        f64,
        f64,
        (f64, f64),
        (f64, f64),
        usize,
    )>::new();
    for arm in rows.keys() {
        let means = cluster_means(&rows, arm)?;
        let (utility_deltas, regret_deltas) =
            paired_cluster_deltas(&means, &baseline_means)?;
        let mut sum = PointMetrics::default();
        for cluster in &clusters {
            let current = means.get(cluster).ok_or("current cluster mean missing")?;
            sum = sum.plus(*current);
        }
        let mean = sum.divide(clusters.len() as f64);
        let utility_ci = bootstrap_paired_ci(
            &utility_deltas,
            stable_seed(seed, &arm.method, arm.per_action_top_k, "utility_delta_vs_no_memory"),
        );
        let regret_ci = bootstrap_paired_ci(
            &regret_deltas,
            stable_seed(seed, &arm.method, arm.per_action_top_k, "regret_delta_vs_no_memory"),
        );
        let mean_utility_delta = utility_deltas.iter().sum::<f64>() / clusters.len() as f64;
        let mean_regret_delta = regret_deltas.iter().sum::<f64>() / clusters.len() as f64;
        summaries.push((
            arm.clone(),
            mean,
            mean_utility_delta,
            mean_regret_delta,
            utility_ci,
            regret_ci,
            clusters.len(),
        ));
    }

    let costs: Vec<ParetoCost> = summaries
        .iter()
        .map(|(_, mean, _, _, _, _, _)| ParetoCost {
            selected_bytes: mean.selected_bytes_lower_bound,
            index_bytes: mean.index_bytes_lower_bound,
            latency_ns: mean.measured_latency_ns,
            regret: mean.absolute_regret,
        })
        .collect();
    let summary_schema_sha256 = hex_digest(
        Sha256::digest(include_bytes!(
            "../../../../../data/schemas/mace_sufficiency_aggregate_v0_1.schema.json"
        ))
        .as_slice(),
    );
    let mut aggregation_hasher = Sha256::new();
    aggregation_hasher.update(include_bytes!("mace_sufficiency_aggregate.rs"));
    aggregation_hasher.update(include_bytes!(
        "../../../../../data/schemas/mace_sufficiency_aggregate_v0_1.schema.json"
    ));
    let aggregation_source_sha256 = hex_digest(aggregation_hasher.finalize().as_slice());
    let mut output = Vec::new();
    for (index, (arm, mean, mean_utility_delta, mean_regret_delta, utility_ci, regret_ci, cluster_count))
        in summaries.iter().enumerate()
    {
        let frontier = !(0..costs.len()).any(|other| other != index && dominates(costs[other], costs[index]));
        let baseline = arm.method == NO_MEMORY && arm.per_action_top_k.is_none();
        output.push(json!({
            "schema_version": OUTPUT_SCHEMA,
            "record_type": "cluster_paired_frontier_summary",
            "input_run_id": provenance.run_id,
            "input_public_seed": provenance.public_seed,
            "input_source_revision": provenance.source_revision,
            "input_source_code_sha256": provenance.source_code_sha256,
            "input_fixture_sha256": provenance.fixture_sha256,
            "input_schema_sha256": provenance.schema_sha256,
            "input_evaluator_jsonl_sha256": input_evaluator_jsonl_sha256,
            "aggregation_source_sha256": aggregation_source_sha256,
            "summary_schema_sha256": summary_schema_sha256,
            "bootstrap_seed": seed,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "independent_world_cluster",
            "within_cluster_aggregation": "equal_weight_mean_over_scenario_families",
            "world_cluster_count": cluster_count,
            "scenario_family_count_per_cluster": expected_families.values().next().map(|x| x.len()).unwrap_or(0),
            "method": arm.method,
            "per_action_top_k": arm.per_action_top_k,
            "mean_utility": mean.utility,
            "mean_absolute_regret": mean.absolute_regret,
            "mean_regret_gap_vs_full_scope_reference": mean.regret_gap_vs_full_scope,
            "regret_only_epsilon_sufficiency_rate": mean.decision_sufficient_by_regret_only,
            "oracle_optimal_rate": mean.oracle_optimal,
            "agreement_with_full_scope_rate": mean.agrees_with_full_scope,
            "mean_selected_experience_struct_bytes_lower_bound": mean.selected_bytes_lower_bound,
            "mean_index_payload_bytes_lower_bound": mean.index_bytes_lower_bound,
            "mean_index_and_selection_latency_ns": mean.measured_latency_ns,
            "mean_utility_delta_vs_no_memory": mean_utility_delta,
            "mean_regret_delta_vs_no_memory": mean_regret_delta,
            "utility_delta_vs_no_memory_cluster_bootstrap_95pct_ci": [utility_ci.0, utility_ci.1],
            "regret_delta_vs_no_memory_cluster_bootstrap_95pct_ci": [regret_ci.0, regret_ci.1],
            "paired_contrast_baseline": "no_memory",
            "multiple_comparison_status": "exploratory_no_multiplicity_correction",
            "pareto_frontier_member": frontier,
            "evaluation_validity": "diagnostic_only",
            "evidence_qualified_sufficiency": false,
            "inference_note": if *cluster_count < 10 { "descriptive_only_fewer_than_10_independent_world_clusters" } else { "cluster_bootstrap_95pct_interval" },
            "baseline_arm": baseline
        }));
    }
    write_jsonl(output_path, &output)?;
    eprintln!(
        "diagnostic-only MACE sufficiency aggregation: clusters={}, arms={}, records={}",
        clusters.len(),
        rows.len(),
        output.len()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(input) = args.first() else {
        eprintln!("usage: mace_sufficiency_aggregate <evaluator.jsonl> [summary.jsonl] [seed]");
        std::process::exit(2);
    };
    let output = args.get(1).map(String::as_str).unwrap_or("mace_sufficiency_summary.jsonl");
    let seed = args
        .get(2)
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("invalid bootstrap seed: {error}");
            std::process::exit(2);
        })
        .unwrap_or(0x4D41_4345_4147_4731);
    if let Err(error) = reduce(input, output, seed) {
        eprintln!("mace_sufficiency_aggregate: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(regret: f64, utility: f64) -> PointMetrics {
        PointMetrics {
            absolute_regret: regret,
            utility,
            ..PointMetrics::default()
        }
    }

    #[test]
    fn cluster_average_gives_each_scenario_family_equal_weight() {
        let arm = ArmBudget { method: NO_MEMORY.into(), per_action_top_k: None };
        let values = BTreeMap::from([(
            arm.clone(),
            BTreeMap::from([(
                1,
                BTreeMap::from([
                    ("rare".to_string(), metric(1.0, 0.0)),
                    ("common".to_string(), metric(0.0, 1.0)),
                ]),
            )]),
        )]);
        let means = cluster_means(&values, &arm).unwrap();
        assert_eq!(means[&1].absolute_regret, 0.5);
        assert_eq!(means[&1].utility, 0.5);
    }

    #[test]
    fn cluster_bootstrap_is_seeded_and_deterministic() {
        let values = [-0.2, 0.0, 0.3, 0.5];
        let first = bootstrap_paired_ci(&values, 17);
        let second = bootstrap_paired_ci(&values, 17);
        assert_eq!(first, second);
        assert!(first.0 <= first.1);
        assert!(first.0 >= -0.2);
        assert!(first.1 <= 0.5);
    }

    #[test]
    fn paired_cluster_deltas_match_worlds_and_report_regret_direction() {
        let treatment = BTreeMap::from([
            (1, PointMetrics { utility: 0.8, absolute_regret: 0.2, ..PointMetrics::default() }),
            (2, PointMetrics { utility: 0.7, absolute_regret: 0.3, ..PointMetrics::default() }),
        ]);
        let baseline = BTreeMap::from([
            (1, PointMetrics { utility: 0.5, absolute_regret: 0.5, ..PointMetrics::default() }),
            (2, PointMetrics { utility: 0.6, absolute_regret: 0.4, ..PointMetrics::default() }),
        ]);
        let (utility, regret) = paired_cluster_deltas(&treatment, &baseline).unwrap();
        assert!((utility[0] - 0.3).abs() < 1e-12);
        assert!((utility[1] - 0.1).abs() < 1e-12);
        assert!((regret[0] + 0.3).abs() < 1e-12);
        assert!((regret[1] + 0.1).abs() < 1e-12);

        let unmatched = BTreeMap::from([(
            1,
            PointMetrics { utility: 0.8, absolute_regret: 0.2, ..PointMetrics::default() },
        )]);
        assert!(paired_cluster_deltas(&unmatched, &baseline).is_err());
    }

    #[test]
    fn dominance_requires_improvement_in_one_dimension_and_no_worsening() {
        let left = ParetoCost { selected_bytes: 10.0, index_bytes: 0.0, latency_ns: 2.0, regret: 0.4 };
        let dominated = ParetoCost { selected_bytes: 20.0, index_bytes: 5.0, latency_ns: 4.0, regret: 0.5 };
        let incomparable = ParetoCost { selected_bytes: 5.0, index_bytes: 10.0, latency_ns: 3.0, regret: 0.3 };
        assert!(dominates(left, dominated));
        assert!(!dominates(left, incomparable));
        assert!(!dominates(left, left));
    }

    #[test]
    fn provenance_requires_lowercase_git_sha() {
        let mut record = json!({
            "schema_version": INPUT_SCHEMA,
            "run_id": "r1",
            "public_seed": 4,
            "source_revision": null,
            "source_code_sha256": "a".repeat(64),
            "fixture_sha256": "b".repeat(64),
            "schema_sha256": "c".repeat(64)
        });
        assert!(parse_provenance(&record).is_ok());
        record["source_revision"] = json!("G".repeat(40));
        assert!(parse_provenance(&record).is_err());
    }
}
