// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analyze paired counterfactual proof-insight outcomes.
//!
//! This binary joins pre-adjudicated base/variant pairs to actual transfer-selectivity
//! outcome rows. It never decides mathematical compatibility and always leaves the
//! confirmatory gate unassessed until real checker receipts and independent review exist.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SUITE_VERSION: &str = "pitb.counterfactual-precondition-suite.v0";
const OUTCOME_VERSION: &str = "pitb.transfer-selectivity-outcome.v0";
const REPORT_VERSION: &str = "pitb.counterfactual-sensitivity-report.v0";
const DEFAULT_BOOTSTRAP_REPLICATES: usize = 10_000;
const DEFAULT_SEED: u64 = 0x5049_5442_4346_4143;

#[derive(Debug, Clone)]
struct PairObservation {
    component_id: String,
    mutation_kind: String,
    base_compatible: bool,
    base_accept: bool,
    base_decision: String,
    variant_compatible: bool,
    variant_accept: bool,
    variant_decision: String,
    noop: bool,
    base_probability: Option<f64>,
    variant_probability: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    base_positive: usize,
    base_accept: usize,
    variant_positive: usize,
    variant_accept: usize,
    variant_negative: usize,
    variant_nonaccept: usize,
    paired_eligible: usize,
    paired_correct: usize,
    noop_eligible: usize,
    noop_stable: usize,
    brier_sum: f64,
    brier_count: usize,
    noop_probability_delta_sum: f64,
    noop_probability_delta_count: usize,
}

impl Counts {
    fn add(&mut self, observation: &PairObservation) {
        if observation.base_compatible {
            self.base_positive += 1;
            if observation.base_accept {
                self.base_accept += 1;
            }
        }
        if observation.variant_compatible {
            self.variant_positive += 1;
            if observation.variant_accept {
                self.variant_accept += 1;
            }
        } else {
            self.variant_negative += 1;
            if !observation.variant_accept {
                self.variant_nonaccept += 1;
            }
        }
        if observation.base_compatible && !observation.variant_compatible {
            self.paired_eligible += 1;
            if observation.base_accept && !observation.variant_accept {
                self.paired_correct += 1;
            }
        }
        if observation.noop && observation.base_compatible && observation.variant_compatible {
            self.noop_eligible += 1;
            if observation.base_decision == observation.variant_decision {
                self.noop_stable += 1;
            }
            if let (Some(base), Some(variant)) =
                (observation.base_probability, observation.variant_probability)
            {
                self.noop_probability_delta_sum += (base - variant).abs();
                self.noop_probability_delta_count += 1;
            }
        }
        for (probability, truth) in [
            (observation.base_probability, observation.base_compatible),
            (observation.variant_probability, observation.variant_compatible),
        ] {
            if let Some(probability) = probability {
                self.brier_sum += (probability - if truth { 1.0 } else { 0.0 }).powi(2);
                self.brier_count += 1;
            }
        }
    }

    fn add_assign(&mut self, other: Counts) {
        self.base_positive += other.base_positive;
        self.base_accept += other.base_accept;
        self.variant_positive += other.variant_positive;
        self.variant_accept += other.variant_accept;
        self.variant_negative += other.variant_negative;
        self.variant_nonaccept += other.variant_nonaccept;
        self.paired_eligible += other.paired_eligible;
        self.paired_correct += other.paired_correct;
        self.noop_eligible += other.noop_eligible;
        self.noop_stable += other.noop_stable;
        self.brier_sum += other.brier_sum;
        self.brier_count += other.brier_count;
        self.noop_probability_delta_sum += other.noop_probability_delta_sum;
        self.noop_probability_delta_count += other.noop_probability_delta_count;
    }

    fn metrics(self) -> BTreeMap<&'static str, (usize, usize, Option<f64>)> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "base_compatible_accept_recall",
            (self.base_accept, self.base_positive, ratio(self.base_accept, self.base_positive)),
        );
        metrics.insert(
            "counterfactual_compatible_accept_recall",
            (
                self.variant_accept,
                self.variant_positive,
                ratio(self.variant_accept, self.variant_positive),
            ),
        );
        metrics.insert(
            "counterfactual_incompatible_sensitivity",
            (
                self.variant_nonaccept,
                self.variant_negative,
                ratio(self.variant_nonaccept, self.variant_negative),
            ),
        );
        metrics.insert(
            "counterfactual_false_accept_rate",
            (
                self.variant_negative.saturating_sub(self.variant_nonaccept),
                self.variant_negative,
                ratio(
                    self.variant_negative.saturating_sub(self.variant_nonaccept),
                    self.variant_negative,
                ),
            ),
        );
        metrics.insert(
            "paired_contract_adherence",
            (
                self.paired_correct,
                self.paired_eligible,
                ratio(self.paired_correct, self.paired_eligible),
            ),
        );
        metrics.insert(
            "semantics_preserving_decision_stability",
            (self.noop_stable, self.noop_eligible, ratio(self.noop_stable, self.noop_eligible)),
        );
        metrics.insert(
            "compatibility_probability_brier_score",
            (
                self.brier_count,
                self.brier_count,
                (self.brier_count > 0).then_some(self.brier_sum / self.brier_count as f64),
            ),
        );
        metrics.insert(
            "semantics_preserving_mean_probability_absolute_delta",
            (
                self.noop_probability_delta_count,
                self.noop_probability_delta_count,
                (self.noop_probability_delta_count > 0).then_some(
                    self.noop_probability_delta_sum / self.noop_probability_delta_count as f64,
                ),
            ),
        );
        metrics
    }
}

#[derive(Debug, Clone)]
struct Metric {
    numerator: usize,
    denominator: usize,
    estimate: Option<f64>,
    ci_95: Option<[f64; 2]>,
    defined_bootstrap_replicates: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 7 || args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
        eprintln!(
            "usage: {} <counterfactual-suite.json> <outcomes.jsonl> <test|external_challenge> <report.json> [bootstrap_replicates] [seed]",
            args.first().map(String::as_str).unwrap_or("analyze_counterfactual_sensitivity")
        );
        if args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
            return Ok(());
        }
        bail!("invalid command-line arguments");
    }

    let suite_bytes = fs::read(&args[1]).with_context(|| format!("read suite {}", args[1]))?;
    let outcome_bytes = fs::read(&args[2]).with_context(|| format!("read outcomes {}", args[2]))?;
    let requested_split = args[3].as_str();
    if !matches!(requested_split, "test" | "external_challenge") {
        bail!(
            "confirmatory analysis accepts only test or external_challenge, found {requested_split:?}"
        );
    }
    let bootstrap_replicates = args
        .get(5)
        .map(|x| x.parse::<usize>().context("parse bootstrap replicate count"))
        .transpose()?
        .unwrap_or(DEFAULT_BOOTSTRAP_REPLICATES);
    if bootstrap_replicates < 1_000 {
        bail!("at least 1000 bootstrap replicates are required");
    }
    let seed = args
        .get(6)
        .map(|x| x.parse::<u64>().context("parse seed"))
        .transpose()?
        .unwrap_or(DEFAULT_SEED);

    let suite: Value = serde_json::from_slice(&suite_bytes)
        .context("parse counterfactual suite JSON")?;
    let suite_id = required_string(&suite, "suite_id", "suite")?.to_owned();
    if required_string(&suite, "schema_version", "suite")? != SUITE_VERSION {
        bail!("unsupported suite schema_version");
    }
    let pairs = suite
        .get("pairs")
        .and_then(Value::as_array)
        .context("suite.pairs must be an array")?;
    if pairs.is_empty() {
        bail!("suite contains no pairs");
    }

    let outcome_text = std::str::from_utf8(&outcome_bytes).context("outcomes JSONL must be UTF-8")?;
    let outcomes = parse_outcomes(outcome_text, requested_split)?;
    let observations = join_pairs(pairs, &outcomes, requested_split)?;
    if observations.is_empty() {
        bail!("no adjudicated base/counterfactual pairs are eligible for analysis");
    }

    let metrics = calculate_metrics(&observations, bootstrap_replicates, seed);
    let counts = aggregate(&observations);
    let mut per_mutation = BTreeMap::new();
    let mut mutation_kinds = BTreeSet::new();
    for observation in &observations {
        mutation_kinds.insert(observation.mutation_kind.clone());
    }
    for kind in mutation_kinds {
        let subset: Vec<_> = observations
            .iter()
            .filter(|x| x.mutation_kind == kind)
            .cloned()
            .collect();
        let subset_counts = aggregate(&subset);
        per_mutation.insert(kind, json!({
            "pair_count": subset.len(),
            "incompatible_variant_count": subset_counts.variant_negative,
            "sensitivity": metric_json(
                subset_counts.variant_nonaccept,
                subset_counts.variant_negative,
                ratio(subset_counts.variant_nonaccept, subset_counts.variant_negative)
            ),
            "false_accept_rate": metric_json(
                subset_counts.variant_negative.saturating_sub(subset_counts.variant_nonaccept),
                subset_counts.variant_negative,
                ratio(
                    subset_counts.variant_negative.saturating_sub(subset_counts.variant_nonaccept),
                    subset_counts.variant_negative
                )
            )
        }));
    }

    let suite_sha256 = sha256_hex(&suite_bytes);
    let outcomes_sha256 = sha256_hex(&outcome_bytes);
    let report = json!({
        "schema_version": REPORT_VERSION,
        "suite_id": suite_id,
        "dataset_kind": suite.get("dataset_kind").and_then(Value::as_str).unwrap_or("unknown"),
        "split": requested_split,
        "suite_sha256": suite_sha256,
        "outcomes_sha256": outcomes_sha256,
        "counts": {
            "suite_pairs_in_requested_split": pairs
                .iter()
                .filter(|p| p.get("split").and_then(Value::as_str) == Some(requested_split))
                .count(),
            "adjudicated_pairs_analyzed": observations.len(),
            "base_compatible_pairs": counts.base_positive,
            "counterfactual_compatible_pairs": counts.variant_positive,
            "counterfactual_incompatible_pairs": counts.variant_negative,
            "paired_contract_eligible_pairs": counts.paired_eligible,
            "semantics_preserving_pairs": counts.noop_eligible,
            "probability_scored_rows": counts.brier_count,
            "components": observations
                .iter()
                .map(|o| o.component_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
        },
        "metrics": metrics,
        "by_mutation_kind": per_mutation,
        "bootstrap": {
            "unit": "component_id",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile_95"
        },
        "proposed_gates": {
            "target_counterfactual_sensitivity_min": 0.90,
            "target_base_positive_recall_min": 0.90,
            "target_semantics_preserving_decision_stability_min": 0.95,
            "overall_confirmatory_decision": "not_evaluated_requires_real_adjudication_and_independent_checker_receipts"
        },
        "limitations": [
            "The analyzer joins outcomes and computes metrics; it does not establish mathematical compatibility.",
            "Synthetic fixtures validate the software contract only and are not benchmark results.",
            "The primary PITB exact-target semantic-gate success endpoint remains separate.",
            "The overall confirmatory decision is never set to pass by this binary."
        ]
    });

    let output = serde_json::to_string_pretty(&report)?;
    fs::write(&args[4], format!("{output}\n"))
        .with_context(|| format!("write report {}", args[4]))?;
    println!(
        "PITB_COUNTERFACTUAL_ANALYSIS_WRITTEN: {} adjudicated pairs, {} components, report {}",
        observations.len(),
        counts_component_count(&observations),
        args[4]
    );
    Ok(())
}

fn parse_outcomes(text: &str, requested_split: &str) -> Result<BTreeMap<String, Value>> {
    let mut outcomes = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            bail!("outcomes JSONL line {} is empty", index + 1);
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse outcome JSONL line {}", index + 1))?;
        let context = format!("outcome line {}", index + 1);
        require_exact_keys(
            &value,
            &[
                "schema_version",
                "pair_id",
                "split",
                "component_id",
                "source",
                "target",
                "adjudication",
                "prediction",
                "run",
            ],
            &[],
            &context,
        )?;
        for (field, required, optional) in [
            ("source", &["record_id", "family_id", "source_commit", "artifact_sha256", "insight_sha256"][..], &[][..]),
            ("target", &["theorem_id", "family_id", "source_commit", "artifact_path", "artifact_sha256", "statement_sha256", "obligation_sha256"][..], &["domain_labels"][..]),
            ("adjudication", &["status", "label", "rubric_version", "reviewer_decisions", "receipt_sha256"][..], &["adjudication_note"][..]),
            ("prediction", &["decision", "predicted_label", "confidence", "compatibility_probability", "mismatch_codes", "precondition_checks", "receipt_sha256"][..], &[][..]),
            ("run", &["model_id", "model_version", "prompt_sha256", "config_sha256", "input_sha256", "output_sha256", "budget"][..], &["actual_model_calls", "actual_tokens", "actual_wall_ms", "actual_cost_microusd"][..]),
        ] {
            let object = value.get(field).with_context(|| format!("{context}.{field} missing"))?;
            require_exact_keys(object, required, optional, &format!("{context}.{field}"))?;
        }
        let reviewer_rows = value.get("adjudication").and_then(|x| x.get("reviewer_decisions")).and_then(Value::as_array)
            .with_context(|| format!("{context}.adjudication.reviewer_decisions must be an array"))?;
        for (reviewer_index, reviewer) in reviewer_rows.iter().enumerate() {
            require_exact_keys(reviewer, &["reviewer_id", "label", "reason_codes", "evidence_sha256"], &[], &format!("{context}.reviewer_decisions[{reviewer_index}]"))?;
        }
        let check_rows = value.get("prediction").and_then(|x| x.get("precondition_checks")).and_then(Value::as_array)
            .with_context(|| format!("{context}.prediction.precondition_checks must be an array"))?;
        for (check_index, check) in check_rows.iter().enumerate() {
            require_exact_keys(check, &["check_id", "outcome", "evidence_sha256"], &[], &format!("{context}.precondition_checks[{check_index}]"))?;
        }
        require_exact_keys(
            value.get("run").and_then(|x| x.get("budget")).context("run budget missing")?,
            &["max_model_calls", "max_tokens", "max_wall_ms", "max_cost_microusd"],
            &[],
            &format!("{context}.run.budget"),
        )?;
        if required_string(&value, "schema_version", &context)? != OUTCOME_VERSION {
            bail!("{context} has unsupported schema_version");
        }
        let pair_id = required_string(&value, "pair_id", &context)?.to_owned();
        if required_string(&value, "split", &context)? != requested_split {
            bail!("{context} is not in requested split {requested_split}");
        }
        if !matches!(required_string_at(&value, &["adjudication", "status"], &context)?, "adjudicated" | "unresolved" | "pending") {
            bail!("{context} has invalid adjudication status");
        }
        if !matches!(required_string_at(&value, &["adjudication", "label"], &context)?, "compatible" | "incompatible" | "partial" | "unknown") {
            bail!("{context} has invalid adjudication label");
        }
        if !matches!(required_string_at(&value, &["prediction", "decision"], &context)?, "accept" | "reject" | "abstain") {
            bail!("{context} has invalid decision");
        }
        validate_probability(value.get("prediction").and_then(|p| p.get("compatibility_probability")), &context)?;
        validate_digest(required_string_at(&value, &["source", "source_commit"], &context)?, 40, &format!("{context}.source.source_commit"))?;
        for (path, name) in [
            (&["source", "artifact_sha256"][..], "source.artifact_sha256"),
            (&["source", "insight_sha256"][..], "source.insight_sha256"),
            (&["target", "artifact_sha256"][..], "target.artifact_sha256"),
            (&["target", "statement_sha256"][..], "target.statement_sha256"),
            (&["target", "obligation_sha256"][..], "target.obligation_sha256"),
            (&["target", "source_commit"][..], "target.source_commit"),
            (&["adjudication", "receipt_sha256"][..], "adjudication.receipt_sha256"),
            (&["prediction", "receipt_sha256"][..], "prediction.receipt_sha256"),
        ] {
            validate_digest(required_string_at(&value, path, &context)?, if name.ends_with("source_commit") {40} else {64}, &format!("{context}.{name}"))?;
        }
        for name in ["record_id", "family_id"] {
            required_string_at(&value, &["source", name], &context)?;
        }
        for name in ["theorem_id", "family_id", "artifact_path"] {
            required_string_at(&value, &["target", name], &context)?;
        }
        validate_prediction_consistency(&value, &context)?;
        if outcomes.insert(pair_id.clone(), value).is_some() {
            bail!("duplicate outcome pair_id {pair_id:?}");
        }
    }
    if outcomes.is_empty() {
        bail!("outcomes JSONL contains no rows");
    }
    Ok(outcomes)
}

fn join_pairs(pairs: &[Value], outcomes: &BTreeMap<String, Value>, requested_split: &str) -> Result<Vec<PairObservation>> {
    let mut all_references = BTreeSet::new();
    let mut observations = Vec::new();
    let mut family_split = BTreeMap::<String, String>::new();
    let mut family_component = BTreeMap::<String, String>::new();
    let mut insight_split = BTreeMap::<String, String>::new();
    for (index, pair) in pairs.iter().enumerate() {
        let ctx = format!("suite.pairs[{index}]");
        let split = required_string(pair, "split", &ctx)?;
        if split != requested_split {
            continue;
        }
        let component_id = required_string(pair, "component_id", &ctx)?.to_owned();
        let source = pair.get("source_insight").context("source_insight missing")?;
        let target = pair.get("target").context("target missing")?;
        let mutation = pair.get("mutation").context("mutation missing")?;
        let base_adj = pair.get("base_adjudication").context("base_adjudication missing")?;
        let variant_adj = pair.get("counterfactual_adjudication").context("counterfactual_adjudication missing")?;
        let base_id = required_string(pair, "base_outcome_pair_id", &ctx)?.to_owned();
        let variant_id = required_string(pair, "counterfactual_outcome_pair_id", &ctx)?.to_owned();
        if base_id == variant_id || !all_references.insert(base_id.clone()) || !all_references.insert(variant_id.clone()) {
            bail!("{ctx} has duplicate outcome links");
        }
        let source_record = required_string(source, "record_id", &format!("{ctx}.source_insight"))?;
        let source_family = required_string(source, "family_id", &format!("{ctx}.source_insight"))?;
        let source_commit = required_string(source, "source_commit", &format!("{ctx}.source_insight"))?;
        let source_digest = required_string(source, "insight_sha256", &format!("{ctx}.source_insight"))?;
        let target_id = required_string(target, "target_id", &format!("{ctx}.target"))?;
        let target_family = required_string(target, "family_id", &format!("{ctx}.target"))?;
        let target_commit = required_string(target, "source_commit", &format!("{ctx}.target"))?;
        let artifact_path = required_string(target, "artifact_path", &format!("{ctx}.target"))?;
        if artifact_path.starts_with('/')
            || artifact_path.split('/').any(|part| part == ".." || part.is_empty()) {
            bail!("{ctx}.target.artifact_path must be a safe repository-relative path");
        }
        let artifact_digest = required_string(target, "artifact_sha256", &format!("{ctx}.target"))?;
        let base_statement_digest = required_string(target, "base_statement_sha256", &format!("{ctx}.target"))?;
        let variant_statement_digest = required_string(target, "counterfactual_statement_sha256", &format!("{ctx}.target"))?;
        let base_status = required_string(base_adj, "status", &format!("{ctx}.base_adjudication"))?;
        let base_label = required_string(base_adj, "label", &format!("{ctx}.base_adjudication"))?;
        let variant_status = required_string(variant_adj, "status", &format!("{ctx}.counterfactual_adjudication"))?;
        let variant_label = required_string(variant_adj, "label", &format!("{ctx}.counterfactual_adjudication"))?;
        for (family, split) in [(source_family, split), (target_family, split)] {
            if let Some(previous) = family_split.insert(family.to_owned(), split.to_owned()) {
                if previous != split {
                    bail!("family {family} crosses suite splits");
                }
            }
            if let Some(previous) = family_component.insert(family.to_owned(), component_id.clone()) {
                if previous != component_id {
                    bail!("family {family} crosses pair components");
                }
            }
        }
        if let Some(previous) = insight_split.insert(source_record.to_owned(), split.to_owned()) {
            if previous != split {
                bail!("source insight {source_record} crosses suite splits");
            }
        }

        let base = outcomes.get(&base_id).with_context(|| format!("{ctx} missing base outcome {base_id}"))?;
        let variant = outcomes.get(&variant_id).with_context(|| format!("{ctx} missing counterfactual outcome {variant_id}"))?;
        validate_matched_arm(base, variant, &ctx)?;
        for (row, expected_id, label, status, statement_digest, side) in [
            (base, &base_id, base_label, base_status, base_statement_digest, "base"),
            (variant, &variant_id, variant_label, variant_status, variant_statement_digest, "counterfactual"),
        ] {
            let row_ctx = format!("{ctx}.{side} outcome {expected_id}");
            if required_string(row, "pair_id", &row_ctx)? != expected_id {
                bail!("{row_ctx} pair_id mismatch");
            }
            if required_string(row, "component_id", &row_ctx)? != component_id {
                bail!("{row_ctx} component_id mismatch");
            }
            if required_string_at(row, &["source", "record_id"], &row_ctx)? != source_record
                || required_string_at(row, &["source", "family_id"], &row_ctx)? != source_family
                || required_string_at(row, &["source", "source_commit"], &row_ctx)? != source_commit
                || required_string_at(row, &["source", "insight_sha256"], &row_ctx)? != source_digest {
                bail!("{row_ctx} source insight provenance mismatch");
            }
            if required_string_at(row, &["target", "theorem_id"], &row_ctx)? != target_id
                || required_string_at(row, &["target", "family_id"], &row_ctx)? != target_family
                || required_string_at(row, &["target", "source_commit"], &row_ctx)? != target_commit
                || required_string_at(row, &["target", "artifact_path"], &row_ctx)? != artifact_path
                || required_string_at(row, &["target", "artifact_sha256"], &row_ctx)? != artifact_digest
                || required_string_at(row, &["target", "statement_sha256"], &row_ctx)? != statement_digest {
                bail!("{row_ctx} target/statement provenance mismatch");
            }
            if required_string(row, "split", &row_ctx)? != requested_split
                || required_string_at(row, &["adjudication", "label"], &row_ctx)? != label
                || required_string_at(row, &["adjudication", "status"], &row_ctx)? != status {
                bail!("{row_ctx} split or gold adjudication mismatch");
            }
            let suite_adjudication = if side == "base" { base_adj } else { variant_adj };
            validate_adjudication_link(suite_adjudication, row, &row_ctx)?;
            validate_run_budgets(row, &row_ctx)?;
        }
        if base_status != "adjudicated" || variant_status != "adjudicated"
            || !matches!(base_label, "compatible" | "incompatible")
            || !matches!(variant_label, "compatible" | "incompatible") {
            continue;
        }
        if base_label != "compatible" {
            bail!("{ctx}: paired counterfactual analysis requires an independently adjudicated compatible base");
        }
        let noop = required_string(mutation, "kind", &format!("{ctx}.mutation"))? == "no_semantic_change";
        if noop && variant_label != "compatible" {
            bail!("{ctx}: semantics-preserving pair has incompatible gold label");
        }
        let base_decision = required_string_at(base, &["prediction", "decision"], &ctx)?;
        let variant_decision = required_string_at(variant, &["prediction", "decision"], &ctx)?;
        observations.push(PairObservation {
            component_id,
            mutation_kind: required_string(mutation, "kind", &format!("{ctx}.mutation"))?.to_owned(),
            base_compatible: base_label == "compatible",
            base_accept: base_decision == "accept",
            base_decision: base_decision.to_owned(),
            variant_compatible: variant_label == "compatible",
            variant_accept: variant_decision == "accept",
            variant_decision: variant_decision.to_owned(),
            noop,
            base_probability: probability(base),
            variant_probability: probability(variant),
        });
    }
    if observations.is_empty() {
        bail!("no fully adjudicated compatible-base pairs in requested split");
    }
    let unused: Vec<_> = outcomes.keys().filter(|key| !all_references.contains(*key)).collect();
    if !unused.is_empty() {
        bail!("outcomes JSONL has {} row(s) not referenced by the requested suite", unused.len());
    }
    Ok(observations)
}

fn calculate_metrics(observations: &[PairObservation], replicates: usize, seed: u64) -> Value {
    let overall = aggregate(observations);
    let mut components = BTreeMap::<String, Vec<PairObservation>>::new();
    for observation in observations {
        components.entry(observation.component_id.clone()).or_default().push(observation.clone());
    }
    let mut bootstrap = BTreeMap::<&'static str, Vec<f64>>::new();
    for name in overall.metrics().keys() {
        bootstrap.insert(*name, Vec::with_capacity(replicates));
    }
    let mut rng = SplitMix64::new(seed);
    let component_values: Vec<_> = components.values().collect();
    for _ in 0..replicates {
        let mut sample = Vec::new();
        for _ in 0..component_values.len() {
            sample.extend(component_values[rng.index(component_values.len())].iter().cloned());
        }
        let metrics = aggregate(&sample).metrics();
        for (name, (_, _, estimate)) in metrics {
            if let Some(estimate) = estimate {
                bootstrap.get_mut(name).expect("registered metric").push(estimate);
            }
        }
    }
    let mut out = serde_json::Map::new();
    for (name, (numerator, denominator, estimate)) in overall.metrics() {
        let values = bootstrap.get_mut(name).expect("registered metric");
        values.sort_by(f64::total_cmp);
        let ci = if values.is_empty() {
            None
        } else {
            Some([percentile(values, 0.025), percentile(values, 0.975)])
        };
        let metric = Metric {
            numerator,
            denominator,
            estimate,
            ci_95: ci,
            defined_bootstrap_replicates: values.len(),
        };
        out.insert(name.to_owned(), metric_json_value(&metric, name));
    }
    Value::Object(out)
}

fn aggregate(observations: &[PairObservation]) -> Counts {
    let mut counts = Counts::default();
    for observation in observations {
        counts.add(observation);
    }
    counts
}

fn metric_json(numerator: usize, denominator: usize, estimate: Option<f64>) -> Value {
    json!({"statistic":"ratio","numerator":numerator,"denominator":denominator,"estimate":estimate,"ci_95":null,"defined_bootstrap_replicates":0})
}

fn metric_json_value(metric: &Metric, name: &str) -> Value {
    let statistic = if name == "compatibility_probability_brier_score"
        || name == "semantics_preserving_mean_probability_absolute_delta" {
        "mean"
    } else {
        "ratio"
    };
    json!({
        "statistic": statistic,
        "numerator": metric.numerator,
        "denominator": metric.denominator,
        "estimate": metric.estimate,
        "ci_95": metric.ci_95,
        "defined_bootstrap_replicates": metric.defined_bootstrap_replicates
    })
}

fn validate_matched_arm(base: &Value, variant: &Value, context: &str) -> Result<()> {
    for field in ["model_id", "model_version", "prompt_sha256", "config_sha256"] {
        if required_string_at(base, &["run", field], context)?
            != required_string_at(variant, &["run", field], context)? {
            bail!("{context}: base/counterfactual rows do not use the same {field}");
        }
    }
    if base.get("run").and_then(|x| x.get("budget"))
        != variant.get("run").and_then(|x| x.get("budget")) {
        bail!("{context}: base/counterfactual run budgets differ");
    }
    Ok(())
}

fn validate_adjudication_link(suite_adj: &Value, outcome: &Value, context: &str) -> Result<()> {
    let outcome_adj = outcome.get("adjudication").context("outcome adjudication missing")?;
    for field in ["status", "label", "rubric_version", "receipt_sha256"] {
        if required_string(suite_adj, field, &format!("{context}.suite_adjudication"))?
            != required_string(outcome_adj, field, &format!("{context}.outcome_adjudication"))? {
            bail!("{context}: adjudication {field} does not match the paired suite record");
        }
    }
    let suite_reviewers = suite_adj.get("reviewer_decisions").and_then(Value::as_array)
        .with_context(|| format!("{context}: suite reviewer_decisions must be an array"))?;
    let outcome_reviewers = outcome_adj.get("reviewer_decisions").and_then(Value::as_array)
        .with_context(|| format!("{context}: outcome reviewer_decisions must be an array"))?;
    let status = required_string(suite_adj, "status", context)?;
    let gold_label = required_string(suite_adj, "label", context)?;
    if status == "adjudicated" {
        if suite_reviewers.len() < 2
            || gold_label == "unknown"
            || suite_reviewers.iter().any(|reviewer| {
                reviewer.get("label").and_then(Value::as_str) != Some(gold_label)
            })
        {
            bail!("{context}: adjudicated label requires two agreeing independent reviewers");
        }
    } else if gold_label != "unknown" {
        bail!("{context}: unresolved/pending adjudication must preserve unknown label");
    }
    if suite_reviewers.len() != outcome_reviewers.len() {
        bail!("{context}: suite/outcome reviewer counts differ");
    }
    let mut expected = BTreeMap::<String, (&str, &str, &str)>::new();
    for reviewer in suite_reviewers {
        let id = required_string(reviewer, "reviewer_id", context)?.to_owned();
        let label = required_string(reviewer, "label", context)?;
        let reason = required_string(reviewer, "reason_code", context)?;
        let evidence = required_string(reviewer, "evidence_sha256", context)?;
        validate_digest(evidence, 64, &format!("{context}.reviewer.evidence_sha256"))?;
        if expected.insert(id, (label, reason, evidence)).is_some() {
            bail!("{context}: duplicate suite reviewer ID");
        }
    }
    let mut observed_ids = BTreeSet::new();
    for reviewer in outcome_reviewers {
        let id = required_string(reviewer, "reviewer_id", context)?.to_owned();
        if !observed_ids.insert(id.clone()) {
            bail!("{context}: duplicate outcome reviewer ID");
        }
        let (label, reason, evidence) = expected.get(&id)
            .with_context(|| format!("{context}: outcome contains unregistered reviewer {id}"))?;
        if required_string(reviewer, "label", context)? != *label
            || required_string(reviewer, "evidence_sha256", context)? != *evidence {
            bail!("{context}: outcome reviewer label/evidence differs from suite adjudication");
        }
        let reasons = reviewer.get("reason_codes").and_then(Value::as_array)
            .with_context(|| format!("{context}: reviewer reason_codes must be an array"))?;
        let reason = *reason;
        let outcome_reason = match reason {
            "preconditions_satisfied" | "no_semantic_change" => "other",
            "precondition_strengthened" => "missing_precondition",
            "conclusion_mismatch" => "predicate_mismatch",
            value => value,
        };
        if !reasons.iter().any(|value| value.as_str() == Some(outcome_reason)) {
            bail!("{context}: outcome reviewer reason_codes omit the mapped suite reason code");
        }
    }
    Ok(())
}

fn validate_run_budgets(row: &Value, context: &str) -> Result<()> {
    let run = row.get("run").with_context(|| format!("{context}.run missing"))?;
    let budget = run.get("budget").with_context(|| format!("{context}.run.budget missing"))?;
    for (actual, limit) in [
        ("actual_model_calls", "max_model_calls"),
        ("actual_tokens", "max_tokens"),
        ("actual_wall_ms", "max_wall_ms"),
        ("actual_cost_microusd", "max_cost_microusd"),
    ] {
        let cap = budget.get(limit).and_then(Value::as_u64).with_context(|| format!("{context}.run.budget.{limit} must be non-negative integer"))?;
        if let Some(actual_value) = run.get(actual) {
            let amount = actual_value.as_u64().with_context(|| format!("{context}.run.{actual} must be non-negative integer"))?;
            if amount > cap {
                bail!("{context}.run.{actual} exceeds {limit}");
            }
        }
    }
    Ok(())
}

fn require_exact_keys(value: &Value, required: &[&str], optional: &[&str], context: &str) -> Result<()> {
    let object = value.as_object().with_context(|| format!("{context} must be an object"))?;
    for key in required {
        if !object.contains_key(*key) {
            bail!("{context}.{key} is required");
        }
    }
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            bail!("{context} has unexpected field {key:?}");
        }
    }
    Ok(())
}

fn validate_prediction_consistency(row: &Value, context: &str) -> Result<()> {
    let decision = required_string_at(row, &["prediction", "decision"], context)?;
    let label = required_string_at(row, &["prediction", "predicted_label"], context)?;
    match decision {
        "accept" if label != "compatible" => bail!("{context}: accept decision must predict compatible"),
        "reject" if !matches!(label, "incompatible" | "partial") => bail!("{context}: reject decision must predict incompatible or partial"),
        "abstain" if label != "unknown" => bail!("{context}: abstain decision must predict unknown"),
        _ => {}
    }
    Ok(())
}

fn validate_probability(value: Option<&Value>, context: &str) -> Result<()> {
    let Some(value) = value else {
        bail!("{context}.prediction.compatibility_probability is required; use null if unscored");
    };
    if value.is_null() {
        return Ok(());
    }
    let p = value.as_f64().with_context(|| format!("{context}.prediction.compatibility_probability must be null or numeric"))?;
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        bail!("{context}.prediction.compatibility_probability outside [0, 1]");
    }
    Ok(())
}

fn probability(row: &Value) -> Option<f64> {
    row.get("prediction")?.get("compatibility_probability")?.as_f64()
}

fn required_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .with_context(|| format!("{context}.{key} must be a non-empty string"))
}

fn required_string_at<'a>(value: &'a Value, path: &[&str], context: &str) -> Result<&'a str> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment).with_context(|| format!("{context}.{} missing", path.join(".")))?;
    }
    current.as_str().filter(|s| !s.trim().is_empty())
        .with_context(|| format!("{context}.{} must be a non-empty string", path.join(".")))
}

fn validate_digest(value: &str, length: usize, context: &str) -> Result<()> {
    if value.len() != length || !value.bytes().all(|x| x.is_ascii_hexdigit() && !x.is_ascii_uppercase()) {
        bail!("{context} must be lowercase hex with length {length}");
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64)
    }
}

fn counts_component_count(observations: &[PairObservation]) -> usize {
    observations.iter().map(|o| o.component_id.as_str()).collect::<BTreeSet<_>>().len()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_metrics_require_both_sides_to_be_correct() {
        let pairs = vec![
            PairObservation {
                component_id: "c1".into(), mutation_kind: "precondition_removed".into(),
                base_compatible: true, base_accept: true, base_decision: "accept".into(),
                variant_compatible: false, variant_accept: false, variant_decision: "reject".into(), noop: false,
                base_probability: Some(0.9), variant_probability: Some(0.1),
            },
            PairObservation {
                component_id: "c2".into(), mutation_kind: "precondition_removed".into(),
                base_compatible: true, base_accept: false, base_decision: "reject".into(),
                variant_compatible: false, variant_accept: false, variant_decision: "reject".into(), noop: false,
                base_probability: Some(0.7), variant_probability: Some(0.2),
            },
            PairObservation {
                component_id: "c3".into(), mutation_kind: "no_semantic_change".into(),
                base_compatible: true, base_accept: true, base_decision: "accept".into(),
                variant_compatible: true, variant_accept: false, variant_decision: "reject".into(), noop: true,
                base_probability: Some(0.8), variant_probability: Some(0.4),
            },
        ];
        let counts = aggregate(&pairs);
        let metrics = counts.metrics();
        assert_eq!(metrics["base_compatible_accept_recall"].2, Some(2.0 / 3.0));
        assert_eq!(metrics["counterfactual_incompatible_sensitivity"].2, Some(1.0));
        assert_eq!(metrics["counterfactual_false_accept_rate"].2, Some(0.0));
        assert_eq!(metrics["paired_contract_adherence"].2, Some(0.5));
        assert_eq!(metrics["semantics_preserving_decision_stability"].2, Some(0.0));
        assert_eq!(metrics["semantics_preserving_mean_probability_absolute_delta"].2, Some(0.4));
    }

    #[test]
    fn goal_noise_or_reject_everything_cannot_pass_paired_adherence() {
        let reject_all = PairObservation {
            component_id: "c1".into(), mutation_kind: "precondition_removed".into(),
            base_compatible: true, base_accept: false, base_decision: "reject".into(),
            variant_compatible: false, variant_accept: false, variant_decision: "reject".into(), noop: false,
            base_probability: None, variant_probability: None,
        };
        let counts = aggregate(&[reject_all]);
        assert_eq!(counts.metrics()["counterfactual_incompatible_sensitivity"].2, Some(1.0));
        assert_eq!(counts.metrics()["base_compatible_accept_recall"].2, Some(0.0));
        assert_eq!(counts.metrics()["paired_contract_adherence"].2, Some(0.0));
    }

    #[test]
    fn reject_and_abstain_are_not_the_same_noop_decision() {
        let changed_action = PairObservation {
            component_id: "noop-component".into(),
            mutation_kind: "no_semantic_change".into(),
            base_compatible: true,
            base_accept: false,
            base_decision: "reject".into(),
            variant_compatible: true,
            variant_accept: false,
            variant_decision: "abstain".into(),
            noop: true,
            base_probability: Some(0.2),
            variant_probability: Some(0.3),
        };
        let metrics = aggregate(&[changed_action]).metrics();
        assert_eq!(metrics["semantics_preserving_decision_stability"].2, Some(0.0));
    }

    #[test]
    fn component_bootstrap_is_deterministic_and_seeded() {
        let pairs = vec![
            PairObservation { component_id:"a".into(), mutation_kind:"x".into(), base_compatible:true, base_accept:true, base_decision:"accept".into(), variant_compatible:false, variant_accept:false, variant_decision:"reject".into(), noop:false, base_probability:None, variant_probability:None },
            PairObservation { component_id:"b".into(), mutation_kind:"x".into(), base_compatible:true, base_accept:true, base_decision:"accept".into(), variant_compatible:false, variant_accept:true, variant_decision:"accept".into(), noop:false, base_probability:None, variant_probability:None },
        ];
        let first = calculate_metrics(&pairs, 1_000, 9);
        let second = calculate_metrics(&pairs, 1_000, 9);
        assert_eq!(first, second);
    }
}
