// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable evaluation for PITB's transfer-selectivity secondary track.
//!
//! This binary analyzes already-adjudicated JSONL outcomes. It does not decide
//! whether a mathematical transfer is valid; it computes metrics against the
//! supplied labels and rejects malformed/inconsistent evaluation inputs.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "pitb.transfer-selectivity-outcome.v0";
const REPORT_SCHEMA_VERSION: &str = "pitb.transfer-selectivity-analysis.v0";
const DEFAULT_BOOTSTRAP_REPLICATES: usize = 10_000;
const DEFAULT_SEED: u64 = 0x5049_5442_5345_4c45;
const MIN_COMPONENTS_PER_CLASS: usize = 20;
const REQUIRED_REVIEWERS: usize = 2;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Split {
    Train,
    Validation,
    Test,
    ExternalChallenge,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::ExternalChallenge => "external_challenge",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Label {
    Compatible,
    Incompatible,
    Partial,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AdjudicationStatus {
    Adjudicated,
    Unresolved,
    Pending,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Decision {
    Accept,
    Reject,
    Abstain,
}

#[derive(Debug, Deserialize)]
struct Outcome {
    schema_version: String,
    pair_id: String,
    split: Split,
    component_id: String,
    source: SourceRef,
    target: TargetRef,
    adjudication: Adjudication,
    prediction: Prediction,
}

#[derive(Debug, Deserialize)]
struct SourceRef {
    record_id: String,
    family_id: String,
}

#[derive(Debug, Deserialize)]
struct TargetRef {
    family_id: String,
    obligation_sha256: String,
}

#[derive(Debug, Deserialize)]
struct Adjudication {
    status: AdjudicationStatus,
    label: Label,
    reviewer_decisions: Vec<ReviewerDecision>,
}

#[derive(Debug, Deserialize)]
struct ReviewerDecision {
    reviewer_id: String,
    label: Label,
}

#[derive(Debug, Deserialize)]
struct Prediction {
    decision: Decision,
    predicted_label: Label,
    #[serde(deserialize_with = "deserialize_required_nullable_f64")]
    compatibility_probability: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
struct Counts {
    rows: usize,
    adjudicated_compatible: usize,
    adjudicated_incompatible: usize,
    adjudicated_partial: usize,
    unresolved_or_unknown: usize,
    pending: usize,
    accept_decisions: usize,
    reject_decisions: usize,
    abstain_decisions: usize,
    binary_evaluable_pairs: usize,
    compatible_components: usize,
    incompatible_components: usize,
    unique_source_families: usize,
    unique_target_families: usize,
    unique_pair_components: usize,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
struct CellCounts {
    compatible: usize,
    compatible_accept: usize,
    compatible_reject: usize,
    compatible_abstain: usize,
    incompatible: usize,
    incompatible_accept: usize,
    incompatible_reject: usize,
    incompatible_abstain: usize,
}

impl CellCounts {
    fn add_outcome(&mut self, outcome: &Outcome) {
        match outcome.adjudication.label {
            Label::Compatible
                if outcome.adjudication.status == AdjudicationStatus::Adjudicated =>
            {
                self.compatible += 1;
                match outcome.prediction.decision {
                    Decision::Accept => self.compatible_accept += 1,
                    Decision::Reject => self.compatible_reject += 1,
                    Decision::Abstain => self.compatible_abstain += 1,
                }
            }
            Label::Incompatible
                if outcome.adjudication.status == AdjudicationStatus::Adjudicated =>
            {
                self.incompatible += 1;
                match outcome.prediction.decision {
                    Decision::Accept => self.incompatible_accept += 1,
                    Decision::Reject => self.incompatible_reject += 1,
                    Decision::Abstain => self.incompatible_abstain += 1,
                }
            }
            _ => {}
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.compatible += other.compatible;
        self.compatible_accept += other.compatible_accept;
        self.compatible_reject += other.compatible_reject;
        self.compatible_abstain += other.compatible_abstain;
        self.incompatible += other.incompatible;
        self.incompatible_accept += other.incompatible_accept;
        self.incompatible_reject += other.incompatible_reject;
        self.incompatible_abstain += other.incompatible_abstain;
    }

    fn metric_values(self) -> BTreeMap<&'static str, Option<f64>> {
        let total_binary = self.compatible + self.incompatible;
        let accepted_or_rejected = self.compatible_accept
            + self.compatible_reject
            + self.incompatible_accept
            + self.incompatible_reject;
        let accepted_compatible = self.compatible_accept;
        let rejected_compatible = self.compatible_reject;
        let accepted_incompatible = self.incompatible_accept;
        let false_decisions = rejected_compatible + accepted_incompatible;

        BTreeMap::from([
            (
                "compatible_accept_recall",
                ratio(accepted_compatible, self.compatible),
            ),
            (
                "incompatible_false_accept_rate",
                ratio(accepted_incompatible, self.incompatible),
            ),
            (
                "compatible_false_reject_rate",
                ratio(rejected_compatible, self.compatible),
            ),
            (
                "compatible_abstention_rate",
                ratio(self.compatible_abstain, self.compatible),
            ),
            (
                "incompatible_abstention_rate",
                ratio(self.incompatible_abstain, self.incompatible),
            ),
            (
                "binary_coverage",
                ratio(accepted_or_rejected, total_binary),
            ),
            (
                "selective_risk",
                ratio(false_decisions, accepted_or_rejected),
            ),
        ])
    }
}

#[derive(Debug, Serialize)]
struct Interval {
    lower: f64,
    upper: f64,
}

#[derive(Debug, Serialize)]
struct MetricReport {
    numerator: usize,
    denominator: usize,
    estimate: Option<f64>,
    ci_95: Option<Interval>,
    bootstrap_defined_replicates: usize,
    bootstrap_undefined_replicates: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    input_sha256: String,
    split: &'static str,
    bootstrap_seed: u64,
    bootstrap_replicates: usize,
    counts: Counts,
    metrics: BTreeMap<String, MetricReport>,
    calibration: CalibrationAssessment,
    proposed_gates: ProposedGates,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct CalibrationAssessment {
    eligible_binary_pairs: usize,
    probability_scored_pairs: usize,
    probability_coverage: f64,
    complete_binary_coverage: bool,
    brier_score: Option<f64>,
    expected_calibration_error_10_bins: Option<f64>,
    reliability_bins: Vec<ReliabilityBin>,
}

#[derive(Debug, Serialize)]
struct ReliabilityBin {
    bin_index: usize,
    lower_bound: f64,
    upper_bound: f64,
    count: usize,
    mean_predicted_compatibility: f64,
    observed_compatibility_rate: f64,
}

#[derive(Debug, Serialize)]
struct ProposedGates {
    minimum_compatible_components: usize,
    observed_compatible_components: usize,
    compatible_component_floor_met: bool,
    minimum_incompatible_components: usize,
    observed_incompatible_components: usize,
    incompatible_component_floor_met: bool,
    minimum_compatible_recall: f64,
    compatible_recall_target_met: bool,
    pair_selectivity_gate: &'static str,
    finite_mutation_suite_gate: &'static str,
    overall_confirmatory_decision: &'static str,
}

#[derive(Debug, Default, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() as u128 * upper_exclusive as u128 >> 64) as usize
    }
}

fn main() -> ExitCode {
    match run_from_args() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB transfer-selectivity analysis failed closed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_from_args() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 6 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "usage: {} <outcomes.jsonl> <test|external_challenge> <report.json> [bootstrap_replicates] [seed]",
            args.first().map(String::as_str).unwrap_or("analyze_transfer_selectivity")
        );
        if args.len() < 4 && args.get(1).is_some_and(|arg| arg == "--help" || arg == "-h") {
            return Ok(());
        }
        bail!("invalid command-line arguments");
    }

    let expected_split: Split = match args[2].as_str() {
        "test" => Split::Test,
        "external_challenge" => Split::ExternalChallenge,
        other => bail!("confirmatory analysis only accepts test or external_challenge, found {other:?}"),
    };
    let replicates = args
        .get(4)
        .map(|value| value.parse::<usize>().context("parse bootstrap replicate count"))
        .transpose()?
        .unwrap_or(DEFAULT_BOOTSTRAP_REPLICATES);
    if replicates < 1_000 {
        bail!("at least 1000 bootstrap replicates are required for an analysis report");
    }
    let seed = args
        .get(5)
        .map(|value| value.parse::<u64>().context("parse bootstrap seed"))
        .transpose()?
        .unwrap_or(DEFAULT_SEED);

    let input_path = &args[1];
    let output_path = &args[3];
    let bytes = fs::read(input_path).with_context(|| format!("read outcomes {}", input_path))?;
    let input_digest = sha256_hex(&bytes);
    let text = std::str::from_utf8(&bytes).context("outcomes JSONL must be UTF-8")?;
    let outcomes = parse_jsonl(text, expected_split)?;
    let report = analyze(&outcomes, expected_split, input_digest, replicates, seed)?;
    let output = serde_json::to_vec_pretty(&report).context("serialize analysis report")?;
    let mut output = output;
    output.push(b'\n');
    fs::write(output_path, output).with_context(|| format!("write report {output_path}"))?;
    println!(
        "PITB_TRANSFER_SELECTIVITY_ANALYSIS: split={} rows={} binary_pairs={} compatible_components={} incompatible_components={} recall={:?} false_accept={:?} decision={}",
        expected_split.as_str(),
        report.counts.rows,
        report.counts.binary_evaluable_pairs,
        report.counts.compatible_components,
        report.counts.incompatible_components,
        report.metrics["compatible_accept_recall"].estimate,
        report.metrics["incompatible_false_accept_rate"].estimate,
        report.proposed_gates.overall_confirmatory_decision
    );
    Ok(())
}

fn parse_jsonl(text: &str, expected_split: Split) -> Result<Vec<Outcome>> {
    let mut outcomes = Vec::new();
    let mut pair_ids = BTreeSet::new();
    let mut pair_keys = BTreeSet::new();
    let mut family_components = BTreeMap::<String, String>::new();

    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let outcome: Outcome = serde_json::from_str(line)
            .with_context(|| format!("parse outcome JSONL line {}", line_index + 1))?;
        if outcome.schema_version != SCHEMA_VERSION {
            bail!(
                "line {} has schema_version {:?}, expected {:?}",
                line_index + 1,
                outcome.schema_version,
                SCHEMA_VERSION
            );
        }
        if outcome.split != expected_split {
            bail!(
                "line {} belongs to split {}, but invocation requested {}",
                line_index + 1,
                outcome.split.as_str(),
                expected_split.as_str()
            );
        }
        if outcome.pair_id.trim().is_empty()
            || outcome.component_id.trim().is_empty()
            || outcome.source.record_id.trim().is_empty()
            || outcome.source.family_id.trim().is_empty()
            || outcome.target.family_id.trim().is_empty()
        {
            bail!("line {} contains an empty identity field", line_index + 1);
        }
        if !pair_ids.insert(outcome.pair_id.clone()) {
            bail!("duplicate pair_id {:?}", outcome.pair_id);
        }
        let pair_key = format!(
            "{}\0{}\0{}",
            outcome.source.record_id, outcome.target.family_id, outcome.target.obligation_sha256
        );
        if !pair_keys.insert(pair_key) {
            bail!(
                "duplicate source-insight/target-obligation pair at {}",
                outcome.pair_id
            );
        }

        let mut reviewer_ids = BTreeSet::new();
        for reviewer in &outcome.adjudication.reviewer_decisions {
            if reviewer.reviewer_id.trim().is_empty() || !reviewer_ids.insert(&reviewer.reviewer_id) {
                bail!(
                    "pair {} has an empty or duplicate reviewer ID",
                    outcome.pair_id
                );
            }
        }
        match outcome.adjudication.status {
            AdjudicationStatus::Adjudicated => {
                if !matches!(
                    outcome.adjudication.label,
                    Label::Compatible | Label::Incompatible | Label::Partial
                ) {
                    bail!("adjudicated pair {} cannot have unknown gold label", outcome.pair_id);
                }
                if reviewer_ids.len() < REQUIRED_REVIEWERS {
                    bail!("adjudicated pair {} needs at least two independent reviewer IDs", outcome.pair_id);
                }
                if outcome
                    .adjudication
                    .reviewer_decisions
                    .iter()
                    .any(|review| review.label != outcome.adjudication.label)
                {
                    bail!(
                        "adjudicated pair {} has reviewer labels inconsistent with the adjudicated label",
                        outcome.pair_id
                    );
                }
            }
            AdjudicationStatus::Unresolved | AdjudicationStatus::Pending => {
                if outcome.adjudication.label != Label::Unknown {
                    bail!(
                        "unresolved/pending pair {} must retain unknown gold label",
                        outcome.pair_id
                    );
                }
            }
        }
        if outcome
            .prediction
            .compatibility_probability
            .is_some_and(|probability| !probability.is_finite() || !(0.0..=1.0).contains(&probability))
        {
            bail!("pair {} has compatibility probability outside [0, 1]", outcome.pair_id);
        }
        match outcome.prediction.decision {
            Decision::Accept if outcome.prediction.predicted_label != Label::Compatible => {
                bail!("pair {} accepts but does not predict compatible", outcome.pair_id);
            }
            Decision::Reject
                if !matches!(
                    outcome.prediction.predicted_label,
                    Label::Incompatible | Label::Partial
                ) =>
            {
                bail!("pair {} rejects without predicting incompatible/partial", outcome.pair_id);
            }
            _ => {}
        }

        for family_id in [&outcome.source.family_id, &outcome.target.family_id] {
            if let Some(previous) = family_components.get(family_id) {
                if previous != &outcome.component_id {
                    bail!(
                        "family {} appears in components {} and {}; component IDs must encode family-connected components",
                        family_id, previous, outcome.component_id
                    );
                }
            } else {
                family_components.insert(family_id.clone(), outcome.component_id.clone());
            }
        }
        outcomes.push(outcome);
    }
    if outcomes.is_empty() {
        bail!("outcome JSONL contains no records");
    }
    Ok(outcomes)
}

fn analyze(
    outcomes: &[Outcome],
    split: Split,
    input_digest: String,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Report> {
    let mut counts = Counts {
        rows: outcomes.len(),
        ..Counts::default()
    };
    let mut source_families = BTreeSet::new();
    let mut target_families = BTreeSet::new();
    let mut all_components = BTreeSet::new();
    let mut class_components = BTreeMap::<String, CellCounts>::new();

    for outcome in outcomes {
        source_families.insert(outcome.source.family_id.clone());
        target_families.insert(outcome.target.family_id.clone());
        all_components.insert(outcome.component_id.clone());
        match outcome.adjudication.status {
            AdjudicationStatus::Adjudicated => match outcome.adjudication.label {
                Label::Compatible => counts.adjudicated_compatible += 1,
                Label::Incompatible => counts.adjudicated_incompatible += 1,
                Label::Partial => counts.adjudicated_partial += 1,
                Label::Unknown => bail!("adjudicated unknown label passed parsing unexpectedly"),
            },
            AdjudicationStatus::Unresolved => counts.unresolved_or_unknown += 1,
            AdjudicationStatus::Pending => counts.pending += 1,
        }
        match outcome.prediction.decision {
            Decision::Accept => counts.accept_decisions += 1,
            Decision::Reject => counts.reject_decisions += 1,
            Decision::Abstain => counts.abstain_decisions += 1,
        }
        class_components
            .entry(outcome.component_id.clone())
            .or_default()
            .add_outcome(outcome);
    }

    counts.unique_source_families = source_families.len();
    counts.unique_target_families = target_families.len();
    counts.unique_pair_components = all_components.len();
    let compatible_component_ids: BTreeSet<&str> = class_components
        .iter()
        .filter(|(_, row)| row.compatible > 0)
        .map(|(id, _)| id.as_str())
        .collect();
    let incompatible_component_ids: BTreeSet<&str> = class_components
        .iter()
        .filter(|(_, row)| row.incompatible > 0)
        .map(|(id, _)| id.as_str())
        .collect();
    counts.compatible_components = compatible_component_ids.len();
    counts.incompatible_components = incompatible_component_ids.len();

    let mut observed = CellCounts::default();
    for row in class_components.values() {
        observed.add_assign(*row);
    }
    counts.binary_evaluable_pairs = observed.compatible + observed.incompatible;
    if counts.binary_evaluable_pairs == 0 {
        bail!("no adjudicated compatible/incompatible outcomes exist in selected split");
    }

    let metric_values = observed.metric_values();
    let mut bootstrap_values: BTreeMap<&'static str, Vec<f64>> = metric_values
        .keys()
        .map(|key| (*key, Vec::with_capacity(bootstrap_replicates)))
        .collect();
    let mut undefined_replicates: BTreeMap<&'static str, usize> =
        metric_values.keys().map(|key| (*key, 0)).collect();
    // Components containing only partial/unknown/pending labels must remain
    // accounted for in the report but cannot contribute zero-valued bootstrap
    // clusters to a binary compatibility metric.
    let clusters: Vec<CellCounts> = class_components
        .values()
        .copied()
        .filter(|row| row.compatible + row.incompatible > 0)
        .collect();
    let mut rng = SplitMix64::new(seed);
    for _ in 0..bootstrap_replicates {
        let mut sample = CellCounts::default();
        for _ in 0..clusters.len() {
            sample.add_assign(clusters[rng.index(clusters.len())]);
        }
        for (name, value) in sample.metric_values() {
            if let Some(value) = value {
                bootstrap_values.get_mut(name).expect("metric registered").push(value);
            } else {
                *undefined_replicates.get_mut(name).expect("metric registered") += 1;
            }
        }
    }

    let mut metrics = BTreeMap::new();
    for (name, estimate) in metric_values {
        let values = bootstrap_values.get_mut(name).expect("metric registered");
        values.sort_by(f64::total_cmp);
        let ci = if values.is_empty() {
            None
        } else {
            Some(Interval {
                lower: quantile(values, 0.025),
                upper: quantile(values, 0.975),
            })
        };
        let (numerator, denominator) = metric_counts(name, observed);
        metrics.insert(
            name.to_string(),
            MetricReport {
                numerator,
                denominator,
                estimate,
                ci_95: ci,
                bootstrap_defined_replicates: values.len(),
                bootstrap_undefined_replicates: undefined_replicates[name],
            },
        );
    }

    let calibration = calibration_assessment(outcomes);
    let recall = metrics["compatible_accept_recall"].estimate;
    let compatible_floor_met = counts.compatible_components >= MIN_COMPONENTS_PER_CLASS;
    let incompatible_floor_met = counts.incompatible_components >= MIN_COMPONENTS_PER_CLASS;
    let recall_target_met = recall.is_some_and(|value| value >= 0.90);
    let pair_gate_met = compatible_floor_met && incompatible_floor_met && recall_target_met;
    let gates = ProposedGates {
        minimum_compatible_components: MIN_COMPONENTS_PER_CLASS,
        observed_compatible_components: counts.compatible_components,
        compatible_component_floor_met: compatible_floor_met,
        minimum_incompatible_components: MIN_COMPONENTS_PER_CLASS,
        observed_incompatible_components: counts.incompatible_components,
        incompatible_component_floor_met: incompatible_floor_met,
        minimum_compatible_recall: 0.90,
        compatible_recall_target_met: recall_target_met,
        pair_selectivity_gate: if pair_gate_met { "pair_track_thresholds_met" } else { "pair_track_thresholds_not_met" },
        finite_mutation_suite_gate: "not_evaluated_by_this_analyzer",
        overall_confirmatory_decision: "not_evaluated_requires_independent_mutation_receipt_and_protocol_audit",
    };

    Ok(Report {
        schema_version: REPORT_SCHEMA_VERSION,
        input_sha256: input_digest,
        split: split.as_str(),
        bootstrap_seed: seed,
        bootstrap_replicates,
        counts,
        metrics,
        calibration,
        proposed_gates: gates,
        limitations: vec![
            "This analyzer computes metrics against supplied adjudication labels; it does not establish that the labels are mathematically correct.",
            "Bootstrap intervals resample declared family-connected components, not individual pairs; incorrect component IDs invalidate the inference.",
            "Unknown, pending, and partial labels are accounted for but excluded from binary compatibility metrics.",
            "Brier score and ten-bin expected calibration error are descriptive and require compatibility_probability coverage; missing probabilities are never imputed.",
            "The finite mutation-control suite must be checked through a separately retained receipt; this analyzer never converts a pass of pair metrics into an overall confirmatory claim.",
            "These outcomes concern transfer selectivity and do not replace the primary verified-proof utility comparison.",
        ],
    })
}

fn calibration_assessment(outcomes: &[Outcome]) -> CalibrationAssessment {
    let binary_rows: Vec<(Option<f64>, bool)> = outcomes
        .iter()
        .filter(|outcome| outcome.adjudication.status == AdjudicationStatus::Adjudicated)
        .filter_map(|outcome| match outcome.adjudication.label {
            Label::Compatible => Some((outcome.prediction.compatibility_probability, true)),
            Label::Incompatible => Some((outcome.prediction.compatibility_probability, false)),
            Label::Partial | Label::Unknown => None,
        })
        .collect();
    let scored: Vec<(f64, bool)> = binary_rows
        .iter()
        .filter_map(|(probability, truth)| probability.map(|value| (value, *truth)))
        .collect();
    let probability_coverage = ratio(scored.len(), binary_rows.len()).unwrap_or(0.0);
    if scored.is_empty() {
        return CalibrationAssessment {
            eligible_binary_pairs: binary_rows.len(),
            probability_scored_pairs: 0,
            probability_coverage,
            complete_binary_coverage: binary_rows.is_empty(),
            brier_score: None,
            expected_calibration_error_10_bins: None,
            reliability_bins: Vec::new(),
        };
    }

    let brier_score = scored
        .iter()
        .map(|(probability, truth)| {
            let target = if *truth { 1.0 } else { 0.0 };
            (probability - target).powi(2)
        })
        .sum::<f64>()
        / scored.len() as f64;

    let mut bins: Vec<Vec<(f64, bool)>> = (0..10).map(|_| Vec::new()).collect();
    for (probability, truth) in &scored {
        let bin_index = ((*probability * 10.0).floor() as usize).min(9);
        bins[bin_index].push((*probability, *truth));
    }

    let mut ece = 0.0;
    let mut reliability_bins = Vec::new();
    for (index, values) in bins.into_iter().enumerate() {
        if values.is_empty() {
            continue;
        }
        let count = values.len();
        let mean_probability = values.iter().map(|(p, _)| p).sum::<f64>() / count as f64;
        let observed_rate =
            values.iter().filter(|(_, truth)| *truth).count() as f64 / count as f64;
        ece += count as f64 / scored.len() as f64 * (mean_probability - observed_rate).abs();
        reliability_bins.push(ReliabilityBin {
            bin_index: index,
            lower_bound: index as f64 / 10.0,
            upper_bound: (index + 1) as f64 / 10.0,
            count,
            mean_predicted_compatibility: mean_probability,
            observed_compatibility_rate: observed_rate,
        });
    }

    CalibrationAssessment {
        eligible_binary_pairs: binary_rows.len(),
        probability_scored_pairs: scored.len(),
        probability_coverage,
        complete_binary_coverage: scored.len() == binary_rows.len(),
        brier_score: Some(brier_score),
        expected_calibration_error_10_bins: Some(ece),
        reliability_bins,
    }
}

fn metric_counts(name: &str, cells: CellCounts) -> (usize, usize) {
    match name {
        "compatible_accept_recall" => (cells.compatible_accept, cells.compatible),
        "incompatible_false_accept_rate" => (cells.incompatible_accept, cells.incompatible),
        "compatible_false_reject_rate" => (cells.compatible_reject, cells.compatible),
        "compatible_abstention_rate" => (cells.compatible_abstain, cells.compatible),
        "incompatible_abstention_rate" => (cells.incompatible_abstain, cells.incompatible),
        "binary_coverage" => (
            cells.compatible_accept
                + cells.compatible_reject
                + cells.incompatible_accept
                + cells.incompatible_reject,
            cells.compatible + cells.incompatible,
        ),
        "selective_risk" => (
            cells.compatible_reject + cells.incompatible_accept,
            cells.compatible_accept
                + cells.compatible_reject
                + cells.incompatible_accept
                + cells.incompatible_reject,
        ),
        _ => unreachable!("only registered metrics are queried"),
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    let position = q * (sorted_values.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let weight = position - low as f64;
    sorted_values[low] + (sorted_values[high] - sorted_values[low]) * weight
}

fn deserialize_required_nullable_f64<'de, D>(deserializer: D) -> std::result::Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    if value.is_some_and(|probability| {
        !probability.is_finite() || !(0.0..=1.0).contains(&probability)
    }) {
        return Err(<D::Error as serde::de::Error>::custom(
            "compatibility_probability must be null or a finite value in [0, 1]",
        ));
    }
    Ok(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outcome(
        pair_id: &str,
        component_id: &str,
        source_family: &str,
        target_family: &str,
        label: Label,
        decision: Decision,
    ) -> Outcome {
        let predicted_label = match decision {
            Decision::Accept => Label::Compatible,
            Decision::Reject => Label::Incompatible,
            Decision::Abstain => Label::Unknown,
        };
        let reviewer_decisions = vec![
            ReviewerDecision { reviewer_id: "reviewer-a".to_string(), label },
            ReviewerDecision { reviewer_id: "reviewer-b".to_string(), label },
        ];
        let json = json!({
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "split": "test",
            "component_id": component_id,
            "source": {"record_id": format!("source-{pair_id}"), "family_id": source_family},
            "target": {"family_id": target_family, "obligation_sha256": format!("{:064x}", pair_id.len())},
            "adjudication": {
                "status": "adjudicated",
                "label": label,
                "reviewer_decisions": reviewer_decisions.iter().map(|review| json!({
                    "reviewer_id": review.reviewer_id,
                    "label": review.label
                })).collect::<Vec<_>>()
            },
            "prediction": {"decision": decision, "predicted_label": predicted_label, "compatibility_probability": if decision == Decision::Accept { Some(0.9) } else if decision == Decision::Reject { Some(0.1) } else { None }}
        });
        serde_json::from_value(json).expect("valid test fixture")
    }

    #[test]
    fn computes_transfer_selectivity_metrics_and_excludes_partial_rows() {
        let positive = outcome("p", "cp", "sf-p", "tf-p", Label::Compatible, Decision::Accept);
        let negative = outcome("n", "cn", "sf-n", "tf-n", Label::Incompatible, Decision::Reject);
        let report = analyze(&[positive, negative], Split::Test, "a".repeat(64), 1_000, 7)
            .expect("analyze synthetic fixture");
        assert_eq!(report.counts.binary_evaluable_pairs, 2);
        assert_eq!(report.metrics["compatible_accept_recall"].estimate, Some(1.0));
        assert_eq!(report.metrics["incompatible_false_accept_rate"].estimate, Some(0.0));
        assert_eq!(report.metrics["binary_coverage"].estimate, Some(1.0));
        assert_eq!(report.metrics["selective_risk"].estimate, Some(0.0));
        assert_eq!(report.calibration.probability_scored_pairs, 2);
        assert!(report.calibration.complete_binary_coverage);
        assert!(report.calibration.brier_score.is_some());
        assert!((report.calibration.brier_score.unwrap() - 0.01).abs() < 1e-12);
        assert!(report.calibration.expected_calibration_error_10_bins.is_some());
        assert_eq!(
            report.proposed_gates.overall_confirmatory_decision,
            "not_evaluated_requires_independent_mutation_receipt_and_protocol_audit"
        );
    }

    #[test]
    fn abstention_is_not_silently_counted_as_accept_or_reject() {
        let positive = outcome("p", "cp", "sf-p", "tf-p", Label::Compatible, Decision::Abstain);
        let negative = outcome("n", "cn", "sf-n", "tf-n", Label::Incompatible, Decision::Abstain);
        let report = analyze(&[positive, negative], Split::Test, "b".repeat(64), 1_000, 8)
            .expect("analyze abstention fixture");
        assert_eq!(report.metrics["compatible_accept_recall"].estimate, Some(0.0));
        assert_eq!(report.metrics["incompatible_false_accept_rate"].estimate, Some(0.0));
        assert_eq!(report.metrics["binary_coverage"].estimate, Some(0.0));
        assert_eq!(report.metrics["selective_risk"].estimate, None);
        assert_eq!(report.calibration.probability_scored_pairs, 0);
        assert_eq!(report.calibration.probability_coverage, 0.0);
    }

    #[test]
    fn deterministic_bootstrap_yields_stable_intervals() {
        let outcomes = vec![
            outcome("p1", "c1", "sf1", "tf1", Label::Compatible, Decision::Accept),
            outcome("n1", "c2", "sf2", "tf2", Label::Incompatible, Decision::Accept),
            outcome("p2", "c3", "sf3", "tf3", Label::Compatible, Decision::Reject),
            outcome("n2", "c4", "sf4", "tf4", Label::Incompatible, Decision::Reject),
        ];
        let first = analyze(&outcomes, Split::Test, "c".repeat(64), 1_000, 42).unwrap();
        let second = analyze(&outcomes, Split::Test, "c".repeat(64), 1_000, 42).unwrap();
        assert_eq!(
            serde_json::to_value(first).unwrap(),
            serde_json::to_value(second).unwrap()
        );
    }

    #[test]
    fn family_component_floor_counts_components_not_pair_rows() {
        let shared = outcome("p", "one-component", "shared-family", "target-p", Label::Compatible, Decision::Accept);
        let different_edge = outcome("n", "one-component", "target-p", "target-n", Label::Incompatible, Decision::Reject);
        let report = analyze(&[shared, different_edge], Split::Test, "d".repeat(64), 1_000, 9).unwrap();
        assert_eq!(report.counts.compatible_components, 1);
        assert_eq!(report.counts.incompatible_components, 1);
        assert!(!report.proposed_gates.compatible_component_floor_met);
        assert!(!report.proposed_gates.incompatible_component_floor_met);
    }

    #[test]
    fn compatibility_probability_must_be_present_even_when_null_is_allowed() {
        let missing = serde_json::from_value::<Prediction>(json!({
            "decision": "accept",
            "predicted_label": "compatible"
        }));
        assert!(missing.is_err());

        let explicit_null = serde_json::from_value::<Prediction>(json!({
            "decision": "abstain",
            "predicted_label": "unknown",
            "compatibility_probability": null
        }));
        assert!(explicit_null.is_ok());

        let out_of_range = serde_json::from_value::<Prediction>(json!({
            "decision": "accept",
            "predicted_label": "compatible",
            "compatibility_probability": 1.01
        }));
        assert!(out_of_range.is_err());
    }

    #[test]
    fn quantile_uses_linear_interpolation() {
        assert!((quantile(&[0.0, 10.0], 0.25) - 2.5).abs() < f64::EPSILON);
    }
}
