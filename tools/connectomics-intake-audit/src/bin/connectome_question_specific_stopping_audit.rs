use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Audit global and question-specific connectome proofreading stopping")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    run_id: String,
    claim_scope: String,
    preregistration: Preregistration,
    query_binding: QueryBinding,
    sampling_frames: Vec<SamplingFrame>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preregistration {
    preregistration_sha256: String,
    frozen_before_sampling: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryBinding {
    query_id: String,
    query_spec_sha256: String,
    decision_loss_sha256: String,
    baseline_decision: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingFrame {
    frame_id: String,
    source_frame_receipt_sha256: String,
    candidate_frame_sha256: String,
    candidate_unit_ids: Vec<String>,
    sampling_design: String,
    sample_size: usize,
    sampled_unit_set_sha256: String,
    sample_outcome_ledger_sha256: String,
    sampling_receipt_sha256: String,
    sampled_units: Vec<SampleUnit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleUnit {
    unit_id: String,
    outcome: String,
    inclusion_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case_id: String,
    dataset_revision_sha256: String,
    frame_id: String,
    frame_coverage_status: String,
    frame_coverage_receipt_sha256: String,
    global_stop_policy: GlobalStopPolicy,
    snapshots: Vec<Snapshot>,
    worlds: Vec<World>,
    confidence_level: f64,
    residual_query_flip_tolerance: f64,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobalStopPolicy {
    relative_delta_tolerance: f64,
    consecutive_stable_transitions: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    snapshot_id: String,
    edit_count: u64,
    graph_invariants: Vec<GraphInvariant>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphInvariant {
    invariant_id: String,
    value: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct World {
    world_id: String,
    biological_disposition: String,
    query_decision: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Expected {
    global_stop_snapshot_id: Option<String>,
    query_world_agreement: bool,
    residual_upper_count: Option<usize>,
    residual_upper_prevalence: Option<f64>,
    query_stop_status: String,
    joint_disposition: String,
}

#[derive(Debug, Serialize, PartialEq)]
struct CaseResult {
    global_stop_snapshot_id: Option<String>,
    query_world_agreement: bool,
    residual_upper_count: usize,
    residual_upper_prevalence: f64,
    query_stop_status: String,
    joint_disposition: String,
}

#[derive(Debug, Serialize)]
struct KnownAnswer {
    finite_frame_population: usize,
    sample_size: usize,
    observed_query_flips: usize,
    confidence_level: f64,
    upper_count: usize,
    upper_prevalence: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    status: &'static str,
    claim_scope: String,
    query_id: String,
    sampling_frame_count: usize,
    case_count: usize,
    results: Vec<CaseReport>,
    known_answer: KnownAnswer,
    claim_ceiling: &'static str,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    case_id: String,
    #[serde(flatten)]
    result: CaseResult,
}

fn sha256_text(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

#[derive(Debug)]
struct FrameStats {
    population_size: usize,
    sample_size: usize,
    observed_query_flips: usize,
    unresolved_sample_units: usize,
}

fn validate_frame(frame: &SamplingFrame) -> Result<FrameStats, String> {
    let ids = &frame.candidate_unit_ids;
    require(!ids.is_empty(), "candidate frame is empty")?;
    require(ids.windows(2).all(|window| window[0] < window[1]), "candidate frame IDs must be unique and sorted")?;
    let population = ids.len();
    require(
        frame.candidate_frame_sha256 == sha256_text(&format!("{}\n", ids.join("\n"))),
        "candidate frame digest mismatch",
    )?;
    require(valid_sha256(&frame.source_frame_receipt_sha256), "source frame receipt digest malformed")?;
    let expected_source_receipt = sha256_text(&format!(
        "synthetic-frame-v1\nframe-size={}\n{}\n",
        population, frame.candidate_frame_sha256
    ));
    require(frame.source_frame_receipt_sha256 == expected_source_receipt, "source frame receipt does not bind the exact candidate frame")?;
    require(frame.sampling_design == "SRS_WITHOUT_REPLACEMENT", "V1 only supports SRS without replacement")?;

    let sampled = &frame.sampled_units;
    require(frame.sample_size == sampled.len(), "declared sample size differs from sample ledger")?;
    require(!sampled.is_empty() && sampled.len() <= population, "sample size is outside the finite frame")?;
    require(
        sampled.windows(2).all(|window| window[0].unit_id < window[1].unit_id),
        "sample unit IDs must be unique and sorted",
    )?;
    let candidate_set: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
    require(sampled.iter().all(|row| candidate_set.contains(row.unit_id.as_str())), "sample contains unit outside frozen candidate frame")?;
    let inclusion = sampled.len() as f64 / population as f64;
    for row in sampled {
        require(
            ["QUERY_FLIP", "NO_QUERY_FLIP", "UNRESOLVED"].contains(&row.outcome.as_str()),
            "unknown sample outcome",
        )?;
        require(row.inclusion_probability.is_finite(), "inclusion probability is not finite")?;
        require(row.inclusion_probability > 0.0 && row.inclusion_probability <= 1.0, "inclusion probability must be positive and at most one")?;
        require((row.inclusion_probability - inclusion).abs() <= 1e-12, "inclusion probability differs from SRSWOR n/N")?;
    }

    let sample_ids: Vec<&str> = sampled.iter().map(|row| row.unit_id.as_str()).collect();
    let sample_ids_sha = sha256_text(&format!("{}\n", sample_ids.join("\n")));
    require(frame.sampled_unit_set_sha256 == sample_ids_sha, "sample unit-set digest mismatch")?;
    let ledger = sampled
        .iter()
        .map(|row| format!("{}\t{}\t{:.6}\n", row.unit_id, row.outcome, row.inclusion_probability))
        .collect::<String>();
    let ledger_sha = sha256_text(&ledger);
    require(frame.sample_outcome_ledger_sha256 == ledger_sha, "sample outcome ledger digest mismatch")?;
    let receipt = format!(
        "SRS_WOR_V1\nN={}\nn={}\n{}\n{}\n{}\n",
        population, sampled.len(), frame.candidate_frame_sha256, sample_ids_sha, ledger_sha
    );
    require(frame.sampling_receipt_sha256 == sha256_text(&receipt), "sampling receipt digest mismatch")?;

    Ok(FrameStats {
        population_size: population,
        sample_size: sampled.len(),
        observed_query_flips: sampled.iter().filter(|row| row.outcome == "QUERY_FLIP").count(),
        unresolved_sample_units: sampled.iter().filter(|row| row.outcome == "UNRESOLVED").count(),
    })
}

fn log_factorial(value: usize) -> f64 {
    (2..=value).map(|number| (number as f64).ln()).sum()
}

fn log_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

fn hypergeometric_cdf(population: usize, positives: usize, sample_size: usize, observed: usize) -> f64 {
    if positives > population || sample_size > population {
        return 0.0;
    }
    let lower = sample_size.saturating_sub(population - positives);
    let upper = observed.min(sample_size).min(positives);
    if upper < lower {
        return 0.0;
    }
    let denominator = log_choose(population, sample_size);
    let log_probabilities: Vec<f64> = (lower..=upper)
        .map(|hits| {
            log_choose(positives, hits)
                + log_choose(population - positives, sample_size - hits)
                - denominator
        })
        .collect();
    let peak = log_probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return 0.0;
    }
    (peak.exp()
        * log_probabilities
            .iter()
            .map(|value| (value - peak).exp())
            .sum::<f64>())
    .clamp(0.0, 1.0)
}

fn upper_population_count(population: usize, sample_size: usize, observed: usize, alpha: f64) -> Result<usize, String> {
    require(population > 0 && sample_size > 0 && sample_size <= population, "invalid finite frame or sample size")?;
    require(observed <= sample_size, "observed flip count exceeds sample size")?;
    require(alpha.is_finite() && alpha > 0.0 && alpha < 1.0, "alpha must be strictly between zero and one")?;
    let mut low = 0usize;
    let mut high = population;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if hypergeometric_cdf(population, middle, sample_size, observed) >= alpha {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

fn global_stop(case: &Case) -> Result<Option<String>, String> {
    require(case.snapshots.len() >= 2, "at least two global snapshots are required")?;
    require(case.global_stop_policy.relative_delta_tolerance.is_finite() && case.global_stop_policy.relative_delta_tolerance > 0.0, "global drift tolerance is invalid")?;
    require(case.global_stop_policy.consecutive_stable_transitions > 0, "stable transition count must be positive")?;
    for pair in case.snapshots.windows(2) {
        require(pair[0].edit_count < pair[1].edit_count, "snapshot edit counts must strictly increase")?;
    }
    let first_names: BTreeSet<&str> = case.snapshots[0].graph_invariants.iter().map(|row| row.invariant_id.as_str()).collect();
    require(!first_names.is_empty(), "graph invariant set is empty")?;
    for snapshot in &case.snapshots {
        let names: BTreeSet<&str> = snapshot.graph_invariants.iter().map(|row| row.invariant_id.as_str()).collect();
        require(names.len() == snapshot.graph_invariants.len(), "duplicate invariant ID in a snapshot")?;
        require(names == first_names, "graph invariant identity changed between checkpoints")?;
        require(snapshot.graph_invariants.iter().all(|row| row.value.is_finite()), "graph invariant value is not finite")?;
    }

    let mut stable_run = 0usize;
    for index in 1..case.snapshots.len() {
        let previous: std::collections::BTreeMap<&str, f64> = case.snapshots[index - 1]
            .graph_invariants.iter().map(|row| (row.invariant_id.as_str(), row.value)).collect();
        let stable = case.snapshots[index].graph_invariants.iter().all(|row| {
            let prior = previous[row.invariant_id.as_str()];
            (row.value - prior).abs() / prior.abs().max(1e-12)
                <= case.global_stop_policy.relative_delta_tolerance
        });
        stable_run = if stable { stable_run + 1 } else { 0 };
        if stable_run >= case.global_stop_policy.consecutive_stable_transitions {
            return Ok(Some(case.snapshots[index].snapshot_id.clone()));
        }
    }
    Ok(None)
}

fn world_agreement(worlds: &[World]) -> Result<(bool, bool), String> {
    require(!worlds.is_empty(), "no reconstruction worlds were supplied")?;
    let mut unresolved = false;
    let mut admissible_decisions = BTreeSet::new();
    for world in worlds {
        match world.biological_disposition.as_str() {
            "ADMISSIBLE" => {
                let decision = world.query_decision.ok_or_else(|| "admissible world lacks query decision".to_owned())?;
                admissible_decisions.insert(decision);
            }
            "INADMISSIBLE" => require(world.query_decision.is_none(), "inadmissible world must not carry query decision")?,
            "INDETERMINATE" | "DISPUTED" => {
                require(world.query_decision.is_none(), "indeterminate/disputed world must not carry promoted query decision")?;
                unresolved = true;
            }
            _ => return Err("unknown biological disposition".to_owned()),
        }
    }
    if unresolved || admissible_decisions.is_empty() {
        return Ok((false, true));
    }
    Ok((admissible_decisions.len() == 1, false))
}

fn evaluate_case(case: &Case, frame: &SamplingFrame) -> Result<CaseResult, String> {
    require(case.frame_id == frame.frame_id, "case references a different sampling frame")?;
    require(valid_sha256(&case.dataset_revision_sha256), "dataset revision digest malformed")?;
    require(valid_sha256(&case.frame_coverage_receipt_sha256), "frame coverage receipt digest malformed")?;
    require(case.frame_coverage_status == "CLOSED" || case.frame_coverage_status == "INCOMPLETE", "unknown frame coverage status")?;
    require(case.confidence_level.is_finite() && case.confidence_level > 0.0 && case.confidence_level < 1.0, "confidence level is invalid")?;
    require(case.residual_query_flip_tolerance.is_finite() && case.residual_query_flip_tolerance > 0.0 && case.residual_query_flip_tolerance <= 1.0, "query-flip tolerance is invalid")?;
    let stats = validate_frame(frame)?;
    let global_snapshot = global_stop(case)?;
    let (agreement, unresolved_worlds) = world_agreement(&case.worlds)?;
    let alpha = 1.0 - case.confidence_level;
    let upper_count = upper_population_count(stats.population_size, stats.sample_size, stats.observed_query_flips, alpha)?;
    let upper_prevalence = upper_count as f64 / stats.population_size as f64;

    let query_status = if unresolved_worlds || stats.unresolved_sample_units > 0 {
        "NOT_ESTIMABLE"
    } else if case.frame_coverage_status == "INCOMPLETE" {
        "FRAME_INCOMPLETE"
    } else if !agreement {
        "QUERY_UNRESOLVED"
    } else if upper_prevalence <= case.residual_query_flip_tolerance {
        "QUERY_ROBUST_WITHIN_DECLARED_FRAME"
    } else {
        "QUERY_UNRESOLVED"
    };
    let joint = if query_status == "QUERY_ROBUST_WITHIN_DECLARED_FRAME" {
        query_status
    } else if query_status == "FRAME_INCOMPLETE" || query_status == "NOT_ESTIMABLE" {
        query_status
    } else if global_snapshot.is_some() {
        "GLOBAL_ONLY"
    } else {
        "QUERY_UNRESOLVED"
    };

    Ok(CaseResult {
        global_stop_snapshot_id: global_snapshot,
        query_world_agreement: agreement,
        residual_upper_count: upper_count,
        residual_upper_prevalence: upper_prevalence,
        query_stop_status: query_status.to_owned(),
        joint_disposition: joint.to_owned(),
    })
}

fn run(input_bytes: &[u8]) -> Result<Report, String> {
    let input: Input = serde_json::from_slice(input_bytes).map_err(|error| format!("invalid strict input: {error}"))?;
    require(input.schema_version == 1, "unsupported schema version")?;
    require(input.claim_scope == "synthetic_known_truth", "V1 only accepts synthetic known-truth fixtures")?;
    require(input.preregistration.frozen_before_sampling, "preregistration was not frozen before sampling")?;
    require(valid_sha256(&input.preregistration.preregistration_sha256), "preregistration digest malformed")?;
    require(!input.run_id.trim().is_empty(), "run ID is empty")?;
    require(!input.query_binding.query_id.trim().is_empty(), "query ID is empty")?;
    require(valid_sha256(&input.query_binding.query_spec_sha256), "query specification digest malformed")?;
    require(valid_sha256(&input.query_binding.decision_loss_sha256), "decision-loss digest malformed")?;

    let mut frames = std::collections::BTreeMap::new();
    for frame in &input.sampling_frames {
        require(frames.insert(frame.frame_id.as_str(), frame).is_none(), "duplicate sampling frame ID")?;
    }
    require(!frames.is_empty(), "no sampling frame was supplied")?;
    for frame in frames.values() {
        validate_frame(frame)?;
    }

    let mut case_ids = BTreeSet::new();
    let mut results = Vec::new();
    for case in &input.cases {
        require(case_ids.insert(case.case_id.as_str()), "duplicate case ID")?;
        let frame = frames.get(case.frame_id.as_str()).ok_or_else(|| "case references unknown sampling frame".to_owned())?;
        let result = evaluate_case(case, frame)?;
        require(result == case.expected_to_result(), "case known-answer expectation mismatch")?;
        results.push(CaseReport { case_id: case.case_id.clone(), result });
    }
    require(results.len() >= 4, "synthetic adversarial corpus must contain at least four cases")?;

    Ok(Report {
        schema_version: 1,
        tool: "query-specific-connectome-stopping-oracle",
        status: "SYNTHETIC_CONTRACT_VALIDATED",
        claim_scope: input.claim_scope,
        query_id: input.query_binding.query_id,
        sampling_frame_count: frames.len(),
        case_count: results.len(),
        results,
        known_answer: KnownAnswer {
            finite_frame_population: 1000,
            sample_size: 300,
            observed_query_flips: 0,
            confidence_level: 0.95,
            upper_count: 8,
            upper_prevalence: 0.008,
        },
        claim_ceiling: "QUESTION_SPECIFIC_ONLY_WITHIN_FROZEN_QUERY_ADMITTED_WORLDS_CLOSED_FRAME_AND_DECLARED_SRSWOR_DESIGN",
    })
}

impl Case {
    fn expected_to_result(&self) -> CaseResult {
        CaseResult {
            global_stop_snapshot_id: self.expected.global_stop_snapshot_id.clone(),
            query_world_agreement: self.expected.query_world_agreement,
            residual_upper_count: self.expected.residual_upper_count.unwrap_or(usize::MAX),
            residual_upper_prevalence: self.expected.residual_upper_prevalence.unwrap_or(f64::NAN),
            query_stop_status: self.expected.query_stop_status.clone(),
            joint_disposition: self.expected.joint_disposition.clone(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input_bytes = fs::read(&args.input)?;
    let report = run(&input_bytes).map_err(std::io::Error::other)?;
    let rendered = serde_json::to_vec_pretty(&report)?;
    fs::write(&args.output, &rendered)?;
    println!(
        "question-specific stopping audit: PASS; cases={}; input_sha256={}",
        report.case_count,
        hex::encode(Sha256::digest(&input_bytes))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_flip_finite_frame_known_answer_is_eight() {
        assert_eq!(upper_population_count(1000, 300, 0, 0.05).unwrap(), 8);
    }

    #[test]
    fn global_stop_requires_two_consecutive_stable_transitions() {
        let case = Case {
            case_id: "global-test".into(),
            dataset_revision_sha256: "a".repeat(64),
            frame_id: "f".into(),
            frame_coverage_status: "CLOSED".into(),
            frame_coverage_receipt_sha256: "b".repeat(64),
            global_stop_policy: GlobalStopPolicy { relative_delta_tolerance: 0.005, consecutive_stable_transitions: 2 },
            snapshots: vec![
                Snapshot { snapshot_id: "S1".into(), edit_count: 1, graph_invariants: vec![GraphInvariant { invariant_id: "density".into(), value: 0.2 }] },
                Snapshot { snapshot_id: "S2".into(), edit_count: 2, graph_invariants: vec![GraphInvariant { invariant_id: "density".into(), value: 0.2008 }] },
                Snapshot { snapshot_id: "S3".into(), edit_count: 3, graph_invariants: vec![GraphInvariant { invariant_id: "density".into(), value: 0.2009 }] },
            ],
            worlds: vec![],
            confidence_level: 0.95,
            residual_query_flip_tolerance: 0.01,
            expected: Expected { global_stop_snapshot_id: Some("S3".into()), query_world_agreement: false, residual_upper_count: None, residual_upper_prevalence: None, query_stop_status: "QUERY_UNRESOLVED".into(), joint_disposition: "GLOBAL_ONLY".into() },
        };
        assert_eq!(global_stop(&case).unwrap(), Some("S3".into()));
    }

    #[test]
    fn indeterminate_world_never_counts_as_agreement() {
        let worlds = vec![
            World { world_id: "a".into(), biological_disposition: "ADMISSIBLE".into(), query_decision: Some(true) },
            World { world_id: "b".into(), biological_disposition: "INDETERMINATE".into(), query_decision: None },
        ];
        assert_eq!(world_agreement(&worlds).unwrap(), (false, true));
    }
}