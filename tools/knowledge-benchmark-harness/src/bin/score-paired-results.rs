use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TOOL_VERSION: &str = "score-paired-results-v0.11.0";
const SCORE_REPORT_SCHEMA: &str =
    include_str!("../../../../benchmarks/benchmark-runs/v0/score-report.schema.json");

#[derive(Debug)]
struct Args {
    plan: PathBuf,
    pairs: PathBuf,
    results: PathBuf,
    case_manifest: PathBuf,
    source_catalog: PathBuf,
    source_pack: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Plan {
    plan_id: String,
    phase: String,
    synthetic_only: bool,
    primary_endpoint: String,
    candidate_system_id: String,
    benchmark: Benchmark,
    systems: Vec<System>,
    preregistration: Preregistration,
    bootstrap: BootstrapPlan,
}

#[derive(Debug, Deserialize)]
struct Benchmark {
    split: String,
    source_family_disjoint: bool,
    rights_review_status: String,
    case_manifest_sha256: String,
    source_catalog_sha256: String,
    source_pack_sha256: String,
}

#[derive(Debug, Deserialize)]
struct System {
    system_id: String,
    baseline_family: String,
}

#[derive(Debug, Deserialize)]
struct Preregistration {
    held_out_access: String,
}

#[derive(Debug, Deserialize)]
struct BootstrapPlan {
    replicates: usize,
    seed: u64,
    confidence_level: f64,
    multiple_comparison_correction: String,
    minimum_independent_clusters: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct PairCase {
    pair_id: String,
    split: String,
    synthetic_only: bool,
    dimension: String,
    expected_relation: String,
    criticality: String,
    source_family_ids: Vec<String>,
    baseline: PairState,
    mutated: PairState,
    acceptance: Acceptance,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct PairState {
    context_id: String,
    tradition_scope: String,
    source_id: String,
    variant_id: String,
    prompt: String,
    rights_status: String,
    access_decision: String,
    evidence_pack: Vec<PairEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct PairEvidence {
    evidence_id: String,
    source_id: String,
    statement: String,
    kind: String,
}

#[derive(Debug, Clone, Deserialize)]
struct SourcePackSnapshot {
    schema_version: String,
    synthetic_only: bool,
    split: String,
    pair_id: String,
    state: String,
    #[serde(flatten)]
    state_data: PairState,
}

#[derive(Debug, Clone, Deserialize)]
struct SourceRecord {
    source_id: String,
    source_family_id: String,
    split: String,
    rights_status: String,
    content_access: String,
    rights_review: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Acceptance {
    must_preserve: Vec<String>,
    must_change: Vec<String>,
    must_not: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Observation {
    schema_version: String,
    result_id: String,
    plan_id: String,
    system_id: String,
    pair_id: String,
    observed_relation: String,
    preservation_satisfied: bool,
    change_satisfied: bool,
    forbidden_behaviors: Vec<String>,
    citation_identity_valid: bool,
    adjudication_status: String,
    notes: String,
}

#[derive(Debug, Serialize)]
struct DimensionMetrics {
    pair_count: usize,
    relation_accuracy: f64,
    preservation_rate: f64,
    change_satisfaction_rate: Option<f64>,
    citation_identity_rate: f64,
    forbidden_behavior_rows: usize,
    paired_intervention_fidelity: f64,
}

#[derive(Debug, Serialize)]
struct SystemMetrics {
    system_id: String,
    baseline_family: String,
    pair_count: usize,
    relation_accuracy: f64,
    preservation_rate: f64,
    change_satisfaction_rate: f64,
    citation_identity_rate: f64,
    forbidden_behavior_rows: usize,
    critical_gate_violations: usize,
    critical_gates_pass: bool,
    paired_intervention_fidelity: f64,
    typed_relation_macro_f1: f64,
    worst_dimension_score: f64,
    primary_metric: String,
    primary_metric_value: f64,
    per_dimension: BTreeMap<String, DimensionMetrics>,
}

#[derive(Debug, Serialize)]
struct PairwiseComparison {
    baseline_system_id: String,
    candidate_system_id: String,
    primary_endpoint: String,
    baseline_value: f64,
    candidate_value: f64,
    absolute_difference: f64,
    confidence_interval_low: f64,
    confidence_interval_high: f64,
    confidence_level: f64,
    source_family_cluster_count: usize,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    minimum_independent_clusters: usize,
    multiple_comparison_correction: String,
    candidate_improvement_supported: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema_version: &'static str,
    tool_version: &'static str,
    plan_id: String,
    phase: String,
    synthetic_only: bool,
    qualification_only: bool,
    primary_endpoint: String,
    candidate_system_id: String,
    candidate_critical_gates_pass: bool,
    report_schema_sha256: String,
    plan_sha256: String,
    pair_manifest_sha256: String,
    result_rows_sha256: String,
    dependency_lock_sha256: String,
    benchmark_case_manifest_sha256: String,
    source_catalog_sha256: String,
    source_pack_sha256: String,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    minimum_independent_clusters: usize,
    confidence_level: f64,
    source_family_cluster_count: usize,
    pair_count: usize,
    system_count: usize,
    systems: Vec<SystemMetrics>,
    comparisons: Vec<PairwiseComparison>,
    nonclaims: Vec<&'static str>,
}

fn usage() -> &'static str {
    "Usage: score-paired-results --plan FILE --pairs JSONL --results JSONL --case-manifest FILE --source-catalog FILE --source-pack FILE --out FILE\n\
     Scores blinded/adjudicated paired-intervention observations against a frozen comparison plan.\n\
     It does not run models, infer labels, or turn synthetic controls into benchmark results."
}

fn parse_args() -> Result<Args, String> {
    let mut raw = env::args().skip(1);
    let mut plan = None;
    let mut pairs = None;
    let mut results = None;
    let mut case_manifest = None;
    let mut source_catalog = None;
    let mut source_pack = None;
    let mut output = None;

    while let Some(arg) = raw.next() {
        let value = |raw: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            raw.next().ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--plan" => plan = Some(PathBuf::from(value(&mut raw, "--plan")?)),
            "--pairs" => pairs = Some(PathBuf::from(value(&mut raw, "--pairs")?)),
            "--results" => results = Some(PathBuf::from(value(&mut raw, "--results")?)),
            "--case-manifest" => {
                case_manifest = Some(PathBuf::from(value(&mut raw, "--case-manifest")?))
            }
            "--source-catalog" => {
                source_catalog = Some(PathBuf::from(value(&mut raw, "--source-catalog")?))
            }
            "--source-pack" => source_pack = Some(PathBuf::from(value(&mut raw, "--source-pack")?)),
            "--out" => output = Some(PathBuf::from(value(&mut raw, "--out")?)),
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(Args {
        plan: plan.ok_or_else(|| format!("missing --plan\n{}", usage()))?,
        pairs: pairs.ok_or_else(|| format!("missing --pairs\n{}", usage()))?,
        results: results.ok_or_else(|| format!("missing --results\n{}", usage()))?,
        case_manifest: case_manifest
            .ok_or_else(|| format!("missing --case-manifest\n{}", usage()))?,
        source_catalog: source_catalog
            .ok_or_else(|| format!("missing --source-catalog\n{}", usage()))?,
        source_pack: source_pack.ok_or_else(|| format!("missing --source-pack\n{}", usage()))?,
        output: output.ok_or_else(|| format!("missing --out\n{}", usage()))?,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn verify_sha256(label: &str, bytes: &[u8], expected: &str) -> Result<String, String> {
    let actual = sha256_hex(bytes);
    if actual != expected {
        return Err(format!(
            "{label} digest mismatch: manifest declares {expected}, actual bytes hash to {actual}"
        ));
    }
    Ok(actual)
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<Vec<T>, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    let mut values = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("{label} contains an empty line at {}", index + 1));
        }
        values.push(
            serde_json::from_str(line).map_err(|error| {
                format!("{label} has invalid JSON at line {}: {error}", index + 1)
            })?,
        );
    }
    if values.is_empty() {
        return Err(format!("{label} contains no records"));
    }
    Ok(values)
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn macro_f1(pairs: &[PairCase], observations: &[&Observation]) -> f64 {
    let labels: BTreeSet<&str> = pairs
        .iter()
        .map(|pair| pair.expected_relation.as_str())
        .collect();
    if labels.is_empty() {
        return 0.0;
    }
    let mut f1_sum = 0.0;
    let mut label_count = 0usize;
    for label in labels {
        let mut true_positive = 0usize;
        let mut false_positive = 0usize;
        let mut false_negative = 0usize;
        for (pair, observation) in pairs.iter().zip(observations.iter()) {
            let expected = pair.expected_relation.as_str();
            let observed = observation.observed_relation.as_str();
            if expected == label.as_str() && observed == label.as_str() {
                true_positive += 1;
            } else {
                if expected != label.as_str() && observed == label.as_str() {
                    false_positive += 1;
                }
                if expected == label && observed != label {
                    false_negative += 1;
                }
            }
        }
        let denominator = 2 * true_positive + false_positive + false_negative;
        if denominator > 0 {
            f1_sum += 2.0 * true_positive as f64 / denominator as f64;
            label_count += 1;
        }
    }
    if label_count == 0 {
        0.0
    } else {
        f1_sum / label_count as f64
    }
}

fn validate_plan_eligibility(plan: &Plan) -> Result<(), String> {
    let systems: HashSet<&str> = plan
        .systems
        .iter()
        .map(|system| system.system_id.as_str())
        .collect();
    let families: HashSet<&str> = plan
        .systems
        .iter()
        .map(|system| system.baseline_family.as_str())
        .collect();
    if plan.systems.len() < 2 || systems.len() != plan.systems.len() {
        return Err("plan must contain at least two systems with unique system_id values".into());
    }
    if families.len() < 2 {
        return Err("plan must include at least two distinct baseline families".into());
    }
    if !systems.contains(plan.candidate_system_id.as_str()) {
        return Err(format!(
            "candidate_system_id {:?} must be declared in systems before evaluation",
            plan.candidate_system_id
        ));
    }

    let digests = [
        plan.benchmark.case_manifest_sha256.as_str(),
        plan.benchmark.source_catalog_sha256.as_str(),
        plan.benchmark.source_pack_sha256.as_str(),
    ];
    if digests.iter().any(|digest| {
        digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    }) {
        return Err("plan must use valid lowercase SHA-256 digests".into());
    }

    if plan.phase == "synthetic_control" {
        if !plan.synthetic_only
            || plan.benchmark.split != "synthetic"
            || plan.preregistration.held_out_access != "not_applicable_synthetic"
        {
            return Err("synthetic plan must be marked synthetic_only, use split=synthetic, and declare held_out_access=not_applicable_synthetic".into());
        }
    } else if plan.phase == "preregistered_held_out" {
        if plan.synthetic_only
            || !matches!(plan.benchmark.split.as_str(), "test" | "holdout")
            || !plan.benchmark.source_family_disjoint
            || plan.benchmark.rights_review_status != "approved"
            || !matches!(
                plan.preregistration.held_out_access.as_str(),
                "sealed_until_freeze" | "independent_evaluator"
            )
        {
            return Err("held-out scoring requires real test/holdout data, source-family disjointness, approved rights, and sealed/independent evaluation access".into());
        }
    } else if plan.phase == "development_pilot" {
        if plan.synthetic_only
            || !matches!(plan.benchmark.split.as_str(), "development" | "validation")
        {
            return Err(
                "development_pilot requires non-synthetic development/validation data".into(),
            );
        }
    } else {
        return Err(format!("unsupported comparison phase {:?}", plan.phase));
    }

    if !plan.synthetic_only && plan.benchmark.rights_review_status == "denied" {
        return Err("cannot score source material whose rights review is denied".into());
    }

    if plan.bootstrap.replicates < 1_000 || plan.bootstrap.replicates > 100_000 {
        return Err("bootstrap replicates must be between 1000 and 100000".into());
    }
    if plan.bootstrap.minimum_independent_clusters < 30 {
        return Err("minimum_independent_clusters must be at least 30".into());
    }
    if !(plan.bootstrap.confidence_level > 0.5 && plan.bootstrap.confidence_level < 1.0) {
        return Err("bootstrap confidence_level must be in (0.5, 1.0)".into());
    }
    if !matches!(
        plan.bootstrap.multiple_comparison_correction.as_str(),
        "none" | "bonferroni"
    ) {
        return Err("multiple_comparison_correction must be none or bonferroni".into());
    }
    if plan.phase == "preregistered_held_out"
        && plan.systems.len() > 2
        && plan.bootstrap.multiple_comparison_correction == "none"
    {
        return Err(
            "held-out comparisons against multiple baselines require familywise correction".into(),
        );
    }

    if !matches!(
        plan.primary_endpoint.as_str(),
        "paired_intervention_fidelity"
            | "citation_identity_rate"
            | "typed_relation_macro_f1"
            | "worst_dimension_score"
    ) {
        return Err(format!(
            "primary endpoint {:?} is not computed by this paired-results scorer",
            plan.primary_endpoint
        ));
    }
    Ok(())
}

fn validate_source_pack_matches_pairs(
    pairs: &[PairCase],
    source_pack: &[SourcePackSnapshot],
    synthetic_only: bool,
) -> Result<(), String> {
    let expected_count = pairs.len() * 2;
    if source_pack.len() != expected_count {
        return Err(format!(
            "source pack contains {} snapshots, expected exactly {expected_count}",
            source_pack.len()
        ));
    }

    let mut snapshots: HashMap<(String, String), &SourcePackSnapshot> = HashMap::new();
    for snapshot in source_pack {
        if snapshot.schema_version != "knowledge-source-pack-v0" {
            return Err(format!(
                "source-pack snapshot for pair {:?} has unsupported schema_version {:?}",
                snapshot.pair_id, snapshot.schema_version
            ));
        }
        if snapshot.synthetic_only != synthetic_only {
            return Err(format!(
                "source-pack snapshot for pair {:?} disagrees with plan synthetic_only={synthetic_only}",
                snapshot.pair_id
            ));
        }
        if !matches!(snapshot.state.as_str(), "baseline" | "mutated") {
            return Err(format!(
                "source-pack snapshot for pair {:?} has invalid state {:?}",
                snapshot.pair_id, snapshot.state
            ));
        }
        let key = (snapshot.pair_id.clone(), snapshot.state.clone());
        if snapshots.insert(key, snapshot).is_some() {
            return Err(format!(
                "duplicate source-pack snapshot for pair {:?} state {:?}",
                snapshot.pair_id, snapshot.state
            ));
        }
    }

    let mut expected_keys = HashSet::new();
    for pair in pairs {
        for (state_name, expected_state) in
            [("baseline", &pair.baseline), ("mutated", &pair.mutated)]
        {
            let key = (pair.pair_id.clone(), state_name.to_string());
            expected_keys.insert(key.clone());
            let snapshot = snapshots.get(&key).ok_or_else(|| {
                format!(
                    "source pack is missing pair {:?} state {state_name}",
                    pair.pair_id
                )
            })?;
            if snapshot.split != pair.split {
                return Err(format!(
                    "source-pack snapshot split does not match pair {:?}",
                    pair.pair_id
                ));
            }
            if snapshot.synthetic_only != pair.synthetic_only {
                return Err(format!(
                    "source-pack snapshot synthetic flag does not match pair {:?}",
                    pair.pair_id
                ));
            }
            if &snapshot.state_data != expected_state {
                return Err(format!(
                    "source-pack snapshot does not match pair {:?} state {state_name}",
                    pair.pair_id
                ));
            }
        }
    }

    if snapshots.keys().any(|key| !expected_keys.contains(key)) {
        return Err("source pack contains an unreferenced pair/state snapshot".into());
    }
    Ok(())
}

fn validate_real_pair_state(
    pair_id: &str,
    split: &str,
    state_name: &str,
    state: &PairState,
    source_catalog: &HashMap<&str, &SourceRecord>,
) -> Result<(), String> {
    if !matches!(
        state.access_decision.as_str(),
        "allow" | "deny" | "require_permission" | "metadata_only"
    ) {
        return Err(format!(
            "pair {pair_id:?} {state_name} has unsupported access_decision {:?}",
            state.access_decision
        ));
    }
    if state.access_decision != "allow" && !state.evidence_pack.is_empty() {
        return Err(format!(
            "pair {pair_id:?} {state_name} has a non-allow access decision but still exposes an evidence pack"
        ));
    }

    let mut referenced_ids = HashSet::new();
    referenced_ids.insert(state.source_id.as_str());
    for evidence in &state.evidence_pack {
        if matches!(
            evidence.kind.as_str(),
            "synthetic_control" | "metadata_only"
        ) {
            return Err(format!(
                "pair {pair_id:?} {state_name} uses inadmissible evidence kind {:?}",
                evidence.kind
            ));
        }
        referenced_ids.insert(evidence.source_id.as_str());
    }

    for source_id in referenced_ids {
        let Some(source) = source_catalog.get(source_id) else {
            return Err(format!(
                "pair {pair_id:?} {state_name} references source {source_id:?} absent from catalog"
            ));
        };
        if source.split != split {
            return Err(format!(
                "pair {pair_id:?} source {source_id:?} is in split {:?}, expected {:?}",
                source.split, split
            ));
        }
        if source.rights_review != "approved_limited_use"
            || matches!(source.rights_status.as_str(), "unknown" | "restricted")
            || !matches!(
                source.content_access.as_str(),
                "verified_open_content" | "permissioned_full_text"
            )
        {
            return Err(format!(
                "pair {pair_id:?} source {source_id:?} lacks approved content access"
            ));
        }
    }

    let primary_source = source_catalog
        .get(state.source_id.as_str())
        .expect("primary source ID was checked above");
    if state.rights_status != primary_source.rights_status {
        return Err(format!(
            "pair {pair_id:?} {state_name} rights_status differs from catalog for source {:?}",
            state.source_id
        ));
    }

    Ok(())
}

fn source_family_clusters(pairs: &[PairCase]) -> Result<Vec<Vec<usize>>, String> {
    if pairs.is_empty() {
        return Err("cannot bootstrap an empty pair manifest".into());
    }

    let mut family_to_pairs: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, pair) in pairs.iter().enumerate() {
        if pair.source_family_ids.is_empty() {
            return Err(format!("pair {:?} has no source_family_ids", pair.pair_id));
        }
        let unique: HashSet<&str> = pair.source_family_ids.iter().map(String::as_str).collect();
        if unique.len() != pair.source_family_ids.len() {
            return Err(format!("pair {:?} repeats source_family_ids", pair.pair_id));
        }
        for family_id in &pair.source_family_ids {
            family_to_pairs
                .entry(family_id.as_str())
                .or_default()
                .push(index);
        }
    }

    // Any pair connecting two families joins those families into one dependency
    // cluster. Bootstrap resampling these connected components avoids treating
    // questions that share source families as independent observations.
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); pairs.len()];
    for indices in family_to_pairs.values() {
        for &a in indices {
            for &b in indices {
                if a != b {
                    adjacency[a].insert(b);
                }
            }
        }
    }

    let mut visited = vec![false; pairs.len()];
    let mut clusters = Vec::new();
    for start in 0..pairs.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut cluster = Vec::new();
        visited[start] = true;
        while let Some(index) = stack.pop() {
            cluster.push(index);
            for &neighbor in &adjacency[index] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        cluster.sort_unstable();
        clusters.push(cluster);
    }
    Ok(clusters)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (((self.next_u64() as u128) * (upper_exclusive as u128)) >> 64) as usize
    }
}

fn fidelity_for(pair: &PairCase, observation: &Observation) -> bool {
    pair.expected_relation == observation.observed_relation
        && observation.preservation_satisfied
        && (pair.acceptance.must_change.is_empty() || observation.change_satisfied)
        && observation.forbidden_behaviors.is_empty()
        && observation.citation_identity_valid
}

fn metric_for_sample(
    endpoint: &str,
    sample_indices: &[usize],
    pairs: &[PairCase],
    observations_by_system: &HashMap<&str, HashMap<&str, &Observation>>,
    system_id: &str,
    fixed_relation_labels: &[String],
    fixed_dimensions: &[String],
) -> Result<f64, String> {
    if sample_indices.is_empty() {
        return Err("bootstrap replicate contains no pairs".into());
    }
    let system_rows = observations_by_system
        .get(system_id)
        .ok_or_else(|| format!("bootstrap has no observations for system {system_id:?}"))?;
    let observation_for = |index: usize| -> Result<&Observation, String> {
        let pair = pairs
            .get(index)
            .ok_or_else(|| format!("bootstrap pair index {index} is out of range"))?;
        system_rows
            .get(pair.pair_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "system {system_id:?} has no observation for pair {:?}",
                    pair.pair_id
                )
            })
    };

    match endpoint {
        "paired_intervention_fidelity" => {
            let mut successes = 0usize;
            for &index in sample_indices {
                if fidelity_for(&pairs[index], observation_for(index)?) {
                    successes += 1;
                }
            }
            Ok(ratio(successes, sample_indices.len()))
        }
        "citation_identity_rate" => {
            let mut successes = 0usize;
            for &index in sample_indices {
                successes += usize::from(observation_for(index)?.citation_identity_valid);
            }
            Ok(ratio(successes, sample_indices.len()))
        }
        "typed_relation_macro_f1" => {
            if fixed_relation_labels.is_empty() {
                return Ok(0.0);
            }
            let mut f1_sum = 0.0;
            for label in fixed_relation_labels {
                let mut true_positive = 0usize;
                let mut false_positive = 0usize;
                let mut false_negative = 0usize;
                for &index in sample_indices {
                    let expected = pairs[index].expected_relation.as_str();
                    let observed = observation_for(index)?.observed_relation.as_str();
                    if expected == label.as_str() && observed == label.as_str() {
                        true_positive += 1;
                    } else {
                        if expected != label.as_str() && observed == label.as_str() {
                            false_positive += 1;
                        }
                        if expected == label && observed != label {
                            false_negative += 1;
                        }
                    }
                }
                let denominator = 2 * true_positive + false_positive + false_negative;
                let f1 = if denominator == 0 {
                    // A preregistered relation absent from a resample contributes
                    // zero rather than disappearing from the metric denominator.
                    0.0
                } else {
                    2.0 * true_positive as f64 / denominator as f64
                };
                f1_sum += f1;
            }
            Ok(f1_sum / fixed_relation_labels.len() as f64)
        }
        "worst_dimension_score" => {
            if fixed_dimensions.is_empty() {
                return Ok(0.0);
            }
            let mut totals: HashMap<&str, (usize, usize)> = fixed_dimensions
                .iter()
                .map(|dimension| (dimension.as_str(), (0usize, 0usize)))
                .collect();
            for &index in sample_indices {
                let pair = &pairs[index];
                let row = observation_for(index)?;
                if let Some((count, success)) = totals.get_mut(pair.dimension.as_str()) {
                    *count += 1;
                    *success += usize::from(fidelity_for(pair, row));
                }
            }
            // A dimension omitted by a cluster bootstrap replicate is assigned
            // zero conservatively rather than silently dropped from worst-group
            // evaluation.
            Ok(totals
                .values()
                .map(|(count, success)| {
                    if *count == 0 {
                        0.0
                    } else {
                        ratio(*success, *count)
                    }
                })
                .fold(1.0_f64, f64::min))
        }
        _ => Err(format!("unsupported bootstrap endpoint {endpoint:?}")),
    }
}

fn percentile(sorted_values: &[f64], probability: f64) -> f64 {
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let position = probability.clamp(0.0, 1.0) * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let fraction = position - lower as f64;
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    }
}

fn compute_pairwise_comparisons(
    plan: &Plan,
    pairs: &[PairCase],
    observations: &[Observation],
    clusters: &[Vec<usize>],
) -> Result<Vec<PairwiseComparison>, String> {
    if clusters.is_empty() {
        return Err("cannot compare systems without source-family clusters".into());
    }
    if plan.phase == "preregistered_held_out"
        && clusters.len() < plan.bootstrap.minimum_independent_clusters
    {
        return Err(format!(
            "held-out inference requires at least {} independent source-family clusters; found {}",
            plan.bootstrap.minimum_independent_clusters,
            clusters.len()
        ));
    }
    let mut observations_by_system: HashMap<&str, HashMap<&str, &Observation>> = HashMap::new();
    for observation in observations {
        let rows = observations_by_system
            .entry(observation.system_id.as_str())
            .or_default();
        if rows
            .insert(observation.pair_id.as_str(), observation)
            .is_some()
        {
            return Err(format!(
                "duplicate observation for system {:?} pair {:?} during bootstrap setup",
                observation.system_id, observation.pair_id
            ));
        }
    }

    let relation_labels: Vec<String> = pairs
        .iter()
        .map(|pair| pair.expected_relation.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let dimensions: Vec<String> = pairs
        .iter()
        .map(|pair| pair.dimension.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let all_indices: Vec<usize> = (0..pairs.len()).collect();
    let candidate_id = plan.candidate_system_id.as_str();
    let candidate_critical_gates_pass = pairs
        .iter()
        .filter(|pair| pair.criticality == "release_blocker")
        .all(|pair| {
            observations_by_system
                .get(candidate_id)
                .and_then(|rows| rows.get(pair.pair_id.as_str()))
                .is_some_and(|observation| fidelity_for(pair, observation))
        });
    let candidate_value = metric_for_sample(
        &plan.primary_endpoint,
        &all_indices,
        pairs,
        &observations_by_system,
        candidate_id,
        &relation_labels,
        &dimensions,
    )?;

    // Generate a single set of cluster resamples and reuse it for each baseline,
    // ensuring that each reported candidate-versus-baseline interval is paired.
    let mut rng = SplitMix64::new(plan.bootstrap.seed);
    let mut bootstrap_samples: Vec<Vec<usize>> = Vec::with_capacity(plan.bootstrap.replicates);
    for _ in 0..plan.bootstrap.replicates {
        let mut sample = Vec::new();
        for _ in 0..clusters.len() {
            let selected_cluster = rng.index(clusters.len());
            sample.extend_from_slice(&clusters[selected_cluster]);
        }
        bootstrap_samples.push(sample);
    }

    let mut comparisons = Vec::new();
    for system in &plan.systems {
        if system.system_id == candidate_id {
            continue;
        }
        let baseline_id = system.system_id.as_str();
        let baseline_value = metric_for_sample(
            &plan.primary_endpoint,
            &all_indices,
            pairs,
            &observations_by_system,
            baseline_id,
            &relation_labels,
            &dimensions,
        )?;
        let mut differences = Vec::with_capacity(bootstrap_samples.len());
        for sample in &bootstrap_samples {
            let candidate_bootstrap = metric_for_sample(
                &plan.primary_endpoint,
                sample,
                pairs,
                &observations_by_system,
                candidate_id,
                &relation_labels,
                &dimensions,
            )?;
            let baseline_bootstrap = metric_for_sample(
                &plan.primary_endpoint,
                sample,
                pairs,
                &observations_by_system,
                baseline_id,
                &relation_labels,
                &dimensions,
            )?;
            differences.push(candidate_bootstrap - baseline_bootstrap);
        }
        differences.sort_by(f64::total_cmp);
        let alpha = 1.0 - plan.bootstrap.confidence_level;
        let comparison_count = (plan.systems.len() - 1).max(1) as f64;
        let adjusted_alpha = if plan.bootstrap.multiple_comparison_correction == "bonferroni" {
            alpha / comparison_count
        } else {
            alpha
        };
        let confidence_interval_low = percentile(&differences, adjusted_alpha / 2.0);
        let confidence_interval_high = percentile(&differences, 1.0 - adjusted_alpha / 2.0);
        comparisons.push(PairwiseComparison {
            baseline_system_id: system.system_id.clone(),
            candidate_system_id: plan.candidate_system_id.clone(),
            primary_endpoint: plan.primary_endpoint.clone(),
            baseline_value,
            candidate_value,
            absolute_difference: candidate_value - baseline_value,
            confidence_interval_low,
            confidence_interval_high,
            confidence_level: plan.bootstrap.confidence_level,
            source_family_cluster_count: clusters.len(),
            bootstrap_replicates: plan.bootstrap.replicates,
            bootstrap_seed: plan.bootstrap.seed,
            multiple_comparison_correction: plan.bootstrap.multiple_comparison_correction.clone(),
            candidate_improvement_supported: !plan.synthetic_only
                && plan.phase == "preregistered_held_out"
                && clusters.len() >= plan.bootstrap.minimum_independent_clusters
                && candidate_critical_gates_pass
                && confidence_interval_low > 0.0,
        });
    }
    Ok(comparisons)
}

fn compute_system_metrics(
    system: &System,
    pair_cases: &[PairCase],
    observations: &[&Observation],
    primary_endpoint: &str,
) -> SystemMetrics {
    let pair_count = pair_cases.len();
    let mut relation_correct = 0usize;
    let mut preservation_correct = 0usize;
    let mut change_denominator = 0usize;
    let mut change_correct = 0usize;
    let mut citation_correct = 0usize;
    let mut forbidden_rows = 0usize;
    let mut critical_violations = 0usize;
    let mut full_fidelity = 0usize;
    let mut per_dimension_counts: BTreeMap<String, Vec<(bool, bool, bool, bool, bool, bool)>> =
        BTreeMap::new();

    for (pair, observation) in pair_cases.iter().zip(observations.iter()) {
        let relation_ok = pair.expected_relation == observation.observed_relation;
        let preservation_ok = observation.preservation_satisfied;
        let change_required = !pair.acceptance.must_change.is_empty();
        let change_ok = !change_required || observation.change_satisfied;
        let forbidden_ok = observation.forbidden_behaviors.is_empty();
        let citation_ok = observation.citation_identity_valid;
        let fidelity_ok =
            relation_ok && preservation_ok && change_ok && forbidden_ok && citation_ok;

        relation_correct += usize::from(relation_ok);
        preservation_correct += usize::from(preservation_ok);
        if change_required {
            change_denominator += 1;
            change_correct += usize::from(observation.change_satisfied);
        }
        citation_correct += usize::from(citation_ok);
        forbidden_rows += usize::from(!forbidden_ok);
        full_fidelity += usize::from(fidelity_ok);
        if pair.criticality == "release_blocker" && !fidelity_ok {
            critical_violations += 1;
        }

        per_dimension_counts
            .entry(pair.dimension.clone())
            .or_default()
            .push((
                relation_ok,
                preservation_ok,
                change_ok,
                citation_ok,
                forbidden_ok,
                fidelity_ok,
            ));
    }

    let mut per_dimension = BTreeMap::new();
    for (dimension, items) in per_dimension_counts {
        let n = items.len();
        let relation = items.iter().filter(|i| i.0).count();
        let preserve = items.iter().filter(|i| i.1).count();
        let change = items.iter().filter(|i| i.2).count();
        let citation = items.iter().filter(|i| i.3).count();
        let forbidden_behavior_rows = items.iter().filter(|i| !i.4).count();
        let fidelity = items.iter().filter(|i| i.5).count();
        let required_change_count = pair_cases
            .iter()
            .filter(|pair| pair.dimension == dimension && !pair.acceptance.must_change.is_empty())
            .count();
        let dimension_change_correct = pair_cases
            .iter()
            .zip(observations.iter())
            .filter(|(pair, _)| {
                pair.dimension == dimension && !pair.acceptance.must_change.is_empty()
            })
            .filter(|(_, observation)| observation.change_satisfied)
            .count();

        per_dimension.insert(
            dimension,
            DimensionMetrics {
                pair_count: n,
                relation_accuracy: ratio(relation, n),
                preservation_rate: ratio(preserve, n),
                change_satisfaction_rate: if required_change_count == 0 {
                    None
                } else {
                    Some(ratio(dimension_change_correct, required_change_count))
                },
                citation_identity_rate: ratio(citation, n),
                forbidden_behavior_rows,
                paired_intervention_fidelity: ratio(fidelity, n),
            },
        );
    }

    let relation_f1 = macro_f1(pair_cases, observations);
    let worst_dimension_score = per_dimension
        .values()
        .map(|metrics| metrics.paired_intervention_fidelity)
        .fold(1.0_f64, f64::min);
    let paired_fidelity = ratio(full_fidelity, pair_count);
    let citation_identity_rate = ratio(citation_correct, pair_count);
    let primary_metric_value = match primary_endpoint {
        "paired_intervention_fidelity" => paired_fidelity,
        "citation_identity_rate" => citation_identity_rate,
        "typed_relation_macro_f1" => relation_f1,
        "worst_dimension_score" => worst_dimension_score,
        _ => unreachable!("endpoint validated before scoring"),
    };

    SystemMetrics {
        system_id: system.system_id.clone(),
        baseline_family: system.baseline_family.clone(),
        pair_count,
        relation_accuracy: ratio(relation_correct, pair_count),
        preservation_rate: ratio(preservation_correct, pair_count),
        change_satisfaction_rate: ratio(change_correct, change_denominator),
        citation_identity_rate,
        forbidden_behavior_rows: forbidden_rows,
        critical_gate_violations: critical_violations,
        critical_gates_pass: critical_violations == 0,
        paired_intervention_fidelity: paired_fidelity,
        typed_relation_macro_f1: relation_f1,
        worst_dimension_score,
        primary_metric: primary_endpoint.to_string(),
        primary_metric_value,
        per_dimension,
    }
}

fn score(
    plan: &Plan,
    pairs: Vec<PairCase>,
    observations: Vec<Observation>,
    source_records: &[SourceRecord],
) -> Result<Vec<SystemMetrics>, String> {
    validate_plan_eligibility(plan)?;

    let mut source_catalog: HashMap<&str, &SourceRecord> = HashMap::new();
    let mut source_family_splits: HashMap<&str, &str> = HashMap::new();
    for source in source_records {
        if source_catalog
            .insert(source.source_id.as_str(), source)
            .is_some()
        {
            return Err(format!(
                "duplicate source catalog ID {:?}",
                source.source_id
            ));
        }
        if let Some(existing_split) =
            source_family_splits.insert(source.source_family_id.as_str(), source.split.as_str())
        {
            if existing_split != source.split.as_str() {
                return Err(format!(
                    "source family {:?} spans multiple splits in the supplied catalog",
                    source.source_family_id
                ));
            }
        }
    }

    let mut pair_by_id = HashMap::new();
    for pair in &pairs {
        if pair.synthetic_only != plan.synthetic_only || pair.split != plan.benchmark.split {
            return Err(format!(
                "pair {:?} split/synthetic status ({:?}, {}) does not match plan ({:?}, {})",
                pair.pair_id,
                pair.split,
                pair.synthetic_only,
                plan.benchmark.split,
                plan.synthetic_only
            ));
        }
        if pair.source_family_ids.is_empty() {
            return Err(format!("pair {:?} has no source_family_ids", pair.pair_id));
        }
        let declared_families: HashSet<&str> =
            pair.source_family_ids.iter().map(String::as_str).collect();
        if declared_families.len() != pair.source_family_ids.len() {
            return Err(format!("pair {:?} repeats source_family_ids", pair.pair_id));
        }
        if pair.synthetic_only {
            for (state_name, state) in [("baseline", &pair.baseline), ("mutated", &pair.mutated)] {
                if state
                    .evidence_pack
                    .iter()
                    .any(|evidence| evidence.kind != "synthetic_control")
                {
                    return Err(format!(
                        "synthetic pair {:?} {state_name} contains non-synthetic evidence",
                        pair.pair_id
                    ));
                }
            }
        } else {
            validate_real_pair_state(
                &pair.pair_id,
                &pair.split,
                "baseline",
                &pair.baseline,
                &source_catalog,
            )?;
            validate_real_pair_state(
                &pair.pair_id,
                &pair.split,
                "mutated",
                &pair.mutated,
                &source_catalog,
            )?;
            let mut observed_families = HashSet::<String>::new();
            for state in [&pair.baseline, &pair.mutated] {
                let source_ids = std::iter::once(state.source_id.as_str()).chain(
                    state
                        .evidence_pack
                        .iter()
                        .map(|evidence| evidence.source_id.as_str()),
                );
                for source_id in source_ids {
                    let source = source_catalog.get(source_id).ok_or_else(|| {
                        format!(
                            "pair {:?} references source {:?} absent from catalog",
                            pair.pair_id, source_id
                        )
                    })?;
                    observed_families.insert(source.source_family_id.clone());
                }
            }
            let observed: HashSet<&str> = observed_families.iter().map(String::as_str).collect();
            if observed != declared_families {
                return Err(format!(
                    "pair {:?} declared source_family_ids do not match catalog references",
                    pair.pair_id
                ));
            }
        }
        if pair.acceptance.must_preserve.is_empty() {
            return Err(format!(
                "pair {:?} must declare at least one must_preserve criterion",
                pair.pair_id
            ));
        }
        if pair.acceptance.must_not.is_empty() {
            return Err(format!(
                "pair {:?} must declare at least one must_not criterion",
                pair.pair_id
            ));
        }
        if pair_by_id.insert(pair.pair_id.as_str(), pair).is_some() {
            return Err(format!("duplicate pair_id {:?}", pair.pair_id));
        }
    }
    if pair_by_id.is_empty() {
        return Err("pair manifest contains no pairs".into());
    }

    let plan_systems: HashMap<&str, &System> = plan
        .systems
        .iter()
        .map(|system| (system.system_id.as_str(), system))
        .collect();
    let mut result_ids = HashSet::new();
    let mut seen_pairs: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut grouped: HashMap<&str, Vec<&Observation>> = HashMap::new();

    for observation in &observations {
        if observation.notes.trim().is_empty() {
            return Err(format!(
                "result {:?} requires non-empty notes",
                observation.result_id
            ));
        }
        if observation.schema_version != "paired-intervention-result-v0" {
            return Err(format!(
                "result {:?} has unsupported schema_version {:?}",
                observation.result_id, observation.schema_version
            ));
        }
        if observation.plan_id != plan.plan_id {
            return Err(format!(
                "result {:?} belongs to plan {:?}, expected {:?}",
                observation.result_id, observation.plan_id, plan.plan_id
            ));
        }
        if !result_ids.insert(observation.result_id.as_str()) {
            return Err(format!("duplicate result_id {:?}", observation.result_id));
        }
        if !plan_systems.contains_key(observation.system_id.as_str()) {
            return Err(format!(
                "result {:?} references unknown system_id {:?}",
                observation.result_id, observation.system_id
            ));
        }
        if !pair_by_id.contains_key(observation.pair_id.as_str()) {
            return Err(format!(
                "result {:?} references unknown pair_id {:?}",
                observation.result_id, observation.pair_id
            ));
        }
        if !seen_pairs
            .entry(observation.system_id.as_str())
            .or_default()
            .insert(observation.pair_id.as_str())
        {
            return Err(format!(
                "system {:?} has duplicate result for pair {:?}",
                observation.system_id, observation.pair_id
            ));
        }
        if plan.synthetic_only {
            if observation.adjudication_status != "synthetic_only" {
                return Err(format!(
                    "synthetic plan result {:?} must be marked synthetic_only",
                    observation.result_id
                ));
            }
        } else if plan.phase == "preregistered_held_out"
            && observation.adjudication_status != "blind_human_adjudicated"
        {
            return Err(format!(
                "held-out result {:?} requires blind_human_adjudicated status",
                observation.result_id
            ));
        } else if !matches!(
            observation.adjudication_status.as_str(),
            "independent_review" | "blind_human_adjudicated"
        ) {
            return Err(format!(
                "non-synthetic result {:?} requires reviewed adjudication status",
                observation.result_id
            ));
        }

        grouped
            .entry(observation.system_id.as_str())
            .or_default()
            .push(observation);
    }

    let mut metrics = Vec::new();
    for system in &plan.systems {
        let expected_pairs: HashSet<&str> = pair_by_id.keys().copied().collect();
        let actual_pairs = seen_pairs.get(system.system_id.as_str());
        if actual_pairs.is_none_or(|actual| actual != &expected_pairs) {
            let observed_count = actual_pairs.map_or(0, |actual| actual.len());
            return Err(format!(
                "system {:?} has incomplete pair coverage: {observed_count}/{}",
                system.system_id,
                expected_pairs.len()
            ));
        }

        let mut ordered_observations: Vec<&Observation> = grouped
            .get(system.system_id.as_str())
            .into_iter()
            .flatten()
            .copied()
            .collect();
        ordered_observations.sort_by_key(|observation| {
            pair_by_id
                .get(observation.pair_id.as_str())
                .map(|pair| pair.pair_id.as_str())
                .unwrap_or("")
        });
        let ordered_pairs: Vec<PairCase> = ordered_observations
            .iter()
            .map(|observation| {
                (**pair_by_id
                    .get(observation.pair_id.as_str())
                    .expect("pair existence was checked"))
                .clone()
            })
            .collect();
        if ordered_pairs.len() != pairs.len() {
            return Err(format!(
                "internal pair alignment failure for system {:?}",
                system.system_id
            ));
        }
        // The zipped vectors are aligned by pair ID above. Output order is deterministic.
        metrics.push(compute_system_metrics(
            system,
            &ordered_pairs,
            &ordered_observations,
            &plan.primary_endpoint,
        ));
    }
    Ok(metrics)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let plan_bytes = fs::read(&args.plan)
        .map_err(|error| format!("cannot read plan {}: {error}", args.plan.display()))?;
    let pair_bytes = fs::read(&args.pairs)
        .map_err(|error| format!("cannot read pairs {}: {error}", args.pairs.display()))?;
    let result_bytes = fs::read(&args.results)
        .map_err(|error| format!("cannot read results {}: {error}", args.results.display()))?;
    let case_manifest_bytes = fs::read(&args.case_manifest).map_err(|error| {
        format!(
            "cannot read case manifest {}: {error}",
            args.case_manifest.display()
        )
    })?;
    let source_catalog_bytes = fs::read(&args.source_catalog).map_err(|error| {
        format!(
            "cannot read source catalog {}: {error}",
            args.source_catalog.display()
        )
    })?;
    let source_pack_bytes = fs::read(&args.source_pack).map_err(|error| {
        format!(
            "cannot read source pack {}: {error}",
            args.source_pack.display()
        )
    })?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)
        .map_err(|error| format!("plan JSON is invalid: {error}"))?;
    if pair_bytes != case_manifest_bytes {
        return Err(
            "paired scorer requires --case-manifest bytes to exactly match --pairs bytes".into(),
        );
    }
    let actual_case_manifest_sha256 = verify_sha256(
        "case manifest",
        &case_manifest_bytes,
        &plan.benchmark.case_manifest_sha256,
    )?;
    let actual_source_catalog_sha256 = verify_sha256(
        "source catalog",
        &source_catalog_bytes,
        &plan.benchmark.source_catalog_sha256,
    )?;
    let actual_source_pack_sha256 = verify_sha256(
        "source pack",
        &source_pack_bytes,
        &plan.benchmark.source_pack_sha256,
    )?;
    let pairs = parse_jsonl::<PairCase>(&pair_bytes, "pair manifest")?;
    let pair_count = pairs.len();
    let results = parse_jsonl::<Observation>(&result_bytes, "paired-intervention results")?;
    let source_records = parse_jsonl::<SourceRecord>(&source_catalog_bytes, "source catalog")?;
    let source_pack_records = parse_jsonl::<SourcePackSnapshot>(&source_pack_bytes, "source pack")?;
    validate_source_pack_matches_pairs(&pairs, &source_pack_records, plan.synthetic_only)?;
    // Validate source rights/splits/family identities before performing any statistical scoring.
    let system_metrics = score(&plan, pairs.clone(), results.clone(), &source_records)?;
    let clusters = source_family_clusters(&pairs)?;
    let comparisons = compute_pairwise_comparisons(&plan, &pairs, &results, &clusters)?;
    let source_family_cluster_count = clusters.len();

    let manifest_lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let lock_bytes = fs::read(&manifest_lock_path).map_err(|error| {
        format!(
            "cannot read dependency lockfile {}: {error}",
            manifest_lock_path.display()
        )
    })?;

    let report = Report {
        report_schema_version: "paired-intervention-score-report-v0",
        tool_version: TOOL_VERSION,
        plan_id: plan.plan_id.clone(),
        phase: plan.phase.clone(),
        synthetic_only: plan.synthetic_only,
        qualification_only: plan.synthetic_only,
        primary_endpoint: plan.primary_endpoint.clone(),
        candidate_system_id: plan.candidate_system_id.clone(),
        candidate_critical_gates_pass: system_metrics
            .iter()
            .find(|metrics| metrics.system_id == plan.candidate_system_id)
            .is_some_and(|metrics| metrics.critical_gates_pass),
        report_schema_sha256: sha256_hex(SCORE_REPORT_SCHEMA.as_bytes()),
        plan_sha256: sha256_hex(&plan_bytes),
        pair_manifest_sha256: sha256_hex(&pair_bytes),
        result_rows_sha256: sha256_hex(&result_bytes),
        dependency_lock_sha256: sha256_hex(&lock_bytes),
        benchmark_case_manifest_sha256: actual_case_manifest_sha256,
        source_catalog_sha256: actual_source_catalog_sha256,
        source_pack_sha256: actual_source_pack_sha256,
        bootstrap_replicates: plan.bootstrap.replicates,
        bootstrap_seed: plan.bootstrap.seed,
        minimum_independent_clusters: plan.bootstrap.minimum_independent_clusters,
        confidence_level: plan.bootstrap.confidence_level,
        source_family_cluster_count,
        pair_count,
        system_count: system_metrics.len(),
        systems: system_metrics,
        comparisons,
        nonclaims: vec![
            "This tool scores adjudicated labels; it does not execute models or verify the labels against source material.",
            "Synthetic-control metrics are qualification fixtures, not evidence of model performance or SOTA.",
            "A high average cannot offset a release-blocker provenance, citation, rights, or lineage failure.",
            "Results generalize only to the documented source-family split and annotation process.",
        ],
    };
    let report_value = serde_json::to_value(&report)
        .map_err(|error| format!("cannot serialize score report: {error}"))?;
    let report_schema: Value = serde_json::from_str(SCORE_REPORT_SCHEMA)
        .map_err(|error| format!("bundled score-report schema is invalid JSON: {error}"))?;
    let meta_validator = jsonschema::meta::validator_for(&report_schema)
        .map_err(|error| format!("cannot compile score-report meta-schema validator: {error}"))?;
    if !meta_validator.evaluate(&report_schema).flag().valid {
        return Err("bundled score-report schema does not validate against its meta-schema".into());
    }
    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&report_schema)
        .map_err(|error| format!("cannot compile score-report schema: {error}"))?;
    let validation_errors: Vec<String> = validator
        .iter_errors(&report_value)
        .map(|error| error.to_string())
        .collect();
    if !validation_errors.is_empty() {
        return Err(format!(
            "generated score report violates its schema: {}",
            validation_errors.join("; ")
        ));
    }
    let bytes = serde_json::to_vec_pretty(&report_value)
        .map_err(|error| format!("cannot serialize score report: {error}"))?;
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create output directory {}: {error}",
                parent.display()
            )
        })?;
    }
    fs::write(&args.output, &bytes)
        .map_err(|error| format!("cannot write report {}: {error}", args.output.display()))?;
    println!(
        "Scored {} paired cases for {} systems; qualification_only={}; candidate_critical_gates_pass={}; report={}",
        report.pair_count,
        report.system_count,
        report.qualification_only,
        report.candidate_critical_gates_pass,
        args.output.display()
    );
    if plan.phase == "preregistered_held_out" && !report.candidate_critical_gates_pass {
        return Err(format!(
            "candidate {:?} failed a non-compensatory critical gate; score report was preserved at {}",
            report.candidate_system_id,
            args.output.display()
        ));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("score-paired-results: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_plan() -> Plan {
        serde_json::from_value(json!({
            "plan_id":"test-plan",
            "phase":"synthetic_control",
            "synthetic_only":true,
            "primary_endpoint":"paired_intervention_fidelity",
            "candidate_system_id":"s2",
            "bootstrap":{"replicates":2000,"seed":7,"confidence_level":0.95,"multiple_comparison_correction":"bonferroni","minimum_independent_clusters":30},
            "benchmark":{
                "split":"synthetic",
                "source_family_disjoint":true,
                "rights_review_status":"metadata_only",
                "case_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "source_catalog_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "source_pack_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "systems":[
                {"system_id":"s1","baseline_family":"bm25"},
                {"system_id":"s2","baseline_family":"typed_graph"}
            ],
            "preregistration":{"held_out_access":"not_applicable_synthetic"}
        })).unwrap()
    }

    fn synthetic_pair_state() -> PairState {
        PairState {
            context_id: "synthetic-context".into(),
            tradition_scope: "synthetic-tradition".into(),
            source_id: "synthetic-source".into(),
            variant_id: "synthetic-variant".into(),
            prompt: "Synthetic control prompt".into(),
            rights_status: "synthetic_only".into(),
            access_decision: "synthetic_only".into(),
            evidence_pack: vec![PairEvidence {
                evidence_id: "synthetic-evidence".into(),
                source_id: "synthetic-source".into(),
                statement: "Synthetic control evidence".into(),
                kind: "synthetic_control".into(),
            }],
        }
    }

    fn pair(
        id: &str,
        dimension: &str,
        expected: &str,
        criticality: &str,
        must_change: bool,
    ) -> PairCase {
        PairCase {
            pair_id: id.into(),
            split: "synthetic".into(),
            synthetic_only: true,
            dimension: dimension.into(),
            expected_relation: expected.into(),
            criticality: criticality.into(),
            source_family_ids: vec![format!("SYN-FAMILY-{id}")],
            baseline: synthetic_pair_state(),
            mutated: synthetic_pair_state(),
            acceptance: Acceptance {
                must_preserve: vec!["source identity".into()],
                must_change: if must_change {
                    vec!["expected behavior changes".into()]
                } else {
                    vec![]
                },
                must_not: vec!["fabricated citation".into()],
            },
        }
    }

    fn observation(_id: &str, system: &str, pair: &str, relation: &str) -> Observation {
        Observation {
            schema_version: "paired-intervention-result-v0".into(),
            result_id: format!("{system}-{pair}"),
            plan_id: "test-plan".into(),
            system_id: system.into(),
            pair_id: pair.into(),
            observed_relation: relation.into(),
            preservation_satisfied: true,
            change_satisfied: true,
            forbidden_behaviors: vec![],
            citation_identity_valid: true,
            adjudication_status: "synthetic_only".into(),
            notes: "synthetic test label".into(),
        }
    }

    #[test]
    fn nonsynthetic_pair_rejects_metadata_only_source() {
        let source = SourceRecord {
            source_id: "source-1".into(),
            source_family_id: "family-1".into(),
            split: "holdout".into(),
            rights_status: "unknown".into(),
            content_access: "metadata_only".into(),
            rights_review: "pending".into(),
        };
        let catalog = HashMap::from([("source-1", &source)]);
        let state = PairState {
            context_id: "ctx".into(),
            tradition_scope: "example".into(),
            source_id: "source-1".into(),
            variant_id: "v1".into(),
            prompt: "summarize source".into(),
            rights_status: "unknown".into(),
            access_decision: "allow".into(),
            evidence_pack: vec![PairEvidence {
                evidence_id: "e1".into(),
                source_id: "source-1".into(),
                statement: "evidence".into(),
                kind: "source_attestation".into(),
            }],
        };
        let error = validate_real_pair_state("pair-1", "holdout", "baseline", &state, &catalog)
            .unwrap_err();
        assert!(error.contains("lacks approved content access"));
    }

    #[test]
    fn nonsynthetic_pair_accepts_approved_content_sources() {
        let source = SourceRecord {
            source_id: "source-1".into(),
            source_family_id: "family-1".into(),
            split: "holdout".into(),
            rights_status: "open_license_verified".into(),
            content_access: "verified_open_content".into(),
            rights_review: "approved_limited_use".into(),
        };
        let catalog = HashMap::from([("source-1", &source)]);
        let state = PairState {
            context_id: "ctx".into(),
            tradition_scope: "example".into(),
            source_id: "source-1".into(),
            variant_id: "v1".into(),
            prompt: "summarize source".into(),
            rights_status: "open_license_verified".into(),
            access_decision: "allow".into(),
            evidence_pack: vec![PairEvidence {
                evidence_id: "e1".into(),
                source_id: "source-1".into(),
                statement: "evidence".into(),
                kind: "source_attestation".into(),
            }],
        };
        assert!(
            validate_real_pair_state("pair-1", "holdout", "baseline", &state, &catalog).is_ok()
        );
    }

    #[test]
    fn source_pack_snapshot_must_match_pair_state_exactly() {
        let case = pair("p1", "context_swap", "local_change", "major", true);
        let mut snapshot = SourcePackSnapshot {
            schema_version: "knowledge-source-pack-v0".into(),
            synthetic_only: true,
            split: "synthetic".into(),
            pair_id: "p1".into(),
            state: "baseline".into(),
            state_data: case.baseline.clone(),
        };
        let mut mutated_snapshot = SourcePackSnapshot {
            schema_version: "knowledge-source-pack-v0".into(),
            synthetic_only: true,
            split: "synthetic".into(),
            pair_id: "p1".into(),
            state: "mutated".into(),
            state_data: case.mutated.clone(),
        };
        assert!(
            validate_source_pack_matches_pairs(
                std::slice::from_ref(&case),
                &[snapshot.clone(), mutated_snapshot.clone()],
                true
            )
            .is_ok()
        );

        mutated_snapshot.state_data.prompt = "altered source-pack prompt".into();
        snapshot.state_data.prompt = "baseline altered prompt".into();
        let error = validate_source_pack_matches_pairs(
            std::slice::from_ref(&case),
            &[snapshot, mutated_snapshot],
            true,
        )
        .unwrap_err();
        assert!(error.contains("does not match pair"));
    }

    #[test]
    fn held_out_inference_rejects_too_few_independent_clusters() {
        let mut plan = fixture_plan();
        plan.phase = "preregistered_held_out".into();
        let error = compute_pairwise_comparisons(&plan, &[], &[], &[vec![0]]).unwrap_err();
        assert!(error.contains("at least 30 independent source-family clusters"));
    }

    #[test]
    fn connected_source_families_form_one_bootstrap_cluster() {
        let mut first = pair("p1", "source_swap", "local_change", "major", true);
        let mut second = pair("p2", "tradition_transfer", "local_change", "major", true);
        first.source_family_ids = vec!["family-A".into(), "family-B".into()];
        second.source_family_ids = vec!["family-B".into(), "family-C".into()];
        let clusters = source_family_clusters(&[first, second]).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0, 1]);
    }

    #[test]
    fn paired_cluster_bootstrap_is_deterministic_and_reports_ci() {
        let plan = fixture_plan();
        let pairs = vec![
            pair("p1", "context_swap", "local_change", "major", true),
            pair("p2", "alias_equivalence", "invariant", "diagnostic", false),
        ];
        let results = vec![
            observation("r1", "s1", "p1", "incorrect"),
            observation("r2", "s1", "p2", "incorrect"),
            observation("r3", "s2", "p1", "local_change"),
            observation("r4", "s2", "p2", "invariant"),
        ];
        let clusters = source_family_clusters(&pairs).unwrap();
        assert_eq!(clusters.len(), 2);
        let first = compute_pairwise_comparisons(&plan, &pairs, &results, &clusters).unwrap();
        let second = compute_pairwise_comparisons(&plan, &pairs, &results, &clusters).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].baseline_system_id, "s1");
        assert_eq!(first[0].candidate_system_id, "s2");
        assert_eq!(first[0].source_family_cluster_count, 2);
        assert_eq!(first[0].bootstrap_replicates, plan.bootstrap.replicates);
        assert_eq!(first[0].confidence_interval_low, 1.0);
        assert_eq!(first[0].confidence_interval_high, 1.0);
        assert!(
            !first[0].candidate_improvement_supported,
            "positive synthetic-control intervals must not support an improvement claim"
        );
    }

    fn held_out_bootstrap_fixture() -> (Plan, Vec<PairCase>, Vec<Observation>) {
        let mut plan = fixture_plan();
        plan.phase = "preregistered_held_out".into();
        plan.synthetic_only = false;
        plan.benchmark.split = "holdout".into();
        plan.benchmark.source_family_disjoint = true;
        plan.benchmark.rights_review_status = "approved".into();
        plan.preregistration.held_out_access = "sealed_until_freeze".into();

        let mut pairs = Vec::new();
        let mut observations = Vec::new();
        for index in 0..30 {
            let pair_id = format!("holdout-{index:02}");
            let mut case = pair(
                &pair_id,
                "context_swap",
                "local_change",
                if index == 0 {
                    "release_blocker"
                } else {
                    "major"
                },
                true,
            );
            case.split = "holdout".into();
            case.synthetic_only = false;
            case.source_family_ids = vec![format!("independent-family-{index:02}")];
            pairs.push(case);
            observations.push(observation("base", "s1", &pair_id, "incorrect"));
            observations.push(observation(
                "candidate",
                "s2",
                &pair_id,
                if index == 0 {
                    "incorrect"
                } else {
                    "local_change"
                },
            ));
        }
        (plan, pairs, observations)
    }

    #[test]
    fn positive_interval_does_not_override_a_candidate_critical_gate_failure() {
        let (plan, pairs, observations) = held_out_bootstrap_fixture();
        let clusters = source_family_clusters(&pairs).unwrap();
        assert_eq!(clusters.len(), 30);
        let comparisons =
            compute_pairwise_comparisons(&plan, &pairs, &observations, &clusters).unwrap();
        assert_eq!(comparisons.len(), 1);
        assert!(
            comparisons[0].confidence_interval_low > 0.0,
            "fixture must show a positive interval to exercise non-compensatory gating"
        );
        assert!(
            !comparisons[0].candidate_improvement_supported,
            "a release-blocker failure must prevent an improvement claim"
        );
    }

    #[test]
    fn sufficiently_supported_held_out_improvement_requires_all_candidate_gates() {
        let (plan, pairs, mut observations) = held_out_bootstrap_fixture();
        for row in &mut observations {
            if row.system_id == "s2" {
                row.observed_relation = "local_change".into();
            }
        }
        let clusters = source_family_clusters(&pairs).unwrap();
        let comparisons =
            compute_pairwise_comparisons(&plan, &pairs, &observations, &clusters).unwrap();
        assert_eq!(comparisons.len(), 1);
        assert!(comparisons[0].confidence_interval_low > 0.0);
        assert!(comparisons[0].candidate_improvement_supported);
    }

    #[test]
    fn declared_input_digest_mismatch_fails_closed() {
        let error = verify_sha256("source pack", b"observed bytes", &"0".repeat(64)).unwrap_err();
        assert!(error.contains("source pack digest mismatch"));
    }

    #[test]
    fn matching_input_digest_returns_exact_content_hash() {
        let bytes = b"fixed source manifest bytes";
        let digest = sha256_hex(bytes);
        assert_eq!(
            verify_sha256("case manifest", bytes, &digest).unwrap(),
            digest
        );
    }

    #[test]
    fn bundled_score_report_schema_is_valid_draft_2020_12() {
        let schema: Value = serde_json::from_str(SCORE_REPORT_SCHEMA).unwrap();
        assert_eq!(
            schema.get("$schema").and_then(Value::as_str),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
        let meta_validator = jsonschema::meta::validator_for(&schema).unwrap();
        assert!(meta_validator.evaluate(&schema).flag().valid);
        jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .unwrap();
    }

    #[test]
    fn paired_fidelity_is_non_compensatory_on_critical_failures() {
        let plan = fixture_plan();
        let pairs = vec![
            pair(
                "p1",
                "context_swap",
                "local_change",
                "release_blocker",
                true,
            ),
            pair("p2", "alias_equivalence", "invariant", "diagnostic", false),
        ];
        let mut results = Vec::new();
        for system in ["s1", "s2"] {
            results.push(observation("r1", system, "p1", "local_change"));
            results.push(observation("r2", system, "p2", "invariant"));
        }
        let index = results
            .iter()
            .position(|row| row.system_id == "s1" && row.pair_id == "p1")
            .unwrap();
        results[index]
            .forbidden_behaviors
            .push("source fabrication".into());
        let metrics = score(&plan, pairs, results, &[]).unwrap();
        let s1 = metrics.iter().find(|m| m.system_id == "s1").unwrap();
        let s2 = metrics.iter().find(|m| m.system_id == "s2").unwrap();
        assert_eq!(s1.critical_gate_violations, 1);
        assert!(!s1.critical_gates_pass);
        assert_eq!(s2.critical_gate_violations, 0);
        assert!(s2.critical_gates_pass);
        assert!(s1.paired_intervention_fidelity < s2.paired_intervention_fidelity);
    }

    #[test]
    fn missing_pair_result_fails_closed() {
        let plan = fixture_plan();
        let pairs = vec![
            pair("p1", "context_swap", "local_change", "major", true),
            pair("p2", "alias_equivalence", "invariant", "diagnostic", false),
        ];
        let results = vec![
            observation("r1", "s1", "p1", "local_change"),
            observation("r2", "s2", "p1", "local_change"),
            observation("r3", "s2", "p2", "invariant"),
        ];
        let error = score(&plan, pairs, results, &[]).unwrap_err();
        assert!(error.contains("incomplete pair coverage"));
    }

    #[test]
    fn synthetic_pairs_cannot_be_scored_as_held_out() {
        let mut plan = fixture_plan();
        plan.phase = "preregistered_held_out".into();
        plan.synthetic_only = false;
        plan.benchmark.split = "holdout".into();
        plan.benchmark.source_family_disjoint = true;
        plan.benchmark.rights_review_status = "approved".into();
        plan.preregistration.held_out_access = "sealed_until_freeze".into();
        let pairs = vec![
            pair(
                "p1",
                "context_swap",
                "local_change",
                "release_blocker",
                true,
            ),
            pair("p2", "alias_equivalence", "invariant", "diagnostic", false),
        ];
        let results = vec![
            observation("r1", "s1", "p1", "local_change"),
            observation("r2", "s1", "p2", "invariant"),
            observation("r3", "s2", "p1", "local_change"),
            observation("r4", "s2", "p2", "invariant"),
        ];
        let error = score(&plan, pairs, results, &[]).unwrap_err();
        assert!(error.contains("split/synthetic status"));
    }

    #[test]
    fn rejects_unknown_multiple_comparison_correction() {
        let mut plan = fixture_plan();
        plan.bootstrap.multiple_comparison_correction = "invented".into();
        let error = validate_plan_eligibility(&plan).unwrap_err();
        assert!(error.contains("multiple_comparison_correction must be none or bonferroni"));
    }

    #[test]
    fn held_out_comparisons_require_familywise_correction_for_multiple_baselines() {
        let mut plan = fixture_plan();
        plan.phase = "preregistered_held_out".into();
        plan.synthetic_only = false;
        plan.benchmark.split = "holdout".into();
        plan.benchmark.source_family_disjoint = true;
        plan.benchmark.rights_review_status = "approved".into();
        plan.preregistration.held_out_access = "sealed_until_freeze".into();
        plan.systems.push(System {
            system_id: "s3".into(),
            baseline_family: "dense_retrieval".into(),
        });
        plan.bootstrap.multiple_comparison_correction = "none".into();
        let error = validate_plan_eligibility(&plan).unwrap_err();
        assert!(error.contains("require familywise correction"));

        plan.bootstrap.multiple_comparison_correction = "bonferroni".into();
        assert!(validate_plan_eligibility(&plan).is_ok());
    }

    #[test]
    fn held_out_scoring_requires_rights_and_source_family_isolation() {
        let mut plan = fixture_plan();
        plan.phase = "preregistered_held_out".into();
        plan.synthetic_only = false;
        plan.benchmark.split = "holdout".into();
        plan.benchmark.source_family_disjoint = false;
        plan.benchmark.rights_review_status = "pending".into();
        plan.preregistration.held_out_access = "not_sealed".into();
        let error = validate_plan_eligibility(&plan).unwrap_err();
        assert!(error.contains("held-out scoring requires"));
    }
}
