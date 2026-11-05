use jsonschema::draft202012;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const TOOL_VERSION: &str = "score-multihall-sa-v0.1.0";
const REPORT_SCHEMA: &str = include_str!("../../../../benchmarks/multi-hall-sa/v0/score-report.schema.json");

#[derive(Debug)]
struct Args {
    plan: PathBuf,
    pairs: PathBuf,
    results: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Plan {
    schema_version: String,
    plan_id: String,
    phase: String,
    synthetic_only: bool,
    candidate_system_id: String,
    dataset: Dataset,
    systems: Vec<SystemPlan>,
    bootstrap: BootstrapPlan,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    dataset_id: String,
    revision: String,
    license: String,
    pair_manifest_sha256: String,
    split: String,
    source_family_disjoint: bool,
    rights_review_status: String,
    held_out_access: String,
    source_cluster_semantics_reviewed: bool,
}

#[derive(Debug, Deserialize)]
struct SystemPlan {
    system_id: String,
    baseline_family: String,
    model_id: String,
    model_revision: String,
    retrieval_profile_id: Option<String>,
    prompt_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BootstrapPlan {
    replicates: usize,
    seed: u64,
    minimum_independent_clusters: usize,
    confidence_level: f64,
    multiple_comparison_correction: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PairMeta {
    schema_version: String,
    pair_id: String,
    file_id: String,
    fact_index: u64,
    language: String,
    hallucination_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Prediction {
    schema_version: String,
    result_id: String,
    plan_id: String,
    system_id: String,
    pair_id: String,
    file_id: String,
    fact_index: u64,
    language: String,
    hallucination_type: String,
    statement_language: String,
    statement_kind: String,
    predicted_label: String,
    p_hallucinated: Option<f64>,
    latency_ms: Option<u64>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    retrieval_calls: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BootstrapCi {
    lower: f64,
    upper: f64,
    confidence_level: f64,
    familywise_adjusted: bool,
    comparison_count: usize,
    replicates: usize,
    seed: u64,
    cluster_count: usize,
}

#[derive(Debug, Serialize)]
struct SliceMetrics {
    language: String,
    hallucination_type: Option<String>,
    statement_count: usize,
    pair_count: usize,
    factual_count: usize,
    hallucinated_count: usize,
    true_positive: usize,
    false_positive: usize,
    true_negative: usize,
    false_negative: usize,
    abstained: usize,
    coverage: Option<f64>,
    accuracy_on_covered: Option<f64>,
    precision_hallucination: Option<f64>,
    recall_hallucination_on_covered: Option<f64>,
    f1_hallucination: Option<f64>,
    f1_factual: Option<f64>,
    macro_f1: Option<f64>,
    false_negative_rate_on_covered: Option<f64>,
    missed_hallucination_rate_unconditional: Option<f64>,
    false_hallucination_rate_unconditional: Option<f64>,
    selective_risk: Option<f64>,
    probability_count: usize,
    probability_coverage: Option<f64>,
    brier_score: Option<f64>,
    expected_calibration_error_10_bins: Option<f64>,
    mean_latency_ms: Option<f64>,
    mean_input_tokens: Option<f64>,
    mean_output_tokens: Option<f64>,
    mean_retrieval_calls: Option<f64>,
}

#[derive(Debug, Serialize)]
struct LanguageGap {
    target_language: String,
    paired_pair_count: usize,
    cluster_count: usize,
    english_unconditional_missed_hallucination_rate: Option<f64>,
    target_unconditional_missed_hallucination_rate: Option<f64>,
    paired_missed_hallucination_gap: Option<f64>,
    bootstrap_ci: Option<BootstrapCi>,
    inference_eligible: bool,
    note: String,
}

#[derive(Debug, Serialize)]
struct SystemComparison {
    baseline_system_id: String,
    candidate_system_id: String,
    language: String,
    pair_count: usize,
    cluster_count: usize,
    baseline_pair_balanced_accuracy: Option<f64>,
    candidate_pair_balanced_accuracy: Option<f64>,
    paired_accuracy_difference: Option<f64>,
    bootstrap_ci: Option<BootstrapCi>,
    inference_eligible: bool,
    note: String,
}

#[derive(Debug, Serialize)]
struct SystemReport {
    system_id: String,
    baseline_family: String,
    model_id: String,
    model_revision: String,
    retrieval_profile_id: Option<String>,
    prompt_sha256: Option<String>,
    is_candidate: bool,
    slices: Vec<SliceMetrics>,
    language_gaps: Vec<LanguageGap>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    tool_version: &'static str,
    plan_id: String,
    phase: String,
    synthetic_only: bool,
    dataset_id: String,
    dataset_revision: String,
    dataset_license: String,
    plan_sha256: String,
    pair_manifest_sha256: String,
    results_sha256: String,
    pair_count: usize,
    result_count: usize,
    cluster_count: usize,
    minimum_independent_clusters: usize,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    confidence_level: f64,
    multiple_comparison_correction: String,
    systems: Vec<SystemReport>,
    system_comparisons: Vec<SystemComparison>,
    nonclaims: Vec<&'static str>,
}

#[derive(Debug, Clone)]
struct Observed {
    pair: PairMeta,
    prediction: Prediction,
}

#[derive(Debug, Clone, Copy)]
struct PairedMiss {
    target_missed: bool,
    english_missed: bool,
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn usage() -> &'static str {
    "Usage: score-multihall-sa --plan FILE --pairs JSONL --results JSONL --out FILE\n\
     Scores statement-level predictions against a digest-pinned Multi-Hall-SA pair manifest.\n\
     It does not fetch dataset text, execute models, verify source truth, or prove source independence."
}

fn parse_args() -> Result<Args, String> {
    let mut raw = env::args().skip(1);
    let mut plan = None;
    let mut pairs = None;
    let mut results = None;
    let mut output = None;

    while let Some(argument) = raw.next() {
        let mut next_value = |flag: &str| {
            raw.next().ok_or_else(|| format!("{flag} requires a path"))
        };
        match argument.as_str() {
            "--plan" => plan = Some(PathBuf::from(next_value("--plan")?)),
            "--pairs" => pairs = Some(PathBuf::from(next_value("--pairs")?)),
            "--results" => results = Some(PathBuf::from(next_value("--results")?)),
            "--out" => output = Some(PathBuf::from(next_value("--out")?)),
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(Args {
        plan: plan.ok_or_else(|| format!("missing --plan\n{}", usage()))?,
        pairs: pairs.ok_or_else(|| format!("missing --pairs\n{}", usage()))?,
        results: results.ok_or_else(|| format!("missing --results\n{}", usage()))?,
        output: output.ok_or_else(|| format!("missing --out\n{}", usage()))?,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_jsonl<T: DeserializeOwned>(bytes: &[u8], label: &str) -> Result<Vec<T>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    let mut rows = Vec::new();

    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("{label} contains an empty line at {}", index + 1));
        }
        let row = serde_json::from_str::<T>(line)
            .map_err(|error| format!("{label} line {} is invalid JSON: {error}", index + 1))?;
        rows.push(row);
    }

    if rows.is_empty() {
        return Err(format!("{label} contains no rows"));
    }
    Ok(rows)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_plan(plan: &Plan) -> Result<(), String> {
    if plan.schema_version != "multi-hall-sa-plan-v0" {
        return Err(format!("unsupported plan schema_version {:?}", plan.schema_version));
    }
    if plan.plan_id.trim().is_empty() || plan.candidate_system_id.trim().is_empty() {
        return Err("plan_id and candidate_system_id must be non-empty".into());
    }
    if !is_sha256(&plan.dataset.pair_manifest_sha256) {
        return Err("dataset.pair_manifest_sha256 must be 64 lowercase hexadecimal characters".into());
    }
    if plan.dataset.dataset_id != "Multi-Hall-SA" && !plan.synthetic_only {
        return Err("non-synthetic plans must identify dataset_id=Multi-Hall-SA".into());
    }
    if plan.bootstrap.replicates < 1_000 || plan.bootstrap.replicates > 100_000 {
        return Err("bootstrap replicates must be between 1000 and 100000".into());
    }
    if plan.bootstrap.minimum_independent_clusters < 30 {
        return Err("minimum_independent_clusters must be at least 30".into());
    }
    if !(plan.bootstrap.confidence_level > 0.5 && plan.bootstrap.confidence_level < 1.0) {
        return Err("confidence_level must be in (0.5, 1.0)".into());
    }
    if !matches!(plan.bootstrap.multiple_comparison_correction.as_str(), "none" | "bonferroni") {
        return Err("multiple_comparison_correction must be none or bonferroni".into());
    }

    match plan.phase.as_str() {
        "synthetic_control" => {
            if !plan.synthetic_only
                || plan.dataset.revision != "synthetic-only"
                || plan.dataset.license != "synthetic"
                || plan.dataset.split != "synthetic"
                || plan.dataset.rights_review_status != "not_applicable_synthetic"
                || plan.dataset.held_out_access != "not_applicable_synthetic"
            {
                return Err("synthetic_control requires synthetic_only=true and synthetic dataset/rights/access declarations".into());
            }
        }
        "development_pilot" => {
            if plan.synthetic_only
                || !matches!(plan.dataset.split.as_str(), "development" | "validation")
                || plan.dataset.rights_review_status != "approved"
            {
                return Err("development_pilot requires non-synthetic development/validation data with approved rights".into());
            }
            if !is_git_sha(&plan.dataset.revision) {
                return Err("non-synthetic dataset revision must be an immutable 40-character commit SHA".into());
            }
        }
        "preregistered_held_out" => {
            if plan.synthetic_only
                || !matches!(plan.dataset.split.as_str(), "test" | "holdout")
                || !plan.dataset.source_family_disjoint
                || plan.dataset.rights_review_status != "approved"
                || !matches!(plan.dataset.held_out_access.as_str(), "sealed_until_freeze" | "independent_evaluator")
                || !plan.dataset.source_cluster_semantics_reviewed
            {
                return Err("held-out scoring requires real test/holdout data, source-family disjointness, approved rights, sealed/independent access and reviewed cluster semantics".into());
            }
            if !is_git_sha(&plan.dataset.revision) {
                return Err("non-synthetic dataset revision must be an immutable 40-character commit SHA".into());
            }
            if plan.systems.len() > 1 && plan.bootstrap.multiple_comparison_correction != "bonferroni" {
                return Err("held-out multi-system comparisons require Bonferroni familywise correction".into());
            }
        }
        other => return Err(format!("unsupported phase {other:?}")),
    }

    let mut system_ids = HashSet::new();
    for system in &plan.systems {
        if system.system_id.trim().is_empty()
            || system.baseline_family.trim().is_empty()
            || system.model_id.trim().is_empty()
            || system.model_revision.trim().is_empty()
        {
            return Err("system_id, baseline_family, model_id and model_revision must be non-empty".into());
        }
        if !system_ids.insert(system.system_id.clone()) {
            return Err(format!("duplicate system_id {:?}", system.system_id));
        }
        if let Some(prompt_sha) = &system.prompt_sha256 {
            if !is_sha256(prompt_sha) {
                return Err(format!("system {:?} has invalid prompt_sha256", system.system_id));
            }
        }
    }
    if !system_ids.contains(&plan.candidate_system_id) {
        return Err("candidate_system_id is not present in systems".into());
    }
    Ok(())
}

fn pair_key_from_fields(
    file_id: &str,
    pair_id: &str,
    fact_index: u64,
    language: &str,
    hallucination_type: &str,
) -> String {
    serde_json::to_string(&(file_id, pair_id, fact_index, language, hallucination_type))
        .expect("serializing a tuple of strings and an integer cannot fail")
}

fn pair_key(pair: &PairMeta) -> String {
    pair_key_from_fields(
        &pair.file_id,
        &pair.pair_id,
        pair.fact_index,
        &pair.language,
        &pair.hallucination_type,
    )
}

fn result_pair_key(result: &Prediction) -> String {
    pair_key_from_fields(
        &result.file_id,
        &result.pair_id,
        result.fact_index,
        &result.language,
        &result.hallucination_type,
    )
}

fn validate_pairs(pairs: &[PairMeta]) -> Result<BTreeMap<String, PairMeta>, String> {
    let allowed_languages = ["isiZulu", "isiXhosa", "Sepedi", "Sesotho"];
    let allowed_types = ["temporal shift", "entity error", "numerical inaccuracy", "location mistake"];
    let mut by_id = BTreeMap::new();
    let mut composite_keys = HashSet::new();

    for pair in pairs {
        if pair.schema_version != "multi-hall-sa-pair-v0" {
            return Err(format!("pair {:?} has unsupported schema_version", pair.pair_id));
        }
        if pair.pair_id.trim().is_empty() || pair.file_id.trim().is_empty() {
            return Err("pair_id and file_id must be non-empty".into());
        }
        if !allowed_languages.contains(&pair.language.as_str()) {
            return Err(format!("pair {:?} has unsupported language {:?}", pair.pair_id, pair.language));
        }
        if !allowed_types.contains(&pair.hallucination_type.as_str()) {
            return Err(format!("pair {:?} has unsupported hallucination_type {:?}", pair.pair_id, pair.hallucination_type));
        }
        // pair_id is not globally unique in the released builder: fact_index is
        // reset for each source document, while file_id is assigned per document.
        // Keep the complete source/fact/language/type tuple as the identity.
        let key = pair_key(pair);
        if !composite_keys.insert(key.clone()) {
            return Err(format!(
                "duplicate composite pair identity for file_id={:?}, pair_id={:?}, fact_index={}, language={:?}, type={:?}",
                pair.file_id, pair.pair_id, pair.fact_index, pair.language, pair.hallucination_type
            ));
        }
        by_id.insert(key, pair.clone());
    }
    Ok(by_id)
}

fn validate_predictions(
    plan: &Plan,
    pairs: &BTreeMap<String, PairMeta>,
    results: &[Prediction],
) -> Result<HashMap<(String, String, String, String), Prediction>, String> {
    let systems: HashSet<&str> = plan.systems.iter().map(|system| system.system_id.as_str()).collect();
    let mut result_ids = HashSet::new();
    let mut keyed = HashMap::new();

    for result in results {
        if result.schema_version != "multi-hall-sa-result-v0" {
            return Err(format!("result {:?} has unsupported schema_version", result.result_id));
        }
        if result.plan_id != plan.plan_id {
            return Err(format!("result {:?} has plan_id mismatch", result.result_id));
        }
        if !systems.contains(result.system_id.as_str()) {
            return Err(format!("result {:?} references unknown system {:?}", result.result_id, result.system_id));
        }
        let pair_key = result_pair_key(result);
        if !pairs.contains_key(&pair_key) {
            return Err(format!(
                "result {:?} references unknown composite pair identity (file_id={:?}, pair_id={:?}, fact_index={}, language={:?}, type={:?})",
                result.result_id, result.file_id, result.pair_id, result.fact_index, result.language, result.hallucination_type
            ));
        }
        if result.result_id.trim().is_empty() || !result_ids.insert(result.result_id.clone()) {
            return Err(format!("empty or duplicate result_id {:?}", result.result_id));
        }
        if !matches!(result.statement_language.as_str(), "english" | "target") {
            return Err(format!("result {:?} has invalid statement_language", result.result_id));
        }
        if !matches!(result.statement_kind.as_str(), "factual" | "hallucinated") {
            return Err(format!("result {:?} has invalid statement_kind", result.result_id));
        }
        if !matches!(result.predicted_label.as_str(), "factual" | "hallucinated" | "abstain") {
            return Err(format!("result {:?} has invalid predicted_label", result.result_id));
        }
        if let Some(probability) = result.p_hallucinated {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(format!("result {:?} has p_hallucinated outside [0,1]", result.result_id));
            }
        }

        let key = (
            result.system_id.clone(),
            pair_key,
            result.statement_language.clone(),
            result.statement_kind.clone(),
        );
        if keyed.insert(key, result.clone()).is_some() {
            return Err(format!("duplicate statement prediction for pair {:?}", result.pair_id));
        }
    }

    for system in &plan.systems {
        for pair_id in pairs.keys() {
            for language in ["english", "target"] {
                for kind in ["factual", "hallucinated"] {
                    let key = (
                        system.system_id.clone(),
                        pair_id.clone(),
                        language.to_string(),
                        kind.to_string(),
                    );
                    if !keyed.contains_key(&key) {
                        return Err(format!("missing prediction: system={}, pair={}, language={}, kind={}", system.system_id, pair_id, language, kind));
                    }
                }
            }
        }
    }

    let expected = plan.systems.len() * pairs.len() * 4;
    if results.len() != expected {
        return Err(format!("prediction row count {} does not equal expected complete matrix {}", results.len(), expected));
    }
    Ok(keyed)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

fn safe_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
}

fn class_f1(tp: usize, fp: usize, false_negative: usize) -> Option<f64> {
    let denominator = 2 * tp + fp + false_negative;
    if denominator == 0 { None } else { Some((2 * tp) as f64 / denominator as f64) }
}

fn calibration_metrics(rows: &[&Observed]) -> (usize, Option<f64>, Option<f64>, Option<f64>) {
    let mut values = Vec::new();
    for row in rows {
        if let Some(probability) = row.prediction.p_hallucinated {
            let actual = if row.prediction.statement_kind == "hallucinated" { 1.0 } else { 0.0 };
            values.push((probability, actual));
        }
    }
    if values.is_empty() {
        return (0, None, None, None);
    }

    let brier = values.iter()
        .map(|(probability, actual)| (probability - actual).powi(2))
        .sum::<f64>() / values.len() as f64;

    let mut bins: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 10];
    for (probability, actual) in &values {
        let index = ((*probability * 10.0).floor() as usize).min(9);
        bins[index].push((*probability, *actual));
    }
    let mut ece = 0.0;
    for bin in bins {
        if bin.is_empty() { continue; }
        let mean_probability = bin.iter().map(|row| row.0).sum::<f64>() / bin.len() as f64;
        let mean_actual = bin.iter().map(|row| row.1).sum::<f64>() / bin.len() as f64;
        ece += (bin.len() as f64 / values.len() as f64) * (mean_probability - mean_actual).abs();
    }
    (values.len(), Some(values.len() as f64 / rows.len() as f64), Some(brier), Some(ece))
}

fn metric_slice(rows: &[&Observed], language: &str, hallucination_type: Option<String>) -> SliceMetrics {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_count = 0usize;
    let mut abstained = 0usize;
    let mut abstained_hallucinated = 0usize;
    let mut factual_count = 0usize;
    let mut hallucinated_count = 0usize;
    let mut latency = Vec::new();
    let mut input_tokens = Vec::new();
    let mut output_tokens = Vec::new();
    let mut retrieval_calls = Vec::new();

    for row in rows {
        let actual_hallucinated = row.prediction.statement_kind == "hallucinated";
        if actual_hallucinated { hallucinated_count += 1; } else { factual_count += 1; }

        match row.prediction.predicted_label.as_str() {
            "hallucinated" if actual_hallucinated => tp += 1,
            "hallucinated" => fp += 1,
            "factual" if actual_hallucinated => fn_count += 1,
            "factual" => tn += 1,
            "abstain" => {
                abstained += 1;
                if actual_hallucinated { abstained_hallucinated += 1; }
            }
            _ => unreachable!("predicted_label is validated before scoring"),
        }

        if let Some(value) = row.prediction.latency_ms { latency.push(value as f64); }
        if let Some(value) = row.prediction.input_tokens { input_tokens.push(value as f64); }
        if let Some(value) = row.prediction.output_tokens { output_tokens.push(value as f64); }
        if let Some(value) = row.prediction.retrieval_calls { retrieval_calls.push(value as f64); }
    }

    let covered = tp + fp + tn + fn_count;
    let correct = tp + tn;
    let errors = fp + fn_count;
    let f1_hall = class_f1(tp, fp, fn_count);
    let f1_factual = class_f1(tn, fn_count, fp);
    let macro_f1 = match (f1_hall, f1_factual) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    };
    let (probability_count, probability_coverage, brier, ece) = calibration_metrics(rows);

    SliceMetrics {
        language: language.to_string(),
        hallucination_type,
        statement_count: rows.len(),
        pair_count: rows.iter().map(|row| pair_key(&row.pair)).collect::<HashSet<_>>().len(),
        factual_count,
        hallucinated_count,
        true_positive: tp,
        false_positive: fp,
        true_negative: tn,
        false_negative: fn_count,
        abstained,
        coverage: ratio(covered, rows.len()),
        accuracy_on_covered: ratio(correct, covered),
        precision_hallucination: ratio(tp, tp + fp),
        recall_hallucination_on_covered: ratio(tp, tp + fn_count),
        f1_hallucination: f1_hall,
        f1_factual,
        macro_f1,
        false_negative_rate_on_covered: ratio(fn_count, tp + fn_count),
        missed_hallucination_rate_unconditional: ratio(fn_count + abstained_hallucinated, hallucinated_count),
        false_hallucination_rate_unconditional: ratio(fp, factual_count),
        selective_risk: ratio(errors, covered),
        probability_count,
        probability_coverage,
        brier_score: brier,
        expected_calibration_error_10_bins: ece,
        mean_latency_ms: safe_mean(&latency),
        mean_input_tokens: safe_mean(&input_tokens),
        mean_output_tokens: safe_mean(&output_tokens),
        mean_retrieval_calls: safe_mean(&retrieval_calls),
    }
}

fn paired_misses(
    system_id: &str,
    target_language: &str,
    pairs: &BTreeMap<String, PairMeta>,
    predictions: &HashMap<(String, String, String, String), Prediction>,
) -> BTreeMap<String, Vec<PairedMiss>> {
    let mut clusters: BTreeMap<String, Vec<PairedMiss>> = BTreeMap::new();
    for pair in pairs.values().filter(|pair| pair.language == target_language) {
        let composite_pair_key = pair_key(pair);
        let english_key = (
            system_id.to_string(), composite_pair_key.clone(), "english".to_string(), "hallucinated".to_string()
        );
        let target_key = (
            system_id.to_string(), composite_pair_key, "target".to_string(), "hallucinated".to_string()
        );
        let english = predictions.get(&english_key).expect("complete result matrix validated");
        let target = predictions.get(&target_key).expect("complete result matrix validated");
        let miss = |result: &Prediction| result.predicted_label != "hallucinated";
        clusters.entry(pair.file_id.clone()).or_default().push(PairedMiss {
            target_missed: miss(target),
            english_missed: miss(english),
        });
    }
    clusters
}

fn paired_rates(clusters: &BTreeMap<String, Vec<PairedMiss>>) -> (usize, Option<f64>, Option<f64>, Option<f64>) {
    let mut target = 0usize;
    let mut english = 0usize;
    let mut count = 0usize;
    for rows in clusters.values() {
        for row in rows {
            if row.target_missed { target += 1; }
            if row.english_missed { english += 1; }
            count += 1;
        }
    }
    let target_rate = ratio(target, count);
    let english_rate = ratio(english, count);
    let gap = match (target_rate, english_rate) {
        (Some(target), Some(english)) => Some(target - english),
        _ => None,
    };
    (count, english_rate, target_rate, gap)
}

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() { return None; }
    let index = ((sorted.len() - 1) as f64 * probability).round() as usize;
    sorted.get(index).copied()
}

fn bootstrap_gap_ci(
    clusters: &BTreeMap<String, Vec<PairedMiss>>,
    plan: &BootstrapPlan,
    comparison_count: usize,
) -> Option<BootstrapCi> {
    if clusters.len() < plan.minimum_independent_clusters || clusters.is_empty() {
        return None;
    }
    let cluster_values: Vec<&Vec<PairedMiss>> = clusters.values().collect();
    let mut rng = SplitMix64::new(plan.seed);
    let familywise_adjusted = plan.multiple_comparison_correction == "bonferroni";
    let alpha = 1.0 - plan.confidence_level;
    let tail = if familywise_adjusted {
        alpha / (2.0 * comparison_count.max(1) as f64)
    } else {
        alpha / 2.0
    };
    let mut gaps = Vec::with_capacity(plan.replicates);

    for _ in 0..plan.replicates {
        let mut target_misses = 0usize;
        let mut english_misses = 0usize;
        let mut total_pairs = 0usize;

        for _ in 0..cluster_values.len() {
            let selected = cluster_values[rng.index(cluster_values.len())];
            for row in selected {
                if row.target_missed { target_misses += 1; }
                if row.english_missed { english_misses += 1; }
                total_pairs += 1;
            }
        }

        if total_pairs > 0 {
            gaps.push((target_misses as f64 / total_pairs as f64) - (english_misses as f64 / total_pairs as f64));
        }
    }

    gaps.sort_by(|left, right| left.total_cmp(right));
    Some(BootstrapCi {
        lower: percentile(&gaps, tail)?,
        upper: percentile(&gaps, 1.0 - tail)?,
        confidence_level: plan.confidence_level,
        familywise_adjusted,
        comparison_count,
        replicates: plan.replicates,
        seed: plan.seed,
        cluster_count: clusters.len(),
    })
}

fn bootstrap_mean_ci(
    clusters: &BTreeMap<String, Vec<f64>>,
    plan: &BootstrapPlan,
    comparison_count: usize,
) -> Option<BootstrapCi> {
    if clusters.len() < plan.minimum_independent_clusters || clusters.is_empty() {
        return None;
    }
    let cluster_values: Vec<&Vec<f64>> = clusters.values().collect();
    let mut rng = SplitMix64::new(plan.seed);
    let familywise_adjusted = plan.multiple_comparison_correction == "bonferroni";
    let alpha = 1.0 - plan.confidence_level;
    let tail = if familywise_adjusted {
        alpha / (2.0 * comparison_count.max(1) as f64)
    } else {
        alpha / 2.0
    };
    let mut means = Vec::with_capacity(plan.replicates);

    for _ in 0..plan.replicates {
        let mut sum = 0.0;
        let mut count = 0usize;
        for _ in 0..cluster_values.len() {
            let selected = cluster_values[rng.index(cluster_values.len())];
            for value in selected {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            means.push(sum / count as f64);
        }
    }

    means.sort_by(|left, right| left.total_cmp(right));
    Some(BootstrapCi {
        lower: percentile(&means, tail)?,
        upper: percentile(&means, 1.0 - tail)?,
        confidence_level: plan.confidence_level,
        familywise_adjusted,
        comparison_count,
        replicates: plan.replicates,
        seed: plan.seed,
        cluster_count: clusters.len(),
    })
}

fn pair_balanced_accuracy(
    system_id: &str,
    pair: &PairMeta,
    statement_language: &str,
    predictions: &HashMap<(String, String, String, String), Prediction>,
) -> f64 {
    let mut correct = 0usize;
    for kind in ["factual", "hallucinated"] {
        let key = (
            system_id.to_string(),
            pair_key(pair),
            statement_language.to_string(),
            kind.to_string(),
        );
        let prediction = predictions.get(&key).expect("complete prediction matrix validated");
        let is_correct = if kind == "factual" {
            prediction.predicted_label == "factual"
        } else {
            prediction.predicted_label == "hallucinated"
        };
        if is_correct {
            correct += 1;
        }
    }
    correct as f64 / 2.0
}

fn build_system_comparisons(
    plan: &Plan,
    pairs: &BTreeMap<String, PairMeta>,
    predictions: &HashMap<(String, String, String, String), Prediction>,
    target_languages: &[String],
    comparison_count: usize,
) -> Vec<SystemComparison> {
    let candidate = &plan.candidate_system_id;
    let mut comparisons = Vec::new();
    let mut conditions = vec!["english".to_string()];
    conditions.extend(target_languages.iter().map(|language| format!("target:{language}")));

    for baseline in plan.systems.iter().filter(|system| system.system_id != *candidate) {
        for condition in &conditions {
            let selected_pairs: Vec<&PairMeta> = pairs.values()
                .filter(|pair| condition == "english" || condition == &format!("target:{}", pair.language))
                .collect();
            let mut clusters: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            let mut baseline_values = Vec::new();
            let mut candidate_values = Vec::new();

            for pair in &selected_pairs {
                let language = if condition == "english" { "english" } else { "target" };
                let baseline_accuracy = pair_balanced_accuracy(
                    &baseline.system_id, pair, language, predictions
                );
                let candidate_accuracy = pair_balanced_accuracy(
                    candidate, pair, language, predictions
                );
                baseline_values.push(baseline_accuracy);
                candidate_values.push(candidate_accuracy);
                clusters.entry(pair.file_id.clone())
                    .or_default()
                    .push(candidate_accuracy - baseline_accuracy);
            }

            let mean = |values: &[f64]| {
                if values.is_empty() { None } else {
                    Some(values.iter().sum::<f64>() / values.len() as f64)
                }
            };
            let differences: Vec<f64> = clusters.values().flatten().copied().collect();
            let paired_difference = mean(&differences);
            let cluster_count = clusters.len();
            let inference_eligible = plan.phase == "preregistered_held_out"
                && cluster_count >= plan.bootstrap.minimum_independent_clusters;
            let ci = if inference_eligible {
                bootstrap_mean_ci(&clusters, &plan.bootstrap, comparison_count)
            } else {
                None
            };

            comparisons.push(SystemComparison {
                baseline_system_id: baseline.system_id.clone(),
                candidate_system_id: candidate.clone(),
                language: condition.clone(),
                pair_count: selected_pairs.len(),
                cluster_count,
                baseline_pair_balanced_accuracy: mean(&baseline_values),
                candidate_pair_balanced_accuracy: mean(&candidate_values),
                paired_accuracy_difference: paired_difference,
                bootstrap_ci: ci,
                inference_eligible,
                note: if inference_eligible {
                    "Paired candidate-minus-baseline accuracy over factual and hallucinated statements, with source-cluster bootstrap; this does not prove external SOTA superiority.".into()
                } else {
                    "Exploratory paired point estimate only; held-out phase and reviewed independent source clusters are required for a qualifying interval.".into()
                },
            });
        }
    }
    comparisons
}

fn build_system_report(
    system: &SystemPlan,
    plan: &Plan,
    pairs: &BTreeMap<String, PairMeta>,
    predictions: &[Observed],
    prediction_map: &HashMap<(String, String, String, String), Prediction>,
    target_languages: &[String],
    comparison_count: usize,
) -> SystemReport {
    let mut slices = Vec::new();
    let system_rows: Vec<&Observed> = predictions.iter()
        .filter(|row| row.prediction.system_id == system.system_id)
        .collect();

    let mut languages = vec!["english".to_string()];
    languages.extend(target_languages.iter().cloned());
    for language in &languages {
        let rows: Vec<&Observed> = system_rows.iter()
            .copied()
            .filter(|row| {
                if language == "english" {
                    row.prediction.statement_language == "english"
                } else {
                    row.prediction.statement_language == "target" && row.pair.language == *language
                }
            })
            .collect();
        slices.push(metric_slice(&rows, language, None));

        let error_types: BTreeSet<String> = rows.iter().map(|row| row.pair.hallucination_type.clone()).collect();
        for error_type in error_types {
            let typed: Vec<&Observed> = rows.iter().copied()
                .filter(|row| row.pair.hallucination_type == error_type)
                .collect();
            slices.push(metric_slice(&typed, language, Some(error_type)));
        }
    }

    let mut gaps = Vec::new();
    for target_language in target_languages {
        let clusters = paired_misses(&system.system_id, target_language, pairs, prediction_map);
        let (count, english_rate, target_rate, gap) = paired_rates(&clusters);
        let inference_eligible = plan.phase == "preregistered_held_out"
            && clusters.len() >= plan.bootstrap.minimum_independent_clusters;
        let ci = if inference_eligible {
            bootstrap_gap_ci(&clusters, &plan.bootstrap, comparison_count)
        } else {
            None
        };
        gaps.push(LanguageGap {
            target_language: target_language.clone(),
            paired_pair_count: count,
            cluster_count: clusters.len(),
            english_unconditional_missed_hallucination_rate: english_rate,
            target_unconditional_missed_hallucination_rate: target_rate,
            paired_missed_hallucination_gap: gap,
            bootstrap_ci: ci,
            inference_eligible,
            note: if inference_eligible {
                "Held-out paired gap; cluster bootstrap is eligible under the declared plan, but does not prove causal or real-world superiority.".into()
            } else {
                "Exploratory point estimate only; insufficient independent source clusters or non-held-out phase prevents a qualifying inference.".into()
            },
        });
    }

    SystemReport {
        system_id: system.system_id.clone(),
        baseline_family: system.baseline_family.clone(),
        model_id: system.model_id.clone(),
        model_revision: system.model_revision.clone(),
        retrieval_profile_id: system.retrieval_profile_id.clone(),
        prompt_sha256: system.prompt_sha256.clone(),
        is_candidate: system.system_id == plan.candidate_system_id,
        slices,
        language_gaps: gaps,
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let plan_bytes = fs::read(&args.plan)
        .map_err(|error| format!("cannot read plan {}: {error}", args.plan.display()))?;
    let pair_bytes = fs::read(&args.pairs)
        .map_err(|error| format!("cannot read pair manifest {}: {error}", args.pairs.display()))?;
    let result_bytes = fs::read(&args.results)
        .map_err(|error| format!("cannot read results {}: {error}", args.results.display()))?;

    let plan: Plan = serde_json::from_slice(&plan_bytes)
        .map_err(|error| format!("plan is invalid JSON: {error}"))?;
    validate_plan(&plan)?;

    let actual_pair_digest = sha256_hex(&pair_bytes);
    if actual_pair_digest != plan.dataset.pair_manifest_sha256 {
        return Err(format!("pair-manifest SHA-256 mismatch: plan={}, actual={}", plan.dataset.pair_manifest_sha256, actual_pair_digest));
    }

    let pairs: Vec<PairMeta> = read_jsonl(&pair_bytes, "pair manifest")?;
    let pair_map = validate_pairs(&pairs)?;
    let result_rows: Vec<Prediction> = read_jsonl(&result_bytes, "prediction results")?;
    let prediction_map = validate_predictions(&plan, &pair_map, &result_rows)?;

    let mut observed = Vec::with_capacity(result_rows.len());
    for result in &result_rows {
        let pair = pair_map.get(&result_pair_key(result))
            .ok_or_else(|| format!("validated composite pair for result {:?} disappeared", result.result_id))?;
        observed.push(Observed { pair: pair.clone(), prediction: result.clone() });
    }

    let target_languages: Vec<String> = pair_map.values().map(|pair| pair.language.clone())
        .collect::<BTreeSet<_>>().into_iter().collect();
    let cluster_count = pair_map.values().map(|pair| pair.file_id.clone())
        .collect::<BTreeSet<_>>().len();
    if plan.phase == "preregistered_held_out" && cluster_count < plan.bootstrap.minimum_independent_clusters {
        return Err(format!(
            "held-out run has only {cluster_count} source clusters; requires {}",
            plan.bootstrap.minimum_independent_clusters
        ));
    }

    let baseline_count = plan.systems.iter()
        .filter(|system| system.system_id != plan.candidate_system_id)
        .count();
    let comparison_count = (
        plan.systems.len().saturating_mul(target_languages.len())
        + baseline_count.saturating_mul(target_languages.len() + 1)
    ).max(1);
    if plan.phase == "preregistered_held_out"
        && comparison_count > 1
        && plan.bootstrap.multiple_comparison_correction != "bonferroni"
    {
        return Err("held-out language-gap confidence intervals require Bonferroni correction across system/language comparisons".into());
    }

    let system_reports: Vec<SystemReport> = plan.systems.iter()
        .map(|system| build_system_report(
            system,
            &plan,
            &pair_map,
            &observed,
            &prediction_map,
            &target_languages,
            comparison_count,
        ))
        .collect();

    let system_comparisons = build_system_comparisons(
        &plan,
        &pair_map,
        &prediction_map,
        &target_languages,
        comparison_count,
    );

    let report = Report {
        schema_version: "multi-hall-sa-score-report-v0",
        tool_version: TOOL_VERSION,
        plan_id: plan.plan_id.clone(),
        phase: plan.phase.clone(),
        synthetic_only: plan.synthetic_only,
        dataset_id: plan.dataset.dataset_id.clone(),
        dataset_revision: plan.dataset.revision.clone(),
        dataset_license: plan.dataset.license.clone(),
        plan_sha256: sha256_hex(&plan_bytes),
        pair_manifest_sha256: actual_pair_digest,
        results_sha256: sha256_hex(&result_bytes),
        pair_count: pair_map.len(),
        result_count: result_rows.len(),
        cluster_count,
        minimum_independent_clusters: plan.bootstrap.minimum_independent_clusters,
        bootstrap_replicates: plan.bootstrap.replicates,
        bootstrap_seed: plan.bootstrap.seed,
        confidence_level: plan.bootstrap.confidence_level,
        multiple_comparison_correction: plan.bootstrap.multiple_comparison_correction.clone(),
        systems: system_reports,
        system_comparisons,
        nonclaims: vec![
            "The scorer does not execute models or retrieve evidence.",
            "The scorer trusts the pinned pair manifest for statement-kind labels and does not independently establish factual truth.",
            "file_id clustering is only as valid as its source-document semantics and metadata review.",
            "Synthetic controls and development pilots are not held-out benchmark results.",
            "A smaller cross-language gap does not imply improved absolute accuracy.",
            "Submitted probabilities are not necessarily calibrated; calibration metrics only describe supplied values.",
        ],
    };

    let report_value = serde_json::to_value(&report)
        .map_err(|error| format!("cannot serialize report: {error}"))?;
    let report_schema: Value = serde_json::from_str(REPORT_SCHEMA)
        .map_err(|error| format!("bundled report schema is invalid JSON: {error}"))?;
    let meta_validator = jsonschema::meta::validator_for(&report_schema)
        .map_err(|error| format!("cannot compile report meta-schema validator: {error}"))?;
    if !meta_validator.evaluate(&report_schema).flag().valid {
        return Err("bundled score report schema is invalid against its meta-schema".into());
    }
    let validator = draft202012::options()
        .should_validate_formats(true)
        .build(&report_schema)
        .map_err(|error| format!("cannot compile report schema: {error}"))?;
    let validation_errors: Vec<String> = validator.iter_errors(&report_value)
        .map(|error| error.to_string()).collect();
    if !validation_errors.is_empty() {
        return Err(format!("generated report violates score-report schema: {}", validation_errors.join("; ")));
    }

    let bytes = serde_json::to_vec_pretty(&report_value)
        .map_err(|error| format!("cannot serialize JSON report: {error}"))?;
    if let Some(parent) = args.output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create output directory {}: {error}", parent.display()))?;
    }
    fs::write(&args.output, &bytes)
        .map_err(|error| format!("cannot write report {}: {error}", args.output.display()))?;

    println!(
        "Scored {} pairs for {} systems; {} clusters; phase={}; synthetic_only={}; report={}",
        report.pair_count,
        report.systems.len(),
        report.cluster_count,
        report.phase,
        report.synthetic_only,
        args.output.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("score-multihall-sa: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_pair_ids_are_disambiguated_by_source_document() {
        let pair = |file_id: &str| PairMeta {
            schema_version: "multi-hall-sa-pair-v0".into(),
            pair_id: "English_isiZulu_fact0_temporal_shift".into(),
            file_id: file_id.into(),
            fact_index: 0,
            language: "isiZulu".into(),
            hallucination_type: "temporal shift".into(),
        };
        let validated = validate_pairs(&[pair("document-0"), pair("document-1")])
            .expect("same pair_id in two source documents is valid");
        assert_eq!(validated.len(), 2);
        assert_ne!(pair_key(&pair("document-0")), pair_key(&pair("document-1")));
    }

    #[test]
    fn splitmix_is_deterministic() {
        let mut left = SplitMix64::new(42);
        let mut right = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn unconditional_miss_treats_abstention_as_not_detected() {
        let target_missed = true; // target abstained
        let english_missed = false; // English detected
        let clusters = BTreeMap::from([(
            "doc-1".to_string(),
            vec![PairedMiss { target_missed, english_missed }],
        )]);
        let (_, english, target, gap) = paired_rates(&clusters);
        assert_eq!(english, Some(0.0));
        assert_eq!(target, Some(1.0));
        assert_eq!(gap, Some(1.0));
    }

    #[test]
    fn abstained_hallucination_is_counted_as_an_unconditional_miss() {
        let pair = PairMeta {
            schema_version: "multi-hall-sa-pair-v0".into(),
            pair_id: "pair-1".into(),
            file_id: "document-1".into(),
            fact_index: 0,
            language: "isiZulu".into(),
            hallucination_type: "temporal shift".into(),
        };
        let factual = Prediction {
            schema_version: "multi-hall-sa-result-v0".into(),
            result_id: "result-factual".into(),
            plan_id: "plan".into(),
            system_id: "system".into(),
            pair_id: "pair-1".into(),
            file_id: "document-1".into(),
            fact_index: 0,
            language: "isiZulu".into(),
            hallucination_type: "temporal shift".into(),
            statement_language: "target".into(),
            statement_kind: "factual".into(),
            predicted_label: "factual".into(),
            p_hallucinated: Some(0.05),
            latency_ms: Some(10),
            input_tokens: Some(20),
            output_tokens: Some(5),
            retrieval_calls: Some(1),
        };
        let hallucinated = Prediction {
            schema_version: "multi-hall-sa-result-v0".into(),
            result_id: "result-hallucinated".into(),
            plan_id: "plan".into(),
            system_id: "system".into(),
            pair_id: "pair-1".into(),
            file_id: "document-1".into(),
            fact_index: 0,
            language: "isiZulu".into(),
            hallucination_type: "temporal shift".into(),
            statement_language: "target".into(),
            statement_kind: "hallucinated".into(),
            predicted_label: "abstain".into(),
            p_hallucinated: Some(0.50),
            latency_ms: Some(11),
            input_tokens: Some(21),
            output_tokens: Some(6),
            retrieval_calls: Some(1),
        };
        let rows = vec![
            Observed { pair: pair.clone(), prediction: factual },
            Observed { pair, prediction: hallucinated },
        ];
        let metrics = metric_slice(&rows, "isiZulu", None);

        assert_eq!(metrics.abstained, 1);
        assert_eq!(metrics.coverage, Some(0.5));
        assert_eq!(metrics.accuracy_on_covered, Some(1.0));
        assert_eq!(metrics.missed_hallucination_rate_unconditional, Some(1.0));
        assert_eq!(metrics.probability_count, 2);
    }

    #[test]
    fn paired_cluster_bootstrap_is_deterministic_and_keeps_cluster_units() {
        let clusters: BTreeMap<String, Vec<PairedMiss>> = (0..30)
            .map(|index| {
                (
                    format!("document-{index}"),
                    vec![PairedMiss {
                        target_missed: true,
                        english_missed: false,
                    }],
                )
            })
            .collect();
        let plan = BootstrapPlan {
            replicates: 1000,
            seed: 7,
            minimum_independent_clusters: 30,
            confidence_level: 0.95,
            multiple_comparison_correction: "none".into(),
        };
        let first = bootstrap_gap_ci(&clusters, &plan, 1).expect("30 clusters qualify for an exploratory interval");
        let second = bootstrap_gap_ci(&clusters, &plan, 1).expect("same input must score twice");
        assert_eq!(first.lower, 1.0);
        assert_eq!(first.upper, 1.0);
        assert_eq!(first.lower, second.lower);
        assert_eq!(first.upper, second.upper);
        assert_eq!(first.cluster_count, 30);
    }

    #[test]
    fn percentile_returns_expected_order_statistic() {
        let values = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        assert_eq!(percentile(&values, 0.0), Some(-1.0));
        assert_eq!(percentile(&values, 1.0), Some(2.0));
    }
}
