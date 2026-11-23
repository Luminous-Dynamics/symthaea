use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
const CLAIM_CEILING: &str = "synthetic-budgeted-selection-and-calibration-semantics-only";
const DEFAULT_SCHEMA: &str = "data/schemas/sc_budgeted_selection_v0_1.schema.json";
const DEFAULT_FIXTURE: &str =
    "tools/superconductivity-discovery-harness/fixtures/budgeted_selection_smoke_v0_1.json";

#[derive(Clone, Debug)]
struct Identity {
    candidate_id: String,
    material_id: String,
    reduced_formula: String,
    family_id: String,
    prototype_id: String,
    source_id: String,
    mechanism_track: String,
    target_semantics: String,
}

#[derive(Clone, Debug)]
struct EvaluationCandidate {
    identity: Identity,
    truth_positive: bool,
    cost_units: usize,
    predicted_probability: Option<f64>,
    domain_status: String,
    abstained: bool,
    heuristic_score: f64,
    random_rank: u64,
}

#[derive(Clone, Debug)]
struct Selection {
    indices: Vec<usize>,
    cost_units: usize,
    true_positives: usize,
}

fn text_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{path}.{key} must be a non-empty string"))
}

fn bool_field(value: &Value, key: &str, path: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{path}.{key} must be a boolean"))
}

fn positive_integer(value: &Value, key: &str, path: &str) -> Result<usize, String> {
    let integer = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{path}.{key} must be a positive integer"))?;
    if integer == 0 || integer > usize::MAX as u64 {
        return Err(format!("{path}.{key} must be a positive integer"));
    }
    Ok(integer as usize)
}

fn finite_number(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{path}.{key} must be a finite number"))?;
    if !number.is_finite() {
        return Err(format!("{path}.{key} must be a finite number"));
    }
    Ok(number)
}

fn optional_probability(value: &Value, key: &str, path: &str) -> Result<Option<f64>, String> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::Number(_)) => {
            let probability = finite_number(value, key, path)?;
            if !(0.0..=1.0).contains(&probability) {
                return Err(format!("{path}.{key} must be in [0, 1]"));
            }
            Ok(Some(probability))
        }
        _ => Err(format!("{path}.{key} must be a number or null")),
    }
}

fn optional_text(value: &Value, key: &str, path: &str) -> Result<Option<String>, String> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(Some(text.clone())),
        _ => Err(format!("{path}.{key} must be a non-empty string or null")),
    }
}

fn identity(value: &Value, path: &str) -> Result<Identity, String> {
    Ok(Identity {
        candidate_id: text_field(value, "candidate_id", path)?.to_owned(),
        material_id: text_field(value, "material_id", path)?.to_owned(),
        reduced_formula: text_field(value, "reduced_formula", path)?.to_owned(),
        family_id: text_field(value, "family_id", path)?.to_owned(),
        prototype_id: text_field(value, "prototype_id", path)?.to_owned(),
        source_id: text_field(value, "source_id", path)?.to_owned(),
        mechanism_track: text_field(value, "mechanism_track", path)?.to_owned(),
        target_semantics: text_field(value, "target_semantics", path)?.to_owned(),
    })
}

fn evaluation_candidate(value: &Value, index: usize) -> Result<EvaluationCandidate, String> {
    let path = format!("evaluation_candidates[{index}]");
    let identity = identity(value, &path)?;
    let cost_units = positive_integer(value, "cost_units", &path)?;
    let predicted_probability = optional_probability(value, "predicted_probability", &path)?;
    let domain_status = text_field(value, "domain_status", &path)?.to_owned();
    let abstained = bool_field(value, "abstained", &path)?;
    let abstain_reason = optional_text(value, "abstain_reason", &path)?;
    let heuristic_score = finite_number(value, "heuristic_score", &path)?;
    if heuristic_score < 0.0 {
        return Err(format!("{path}.heuristic_score must be non-negative"));
    }
    let random_rank = value
        .get("random_rank")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{path}.random_rank must be a non-negative integer"))?;
    match domain_status.as_str() {
        "in_domain" => {
            if abstained || predicted_probability.is_none() || abstain_reason.is_some() {
                return Err(format!(
                    "{path}: in-domain candidates require a probability and must not abstain"
                ));
            }
        }
        "out_of_domain" => {
            if !abstained || predicted_probability.is_some() || abstain_reason.is_none() {
                return Err(format!(
                    "{path}: out-of-domain candidates must abstain, omit probability and declare a reason"
                ));
            }
        }
        _ => return Err(format!("{path}.domain_status is unsupported")),
    }
    Ok(EvaluationCandidate {
        identity,
        truth_positive: bool_field(value, "truth_positive", &path)?,
        cost_units,
        predicted_probability,
        domain_status,
        abstained,
        heuristic_score,
        random_rank,
    })
}

fn collect_group<'a, I>(identities: I, field: &str) -> BTreeSet<String>
where
    I: Iterator<Item = &'a Identity>,
{
    identities
        .map(|identity| match field {
            "family_id" => identity.family_id.clone(),
            "prototype_id" => identity.prototype_id.clone(),
            "reduced_formula" => identity.reduced_formula.clone(),
            "source_id" => identity.source_id.clone(),
            _ => unreachable!("validated group field"),
        })
        .collect()
}

fn require_group_disjoint(
    training: &BTreeSet<String>,
    evaluation: &BTreeSet<String>,
    label: &str,
) -> Result<(), String> {
    if let Some(shared) = training.intersection(evaluation).next() {
        return Err(format!(
            "{label} leakage between training and evaluation: {shared}"
        ));
    }
    Ok(())
}

fn identity_index(
    records: &[Value],
    track: &str,
    semantics: &str,
) -> Result<Vec<Identity>, String> {
    let mut result = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let path = format!("training_candidates[{index}]");
        let identity = identity(record, &path)?;
        if identity.mechanism_track != track || identity.target_semantics != semantics {
            return Err(format!(
                "{path}: mechanism/target semantics do not match the benchmark target track"
            ));
        }
        result.push(identity);
    }
    Ok(result)
}

fn validate_unique_identities(
    training: &[Identity],
    evaluation: &[EvaluationCandidate],
) -> Result<(), String> {
    let mut candidate_ids = HashSet::new();
    let mut material_ids = HashSet::new();
    for identity in training
        .iter()
        .chain(evaluation.iter().map(|candidate| &candidate.identity))
    {
        if !candidate_ids.insert(identity.candidate_id.as_str()) {
            return Err(format!("duplicate candidate_id {}", identity.candidate_id));
        }
        if !material_ids.insert(identity.material_id.as_str()) {
            return Err(format!("duplicate material_id {}", identity.material_id));
        }
    }
    let mut ranks = HashSet::new();
    for candidate in evaluation {
        if !ranks.insert(candidate.random_rank) {
            return Err(format!(
                "duplicate random_rank {} in evaluation candidates",
                candidate.random_rank
            ));
        }
    }
    Ok(())
}

fn validate_family_holdouts(
    training: &[Identity],
    evaluation: &[EvaluationCandidate],
) -> Result<(), String> {
    for field in ["family_id", "prototype_id", "reduced_formula", "source_id"] {
        let training_values = collect_group(training.iter(), field);
        let evaluation_values = collect_group(
            evaluation.iter().map(|candidate| &candidate.identity),
            field,
        );
        require_group_disjoint(&training_values, &evaluation_values, field)?;
    }
    Ok(())
}

fn order_by_efficiency<F>(
    candidates: &[EvaluationCandidate],
    score: F,
    require_in_domain: bool,
) -> Vec<usize>
where
    F: Fn(&EvaluationCandidate) -> f64,
{
    let mut indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            !require_in_domain || (candidate.domain_status == "in_domain" && !candidate.abstained)
        })
        .map(|(index, _)| index)
        .collect();
    indices.sort_by(|left, right| {
        let left_candidate = &candidates[*left];
        let right_candidate = &candidates[*right];
        let left_efficiency = score(left_candidate) / left_candidate.cost_units as f64;
        let right_efficiency = score(right_candidate) / right_candidate.cost_units as f64;
        right_efficiency.total_cmp(&left_efficiency).then_with(|| {
            left_candidate
                .identity
                .candidate_id
                .cmp(&right_candidate.identity.candidate_id)
        })
    });
    indices
}

fn select_greedily(
    candidates: &[EvaluationCandidate],
    order: Vec<usize>,
    budget: usize,
) -> Selection {
    let mut selected = Vec::new();
    let mut cost_units = 0usize;
    let mut true_positives = 0usize;
    for index in order {
        let candidate = &candidates[index];
        if candidate.cost_units <= budget.saturating_sub(cost_units) {
            selected.push(index);
            cost_units += candidate.cost_units;
            if candidate.truth_positive {
                true_positives += 1;
            }
        }
    }
    Selection {
        indices: selected,
        cost_units,
        true_positives,
    }
}

fn select_random(candidates: &[EvaluationCandidate], budget: usize) -> Selection {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|left, right| {
        candidates[*left]
            .random_rank
            .cmp(&candidates[*right].random_rank)
            .then_with(|| {
                candidates[*left]
                    .identity
                    .candidate_id
                    .cmp(&candidates[*right].identity.candidate_id)
            })
    });
    select_greedily(candidates, indices, budget)
}

fn select_oracle(candidates: &[EvaluationCandidate], budget: usize) -> Selection {
    let mut states: Vec<Option<Vec<usize>>> = vec![None; budget + 1];
    states[0] = Some(Vec::new());
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.truth_positive || candidate.cost_units > budget {
            continue;
        }
        for used in (candidate.cost_units..=budget).rev() {
            if let Some(previous) = states[used - candidate.cost_units].clone() {
                let mut proposal = previous;
                proposal.push(index);
                let replace = states[used]
                    .as_ref()
                    .is_none_or(|current| proposal.len() > current.len());
                if replace {
                    states[used] = Some(proposal);
                }
            }
        }
    }
    let mut best = Vec::new();
    let mut best_cost = 0usize;
    for (cost, state) in states.into_iter().enumerate() {
        if let Some(indices) = state
            && (indices.len() > best.len() || (indices.len() == best.len() && cost < best_cost))
        {
            best = indices;
            best_cost = cost;
        }
    }
    Selection {
        true_positives: best.len(),
        cost_units: best_cost,
        indices: best,
    }
}

fn selection_value(
    candidates: &[EvaluationCandidate],
    selection: &Selection,
    total_positives: usize,
    budget: usize,
) -> Value {
    let selected_ids: Vec<&str> = selection
        .indices
        .iter()
        .map(|index| candidates[*index].identity.candidate_id.as_str())
        .collect();
    let precision = if selection.indices.is_empty() {
        0.0
    } else {
        selection.true_positives as f64 / selection.indices.len() as f64
    };
    let recall = if total_positives == 0 {
        0.0
    } else {
        selection.true_positives as f64 / total_positives as f64
    };
    let positives_per_cost = if selection.cost_units == 0 {
        0.0
    } else {
        selection.true_positives as f64 / selection.cost_units as f64
    };
    json!({
        "selected_candidate_ids": selected_ids,
        "selected_count": selection.indices.len(),
        "selected_true_positives": selection.true_positives,
        "cost_units_used": selection.cost_units,
        "budget_units": budget,
        "precision_under_budget": precision,
        "recall_under_budget": recall,
        "true_positives_per_cost_unit": positives_per_cost
    })
}

fn calibration_metrics(candidates: &[EvaluationCandidate]) -> Result<Value, String> {
    let scored: Vec<&EvaluationCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.predicted_probability.is_some())
        .collect();
    if scored.is_empty() {
        return Err(
            "calibration cannot be evaluated with zero non-abstained candidates".to_owned(),
        );
    }
    let mut brier_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mut bin_counts = [0usize; 5];
    let mut bin_confidence = [0.0f64; 5];
    let mut bin_frequency = [0.0f64; 5];
    for candidate in &scored {
        let probability = candidate
            .predicted_probability
            .expect("scored candidate has probability");
        let outcome = if candidate.truth_positive { 1.0 } else { 0.0 };
        let error = probability - outcome;
        brier_sum += error * error;
        if (probability >= 0.5) != candidate.truth_positive {
            absolute_error_sum += 1.0;
        }
        let bin = ((probability * 5.0).floor() as usize).min(4);
        bin_counts[bin] += 1;
        bin_confidence[bin] += probability;
        bin_frequency[bin] += outcome;
    }
    let mut expected_calibration_error = 0.0;
    for (index, bin_count) in bin_counts.iter().enumerate() {
        if *bin_count > 0 {
            let count = *bin_count as f64;
            let confidence = bin_confidence[index] / count;
            let frequency = bin_frequency[index] / count;
            expected_calibration_error +=
                count / scored.len() as f64 * (confidence - frequency).abs();
        }
    }
    Ok(json!({
        "scored_candidate_count": scored.len(),
        "abstained_candidate_count": candidates.len() - scored.len(),
        "coverage": scored.len() as f64 / candidates.len() as f64,
        "brier_score": brier_sum / scored.len() as f64,
        "expected_calibration_error_5_bins": expected_calibration_error,
        "selective_classification_error_at_0_5": absolute_error_sum / scored.len() as f64
    }))
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "fixture schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture.get("evidence_class").and_then(Value::as_str) != Some("synthetic") {
        return Err("auditor accepts synthetic fixtures only".to_owned());
    }
    if fixture.get("claim_ceiling").and_then(Value::as_str) != Some(CLAIM_CEILING) {
        return Err("fixture claim ceiling does not match the budgeted selection audit".to_owned());
    }
    let benchmark_id = text_field(&fixture, "benchmark_id", "$")?;
    let target = fixture
        .get("target_track")
        .ok_or_else(|| "target_track is required".to_owned())?;
    let track = text_field(target, "mechanism_track", "target_track")?;
    let semantics = text_field(target, "target_semantics", "target_track")?;
    let budget = positive_integer(&fixture, "budget_units", "$")?;
    let training_values = fixture
        .get("training_candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "training_candidates must be an array".to_owned())?;
    let evaluation_values = fixture
        .get("evaluation_candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "evaluation_candidates must be an array".to_owned())?;
    if training_values.is_empty() || evaluation_values.is_empty() {
        return Err("training and evaluation candidates must both be non-empty".to_owned());
    }
    let training = identity_index(training_values, track, semantics)?;
    let evaluation: Vec<EvaluationCandidate> = evaluation_values
        .iter()
        .enumerate()
        .map(|(index, value)| evaluation_candidate(value, index))
        .collect::<Result<_, _>>()?;
    for candidate in &evaluation {
        if candidate.identity.mechanism_track != track
            || candidate.identity.target_semantics != semantics
        {
            return Err(format!(
                "{}: evaluation target track differs from benchmark target",
                candidate.identity.candidate_id
            ));
        }
    }
    validate_unique_identities(&training, &evaluation)?;
    validate_family_holdouts(&training, &evaluation)?;
    let total_positives = evaluation
        .iter()
        .filter(|candidate| candidate.truth_positive)
        .count();
    if total_positives == 0 {
        return Err("evaluation fixture requires at least one positive outcome".to_owned());
    }
    let model_order = order_by_efficiency(
        &evaluation,
        |candidate| candidate.predicted_probability.unwrap_or(0.0),
        true,
    );
    let heuristic_order =
        order_by_efficiency(&evaluation, |candidate| candidate.heuristic_score, false);
    let model_selection = select_greedily(&evaluation, model_order, budget);
    let heuristic_selection = select_greedily(&evaluation, heuristic_order, budget);
    let random_selection = select_random(&evaluation, budget);
    let oracle_selection = select_oracle(&evaluation, budget);
    let model_precision = if model_selection.indices.is_empty() {
        0.0
    } else {
        model_selection.true_positives as f64 / model_selection.indices.len() as f64
    };
    let random_precision = if random_selection.indices.is_empty() {
        0.0
    } else {
        random_selection.true_positives as f64 / random_selection.indices.len() as f64
    };
    let enrichment_vs_random = if random_precision > 0.0 {
        Some(model_precision / random_precision)
    } else {
        None
    };
    let oracle_regret = if oracle_selection.true_positives == 0 {
        0.0
    } else {
        (oracle_selection
            .true_positives
            .saturating_sub(model_selection.true_positives)) as f64
            / oracle_selection.true_positives as f64
    };
    let calibration = calibration_metrics(&evaluation)?;
    Ok(json!({
        "schema_version": "sc-budgeted-selection-receipt.v0.1",
        "benchmark_id": benchmark_id,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "target_track": {
            "mechanism_track": track,
            "target_semantics": semantics
        },
        "training_candidate_count": training.len(),
        "evaluation_candidate_count": evaluation.len(),
        "training_family_count": collect_group(training.iter(), "family_id").len(),
        "held_out_family_count": collect_group(evaluation.iter().map(|candidate| &candidate.identity), "family_id").len(),
        "evaluation_family_count": collect_group(evaluation.iter().map(|candidate| &candidate.identity), "family_id").len(),
        "evaluation_positive_count": total_positives,
        "budgeted_selection": {
            "model_probability_per_cost": selection_value(&evaluation, &model_selection, total_positives, budget),
            "chemistry_heuristic_per_cost": selection_value(&evaluation, &heuristic_selection, total_positives, budget),
            "deterministic_random": selection_value(&evaluation, &random_selection, total_positives, budget),
            "synthetic_truth_oracle": selection_value(&evaluation, &oracle_selection, total_positives, budget),
            "model_enrichment_vs_random": enrichment_vs_random,
            "model_oracle_regret_fraction": oracle_regret
        },
        "calibration": calibration,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>), String> {
    let mut schema = PathBuf::from(DEFAULT_SCHEMA);
    let mut fixture = PathBuf::from(DEFAULT_FIXTURE);
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for argument {argument}"))?;
        match argument.as_str() {
            "--schema" => schema = PathBuf::from(value),
            "--fixture" => fixture = PathBuf::from(value),
            "--out" => output = Some(PathBuf::from(value)),
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    Ok((schema, fixture, output))
}

fn run() -> Result<(), String> {
    let (schema_path, fixture_path, output_path) = parse_args()?;
    let schema_bytes = fs::read(&schema_path)
        .map_err(|error| format!("cannot read schema {}: {error}", schema_path.display()))?;
    let fixture_bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?;
    let report = audit(&schema_bytes, &fixture_bytes)?;
    let encoded = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize receipt: {error}"))?;
    if let Some(path) = output_path {
        fs::write(&path, &encoded)
            .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
    } else {
        println!(
            "{}",
            String::from_utf8(encoded).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sc_budgeted_selection_audit: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_bytes() -> &'static [u8] {
        include_bytes!("../../../../data/schemas/sc_budgeted_selection_v0_1.schema.json")
    }

    fn fixture_bytes() -> &'static [u8] {
        include_bytes!("../../fixtures/budgeted_selection_smoke_v0_1.json")
    }

    fn fixture_value() -> Value {
        serde_json::from_slice(fixture_bytes()).expect("fixture JSON must parse")
    }

    fn audit_fixture(fixture: &Value) -> Result<Value, String> {
        let bytes = serde_json::to_vec(fixture).expect("fixture serializes");
        audit(schema_bytes(), &bytes)
    }

    fn run_fixture() -> Value {
        audit(schema_bytes(), fixture_bytes()).expect("synthetic selection fixture must pass")
    }

    #[test]
    fn fixture_executes_fixed_budget_comparisons_and_receipt() {
        let receipt = run_fixture();
        assert_eq!(receipt["training_candidate_count"], 4);
        assert_eq!(receipt["evaluation_candidate_count"], 8);
        assert_eq!(receipt["evaluation_family_count"], 4);
        assert_eq!(receipt["evaluation_positive_count"], 4);
        assert_eq!(
            receipt["budgeted_selection"]["model_probability_per_cost"]["selected_true_positives"],
            2
        );
        assert_eq!(
            receipt["budgeted_selection"]["synthetic_truth_oracle"]["selected_true_positives"],
            3
        );
        assert_eq!(
            receipt["budgeted_selection"]["model_oracle_regret_fraction"],
            1.0 / 3.0
        );
        assert_eq!(
            receipt["budgeted_selection"]["model_enrichment_vs_random"],
            2.0
        );
        assert_eq!(
            receipt["claim_ceiling"],
            "synthetic-budgeted-selection-and-calibration-semantics-only"
        );
    }

    #[test]
    fn all_policies_respect_the_same_fixed_cost_budget() {
        let receipt = run_fixture();
        for key in [
            "model_probability_per_cost",
            "chemistry_heuristic_per_cost",
            "deterministic_random",
            "synthetic_truth_oracle",
        ] {
            assert!(
                receipt["budgeted_selection"][key]["cost_units_used"]
                    .as_u64()
                    .expect("cost is numeric")
                    <= 5
            );
        }
    }

    #[test]
    fn out_of_domain_candidates_are_never_selected_by_the_model_policy() {
        let receipt = run_fixture();
        let selected =
            receipt["budgeted_selection"]["model_probability_per_cost"]["selected_candidate_ids"]
                .as_array()
                .expect("selected IDs are an array");
        assert!(!selected.iter().any(|id| id == "EV-006"));
    }

    #[test]
    fn oracle_never_finds_fewer_true_positives_than_a_policy() {
        let receipt = run_fixture();
        let oracle =
            receipt["budgeted_selection"]["synthetic_truth_oracle"]["selected_true_positives"]
                .as_u64()
                .expect("oracle positives are numeric");
        for key in [
            "model_probability_per_cost",
            "chemistry_heuristic_per_cost",
            "deterministic_random",
        ] {
            let selected = receipt["budgeted_selection"][key]["selected_true_positives"]
                .as_u64()
                .expect("selected positives are numeric");
            assert!(oracle >= selected);
        }
    }

    #[test]
    fn calibration_is_reported_only_on_non_abstained_candidates() {
        let receipt = run_fixture();
        assert_eq!(receipt["calibration"]["scored_candidate_count"], 7);
        assert_eq!(receipt["calibration"]["abstained_candidate_count"], 1);
        assert_eq!(receipt["calibration"]["coverage"], 0.875);
    }

    #[test]
    fn rejects_family_overlap() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["family_id"] = json!("TRAIN-FAM-A");
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("family_id leakage")
        );
    }

    #[test]
    fn rejects_prototype_overlap() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["prototype_id"] = json!("PROTO-A");
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("prototype_id leakage")
        );
    }

    #[test]
    fn rejects_source_paper_overlap() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["source_id"] = json!("SRC-TR-A");
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("source_id leakage")
        );
    }

    #[test]
    fn rejects_reduced_formula_overlap() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["reduced_formula"] = json!("A2B");
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("reduced_formula leakage")
        );
    }

    #[test]
    fn rejects_duplicate_material_identity() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["material_id"] = json!("MAT-TR-001");
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("duplicate material_id")
        );
    }

    #[test]
    fn rejects_out_of_domain_candidate_with_prediction() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][5]["predicted_probability"] = json!(0.5);
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("out-of-domain candidates must abstain")
        );
    }

    #[test]
    fn rejects_out_of_domain_candidate_without_abstention() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][5]["abstained"] = json!(false);
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("out-of-domain candidates must abstain")
        );
    }

    #[test]
    fn rejects_in_domain_candidate_that_abstains_without_ood() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["abstained"] = json!(true);
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("in-domain candidates require a probability")
        );
    }

    #[test]
    fn rejects_probability_outside_unit_interval() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["predicted_probability"] = json!(1.2);
        assert!(audit_fixture(&fixture).is_err());
    }

    #[test]
    fn rejects_duplicate_random_rank() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][1]["random_rank"] = json!(5);
        assert!(
            audit_fixture(&fixture)
                .unwrap_err()
                .contains("duplicate random_rank")
        );
    }

    #[test]
    fn rejects_candidate_identity_in_wrong_target_track() {
        let mut fixture = fixture_value();
        fixture["evaluation_candidates"][0]["target_semantics"] = json!("experimental_measured_tc");
        assert!(audit_fixture(&fixture).is_err());
    }

    #[test]
    fn rejects_budget_over_one_thousand() {
        let mut fixture = fixture_value();
        fixture["budget_units"] = json!(1001);
        assert!(audit_fixture(&fixture).is_err());
    }
}
