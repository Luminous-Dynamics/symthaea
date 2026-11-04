use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const OUTCOME_SCHEMA: &str = "pitb.support-ablation-outcome.v0";
const PREDICTION_SCHEMA: &str = "pitb.support-antichain-prediction.v0";
const REPORT_SCHEMA: &str = "pitb.support-antichain-report.v0";
const DEFAULT_MAX_VARIABLES: usize = 12;
const HARD_MAX_VARIABLES: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Outcome {
    schema_version: String,
    study_id: String,
    component_id: String,
    target_id: String,
    analysis_endpoint: String,
    target_statement_sha256: String,
    environment_sha256: String,
    toolchain_sha256: String,
    fixed_context_sha256: String,
    budget_sha256: String,
    solver_policy_sha256: String,
    mutable_declarations: Vec<String>,
    available_declarations: Vec<String>,
    local_context_sha256: String,
    outcome: String,
    terminal_reason: String,
    terminal_reason_evidence_sha256: String,
    attempt_finished: bool,
    kernel_passed: bool,
    semantic_gate_passed: bool,
    state_receipt_sha256: String,
    run_receipt_sha256: String,
    checker_receipt_sha256: String,
    semantic_gate_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Prediction {
    schema_version: String,
    group_id: String,
    model_id: String,
    model_version: String,
    abstained: bool,
    claimed_necessary_declarations: Vec<String>,
    proposed_sufficient_route: Option<Vec<String>>,
    route_sufficiency_probability: Option<f64>,
    run_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct Identity {
    study_id: String,
    component_id: String,
    target_id: String,
    analysis_endpoint: String,
    target_statement_sha256: String,
    environment_sha256: String,
    toolchain_sha256: String,
    fixed_context_sha256: String,
    budget_sha256: String,
    solver_policy_sha256: String,
    mutable_declarations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DeclarationRole {
    declaration: String,
    in_all_minimal_support_sets: bool,
    in_some_minimal_support_sets: bool,
    minimal_support_set_frequency: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GroupReport {
    group_id: String,
    study_id: String,
    component_id: String,
    target_id: String,
    analysis_endpoint: String,
    target_statement_sha256: String,
    environment_sha256: String,
    toolchain_sha256: String,
    fixed_context_sha256: String,
    budget_sha256: String,
    solver_policy_sha256: String,
    mutable_declaration_universe: Vec<String>,
    subset_lattice_size: usize,
    verified_success_subset_count: usize,
    non_solve_subset_count: usize,
    full_context_verified_success: bool,
    minimal_sufficient_support_antichain: Vec<Vec<String>>,
    minimal_support_set_count: usize,
    necessary_across_all_observed_minimal_routes: Vec<String>,
    possible_across_observed_minimal_routes: Vec<String>,
    declaration_roles: Vec<DeclarationRole>,
    failed_supersets_of_verified_success_count: usize,
    observed_success_monotonicity: String,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct Confusion {
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    true_negative: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PredictionAnalysis {
    model_id: Option<String>,
    model_version: Option<String>,
    prediction_count: usize,
    evaluated_prediction_count: usize,
    prediction_coverage: f64,
    necessary_declaration_confusion: Confusion,
    necessary_declaration_precision: Option<f64>,
    necessary_declaration_recall: Option<f64>,
    necessary_declaration_f1: Option<f64>,
    route_sufficiency_rate: Option<f64>,
    exact_minimal_route_rate: Option<f64>,
    route_sufficiency_brier_score: Option<f64>,
    route_sufficiency_probability_scored_count: usize,
    selective_route_risk: Option<f64>,
    interpretation_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    outcomes_jsonl_sha256: String,
    predictions_jsonl_sha256: Option<String>,
    group_count: usize,
    exhaustive_group_count: usize,
    total_ablation_rows: usize,
    max_mutable_declarations: usize,
    groups: Vec<GroupReport>,
    prediction_analysis: Option<PredictionAnalysis>,
    receipt_verification_boundary: String,
    overall_interpretation_boundary: String,
}

#[derive(Debug, Clone)]
struct LoadedRows<T> {
    rows: Vec<T>,
    digest: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest_field(value: &str, label: &str) -> Result<(), String> {
    if valid_sha256(value) {
        Ok(())
    } else {
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if !value.trim().is_empty() {
        Ok(())
    } else {
        Err(format!("{label} must be non-empty"))
    }
}

fn validate_sorted_unique(values: &[String], label: &str) -> Result<(), String> {
    if values.iter().any(|x| x.trim().is_empty()) {
        return Err(format!("{label} contains an empty declaration ID"));
    }
    if !values.windows(2).all(|w| w[0] < w[1]) {
        return Err(format!("{label} must be sorted and unique"));
    }
    Ok(())
}

fn identity(row: &Outcome) -> Identity {
    Identity {
        study_id: row.study_id.clone(),
        component_id: row.component_id.clone(),
        target_id: row.target_id.clone(),
        analysis_endpoint: row.analysis_endpoint.clone(),
        target_statement_sha256: row.target_statement_sha256.clone(),
        environment_sha256: row.environment_sha256.clone(),
        toolchain_sha256: row.toolchain_sha256.clone(),
        fixed_context_sha256: row.fixed_context_sha256.clone(),
        budget_sha256: row.budget_sha256.clone(),
        solver_policy_sha256: row.solver_policy_sha256.clone(),
        mutable_declarations: row.mutable_declarations.clone(),
    }
}

fn group_id(identity: &Identity) -> Result<String, String> {
    let value = serde_json::to_value(identity).map_err(|e| format!("serialize group identity: {e}"))?;
    let bytes = serde_json::to_vec(&value).map_err(|e| format!("encode group identity: {e}"))?;
    Ok(sha256(&bytes))
}

fn validate_outcome(row: &Outcome, index: usize, max_variables: usize) -> Result<(), String> {
    let label = format!("outcome row {index}");
    if row.schema_version != OUTCOME_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    for (field, value) in [
        ("study_id", &row.study_id),
        ("component_id", &row.component_id),
        ("target_id", &row.target_id),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    if row.analysis_endpoint != "exact_target_replan" {
        return Err(format!("{label}.analysis_endpoint is unsupported; v0 requires exact_target_replan"));
    }
    for (field, value) in [
        ("target_statement_sha256", &row.target_statement_sha256),
        ("environment_sha256", &row.environment_sha256),
        ("toolchain_sha256", &row.toolchain_sha256),
        ("fixed_context_sha256", &row.fixed_context_sha256),
        ("budget_sha256", &row.budget_sha256),
        ("solver_policy_sha256", &row.solver_policy_sha256),
        ("local_context_sha256", &row.local_context_sha256),
        ("terminal_reason_evidence_sha256", &row.terminal_reason_evidence_sha256),
        ("state_receipt_sha256", &row.state_receipt_sha256),
        ("run_receipt_sha256", &row.run_receipt_sha256),
        ("checker_receipt_sha256", &row.checker_receipt_sha256),
        ("semantic_gate_receipt_sha256", &row.semantic_gate_receipt_sha256),
    ] {
        digest_field(value, &format!("{label}.{field}"))?;
    }
    validate_sorted_unique(&row.mutable_declarations, &format!("{label}.mutable_declarations"))?;
    validate_sorted_unique(&row.available_declarations, &format!("{label}.available_declarations"))?;
    if row.mutable_declarations.len() > max_variables {
        return Err(format!(
            "{label} has {} mutable declarations; configured exhaustive cap is {max_variables}",
            row.mutable_declarations.len()
        ));
    }
    let universe: BTreeSet<&str> = row.mutable_declarations.iter().map(String::as_str).collect();
    if row.available_declarations.iter().any(|x| !universe.contains(x.as_str())) {
        return Err(format!("{label} contains a declaration outside its mutable universe"));
    }
    if !row.attempt_finished {
        return Err(format!("{label} is interrupted or unfinished and cannot enter an exhaustive support lattice"));
    }
    if !matches!(
        row.terminal_reason.as_str(),
        "verified_success" | "budget_exhausted" | "search_exhausted"
    ) {
        return Err(format!(
            "{label} terminal_reason={} is not a completed proof-search outcome; infrastructure failures, invalid targets, checker failures, unresolved runs and interruptions must be excluded and rerun",
            row.terminal_reason
        ));
    }
    if !matches!(row.outcome.as_str(), "verified_success" | "non_solve") {
        return Err(format!("{label} has unresolved or unsupported outcome"));
    }
    let passed = row.kernel_passed && row.semantic_gate_passed;
    if row.outcome == "verified_success" {
        if row.terminal_reason != "verified_success" || !passed {
            return Err(format!("{label} verified_success requires a verified_success terminal reason and both proof gates"));
        }
    } else {
        if !matches!(row.terminal_reason.as_str(), "budget_exhausted" | "search_exhausted") {
            return Err(format!("{label} non_solve must be a completed budget_exhausted or search_exhausted attempt"));
        }
        if row.kernel_passed || row.semantic_gate_passed {
            return Err(format!("{label} non_solve cannot claim a passed kernel or semantic gate"));
        }
    }
    Ok(())
}

fn combinations(universe: &[String], size: usize, start: usize, current: &mut Vec<String>, out: &mut BTreeSet<Vec<String>>) {
    if current.len() == size {
        out.insert(current.clone());
        return;
    }
    let needed = size - current.len();
    if universe.len().saturating_sub(start) < needed {
        return;
    }
    for index in start..universe.len() {
        current.push(universe[index].clone());
        combinations(universe, size, index + 1, current, out);
        current.pop();
    }
}

fn powerset(universe: &[String]) -> BTreeSet<Vec<String>> {
    let mut out = BTreeSet::new();
    for size in 0..=universe.len() {
        combinations(universe, size, 0, &mut Vec::new(), &mut out);
    }
    out
}

fn is_subset(left: &[String], right: &[String]) -> bool {
    left.iter().all(|value| right.binary_search(value).is_ok())
}

fn analyze_group(gid: &str, rows: &[Outcome], max_variables: usize) -> Result<GroupReport, String> {
    let first = rows.first().ok_or("support group cannot be empty")?;
    let id = identity(first);
    let universe = &first.mutable_declarations;
    if universe.len() > max_variables {
        return Err(format!("group {gid} exceeds exhaustive variable cap"));
    }
    let expected = powerset(universe);
    let mut seen = BTreeMap::<Vec<String>, &Outcome>::new();
    let mut contexts = BTreeSet::<String>::new();
    for (index, row) in rows.iter().enumerate() {
        if identity(row) != id {
            return Err(format!("group {gid} mixes target, environment, toolchain, context, budget or solver policy"));
        }
        if !expected.contains(&row.available_declarations) {
            return Err(format!("group {gid} has invalid ablation subset at row {index}"));
        }
        if seen.insert(row.available_declarations.clone(), row).is_some() {
            return Err(format!("group {gid} duplicates subset {:?}", row.available_declarations));
        }
        if !contexts.insert(row.local_context_sha256.clone()) {
            return Err(format!("group {gid} reuses local-context digest across distinct subsets"));
        }
    }
    let seen_subsets: BTreeSet<Vec<String>> = seen.keys().cloned().collect();
    let missing: Vec<_> = expected.difference(&seen_subsets).take(8).cloned().collect();
    if !missing.is_empty() || seen.len() != expected.len() {
        return Err(format!("group {gid} is not exhaustive; missing subsets {missing:?}"));
    }
    let full = universe.clone();
    if seen.get(&full).map_or(true, |row| row.outcome != "verified_success") {
        return Err(format!("group {gid} full-context baseline is not a verified exact-target success"));
    }

    let mut successful: Vec<Vec<String>> = seen
        .iter()
        .filter(|(_, row)| row.outcome == "verified_success")
        .map(|(subset, _)| subset.clone())
        .collect();
    successful.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    let mut minimal: Vec<Vec<String>> = Vec::new();
    for candidate in &successful {
        if !minimal.iter().any(|known| is_subset(known, candidate)) {
            minimal.push(candidate.clone());
        }
    }
    if minimal.is_empty() {
        return Err(format!("group {gid} contains no minimal successful support route"));
    }

    let necessary: BTreeSet<String> = minimal[0].iter().cloned()
        .filter(|d| minimal.iter().all(|route| route.binary_search(d).is_ok()))
        .collect();
    let possible: BTreeSet<String> = minimal.iter().flatten().cloned().collect();
    let failed_supersets = seen
        .iter()
        .filter(|(subset, row)| {
            row.outcome == "non_solve"
                && minimal.iter().any(|route| route.len() < subset.len() && is_subset(route, subset))
        })
        .count();
    let roles = universe.iter().map(|declaration| DeclarationRole {
        declaration: declaration.clone(),
        in_all_minimal_support_sets: necessary.contains(declaration),
        in_some_minimal_support_sets: possible.contains(declaration),
        minimal_support_set_frequency: minimal.iter().filter(|route| route.binary_search(declaration).is_ok()).count() as f64
            / minimal.len() as f64,
    }).collect();
    let success_count = successful.len();
    Ok(GroupReport {
        group_id: gid.to_owned(),
        study_id: id.study_id,
        component_id: id.component_id,
        target_id: id.target_id,
        analysis_endpoint: id.analysis_endpoint,
        target_statement_sha256: id.target_statement_sha256,
        environment_sha256: id.environment_sha256,
        toolchain_sha256: id.toolchain_sha256,
        fixed_context_sha256: id.fixed_context_sha256,
        budget_sha256: id.budget_sha256,
        solver_policy_sha256: id.solver_policy_sha256,
        mutable_declaration_universe: universe.clone(),
        subset_lattice_size: expected.len(),
        verified_success_subset_count: success_count,
        non_solve_subset_count: expected.len() - success_count,
        full_context_verified_success: true,
        minimal_sufficient_support_antichain: minimal.clone(),
        minimal_support_set_count: minimal.len(),
        necessary_across_all_observed_minimal_routes: necessary.into_iter().collect(),
        possible_across_observed_minimal_routes: possible.into_iter().collect(),
        declaration_roles: roles,
        failed_supersets_of_verified_success_count: failed_supersets,
        observed_success_monotonicity: if failed_supersets > 0 {
            "violations_present".into()
        } else {
            "no_violations_on_exhaustive_lattice".into()
        },
        interpretation_boundary: "Finite, policy-, budget-, toolchain-, target- and context-specific minimal sufficient routes; not logical necessity, proof-term support or a global theorem dependency certificate.".into(),
    })
}

fn validate_prediction(prediction: &Prediction, index: usize) -> Result<(), String> {
    let label = format!("prediction row {index}");
    if prediction.schema_version != PREDICTION_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    for (field, value) in [
        ("group_id", &prediction.group_id),
        ("model_id", &prediction.model_id),
        ("model_version", &prediction.model_version),
    ] {
        non_empty(value, &format!("{label}.{field}"))?;
    }
    for (field, value) in [
        ("group_id", &prediction.group_id),
        ("run_receipt_sha256", &prediction.run_receipt_sha256),
    ] {
        digest_field(value, &format!("{label}.{field}"))?;
    }
    validate_sorted_unique(
        &prediction.claimed_necessary_declarations,
        &format!("{label}.claimed_necessary_declarations"),
    )?;
    if let Some(route) = &prediction.proposed_sufficient_route {
        validate_sorted_unique(route, &format!("{label}.proposed_sufficient_route"))?;
    }
    if let Some(probability) = prediction.route_sufficiency_probability {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(format!("{label}.route_sufficiency_probability must lie in [0,1]"));
        }
    }
    if prediction.abstained {
        if !prediction.claimed_necessary_declarations.is_empty()
            || prediction.proposed_sufficient_route.is_some()
            || prediction.route_sufficiency_probability.is_some()
        {
            return Err(format!("{label} abstention cannot include claims or route probability"));
        }
    } else if prediction.proposed_sufficient_route.is_none() {
        return Err(format!("{label} non-abstaining prediction requires a proposed route"));
    }
    Ok(())
}

fn score_predictions(predictions: &[Prediction], groups: &[GroupReport]) -> Result<PredictionAnalysis, String> {
    let group_map: BTreeMap<&str, &GroupReport> = groups.iter().map(|g| (g.group_id.as_str(), g)).collect();
    let mut pred_map = BTreeMap::<String, &Prediction>::new();
    let mut versions = BTreeSet::<(String, String)>::new();
    for (index, prediction) in predictions.iter().enumerate() {
        validate_prediction(prediction, index + 1)?;
        if !group_map.contains_key(prediction.group_id.as_str()) {
            return Err(format!("prediction {} references unknown group", index + 1));
        }
        if pred_map.insert(prediction.group_id.clone(), prediction).is_some() {
            return Err(format!("duplicate prediction for group {}", prediction.group_id));
        }
        versions.insert((prediction.model_id.clone(), prediction.model_version.clone()));
    }
    if versions.len() != 1 {
        return Err("one prediction ledger must contain exactly one model_id/model_version pair".into());
    }
    if pred_map.len() != group_map.len() {
        let missing: Vec<_> = group_map.keys().filter(|id| !pred_map.contains_key(**id)).take(8).collect();
        return Err(format!("prediction ledger must cover every group exactly once; missing {missing:?}"));
    }

    let mut confusion = Confusion { true_positive: 0, false_positive: 0, false_negative: 0, true_negative: 0 };
    let mut evaluated = 0usize;
    let mut route_valid = 0usize;
    let mut exact_route = 0usize;
    let mut brier_sum = 0.0_f64;
    let mut brier_count = 0usize;
    for group in groups {
        let prediction = pred_map[group.group_id.as_str()];
        if prediction.abstained {
            continue;
        }
        evaluated += 1;
        let universe: BTreeSet<&str> = group.mutable_declaration_universe.iter().map(String::as_str).collect();
        if prediction.claimed_necessary_declarations.iter().any(|x| !universe.contains(x.as_str())) {
            return Err(format!("prediction {} claims necessary declaration outside group universe", group.group_id));
        }
        let route = prediction.proposed_sufficient_route.as_ref().ok_or("route missing after validation")?;
        if route.iter().any(|x| !universe.contains(x.as_str())) {
            return Err(format!("prediction {} proposes declaration outside group universe", group.group_id));
        }
        let gold: BTreeSet<&str> = group.necessary_across_all_observed_minimal_routes.iter().map(String::as_str).collect();
        let predicted: BTreeSet<&str> = prediction.claimed_necessary_declarations.iter().map(String::as_str).collect();
        confusion.true_positive += gold.intersection(&predicted).count();
        confusion.false_positive += predicted.difference(&gold).count();
        confusion.false_negative += gold.difference(&predicted).count();
        let predicted_union: BTreeSet<&str> = gold.union(&predicted).copied().collect();
        confusion.true_negative += universe.difference(&predicted_union).count();

        let valid = group.minimal_sufficient_support_antichain.iter().any(|minimal| is_subset(minimal, route));
        let exact = group.minimal_sufficient_support_antichain.iter().any(|minimal| minimal == route);
        route_valid += usize::from(valid);
        exact_route += usize::from(exact);
        if let Some(probability) = prediction.route_sufficiency_probability {
            let observed = if valid { 1.0 } else { 0.0 };
            brier_sum += (probability - observed).powi(2);
            brier_count += 1;
        }
    }
    let precision = ratio(confusion.true_positive, confusion.true_positive + confusion.false_positive);
    let recall = ratio(confusion.true_positive, confusion.true_positive + confusion.false_negative);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    let model = versions.into_iter().next();
    Ok(PredictionAnalysis {
        model_id: model.as_ref().map(|x| x.0.clone()),
        model_version: model.as_ref().map(|x| x.1.clone()),
        prediction_count: predictions.len(),
        evaluated_prediction_count: evaluated,
        prediction_coverage: if predictions.is_empty() { 0.0 } else { evaluated as f64 / predictions.len() as f64 },
        necessary_declaration_confusion: confusion,
        necessary_declaration_precision: precision,
        necessary_declaration_recall: recall,
        necessary_declaration_f1: f1,
        route_sufficiency_rate: ratio(route_valid, evaluated),
        exact_minimal_route_rate: ratio(exact_route, evaluated),
        route_sufficiency_brier_score: if brier_count > 0 { Some(brier_sum / brier_count as f64) } else { None },
        route_sufficiency_probability_scored_count: brier_count,
        selective_route_risk: ratio(evaluated - route_valid, evaluated),
        interpretation_boundary: "Necessity is across a finite exhaustive antichain under the pinned policy; it is not theorem-level logical independence.".into(),
    })
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf, label: &str) -> Result<LoadedRows<T>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {label}: {e}"))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{label} is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let row: T = serde_json::from_str(line)
            .map_err(|e| format!("{label} line {}: {e}", index + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{label} must contain at least one JSONL row"));
    }
    Ok(LoadedRows { rows, digest: sha256(&bytes) })
}

fn analyze(
    outcomes: &[Outcome],
    outcomes_digest: String,
    predictions: Option<&[Prediction]>,
    predictions_digest: Option<String>,
    max_variables: usize,
) -> Result<Report, String> {
    if max_variables == 0 || max_variables > HARD_MAX_VARIABLES {
        return Err(format!("max_variables must be between 1 and {HARD_MAX_VARIABLES}"));
    }
    let mut grouped = BTreeMap::<String, Vec<Outcome>>::new();
    for (index, outcome) in outcomes.iter().enumerate() {
        validate_outcome(outcome, index + 1, max_variables)?;
        let gid = group_id(&identity(outcome))?;
        grouped.entry(gid).or_default().push(outcome.clone());
    }
    let mut reports = Vec::new();
    for (gid, rows) in &grouped {
        reports.push(analyze_group(gid, rows, max_variables)?);
    }
    let prediction_analysis = predictions
        .map(|rows| score_predictions(rows, &reports))
        .transpose()?;
    let max_declarations = reports.iter().map(|g| g.mutable_declaration_universe.len()).max().unwrap_or(0);
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        outcomes_jsonl_sha256: outcomes_digest,
        predictions_jsonl_sha256: predictions_digest,
        group_count: reports.len(),
        exhaustive_group_count: reports.len(),
        total_ablation_rows: outcomes.len(),
        max_mutable_declarations: max_declarations,
        groups: reports,
        prediction_analysis,
        receipt_verification_boundary: "Digest syntax and row binding only. Referenced state/run/checker/semantic receipt bytes must be independently opened and authenticated before benchmark use.".into(),
        overall_interpretation_boundary: "Finite exhaustive support ablation under fixed target, environment, toolchain, fixed context, prover policy and budget; not logical necessity or global mathematical necessity.".into(),
    })
}

fn fake_digest(label: &str) -> String {
    sha256(label.as_bytes())
}

fn build_synthetic_rows(
    study: &str,
    target: &str,
    component: &str,
    universe: &[&str],
    predicate: impl Fn(&BTreeSet<String>) -> bool,
) -> Vec<Outcome> {
    let declarations: Vec<String> = universe.iter().map(|x| x.to_string()).collect();
    let mut rows = Vec::new();
    for subset in powerset(&declarations) {
        let set: BTreeSet<String> = subset.iter().cloned().collect();
        let pass = predicate(&set);
        let available_json = serde_json::to_string(&subset).unwrap();
        rows.push(Outcome {
            schema_version: OUTCOME_SCHEMA.into(),
            study_id: study.into(),
            component_id: component.into(),
            target_id: target.into(),
            analysis_endpoint: "exact_target_replan".into(),
            target_statement_sha256: fake_digest(&format!("{study}:{target}:target")),
            environment_sha256: fake_digest(&format!("{study}:{target}:environment")),
            toolchain_sha256: fake_digest(&format!("{study}:{target}:toolchain")),
            fixed_context_sha256: fake_digest(&format!("{study}:{target}:fixed")),
            budget_sha256: fake_digest(&format!("{study}:{target}:budget")),
            solver_policy_sha256: fake_digest(&format!("{study}:{target}:policy")),
            mutable_declarations: declarations.clone(),
            available_declarations: subset,
            local_context_sha256: fake_digest(&format!("{study}:{target}:{available_json}")),
            outcome: if pass { "verified_success" } else { "non_solve" }.into(),
            terminal_reason: if pass { "verified_success" } else { "budget_exhausted" }.into(),
            terminal_reason_evidence_sha256: fake_digest(&format!("terminal:{target}:{available_json}")),
            attempt_finished: true,
            kernel_passed: pass,
            semantic_gate_passed: pass,
            state_receipt_sha256: fake_digest(&format!("state:{target}:{available_json}")),
            run_receipt_sha256: fake_digest(&format!("run:{target}:{available_json}")),
            checker_receipt_sha256: fake_digest(&format!("checker:{target}:{available_json}")),
            semantic_gate_receipt_sha256: fake_digest(&format!("semantic:{target}:{available_json}")),
        });
    }
    rows
}

fn run_self_tests() -> Result<(), String> {
    let alternative = build_synthetic_rows("self", "alternative", "component-a", &["d", "h", "hEq"], |set| {
        set.contains("h") || set.contains("hEq")
    });
    let conjunctive = build_synthetic_rows("self", "conjunctive", "component-b", &["h1", "h2", "irrelevant"], |set| {
        set.contains("h1") && set.contains("h2")
    });
    let a = analyze_group(&group_id(&identity(&alternative[0]))?, &alternative, 12)?;
    let b = analyze_group(&group_id(&identity(&conjunctive[0]))?, &conjunctive, 12)?;
    if a.minimal_sufficient_support_antichain != vec![vec!["h".to_owned()], vec!["hEq".to_owned()]]
        || !a.necessary_across_all_observed_minimal_routes.is_empty()
        || a.possible_across_observed_minimal_routes != vec!["h".to_owned(), "hEq".to_owned()]
    {
        return Err("alternative-route antichain control failed".into());
    }
    println!("SELF_TEST_PASS alternative routes preserve both minimal supports and empty intersection");
    if b.minimal_sufficient_support_antichain != vec![vec!["h1".to_owned(), "h2".to_owned()]]
        || b.necessary_across_all_observed_minimal_routes != vec!["h1".to_owned(), "h2".to_owned()]
    {
        return Err("conjunctive-support antichain control failed".into());
    }
    println!("SELF_TEST_PASS conjunctive support requires the full minimal set");

    let mut incomplete = alternative.clone();
    incomplete.pop();
    if analyze_group(&group_id(&identity(&alternative[0]))?, &incomplete, 12).is_ok() {
        return Err("incomplete powerset lattice was accepted".into());
    }
    println!("SELF_TEST_PASS incomplete powerset is rejected");

    let mut mixed = alternative.clone();
    mixed[0].budget_sha256 = fake_digest("changed-budget");
    if analyze_group(&group_id(&identity(&alternative[0]))?, &mixed, 12).is_ok() {
        return Err("mixed budget identities were accepted".into());
    }
    println!("SELF_TEST_PASS mixed budget identity is rejected");

    let mut duplicate_context = alternative.clone();
    duplicate_context[1].local_context_sha256 = duplicate_context[0].local_context_sha256.clone();
    if analyze_group(&group_id(&identity(&alternative[0]))?, &duplicate_context, 12).is_ok() {
        return Err("duplicate local context identities were accepted".into());
    }
    println!("SELF_TEST_PASS repeated local-context digest is rejected");

    let mut bad_gate = alternative[0].clone();
    bad_gate.outcome = "verified_success".into();
    bad_gate.kernel_passed = false;
    if validate_outcome(&bad_gate, 999, 12).is_ok() {
        return Err("success without both proof gates was accepted".into());
    }
    println!("SELF_TEST_PASS success requires kernel and semantic gates");

    let mut interrupted = alternative[0].clone();
    interrupted.outcome = "non_solve".into();
    interrupted.terminal_reason = "interrupted".into();
    interrupted.kernel_passed = false;
    interrupted.semantic_gate_passed = false;
    if validate_outcome(&interrupted, 1000, 12).is_ok() {
        return Err("interrupted search was accepted as an exhaustive non-solve".into());
    }
    println!("SELF_TEST_PASS interrupted search cannot be counted as an insufficient support set");

    let mut infrastructure = alternative[0].clone();
    infrastructure.outcome = "non_solve".into();
    infrastructure.terminal_reason = "infrastructure_failure".into();
    infrastructure.kernel_passed = false;
    infrastructure.semantic_gate_passed = false;
    if validate_outcome(&infrastructure, 1001, 12).is_ok() {
        return Err("infrastructure failure was accepted as an insufficient support set".into());
    }
    println!("SELF_TEST_PASS infrastructure failure must be excluded and rerun");

    let mut fake_non_solve = alternative[0].clone();
    fake_non_solve.outcome = "non_solve".into();
    fake_non_solve.terminal_reason = "verified_success".into();
    fake_non_solve.kernel_passed = false;
    fake_non_solve.semantic_gate_passed = false;
    if validate_outcome(&fake_non_solve, 1002, 12).is_ok() {
        return Err("verified-success terminal reason was accepted as a non-solve".into());
    }
    println!("SELF_TEST_PASS terminal reason must agree with outcome and proof gates");

    let mut unfinished = alternative[0].clone();
    unfinished.outcome = "non_solve".into();
    unfinished.terminal_reason = "budget_exhausted".into();
    unfinished.attempt_finished = false;
    unfinished.kernel_passed = false;
    unfinished.semantic_gate_passed = false;
    if validate_outcome(&unfinished, 1003, 12).is_ok() {
        return Err("unfinished budget-limited run was accepted in the support lattice".into());
    }
    println!("SELF_TEST_PASS unfinished attempts cannot contaminate exhaustive support analysis");

    // Explicitly test the configured cap without generating a powerset.
    let mut row = alternative[0].clone();
    row.mutable_declarations = (0..17).map(|i| format!("d{i:02}")).collect();
    row.available_declarations = vec![];
    if validate_outcome(&row, 1000, 12).is_ok() {
        return Err("over-cap exhaustive lattice was accepted".into());
    }
    println!("SELF_TEST_PASS over-cap exhaustive lattice is rejected");

    let groups = vec![a, b];
    let predictions = vec![
        Prediction {
            schema_version: PREDICTION_SCHEMA.into(),
            group_id: groups[0].group_id.clone(),
            model_id: "synthetic-model".into(),
            model_version: "v0".into(),
            abstained: false,
            claimed_necessary_declarations: vec![],
            proposed_sufficient_route: Some(vec!["h".into()]),
            route_sufficiency_probability: Some(0.8),
            run_receipt_sha256: fake_digest("prediction-alt"),
        },
        Prediction {
            schema_version: PREDICTION_SCHEMA.into(),
            group_id: groups[1].group_id.clone(),
            model_id: "synthetic-model".into(),
            model_version: "v0".into(),
            abstained: false,
            claimed_necessary_declarations: vec!["h1".into(), "h2".into()],
            proposed_sufficient_route: Some(vec!["h1".into(), "h2".into()]),
            route_sufficiency_probability: Some(0.8),
            run_receipt_sha256: fake_digest("prediction-conj"),
        },
    ];
    let score = score_predictions(&predictions, &groups)?;
    if score.prediction_coverage != 1.0
        || score.route_sufficiency_rate != Some(1.0)
        || score.exact_minimal_route_rate != Some(1.0)
        || score.route_sufficiency_brier_score.map_or(true, |value| (value - 0.04).abs() > 1e-12)
    {
        return Err("prediction scoring control failed".into());
    }
    println!("SELF_TEST_PASS prediction coverage, route sufficiency, exact route and Brier score");
    Ok(())
}

fn synthetic_report() -> Result<Report, String> {
    let alternative = build_synthetic_rows("self", "alternative", "component-a", &["d", "h", "hEq"], |set| {
        set.contains("h") || set.contains("hEq")
    });
    let conjunctive = build_synthetic_rows("self", "conjunctive", "component-b", &["h1", "h2", "irrelevant"], |set| {
        set.contains("h1") && set.contains("h2")
    });
    let mut outcomes = alternative;
    outcomes.extend(conjunctive);
    let reports = {
        let mut grouped = BTreeMap::<String, Vec<Outcome>>::new();
        for row in &outcomes {
            grouped.entry(group_id(&identity(row))?).or_default().push(row.clone());
        }
        let mut reports = Vec::new();
        for (gid, rows) in grouped {
            reports.push(analyze_group(&gid, &rows, 12)?);
        }
        reports
    };
    let alternative_group_id = reports.iter()
        .find(|group| group.target_id == "alternative")
        .ok_or("synthetic alternative group missing")?
        .group_id.clone();
    let conjunctive_group_id = reports.iter()
        .find(|group| group.target_id == "conjunctive")
        .ok_or("synthetic conjunctive group missing")?
        .group_id.clone();
    let predictions = vec![
        Prediction {
            schema_version: PREDICTION_SCHEMA.into(),
            group_id: alternative_group_id,
            model_id: "synthetic-model".into(),
            model_version: "self-test".into(),
            abstained: false,
            claimed_necessary_declarations: vec![],
            proposed_sufficient_route: Some(vec!["h".into()]),
            route_sufficiency_probability: Some(0.8),
            run_receipt_sha256: fake_digest("synthetic-prediction-alternative"),
        },
        Prediction {
            schema_version: PREDICTION_SCHEMA.into(),
            group_id: conjunctive_group_id,
            model_id: "synthetic-model".into(),
            model_version: "self-test".into(),
            abstained: false,
            claimed_necessary_declarations: vec!["h1".into(), "h2".into()],
            proposed_sufficient_route: Some(vec!["h1".into(), "h2".into()]),
            route_sufficiency_probability: Some(0.8),
            run_receipt_sha256: fake_digest("synthetic-prediction-conjunctive"),
        },
    ];
    let outcome_bytes = serde_json::to_vec(&outcomes).map_err(|e| format!("serialize synthetic outcomes: {e}"))?;
    let prediction_bytes = serde_json::to_vec(&predictions).map_err(|e| format!("serialize synthetic predictions: {e}"))?;
    analyze(
        &outcomes,
        sha256(&outcome_bytes),
        Some(&predictions),
        Some(sha256(&prediction_bytes)),
        12,
    )
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, Option<PathBuf>, usize), String> {
    let mut outcomes = None;
    let mut predictions = None;
    let mut report = None;
    let mut max_variables = DEFAULT_MAX_VARIABLES;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--outcomes" if index + 1 < args.len() => {
                outcomes = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--predictions" if index + 1 < args.len() => {
                predictions = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--report" if index + 1 < args.len() => {
                report = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--max-variables" if index + 1 < args.len() => {
                max_variables = args[index + 1].parse::<usize>()
                    .map_err(|_| "--max-variables must be an integer".to_owned())?;
                index += 2;
            }
            _ => return Err(format!("unknown or incomplete option {}", args[index])),
        }
    }
    let outcomes = outcomes.ok_or("--outcomes is required")?;
    let report = report.ok_or("--report is required")?;
    if max_variables == 0 || max_variables > HARD_MAX_VARIABLES {
        return Err(format!("--max-variables must be between 1 and {HARD_MAX_VARIABLES}"));
    }
    Ok((outcomes, report, predictions, max_variables))
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SUPPORT_ANTICHAIN_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let report = synthetic_report()?;
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic report: {e}"))?;
        println!("PITB_SUPPORT_ANTICHAIN_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    let (outcome_path, report_path, prediction_path, max_variables) = parse_args(&args)?;
    let outcomes: LoadedRows<Outcome> = load_jsonl(&outcome_path, "outcomes JSONL")?;
    let predictions: Option<LoadedRows<Prediction>> = prediction_path.as_ref()
        .map(|path| load_jsonl(path, "predictions JSONL"))
        .transpose()?;
    let report = analyze(
        &outcomes.rows,
        outcomes.digest,
        predictions.as_ref().map(|x| x.rows.as_slice()),
        predictions.as_ref().map(|x| x.digest.clone()),
        max_variables,
    )?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(&report_path, bytes).map_err(|e| format!("write report: {e}"))?;
    println!("PITB_SUPPORT_ANTICHAIN_ANALYSIS_OK");
    println!("groups={} rows={} max_variables={}", report.group_count, report.total_ablation_rows, report.max_mutable_declarations);
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_support_antichain_controls_pass() {
        run_self_tests().expect("all support-antichain controls must pass");
    }

    #[test]
    fn incomplete_prediction_coverage_is_rejected() {
        let rows_a = build_synthetic_rows("self", "a", "ca", &["h", "irrelevant"], |s| s.contains("h"));
        let rows_b = build_synthetic_rows("self", "b", "cb", &["h1", "h2"], |s| s.contains("h1") && s.contains("h2"));
        let a = analyze_group(&group_id(&identity(&rows_a[0])).unwrap(), &rows_a, 12).unwrap();
        let b = analyze_group(&group_id(&identity(&rows_b[0])).unwrap(), &rows_b, 12).unwrap();
        let prediction = Prediction {
            schema_version: PREDICTION_SCHEMA.into(),
            group_id: a.group_id.clone(),
            model_id: "model".into(),
            model_version: "v1".into(),
            abstained: false,
            claimed_necessary_declarations: vec![],
            proposed_sufficient_route: Some(vec!["h".into()]),
            route_sufficiency_probability: Some(0.9),
            run_receipt_sha256: fake_digest("pred"),
        };
        assert!(score_predictions(&[prediction], &[a, b]).is_err());
    }
}
