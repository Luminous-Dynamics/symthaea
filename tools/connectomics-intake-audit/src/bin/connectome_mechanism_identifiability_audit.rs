//! Exact finite-family, finite-sample mechanism identifiability audit.
//!
//! This V1 binary validates synthetic known-answer inputs. It distinguishes
//! scientific-query identification from mechanism-class identification and
//! refuses to certify any result outside the declared nuisance envelope.

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-mechanism-identifiability-audit",
    version,
    about = "Evaluate finite-sample mechanism identifiability under declared nuisance profiles"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    run_id: String,
    claim_scope: String,
    query: Query,
    mechanisms: Vec<Mechanism>,
    nuisance_profiles: Vec<NuisanceProfile>,
    interventions: Vec<Intervention>,
    predictions: Vec<Prediction>,
    replicate_design: ReplicateDesign,
    alpha: f64,
    known_answer: KnownAnswer,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    query_spec_sha256: String,
    decision_rule_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mechanism {
    mechanism_id: String,
    mechanism_class: String,
    query_decision: bool,
    prior_probability: f64,
    source_evidence_sha256: String,
    model_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NuisanceProfile {
    nuisance_profile_id: String,
    profile_sha256: String,
    description: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intervention {
    intervention_id: String,
    action_kind: String,
    cost_units: f64,
    feasibility_status: String,
    feasibility_receipt_sha256: String,
    safety_profile_sha256: String,
    outcome_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prediction {
    intervention_id: String,
    mechanism_id: String,
    nuisance_profile_id: String,
    observation_model_sha256: String,
    outcomes: Vec<Outcome>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Outcome {
    outcome_id: String,
    probability: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplicateDesign {
    independent_unit: String,
    replicate_count: usize,
    independence_scope: String,
    split_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct KnownAnswer {
    expected_selected_intervention_id: String,
    expected_action_statuses: BTreeMap<String, ExpectedActionStatus>,
    expected_pairwise_risk: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedActionStatus {
    query_identified: bool,
    mechanism_identified: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PairwiseResult {
    mechanism_a: String,
    mechanism_b: String,
    different_query_decision: bool,
    different_mechanism_class: bool,
    tv_by_nuisance_profile: BTreeMap<String, f64>,
    robust_minimum_tv: f64,
    testing_error_by_nuisance_profile: BTreeMap<String, f64>,
    worst_case_testing_error: f64,
    pair_identified_at_budget: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ActionReport {
    feasibility_status: String,
    cost_units: f64,
    query_identified: bool,
    mechanism_identified: bool,
    status: String,
    pairwise_results: Vec<PairwiseResult>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: u32,
    tool: String,
    run_id: String,
    claim_scope: String,
    input_sha256: String,
    replicate_count: usize,
    alpha: f64,
    mechanism_count: usize,
    nuisance_profile_count: usize,
    intervention_count: usize,
    prediction_cell_count: usize,
    selected_intervention_id: Option<String>,
    action_reports: BTreeMap<String, ActionReport>,
    pairwise_worst_case_testing_error: BTreeMap<String, f64>,
    query_vs_mechanism_semantics: String,
    nuisance_scope: String,
    biological_mechanism: String,
    status: String,
}

type PredictionKey = (String, String, String);
type PredictionMap = BTreeMap<PredictionKey, Vec<f64>>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha(value: &str, label: &str) -> Result<(), String> {
    if valid_sha(value) {
        Ok(())
    } else {
        Err(format!("{label} must be 64 lowercase hexadecimal characters"))
    }
}

fn require_unique<I, S>(values: I, label: &str) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut seen = BTreeSet::new();
    for value in values {
        let value = value.into();
        if value.trim().is_empty() || !seen.insert(value.clone()) {
            return Err(format!("{label} contains an empty or duplicate identity: {value}"));
        }
    }
    Ok(())
}

fn pair_key(action: &str, left: &str, right: &str) -> String {
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!("{action}|{first}|{second}")
}

fn product_overlap_sum(
    index: usize,
    n: usize,
    left: &[f64],
    right: &[f64],
    p_left: f64,
    p_right: f64,
    overlap: &mut f64,
) {
    if index == n {
        *overlap += p_left.min(p_right);
        return;
    }
    for outcome_index in 0..left.len() {
        product_overlap_sum(
            index + 1,
            n,
            left,
            right,
            p_left * left[outcome_index],
            p_right * right[outcome_index],
            overlap,
        );
    }
}

fn product_testing_error(left: &[f64], right: &[f64], n: usize) -> f64 {
    let mut overlap = 0.0;
    product_overlap_sum(0, n, left, right, 1.0, 1.0, &mut overlap);
    0.5 * overlap
}

fn total_variation(left: &[f64], right: &[f64]) -> f64 {
    0.5 * left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
}

fn validate_input(input: &Input) -> Result<(PredictionMap, BTreeMap<String, &Intervention>), String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if input.claim_scope != "synthetic_known_truth" {
        return Err("V1 executable audit accepts synthetic_known_truth only".into());
    }
    if input.run_id.trim().is_empty() || input.query.query_id.trim().is_empty() {
        return Err("run_id and query_id must be non-empty".into());
    }
    require_sha(&input.query.query_spec_sha256, "query specification digest")?;
    require_sha(&input.query.decision_rule_sha256, "query decision-rule digest")?;
    require_sha(&input.replicate_design.split_manifest_sha256, "split-manifest digest")?;
    if input.replicate_design.independence_scope != "SYNTHETIC_IID_KNOWN_TRUTH"
        || input.replicate_design.independent_unit != "synthetic_replicate"
    {
        return Err("synthetic IID independent-replicate design is required".into());
    }
    if !(1..=8).contains(&input.replicate_design.replicate_count) {
        return Err("replicate_count must be within the exact V1 limit 1..=8".into());
    }
    if !input.alpha.is_finite() || input.alpha <= 0.0 || input.alpha >= 0.5 {
        return Err("alpha must be finite and between zero and one half".into());
    }
    if input.mechanisms.len() < 2 || input.nuisance_profiles.is_empty() {
        return Err("at least two mechanisms and one nuisance profile are required".into());
    }

    require_unique(
        input.mechanisms.iter().map(|row| row.mechanism_id.clone()),
        "mechanism IDs",
    )?;
    require_unique(
        input
            .nuisance_profiles
            .iter()
            .map(|row| row.nuisance_profile_id.clone()),
        "nuisance-profile IDs",
    )?;
    require_unique(
        input
            .interventions
            .iter()
            .map(|row| row.intervention_id.clone()),
        "intervention IDs",
    )?;
    if input.interventions.is_empty() {
        return Err("at least one intervention is required".into());
    }

    let prior_sum = input
        .mechanisms
        .iter()
        .map(|row| row.prior_probability)
        .sum::<f64>();
    if (prior_sum - 1.0).abs() > 1e-9 {
        return Err("mechanism prior probabilities must sum to one".into());
    }

    let allowed_classes = [
        "DIRECT_SYNAPTIC",
        "POLYSYNAPTIC",
        "EXTRASYNAPTIC_LATENT",
        "SHARED_INPUT",
        "CELL_INTRINSIC_STATE",
        "OBSERVATION_ARTIFACT",
        "MIXTURE_UNRESOLVED",
        "OTHER",
    ];
    for mechanism in &input.mechanisms {
        if !allowed_classes.contains(&mechanism.mechanism_class.as_str()) {
            return Err(format!(
                "mechanism {} has unsupported class {}",
                mechanism.mechanism_id, mechanism.mechanism_class
            ));
        }
        if !mechanism.prior_probability.is_finite()
            || mechanism.prior_probability < 0.0
            || mechanism.prior_probability > 1.0
        {
            return Err(format!(
                "mechanism {} has invalid prior probability",
                mechanism.mechanism_id
            ));
        }
        require_sha(&mechanism.source_evidence_sha256, "mechanism source digest")?;
        require_sha(&mechanism.model_sha256, "mechanism model digest")?;
    }
    for profile in &input.nuisance_profiles {
        require_sha(&profile.profile_sha256, "nuisance profile digest")?;
        if profile.description.trim().is_empty() {
            return Err(format!(
                "nuisance profile {} has an empty description",
                profile.nuisance_profile_id
            ));
        }
    }

    let mut intervention_map = BTreeMap::new();
    for action in &input.interventions {
        if !action.cost_units.is_finite() || action.cost_units <= 0.0 {
            return Err(format!(
                "intervention {} has invalid cost",
                action.intervention_id
            ));
        }
        if !matches!(
            action.feasibility_status.as_str(),
            "FEASIBLE" | "DISALLOWED" | "INACCESSIBLE" | "NOT_ASSESSED"
        ) {
            return Err(format!(
                "intervention {} has unknown feasibility status",
                action.intervention_id
            ));
        }
        require_sha(
            &action.feasibility_receipt_sha256,
            "intervention feasibility receipt",
        )?;
        require_sha(
            &action.safety_profile_sha256,
            "intervention safety-profile digest",
        )?;
        if action.outcome_ids.len() < 2 {
            return Err(format!(
                "intervention {} requires at least two outcomes",
                action.intervention_id
            ));
        }
        require_unique(action.outcome_ids.clone(), "intervention outcome IDs")?;
        intervention_map.insert(action.intervention_id.as_str().to_string(), action);
    }

    let expected_cells = input.interventions.len()
        * input.mechanisms.len()
        * input.nuisance_profiles.len();
    if input.predictions.len() != expected_cells {
        return Err(format!(
            "prediction matrix has {} cells; expected complete Cartesian product of {expected_cells}",
            input.predictions.len()
        ));
    }

    let mechanism_ids = input
        .mechanisms
        .iter()
        .map(|row| row.mechanism_id.as_str())
        .collect::<BTreeSet<_>>();
    let nuisance_ids = input
        .nuisance_profiles
        .iter()
        .map(|row| row.nuisance_profile_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut predictions = BTreeMap::new();
    for prediction in &input.predictions {
        let key = (
            prediction.intervention_id.clone(),
            prediction.mechanism_id.clone(),
            prediction.nuisance_profile_id.clone(),
        );
        if !intervention_map.contains_key(&prediction.intervention_id) {
            return Err(format!(
                "prediction references unknown intervention {}",
                prediction.intervention_id
            ));
        }
        if !mechanism_ids.contains(prediction.mechanism_id.as_str()) {
            return Err(format!(
                "prediction references unknown mechanism {}",
                prediction.mechanism_id
            ));
        }
        if !nuisance_ids.contains(prediction.nuisance_profile_id.as_str()) {
            return Err(format!(
                "prediction references unknown nuisance profile {}",
                prediction.nuisance_profile_id
            ));
        }
        require_sha(
            &prediction.observation_model_sha256,
            "observation-model digest",
        )?;
        let action = intervention_map[&prediction.intervention_id];
        let outcome_ids = prediction
            .outcomes
            .iter()
            .map(|row| row.outcome_id.clone())
            .collect::<Vec<_>>();
        require_unique(outcome_ids.clone(), "predicted outcome IDs")?;
        let declared = action.outcome_ids.iter().cloned().collect::<BTreeSet<_>>();
        let observed = outcome_ids.iter().cloned().collect::<BTreeSet<_>>();
        if declared != observed {
            return Err(format!(
                "prediction outcome vocabulary differs for intervention {}",
                prediction.intervention_id
            ));
        }
        let by_id = prediction
            .outcomes
            .iter()
            .map(|row| (row.outcome_id.as_str(), row.probability))
            .collect::<BTreeMap<_, _>>();
        let vector = action
            .outcome_ids
            .iter()
            .map(|id| by_id[id.as_str()])
            .collect::<Vec<_>>();
        if vector
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        {
            return Err("prediction contains an invalid probability".into());
        }
        if (vector.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            return Err(format!(
                "prediction probabilities do not sum to one for intervention {} mechanism {} profile {}",
                prediction.intervention_id,
                prediction.mechanism_id,
                prediction.nuisance_profile_id
            ));
        }
        if predictions.insert(key.clone(), vector).is_some() {
            return Err(format!("duplicate prediction cell: {key:?}"));
        }
    }
    if predictions.len() != expected_cells {
        return Err("prediction matrix is missing one or more Cartesian cells".into());
    }

    Ok((predictions, intervention_map))
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let (predictions, intervention_map) = validate_input(input)?;
    let replicate_count = input.replicate_design.replicate_count;
    let mut action_reports = BTreeMap::new();
    let mut pairwise_risk_map = BTreeMap::new();

    for action in &input.interventions {
        let mut pairwise_results = Vec::new();
        for index in 0..input.mechanisms.len() {
            for right_index in (index + 1)..input.mechanisms.len() {
                let left_mechanism = &input.mechanisms[index];
                let right_mechanism = &input.mechanisms[right_index];
                let mut tv_by_profile = BTreeMap::new();
                let mut error_by_profile = BTreeMap::new();
                for profile in &input.nuisance_profiles {
                    let left = &predictions[&(
                        action.intervention_id.clone(),
                        left_mechanism.mechanism_id.clone(),
                        profile.nuisance_profile_id.clone(),
                    )];
                    let right = &predictions[&(
                        action.intervention_id.clone(),
                        right_mechanism.mechanism_id.clone(),
                        profile.nuisance_profile_id.clone(),
                    )];
                    tv_by_profile.insert(
                        profile.nuisance_profile_id.clone(),
                        total_variation(left, right),
                    );
                    error_by_profile.insert(
                        profile.nuisance_profile_id.clone(),
                        product_testing_error(left, right, replicate_count),
                    );
                }
                let robust_minimum_tv = tv_by_profile
                    .values()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let worst_case_testing_error = error_by_profile
                    .values()
                    .copied()
                    .fold(0.0, f64::max);
                let key = pair_key(
                    &action.intervention_id,
                    &left_mechanism.mechanism_id,
                    &right_mechanism.mechanism_id,
                );
                pairwise_risk_map.insert(key, worst_case_testing_error);
                pairwise_results.push(PairwiseResult {
                    mechanism_a: left_mechanism.mechanism_id.clone(),
                    mechanism_b: right_mechanism.mechanism_id.clone(),
                    different_query_decision: left_mechanism.query_decision
                        != right_mechanism.query_decision,
                    different_mechanism_class: left_mechanism.mechanism_class
                        != right_mechanism.mechanism_class,
                    tv_by_nuisance_profile: tv_by_profile,
                    robust_minimum_tv,
                    testing_error_by_nuisance_profile: error_by_profile,
                    worst_case_testing_error,
                    pair_identified_at_budget: worst_case_testing_error <= input.alpha,
                });
            }
        }
        let query_pairs = pairwise_results
            .iter()
            .filter(|row| row.different_query_decision)
            .collect::<Vec<_>>();
        let mechanism_pairs = pairwise_results
            .iter()
            .filter(|row| row.different_mechanism_class)
            .collect::<Vec<_>>();
        let query_identified = !query_pairs.is_empty()
            && query_pairs.iter().all(|row| row.pair_identified_at_budget);
        let mechanism_identified = !mechanism_pairs.is_empty()
            && mechanism_pairs.iter().all(|row| row.pair_identified_at_budget);
        let status = if query_identified && mechanism_identified {
            "QUERY_AND_MECHANISM_IDENTIFIED"
        } else if query_identified {
            "QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED"
        } else {
            "QUERY_UNIDENTIFIED"
        };
        action_reports.insert(
            action.intervention_id.clone(),
            ActionReport {
                feasibility_status: action.feasibility_status.clone(),
                cost_units: action.cost_units,
                query_identified,
                mechanism_identified,
                status: status.into(),
                pairwise_results,
            },
        );
    }

    let mut feasible_joint = input
        .interventions
        .iter()
        .filter(|action| {
            let report = &action_reports[&action.intervention_id];
            report.feasibility_status == "FEASIBLE"
                && report.query_identified
                && report.mechanism_identified
        })
        .collect::<Vec<_>>();
    feasible_joint.sort_by(|left, right| {
        left.cost_units
            .total_cmp(&right.cost_units)
            .then_with(|| left.intervention_id.cmp(&right.intervention_id))
    });
    let selected = feasible_joint
        .first()
        .map(|action| action.intervention_id.clone());

    let expected_selected = &input.known_answer.expected_selected_intervention_id;
    if selected.as_deref() != Some(expected_selected.as_str()) {
        return Err(format!(
            "minimum-cost feasible joint-identification action mismatch: expected {expected_selected}, got {selected:?}"
        ));
    }
    for (action_id, expected) in &input.known_answer.expected_action_statuses {
        let actual = action_reports
            .get(action_id)
            .ok_or_else(|| format!("known answer references missing action {action_id}"))?;
        if actual.query_identified != expected.query_identified
            || actual.mechanism_identified != expected.mechanism_identified
        {
            return Err(format!("known-answer status mismatch for action {action_id}"));
        }
    }
    for (key, expected) in &input.known_answer.expected_pairwise_risk {
        let actual = pairwise_risk_map
            .get(key)
            .ok_or_else(|| format!("known answer references missing pairwise result {key}"))?;
        if (actual - expected).abs() > 1e-8 {
            return Err(format!("known-answer pairwise risk mismatch for {key}: {actual} != {expected}"));
        }
    }
    if intervention_map[expected_selected].feasibility_status != "FEASIBLE" {
        return Err("selected intervention is not feasible".into());
    }

    Ok(Report {
        schema_version: 1,
        tool: "connectome-mechanism-identifiability-audit".into(),
        run_id: input.run_id.clone(),
        claim_scope: input.claim_scope.clone(),
        input_sha256: sha256(input_bytes),
        replicate_count,
        alpha: input.alpha,
        mechanism_count: input.mechanisms.len(),
        nuisance_profile_count: input.nuisance_profiles.len(),
        intervention_count: input.interventions.len(),
        prediction_cell_count: predictions.len(),
        selected_intervention_id: selected,
        action_reports,
        pairwise_worst_case_testing_error: pairwise_risk_map,
        query_vs_mechanism_semantics: "SEPARATE_GATES".into(),
        nuisance_scope: "DECLARED_COMMON_PROFILES_ONLY".into(),
        biological_mechanism: "NOT_ESTABLISHED_BY_SYNTHETIC_ORACLE".into(),
        status: "SYNTHETIC_MECHANISM_IDENTIFIABILITY_CONTRACT_VALIDATED".into(),
    })
}

fn run(bytes: &[u8]) -> Result<Report, String> {
    let input: Input = serde_json::from_slice(bytes)
        .map_err(|error| format!("input JSON decoding failed: {error}"))?;
    evaluate(&input, bytes)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", args.input.display());
            return ExitCode::FAILURE;
        }
    };
    let report = match run(&input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("mechanism-identifiability audit rejected input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(value) => format!("{value}\n"),
        Err(error) => {
            eprintln!("failed to serialize mechanism audit report: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, rendered) {
            eprintln!("failed to write {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_MECHANISM_IDENTIFIABILITY.example.json"
        ))
        .unwrap()
    }

    fn run_value(value: Value) -> Result<Report, String> {
        let bytes = serde_json::to_vec(&value).unwrap();
        run(&bytes)
    }

    #[test]
    fn query_can_be_identified_while_mechanism_remains_unidentified() {
        let report = run_value(fixture()).unwrap();
        let action = &report.action_reports["query-probe"];
        assert!(action.query_identified);
        assert!(!action.mechanism_identified);
        assert_eq!(
            action.status,
            "QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED"
        );
    }

    #[test]
    fn nuisance_profile_blocks_pathway_selective_mechanism_claim() {
        let report = run_value(fixture()).unwrap();
        let action = &report.action_reports["selective-pathway"];
        assert!(!action.query_identified);
        assert!(!action.mechanism_identified);
        assert_eq!(
            action.pairwise_results[0].robust_minimum_tv,
            0.0
        );
    }

    #[test]
    fn orthogonal_action_identifies_query_and_mechanism() {
        let report = run_value(fixture()).unwrap();
        assert_eq!(
            report.selected_intervention_id.as_deref(),
            Some("orthogonal-joint")
        );
        assert!(report.action_reports["orthogonal-joint"].query_identified);
        assert!(report.action_reports["orthogonal-joint"].mechanism_identified);
    }

    #[test]
    fn disallowed_perfect_action_is_never_selected() {
        let report = run_value(fixture()).unwrap();
        assert_eq!(
            report.action_reports["infeasible-perfect"].status,
            "QUERY_AND_MECHANISM_IDENTIFIED"
        );
        assert_ne!(
            report.selected_intervention_id.as_deref(),
            Some("infeasible-perfect")
        );
    }

    #[test]
    fn missing_prediction_cell_is_rejected() {
        let mut value = fixture();
        value["predictions"].as_array_mut().unwrap().pop();
        assert!(run_value(value).unwrap_err().contains("complete Cartesian product"));
    }

    #[test]
    fn duplicate_mechanism_identity_is_rejected() {
        let mut value = fixture();
        value["mechanisms"][1]["mechanism_id"] =
            value["mechanisms"][0]["mechanism_id"].clone();
        assert!(run_value(value).unwrap_err().contains("duplicate identity"));
    }

    #[test]
    fn underpowered_single_replicate_does_not_retain_known_answer() {
        let mut value = fixture();
        value["replicate_design"]["replicate_count"] = json!(1);
        assert!(run_value(value).is_err());
    }

    #[test]
    fn malformed_probability_is_rejected() {
        let mut value = fixture();
        value["predictions"][0]["outcomes"][0]["probability"] = json!(1.5);
        assert!(run_value(value).unwrap_err().contains("invalid probability"));
    }

    #[test]
    fn probabilities_must_sum_to_one() {
        let mut value = fixture();
        value["predictions"][0]["outcomes"][0]["probability"] = json!(0.5);
        assert!(run_value(value).unwrap_err().contains("do not sum to one"));
    }

    #[test]
    fn infeasible_action_cannot_be_promoted_after_status_mutation() {
        let mut value = fixture();
        value["interventions"][4]["feasibility_status"] = json!("FEASIBLE");
        assert!(run_value(value).is_err());
    }

    #[test]
    fn input_report_is_bound_to_exact_bytes() {
        let value = fixture();
        let bytes = serde_json::to_vec(&value).unwrap();
        let report = run(&bytes).unwrap();
        assert_eq!(report.input_sha256, sha256(&bytes));
    }
}
