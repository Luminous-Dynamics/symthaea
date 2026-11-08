//! Offline semantic validator for the ansatz-free structural-discovery contract.
//!
//! This crate validates synthetic protocol fixtures only. It does not infer
//! equations, execute a world, evaluate a scientific model, or establish
//! empirical/physical truth. The checked-in fixtures are public and are not a
//! blind benchmark or an Unsatz reproduction.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "open-world-ansatz-structural-discovery-fixtures-v0";
const REQUIRED_STATUS: &str =
    "SYNTHETIC_CONTRACT_FIXTURES_NOT_UNSATZ_REPRODUCTION_OR_AGENT_RESULTS";
const REQUIRED_ISSUE: &str = "#11042";
const EXPECTED_FIXTURE_COUNT: usize = 10;

const ALLOWED_AGENT_FIELDS: &[&str] = &[
    "task_objective",
    "units",
    "measurement_metadata",
    "partial_measurement_metadata",
    "observations_after_acquisition",
    "legal_experiments",
    "remaining_budget",
    "supplied_model_structure",
    "source_world_model_artifact",
];

const FORBIDDEN_AGENT_FIELDS: &[&str] = &[
    "world_seed_id",
    "evaluator_structure",
    "latent_truth",
    "latent_state_truth",
    "hidden_structure",
    "hidden_truth",
    "confirmation_outcomes",
    "oracle_seed",
    "evaluator_equivalence_class",
    "ground_truth",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationContract {
    pub structural_score_unit: String,
    pub coefficient_score_separate: bool,
    pub non_identifiable_policy: String,
    pub hidden_truth_boundary: String,
    pub cost_boundary: String,
    pub task_claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixturePack {
    pub schema_version: String,
    pub status: String,
    pub source_issue: String,
    pub purpose: String,
    pub evaluation_contract: EvaluationContract,
    pub fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub fixture_id: String,
    pub tier: String,
    pub world_id: String,
    pub paired_control_group: Option<String>,
    pub world_seed_id: String,
    pub discovery_split_id: String,
    pub calibration_split_id: String,
    pub confirmation_split_id: String,
    pub structure_input: String,
    pub measurement_channel: String,
    pub identifiability: String,
    pub hidden_state_present: bool,
    pub intervention_access: bool,
    pub long_horizon_holdout: bool,
    pub transfer_relation: String,
    pub experiment_budget: u64,
    pub measurement_budget: u64,
    pub agent_visible_fields: Vec<String>,
    pub expected_disposition: String,
    pub required_evidence: Vec<String>,
    pub prohibited_inference: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub valid: bool,
    pub fixture_count: usize,
    pub paired_control_groups: usize,
    pub tier_counts: [usize; 6],
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

const CANONICAL_CASES: [(&str, &str, &str, &str, &str, &str); EXPECTED_FIXTURE_COUNT] = [
    (
        "ANS-A0-01",
        "A0",
        "supplied",
        "ideal",
        "identifiable",
        "fit_parameters_given_structure",
    ),
    (
        "ANS-A2-01",
        "A2",
        "withheld",
        "ideal",
        "identifiable",
        "propose_structural_model",
    ),
    (
        "ANS-A2-02",
        "A2",
        "withheld",
        "ideal",
        "identifiable",
        "propose_latent_state_and_interaction_model",
    ),
    (
        "ANS-A3-01",
        "A3",
        "withheld",
        "distorted",
        "identifiable",
        "jointly_identify_measurement_and_state_model",
    ),
    (
        "ANS-A3-02",
        "A3",
        "withheld",
        "uninformative",
        "not_identifiable",
        "abstain_request_informative_measurement",
    ),
    (
        "ANS-A2-03",
        "A2",
        "withheld",
        "ideal",
        "equivalence_class",
        "report_observational_equivalence_class",
    ),
    (
        "ANS-A4-01",
        "A4",
        "withheld",
        "ideal",
        "identifiable",
        "discriminate_confounding_with_intervention",
    ),
    (
        "ANS-A4-02",
        "A4",
        "withheld",
        "ideal",
        "identifiable",
        "prefer_intervention_and_long_horizon_generalization",
    ),
    (
        "ANS-A5-01",
        "A5",
        "withheld",
        "ideal",
        "identifiable",
        "test_and_adapt_related_mechanism_transfer",
    ),
    (
        "ANS-A5-02",
        "A5",
        "withheld",
        "ideal",
        "identifiable",
        "reject_negative_transfer_and_reinfer",
    ),
];

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn tier_index(tier: &str) -> Option<usize> {
    match tier {
        "A0" => Some(0),
        "A1" => Some(1),
        "A2" => Some(2),
        "A3" => Some(3),
        "A4" => Some(4),
        "A5" => Some(5),
        _ => None,
    }
}

fn canonical_case(id: &str) -> Option<&'static (&'static str, &'static str, &'static str, &'static str, &'static str, &'static str)> {
    CANONICAL_CASES.iter().find(|case| case.0 == id)
}

fn validate_fixture(fixture: &Fixture, issues: &mut Vec<String>, tier_counts: &mut [usize; 6]) {
    if let Some(index) = tier_index(&fixture.tier) {
        tier_counts[index] += 1;
    } else {
        issues.push(format!("{}: unsupported tier", fixture.fixture_id));
    }

    for (name, value) in [
        ("world_id", fixture.world_id.as_str()),
        ("world_seed_id", fixture.world_seed_id.as_str()),
        ("discovery_split_id", fixture.discovery_split_id.as_str()),
        ("calibration_split_id", fixture.calibration_split_id.as_str()),
        ("confirmation_split_id", fixture.confirmation_split_id.as_str()),
        ("expected_disposition", fixture.expected_disposition.as_str()),
    ] {
        if !non_empty(value) {
            issues.push(format!("{}: empty {name}", fixture.fixture_id));
        }
    }

    if fixture.experiment_budget == 0 || fixture.measurement_budget == 0 {
        issues.push(format!("{}: budgets must be positive", fixture.fixture_id));
    }
    if fixture.discovery_split_id == fixture.calibration_split_id
        || fixture.discovery_split_id == fixture.confirmation_split_id
        || fixture.calibration_split_id == fixture.confirmation_split_id
    {
        issues.push(format!("{}: discovery, calibration and confirmation splits overlap", fixture.fixture_id));
    }

    if fixture.required_evidence.is_empty()
        || fixture.required_evidence.iter().any(|item| !non_empty(item))
    {
        issues.push(format!("{}: required_evidence is empty or invalid", fixture.fixture_id));
    }
    if fixture.prohibited_inference.is_empty()
        || fixture.prohibited_inference.iter().any(|item| !non_empty(item))
    {
        issues.push(format!("{}: prohibited_inference is empty or invalid", fixture.fixture_id));
    }

    let mut seen_fields = BTreeSet::new();
    for field in &fixture.agent_visible_fields {
        if !seen_fields.insert(field.as_str()) {
            issues.push(format!("{}: duplicate agent-visible field {field}", fixture.fixture_id));
        }
        if FORBIDDEN_AGENT_FIELDS.contains(&field.as_str()) {
            issues.push(format!("{}: evaluator-only field leaked to agent interface: {field}", fixture.fixture_id));
        }
        if !ALLOWED_AGENT_FIELDS.contains(&field.as_str()) {
            issues.push(format!("{}: unknown agent-visible field: {field}", fixture.fixture_id));
        }
    }

    match canonical_case(&fixture.fixture_id) {
        Some((_, tier, structure_input, channel, identifiability, disposition)) => {
            if fixture.tier != *tier
                || fixture.structure_input != *structure_input
                || fixture.measurement_channel != *channel
                || fixture.identifiability != *identifiability
                || fixture.expected_disposition != *disposition
            {
                issues.push(format!("{}: canonical tier / structure / channel / identifiability / disposition changed", fixture.fixture_id));
            }
        }
        None => issues.push(format!("{}: unknown fixture identity", fixture.fixture_id)),
    }

    if (fixture.tier == "A0") != (fixture.structure_input == "supplied") {
        issues.push(format!("{}: only A0 parameter controls may receive the true structure", fixture.fixture_id));
    }
    if fixture.tier != "A0" && fixture.structure_input != "withheld" {
        issues.push(format!("{}: structure must be withheld outside A0", fixture.fixture_id));
    }
    if fixture.identifiability == "not_identifiable"
        && !fixture.expected_disposition.starts_with("abstain_")
    {
        issues.push(format!("{}: non-identifiable world must require abstention", fixture.fixture_id));
    }
    if fixture.identifiability == "equivalence_class"
        && fixture.expected_disposition != "report_observational_equivalence_class"
    {
        issues.push(format!("{}: equivalence-class world must preserve set-valued structure", fixture.fixture_id));
    }
    if fixture.tier == "A3" && fixture.measurement_channel == "uninformative"
        && fixture.identifiability != "not_identifiable"
    {
        issues.push(format!("{}: uninformative channel must not imply unique identification", fixture.fixture_id));
    }
    if fixture.tier == "A4" && (!fixture.intervention_access || !fixture.long_horizon_holdout) {
        issues.push(format!("{}: A4 requires intervention access and a long-horizon holdout", fixture.fixture_id));
    }
    if fixture.tier == "A5" && fixture.transfer_relation == "none" {
        issues.push(format!("{}: A5 requires an explicitly related or unrelated transfer source", fixture.fixture_id));
    }
    if fixture.tier != "A5" && fixture.transfer_relation != "none" {
        issues.push(format!("{}: transfer relation is reserved for A5", fixture.fixture_id));
    }
    if fixture.tier == "A5"
        && fixture.transfer_relation == "related"
        && !fixture.expected_disposition.contains("related")
    {
        issues.push(format!("{}: related transfer must be tested before acceptance", fixture.fixture_id));
    }
    if fixture.tier == "A5"
        && fixture.transfer_relation == "unrelated"
        && !fixture.expected_disposition.contains("reject_negative_transfer")
    {
        issues.push(format!("{}: unrelated transfer must be rejected/quarantined", fixture.fixture_id));
    }
}

pub fn validate_pack(pack: &FixturePack) -> ValidationReport {
    let mut issues = Vec::new();
    let mut fixture_ids = BTreeSet::new();
    let mut world_seeds: BTreeMap<&str, Vec<&Fixture>> = BTreeMap::new();
    let mut paired_groups: BTreeMap<&str, Vec<&Fixture>> = BTreeMap::new();
    let mut split_owners: BTreeMap<&str, Vec<&Fixture>> = BTreeMap::new();
    let mut tier_counts = [0usize; 6];

    if pack.schema_version != SCHEMA_VERSION {
        issues.push("SCHEMA_VERSION: unsupported fixture schema".to_owned());
    }
    if pack.status != REQUIRED_STATUS {
        issues.push("STATUS: pack must remain explicitly synthetic and non-blind".to_owned());
    }
    if pack.source_issue != REQUIRED_ISSUE {
        issues.push("SOURCE_ISSUE: expected #11042".to_owned());
    }
    if !non_empty(&pack.purpose) {
        issues.push("PURPOSE: missing".to_owned());
    }
    if pack.evaluation_contract.structural_score_unit != "identified_structural_equivalence_class"
        || !pack.evaluation_contract.coefficient_score_separate
        || pack.evaluation_contract.non_identifiable_policy.trim().is_empty()
        || pack.evaluation_contract.hidden_truth_boundary.trim().is_empty()
        || pack.evaluation_contract.cost_boundary.trim().is_empty()
        || pack.evaluation_contract.task_claim_boundary.trim().is_empty()
    {
        issues.push("EVALUATION_CONTRACT: structure, coefficient and non-identifiability boundaries are incomplete".to_owned());
    }
    if pack.fixtures.len() != EXPECTED_FIXTURE_COUNT {
        issues.push(format!("FIXTURE_COUNT: expected {EXPECTED_FIXTURE_COUNT}, found {}", pack.fixtures.len()));
    }

    for fixture in &pack.fixtures {
        if !fixture_ids.insert(fixture.fixture_id.as_str()) {
            issues.push(format!("DUPLICATE_FIXTURE_ID: {}", fixture.fixture_id));
        }
        if !fixture.world_seed_id.is_empty() {
            world_seeds.entry(&fixture.world_seed_id).or_default().push(fixture);
        }
        for split in [
            fixture.discovery_split_id.as_str(),
            fixture.calibration_split_id.as_str(),
            fixture.confirmation_split_id.as_str(),
        ] {
            if !split.is_empty() {
                split_owners.entry(split).or_default().push(fixture);
            }
        }
        if let Some(group) = fixture.paired_control_group.as_deref() {
            paired_groups.entry(group).or_default().push(fixture);
        }
        validate_fixture(fixture, &mut issues, &mut tier_counts);
    }

    for (seed, members) in &world_seeds {
        if members.len() > 1 {
            let group = members.first().and_then(|fixture| fixture.paired_control_group.as_deref());
            let is_valid_pair = members.len() == 2
                && group.is_some()
                && members.iter().all(|fixture| fixture.paired_control_group.as_deref() == group)
                && members.iter().any(|fixture| fixture.tier == "A0")
                && members.iter().any(|fixture| fixture.tier == "A2")
                && members[0].world_id == members[1].world_id;
            if !is_valid_pair {
                issues.push(format!("WORLD_SEED_REUSE: {seed} is reused outside the matched A0/A2 control"));
            }
        }
    }

    for (split, owners) in &split_owners {
        if owners.len() > 1 {
            let group = owners.first().and_then(|fixture| fixture.paired_control_group.as_deref());
            let is_valid_pair = owners.len() == 2
                && group.is_some()
                && owners.iter().all(|fixture| fixture.paired_control_group.as_deref() == group)
                && owners.iter().any(|fixture| fixture.tier == "A0")
                && owners.iter().any(|fixture| fixture.tier == "A2");
            if !is_valid_pair {
                issues.push(format!("SPLIT_REUSE: {split} appears in unrelated fixture cases"));
            }
        }
    }

    if paired_groups.len() != 1 {
        issues.push(format!("PAIRED_GROUP_COUNT: expected one matched A0/A2 control, found {}", paired_groups.len()));
    }
    for (group, members) in &paired_groups {
        if members.len() != 2 {
            issues.push(format!("{group}: paired control must contain exactly two cases"));
            continue;
        }
        let a0 = members.iter().find(|fixture| fixture.tier == "A0");
        let a2 = members.iter().find(|fixture| fixture.tier == "A2");
        let (Some(a0), Some(a2)) = (a0, a2) else {
            issues.push(format!("{group}: pair must include exactly one A0 and one A2 case"));
            continue;
        };
        let matched = a0.world_id == a2.world_id
            && a0.world_seed_id == a2.world_seed_id
            && a0.discovery_split_id == a2.discovery_split_id
            && a0.calibration_split_id == a2.calibration_split_id
            && a0.confirmation_split_id == a2.confirmation_split_id
            && a0.experiment_budget == a2.experiment_budget
            && a0.measurement_budget == a2.measurement_budget
            && a0.measurement_channel == a2.measurement_channel
            && a0.identifiability == a2.identifiability
            && a0.hidden_state_present == a2.hidden_state_present
            && a0.intervention_access == a2.intervention_access
            && a0.long_horizon_holdout == a2.long_horizon_holdout
            && a0.transfer_relation == a2.transfer_relation
            && a0.structure_input != a2.structure_input;
        if !matched {
            issues.push(format!("{group}: A0/A2 pair must share world, seed, splits and inclusive experiment budgets; structure availability is the treatment"));
        }
        if !a0.agent_visible_fields.iter().any(|field| field == "supplied_model_structure")
            || a2.agent_visible_fields.iter().any(|field| field == "supplied_model_structure")
        {
            issues.push(format!("{group}: only the A0 agent-visible packet may include supplied_model_structure"));
        }
    }

    for required in ["ANS-A0-01","ANS-A2-01","ANS-A2-02","ANS-A3-01","ANS-A3-02","ANS-A2-03","ANS-A4-01","ANS-A4-02","ANS-A5-01","ANS-A5-02"] {
        if !fixture_ids.contains(required) {
            issues.push(format!("REQUIRED_CASE_MISSING: {required}"));
        }
    }

    issues.sort();
    issues.dedup();
    ValidationReport {
        valid: issues.is_empty(),
        fixture_count: pack.fixtures.len(),
        paired_control_groups: paired_groups.len(),
        tier_counts,
        issues,
        limitations: vec![
            "These ten cases are public synthetic contract fixtures, not blind worlds or an Unsatz reproduction.".to_owned(),
            "A valid pack does not show that an agent can infer a hidden structure or predict a real system.".to_owned(),
            "The A0/A2 pair tests experimental-design matching at the contract level; it does not execute a parameter fitter.".to_owned(),
            "The validator checks declared split identifiers, not hidden-data custody; true blindness needs an independent evaluator boundary.".to_owned(),
            "A symbolic structure is scored only up to the declared observable equivalence class where unique identification is impossible.".to_owned(),
        ],
    }
}

pub fn parse_and_validate(contents: &str) -> Result<ValidationReport, serde_json::Error> {
    let pack: FixturePack = serde_json::from_str(contents)?;
    Ok(validate_pack(&pack))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURES: &str =
        include_str!("../../../docs/research/fixtures/OPEN_WORLD_ANSATZ_STRUCTURAL_DISCOVERY_FIXTURES_V0_2026-11-07.json");

    fn valid_pack() -> FixturePack {
        serde_json::from_str(FIXTURES).expect("checked-in ansatz fixture pack parses")
    }

    #[test]
    fn accepts_checked_in_ansatz_contract_pack() {
        let report = validate_pack(&valid_pack());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.fixture_count, 10);
        assert_eq!(report.paired_control_groups, 1);
        assert_eq!(report.tier_counts, [1, 0, 3, 2, 2, 2]);
    }

    #[test]
    fn rejects_agent_visible_hidden_truth_field() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][1]["agent_visible_fields"].as_array_mut()
            .expect("agent field list").push(Value::String("latent_truth".to_owned()));
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("evaluator-only field leaked")));
    }

    #[test]
    fn rejects_overlapping_discovery_and_confirmation_split() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][2]["confirmation_split_id"] = value["fixtures"][2]["discovery_split_id"].clone();
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("splits overlap")));
    }

    #[test]
    fn rejects_unmatched_a0_a2_budget() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][1]["experiment_budget"] = Value::from(11);
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("A0/A2 pair must share")));
    }

    #[test]
    fn rejects_unique_structure_on_nonidentifiable_world() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][4]["expected_disposition"] =
            Value::String("propose_structural_model".to_owned());
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("canonical tier / structure")));
        assert!(report.issues.iter().any(|issue| issue.contains("non-identifiable world must require abstention")));
    }

    #[test]
    fn rejects_unrelated_transfer_without_negative_transfer_rejection() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][9]["expected_disposition"] =
            Value::String("accept_transfer_without_validation".to_owned());
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("canonical tier / structure")));
        assert!(report.issues.iter().any(|issue| issue.contains("unrelated transfer must be rejected")));
    }

    #[test]
    fn rejects_duplicate_world_seed_outside_matched_pair() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][2]["world_seed_id"] = Value::String("world-seed-microcosm-01".to_owned());
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("WORLD_SEED_REUSE")));
    }

    #[test]
    fn rejects_missing_structural_evidence_requirements() {
        let mut value: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
        value["fixtures"][3]["required_evidence"] = Value::Array(Vec::new());
        let pack: FixturePack = serde_json::from_value(value).expect("mutated fixture pack parses");
        let report = validate_pack(&pack);
        assert!(report.issues.iter().any(|issue| issue.contains("required_evidence is empty")));
    }
}
