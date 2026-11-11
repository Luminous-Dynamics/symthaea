use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const BENCHMARK_ID: &str = "IG-RISK-ACTION-001";
const SCHEMA_VERSION: &str = "ig-risk-action-fixtures-v0.1";
const SCHEMA_ID: &str = "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-risk-action-001/IG_RISK_ACTION_FIXTURES_V0_1.schema.json";
const ACTIONS: [&str; 5] = ["Answer", "Abstain", "RequestEvidence", "Escalate", "Defer"];
const EXPECTED_FIXTURES: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rational { numerator: i128, denominator: i128 }

impl Rational {
    fn new(numerator: i128, denominator: i128) -> Result<Self, String> {
        if denominator <= 0 { return Err("rational denominator must be positive".into()); }
        let divisor = gcd(numerator.abs(), denominator);
        Ok(Self { numerator: numerator / divisor, denominator: denominator / divisor })
    }
    fn compare(self, other: Self) -> Ordering {
        (self.numerator * other.denominator).cmp(&(other.numerator * self.denominator))
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 { let r = a % b; a = b; b = r; }
    if a == 0 { 1 } else { a.abs() }
}

fn field_i64(value: &Value, key: &str) -> Result<i64, String> {
    value.get(key).and_then(Value::as_i64).ok_or_else(|| format!("{key} must be a signed integer"))
}
fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be an unsigned integer"))
}
fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|text| !text.trim().is_empty())
}
fn probability(value: &Value) -> Result<Rational, String> {
    let numerator = field_u64(value, "numerator")? as i128;
    let denominator = field_u64(value, "denominator")? as i128;
    if denominator == 0 || numerator > denominator || denominator > 1_000_000 {
        return Err("probability must be in [0,1] with denominator <= 1000000".into());
    }
    Rational::new(numerator, denominator)
}
fn expected_utility(p: Rational, reward: i128, penalty: i128, cost: i128) -> Result<Rational, String> {
    let numerator = p.numerator.checked_mul(reward)
        .and_then(|value| (p.denominator-p.numerator).checked_mul(penalty).and_then(|wrong| value.checked_sub(wrong)))
        .and_then(|value| p.denominator.checked_mul(cost).and_then(|amount| value.checked_sub(amount)))
        .ok_or_else(|| "expected-utility arithmetic overflow".to_string())?;
    Rational::new(numerator, p.denominator)
}
fn parse_expected(value: &Value) -> Result<Rational, String> {
    Rational::new(field_i64(value, "numerator")? as i128, field_u64(value, "denominator")? as i128)
}

fn validate_schema(schema: &Value) -> Result<(), String> {
    if string_field(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema")
        || string_field(schema, "$id") != Some(SCHEMA_ID)
        || string_field(schema, "type") != Some("object")
        || schema["additionalProperties"] != false
        || schema["properties"]["fixtures"]["type"] != "array" {
        return Err("schema identity or bounded top-level contract mismatch".into());
    }
    Ok(())
}

fn calculate_utilities(fixture: &Value) -> Result<BTreeMap<String, Rational>, String> {
    let probs = &fixture["probabilities"];
    let loss = &fixture["loss_matrix"];
    let reward = field_i64(loss, "correct_reward")? as i128;
    let wrong = field_i64(loss, "wrong_answer_penalty")? as i128;
    let abstain = field_i64(loss, "abstain_cost")? as i128;
    let evidence = field_i64(loss, "request_evidence_cost")? as i128;
    let escalation = field_i64(loss, "escalation_cost")? as i128;
    let delay = field_i64(loss, "delay_cost")? as i128;
    if reward <= 0 || [wrong, abstain, evidence, escalation, delay].iter().any(|x| *x < 0)
        || [reward, wrong, abstain, evidence, escalation, delay].iter().any(|x| x.abs() > 1_000_000) {
        return Err("loss matrix is outside the bounded non-negative cost profile".into());
    }
    let mut out = BTreeMap::new();
    out.insert("Answer".into(), expected_utility(probability(&probs["answer_correctness"])?, reward, wrong, 0)?);
    out.insert("Abstain".into(), Rational::new(-abstain, 1)?);
    out.insert("RequestEvidence".into(), expected_utility(probability(&probs["evidence_adjusted_correctness"])?, reward, wrong, evidence)?);
    out.insert("Escalate".into(), expected_utility(probability(&probs["reviewer_correctness"])?, reward, wrong, escalation)?);
    out.insert("Defer".into(), expected_utility(probability(&probs["deferred_correctness"])?, reward, wrong, delay)?);
    Ok(out)
}

fn choose_action(values: &BTreeMap<String, Rational>, order: &[String]) -> Result<String, String> {
    let mut iter = order.iter();
    let mut best = iter.next().ok_or_else(|| "tie_break_order cannot be empty".to_string())?.clone();
    let mut best_value = *values.get(&best).ok_or_else(|| format!("missing utility for {best}"))?;
    for action in iter {
        let value = *values.get(action).ok_or_else(|| format!("missing utility for {action}"))?;
        if value.compare(best_value) == Ordering::Greater { best = action.clone(); best_value = value; }
    }
    Ok(best)
}

fn validate_fixture(fixture: &Value) -> Result<(String, String), String> {
    let id = string_field(fixture, "scenario_id").ok_or_else(|| "scenario_id missing".to_string())?;
    if string_field(fixture, "probability_source") != Some("SyntheticScenarioOracle")
        || string_field(fixture, "probability_target") != Some("scenario_defined_action_outcome_success") {
        return Err("probability source or confidence target mismatch".into());
    }
    let family = string_field(fixture, "task_family").ok_or_else(|| "task_family missing".to_string())?;
    if !["factual_reasoning", "retrieval", "mechanical_diagnosis", "institutional_process"].contains(&family) {
        return Err(format!("unknown task family {family}"));
    }
    let values = calculate_utilities(fixture)?;
    let declared = fixture["expected_utilities"].as_object().ok_or_else(|| "expected_utilities must be an object".to_string())?;
    if declared.len() != ACTIONS.len() { return Err("expected_utilities must contain exactly five actions".into()); }
    for action in ACTIONS {
        let actual = parse_expected(declared.get(action).ok_or_else(|| format!("missing utility for {action}"))?)?;
        if values.get(action).copied() != Some(actual) { return Err(format!("expected utility mismatch for {action}")); }
    }
    let order = fixture["tie_break_order"].as_array().ok_or_else(|| "tie_break_order must be an array".to_string())?
        .iter().map(|v| v.as_str().map(ToOwned::to_owned).ok_or_else(|| "non-string tie-break action".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let set = order.iter().cloned().collect::<BTreeSet<_>>();
    if order.len() != 5 || set.len() != 5 || order.iter().any(|a| !ACTIONS.contains(&a.as_str())) {
        return Err("tie_break_order must be a permutation of the five actions".into());
    }
    let chosen = choose_action(&values, &order)?;
    if string_field(fixture, "expected_action") != Some(chosen.as_str()) {
        return Err(format!("action oracle mismatch: computed {chosen}"));
    }
    let pair = string_field(fixture, "sensitivity_pair_id").unwrap_or_default();
    let level = string_field(fixture, "sensitivity_pair_level").unwrap_or_default();
    if pair.is_empty() != level.is_empty() || (!pair.is_empty() && !["low_penalty", "high_penalty"].contains(&level)) {
        return Err("sensitivity pair identity/level is incomplete or invalid".into());
    }
    Ok((id.to_string(), pair.to_string()))
}

fn validate_manifest(manifest: &Value) -> Result<BTreeMap<String, usize>, Vec<String>> {
    let mut errors = Vec::new();
    let expected_action_set = ACTIONS.iter().copied().collect::<BTreeSet<_>>();
    let actual_action_array = manifest["action_set"].as_array();
    let actual_action_set = actual_action_array.map(|xs| xs.iter().filter_map(Value::as_str).collect::<BTreeSet<_>>());
    if string_field(manifest, "schema_version") != Some(SCHEMA_VERSION)
        || string_field(manifest, "benchmark_id") != Some(BENCHMARK_ID)
        || string_field(manifest, "status") != Some("synthetic_known_answer_unexecuted")
        || manifest["fixture_count"].as_u64() != Some(EXPECTED_FIXTURES as u64)
        || actual_action_array.map(Vec::len) != Some(ACTIONS.len())
        || actual_action_set != Some(expected_action_set) {
        errors.push("manifest identity, status, fixture count or action set mismatch".into());
    }
    let Some(fixtures) = manifest["fixtures"].as_array() else { return Err(vec!["fixtures must be an array".into()]); };
    if fixtures.len() != EXPECTED_FIXTURES { errors.push(format!("expected {EXPECTED_FIXTURES} fixtures, got {}", fixtures.len())); }
    let mut ids = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut pairs = BTreeMap::<String, Vec<&Value>>::new();
    for (index, fixture) in fixtures.iter().enumerate() {
        match validate_fixture(fixture) {
            Ok((id, pair)) => {
                if !ids.insert(id.clone()) { errors.push(format!("fixtures[{index}] duplicates scenario ID {id}")); }
                let action = string_field(fixture, "expected_action").unwrap_or_default().to_string();
                *counts.entry(action).or_default() += 1;
                if !pair.is_empty() { pairs.entry(pair).or_default().push(fixture); }
            }
            Err(error) => errors.push(format!("fixtures[{index}]: {error}")),
        }
    }
    for action in ACTIONS { if counts.get(action).copied().unwrap_or(0)==0 { errors.push(format!("no fixture makes {action} optimal")); } }
    if pairs.len()!=5 { errors.push(format!("expected five wrong-penalty pairs, got {}",pairs.len())); }
    for (id,members) in pairs {
        if members.len()!=2 { errors.push(format!("pair {id} must have exactly two cases")); continue; }
        let low=members.iter().find(|x|string_field(x,"sensitivity_pair_level")==Some("low_penalty"));
        let high=members.iter().find(|x|string_field(x,"sensitivity_pair_level")==Some("high_penalty"));
        let (Some(low),Some(high))=(low,high) else { errors.push(format!("pair {id} needs low and high cases")); continue; };
        let mut low_loss=low["loss_matrix"].clone(); let mut high_loss=high["loss_matrix"].clone();
        low_loss.as_object_mut().unwrap().remove("wrong_answer_penalty");
        high_loss.as_object_mut().unwrap().remove("wrong_answer_penalty");
        let lp=low["loss_matrix"]["wrong_answer_penalty"].as_i64();
        let hp=high["loss_matrix"]["wrong_answer_penalty"].as_i64();
        if low["task_family"] != high["task_family"]
            || low["probability_source"] != high["probability_source"]
            || low["probability_target"] != high["probability_target"]
            || low["probabilities"]!=high["probabilities"] || low_loss!=high_loss
            || lp.zip(hp).is_none_or(|(a,b)|a>=b) || low["expected_action"]==high["expected_action"] {
            errors.push(format!("pair {id} must change only wrong-answer penalty and change optimal action"));
        }
    }
    if errors.is_empty() { Ok(counts) } else { errors.sort(); errors.dedup(); Err(errors) }
}

fn read_json(path: &str) -> Result<Value, String> {
    let source=fs::read_to_string(path).map_err(|e|format!("{path}: {e}"))?;
    serde_json::from_str(&source).map_err(|e|format!("{path}: invalid JSON: {e}"))
}
fn main() -> ExitCode {
    let args:Vec<String>=env::args().collect();
    if args.len()!=3 { eprintln!("usage: ig-risk-action-qualifier <fixtures.json> <schema.json>"); return ExitCode::from(2); }
    let manifest=match read_json(&args[1]) { Ok(x)=>x, Err(e)=>{eprintln!("{e}");return ExitCode::FAILURE;} };
    let schema=match read_json(&args[2]) { Ok(x)=>x, Err(e)=>{eprintln!("{e}");return ExitCode::FAILURE;} };
    if let Err(e)=validate_schema(&schema) { eprintln!("SCHEMA QUALIFICATION FAILED: {e}");return ExitCode::FAILURE; }
    match validate_manifest(&manifest) {
        Ok(counts)=>{println!("QUALIFIED SYNTHETIC ACTION FIXTURES: total=40");for a in ACTIONS {println!("  {a}: {}",counts.get(a).copied().unwrap_or(0));}println!("NOT EXECUTED: no model calibration or behavioral trial is implied.");ExitCode::SUCCESS}
        Err(errors)=>{eprintln!("RISK-ACTION QUALIFICATION FAILED ({} findings):",errors.len());for e in errors {eprintln!(" - {e}");}ExitCode::FAILURE}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const FIXTURES:&str=include_str!("../../IG_RISK_ACTION_FIXTURES_V0_1.json");
    fn parsed()->Value { serde_json::from_str(FIXTURES).expect("fixtures parse") }

    #[test] fn all_forty_fixtures_and_five_actions_validate() {
        let counts=validate_manifest(&parsed()).expect("fixtures validate");
        assert_eq!(counts.values().sum::<usize>(),40); assert_eq!(counts.len(),5);
    }
    #[test] fn duplicate_top_level_action_is_rejected() {
        let mut v=parsed();
        let duplicate=v["action_set"][0].clone();
        v["action_set"][4]=duplicate;
        assert!(validate_manifest(&v).is_err());
    }
    #[test] fn matched_pair_must_preserve_task_family() {
        let mut v=parsed();
        v["fixtures"][0]["task_family"]=Value::String("retrieval".into());
        assert!(validate_manifest(&v).is_err());
    }
    #[test] fn missing_fixture_is_rejected() {
        let mut v=parsed();v["fixtures"].as_array_mut().unwrap().pop();assert!(validate_manifest(&v).is_err());
    }
    #[test] fn utility_tampering_is_rejected() {
        let mut v=parsed();v["fixtures"][0]["expected_utilities"]["Answer"]["numerator"]=Value::from(999);assert!(validate_manifest(&v).is_err());
    }
    #[test] fn probability_target_substitution_is_rejected() {
        let mut v=parsed();v["fixtures"][0]["probability_target"]=Value::String("pre_reasoning_success".into());assert!(validate_manifest(&v).is_err());
    }
    #[test] fn invalid_probability_is_rejected() {
        let mut v=parsed();v["fixtures"][0]["probabilities"]["answer_correctness"]["numerator"]=Value::from(101);assert!(validate_manifest(&v).is_err());
    }
    #[test] fn duplicate_tie_break_action_is_rejected() {
        let mut v=parsed();let duplicate=v["fixtures"][0]["tie_break_order"][3].clone();v["fixtures"][0]["tie_break_order"][4]=duplicate;assert!(validate_manifest(&v).is_err());
    }
    #[test] fn always_abstain_mutation_is_rejected() {
        let mut v=parsed();for f in v["fixtures"].as_array_mut().unwrap(){f["expected_action"]=Value::String("Abstain".into());}assert!(validate_manifest(&v).is_err());
    }
    #[test] fn wrong_penalty_pairs_hold_other_inputs_constant() {
        let mut v=parsed();v["fixtures"][1]["loss_matrix"]["escalation_cost"]=Value::from(7);assert!(validate_manifest(&v).is_err());
    }
}
