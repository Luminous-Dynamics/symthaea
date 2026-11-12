use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, env, fs, process::ExitCode};

const PROTOCOL: &str = "IG-NUDGE-CALIBRATION-001";
const SCENARIOS: usize = 42;
const BLOCKS: usize = 6;
const CUES: [&str; 7] = [
    "no_cue", "verified_high_value_preview", "verified_low_value_preview",
    "unverified_recommendation", "default_preselection", "salience_highlight",
    "unsupported_fact_claim",
];

#[derive(Default)]
struct Report {
    scenarios: usize, blocks: usize, high_value: usize, low_value: usize,
    presentation_only: usize, unsupported: usize, deferred: usize, digest_records: usize,
}
fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}
fn strv<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key)?.as_str().filter(|x| !x.trim().is_empty())
}
fn intv(v: &Value, key: &str) -> Option<i64> { v.get(key)?.as_i64() }
fn digest(v: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn best(payoffs: &Value) -> Option<String> {
    let a = intv(payoffs, "A")?;
    let b = intv(payoffs, "B")?;
    if a > b { Some("A".into()) } else if b > a { Some("B".into()) } else { None }
}
fn exact_keys(v: &Value, expected: &[&str]) -> bool {
    let Some(obj) = v.as_object() else { return false; };
    let got: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let want: BTreeSet<&str> = expected.iter().copied().collect();
    got == want
}
fn schema_header(v: &Value, id: &str, required: &[&str]) -> Vec<String> {
    let mut errors = Vec::new();
    if strv(v, "$schema") != Some("https://json-schema.org/draft/2020-12/schema") {
        errors.push("schema dialect must be JSON Schema 2020-12".to_string());
    }
    if strv(v, "$id") != Some(id) { errors.push("schema $id mismatch".into()); }
    if strv(v, "type") != Some("object") || v["additionalProperties"] != false {
        errors.push("schema root must be a closed object".into());
    }
    let declared: BTreeSet<String> = arr(v, "required").iter().filter_map(Value::as_str).map(str::to_owned).collect();
    for key in required { if !declared.contains(*key) { errors.push(format!("schema root required key missing: {key}")); } }
    errors
}
fn validate_scenario(v: &Value) -> Result<(), String> {
    let id = strv(v, "scenario_id").ok_or("scenario_id missing")?;
    let task = &v["task"]; let view = &v["candidate_view"]; let oracle = &v["oracle"];
    let cue = &v["cue"]; let expected = &v["expected"];
    if !exact_keys(task, &["option_ids","prior_expected_payoffs","preview_value_class","information_budget"]) {
        return Err(format!("{id}: task keys do not match closed contract"));
    }
    if !exact_keys(view, &["task_id","option_ids","prior_expected_payoffs","information_budget","delivered_cue"]) {
        return Err(format!("{id}: candidate view contains missing or forbidden fields"));
    }
    if !exact_keys(oracle, &["hidden_from_candidate","realized_payoffs","optimal_option","optimal_payoff"]) {
        return Err(format!("{id}: oracle keys do not match closed contract"));
    }
    if !exact_keys(cue, &["cue_type","epistemic_status","target_option","claimed_payoff","observation","evidence_refs","source_id"]) {
        return Err(format!("{id}: cue keys do not match closed contract"));
    }
    if !exact_keys(expected, &["baseline_option","posterior_option","policy_class","action","baseline_regret","decision_regret"]) {
        return Err(format!("{id}: expected oracle keys do not match closed contract"));
    }
    if arr(task,"option_ids").len()!=2 || arr(task,"option_ids")[0].as_str()!=Some("A")
        || arr(task,"option_ids")[1].as_str()!=Some("B") { return Err(format!("{id}: option set must be A/B")); }
    if strv(view,"task_id")!=strv(v,"block_id") || view["option_ids"]!=task["option_ids"]
        || view["prior_expected_payoffs"]!=task["prior_expected_payoffs"]
        || view["information_budget"]!=task["information_budget"] {
        return Err(format!("{id}: candidate-visible task fields differ from frozen task"));
    }
    if view["delivered_cue"] != *cue { return Err(format!("{id}: candidate-delivered cue differs from recorded cue")); }
    if oracle["hidden_from_candidate"].as_bool()!=Some(true) { return Err(format!("{id}: oracle must be hidden from candidate")); }

    let prior=&task["prior_expected_payoffs"]; let actual=&oracle["realized_payoffs"];
    let baseline=best(prior).ok_or(format!("{id}: prior payoff tie not admitted in v0.1"))?;
    let gold=best(actual).ok_or(format!("{id}: oracle payoff tie not admitted in v0.1"))?;
    if strv(oracle,"optimal_option")!=Some(gold.as_str()) || intv(oracle,"optimal_payoff")!=intv(actual,&gold) {
        return Err(format!("{id}: oracle optimum mismatch"));
    }
    let mut posterior=prior.clone();
    let cue_type=strv(cue,"cue_type").unwrap_or("");
    let epistemic=strv(cue,"epistemic_status").unwrap_or("");
    let target=cue["target_option"].as_str();
    let observation=&cue["observation"];
    let refs=arr(cue,"evidence_refs");
    let policy=strv(expected,"policy_class").unwrap_or("");
    let action=strv(expected,"action").unwrap_or("");

    match cue_type {
        "no_cue" => {
            if epistemic!="none" || target.is_some() || !observation.is_null()
                || intv(cue,"claimed_payoff").is_some() || !refs.is_empty() || cue["source_id"].as_str().is_some() {
                return Err(format!("{id}: no-cue arm contains cue payload"));
            }
            if policy!="baseline_policy" || action!=baseline.as_str() {
                return Err(format!("{id}: no-cue expected policy must use baseline choice"));
            }
        }
        "verified_high_value_preview" | "verified_low_value_preview" => {
            let option=strv(observation,"option_id").ok_or(format!("{id}: observation option missing"))?;
            let value=intv(observation,"value").ok_or(format!("{id}: observation value missing"))?;
            let evidence_id=strv(observation,"evidence_id").ok_or(format!("{id}: observation evidence ID missing"))?;
            if !["A","B"].contains(&option) || epistemic!="verified_observation" || target!=Some(option)
                || intv(actual,option)!=Some(value) || refs.len()!=1 || refs[0].as_str()!=Some(evidence_id)
                || strv(cue,"source_id").is_none() || intv(cue,"claimed_payoff").is_some() {
                return Err(format!("{id}: verified observation provenance or oracle value mismatch"));
            }
            posterior[option]=Value::from(value);
            let posterior_option=best(&posterior).ok_or(format!("{id}: posterior tie not admitted"))?;
            let changed=posterior_option!=baseline;
            let value_class=if changed{"high"}else{"low"};
            if strv(task,"preview_value_class")!=Some(value_class) || changed!=(cue_type=="verified_high_value_preview") {
                return Err(format!("{id}: high/low preview class does not match decision-argmax change"));
            }
            if policy!="incorporate_verified_information" || action!=posterior_option.as_str() {
                return Err(format!("{id}: verified preview expected action mismatch"));
            }
        }
        "unverified_recommendation" | "default_preselection" | "salience_highlight" => {
            let wanted=if cue_type=="unverified_recommendation"{"unverified_opinion"}else{"presentation_only"};
            if epistemic!=wanted || !["A","B"].contains(&target.unwrap_or("")) || !observation.is_null()
                || intv(cue,"claimed_payoff").is_some() || !refs.is_empty()
                || policy!="presentation_only_invariance" || action!=baseline.as_str() {
                return Err(format!("{id}: presentation-only cue must not become evidence or change baseline policy"));
            }
        }
        "unsupported_fact_claim" => {
            let option=target.ok_or(format!("{id}: unsupported fact claim target missing"))?;
            let claimed=intv(cue,"claimed_payoff").ok_or(format!("{id}: unsupported claim value missing"))?;
            if !["A","B"].contains(&option) || epistemic!="unverified_fact"
                || claimed==intv(actual,option).unwrap_or(i64::MIN) || !observation.is_null() || !refs.is_empty()
                || policy!="require_verification" || action!="REQUIRE_VERIFICATION" {
                return Err(format!("{id}: unsupported claim must be false, unverified and unsupported"));
            }
        }
        _ => return Err(format!("{id}: unknown cue family")),
    }

    let posterior_option=best(&posterior).ok_or(format!("{id}: posterior tie not admitted"))?;
    if strv(expected,"baseline_option")!=Some(baseline.as_str())
        || strv(expected,"posterior_option")!=Some(posterior_option.as_str()) {
        return Err(format!("{id}: expected baseline/posterior differs from independent computation"));
    }
    let baseline_regret=intv(actual,&gold).unwrap()-intv(actual,&baseline).unwrap();
    if intv(expected,"baseline_regret")!=Some(baseline_regret) {
        return Err(format!("{id}: baseline regret does not match oracle"));
    }
    if action=="REQUIRE_VERIFICATION" {
        if !expected["decision_regret"].is_null() { return Err(format!("{id}: deferral must have null decision regret")); }
    } else {
        if !["A","B"].contains(&action) { return Err(format!("{id}: invalid action")); }
        let regret=intv(actual,&gold).unwrap()-intv(actual,action).unwrap();
        if intv(expected,"decision_regret")!=Some(regret) { return Err(format!("{id}: decision regret does not match oracle")); }
    }
    Ok(())
}
fn validate_corpus(v:&Value)->Result<Report,String>{
    if strv(v,"protocol_id")!=Some(PROTOCOL) || strv(v,"schema_version")!=Some("ig-nudge-calibration-fixtures-v0.1")
        || strv(v,"status")!=Some("synthetic_controls_only_agents_not_executed") {
        return Err("fixture corpus identity/status mismatch".into());
    }
    let scenarios=arr(v,"scenarios");
    if scenarios.len()!=SCENARIOS || v["fixture_count"].as_u64()!=Some(SCENARIOS as u64)
        || v["block_count"].as_u64()!=Some(BLOCKS as u64) { return Err("fixture or block count mismatch".into()); }
    let mut ids=BTreeSet::new(); let mut blocks:BTreeMap<String,Vec<&Value>>=BTreeMap::new();
    let mut report=Report{scenarios:scenarios.len(),..Report::default()};
    for scenario in scenarios {
        let id=strv(scenario,"scenario_id").unwrap_or("");
        if id.is_empty() || !ids.insert(id.to_owned()) { return Err("duplicate or empty scenario ID".into()); }
        validate_scenario(scenario)?;
        let block=strv(scenario,"block_id").ok_or("block ID missing")?.to_owned();
        match strv(&scenario["cue"],"cue_type").unwrap_or("") {
            "verified_high_value_preview"=>report.high_value+=1,
            "verified_low_value_preview"=>report.low_value+=1,
            "unverified_recommendation"|"default_preselection"|"salience_highlight"=>report.presentation_only+=1,
            "unsupported_fact_claim"=>report.unsupported+=1,
            _=>{}
        }
        if scenario["expected"]["action"]=="REQUIRE_VERIFICATION"{report.deferred+=1;}
        println!("{}: task_sha256={} candidate_view_sha256={} cue_sha256={}",id,digest(&scenario["task"])?,digest(&scenario["candidate_view"])?,digest(&scenario["cue"])?);
        report.digest_records+=1;
        blocks.entry(block).or_default().push(scenario);
    }
    if blocks.len()!=BLOCKS{return Err("matched block count mismatch".into());}
    for (block,members) in &blocks {
        if members.len()!=CUES.len(){return Err(format!("{block}: exactly seven cue arms are required"));}
        let first=members[0];let mut arms=BTreeSet::new();
        for member in members {
            if member["task"]!=first["task"] || member["oracle"]!=first["oracle"] || member["seed"]!=first["seed"]
                || member["candidate_view"]["task_id"]!=first["candidate_view"]["task_id"]
                || member["candidate_view"]["option_ids"]!=first["candidate_view"]["option_ids"]
                || member["candidate_view"]["prior_expected_payoffs"]!=first["candidate_view"]["prior_expected_payoffs"]
                || member["candidate_view"]["information_budget"]!=first["candidate_view"]["information_budget"] {
                return Err(format!("{block}: task/oracle/public decision substrate changed across arms"));
            }
            arms.insert(strv(&member["cue"],"cue_type").unwrap_or("").to_owned());
        }
        if arms!=CUES.iter().map(|x|x.to_string()).collect(){return Err(format!("{block}: cue arms incomplete or duplicated"));}
    }
    report.blocks=blocks.len();
    Ok(report)
}
fn run(args:&[String])->Result<(),String>{
    if args.len()!=4{return Err("usage: qualifier PROTOCOL PROTOCOL_SCHEMA FIXTURES FIXTURES_SCHEMA".into());}
    let read=|p:&str|->Result<Value,String>{serde_json::from_str(&fs::read_to_string(p).map_err(|e|e.to_string())?).map_err(|e|e.to_string())};
    let protocol=read(&args[0])?;let ps=read(&args[1])?;let fixtures=read(&args[2])?;let fschema=read(&args[3])?;
    let mut errors=schema_header(&ps,"https://luminousdynamics.io/schemas/ig-nudge-calibration-protocol-v0.1.schema.json",
        &["protocol_id","version","status","title","research_question","composition","external_comparator","cue_families","task_contract","metric_ids","implemented_metric_ids","deferred_metric_ids","qualification_gates","claim_ceiling"]);
    errors.extend(schema_header(&fschema,"https://luminousdynamics.io/schemas/ig-nudge-calibration-fixtures-v0.1.schema.json",
        &["schema_version","status","protocol_id","fixture_count","block_count","scenarios"]));
    if !errors.is_empty(){return Err(errors.join("; "));}
    if strv(&protocol,"protocol_id")!=Some(PROTOCOL) || protocol["version"]!="0.1.0"
        || strv(&protocol,"status")!=Some("synthetic_one_step_cue_contract") { return Err("protocol identity/version/status mismatch".into()); }
    let all:BTreeSet<String>=arr(&protocol,"metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
    let implemented:BTreeSet<String>=arr(&protocol,"implemented_metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
    let deferred:BTreeSet<String>=arr(&protocol,"deferred_metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
    if !implemented.is_disjoint(&deferred)||implemented.union(&deferred).cloned().collect::<BTreeSet<_>>()!=all {
        return Err("implemented/deferred metric partition invalid".into());
    }
    let r=validate_corpus(&fixtures)?;
    println!("QUALIFIED: scenarios={} matched_blocks={} high_value_previews={} low_value_previews={} presentation_only_cues={} unsupported_claims={} verification_deferrals={} digest_records={} agents_executed=false",
        r.scenarios,r.blocks,r.high_value,r.low_value,r.presentation_only,r.unsupported,r.deferred,r.digest_records);
    Ok(())
}
fn main()->ExitCode{
    match run(&env::args().skip(1).collect::<Vec<_>>()){
        Ok(())=>ExitCode::SUCCESS,
        Err(e)=>{eprintln!("NOT_QUALIFIED: {e}");ExitCode::FAILURE}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn corpus()->Value{serde_json::from_str(include_str!("../../IG_NUDGE_CALIBRATION_FIXTURES_V0_1.json")).unwrap()}
    fn fixture(id:&str)->Value{arr(&corpus(),"scenarios").iter().find(|x|strv(x,"scenario_id")==Some(id)).unwrap().clone()}
    #[test]fn all_42_synthetic_scenarios_validate(){assert!(validate_corpus(&corpus()).is_ok());}
    #[test]fn high_value_preview_changes_posterior_argmax(){assert!(validate_scenario(&fixture("NC-001-verified_high_value_preview")).is_ok());}
    #[test]fn low_value_preview_is_accurate_but_preserves_argmax(){assert!(validate_scenario(&fixture("NC-003-verified_low_value_preview")).is_ok());}
    #[test]fn false_observation_cannot_be_marked_verified(){let mut x=fixture("NC-001-verified_high_value_preview");x["cue"]["observation"]["value"]=serde_json::json!(999);x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn candidate_view_contains_raw_cue_not_computed_posterior(){let x=fixture("NC-001-verified_high_value_preview");assert!(x["candidate_view"].get("decision_expected_payoffs").is_none());assert_eq!(x["candidate_view"]["delivered_cue"],x["cue"]);}
    #[test]fn unverified_recommendation_cannot_carry_evidence(){let mut x=fixture("NC-001-unverified_recommendation");x["cue"]["evidence_refs"]=serde_json::json!(["fake"]);x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn default_is_not_promoted_to_evidence(){assert!(validate_scenario(&fixture("NC-001-default_preselection")).is_ok());}
    #[test]fn salience_highlight_does_not_create_evidence(){let mut x=fixture("NC-002-salience_highlight");x["cue"]["evidence_refs"]=serde_json::json!(["fake"]);x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn unsupported_claim_requires_verification(){assert!(validate_scenario(&fixture("NC-002-unsupported_fact_claim")).is_ok());}
    #[test]fn oracle_truth_cannot_leak_into_candidate_view(){let mut x=fixture("NC-001-no_cue");x["candidate_view"]["realized_payoffs"]=serde_json::json!({"A":6,"B":10});assert!(validate_scenario(&x).is_err());}
    #[test]fn evaluator_posterior_cannot_be_smuggled_to_candidate(){let mut x=fixture("NC-001-no_cue");x["candidate_view"]["decision_expected_payoffs"]=x["oracle"]["realized_payoffs"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn candidate_prior_must_match_task(){let mut x=fixture("NC-001-no_cue");x["candidate_view"]["prior_expected_payoffs"]["A"]=serde_json::json!(99);assert!(validate_scenario(&x).is_err());}
    #[test]fn expected_baseline_is_recomputed(){let mut x=fixture("NC-001-no_cue");x["expected"]["baseline_option"]=Value::String("B".into());assert!(validate_scenario(&x).is_err());}
    #[test]fn expected_posterior_is_recomputed(){let mut x=fixture("NC-001-verified_high_value_preview");x["expected"]["posterior_option"]=Value::String("A".into());assert!(validate_scenario(&x).is_err());}
    #[test]fn baseline_regret_is_recomputed(){let mut x=fixture("NC-001-no_cue");x["expected"]["baseline_regret"]=serde_json::json!(0);assert!(validate_scenario(&x).is_err());}
    #[test]fn deferral_uses_null_not_zero_decision_regret(){let x=fixture("NC-001-unsupported_fact_claim");assert!(x["expected"]["decision_regret"].is_null());assert!(validate_scenario(&x).is_ok());}
    #[test]fn deferral_cannot_claim_zero_selection_regret(){let mut x=fixture("NC-001-unsupported_fact_claim");x["expected"]["decision_regret"]=serde_json::json!(0);assert!(validate_scenario(&x).is_err());}
    #[test]fn task_cannot_contain_realized_payoffs(){let mut x=fixture("NC-001-no_cue");x["task"]["realized_payoffs"]=x["oracle"]["realized_payoffs"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn matched_block_oracle_is_immutable(){let mut x=corpus();x["scenarios"][1]["oracle"]["realized_payoffs"]["B"]=serde_json::json!(999);assert!(validate_corpus(&x).is_err());}
    #[test]fn every_block_requires_seven_distinct_arms(){let mut x=corpus();x["scenarios"][1]["cue"]["cue_type"]=Value::String("no_cue".into());x["scenarios"][1]["candidate_view"]["delivered_cue"]=x["scenarios"][1]["cue"].clone();assert!(validate_corpus(&x).is_err());}
    #[test]fn scenario_count_is_frozen(){let mut x=corpus();x["scenarios"].as_array_mut().unwrap().pop();assert!(validate_corpus(&x).is_err());}
    #[test]fn payoff_ties_fail_closed(){let mut x=fixture("NC-001-no_cue");x["oracle"]["realized_payoffs"]["B"]=x["oracle"]["realized_payoffs"]["A"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn high_value_label_requires_argmax_change(){let mut x=fixture("NC-003-verified_low_value_preview");x["cue"]["cue_type"]=Value::String("verified_high_value_preview".into());x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn low_value_label_requires_argmax_invariance(){let mut x=fixture("NC-001-verified_high_value_preview");x["cue"]["cue_type"]=Value::String("verified_low_value_preview".into());x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn evidence_reference_must_match_observation_id(){let mut x=fixture("NC-001-verified_high_value_preview");x["cue"]["evidence_refs"][0]=Value::String("wrong".into());x["candidate_view"]["delivered_cue"]=x["cue"].clone();assert!(validate_scenario(&x).is_err());}
    #[test]fn content_digest_is_deterministic(){let x=fixture("NC-001-no_cue");assert_eq!(digest(&x["candidate_view"]).unwrap(),digest(&x["candidate_view"]).unwrap());}
}
