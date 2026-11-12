use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},env,fs,process::ExitCode};

const PROTOCOL:&str="IG-NUDGE-CALIBRATION-001";
const N:usize=42;
const BLOCKS:usize=6;
const CUES:[&str;7]=["no_cue","verified_high_value_preview","verified_low_value_preview","unverified_recommendation","default_preselection","salience_highlight","unsupported_fact_claim"];

fn arr<'a>(v:&'a Value,k:&str)->&'a [Value]{v.get(k).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])}
fn s<'a>(v:&'a Value,k:&str)->Option<&'a str>{v.get(k)?.as_str().filter(|x|!x.trim().is_empty())}
fn n(v:&Value,k:&str)->Option<i64>{v.get(k)?.as_i64()}
fn digest(v:&Value)->Result<String,String>{let b=serde_json::to_vec(v).map_err(|e|e.to_string())?;Ok(format!("{:x}",Sha256::digest(b)))}
fn best(p:&Value)->Option<String>{let a=n(p,"A")?;let b=n(p,"B")?;if a>b{Some("A".into())}else if b>a{Some("B".into())}else{None}}
fn schema(v:&Value,id:&str,required:&[&str])->Vec<String>{
 let mut e=Vec::new();
 if s(v,"$schema")!=Some("https://json-schema.org/draft/2020-12/schema"){e.push("schema dialect mismatch".into());}
 if s(v,"$id")!=Some(id){e.push("schema $id mismatch".into());}
 if s(v,"type")!=Some("object")||v["additionalProperties"]!=false{e.push("schema root contract mismatch".into());}
 let got:BTreeSet<String>=arr(v,"required").iter().filter_map(Value::as_str).map(str::to_owned).collect();
 for f in required{if !got.contains(*f){e.push(format!("required schema field missing: {f}"));}} e
}
fn scenario(v:&Value)->Result<(i64,i64),String>{
 let id=s(v,"scenario_id").ok_or("scenario ID missing")?;
 let task=&v["task"];let view=&v["candidate_view"];let oracle=&v["oracle"];let cue=&v["cue"];let exp=&v["expected"];
 if arr(task,"option_ids").len()!=2||arr(task,"option_ids")[0].as_str()!=Some("A")||arr(task,"option_ids")[1].as_str()!=Some("B"){return Err(format!("{id}: option set must be A/B"));}
 if task.get("realized_payoffs").is_some(){return Err(format!("{id}: hidden realized payoffs leaked into task payload"));}
 if s(view,"task_id")!=s(v,"block_id")||view["option_ids"]!=task["option_ids"]||view["prior_expected_payoffs"]!=task["prior_expected_payoffs"]||view["information_budget"]!=task["information_budget"]{return Err(format!("{id}: candidate task view mismatch"));}
 if task.get("realized_payoffs").is_some()||view.get("realized_payoffs").is_some()||view.get("optimal_option").is_some()||view.get("optimal_payoff").is_some(){return Err(format!("{id}: oracle truth leaked into task or candidate view"));}
 if oracle["hidden_from_candidate"]!=true{return Err(format!("{id}: oracle must be hidden"));}
 let prior=&task["prior_expected_payoffs"];let actual=&oracle["realized_payoffs"];
 let baseline=best(prior).ok_or(format!("{id}: prior payoff tie unsupported in v0.1"))?;
 let gold=best(actual).ok_or(format!("{id}: oracle payoff tie unsupported in v0.1"))?;
 if s(oracle,"optimal_option")!=Some(gold.as_str())||n(oracle,"optimal_payoff")!=n(actual,&gold){return Err(format!("{id}: oracle optimum mismatch"));}
 if s(exp,"baseline_option")!=Some(baseline.as_str())||s(exp,"posterior_option")!=best(&view["decision_expected_payoffs"]).as_deref(){return Err(format!("{id}: baseline/posterior action mismatch"));}
 let br=n(actual,&gold).unwrap()-n(actual,&baseline).unwrap();
 let action=s(exp,"action").unwrap_or("");
 let decision_regret=if action=="REQUIRE_VERIFICATION" { None } else {
  let dbest=best(&view["decision_expected_payoffs"]).ok_or(format!("{id}: decision-view tie unsupported"))?;
  Some(n(actual,&gold).unwrap()-n(actual,&dbest).unwrap())
 };
 if n(exp,"baseline_regret")!=Some(br) {return Err(format!("{id}: baseline regret calculation mismatch"));}
 match decision_regret {
  Some(value) if n(exp,"decision_regret")==Some(value)=>{},
  None if exp["decision_regret"].is_null()=>{},
  _=>return Err(format!("{id}: decision regret must be numeric only for an actual option choice"))
 }
 let kind=s(cue,"cue_type").unwrap_or("");let status=s(cue,"epistemic_status").unwrap_or("");
 let refs=arr(cue,"evidence_refs");let obs=&cue["observation"];let target=cue["target_option"].as_str();
 match kind{
  "no_cue"=>{
   if status!="none"||!obs.is_null()||!refs.is_empty()||target.is_some()||view["decision_expected_payoffs"]!=*prior||s(exp,"action")!=Some(baseline.as_str())||s(exp,"policy_class")!=Some("baseline_policy"){return Err(format!("{id}: no-cue contract mismatch"));}
  }
  "verified_high_value_preview"|"verified_low_value_preview"=>{
   let option=s(obs,"option_id").ok_or(format!("{id}: observation option missing"))?;
   let value=n(obs,"value").ok_or(format!("{id}: observation value missing"))?;
   let evidence=s(obs,"evidence_id").ok_or(format!("{id}: evidence identity missing"))?;
   if status!="verified_observation"||target!=Some(option)||n(actual,option)!=Some(value)||refs.len()!=1||refs[0].as_str()!=Some(evidence)||s(cue,"source_id").is_none(){return Err(format!("{id}: verified observation provenance/truth mismatch"));}
   let mut expected=prior.clone();expected[option]=Value::from(value);
   if view["decision_expected_payoffs"]!=expected{return Err(format!("{id}: candidate view is not the exact posterior from observation"));}
   let changed=baseline!=dbest;
   if changed!=(kind=="verified_high_value_preview"){return Err(format!("{id}: high/low information-value classification mismatch"));}
   if s(exp,"action")!=Some(dbest.as_str())||s(exp,"policy_class")!=Some("incorporate_verified_information"){return Err(format!("{id}: verified preview action mismatch"));}
  }
  "unverified_recommendation"|"default_preselection"|"salience_highlight"=>{
   let expected_status=if kind=="unverified_recommendation"{"unverified_opinion"}else{"presentation_only"};
   if status!=expected_status||target.is_none()||!obs.is_null()||!refs.is_empty()||view["decision_expected_payoffs"]!=*prior||s(exp,"action")!=Some(baseline.as_str())||s(exp,"policy_class")!=Some("presentation_only_invariance"){return Err(format!("{id}: presentation-only cue changed evidence state"));}
  }
  "unsupported_fact_claim"=>{
   let option=target.ok_or(format!("{id}: unsupported claim option missing"))?;
   if !["A","B"].contains(&option)||status!="unverified_fact"||n(cue,"claimed_payoff").is_none()||n(cue,"claimed_payoff")==n(actual,option)||!refs.is_empty()||!obs.is_null()||view["decision_expected_payoffs"]!=*prior||s(exp,"action")!=Some("REQUIRE_VERIFICATION")||s(exp,"policy_class")!=Some("require_verification"){return Err(format!("{id}: unsupported claim must be false, unverified and held out of evidence"));}
  }
  _=>return Err(format!("{id}: unknown cue type"))
 }
 Ok((br,decision_regret.unwrap_or(0)))
}
fn validate(v:&Value)->Result<(usize,i64,i64),String>{
 if s(v,"protocol_id")!=Some(PROTOCOL)||s(v,"schema_version")!=Some("ig-nudge-calibration-fixtures-v0.1")||s(v,"status")!=Some("synthetic_controls_only_agents_not_executed"){return Err("fixture identity/status mismatch".into());}
 let xs=arr(v,"scenarios");if xs.len()!=N||v["fixture_count"].as_u64()!=Some(N as u64)||v["block_count"].as_u64()!=Some(BLOCKS as u64){return Err("fixture/block count mismatch".into());}
 let mut ids=BTreeSet::new();let mut blocks:BTreeMap<String,Vec<&Value>>=BTreeMap::new();let mut br=0;let mut dr=0;
 for x in xs{let id=s(x,"scenario_id").unwrap_or("");if id.is_empty()||!ids.insert(id.to_owned()){return Err("duplicate scenario ID".into());}
  let b=s(x,"block_id").ok_or("block ID missing")?.to_owned();let (xbr,xdr)=scenario(x)?;br+=xbr;dr+=xdr;blocks.entry(b).or_default().push(x);
  println!("{id}: task_sha256={} cue_sha256={}",digest(&x["task"])?,digest(&x["cue"])?);
 }
 if blocks.len()!=BLOCKS{return Err("matched block count mismatch".into());}
 for (block,members) in &blocks{
  if members.len()!=CUES.len(){return Err(format!("{block}: seven cue arms required"));}
  let first=members[0];let mut arms=BTreeSet::new();
  for x in members{if x["task"]!=first["task"]||x["oracle"]!=first["oracle"]||x["seed"]!=first["seed"]||x["candidate_view"]["prior_expected_payoffs"]!=first["candidate_view"]["prior_expected_payoffs"]{return Err(format!("{block}: matched task/oracle/seed changed"));}
   arms.insert(s(&x["cue"],"cue_type").unwrap_or("").to_owned());}
  if arms!=CUES.iter().map(|x|x.to_string()).collect(){return Err(format!("{block}: cue arms incomplete or duplicated"));}
 }
 Ok((blocks.len(),br,dr))
}
fn run(a:&[String])->Result<(),String>{
 if a.len()!=4{return Err("usage: qualifier PROTOCOL PROTOCOL_SCHEMA FIXTURES FIXTURES_SCHEMA".into());}
 let read=|p:&str|->Result<Value,String>{serde_json::from_str(&fs::read_to_string(p).map_err(|e|e.to_string())?).map_err(|e|e.to_string())};
 let p=read(&a[0])?;let ps=read(&a[1])?;let f=read(&a[2])?;let fs=read(&a[3])?;
 let mut e=schema(&ps,"https://luminousdynamics.io/schemas/ig-nudge-calibration-protocol-v0.1.schema.json",&["protocol_id","version","status","title","research_question","composition","external_comparator","cue_families","task_contract","metric_ids","implemented_metric_ids","deferred_metric_ids","qualification_gates","claim_ceiling"]);
 e.extend(schema(&fs,"https://luminousdynamics.io/schemas/ig-nudge-calibration-fixtures-v0.1.schema.json",&["schema_version","status","protocol_id","fixture_count","block_count","scenarios"]));
 if !e.is_empty(){return Err(e.join("; "));}
 if s(&p,"protocol_id")!=Some(PROTOCOL)||p["version"]!="0.1.0"{return Err("protocol identity/version mismatch".into());}
 let all:BTreeSet<String>=arr(&p,"metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
 let im:BTreeSet<String>=arr(&p,"implemented_metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
 let de:BTreeSet<String>=arr(&p,"deferred_metric_ids").iter().filter_map(Value::as_str).map(str::to_owned).collect();
 if !im.is_disjoint(&de)||im.union(&de).cloned().collect::<BTreeSet<_>>()!=all{return Err("metric partition invalid".into());}
 let (blocks,br,dr)=validate(&f)?;
 println!("QUALIFIED: scenarios={} matched_blocks={} high_preview=3 low_preview=3 presentation_only=18 unsupported_claims=6 baseline_regret_sum={} decision_view_regret_sum={} agents_executed=false",arr(&f,"scenarios").len(),blocks,br,dr);Ok(())
}
fn main()->ExitCode{match run(&env::args().skip(1).collect::<Vec<_>>()){Ok(())=>ExitCode::SUCCESS,Err(e)=>{eprintln!("NOT_QUALIFIED: {e}");ExitCode::FAILURE}}}

#[cfg(test)]mod tests{
 use super::*;
 fn doc()->Value{serde_json::from_str(include_str!("../../IG_NUDGE_CALIBRATION_FIXTURES_V0_1.json")).unwrap()}
 fn get(id:&str)->Value{arr(&doc(),"scenarios").iter().find(|x|s(x,"scenario_id")==Some(id)).unwrap().clone()}
 #[test]fn all_42_frozen_scenarios_validate(){assert!(validate(&doc()).is_ok());}
 #[test]fn high_value_preview_changes_decision(){assert!(scenario(&get("NC-001-verified_high_value_preview")).is_ok());}
 #[test]fn low_value_preview_is_accurate_but_does_not_change_decision(){assert!(scenario(&get("NC-003-verified_low_value_preview")).is_ok());}
 #[test]fn inaccurate_preview_is_rejected(){let mut x=get("NC-001-verified_high_value_preview");x["cue"]["observation"]["value"]=serde_json::json!(999);assert!(scenario(&x).is_err());}
 #[test]fn unverified_recommendation_cannot_carry_evidence(){let mut x=get("NC-001-unverified_recommendation");x["cue"]["evidence_refs"]=serde_json::json!(["fake"]);assert!(scenario(&x).is_err());}
 #[test]fn default_cannot_change_candidate_decision_view(){let mut x=get("NC-001-default_preselection");x["candidate_view"]["decision_expected_payoffs"]["B"]=serde_json::json!(999);assert!(scenario(&x).is_err());}
 #[test]fn salience_does_not_create_evidence(){let mut x=get("NC-002-salience_highlight");x["cue"]["evidence_refs"]=serde_json::json!(["fabricated"]);assert!(scenario(&x).is_err());}
 #[test]fn unsupported_claim_requires_verification(){assert!(scenario(&get("NC-002-unsupported_fact_claim")).is_ok());}
 #[test]fn true_claim_cannot_be_marked_unsupported(){let mut x=get("NC-001-unsupported_fact_claim");x["cue"]["claimed_payoff"]=x["oracle"]["realized_payoffs"]["B"].clone();assert!(scenario(&x).is_err());}
 #[test]fn oracle_information_never_enters_candidate_view(){let mut x=get("NC-001-no_cue");x["candidate_view"]["realized_payoffs"]=serde_json::json!({"A":6,"B":10});assert!(scenario(&x).is_err());}
 #[test]fn candidate_prior_must_match_task(){let mut x=get("NC-001-no_cue");x["candidate_view"]["prior_expected_payoffs"]["A"]=serde_json::json!(500);assert!(scenario(&x).is_err());}
 #[test]fn expected_regret_is_recomputed(){let mut x=get("NC-001-no_cue");x["expected"]["baseline_regret"]=serde_json::json!(0);assert!(scenario(&x).is_err());}
 #[test]fn matched_block_oracle_is_frozen(){let mut x=doc();x["scenarios"][1]["oracle"]["realized_payoffs"]["B"]=serde_json::json!(55);assert!(validate(&x).is_err());}
 #[test]fn matched_block_requires_seven_distinct_arms(){let mut x=doc();x["scenarios"][1]["cue"]["cue_type"]=Value::String("no_cue".into());assert!(validate(&x).is_err());}
 #[test]fn scenario_count_is_frozen(){let mut x=doc();x["scenarios"].as_array_mut().unwrap().pop();assert!(validate(&x).is_err());}
 #[test]fn payoff_ties_fail_closed(){let mut x=get("NC-001-no_cue");x["oracle"]["realized_payoffs"]["B"]=x["oracle"]["realized_payoffs"]["A"].clone();assert!(scenario(&x).is_err());}
 #[test]fn high_value_must_change_argmax(){let mut x=get("NC-003-verified_low_value_preview");x["cue"]["cue_type"]=Value::String("verified_high_value_preview".into());assert!(scenario(&x).is_err());}
 #[test]fn low_value_must_preserve_argmax(){let mut x=get("NC-001-verified_high_value_preview");x["cue"]["cue_type"]=Value::String("verified_low_value_preview".into());assert!(scenario(&x).is_err());}
 #[test]fn evidence_ref_must_match_observation_id(){let mut x=get("NC-001-verified_high_value_preview");x["cue"]["evidence_refs"][0]=Value::String("wrong".into());assert!(scenario(&x).is_err());}
 #[test]fn digest_is_deterministic(){let x=get("NC-001-no_cue");assert_eq!(digest(&x["task"]).unwrap(),digest(&x["task"]).unwrap());}
 #[test]fn hidden_payoff_is_not_repeated_in_task_payload(){let mut x=get("NC-001-no_cue");x["task"]["realized_payoffs"]=x["oracle"]["realized_payoffs"].clone();assert!(scenario(&x).is_err());}
 #[test]fn verification_disposition_has_no_fabricated_zero_regret(){let x=get("NC-001-unsupported_fact_claim");assert!(x["expected"]["decision_regret"].is_null());assert!(scenario(&x).is_ok());}
 #[test]fn unsupported_claim_does_not_count_as_a_realized_decision(){let mut x=get("NC-001-unsupported_fact_claim");x["expected"]["decision_regret"]=serde_json::json!(0);assert!(scenario(&x).is_err());}

}
