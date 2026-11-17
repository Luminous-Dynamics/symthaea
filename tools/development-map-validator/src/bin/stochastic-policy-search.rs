//! D6.17.2.2.4-alpha exact finite stochastic policy search.
//!
//! Enumerates finite adaptive trees within the declared option/depth/node and
//! evaluator candidate-count limits, then scores every tree through the
//! authoritative D6.17.2.2.1 scorer. A capped search is never called optimal.

#[allow(dead_code)]
#[path="stochastic-adaptive-policy-scorer.rs"]
mod scorer;

use serde_json::{json,Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap,BTreeSet};
use std::{env,fs,process};

const MAX_OPTIONS:usize=6;
const MAX_NODES:usize=64;
const MAX_POLICY_CAP:usize=50_000;
const MAX_FRONTIER_REPORT:usize=50;
const CLAIM_CEILING:&str="Exact lexicographic search over the finite policy-tree class completely enumerated under the evaluator-owned candidate cap, option/depth limits and 64-node tree limit. Every tree is scored by the D6.17.2.2.1 stochastic policy evaluator. If search_complete is false, no global optimum or exact regret claim is made. It does not optimize beyond the registered finite policy class, establish model completeness or calibration, model stochastic execution costs, handle shared-latent cross-hypothesis ambiguity, or authorize actions.";

#[derive(Clone,Debug,Eq,PartialEq,Ord,PartialOrd)]
enum Policy{
 Stop{action_id:String},
 Acquire{option_id:String,branches:BTreeMap<String,Box<Policy>>}
}
#[derive(Clone,Debug)]
struct Generation{policies:Vec<Policy>,complete:bool}
#[derive(Clone,Debug)]
struct Entry{tree:Value,report:Value,canonical:String}

fn object<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
 v.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn get<'a>(v:&'a Value,key:&str,label:&str)->Result<&'a Value,String>{
 v.get(key).ok_or_else(||format!("{label}.{key} is required"))
}
fn string(v:&Value,key:&str,label:&str)->Result<String,String>{
 get(v,key,label)?.as_str().filter(|x|!x.trim().is_empty()).map(str::to_owned)
  .ok_or_else(||format!("{label}.{key} must be a non-empty string"))
}
fn uint(v:&Value,key:&str,label:&str)->Result<u64,String>{
 get(v,key,label)?.as_u64().ok_or_else(||format!("{label}.{key} must be a non-negative integer"))
}
fn allowed(v:&Value,keys:&[&str],label:&str)->Result<(),String>{
 for key in object(v,label)?.keys(){if !keys.contains(&key.as_str()){return Err(format!("{label} contains unknown field {key:?}"));}}
 Ok(())
}
fn push_limited(out:&mut Vec<Policy>,policy:Policy,cap:usize,complete:&mut bool){
 if out.len()<cap{out.push(policy)}else{*complete=false;}
}
fn cartesian_branches(
 outcomes:&[String],
 child_policies:&[Policy],
 index:usize,
 current:&mut BTreeMap<String,Box<Policy>>,
 output:&mut Vec<Policy>,
 option_id:&str,
 cap:usize,
 complete:&mut bool
){
 if output.len()>=cap{*complete=false;return;}
 if index==outcomes.len(){
  let branches=current.clone();
  output.push(Policy::Acquire{option_id:option_id.to_owned(),branches});
  return;
 }
 for child in child_policies{
  current.insert(outcomes[index].clone(),Box::new(child.clone()));
  cartesian_branches(outcomes,child_policies,index+1,current,output,option_id,cap,complete);
  current.remove(&outcomes[index]);
  if output.len()>=cap{
   if child_policies.last().is_some_and(|last|child!=last)||index+1<outcomes.len(){*complete=false;}
   return;
  }
 }
}
fn enumerate_policies(
 remaining_options:&[String],
 remaining_acquisitions:usize,
 outcomes:&BTreeMap<String,Vec<String>>,
 action_ids:&[String],
 cap:usize
)->Generation{
 let mut generated=Vec::new();let mut complete=true;
 for action in action_ids{
  push_limited(&mut generated,Policy::Stop{action_id:action.clone()},cap,&mut complete);
  if !complete{return Generation{policies:generated,complete:false};}
 }
 if remaining_acquisitions==0{return Generation{policies:generated,complete};}
 for option in remaining_options{
  let child_options:Vec<String>=remaining_options.iter().filter(|x|*x!=option).cloned().collect();
  let children=enumerate_policies(&child_options,remaining_acquisitions-1,outcomes,action_ids,cap);
  if !children.complete{complete=false;}
  if children.policies.is_empty(){continue;}
  let option_outcomes=outcomes.get(option).ok_or_else(||format!("missing registered outcomes for {option:?}"));
  let option_outcomes=match option_outcomes{Ok(x)=>x,Err(_)=>{complete=false;continue;}};
  let mut branches=BTreeMap::new();let mut combinations=Vec::new();
  cartesian_branches(option_outcomes,&children.policies,0,&mut branches,&mut combinations,option,cap,&mut complete);
  for policy in combinations{
   push_limited(&mut generated,policy,cap,&mut complete);
   if !complete{break;}
  }
  if !complete{break;}
 }
 Generation{policies:generated,complete}
}
fn encode_tree(
 policy:&Policy,
 option_times:&BTreeMap<String,String>,
 action_read_only:&BTreeMap<String,bool>,
 nodes:&mut Vec<Value>,
 next_id:&mut u64
)->Result<String,String>{
 let node_id=format!("n{}",*next_id);
 *next_id=(*next_id).checked_add(1).ok_or("policy node ID overflow")?;
 match policy{
  Policy::Stop{action_id}=>{
   let read_only=*action_read_only.get(action_id).ok_or_else(||format!("unknown action {action_id:?}"))?;
   nodes.push(json!({
    "node_id":node_id,"kind":"stop",
    "decision_kind":if read_only{"defer"}else{"act"},
    "action_id":action_id,"claimed_residual_risk_bps":0
   }));
  },
  Policy::Acquire{option_id,branches}=>{
   let at=option_times.get(option_id).ok_or_else(||format!("unknown option {option_id:?}"))?;
   let mut encoded_branches=BTreeMap::new();
   for (outcome,child) in branches{
    let child_id=encode_tree(child,option_times,action_read_only,nodes,next_id)?;
    encoded_branches.insert(outcome.clone(),child_id);
   }
   nodes.push(json!({
    "node_id":node_id,"kind":"acquire","option_id":option_id,"at":at,
    "branches":encoded_branches
   }));
  }
 }
 Ok(node_id)
}
fn as_tree(
 policy:&Policy,
 option_times:&BTreeMap<String,String>,
 action_read_only:&BTreeMap<String,bool>
)->Result<Value,String>{
 let mut nodes=Vec::new();let mut next=0u64;
 let root=encode_tree(policy,option_times,action_read_only,&mut nodes,&mut next)?;
 if nodes.len()>MAX_NODES{return Err(format!("policy has {} nodes; evaluator limit is {MAX_NODES}",nodes.len()));}
 Ok(json!({"root_node_id":root,"nodes":nodes}))
}
fn scoring_bundle(candidate:&Value,oracle:&Value,tree:&Value)->Result<String,String>{
 let mut c=object(candidate,"candidate_view")?.clone();
 if c.contains_key("policy_tree"){return Err("search input must use submitted_policy_tree, not candidate policy_tree".to_owned());}
 c.remove("submitted_policy_tree");
 c.insert("policy_tree".to_owned(),tree.clone());
 let mut o=object(oracle,"oracle")?.clone();o.remove("search_limits");
 serde_json::to_string(&json!({"candidate_view":Value::Object(c),"oracle":Value::Object(o)}))
  .map_err(|e|format!("scoring bundle serialization failed: {e}"))
}
fn metric(report:&Value,path:&[&str])->Result<u64,String>{
 let mut current=report;
 for key in path{current=current.get(*key).ok_or_else(||format!("scorer omitted metric {key:?}"))?;}
 current.as_u64().ok_or_else(||format!("scorer metric {path:?} must be an unsigned integer"))
}
fn feasible(report:&Value)->bool{report["submitted_policy"]["policy_feasible"].as_bool()==Some(true)}
fn better(a:&Entry,b:&Entry)->Ordering{
 let a_feasible=feasible(&a.report);let b_feasible=feasible(&b.report);
 a_feasible.cmp(&b_feasible)
  .then_with(||metric(&a.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0)
   .cmp(&metric(&b.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0)))
  .then_with(||metric(&a.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0)
   .cmp(&metric(&b.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0)))
  .then_with(||metric(&b.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(u64::MAX)
   .cmp(&metric(&a.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(u64::MAX)))
  .then_with(||metric(&b.report,&["expected_cost","nominal_expected_cost_micros"]).unwrap_or(u64::MAX)
   .cmp(&metric(&a.report,&["expected_cost","nominal_expected_cost_micros"]).unwrap_or(u64::MAX)))
  .then_with(||metric(&b.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(u64::MAX)
   .cmp(&metric(&a.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(u64::MAX)))
  .then_with(||b.canonical.cmp(&a.canonical))
}
fn dominates(a:&Entry,b:&Entry)->bool{
 if !feasible(&a.report){return false;}
 let ar=metric(&a.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0);
 let br=metric(&b.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0);
 let an=metric(&a.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0);
 let bn=metric(&b.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0);
 let ac=metric(&a.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(u64::MAX);
 let bc=metric(&b.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(u64::MAX);
 let ap=metric(&a.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(u64::MAX);
 let bp=metric(&b.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(u64::MAX);
 ar>=br&&an>=bn&&ac<=bc&&ap<=bp&&(ar>br||an>bn||ac<bc||ap<bp)
}
fn evaluate(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=get(&bundle,"candidate_view","bundle")?;
 let oracle=get(&bundle,"oracle","bundle")?;
 allowed(candidate,&["scenario_id","decision_cutoff","budget_micros","max_depth","actions","hypothesis_bounds","options","factors","submitted_policy_tree"],"candidate_view")?;
 allowed(oracle,&["actual_hypothesis_id","decision_cutoff","budget_micros","actions","hypothesis_bounds","options","factors","hypothesis_safety","calibrations","world_kernels","cost_options","billing_currency","search_limits"],"oracle")?;
 let limits=get(oracle,"search_limits","oracle")?;
 allowed(limits,&["max_generated_policies"],"oracle.search_limits")?;
 let cap=uint(limits,"max_generated_policies","oracle.search_limits")? as usize;
 if cap==0||cap>MAX_POLICY_CAP{return Err(format!("max_generated_policies must be in 1..={MAX_POLICY_CAP}"));}
 let max_depth=uint(candidate,"max_depth","candidate_view")? as usize;
 if max_depth==0||max_depth>5{return Err("max_depth must be in 1..=5".to_owned());}
 let decision_cutoff=string(candidate,"decision_cutoff","candidate_view")?;
 let options_raw=get(candidate,"options","candidate_view")?.as_array().ok_or("candidate options must be array")?;
 if options_raw.is_empty()||options_raw.len()>MAX_OPTIONS{return Err("search candidate must contain 1..=6 options".to_owned());}
 let mut option_times=BTreeMap::new();let mut option_factors=BTreeMap::new();
 for (i,row) in options_raw.iter().enumerate(){
  let label=format!("candidate_view.options[{i}]");
  let id=string(row,"option_id",&label)?;
  let _available_at=string(row,"available_at",&label)?;
  let factor=string(row,"factor_id",&label)?;
  // Search scores policy order, not wall-clock scheduling. Every selected
  // measurement is represented as available at the registered decision
  // cutoff, so a later tree depth never receives a regressing timestamp.
  if option_times.insert(id.clone(),decision_cutoff.clone()).is_some(){return Err(format!("duplicate option {id:?}"));}
  option_factors.insert(id,factor);
 }
 let factor_rows=get(candidate,"factors","candidate_view")?.as_array().ok_or("candidate factors must be array")?;
 let mut outcomes=BTreeMap::new();
 for factor in factor_rows{
  let factor_id=string(factor,"factor_id","candidate factor")?;
  let likelihoods=get(factor,"likelihoods","candidate factor")?.as_array().ok_or("factor likelihoods must be array")?;
  let first=likelihoods.first().ok_or("factor likelihoods cannot be empty")?;
  let names=object(get(first,"outcomes","factor likelihood")?,"factor outcomes")?.keys().cloned().collect::<Vec<_>>();
  outcomes.insert(factor_id,names);
 }
 let mut option_outcomes=BTreeMap::new();
 for (id,factor) in &option_factors{
  option_outcomes.insert(id.clone(),outcomes.get(factor).cloned().ok_or_else(||format!("missing factor outcomes for {factor:?}"))?);
 }
 let mut action_read_only=BTreeMap::new();let mut action_ids=Vec::new();
 for (i,action) in get(candidate,"actions","candidate_view")?.as_array().ok_or("actions must be array")?.iter().enumerate(){
  let label=format!("candidate_view.actions[{i}]");
  let id=string(action,"action_id",&label)?;
  let read_only=get(action,"read_only",&label)?.as_bool().ok_or("action read_only must be boolean")?;
  if action_read_only.insert(id.clone(),read_only).is_some(){return Err(format!("duplicate action {id:?}"));}
  action_ids.push(id);
 }
 if action_ids.is_empty(){return Err("search requires at least one action".to_owned());}
 action_ids.sort();
 let option_ids:Vec<String>=option_times.keys().cloned().collect();
 // The scorer counts root depth as zero; a stop child at depth N
 // therefore permits N acquisition nodes along its path.
 let max_acquisitions=max_depth;
 let generation=enumerate_policies(&option_ids,max_acquisitions,&option_outcomes,&action_ids,cap);
 let mut entries=Vec::new();let mut invalid_node_count=0usize;
 for policy in &generation.policies{
  let tree=match as_tree(policy,&option_times,&action_read_only){
   Ok(x)=>x,Err(_)=>{invalid_node_count+=1;continue;}
  };
  let scoring_raw=scoring_bundle(candidate,oracle,&tree)?;
  let report=scorer::score_stochastic_adaptive_policy_json(&scoring_raw)
   .map_err(|e|format!("generated policy failed scorer validation: {e}"))?;
  let canonical=serde_json::to_string(&tree).map_err(|e|format!("tree canonicalization failed: {e}"))?;
  entries.push(Entry{tree,report,canonical});
 }
 if entries.is_empty(){return Err("no generated policy fits the evaluator tree limits".to_owned());}
 let submitted_tree=get(candidate,"submitted_policy_tree","candidate_view")?.clone();
 let baseline_raw=scoring_bundle(candidate,oracle,&submitted_tree)?;
 let baseline_report=scorer::score_stochastic_adaptive_policy_json(&baseline_raw)
  .map_err(|e|format!("submitted policy failed scorer validation: {e}"))?;
 let baseline_canonical=serde_json::to_string(&submitted_tree).map_err(|e|format!("submitted tree serialization failed: {e}"))?;
 let baseline=Entry{tree:submitted_tree,report:baseline_report,canonical:baseline_canonical};
 entries.sort_by(|a,b|better(b,a));
 let best=entries.first().ok_or("best policy missing")?;
 let feasible_count=entries.iter().filter(|x|feasible(&x.report)).count();
 let mut frontier:Vec<&Entry>=entries.iter().filter(|x|feasible(&x.report)).collect();
 frontier.retain(|candidate_entry|{
  !entries.iter().any(|other|other.canonical!=candidate_entry.canonical&&dominates(other,candidate_entry))
 });
 frontier.sort_by(|a,b|better(b,a));
 let frontier_truncated=frontier.len()>MAX_FRONTIER_REPORT;
 let frontier_report:Vec<Value>=frontier.iter().take(MAX_FRONTIER_REPORT).map(|entry|json!({
  "robust_coverage_lower_bps":metric(&entry.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0),
  "nominal_coverage_bps":metric(&entry.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0),
  "worst_case_expected_cost_micros":metric(&entry.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(0),
  "nominal_expected_cost_micros":metric(&entry.report,&["expected_cost","nominal_expected_cost_micros"]).unwrap_or(0),
  "worst_case_path_cost_micros":metric(&entry.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(0),
  "policy_tree":entry.tree
 })).collect();
 let complete=generation.complete;
 let regret=if complete{
  Some(json!({
   "robust_coverage_shortfall_bps":metric(&best.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0)
    .saturating_sub(metric(&baseline.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"]).unwrap_or(0)),
   "nominal_coverage_shortfall_bps":metric(&best.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0)
    .saturating_sub(metric(&baseline.report,&["submitted_policy","nominal_justified_action_coverage_bps"]).unwrap_or(0)),
   "worst_case_cost_excess_micros":metric(&baseline.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(0)
    .saturating_sub(metric(&best.report,&["expected_cost","worst_case_expected_cost_micros"]).unwrap_or(0)),
   "nominal_cost_excess_micros":metric(&baseline.report,&["expected_cost","nominal_expected_cost_micros"]).unwrap_or(0)
    .saturating_sub(metric(&best.report,&["expected_cost","nominal_expected_cost_micros"]).unwrap_or(0)),
   "worst_path_cost_excess_micros":metric(&baseline.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(0)
    .saturating_sub(metric(&best.report,&["submitted_policy","worst_case_path_cost_micros"]).unwrap_or(0))
  }))
 }else{None};
 Ok(json!({
  "adjudication_status":"finite_stochastic_policy_search",
  "search_complete":complete,
  "optimality_claim":if complete{"exact_optimum_over_registered_finite_tree_class"}else{"none_search_incomplete"},
  "objective_order":[
   "feasible_only","maximize_robust_lower_justified_coverage","maximize_nominal_justified_coverage",
   "minimize_worst_case_expected_cost","minimize_nominal_expected_cost","minimize_worst_case_path_cost","canonical_tree_order"
  ],
  "generated_policy_count":generation.policies.len(),
  "evaluated_policy_count":entries.len(),
  "invalid_node_limit_policy_count":invalid_node_count,
  "feasible_policy_count":feasible_count,
  "evaluator_candidate_cap":cap,
  "pareto_frontier_count":frontier.len(),
  "pareto_frontier_truncated":frontier_truncated,
  "pareto_frontier":frontier_report,
  "best_policy":{
   "policy_feasible":feasible(&best.report),
   "robust_coverage_lower_bps":metric(&best.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"])?,
   "nominal_coverage_bps":metric(&best.report,&["submitted_policy","nominal_justified_action_coverage_bps"])?,
   "worst_case_expected_cost_micros":metric(&best.report,&["expected_cost","worst_case_expected_cost_micros"])?,
   "nominal_expected_cost_micros":metric(&best.report,&["expected_cost","nominal_expected_cost_micros"])?,
   "worst_case_path_cost_micros":metric(&best.report,&["submitted_policy","worst_case_path_cost_micros"])?,
   "policy_tree":best.tree,
   "scorer_report":best.report
  },
  "submitted_policy":{
   "policy_feasible":feasible(&baseline.report),
   "robust_coverage_lower_bps":metric(&baseline.report,&["submitted_policy","justified_action_coverage_bounds_bps","lower"])?,
   "nominal_coverage_bps":metric(&baseline.report,&["submitted_policy","nominal_justified_action_coverage_bps"])?,
   "worst_case_expected_cost_micros":metric(&baseline.report,&["expected_cost","worst_case_expected_cost_micros"])?,
   "nominal_expected_cost_micros":metric(&baseline.report,&["expected_cost","nominal_expected_cost_micros"])?,
   "worst_case_path_cost_micros":metric(&baseline.report,&["submitted_policy","worst_case_path_cost_micros"])?,
   "scorer_report":baseline.report
  },
  "exact_regret":regret,
  "claim_ceiling":CLAIM_CEILING
 }))
}
pub fn search_stochastic_policy_json(raw:&str)->Result<Value,String>{evaluate(raw)}
fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{
  eprintln!("Usage: stochastic-policy-search <candidate-and-oracle-bundle.json>");
  process::exit(if args.len()==1{0}else{2});
 }
 let raw=match fs::read_to_string(&args[0]){Ok(x)=>x,Err(e)=>{eprintln!("ERROR D6.17.2.2.4-IO-001 {}: {e}",args[0]);process::exit(2)}};
 match evaluate(&raw){
  Ok(report)=>match serde_json::to_string_pretty(&report){Ok(output)=>println!("{output}"),Err(e)=>{eprintln!("ERROR D6.17.2.2.4-JSON-001: {e}");process::exit(2)}},
  Err(e)=>{eprintln!("ERROR D6.17.2.2.4-EVALUATION-001: {e}");process::exit(1)}
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 const FIXTURE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-search-deterministic-v0.1.json");
 const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-search-deterministic-v0.1.expected.json");
 fn report()->Value{evaluate(FIXTURE).expect("exact policy search fixture validates")}
 #[test]fn report_matches_exact_search_known_answer(){
  let actual=report();
  let expected:Value=serde_json::from_str(EXPECTED).expect("expected report");
  for key in ["adjudication_status","search_complete","optimality_claim","generated_policy_count","evaluated_policy_count","invalid_node_limit_policy_count"]{
   assert_eq!(actual[key],expected[key], "report field {key}");
  }
  assert_eq!(actual["best_policy"]["robust_coverage_lower_bps"],expected["best_policy"]["robust_coverage_lower_bps"]);
  assert_eq!(actual["best_policy"]["nominal_coverage_bps"],expected["best_policy"]["nominal_coverage_bps"]);
  assert_eq!(actual["best_policy"]["worst_case_expected_cost_micros"],expected["best_policy"]["worst_case_expected_cost_micros"]);
  assert_eq!(actual["exact_regret"],expected["exact_regret"]);
 }
 #[test]fn complete_search_claims_optimality_only_when_all_policies_evaluated(){
  let r=report();
  assert_eq!(r["search_complete"],true);
  assert_eq!(r["optimality_claim"],"exact_optimum_over_registered_finite_tree_class");
  assert_eq!(r["evaluated_policy_count"],74);
  assert_eq!(r["generated_policy_count"],74);
 }
 #[test]fn exact_search_does_not_lose_deterministic_adaptive_baseline(){
  let r=report();
  assert_eq!(r["best_policy"]["robust_coverage_lower_bps"],5000);
  assert_eq!(r["best_policy"]["nominal_coverage_bps"],6000);
  assert_eq!(r["best_policy"]["worst_case_expected_cost_micros"],1420);
 }
 #[test]fn capped_search_never_claims_optimality(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["search_limits"]["max_generated_policies"]=json!(10);
  let r=evaluate(&b.to_string()).expect("partial search returns report");
  assert_eq!(r["search_complete"],false);
  assert_eq!(r["optimality_claim"],"none_search_incomplete");
  assert_eq!(r["exact_regret"],Value::Null);
 }
 #[test]fn candidate_cannot_raise_evaluator_search_cap(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["candidate_view"]["search_limits"]=json!({"max_generated_policies":50000});
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));
 }
 #[test]fn node_limit_exclusions_are_counted(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["search_limits"]["max_generated_policies"]=json!(10000);
  let r=evaluate(&b.to_string()).expect("search report");
  assert_eq!(r["invalid_node_limit_policy_count"],0);
 }
}
