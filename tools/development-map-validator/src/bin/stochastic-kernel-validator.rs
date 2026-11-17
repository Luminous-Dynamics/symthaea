//! D6.17.2.2 stochastic joint-outcome credal-kernel validator.
//!
//! Validates evaluator-owned bounded-simplex distributions over complete
//! multi-channel outcome vectors and proves that each registered D6.17
//! per-channel likelihood interval conservatively contains the induced
//! marginal interval. Joint rows preserve cross-channel dependence; no
//! independence assumption is made by this validator.

use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet},env,fs,process};

const SCALE:u64=10_000;
const MAX_OPTIONS:usize=6;
const MAX_OUTCOMES_PER_OPTION:usize=4;
const MAX_JOINT_ROWS:usize=4_096;
const MAX_HYPOTHESES:usize=64;

#[derive(Clone,Debug)]
struct Probability{lower:u64,upper:u64,nominal:u64}
#[derive(Clone,Debug)]
struct JointRow{outcomes:BTreeMap<String,String>,probability:Probability}
#[derive(Clone,Debug)]
struct Kernel{hypothesis_id:String,kernel_id:String,authority:String,valid_from:String,valid_until:String,revoked_at:Option<String>,rows:Vec<JointRow>}
#[derive(Clone,Debug)]
struct Factor{outcomes:BTreeSet<String>,likelihoods:BTreeMap<String,BTreeMap<String,Probability>>}
#[derive(Clone,Debug)]
struct Marginal{probability:Probability}
#[derive(Clone,Debug)]
struct KernelModel{scenario_id:String,decision_cutoff:String,options:BTreeMap<String,String>,factors:BTreeMap<String,Factor>,kernels:BTreeMap<String,Kernel>}

fn obj<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
 v.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn allowed(v:&Value,keys:&[&str],label:&str)->Result<(),String>{
 for key in obj(v,label)?.keys(){if !keys.contains(&key.as_str()){return Err(format!("{label} contains unknown field {key:?}"));}}
 Ok(())
}
fn get<'a>(v:&'a Value,key:&str,label:&str)->Result<&'a Value,String>{
 v.get(key).ok_or_else(||format!("{label}.{key} is required"))
}
fn string(v:&Value,key:&str,label:&str)->Result<String,String>{
 get(v,key,label)?.as_str().filter(|x|!x.trim().is_empty()).map(str::to_owned)
  .ok_or_else(||format!("{label}.{key} must be a non-empty string"))
}
fn integer(v:&Value,key:&str,label:&str)->Result<u64,String>{
 get(v,key,label)?.as_u64().ok_or_else(||format!("{label}.{key} must be a non-negative integer"))
}
fn time(raw:&str,label:&str)->Result<chrono::DateTime<chrono::FixedOffset>,String>{
 chrono::DateTime::parse_from_rfc3339(raw).map_err(|e|format!("{label} must be RFC 3339: {e}"))
}
fn parse_probability(v:&Value,label:&str)->Result<Probability,String>{
 allowed(v,&["lower_probability_bps","upper_probability_bps","nominal_probability_bps"],label)?;
 let p=Probability{
  lower:integer(v,"lower_probability_bps",label)?,
  upper:integer(v,"upper_probability_bps",label)?,
  nominal:integer(v,"nominal_probability_bps",label)?
 };
 if p.lower>p.nominal||p.nominal>p.upper||p.upper>SCALE{
  return Err(format!("{label} must satisfy 0 <= lower <= nominal <= upper <= 10000"));
 }
 Ok(p)
}
fn parse_likelihood(v:&Value,label:&str)->Result<Probability,String>{
 allowed(v,&["lower_likelihood_bps","upper_likelihood_bps","nominal_likelihood_bps"],label)?;
 let p=Probability{
  lower:integer(v,"lower_likelihood_bps",label)?,
  upper:integer(v,"upper_likelihood_bps",label)?,
  nominal:integer(v,"nominal_likelihood_bps",label)?
 };
 if p.lower>p.nominal||p.nominal>p.upper||p.upper>SCALE{
  return Err(format!("{label} must satisfy 0 <= lower <= nominal <= upper <= 10000"));
 }
 Ok(p)
}
fn validate_partition(rows:&[JointRow],label:&str)->Result<(),String>{
 if rows.is_empty()||rows.len()>MAX_JOINT_ROWS{return Err(format!("{label} must contain 1..={MAX_JOINT_ROWS} joint rows"));}
 let lower:u64=rows.iter().map(|r|r.probability.lower).sum();
 let upper:u64=rows.iter().map(|r|r.probability.upper).sum();
 let nominal:u64=rows.iter().map(|r|r.probability.nominal).sum();
 if lower>SCALE||upper<SCALE||nominal!=SCALE{
  return Err(format!("{label} is not a non-empty bounded simplex: lower_sum={lower}, upper_sum={upper}, nominal_sum={nominal}"));
 }
 Ok(())
}
fn event_bounds(rows:&[JointRow],event:impl Fn(&JointRow)->bool)->Result<Probability,String>{
 let mut lower_event=0u64;let mut upper_event=0u64;let mut lower_other=0u64;let mut upper_other=0u64;let mut nominal_event=0u64;
 for row in rows{
  if event(row){
   lower_event=lower_event.checked_add(row.probability.lower).ok_or("event lower sum overflow")?;
   upper_event=upper_event.checked_add(row.probability.upper).ok_or("event upper sum overflow")?;
   nominal_event=nominal_event.checked_add(row.probability.nominal).ok_or("event nominal sum overflow")?;
  }else{
   lower_other=lower_other.checked_add(row.probability.lower).ok_or("complement lower sum overflow")?;
   upper_other=upper_other.checked_add(row.probability.upper).ok_or("complement upper sum overflow")?;
  }
 }
 let lower=lower_event.max(SCALE.saturating_sub(upper_other));
 let upper=upper_event.min(SCALE.saturating_sub(lower_other));
 if lower>upper||nominal_event<lower||nominal_event>upper{
  return Err("derived event probability interval is infeasible".to_owned());
 }
 Ok(Probability{lower,upper,nominal:nominal_event})
}
fn parse_candidate(v:&Value)->Result<(String,String,BTreeMap<String,String>,BTreeMap<String,Factor>),String>{
 allowed(v,&["scenario_id","decision_cutoff","options","factors"],"candidate_view")?;
 let scenario=string(v,"scenario_id","candidate_view")?;
 let cutoff=string(v,"decision_cutoff","candidate_view")?;
 let cutoff_time=time(&cutoff,"candidate_view.decision_cutoff")?;
 let option_rows=get(v,"options","candidate_view")?.as_array().ok_or("candidate options must be an array")?;
 if option_rows.is_empty()||option_rows.len()>MAX_OPTIONS{return Err(format!("candidate must register 1..={MAX_OPTIONS} options"));}
 let mut options=BTreeMap::new();
 for (i,row) in option_rows.iter().enumerate(){
  let label=format!("candidate_view.options[{i}]");allowed(row,&["option_id","factor_id"],&label)?;
  let option=string(row,"option_id",&label)?;let factor=string(row,"factor_id",&label)?;
  if options.insert(option.clone(),factor).is_some(){return Err(format!("duplicate option ID {option:?}"));}
 }
 let factor_rows=get(v,"factors","candidate_view")?.as_array().ok_or("candidate factors must be an array")?;
 if factor_rows.is_empty()||factor_rows.len()>MAX_OPTIONS{return Err(format!("candidate must register 1..={MAX_OPTIONS} factors"));}
 let mut factors=BTreeMap::new();
 for (i,row) in factor_rows.iter().enumerate(){
  let label=format!("candidate_view.factors[{i}]");
  allowed(row,&["factor_id","dependence_group","source_ids","calibration_ref","likelihoods"],&label)?;
  let factor_id=string(row,"factor_id",&label)?;
  let likelihood_rows=get(row,"likelihoods",&label)?.as_array().ok_or("factor likelihoods must be an array")?;
  if likelihood_rows.is_empty()||likelihood_rows.len()>MAX_HYPOTHESES{return Err("factor likelihood rows must contain 1..=64 hypotheses".to_owned());}
  let mut likelihoods=BTreeMap::new();let mut outcomes:BTreeSet<String>=BTreeSet::new();
  for (j,hypothesis) in likelihood_rows.iter().enumerate(){
   let hl=format!("{label}.likelihoods[{j}]");
   allowed(hypothesis,&["hypothesis_id","outcomes"],&hl)?;
   let hypothesis_id=string(hypothesis,"hypothesis_id",&hl)?;
   let outcome_rows=obj(get(hypothesis,"outcomes",&hl)?,&format!("{hl}.outcomes"))?;
   if outcome_rows.len()<2||outcome_rows.len()>MAX_OUTCOMES_PER_OPTION{
    return Err(format!("{hl}.outcomes must have 2..={MAX_OUTCOMES_PER_OPTION} registered categories"));
   }
   let mut parsed=BTreeMap::new();
   for (outcome,value) in outcome_rows{
    let probability=parse_likelihood(value,&format!("{hl}.outcomes.{outcome}"))?;
    parsed.insert(outcome.clone(),probability);outcomes.insert(outcome.clone());
   }
   if likelihoods.insert(hypothesis_id.clone(),parsed).is_some(){return Err(format!("duplicate hypothesis {hypothesis_id:?} in factor {factor_id:?}"));}
  }
  if factors.insert(factor_id.clone(),Factor{outcomes,likelihoods}).is_some(){return Err(format!("duplicate factor ID {factor_id:?}"));}
 }
 for (option,factor) in &options{if !factors.contains_key(factor){return Err(format!("option {option:?} references missing factor {factor:?}"));}}
 let referenced:BTreeSet<String>=options.values().cloned().collect();
 if referenced!=factors.keys().cloned().collect(){return Err("candidate factors must be referenced by the exact option factor set".to_owned());}
 for factor in factors.values(){
  let hypotheses:Vec<String>=factor.likelihoods.keys().cloned().collect();
  for hypothesis in &hypotheses{
   let row=&factor.likelihoods[hypothesis];
   if row.values().map(|x|x.lower).sum::<u64>()>SCALE
    ||row.values().map(|x|x.upper).sum::<u64>()<SCALE
    ||row.values().map(|x|x.nominal).sum::<u64>()!=SCALE
   {return Err(format!("factor likelihood row for {hypothesis:?} is not a feasible outcome partition"));}
  }
 }
 let _=cutoff_time;
 Ok((scenario,cutoff,options,factors))
}
fn parse_oracle(v:&Value,cutoff:&str,options:&BTreeMap<String,String>,factors:&BTreeMap<String,Factor>)->Result<BTreeMap<String,Kernel>,String>{
 allowed(v,&["world_kernels"],"oracle")?;
 let kernels=get(v,"world_kernels","oracle")?.as_array().ok_or("oracle.world_kernels must be an array")?;
 if kernels.is_empty()||kernels.len()>MAX_HYPOTHESES{return Err("oracle.world_kernels must contain 1..=64 kernels".to_owned());}
 let cutoff_time=time(cutoff,"decision cutoff")?;
 let expected_hypotheses:BTreeSet<String>=factors.values().next().ok_or("no factors")?.likelihoods.keys().cloned().collect();
 for factor in factors.values(){if factor.likelihoods.keys().cloned().collect::<BTreeSet<_>>()!=expected_hypotheses{return Err("all factors must cover the same exact hypothesis set".to_owned());}}
 let mut out=BTreeMap::new();
 for (i,value) in kernels.iter().enumerate(){
  let label=format!("oracle.world_kernels[{i}]");
  allowed(value,&["hypothesis_id","kernel_id","authority","valid_from","valid_until","revoked_at","joint_outcomes"],&label)?;
  let hypothesis=string(value,"hypothesis_id",&label)?;
  if !expected_hypotheses.contains(&hypothesis){return Err(format!("kernel references unknown hypothesis {hypothesis:?}"));}
  let kernel_id=string(value,"kernel_id",&label)?;let authority=string(value,"authority",&label)?;
  let valid_from=string(value,"valid_from",&label)?;let valid_until=string(value,"valid_until",&label)?;
  let from=time(&valid_from,&label)?;let until=time(&valid_until,&label)?;
  if from>cutoff_time||until<cutoff_time||from>until{return Err(format!("kernel {kernel_id:?} is stale or invalid at decision cutoff"));}
  let revoked_at=match value.get("revoked_at"){
   None|Some(Value::Null)=>None,
   Some(Value::String(raw))=>Some(raw.clone()),
   _=>return Err(format!("{label}.revoked_at must be RFC3339 string or null"))
  };
  if let Some(revoked)=&revoked_at{if time(revoked,&label)?<=cutoff_time{return Err(format!("kernel {kernel_id:?} is revoked at decision cutoff"));}}
  let row_values=get(value,"joint_outcomes",&label)?.as_array().ok_or("joint_outcomes must be an array")?;
  if row_values.is_empty()||row_values.len()>MAX_JOINT_ROWS{return Err(format!("kernel rows must contain 1..={MAX_JOINT_ROWS} entries"));}
  let mut rows=Vec::new();let mut signatures=BTreeSet::new();
  for (j,row) in row_values.iter().enumerate(){
   let rl=format!("{label}.joint_outcomes[{j}]");
   allowed(row,&["outcomes","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&rl)?;
   let outcomes_value=get(row,"outcomes",&rl)?;
   let outcome_object=obj(outcomes_value,&format!("{rl}.outcomes"))?;
   let actual_options: BTreeSet<String>=outcome_object.keys().cloned().collect();
   if actual_options!=options.keys().cloned().collect(){return Err(format!("{rl}.outcomes must cover every registered option exactly"));}
   let mut outcomes=BTreeMap::new();
   for (option,outcome_value) in outcome_object{
    let outcome=outcome_value.as_str().filter(|x|!x.trim().is_empty()).ok_or("joint outcome category must be a non-empty string")?.to_owned();
    let factor=&options[option];if !factors[factor].outcomes.contains(&outcome){return Err(format!("{rl} uses unregistered outcome {outcome:?} for option {option:?}"));}
    outcomes.insert(option.clone(),outcome);
   }
   let signature=serde_json::to_string(&outcomes).map_err(|e|format!("cannot canonicalize joint outcome: {e}"))?;
   if !signatures.insert(signature){return Err(format!("{label} contains duplicate joint outcome vectors"));}
   let probability=parse_probability(row,&rl)?;
   rows.push(JointRow{outcomes,probability});
  }
  validate_partition(&rows,&label)?;
  if out.insert(hypothesis.clone(),Kernel{hypothesis_id:hypothesis.clone(),kernel_id,authority,valid_from,valid_until,revoked_at,rows}).is_some(){
   return Err(format!("duplicate joint kernel for hypothesis {hypothesis:?}"));
  }
 }
 if out.keys().cloned().collect::<BTreeSet<_>>()!=expected_hypotheses{return Err("joint kernels must cover the exact candidate hypothesis set".to_owned());}
 Ok(out)
}
fn check_marginals(model:&KernelModel)->Result<(Vec<Value>,Vec<Value>),String>{
 let mut marginals=Vec::new();let mut dependence=Vec::new();
 for (hypothesis,kernel) in &model.kernels{
  for (option,factor_id) in &model.options{
   let factor=&model.factors[factor_id];
   let registered=factor.likelihoods.get(hypothesis).ok_or_else(||format!("factor {factor_id:?} missing hypothesis {hypothesis:?}"))?;
   for outcome in &factor.outcomes{
    let derived=event_bounds(&kernel.rows,|row|row.outcomes.get(option)==Some(outcome))?;
    let channel=registered.get(outcome).ok_or_else(||format!("factor {factor_id:?} missing outcome {outcome:?}"))?;
    if channel.lower>derived.lower||channel.upper<derived.upper||channel.nominal!=derived.nominal{
     return Err(format!("factor {factor_id:?} likelihood envelope for {hypothesis:?}/{outcome:?} does not conservatively contain joint-kernel marginal [{},{}] nominal {}; registered [{},{}/{}]",
      derived.lower,derived.upper,derived.nominal,channel.lower,channel.upper,channel.nominal));
    }
    marginals.push(json!({
     "hypothesis_id":hypothesis,"option_id":option,"factor_id":factor_id,"outcome":outcome,
     "joint_kernel_marginal_lower_bps":derived.lower,"joint_kernel_marginal_upper_bps":derived.upper,
     "nominal_probability_bps":derived.nominal,
     "registered_channel_lower_bps":channel.lower,"registered_channel_upper_bps":channel.upper,
     "registered_channel_nominal_bps":channel.nominal
    }));
   }
  }
  let option_ids:Vec<String>=model.options.keys().cloned().collect();
  for i in 0..option_ids.len(){
   for j in (i+1)..option_ids.len(){
    let left=&option_ids[i];let right=&option_ids[j];
    let left_factor=&model.factors[&model.options[left]];
    let right_factor=&model.factors[&model.options[right]];
    for left_outcome in &left_factor.outcomes{
     let left_marginal=event_bounds(&kernel.rows,|row|row.outcomes.get(left)==Some(left_outcome))?;
     for right_outcome in &right_factor.outcomes{
      let right_marginal=event_bounds(&kernel.rows,|row|row.outcomes.get(right)==Some(right_outcome))?;
      let joint=event_bounds(&kernel.rows,|row|row.outcomes.get(left)==Some(left_outcome)&&row.outcomes.get(right)==Some(right_outcome))?;
      let frechet_lower=left_marginal.lower.saturating_add(right_marginal.lower).saturating_sub(SCALE);
      let frechet_upper=left_marginal.upper.min(right_marginal.upper);
      if joint.lower<frechet_lower||joint.upper>frechet_upper{return Err("joint outcome interval violates its marginal Fréchet envelope".to_owned());}
      dependence.push(json!({
       "hypothesis_id":hypothesis,"left_option_id":left,"left_outcome":left_outcome,
       "right_option_id":right,"right_outcome":right_outcome,
       "joint_lower_bps":joint.lower,"joint_upper_bps":joint.upper,"joint_nominal_bps":joint.nominal,
       "frechet_lower_bps":frechet_lower,"frechet_upper_bps":frechet_upper,
       "frechet_width_bps":frechet_upper-frechet_lower,
       "joint_width_bps":joint.upper-joint.lower
      }));
     }
    }
   }
  }
 }
 Ok((marginals,dependence))
}
fn evaluate(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=get(&bundle,"candidate_view","bundle")?;let oracle=get(&bundle,"oracle","bundle")?;
 let (scenario_id,decision_cutoff,options,factors)=parse_candidate(candidate)?;
 let kernels=parse_oracle(oracle,&decision_cutoff,&options,&factors)?;
 let model=KernelModel{scenario_id,decision_cutoff,options,factors,kernels};
 let (marginals,dependence)=check_marginals(&model)?;
 let row_count:usize=model.kernels.values().map(|x|x.rows.len()).sum();
 Ok(json!({
  "scenario_id":model.scenario_id,
  "adjudication_status":"synthetic_known_answer_stochastic_joint_kernel_validation",
  "decision_cutoff":model.decision_cutoff,
  "hypothesis_count":model.kernels.len(),
  "option_count":model.options.len(),
  "factor_count":model.factors.len(),
  "joint_row_count":row_count,
  "joint_kernel_count":model.kernels.len(),
  "marginal_check_count":marginals.len(),
  "pairwise_dependence_check_count":dependence.len(),
  "marginal_checks":marginals,
  "pairwise_dependence":dependence,
  "claim_ceiling":"Validates evaluator-owned bounded-simplex distributions over complete joint outcome vectors and verifies conservative marginal consistency with registered D6.17 channel likelihood intervals. It does not establish calibration of those kernels against real observations, prove model completeness, estimate transition drift, solve an adaptive policy, authorize actions, or make a production reliability claim."
 }))
}
pub fn validate_stochastic_joint_kernel_json(raw:&str)->Result<Value,String>{evaluate(raw)}
fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{
  eprintln!("Usage: stochastic-kernel-validator <candidate-and-oracle-bundle.json>");
  process::exit(if args.len()==1{0}else{2});
 }
 let raw=match fs::read_to_string(&args[0]){
  Ok(x)=>x,
  Err(e)=>{eprintln!("ERROR D6.17.2.2-IO-001 {}: {e}",args[0]);process::exit(2)}
 };
 match evaluate(&raw){
  Ok(report)=>match serde_json::to_string_pretty(&report){
   Ok(text)=>println!("{text}"),
   Err(e)=>{eprintln!("ERROR D6.17.2.2-JSON-001: {e}");process::exit(2)}
  },
  Err(e)=>{eprintln!("ERROR D6.17.2.2-EVALUATION-001: {e}");process::exit(1)}
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 const FIXTURE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-joint-kernel-v0.1.json");
 const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-joint-kernel-v0.1.expected.json");
 fn report()->Value{evaluate(FIXTURE).expect("stochastic joint kernel fixture validates")}
 #[test]fn report_matches_known_answer(){let expected:Value=serde_json::from_str(EXPECTED).unwrap();assert_eq!(report(),expected);}
 #[test]fn joint_kernel_preserves_common_cause_dependence(){
  let r=report();let pair=r["pairwise_dependence"].as_array().unwrap().iter()
   .find(|x|x["left_option_id"]=="gate-a"&&x["left_outcome"]=="fail"&&x["right_option_id"]=="gate-b"&&x["right_outcome"]=="fail").unwrap();
  assert_eq!(pair["joint_lower_bps"],4000);assert_eq!(pair["joint_upper_bps"],5000);
  assert_eq!(pair["frechet_lower_bps"],0);assert_eq!(pair["frechet_upper_bps"],6000);
 }
 #[test]fn nominal_marginals_match_registered_channel(){let r=report();assert_eq!(r["marginal_check_count"],4);}
 #[test]fn candidate_cannot_inject_oracle_fields(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["world_kernels"]=json!([]);assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));}
 #[test]fn duplicate_joint_outcome_vector_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();let row=b["oracle"]["world_kernels"][0]["joint_outcomes"][0].clone();b["oracle"]["world_kernels"][0]["joint_outcomes"].as_array_mut().unwrap().push(row);assert!(evaluate(&b.to_string()).unwrap_err().contains("duplicate joint outcome"));}
 #[test]fn missing_option_dimension_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["oracle"]["world_kernels"][0]["joint_outcomes"][0]["outcomes"].as_object_mut().unwrap().remove("gate-b");assert!(evaluate(&b.to_string()).unwrap_err().contains("cover every registered option"));}
 #[test]fn unregistered_outcome_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["oracle"]["world_kernels"][0]["joint_outcomes"][0]["outcomes"]["gate-a"]=json!("unknown");assert!(evaluate(&b.to_string()).unwrap_err().contains("unregistered outcome"));}
 #[test]fn kernel_must_be_valid_at_decision_cutoff(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["oracle"]["world_kernels"][0]["valid_until"]=json!("2026-11-16T00:00:00Z");assert!(evaluate(&b.to_string()).unwrap_err().contains("stale or invalid"));}
 #[test]fn revoked_kernel_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["oracle"]["world_kernels"][0]["revoked_at"]=json!("2026-11-17T10:00:00Z");assert!(evaluate(&b.to_string()).unwrap_err().contains("revoked"));}
 #[test]fn nominal_mass_must_sum_to_one(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["oracle"]["world_kernels"][0]["joint_outcomes"][0]["nominal_probability_bps"]=json!(4000);assert!(evaluate(&b.to_string()).unwrap_err().contains("not a non-empty bounded simplex"));}
 #[test]fn underclaimed_channel_marginal_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["factors"][0]["likelihoods"][0]["outcomes"]["fail"]["upper_likelihood_bps"]=json!(4900);assert!(evaluate(&b.to_string()).unwrap_err().contains("does not conservatively contain"));}
 #[test]fn mismatched_nominal_channel_marginal_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["factors"][1]["likelihoods"][0]["outcomes"]["fail"]["nominal_likelihood_bps"]=json!(4901);assert!(evaluate(&b.to_string()).unwrap_err().contains("does not conservatively contain"));}
}
