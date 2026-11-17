//! D6.17.2.2.3-alpha shared-latent joint-kernel validator.
//!
//! One evaluator-owned latent-weight credal simplex is shared by every
//! hypothesis. Each hypothesis/latent-state pair has a conditional joint
//! distribution over complete option-outcome vectors. This preserves
//! cross-hypothesis coupling rather than replacing it with independent
//! per-hypothesis intervals.

use chrono::{DateTime,FixedOffset};
use serde_json::{json,Value};
use std::collections::{BTreeMap,BTreeSet};
use std::{env,fs,process};

const SCALE:u64=10_000;
const MAX_HYPOTHESES:usize=32;
const MAX_OPTIONS:usize=6;
const MAX_LATENT_STATES:usize=8;
const MAX_ROWS:usize=4096;
const CLAIM_CEILING:&str="Validates an evaluator-owned shared latent-weight credal simplex and exact conditional joint outcome distributions for every hypothesis/latent-state pair. It derives cross-hypothesis coupled marginal and pairwise event intervals and checks candidate channel consistency. It does not estimate latent weights from real measurements, prove latent-state completeness, optimize adaptive policies, validate drift, authenticate the evaluator's oracle, or authorize actions.";

#[derive(Clone,Debug)]
struct Bounds{lower:u64,upper:u64,nominal:u64}
#[derive(Clone,Debug)]
struct OptionDef{factor_id:String}
#[derive(Clone,Debug)]
struct Factor{outcomes:BTreeSet<String>,likelihoods:BTreeMap<String,BTreeMap<String,Bounds>>}
#[derive(Clone,Debug)]
struct LatentState{id:String,probability:Bounds}
#[derive(Clone,Debug)]
struct JointRow{outcomes:BTreeMap<String,String>,probability:u64}
#[derive(Clone,Debug)]
struct ConditionalKernel{
 hypothesis_id:String,latent_state_id:String,kernel_id:String,authority:String,
 rows:Vec<JointRow>
}

fn object<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
 v.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn allowed(v:&Value,keys:&[&str],label:&str)->Result<(),String>{
 for key in object(v,label)?.keys(){if !keys.contains(&key.as_str()){return Err(format!("{label} contains unknown field {key:?}"));}}
 Ok(())
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
fn time(raw:&str,label:&str)->Result<DateTime<FixedOffset>,String>{
 DateTime::parse_from_rfc3339(raw).map_err(|e|format!("{label} must be RFC 3339: {e}"))
}
fn interval(v:&Value,label:&str)->Result<Bounds,String>{
 allowed(v,&["lower_likelihood_bps","upper_likelihood_bps","nominal_likelihood_bps"],label)?;
 let b=Bounds{
  lower:uint(v,"lower_likelihood_bps",label)?,
  upper:uint(v,"upper_likelihood_bps",label)?,
  nominal:uint(v,"nominal_likelihood_bps",label)?
 };
 if b.lower>b.nominal||b.nominal>b.upper||b.upper>SCALE{
  return Err(format!("{label} must satisfy 0 <= lower <= nominal <= upper <= 10000"));
 }
 Ok(b)
}
fn bounded_simplex(values:&[Bounds],label:&str)->Result<(),String>{
 if values.is_empty(){return Err(format!("{label} cannot be empty"));}
 let lower:u64=values.iter().map(|x|x.lower).sum();
 let upper:u64=values.iter().map(|x|x.upper).sum();
 let nominal:u64=values.iter().map(|x|x.nominal).sum();
 if lower>SCALE||upper<SCALE||nominal!=SCALE{
  return Err(format!("{label} is infeasible: lower={lower}, upper={upper}, nominal={nominal}"));
 }
 Ok(())
}
fn weighted_extreme(weights:&[Bounds],coefficients:&[u64],maximize:bool)->Result<u64,String>{
 if weights.len()!=coefficients.len()||weights.is_empty(){return Err("shared latent weight/coefficient dimensions differ".to_owned());}
 let lower_sum:u64=weights.iter().map(|x|x.lower).sum();
 if lower_sum>SCALE{return Err("shared latent lower weights exceed 10000".to_owned());}
 let mut numerator=weights.iter().zip(coefficients).try_fold(0u128,|sum,(w,c)|{
  sum.checked_add(w.lower as u128*(*c as u128)).ok_or_else(||"latent event expectation overflow".to_owned())
 })?;
 let mut remaining=SCALE-lower_sum;
 let mut order:Vec<usize>=(0..weights.len()).collect();
 if maximize{
  order.sort_by(|a,b|coefficients[*b].cmp(&coefficients[*a]).then_with(||a.cmp(b)));
 }else{
  order.sort_by(|a,b|coefficients[*a].cmp(&coefficients[*b]).then_with(||a.cmp(b)));
 }
 for i in order{
  let take=remaining.min(weights[i].upper-weights[i].lower);
  numerator=numerator.checked_add(take as u128*coefficients[i] as u128)
   .ok_or("latent event expectation overflow")?;
  remaining-=take;if remaining==0{break;}
 }
 if remaining>0{return Err("shared latent upper weights cannot allocate unit mass".to_owned());}
 if maximize{
  Ok(((numerator+SCALE as u128-1)/SCALE as u128).min(SCALE as u128) as u64)
 }else{
  Ok((numerator/SCALE as u128) as u64)
 }
}
fn weighted_nominal(weights:&[Bounds],coefficients:&[u64])->Result<u64,String>{
 if weights.len()!=coefficients.len()||weights.is_empty(){return Err("shared latent weight/coefficient dimensions differ".to_owned());}
 let numerator=weights.iter().zip(coefficients).try_fold(0u128,|sum,(w,c)|{
  sum.checked_add(w.nominal as u128*(*c as u128)).ok_or_else(||"nominal latent expectation overflow".to_owned())
 })?;
 if numerator%(SCALE as u128)!=0{
  return Err("nominal shared-latent event is not exactly representable in basis points".to_owned());
 }
 Ok((numerator/SCALE as u128) as u64)
}
fn parse_candidate(candidate:&Value)->Result<(String,String,BTreeMap<String,OptionDef>,BTreeMap<String,Factor>,BTreeSet<String>),String>{
 allowed(candidate,&["scenario_id","decision_cutoff","options","factors"],"candidate_view")?;
 let scenario=string(candidate,"scenario_id","candidate_view")?;
 let cutoff=string(candidate,"decision_cutoff","candidate_view")?;
 let _=time(&cutoff,"candidate_view.decision_cutoff")?;
 let option_rows=get(candidate,"options","candidate_view")?.as_array().ok_or("candidate options must be array")?;
 if option_rows.is_empty()||option_rows.len()>MAX_OPTIONS{return Err(format!("candidate options must contain 1..={MAX_OPTIONS} entries"));}
 let mut options=BTreeMap::new();
 for (i,row) in option_rows.iter().enumerate(){
  let label=format!("candidate_view.options[{i}]");
  allowed(row,&["option_id","factor_id"],&label)?;
  let id=string(row,"option_id",&label)?;let factor=string(row,"factor_id",&label)?;
  if options.insert(id.clone(),OptionDef{factor_id:factor}).is_some(){return Err(format!("duplicate option ID {id:?}"));}
 }
 let factor_rows=get(candidate,"factors","candidate_view")?.as_array().ok_or("candidate factors must be array")?;
 if factor_rows.is_empty()||factor_rows.len()>MAX_OPTIONS{return Err(format!("candidate factors must contain 1..={MAX_OPTIONS} entries"));}
 let mut factors=BTreeMap::new();let mut hypothesis_ids:Option<BTreeSet<String>>=None;
 for (i,row) in factor_rows.iter().enumerate(){
  let label=format!("candidate_view.factors[{i}]");
  allowed(row,&["factor_id","likelihoods"],&label)?;
  let factor_id=string(row,"factor_id",&label)?;
  let likelihood_rows=get(row,"likelihoods",&label)?.as_array().ok_or("factor likelihoods must be array")?;
  if likelihood_rows.is_empty()||likelihood_rows.len()>MAX_HYPOTHESES{return Err("factor likelihoods must contain 1..=32 hypotheses".to_owned());}
  let mut likelihoods=BTreeMap::new();let mut outcomes:BTreeSet<String>=BTreeSet::new();
  let mut local_hypotheses=BTreeSet::new();
  for (j,hypothesis) in likelihood_rows.iter().enumerate(){
   let hl=format!("{label}.likelihoods[{j}]");
   allowed(hypothesis,&["hypothesis_id","outcomes"],&hl)?;
   let hid=string(hypothesis,"hypothesis_id",&hl)?;
   if !local_hypotheses.insert(hid.clone()){return Err(format!("duplicate hypothesis {hid:?} in factor {factor_id:?}"));}
   let raw=obj(get(hypothesis,"outcomes",&hl)?,&format!("{hl}.outcomes"))?;
   if raw.len()<2||raw.len()>4{return Err(format!("{hl}.outcomes must contain 2..=4 outcome categories"));}
   let names=raw.keys().cloned().collect::<BTreeSet<_>>();
   if !outcomes.is_empty()&&outcomes!=names{return Err(format!("{label} outcome categories differ across hypotheses"));}
   outcomes=names;
   let mut parsed=BTreeMap::new();
   for (outcome,raw_interval) in raw{
    parsed.insert(outcome.clone(),interval(raw_interval,&format!("{hl}.outcomes.{outcome}"))?);
   }
   bounded_simplex(&parsed.values().cloned().collect::<Vec<_>>(),&format!("{hl}.outcomes"))?;
   if likelihoods.insert(hid.clone(),parsed).is_some(){return Err(format!("duplicate hypothesis {hid:?}"));}
  }
  if let Some(expected)=&hypothesis_ids{
   if expected!=&local_hypotheses{return Err(format!("{label} must cover the exact candidate hypothesis set"));}
  }else{hypothesis_ids=Some(local_hypotheses);}
  if factors.insert(factor_id.clone(),Factor{outcomes,likelihoods}).is_some(){return Err(format!("duplicate factor ID {factor_id:?}"));}
 }
 let hypotheses=hypothesis_ids.ok_or("candidate has no hypotheses")?;
 if hypotheses.is_empty()||hypotheses.len()>MAX_HYPOTHESES{return Err("candidate hypothesis count is outside 1..=32".to_owned());}
 let referenced: BTreeSet<String>=options.values().map(|o|o.factor_id.clone()).collect();
 if referenced.len()!=options.len()||referenced!=factors.keys().cloned().collect(){
  return Err("each option must map to one unique factor, and factor set must match options exactly".to_owned());
 }
 Ok((scenario,cutoff,options,factors,hypotheses))
}
fn parse_latents(oracle:&Value,cutoff:&str)->Result<Vec<LatentState>,String>{
 let rows=get(oracle,"latent_states","oracle")?.as_array().ok_or("oracle.latent_states must be array")?;
 if rows.is_empty()||rows.len()>MAX_LATENT_STATES{return Err(format!("latent_states must contain 1..={MAX_LATENT_STATES} states"));}
 let cutoff_time=time(cutoff,"decision cutoff")?;
 let mut out=Vec::new();let mut ids=BTreeSet::new();
 for (i,row) in rows.iter().enumerate(){
  let label=format!("oracle.latent_states[{i}]");
  allowed(row,&["latent_state_id","lower_probability_bps","upper_probability_bps","nominal_probability_bps","authority","valid_from","valid_until","revoked_at"],&label)?;
  let id=string(row,"latent_state_id",&label)?;
  if !ids.insert(id.clone()){return Err(format!("duplicate latent state {id:?}"));}
  let authority=string(row,"authority",&label)?;
  let from=time(&string(row,"valid_from",&label)?,&label)?;
  let until=time(&string(row,"valid_until",&label)?,&label)?;
  let revoked=match get(row,"revoked_at",&label)?{
   Value::Null=>None,Value::String(x)=>Some(time(x,&label)?),_=>return Err(format!("{label}.revoked_at must be string or null"))
  };
  if authority.trim().is_empty()||from>cutoff_time||until<cutoff_time||from>until||revoked.as_ref().is_some_and(|x|x<=&cutoff_time){
   return Err(format!("latent state {id:?} is not evaluator-valid at decision cutoff"));
  }
  let probability=Bounds{
   lower:uint(row,"lower_probability_bps",&label)?,
   upper:uint(row,"upper_probability_bps",&label)?,
   nominal:uint(row,"nominal_probability_bps",&label)?
  };
  if probability.lower>probability.nominal||probability.nominal>probability.upper||probability.upper>SCALE{
   return Err(format!("{label} violates probability interval ordering"));
  }
  out.push(LatentState{id,probability});
 }
 bounded_simplex(&out.iter().map(|x|x.probability.clone()).collect::<Vec<_>>(),"oracle.latent_states")?;
 Ok(out)
}
fn parse_kernels(
 oracle:&Value,cutoff:&str,options:&BTreeMap<String,OptionDef>,factors:&BTreeMap<String,Factor>,
 hypotheses:&BTreeSet<String>,latents:&[LatentState]
)->Result<BTreeMap<(String,String),ConditionalKernel>,String>{
 let rows=get(oracle,"conditional_kernels","oracle")?.as_array().ok_or("oracle.conditional_kernels must be array")?;
 let expected_count=hypotheses.len().checked_mul(latents.len()).ok_or("conditional kernel count overflow")?;
 if rows.len()!=expected_count{return Err(format!("conditional kernels must cover exact hypothesis x latent-state product: expected {expected_count}, got {}",rows.len()));}
 let cutoff_time=time(cutoff,"decision cutoff")?;
 let latent_ids: BTreeSet<String>=latents.iter().map(|x|x.id.clone()).collect();
 let mut out=BTreeMap::new();
 for (i,kernel) in rows.iter().enumerate(){
  let label=format!("oracle.conditional_kernels[{i}]");
  allowed(kernel,&["hypothesis_id","latent_state_id","kernel_id","authority","valid_from","valid_until","revoked_at","joint_outcomes"],&label)?;
  let hypothesis=string(kernel,"hypothesis_id",&label)?;
  let latent=string(kernel,"latent_state_id",&label)?;
  if !hypotheses.contains(&hypothesis)||!latent_ids.contains(&latent){return Err(format!("{label} references unknown hypothesis or latent state"));}
  let kernel_id=string(kernel,"kernel_id",&label)?;let authority=string(kernel,"authority",&label)?;
  let from=time(&string(kernel,"valid_from",&label)?,&label)?;
  let until=time(&string(kernel,"valid_until",&label)?,&label)?;
  let revoked=match get(kernel,"revoked_at",&label)?{
   Value::Null=>None,Value::String(x)=>Some(time(x,&label)?),_=>return Err(format!("{label}.revoked_at must be string or null"))
  };
  if authority.trim().is_empty()||from>cutoff_time||until<cutoff_time||from>until||revoked.as_ref().is_some_and(|x|x<=&cutoff_time){
   return Err(format!("conditional kernel {kernel_id:?} is stale or revoked at decision cutoff"));
  }
  let joint=get(kernel,"joint_outcomes",&label)?.as_array().ok_or("joint_outcomes must be array")?;
  if joint.is_empty()||joint.len()>MAX_ROWS{return Err(format!("{label}.joint_outcomes must contain 1..={MAX_ROWS} rows"));}
  let expected_options: BTreeSet<String>=options.keys().cloned().collect();
  let mut signatures=BTreeSet::new();let mut parsed=Vec::new();let mut total=0u64;
  for (j,row) in joint.iter().enumerate(){
   let rl=format!("{label}.joint_outcomes[{j}]");
   allowed(row,&["outcomes","probability_bps"],&rl)?;
   let raw=obj(get(row,"outcomes",&rl)?,&format!("{rl}.outcomes"))?;
   if raw.keys().cloned().collect::<BTreeSet<_>>()!=expected_options{return Err(format!("{rl}.outcomes must cover all registered options exactly"));}
   let mut outcomes=BTreeMap::new();
   for (option,raw_outcome) in raw{
    let outcome=raw_outcome.as_str().filter(|x|!x.trim().is_empty()).ok_or("joint outcome must be non-empty string")?.to_owned();
    let factor=&options[option].factor_id;
    if !factors[factor].outcomes.contains(&outcome){return Err(format!("{rl} contains unregistered outcome {outcome:?} for {option:?}"));}
    outcomes.insert(option.clone(),outcome);
   }
   let signature=serde_json::to_string(&outcomes).map_err(|e|format!("joint vector serialization failed: {e}"))?;
   if !signatures.insert(signature){return Err(format!("{label} contains duplicate joint outcome vector"));}
   let probability=uint(row,"probability_bps",&rl)?;
   if probability>SCALE{return Err(format!("{rl}.probability_bps must be <= 10000"));}
   total=total.checked_add(probability).ok_or("conditional kernel mass overflow")?;
   parsed.push(JointRow{outcomes,probability});
  }
  if total!=SCALE{return Err(format!("{label} conditional joint outcomes must sum to 10000, got {total}"));}
  let key=(hypothesis.clone(),latent.clone());
  if out.insert(key,ConditionalKernel{hypothesis_id:hypothesis,latent_state_id:latent,kernel_id,authority,rows:parsed}).is_some(){
   return Err(format!("{label} duplicates a hypothesis/latent kernel pair"));
  }
 }
 for hypothesis in hypotheses{
  for latent in &latent_ids{
   if !out.contains_key(&(hypothesis.clone(),latent.clone())){
    return Err(format!("missing conditional kernel for ({hypothesis:?}, {latent:?})"));
   }
  }
 }
 Ok(out)
}
fn marginal_coefficients(kernel:&ConditionalKernel,option:&str,outcome:&str)->u64{
 kernel.rows.iter().filter(|row|row.outcomes.get(option).is_some_and(|x|x==outcome))
  .map(|row|row.probability).sum()
}
fn mixture_interval(weights:&[LatentState],coefficients:&[u64])->Result<Bounds,String>{
 let bounds:Vec<Bounds>=weights.iter().map(|x|x.probability.clone()).collect();
 let lower=weighted_extreme(&bounds,coefficients,false)?;
 let upper=weighted_extreme(&bounds,coefficients,true)?;
 let nominal=weighted_nominal(&bounds,coefficients)?;
 if lower>nominal||nominal>upper{return Err("shared-latent mixture interval does not contain nominal point".to_owned());}
 Ok(Bounds{lower,upper,nominal})
}
fn evaluate(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=get(&bundle,"candidate_view","bundle")?;
 let oracle=get(&bundle,"oracle","bundle")?;
 allowed(oracle,&["decision_cutoff","latent_states","conditional_kernels"],"oracle")?;
 let (scenario,cutoff,options,factors,hypotheses)=parse_candidate(candidate)?;
 if string(oracle,"decision_cutoff","oracle")?!=cutoff{return Err("candidate and evaluator decision cutoffs differ".to_owned());}
 let latents=parse_latents(oracle,&cutoff)?;
 let kernels=parse_kernels(oracle,&cutoff,&options,&factors,&hypotheses,&latents)?;
 let mut marginal_checks=Vec::new();
 for (factor_id,factor) in &factors{
  let option=options.iter().find(|(_,x)|&x.factor_id==factor_id).map(|(id,_)|id.clone())
   .ok_or_else(||format!("factor {factor_id:?} has no option"))?;
  for hypothesis in &hypotheses{
   for outcome in &factor.outcomes{
    let coefficients:Vec<u64>=latents.iter().map(|latent|{
     marginal_coefficients(&kernels[&(hypothesis.clone(),latent.id.clone())],&option,outcome)
    }).collect();
    let mixture=mixture_interval(&latents,&coefficients)?;
    let registered=&factor.likelihoods[hypothesis][outcome];
    if registered.lower>mixture.lower||registered.upper<mixture.upper||registered.nominal!=mixture.nominal{
     return Err(format!("candidate channel marginal underclaims shared-latent mixture for {factor_id:?}/{hypothesis:?}/{outcome:?}: mixture [{},{}] nominal {}, candidate [{},{}] nominal {}",
      mixture.lower,mixture.upper,mixture.nominal,registered.lower,registered.upper,registered.nominal));
    }
    marginal_checks.push(json!({
     "hypothesis_id":hypothesis,"option_id":option,"factor_id":factor_id,"outcome":outcome,
     "conditional_probabilities_by_latent_state":latents.iter().zip(&coefficients).map(|(z,p)|json!({"latent_state_id":z.id,"probability_bps":p})).collect::<Vec<_>>(),
     "shared_latent_marginal_lower_bps":mixture.lower,"shared_latent_marginal_nominal_bps":mixture.nominal,
     "shared_latent_marginal_upper_bps":mixture.upper,
     "registered_channel_lower_bps":registered.lower,"registered_channel_nominal_bps":registered.nominal,
     "registered_channel_upper_bps":registered.upper
    }));
   }
  }
 }
 let option_ids:Vec<String>=options.keys().cloned().collect();
 let mut pairwise_checks=Vec::new();
 for hypothesis in &hypotheses{
  for i in 0..option_ids.len(){
   for j in (i+1)..option_ids.len(){
    let left=&option_ids[i];let right=&option_ids[j];
    let left_factor=&factors[&options[left].factor_id];
    let right_factor=&factors[&options[right].factor_id];
    for left_outcome in &left_factor.outcomes{
     let left_coefficients:Vec<u64>=latents.iter().map(|z|{
      marginal_coefficients(&kernels[&(hypothesis.clone(),z.id.clone())],left,left_outcome)
     }).collect();
     let left_mix=mixture_interval(&latents,&left_coefficients)?;
     for right_outcome in &right_factor.outcomes{
      let right_coefficients:Vec<u64>=latents.iter().map(|z|{
       marginal_coefficients(&kernels[&(hypothesis.clone(),z.id.clone())],right,right_outcome)
      }).collect();
      let right_mix=mixture_interval(&latents,&right_coefficients)?;
      let joint_coefficients:Vec<u64>=latents.iter().map(|z|{
       kernels[&(hypothesis.clone(),z.id.clone())].rows.iter()
        .filter(|row|row.outcomes.get(left)==Some(left_outcome)&&row.outcomes.get(right)==Some(right_outcome))
        .map(|row|row.probability).sum()
      }).collect();
      let joint=mixture_interval(&latents,&joint_coefficients)?;
      let frechet_lower=left_mix.lower.saturating_add(right_mix.lower).saturating_sub(SCALE);
      let frechet_upper=left_mix.upper.min(right_mix.upper);
      if joint.lower<frechet_lower||joint.upper>frechet_upper{return Err("shared-latent joint event violates its marginal Frechet envelope".to_owned());}
      pairwise_checks.push(json!({
       "hypothesis_id":hypothesis,"left_option_id":left,"left_outcome":left_outcome,
       "right_option_id":right,"right_outcome":right_outcome,
       "joint_lower_bps":joint.lower,"joint_nominal_bps":joint.nominal,"joint_upper_bps":joint.upper,
       "frechet_lower_bps":frechet_lower,"frechet_upper_bps":frechet_upper
      }));
     }
    }
   }
  }
 }
 let provenance=kernels.values().map(|k|json!({
  "hypothesis_id":k.hypothesis_id,"latent_state_id":k.latent_state_id,
  "kernel_id":k.kernel_id,"authority":k.authority,"joint_row_count":k.rows.len()
 })).collect::<Vec<_>>();
 Ok(json!({
  "scenario_id":scenario,
  "adjudication_status":"shared_latent_joint_kernel_validation",
  "decision_cutoff":cutoff,
  "coupling_model":{
   "kind":"one_shared_latent_weight_credal_simplex",
   "scope":"The same evaluator-owned latent-state weights apply to every hypothesis and every registered option in this decision window.",
   "latent_weights":latents.iter().map(|z|json!({
    "latent_state_id":z.id,"lower_probability_bps":z.probability.lower,
    "nominal_probability_bps":z.probability.nominal,"upper_probability_bps":z.probability.upper
   })).collect::<Vec<_>>()
  },
  "hypothesis_count":hypotheses.len(),"latent_state_count":latents.len(),
  "option_count":options.len(),"conditional_kernel_count":kernels.len(),
  "conditional_joint_row_count":kernels.values().map(|k|k.rows.len()).sum::<usize>(),
  "marginal_check_count":marginal_checks.len(),"pairwise_joint_check_count":pairwise_checks.len(),
  "kernel_provenance":provenance,
  "marginal_checks":marginal_checks,
  "pairwise_joint_checks":pairwise_checks,
  "claim_ceiling":CLAIM_CEILING
 }))
}
pub fn validate_shared_latent_kernels_json(raw:&str)->Result<Value,String>{evaluate(raw)}
fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{
  eprintln!("Usage: shared-latent-kernel-validator <candidate-and-oracle-bundle.json>");
  process::exit(if args.len()==1{0}else{2});
 }
 let raw=match fs::read_to_string(&args[0]){Ok(x)=>x,Err(e)=>{eprintln!("ERROR D6.17.2.2.3-IO-001 {}: {e}",args[0]);process::exit(2)}};
 match evaluate(&raw){
  Ok(report)=>match serde_json::to_string_pretty(&report){Ok(output)=>println!("{output}"),Err(e)=>{eprintln!("ERROR D6.17.2.2.3-JSON-001: {e}");process::exit(2)}},
  Err(e)=>{eprintln!("ERROR D6.17.2.2.3-EVALUATION-001: {e}");process::exit(1)}
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 const FIXTURE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-shared-latent-kernel-v0.1.json");
 const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-shared-latent-kernel-v0.1.expected.json");
 fn report()->Value{evaluate(FIXTURE).expect("shared latent fixture validates")}
 #[test]fn report_matches_shared_latent_known_answer(){
  let expected:Value=serde_json::from_str(EXPECTED).unwrap();assert_eq!(report(),expected);
 }
 #[test]fn one_shared_weight_vector_is_used_for_all_hypotheses(){
  let r=report();
  assert_eq!(r["coupling_model"]["kind"],"one_shared_latent_weight_credal_simplex");
  assert_eq!(r["conditional_kernel_count"],4);
  assert_eq!(r["coupling_model"]["latent_weights"][0]["lower_probability_bps"],4000);
 }
 #[test]fn common_latent_mixture_marginals_match_candidate_channels(){
  let r=report();
  let safe_a=r["marginal_checks"].as_array().unwrap().iter()
   .find(|x|x["hypothesis_id"]=="safe-world"&&x["option_id"]=="gate-a"&&x["outcome"]=="pass").unwrap();
  assert_eq!(safe_a["shared_latent_marginal_lower_bps"],5000);
  assert_eq!(safe_a["shared_latent_marginal_nominal_bps"],5500);
  assert_eq!(safe_a["shared_latent_marginal_upper_bps"],6000);
 }
 #[test]fn common_cause_joint_pass_pass_is_not_reconstructed_from_marginals(){
  let r=report();
  let pair=r["pairwise_joint_checks"].as_array().unwrap().iter()
   .find(|x|x["hypothesis_id"]=="safe-world"&&x["left_option_id"]=="gate-a"&&x["left_outcome"]=="pass"&&x["right_option_id"]=="gate-b"&&x["right_outcome"]=="pass").unwrap();
  assert_eq!(pair["joint_lower_bps"],4400);
  assert_eq!(pair["joint_nominal_bps"],5000);
  assert_eq!(pair["joint_upper_bps"],5600);
 }
 #[test]fn candidate_cannot_inject_shared_latent_oracle(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["candidate_view"]["latent_states"]=json!([]);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));
 }
 #[test]fn latent_probability_simplex_must_be_feasible(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["latent_states"][0]["lower_probability_bps"]=json!(6000);
  b["oracle"]["latent_states"][1]["lower_probability_bps"]=json!(6000);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("infeasible"));
 }
 #[test]fn missing_hypothesis_latent_pair_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["conditional_kernels"].as_array_mut().unwrap().pop();
  assert!(evaluate(&b.to_string()).unwrap_err().contains("exact hypothesis x latent-state product"));
 }
 #[test]fn non_normalized_conditional_kernel_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["conditional_kernels"][0]["joint_outcomes"][0]["probability_bps"]=json!(1999);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("must sum to 10000"));
 }
 #[test]fn stale_latent_state_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["latent_states"][0]["valid_until"]=json!("2026-11-16T23:59:00Z");
  assert!(evaluate(&b.to_string()).unwrap_err().contains("not evaluator-valid"));
 }
 #[test]fn revoked_conditional_kernel_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["conditional_kernels"][0]["revoked_at"]=json!("2026-11-17T09:59:00Z");
  assert!(evaluate(&b.to_string()).unwrap_err().contains("stale or revoked"));
 }
 #[test]fn underclaimed_marginal_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["candidate_view"]["factors"][0]["likelihoods"][0]["outcomes"]["pass"]["upper_likelihood_bps"]=json!(5999);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("underclaims shared-latent mixture"));
 }
 #[test]fn duplicate_joint_vector_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  let row=b["oracle"]["conditional_kernels"][0]["joint_outcomes"][0].clone();
  b["oracle"]["conditional_kernels"][0]["joint_outcomes"].as_array_mut().unwrap().push(row);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("duplicate joint outcome vector"));
 }
}
