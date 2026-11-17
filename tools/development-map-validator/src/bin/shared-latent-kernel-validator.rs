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
 // The source probabilities are in basis points. Round half-up to the nearest
 // basis point for the human-readable marginal projection; risk probes retain
 // the full integer numerator and do not use this rounded value.
 Ok(((numerator+(SCALE as u128)/2)/SCALE as u128).min(SCALE as u128) as u64)
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
   let raw=object(get(hypothesis,"outcomes",&hl)?,&format!("{hl}.outcomes"))?;
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
   let raw=object(get(row,"outcomes",&rl)?,&format!("{rl}.outcomes"))?;
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
fn simplex_vertices(weights:&[LatentState])->Result<Vec<Vec<u64>>,String>{
 if weights.is_empty()||weights.len()>MAX_LATENT_STATES{return Err("latent vertex enumeration dimensions invalid".to_owned());}
 let mut unique:BTreeMap<String,Vec<u64>>=BTreeMap::new();
 let dimension=weights.len();
 let assignment_count=1usize.checked_shl((dimension-1) as u32).ok_or("latent vertex assignment count overflow")?;
 for pivot in 0..dimension{
  let others:Vec<usize>=(0..dimension).filter(|i|*i!=pivot).collect();
  for mask in 0..assignment_count{
   let mut vertex=vec![0u64;dimension];let mut assigned=0u64;
   for (bit,index) in others.iter().enumerate(){
    let value=if (mask>>bit)&1==0{weights[*index].probability.lower}else{weights[*index].probability.upper};
    vertex[*index]=value;assigned=assigned.checked_add(value).ok_or("latent vertex mass overflow")?;
   }
   if assigned>SCALE{continue;}
   let residual=SCALE-assigned;
   if residual<weights[pivot].probability.lower||residual>weights[pivot].probability.upper{continue;}
   vertex[pivot]=residual;
   let key=serde_json::to_string(&vertex).map_err(|e|format!("latent vertex serialization failed: {e}"))?;
   unique.insert(key,vertex);
  }
 }
 if unique.is_empty(){return Err("shared latent credal simplex has no enumerated vertices".to_owned());}
 Ok(unique.into_values().collect())
}
fn prior_extreme(
 priors:&BTreeMap<String,Bounds>,
 coefficients:&BTreeMap<String,i128>,
 maximize:bool
)->Result<i128,String>{
 if priors.keys().cloned().collect::<BTreeSet<_>>()!=coefficients.keys().cloned().collect::<BTreeSet<_>>(){
  return Err("prior coefficients must cover exact hypothesis set".to_owned());
 }
 let lower_sum:u64=priors.values().map(|x|x.lower).sum();
 if lower_sum>SCALE{return Err("prior lower mass exceeds scale".to_owned());}
 let mut total=priors.iter().try_fold(0i128,|sum,(id,p)|{
  sum.checked_add(p.lower as i128*coefficients[id]).ok_or_else(||"prior linear objective overflow".to_owned())
 })?;
 let mut remaining=SCALE-lower_sum;
 let mut ids:Vec<String>=priors.keys().cloned().collect();
 if maximize{
  ids.sort_by(|a,b|coefficients[b].cmp(&coefficients[a]).then_with(||a.cmp(b)));
 }else{
  ids.sort_by(|a,b|coefficients[a].cmp(&coefficients[b]).then_with(||a.cmp(b)));
 }
 for id in ids{
  let p=&priors[&id];let take=remaining.min(p.upper-p.lower);
  total=total.checked_add(take as i128*coefficients[&id]).ok_or("prior linear objective overflow")?;
  remaining-=take;if remaining==0{break;}
 }
 if remaining>0{return Err("prior upper bounds cannot allocate unit probability".to_owned());}
 Ok(total)
}
fn shared_likelihood_numerator(vertex:&[u64],likelihoods:&[u64])->Result<i128,String>{
 if vertex.len()!=likelihoods.len(){return Err("latent weight and conditional likelihood dimensions differ".to_owned());}
 vertex.iter().zip(likelihoods).try_fold(0i128,|sum,(weight,likelihood)|{
  sum.checked_add(*weight as i128*(*likelihood as i128)).ok_or_else(||"shared likelihood numerator overflow".to_owned())
 })
}
fn shared_support_lower(
 priors:&BTreeMap<String,Bounds>,
 weights:&[LatentState],
 likelihoods:&BTreeMap<String,Vec<u64>>
)->Result<u64,String>{
 let vertices=simplex_vertices(weights)?;
 let mut minimum=i128::MAX;
 for vertex in vertices{
  let coefficients=likelihoods.iter().map(|(hypothesis,values)|{
   Ok((hypothesis.clone(),shared_likelihood_numerator(&vertex,values)?))
  }).collect::<Result<BTreeMap<_,_>,String>>()?;
  minimum=minimum.min(prior_extreme(priors,&coefficients,false)?);
 }
 if minimum<0{return Err("shared event support became negative".to_owned());}
 Ok((minimum as u128/((SCALE as u128)*(SCALE as u128))) as u64)
}
fn shared_nominal_risk(
 priors:&BTreeMap<String,Bounds>,
 weights:&[LatentState],
 likelihoods:&BTreeMap<String,Vec<u64>>,
 safe:&BTreeSet<String>
)->Result<u64,String>{
 let mut numerator=0u128;let mut denominator=0u128;
 for (hypothesis,prior) in priors{
  let likelihood_numerator=shared_likelihood_numerator(
   &weights.iter().map(|z|z.probability.nominal).collect::<Vec<_>>(),
   &likelihoods[hypothesis]
  )?;
  let mass=prior.nominal as u128*(likelihood_numerator as u128);
  denominator+=mass;if !safe.contains(hypothesis){numerator+=mass;}
 }
 if denominator==0{return Ok(SCALE);}
 Ok(((numerator*SCALE as u128+denominator-1)/denominator).min(SCALE as u128) as u64)
}
fn shared_max_surplus(
 risk:u64,
 priors:&BTreeMap<String,Bounds>,
 weights:&[LatentState],
 likelihoods:&BTreeMap<String,Vec<u64>>,
 safe:&BTreeSet<String>
)->Result<i128,String>{
 let mut maximum=i128::MIN;
 for vertex in simplex_vertices(weights)?{
  let mut coefficients=BTreeMap::new();
  for (hypothesis,values) in likelihoods{
   let likelihood=shared_likelihood_numerator(&vertex,values)?;
   let coefficient=if safe.contains(hypothesis){
    -(risk as i128)*likelihood
   }else{
    (SCALE as i128-risk as i128)*likelihood
   };
   coefficients.insert(hypothesis.clone(),coefficient);
  }
  maximum=maximum.max(prior_extreme(priors,&coefficients,true)?);
 }
 Ok(maximum)
}
fn shared_robust_risk(
 priors:&BTreeMap<String,Bounds>,
 weights:&[LatentState],
 likelihoods:&BTreeMap<String,Vec<u64>>,
 safe:&BTreeSet<String>
)->Result<u64,String>{
 let support=shared_support_lower(priors,weights,likelihoods)?;
 if support==0{return Ok(SCALE);}
 let mut low=0u64;let mut high=SCALE;
 while low<high{
  let mid=low+(high-low)/2;
  if shared_max_surplus(mid,priors,weights,likelihoods,safe)?<=0{high=mid}else{low=mid+1;}
 }
 Ok(low)
}
fn rectangular_max_surplus(
 risk:u64,priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>,safe:&BTreeSet<String>
)->Result<i128,String>{
 let coefficients=likelihoods.iter().map(|(hypothesis,bounds)|{
  let coefficient=if safe.contains(hypothesis){
   -(risk as i128)*(bounds.lower as i128)
  }else{
   ((SCALE-risk) as i128)*(bounds.upper as i128)
  };
  (hypothesis.clone(),coefficient)
 }).collect::<BTreeMap<_,_>>();
 prior_extreme(priors,&coefficients,true)
}
fn rectangular_robust_risk(
 priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>,safe:&BTreeSet<String>
)->Result<u64,String>{
 let mut low=0u64;let mut high=SCALE;
 while low<high{
  let mid=low+(high-low)/2;
  if rectangular_max_surplus(mid,priors,likelihoods,safe)?<=0{high=mid}else{low=mid+1;}
 }
 Ok(low)
}
fn parse_risk_queries(
 oracle:&Value,
 hypotheses:&BTreeSet<String>,
 options:&BTreeMap<String,OptionDef>,
 factors:&BTreeMap<String,Factor>,
 latents:&[LatentState],
 kernels:&BTreeMap<(String,String),ConditionalKernel>
)->Result<Vec<Value>,String>{
 let rows=match oracle.get("risk_queries"){
  None=>return Ok(Vec::new()),
  Some(v)=>v.as_array().ok_or("oracle.risk_queries must be array")?
 };
 if rows.len()>64{return Err("oracle.risk_queries must contain at most 64 queries".to_owned());}
 let option_ids:BTreeSet<String>=options.keys().cloned().collect();
 let latent_weights:Vec<Bounds>=latents.iter().map(|z|z.probability.clone()).collect();
 let mut reports=Vec::new();let mut query_ids=BTreeSet::new();
 for (i,query) in rows.iter().enumerate(){
  let label=format!("oracle.risk_queries[{i}]");
  allowed(query,&["query_id","action_id","safe_hypotheses","prior_bounds","observed_outcomes"],&label)?;
  let query_id=string(query,"query_id",&label)?;
  if !query_ids.insert(query_id.clone()){return Err(format!("duplicate risk query {query_id:?}"));}
  let action_id=string(query,"action_id",&label)?;
  let safe_rows=get(query,"safe_hypotheses",&label)?.as_array().ok_or("safe_hypotheses must be array")?;
  let mut safe=BTreeSet::new();
  for raw in safe_rows{
   let id=raw.as_str().filter(|x|!x.trim().is_empty()).ok_or("safe hypothesis ID must be non-empty string")?;
   if !hypotheses.contains(id)||!safe.insert(id.to_owned()){return Err(format!("{label}.safe_hypotheses contains unknown or duplicate hypothesis"));}
  }
  let prior_rows=get(query,"prior_bounds",&label)?.as_array().ok_or("prior_bounds must be array")?;
  if prior_rows.len()!=hypotheses.len(){return Err(format!("{label}.prior_bounds must cover exact hypothesis set"));}
  let mut priors=BTreeMap::new();
  for (j,row) in prior_rows.iter().enumerate(){
   let pl=format!("{label}.prior_bounds[{j}]");
   allowed(row,&["hypothesis_id","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&pl)?;
   let id=string(row,"hypothesis_id",&pl)?;
   if !hypotheses.contains(&id){return Err(format!("{pl} references unknown hypothesis"));}
   let b=Bounds{lower:uint(row,"lower_probability_bps",&pl)?,upper:uint(row,"upper_probability_bps",&pl)?,nominal:uint(row,"nominal_probability_bps",&pl)?};
   if b.lower>b.nominal||b.nominal>b.upper||b.upper>SCALE{return Err(format!("{pl} violates probability interval ordering"));}
   if priors.insert(id.clone(),b).is_some(){return Err(format!("{pl} duplicates hypothesis {id:?}"));}
  }
  bounded_simplex(&priors.values().cloned().collect::<Vec<_>>(),&format!("{label}.prior_bounds"))?;
  if priors.keys().cloned().collect::<BTreeSet<_>>()!=*hypotheses{return Err(format!("{label}.prior_bounds must cover exact hypothesis set"));}
  let observed=object(get(query,"observed_outcomes",&label)?,"observed_outcomes")?;
  if observed.keys().cloned().collect::<BTreeSet<_>>()!=option_ids{return Err(format!("{label}.observed_outcomes must cover every option exactly"));}
  for (option,value) in observed{
   let outcome=value.as_str().filter(|x|!x.trim().is_empty()).ok_or("observed outcome must be non-empty string")?;
   if !factors[&options[option].factor_id].outcomes.contains(outcome){return Err(format!("{label} references unknown outcome {outcome:?}"));}
  }
  let mut likelihoods=BTreeMap::new();
  for hypothesis in hypotheses{
   let mut by_latent=Vec::new();
   for latent in latents{
    let kernel=&kernels[&(hypothesis.clone(),latent.id.clone())];
    let probability=kernel.rows.iter().filter(|row|{
     observed.iter().all(|(option,outcome)|row.outcomes.get(option).map(String::as_str)==Some(outcome.as_str()))
    }).map(|row|row.probability).sum();
    by_latent.push(probability);
   }
   likelihoods.insert(hypothesis.clone(),by_latent);
  }
  let shared_support=shared_support_lower(&priors,latents,&likelihoods)?;
  let nominal_risk=shared_nominal_risk(&priors,latents,&likelihoods,&safe)?;
  let shared_risk=shared_robust_risk(&priors,latents,&likelihoods,&safe)?;
  let mut rectangular_likelihoods=BTreeMap::new();
  for hypothesis in hypotheses{
   let coeffs=&likelihoods[hypothesis];
   rectangular_likelihoods.insert(hypothesis.clone(),Bounds{
    lower:weighted_extreme(&latent_weights,coeffs,false)?,
    upper:weighted_extreme(&latent_weights,coeffs,true)?,
    nominal:weighted_nominal(&latent_weights,coeffs)?
   });
  }
  let rectangular_risk=rectangular_robust_risk(&priors,&rectangular_likelihoods,&safe)?;
  reports.push(json!({
   "query_id":query_id,"action_id":action_id,
   "safe_hypotheses":safe.iter().collect::<Vec<_>>(),
   "observed_outcomes":observed,
   "support_lower_bps":shared_support,
   "nominal_posterior_risk_bps":nominal_risk,
   "shared_latent_worst_case_posterior_risk_bps":shared_risk,
   "rectangular_hull_worst_case_posterior_risk_bps":rectangular_risk,
   "rectangular_excess_risk_bps":rectangular_risk.saturating_sub(shared_risk),
   "per_hypothesis_likelihood_bounds":rectangular_likelihoods.iter().map(|(h,b)|json!({
    "hypothesis_id":h,"lower":b.lower,"nominal":b.nominal,"upper":b.upper
   })).collect::<Vec<_>>()
  }));
 }
 Ok(reports)
}
const MAX_DECISION_VOI_QUERIES:usize=16;
const MAX_DECISION_RULES:usize=10_000;
const MAX_LOSS_MICROS:u64=1_000_000_000_000;

fn ceil_ratio(numerator:u128,denominator:u128,label:&str)->Result<u64,String>{
 if denominator==0{return Err(format!("{label} denominator is zero"));}
 let value=(numerator+denominator-1)/denominator;
 u64::try_from(value).map_err(|_|format!("{label} exceeds u64"))
}
fn signed_net_value(stop_loss:u64,post_loss:u64,cost:u64,label:&str)->Result<i64,String>{
 let value=stop_loss as i128-post_loss as i128-cost as i128;
 i64::try_from(value).map_err(|_|format!("{label} exceeds signed 64-bit range"))
}
fn enumerate_decision_rules(
 outcomes:&[String],
 actions:&[String],
 index:usize,
 current:&mut BTreeMap<String,String>,
 output:&mut Vec<BTreeMap<String,String>>
)->Result<(),String>{
 if output.len()>MAX_DECISION_RULES{return Err("decision rule count exceeds evaluator cap".to_owned());}
 if index==outcomes.len(){output.push(current.clone());return Ok(());}
 for action in actions{
  current.insert(outcomes[index].clone(),action.clone());
  enumerate_decision_rules(outcomes,actions,index+1,current,output)?;
  current.remove(&outcomes[index]);
 }
 Ok(())
}
fn robust_stop_loss(
 priors:&BTreeMap<String,Bounds>,
 losses:&BTreeMap<String,BTreeMap<String,u64>>,
 action:&str
)->Result<u64,String>{
 let coefficients=losses.iter().map(|(hypothesis,actions)|{
  Ok((hypothesis.clone(),*actions.get(action).ok_or_else(||format!("loss matrix missing action {action:?}"))? as i128))
 }).collect::<Result<BTreeMap<_,_>,String>>()?;
 let worst=prior_extreme(priors,&coefficients,true)?;
 ceil_ratio(worst as u128,SCALE as u128,"robust stop loss")
}
fn nominal_stop_loss(
 priors:&BTreeMap<String,Bounds>,
 losses:&BTreeMap<String,BTreeMap<String,u64>>,
 action:&str
)->Result<u64,String>{
 let numerator=priors.iter().try_fold(0u128,|sum,(hypothesis,prior)|{
  let loss=*losses[hypothesis].get(action).ok_or_else(||format!("loss matrix missing action {action:?}"))?;
  sum.checked_add(prior.nominal as u128*loss as u128).ok_or_else(||"nominal stop loss overflow".to_owned())
 })?;
 ceil_ratio(numerator,SCALE as u128,"nominal stop loss")
}
fn decision_rule_coefficients(
 hypothesis:&str,
 vertex:&[u64],
 latents:&[LatentState],
 kernels:&BTreeMap<(String,String),ConditionalKernel>,
 option:&str,
 rule:&BTreeMap<String,String>,
 losses:&BTreeMap<String,BTreeMap<String,u64>>
)->Result<i128,String>{
 if vertex.len()!=latents.len(){return Err("decision rule latent vertex dimensions differ".to_owned());}
 let mut total=0i128;
 for (latent_index,latent) in latents.iter().enumerate(){
  let kernel=kernels.get(&(hypothesis.to_owned(),latent.id.clone()))
   .ok_or_else(||format!("missing conditional kernel for {hypothesis:?}/{:?}",latent.id))?;
  let conditional=kernel.rows.iter().try_fold(0i128,|sum,row|{
   let outcome=row.outcomes.get(option).ok_or_else(||format!("kernel row lacks option {option:?}"))?;
   let action=rule.get(outcome).ok_or_else(||format!("decision rule lacks outcome {outcome:?}"))?;
   let loss=*losses.get(hypothesis).and_then(|x|x.get(action))
    .ok_or_else(||format!("loss matrix missing {hypothesis:?}/{action:?}"))?;
   sum.checked_add(row.probability as i128*loss as i128)
    .ok_or_else(||"conditional decision loss overflow".to_owned())
  })?;
  total=total.checked_add(vertex[latent_index] as i128*conditional)
   .ok_or_else(||"shared latent decision loss overflow".to_owned())?;
 }
 Ok(total)
}
fn robust_rule_loss(
 priors:&BTreeMap<String,Bounds>,
 latents:&[LatentState],
 kernels:&BTreeMap<(String,String),ConditionalKernel>,
 option:&str,
 rule:&BTreeMap<String,String>,
 losses:&BTreeMap<String,BTreeMap<String,u64>>
)->Result<u64,String>{
 let mut worst=i128::MIN;
 for vertex in simplex_vertices(latents)?{
  let mut coefficients=BTreeMap::new();
  for hypothesis in priors.keys(){
   coefficients.insert(hypothesis.clone(),decision_rule_coefficients(
    hypothesis,&vertex,latents,kernels,option,rule,losses
   )?);
  }
  worst=worst.max(prior_extreme(priors,&coefficients,true)?);
 }
 if worst<0{return Err("robust expected loss became negative".to_owned());}
 ceil_ratio(worst as u128,(SCALE as u128).pow(3),"robust post-measurement loss")
}
fn nominal_rule_loss(
 priors:&BTreeMap<String,Bounds>,
 latents:&[LatentState],
 kernels:&BTreeMap<(String,String),ConditionalKernel>,
 option:&str,
 rule:&BTreeMap<String,String>,
 losses:&BTreeMap<String,BTreeMap<String,u64>>
)->Result<u64,String>{
 let vertex:Vec<u64>=latents.iter().map(|z|z.probability.nominal).collect();
 let numerator=priors.iter().try_fold(0u128,|sum,(hypothesis,prior)|{
  let conditional=decision_rule_coefficients(hypothesis,&vertex,latents,kernels,option,rule,losses)?;
  sum.checked_add(prior.nominal as u128*conditional as u128)
   .ok_or_else(||"nominal post-measurement loss overflow".to_owned())
 })?;
 ceil_ratio(numerator,(SCALE as u128).pow(3),"nominal post-measurement loss")
}
fn parse_decision_voi_queries(
 oracle:&Value,
 hypotheses:&BTreeSet<String>,
 options:&BTreeMap<String,OptionDef>,
 factors:&BTreeMap<String,Factor>,
 latents:&[LatentState],
 kernels:&BTreeMap<(String,String),ConditionalKernel>
)->Result<Vec<Value>,String>{
 let rows=match oracle.get("decision_voi_queries"){
  None=>return Ok(Vec::new()),
  Some(v)=>v.as_array().ok_or("oracle.decision_voi_queries must be array")?
 };
 if rows.len()>MAX_DECISION_VOI_QUERIES{return Err(format!("decision_voi_queries exceeds {MAX_DECISION_VOI_QUERIES}"));}
 let mut reports=Vec::new();let mut ids=BTreeSet::new();
 for (i,query) in rows.iter().enumerate(){
  let label=format!("oracle.decision_voi_queries[{i}]");
  allowed(query,&["query_id","loss_currency","prior_bounds","action_losses","option_costs"],&label)?;
  let query_id=string(query,"query_id",&label)?;
  if !ids.insert(query_id.clone()){return Err(format!("duplicate decision VOI query {query_id:?}"));}
  let currency=string(query,"loss_currency",&label)?;
  if currency.len()!=3||!currency.chars().all(|x|x.is_ascii_uppercase()){
   return Err(format!("{label}.loss_currency must be a three-letter uppercase ISO 4217 code"));
  }
  let prior_rows=get(query,"prior_bounds",&label)?.as_array().ok_or("decision VOI prior_bounds must be array")?;
  if prior_rows.len()!=hypotheses.len(){return Err(format!("{label}.prior_bounds must cover exact hypothesis set"));}
  let mut priors=BTreeMap::new();
  for (j,row) in prior_rows.iter().enumerate(){
   let pl=format!("{label}.prior_bounds[{j}]");
   allowed(row,&["hypothesis_id","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&pl)?;
   let id=string(row,"hypothesis_id",&pl)?;
   if !hypotheses.contains(&id){return Err(format!("{pl} references unknown hypothesis"));}
   let b=Bounds{lower:uint(row,"lower_probability_bps",&pl)?,upper:uint(row,"upper_probability_bps",&pl)?,nominal:uint(row,"nominal_probability_bps",&pl)?};
   if b.lower>b.nominal||b.nominal>b.upper||b.upper>SCALE{return Err(format!("{pl} violates probability interval ordering"));}
   if priors.insert(id.clone(),b).is_some(){return Err(format!("{pl} duplicates hypothesis {id:?}"));}
  }
  bounded_simplex(&priors.values().cloned().collect::<Vec<_>>(),&format!("{label}.prior_bounds"))?;
  if priors.keys().cloned().collect::<BTreeSet<_>>()!=*hypotheses{return Err(format!("{label}.prior_bounds must cover exact hypothesis set"));}
  let raw_losses=get(query,"action_losses",&label)?.as_array().ok_or("action_losses must be array")?;
  if raw_losses.is_empty()||raw_losses.len()>hypotheses.len().saturating_mul(16){
   return Err(format!("{label}.action_losses has invalid row count"));
  }
  let mut losses:BTreeMap<String,BTreeMap<String,u64>>=BTreeMap::new();
  for (j,row) in raw_losses.iter().enumerate(){
   let ll=format!("{label}.action_losses[{j}]");
   allowed(row,&["hypothesis_id","action_id","loss_micros"],&ll)?;
   let hypothesis=string(row,"hypothesis_id",&ll)?;
   let action=string(row,"action_id",&ll)?;
   if !hypotheses.contains(&hypothesis){return Err(format!("{ll} references unknown hypothesis"));}
   let value=uint(row,"loss_micros",&ll)?;
   if value>MAX_LOSS_MICROS{return Err(format!("{ll}.loss_micros exceeds registered bound"));}
   if losses.entry(hypothesis.clone()).or_default().insert(action.clone(),value).is_some(){
    return Err(format!("{ll} duplicates hypothesis/action loss"));
   }
  }
  let action_ids=losses.values().flat_map(|x|x.keys().cloned()).collect::<BTreeSet<_>>();
  if action_ids.is_empty()||action_ids.len()>16{return Err(format!("{label} requires 1..=16 actions"));}
  if losses.keys().cloned().collect::<BTreeSet<_>>()!=*hypotheses||
   losses.values().any(|x|x.keys().cloned().collect::<BTreeSet<_>>()!=action_ids){
   return Err(format!("{label}.action_losses must cover the exact hypothesis x action product"));
  }
  let cost_rows=get(query,"option_costs",&label)?.as_array().ok_or("option_costs must be array")?;
  if cost_rows.len()!=options.len(){return Err(format!("{label}.option_costs must cover exact option set"));}
  let mut option_costs=BTreeMap::new();
  for (j,row) in cost_rows.iter().enumerate(){
   let cl=format!("{label}.option_costs[{j}]");
   allowed(row,&["option_id","cost_micros"],&cl)?;
   let option=string(row,"option_id",&cl)?;
   if !options.contains_key(&option){return Err(format!("{cl} references unknown option"));}
   let cost=uint(row,"cost_micros",&cl)?;
   if option_costs.insert(option.clone(),cost).is_some(){return Err(format!("{cl} duplicates option"));}
  }
  if option_costs.keys().cloned().collect::<BTreeSet<_>>()!=options.keys().cloned().collect(){
   return Err(format!("{label}.option_costs must cover exact option set"));
  }
  let mut stop_scores=Vec::new();
  for action in &action_ids{
   let robust=robust_stop_loss(&priors,&losses,action)?;
   let nominal=nominal_stop_loss(&priors,&losses,action)?;
   stop_scores.push((robust,nominal,action.clone()));
  }
  stop_scores.sort_by(|a,b|a.0.cmp(&b.0).then_with(||a.1.cmp(&b.1)).then_with(||a.2.cmp(&b.2)));
  let (stop_robust,stop_nominal,stop_action)=stop_scores.first().ok_or("no stop action score")?.clone();
  let mut option_reports=Vec::new();
  for (option_id,cost) in &option_costs{
   let factor_id=&options[option_id].factor_id;
   let option_outcomes=factors[factor_id].outcomes.iter().cloned().collect::<Vec<_>>();
   let mut rules=Vec::new();
   enumerate_decision_rules(&option_outcomes,&action_ids.iter().cloned().collect::<Vec<_>>(),0,&mut BTreeMap::new(),&mut rules)?;
   if rules.len()>MAX_DECISION_RULES{return Err(format!("{label} decision rules exceed evaluator cap"));}
   let mut scored=Vec::new();
   for rule in rules{
    let robust=robust_rule_loss(&priors,latents,kernels,option_id,&rule,&losses)?;
    let nominal=nominal_rule_loss(&priors,latents,kernels,option_id,&rule,&losses)?;
    let canonical=serde_json::to_string(&rule).map_err(|e|format!("decision rule serialization failed: {e}"))?;
    scored.push((robust,nominal,canonical,rule));
   }
   scored.sort_by(|a,b|a.0.cmp(&b.0).then_with(||a.1.cmp(&b.1)).then_with(||a.2.cmp(&b.2)));
   let decision_rule_count=scored.len();
   let (post_robust,post_nominal,_,best_rule)=scored.into_iter().next().ok_or("no decision rules generated")?;
   let net_robust=signed_net_value(stop_robust,post_robust,*cost,"net robust VOI")?;
   let net_nominal=signed_net_value(stop_nominal,post_nominal,*cost,"net nominal VOI")?;
   option_reports.push(json!({
    "option_id":option_id,"measurement_cost_micros":cost,
    "robust_post_measurement_loss_micros":post_robust,
    "nominal_post_measurement_loss_micros":post_nominal,
    "net_robust_decision_value_micros":net_robust,
    "net_nominal_decision_value_micros":net_nominal,
    "continue_recommended":net_robust>0,
    "decision_rule":best_rule,
    "decision_rule_count":decision_rule_count
   }));
  }
  option_reports.sort_by(|a,b|{
   b["net_robust_decision_value_micros"].as_i64().unwrap_or(i64::MIN)
    .cmp(&a["net_robust_decision_value_micros"].as_i64().unwrap_or(i64::MIN))
    .then_with(||a["option_id"].as_str().cmp(&b["option_id"].as_str()))
  });
  let recommended=option_reports.first().ok_or("no option report")?;
  let should_continue=recommended["net_robust_decision_value_micros"].as_i64().unwrap_or(i64::MIN)>0;
  reports.push(json!({
   "query_id":query_id,"loss_currency":currency,
   "loss_unit":"micro-units of the declared currency",
   "robust_stop_action_id":stop_action,
   "robust_stop_loss_micros":stop_robust,
   "nominal_stop_loss_micros":stop_nominal,
   "recommended_next_step":if should_continue{"measure"}else{"stop"},
   "recommended_option_id":if should_continue{recommended["option_id"].clone()}else{Value::Null},
   "best_net_robust_decision_value_micros":recommended["net_robust_decision_value_micros"],
   "option_evaluations":option_reports
  }));
 }
 Ok(reports)
}
fn evaluate(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=get(&bundle,"candidate_view","bundle")?;
 let oracle=get(&bundle,"oracle","bundle")?;
 allowed(oracle,&["decision_cutoff","latent_states","conditional_kernels","risk_queries","decision_voi_queries"],"oracle")?;
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
 let risk_queries=parse_risk_queries(&oracle,&hypotheses,&options,&factors,&latents,&kernels)?;
 let decision_voi=parse_decision_voi_queries(&oracle,&hypotheses,&options,&factors,&latents,&kernels)?;
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
  "risk_queries":risk_queries,
  "decision_voi":decision_voi,
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
 #[test]fn nonrepresentable_nominal_mixture_rounds_without_rejecting(){
  let rounded=weighted_nominal(
   &[Bounds{lower:4000,upper:6000,nominal:5000},Bounds{lower:4000,upper:6000,nominal:5000}],
   &[3333,3334]
  ).expect("rounding is defined");
  assert_eq!(rounded,3334);
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
 #[test]fn shared_latent_posterior_risk_is_tighter_than_rectangular_hull(){
  // Exact prior: safe-world=hazard=50%. For pass/pass:
  // shared w0=0.4 gives safe=0.56 and hazard=0.34, the worst coupled ratio.
  // The independent product hull combines safe lower=0.44 with hazard upper=0.34,
  // although those extrema cannot occur at the same shared latent weight.
  let coupled=(3400u64*10000+9000-1)/9000;
  let rectangular=(3400u64*10000+3400+4400-1)/(3400+4400);
  assert_eq!(coupled,3778);
  assert_eq!(rectangular,4359);
  assert!(coupled<rectangular);
 }
 #[test]fn risk_probe_reports_shared_and_rectangular_posterior_bounds(){
  let r=report();
  let q=&r["risk_queries"][0];
  assert_eq!(q["query_id"],"pass-pass-release");
  assert_eq!(q["support_lower_bps"],3500);
  assert_eq!(q["nominal_posterior_risk_bps"],3750);
  assert_eq!(q["shared_latent_worst_case_posterior_risk_bps"],3778);
  assert_eq!(q["rectangular_hull_worst_case_posterior_risk_bps"],4359);
  assert_eq!(q["rectangular_excess_risk_bps"],581);
 }
 #[test]fn risk_probe_cannot_be_injected_by_candidate(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["candidate_view"]["risk_queries"]=b["oracle"]["risk_queries"].clone();
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));
 }
 #[test]fn risk_probe_rejects_duplicate_query_id(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  let q=b["oracle"]["risk_queries"][0].clone();
  b["oracle"]["risk_queries"].as_array_mut().unwrap().push(q);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("duplicate risk query"));
 }
 #[test]fn risk_probe_requires_exact_registered_outcome_vector(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["risk_queries"][0]["observed_outcomes"]["gate-b"]=json!("unknown");
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown outcome"));
 }
 #[test]fn risk_probe_prior_must_cover_all_hypotheses_and_normalize(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["risk_queries"][0]["prior_bounds"][0]["lower_probability_bps"]=json!(6000);
  b["oracle"]["risk_queries"][0]["prior_bounds"][0]["upper_probability_bps"]=json!(6000);
  b["oracle"]["risk_queries"][0]["prior_bounds"][0]["nominal_probability_bps"]=json!(6000);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("infeasible"));
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
