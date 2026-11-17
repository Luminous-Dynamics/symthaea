//! D6.17.2.2.1 exact scoring of a submitted adaptive policy over stochastic joint kernels.
//!
//! Every complete evaluator-owned joint outcome vector is replayed through the
//! submitted adaptive tree. Branch likelihood intervals are derived from the
//! full joint bounded-simplex kernel; channel marginals are never multiplied.
//! Expected cost is optimized exactly over each conditional kernel and then
//! over the prior credal set, under an explicit product-set assumption across
//! hypotheses. This is a fixed-policy scorer, not a policy-search oracle.

#[allow(dead_code)]
#[path="noisy-channel-risk.rs"]
mod noisy_channel;
#[allow(dead_code)]
#[path="stochastic-kernel-validator.rs"]
mod joint_kernel;

use chrono::{DateTime,FixedOffset};
use serde_json::{json,Value};
use std::collections::{BTreeMap,BTreeSet};
use std::{env,fs,process};

const SCALE:u64=10_000;
const MAX_HYPOTHESES:usize=64;
const MAX_OPTIONS:usize=6;
const MAX_NODES:usize=64;
const MAX_JOINT_ROWS:usize=4096;
const CLAIM_CEILING:&str="Exact evaluation of a submitted finite adaptive policy over the evaluator-owned finite joint-outcome credal kernels and bounded prior set. Branch event intervals and conditional expected costs are exact for the registered bounded-simplex sets; the outer expected-cost bound assumes kernel ambiguity sets are independent across hypotheses. It does not optimize over all policies, model stochastic execution costs, authenticate joint-kernel calibration, establish model completeness, or authorize production actions.";

#[derive(Clone,Debug)]
struct Bounds{lower:u64,upper:u64,nominal:u64}
#[derive(Clone,Debug)]
struct Action{read_only:bool,irreversible:bool,hard_gate:bool,max_risk:u64}
#[derive(Clone,Debug)]
struct OptionDef{factor_id:String,available_at:DateTime<FixedOffset>,cost_micros:u64}
#[derive(Clone,Debug)]
struct JointRow{outcomes:BTreeMap<String,String>,probability:Bounds}
#[derive(Clone,Debug)]
struct Kernel{kernel_id:String,authority:String,rows:Vec<JointRow>}
#[derive(Clone,Debug)]
struct Node{
 id:String,kind:String,option_id:Option<String>,at:Option<String>,
 branches:BTreeMap<String,String>,decision_kind:Option<String>,
 action_id:Option<String>,claimed:u64
}
#[derive(Clone,Debug)]
struct PolicyTree{root:String,nodes:BTreeMap<String,Node>}
#[derive(Clone,Debug)]
struct LeafPath{
 leaf_id:String,decision_kind:String,action_id:String,claimed:u64,
 option_ids:Vec<String>,world_row_count:usize,path_cost:u64
}
#[derive(Clone,Debug)]
struct PolicyMetrics{
 report:Value,coverage:u64,nominal_expected_cost:u64,worst_expected_cost:u64,
 max_path_cost:u64,feasible:bool
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
fn probability(v:&Value,label:&str)->Result<Bounds,String>{
 let (lk,uk,nk)=("lower_probability_bps","upper_probability_bps","nominal_probability_bps");
 allowed(v,&[lk,uk,nk],label)?;
 let b=Bounds{lower:uint(v,lk,label)?,upper:uint(v,uk,label)?,nominal:uint(v,nk,label)?};
 if b.lower>b.nominal||b.nominal>b.upper||b.upper>SCALE{
  return Err(format!("{label} must satisfy 0 <= lower <= nominal <= upper <= 10000"));
 }
 Ok(b)
}
fn parse_priors(candidate:&Value,oracle:&Value)->Result<BTreeMap<String,Bounds>,String>{
 let cv=get(candidate,"hypothesis_bounds","candidate_view")?.as_array().ok_or("candidate hypothesis_bounds must be array")?;
 let ov=get(oracle,"hypothesis_bounds","oracle")?.as_array().ok_or("oracle hypothesis_bounds must be array")?;
 if cv.is_empty()||cv.len()>MAX_HYPOTHESES||cv!=ov{return Err("candidate prior bounds must exactly match 1..=64 evaluator-authoritative bounds".to_owned());}
 let mut out=BTreeMap::new();let mut lower=0u64;let mut upper=0u64;let mut nominal=0u64;
 for (i,row) in cv.iter().enumerate(){
  let label=format!("candidate_view.hypothesis_bounds[{i}]");
  allowed(row,&["hypothesis_id","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&label)?;
  let id=string(row,"hypothesis_id",&label)?;
  let b=probability(row,&label)?;
  lower=lower.checked_add(b.lower).ok_or("prior lower mass overflow")?;
  upper=upper.checked_add(b.upper).ok_or("prior upper mass overflow")?;
  nominal=nominal.checked_add(b.nominal).ok_or("prior nominal mass overflow")?;
  if out.insert(id.clone(),b).is_some(){return Err(format!("duplicate hypothesis ID {id:?}"));}
 }
 if lower>SCALE||upper<SCALE||nominal!=SCALE{return Err(format!("prior bounded simplex is infeasible: lower={lower}, upper={upper}, nominal={nominal}"));}
 Ok(out)
}
fn parse_actions(candidate:&Value,oracle:&Value)->Result<BTreeMap<String,Action>,String>{
 let cv=get(candidate,"actions","candidate_view")?.as_array().ok_or("candidate actions must be array")?;
 let ov=get(oracle,"actions","oracle")?.as_array().ok_or("oracle actions must be array")?;
 if cv.is_empty()||cv.len()>16||cv!=ov{return Err("candidate actions must exactly match evaluator-authoritative 1..=16 policies".to_owned());}
 let mut out=BTreeMap::new();
 for (i,row) in cv.iter().enumerate(){
  let label=format!("candidate_view.actions[{i}]");
  allowed(row,&["action_id","read_only","irreversible","hard_gate","max_residual_risk_bps"],&label)?;
  let id=string(row,"action_id",&label)?;
  let read_only=get(row,"read_only",&label)?.as_bool().ok_or("read_only must be boolean")?;
  let irreversible=get(row,"irreversible",&label)?.as_bool().ok_or("irreversible must be boolean")?;
  let hard=get(row,"hard_gate",&label)?.as_bool().ok_or("hard_gate must be boolean")?;
  let max_risk=uint(row,"max_residual_risk_bps",&label)?;
  if max_risk>SCALE||(hard&&max_risk!=0)||(irreversible&&!hard)||(read_only&&(irreversible||hard||max_risk!=0)){
   return Err(format!("invalid action policy {id:?}"));
  }
  if out.insert(id,Action{read_only,irreversible,hard_gate:hard,max_risk}).is_some(){return Err("duplicate action ID".to_owned());}
 }
 Ok(out)
}
fn canonical_defer(actions:&BTreeMap<String,Action>,safe:&BTreeMap<String,BTreeSet<String>>)->Result<String,String>{
 let ids:Vec<String>=actions.iter().filter(|(id,a)|{
  a.read_only&&!a.irreversible&&!a.hard_gate&&a.max_risk==0&&safe.values().all(|v|v.contains(*id))
 }).map(|(id,_)|id.clone()).collect();
 if ids.len()!=1{return Err(format!("exactly one globally safe read-only defer action is required; found {}",ids.len()));}
 Ok(ids[0].clone())
}
fn parse_safety(oracle:&Value,ids:&BTreeSet<String>,actions:&BTreeMap<String,Action>)->Result<BTreeMap<String,BTreeSet<String>>,String>{
 let values=get(oracle,"hypothesis_safety","oracle")?.as_array().ok_or("hypothesis_safety must be an array")?;
 let mut out=BTreeMap::new();
 for (i,row) in values.iter().enumerate(){
  let label=format!("oracle.hypothesis_safety[{i}]");
  allowed(row,&["hypothesis_id","safe_actions"],&label)?;
  let id=string(row,"hypothesis_id",&label)?;
  if !ids.contains(&id){return Err(format!("{label} references unknown hypothesis {id:?}"));}
  let raw=get(row,"safe_actions",&label)?.as_array().ok_or("safe_actions must be array")?;
  let mut set=BTreeSet::new();
  for a in raw{
   let a=a.as_str().filter(|x|!x.trim().is_empty()).ok_or("safe action IDs must be non-empty strings")?;
   if !actions.contains_key(a){return Err(format!("{label} references unknown action {a:?}"));}
   set.insert(a.to_owned());
  }
  if out.insert(id.clone(),set).is_some(){return Err(format!("duplicate safety label for {id:?}"));}
 }
 if out.keys().cloned().collect::<BTreeSet<_>>()!=*ids{return Err("oracle safety labels must cover the exact hypothesis set".to_owned());}
 Ok(out)
}
fn parse_options(candidate:&Value,oracle:&Value,cutoff:&DateTime<FixedOffset>)->Result<BTreeMap<String,OptionDef>,String>{
 let cv=get(candidate,"options","candidate_view")?.as_array().ok_or("candidate options must be array")?;
 let ov=get(oracle,"options","oracle")?.as_array().ok_or("oracle options must be array")?;
 if cv.is_empty()||cv.len()>MAX_OPTIONS||cv!=ov{return Err("candidate options must exactly match evaluator-authoritative 1..=6 options".to_owned());}
 let costs=get(oracle,"cost_options","oracle")?.as_array().ok_or("oracle cost_options must be array")?;
 if costs.len()!=cv.len(){return Err("cost_options must cover the exact option set".to_owned());}
 let mut cost_map=BTreeMap::new();
 for (i,row) in costs.iter().enumerate(){
  let label=format!("oracle.cost_options[{i}]");
  allowed(row,&["option_id","cost_micros"],&label)?;
  let id=string(row,"option_id",&label)?;let cost=uint(row,"cost_micros",&label)?;
  if cost_map.insert(id.clone(),cost).is_some(){return Err(format!("duplicate cost option {id:?}"));}
 }
 let mut out=BTreeMap::new();
 for (i,row) in cv.iter().enumerate(){
  let label=format!("candidate_view.options[{i}]");
  allowed(row,&["option_id","available_at","factor_id"],&label)?;
  let id=string(row,"option_id",&label)?;
  let available=string(row,"available_at",&label)?;
  let available_at=time(&available,&label)?;
  let factor=string(row,"factor_id",&label)?;
  if available_at> *cutoff{return Err(format!("option {id:?} is unavailable at decision cutoff"));}
  let cost=*cost_map.get(&id).ok_or_else(||format!("missing evaluator cost for option {id:?}"))?;
  if out.insert(id.clone(),OptionDef{factor_id:factor,available_at,cost_micros:cost}).is_some(){return Err(format!("duplicate option {id:?}"));}
 }
 if cost_map.keys().cloned().collect::<BTreeSet<_>>()!=out.keys().cloned().collect(){return Err("cost options must exactly cover candidate options".to_owned());}
 Ok(out)
}
fn factor_outcomes(candidate:&Value,options:&BTreeMap<String,OptionDef>)->Result<BTreeMap<String,BTreeSet<String>>,String>{
 let factors=get(candidate,"factors","candidate_view")?.as_array().ok_or("candidate factors must be array")?;
 let mut out=BTreeMap::new();
 for (i,row) in factors.iter().enumerate(){
  let label=format!("candidate_view.factors[{i}]");
  allowed(row,&["factor_id","dependence_group","source_ids","calibration_ref","likelihoods"],&label)?;
  let id=string(row,"factor_id",&label)?;
  let likelihoods=get(row,"likelihoods",&label)?.as_array().ok_or("factor likelihoods must be array")?;
  let first=likelihoods.first().ok_or("factor likelihoods cannot be empty")?;
  let names=object(get(first,"outcomes",&label)?,"factor outcomes")?.keys().cloned().collect::<BTreeSet<_>>();
  if names.is_empty(){return Err(format!("{label} has no outcomes"));}
  for hyp in likelihoods{
   let other=object(get(hyp,"outcomes",&label)?,"factor outcomes")?.keys().cloned().collect::<BTreeSet<_>>();
   if other!=names{return Err(format!("{label} outcome sets differ across hypotheses"));}
  }
  if out.insert(id.clone(),names).is_some(){return Err(format!("duplicate factor {id:?}"));}
 }
 let referenced: BTreeSet<String>=options.values().map(|x|x.factor_id.clone()).collect();
 if referenced!=out.keys().cloned().collect(){return Err("factor registry differs from option factor references".to_owned());}
 Ok(out)
}
fn parse_kernels(candidate:&Value,oracle:&Value,options:&BTreeMap<String,OptionDef>,cutoff:&str)->Result<BTreeMap<String,Kernel>,String>{
 let compact_candidate=json!({
  "scenario_id":candidate["scenario_id"],"decision_cutoff":cutoff,
  "options":candidate["options"],"factors":candidate["factors"]
 });
 let compact_oracle=json!({"world_kernels":oracle["world_kernels"]});
 let validation=json!({"candidate_view":compact_candidate,"oracle":compact_oracle});
 joint_kernel::validate_stochastic_joint_kernel_json(&validation.to_string())?;
 let values=get(oracle,"world_kernels","oracle")?.as_array().ok_or("world_kernels must be array")?;
 let mut out=BTreeMap::new();
 for (i,kernel) in values.iter().enumerate(){
  let label=format!("oracle.world_kernels[{i}]");
  let hypothesis=string(kernel,"hypothesis_id",&label)?;
  let kernel_id=string(kernel,"kernel_id",&label)?;
  let authority=string(kernel,"authority",&label)?;
  let rows=get(kernel,"joint_outcomes",&label)?.as_array().ok_or("joint_outcomes must be array")?;
  if rows.is_empty()||rows.len()>MAX_JOINT_ROWS{return Err(format!("{label} has invalid row count"));}
  let mut parsed=Vec::new();
  for (j,row) in rows.iter().enumerate(){
   let rl=format!("{label}.joint_outcomes[{j}]");
   allowed(row,&["outcomes","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&rl)?;
   let o=object(get(row,"outcomes",&rl)?,"joint outcomes")?;
   let mut outcomes=BTreeMap::new();
   for (option,value) in o{
    if !options.contains_key(option){return Err(format!("{rl} references unknown option {option:?}"));}
    let outcome=value.as_str().filter(|x|!x.trim().is_empty()).ok_or("joint outcome must be a non-empty string")?;
    outcomes.insert(option.clone(),outcome.to_owned());
   }
   let b=Bounds{lower:uint(row,"lower_probability_bps",&rl)?,upper:uint(row,"upper_probability_bps",&rl)?,nominal:uint(row,"nominal_probability_bps",&rl)?};
   parsed.push(JointRow{outcomes,probability:b});
  }
  if out.insert(hypothesis.clone(),Kernel{kernel_id,authority,rows:parsed}).is_some(){return Err(format!("duplicate kernel for {hypothesis:?}"));}
 }
 Ok(out)
}
fn event_bounds(rows:&[JointRow],membership:&BTreeSet<usize>)->Result<Bounds,String>{
 let mut lower_event=0u64;let mut upper_event=0u64;let mut lower_other=0u64;let mut upper_other=0u64;let mut nominal=0u64;
 for (i,row) in rows.iter().enumerate(){
  if membership.contains(&i){
   lower_event=lower_event.checked_add(row.probability.lower).ok_or("event lower overflow")?;
   upper_event=upper_event.checked_add(row.probability.upper).ok_or("event upper overflow")?;
   nominal=nominal.checked_add(row.probability.nominal).ok_or("event nominal overflow")?;
  }else{
   lower_other=lower_other.checked_add(row.probability.lower).ok_or("complement lower overflow")?;
   upper_other=upper_other.checked_add(row.probability.upper).ok_or("complement upper overflow")?;
  }
 }
 let lower=lower_event.max(SCALE.saturating_sub(upper_other));
 let upper=upper_event.min(SCALE.saturating_sub(lower_other));
 if lower>upper||nominal<lower||nominal>upper{return Err("joint-kernel event interval is infeasible".to_owned());}
 Ok(Bounds{lower,upper,nominal})
}
fn support_lower(priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>)->Result<u64,String>{
 let lower_mass: u64=priors.values().map(|x|x.lower).sum();
 let mut value:u128=priors.iter().map(|(id,p)|p.lower as u128*likelihoods[id].lower as u128).sum();
 let mut remaining=SCALE.checked_sub(lower_mass).ok_or("prior lower sum exceeds scale")?;
 let mut rows:Vec<(String,u64,u64,u64)>=priors.iter().map(|(id,p)|(id.clone(),p.lower,p.upper,likelihoods[id].lower)).collect();
 rows.sort_by(|a,b|a.3.cmp(&b.3).then_with(||a.0.cmp(&b.0)));
 for row in rows{
  let take=remaining.min(row.2-row.1);value+=take as u128*row.3 as u128;remaining-=take;if remaining==0{break;}
 }
 Ok((value/SCALE as u128) as u64)
}
fn max_surplus(risk:u64,priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>,safe:&BTreeMap<String,BTreeSet<String>>,action:&str)->i128{
 let mut rows:Vec<(String,u64,u64,i128)>=priors.iter().map(|(id,p)|{
  let l=&likelihoods[id];let coefficient=if safe[id].contains(action){
   -((risk as i128)*(l.lower as i128))
  }else{(((SCALE-risk) as i128)*(l.upper as i128))};
  (id.clone(),p.lower,p.upper,coefficient)
 }).collect();
 let mut value:i128=rows.iter().map(|x|x.1 as i128*x.3).sum();
 let mut remaining=SCALE-rows.iter().map(|x|x.1).sum::<u64>();
 rows.sort_by(|a,b|b.3.cmp(&a.3).then_with(||a.0.cmp(&b.0)));
 for row in rows{let take=remaining.min(row.2-row.1);value+=take as i128*row.3;remaining-=take;if remaining==0{break;}}
 value
}
fn robust_risk(priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>,safe:&BTreeMap<String,BTreeSet<String>>,action:&str)->u64{
 let mut low=0;let mut high=SCALE;
 while low<high{
  let mid=low+(high-low)/2;
  if max_surplus(mid,priors,likelihoods,safe,action)<=0{high=mid;}else{low=mid+1;}
 }
 low
}
fn nominal_risk(priors:&BTreeMap<String,Bounds>,likelihoods:&BTreeMap<String,Bounds>,safe:&BTreeMap<String,BTreeSet<String>>,action:&str)->u64{
 let mut numerator=0u128;let mut denominator=0u128;
 for (id,p) in priors{
  let mass=p.nominal as u128*likelihoods[id].nominal as u128;
  denominator+=mass;if !safe[id].contains(action){numerator+=mass;}
 }
 if denominator==0{return SCALE;}
 ((numerator*SCALE as u128+denominator-1)/denominator).min(SCALE as u128) as u64
}
fn conditional_cost(rows:&[JointRow],costs:&[u64])->Result<(u128,u128),String>{
 if rows.len()!=costs.len(){return Err("kernel cost vector mismatch".to_owned());}
 let lower_sum: u64=rows.iter().map(|x|x.probability.lower).sum();
 if lower_sum>SCALE{return Err("kernel lower mass infeasible".to_owned());}
 let nominal=rows.iter().zip(costs).try_fold(0u128,|sum,(row,cost)|{
  sum.checked_add(row.probability.nominal as u128*(*cost as u128)).ok_or_else(||"nominal kernel cost overflow".to_owned())
 })?;
 let mut worst=rows.iter().zip(costs).try_fold(0u128,|sum,(row,cost)|{
  sum.checked_add(row.probability.lower as u128*(*cost as u128)).ok_or_else(||"worst kernel cost overflow".to_owned())
 })?;
 let mut remaining=SCALE-lower_sum;
 let mut order:Vec<usize>=(0..rows.len()).collect();
 order.sort_by(|a,b|costs[*b].cmp(&costs[*a]).then_with(||a.cmp(b)));
 for index in order{
  let row=&rows[index];let room=row.probability.upper-row.probability.lower;
  let take=remaining.min(room);
  worst=worst.checked_add(take as u128*costs[index] as u128).ok_or("worst kernel cost overflow")?;
  remaining-=take;if remaining==0{break;}
 }
 if remaining>0{return Err("kernel upper masses cannot allocate unit probability".to_owned());}
 Ok((nominal,worst))
}
fn outer_worst_cost(priors:&BTreeMap<String,Bounds>,conditional:&BTreeMap<String,u128>)->Result<u128,String>{
 let lower_sum:u64=priors.values().map(|x|x.lower).sum();
 if lower_sum>SCALE{return Err("prior lower mass infeasible".to_owned());}
 let mut value:u128=priors.iter().try_fold(0u128,|sum,(id,p)|{
  sum.checked_add(p.lower as u128*conditional[id]).ok_or_else(||"outer cost overflow".to_owned())
 })?;
 let mut remaining=SCALE-lower_sum;
 let mut order:Vec<String>=priors.keys().cloned().collect();
 order.sort_by(|a,b|conditional[b].cmp(&conditional[a]).then_with(||a.cmp(b)));
 for id in order{
  let p=&priors[&id];let take=remaining.min(p.upper-p.lower);
  value=value.checked_add(take as u128*conditional[&id]).ok_or("outer cost overflow")?;
  remaining-=take;if remaining==0{break;}
 }
 if remaining>0{return Err("prior upper masses cannot allocate unit probability".to_owned());}
 Ok(value)
}
fn ceil_scale2(value:u128)->u64{
 let divisor=(SCALE as u128)*(SCALE as u128);
 ((value+divisor-1)/divisor).min(u64::MAX as u128) as u64
}
fn parse_tree(tree:&Value,options:&BTreeMap<String,OptionDef>,outcomes:&BTreeMap<String,BTreeSet<String>>,actions:&BTreeMap<String,Action>,max_depth:usize)->Result<PolicyTree,String>{
 allowed(tree,&["root_node_id","nodes"],"policy_tree")?;
 let root=string(tree,"root_node_id","policy_tree")?;
 let rows=get(tree,"nodes","policy_tree")?.as_array().ok_or("policy_tree.nodes must be array")?;
 if rows.is_empty()||rows.len()>MAX_NODES{return Err(format!("policy_tree.nodes must contain 1..={MAX_NODES} nodes"));}
 let mut nodes=BTreeMap::new();
 for (i,row) in rows.iter().enumerate(){
  let label=format!("policy_tree.nodes[{i}]");
  let id=string(row,"node_id",&label)?;let kind=string(row,"kind",&label)?;
  let node=if kind=="acquire"{
   allowed(row,&["node_id","kind","option_id","at","branches"],&label)?;
   let option=string(row,"option_id",&label)?;
   let opt=options.get(&option).ok_or_else(||format!("unknown option {option:?}"))?;
   let at=string(row,"at",&label)?;let _=time(&at,&label)?;
   let branch_object=object(get(row,"branches",&label)?,"branches")?;
   let branches=branch_object.iter().map(|(k,v)|Ok((k.clone(),v.as_str().ok_or("branch target must be string")?.to_owned()))).collect::<Result<BTreeMap<_,_>,String>>()?;
   let registered=outcomes.get(&opt.factor_id).ok_or_else(||format!("missing outcome partition for {:?}",opt.factor_id))?;
   let actual=branches.keys().cloned().collect::<BTreeSet<_>>();
   if actual!=*registered{return Err(format!("{label}.branches must cover every registered outcome exactly"));}
   Node{id:id.clone(),kind,option_id:Some(option),at:Some(at),branches,decision_kind:None,action_id:None,claimed:0}
  }else if kind=="stop"{
   allowed(row,&["node_id","kind","decision_kind","action_id","claimed_residual_risk_bps"],&label)?;
   let decision=string(row,"decision_kind",&label)?;
   if decision!="act"&&decision!="defer"{return Err("decision_kind must be act or defer".to_owned());}
   let action=string(row,"action_id",&label)?;
   if !actions.contains_key(&action){return Err(format!("unknown action {action:?}"));}
   let claimed=uint(row,"claimed_residual_risk_bps",&label)?;
   if claimed>SCALE{return Err("claimed risk must be <=10000".to_owned());}
   if decision=="defer"{
    let a=&actions[&action];
    if !a.read_only||a.irreversible||a.hard_gate||a.max_risk!=0{return Err("defer must use canonical read-only zero-risk action".to_owned());}
   }
   Node{id:id.clone(),kind,option_id:None,at:None,branches:BTreeMap::new(),decision_kind:Some(decision),action_id:Some(action),claimed}
  }else{return Err(format!("{label}.kind must be acquire or stop"));};
  if nodes.insert(id.clone(),node).is_some(){return Err(format!("duplicate node ID {id:?}"));}
 }
 if !nodes.contains_key(&root){return Err("policy tree root missing".to_owned());}
 let mut parents:BTreeMap<String,usize>=BTreeMap::new();
 for node in nodes.values(){for child in node.branches.values(){
  if !nodes.contains_key(child){return Err(format!("missing child node {child:?}"));}
  *parents.entry(child.clone()).or_default()+=1;
 }}
 if parents.values().any(|x|*x>1){return Err("policy graph contains shared child nodes; expected a tree".to_owned());}
 let mut visited=BTreeSet::new();let mut stack=vec![(root.clone(),0usize)];
 while let Some((id,depth))=stack.pop(){
  if depth>max_depth{return Err("policy tree exceeds max_depth".to_owned());}
  if !visited.insert(id.clone()){return Err("policy tree contains a cycle or repeated node".to_owned());}
  for child in nodes[&id].branches.values(){stack.push((child.clone(),depth+1));}
 }
 if visited.len()!=nodes.len(){return Err("policy tree contains unreachable nodes".to_owned());}
 Ok(PolicyTree{root,nodes})
}
fn leaf_risk(
 priors:&BTreeMap<String,Bounds>,safe:&BTreeMap<String,BTreeSet<String>>,
 kernels:&BTreeMap<String,Kernel>,membership:&BTreeMap<String,BTreeSet<usize>>,
 action:&str
)->Result<(u64,u64,u64,BTreeMap<String,Bounds>),String>{
 let mut likelihoods=BTreeMap::new();
 for (hypothesis,kernel) in kernels{
  let selected=membership.get(hypothesis).cloned().unwrap_or_default();
  likelihoods.insert(hypothesis.clone(),event_bounds(&kernel.rows,&selected)?);
 }
 let support=support_lower(priors,&likelihoods)?;
 let nominal=nominal_risk(priors,&likelihoods,safe,action);
 let robust=if support==0{SCALE}else{robust_risk(priors,&likelihoods,safe,action)};
 Ok((nominal,robust,support,likelihoods))
}
fn parse_bundle(raw:&str)->Result<(Value,Value),String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 Ok((get(&bundle,"candidate_view","bundle")?.clone(),get(&bundle,"oracle","bundle")?.clone()))
}
fn evaluate(raw:&str)->Result<Value,String>{
 let (candidate,oracle)=parse_bundle(raw)?;
 allowed(&candidate,&["scenario_id","decision_cutoff","budget_micros","max_depth","actions","hypothesis_bounds","options","factors","policy_tree"],"candidate_view")?;
 allowed(&oracle,&["actual_hypothesis_id","decision_cutoff","budget_micros","actions","hypothesis_bounds","options","factors","hypothesis_safety","calibrations","factorization_certificates","world_kernels","cost_options","billing_currency"],"oracle")?;
 let scenario=string(&candidate,"scenario_id","candidate_view")?;
 let cutoff_raw=string(&candidate,"decision_cutoff","candidate_view")?;let cutoff=time(&cutoff_raw,"candidate_view.decision_cutoff")?;
 if cutoff_raw!=string(&oracle,"decision_cutoff","oracle")?{return Err("candidate cutoff differs from evaluator cutoff".to_owned());}
 let budget=uint(&candidate,"budget_micros","candidate_view")?;
 if budget!=uint(&oracle,"budget_micros","oracle")?{return Err("candidate budget differs from evaluator budget".to_owned());}
 let currency=string(&oracle,"billing_currency","oracle")?;
 if currency.len()!=3||!currency.chars().all(|x|x.is_ascii_uppercase()){return Err("billing_currency must be a three-letter uppercase code".to_owned());}
 for field in ["actions","hypothesis_bounds","options","factors"]{
  if candidate[field]!=oracle[field]{return Err(format!("candidate {field} differs from evaluator-authoritative copy"));}
 }
 let priors=parse_priors(&candidate,&oracle)?;
 let ids=priors.keys().cloned().collect::<BTreeSet<_>>();
 let actions=parse_actions(&candidate,&oracle)?;
 let safety=parse_safety(&oracle,&ids,&actions)?;
 let defer=canonical_defer(&actions,&safety)?;
 let targets:Vec<String>=actions.iter().filter(|(_,x)|x.irreversible).map(|(id,_)|id.clone()).collect();
 if targets.len()!=1{return Err("v0.1 requires exactly one evaluator-designated irreversible target action".to_owned());}
 let target_action=targets[0].clone();
 let options=parse_options(&candidate,&oracle,&cutoff)?;
 let outcomes=factor_outcomes(&candidate,&options)?;
 let actual=string(&oracle,"actual_hypothesis_id","oracle")?;
 if !ids.contains(&actual){return Err("actual_hypothesis_id is outside candidate hypothesis set".to_owned());}
 let kernels=parse_kernels(&candidate,&oracle,&options,&cutoff_raw)?;
 if kernels.keys().cloned().collect::<BTreeSet<_>>()!=ids{return Err("joint kernels must cover the exact hypothesis set".to_owned());}
 let calibrations=get(&oracle,"calibrations","oracle")?.as_array().ok_or("calibrations must be an array")?;
 for factor in get(&candidate,"factors","candidate_view")?.as_array().ok_or("factors must be array")?{
  let fid=string(factor,"factor_id","factor")?;let cid=string(factor,"calibration_ref","factor")?;
  let calibration=calibrations.iter().find(|x|x["calibration_id"].as_str()==Some(cid.as_str())).ok_or_else(||format!("missing calibration {cid:?}"))?;
  let authority=string(calibration,"authority","calibration")?;
  let from=time(&string(calibration,"valid_from","calibration")?,"calibration.valid_from")?;
  let until=time(&string(calibration,"valid_until","calibration")?,"calibration.valid_until")?;
  let revoked=match calibration.get("revoked_at"){None|Some(Value::Null)=>None,Some(Value::String(x))=>Some(time(x,"calibration.revoked_at")?),_=>return Err("calibration revoked_at invalid".to_owned())};
  let factor_ids=get(calibration,"factor_ids","calibration")?.as_array().ok_or("calibration.factor_ids must be array")?;
  if authority.trim().is_empty()||!factor_ids.iter().any(|x|x.as_str()==Some(fid.as_str()))||from>cutoff||until<cutoff||revoked.is_some_and(|x|x<=cutoff){
   return Err(format!("factor {fid:?} calibration is not independently valid at cutoff"));
  }
 }
 let max_depth_raw=uint(&candidate,"max_depth","candidate_view")?;
 if max_depth_raw==0||max_depth_raw>5{return Err("max_depth must be 1..=5".to_owned());}
 let max_depth=max_depth_raw as usize;
 let tree=parse_tree(get(&candidate,"policy_tree","candidate_view")?,&options,&outcomes,&actions,max_depth)?;
 for node in tree.nodes.values(){
  if node.kind=="stop"&&node.decision_kind.as_deref()==Some("defer")&&node.action_id.as_deref()!=Some(defer.as_str()){
   return Err("all defer leaves must use the canonical read-only action".to_owned());
  }
  if node.kind=="stop"&&node.decision_kind.as_deref()==Some("act")&&node.action_id.as_deref()==Some(defer.as_str()){
   return Err("canonical defer action cannot be an act leaf".to_owned());
  }
 }
 let mut atoms_by_leaf:BTreeMap<String,BTreeMap<String,BTreeSet<usize>>>=BTreeMap::new();
 let mut leaf_paths:BTreeMap<String,LeafPath>=BTreeMap::new();
 let mut row_costs:BTreeMap<String,Vec<u64>>=BTreeMap::new();
 let mut max_path=0u64;
 let mut actual_row_leafs:BTreeMap<usize,String>=BTreeMap::new();
 for (hypothesis,kernel) in &kernels{
  let mut costs=Vec::new();
  for (row_index,row) in kernel.rows.iter().enumerate(){
   let mut node_id=tree.root.clone();let mut seen_options=BTreeSet::new();let mut path_options=Vec::new();
   let mut path_cost=0u64;let mut previous:Option<DateTime<FixedOffset>>=None;let mut depth=0usize;
   loop{
    depth+=1;if depth>max_depth+1{return Err("tree traversal depth exceeded".to_owned());}
    let node=tree.nodes.get(&node_id).ok_or_else(||format!("missing node {node_id:?}"))?;
    if node.kind=="stop"{
     let decision=node.decision_kind.clone().ok_or("stop node missing decision kind")?;
     let action=node.action_id.clone().ok_or("stop node missing action ID")?;
     if let Some(existing)=leaf_paths.get(&node_id){
      if existing.path_cost!=path_cost||existing.action_id!=action||existing.decision_kind!=decision||existing.option_ids!=path_options{
       return Err(format!("leaf {node_id:?} combines inconsistent acquisition histories"));
      }
     }else{
      leaf_paths.insert(node_id.clone(),LeafPath{leaf_id:node_id.clone(),decision_kind:decision,action_id:action,claimed:node.claimed,option_ids:path_options.clone(),world_row_count:0,path_cost});
     }
     atoms_by_leaf.entry(node_id.clone()).or_default().entry(hypothesis.clone()).or_default().insert(row_index);
     leaf_paths.get_mut(&node_id).ok_or("leaf insertion failed")?.world_row_count+=1;
     costs.push(path_cost);
     if row.probability.upper>0{max_path=max_path.max(path_cost);}
     if hypothesis==&actual&&row.probability.upper>0{actual_row_leafs.insert(row_index,node_id.clone());}
     break;
    }
    let option_id=node.option_id.as_ref().ok_or("acquire node lacks option")?;
    if !seen_options.insert(option_id.clone()){return Err(format!("policy repeats option {option_id:?}"));}
    let at_raw=node.at.as_ref().ok_or("acquire node lacks timestamp")?;
    let at=time(at_raw,"policy node at")?;
    let option=options.get(option_id).ok_or_else(||format!("policy uses unknown option {option_id:?}"))?;
    if at<option.available_at||at>cutoff{return Err(format!("option {option_id:?} is unavailable at submitted time"));}
    if previous.as_ref().is_some_and(|prev|&at<prev){return Err("policy acquisition timestamps regress".to_owned());}
    previous=Some(at);
    let observed=row.outcomes.get(option_id).ok_or_else(||format!("kernel row missing option {option_id:?}"))?;
    let child=node.branches.get(observed).ok_or_else(||format!("policy branch missing outcome {observed:?}"))?;
    path_cost=path_cost.checked_add(option.cost_micros).ok_or("path cost overflow")?;
    path_options.push(option_id.clone());node_id=child.clone();
   }
  }
  row_costs.insert(hypothesis.clone(),costs);
 }
 let mut conditional_nominal=BTreeMap::new();let mut conditional_worst=BTreeMap::new();
 let mut cost_details=Vec::new();
 for (hypothesis,kernel) in &kernels{
  let costs=row_costs.get(hypothesis).ok_or("missing per-kernel cost vector")?;
  let (nom,worst)=conditional_cost(&kernel.rows,costs)?;
  conditional_nominal.insert(hypothesis.clone(),nom);conditional_worst.insert(hypothesis.clone(),worst);
  cost_details.push(json!({"hypothesis_id":hypothesis,"nominal_conditional_cost_numerator":nom,"worst_case_conditional_cost_numerator":worst}));
 }
 let nominal_outer=priors.iter().try_fold(0u128,|sum,(id,p)|{
  sum.checked_add(p.nominal as u128*conditional_nominal[id]).ok_or_else(||"nominal total cost overflow".to_owned())
 })?;
 let worst_outer=outer_worst_cost(&priors,&conditional_worst)?;
 let nominal_cost=ceil_scale2(nominal_outer);let worst_cost=ceil_scale2(worst_outer);
 let full_cost=options.values().try_fold(0u64,|sum,o|sum.checked_add(o.cost_micros).ok_or("full revalidation cost overflow"))?;
 let mut leaf_reports=Vec::new();let mut risk_violations=0usize;let mut budget_violations=0usize;
 let mut coverage_numerator=0u128;let mut unnecessary_numerator=0u128;let mut unsafe_numerator=0u128;
 let mut actual_unsafe_nominal=0u64;let mut actual_unsafe_upper=0u64;
 for (leaf_id,path) in &leaf_paths{
  let membership=atoms_by_leaf.get(leaf_id).ok_or_else(||format!("leaf {leaf_id:?} has no joint rows"))?;
  let (nominal,robust,support,likelihoods)=leaf_risk(&priors,&safety,&kernels,membership,&path.action_id)?;
  let action=&actions[&path.action_id];
  let justified=support>0&&robust<=action.max_risk&&(!action.hard_gate||robust==0);
  let within=path.path_cost<=budget;
  if path.decision_kind=="act"&&!justified{risk_violations+=1;}
  if !within{budget_violations+=1;}
  let mut leaf_nominal_numerator=0u128;let mut leaf_unsafe_numerator=0u128;
  for (hypothesis,prior) in &priors{
   let probability=likelihoods[hypothesis].nominal as u128;
   leaf_nominal_numerator+=prior.nominal as u128*probability;
   if !safety[hypothesis].contains(&path.action_id){leaf_unsafe_numerator+=prior.nominal as u128*probability;}
  }
  let leaf_nominal_bps=(leaf_nominal_numerator/SCALE as u128) as u64;
  if path.decision_kind=="act"&&justified&&within{coverage_numerator+=leaf_nominal_numerator;}
  if path.decision_kind=="act"{unsafe_numerator+=leaf_unsafe_numerator;}
  let target_risk=if path.decision_kind=="defer"{
   let (tn,tr,ts,_)=leaf_risk(&priors,&safety,&kernels,membership,&target_action)?;
   if ts>0&&tr<=actions[&target_action].max_risk&&within{unnecessary_numerator+=leaf_nominal_numerator;}
   Some((tn,tr,ts))
  }else{None};
  let actual_likelihood=likelihoods.get(&actual).ok_or("actual hypothesis likelihood missing")?;
  let actual_possible=actual_likelihood.upper>0;
  leaf_reports.push(json!({
   "leaf_node_id":leaf_id,"decision_kind":path.decision_kind,"action_id":path.action_id,
   "acquisition_options":path.option_ids,"joint_row_count":path.world_row_count,
   "nominal_path_probability_bps":leaf_nominal_bps,
   "nominal_posterior_risk_bps":nominal,"worst_case_posterior_risk_bps":robust,
   "evidence_support_lower_bps":support,"actual_hypothesis_path_possible":actual_possible,
   "action_justified":justified,"path_cost_micros":path.path_cost,"within_budget":within,
   "target_action_counterfactual":target_risk.map(|(n,r,s)|json!({"nominal_risk_bps":n,"worst_case_risk_bps":r,"support_lower_bps":s}))
  }));
 }
 let actual_kernel=kernels.get(&actual).ok_or("actual joint kernel missing")?;
 let actual_unsafe_rows:BTreeSet<usize>=actual_row_leafs.iter().filter_map(|(row_index,leaf_id)|{
  let path=leaf_paths.get(leaf_id)?;
  if path.decision_kind=="act"&&!safety[&actual].contains(&path.action_id){Some(*row_index)}else{None}
 }).collect();
 let actual_unsafe_interval=event_bounds(&actual_kernel.rows,&actual_unsafe_rows)?;
 actual_unsafe_nominal=actual_unsafe_interval.nominal;
 actual_unsafe_upper=actual_unsafe_interval.upper;
 let coverage=(coverage_numerator/SCALE as u128).min(SCALE as u128) as u64;
 let unnecessary=(unnecessary_numerator/SCALE as u128).min(SCALE as u128) as u64;
 let unsafe_bps=(unsafe_numerator/SCALE as u128).min(SCALE as u128) as u64;
 let policy_feasible=risk_violations==0&&budget_violations==0;
 let report=json!({
  "scenario_id":scenario,"adjudication_status":"stochastic_joint_kernel_adaptive_policy_scoring",
  "billing_currency":currency,"world_count":kernels.len(),"joint_row_count":kernels.values().map(|x|x.rows.len()).sum::<usize>(),
  "option_count":options.len(),"leaf_count":leaf_paths.len(),"max_depth":max_depth,
  "submitted_policy":{
   "policy_feasible":policy_feasible,"nominal_justified_action_coverage_bps":coverage,
   "nominal_unsafe_action_mass_bps":unsafe_bps,"unnecessary_abstention_bps":unnecessary,
   "risk_violation_leaf_count":risk_violations,"over_budget_leaf_count":budget_violations,
   "actual_hypothesis_unsafe_probability_nominal_bps":actual_unsafe_nominal,
   "actual_hypothesis_unsafe_probability_upper_bps":actual_unsafe_upper,
   "worst_case_path_cost_micros":max_path,"nominal_expected_cost_micros":nominal_cost,
   "worst_case_expected_cost_micros":worst_cost,"budget_micros":budget,
   "leaf_reports":leaf_reports
  },
  "expected_cost":{
   "assumption":"Per-hypothesis joint-kernel credal sets are independent across hypotheses; the evaluator then optimizes the outer prior credal set. Cross-hypothesis shared latent ambiguity requires a coupled joint set.",
   "nominal_expected_cost_micros":nominal_cost,"worst_case_expected_cost_micros":worst_cost,
   "full_revalidation_cost_micros":full_cost,
   "nominal_savings_vs_full_revalidation_micros":full_cost.saturating_sub(nominal_cost),
   "conditional_world_cost_bounds":cost_details
  },
  "claim_ceiling":CLAIM_CEILING
 });
 Ok(report)
}
pub fn score_stochastic_adaptive_policy_json(raw:&str)->Result<Value,String>{evaluate(raw)}
fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{
  eprintln!("Usage: stochastic-adaptive-policy-scorer <candidate-and-oracle-bundle.json>");
  process::exit(if args.len()==1{0}else{2});
 }
 let raw=match fs::read_to_string(&args[0]){Ok(x)=>x,Err(e)=>{eprintln!("ERROR D6.17.2.2.1-IO-001 {}: {e}",args[0]);process::exit(2)}};
 match evaluate(&raw){
  Ok(report)=>match serde_json::to_string_pretty(&report){Ok(output)=>println!("{output}"),Err(e)=>{eprintln!("ERROR D6.17.2.2.1-JSON-001: {e}");process::exit(2)}},
  Err(e)=>{eprintln!("ERROR D6.17.2.2.1-EVALUATION-001: {e}");process::exit(1)}
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 const DETERMINISTIC:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-deterministic-v0.1.json");
 const DETERMINISTIC_EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-deterministic-v0.1.expected.json");
 const COMMON_CAUSE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-common-cause-v0.1.json");
 const COMMON_CAUSE_EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-stochastic-policy-common-cause-v0.1.expected.json");
 #[test]fn deterministic_limit_matches_d6_17_2_1_known_answer(){
  let actual=evaluate(DETERMINISTIC).expect("deterministic limit validates");
  let expected:Value=serde_json::from_str(DETERMINISTIC_EXPECTED).unwrap();
  assert_eq!(actual,expected);
  assert_eq!(actual["submitted_policy"]["nominal_justified_action_coverage_bps"],6000);
  assert_eq!(actual["expected_cost"]["nominal_expected_cost_micros"],1300);
  assert_eq!(actual["expected_cost"]["worst_case_expected_cost_micros"],1420);
 }
 #[test]fn common_cause_joint_kernel_does_not_multiply_marginals(){
  let r=evaluate(COMMON_CAUSE).expect("common cause fixture validates");
  let expected:Value=serde_json::from_str(COMMON_CAUSE_EXPECTED).unwrap();
  assert_eq!(r,expected);
  assert_eq!(r["submitted_policy"]["nominal_justified_action_coverage_bps"],4500);
 }
 #[test]fn candidate_cannot_override_joint_kernels(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["world_kernels"]=json!([]);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));
 }
 #[test]fn candidate_cannot_override_costs(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["cost_options"]=json!([]);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("unknown field"));
 }
 #[test]fn candidate_cannot_change_prior_bounds(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["hypothesis_bounds"][0]["upper_probability_bps"]=json!(9999);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("differs from evaluator-authoritative"));
 }
 #[test]fn missing_policy_branch_is_rejected(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["policy_tree"]["nodes"][0]["branches"].as_object_mut().unwrap().remove("fail");
  assert!(evaluate(&b.to_string()).unwrap_err().contains("cover every registered outcome"));
 }
 #[test]fn repeated_acquisition_on_path_is_rejected(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["policy_tree"]["nodes"][1]["option_id"]=json!("gate-a");
  assert!(evaluate(&b.to_string()).unwrap_err().contains("policy repeats option"));
 }
 #[test]fn invalid_joint_marginal_is_rejected_before_scoring(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["factors"][0]["likelihoods"][0]["outcomes"]["pass"]["upper_likelihood_bps"]=json!(1);
  assert!(evaluate(&b.to_string()).unwrap_err().contains("does not conservatively contain"));
 }
 #[test]fn policy_that_exceeds_budget_is_not_feasible(){
  let mut b:Value=serde_json::from_str(DETERMINISTIC).unwrap();
  b["candidate_view"]["budget_micros"]=json!(100);
  b["oracle"]["budget_micros"]=json!(100);
  let r=evaluate(&b.to_string()).expect("over-budget policy still reports");
  assert_eq!(r["submitted_policy"]["policy_feasible"],false);
  assert!(r["submitted_policy"]["over_budget_leaf_count"].as_u64().unwrap()>0);
 }
}
