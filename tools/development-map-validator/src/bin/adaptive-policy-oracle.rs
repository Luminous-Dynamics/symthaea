//! D6.17.2.1 exact small-tree adaptive evidence-policy oracle.
//!
//! The evaluator validates a submitted finite policy tree, reuses D6.17 robust
//! risk at every reachable leaf, and exhaustively enumerates all bounded-depth
//! deterministic policies on small finite observation worlds. Exactness is
//! conditional on the registered finite world/outcome model and search cap.

#[allow(dead_code)]
#[path = "noisy-channel-risk.rs"]
mod noisy_channel;

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCALE:u64=10_000;
const MAX_WORLDS:usize=8;
const MAX_OPTIONS:usize=6;
const MAX_DEPTH:usize=5;
const MAX_POLICIES:usize=50_000;
const CLAIM_CEILING:&str="Exact exhaustive policy oracle over the registered finite deterministic observation-world fixture and search limits. Risk at each leaf is delegated to D6.17.1. It does not establish optimality outside the finite model, stochastic transition distributions, arbitrary noisy channels beyond the reused risk evaluator, or measured production cost.";

#[derive(Clone,Debug)]
struct Bounds{lower:u64,upper:u64,nominal:u64}
#[derive(Clone,Debug)]
struct World{ safe_actions:BTreeSet<String>, observations:BTreeMap<String,String> }
#[derive(Clone,Debug)]
struct OptionDef{factor_id:String,available_at:DateTime<FixedOffset>,available_at_raw:String,cost_p95:u64}
#[derive(Clone,Debug)]
struct Action{irreversible:bool,hard_gate:bool,max_risk:u64}
#[derive(Clone,Debug)]
struct Step{option_id:String,outcome:String,at:String}
#[derive(Clone,Debug)]
struct Node{id:String,kind:String,option_id:Option<String>,at:Option<String>,branches:BTreeMap<String,String>,decision_kind:Option<String>,action_id:Option<String>,claimed:u64}
#[derive(Clone,Debug)]
struct PolicyTree{root:String,nodes:BTreeMap<String,Node>}
#[derive(Clone,Debug)]
struct LeafPath{leaf_id:String,steps:Vec<Step>,decision_kind:String,action_id:String,claimed:u64,world_ids:Vec<String>,cost:u64}
#[derive(Clone,Debug)]
struct SubPolicy{root:String,nodes:Vec<Value>}
#[derive(Clone,Debug)]
struct PolicyMetrics{report:Value,coverage:u64,worst_expected_cost:u64,nominal_expected_cost:u64,max_path_cost:u64,feasible:bool}

fn object<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{v.as_object().ok_or_else(||format!("{label} must be an object"))}
fn allowed(v:&Value,keys:&[&str],label:&str)->Result<(),String>{for k in object(v,label)?.keys(){if !keys.contains(&k.as_str()){return Err(format!("{label} contains unknown field {k:?}"));}}Ok(())}
fn val<'a>(v:&'a Value,k:&str,l:&str)->Result<&'a Value,String>{v.get(k).ok_or_else(||format!("{l}.{k} is required"))}
fn s(v:&Value,k:&str,l:&str)->Result<String,String>{val(v,k,l)?.as_str().filter(|x|!x.trim().is_empty()).map(str::to_owned).ok_or_else(||format!("{l}.{k} must be a non-empty string"))}
fn u(v:&Value,k:&str,l:&str)->Result<u64,String>{val(v,k,l)?.as_u64().ok_or_else(||format!("{l}.{k} must be a non-negative integer"))}
fn parse_time(raw:&str,label:&str)->Result<DateTime<FixedOffset>,String>{DateTime::parse_from_rfc3339(raw).map_err(|e|format!("{label} must be RFC 3339: {e}"))}
fn p95(xs:&[u64])->Result<u64,String>{if xs.is_empty(){return Err("cost sample list is empty; unknown cost is not zero".to_owned());}let mut a=xs.to_vec();a.sort_unstable();let rank=(a.len()*95+99)/100;Ok(a[rank.saturating_sub(1)])}

fn parse_bounds(v:&Value)->Result<BTreeMap<String,Bounds>,String>{
 let rows=v.as_array().ok_or("hypothesis_bounds must be an array")?;if rows.is_empty()||rows.len()>MAX_WORLDS{return Err("hypothesis_bounds must contain 1..=8 entries".to_owned());}
 let mut out=BTreeMap::new();let mut lo=0u64;let mut hi=0u64;let mut nom=0u64;
 for (i,r) in rows.iter().enumerate(){let l=format!("hypothesis_bounds[{i}]");allowed(r,&["hypothesis_id","lower_probability_bps","upper_probability_bps","nominal_probability_bps"],&l)?;
  let id=s(r,"hypothesis_id",&l)?;let b=Bounds{lower:u(r,"lower_probability_bps",&l)?,upper:u(r,"upper_probability_bps",&l)?,nominal:u(r,"nominal_probability_bps",&l)?};
  if b.lower>b.nominal||b.nominal>b.upper||b.upper>SCALE{return Err(format!("{l}: require 0 <= lower <= nominal <= upper <= 10000"));}
  lo+=b.lower;hi+=b.upper;nom+=b.nominal;if out.insert(id.clone(),b).is_some(){return Err(format!("duplicate hypothesis {id:?}"));}
 }
 if lo>SCALE||hi<SCALE||nom!=SCALE{return Err(format!("infeasible prior bounds: lower={lo}, upper={hi}, nominal={nom}"));}
 Ok(out)
}
fn parse_options(candidate:&Value,oracle:&Value)->Result<BTreeMap<String,OptionDef>,String>{
 let cv=val(candidate,"options","candidate_view")?.as_array().ok_or("candidate options must be an array")?;
 let ov=val(oracle,"cost_options","oracle")?.as_array().ok_or("oracle cost_options must be an array")?;
 if cv.len()!=ov.len()||cv.is_empty()||cv.len()>MAX_OPTIONS{return Err("candidate options and oracle cost_options must have the same 1..=6 entries".to_owned());}
 let mut oracle_ids=BTreeSet::new();
 for row in ov{
  let id=s(row,"option_id","oracle cost option")?;
  if !oracle_ids.insert(id){return Err("duplicate oracle cost option ID".to_owned());}
 }
 let candidate_ids:BTreeSet<String>=cv.iter().map(|row|s(row,"option_id","candidate option")).collect::<Result<_,_>>()?;
 if candidate_ids!=oracle_ids{return Err("candidate and oracle cost option IDs differ".to_owned());}
 let mut out=BTreeMap::new();
 for (i,co) in cv.iter().enumerate(){
  let l=format!("candidate options[{i}]");allowed(co,&["option_id","available_at","factor_id"],&l)?;
  let id=s(co,"option_id",&l)?;let at=s(co,"available_at",&l)?;let parsed=parse_time(&at,&l)?;let factor=s(co,"factor_id",&l)?;
  let oo=ov.iter().find(|x|x["option_id"].as_str()==Some(id.as_str())).ok_or_else(||format!("oracle cost option missing {id:?}"))?;
  allowed(oo,&["option_id","available_at","cost_samples_micros"],"oracle cost option")?;
  if s(oo,"available_at","oracle cost option")?!=at{return Err(format!("option {id:?} availability mismatch"));}
  let samples=val(oo,"cost_samples_micros","oracle cost option")?;
  allowed(samples,&["cold"],"cost_samples_micros")?;
  let raw=val(samples,"cold","cost_samples_micros")?.as_array().ok_or("cold cost samples must be an array")?;
  if raw.len()>1000{return Err("cost samples exceed 1000 entries".to_owned());}
  let costs=raw.iter().map(|x|x.as_u64().filter(|n|*n>0).ok_or("cost sample must be positive")).collect::<Result<Vec<_>,_>>()?;
  let cost=p95(&costs)?;
  if out.insert(id.clone(),OptionDef{factor_id:factor,available_at:parsed,available_at_raw:at,cost_p95:cost}).is_some(){return Err(format!("duplicate option {id:?}"));}
 }
 Ok(out)
}
fn parse_worlds(oracle:&Value,ids:&BTreeSet<String>,option_ids:&BTreeSet<String>,actions:&BTreeMap<String,Action>)->Result<BTreeMap<String,World>,String>{
 let values=val(oracle,"worlds","oracle")?.as_array().ok_or("oracle.worlds must be an array")?;
 if values.len()!=ids.len()||values.is_empty()||values.len()>MAX_WORLDS{return Err("oracle worlds must cover all hypotheses exactly (1..=8)".to_owned());}
 let mut out=BTreeMap::new();
 for (i,w) in values.iter().enumerate(){let l=format!("oracle.worlds[{i}]");allowed(w,&["world_id","safe_actions","observations"],&l)?;
  let id=s(w,"world_id",&l)?;if !ids.contains(&id){return Err(format!("unknown world id {id:?}"));}
  let safe=w["safe_actions"].as_array().ok_or("world.safe_actions must be an array")?;
  let mut safe_set=BTreeSet::new();for a in safe{let a=a.as_str().ok_or("safe action must be string")?;if !actions.contains_key(a){return Err(format!("world references unknown action {a:?}"));}safe_set.insert(a.to_owned());}
  let obs=object(val(w,"observations",&l)?,&format!("{l}.observations"))?;
  let observed_options=obs.keys().cloned().collect::<BTreeSet<_>>();
  if observed_options.len()!=option_ids.len()||!observed_options.is_subset(option_ids){return Err(format!("world {id:?} observation option set mismatch"));}
  let observations=obs.iter().map(|(k,v)|Ok((k.clone(),v.as_str().ok_or("observation must be string")?.to_owned()))).collect::<Result<BTreeMap<_,_>,String>>()?;
  if out.insert(id.clone(),World{safe_actions:safe_set,observations}).is_some(){return Err(format!("duplicate world {id:?}"));}
 }
 Ok(out)
}
fn parse_actions(v:&Value)->Result<BTreeMap<String,Action>,String>{
 let items=val(v,"actions","candidate_view")?.as_array().ok_or("actions must be array")?;
 if items.is_empty()||items.len()>16{return Err("actions must contain 1..=16 entries".to_owned());}
 let mut out=BTreeMap::new();
 for (i,a) in items.iter().enumerate(){let l=format!("actions[{i}]");allowed(a,&["action_id","irreversible","hard_gate","max_residual_risk_bps"],&l)?;
  let id=s(a,"action_id",&l)?;let irreversible=val(a,"irreversible",&l)?.as_bool().ok_or("irreversible must be boolean")?;let hard=val(a,"hard_gate",&l)?.as_bool().ok_or("hard_gate must be boolean")?;let risk=u(a,"max_residual_risk_bps",&l)?;
  if risk>SCALE||(hard&&risk!=0)||(irreversible&&!hard){return Err(format!("invalid action policy {id:?}"));}
  if out.insert(id,Action{irreversible,hard_gate:hard,max_risk:risk}).is_some(){return Err("duplicate action".to_owned());}
 }
 Ok(out)
}
fn parse_nodes(tree:&Value,options:&BTreeMap<String,OptionDef>,factor_outcomes:&BTreeMap<String,BTreeSet<String>>,actions:&BTreeMap<String,Action>)->Result<PolicyTree,String>{
 allowed(tree,&["root_node_id","nodes"],"policy_tree")?;
 let root=s(tree,"root_node_id","policy_tree")?;
 let vals=val(tree,"nodes","policy_tree")?.as_array().ok_or("policy_tree.nodes must be an array")?;
 if vals.is_empty()||vals.len()>64{return Err("policy_tree.nodes must contain 1..=64 entries".to_owned());}
 let mut nodes=BTreeMap::new();
 for (i,v) in vals.iter().enumerate(){let l=format!("policy_tree.nodes[{i}]");let id=s(v,"node_id",&l)?;let kind=s(v,"kind",&l)?;
  let node=if kind=="acquire"{
   allowed(v,&["node_id","kind","option_id","at","branches"],&l)?;
   let option=s(v,"option_id",&l)?;let opt=options.get(&option).ok_or_else(||format!("unknown option {option:?}"))?;
   let at=s(v,"at",&l)?;let _=parse_time(&at,&l)?;
   let raw=object(val(v,"branches",&l)?,&format!("{l}.branches"))?;
   let branches=raw.iter().map(|(k,x)|Ok((k.clone(),x.as_str().ok_or("branch target must be string")?.to_owned()))).collect::<Result<BTreeMap<_,_>,String>>()?;
   let factor=factor_outcomes.get(&opt.factor_id).ok_or_else(||format!("missing factor outcome set for {:?}",opt.factor_id))?;
   let branch_outcomes=branches.keys().cloned().collect::<BTreeSet<_>>();
   if branch_outcomes.len()!=factor.len()||!branch_outcomes.is_subset(factor){return Err(format!("{l}.branches must cover every factor outcome exactly"));}
   Node{id:id.clone(),kind:kind.clone(),option_id:Some(option),at:Some(at),branches,decision_kind:None,action_id:None,claimed:0}
  }else if kind=="stop"{
   allowed(v,&["node_id","kind","decision_kind","action_id","claimed_residual_risk_bps"],&l)?;
   let decision=s(v,"decision_kind",&l)?;if decision!="act"&&decision!="defer"{return Err("decision_kind must be act or defer".to_owned());}
   let action=s(v,"action_id",&l)?;if !actions.contains_key(&action){return Err(format!("unknown action {action:?}"));}
   let claimed=u(v,"claimed_residual_risk_bps",&l)?;if claimed>SCALE{return Err("claimed risk exceeds 10000".to_owned());}
   if decision=="defer"{
    let policy=&actions[&action];
    if policy.irreversible||policy.hard_gate||policy.max_risk!=0{
     return Err("defer must use a non-irreversible, non-hard-gated zero-risk action".to_owned());
    }
   }
   Node{id:id.clone(),kind:kind.clone(),option_id:None,at:None,branches:BTreeMap::new(),decision_kind:Some(decision),action_id:Some(action),claimed}
  }else{return Err(format!("{l}.kind must be acquire or stop"));};
  if nodes.insert(id.clone(),node).is_some(){return Err(format!("duplicate node id {id:?}"));}
 }
 if !nodes.contains_key(&root){return Err("root node does not exist".to_owned());}
 let mut parent_count:BTreeMap<String,usize>=BTreeMap::new();
 for n in nodes.values(){for child in n.branches.values(){if !nodes.contains_key(child){return Err(format!("node {:?} references missing child {child:?}",n.id));}*parent_count.entry(child.clone()).or_insert(0)+=1;}}
 if parent_count.values().any(|x|*x>1){return Err("policy must be a tree; shared child nodes are not allowed".to_owned());}
 let mut visiting=BTreeSet::new();let mut visited=BTreeSet::new();
 fn dfs(id:&str,nodes:&BTreeMap<String,Node>,visiting:&mut BTreeSet<String>,visited:&mut BTreeSet<String>,depth:usize)->Result<(),String>{
  if depth>MAX_DEPTH+1{return Err("policy exceeds maximum depth 5".to_owned());}
  if visiting.contains(id){return Err("policy tree contains a cycle".to_owned());}
  if !visited.insert(id.to_owned()){return Ok(());}
  visiting.insert(id.to_owned());let n=&nodes[id];for child in n.branches.values(){dfs(child,nodes,visiting,visited,depth+1)?;}visiting.remove(id);Ok(())
 }
 dfs(&root,&nodes,&mut visiting,&mut visited,0)?;
 if visited.len()!=nodes.len(){return Err("policy tree contains unreachable nodes".to_owned());}
 Ok(PolicyTree{root,nodes})
}

fn factor_outcomes(candidate:&Value)->Result<BTreeMap<String,BTreeSet<String>>,String>{
 let mut out=BTreeMap::new();for f in val(candidate,"factors","candidate_view")?.as_array().ok_or("factors must be array")?{
  let id=s(f,"factor_id","factor")?;let rows=val(f,"likelihoods","factor")?.as_array().ok_or("likelihoods must be array")?;
  let first=rows.first().ok_or("factor has no likelihood rows")?;let names=object(val(first,"outcomes","likelihood")?,"outcomes")?.keys().cloned().collect();
  for row in rows{let names2=object(val(row,"outcomes","likelihood")?,"outcomes")?.keys().cloned().collect::<BTreeSet<_>>();if names2!=names{return Err(format!("factor {id:?} has inconsistent outcomes"));}}
  out.insert(id,names);
 }Ok(out)
}
fn parse_world_bounds(v:&Value)->Result<BTreeMap<String,Bounds>,String>{parse_bounds(val(v,"hypothesis_bounds","candidate_view")?)}
fn risk_for_path(candidate:&Value,oracle:&Value,steps:&[Step],representative:&str,action:&str,claimed:u64)->Result<Value,String>{
 let trace=json!({"trace_id":format!("risk-{}",steps.iter().map(|s|s.option_id.as_str()).collect::<Vec<_>>().join("-")),"acquisitions":steps.iter().map(|x|json!({"option_id":x.option_id,"observed_value":x.outcome,"at":x.at})).collect::<Vec<_>>(),"decision":{"kind":"act","action_id":action,"claimed_residual_risk_bps":claimed}});
 let cv=json!({"scenario_id":candidate["scenario_id"],"decision_cutoff":candidate["decision_cutoff"],"actions":candidate["actions"],"hypothesis_bounds":candidate["hypothesis_bounds"],"options":candidate["options"],"factors":candidate["factors"],"traces":[trace]});
 let ov=json!({"actual_hypothesis_id":representative,"decision_cutoff":oracle["decision_cutoff"],"actions":oracle["actions"],"hypothesis_bounds":oracle["hypothesis_bounds"],"options":oracle["options"],"factors":oracle["factors"],"hypothesis_safety":oracle["hypothesis_safety"],"calibrations":oracle["calibrations"],"factorization_certificates":oracle["factorization_certificates"]});
 let report=noisy_channel::evaluate_noisy_channel_parts_json(&cv.to_string(),&ov.to_string())?;
 Ok(report["traces"][0].clone())
}
fn canonical_defer_action(
 actions:&BTreeMap<String,Action>,
 worlds:&BTreeMap<String,World>
)->Result<String,String>{
 let matches:Vec<String>=actions.iter().filter(|(id,action)|{
  !action.irreversible&&!action.hard_gate&&action.max_risk==0
   &&worlds.values().all(|world|world.safe_actions.contains(*id))
 }).map(|(id,_)|id.clone()).collect();
 if matches.len()!=1{
  return Err(format!("exact policy oracle requires exactly one evaluator-safe read-only defer action; found {}",matches.len()));
 }
 Ok(matches[0].clone())
}

fn parse_candidate_oracle(candidate:&Value,oracle:&Value)->Result<(BTreeMap<String,Bounds>,BTreeMap<String,Action>,BTreeMap<String,OptionDef>,BTreeMap<String,World>,BTreeMap<String,BTreeSet<String>>,String,u64,usize),String>{
 allowed(candidate,&["scenario_id","decision_cutoff","budget_micros","max_depth","actions","hypothesis_bounds","options","factors","policy_tree"],"candidate_view")?;
 allowed(oracle,&["actual_hypothesis_id","decision_cutoff","budget_micros","actions","hypothesis_bounds","options","factors","hypothesis_safety","calibrations","factorization_certificates","worlds","cost_options"],"oracle")?;
 let cutoff=s(candidate,"decision_cutoff","candidate_view")?;if cutoff!=s(oracle,"decision_cutoff","oracle")?{return Err("candidate cutoff differs from oracle".to_owned());}
 let budget=u(candidate,"budget_micros","candidate_view")?;if budget!=u(oracle,"budget_micros","oracle")?{return Err("candidate budget differs from evaluator budget".to_owned());}
 for k in ["actions","hypothesis_bounds","options","factors"]{if candidate[k]!=oracle[k]{return Err(format!("candidate {k} differ from evaluator-authoritative copies"));}}
 let max_depth=u(candidate,"max_depth","candidate_view")? as usize;if max_depth==0||max_depth>MAX_DEPTH{return Err("max_depth must be 1..=5".to_owned());}
 let bounds=parse_world_bounds(candidate)?;let actions=parse_actions(candidate)?;let options=parse_options(candidate,oracle)?;
 let ids=bounds.keys().cloned().collect::<BTreeSet<_>>();let option_ids=options.keys().cloned().collect::<BTreeSet<_>>();
 let worlds=parse_worlds(oracle,&ids,&option_ids,&actions)?;
 let _defer_action=canonical_defer_action(&actions,&worlds)?;
 let safe_values=val(oracle,"hypothesis_safety","oracle")?.as_array().ok_or("hypothesis_safety must be array")?;
 let safety:BTreeMap<String,BTreeSet<String>>=safe_values.iter().map(|x|Ok((s(x,"hypothesis_id","safety")?,x["safe_actions"].as_array().ok_or("safe_actions must be array")?.iter().map(|a|a.as_str().ok_or("safe action must be string").map(str::to_owned)).collect::<Result<BTreeSet<_>,_>>()?))).collect::<Result<_,String>>()?;
 if safety.keys().cloned().collect::<BTreeSet<_>>()!=ids {
  return Err("oracle hypothesis_safety must cover the exact world set".to_owned());
 }
 for (id,world) in &worlds {
  if safety[id]!=world.safe_actions {
   return Err(format!("oracle world and hypothesis_safety labels differ for {id:?}"));
  }
 }
 let actual=s(oracle,"actual_hypothesis_id","oracle")?;if !worlds.contains_key(&actual){return Err("actual_hypothesis_id is not a world".to_owned());}
 let outcomes=factor_outcomes(candidate)?;
 let factor_rows=candidate["factors"].as_array().ok_or("candidate factors must be array")?;
 for (id,world) in &worlds{
  for (option_id,outcome) in &world.observations{
   let factor_id=&options[option_id].factor_id;
   if !outcomes[factor_id].contains(outcome){
    return Err(format!("world {id:?} has unknown outcome {outcome:?} for {option_id:?}"));
   }
   let factor=factor_rows.iter().find(|f|f["factor_id"].as_str()==Some(factor_id.as_str()))
    .ok_or_else(||format!("missing factor {factor_id:?}"))?;
   let row=factor["likelihoods"].as_array().ok_or("factor likelihoods must be array")?.iter()
    .find(|r|r["hypothesis_id"].as_str()==Some(id.as_str()))
    .ok_or_else(||format!("factor {factor_id:?} missing likelihood for world {id:?}"))?;
   let upper=row["outcomes"][outcome]["upper_likelihood_bps"].as_u64()
    .ok_or_else(||format!("factor {factor_id:?} outcome {outcome:?} is invalid"))?;
   if upper==0{
    return Err(format!("world {id:?} emits outcome {outcome:?} despite zero likelihood in factor {factor_id:?}"));
   }
  }
 }
 Ok((bounds,actions,options,worlds,outcomes,actual,budget,max_depth))
}
fn nearest_rank_expected_cost(bounds:&BTreeMap<String,Bounds>,costs:&BTreeMap<String,u64>)->Result<u64,String>{
 let mut rows:Vec<(String,u64,u64,u64,u64)>=bounds.iter().map(|(id,b)|(id.clone(),b.lower,b.upper,b.nominal,*costs.get(id).unwrap_or(&0))).collect();
 let lower_sum:u64=rows.iter().map(|r|r.1).sum();let mut value:u128=rows.iter().map(|r|r.1 as u128*r.4 as u128).sum();
 let mut remaining=SCALE-lower_sum;rows.sort_by(|a,b|b.4.cmp(&a.4).then_with(||a.0.cmp(&b.0)));
 for row in &rows{let take=remaining.min(row.2-row.1);value+=take as u128*row.4 as u128;remaining-=take;if remaining==0{break;}}
 if remaining>0{return Err("expected-cost prior allocation infeasible".to_owned());}
 Ok(((value+SCALE as u128-1)/SCALE as u128) as u64)
}
fn evaluate_policy(candidate:&Value,oracle:&Value,tree:&PolicyTree,bounds:&BTreeMap<String,Bounds>,actions:&BTreeMap<String,Action>,options:&BTreeMap<String,OptionDef>,worlds:&BTreeMap<String,World>,budget:u64,target_action:&str)->Result<PolicyMetrics,String>{
 let cutoff=parse_time(candidate["decision_cutoff"].as_str().ok_or("candidate cutoff missing")?,"candidate cutoff")?;
 let mut leaf_paths:BTreeMap<String,LeafPath>=BTreeMap::new();let mut world_leaf=BTreeMap::new();let mut world_costs=BTreeMap::new();let mut max_path=0u64;
 for (world_id,world) in worlds{
  let mut node_id=tree.root.clone();let mut steps=Vec::new();let mut seen=BTreeSet::new();let mut previous:Option<DateTime<FixedOffset>>=None;let mut total=0u64;let mut depth=0usize;
  loop{
   depth+=1;if depth>MAX_DEPTH+1{return Err("tree traversal exceeds depth limit".to_owned());}
   if depth.saturating_sub(1)>candidate["max_depth"].as_u64().unwrap_or(0) as usize{
    return Err("submitted policy exceeds candidate max_depth".to_owned());
   }
   let node=&tree.nodes[&node_id];
   if node.kind=="stop"{
    let path=LeafPath{leaf_id:node.id.clone(),steps:steps.clone(),decision_kind:node.decision_kind.clone().unwrap(),action_id:node.action_id.clone().unwrap(),claimed:node.claimed,world_ids:vec![world_id.clone()],cost:total};
    leaf_paths.entry(node.id.clone()).and_modify(|p|p.world_ids.push(world_id.clone())).or_insert(path);
    world_leaf.insert(world_id.clone(),node.id.clone());world_costs.insert(world_id.clone(),total);max_path=max_path.max(total);break;
   }
   let option=node.option_id.as_ref().unwrap();if !seen.insert(option.clone()){return Err(format!("policy repeats option {option:?} on path for {world_id:?}"));}
   let at_raw=node.at.as_ref().unwrap();let at=parse_time(at_raw,"policy acquisition at")?;let opt=&options[option];
   if &at<&opt.available_at||at>cutoff{return Err(format!("option {option:?} is not available at submitted timestamp"));}
   if previous.as_ref().is_some_and(|t|&at<t){return Err("acquisition times must be nondecreasing on each path".to_owned());}
   previous=Some(at);
   let outcome=world.observations.get(option).ok_or_else(||format!("world {world_id:?} has no observation for {option:?}"))?.clone();
   let child=node.branches.get(&outcome).ok_or_else(||format!("policy has no branch for observed outcome {outcome:?}"))?.clone();
   total=total.checked_add(opt.cost_p95).ok_or("path cost overflow")?;
   steps.push(Step{option_id:option.clone(),outcome,at:at_raw.clone()});
   node_id=child;
  }
 }
 let mut leaf_reports=Vec::new();
 let mut leaf_feasible:BTreeMap<String,bool>=BTreeMap::new();
 let mut robust_violations=0usize;let mut budget_violations=0usize;let mut unnecessary=0u64;
 for path in leaf_paths.values(){
  let representative=path.world_ids.first().ok_or("leaf has no reachable worlds")?;
  let risk=risk_for_path(candidate,oracle,&path.steps,representative,&path.action_id,path.claimed)?;
  let action_policy=&actions[&path.action_id];
  let acts=path.decision_kind=="act";
  let risk_ok=risk["evidence_accepted"].as_bool()==Some(true)
   &&risk["robust_action_justified"].as_bool()==Some(true)
   &&(!action_policy.hard_gate||risk["worst_case_posterior_risk_bps"].as_u64()==Some(0));
  let within=path.cost<=budget;
  if !risk_ok{robust_violations+=1;}if !within{budget_violations+=1;}
  leaf_feasible.insert(path.leaf_id.clone(),risk_ok&&within);
  if !acts{
   let target_risk=risk_for_path(candidate,oracle,&path.steps,representative,target_action,SCALE)?;
   let target_justified=target_risk["evidence_accepted"].as_bool()==Some(true)
    &&target_risk["robust_action_justified"].as_bool()==Some(true);
   if target_justified&&within{
    unnecessary=unnecessary.saturating_add(path.world_ids.iter().map(|id|bounds[id].nominal).sum::<u64>());
   }
  }
  leaf_reports.push(json!({
   "leaf_node_id":path.leaf_id,
   "world_ids":path.world_ids,
   "decision_kind":path.decision_kind,
   "action_id":path.action_id,
   "nominal_risk_bps":risk["nominal_posterior_risk_bps"],
   "worst_case_posterior_risk_bps":risk["worst_case_posterior_risk_bps"],
   "risk_qualified":risk_ok,
   "path_cost_bound_micros":path.cost,
   "within_budget":within,
   "feasible":risk_ok&&within
  }));
 }
 let mut coverage=0u64;let mut unsafe_worlds=0usize;
 for (world_id,leaf_id) in &world_leaf{
  let path=&leaf_paths[leaf_id];
  if path.decision_kind=="act"{
   if !worlds[world_id].safe_actions.contains(&path.action_id){unsafe_worlds+=1;}
   if path.action_id==target_action
    &&worlds[world_id].safe_actions.contains(target_action)
    &&leaf_feasible.get(leaf_id)==Some(&true)
   {
    coverage+=bounds[world_id].nominal;
   }
  }
 }
 let nominal_cost:u128=bounds.iter().map(|(id,b)|b.nominal as u128*world_costs[id] as u128).sum();
 let nominal_cost=((nominal_cost+SCALE as u128-1)/SCALE as u128) as u64;
 let worst_expected=nearest_rank_expected_cost(bounds,&world_costs)?;
 let feasible=robust_violations==0&&budget_violations==0;
 let report=json!({
  "policy_feasible":feasible,
  "action_coverage_nominal_bps":coverage,
  "unnecessary_abstention_nominal_bps":unnecessary,
  "unsafe_world_action_count":unsafe_worlds,
  "robust_risk_violation_leaf_count":robust_violations,
  "over_budget_leaf_count":budget_violations,
  "worst_case_path_cost_bound_micros":max_path,
  "nominal_expected_cost_bound_micros":nominal_cost,
  "worst_case_expected_cost_bound_micros":worst_expected,
  "budget_micros":budget,
  "leaf_reports":leaf_reports,
  "world_paths":world_leaf.iter().map(|(id,leaf)|json!({"world_id":id,"leaf_node_id":leaf,"path_cost_bound_micros":world_costs[id]})).collect::<Vec<_>>()
 });
 Ok(PolicyMetrics{report,coverage,worst_expected_cost:worst_expected,nominal_expected_cost:nominal_cost,max_path_cost:max_path,feasible})
}
fn combine_children(
 outcome_keys:&[String],children:&[Vec<SubPolicy>],index:usize,
 branches:&mut BTreeMap<String,String>,nodes:&mut Vec<Value>,root_id:&str,option_id:&str,at:&str,
 output:&mut Vec<SubPolicy>,counter:&mut usize
)->Result<(),String>{
 if output.len()>=MAX_POLICIES{return Err("exact policy enumeration reached 50,000-policy cap".to_owned());}
 if index==children.len(){
  let root=json!({"node_id":root_id,"kind":"acquire","option_id":option_id,"at":at,"branches":branches});
  let mut all=vec![root];all.extend(nodes.clone());output.push(SubPolicy{root:root_id.to_owned(),nodes:all});return Ok(());
 }
 let outcome=&outcome_keys[index];
 for child in &children[index]{
  branches.insert(outcome.clone(),child.root.clone());
  let old=nodes.len();nodes.extend(child.nodes.clone());
  combine_children(outcome_keys,children,index+1,branches,nodes,root_id,option_id,at,output,counter)?;
  nodes.truncate(old);branches.remove(outcome);
 }
 let _=counter;Ok(())
}
fn enumerate_policies(
 remaining:&[String],steps:&[Step],used:&BTreeSet<String>,depth:usize,
 candidate:&Value,oracle:&Value,actions:&BTreeMap<String,Action>,
 options:&BTreeMap<String,OptionDef>,worlds:&BTreeMap<String,World>,outcomes:&BTreeMap<String,BTreeSet<String>>,
 counter:&mut usize,total_generated:&mut usize
)->Result<Vec<SubPolicy>,String>{
 if *total_generated>MAX_POLICIES{return Err("exact policy enumeration cap reached".to_owned());}
 let mut output=Vec::new();
 let defer_action=canonical_defer_action(actions,worlds)?;
 let defer_id=format!("stop-defer-{}",*counter);*counter+=1;
 output.push(SubPolicy{root:defer_id.clone(),nodes:vec![json!({"node_id":defer_id,"kind":"stop","decision_kind":"defer","action_id":defer_action,"claimed_residual_risk_bps":0})]});
 if !remaining.is_empty(){
  for action_id in actions.keys(){
   if action_id==&defer_action{continue;}
   let risk=risk_for_path(candidate,oracle,steps,&remaining[0],action_id,SCALE)?;
   if risk["evidence_accepted"]==true&&risk["robust_action_justified"]==true{
    let justified_risk=risk["worst_case_posterior_risk_bps"].as_u64().unwrap_or(SCALE);
    let id=format!("stop-act-{}",*counter);*counter+=1;
    output.push(SubPolicy{root:id.clone(),nodes:vec![json!({"node_id":id,"kind":"stop","decision_kind":"act","action_id":action_id,"claimed_residual_risk_bps":justified_risk})]});
   }
  }
 }
 if depth>=candidate["max_depth"].as_u64().unwrap_or(0) as usize||remaining.is_empty(){*total_generated+=output.len();return Ok(output);}
 for (option_id,opt) in options{
  if used.contains(option_id)||opt.available_at>parse_time(candidate["decision_cutoff"].as_str().unwrap(),"cutoff")?{continue;}
  let names=outcomes.get(&opt.factor_id).ok_or("missing factor outcomes")?;
  let mut partitions:BTreeMap<String,Vec<String>>=names.iter().map(|o|(o.clone(),Vec::new())).collect();
  for world_id in remaining{let observed=&worlds[world_id].observations[option_id];partitions.get_mut(observed).ok_or("world observation outside factor outcomes")?.push(world_id.clone());}
  let nonempty=partitions.values().filter(|v|!v.is_empty()).count();if nonempty<2{continue;}
  let mut new_used=used.clone();new_used.insert(option_id.clone());
  let at=opt.available_at_raw.clone();
  let mut child_lists=Vec::new();let mut outcome_keys=Vec::new();
  for (outcome,part) in &partitions{
   outcome_keys.push(outcome.clone());
   if part.is_empty(){
    let id=format!("stop-unreachable-{}",*counter);*counter+=1;
    child_lists.push(vec![SubPolicy{root:id.clone(),nodes:vec![json!({"node_id":id,"kind":"stop","decision_kind":"defer","action_id":defer_action,"claimed_residual_risk_bps":0})]}]);
   }else{
    let mut new_steps=steps.to_vec();new_steps.push(Step{option_id:option_id.clone(),outcome:outcome.clone(),at:at.clone()});
    let subs=enumerate_policies(part,&new_steps,&new_used,depth+1,candidate,oracle,actions,options,worlds,outcomes,counter,total_generated)?;
    child_lists.push(subs);
   }
  }
  if child_lists.iter().any(|x|x.is_empty()){continue;}
  let mut combinations=1usize;
  for child in &child_lists{
   combinations=combinations.checked_mul(child.len()).ok_or("exact policy combination count overflow")?;
   if combinations>MAX_POLICIES-output.len(){return Err("exact policy combination cap reached".to_owned());}
  }
  let root_id=format!("acquire-{}",*counter);*counter+=1;
  let mut branches=BTreeMap::new();let mut child_nodes=Vec::new();
  let mut combined=Vec::new();
  combine_children(&outcome_keys,&child_lists,0,&mut branches,&mut child_nodes,&root_id,option_id,&at,&mut combined,counter)?;
  *total_generated+=combined.len();
  output.extend(combined);
  if output.len()>MAX_POLICIES{return Err("exact policy enumeration cap reached".to_owned());}
 }
 Ok(output)
}
fn evaluate_exact(candidate:&Value,oracle:&Value,bounds:&BTreeMap<String,Bounds>,actions:&BTreeMap<String,Action>,options:&BTreeMap<String,OptionDef>,worlds:&BTreeMap<String,World>,outcomes:&BTreeMap<String,BTreeSet<String>>,budget:u64,max_depth:usize)->Result<(PolicyMetrics,usize,PolicyTree),String>{
 let root_worlds=worlds.keys().cloned().collect::<Vec<_>>();
 let mut counter=0usize;let mut total=0usize;
 let policies=enumerate_policies(&root_worlds,&[],&BTreeSet::new(),0,candidate,oracle,actions,options,worlds,outcomes,&mut counter,&mut total)?;
 let mut best:Option<(PolicyMetrics,PolicyTree)>=None;
 let mut feasible_count=0usize;
 for p in policies{
  let tree_json=json!({"root_node_id":p.root,"nodes":p.nodes});
  let tree=parse_nodes(&tree_json,options,outcomes,actions)?;
  let metrics=evaluate_policy(candidate,oracle,&tree,bounds,actions,options,worlds,budget,"release")?;
  if !metrics.feasible{continue;}feasible_count+=1;
  let replace=best.as_ref().map(|(b,_)|{
   metrics.coverage>b.coverage||
    (metrics.coverage==b.coverage&&(metrics.worst_expected_cost<b.worst_expected_cost||
     (metrics.worst_expected_cost==b.worst_expected_cost&&(metrics.nominal_expected_cost<b.nominal_expected_cost||
      (metrics.nominal_expected_cost==b.nominal_expected_cost&&metrics.max_path_cost<b.max_path_cost)))))
  }).unwrap_or(true);
  if replace{best=Some((metrics,tree));}
 }
 let (metrics,tree)=best.ok_or("exact search found no feasible policy (defer baseline should always exist)")?;
 let _=max_depth;
 Ok((metrics,feasible_count,tree))
}

pub fn evaluate_adaptive_policy_json(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=val(&bundle,"candidate_view","bundle")?;let oracle=val(&bundle,"oracle","bundle")?;
 let (bounds,actions,options,worlds,outcomes,actual,budget,max_depth)=parse_candidate_oracle(candidate,oracle)?;
 let tree=parse_nodes(val(candidate,"policy_tree","candidate_view")?,&options,&outcomes,&actions)?;
 let defer_action=canonical_defer_action(&actions,&worlds)?;
 for node in tree.nodes.values(){
  if node.kind=="stop"{
   if node.decision_kind.as_deref()==Some("defer")&&node.action_id.as_deref()!=Some(defer_action.as_str()){
    return Err("defer leaves must use the evaluator's canonical read-only action".to_owned());
   }
   if node.decision_kind.as_deref()==Some("act")&&node.action_id.as_deref()==Some(defer_action.as_str()){
    return Err("canonical defer action cannot be submitted as an act leaf".to_owned());
   }
  }
 }
 let submitted=evaluate_policy(candidate,oracle,&tree,&bounds,&actions,&options,&worlds,budget,"release")?;
 let (optimal,optimal_count,optimal_tree)=evaluate_exact(candidate,oracle,&bounds,&actions,&options,&worlds,&outcomes,budget,max_depth)?;
 let cutoff_raw=s(candidate,"decision_cutoff","candidate_view")?;
 let cutoff=parse_time(&cutoff_raw,"candidate decision cutoff")?;
 let full_revalidation_cost=options.values()
  .filter(|option|option.available_at<=cutoff)
  .try_fold(0u64,|sum,option|sum.checked_add(option.cost_p95).ok_or("full revalidation cost overflow"))?;
 let coverage_regret=optimal.coverage.saturating_sub(submitted.coverage);
 let cost_regret=submitted.worst_expected_cost.saturating_sub(optimal.worst_expected_cost);
 let nominal_regret=submitted.nominal_expected_cost.saturating_sub(optimal.nominal_expected_cost);
 Ok(json!({
  "scenario_id":s(candidate,"scenario_id","candidate_view")?,
  "adjudication_status":"synthetic_known_answer_exact_small_tree_adaptive_policy_evaluation",
  "actual_hypothesis_id_evaluator_only":actual,
  "world_count":worlds.len(),
  "option_count":options.len(),
  "max_depth":max_depth,
  "submitted_policy":submitted.report,
  "exact_oracle":{
   "search_complete":true,
   "feasible_policy_count":optimal_count,
   "objective":"lexicographic: maximize nominal justified target-action coverage; minimize worst-case expected acquisition-cost bound; then nominal expected cost and worst path cost",
   "action_coverage_nominal_bps":optimal.coverage,
   "worst_case_expected_cost_bound_micros":optimal.worst_expected_cost,
   "nominal_expected_cost_bound_micros":optimal.nominal_expected_cost,
   "worst_case_path_cost_bound_micros":optimal.max_path_cost,
   "full_revalidation_comparator":{
    "acquisition_count":options.values().filter(|option|option.available_at<=parse_time(candidate["decision_cutoff"].as_str().unwrap(),"cutoff").unwrap()).count(),
    "cost_bound_micros":full_revalidation_cost,
    "nominal_expected_cost_bound_micros":full_revalidation_cost,
    "worst_case_expected_cost_bound_micros":full_revalidation_cost,
    "action_coverage_nominal_bps":optimal.coverage,
    "nominal_cost_savings_micros":full_revalidation_cost.saturating_sub(optimal.nominal_expected_cost)
   },
   "policy_tree":{"root_node_id":optimal_tree.root,"nodes":optimal_tree.nodes.values().map(|n|node_to_json(n)).collect::<Vec<_>>()}
  },
  "regret":{
   "target_action_coverage_bps":coverage_regret,
   "worst_case_expected_cost_bound_micros":cost_regret,
   "nominal_expected_cost_bound_micros":nominal_regret
  },
  "claim_ceiling":CLAIM_CEILING
 }))
}
fn node_to_json(n:&Node)->Value{
 if n.kind=="acquire"{json!({"node_id":n.id,"kind":"acquire","option_id":n.option_id,"at":n.at,"branches":n.branches})}
 else{json!({"node_id":n.id,"kind":"stop","decision_kind":n.decision_kind,"action_id":n.action_id,"claimed_residual_risk_bps":n.claimed})}
}

fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{eprintln!("Usage: adaptive-policy-oracle <candidate-and-oracle-bundle.json>");process::exit(if args.len()==1{0}else{2});}
 let raw=match fs::read_to_string(&args[0]){Ok(v)=>v,Err(e)=>{eprintln!("ERROR D6.17.2.1-IO-001 {}: {e}",args[0]);process::exit(2)}};
 match evaluate_adaptive_policy_json(&raw){Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),Err(e)=>{eprintln!("ERROR D6.17.2.1-EVALUATION-001: {e}");process::exit(1)}}
}

#[cfg(test)]
mod tests{
 use super::*;
 const FIXTURE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-adaptive-policy-v0.1.json");
 fn report()->Value{evaluate_adaptive_policy_json(FIXTURE).expect("adaptive policy fixture evaluates")}
 #[test]fn complete_report_matches_known_answer(){let r=report();let expected:Value=serde_json::from_str(include_str!("../../../../docs/development-map/fixtures/egddm-adaptive-policy-v0.1.expected.json")).unwrap();assert_eq!(r,expected);}
 #[test]fn adaptive_policy_stops_after_first_gate_failure(){let r=report();let world=&r["submitted_policy"]["world_paths"];let haz=world.as_array().unwrap().iter().find(|x|x["world_id"]=="hazard-b").unwrap();assert_eq!(haz["path_cost_bound_micros"],700);}
 #[test]fn positive_world_requires_both_checks_before_release(){let r=report();let safe=r["submitted_policy"]["world_paths"].as_array().unwrap().iter().find(|x|x["world_id"]=="safe").unwrap();assert_eq!(safe["path_cost_bound_micros"],1500);assert_eq!(r["submitted_policy"]["action_coverage_nominal_bps"],6000);}
 #[test]fn exact_oracle_finds_complete_coverage_with_lower_cost_than_full_revalidation(){let r=report();assert_eq!(r["exact_oracle"]["search_complete"],true);assert_eq!(r["exact_oracle"]["action_coverage_nominal_bps"],6000);assert_eq!(r["exact_oracle"]["nominal_expected_cost_bound_micros"],1300);assert_eq!(r["exact_oracle"]["worst_case_path_cost_bound_micros"],1500);}
 #[test]fn exact_policy_cost_is_below_full_revalidation(){
  let r=report();
  assert_eq!(r["exact_oracle"]["worst_case_expected_cost_bound_micros"],1420);
  assert_eq!(r["exact_oracle"]["nominal_expected_cost_bound_micros"],1300);
  assert_eq!(r["exact_oracle"]["full_revalidation_comparator"]["cost_bound_micros"],1500);
  assert_eq!(r["exact_oracle"]["full_revalidation_comparator"]["nominal_cost_savings_micros"],200);
  assert_eq!(r["exact_oracle"]["feasible_policy_count"],13);
 }
 #[test]fn candidate_cannot_change_budget(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["budget_micros"]=json!(99999);assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("budget differs"));}
 #[test]fn candidate_cannot_change_world_probability_bounds(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["hypothesis_bounds"][0]["upper_probability_bps"]=json!(9999);assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("differ from evaluator-authoritative"));}
 #[test]fn missing_observation_branch_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["policy_tree"]["nodes"][0]["branches"].as_object_mut().unwrap().remove("fail");assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("cover every factor outcome"));}
 #[test]fn repeated_option_on_a_path_is_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["policy_tree"]["nodes"][1]["option_id"]=json!("gate-a");assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("repeats option"));}
 #[test]fn more_than_one_global_safe_defer_action_is_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  let extra=json!({"action_id":"inspect-second","irreversible":false,"hard_gate":false,"max_residual_risk_bps":0});
  b["candidate_view"]["actions"].as_array_mut().unwrap().push(extra.clone());
  b["oracle"]["actions"].as_array_mut().unwrap().push(extra);
  for world in b["oracle"]["worlds"].as_array_mut().unwrap(){
   world["safe_actions"].as_array_mut().unwrap().push(json!("inspect-second"));
  }
  for row in b["oracle"]["hypothesis_safety"].as_array_mut().unwrap(){
   row["safe_actions"].as_array_mut().unwrap().push(json!("inspect-second"));
  }
  assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("exactly one evaluator-safe read-only defer action"));
 }
 #[test]fn oracle_cost_options_must_match_candidate_ids_exactly(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  b["oracle"]["cost_options"][1]["option_id"]=json!("gate-a");
  assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("cost option IDs differ"));
 }
 #[test]fn world_outcomes_with_zero_registered_likelihood_are_rejected(){
  let mut b:Value=serde_json::from_str(FIXTURE).unwrap();
  for root in [&mut b["candidate_view"],&mut b["oracle"]]{ 
   let row=root["factors"][0]["likelihoods"].as_array_mut().unwrap().iter_mut()
    .find(|x|x["hypothesis_id"]=="safe").unwrap();
   row["outcomes"]["pass"]=json!({"lower_likelihood_bps":0,"upper_likelihood_bps":0,"nominal_likelihood_bps":0});
   row["outcomes"]["fail"]=json!({"lower_likelihood_bps":10000,"upper_likelihood_bps":10000,"nominal_likelihood_bps":10000});
  }
  assert!(evaluate_adaptive_policy_json(&b.to_string()).unwrap_err().contains("zero likelihood"));
 }
 #[test]fn cycles_are_rejected(){let mut b:Value=serde_json::from_str(FIXTURE).unwrap();b["candidate_view"]["policy_tree"]["nodes"][0]["branches"]["pass"]=json!("root");b["candidate_view"]["policy_tree"]["nodes"][0]["node_id"]=json!("root");b["candidate_view"]["policy_tree"]["root_node_id"]=json!("root");assert!(evaluate_adaptive_policy_json(&b.to_string()).is_err());}
}
