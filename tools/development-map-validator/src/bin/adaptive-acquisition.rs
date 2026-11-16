//! D6.16 evaluator-only exact oracle for adaptive risk-constrained evidence acquisition.
//! Small finite POMDP; evaluator-supplied priors, likelihoods, action safety and query costs.
//! No production planning, authorization, or side effects.

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::{env, fs, process};

const CLAIM_CEILING: &str = "Exact dynamic-programming oracle over a bounded finite POMDP with evaluator-supplied world priors, conditional observation likelihoods, action-safety labels and p95 query costs. Does not prove real-world probability calibration, general-POMDP optimality outside this finite model, production authority, safe autonomy, or external benchmark superiority.";

#[derive(Clone)]
struct Action { hard_gate: bool, max_risk_bps: u64 }
#[derive(Clone)]
struct OptionModel { id: String, available_at: DateTime<FixedOffset>, p95_cost: u64, likelihoods: Vec<BTreeMap<String,u64>> }
#[derive(Clone)]
struct World { id: String, prior_bps: u64, safe_actions: BTreeSet<String> }
#[derive(Clone)]
struct Contract {
    cutoff: DateTime<FixedOffset>, budget: u64, defer_penalty: u64,
    max_acquisitions: usize, actions: BTreeMap<String,Action>, options: Vec<OptionModel>,
}
#[derive(Clone)]
struct Outcome { label: String, probability: f64, belief: Vec<u128> }
#[derive(Clone)]
struct Plan {
    total: f64, acquisition: f64, defer: f64, queries: f64, max_queries: usize, policy: Value,
}
#[derive(Clone, Hash, PartialEq, Eq)]
struct DpKey { belief: Vec<u128>, remaining: u16, budget: u64 }
#[derive(Clone)]
struct Metrics {
    total: f64, acquisition: f64, defer: f64, queries: f64, max_queries: usize, leaves: usize,
    unsupported: f64, hard_gate: f64, risk_violation: f64, actual_unsafe: f64, unnecessary_defer: f64,
}

fn obj<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
    v.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn keys(v:&Value,allowed:&[&str],label:&str)->Result<(),String>{
    for k in obj(v,label)?.keys(){if !allowed.contains(&k.as_str()){return Err(format!("{label} contains unknown field {k:?}"));}}
    Ok(())
}
fn req<'a>(v:&'a Value,k:&str,label:&str)->Result<&'a Value,String>{
    v.get(k).ok_or_else(||format!("{label}.{k} is required"))
}
fn strv(v:&Value,k:&str,label:&str)->Result<String,String>{
    req(v,k,label)?.as_str().filter(|s|!s.trim().is_empty()).map(str::to_owned)
        .ok_or_else(||format!("{label}.{k} must be a non-empty string"))
}
fn num(v:&Value,k:&str,label:&str)->Result<u64,String>{
    req(v,k,label)?.as_u64().ok_or_else(||format!("{label}.{k} must be a non-negative integer"))
}
fn boolv(v:&Value,k:&str,label:&str)->Result<bool,String>{
    req(v,k,label)?.as_bool().ok_or_else(||format!("{label}.{k} must be boolean"))
}
fn timev(v:&Value,k:&str,label:&str)->Result<DateTime<FixedOffset>,String>{
    let raw=strv(v,k,label)?;DateTime::parse_from_rfc3339(&raw).map_err(|e|format!("{label}.{k} must be RFC 3339: {e}"))
}
fn setv(v:&Value,label:&str)->Result<BTreeSet<String>,String>{
    let a=v.as_array().ok_or_else(||format!("{label} must be an array"))?;let mut out=BTreeSet::new();
    for (i,x) in a.iter().enumerate(){let s=x.as_str().filter(|s|!s.trim().is_empty()).ok_or_else(||format!("{label}[{i}] must be a non-empty string"))?;
        if !out.insert(s.to_owned()){return Err(format!("{label} contains duplicate {s:?}"));}}
    Ok(out)
}
fn p95(v:&Value,label:&str)->Result<u64,String>{
    let a=v.as_array().ok_or_else(||format!("{label} must be an array"))?;
    if a.is_empty(){return Err(format!("{label} must not be empty; unknown cost is not zero"));}
    let mut xs=Vec::new();for (i,x) in a.iter().enumerate(){xs.push(x.as_u64().filter(|n|*n>0).ok_or_else(||format!("{label}[{i}] must be positive"))?);}
    xs.sort_unstable();let rank=(xs.len()*95+99)/100;Ok(xs[rank.saturating_sub(1)])
}
fn gcd(mut a:u128,mut b:u128)->u128{while b!=0{let r=a%b;a=b;b=r;}a}
fn normalize(xs:&mut[u128]){let g=xs.iter().copied().filter(|n|*n>0).reduce(gcd).unwrap_or(1);if g>1{for x in xs{*x/=g;}}}
fn mass(xs:&[u128])->u128{xs.iter().copied().sum()}
fn risk(belief:&[u128],worlds:&[World],action_id:&str)->(u64,u128,u128){
    let total=mass(belief);if total==0{return(10_000,0,0);}
    let unsafe_mass=belief.iter().zip(worlds).filter(|(m,w)|**m>0&&!w.safe_actions.contains(action_id)).map(|(m,_)|*m).sum::<u128>();
    let bps=if unsafe_mass==0{0}else{((unsafe_mass as f64/total as f64)*10_000.0).ceil().clamp(0.0,10_000.0)as u64};
    (bps,unsafe_mass,total)
}
fn justified(action:&Action,r:u64,unsafe_mass:u128)->bool{if action.hard_gate{unsafe_mass==0}else{r<=action.max_risk_bps}}
fn safe_action(belief:&[u128],worlds:&[World],actions:&BTreeMap<String,Action>)->Option<String>{
    actions.iter().find_map(|(id,a)|{let(r,u,_)=risk(belief,worlds,id);justified(a,r,u).then(||id.clone())})
}
fn outcomes(belief:&[u128],option_index:usize,contract:&Contract)->Result<Vec<Outcome>,String>{
    let option=contract.options.get(option_index).ok_or("query option index missing")?;
    let total=mass(belief);if total==0{return Err("empty belief is invalid".to_owned());}
    let mut labels=BTreeSet::new();
    for(i,m)in belief.iter().enumerate(){if *m>0{for(l,p)in &option.likelihoods[i]{if *p>0{labels.insert(l.clone());}}}}
    let denom=total as f64*10_000.0;let mut result=Vec::new();
    for label in labels{
        let mut next=Vec::with_capacity(belief.len());
        for(i,m)in belief.iter().enumerate(){let p=*option.likelihoods[i].get(&label).unwrap_or(&0)as u128;next.push(m.checked_mul(p).ok_or("belief mass overflow")?);}
        let next_mass=mass(&next);if next_mass==0{continue;}normalize(&mut next);
        result.push(Outcome{label,probability:next_mass as f64/denom,belief:next});
    }
    Ok(result)
}
fn parse_contract(value:&Value,oracle:&Value)->Result<Contract,String>{
    keys(value,&["decision_cutoff","budget_micros","defer_penalty_micros","max_acquisitions","actions","options"],"trusted_contract")?;
    let cutoff=timev(value,"decision_cutoff","trusted_contract")?;let budget=num(value,"budget_micros","trusted_contract")?;
    let defer_penalty=num(value,"defer_penalty_micros","trusted_contract")?;
    let max_acquisitions=num(value,"max_acquisitions","trusted_contract")? as usize;
    if max_acquisitions==0||max_acquisitions>8{return Err("max_acquisitions must be within 1..=8".to_owned());}
    let av=req(value,"actions","trusted_contract")?.as_array().ok_or("actions must be an array")?;
    if av.is_empty(){return Err("trusted_contract.actions must not be empty".to_owned());}
    let mut actions=BTreeMap::new();
    for(i,item)in av.iter().enumerate(){let l=format!("trusted_contract.actions[{i}]");keys(item,&["action_id","hard_gate","irreversible","max_residual_risk_bps"],&l)?;
        let id=strv(item,"action_id",&l)?;let hard_gate=boolv(item,"hard_gate",&l)?;let irreversible=boolv(item,"irreversible",&l)?;
        let max_risk_bps=num(item,"max_residual_risk_bps",&l)?;if max_risk_bps>10_000{return Err(format!("{l}.max_residual_risk_bps > 10000"));}
        if hard_gate&&max_risk_bps!=0{return Err(format!("{l}: hard gate must have zero residual-risk allowance"));}
        if irreversible&&!hard_gate{return Err(format!("{l}: irreversible action requires hard_gate"));}
        if actions.insert(id.clone(),Action{hard_gate,max_risk_bps}).is_some(){return Err(format!("duplicate action_id {id:?}"));}
    }
    let ovs=req(value,"options","trusted_contract")?.as_array().ok_or("options must be an array")?;
    if ovs.is_empty()||ovs.len()>8{return Err("trusted_contract.options must contain 1..=8 options".to_owned());}
    let raw_models=req(oracle,"observation_models","oracle")?.as_array().ok_or("observation_models must be an array")?;
    if raw_models.len()!=ovs.len(){return Err("observation model set must match trusted options".to_owned());}
    let mut options=BTreeMap::new();
    for(i,item)in ovs.iter().enumerate(){let l=format!("trusted_contract.options[{i}]");keys(item,&["option_id","available_at"],&l)?;
        let id=strv(item,"option_id",&l)?;let available_at=timev(item,"available_at",&l)?;
        let model=raw_models.iter().find(|m|m.get("option_id").and_then(Value::as_str)==Some(id.as_str())).ok_or_else(||format!("missing observation model for {id}"))?;
        let cost_obj=req(model,"cost_samples_micros","observation_model")?;keys(cost_obj,&["cold"],"cost_samples_micros")?;
        let cost=p95(req(cost_obj,"cold","cost_samples_micros")?,"cost_samples_micros.cold")?;
        if options.insert(id.clone(),OptionModel{id:id.clone(),available_at,p95_cost:cost,likelihoods:Vec::new()}).is_some(){return Err(format!("duplicate option_id {id:?}"));}
    }
    Ok(Contract{cutoff,budget,defer_penalty,max_acquisitions,actions,options:options.into_values().collect()})
}
fn parse_worlds(oracle:&Value,actions:&BTreeMap<String,Action>)->Result<Vec<World>,String>{
    let a=req(oracle,"worlds","oracle")?.as_array().ok_or("oracle.worlds must be array")?;
    if a.is_empty()||a.len()>8{return Err("oracle.worlds must contain 1..=8 worlds".to_owned());}
    let mut worlds=BTreeMap::new();let mut sum=0u64;
    for(i,item)in a.iter().enumerate(){let l=format!("oracle.worlds[{i}]");keys(item,&["world_id","prior_probability_bps","safe_actions"],&l)?;
        let id=strv(item,"world_id",&l)?;let prior=num(item,"prior_probability_bps",&l)?;
        if prior==0||prior>10_000{return Err(format!("{l}.prior_probability_bps must be 1..=10000"));}sum=sum.checked_add(prior).ok_or("prior overflow")?;
        let safe=setv(req(item,"safe_actions",&l)?,&format!("{l}.safe_actions"))?;
        if safe.iter().any(|x|!actions.contains_key(x)){return Err(format!("{l}.safe_actions contains unknown action"));}
        if worlds.insert(id.clone(),World{id:id.clone(),prior_bps:prior,safe_actions:safe}).is_some(){return Err(format!("duplicate world {id:?}"));}
    }
    if sum!=10_000{return Err(format!("world priors must sum to 10000, got {sum}"));}Ok(worlds.into_values().collect())
}
fn parse_models(oracle:&Value,worlds:&[World],contract:&mut Contract)->Result<(),String>{
    let raw=req(oracle,"observation_models","oracle")?.as_array().ok_or("observation_models must be array")?;
    if raw.len()!=contract.options.len(){return Err("observation_models must match options exactly".to_owned());}
    let world_ids=worlds.iter().map(|w|w.id.clone()).collect::<BTreeSet<_>>();
    for option in &mut contract.options{
        let item=raw.iter().find(|m|m.get("option_id").and_then(Value::as_str)==Some(option.id.as_str())).ok_or_else(||format!("missing model for {}",option.id))?;
        keys(item,&["option_id","cost_samples_micros","by_world"],"observation_model")?;
        let costs=req(item,"cost_samples_micros","observation_model")?;keys(costs,&["cold"],"cost_samples_micros")?;
        let cost=p95(req(costs,"cold","cost_samples_micros")?,"cost_samples_micros.cold")?;
        if cost!=option.p95_cost{return Err("inconsistent cost samples during model binding".to_owned());}
        let entries=req(item,"by_world","observation_model")?.as_array().ok_or("by_world must be array")?;
        if entries.len()!=worlds.len(){return Err(format!("{} must cover all worlds",option.id));}
        let mut map=BTreeMap::new();
        for(i,entry)in entries.iter().enumerate(){let l=format!("{}.by_world[{i}]",option.id);keys(entry,&["world_id","outcomes"],&l)?;
            let wid=strv(entry,"world_id",&l)?;if !world_ids.contains(&wid){return Err(format!("{l} unknown world {wid:?}"));}
            let distribution=obj(req(entry,"outcomes",&l)?,&format!("{l}.outcomes"))?;if distribution.is_empty(){return Err(format!("{l}.outcomes must not be empty"));}
            let mut parsed=BTreeMap::new();let mut total=0u64;
            for(label,p)in distribution{if label.trim().is_empty(){return Err("empty outcome label".to_owned());}
                let n=p.as_u64().ok_or_else(||format!("{l}.outcomes.{label} must be integer"))?;if n>10_000{return Err(format!("{l}.outcomes.{label} > 10000"));}total+=n;parsed.insert(label.clone(),n);}
            if total!=10_000{return Err(format!("{l}.outcomes must sum to 10000, got {total}"));}
            if map.insert(wid,parsed).is_some(){return Err(format!("{l} duplicates world"));}
        }
        if map.len()!=worlds.len(){return Err(format!("{} observation model missing world",option.id));}
        option.likelihoods=worlds.iter().map(|w|map.get(&w.id).cloned().expect("world set checked")).collect();
    }Ok(())
}
fn initial(worlds:&[World])->Vec<u128>{worlds.iter().map(|w|w.prior_bps as u128).collect()}
fn defer_plan(p:u64)->Plan{Plan{total:p as f64,acquisition:0.0,defer:p as f64,queries:0.0,max_queries:0,policy:json!({"kind":"defer","reason_code":"no_safe_action_within_budget"})}}
fn act_plan(id:String)->Plan{Plan{total:0.0,acquisition:0.0,defer:0.0,queries:0.0,max_queries:0,policy:json!({"kind":"act","action_id":id})}}
fn is_better(a:&Plan,b:&Plan)->bool{a.total<b.total-1e-9||((a.total-b.total).abs()<=1e-9&&a.policy.to_string()<b.policy.to_string())}
fn solve(belief:Vec<u128>,mask:u16,budget:u64,contract:&Contract,worlds:&[World],memo:&mut HashMap<DpKey,Plan>)->Result<Plan,String>{
    if mass(&belief)==0{return Err("empty belief in exact solver".to_owned());}
    let key=DpKey{belief:belief.clone(),remaining:mask,budget};if let Some(p)=memo.get(&key){return Ok(p.clone());}
    if let Some(id)=safe_action(&belief,worlds,&contract.actions){let p=act_plan(id);memo.insert(key,p.clone());return Ok(p);}
    let mut best=defer_plan(contract.defer_penalty);let used=contract.options.len()-mask.count_ones()as usize;
    if used<contract.max_acquisitions{
        for(i,option)in contract.options.iter().enumerate(){let bit=1u16<<i;if mask&bit==0||option.available_at>contract.cutoff||option.p95_cost>budget{continue;}
            let branches=outcomes(&belief,i,contract)?;if branches.is_empty(){continue;}
            let mut p=Plan{total:option.p95_cost as f64,acquisition:option.p95_cost as f64,defer:0.0,queries:1.0,max_queries:0,policy:Value::Null};
            let mut policy_branches=serde_json::Map::new();
            for branch in branches{let child=solve(branch.belief,mask&!bit,budget-option.p95_cost,contract,worlds,memo)?;
                p.total+=branch.probability*child.total;p.acquisition+=branch.probability*child.acquisition;
                p.defer+=branch.probability*child.defer;p.queries+=branch.probability*child.queries;
                p.max_queries=p.max_queries.max(1+child.max_queries);policy_branches.insert(branch.label,child.policy);}
            p.policy=json!({"kind":"acquire","option_id":option.id,"outcome_policies":policy_branches});
            if is_better(&p,&best){best=p;}
        }
    }
    memo.insert(key,best.clone());Ok(best)
}
fn evaluate_node(
    id:&str,nodes:&BTreeMap<String,Value>,belief:Vec<u128>,used:u16,spent:u64,prev:Option<DateTime<FixedOffset>>,
    visited:&mut BTreeSet<String>,contract:&Contract,worlds:&[World],
)->Result<Metrics,String>{
    let node=nodes.get(id).ok_or_else(||format!("policy references missing node {id:?}"))?;
    if !visited.insert(id.to_owned()){return Err(format!("policy is not a tree: repeated/cyclic node {id:?}"));}
    let label=format!("policy node {id}");let kind=strv(node,"kind",&label)?;
    match kind.as_str(){
        "act"=>{
            keys(node,&["node_id","kind","action_id"],&label)?;let action_id=strv(node,"action_id",&label)?;
            let action=contract.actions.get(&action_id).ok_or_else(||format!("{label} unknown action {action_id:?}"))?;
            let(r,u,t)=risk(&belief,worlds,&action_id);let ok=justified(action,r,u);
            Ok(Metrics{total:0.0,acquisition:0.0,defer:0.0,queries:0.0,max_queries:0,leaves:1,
                unsupported:if ok{0.0}else{1.0},hard_gate:if action.hard_gate&&!ok{1.0}else{0.0},
                risk_violation:if !action.hard_gate&&!ok{1.0}else{0.0},
                actual_unsafe:u as f64/t as f64,unnecessary_defer:0.0})
        },
        "defer"=>{
            keys(node,&["node_id","kind","reason_code"],&label)?;let _reason=strv(node,"reason_code",&label)?;
            let unnecessary=safe_action(&belief,worlds,&contract.actions).is_some();
            Ok(Metrics{total:contract.defer_penalty as f64,acquisition:0.0,defer:contract.defer_penalty as f64,
                queries:0.0,max_queries:0,leaves:1,unsupported:0.0,hard_gate:0.0,risk_violation:0.0,
                actual_unsafe:0.0,unnecessary_defer:if unnecessary{1.0}else{0.0}})
        },
        "acquire"=>{
            keys(node,&["node_id","kind","option_id","at","outcome_edges"],&label)?;
            let option_id=strv(node,"option_id",&label)?;
            let index=contract.options.iter().position(|o|o.id==option_id).ok_or_else(||format!("{label} unknown option {option_id:?}"))?;
            let option=&contract.options[index];let bit=1u16<<index;
            if used&bit!=0{return Err(format!("{label} re-acquires {option_id:?} on one path"));}
            if used.count_ones()as usize>=contract.max_acquisitions{return Err(format!("{label} exceeds max_acquisitions"));}
            if option.available_at>contract.cutoff{return Err(format!("{label} option unavailable by cutoff"));}
            let at=timev(node,"at",&label)?;
            if at<option.available_at||at>contract.cutoff{return Err(format!("{label}.at outside option/cutoff availability"));}
            if prev.as_ref().is_some_and(|x|at<*x){return Err(format!("{label}.at moves backwards"));}
            let new_spent=spent.checked_add(option.p95_cost).ok_or("cost overflow")?;
            if new_spent>contract.budget{return Err(format!("{label} exceeds path budget"));}
            let edges=obj(req(node,"outcome_edges",&label)?,&format!("{label}.outcome_edges"))?;
            let branches=outcomes(&belief,index,contract)?;
            let possible=branches.iter().map(|b|b.label.clone()).collect::<BTreeSet<_>>();
            let supplied=edges.keys().cloned().collect::<BTreeSet<_>>();
            if possible!=supplied{return Err(format!("{label} outcome_edges must exactly cover possible outcomes; possible={possible:?}, supplied={supplied:?}"));}
            let mut result=Metrics{total:option.p95_cost as f64,acquisition:option.p95_cost as f64,defer:0.0,
                queries:1.0,max_queries:0,leaves:0,unsupported:0.0,hard_gate:0.0,risk_violation:0.0,actual_unsafe:0.0,unnecessary_defer:0.0};
            for branch in branches{
                let child_id=edges.get(&branch.label).and_then(Value::as_str).ok_or_else(||format!("{label} invalid edge for {}",branch.label))?;
                let child=evaluate_node(child_id,nodes,branch.belief,used|bit,new_spent,Some(at),visited,contract,worlds)?;
                result.total+=branch.probability*child.total;result.acquisition+=branch.probability*child.acquisition;
                result.defer+=branch.probability*child.defer;result.queries+=branch.probability*child.queries;
                result.max_queries=result.max_queries.max(1+child.max_queries);result.leaves+=child.leaves;
                result.unsupported+=branch.probability*child.unsupported;result.hard_gate+=branch.probability*child.hard_gate;
                result.risk_violation+=branch.probability*child.risk_violation;
                result.actual_unsafe+=branch.probability*child.actual_unsafe;
                result.unnecessary_defer+=branch.probability*child.unnecessary_defer;
            }Ok(result)
        },
        other=>Err(format!("{label} has unsupported kind {other:?}")),
    }
}
fn bp(p:f64)->u64{(p.clamp(0.0,1.0)*10_000.0).round()as u64}

pub fn evaluate_adaptive_acquisition_json(raw:&str)->Result<Value,String>{
    let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
    keys(&bundle,&["candidate_view","oracle"],"bundle")?;
    let candidate=req(&bundle,"candidate_view","bundle")?;let oracle=req(&bundle,"oracle","bundle")?;
    keys(candidate,&["scenario_id","trusted_contract","policies"],"candidate_view")?;
    keys(oracle,&["trusted_contract","worlds","observation_models"],"oracle")?;
    let scenario_id=strv(candidate,"scenario_id","candidate_view")?;
    let cc=req(candidate,"trusted_contract","candidate_view")?;let oc=req(oracle,"trusted_contract","oracle")?;
    if cc!=oc{return Err("candidate trusted_contract differs from evaluator-authoritative contract".to_owned());}
    let mut contract=parse_contract(oc,oracle)?;
    let worlds=parse_worlds(oracle,&contract.actions)?;parse_models(oracle,&worlds,&mut contract)?;
    let start=initial(&worlds);let mask=(1u16<<contract.options.len())-1;let mut memo=HashMap::new();
    let exact=solve(start.clone(),mask,contract.budget,&contract,&worlds,&mut memo)?;
    let policies=req(candidate,"policies","candidate_view")?.as_array().ok_or("candidate_view.policies must be array")?;
    if policies.is_empty()||policies.len()>32{return Err("candidate_view.policies must contain 1..=32 policies".to_owned());}
    let mut seen=BTreeSet::new();let mut reports=Vec::new();
    for(i,p)in policies.iter().enumerate(){
        let l=format!("candidate_view.policies[{i}]");keys(p,&["policy_id","root_node_id","nodes"],&l)?;
        let policy_id=strv(p,"policy_id",&l)?;if !seen.insert(policy_id.clone()){return Err(format!("duplicate policy_id {policy_id:?}"));}
        let root=strv(p,"root_node_id",&l)?;let raw_nodes=req(p,"nodes",&l)?.as_array().ok_or("policy nodes must be array")?;
        if raw_nodes.is_empty()||raw_nodes.len()>256{return Err("policy nodes must contain 1..=256".to_owned());}
        let mut nodes=BTreeMap::new();
        for(n,node)in raw_nodes.iter().enumerate(){let nl=format!("{l}.nodes[{n}]");let id=strv(node,"node_id",&nl)?;
            if nodes.insert(id.clone(),node.clone()).is_some(){return Err(format!("duplicate node_id {id:?}"));}}
        let mut visited=BTreeSet::new();let m=evaluate_node(&root,&nodes,start.clone(),0,0,None,&mut visited,&contract,&worlds)?;
        if visited.len()!=nodes.len(){return Err(format!("{l} contains unreachable nodes"));}
        let safe=m.unsupported==0.0;
        reports.push(json!({
            "policy_id":policy_id,"expected_total_cost_micros":m.total,
            "expected_acquisition_cost_micros":m.acquisition,"expected_defer_penalty_micros":m.defer,
            "expected_acquisition_count":m.queries,"max_acquisitions_on_path":m.max_queries,"leaf_count":m.leaves,
            "unsupported_action_probability_bps":bp(m.unsupported),"hard_gate_violation_probability_bps":bp(m.hard_gate),
            "risk_envelope_violation_probability_bps":bp(m.risk_violation),"actual_unsafe_action_probability_bps":bp(m.actual_unsafe),
            "unnecessary_defer_probability_bps":bp(m.unnecessary_defer),
            "safe_policy_cost_regret_micros":if safe{json!(m.total-exact.total)}else{Value::Null}
        }));
    }
    Ok(json!({
        "scenario_id":scenario_id,"adjudication_status":"synthetic_finite_pomdp_known_answer",
        "world_count":worlds.len(),"option_count":contract.options.len(),"action_count":contract.actions.len(),
        "candidate_policy_count":reports.len(),
        "exact_oracle":{
            "expected_total_cost_micros":exact.total,"expected_acquisition_cost_micros":exact.acquisition,
            "expected_defer_penalty_micros":exact.defer,"expected_acquisition_count":exact.queries,
            "max_acquisitions_on_path":exact.max_queries,"memoized_belief_states":memo.len(),"policy":exact.policy
        },
        "policies":reports,"claim_ceiling":CLAIM_CEILING
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    const FIXTURE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-adaptive-acquisition-v0.1.json");
    const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-adaptive-acquisition-v0.1.expected.json");
    fn report()->Value{evaluate_adaptive_acquisition_json(FIXTURE).expect("D6.16 fixture")}
    fn policy<'a>(r:&'a Value,id:&str)->&'a Value{r["policies"].as_array().unwrap().iter().find(|p|p["policy_id"]==id).expect("policy")}
    #[test]fn expected_report_matches_independent_known_answer(){let a=report();let e:Value=serde_json::from_str(EXPECTED).unwrap();assert_eq!(a,e);}
    #[test]fn exact_oracle_finds_branching_policy(){let r=report();assert_eq!(r["exact_oracle"]["expected_total_cost_micros"],1100.0);assert_eq!(r["exact_oracle"]["expected_acquisition_count"],1.4);assert_eq!(r["exact_oracle"]["policy"]["option_id"],"verify-release");}
    #[test]fn adaptive_policy_matches_oracle_and_beats_fixed_classifier(){let r=report();assert_eq!(policy(&r,"adaptive-verify-then-classify")["safe_policy_cost_regret_micros"],0.0);assert_eq!(policy(&r,"fixed-classifier")["expected_total_cost_micros"],1500.0);}
    #[test]fn noisy_screen_cannot_authorize_irreversible_release(){let p=policy(&report(),"greedy-screen-only");assert_eq!(p["hard_gate_violation_probability_bps"],5800);assert_eq!(p["actual_unsafe_action_probability_bps"],400);}
    #[test]fn full_static_checks_are_safe_but_more_expensive(){let r=report();let full=policy(&r,"full-static-acquisition");assert_eq!(full["expected_total_cost_micros"],2300.0);assert!(full["expected_total_cost_micros"].as_f64().unwrap()>1100.0);}
    #[test]fn always_act_and_always_defer_are_both_measured(){let r=report();assert_eq!(policy(&r,"always-act")["hard_gate_violation_probability_bps"],10000);assert_eq!(policy(&r,"always-act")["actual_unsafe_action_probability_bps"],4000);assert_eq!(policy(&r,"always-defer")["expected_defer_penalty_micros"],10000.0);}
    #[test]fn candidate_cannot_inflate_budget_or_change_cutoff(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["candidate_view"]["trusted_contract"]["budget_micros"]=json!(999999);assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("differs"));}
    #[test]fn missing_possible_observation_branch_is_rejected(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["candidate_view"]["policies"][0]["nodes"][0]["outcome_edges"].as_object_mut().unwrap().remove("fail");assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("exactly cover possible outcomes"));}
    #[test]fn unknown_cost_is_not_zero(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["oracle"]["observation_models"][0]["cost_samples_micros"]["cold"]=json!([]);assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("must not be empty"));}
    #[test]fn observation_distributions_must_sum_to_one(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["oracle"]["observation_models"][0]["by_world"][0]["outcomes"]["ghost"]=json!(100);assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("sum to 10000"));}
    #[test]fn query_time_must_be_available_and_monotone(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["candidate_view"]["policies"][0]["nodes"][0]["at"]=json!("2026-11-16T07:00:00Z");assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("outside option/cutoff"));}
    #[test]fn repeated_option_on_a_path_is_rejected(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["candidate_view"]["policies"][0]["nodes"][2]["option_id"]=json!("verify-release");assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("re-acquires"));}
    #[test]fn cycles_and_shared_nodes_are_rejected(){let mut v:Value=serde_json::from_str(FIXTURE).unwrap();v["candidate_view"]["policies"][0]["nodes"][0]["outcome_edges"]["pass"]=json!("root");assert!(evaluate_adaptive_acquisition_json(&v.to_string()).unwrap_err().contains("repeated/cyclic"));}
}

fn main(){
    let args:Vec<String>=env::args().skip(1).collect();
    if args.len()!=1||args[0]=="--help"||args[0]=="-h"{eprintln!("Usage: adaptive-acquisition <bundle.json>");eprintln!("Evaluator-only D6.16 finite-POMDP policy evaluator.");process::exit(if args.len()==1{0}else{2});}
    let path=&args[0];let input=match fs::read_to_string(path){Ok(s)=>s,Err(e)=>{eprintln!("ERROR D6.16-IO-001 {path}: {e}");process::exit(2);}};
    match evaluate_adaptive_acquisition_json(&input){Ok(report)=>match serde_json::to_string_pretty(&report){Ok(s)=>println!("{s}"),Err(e)=>{eprintln!("ERROR D6.16-JSON-001 {path}: {e}");process::exit(2);}},
        Err(e)=>{eprintln!("ERROR D6.16-ADAPTIVE-001 {path}: {e}");process::exit(1);}}
}
