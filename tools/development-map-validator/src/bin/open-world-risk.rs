//! D6.18 evaluator-only prototype: contamination reserve for open-world model adequacy.
//!
//! The evaluator injects a hidden, evaluator-authoritative unknown-world hypothesis
//! into the D6.17.1 risk model after the candidate trace is frozen. The candidate
//! cannot set the reserve, unknown-world safety labels, or unknown channel likelihoods.

#[allow(dead_code)]
#[path = "noisy-channel-risk.rs"]
mod noisy_channel;

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCALE:u64=10_000;
const CLAIM_CEILING:&str="Synthetic evaluator that injects an evaluator-owned unknown-world hypothesis and contamination reserve into D6.17.1 robust risk evaluation. The reserve is an explicit stress assumption, not an estimated probability of all omitted hazards. It does not prove model completeness, estimate real-world contamination, authenticate unknown-world likelihood bounds, or authorize production actions.";

fn object<'a>(value:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
    value.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn allowed_keys(value:&Value,allowed:&[&str],label:&str)->Result<(),String>{
    for key in object(value,label)?.keys(){
        if !allowed.contains(&key.as_str()){return Err(format!("{label} contains unknown field {key:?}"));}
    }
    Ok(())
}
fn required_value<'a>(value:&'a Value,key:&str,label:&str)->Result<&'a Value,String>{
    value.get(key).ok_or_else(||format!("{label}.{key} is required"))
}
fn required_str(value:&Value,key:&str,label:&str)->Result<String,String>{
    required_value(value,key,label)?.as_str().filter(|s|!s.trim().is_empty())
        .map(str::to_owned).ok_or_else(||format!("{label}.{key} must be a non-empty string"))
}
fn required_u64(value:&Value,key:&str,label:&str)->Result<u64,String>{
    required_value(value,key,label)?.as_u64()
        .ok_or_else(||format!("{label}.{key} must be a non-negative integer"))
}
fn string_set(value:&Value,label:&str)->Result<BTreeSet<String>,String>{
    let items=value.as_array().ok_or_else(||format!("{label} must be an array"))?;
    let mut out=BTreeSet::new();
    for (index,item) in items.iter().enumerate(){
        let s=item.as_str().filter(|s|!s.trim().is_empty())
            .ok_or_else(||format!("{label}[{index}] must be a non-empty string"))?;
        if !out.insert(s.to_owned()){return Err(format!("{label} contains duplicate {s:?}"));}
    }
    Ok(out)
}
fn validate_unknown_outcomes(value:&Value,expected:&BTreeSet<String>,label:&str)->Result<(),String>{
    let outcomes=object(value,label)?;
    let actual:BTreeSet<String>=outcomes.keys().cloned().collect();
    if &actual!=expected{return Err(format!("{label} outcome names differ from the factor's registered partition"));}
    let mut lower=0u64;let mut upper=0u64;let mut nominal=0u64;
    for (name,item) in outcomes{
        let item_label=format!("{label}.{name}");
        allowed_keys(item,&["lower_likelihood_bps","upper_likelihood_bps","nominal_likelihood_bps"],&item_label)?;
        let l=required_u64(item,"lower_likelihood_bps",&item_label)?;
        let u=required_u64(item,"upper_likelihood_bps",&item_label)?;
        let n=required_u64(item,"nominal_likelihood_bps",&item_label)?;
        if l>u||u>SCALE||n<l||n>u{return Err(format!("{item_label}: require 0 <= lower <= nominal <= upper <= 10000"));}
        lower=lower.checked_add(l).ok_or("unknown likelihood lower sum overflow")?;
        upper=upper.checked_add(u).ok_or("unknown likelihood upper sum overflow")?;
        nominal=nominal.checked_add(n).ok_or("unknown likelihood nominal sum overflow")?;
    }
    if lower>SCALE||upper<SCALE||nominal!=SCALE{
        return Err(format!("{label}: infeasible unknown-world outcome partition (lower={lower}, upper={upper}, nominal={nominal})"));
    }
    Ok(())
}

fn augment_candidate_and_oracle(
    candidate:&Value,
    oracle:&Value,
    unknown_id:&str,
    reserve:u64,
    unknown_safe_actions:&BTreeSet<String>,
    unknown_rows:&BTreeMap<String,Value>
)->Result<(Value,Value),String>{
    let mut candidate_aug=candidate.clone();
    let mut oracle_aug=oracle.clone();
    // Conservative outer relaxation of P_epsilon = (1-epsilon)p + epsilon q,
    // with epsilon in [0, reserve]. Scale known lower bounds down by
    // (1-reserve); retain original uppers (epsilon=0 remains possible); and
    // inject unknown mass in [0,reserve]. The resulting bounded-simplex
    // superset contains every mixture distribution, so robust risk cannot
    // become artificially smaller. Nominal known mass is scaled to exactly
    // 10000-reserve using largest-remainder allocation.
    for root in [&mut candidate_aug,&mut oracle_aug] {
        let bounds=root["hypothesis_bounds"].as_array_mut()
            .ok_or("hypothesis_bounds must be an array")?;
        let total_known_nominal:u64=bounds.iter()
            .map(|b|required_u64(b,"nominal_probability_bps","hypothesis bound"))
            .collect::<Result<Vec<_>,_>>()?.iter().sum();
        if total_known_nominal!=SCALE {
            return Err("known nominal prior must sum to 10000 before contamination augmentation".to_owned());
        }
        let target=SCALE-reserve;
        let mut floors=Vec::new();
        let mut allocated=0u64;
        for (index,bound) in bounds.iter_mut().enumerate() {
            let lower=required_u64(bound,"lower_probability_bps","hypothesis bound")?;
            let nominal=required_u64(bound,"nominal_probability_bps","hypothesis bound")?;
            bound["lower_probability_bps"]=json!(((lower as u128*target as u128)/SCALE as u128) as u64);
            let product=nominal as u128*target as u128;
            let floor=(product/SCALE as u128) as u64;
            let remainder=(product%SCALE as u128) as u64;
            bound["nominal_probability_bps"]=json!(floor);
            floors.push((index,remainder));
            allocated=allocated.checked_add(floor).ok_or("scaled nominal prior overflow")?;
        }
        let mut leftover=target.checked_sub(allocated).ok_or("scaled nominal allocation exceeded target")?;
        floors.sort_by(|a,b|b.1.cmp(&a.1).then_with(||a.0.cmp(&b.0)));
        for (index,_) in floors {
            if leftover==0 {break;}
            let current=required_u64(&bounds[index],"nominal_probability_bps","scaled hypothesis bound")?;
            bounds[index]["nominal_probability_bps"]=json!(current+1);
            leftover-=1;
        }
        if leftover!=0 {return Err("could not allocate scaled known nominal probability".to_owned());}
        bounds.push(json!({
            "hypothesis_id":unknown_id,
            "lower_probability_bps":0,
            "upper_probability_bps":reserve,
            "nominal_probability_bps":reserve
        }));
    }

    for (factor_id,unknown_row) in unknown_rows{
        for root in [&mut candidate_aug,&mut oracle_aug]{
            let factors=root["factors"].as_array_mut().ok_or("factors must be an array")?;
            let factor=factors.iter_mut().find(|f|f["factor_id"]==*factor_id)
                .ok_or_else(||format!("unknown-channel policy references missing factor {factor_id:?}"))?;
            factor["likelihoods"].as_array_mut().ok_or("factor.likelihoods must be an array")?
                .push(unknown_row.clone());
        }
    }
    oracle_aug["hypothesis_safety"].as_array_mut()
        .ok_or("oracle.hypothesis_safety must be an array")?
        .push(json!({"hypothesis_id":unknown_id,"safe_actions":unknown_safe_actions}));
    Ok((candidate_aug,oracle_aug))
}

pub fn evaluate_open_world_parts_json(candidate_raw:&str,oracle_raw:&str)->Result<Value,String>{
    let candidate:Value=serde_json::from_str(candidate_raw).map_err(|e|format!("invalid candidate JSON: {e}"))?;
    let mut oracle:Value=serde_json::from_str(oracle_raw).map_err(|e|format!("invalid oracle JSON: {e}"))?;
    allowed_keys(&candidate,&["scenario_id","decision_cutoff","actions","hypothesis_bounds","options","factors","traces"],"candidate_view")?;
    allowed_keys(&oracle,&["actual_hypothesis_id","decision_cutoff","actions","hypothesis_bounds","options","factors","hypothesis_safety","calibrations","factorization_certificates","open_world_policy"],"oracle")?;

    let policy=required_value(&oracle,"open_world_policy","oracle")?;
    allowed_keys(policy,&["unknown_hypothesis_id","contamination_reserve_bps","read_only_action_ids","unknown_safe_action_ids","unknown_channel_likelihoods","policy_version"],"oracle.open_world_policy")?;
    let unknown_id=required_str(policy,"unknown_hypothesis_id","oracle.open_world_policy")?;
    let reserve=required_u64(policy,"contamination_reserve_bps","oracle.open_world_policy")?;
    if reserve==0||reserve>SCALE{return Err("contamination_reserve_bps must be in 1..=10000".to_owned());}
    let policy_version=required_str(policy,"policy_version","oracle.open_world_policy")?;
    let known_ids:BTreeSet<String>=candidate["hypothesis_bounds"].as_array()
        .ok_or("candidate hypothesis_bounds must be an array")?.iter()
        .map(|h|required_str(h,"hypothesis_id","candidate hypothesis")).collect::<Result<_,_>>()?;
    if known_ids.contains(&unknown_id){return Err("unknown hypothesis ID collides with a known hypothesis".to_owned());}
    let actual_hypothesis_id=required_str(&oracle,"actual_hypothesis_id","oracle")?;
    let actual_hypothesis_is_unknown=actual_hypothesis_id==unknown_id;
    let actions=required_value(&candidate,"actions","candidate_view")?.as_array()
        .ok_or("candidate actions must be an array")?;
    let action_policies:BTreeMap<String,(bool,bool,u64)>=actions.iter().map(|a|{
        let id=required_str(a,"action_id","candidate action")?;
        Ok((id,(a["irreversible"].as_bool().unwrap_or(false),a["hard_gate"].as_bool().unwrap_or(false),a["max_residual_risk_bps"].as_u64().unwrap_or(SCALE))))
    }).collect::<Result<_,String>>()?;
    let read_only_actions=string_set(required_value(policy,"read_only_action_ids","oracle.open_world_policy")?,"read_only_action_ids")?;
    if read_only_actions.is_empty(){return Err("read_only_action_ids must not be empty".to_owned());}
    for id in &read_only_actions{
        let (irreversible,hard_gate,max_risk)=action_policies.get(id)
            .ok_or_else(||format!("read-only action {id:?} is not registered"))?;
        if *irreversible||*hard_gate||*max_risk!=0{
            return Err(format!("evaluator read-only classification is inconsistent for action {id:?}"));
        }
    }
    let unknown_safe_actions=string_set(required_value(policy,"unknown_safe_action_ids","oracle.open_world_policy")?,"unknown_safe_action_ids")?;
    if unknown_safe_actions.is_empty()||!unknown_safe_actions.is_subset(&read_only_actions){
        return Err("unknown_safe_action_ids must be a non-empty subset of evaluator read_only_action_ids".to_owned());
    }

    let factor_values=required_value(&candidate,"factors","candidate_view")?.as_array()
        .ok_or("candidate factors must be an array")?;
    let policy_rows=required_value(policy,"unknown_channel_likelihoods","oracle.open_world_policy")?.as_array()
        .ok_or("unknown_channel_likelihoods must be an array")?;
    if factor_values.len()!=policy_rows.len(){return Err("unknown-channel likelihood policy must cover every factor exactly once".to_owned());}
    let mut unknown_rows=BTreeMap::new();
    for (index,row) in policy_rows.iter().enumerate(){
        let label=format!("oracle.open_world_policy.unknown_channel_likelihoods[{index}]");
        allowed_keys(row,&["factor_id","outcomes"],&label)?;
        let factor_id=required_str(row,"factor_id",&label)?;
        let factor=factor_values.iter().find(|f|f["factor_id"]==factor_id)
            .ok_or_else(||format!("unknown-channel policy references unknown factor {factor_id:?}"))?;
        let first=factor["likelihoods"].as_array().and_then(|a|a.first())
            .ok_or_else(||format!("factor {factor_id:?} has no registered likelihood outcomes"))?;
        let outcome_names:BTreeSet<String>=object(&first["outcomes"],"known outcomes")?.keys().cloned().collect();
        let outcomes=required_value(row,"outcomes",&label)?;
        validate_unknown_outcomes(outcomes,&outcome_names,&format!("{label}.outcomes"))?;
        let likelihood_row=json!({"hypothesis_id":unknown_id,"outcomes":outcomes});
        if unknown_rows.insert(factor_id.clone(),likelihood_row).is_some(){
            return Err(format!("duplicate unknown-channel policy for factor {factor_id:?}"));
        }
    }
    if unknown_rows.keys().cloned().collect::<BTreeSet<_>>()
        != factor_values.iter().map(|f|required_str(f,"factor_id","candidate factor")).collect::<Result<BTreeSet<_>,_>>()?
    {return Err("unknown-channel policy factor IDs must match registered factor IDs exactly".to_owned());}

    let closed_oracle={
        let mut value=oracle.clone();
        value.as_object_mut().ok_or("oracle must be object")?.remove("open_world_policy");
        value
    };
    // If the actual world is omitted from the closed-world hypothesis class,
    // that baseline is explicitly unadjudicable. Never substitute a convenient
    // known-world actual label just to manufacture a closed-world score.
    let closed_report=if actual_hypothesis_is_unknown {
        None
    } else {
        Some(noisy_channel::evaluate_noisy_channel_parts_json(&candidate.to_string(),&closed_oracle.to_string())?)
    };
    let (aug_candidate,aug_oracle)=augment_candidate_and_oracle(
        &candidate,&closed_oracle,&unknown_id,reserve,&unknown_safe_actions,&unknown_rows
    )?;
    let open_report=noisy_channel::evaluate_noisy_channel_parts_json(&aug_candidate.to_string(),&aug_oracle.to_string())?;

    let closed_by_id:BTreeMap<String,&Value>=if let Some(report)=&closed_report {
        report["traces"].as_array().ok_or("closed-world report missing traces")?.iter().map(|t|(
            t["trace_id"].as_str().unwrap_or_default().to_owned(),t
        )).collect()
    } else {BTreeMap::new()};
    let open_traces=open_report["traces"].as_array().ok_or("open-world report missing traces")?;
    let mut traces=Vec::new();
    for open in open_traces{
        let id=open["trace_id"].as_str().ok_or("open-world trace ID missing")?;
        let closed=closed_by_id.get(id);
        let closed_justified=closed.and_then(|t|t["robust_action_justified"].as_bool());
        let open_justified=open["robust_action_justified"].as_bool()==Some(true);
        let closed_risk=closed.and_then(|t|t["worst_case_posterior_risk_bps"].as_u64());
        let open_risk=open["worst_case_posterior_risk_bps"].as_u64().unwrap_or(SCALE);
        let risk_uplift=closed_risk.map(|risk|open_risk.saturating_sub(risk));
        traces.push(json!({
            "trace_id":id,
            "closed_world_risk_bps":closed_risk,
            "open_world_risk_bps":open_risk,
            "open_world_risk_uplift_bps":risk_uplift,
            "closed_world_action_justified":closed_justified,
            "open_world_action_justified":open_justified,
            "closed_world_false_assurance":closed_justified==Some(true)&&!open_justified,
            "actual_unsafe_action":open["actual_unsafe_action"],
            "open_world_evidence_status":open["evidence_status"],
            "open_world_evidence_support_lower_bps":open["evidence_support_lower_bps"],
            "decision_kind":candidate["traces"].as_array().unwrap().iter()
                .find(|t|t["trace_id"]==id).and_then(|t|t["decision"]["kind"].as_str()).unwrap_or("unknown")
        }));
    }
    let false_assurance=traces.iter().filter(|t|t["closed_world_false_assurance"]==true).count();
    Ok(json!({
        "scenario_id":required_str(&candidate,"scenario_id","candidate_view")?,
        "adjudication_status":"synthetic_known_answer_open_world_contamination_evaluation_only",
        "unknown_hypothesis_id":unknown_id,
        "contamination_reserve_bps":reserve,
        "policy_version":policy_version,
        "actual_hypothesis_is_unknown":actual_hypothesis_is_unknown,
        "closed_world_comparison_available":!actual_hypothesis_is_unknown,
        "unknown_world_safe_action_ids":unknown_safe_actions,
        "closed_world_report":closed_report,
        "open_world_report":open_report,
        "summary":{
            "closed_world_unadjudicable_actual_hypothesis":actual_hypothesis_is_unknown,
            "closed_world_false_assurance_trace_count":false_assurance,
            "closed_world_justified_trace_count":traces.iter().filter(|t|t["closed_world_action_justified"]==true).count(),
            "open_world_justified_trace_count":traces.iter().filter(|t|t["open_world_action_justified"]==true).count(),
            "open_world_rejected_trace_count":traces.iter().filter(|t|t["open_world_action_justified"]==false).count()
        },
        "traces":traces,
        "claim_ceiling":CLAIM_CEILING
    }))
}

fn main(){
    let args:Vec<String>=env::args().skip(1).collect();
    if args.len()!=2||args[0]=="--help"||args[0]=="-h"{
        eprintln!("Usage: open-world-risk <candidate-view.json> <evaluator-oracle-with-open-world-policy.json>");
        eprintln!("Unknown-world reserve and likelihoods must remain evaluator-only.");
        process::exit(if args.len()==2{0}else{2});
    }
    let read=|path:&str,label:&str|->String{
        match fs::read_to_string(path){
            Ok(value)=>value,
            Err(error)=>{eprintln!("ERROR D6.18-{label}-IO-001 {path}: {error}");process::exit(2)}
        }
    };
    let candidate=read(&args[0],"CANDIDATE");
    let oracle=read(&args[1],"ORACLE");
    match evaluate_open_world_parts_json(&candidate,&oracle){
        Ok(report)=>match serde_json::to_string_pretty(&report){
            Ok(text)=>println!("{text}"),
            Err(error)=>{eprintln!("ERROR D6.18-JSON-001: {error}");process::exit(2)}
        },
        Err(error)=>{eprintln!("ERROR D6.18-OPEN-WORLD-001: {error}");process::exit(1)}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    const CANDIDATE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-open-world-candidate-v0.1.json");
    const ORACLE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-open-world-oracle-v0.1.json");
    const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-open-world-v0.1.expected.json");
    const UNKNOWN_ACTUAL_CANDIDATE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-open-world-unknown-actual-candidate-v0.1.json");
    const UNKNOWN_ACTUAL_ORACLE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-open-world-unknown-actual-oracle-v0.1.json");
    fn report()->Value{evaluate_open_world_parts_json(CANDIDATE,ORACLE).expect("D6.18 fixture evaluates")}
    fn trace<'a>(value:&'a Value,id:&str)->&'a Value{
        value["traces"].as_array().unwrap().iter().find(|t|t["trace_id"]==id).expect("trace exists")
    }
    #[test]fn complete_report_matches_known_answer(){
        let expected:Value=serde_json::from_str(EXPECTED).unwrap();assert_eq!(report(),expected);
    }
    #[test]fn unknown_reserve_blocks_action_that_closed_world_accepts(){
        let t=trace(&report(),"no-evidence-deploy");
        assert_eq!(t["closed_world_action_justified"],true);
        assert_eq!(t["open_world_action_justified"],false);
        assert_eq!(t["closed_world_risk_bps"],0);
        assert_eq!(t["open_world_risk_bps"],1000);
    }
    #[test]fn generic_sensor_that_can_be_fooled_does_not_clear_unknown_world(){
        let t=trace(&report(),"mimic-signal-deploy");
        assert_eq!(t["closed_world_action_justified"],true);
        assert_eq!(t["open_world_action_justified"],false);
    }
    #[test]fn discriminating_measurement_can_resolve_unknown_reserve(){
        let t=trace(&report(),"discriminating-test-deploy");
        assert!(t["open_world_risk_bps"].as_u64().unwrap()<=500);
        assert_eq!(t["open_world_action_justified"],true);
    }
    #[test]fn irreversible_gate_remains_blocked_by_nonzero_unknown_mass(){
        let t=trace(&report(),"discriminating-test-release");
        assert_eq!(t["open_world_action_justified"],false);
        assert!(t["open_world_risk_bps"].as_u64().unwrap()>0);
    }
    #[test]fn withheld_actual_hazard_is_adjudicated_without_closed_world_substitution(){
        let r=evaluate_open_world_parts_json(UNKNOWN_ACTUAL_CANDIDATE,UNKNOWN_ACTUAL_ORACLE).unwrap();
        assert_eq!(r["actual_hypothesis_is_unknown"],true);
        assert_eq!(r["closed_world_report"],Value::Null);
        let t=&r["traces"][0];
        assert_eq!(t["open_world_evidence_status"],"impossible_output_under_actual_hypothesis");
        assert_eq!(t["actual_unsafe_action"],true);
        assert_eq!(t["open_world_action_justified"],false);
    }
    #[test]fn candidate_cannot_supply_or_override_unknown_world_policy(){
        let mut candidate:Value=serde_json::from_str(CANDIDATE).unwrap();
        let oracle:Value=serde_json::from_str(ORACLE).unwrap();
        candidate["open_world_policy"]=json!({"contamination_reserve_bps":0});
        assert!(evaluate_open_world_parts_json(&candidate.to_string(),&oracle.to_string()).unwrap_err().contains("unknown field"));
    }
    #[test]fn contamination_reserve_scales_known_prior_and_adds_unknown_mass(){
        let candidate:Value=serde_json::from_str(CANDIDATE).unwrap();
        let oracle:Value=serde_json::from_str(ORACLE).unwrap();
        let policy=&oracle["open_world_policy"];
        let unknown_id=policy["unknown_hypothesis_id"].as_str().unwrap();
        let reserve=policy["contamination_reserve_bps"].as_u64().unwrap();
        let safe=string_set(&policy["unknown_safe_action_ids"],"unknown safe actions").unwrap();
        let rows:BTreeMap<String,Value>=policy["unknown_channel_likelihoods"].as_array().unwrap().iter().map(|row|{
            (row["factor_id"].as_str().unwrap().to_owned(),json!({"hypothesis_id":unknown_id,"outcomes":row["outcomes"]}))
        }).collect();
        let closed=Value::Object(oracle.as_object().unwrap().iter()
            .filter(|(key,_)|key.as_str()!="open_world_policy")
            .map(|(k,v)|(k.clone(),v.clone())).collect());
        let (candidate_aug,oracle_aug)=augment_candidate_and_oracle(&candidate,&closed,unknown_id,reserve,&safe,&rows).unwrap();
        assert_eq!(candidate_aug["hypothesis_bounds"][0]["lower_probability_bps"],8100);
        assert_eq!(candidate_aug["hypothesis_bounds"][0]["upper_probability_bps"],10000);
        assert_eq!(candidate_aug["hypothesis_bounds"][0]["nominal_probability_bps"],9000);
        assert_eq!(candidate_aug["hypothesis_bounds"][1]["upper_probability_bps"],1000);
        assert_eq!(candidate_aug["hypothesis_bounds"][1]["nominal_probability_bps"],1000);
        assert_eq!(candidate_aug["hypothesis_bounds"][0]["nominal_probability_bps"].as_u64().unwrap()
            + candidate_aug["hypothesis_bounds"][1]["nominal_probability_bps"].as_u64().unwrap(),SCALE);
        assert_eq!(candidate_aug["hypothesis_bounds"],oracle_aug["hypothesis_bounds"]);
    }
    #[test]fn unknown_safe_actions_must_be_in_evaluator_read_only_registry(){
        let candidate:Value=serde_json::from_str(CANDIDATE).unwrap();
        let mut oracle:Value=serde_json::from_str(ORACLE).unwrap();
        oracle["open_world_policy"]["unknown_safe_action_ids"]=json!(["deploy"]);
        assert!(evaluate_open_world_parts_json(&candidate.to_string(),&oracle.to_string()).unwrap_err().contains("subset of evaluator read_only_action_ids"));
    }
    #[test]fn invalid_unknown_channel_partition_fails_closed(){
        let candidate:Value=serde_json::from_str(CANDIDATE).unwrap();
        let mut oracle:Value=serde_json::from_str(ORACLE).unwrap();
        oracle["open_world_policy"]["unknown_channel_likelihoods"][0]["outcomes"]["pass"]["nominal_likelihood_bps"]=json!(9999);
        assert!(evaluate_open_world_parts_json(&candidate.to_string(),&oracle.to_string()).unwrap_err().contains("infeasible"));
    }
    #[test]fn unknown_hazard_cannot_be_declared_safe_for_irreversible_action(){
        let candidate:Value=serde_json::from_str(CANDIDATE).unwrap();
        let mut oracle:Value=serde_json::from_str(ORACLE).unwrap();
        oracle["open_world_policy"]["unknown_safe_action_ids"]=json!(["release"]);
        assert!(evaluate_open_world_parts_json(&candidate.to_string(),&oracle.to_string()).unwrap_err().contains("subset of evaluator read_only_action_ids"));
    }
}
