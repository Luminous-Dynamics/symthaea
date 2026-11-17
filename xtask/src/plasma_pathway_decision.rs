use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MIN_DRAWS: u32 = 10_000;
const OBJECTIVES: [&str; 3] = ["cost", "grid_energy", "ghg"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    decision_id: String,
    evidence_class: String,
    seed: u64,
    draws: u32,
    min_feasibility: f64,
    groups: Vec<Group>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    group_id: String,
    service_status: String,
    functional_unit: String,
    product_contract: String,
    feedstock: String,
    boundary: String,
    currency: String,
    credit_policy: String,
    routes: Vec<Route>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Route {
    route_id: String,
    technology: String,
    evidence_class: String,
    functional_unit: String,
    product_contract: String,
    feedstock: String,
    boundary: String,
    currency: String,
    credit_policy: String,
    metrics: Vec<Band>,
    constraints: Vec<Limit>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Band {
    id: String,
    unit: String,
    low: f64,
    mode: f64,
    high: f64,
    pairing: String,
    key: String,
}
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Limit {
    metric: String,
    op: String,
    threshold: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct DecisionReport {
    pub decision_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub draws: u32,
    pub seed: u64,
    pub groups: Vec<GroupReport>,
}
#[derive(Debug, Clone, Serialize)]
pub struct GroupReport {
    pub group_id: String,
    pub disposition: String,
    pub reasons: Vec<String>,
    pub routes: Vec<RouteReport>,
    pub pairwise_dominance: Vec<PairReport>,
}
#[derive(Debug, Clone, Serialize)]
pub struct RouteReport {
    pub route_id: String,
    pub technology: String,
    pub feasibility_probability: f64,
    pub feasibility_lower_95: f64,
    pub robustly_feasible: bool,
    pub pareto_membership_probability: Option<f64>,
    pub mean_regret: BTreeMap<String, Option<f64>>,
    pub p95_regret: BTreeMap<String, Option<f64>>,
}
#[derive(Debug, Clone, Serialize)]
pub struct PairReport {
    pub route_a: String,
    pub route_b: String,
    pub a_dominates_b_probability: f64,
    pub b_dominates_a_probability: f64,
    pub jointly_feasible_draws: u32,
}
#[derive(Debug, Clone)]
struct Sample { values: BTreeMap<String, f64>, feasible: bool }

fn nonempty(s: &str) -> bool { !s.trim().is_empty() }
fn hash(s: &str) -> u64 {
    let mut h=0xcbf29ce484222325_u64;
    for b in s.as_bytes(){h^=u64::from(*b);h=h.wrapping_mul(0x100000001b3);}
    h
}
fn mix(mut x:u64)->u64 {
    x=x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z=x; z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb); z^(z>>31)
}
fn uniform(seed:u64, draw:u32, route:&str, band:&Band)->f64 {
    let scope=if band.pairing=="COMMON_EXOGENOUS" {hash(&band.key)} else {hash(&format!("{route}:{}",band.key))};
    (mix(seed ^ u64::from(draw).wrapping_mul(0xd6e8feb86659fd93) ^ scope)>>11) as f64 / ((1_u64<<53) as f64)
}
fn triangular(lo:f64, mode:f64, hi:f64, u:f64)->f64 {
    if (hi-lo).abs()<=f64::EPSILON{return lo;}
    let split=(mode-lo)/(hi-lo);
    if u<split {lo+(u*(hi-lo)*(mode-lo)).sqrt()} else {hi-((1.0-u)*(hi-lo)*(hi-mode)).sqrt()}
}
fn dominates(a:&BTreeMap<String,f64>,b:&BTreeMap<String,f64>)->bool {
    let mut strict=false;
    for m in OBJECTIVES {let x=a[m];let y=b[m];if x>y{return false;}if x<y{strict=true;}}
    strict
}
fn p95(values:&mut [f64])->Option<f64>{
    if values.is_empty(){return None;} values.sort_by(f64::total_cmp);
    values.get(((values.len()-1) as f64*0.95).ceil() as usize).copied()
}
fn valid_group(g:&Group,min_feasibility:f64)->Result<(),Vec<String>>{
    let mut e=Vec::new();
    if g.service_status!="MATCHED"{e.push("service contract is not matched".to_owned());}
    if g.credit_policy!="NO_CREDIT"{e.push("mandatory NO_CREDIT baseline absent".to_owned());}
    if g.routes.len()<2{e.push("at least two routes required".to_owned());}
    if !(min_feasibility>0.0&&min_feasibility<=1.0){e.push("invalid feasibility threshold".to_owned());}
    let mut ids=BTreeSet::new();let mut plasma=0;let mut incumbent=0;
    let mut ref_metrics:Option<BTreeSet<String>>=None;
    let mut ref_meta:BTreeMap<String,(String,String,String)>=BTreeMap::new();
    let mut ref_limits:Option<Vec<Limit>>=None;
    for r in &g.routes{
        if !nonempty(&r.route_id)||!ids.insert(r.route_id.as_str()){e.push("route IDs must be unique/non-empty".to_owned());}
        if r.technology=="PLASMA"{plasma+=1;}else{incumbent+=1;}
        if r.evidence_class!="SYNTHETIC"{e.push(format!("{} is not synthetic-only evidence",r.route_id));}
        if r.credit_policy!="NO_CREDIT"{e.push(format!("{} lacks no-credit policy",r.route_id));}
        if r.functional_unit!=g.functional_unit||r.product_contract!=g.product_contract||r.feedstock!=g.feedstock||r.boundary!=g.boundary||r.currency!=g.currency{
            e.push(format!("{} has mismatched product/feed/boundary/currency contract",r.route_id));
        }
        let mut metrics=BTreeSet::new();
        for b in &r.metrics{
            if !nonempty(&b.id)||!nonempty(&b.unit)||!nonempty(&b.key)||!b.low.is_finite()||!b.mode.is_finite()||!b.high.is_finite()||b.low>b.mode||b.mode>b.high||!matches!(b.pairing.as_str(),"COMMON_EXOGENOUS"|"ROUTE_SPECIFIC"){
                e.push(format!("{} has invalid uncertainty band {}",r.route_id,b.id));
            }
            if !metrics.insert(b.id.clone()){e.push(format!("{} duplicates metric {}",r.route_id,b.id));}
            let meta=(b.unit.clone(),b.pairing.clone(),b.key.clone());
            if let Some(old)=ref_meta.get(&b.id){if old!=&meta{e.push(format!("{} breaks paired scenario/unit for {}",r.route_id,b.id));}}
            else{ref_meta.insert(b.id.clone(),meta);}
        }
        if let Some(old)=&ref_metrics{if old!=&metrics{e.push(format!("{} metric set differs",r.route_id));}}else{ref_metrics=Some(metrics);}
        let mut limit_ids=BTreeSet::new();
        for l in &r.constraints{
            if !limit_ids.insert(l.metric.clone())||!matches!(l.op.as_str(),"MIN"|"MAX")||!l.threshold.is_finite()||!r.metrics.iter().any(|b|b.id==l.metric){
                e.push(format!("{} has invalid hard constraint {}",r.route_id,l.metric));
            }
        }
        if let Some(old)=&ref_limits{if old!=&r.constraints{e.push(format!("{} has different product limits",r.route_id));}}else{ref_limits=Some(r.constraints.clone());}
    }
    if plasma==0||incumbent==0{e.push("include a plasma route and a non-plasma incumbent".to_owned());}
    for m in OBJECTIVES{if !ref_metrics.as_ref().is_some_and(|s|s.contains(m)){e.push(format!("missing Pareto objective {m}"));}}
    if e.is_empty(){Ok(())}else{Err(e)}
}
fn wilson_lower_95(successes: u32, trials: u32) -> f64 {
    if trials == 0 {
        return 0.0;
    }
    let n = f64::from(trials);
    let p = f64::from(successes.min(trials)) / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt());
    ((center - radius) / denominator).clamp(0.0, 1.0)
}

fn constraint_passes(limit:&Limit,values:&BTreeMap<String,f64>)->bool{
    match (limit.op.as_str(),values.get(&limit.metric)){
        ("MIN",Some(v))=>*v>=limit.threshold,("MAX",Some(v))=>*v<=limit.threshold,_=>false
    }
}
fn evaluate_group(g:&Group,m:&Manifest)->GroupReport{
    if g.service_status!="MATCHED"{
        return GroupReport{group_id:g.group_id.clone(),disposition:"ABSTAIN_NOT_COMPARABLE".to_owned(),reasons:vec!["routes lack a matched service contract; no ranking permitted".to_owned()],routes:vec![],pairwise_dominance:vec![]};
    }
    if let Err(reasons)=valid_group(g,m.min_feasibility){
        return GroupReport{group_id:g.group_id.clone(),disposition:"BLOCKED".to_owned(),reasons,routes:vec![],pairwise_dominance:vec![]};
    }
    let n=g.routes.len();let mut draws:Vec<Vec<Sample>>=(0..n).map(|_|Vec::with_capacity(m.draws as usize)).collect();let mut ok=vec![0_u32;n];
    for d in 0..m.draws{
        for (i,r) in g.routes.iter().enumerate(){
            let values=r.metrics.iter().map(|b|(b.id.clone(),triangular(b.low,b.mode,b.high,uniform(m.seed,d,&r.route_id,b)))).collect::<BTreeMap<_,_>>();
            let feasible=r.constraints.iter().all(|c|constraint_passes(c,&values));if feasible{ok[i]+=1;}
            draws[i].push(Sample{values,feasible});
        }
    }
    let probs:Vec<f64>=ok.iter().map(|x|f64::from(*x)/f64::from(m.draws)).collect();
    let lower_95:Vec<f64>=ok.iter().map(|x|wilson_lower_95(*x,m.draws)).collect();
    let robust:Vec<bool>=lower_95.iter().map(|lower|*lower>=m.min_feasibility).collect();
    let qualified:Vec<usize>=robust.iter().enumerate().filter_map(|(i,x)|if *x{Some(i)}else{None}).collect();
    if qualified.len()<2{
        let routes=g.routes.iter().enumerate().map(|(i,r)|{
            let empty: BTreeMap<String, Option<f64>> = OBJECTIVES.iter().map(|x|(x.to_string(),None)).collect();
            RouteReport{route_id:r.route_id.clone(),technology:r.technology.clone(),feasibility_probability:probs[i],feasibility_lower_95:lower_95[i],robustly_feasible:robust[i],pareto_membership_probability:None,mean_regret:empty.clone(),p95_regret:empty}
        }).collect();
        return GroupReport{group_id:g.group_id.clone(),disposition:"ABSTAIN_INSUFFICIENT_FEASIBLE_ROUTES".to_owned(),reasons:vec!["fewer than two routes satisfy all hard constraints at the required probability".to_owned()],routes,pairwise_dominance:vec![]};
    }
    let mut pareto=vec![0_u32;n];
    let mut regrets:Vec<BTreeMap<String,Vec<f64>>>=(0..n).map(|_|OBJECTIVES.iter().map(|x|(x.to_string(),Vec::new())).collect()).collect();
    let mut pairs:BTreeMap<(usize,usize),(u32,u32,u32)>=BTreeMap::new();
    let mut route_iters: Vec<_> = draws.iter().map(|route_draws| route_draws.iter()).collect();
    for _ in 0..m.draws {
        let current: Vec<&Sample> = route_iters
            .iter_mut()
            .map(|iter| iter.next().expect("one sample per route and draw"))
            .collect();
        let feasible: Vec<usize> = qualified
            .iter()
            .copied()
            .filter(|i| current[*i].feasible)
            .collect();

        for i in &feasible {
            if !feasible.iter().any(|j| {
                j != i && dominates(&current[*j].values, &current[*i].values)
            }) {
                pareto[*i] += 1;
            }
        }

        for i in &feasible {
            for metric in OBJECTIVES {
                let best = feasible
                    .iter()
                    .map(|j| current[*j].values[metric])
                    .fold(f64::INFINITY, f64::min);
                regrets[*i]
                    .get_mut(metric)
                    .unwrap()
                    .push((current[*i].values[metric] - best).max(0.0));
            }
        }

        for p in 0..feasible.len() {
            for q in (p + 1)..feasible.len() {
                let a = feasible[p];
                let b = feasible[q];
                let counts = pairs.entry((a, b)).or_insert((0, 0, 0));
                counts.0 += 1;
                if dominates(&current[a].values, &current[b].values) {
                    counts.1 += 1;
                } else if dominates(&current[b].values, &current[a].values) {
                    counts.2 += 1;
                }
            }
        }
    }
    let routes=g.routes.iter().enumerate().map(|(i,r)|{
        let mut mean=BTreeMap::new();let mut tail=BTreeMap::new();
        for metric in OBJECTIVES{let values=regrets[i].get_mut(metric).unwrap();mean.insert(metric.to_owned(),if values.is_empty(){None}else{Some(values.iter().sum::<f64>()/values.len() as f64)});tail.insert(metric.to_owned(),p95(values));}
        RouteReport{route_id:r.route_id.clone(),technology:r.technology.clone(),feasibility_probability:probs[i],feasibility_lower_95:lower_95[i],robustly_feasible:robust[i],pareto_membership_probability:if robust[i]{Some(f64::from(pareto[i])/f64::from(m.draws))}else{None},mean_regret:mean,p95_regret:tail}
    }).collect();
    let pairwise_dominance=pairs.into_iter().map(|((a,b),(joint,aw,bw))|PairReport{
        route_a:g.routes[a].route_id.clone(),route_b:g.routes[b].route_id.clone(),
        a_dominates_b_probability:f64::from(aw)/f64::from(joint),b_dominates_a_probability:f64::from(bw)/f64::from(joint),jointly_feasible_draws:joint
    }).collect();
    GroupReport{group_id:g.group_id.clone(),disposition:"SYNTHETIC_PAIRED_PARETO_ANALYSIS".to_owned(),reasons:vec!["synthetic algorithm exercise only; no physical or commercial claim".to_owned(),"Pareto probability, pairwise dominance and per-metric regret are separate; no scalar score".to_owned()],routes,pairwise_dominance}
}

pub fn evaluate_decision_manifest_json(source:&str)->Result<DecisionReport,String>{
    let m:Manifest=serde_json::from_str(source).map_err(|e|format!("invalid decision manifest: {e}"))?;
    if m.schema_version!="luminous.plasma-pathway-decision.v0"{return Err("unsupported decision schema".into());}
    if !nonempty(&m.decision_id)||m.evidence_class!="SYNTHETIC"{return Err("v0 engine accepts synthetic manifests only".into());}
    if m.draws<MIN_DRAWS{return Err(format!("at least {MIN_DRAWS} paired draws required"));}
    if m.groups.is_empty(){return Err("no decision groups".into());}
    let mut ids=BTreeSet::new();if m.groups.iter().any(|g|!nonempty(&g.group_id)||!ids.insert(g.group_id.as_str())){return Err("group IDs must be unique and non-empty".into());}
    let groups=m.groups.iter().map(|g|evaluate_group(g,&m)).collect();
    Ok(DecisionReport{decision_id:m.decision_id,evidence_class:m.evidence_class,claim_ceiling:"SYNTHETIC_ONLY_NO_PHYSICAL_RESULT_OR_ROUTE_SUPERIORITY".into(),draws:m.draws,seed:m.seed,groups})
}

pub fn evaluate_registered_target_json(source:&str)->Result<(String,Vec<(String,String)>),String>{
    let root:serde_json::Value=serde_json::from_str(source).map_err(|e|format!("invalid register: {e}"))?;
    let target=root.get("benchmark_targets").and_then(serde_json::Value::as_array).and_then(|x|x.iter().find(|t|t["target_id"]=="PLASMA-PROCESS-PATHWAY-SELECTION-001")).ok_or("pathway selection target missing")?;
    if target["status"]!="RESEARCH_TARGET_NOT_RESULT"{return Err("target must remain RESEARCH_TARGET_NOT_RESULT".into());}
    let groups=target["candidate_groups"].as_array().ok_or("candidate_groups missing")?;
    let mut results=Vec::new();
    for g in groups{
        let id=g["group_id"].as_str().ok_or("group id missing")?;
        let routes=g["candidate_routes"].as_array().ok_or("candidate routes missing")?;
        let has_p=routes.iter().any(|r|r["technology_class"]=="PLASMA");
        let has_n=routes.iter().any(|r|r["technology_class"]!="PLASMA");
        let status=if routes.len()<2||!has_p||!has_n{"BLOCKED"}else if g["common_service_status"]!="MATCHED"{"ABSTAIN_NOT_COMPARABLE"}else{"READY_FOR_ELIGIBILITY_CHECKS_NOT_A_RESULT"};
        results.push((id.to_owned(),status.to_owned()));
    }
    let overall=if results.iter().any(|x|x.1=="BLOCKED"){"BLOCKED"}else if results.iter().any(|x|x.1.starts_with("READY")){"READY_FOR_ELIGIBILITY_CHECKS_NOT_A_RESULT"}else{"ABSTAIN_ALL_GROUPS_NOT_COMPARABLE"};
    Ok((overall.to_owned(),results))
}

#[cfg(test)]
mod tests{
    use super::*;
    use serde_json::Value;
    const FIXTURE:&str=include_str!("../plasma-audit-harness/fixtures/pathway-selection-synthetic-v0.json");
    const SOURCE:&str=include_str!("../../data/research/plasma_energy_reference_cases_v0_1.json");
    fn fixture()->Value{serde_json::from_str(FIXTURE).unwrap()}

    #[test] fn synthetic_fixture_executes_10000_paired_draws(){
        let r=evaluate_decision_manifest_json(FIXTURE).unwrap();assert_eq!(r.draws,10000);assert_eq!(r.groups[0].disposition,"SYNTHETIC_PAIRED_PARETO_ANALYSIS");assert_eq!(r.groups[0].routes.len(),3);assert_eq!(r.groups[0].pairwise_dominance.len(),3);
    }
    #[test] fn fixed_seed_is_reproducible(){
        let a=serde_json::to_string(&evaluate_decision_manifest_json(FIXTURE).unwrap()).unwrap();let b=serde_json::to_string(&evaluate_decision_manifest_json(FIXTURE).unwrap()).unwrap();assert_eq!(a,b);
    }
    #[test] fn actual_source_candidates_abstain_until_matched(){
        let (overall,groups)=evaluate_registered_target_json(SOURCE).unwrap();assert_eq!(overall,"ABSTAIN_ALL_GROUPS_NOT_COMPARABLE");assert_eq!(groups.len(),3);assert!(groups.iter().all(|x|x.1=="ABSTAIN_NOT_COMPARABLE"));
    }
    #[test] fn missing_non_plasma_baseline_is_blocked(){
        let mut v=fixture();v["groups"][0]["routes"].as_array_mut().unwrap().retain(|r|r["technology"]=="PLASMA");let r=evaluate_decision_manifest_json(&v.to_string()).unwrap();assert_eq!(r.groups[0].disposition,"BLOCKED");
    }
    #[test] fn mismatched_functional_unit_is_blocked(){
        let mut v=fixture();v["groups"][0]["routes"][1]["functional_unit"]=serde_json::json!("unpurified H2");let r=evaluate_decision_manifest_json(&v.to_string()).unwrap();assert_eq!(r.groups[0].disposition,"BLOCKED");
    }
    #[test] fn changed_pairing_key_is_blocked(){
        let mut v=fixture();v["groups"][0]["routes"][1]["metrics"][0]["key"]=serde_json::json!("different-electricity-scenario");let r=evaluate_decision_manifest_json(&v.to_string()).unwrap();assert_eq!(r.groups[0].disposition,"BLOCKED");
    }
    #[test] fn product_quality_constraint_is_non_compensatory(){
        let mut v=fixture();for r in v["groups"][0]["routes"].as_array_mut().unwrap(){r["metrics"][3]["low"]=serde_json::json!(0.90);r["metrics"][3]["mode"]=serde_json::json!(0.91);r["metrics"][3]["high"]=serde_json::json!(0.92);}
        let r=evaluate_decision_manifest_json(&v.to_string()).unwrap();assert_eq!(r.groups[0].disposition,"ABSTAIN_INSUFFICIENT_FEASIBLE_ROUTES");
    }
    #[test] fn no_coproduct_credit_is_mandatory(){
        let mut v=fixture();v["groups"][0]["routes"][0]["credit_policy"]=serde_json::json!("AVOIDED_PRODUCT_CREDIT");let r=evaluate_decision_manifest_json(&v.to_string()).unwrap();assert_eq!(r.groups[0].disposition,"BLOCKED");
    }
    #[test]
    fn wilson_lower_bound_is_conservative_at_the_feasibility_threshold() {
        let lower = wilson_lower_95(9_900, 10_000);
        assert!(lower < 0.99);
        assert!(wilson_lower_95(10_000, 10_000) > 0.99);
    }

    #[test]
    fn wilson_lower_bound_handles_zero_trials_and_zero_successes() {
        assert_eq!(wilson_lower_95(0, 0), 0.0);
        assert!(wilson_lower_95(0, 10_000) < 0.001);
    }

    #[test]
    fn route_report_exposes_point_estimate_and_confidence_lower_bound() {
        let report = evaluate_decision_manifest_json(FIXTURE).unwrap();
        let route = &report.groups[0].routes[0];
        assert!(route.feasibility_lower_95 <= route.feasibility_probability);
        assert!(route.feasibility_lower_95 >= 0.0);
    }

    #[test] fn less_than_10000_draws_is_rejected(){
        let mut v=fixture();v["draws"]=serde_json::json!(9999);assert!(evaluate_decision_manifest_json(&v.to_string()).is_err());
    }
}
