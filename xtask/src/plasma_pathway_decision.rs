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
pub struct ProbabilityEstimate {
    pub successes: u32,
    pub trials: u32,
    pub probability: f64,
    pub monte_carlo_standard_error: f64,
    pub ci95_lower: f64,
    pub ci95_upper: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineReport {
    pub method: String,
    pub ranked_route_ids: Vec<String>,
    pub scores: BTreeMap<String, f64>,
    pub eligible_route_count: usize,
    pub diagnostic_only: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupReport {
    pub group_id: String,
    pub disposition: String,
    pub reasons: Vec<String>,
    pub routes: Vec<RouteReport>,
    pub pairwise_dominance: Vec<PairReport>,
    pub diagnostic_baselines: Vec<BaselineReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteReport {
    pub route_id: String,
    pub technology: String,
    pub feasibility_probability: f64,
    pub feasibility_estimate: ProbabilityEstimate,
    pub robustly_feasible: bool,
    pub pareto_membership_probability: Option<f64>,
    pub pareto_membership_estimate: Option<ProbabilityEstimate>,
    pub mean_regret: BTreeMap<String, Option<f64>>,
    pub mean_regret_mcse: BTreeMap<String, Option<f64>>,
    pub p95_regret: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairReport {
    pub route_a: String,
    pub route_b: String,
    pub a_dominates_b_probability: f64,
    pub a_dominates_b_estimate: ProbabilityEstimate,
    pub b_dominates_a_probability: f64,
    pub b_dominates_a_estimate: ProbabilityEstimate,
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

fn probability_estimate(successes: u32, trials: u32) -> ProbabilityEstimate {
    if trials == 0 {
        return ProbabilityEstimate {
            successes,
            trials,
            probability: 0.0,
            monte_carlo_standard_error: 0.0,
            ci95_lower: 0.0,
            ci95_upper: 1.0,
        };
    }

    let n = f64::from(trials);
    let p = f64::from(successes.min(trials)) / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z
        * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt()
        / denominator;
    let standard_error = (p * (1.0 - p) / n).sqrt();

    ProbabilityEstimate {
        successes,
        trials,
        probability: p,
        monte_carlo_standard_error: standard_error,
        ci95_lower: (center - half_width).clamp(0.0, 1.0),
        ci95_upper: (center + half_width).clamp(0.0, 1.0),
    }
}

fn mean_and_mcse(values: &[f64]) -> (Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (Some(mean), None);
    }
    let sum_squared_deviation = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let sample_variance = sum_squared_deviation / (values.len() - 1) as f64;
    (Some(mean), Some((sample_variance / values.len() as f64).sqrt()))
}

fn diagnostic_baselines(
    group: &Group,
    draws: &[Vec<Sample>],
    qualified: &[usize],
) -> Vec<BaselineReport> {
    let mut means: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
    for &index in qualified {
        let feasible_values: Vec<&Sample> =
            draws[index].iter().filter(|sample| sample.feasible).collect();
        let mut metrics = BTreeMap::new();
        for metric in OBJECTIVES {
            let values: Vec<f64> = feasible_values
                .iter()
                .filter_map(|sample| sample.values.get(metric).copied())
                .collect();
            if !values.is_empty() {
                metrics.insert(
                    metric.to_owned(),
                    values.iter().sum::<f64>() / values.len() as f64,
                );
            }
        }
        if OBJECTIVES.iter().all(|metric| metrics.contains_key(*metric)) {
            means.insert(index, metrics);
        }
    }

    let mut reports = Vec::new();
    let eligible: Vec<usize> = qualified
        .iter()
        .copied()
        .filter(|index| means.contains_key(index))
        .collect();

    let mut incumbents: Vec<usize> = eligible
        .iter()
        .copied()
        .filter(|index| group.routes[*index].technology != "PLASMA")
        .collect();
    incumbents.sort_by(|a, b| {
        means[a]["cost"]
            .total_cmp(&means[b]["cost"])
            .then_with(|| group.routes[*a].route_id.cmp(&group.routes[*b].route_id))
    });
    let incumbent_scores = incumbents
        .iter()
        .map(|index| (group.routes[*index].route_id.clone(), means[index]["cost"]))
        .collect();
    reports.push(BaselineReport {
        method: "BEST_QUALIFIED_NON_PLASMA_INCUMBENT_BY_MEAN_COST".to_owned(),
        ranked_route_ids: incumbents
            .iter()
            .map(|index| group.routes[*index].route_id.clone())
            .collect(),
        scores: incumbent_scores,
        eligible_route_count: incumbents.len(),
        diagnostic_only: true,
        note: "Reference-only incumbent ranking among robustly feasible non-plasma routes; it does not select a plasma route or override the Pareto/abstention result.".to_owned(),
    });

    let mut weighted_scores = BTreeMap::new();
    let mut topsis_scores = BTreeMap::new();
    let mut normalized: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();

    for metric in OBJECTIVES {
        let min = eligible
            .iter()
            .map(|index| means[index][metric])
            .fold(f64::INFINITY, f64::min);
        let max = eligible
            .iter()
            .map(|index| means[index][metric])
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        for index in &eligible {
            let value = if range.abs() <= f64::EPSILON {
                0.5
            } else {
                (means[index][metric] - min) / range
            };
            normalized
                .entry(*index)
                .or_default()
                .insert(metric.to_owned(), value);
        }
    }

    for index in &eligible {
        let score = OBJECTIVES
            .iter()
            .map(|metric| normalized[index][*metric])
            .sum::<f64>()
            / OBJECTIVES.len() as f64;
        weighted_scores.insert(group.routes[*index].route_id.clone(), score);
    }

    let mut weighted_ranking = eligible.clone();
    weighted_ranking.sort_by(|a, b| {
        weighted_scores[&group.routes[*a].route_id]
            .total_cmp(&weighted_scores[&group.routes[*b].route_id])
            .then_with(|| group.routes[*a].route_id.cmp(&group.routes[*b].route_id))
    });
    reports.push(BaselineReport {
        method: "FIXED_EQUAL_WEIGHT_NORMALIZED_SUM".to_owned(),
        ranked_route_ids: weighted_ranking
            .iter()
            .map(|index| group.routes[*index].route_id.clone())
            .collect(),
        scores: weighted_scores,
        eligible_route_count: eligible.len(),
        diagnostic_only: true,
        note: "Equal weights are a deliberately simple diagnostic baseline; normalization and weights are not a normative preference model.".to_owned(),
    });

    let norms: BTreeMap<String, f64> = OBJECTIVES
        .iter()
        .map(|metric| {
            let norm = eligible
                .iter()
                .map(|index| means[index][*metric].powi(2))
                .sum::<f64>()
                .sqrt();
            (metric.to_string(), norm)
        })
        .collect();
    let mut normalized_vector: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
    for index in &eligible {
        for metric in OBJECTIVES {
            let norm = norms[metric];
            normalized_vector
                .entry(*index)
                .or_default()
                .insert(
                    metric.to_owned(),
                    if norm <= f64::EPSILON {
                        0.0
                    } else {
                        means[index][metric] / norm
                    },
                );
        }
    }
    let ideals: BTreeMap<String, f64> = OBJECTIVES
        .iter()
        .map(|metric| {
            (
                metric.to_string(),
                eligible
                    .iter()
                    .map(|index| normalized_vector[index][*metric])
                    .fold(f64::INFINITY, f64::min),
            )
        })
        .collect();
    let anti_ideals: BTreeMap<String, f64> = OBJECTIVES
        .iter()
        .map(|metric| {
            (
                metric.to_string(),
                eligible
                    .iter()
                    .map(|index| normalized_vector[index][*metric])
                    .fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect();

    for index in &eligible {
        let distance_to_ideal = OBJECTIVES
            .iter()
            .map(|metric| {
                (normalized_vector[index][*metric] - ideals[*metric]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let distance_to_anti_ideal = OBJECTIVES
            .iter()
            .map(|metric| {
                (normalized_vector[index][*metric] - anti_ideals[*metric]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let denominator = distance_to_ideal + distance_to_anti_ideal;
        let score = if denominator <= f64::EPSILON {
            0.5
        } else {
            distance_to_anti_ideal / denominator
        };
        topsis_scores.insert(group.routes[*index].route_id.clone(), score);
    }
    let mut topsis_ranking = eligible.clone();
    topsis_ranking.sort_by(|a, b| {
        topsis_scores[&group.routes[*b].route_id]
            .total_cmp(&topsis_scores[&group.routes[*a].route_id])
            .then_with(|| group.routes[*a].route_id.cmp(&group.routes[*b].route_id))
    });
    reports.push(BaselineReport {
        method: "TOPSIS_EQUAL_METRIC_DIRECTION".to_owned(),
        ranked_route_ids: topsis_ranking
            .iter()
            .map(|index| group.routes[*index].route_id.clone())
            .collect(),
        scores: topsis_scores,
        eligible_route_count: eligible.len(),
        diagnostic_only: true,
        note: "TOPSIS uses equal metric direction and vector normalization over cost, grid energy and GHG; preference assumptions are not validated.".to_owned(),
    });

    let mut dominator_counts: BTreeMap<usize, usize> =
        eligible.iter().map(|index| (*index, 0)).collect();
    for &candidate in &eligible {
        for &other in &eligible {
            if candidate != other && dominates(&means[&other], &means[&candidate]) {
                *dominator_counts.get_mut(&candidate).unwrap() += 1;
            }
        }
    }
    let mut pareto_ranking = eligible.clone();
    pareto_ranking.sort_by(|a, b| {
        dominator_counts[a]
            .cmp(&dominator_counts[b])
            .then_with(|| group.routes[*a].route_id.cmp(&group.routes[*b].route_id))
    });
    let pareto_scores = dominator_counts
        .iter()
        .map(|(index, count)| (group.routes[*index].route_id.clone(), *count as f64))
        .collect();
    reports.push(BaselineReport {
        method: "CENTRAL_MEAN_NONDOMINATED_SORT".to_owned(),
        ranked_route_ids: pareto_ranking
            .iter()
            .map(|index| group.routes[*index].route_id.clone())
            .collect(),
        scores: pareto_scores,
        eligible_route_count: eligible.len(),
        diagnostic_only: true,
        note: "Deterministic non-dominated sorting on conditional mean metrics; this is not NSGA-II and is not the robust stochastic Pareto result.".to_owned(),
    });

    reports
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
fn constraint_passes(limit:&Limit,values:&BTreeMap<String,f64>)->bool{
    match (limit.op.as_str(),values.get(&limit.metric)){
        ("MIN",Some(v))=>*v>=limit.threshold,("MAX",Some(v))=>*v<=limit.threshold,_=>false
    }
}
fn evaluate_group(g:&Group,m:&Manifest)->GroupReport{
    if g.service_status!="MATCHED"{
        return GroupReport{group_id:g.group_id.clone(),disposition:"ABSTAIN_NOT_COMPARABLE".to_owned(),reasons:vec!["routes lack a matched service contract; no ranking permitted".to_owned()],routes:vec![],pairwise_dominance:vec![],diagnostic_baselines:vec![]};
    }
    if let Err(reasons)=valid_group(g,m.min_feasibility){
        return GroupReport{group_id:g.group_id.clone(),disposition:"BLOCKED".to_owned(),reasons,routes:vec![],pairwise_dominance:vec![],diagnostic_baselines:vec![]};
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
    let robust:Vec<bool>=probs.iter().map(|p|*p>=m.min_feasibility).collect();
    let qualified:Vec<usize>=robust.iter().enumerate().filter_map(|(i,x)|if *x{Some(i)}else{None}).collect();
    if qualified.len()<2{
        let routes=g.routes.iter().enumerate().map(|(i,r)|{
            let empty: BTreeMap<String, Option<f64>> = OBJECTIVES.iter().map(|x|(x.to_string(),None)).collect();
            RouteReport{
                route_id:r.route_id.clone(),
                technology:r.technology.clone(),
                feasibility_probability:probs[i],
                feasibility_estimate:probability_estimate(ok[i],m.draws),
                robustly_feasible:robust[i],
                pareto_membership_probability:None,
                pareto_membership_estimate:None,
                mean_regret:empty.clone(),
                mean_regret_mcse:empty.clone(),
                p95_regret:empty,
            }
        }).collect();
        let diagnostic_baselines=diagnostic_baselines(g,&draws,&qualified);
        return GroupReport{
            group_id:g.group_id.clone(),
            disposition:"ABSTAIN_INSUFFICIENT_FEASIBLE_ROUTES".to_owned(),
            reasons:vec!["fewer than two routes satisfy all hard constraints at the required probability".to_owned()],
            routes,
            pairwise_dominance:vec![],
            diagnostic_baselines,
        };
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
        let mut mean=BTreeMap::new();
        let mut mean_mcse=BTreeMap::new();
        let mut tail=BTreeMap::new();
        for metric in OBJECTIVES {
            let values=regrets[i].get_mut(metric).unwrap();
            let (average,mcse)=mean_and_mcse(values);
            mean.insert(metric.to_owned(),average);
            mean_mcse.insert(metric.to_owned(),mcse);
            tail.insert(metric.to_owned(),p95(values));
        }
        let pareto_estimate=if robust[i] {
            Some(probability_estimate(pareto[i],m.draws))
        } else {
            None
        };
        RouteReport{
            route_id:r.route_id.clone(),
            technology:r.technology.clone(),
            feasibility_probability:probs[i],
            feasibility_estimate:probability_estimate(ok[i],m.draws),
            robustly_feasible:robust[i],
            pareto_membership_probability:pareto_estimate.as_ref().map(|estimate|estimate.probability),
            pareto_membership_estimate:pareto_estimate,
            mean_regret:mean,
            mean_regret_mcse:mean_mcse,
            p95_regret:tail,
        }
    }).collect();
    let pairwise_dominance=pairs.into_iter().map(|((a,b),(joint,aw,bw))|{
        let a_estimate=probability_estimate(aw,joint);
        let b_estimate=probability_estimate(bw,joint);
        PairReport{
            route_a:g.routes[a].route_id.clone(),
            route_b:g.routes[b].route_id.clone(),
            a_dominates_b_probability:a_estimate.probability,
            a_dominates_b_estimate:a_estimate,
            b_dominates_a_probability:b_estimate.probability,
            b_dominates_a_estimate:b_estimate,
            jointly_feasible_draws:joint
        }
    }).collect();
    let diagnostic_baselines=diagnostic_baselines(g,&draws,&qualified);
    GroupReport{
        group_id:g.group_id.clone(),
        disposition:"SYNTHETIC_PAIRED_PARETO_ANALYSIS".to_owned(),
        reasons:vec![
            "synthetic algorithm exercise only; no physical or commercial claim".to_owned(),
            "Pareto probability, pairwise dominance and per-metric regret are separate; no scalar score".to_owned(),
            "diagnostic baseline rankings are not authoritative and cannot override hard constraints or abstention".to_owned(),
            "probability intervals quantify Monte Carlo sampling error under the declared synthetic distributions, not physical uncertainty or model validity".to_owned()
        ],
        routes,
        pairwise_dominance,
        diagnostic_baselines
    }
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


#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SensitivitySettings {
    schema_version: String,
    sensitivity_id: String,
    seeds: Vec<u64>,
    draw_counts: Vec<u32>,
    max_feasibility_probability_range: f64,
    max_pareto_probability_range: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensitivityReport {
    pub sensitivity_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub seed_count: usize,
    pub draw_counts: Vec<u32>,
    pub run_count: usize,
    pub groups: Vec<GroupSensitivityReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupSensitivityReport {
    pub group_id: String,
    pub run_count: u32,
    pub disposition_counts: BTreeMap<String, u32>,
    pub routes: Vec<RouteSensitivityReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteSensitivityReport {
    pub route_id: String,
    pub technology: String,
    pub run_count: u32,
    pub feasibility_probability_min: f64,
    pub feasibility_probability_max: f64,
    pub feasibility_probability_range: f64,
    pub feasibility_range_within_tolerance: bool,
    pub robustly_feasible_runs: u32,
    pub robust_classification_stable: bool,
    pub pareto_membership_available_runs: u32,
    pub pareto_membership_probability_min: Option<f64>,
    pub pareto_membership_probability_max: Option<f64>,
    pub pareto_membership_probability_range: Option<f64>,
    pub pareto_range_within_tolerance: Option<bool>,
    pub diagnostic_top_rank_runs: BTreeMap<String, u32>,
}

#[derive(Debug, Default)]
struct RouteSensitivityAccumulator {
    technology: Option<String>,
    feasibility_probabilities: Vec<f64>,
    robust_flags: Vec<bool>,
    pareto_probabilities: Vec<f64>,
    diagnostic_top_rank_runs: BTreeMap<String, u32>,
}

fn value_range(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((min, max))
}

/// Re-evaluates one explicitly synthetic decision fixture across multiple seeds and
/// draw budgets. This diagnoses Monte Carlo stability only; it does not calibrate
/// or validate the input distributions against physical data.
pub fn evaluate_decision_sensitivity_json(
    decision_source: &str,
    settings_source: &str,
) -> Result<SensitivityReport, String> {
    let settings: SensitivitySettings = serde_json::from_str(settings_source)
        .map_err(|error| format!("invalid sensitivity settings: {error}"))?;
    if settings.schema_version != "luminous.plasma-pathway-sensitivity.v0" {
        return Err("unsupported pathway sensitivity schema".into());
    }
    if !nonempty(&settings.sensitivity_id) {
        return Err("sensitivity_id must be non-empty".into());
    }
    if settings.seeds.len() < 3 {
        return Err("at least three distinct seeds are required".into());
    }
    let mut unique_seeds = BTreeSet::new();
    if settings.seeds.iter().any(|seed| !unique_seeds.insert(*seed)) {
        return Err("sensitivity seeds must be unique".into());
    }
    if settings.draw_counts.len() < 2 {
        return Err("at least two draw budgets are required".into());
    }
    if settings.draw_counts.iter().any(|draws| *draws < MIN_DRAWS || *draws > 1_000_000) {
        return Err(format!(
            "draw budgets must be between {MIN_DRAWS} and 1000000"
        ));
    }
    if settings.draw_counts.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("draw budgets must be strictly increasing".into());
    }
    if !settings.max_feasibility_probability_range.is_finite()
        || !(0.0..=1.0).contains(&settings.max_feasibility_probability_range)
        || !settings.max_pareto_probability_range.is_finite()
        || !(0.0..=1.0).contains(&settings.max_pareto_probability_range)
    {
        return Err("sensitivity probability-range tolerances must be within [0, 1]".into());
    }

    let mut base: serde_json::Value = serde_json::from_str(decision_source)
        .map_err(|error| format!("invalid decision manifest: {error}"))?;
    if base.get("evidence_class").and_then(serde_json::Value::as_str) != Some("SYNTHETIC") {
        return Err("sensitivity v0 accepts synthetic decision manifests only".into());
    }

    let run_count = settings.seeds.len() * settings.draw_counts.len();
    let mut route_accumulators: BTreeMap<(String, String), RouteSensitivityAccumulator> =
        BTreeMap::new();
    let mut group_dispositions: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    let mut group_run_counts: BTreeMap<String, u32> = BTreeMap::new();

    for seed in &settings.seeds {
        for draws in &settings.draw_counts {
            base["seed"] = serde_json::json!(seed);
            base["draws"] = serde_json::json!(draws);
            let report = evaluate_decision_manifest_json(&base.to_string())?;
            for group in report.groups {
                *group_run_counts.entry(group.group_id.clone()).or_default() += 1;
                *group_dispositions
                    .entry(group.group_id.clone())
                    .or_default()
                    .entry(group.disposition.clone())
                    .or_default() += 1;

                for route in group.routes {
                    let key = (group.group_id.clone(), route.route_id.clone());
                    let accumulator = route_accumulators.entry(key).or_default();
                    if let Some(existing) = &accumulator.technology {
                        if existing != &route.technology {
                            return Err("route technology changed across sensitivity runs".into());
                        }
                    } else {
                        accumulator.technology = Some(route.technology.clone());
                    }
                    accumulator.feasibility_probabilities.push(route.feasibility_probability);
                    accumulator.robust_flags.push(route.robustly_feasible);
                    if let Some(probability) = route.pareto_membership_probability {
                        accumulator.pareto_probabilities.push(probability);
                    }
                }

                for baseline in group.diagnostic_baselines {
                    if let Some(top_route_id) = baseline.ranked_route_ids.first() {
                        let key = (group.group_id.clone(), top_route_id.clone());
                        let accumulator = route_accumulators.entry(key).or_default();
                        *accumulator
                            .diagnostic_top_rank_runs
                            .entry(baseline.method)
                            .or_default() += 1;
                    }
                }
            }
        }
    }

    let mut groups = Vec::new();
    for (group_id, run_count_for_group) in group_run_counts {
        let mut routes = Vec::new();
        for ((candidate_group, route_id), accumulator) in &route_accumulators {
            if candidate_group != &group_id {
                continue;
            }
            let Some((feasibility_min, feasibility_max)) =
                value_range(&accumulator.feasibility_probabilities)
            else {
                return Err(format!("route {route_id} has no valid feasibility estimates"));
            };
            let feasibility_range = feasibility_max - feasibility_min;
            let robustly_feasible_runs = accumulator
                .robust_flags
                .iter()
                .filter(|value| **value)
                .count() as u32;
            let robust_classification_stable = robustly_feasible_runs == 0
                || robustly_feasible_runs == accumulator.robust_flags.len() as u32;
            let pareto_range = value_range(&accumulator.pareto_probabilities);
            let (pareto_min, pareto_max, pareto_width, pareto_within_tolerance) =
                match pareto_range {
                    Some((min, max)) => (
                        Some(min),
                        Some(max),
                        Some(max - min),
                        Some(max - min <= settings.max_pareto_probability_range),
                    ),
                    None => (None, None, None, None),
                };

            routes.push(RouteSensitivityReport {
                route_id: route_id.clone(),
                technology: accumulator.technology.clone().unwrap_or_default(),
                run_count: accumulator.feasibility_probabilities.len() as u32,
                feasibility_probability_min: feasibility_min,
                feasibility_probability_max: feasibility_max,
                feasibility_probability_range: feasibility_range,
                feasibility_range_within_tolerance: feasibility_range
                    <= settings.max_feasibility_probability_range,
                robustly_feasible_runs,
                robust_classification_stable,
                pareto_membership_available_runs: accumulator.pareto_probabilities.len() as u32,
                pareto_membership_probability_min: pareto_min,
                pareto_membership_probability_max: pareto_max,
                pareto_membership_probability_range: pareto_width,
                pareto_range_within_tolerance: pareto_within_tolerance,
                diagnostic_top_rank_runs: accumulator.diagnostic_top_rank_runs.clone(),
            });
        }
        routes.sort_by(|left, right| left.route_id.cmp(&right.route_id));
        groups.push(GroupSensitivityReport {
            group_id: group_id.clone(),
            run_count: run_count_for_group,
            disposition_counts: group_dispositions.remove(&group_id).unwrap_or_default(),
            routes,
        });
    }

    groups.sort_by(|left, right| left.group_id.cmp(&right.group_id));
    Ok(SensitivityReport {
        sensitivity_id: settings.sensitivity_id,
        evidence_class: "SYNTHETIC".to_owned(),
        claim_ceiling: "MONTE_CARLO_STABILITY_ONLY_NOT_EMPIRICAL_CALIBRATION_OR_PHYSICAL_VALIDATION".to_owned(),
        seed_count: settings.seeds.len(),
        draw_counts: settings.draw_counts,
        run_count,
        groups,
    })
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
    const SENSITIVITY:&str=include_str!("../plasma-audit-harness/fixtures/pathway-sensitivity-v0.json");
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
    fn feasibility_probability_reports_wilson_interval_and_mcse() {
        let report=evaluate_decision_manifest_json(FIXTURE).unwrap();
        let estimate=&report.groups[0].routes[0].feasibility_estimate;
        assert_eq!(estimate.trials,10_000);
        assert_eq!(estimate.probability,report.groups[0].routes[0].feasibility_probability);
        assert!(estimate.ci95_lower<=estimate.probability);
        assert!(estimate.probability<=estimate.ci95_upper);
        assert!(estimate.monte_carlo_standard_error>=0.0);
    }

    #[test]
    fn wilson_interval_narrows_with_more_draws() {
        let small=probability_estimate(900,1_000);
        let large=probability_estimate(9_000,10_000);
        let small_width=small.ci95_upper-small.ci95_lower;
        let large_width=large.ci95_upper-large.ci95_lower;
        assert!(large_width<small_width);
    }

    #[test]
    fn wilson_interval_handles_all_success_and_all_failure_edges() {
        let all_success=probability_estimate(100,100);
        let all_failure=probability_estimate(0,100);
        assert_eq!(all_success.probability,1.0);
        assert_eq!(all_failure.probability,0.0);
        assert!(all_success.ci95_upper<=1.0);
        assert!(all_failure.ci95_lower>=0.0);
        assert!(all_success.ci95_lower<1.0);
        assert!(all_failure.ci95_upper>0.0);
    }

    #[test]
    fn pairwise_dominance_reports_joint_feasible_confidence_denominator() {
        let report=evaluate_decision_manifest_json(FIXTURE).unwrap();
        let pair=&report.groups[0].pairwise_dominance[0];
        assert!(pair.jointly_feasible_draws<=report.draws);
        assert_eq!(pair.a_dominates_b_estimate.trials,pair.jointly_feasible_draws);
        assert_eq!(pair.b_dominates_a_estimate.trials,pair.jointly_feasible_draws);
    }

    #[test]
    fn baseline_rankings_are_diagnostic_only_and_do_not_override_pareto() {
        let report=evaluate_decision_manifest_json(FIXTURE).unwrap();
        let group=&report.groups[0];
        assert_eq!(group.disposition,"SYNTHETIC_PAIRED_PARETO_ANALYSIS");
        assert_eq!(group.diagnostic_baselines.len(),4);
        assert!(group.diagnostic_baselines.iter().all(|baseline|baseline.diagnostic_only));
        assert!(group.diagnostic_baselines.iter().any(|baseline|baseline.method=="TOPSIS_EQUAL_METRIC_DIRECTION"));
        assert!(group.diagnostic_baselines.iter().any(|baseline|baseline.method=="FIXED_EQUAL_WEIGHT_NORMALIZED_SUM"));
    }

    #[test]
    fn abstention_remains_authoritative_when_hard_constraints_fail() {
        let mut value=fixture();
        for route in value["groups"][0]["routes"].as_array_mut().unwrap() {
            route["metrics"][3]["low"]=serde_json::json!(0.90);
            route["metrics"][3]["mode"]=serde_json::json!(0.91);
            route["metrics"][3]["high"]=serde_json::json!(0.92);
        }
        let report=evaluate_decision_manifest_json(&value.to_string()).unwrap();
        assert_eq!(report.groups[0].disposition,"ABSTAIN_INSUFFICIENT_FEASIBLE_ROUTES");
        assert!(report.groups[0].routes.iter().all(|route|!route.robustly_feasible));
    }


    #[test]
    fn sensitivity_runs_multiple_seeds_and_draw_budgets() {
        let report = evaluate_decision_sensitivity_json(FIXTURE, SENSITIVITY).unwrap();
        assert_eq!(report.seed_count, 3);
        assert_eq!(report.draw_counts, vec![10_000, 20_000]);
        assert_eq!(report.run_count, 6);
        assert_eq!(report.groups.len(), 1);
        assert_eq!(report.groups[0].run_count, 6);
        assert_eq!(report.groups[0].routes.len(), 3);
        assert_eq!(
            report.claim_ceiling,
            "MONTE_CARLO_STABILITY_ONLY_NOT_EMPIRICAL_CALIBRATION_OR_PHYSICAL_VALIDATION"
        );
    }

    #[test]
    fn sensitivity_is_reproducible_for_a_pinned_seed_grid() {
        let first = serde_json::to_string(
            &evaluate_decision_sensitivity_json(FIXTURE, SENSITIVITY).unwrap()
        )
        .unwrap();
        let second = serde_json::to_string(
            &evaluate_decision_sensitivity_json(FIXTURE, SENSITIVITY).unwrap()
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn sensitivity_rejects_duplicate_seeds() {
        let mut value: Value = serde_json::from_str(SENSITIVITY).unwrap();
        value["seeds"][1] = value["seeds"][0].clone();
        assert!(evaluate_decision_sensitivity_json(FIXTURE, &value.to_string()).is_err());
    }

    #[test]
    fn sensitivity_rejects_non_increasing_draw_budgets() {
        let mut value: Value = serde_json::from_str(SENSITIVITY).unwrap();
        value["draw_counts"] = serde_json::json!([20_000, 10_000]);
        assert!(evaluate_decision_sensitivity_json(FIXTURE, &value.to_string()).is_err());
    }

    #[test]
    fn sensitivity_reports_robust_feasibility_flip_count() {
        let report = evaluate_decision_sensitivity_json(FIXTURE, SENSITIVITY).unwrap();
        for route in &report.groups[0].routes {
            assert!(route.robustly_feasible_runs <= route.run_count);
            assert_eq!(
                route.robust_classification_stable,
                route.robustly_feasible_runs == 0
                    || route.robustly_feasible_runs == route.run_count
            );
            assert!(route.feasibility_probability_min <= route.feasibility_probability_max);
        }
    }

    #[test] fn less_than_10000_draws_is_rejected(){
        let mut v=fixture();v["draws"]=serde_json::json!(9999);assert!(evaluate_decision_manifest_json(&v.to_string()).is_err());
    }
}
