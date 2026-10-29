//! VIS-DVPC-044: reward-predictor control-variate identity and wealth-floor audit.
//!
//! This is a proof-obligation / counterexample search, not a MOPE interval
//! implementation. It checks the exact control-variate transform used by the
//! public MOPE reward-predictor path and compares its encoded feasible set with
//! action-specific half-wealth constraints.
#![deny(unsafe_code)]

use std::fmt::Write as _;

const TOL: f64 = 1e-9;
const WEALTH_FLOOR: f64 = 0.5;

#[derive(Clone, Debug)]
struct Dgp {
    name: &'static str,
    h: Vec<f64>,
    pi: Vec<f64>,
    reward_mean: Vec<f64>,
    predictor: Vec<f64>,
    max_weight: f64,
}

impl Dgp {
    fn healthy_oracle() -> Self {
        Self { name: "healthy_oracle", h: vec![0.5,0.5], pi: vec![0.5,0.5],
            reward_mean: vec![0.2,0.8], predictor: vec![0.2,0.8], max_weight: 1.0 }
    }
    fn weak_oracle() -> Self {
        Self { name: "weak_overlap_oracle", h: vec![0.01,0.99], pi: vec![0.5,0.5],
            reward_mean: vec![0.2,0.8], predictor: vec![0.2,0.8], max_weight: 50.0 }
    }
    fn weak_reversed() -> Self {
        Self { name: "weak_overlap_reversed", h: vec![0.01,0.99], pi: vec![0.5,0.5],
            reward_mean: vec![0.2,0.8], predictor: vec![0.8,0.2], max_weight: 50.0 }
    }
    fn weak_constant() -> Self {
        Self { name: "weak_overlap_constant", h: vec![0.01,0.99], pi: vec![0.5,0.5],
            reward_mean: vec![0.2,0.8], predictor: vec![0.5,0.5], max_weight: 50.0 }
    }
    fn balanced_constant() -> Self {
        Self { name: "balanced_constant", h: vec![0.5,0.5], pi: vec![0.5,0.5],
            reward_mean: vec![0.2,0.8], predictor: vec![0.5,0.5], max_weight: 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActionControl {
    h: f64,
    pi: f64,
    w: f64,
    q: f64,
    c: f64,
    residual_min: f64,
    residual_max: f64,
    reward_mean: f64,
}

#[derive(Clone, Debug)]
struct Audit {
    name: &'static str,
    q_pi: f64,
    true_value: f64,
    adjusted_mean: f64,
    raw_variance: f64,
    residual_variance: f64,
    min_residual: f64,
    max_residual: f64,
    actions: Vec<ActionControl>,
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a-b).abs() <= tol*(1.0+a.abs().max(b.abs()))
}

fn probability_vector_ok(p: &[f64]) -> bool {
    !p.is_empty() && p.iter().all(|x| x.is_finite() && *x>=0.0 && *x<=1.0)
        && close(p.iter().sum(),1.0,1e-10)
}

fn audit(dgp: &Dgp) -> Result<Audit,String> {
    if dgp.h.len()!=dgp.pi.len() || dgp.pi.len()!=dgp.reward_mean.len()
        || dgp.pi.len()!=dgp.predictor.len() || dgp.h.is_empty() {
        return Err(format!("{}: action vector lengths differ",dgp.name));
    }
    if !probability_vector_ok(&dgp.h) || !probability_vector_ok(&dgp.pi)
        || dgp.reward_mean.iter().chain(dgp.predictor.iter())
            .any(|x| !x.is_finite() || *x<0.0 || *x>1.0) {
        return Err(format!("{}: invalid probability, reward mean, or predictor",dgp.name));
    }

    for i in 0..dgp.h.len() {
        if dgp.h[i]==0.0 && dgp.pi[i]>0.0 {
            return Err(format!("{}: target action {} has zero logger support",dgp.name,i));
        }
    }

    let q_pi:f64=dgp.pi.iter().zip(dgp.predictor.iter()).map(|(p,q)|p*q).sum();
    let true_value:f64=dgp.pi.iter().zip(dgp.reward_mean.iter()).map(|(p,m)|p*m).sum();
    let mut actions=Vec::new();
    let mut raw_first=0.0;
    let mut raw_second=0.0;
    let mut residual_first=0.0;
    let mut residual_second=0.0;

    for i in 0..dgp.h.len() {
        let h=dgp.h[i];
        if h==0.0 { continue; }
        let pi=dgp.pi[i];
        let w=pi/h;
        if !w.is_finite() || w>dgp.max_weight+TOL {
            return Err(format!("{}: importance ratio exceeds declared bound",dgp.name));
        }
        let q=dgp.predictor[i];
        let c=w*q-q_pi;
        let ymin=-c;
        let ymax=w-c;
        actions.push(ActionControl{h,pi,w,q,c,residual_min:ymin,residual_max:ymax,reward_mean:dgp.reward_mean[i]});

        for (reward,prob_r) in [(0.0,1.0-dgp.reward_mean[i]),(1.0,dgp.reward_mean[i])] {
            let p=h*prob_r;
            let raw=w*reward;
            let adjusted=raw-c;
            raw_first+=p*raw;
            raw_second+=p*raw*raw;
            residual_first+=p*adjusted;
            residual_second+=p*adjusted*adjusted;
        }
    }
    let min_residual=actions.iter().map(|a|a.residual_min).fold(f64::INFINITY,f64::min);
    let max_residual=actions.iter().map(|a|a.residual_max).fold(f64::NEG_INFINITY,f64::max);
    Ok(Audit{
        name:dgp.name,q_pi,true_value,adjusted_mean:residual_first,
        raw_variance:raw_second-raw_first*raw_first,
        residual_variance:residual_second-residual_first*residual_first,
        min_residual,max_residual,actions,
    })
}

#[derive(Clone,Copy,Debug)]
struct Bet { x:f64, y:f64 }

fn upstream_encoded_feasible(bet: Bet, w_max:f64) -> bool {
    // Directly translates the two inequalities in n17s/mope/opebetrp.py::solve2:
    // G=[(-1,-1),(W-1,-W)], G @ bet >= -1/2, bet[1] >= 0.
    bet.y>=-TOL
        && -bet.x-bet.y>=-WEALTH_FLOOR-TOL
        && (w_max-1.0)*bet.x-w_max*bet.y>=-WEALTH_FLOOR-TOL
}

fn actual_factor(action: ActionControl, reward:f64, candidate:f64, bet:Bet)->f64 {
    1.0+bet.x*(action.w-1.0)+bet.y*(action.w*reward-action.c-candidate)
}

fn support_aware_feasible(bet:Bet, audit:&Audit)->bool {
    if !bet.x.is_finite() || !bet.y.is_finite() || bet.y < -TOL { return false; }
    audit.actions.iter().all(|a| {
        // Because bet.y >= 0, the minimum factor for this action occurs at
        // reward=0 and candidate value=1.
        bet.x*(a.w-1.0)+bet.y*(a.residual_min-1.0)>=-WEALTH_FLOOR-TOL
    })
}

fn support_aware_minimum_factor(bet:Bet,audit:&Audit)->f64 {
    audit.actions.iter().map(|a| {
        1.0+bet.x*(a.w-1.0)+bet.y*(a.residual_min-1.0)
    }).fold(f64::INFINITY,f64::min)
}
fn support_aware_corners_hold(bet:Bet,audit:&Audit)->bool{
    audit.actions.iter().all(|action|{
        [0.0,1.0].iter().all(|reward|{
            [0.0,1.0].iter().all(|candidate|{
                actual_factor(*action,*reward,*candidate,bet)>=WEALTH_FLOOR-TOL
            })
        })
    })
}

fn upstream_counterexample() -> Result<(Audit,Bet,f64),String> {
    let dgp=Dgp{
        name:"legacy_polytope_factor_floor_counterexample",
        h:vec![0.01,0.99],pi:vec![0.5,0.5],
        reward_mean:vec![0.2,0.8],predictor:vec![0.999,0.001],max_weight:50.0,
    };
    let a=audit(&dgp)?;
    let w_max=50.0;
    let lambda_y=0.1;
    let lambda_x=(w_max*lambda_y-WEALTH_FLOOR)/(w_max-1.0);
    let bet=Bet{x:lambda_x,y:lambda_y};
    if !upstream_encoded_feasible(bet,w_max) {
        return Err("frozen legacy witness was not feasible under encoded inequalities".into());
    }
    let rare_action=a.actions.iter().find(|x|close(x.w,50.0,1e-12)).ok_or("rare action missing")?;
    let factor=actual_factor(*rare_action,0.0,1.0,bet);
    if support_aware_feasible(bet,&a) {
        return Err("support-aware constraints unexpectedly accepted legacy witness".into());
    }
    if !close(factor,0.455,1e-8) {
        return Err(format!("legacy factor witness changed: {factor}"));
    }
    Ok((a,bet,factor))
}

#[derive(Clone,Copy,Debug)]
struct Constraint { a:f64, b:f64, c:f64 }
#[derive(Clone,Copy,Debug,Default)]
struct QuadraticStats { a00:f64, a01:f64, a11:f64, b0:f64, b1:f64 }
#[derive(Clone,Copy,Debug)]
struct Sample { w:f64, r:f64, c:f64 }
#[derive(Clone,Copy,Debug)]
struct Optimum { bet:Bet, objective:f64, candidates:usize }

fn action_specific_constraints(audit:&Audit)->Vec<Constraint>{
    let mut out=Vec::with_capacity(audit.actions.len()+3);
    // lambda_2 >= 0.
    out.push(Constraint{a:0.0,b:-1.0,c:0.0});
    for action in &audit.actions{
        // lambda_1(w-1)+lambda_2(y_min-1) >= -1/2
        // is equivalent to (1-w)lambda_1+(c+1)lambda_2 <= 1/2.
        out.push(Constraint{a:1.0-action.w,b:action.c+1.0,c:WEALTH_FLOOR});
    }
    if audit.actions.iter().all(|a|close(a.w,1.0,1e-10)){
        // lambda_1 has no wealth effect when all importance weights equal one.
        // Pin it to zero to eliminate an otherwise flat/unbounded direction.
        out.push(Constraint{a:1.0,b:0.0,c:0.0});
        out.push(Constraint{a:-1.0,b:0.0,c:0.0});
    }
    out
}

fn feasible_constraints(bet:Bet,constraints:&[Constraint])->bool{
    bet.x.is_finite() && bet.y.is_finite()
        && constraints.iter().all(|c|c.a*bet.x+c.b*bet.y<=c.c+1e-8)
}
fn qstats(samples:&[Sample],value:f64)->QuadraticStats{
    let mut s=QuadraticStats::default();
    for sample in samples{
        let x=sample.w-1.0;
        let y=sample.w*sample.r-sample.c-value;
        s.a00+=x*x; s.a01+=x*y; s.a11+=y*y; s.b0+=x; s.b1+=y;
    }
    s
}
fn objective(stats:QuadraticStats,bet:Bet)->f64{
    let psi=2.0-4.0*std::f64::consts::LN_2;
    psi*(stats.a00*bet.x*bet.x+2.0*stats.a01*bet.x*bet.y+stats.a11*bet.y*bet.y)
        +stats.b0*bet.x+stats.b1*bet.y
}
fn feasible_vertices(constraints:&[Constraint])->Vec<Bet>{
    let mut vertices=Vec::new();
    for i in 0..constraints.len(){
        for j in (i+1)..constraints.len(){
            let p=constraints[i]; let q=constraints[j];
            let det=p.a*q.b-p.b*q.a;
            if det.abs()<=1e-12{continue;}
            let x=(p.c*q.b-p.b*q.c)/det;
            let y=(p.a*q.c-p.c*q.a)/det;
            let candidate=Bet{x,y};
            if feasible_constraints(candidate,constraints)
                && !vertices.iter().any(|v:&Bet|close(v.x,x,1e-8)&&close(v.y,y,1e-8)){
                vertices.push(candidate);
            }
        }
    }
    vertices
}
fn optimize_support_polytope(stats:QuadraticStats,constraints:&[Constraint])->Result<Optimum,String>{
    let vertices=feasible_vertices(constraints);
    if vertices.is_empty(){return Err("support-aware feasible polygon has no enumerated vertices".into());}
    let mut candidates=vertices.clone();

    // Include the unconstrained stationary point when the matrix is well-conditioned.
    let scale=stats.a00.abs().max(stats.a01.abs()).max(stats.a11.abs());
    if scale>0.0{
        let (a00,a01,a11)=(stats.a00/scale,stats.a01/scale,stats.a11/scale);
        let det=a00*a11-a01*a01;
        if det>1e-10{
            let psi=2.0-4.0*std::f64::consts::LN_2;
            let den=2.0*psi*scale*det;
            let stationary=Bet{
                x:-(a11*stats.b0-a01*stats.b1)/den,
                y:-(-a01*stats.b0+a00*stats.b1)/den,
            };
            if feasible_constraints(stationary,constraints){candidates.push(stationary);}
        }
    }

    // Each edge optimum is its exact 1-D stationary point clamped to the feasible segment.
    let psi=2.0-4.0*std::f64::consts::LN_2;
    for constraint in constraints{
        let normal_sq=constraint.a*constraint.a+constraint.b*constraint.b;
        if normal_sq<=1e-16{continue;}
        let on_edge:Vec<Bet>=vertices.iter().copied()
            .filter(|v|(constraint.a*v.x+constraint.b*v.y-constraint.c).abs()<=1e-7).collect();
        if on_edge.len()<2{continue;}
        let direction=Bet{x:-constraint.b,y:constraint.a};
        let direction_sq=direction.x*direction.x+direction.y*direction.y;
        let mut low=f64::INFINITY; let mut high=f64::NEG_INFINITY;
        for v in &on_edge{
            let t=(v.x*direction.x+v.y*direction.y)/direction_sq;
            low=low.min(t); high=high.max(t);
        }
        if !(low.is_finite()&&high.is_finite())||low>high{continue;}
        let origin=Bet{x:constraint.a*constraint.c/normal_sq,y:constraint.b*constraint.c/normal_sq};
        let a_dir=stats.a00*direction.x*direction.x
            +2.0*stats.a01*direction.x*direction.y+stats.a11*direction.y*direction.y;
        let quad=psi*a_dir;
        let origin_a_dir=stats.a00*origin.x*direction.x
            +stats.a01*(origin.x*direction.y+origin.y*direction.x)
            +stats.a11*origin.y*direction.y;
        let linear=2.0*psi*origin_a_dir+stats.b0*direction.x+stats.b1*direction.y;
        let t=if quad < -1e-14 {(-linear/(2.0*quad)).clamp(low,high)}
            else if linear>0.0 {high} else {low};
        let edge=Bet{x:origin.x+t*direction.x,y:origin.y+t*direction.y};
        if feasible_constraints(edge,constraints){candidates.push(edge);}
    }

    let (bet,obj)=candidates.iter().copied()
        .filter(|candidate|feasible_constraints(*candidate,constraints))
        .map(|candidate|(candidate,objective(stats,candidate)))
        .max_by(|a,b|a.1.total_cmp(&b.1))
        .ok_or_else(||"no feasible optimizer candidate survived validation".to_owned())?;
    if !obj.is_finite(){return Err("non-finite support-aware objective".into());}
    Ok(Optimum{bet,objective:obj,candidates:candidates.len()})
}

fn cv_samples(audit:&Audit)->Vec<Sample>{
    let mut samples=Vec::new();
    for action in &audit.actions{
        for reward in [0.0,1.0]{
            samples.push(Sample{w:action.w,r:reward,c:action.c});
        }
    }
    samples
}
fn dense_grid_best(stats:QuadraticStats,constraints:&[Constraint])->f64{
    let mut best=f64::NEG_INFINITY;
    for ix in -100..=300{
        let x=ix as f64/200.0; // [-0.5, 1.5] in 0.005 steps.
        for iy in 0..=200{
            let y=iy as f64/200.0; // [0,1] in 0.005 steps.
            let bet=Bet{x,y};
            if feasible_constraints(bet,constraints){best=best.max(objective(stats,bet));}
        }
    }
    best
}

fn frozen_cases()->Vec<Dgp>{
    vec![Dgp::healthy_oracle(),Dgp::healthy_constant(),Dgp::weak_oracle(),Dgp::weak_reversed(),Dgp::weak_constant()]
}

fn run()->Result<String,String>{
    let mut out=String::from("record_type\tcase_id\tfield\tvalue\n");
    for dgp in frozen_cases(){
        let a=audit(&dgp)?;
        if !close(a.adjusted_mean,a.true_value,1e-10){
            return Err(format!("{}: control variate changed expected policy value",a.name));
        }
        let variance_ratio=if a.raw_variance>0.0{a.residual_variance/a.raw_variance}else{0.0};
        let _=writeln!(out,"PREDICTOR_CASE\t{}\ttrue_value\t{:.12}",a.name,a.true_value);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tq_pi\t{:.12}",a.name,a.q_pi);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tadjusted_mean\t{:.12}",a.name,a.adjusted_mean);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\traw_variance\t{:.12}",a.name,a.raw_variance.max(0.0));
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tresidual_variance\t{:.12}",a.name,a.residual_variance.max(0.0));
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tvariance_ratio\t{:.12}",a.name,variance_ratio);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tresidual_min\t{:.12}",a.name,a.min_residual);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tresidual_max\t{:.12}",a.name,a.max_residual);
        let _=writeln!(out,"PREDICTOR_CASE\t{}\tresidual_range_width\t{:.12}",a.name,a.max_residual-a.min_residual);
        for (i,action) in a.actions.iter().enumerate(){
            let _=writeln!(out,"ACTION\t{}\t{}\th\t{:.12}\tpi\t{:.12}\tw\t{:.12}\tq\t{:.12}\tc\t{:.12}\tymin\t{:.12}\tymax\t{:.12}",
                a.name,i,action.h,action.pi,action.w,action.q,action.c,action.residual_min,action.residual_max);
        }
        let _=writeln!(out,"CASE\t{}\tstatus\tOK",a.name);
    }
    let (a,bet,factor)=upstream_counterexample()?;
    let _=writeln!(out,"LEGACY_WITNESS\tfeasible_under_encoded_constraints\ttrue");
    let _=writeln!(out,"LEGACY_WITNESS\tlambda1\t{:.12}",bet.x);
    let _=writeln!(out,"LEGACY_WITNESS\tlambda2\t{:.12}",bet.y);
    let _=writeln!(out,"LEGACY_WITNESS\tactual_minimum_factor\t{:.12}",factor);
    let _=writeln!(out,"LEGACY_WITNESS\tviolates_half_wealth_floor\t{}",factor<WEALTH_FLOOR);
    let _=writeln!(out,"SAFE_REGION\twitness_rejected\t{}",!support_aware_feasible(bet,&a));
    let mut checked=0usize;
    let constraints=action_specific_constraints(&a);
    for xi in -100..=300 {
        let x=xi as f64/200.0;
        for yi in 0..=200 {
            let y=yi as f64/200.0;
            let bet=Bet{x,y};
            if support_aware_feasible(bet,&a){
                checked+=1;
                if support_aware_minimum_factor(bet,&a)<WEALTH_FLOOR-TOL
                    || !support_aware_corners_hold(bet,&a) {
                    return Err("support-aware feasible-set grid found a factor-floor violation".into());
                }
            }
        }
    }
    let samples=cv_samples(&a);
    let stats=qstats(&samples,0.5);
    let optimum=optimize_support_polytope(stats,&constraints)?;
    if !support_aware_feasible(optimum.bet,&a)
        || !support_aware_corners_hold(optimum.bet,&a) {
        return Err("support-aware optimizer returned a bet that violates the factor floor".into());
    }
    let dense=dense_grid_best(stats,&constraints);
    if optimum.objective+1e-7<dense{
        return Err(format!("support-aware optimizer underperformed dense feasible grid: {} < {}",optimum.objective,dense));
    }
    let _=writeln!(out,"SAFE_REGION\tfeasible_grid_points_checked\t{checked}");
    let _=writeln!(out,"SAFE_BET_OPTIMIZER\tlambda1\t{:.12}",optimum.bet.x);
    let _=writeln!(out,"SAFE_BET_OPTIMIZER\tlambda2\t{:.12}",optimum.bet.y);
    let _=writeln!(out,"SAFE_BET_OPTIMIZER\tobjective\t{:.12}",optimum.objective);
    let _=writeln!(out,"SAFE_BET_OPTIMIZER\tdense_grid_best\t{:.12}",dense);
    let _=writeln!(out,"SAFE_BET_OPTIMIZER\tminimum_factor\t{:.12}",support_aware_minimum_factor(optimum.bet,&a));
    Ok(out)
}

fn main(){
    match run(){
        Ok(out)=>print!("{out}"),
        Err(err)=>{eprintln!("VIS_DVPC_044_ERROR\t{err}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn control_variate_is_unbiased_for_oracle_and_misspecified_predictors(){
        for dgp in frozen_cases(){
            let a=audit(&dgp).unwrap();
            assert!(close(a.adjusted_mean,a.true_value,1e-10),"case={} adjusted={} true={}",a.name,a.adjusted_mean,a.true_value);
        }
    }

    #[test]
    fn oracle_predictor_reduces_variance_in_frozen_controls(){
        let healthy=audit(&Dgp::healthy_oracle()).unwrap();
        let weak=audit(&Dgp::weak_oracle()).unwrap();
        assert!(healthy.residual_variance<healthy.raw_variance);
        assert!(weak.residual_variance<weak.raw_variance);
    }

    #[test]
    fn legacy_encoded_feasible_region_has_a_candidate_factor_below_half(){
        let (_audit,bet,factor)=upstream_counterexample().unwrap();
        assert!(upstream_encoded_feasible(bet,50.0));
        assert!(close(factor,0.455,1e-8));
        assert!(factor<WEALTH_FLOOR);
    }

    #[test]
    fn action_specific_constraints_reject_legacy_witness_and_preserve_half_wealth_floor(){
        let (a,bet,_)=upstream_counterexample().unwrap();
        assert!(!support_aware_feasible(bet,&a));
        let checked=run().unwrap();
        assert!(checked.contains("SAFE_REGION\tfeasible_grid_points_checked"));
        for x_i in -100..=300{
            for y_i in 0..=200{
                let candidate=Bet{x:x_i as f64/200.0,y:y_i as f64/200.0};
                if support_aware_feasible(candidate,&a){
                    assert!(support_aware_minimum_factor(candidate,&a)>=WEALTH_FLOOR-TOL);
                    assert!(support_aware_corners_hold(candidate,&a));
                }
            }
        }
    }

    #[test]
    fn support_aware_polygon_optimizer_returns_safe_bet_no_worse_than_dense_grid(){
        let dgp=Dgp{
            name:"optimizer_control",
            h:vec![0.01,0.99],pi:vec![0.5,0.5],
            reward_mean:vec![0.2,0.8],predictor:vec![0.999,0.001],max_weight:50.0,
        };
        let a=audit(&dgp).unwrap();
        let constraints=action_specific_constraints(&a);
        let samples=cv_samples(&a);
        let stats=qstats(&samples,0.5);
        let optimum=optimize_support_polytope(stats,&constraints).unwrap();
        assert!(support_aware_feasible(optimum.bet,&a));
        assert!(support_aware_corners_hold(optimum.bet,&a));
        assert!(support_aware_minimum_factor(optimum.bet,&a)>=WEALTH_FLOOR-TOL);
        assert!(optimum.objective+1e-7>=dense_grid_best(stats,&constraints));
    }

    #[test]
    fn predictor_profile_does_not_change_declared_importance_support(){
        for dgp in frozen_cases(){
            let a=audit(&dgp).unwrap();
            for action in a.actions{
                assert!(action.w.is_finite());
                assert!(action.w<=dgp.max_weight+TOL);
            }
        }
    }
}
