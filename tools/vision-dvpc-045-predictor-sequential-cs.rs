//! VIS-DVPC-045: sequential MOPE with frozen reward-predictor control variates.
//!
//! Compares raw MOPE with a predictor-adjusted two-sided confidence sequence on
//! identical seeded samples. The adjusted process uses the action-specific
//! half-wealth constraints audited in VIS-DVPC-044.
#![deny(unsafe_code)]

#[path = "vision-dvpc-040-mope-sufficient-stats.rs"]
mod stream;
#[path = "vision-dvpc-044-control-variate-feasibility-audit.rs"]
mod predictor;

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const ALPHA: f64 = 0.05;
const HORIZON: usize = 512;
const REPLICATES: usize = 2_000;
const LOOKS: [usize;3] = [32,128,512];
const PROMOTION_THRESHOLD: f64 = 0.55;
const PSI: f64 = 2.0-4.0*std::f64::consts::LN_2;
const EPS: f64 = 1e-10;
const ROOT_TOL: f64 = 3e-7;
const WEALTH_TOL: f64 = 3e-8;
const VALUE_GRID: [f64;21]=[
    0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,
    0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,
];

#[derive(Clone,Copy,Debug)]
struct SplitMix64{state:u64}
impl SplitMix64{
    fn new(seed:u64)->Self{Self{state:seed}}
    fn next_u64(&mut self)->u64{
        self.state=self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z=self.state;
        z=(z^(z>>30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z=(z^(z>>27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z^(z>>31)
    }
    fn unit(&mut self)->f64{((self.next_u64()>>11) as f64)*(1.0/((1u64<<53) as f64))}
}

#[derive(Clone,Copy,Debug)]
struct Interval{lower:f64,upper:f64}
impl Interval{
    fn width(self)->f64{(self.upper-self.lower).max(0.0)}
    fn covers(self,value:f64)->bool{
        self.lower<=self.upper+EPS && self.lower<=value+EPS && self.upper>=value-EPS
    }
    fn reversed(self)->bool{self.lower>self.upper+EPS}
}

#[derive(Clone,Copy,Debug)]
struct Record{w:f64,score:f64,bet:predictor::Bet}

#[derive(Clone,Copy,Debug,Default)]
struct Sufficient{
    n:u64,
    sum_z:f64,
    sum_score:f64,
    sum_z2:f64,
    sum_z_score:f64,
    sum_score2:f64,
}
impl Sufficient{
    fn update(&mut self,w:f64,score:f64){
        let z=w-1.0;
        self.n+=1;
        self.sum_z+=z;
        self.sum_score+=score;
        self.sum_z2+=z*z;
        self.sum_z_score+=z*score;
        self.sum_score2+=score*score;
    }
    fn at(self,value:f64)->predictor::QuadraticStats{
        predictor::QuadraticStats{
            a00:self.sum_z2,
            a01:self.sum_z_score-value*self.sum_z,
            a11:self.sum_score2-2.0*value*self.sum_score+(self.n as f64)*value*value,
            b0:self.sum_z,
            b1:self.sum_score-(self.n as f64)*value,
        }
    }
}

#[derive(Clone,Copy,Debug,Default)]
struct BoundPolynomial{a:f64,b:f64,c:f64}
impl BoundPolynomial{
    fn update(&mut self,w:f64,score:f64,bet:predictor::Bet){
        let z=w-1.0;
        self.a+=PSI*bet.y*bet.y;
        self.b+=-bet.y+PSI*(-2.0*bet.x*bet.y*z-2.0*bet.y*bet.y*score);
        self.c+=bet.x*z+bet.y*score
            +PSI*(bet.x*bet.x*z*z+2.0*bet.x*bet.y*z*score+bet.y*bet.y*score*score);
    }
    fn value(self,v:f64)->f64{self.a*v*v+self.b*v+self.c}
}

fn real_roots(a:f64,b:f64,c:f64)->Vec<f64>{
    if [a,b,c].iter().any(|x|!x.is_finite()){return Vec::new();}
    let scale=a.abs().max(b.abs()).max(c.abs());
    if scale==0.0{return Vec::new();}
    let (a,b,c)=(a/scale,b/scale,c/scale);
    if a.abs()<=1e-12{
        if b.abs()<=1e-12{return Vec::new();}
        return vec![-c/b];
    }
    let disc=b*b-4.0*a*c;
    if disc < -1e-12{return Vec::new();}
    let root_disc=disc.max(0.0).sqrt();
    let q=-0.5*(b+b.signum()*root_disc);
    if q.abs()<=1e-14{return vec![-b/(2.0*a)];}
    let mut roots=vec![q/a,c/q];
    roots.sort_by(f64::total_cmp);
    if (roots[0]-roots[1]).abs()<=1e-10{roots.truncate(1);}
    roots
}

fn sample_score(sample:predictor::Sample,reflected:bool)->f64{
    let raw=sample.w*sample.r-sample.c;
    if reflected{1.0-raw}else{raw}
}
fn close(a:f64,b:f64,tol:f64)->bool{
    (a-b).abs()<=tol*(1.0+a.abs().max(b.abs()))
}
fn direct_log_wealth(records:&[Record],value:f64)->Option<f64>{
    let mut sum=0.0;
    for record in records{
        let factor=1.0+record.bet.x*(record.w-1.0)+record.bet.y*(record.score-value);
        if !factor.is_finite()||factor<0.5-WEALTH_TOL{return None;}
        sum+=factor.ln();
    }
    Some(sum)
}
fn direct_polynomial(records:&[Record],value:f64)->f64{
    records.iter().map(|record|{
        let z=record.w-1.0;
        let d=record.score-value;
        record.bet.x*z+record.bet.y*d
            +PSI*(record.bet.x*record.bet.x*z*z+2.0*record.bet.x*record.bet.y*z*d
                +record.bet.y*record.bet.y*d*d)
    }).sum()
}
fn certify_root(records:&[Record],root:f64,level:f64)->bool{
    if !root.is_finite()||!(0.0..=1.0).contains(&root){return false;}
    if records.iter().any(|record|record.bet.y < -WEALTH_TOL){return false;}
    if !close(direct_polynomial(records,root),level,ROOT_TOL){return false;}
    if !matches!(direct_log_wealth(records,root),Some(x) if x+WEALTH_TOL>=level){return false;}
    for v in VALUE_GRID.iter().copied().filter(|v|*v<=root+EPS){
        if !matches!(direct_log_wealth(records,v),Some(x) if x+WEALTH_TOL>=level){return false;}
    }
    true
}

#[derive(Clone,Copy,Debug,Default)]
struct Diagnostics{root_checks:u64,root_failures:u64,endpoint_collisions:u64}

struct PredictorSide{
    audit:predictor::Audit,
    reflected:bool,
    constraints:Vec<predictor::Constraint>,
    records:Vec<Record>,
    stats:Sufficient,
    polynomial:BoundPolynomial,
    current_bet:predictor::Bet,
    endpoint:f64,
    diagnostics:Diagnostics,
}
impl PredictorSide{
    fn new(audit:predictor::Audit,reflected:bool,capacity:usize)->Self{
        let constraints=predictor::action_specific_constraints(&audit,reflected);
        Self{
            audit,reflected,constraints,
            records:Vec::with_capacity(capacity),
            stats:Sufficient::default(),polynomial:BoundPolynomial::default(),
            current_bet:predictor::Bet{x:0.0,y:0.0},endpoint:0.0,
            diagnostics:Diagnostics::default(),
        }
    }
    fn observe(&mut self,sample:predictor::Sample)->Result<(),String>{
        let score=sample_score(sample,self.reflected);
        let bet=self.current_bet;
        for candidate in [0.0,self.endpoint,1.0]{
            let factor=1.0+bet.x*(sample.w-1.0)+bet.y*(score-candidate);
            if !factor.is_finite()||factor<0.5-WEALTH_TOL{
                return Err("predictor betting factor fell below 1/2 on an observed step".into());
            }
        }
        self.records.push(Record{w:sample.w,score,bet});
        self.stats.update(sample.w,score);
        self.polynomial.update(sample.w,score,bet);
        Ok(())
    }
    fn certified_root_candidate(&mut self,level:f64)->Option<f64>{
        let roots=real_roots(self.polynomial.a,self.polynomial.b,self.polynomial.c-level);
        let root=stream::safe_lower_root(&roots,self.endpoint)?;
        if root<=self.endpoint+EPS{return None;}
        self.diagnostics.root_checks+=1;
        if certify_root(&self.records,root,level){Some(root)}
        else{self.diagnostics.root_failures+=1;None}
    }
    fn set_endpoint(&mut self,value:f64){self.endpoint=value.clamp(self.endpoint,1.0);}
    fn choose_next_bet(&mut self)->Result<(),String>{
        let stats=self.stats.at(self.endpoint);
        let optimum=predictor::optimize_support_polytope(stats,&self.constraints)?;
        if !predictor::support_aware_feasible(optimum.bet,&self.audit,self.reflected)
            || !predictor::support_aware_corners_hold(optimum.bet,&self.audit,self.reflected)
            || predictor::support_aware_minimum_factor(optimum.bet,&self.audit,self.reflected)<0.5-WEALTH_TOL{
            return Err("support-aware optimizer emitted a bet below the half-wealth floor".into());
        }
        self.current_bet=optimum.bet;
        Ok(())
    }
}

struct PredictorPair{
    lower:PredictorSide,
    complement:PredictorSide,
    diagnostics:Diagnostics,
}
impl PredictorPair{
    fn new(audit:&predictor::Audit,capacity:usize)->Self{
        Self{
            lower:PredictorSide::new(audit.clone(),false,capacity),
            complement:PredictorSide::new(audit.clone(),true,capacity),
            diagnostics:Diagnostics::default(),
        }
    }
    fn update(&mut self,sample:predictor::Sample)->Result<Interval,String>{
        let level=(2.0/ALPHA).ln();
        self.lower.observe(sample)?;
        self.complement.observe(sample)?;
        let lo_root=self.lower.certified_root_candidate(level);
        let hi_root=self.complement.certified_root_candidate(level);

        let old_lower=self.lower.endpoint;
        let old_complement=self.complement.endpoint;
        let proposed_lower=lo_root.map(|root|old_lower.max(root)).unwrap_or(old_lower);
        let lower_cap=1.0-old_complement;
        if proposed_lower>lower_cap+EPS{self.diagnostics.endpoint_collisions+=1;}
        let new_lower=proposed_lower.min(lower_cap);
        self.lower.set_endpoint(new_lower);

        let proposed_complement=hi_root.map(|root|old_complement.max(root)).unwrap_or(old_complement);
        let complement_cap=1.0-new_lower;
        if proposed_complement>complement_cap+EPS{self.diagnostics.endpoint_collisions+=1;}
        let new_complement=proposed_complement.min(complement_cap);
        self.complement.set_endpoint(new_complement);

        self.lower.choose_next_bet()?;
        self.complement.choose_next_bet()?;
        self.diagnostics.root_checks=self.lower.diagnostics.root_checks+self.complement.diagnostics.root_checks;
        self.diagnostics.root_failures=self.lower.diagnostics.root_failures+self.complement.diagnostics.root_failures;
        Ok(Interval{lower:self.lower.endpoint,upper:1.0-self.complement.endpoint})
    }
}

#[derive(Clone,Copy,Debug)]
struct RawRecord{w:f64,r:f64,bet:predictor::Bet}
struct RawSide{
    stream:stream::StreamState,
    records:Vec<RawRecord>,
    endpoint:f64,
    last_root:Option<f64>,
    diagnostics:Diagnostics,
}
impl RawSide{
    fn new(max_weight:f64,capacity:usize)->Result<Self,String>{
        Ok(Self{stream:stream::StreamState::new(max_weight,ALPHA)?,records:Vec::with_capacity(capacity),endpoint:0.0,last_root:None,diagnostics:Diagnostics::default()})
    }
    fn observe(&mut self,w:f64,r:f64)->Result<(),String>{
        let lambda=self.stream.current_lambda();
        let report=self.stream.observe(w,r)?;
        self.records.push(RawRecord{w,r,bet:predictor::Bet{x:lambda.x,y:lambda.y}});
        self.last_root=report.selected_safe_root;
        Ok(())
    }
    fn record_view(&self)->Vec<Record>{
        self.records.iter().map(|r|Record{w:r.w,score:r.w*r.r,bet:r.bet}).collect()
    }
    fn certified_root_candidate(&mut self,level:f64)->Option<f64>{
        let root=self.last_root?;
        if root<=self.endpoint+EPS{return None;}
        self.diagnostics.root_checks+=1;
        if certify_root(&self.record_view(),root,level){Some(root)}
        else{self.diagnostics.root_failures+=1;None}
    }
    fn set_endpoint(&mut self,value:f64){self.endpoint=value.clamp(self.endpoint,1.0);}
    fn choose_next_bet(&mut self)->Result<(),String>{
        self.stream.choose_next_bet(self.endpoint)?;
        Ok(())
    }
}
struct RawPair{lower:RawSide,complement:RawSide,diagnostics:Diagnostics}
impl RawPair{
    fn new(max_weight:f64,capacity:usize)->Result<Self,String>{
        Ok(Self{lower:RawSide::new(max_weight,capacity)?,complement:RawSide::new(max_weight,capacity)?,diagnostics:Diagnostics::default()})
    }
    fn update(&mut self,w:f64,r:f64)->Result<Interval,String>{
        let level=(2.0/ALPHA).ln();
        self.lower.observe(w,r)?;
        self.complement.observe(w,1.0-r)?;
        let lo_root=self.lower.certified_root_candidate(level);
        let hi_root=self.complement.certified_root_candidate(level);
        let old_lower=self.lower.endpoint;
        let old_complement=self.complement.endpoint;
        let proposed_lower=lo_root.map(|x|old_lower.max(x)).unwrap_or(old_lower);
        let lower_cap=1.0-old_complement;
        if proposed_lower>lower_cap+EPS{self.diagnostics.endpoint_collisions+=1;}
        let new_lower=proposed_lower.min(lower_cap);
        self.lower.set_endpoint(new_lower);
        let proposed_complement=hi_root.map(|x|old_complement.max(x)).unwrap_or(old_complement);
        let upper_cap=1.0-new_lower;
        if proposed_complement>upper_cap+EPS{self.diagnostics.endpoint_collisions+=1;}
        let new_complement=proposed_complement.min(upper_cap);
        self.complement.set_endpoint(new_complement);
        self.lower.choose_next_bet()?;
        self.complement.choose_next_bet()?;
        self.diagnostics.root_checks=self.lower.diagnostics.root_checks+self.complement.diagnostics.root_checks;
        self.diagnostics.root_failures=self.lower.diagnostics.root_failures+self.complement.diagnostics.root_failures;
        Ok(Interval{lower:self.lower.endpoint,upper:1.0-self.complement.endpoint})
    }
}

fn make_case(name:&'static str,weak:bool,means:[f64;2],predictor_kind:u8)->predictor::Dgp{
    let q=match predictor_kind{
        0=>vec![means[0],means[1]],
        1=>vec![0.5,0.5],
        _=>vec![means[1],means[0]],
    };
    predictor::Dgp{
        name,
        h:if weak{vec![0.01,0.99]}else{vec![0.5,0.5]},
        pi:vec![0.5,0.5],
        reward_mean:means.to_vec(),
        predictor:q,
        max_weight:if weak{50.0}else{1.0},
    }
}
fn cases()->Vec<predictor::Dgp>{
    vec![
        make_case("healthy_null_oracle",false,[0.2,0.8],0),
        make_case("healthy_null_constant",false,[0.2,0.8],1),
        make_case("healthy_null_reversed",false,[0.2,0.8],2),
        make_case("healthy_alt_oracle",false,[0.5,0.8],0),
        make_case("healthy_alt_constant",false,[0.5,0.8],1),
        make_case("healthy_alt_reversed",false,[0.5,0.8],2),
        make_case("weak_null_oracle",true,[0.2,0.8],0),
        make_case("weak_null_constant",true,[0.2,0.8],1),
        make_case("weak_null_reversed",true,[0.2,0.8],2),
        make_case("weak_alt_oracle",true,[0.5,0.8],0),
        make_case("weak_alt_constant",true,[0.5,0.8],1),
        make_case("weak_alt_reversed",true,[0.5,0.8],2),
    ]
}
fn seed_group(name:&str)->usize{
    if name.starts_with("healthy_null_"){0}
    else if name.starts_with("healthy_alt_"){1}
    else if name.starts_with("weak_null_"){2}
    else if name.starts_with("weak_alt_"){3}
    else{panic!("unregistered benchmark case: {name}")}
}
fn seed_for(group_index:usize,replicate:usize)->u64{
    0x8b8b_8b8b_0450_0045
        ^ ((group_index as u64+1).wrapping_mul(0xD6E8_FEB8_6659_FD93))
        ^ (replicate as u64).wrapping_mul(0xA076_1D64_78BD_642F)
}

#[derive(Clone,Debug,Default)]
struct Aggregate{
    point_hits:usize,sequence_hits:usize,promotion_hits:usize,reversed_hits:usize,
    width_sum:f64,width_sq_sum:f64,lower_sum:f64,upper_sum:f64,
    elapsed_sum:f64,elapsed_sq_sum:f64,promotion_times:Vec<usize>,trials:usize,
    root_checks:u64,root_failures:u64,endpoint_collisions:u64,
}
impl Aggregate{
    fn add(&mut self,interval:Interval,point:bool,sequence:bool,promotion:Option<usize>,elapsed:f64,diagnostics:Diagnostics){
        self.trials+=1;
        self.root_checks+=diagnostics.root_checks;
        self.root_failures+=diagnostics.root_failures;
        self.endpoint_collisions+=diagnostics.endpoint_collisions;
        if point{self.point_hits+=1;}
        if sequence{self.sequence_hits+=1;}
        if interval.reversed(){self.reversed_hits+=1;}
        let width=interval.width();
        self.width_sum+=width;self.width_sq_sum+=width*width;
        self.lower_sum+=interval.lower;self.upper_sum+=interval.upper;
        self.elapsed_sum+=elapsed;self.elapsed_sq_sum+=elapsed*elapsed;
        if let Some(t)=promotion{self.promotion_hits+=1;self.promotion_times.push(t);}
    }
}
fn wilson(success:usize,total:usize)->(f64,f64,f64,f64){
    if total==0{return(0.0,0.0,0.0,1.0);}
    let n=total as f64;let p=success as f64/n;let se=(p*(1.0-p)/n).sqrt();
    let z=1.959963984540054;let den=1.0+z*z/n;
    let center=(p+z*z/(2.0*n))/den;
    let half=z*((p*(1.0-p)/n+z*z/(4.0*n*n)).sqrt())/den;
    (p,se,(center-half).max(0.0),(center+half).min(1.0))
}
fn mean_se(sum:f64,sum_sq:f64,n:usize)->(f64,f64){
    if n==0{return(0.0,0.0);}
    let mean=sum/n as f64;if n<2{return(mean,0.0);}
    let var=((sum_sq-sum*sum/n as f64).max(0.0))/(n-1) as f64;
    (mean,(var/n as f64).sqrt())
}
fn median(values:&[usize])->Option<usize>{
    if values.is_empty(){return None;}
    let mut sorted=values.to_vec();sorted.sort_unstable();Some(sorted[(sorted.len()-1)/2])
}
fn percentile90(values:&[usize])->Option<usize>{
    if values.is_empty(){return None;}
    let mut sorted=values.to_vec();sorted.sort_unstable();
    Some(sorted[((sorted.len()*9+9)/10).saturating_sub(1).min(sorted.len()-1)])
}

struct CaseResult{
    name:String,true_value:f64,w_max:f64,aggregates:Vec<Vec<Aggregate>>,
    elapsed_seconds:f64,observations:usize,root_checks:[u64;2],root_failures:[u64;2],
}
fn simulate_case(dgp:predictor::Dgp,reps:usize,looks:&[usize])->Result<CaseResult,String>{
    let audit=predictor::audit(&dgp)?;
    let max_look=*looks.iter().max().ok_or("no observation look")?;
    let started=Instant::now();
    let mut aggregates=vec![vec![Aggregate::default();looks.len()];2];
    let mut root_checks=[0u64;2];let mut root_failures=[0u64;2];
    let mut obs_count=0;
    for rep in 0..reps{
        let mut rng=SplitMix64::new(seed_for(seed_group(dgp.name),rep));
        let mut raw=RawPair::new(dgp.max_weight,max_look)?;
        let mut adjusted=PredictorPair::new(&audit,max_look);
        let mut sequence_covered=[true;2];
        let mut promotion_times:[Option<usize>;2]=[None;2];
        let mut elapsed=[0.0_f64;2];
        let mut look_index=0;
        for t in 1..=max_look{
            let action=if rng.unit()<dgp.h[1]{1}else{0};
            let reward=if rng.unit()<dgp.reward_mean[action]{1.0}else{0.0};
            let w=dgp.pi[action]/dgp.h[action];
            let c=w*dgp.predictor[action]-audit.q_pi;
            let sample=predictor::Sample{w,r:reward,c};
            let start_raw=Instant::now();
            let raw_interval=raw.update(w,reward)?;
            elapsed[0]+=start_raw.elapsed().as_secs_f64();
            let start_adjusted=Instant::now();
            let adjusted_interval=adjusted.update(sample)?;
            elapsed[1]+=start_adjusted.elapsed().as_secs_f64();
            let intervals=[raw_interval,adjusted_interval];
            for method in 0..2{
                sequence_covered[method]&=intervals[method].covers(audit.true_value);
                if promotion_times[method].is_none()&&!intervals[method].reversed()
                    && intervals[method].lower>PROMOTION_THRESHOLD{
                    promotion_times[method]=Some(t);
                }
            }
            while look_index<looks.len()&&looks[look_index]==t{
                for method in 0..2{
                    let diagnostics=if method==0{raw.diagnostics}else{adjusted.diagnostics};
                    aggregates[method][look_index].add(intervals[method],intervals[method].covers(audit.true_value),
                        sequence_covered[method],promotion_times[method],elapsed[method],diagnostics);
                }
                look_index+=1;
            }
            obs_count+=1;
        }
        root_checks[0]+=raw.diagnostics.root_checks;
        root_failures[0]+=raw.diagnostics.root_failures;
        root_checks[1]+=adjusted.diagnostics.root_checks;
        root_failures[1]+=adjusted.diagnostics.root_failures;
    }
    Ok(CaseResult{
        name:dgp.name.to_string(),true_value:audit.true_value,w_max:dgp.max_weight,aggregates,
        elapsed_seconds:started.elapsed().as_secs_f64(),observations:obs_count,root_checks,root_failures,
    })
}
fn emit_case(out:&mut String,result:&CaseResult,looks:&[usize]){
    let _=writeln!(out,"DGP\t{}\treplicates\t{}\ttrue_value\t{:.8}\tW\t{:.8}",result.name,result.aggregates[0][0].trials,result.true_value,result.w_max);
    let methods=["MOPE_raw","MOPE_predictor_safe"];
    for method in 0..2{
        for (li,look) in looks.iter().copied().enumerate(){
            let a=&result.aggregates[method][li];
            let (point,point_se,point_lo,point_hi)=wilson(a.point_hits,a.trials);
            let (seq,seq_se,seq_lo,seq_hi)=wilson(a.sequence_hits,a.trials);
            let (false_exclusion,false_se,false_lo,false_hi)=wilson(a.trials-a.sequence_hits,a.trials);
            let (promotion,prom_se,prom_lo,prom_hi)=wilson(a.promotion_hits,a.trials);
            let (width,width_se)=mean_se(a.width_sum,a.width_sq_sum,a.trials);
            let mean_lower=a.lower_sum/a.trials as f64;
            let mean_upper=a.upper_sum/a.trials as f64;
            let promotion_name=if result.true_value<PROMOTION_THRESHOLD{"false_promotion_probability"}else{"promotion_power"};
            for (metric,val,se,lo,hi) in [
                ("pointwise_coverage",point,point_se,point_lo,point_hi),
                ("sequence_coverage",seq,seq_se,seq_lo,seq_hi),
                ("sequence_false_exclusion_probability",false_exclusion,false_se,false_lo,false_hi),
                (promotion_name,promotion,prom_se,prom_lo,prom_hi),
            ]{
                let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",result.name,methods[method],look,metric,val,se,lo,hi);
            }
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_interval_width\t{:.8}\t{:.8}\t-\t-",result.name,methods[method],look,width,width_se);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_lower_endpoint\t{:.8}\t-\t-\t-",result.name,methods[method],look,mean_lower);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_upper_endpoint\t{:.8}\t-\t-\t-",result.name,methods[method],look,mean_upper);
            let reversed=a.reversed_hits as f64/a.trials as f64;
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\treversed_interval_probability\t{:.8}\t-\t-\t-",result.name,methods[method],look,reversed);
            let (elapsed,elapsed_se)=mean_se(a.elapsed_sum,a.elapsed_sq_sum,a.trials);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_method_elapsed_seconds\t{:.8}\t{:.8}\t-\t-",result.name,methods[method],look,elapsed,elapsed_se);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tpromotion_time_median_conditional\t{}\t-\t-\t-",result.name,methods[method],look,median(&a.promotion_times).map(|x|x.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tpromotion_time_p90_conditional\t{}\t-\t-\t-",result.name,methods[method],look,percentile90(&a.promotion_times).map(|x|x.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
            let _=writeln!(out,"ROOT_METRIC\t{}\t{}\t{}\tchecks\t{}\tfailures\t{}\tcollisions\t{}",result.name,look,methods[method],a.root_checks,a.root_failures,a.endpoint_collisions);
        }
    }
    let _=writeln!(out,"SYSTEM\t{}\tseconds\t{:.6}\tobservations\t{}\tobservations_per_second\t{:.3}",result.name,result.elapsed_seconds,result.observations,result.observations as f64/result.elapsed_seconds.max(1e-12));
}
fn benchmark(reps:usize,looks:&[usize])->Result<String,String>{
    let cases=cases();
    let mut out=String::from("record_type\tscenario\tmethod\tlook\tmetric\testimate\tstandard_error\tci_low\tci_high\n");
    let _=writeln!(out,"SEED_SCHEDULE\tbase_seed\t0x8b8b8b8b04500045\tformula\tBASE xor DGP_GROUP*D6E8FEB86659FD93 xor replicate*A0761D6478BD642F");
    let _=writeln!(out,"BENCHMARK_CONFIG\treplicates\t{reps}\tmax_horizon\t{}",looks.iter().max().unwrap_or(&0));
    for dgp in cases{
        let result=simulate_case(dgp,reps,looks)?;
        emit_case(&mut out,&result,looks);
    }
    Ok(out)
}
fn seed_smoke()->String{
    let mut rng=SplitMix64::new(0x8b8b_8b8b_0450_0045);
    let mut out=String::from("VIS_DVPC_045_SEED_SMOKE\n");
    for _ in 0..64{let _=writeln!(out,"{:016x}",rng.next_u64());}
    out
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let result=match args.get(1).map(String::as_str){
        Some("--benchmark")=>benchmark(REPLICATES,&LOOKS),
        Some("--smoke")=>benchmark(8,&[8]),
        Some("--seed-smoke")=>Ok(seed_smoke()),
        _=>Err("usage: vision-dvpc-045-predictor-sequential-cs --benchmark|--smoke|--seed-smoke".into())
    };
    match result{
        Ok(out)=>print!("{out}"),
        Err(err)=>{eprintln!("VIS_DVPC_045_ERROR\t{err}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn predictor_variants_share_the_same_registered_seed_group(){
        assert_eq!(seed_group("healthy_null_oracle"),seed_group("healthy_null_constant"));
        assert_eq!(seed_group("healthy_null_constant"),seed_group("healthy_null_reversed"));
        assert_eq!(seed_group("weak_alt_oracle"),seed_group("weak_alt_reversed"));
        assert_ne!(seed_group("healthy_null_oracle"),seed_group("healthy_alt_oracle"));
    }

    #[test]
    fn frozen_control_variate_remains_unbiased(){
        for dgp in cases(){
            let a=predictor::audit(&dgp).unwrap();
            assert!(close(a.adjusted_mean,a.true_value,1e-10));
        }
    }

    #[test]
    fn action_specific_constraints_hold_for_both_sides(){
        for dgp in cases(){
            let audit=predictor::audit(&dgp).unwrap();
            for reflected in [false,true]{
                let constraints=predictor::action_specific_constraints(&audit,reflected);
                let stats=predictor::QuadraticStats{a00:1.0,a01:0.1,a11:1.0,b0:0.2,b1:0.5};
                let optimum=predictor::optimize_support_polytope(stats,&constraints).unwrap();
                assert!(predictor::support_aware_feasible(optimum.bet,&audit,reflected));
                assert!(predictor::support_aware_corners_hold(optimum.bet,&audit,reflected));
                assert!(predictor::support_aware_minimum_factor(optimum.bet,&audit,reflected)>=0.5-WEALTH_TOL);
            }
        }
    }

    #[test]
    fn predictor_pair_is_nested_and_root_certificates_hold_on_smoke_stream(){
        let dgp=make_case("healthy_null_oracle",false,[0.2,0.8],0);
        let audit=predictor::audit(&dgp).unwrap();
        let mut pair=PredictorPair::new(&audit,32);
        let mut lower=0.0;
        let mut upper=1.0;
        for i in 0..32{
            let action=i%2;
            let reward=if action==0{if i%5==0{1.0}else{0.0}}else if i%3==0{0.0}else{1.0};
            let w=dgp.pi[action]/dgp.h[action];
            let c=w*dgp.predictor[action]-audit.q_pi;
            let interval=pair.update(predictor::Sample{w,r:reward,c}).unwrap();
            assert!(interval.lower+EPS>=lower);
            assert!(interval.upper<=upper+EPS);
            assert!(interval.lower<=interval.upper+EPS);
            assert_eq!(pair.diagnostics.root_failures,0);
            lower=interval.lower;upper=interval.upper;
        }
    }

    #[test]
    fn deterministic_paired_benchmark_replays(){
        let dgp=make_case("weak_null_oracle",true,[0.2,0.8],0);
        let a=simulate_case(dgp.clone(),5,&[8]).unwrap();
        let b=simulate_case(dgp,5,&[8]).unwrap();
        assert_eq!(a.aggregates[0][0].sequence_hits,b.aggregates[0][0].sequence_hits);
        assert_eq!(a.aggregates[1][0].sequence_hits,b.aggregates[1][0].sequence_hits);
        assert_eq!(a.aggregates[0][0].point_hits,b.aggregates[0][0].point_hits);
        assert_eq!(a.aggregates[1][0].point_hits,b.aggregates[1][0].point_hits);
        assert_eq!(a.root_failures,[0,0]);
    }
}
