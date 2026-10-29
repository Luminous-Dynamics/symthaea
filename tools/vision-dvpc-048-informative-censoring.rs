//! VIS-DVPC-048: deadline censoring, inverse-probability correction and identification.
//!
//! The censor indicator is finalized at a predeclared follow-up deadline.
//! On MAR fixtures, it is independent of reward conditional on context/action,
//! so IPCW scores are observable, bounded and unbiased. Informative (MNAR)
//! censoring and target-action positivity failure fail closed.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const REPLICATES:usize=1024;
const HORIZONS:[usize;3]=[128,512,4096];
const SMOKE_HORIZONS:[usize;2]=[16,32];
const ALPHA:f64=0.05;
const EPS:f64=1e-10;
const IPS_BOUND:f64=2.0;
const TARGET_ACTION:[usize;2]=[1,0];
const REWARD_MEAN:[[f64;2];2]=[[0.20,0.80],[0.75,0.30]];

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
fn bernoulli(rng:&mut SplitMix64,p:f64)->bool{rng.unit()<p}
fn close(a:f64,b:f64,tol:f64)->bool{(a-b).abs()<=tol*(1.0+a.abs().max(b.abs()))}

#[derive(Clone,Copy,Debug,PartialEq)]
enum CensorModel{IndependentByCell,RewardDependent}
#[derive(Clone,Copy,Debug)]
struct Fixture{
    name:&'static str,
    q:[[f64;2];2],
    model:CensorModel,
}
impl Fixture{
    fn no_censoring()->Self{Self{name:"no_censoring",q:[[1.0,1.0],[1.0,1.0]],model:CensorModel::IndependentByCell}}
    fn uniform_censoring()->Self{Self{name:"uniform_censoring",q:[[0.5,0.5],[0.5,0.5]],model:CensorModel::IndependentByCell}}
    fn cell_dependent()->Self{Self{name:"context_action_dependent_censoring",q:[[0.95,0.20],[0.25,0.90]],model:CensorModel::IndependentByCell}}
    fn positivity_failure()->Self{Self{name:"target_cell_never_matures",q:[[0.95,0.0],[0.25,0.90]],model:CensorModel::IndependentByCell}}
    fn outcome_dependent()->Self{Self{name:"reward_dependent_mnar_censoring",q:[[0.5,0.5],[0.5,0.5]],model:CensorModel::RewardDependent}}
}
fn fixtures()->Vec<Fixture>{vec![Fixture::no_censoring(),Fixture::uniform_censoring(),Fixture::cell_dependent(),Fixture::positivity_failure(),Fixture::outcome_dependent()]}
fn true_value()->f64{0.5*(REWARD_MEAN[0][TARGET_ACTION[0]]+REWARD_MEAN[1][TARGET_ACTION[1]])}
fn target_action(context:usize)->usize{TARGET_ACTION[context]}
fn target_q_min(f:Fixture)->f64{
    (0..2).map(|c|f.q[c][target_action(c)]).fold(f64::INFINITY,f64::min)
}
fn qualification(f:Fixture)->&'static str{
    if f.model==CensorModel::RewardDependent{"NOT_IDENTIFIED_MNAR"}
    else if target_q_min(f)<=0.0{"NOT_IDENTIFIED_POSITIVITY"}
    else{"IDENTIFIED_MAR"}
}
fn ipcw_bound(f:Fixture)->Option<f64>{
    if qualification(f)!="IDENTIFIED_MAR"{None}else{Some(IPS_BOUND/target_q_min(f))}
}
fn maturity_probability(f:Fixture,c:usize,a:usize,r:bool)->f64{
    match f.model{
        CensorModel::IndependentByCell=>f.q[c][a],
        CensorModel::RewardDependent=>if r{0.20}else{0.90},
    }
}

#[derive(Clone,Copy,Debug)]
struct Observation{context:usize,action:usize,reward:bool,mature:bool}
fn draw_observation(f:Fixture,rng:&mut SplitMix64)->Observation{
    let context=if rng.unit()<0.5{0}else{1};
    let action=if rng.unit()<0.5{0}else{1};
    let reward=bernoulli(rng,REWARD_MEAN[context][action]);
    let mature=bernoulli(rng,maturity_probability(f,context,action,reward));
    Observation{context,action,reward,mature}
}
fn ips_score(o:Observation)->f64{
    if o.action==target_action(o.context)&&o.reward{IPS_BOUND}else{0.0}
}
fn ipcw_score(f:Fixture,o:Observation)->Option<f64>{
    if qualification(f)!="IDENTIFIED_MAR"{return None;}
    if !o.mature{return Some(0.0);}
    if o.action!=target_action(o.context){return Some(0.0);}
    let q=f.q[o.context][o.action];
    if q<=0.0{return None;}
    Some(ips_score(o)/q)
}
fn available_case_wald(sum:f64,sum_sq:f64,n:usize)->(f64,f64,f64,f64){
    if n==0{return(0.0,1.0,1.0,0.5);}
    let mean=sum/n as f64;
    if n<2{return(0.0,1.0,1.0,mean);}
    let variance=((sum_sq-sum*sum/n as f64).max(0.0))/(n-1) as f64;
    let radius=1.959_963_984_540_054*(variance/n as f64).sqrt();
    let lower=(mean-radius).max(0.0).min(1.0);
    let upper=(mean+radius).max(0.0).min(1.0);
    (lower,upper,(upper-lower).max(0.0),mean)
}
#[derive(Clone,Copy,Debug)]
struct Interval{lower:f64,upper:f64}
impl Interval{
    fn full()->Self{Self{lower:0.0,upper:1.0}}
    fn width(self)->f64{(self.upper-self.lower).max(0.0)}
    fn covers(self,v:f64)->bool{self.lower<=self.upper+EPS&&self.lower<=v+EPS&&self.upper>=v-EPS}
    fn intersect(self,other:Self)->Self{Self{lower:self.lower.max(other.lower),upper:self.upper.min(other.upper)}}
}
fn ipcw_hoeffding_raw(sum:f64,n:usize,bound:f64)->Interval{
    if n==0{return Interval::full();}
    let t=n as f64;
    let delta=ALPHA/(2.0*t*(t+1.0));
    let mean=sum/t;
    let radius=bound*((1.0/delta).ln()/(2.0*t)).sqrt();
    Interval{lower:(mean-radius).max(0.0).min(1.0),upper:(mean+radius).max(0.0).min(1.0)}
}
fn ipcw_hoeffding_cs(sum:f64,n:usize,bound:f64,previous:Interval)->Interval{
    previous.intersect(ipcw_hoeffding_raw(sum,n,bound))
}
fn seed_for(rep:usize)->u64{
    0x8b8b_8b8b_0480_0048 ^ (rep as u64).wrapping_mul(0xA076_1D64_78BD_642F)
}
const COMMON_NAMES:[&str;13]=[
    "available_case_ips_estimate",
    "available_case_ips_bias",
    "available_case_ips_absolute_error",
    "available_case_ips_squared_error",
    "available_case_wald_coverage",
    "available_case_wald_width",
    "zero_filled_ips_estimate",
    "zero_filled_ips_bias",
    "zero_filled_ips_absolute_error",
    "zero_filled_ips_squared_error",
    "matured_fraction",
    "target_action_matured_fraction",
    "matured_sample_count",
];
const IPCW_NAMES:[&str;9]=[
    "ipcw_estimate",
    "ipcw_bias",
    "ipcw_absolute_error",
    "ipcw_squared_error",
    "ipcw_pointwise_hoeffding_coverage",
    "ipcw_cs_sequence_coverage",
    "ipcw_cs_width",
    "ipcw_effective_sample_size",
    "ipcw_score_bound",
];
#[derive(Clone,Copy,Debug,Default)]
struct MetricAgg{sum:f64,sum_sq:f64,n:usize}
impl MetricAgg{
    fn add(&mut self,x:f64){self.sum+=x;self.sum_sq+=x*x;self.n+=1;}
    fn mean(self)->f64{if self.n==0{0.0}else{self.sum/self.n as f64}}
    fn se(self)->f64{
        if self.n<2{return 0.0;}
        let n=self.n as f64;
        let var=((self.sum_sq-self.sum*self.sum/n).max(0.0))/(n-1.0);
        (var/n).sqrt()
    }
}
#[derive(Clone,Debug)]
struct ResultFixture{
    fixture:Fixture,
    true_value:f64,
    qualification:&'static str,
    bound:Option<f64>,
    common:Vec<Vec<MetricAgg>>,
    ipcw:Vec<Vec<MetricAgg>>,
    elapsed:f64,
}
fn simulate(f:Fixture,reps:usize,looks:&[usize])->Result<ResultFixture,String>{
    if reps==0||looks.is_empty()||looks.iter().any(|x|*x==0)||looks.windows(2).any(|x|x[0]>=x[1]){
        return Err("replicates and strictly increasing positive horizons are required".into());
    }
    let max_n=*looks.last().unwrap();
    let start=Instant::now();
    let mut common=vec![vec![MetricAgg::default();COMMON_NAMES.len()];looks.len()];
    let mut ipcw=vec![vec![MetricAgg::default();IPCW_NAMES.len()];looks.len()];
    let is_identified=qualification(f)=="IDENTIFIED_MAR";
    let bound=ipcw_bound(f);
    for rep in 0..reps{
        let mut rng=SplitMix64::new(seed_for(rep));
        let mut available_sum=0.0;let mut available_sq=0.0;let mut mature_count=0usize;
        let mut zero_fill_sum=0.0;let mut target_seen=0usize;let mut target_matured=0usize;
        let mut ipcw_sum=0.0;let mut ipcw_weight_sum=0.0;let mut ipcw_weight_sq_sum=0.0;
        let mut cs=Interval::full();let mut current_raw_cs=Interval::full();let mut cs_sequence_coverage=true;
        let mut look_i=0usize;
        for n in 1..=max_n{
            let o=draw_observation(f,&mut rng);
            let target=o.action==target_action(o.context);
            if target{target_seen+=1;}
            if o.mature{
                mature_count+=1;
                let x=ips_score(o);
                available_sum+=x;available_sq+=x*x;
                zero_fill_sum+=x;
                if target{target_matured+=1;}
            }
            if is_identified{
                let z=ipcw_score(f,o).ok_or("identified fixture unexpectedly produced unavailable IPCW score")?;
                ipcw_sum+=z;
                if o.mature&&target{
                    let w=IPS_BOUND/f.q[o.context][o.action];
                    ipcw_weight_sum+=w;ipcw_weight_sq_sum+=w*w;
                }
                current_raw_cs=ipcw_hoeffding_raw(ipcw_sum,n,bound.unwrap());
                cs=ipcw_hoeffding_cs(ipcw_sum,n,bound.unwrap(),cs);
                cs_sequence_coverage&=cs.covers(true_value());
            }
            if look_i<looks.len()&&n==looks[look_i]{
                let estimate=if mature_count==0{0.0}else{available_sum/mature_count as f64};
                let (lo,hi,width,wald_estimate)=available_case_wald(available_sum,available_sq,mature_count);
                let available_error=estimate-true_value();
                let zero_estimate=zero_fill_sum/n as f64;
                let zero_error=zero_estimate-true_value();
                let common_values=[
                    estimate,
                    available_error,
                    available_error.abs(),
                    available_error*available_error,
                    (lo<=true_value()+EPS&&hi>=true_value()-EPS) as u8 as f64,
                    width,
                    zero_estimate,
                    zero_error,
                    zero_error.abs(),
                    zero_error*zero_error,
                    mature_count as f64/n as f64,
                    if target_seen==0{0.0}else{target_matured as f64/target_seen as f64},
                    mature_count as f64,
                ];
                let _=wald_estimate;
                for i in 0..COMMON_NAMES.len(){common[look_i][i].add(common_values[i]);}
                if is_identified{
                    let ipcw_est=ipcw_sum/n as f64;
                    let ess=if ipcw_weight_sq_sum<=0.0{0.0}else{ipcw_weight_sum*ipcw_weight_sum/ipcw_weight_sq_sum};
                    let ipcw_error=ipcw_est-true_value();
                    let values=[
                        ipcw_est,
                        ipcw_error,
                        ipcw_error.abs(),
                        ipcw_error*ipcw_error,
                        current_raw_cs.covers(true_value()) as u8 as f64,
                        cs_sequence_coverage as u8 as f64,
                        cs.width(),
                        ess,
                        bound.unwrap(),
                    ];
                    for i in 0..IPCW_NAMES.len(){ipcw[look_i][i].add(values[i]);}
                }
                look_i+=1;
            }
        }
    }
    Ok(ResultFixture{fixture:f,true_value:true_value(),qualification:qualification(f),bound,common,ipcw,elapsed:start.elapsed().as_secs_f64()})
}
fn emit(out:&mut String,r:&ResultFixture,looks:&[usize]){
    let qmin=target_q_min(r.fixture);
    let model=if r.fixture.model==CensorModel::RewardDependent{"reward_dependent"}else{"independent_cell"};
    let _=writeln!(out,"FIXTURE\t{}\tqualification\t{}\ttrue_value\t{:.8}\tcensor_model\t{}\tmin_target_maturity_probability\t{:.8}\tipcw_score_bound\t{}",
        r.fixture.name,r.qualification,r.true_value,model,qmin,r.bound.map(|x|format!("{x:.8}")).unwrap_or_else(||"SUPPRESSED".into()));
    for (li,n) in looks.iter().copied().enumerate(){
        for (i,name) in COMMON_NAMES.iter().enumerate(){
            let a=r.common[li][i];
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{:.8}\t{:.8}",r.fixture.name,n,name,a.mean(),a.se());
        }
        if r.qualification=="IDENTIFIED_MAR"{
            for (i,name) in IPCW_NAMES.iter().enumerate(){
                let a=r.ipcw[li][i];
                let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{:.8}\t{:.8}",r.fixture.name,n,name,a.mean(),a.se());
            }
        }
    }
    let _=writeln!(out,"SYSTEM\t{}\tseconds\t{:.6}\tqualification\t{}",r.fixture.name,r.elapsed,r.qualification);
}
fn benchmark(reps:usize,looks:&[usize])->Result<String,String>{
    let mut out=String::from("record_type\tfixture\thorizon\tmetric\testimate\tstandard_error\n");
    let _=writeln!(out,"CONFIG\tseed\t0x8b8b8b8b04800048\treplicates\t{}\ttruth\t{:.8}",reps,true_value());
    for f in fixtures(){emit(&mut out,&simulate(f,reps,looks)?,looks);}
    Ok(out)
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let result=match args.get(1).map(String::as_str){
        Some("--benchmark")=>benchmark(REPLICATES,&HORIZONS),
        Some("--smoke")=>benchmark(8,&SMOKE_HORIZONS),
        _=>Err("usage: vision-dvpc-048-informative-censoring --benchmark|--smoke".into())
    };
    match result{
        Ok(out)=>print!("{out}"),
        Err(e)=>{eprintln!("VIS_DVPC_048_ERROR\t{e}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn fixed_truth_and_target_policy_are_frozen(){
        assert!(close(true_value(),0.775,1e-12));
        assert_eq!(target_action(0),1);
        assert_eq!(target_action(1),0);
    }
    #[test]
    fn censoring_controls_have_explicit_identification_status(){
        assert_eq!(qualification(Fixture::no_censoring()),"IDENTIFIED_MAR");
        assert_eq!(qualification(Fixture::uniform_censoring()),"IDENTIFIED_MAR");
        assert_eq!(qualification(Fixture::cell_dependent()),"IDENTIFIED_MAR");
        assert_eq!(qualification(Fixture::positivity_failure()),"NOT_IDENTIFIED_POSITIVITY");
        assert_eq!(qualification(Fixture::outcome_dependent()),"NOT_IDENTIFIED_MNAR");
        assert!(ipcw_bound(Fixture::positivity_failure()).is_none());
        assert!(ipcw_bound(Fixture::outcome_dependent()).is_none());
    }
    #[test]
    fn analytical_ipcw_expectation_is_true_value_under_all_mar_fixtures(){
        for f in [Fixture::no_censoring(),Fixture::uniform_censoring(),Fixture::cell_dependent()]{
            let mut expectation=0.0;
            for c in 0..2{
                let a=target_action(c);
                expectation+=0.25*f.q[c][a]*(IPS_BOUND*REWARD_MEAN[c][a]/f.q[c][a]);
            }
            assert!(close(expectation,true_value(),1e-12));
        }
    }
    #[test]
    fn available_case_is_biased_by_context_action_dependent_maturity(){
        let f=Fixture::cell_dependent();
        let mut numerator=0.0;let mut denominator=0.0;
        for c in 0..2{
            for a in 0..2{
                denominator+=0.25*f.q[c][a];
                if a==target_action(c){numerator+=0.25*f.q[c][a]*IPS_BOUND*REWARD_MEAN[c][a];}
            }
        }
        let complete_case=numerator/denominator;
        assert!((complete_case-true_value()).abs()>0.2);
    }
    #[test]
    fn ipc_weights_stay_inside_registered_bound(){
        for f in [Fixture::no_censoring(),Fixture::uniform_censoring(),Fixture::cell_dependent()]{
            let b=ipcw_bound(f).unwrap();
            for c in 0..2{
                let a=target_action(c);
                assert!(IPS_BOUND/f.q[c][a]<=b+EPS);
            }
        }
    }
    #[test]
    fn every_simulated_ipcw_score_respects_its_registered_bound(){
        for f in [Fixture::no_censoring(),Fixture::uniform_censoring(),Fixture::cell_dependent()]{
            let b=ipcw_bound(f).unwrap();
            let mut rng=SplitMix64::new(seed_for(31));
            for _ in 0..10_000{
                let o=draw_observation(f,&mut rng);
                let z=ipcw_score(f,o).unwrap();
                assert!(z>=0.0&&z<=b+EPS,"score {z} exceeded bound {b} for {}",f.name);
            }
        }
    }

    #[test]
    fn zero_filled_expectation_is_downward_biased_under_any_uniform_censoring(){
        let f=Fixture::uniform_censoring();
        let mut expected_zero=0.0;
        for context in 0..2{
            let action=target_action(context);
            expected_zero+=0.25*f.q[context][action]*IPS_BOUND*REWARD_MEAN[context][action];
        }
        assert!(close(expected_zero,true_value()*0.5,1e-12));
        let no_censor=Fixture::no_censoring();
        let mut expected_no_censor=0.0;
        for context in 0..2{
            let action=target_action(context);
            expected_no_censor+=0.25*no_censor.q[context][action]*IPS_BOUND*REWARD_MEAN[context][action];
        }
        assert!(close(expected_no_censor,true_value(),1e-12));
    }

    #[test]
    fn confidence_sequence_is_nested_and_uses_full_assigned_horizon(){
        let f=Fixture::cell_dependent();
        let b=ipcw_bound(f).unwrap();
        let mut rng=SplitMix64::new(17);
        let mut sum=0.0;let mut cs=Interval::full();let mut old=cs;
        for n in 1..=256{
            let o=draw_observation(f,&mut rng);
            sum+=ipcw_score(f,o).unwrap();
            old=cs;cs=ipcw_hoeffding_cs(sum,n,b,cs);
            assert!(cs.lower>=old.lower-EPS);
            assert!(cs.upper<=old.upper+EPS);
            assert!(cs.lower>=0.0&&cs.upper<=1.0);
        }
    }
    #[test]
    fn deterministic_replay_repeats_statistics(){
        let f=Fixture::cell_dependent();
        let a=simulate(f,8,&[16,32]).unwrap();
        let b=simulate(f,8,&[16,32]).unwrap();
        assert_eq!(a.common[1][0].mean(),b.common[1][0].mean());
        assert_eq!(a.ipcw[1][0].mean(),b.ipcw[1][0].mean());
    }
}
