//! VIS-DVPC-047: reward alignment, online learnability, headroom and delay diagnostics.
//!
//! Synthetic ground-truth benchmark. Proxy rewards drive a tabular contextual
//! UCB learner; the north-star outcome determines business value. A separate
//! uniform logged sample estimates the final greedy proxy policy using IPS.
//! No real-data or production claim follows from these fixtures.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const CONTEXTS:usize=2;
const ARMS:usize=4;
const REPLICATES:usize=1024;
const HORIZON:usize=4096;
const LOOKS:[usize;3]=[128,512,4096];
const SEED:u64=0x8b8b_8b8b_0470_0047;
const EPS:f64=1e-10;
const IPS_MAX_WEIGHT:f64=4.0;

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
fn bernoulli(rng:&mut SplitMix64,p:f64)->f64{if rng.unit()<p{1.0}else{0.0}}
fn close(a:f64,b:f64,tol:f64)->bool{(a-b).abs()<=tol*(1.0+a.abs().max(b.abs()))}

#[derive(Clone,Copy,Debug)]
enum ProxyKind{AlignedDense,AlignedSparse,MisalignedDense}
#[derive(Clone,Copy,Debug)]
enum DelayKind{Short,HeavyTail}
#[derive(Clone,Copy,Debug)]
struct Fixture{
    name:&'static str,
    north_star:[[f64;ARMS];CONTEXTS],
    proxy_kind:ProxyKind,
    delay:DelayKind,
}
impl Fixture{
    fn aligned_dense()->Self{Self{name:"aligned_dense_personalization",north_star:[[0.35,0.55,0.75,0.65],[0.80,0.60,0.40,0.55]],proxy_kind:ProxyKind::AlignedDense,delay:DelayKind::Short}}
    fn aligned_sparse()->Self{Self{name:"aligned_sparse_personalization",north_star:[[0.35,0.55,0.75,0.65],[0.80,0.60,0.40,0.55]],proxy_kind:ProxyKind::AlignedSparse,delay:DelayKind::Short}}
    fn misaligned_dense()->Self{Self{name:"misaligned_dense_personalization",north_star:[[0.35,0.55,0.75,0.65],[0.80,0.60,0.40,0.55]],proxy_kind:ProxyKind::MisalignedDense,delay:DelayKind::Short}}
    fn zero_headroom()->Self{Self{name:"zero_personalization_headroom",north_star:[[0.35,0.55,0.75,0.65],[0.35,0.55,0.75,0.65]],proxy_kind:ProxyKind::AlignedDense,delay:DelayKind::Short}}
    fn delayed_sparse()->Self{Self{name:"aligned_sparse_delayed_outcome",north_star:[[0.35,0.55,0.75,0.65],[0.80,0.60,0.40,0.55]],proxy_kind:ProxyKind::AlignedSparse,delay:DelayKind::HeavyTail}}
}
fn fixtures()->Vec<Fixture>{vec![Fixture::aligned_dense(),Fixture::aligned_sparse(),Fixture::misaligned_dense(),Fixture::zero_headroom(),Fixture::delayed_sparse()]}
fn proxy_mean(f:Fixture,context:usize,arm:usize)->f64{
    let value=f.north_star[context][arm];
    match f.proxy_kind{
        ProxyKind::AlignedDense=>value,
        ProxyKind::AlignedSparse=>0.12*value,
        ProxyKind::MisalignedDense=>1.0-value,
    }
}
fn argmax(values:&[f64;ARMS])->usize{
    (0..ARMS).max_by(|a,b|values[*a].total_cmp(&values[*b])).unwrap()
}
fn best_action(f:Fixture,context:usize,proxy:bool)->usize{
    if proxy{
        argmax(&std::array::from_fn(|a|proxy_mean(f,context,a)))
    }else{argmax(&f.north_star[context])}
}
fn oracle_contextual_value(f:Fixture,proxy:bool)->f64{
    0.5*( (0..ARMS).map(|a|if proxy{proxy_mean(f,0,a)}else{f.north_star[0][a]}).fold(f64::NEG_INFINITY,f64::max)
        +(0..ARMS).map(|a|if proxy{proxy_mean(f,1,a)}else{f.north_star[1][a]}).fold(f64::NEG_INFINITY,f64::max))
}
fn global_best_value(f:Fixture)->f64{
    (0..ARMS).map(|a|0.5*(f.north_star[0][a]+f.north_star[1][a])).fold(f64::NEG_INFINITY,f64::max)
}
fn global_best_action(f:Fixture)->usize{
    (0..ARMS).max_by(|a,b|{
        (f.north_star[0][*a]+f.north_star[1][*a]).total_cmp(&(f.north_star[0][*b]+f.north_star[1][*b]))
    }).unwrap()
}
fn personalization_headroom(f:Fixture)->f64{oracle_contextual_value(f,false)-global_best_value(f)}
fn alignment_pairwise_accuracy(f:Fixture)->f64{
    let mut hits=0.0;let mut pairs=0.0;
    for c in 0..CONTEXTS{
        for a in 0..ARMS{
            for b in (a+1)..ARMS{
                let ns=f.north_star[c][a]-f.north_star[c][b];
                let proxy=proxy_mean(f,c,a)-proxy_mean(f,c,b);
                if ns.abs()<=EPS||proxy.abs()<=EPS{continue;}
                pairs+=1.0;
                if ns.signum()==proxy.signum(){hits+=1.0;}
            }
        }
    }
    if pairs==0.0{0.0}else{hits/pairs}
}
fn fixture_delay_name(f:Fixture)->&'static str{
    match f.delay{DelayKind::Short=>"fixed_8_step",DelayKind::HeavyTail=>"mixture_4_or_512_step"}
}
fn delay_steps(f:Fixture,draw:f64)->usize{
    match f.delay{DelayKind::Short=>{let _=draw;8},DelayKind::HeavyTail=>if draw<0.70{4}else{512}}
}

#[derive(Clone,Copy,Debug)]
struct LoggedObservation{context:usize,action:usize,reward:f64}
fn logged_observation(rng:&mut SplitMix64,f:Fixture)->LoggedObservation{
    let context=if rng.unit()<0.5{0}else{1};
    let action=(rng.unit()*ARMS as f64).floor() as usize;
    let reward=bernoulli(rng,f.north_star[context][action]);
    LoggedObservation{context,action,reward}
}
fn choose_ucb(context:usize,context_count:usize,counts:&[[usize;ARMS];CONTEXTS],sums:&[[f64;ARMS];CONTEXTS])->usize{
    for arm in 0..ARMS{if counts[context][arm]==0{return arm;}}
    let logterm=(context_count.max(1) as f64).ln();
    (0..ARMS).max_by(|a,b|{
        let score=|arm:usize|sums[context][arm]/counts[context][arm] as f64
            +(2.0*logterm/counts[context][arm] as f64).sqrt();
        score(*a).total_cmp(&score(*b))
    }).unwrap()
}
fn estimated_greedy_actions(counts:&[[usize;ARMS];CONTEXTS],sums:&[[f64;ARMS];CONTEXTS])->[usize;CONTEXTS]{
    std::array::from_fn(|c|{
        argmax(&std::array::from_fn(|a|if counts[c][a]==0{-1.0}else{sums[c][a]/counts[c][a] as f64}))
    })
}
fn policy_code(policy:[usize;CONTEXTS])->usize{policy[0]*ARMS+policy[1]}
fn policy_value(f:Fixture,policy:[usize;CONTEXTS])->f64{
    0.5*(f.north_star[0][policy[0]]+f.north_star[1][policy[1]])
}
fn ips_interval(sum:f64,sum_sq:f64,n:usize)->(f64,f64,f64){
    if n<2{return(0.0,1.0,1.0);}
    let mean=sum/n as f64;
    let variance=((sum_sq-sum*sum/n as f64).max(0.0))/(n-1) as f64;
    let radius=1.959_963_984_540_054*(variance/n as f64).sqrt();
    let lower=(mean-radius).max(0.0).min(1.0);
    let upper=(mean+radius).max(0.0).min(1.0);
    (lower,upper,(upper-lower).max(0.0))
}
fn seed_for(replicate:usize)->u64{
    SEED ^ (replicate as u64).wrapping_mul(0xA076_1D64_78BD_642F)
}
fn evaluation_seed_for(replicate:usize)->u64{
    SEED.rotate_left(17) ^ 0xE1A1_0470_0000_0001
        ^ (replicate as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

const METRIC_NAMES:[&str;12]=[
    "online_north_star_value",
    "online_north_star_regret_per_decision",
    "online_proxy_reward_value",
    "online_proxy_regret_per_decision",
    "proxy_policy_identification_rate",
    "north_star_policy_identification_rate",
    "online_personalization_premium_vs_global",
    "matured_outcome_fraction",
    "matured_north_star_mean",
    "batch_ips_absolute_error_for_learned_greedy_policy",
    "batch_ips_pointwise_wald_coverage",
    "batch_ips_wald_interval_width",
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
struct FixtureResult{
    name:&'static str,delay:&'static str,
    north_star_contextual_value:f64,global_value:f64,headroom:f64,
    proxy_contextual_value:f64,alignment:f64,metrics:Vec<Vec<MetricAgg>>,
    elapsed_seconds:f64,replicates:usize,
}
fn snapshot_metrics(f:Fixture, t:usize,
    counts:&[[usize;ARMS];CONTEXTS],proxy_sums:&[[f64;ARMS];CONTEXTS],
    ns_sum:f64,proxy_sum:f64,ns_regret:f64,proxy_regret:f64,
    matured_count:usize,matured_sum:f64,
    ips_sum:&[f64;ARMS*ARMS],
    ips_sq:&[f64;ARMS*ARMS]) -> [f64;12]
{
    let greedy=estimated_greedy_actions(counts,proxy_sums);
    let proxy_best=[best_action(f,0,true),best_action(f,1,true)];
    let ns_best=[best_action(f,0,false),best_action(f,1,false)];
    let proxy_hits=(0..CONTEXTS).filter(|c|greedy[*c]==proxy_best[*c]).count() as f64/CONTEXTS as f64;
    let ns_hits=(0..CONTEXTS).filter(|c|greedy[*c]==ns_best[*c]).count() as f64/CONTEXTS as f64;
    let learned_true=policy_value(f,greedy);
    let code=policy_code(greedy);
    let offset=code;
    // All 16 deterministic context-to-action policies share the same independent
    // uniform evaluation log. The flattened index is policy_code * 1.
    let ips_estimate=ips_sum[offset]/t as f64;
    let (ips_lower,ips_upper,ips_width)=ips_interval(ips_sum[offset],ips_sq[offset],t);
    let ips_covered=(ips_lower<=learned_true+EPS&&ips_upper>=learned_true-EPS) as u8 as f64;
    [
        ns_sum/t as f64,
        ns_regret/t as f64,
        proxy_sum/t as f64,
        proxy_regret/t as f64,
        proxy_hits,
        ns_hits,
        ns_sum/t as f64-global_best_value(f),
        matured_count as f64/t as f64,
        if matured_count==0{0.0}else{matured_sum/matured_count as f64},
        (ips_estimate-learned_true).abs(),
        ips_covered,
        ips_width,
    ]
}

fn simulate_fixture(f:Fixture,reps:usize,looks:&[usize])->Result<FixtureResult,String>{
    if reps==0||looks.is_empty()||looks.windows(2).any(|x|x[0]>=x[1])||looks.iter().any(|x|*x==0){
        return Err("replicate count and look schedule must be positive and increasing".into());
    }
    let max_look=*looks.last().unwrap();
    let started=Instant::now();
    let mut metrics=vec![vec![MetricAgg::default();METRIC_NAMES.len()];looks.len()];
    for rep in 0..reps{
        let mut rng=SplitMix64::new(seed_for(rep));
        let mut eval_rng=SplitMix64::new(evaluation_seed_for(rep));
        let eval_log:Vec<LoggedObservation>=(0..max_look).map(|_|logged_observation(&mut eval_rng,f)).collect();
        let mut counts=[[0usize;ARMS];CONTEXTS];
        let mut context_counts=[0usize;CONTEXTS];
        let mut proxy_sums=[[0.0_f64;ARMS];CONTEXTS];
        let mut ns_sum=0.0;let mut proxy_sum=0.0;let mut ns_regret=0.0;let mut proxy_regret=0.0;
        let mut matured_count=0usize;let mut matured_sum=0.0;
        // Store maturity counts/sums rather than one Vec allocation per
        // decision. This keeps the 5x1,024x4,096 sweep allocation-light.
        let mut maturity_count=vec![0usize;max_look+513];
        let mut maturity_sum=vec![0.0_f64;max_look+513];
        let mut ips_sum=[0.0_f64;ARMS*ARMS];
        let mut ips_sq=[0.0_f64;ARMS*ARMS];
        let mut look_i=0usize;
        for t in 1..=max_look{
            let obs=eval_log[t-1];
            for code in 0..ARMS*ARMS{
                let a0=code/ARMS;
                let a1=code%ARMS;
                let target=if obs.context==0{a0}else{a1};
                let score=if obs.action==target{IPS_MAX_WEIGHT*obs.reward}else{0.0};
                ips_sum[code]+=score;ips_sq[code]+=score*score;
            }

            let context=if rng.unit()<0.5{0}else{1};
            context_counts[context]+=1;
            let action=choose_ucb(context,context_counts[context]-1,&counts,&proxy_sums);
            let p_proxy=proxy_mean(f,context,action);
            let p_ns=f.north_star[context][action];
            let proxy_reward=bernoulli(&mut rng,p_proxy);
            let ns_reward=bernoulli(&mut rng,p_ns);
            let delay_draw=rng.unit();
            let delay=delay_steps(f,delay_draw);
            let mature_at=t+delay;
            if mature_at<maturity_count.len(){
                maturity_count[mature_at]+=1;
                maturity_sum[mature_at]+=ns_reward;
            }
            matured_count+=maturity_count[t];
            matured_sum+=maturity_sum[t];

            counts[context][action]+=1;
            proxy_sums[context][action]+=proxy_reward;
            ns_sum+=p_ns;proxy_sum+=p_proxy;
            ns_regret+=f.north_star[context][best_action(f,context,false)]-p_ns;
            proxy_regret+=proxy_mean(f,context,best_action(f,context,true))-p_proxy;

            if t==looks.get(look_i).copied().unwrap_or(usize::MAX){
                let snapshot=snapshot_metrics(f,t,&counts,&proxy_sums,ns_sum,proxy_sum,ns_regret,proxy_regret,
                    matured_count,matured_sum,&ips_sum,&ips_sq);
                for i in 0..METRIC_NAMES.len(){metrics[look_i][i].add(snapshot[i]);}
                look_i+=1;
            }
        }
        if look_i!=looks.len(){return Err("not all registered looks were reached".into());}
    }
    Ok(FixtureResult{
        name:f.name,delay:fixture_delay_name(f),north_star_contextual_value:oracle_contextual_value(f,false),
        global_value:global_best_value(f),headroom:personalization_headroom(f),
        proxy_contextual_value:oracle_contextual_value(f,true),alignment:alignment_pairwise_accuracy(f),
        metrics,elapsed_seconds:started.elapsed().as_secs_f64(),replicates:reps,
    })
}

fn emit(out:&mut String,r:&FixtureResult,looks:&[usize]){
    use std::fmt::Write as _;
    let _=writeln!(out,"FIXTURE\t{}\tdelay\t{}\treplicates\t{}\tnorth_star_contextual_value\t{:.8}\tbest_global_value\t{:.8}\tpersonalization_headroom\t{:.8}\tproxy_contextual_value\t{:.8}\tpairwise_alignment_accuracy\t{:.8}",
        r.name,r.delay,r.replicates,r.north_star_contextual_value,r.global_value,r.headroom,r.proxy_contextual_value,r.alignment);
    for (li,look) in looks.iter().copied().enumerate(){
        for i in 0..METRIC_NAMES.len(){
            let a=r.metrics[li][i];
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{:.8}\t{:.8}",r.name,look,METRIC_NAMES[i],a.mean(),a.se());
        }
    }
    let _=writeln!(out,"SYSTEM\t{}\tseconds\t{:.6}\treplicates\t{}",r.name,r.elapsed_seconds,r.replicates);
}
fn benchmark(reps:usize,looks:&[usize])->Result<String,String>{
    let mut out=String::from("record_type\tfixture\tlook\tmetric\testimate\tstandard_error\n");
    let _=writeln!(out,"CONFIG\tseed\t0x8b8b8b8b04700047\treplicates\t{}\thorizon\t{}",reps,looks.iter().max().unwrap_or(&0));
    for f in fixtures(){let r=simulate_fixture(f,reps,looks)?;emit(&mut out,&r,looks);}
    Ok(out)
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let result=match args.get(1).map(String::as_str){
        Some("--benchmark")=>benchmark(REPLICATES,&LOOKS),
        Some("--smoke")=>benchmark(8,&[16,32]),
        _=>Err("usage: vision-dvpc-047-reward-alignment --benchmark|--smoke".into())
    };
    match result{
        Ok(s)=>print!("{s}"),
        Err(e)=>{eprintln!("VIS_DVPC_047_ERROR\t{e}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn proxy_alignment_fixtures_have_distinct_expected_rank_signals(){
        assert!(alignment_pairwise_accuracy(Fixture::aligned_dense())>0.99);
        assert!(alignment_pairwise_accuracy(Fixture::aligned_sparse())>0.99);
        assert!(alignment_pairwise_accuracy(Fixture::misaligned_dense())<0.01);
    }
    #[test]
    fn zero_personalization_headroom_is_exactly_zero(){
        let f=Fixture::zero_headroom();
        assert!(close(personalization_headroom(f),0.0,1e-12));
        assert!(close(global_best_value(f),oracle_contextual_value(f,false),1e-12));
    }
    #[test]
    fn sparse_proxy_preserves_alignment_but_scales_signal(){
        let dense=Fixture::aligned_dense();let sparse=Fixture::aligned_sparse();
        assert!(close(alignment_pairwise_accuracy(dense),alignment_pairwise_accuracy(sparse),1e-12));
        assert!(proxy_mean(sparse,0,2)<proxy_mean(dense,0,2));
        assert_eq!(best_action(dense,0,true),best_action(sparse,0,true));
    }
    #[test]
    fn uniform_ope_score_is_bounded_and_targets_the_fixed_policy_value(){
        let f=Fixture::aligned_dense();
        let pol=[best_action(f,0,true),best_action(f,1,true)];
        for c in 0..2{
            for a in 0..ARMS{
                let w=if a==pol[c]{IPS_MAX_WEIGHT}else{0.0};
                assert!((0.0..=IPS_MAX_WEIGHT).contains(&(w*f.north_star[c][a])));
            }
        }
        assert!(close(policy_value(f,pol),0.5*(f.north_star[0][pol[0]]+f.north_star[1][pol[1]]),1e-12));
    }
    #[test]
    fn zero_headroom_never_exceeds_best_global_value_in_expectation(){
        let f=Fixture::zero_headroom();
        let r=simulate_fixture(f,16,&[16,32]).unwrap();
        for look in 0..2{
            let online_value=r.metrics[look][0].mean();
            assert!(online_value<=global_best_value(f)+1e-9);
            assert!(close(r.headroom,0.0,1e-12));
        }
    }
    #[test]
    fn deterministic_seed_replays(){
        let f=Fixture::aligned_sparse();
        let a=simulate_fixture(f,4,&[16,32]).unwrap();
        let b=simulate_fixture(f,4,&[16,32]).unwrap();
        assert_eq!(a.metrics[1][0].mean(),b.metrics[1][0].mean());
        assert_eq!(a.metrics[1][0].se(),b.metrics[1][0].se());
    }
}
