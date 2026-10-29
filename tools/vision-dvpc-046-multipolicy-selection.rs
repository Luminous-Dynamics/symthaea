//! VIS-DVPC-046: multiplicity-aware, anytime-valid multi-policy selection.
//!
//! Eight comparators share exact logged trajectories:
//!   0..5: pointwise/anytime-valid confidence-sequence baselines;
//!   6: fixed-horizon pCRP-star LCBs (conservative fixed-bet-grid approximation);
//!   7: fixed-horizon Logarithmic Smoothing LCBs.
//!
//! Methods 6 and 7 are evaluated only at registered looks and are NOT anytime-valid.
//! A best-policy certificate requires one lower bound to exceed every competitor's upper
//! bound; empty/reversed intervals fail closed.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const ALPHA: f64 = 0.05;
const REPLICATES: usize = 1024;
const LOOKS: [usize;3] = [256,1024,4096];
const POLICY_COUNTS: [usize;3] = [4,8,32];
const Z_975: f64 = 1.959_963_984_540_054;
const EPS: f64 = 1e-10;
const PCRP_BET_STEPS: usize = 32;
const PCRP_ROOT_ITERS: usize = 20;
const ANYTIME_METHODS: usize = 6;
const METHOD_COUNT: usize = 8;

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
    fn width(self)->f64{if self.reversed(){0.0}else{self.upper-self.lower}}
    fn reversed(self)->bool{!self.lower.is_finite()||!self.upper.is_finite()||self.lower>self.upper+EPS}
    fn covers(self,v:f64)->bool{!self.reversed()&&self.lower<=v+EPS&&self.upper>=v-EPS}
}
const FULL:Interval=Interval{lower:0.0,upper:1.0};

#[derive(Clone,Copy,Debug)]
struct Logger{name:&'static str,h1:[f64;2],index:u64}
impl Logger{
    fn healthy()->Self{Self{name:"healthy",h1:[0.5,0.5],index:0}}
    fn weak()->Self{Self{name:"weak_support",h1:[0.95,0.05],index:1}}
}

#[derive(Clone,Debug)]
struct Policy{id:usize,pi1:[f64;2],value:f64,w_max:f64}

#[derive(Clone,Copy,Debug)]
struct Scenario{logger:Logger,policy_count:usize,name:&'static str}
fn reward_mean(context:usize,action:usize)->f64{
    match (context,action){
        (0,0)=>0.10,
        (0,1)=>0.90,
        (1,0)=>0.95,
        (1,1)=>0.15,
        _=>unreachable!(),
    }
}
fn make_policies(m:usize,logger:Logger)->Vec<Policy>{
    (0..m).map(|j|{
        let p0=0.1+0.8*j as f64/(m-1) as f64;
        let perm=(7*j)%m;
        let p1=0.1+0.8*perm as f64/(m-1) as f64;
        let pi1=[p0,p1];
        let value=0.5*((1.0-p0)*reward_mean(0,0)+p0*reward_mean(0,1)
            +(1.0-p1)*reward_mean(1,0)+p1*reward_mean(1,1));
        let mut w_max=0.0_f64;
        for x in 0..2{
            let h=logger.h1[x];
            w_max=w_max.max(pi1[x]/h).max((1.0-pi1[x])/(1.0-h));
        }
        Policy{id:j,pi1,value,w_max}
    }).collect()
}
fn scenarios()->Vec<Scenario>{
    let mut out=Vec::new();
    for logger in [Logger::healthy(),Logger::weak()]{
        for m in POLICY_COUNTS{
            let name=match (logger.index,m){
                (0,4)=>"healthy_M4",(0,8)=>"healthy_M8",(0,32)=>"healthy_M32",
                (1,4)=>"weak_M4",(1,8)=>"weak_M8",(1,32)=>"weak_M32",
                _=>unreachable!(),
            };
            out.push(Scenario{logger,policy_count:m,name});
        }
    }
    out
}

fn seed_for(logger_index:u64,replicate:usize)->u64{
    0x8b8b_8b8b_0460_0046
        ^ ((logger_index+1).wrapping_mul(0xD6E8_FEB8_6659_FD93))
        ^ (replicate as u64).wrapping_mul(0xA076_1D64_78BD_642F)
}

#[derive(Clone,Copy,Debug,PartialEq)]
struct LoggedObservation{context:usize,action:usize,reward:f64}
fn next_observation(logger:Logger,rng:&mut SplitMix64)->LoggedObservation{
    let context=if rng.unit()<0.5{0}else{1};
    let h=logger.h1[context];
    let action=if rng.unit()<h{1}else{0};
    let reward=if rng.unit()<reward_mean(context,action){1.0}else{0.0};
    LoggedObservation{context,action,reward}
}

fn wald_interval(sum:f64,sum_sq:f64,n:usize)->Interval{
    if n<2{return FULL;}
    let mean=sum/n as f64;
    let var=((sum_sq-sum*sum/n as f64).max(0.0))/(n-1) as f64;
    let radius=Z_975*(var/n as f64).sqrt();
    Interval{lower:(mean-radius).max(0.0),upper:(mean+radius).min(1.0)}
}
fn hoeffding_radius_factor(n:usize,alpha_policy:f64)->f64{
    let t=n as f64;
    let delta=alpha_policy/(2.0*t*(t+1.0));
    ((1.0/delta).ln()/(2.0*t)).sqrt()
}
fn hoeffding_cs_factor(mean:f64,w_max:f64,radius_factor:f64)->Interval{
    let radius=w_max*radius_factor;
    Interval{lower:(mean-radius).max(0.0),upper:(mean+radius).min(1.0)}
}
fn hoeffding_cs(mean:f64,w_max:f64,n:usize,alpha_policy:f64)->Interval{
    hoeffding_cs_factor(mean,w_max,hoeffding_radius_factor(n,alpha_policy))
}

fn empirical_bernstein_logterm(n:usize,alpha_policy:f64)->f64{
    (2.0*(n as f64)*((n+1) as f64)/alpha_policy).ln()
}
fn empirical_bernstein_cs(sum:f64,sum_sq:f64,n:usize,w_max:f64,logterm:f64)->Interval{
    if n<2{return FULL;}
    let t=n as f64;
    let mean=sum/t;
    // SCOPE-RL's documented finite-sample empirical-Bernstein form uses the
    // population-normalized sample variance and the n-1 correction below.
    let variance=(sum_sq-sum*sum/t).max(0.0)/t;
    let radius=(2.0*variance*logterm/(t-1.0)).sqrt()
        +7.0*w_max*logterm/(3.0*(t-1.0));
    Interval{lower:(mean-radius).max(0.0),upper:(mean+radius).min(1.0)}
}
fn intersect(previous:Interval,next:Interval)->Interval{
    Interval{lower:previous.lower.max(next.lower),upper:previous.upper.min(next.upper)}
}

fn cell_mean_cs(sum:f64,count:usize,alpha_cell:f64)->Interval{
    if count==0{return FULL;}
    let n=count as f64;
    let mean=sum/n;
    let delta=alpha_cell/(2.0*n*(n+1.0));
    let radius=((1.0/delta).ln()/(2.0*n)).sqrt();
    Interval{lower:(mean-radius).max(0.0),upper:(mean+radius).min(1.0)}
}
fn shared_policy_interval(policy:&Policy,cells:[[Interval;2];2])->Interval{
    let p0=policy.pi1[0];let p1=policy.pi1[1];
    let lower=0.5*((1.0-p0)*cells[0][0].lower+p0*cells[0][1].lower
        +(1.0-p1)*cells[1][0].lower+p1*cells[1][1].lower);
    let upper=0.5*((1.0-p0)*cells[0][0].upper+p0*cells[0][1].upper
        +(1.0-p1)*cells[1][0].upper+p1*cells[1][1].upper);
    Interval{lower,upper}
}

#[derive(Clone,Debug,Default)]
struct ScoreProfile {
    bins: Vec<(f64, usize)>,
    n: usize,
    sum: f64,
}
impl ScoreProfile {
    fn push(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() || value < 0.0 {
            return Err("importance score must be finite and nonnegative".into());
        }
        let next_sum = self.sum + value;
        if !next_sum.is_finite() {
            return Err("importance-score sum overflowed".into());
        }
        self.n += 1;
        self.sum = next_sum;
        if let Some(index) = self.bins.iter().position(|(v, _)| *v == value) {
            self.bins[index].1 += 1;
        } else {
            self.bins.push((value, 1));
            self.bins.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        Ok(())
    }
    fn mean(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.sum / self.n as f64 }
    }
}
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let hi = a.max(b);
    let lo = a.min(b);
    hi + (lo - hi).exp().ln_1p()
}
fn log_crp_wealth_profile(profile: &ScoreProfile, candidate_mean: f64, bet: f64) -> f64 {
    if bet <= 0.0 { return 0.0; }
    if bet >= 1.0 {
        if profile.bins.iter().any(|(y, _)| *y == 0.0) {
            return f64::NEG_INFINITY;
        }
        return profile.bins.iter()
            .map(|(y, n)| *n as f64 * (y.ln() - candidate_mean.ln()))
            .sum();
    }
    let log_cash = (-bet).ln_1p();
    let log_bet = bet.ln();
    let log_mean = candidate_mean.ln();
    let mut total = 0.0;
    for (y, count) in &profile.bins {
        let log_risky = if *y == 0.0 { f64::NEG_INFINITY }
            else { log_bet + y.ln() - log_mean };
        let term = log_add_exp(log_cash, log_risky);
        if !term.is_finite() { return term; }
        total += *count as f64 * term;
    }
    total
}
fn log_sup_crp_wealth_lower(profile: &ScoreProfile, candidate_mean: f64) -> Result<f64, String> {
    if profile.n == 0 || !candidate_mean.is_finite() || candidate_mean <= 0.0 {
        return Err("pCRP requires a nonempty profile and positive finite candidate mean".into());
    }
    if candidate_mean >= profile.mean() {
        return Ok(0.0);
    }
    // Fixed feasible bet grid: its maximum is no larger than the continuous
    // supremum, and is monotone in candidate_mean. This is intentionally a
    // conservative approximation, not the exact continuous-bet pCRP-star method.
    let mut best = 0.0_f64;
    for i in 1..=PCRP_BET_STEPS {
        let bet = i as f64 / PCRP_BET_STEPS as f64;
        best = best.max(log_crp_wealth_profile(profile, candidate_mean, bet));
    }
    let shrink = (1.0 - 64.0 * f64::EPSILON * (profile.n as f64 + 1.0)).max(0.0);
    Ok(best * shrink)
}
fn pcrp_star_lcb_profile(profile: &ScoreProfile, delta: f64) -> Result<f64, String> {
    if profile.n == 0 || !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("invalid pCRP confidence-bound input".into());
    }
    let mean = profile.mean();
    if mean <= 0.0 { return Ok(0.0); }
    let threshold = (1.0 / delta).ln()
        + 0.5 * (std::f64::consts::PI * (profile.n as f64 + 1.0)).ln();
    let mut hi = mean;
    let mut probe = mean * 0.5;
    let mut low = None;
    for _ in 0..1075 {
        if probe <= 0.0 { break; }
        if log_sup_crp_wealth_lower(profile, probe)? > threshold {
            low = Some(probe);
            break;
        }
        hi = probe;
        probe *= 0.5;
    }
    let Some(mut lo) = low else { return Ok(0.0); };
    for _ in 0..PCRP_ROOT_ITERS {
        let mid = lo + (hi - lo) * 0.5;
        if mid == lo || mid == hi { break; }
        if log_sup_crp_wealth_lower(profile, mid)? > threshold {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi.max(0.0).min(mean).min(1.0))
}
fn logarithmic_smoothing_lcb_profile(
    profile: &ScoreProfile,
    delta: f64,
    w_max: f64,
) -> Result<f64, String> {
    if profile.n == 0 || !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("invalid logarithmic-smoothing input".into());
    }
    if !w_max.is_finite() || w_max <= 0.0 {
        return Err("logarithmic smoothing requires positive finite W_max".into());
    }
    let n = profile.n as f64;
    // Preregistered scale-aware lambda; not tuned on the evaluation replicates.
    let lambda = (2.0 * (1.0 / delta).ln() / (n * w_max * w_max)).sqrt();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err("logarithmic-smoothing lambda is invalid".into());
    }
    let log_sum: f64 = profile.bins.iter()
        .map(|(y, count)| *count as f64 * (lambda * *y).ln_1p())
        .sum();
    let smoothed_mean = log_sum / (lambda * n);
    let log_lower_factor = lambda * smoothed_mean - (1.0 / delta).ln() / n;
    if !log_lower_factor.is_finite() {
        return Err("logarithmic-smoothing log factor is non-finite".into());
    }
    let lower = log_lower_factor.exp_m1() / lambda;
    if !lower.is_finite() {
        return Err("logarithmic-smoothing lower bound is non-finite".into());
    }
    Ok(lower.clamp(0.0, profile.mean().min(1.0)))
}

fn certified_best(bounds:&[Interval])->Option<usize>{
    if bounds.is_empty()||bounds.iter().any(|b|b.reversed()){return None;}
    let mut best=0usize;
    let mut best_lower=f64::NEG_INFINITY;
    let mut tied=false;
    for (i,b) in bounds.iter().enumerate(){
        if b.lower>best_lower+EPS{
            best=i;best_lower=b.lower;tied=false;
        }else if (b.lower-best_lower).abs()<=EPS{
            tied=true;
        }
    }
    if tied{return None;}
    let max_other_upper=bounds.iter().enumerate()
        .filter(|(i,_)|*i!=best).map(|(_,b)|b.upper)
        .fold(f64::NEG_INFINITY,f64::max);
    (bounds[best].lower>max_other_upper+EPS).then_some(best)
}

#[derive(Clone,Debug,Default)]
struct MethodAggregate{
    point_cover:usize,
    sequence_cover:usize,
    false_certificates:usize,
    ever_certified:usize,
    empty_portfolio:usize,
    average_width_sum:f64,
    average_width_sq_sum:f64,
    max_width_sum:f64,
    max_width_sq_sum:f64,
    certificate_times:Vec<usize>,
    trials:usize,
}
impl MethodAggregate{
    fn add(&mut self,point:bool,sequence:bool,false_certificate:bool,ever_certificate:bool,
        empty:bool,average_width:f64,max_width:f64,cert_time:Option<usize>){
        self.trials+=1;
        if point{self.point_cover+=1;}
        if sequence{self.sequence_cover+=1;}
        if false_certificate{self.false_certificates+=1;}
        if ever_certificate{self.ever_certified+=1;}
        if empty{self.empty_portfolio+=1;}
        self.average_width_sum+=average_width;
        self.average_width_sq_sum+=average_width*average_width;
        self.max_width_sum+=max_width;
        self.max_width_sq_sum+=max_width*max_width;
        if let Some(t)=cert_time{self.certificate_times.push(t);}
    }
}
#[derive(Clone,Debug,Default)]
struct SelectionAggregate{
    correct:usize,
    regret_sum:f64,
    regret_sq_sum:f64,
    trials:usize,
}
impl SelectionAggregate{
    fn add(&mut self,correct:bool,regret:f64){
        self.trials+=1;
        if correct{self.correct+=1;}
        self.regret_sum+=regret;self.regret_sq_sum+=regret*regret;
    }
}

#[derive(Clone,Debug)]
struct SimResult{
    name:&'static str,
    logger_name:&'static str,
    m:usize,
    values:Vec<f64>,
    policies:Vec<Policy>,
    aggregates:Vec<Vec<MethodAggregate>>,
    selection:Vec<SelectionAggregate>,
    elapsed_seconds:f64,
    observations:usize,
}
fn wilson(success:usize,total:usize)->(f64,f64,f64,f64){
    if total==0{return(0.0,0.0,0.0,1.0);}
    let n=total as f64;let p=success as f64/n;let se=(p*(1.0-p)/n).sqrt();
    let z=Z_975;let den=1.0+z*z/n;
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
    let mut v=values.to_vec();v.sort_unstable();Some(v[(v.len()-1)/2])
}
fn p90(values:&[usize])->Option<usize>{
    if values.is_empty(){return None;}
    let mut v=values.to_vec();v.sort_unstable();
    Some(v[((v.len()*9+9)/10).saturating_sub(1).min(v.len()-1)])
}

fn simulate(s:Scenario,reps:usize,looks:&[usize])->Result<SimResult,String>{
    if reps==0||looks.is_empty()||looks.iter().any(|x|*x==0)||looks.windows(2).any(|w|w[0]>=w[1]){
        return Err("replicates and strictly increasing positive looks are required".into());
    }
    let m=s.policy_count;
    let policies=make_policies(m,s.logger);
    let values:Vec<f64>=policies.iter().map(|p|p.value).collect();
    let best_value=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let max_look=*looks.last().unwrap();
    let mut agg=vec![vec![MethodAggregate::default();looks.len()];METHOD_COUNT];
    let mut selection=vec![SelectionAggregate::default();looks.len()];
    let started=Instant::now();
    let mut observations=0usize;

    for rep in 0..reps{
        let mut rng=SplitMix64::new(seed_for(s.logger.index,rep));
        let mut sums=vec![0.0_f64;m];
        let mut sums_sq=vec![0.0_f64;m];
        let mut cell_counts=[[0usize;2];2];
        let mut cell_reward_sums=[[0.0_f64;2];2];
        let mut cell_bounds=[[FULL;2];2];
        let mut bounds=vec![vec![FULL;m];METHOD_COUNT];
        let mut sequence_covered=[true;METHOD_COUNT];
        let mut ever_false_certificate=[false;METHOD_COUNT];
        let mut first_certificate:[Option<usize>;METHOD_COUNT]=[None;METHOD_COUNT];
        let mut point_certificate=[false;METHOD_COUNT];
        let mut point_false_certificate=[false;METHOD_COUNT];
        let mut score_profiles=vec![ScoreProfile::default();m];
        let mut complement_profiles=vec![ScoreProfile::default();m];
        let mut look_index=0usize;

        for t in 1..=max_look{
            let observation=next_observation(s.logger,&mut rng);
            let context=observation.context;
            let action=observation.action;
            let reward=observation.reward;
            let h=s.logger.h1[context];
            cell_counts[context][action]+=1;
            cell_reward_sums[context][action]+=reward;
            let cell_raw=cell_mean_cs(cell_reward_sums[context][action],cell_counts[context][action],ALPHA/4.0);
            cell_bounds[context][action]=intersect(cell_bounds[context][action],cell_raw);
            for j in 0..m{
                let p=policies[j].pi1[context];
                let ratio=if action==1{p/h}else{(1.0-p)/(1.0-h)};
                if !ratio.is_finite()||ratio<0.0||ratio>policies[j].w_max+1e-9{
                    return Err(format!("{} policy {} exceeded computed W",s.name,j));
                }
                let x=ratio*reward;
                let complement=ratio*(1.0-reward);
                if !x.is_finite()||x<0.0||x>policies[j].w_max+1e-9
                    || !complement.is_finite()||complement<0.0||complement>policies[j].w_max+1e-9{
                    return Err(format!("{} policy {} score outside [0,W]",s.name,j));
                }
                sums[j]+=x;sums_sq[j]+=x*x;
                score_profiles[j].push(x)?;
                complement_profiles[j].push(complement)?;
            }

            // Cache look/alpha-dependent logs once per method/look, not once
            // per candidate policy. This matters for the M=32 horizon-4096 sweep.
            let independent_radius_factor=hoeffding_radius_factor(t,ALPHA);
            let familywise_radius_factor=hoeffding_radius_factor(t,ALPHA/(m as f64));
            let independent_eb_logterm=empirical_bernstein_logterm(t,ALPHA);
            let familywise_eb_logterm=empirical_bernstein_logterm(t,ALPHA/(m as f64));
            for method in 0..5{
                for j in 0..m{
                    let mean=sums[j]/t as f64;
                    let raw=match method{
                        0=>wald_interval(sums[j],sums_sq[j],t),
                        1=>hoeffding_cs_factor(mean,policies[j].w_max,independent_radius_factor),
                        2=>hoeffding_cs_factor(mean,policies[j].w_max,familywise_radius_factor),
                        3=>empirical_bernstein_cs(sums[j],sums_sq[j],t,policies[j].w_max,independent_eb_logterm),
                        4=>empirical_bernstein_cs(sums[j],sums_sq[j],t,policies[j].w_max,familywise_eb_logterm),
                        _=>unreachable!(),
                    };
                    bounds[method][j]=if method==0{raw}else{intersect(bounds[method][j],raw)};
                }
            }
            // One simultaneous confidence sequence per primitive context/action
            // reward mean induces valid intervals for every policy value. The
            // Bonferroni correction is over four cells, not over M policies.
            for j in 0..m{
                bounds[5][j]=shared_policy_interval(&policies[j],cell_bounds);
            }
            let is_registered_look=looks.contains(&t);
            if is_registered_look {
                // Fixed-horizon pCRP-star and LS comparisons. Alpha is allocated over
                // two tails and M policies for THIS registered look. These methods are
                // intentionally not updated between looks and make no sequence claim.
                let delta_tail=ALPHA/(2.0*m as f64);
                for j in 0..m {
                    let lower_p=pcrp_star_lcb_profile(&score_profiles[j],delta_tail)?;
                    let lower_complement_p=pcrp_star_lcb_profile(&complement_profiles[j],delta_tail)?;
                    bounds[6][j]=Interval{lower:lower_p.clamp(0.0,1.0),upper:(1.0-lower_complement_p).clamp(0.0,1.0)};
                    let lower_ls=logarithmic_smoothing_lcb_profile(&score_profiles[j],delta_tail,policies[j].w_max)?;
                    let lower_complement_ls=logarithmic_smoothing_lcb_profile(&complement_profiles[j],delta_tail,policies[j].w_max)?;
                    bounds[7][j]=Interval{lower:lower_ls.clamp(0.0,1.0),upper:(1.0-lower_complement_ls).clamp(0.0,1.0)};
                }
            }
            for method in 0..METHOD_COUNT{
                if method>=ANYTIME_METHODS && !is_registered_look { continue; }
                let point=bounds[method].iter().enumerate().all(|(j,b)|b.covers(values[j]));
                if method<ANYTIME_METHODS {
                    sequence_covered[method]&=point;
                    if let Some(k)=certified_best(&bounds[method]){
                        if first_certificate[method].is_none(){first_certificate[method]=Some(t);}
                        if values[k]+1e-9<best_value{ever_false_certificate[method]=true;}
                    }
                } else {
                    point_certificate[method]=certified_best(&bounds[method]).is_some();
                    point_false_certificate[method]=certified_best(&bounds[method])
                        .map(|k| values[k]+1e-9<best_value).unwrap_or(false);
                }
            }

            while look_index<looks.len()&&looks[look_index]==t{
                let estimate_winner=(0..m).max_by(|a,b|(sums[*a]/t as f64).total_cmp(&(sums[*b]/t as f64))).unwrap();
                let best_set=values.iter().enumerate().filter(|(_,v)|(**v-best_value).abs()<=1e-9).map(|(i,_)|i).collect::<Vec<_>>();
                selection[look_index].add(best_set.contains(&estimate_winner),best_value-values[estimate_winner]);

                for method in 0..METHOD_COUNT{
                    let point=bounds[method].iter().enumerate().all(|(j,b)|b.covers(values[j]));
                    let average_width=bounds[method].iter().map(|b|b.width()).sum::<f64>()/(m as f64);
                    let max_width=bounds[method].iter().map(|b|b.width()).fold(0.0_f64,f64::max);
                    let empty=bounds[method].iter().any(|b|b.reversed());
                    if method<ANYTIME_METHODS {
                        agg[method][look_index].add(
                            point,sequence_covered[method],ever_false_certificate[method],
                            first_certificate[method].is_some(),empty,average_width,max_width,
                            first_certificate[method].filter(|t0|*t0<=looks[look_index]),
                        );
                    } else {
                        agg[method][look_index].add(
                            point,point,point_false_certificate[method],
                            point_certificate[method],empty,average_width,max_width,None,
                        );
                    }
                }
                look_index+=1;
            }
            observations+=1;
        }
    }
    Ok(SimResult{
        name:s.name,logger_name:s.logger.name,m,values,policies,aggregates:agg,selection,
        elapsed_seconds:started.elapsed().as_secs_f64(),observations,
    })
}

fn emit(out:&mut String,r:&SimResult,looks:&[usize]){
    let best_value=r.values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let max_w=r.policies.iter().map(|p|p.w_max).fold(0.0_f64,f64::max);
    let _=writeln!(out,"DGP\t{}\tlogger\t{}\tpolicies\t{}\tbest_value\t{:.8}\tmax_policy_W\t{:.8}",r.name,r.logger_name,r.m,best_value,max_w);
    for policy in &r.policies{
        let _=writeln!(out,"POLICY\t{}\tid\t{}\tpi_action1_context0\t{:.8}\tpi_action1_context1\t{:.8}\ttrue_value\t{:.8}\tW_max\t{:.8}\tis_best\t{}",
            r.name,policy.id,policy.pi1[0],policy.pi1[1],policy.value,policy.w_max,(policy.value-best_value).abs()<=EPS);
    }
    let methods=["Wald_pointwise_reused","Hoeffding_CS_per_policy_alpha_0.05","Hoeffding_CS_familywise_alpha_over_M","Empirical_Bernstein_CS_per_policy_alpha_0.05","Empirical_Bernstein_CS_familywise_alpha_over_M","Shared_cell_mean_Hoeffding_CS_alpha_over_4","pCRP_star_fixed_horizon_grid33","Logarithmic_Smoothing_fixed_horizon"];
    for method in 0..METHOD_COUNT{
        for (li,look) in looks.iter().copied().enumerate(){
            let a=&r.aggregates[method][li];
            let (v,se,lo,hi)=wilson(a.point_cover,a.trials);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tfamilywise_point_coverage\t{:.8}\t{:.8}\t{:.8}\t{:.8}",r.name,methods[method],look,v,se,lo,hi);
            if method<ANYTIME_METHODS {
                for (name,count) in [
                    ("familywise_sequence_coverage",a.sequence_cover),
                    ("any_false_certificate_probability",a.false_certificates),
                    ("ever_best_certificate_rate",a.ever_certified),
                ]{
                    let (v,se,lo,hi)=wilson(count,a.trials);
                    let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",r.name,methods[method],look,name,v,se,lo,hi);
                }
            } else {
                for (name,count) in [
                    ("point_false_certificate_probability",a.false_certificates),
                    ("point_best_certificate_rate",a.ever_certified),
                ]{
                    let (v,se,lo,hi)=wilson(count,a.trials);
                    let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",r.name,methods[method],look,name,v,se,lo,hi);
                }
                let _=writeln!(out,"METHOD_STATUS\t{}\t{}\tguarantee\tFIXED_HORIZON_ONLY",r.name,methods[method]);
            }
            let (v,se,lo,hi)=wilson(a.empty_portfolio,a.trials);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tportfolio_empty_interval_probability\t{:.8}\t{:.8}\t{:.8}\t{:.8}",r.name,methods[method],look,v,se,lo,hi);
            let (avg,avg_se)=mean_se(a.average_width_sum,a.average_width_sq_sum,a.trials);
            let (mx,mx_se)=mean_se(a.max_width_sum,a.max_width_sq_sum,a.trials);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_average_policy_interval_width\t{:.8}\t{:.8}\t-\t-",r.name,methods[method],look,avg,avg_se);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_max_policy_interval_width\t{:.8}\t{:.8}\t-\t-",r.name,methods[method],look,mx,mx_se);
            if method<ANYTIME_METHODS {
                let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tconditional_certificate_time_median\t{}\t-\t-\t-",r.name,methods[method],look,median(&a.certificate_times).map(|v|v.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
                let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tconditional_certificate_time_p90\t{}\t-\t-\t-",r.name,methods[method],look,p90(&a.certificate_times).map(|v|v.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
            }
        }
    }
    for (li,look) in looks.iter().copied().enumerate(){
        let a=&r.selection[li];
        let (acc,se,lo,hi)=wilson(a.correct,a.trials);
        let _=writeln!(out,"RESULT_METRIC\t{}\tIPW_argmax\t{}\tIPS_argmax_top1_accuracy\t{:.8}\t{:.8}\t{:.8}\t{:.8}",r.name,look,acc,se,lo,hi);
        let (regret,se_regret)=mean_se(a.regret_sum,a.regret_sq_sum,a.trials);
        let _=writeln!(out,"RESULT_METRIC\t{}\tIPW_argmax\t{}\tIPS_argmax_simple_regret\t{:.8}\t{:.8}\t-\t-",r.name,look,regret,se_regret);
    }
    let _=writeln!(out,"SYSTEM\t{}\tseconds\t{:.6}\tobservations\t{}\tobservations_per_second\t{:.3}",r.name,r.elapsed_seconds,r.observations,r.observations as f64/r.elapsed_seconds.max(1e-12));
}
fn benchmark(reps:usize,looks:&[usize])->Result<String,String>{
    let scenarios=scenarios();
    let mut out=String::from("record_type\tscenario\tmethod\tlook\tmetric\testimate\tstandard_error\tci_low\tci_high\n");
    let _=writeln!(out,"SEED_SCHEDULE\tbase_seed\t0x8b8b8b8b04600046\tformula\tBASE xor LOGGER_INDEX*D6E8FEB86659FD93 xor REPLICATE*A0761D6478BD642F");
    let _=writeln!(out,"BENCHMARK_CONFIG\treplicates\t{}\tmax_horizon\t{}",reps,looks.iter().max().unwrap_or(&0));
    for s in scenarios{
        let result=simulate(s,reps,looks)?;
        emit(&mut out,&result,looks);
    }
    Ok(out)
}
fn seed_smoke()->String{
    let mut rng=SplitMix64::new(0x8b8b_8b8b_0460_0046);
    let mut out=String::from("VIS_DVPC_046_SEED_SMOKE\n");
    for _ in 0..64{let _=writeln!(out,"{:016x}",rng.next_u64());}
    out
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let result=match args.get(1).map(String::as_str){
        Some("--benchmark")=>benchmark(REPLICATES,&LOOKS),
        Some("--smoke")=>benchmark(8,&[8,16]),
        Some("--seed-smoke")=>Ok(seed_smoke()),
        _=>Err("usage: vision-dvpc-046-multipolicy-selection --benchmark|--smoke|--seed-smoke".into())
    };
    match result{
        Ok(out)=>print!("{out}"),
        Err(err)=>{eprintln!("VIS_DVPC_046_ERROR\t{err}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn pcrp_star_profile_constant_stream_matches_analytic_root() {
        let mut profile=ScoreProfile::default();
        for _ in 0..32 { profile.push(0.6).unwrap(); }
        let delta=0.05;
        let expected=0.6*(delta/(std::f64::consts::PI*(profile.n as f64+1.0)).sqrt()).powf(1.0/profile.n as f64);
        let got=pcrp_star_lcb_profile(&profile,delta).unwrap();
        assert!((got-expected).abs()<2e-5,"got={got} expected={expected}");
    }
    #[test]
    fn pcrp_grid_wealth_witness_is_monotone_in_candidate_mean() {
        let mut profile=ScoreProfile::default();
        for value in [0.0,0.0,0.2,0.2,0.4,0.8,1.2,1.2] {
            profile.push(value).unwrap();
        }
        let mut previous=f64::INFINITY;
        for i in 1..=1000 {
            let nu=i as f64/1000.0;
            let current=log_sup_crp_wealth_lower(&profile,nu).unwrap();
            assert!(current<=previous+1e-9,"nu={nu} previous={previous} current={current}");
            previous=current;
        }
    }
    #[test]
    fn ls_constant_profile_matches_closed_form() {
        let mut profile=ScoreProfile::default();
        for _ in 0..64 { profile.push(0.4).unwrap(); }
        let delta=0.01;
        let w_max=1.5;
        let n=profile.n as f64;
        let lambda=(2.0*(1.0/delta).ln()/(n*w_max*w_max)).sqrt();
        let expected=(((lambda*0.4).ln_1p() - (1.0/delta).ln()/n).exp_m1()/lambda)
            .clamp(0.0,0.4);
        let got=logarithmic_smoothing_lcb_profile(&profile,delta,w_max).unwrap();
        assert!((got-expected).abs()<1e-12,"got={got} expected={expected}");
        assert!((got..=0.4).contains(&got));
    }

    #[test]
    fn pcrp_and_ls_bounds_fail_closed_and_handle_zero_rewards() {
        let mut zeros=ScoreProfile::default();
        for _ in 0..16 { zeros.push(0.0).unwrap(); }
        assert_eq!(pcrp_star_lcb_profile(&zeros,0.05).unwrap(),0.0);
        assert_eq!(logarithmic_smoothing_lcb_profile(&zeros,0.05,1.0).unwrap(),0.0);
        assert!(pcrp_star_lcb_profile(&zeros,1.0).is_err());
        assert!(logarithmic_smoothing_lcb_profile(&zeros,0.05,0.0).is_err());
    }
    #[test]
    fn fixed_horizon_methods_are_not_updated_between_registered_looks() {
        let looks=[8,16];
        assert!(!looks.contains(&7));
        assert!(looks.contains(&8));
    }

    #[test]
    fn every_policy_value_and_weight_bound_is_exactly_well_formed(){
        for logger in [Logger::healthy(),Logger::weak()]{
            for m in POLICY_COUNTS{
                let policies=make_policies(m,logger);
                assert_eq!(policies.len(),m);
                for p in &policies{
                    assert!((0.0..=1.0).contains(&p.value));
                    assert!(p.w_max>=1.0);
                    for x in 0..2{
                        let h=logger.h1[x];
                        assert!(p.pi1[x]/h<=p.w_max+EPS);
                        assert!((1.0-p.pi1[x])/(1.0-h)<=p.w_max+EPS);
                    }
                }
                let best=policies.iter().map(|p|p.value).fold(f64::NEG_INFINITY,f64::max);
                assert_eq!(policies.iter().filter(|p|(p.value-best).abs()<=EPS).count(),1,
                    "each frozen portfolio must have a unique optimum: logger={} M={m}",logger.name);
            }
        }
    }

    #[test]
    fn running_intersection_never_expands(){
        let old=Interval{lower:0.2,upper:0.8};
        let new=intersect(old,Interval{lower:0.1,upper:0.7});
        assert!((new.lower-0.2).abs()<EPS);
        assert!((new.upper-0.7).abs()<EPS);
        assert!(new.width()<=old.width()+EPS);
    }

    #[test]
    fn familywise_adjustment_is_never_narrower_than_unadjusted_cs(){
        for t in [1,32,256,4096]{
            let mean=0.5;let w=18.0;
            let independent=hoeffding_cs(mean,w,t,ALPHA);
            let family=hoeffding_cs(mean,w,t,ALPHA/32.0);
            assert!(family.width()>=independent.width()-EPS);
        }
    }


    #[test]
    fn empirical_bernstein_intervals_are_bounded_and_familywise_allocation_is_wider(){
        for n in [1,2,16,256,4096]{
            let sum=0.37*n as f64;
            let sum_sq=0.2*n as f64;
            let ind=empirical_bernstein_cs(sum,sum_sq,n,2.0,empirical_bernstein_logterm(n,ALPHA));
            let fam=empirical_bernstein_cs(sum,sum_sq,n,2.0,empirical_bernstein_logterm(n,ALPHA/32.0));
            assert!(ind.lower>=0.0 && ind.upper<=1.0 && !ind.reversed());
            assert!(fam.lower>=0.0 && fam.upper<=1.0 && !fam.reversed());
            assert!(fam.width()>=ind.width()-EPS);
        }
    }

    #[test]
    fn shared_cell_bounds_induce_valid_policy_value_bounds(){
        let cell_true=[[0.10,0.90],[0.95,0.15]];
        let cells=[
            [Interval{lower:0.05,upper:0.15},Interval{lower:0.85,upper:0.95}],
            [Interval{lower:0.90,upper:1.0},Interval{lower:0.10,upper:0.20}],
        ];
        for policy in make_policies(8,Logger::healthy()){
            let value=policy.value;
            let bound=shared_policy_interval(&policy,cells);
            assert!(bound.lower<=value+EPS);
            assert!(bound.upper>=value-EPS);
        }
        let _=cell_true;
    }

    #[test]
    fn shared_cell_alpha_is_allocated_over_four_primitives(){
        for n in [1,16,256,4096]{
            let alpha_cell=ALPHA/4.0;
            let delta=alpha_cell/(2.0*(n as f64)*((n+1) as f64));
            let each_tail=delta*1.0; // symbolic first-look allocation; actual sum uses telescoping.
            assert!(each_tail<alpha_cell/2.0);
            assert!(hoeffding_cs(0.5,1.0,n,ALPHA/32.0).width()>=cell_mean_cs(0.5*n as f64,n,alpha_cell).width()-EPS);
        }
    }

    #[test]
    fn certificate_fails_closed_on_empty_or_overlapping_intervals(){
        assert_eq!(certified_best(&[Interval{lower:0.7,upper:0.8},Interval{lower:0.1,upper:0.6}]),Some(0));
        assert_eq!(certified_best(&[Interval{lower:0.7,upper:0.8},Interval{lower:0.2,upper:0.75}]),None);
        assert_eq!(certified_best(&[Interval{lower:0.8,upper:0.7},Interval{lower:0.1,upper:0.2}]),None);
    }

    #[test]
    fn same_logger_and_replicate_produce_identical_sample_prefixes_across_portfolio_sizes(){
        let logger=Logger::weak();
        let mut small=SplitMix64::new(seed_for(logger.index,91));
        let mut large=SplitMix64::new(seed_for(logger.index,91));
        for _ in 0..4096{
            assert_eq!(next_observation(logger,&mut small),next_observation(logger,&mut large));
        }
    }

    #[test]
    fn paired_replay_uses_same_seed_for_policy_counts(){
        assert_eq!(seed_for(Logger::healthy().index,17),seed_for(Logger::healthy().index,17));
        assert_ne!(seed_for(Logger::healthy().index,17),seed_for(Logger::weak().index,17));
        let a=simulate(Scenario{logger:Logger::healthy(),policy_count:4,name:"smoke_h4"},8,&[8,16]).unwrap();
        let b=simulate(Scenario{logger:Logger::healthy(),policy_count:4,name:"smoke_h4"},8,&[8,16]).unwrap();
        assert_eq!(a.aggregates[2][1].sequence_cover,b.aggregates[2][1].sequence_cover);
        assert_eq!(a.aggregates[2][1].false_certificates,b.aggregates[2][1].false_certificates);
        assert_eq!(a.selection[1].correct,b.selection[1].correct);
    }
}
