//! VIS-DVPC-042: reproducible long-horizon OPE coverage/power/width frontier.
//!
//! Methods compared on identical seeded streams:
//! - MOPE two-tail confidence sequence,
//! - reused pointwise Wald interval (non-anytime-valid comparator),
//! - two-sided Hoeffding alpha-spending sequence,
//! - two-sided empirical-Bernstein alpha-spending interval.
//!
//! Results are Monte Carlo diagnostics with binomial uncertainty, not a finite-sample proof.
#![deny(unsafe_code)]

#[path = "vision-dvpc-040-mope-sufficient-stats.rs"]
mod stream;

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const ALPHA: f64 = 0.05;
const ALPHA_SIDE: f64 = 0.025;
const MAX_HORIZON: usize = 512;
const REPLICATES: usize = 2_000;
const HORIZONS: [usize; 3] = [32, 128, 512];
const PROMOTION_THRESHOLD: f64 = 0.55;
const TRUE_NULL_VALUE: f64 = 0.50;
const BASE_SEED: u64 = 0x8b8b_8b8b_0410_0042;
const PSI: f64 = 2.0 - 4.0 * std::f64::consts::LN_2;
const ROOT_TOL: f64 = 2e-7;
const WEALTH_TOL: f64 = 2e-8;
const ENDPOINT_TOL: f64 = 1e-10;
const Z_975: f64 = 1.959_963_984_540_054;

const GRID_SIZE: usize = 1000;
const METHOD_NAMES: [&str; 6] = [
    "MOPE_analytic_root_audited",
    "MOPE_quadratic_bound_grid_1000",
    "MOPE_upstream_style_actual_wealth_grid_1000",
    "Wald_pointwise_reused",
    "Hoeffding_alpha_spending_intersection",
    "Empirical_Bernstein_alpha_spending_intersection",
];

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    behavior_action_one: f64,
    reward_mean_action_zero: f64,
    reward_mean_action_one: f64,
    true_value: f64,
    max_weight: f64,
}

impl Scenario {
    fn healthy_null() -> Self {
        Self { name: "healthy_null", behavior_action_one: 0.50, reward_mean_action_zero: 0.20,
            reward_mean_action_one: 0.80, true_value: 0.50, max_weight: 1.0 }
    }
    fn healthy_alternative() -> Self {
        Self { name: "healthy_alt", behavior_action_one: 0.50, reward_mean_action_zero: 0.50,
            reward_mean_action_one: 0.80, true_value: 0.65, max_weight: 1.0 }
    }
    fn weak_overlap_null() -> Self {
        Self { name: "weak_overlap_null", behavior_action_one: 0.99, reward_mean_action_zero: 0.20,
            reward_mean_action_one: 0.80, true_value: 0.50, max_weight: 50.0 }
    }
    fn weak_overlap_alternative() -> Self {
        Self { name: "weak_overlap_alt", behavior_action_one: 0.99, reward_mean_action_zero: 0.50,
            reward_mean_action_one: 0.80, true_value: 0.65, max_weight: 50.0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[derive(Clone, Copy, Debug)]
struct BetRecord { w: f64, r: f64, lambda1: f64, lambda2: f64 }

#[derive(Clone, Copy, Debug)]
struct Interval { lower: f64, upper: f64 }
impl Interval {
    fn covers(self, value: f64) -> bool {
        self.lower <= self.upper + ENDPOINT_TOL
            && self.lower <= value + ENDPOINT_TOL
            && self.upper >= value - ENDPOINT_TOL
    }
    fn width(self) -> f64 { (self.upper - self.lower).max(0.0) }
    fn reversed(self) -> bool { self.lower > self.upper + ENDPOINT_TOL }
}

#[derive(Clone, Copy, Debug, Default)]
struct MopeDiagnostics {
    root_checks: u64,
    root_failures: u64,
    endpoint_collisions: u64,
}
#[derive(Clone, Copy, Debug, Default)]
struct GridDiagnostics { endpoint_advances: u64 }

struct MopePair {
    lower_stream: stream::StreamState,
    complement_stream: stream::StreamState,
    lower_endpoint: f64,
    complement_endpoint: f64,
    lower_history: Vec<BetRecord>,
    complement_history: Vec<BetRecord>,
    diagnostics: MopeDiagnostics,
}

impl MopePair {
    fn new(max_weight: f64, capacity: usize) -> Result<Self, String> {
        Ok(Self {
            lower_stream: stream::StreamState::new(max_weight, ALPHA_SIDE)?,
            complement_stream: stream::StreamState::new(max_weight, ALPHA_SIDE)?,
            lower_endpoint: 0.0, complement_endpoint: 0.0,
            lower_history: Vec::with_capacity(capacity),
            complement_history: Vec::with_capacity(capacity),
            diagnostics: MopeDiagnostics::default(),
        })
    }
    fn update(&mut self, w: f64, r: f64) -> Result<Interval, String> {
        let lo_lambda = self.lower_stream.current_lambda();
        let hi_lambda = self.complement_stream.current_lambda();
        let lo_report = self.lower_stream.observe(w, r)?;
        let hi_report = self.complement_stream.observe(w, 1.0 - r)?;
        self.lower_history.push(BetRecord { w, r, lambda1: lo_lambda.x, lambda2: lo_lambda.y });
        self.complement_history.push(BetRecord { w, r: 1.0-r, lambda1: hi_lambda.x, lambda2: hi_lambda.y });
        let level = (2.0 / ALPHA_SIDE).ln();

        if let Some(root) = lo_report.selected_safe_root {
            if root > self.lower_endpoint + ENDPOINT_TOL {
                self.diagnostics.root_checks += 1;
                if certify_root(&self.lower_history, root, level) {
                    let endpoint_cap = 1.0 - self.complement_endpoint;
                    if root > endpoint_cap + ENDPOINT_TOL {
                        self.diagnostics.endpoint_collisions += 1;
                    }
                    self.lower_endpoint = self.lower_endpoint.max(root).min(endpoint_cap);
                } else {
                    self.diagnostics.root_failures += 1;
                }
            }
        }
        if let Some(root) = hi_report.selected_safe_root {
            if root > self.complement_endpoint + ENDPOINT_TOL {
                self.diagnostics.root_checks += 1;
                if certify_root(&self.complement_history, root, level) {
                    let endpoint_cap = 1.0 - self.lower_endpoint;
                    if root > endpoint_cap + ENDPOINT_TOL {
                        self.diagnostics.endpoint_collisions += 1;
                    }
                    self.complement_endpoint = self.complement_endpoint.max(root).min(endpoint_cap);
                } else {
                    self.diagnostics.root_failures += 1;
                }
            }
        }
        self.lower_stream.choose_next_bet(self.lower_endpoint)?;
        self.complement_stream.choose_next_bet(self.complement_endpoint)?;
        Ok(Interval { lower: self.lower_endpoint, upper: 1.0 - self.complement_endpoint })
    }
}

/// Upstream-style MOPE root lookup: maintain log wealth on 1,000 values in [0,1]
/// and select the first grid value whose half-initialized log wealth is below
/// log(1/alpha). Uses the same streaming bet optimizer as the analytic-root method.
struct MopeGridPair {
    lower_stream: stream::StreamState,
    complement_stream: stream::StreamState,
    values: Vec<f64>,
    lower_log_wealth: Vec<f64>,
    complement_log_wealth: Vec<f64>,
    lower_endpoint: f64,
    complement_endpoint: f64,
    diagnostics: GridDiagnostics,
}

impl MopeGridPair {
    fn new(max_weight: f64) -> Result<Self, String> {
        let values: Vec<f64> = (0..GRID_SIZE)
            .map(|i| i as f64 / (GRID_SIZE - 1) as f64)
            .collect();
        Ok(Self {
            lower_stream: stream::StreamState::new(max_weight, ALPHA_SIDE)?,
            complement_stream: stream::StreamState::new(max_weight, ALPHA_SIDE)?,
            lower_log_wealth: vec![0.5_f64.ln(); GRID_SIZE],
            complement_log_wealth: vec![0.5_f64.ln(); GRID_SIZE],
            values,
            lower_endpoint: 0.0,
            complement_endpoint: 0.0,
            diagnostics: GridDiagnostics::default(),
        })
    }

    fn update(&mut self, w: f64, r: f64) -> Result<Interval, String> {
        let lo_lambda = self.lower_stream.current_lambda();
        let hi_lambda = self.complement_stream.current_lambda();
        let _ = self.lower_stream.observe(w, r)?;
        let _ = self.complement_stream.observe(w, 1.0-r)?;

        for i in 0..GRID_SIZE {
            let value = self.values[i];
            let factor_lo = 1.0 + lo_lambda.x * (w-1.0) + lo_lambda.y * (w*r-value);
            let factor_hi = 1.0 + hi_lambda.x * (w-1.0) + hi_lambda.y * (w*(1.0-r)-value);
            if !factor_lo.is_finite() || factor_lo < 0.5-WEALTH_TOL
                || !factor_hi.is_finite() || factor_hi < 0.5-WEALTH_TOL {
                return Err("grid MOPE bet violated the half-wealth constraint".to_owned());
            }
            self.lower_log_wealth[i] += factor_lo.ln();
            self.complement_log_wealth[i] += factor_hi.ln();
        }

        let log_threshold = (1.0/ALPHA_SIDE).ln();
        let grid_lower = self.lower_log_wealth.iter()
            .position(|wealth| *wealth < log_threshold)
            .unwrap_or(0);
        let grid_complement = self.complement_log_wealth.iter()
            .position(|wealth| *wealth < log_threshold)
            .unwrap_or(0);
        let proposed_lower = self.lower_endpoint.max(self.values[grid_lower]);
        let endpoint_cap = 1.0-self.complement_endpoint;
        let updated_lower = proposed_lower.min(endpoint_cap);
        if updated_lower > self.lower_endpoint + ENDPOINT_TOL {
            self.diagnostics.endpoint_advances += 1;
        }
        self.lower_endpoint = updated_lower;

        let proposed_complement = self.complement_endpoint.max(self.values[grid_complement]);
        let complement_cap = 1.0-self.lower_endpoint;
        let updated_complement = proposed_complement.min(complement_cap);
        if updated_complement > self.complement_endpoint + ENDPOINT_TOL {
            self.diagnostics.endpoint_advances += 1;
        }
        self.complement_endpoint = updated_complement;

        self.lower_stream.choose_next_bet(self.lower_endpoint)?;
        self.complement_stream.choose_next_bet(self.complement_endpoint)?;
        Ok(Interval { lower: self.lower_endpoint, upper: 1.0-self.complement_endpoint })
    }
}

/// Fair precision comparator: evaluates the same quadratic lower-bound
/// polynomial used by the analytic-root method on GRID_SIZE candidates.
/// This isolates root finding versus grid discretization from actual wealth.
struct MopeBoundGridPair {
    lower_stream: stream::StreamState,
    complement_stream: stream::StreamState,
    values: Vec<f64>,
    lower_bound: Vec<f64>,
    complement_bound: Vec<f64>,
    lower_endpoint: f64,
    complement_endpoint: f64,
    diagnostics: GridDiagnostics,
}

impl MopeBoundGridPair {
    fn new(max_weight: f64) -> Result<Self, String> {
        let values: Vec<f64>=(0..GRID_SIZE).map(|i|i as f64/(GRID_SIZE-1) as f64).collect();
        Ok(Self {
            lower_stream:stream::StreamState::new(max_weight,ALPHA_SIDE)?,
            complement_stream:stream::StreamState::new(max_weight,ALPHA_SIDE)?,
            lower_bound:vec![0.0;GRID_SIZE],
            complement_bound:vec![0.0;GRID_SIZE],
            values,
            lower_endpoint:0.0,
            complement_endpoint:0.0,
            diagnostics:GridDiagnostics::default(),
        })
    }

    fn update(&mut self,w:f64,r:f64)->Result<Interval,String>{
        let lo=self.lower_stream.current_lambda();
        let hi=self.complement_stream.current_lambda();
        let _=self.lower_stream.observe(w,r)?;
        let _=self.complement_stream.observe(w,1.0-r)?;
        let z=w-1.0;
        for i in 0..GRID_SIZE{
            let v=self.values[i];
            let d=w*r-v;
            let contribution=lo.x*z+lo.y*d+PSI*(lo.x*lo.x*z*z+2.0*lo.x*lo.y*z*d+lo.y*lo.y*d*d);
            self.lower_bound[i]+=contribution;
            let dc=w*(1.0-r)-v;
            let contribution_c=hi.x*z+hi.y*dc+PSI*(hi.x*hi.x*z*z+2.0*hi.x*hi.y*z*dc+hi.y*hi.y*dc*dc);
            self.complement_bound[i]+=contribution_c;
        }
        let level=(2.0/ALPHA_SIDE).ln();
        let lower_index=self.lower_bound.iter().position(|p|*p<level).unwrap_or(0);
        let complement_index=self.complement_bound.iter().position(|p|*p<level).unwrap_or(0);
        let proposed_lower=self.lower_endpoint.max(self.values[lower_index]);
        let lower_cap=1.0-self.complement_endpoint;
        let updated_lower=proposed_lower.min(lower_cap);
        if updated_lower>self.lower_endpoint+ENDPOINT_TOL{self.diagnostics.endpoint_advances+=1;}
        self.lower_endpoint=updated_lower;
        let proposed_complement=self.complement_endpoint.max(self.values[complement_index]);
        let complement_cap=1.0-self.lower_endpoint;
        let updated_complement=proposed_complement.min(complement_cap);
        if updated_complement>self.complement_endpoint+ENDPOINT_TOL{self.diagnostics.endpoint_advances+=1;}
        self.complement_endpoint=updated_complement;
        self.lower_stream.choose_next_bet(self.lower_endpoint)?;
        self.complement_stream.choose_next_bet(self.complement_endpoint)?;
        Ok(Interval{lower:self.lower_endpoint,upper:1.0-self.complement_endpoint})
    }
}

fn direct_polynomial(history: &[BetRecord], value: f64) -> f64 {
    history.iter().map(|x| {
        let z = x.w - 1.0;
        let d = x.w * x.r - value;
        x.lambda1 * z + x.lambda2 * d
            + PSI * (x.lambda1*x.lambda1*z*z + 2.0*x.lambda1*x.lambda2*z*d + x.lambda2*x.lambda2*d*d)
    }).sum()
}
fn direct_log_wealth(history: &[BetRecord], value: f64) -> Option<f64> {
    let mut out = 0.0;
    for x in history {
        let factor = 1.0 + x.lambda1*(x.w-1.0) + x.lambda2*(x.w*x.r-value);
        if !factor.is_finite() || factor < 0.5 - WEALTH_TOL { return None; }
        out += factor.ln();
    }
    Some(out)
}
fn certify_root(history: &[BetRecord], root: f64, level: f64) -> bool {
    if !root.is_finite() || !(0.0..=1.0).contains(&root) { return false; }
    if history.iter().any(|x| x.lambda2 < -WEALTH_TOL) { return false; }
    if (direct_polynomial(history, root) - level).abs() > ROOT_TOL * (1.0 + level.abs()) { return false; }
    matches!(direct_log_wealth(history, root), Some(value) if value + WEALTH_TOL >= level)
}

#[derive(Clone, Copy, Debug, Default)]
struct OpeMoments { n: usize, sum: f64, sum_sq: f64 }
impl OpeMoments {
    fn update(&mut self, value: f64) { self.n += 1; self.sum += value; self.sum_sq += value*value; }
    fn mean(self) -> f64 { if self.n == 0 { 0.0 } else { self.sum / self.n as f64 } }
    fn unbiased_variance(self) -> f64 {
        if self.n < 2 { return 0.0; }
        ((self.sum_sq - self.sum*self.sum/self.n as f64).max(0.0))/(self.n-1) as f64
    }
}

fn clip_interval(lower: f64, upper: f64) -> Interval {
    Interval { lower: lower.clamp(0.0,1.0), upper: upper.clamp(0.0,1.0) }
}
fn running_intersection(previous: Interval, raw: Interval) -> Interval {
    Interval { lower: previous.lower.max(raw.lower), upper: previous.upper.min(raw.upper) }
}
fn baseline_interval(method: usize, moments: OpeMoments, w_max: f64) -> Interval {
    let n = moments.n as f64;
    let mean = moments.mean();
    if moments.n == 0 { return Interval { lower: 0.0, upper: 1.0 }; }
    match method {
        1 => {
            if moments.n < 2 { return Interval { lower: 0.0, upper: 1.0 }; }
            let radius = Z_975 * (moments.unbiased_variance()/n).sqrt();
            clip_interval(mean-radius, mean+radius)
        }
        2 => {
            let t = n;
            let delta_side = ALPHA / (2.0*t*(t+1.0));
            let radius = w_max * ((1.0/delta_side).ln()/(2.0*t)).sqrt();
            clip_interval(mean-radius, mean+radius)
        }
        3 => {
            if moments.n < 2 { return Interval { lower: 0.0, upper: 1.0 }; }
            let t = n;
            // The empirical-Bernstein formula is a two-sided interval. Spend
            // alpha_t = alpha/[t(t+1)] per look (sum_t alpha_t = alpha);
            // unlike Hoeffding above, do not split this allocation between tails.
            let delta_interval = ALPHA / (t*(t+1.0));
            let logterm = (2.0/delta_interval).ln();
            let variance = moments.unbiased_variance();
            let radius = (2.0*variance*logterm/(t-1.0)).sqrt()
                + 7.0*w_max*logterm/(3.0*(t-1.0));
            clip_interval(mean-radius, mean+radius)
        }
        _ => unreachable!("MOPE is updated separately"),
    }
}

fn wilson(successes: usize, trials: usize) -> (f64, f64, f64, f64) {
    if trials == 0 { return (0.0, 0.0, 0.0, 1.0); }
    let n = trials as f64;
    let p = successes as f64/n;
    let se = (p*(1.0-p)/n).sqrt();
    let z = Z_975;
    let den = 1.0+z*z/n;
    let center = (p+z*z/(2.0*n))/den;
    let half = z*((p*(1.0-p)/n+z*z/(4.0*n*n)).sqrt())/den;
    (p,se,(center-half).max(0.0),(center+half).min(1.0))
}
fn mean_se(sum: f64, sum_sq: f64, n: usize) -> (f64, f64) {
    if n == 0 { return (0.0,0.0); }
    let m=sum/n as f64;
    if n<2 { return (m,0.0); }
    let var=((sum_sq-sum*sum/n as f64).max(0.0))/(n-1) as f64;
    (m,(var/n as f64).sqrt())
}

#[derive(Clone, Debug, Default)]
struct Aggregate {
    point_hits: usize,
    sequence_hits: usize,
    promotion_hits: usize,
    reversed_hits: usize,
    width_sum: f64,
    width_sq_sum: f64,
    lower_sum: f64,
    lower_sq_sum: f64,
    upper_sum: f64,
    upper_sq_sum: f64,
    promotion_times: Vec<usize>,
    root_checks: u64,
    root_failures: u64,
    endpoint_collisions: u64,
    grid_endpoint_advances: u64,
    elapsed_seconds_sum: f64,
    elapsed_seconds_sq_sum: f64,
    trials: usize,
}
impl Aggregate {
    fn add(&mut self, interval: Interval, point_cover: bool, sequence_cover: bool,
           promotion_time: Option<usize>, root_checks: u64, root_failures: u64,
           endpoint_collisions: u64, grid_endpoint_advances: u64, elapsed_seconds: f64) {
        self.trials += 1;
        self.endpoint_collisions += endpoint_collisions;
        self.grid_endpoint_advances += grid_endpoint_advances;
        self.elapsed_seconds_sum += elapsed_seconds;
        self.elapsed_seconds_sq_sum += elapsed_seconds * elapsed_seconds;
        if point_cover { self.point_hits += 1; }
        if sequence_cover { self.sequence_hits += 1; }
        if interval.reversed() { self.reversed_hits += 1; }
        let w=interval.width();
        self.width_sum+=w; self.width_sq_sum+=w*w;
        self.lower_sum+=interval.lower; self.lower_sq_sum+=interval.lower*interval.lower;
        self.upper_sum+=interval.upper; self.upper_sq_sum+=interval.upper*interval.upper;
        if let Some(t)=promotion_time { self.promotion_hits += 1; self.promotion_times.push(t); }
        self.root_checks += root_checks;
        self.root_failures += root_failures;
    }
}

#[derive(Clone, Debug)]
struct ScenarioOutput {
    scenario: &'static str,
    true_value: f64,
    w_max: f64,
    aggregates: Vec<Vec<Aggregate>>, // method x look
    elapsed_seconds: f64,
    sample_count: usize,
    path_count: usize,
}

fn seed_for(scenario_index: usize, replicate: usize) -> u64 {
    BASE_SEED ^ ((scenario_index as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93))
        ^ (replicate as u64).wrapping_mul(0xA076_1D64_78BD_642F)
}
fn simulate_scenario(scenario: Scenario, scenario_index: usize, replicates: usize, horizons: &[usize]) -> Result<ScenarioOutput, String> {
    if replicates == 0 || horizons.is_empty() { return Err("replicates and horizons must be nonzero".to_owned()); }
    let max_horizon=*horizons.iter().max().unwrap();
    if max_horizon > MAX_HORIZON || horizons.iter().any(|h| *h==0) { return Err("horizon outside [1,512]".to_owned()); }
    let start=Instant::now();
    let mut aggregates=vec![vec![Aggregate::default();horizons.len()];5];
    let mut samples_processed=0usize;

    for replicate in 0..replicates {
        let mut rng=SplitMix64::new(seed_for(scenario_index,replicate));
        let mut mope=MopePair::new(scenario.max_weight,max_horizon)?;
        let mut mope_grid=MopeGridPair::new(scenario.max_weight)?;
        let mut moments=OpeMoments::default();
        let mut seq_covered=[true;5];
        let mut promotion_time:[Option<usize>;5]=[None;5];
        let mut previous_intervals=[Interval{lower:0.0,upper:1.0};5];
        let mut method_elapsed_seconds=[0.0_f64;5];
        let mut look_index=0usize;

        for t in 1..=max_horizon {
            let action_one=rng.unit()<scenario.behavior_action_one;
            let reward_mean=if action_one {scenario.reward_mean_action_one} else {scenario.reward_mean_action_zero};
            let reward=if rng.unit()<reward_mean {1.0} else {0.0};
            let behavior_probability=if action_one {scenario.behavior_action_one} else {1.0-scenario.behavior_action_one};
            let w=POLICY_ACTION_ONE / behavior_probability;
            if !w.is_finite() || w<0.0 || w>scenario.max_weight+1e-10 { return Err("generated importance ratio outside frozen support".to_owned()); }

            let mope_start=Instant::now();
            let mope_interval=mope.update(w,reward)?;
            method_elapsed_seconds[0]+=mope_start.elapsed().as_secs_f64();
            let grid_start=Instant::now();
            let grid_interval=mope_grid.update(w,reward)?;
            method_elapsed_seconds[1]+=grid_start.elapsed().as_secs_f64();
            moments.update(w*reward);
            let mut intervals=[Interval{lower:0.0,upper:1.0};5];
            intervals[0]=mope_interval;
            intervals[1]=grid_interval;
            // Reused Wald remains pointwise by design. Time-uniform methods use
            // the running intersection, matching the endpoint semantics of MOPE.
            let wald_start=Instant::now();
            intervals[2]=baseline_interval(1,moments,scenario.max_weight);
            method_elapsed_seconds[2]+=wald_start.elapsed().as_secs_f64();
            for method in 3..5 {
                let method_start=Instant::now();
                let raw=baseline_interval(method-1,moments,scenario.max_weight);
                intervals[method]=running_intersection(previous_intervals[method],raw);
                method_elapsed_seconds[method]+=method_start.elapsed().as_secs_f64();
            }

            for method in 0..5 {
                let interval=intervals[method];
                let covers=interval.covers(scenario.true_value);
                seq_covered[method] &= covers;
                if promotion_time[method].is_none() && !interval.reversed()
                    && interval.lower > PROMOTION_THRESHOLD {
                    promotion_time[method]=Some(t);
                }
            }
            while look_index<horizons.len() && horizons[look_index]==t {
                for method in 0..5 {
                    let interval=intervals[method];
                    let diag=if method==0 {mope.diagnostics} else {MopeDiagnostics::default()};
                    let grid_advances=if method==1 {mope_grid.diagnostics.endpoint_advances} else {0};
                    aggregates[method][look_index].add(interval,interval.covers(scenario.true_value),
                        seq_covered[method],promotion_time[method],diag.root_checks,diag.root_failures,
                        diag.endpoint_collisions,grid_advances,method_elapsed_seconds[method]);
                }
                look_index+=1;
            }
            previous_intervals=intervals;
            samples_processed+=1;
        }
    }
    Ok(ScenarioOutput {
        scenario:scenario.name,true_value:scenario.true_value,w_max:scenario.max_weight,
        aggregates,elapsed_seconds:start.elapsed().as_secs_f64(),sample_count:samples_processed,path_count:replicates,
    })
}

fn median(values: &[usize]) -> Option<usize> {
    if values.is_empty() {return None;}
    let mut values=values.to_vec(); values.sort_unstable(); Some(values[(values.len()-1)/2])
}
fn percentile90(values: &[usize]) -> Option<usize> {
    if values.is_empty() { return None; }
    let mut sorted=values.to_vec();
    sorted.sort_unstable();
    let index=((sorted.len()*9+9)/10).saturating_sub(1).min(sorted.len()-1);
    Some(sorted[index])
}

fn emit_scenario(out: &mut String, result: &ScenarioOutput, horizons: &[usize]) {
    use std::fmt::Write as _;
    let _=writeln!(out,"DGP\t{}\treplicates\t{}\ttrue_value\t{:.6}\tW\t{:.6}",result.scenario,result.path_count,result.true_value,result.w_max);
    for (method_idx,method) in METHOD_NAMES.iter().enumerate() {
        for (hidx,&horizon) in horizons.iter().enumerate() {
            let a=&result.aggregates[method_idx][hidx];
            let (point,se_point,lo_point,hi_point)=wilson(a.point_hits,a.trials);
            let (seq,se_seq,lo_seq,hi_seq)=wilson(a.sequence_hits,a.trials);
            let (false_exclusion,se_false_exclusion,lo_false_exclusion,hi_false_exclusion)=wilson(a.trials-a.sequence_hits,a.trials);
            let (prom,se_prom,lo_prom,hi_prom)=wilson(a.promotion_hits,a.trials);
            let (rev,_,_,_)=wilson(a.reversed_hits,a.trials);
            let (width,se_width)=mean_se(a.width_sum,a.width_sq_sum,a.trials);
            let (mean_lower,se_lower)=mean_se(a.lower_sum,a.lower_sq_sum,a.trials);
            let (mean_upper,se_upper)=mean_se(a.upper_sum,a.upper_sq_sum,a.trials);
            let (mean_elapsed,se_elapsed)=mean_se(a.elapsed_seconds_sum,a.elapsed_seconds_sq_sum,a.trials);
            let method_samples_per_second=(a.trials as f64*horizon as f64)/a.elapsed_seconds_sum.max(1e-12);
            let promotion_metric=if result.true_value < PROMOTION_THRESHOLD {"false_promotion_probability"} else {"promotion_power"};
            for (name,estimate,se,lo,hi) in [
                ("pointwise_coverage",point,se_point,lo_point,hi_point),
                ("sequence_coverage",seq,se_seq,lo_seq,hi_seq),
                ("sequence_false_exclusion_probability",false_exclusion,se_false_exclusion,lo_false_exclusion,hi_false_exclusion),
                (promotion_metric,prom,se_prom,lo_prom,hi_prom),
            ] {
                let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",result.scenario,method,horizon,name,estimate,se,lo,hi);
            }
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_interval_width\t{:.8}\t{:.8}\t-\t-",result.scenario,method,horizon,width,se_width);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_lower_endpoint\t{:.8}\t{:.8}\t-\t-",result.scenario,method,horizon,mean_lower,se_lower);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_upper_endpoint\t{:.8}\t{:.8}\t-\t-",result.scenario,method,horizon,mean_upper,se_upper);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tempty_or_reversed_probability\t{:.8}\t-\t-\t-",result.scenario,method,horizon,rev);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmean_method_elapsed_seconds\t{:.8}\t{:.8}\t-\t-",result.scenario,method,horizon,mean_elapsed,se_elapsed);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tmethod_observations_per_second\t{:.3}\t-\t-\t-",result.scenario,method,horizon,method_samples_per_second);
            let times=&a.promotion_times;
            let med=median(times);
            let p90=percentile90(times);
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tpromotion_time_median_conditional\t{}\t-\t-\t-",result.scenario,method,horizon,med.map(|x|x.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
            let _=writeln!(out,"RESULT_METRIC\t{}\t{}\t{}\tpromotion_time_p90_conditional\t{}\t-\t-\t-",result.scenario,method,horizon,p90.map(|x|x.to_string()).unwrap_or_else(||"NOT_REACHED".into()));
            if method_idx==0 {
                let _=writeln!(out,"ROOT_METRIC\t{}\t{}\tchecks\t{}\tfailures\t{}\tcollisions\t{}",result.scenario,horizon,a.root_checks,a.root_failures,a.endpoint_collisions);
            }
            if method_idx==1 {
                let _=writeln!(out,"GRID_METRIC\t{}\t{}\tendpoint_advances\t{}",result.scenario,horizon,a.grid_endpoint_advances);
            }
        }
    }
    let _=writeln!(out,"SYSTEM\t{}\tseconds\t{:.6}\tobservations\t{}\tobservations_per_second\t{:.3}",result.scenario,result.elapsed_seconds,result.sample_count,result.sample_count as f64/result.elapsed_seconds.max(1e-9));
}

fn benchmark(replicates: usize, horizons: &[usize]) -> Result<String,String> {
    let scenarios=[Scenario::healthy_null(),Scenario::healthy_alternative(),Scenario::weak_overlap_null(),Scenario::weak_overlap_alternative()];
    let mut out=String::from("record_type\tscenario\tmethod_or_field\thorizon\tmetric\testimate\tstandard_error\tci_low\tci_high\n");
    let _=writeln!(out,"SEED_SCHEDULE\tbase_seed\t{BASE_SEED}\tformula\tBASE_SEED xor scenario*D6E8FEB86659FD93 xor replicate*A0761D6478BD642F");
    let _=writeln!(out,"BENCHMARK_CONFIG\treplicates\t{replicates}\tmax_horizon\t{}",horizons.iter().max().unwrap_or(&0));
    for (idx,scenario) in scenarios.into_iter().enumerate() {
        let result=simulate_scenario(scenario,idx,replicates,horizons)?;
        emit_scenario(&mut out,&result,horizons);
    }
    Ok(out)
}

fn seed_smoke() -> String {
    let mut rng=SplitMix64::new(BASE_SEED);
    let mut out=String::from("VIS_DVPC_042_SEED_SMOKE\\n");
    for _ in 0..64 { let _=writeln!(out,"{:016x}",rng.next_u64()); }
    out
}

fn main() {
    let args:Vec<String>=env::args().collect();
    let result=match args.get(1).map(String::as_str) {
        Some("--benchmark")=>benchmark(REPLICATES,&HORIZONS),
        Some("--smoke")=>benchmark(8,&[8]),
        Some("--seed-smoke")=>Ok(seed_smoke()),
        _=>Err("usage: vision-dvpc-042-mope-long-horizon-frontier --benchmark|--smoke".to_owned()),
    };
    match result {
        Ok(out)=>print!("{out}"),
        Err(err)=>{eprintln!("VIS_DVPC_042_ERROR\t{err}");std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_seed_replays_identically() {
        let mut a=SplitMix64::new(BASE_SEED);
        let mut b=SplitMix64::new(BASE_SEED);
        for _ in 0..128 { assert_eq!(a.next_u64(),b.next_u64()); }
    }

    #[test]
    fn running_intersection_is_nested_and_never_wider() {
        let previous=Interval{lower:0.2,upper:0.8};
        let raw=Interval{lower:0.1,upper:0.7};
        let got=running_intersection(previous,raw);
        assert!(close(got.lower,0.2,1e-12));
        assert!(close(got.upper,0.7,1e-12));
        assert!(got.width()<=previous.width()+1e-12);
    }

    #[test]
    fn upstream_grid_comparator_is_bounded_and_reports_advances() {
        let mut grid=MopeGridPair::new(1.0).unwrap();
        assert_eq!(grid.values.len(),GRID_SIZE);
        assert!(close(grid.values[0],0.0,1e-12));
        assert!(close(grid.values[GRID_SIZE-1],1.0,1e-12));
        let interval=grid.update(1.0,1.0).unwrap();
        assert!(interval.lower>=0.0 && interval.upper<=1.0);
        assert!(!interval.reversed());
        assert_eq!(grid.diagnostics.endpoint_advances,0);
    }

    #[test]
    fn baseline_intervals_are_bounded_and_spending_schedule_is_two_sided() {
        let mut moments=OpeMoments::default();
        for x in [0.0,0.25,0.5,1.0] { moments.update(x); }
        for method in 1..4 {
            let ci=baseline_interval(method,moments,1.0);
            assert!((0.0..=1.0).contains(&ci.lower));
            assert!((0.0..=1.0).contains(&ci.upper));
        }
        let one_sided_allocations:f64=(1..100_000).map(|t|ALPHA/(2.0*(t as f64)*((t+1) as f64))).sum();
        let two_sided_interval_allocations:f64=(1..100_000).map(|t|ALPHA/((t as f64)*((t+1) as f64))).sum();
        assert!(one_sided_allocations<=ALPHA/2.0+1e-5);
        assert!(two_sided_interval_allocations<=ALPHA+1e-5);
    }

    #[test]
    fn short_seeded_benchmark_replays_and_has_no_mope_root_failures() {
        let scenarios=[Scenario::healthy_null(),Scenario::healthy_alternative(),Scenario::weak_overlap_null(),Scenario::weak_overlap_alternative()];
        for (index,s) in scenarios.into_iter().enumerate() {
            let first=simulate_scenario(s,index,8,&[8]).unwrap();
            let second=simulate_scenario(s,index,8,&[8]).unwrap();
            assert_eq!(first.path_count,second.path_count);
            assert_eq!(first.sample_count,second.sample_count);
            assert_eq!(first.aggregates[0][0].point_hits,second.aggregates[0][0].point_hits);
            assert_eq!(first.aggregates[0][0].sequence_hits,second.aggregates[0][0].sequence_hits);
            assert_eq!(first.aggregates[0][0].root_failures,0);
        }
    }
}
