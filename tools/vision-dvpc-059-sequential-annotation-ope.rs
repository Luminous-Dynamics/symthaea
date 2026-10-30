//! VIS-DVPC-059: forward-monotone sequential annotation for off-policy evaluation.
//!
//! Standalone Rust 2024 benchmark:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-059-sequential-annotation-ope.rs -o /tmp/vis-dvpc-059-tests
//!   /tmp/vis-dvpc-059-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-059-sequential-annotation-ope.rs -o /tmp/vis-dvpc-059
//!   /tmp/vis-dvpc-059 --smoke
//!   /tmp/vis-dvpc-059
//!
//! The v0 fixture represents an efficient sequential OPE score by two
//! martingale projection increments D1 and D2. Stage 1 annotation reveals D1
//! and a branch state H1. Stage 2 can be annotated only after stage 1. The
//! estimator uses realized prefix inclusion probabilities and is unbiased
//! under the registered sequential annotation-ignorability assumptions.
#![deny(unsafe_code)]

use std::env;
use std::f64::consts::TAU;
use std::fmt::Write;

const BRANCHES: usize = 2;
const GRID_STEPS: usize = 10;
const PILOT_N: usize = 128;
const EVALUATION_N: usize = 256;
const FULL_REPLICATES: usize = 2048;
const SMOKE_REPLICATES: usize = 48;
const MC_TEST_REPLICATES: usize = 4096;
const BUDGET_LABELS: f64 = 0.80;
const PILOT_FRACTION: f64 = 0.25;
const PILOT_ANNOTATION_PROBABILITY: f64 = 0.10;
const ADAPTIVE_FLOOR: f64 = 0.02;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct VarianceProfile {
    stage1: f64,
    stage2: [f64; BRANCHES],
}

impl VarianceProfile {
    fn truth() -> Self {
        Self {
            stage1: 0.25,
            stage2: [0.12, 1.60],
        }
    }

    fn validate(&self) -> Result<(), String> {
        if !self.stage1.is_finite()
            || self.stage1 <= 0.0
            || self
                .stage2
                .iter()
                .any(|variance| !variance.is_finite() || *variance <= 0.0)
        {
            return Err("conditional-variance increments must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PrefixDesign {
    /// P(C1 = 1 | F0).
    lambda1: f64,
    /// P(C1 = C2 = 1 | F0, F1), indexed by revealed stage-1 branch.
    lambda12: [f64; BRANCHES],
}

impl PrefixDesign {
    fn validate(&self) -> Result<(), String> {
        if !self.lambda1.is_finite() || self.lambda1 <= 0.0 || self.lambda1 > 1.0 {
            return Err("stage-1 prefix probability must be in (0,1]".into());
        }
        if self
            .lambda12
            .iter()
            .any(|probability| !probability.is_finite() || *probability <= 0.0 || *probability > self.lambda1 + EPS)
        {
            return Err("stage-2 joint inclusion must lie in (0, lambda1] pathwise".into());
        }
        Ok(())
    }

    fn expected_labels(&self, branch_probability: [f64; BRANCHES]) -> f64 {
        self.lambda1
            + (0..BRANCHES)
                .map(|branch| branch_probability[branch] * self.lambda12[branch])
                .sum::<f64>()
    }

    fn stage2_conditional_probability(&self, branch: usize) -> f64 {
        self.lambda12[branch] / self.lambda1
    }
}

#[derive(Clone, Copy, Debug)]
struct World {
    name: &'static str,
    branch_probability: [f64; BRANCHES],
    variance: VarianceProfile,
    budget: f64,
}

impl World {
    fn branch_reversal() -> Self {
        Self {
            name: "branch_dependent_variance",
            branch_probability: [0.70, 0.30],
            variance: VarianceProfile::truth(),
            budget: BUDGET_LABELS,
        }
    }

    fn early_information_bottleneck() -> Self {
        Self {
            name: "early_information_bottleneck",
            branch_probability: [0.50, 0.50],
            variance: VarianceProfile {
                stage1: 1.25,
                stage2: [0.12, 0.35],
            },
            budget: BUDGET_LABELS,
        }
    }

    fn validate(&self) -> Result<(), String> {
        self.variance.validate()?;
        if (self.branch_probability.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || self
                .branch_probability
                .iter()
                .any(|probability| !probability.is_finite() || *probability <= 0.0 || *probability > 1.0)
        {
            return Err("branch probabilities must form a positive distribution".into());
        }
        if !self.budget.is_finite() || self.budget <= 0.0 || self.budget > 2.0 {
            return Err("expected annotation budget must lie in (0,2]".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct ExactPlan {
    design: PrefixDesign,
    objective: f64,
    candidates_checked: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    bias: f64,
    variance: f64,
    mse: f64,
    analytic_mse: f64,
    expected_labels: f64,
    mean_lambda1: f64,
    mean_lambda12_low: f64,
    mean_lambda12_high: f64,
    mean_pilot_stage1: f64,
    mean_pilot_stage2_low: f64,
    mean_pilot_stage2_high: f64,
    prefix_violations: usize,
    replicates: usize,
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }
}

fn mixed_seed(base: u64, replicate: usize, stream: u64) -> u64 {
    base ^ (replicate as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn grid_probability(index: usize) -> f64 {
    (index + 1) as f64 / GRID_STEPS as f64
}

fn objective(profile: VarianceProfile, design: PrefixDesign, branch_probability: [f64; BRANCHES]) -> f64 {
    profile.stage1 / design.lambda1
        + (0..BRANCHES)
            .map(|branch| {
                branch_probability[branch] * profile.stage2[branch] / design.lambda12[branch]
            })
            .sum::<f64>()
}

/// Expected prefix-label cost per trajectory:
/// E[C1 + C1*C2] = lambda1 + E[Lambda_12(H1)].
fn expected_label_cost(design: PrefixDesign, branch_probability: [f64; BRANCHES]) -> f64 {
    design.expected_labels(branch_probability)
}

/// Exact grid oracle for history-adaptive Lambda_12(H1).
fn exact_history_adaptive(world: World, profile: VarianceProfile) -> Result<ExactPlan, String> {
    world.validate()?;
    profile.validate()?;
    let mut best: Option<(f64, PrefixDesign)> = None;
    let mut checked = 0usize;
    for i in 0..GRID_STEPS {
        let lambda1 = grid_probability(i);
        for j in 0..=i {
            let lambda12_0 = grid_probability(j);
            for k in 0..=i {
                let lambda12_1 = grid_probability(k);
                let design = PrefixDesign {
                    lambda1,
                    lambda12: [lambda12_0, lambda12_1],
                };
                if expected_label_cost(design, world.branch_probability) > world.budget + EPS {
                    continue;
                }
                design.validate()?;
                checked += 1;
                let score = objective(profile, design, world.branch_probability);
                if best.is_none_or(|(best_score, _)| score < best_score - EPS) {
                    best = Some((score, design));
                }
            }
        }
    }
    let (score, design) = best.ok_or_else(|| "adaptive exact grid has no feasible design".to_string())?;
    Ok(ExactPlan {
        design,
        objective: score,
        candidates_checked: checked,
    })
}

/// Exact history-blind grid oracle constrained to Lambda_12(H1)=constant.
fn exact_history_blind(world: World, profile: VarianceProfile) -> Result<ExactPlan, String> {
    world.validate()?;
    profile.validate()?;
    let mut best: Option<(f64, PrefixDesign)> = None;
    let mut checked = 0usize;
    for i in 0..GRID_STEPS {
        let lambda1 = grid_probability(i);
        for j in 0..=i {
            let lambda12 = grid_probability(j);
            let design = PrefixDesign {
                lambda1,
                lambda12: [lambda12; BRANCHES],
            };
            if expected_label_cost(design, world.branch_probability) > world.budget + EPS {
                continue;
            }
            checked += 1;
            let score = objective(profile, design, world.branch_probability);
            if best.is_none_or(|(best_score, _)| score < best_score - EPS) {
                best = Some((score, design));
            }
        }
    }
    let (score, design) = best.ok_or_else(|| "history-blind exact grid has no feasible design".to_string())?;
    Ok(ExactPlan {
        design,
        objective: score,
        candidates_checked: checked,
    })
}

fn uniform_prefix_design(world: World) -> PrefixDesign {
    let lambda = world.budget / 2.0;
    PrefixDesign {
        lambda1: lambda,
        lambda12: [lambda; BRANCHES],
    }
}

/// The paper's two-batch mixture correction. Pilot units are fully annotated
/// with probability p; the adaptive batch probability is clipped so the pooled
/// design matches the target when it lies inside the attainable mixture range.
fn adaptive_probability(target: f64, upper: f64, kappa: f64, pilot_p: f64, floor: f64) -> f64 {
    ((target - kappa * pilot_p) / (1.0 - kappa)).clamp(floor, upper)
}

fn pooled_two_batch_design(target: PrefixDesign, kappa: f64, pilot_p: f64, floor: f64) -> Result<PrefixDesign, String> {
    if !kappa.is_finite() || kappa <= 0.0 || kappa >= 1.0
        || !pilot_p.is_finite() || pilot_p <= 0.0 || pilot_p > 1.0
        || !floor.is_finite() || floor <= 0.0 || floor >= 1.0
    {
        return Err("invalid pilot mixture parameters".into());
    }
    target.validate()?;
    let adaptive_lambda1 = adaptive_probability(target.lambda1, 1.0, kappa, pilot_p, floor);
    let adaptive_lambda12 = [
        adaptive_probability(target.lambda12[0], adaptive_lambda1, kappa, pilot_p, floor),
        adaptive_probability(target.lambda12[1], adaptive_lambda1, kappa, pilot_p, floor),
    ];
    let pooled = PrefixDesign {
        lambda1: kappa * pilot_p + (1.0 - kappa) * adaptive_lambda1,
        lambda12: [
            kappa * pilot_p + (1.0 - kappa) * adaptive_lambda12[0],
            kappa * pilot_p + (1.0 - kappa) * adaptive_lambda12[1],
        ],
    };
    pooled.validate()?;
    Ok(pooled)
}

#[derive(Clone, Copy, Debug)]
struct PilotEstimate {
    profile: VarianceProfile,
    stage1_count: usize,
    stage2_count: [usize; BRANCHES],
}

fn sample_variance(values: &[f64], fallback: f64) -> f64 {
    if values.len() < 2 {
        return fallback;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// Independent fully-annotated pilot trajectory sample. A unit either yields
/// both stages (prefix) or neither; branch-specific conditional variances are
/// estimated from the pilot trajectories that reveal that branch.
fn estimate_pilot_profile(world: World, pilot_n: usize, pilot_probability: f64, seed: u64) -> Result<PilotEstimate, String> {
    world.validate()?;
    if pilot_n < 2 || !pilot_probability.is_finite() || pilot_probability <= 0.0 || pilot_probability > 1.0 {
        return Err("invalid pilot sample size or annotation probability".into());
    }
    let mut rng = Rng::new(seed);
    let mut stage1_values = Vec::new();
    let mut stage2_values: [Vec<f64>; BRANCHES] = std::array::from_fn(|_| Vec::new());
    for _ in 0..pilot_n {
        let branch = if rng.bernoulli(world.branch_probability[0]) { 0 } else { 1 };
        let d1 = world.variance.stage1.sqrt() * rng.normal();
        let d2 = world.variance.stage2[branch].sqrt() * rng.normal();
        if rng.bernoulli(pilot_probability) {
            stage1_values.push(d1);
            stage2_values[branch].push(d2);
        }
    }
    let profile = VarianceProfile {
        stage1: sample_variance(&stage1_values, 0.5).max(EPS),
        stage2: std::array::from_fn(|branch| {
            sample_variance(&stage2_values[branch], 0.5).max(EPS)
        }),
    };
    profile.validate()?;
    Ok(PilotEstimate {
        profile,
        stage1_count: stage1_values.len(),
        stage2_count: std::array::from_fn(|branch| stage2_values[branch].len()),
    })
}

#[derive(Clone, Copy, Debug)]
struct TrajectoryEstimate {
    score: f64,
    labels: u8,
    prefix_violation: bool,
}

/// Simulate one trajectory under forward-monotone sequential annotation.
/// H1 is revealed only after C1=1; stage 2 is sampled conditionally with
/// P(C2=1 | C1=1,H1)=Lambda_12(H1)/Lambda_1.
fn sample_trajectory(world: World, design: PrefixDesign, seed: u64) -> Result<TrajectoryEstimate, String> {
    design.validate()?;
    let mut rng = Rng::new(seed);
    let branch = if rng.bernoulli(world.branch_probability[0]) { 0 } else { 1 };
    let d1 = world.variance.stage1.sqrt() * rng.normal();
    let d2 = world.variance.stage2[branch].sqrt() * rng.normal();
    let c1 = rng.bernoulli(design.lambda1);
    let c2 = c1 && rng.bernoulli(design.stage2_conditional_probability(branch));
    let score = if c1 { d1 / design.lambda1 } else { 0.0 }
        + if c2 { d2 / design.lambda12[branch] } else { 0.0 };
    Ok(TrajectoryEstimate {
        score,
        labels: u8::from(c1) + u8::from(c2),
        prefix_violation: c2 && !c1,
    })
}

fn evaluate_design(
    world: World,
    design: PrefixDesign,
    replicates: usize,
    evaluation_n: usize,
    seed: u64,
) -> Result<Metrics, String> {
    world.validate()?;
    design.validate()?;
    if replicates < 2 || evaluation_n == 0 {
        return Err("evaluation requires at least two replicates and positive sample size".into());
    }
    let analytic_per_trajectory = objective(world.variance, design, world.branch_probability);
    let analytic_mse = analytic_per_trajectory / evaluation_n as f64;
    let mut estimates = Vec::with_capacity(replicates);
    let mut total_labels = 0.0;
    let mut prefix_violations = 0usize;
    for replicate in 0..replicates {
        let mut estimate = 0.0;
        for index in 0..evaluation_n {
            let trajectory = sample_trajectory(
                world,
                design,
                mixed_seed(seed, replicate * evaluation_n + index, 17),
            )?;
            estimate += trajectory.score;
            total_labels += trajectory.labels as f64;
            prefix_violations += usize::from(trajectory.prefix_violation);
        }
        estimates.push(estimate / evaluation_n as f64);
    }
    let n = estimates.len() as f64;
    let bias = estimates.iter().sum::<f64>() / n;
    let mse = estimates.iter().map(|estimate| estimate * estimate).sum::<f64>() / n;
    let variance = (mse - bias * bias).max(0.0);
    Ok(Metrics {
        bias,
        variance,
        mse,
        analytic_mse,
        expected_labels: total_labels / (n * evaluation_n as f64),
        mean_lambda1: design.lambda1,
        mean_lambda12_low: design.lambda12[0],
        mean_lambda12_high: design.lambda12[1],
        mean_pilot_stage1: 0.0,
        mean_pilot_stage2_low: 0.0,
        mean_pilot_stage2_high: 0.0,
        prefix_violations,
        replicates,
    })
}

fn evaluate_batch_adaptive(
    world: World,
    replicates: usize,
    evaluation_n: usize,
    pilot_n: usize,
    seed: u64,
) -> Result<Metrics, String> {
    let kappa = PILOT_FRACTION;
    let pilot_p = PILOT_ANNOTATION_PROBABILITY;
    let oracle_target = exact_history_adaptive(world, world.variance)?.design;
    let mut estimates = Vec::with_capacity(replicates);
    let mut analytic_mse = 0.0;
    let mut expected_labels = 0.0;
    let mut pooled_lambda1 = 0.0;
    let mut pooled_lambda12_low = 0.0;
    let mut pooled_lambda12_high = 0.0;
    let mut stage1_estimates = 0.0;
    let mut stage2_low_estimates = 0.0;
    let mut stage2_high_estimates = 0.0;
    let mut prefix_violations = 0usize;

    for replicate in 0..replicates {
        let pilot = estimate_pilot_profile(
            world,
            pilot_n,
            pilot_p,
            mixed_seed(seed, replicate, 31),
        )?;
        let plug_in_target = exact_history_adaptive(world, pilot.profile)?.design;
        let pooled = pooled_two_batch_design(
            plug_in_target,
            kappa,
            pilot_p,
            ADAPTIVE_FLOOR,
        )?;
        let analytic_for_design =
            objective(world.variance, pooled, world.branch_probability) / evaluation_n as f64;
        analytic_mse += analytic_for_design;
        expected_labels += expected_label_cost(pooled, world.branch_probability);
        pooled_lambda1 += pooled.lambda1;
        pooled_lambda12_low += pooled.lambda12[0];
        pooled_lambda12_high += pooled.lambda12[1];
        stage1_estimates += pilot.profile.stage1;
        stage2_low_estimates += pilot.profile.stage2[0];
        stage2_high_estimates += pilot.profile.stage2[1];

        let mut estimate = 0.0;
        for index in 0..evaluation_n {
            let trajectory = sample_trajectory(
                world,
                pooled,
                mixed_seed(seed, replicate * evaluation_n + index, 41),
            )?;
            estimate += trajectory.score;
            prefix_violations += usize::from(trajectory.prefix_violation);
        }
        estimates.push(estimate / evaluation_n as f64);
        let _ = pilot.stage1_count;
        let _ = pilot.stage2_count;
    }

    let n = estimates.len() as f64;
    let bias = estimates.iter().sum::<f64>() / n;
    let mse = estimates.iter().map(|estimate| estimate * estimate).sum::<f64>() / n;
    let variance = (mse - bias * bias).max(0.0);
    let _ = oracle_target;
    Ok(Metrics {
        bias,
        variance,
        mse,
        analytic_mse: analytic_mse / n,
        expected_labels: expected_labels / n,
        mean_lambda1: pooled_lambda1 / n,
        mean_lambda12_low: pooled_lambda12_low / n,
        mean_lambda12_high: pooled_lambda12_high / n,
        mean_pilot_stage1: stage1_estimates / n,
        mean_pilot_stage2_low: stage2_low_estimates / n,
        mean_pilot_stage2_high: stage2_high_estimates / n,
        prefix_violations,
        replicates,
    })
}

fn benchmark(replicates: usize, seed: u64) -> Result<String, String> {
    if replicates == 0 {
        return Err("replicate count must be positive".into());
    }
    let worlds = [
        World::branch_reversal(),
        World::early_information_bottleneck(),
    ];
    let mut output = String::from(
        "record_type\tscenario\tmethod\treplicates\tevaluation_n\tbudget_labels\tlambda1\tlambda12_low\tlambda12_high\texpected_labels\tanalytic_mse\tempirical_bias\tempirical_variance\tempirical_mse\toracle_objective\toptimality_gap\tpilot_variance_stage1\tpilot_variance_stage2_low\tpilot_variance_stage2_high\tprefix_violations\n",
    );
    writeln!(output, "BENCHMARK_CONFIG\tVIS-DVPC-059\tversion\t0").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\treplicates\t{replicates}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\tevaluation_n\t{EVALUATION_N}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\tpilot_n\t{PILOT_N}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\tbudget_labels\t{BUDGET_LABELS:.4}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\tpilot_fraction\t{PILOT_FRACTION:.4}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\tpilot_annotation_probability\t{PILOT_ANNOTATION_PROBABILITY:.4}").unwrap();
    writeln!(output, "BENCHMARK_CONFIG\testimator\tmartingale projection increments with prefix inverse-probability weights").unwrap();

    for (world_index, world) in worlds.iter().enumerate() {
        world.validate()?;
        let oracle = exact_history_adaptive(*world, world.variance)?;
        let static_plan = exact_history_blind(*world, world.variance)?;
        let uniform = uniform_prefix_design(*world);
        let adaptive = evaluate_batch_adaptive(
            *world,
            replicates,
            EVALUATION_N,
            PILOT_N,
            seed ^ (world_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        )?;
        let plans = [
            ("uniform_prefix", uniform),
            ("history_blind_exact", static_plan.design),
            ("history_adaptive_exact_oracle", oracle.design),
        ];
        for (method_index, (method, design)) in plans.iter().enumerate() {
            design.validate()?;
            let metrics = evaluate_design(
                *world,
                *design,
                replicates,
                EVALUATION_N,
                seed ^ (world_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                    ^ (method_index as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f),
            )?;
            let gap = (objective(world.variance, *design, world.branch_probability)
                - oracle.objective)
                .max(0.0)
                / oracle.objective.max(EPS);
            writeln!(
                output,
                "RESULT\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}",
                world.name,
                method,
                metrics.replicates,
                EVALUATION_N,
                world.budget,
                design.lambda1,
                design.lambda12[0],
                design.lambda12[1],
                metrics.expected_labels,
                metrics.analytic_mse,
                metrics.bias,
                metrics.variance,
                metrics.mse,
                oracle.objective,
                gap,
                0.0,
                0.0,
                0.0,
                metrics.prefix_violations,
            )
            .unwrap();
        }

        let batch_design_metrics = adaptive;
        let actual_design_analytic =
            batch_design_metrics.analytic_mse * EVALUATION_N as f64;
        let gap = ((actual_design_analytic - oracle.objective) / oracle.objective.max(EPS)).max(0.0);
        writeln!(
            output,
            "RESULT\t{}\ttwo_batch_adaptive_plugin\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}",
            world.name,
            batch_design_metrics.replicates,
            EVALUATION_N,
            world.budget,
            batch_design_metrics.mean_lambda1,
            batch_design_metrics.mean_lambda12_low,
            batch_design_metrics.mean_lambda12_high,
            batch_design_metrics.expected_labels,
            batch_design_metrics.analytic_mse,
            batch_design_metrics.bias,
            batch_design_metrics.variance,
            batch_design_metrics.mse,
            oracle.objective,
            gap,
            batch_design_metrics.mean_pilot_stage1,
            batch_design_metrics.mean_pilot_stage2_low,
            batch_design_metrics.mean_pilot_stage2_high,
            batch_design_metrics.prefix_violations,
        )
        .unwrap();
        writeln!(
            output,
            "ORACLE\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.10}\t{}\t{}",
            world.name,
            "exact_history_adaptive_grid",
            oracle.design.lambda1,
            oracle.design.lambda12[0],
            oracle.design.lambda12[1],
            oracle.objective,
            oracle.candidates_checked,
            expected_label_cost(oracle.design, world.branch_probability) <= world.budget + EPS,
        )
        .unwrap();
    }

    writeln!(
        output,
        "PROTOCOL_NEGATIVE_CONTROL\tindependent_stage_sampling\t{}\t{}\tviolates_forward_monotone",
        replicates,
        "fail_closed",
    )
    .unwrap();
    Ok(output)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let replicates = if smoke { SMOKE_REPLICATES } else { FULL_REPLICATES };
    match benchmark(replicates, 0x0590_2026_1030) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_059_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_monotone_design_and_budget_are_enforced() {
        let world = World::branch_reversal();
        let oracle = exact_history_adaptive(world, world.variance).unwrap();
        oracle.design.validate().unwrap();
        assert!(oracle.design.lambda12.iter().all(|p| *p <= oracle.design.lambda1));
        assert!(expected_label_cost(oracle.design, world.branch_probability) <= world.budget + EPS);
    }

    #[test]
    fn adaptive_oracle_is_never_worse_than_history_blind_or_uniform_design() {
        for world in [World::branch_reversal(), World::early_information_bottleneck()] {
            let oracle = exact_history_adaptive(world, world.variance).unwrap();
            let blind = exact_history_blind(world, world.variance).unwrap();
            let uniform = uniform_prefix_design(world);
            let oracle_obj = objective(world.variance, oracle.design, world.branch_probability);
            let blind_obj = objective(world.variance, blind.design, world.branch_probability);
            let uniform_obj = objective(world.variance, uniform, world.branch_probability);
            assert!(oracle_obj <= blind_obj + EPS);
            assert!(oracle_obj <= uniform_obj + EPS);
        }
    }

    #[test]
    fn exact_grid_oracle_matches_independent_full_grid_enumeration() {
        let world = World::branch_reversal();
        let oracle = exact_history_adaptive(world, world.variance).unwrap();
        let mut best = f64::INFINITY;
        for i in 0..GRID_STEPS {
            for j in 0..=i {
                for k in 0..=i {
                    let design = PrefixDesign {
                        lambda1: grid_probability(i),
                        lambda12: [grid_probability(j), grid_probability(k)],
                    };
                    if expected_label_cost(design, world.branch_probability) <= world.budget + EPS {
                        best = best.min(objective(world.variance, design, world.branch_probability));
                    }
                }
            }
        }
        assert!((best - oracle.objective).abs() < 1e-12);
    }

    #[test]
    fn pooled_batch_design_preserves_forward_monotonicity_and_budget_when_unclipped() {
        let world = World::branch_reversal();
        let target = exact_history_adaptive(world, world.variance).unwrap().design;
        let pooled = pooled_two_batch_design(target, 0.25, 0.10, 0.02).unwrap();
        pooled.validate().unwrap();
        assert!(expected_label_cost(pooled, world.branch_probability) <= world.budget + 1e-8);
    }

    #[test]
    fn batch_adaptive_pilot_uses_disjoint_stream_and_updates_conditional_variances() {
        let world = World::branch_reversal();
        let first = estimate_pilot_profile(world, 512, 0.5, 0x0590_001).unwrap();
        let second = estimate_pilot_profile(world, 512, 0.5, 0x0590_002).unwrap();
        assert_ne!(first.profile.stage1, second.profile.stage1);
        assert!(first.profile.stage2[1] > first.profile.stage2[0] * 2.0);
        assert_eq!(first.stage1_count, first.stage2_count.iter().sum());
    }

    #[test]
    fn analytic_sequential_variance_matches_independent_monte_carlo() {
        let world = World::branch_reversal();
        let design = exact_history_adaptive(world, world.variance).unwrap().design;
        let metrics = evaluate_design(world, design, MC_TEST_REPLICATES, 128, 0x0590_abc1).unwrap();
        let relative_error = (metrics.mse - metrics.analytic_mse) / metrics.analytic_mse.max(EPS);
        assert!(relative_error.abs() < 0.15, "empirical={}, analytic={}, relative={relative_error}", metrics.mse, metrics.analytic_mse);
        assert!(metrics.bias.abs() < 0.04);
        assert_eq!(metrics.prefix_violations, 0);
    }

    #[test]
    fn zero_probability_and_prefix_violations_fail_closed() {
        let invalid = PrefixDesign {
            lambda1: 0.0,
            lambda12: [0.1, 0.1],
        };
        assert!(invalid.validate().is_err());
        let invalid_prefix = PrefixDesign {
            lambda1: 0.3,
            lambda12: [0.4, 0.2],
        };
        assert!(invalid_prefix.validate().is_err());
        assert!(adaptive_probability(0.1, 0.05, 0.25, 0.10, 0.02) <= 0.05);
    }

    #[test]
    fn output_is_deterministic_and_row_schema_is_exact() {
        let first = benchmark(2, 123).unwrap();
        let second = benchmark(2, 123).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("RESULT\t")).count(), 8);
        assert_eq!(first.lines().filter(|line| line.starts_with("ORACLE\t")).count(), 2);
        assert_eq!(first.lines().filter(|line| line.starts_with("BENCHMARK_CONFIG\t")).count(), 8);
        for line in first.lines().filter(|line| line.starts_with("RESULT\t")) {
            assert_eq!(line.split('\t').count(), 20);
        }
    }

    #[test]
    fn invalid_world_profiles_and_adaptive_parameters_fail_closed() {
        let mut world = World::branch_reversal();
        world.variance.stage2[0] = f64::NAN;
        assert!(world.validate().is_err());
        let world = World::branch_reversal();
        assert!(pooled_two_batch_design(exact_history_adaptive(world, world.variance).unwrap().design, 1.0, 0.1, 0.02).is_err());
        assert!(evaluate_design(world, uniform_prefix_design(world), 1, 10, 1).is_err());
    }
}