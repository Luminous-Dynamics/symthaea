//! Multi-seed stability calibration for the Gaussian terminal-test frontier.
//!
//! Repeats the same fixed simple Gaussian null/alternative campaign over
//! several frozen root seeds. Every policy within a path sees the same
//! observation sequence, while each stopping threshold receives a separate
//! randomizer stream. This is a numerical stability diagnostic, not a new
//! sequential-testing method or a physical measurement validation.

use falsification_power_reference::gaussian_terminal_induction::GaussianTerminalInducedTest;
use std::collections::BTreeSet;
use std::error::Error;
use std::f64::consts::TAU;

const PATHS_PER_ROOT_SEED: u64 = 20_000;
const HORIZON: usize = 80;
const ALPHA: f64 = 0.05;
const DISCREPANCY: f64 = 0.22;
const SIGMA: f64 = 0.80;
const STOP_THRESHOLDS: [f64; 3] = [0.75, 0.90, 0.99];
const RANDOMIZER_SALTS: [u64; 3] = [
    0xa5a5_5a5a_1234_9876,
    0x6a09_e667_f3bc_c909,
    0xbb67_ae85_84ca_a73b,
];
const ROOT_SEEDS: [u64; 5] = [
    0x4f1b_bcdd_6a09_e667,
    0xeabe_e687_783d_7e11,
    0x2512_5aba_99b5_2f6e,
    0xf47c_1258_eec3_415c,
    0x7375_4faf_949d_1e4c,
];
const PATH_SEED_MULTIPLIER: u64 = 0xd6e8_feb8_6659_fd93;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Copy)]
enum Scenario {
    Null,
    Alternative,
}

impl Scenario {
    fn id(self) -> &'static str {
        match self {
            Self::Null => "fixed_gaussian_simple_null",
            Self::Alternative => "fixed_gaussian_declared_alternative",
        }
    }

    fn seed_offset(self) -> u64 {
        match self {
            Self::Null => 0x1111_2222_3333_4444,
            Self::Alternative => 0x9999_aaaa_bbbb_cccc,
        }
    }

    fn mean(self) -> f64 {
        match self {
            Self::Null => 0.0,
            Self::Alternative => DISCREPANCY,
        }
    }
}

/// SplitMix64 / Box-Muller stream for reproducible simulation only.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(0.5 / 9_007_199_254_740_992.0);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

#[derive(Debug, Clone, Default)]
struct PolicyCounts {
    randomized_rejections: u64,
    early_stops: u64,
    total_stop_looks: u64,
}

#[derive(Debug, Clone)]
struct Counts {
    terminal_np_rejections: u64,
    terminal_induced_mismatches: u64,
    policies: Vec<PolicyCounts>,
}

impl Counts {
    fn new() -> Self {
        Self {
            terminal_np_rejections: 0,
            terminal_induced_mismatches: 0,
            policies: vec![PolicyCounts::default(); STOP_THRESHOLDS.len()],
        }
    }

    fn merge(&mut self, other: &Self) {
        self.terminal_np_rejections += other.terminal_np_rejections;
        self.terminal_induced_mismatches += other.terminal_induced_mismatches;
        for (left, right) in self.policies.iter_mut().zip(&other.policies) {
            left.randomized_rejections += right.randomized_rejections;
            left.early_stops += right.early_stops;
            left.total_stop_looks += right.total_stop_looks;
        }
    }
}

struct SeedResult {
    root_seed: u64,
    counts: Counts,
}

fn wilson_95(successes: u64, trials: u64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_Z_95 * WILSON_Z_95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = WILSON_Z_95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn policy_json(counts: &Counts, paths: u64) -> String {
    counts
        .policies
        .iter()
        .zip(STOP_THRESHOLDS)
        .map(|(policy, threshold)| {
            let (low, high) = wilson_95(policy.randomized_rejections, paths);
            format!(
                concat!(
                    "{{",
                    "\"stop_threshold\":{:.4},",
                    "\"randomized_rejection_rate\":{:.8},",
                    "\"randomized_rejection_wilson95\":[{:.8},{:.8}],",
                    "\"early_stop_rate\":{:.8},",
                    "\"mean_stop_look\":{:.4}",
                    "}}"
                ),
                threshold,
                policy.randomized_rejections as f64 / paths as f64,
                low,
                high,
                policy.early_stops as f64 / paths as f64,
                policy.total_stop_looks as f64 / paths as f64
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn run_seed(
    scenario: Scenario,
    root_seed: u64,
    design: &GaussianTerminalInducedTest,
) -> Result<SeedResult, Box<dyn Error>> {
    let mut counts = Counts::new();
    for path in 0..PATHS_PER_ROOT_SEED {
        let seed = root_seed ^ scenario.seed_offset() ^ path.wrapping_mul(PATH_SEED_MULTIPLIER);
        let mut observation_rng = SplitMix64::new(seed);
        let mut observations = Vec::with_capacity(HORIZON);
        let mut induced_stops = vec![None; STOP_THRESHOLDS.len()];

        for look in 1..=HORIZON {
            let observed = scenario.mean() + SIGMA * observation_rng.standard_normal();
            observations.push(observed);
            let prefix = design.evaluate_prefix(&observations)?;
            for (policy_index, threshold) in STOP_THRESHOLDS.iter().copied().enumerate() {
                if induced_stops[policy_index].is_none()
                    && (prefix.conditional_rejection_probability >= threshold || look == HORIZON)
                {
                    induced_stops[policy_index] = Some(prefix);
                }
            }
        }

        let terminal = design.evaluate_prefix(&observations)?;
        let terminal_np_reject = terminal.conditional_rejection_probability >= 0.5;
        if terminal.conditional_rejection_probability != if terminal_np_reject { 1.0 } else { 0.0 } {
            counts.terminal_induced_mismatches += 1;
        }
        if terminal_np_reject {
            counts.terminal_np_rejections += 1;
        }

        for (policy_index, policy) in counts.policies.iter_mut().enumerate() {
            let stop = induced_stops[policy_index]
                .expect("the fixed horizon always produces a stop value");
            if stop.observations < HORIZON {
                policy.early_stops += 1;
            }
            policy.total_stop_looks += stop.observations as u64;
            let mut randomizer_rng = SplitMix64::new(seed ^ RANDOMIZER_SALTS[policy_index]);
            if stop.randomized_reject(randomizer_rng.uniform())? {
                policy.randomized_rejections += 1;
            }
        }
    }

    Ok(SeedResult { root_seed, counts })
}

fn seed_json(result: &SeedResult, analytic_power: f64) -> String {
    let paths = PATHS_PER_ROOT_SEED;
    let (low, high) = wilson_95(result.counts.terminal_np_rejections, paths);
    format!(
        concat!(
            "{{",
            "\"root_seed\":\"0x{:016x}\",",
            "\"paths\":{},",
            "\"terminal_np_rejection_rate\":{:.8},",
            "\"terminal_np_wilson95\":[{:.8},{:.8}],",
            "\"analytic_terminal_np_power\":{:.8},",
            "\"induced_stop_policies\":[{}],",
            "\"terminal_induced_mismatches\":{}",
            "}}"
        ),
        result.root_seed,
        paths,
        result.counts.terminal_np_rejections as f64 / paths as f64,
        low,
        high,
        analytic_power,
        policy_json(&result.counts, paths),
        result.counts.terminal_induced_mismatches
    )
}

fn scenario_json(scenario: Scenario, results: &[SeedResult], analytic_power: f64) -> String {
    let mut pooled = Counts::new();
    for result in results {
        pooled.merge(&result.counts);
    }
    let total_paths = PATHS_PER_ROOT_SEED * results.len() as u64;
    let (terminal_low, terminal_high) = wilson_95(pooled.terminal_np_rejections, total_paths);
    let per_seed = results
        .iter()
        .map(|result| seed_json(result, analytic_power))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",",
            "\"paths_per_root_seed\":{},",
            "\"paths\":{},",
            "\"horizon\":{},",
            "\"terminal_np_rejection_rate\":{:.8},",
            "\"terminal_np_wilson95\":[{:.8},{:.8}],",
            "\"analytic_terminal_np_power\":{:.8},",
            "\"induced_stop_policies\":[{}],",
            "\"terminal_induced_mismatches\":{},",
            "\"per_seed\":[{}]",
            "}}"
        ),
        scenario.id(),
        PATHS_PER_ROOT_SEED,
        total_paths,
        HORIZON,
        pooled.terminal_np_rejections as f64 / total_paths as f64,
        terminal_low,
        terminal_high,
        analytic_power,
        policy_json(&pooled, total_paths),
        pooled.terminal_induced_mismatches,
        per_seed
    )
}

fn run_scenario(
    scenario: Scenario,
    design: &GaussianTerminalInducedTest,
) -> Result<Vec<SeedResult>, Box<dyn Error>> {
    ROOT_SEEDS
        .iter()
        .copied()
        .map(|seed| run_seed(scenario, seed, design))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let unique_seeds = ROOT_SEEDS.iter().copied().collect::<BTreeSet<_>>();
    if unique_seeds.len() != ROOT_SEEDS.len() {
        return Err("root seed list contains duplicates".into());
    }
    let design =
        GaussianTerminalInducedTest::new(ALPHA, vec![DISCREPANCY; HORIZON], vec![SIGMA; HORIZON])?;
    let null = run_scenario(Scenario::Null, &design)?;
    let alternative = run_scenario(Scenario::Alternative, &design)?;
    let seeds_json = ROOT_SEEDS
        .iter()
        .map(|seed| format!("\"0x{seed:016x}\""))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        concat!(
            "{{",
            "\"schema\":\"gaussian_terminal_test_induction_multiseed_v0\",",
            "\"root_seed_count\":{},",
            "\"root_seeds\":[{}],",
            "\"paths_per_root_seed\":{},",
            "\"paths_per_scenario\":{},",
            "\"maximum_looks\":{},",
            "\"nominal_alpha\":{:.8},",
            "\"induced_stop_thresholds\":[{:.4},{:.4},{:.4}],",
            "\"design_information\":{:.8},",
            "\"analytic_terminal_np_power\":{:.8},",
            "\"claim\":\"multi-seed synthetic calibration stability diagnostic for one frozen simple Gaussian design only\",",
            "\"scenarios\":[{},{}]",
            "}}"
        ),
        ROOT_SEEDS.len(),
        seeds_json,
        PATHS_PER_ROOT_SEED,
        PATHS_PER_ROOT_SEED * ROOT_SEEDS.len() as u64,
        HORIZON,
        ALPHA,
        STOP_THRESHOLDS[0],
        STOP_THRESHOLDS[1],
        STOP_THRESHOLDS[2],
        design.total_information(),
        design.terminal_power(),
        scenario_json(Scenario::Null, &null, design.terminal_power()),
        scenario_json(Scenario::Alternative, &alternative, design.terminal_power())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_root_seeds_are_unique_and_reproducible() {
        let unique = ROOT_SEEDS.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), ROOT_SEEDS.len());
        let mut first = SplitMix64::new(ROOT_SEEDS[0]);
        let mut replay = SplitMix64::new(ROOT_SEEDS[0]);
        let mut other = SplitMix64::new(ROOT_SEEDS[1]);
        for _ in 0..16 {
            assert_eq!(first.next_u64(), replay.next_u64());
        }
        assert_ne!(replay.next_u64(), other.next_u64());
    }

    #[test]
    fn analytic_terminal_power_matches_frozen_known_answer() {
        let design = GaussianTerminalInducedTest::new(
            ALPHA,
            vec![DISCREPANCY; HORIZON],
            vec![SIGMA; HORIZON],
        )
        .unwrap();
        assert!((design.total_information() - 6.05).abs() < 1.0e-12);
        assert!((design.terminal_power() - 0.792_412_72).abs() < 1.0e-7);
    }
}
