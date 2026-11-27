#![forbid(unsafe_code)]

//! Paired Monte Carlo comparison of CMAQ-031 fixed-bet and CMAQ-033 ONS PRMs.
//! The deterministic RNG is simulation-only and is not suitable for production.

use std::collections::HashSet;

use cmaq_031_anytime_calibration_monitor::{
    AnytimeCalibrationMonitor, CalibrationRankReference, CalibrationRankScore,
    DeploymentRankObservation, MonitorUpdate, PreviousOutcome, RankFeature, centered_rank_payoff,
};
use cmaq_033_prm_ons_monitor::{OnsPredictiveRankMonitor, OnsUpdate};

const BASE_SCORE: u64 = 10_000;
const SCORE_WIDTH: u64 = 1_000;
const PROFILE_SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const SCOPE_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const CALIBRATION_COUNT: usize = 99;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub replications: usize,
    pub horizon: usize,
    pub seed: u64,
}

impl BenchmarkConfig {
    pub fn validate(self) -> Result<Self, String> {
        if self.replications == 0 {
            return Err("replications must be positive".into());
        }
        if self.horizon == 0 {
            return Err("horizon must be positive".into());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    Null,
    UpperLocationShift,
    LowerLocationShift,
    DispersionExpansion,
    DispersionContraction,
    OrthogonalTrimodal,
}

impl Scenario {
    pub const ALL: [Self; 6] = [
        Self::Null,
        Self::UpperLocationShift,
        Self::LowerLocationShift,
        Self::DispersionExpansion,
        Self::DispersionContraction,
        Self::OrthogonalTrimodal,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::UpperLocationShift => "upper_location_shift",
            Self::LowerLocationShift => "lower_location_shift",
            Self::DispersionExpansion => "dispersion_expansion",
            Self::DispersionContraction => "dispersion_contraction",
            Self::OrthogonalTrimodal => "orthogonal_trimodal",
        }
    }

    fn stream_id(self) -> u64 {
        match self {
            Self::Null => 0x0101,
            Self::UpperLocationShift => 0x0202,
            Self::LowerLocationShift => 0x0303,
            Self::DispersionExpansion => 0x0404,
            Self::DispersionContraction => 0x0505,
            Self::OrthogonalTrimodal => 0x0606,
        }
    }

    fn sample(self, rng: &mut SplitMix64) -> u64 {
        let draw = rng.below(SCORE_WIDTH) as i64;
        let centered = draw - (SCORE_WIDTH as i64 / 2);
        match self {
            Self::Null => BASE_SCORE + draw as u64,
            Self::UpperLocationShift => BASE_SCORE + 250 + draw as u64,
            Self::LowerLocationShift => BASE_SCORE - 250 + draw as u64,
            Self::DispersionExpansion => (BASE_SCORE as i64 + 500 + 2 * centered) as u64,
            Self::DispersionContraction => (BASE_SCORE as i64 + 500 + centered / 2) as u64,
            Self::OrthogonalTrimodal => {
                // Three symmetric rank masses preserve ideal mean rank and mean
                // absolute rank deviation while changing higher-order shape.
                let category = rng.below(10_000);
                if category < 3_125 {
                    BASE_SCORE + 100
                } else if category < 6_875 {
                    BASE_SCORE + 500
                } else {
                    BASE_SCORE + 900
                }
            }
        }
    }
}

/// SplitMix64 is used strictly as a deterministic simulation PRNG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    pub fn below(&mut self, upper_exclusive: u64) -> u64 {
        assert!(upper_exclusive > 0);
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % upper_exclusive;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MethodSummary {
    pub method: &'static str,
    pub alarm_delays: Vec<usize>,
    pub replications: usize,
    pub horizon: usize,
}

impl MethodSummary {
    pub fn alarm_count(&self) -> usize {
        self.alarm_delays.len()
    }

    pub fn detection_rate(&self) -> f64 {
        self.alarm_count() as f64 / self.replications as f64
    }

    pub fn median_delay(&self) -> Option<usize> {
        quantile(&self.alarm_delays, 0.50)
    }

    pub fn p90_delay(&self) -> Option<usize> {
        quantile(&self.alarm_delays, 0.90)
    }

    pub fn p95_delay(&self) -> Option<usize> {
        quantile(&self.alarm_delays, 0.95)
    }

    pub fn wilson_interval(&self) -> (f64, f64) {
        wilson_interval(self.alarm_count(), self.replications)
    }
}

/// Fixed-bet control using exactly the ONS Order + Dispersion features.
/// This separates the effect of adaptive betting from the effect of feature count.
#[derive(Clone, Debug, PartialEq)]
struct MatchedFixedBetMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    log_wealth: [f64; 2],
    quarantined: bool,
}

impl MatchedFixedBetMonitor {
    fn new(
        calibration_count: usize,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, String> {
        if calibration_count == 0
            || alpha_numerator == 0
            || alpha_denominator == 0
            || alpha_numerator >= alpha_denominator
        {
            return Err("invalid matched fixed-bet configuration".into());
        }
        Ok(Self {
            calibration_count,
            alpha_numerator,
            alpha_denominator,
            rank_counts: vec![0; calibration_count + 1],
            observations_seen: 0,
            log_wealth: [0.0; 2],
            quarantined: false,
        })
    }

    fn observe_rank(&mut self, rank_one_based: usize) -> Result<bool, String> {
        if self.quarantined {
            return Err("matched fixed-bet monitor already quarantined".into());
        }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err("invalid rank for matched fixed-bet monitor".into());
        }
        let next_seen = self
            .observations_seen
            .checked_add(1)
            .ok_or_else(|| "matched fixed-bet observation counter overflow".to_string())?;
        let mut next_log_wealth = self.log_wealth;

        for (index, feature) in [RankFeature::UpperOrder, RankFeature::TailDispersion]
            .into_iter()
            .enumerate()
        {
            let payoff = centered_rank_payoff(
                self.calibration_count,
                &self.rank_counts,
                next_seen,
                rank_one_based,
                feature,
            )
            .map_err(|error| format!("matched fixed-bet payoff: {error:?}"))?;
            let factor = 1.0 + 0.5 * payoff;
            if !factor.is_finite() || factor < 0.5 || !payoff.is_finite() {
                return Err("invalid matched fixed-bet e-process factor".into());
            }
            next_log_wealth[index] += factor.ln();
            if !next_log_wealth[index].is_finite() {
                return Err("matched fixed-bet log wealth overflow".into());
            }
        }

        let max_log = next_log_wealth[0].max(next_log_wealth[1]);
        let log_mixture = max_log
            + (((next_log_wealth[0] - max_log).exp() + (next_log_wealth[1] - max_log).exp()) / 2.0)
                .ln();
        if !log_mixture.is_finite() {
            return Err("matched fixed-bet mixture wealth overflow".into());
        }

        let index = rank_one_based - 1;
        let next_count = self.rank_counts[index]
            .checked_add(1)
            .ok_or_else(|| "matched fixed-bet rank counter overflow".to_string())?;
        self.rank_counts[index] = next_count;
        self.observations_seen = next_seen;
        self.log_wealth = next_log_wealth;

        let threshold = (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        if log_mixture >= threshold {
            self.quarantined = true;
        }
        Ok(self.quarantined)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioSummary {
    pub scenario: Scenario,
    pub fixed_four: MethodSummary,
    pub fixed_two: MethodSummary,
    pub ons: MethodSummary,
    pub ons_wins_vs_fixed_two: usize,
    pub fixed_two_wins: usize,
    pub ties_vs_fixed_two: usize,
    pub ons_wins_vs_fixed_four: usize,
    pub fixed_four_wins: usize,
    pub ties_vs_fixed_four: usize,
}

impl ScenarioSummary {
    pub fn csv_rows(&self) -> [String; 3] {
        [
            format_method_row(self.scenario, &self.fixed_four),
            format_method_row(self.scenario, &self.fixed_two),
            format_method_row(self.scenario, &self.ons),
        ]
    }
}

pub fn run_scenario(
    config: BenchmarkConfig,
    scenario: Scenario,
) -> Result<ScenarioSummary, String> {
    let config = config.validate()?;
    let mut fixed_four = MethodSummary {
        method: "cmaq031_fixed_four_features",
        alarm_delays: Vec::new(),
        replications: config.replications,
        horizon: config.horizon,
    };
    let mut fixed_two = MethodSummary {
        method: "cmaq035_fixed_two_features",
        alarm_delays: Vec::new(),
        replications: config.replications,
        horizon: config.horizon,
    };
    let mut ons = MethodSummary {
        method: "cmaq033_ons_two_features",
        alarm_delays: Vec::new(),
        replications: config.replications,
        horizon: config.horizon,
    };
    let mut ons_wins_vs_fixed_two = 0;
    let mut fixed_two_wins = 0;
    let mut ties_vs_fixed_two = 0;
    let mut ons_wins_vs_fixed_four = 0;
    let mut fixed_four_wins = 0;
    let mut ties_vs_fixed_four = 0;

    for replication in 0..config.replications {
        let replication_seed = config
            .seed
            .wrapping_add(scenario.stream_id())
            .wrapping_add((replication as u64).wrapping_mul(0xD1342543DE82EF95));
        let pair = run_pair(replication_seed, config.horizon, scenario)?;
        let four_delay = pair.fixed_four;
        let two_delay = pair.fixed_two;
        let ons_delay = pair.ons;
        if let Some(delay) = four_delay {
            fixed_four.alarm_delays.push(delay);
        }
        if let Some(delay) = two_delay {
            fixed_two.alarm_delays.push(delay);
        }
        if let Some(delay) = ons_delay {
            ons.alarm_delays.push(delay);
        }

        if scenario != Scenario::Null {
            compare_delays(
                ons_delay,
                two_delay,
                &mut ons_wins_vs_fixed_two,
                &mut fixed_two_wins,
                &mut ties_vs_fixed_two,
            );
            compare_delays(
                ons_delay,
                four_delay,
                &mut ons_wins_vs_fixed_four,
                &mut fixed_four_wins,
                &mut ties_vs_fixed_four,
            );
        }
    }

    Ok(ScenarioSummary {
        scenario,
        fixed_four,
        fixed_two,
        ons,
        ons_wins_vs_fixed_two,
        fixed_two_wins,
        ties_vs_fixed_two,
        ons_wins_vs_fixed_four,
        fixed_four_wins,
        ties_vs_fixed_four,
    })
}

fn compare_delays(
    ons_delay: Option<usize>,
    fixed_delay: Option<usize>,
    ons_wins: &mut usize,
    fixed_wins: &mut usize,
    ties: &mut usize,
) {
    match (ons_delay, fixed_delay) {
        (Some(ons), Some(fixed)) if ons < fixed => *ons_wins += 1,
        (Some(ons), Some(fixed)) if fixed < ons => *fixed_wins += 1,
        (Some(_), None) => *ons_wins += 1,
        (None, Some(_)) => *fixed_wins += 1,
        _ => *ties += 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairDelays {
    fixed_four: Option<usize>,
    fixed_two: Option<usize>,
    ons: Option<usize>,
}

fn run_pair(seed: u64, horizon: usize, scenario: Scenario) -> Result<PairDelays, String> {
    // Deterministic domain separation prevents accidental stream reuse in the
    // simulation. Neither generator is cryptographic.
    let mut score_rng = SplitMix64::new(seed ^ 0x53434F52455F524E);
    let mut tie_rng = SplitMix64::new(seed ^ 0x5449455F4B455953);
    let mut used_tie_keys = HashSet::with_capacity(CALIBRATION_COUNT + horizon);
    let mut scores = Vec::with_capacity(CALIBRATION_COUNT);

    for index in 0..CALIBRATION_COUNT {
        scores.push(CalibrationRankScore {
            independent_cluster_id: format!("calibration-{seed}-{index}"),
            score_units: Scenario::Null.sample(&mut score_rng),
            tie_break_key: unique_tie_key(&mut tie_rng, &mut used_tie_keys),
            block_manifest_sha256: digest_u64(seed ^ (index as u64) ^ 0x424C4F434B),
            aggregate_score_receipt_sha256: digest_u64(seed ^ (index as u64) ^ 0x53434F5245),
            tie_break_receipt_sha256: digest_u64(seed ^ (index as u64) ^ 0x544945524543),
        });
    }
    scores.sort_by_key(|score| (score.score_units, score.tie_break_key));

    let reference = CalibrationRankReference {
        calibration_profile_sha256: PROFILE_SHA256.to_string(),
        calibration_manifest_sha256: digest_u64(seed ^ 0x43414C5F4D414E49),
        scope_sha256: SCOPE_SHA256.to_string(),
        scores: scores.clone(),
    };
    let mut fixed_four_monitor =
        AnytimeCalibrationMonitor::new(reference, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
            .map_err(|error| format!("fixed four-feature monitor init: {error:?}"))?;
    let mut fixed_two_monitor =
        MatchedFixedBetMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)?;
    let mut ons_monitor =
        OnsPredictiveRankMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
            .map_err(|error| format!("ONS monitor init: {error:?}"))?;

    let mut four_delay = None;
    let mut two_delay = None;
    let mut ons_delay = None;
    for step in 1..=horizon {
        let score = scenario.sample(&mut score_rng);
        let tie_key = unique_tie_key(&mut tie_rng, &mut used_tie_keys);
        let rank = rank_against_fixed_reference(&scores, score, tie_key);

        if four_delay.is_none() {
            let observation = DeploymentRankObservation {
                previous_outcome: PreviousOutcome::Reconciled,
                calibration_profile_sha256: PROFILE_SHA256.to_string(),
                scope_sha256: SCOPE_SHA256.to_string(),
                independent_cluster_id: format!("deployment-{seed}-{step}"),
                score_units: score,
                tie_break_key: tie_key,
                block_manifest_sha256: digest_u64(seed ^ step as u64 ^ 0x4445504C4F59),
                candidate_margin_block_sha256: digest_u64(seed ^ step as u64 ^ 0x43414E444944),
                aggregate_score_receipt_sha256: digest_u64(seed ^ step as u64 ^ 0x41474753434F52),
                tie_break_receipt_sha256: digest_u64(seed ^ step as u64 ^ 0x54494542524543),
            };
            match fixed_four_monitor
                .observe(observation)
                .map_err(|error| format!("fixed four-feature monitor observe: {error:?}"))?
            {
                MonitorUpdate::Quarantined { .. } => four_delay = Some(step),
                MonitorUpdate::Accepted { .. } => {}
                MonitorUpdate::BlockedUntilReconciliation | MonitorUpdate::SkippedNotExecuted => {
                    return Err("reconciled synthetic event was unexpectedly skipped".into());
                }
            }
        }

        if two_delay.is_none() && fixed_two_monitor.observe_rank(rank)? {
            two_delay = Some(step);
        }
        if ons_delay.is_none() {
            match ons_monitor
                .observe_rank(rank)
                .map_err(|error| format!("ONS observe: {error:?}"))?
            {
                OnsUpdate::Quarantined { .. } => ons_delay = Some(step),
                OnsUpdate::Accepted { .. } => {}
            }
        }

        if four_delay.is_some() && two_delay.is_some() && ons_delay.is_some() {
            break;
        }
    }
    Ok(PairDelays {
        fixed_four: four_delay,
        fixed_two: two_delay,
        ons: ons_delay,
    })
}

fn rank_against_fixed_reference(
    sorted_reference: &[CalibrationRankScore],
    score_units: u64,
    tie_break_key: u64,
) -> usize {
    sorted_reference.partition_point(|reference| {
        (reference.score_units, reference.tie_break_key) < (score_units, tie_break_key)
    }) + 1
}

fn unique_tie_key(rng: &mut SplitMix64, seen: &mut HashSet<u64>) -> u64 {
    loop {
        let key = rng.next_u64();
        if seen.insert(key) {
            return key;
        }
    }
}

fn digest_u64(value: u64) -> String {
    format!("{value:064x}")
}

fn quantile(values: &[usize], probability: f64) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((probability * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[rank])
}

pub fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    let lower = if successes == 0 {
        0.0
    } else {
        (center - half).max(0.0)
    };
    let upper = if successes == trials {
        1.0
    } else {
        (center + half).min(1.0)
    };
    (lower, upper)
}

fn format_method_row(scenario: Scenario, summary: &MethodSummary) -> String {
    let (low, high) = summary.wilson_interval();
    format!(
        "{},{},{},{},{:.6},{:.6},{:.6},{},{},{}",
        scenario.as_str(),
        summary.method,
        summary.replications,
        summary.horizon,
        summary.detection_rate(),
        low,
        high,
        summary
            .median_delay()
            .map_or_else(String::new, |value| value.to_string()),
        summary
            .p90_delay()
            .map_or_else(String::new, |value| value.to_string()),
        summary
            .p95_delay()
            .map_or_else(String::new, |value| value.to_string()),
    )
}

pub fn run_benchmark(config: BenchmarkConfig) -> Result<Vec<ScenarioSummary>, String> {
    config.validate()?;
    Scenario::ALL
        .into_iter()
        .map(|scenario| run_scenario(config, scenario))
        .collect()
}

/// Positional CLI: cmaq-035-prm-paired-benchmark [replications] [horizon] [seed].
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let replications = args.first().map_or(Ok(10_000), |value| value.parse())?;
    let horizon = args.get(1).map_or(Ok(500), |value| value.parse())?;
    let seed = args
        .get(2)
        .map_or(Ok(20_261_127_u64), |value| value.parse())?;
    let config = BenchmarkConfig {
        replications,
        horizon,
        seed,
    };
    let results = run_benchmark(config).map_err(std::io::Error::other)?;

    println!(
        "scenario,method,replications,horizon,detection_rate,wilson95_low,wilson95_high,median_delay,p90_delay,p95_delay"
    );
    for result in &results {
        for row in result.csv_rows() {
            println!("{row}");
        }
        if result.scenario == Scenario::Null {
            println!("#paired,null,comparison=not_applicable");
        } else {
            println!(
                "#paired,{},ons_vs_fixed_two_wins={},fixed_two_wins={},ties={}",
                result.scenario.as_str(),
                result.ons_wins_vs_fixed_two,
                result.fixed_two_wins,
                result.ties_vs_fixed_two
            );
            println!(
                "#paired,{},ons_vs_fixed_four_wins={},fixed_four_wins={},ties={}",
                result.scenario.as_str(),
                result.ons_wins_vs_fixed_four,
                result.fixed_four_wins,
                result.ties_vs_fixed_four
            );
        }
    }
    eprintln!(
        "SIMULATION ONLY: replications={} horizon={} seed={} calibration_n={} alpha={}/{} rng=SplitMix64(non-cryptographic)",
        config.replications,
        config.horizon,
        config.seed,
        CALIBRATION_COUNT,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_reproducible() {
        let mut first = SplitMix64::new(0x1234);
        let mut second = SplitMix64::new(0x1234);
        for _ in 0..100 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn rank_respects_score_then_independent_tie_key() {
        let scores = vec![
            CalibrationRankScore {
                independent_cluster_id: "a".into(),
                score_units: 10,
                tie_break_key: 3,
                block_manifest_sha256: digest_u64(1),
                aggregate_score_receipt_sha256: digest_u64(2),
                tie_break_receipt_sha256: digest_u64(3),
            },
            CalibrationRankScore {
                independent_cluster_id: "b".into(),
                score_units: 10,
                tie_break_key: 7,
                block_manifest_sha256: digest_u64(4),
                aggregate_score_receipt_sha256: digest_u64(5),
                tie_break_receipt_sha256: digest_u64(6),
            },
            CalibrationRankScore {
                independent_cluster_id: "c".into(),
                score_units: 20,
                tie_break_key: 1,
                block_manifest_sha256: digest_u64(7),
                aggregate_score_receipt_sha256: digest_u64(8),
                tie_break_receipt_sha256: digest_u64(9),
            },
        ];
        assert_eq!(rank_against_fixed_reference(&scores, 10, 1), 1);
        assert_eq!(rank_against_fixed_reference(&scores, 10, 5), 2);
        assert_eq!(rank_against_fixed_reference(&scores, 10, 9), 3);
        assert_eq!(rank_against_fixed_reference(&scores, 20, 1), 3);
        assert_eq!(rank_against_fixed_reference(&scores, 30, 1), 4);
    }

    #[test]
    fn simulation_configuration_rejects_zero_counts() {
        assert!(
            BenchmarkConfig {
                replications: 0,
                horizon: 100,
                seed: 1,
            }
            .validate()
            .is_err()
        );
        assert!(
            BenchmarkConfig {
                replications: 1,
                horizon: 0,
                seed: 1,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn wilson_interval_contains_observed_rate() {
        let (low, high) = wilson_interval(50, 100);
        assert!(low < 0.5 && high > 0.5);
        assert_eq!(wilson_interval(0, 100).0, 0.0);
        assert_eq!(wilson_interval(100, 100).1, 1.0);
    }

    #[test]
    fn paired_simulation_is_seed_reproducible() {
        let config = BenchmarkConfig {
            replications: 20,
            horizon: 100,
            seed: 20261127,
        };
        assert_eq!(
            run_scenario(config, Scenario::DispersionExpansion).unwrap(),
            run_scenario(config, Scenario::DispersionExpansion).unwrap()
        );
    }

    #[test]
    fn null_and_orthogonal_scenarios_are_distinct() {
        let config = BenchmarkConfig {
            replications: 8,
            horizon: 20,
            seed: 42,
        };
        assert_ne!(
            run_scenario(config, Scenario::Null).unwrap(),
            run_scenario(config, Scenario::OrthogonalTrimodal).unwrap()
        );
    }
}
