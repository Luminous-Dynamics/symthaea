#![forbid(unsafe_code)]

//! Paired rank-grid probe: CMAQ-031 fixed-four PRM vs CMAQ-038 Haar spectrum.
//! The Pólya-urn null is exactly the predictive-rank exchangeability law;
//! the tri-modal arm is a deliberate higher-order shape alternative.

use std::env;
use std::fmt::Display;
use std::str::FromStr;

use cmaq_031_anytime_calibration_monitor::{centered_rank_payoff, RankFeature};
use cmaq_038_rank_spectrum_monitor::{RankSpectrumMonitor, SpectrumUpdate};

const CALIBRATION_COUNT: usize = 99;
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const Z_975: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug)]
struct Config { replications: usize, horizon: usize, seed: u64 }

#[derive(Clone, Copy, Debug)]
enum Scenario { PredictiveNull, TrimodalShape }

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::PredictiveNull => "predictive_rank_null",
            Self::TrimodalShape => "trimodal_rank_grid",
        }
    }
    fn id(self) -> u64 {
        match self {
            Self::PredictiveNull => 0x0380_0001,
            Self::TrimodalShape => 0x0380_0002,
        }
    }
}

#[derive(Clone, Debug)]
struct MethodSummary {
    method: &'static str,
    alarm_delays: Vec<usize>,
    replications: usize,
    horizon: usize,
}
impl MethodSummary {
    fn alarm_count(&self) -> usize { self.alarm_delays.len() }
    fn rate(&self) -> f64 { self.alarm_count() as f64 / self.replications as f64 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairCounts { spectrum_only: usize, fixed_only: usize, both: usize, neither: usize }

#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    fn below(&mut self, upper_exclusive: u64) -> u64 {
        assert!(upper_exclusive > 0, "upper bound must be nonzero");
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold { return value % upper_exclusive; }
        }
    }
}

/// CMAQ-031 four-feature fixed-bet monitor, using the actual public
/// centered_rank_payoff helper and the same lambda=1/2 equal mixture.
struct FixedFourMonitor {
    calibration_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    rank_counts: Vec<u64>,
    observations_seen: u64,
    log_wealth: [f64; 4],
    quarantined: bool,
}
impl FixedFourMonitor {
    fn new(calibration_count: usize, alpha_numerator: u32, alpha_denominator: u32) -> Result<Self, String> {
        if calibration_count == 0 || alpha_numerator == 0 || alpha_denominator == 0 || alpha_numerator >= alpha_denominator {
            return Err("invalid fixed-four monitor configuration".into());
        }
        Ok(Self {
            calibration_count, alpha_numerator, alpha_denominator,
            rank_counts: vec![0; calibration_count + 1],
            observations_seen: 0, log_wealth: [0.0; 4], quarantined: false,
        })
    }

    fn observe_rank(&mut self, rank_one_based: usize) -> Result<bool, String> {
        if self.quarantined { return Err("fixed-four monitor already quarantined".into()); }
        if rank_one_based == 0 || rank_one_based > self.calibration_count + 1 {
            return Err("invalid predictive rank".into());
        }
        let next_seen = self.observations_seen.checked_add(1)
            .ok_or_else(|| "fixed-four observation counter overflow".to_string())?;
        let features = [
            RankFeature::UpperOrder, RankFeature::LowerOrder,
            RankFeature::TailDispersion, RankFeature::CenterConcentration,
        ];
        let mut next_wealth = self.log_wealth;
        for (index, feature) in features.into_iter().enumerate() {
            let payoff = centered_rank_payoff(
                self.calibration_count, &self.rank_counts, next_seen, rank_one_based, feature,
            ).map_err(|error| format!("CMAQ-031 payoff error: {error:?}"))?;
            let factor = 1.0 + 0.5 * payoff;
            if !payoff.is_finite() || !factor.is_finite() || factor < 0.5 - 1e-12 {
                return Err("invalid fixed-four betting factor".into());
            }
            next_wealth[index] += factor.ln();
            if !next_wealth[index].is_finite() { return Err("fixed-four log wealth overflow".into()); }
        }
        let log_mixture = log_mean_exp(&next_wealth)?;
        let threshold = (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        let next_count = self.rank_counts[rank_one_based - 1].checked_add(1)
            .ok_or_else(|| "fixed-four rank counter overflow".to_string())?;

        // Commit only after all feature and mixture arithmetic succeeds.
        self.rank_counts[rank_one_based - 1] = next_count;
        self.observations_seen = next_seen;
        self.log_wealth = next_wealth;
        if log_mixture >= threshold { self.quarantined = true; }
        Ok(self.quarantined)
    }
}

fn log_mean_exp(log_values: &[f64]) -> Result<f64, String> {
    if log_values.is_empty() || log_values.iter().any(|value| !value.is_finite()) {
        return Err("invalid log wealth vector".into());
    }
    let maximum = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum: f64 = log_values.iter().map(|value| (value - maximum).exp()).sum();
    let result = maximum + (scaled_sum / log_values.len() as f64).ln();
    if !result.is_finite() { return Err("log mixture overflow".into()); }
    Ok(result)
}

/// Under null, samples the exact Pólya predictive law induced by the fixed
/// calibration set: bucket j has weight 1 + the number of prior online ranks.
fn sample_rank(
    scenario: Scenario,
    calibration_count: usize,
    observations_seen: usize,
    previous_counts: &[u64],
    rng: &mut SplitMix64,
) -> Result<usize, String> {
    match scenario {
        Scenario::PredictiveNull => {
            let total = u64::try_from(calibration_count)
                .map_err(|_| "calibration count conversion overflow".to_string())?
                .checked_add(u64::try_from(observations_seen)
                    .map_err(|_| "observation count conversion overflow".to_string())?)
                .ok_or_else(|| "predictive rank denominator overflow".to_string())?;
            let mut draw = rng.below(total);
            for (index, count) in previous_counts.iter().enumerate() {
                let weight = count.checked_add(1)
                    .ok_or_else(|| "predictive rank bucket overflow".to_string())?;
                if draw < weight { return Ok(index + 1); }
                draw -= weight;
            }
            Err("predictive rank draw escaped its support".into())
        }
        Scenario::TrimodalShape => {
            let draw = rng.below(10_000);
            if draw < 3_125 { Ok(11) }
            else if draw < 6_875 { Ok(51) }
            else { Ok(91) }
        }
    }
}

fn run_scenario(config: Config, scenario: Scenario) -> Result<(MethodSummary, MethodSummary, PairCounts), String> {
    let mut fixed = MethodSummary {
        method: "cmaq031_fixed_four", alarm_delays: Vec::with_capacity(config.replications),
        replications: config.replications, horizon: config.horizon,
    };
    let mut spectrum = MethodSummary {
        method: "cmaq038_haar_spectrum", alarm_delays: Vec::with_capacity(config.replications),
        replications: config.replications, horizon: config.horizon,
    };
    let mut pairs = PairCounts { spectrum_only: 0, fixed_only: 0, both: 0, neither: 0 };

    for replication in 0..config.replications {
        let replication_seed = config.seed
            ^ scenario.id().rotate_left(17)
            ^ (replication as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = SplitMix64::new(replication_seed);
        let mut generator_counts = vec![0u64; CALIBRATION_COUNT + 1];
        let mut fixed_monitor = FixedFourMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)?;
        let mut spectrum_monitor = RankSpectrumMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
            .map_err(|error| format!("CMAQ-038 construction error: {error:?}"))?;
        let mut fixed_delay = None;
        let mut spectrum_delay = None;

        for step in 1..=config.horizon {
            let rank = sample_rank(scenario, CALIBRATION_COUNT, step, &generator_counts, &mut rng)?;
            if fixed_delay.is_none() && fixed_monitor.observe_rank(rank)? { fixed_delay = Some(step); }
            if spectrum_delay.is_none() {
                match spectrum_monitor.observe_rank(rank).map_err(|error| format!("CMAQ-038 update error: {error:?}"))? {
                    SpectrumUpdate::Accepted { .. } => {}
                    SpectrumUpdate::Quarantined { .. } => spectrum_delay = Some(step),
                }
            }
            generator_counts[rank - 1] = generator_counts[rank - 1].checked_add(1)
                .ok_or_else(|| "generator count overflow".to_string())?;
            if fixed_delay.is_some() && spectrum_delay.is_some() { break; }
        }

        if let Some(delay) = fixed_delay { fixed.alarm_delays.push(delay); }
        if let Some(delay) = spectrum_delay { spectrum.alarm_delays.push(delay); }
        match (fixed_delay.is_some(), spectrum_delay.is_some()) {
            (true, true) => pairs.both += 1,
            (true, false) => pairs.fixed_only += 1,
            (false, true) => pairs.spectrum_only += 1,
            (false, false) => pairs.neither += 1,
        }
    }
    Ok((fixed, spectrum, pairs))
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z_975 * Z_975;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radius = Z_975 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

fn quantile(values: &[usize], probability: f64) -> Option<usize> {
    if values.is_empty() { return None; }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((probability * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1).min(sorted.len() - 1);
    sorted.get(index).copied()
}

fn optional_number(value: Option<usize>) -> String {
    value.map_or_else(String::new, |number| number.to_string())
}

fn print_summary(scenario: Scenario, summary: &MethodSummary) {
    let (low, high) = wilson_interval(summary.alarm_count(), summary.replications);
    println!("{},{},{},{},{:.6},{:.6},{:.6},{},{},{}",
        scenario.name(), summary.method, summary.replications, summary.horizon,
        summary.rate(), low, high,
        optional_number(quantile(&summary.alarm_delays, 0.50)),
        optional_number(quantile(&summary.alarm_delays, 0.90)),
        optional_number(quantile(&summary.alarm_delays, 0.95)));
}

fn parse_arg<T>(index: usize, default: T, name: &str) -> Result<T, String>
where T: FromStr, T::Err: Display {
    match env::args().nth(index) {
        Some(value) => value.parse::<T>().map_err(|error| format!("invalid {name}: {error}")),
        None => Ok(default),
    }
}

fn run() -> Result<(), String> {
    let config = Config {
        replications: parse_arg(1, 1_000usize, "replications")?,
        horizon: parse_arg(2, 500usize, "horizon")?,
        seed: parse_arg(3, 20_261_127u64, "seed")?,
    };
    if config.replications == 0 || config.horizon == 0 {
        return Err("replications and horizon must be positive".into());
    }
    println!("scenario,method,replications,horizon,alarm_rate,wilson95_low,wilson95_high,median_delay,p90_delay,p95_delay");
    for scenario in [Scenario::PredictiveNull, Scenario::TrimodalShape] {
        let (fixed, spectrum, pairs) = run_scenario(config, scenario)?;
        print_summary(scenario, &fixed);
        print_summary(scenario, &spectrum);
        eprintln!("paired scenario={} spectrum_only={} fixed_only={} both={} neither={}",
            scenario.name(), pairs.spectrum_only, pairs.fixed_only, pairs.both, pairs.neither);
    }
    eprintln!("config replications={} horizon={} seed={} calibration_n={} alpha={}/{} rng=SplitMix64(simulation-only)",
        config.replications, config.horizon, config.seed, CALIBRATION_COUNT,
        ALPHA_NUMERATOR, ALPHA_DENOMINATOR);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
