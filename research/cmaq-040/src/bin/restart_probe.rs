#![forbid(unsafe_code)]

//! Paired smoke probe: fixed CMAQ-038 spectrum versus CMAQ-040 restarted spectrum.
//! Null ranks are generated from the exact Pólya-urn predictive law. The delayed
//! arm changes to a constructed three-mode rank-grid alternative after a frozen
//! shift time. The simulation RNG is deterministic and non-cryptographic.

use std::env;
use std::fmt::Write as _;

use cmaq_038_rank_spectrum_monitor::{RankSpectrumMonitor, SpectrumUpdate};
use cmaq_040_restarted_rank_spectrum::{RestartUpdate, RestartedRankSpectrumMonitor};

const CALIBRATION_COUNT: usize = 99;
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const SHAPE_RANKS: [usize; 3] = [11, 51, 91];
const SHAPE_WEIGHTS: [u64; 3] = [5, 6, 5];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    PredictiveNull,
    DelayedTrimodal,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::PredictiveNull => "predictive_rank_null",
            Self::DelayedTrimodal => "delayed_trimodal_rank_grid",
        }
    }
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

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
            if value >= threshold {
                return value % upper_exclusive;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    replications: usize,
    horizon: usize,
    seed: u64,
    shift_after: usize,
}

fn parse_config() -> Result<Config, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    if values.len() > 4 {
        return Err("usage: restart_probe [replications] [horizon] [seed] [shift_after]".into());
    }
    let parse = |index: usize, default: &str, name: &str| -> Result<u64, String> {
        values
            .get(index)
            .map(String::as_str)
            .unwrap_or(default)
            .parse::<u64>()
            .map_err(|_| format!("invalid {name}"))
    };
    let replications = usize::try_from(parse(0, "64", "replications")?)
        .map_err(|_| "replications exceed usize".to_string())?;
    let horizon = usize::try_from(parse(1, "1000", "horizon")?)
        .map_err(|_| "horizon exceeds usize".to_string())?;
    let seed = parse(2, "20261127", "seed")?;
    let shift_after = usize::try_from(parse(3, "400", "shift_after")?)
        .map_err(|_| "shift_after exceeds usize".to_string())?;
    if replications == 0 || horizon == 0 || shift_after >= horizon {
        return Err("replications and horizon must be positive; shift_after must be below horizon".into());
    }
    Ok(Config {
        replications,
        horizon,
        seed,
        shift_after,
    })
}

fn draw_predictive_rank(rng: &mut SplitMix64, counts: &[u64]) -> usize {
    let total: u64 = counts.iter().map(|count| *count + 1).sum();
    let mut draw = rng.below(total);
    for (index, count) in counts.iter().enumerate() {
        let bucket_mass = count + 1;
        if draw < bucket_mass {
            return index + 1;
        }
        draw -= bucket_mass;
    }
    unreachable!("draw is bounded by the sum of the predictive rank masses")
}

fn draw_trimodal_rank(rng: &mut SplitMix64) -> usize {
    let draw = rng.below(SHAPE_WEIGHTS.iter().sum());
    let mut cumulative = 0;
    for (rank, weight) in SHAPE_RANKS.into_iter().zip(SHAPE_WEIGHTS) {
        cumulative += weight;
        if draw < cumulative {
            return rank;
        }
    }
    unreachable!("trimodal draw is bounded by total mass")
}

fn observe_fixed(
    monitor: &mut RankSpectrumMonitor,
    rank: usize,
) -> Result<Option<usize>, String> {
    match monitor
        .observe_rank(rank)
        .map_err(|error| format!("CMAQ-038 update failed: {error:?}"))?
    {
        SpectrumUpdate::Accepted { .. } => Ok(None),
        SpectrumUpdate::Quarantined { .. } => Ok(Some(
            usize::try_from(monitor.observations_seen())
                .map_err(|_| "fixed monitor time exceeds usize".to_string())?,
        )),
    }
}

fn observe_restarted(
    monitor: &mut RestartedRankSpectrumMonitor,
    rank: usize,
) -> Result<Option<usize>, String> {
    match monitor
        .observe_rank(rank)
        .map_err(|error| format!("CMAQ-040 update failed: {error:?}"))?
    {
        RestartUpdate::Accepted { .. } => Ok(None),
        RestartUpdate::Quarantined { .. } => Ok(Some(
            usize::try_from(monitor.observations_seen())
                .map_err(|_| "restarted monitor time exceeds usize".to_string())?,
        )),
    }
}

fn run_replication(
    scenario: Scenario,
    replication: usize,
    config: Config,
    output: &mut String,
) -> Result<(Option<usize>, Option<usize>), String> {
    let scenario_seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::PredictiveNull => 0x0400_0001,
            Scenario::DelayedTrimodal => 0x0400_0002,
        });
    let mut rng = SplitMix64::new(scenario_seed);
    let mut urn_counts = vec![0u64; CALIBRATION_COUNT + 1];
    let mut fixed = RankSpectrumMonitor::new(
        CALIBRATION_COUNT,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
    )
    .map_err(|error| format!("CMAQ-038 construction failed: {error:?}"))?;
    let mut restarted = RestartedRankSpectrumMonitor::new(
        CALIBRATION_COUNT,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
    )
    .map_err(|error| format!("CMAQ-040 construction failed: {error:?}"))?;

    let mut fixed_alarm = None;
    let mut restarted_alarm = None;

    for time in 1..=config.horizon {
        let rank = match scenario {
            Scenario::PredictiveNull => draw_predictive_rank(&mut rng, &urn_counts),
            Scenario::DelayedTrimodal if time <= config.shift_after => {
                draw_predictive_rank(&mut rng, &urn_counts)
            }
            Scenario::DelayedTrimodal => draw_trimodal_rank(&mut rng),
        };
        let count = &mut urn_counts[rank - 1];
        *count = (*count)
            .checked_add(1)
            .ok_or_else(|| "probe urn counter overflow".to_string())?;

        if fixed_alarm.is_none() {
            fixed_alarm = observe_fixed(&mut fixed, rank)?;
        }
        if restarted_alarm.is_none() {
            restarted_alarm = observe_restarted(&mut restarted, rank)?;
        }
        if fixed_alarm.is_some() && restarted_alarm.is_some() {
            break;
        }
    }

    let fixed_text = fixed_alarm.map_or_else(|| "NA".into(), |time| time.to_string());
    let restarted_text = restarted_alarm.map_or_else(|| "NA".into(), |time| time.to_string());
    writeln!(
        output,
        "{},{},cmaq038_fixed_spectrum,{}",
        scenario.name(),
        replication,
        fixed_text
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        output,
        "{},{},cmaq040_restarted_spectrum,{}",
        scenario.name(),
        replication,
        restarted_text
    )
    .map_err(|error| error.to_string())?;

    Ok((fixed_alarm, restarted_alarm))
}

fn summarize(
    label: &str,
    fixed: &[Option<usize>],
    restarted: &[Option<usize>],
    shift_after: Option<usize>,
) {
    let fixed_detected = fixed.iter().filter(|time| time.is_some()).count();
    let restarted_detected = restarted.iter().filter(|time| time.is_some()).count();
    let fixed_delay = median_delay(fixed, shift_after);
    let restarted_delay = median_delay(restarted, shift_after);
    eprintln!(
        "{label}: replications={} fixed_alarms={} restarted_alarms={} fixed_median_delay={} restarted_median_delay={}",
        fixed.len(),
        fixed_detected,
        restarted_detected,
        fixed_delay.map_or_else(|| "NA".into(), |value| value.to_string()),
        restarted_delay.map_or_else(|| "NA".into(), |value| value.to_string())
    );
}

fn median_delay(alarms: &[Option<usize>], shift_after: Option<usize>) -> Option<usize> {
    let mut delays = alarms
        .iter()
        .filter_map(|time| {
            time.as_ref().and_then(|alarm| {
                let origin = shift_after.unwrap_or(0);
                (*alarm > origin).then_some(*alarm - origin)
            })
        })
        .collect::<Vec<_>>();
    if delays.is_empty() {
        return None;
    }
    delays.sort_unstable();
    Some(delays[(delays.len() - 1) / 2])
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut output = String::from("scenario,replication,method,alarm_time_one_based\n");

    for scenario in [Scenario::PredictiveNull, Scenario::DelayedTrimodal] {
        let mut fixed = Vec::with_capacity(config.replications);
        let mut restarted = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            let (fixed_alarm, restarted_alarm) =
                run_replication(scenario, replication, config, &mut output)?;
            fixed.push(fixed_alarm);
            restarted.push(restarted_alarm);
        }
        summarize(
            scenario.name(),
            &fixed,
            &restarted,
            match scenario {
                Scenario::PredictiveNull => None,
                Scenario::DelayedTrimodal => Some(config.shift_after),
            },
        );
    }

    print!("{output}");
    Ok(())
}
