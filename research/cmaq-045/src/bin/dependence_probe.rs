#![forbid(unsafe_code)]

//! Paired probe for marginally stationary temporal dependence shifts.
//!
//! The null arm uses the exact Pólya-urn predictive rank process. The
//! alternative has a uniform stationary marginal over rank buckets but a
//! sticky Markov transition: it repeats the previous rank with probability
//! 0.8 and otherwise redraws uniformly. This is a deliberately simple
//! dependence-shift diagnostic, not a SOTA or deployment benchmark.

use std::env;
use std::fmt::Write as _;

use cmaq_038_rank_spectrum_monitor::{RankSpectrumMonitor, SpectrumUpdate};
use cmaq_045_conditional_rank_spectrum::{ConditionalRankSpectrumMonitor, ConditionalUpdate};

const CALIBRATION_COUNT: usize = 99;
const RANK_BUCKETS: usize = CALIBRATION_COUNT + 1;
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    PredictiveNull,
    StationarySticky,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::PredictiveNull => "predictive_rank_null",
            Self::StationarySticky => "stationary_uniform_sticky_dependence",
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
        return Err("usage: dependence_probe [replications] [horizon] [seed] [shift_after]".into());
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
    let horizon = usize::try_from(parse(1, "800", "horizon")?)
        .map_err(|_| "horizon exceeds usize".to_string())?;
    let seed = parse(2, "20261127", "seed")?;
    let shift_after = usize::try_from(parse(3, "300", "shift_after")?)
        .map_err(|_| "shift_after exceeds usize".to_string())?;
    if replications == 0 || horizon == 0 || shift_after >= horizon {
        return Err(
            "replications and horizon must be positive; shift_after must be below horizon".into(),
        );
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
        let mass = *count + 1;
        if draw < mass {
            return index + 1;
        }
        draw -= mass;
    }
    unreachable!("draw is bounded by total predictive rank mass")
}

fn draw_stationary_sticky_rank(rng: &mut SplitMix64, previous_rank: usize) -> usize {
    // 80% retain the current rank; 20% refresh from the uniform stationary law.
    if rng.below(5) != 0 {
        previous_rank
    } else {
        usize::try_from(rng.below(RANK_BUCKETS as u64) + 1).expect("rank bucket index fits usize")
    }
}

fn observe_fixed(monitor: &mut RankSpectrumMonitor, rank: usize) -> Result<Option<usize>, String> {
    match monitor
        .observe_rank(rank)
        .map_err(|error| format!("CMAQ-038 update failed: {error:?}"))?
    {
        SpectrumUpdate::Accepted { .. } => Ok(None),
        SpectrumUpdate::Quarantined { .. } => Ok(Some(
            usize::try_from(monitor.observations_seen())
                .map_err(|_| "CMAQ-038 time exceeds usize".to_string())?,
        )),
    }
}

fn observe_conditional(
    monitor: &mut ConditionalRankSpectrumMonitor,
    rank: usize,
) -> Result<Option<usize>, String> {
    match monitor
        .observe_rank(rank)
        .map_err(|error| format!("CMAQ-045 update failed: {error:?}"))?
    {
        ConditionalUpdate::Accepted { .. } => Ok(None),
        ConditionalUpdate::Quarantined { .. } => Ok(Some(
            usize::try_from(monitor.observations_seen())
                .map_err(|_| "CMAQ-045 time exceeds usize".to_string())?,
        )),
    }
}

fn run_replication(
    scenario: Scenario,
    replication: usize,
    config: Config,
    csv: &mut String,
) -> Result<(Option<usize>, Option<usize>), String> {
    let seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::PredictiveNull => 0x0450_0001,
            Scenario::StationarySticky => 0x0450_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let mut predictive_counts = vec![0u64; RANK_BUCKETS];
    let mut fixed = RankSpectrumMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
        .map_err(|error| format!("CMAQ-038 construction failed: {error:?}"))?;
    let mut conditional =
        ConditionalRankSpectrumMonitor::new(CALIBRATION_COUNT, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
            .map_err(|error| format!("CMAQ-045 construction failed: {error:?}"))?;

    let mut fixed_alarm = None;
    let mut conditional_alarm = None;
    let mut previous_rank = None;

    for time in 1..=config.horizon {
        let rank = match scenario {
            Scenario::PredictiveNull => draw_predictive_rank(&mut rng, &predictive_counts),
            Scenario::StationarySticky if time <= config.shift_after => {
                draw_predictive_rank(&mut rng, &predictive_counts)
            }
            Scenario::StationarySticky if time == config.shift_after + 1 => {
                // Start the new Markov regime from its exact uniform stationary
                // law rather than inheriting the final nonuniform Pólya-null rank.
                usize::try_from(rng.below(RANK_BUCKETS as u64) + 1)
                    .expect("rank bucket index fits usize")
            }
            Scenario::StationarySticky => {
                let previous = previous_rank.expect("post-change chain has an initialized state");
                draw_stationary_sticky_rank(&mut rng, previous)
            }
        };

        if fixed_alarm.is_none() {
            fixed_alarm = observe_fixed(&mut fixed, rank)?;
        }
        if conditional_alarm.is_none() {
            conditional_alarm = observe_conditional(&mut conditional, rank)?;
        }

        predictive_counts[rank - 1] = predictive_counts[rank - 1]
            .checked_add(1)
            .ok_or_else(|| "probe predictive count overflow".to_string())?;
        previous_rank = Some(rank);

        if fixed_alarm.is_some() && conditional_alarm.is_some() {
            break;
        }
    }

    for (method, alarm) in [
        ("cmaq038_histogram_spectrum", fixed_alarm),
        ("cmaq045_conditional_context_spectrum", conditional_alarm),
    ] {
        writeln!(
            csv,
            "{},{},{},{}",
            scenario.name(),
            replication,
            method,
            alarm.map_or_else(|| "NA".into(), |value| value.to_string())
        )
        .map_err(|error| error.to_string())?;
    }

    Ok((fixed_alarm, conditional_alarm))
}

fn median_delay(alarms: &[Option<usize>], shift_after: usize) -> Option<usize> {
    let mut delays = alarms
        .iter()
        .filter_map(|time| {
            time.as_ref()
                .and_then(|alarm| (*alarm > shift_after).then_some(*alarm - shift_after))
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
    let mut csv = String::from("scenario,replication,method,alarm_time_one_based\n");

    for scenario in [Scenario::PredictiveNull, Scenario::StationarySticky] {
        let mut fixed = Vec::with_capacity(config.replications);
        let mut conditional = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            let (fixed_alarm, conditional_alarm) =
                run_replication(scenario, replication, config, &mut csv)?;
            fixed.push(fixed_alarm);
            conditional.push(conditional_alarm);
        }

        let fixed_prechange_alarms = fixed
            .iter()
            .filter(|time| time.is_some_and(|alarm| alarm <= config.shift_after))
            .count();
        let conditional_prechange_alarms = conditional
            .iter()
            .filter(|time| time.is_some_and(|alarm| alarm <= config.shift_after))
            .count();
        let fixed_postchange_detections = fixed
            .iter()
            .filter(|time| time.is_some_and(|alarm| alarm > config.shift_after))
            .count();
        let conditional_postchange_detections = conditional
            .iter()
            .filter(|time| time.is_some_and(|alarm| alarm > config.shift_after))
            .count();
        let (fixed_median, conditional_median) = match scenario {
            Scenario::PredictiveNull => ("NA".to_string(), "NA".to_string()),
            Scenario::StationarySticky => (
                median_delay(&fixed, config.shift_after)
                    .map_or_else(|| "NA".into(), |value| value.to_string()),
                median_delay(&conditional, config.shift_after)
                    .map_or_else(|| "NA".into(), |value| value.to_string()),
            ),
        };
        eprintln!(
            "{}: replications={} cmaq038_total_alarms={} cmaq045_total_alarms={} cmaq038_prechange_alarms={} cmaq045_prechange_alarms={} cmaq038_postchange_detections={} cmaq045_postchange_detections={} cmaq038_misses={} cmaq045_misses={} cmaq038_median_delay_conditional={} cmaq045_median_delay_conditional={}",
            scenario.name(),
            config.replications,
            fixed.iter().filter(|time| time.is_some()).count(),
            conditional.iter().filter(|time| time.is_some()).count(),
            fixed_prechange_alarms,
            conditional_prechange_alarms,
            fixed_postchange_detections,
            conditional_postchange_detections,
            config.replications - fixed_postchange_detections,
            config.replications - conditional_postchange_detections,
            fixed_median,
            conditional_median,
        );
    }

    print!("{csv}");
    Ok(())
}
