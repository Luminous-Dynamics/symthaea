#![forbid(unsafe_code)]

//! Paired probe: adaptive OCO witness versus a fixed compass-direction e-mixture.
//!
//! The conditional-mean null has zero mean, inside the frozen box. After a
//! pre-change prefix, the alternative shifts the mean in one of eight
//! orientations unknown to either detector. Both methods process identical
//! bounded residual vectors with the same alpha and residual bound.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::{
    BoxTarget, ConvexTargetApproachabilityMonitor,
};

const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const OBSERVATION_NORM_BOUND: f64 = 1.0;
const NOISE_MAGNITUDE: f64 = 0.2;
const SHIFT_MAGNITUDE: f64 = 0.35;
const DIAGONAL: f64 = std::f64::consts::FRAC_1_SQRT_2;
const DIRECTIONS: [[f64; 2]; 8] = [
    [1.0, 0.0],
    [DIAGONAL, DIAGONAL],
    [0.0, 1.0],
    [-DIAGONAL, DIAGONAL],
    [-1.0, 0.0],
    [-DIAGONAL, -DIAGONAL],
    [0.0, -1.0],
    [DIAGONAL, -DIAGONAL],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    ConditionalMeanNull,
    UnknownDirectionShift,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::ConditionalMeanNull => "conditional_mean_null",
            Self::UnknownDirectionShift => "unknown_direction_mean_shift",
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

    fn symmetric_sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 }
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
        return Err("usage: approachability_probe [replications] [horizon] [seed] [shift_after]".into());
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

fn target() -> Result<BoxTarget, String> {
    BoxTarget::new(vec![-0.05, -0.05], vec![0.05, 0.05])
        .map_err(|error| format!("box target invalid: {error:?}"))
}

fn dot(left: &[f64; 2], right: &[f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

fn log_mean_exp(log_values: &[f64]) -> Result<f64, String> {
    let maximum = log_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err("fixed-direction log wealth is non-finite".into());
    }
    let sum = log_values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f64>();
    let result = maximum + sum.ln() - (log_values.len() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err("fixed-direction mixture wealth is non-finite".into())
    }
}

fn update_fixed_portfolio(
    log_wealths: &mut [f64],
    target: &BoxTarget,
    residual_bound: f64,
    observation: &[f64; 2],
) -> Result<bool, String> {
    for (log_wealth, direction) in log_wealths.iter_mut().zip(DIRECTIONS) {
        let support = target
            .support(&direction)
            .map_err(|error| format!("support calculation failed: {error:?}"))?;
        let residual = dot(&direction, observation) - support;
        let normalized = residual / residual_bound;
        if !normalized.is_finite() || !(-1.0 - 1e-10..=1.0 + 1e-10).contains(&normalized) {
            return Err("fixed directional residual exceeded its bound".into());
        }
        let factor = 1.0 + 0.5 * normalized;
        if !factor.is_finite() || factor < 0.5 - 1e-12 {
            return Err("fixed directional factor violated nonnegativity".into());
        }
        *log_wealth += factor.ln();
        if !log_wealth.is_finite() {
            return Err("fixed directional wealth overflow".into());
        }
    }
    let log_mixture = log_mean_exp(log_wealths)?;
    let log_threshold =
        (f64::from(ALPHA_DENOMINATOR) / f64::from(ALPHA_NUMERATOR)).ln();
    Ok(log_mixture >= log_threshold)
}

fn draw_observation(
    rng: &mut SplitMix64,
    scenario: Scenario,
    time_one_based: usize,
    shift_after: usize,
    direction: [f64; 2],
) -> [f64; 2] {
    let mean = match scenario {
        Scenario::ConditionalMeanNull => [0.0, 0.0],
        Scenario::UnknownDirectionShift if time_one_based <= shift_after => [0.0, 0.0],
        Scenario::UnknownDirectionShift => [
            SHIFT_MAGNITUDE * direction[0],
            SHIFT_MAGNITUDE * direction[1],
        ],
    };
    [
        mean[0] + NOISE_MAGNITUDE * rng.symmetric_sign(),
        mean[1] + NOISE_MAGNITUDE * rng.symmetric_sign(),
    ]
}

fn run_replication(
    scenario: Scenario,
    replication: usize,
    config: Config,
    csv: &mut String,
) -> Result<(Option<usize>, Option<usize>), String> {
    let orientation = replication % DIRECTIONS.len();
    let direction = DIRECTIONS[orientation];
    let seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::ConditionalMeanNull => 0x0460_0001,
            Scenario::UnknownDirectionShift => 0x0460_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let target = target()?;
    let mut adaptive = ConvexTargetApproachabilityMonitor::new(
        target.clone(),
        OBSERVATION_NORM_BOUND,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
    )
    .map_err(|error| format!("adaptive monitor construction failed: {error:?}"))?;
    let residual_bound = OBSERVATION_NORM_BOUND + target.maximum_corner_norm();
    let mut fixed_logs = vec![0.0f64; DIRECTIONS.len()];
    let mut adaptive_alarm = None;
    let mut fixed_alarm = None;

    for time in 1..=config.horizon {
        let observation = draw_observation(
            &mut rng,
            scenario,
            time,
            config.shift_after,
            direction,
        );
        if adaptive_alarm.is_none() {
            let update = adaptive
                .observe(&observation)
                .map_err(|error| format!("adaptive monitor update failed: {error:?}"))?;
            if update.quarantined {
                adaptive_alarm = Some(time);
            }
        }
        if fixed_alarm.is_none()
            && update_fixed_portfolio(
                &mut fixed_logs,
                &target,
                residual_bound,
                &observation,
            )?
        {
            fixed_alarm = Some(time);
        }
        if adaptive_alarm.is_some() && fixed_alarm.is_some() {
            break;
        }
    }

    for (arm, alarm) in [
        ("adaptive_oco_approachability", adaptive_alarm),
        ("fixed_eight_direction_mixture", fixed_alarm),
    ] {
        writeln!(
            csv,
            "{},{},{},{},{}",
            scenario.name(),
            replication,
            orientation,
            arm,
            alarm.map_or_else(|| "NA".into(), |time| time.to_string())
        )
        .map_err(|error| error.to_string())?;
    }
    Ok((adaptive_alarm, fixed_alarm))
}

fn median(values: &[Option<usize>], shift_after: usize) -> Option<usize> {
    let mut delays = values
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
    let mut csv = String::from("scenario,replication,orientation,arm,first_alarm_time_one_based\n");

    for scenario in [Scenario::ConditionalMeanNull, Scenario::UnknownDirectionShift] {
        let mut adaptive = Vec::with_capacity(config.replications);
        let mut fixed = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            let (adaptive_alarm, fixed_alarm) =
                run_replication(scenario, replication, config, &mut csv)?;
            adaptive.push(adaptive_alarm);
            fixed.push(fixed_alarm);
        }

        if scenario == Scenario::ConditionalMeanNull {
            eprintln!(
                "conditional_mean_null: replications={} adaptive_false_alarms={} fixed_mixture_false_alarms={}",
                config.replications,
                adaptive.iter().filter(|alarm| alarm.is_some()).count(),
                fixed.iter().filter(|alarm| alarm.is_some()).count(),
            );
        } else {
            let adaptive_prechange = adaptive
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time <= config.shift_after))
                .count();
            let fixed_prechange = fixed
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time <= config.shift_after))
                .count();
            let adaptive_postchange = adaptive
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time > config.shift_after))
                .count();
            let fixed_postchange = fixed
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time > config.shift_after))
                .count();
            eprintln!(
                "unknown_direction_mean_shift: reps={} adaptive_prechange_false={} fixed_prechange_false={} adaptive_detected={} fixed_detected={} adaptive_misses={} fixed_misses={} adaptive_median_delay={} fixed_median_delay={}",
                config.replications,
                adaptive_prechange,
                fixed_prechange,
                adaptive_postchange,
                fixed_postchange,
                config.replications - adaptive_postchange,
                config.replications - fixed_postchange,
                median(&adaptive, config.shift_after)
                    .map_or_else(|| "NA".into(), |time| time.to_string()),
                median(&fixed, config.shift_after)
                    .map_or_else(|| "NA".into(), |time| time.to_string()),
            );
        }
    }

    print!("{csv}");
    Ok(())
}
