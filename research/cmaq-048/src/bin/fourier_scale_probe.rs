#![forbid(unsafe_code)]

//! Paired smoke probe: raw-mean approachability vs fixed Fourier mean embedding.
//!
//! Under both null and alternative, X is symmetric around zero so the raw
//! conditional mean remains zero. The scale changes from +/-0.2 to +/-0.8.
//! The registered Fourier bank {1,3} turns that scale change into a kernel
//! feature-mean shift. Both monitors receive the exact same input sequence,
//! alpha and horizon.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::{
    BoxTarget, ConvexTargetApproachabilityMonitor,
};
use cmaq_048_fourier_kernel_approachability::{
    FourierFeatureMap, KernelMeanApproachabilityMonitor,
};

const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TARGET_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const REFERENCE_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const NULL_SCALE: f64 = 0.2;
const SHIFTED_SCALE: f64 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    ConditionalNull,
    ScaleOnlyShift,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::ConditionalNull => "conditional_null",
            Self::ScaleOnlyShift => "scale_only_shift",
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
        return Err(
            "usage: fourier_scale_probe [replications] [horizon] [seed] [shift_after]".into(),
        );
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
    let seed = parse(2, "20261128", "seed")?;
    let shift_after = usize::try_from(parse(3, "500", "shift_after")?)
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

fn raw_monitor() -> Result<ConvexTargetApproachabilityMonitor, String> {
    let target = BoxTarget::new(vec![-0.05], vec![0.05])
        .map_err(|error| format!("raw target invalid: {error:?}"))?;
    ConvexTargetApproachabilityMonitor::new(target, 1.0, ALPHA_NUMERATOR, ALPHA_DENOMINATOR)
        .map_err(|error| format!("raw monitor invalid: {error:?}"))
}

fn kernel_monitor() -> Result<KernelMeanApproachabilityMonitor, String> {
    let scale = 2.0f64.sqrt().recip();
    let means = [(NULL_SCALE).cos() * scale, (3.0 * NULL_SCALE).cos() * scale];
    let tolerance = 0.01;
    let target = BoxTarget::new(
        vec![
            means[0] - tolerance,
            -tolerance,
            means[1] - tolerance,
            -tolerance,
        ],
        vec![
            means[0] + tolerance,
            tolerance,
            means[1] + tolerance,
            tolerance,
        ],
    )
    .map_err(|error| format!("kernel target invalid: {error:?}"))?;
    let map = FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], MAP_SHA.into())
        .map_err(|error| format!("Fourier map invalid: {error:?}"))?;
    KernelMeanApproachabilityMonitor::new(
        map,
        target,
        1.0,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
        TARGET_SHA.into(),
        REFERENCE_SHA.into(),
    )
    .map_err(|error| format!("kernel monitor invalid: {error:?}"))
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
            Scenario::ConditionalNull => 0x0480_0001,
            Scenario::ScaleOnlyShift => 0x0480_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let mut raw = raw_monitor()?;
    let mut kernel = kernel_monitor()?;
    let mut raw_alarm = None;
    let mut kernel_alarm = None;

    for time in 1..=config.horizon {
        let scale = match scenario {
            Scenario::ConditionalNull => NULL_SCALE,
            Scenario::ScaleOnlyShift if time <= config.shift_after => NULL_SCALE,
            Scenario::ScaleOnlyShift => SHIFTED_SCALE,
        };
        let sample = scale * rng.symmetric_sign();

        if raw_alarm.is_none() {
            let update = raw
                .observe(&[sample])
                .map_err(|error| format!("raw monitor update failed: {error:?}"))?;
            if update.quarantined {
                raw_alarm = Some(time);
            }
        }

        if kernel_alarm.is_none() {
            let update = kernel
                .observe(&[sample])
                .map_err(|error| format!("kernel monitor update failed: {error:?}"))?;
            if update.evidence.quarantined {
                kernel_alarm = Some(time);
            }
        }

        if raw_alarm.is_some() && kernel_alarm.is_some() {
            break;
        }
    }

    for (arm, alarm) in [
        ("raw_mean", raw_alarm),
        ("fourier_kernel_mean", kernel_alarm),
    ] {
        writeln!(
            csv,
            "{},{},{},{},{}",
            scenario.name(),
            replication,
            arm,
            alarm.map_or_else(|| "NA".into(), |time| time.to_string()),
            config.shift_after,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok((raw_alarm, kernel_alarm))
}

fn median_delay(values: &[Option<usize>], shift_after: usize) -> Option<usize> {
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
    let mut csv = String::from(
        "scenario,replication,arm,first_alarm_time_one_based,shift_after\n",
    );
    for scenario in [Scenario::ConditionalNull, Scenario::ScaleOnlyShift] {
        let mut raw_alarms = Vec::with_capacity(config.replications);
        let mut kernel_alarms = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            let (raw_alarm, kernel_alarm) =
                run_replication(scenario, replication, config, &mut csv)?;
            raw_alarms.push(raw_alarm);
            kernel_alarms.push(kernel_alarm);
        }

        let raw_false_or_prechange = raw_alarms
            .iter()
            .filter(|alarm| alarm.is_some_and(|time| {
                scenario == Scenario::ConditionalNull || time <= config.shift_after
            }))
            .count();
        let kernel_false_or_prechange = kernel_alarms
            .iter()
            .filter(|alarm| alarm.is_some_and(|time| {
                scenario == Scenario::ConditionalNull || time <= config.shift_after
            }))
            .count();

        if scenario == Scenario::ConditionalNull {
            eprintln!(
                "conditional_null: replications={} raw_false_alarms={} kernel_false_alarms={}",
                config.replications,
                raw_false_or_prechange,
                kernel_false_or_prechange,
            );
        } else {
            let raw_detected = raw_alarms
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time > config.shift_after))
                .count();
            let kernel_detected = kernel_alarms
                .iter()
                .filter(|alarm| alarm.is_some_and(|time| time > config.shift_after))
                .count();
            eprintln!(
                "scale_only_shift: reps={} raw_prechange_false={} kernel_prechange_false={} raw_detected={} kernel_detected={} raw_misses={} kernel_misses={} raw_median_delay={} kernel_median_delay={}",
                config.replications,
                raw_false_or_prechange,
                kernel_false_or_prechange,
                raw_detected,
                kernel_detected,
                config.replications - raw_detected,
                config.replications - kernel_detected,
                median_delay(&raw_alarms, config.shift_after)
                    .map_or_else(|| "NA".into(), |time| time.to_string()),
                median_delay(&kernel_alarms, config.shift_after)
                    .map_or_else(|| "NA".into(), |time| time.to_string()),
            );
        }
    }
    print!("{csv}");
    Ok(())
}
