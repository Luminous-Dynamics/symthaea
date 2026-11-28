#![forbid(unsafe_code)]

//! Paired probe for reference-target uncertainty and alpha allocation.
//!
//! Each replication has an independent reference split and test stream.
//! The known synthetic null mean is zero. Three target choices are compared:
//! calibrated Hoeffding envelope, oracle point target, and an invalid
//! plug-in point target centered on the sample mean (included only as a
//! negative control).

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::{
    ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};
use cmaq_052_reference_uncertainty_alpha_split::{
    CalibratedReferenceApproachabilityMonitor, ReferenceBlock, RiskFraction,
};

const REFERENCE_COUNT: usize = 256;
const OBSERVATION_BOUND: f64 = 1.0;
const ALPHA_TEST_NUM: u32 = 1;
const ALPHA_TEST_DEN: u32 = 40;
const DELTA_NUM: u32 = 1;
const DELTA_DEN: u32 = 40;
const ALPHA_TOTAL_NUM: u32 = 1;
const ALPHA_TOTAL_DEN: u32 = 20;
const REFERENCE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TARGET_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    IidNull,
    HeteroskedasticMartingaleNull,
    MeanShift,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::IidNull => "iid_null",
            Self::HeteroskedasticMartingaleNull => "heteroskedastic_martingale_null",
            Self::MeanShift => "mean_shift",
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

    fn sign(&mut self) -> f64 {
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
            "usage: reference_envelope_probe [replications] [horizon] [seed] [shift_after]".into(),
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

fn references(rng: &mut SplitMix64, seed: u64) -> (Vec<ReferenceBlock>, f64) {
    let mut previous_sign = rng.sign();
    let mut blocks = Vec::with_capacity(REFERENCE_COUNT);
    for index in 0..REFERENCE_COUNT {
        // The conditional variance depends on the prior sign; the fresh sign
        // remains symmetric, so the conditional mean is exactly zero.
        let amplitude = if previous_sign > 0.0 { 0.6 } else { 0.2 };
        let sign = rng.sign();
        blocks.push(ReferenceBlock {
            block_id: format!("reference-{seed}-{index}"),
            trace_sha256: format!("{:064x}", seed.wrapping_add(index as u64 + 1)),
            value: vec![amplitude * sign],
        });
        previous_sign = sign;
    }
    (blocks, previous_sign)
}

fn calibrated_monitor(
    blocks: Vec<ReferenceBlock>,
) -> Result<CalibratedReferenceApproachabilityMonitor, String> {
    CalibratedReferenceApproachabilityMonitor::new(
        blocks,
        OBSERVATION_BOUND,
        RiskFraction::new(ALPHA_TEST_NUM, ALPHA_TEST_DEN)
            .map_err(|error| format!("alpha_test invalid: {error:?}"))?,
        RiskFraction::new(DELTA_NUM, DELTA_DEN)
            .map_err(|error| format!("delta invalid: {error:?}"))?,
        RiskFraction::new(ALPHA_TOTAL_NUM, ALPHA_TOTAL_DEN)
            .map_err(|error| format!("alpha_total invalid: {error:?}"))?,
        REFERENCE_SHA.into(),
        TARGET_SHA.into(),
    )
    .map_err(|error| format!("calibrated monitor invalid: {error:?}"))
}

fn point_monitor(mean: f64) -> Result<ConvexTargetApproachabilityMonitor, String> {
    let target = BoxTarget::new(vec![mean], vec![mean])
        .map_err(|error| format!("point target invalid: {error:?}"))?;
    ConvexTargetApproachabilityMonitor::new(
        target,
        OBSERVATION_BOUND,
        ALPHA_TEST_NUM,
        ALPHA_TEST_DEN,
    )
    .map_err(|error| format!("point monitor invalid: {error:?}"))
}

fn observe_if_active(
    monitor: &mut ConvexTargetApproachabilityMonitor,
    sample: f64,
    time: usize,
) -> Result<Option<ApproachabilityUpdate>, String> {
    if monitor.is_quarantined() {
        return Ok(None);
    }
    monitor
        .observe(&[sample])
        .map(Some)
        .map_err(|error| format!("point monitor update at {time} failed: {error:?}"))
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

fn run_replication(
    scenario: Scenario,
    replication: usize,
    config: Config,
    csv: &mut String,
) -> Result<(bool, [Option<usize>; 3]), String> {
    let base_seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::IidNull => 0x0520_0001,
            Scenario::MeanShift => 0x0520_0002,
        });
    let mut reference_rng = SplitMix64::new(base_seed ^ 0xA11C_E001);
    let mut test_rng = SplitMix64::new(base_seed ^ 0x7E57_0002);
    let (reference_blocks, mut previous_sign) = references(&mut reference_rng, base_seed);
    let mut calibrated = calibrated_monitor(reference_blocks.clone())?;

    let reference_mean = reference_blocks
        .iter()
        .map(|block| block.value[0])
        .sum::<f64>()
        / REFERENCE_COUNT as f64;
    let mut naive = point_monitor(reference_mean)?;
    let mut oracle = point_monitor(0.0)?;
    let coverage = calibrated.target().lower()[0] <= 0.0 && calibrated.target().upper()[0] >= 0.0;

    let mut calibrated_alarm = None;
    let mut naive_alarm = None;
    let mut oracle_alarm = None;

    for time in 1..=config.horizon {
        let sample = match scenario {
            Scenario::IidNull => 0.2 * test_rng.sign(),
            Scenario::HeteroskedasticMartingaleNull => {
                let amplitude = if previous_sign > 0.0 { 0.6 } else { 0.2 };
                amplitude * test_rng.sign()
            }
            Scenario::MeanShift if time <= config.shift_after => {
                let amplitude = if previous_sign > 0.0 { 0.6 } else { 0.2 };
                amplitude * test_rng.sign()
            }
            Scenario::MeanShift => {
                let amplitude = if previous_sign > 0.0 { 0.6 } else { 0.2 };
                0.3 + amplitude * test_rng.sign()
            }
        };
        previous_sign = if sample >= 0.0 { 1.0 } else { -1.0 };

        if calibrated_alarm.is_none() && !calibrated.is_quarantined() {
            let update = calibrated
                .observe(&format!("test-{replication}-{time}"), &[sample])
                .map_err(|error| format!("calibrated update failed: {error:?}"))?;
            if update.quarantined {
                calibrated_alarm = Some(time);
            }
        }
        if naive_alarm.is_none() {
            if let Some(update) = observe_if_active(&mut naive, sample, time)? {
                if update.quarantined {
                    naive_alarm = Some(time);
                }
            }
        }
        if oracle_alarm.is_none() {
            if let Some(update) = observe_if_active(&mut oracle, sample, time)? {
                if update.quarantined {
                    oracle_alarm = Some(time);
                }
            }
        }
    }

    for (arm, alarm) in [
        ("calibrated_hoeffding_target", calibrated_alarm),
        ("naive_point_plugin_invalid_control", naive_alarm),
        ("oracle_known_mean_target", oracle_alarm),
    ] {
        writeln!(
            csv,
            "{},{},{},{},{}",
            scenario.name(),
            replication,
            coverage,
            arm,
            alarm.map_or_else(|| "NA".into(), |time| time.to_string())
        )
        .map_err(|error| error.to_string())?;
    }
    Ok((coverage, [calibrated_alarm, naive_alarm, oracle_alarm]))
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut csv = String::from(
        "scenario,replication,reference_target_covers_true_mean,arm,first_alarm_time_one_based\n",
    );
    for scenario in [
        Scenario::IidNull,
        Scenario::HeteroskedasticMartingaleNull,
        Scenario::MeanShift,
    ] {
        let mut coverage_count = 0;
        let mut alarms: [Vec<Option<usize>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for replication in 0..config.replications {
            let (coverage, replication_alarms) =
                run_replication(scenario, replication, config, &mut csv)?;
            if coverage {
                coverage_count += 1;
            }
            for (index, alarm) in replication_alarms.into_iter().enumerate() {
                alarms[index].push(alarm);
            }
        }

        let counts = alarms
            .iter()
            .map(|arm| arm.iter().filter(|alarm| alarm.is_some()).count())
            .collect::<Vec<_>>();
        if scenario == Scenario::IidNull || scenario == Scenario::HeteroskedasticMartingaleNull {
            eprintln!(
                "{}: reps={} reference_coverage={}/{} calibrated_false_alarms={} naive_invalid_false_alarms={} oracle_false_alarms={}",
                scenario.name(),
                config.replications,
                coverage_count,
                config.replications,
                counts[0],
                counts[1],
                counts[2],
            );
        } else {
            let prechange = alarms
                .iter()
                .map(|arm| {
                    arm.iter()
                        .filter(|alarm| alarm.is_some_and(|time| time <= config.shift_after))
                        .count()
                })
                .collect::<Vec<_>>();
            let detected = alarms
                .iter()
                .map(|arm| {
                    arm.iter()
                        .filter(|alarm| alarm.is_some_and(|time| time > config.shift_after))
                        .count()
                })
                .collect::<Vec<_>>();
            eprintln!(
                "mean_shift: reps={} calibrated_prechange_false={} naive_prechange_false={} oracle_prechange_false={} calibrated_detected={} calibrated_misses={} calibrated_median_delay={} naive_detected={} oracle_detected={}",
                config.replications,
                prechange[0],
                prechange[1],
                prechange[2],
                detected[0],
                config.replications - detected[0],
                median_delay(&alarms[0], config.shift_after)
                    .map_or_else(|| "NA".into(), |time| time.to_string()),
                detected[1],
                detected[2],
            );
        }
    }
    print!("{csv}");
    Ok(())
}
