#![forbid(unsafe_code)]

//! Paired conditional-kernel smoke probe.
//!
//! The alternative has the same marginal X/Y mixture across contexts, but
//! different conditional pair distributions within each registered context.
//! The null independently draws X and Y from the same symmetric conditional
//! distribution in each context.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_049_contextual_paired_kernel::ContextualPairedKernelMonitor;

const MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CONTEXT_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const TARGET_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const REFERENCE_SHA: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const CONTEXTS: usize = 2;
const INPUT_SCALE: f64 = 0.2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    ConditionalPairNull,
    MarginallyEqualConditionalShift,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::ConditionalPairNull => "conditional_pair_null",
            Self::MarginallyEqualConditionalShift => "marginally_equal_conditional_shift",
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
}

fn parse_config() -> Result<Config, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    if values.len() > 3 {
        return Err("usage: conditional_pair_probe [replications] [even_horizon] [seed]".into());
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
    let horizon = usize::try_from(parse(1, "500", "horizon")?)
        .map_err(|_| "horizon exceeds usize".to_string())?;
    let seed = parse(2, "20261128", "seed")?;
    if replications == 0 || horizon == 0 || horizon % 2 != 0 {
        return Err("replications must be positive and horizon must be positive and even".into());
    }
    Ok(Config {
        replications,
        horizon,
        seed,
    })
}

fn monitor() -> Result<ContextualPairedKernelMonitor, String> {
    let feature_map = FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], MAP_SHA.into())
        .map_err(|error| format!("feature map invalid: {error:?}"))?;
    let target = BoxTarget::new(vec![-0.005; CONTEXTS * 4], vec![0.005; CONTEXTS * 4])
        .map_err(|error| format!("paired target invalid: {error:?}"))?;
    ContextualPairedKernelMonitor::new(
        feature_map,
        CONTEXTS,
        target,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
        CONTEXT_SHA.into(),
        TARGET_SHA.into(),
        REFERENCE_SHA.into(),
    )
    .map_err(|error| format!("paired monitor invalid: {error:?}"))
}

fn pair_values(scenario: Scenario, context: usize, rng: &mut SplitMix64) -> (f64, f64) {
    match scenario {
        Scenario::ConditionalPairNull => (INPUT_SCALE * rng.sign(), INPUT_SCALE * rng.sign()),
        Scenario::MarginallyEqualConditionalShift if context == 0 => (0.2, 0.8),
        Scenario::MarginallyEqualConditionalShift => (0.8, 0.2),
    }
}

fn median(values: &[Option<usize>]) -> Option<usize> {
    let mut detections = values.iter().flatten().copied().collect::<Vec<_>>();
    if detections.is_empty() {
        return None;
    }
    detections.sort_unstable();
    Some(detections[(detections.len() - 1) / 2])
}

fn run_replication(
    scenario: Scenario,
    replication: usize,
    config: Config,
    csv: &mut String,
) -> Result<Option<usize>, String> {
    let seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::ConditionalPairNull => 0x0490_0001,
            Scenario::MarginallyEqualConditionalShift => 0x0490_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let mut monitor = monitor()?;
    let mut alarm = None;

    for sequence in 1..=config.horizon {
        let context = (sequence - 1) % CONTEXTS;
        let sequence_u64 =
            u64::try_from(sequence).map_err(|_| "sequence exceeds u64".to_string())?;
        let (x, y) = pair_values(scenario, context, &mut rng);
        let update = monitor
            .observe_pair(
                sequence_u64,
                &format!("{}-{replication}-{sequence}", scenario.name()),
                context,
                &[x],
                &[y],
            )
            .map_err(|error| format!("paired monitor update failed: {error:?}"))?;
        writeln!(
            csv,
            "{},{},{},{},{:.9},{:.9},{:.9}",
            scenario.name(),
            replication,
            sequence,
            context,
            x,
            y,
            update.evidence.log_e_value,
        )
        .map_err(|error| error.to_string())?;
        if update.evidence.quarantined {
            alarm = Some(sequence);
            break;
        }
    }
    Ok(alarm)
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut csv = String::from("scenario,replication,sequence,context,x,y,log_e_value\n");
    for scenario in [
        Scenario::ConditionalPairNull,
        Scenario::MarginallyEqualConditionalShift,
    ] {
        let mut alarms = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            alarms.push(run_replication(scenario, replication, config, &mut csv)?);
        }
        let alarm_count = alarms.iter().filter(|alarm| alarm.is_some()).count();
        if scenario == Scenario::ConditionalPairNull {
            eprintln!(
                "conditional_pair_null: reps={} false_alarms={}",
                config.replications, alarm_count
            );
        } else {
            eprintln!(
                "marginally_equal_conditional_shift: reps={} detected={} misses={} median_alarm_sequence={}",
                config.replications,
                alarm_count,
                config.replications - alarm_count,
                median(&alarms).map_or_else(|| "NA".into(), |value| value.to_string())
            );
        }
    }
    print!("{csv}");
    Ok(())
}
