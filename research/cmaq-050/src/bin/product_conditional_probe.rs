#![forbid(unsafe_code)]

//! Paired probe for continuous conditional effects versus a pooled discrepancy.
//!
//! Z is drawn independently at each time. Under the alternative, X and Y have
//! identical marginal mixtures but their pairing depends on Z. The pooled
//! CMAQ-049 one-context arm cannot use that association; CMAQ-050 multiplies
//! the response difference by covariate Fourier features.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_049_contextual_paired_kernel::ContextualPairedKernelMonitor;
use cmaq_050_product_fourier_conditional_pairs::ProductFourierConditionalMonitor;

const MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const COVARIATE_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const TARGET_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const RESPONSE_REFERENCE_SHA: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const COVARIATE_REFERENCE_SHA: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const POOLED_TARGET_SHA: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    ConditionalPairNull,
    EqualMarginalConditionalShift,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::ConditionalPairNull => "conditional_pair_null",
            Self::EqualMarginalConditionalShift => "equal_marginal_conditional_shift",
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
        return Err("usage: product_conditional_probe [replications] [horizon] [seed]".into());
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
    if replications == 0 || horizon == 0 {
        return Err("replications and horizon must be positive".into());
    }
    Ok(Config { replications, horizon, seed })
}

fn response_map() -> Result<FourierFeatureMap, String> {
    FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], MAP_SHA.into())
        .map_err(|error| format!("response map invalid: {error:?}"))
}

fn covariate_map() -> Result<FourierFeatureMap, String> {
    FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], COVARIATE_SHA.into())
        .map_err(|error| format!("covariate map invalid: {error:?}"))
}

fn product_monitor() -> Result<ProductFourierConditionalMonitor, String> {
    let target = BoxTarget::new(vec![-0.005; 16], vec![0.005; 16])
        .map_err(|error| format!("product target invalid: {error:?}"))?;
    ProductFourierConditionalMonitor::new(
        response_map()?,
        covariate_map()?,
        target,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
        TARGET_SHA.into(),
        RESPONSE_REFERENCE_SHA.into(),
        COVARIATE_REFERENCE_SHA.into(),
    )
    .map_err(|error| format!("product monitor invalid: {error:?}"))
}

fn pooled_monitor() -> Result<ContextualPairedKernelMonitor, String> {
    let target = BoxTarget::new(vec![-0.005; 4], vec![0.005; 4])
        .map_err(|error| format!("pooled target invalid: {error:?}"))?;
    ContextualPairedKernelMonitor::new(
        response_map()?,
        1,
        target,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
        POOLED_TARGET_SHA.into(),
        TARGET_SHA.into(),
        RESPONSE_REFERENCE_SHA.into(),
    )
    .map_err(|error| format!("pooled monitor invalid: {error:?}"))
}

fn draw_pair(scenario: Scenario, rng: &mut SplitMix64) -> (f64, f64, f64) {
    let z = 0.2 * rng.sign();
    match scenario {
        Scenario::ConditionalPairNull => (
            z,
            0.2 * rng.sign(),
            0.2 * rng.sign(),
        ),
        Scenario::EqualMarginalConditionalShift if z > 0.0 => (z, 0.2, 0.8),
        Scenario::EqualMarginalConditionalShift => (z, 0.8, 0.2),
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
) -> Result<(Option<usize>, Option<usize>), String> {
    let seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9))
        .wrapping_add(match scenario {
            Scenario::ConditionalPairNull => 0x0500_0001,
            Scenario::EqualMarginalConditionalShift => 0x0500_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let mut product = product_monitor()?;
    let mut pooled = pooled_monitor()?;
    let mut product_alarm = None;
    let mut pooled_alarm = None;

    for time in 1..=config.horizon {
        let sequence = u64::try_from(time).map_err(|_| "sequence exceeds u64".to_string())?;
        let (z, x, y) = draw_pair(scenario, &mut rng);
        if product_alarm.is_none() {
            let update = product
                .observe_pair(sequence, &format!("product-{replication}-{time}"), &[z], &[x], &[y])
                .map_err(|error| format!("product monitor update failed: {error:?}"))?;
            if update.evidence.quarantined {
                product_alarm = Some(time);
            }
        }
        if pooled_alarm.is_none() {
            let update = pooled
                .observe_pair(sequence, &format!("pooled-{replication}-{time}"), 0, &[x], &[y])
                .map_err(|error| format!("pooled monitor update failed: {error:?}"))?;
            if update.evidence.quarantined {
                pooled_alarm = Some(time);
            }
        }
        for (arm, log_e) in [
            ("product_fourier_conditional", product.log_e_value()),
            ("pooled_response_fourier", pooled.log_e_value()),
        ] {
            writeln!(
                csv,
                "{},{},{},{},{:.9},{:.9},{:.9},{:.9}",
                scenario.name(),
                replication,
                time,
                arm,
                z,
                x,
                y,
                log_e,
            )
            .map_err(|error| error.to_string())?;
        }
        if product_alarm.is_some() && pooled_alarm.is_some() {
            break;
        }
    }
    Ok((product_alarm, pooled_alarm))
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut csv = String::from(
        "scenario,replication,time,arm,covariate,x,y,log_e_value\n",
    );
    for scenario in [Scenario::ConditionalPairNull, Scenario::EqualMarginalConditionalShift] {
        let mut product_alarms = Vec::with_capacity(config.replications);
        let mut pooled_alarms = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            let (product_alarm, pooled_alarm) =
                run_replication(scenario, replication, config, &mut csv)?;
            product_alarms.push(product_alarm);
            pooled_alarms.push(pooled_alarm);
        }
        if scenario == Scenario::ConditionalPairNull {
            eprintln!(
                "conditional_pair_null: reps={} product_false_alarms={} pooled_false_alarms={}",
                config.replications,
                product_alarms.iter().filter(|value| value.is_some()).count(),
                pooled_alarms.iter().filter(|value| value.is_some()).count(),
            );
        } else {
            let product_detections = product_alarms.iter().filter(|value| value.is_some()).count();
            let pooled_detections = pooled_alarms.iter().filter(|value| value.is_some()).count();
            eprintln!(
                "equal_marginal_conditional_shift: reps={} product_detected={} product_misses={} product_median_time={} pooled_detected={} pooled_misses={} pooled_median_time={}",
                config.replications,
                product_detections,
                config.replications - product_detections,
                median(&product_alarms).map_or_else(|| "NA".into(), |value| value.to_string()),
                pooled_detections,
                config.replications - pooled_detections,
                median(&pooled_alarms).map_or_else(|| "NA".into(), |value| value.to_string()),
            );
        }
    }
    print!("{csv}");
    Ok(())
}
