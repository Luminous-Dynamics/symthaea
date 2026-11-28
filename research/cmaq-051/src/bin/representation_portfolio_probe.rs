#![forbid(unsafe_code)]

//! Smoke probe for a fixed equal-weight context/product representation portfolio.
//!
//! Covariate signs are independently randomized each time. Under the
//! conditional-pair null, X and Y are independent draws from the same
//! symmetric response law. Under the alternative, the pair relationship
//! reverses with the covariate sign, preserving the X and Y marginal mixtures.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_051_representation_eportfolio::RepresentationEPortfolio;

const RESPONSE_MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const COVARIATE_MAP_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const CONTEXT_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const PORTFOLIO_SHA: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const CONTEXT_TARGET_SHA: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const PRODUCT_TARGET_SHA: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const RESPONSE_REFERENCE_SHA: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const COVARIATE_REFERENCE_SHA: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const CONTEXT_COUNT: usize = 2;

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
        return Err("usage: representation_portfolio_probe [replications] [horizon] [seed]".into());
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
    Ok(Config {
        replications,
        horizon,
        seed,
    })
}

fn monitor() -> Result<RepresentationEPortfolio, String> {
    let response_map = FourierFeatureMap::new(
        1,
        vec![vec![1.0], vec![3.0]],
        RESPONSE_MAP_SHA.into(),
    )
    .map_err(|error| format!("response map invalid: {error:?}"))?;
    let covariate_map = FourierFeatureMap::new(
        1,
        vec![vec![1.0], vec![3.0]],
        COVARIATE_MAP_SHA.into(),
    )
    .map_err(|error| format!("covariate map invalid: {error:?}"))?;

    let contextual_target = BoxTarget::new(
        vec![-0.005; CONTEXT_COUNT * 4],
        vec![0.005; CONTEXT_COUNT * 4],
    )
    .map_err(|error| format!("context target invalid: {error:?}"))?;
    let product_target = BoxTarget::new(vec![-0.005; 16], vec![0.005; 16])
        .map_err(|error| format!("product target invalid: {error:?}"))?;

    RepresentationEPortfolio::new(
        response_map,
        covariate_map,
        CONTEXT_COUNT,
        contextual_target,
        product_target,
        ALPHA_NUMERATOR,
        ALPHA_DENOMINATOR,
        CONTEXT_SHA.into(),
        PORTFOLIO_SHA.into(),
        CONTEXT_TARGET_SHA.into(),
        PRODUCT_TARGET_SHA.into(),
        RESPONSE_REFERENCE_SHA.into(),
        COVARIATE_REFERENCE_SHA.into(),
    )
    .map_err(|error| format!("portfolio construction failed: {error:?}"))
}

fn draw_pair(
    scenario: Scenario,
    rng: &mut SplitMix64,
) -> (f64, usize, f64, f64) {
    let covariate = 0.2 * rng.sign();
    let context = if covariate > 0.0 { 0 } else { 1 };
    match scenario {
        Scenario::ConditionalPairNull => (
            covariate,
            context,
            0.2 * rng.sign(),
            0.2 * rng.sign(),
        ),
        Scenario::EqualMarginalConditionalShift if covariate > 0.0 => {
            (covariate, context, 0.2, 0.8)
        }
        Scenario::EqualMarginalConditionalShift => (covariate, context, 0.8, 0.2),
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
            Scenario::ConditionalPairNull => 0x0510_0001,
            Scenario::EqualMarginalConditionalShift => 0x0510_0002,
        });
    let mut rng = SplitMix64::new(seed);
    let mut portfolio = monitor()?;
    let mut alarm = None;

    for time in 1..=config.horizon {
        let sequence = u64::try_from(time).map_err(|_| "sequence exceeds u64".to_string())?;
        let (covariate, context, x, y) = draw_pair(scenario, &mut rng);
        let update = portfolio
            .observe_pair(
                sequence,
                &format!("{}-{replication}-{time}", scenario.name()),
                context,
                &[covariate],
                &[x],
                &[y],
            )
            .map_err(|error| format!("portfolio update failed: {error:?}"))?;

        writeln!(
            csv,
            "{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            scenario.name(),
            replication,
            sequence,
            context,
            covariate,
            x,
            y,
            update.contextual_member_log_e,
            update.product_member_log_e,
            update.mixture_log_e,
        )
        .map_err(|error| error.to_string())?;

        if update.quarantined {
            alarm = Some(time);
            break;
        }
    }
    Ok(alarm)
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut csv = String::from(
        "scenario,replication,sequence,context,covariate,x,y,log_e_context,log_e_product,log_e_mixture\n",
    );
    for scenario in [
        Scenario::ConditionalPairNull,
        Scenario::EqualMarginalConditionalShift,
    ] {
        let mut alarms = Vec::with_capacity(config.replications);
        for replication in 0..config.replications {
            alarms.push(run_replication(scenario, replication, config, &mut csv)?);
        }
        let alarm_count = alarms.iter().filter(|alarm| alarm.is_some()).count();
        if scenario == Scenario::ConditionalPairNull {
            eprintln!(
                "conditional_pair_null: reps={} portfolio_false_alarms={}",
                config.replications, alarm_count
            );
        } else {
            eprintln!(
                "equal_marginal_conditional_shift: reps={} detected={} misses={} median_alarm_sequence={}",
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
