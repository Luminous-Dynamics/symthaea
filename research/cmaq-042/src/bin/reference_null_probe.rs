#![forbid(unsafe_code)]

//! Smoke experiment for finite-horizon reference-null calibration.
//!
//! Uses independent fair-coin paths and the nonnegative martingale
//! M_t = product_{i<=t} (1 + 0.5 X_i), X_i in {-1,+1}.
//! This is a generic simple-null e-process experiment, not a PRM or a
//! deployment benchmark. Each test path receives a fresh independent bank.

use std::env;
use std::fmt::Write as _;

use cmaq_042_finite_horizon_threshold::{
    FiniteHorizonGate, ReferenceNullBank, ReferencePathMaximum, TestPathMaximum,
    ThresholdDecision,
};

const ALPHA_NUMERATOR: u32 = 1;
const ALPHA_DENOMINATOR: u32 = 20;
const REFERENCE_COUNT: usize = 99;
const MONITOR_BUILD: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SCORE_DEFINITION: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const RANK_REFERENCE: &str =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const NULL_SIMULATOR: &str =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

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

    fn fair_sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    replications: usize,
    horizon: u64,
    seed: u64,
}

fn parse_config() -> Result<Config, String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 3 {
        return Err("usage: reference_null_probe [replications] [horizon] [seed]".into());
    }
    let parse = |index: usize, default: &str, name: &str| -> Result<u64, String> {
        args.get(index)
            .map(String::as_str)
            .unwrap_or(default)
            .parse::<u64>()
            .map_err(|_| format!("invalid {name}"))
    };
    let replications = usize::try_from(parse(0, "64", "replications")?)
        .map_err(|_| "replications exceed usize".to_string())?;
    let horizon = parse(1, "256", "horizon")?;
    let seed = parse(2, "20261127", "seed")?;
    if replications == 0 || horizon == 0 {
        return Err("replications and horizon must be positive".into());
    }
    Ok(Config {
        replications,
        horizon,
        seed,
    })
}

fn path_log_maximum(rng: &mut SplitMix64, horizon: u64) -> f64 {
    let log_up = 1.5f64.ln();
    let log_down = 0.5f64.ln();
    let mut log_wealth = 0.0;
    let mut maximum = 0.0;

    for _ in 0..horizon {
        log_wealth += if rng.fair_sign() > 0.0 { log_up } else { log_down };
        maximum = maximum.max(log_wealth);
    }
    maximum
}

fn digest_for(seed: u64) -> String {
    format!("{seed:064x}")
}

fn run_replication(
    replication: usize,
    config: Config,
    output: &mut String,
) -> Result<(bool, bool), String> {
    let replication_seed = config
        .seed
        .wrapping_add((replication as u64).wrapping_mul(0x9E37_79B9));

    let mut references = Vec::with_capacity(REFERENCE_COUNT);
    for index in 0..REFERENCE_COUNT {
        // Disjoint deterministic seeds make the simulated reference paths
        // separate pseudo-random streams; the test path uses another seed.
        let reference_key = replication_seed.wrapping_add(index as u64 + 1);
        let mut reference_rng = SplitMix64::new(reference_key);
        let log_maximum = path_log_maximum(&mut reference_rng, config.horizon);
        references.push(ReferencePathMaximum {
            trajectory_id: format!("null-{replication}-{index}"),
            trace_sha256: digest_for(reference_key),
            log_max_e_value: log_maximum,
        });
    }

    let test_key = replication_seed.wrapping_add(1_000);
    let mut test_rng = SplitMix64::new(test_key);
    let test_log_maximum = path_log_maximum(&mut test_rng, config.horizon);
    let bank = ReferenceNullBank {
        bank_id: format!("mission-bank-{replication}"),
        horizon: config.horizon,
        calibration_reference_size: 99,
        alpha_numerator: ALPHA_NUMERATOR,
        alpha_denominator: ALPHA_DENOMINATOR,
        monitor_build_sha256: MONITOR_BUILD.into(),
        score_definition_sha256: SCORE_DEFINITION.into(),
        rank_reference_sha256: RANK_REFERENCE.into(),
        null_simulator_sha256: NULL_SIMULATOR.into(),
        references,
    };
    let mut gate = FiniteHorizonGate::calibrate(bank)
        .map_err(|error| format!("reference bank rejected: {error:?}"))?;
    let test = TestPathMaximum {
        trajectory_id: format!("deployment-{replication}"),
        trace_sha256: digest_for(test_key),
        horizon: config.horizon,
        calibration_reference_size: 99,
        monitor_build_sha256: MONITOR_BUILD.into(),
        score_definition_sha256: SCORE_DEFINITION.into(),
        rank_reference_sha256: RANK_REFERENCE.into(),
        log_max_e_value: test_log_maximum,
    };
    let decision = gate
        .evaluate_once(&test)
        .map_err(|error| format!("test path rejected: {error:?}"))?;
    let reference_reject = matches!(decision, ThresholdDecision::Reject { .. });
    let log_threshold = gate.log_threshold().unwrap_or(f64::INFINITY);
    let ville_reject = test_log_maximum >= (f64::from(ALPHA_DENOMINATOR)
        / f64::from(ALPHA_NUMERATOR))
        .ln();

    writeln!(
        output,
        "{replication},{test_log_maximum:.12},{log_threshold:.12},{reference_reject},{ville_reject}"
    )
    .map_err(|error| error.to_string())?;
    Ok((reference_reject, ville_reject))
}

fn main() -> Result<(), String> {
    let config = parse_config()?;
    let mut output = String::from(
        "replication,test_log_maximum,reference_log_threshold,reference_null_reject,ville_reject\n",
    );
    let mut reference_rejects = 0usize;
    let mut ville_rejects = 0usize;

    for replication in 0..config.replications {
        let (reference_reject, ville_reject) =
            run_replication(replication, config, &mut output)?;
        reference_rejects += usize::from(reference_reject);
        ville_rejects += usize::from(ville_reject);
    }

    let reference_rate = reference_rejects as f64 / config.replications as f64;
    let ville_rate = ville_rejects as f64 / config.replications as f64;
    eprintln!(
        "replications={} horizon={} bank_size={} alpha=0.05 reference_null_rejections={} rate={:.6} ville_rejections={} rate={:.6}",
        config.replications,
        config.horizon,
        REFERENCE_COUNT,
        reference_rejects,
        reference_rate,
        ville_rejects,
        ville_rate
    );
    print!("{output}");
    Ok(())
}
