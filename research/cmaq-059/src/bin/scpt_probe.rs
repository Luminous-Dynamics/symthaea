#![forbid(unsafe_code)]

//! Deterministic wiring probe for the SCPT reference comparator.
//! This is not a false-alarm calibration or benchmark result.

use std::error::Error;

use cmaq_059_finite_sample_calibration::scenarios::{generate_stream, ScenarioConfig, ScenarioKind};
use cmaq_059_finite_sample_calibration::scpt::ScptReferenceMonitor;

fn run_scenario(kind: ScenarioKind, label: &str) -> Result<(), Box<dyn Error>> {
    let config = ScenarioConfig::standard(kind, 128);
    let stream = generate_stream("cmaq-059-scpt-probe-v1", &config, 7)?;
    let warmup = &stream.observations[..50];
    let mut monitor = ScptReferenceMonitor::from_warmup(warmup, 1e-6, 0.05)?;

    for observation in stream.observations.iter().skip(50) {
        let update = monitor.observe_pair(
            observation.sequence,
            &format!("{label}-{}", observation.sequence),
            &observation.covariate,
            &observation.x,
            &observation.y,
        )?;
        let lambdas = update.configurations.iter()
            .map(|item| format!("{:.8}", item.betting_fraction))
            .collect::<Vec<_>>()
            .join(";");
        let payoffs = update.configurations.iter()
            .map(|item| format!("{:.8}", item.payoff))
            .collect::<Vec<_>>()
            .join(";");
        let neighbors = update.configurations.iter()
            .map(|item| item.neighbor_count.to_string())
            .collect::<Vec<_>>()
            .join(";");
        println!(
            "{label},{},{:.12},{:.12},{},{},{},{},{}",
            update.sequence,
            update.mixture_factor,
            update.log_e_value,
            update.quarantined,
            stream.seed_sha256,
            stream.stream_sha256,
            lambdas,
            payoffs
        );
        if update.quarantined {
            break;
        }
        let _ = neighbors;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("scenario,sequence,mixture_factor,log_e_value,quarantined,seed_sha256,stream_sha256,lambdas,payoffs");
    run_scenario(ScenarioKind::EqualConditionalIid, "equal-conditional-iid")?;
    run_scenario(
        ScenarioKind::EqualMarginalConditionalPairingShift,
        "equal-marginal-conditional-pairing-shift",
    )?;
    run_scenario(ScenarioKind::ConditionalScaleShift, "conditional-scale-shift")?;
    Ok(())
}
