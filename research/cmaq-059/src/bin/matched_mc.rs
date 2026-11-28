#![forbid(unsafe_code)]

//! Matched-stream Monte Carlo comparison of C058 and the SCPT reference arm.
//!
//! This runner writes raw per-replicate outcomes and aggregate exact intervals.
//! It is a reproducible benchmark instrument, not an independent validation
//! that synthetic paths satisfy external data-generating assumptions.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::{FleetBudgetPlan, FleetScope, Rational};
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{KernelFamilyManifest, NullCalibration, TypedRiskPlan};
use cmaq_058_exact_restart_renewal::CapitalRestartRenewalMonitor;
use cmaq_059_finite_sample_calibration::{
    clopper_pearson, summarize_detection_trials, DetectionTrial,
};
use cmaq_059_finite_sample_calibration::scenarios::{
    generate_stream, ScenarioConfig, ScenarioKind,
};
use cmaq_059_finite_sample_calibration::scpt::{median_heuristic_sigma, ScptReferenceMonitor};

const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const HASH_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const HASH_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const HASH_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const EXPERIMENT_ID: &str = "cmaq-059-matched-scpt-c058-v1";

#[derive(Clone, Copy, Debug)]
enum Arm {
    Cmaq058,
    ScptFixedBandwidth,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Cmaq058 => "CMAQ-058-capital-only",
            Self::ScptFixedBandwidth => "SCPT-fixed-bandwidth",
        }
    }
}

#[derive(Clone, Debug)]
struct ArmOutcome {
    first_alarm: Option<u64>,
    feature_coordinates: u64,
    kernel_evaluations: u64,
    update_latencies_ns: Vec<u64>,
}

fn configs() -> Result<Vec<KernelExpertConfig>, Box<dyn Error>> {
    let target = BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])?;
    Ok(vec![
        KernelExpertConfig {
            expert_id: "low-frequency".into(),
            response_map: FourierFeatureMap::new(1, vec![vec![0.5]], HASH_A.into())?,
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], HASH_B.into())?,
            target: target.clone(),
            prior_weight: Rational::new(1, 2)?,
            target_sha256: HASH_C.into(),
            response_reference_sha256: HASH_E.into(),
            covariate_reference_sha256: HASH_F.into(),
        },
        KernelExpertConfig {
            expert_id: "high-frequency".into(),
            response_map: FourierFeatureMap::new(1, vec![vec![2.0]], HASH_D.into())?,
            covariate_map: FourierFeatureMap::new(1, vec![vec![1.5]], HASH_E.into())?,
            target,
            prior_weight: Rational::new(1, 2)?,
            target_sha256: HASH_C.into(),
            response_reference_sha256: HASH_E.into(),
            covariate_reference_sha256: HASH_F.into(),
        },
    ])
}

fn alpha_and_plan(
    fleet: &FleetBudgetPlan,
    manifest: &KernelFamilyManifest,
    stream_id: String,
) -> Result<(TypedRiskPlan, Rational), Box<dyn Error>> {
    let scope = FleetScope {
        stream_id,
        asset_id: "synthetic-asset".into(),
        evidence_channel_id: "conditional-paired-response".into(),
        asset_index: 1,
        evidence_channel_index: 1,
        deployment_index: 1,
    };
    let local = fleet.local_alpha_for(&scope)?;
    let alpha_test = local.divided_by_integer(2)?;
    let plan = TypedRiskPlan::new(fleet, scope, alpha_test, manifest)?;
    Ok((plan, alpha_test))
}

fn coordinates_per_c058_pair(configurations: &[KernelExpertConfig]) -> u64 {
    configurations.iter().map(|config| {
        let response = config.response_map.feature_dimension() as u64;
        let covariate = config.covariate_map.feature_dimension() as u64;
        2 * response + covariate + response * covariate
    }).sum()
}

fn parse_argument(args: &[String], name: &str, default: u64) -> Result<u64, Box<dyn Error>> {
    match args.iter().position(|value| value == name) {
        Some(index) => Ok(args.get(index + 1).ok_or("missing argument value")?.parse()?),
        None => Ok(default),
    }
}

fn nearest_rank(values: &mut [u64], quantile: f64) -> u64 {
    if values.is_empty() { return 0; }
    values.sort_unstable();
    let rank = ((quantile * values.len() as f64).ceil() as usize)
        .max(1)
        .min(values.len());
    values[rank - 1]
}

fn calibration_sigma() -> Result<(f64, String), Box<dyn Error>> {
    // Separate calibration stream: it is never used as a scored test stream.
    let mut config = ScenarioConfig::standard(ScenarioKind::EqualConditionalIid, 50);
    config.covariate_dimension = 1;
    let stream = generate_stream("cmaq-059-scpt-bandwidth-calibration-v1", &config, 0)?;
    let sigma = median_heuristic_sigma(&stream.observations)?;
    Ok((sigma, stream.stream_sha256))
}

fn scenario_config(kind: ScenarioKind, horizon: u64) -> ScenarioConfig {
    let mut config = ScenarioConfig::standard(kind, horizon);
    config.covariate_dimension = 1;
    config.amplitude = 0.5;
    if matches!(
        kind,
        ScenarioKind::EqualMarginalConditionalPairingShift
            | ScenarioKind::SparseConditionalShift
            | ScenarioKind::HighFrequencyConditionalShift
    ) {
        config.change_point = 0;
    }
    config
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let replicates = parse_argument(&args, "--replicates", 16)?;
    let horizon = parse_argument(&args, "--horizon", 128)?;
    if replicates == 0 || replicates > 10_000 || horizon == 0 || horizon > 1_024 {
        return Err("replicates must be 1..=10000 and horizon 1..=1024".into());
    }

    let configurations = configs()?;
    let manifest = KernelFamilyManifest::build(&configurations, NullCalibration::direct_v1())?;
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        HASH_A.into(),
        HASH_B.into(),
    )?;
    let (_, alpha_test) =
        alpha_and_plan(&fleet, &manifest, "matched-calibration-scope".into())?;
    let alpha = alpha_test.numerator() as f64 / alpha_test.denominator() as f64;
    let (sigma, calibration_stream_sha256) = calibration_sigma()?;

    let raw_file = File::create("cmaq-059-matched-mc-raw.csv")?;
    let summary_file = File::create("cmaq-059-matched-mc-summary.csv")?;
    let mut raw = BufWriter::new(raw_file);
    let mut summary = BufWriter::new(summary_file);

    writeln!(
        raw,
        "experiment,scenario,replicate,arm,horizon,change_point,first_alarm,is_false_alarm,is_detection,restricted_delay,feature_coordinates,kernel_evaluations,update_latency_p99_ns,seed_sha256,stream_sha256"
    )?;
    writeln!(
        summary,
        "experiment,scenario,arm,horizon,replicates,alpha,false_alarms,false_alarm_rate,false_alarm_lower,false_alarm_upper,detections,early_alarms,unconditional_power,power_lower,power_upper,restricted_mean_delay,delay_p50,delay_p90,delay_p95,feature_coordinates_total,kernel_evaluations_total,update_latency_p50_ns,update_latency_p99_ns,bandwidth,bandwidth_calibration_stream_sha256"
    )?;

    let scenarios = ScenarioKind::ALL;
    for kind in scenarios {
        let config = scenario_config(kind, horizon);
        let change_point = config.change_point;
        let mut c058_outcomes = Vec::with_capacity(replicates as usize);
        let mut scpt_outcomes = Vec::with_capacity(replicates as usize);
        let mut detections_c058 = Vec::with_capacity(replicates as usize);
        let mut detections_scpt = Vec::with_capacity(replicates as usize);
        let mut false_alarms_c058 = 0_u64;
        let mut false_alarms_scpt = 0_u64;

        for replicate in 0..replicates {
            let stream = generate_stream(EXPERIMENT_ID, &config, replicate)?;
            let stream_id = format!("{}-replicate-{replicate}", kind.id());
            let (risk_plan, _) = alpha_and_plan(&fleet, &manifest, stream_id)?;
            let mut c058 = CapitalRestartRenewalMonitor::new(
                &risk_plan,
                &manifest,
                &configurations,
                0.5,
                horizon,
            )?;
            let mut scpt = ScptReferenceMonitor::paper_default(sigma, 1e-6, alpha)?;

            let mut c058_alarm = None;
            let mut scpt_alarm = None;
            let mut c058_coordinates = 0_u64;
            let mut scpt_kernel_evaluations = 0_u64;
            let mut c058_latencies = Vec::new();
            let mut scpt_latencies = Vec::new();

            for observation in &stream.observations {
                let pair_id = format!("{}-{}", kind.id(), observation.sequence);

                if c058_alarm.is_none() {
                    let start = Instant::now();
                    let update = c058.observe_pair(
                        observation.sequence,
                        &pair_id,
                        &observation.covariate,
                        &observation.x,
                        &observation.y,
                    )?;
                    c058_latencies.push(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                    c058_coordinates = c058_coordinates.saturating_add(c058_coordinates_per_pair(&configurations));
                    if update.quarantined { c058_alarm = Some(observation.sequence); }
                }

                if scpt_alarm.is_none() {
                    let start = Instant::now();
                    let update = scpt.observe_pair(
                        observation.sequence,
                        &pair_id,
                        &observation.covariate,
                        &observation.x,
                        &observation.y,
                    )?;
                    scpt_latencies.push(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                    for configuration in &update.configurations {
                        let k = configuration.neighbor_count as u64;
                        scpt_kernel_evaluations = scpt_kernel_evaluations
                            .saturating_add(4_u64.saturating_mul(k.saturating_add(k.saturating_mul(k))));
                    }
                    if update.quarantined { scpt_alarm = Some(observation.sequence); }
                }
            }

            let c058_p99 = nearest_rank(&mut c058_latencies, 0.99);
            let scpt_p99 = nearest_rank(&mut scpt_latencies, 0.99);
            let c058_outcome = ArmOutcome {
                first_alarm: c058_alarm,
                feature_coordinates: c058_coordinates,
                kernel_evaluations: 0,
                update_latencies_ns: c058_latencies,
            };
            let scpt_outcome = ArmOutcome {
                first_alarm: scpt_alarm,
                feature_coordinates: 0,
                kernel_evaluations: scpt_kernel_evaluations,
                update_latencies_ns: scpt_latencies,
            };

            if kind.is_null() {
                false_alarms_c058 += u64::from(c058_alarm.is_some());
                false_alarms_scpt += u64::from(scpt_alarm.is_some());
            } else {
                detections_c058.push(DetectionTrial {
                    horizon,
                    change_point,
                    first_alarm: c058_alarm,
                });
                detections_scpt.push(DetectionTrial {
                    horizon,
                    change_point,
                    first_alarm: scpt_alarm,
                });
            }

            for (arm, outcome) in [
                (Arm::Cmaq058, &c058_outcome),
                (Arm::ScptFixedBandwidth, &scpt_outcome),
            ] {
                let is_detection = !kind.is_null()
                    && outcome.first_alarm.is_some_and(|alarm| alarm > change_point);
                let restricted_delay = if kind.is_null() {
                    String::new()
                } else if is_detection {
                    (outcome.first_alarm.unwrap() - change_point).to_string()
                } else {
                    horizon.saturating_add(1).to_string()
                };
                let first_alarm = outcome.first_alarm.map_or(String::new(), |x| x.to_string());
                let is_false_alarm = kind.is_null() && outcome.first_alarm.is_some();
                let latency_p99 = outcome.update_latencies_ns.first().copied().unwrap_or(0);
                writeln!(
                    raw,
                    "{EXPERIMENT_ID},{},{replicate},{},{horizon},{change_point},{},{},{},{},{},{},{},{},{}",
                    kind.id(),
                    arm.name(),
                    first_alarm,
                    is_false_alarm,
                    is_detection,
                    restricted_delay,
                    outcome.feature_coordinates,
                    outcome.kernel_evaluations,
                    latency_p99,
                    stream.seed_sha256,
                    stream.stream_sha256,
                )?;
            }

            c058_outcomes.push(c058_outcome);
            scpt_outcomes.push(scpt_outcome);
        }

        for (arm, outcomes) in [
            (Arm::Cmaq058, &c058_outcomes),
            (Arm::ScptFixedBandwidth, &scpt_outcomes),
        ] {
            let feature_coordinates_total =
                outcomes.iter().map(|x| x.feature_coordinates).sum::<u64>();
            let kernel_evaluations_total =
                outcomes.iter().map(|x| x.kernel_evaluations).sum::<u64>();
            let mut latencies = outcomes.iter()
                .flat_map(|x| x.update_latencies_ns.iter().copied())
                .collect::<Vec<_>>();
            let mut latency_copy = latencies.clone();
            let latency_p50 = nearest_rank(&mut latency_copy, 0.50);
            let latency_p99 = nearest_rank(&mut latencies, 0.99);

            if kind.is_null() {
                let false_alarms = match arm {
                    Arm::Cmaq058 => false_alarms_c058,
                    Arm::ScptFixedBandwidth => false_alarms_scpt,
                };
                let interval = clopper_pearson(false_alarms, replicates, 0.95)?;
                let fields = vec![
                    EXPERIMENT_ID.to_string(),
                    kind.id().to_string(),
                    arm.name().to_string(),
                    horizon.to_string(),
                    replicates.to_string(),
                    format!("{alpha:.12}"),
                    false_alarms.to_string(),
                    format!("{:.12}", interval.estimate),
                    format!("{:.12}", interval.lower),
                    format!("{:.12}", interval.upper),
                    String::new(), String::new(), String::new(), String::new(),
                    String::new(), String::new(), String::new(), String::new(),
                    String::new(),
                    feature_coordinates_total.to_string(),
                    kernel_evaluations_total.to_string(),
                    latency_p50.to_string(),
                    latency_p99.to_string(),
                    format!("{sigma:.12}"),
                    calibration_stream_sha256.clone(),
                ];
                writeln!(summary, "{}", fields.join(","))?;
            } else {
                let trials = match arm {
                    Arm::Cmaq058 => &detections_c058,
                    Arm::ScptFixedBandwidth => &detections_scpt,
                };
                let detection = summarize_detection_trials(trials, 0.95)?;
                let fields = vec![
                    EXPERIMENT_ID.to_string(),
                    kind.id().to_string(),
                    arm.name().to_string(),
                    horizon.to_string(),
                    replicates.to_string(),
                    format!("{alpha:.12}"),
                    String::new(), String::new(), String::new(), String::new(),
                    detection.detections.to_string(),
                    detection.early_alarms.to_string(),
                    format!("{:.12}", detection.unconditional_power),
                    format!("{:.12}", detection.power_interval.lower),
                    format!("{:.12}", detection.power_interval.upper),
                    format!("{:.4}", detection.restricted_mean_delay),
                    detection.restricted_delay_p50.to_string(),
                    detection.restricted_delay_p90.to_string(),
                    detection.restricted_delay_p95.to_string(),
                    feature_coordinates_total.to_string(),
                    kernel_evaluations_total.to_string(),
                    latency_p50.to_string(),
                    latency_p99.to_string(),
                    format!("{sigma:.12}"),
                    calibration_stream_sha256.clone(),
                ];
                writeln!(summary, "{}", fields.join(","))?;
            }
            }
        }
    }

    raw.flush()?;
    summary.flush()?;
    eprintln!(
        "completed experiment={} replicates={} horizon={} alpha={} sigma={} calibration_stream={}",
        EXPERIMENT_ID, replicates, horizon, alpha, sigma, calibration_stream_sha256
    );
    Ok(())
}
