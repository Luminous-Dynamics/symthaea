#![forbid(unsafe_code)]

//! Reproducible wiring probe for compute-sparse predictable expert selection.
//! This is not a false-alarm calibration or power benchmark.

use std::error::Error;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::{FleetBudgetPlan, FleetScope, Rational};
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{KernelFamilyManifest, NullCalibration, TypedRiskPlan};
use cmaq_059_finite_sample_calibration::scenarios::{generate_stream, ScenarioConfig, ScenarioKind};
use cmaq_059_finite_sample_calibration::paired_stream_seed;
use cmaq_060_sparse_expert_selection::SparsePredictableExpertMonitor;

const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const HASH_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const HASH_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const HASH_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

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

fn monitor(seed: [u8; 32], configurations: &[KernelExpertConfig]) -> Result<SparsePredictableExpertMonitor, Box<dyn Error>> {
    let manifest = KernelFamilyManifest::build(configurations, NullCalibration::direct_v1())?;
    let fleet = FleetBudgetPlan::new(Rational::probability(1, 20)?, HASH_A.into(), HASH_B.into())?;
    let scope = FleetScope {
        stream_id: "cmaq-060-probe".into(),
        asset_id: "synthetic-asset".into(),
        evidence_channel_id: "paired-responses".into(),
        asset_index: 1,
        evidence_channel_index: 1,
        deployment_index: 1,
    };
    let local = fleet.local_alpha_for(&scope)?;
    let risk_plan = TypedRiskPlan::new(
        &fleet,
        scope,
        local.divided_by_integer(2)?,
        &manifest,
    )?;
    Ok(SparsePredictableExpertMonitor::new(
        &risk_plan,
        &manifest,
        configurations,
        0.5,
        0.1,
        256,
        seed,
    )?)
}

fn run(kind: ScenarioKind, label: &str) -> Result<(), Box<dyn Error>> {
    let mut config = ScenarioConfig::standard(kind, 256);
    config.covariate_dimension = 1;
    config.amplitude = 0.5;
    let stream = generate_stream("cmaq-060-probe-v1", &config, 17)?;
    let seed = paired_stream_seed("cmaq-060-selector-v1", label, 17)?;
    let configurations = configs()?;
    let mut monitor = monitor(seed, &configurations)?;

    for observation in &stream.observations {
        let update = monitor.observe_pair(
            observation.sequence,
            &format!("{label}-{}", observation.sequence),
            &observation.covariate,
            &observation.x,
            &observation.y,
        )?;
        println!(
            "{label},{},{},{:.8},{:.8},{:.12},{:.12},{},{},{}",
            update.sequence,
            update.selected_expert_id,
            update.selection_probability,
            update.factor,
            update.normalized_bandit_reward,
            update.log_e_value,
            update.feature_coordinates_evaluated,
            update.quarantined,
            update.policy_trace_sha256,
        );
        if update.quarantined { break; }
    }
    eprintln!(
        "scenario={} seed={} stream={} selected_counts={:?}",
        label,
        stream.seed_sha256,
        stream.stream_sha256,
        monitor.selected_counts(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("scenario,sequence,selected_expert,selection_probability,factor,normalized_reward,log_e_value,feature_coordinates,quarantined,trace_sha256");
    run(ScenarioKind::EqualConditionalIid, "equal-conditional-iid")?;
    run(ScenarioKind::SparseConditionalShift, "sparse-conditional-shift")?;
    run(
        ScenarioKind::EqualMarginalConditionalPairingShift,
        "equal-marginal-pairing-shift",
    )?;
    Ok(())
}
