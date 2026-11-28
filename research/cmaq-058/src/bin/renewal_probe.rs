#![forbid(unsafe_code)]

//! Deterministic wiring probe for capital-only exact restart renewal.
//! Includes an equal-pair null and equal-marginal conditional-pairing shift.
//! This is not a Type-I calibration or power benchmark.

use std::error::Error;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::{
    FleetBudgetPlan, FleetScope, Rational,
};
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{
    KernelFamilyManifest, NullCalibration, TypedRiskPlan,
};
use cmaq_058_exact_restart_renewal::CapitalRestartRenewalMonitor;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn configurations() -> Vec<KernelExpertConfig> {
    vec![
        KernelExpertConfig {
            expert_id: "low-frequency".into(),
            response_map: FourierFeatureMap::new(1, vec![vec![1.0]], SHA_A.into())
                .expect("fixed response map is valid"),
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], SHA_B.into())
                .expect("fixed covariate map is valid"),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])
                .expect("target contains zero"),
            prior_weight: Rational::new(1, 2).expect("positive prior"),
            target_sha256: SHA_C.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        },
        KernelExpertConfig {
            expert_id: "high-frequency".into(),
            response_map: FourierFeatureMap::new(1, vec![vec![0.5]], SHA_D.into())
                .expect("fixed response map is valid"),
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], SHA_E.into())
                .expect("fixed covariate map is valid"),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])
                .expect("target contains zero"),
            prior_weight: Rational::new(1, 2).expect("positive prior"),
            target_sha256: SHA_F.into(),
            response_reference_sha256: SHA_C.into(),
            covariate_reference_sha256: SHA_D.into(),
        },
    ]
}

fn new_monitor(
    configs: &[KernelExpertConfig],
    horizon: u64,
) -> Result<CapitalRestartRenewalMonitor, Box<dyn Error>> {
    let manifest = KernelFamilyManifest::build(
        configs,
        NullCalibration::direct_v1(),
    )?;
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        SHA_A.into(),
        SHA_B.into(),
    )?;
    let scope = FleetScope {
        stream_id: "cmaq-058-probe".into(),
        asset_id: "probe-asset".into(),
        channel_id: "conditional-pair".into(),
        deployment_id: "probe-deployment".into(),
        asset_index: 1,
        channel_index: 1,
        deployment_index: 1,
    };
    let local = fleet.local_alpha_for(&scope)?;
    let plan = TypedRiskPlan::new(
        &fleet,
        scope,
        local.divided_by_integer(2)?,
        &manifest,
    )?;
    Ok(CapitalRestartRenewalMonitor::new(
        &plan,
        &manifest,
        configs,
        0.5,
        horizon,
    )?)
}

fn emit(
    scenario: &str,
    update: &cmaq_058_exact_restart_renewal::RenewalUpdate,
) {
    let best = update
        .kernels
        .iter()
        .max_by(|left, right| left.posterior_weight.total_cmp(&right.posterior_weight));
    let (best_id, best_posterior) = best
        .map(|kernel| (kernel.expert_id.as_str(), kernel.posterior_weight))
        .unwrap_or(("none", 0.0));
    println!(
        "{},{},{:.12},{:.12},{},{:.12},{},{},{}",
        scenario,
        update.sequence,
        update.log_mixture_e_value,
        update.future_start_posterior,
        best_id,
        best_posterior,
        update.quarantined,
        update.factor_batch_sha256,
        update.policy_trace_sha256
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "scenario,sequence,log_mixture_e,future_start_posterior,best_kernel,best_kernel_posterior,quarantined,factor_batch_sha256,policy_trace_sha256"
    );
    let configs = configurations();

    let mut null_monitor = new_monitor(&configs, 64)?;
    for sequence in 1..=32u64 {
        let update = null_monitor.observe_pair(
            sequence,
            &format!("null-{sequence}"),
            &[(sequence as f64 / 17.0).sin()],
            &[0.25],
            &[0.25],
        )?;
        emit("exact-pair-null-wiring", &update);
    }

    let mut shifted_monitor = new_monitor(&configs, 128)?;
    for sequence in 1..=128u64 {
        // Over each two-pair block, X and Y have identical marginals while
        // their association with the predictable covariate differs.
        let (covariate, x, y) = if sequence % 2 == 0 {
            (std::f64::consts::PI, std::f64::consts::PI, 0.0)
        } else {
            (0.0, 0.0, std::f64::consts::PI)
        };
        let update = shifted_monitor.observe_pair(
            sequence,
            &format!("conditional-shift-{sequence}"),
            &[covariate],
            &[x],
            &[y],
        )?;
        emit("equal-marginal-conditional-shift-wiring", &update);
        if update.quarantined {
            break;
        }
    }
    Ok(())
}
