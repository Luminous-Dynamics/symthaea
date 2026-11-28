#![forbid(unsafe_code)]

//! Deterministic wiring probe for predictable adaptive single-bet monitoring.
//! It demonstrates lagged expert selection on an equal-marginal conditional
//! pairing alternative and zero wealth movement on exact paired equality.
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
use cmaq_057_predictable_adaptive_kernel_betting::PredictableAdaptiveKernelMonitor;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn configurations() -> Vec<KernelExpertConfig> {
    vec![
        KernelExpertConfig {
            expert_id: "high-frequency".into(),
            response_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                SHA_A.into(),
            )
            .expect("fixed response Fourier map is valid"),
            covariate_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                SHA_B.into(),
            )
            .expect("fixed covariate Fourier map is valid"),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])
                .expect("fixed target contains zero"),
            prior_weight: Rational::new(1, 2).expect("positive rational prior"),
            target_sha256: SHA_C.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        },
        KernelExpertConfig {
            expert_id: "low-frequency".into(),
            response_map: FourierFeatureMap::new(
                1,
                vec![vec![1.0]],
                SHA_D.into(),
            )
            .expect("fixed response Fourier map is valid"),
            covariate_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                SHA_E.into(),
            )
            .expect("fixed covariate Fourier map is valid"),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])
                .expect("fixed target contains zero"),
            prior_weight: Rational::new(1, 2).expect("positive rational prior"),
            target_sha256: SHA_F.into(),
            response_reference_sha256: SHA_C.into(),
            covariate_reference_sha256: SHA_D.into(),
        },
    ]
}

fn new_monitor(
    configs: &[KernelExpertConfig],
) -> Result<PredictableAdaptiveKernelMonitor, Box<dyn Error>> {
    let manifest =
        KernelFamilyManifest::build(configs, NullCalibration::direct_v1())?;
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        SHA_A.into(),
        SHA_B.into(),
    )?;
    let scope = FleetScope {
        stream_id: "cmaq-057-probe".into(),
        asset_id: "probe-asset".into(),
        channel_id: "conditional-paired-monitor".into(),
        deployment_id: "probe-deployment".into(),
        asset_index: 1,
        channel_index: 1,
        deployment_index: 1,
    };
    let local = fleet.local_alpha_for(&scope)?;
    let risk_plan = TypedRiskPlan::new(
        &fleet,
        scope,
        local.divided_by_integer(2)?,
        &manifest,
    )?;
    Ok(PredictableAdaptiveKernelMonitor::new(
        &risk_plan,
        &manifest,
        configs,
        0.5,
    )?)
}

fn emit(
    scenario: &str,
    update: &cmaq_057_predictable_adaptive_kernel_betting::PredictableAdaptiveUpdate,
) {
    println!(
        "{},{},{},{:.9},{:.9},{:.9},{:.9},{},{}",
        scenario,
        update.action.sequence,
        update.action.selected_expert_id,
        update.action.historical_score_used,
        update.action.projected_feature_difference,
        update.action.log_e_value,
        update.action.direction_norm,
        update.action.quarantined,
        update.policy_trace_sha256,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "scenario,sequence,selected_expert,lagged_score,projected_difference,log_e_value,direction_norm,quarantined,policy_trace_sha256"
    );
    let configs = configurations();

    let mut null_monitor = new_monitor(&configs)?;
    for sequence in 1..=64u64 {
        let update = null_monitor.observe_pair(
            sequence,
            &format!("null-pair-{sequence}"),
            &[(sequence as f64 / 13.0).sin()],
            &[0.25],
            &[0.25],
        )?;
        emit("exact-paired-equality-wiring", &update);
    }

    let mut adaptive_monitor = new_monitor(&configs)?;
    for sequence in 1..=64u64 {
        // X and Y have identical marginals over every two-pair block:
        // X=(0,pi), Y=(pi,0). Their pairing with the predictable covariate
        // changes, so a conditional representation can distinguish them.
        let (covariate, x, y) = if sequence % 2 == 0 {
            (std::f64::consts::PI, std::f64::consts::PI, 0.0)
        } else {
            (0.0, 0.0, std::f64::consts::PI)
        };
        let update = adaptive_monitor.observe_pair(
            sequence,
            &format!("conditional-shift-{sequence}"),
            &[covariate],
            &[x],
            &[y],
        )?;
        emit("equal-marginal-conditional-pairing-shift-wiring", &update);
        if update.action.quarantined {
            break;
        }
    }
    Ok(())
}
