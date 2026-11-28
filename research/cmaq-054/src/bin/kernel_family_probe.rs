#![forbid(unsafe_code)]

//! Deterministic wiring probe for evidence-posterior reweighting across a
//! fixed two-scale conditional Fourier family. This is not a null calibration.

use cmaq_053_compositional_risk_budget::{
    ComposedRiskPlan, FleetBudgetPlan, FleetScope, Rational, RepresentationWeight,
    RiskPlanBindings,
};
use cmaq_054_evidence_adaptive_kernel_family::{
    KernelExpertConfig, KernelFamilyPortfolio,
};
use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const SHA_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const SHA_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

fn risk_plan() -> Result<ComposedRiskPlan, Box<dyn std::error::Error>> {
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        SHA_A.into(),
        SHA_B.into(),
    )?;
    let scope = FleetScope {
        stream_id: "probe-stream".into(),
        asset_id: "probe-asset".into(),
        channel_id: "paired-model-comparison".into(),
        deployment_id: "probe-deployment".into(),
        asset_index: 1,
        channel_index: 1,
        deployment_index: 1,
    };
    let local = fleet.local_alpha_for(&scope)?;
    let alpha_test = local.divided_by_integer(2)?;
    let delta_reference = local.divided_by_integer(2)?;
    Ok(ComposedRiskPlan::new(
        &fleet,
        scope,
        alpha_test,
        delta_reference,
        vec![
            RepresentationWeight {
                member_id: "low-frequency".into(),
                weight: Rational::new(1, 2)?,
            },
            RepresentationWeight {
                member_id: "high-frequency".into(),
                weight: Rational::new(1, 2)?,
            },
        ],
        RiskPlanBindings {
            deployment_plan_sha256: SHA_C.into(),
            reference_manifest_sha256: SHA_D.into(),
            target_envelope_sha256: SHA_E.into(),
            monitor_build_sha256: SHA_F.into(),
            representation_inventory_sha256: SHA_1.into(),
            calibration_method_sha256: SHA_2.into(),
        },
    )?)
}

fn expert(
    id: &str,
    response_frequency: f64,
    response_map_sha: &str,
    covariate_map_sha: &str,
    target_sha: &str,
    prior_weight: Rational,
) -> Result<KernelExpertConfig, Box<dyn std::error::Error>> {
    Ok(KernelExpertConfig {
        expert_id: id.into(),
        response_map: FourierFeatureMap::new(
            1,
            vec![vec![response_frequency]],
            response_map_sha.into(),
        )?,
        covariate_map: FourierFeatureMap::new(
            1,
            vec![vec![0.0]],
            covariate_map_sha.into(),
        )?,
        target: BoxTarget::new(vec![-0.005; 4], vec![0.005; 4])?,
        prior_weight,
        target_sha256: target_sha.into(),
        response_reference_sha256: SHA_E.into(),
        covariate_reference_sha256: SHA_F.into(),
    })
}

fn new_family() -> Result<KernelFamilyPortfolio, Box<dyn std::error::Error>> {
    let plan = risk_plan()?;
    let configurations = vec![
        expert("low-frequency", 1.0, SHA_A, SHA_B, SHA_C, Rational::new(1, 2)?)?,
        expert("high-frequency", 0.5, SHA_D, SHA_E, SHA_F, Rational::new(1, 2)?)?,
    ];
    Ok(KernelFamilyPortfolio::new(&plan, SHA_1.into(), configurations)?)
}

fn emit(
    scenario: &str,
    update: &cmaq_054_evidence_adaptive_kernel_family::KernelFamilyUpdate,
) {
    let low = update
        .experts
        .iter()
        .find(|expert| expert.expert_id == "low-frequency")
        .expect("registered low-frequency expert");
    let high = update
        .experts
        .iter()
        .find(|expert| expert.expert_id == "high-frequency")
        .expect("registered high-frequency expert");
    println!(
        "{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
        scenario,
        update.sequence,
        update.log_mixture_e_value,
        low.log_e_value,
        high.log_e_value,
        low.posterior_weight,
        high.posterior_weight,
        update.quarantined
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("scenario,sequence,log_mixture_e,low_log_e,high_log_e,low_posterior,high_posterior,quarantined");

    let mut null_family = new_family()?;
    for sequence in 1..=128u64 {
        let update = null_family.observe_pair(
            sequence,
            &format!("null-{sequence}"),
            &[(sequence as f64).sin()],
            &[0.25],
            &[0.25],
        )?;
        emit("equal_pair_null", &update);
    }

    let mut shifted_family = new_family()?;
    for sequence in 1..=200u64 {
        let update = shifted_family.observe_pair(
            sequence,
            &format!("scale-shift-{sequence}"),
            &[0.0],
            &[0.0],
            &[std::f64::consts::TAU],
        )?;
        emit("low_frequency_null_space_high_frequency_shift", &update);
        if update.quarantined {
            break;
        }
    }
    Ok(())
}
