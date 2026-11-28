#![forbid(unsafe_code)]

//! Deterministic end-to-end wiring probe for the all-start kernel-family oracle.
//! It demonstrates unit wealth under equal pairs and posterior movement under
//! a registered conditional-pairing alternative. It is not a null calibration
//! or a power benchmark.

use std::error::Error;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::{
    ComposedRiskPlan, FleetBudgetPlan, FleetScope, Rational, RepresentationWeight,
    RiskPlanBindings,
};
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_055_dyadic_start_time_prior::RestartableKernelFamilyPortfolio;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const SHA_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const SHA_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

fn make_risk_plan() -> Result<ComposedRiskPlan, Box<dyn Error>> {
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        SHA_A.into(),
        SHA_B.into(),
    )?;
    let scope = FleetScope {
        stream_id: "cmaq-055-probe".into(),
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

fn make_expert(
    id: &str,
    frequency: f64,
    prior: Rational,
    response_sha: &str,
    covariate_sha: &str,
    target_sha: &str,
) -> KernelExpertConfig {
    KernelExpertConfig {
        expert_id: id.into(),
        response_map: FourierFeatureMap::new(
            1,
            vec![vec![frequency]],
            response_sha.into(),
        )
        .expect("fixed response frequency map is valid"),
        covariate_map: FourierFeatureMap::new(
            1,
            vec![vec![0.5]],
            covariate_sha.into(),
        )
        .expect("fixed covariate frequency map is valid"),
        target: BoxTarget::new(vec![-0.005; 4], vec![0.005; 4])
            .expect("fixed target box is valid"),
        prior_weight: prior,
        target_sha256: target_sha.into(),
        response_reference_sha256: SHA_E.into(),
        covariate_reference_sha256: SHA_F.into(),
    }
}

fn make_portfolio(
    horizon: u64,
) -> Result<RestartableKernelFamilyPortfolio, Box<dyn Error>> {
    let experts = vec![
        make_expert(
            "low-frequency",
            1.0,
            Rational::new(1, 2)?,
            SHA_A,
            SHA_B,
            SHA_C,
        ),
        make_expert(
            "high-frequency",
            0.5,
            Rational::new(1, 2)?,
            SHA_D,
            SHA_E,
            SHA_F,
        ),
    ];
    Ok(RestartableKernelFamilyPortfolio::new(
        make_risk_plan()?,
        SHA_1.into(),
        experts,
        horizon,
    )?)
}

fn emit(
    scenario: &str,
    update: &cmaq_055_dyadic_start_time_prior::RestartPortfolioUpdate,
) {
    let best = update
        .kernel_start_posteriors
        .iter()
        .max_by(|left, right| left.joint_posterior.total_cmp(&right.joint_posterior));
    let (best_start, best_expert, best_joint) = best
        .map(|item| {
            (
                item.start_time.to_string(),
                item.expert_id.as_str(),
                item.joint_posterior,
            )
        })
        .unwrap_or_else(|| ("0".to_string(), "none", 0.0));
    println!(
        "{},{},{:.12},{:.12},{},{},{:.12},{}",
        scenario,
        update.sequence,
        update.log_mixture_e_value,
        update.future_start_posterior,
        best_start,
        best_expert,
        best_joint,
        update.quarantined
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "scenario,sequence,log_mixture_e,future_start_posterior,best_start,best_expert,best_joint_posterior,quarantined"
    );

    let mut null_portfolio = make_portfolio(64)?;
    for sequence in 1..=32u64 {
        let update = null_portfolio.observe_pair(
            sequence,
            &format!("null-pair-{sequence}"),
            &[(sequence as f64 / 10.0).sin()],
            &[0.25],
            &[0.25],
        )?;
        emit("equal-pair-null-wiring", &update);
    }

    let mut shifted_portfolio = make_portfolio(128)?;
    for sequence in 1..=128u64 {
        // Over each two-pair block, X and Y have identical marginals.
        // Their association with the predictable covariate is changed.
        let (covariate, x, y) = if sequence % 2 == 0 {
            (std::f64::consts::PI, std::f64::consts::TAU, 0.0)
        } else {
            (0.0, 0.0, std::f64::consts::TAU)
        };
        let update = shifted_portfolio.observe_pair(
            sequence,
            &format!("conditional-shift-pair-{sequence}"),
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
