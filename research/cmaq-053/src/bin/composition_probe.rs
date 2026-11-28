#![forbid(unsafe_code)]

//! Emits a small deterministic fleet allocation table.
//!
//! This is an arithmetic/receipt wiring probe only. It is not a statistical
//! false-alarm experiment and does not simulate the child e-processes.

use cmaq_053_compositional_risk_budget::{
    ComposedRiskPlan, CompositionLedger, FleetBudgetPlan, FleetScope, Rational,
    RepresentationWeight, RiskPlanBindings,
};

fn digest(value: u64) -> String {
    format!("{value:064x}")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fleet = FleetBudgetPlan::new(
        Rational::probability(1, 20)?,
        digest(1),
        digest(2),
    )?;
    let mut ledger = CompositionLedger::new(&fleet);
    println!(
        "asset_index,channel_index,deployment_index,local_alpha,alpha_test,delta_reference,member_weight_sum"
    );

    let mut plan_index = 10u64;
    for asset in 1..=3u64 {
        for channel in 1..=2u64 {
            for deployment in 1..=2u64 {
                let scope = FleetScope {
                    stream_id: format!("stream-{asset}-{channel}-{deployment}"),
                    asset_id: format!("asset-{asset}"),
                    channel_id: format!("channel-{channel}"),
                    deployment_id: format!("deployment-{deployment}"),
                    asset_index: asset,
                    channel_index: channel,
                    deployment_index: deployment,
                };
                let local = fleet.local_alpha_for(&scope)?;
                let alpha_test = local.divided_by_integer(2)?;
                let delta = local.divided_by_integer(2)?;
                let bindings = RiskPlanBindings {
                    deployment_plan_sha256: digest(plan_index),
                    reference_manifest_sha256: digest(plan_index + 100),
                    target_envelope_sha256: digest(plan_index + 200),
                    monitor_build_sha256: digest(plan_index + 300),
                    representation_inventory_sha256: digest(plan_index + 400),
                    calibration_method_sha256: digest(plan_index + 500),
                };
                plan_index = plan_index.checked_add(1).expect("plan index overflow");

                let plan = ComposedRiskPlan::new(
                    &fleet,
                    scope,
                    alpha_test,
                    delta,
                    vec![
                        RepresentationWeight {
                            member_id: "context".into(),
                            weight: Rational::new(1, 2)?,
                        },
                        RepresentationWeight {
                            member_id: "continuous".into(),
                            weight: Rational::new(1, 2)?,
                        },
                    ],
                    bindings,
                )?;
                let receipt = ledger.register_once(&fleet, &plan)?;
                let member_weight_sum = receipt
                    .representation_weights
                    .iter()
                    .map(|member| member.weight)
                    .reduce(|left, right| left.checked_add(right).expect("validated weights"))
                    .expect("nonempty representation portfolio");
                println!(
                    "{},{},{},{}/{},{}/{},{}/{},{} / {}",
                    receipt.scope.asset_index,
                    receipt.scope.channel_index,
                    receipt.scope.deployment_index,
                    receipt.local_alpha.numerator(),
                    receipt.local_alpha.denominator(),
                    receipt.alpha_test.numerator(),
                    receipt.alpha_test.denominator(),
                    receipt.delta_reference_failure.numerator(),
                    receipt.delta_reference_failure.denominator(),
                    member_weight_sum.numerator(),
                    member_weight_sum.denominator(),
                );
            }
        }
    }
    eprintln!("registered {} immutable deployment receipts", ledger.registration_count());
    Ok(())
}
