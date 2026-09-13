// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic evidence reporter for the synthetic HYPERSPACE-001 campaign.
//!
//! This executable emits a machine-readable receipt. It does not widen the
//! scientific claim: all results concern the exact synthetic Euclidean benchmark.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousValidityOracle, validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceBenchmarkError, HyperspaceDimension,
    canonical_shell_problem, certify_radial_separation, evaluator_reference_escape_path,
    qualify_projection_trap,
};
use symthaea_core::hyperspace_campaign::{
    ArmTruth, HyperspaceArm, HyperspaceCampaignConfig, PlannerObservation,
    run_hyperspace_campaign,
};
use symthaea_core::hyperspace_fixed_axis::qualify_fixed_w_h3;
use symthaea_core::reachability::UnknownReason;
use symthaea_core::sampling_reachability::RrtConfig;

const SCHEMA: &str = "symthaea.hyperspace-001.evidence.v1";
const W_SHELL: f64 = 0.25;
const ACCESS_INCREMENT: f64 = 0.000_976_562_5; // 2^-10
const CLEARANCE_MARGIN: f64 = 0.000_488_281_25; // 2^-11

fn main() -> Result<(), Box<dyn Error>> {
    let (subject, output) = parse_args()?;

    let campaign = campaign_report();
    let fixed_h3 = fixed_h3_report();
    let threshold = threshold_report();
    let pass = component_pass(&campaign) && component_pass(&fixed_h3) && component_pass(&threshold);

    let report = json!({
        "schema": SCHEMA,
        "source_subject": subject,
        "package": {
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "claim_scope": "synthetic Euclidean navigation competence only; not evidence for physically accessible extra dimensions",
        "verdict": if pass { "PASS" } else { "FAIL" },
        "campaign": campaign,
        "fixed_w_h3": fixed_h3,
        "access_threshold": threshold,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("HYPERSPACE-001 verdict: {}", report["verdict"].as_str().unwrap_or("FAIL"));
    println!("receipt: {}", output.display());
    Ok(())
}

fn parse_args() -> Result<(String, PathBuf), Box<dyn Error>> {
    let mut subject = None;
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject" => subject = args.next(),
            "--output" => output = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let subject = subject.ok_or("--subject is required")?;
    if subject.len() != 40 || !subject.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--subject must be an exact 40-hex commit SHA".into());
    }
    let output = output.ok_or("--output is required")?;
    Ok((subject.to_ascii_lowercase(), output))
}

fn component_pass(value: &Value) -> bool {
    value.get("pass").and_then(Value::as_bool) == Some(true)
}

fn campaign_report() -> Value {
    let config = HyperspaceCampaignConfig::reference_v1();
    let result = match run_hyperspace_campaign(config) {
        Ok(result) => result,
        Err(error) => {
            return json!({
                "pass": false,
                "status": "ERROR",
                "error": error.to_string(),
                "config_identity": hex32(config.identity()),
            });
        }
    };

    let expected_order = [
        HyperspaceArm::H0,
        HyperspaceArm::H1,
        HyperspaceArm::H2,
        HyperspaceArm::H3,
        HyperspaceArm::H4,
        HyperspaceArm::H5,
        HyperspaceArm::H6,
    ];
    let order_ok = result.arms().len() == expected_order.len()
        && result
            .arms()
            .iter()
            .zip(expected_order)
            .all(|(receipt, expected)| receipt.arm() == expected);

    let h0 = result.arm(HyperspaceArm::H0);
    let h1 = result.arm(HyperspaceArm::H1);
    let h2 = result.arm(HyperspaceArm::H2);
    let h3 = result.arm(HyperspaceArm::H3);
    let h4 = result.arm(HyperspaceArm::H4);
    let h5 = result.arm(HyperspaceArm::H5);
    let h6 = result.arm(HyperspaceArm::H6);

    let h0_ok = h0.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::ConstructivelyFeasible { .. })
            && matches!(r.planner(), PlannerObservation::Feasible { .. })
    });
    let h1_ok = h1.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::AnalyticallyBlocked { .. })
            && matches!(r.planner(), PlannerObservation::Unknown { .. })
    });
    let h2_ok = h2.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::ConstructivelyFeasible { .. })
            && matches!(r.planner(), PlannerObservation::NotRun { .. })
    });
    let h3_ok = h3.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::AnalyticallyBlocked { .. })
            && matches!(r.planner(), PlannerObservation::NotRun { .. })
    });
    let h4_ok = h4.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::AnalyticallyBlocked { .. })
            && matches!(r.planner(), PlannerObservation::Unknown { .. })
    });
    let h5_ok = h5.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::ConstructivelyFeasible { .. })
            && matches!(
                r.planner(),
                PlannerObservation::Feasible {
                    max_abs_w: Some(max_abs_w),
                    ..
                } if *max_abs_w > W_SHELL
            )
    });
    let h6_ok = h6.is_some_and(|r| {
        matches!(r.truth(), ArmTruth::ProjectionTrapEstablished { .. })
    });

    let arms: Vec<Value> = result
        .arms()
        .iter()
        .map(|receipt| {
            json!({
                "arm": arm_name(receipt.arm()),
                "problem_identity": hex32(receipt.problem_identity()),
                "oracle_identity": hex32(receipt.oracle_identity()),
                "receipt_identity": hex32(receipt.identity()),
                "truth": truth_json(receipt.truth()),
                "planner": planner_json(receipt.planner()),
            })
        })
        .collect();

    let pass = order_ok && h0_ok && h1_ok && h2_ok && h3_ok && h4_ok && h5_ok && h6_ok;
    json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "config_identity": hex32(result.config_identity()),
        "campaign_identity": hex32(result.identity()),
        "gates": {
            "ordered_h0_h6": order_ok,
            "h0_warmup_feasible": h0_ok,
            "h1_r3_blocked_no_false_feasible": h1_ok,
            "h2_r4_constructive_existence": h2_ok,
            "h3_base_analytic_blocked": h3_ok,
            "h4_insufficient_w_blocked_no_false_feasible": h4_ok,
            "h5_target_discovers_material_w_route": h5_ok,
            "h6_target_path_projection_trap": h6_ok,
        },
        "arms": arms,
    })
}

fn fixed_h3_report() -> Value {
    let config = match RrtConfig::new(11, 1_000, 0.4, 0.05, 0.6) {
        Ok(config) => config,
        Err(error) => return json!({"pass": false, "status": "ERROR", "error": error.to_string()}),
    };
    match qualify_fixed_w_h3(config) {
        Ok(receipt) => {
            let pass = receipt.fixed_w().to_bits() == 0.0_f64.to_bits()
                && receipt.planner_reason() == &UnknownReason::BudgetExceeded
                && receipt.iterations() == 1_000
                && receipt.accepted_nodes() > 0
                && receipt.state_queries() > 0
                && receipt.segment_queries() > 0;
            json!({
                "pass": pass,
                "status": if pass { "PASS" } else { "FAIL" },
                "problem_identity": hex32(receipt.problem_identity()),
                "oracle_identity": hex32(receipt.oracle_identity()),
                "certificate_identity": hex32(receipt.certificate_identity()),
                "reduction_identity": hex32(receipt.reduction_identity()),
                "solver_identity": hex32(receipt.solver_identity()),
                "receipt_identity": hex32(receipt.identity()),
                "planner_reason": unknown_reason_name(receipt.planner_reason()),
                "iterations": receipt.iterations(),
                "accepted_nodes": receipt.accepted_nodes(),
                "state_queries": receipt.state_queries(),
                "segment_queries": receipt.segment_queries(),
                "fixed_w_bits": format!("{:016x}", receipt.fixed_w().to_bits()),
            })
        }
        Err(error) => json!({
            "pass": false,
            "status": "ERROR",
            "error": error.to_string(),
            "solver_config_identity": hex32(config.identity()),
        }),
    }
}

fn threshold_report() -> Value {
    let result = (|| -> Result<Value, Box<dyn Error>> {
        let blocked_oracle = FiniteWShellOracle::new(
            HyperspaceDimension::R4,
            1.0,
            2.0,
            W_SHELL,
            4.0,
            W_SHELL,
        )?;
        let blocked_problem = canonical_shell_problem(&blocked_oracle, 3.0)?;
        let blocked_certificate = certify_radial_separation(&blocked_problem, &blocked_oracle)?;

        let feasible_bound = W_SHELL + ACCESS_INCREMENT;
        let feasible_oracle = FiniteWShellOracle::new(
            HyperspaceDimension::R4,
            1.0,
            2.0,
            W_SHELL,
            4.0,
            feasible_bound,
        )?;
        let feasible_problem = canonical_shell_problem(&feasible_oracle, 3.0)?;
        let theorem_rejected_above_threshold = matches!(
            certify_radial_separation(&feasible_problem, &feasible_oracle),
            Err(HyperspaceBenchmarkError::SeparationNotApplicable { .. })
        );

        let path = evaluator_reference_escape_path(
            &feasible_problem,
            &feasible_oracle,
            CLEARANCE_MARGIN,
        )?;
        let validation = match validate_euclidean_path(&feasible_problem, &feasible_oracle, &path)? {
            ContinuousPathReplay::Valid(validation) => validation,
            other => return Err(format!("above-threshold witness replay was {other:?}").into()),
        };
        let projection = qualify_projection_trap(&feasible_problem, &feasible_oracle, &path)?;
        let witness_w = W_SHELL + CLEARANCE_MARGIN;
        let max_abs_w = path
            .iter()
            .map(|state| state[3].abs())
            .fold(0.0_f64, f64::max);
        let shell_held_fixed = blocked_oracle.inner_radius().to_bits() == feasible_oracle.inner_radius().to_bits()
            && blocked_oracle.outer_radius().to_bits() == feasible_oracle.outer_radius().to_bits()
            && blocked_oracle.w_obstacle_half_thickness().to_bits()
                == feasible_oracle.w_obstacle_half_thickness().to_bits()
            && blocked_oracle.xyz_bound().to_bits() == feasible_oracle.xyz_bound().to_bits();
        let pass = theorem_rejected_above_threshold
            && shell_held_fixed
            && max_abs_w.to_bits() == witness_w.to_bits()
            && projection.full_path_identity() == validation.path_identity();

        Ok(json!({
            "pass": pass,
            "status": if pass { "PASS" } else { "FAIL" },
            "w_shell_bits": format!("{:016x}", W_SHELL.to_bits()),
            "access_increment_bits": format!("{:016x}", ACCESS_INCREMENT.to_bits()),
            "clearance_margin_bits": format!("{:016x}", CLEARANCE_MARGIN.to_bits()),
            "blocked_problem_identity": hex32(blocked_problem.identity()),
            "blocked_oracle_identity": hex32(blocked_oracle.profile().identity()),
            "blocked_certificate_identity": hex32(blocked_certificate.identity()),
            "feasible_problem_identity": hex32(feasible_problem.identity()),
            "feasible_oracle_identity": hex32(feasible_oracle.profile().identity()),
            "feasible_path_identity": hex32(validation.path_identity()),
            "projection_trap_identity": hex32(projection.identity()),
            "shell_held_fixed": shell_held_fixed,
            "blocking_theorem_rejected_above_threshold": theorem_rejected_above_threshold,
            "max_abs_w_bits": format!("{:016x}", max_abs_w.to_bits()),
        }))
    })();

    match result {
        Ok(value) => value,
        Err(error) => json!({"pass": false, "status": "ERROR", "error": error.to_string()}),
    }
}

fn arm_name(arm: HyperspaceArm) -> &'static str {
    match arm {
        HyperspaceArm::H0 => "H0",
        HyperspaceArm::H1 => "H1",
        HyperspaceArm::H2 => "H2",
        HyperspaceArm::H3 => "H3",
        HyperspaceArm::H4 => "H4",
        HyperspaceArm::H5 => "H5",
        HyperspaceArm::H6 => "H6",
    }
}

fn truth_json(truth: &ArmTruth) -> Value {
    match truth {
        ArmTruth::ConstructivelyFeasible { path_identity, total_cost } => json!({
            "kind": "ConstructivelyFeasible",
            "path_identity": hex32(*path_identity),
            "total_cost": total_cost,
        }),
        ArmTruth::AnalyticallyBlocked { certificate_identity } => json!({
            "kind": "AnalyticallyBlocked",
            "certificate_identity": hex32(*certificate_identity),
        }),
        ArmTruth::ProjectionTrapEstablished { receipt_identity, full_path_identity } => json!({
            "kind": "ProjectionTrapEstablished",
            "receipt_identity": hex32(*receipt_identity),
            "full_path_identity": hex32(*full_path_identity),
        }),
        ArmTruth::NotEstablished => json!({"kind": "NotEstablished"}),
    }
}

fn planner_json(planner: &PlannerObservation) -> Value {
    match planner {
        PlannerObservation::Feasible {
            path_identity,
            total_cost,
            solver_identity,
            iterations,
            max_abs_w,
        } => json!({
            "kind": "Feasible",
            "path_identity": hex32(*path_identity),
            "total_cost": total_cost,
            "solver_identity": hex32(*solver_identity),
            "iterations": iterations,
            "max_abs_w": max_abs_w,
        }),
        PlannerObservation::Unknown {
            reason,
            solver_identity,
            iterations,
            accepted_nodes,
        } => json!({
            "kind": "Unknown",
            "reason": unknown_reason_name(reason),
            "solver_identity": hex32(*solver_identity),
            "iterations": iterations,
            "accepted_nodes": accepted_nodes,
        }),
        PlannerObservation::NotRun { reason } => json!({
            "kind": "NotRun",
            "reason": reason,
        }),
    }
}

fn unknown_reason_name(reason: &UnknownReason) -> &'static str {
    match reason {
        UnknownReason::BudgetExceeded => "BudgetExceeded",
        UnknownReason::NoPathFound => "NoPathFound",
        UnknownReason::ValidityOracleIncomplete => "ValidityOracleIncomplete",
        UnknownReason::NumericalFailure => "NumericalFailure",
        UnknownReason::ModelOutOfDomain => "ModelOutOfDomain",
        UnknownReason::ConstraintProjectionFailure => "ConstraintProjectionFailure",
        UnknownReason::UnsupportedSpace => "UnsupportedSpace",
    }
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
