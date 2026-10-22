//! Deterministic tests for assumption-bound certificate freshness.
//!
//! Usage: allostatic-certificate-lease-bench <raw.csv> <summary.csv>
//! The benchmark validates certificate lifecycle mechanics only. It is not a
//! proof that the bound state/model/evidence is physically complete or correct.

use allostatic_control_bench::certificate_lease::{
    CertificateContext, CertificateLease, CertificateRejection,
};
use allostatic_control_bench::multistep_viability::{
    evaluate_horizon_action_set, HorizonViableActionSet,
};
use allostatic_control_bench::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;

const RAW_HEADER: &str = "case,issue_succeeded,current_context_valid,action,expected_status,observed_status,accepted,expires_at_step,current_step,issued_horizon,requested_horizon";
const SUMMARY_HEADER: &str = "case,issue_succeeded,expected_status,observed_status,accepted,match";

fn interval(lower: f64, upper: f64) -> Interval {
    Interval::new(lower, upper).expect("valid interval")
}

fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
    ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid effects")
}

fn context(
    step: u64,
    state_revision: u64,
    dynamics_revision: u64,
    limits_revision: u64,
    calibration_revision: u64,
    authority_revision: u64,
    monitor_epoch: u64,
    horizon: usize,
    evidence_digest: [u8; 32],
) -> Option<CertificateContext> {
    CertificateContext::new(
        step,
        state_revision,
        dynamics_revision,
        limits_revision,
        calibration_revision,
        authority_revision,
        monitor_epoch,
        horizon,
        evidence_digest,
    )
}

fn balanced_action_set() -> HorizonViableActionSet {
    let state = StateEnvelope::new(interval(0.50, 0.50), interval(0.50, 0.50))
        .expect("valid state");
    let dynamics = DynamicsBounds::new(
        effects((-0.02, -0.01), (0.01, 0.02)),
        effects((0.02, 0.03), (-0.02, -0.01)),
        effects((0.08, 0.10), (-0.10, -0.08)),
    );
    evaluate_horizon_action_set(
        state,
        dynamics,
        ViabilityLimits::new(0.15, 0.88).expect("valid limits"),
        true,
        3,
    )
    .expect("valid horizon evaluation")
}

fn no_push_horizon_set() -> HorizonViableActionSet {
    let state = StateEnvelope::new(interval(0.16, 0.16), interval(0.87, 0.87))
        .expect("valid state");
    let dynamics = DynamicsBounds::new(
        effects((-0.006, -0.005), (0.01, 0.01)),
        effects((0.0, 0.0), (0.005, 0.005)),
        effects((-0.02, -0.02), (-0.30, -0.25)),
    );
    evaluate_horizon_action_set(
        state,
        dynamics,
        ViabilityLimits::new(0.15, 0.88).expect("valid limits"),
        true,
        2,
    )
    .expect("valid horizon evaluation")
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::Push => "push",
        Action::Maintain => "maintain",
        Action::EmergencyRecover => "emergency_recover",
    }
}

fn bool_text(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

struct Case {
    name: &'static str,
    current: Option<CertificateContext>,
    action: Action,
    issue_context: CertificateContext,
    action_set: HorizonViableActionSet,
    expected_status: &'static str,
}

fn cases() -> Vec<Case> {
    let base = context(100, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32])
        .expect("base context");
    let set = balanced_action_set();
    vec![
        Case {
            name: "valid_within_age",
            current: context(101, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "accepted",
        },
        Case {
            name: "state_revision_changed",
            current: context(100, 8, 11, 13, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "state_revision_changed",
        },
        Case {
            name: "dynamics_revision_changed",
            current: context(100, 7, 12, 13, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "dynamics_revision_changed",
        },
        Case {
            name: "limits_revision_changed",
            current: context(100, 7, 11, 14, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "limits_revision_changed",
        },
        Case {
            name: "calibration_revision_changed",
            current: context(100, 7, 11, 13, 18, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "calibration_revision_changed",
        },
        Case {
            name: "authority_revision_changed",
            current: context(100, 7, 11, 13, 17, 20, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "authority_revision_changed",
        },
        Case {
            name: "monitor_epoch_changed",
            current: context(100, 7, 11, 13, 17, 19, 24, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "monitor_epoch_changed",
        },
        Case {
            name: "supporting_evidence_changed",
            current: context(100, 7, 11, 13, 17, 19, 23, 3, [0x2b; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "supporting_evidence_changed",
        },
        Case {
            name: "lease_expired",
            current: context(102, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "expired",
        },
        Case {
            name: "clock_regressed",
            current: context(99, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "clock_regressed",
        },
        Case {
            name: "longer_horizon_requested",
            current: context(100, 7, 11, 13, 17, 19, 23, 4, [0x2a; 32]),
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "horizon_insufficient",
        },
        Case {
            name: "uncertified_action_not_issued",
            current: Some(base),
            action: Action::Push,
            issue_context: context(100, 7, 11, 13, 17, 19, 23, 2, [0x2a; 32])
                .expect("valid short context"),
            action_set: no_push_horizon_set(),
            expected_status: "issue_rejected",
        },
        Case {
            name: "zero_digest_context_rejected",
            current: None,
            action: Action::Maintain,
            issue_context: base,
            action_set: set,
            expected_status: "invalid_context",
        },
    ]
}

fn status_for(case: &Case) -> (bool, bool, &'static str, u64) {
    let Some(current) = case.current else {
        return (false, false, "invalid_context", 0);
    };
    let lease = CertificateLease::issue(case.action, case.issue_context, case.action_set, 1);
    let Some(lease) = lease else {
        return (false, true, "issue_rejected", 0);
    };
    let status = match lease.validate(current) {
        Ok(()) => "accepted",
        Err(reason) => reason.as_str(),
    };
    (true, true, status, lease.expires_at_step())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-certificate-lease-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let mut raw = String::from(RAW_HEADER);
    raw.push('\n');
    let mut summary = String::from(SUMMARY_HEADER);
    summary.push('\n');

    for case in cases() {
        let (issue_succeeded, current_valid, observed_status, expiry) = status_for(&case);
        let (current_step, requested_horizon) = case
            .current
            .map(|context| (context.step(), context.horizon()))
            .unwrap_or((100, 3));
        let accepted = observed_status == "accepted";
        let matches = observed_status == case.expected_status;
        raw.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            case.name,
            bool_text(issue_succeeded),
            bool_text(current_valid),
            action_name(case.action),
            case.expected_status,
            observed_status,
            bool_text(accepted),
            expiry,
            current_step,
            case.issue_context.horizon(),
            requested_horizon,
        ));
        summary.push_str(&format!(
            "{},{},{},{},{},{}\n",
            case.name,
            bool_text(issue_succeeded),
            case.expected_status,
            observed_status,
            bool_text(accepted),
            bool_text(matches),
        ));
    }

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    println!("Certificate lease raw results: {}", args[1]);
    println!("Certificate lease summary: {}", args[2]);
    println!("Cases: {}", cases().len());
    Ok(())
}
