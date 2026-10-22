//! Independent verification contract suite for multi-step action certificates.
//!
//! Usage: allostatic-independent-verifier-bench <raw.csv> <summary.csv>
//! Tests the independent checker against valid, falsified, stale, and tampered
//! witnesses. Passing proves only these software contracts, not physical safety.

use allostatic_control_bench::certificate_lease::{
    CertificateContext, CertificateLease, CertificateRejection,
};
use allostatic_control_bench::independent_verifier::{
    verify_leased_witness, verify_sequence, LeasedVerificationFailure,
};
use allostatic_control_bench::multistep_viability::evaluate_horizon_action_set;
use allostatic_control_bench::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;

const RAW_HEADER: &str = "case,expected,observed,accepted,steps_verified,first_action,final_reserve_lower,final_load_upper";
const SUMMARY_HEADER: &str = "case,expected,observed,match";

fn interval(lower: f64, upper: f64) -> Interval {
    Interval::new(lower, upper).expect("valid interval")
}

fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
    ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid effects")
}

fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
    StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid state")
}

fn limits() -> ViabilityLimits {
    ViabilityLimits::new(0.15, 0.88).expect("valid limits")
}

fn nominal_dynamics() -> DynamicsBounds {
    DynamicsBounds::new(
        effects((-0.02, -0.01), (0.01, 0.02)),
        effects((0.02, 0.03), (-0.02, -0.01)),
        effects((0.08, 0.10), (-0.10, -0.08)),
    )
}

fn certificate_context(
    step: u64,
    dynamics_revision: u64,
    evidence_digest: [u8; 32],
) -> CertificateContext {
    CertificateContext::new(step, 7, dynamics_revision, 13, 17, 19, 23, 3, evidence_digest)
        .expect("valid context")
}

fn safe_lease(
    action: Action,
    current_state: StateEnvelope,
    model: DynamicsBounds,
    context: CertificateContext,
) -> CertificateLease {
    let actions = evaluate_horizon_action_set(current_state, model, limits(), true, 3)
        .expect("valid planner search");
    CertificateLease::issue(action, context, actions, 0).expect("action is planner-certified")
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::Push) => "push",
        Some(Action::Maintain) => "maintain",
        Some(Action::EmergencyRecover) => "emergency_recover",
        None => "none",
    }
}

fn render_result(
    case: &str,
    expected: &str,
    observed: &str,
    steps: usize,
    first_action: Option<Action>,
    final_reserve_lower: f64,
    final_load_upper: f64,
    raw: &mut String,
    summary: &mut String,
) {
    let accepted = observed == "accepted";
    raw.push_str(&format!(
        "{},{},{},{},{},{},{:.6},{:.6}\n",
        case,
        expected,
        observed,
        accepted,
        steps,
        action_name(first_action),
        final_reserve_lower,
        final_load_upper,
    ));
    summary.push_str(&format!("{},{},{},{}\n", case, expected, observed, expected == observed));
}

fn direct_case(
    case: &str,
    expected: &str,
    initial: StateEnvelope,
    actions: &[Action],
    model: DynamicsBounds,
    authority: bool,
    horizon: usize,
    raw: &mut String,
    summary: &mut String,
) {
    match verify_sequence(initial, actions, model, limits(), authority, horizon) {
        Ok(receipt) => render_result(
            case, expected, "accepted", receipt.steps_verified(),
            Some(receipt.first_action()), receipt.final_state().reserve().lower(),
            receipt.final_state().load().upper(), raw, summary,
        ),
        Err(reason) => render_result(
            case, expected, reason.as_str(), 0, actions.first().copied(),
            -1.0, -1.0, raw, summary,
        ),
    }
}

fn leased_case(
    case: &str,
    expected: &str,
    lease: CertificateLease,
    current_context: CertificateContext,
    initial: StateEnvelope,
    actions: &[Action],
    model: DynamicsBounds,
    authority: bool,
    raw: &mut String,
    summary: &mut String,
) {
    match verify_leased_witness(lease, current_context, initial, actions, model, limits(), authority) {
        Ok(receipt) => render_result(
            case, expected, "accepted", receipt.steps_verified(),
            Some(receipt.first_action()), receipt.final_state().reserve().lower(),
            receipt.final_state().load().upper(), raw, summary,
        ),
        Err(LeasedVerificationFailure::LeaseInvalid(CertificateRejection::DynamicsRevisionChanged)) => {
            render_result(case, expected, "dynamics_revision_changed", 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
        Err(LeasedVerificationFailure::LeaseInvalid(CertificateRejection::SupportingEvidenceChanged)) => {
            render_result(case, expected, "supporting_evidence_changed", 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
        Err(LeasedVerificationFailure::LeaseInvalid(other)) => {
            render_result(case, expected, other.as_str(), 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
        Err(LeasedVerificationFailure::WitnessActionMismatch) => {
            render_result(case, expected, "witness_action_mismatch", 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
        Err(LeasedVerificationFailure::WitnessHorizonMismatch) => {
            render_result(case, expected, "witness_horizon_mismatch", 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
        Err(LeasedVerificationFailure::WitnessRejected(reason)) => {
            render_result(case, expected, reason.as_str(), 0, actions.first().copied(), -1.0, -1.0, raw, summary)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-independent-verifier-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let balanced = state((0.50, 0.50), (0.50, 0.50));
    let boundary = state((0.16, 0.16), (0.87, 0.87));
    let model = nominal_dynamics();
    let mut raw = String::from(RAW_HEADER);
    raw.push('\n');
    let mut summary = String::from(SUMMARY_HEADER);
    summary.push('\n');

    direct_case("safe_full_horizon", "accepted", balanced,
        &[Action::Maintain, Action::Maintain, Action::Push], model, true, 3, &mut raw, &mut summary);

    let unsafe_model = DynamicsBounds::new(
        effects((-0.005, -0.005), (0.09, 0.10)),
        effects((0.02, 0.03), (0.03, 0.04)),
        effects((0.02, 0.03), (-0.30, -0.25)),
    );
    direct_case("unsafe_first_step", "load_constraint_violated", boundary,
        &[Action::Push], unsafe_model, true, 1, &mut raw, &mut summary);
    direct_case("unsafe_intermediate_then_recover", "load_constraint_violated", boundary,
        &[Action::Push, Action::EmergencyRecover], unsafe_model, true, 2, &mut raw, &mut summary);
    direct_case("authority_revoked", "recovery_authority_missing", balanced,
        &[Action::Maintain, Action::EmergencyRecover], model, false, 2, &mut raw, &mut summary);
    direct_case("horizon_mismatch", "horizon_mismatch", balanced,
        &[Action::Maintain, Action::Push], model, true, 3, &mut raw, &mut summary);
    direct_case("unsafe_initial_state", "reserve_constraint_violated", state((0.14, 0.18), (0.80, 0.82)),
        &[Action::EmergencyRecover], model, true, 1, &mut raw, &mut summary);

    let issue_context = certificate_context(100, 11, [0x2a; 32]);
    let lease = safe_lease(Action::Maintain, balanced, model, issue_context);
    leased_case("fresh_lease_and_witness", "accepted", lease,
        certificate_context(100, 11, [0x2a; 32]), balanced,
        &[Action::Maintain, Action::Maintain, Action::Push], model, true, &mut raw, &mut summary);
    leased_case("stale_dynamics_revision", "dynamics_revision_changed", lease,
        certificate_context(100, 12, [0x2a; 32]), balanced,
        &[Action::Maintain, Action::Maintain, Action::Push], model, true, &mut raw, &mut summary);
    leased_case("stale_evidence_digest", "supporting_evidence_changed", lease,
        certificate_context(100, 11, [0x2b; 32]), balanced,
        &[Action::Maintain, Action::Maintain, Action::Push], model, true, &mut raw, &mut summary);
    leased_case("tampered_first_action", "witness_action_mismatch", lease,
        certificate_context(100, 11, [0x2a; 32]), balanced,
        &[Action::Push, Action::Maintain, Action::Maintain], model, true, &mut raw, &mut summary);

    // A forged integration context cannot make an unsafe witness pass: the
    // independent transition checker re-evaluates the actual supplied model.
    let near_boundary = state((0.50, 0.50), (0.87, 0.87));
    let benign_push = DynamicsBounds::new(
        effects((0.0, 0.0), (0.002, 0.002)),
        effects((0.002, 0.002), (-0.002, -0.002)),
        effects((0.02, 0.02), (-0.10, -0.10)),
    );
    let adverse_push = DynamicsBounds::new(
        effects((0.0, 0.0), (0.02, 0.02)),
        effects((0.002, 0.002), (-0.002, -0.002)),
        effects((0.02, 0.02), (-0.10, -0.10)),
    );
    let near_ctx = certificate_context(100, 33, [0x33; 32]);
    let near_lease = safe_lease(Action::Push, near_boundary, benign_push, near_ctx);
    leased_case("actual_transition_disagrees_with_claimed_model", "load_constraint_violated", near_lease,
        near_ctx, near_boundary, &[Action::Push, Action::Maintain, Action::Maintain],
        adverse_push, true, &mut raw, &mut summary);

    let raw_cases = raw.lines().count() - 1;
    let summary_cases = summary.lines().count() - 1;
    if raw_cases != 11 || summary_cases != 11 {
        return Err(format!("expected 11 cases, got raw={raw_cases}, summary={summary_cases}").into());
    }
    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    println!("Independent verification raw results: {}", args[1]);
    println!("Independent verification summary: {}", args[2]);
    println!("Cases: {}", raw_cases);
    Ok(())
}
