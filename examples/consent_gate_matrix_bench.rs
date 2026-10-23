// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exhaustive deterministic matrix for the consent-specific gate and its
//! point-of-use wrapper. This is a development invariant probe, not a security
//! certification or a substitute for real dispatcher integration.
//!
//! Run with: cargo run --example consent_gate_matrix_bench
//!
//! The expected outcome is defined independently from the gate output. The
//! matrix crosses requirement state, consent state, trust flags, revocation,
//! evidence presence, request/evidence scope mutations, and validity windows.
//! It deliberately uses a toy fixed clock and synthetic records.

use symthaea::hdc::moral_algebra::{
    ConsentEvidence, ConsentGateOutcome, ConsentRequest, ConsentRequirement, ConsentState,
    evaluate_consent_gate, execute_after_consent_check,
};

const NOW: u64 = 1_800_000_000;

#[derive(Default)]
struct Metrics {
    cases: usize,
    expected_execute: usize,
    expected_block: usize,
    false_permits: usize,
    false_blocks: usize,
    closure_invocations_on_block: usize,
    closure_not_invoked_on_allow: usize,
    outcome_mismatches: usize,
}

fn make_evidence(
    state: ConsentState,
    trust_mask: u8,
    revoked: bool,
    evidence_scope_case: usize,
    time_case: usize,
) -> ConsentEvidence<'static> {
    let mut evidence = ConsentEvidence {
        state,
        subject_id: "subject:alice",
        action_id: "action:export-record",
        purpose_id: "purpose:care-coordination",
        record_id: "record:consent-17",
        record_verified: trust_mask & 0b001 != 0,
        identity_verified: trust_mask & 0b010 != 0,
        authority_verified: trust_mask & 0b100 != 0,
        valid_from_unix_s: NOW - 60,
        expires_at_unix_s: Some(NOW + 60),
        revoked,
    };

    match evidence_scope_case {
        0 => {}
        1 => evidence.subject_id = "subject:bob",
        2 => evidence.action_id = "action:publish-record",
        3 => evidence.purpose_id = "purpose:advertising",
        4 => evidence.record_id = "",
        _ => unreachable!("evidence scope case is bounded by the matrix"),
    }

    match time_case {
        0 => {}
        1 => {
            evidence.valid_from_unix_s = NOW + 1;
            evidence.expires_at_unix_s = Some(NOW + 60);
        }
        2 => evidence.valid_from_unix_s = NOW - 60,
        3 => {
            evidence.valid_from_unix_s = NOW;
            evidence.expires_at_unix_s = Some(NOW);
        }
        _ => unreachable!("time case is bounded by the matrix"),
    }

    if time_case == 2 {
        evidence.expires_at_unix_s = Some(NOW);
    }

    evidence
}

fn make_request(request_scope_case: usize) -> ConsentRequest<'static> {
    let mut request = ConsentRequest {
        subject_id: "subject:alice",
        action_id: "action:export-record",
        purpose_id: "purpose:care-coordination",
    };
    match request_scope_case {
        0 => {}
        1 => request.subject_id = "subject:request-bob",
        2 => request.action_id = "action:request-publish-record",
        3 => request.purpose_id = "purpose:request-advertising",
        4 => request.action_id = "",
        _ => unreachable!("request scope case is bounded by the matrix"),
    }
    request
}

fn expected_consent_outcome(
    requirement: ConsentRequirement,
    state: ConsentState,
    trust_mask: u8,
    revoked: bool,
    evidence_present: bool,
    request_scope_case: usize,
    evidence_scope_case: usize,
    time_case: usize,
) -> ConsentGateOutcome {
    match requirement {
        ConsentRequirement::NotRequired => ConsentGateOutcome::NotRequired,
        ConsentRequirement::Unresolved => {
            ConsentGateOutcome::Blocked(
                symthaea::hdc::moral_algebra::ConsentBlockReason::RequirementUnresolved,
            )
        }
        ConsentRequirement::Required => {
            if !evidence_present {
                return ConsentGateOutcome::Blocked(
                    symthaea::hdc::moral_algebra::ConsentBlockReason::MissingEvidence,
                );
            }
            // A Required request can satisfy the consent precondition only if
            // every independent condition is valid. For invalid cases the exact
            // blocking reason is less important than the invariant that it blocks.
            if state == ConsentState::Given
                && trust_mask == 0b111
                && !revoked
                && request_scope_case == 0
                && evidence_scope_case == 0
                && time_case == 0
            {
                ConsentGateOutcome::Satisfied
            } else {
                ConsentGateOutcome::Blocked(
                    symthaea::hdc::moral_algebra::ConsentBlockReason::UnknownState,
                )
            }
        }
    }
}

fn main() {
    let requirements = [
        ConsentRequirement::NotRequired,
        ConsentRequirement::Required,
        ConsentRequirement::Unresolved,
    ];
    let states = [
        ConsentState::Given,
        ConsentState::Denied,
        ConsentState::Absent,
        ConsentState::Implied,
        ConsentState::Unknown,
    ];

    let mut metrics = Metrics::default();

    for requirement in requirements {
        for state in states {
            for trust_mask in 0_u8..8 {
                for revoked in [false, true] {
                    for evidence_present in [false, true] {
                        for request_scope_case in 0_usize..5 {
                            for evidence_scope_case in 0_usize..5 {
                                for time_case in 0_usize..4 {
                                    metrics.cases += 1;
                                    let request = make_request(request_scope_case);
                                    let evidence = make_evidence(
                                        state,
                                        trust_mask,
                                        revoked,
                                        evidence_scope_case,
                                        time_case,
                                    );
                                    let evidence_ref = evidence_present.then_some(&evidence);
                                    let expected = expected_consent_outcome(
                                        requirement,
                                        state,
                                        trust_mask,
                                        revoked,
                                        evidence_present,
                                        request_scope_case,
                                        evidence_scope_case,
                                        time_case,
                                    );

                                    let observed = evaluate_consent_gate(
                                        requirement,
                                        request,
                                        evidence_ref,
                                        NOW,
                                    );

                                    if matches!(expected, ConsentGateOutcome::Blocked(_)) {
                                        metrics.expected_block += 1;
                                        if !matches!(observed, ConsentGateOutcome::Blocked(_)) {
                                            metrics.false_permits += 1;
                                        }
                                    } else {
                                        metrics.expected_execute += 1;
                                        if observed != expected {
                                            metrics.outcome_mismatches += 1;
                                        }
                                    }

                                    let should_execute =
                                        !matches!(expected, ConsentGateOutcome::Blocked(_));
                                    let mut invocations = 0_u32;
                                    let wrapped = execute_after_consent_check(
                                        requirement,
                                        request,
                                        evidence_ref,
                                        NOW,
                                        || {
                                            invocations += 1;
                                        },
                                    );

                                    if should_execute && invocations != 1 {
                                        metrics.closure_not_invoked_on_allow += 1;
                                    }
                                    if !should_execute && invocations != 0 {
                                        metrics.closure_invocations_on_block += 1;
                                    }

                                    if should_execute {
                                        if let Ok(((), outcome)) = wrapped {
                                            if outcome != expected {
                                                metrics.outcome_mismatches += 1;
                                            }
                                        } else {
                                            metrics.outcome_mismatches += 1;
                                        }
                                    } else if wrapped.is_ok() {
                                        metrics.false_permits += 1;
                                    }

                                    // Also catch unexpected blocks for a valid Required
                                    // case, independently of the expected-output branch.
                                    if requirement == ConsentRequirement::Required
                                        && evidence_present
                                        && state == ConsentState::Given
                                        && trust_mask == 0b111
                                        && !revoked
                                        && request_scope_case == 0
                                        && evidence_scope_case == 0
                                        && time_case == 0
                                        && observed != ConsentGateOutcome::Satisfied
                                    {
                                        metrics.false_blocks += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Consent point-of-use gate matrix");
    println!("cases={}", metrics.cases);
    println!("expected_block={}", metrics.expected_block);
    println!("expected_execute={}", metrics.expected_execute);
    println!("false_permits={}", metrics.false_permits);
    println!("false_blocks={}", metrics.false_blocks);
    println!(
        "closure_invocations_on_block={}",
        metrics.closure_invocations_on_block
    );
    println!(
        "closure_not_invoked_on_allow={}",
        metrics.closure_not_invoked_on_allow
    );
    println!("outcome_mismatches={}", metrics.outcome_mismatches);
    println!("scope: synthetic fixed-clock consent precondition only; not end-to-end authorization");

    if metrics.false_permits > 0
        || metrics.false_blocks > 0
        || metrics.closure_invocations_on_block > 0
        || metrics.closure_not_invoked_on_allow > 0
        || metrics.outcome_mismatches > 0
    {
        eprintln!("FAIL: consent gate matrix violated an expected invariant");
        std::process::exit(1);
    }
    println!("PASS: deterministic consent-gate invariants held for this matrix");
}
