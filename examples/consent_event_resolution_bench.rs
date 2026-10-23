// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic property-oriented probe for scoped consent event resolution.
//!
//! Run with: cargo run --example consent_event_resolution_bench
//!
//! This probe checks resolver contracts on synthetic fixtures. It does not
//! evaluate natural-language extraction, identity authentication, provenance,
//! trusted chronology, policy integration, or production dispatch. Fixture
//! success is not a safety certification or a state-of-the-art comparison.

use std::time::Instant;
use symthaea::hdc::moral_algebra::{
    resolve_latest_consent_candidate, ConsentEvidenceEvent, ConsentEvidenceResolution,
    ConsentEvidenceScope, ConsentEventOrder, ConsentModality, ConsentResolutionUncertainty,
    ConsentScope, ConsentTransition,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum ResolutionKey {
    NoRelevantEvidence,
    Unknown(ConsentResolutionUncertainty),
    Candidate(ConsentTransition, ConsentEventOrder),
}

impl From<ConsentEvidenceResolution> for ResolutionKey {
    fn from(value: ConsentEvidenceResolution) -> Self {
        match value {
            ConsentEvidenceResolution::NoRelevantEvidence => Self::NoRelevantEvidence,
            ConsentEvidenceResolution::Unknown(reason) => Self::Unknown(reason),
            ConsentEvidenceResolution::LatestCandidate {
                transition, order, ..
            } => Self::Candidate(transition, order),
        }
    }
}

#[derive(Debug)]
struct Fixture {
    id: &'static str,
    events: Vec<ConsentEvidenceEvent>,
    expected: ResolutionKey,
}

fn request() -> ConsentScope {
    ConsentScope::new(
        "person:alice",
        "action:share-health",
        "purpose:care",
    )
}

fn event(
    scope: &ConsentScope,
    transition: ConsentTransition,
    modality: ConsentModality,
    order: Option<u64>,
) -> ConsentEvidenceEvent {
    ConsentEvidenceEvent::new(
        ConsentEvidenceScope::exact(scope),
        Some("person:alice".to_owned()),
        transition,
        modality,
        order,
        None,
    )
}

fn resolve(events: &[ConsentEvidenceEvent], scope: &ConsentScope) -> ResolutionKey {
    resolve_latest_consent_candidate(events, scope).into()
}

fn permutations(events: &[ConsentEvidenceEvent]) -> Vec<Vec<ConsentEvidenceEvent>> {
    fn visit(
        remaining: &mut Vec<ConsentEvidenceEvent>,
        prefix: &mut Vec<ConsentEvidenceEvent>,
        output: &mut Vec<Vec<ConsentEvidenceEvent>>,
    ) {
        if remaining.is_empty() {
            output.push(prefix.clone());
            return;
        }

        for index in 0..remaining.len() {
            let event = remaining.remove(index);
            prefix.push(event.clone());
            visit(remaining, prefix, output);
            prefix.pop();
            remaining.insert(index, event);
        }
    }

    let mut output = Vec::new();
    visit(&mut events.to_vec(), &mut Vec::new(), &mut output);
    output
}

fn fixtures(scope: &ConsentScope) -> Vec<Fixture> {
    let mut bob = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(1),
    );
    bob.scope.subject_id = Some("person:bob".to_owned());

    let mut wrong_action = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(1),
    );
    wrong_action.scope.action_id = Some("action:publish".to_owned());

    let mut wrong_purpose = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(1),
    );
    wrong_purpose.scope.purpose_id = Some("purpose:advertising".to_owned());

    let mut partial_scope = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(1),
    );
    partial_scope.scope.action_id = None;

    let mut known_mismatch_and_missing = bob.clone();
    known_mismatch_and_missing.scope.action_id = None;

    vec![
        Fixture {
            id: "no-evidence",
            events: vec![],
            expected: ResolutionKey::NoRelevantEvidence,
        },
        Fixture {
            id: "single-grant",
            events: vec![event(
                scope,
                ConsentTransition::Granted,
                ConsentModality::Asserted,
                Some(10),
            )],
            expected: ResolutionKey::Candidate(ConsentTransition::Granted, 10),
        },
        Fixture {
            id: "later-withdrawal-wins-not-input-order",
            events: vec![
                event(
                    scope,
                    ConsentTransition::Withdrawn,
                    ConsentModality::Asserted,
                    Some(20),
                ),
                event(
                    scope,
                    ConsentTransition::Granted,
                    ConsentModality::Asserted,
                    Some(10),
                ),
            ],
            expected: ResolutionKey::Candidate(ConsentTransition::Withdrawn, 20),
        },
        Fixture {
            id: "wrong-subject-does-not-transfer",
            events: vec![bob],
            expected: ResolutionKey::NoRelevantEvidence,
        },
        Fixture {
            id: "wrong-action-does-not-transfer",
            events: vec![wrong_action],
            expected: ResolutionKey::NoRelevantEvidence,
        },
        Fixture {
            id: "wrong-purpose-does-not-transfer",
            events: vec![wrong_purpose],
            expected: ResolutionKey::NoRelevantEvidence,
        },
        Fixture {
            id: "partial-scope-is-unknown",
            events: vec![partial_scope],
            expected: ResolutionKey::Unknown(
                ConsentResolutionUncertainty::AmbiguousEventScope,
            ),
        },
        Fixture {
            id: "quoted-claim-is-not-asserted-grant",
            events: vec![event(
                scope,
                ConsentTransition::Granted,
                ConsentModality::Quoted,
                Some(10),
            )],
            expected: ResolutionKey::Unknown(
                ConsentResolutionUncertainty::NonAssertedEvidence,
            ),
        },
        Fixture {
            id: "missing-order-is-unknown",
            events: vec![event(
                scope,
                ConsentTransition::Granted,
                ConsentModality::Asserted,
                None,
            )],
            expected: ResolutionKey::Unknown(
                ConsentResolutionUncertainty::MissingTemporalOrder,
            ),
        },
        Fixture {
            id: "duplicate-order-is-unknown",
            events: vec![
                event(
                    scope,
                    ConsentTransition::Granted,
                    ConsentModality::Asserted,
                    Some(10),
                ),
                event(
                    scope,
                    ConsentTransition::Revoked,
                    ConsentModality::Asserted,
                    Some(10),
                ),
            ],
            expected: ResolutionKey::Unknown(
                ConsentResolutionUncertainty::DuplicateTemporalOrder,
            ),
        },
        Fixture {
            id: "known-mismatch-excludes-even-with-missing-field",
            events: vec![known_mismatch_and_missing],
            expected: ResolutionKey::NoRelevantEvidence,
        },
        Fixture {
            id: "latest-revocation-wins",
            events: vec![
                event(
                    scope,
                    ConsentTransition::Granted,
                    ConsentModality::Asserted,
                    Some(1),
                ),
                event(
                    scope,
                    ConsentTransition::Revoked,
                    ConsentModality::Asserted,
                    Some(2),
                ),
            ],
            expected: ResolutionKey::Candidate(ConsentTransition::Revoked, 2),
        },
    ]
}

fn counterfactual_pairs_pass(scope: &ConsentScope) -> (usize, usize) {
    let grant = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(1),
    );
    let base = resolve(std::slice::from_ref(&grant), scope);

    let withdrawal = event(
        scope,
        ConsentTransition::Withdrawn,
        ConsentModality::Asserted,
        Some(2),
    );
    let with_withdrawal = vec![grant.clone(), withdrawal];
    let withdrawal_result = resolve(&with_withdrawal, scope);

    let mut different_purpose = grant.clone();
    different_purpose.scope.purpose_id = Some("purpose:advertising".to_owned());
    let wrong_scope_result = resolve(&[different_purpose], scope);

    let mut quoted = grant.clone();
    quoted.modality = ConsentModality::Quoted;
    let quote_result = resolve(&[quoted], scope);

    let mut no_order = grant.clone();
    no_order.order = None;
    let order_result = resolve(&[no_order], scope);

    let pairs = [
        (
            base == ResolutionKey::Candidate(ConsentTransition::Granted, 1)
                && withdrawal_result
                    == ResolutionKey::Candidate(ConsentTransition::Withdrawn, 2),
            "add-later-withdrawal-changes-candidate",
        ),
        (
            wrong_scope_result == ResolutionKey::NoRelevantEvidence,
            "change-purpose-removes-scope-match",
        ),
        (
            quote_result
                == ResolutionKey::Unknown(ConsentResolutionUncertainty::NonAssertedEvidence),
            "asserted-to-quoted-changes-to-unknown",
        ),
        (
            order_result
                == ResolutionKey::Unknown(ConsentResolutionUncertainty::MissingTemporalOrder),
            "remove-order-changes-to-unknown",
        ),
    ];

    for (passed, id) in pairs {
        println!("counterfactual,{id},{}", if passed { "PASS" } else { "FAIL" });
    }
    (pairs.iter().filter(|(passed, _)| *passed).count(), pairs.len())
}

fn main() {
    let started = Instant::now();
    let scope = request();
    let fixtures = fixtures(&scope);
    let mut fixture_passes = 0usize;
    let mut permutation_cases = 0usize;
    let mut permutation_trials = 0usize;
    let mut permutation_failures = 0usize;

    println!("consent_event_resolution_bench=v1");
    println!("scope=synthetic; authorization_claim=false; natural_language_eval=false");
    for fixture in &fixtures {
        let actual = resolve(&fixture.events, &scope);
        let fixture_pass = actual == fixture.expected;
        if fixture_pass {
            fixture_passes += 1;
        }
        println!(
            "fixture,{},expected={:?},actual={:?},{}",
            fixture.id,
            fixture.expected,
            actual,
            if fixture_pass { "PASS" } else { "FAIL" }
        );

        let variants = permutations(&fixture.events);
        let canonical = actual.clone();
        let mut invariant = true;
        for variant in &variants {
            permutation_trials += 1;
            if resolve(variant, &scope) != canonical {
                invariant = false;
                permutation_failures += 1;
            }
        }
        if invariant {
            permutation_cases += 1;
        } else {
            println!("permutation_invariance,{},FAIL", fixture.id);
        }
    }

    let (counterfactual_passes, counterfactual_total) = counterfactual_pairs_pass(&scope);
    let fixture_total = fixtures.len();
    println!(
        "summary,fixtures={fixture_passes}/{fixture_total},permutation_invariant_fixtures={permutation_cases}/{fixture_total},permutation_trials={permutation_trials},permutation_failures={permutation_failures},counterfactual_pairs={counterfactual_passes}/{counterfactual_total},elapsed_us={}",
        started.elapsed().as_micros()
    );

    if fixture_passes != fixture_total
        || permutation_failures != 0
        || permutation_cases != fixture_total
        || counterfactual_passes != counterfactual_total
    {
        std::process::exit(1);
    }
}
