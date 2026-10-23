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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldCase {
    Exact,
    Mismatch,
    Missing,
    Empty,
}

const FIELD_CASES: [FieldCase; 4] = [
    FieldCase::Exact,
    FieldCase::Mismatch,
    FieldCase::Missing,
    FieldCase::Empty,
];
const TRANSITIONS: [ConsentTransition; 4] = [
    ConsentTransition::Granted,
    ConsentTransition::Refused,
    ConsentTransition::Withdrawn,
    ConsentTransition::Revoked,
];
const MODALITIES: [ConsentModality; 6] = [
    ConsentModality::Asserted,
    ConsentModality::Asked,
    ConsentModality::Quoted,
    ConsentModality::Reported,
    ConsentModality::Conditional,
    ConsentModality::Hypothetical,
];
const ORDERS: [Option<u64>; 3] = [None, Some(0), Some(1)];

fn mutate_field(
    field: &mut Option<String>,
    case: FieldCase,
    expected: &str,
    mismatch: &str,
) {
    *field = match case {
        FieldCase::Exact => Some(expected.to_owned()),
        FieldCase::Mismatch => Some(mismatch.to_owned()),
        FieldCase::Missing => None,
        FieldCase::Empty => Some(String::new()),
    };
}

fn matrix_event(
    scope: &ConsentScope,
    subject: FieldCase,
    action: FieldCase,
    purpose: FieldCase,
    transition: ConsentTransition,
    modality: ConsentModality,
    order: Option<u64>,
) -> ConsentEvidenceEvent {
    let mut candidate = event(scope, transition, modality, order);
    mutate_field(
        &mut candidate.scope.subject_id,
        subject,
        &scope.subject_id,
        "person:other",
    );
    mutate_field(
        &mut candidate.scope.action_id,
        action,
        &scope.action_id,
        "action:other",
    );
    mutate_field(
        &mut candidate.scope.purpose_id,
        purpose,
        &scope.purpose_id,
        "purpose:other",
    );
    candidate
}

/// Independent, high-level oracle for a single synthetic event. This is not
/// the implementation itself: it classifies mutation cases before inspecting
/// the event and never invokes resolver helpers.
fn expected_single(
    event: &ConsentEvidenceEvent,
    fields: [FieldCase; 3],
) -> ResolutionKey {
    if fields.contains(&FieldCase::Mismatch) {
        return ResolutionKey::NoRelevantEvidence;
    }
    if fields
        .iter()
        .any(|case| matches!(case, FieldCase::Missing | FieldCase::Empty))
    {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::AmbiguousEventScope);
    }
    if event.modality != ConsentModality::Asserted {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::NonAssertedEvidence);
    }
    match event.order {
        None => ResolutionKey::Unknown(ConsentResolutionUncertainty::MissingTemporalOrder),
        Some(order) => ResolutionKey::Candidate(event.transition, order),
    }
}

fn expected_exact_pair(events: &[ConsentEvidenceEvent; 2]) -> ResolutionKey {
    if events.iter().any(|event| event.modality != ConsentModality::Asserted) {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::NonAssertedEvidence);
    }
    let [Some(left), Some(right)] = [events[0].order, events[1].order] else {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::MissingTemporalOrder);
    };
    if left == right {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::DuplicateTemporalOrder);
    }
    let latest = if left > right { &events[0] } else { &events[1] };
    ResolutionKey::Candidate(latest.transition, latest.order.unwrap_or_default())
}

fn exhaustive_single_event_matrix(scope: &ConsentScope) -> (usize, usize) {
    let mut passes = 0usize;
    let mut total = 0usize;
    for subject in FIELD_CASES {
        for action in FIELD_CASES {
            for purpose in FIELD_CASES {
                let fields = [subject, action, purpose];
                for transition in TRANSITIONS {
                    for modality in MODALITIES {
                        for order in ORDERS {
                            let candidate = matrix_event(
                                scope, subject, action, purpose, transition, modality, order,
                            );
                            let expected = expected_single(&candidate, fields);
                            let actual = resolve(std::slice::from_ref(&candidate), scope);
                            total += 1;
                            if actual == expected {
                                passes += 1;
                            } else if total <= 20 {
                                println!(
                                    "single_matrix_mismatch,case={total},fields={fields:?},transition={transition:?},modality={modality:?},order={order:?},expected={expected:?},actual={actual:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "single_event_matrix,passes={passes},total={total},scope_mutation_combinations=64"
    );
    (passes, total)
}

fn exhaustive_exact_pair_matrix(scope: &ConsentScope) -> (usize, usize, usize) {
    let mut candidates = Vec::new();
    for transition in TRANSITIONS {
        for modality in MODALITIES {
            for order in ORDERS {
                candidates.push(event(scope, transition, modality, order));
            }
        }
    }

    let mut expected_passes = 0usize;
    let mut permutation_passes = 0usize;
    let mut total = 0usize;
    for left in &candidates {
        for right in &candidates {
            let pair = [left.clone(), right.clone()];
            let expected = expected_exact_pair(&pair);
            let actual = resolve(&pair, scope);
            total += 1;
            if actual == expected {
                expected_passes += 1;
            } else if total <= 20 {
                println!(
                    "pair_matrix_mismatch,case={total},expected={expected:?},actual={actual:?}"
                );
            }

            let reversed = [right.clone(), left.clone()];
            if resolve(&reversed, scope) == actual {
                permutation_passes += 1;
            } else if total <= 20 {
                println!("pair_matrix_permutation_mismatch,case={total}");
            }
        }
    }
    println!(
        "exact_scope_pair_matrix,oracle_passes={expected_passes}/{total},permutation_passes={permutation_passes}/{total}"
    );
    (expected_passes, permutation_passes, total)
}

fn expected_mixed_scope_pair(
    candidate: &ConsentEvidenceEvent,
    fields: [FieldCase; 3],
) -> ResolutionKey {
    // A known mismatch means the event is out of scope even if another scope
    // field is malformed. The exact in-scope baseline event remains applicable.
    if fields.contains(&FieldCase::Mismatch) {
        return ResolutionKey::Candidate(ConsentTransition::Granted, 0);
    }
    if fields
        .iter()
        .any(|case| matches!(case, FieldCase::Missing | FieldCase::Empty))
    {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::AmbiguousEventScope);
    }
    if candidate.modality != ConsentModality::Asserted {
        return ResolutionKey::Unknown(ConsentResolutionUncertainty::NonAssertedEvidence);
    }
    match candidate.order {
        None => ResolutionKey::Unknown(ConsentResolutionUncertainty::MissingTemporalOrder),
        Some(0) => ResolutionKey::Unknown(ConsentResolutionUncertainty::DuplicateTemporalOrder),
        Some(1) => ResolutionKey::Candidate(candidate.transition, 1),
        Some(_) => unreachable!("matrix order is limited to 0 and 1"),
    }
}

fn exhaustive_mixed_scope_pair_matrix(scope: &ConsentScope) -> (usize, usize, usize) {
    let baseline = event(
        scope,
        ConsentTransition::Granted,
        ConsentModality::Asserted,
        Some(0),
    );
    let mut oracle_passes = 0usize;
    let mut permutation_passes = 0usize;
    let mut total = 0usize;

    for subject in FIELD_CASES {
        for action in FIELD_CASES {
            for purpose in FIELD_CASES {
                let fields = [subject, action, purpose];
                for transition in TRANSITIONS {
                    for modality in MODALITIES {
                        for order in ORDERS {
                            let candidate = matrix_event(
                                scope, subject, action, purpose, transition, modality, order,
                            );
                            let expected = expected_mixed_scope_pair(&candidate, fields);
                            let pair = [candidate.clone(), baseline.clone()];
                            let actual = resolve(&pair, scope);
                            total += 1;
                            if actual == expected {
                                oracle_passes += 1;
                            } else if total <= 30 {
                                println!(
                                    "mixed_scope_pair_mismatch,case={total},fields={fields:?},transition={transition:?},modality={modality:?},order={order:?},expected={expected:?},actual={actual:?}"
                                );
                            }

                            let reversed = [baseline.clone(), candidate];
                            if resolve(&reversed, scope) == actual {
                                permutation_passes += 1;
                            } else if total <= 30 {
                                println!(
                                    "mixed_scope_pair_permutation_mismatch,case={total},fields={fields:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "mixed_scope_pair_matrix,oracle_passes={oracle_passes}/{total},permutation_passes={permutation_passes}/{total}"
    );
    (oracle_passes, permutation_passes, total)
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
    let (single_passes, single_total) = exhaustive_single_event_matrix(&scope);
    let (pair_passes, pair_permutation_passes, pair_total) =
        exhaustive_exact_pair_matrix(&scope);
    let (mixed_passes, mixed_permutation_passes, mixed_total) =
        exhaustive_mixed_scope_pair_matrix(&scope);
    let fixture_total = fixtures.len();
    println!(
        "summary,fixtures={fixture_passes}/{fixture_total},permutation_invariant_fixtures={permutation_cases}/{fixture_total},permutation_trials={permutation_trials},permutation_failures={permutation_failures},counterfactual_pairs={counterfactual_passes}/{counterfactual_total},single_event_matrix={single_passes}/{single_total},exact_pair_matrix={pair_passes}/{pair_total},exact_pair_permutations={pair_permutation_passes}/{pair_total},mixed_scope_pair_matrix={mixed_passes}/{mixed_total},mixed_scope_pair_permutations={mixed_permutation_passes}/{mixed_total},elapsed_us={}",
        started.elapsed().as_micros()
    );

    if fixture_passes != fixture_total
        || permutation_failures != 0
        || permutation_cases != fixture_total
        || counterfactual_passes != counterfactual_total
        || single_passes != single_total
        || pair_passes != pair_total
        || pair_permutation_passes != pair_total
        || mixed_passes != mixed_total
        || mixed_permutation_passes != mixed_total
    {
        std::process::exit(1);
    }
}
