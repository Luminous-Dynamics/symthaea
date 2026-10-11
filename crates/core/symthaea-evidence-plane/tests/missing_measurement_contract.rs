// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Expected-red regression for evidence-plane measurement presence.
//!
//! A negative experimental result requires an observed measurement whose value
//! is zero (or below a threshold). An absent counter is not evidence that the
//! mechanism was observed inactive.

use std::collections::{BTreeMap, HashMap};

use symthaea_evidence_plane::{
    EvidenceCounters, Expectation, RunEvidence, RunId, check_integrity,
};

#[test]
fn missing_must_be_zero_measurement_is_not_an_observed_zero() {
    let declared = HashMap::from([("target_hook".to_string(), Expectation::MustBeZero)]);
    let measured = EvidenceCounters::new();

    assert!(
        check_integrity(&declared, &measured).is_err(),
        "a counter that was never observed must not satisfy MustBeZero"
    );
}

#[test]
fn explicitly_observed_zero_still_satisfies_must_be_zero() {
    let declared = HashMap::from([("target_hook".to_string(), Expectation::MustBeZero)]);
    let mut measured = EvidenceCounters::new();
    measured.record("target_hook", 0.0);

    assert!(check_integrity(&declared, &measured).is_ok());
}

#[test]
fn misspelled_negative_probe_name_fails_instead_of_becoming_zero() {
    let declared = HashMap::from([(
        "target_hook_fired_fraction".to_string(),
        Expectation::MustBeBelow(0.1),
    )]);
    let mut measured = EvidenceCounters::new();
    measured.record("target_hook_fire_fraction", 0.0); // deliberately misspelled

    assert!(
        check_integrity(&declared, &measured).is_err(),
        "a missing declared probe must not inherit zero from a different counter"
    );
}

#[test]
fn run_evidence_does_not_cache_missing_negative_measurement_as_satisfied() {
    let declared = BTreeMap::from([("target_hook".to_string(), Expectation::MustBeZero)]);
    let measured = EvidenceCounters::new();

    let evidence = RunEvidence::new(RunId::from("missing-negative-probe"), &(), declared, measured);

    assert!(
        !evidence.satisfied,
        "RunEvidence must not record positive integrity when a declared negative probe is absent"
    );
    assert!(
        !evidence.violations.is_empty(),
        "the missing measurement must remain visible in the audit record"
    );
}
