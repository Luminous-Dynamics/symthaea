// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Expected-red regression for finite numeric evidence.
//!
//! IEEE-754 infinities may satisfy ordinary comparisons, but they are invalid
//! experimental measurements/thresholds and must fail before predicate logic.

use std::collections::HashMap;

use symthaea_evidence_plane::{EvidenceCounters, Expectation, check_integrity};

fn check(value: f64, expectation: Expectation) -> bool {
    let declared = HashMap::from([("probe".to_string(), expectation)]);
    let mut measured = EvidenceCounters::new();
    measured.record("probe", value);
    check_integrity(&declared, &measured).is_ok()
}

#[test]
fn positive_infinity_cannot_satisfy_must_be_positive() {
    assert!(
        !check(f64::INFINITY, Expectation::MustBePositive),
        "+infinity is invalid evidence, not an observed positive measurement"
    );
}

#[test]
fn positive_infinity_cannot_satisfy_must_exceed() {
    assert!(
        !check(f64::INFINITY, Expectation::MustExceed(0.9)),
        "+infinity must fail validation before threshold comparison"
    );
}

#[test]
fn negative_infinity_cannot_satisfy_must_be_below() {
    assert!(
        !check(f64::NEG_INFINITY, Expectation::MustBeBelow(0.1)),
        "-infinity is invalid evidence, not a qualified low measurement"
    );
}

#[test]
fn infinite_upper_threshold_cannot_trivialize_must_be_below() {
    assert!(
        !check(0.5, Expectation::MustBeBelow(f64::INFINITY)),
        "a non-finite expectation threshold is malformed policy, not an automatic pass"
    );
}

#[test]
fn negative_infinite_lower_threshold_cannot_trivialize_must_exceed() {
    assert!(
        !check(0.5, Expectation::MustExceed(f64::NEG_INFINITY)),
        "a non-finite expectation threshold is malformed policy, not an automatic pass"
    );
}

#[test]
fn finite_values_keep_existing_predicate_semantics() {
    assert!(check(1.0, Expectation::MustBePositive));
    assert!(check(0.95, Expectation::MustExceed(0.9)));
    assert!(check(0.05, Expectation::MustBeBelow(0.1)));
    assert!(check(0.0, Expectation::MustBeZero));
}
