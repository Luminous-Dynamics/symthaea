// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Persistence-boundary diagnostics for `RunEvidence`.
//!
//! These tests establish an evidence contract only. They do not claim that a
//! successful evidence-plane integrity check is a scientific result.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

fn declared_positive_counter() -> BTreeMap<String, Expectation> {
    let mut declared = BTreeMap::new();
    declared.insert("must_fire".to_string(), Expectation::MustBePositive);
    declared
}

fn forged_positive_record() -> RunEvidence {
    RunEvidence {
        run_id: RunId::new("forged-positive-record"),
        config_hash: "diagnostic-only-not-authority".to_string(),
        declared: declared_positive_counter(),
        measured: EvidenceCounters::new(), // missing => honestly 0.0; MustBePositive fails
        satisfied: true,                   // forged cached positive state
        violations: Vec::new(),            // forged cached empty failure list
    }
}

#[test]
fn forged_positive_in_memory_record_cannot_bypass_enforcement() {
    let forged = forged_positive_record();
    let result = catch_unwind(AssertUnwindSafe(|| forged.enforce()));

    assert!(
        result.is_err(),
        "RunEvidence::enforce must recompute declared-vs-measured integrity; \
         caller-supplied satisfied=true must not mint current positive integrity"
    );
}

#[test]
fn forged_deserialized_positive_record_cannot_bypass_enforcement() {
    let bytes = serde_json::to_vec(&forged_positive_record()).expect("serialize forged record");
    let loaded: RunEvidence = serde_json::from_slice(&bytes).expect("deserialize forged record");

    let result = catch_unwind(AssertUnwindSafe(|| loaded.enforce()));
    assert!(
        result.is_err(),
        "deserializing an audit record must not deserialize positive verifier authority"
    );
}

#[test]
fn valid_round_trip_remains_revalidatable() {
    let mut measured = EvidenceCounters::new();
    measured.record("must_fire", 1.0);

    let original = RunEvidence::new(
        RunId::new("valid-round-trip"),
        &"diagnostic-config",
        declared_positive_counter(),
        measured,
    );
    assert!(original.satisfied);

    let bytes = serde_json::to_vec(&original).expect("serialize valid record");
    let loaded: RunEvidence = serde_json::from_slice(&bytes).expect("deserialize valid record");

    let result = catch_unwind(AssertUnwindSafe(|| loaded.enforce()));
    assert!(
        result.is_ok(),
        "a valid persisted record should remain usable after verifier-owned revalidation"
    );
}

#[test]
fn stale_cached_violation_state_cannot_define_current_integrity() {
    let mut measured = EvidenceCounters::new();
    measured.record("must_fire", 1.0); // current raw evidence satisfies the declaration

    let stale = RunEvidence {
        run_id: RunId::new("stale-negative-cache"),
        config_hash: "diagnostic-only-not-authority".to_string(),
        declared: declared_positive_counter(),
        measured,
        satisfied: false,
        violations: Vec::new(), // deliberately contradictory cached state
    };

    let result = catch_unwind(AssertUnwindSafe(|| stale.enforce()));
    assert!(
        result.is_ok(),
        "cached satisfied/violations are audit fields, not authority; current raw evidence \
         should be revalidated rather than trusting stale negative cache state"
    );
}
