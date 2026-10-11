// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Downward-only ratchet for known pre-ETK engineering authority bypasses.
//!
//! This is intentionally **not** a closure theorem. The baseline is legacy
//! debt recorded in `ENGINEERING_TRUST_KERNEL_AUTHORITY_BYPASS_CENSUS_V1.md`.
//! These tests make that debt non-expanding while ETK-3A..3D remove it.

const ENGINEERING: &str = include_str!("../../../domains/symthaea-engineering/src/lib.rs");
const FORMAL_SAFETY: &str = include_str!("../../../domains/symthaea-formal-safety/src/lib.rs");
const SOVEREIGN_EXAMPLE: &str = include_str!("../../../../examples/sovereign_design_loop.rs");

fn assert_at_most(source: &str, needle: &str, baseline: usize, label: &str) {
    let observed = source.matches(needle).count();
    assert!(
        observed <= baseline,
        "ETK authority-bypass ratchet regressed for {label}: observed {observed}, baseline {baseline}; new authority creation/amplification must cross the Engineering Trust Kernel instead of increasing legacy debt"
    );
}

#[test]
fn engineering_direct_discharge_writes_do_not_increase() {
    // Current legacy sites: evaluate_concept + structural + electrical +
    // thermofluid + shared native-discipline discharge helper.
    assert_at_most(
        ENGINEERING,
        "status = formal_safety::ObligationStatus::Discharged;",
        5,
        "engineering direct Discharged writes",
    );
}

#[test]
fn formal_safety_free_form_discharge_write_does_not_increase() {
    // Legacy ProofObligation::discharge(String). ETK-qualified paths must not
    // add peers to this mutable/free-form authority transition.
    assert_at_most(
        FORMAL_SAFETY,
        "self.status = ObligationStatus::Discharged;",
        1,
        "formal-safety free-form discharge writes",
    );
}

#[test]
fn example_direct_discharge_shortcuts_do_not_increase() {
    assert_at_most(
        SOVEREIGN_EXAMPLE,
        "symthaea_formal_safety::ObligationStatus::Discharged;",
        1,
        "example direct Discharged writes",
    );
}

#[test]
fn mutable_discharge_consumers_do_not_increase() {
    // Current consumers include placeholder Lean rendering and proof gossip.
    // ETK-3D should drive this count down as both move to typed receipts.
    assert_at_most(
        ENGINEERING,
        "obligation.status == formal_safety::ObligationStatus::Discharged",
        2,
        "mutable Discharged consumers",
    );
}

#[test]
fn verified_proof_amplification_does_not_increase() {
    // One legacy proof-gossip path currently emits verified=true from mutable
    // discharge state. New paths must carry explicit verification authority.
    assert_at_most(
        ENGINEERING,
        "verified: true",
        1,
        "unqualified verified proof amplification",
    );
}

#[test]
fn report_authority_overclaims_do_not_increase() {
    assert_at_most(
        ENGINEERING,
        "Status: **DISCHARGED**",
        1,
        "hard-coded DISCHARGED report labels",
    );
    assert_at_most(
        ENGINEERING,
        "mathematically proven",
        1,
        "unqualified mathematically-proven wording",
    );
}
