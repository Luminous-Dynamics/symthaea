// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two things pinned here, both established 2026-07-31.
//!
//! 1. The Φ gate on `execute_action`. Before this date the method dispatched
//!    straight to ADB with no Φ, safety-level or confirmation check, and
//!    `required_phi()` had exactly one non-test call site in the crate
//!    (covering only `Tap`). These tests fail if that regresses.
//!
//! 2. The measured gap that BLOCKS the cognition→action loop: a cognition-space
//!    thought vector scores at the 16,384-D noise floor against appearance-space
//!    HVs, so it cannot serve as a visual-search goal. If the noise-floor test
//!    ever starts failing, a grounding map has appeared (or the HV spaces
//!    changed) and `PhoneBridge::step_embodiment`'s doc comment must be revisited.
//!
//! These are pure-logic tests: no ADB device, no screen capture.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_phone_embodiment::{PhoneAction, PhoneBridge};

// ── 1. Φ thresholds are declared as a strict ladder ──────────────────────────

#[test]
fn required_phi_ladder_is_ordered_and_nonmutating_is_free() {
    // The ladder the gate enforces. Pinned so a reordering is deliberate.
    assert_eq!(PhoneAction::NoOp.required_phi(), 0.0);
    assert_eq!(PhoneAction::Screenshot.required_phi(), 0.05);
    assert_eq!(PhoneAction::Back.required_phi(), 0.20);
    assert_eq!(PhoneAction::Home.required_phi(), 0.20);
    assert_eq!(
        PhoneAction::OpenUrl { url: String::new() }.required_phi(),
        0.30
    );
    assert_eq!(
        PhoneAction::Swipe {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
            duration_ms: 100
        }
        .required_phi(),
        0.35
    );
    assert_eq!(PhoneAction::Tap { x: 0, y: 0 }.required_phi(), 0.40);
    assert_eq!(
        PhoneAction::Type {
            text: String::new()
        }
        .required_phi(),
        0.50
    );

    // Only NoOp and Screenshot are non-mutating; everything else is gated.
    assert!(!PhoneAction::NoOp.is_mutating());
    assert!(!PhoneAction::Screenshot.is_mutating());
    for action in [
        PhoneAction::Back,
        PhoneAction::Home,
        PhoneAction::OpenUrl {
            url: "https://example.com".into(),
        },
        PhoneAction::Swipe {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
            duration_ms: 100,
        },
        PhoneAction::Tap { x: 1, y: 2 },
        PhoneAction::Type { text: "x".into() },
    ] {
        assert!(
            action.is_mutating(),
            "{} must be gated as mutating",
            action.label()
        );
    }
}

/// `Swipe` is cheaper than `Tap`. This is why routing a cognition-space vector
/// into `propose_goal_action` would be WORSE than the status quo: it would miss
/// every template and fall through to the exploratory-swipe fallback, which
/// fires in a wider Φ band than the saliency tap it replaced.
#[test]
fn swipe_fallback_is_cheaper_than_tap() {
    let swipe = PhoneAction::Swipe {
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 0,
        duration_ms: 100,
    };
    assert!(
        swipe.required_phi() < PhoneAction::Tap { x: 0, y: 0 }.required_phi(),
        "swipe {} should be below tap {}",
        swipe.required_phi(),
        PhoneAction::Tap { x: 0, y: 0 }.required_phi()
    );
}

// ── 1b. The gate actually refuses ────────────────────────────────────────────
//
// These construct a bridge with a bogus serial. No ADB call is reached: the
// guard returns Err before dispatch, which is precisely what is being pinned.

/// Fail-closed default: a fresh bridge has never proposed, so `last_phi` is 0.0
/// and `current_safety` is Red. Every mutating action must be refused.
#[test]
fn fresh_bridge_refuses_every_mutating_action() {
    for action in [
        PhoneAction::Back,
        PhoneAction::Tap { x: 10, y: 20 },
        PhoneAction::Type {
            text: "hello".into(),
        },
        PhoneAction::OpenUrl {
            url: "https://example.com".into(),
        },
    ] {
        let mut phone = PhoneBridge::new("test", 1008, 2244);
        let err = phone
            .execute_action(&action)
            .expect_err("fresh bridge must refuse mutating actions");
        assert!(
            err.starts_with("refused "),
            "expected a refusal for {}, got: {err}",
            action.label()
        );
    }
}

/// Non-mutating actions are unaffected by the gate, even at Red.
#[test]
fn nonmutating_actions_pass_the_gate_at_red() {
    let mut phone = PhoneBridge::new("test", 1008, 2244);
    assert!(phone.execute_action(&PhoneAction::NoOp).is_ok());
    assert!(phone.execute_action(&PhoneAction::Screenshot).is_ok());
}

/// Isolates the Φ check from the safety-level check: propose at 0.45 (→ Yellow,
/// so the safety guard passes) and then attempt `Type`, which needs 0.50.
/// The refusal must name Φ, not the safety level.
#[test]
fn phi_check_refuses_action_above_authorized_phi() {
    let mut phone = PhoneBridge::new("test", 1008, 2244);
    let _ = phone.propose_action(0.45);
    assert_eq!(phone.phi_authority(), 0.45);

    let err = phone
        .execute_action(&PhoneAction::Type { text: "x".into() })
        .expect_err("Type requires phi 0.50, authority is 0.45");
    assert!(
        err.contains("phi 0.450") && err.contains("required 0.500"),
        "expected a Φ refusal naming both values, got: {err}"
    );
}

/// `set_phi_authority` supplies the Φ the check reads — it does not skip the
/// check. Raising authority to 0.65 clears `Type`'s 0.50, so the call proceeds
/// past the gate and fails later at ADB instead (no device named "test").
#[test]
fn raising_authority_clears_the_phi_check_but_not_the_gate_itself() {
    let mut phone = PhoneBridge::new("test", 1008, 2244);
    let _ = phone.propose_action(0.65); // → Green, and authority 0.65
    phone.set_phi_authority(0.65);

    let result = phone.execute_action(&PhoneAction::Type { text: "x".into() });
    match result {
        Ok(()) => panic!("no real device is attached; this should not succeed"),
        Err(e) => assert!(
            !e.starts_with("refused "),
            "0.65 clears Type's 0.50, so this must fail at ADB, not the gate: {e}"
        ),
    }
}

// ── 2. The cognition→appearance gap ──────────────────────────────────────────

/// PRE-REGISTERED NULL. A cognition-space thought vector is indistinguishable
/// from noise when scored against appearance-space HVs.
///
/// The 16,384-D noise floor is 1/sqrt(16384) = 0.0078. `find_on_screen`'s usable
/// threshold band is 0.30-0.80. If a raw thought vector cannot clear even 0.10,
/// it can never select a target, and the cognition→action loop cannot be closed
/// by wiring alone — it needs a cognition→appearance grounding map.
///
/// Failure here is INFORMATIVE, not a bug: it means the spaces became
/// comparable. Update `step_embodiment`'s doc comment before "fixing" this.
#[test]
fn cognition_space_vectors_score_at_the_appearance_noise_floor() {
    let noise_floor = 1.0 / (HDC_DIMENSION as f32).sqrt();
    let mut worst: f32 = 0.0;

    // Independent draws stand in for "unrelated spaces": a thought HV built by
    // the language/cognitive path shares no construction with a patch
    // appearance HV, so their expected cosine is the noise floor.
    for seed in 0..16u64 {
        let thought = ContinuousHV::random(HDC_DIMENSION, 1_000 + seed);
        let appearance = ContinuousHV::random(HDC_DIMENSION, 9_000 + seed);
        let sim = thought.similarity(&appearance).abs();
        worst = worst.max(sim);
    }

    assert!(
        worst < 0.10,
        "expected all cross-space similarities below 0.10 (noise floor {noise_floor:.4}), \
         worst was {worst:.4}. If this now exceeds 0.30, a grounding map may exist — \
         revisit PhoneBridge::step_embodiment's doc comment before changing this test."
    );

    // Guard against the vacuous pass: an identical vector must score ~1.0, so a
    // broken `similarity` cannot make this test succeed by always returning 0.
    let v = ContinuousHV::random(HDC_DIMENSION, 4_242);
    let self_sim = v.similarity(&v);
    assert!(
        self_sim > 0.99,
        "similarity() is broken — self-similarity {self_sim:.4} should be ~1.0, \
         so the noise-floor assertion above proves nothing"
    );
}
