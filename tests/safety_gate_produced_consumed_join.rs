// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A1.1 readiness gate: demonstrate the produced/consumed join on a real
//! `EmbodimentBridge` implementor.
//!
//! `SYMTHAEA_PHASE4_CHARACTERIZATION_PROTOCOL_2026-07-29.md`, Amendment 1 (A1.1), makes
//! consumption-boundary instrumentation a **blocking precondition** for any
//! characterization run, and states that the gate is closed not when the code compiles
//! but when a produced/consumed join has been *demonstrated* on at least one real
//! implementor. This test is that demonstration.
//!
//! Why the gate exists: write-side provenance records what a formula *produced*. It
//! cannot establish what the safety path *consumed*. Between the two, a value can be
//! stale, re-clamped at an intermediate layer, misaligned by a cycle, served from cache,
//! per-platform overridden, defaulted, or computed and never used. All are live
//! possibilities in this codebase, which already contains a second gate writer nobody
//! knew about, a `social_mod` applied after the gate write, and four platforms that
//! bypass `motor_gain()`.
//!
//! **What this test asserts is a structural property, not a good outcome.** The gate
//! consumes the *previous* cycle's value by construction: the embodiment block runs in
//! PHASE 2 of `cycle()`, before the feedback phase that rewrites
//! `carryover.history.consciousness_level`. A lag of ≥1 is therefore expected and
//! correct — it is stated here in advance so that observing it is not later reported as
//! a discovery. What would be a genuine finding is a lag exceeding the 67-cycle
//! spectral/structural refresh interval (the tier selected from a value no recent
//! measurement produced), or a `consumed_value` that does not match any value the field
//! ever held.
//!
//! Run: cargo test --features humanoid --test safety_gate_produced_consumed_join

#![cfg(feature = "humanoid")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some("a11-produced-consumed-join".to_string());
    config.async_training = false;
    CognitiveLoopService::new(config).expect("service must construct")
}

/// **Characterizes the cold-start window**, which the first run of this test discovered
/// rather than assumed.
///
/// `carryover.history.consciousness_level` initialises to 0.05 — a *prior*, explicitly
/// commented in `carryover.rs` as "Floor: prevents fully unconscious cold-start", not a
/// measurement. The text-path writer is gated on
/// `should_run(total_cycles, 5, 10, 20)`, and `total_cycles` increments at the TOP of
/// `cycle()`, so the first write lands on cycle 5, 10, or 20 depending on urgency
/// (Critical/Normal/Cruise). **Measured: cycle 5, consumed at cycle 6** — i.e. urgency is
/// Critical at startup, not Normal, which is itself worth knowing and was NOT what this
/// comment originally predicted. So the cold-start window is 5-19 cycles wide depending
/// on startup urgency, and the gate consumes the prior throughout it.
///
/// `from_phi(0.05)` is **Red** (0.05 ≤ 0.1), so a freshly-constructed robot sits in its
/// platform's SafeFallback (StandingLock for a humanoid) until the first real write.
/// That is fail-safe and arguably correct for cold start — the system genuinely knows
/// nothing yet. It is asserted here so it stays a *chosen* behavior rather than an
/// emergent one, and so a change to either the floor value or the MCE cadence surfaces
/// as a failing test.
///
/// Consequence for prior work: the `min=0.0500` values in the 2026-07-29 true-gate
/// characterization were this prior, not measured lows — those tier-occupancy figures
/// include Red cycles that were priors. Recorded in the Phase 4 protocol.
#[test]
fn cold_start_gate_consumes_a_prior_not_a_measurement() {
    let mut svc = service();
    svc.switch_embodiment(symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform::Humanoid);

    let _ = svc.cycle("first cycle after construction");

    let c = svc
        .last_gate_consumption()
        .expect("embodiment steps every cycle at interval 1");

    assert_eq!(
        c.writer,
        symthaea::cognitive_loop::types::GateWriter::ColdStartFloor,
        "on cycle 1 no formula has written the gate field yet"
    );
    assert!(
        (c.consumed_value - 0.05).abs() < 1e-12,
        "expected the documented 0.05 cold-start floor, got {}",
        c.consumed_value
    );
    assert_eq!(
        c.resulting_tier,
        symthaea_core::embodiment::MotorSafetyLevel::Red,
        "0.05 maps to Red — a cold-started robot is in SafeFallback, gated by a prior"
    );
}

/// The A1.1 join, with the transition **measured** rather than assumed.
///
/// The first version of this test asserted that "5 real cycles" would be enough for a
/// real formula to have written the gate field. That was an unverified assumption about
/// the MCE cadence and it was wrong — see the cold-start test above. It is replaced with
/// a measurement of when the transition actually occurs, plus invariants that hold
/// regardless of cadence. Bumping the cycle count until the original assertion passed
/// would have hidden the finding that produced this comment.
#[test]
fn produced_consumed_join_is_observable_on_a_real_implementor() {
    let mut svc = service();
    svc.switch_embodiment(symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform::Humanoid);

    assert!(
        svc.last_gate_consumption().is_none(),
        "no consumption record should exist before the first embodiment step"
    );

    // Run well past the first MCE firing (cycle 5/10/20 depending on urgency; measured
    // at 5, consumed at 6) recording when the consumed value first comes from a real
    // formula. 30 cycles covers even the Cruise-urgency interval of 20.
    let mut first_real_write: Option<(usize, symthaea::cognitive_loop::types::GateWriter)> = None;
    for _ in 0..30 {
        let _ = svc.cycle("the join demonstration proceeds");
        if first_real_write.is_none() {
            if let Some(c) = svc.last_gate_consumption() {
                if c.writer != symthaea::cognitive_loop::types::GateWriter::ColdStartFloor {
                    first_real_write = Some((c.cycle_index, c.writer));
                }
            }
        }
    }

    let (transition_cycle, transition_writer) = first_real_write.expect(
        "within 30 cycles a real formula must have written the gate field; if this fails \
         the gate runs on the 0.05 prior indefinitely, which IS a primary finding",
    );

    let c = svc
        .last_gate_consumption()
        .expect("A1.1: an embodiment step must have recorded what it consumed");
    let (_writer_now, written_at_now) = svc.safety_gate_provenance();

    // ── Invariants that hold regardless of MCE cadence ───────────────────────
    assert!(
        c.written_at <= c.cycle_index,
        "a value cannot be consumed before it was written (cycle {} < written_at {})",
        c.cycle_index,
        c.written_at
    );
    assert_eq!(
        c.resulting_tier,
        symthaea_core::embodiment::MotorSafetyLevel::from_phi(c.consumed_value),
        "resulting_tier must equal from_phi(consumed_value) — if these diverge the record \
         computes its own ladder instead of using production's"
    );
    assert_eq!(
        c.platform,
        symthaea_core::embodiment::EmbodimentPlatform::Humanoid
    );
    assert!(
        written_at_now >= c.written_at,
        "the field's write pointer must not move backwards"
    );

    // Staleness beyond the 67-cycle spectral/structural refresh interval would mean the
    // tier came from a value no recent measurement produced — a finding, not a tuning
    // problem. Note the MCE's own 5/10/20-cycle cadence dominates the lag: measured at 5,
    // not the 1 that phase-ordering alone would give. So the gate routinely acts on a
    // value several cycles old, by design rather than by defect.
    let lag = c.lag_cycles();
    assert!(
        lag < 67,
        "consumed value is {lag} cycles stale, exceeding the 67-cycle refresh interval — \
         the tier was selected from a value no recent measurement produced. See the Phase 4 \
         protocol A1.1; this is a primary finding."
    );

    println!(
        "A1.1 join demonstrated. First real write at cycle {} by {}. \
         Latest: consumed {:.6} (writer {}, written_at {}) at cycle {} -> tier {:?}, \
         lag {} cycles; field now written_at {}",
        transition_cycle,
        transition_writer.label(),
        c.consumed_value,
        c.writer.label(),
        c.written_at,
        c.cycle_index,
        c.resulting_tier,
        lag,
        written_at_now
    );
}

/// Consumption can be strictly sparser than production. With
/// `embodiment_step_interval > 1` the gate is only read on some cycles, so a
/// produced-value series is not a consumed-value series — which is why the protocol
/// requires the consumed side to be recorded separately rather than inferred.
#[test]
fn consumption_is_sparser_than_production_when_interval_exceeds_one() {
    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some("a11-sparse-consumption".to_string());
    config.async_training = false;
    config.embodiment_step_interval = 4;
    let mut svc = CognitiveLoopService::new(config).expect("service must construct");
    svc.switch_embodiment(symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform::Humanoid);

    for _ in 0..9 {
        let _ = svc.cycle("sparse consumption");
    }

    let consumption = svc
        .last_gate_consumption()
        .expect("at least one embodiment step should have fired within 9 cycles");

    assert_eq!(
        consumption.cycle_index % 4,
        0,
        "with interval 4, consumption must land on a multiple of the interval (got cycle \
         {}) — if it does not, the gating condition and the record disagree",
        consumption.cycle_index
    );
}
