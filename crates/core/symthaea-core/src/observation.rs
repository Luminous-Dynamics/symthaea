// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A narrow, Tier-1 observation contract for evidence-gathering consumers.
//!
//! Phase 2 of `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`.
//!
//! Mirrors [`crate::embodiment::EmbodimentBridge`]'s proven pattern (a narrow trait
//! living at the Tier-1 substrate level that consumers depend on instead of a
//! concrete, fast-growing implementation struct). The problem this solves: today
//! `symthaea-psych-bench` and `symthaea-pulse` depend on the main `symthaea` crate
//! in the *opposite* direction and drive `CognitiveLoopService` directly through its
//! full public method surface (`cycle_with_hv`, `cycle`, `state_dim`, `provide_reward`,
//! `CycleResult.metadata.structural.*`) -- a struct that has grown from a documented
//! 38 fields to ~124-135. Every evidence-gathering consumer built that way is exposed
//! to that churn. This trait is deliberately much smaller than that surface.
//!
//! Because `symthaea-core` must never depend on the main `symthaea` crate (that's
//! the whole point of it being Tier-1 substrate), every method here returns only
//! types that live in `symthaea-core` or `std`. Implementors translate their own
//! internal types into these generic shapes rather than exposing them directly.

/// A narrow, stable observation contract for evidence-gathering consumers.
///
/// Deliberately NOT a second copy of a service's full public API -- keep this
/// trait small. Add a method only when a real, external evidence-gathering
/// consumer needs it, not speculatively.
pub trait CognitiveObservation {
    /// Dimensionality of the primary cognitive state representation.
    fn state_dimensions(&self) -> usize;
}

// ── Removed 2026-07-29: three speculative methods, per this trait's own
// kill-condition ──────────────────────────────────────────────────────────────
//
// `active_backend_label() -> String`, `mechanism_activation() -> EvidenceCounters`,
// and `run_identity() -> (RunId, String)` were part of the original Phase 2 landing.
// The reconciliation plan recorded an explicit expiry for them: *"if those three
// still have no consumer by the time Phase 4's characterization work lands, delete
// them and keep the trait minimal... Phase 4's harness is the natural first customer
// for run_identity() and mechanism_activation() -- if it doesn't use them, that is
// strong evidence they were speculative."*
//
// Phase 4's characterization landed (`examples/safety_signal_characterization.rs`).
// It used none of them; workspace-wide consumer count was 0/0/0. So they were
// deleted, exactly as promised.
//
// This is recorded rather than silently dropped because an expiry condition nobody
// executes is itself decorative -- the same failure this program documented in
// `gate_civic()` (~475 call sites that never checked the returned eligibility) and
// in the motor-safety fail-safe that is unreachable on partial failure. The point of
// writing a kill-condition is to honour it when it fires.
//
// Re-add any of them the moment a REAL consumer needs one -- but add it with that
// consumer in the same change, not ahead of it.

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal fixture proving the trait is genuinely implementable with
    /// real (not stubbed) values, without depending on the main crate.
    struct Fixture {
        dims: usize,
    }

    impl CognitiveObservation for Fixture {
        fn state_dimensions(&self) -> usize {
            self.dims
        }
    }

    #[test]
    fn fixture_reports_real_values() {
        let f = Fixture { dims: 16_384 };
        assert_eq!(f.state_dimensions(), 16_384);
    }
}
