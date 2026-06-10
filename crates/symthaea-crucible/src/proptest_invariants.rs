// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Property-Based Testing for Crucible Invariants (Option B)
//!
//! Uses proptest to throw arbitrary text at the moral algebra math
//! and verify universal invariants hold for ALL inputs.
//!
//! In a continuous 50Hz loop, a single NaN or out-of-bounds float
//! will panic the entire system. Proptest finds those fractures.

#[cfg(test)]
mod tests {
    use crate::harness::CrucibleEngine;
    use crate::invariants;
    use crate::scenarios::ScenarioStep;
    use proptest::prelude::*;

    // ── Strategy: arbitrary moral text ──────────────────────────────────────

    /// Generate random text that might break the encoder or math.
    fn arb_text() -> impl Strategy<Value = String> {
        prop_oneof![
            // Normal text
            "[a-z ]{1,200}",
            // Empty / whitespace
            Just("".to_string()),
            Just(" ".to_string()),
            Just("   \t\n  ".to_string()),
            // Unicode stress
            "[\u{0000}-\u{FFFF}]{1,100}",
            // Repeated characters
            "[a]{1,500}",
            // Numbers and symbols
            "[0-9!@#$%^&*()_+=]{1,100}",
            // Very long text
            "[a-z ]{500,1000}",
            // Mixed moral keywords (positive + negative)
            Just("help destroy care harm support exploit love hate".to_string()),
            Just("consent force willing against voluntary involuntary".to_string()),
        ]
    }

    // ── Property: universal invariants hold for any single input ────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_single_step_invariants(text in arb_text()) {
            let mut engine = CrucibleEngine::new();
            let steps = vec![ScenarioStep::new(&text)];
            let report = engine.run_scenario("proptest_single", &steps);

            for telemetry in &report.cycle_telemetry {
                let violations = invariants::check_universal(telemetry);
                prop_assert!(
                    violations.is_empty(),
                    "Input {:?} violated invariants: {:?}",
                    &text[..text.len().min(80)],
                    violations
                );
            }
        }

        #[test]
        fn prop_moral_score_bounded(text in arb_text()) {
            let mut engine = CrucibleEngine::new();
            let steps = vec![ScenarioStep::new(&text)];
            let report = engine.run_scenario("proptest_bounds", &steps);

            for t in &report.cycle_telemetry {
                prop_assert!(
                    t.moral_score >= -1.0 && t.moral_score <= 1.0,
                    "moral_score {} out of [-1,1] for input {:?}",
                    t.moral_score,
                    &text[..text.len().min(80)]
                );
            }
        }

        #[test]
        fn prop_free_energy_non_negative(text in arb_text()) {
            let mut engine = CrucibleEngine::new();
            let steps = vec![ScenarioStep::new(&text)];
            let report = engine.run_scenario("proptest_fe", &steps);

            for t in &report.cycle_telemetry {
                prop_assert!(
                    t.moral_free_energy >= -1e-9,
                    "Negative free energy {} for input {:?}",
                    t.moral_free_energy,
                    &text[..text.len().min(80)]
                );
            }
        }

        #[test]
        fn prop_anomaly_score_bounded(text in arb_text()) {
            let mut engine = CrucibleEngine::new();
            let steps = vec![ScenarioStep::new(&text)];
            let report = engine.run_scenario("proptest_anomaly", &steps);

            for t in &report.cycle_telemetry {
                prop_assert!(
                    t.anomaly_score >= 0.0 && t.anomaly_score <= 1.0,
                    "anomaly_score {} out of [0,1] for input {:?}",
                    t.anomaly_score,
                    &text[..text.len().min(80)]
                );
            }
        }

        #[test]
        fn prop_harmony_coordinates_finite(text in arb_text()) {
            let mut engine = CrucibleEngine::new();
            let steps = vec![ScenarioStep::new(&text)];
            let report = engine.run_scenario("proptest_coords", &steps);

            for t in &report.cycle_telemetry {
                for (i, &c) in t.harmony_coordinates.iter().enumerate() {
                    prop_assert!(
                        c.is_finite(),
                        "harmony_coordinates[{}] = {} (not finite) for input {:?}",
                        i, c,
                        &text[..text.len().min(80)]
                    );
                }
            }
        }
    }

    // ── Property: multi-step sequences ──────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_multi_step_no_panic(
            texts in prop::collection::vec(arb_text(), 2..20)
        ) {
            let mut engine = CrucibleEngine::new();
            let steps: Vec<ScenarioStep> = texts.iter().map(|t| ScenarioStep::new(t)).collect();
            let report = engine.run_scenario("proptest_multi", &steps);

            // All per-cycle invariants must hold
            for telemetry in &report.cycle_telemetry {
                let violations = invariants::check_universal(telemetry);
                prop_assert!(
                    violations.is_empty(),
                    "Multi-step violated invariants: {:?}",
                    violations
                );
            }
        }

        #[test]
        fn prop_engine_reset_is_clean(
            text_a in arb_text(),
            text_b in arb_text(),
        ) {
            let mut engine = CrucibleEngine::new();

            // Run first scenario
            let steps_a = vec![ScenarioStep::new(&text_a)];
            let _report_a = engine.run_scenario("proptest_a", &steps_a);

            // Reset
            engine.reset();

            // Run second scenario
            let steps_b = vec![ScenarioStep::new(&text_b)];
            let report_b = engine.run_scenario("proptest_b", &steps_b);

            // Post-reset scenario should pass invariants independently
            for t in &report_b.cycle_telemetry {
                let violations = invariants::check_universal(t);
                prop_assert!(
                    violations.is_empty(),
                    "Post-reset invariant violation: {:?}",
                    violations
                );
            }
        }
    }

    // ── Property: softmax and KL are numerically stable ─────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_encode_never_panics(text in arb_text()) {
            let engine = CrucibleEngine::new();
            let hv = engine.encode(&text);
            let coords = engine.project(&hv);
            for &c in &coords {
                prop_assert!(c.is_finite(), "Non-finite coord from {:?}", &text[..text.len().min(50)]);
            }
        }
    }
}
