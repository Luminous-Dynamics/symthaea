// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral Algebra Demo
//!
//! Demonstrates compositional moral reasoning using HDC primitives.
//! Shows how the 7 moral primitives and 5 operators can reason about:
//! - Consent violations (commonsense ethics)
//! - Proportionality judgments (justice)
//! - Excuse validity (deontology)
//!
//! Run with: cargo run --example demo_moral_algebra

use symthaea::hdc::moral_algebra::{ConsentState, Magnitude, MoralAlgebra, MoralIntent};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Moral Algebra in Hyperdimensional Space            ║");
    println!("║        Compositional Ethical Reasoning via HDC               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let algebra = MoralAlgebra::default_dim();

    // ========================================================================
    // Demo 1: Consent Reasoning (Commonsense Ethics)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: CONSENT REASONING (Commonsense Ethics)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Scenario: Discussing someone's health information with others\n");

    // With consent
    let with_consent = algebra.encode_consent_action(
        "discuss health information",
        "daughter",
        ConsentState::Given,
    );

    // Without consent
    let without_consent = algebra.encode_consent_action(
        "discuss health information",
        "daughter",
        ConsentState::Absent,
    );

    let violation_proto = algebra.consent_violation_prototype();

    let with_sim = with_consent.similarity(&violation_proto);
    let without_sim = without_consent.similarity(&violation_proto);

    println!("  Case A: \"I discussed my daughter's health after asking her first\"");
    println!("    → Consent: Given");
    println!("    → Similarity to violation prototype: {:.3}", with_sim);
    println!(
        "    → Verdict: {}\n",
        if with_sim < 0.3 {
            "✓ Ethical"
        } else {
            "✗ Problematic"
        }
    );

    println!("  Case B: \"I discussed my daughter's health without asking first\"");
    println!("    → Consent: Absent");
    println!(
        "    → Similarity to violation prototype: {:.3}",
        without_sim
    );
    println!(
        "    → Verdict: {}\n",
        if without_sim > with_sim {
            "✗ Consent Violation"
        } else {
            "✓ Ethical"
        }
    );

    println!(
        "  Analysis: Without-consent case is {:.1}x more similar to violation pattern\n",
        without_sim / with_sim.max(0.001)
    );

    // ========================================================================
    // Demo 2: Proportionality Reasoning (Justice)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: PROPORTIONALITY REASONING (Justice)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Scenario: Evaluating whether rewards are proportional to effort\n");

    // Fair case: medium effort, medium reward
    let fair = algebra.encode_proportionality(
        "clean house daily",
        Magnitude::Medium,
        "fair wage",
        Magnitude::Medium,
    );

    // Unfair case: small effort, huge reward
    let unfair = algebra.encode_proportionality(
        "clean house once",
        Magnitude::Small,
        "brand new car",
        Magnitude::Huge,
    );

    let fair_judgment = algebra.judge_proportionality(&fair);
    let unfair_judgment = algebra.judge_proportionality(&unfair);

    println!("  Case A: \"I deserve fair wage because I clean house daily\"");
    println!("    → Effort: Medium, Reward: Medium");
    println!("    → Is proportional: {}", fair.is_proportional);
    println!(
        "    → Fair similarity: {:.3}",
        fair_judgment.fair_similarity
    );
    println!(
        "    → Verdict: {}\n",
        if fair_judgment.is_just {
            "✓ Just"
        } else {
            "✗ Unjust"
        }
    );

    println!("  Case B: \"I deserve a new car because I cleaned house once\"");
    println!("    → Effort: Small, Reward: Huge");
    println!("    → Is proportional: {}", unfair.is_proportional);
    println!(
        "    → Magnitude difference: {:.2}",
        unfair_judgment.magnitude_difference
    );
    println!(
        "    → Verdict: {}\n",
        if unfair_judgment.is_just {
            "✓ Just"
        } else {
            "✗ Unjust"
        }
    );

    // ========================================================================
    // Demo 3: Excuse Validity (Deontology)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: EXCUSE VALIDITY (Deontology)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Scenario: \"Aren't you supposed to prepare for the meeting?\"\n");

    // Valid excuse: addresses the obligation
    let valid = algebra.encode_excuse_validity(
        "prepare for meeting",
        "already set up conference room",
        true,
    );

    // Invalid excuse: doesn't address obligation
    let invalid = algebra.encode_excuse_validity("prepare for meeting", "not in the mood", false);

    println!("  Case A: \"No, because I already set up the conference room\"");
    println!("    → Addresses obligation: Yes");
    println!(
        "    → Verdict: {}\n",
        if valid.is_valid {
            "✓ Valid Excuse"
        } else {
            "✗ Invalid"
        }
    );

    println!("  Case B: \"No, because I'm not in the mood\"");
    println!("    → Addresses obligation: No");
    println!(
        "    → Verdict: {}\n",
        if invalid.is_valid {
            "✓ Valid Excuse"
        } else {
            "✗ Invalid Excuse"
        }
    );

    // Show HV difference
    let excuse_sim = valid.composed.similarity(&invalid.composed);
    println!(
        "  HDC Analysis: Valid and invalid excuses have similarity {:.3}",
        excuse_sim
    );
    println!("  (Low similarity = good discrimination between valid/invalid)\n");

    // ========================================================================
    // Demo 4: Full Action Judgment
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: FULL ACTION JUDGMENT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Good action
    let good_action =
        algebra.encode_action_structure("helper", "assist", "stranger", MoralIntent::Good);

    // Bad action
    let bad_action = algebra.encode_action_structure("thief", "steal", "victim", MoralIntent::Bad);

    let good_judgment = algebra.judge_action(&good_action);
    let bad_judgment = algebra.judge_action(&bad_action);

    println!("  Case A: Helper assists stranger (good intent)");
    println!(
        "    → Good similarity: {:.3}",
        good_judgment.good_similarity
    );
    println!("    → Bad similarity: {:.3}", good_judgment.bad_similarity);
    println!("    → Verdict: {:?}\n", good_judgment.verdict);

    println!("  Case B: Thief steals from victim (bad intent)");
    println!("    → Good similarity: {:.3}", bad_judgment.good_similarity);
    println!("    → Bad similarity: {:.3}", bad_judgment.bad_similarity);
    println!("    → Verdict: {:?}\n", bad_judgment.verdict);

    // ========================================================================
    // Demo 5: Negation Operator
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 5: NEGATION OPERATOR");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let consent_given = algebra.encode_consent(ConsentState::Given);
    let consent_negated = algebra.negate(&consent_given);
    let double_negated = algebra.negate(&consent_negated);

    let original_vs_negated = consent_given.similarity(&consent_negated);
    let original_vs_double = consent_given.similarity(&double_negated);

    println!("  Consent(Given) vs NEGATE(Consent(Given))");
    println!(
        "    → Similarity: {:.3} (should be low - opposite meanings)\n",
        original_vs_negated
    );

    println!("  Consent(Given) vs NEGATE(NEGATE(Consent(Given)))");
    println!(
        "    → Similarity: {:.3} (should be higher - double negation)",
        original_vs_double
    );
    println!(
        "    → Recovery ratio: {:.2}x\n",
        original_vs_double / original_vs_negated.max(0.001)
    );

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Moral Primitives: AGENT, PATIENT, ACTION, INTENT,           ║");
    println!("║                    CONSENT, OBLIGATION, MAGNITUDE            ║");
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!("║  Moral Operators:  CAUSES, VIOLATES, SATISFIES,              ║");
    println!("║                    PROPORTIONAL, NEGATES                     ║");
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!("║  This algebraic approach enables compositional reasoning     ║");
    println!("║  that n-gram HDC cannot achieve, addressing the gap between  ║");
    println!("║  virtue ethics (80%) and other categories (~50%).            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}