// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral Parser Demo
//!
//! Demonstrates natural language parsing into moral algebra structures.
//! Shows how the MoralParser extracts ethical primitives from text and
//! encodes them for compositional reasoning.
//!
//! Run with: cargo run --example demo_moral_parser

use symthaea::hdc::moral_algebra::MoralAlgebra;
use symthaea::hdc::moral_parser::MoralParser;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Moral Parser: Natural Language to HDC                ║");
    println!("║      Extracting Ethical Primitives from Text                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let parser = MoralParser::new();
    let algebra = MoralAlgebra::default_dim();

    // ========================================================================
    // Demo 1: Consent Violation Scenarios (Commonsense Ethics)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: CONSENT SCENARIOS (Commonsense Ethics)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let consent_scenarios = [
        "I discussed my daughter's health without asking first",
        "After asking my daughter, I discussed her health with the doctor",
        "I shared my friend's secret without permission",
        "I told everyone about the surprise party after getting approval",
    ];

    let violation_proto = algebra.consent_violation_prototype();

    for scenario in &consent_scenarios {
        let encoded = parser.parse_and_encode(scenario, &algebra);
        let parsed = &encoded.parsed;

        println!("  Scenario: \"{}\"", scenario);
        println!("    Detected:");
        println!("      Agent: {:?}", parsed.agent);
        println!("      Action: {:?}", parsed.action);
        println!("      Patient: {:?}", parsed.patient);
        println!("      Consent: {:?}", parsed.consent);
        println!("      Confidence: {:.0}%", parsed.confidence * 100.0);

        // Use the fixed is_consent_violation() method which directly checks
        // consent state rather than relying on HDC similarity
        let is_violation = encoded.is_consent_violation();
        let verdict = if is_violation {
            "⚠️  Potential Consent Violation"
        } else {
            "✓ Consent Respected"
        };

        // Also show HDC similarity for comparison
        if let Some(consent_hv) = &encoded.consent_hv {
            let violation_sim = consent_hv.similarity(&violation_proto);
            println!("    → HDC similarity: {:.3}", violation_sim);
        }

        println!("    → Consent violation: {}", is_violation);
        println!("    → Verdict: {}\n", verdict);
    }

    // ========================================================================
    // Demo 2: Intent Recognition
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: INTENT RECOGNITION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let intent_scenarios = [
        "I helped the elderly woman cross the street",
        "I stole money from the cash register",
        "I walked to the store to buy groceries",
        "I lied to my friend about where I was",
        "I saved the child from the burning building",
    ];

    for scenario in &intent_scenarios {
        let parsed = parser.parse(scenario);

        let intent_symbol = match parsed.intent {
            symthaea::hdc::moral_algebra::MoralIntent::Good => "😇 Good",
            symthaea::hdc::moral_algebra::MoralIntent::Bad => "😈 Bad",
            symthaea::hdc::moral_algebra::MoralIntent::Neutral => "😐 Neutral",
            symthaea::hdc::moral_algebra::MoralIntent::Unknown => "❓ Unknown",
        };

        println!("  \"{}\"", scenario);
        println!("    → Action: {:?}", parsed.action);
        println!("    → Intent: {}", intent_symbol);
        println!(
            "    → Negation: {}\n",
            if parsed.has_negation { "Yes" } else { "No" }
        );
    }

    // ========================================================================
    // Demo 3: Proportionality Judgments (Justice)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: PROPORTIONALITY (Justice Scenarios)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let justice_scenarios = [
        "I deserve a small reward because I helped once",
        "I deserve a huge bonus because I worked daily for years",
        "I deserve a brand new car because I cleaned the house once",
        "I deserve recognition because I constantly support the team",
    ];

    for scenario in &justice_scenarios {
        let parsed = parser.parse(scenario);

        let magnitude_str = match parsed.magnitude {
            Some(symthaea::hdc::moral_algebra::Magnitude::Tiny) => "Tiny",
            Some(symthaea::hdc::moral_algebra::Magnitude::Small) => "Small",
            Some(symthaea::hdc::moral_algebra::Magnitude::Medium) => "Medium",
            Some(symthaea::hdc::moral_algebra::Magnitude::Large) => "Large",
            Some(symthaea::hdc::moral_algebra::Magnitude::Huge) => "Huge",
            None => "Unspecified",
        };

        println!("  \"{}\"", scenario);
        println!("    → Action: {:?}", parsed.action);
        println!("    → Magnitude detected: {}", magnitude_str);
        println!("    → Confidence: {:.0}%\n", parsed.confidence * 100.0);
    }

    // ========================================================================
    // Demo 4: Negation Handling
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: NEGATION HANDLING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let negation_pairs = [
        ("I helped my friend move", "I didn't help my friend move"),
        ("I told the truth", "I never told the truth"),
        ("I shared the resources", "I refused to share the resources"),
    ];

    for (positive, negative) in &negation_pairs {
        let pos_encoded = parser.parse_and_encode(positive, &algebra);
        let neg_encoded = parser.parse_and_encode(negative, &algebra);

        println!("  Positive: \"{}\"", positive);
        println!("    → Negation: {}", pos_encoded.parsed.has_negation);
        println!("    → Intent: {:?}", pos_encoded.parsed.intent);

        println!("  Negative: \"{}\"", negative);
        println!("    → Negation: {}", neg_encoded.parsed.has_negation);
        println!("    → Intent: {:?}", neg_encoded.parsed.intent);

        // If both have action HVs, show how negation affects similarity
        if let (Some(pos_hv), Some(neg_hv)) = (&pos_encoded.action_hv, &neg_encoded.action_hv) {
            let similarity = pos_hv.similarity(neg_hv);
            println!(
                "    → HDC Similarity: {:.3} (lower = more opposite)\n",
                similarity
            );
        } else {
            println!();
        }
    }

    // ========================================================================
    // Demo 5: Full Pipeline - Ensemble Judgment
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 5: ENSEMBLE JUDGMENT (HDC + Intent + Deontology)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let full_scenarios = [
        "I kindly helped the stranger find their way",
        "I cruelly harmed the innocent victim",
        "I discussed the project without telling anyone",
        "I saved the child from drowning",
        "I stole money from the elderly woman",
    ];

    for scenario in &full_scenarios {
        let encoded = parser.parse_and_encode(scenario, &algebra);

        println!("  Input: \"{}\"", scenario);
        println!("  ┌─ Parsed Structure ─────────────────────────────────");
        println!("  │ Agent:    {:?}", encoded.parsed.agent);
        println!("  │ Action:   {:?}", encoded.parsed.action);
        println!("  │ Patient:  {:?}", encoded.parsed.patient);
        println!("  │ Intent:   {:?}", encoded.parsed.intent);
        println!("  └──────────────────────────────────────────────────────");

        // Use ensemble judgment for better accuracy
        let ensemble = encoded.judge_ensemble(&algebra, scenario);

        println!("  ┌─ Ensemble Voting ─────────────────────────────────");
        println!(
            "  │ HDC:        {:?} (conf: {:+.3})",
            ensemble
                .hdc_verdict
                .map(|v| format!("{:?}", v))
                .unwrap_or("N/A".to_string()),
            ensemble.hdc_confidence.unwrap_or(0.0)
        );
        println!("  │ Intent:     {:?}", ensemble.intent_verdict);
        println!(
            "  │ Deontology: {:?} (score: {:+.2})",
            ensemble.deonto_verdict, ensemble.deonto_score
        );
        println!("  └──────────────────────────────────────────────────────");

        let verdict_emoji = match ensemble.final_verdict {
            symthaea::hdc::moral_algebra::MoralVerdict::Good => "✓ Good",
            symthaea::hdc::moral_algebra::MoralVerdict::Bad => "✗ Bad",
            symthaea::hdc::moral_algebra::MoralVerdict::Neutral => "○ Neutral",
            symthaea::hdc::moral_algebra::MoralVerdict::ConsentViolation => "⚠ Consent Violation",
        };

        let unanimous = if ensemble.is_unanimous() {
            " (unanimous)"
        } else {
            ""
        };
        println!(
            "  → Final: {} (confidence: {:.0}%){}\n",
            verdict_emoji,
            ensemble.confidence * 100.0,
            unanimous
        );
    }

    // ========================================================================
    // Demo 6: Deontological Judgment (Obligation Rules)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 6: DEONTOLOGICAL JUDGMENT (Obligation Rules)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let deontology_scenarios = [
        "I lied to protect my friend from harm",
        "I stole food to feed my starving children",
        "I broke my promise because I had an emergency",
        "I helped the neighbor without expecting anything",
        "I cheated on the exam to get ahead",
    ];

    for scenario in &deontology_scenarios {
        let judgment = algebra.judge_deontological(scenario);

        println!("  \"{}\"", scenario);
        println!("    ┌─ Violations ─────────────────────────────────────────");
        if judgment.violations.is_empty() {
            println!("    │ (none)");
        } else {
            for v in &judgment.violations {
                println!("    │ {} ({})", v.rule_name, v.matched_phrase);
            }
        }
        println!("    └─ Satisfactions ──────────────────────────────────────");
        if judgment.satisfactions.is_empty() {
            println!("    │ (none)");
        } else {
            for s in &judgment.satisfactions {
                println!("    │ {} ({})", s.rule_name, s.matched_phrase);
            }
        }

        let verdict_emoji = match judgment.verdict {
            symthaea::hdc::moral_algebra::DeontologicalVerdict::RightDutyFulfilled => {
                "✓ Right (Duty Fulfilled)"
            }
            symthaea::hdc::moral_algebra::DeontologicalVerdict::WrongPerfectDutyViolated => {
                "✗ Wrong (Perfect Duty Violated)"
            }
            symthaea::hdc::moral_algebra::DeontologicalVerdict::WrongImperfectDutyViolated => {
                "⚠ Wrong (Imperfect Duty Violated)"
            }
            symthaea::hdc::moral_algebra::DeontologicalVerdict::Neutral => "○ Neutral",
        };
        println!(
            "    └─ Verdict: {} (score: {:.2})\n",
            verdict_emoji, judgment.score
        );
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Parser extracts from natural language:                      ║");
    println!("║    • AGENT (who acts)                                        ║");
    println!("║    • ACTION (what happens)                                   ║");
    println!("║    • PATIENT (who is affected)                               ║");
    println!("║    • INTENT (good/bad/neutral)                               ║");
    println!("║    • CONSENT (given/absent/implied)                          ║");
    println!("║    • MAGNITUDE (tiny to huge)                                ║");
    println!("║    • NEGATION (inverts moral meaning)                        ║");
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!("║  Moral Algebra then enables:                                 ║");
    println!("║    • Consent violation detection                             ║");
    println!("║    • Proportionality judgments (justice)                     ║");
    println!("║    • Deontological reasoning (7 obligation rules)            ║");
    println!("║    • Good/bad action classification                          ║");
    println!("║    • Compositional reasoning via HDC                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}