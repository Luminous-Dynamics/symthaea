// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual Planning Demo
//!
//! Demonstrates how causal reasoning integrates with temporal planning:
//!   1. Build a causal DAG from observed relationships
//!   2. Query for causal effects (backdoor/frontdoor identification)
//!   3. Use HDC graph surgery to compute interventions
//!   4. Apply semantic role substitution for "what-if" scenarios
//!   5. Compose with the reasoning engine for planning decisions
//!
//! Run: cargo run --example counterfactual_planning --features reasoning_engine

#[cfg(feature = "reasoning_engine")]
fn main() {
    use symthaea::consciousness::counterfactual::semantic_roles::{RoleSubstitution, SemanticRole};
    use symthaea::consciousness::counterfactual::{
        CausalDAG, CausalQuery, CausalReferenceHarness, CounterfactualReasoner,
        composer::CounterfactualComposer,
    };
    use symthaea_core::hdc::binary_hv::BinaryHV;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Symthaea Counterfactual Planning Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    let reasoner = CounterfactualReasoner::new();
    let composer = CounterfactualComposer::new();

    // ── Example 1: Simple causal chain ───────────────────────────────
    println!("── Example 1: Simple causal chain (X → Y) ──");
    let dag1 = CausalDAG::new(vec!["Treatment".into(), "Outcome".into()], vec![(0, 1)]);
    let query1 = CausalQuery {
        treatment: 0,
        outcome: 1,
        conditioning: vec![],
    };
    let outcome1 = reasoner.query(&dag1, &query1);
    print_outcome("Treatment → Outcome", &outcome1);
    println!();

    // ── Example 2: Confounded relationship ───────────────────────────
    println!("── Example 2: Confounded relationship ──");
    println!("  DAG: Confounder → Treatment, Confounder → Outcome, Treatment → Outcome");
    let dag2 = CausalDAG::new(
        vec!["Treatment".into(), "Outcome".into(), "Confounder".into()],
        vec![(2, 0), (2, 1), (0, 1)],
    );
    let query2 = CausalQuery {
        treatment: 0,
        outcome: 1,
        conditioning: vec![],
    };
    let outcome2 = reasoner.query(&dag2, &query2);
    print_outcome("Treatment → Outcome (adjusting for Confounder)", &outcome2);
    println!();

    // ── Example 3: Unidentifiable query ──────────────────────────────
    println!("── Example 3: Disconnected nodes ──");
    let dag3 = CausalDAG::new(
        vec!["A".into(), "B".into()],
        vec![], // no edges
    );
    let query3 = CausalQuery {
        treatment: 0,
        outcome: 1,
        conditioning: vec![],
    };
    let outcome3 = reasoner.query(&dag3, &query3);
    print_outcome("A ⊥ B (no causal path)", &outcome3);
    println!();

    // ── Example 4: Counterfactual composition ────────────────────────
    println!("── Example 4: Composed counterfactual analysis ──");
    let analysis = composer.analyze(&dag1, &query1);
    println!("  Actionable:   {}", analysis.actionable);
    println!("  Uncertainty:  {:.3}", analysis.uncertainty);
    println!("  Narrative:    {}", analysis.narrative);
    println!();

    // ── Example 5: Semantic role substitution ────────────────────────
    println!("── Example 5: Role substitution ──");
    println!("  \"What if a different agent performed the same action?\"");
    let agent_a = BinaryHV::random(100);
    let agent_b = BinaryHV::random(200);
    let context = BinaryHV::random(300);

    // Create causal vector with agent_a
    let causal_vector = context.bind(&agent_a);

    // Substitute agent_b for agent_a
    let sub = RoleSubstitution::new(SemanticRole::Agent, agent_a, agent_b);
    let substituted = sub.apply(&causal_vector);

    // Verify preservation
    let preservation = sub.verify_preservation(&causal_vector, &substituted);
    println!("  Original agent:      Agent A");
    println!("  Substituted agent:   Agent B");
    println!("  Structural similarity: {:.4}", preservation);
    println!("  Vectors differ:      {}", causal_vector != substituted);
    println!();

    // ── Example 6: Reference harness validation ──────────────────────
    println!("── Example 6: Reference harness validation ──");
    let mut harness = CausalReferenceHarness::new();
    let test_count = harness.test_count();
    let result = harness.validate(&reasoner);
    println!("  Harness result: {:?}", result);
    println!(
        "  Test cases:     {} (DAGs ≤8 nodes, brute-force verified)",
        test_count
    );
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Demo complete. Causal queries return honest outcomes:");
    println!("  Identified / Unidentified / AssumptionRequired");
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(feature = "reasoning_engine")]
fn print_outcome(
    label: &str,
    outcome: &symthaea::consciousness::counterfactual::CausalQueryOutcome,
) {
    use symthaea::consciousness::counterfactual::CausalQueryOutcome;
    match outcome {
        CausalQueryOutcome::Identified {
            method,
            confidence,
            estimand,
        } => {
            println!("  Query:  {}", label);
            println!("  Result: IDENTIFIED via {:?}", method);
            println!("  Conf:   {:.2}", confidence);
            println!("  Est:    {}", estimand.description);
        }
        CausalQueryOutcome::Unidentified {
            reason,
            suggestions,
            ..
        } => {
            println!("  Query:  {}", label);
            println!("  Result: UNIDENTIFIED ({:?})", reason);
            println!("  Hints:  {}", suggestions.join("; "));
        }
        CausalQueryOutcome::AssumptionRequired {
            assumption,
            plausibility,
            ..
        } => {
            println!("  Query:  {}", label);
            println!("  Result: ASSUMPTION REQUIRED");
            println!("  Cond:   {}", assumption.condition);
            println!("  Plaus:  {:.2}", plausibility);
        }
    }
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("This example requires the 'reasoning_engine' feature:");
    eprintln!("  cargo run --example counterfactual_planning --features reasoning_engine");
}
