// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 1 comparison harness for `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`.
//!
//! Not part of the library API. Runs the SAME seed reactant set through all
//! three `ScopePolicy` implementations and prints a side-by-side comparison
//! -- the deliverable "let's do all three and compare" actually asked for,
//! not a single chosen design.
//!
//! **Phase 1.1 hardening**: consumes `outcome.certificates` (the actual
//! survivor output), not raw candidates -- an external review found the
//! prior version bypassed `ProcessCertificate` entirely.

use symthaea_process_discovery::hazard_heuristics::ExternalScopeConfig;
use symthaea_process_discovery::oracle::GateOutcome;
use symthaea_process_discovery::policy::{
    AllowlistOnlyPolicy, HybridAllowlistReactantsPolicy, OpenWithHeuristicScreenPolicy,
    ReactantLibrary, ScopePolicy,
};
use symthaea_process_discovery::search::run_search;
use symthaea_process_discovery::types::SearchConfig;

fn seed_config() -> SearchConfig {
    SearchConfig {
        seed_reactants: vec![
            "C=C".into(),              // ethylene
            "CC=C".into(),             // propylene
            "c1ccccc1".into(),         // benzene
            "CCO".into(),              // ethanol
            "CC(=O)O".into(),          // acetic acid
            "CO".into(),               // methanol
            "C=CC#N".into(),           // acrylonitrile
            "c1ccc(cc1)O".into(),      // phenol
            "OC(=O)CCCCC(=O)O".into(), // adipic acid
            "O=C1CCCCCN1".into(),      // caprolactam
        ],
        candidate_cap: 200,
    }
}

fn run_and_report(label: &str, policy: &dyn ScopePolicy) {
    let outcome = run_search(&seed_config(), policy);
    println!("=== {label} ({}) ===", policy.name());
    println!(
        "  attempted={}  failed_validity={}  blocked_by_scope={}  survived={}",
        outcome.stats.candidates_attempted,
        outcome.stats.failed_validity,
        outcome.stats.blocked_by_scope,
        outcome.stats.survived
    );
    if !outcome.unparseable_seeds.is_empty() {
        for (s, e) in &outcome.unparseable_seeds {
            println!("    UNPARSEABLE SEED: {s:?} -- {e}");
        }
    }
    for cert in &outcome.certificates {
        println!("    CERTIFICATE {}", cert.summary());
        for note in &cert.composition_model_notes {
            println!("      composition model (advisory only): {note}");
        }
    }
    let scope_blocked_examples: Vec<_> = outcome
        .all_attempts
        .iter()
        .filter(|a| matches!(a.outcome, GateOutcome::FailedScope(_)))
        .take(3)
        .collect();
    for attempt in scope_blocked_examples {
        if let GateOutcome::FailedScope(reason) = &attempt.outcome {
            println!(
                "    blocked example [{}] -> {}: {reason}",
                attempt.template,
                attempt.product_formulas.join(" + ")
            );
        }
    }
    println!();
}

fn main() {
    let allow_only = AllowlistOnlyPolicy {
        library: ReactantLibrary::phase0_feedstocks(),
    };
    let open = OpenWithHeuristicScreenPolicy {
        external: ExternalScopeConfig::default(),
    };
    let hybrid = HybridAllowlistReactantsPolicy {
        library: ReactantLibrary::phase0_feedstocks(),
        external: ExternalScopeConfig::default(),
    };

    run_and_report("Policy 1: allowlist-only", &allow_only);
    run_and_report("Policy 2: open+heuristic-screen", &open);
    run_and_report("Policy 3: hybrid-allowlist-reactants", &hybrid);

    println!("=== Honest caveat ===");
    println!("Phase 1's generator only enumerates candidates from the fixed seed list above,");
    println!("via 2 reaction templates, for ALL THREE policies -- it never invents new");
    println!("reactants, even under open+heuristic-screen's design. So policies 2 and 3 are");
    println!("expected to behave identically here; the real comparison this run demonstrates is");
    println!("allowlist-only's product restriction vs. the two open-product policies. A");
    println!("generator that actually invents novel starting materials is separate, higher-risk");
    println!("scope, deliberately not built without further explicit sign-off (see search.rs).");
    println!();
    println!("Every survivor above is a ProcessCertificate -- full structural detail (atoms/");
    println!("bonds, not just formula), gate evidence, and the composition-model estimate marked");
    println!("advisory-only (it cannot reject anything by construction, see oracle.rs). Call");
    println!(".to_json_pretty() on any certificate for the full serialized record.");
}
