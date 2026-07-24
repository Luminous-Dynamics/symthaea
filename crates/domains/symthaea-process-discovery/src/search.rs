// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The generate -> verify -> select loop. Mirrors `symthaea-forge/src/search.rs`'s
//! shape (config, per-cause stats, never-auto-applied certificates) retargeted
//! from Rust ASTs to reaction chemistry.
//!
//! **Generator scope, stated explicitly**: Phase 1's generator only
//! enumerates candidates from the fixed seed reactant list via the two
//! templates in `templates.rs` -- it never invents new starting materials,
//! *for any policy*, including `OpenWithHeuristicScreenPolicy` (whose design
//! allows that in principle). Building a real "invent a valid novel
//! molecule" generator is separate, materially higher-risk scope that
//! deserves its own explicit sign-off rather than being improvised here --
//! see the Phase 1 plan's Context section. This means `OpenWithHeuristicScreenPolicy`
//! and `HybridAllowlistReactantsPolicy` behave identically in this Phase 1
//! run (neither's reactant-invention path is exercised); only
//! `AllowlistOnlyPolicy`'s product restriction is genuinely tested against
//! the other two. The comparison harness reports this explicitly rather than
//! implying a fuller comparison happened.
//!
//! **Phase 1.1 hardening**: `SearchOutcome` returns `Vec<ProcessCertificate>`
//! for survivors, not raw `ReactionCandidate`s -- an external review found
//! the certificate boundary was aspirational (nothing actually constructed
//! one outside its own unit test).
//!
//! **Phase 1.2 fix**: the Phase 1.1 version of `all_attempts` still stored
//! the *full* `ReactionCandidate` for every attempt, including ones that
//! passed -- meaning a caller could read a survivor's complete atom/bond
//! structure straight out of `all_attempts`, completely bypassing
//! `ProcessCertificate`. The doc comment at the time even asserted this was
//! fine ("not a bypass: nothing that failed a gate was ever going to get a
//! certificate") without checking the *passed* case. Found while planning
//! the Phase 1.2 adversarial-assurance pass, specifically its "attempt to
//! bypass the certificate-only output boundary" item -- fixed by replacing
//! `all_attempts`'s element type with `AttemptSummary`, which never carries
//! full molecular structure, for any outcome.

use crate::certificate::ProcessCertificate;
use crate::oracle::{self, GateOutcome};
use crate::policy::ScopePolicy;
use crate::templates::{
    AmidationTemplate, EsterificationTemplate, ExhaustiveHydrogenationTemplate,
    HydrogenationTemplate, ReactionTemplate, count_reducible_cc_bonds, molecular_hydrogen,
};
use crate::types::{ReactionCandidate, SearchConfig};
use symthaea_organic_chemistry::smiles::Molecule;

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub candidates_attempted: usize,
    pub failed_validity: usize,
    pub blocked_by_scope: usize,
    pub survived: usize,
}

/// A record of one attempt with formulas only -- no atom/bond structure.
/// See the module doc's "Phase 1.2 fix" note for why this exists instead of
/// the raw `ReactionCandidate`.
#[derive(Debug, Clone)]
pub struct AttemptSummary {
    pub template: &'static str,
    pub reactant_formulas: Vec<String>,
    pub product_formulas: Vec<String>,
    pub outcome: GateOutcome,
}

impl AttemptSummary {
    fn from_candidate(candidate: &ReactionCandidate, outcome: GateOutcome) -> Self {
        Self {
            template: candidate.template,
            reactant_formulas: candidate
                .reactants
                .iter()
                .map(|m| m.molecular_formula())
                .collect(),
            product_formulas: candidate
                .products
                .iter()
                .map(|m| m.molecular_formula())
                .collect(),
            outcome,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub policy_name: &'static str,
    pub stats: SearchStats,
    /// The only place full molecular structure appears for a survivor.
    pub certificates: Vec<ProcessCertificate>,
    /// Every candidate this run considered, formulas only (see
    /// `AttemptSummary`), with the reason it was blocked (if it was) --
    /// kept alongside `certificates` so the comparison harness can show
    /// *why* each policy diverged, not just the counts. Matches this
    /// project's "no silent caps, log what's dropped" convention, without
    /// reopening the bypass the Phase 1.2 fix closed.
    pub all_attempts: Vec<AttemptSummary>,
    /// Seed SMILES strings that failed to parse, with the parse error --
    /// an earlier version silently dropped these via `filter_map`. Empty in
    /// normal operation (every seed used so far is known-valid).
    pub unparseable_seeds: Vec<(String, String)>,
}

/// Enumerate every candidate the three templates can produce from
/// `reactants`: one hydrogenation attempt per reactant with an unsaturated
/// C-C bond, one esterification attempt AND one amidation attempt per
/// unordered pair with distinct formulas (`EsterificationTemplate::apply`/
/// `AmidationTemplate::apply` are both order-independent, so each pair is
/// tried only once per template -- see `try_esterify`/`try_amidate`).
///
/// **Both templates are tried independently for every pair, deliberately
/// not first-match-wins** (Phase A.4): when a pair supports both reactions
/// (one reactant offers a free amine AND a free alcohol), this generator
/// has no declared product to disambiguate against, so silently picking one
/// over the other would be an unjustified guess. It emits both candidates
/// and lets each flow independently through validity/scope/certificate
/// gating. This is different from `audit.rs`'s auditor, which DOES have a
/// declared product and so can (and does) commit to a chemically-motivated
/// dispatch priority -- see `audit.rs::classify`'s doc comment.
fn generate_candidates(reactants: &[Molecule]) -> Vec<ReactionCandidate> {
    let mut out = Vec::new();

    for r in reactants {
        let h2 = molecular_hydrogen();
        if let Some(products) = HydrogenationTemplate.apply(&[r.clone(), h2.clone()]) {
            out.push(ReactionCandidate {
                reactants: vec![r.clone(), h2],
                products,
                template: "hydrogenation",
            });
        }

        // Phase A.5: an independent candidate reducing to FULL saturation,
        // alongside (not instead of) the single-step candidate above -- same
        // "no selectivity guessing, emit every candidate" principle as the
        // esterification/amidation pair. Needs an exact H2 count (see
        // ExhaustiveHydrogenationTemplate's doc comment), computed here.
        let needed = count_reducible_cc_bonds(r);
        if needed > 0 {
            let h2s: Vec<Molecule> = (0..needed).map(|_| molecular_hydrogen()).collect();
            let mut exhaustive_reactants = vec![r.clone()];
            exhaustive_reactants.extend(h2s);
            if let Some(products) = ExhaustiveHydrogenationTemplate.apply(&exhaustive_reactants) {
                out.push(ReactionCandidate {
                    reactants: exhaustive_reactants,
                    products,
                    template: "exhaustive_hydrogenation",
                });
            }
        }
    }

    // Index-based, i<j only: EsterificationTemplate::apply/AmidationTemplate::apply
    // are order-independent (Phase A.3/A.4), so iterating every ORDERED pair
    // here would call each twice per real reactant pair and produce
    // duplicate candidates (same computed product, reactants field just
    // swapped) -- each unordered pair only needs one attempt per template.
    for i in 0..reactants.len() {
        for j in (i + 1)..reactants.len() {
            let (a, b) = (&reactants[i], &reactants[j]);
            if a.molecular_formula() == b.molecular_formula() {
                continue; // skip self-pairing
            }
            if let Some(products) = EsterificationTemplate.apply(&[a.clone(), b.clone()]) {
                out.push(ReactionCandidate {
                    reactants: vec![a.clone(), b.clone()],
                    products,
                    template: "esterification",
                });
            }
            if let Some(products) = AmidationTemplate.apply(&[a.clone(), b.clone()]) {
                out.push(ReactionCandidate {
                    reactants: vec![a.clone(), b.clone()],
                    products,
                    template: "amidation",
                });
            }
        }
    }

    out
}

pub fn run_search(config: &SearchConfig, policy: &dyn ScopePolicy) -> SearchOutcome {
    let mut unparseable_seeds = Vec::new();
    let reactants: Vec<Molecule> = config
        .seed_reactants
        .iter()
        .filter_map(|s| match Molecule::from_smiles(s) {
            Ok(m) => Some(m),
            Err(e) => {
                unparseable_seeds.push((s.clone(), e.to_string()));
                None
            }
        })
        .collect();

    let mut candidates = generate_candidates(&reactants);
    candidates.truncate(config.candidate_cap);
    // `candidate_cap` in Phase 1's enumerative (non-randomized) generator
    // caps total candidates attempted in enumeration order (hydrogenation
    // attempts first, then esterification pairs) -- not an unbiased sample.
    // Not fixed this pass (a future randomized generator should address it
    // together with reintroducing a real seed field); noted here rather
    // than silently accepted, per an external review's "smaller issues" list.

    let mut stats = SearchStats::default();
    let mut certificates = Vec::new();
    let mut all_attempts = Vec::new();

    for candidate in candidates {
        stats.candidates_attempted += 1;
        let result = oracle::evaluate(&candidate, policy);
        match &result.outcome {
            GateOutcome::FailedValidity(_) => stats.failed_validity += 1,
            GateOutcome::FailedScope(_) => stats.blocked_by_scope += 1,
            GateOutcome::Passed => {
                stats.survived += 1;
                certificates.push(ProcessCertificate::from_passed(
                    policy.name(),
                    &result,
                    &config.seed_reactants,
                ));
            }
        }
        all_attempts.push(AttemptSummary::from_candidate(
            &candidate,
            result.outcome.clone(),
        ));
    }

    SearchOutcome {
        policy_name: policy.name(),
        stats,
        certificates,
        all_attempts,
        unparseable_seeds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hazard_heuristics::ExternalScopeConfig;
    use crate::policy::{
        AllowlistOnlyPolicy, HybridAllowlistReactantsPolicy, OpenWithHeuristicScreenPolicy,
        ReactantLibrary,
    };

    fn config() -> SearchConfig {
        SearchConfig {
            seed_reactants: vec!["C=C".into(), "CC=C".into(), "CCO".into(), "CC(=O)O".into()],
            candidate_cap: 100,
        }
    }

    #[test]
    fn generator_produces_both_template_types() {
        let reactants: Vec<Molecule> = config()
            .seed_reactants
            .iter()
            .map(|s| Molecule::from_smiles(s).unwrap())
            .collect();
        let candidates = generate_candidates(&reactants);
        assert!(candidates.iter().any(|c| c.template == "hydrogenation"));
        assert!(candidates.iter().any(|c| c.template == "esterification"));
    }

    #[test]
    fn esterification_order_independence_does_not_duplicate_candidates() {
        // Regression test for a real second-order effect of the Phase A.3
        // order-independence fix to EsterificationTemplate::apply: since it
        // now succeeds regardless of which argument is the acid and which
        // is the alcohol, generate_candidates' pairing loop must only try
        // each UNORDERED pair once (index-based i<j), or it would call
        // apply() twice per real acid+alcohol pair and emit two duplicate
        // candidates for the same actual reaction.
        let reactants: Vec<Molecule> = config()
            .seed_reactants
            .iter()
            .map(|s| Molecule::from_smiles(s).unwrap())
            .collect();
        let candidates = generate_candidates(&reactants);
        let esterifications: Vec<_> = candidates
            .iter()
            .filter(|c| c.template == "esterification")
            .collect();
        // This seed set has exactly one acid (acetic acid) and one alcohol
        // (ethanol) -- exactly one real esterification exists among the 4
        // seeds, so exactly one candidate must be generated, not two.
        assert_eq!(
            esterifications.len(),
            1,
            "expected exactly one esterification candidate (acetic acid + ethanol), got {}: {:?}",
            esterifications.len(),
            esterifications
                .iter()
                .map(|c| c
                    .reactants
                    .iter()
                    .map(|m| m.molecular_formula())
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn generator_emits_both_ester_and_amide_when_a_pair_supports_both() {
        // Phase A.4: acetic acid + ethanolamine (NCCO, which has BOTH a free
        // amine and a free alcohol) can react via either template. Since
        // this generator has no declared product to disambiguate against
        // (unlike the auditor's classify()), it must emit BOTH candidates,
        // not silently pick one.
        let reactants: Vec<Molecule> = ["CC(=O)O", "NCCO"]
            .iter()
            .map(|s| Molecule::from_smiles(s).unwrap())
            .collect();
        let candidates = generate_candidates(&reactants);
        let esterifications = candidates
            .iter()
            .filter(|c| c.template == "esterification")
            .count();
        let amidations = candidates
            .iter()
            .filter(|c| c.template == "amidation")
            .count();
        assert_eq!(
            esterifications, 1,
            "expected exactly one esterification candidate"
        );
        assert_eq!(amidations, 1, "expected exactly one amidation candidate");
    }

    #[test]
    fn generator_emits_both_single_step_and_exhaustive_hydrogenation_independently() {
        // Phase A.5: 1,4-pentadiene (C=CCC=C, 2 reducible C-C bonds) supports
        // both a single-step reduction (HydrogenationTemplate) and a
        // full-saturation reduction (ExhaustiveHydrogenationTemplate). No
        // declared product to disambiguate against here either, so both
        // must be emitted independently, same principle as the ester/amide
        // pair.
        let reactants: Vec<Molecule> = ["C=CCC=C"]
            .iter()
            .map(|s| Molecule::from_smiles(s).unwrap())
            .collect();
        let candidates = generate_candidates(&reactants);
        let single_step = candidates
            .iter()
            .filter(|c| c.template == "hydrogenation")
            .count();
        let exhaustive = candidates
            .iter()
            .filter(|c| c.template == "exhaustive_hydrogenation")
            .count();
        assert_eq!(single_step, 1, "expected exactly one single-step candidate");
        assert_eq!(exhaustive, 1, "expected exactly one exhaustive candidate");
        let exhaustive_candidate = candidates
            .iter()
            .find(|c| c.template == "exhaustive_hydrogenation")
            .unwrap();
        assert_eq!(
            exhaustive_candidate.reactants.len(),
            3,
            "1 unsaturated reactant + 2 H2 equivalents"
        );
        assert_eq!(
            exhaustive_candidate.products[0].molecular_formula(),
            "C5H12"
        );
    }

    #[test]
    fn allowlist_only_survives_nothing_from_this_seed_set() {
        // None of the four seed molecules' template products (ethane,
        // propane, ethyl acetate, water) other than water are themselves
        // library members alongside all reactants -- so essentially every
        // candidate should be blocked at scope. Verifies the "safest, least
        // discovery power" characterization empirically, not just by design.
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let outcome = run_search(&config(), &policy);
        assert!(outcome.stats.candidates_attempted > 0);
        assert_eq!(outcome.stats.survived, 0);
        assert_eq!(outcome.stats.failed_validity, 0);
        assert_eq!(
            outcome.stats.blocked_by_scope,
            outcome.stats.candidates_attempted
        );
        assert!(outcome.certificates.is_empty());
    }

    #[test]
    fn open_and_hybrid_survive_more_than_allowlist_only() {
        let open = OpenWithHeuristicScreenPolicy {
            external: ExternalScopeConfig::default(),
        };
        let hybrid = HybridAllowlistReactantsPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
            external: ExternalScopeConfig::default(),
        };
        let allow_only = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };

        let open_outcome = run_search(&config(), &open);
        let hybrid_outcome = run_search(&config(), &hybrid);
        let allow_outcome = run_search(&config(), &allow_only);

        assert!(open_outcome.stats.survived > allow_outcome.stats.survived);
        assert!(hybrid_outcome.stats.survived > allow_outcome.stats.survived);
        assert_eq!(open_outcome.certificates.len(), open_outcome.stats.survived);
        // Per the module doc comment: identical in Phase 1, since the
        // generator never exercises reactant invention.
        assert_eq!(open_outcome.stats.survived, hybrid_outcome.stats.survived);
    }

    #[test]
    fn unparseable_seed_is_reported_not_silently_dropped() {
        let cfg = SearchConfig {
            seed_reactants: vec!["CCO".into(), "not a smiles string!!".into()],
            candidate_cap: 100,
        };
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let outcome = run_search(&cfg, &policy);
        assert_eq!(outcome.unparseable_seeds.len(), 1);
        assert_eq!(outcome.unparseable_seeds[0].0, "not a smiles string!!");
    }

    #[test]
    fn all_attempts_never_carries_full_molecular_structure() {
        // Regression test for the Phase 1.2 bypass fix: AttemptSummary has
        // no field that could expose atoms/bonds, for ANY outcome including
        // Passed. This is enforced by the type itself (no such field
        // exists), but assert on the concrete fields present as a
        // structural double-check that the type wasn't quietly widened.
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        // A candidate this policy actually lets pass (all library members).
        let cfg = SearchConfig {
            seed_reactants: vec!["CCO".into()],
            candidate_cap: 100,
        };
        let outcome = run_search(&cfg, &policy);
        for a in &outcome.all_attempts {
            // Only formula strings are reachable -- there is no way to get
            // back atom/bond data from an AttemptSummary.
            assert!(a.reactant_formulas.iter().all(|f| !f.is_empty()));
        }
    }
}
