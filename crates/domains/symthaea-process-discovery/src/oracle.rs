// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layered gates, cheapest first (mirrors `symthaea-forge/src/fitness.rs`'s
//! "cheap gate first" ordering): validity -> scope -> stability (advisory).
//!
//! **Scope cut, stated explicitly** (per the Phase 1 plan): no
//! quantum-chemistry feasibility gate here. It needs 3D atom geometry this
//! crate has no way to generate (Phase 0's audit hand-built every geometry
//! from literature values) -- rather than fake or silently skip it, the
//! chain stops at the stability estimate. Quantum-chemistry feasibility is a
//! named Phase 2 blocker: it needs a 2D-graph -> 3D-geometry embedder that
//! doesn't exist anywhere in this codebase yet.
//!
//! **The stability estimate does not gate (Phase 1.1 fix).** An external
//! review pointed out that `symthaea_materials::compound_stability`'s model
//! is mathematically guaranteed to never reject a multi-element composition:
//! `formation_energy` is `-(weighted variance) <= 0` by construction and
//! `mixing_entropy` is always `>= 0` for `0 < x < 1`, so
//! `free_energy = formation_energy - T * mixing_entropy` is always `<= 0`
//! for any candidate with two or more distinct elements -- `is_stable`
//! cannot ever come back `false` here, regardless of the actual candidate.
//! That's not a threshold-tuning problem; there is no threshold that fixes
//! a model whose sign is fixed by its own formula. It also wasn't built for
//! organic covalent chemistry in the first place (a Miedema-style
//! alloy/mixing model). Kept as informational telemetry -- real numbers,
//! clearly labeled as non-gating -- rather than removed outright, since the
//! formation-energy magnitude may still be a useful ranking signal even
//! though it can't reject anything.

use crate::normalization::{self, NormalizedCandidate};
use crate::policy::ScopePolicy;
use crate::types::ReactionCandidate;
use crate::validity;
use symthaea_materials::compound_stability::{StabilityPrediction, predict_stability};
use symthaea_organic_chemistry::element;
use symthaea_organic_chemistry::smiles::Molecule;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateOutcome {
    Passed,
    FailedValidity(String),
    FailedScope(String),
}

impl GateOutcome {
    pub fn passed(&self) -> bool {
        matches!(self, GateOutcome::Passed)
    }
}

#[derive(Debug, Clone)]
pub struct OracleResult {
    pub outcome: GateOutcome,
    /// The scope policy's own justification, kept regardless of pass/fail
    /// (empty if validity failed before scope ever ran) -- so a certificate
    /// can show *why* a candidate was allowed, not just that it was.
    pub scope_reason: String,
    /// Advisory only (see module doc) -- populated whenever a candidate
    /// reaches this stage, regardless of the values. Never gates.
    pub composition_model_prediction: Vec<StabilityPrediction>,
    /// The candidate that was actually validated/scoped -- after
    /// normalization (see `normalization.rs`'s module doc), not the raw
    /// input. This is what `certificate.rs` builds structure records from,
    /// with the normalization records attached as evidence.
    pub normalized: NormalizedCandidate,
}

/// Element composition (atomic_number, mole_fraction) for one molecule,
/// heavy atoms + implicit hydrogens both counted -- the input shape
/// `symthaea_materials::compound_stability::predict_stability` expects.
fn composition(m: &Molecule) -> Vec<(u16, f64)> {
    let mut counts: Vec<(u16, u32)> = Vec::new();
    let mut bump = |z: u16, n: u32| {
        if let Some(entry) = counts.iter_mut().find(|(zz, _)| *zz == z) {
            entry.1 += n;
        } else {
            counts.push((z, n));
        }
    };
    for atom in &m.atoms {
        if let Some(e) = element::lookup(atom.element) {
            bump(e.atomic_number as u16, 1);
            if atom.hydrogens > 0 {
                bump(1, atom.hydrogens as u32); // H, Z=1
            }
        }
    }
    let total: u32 = counts.iter().map(|(_, n)| *n).sum();
    if total == 0 {
        return vec![];
    }
    counts
        .into_iter()
        .map(|(z, n)| (z, n as f64 / total as f64))
        .collect()
}

/// Run a candidate through normalization, then validity, then the scope
/// policy. The composition stability estimate is always computed for any
/// candidate that reaches this point (for its own sake as telemetry) but
/// never changes the outcome.
///
/// **Stage 0: normalization** (added for the Reaction Corpus Auditor's
/// Phase A.1 hardening pass). Runs before validity, not as a separate
/// optional step some callers could skip -- every `ReactionTemplate` output
/// flows through this single function (the threat model's "single most
/// important assumption to preserve"), so wiring normalization in here
/// means both `search.rs`'s generator and `audit.rs`'s auditor get it
/// automatically, with no risk of a caller bypassing it.
pub fn evaluate(candidate: &ReactionCandidate, policy: &dyn ScopePolicy) -> OracleResult {
    let normalized = normalization::normalize_candidate(candidate);

    if let Err(reason) = validity::check_candidate(&normalized.candidate) {
        return OracleResult {
            outcome: GateOutcome::FailedValidity(reason),
            scope_reason: String::new(), // scope never ran
            composition_model_prediction: vec![],
            normalized,
        };
    }

    let scope_decision = policy.check_candidate(&normalized.candidate);
    if !scope_decision.allowed {
        return OracleResult {
            outcome: GateOutcome::FailedScope(scope_decision.reason.clone()),
            scope_reason: scope_decision.reason,
            composition_model_prediction: vec![],
            normalized,
        };
    }

    let composition_model_prediction: Vec<StabilityPrediction> = normalized
        .candidate
        .products
        .iter()
        .filter_map(|p| {
            let comp = composition(p);
            if comp.len() < 2 {
                None // pure element or empty -- the model doesn't apply
            } else {
                Some(predict_stability(&comp, 298.15))
            }
        })
        .collect();

    OracleResult {
        outcome: GateOutcome::Passed,
        scope_reason: scope_decision.reason,
        composition_model_prediction,
        normalized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{AllowlistOnlyPolicy, ReactantLibrary};
    use crate::types::ReactionCandidate;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn composition_sums_to_one() {
        let m = mol("CCO"); // ethanol: C2H6O
        let comp = composition(&m);
        let total: f64 = comp.iter().map(|(_, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn novel_product_fails_at_scope_gate_after_validity_passes() {
        // Hydrogenation's real product (ethane) isn't a phase0 library
        // member, so AllowlistOnlyPolicy must reject it -- and the failure
        // must come from the scope gate specifically, after validity (a
        // structurally sane candidate) already passed.
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let ethylene = mol("C=C");
        let h2 = crate::templates::molecular_hydrogen();
        let ethane = mol("CC");
        let candidate = ReactionCandidate {
            reactants: vec![ethylene, h2],
            products: vec![ethane],
            template: "hydrogenation",
        };
        let result = evaluate(&candidate, &policy);
        assert!(matches!(result.outcome, GateOutcome::FailedScope(_)));
    }

    #[test]
    fn structurally_invalid_candidate_fails_before_scope() {
        // Reactants/products with different formulas -- an unconserved,
        // structurally invalid "reaction" (as if a template had a bug).
        // Even a permissive AllowlistOnlyPolicy over a library containing
        // BOTH molecules must never see this candidate; validity rejects
        // it first.
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::from_smiles(&["CCO", "CCC"]),
        };
        let candidate = ReactionCandidate {
            reactants: vec![mol("CCO")],
            products: vec![mol("CCC")],
            template: "buggy",
        };
        let result = evaluate(&candidate, &policy);
        assert!(matches!(result.outcome, GateOutcome::FailedValidity(_)));
    }

    #[test]
    fn all_library_members_pass_every_gate() {
        // Hand-built (not template-derived) candidate where every reactant
        // and product is already a library member, isolating the oracle
        // chain's own correctness from the templates' output.
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let ethanol_in = mol("CCO");
        let ethanol_out = mol("CCO");
        let candidate = ReactionCandidate {
            reactants: vec![ethanol_in],
            products: vec![ethanol_out],
            template: "identity",
        };
        let result = evaluate(&candidate, &policy);
        assert_eq!(result.outcome, GateOutcome::Passed);
    }
}
