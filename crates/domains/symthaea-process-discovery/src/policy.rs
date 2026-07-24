// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `ScopePolicy`: the guardrail. Three implementations, compared empirically
//! by `examples/policy_comparison.rs` rather than this plan picking a
//! winner in advance -- see `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`
//! Phase 1.

use crate::hazard_heuristics::{ExternalScopeConfig, score};
use crate::isomorphism::is_isomorphic;
use crate::types::{ReactionCandidate, ScopeDecision};
use std::collections::HashMap;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

pub trait ScopePolicy: Send + Sync {
    fn name(&self) -> &'static str;
    fn check_reactant(&self, mol: &Molecule) -> ScopeDecision;
    fn check_candidate(&self, candidate: &ReactionCandidate) -> ScopeDecision;
}

fn bond_order_code(o: BondOrder) -> char {
    match o {
        BondOrder::Single => '-',
        BondOrder::Double => '=',
        BondOrder::Triple => '#',
        BondOrder::Aromatic => ':',
    }
}

/// A one-round structural invariant for `m`: per atom, `(element, hydrogens,
/// charge, sorted (bond_order, neighbor_element) pairs)`, joined and sorted
/// so atom insertion order doesn't affect the result, prefixed by the
/// molecular formula.
///
/// **Deliberately not named/treated as "canonical."** A second review pass
/// correctly pushed back on the first review's own fix here: this is
/// structural identity with a bounded current representation, not a
/// collision-free canonical graph identity. It's one round of local graph
/// refinement (each atom's own label + its immediate neighbors), not full
/// canonicalization (iterated Morgan/Weisfeiler-Leman to a fixed point, or a
/// real canonical-SMILES writer -- `organic-chemistry` has neither). It
/// correctly distinguishes simple constitutional isomers -- verified here
/// against ethanol (CCO) vs. dimethyl ether (COC), same formula C2H6O,
/// different connectivity -- but larger or more symmetric molecules can in
/// principle share a one-round invariant while being genuinely distinct
/// graphs. **Phase 1.2: now used as a bucket index, not the equality test
/// itself.** `ReactantLibrary` groups molecules by this key and resolves any
/// collision within a bucket via [`crate::isomorphism::is_isomorphic`] --
/// the two-stage design this comment used to describe as future work. See
/// `isomorphism.rs` for a real, constructed collision this key alone cannot
/// distinguish (the triangular-prism graph vs. K3,3) and confirmation that
/// the two-stage lookup resolves it correctly.
fn structural_key(m: &Molecule) -> String {
    let mut atom_keys: Vec<String> = m
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let mut neighbor_codes: Vec<String> = m
                .neighbors(i)
                .iter()
                .map(|(j, order)| format!("{}{}", bond_order_code(*order), m.atoms[*j].element))
                .collect();
            neighbor_codes.sort();
            format!(
                "{}h{}c{}[{}]",
                atom.element,
                atom.hydrogens,
                atom.charge,
                neighbor_codes.join(",")
            )
        })
        .collect();
    atom_keys.sort();
    format!("{}|{}", m.molecular_formula(), atom_keys.join(";"))
}

/// A curated set of allowed molecules. Two-stage membership test (Phase
/// 1.2): [`structural_key`] buckets candidates for a cheap lookup, and any
/// molecule sharing a bucket with a library member is resolved by exact
/// [`is_isomorphic`] rather than trusted on the key alone -- closing the
/// collision `structural_key`'s doc comment documents (verified: the
/// triangular-prism graph and K3,3 collide on the key but `is_isomorphic`
/// correctly tells them apart, see the test below and `isomorphism.rs`).
/// Previously compared by molecular formula alone, which a review correctly
/// pointed out would treat any structural isomer of a library member as
/// that member.
#[derive(Debug, Clone)]
pub struct ReactantLibrary {
    buckets: HashMap<String, Vec<Molecule>>,
}

impl ReactantLibrary {
    pub fn from_smiles(entries: &[&str]) -> Self {
        let mut buckets: HashMap<String, Vec<Molecule>> = HashMap::new();
        for m in entries.iter().filter_map(|s| Molecule::from_smiles(s).ok()) {
            buckets.entry(structural_key(&m)).or_default().push(m);
        }
        Self { buckets }
    }

    pub fn contains(&self, mol: &Molecule) -> bool {
        match self.buckets.get(&structural_key(mol)) {
            Some(bucket) => bucket.iter().any(|candidate| is_isomorphic(mol, candidate)),
            None => false,
        }
    }

    /// The 10 feedstocks validated in Phase 0
    /// (`symthaea-organic-chemistry/examples/phase0_audit.rs`), plus water
    /// and H2 (needed as reaction partners for the two Phase 1 templates).
    pub fn phase0_feedstocks() -> Self {
        Self::from_smiles(&[
            "C=C",              // ethylene
            "CC=C",             // propylene
            "c1ccccc1",         // benzene
            "CCO",              // ethanol
            "CC(=O)O",          // acetic acid
            "CO",               // methanol
            "C=CC#N",           // acrylonitrile
            "c1ccc(cc1)O",      // phenol
            "OC(=O)CCCCC(=O)O", // adipic acid
            "O=C1CCCCCN1",      // caprolactam
            "O",                // water
            "[H][H]",           // H2 (explicit-H2 bracket form)
        ])
    }
}

/// Policy 1: reactants AND products must both be library members. The
/// generator can only verify/rank *known* reaction pairs -- it can never
/// produce a genuinely new molecule under this policy. Safest, least
/// discovery power.
pub struct AllowlistOnlyPolicy {
    pub library: ReactantLibrary,
}

impl ScopePolicy for AllowlistOnlyPolicy {
    fn name(&self) -> &'static str {
        "allowlist-only"
    }

    fn check_reactant(&self, mol: &Molecule) -> ScopeDecision {
        if self.library.contains(mol) {
            ScopeDecision::allow("reactant is a library member")
        } else {
            ScopeDecision::deny("reactant not in the curated library")
        }
    }

    fn check_candidate(&self, candidate: &ReactionCandidate) -> ScopeDecision {
        for r in &candidate.reactants {
            let d = self.check_reactant(r);
            if !d.allowed {
                return d;
            }
        }
        for p in &candidate.products {
            if !self.library.contains(p) {
                return ScopeDecision::deny(format!(
                    "product {} not in the curated library -- allowlist-only cannot produce new molecules",
                    p.molecular_formula()
                ));
            }
        }
        ScopeDecision::allow("all reactants and products are library members")
    }
}

/// Policy 2: any valid H/C/N/O/F structure is allowed as a reactant; every
/// candidate is screened by the generic hazard heuristics before a
/// certificate is ever written. Broadest search space, materially higher
/// risk, only as reliable as `hazard_heuristics.rs`.
pub struct OpenWithHeuristicScreenPolicy {
    pub external: ExternalScopeConfig,
}

impl ScopePolicy for OpenWithHeuristicScreenPolicy {
    fn name(&self) -> &'static str {
        "open+heuristic-screen"
    }

    fn check_reactant(&self, mol: &Molecule) -> ScopeDecision {
        let s = score(mol);
        if s.exceeds_conservative_threshold() {
            ScopeDecision::deny(format!("reactant flagged by hazard heuristics: {s:?}"))
        } else {
            ScopeDecision::allow("reactant passed hazard heuristics")
        }
    }

    fn check_candidate(&self, candidate: &ReactionCandidate) -> ScopeDecision {
        for r in &candidate.reactants {
            let d = self.check_reactant(r);
            if !d.allowed {
                return d;
            }
        }
        for p in &candidate.products {
            let s = score(p);
            if s.exceeds_conservative_threshold() {
                return ScopeDecision::deny(format!(
                    "product {} flagged by hazard heuristics: {s:?}",
                    p.molecular_formula()
                ));
            }
        }
        if self.external.extra_patterns_path.is_some() {
            // Extension point wired but no matcher implemented -- see
            // ExternalScopeConfig doc comment. Not reached in Phase 1 since
            // nothing populates this path yet.
            return ScopeDecision::deny(
                "external pattern reference configured but no matcher implemented in Phase 1",
            );
        }
        ScopeDecision::allow("all reactants and products passed hazard heuristics")
    }
}

/// Policy 3: reactants restricted to the curated library (can't invent new
/// starting materials), but products are open -- so genuinely novel product
/// molecules can appear. Screened by the same hazard heuristics as defense
/// in depth. Closest to how real process chemistry works (fixed feedstocks,
/// novel products).
pub struct HybridAllowlistReactantsPolicy {
    pub library: ReactantLibrary,
    pub external: ExternalScopeConfig,
}

impl ScopePolicy for HybridAllowlistReactantsPolicy {
    fn name(&self) -> &'static str {
        "hybrid-allowlist-reactants"
    }

    fn check_reactant(&self, mol: &Molecule) -> ScopeDecision {
        if self.library.contains(mol) {
            ScopeDecision::allow("reactant is a library member")
        } else {
            ScopeDecision::deny("reactant not in the curated library")
        }
    }

    fn check_candidate(&self, candidate: &ReactionCandidate) -> ScopeDecision {
        for r in &candidate.reactants {
            let d = self.check_reactant(r);
            if !d.allowed {
                return d;
            }
        }
        for p in &candidate.products {
            let s = score(p);
            if s.exceeds_conservative_threshold() {
                return ScopeDecision::deny(format!(
                    "product {} flagged by hazard heuristics: {s:?}",
                    p.molecular_formula()
                ));
            }
        }
        if self.external.extra_patterns_path.is_some() {
            return ScopeDecision::deny(
                "external pattern reference configured but no matcher implemented in Phase 1",
            );
        }
        ScopeDecision::allow("reactants are library members, products passed hazard heuristics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::{EsterificationTemplate, ReactionTemplate};

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    fn candidate(reactants: Vec<Molecule>, products: Vec<Molecule>) -> ReactionCandidate {
        ReactionCandidate {
            reactants,
            products,
            template: "test",
        }
    }

    #[test]
    fn allowlist_only_allows_known_reaction() {
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        // acetic acid + ethanol -> ethyl acetate + water is NOT in the
        // library (ethyl acetate isn't a seed feedstock) -- must be denied.
        let acid = mol("CC(=O)O");
        let alcohol = mol("CCO");
        let products = EsterificationTemplate
            .apply(&[acid.clone(), alcohol.clone()])
            .unwrap();
        let c = candidate(vec![acid, alcohol], products);
        let decision = policy.check_candidate(&c);
        assert!(
            !decision.allowed,
            "allowlist-only must reject a novel product"
        );
    }

    #[test]
    fn allowlist_only_rejects_unknown_reactant() {
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let unknown = mol("CCCCCCCC"); // octane, not in the library
        let decision = policy.check_reactant(&unknown);
        assert!(!decision.allowed);
    }

    #[test]
    fn open_with_heuristic_screen_allows_benign_and_blocks_nitro() {
        let policy = OpenWithHeuristicScreenPolicy {
            external: ExternalScopeConfig::default(),
        };
        let benign = mol("CCO");
        assert!(policy.check_reactant(&benign).allowed);

        let nitro = mol("CN(=O)=O");
        assert!(!policy.check_reactant(&nitro).allowed);
    }

    #[test]
    fn structural_library_distinguishes_isomers() {
        // Ethanol vs. dimethyl ether: same formula (C2H6O), different
        // connectivity. A formula-only library would wrongly treat one as
        // the other -- the exact gap an external review flagged.
        let library = ReactantLibrary::from_smiles(&["CCO"]); // ethanol only
        let ethanol = mol("CCO");
        let dimethyl_ether = mol("COC");
        assert!(library.contains(&ethanol));
        assert!(
            !library.contains(&dimethyl_ether),
            "dimethyl ether must NOT be treated as ethanol just because they share a formula"
        );
    }

    #[test]
    fn library_resolves_structural_key_collision_via_isomorphism() {
        // The real, constructed collision `isomorphism.rs` exists to
        // resolve: the triangular-prism graph and K3,3 are both connected,
        // 3-regular, all-carbon, all-single-bond, 6-atom graphs -- they
        // share a `structural_key` (verified in isomorphism.rs) but are NOT
        // the same molecule. A library containing only the prism graph must
        // not report K3,3 as a member just because they land in the same
        // bucket.
        use crate::isomorphism::{k33_graph, prism_graph};
        let prism = prism_graph();
        let k33 = k33_graph();

        let mut library = ReactantLibrary::from_smiles(&[]);
        library
            .buckets
            .entry(super::structural_key(&prism))
            .or_default()
            .push(prism.clone());

        assert!(
            library.contains(&prism),
            "the library must recognize its own member"
        );
        assert!(
            !library.contains(&k33),
            "K3,3 must not be falsely admitted just because it collides with the prism graph's structural_key"
        );
    }

    #[test]
    fn hybrid_allows_novel_product_from_library_reactants() {
        let policy = HybridAllowlistReactantsPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
            external: ExternalScopeConfig::default(),
        };
        let acid = mol("CC(=O)O");
        let alcohol = mol("CCO");
        let products = EsterificationTemplate
            .apply(&[acid.clone(), alcohol.clone()])
            .unwrap();
        let c = candidate(vec![acid, alcohol], products);
        let decision = policy.check_candidate(&c);
        assert!(
            decision.allowed,
            "hybrid must allow a novel benign product from library reactants: {decision:?}"
        );
    }

    #[test]
    fn hybrid_rejects_non_library_reactant() {
        let policy = HybridAllowlistReactantsPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
            external: ExternalScopeConfig::default(),
        };
        let unknown = mol("CCCCCCCC");
        assert!(!policy.check_reactant(&unknown).allowed);
    }
}
