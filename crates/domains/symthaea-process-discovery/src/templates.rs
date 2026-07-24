// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal reaction-template engine.
//!
//! `symthaea-organic-chemistry` has no bond-forming/breaking operators at
//! all -- this is genuinely new code. Deliberately kept to two templates for
//! Phase 1: enough to make the three-`ScopePolicy` comparison meaningful
//! without combinatorial explosion or ambiguous transformation semantics.

use symthaea_organic_chemistry::smiles::{Atom, Bond, BondOrder, Molecule};

/// One reaction transformation: takes 1-2 reactants, returns products (and a
/// template name for `ReactionCandidate`) or `None` if the reactants don't
/// match this template's required functional groups.
pub trait ReactionTemplate: Send + Sync {
    fn name(&self) -> &'static str;
    fn apply(&self, reactants: &[Molecule]) -> Option<Vec<Molecule>>;
}

/// Standalone water molecule (O with 2 implicit H) -- constructed directly
/// rather than parsed, since it's a fixed byproduct of `EsterificationTemplate`.
fn water() -> Molecule {
    Molecule {
        atoms: vec![Atom {
            element: "O",
            aromatic: false,
            charge: 0,
            hydrogens: 2,
        }],
        bonds: vec![],
    }
}

/// Molecular hydrogen. `organic-chemistry`'s `Molecule` models heavy atoms
/// only (hydrogens are an implicit per-atom count) -- H2 has zero heavy
/// atoms and genuinely cannot be represented that way. This constructs H2 as
/// two explicit `Atom { element: "H", .. }` nodes bonded together, which
/// works fine for `molecular_formula()`/`molecular_weight()` even though it
/// bends the "atoms are never hydrogen" convention documented on
/// `smiles::Molecule` -- a deliberate, local, honest exception for this one
/// molecule, not a modification to that crate.
fn h2() -> Molecule {
    Molecule {
        atoms: vec![
            Atom {
                element: "H",
                aromatic: false,
                charge: 0,
                hydrogens: 0,
            },
            Atom {
                element: "H",
                aromatic: false,
                charge: 0,
                hydrogens: 0,
            },
        ],
        bonds: vec![Bond {
            a: 0,
            b: 1,
            order: BondOrder::Single,
        }],
    }
}

/// Find a carboxyl carbon: `C` with one `C=O` neighbor and one `C-O(H)`
/// neighbor. Returns `(carbon_idx, hydroxyl_o_idx)`.
fn find_carboxyl(m: &Molecule) -> Option<(usize, usize)> {
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "C" {
            continue;
        }
        let neighbors = m.neighbors(i);
        let has_carbonyl_o = neighbors
            .iter()
            .any(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O");
        if !has_carbonyl_o {
            continue;
        }
        if let Some((oh_idx, _)) = neighbors.iter().find(|(j, o)| {
            *o == BondOrder::Single && m.atoms[*j].element == "O" && m.atoms[*j].hydrogens >= 1
        }) {
            return Some((i, *oh_idx));
        }
    }
    None
}

/// Find a hydroxyl oxygen that is NOT part of a carboxyl group: `O` with
/// exactly one bond (to a carbon), hydrogens >= 1, whose carbon has no
/// carbonyl (=O) neighbor.
fn find_alcohol_hydroxyl(m: &Molecule) -> Option<usize> {
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "O" || atom.hydrogens == 0 {
            continue;
        }
        let neighbors = m.neighbors(i);
        if neighbors.len() != 1 {
            continue;
        }
        let (c_idx, order) = neighbors[0];
        if order != BondOrder::Single || m.atoms[c_idx].element != "C" {
            continue;
        }
        let c_neighbors = m.neighbors(c_idx);
        let is_carbonyl_carbon = c_neighbors
            .iter()
            .any(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O");
        if !is_carbonyl_carbon {
            return Some(i);
        }
    }
    None
}

/// Find a free (unreacted) amine nitrogen: aliphatic (not aromatic), neutral,
/// at least one hydrogen (primary/secondary -- a tertiary amine has no H to
/// lose in this simple substitution), and NOT already bonded to a carbonyl
/// carbon (which would make it an existing amide/carbamate/urea nitrogen,
/// not a free amine available for a *new* bond). Same first-match-wins
/// convention as `find_alcohol_hydroxyl` above.
///
/// **Disclosed scope limit**: charged ammonium (`[NH3+]`) is deliberately
/// excluded, matching this file's existing charged-atom exemptions -- real
/// records built around a protonated amine (e.g. Phase A.3's frozen row 490)
/// are not matched by this template. Not a bug; a documented boundary, same
/// as `find_carboxyl`/`find_alcohol_hydroxyl` never handling charged atoms.
fn find_free_amine(m: &Molecule) -> Option<usize> {
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "N" || atom.aromatic || atom.charge != 0 || atom.hydrogens == 0 {
            continue;
        }
        let already_amide = m.neighbors(i).iter().any(|&(j, order)| {
            order != BondOrder::Aromatic
                && m.atoms[j].element == "C"
                && m.neighbors(j)
                    .iter()
                    .any(|&(k, o2)| o2 == BondOrder::Double && m.atoms[k].element == "O")
        });
        if !already_amide {
            return Some(i);
        }
    }
    None
}

/// R-COOH + R'-OH -> R-COO-R' + H2O.
///
/// Mass balance is exact by construction: the acid's hydroxyl O (+ its 1 H)
/// is fully removed, the alcohol's hydroxyl O loses exactly 1 H (it now
/// bonds to the ester carbon instead of just H) -- together that's 1 O + 2 H,
/// exactly `water()`.
pub struct EsterificationTemplate;

impl ReactionTemplate for EsterificationTemplate {
    fn name(&self) -> &'static str {
        "esterification"
    }

    /// **Order-independent (fixed 2026-07-14).** Real esterification doesn't
    /// care which reactant is listed first; a live 1,282-record external
    /// evaluation (Phase A.3) found real, correctly-shaped esterifications
    /// were misclassified as "unsupported" purely because the source data
    /// listed the alcohol before the acid -- at least 73/186 (39%,
    /// conservative estimate) of that evaluation's "unsupported" bucket
    /// showed this exact pattern. Tries `[a, b]` first, then `[b, a]` --
    /// deterministic tie-breaking if a pathological input could match
    /// either way (e.g. a molecule with both a carboxyl and an alcohol
    /// group on each side), matching this template's existing
    /// first-candidate-wins convention (`find_carboxyl`/
    /// `find_alcohol_hydroxyl` already pick the first match within one
    /// molecule the same way).
    fn apply(&self, reactants: &[Molecule]) -> Option<Vec<Molecule>> {
        let [a, b] = reactants else {
            return None;
        };
        try_esterify(a, b).or_else(|| try_esterify(b, a))
    }
}

/// Attempts esterification with a specific acid/alcohol assignment. Returns
/// `None` if `acid` has no carboxyl group or `alcohol` has no non-acid
/// hydroxyl -- callers try both argument orderings (see `apply` above).
fn try_esterify(acid: &Molecule, alcohol: &Molecule) -> Option<Vec<Molecule>> {
    let (acid_c, acid_oh) = find_carboxyl(acid)?;
    let alcohol_oh = find_alcohol_hydroxyl(alcohol)?;

    // Build the ester: acid's atoms with its hydroxyl-O removed, plus
    // alcohol's atoms (reindexed), with alcohol's hydroxyl-O losing 1 H,
    // plus a new bond from the acid's carboxyl carbon to that oxygen.
    let mut atoms: Vec<Atom> = Vec::with_capacity(acid.atoms.len() - 1 + alcohol.atoms.len());
    let mut remap = vec![usize::MAX; acid.atoms.len()];
    for (i, a) in acid.atoms.iter().enumerate() {
        if i == acid_oh {
            continue; // removed -- leaves as part of water
        }
        remap[i] = atoms.len();
        atoms.push(a.clone());
    }
    let new_acid_c = remap[acid_c];

    let mut alcohol_remap = vec![usize::MAX; alcohol.atoms.len()];
    for (i, a) in alcohol.atoms.iter().enumerate() {
        let mut a = a.clone();
        if i == alcohol_oh {
            a.hydrogens = a.hydrogens.saturating_sub(1);
        }
        alcohol_remap[i] = atoms.len();
        atoms.push(a);
    }
    let new_alcohol_o = alcohol_remap[alcohol_oh];

    let mut bonds: Vec<Bond> = Vec::new();
    for b in &acid.bonds {
        if b.a == acid_oh || b.b == acid_oh {
            continue; // the bond to the removed oxygen
        }
        bonds.push(Bond {
            a: remap[b.a],
            b: remap[b.b],
            order: b.order,
        });
    }
    for b in &alcohol.bonds {
        bonds.push(Bond {
            a: alcohol_remap[b.a],
            b: alcohol_remap[b.b],
            order: b.order,
        });
    }
    bonds.push(Bond {
        a: new_acid_c,
        b: new_alcohol_o,
        order: BondOrder::Single,
    });

    let ester = Molecule { atoms, bonds };
    Some(vec![ester, water()])
}

/// R-COOH + R'-NH2/R'R''NH -> R-CO-NR'(R'') + H2O.
///
/// Added in Phase A.4: a full-scale scan of `structurally_shaped_wrong_transformation`
/// records from a real 1,282-record USPTO-50K evaluation found that 17/20
/// (85%) of esterification-kind wrong-transformation records have a free
/// amine competing with the alcohol `EsterificationTemplate` searches for --
/// the real reaction usually forms this amide, not an ester.
///
/// Mass balance is exact by construction, same reasoning as
/// `EsterificationTemplate`: the acid's hydroxyl O (+ its 1 H) is fully
/// removed, the amine's nitrogen loses exactly 1 H (it now bonds to the
/// amide carbon instead of just H) -- together that's 1 O + 2 H, exactly
/// `water()`.
pub struct AmidationTemplate;

impl ReactionTemplate for AmidationTemplate {
    fn name(&self) -> &'static str {
        "amidation"
    }

    /// Order-independent from the start (unlike `EsterificationTemplate`,
    /// which had to be fixed to become order-independent after a real
    /// 1,282-record evaluation found it wasn't) -- no reason to reintroduce
    /// that class of bug in a new template.
    fn apply(&self, reactants: &[Molecule]) -> Option<Vec<Molecule>> {
        let [a, b] = reactants else {
            return None;
        };
        try_amidate(a, b).or_else(|| try_amidate(b, a))
    }
}

/// Attempts amidation with a specific acid/amine assignment. Returns `None`
/// if `acid` has no carboxyl group or `amine` has no free amine nitrogen --
/// callers try both argument orderings (see `apply` above). Selectivity
/// against a competing alcohol on the same molecule comes entirely from
/// `find_free_amine` only ever looking for nitrogen -- it never needs to
/// know `find_alcohol_hydroxyl` exists.
fn try_amidate(acid: &Molecule, amine: &Molecule) -> Option<Vec<Molecule>> {
    let (acid_c, acid_oh) = find_carboxyl(acid)?;
    let amine_n = find_free_amine(amine)?;

    // Build the amide: acid's atoms with its hydroxyl-O removed, plus
    // amine's atoms (reindexed), with the amine nitrogen losing 1 H, plus a
    // new bond from the acid's carboxyl carbon to that nitrogen.
    let mut atoms: Vec<Atom> = Vec::with_capacity(acid.atoms.len() - 1 + amine.atoms.len());
    let mut remap = vec![usize::MAX; acid.atoms.len()];
    for (i, a) in acid.atoms.iter().enumerate() {
        if i == acid_oh {
            continue; // removed -- leaves as part of water
        }
        remap[i] = atoms.len();
        atoms.push(a.clone());
    }
    let new_acid_c = remap[acid_c];

    let mut amine_remap = vec![usize::MAX; amine.atoms.len()];
    for (i, a) in amine.atoms.iter().enumerate() {
        let mut a = a.clone();
        if i == amine_n {
            a.hydrogens = a.hydrogens.saturating_sub(1);
        }
        amine_remap[i] = atoms.len();
        atoms.push(a);
    }
    let new_amine_n = amine_remap[amine_n];

    let mut bonds: Vec<Bond> = Vec::new();
    for b in &acid.bonds {
        if b.a == acid_oh || b.b == acid_oh {
            continue; // the bond to the removed oxygen
        }
        bonds.push(Bond {
            a: remap[b.a],
            b: remap[b.b],
            order: b.order,
        });
    }
    for b in &amine.bonds {
        bonds.push(Bond {
            a: amine_remap[b.a],
            b: amine_remap[b.b],
            order: b.order,
        });
    }
    bonds.push(Bond {
        a: new_acid_c,
        b: new_amine_n,
        order: BondOrder::Single,
    });

    let amide = Molecule { atoms, bonds };
    Some(vec![amide, water()])
}

/// True if `m` is exactly `h2()`: 2 heavy atoms, both H, one single bond.
/// `apply()` uses this to validate its own H2 argument rather than silently
/// ignoring it (a real API asymmetry an external review caught: the old
/// version took only `[unsaturated]` while `search.rs` appended H2 to the
/// displayed reactant list without it ever flowing through `apply`).
fn is_molecular_hydrogen(m: &Molecule) -> bool {
    m.atoms.len() == 2 && m.atoms.iter().all(|a| a.element == "H") && m.bonds.len() == 1
}

/// C-C double or triple bond + H2 -> one degree of saturation removed (one
/// hydrogenation step: triple->double or double->single), each bonded carbon
/// gains 1 H.
///
/// **Scoped to carbon-carbon multiple bonds only.** An earlier version
/// matched the *first* Double/Triple bond regardless of element -- it would
/// have reduced a C=O carbonyl or a C#N nitrile just as readily as a C=C
/// alkene, none of which are the same reaction or share the same conditions.
/// An external review caught this (the only reason it hadn't surfaced yet:
/// every test happened to use alkenes). Reducing carbonyls/nitriles/N=O is
/// real chemistry but belongs in separate, separately-tested templates.
pub struct HydrogenationTemplate;

impl ReactionTemplate for HydrogenationTemplate {
    fn name(&self) -> &'static str {
        "hydrogenation"
    }

    fn apply(&self, reactants: &[Molecule]) -> Option<Vec<Molecule>> {
        let [unsaturated, h2] = reactants else {
            return None;
        };
        if !is_molecular_hydrogen(h2) {
            return None;
        }
        let bond_idx = unsaturated.bonds.iter().position(|b| {
            matches!(b.order, BondOrder::Double | BondOrder::Triple)
                && unsaturated.atoms[b.a].element == "C"
                && unsaturated.atoms[b.b].element == "C"
        })?;

        let mut atoms = unsaturated.atoms.clone();
        let mut bonds = unsaturated.bonds.clone();
        let (a_idx, b_idx, new_order) = {
            let b = &bonds[bond_idx];
            let new_order = match b.order {
                BondOrder::Triple => BondOrder::Double,
                BondOrder::Double => BondOrder::Single,
                _ => unreachable!("filtered above"),
            };
            (b.a, b.b, new_order)
        };
        bonds[bond_idx].order = new_order;
        atoms[a_idx].hydrogens += 1;
        atoms[b_idx].hydrogens += 1;

        let hydrogenated = Molecule { atoms, bonds };
        Some(vec![hydrogenated])
    }
}

/// Sum of C-C reducible-bond "equivalents" in `m`: 1 per C=C, 2 per C#C (a
/// triple bond needs two successive reduction steps -- triple->double,
/// double->single -- to reach full saturation). Same carbon-carbon-only
/// restriction `HydrogenationTemplate::apply` already enforces, just counted
/// instead of applied.
///
/// This exists so a caller can determine, up front, exactly how many
/// `molecular_hydrogen()` equivalents `ExhaustiveHydrogenationTemplate`
/// needs before calling it -- see that template's doc comment for why an
/// exact count (not a guess) matters for mass balance.
pub fn count_reducible_cc_bonds(m: &Molecule) -> usize {
    m.bonds
        .iter()
        .filter_map(|b| {
            if m.atoms[b.a].element != "C" || m.atoms[b.b].element != "C" {
                return None;
            }
            match b.order {
                BondOrder::Double => Some(1),
                BondOrder::Triple => Some(2),
                _ => None,
            }
        })
        .sum()
}

/// Reduces a molecule to full C-C saturation in one candidate, not one step
/// at a time.
///
/// Phase A.5: a full 1,282-record USPTO evaluation found ~69% (82/119) of
/// hydrogenation's `structurally_shaped_wrong_transformation` records share
/// one signature -- the declared product has exactly 2 more H than
/// `HydrogenationTemplate` computes, because the real reaction fully
/// reduces an alkyne to an alkane (standard excess-H2/catalyst practice)
/// while that template only removes one degree of unsaturation per
/// application.
///
/// **Requires an EXACT H2 count, not a generous upper bound.** Reducing N
/// degrees of unsaturation consumes N equivalents of H2; supplying more or
/// fewer than `count_reducible_cc_bonds` demands would make the resulting
/// `ReactionCandidate` misrepresent what it actually consumed, and
/// `validity::check_conservation` would (correctly) reject it. Callers must
/// compute the exact count via `count_reducible_cc_bonds` first -- this
/// template fails closed (returns `None`) rather than silently accepting a
/// wrong count.
///
/// **Not a claim that real hydrogenations always go to full saturation.**
/// Partial/chemoselective reduction (e.g. Lindlar-catalyst alkyne->alkene)
/// is real, common, named chemistry, not a rare exception -- Phase A.4's
/// `AmidationTemplate` regression already taught this project not to pick
/// one behavior as a fixed default when the real answer is genuinely
/// context-dependent. This template is registered as an INDEPENDENT
/// candidate alongside `HydrogenationTemplate`, not a replacement for it;
/// the auditor's `classify_all` tries both and accepts whichever matches
/// the declared product, and the generator emits both independently.
pub struct ExhaustiveHydrogenationTemplate;

impl ReactionTemplate for ExhaustiveHydrogenationTemplate {
    fn name(&self) -> &'static str {
        "exhaustive_hydrogenation"
    }

    fn apply(&self, reactants: &[Molecule]) -> Option<Vec<Molecule>> {
        let [unsaturated, rest @ ..] = reactants else {
            return None;
        };
        if rest.is_empty() || !rest.iter().all(is_molecular_hydrogen) {
            return None;
        }
        let needed = count_reducible_cc_bonds(unsaturated);
        if needed == 0 || rest.len() != needed {
            return None;
        }

        let mut current = unsaturated.clone();
        for _ in 0..needed {
            let step = HydrogenationTemplate.apply(&[current, molecular_hydrogen()])?;
            current = step.into_iter().next()?;
        }
        Some(vec![current])
    }
}

/// Convenience: the H2 "reactant" molecule for candidates using
/// `HydrogenationTemplate`, and the fixed water byproduct for
/// `EsterificationTemplate` -- exposed so `search.rs` can build a complete
/// `ReactionCandidate` (both explicit reactants/products) without
/// duplicating these constructors.
pub fn molecular_hydrogen() -> Molecule {
    h2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ReactionCandidate;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn esterification_mass_balances() {
        let acid = mol("CC(=O)O"); // acetic acid
        let alcohol = mol("CCO"); // ethanol
        let lhs_atoms = acid.total_atom_count() + alcohol.total_atom_count();

        let products = EsterificationTemplate.apply(&[acid, alcohol]).unwrap();
        let rhs_atoms: usize = products.iter().map(|p| p.total_atom_count()).sum();

        assert_eq!(lhs_atoms, rhs_atoms, "mass balance violated");
        // Ethyl acetate: C4H8O2
        let ester = &products[0];
        assert_eq!(ester.molecular_formula(), "C4H8O2");
        let water_product = &products[1];
        assert_eq!(water_product.molecular_formula(), "H2O");
    }

    #[test]
    fn esterification_rejects_non_matching_reactants() {
        let a = mol("CCO");
        let b = mol("CCO");
        assert!(EsterificationTemplate.apply(&[a, b]).is_none());
    }

    #[test]
    fn esterification_is_order_independent() {
        // Regression test for the Phase A.3 finding: real esterifications
        // with the alcohol listed first were previously misclassified as
        // "no template matched" purely due to argument order, which isn't
        // a real chemical distinction.
        let acid = mol("CC(=O)O");
        let alcohol = mol("CCO");

        let acid_first = EsterificationTemplate
            .apply(&[acid.clone(), alcohol.clone()])
            .unwrap();
        let alcohol_first = EsterificationTemplate.apply(&[alcohol, acid]).unwrap();

        assert_eq!(acid_first[0].molecular_formula(), "C4H8O2");
        assert_eq!(
            alcohol_first[0].molecular_formula(),
            "C4H8O2",
            "reversed reactant order must produce the same ester"
        );
        assert_eq!(alcohol_first[1].molecular_formula(), "H2O");
    }

    #[test]
    fn esterification_still_rejects_when_neither_order_matches() {
        // Two plain alcohols, in either order -- neither ordering should
        // spuriously succeed just because the order-independence fix tries
        // both.
        let a = mol("CCO");
        let b = mol("CO");
        assert!(
            EsterificationTemplate
                .apply(&[a.clone(), b.clone()])
                .is_none()
        );
        assert!(EsterificationTemplate.apply(&[b, a]).is_none());
    }

    #[test]
    fn amidation_mass_balances() {
        let acid = mol("CC(=O)O"); // acetic acid
        let amine = mol("CN"); // methylamine
        let lhs_atoms = acid.total_atom_count() + amine.total_atom_count();

        let products = AmidationTemplate.apply(&[acid, amine]).unwrap();
        let rhs_atoms: usize = products.iter().map(|p| p.total_atom_count()).sum();

        assert_eq!(lhs_atoms, rhs_atoms, "mass balance violated");
        // N-methylacetamide: C3H7NO
        let amide = &products[0];
        assert_eq!(amide.molecular_formula(), "C3H7NO");
        let water_product = &products[1];
        assert_eq!(water_product.molecular_formula(), "H2O");
    }

    #[test]
    fn amidation_rejects_non_matching_reactants() {
        let a = mol("CCO"); // an alcohol, no free amine
        let b = mol("CCO");
        assert!(AmidationTemplate.apply(&[a, b]).is_none());
    }

    #[test]
    fn amidation_is_order_independent() {
        let acid = mol("CC(=O)O");
        let amine = mol("CN");

        let acid_first = AmidationTemplate
            .apply(&[acid.clone(), amine.clone()])
            .unwrap();
        let amine_first = AmidationTemplate.apply(&[amine, acid]).unwrap();

        assert_eq!(acid_first[0].molecular_formula(), "C3H7NO");
        assert_eq!(
            amine_first[0].molecular_formula(),
            "C3H7NO",
            "reversed reactant order must produce the same amide"
        );
        assert_eq!(amine_first[1].molecular_formula(), "H2O");
    }

    #[test]
    fn amidation_still_rejects_when_neither_order_matches() {
        let a = mol("CCO");
        let b = mol("CO");
        assert!(AmidationTemplate.apply(&[a.clone(), b.clone()]).is_none());
        assert!(AmidationTemplate.apply(&[b, a]).is_none());
    }

    #[test]
    fn amidation_prefers_the_amine_over_a_coexisting_alcohol_on_the_same_molecule() {
        // Ethanolamine (NCCO) has BOTH a free amine and a free alcohol.
        // Mirrors the real Phase A.3 records (rows 38/283/473/490) where
        // EsterificationTemplate silently esterified the alcohol instead of
        // the amide the real reaction actually formed. AmidationTemplate
        // reacts at the amine and leaves the alcohol untouched -- not by
        // choosing between them, but because find_free_amine only ever
        // looks for nitrogen and has no code path that could touch oxygen.
        let acid = mol("CC(=O)O"); // acetic acid
        let ethanolamine = mol("NCCO");
        let products = AmidationTemplate
            .apply(&[acid, ethanolamine])
            .expect("must find the free amine and form the amide");
        // N-(2-hydroxyethyl)acetamide: C4H9NO2.
        let amide = &products[0];
        assert_eq!(amide.molecular_formula(), "C4H9NO2");

        let amide_n = amide
            .atoms
            .iter()
            .find(|a| a.element == "N")
            .expect("amide nitrogen must be present");
        assert_eq!(
            amide_n.hydrogens, 1,
            "the amine lost exactly one H, from 2 to 1"
        );

        let surviving_alcohol_o_count = amide
            .atoms
            .iter()
            .filter(|a| a.element == "O" && a.hydrogens == 1)
            .count();
        assert_eq!(
            surviving_alcohol_o_count, 1,
            "the alcohol's oxygen must survive untouched, with its hydrogen intact"
        );
    }

    #[test]
    fn hydrogenation_ethylene_to_ethane() {
        let ethylene = mol("C=C");
        let products = HydrogenationTemplate
            .apply(&[ethylene, molecular_hydrogen()])
            .unwrap();
        assert_eq!(products.len(), 1);
        assert_eq!(products[0].molecular_formula(), "C2H6"); // ethane
    }

    #[test]
    fn hydrogenation_propylene_to_propane() {
        let propylene = mol("CC=C");
        let products = HydrogenationTemplate
            .apply(&[propylene, molecular_hydrogen()])
            .unwrap();
        assert_eq!(products[0].molecular_formula(), "C3H8"); // propane
    }

    #[test]
    fn hydrogenation_rejects_saturated_input() {
        let ethane = mol("CC");
        assert!(
            HydrogenationTemplate
                .apply(&[ethane, molecular_hydrogen()])
                .is_none()
        );
    }

    #[test]
    fn hydrogenation_rejects_missing_h2() {
        let ethylene = mol("C=C");
        let not_h2 = mol("CCO"); // wrong second reactant
        assert!(HydrogenationTemplate.apply(&[ethylene, not_h2]).is_none());
    }

    #[test]
    fn hydrogenation_skips_carbonyl_and_nitrile_reduces_only_the_c_c_bond() {
        // Acrylonitrile: CH2=CH-C#N. Has both a C=C (alkene) and a C#N
        // (nitrile) multiple bond. HydrogenationTemplate must reduce ONLY
        // the C=C, never the C#N. NOTE (found via the Phase 1.2 mutation-
        // testing pass): for THIS specific molecule, the C=C also happens
        // to be first in the parser's bond list, so this test alone does
        // NOT independently prove the C-C restriction is doing the work --
        // it would pass even with the restriction removed, by bond-order
        // coincidence. See `hydrogenation_carbonyl_before_cc_bond_in_atom_order`
        // below for the test that actually depends on the restriction.
        let acrylonitrile = mol("C=CC#N");
        let products = HydrogenationTemplate
            .apply(&[acrylonitrile, molecular_hydrogen()])
            .unwrap();
        // Propionitrile (CH3-CH2-C#N): the C=C is saturated, the C#N survives.
        assert_eq!(products[0].molecular_formula(), "C3H5N");
        let triple_bonds = products[0]
            .bonds
            .iter()
            .filter(|b| b.order == BondOrder::Triple)
            .count();
        assert_eq!(
            triple_bonds, 1,
            "the nitrile triple bond must survive intact"
        );
    }

    #[test]
    fn hydrogenation_carbonyl_before_cc_bond_in_atom_order() {
        // Acrolein (prop-2-enal), CH2=CH-CHO, written so the C=O carbonyl
        // bond is FIRST in the parser's bond list and the C=C alkene is
        // LAST -- the opposite order from the acrylonitrile test above.
        // Without the C-C restriction, `.position()` would pick the
        // carbonyl bond (wrong: reduces the aldehyde) instead of the
        // alkene. This is the test that actually depends on the
        // restriction, found missing during the Phase 1.2 mutation-testing
        // pass (mutation 4 removed the restriction and this was the only
        // test of the two that caught it).
        let acrolein = mol("O=CC=C");
        let products = HydrogenationTemplate
            .apply(&[acrolein, molecular_hydrogen()])
            .unwrap();
        // Propanal (CH3-CH2-CHO): the C=C is saturated, the C=O survives.
        assert_eq!(products[0].molecular_formula(), "C3H6O");
        let carbonyl_survives = products[0].bonds.iter().any(|b| {
            b.order == BondOrder::Double
                && (products[0].atoms[b.a].element == "O" || products[0].atoms[b.b].element == "O")
        });
        assert!(carbonyl_survives, "the C=O carbonyl must survive intact");
    }

    #[test]
    fn hydrogenation_rejects_pure_carbonyl_no_cc_bond() {
        // Formaldehyde CH2=O: the only multiple bond is C=O, not C=C/C#C.
        // HydrogenationTemplate must find nothing to reduce.
        let formaldehyde = mol("C=O");
        assert!(
            HydrogenationTemplate
                .apply(&[formaldehyde, molecular_hydrogen()])
                .is_none()
        );
    }

    #[test]
    fn count_reducible_cc_bonds_counts_correctly() {
        assert_eq!(
            count_reducible_cc_bonds(&mol("CC")),
            0,
            "ethane: fully saturated"
        );
        assert_eq!(
            count_reducible_cc_bonds(&mol("C=C")),
            1,
            "ethylene: one C=C"
        );
        assert_eq!(
            count_reducible_cc_bonds(&mol("C#C")),
            2,
            "acetylene: one C#C = 2 steps"
        );
        assert_eq!(
            count_reducible_cc_bonds(&mol("C=CC#C")),
            3,
            "vinylacetylene: 1 (C=C) + 2 (C#C) = 3"
        );
        assert_eq!(
            count_reducible_cc_bonds(&mol("C=O")),
            0,
            "C=O is not a C-C bond, must not count"
        );
        assert_eq!(
            count_reducible_cc_bonds(&mol("C=CC#N")),
            1,
            "only the C=C counts; C#N is carbon-nitrogen, not carbon-carbon"
        );
    }

    #[test]
    fn exhaustive_hydrogenation_fully_reduces_a_diene() {
        let pentadiene = mol("C=CCC=C"); // 1,4-pentadiene, C5H8
        let needed = count_reducible_cc_bonds(&pentadiene);
        assert_eq!(needed, 2);
        let h2s: Vec<Molecule> = (0..needed).map(|_| molecular_hydrogen()).collect();
        let mut reactants = vec![pentadiene];
        reactants.extend(h2s);
        let products = ExhaustiveHydrogenationTemplate.apply(&reactants).unwrap();
        assert_eq!(products.len(), 1);
        assert_eq!(
            products[0].molecular_formula(),
            "C5H12",
            "pentane, fully saturated"
        );
        assert_eq!(count_reducible_cc_bonds(&products[0]), 0);
    }

    #[test]
    fn exhaustive_hydrogenation_fully_reduces_an_alkyne() {
        let propyne = mol("C#CC"); // C3H4
        let needed = count_reducible_cc_bonds(&propyne);
        assert_eq!(needed, 2);
        let h2s: Vec<Molecule> = (0..needed).map(|_| molecular_hydrogen()).collect();
        let mut reactants = vec![propyne];
        reactants.extend(h2s);
        let products = ExhaustiveHydrogenationTemplate.apply(&reactants).unwrap();
        assert_eq!(
            products[0].molecular_formula(),
            "C3H8",
            "propane, fully saturated"
        );
    }

    #[test]
    fn exhaustive_hydrogenation_rejects_wrong_h2_count() {
        // Propyne needs exactly 2 H2 equivalents -- supplying 1 (too few) or
        // 3 (too many) must both fail closed, never silently over/under-consume.
        let propyne = mol("C#CC");
        assert!(
            ExhaustiveHydrogenationTemplate
                .apply(&[propyne.clone(), molecular_hydrogen()])
                .is_none(),
            "1 H2 is too few for a triple bond"
        );
        assert!(
            ExhaustiveHydrogenationTemplate
                .apply(&[
                    propyne,
                    molecular_hydrogen(),
                    molecular_hydrogen(),
                    molecular_hydrogen()
                ])
                .is_none(),
            "3 H2 is too many"
        );
    }

    #[test]
    fn exhaustive_hydrogenation_rejects_saturated_input() {
        let propane = mol("CCC");
        assert!(ExhaustiveHydrogenationTemplate.apply(&[propane]).is_none());
    }

    #[test]
    fn exhaustive_hydrogenation_skips_nitrile_reduces_only_the_c_c_bond() {
        // Acrylonitrile CH2=CH-C#N: only the C=C counts (C#N is C-N, not
        // C-C) -- ExhaustiveHydrogenationTemplate needs exactly 1 H2 and
        // must leave the nitrile intact, same restriction
        // HydrogenationTemplate::apply already enforces per-step.
        let acrylonitrile = mol("C=CC#N");
        assert_eq!(count_reducible_cc_bonds(&acrylonitrile), 1);
        let products = ExhaustiveHydrogenationTemplate
            .apply(&[acrylonitrile, molecular_hydrogen()])
            .unwrap();
        assert_eq!(products[0].molecular_formula(), "C3H5N", "propionitrile");
        let triple_bonds = products[0]
            .bonds
            .iter()
            .filter(|b| b.order == BondOrder::Triple)
            .count();
        assert_eq!(
            triple_bonds, 1,
            "the nitrile triple bond must survive intact"
        );
    }

    #[test]
    fn molecular_hydrogen_formula() {
        assert_eq!(molecular_hydrogen().molecular_formula(), "H2");
    }

    /// Same seed pool as `validity.rs`'s property tests, duplicated locally
    /// (small, avoids introducing a cross-module test-fixture dependency
    /// for a one-line array).
    const SEED_SMILES: &[&str] = &[
        "C=C",
        "CC=C",
        "c1ccccc1",
        "CCO",
        "CC(=O)O",
        "CO",
        "C=CC#N",
        "c1ccc(cc1)O",
        "OC(=O)CCCCC(=O)O",
        "O=C1CCCCCN1",
        "CCC",
        "CCCC",
    ];

    // "Randomized conservation tests across every transformation" (Phase
    // 1.2): whenever EITHER template returns `Some(products)` for a
    // randomly sampled pair of real reactant molecules, element and charge
    // conservation must hold. This is exactly the invariant
    // `validity::check_conservation` enforces in the real oracle pipeline --
    // this test instead checks the templates' own construction logic
    // satisfies it directly, many random samples at a time, rather than
    // relying only on the two hand-picked cases in
    // `esterification_mass_balances`/hand-checked hydrogenation cases above.
    proptest::proptest! {
        #[test]
        fn prop_hydrogenation_always_conserves(idx in 0usize..SEED_SMILES.len()) {
            let reactant = mol(SEED_SMILES[idx]);
            let h2 = molecular_hydrogen();
            if let Some(products) = HydrogenationTemplate.apply(&[reactant.clone(), h2.clone()]) {
                let candidate = ReactionCandidate {
                    reactants: vec![reactant, h2],
                    products,
                    template: "hydrogenation",
                };
                proptest::prop_assert!(crate::validity::check_candidate(&candidate).is_ok());
            }
        }

        #[test]
        fn prop_esterification_always_conserves(
            i in 0usize..SEED_SMILES.len(),
            j in 0usize..SEED_SMILES.len(),
        ) {
            let a = mol(SEED_SMILES[i]);
            let b = mol(SEED_SMILES[j]);
            if let Some(products) = EsterificationTemplate.apply(&[a.clone(), b.clone()]) {
                let candidate = ReactionCandidate {
                    reactants: vec![a, b],
                    products,
                    template: "esterification",
                };
                proptest::prop_assert!(crate::validity::check_candidate(&candidate).is_ok());
            }
        }

        /// The core correctness invariant for ExhaustiveHydrogenationTemplate:
        /// whenever it succeeds, the output must have ZERO remaining
        /// reducible C-C bonds (it actually reached full saturation, not a
        /// partial reduction that happened to satisfy the H2 count check),
        /// and the candidate must still conserve elements/charge exactly.
        #[test]
        fn prop_exhaustive_hydrogenation_always_reaches_full_saturation_and_conserves(
            idx in 0usize..SEED_SMILES.len(),
        ) {
            let reactant = mol(SEED_SMILES[idx]);
            let needed = count_reducible_cc_bonds(&reactant);
            if needed == 0 {
                return Ok(());
            }
            let h2s: Vec<Molecule> = (0..needed).map(|_| molecular_hydrogen()).collect();
            let mut reactants = vec![reactant];
            reactants.extend(h2s.clone());
            if let Some(products) = ExhaustiveHydrogenationTemplate.apply(&reactants) {
                proptest::prop_assert_eq!(count_reducible_cc_bonds(&products[0]), 0);
                let candidate = ReactionCandidate {
                    reactants,
                    products,
                    template: "exhaustive_hydrogenation",
                };
                proptest::prop_assert!(crate::validity::check_candidate(&candidate).is_ok());
            }
        }
    }
}
