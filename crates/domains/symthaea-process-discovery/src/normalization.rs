// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Representation normalization: converts recognized alternative encodings
//! of the same chemical identity into one canonical internal form *before*
//! authoritative validity checking, rather than loosening the validity
//! rules themselves.
//!
//! Motivated by a real gap the Reaction Corpus Auditor's first live run
//! surfaced: a neutrally-drawn nitro group (`N(=O)=O`, a common textbook
//! shorthand) fails `validity::check_molecule`'s neutral-atom valence check
//! (bonded valence 5 vs. nitrogen's normal valence 3), even though this is
//! real, common, valid chemistry -- just drawn as an informal resonance
//! shorthand rather than the formally correct charge-separated Lewis
//! structure (`[N+](=O)[O-]`). The fix is **not** to loosen the valence
//! check (that would also start accepting genuinely malformed structures)
//! -- it's to recognize the specific shorthand and convert it to the
//! structure it actually represents, then validate that.
//!
//! Every normalization is recorded, never silent: `normalize_candidate`
//! returns which rule fired on which molecule, so a certificate can
//! distinguish "valid exactly as supplied" from "valid after a recognized,
//! logged normalization" -- see `certificate.rs`.

use serde::Serialize;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NormalizationRecord {
    pub rule: &'static str,
    /// How many times this rule fired within the one molecule (e.g. 2 for a
    /// dinitro compound) -- kept as a count rather than one record per
    /// occurrence, since the rule identity is what a reviewer needs, not a
    /// per-atom log.
    pub applied_count: usize,
}

/// Convert every neutral, formally-pentavalent-drawn nitro group (`N` with
/// two `N=O` double bonds and net atom charge 0) into the charge-separated
/// form (`N` charge +1, one `N=O` becomes `N-O` single bond, that `O`
/// charge -1) that real chemistry (and `validity.rs`'s existing
/// charged-atom exemption) already treats as well-formed. Mutates `m` in
/// place; returns how many nitro groups were converted.
///
/// Deliberately conservative: only fires when a nitrogen has *exactly* two
/// double bonds to charge-0, non-hydrogenated oxygen atoms and is itself
/// charge-0. A nitrogen with some other, unrecognized bonding pattern is
/// left untouched -- it either isn't a nitro group or is malformed in some
/// other way `validity.rs` should catch on its own, not something this
/// rule should guess about.
fn normalize_neutral_nitro_groups(m: &mut Molecule) -> usize {
    let mut count = 0;
    for i in 0..m.atoms.len() {
        if m.atoms[i].element != "N" || m.atoms[i].charge != 0 {
            continue;
        }
        let double_bond_o_indices: Vec<usize> = m
            .bonds
            .iter()
            .enumerate()
            .filter(|(_, b)| b.order == BondOrder::Double && (b.a == i || b.b == i))
            .filter_map(|(bi, b)| {
                let o_idx = if b.a == i { b.b } else { b.a };
                let o = &m.atoms[o_idx];
                if o.element == "O" && o.charge == 0 && o.hydrogens == 0 {
                    Some(bi)
                } else {
                    None
                }
            })
            .collect();
        if double_bond_o_indices.len() != 2 {
            continue;
        }
        // Deterministic choice: convert the higher-indexed bond, so the
        // result doesn't depend on iteration/HashMap ordering anywhere
        // upstream -- same input always normalizes to the same output.
        let bond_idx = *double_bond_o_indices.iter().max().unwrap();
        let o_idx = if m.bonds[bond_idx].a == i {
            m.bonds[bond_idx].b
        } else {
            m.bonds[bond_idx].a
        };
        m.bonds[bond_idx].order = BondOrder::Single;
        m.atoms[i].charge = 1;
        m.atoms[o_idx].charge = -1;
        count += 1;
    }
    count
}

/// Normalize one molecule. Element counts, hydrogen counts, and net charge
/// are unchanged by every rule here (only bond order and *which* atom
/// carries a given formal charge change) -- so `molecular_formula()` and
/// `validity::check_conservation` are unaffected by normalization, by
/// construction, not by coincidence.
pub fn normalize_molecule(m: &Molecule) -> (Molecule, Vec<NormalizationRecord>) {
    let mut result = m.clone();
    let mut records = Vec::new();
    let nitro_count = normalize_neutral_nitro_groups(&mut result);
    if nitro_count > 0 {
        records.push(NormalizationRecord {
            rule: "neutral_nitro_to_charge_separated",
            applied_count: nitro_count,
        });
    }
    (result, records)
}

/// A candidate after normalization, paired with which normalization
/// records apply to which reactant/product (parallel to
/// `candidate.reactants`/`candidate.products`, same length, empty vec where
/// nothing fired). This is the structure the oracle actually validates and
/// certifies -- see `oracle.rs`'s "stage 0" and `certificate.rs`.
#[derive(Debug, Clone)]
pub struct NormalizedCandidate {
    pub candidate: crate::types::ReactionCandidate,
    pub reactant_normalizations: Vec<Vec<NormalizationRecord>>,
    pub product_normalizations: Vec<Vec<NormalizationRecord>>,
}

impl NormalizedCandidate {
    /// True if every reactant and product needed zero normalization --
    /// i.e. the candidate is "valid exactly as supplied," not "valid after
    /// a recognized normalization."
    pub fn any_normalization_applied(&self) -> bool {
        self.reactant_normalizations
            .iter()
            .chain(self.product_normalizations.iter())
            .any(|recs| !recs.is_empty())
    }
}

pub fn normalize_candidate(candidate: &crate::types::ReactionCandidate) -> NormalizedCandidate {
    let mut reactants = Vec::with_capacity(candidate.reactants.len());
    let mut reactant_normalizations = Vec::with_capacity(candidate.reactants.len());
    for m in &candidate.reactants {
        let (normalized, records) = normalize_molecule(m);
        reactants.push(normalized);
        reactant_normalizations.push(records);
    }
    let mut products = Vec::with_capacity(candidate.products.len());
    let mut product_normalizations = Vec::with_capacity(candidate.products.len());
    for m in &candidate.products {
        let (normalized, records) = normalize_molecule(m);
        products.push(normalized);
        product_normalizations.push(records);
    }
    NormalizedCandidate {
        candidate: crate::types::ReactionCandidate {
            reactants,
            products,
            template: candidate.template,
        },
        reactant_normalizations,
        product_normalizations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn neutral_nitro_is_normalized_to_charge_separated() {
        let (normalized, records) = normalize_molecule(&mol("CN(=O)=O"));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].rule, "neutral_nitro_to_charge_separated");
        assert_eq!(records[0].applied_count, 1);

        let n = normalized.atoms.iter().find(|a| a.element == "N").unwrap();
        assert_eq!(n.charge, 1);
        let negative_o_count = normalized
            .atoms
            .iter()
            .filter(|a| a.element == "O" && a.charge == -1)
            .count();
        assert_eq!(negative_o_count, 1);
        let double_bond_o_count = normalized
            .bonds
            .iter()
            .filter(|b| b.order == BondOrder::Double)
            .count();
        assert_eq!(
            double_bond_o_count, 1,
            "exactly one N=O double bond should remain after normalization"
        );
    }

    #[test]
    fn normalization_preserves_formula_and_charge_neutral_total() {
        let original = mol("CN(=O)=O");
        let (normalized, _) = normalize_molecule(&original);
        assert_eq!(original.molecular_formula(), normalized.molecular_formula());
        let total_charge: i32 = normalized.atoms.iter().map(|a| a.charge as i32).sum();
        assert_eq!(
            total_charge, 0,
            "nitro normalization must not change net molecular charge"
        );
    }

    #[test]
    fn already_charge_separated_nitro_is_left_alone_no_double_normalization() {
        let (_, records) = normalize_molecule(&mol("C[N+](=O)[O-]"));
        assert!(
            records.is_empty(),
            "an already-correct charge-separated nitro must not be re-normalized: {records:?}"
        );
    }

    #[test]
    fn dinitro_molecule_normalizes_both_groups() {
        let (_, records) = normalize_molecule(&mol("c1cc(ccc1N(=O)=O)N(=O)=O"));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].applied_count, 2);
    }

    #[test]
    fn benign_molecule_is_never_touched() {
        let (normalized, records) = normalize_molecule(&mol("CCO"));
        assert!(records.is_empty());
        assert_eq!(normalized.molecular_formula(), "C2H6O");
    }

    #[test]
    fn post_normalization_molecule_passes_validity() {
        // The whole point: validity.rs's own neutral-atom valence check
        // must accept the normalized structure (it already exempts charged
        // atoms -- this rule exists specifically to route into that
        // existing exemption for real chemistry, not to bypass validity).
        let (normalized, _) = normalize_molecule(&mol("c1ccccc1N(=O)=O"));
        assert!(crate::validity::check_molecule(&normalized).is_ok());
    }

    #[test]
    fn pre_normalization_neutral_nitro_fails_validity_documenting_the_gap_this_fixes() {
        let unnormalized = mol("c1ccccc1N(=O)=O");
        assert!(
            crate::validity::check_molecule(&unnormalized).is_err(),
            "this test documents the exact gap normalization fixes -- if this ever starts \
             passing, validity.rs's valence check changed and this module's premise should be \
             re-examined"
        );
    }

    #[test]
    fn candidate_normalization_tracks_which_reactant_and_product_changed() {
        use crate::types::ReactionCandidate;
        let candidate = ReactionCandidate {
            reactants: vec![mol("c1ccccc1N(=O)=O"), mol("[H][H]")],
            products: vec![mol("c1ccccc1N")],
            template: "test",
        };
        let normalized = normalize_candidate(&candidate);
        assert!(normalized.any_normalization_applied());
        assert_eq!(normalized.reactant_normalizations.len(), 2);
        assert!(!normalized.reactant_normalizations[0].is_empty());
        assert!(normalized.reactant_normalizations[1].is_empty());
        assert_eq!(normalized.product_normalizations.len(), 1);
        assert!(normalized.product_normalizations[0].is_empty());
        assert!(crate::validity::check_molecule(&normalized.candidate.reactants[0]).is_ok());
    }

    #[test]
    fn untouched_candidate_reports_no_normalization_applied() {
        use crate::types::ReactionCandidate;
        let candidate = ReactionCandidate {
            reactants: vec![mol("CCO")],
            products: vec![mol("CCO")],
            template: "identity",
        };
        let normalized = normalize_candidate(&candidate);
        assert!(!normalized.any_normalization_applied());
    }

    // "Multiple accepted encodings of the same compound should converge on
    // the same authoritative graph identity" -- the representation-
    // equivalence property this module exists to guarantee. Nitrobenzene
    // written as the common neutral shorthand vs. the formally correct
    // charge-separated Lewis structure must normalize to an IDENTICAL atom/
    // bond graph (same charges, same bond orders, same formula) even though
    // only one of the two inputs needed a normalization record to get
    // there.
    #[test]
    fn equivalent_encodings_of_nitrobenzene_converge_to_identical_normalized_structure() {
        let (from_neutral, neutral_records) = normalize_molecule(&mol("c1ccccc1N(=O)=O"));
        let (from_charge_separated, charge_separated_records) =
            normalize_molecule(&mol("c1ccccc1[N+](=O)[O-]"));

        assert!(
            !neutral_records.is_empty(),
            "the neutral encoding must be recognized as needing normalization"
        );
        assert!(
            charge_separated_records.is_empty(),
            "the already-correct encoding must need no normalization"
        );

        assert_eq!(
            from_neutral.molecular_formula(),
            from_charge_separated.molecular_formula()
        );
        let mut neutral_charges: Vec<i8> = from_neutral.atoms.iter().map(|a| a.charge).collect();
        let mut charge_separated_charges: Vec<i8> = from_charge_separated
            .atoms
            .iter()
            .map(|a| a.charge)
            .collect();
        neutral_charges.sort();
        charge_separated_charges.sort();
        assert_eq!(
            neutral_charges, charge_separated_charges,
            "both encodings must normalize to the same multiset of formal charges"
        );

        let mut neutral_orders: Vec<BondOrder> =
            from_neutral.bonds.iter().map(|b| b.order).collect();
        let mut charge_separated_orders: Vec<BondOrder> = from_charge_separated
            .bonds
            .iter()
            .map(|b| b.order)
            .collect();
        neutral_orders.sort_by_key(|o| format!("{o:?}"));
        charge_separated_orders.sort_by_key(|o| format!("{o:?}"));
        assert_eq!(
            neutral_orders, charge_separated_orders,
            "both encodings must normalize to the same multiset of bond orders"
        );

        assert!(crate::validity::check_molecule(&from_neutral).is_ok());
        assert!(crate::validity::check_molecule(&from_charge_separated).is_ok());
    }
}
