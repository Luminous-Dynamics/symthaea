// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic structural validity checks for a candidate, run before scope and
//! stability. Added in the Phase 1.1 hardening pass: an external review
//! correctly pointed out that `GateOutcome::FailedValidity` existed as a
//! type but nothing ever produced it -- the oracle assumed "parsed by
//! `Molecule::from_smiles` or emitted by a `ReactionTemplate`" was enough,
//! which is not true once templates directly manipulate atoms/bonds/
//! hydrogens by hand. A future template with a bug (wrong index, dropped
//! atom, unconserved charge) would have sailed through undetected.
//!
//! These checks are generic (apply to any candidate, not specific to any
//! one template) and structural (element set, bond-index sanity, valence
//! plausibility, and -- the most valuable one -- exact elemental and charge
//! conservation between reactants and products).

use crate::aromaticity;
use crate::types::ReactionCandidate;
use symthaea_organic_chemistry::element;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

/// Upper-bounded by `symthaea-quantum-chemistry`'s STO-3G element coverage
/// (Z=1-20, 31-36 as of Phase A.7), but deliberately NOT a 1:1 mirror of
/// it: quantum-chemistry can also build a basis set for Na/Mg/Al/K/Ca/Ar/Kr,
/// none of which legitimately appear as part of a covalently-bonded organic
/// reactant/product structure in this crate's domain (alkali/alkaline-earth
/// metals show up only as salt counterions -- already excluded by the
/// existing disconnected-structure rejection upstream in the SMILES parser
/// -- and noble gases are inert). This allowlist's actual boundary is
/// chemistry-appropriateness, not quantum-chemistry-computability; a
/// candidate that passes here is guaranteed (not just intended) to also be
/// eligible for a future quantum-chemistry feasibility gate, since every
/// element here already has real STO-3G basis data.
const ALLOWED_ELEMENTS: &[&str] = &["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"];

/// Chemically-realistic upper bound on |formal charge|. Organic ions rarely
/// exceed +-2; this is deliberately generous while still catching absurd
/// values (an `i8` field otherwise permits up to +-127 with nothing to stop
/// a template bug or a hostile generator from producing one).
const MAX_ABS_CHARGE: i8 = 3;

/// Structural sanity for one molecule: known/allowed elements, in-range bond
/// indices, and (for neutral atoms) valence conservation -- bonded valence
/// plus attached hydrogens should equal the element's normal valence,
/// exactly what `organic-chemistry`'s own implicit-hydrogen derivation
/// assumes but that a hand-built (template-produced) molecule never runs
/// through.
pub fn check_molecule(m: &Molecule) -> Result<(), String> {
    for (i, atom) in m.atoms.iter().enumerate() {
        if !ALLOWED_ELEMENTS.contains(&atom.element) {
            return Err(format!(
                "atom {i} has element {} outside the allowed set {ALLOWED_ELEMENTS:?}",
                atom.element
            ));
        }
        if atom.charge.unsigned_abs() > MAX_ABS_CHARGE as u8 {
            return Err(format!(
                "atom {i} has charge {} outside the allowed range +-{MAX_ABS_CHARGE}",
                atom.charge
            ));
        }
    }
    for (bi, b) in m.bonds.iter().enumerate() {
        if b.a >= m.atoms.len() || b.b >= m.atoms.len() {
            return Err(format!(
                "bond {bi} references out-of-range atom index ({}, {}) for {} atoms",
                b.a,
                b.b,
                m.atoms.len()
            ));
        }
        if b.a == b.b {
            return Err(format!("bond {bi} is a self-loop on atom {}", b.a));
        }
    }
    // Duplicate bonds between the same atom pair are chemically invalid --
    // two independent single bonds between the same two atoms isn't a real
    // structure, it's either a data-modeling mistake or should have been a
    // single higher-order bond.
    for i in 0..m.bonds.len() {
        for j in (i + 1)..m.bonds.len() {
            let (a1, b1) = (m.bonds[i].a, m.bonds[i].b);
            let (a2, b2) = (m.bonds[j].a, m.bonds[j].b);
            if (a1, b1) == (a2, b2) || (a1, b1) == (b2, a2) {
                return Err(format!(
                    "duplicate bond between atoms {a1} and {b1} (bond indices {i} and {j})"
                ));
            }
        }
    }
    // A single molecule must be one connected component. A disconnected
    // graph is really multiple molecules, not one -- found while designing
    // the Phase 1.2 adversarial tests: a hand-built two-disconnected-rings
    // structure collides with a genuine one-ring structure under
    // `structural_key` (same formula, same per-atom local invariants) and,
    // before this check, would have passed validity too. Rejecting
    // disconnected inputs here closes that specific attack surface at the
    // source rather than relying on the (separately fixed) exact-isomorphism
    // check alone.
    if !m.atoms.is_empty() && m.connected_components() != 1 {
        return Err(format!(
            "molecule has {} connected components, expected exactly 1",
            m.connected_components()
        ));
    }

    let mut bond_sum = vec![0.0f64; m.atoms.len()];
    let mut has_aromatic_bond = vec![false; m.atoms.len()];
    for b in &m.bonds {
        let c = b.order.valence_contribution();
        bond_sum[b.a] += c;
        bond_sum[b.b] += c;
        if b.order == BondOrder::Aromatic {
            has_aromatic_bond[b.a] = true;
            has_aromatic_bond[b.b] = true;
        }
    }
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.charge != 0 {
            continue; // charged-atom valence rules are more complex; not checked here
        }
        if has_aromatic_bond[i] {
            // The naive `1.5 * aromatic_bond_count` sum used below is only
            // correct for a plain 2-ring-bond atom; it silently rejects real
            // ring-fusion atoms and overcounts "pyrrole-type" heteroatoms.
            // `aromaticity::check_aromatic_valence` validates every
            // aromatic atom in this molecule (not just this one) via a real
            // Kekule-matching search, so once it's run once below the loop
            // there's nothing further to check here for this atom.
            continue;
        }
        let Some(e) = element::lookup(atom.element) else {
            continue; // already caught by the allowed-elements check above
        };
        let total = bond_sum[i] + atom.hydrogens as f64;
        if (total - e.normal_valence as f64).abs() > 1e-9 {
            return Err(format!(
                "atom {i} ({}) has bonded valence {} + {} H = {total}, expected {}",
                atom.element, bond_sum[i], atom.hydrogens, e.normal_valence
            ));
        }
    }
    aromaticity::check_aromatic_valence(m)?;
    Ok(())
}

/// Element (heavy + implicit H) counts, as an order-independent multiset.
fn element_counts(m: &Molecule) -> Vec<(&'static str, u32)> {
    let mut counts: Vec<(&'static str, u32)> = Vec::new();
    let mut bump = |sym: &'static str, n: u32| {
        if let Some(entry) = counts.iter_mut().find(|(s, _)| *s == sym) {
            entry.1 += n;
        } else {
            counts.push((sym, n));
        }
    };
    for a in &m.atoms {
        bump(a.element, 1);
        if a.hydrogens > 0 {
            bump("H", a.hydrogens as u32);
        }
    }
    counts.sort();
    counts
}

fn total_counts(mols: &[Molecule]) -> Vec<(&'static str, u32)> {
    let mut merged: Vec<(&'static str, u32)> = Vec::new();
    for m in mols {
        for (sym, n) in element_counts(m) {
            if let Some(entry) = merged.iter_mut().find(|(s, _)| *s == sym) {
                entry.1 += n;
            } else {
                merged.push((sym, n));
            }
        }
    }
    merged.sort();
    merged
}

/// The reaction-level invariant that actually catches template bugs: total
/// element counts and total net charge must match exactly between reactants
/// and products. Every `ReactionTemplate` should already satisfy this by
/// construction (see `templates.rs`'s mass-balance doc comments) -- this
/// check exists so a *future* template that doesn't gets caught here rather
/// than silently producing an unbalanced candidate.
pub fn check_conservation(candidate: &ReactionCandidate) -> Result<(), String> {
    let lhs = total_counts(&candidate.reactants);
    let rhs = total_counts(&candidate.products);
    if lhs != rhs {
        return Err(format!(
            "element counts not conserved: reactants={lhs:?} products={rhs:?}"
        ));
    }
    let lhs_charge: i64 = candidate
        .reactants
        .iter()
        .flat_map(|m| m.atoms.iter())
        .map(|a| a.charge as i64)
        .sum();
    let rhs_charge: i64 = candidate
        .products
        .iter()
        .flat_map(|m| m.atoms.iter())
        .map(|a| a.charge as i64)
        .sum();
    if lhs_charge != rhs_charge {
        return Err(format!(
            "net charge not conserved: reactants={lhs_charge} products={rhs_charge}"
        ));
    }
    Ok(())
}

/// Full validity check for a candidate: every reactant and product molecule
/// individually, then the reaction-level conservation invariant.
pub fn check_candidate(candidate: &ReactionCandidate) -> Result<(), String> {
    for m in candidate.reactants.iter().chain(candidate.products.iter()) {
        check_molecule(m)?;
    }
    check_conservation(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::{
        EsterificationTemplate, HydrogenationTemplate, ReactionTemplate, molecular_hydrogen,
    };
    use symthaea_organic_chemistry::smiles::{Atom, Bond, BondOrder};

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn benign_molecule_passes() {
        assert!(check_molecule(&mol("CCO")).is_ok());
        assert!(check_molecule(&mol("c1ccccc1")).is_ok());
    }

    #[test]
    fn disallowed_element_rejected() {
        // Iodine is in organic-chemistry's SMILES subset but outside this
        // crate's allowed element set (Phase A.7: H/C/N/O/F/Si/P/S/Cl/Br --
        // iodine has no STO-3G basis data and is chemically rare enough in
        // this corpus's reaction classes not to warrant adding it).
        let m = mol("CCI");
        assert!(check_molecule(&m).is_err());
    }

    #[test]
    fn out_of_range_bond_index_rejected() {
        let bad = Molecule {
            atoms: vec![Atom {
                element: "C",
                aromatic: false,
                charge: 0,
                hydrogens: 4,
            }],
            bonds: vec![Bond {
                a: 0,
                b: 5, // out of range
                order: BondOrder::Single,
            }],
        };
        assert!(check_molecule(&bad).is_err());
    }

    #[test]
    fn valence_violation_rejected() {
        // Carbon with 5 bonds worth of valence (impossible, normal valence 4).
        let bad = Molecule {
            atoms: vec![
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 4,
                },
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                },
            ],
            bonds: vec![Bond {
                a: 0,
                b: 1,
                order: BondOrder::Single,
            }],
        };
        // atom 0: bond_sum=1.0 + hydrogens=4 = 5, expected 4 -- violates valence.
        assert!(check_molecule(&bad).is_err());
    }

    #[test]
    fn disconnected_graph_rejected() {
        // Two separate 3-membered carbon rings as ONE "molecule" -- this is
        // exactly the pathological input `structural_key` (policy.rs) can
        // collide with a real single 6-ring on (same formula, same local
        // per-atom invariants). Rejecting it here at validity closes that
        // attack surface at the source.
        let c = |h: u8| Atom {
            element: "C",
            aromatic: false,
            charge: 0,
            hydrogens: h,
        };
        let b = |a: usize, b: usize| Bond {
            a,
            b,
            order: BondOrder::Single,
        };
        let two_disjoint_triangles = Molecule {
            atoms: vec![c(2), c(2), c(2), c(2), c(2), c(2)],
            bonds: vec![
                b(0, 1),
                b(1, 2),
                b(2, 0), // ring 1: atoms 0,1,2
                b(3, 4),
                b(4, 5),
                b(5, 3), // ring 2: atoms 3,4,5 -- disconnected from ring 1
            ],
        };
        assert_eq!(two_disjoint_triangles.connected_components(), 2);
        assert!(check_molecule(&two_disjoint_triangles).is_err());
    }

    #[test]
    fn duplicate_bond_rejected() {
        let bad = Molecule {
            atoms: vec![
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                },
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                },
            ],
            bonds: vec![
                Bond {
                    a: 0,
                    b: 1,
                    order: BondOrder::Single,
                },
                Bond {
                    a: 0,
                    b: 1,
                    order: BondOrder::Single,
                }, // duplicate of the same pair
            ],
        };
        assert!(check_molecule(&bad).is_err());
    }

    #[test]
    fn excessive_charge_rejected() {
        let bad = Molecule {
            atoms: vec![Atom {
                element: "N",
                aromatic: false,
                charge: 100, // absurd -- i8 permits it, chemistry doesn't
                hydrogens: 0,
            }],
            bonds: vec![],
        };
        assert!(check_molecule(&bad).is_err());
    }

    // The charged-atom valence exemption (`if atom.charge != 0 { continue }`
    // above) went from theoretical surface area to actively load-bearing
    // once `normalization.rs` started routing real molecules through it
    // (Phase A.1) -- an external review asked for both directions to be
    // tested explicitly, not just the "happy path" the normalization
    // equivalence test already covers indirectly.

    #[test]
    fn legitimate_charge_separated_structure_passes_without_going_through_normalization() {
        // Fed directly to check_molecule, bypassing normalization.rs
        // entirely -- confirms the exemption itself accepts real,
        // already-correct charge-separated chemistry on its own, not only
        // as normalization's downstream beneficiary.
        assert!(check_molecule(&mol("C[N+](=O)[O-]")).is_ok());
    }

    #[test]
    fn malformed_charged_atom_still_rejected_by_the_checks_that_do_apply_to_it() {
        // The valence check specifically skips charged atoms, but every
        // OTHER structural check still applies to them: element allowlist,
        // charge-magnitude bound, and bond-index sanity are all evaluated
        // before the per-atom loop reaches the charge-skip branch (element/
        // charge checks) or don't depend on charge at all (bond checks).
        let disallowed_element_charged = Molecule {
            atoms: vec![Atom {
                element: "I", // outside ALLOWED_ELEMENTS, charge irrelevant
                aromatic: false,
                charge: 1,
                hydrogens: 0,
            }],
            bonds: vec![],
        };
        assert!(check_molecule(&disallowed_element_charged).is_err());

        let excessive_charge = Molecule {
            atoms: vec![Atom {
                element: "N",
                aromatic: false,
                charge: 100,
                hydrogens: 0,
            }],
            bonds: vec![],
        };
        assert!(check_molecule(&excessive_charge).is_err());
    }

    #[test]
    fn malformed_valence_on_a_charged_atom_is_not_currently_caught_documenting_the_known_gap() {
        // Honest documentation of the residual risk this exemption
        // reopened (see PROCESS_DISCOVERY_THREAT_MODEL's "Residual risks"
        // section): a charged atom with an absurd bonded valence -- N+
        // (which even ionized should never bond 4 separate carbons this
        // way) with FOUR single bonds -- currently PASSES, because the
        // valence check is unconditionally skipped once `atom.charge != 0`.
        // This is not a bug this pass fixes (a real charged-atom valence
        // model is separate, future work) -- this test exists so that if
        // someone later adds that model, this assertion breaks and tells
        // them exactly what changed, rather than the gap silently
        // persisting unnoticed.
        let c = |h: u8| Atom {
            element: "C",
            aromatic: false,
            charge: 0,
            hydrogens: h,
        };
        let bad_valence_but_charged = Molecule {
            atoms: vec![
                Atom {
                    element: "N",
                    aromatic: false,
                    charge: 1,
                    hydrogens: 0,
                },
                c(3),
                c(3),
                c(3),
                c(3),
            ],
            bonds: (1..=4)
                .map(|i| Bond {
                    a: 0,
                    b: i,
                    order: BondOrder::Single,
                })
                .collect(),
        };
        assert!(
            check_molecule(&bad_valence_but_charged).is_ok(),
            "if this now fails, a charged-atom valence check was added -- update the threat \
             model's residual-risks section to remove this gap, then delete this test"
        );
    }

    #[test]
    fn esterification_conserves_elements_and_charge() {
        let acid = mol("CC(=O)O");
        let alcohol = mol("CCO");
        let products = EsterificationTemplate
            .apply(&[acid.clone(), alcohol.clone()])
            .unwrap();
        let candidate = ReactionCandidate {
            reactants: vec![acid, alcohol],
            products,
            template: "esterification",
        };
        assert!(check_candidate(&candidate).is_ok());
    }

    #[test]
    fn hydrogenation_conserves_elements_and_charge() {
        let ethylene = mol("C=C");
        let h2 = molecular_hydrogen();
        let products = HydrogenationTemplate
            .apply(&[ethylene.clone(), h2.clone()])
            .unwrap();
        let candidate = ReactionCandidate {
            reactants: vec![ethylene, h2],
            products,
            template: "hydrogenation",
        };
        assert!(check_candidate(&candidate).is_ok());
    }

    #[test]
    fn unconserved_elements_rejected() {
        let candidate = ReactionCandidate {
            reactants: vec![mol("CCO")],
            products: vec![mol("CCC")], // different formula -- a template bug
            template: "buggy",
        };
        assert!(check_candidate(&candidate).is_err());
    }

    /// Seed pool for the property tests below: real, known-valid molecules
    /// (the Phase 0 feedstocks plus a couple of extras) rather than fully
    /// random atom/bond soup -- generating an arbitrary *valid* organic
    /// molecule from scratch is a much bigger undertaking than this
    /// assurance pass needs; sampling real valid structures and then
    /// deliberately corrupting them exercises the same soundness property
    /// (validity accepts real molecules, rejects broken ones) without that
    /// extra machinery.
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
        "O",
        "CCC",
        "CCCC",
    ];

    proptest::proptest! {
        /// Every real, unmodified seed molecule passes validity -- the
        /// "doesn't reject good input" half of soundness.
        #[test]
        fn prop_valid_seed_molecules_always_pass(idx in 0usize..SEED_SMILES.len()) {
            let m = mol(SEED_SMILES[idx]);
            proptest::prop_assert!(check_molecule(&m).is_ok());
        }

        /// Every one of 5 deliberate corruption kinds, applied to any seed
        /// molecule, is always caught -- the "does reject broken input"
        /// half of soundness, exercised across many random (seed,
        /// corruption, magnitude) combinations rather than the fixed
        /// hand-picked cases above.
        #[test]
        fn prop_corrupted_molecules_are_always_rejected(
            idx in 0usize..SEED_SMILES.len(),
            corruption in 0usize..5,
            delta in 1u8..5,
        ) {
            let mut m = mol(SEED_SMILES[idx]);
            if m.atoms.is_empty() {
                return Ok(());
            }
            match corruption {
                0 => {
                    // Break valence: bump one atom's hydrogen count.
                    m.atoms[0].hydrogens = m.atoms[0].hydrogens.saturating_add(delta);
                }
                1 => {
                    // Disallowed element (I is in organic-chemistry's SMILES
                    // subset but outside this crate's allowed set).
                    m.atoms[0].element = "I";
                }
                2 => {
                    // Out-of-range bond index.
                    m.bonds.push(Bond {
                        a: 0,
                        b: m.atoms.len() + 5,
                        order: BondOrder::Single,
                    });
                }
                3 => {
                    // Duplicate an existing bond.
                    if let Some(b) = m.bonds.first().cloned() {
                        m.bonds.push(b);
                    } else {
                        return Ok(());
                    }
                }
                4 => {
                    // Disconnect: append a self-contained but separate atom.
                    m.atoms.push(Atom {
                        element: "C",
                        aromatic: false,
                        charge: 0,
                        hydrogens: 4,
                    });
                }
                _ => unreachable!(),
            }
            proptest::prop_assert!(check_molecule(&m).is_err());
        }
    }
}
