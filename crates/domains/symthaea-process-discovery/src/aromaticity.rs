// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Correct per-atom valence checking for aromatic atoms, replacing
//! `validity.rs`'s original naive "each aromatic bond contributes 1.5
//! valence" sum.
//!
//! **Why the naive sum is wrong.** The 1.5-per-aromatic-bond average
//! (`BondOrder::valence_contribution`) is only numerically correct for the
//! common case of a plain ring atom with exactly 2 aromatic bonds, because
//! `1.5 * 2 = 3.0` happens to land on an integer. It silently breaks for two
//! real, frequently-occurring structures, both confirmed against real USPTO
//! data during the Phase A.3 evaluation (`PROCESS_DISCOVERY_PHASE_A3_ADJUDICATION_2026-07-13.md`,
//! `PROCESS_DISCOVERY_PHASE_A3_V2_ADJUDICATION_2026-07-15.md`):
//! - **Ring-fusion atoms** (e.g. naphthalene's shared carbons) have 3
//!   aromatic bonds. `1.5 * 3 = 4.5`, a fractional total that can never equal
//!   an integer `normal_valence`, so the naive check always rejects these
//!   even when the structure is perfectly valid (real Kekule structures give
//!   these atoms exactly one double bond among their three ring bonds,
//!   summing to 1+1+2=4).
//! - **"Pyrrole-type" heteroatoms** (e.g. the substituted nitrogen in
//!   N-methylpyrazole) donate a lone pair to the ring instead of forming a
//!   ring double bond -- both their ring bonds are single in every valid
//!   Kekule structure. The naive model still charges them 1.5 per ring bond,
//!   overcounting by a full valence unit.
//!
//! **The fix**: compute, per neutral aromatic atom, how many of its aromatic
//! bonds must be "double" for its own valence to work out (0 or 1 -- this is
//! `aromatic_demand` below, a purely local quantity). Then search for a
//! perfect matching over the demand-1 atoms using only aromatic edges
//! between two demand-1 atoms -- this is exactly the graph-theoretic
//! condition for "a valid Kekule (alternating double-bond) assignment
//! exists." If it does, every atom's real valence is achievable
//! simultaneously and the molecule is accepted; if it provably doesn't
//! (odd demand-1 count, or the matching search is exhausted), it's rejected
//! as a genuine aromaticity violation, not a naive-model artifact.
//!
//! Matching is general-graph (not assumed bipartite), since fused
//! odd-membered rings (pyrrole/furan fused to a 6-ring) make some real
//! aromatic systems non-bipartite. Bounded and fail-closed, mirroring
//! `isomorphism.rs`'s `MAX_ATOMS_FOR_EXACT_CHECK`/`MAX_BACKTRACK_STEPS`
//! precedent: exceeding either bound is reported as a failure, never
//! silently accepted.

use std::collections::HashMap;
use symthaea_organic_chemistry::element;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

/// Upper bound on the number of demand-1 atoms considered in one matching
/// search. Real aromatic systems in organic molecules are small (a handful
/// of fused rings); this is generous headroom while keeping worst-case
/// backtracking cheap.
const MAX_DEMAND_ATOMS: usize = 32;

/// Shared step budget across one matching search. Low-degree ring graphs
/// (each atom has 2-3 aromatic neighbors) are fast to search; this bound
/// exists purely as a defense against a pathological/adversarial input, not
/// because real inputs are expected to approach it.
const MAX_BACKTRACK_STEPS: u64 = 200_000;

/// Checks that every neutral atom with at least one aromatic bond has a
/// chemically achievable, mutually-consistent Kekule valence assignment.
/// Atoms with no aromatic bonds are untouched by this function (their
/// valence is checked by `validity::check_molecule`'s exact-integer path).
/// Charged atoms are skipped, matching `check_molecule`'s existing
/// charged-atom exemption -- their bonds still count toward neighbors'
/// local sums, but bonds incident to a charged atom are never used as a
/// candidate double bond in the matching (that atom's own Kekule role isn't
/// being validated here, so nothing should rely on it).
pub fn check_aromatic_valence(m: &Molecule) -> Result<(), String> {
    let mut aromatic_neighbors: Vec<Vec<usize>> = vec![Vec::new(); m.atoms.len()];
    let mut any_aromatic = false;
    for b in &m.bonds {
        if b.order == BondOrder::Aromatic {
            aromatic_neighbors[b.a].push(b.b);
            aromatic_neighbors[b.b].push(b.a);
            any_aromatic = true;
        }
    }
    if !any_aromatic {
        return Ok(());
    }

    // demand[i]: None = not aromatic, not neutral-and-known, or excluded
    // (charged) -- never placed in the matching. Some(0)/Some(1) = a real,
    // locally-computed requirement.
    let mut demand: Vec<Option<u8>> = vec![None; m.atoms.len()];
    for (i, atom) in m.atoms.iter().enumerate() {
        if aromatic_neighbors[i].is_empty() {
            continue;
        }
        if atom.charge != 0 {
            continue; // matches check_molecule's charged-atom exemption
        }
        let Some(e) = element::lookup(atom.element) else {
            continue; // already caught by the allowed-elements check
        };
        let non_aromatic_sum: f64 = m
            .neighbors(i)
            .iter()
            .filter(|(_, o)| *o != BondOrder::Aromatic)
            .map(|(_, o)| o.valence_contribution())
            .sum();
        let aromatic_count = aromatic_neighbors[i].len() as f64;
        let capacity = e.normal_valence as f64 - non_aromatic_sum - atom.hydrogens as f64;
        let d = capacity - aromatic_count;
        if (d - 0.0).abs() < 1e-9 {
            demand[i] = Some(0);
        } else if (d - 1.0).abs() < 1e-9 {
            demand[i] = Some(1);
        } else {
            return Err(format!(
                "atom {i} ({}) has an aromatic valence demand of {d} (expected exactly 0 or 1 \
                 double bond among its aromatic ring bonds) -- not a chemically valid aromatic \
                 assignment",
                atom.element
            ));
        }
    }

    let demand1: Vec<usize> = (0..m.atoms.len())
        .filter(|&i| demand[i] == Some(1))
        .collect();
    if demand1.len() > MAX_DEMAND_ATOMS {
        return Err(format!(
            "aromatic ring system has {} atoms requiring a double bond, exceeding the bounded \
             Kekulization limit of {MAX_DEMAND_ATOMS} -- cannot verify, failing closed",
            demand1.len()
        ));
    }
    if demand1.len() % 2 != 0 {
        return Err(format!(
            "aromatic ring system has an odd number ({}) of atoms each requiring exactly one \
             double bond -- no valid Kekule assignment can exist",
            demand1.len()
        ));
    }

    let index_of: HashMap<usize, usize> = demand1
        .iter()
        .enumerate()
        .map(|(local, &atom_idx)| (atom_idx, local))
        .collect();
    let n = demand1.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (local_i, &atom_idx) in demand1.iter().enumerate() {
        for &nbr in &aromatic_neighbors[atom_idx] {
            if let Some(&local_j) = index_of.get(&nbr) {
                adj[local_i].push(local_j);
            }
        }
    }

    let mut matched = vec![false; n];
    let mut steps: u64 = 0;
    if find_perfect_matching(&adj, &mut matched, &mut steps) {
        return Ok(());
    }
    if steps >= MAX_BACKTRACK_STEPS {
        return Err(format!(
            "aromatic Kekule-matching search exceeded the {MAX_BACKTRACK_STEPS}-step budget -- \
             cannot verify, failing closed"
        ));
    }
    Err(
        "no valid Kekule (alternating double-bond) assignment exists for this aromatic ring \
         system"
            .to_string(),
    )
}

/// Backtracking search for a perfect matching over `adj` (undirected
/// adjacency by local index into the demand-1 atom set). General graphs, not
/// just bipartite ones, are supported -- fused polycyclic aromatics
/// involving an odd-membered ring (pyrrole/furan fused to a benzo ring) are
/// not bipartite. `steps` is a shared budget counter for the whole search;
/// the caller distinguishes "search exhausted, no matching exists" from
/// "budget exceeded" by checking it against `MAX_BACKTRACK_STEPS` after this
/// returns `false`.
fn find_perfect_matching(adj: &[Vec<usize>], matched: &mut [bool], steps: &mut u64) -> bool {
    let Some(i) = matched.iter().position(|&done| !done) else {
        return true; // every demand-1 atom is matched
    };
    for &j in &adj[i] {
        if matched[j] {
            continue;
        }
        *steps += 1;
        if *steps > MAX_BACKTRACK_STEPS {
            return false;
        }
        matched[i] = true;
        matched[j] = true;
        if find_perfect_matching(adj, matched, steps) {
            return true;
        }
        matched[i] = false;
        matched[j] = false;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_organic_chemistry::smiles::Molecule;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap_or_else(|e| panic!("failed to parse {s:?}: {e}"))
    }

    #[test]
    fn plain_benzene_ring_still_passes() {
        assert!(check_aromatic_valence(&mol("c1ccccc1")).is_ok());
    }

    #[test]
    fn naphthalene_ring_fusion_carbons_now_pass() {
        // Each fusion carbon has 3 aromatic bonds -- this is exactly the
        // case the naive 1.5-per-bond sum could never satisfy (4.5 != 4).
        assert!(check_aromatic_valence(&mol("c1ccc2ccccc2c1")).is_ok());
    }

    #[test]
    fn pyridine_type_nitrogen_passes() {
        // Ring N with no substituent/H: pyridine-type, needs one double bond.
        assert!(check_aromatic_valence(&mol("c1ccncc1")).is_ok());
    }

    #[test]
    fn pyrrole_type_nitrogen_passes() {
        // Ring N-H: pyrrole-type, donates its lone pair, needs zero double
        // bonds among its ring bonds -- the naive model overcounts this by
        // a full valence unit (see module doc).
        assert!(check_aromatic_valence(&mol("c1cc[nH]c1")).is_ok());
    }

    #[test]
    fn n_substituted_pyrazole_with_pyridine_type_second_nitrogen_passes() {
        // The real record (row 97 of the frozen USPTO evaluation) this fix
        // targets: N-methylpyrazole ring, N1 pyrrole-type (substituted),
        // N2 pyridine-type.
        assert!(check_aromatic_valence(&mol("Cn1ccc([N+](=O)[O-])n1")).is_ok());
    }

    #[test]
    fn non_aromatic_molecule_is_a_no_op() {
        assert!(check_aromatic_valence(&mol("CCO")).is_ok());
    }

    #[test]
    fn aromatic_carbon_with_impossible_demand_is_rejected() {
        // Hand-built: an aromatic carbon with 3 non-aromatic single-bond
        // substituents plus 2 aromatic ring bonds. capacity = 4-3-0 = 1,
        // aromatic_count = 2, demand = -1 -- outside {0,1}, must reject.
        use symthaea_organic_chemistry::smiles::{Atom, Bond};
        let m = Molecule {
            atoms: vec![
                Atom {
                    element: "C",
                    aromatic: true,
                    charge: 0,
                    hydrogens: 0,
                }, // 0: the overloaded ring atom
                Atom {
                    element: "C",
                    aromatic: true,
                    charge: 0,
                    hydrogens: 1,
                }, // 1: ring partner
                Atom {
                    element: "C",
                    aromatic: true,
                    charge: 0,
                    hydrogens: 1,
                }, // 2: ring partner
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                }, // 3: substituent A
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                }, // 4: substituent B
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                }, // 5: substituent C
            ],
            bonds: vec![
                Bond {
                    a: 0,
                    b: 1,
                    order: BondOrder::Aromatic,
                },
                Bond {
                    a: 0,
                    b: 2,
                    order: BondOrder::Aromatic,
                },
                Bond {
                    a: 0,
                    b: 3,
                    order: BondOrder::Single,
                },
                Bond {
                    a: 0,
                    b: 4,
                    order: BondOrder::Single,
                },
                Bond {
                    a: 0,
                    b: 5,
                    order: BondOrder::Single,
                },
            ],
        };
        let err = check_aromatic_valence(&m).unwrap_err();
        assert!(
            err.contains("aromatic valence demand"),
            "expected an aromatic-demand error, got: {err}"
        );
    }

    #[test]
    fn matching_search_correctly_rejects_an_unmatchable_even_demand_set() {
        // Star graph: node 0 adjacent to 1, 2, 3; no edges among 1/2/3. All
        // 4 demand-1 -- even count, but node 0 can only match one of its
        // three neighbors, stranding the other two with no edge between
        // them. Tests the matcher directly (real chemistry rarely produces
        // this shape, but the algorithm must handle it correctly).
        let adj = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
        let mut matched = vec![false; 4];
        let mut steps = 0u64;
        assert!(!find_perfect_matching(&adj, &mut matched, &mut steps));
    }

    #[test]
    fn matching_search_finds_a_path_graph_matching() {
        // Path 0-1-2-3 (pyrrole's 4 demand-1 carbons after excluding the
        // pyrrole-type N): a valid perfect matching {0-1, 2-3} exists.
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let mut matched = vec![false; 4];
        let mut steps = 0u64;
        assert!(find_perfect_matching(&adj, &mut matched, &mut steps));
        assert!(matched.iter().all(|&m| m));
    }
}
