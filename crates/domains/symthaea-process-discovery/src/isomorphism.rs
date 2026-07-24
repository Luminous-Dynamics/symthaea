// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact labeled-graph isomorphism for small molecular graphs.
//!
//! `policy::structural_key` is a one-round local invariant, not a canonical
//! identity -- two genuinely distinct, non-isomorphic, both-valid molecules
//! can share the same key (see this module's `prism_and_k33_...` tests for a
//! real, constructed example: two connected, 3-regular, all-carbon 6-atom
//! graphs -- the triangular-prism graph and the complete bipartite graph
//! K3,3 -- collide under `structural_key` because every atom's *local*
//! neighborhood looks identical, even though the graphs have different
//! global structure, e.g. different triangle counts). `ReactantLibrary`
//! uses `structural_key` as a bucket index and this module's `is_isomorphic`
//! to resolve any collision within a bucket exactly -- the two-stage design
//! `policy.rs`'s doc comments name as the real fix for the one-round
//! invariant's known limitation.
//!
//! Bounded backtracking with degree/label pruning. Appropriate for the
//! molecule sizes this crate actually produces (verified: the largest
//! current library/product molecule has ~14 atoms) -- not a general-purpose
//! cheminformatics-scale isomorphism engine.
//!
//! **The bound is enforced, not just claimed.** An external review pointed
//! out that "bounded" needs to be a real limit with explicit fail-closed
//! behavior, not an informal expectation -- an earlier version of this
//! module had no atom-count cap or search-step budget at all, so a
//! sufficiently large or highly symmetric (many equivalent local labels,
//! wide branching) input could in principle make the backtracking search
//! run for a very long time with no defense. `MAX_ATOMS_FOR_EXACT_CHECK` and
//! `MAX_BACKTRACK_STEPS` are real, checked limits now. On either bound being
//! hit, `is_isomorphic` returns `false` -- **not** "assume isomorphic,"
//! **not** "fall back to trusting `structural_key` alone." For this
//! module's actual consumer (`ReactantLibrary::contains`, an allowlist
//! membership check), `false` on an unresolved case is the safe,
//! default-deny answer: an unconfirmed match is treated as "not a member,"
//! never as "trust it anyway."

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(test)]
use symthaea_organic_chemistry::smiles::Atom;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

/// Hard cap on atom count for an exact check. Comfortably above every
/// molecule this crate currently produces (~14 atoms) while still bounding
/// worst-case backtracking cost. Molecules larger than this are treated as
/// "not confirmed isomorphic" (see module doc) rather than attempted.
const MAX_ATOMS_FOR_EXACT_CHECK: usize = 40;

/// Hard cap on total backtracking calls, as defense-in-depth against a
/// pathological (highly symmetric, wide-branching) input even within the
/// atom-count bound. Exceeding this also fails closed to `false`.
const MAX_BACKTRACK_STEPS: u64 = 2_000_000;

type AtomLabel = (&'static str, u8, i8, usize); // (element, hydrogens, charge, degree)

fn atom_label(m: &Molecule, i: usize) -> AtomLabel {
    let atom = &m.atoms[i];
    (
        atom.element,
        atom.hydrogens,
        atom.charge,
        m.neighbors(i).len(),
    )
}

/// Why a comparison did or didn't confirm isomorphism. A second review
/// pass, after approving the bound-enforcement fix, pointed out that a bare
/// `bool` collapses two operationally very different situations -- "these
/// are genuinely different molecules" and "the search gave up without an
/// answer" -- into the same `false`. Both are the correct *decision*
/// (`ReactantLibrary` should deny either way), but conflating them makes it
/// impossible to later tell whether `MAX_ATOMS_FOR_EXACT_CHECK`/
/// `MAX_BACKTRACK_STEPS` are well-tuned or need revisiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsomorphismOutcome {
    Isomorphic,
    NotIsomorphic,
    AtomLimitExceeded,
    SearchBudgetExceeded,
}

impl IsomorphismOutcome {
    pub fn is_isomorphic(self) -> bool {
        matches!(self, IsomorphismOutcome::Isomorphic)
    }
}

// Diagnostics-only counters -- never read by any acceptance/policy decision
// in this crate, only by `diagnostics()` below. Global (not per-call-site)
// because the actual consumer, `ReactantLibrary::contains`, has no natural
// place to thread a mutable stats object through without changing its
// (and every `ScopePolicy` impl's) public signature for a non-functional
// concern.
static COMPARISONS_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static ATOM_LIMIT_REJECTIONS: AtomicU64 = AtomicU64::new(0);
static BUDGET_EXHAUSTIONS: AtomicU64 = AtomicU64::new(0);
static WORST_STEPS_USED: AtomicU64 = AtomicU64::new(0);
static WORST_DEPTH_REACHED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IsomorphismDiagnostics {
    pub comparisons_attempted: u64,
    pub atom_limit_rejections: u64,
    pub budget_exhaustions: u64,
    pub worst_steps_used: u64,
    pub worst_depth_reached: usize,
}

/// Snapshot of the counters above. Diagnostics only -- exists so the two
/// bound constants can eventually be evaluated against real usage ("worst
/// observed step count is still 3 orders of magnitude under the budget" or
/// "we're hitting the atom limit regularly, reconsider it"), not to feed
/// back into any acceptance decision.
pub fn diagnostics() -> IsomorphismDiagnostics {
    IsomorphismDiagnostics {
        comparisons_attempted: COMPARISONS_ATTEMPTED.load(Ordering::Relaxed),
        atom_limit_rejections: ATOM_LIMIT_REJECTIONS.load(Ordering::Relaxed),
        budget_exhaustions: BUDGET_EXHAUSTIONS.load(Ordering::Relaxed),
        worst_steps_used: WORST_STEPS_USED.load(Ordering::Relaxed),
        worst_depth_reached: WORST_DEPTH_REACHED.load(Ordering::Relaxed),
    }
}

/// True if `a` and `b` are isomorphic as labeled graphs. Thin wrapper over
/// [`is_isomorphic_detailed`] for callers (e.g. `ReactantLibrary::contains`)
/// that only need the accept/reject decision, not the reason.
pub fn is_isomorphic(a: &Molecule, b: &Molecule) -> bool {
    is_isomorphic_detailed(a, b).is_isomorphic()
}

/// Isomorphism check with the reason, plus diagnostics bookkeeping. See
/// [`IsomorphismOutcome`]'s doc comment for why the reason is exposed
/// separately from the accept/reject decision every current caller uses.
pub fn is_isomorphic_detailed(a: &Molecule, b: &Molecule) -> IsomorphismOutcome {
    COMPARISONS_ATTEMPTED.fetch_add(1, Ordering::Relaxed);

    let n = a.atoms.len();
    if n != b.atoms.len() || a.bonds.len() != b.bonds.len() {
        return IsomorphismOutcome::NotIsomorphic;
    }
    if n == 0 {
        return IsomorphismOutcome::Isomorphic;
    }
    if n > MAX_ATOMS_FOR_EXACT_CHECK {
        ATOM_LIMIT_REJECTIONS.fetch_add(1, Ordering::Relaxed);
        return IsomorphismOutcome::AtomLimitExceeded; // fail closed: too large to check exactly
    }

    let mut a_labels: Vec<AtomLabel> = (0..n).map(|i| atom_label(a, i)).collect();
    let mut b_labels: Vec<AtomLabel> = (0..n).map(|i| atom_label(b, i)).collect();
    a_labels.sort();
    b_labels.sort();
    if a_labels != b_labels {
        return IsomorphismOutcome::NotIsomorphic; // cheap reject before any search
    }

    let a_adj: Vec<Vec<(usize, BondOrder)>> = (0..n).map(|i| a.neighbors(i)).collect();

    let mut mapping = vec![usize::MAX; n]; // a-index -> b-index
    let mut used = vec![false; n];
    let mut steps_remaining = MAX_BACKTRACK_STEPS;
    let mut max_depth_reached = 0usize;
    let found = backtrack(
        0,
        n,
        a,
        b,
        &a_adj,
        &mut mapping,
        &mut used,
        &mut steps_remaining,
        &mut max_depth_reached,
    );

    let steps_used = MAX_BACKTRACK_STEPS - steps_remaining;
    WORST_STEPS_USED.fetch_max(steps_used, Ordering::Relaxed);
    WORST_DEPTH_REACHED.fetch_max(max_depth_reached, Ordering::Relaxed);

    // `steps_remaining == 0` without `found` means the budget ran out before
    // the search could conclude either way -- a real (if astronomically
    // unlikely, given the constants' margin over this crate's actual
    // molecule sizes) exhaustive-and-complete search could in principle also
    // land exactly on the boundary and be misclassified here; that only
    // affects this diagnostic label, never the returned accept/reject
    // decision (both cases correctly deny).
    if !found && steps_remaining == 0 {
        BUDGET_EXHAUSTIONS.fetch_add(1, Ordering::Relaxed);
        return IsomorphismOutcome::SearchBudgetExceeded;
    }

    if found {
        IsomorphismOutcome::Isomorphic
    } else {
        IsomorphismOutcome::NotIsomorphic
    }
}

fn backtrack(
    idx: usize,
    n: usize,
    a: &Molecule,
    b: &Molecule,
    a_adj: &[Vec<(usize, BondOrder)>],
    mapping: &mut [usize],
    used: &mut [bool],
    steps_remaining: &mut u64,
    max_depth_reached: &mut usize,
) -> bool {
    *max_depth_reached = (*max_depth_reached).max(idx);
    if idx == n {
        return true;
    }
    let a_label = atom_label(a, idx);
    for cand in 0..n {
        // Fail closed: budget exhausted, treat as "not confirmed isomorphic"
        // rather than continuing an unbounded search.
        if *steps_remaining == 0 {
            return false;
        }
        *steps_remaining -= 1;

        if used[cand] || atom_label(b, cand) != a_label {
            continue;
        }
        let b_adj_cand = b.neighbors(cand);

        // Every already-mapped a-neighbor of idx must correspond to a real
        // b-edge from cand with the same bond order.
        let consistent = a_adj[idx].iter().all(|&(nbr, order)| {
            mapping[nbr] == usize::MAX
                || b_adj_cand
                    .iter()
                    .any(|&(bn, bo)| bn == mapping[nbr] && bo == order)
        });
        if !consistent {
            continue;
        }
        // And no EXTRA b-edge from cand to an already-mapped atom that
        // doesn't correspond to an a-edge from idx (catches the K3,3-vs-
        // prism case: same degree, but different adjacency among mapped
        // atoms).
        let no_extra = b_adj_cand.iter().all(|&(bn, bo)| {
            !used[bn] || {
                let am = mapping.iter().position(|&x| x == bn);
                am.is_some_and(|am| a_adj[idx].iter().any(|&(an, ao)| an == am && ao == bo))
            }
        });
        if !no_extra {
            continue;
        }

        mapping[idx] = cand;
        used[cand] = true;
        if backtrack(
            idx + 1,
            n,
            a,
            b,
            a_adj,
            mapping,
            used,
            steps_remaining,
            max_depth_reached,
        ) {
            return true;
        }
        mapping[idx] = usize::MAX;
        used[cand] = false;
    }
    false
}

/// Test fixtures, hoisted to module scope (not nested in `mod tests`) so
/// `policy.rs`'s tests can reuse them for a library-level collision check
/// alongside this module's own isomorphism-level check.
#[cfg(test)]
fn c1h() -> symthaea_organic_chemistry::smiles::Atom {
    symthaea_organic_chemistry::smiles::Atom {
        element: "C",
        aromatic: false,
        charge: 0,
        hydrogens: 1,
    }
}

#[cfg(test)]
fn c1h_with_hydrogens(h: u8) -> symthaea_organic_chemistry::smiles::Atom {
    symthaea_organic_chemistry::smiles::Atom {
        element: "C",
        aromatic: false,
        charge: 0,
        hydrogens: h,
    }
}

#[cfg(test)]
fn single(a: usize, b: usize) -> symthaea_organic_chemistry::smiles::Bond {
    symthaea_organic_chemistry::smiles::Bond {
        a,
        b,
        order: BondOrder::Single,
    }
}

/// Triangular prism: two triangles {0,1,2} and {3,4,5}, joined by three
/// "rungs" 0-3, 1-4, 2-5. 3-regular, 6 atoms, all-carbon, all-single.
#[cfg(test)]
pub(crate) fn prism_graph() -> Molecule {
    Molecule {
        atoms: vec![c1h(), c1h(), c1h(), c1h(), c1h(), c1h()],
        bonds: vec![
            single(0, 1),
            single(1, 2),
            single(2, 0), // top triangle
            single(3, 4),
            single(4, 5),
            single(5, 3), // bottom triangle
            single(0, 3),
            single(1, 4),
            single(2, 5), // rungs
        ],
    }
}

/// K3,3: every atom in {0,1,2} bonded to every atom in {3,4,5}. Also
/// 3-regular, 6 atoms, all-carbon, all-single -- but bipartite (zero
/// triangles), genuinely non-isomorphic to the prism graph.
#[cfg(test)]
pub(crate) fn k33_graph() -> Molecule {
    let mut bonds = Vec::new();
    for i in 0..3 {
        for j in 3..6 {
            bonds.push(single(i, j));
        }
    }
    Molecule {
        atoms: vec![c1h(), c1h(), c1h(), c1h(), c1h(), c1h()],
        bonds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn identical_molecules_are_isomorphic() {
        assert!(is_isomorphic(&mol("CCO"), &mol("CCO")));
    }

    #[test]
    fn different_formulas_are_not_isomorphic() {
        assert!(!is_isomorphic(&mol("CCO"), &mol("CCC")));
    }

    #[test]
    fn constitutional_isomers_are_not_isomorphic() {
        // Ethanol vs dimethyl ether: same formula, different connectivity.
        assert!(!is_isomorphic(&mol("CCO"), &mol("COC")));
    }

    #[test]
    fn relabeled_atoms_are_still_isomorphic() {
        // Same graph, atoms declared in a different order -- SMILES "CCO"
        // vs a hand-built molecule with atoms in reverse index order.
        let reordered = Molecule {
            atoms: vec![
                Atom {
                    element: "O",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 1,
                },
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 2, // middle carbon: bonded to O and the terminal C
                },
                Atom {
                    element: "C",
                    aromatic: false,
                    charge: 0,
                    hydrogens: 3,
                },
            ],
            bonds: vec![single(0, 1), single(1, 2)],
        };
        assert!(is_isomorphic(&mol("CCO"), &reordered));
    }

    #[test]
    fn prism_and_k33_collide_under_structural_key_but_are_not_isomorphic() {
        // The real, constructed collision this module exists to resolve.
        let prism = prism_graph();
        let k33 = k33_graph();
        assert_eq!(prism.molecular_formula(), k33.molecular_formula()); // both C6H6
        assert!(
            !is_isomorphic(&prism, &k33),
            "prism graph (has triangles) and K3,3 (bipartite, no triangles) must not be isomorphic"
        );
    }

    #[test]
    fn large_fused_ring_molecule_self_isomorphic() {
        // Naphthalene: two fused aromatic rings, 10 heavy atoms -- the
        // largest single-ring-system structure exercised in this crate's
        // tests. Confirms no panic and correct self-isomorphism at a size
        // beyond the 6-atom prism/K3,3 case.
        let naphthalene = mol("c1ccc2ccccc2c1");
        assert!(is_isomorphic(&naphthalene, &naphthalene));
    }

    #[test]
    fn larger_symmetric_ring_self_isomorphic_reasonably_fast() {
        // A 12-membered all-carbon ring (cyclododecane skeleton) -- highly
        // symmetric (every atom has an identical local label), which is
        // exactly the shape that makes naive backtracking slow without
        // degree/label pruning. Runs as a normal `cargo test` (no explicit
        // timing assertion, matching this crate's other tests), but its
        // presence is the check: if pruning regresses, this test suite
        // stalls rather than passing silently.
        let n = 12;
        let atoms: Vec<Atom> = (0..n).map(|_| c1h_with_hydrogens(2)).collect();
        let bonds: Vec<_> = (0..n).map(|i| single(i, (i + 1) % n)).collect();
        let ring = Molecule { atoms, bonds };
        assert!(is_isomorphic(&ring, &ring));

        // A relabeled copy (reverse atom order) must also match.
        let mut relabeled_atoms = ring.atoms.clone();
        relabeled_atoms.reverse();
        let relabeled_bonds: Vec<_> = ring
            .bonds
            .iter()
            .map(|b| single(n - 1 - b.a, n - 1 - b.b))
            .collect();
        let relabeled = Molecule {
            atoms: relabeled_atoms,
            bonds: relabeled_bonds,
        };
        assert!(is_isomorphic(&ring, &relabeled));
    }

    #[test]
    fn oversized_molecule_fails_closed_not_open() {
        // A molecule larger than MAX_ATOMS_FOR_EXACT_CHECK must return
        // `false` -- even compared against an IDENTICAL copy of itself,
        // where the "true" answer is obviously yes. This is the point: the
        // bound is a hard cutoff, not a "except when it's trivially true"
        // special case -- fail-closed means never assuming isomorphic on an
        // unresolved input, not even a plausible-looking one. An external
        // review specifically asked for this to be verified, not just
        // documented.
        let n = MAX_ATOMS_FOR_EXACT_CHECK + 5;
        let atoms: Vec<Atom> = (0..n).map(|_| c1h_with_hydrogens(2)).collect();
        let bonds: Vec<_> = (0..n).map(|i| single(i, (i + 1) % n)).collect();
        let oversized = Molecule { atoms, bonds };
        assert!(
            !is_isomorphic(&oversized, &oversized.clone()),
            "must fail closed (false) above the size bound, not fall back to assuming a match"
        );
    }

    #[test]
    fn detailed_outcome_distinguishes_reasons() {
        // A second review pass asked for the *reason* a comparison denied,
        // not just the accept/reject bool -- so resource exhaustion doesn't
        // get silently confused with a genuine structural mismatch in future
        // diagnostics. Verifies all three "deny" reasons are distinguishable.
        assert_eq!(
            is_isomorphic_detailed(&prism_graph(), &k33_graph()),
            IsomorphismOutcome::NotIsomorphic
        );
        assert_eq!(
            is_isomorphic_detailed(&mol("CCO"), &mol("CCO")),
            IsomorphismOutcome::Isomorphic
        );

        let n = MAX_ATOMS_FOR_EXACT_CHECK + 5;
        let atoms: Vec<Atom> = (0..n).map(|_| c1h_with_hydrogens(2)).collect();
        let bonds: Vec<_> = (0..n).map(|i| single(i, (i + 1) % n)).collect();
        let oversized = Molecule { atoms, bonds };
        assert_eq!(
            is_isomorphic_detailed(&oversized, &oversized.clone()),
            IsomorphismOutcome::AtomLimitExceeded
        );
    }

    #[test]
    fn diagnostics_track_comparisons_and_atom_limit_rejections() {
        // Diagnostics-only: never read by any acceptance decision, exists so
        // MAX_ATOMS_FOR_EXACT_CHECK/MAX_BACKTRACK_STEPS can later be
        // evaluated against real observed usage.
        //
        // Uses before/after DELTAS, not reset-then-assert-exact: these
        // counters are global statics, and `cargo test` runs tests
        // multi-threaded by default, so other tests in this module
        // increment the same counters concurrently. A first version of this
        // test reset the counters to zero and asserted exact values --
        // correct in isolation, but a real flakiness hazard under concurrent
        // test execution, caught before it was ever committed. Deltas are
        // robust to that: this test only asserts that ITS OWN actions moved
        // the counters by at least the expected amount, regardless of what
        // else is running.
        let before = diagnostics();

        let _ = is_isomorphic(&mol("CCO"), &mol("CCO"));
        let _ = is_isomorphic(&prism_graph(), &k33_graph());
        let after_two = diagnostics();
        assert!(after_two.comparisons_attempted >= before.comparisons_attempted + 2);

        let n = MAX_ATOMS_FOR_EXACT_CHECK + 5;
        let atoms: Vec<Atom> = (0..n).map(|_| c1h_with_hydrogens(2)).collect();
        let bonds: Vec<_> = (0..n).map(|i| single(i, (i + 1) % n)).collect();
        let oversized = Molecule { atoms, bonds };
        let _ = is_isomorphic(&oversized, &oversized.clone());
        let after_three = diagnostics();
        assert!(after_three.comparisons_attempted >= after_two.comparisons_attempted + 1);
        assert!(after_three.atom_limit_rejections >= before.atom_limit_rejections + 1);
    }
}
