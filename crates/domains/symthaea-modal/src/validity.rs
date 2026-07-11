// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modal systems (K/T/S4/S5) and bounded validity / countermodel search.
//!
//! Validity is checked by exhaustively searching all frames (up to a small world
//! bound) whose accessibility relation satisfies the system's constraints, over
//! all valuations of the formula's variables, for a world where the formula
//! fails. Finding a countermodel is a *sound* disproof of validity; K/T/S4/S5 all
//! have the finite-model property, so for the small formulas here the bounded
//! search also confirms validity (no small countermodel ⇒ valid).

use crate::kripke::{Formula, KripkeModel};
use std::collections::BTreeSet;

/// A normal modal system, defined by its frame conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum System {
    /// K — any accessibility relation.
    K,
    /// T — reflexive.
    T,
    /// S4 — reflexive + transitive.
    S4,
    /// S5 — reflexive + symmetric + transitive (an equivalence relation).
    S5,
}

fn to_matrix(worlds: usize, relation: &[(usize, usize)]) -> Vec<Vec<bool>> {
    let mut m = vec![vec![false; worlds]; worlds];
    for &(i, j) in relation {
        if i < worlds && j < worlds {
            m[i][j] = true;
        }
    }
    m
}

fn frame_ok(worlds: usize, m: &[Vec<bool>], system: System) -> bool {
    let reflexive = || (0..worlds).all(|i| m[i][i]);
    let symmetric = || (0..worlds).all(|i| (0..worlds).all(|j| !m[i][j] || m[j][i]));
    let transitive = || {
        (0..worlds)
            .all(|i| (0..worlds).all(|j| (0..worlds).all(|k| !(m[i][j] && m[j][k]) || m[i][k])))
    };
    match system {
        System::K => true,
        System::T => reflexive(),
        System::S4 => reflexive() && transitive(),
        System::S5 => reflexive() && symmetric() && transitive(),
    }
}

impl std::str::FromStr for System {
    type Err = String;

    /// Case-insensitive: `k`, `t`, `s4`, `s5`.
    fn from_str(s: &str) -> Result<System, String> {
        match s.trim().to_lowercase().as_str() {
            "k" => Ok(System::K),
            "t" => Ok(System::T),
            "s4" => Ok(System::S4),
            "s5" => Ok(System::S5),
            other => Err(format!(
                "unknown modal system '{other}' (expected K, T, S4, or S5)"
            )),
        }
    }
}

/// Search for a countermodel to `formula` in `system` using at most `max_worlds`
/// worlds. `Some(model)` disproves validity; `None` means no small counterexample.
pub fn find_countermodel(
    formula: &Formula,
    system: System,
    max_worlds: usize,
) -> Option<KripkeModel> {
    let vars = formula.variables();
    let nv = vars.len();

    for w in 1..=max_worlds {
        let pairs: Vec<(usize, usize)> = (0..w).flat_map(|i| (0..w).map(move |j| (i, j))).collect();
        let n_pairs = pairs.len();

        for rmask in 0u32..(1u32 << n_pairs) {
            let relation: Vec<(usize, usize)> = pairs
                .iter()
                .enumerate()
                .filter(|(k, _)| rmask & (1 << k) != 0)
                .map(|(_, &p)| p)
                .collect();

            let matrix = to_matrix(w, &relation);
            if !frame_ok(w, &matrix, system) {
                continue;
            }

            let val_bits = w * nv;
            for vmask in 0u64..(1u64 << val_bits) {
                let mut true_atoms = BTreeSet::new();
                for world in 0..w {
                    for (vi, name) in vars.iter().enumerate() {
                        if vmask & (1u64 << (world * nv + vi)) != 0 {
                            true_atoms.insert((world, name.clone()));
                        }
                    }
                }
                let model = KripkeModel {
                    worlds: w,
                    relation: relation.clone(),
                    true_atoms,
                };
                if (0..w).any(|world| !model.satisfies(world, formula)) {
                    return Some(model);
                }
            }
        }
    }
    None
}

/// Whether `formula` is valid in `system` (no countermodel up to `max_worlds`).
pub fn is_valid_bounded(formula: &Formula, system: System, max_worlds: usize) -> bool {
    find_countermodel(formula, system, max_worlds).is_none()
}

/// Validity with a default bound of 3 worlds (sufficient for the standard axioms).
pub fn is_valid(formula: &Formula, system: System) -> bool {
    is_valid_bounded(formula, system, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kripke::{implies, necessarily, possibly, var};

    fn axiom_t() -> Formula {
        // □p → p
        implies(necessarily(var("p")), var("p"))
    }
    fn axiom_4() -> Formula {
        // □p → □□p
        implies(necessarily(var("p")), necessarily(necessarily(var("p"))))
    }
    fn axiom_5() -> Formula {
        // ◇p → □◇p
        implies(possibly(var("p")), necessarily(possibly(var("p"))))
    }
    fn axiom_k() -> Formula {
        // □(p→q) → (□p → □q)
        implies(
            necessarily(implies(var("p"), var("q"))),
            implies(necessarily(var("p")), necessarily(var("q"))),
        )
    }

    #[test]
    fn k_axiom_valid_everywhere() {
        assert!(is_valid(&axiom_k(), System::K));
    }

    #[test]
    fn t_axiom_separates_k_from_t() {
        // □p → p fails in K but holds in T (and above).
        assert!(!is_valid(&axiom_t(), System::K));
        assert!(is_valid(&axiom_t(), System::T));
        assert!(is_valid(&axiom_t(), System::S4));
    }

    #[test]
    fn four_axiom_separates_t_from_s4() {
        // □p → □□p fails in T (non-transitive) but holds in S4.
        assert!(!is_valid(&axiom_4(), System::T));
        assert!(is_valid(&axiom_4(), System::S4));
    }

    #[test]
    fn five_axiom_separates_s4_from_s5() {
        // ◇p → □◇p fails in S4 but holds in S5.
        assert!(!is_valid(&axiom_5(), System::S4));
        assert!(is_valid(&axiom_5(), System::S5));
    }

    #[test]
    fn countermodel_is_a_real_witness() {
        // The disproof of □p→p in K must actually falsify it at some world.
        let cm = find_countermodel(&axiom_t(), System::K, 3).unwrap();
        assert!((0..cm.worlds).any(|w| !cm.satisfies(w, &axiom_t())));
    }
}
