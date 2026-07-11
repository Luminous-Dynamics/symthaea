// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Propositional modal formulas and Kripke-model evaluation.

use std::collections::BTreeSet;

/// A propositional modal formula. `Box` = necessity (□), `Diamond` = possibility (◇).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula {
    Var(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Nec(Box<Formula>),
    Pos(Box<Formula>),
}

// Ergonomic constructors.
pub fn var(name: &str) -> Formula {
    Formula::Var(name.to_string())
}
pub fn not(f: Formula) -> Formula {
    Formula::Not(Box::new(f))
}
pub fn and(a: Formula, b: Formula) -> Formula {
    Formula::And(Box::new(a), Box::new(b))
}
pub fn or(a: Formula, b: Formula) -> Formula {
    Formula::Or(Box::new(a), Box::new(b))
}
pub fn implies(a: Formula, b: Formula) -> Formula {
    Formula::Implies(Box::new(a), Box::new(b))
}
/// Necessity (□).
pub fn necessarily(f: Formula) -> Formula {
    Formula::Nec(Box::new(f))
}
/// Possibility (◇).
pub fn possibly(f: Formula) -> Formula {
    Formula::Pos(Box::new(f))
}

impl Formula {
    /// Propositional variables occurring in the formula (sorted, unique).
    pub fn variables(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        self.collect_vars(&mut set);
        set.into_iter().collect()
    }

    fn collect_vars(&self, set: &mut BTreeSet<String>) {
        match self {
            Formula::Var(x) => {
                set.insert(x.clone());
            }
            Formula::Not(a) | Formula::Nec(a) | Formula::Pos(a) => a.collect_vars(set),
            Formula::And(a, b) | Formula::Or(a, b) | Formula::Implies(a, b) => {
                a.collect_vars(set);
                b.collect_vars(set);
            }
        }
    }
}

/// A Kripke model: `worlds` numbered `0..worlds`, an accessibility relation, and
/// the set of true atoms `(world, var)`.
#[derive(Debug, Clone)]
pub struct KripkeModel {
    pub worlds: usize,
    pub relation: Vec<(usize, usize)>,
    pub true_atoms: BTreeSet<(usize, String)>,
}

impl KripkeModel {
    pub fn new(worlds: usize) -> KripkeModel {
        KripkeModel {
            worlds,
            relation: Vec::new(),
            true_atoms: BTreeSet::new(),
        }
    }

    pub fn accessible(mut self, from: usize, to: usize) -> KripkeModel {
        self.relation.push((from, to));
        self
    }

    pub fn set_true(mut self, world: usize, var: &str) -> KripkeModel {
        self.true_atoms.insert((world, var.to_string()));
        self
    }

    fn successors(&self, world: usize) -> Vec<usize> {
        self.relation
            .iter()
            .filter(|(a, _)| *a == world)
            .map(|(_, b)| *b)
            .collect()
    }

    /// Whether `formula` holds at `world` in this model.
    pub fn satisfies(&self, world: usize, formula: &Formula) -> bool {
        match formula {
            Formula::Var(x) => self.true_atoms.contains(&(world, x.clone())),
            Formula::Not(a) => !self.satisfies(world, a),
            Formula::And(a, b) => self.satisfies(world, a) && self.satisfies(world, b),
            Formula::Or(a, b) => self.satisfies(world, a) || self.satisfies(world, b),
            Formula::Implies(a, b) => !self.satisfies(world, a) || self.satisfies(world, b),
            Formula::Nec(a) => self.successors(world).iter().all(|&v| self.satisfies(v, a)),
            Formula::Pos(a) => self.successors(world).iter().any(|&v| self.satisfies(v, a)),
        }
    }

    /// Whether the formula holds at *every* world of the model.
    pub fn valid_in_model(&self, formula: &Formula) -> bool {
        (0..self.worlds).all(|w| self.satisfies(w, formula))
    }

    /// A compact human-readable rendering of the model — useful for reporting a
    /// countermodel back to a user ("here is *why* it fails"). Lists the worlds,
    /// the accessibility edges, and which atoms are true in which world.
    pub fn describe(&self) -> String {
        let edges: Vec<String> = self
            .relation
            .iter()
            .map(|(a, b)| format!("w{a}→w{b}"))
            .collect();
        let edge_s = if edges.is_empty() {
            "no accessibility edges".to_string()
        } else {
            edges.join(", ")
        };
        let atoms: Vec<String> = self
            .true_atoms
            .iter()
            .map(|(w, v)| format!("{v}@w{w}"))
            .collect();
        let atom_s = if atoms.is_empty() {
            "all atoms false".to_string()
        } else {
            atoms.join(", ")
        };
        format!("{} world(s); {edge_s}; true: {atom_s}", self.worlds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_renders_worlds_edges_and_atoms() {
        let m = KripkeModel::new(2).accessible(0, 1).set_true(1, "p");
        let d = m.describe();
        assert!(d.contains("2 world(s)"), "{d}");
        assert!(d.contains("w0→w1"), "{d}");
        assert!(d.contains("p@w1"), "{d}");
    }

    #[test]
    fn describe_handles_empty_frame() {
        let d = KripkeModel::new(1).describe();
        assert!(d.contains("no accessibility edges"), "{d}");
        assert!(d.contains("all atoms false"), "{d}");
    }

    #[test]
    fn box_is_vacuously_true_at_a_dead_end() {
        // A world with no successors satisfies □anything, but not necessarily p.
        let m = KripkeModel::new(1); // world 0, no edges, p false
        assert!(m.satisfies(0, &necessarily(var("p"))));
        assert!(!m.satisfies(0, &var("p")));
    }

    #[test]
    fn diamond_needs_a_witness() {
        let m = KripkeModel::new(2).accessible(0, 1).set_true(1, "p");
        assert!(m.satisfies(0, &possibly(var("p")))); // world 1 witnesses ◇p
        let m2 = KripkeModel::new(2).accessible(0, 1); // p false at 1
        assert!(!m2.satisfies(0, &possibly(var("p"))));
    }

    #[test]
    fn box_quantifies_over_all_successors() {
        // □p false if any successor lacks p.
        let m = KripkeModel::new(3)
            .accessible(0, 1)
            .accessible(0, 2)
            .set_true(1, "p"); // world 2 lacks p
        assert!(!m.satisfies(0, &necessarily(var("p"))));
    }
}
