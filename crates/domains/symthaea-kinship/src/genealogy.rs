// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Genealogy graph and kin-relation resolution.

use crate::terms::{in_law, term_from_mn};
use std::collections::{HashMap, VecDeque};

/// Biological sex, used to choose sexed kin terms (father vs mother, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sex {
    Male,
    Female,
    Unknown,
}

/// A genealogy: people with sex, parent→child edges, and marriages.
#[derive(Debug, Clone, Default)]
pub struct Genealogy {
    sex: HashMap<String, Sex>,
    /// child → its parents.
    parents: HashMap<String, Vec<String>>,
    /// person → spouses.
    spouses: HashMap<String, Vec<String>>,
}

impl Genealogy {
    pub fn new() -> Genealogy {
        Genealogy::default()
    }

    /// Register a person with a sex (idempotent).
    pub fn person(&mut self, id: &str, sex: Sex) -> &mut Genealogy {
        self.sex.insert(id.to_string(), sex);
        self
    }

    /// Record that `parent` is a parent of `child` (both auto-registered).
    pub fn parent_of(&mut self, parent: &str, child: &str) -> &mut Genealogy {
        self.sex.entry(parent.to_string()).or_insert(Sex::Unknown);
        self.sex.entry(child.to_string()).or_insert(Sex::Unknown);
        let ps = self.parents.entry(child.to_string()).or_default();
        if !ps.iter().any(|p| p == parent) {
            ps.push(parent.to_string());
        }
        self
    }

    /// Record a marriage (symmetric).
    pub fn marriage(&mut self, a: &str, b: &str) -> &mut Genealogy {
        self.spouses
            .entry(a.to_string())
            .or_default()
            .push(b.to_string());
        self.spouses
            .entry(b.to_string())
            .or_default()
            .push(a.to_string());
        self
    }

    fn sex_of(&self, id: &str) -> Sex {
        self.sex.get(id).copied().unwrap_or(Sex::Unknown)
    }

    /// All ancestors of `start` (including `start` at distance 0) → generations.
    fn ancestors(&self, start: &str) -> HashMap<String, u32> {
        let mut dist = HashMap::new();
        let mut queue = VecDeque::new();
        dist.insert(start.to_string(), 0u32);
        queue.push_back(start.to_string());
        while let Some(p) = queue.pop_front() {
            let d = dist[&p];
            if let Some(parents) = self.parents.get(&p) {
                for parent in parents {
                    if !dist.contains_key(parent) {
                        dist.insert(parent.clone(), d + 1);
                        queue.push_back(parent.clone());
                    }
                }
            }
        }
        dist
    }

    /// Triangle coordinates `(m, n)` for the nearest common ancestor of ego and
    /// alter, or `None` if they share no ancestor (unrelated by blood).
    fn triangle(&self, ego: &str, alter: &str) -> Option<(u32, u32)> {
        let ae = self.ancestors(ego);
        let aa = self.ancestors(alter);
        let mut best: Option<(u32, u32, u32)> = None; // (total, m, n)
        for (anc, &m) in &ae {
            if let Some(&n) = aa.get(anc) {
                let total = m + n;
                if best.is_none_or(|(bt, _, _)| total < bt) {
                    best = Some((total, m, n));
                }
            }
        }
        best.map(|(_, m, n)| (m, n))
    }

    /// The consanguineal (blood) kin term for `alter` from `ego`'s view.
    fn blood_term(&self, ego: &str, alter: &str) -> Option<String> {
        let (m, n) = self.triangle(ego, alter)?;
        Some(term_from_mn(m, n, self.sex_of(alter)))
    }

    /// The kin term for `alter` from `ego`'s perspective — consanguineal first,
    /// then spouse, then in-laws. `None` if unrelated.
    pub fn relation(&self, ego: &str, alter: &str) -> Option<String> {
        // 1. Blood relation.
        if let Some(t) = self.blood_term(ego, alter) {
            return Some(t);
        }
        // 2. Direct spouse.
        if self
            .spouses
            .get(ego)
            .is_some_and(|s| s.iter().any(|x| x == alter))
        {
            return Some(
                match self.sex_of(alter) {
                    Sex::Male => "husband",
                    Sex::Female => "wife",
                    Sex::Unknown => "spouse",
                }
                .to_string(),
            );
        }
        // 3. Alter is a blood relative of ego's spouse → in-law (alter's sex is
        //    already used by blood_term, which is correct here).
        if let Some(spouses) = self.spouses.get(ego) {
            for sp in spouses {
                if let Some(t) = self.blood_term(sp, alter) {
                    return Some(in_law(&t));
                }
            }
        }
        // 4. Alter is the spouse of ego's blood relative → in-law, but the term
        //    must use ALTER's sex, not the blood relative's.
        if let Some(alter_spouses) = self.spouses.get(alter) {
            for asp in alter_spouses {
                if let Some((m, n)) = self.triangle(ego, asp) {
                    let base = term_from_mn(m, n, self.sex_of(alter));
                    return Some(in_law(&base));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three-generation family with a cousin branch and affines.
    fn family() -> Genealogy {
        let mut g = Genealogy::new();
        g.person("grandpa", Sex::Male)
            .person("grandma", Sex::Female)
            .marriage("grandpa", "grandma");
        // grandpa+grandma → dad, uncle
        for child in ["dad", "uncle"] {
            g.parent_of("grandpa", child).parent_of("grandma", child);
        }
        g.person("dad", Sex::Male).person("uncle", Sex::Male);
        g.person("mom", Sex::Female).marriage("dad", "mom");
        g.person("aunt", Sex::Female).marriage("uncle", "aunt");
        // dad+mom → ego, sis
        for child in ["ego", "sis"] {
            g.parent_of("dad", child).parent_of("mom", child);
        }
        g.person("ego", Sex::Male).person("sis", Sex::Female);
        // uncle+aunt → cousin
        g.parent_of("uncle", "cousin").parent_of("aunt", "cousin");
        g.person("cousin", Sex::Male);
        // ego+wife → son
        g.person("wife", Sex::Female).marriage("ego", "wife");
        g.parent_of("ego", "son").parent_of("wife", "son");
        g.person("son", Sex::Male);
        // affines: wife's father & brother; sis's husband; son's wife
        g.person("wife_dad", Sex::Male)
            .parent_of("wife_dad", "wife");
        g.person("wife_bro", Sex::Male)
            .parent_of("wife_dad", "wife_bro");
        g.person("sis_husband", Sex::Male)
            .marriage("sis", "sis_husband");
        g.person("son_wife", Sex::Female)
            .marriage("son", "son_wife");
        g
    }

    fn rel(g: &Genealogy, alter: &str) -> String {
        g.relation("ego", alter)
            .unwrap_or_else(|| "unrelated".into())
    }

    #[test]
    fn lineal_relations() {
        let g = family();
        assert_eq!(rel(&g, "dad"), "father");
        assert_eq!(rel(&g, "mom"), "mother");
        assert_eq!(rel(&g, "grandpa"), "grandfather");
        assert_eq!(rel(&g, "grandma"), "grandmother");
        assert_eq!(rel(&g, "son"), "son");
        assert_eq!(rel(&g, "ego"), "self");
    }

    #[test]
    fn collateral_relations() {
        let g = family();
        assert_eq!(rel(&g, "sis"), "sister");
        assert_eq!(rel(&g, "uncle"), "uncle");
        assert_eq!(rel(&g, "cousin"), "first cousin");
    }

    #[test]
    fn nephew_view_is_symmetric_in_structure() {
        // From the cousin's uncle (ego is dad's child; cousin's father is uncle):
        // ego is the cousin's "first cousin" too, and ego→uncle's child cousin.
        let g = family();
        assert_eq!(g.relation("cousin", "ego").unwrap(), "first cousin");
        // Ego is a nephew of uncle; uncle sees ego as nephew.
        assert_eq!(g.relation("uncle", "ego").unwrap(), "nephew");
    }

    #[test]
    fn affinal_relations() {
        let g = family();
        assert_eq!(rel(&g, "wife"), "wife");
        assert_eq!(rel(&g, "wife_dad"), "father-in-law");
        assert_eq!(rel(&g, "wife_bro"), "brother-in-law");
        assert_eq!(rel(&g, "sis_husband"), "brother-in-law"); // sister's husband
        assert_eq!(rel(&g, "son_wife"), "daughter-in-law"); // son's wife
    }

    #[test]
    fn unrelated_is_none() {
        let g = family();
        let mut g2 = g.clone();
        g2.person("stranger", Sex::Male);
        assert_eq!(g2.relation("ego", "stranger"), None);
    }
}
