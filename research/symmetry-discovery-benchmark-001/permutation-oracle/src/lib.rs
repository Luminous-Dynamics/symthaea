// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact exhaustive oracle for the S3 permutation action on R^3.
//!
//! This is a complementary benchmark lane to the D4 planar point-group oracle.
//! It proves no continuous-group or physical claim.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Permutation3(pub [usize; 3]);

impl Permutation3 {
    pub fn apply(self, point: [i64; 3]) -> [i64; 3] {
        [
            point[self.0[0]],
            point[self.0[1]],
            point[self.0[2]],
        ]
    }

    /// Composition for the action apply(p, x)[i] = x[p[i]].
    /// The resulting action is self.apply(rhs.apply(x)).
    pub fn compose(self, rhs: Self) -> Self {
        Self([
            rhs.0[self.0[0]],
            rhs.0[self.0[1]],
            rhs.0[self.0[2]],
        ])
    }

    pub fn inverse(self) -> Self {
        let mut inverse = [0; 3];
        for i in 0..3 {
            inverse[self.0[i]] = i;
        }
        Self(inverse)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Element {
    pub id: String,
    pub permutation: [usize; 3],
}

impl Element {
    fn action(&self) -> Permutation3 {
        Permutation3(self.permutation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Product {
    pub left: String,
    pub right: String,
    pub product: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub case_id: String,
    pub split: String,
    pub target_subgroup_ids: Vec<String>,
    pub expected_subgroup_label: String,
    pub point_orbit: Vec<[i64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: String,
    pub benchmark_id: String,
    pub action: Action,
    pub seed_point: [i64; 3],
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    pub group_id: String,
    pub action_kind: String,
    pub identity_element_id: String,
    pub elements: Vec<Element>,
    pub multiplication_table: Vec<Product>,
    pub inverse_table: Vec<Inverse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inverse {
    pub element: String,
    pub inverse: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Witness {
    OrbitPermutation { source_to_target: Vec<usize> },
    MovedOrbitPoint { source: [i64; 3], transformed: [i64; 3] },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub element_id: String,
    pub proposed: bool,
    pub exact: bool,
    pub witness: Witness,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Audit {
    pub case_id: String,
    pub proposed_subgroup_ids: Vec<String>,
    pub exact_stabilizer_ids: Vec<String>,
    pub group_axioms_satisfied: bool,
    pub axiom_violations: Vec<String>,
    pub false_promotion_elements: Vec<String>,
    pub missed_symmetry_elements: Vec<String>,
    pub decisions: Vec<Decision>,
}

#[derive(Debug, Clone)]
pub struct Oracle {
    pub manifest: Manifest,
    by_id: BTreeMap<String, Permutation3>,
    multiplication: BTreeMap<(String, String), String>,
    inverses: BTreeMap<String, String>,
}

fn factorial_permutations() -> Vec<Permutation3> {
    vec![
        Permutation3([0, 1, 2]),
        Permutation3([0, 2, 1]),
        Permutation3([1, 0, 2]),
        Permutation3([1, 2, 0]),
        Permutation3([2, 0, 1]),
        Permutation3([2, 1, 0]),
    ]
}

impl Oracle {
    pub fn from_manifest(manifest: Manifest) -> Result<Self, String> {
        if manifest.schema_version != "0.1.0"
            || manifest.benchmark_id != "LUMINOUS-SYMM-PERMUTATION-001"
            || manifest.action.group_id != "S3"
            || manifest.action.action_kind != "coordinate_permutation_on_R3"
        {
            return Err("unsupported S3 permutation manifest identity".to_owned());
        }
        let mut by_id = BTreeMap::new();
        let mut by_perm = BTreeMap::new();
        for element in &manifest.action.elements {
            let permutation = element.action();
            if !factorial_permutations().contains(&permutation) {
                return Err(format!("invalid permutation {}", element.id));
            }
            if by_id.insert(element.id.clone(), permutation).is_some()
                || by_perm.insert(permutation, element.id.clone()).is_some()
            {
                return Err("duplicate permutation ID or action".to_owned());
            }
        }
        if by_id.len() != 6 {
            return Err("S3 must contain exactly six elements".to_owned());
        }
        let identity = Permutation3([0, 1, 2]);
        if by_id.get(&manifest.action.identity_element_id) != Some(&identity) {
            return Err("declared identity is not identity permutation".to_owned());
        }

        let mut multiplication = BTreeMap::new();
        for product in &manifest.action.multiplication_table {
            if multiplication
                .insert((product.left.clone(), product.right.clone()), product.product.clone())
                .is_some()
            {
                return Err("duplicate S3 multiplication entry".to_owned());
            }
        }
        if multiplication.len() != 36 {
            return Err("S3 multiplication table must contain 36 products".to_owned());
        }

        let mut inverses = BTreeMap::new();
        for inverse in &manifest.action.inverse_table {
            if inverses.insert(inverse.element.clone(), inverse.inverse.clone()).is_some() {
                return Err("duplicate S3 inverse entry".to_owned());
            }
        }
        if inverses.len() != 6 {
            return Err("S3 inverse table must contain six entries".to_owned());
        }

        let oracle = Self { manifest, by_id, multiplication, inverses };
        oracle.verify_group_axioms()?;
        oracle.verify_all_cases()?;
        Ok(oracle)
    }

    pub fn verify_group_axioms(&self) -> Result<(), String> {
        let ids: Vec<String> = self.by_id.keys().cloned().collect();
        for left in &ids {
            for right in &ids {
                let computed = self.by_id[left].compose(self.by_id[right]);
                let computed_id = self.by_id
                    .iter()
                    .find(|(_, permutation)| **permutation == computed)
                    .map(|(id, _)| id.clone())
                    .ok_or_else(|| format!("S3 is not closed at {left} * {right}"))?;
                let declared = self.multiplication
                    .get(&(left.clone(), right.clone()))
                    .ok_or_else(|| format!("missing S3 product {left} * {right}"))?;
                if *declared != computed_id {
                    return Err(format!("wrong S3 product entry {left} * {right}"));
                }
            }
            let inverse_id = self.inverses.get(left)
                .ok_or_else(|| format!("missing S3 inverse for {left}"))?;
            let inverse = self.by_id.get(inverse_id)
                .ok_or_else(|| format!("unknown inverse {inverse_id}"))?;
            if self.by_id[left].compose(*inverse) != Permutation3([0, 1, 2])
                || inverse.compose(self.by_id[left]) != Permutation3([0, 1, 2])
            {
                return Err(format!("two-sided inverse failed for {left}"));
            }
            if self.by_id[left].inverse() != *inverse {
                return Err(format!("inverse algorithm disagrees for {left}"));
            }
        }
        for a in &ids {
            for b in &ids {
                for c in &ids {
                    let ab = self.multiplication[&(a.clone(), b.clone())].clone();
                    let bc = self.multiplication[&(b.clone(), c.clone())].clone();
                    let lhs = self.multiplication[&(ab, c.clone())].clone();
                    let rhs = self.multiplication[&(a.clone(), bc)].clone();
                    if lhs != rhs {
                        return Err(format!("S3 associativity failed for {a}, {b}, {c}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn subgroup_valid(&self, members: &BTreeSet<String>) -> Vec<String> {
        let mut errors = Vec::new();
        if !members.contains(&self.manifest.action.identity_element_id) {
            errors.push("missing_identity".to_owned());
        }
        for left in members {
            if !self.inverses.get(left).is_some_and(|inverse| members.contains(inverse)) {
                errors.push(format!("missing_inverse:{left}"));
            }
            for right in members {
                if !self.multiplication
                    .get(&(left.clone(), right.clone()))
                    .is_some_and(|product| members.contains(product))
                {
                    errors.push(format!("missing_product:{left}:{right}"));
                }
            }
        }
        errors
    }

    pub fn enumerate_subgroups(&self) -> Vec<BTreeSet<String>> {
        let ids: Vec<String> = self.by_id.keys().cloned().collect();
        let mut groups = Vec::new();
        for mask in 0_u32..(1_u32 << ids.len()) {
            let members: BTreeSet<String> = ids.iter().enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, id)| id.clone())
                .collect();
            if self.subgroup_valid(&members).is_empty() {
                groups.push(members);
            }
        }
        groups
    }

    pub fn orbit(&self, members: &BTreeSet<String>) -> BTreeSet<[i64; 3]> {
        members.iter()
            .map(|id| self.by_id[id].apply(self.manifest.seed_point))
            .collect()
    }

    pub fn stabilizer(&self, orbit: &BTreeSet<[i64; 3]>) -> BTreeSet<String> {
        self.by_id.iter()
            .filter_map(|(id, permutation)| {
                let transformed: BTreeSet<[i64; 3]> =
                    orbit.iter().map(|point| permutation.apply(*point)).collect();
                (transformed == *orbit).then_some(id.clone())
            })
            .collect()
    }

    pub fn verify_all_cases(&self) -> Result<(), String> {
        let subgroups = self.enumerate_subgroups();
        if subgroups.len() != 6 {
            return Err(format!("expected six S3 subgroups, got {}", subgroups.len()));
        }
        let mut case_ids = BTreeSet::new();
        let mut observed = BTreeSet::new();
        for case in &self.manifest.cases {
            if !case_ids.insert(case.case_id.clone()) {
                return Err(format!("duplicate S3 case ID {}", case.case_id));
            }
            let declared: BTreeSet<String> = case.target_subgroup_ids.iter().cloned().collect();
            if declared.len() != case.target_subgroup_ids.len() || !subgroups.contains(&declared) {
                return Err(format!("{} declares a non-subgroup target", case.case_id));
            }
            let expected_label = match declared.len() {
                1 => "C1".to_owned(),
                3 => "C3".to_owned(),
                6 => "S3".to_owned(),
                2 => format!(
                    "C2_{}",
                    declared
                        .iter()
                        .find(|element_id| *element_id != &self.manifest.action.identity_element_id)
                        .ok_or_else(|| format!("{} order-two subgroup lacks generator", case.case_id))?
                ),
                _ => return Err(format!("{} has unsupported S3 subgroup order", case.case_id)),
            };
            if case.expected_subgroup_label != expected_label {
                return Err(format!(
                    "{} label {} does not match exact subgroup {}",
                    case.case_id, case.expected_subgroup_label, expected_label
                ));
            }
            if case.split != "fixture_sanity" {
                return Err(format!("{} must remain isolated as fixture_sanity", case.case_id));
            }
            let orbit = self.orbit(&declared);
            let fixture_orbit: BTreeSet<[i64; 3]> = case.point_orbit.iter().copied().collect();
            if orbit != fixture_orbit || fixture_orbit.len() != case.point_orbit.len() {
                return Err(format!("{} point orbit disagrees with exact action", case.case_id));
            }
            let exact = self.stabilizer(&orbit);
            if exact != declared {
                return Err(format!("{} stabilizer is not its declared subgroup", case.case_id));
            }
            if !observed.insert(declared) {
                return Err(format!("duplicate subgroup fixture in {}", case.case_id));
            }
        }
        if observed != subgroups {
            return Err("S3 cases do not cover every distinct subgroup".to_owned());
        }
        Ok(())
    }

    pub fn audit_proposal(&self, case_id: &str, proposal_ids: &[String]) -> Result<Audit, String> {
        let case = self.manifest.cases.iter()
            .find(|case| case.case_id == case_id)
            .ok_or_else(|| format!("unknown S3 case {case_id}"))?;
        let mut proposed = BTreeSet::new();
        for id in proposal_ids {
            if !self.by_id.contains_key(id) {
                return Err(format!("unknown S3 proposal element {id}"));
            }
            if !proposed.insert(id.clone()) {
                return Err(format!("duplicate S3 proposal element {id}"));
            }
        }
        let exact: BTreeSet<String> = case.target_subgroup_ids.iter().cloned().collect();
        let axiom_violations = self.subgroup_valid(&proposed);
        let orbit: BTreeSet<[i64; 3]> = case.point_orbit.iter().copied().collect();
        let mut decisions = Vec::new();
        let mut false_promotion_elements = Vec::new();
        let mut missed_symmetry_elements = Vec::new();

        for (id, permutation) in &self.by_id {
            let transformed: BTreeSet<[i64; 3]> =
                orbit.iter().map(|point| permutation.apply(*point)).collect();
            let accepted = transformed == orbit;
            let is_proposed = proposed.contains(id);
            let witness = if accepted {
                let source_to_target = orbit.iter().map(|point| {
                    let image = permutation.apply(*point);
                    orbit.iter().position(|candidate| candidate == &image).unwrap()
                }).collect();
                Witness::OrbitPermutation { source_to_target }
            } else {
                let (source, transformed) = orbit.iter()
                    .map(|point| (*point, permutation.apply(*point)))
                    .find(|(_, image)| !orbit.contains(image))
                    .ok_or_else(|| "rejected action lacks moved-point witness".to_owned())?;
                Witness::MovedOrbitPoint { source, transformed }
            };
            if is_proposed && !accepted { false_promotion_elements.push(id.clone()); }
            if !is_proposed && accepted { missed_symmetry_elements.push(id.clone()); }
            decisions.push(Decision { element_id:id.clone(), proposed:is_proposed, exact:accepted, witness });
        }
        Ok(Audit {
            case_id:case_id.to_owned(),
            proposed_subgroup_ids:proposed.into_iter().collect(),
            exact_stabilizer_ids:exact.into_iter().collect(),
            group_axioms_satisfied:axiom_violations.is_empty(),
            axiom_violations,
            false_promotion_elements,
            missed_symmetry_elements,
            decisions
        })
    }

    pub fn verify_audit(&self, case_id: &str, audit: &Audit) -> Result<(), String> {
        let replayed = self.audit_proposal(case_id, &audit.proposed_subgroup_ids)?;
        if replayed == *audit { Ok(()) } else { Err("S3 audit replay mismatch".to_owned()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle() -> Oracle {
        let manifest: Manifest = serde_json::from_str(include_str!("../../fixtures/s3_permutation_v0.1.json")).unwrap();
        Oracle::from_manifest(manifest).unwrap()
    }

    #[test]
    fn s3_group_table_has_six_elements_and_36_exact_products() {
        let oracle = oracle();
        assert_eq!(oracle.by_id.len(), 6);
        assert_eq!(oracle.multiplication.len(), 36);
        assert_eq!(oracle.inverses.len(), 6);
    }

    #[test]
    fn exhaustive_subgroup_lattice_contains_exactly_six_subgroups() {
        let oracle = oracle();
        let groups = oracle.enumerate_subgroups();
        assert_eq!(groups.len(), 6);
        let mut sizes = groups.iter().map(BTreeSet::len).collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 2, 2, 2, 3, 6]);
    }

    #[test]
    fn every_subgroup_orbit_has_exactly_that_stabilizer() {
        let oracle = oracle();
        for subgroup in oracle.enumerate_subgroups() {
            let orbit = oracle.orbit(&subgroup);
            assert_eq!(oracle.stabilizer(&orbit), subgroup);
        }
    }

    #[test]
    fn all_six_subgroup_labels_are_exact_and_nonduplicated() {
        let oracle = oracle();
        let labels = oracle
            .manifest
            .cases
            .iter()
            .map(|case| case.expected_subgroup_label.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            labels,
            BTreeSet::from([
                "C1".to_owned(),
                "C2_s12".to_owned(),
                "C2_s01".to_owned(),
                "C2_s02".to_owned(),
                "C3".to_owned(),
                "S3".to_owned(),
            ])
        );
    }

    #[test]
    fn c3_object_rejects_three_transposition_promotions() {
        let oracle = oracle();
        let audit = oracle.audit_proposal(
            "S3-C3",
            &vec!["e","s12","s01","c012","c021","s02"].into_iter().map(str::to_owned).collect::<Vec<_>>()
        ).unwrap();
        assert!(audit.group_axioms_satisfied);
        assert_eq!(audit.false_promotion_elements.len(), 3);
        assert!(audit.missed_symmetry_elements.is_empty());
        oracle.verify_audit("S3-C3",&audit).unwrap();
    }

    #[test]
    fn s3_object_detects_c2_underpromotion_and_invalid_proposal_sets() {
        let oracle = oracle();
        let audit = oracle.audit_proposal("S3-S3",&vec!["e","s12"].into_iter().map(str::to_owned).collect::<Vec<_>>()).unwrap();
        assert!(audit.group_axioms_satisfied);
        assert_eq!(audit.missed_symmetry_elements.len(),4);
        let invalid = oracle.audit_proposal("S3-C3",&vec!["e","c012"].into_iter().map(str::to_owned).collect::<Vec<_>>()).unwrap();
        assert!(!invalid.group_axioms_satisfied);
    }

    #[test]
    fn forged_s3_audit_witness_is_rejected() {
        let oracle = oracle();
        let mut audit = oracle.audit_proposal("S3-C3",&vec!["e","s12","s01","c012","c021","s02"].into_iter().map(str::to_owned).collect::<Vec<_>>()).unwrap();
        if let Witness::MovedOrbitPoint { transformed, .. } = &mut audit.decisions.iter_mut().find(|d|d.proposed&&!d.exact).unwrap().witness {
            transformed[0]+=1;
        }
        assert!(oracle.verify_audit("S3-C3",&audit).is_err());
    }
}
