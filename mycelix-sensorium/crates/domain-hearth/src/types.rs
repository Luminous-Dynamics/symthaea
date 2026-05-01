// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hearth shared types — mirrors mycelix-hearth zome entries.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KinshipBond {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub bond_type: BondType,
    pub strength: f64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BondType { Parent, Child, Sibling, Partner, Chosen, Mentor, Elder, Companion }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GratitudeNote {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub message: String,
    pub tags: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilyDecision {
    pub id: String,
    pub title: String,
    pub description: String,
    pub participants: Vec<String>,
    pub status: DecisionStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DecisionStatus { Proposed, Deliberating, Decided, Enacted, Withdrawn }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Story {
    pub id: String,
    pub author_did: String,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub visibility: Visibility,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Visibility { Private, Family, Community, Public }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Milestone {
    pub id: String,
    pub person_did: String,
    pub title: String,
    pub description: String,
    pub milestone_type: MilestoneType,
    pub date: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MilestoneType { Birth, FirstStep, FirstWord, School, Graduation, Marriage, Career, Retirement, Memorial, Custom }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kinship_bond_roundtrip() {
        let bond = KinshipBond { id: "kb-1".into(), from_did: "a".into(), to_did: "b".into(), bond_type: BondType::Partner, strength: 0.9, created_at: 0 };
        let bytes = rmp_serde::to_vec_named(&bond).unwrap();
        let decoded: KinshipBond = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.bond_type, BondType::Partner);
    }
}
