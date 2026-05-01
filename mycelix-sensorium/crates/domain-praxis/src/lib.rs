// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Praxis domain module — decentralized learning with consciousness-aware adaptivity.

pub mod types;

use sensorium_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct PraxisDomain;

impl DomainModule for PraxisDomain {
    fn id(&self) -> &'static str { "praxis" }
    fn name(&self) -> &'static str { "Education" }
    fn bio_name(&self) -> &'static str { "Growth" }
    fn description(&self) -> &'static str {
        "K-to-PhD adaptive learning: 2,002 curriculum nodes, BKT mastery tracking, spaced repetition, and consciousness-gated credential issuance."
    }

    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#2563EB", glow: "#60A5FA" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Dashboard", bio_label: "Growth Mirror", path: "/praxis/dashboard" },
            NavItem { label: "Courses", bio_label: "Pathways", path: "/praxis/courses" },
            NavItem { label: "Review", bio_label: "Memory Garden", path: "/praxis/review" },
            NavItem { label: "Skill Map", bio_label: "Constellation", path: "/praxis/skill-map" },
            NavItem { label: "Governance", bio_label: "Student Voice", path: "/praxis/governance" },
            NavItem { label: "Credentials", bio_label: "Proof of Growth", path: "/praxis/credentials" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Participant }
    fn key_context(&self) -> &'static [u8] { b"mycelix-praxis-v1" }
    fn happ_role(&self) -> &'static str { "praxis" }

    fn zomes(&self) -> &'static [&'static str] {
        &["learning", "fl", "credential", "dao", "srs", "gamification", "adaptive", "integration"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Credential issuance linked to DID", required: true },
            ClusterDependency { cluster_id: "governance", reason: "Curriculum DAO voting", required: false },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Learning Record", zome: "learning", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Credential", zome: "credential", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Mastery Score", zome: "adaptive", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Review Card", zome: "srs", sensitivity: DataSensitivity::Private },
        ]
    }
}
