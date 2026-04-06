// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Praxis domain module — decentralized learning with consciousness-aware adaptivity.

pub mod types;

use portal_domain_trait::{ColorFamily, ConsciousnessTier, DomainModule, NavItem};

pub struct PraxisDomain;

impl DomainModule for PraxisDomain {
    fn id(&self) -> &'static str { "praxis" }
    fn name(&self) -> &'static str { "Education" }
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

    fn min_tier(&self) -> ConsciousnessTier { ConsciousnessTier::Participant }
    fn key_context(&self) -> &'static [u8] { b"mycelix-praxis-v1" }
    fn happ_role(&self) -> &'static str { "praxis" }

    fn zomes(&self) -> &'static [&'static str] {
        &["learning", "fl", "credential", "dao", "srs", "gamification", "adaptive", "integration"]
    }
}
