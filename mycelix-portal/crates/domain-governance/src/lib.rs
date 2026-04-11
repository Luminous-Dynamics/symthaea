// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance domain module for Mycelix Portal.
//!
//! Implements the `DomainModule` trait to surface proposals, voting,
//! councils, constitution, and budgeting through the consciousness orb.
//!
//! Types mirror the governance zome integrity entries (no HDI dependency).

pub mod types;

use portal_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

/// Governance domain module.
pub struct GovernanceDomain;

impl DomainModule for GovernanceDomain {
    fn id(&self) -> &'static str { "governance" }

    fn name(&self) -> &'static str { "Governance" }

    fn bio_name(&self) -> &'static str { "Consensus" }

    fn description(&self) -> &'static str {
        "Anti-tyranny governance with Phi-weighted voting, council coordination, constitutional amendments, and transparent budget execution."
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Voter DID verification", required: true },
            ClusterDependency { cluster_id: "finance", reason: "Budget proposal execution", required: false },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Proposal", zome: "proposals", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Vote", zome: "voting", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Council Seat", zome: "councils", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Constitutional Amendment", zome: "constitution", sensitivity: DataSensitivity::Public },
            EntryTypeInfo { label: "Budget Line", zome: "budgeting", sensitivity: DataSensitivity::Community },
        ]
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#7C3AED",
            glow: "#A78BFA",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Proposals",
                bio_label: "Intentions",
                path: "/governance/proposals",
            },
            NavItem {
                label: "Voting",
                bio_label: "Consensus",
                path: "/governance/voting",
            },
            NavItem {
                label: "Councils",
                bio_label: "Mycorrhizal Nodes",
                path: "/governance/councils",
            },
            NavItem {
                label: "Constitution",
                bio_label: "Root Code",
                path: "/governance/constitution",
            },
            NavItem {
                label: "Budget",
                bio_label: "Metabolism",
                path: "/governance/budget",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Citizen
    }

    fn key_context(&self) -> &'static [u8] {
        b"mycelix-governance-v1"
    }

    fn happ_role(&self) -> &'static str {
        "governance"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &["proposals", "voting", "councils", "constitution", "budgeting", "execution"]
    }
}
