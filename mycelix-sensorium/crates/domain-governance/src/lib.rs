// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance domain module for Mycelix Sensorium.
//!
//! Implements the `DomainModule` trait to surface proposals, voting,
//! councils, constitution, and budgeting through the consciousness orb.
//!
//! Types mirror the governance zome integrity entries (no HDI dependency).

pub mod types;

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

/// Governance domain module.
pub struct GovernanceDomain;

impl DomainModule for GovernanceDomain {
    fn id(&self) -> &'static str {
        "governance"
    }

    fn name(&self) -> &'static str {
        "Governance"
    }

    fn bio_name(&self) -> &'static str {
        "Consensus"
    }

    fn description(&self) -> &'static str {
        "Anti-tyranny governance with Phi-weighted voting, council coordination, constitutional amendments, and transparent budget execution."
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "Voter DID verification",
                required: true,
            },
            ClusterDependency {
                cluster_id: "finance",
                reason: "Budget proposal execution",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Proposal",
                zome: "proposals",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Vote",
                zome: "voting",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Council Seat",
                zome: "councils",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Constitutional Amendment",
                zome: "constitution",
                sensitivity: DataSensitivity::Public,
            },
            EntryTypeInfo {
                label: "Budget Line",
                zome: "budgeting",
                sensitivity: DataSensitivity::Community,
            },
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
                path: "/proposals",
            },
            NavItem {
                label: "Voting",
                bio_label: "Consensus",
                path: "/voting",
            },
            NavItem {
                label: "Councils",
                bio_label: "Mycorrhizal Nodes",
                path: "/councils",
            },
            NavItem {
                label: "Constitution",
                bio_label: "Root Code",
                path: "/constitution",
            },
            NavItem {
                label: "Budget",
                bio_label: "Metabolism",
                path: "/budgeting",
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
        &[
            "proposals",
            "voting",
            "councils",
            "constitution",
            "budgeting",
            "execution",
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "proposals",
            label: "Open Proposals",
            path: "/proposals",
            kind: LaunchKind::InternalRoute,
            requires_unlock: false,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "voting",
                label: "Voting",
                path: "/voting",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "budgeting",
                label: "Budgeting",
                path: "/budgeting",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "constitution",
                label: "Constitution",
                path: "/constitution",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "governance",
            title: "Consensus State".into(),
            availability: DomainAvailability::Mock,
            status_line:
                "Governance can summarize proposal load, council activity, and execution posture before you enter the full civic workspace."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "proposals",
                    label: "Active Proposals".into(),
                    value: "2".into(),
                    hint: Some("1 executed recently".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "councils",
                    label: "Councils".into(),
                    value: "3".into(),
                    hint: Some("active".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "quorum",
                    label: "Quorum".into(),
                    value: "Met".into(),
                    hint: Some("for current cycle".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "budget",
                    label: "Budget Items".into(),
                    value: "4".into(),
                    hint: Some("pending review".into()),
                    tone: Some("notice"),
                },
            ],
            attention: vec![DomainAttentionItem {
                id: "gov-vote".into(),
                label: "Votes pending".into(),
                detail: "At least one active proposal still needs participation before its window closes."
                    .into(),
                level: AttentionLevel::Notice,
                path: Some("/voting".into()),
            }],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
