// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge domain module — epistemic commons, claims, fact-checking, inference.

pub mod types;

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

pub struct KnowledgeDomain;

impl DomainModule for KnowledgeDomain {
    fn id(&self) -> &'static str {
        "knowledge"
    }
    fn name(&self) -> &'static str {
        "Knowledge"
    }
    fn bio_name(&self) -> &'static str {
        "Noosphere"
    }
    fn description(&self) -> &'static str {
        "Epistemic commons: distributed knowledge claims, inference graphs, decentralized fact-checking, and prediction markets for collective intelligence."
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#0891B2",
            glow: "#22D3EE",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Claims",
                bio_label: "Spores",
                path: "/knowledge/claims",
            },
            NavItem {
                label: "Graph",
                bio_label: "Mycelial Web",
                path: "/knowledge/graph",
            },
            NavItem {
                label: "Fact Check",
                bio_label: "Immune Response",
                path: "/knowledge/factcheck",
            },
            NavItem {
                label: "Markets",
                bio_label: "Prediction Soil",
                path: "/knowledge/markets",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Steward
    }
    fn key_context(&self) -> &'static [u8] {
        b"mycelix-knowledge-v1"
    }
    fn happ_role(&self) -> &'static str {
        "knowledge"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &[
            "claims",
            "graph",
            "query",
            "inference",
            "factcheck",
            "markets",
            "dkg",
            "bridge",
        ]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "Claim author DID attestation",
                required: true,
            },
            ClusterDependency {
                cluster_id: "governance",
                reason: "Epistemic standards proposals",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Knowledge Claim",
                zome: "claims",
                sensitivity: DataSensitivity::Public,
            },
            EntryTypeInfo {
                label: "Inference Rule",
                zome: "inference",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Fact Check",
                zome: "factcheck",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Prediction Market",
                zome: "markets",
                sensitivity: DataSensitivity::Community,
            },
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "browse",
            label: "Browse Claims",
            path: "/browse",
            kind: LaunchKind::InternalRoute,
            requires_unlock: false,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "fact-check",
                label: "Fact Check",
                path: "/fact-check",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "submit",
                label: "Submit Claim",
                path: "/submit",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "knowledge",
            title: "Epistemic Commons".into(),
            availability: DomainAvailability::Mock,
            status_line:
                "Knowledge can summarize claim flow, review load, and prediction activity before you enter the full epistemic workspace."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "claims",
                    label: "Claims".into(),
                    value: "4".into(),
                    hint: Some("current working set".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "verified",
                    label: "Verified".into(),
                    value: "2".into(),
                    hint: Some("high-confidence".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "review",
                    label: "Under Review".into(),
                    value: "1".into(),
                    hint: Some("needs attention".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "markets",
                    label: "Markets".into(),
                    value: "3".into(),
                    hint: Some("active predictions".into()),
                    tone: None,
                },
            ],
            attention: vec![DomainAttentionItem {
                id: "knowledge-review".into(),
                label: "Review queue active".into(),
                detail: "At least one claim is under review and should be resolved or escalated.".into(),
                level: AttentionLevel::Notice,
                path: Some("/fact-check".into()),
            }],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
