// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance domain module — economic metabolism of the Mycelix network.

pub mod types;

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

pub struct FinanceDomain;

impl DomainModule for FinanceDomain {
    fn id(&self) -> &'static str {
        "finance"
    }
    fn name(&self) -> &'static str {
        "Finance"
    }
    fn bio_name(&self) -> &'static str {
        "Metabolism"
    }
    fn description(&self) -> &'static str {
        "Economic metabolism: SAP/TEND/MYCEL payments, community treasury, staking, and recognition for contributions to the commons."
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#D97706",
            glow: "#FBBF24",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Payments",
                bio_label: "Circulation",
                path: "/finance/payments",
            },
            NavItem {
                label: "Treasury",
                bio_label: "Reserve",
                path: "/finance/treasury",
            },
            NavItem {
                label: "Staking",
                bio_label: "Root System",
                path: "/finance/staking",
            },
            NavItem {
                label: "Recognition",
                bio_label: "Fruiting",
                path: "/finance/recognition",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Citizen
    }
    fn key_context(&self) -> &'static [u8] {
        b"mycelix-finance-v1"
    }
    fn happ_role(&self) -> &'static str {
        "finance"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &["payments", "treasury", "staking", "recognition"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "Payment recipient DID",
                required: true,
            },
            ClusterDependency {
                cluster_id: "governance",
                reason: "Treasury proposals",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Payment",
                zome: "payments",
                sensitivity: DataSensitivity::Private,
            },
            EntryTypeInfo {
                label: "Treasury Entry",
                zome: "treasury",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Stake",
                zome: "staking",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Recognition Award",
                zome: "recognition",
                sensitivity: DataSensitivity::Community,
            },
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "tend",
            label: "Open TEND",
            path: "/tend",
            kind: LaunchKind::InternalRoute,
            requires_unlock: false,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "treasury",
                label: "Treasury",
                path: "/treasury",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "staking",
                label: "Staking",
                path: "/staking",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "finance",
            title: "Economic Metabolism".into(),
            availability: DomainAvailability::Mock,
            status_line:
                "Finance can surface current balance, staking posture, and recent flow without opening the full domain app."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "balance",
                    label: "Balance".into(),
                    value: "4 SAP".into(),
                    hint: Some("demo-scale summary".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "stake",
                    label: "Staked".into(),
                    value: "1 SAP".into(),
                    hint: Some("active".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "rewards",
                    label: "Pending Rewards".into(),
                    value: "0.045 SAP".into(),
                    hint: Some("claimable".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "mycel",
                    label: "MYCEL".into(),
                    value: "0.72".into(),
                    hint: Some("reputation score".into()),
                    tone: None,
                },
            ],
            attention: vec![DomainAttentionItem {
                id: "finance-rewards".into(),
                label: "Rewards available".into(),
                detail: "Staking rewards are available to claim from the Finance app.".into(),
                level: AttentionLevel::Notice,
                path: Some("/staking".into()),
            }],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
