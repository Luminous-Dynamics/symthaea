// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Personal domain module — sovereign vault posture, credentials, consent,
//! disclosure history, and handoff into deeper domain workflows.

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

pub struct PersonalDomain;

impl DomainModule for PersonalDomain {
    fn id(&self) -> &'static str {
        "personal"
    }

    fn name(&self) -> &'static str {
        "Personal"
    }

    fn bio_name(&self) -> &'static str {
        "Sovereign Vault"
    }

    fn description(&self) -> &'static str {
        "Unified vault posture for identity, credentials, health privacy, and disclosure controls across Mycelix."
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#A16207",
            glow: "#F59E0B",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Vault",
                bio_label: "Shell",
                path: "/",
            },
            NavItem {
                label: "Identity",
                bio_label: "Face",
                path: "/identity",
            },
            NavItem {
                label: "Wallet",
                bio_label: "Spores",
                path: "/wallet",
            },
            NavItem {
                label: "Health",
                bio_label: "Body",
                path: "/health",
            },
            NavItem {
                label: "Preferences",
                bio_label: "Membrane",
                path: "/preferences",
            },
            NavItem {
                label: "Activity",
                bio_label: "Mycelial Trace",
                path: "/activity",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Observer
    }

    fn key_context(&self) -> &'static [u8] {
        b"mycelix-personal-v1"
    }

    fn happ_role(&self) -> &'static str {
        "personal"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &[
            "identity_vault",
            "health_vault",
            "credential_wallet",
            "data_preferences",
            "personal_bridge",
        ]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "Identity handoff for DID, MFA, and recovery flows",
                required: false,
            },
            ClusterDependency {
                cluster_id: "health",
                reason: "Deep health workflows live in the Health app",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Profile",
                zome: "identity_vault",
                sensitivity: DataSensitivity::Private,
            },
            EntryTypeInfo {
                label: "Master Key",
                zome: "identity_vault",
                sensitivity: DataSensitivity::Sensitive,
            },
            EntryTypeInfo {
                label: "Stored Credential",
                zome: "credential_wallet",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Consent Grant",
                zome: "health_vault",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Preference Policy",
                zome: "data_preferences",
                sensitivity: DataSensitivity::Private,
            },
            EntryTypeInfo {
                label: "Bridge Event",
                zome: "personal_bridge",
                sensitivity: DataSensitivity::Protected,
            },
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "vault",
            label: "Open Vault",
            path: "/",
            kind: LaunchKind::InternalRoute,
            requires_unlock: true,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "identity",
                label: "Identity",
                path: "/identity",
                kind: LaunchKind::InternalRoute,
                requires_unlock: true,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "preferences",
                label: "Preferences",
                path: "/preferences",
                kind: LaunchKind::InternalRoute,
                requires_unlock: true,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "activity",
                label: "Activity",
                path: "/activity",
                kind: LaunchKind::InternalRoute,
                requires_unlock: true,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "personal",
            title: "Sovereign Vault".into(),
            availability: DomainAvailability::Locked,
            status_line:
                "Vault posture is available, but sensitive inventory and disclosure detail remain hidden until unlock."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "credentials",
                    label: "Credentials".into(),
                    value: "8".into(),
                    hint: Some("2 health, 3 identity, 3 trust".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "consents",
                    label: "Active Consents".into(),
                    value: "3".into(),
                    hint: Some("1 changed recently".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "health",
                    label: "Health Records".into(),
                    value: "14".into(),
                    hint: Some("summary only in Personal".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "activity",
                    label: "Recent Disclosures".into(),
                    value: "2".into(),
                    hint: Some("last 7 days".into()),
                    tone: Some("notice"),
                },
            ],
            attention: vec![
                DomainAttentionItem {
                    id: "vault-locked".into(),
                    label: "Unlock required".into(),
                    detail: "Unlock the vault before reviewing preferences, disclosure history, or credential detail."
                        .into(),
                    level: AttentionLevel::ActionNeeded,
                    path: Some("/unlock".into()),
                },
                DomainAttentionItem {
                    id: "consent-review".into(),
                    label: "Consent posture changed".into(),
                    detail: "One health consent was updated recently and should be reviewed.".into(),
                    level: AttentionLevel::Notice,
                    path: Some("/health".into()),
                },
            ],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
