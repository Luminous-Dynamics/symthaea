// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health domain module for the Mycelix Sensorium.
//!
//! Provides: patient records, consent management, privacy/FL dashboard,
//! data dividends, clinical trials.

pub mod types;

use sensorium_domain_trait::*;

pub struct HealthDomain;

impl DomainModule for HealthDomain {
    fn id(&self) -> &'static str {
        "health"
    }

    fn name(&self) -> &'static str {
        "Health"
    }

    fn bio_name(&self) -> &'static str {
        "Homeostasis"
    }

    fn description(&self) -> &'static str {
        "Sovereign health records, consent-gated data sharing, federated learning data dividends, and clinical trial participation — your body, your data."
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "DID-linked patient records",
                required: true,
            },
            ClusterDependency {
                cluster_id: "finance",
                reason: "Data dividend payouts",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Patient Record",
                zome: "patient",
                sensitivity: DataSensitivity::Sensitive,
            },
            EntryTypeInfo {
                label: "Consent Grant",
                zome: "consent",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Clinical Trial",
                zome: "trials",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Prescription",
                zome: "prescriptions",
                sensitivity: DataSensitivity::Sensitive,
            },
            EntryTypeInfo {
                label: "FL Gradient",
                zome: "dividends",
                sensitivity: DataSensitivity::Private,
            },
        ]
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#0D7377",
            glow: "#06D6C8",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Home",
                bio_label: "Homeostasis",
                path: "/health",
            },
            NavItem {
                label: "Records",
                bio_label: "Tissue",
                path: "/health/records",
            },
            NavItem {
                label: "Consent",
                bio_label: "Symbiosis",
                path: "/health/consent",
            },
            NavItem {
                label: "Privacy",
                bio_label: "Membrane",
                path: "/health/privacy",
            },
            NavItem {
                label: "Metabolism",
                bio_label: "Yield",
                path: "/health/metabolism",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Participant
    }

    fn key_context(&self) -> &'static [u8] {
        // Backward compatible with existing health vault keys
        b"mycelix-health-v1-patient-encryption"
    }

    fn happ_role(&self) -> &'static str {
        "health"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &[
            "patient",
            "provider",
            "records",
            "consent",
            "prescriptions",
            "trials",
            "dividends",
            "fhir_bridge",
            "fhir_mapping",
            "cds",
            "credentials",
            "insurance",
            "nutrition",
            "telehealth",
            "mental_health",
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "records",
            label: "Open Records",
            path: "/records",
            kind: LaunchKind::InternalRoute,
            requires_unlock: true,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "consent",
                label: "Review Consent",
                path: "/consent",
                kind: LaunchKind::InternalRoute,
                requires_unlock: true,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "privacy",
                label: "Privacy Budget",
                path: "/privacy",
                kind: LaunchKind::InternalRoute,
                requires_unlock: true,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "health",
            title: "Health Posture".into(),
            availability: DomainAvailability::Locked,
            status_line:
                "Health vault is present, but records and consent details remain protected until unlock."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "records",
                    label: "Records".into(),
                    value: "4".into(),
                    hint: Some("encrypted in vault".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "consents",
                    label: "Active Consents".into(),
                    value: "2".into(),
                    hint: Some("1 revoked".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "privacy",
                    label: "Privacy Budget".into(),
                    value: "2.4 ε".into(),
                    hint: Some("remaining".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "yield",
                    label: "FL Dividends".into(),
                    value: "128 SAP".into(),
                    hint: Some("research payouts".into()),
                    tone: None,
                },
            ],
            attention: vec![
                DomainAttentionItem {
                    id: "health-locked".into(),
                    label: "Vault locked".into(),
                    detail: "Unlock is required before reviewing records, privacy budget, or clinical sharing state."
                        .into(),
                    level: AttentionLevel::ActionNeeded,
                    path: Some("/records".into()),
                },
                DomainAttentionItem {
                    id: "consent-expiry".into(),
                    label: "Consent review recommended".into(),
                    detail: "At least one active consent should be reviewed soon for scope and expiry."
                        .into(),
                    level: AttentionLevel::Notice,
                    path: Some("/consent".into()),
                },
            ],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
