// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain module — PQC-encrypted decentralized email via Mycelix Pulse.

pub mod types;

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

pub struct MailDomain;

impl DomainModule for MailDomain {
    fn id(&self) -> &'static str {
        "mail"
    }
    fn name(&self) -> &'static str {
        "Mail"
    }
    fn bio_name(&self) -> &'static str {
        "Signal Stream"
    }
    fn description(&self) -> &'static str {
        "Post-quantum encrypted decentralized communications: email, chat, meet, and calendar — all peer-to-peer with web-of-trust spam filtering."
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#DC2626",
            glow: "#FCA5A5",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Inbox",
                bio_label: "Signal Stream",
                path: "/mail/inbox",
            },
            NavItem {
                label: "Trust",
                bio_label: "Web of Trust",
                path: "/mail/trust",
            },
            NavItem {
                label: "Compose",
                bio_label: "Emanate",
                path: "/mail/compose",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Participant
    }
    fn key_context(&self) -> &'static [u8] {
        b"mycelix-mail-v1"
    }
    fn happ_role(&self) -> &'static str {
        "mail"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &["mail_messages", "trust", "contacts", "profiles"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[ClusterDependency {
            cluster_id: "identity",
            reason: "Sender/recipient DID resolution",
            required: true,
        }]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Message",
                zome: "mail_messages",
                sensitivity: DataSensitivity::Sensitive,
            },
            EntryTypeInfo {
                label: "Contact",
                zome: "contacts",
                sensitivity: DataSensitivity::Private,
            },
            EntryTypeInfo {
                label: "Trust Score",
                zome: "trust",
                sensitivity: DataSensitivity::Protected,
            },
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "inbox",
            label: "Open Inbox",
            path: "/mail/inbox",
            kind: LaunchKind::InternalRoute,
            requires_unlock: false,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "trust",
                label: "Trust Network",
                path: "/mail/trust",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "compose",
                label: "Compose",
                path: "/mail/compose",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "mail",
            title: "Signal Stream".into(),
            availability: DomainAvailability::Mock,
            status_line:
                "Pulse should surface unread pressure, trust filtering, and urgent thread posture before opening the full communication shell."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "unread",
                    label: "Unread".into(),
                    value: "12".into(),
                    hint: Some("mock inbox pressure".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "quarantine",
                    label: "Quarantined".into(),
                    value: "3".into(),
                    hint: Some("low-trust held".into()),
                    tone: Some("warning"),
                },
                DomainMetric {
                    id: "pending",
                    label: "Introductions".into(),
                    value: "2".into(),
                    hint: Some("awaiting trust review".into()),
                    tone: Some("notice"),
                },
                DomainMetric {
                    id: "trust",
                    label: "Trust Health".into(),
                    value: "78%".into(),
                    hint: Some("network average".into()),
                    tone: None,
                },
            ],
            attention: vec![
                DomainAttentionItem {
                    id: "pulse-unread".into(),
                    label: "Unread triage building".into(),
                    detail: "Unread communication pressure is high enough to warrant a Pulse review."
                        .into(),
                    level: AttentionLevel::ActionNeeded,
                    path: Some("/mail/inbox".into()),
                },
                DomainAttentionItem {
                    id: "pulse-trust".into(),
                    label: "Trust quarantine active".into(),
                    detail: "Low-trust or unintroduced senders are currently being held for review."
                        .into(),
                    level: AttentionLevel::Notice,
                    path: Some("/mail/trust".into()),
                },
            ],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
