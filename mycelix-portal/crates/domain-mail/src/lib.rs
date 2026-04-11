// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain module — PQC-encrypted decentralized email via Mycelix Pulse.

pub mod types;

use portal_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct MailDomain;

impl DomainModule for MailDomain {
    fn id(&self) -> &'static str { "mail" }
    fn name(&self) -> &'static str { "Mail" }
    fn bio_name(&self) -> &'static str { "Signal Stream" }
    fn description(&self) -> &'static str {
        "Post-quantum encrypted decentralized communications: email, chat, meet, and calendar — all peer-to-peer with web-of-trust spam filtering."
    }

    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#DC2626", glow: "#FCA5A5" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Inbox", bio_label: "Signal Stream", path: "/mail/inbox" },
            NavItem { label: "Trust", bio_label: "Web of Trust", path: "/mail/trust" },
            NavItem { label: "Compose", bio_label: "Emanate", path: "/mail/compose" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Participant }
    fn key_context(&self) -> &'static [u8] { b"mycelix-mail-v1" }
    fn happ_role(&self) -> &'static str { "mail" }

    fn zomes(&self) -> &'static [&'static str] {
        &["mail_messages", "trust", "contacts", "profiles"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Sender/recipient DID resolution", required: true },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Message", zome: "mail_messages", sensitivity: DataSensitivity::Sensitive },
            EntryTypeInfo { label: "Contact", zome: "contacts", sensitivity: DataSensitivity::Private },
            EntryTypeInfo { label: "Trust Score", zome: "trust", sensitivity: DataSensitivity::Protected },
        ]
    }
}
