// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail domain module — inbox summary via REST API bridge to Mail React app.
//!
//! The Mail frontend (90K LOC React) stays as-is. This domain module
//! consumes Mail's REST API to show inbox summary, trust scores, and
//! basic compose in the unified portal.

pub mod types;

use portal_domain_trait::{ColorFamily, CivicTier, DomainModule, NavItem};

pub struct MailDomain;

impl DomainModule for MailDomain {
    fn id(&self) -> &'static str { "mail" }
    fn name(&self) -> &'static str { "Mail" }
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
    fn zomes(&self) -> &'static [&'static str] { &["mail_messages", "trust", "contacts", "profiles"] }
}
