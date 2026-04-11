// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hearth domain module — kinship, gratitude, care, autonomy, decisions, stories.

pub mod types;

use portal_domain_trait::{ColorFamily, CivicTier, DomainModule, NavItem};

pub struct HearthDomain;

impl DomainModule for HearthDomain {
    fn id(&self) -> &'static str { "hearth" }
    fn name(&self) -> &'static str { "Hearth" }
    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#DB2777", glow: "#F472B6" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Kinship", bio_label: "Root Network", path: "/hearth/kinship" },
            NavItem { label: "Gratitude", bio_label: "Fruiting", path: "/hearth/gratitude" },
            NavItem { label: "Care Circle", bio_label: "Tending", path: "/hearth/care" },
            NavItem { label: "Decisions", bio_label: "Council Fire", path: "/hearth/decisions" },
            NavItem { label: "Stories", bio_label: "Oral Tradition", path: "/hearth/stories" },
            NavItem { label: "Milestones", bio_label: "Growth Rings", path: "/hearth/milestones" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Participant }
    fn key_context(&self) -> &'static [u8] { b"mycelix-hearth-v1" }
    fn happ_role(&self) -> &'static str { "hearth" }

    fn zomes(&self) -> &'static [&'static str] {
        &["kinship", "gratitude", "care", "autonomy", "decisions", "stories",
          "milestones", "rhythms", "emergency", "resources", "bridge"]
    }
}
