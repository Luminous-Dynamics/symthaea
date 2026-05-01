// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hearth domain module — kinship, gratitude, care, autonomy, decisions, stories.

pub mod types;

use sensorium_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct HearthDomain;

impl DomainModule for HearthDomain {
    fn id(&self) -> &'static str { "hearth" }
    fn name(&self) -> &'static str { "Hearth" }
    fn bio_name(&self) -> &'static str { "Kinship" }
    fn description(&self) -> &'static str {
        "The intimate layer: family and chosen-kin networks, gratitude webs, care coordination, family council decisions, and life story archives."
    }

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

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Kin network linked to DID", required: true },
            ClusterDependency { cluster_id: "commons", reason: "Household resource sharing", required: false },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Kinship Bond", zome: "kinship", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Gratitude Note", zome: "gratitude", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Care Record", zome: "care", sensitivity: DataSensitivity::Sensitive },
            EntryTypeInfo { label: "Family Decision", zome: "decisions", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Story", zome: "stories", sensitivity: DataSensitivity::Protected },
        ]
    }
}
