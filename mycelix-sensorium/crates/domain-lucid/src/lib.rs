// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LUCID domain module — consciousness dashboard, collective sensemaking.

pub mod types;

use sensorium_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct LucidDomain;

impl DomainModule for LucidDomain {
    fn id(&self) -> &'static str { "lucid" }
    fn name(&self) -> &'static str { "LUCID" }
    fn bio_name(&self) -> &'static str { "Inner Eye" }
    fn description(&self) -> &'static str {
        "Consciousness dashboard and collective sensemaking: thought streams, relationship graphs, collective knowledge building, and reasoning traces."
    }

    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#8B5CF6", glow: "#C4B5FD" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Thoughts", bio_label: "Spore Clouds", path: "/lucid/thoughts" },
            NavItem { label: "Relationships", bio_label: "Mycelial Bonds", path: "/lucid/relationships" },
            NavItem { label: "Knowledge Graph", bio_label: "Noosphere", path: "/lucid/graph" },
            NavItem { label: "Collective", bio_label: "Hive Mind", path: "/lucid/collective" },
            NavItem { label: "Reasoning", bio_label: "Logic Threads", path: "/lucid/reasoning" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Participant }
    fn key_context(&self) -> &'static [u8] { b"mycelix-lucid-v1" }
    fn happ_role(&self) -> &'static str { "lucid" }

    fn zomes(&self) -> &'static [&'static str] {
        &["lucid", "bridge", "collective", "reasoning", "temporal",
          "temporal_consciousness", "sources", "privacy"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Thought authorship linked to DID", required: true },
            ClusterDependency { cluster_id: "knowledge", reason: "Cross-domain knowledge graph", required: false },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Thought Stream", zome: "lucid", sensitivity: DataSensitivity::Private },
            EntryTypeInfo { label: "Relationship", zome: "lucid", sensitivity: DataSensitivity::Protected },
            EntryTypeInfo { label: "Collective Post", zome: "collective", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Reasoning Trace", zome: "reasoning", sensitivity: DataSensitivity::Protected },
        ]
    }
}
