// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LUCID domain module — consciousness dashboard, collective sensemaking.
//!
//! Migrates the SvelteKit LUCID dashboard to a portal domain module.
//! Uses portal-viz ForceGraph for agent relationship networks,
//! KaTeX bridge for math rendering, and Candle WASM for local NLP.

pub mod types;

use portal_domain_trait::{ColorFamily, CivicTier, DomainModule, NavItem};

pub struct LucidDomain;

impl DomainModule for LucidDomain {
    fn id(&self) -> &'static str { "lucid" }
    fn name(&self) -> &'static str { "LUCID" }
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
}
