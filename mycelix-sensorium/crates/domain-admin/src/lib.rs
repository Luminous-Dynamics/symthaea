// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admin/IT domain module — system health, conductor monitoring, cluster catalog.

pub mod types;

use sensorium_domain_trait::{
    ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct AdminDomain;

impl DomainModule for AdminDomain {
    fn id(&self) -> &'static str { "admin" }
    fn name(&self) -> &'static str { "System" }
    fn bio_name(&self) -> &'static str { "Nervous System" }
    fn description(&self) -> &'static str {
        "System administration: conductor health, installed hApp status, WASM cell inspection, cluster catalog, and data sovereignty controls."
    }

    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#475569", glow: "#94A3B8" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Health", bio_label: "Vital Signs", path: "/admin/health" },
            NavItem { label: "Conductors", bio_label: "Nerve Centers", path: "/admin/conductors" },
            NavItem { label: "Catalog", bio_label: "Organism Map", path: "/admin/catalog" },
            NavItem { label: "Sovereignty", bio_label: "Membrane Control", path: "/admin/sovereignty" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Guardian }
    fn key_context(&self) -> &'static [u8] { b"mycelix-admin-v1" }
    fn happ_role(&self) -> &'static str { "admin" }
    fn zomes(&self) -> &'static [&'static str] { &[] }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Conductor Config", zome: "admin", sensitivity: DataSensitivity::Private },
        ]
    }
}
