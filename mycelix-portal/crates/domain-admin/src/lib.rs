// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admin/IT domain module — system health, monitoring, RDP web viewer stub.
//!
//! Consolidates the SvelteKit Observatory dashboard into a portal domain module.
//! Also provides the architecture for a web-based RDP session viewer
//! (view-only remote desktop via WebSocket relay).

pub mod types;

use portal_domain_trait::{ColorFamily, CivicTier, DomainModule, NavItem};

pub struct AdminDomain;

impl DomainModule for AdminDomain {
    fn id(&self) -> &'static str { "admin" }
    fn name(&self) -> &'static str { "System" }
    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#475569", glow: "#94A3B8" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Health", bio_label: "Vital Signs", path: "/admin/health" },
            NavItem { label: "Conductors", bio_label: "Nerve Centers", path: "/admin/conductors" },
            NavItem { label: "WASM", bio_label: "Cell Membranes", path: "/admin/wasm" },
            NavItem { label: "Remote", bio_label: "Astral Projection", path: "/admin/remote" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Guardian }
    fn key_context(&self) -> &'static [u8] { b"mycelix-admin-v1" }
    fn happ_role(&self) -> &'static str { "admin" }
    fn zomes(&self) -> &'static [&'static str] { &[] }
}
