// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commons domain module — shared resources for community wellbeing.
//!
//! Covers 39 zomes across property, housing, care, mutual aid, water,
//! food, transport, mesh-time, and resource-mesh.

pub mod types;

use portal_domain_trait::{
    ColorFamily, CivicTier, DomainModule, NavItem,
};

pub struct CommonsDomain;

impl DomainModule for CommonsDomain {
    fn id(&self) -> &'static str { "commons" }
    fn name(&self) -> &'static str { "Commons" }

    fn color_family(&self) -> ColorFamily {
        ColorFamily { primary: "#059669", glow: "#34D399" }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Property", bio_label: "Territory", path: "/commons/property" },
            NavItem { label: "Housing", bio_label: "Shelter", path: "/commons/housing" },
            NavItem { label: "Care", bio_label: "Tending", path: "/commons/care" },
            NavItem { label: "Mutual Aid", bio_label: "Symbiosis", path: "/commons/mutual-aid" },
            NavItem { label: "Water", bio_label: "Flow", path: "/commons/water" },
            NavItem { label: "Food", bio_label: "Nourishment", path: "/commons/food" },
            NavItem { label: "Transport", bio_label: "Mycelial Paths", path: "/commons/transport" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Citizen }
    fn key_context(&self) -> &'static [u8] { b"mycelix-commons-v1" }
    fn happ_role(&self) -> &'static str { "commons" }

    fn zomes(&self) -> &'static [&'static str] {
        &[
            "property_registry", "housing_units", "care_plans",
            "mutualaid_requests", "water_flow", "food_distribution",
            "transport_routes", "mesh_time", "resource_mesh",
        ]
    }
}
