// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance domain module — economic metabolism of the Mycelix network.

pub mod types;

use portal_domain_trait::{
    ColorFamily, CivicTier, DomainModule, NavItem,
};

pub struct FinanceDomain;

impl DomainModule for FinanceDomain {
    fn id(&self) -> &'static str { "finance" }
    fn name(&self) -> &'static str { "Finance" }

    fn color_family(&self) -> ColorFamily {
        ColorFamily { primary: "#D97706", glow: "#FBBF24" }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Payments", bio_label: "Circulation", path: "/finance/payments" },
            NavItem { label: "Treasury", bio_label: "Reserve", path: "/finance/treasury" },
            NavItem { label: "Staking", bio_label: "Root System", path: "/finance/staking" },
            NavItem { label: "Recognition", bio_label: "Fruiting", path: "/finance/recognition" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Citizen }
    fn key_context(&self) -> &'static [u8] { b"mycelix-finance-v1" }
    fn happ_role(&self) -> &'static str { "finance" }

    fn zomes(&self) -> &'static [&'static str] {
        &["payments", "treasury", "staking", "recognition"]
    }
}
