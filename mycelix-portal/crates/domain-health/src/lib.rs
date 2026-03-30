// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health domain module for the Mycelix Portal.
//!
//! Provides: patient records, consent management, privacy/FL dashboard,
//! data dividends, clinical trials.

pub mod types;

use portal_domain_trait::*;

pub struct HealthDomain;

impl DomainModule for HealthDomain {
    fn id(&self) -> &'static str { "health" }

    fn name(&self) -> &'static str { "Health" }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#0D7377",
            glow: "#06D6C8",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Home", bio_label: "Homeostasis", path: "/health" },
            NavItem { label: "Records", bio_label: "Tissue", path: "/health/records" },
            NavItem { label: "Consent", bio_label: "Symbiosis", path: "/health/consent" },
            NavItem { label: "Privacy", bio_label: "Membrane", path: "/health/privacy" },
            NavItem { label: "Metabolism", bio_label: "Yield", path: "/health/metabolism" },
        ]
    }

    fn min_tier(&self) -> ConsciousnessTier {
        ConsciousnessTier::Participant
    }

    fn key_context(&self) -> &'static [u8] {
        // Backward compatible with existing health vault keys
        b"mycelix-health-v1-patient-encryption"
    }

    fn happ_role(&self) -> &'static str { "health" }

    fn zomes(&self) -> &'static [&'static str] {
        &["patient", "provider", "records", "consent", "prescriptions",
          "trials", "dividends", "fhir_bridge", "fhir_mapping",
          "cds", "credentials", "insurance", "nutrition", "telehealth",
          "mental_health"]
    }
}
