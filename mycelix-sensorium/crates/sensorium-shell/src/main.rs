// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Sensorium — the Consciousness Orb.

pub mod ai;
mod app;
mod background;
mod identity;
mod pages;
mod summary;

/// Build the domain registry from all compile-time feature-enabled domains.
/// Called by catalog and sovereignty pages to enumerate installed clusters.
pub fn build_registry() -> sensorium_domain_trait::DomainRegistry {
    let mut registry = sensorium_domain_trait::DomainRegistry::new();
    #[cfg(feature = "personal")]
    registry.register(Box::new(domain_personal::PersonalDomain));
    #[cfg(feature = "health")]
    registry.register(Box::new(domain_health::HealthDomain));
    #[cfg(feature = "governance")]
    registry.register(Box::new(domain_governance::GovernanceDomain));
    #[cfg(feature = "commons")]
    registry.register(Box::new(domain_commons::CommonsDomain));
    #[cfg(feature = "finance")]
    registry.register(Box::new(domain_finance::FinanceDomain));
    #[cfg(feature = "praxis")]
    registry.register(Box::new(domain_praxis::PraxisDomain));
    #[cfg(feature = "hearth")]
    registry.register(Box::new(domain_hearth::HearthDomain));
    #[cfg(feature = "knowledge")]
    registry.register(Box::new(domain_knowledge::KnowledgeDomain));
    #[cfg(feature = "lucid")]
    registry.register(Box::new(domain_lucid::LucidDomain));
    #[cfg(feature = "mail")]
    registry.register(Box::new(domain_mail::MailDomain));
    #[cfg(feature = "admin")]
    registry.register(Box::new(domain_admin::AdminDomain));
    registry
}

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::App);
}
