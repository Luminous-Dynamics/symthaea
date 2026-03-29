// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain module trait — the contract every Mycelix cluster implements
//! to integrate with the unified portal.
//!
//! Adding a new domain to the portal requires:
//! 1. Create a crate implementing `DomainModule`
//! 2. Add a feature flag to the portal's Cargo.toml
//! 3. Add a CSS color override for `[data-domain="your-id"]`
//!
//! No changes to the portal shell needed.

/// Consciousness tier — mirrors mycelix-bridge-common::ConsciousnessTier.
/// Duplicated here to avoid depending on HDK in the portal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsciousnessTier {
    /// Score < 0.3 — can see Identity, Personal
    Observer = 0,
    /// Score >= 0.3 — can see Health, EduNet, Hearth
    Participant = 1,
    /// Score >= 0.4 — can see Governance, Commons, Finance
    Citizen = 2,
    /// Score >= 0.6 — can see Civic, Knowledge, Mail
    Steward = 3,
    /// Score >= 0.8 — can see all domains
    Guardian = 4,
}

impl ConsciousnessTier {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 { Self::Guardian }
        else if score >= 0.6 { Self::Steward }
        else if score >= 0.4 { Self::Citizen }
        else if score >= 0.3 { Self::Participant }
        else { Self::Observer }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Seed",
            Self::Participant => "Sprout",
            Self::Citizen => "Fruiting Body",
            Self::Steward => "Mycelial Network",
            Self::Guardian => "Forest",
        }
    }
}

/// Color family for a domain — primary and glow colors.
#[derive(Clone, Debug)]
pub struct ColorFamily {
    /// CSS color for primary elements (e.g., "#0D7377")
    pub primary: &'static str,
    /// CSS color for glow/accent (e.g., "#06D6C8")
    pub glow: &'static str,
}

/// A route contributed by a domain module.
#[derive(Clone, Debug)]
pub struct DomainRoute {
    /// URL path (e.g., "/health/records")
    pub path: &'static str,
    /// Label shown in breadcrumbs/tabs
    pub label: &'static str,
}

/// A navigation item for the sidebar.
#[derive(Clone, Debug)]
pub struct NavItem {
    /// Display label
    pub label: &'static str,
    /// Biological metaphor label (e.g., "Tissue" for Records)
    pub bio_label: &'static str,
    /// URL path
    pub path: &'static str,
}

/// The contract every Mycelix domain implements for portal integration.
pub trait DomainModule {
    /// Unique identifier (e.g., "health", "governance", "edunet").
    fn id(&self) -> &'static str;

    /// Human-readable name (e.g., "Health", "Governance").
    fn name(&self) -> &'static str;

    /// Domain color palette.
    fn color_family(&self) -> ColorFamily;

    /// Navigation items this domain contributes to the sidebar.
    fn nav_items(&self) -> Vec<NavItem>;

    /// Minimum consciousness tier to see this domain.
    /// Domains below the user's tier appear as dim locked nodes.
    fn min_tier(&self) -> ConsciousnessTier;

    /// HKDF context bytes for deriving the domain-specific key.
    /// Must be unique per domain and stable across versions.
    fn key_context(&self) -> &'static [u8];

    /// Holochain hApp role name for conductor calls.
    fn happ_role(&self) -> &'static str;

    /// Zome names this domain can call.
    fn zomes(&self) -> &'static [&'static str];
}

/// Registry of all compiled domain modules.
/// Built at compile time via feature flags.
pub struct DomainRegistry {
    pub domains: Vec<Box<dyn DomainModule>>,
}

impl DomainRegistry {
    pub fn new() -> Self {
        Self { domains: vec![] }
    }

    pub fn register(&mut self, domain: Box<dyn DomainModule>) {
        self.domains.push(domain);
    }

    /// Get domains visible at a given tier.
    pub fn visible_at(&self, tier: ConsciousnessTier) -> Vec<&dyn DomainModule> {
        self.domains.iter()
            .filter(|d| d.min_tier() <= tier)
            .map(|d| d.as_ref())
            .collect()
    }

    /// Find a domain by ID.
    pub fn get(&self, id: &str) -> Option<&dyn DomainModule> {
        self.domains.iter().find(|d| d.id() == id).map(|d| d.as_ref())
    }
}
