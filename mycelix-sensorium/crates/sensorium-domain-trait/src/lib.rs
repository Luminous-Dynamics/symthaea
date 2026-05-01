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
//! No changes to the Sensorium shell needed.

/// Consciousness tier — Civic tier for domain access gating.
/// Duplicated here to avoid depending on HDK in the portal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CivicTier {
    /// Score < 0.3 — can see Identity, Personal
    Observer = 0,
    /// Score >= 0.3 — can see Health, Praxis, Hearth
    Participant = 1,
    /// Score >= 0.4 — can see Governance, Commons, Finance
    Citizen = 2,
    /// Score >= 0.6 — can see Civic, Knowledge, Mail
    Steward = 3,
    /// Score >= 0.8 — can see all domains
    Guardian = 4,
}

impl CivicTier {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Guardian
        } else if score >= 0.6 {
            Self::Steward
        } else if score >= 0.4 {
            Self::Citizen
        } else if score >= 0.3 {
            Self::Participant
        } else {
            Self::Observer
        }
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

/// Dependency on another cluster.
#[derive(Clone, Debug)]
pub struct ClusterDependency {
    /// ID of the required cluster (e.g., "identity").
    pub cluster_id: &'static str,
    /// Human-readable reason for the dependency.
    pub reason: &'static str,
    /// If `true`, this domain cannot function without the dependency.
    pub required: bool,
}

/// Data sensitivity classification for sovereignty dashboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataSensitivity {
    /// Publicly visible to all network participants.
    Public,
    /// Visible to community members with appropriate tier.
    Community,
    /// Shared only with explicit consent grants.
    Protected,
    /// Only accessible to the owner.
    Private,
    /// Highly sensitive — encrypted at rest, ZKP for any disclosure.
    Sensitive,
}

impl DataSensitivity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Public => "Public",
            Self::Community => "Community",
            Self::Protected => "Protected",
            Self::Private => "Private",
            Self::Sensitive => "Sensitive",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Public => "sensitivity-public",
            Self::Community => "sensitivity-community",
            Self::Protected => "sensitivity-protected",
            Self::Private => "sensitivity-private",
            Self::Sensitive => "sensitivity-sensitive",
        }
    }
}

/// Metadata about a Holochain entry type — for sovereignty dashboard.
#[derive(Clone, Debug)]
pub struct EntryTypeInfo {
    /// Human-readable label (e.g., "Patient Record").
    pub label: &'static str,
    /// Which zome owns this entry type.
    pub zome: &'static str,
    /// Sensitivity classification.
    pub sensitivity: DataSensitivity,
}

/// Live-state availability for Sensorium summary cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainAvailability {
    Live,
    Mock,
    Empty,
    Locked,
    Degraded,
    Unavailable,
}

impl DomainAvailability {
    pub fn label(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Mock => "Mock",
            Self::Empty => "Empty",
            Self::Locked => "Locked",
            Self::Degraded => "Degraded",
            Self::Unavailable => "Unavailable",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Live => "availability-live",
            Self::Mock => "availability-mock",
            Self::Empty => "availability-empty",
            Self::Locked => "availability-locked",
            Self::Degraded => "availability-degraded",
            Self::Unavailable => "availability-unavailable",
        }
    }
}

/// Launch behavior for a portal launch target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchKind {
    InternalRoute,
    ExternalApp,
    Disabled,
}

/// Attention level for domain summary items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionLevel {
    Quiet,
    Notice,
    ActionNeeded,
    Urgent,
}

impl AttentionLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Quiet => "Quiet",
            Self::Notice => "Notice",
            Self::ActionNeeded => "Action Needed",
            Self::Urgent => "Urgent",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Quiet => "attention-quiet",
            Self::Notice => "attention-notice",
            Self::ActionNeeded => "attention-action-needed",
            Self::Urgent => "attention-urgent",
        }
    }
}

/// Primary or secondary launch action surfaced by the portal.
#[derive(Clone, Debug)]
pub struct DomainLaunchTarget {
    pub id: &'static str,
    pub label: &'static str,
    pub path: &'static str,
    pub kind: LaunchKind,
    pub requires_unlock: bool,
    pub recommended: bool,
}

/// Small headline metric shown on a domain summary card.
#[derive(Clone, Debug)]
pub struct DomainMetric {
    pub id: &'static str,
    pub label: String,
    pub value: String,
    pub hint: Option<String>,
    pub tone: Option<&'static str>,
}

/// Action-worthy or informative summary item for the shell.
#[derive(Clone, Debug)]
pub struct DomainAttentionItem {
    pub id: String,
    pub label: String,
    pub detail: String,
    pub level: AttentionLevel,
    pub path: Option<String>,
}

/// Generic summary payload rendered by the Sensorium shell.
#[derive(Clone, Debug)]
pub struct DomainSummaryCard {
    pub domain_id: &'static str,
    pub title: String,
    pub availability: DomainAvailability,
    pub status_line: String,
    pub metrics: Vec<DomainMetric>,
    pub attention: Vec<DomainAttentionItem>,
    pub primary_launch: Option<DomainLaunchTarget>,
    pub secondary_launches: Vec<DomainLaunchTarget>,
    pub updated_at: Option<i64>,
}

/// The contract every Mycelix domain implements for Sensorium integration.
pub trait DomainModule {
    /// Unique identifier (e.g., "health", "governance", "praxis").
    fn id(&self) -> &'static str;

    /// Human-readable name (e.g., "Health", "Governance").
    fn name(&self) -> &'static str;

    /// Biological metaphor name (e.g., "Homeostasis", "Consensus").
    fn bio_name(&self) -> &'static str;

    /// One-sentence description of the domain's purpose.
    fn description(&self) -> &'static str;

    /// Domain color palette.
    fn color_family(&self) -> ColorFamily;

    /// Navigation items this domain contributes to the sidebar.
    fn nav_items(&self) -> Vec<NavItem>;

    /// Minimum consciousness tier to see this domain.
    /// Domains below the user's tier appear as dim locked nodes.
    fn min_tier(&self) -> CivicTier;

    /// HKDF context bytes for deriving the domain-specific key.
    /// Must be unique per domain and stable across versions.
    fn key_context(&self) -> &'static [u8];

    /// Holochain hApp role name for conductor calls.
    fn happ_role(&self) -> &'static str;

    /// Zome names this domain can call.
    fn zomes(&self) -> &'static [&'static str];

    /// Other clusters this domain depends on.
    /// Used by the catalog and sovereignty dashboard.
    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[]
    }

    /// Key entry types managed by this domain.
    /// Used by the sovereignty dashboard to build the data inventory.
    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[]
    }

    /// Primary launch action from the Sensorium shell.
    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        None
    }

    /// Secondary launch actions from the Sensorium shell.
    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![]
    }

    /// Typed summary payload for the Sensorium shell.
    fn summary_card(&self) -> Option<DomainSummaryCard> {
        None
    }
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
    pub fn visible_at(&self, tier: CivicTier) -> Vec<&dyn DomainModule> {
        self.domains
            .iter()
            .filter(|d| d.min_tier() <= tier)
            .map(|d| d.as_ref())
            .collect()
    }

    /// Find a domain by ID.
    pub fn get(&self, id: &str) -> Option<&dyn DomainModule> {
        self.domains
            .iter()
            .find(|d| d.id() == id)
            .map(|d| d.as_ref())
    }
}
