// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Centralized routing registry for cross-cluster bridge dispatch.
//!
//! Consolidates the hardcoded allowlists from the 4 bridge coordinator zomes
//! (commons-bridge, civic-bridge, hearth-bridge, personal-bridge) into a
//! single source of truth.
//!
//! ## Usage
//!
//! Each bridge coordinator can replace its local `ALLOWED_*_ZOMES` constants
//! and role-name strings with calls to this registry:
//!
//! ```rust,ignore
//! use mycelix_bridge_common::routing_registry;
//!
//! // Check if a zome is reachable from this cluster
//! if routing_registry::is_allowed(
//!     CrossClusterRole::Commons,
//!     CrossClusterRole::Civic,
//!     "justice_cases",
//! ) { /* dispatch */ }
//!
//! // Get the full allowlist
//! let zomes = routing_registry::get_allowed_zomes(
//!     CrossClusterRole::Commons,
//!     CrossClusterRole::Civic,
//! );
//!
//! // Get the hApp role name string
//! let role = routing_registry::role_name(CrossClusterRole::Civic); // "civic"
//! ```

use crate::routing::CrossClusterRole;

// ============================================================================
// Local (intra-cluster) allowlists
// ============================================================================

/// Zomes allowed for local dispatch within the Commons cluster.
pub const COMMONS_LOCAL_ZOMES: &[&str] = &[
    // Property domain (4)
    "property_registry",
    "property_transfer",
    "property_disputes",
    "property_commons",
    // Housing domain (6)
    "housing_units",
    "housing_membership",
    "housing_finances",
    "housing_maintenance",
    "housing_clt",
    "housing_governance",
    // Care domain (5)
    "care_timebank",
    "care_circles",
    "care_matching",
    "care_plans",
    "care_credentials",
    // Mutual aid domain (7)
    "mutualaid_needs",
    "mutualaid_circles",
    "mutualaid_governance",
    "mutualaid_pools",
    "mutualaid_requests",
    "mutualaid_resources",
    "mutualaid_timebank",
    // Water domain (5)
    "water_flow",
    "water_purity",
    "water_capture",
    "water_steward",
    "water_wisdom",
    // Food domain (4)
    "food_production",
    "food_distribution",
    "food_preservation",
    "food_knowledge",
    // Transport domain (3)
    "transport_routes",
    "transport_sharing",
    "transport_impact",
    // Support domain (3)
    "support_knowledge",
    "support_tickets",
    "support_diagnostics",
    // Space (1)
    "space",
    // Mesh-Time (1)
    "mesh_time",
    // Resource-Mesh (1)
    "resource_mesh",
];

/// Zomes allowed for local dispatch within the Civic cluster.
pub const CIVIC_LOCAL_ZOMES: &[&str] = &[
    // Justice domain (5)
    "justice_cases",
    "justice_evidence",
    "justice_arbitration",
    "justice_restorative",
    "justice_enforcement",
    // Emergency domain (6)
    "emergency_incidents",
    "emergency_triage",
    "emergency_resources",
    "emergency_coordination",
    "emergency_shelters",
    "emergency_comms",
    // Media domain (4)
    "media_publication",
    "media_attribution",
    "media_factcheck",
    "media_curation",
    // Resonance domain (1)
    "resonance_feed",
    // Mediation domain (1)
    "mediation",
];

/// Zomes allowed for local dispatch within the Music cluster.
pub const MUSIC_LOCAL_ZOMES: &[&str] = &["catalog", "plays", "balances", "trust"];

/// Zomes allowed for local dispatch within the Hearth cluster.
pub const HEARTH_LOCAL_ZOMES: &[&str] = &[
    "hearth_kinship",
    "hearth_gratitude",
    "hearth_stories",
    "hearth_care",
    "hearth_autonomy",
    "hearth_emergency",
    "hearth_decisions",
    "hearth_resources",
    "hearth_milestones",
    "hearth_rhythms",
];

/// Zomes allowed for local dispatch within the Personal cluster.
pub const PERSONAL_LOCAL_ZOMES: &[&str] = &["identity_vault", "health_vault", "credential_wallet"];

// ============================================================================
// Commons sub-cluster partitioning
// ============================================================================

/// Zomes that live in the commons_land sub-cluster DNA (physical infrastructure).
pub const COMMONS_LAND_ZOMES: &[&str] = &[
    // Property domain
    "property_registry",
    "property_transfer",
    "property_disputes",
    "property_commons",
    // Housing domain
    "housing_units",
    "housing_membership",
    "housing_finances",
    "housing_maintenance",
    "housing_clt",
    "housing_governance",
    // Water domain
    "water_flow",
    "water_purity",
    "water_capture",
    "water_steward",
    "water_wisdom",
    // Food domain
    "food_production",
    "food_distribution",
    "food_preservation",
    "food_knowledge",
];

/// Zomes that live in the commons_care sub-cluster DNA (social/care).
pub const COMMONS_CARE_ZOMES: &[&str] = &[
    // Care domain
    "care_timebank",
    "care_circles",
    "care_matching",
    "care_plans",
    "care_credentials",
    // Mutual aid domain
    "mutualaid_needs",
    "mutualaid_circles",
    "mutualaid_governance",
    "mutualaid_pools",
    "mutualaid_requests",
    "mutualaid_resources",
    "mutualaid_timebank",
    // Transport domain
    "transport_routes",
    "transport_sharing",
    "transport_impact",
    // Support domain
    "support_knowledge",
    "support_tickets",
    "support_diagnostics",
    // Space
    "space",
    // Mesh-Time
    "mesh_time",
    // Resource-Mesh
    "resource_mesh",
];

/// Zomes allowed for local dispatch within the Health cluster.
pub const HEALTH_LOCAL_ZOMES: &[&str] = &[
    // MVP (7)
    "patient",
    "provider",
    "records",
    "prescriptions",
    "consent",
    "bridge",
    "shared",
    // Tier 2 (9)
    "credentials",
    "trials",
    "insurance",
    "fhir_mapping",
    "fhir_bridge",
    "cds",
    "provider_directory",
    "telehealth",
    "nutrition",
];

/// Zomes allowed for local dispatch within the Energy cluster.
pub const ENERGY_LOCAL_ZOMES: &[&str] = &[
    "projects",
    "investments",
    "regenerative",
    "grid",
    "energy_bridge",
];

/// Zomes allowed for local dispatch within the Knowledge cluster.
pub const KNOWLEDGE_LOCAL_ZOMES: &[&str] = &[
    "claims",
    "graph",
    "query",
    "inference",
    "factcheck",
    "markets_integration",
    "dkg",
    "knowledge_bridge",
];

/// Zomes allowed for local dispatch within the Climate cluster.
pub const CLIMATE_LOCAL_ZOMES: &[&str] = &["carbon", "projects", "bridge"];

/// Zomes allowed for local dispatch within the Manufacturing cluster.
pub const MANUFACTURING_LOCAL_ZOMES: &[&str] = &[
    "bom",
    "machines",
    "operations",
    "planning",
    "workorders",
    "manufacturing_bridge",
];

/// Zomes allowed for local dispatch within the Supplychain cluster.
pub const SUPPLYCHAIN_LOCAL_ZOMES: &[&str] = &[
    "bridge_coordinator",
    "claims_coordinator",
    "inventory_coordinator",
    "logistics_coordinator",
    "payments_coordinator",
    "procurement_coordinator",
    "trust_coordinator",
    "verification_coordinator",
];

/// Zomes allowed for local dispatch within the Praxis cluster.
pub const PRAXIS_LOCAL_ZOMES: &[&str] = &[
    "learning_zome",
    "fl_zome",
    "credential_zome",
    "dao_zome",
    "srs_zome",
    "gamification_zome",
    "adaptive_zome",
    "pods_zome",
    "knowledge_zome",
    "integration_zome",
    "mentorship_zome",
    "praxis_bridge",
];

/// Zomes allowed for local dispatch within the Craft cluster.
pub const CRAFT_LOCAL_ZOMES: &[&str] = &[
    "craft_graph",
    "job_postings_coordinator",
    "work_history_coordinator",
    "connection_graph_coordinator",
    "applications_coordinator",
    "guild_coordinator",
];

/// Zomes allowed for local dispatch within the Legacy cluster.
pub const LEGACY_LOCAL_ZOMES: &[&str] = &[
    "legacy_testament",
    "legacy_probate",
    "legacy_activation",
    "legacy_memorial",
    "legacy_bridge",
];

/// Zomes allowed for local dispatch within the Identity cluster.
pub const IDENTITY_LOCAL_ZOMES: &[&str] = &[
    "did_registry",
    "trust_credential",
    "mfa",
    "verifiable_credential",
    "credential_schema",
    "education",
    "revocation",
    "recovery",
    "name_registry",
    "web_of_trust",
    "reputation_aggregator",
    "identity_bridge",
];

/// Zomes allowed for local dispatch within the Finance cluster.
pub const FINANCE_LOCAL_ZOMES: &[&str] = &[
    "finance_bridge",
    "currency_mint",
    "payments",
    "price_oracle",
    "recognition",
    "staking",
    "tend",
    "treasury",
];

/// Zomes allowed for local dispatch within the Governance cluster.
pub const GOVERNANCE_LOCAL_ZOMES: &[&str] = &[
    "governance_bridge",
    "budgeting",
    "constitution",
    "councils",
    "execution",
    "proposals",
    "threshold_signing",
    "voting",
];

/// Zomes allowed for local dispatch within the Cafe cluster.
pub const CAFE_LOCAL_ZOMES: &[&str] = &["cafe_pos", "cafe_inventory", "cafe_roster", "cafe_bridge"];

/// Zomes allowed for local dispatch within the Atlas cluster.
pub const ATLAS_LOCAL_ZOMES: &[&str] = &["atlas_sites", "atlas_infrastructure", "atlas_bridge"];

/// Zomes allowed for local dispatch within the Attribution cluster.
pub const ATTRIBUTION_LOCAL_ZOMES: &[&str] = &["reciprocity", "registry", "usage"];

/// Zomes allowed for local dispatch within the Core cluster (0TML federated learning).
pub const CORE_LOCAL_ZOMES: &[&str] = &[
    "agents",
    "bridge",
    "dkg",
    "epistemic_storage",
    "federated_learning",
    "pogq_validation",
];

/// Zomes allowed for local dispatch within the DeSci cluster (REST API modules).
pub const DESCI_LOCAL_ZOMES: &[&str] = &["claims", "query", "system", "trust"];

/// Zomes allowed for local dispatch within the Mail cluster (PQC-encrypted decentralized email).
pub const MAIL_LOCAL_ZOMES: &[&str] = &[
    "audit",
    "backup",
    "capabilities",
    "contacts",
    "federation",
    "keys",
    "mail_bridge",
    "messages",
    "profiles",
    "scheduler",
    "search",
    "sync",
    "trust",
];

/// Zomes allowed for local dispatch within the Marketplace cluster.
pub const MARKETPLACE_LOCAL_ZOMES: &[&str] = &[
    "arbitration",
    "listings",
    "messaging",
    "monitoring",
    "notifications",
    "reputation",
    "search",
    "security",
    "transactions",
];

/// Zomes allowed for local dispatch within the Position cluster (GPS-without-GPS).
pub const POSITION_LOCAL_ZOMES: &[&str] = &["anchor_registry", "position_estimates", "ranging"];

/// Zomes allowed for local dispatch within the Space cluster (orbital mechanics).
pub const SPACE_LOCAL_ZOMES: &[&str] = &[
    "conjunctions",
    "debris_bounties",
    "observations",
    "orbital_objects",
    "traffic_control",
];

// Note: Lunar, MultiworldSim, Portal, and Workspace are either stubs,
// pure simulations, frontends, or utility crates with no Holochain zomes.
// They have empty local zome lists but are included in the enum for
// routing completeness.

/// Zomes allowed for local dispatch within the Lunar cluster (stub/dormant).
pub const LUNAR_LOCAL_ZOMES: &[&str] = &[];

/// Zomes allowed for local dispatch within the MultiworldSim cluster (pure simulation, no Holochain zomes).
pub const MULTIWORLD_SIM_LOCAL_ZOMES: &[&str] = &[];

/// Zomes allowed for local dispatch within the Portal cluster (Leptos frontend, no Holochain zomes).
pub const PORTAL_LOCAL_ZOMES: &[&str] = &[];

/// Zomes allowed for local dispatch within the Workspace cluster (utility crates, no direct zomes).
pub const WORKSPACE_LOCAL_ZOMES: &[&str] = &[];

// ============================================================================
// Cross-cluster allowlists
// ============================================================================

/// Civic-side zomes that commons-bridge is allowed to call cross-cluster.
const COMMONS_TO_CIVIC: &[&str] = &[
    // Justice domain
    "justice_cases",
    "justice_evidence",
    "justice_arbitration",
    "justice_restorative",
    "justice_enforcement",
    // Emergency domain
    "emergency_incidents",
    "emergency_triage",
    "emergency_resources",
    "emergency_coordination",
    "emergency_shelters",
    "emergency_comms",
    // Media domain
    "media_publication",
    "media_attribution",
    "media_factcheck",
    "media_curation",
    // Civic bridge
    "civic_bridge",
];

/// Identity-side zomes that commons-bridge is allowed to call cross-cluster.
const COMMONS_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry"];

/// Finance-side zomes that commons-bridge is allowed to call cross-cluster.
const COMMONS_TO_FINANCE: &[&str] = &[
    "finance_bridge",
    "currency_mint",
    "payments",
    "treasury",
    "staking",
    "recognition",
];

/// Commons-side zomes that civic-bridge is allowed to call cross-cluster.
const CIVIC_TO_COMMONS: &[&str] = &[
    // Property domain
    "property_registry",
    "property_transfer",
    "property_disputes",
    "property_commons",
    // Housing domain
    "housing_units",
    "housing_membership",
    "housing_finances",
    "housing_maintenance",
    "housing_clt",
    "housing_governance",
    // Care domain
    "care_timebank",
    "care_circles",
    "care_matching",
    "care_plans",
    "care_credentials",
    // Mutual aid domain
    "mutualaid_needs",
    "mutualaid_circles",
    "mutualaid_governance",
    "mutualaid_pools",
    "mutualaid_requests",
    "mutualaid_resources",
    "mutualaid_timebank",
    // Water domain
    "water_flow",
    "water_purity",
    "water_capture",
    "water_steward",
    "water_wisdom",
    // Food domain
    "food_production",
    "food_distribution",
    "food_preservation",
    "food_knowledge",
    // Transport domain
    "transport_routes",
    "transport_sharing",
    "transport_impact",
    // Commons bridge
    "commons_bridge",
];

/// Identity-side zomes that civic-bridge is allowed to call cross-cluster.
const CIVIC_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry"];

/// Personal-side zomes that hearth-bridge is allowed to call cross-cluster.
const HEARTH_TO_PERSONAL: &[&str] = &["personal_bridge"];

/// Identity-side zomes that hearth-bridge is allowed to call cross-cluster.
const HEARTH_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry", "recovery"];

/// Commons-side zomes that hearth-bridge is allowed to call cross-cluster.
const HEARTH_TO_COMMONS: &[&str] = &["commons_bridge"];

/// Civic-side zomes that hearth-bridge is allowed to call cross-cluster.
const HEARTH_TO_CIVIC: &[&str] = &["civic_bridge"];

/// Commons-side zomes that personal-bridge is allowed to call cross-cluster.
const PERSONAL_TO_COMMONS: &[&str] = &["commons_bridge"];

/// Civic-side zomes that personal-bridge is allowed to call cross-cluster.
const PERSONAL_TO_CIVIC: &[&str] = &["civic_bridge"];

/// Governance-side zomes that personal-bridge is allowed to call cross-cluster.
const PERSONAL_TO_GOVERNANCE: &[&str] = &["governance_bridge"];

/// Identity-side zomes that personal-bridge is allowed to call cross-cluster.
const PERSONAL_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry", "verifiable_credential"];

// --- Group A: Routes already implemented in bridge code but not previously registered ---

/// Governance-side zomes that finance-bridge is allowed to call cross-cluster.
const FINANCE_TO_GOVERNANCE: &[&str] = &["governance_bridge", "proposals", "voting", "councils"];

/// Personal-side zomes that governance-bridge is allowed to call cross-cluster.
const GOVERNANCE_TO_PERSONAL: &[&str] = &["personal_bridge"];

/// Identity-side zomes that governance-bridge is allowed to call cross-cluster.
const GOVERNANCE_TO_IDENTITY: &[&str] =
    &["identity_bridge", "did_registry", "verifiable_credential"];

/// Finance-side zomes that energy-bridge is allowed to call cross-cluster.
const ENERGY_TO_FINANCE: &[&str] = &["finance_bridge", "payments", "staking"];

/// Supplychain-side zomes that energy-bridge is allowed to call cross-cluster.
const ENERGY_TO_SUPPLYCHAIN: &[&str] = &[
    "bridge_coordinator",
    "inventory_coordinator",
    "procurement_coordinator",
];

/// Supplychain-side zomes that climate-bridge is allowed to call cross-cluster.
const CLIMATE_TO_SUPPLYCHAIN: &[&str] = &[
    "bridge_coordinator",
    "claims_coordinator",
    "verification_coordinator",
];

/// Commons-side zomes that manufacturing-bridge is allowed to call cross-cluster.
const MANUFACTURING_TO_COMMONS: &[&str] = &["commons_bridge", "resource_mesh"];

/// Supplychain-side zomes that manufacturing-bridge is allowed to call cross-cluster.
const MANUFACTURING_TO_SUPPLYCHAIN: &[&str] = &[
    "bridge_coordinator",
    "procurement_coordinator",
    "inventory_coordinator",
];

// --- Group B: Routes needed for governance execution loop ---

/// Commons-side zomes that governance-bridge is allowed to call cross-cluster.
const GOVERNANCE_TO_COMMONS: &[&str] = &[
    "commons_bridge",
    "property_registry",
    "housing_governance",
    "water_steward",
];

/// Civic-side zomes that governance-bridge is allowed to call cross-cluster.
const GOVERNANCE_TO_CIVIC: &[&str] = &["civic_bridge", "justice_cases", "emergency_coordination"];

/// Finance-side zomes that governance-bridge is allowed to call cross-cluster.
const GOVERNANCE_TO_FINANCE: &[&str] = &["finance_bridge", "treasury", "payments"];

/// Commons-side zomes that finance-bridge is allowed to call cross-cluster.
const FINANCE_TO_COMMONS: &[&str] = &["commons_bridge", "housing_finances", "mutualaid_pools"];

/// Civic-side zomes that finance-bridge is allowed to call cross-cluster.
const FINANCE_TO_CIVIC: &[&str] = &["civic_bridge", "justice_enforcement"];

/// Finance-side zomes that civic-bridge is allowed to call cross-cluster.
const CIVIC_TO_FINANCE: &[&str] = &["finance_bridge", "payments", "treasury"];

/// Hearth-side zomes that civic-bridge is allowed to call cross-cluster.
const CIVIC_TO_HEARTH: &[&str] = &["hearth_bridge", "hearth_care", "hearth_emergency"];

/// Hearth-side zomes that commons-bridge is allowed to call cross-cluster.
const COMMONS_TO_HEARTH: &[&str] = &["hearth_bridge", "hearth_care", "hearth_resources"];

// --- Group C: Cross-domain integration routes ---

/// Commons-side zomes that identity-bridge is allowed to call cross-cluster.
const IDENTITY_TO_COMMONS: &[&str] = &["commons_bridge"];

/// Civic-side zomes that identity-bridge is allowed to call cross-cluster.
const IDENTITY_TO_CIVIC: &[&str] = &["civic_bridge"];

/// Finance-side zomes that identity-bridge is allowed to call cross-cluster.
const IDENTITY_TO_FINANCE: &[&str] = &["finance_bridge"];

/// Governance-side zomes that identity-bridge is allowed to call cross-cluster.
const IDENTITY_TO_GOVERNANCE: &[&str] = &["governance_bridge"];

/// Personal-side zomes that health-bridge is allowed to call cross-cluster.
const HEALTH_TO_PERSONAL: &[&str] = &["personal_bridge", "health_vault"];

/// Identity-side zomes that health-bridge is allowed to call cross-cluster.
const HEALTH_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry"];

/// Identity-side zomes that praxis-bridge is allowed to call cross-cluster.
const PRAXIS_TO_IDENTITY: &[&str] = &["identity_bridge", "verifiable_credential"];

/// Finance-side zomes that praxis-bridge is allowed to call cross-cluster (TEND learning credits).
const PRAXIS_TO_FINANCE: &[&str] = &["finance_bridge", "tend"];

/// Praxis-side zomes that finance-bridge is allowed to call cross-cluster (payment confirmations).
const FINANCE_TO_PRAXIS: &[&str] = &["praxis_bridge"];

// --- Craft cluster routes ---

/// Craft-side zomes that praxis-bridge is allowed to call cross-cluster.
const PRAXIS_TO_CRAFT: &[&str] = &["craft_graph", "job_postings_coordinator"];

/// Praxis-side zomes that craft can call cross-cluster (credential verification).
const CRAFT_TO_PRAXIS: &[&str] = &["praxis_bridge", "credential_coordinator"];

/// Identity-side zomes that craft can call cross-cluster.
const CRAFT_TO_IDENTITY: &[&str] = &["identity_bridge", "verifiable_credential", "did_registry"];

/// Craft-side zomes that identity can call cross-cluster.
const IDENTITY_TO_CRAFT: &[&str] = &["craft_graph"];

/// Finance-side zomes that craft can call (staked recommendations, SAP escrow).
const CRAFT_TO_FINANCE: &[&str] = &["staking", "finance_bridge"];

/// Craft-side zomes that finance can call (recommendation verification).
const FINANCE_TO_CRAFT: &[&str] = &["connection_graph_coordinator", "craft_bridge"];

/// Finance-side zomes that cafe-bridge is allowed to call cross-cluster.
const CAFE_TO_FINANCE: &[&str] = &["finance_bridge", "payments", "tend", "price_oracle"];

/// Identity-side zomes that cafe-bridge is allowed to call cross-cluster.
const CAFE_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry"];

/// Cafe-side zomes that finance-bridge is allowed to call cross-cluster (payment confirmations).
const FINANCE_TO_CAFE: &[&str] = &["cafe_bridge"];

/// Cafe-side zomes that identity-bridge is allowed to call cross-cluster (credential notifications).
const IDENTITY_TO_CAFE: &[&str] = &["cafe_bridge"];

// --- Group D: Newly registered cluster routes ---

/// Identity-side zomes that mail-bridge is allowed to call cross-cluster (DID resolution, trust).
const MAIL_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry", "trust_credential"];

/// Finance-side zomes that marketplace transactions are allowed to call cross-cluster (payment settlement).
const MARKETPLACE_TO_FINANCE: &[&str] = &["finance_bridge", "payments"];

/// Identity-side zomes that space observations is allowed to call cross-cluster (trust lookups).
const SPACE_TO_IDENTITY: &[&str] = &["identity_bridge", "trust_credential"];

/// Identity-side zomes that attribution is allowed to call cross-cluster (creator verification).
const ATTRIBUTION_TO_IDENTITY: &[&str] = &["identity_bridge", "did_registry"];

/// Finance-side zomes that attribution is allowed to call cross-cluster (reciprocity payments).
const ATTRIBUTION_TO_FINANCE: &[&str] = &["finance_bridge", "payments"];

// --- Group E: Sovereign Profile collector routes ---
// identity-bridge aggregates 8D sovereign profile scores from sibling clusters.
// These routes were active in identity-bridge code but not previously declared here.

/// Knowledge-side zomes that identity-bridge queries for epistemic integrity score.
const IDENTITY_TO_KNOWLEDGE: &[&str] = &["claims"];

/// Attribution-side zomes that identity-bridge queries for stewardship-care score.
const IDENTITY_TO_ATTRIBUTION: &[&str] = &["reciprocity"];

/// Energy-side zomes that identity-bridge queries for thermodynamic-yield score.
const IDENTITY_TO_ENERGY: &[&str] = &["grid"];

/// Core-FL side zomes that identity-bridge queries for FL participation status
/// (domain-competence signal in the 8D sovereign profile).
const IDENTITY_TO_CORE: &[&str] = &["h_fl"];

// --- Group F: Finance outbound routes not previously declared ---

/// Energy-side zomes that finance-bridge is allowed to call for project reconciliation.
const FINANCE_TO_ENERGY: &[&str] = &["energy_bridge"];

/// Hearth-side zomes that finance-payments queries for membership gating.
const FINANCE_TO_HEARTH: &[&str] = &["hearth_bridge"];

/// Identity-side zomes that finance uses for consciousness tier gating
/// (price-oracle + shared participant checks).
const FINANCE_TO_IDENTITY: &[&str] = &["identity_bridge", "consciousness_gating"];

// --- Group G: Climate → Praxis (stake-claim credential verification) ---

/// Praxis-side zomes that climate-bridge queries for credential verification
/// when validating climate-stake educational prerequisites.
const CLIMATE_TO_PRAXIS: &[&str] = &["credential_zome"];

// ============================================================================
// Public API
// ============================================================================

/// Get the hApp role name string for a cluster.
///
/// This is the string used in `CallTargetCell::OtherRole(role_name(...).into())`.
pub const fn role_name(cluster: CrossClusterRole) -> &'static str {
    cluster.as_str()
}

/// Get the list of allowed zomes for a cross-cluster call from `initiator` to `target`.
///
/// Returns an empty slice if there is no registered route between the two clusters,
/// or if `initiator == target` (use the local allowlist for intra-cluster dispatch).
pub const fn get_allowed_zomes(
    initiator: CrossClusterRole,
    target: CrossClusterRole,
) -> &'static [&'static str] {
    match (initiator, target) {
        // Commons outbound
        (CrossClusterRole::Commons, CrossClusterRole::Civic) => COMMONS_TO_CIVIC,
        (CrossClusterRole::Commons, CrossClusterRole::Identity) => COMMONS_TO_IDENTITY,
        (CrossClusterRole::Commons, CrossClusterRole::Finance) => COMMONS_TO_FINANCE,
        (CrossClusterRole::Commons, CrossClusterRole::Hearth) => COMMONS_TO_HEARTH,

        // Civic outbound
        (CrossClusterRole::Civic, CrossClusterRole::Commons) => CIVIC_TO_COMMONS,
        (CrossClusterRole::Civic, CrossClusterRole::Identity) => CIVIC_TO_IDENTITY,
        (CrossClusterRole::Civic, CrossClusterRole::Finance) => CIVIC_TO_FINANCE,
        (CrossClusterRole::Civic, CrossClusterRole::Hearth) => CIVIC_TO_HEARTH,

        // Hearth outbound
        (CrossClusterRole::Hearth, CrossClusterRole::Personal) => HEARTH_TO_PERSONAL,
        (CrossClusterRole::Hearth, CrossClusterRole::Identity) => HEARTH_TO_IDENTITY,
        (CrossClusterRole::Hearth, CrossClusterRole::Commons) => HEARTH_TO_COMMONS,
        (CrossClusterRole::Hearth, CrossClusterRole::Civic) => HEARTH_TO_CIVIC,

        // Personal outbound
        (CrossClusterRole::Personal, CrossClusterRole::Commons) => PERSONAL_TO_COMMONS,
        (CrossClusterRole::Personal, CrossClusterRole::Civic) => PERSONAL_TO_CIVIC,
        (CrossClusterRole::Personal, CrossClusterRole::Governance) => PERSONAL_TO_GOVERNANCE,
        (CrossClusterRole::Personal, CrossClusterRole::Identity) => PERSONAL_TO_IDENTITY,

        // Finance outbound
        (CrossClusterRole::Finance, CrossClusterRole::Governance) => FINANCE_TO_GOVERNANCE,
        (CrossClusterRole::Finance, CrossClusterRole::Commons) => FINANCE_TO_COMMONS,
        (CrossClusterRole::Finance, CrossClusterRole::Civic) => FINANCE_TO_CIVIC,
        (CrossClusterRole::Finance, CrossClusterRole::Energy) => FINANCE_TO_ENERGY,
        (CrossClusterRole::Finance, CrossClusterRole::Hearth) => FINANCE_TO_HEARTH,
        (CrossClusterRole::Finance, CrossClusterRole::Identity) => FINANCE_TO_IDENTITY,

        // Governance outbound
        (CrossClusterRole::Governance, CrossClusterRole::Personal) => GOVERNANCE_TO_PERSONAL,
        (CrossClusterRole::Governance, CrossClusterRole::Identity) => GOVERNANCE_TO_IDENTITY,
        (CrossClusterRole::Governance, CrossClusterRole::Commons) => GOVERNANCE_TO_COMMONS,
        (CrossClusterRole::Governance, CrossClusterRole::Civic) => GOVERNANCE_TO_CIVIC,
        (CrossClusterRole::Governance, CrossClusterRole::Finance) => GOVERNANCE_TO_FINANCE,

        // Identity outbound
        (CrossClusterRole::Identity, CrossClusterRole::Commons) => IDENTITY_TO_COMMONS,
        (CrossClusterRole::Identity, CrossClusterRole::Civic) => IDENTITY_TO_CIVIC,
        (CrossClusterRole::Identity, CrossClusterRole::Finance) => IDENTITY_TO_FINANCE,
        (CrossClusterRole::Identity, CrossClusterRole::Governance) => IDENTITY_TO_GOVERNANCE,
        (CrossClusterRole::Identity, CrossClusterRole::Knowledge) => IDENTITY_TO_KNOWLEDGE,
        (CrossClusterRole::Identity, CrossClusterRole::Attribution) => IDENTITY_TO_ATTRIBUTION,
        (CrossClusterRole::Identity, CrossClusterRole::Energy) => IDENTITY_TO_ENERGY,
        (CrossClusterRole::Identity, CrossClusterRole::Core) => IDENTITY_TO_CORE,

        // Energy outbound
        (CrossClusterRole::Energy, CrossClusterRole::Finance) => ENERGY_TO_FINANCE,
        (CrossClusterRole::Energy, CrossClusterRole::Supplychain) => ENERGY_TO_SUPPLYCHAIN,

        // Climate outbound
        (CrossClusterRole::Climate, CrossClusterRole::Supplychain) => CLIMATE_TO_SUPPLYCHAIN,
        (CrossClusterRole::Climate, CrossClusterRole::Praxis) => CLIMATE_TO_PRAXIS,

        // Manufacturing outbound
        (CrossClusterRole::Manufacturing, CrossClusterRole::Commons) => MANUFACTURING_TO_COMMONS,
        (CrossClusterRole::Manufacturing, CrossClusterRole::Supplychain) => {
            MANUFACTURING_TO_SUPPLYCHAIN
        }

        // Health outbound
        (CrossClusterRole::Health, CrossClusterRole::Personal) => HEALTH_TO_PERSONAL,
        (CrossClusterRole::Health, CrossClusterRole::Identity) => HEALTH_TO_IDENTITY,

        // Praxis outbound
        (CrossClusterRole::Praxis, CrossClusterRole::Identity) => PRAXIS_TO_IDENTITY,
        (CrossClusterRole::Praxis, CrossClusterRole::Finance) => PRAXIS_TO_FINANCE,

        // Inbound to Praxis
        (CrossClusterRole::Finance, CrossClusterRole::Praxis) => FINANCE_TO_PRAXIS,

        // Craft outbound
        (CrossClusterRole::Craft, CrossClusterRole::Praxis) => CRAFT_TO_PRAXIS,
        (CrossClusterRole::Craft, CrossClusterRole::Identity) => CRAFT_TO_IDENTITY,
        (CrossClusterRole::Craft, CrossClusterRole::Finance) => CRAFT_TO_FINANCE,

        // Praxis <-> Craft
        (CrossClusterRole::Praxis, CrossClusterRole::Craft) => PRAXIS_TO_CRAFT,

        // Inbound to Craft
        (CrossClusterRole::Identity, CrossClusterRole::Craft) => IDENTITY_TO_CRAFT,
        (CrossClusterRole::Finance, CrossClusterRole::Craft) => FINANCE_TO_CRAFT,

        // Cafe outbound
        (CrossClusterRole::Cafe, CrossClusterRole::Finance) => CAFE_TO_FINANCE,
        (CrossClusterRole::Cafe, CrossClusterRole::Identity) => CAFE_TO_IDENTITY,

        // Inbound to Cafe
        (CrossClusterRole::Finance, CrossClusterRole::Cafe) => FINANCE_TO_CAFE,
        (CrossClusterRole::Identity, CrossClusterRole::Cafe) => IDENTITY_TO_CAFE,

        // Mail outbound
        (CrossClusterRole::Mail, CrossClusterRole::Identity) => MAIL_TO_IDENTITY,

        // Marketplace outbound
        (CrossClusterRole::Marketplace, CrossClusterRole::Finance) => MARKETPLACE_TO_FINANCE,

        // Space outbound
        (CrossClusterRole::Space, CrossClusterRole::Identity) => SPACE_TO_IDENTITY,

        // Attribution outbound
        (CrossClusterRole::Attribution, CrossClusterRole::Identity) => ATTRIBUTION_TO_IDENTITY,
        (CrossClusterRole::Attribution, CrossClusterRole::Finance) => ATTRIBUTION_TO_FINANCE,

        // No registered route (including self→self)
        _ => &[],
    }
}

/// Check whether a specific zome name is allowed for a cross-cluster call
/// from `initiator` to `target`.
pub fn is_allowed(initiator: CrossClusterRole, target: CrossClusterRole, zome_name: &str) -> bool {
    let allowed = get_allowed_zomes(initiator, target);
    let mut i = 0;
    while i < allowed.len() {
        if str_eq(allowed[i], zome_name) {
            return true;
        }
        i += 1;
    }
    false
}

/// Const-compatible string equality (slice::contains requires non-const traits).
const fn str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Get the local (intra-cluster) allowlist for a given cluster.
///
/// Returns `None` for clusters whose local zomes are not tracked here
/// (e.g., Identity, Finance, Governance have their own internal dispatch).
pub const fn get_local_zomes(cluster: CrossClusterRole) -> Option<&'static [&'static str]> {
    match cluster {
        CrossClusterRole::Commons => Some(COMMONS_LOCAL_ZOMES),
        CrossClusterRole::Civic => Some(CIVIC_LOCAL_ZOMES),
        CrossClusterRole::Hearth => Some(HEARTH_LOCAL_ZOMES),
        CrossClusterRole::Personal => Some(PERSONAL_LOCAL_ZOMES),
        CrossClusterRole::Music => Some(MUSIC_LOCAL_ZOMES),
        CrossClusterRole::Health => Some(HEALTH_LOCAL_ZOMES),
        CrossClusterRole::Energy => Some(ENERGY_LOCAL_ZOMES),
        CrossClusterRole::Knowledge => Some(KNOWLEDGE_LOCAL_ZOMES),
        CrossClusterRole::Climate => Some(CLIMATE_LOCAL_ZOMES),
        CrossClusterRole::Craft => Some(CRAFT_LOCAL_ZOMES),
        CrossClusterRole::Manufacturing => Some(MANUFACTURING_LOCAL_ZOMES),
        CrossClusterRole::Supplychain => Some(SUPPLYCHAIN_LOCAL_ZOMES),
        CrossClusterRole::Praxis => Some(PRAXIS_LOCAL_ZOMES),
        CrossClusterRole::Legacy => Some(LEGACY_LOCAL_ZOMES),
        CrossClusterRole::Identity => Some(IDENTITY_LOCAL_ZOMES),
        CrossClusterRole::Finance => Some(FINANCE_LOCAL_ZOMES),
        CrossClusterRole::Governance => Some(GOVERNANCE_LOCAL_ZOMES),
        CrossClusterRole::Cafe => Some(CAFE_LOCAL_ZOMES),
        CrossClusterRole::Atlas => Some(ATLAS_LOCAL_ZOMES),
        CrossClusterRole::Attribution => Some(ATTRIBUTION_LOCAL_ZOMES),
        CrossClusterRole::Core => Some(CORE_LOCAL_ZOMES),
        CrossClusterRole::Desci => Some(DESCI_LOCAL_ZOMES),
        CrossClusterRole::Mail => Some(MAIL_LOCAL_ZOMES),
        CrossClusterRole::Marketplace => Some(MARKETPLACE_LOCAL_ZOMES),
        CrossClusterRole::Position => Some(POSITION_LOCAL_ZOMES),
        CrossClusterRole::Space => Some(SPACE_LOCAL_ZOMES),
        CrossClusterRole::Lunar => Some(LUNAR_LOCAL_ZOMES),
        CrossClusterRole::MultiworldSim => Some(MULTIWORLD_SIM_LOCAL_ZOMES),
        CrossClusterRole::Portal => Some(PORTAL_LOCAL_ZOMES),
        CrossClusterRole::Workspace => Some(WORKSPACE_LOCAL_ZOMES),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Zome count tests ----

    #[test]
    fn commons_local_zome_count() {
        assert_eq!(COMMONS_LOCAL_ZOMES.len(), 40);
    }

    #[test]
    fn civic_local_zome_count() {
        assert_eq!(CIVIC_LOCAL_ZOMES.len(), 17);
    }

    #[test]
    fn hearth_local_zome_count() {
        assert_eq!(HEARTH_LOCAL_ZOMES.len(), 10);
    }

    #[test]
    fn personal_local_zome_count() {
        assert_eq!(PERSONAL_LOCAL_ZOMES.len(), 3);
    }

    #[test]
    fn commons_land_zome_count() {
        assert_eq!(COMMONS_LAND_ZOMES.len(), 19);
    }

    #[test]
    fn commons_care_zome_count() {
        assert_eq!(COMMONS_CARE_ZOMES.len(), 21);
    }

    #[test]
    fn commons_sub_clusters_partition_local() {
        // LAND + CARE must equal LOCAL (no overlap, no gaps)
        let mut combined: Vec<&&str> = COMMONS_LAND_ZOMES
            .iter()
            .chain(COMMONS_CARE_ZOMES.iter())
            .collect();
        combined.sort();
        let mut local: Vec<&&str> = COMMONS_LOCAL_ZOMES.iter().collect();
        local.sort();
        assert_eq!(combined, local, "LAND + CARE must equal LOCAL");
    }

    // ---- Cross-cluster allowlist counts ----

    #[test]
    fn commons_to_civic_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Civic).len(),
            16,
        );
    }

    #[test]
    fn commons_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Identity).len(),
            2,
        );
    }

    #[test]
    fn commons_to_finance_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Finance).len(),
            6,
        );
    }

    #[test]
    fn civic_to_commons_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Commons).len(),
            35,
        );
    }

    #[test]
    fn civic_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Identity).len(),
            2,
        );
    }

    #[test]
    fn hearth_to_personal_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Hearth, CrossClusterRole::Personal).len(),
            1,
        );
    }

    #[test]
    fn hearth_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Hearth, CrossClusterRole::Identity).len(),
            3,
        );
    }

    #[test]
    fn hearth_to_commons_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Hearth, CrossClusterRole::Commons).len(),
            1,
        );
    }

    #[test]
    fn hearth_to_civic_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Hearth, CrossClusterRole::Civic).len(),
            1,
        );
    }

    #[test]
    fn personal_to_commons_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Personal, CrossClusterRole::Commons).len(),
            1,
        );
    }

    #[test]
    fn personal_to_civic_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Personal, CrossClusterRole::Civic).len(),
            1,
        );
    }

    #[test]
    fn personal_to_governance_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Personal, CrossClusterRole::Governance).len(),
            1,
        );
    }

    #[test]
    fn personal_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Personal, CrossClusterRole::Identity).len(),
            3,
        );
    }

    // ---- is_allowed positive tests ----

    #[test]
    fn is_allowed_commons_to_civic_justice_cases() {
        assert!(is_allowed(
            CrossClusterRole::Commons,
            CrossClusterRole::Civic,
            "justice_cases",
        ));
    }

    #[test]
    fn is_allowed_civic_to_commons_property_registry() {
        assert!(is_allowed(
            CrossClusterRole::Civic,
            CrossClusterRole::Commons,
            "property_registry",
        ));
    }

    #[test]
    fn is_allowed_hearth_to_identity_recovery() {
        assert!(is_allowed(
            CrossClusterRole::Hearth,
            CrossClusterRole::Identity,
            "recovery",
        ));
    }

    #[test]
    fn is_allowed_personal_to_identity_verifiable_credential() {
        assert!(is_allowed(
            CrossClusterRole::Personal,
            CrossClusterRole::Identity,
            "verifiable_credential",
        ));
    }

    #[test]
    fn is_allowed_commons_to_finance_treasury() {
        assert!(is_allowed(
            CrossClusterRole::Commons,
            CrossClusterRole::Finance,
            "treasury",
        ));
    }

    // ---- is_allowed negative tests ----

    #[test]
    fn is_not_allowed_nonexistent_pair() {
        assert!(!is_allowed(
            CrossClusterRole::Finance,
            CrossClusterRole::Hearth,
            "anything",
        ));
    }

    #[test]
    fn is_not_allowed_self_dispatch() {
        assert!(!is_allowed(
            CrossClusterRole::Commons,
            CrossClusterRole::Commons,
            "property_registry",
        ));
    }

    #[test]
    fn is_not_allowed_wrong_zome() {
        // commons can reach civic, but not a fake zome
        assert!(!is_allowed(
            CrossClusterRole::Commons,
            CrossClusterRole::Civic,
            "nonexistent_zome",
        ));
    }

    #[test]
    fn is_allowed_civic_to_finance() {
        // Civic→Finance route is now registered
        assert!(is_allowed(
            CrossClusterRole::Civic,
            CrossClusterRole::Finance,
            "finance_bridge",
        ));
    }

    #[test]
    fn is_not_allowed_hearth_to_governance() {
        // Hearth has no route to governance
        assert!(!is_allowed(
            CrossClusterRole::Hearth,
            CrossClusterRole::Governance,
            "governance_bridge",
        ));
    }

    // ---- role_name tests ----

    #[test]
    fn role_name_returns_correct_strings() {
        assert_eq!(role_name(CrossClusterRole::Commons), "commons");
        assert_eq!(role_name(CrossClusterRole::Civic), "civic");
        assert_eq!(role_name(CrossClusterRole::Identity), "identity");
        assert_eq!(role_name(CrossClusterRole::Hearth), "hearth");
        assert_eq!(role_name(CrossClusterRole::Personal), "personal");
        assert_eq!(role_name(CrossClusterRole::Music), "music");
        assert_eq!(role_name(CrossClusterRole::Finance), "finance");
        assert_eq!(role_name(CrossClusterRole::Governance), "governance");
        assert_eq!(role_name(CrossClusterRole::Health), "health");
        assert_eq!(role_name(CrossClusterRole::Energy), "energy");
        assert_eq!(role_name(CrossClusterRole::Knowledge), "knowledge");
        assert_eq!(role_name(CrossClusterRole::Climate), "climate");
        assert_eq!(role_name(CrossClusterRole::Manufacturing), "manufacturing");
        assert_eq!(role_name(CrossClusterRole::Supplychain), "supplychain");
        assert_eq!(role_name(CrossClusterRole::Praxis), "praxis");
        assert_eq!(role_name(CrossClusterRole::Legacy), "legacy");
        assert_eq!(role_name(CrossClusterRole::Atlas), "atlas");
        assert_eq!(role_name(CrossClusterRole::Attribution), "attribution");
        assert_eq!(role_name(CrossClusterRole::Core), "core_fl");
        assert_eq!(role_name(CrossClusterRole::Desci), "desci");
        assert_eq!(role_name(CrossClusterRole::Mail), "mail");
        assert_eq!(role_name(CrossClusterRole::Marketplace), "marketplace");
        assert_eq!(role_name(CrossClusterRole::Position), "position");
        assert_eq!(role_name(CrossClusterRole::Space), "space");
        assert_eq!(role_name(CrossClusterRole::MultiworldSim), "multiworld_sim");
        assert_eq!(role_name(CrossClusterRole::Portal), "portal");
        assert_eq!(role_name(CrossClusterRole::Workspace), "workspace");
        assert_eq!(role_name(CrossClusterRole::Lunar), "lunar");
    }

    // ---- No duplicates in any allowlist ----

    #[test]
    fn no_duplicates_in_any_allowlist() {
        let lists: &[(&str, &[&str])] = &[
            // Local allowlists
            ("COMMONS_LOCAL_ZOMES", COMMONS_LOCAL_ZOMES),
            ("CIVIC_LOCAL_ZOMES", CIVIC_LOCAL_ZOMES),
            ("HEARTH_LOCAL_ZOMES", HEARTH_LOCAL_ZOMES),
            ("PERSONAL_LOCAL_ZOMES", PERSONAL_LOCAL_ZOMES),
            ("COMMONS_LAND_ZOMES", COMMONS_LAND_ZOMES),
            ("COMMONS_CARE_ZOMES", COMMONS_CARE_ZOMES),
            ("HEALTH_LOCAL_ZOMES", HEALTH_LOCAL_ZOMES),
            ("ENERGY_LOCAL_ZOMES", ENERGY_LOCAL_ZOMES),
            ("KNOWLEDGE_LOCAL_ZOMES", KNOWLEDGE_LOCAL_ZOMES),
            ("CLIMATE_LOCAL_ZOMES", CLIMATE_LOCAL_ZOMES),
            ("CRAFT_LOCAL_ZOMES", CRAFT_LOCAL_ZOMES),
            ("MANUFACTURING_LOCAL_ZOMES", MANUFACTURING_LOCAL_ZOMES),
            ("SUPPLYCHAIN_LOCAL_ZOMES", SUPPLYCHAIN_LOCAL_ZOMES),
            ("PRAXIS_LOCAL_ZOMES", PRAXIS_LOCAL_ZOMES),
            ("LEGACY_LOCAL_ZOMES", LEGACY_LOCAL_ZOMES),
            ("IDENTITY_LOCAL_ZOMES", IDENTITY_LOCAL_ZOMES),
            ("FINANCE_LOCAL_ZOMES", FINANCE_LOCAL_ZOMES),
            ("GOVERNANCE_LOCAL_ZOMES", GOVERNANCE_LOCAL_ZOMES),
            // Original cross-cluster routes
            ("COMMONS_TO_CIVIC", COMMONS_TO_CIVIC),
            ("COMMONS_TO_IDENTITY", COMMONS_TO_IDENTITY),
            ("COMMONS_TO_FINANCE", COMMONS_TO_FINANCE),
            ("COMMONS_TO_HEARTH", COMMONS_TO_HEARTH),
            ("CIVIC_TO_COMMONS", CIVIC_TO_COMMONS),
            ("CIVIC_TO_IDENTITY", CIVIC_TO_IDENTITY),
            ("CIVIC_TO_FINANCE", CIVIC_TO_FINANCE),
            ("CIVIC_TO_HEARTH", CIVIC_TO_HEARTH),
            ("HEARTH_TO_PERSONAL", HEARTH_TO_PERSONAL),
            ("HEARTH_TO_IDENTITY", HEARTH_TO_IDENTITY),
            ("HEARTH_TO_COMMONS", HEARTH_TO_COMMONS),
            ("HEARTH_TO_CIVIC", HEARTH_TO_CIVIC),
            ("PERSONAL_TO_COMMONS", PERSONAL_TO_COMMONS),
            ("PERSONAL_TO_CIVIC", PERSONAL_TO_CIVIC),
            ("PERSONAL_TO_GOVERNANCE", PERSONAL_TO_GOVERNANCE),
            ("PERSONAL_TO_IDENTITY", PERSONAL_TO_IDENTITY),
            // Group A: Already-implemented routes
            ("FINANCE_TO_GOVERNANCE", FINANCE_TO_GOVERNANCE),
            ("GOVERNANCE_TO_PERSONAL", GOVERNANCE_TO_PERSONAL),
            ("GOVERNANCE_TO_IDENTITY", GOVERNANCE_TO_IDENTITY),
            ("ENERGY_TO_FINANCE", ENERGY_TO_FINANCE),
            ("ENERGY_TO_SUPPLYCHAIN", ENERGY_TO_SUPPLYCHAIN),
            ("CLIMATE_TO_SUPPLYCHAIN", CLIMATE_TO_SUPPLYCHAIN),
            ("MANUFACTURING_TO_COMMONS", MANUFACTURING_TO_COMMONS),
            ("MANUFACTURING_TO_SUPPLYCHAIN", MANUFACTURING_TO_SUPPLYCHAIN),
            // Group B: Governance execution loop
            ("GOVERNANCE_TO_COMMONS", GOVERNANCE_TO_COMMONS),
            ("GOVERNANCE_TO_CIVIC", GOVERNANCE_TO_CIVIC),
            ("GOVERNANCE_TO_FINANCE", GOVERNANCE_TO_FINANCE),
            ("FINANCE_TO_COMMONS", FINANCE_TO_COMMONS),
            ("FINANCE_TO_CIVIC", FINANCE_TO_CIVIC),
            ("CIVIC_TO_FINANCE", CIVIC_TO_FINANCE),
            ("CIVIC_TO_HEARTH", CIVIC_TO_HEARTH),
            // Group C: Cross-domain integration
            ("IDENTITY_TO_COMMONS", IDENTITY_TO_COMMONS),
            ("IDENTITY_TO_CIVIC", IDENTITY_TO_CIVIC),
            ("IDENTITY_TO_FINANCE", IDENTITY_TO_FINANCE),
            ("IDENTITY_TO_GOVERNANCE", IDENTITY_TO_GOVERNANCE),
            ("HEALTH_TO_PERSONAL", HEALTH_TO_PERSONAL),
            ("HEALTH_TO_IDENTITY", HEALTH_TO_IDENTITY),
            ("PRAXIS_TO_IDENTITY", PRAXIS_TO_IDENTITY),
            ("PRAXIS_TO_CRAFT", PRAXIS_TO_CRAFT),
            ("CRAFT_TO_PRAXIS", CRAFT_TO_PRAXIS),
            ("CRAFT_TO_FINANCE", CRAFT_TO_FINANCE),
            ("FINANCE_TO_CRAFT", FINANCE_TO_CRAFT),
            ("CRAFT_TO_IDENTITY", CRAFT_TO_IDENTITY),
            ("IDENTITY_TO_CRAFT", IDENTITY_TO_CRAFT),
            ("CAFE_LOCAL_ZOMES", CAFE_LOCAL_ZOMES),
            ("CAFE_TO_FINANCE", CAFE_TO_FINANCE),
            ("CAFE_TO_IDENTITY", CAFE_TO_IDENTITY),
            ("FINANCE_TO_CAFE", FINANCE_TO_CAFE),
            ("IDENTITY_TO_CAFE", IDENTITY_TO_CAFE),
            // Group D: Newly registered cluster routes
            ("ATLAS_LOCAL_ZOMES", ATLAS_LOCAL_ZOMES),
            ("ATTRIBUTION_LOCAL_ZOMES", ATTRIBUTION_LOCAL_ZOMES),
            ("CORE_LOCAL_ZOMES", CORE_LOCAL_ZOMES),
            ("DESCI_LOCAL_ZOMES", DESCI_LOCAL_ZOMES),
            ("MAIL_LOCAL_ZOMES", MAIL_LOCAL_ZOMES),
            ("MARKETPLACE_LOCAL_ZOMES", MARKETPLACE_LOCAL_ZOMES),
            ("POSITION_LOCAL_ZOMES", POSITION_LOCAL_ZOMES),
            ("SPACE_LOCAL_ZOMES", SPACE_LOCAL_ZOMES),
            ("MAIL_TO_IDENTITY", MAIL_TO_IDENTITY),
            ("MARKETPLACE_TO_FINANCE", MARKETPLACE_TO_FINANCE),
            ("SPACE_TO_IDENTITY", SPACE_TO_IDENTITY),
            ("ATTRIBUTION_TO_IDENTITY", ATTRIBUTION_TO_IDENTITY),
            ("ATTRIBUTION_TO_FINANCE", ATTRIBUTION_TO_FINANCE),
        ];
        for (name, list) in lists {
            let mut seen = std::collections::HashSet::new();
            for zome in *list {
                assert!(seen.insert(zome), "Duplicate in {}: '{}'", name, zome);
            }
        }
    }

    // ---- No empty strings in any allowlist ----

    #[test]
    fn no_empty_entries_in_any_allowlist() {
        // Use get_local_zomes to cover all clusters exhaustively
        let lists: Vec<(&str, &[&str])> = CrossClusterRole::ALL
            .iter()
            .filter_map(|c| get_local_zomes(*c).map(|z| (c.as_str(), z)))
            .collect();
        let lists: &[(&str, &[&str])] = &lists;
        for (name, list) in lists {
            for zome in *list {
                assert!(!zome.is_empty(), "{} contains an empty string", name);
                assert!(
                    !zome.contains(char::is_whitespace),
                    "{} entry '{}' contains whitespace",
                    name,
                    zome,
                );
            }
        }
    }

    // ---- get_local_zomes coverage ----

    #[test]
    fn get_local_zomes_returns_expected() {
        // All 29 clusters now have local zome lists
        assert_eq!(
            get_local_zomes(CrossClusterRole::Commons).unwrap().len(),
            40
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Civic).unwrap().len(), 17);
        assert_eq!(get_local_zomes(CrossClusterRole::Hearth).unwrap().len(), 10);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Personal).unwrap().len(),
            3
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Music).unwrap().len(), 4);
        assert_eq!(get_local_zomes(CrossClusterRole::Health).unwrap().len(), 16);
        assert_eq!(get_local_zomes(CrossClusterRole::Energy).unwrap().len(), 5);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Knowledge).unwrap().len(),
            8
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Climate).unwrap().len(), 3);
        assert_eq!(get_local_zomes(CrossClusterRole::Craft).unwrap().len(), 6);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Manufacturing)
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            get_local_zomes(CrossClusterRole::Supplychain)
                .unwrap()
                .len(),
            8
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Praxis).unwrap().len(), 12);
        assert_eq!(get_local_zomes(CrossClusterRole::Legacy).unwrap().len(), 5);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Identity).unwrap().len(),
            12
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Finance).unwrap().len(), 8);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Governance).unwrap().len(),
            8
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Cafe).unwrap().len(), 4);
        assert_eq!(get_local_zomes(CrossClusterRole::Atlas).unwrap().len(), 3);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Attribution)
                .unwrap()
                .len(),
            3
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Core).unwrap().len(), 6);
        assert_eq!(get_local_zomes(CrossClusterRole::Desci).unwrap().len(), 4);
        assert_eq!(get_local_zomes(CrossClusterRole::Mail).unwrap().len(), 13);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Marketplace)
                .unwrap()
                .len(),
            9
        );
        assert_eq!(
            get_local_zomes(CrossClusterRole::Position).unwrap().len(),
            3
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Space).unwrap().len(), 5);
        assert_eq!(get_local_zomes(CrossClusterRole::Lunar).unwrap().len(), 0);
        assert_eq!(
            get_local_zomes(CrossClusterRole::MultiworldSim)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(get_local_zomes(CrossClusterRole::Portal).unwrap().len(), 0);
        assert_eq!(
            get_local_zomes(CrossClusterRole::Workspace).unwrap().len(),
            0
        );
    }

    #[test]
    fn all_clusters_have_local_zomes() {
        for cluster in CrossClusterRole::ALL {
            assert!(
                get_local_zomes(*cluster).is_some(),
                "Cluster {:?} has no local zomes registered",
                cluster,
            );
        }
    }

    // ---- Empty result for unregistered routes ----

    #[test]
    fn unregistered_routes_return_empty() {
        // Spot-check routes that are intentionally NOT registered
        assert!(get_allowed_zomes(CrossClusterRole::Legacy, CrossClusterRole::Music).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Music, CrossClusterRole::Legacy).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Climate, CrossClusterRole::Hearth).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Lunar, CrossClusterRole::Music).is_empty());
        assert!(
            get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Personal).is_empty()
        );
        // Finance → Hearth is now registered (Group F, 2026-04-18); use a still-unregistered pair.
        assert!(get_allowed_zomes(CrossClusterRole::Hearth, CrossClusterRole::Music).is_empty());
    }

    // ---- New route count tests ----

    #[test]
    fn finance_to_governance_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Finance, CrossClusterRole::Governance).len(),
            4
        );
    }

    #[test]
    fn governance_to_commons_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Commons).len(),
            4
        );
    }

    #[test]
    fn governance_to_finance_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Finance).len(),
            3
        );
    }

    #[test]
    fn identity_outbound_counts() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Commons).len(),
            1
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Civic).len(),
            1
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Finance).len(),
            1
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Governance).len(),
            1
        );
    }

    #[test]
    fn health_outbound_counts() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Health, CrossClusterRole::Personal).len(),
            2
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Health, CrossClusterRole::Identity).len(),
            2
        );
    }

    #[test]
    fn energy_outbound_counts() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Energy, CrossClusterRole::Finance).len(),
            3
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Energy, CrossClusterRole::Supplychain).len(),
            3
        );
    }

    #[test]
    fn manufacturing_outbound_counts() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Manufacturing, CrossClusterRole::Commons).len(),
            2
        );
        assert_eq!(
            get_allowed_zomes(
                CrossClusterRole::Manufacturing,
                CrossClusterRole::Supplychain
            )
            .len(),
            3
        );
    }

    #[test]
    fn praxis_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Praxis, CrossClusterRole::Identity).len(),
            2
        );
    }

    // ---- is_allowed positive tests for new routes ----

    #[test]
    fn is_allowed_finance_to_governance_proposals() {
        assert!(is_allowed(
            CrossClusterRole::Finance,
            CrossClusterRole::Governance,
            "proposals"
        ));
    }

    #[test]
    fn is_allowed_governance_to_commons_property_registry() {
        assert!(is_allowed(
            CrossClusterRole::Governance,
            CrossClusterRole::Commons,
            "property_registry"
        ));
    }

    #[test]
    fn is_allowed_identity_to_commons_bridge() {
        assert!(is_allowed(
            CrossClusterRole::Identity,
            CrossClusterRole::Commons,
            "commons_bridge"
        ));
    }

    #[test]
    fn is_allowed_health_to_personal_health_vault() {
        assert!(is_allowed(
            CrossClusterRole::Health,
            CrossClusterRole::Personal,
            "health_vault"
        ));
    }

    #[test]
    fn is_allowed_praxis_to_identity_verifiable_credential() {
        assert!(is_allowed(
            CrossClusterRole::Praxis,
            CrossClusterRole::Identity,
            "verifiable_credential"
        ));
    }

    // ---- New cluster route count tests ----

    #[test]
    fn mail_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Mail, CrossClusterRole::Identity).len(),
            3
        );
    }

    #[test]
    fn marketplace_to_finance_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Marketplace, CrossClusterRole::Finance).len(),
            2
        );
    }

    #[test]
    fn space_to_identity_count() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Space, CrossClusterRole::Identity).len(),
            2
        );
    }

    #[test]
    fn attribution_outbound_counts() {
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Attribution, CrossClusterRole::Identity).len(),
            2
        );
        assert_eq!(
            get_allowed_zomes(CrossClusterRole::Attribution, CrossClusterRole::Finance).len(),
            2
        );
    }

    // ---- is_allowed positive tests for new cluster routes ----

    #[test]
    fn is_allowed_mail_to_identity_did_registry() {
        assert!(is_allowed(
            CrossClusterRole::Mail,
            CrossClusterRole::Identity,
            "did_registry"
        ));
    }

    #[test]
    fn is_allowed_marketplace_to_finance_payments() {
        assert!(is_allowed(
            CrossClusterRole::Marketplace,
            CrossClusterRole::Finance,
            "payments"
        ));
    }

    #[test]
    fn is_allowed_space_to_identity_trust_credential() {
        assert!(is_allowed(
            CrossClusterRole::Space,
            CrossClusterRole::Identity,
            "trust_credential"
        ));
    }

    #[test]
    fn is_allowed_attribution_to_identity_did_registry() {
        assert!(is_allowed(
            CrossClusterRole::Attribution,
            CrossClusterRole::Identity,
            "did_registry"
        ));
    }

    // ---- Total route count ----

    #[test]
    fn total_registered_route_count() {
        let mut count = 0;
        for initiator in CrossClusterRole::ALL {
            for target in CrossClusterRole::ALL {
                if !get_allowed_zomes(*initiator, *target).is_empty() {
                    count += 1;
                }
            }
        }
        // 60 registered directional routes:
        //   36 original + 4 Cafe + 5 new (Mail→Identity, Marketplace→Finance,
        //   Space→Identity, Attribution→Identity, Attribution→Finance)
        //   + 2 additional routes added for Health↔Identity and Praxis
        //   + 4 Craft routes (Craft↔Praxis, Craft→Identity, Identity→Craft)
        //   + 8 code-vs-registry gap closes (2026-04-18, Groups E/F/G):
        //     Identity→{Knowledge, Attribution, Energy, Core},
        //     Finance→{Energy, Hearth, Identity},
        //     Climate→Praxis
        assert_eq!(count, 61, "Expected 61 registered cross-cluster routes");
    }
}
