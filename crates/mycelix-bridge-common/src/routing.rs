//! Type-safe bridge dispatch routing.
//!
//! Replaces the fragile `query_type.contains("transfer")` substring matching
//! in each bridge's `resolve_domain_zome()` with strongly-typed enums and
//! case-insensitive routing.
//!
//! ## Enums
//!
//! - [`BridgeDomain`] — 12 domain variants (property, housing, … media)
//! - [`CommonsZome`] — 38 zome variants in the commons cluster
//! - [`CivicZome`] — 15 zome variants in the civic cluster
//! - [`CrossClusterRole`] — commons / civic / identity hApp roles
//!
//! ## Routing Functions
//!
//! - [`resolve_commons_zome`] — domain + query_type → `CommonsZome`
//! - [`resolve_civic_zome`] — domain + query_type → `CivicZome`

use serde::{Deserialize, Serialize};

// ============================================================================
// Domain enum
// ============================================================================

/// All 12 Mycelix domains across both cluster DNAs.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BridgeDomain {
    Property,
    Housing,
    Care,
    Mutualaid,
    Water,
    Food,
    Transport,
    Support,
    Space,
    Justice,
    Emergency,
    Media,
}

/// Domains that belong to the Commons cluster.
pub const COMMONS_DOMAINS: &[BridgeDomain] = &[
    BridgeDomain::Property,
    BridgeDomain::Housing,
    BridgeDomain::Care,
    BridgeDomain::Mutualaid,
    BridgeDomain::Water,
    BridgeDomain::Food,
    BridgeDomain::Transport,
    BridgeDomain::Support,
    BridgeDomain::Space,
];

/// Domains that belong to the Civic cluster.
pub const CIVIC_DOMAINS: &[BridgeDomain] = &[
    BridgeDomain::Justice,
    BridgeDomain::Emergency,
    BridgeDomain::Media,
];

impl BridgeDomain {
    /// Parse a domain string case-insensitively.
    ///
    /// Fixes the current bug where `"Property"` or `"HOUSING"` would fail
    /// the exact-match `domain == "property"` check.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "property" => Some(Self::Property),
            "housing" => Some(Self::Housing),
            "care" => Some(Self::Care),
            "mutualaid" | "mutual_aid" | "mutual-aid" => Some(Self::Mutualaid),
            "water" => Some(Self::Water),
            "food" => Some(Self::Food),
            "transport" => Some(Self::Transport),
            "support" => Some(Self::Support),
            "space" => Some(Self::Space),
            "justice" => Some(Self::Justice),
            "emergency" => Some(Self::Emergency),
            "media" => Some(Self::Media),
            _ => None,
        }
    }

    /// Canonical lowercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Property => "property",
            Self::Housing => "housing",
            Self::Care => "care",
            Self::Mutualaid => "mutualaid",
            Self::Water => "water",
            Self::Food => "food",
            Self::Transport => "transport",
            Self::Support => "support",
            Self::Space => "space",
            Self::Justice => "justice",
            Self::Emergency => "emergency",
            Self::Media => "media",
        }
    }

    /// Whether this domain belongs to the Commons cluster.
    pub fn is_commons(&self) -> bool {
        COMMONS_DOMAINS.contains(self)
    }

    /// Whether this domain belongs to the Civic cluster.
    pub fn is_civic(&self) -> bool {
        CIVIC_DOMAINS.contains(self)
    }
}

// ============================================================================
// Commons zome enum (38 variants)
// ============================================================================

/// All 38 coordinator zomes in the commons cluster.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommonsZome {
    // Property domain (4)
    PropertyRegistry,
    PropertyTransfer,
    PropertyDisputes,
    PropertyCommons,
    // Housing domain (6)
    HousingUnits,
    HousingMembership,
    HousingFinances,
    HousingMaintenance,
    HousingClt,
    HousingGovernance,
    // Care domain (5)
    CareTimebank,
    CareCircles,
    CareMatching,
    CarePlans,
    CareCredentials,
    // Mutual aid domain (7)
    MutualaidNeeds,
    MutualaidCircles,
    MutualaidGovernance,
    MutualaidPools,
    MutualaidRequests,
    MutualaidResources,
    MutualaidTimebank,
    // Water domain (5)
    WaterFlow,
    WaterPurity,
    WaterCapture,
    WaterSteward,
    WaterWisdom,
    // Food domain (4)
    FoodProduction,
    FoodDistribution,
    FoodPreservation,
    FoodKnowledge,
    // Transport domain (3)
    TransportRoutes,
    TransportSharing,
    TransportImpact,
    // Support domain (3)
    SupportKnowledge,
    SupportTickets,
    SupportDiagnostics,
    // Space domain (1)
    Space,
}

impl CommonsZome {
    /// Snake_case string matching the actual Holochain zome name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PropertyRegistry => "property_registry",
            Self::PropertyTransfer => "property_transfer",
            Self::PropertyDisputes => "property_disputes",
            Self::PropertyCommons => "property_commons",
            Self::HousingUnits => "housing_units",
            Self::HousingMembership => "housing_membership",
            Self::HousingFinances => "housing_finances",
            Self::HousingMaintenance => "housing_maintenance",
            Self::HousingClt => "housing_clt",
            Self::HousingGovernance => "housing_governance",
            Self::CareTimebank => "care_timebank",
            Self::CareCircles => "care_circles",
            Self::CareMatching => "care_matching",
            Self::CarePlans => "care_plans",
            Self::CareCredentials => "care_credentials",
            Self::MutualaidNeeds => "mutualaid_needs",
            Self::MutualaidCircles => "mutualaid_circles",
            Self::MutualaidGovernance => "mutualaid_governance",
            Self::MutualaidPools => "mutualaid_pools",
            Self::MutualaidRequests => "mutualaid_requests",
            Self::MutualaidResources => "mutualaid_resources",
            Self::MutualaidTimebank => "mutualaid_timebank",
            Self::WaterFlow => "water_flow",
            Self::WaterPurity => "water_purity",
            Self::WaterCapture => "water_capture",
            Self::WaterSteward => "water_steward",
            Self::WaterWisdom => "water_wisdom",
            Self::FoodProduction => "food_production",
            Self::FoodDistribution => "food_distribution",
            Self::FoodPreservation => "food_preservation",
            Self::FoodKnowledge => "food_knowledge",
            Self::TransportRoutes => "transport_routes",
            Self::TransportSharing => "transport_sharing",
            Self::TransportImpact => "transport_impact",
            Self::SupportKnowledge => "support_knowledge",
            Self::SupportTickets => "support_tickets",
            Self::SupportDiagnostics => "support_diagnostics",
            Self::Space => "space",
        }
    }

    /// Whether this zome belongs to the commons_land sub-cluster DNA.
    pub fn is_land(&self) -> bool {
        matches!(
            self,
            Self::PropertyRegistry
                | Self::PropertyTransfer
                | Self::PropertyDisputes
                | Self::PropertyCommons
                | Self::HousingUnits
                | Self::HousingMembership
                | Self::HousingFinances
                | Self::HousingMaintenance
                | Self::HousingClt
                | Self::HousingGovernance
                | Self::WaterFlow
                | Self::WaterPurity
                | Self::WaterCapture
                | Self::WaterSteward
                | Self::WaterWisdom
                | Self::FoodProduction
                | Self::FoodDistribution
                | Self::FoodPreservation
                | Self::FoodKnowledge
        )
    }

    /// Default zome for a commons domain when no query_type routing matches.
    pub fn default_for_domain(domain: BridgeDomain) -> Option<Self> {
        match domain {
            BridgeDomain::Property => Some(Self::PropertyRegistry),
            BridgeDomain::Housing => Some(Self::HousingUnits),
            BridgeDomain::Care => Some(Self::CareTimebank),
            BridgeDomain::Mutualaid => Some(Self::MutualaidTimebank),
            BridgeDomain::Water => Some(Self::WaterFlow),
            BridgeDomain::Food => Some(Self::FoodProduction),
            BridgeDomain::Transport => Some(Self::TransportRoutes),
            BridgeDomain::Support => Some(Self::SupportKnowledge),
            BridgeDomain::Space => Some(Self::Space),
            // Civic domains have no commons zome
            _ => None,
        }
    }

    /// All 38 variants, for consistency checks.
    pub const ALL: &'static [CommonsZome] = &[
        Self::PropertyRegistry,
        Self::PropertyTransfer,
        Self::PropertyDisputes,
        Self::PropertyCommons,
        Self::HousingUnits,
        Self::HousingMembership,
        Self::HousingFinances,
        Self::HousingMaintenance,
        Self::HousingClt,
        Self::HousingGovernance,
        Self::CareTimebank,
        Self::CareCircles,
        Self::CareMatching,
        Self::CarePlans,
        Self::CareCredentials,
        Self::MutualaidNeeds,
        Self::MutualaidCircles,
        Self::MutualaidGovernance,
        Self::MutualaidPools,
        Self::MutualaidRequests,
        Self::MutualaidResources,
        Self::MutualaidTimebank,
        Self::WaterFlow,
        Self::WaterPurity,
        Self::WaterCapture,
        Self::WaterSteward,
        Self::WaterWisdom,
        Self::FoodProduction,
        Self::FoodDistribution,
        Self::FoodPreservation,
        Self::FoodKnowledge,
        Self::TransportRoutes,
        Self::TransportSharing,
        Self::TransportImpact,
        Self::SupportKnowledge,
        Self::SupportTickets,
        Self::SupportDiagnostics,
        Self::Space,
    ];

    /// Resolve a commons zome name string to the correct sub-cluster hApp role.
    ///
    /// Returns `"commons_land"` for physical infrastructure zomes (property,
    /// housing, water, food) and `"commons_care"` for social/care zomes
    /// (care, mutualaid, transport, support, space).
    ///
    /// Returns `None` if the zome name is not a recognized commons zome.
    /// Also handles the `commons_bridge` zome which exists in both sub-clusters.
    pub fn resolve_role(zome_name: &str) -> Option<&'static str> {
        // Try to match against known zome names
        if let Some(zome) = Self::from_str(zome_name) {
            if zome.is_land() {
                Some("commons_land")
            } else {
                Some("commons_care")
            }
        } else if zome_name == "commons_bridge" {
            // Bridge exists in both; default to land (caller can override)
            Some("commons_land")
        } else {
            None
        }
    }

    /// Parse a zome name string into a CommonsZome variant.
    pub fn from_str(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|z| z.as_str() == s)
    }
}

// ============================================================================
// Civic zome enum (15 variants)
// ============================================================================

/// All 15 coordinator zomes in the civic cluster.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CivicZome {
    // Justice domain (5)
    JusticeCases,
    JusticeEvidence,
    JusticeArbitration,
    JusticeRestorative,
    JusticeEnforcement,
    // Emergency domain (6)
    EmergencyIncidents,
    EmergencyTriage,
    EmergencyResources,
    EmergencyCoordination,
    EmergencyShelters,
    EmergencyComms,
    // Media domain (4)
    MediaPublication,
    MediaAttribution,
    MediaFactcheck,
    MediaCuration,
}

impl CivicZome {
    /// Snake_case string matching the actual Holochain zome name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::JusticeCases => "justice_cases",
            Self::JusticeEvidence => "justice_evidence",
            Self::JusticeArbitration => "justice_arbitration",
            Self::JusticeRestorative => "justice_restorative",
            Self::JusticeEnforcement => "justice_enforcement",
            Self::EmergencyIncidents => "emergency_incidents",
            Self::EmergencyTriage => "emergency_triage",
            Self::EmergencyResources => "emergency_resources",
            Self::EmergencyCoordination => "emergency_coordination",
            Self::EmergencyShelters => "emergency_shelters",
            Self::EmergencyComms => "emergency_comms",
            Self::MediaPublication => "media_publication",
            Self::MediaAttribution => "media_attribution",
            Self::MediaFactcheck => "media_factcheck",
            Self::MediaCuration => "media_curation",
        }
    }

    /// Default zome for a civic domain when no query_type routing matches.
    pub fn default_for_domain(domain: BridgeDomain) -> Option<Self> {
        match domain {
            BridgeDomain::Justice => Some(Self::JusticeCases),
            BridgeDomain::Emergency => Some(Self::EmergencyIncidents),
            BridgeDomain::Media => Some(Self::MediaPublication),
            // Commons domains have no civic zome
            _ => None,
        }
    }

    /// All 15 variants, for consistency checks.
    pub const ALL: &'static [CivicZome] = &[
        Self::JusticeCases,
        Self::JusticeEvidence,
        Self::JusticeArbitration,
        Self::JusticeRestorative,
        Self::JusticeEnforcement,
        Self::EmergencyIncidents,
        Self::EmergencyTriage,
        Self::EmergencyResources,
        Self::EmergencyCoordination,
        Self::EmergencyShelters,
        Self::EmergencyComms,
        Self::MediaPublication,
        Self::MediaAttribution,
        Self::MediaFactcheck,
        Self::MediaCuration,
    ];
}

// ============================================================================
// Cross-cluster role
// ============================================================================

/// hApp role names for cross-cluster dispatch via `CallTargetCell::OtherRole`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CrossClusterRole {
    Commons,
    Civic,
    Identity,
}

impl CrossClusterRole {
    /// String matching the hApp role name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Commons => "commons",
            Self::Civic => "civic",
            Self::Identity => "identity",
        }
    }
}

// ============================================================================
// Routing: Commons
// ============================================================================

/// Resolve a commons domain + query_type to a specific zome.
///
/// This is the type-safe replacement for the `resolve_domain_zome()` function
/// in `commons-bridge/coordinator/src/lib.rs` (lines 368-427).
///
/// Query type matching is case-insensitive (lowercased before comparison).
/// Falls back to the domain's default zome if no keyword matches.
pub fn resolve_commons_zome(domain: BridgeDomain, query_type: &str) -> Option<CommonsZome> {
    let qt = query_type.to_ascii_lowercase();
    match domain {
        BridgeDomain::Property => Some(resolve_property(&qt)),
        BridgeDomain::Housing => Some(resolve_housing(&qt)),
        BridgeDomain::Care => Some(resolve_care(&qt)),
        BridgeDomain::Mutualaid => Some(resolve_mutualaid(&qt)),
        BridgeDomain::Water => Some(resolve_water(&qt)),
        BridgeDomain::Food => Some(resolve_food(&qt)),
        BridgeDomain::Transport => Some(resolve_transport(&qt)),
        BridgeDomain::Support => Some(resolve_support(&qt)),
        BridgeDomain::Space => Some(CommonsZome::Space),
        // Civic domains → None
        BridgeDomain::Justice | BridgeDomain::Emergency | BridgeDomain::Media => None,
    }
}

fn resolve_property(qt: &str) -> CommonsZome {
    if qt.contains("transfer") || qt.contains("ownership") {
        CommonsZome::PropertyTransfer
    } else if qt.contains("dispute") {
        CommonsZome::PropertyDisputes
    } else if qt.contains("encumbrance") || qt.contains("title") {
        CommonsZome::PropertyRegistry
    } else {
        CommonsZome::PropertyRegistry
    }
}

fn resolve_housing(qt: &str) -> CommonsZome {
    if qt.contains("clt") || qt.contains("lease") || qt.contains("resale") {
        CommonsZome::HousingClt
    } else if qt.contains("member") {
        CommonsZome::HousingMembership
    } else if qt.contains("finance") || qt.contains("fee") {
        CommonsZome::HousingFinances
    } else if qt.contains("maintenance") || qt.contains("repair") {
        CommonsZome::HousingMaintenance
    } else if qt.contains("governance") || qt.contains("proposal") {
        CommonsZome::HousingGovernance
    } else {
        CommonsZome::HousingUnits
    }
}

fn resolve_care(qt: &str) -> CommonsZome {
    if qt.contains("match") {
        CommonsZome::CareMatching
    } else if qt.contains("circle") {
        CommonsZome::CareCircles
    } else if qt.contains("credential") {
        CommonsZome::CareCredentials
    } else if qt.contains("plan") {
        CommonsZome::CarePlans
    } else {
        CommonsZome::CareTimebank
    }
}

fn resolve_mutualaid(qt: &str) -> CommonsZome {
    if qt.contains("resource") || qt.contains("booking") {
        CommonsZome::MutualaidResources
    } else if qt.contains("need") || qt.contains("handoff") {
        CommonsZome::MutualaidNeeds
    } else if qt.contains("pool") {
        CommonsZome::MutualaidPools
    } else if qt.contains("request") {
        CommonsZome::MutualaidRequests
    } else if qt.contains("circle") {
        CommonsZome::MutualaidCircles
    } else if qt.contains("governance") || qt.contains("proposal") {
        CommonsZome::MutualaidGovernance
    } else {
        CommonsZome::MutualaidTimebank
    }
}

fn resolve_water(qt: &str) -> CommonsZome {
    if qt.contains("purity") || qt.contains("quality") {
        CommonsZome::WaterPurity
    } else if qt.contains("capture") || qt.contains("harvest") {
        CommonsZome::WaterCapture
    } else if qt.contains("steward") || qt.contains("guardian") {
        CommonsZome::WaterSteward
    } else if qt.contains("wisdom") || qt.contains("knowledge") {
        CommonsZome::WaterWisdom
    } else {
        CommonsZome::WaterFlow
    }
}

fn resolve_food(qt: &str) -> CommonsZome {
    if qt.contains("distribution")
        || qt.contains("market")
        || qt.contains("order")
        || qt.contains("allergen")
        || qt.contains("product")
        || qt.contains("listing")
    {
        CommonsZome::FoodDistribution
    } else if qt.contains("preservation") || qt.contains("batch") || qt.contains("storage") {
        CommonsZome::FoodPreservation
    } else if qt.contains("knowledge")
        || qt.contains("seed")
        || qt.contains("recipe")
        || qt.contains("nutrient")
        || qt.contains("rating")
        || qt.contains("practice")
    {
        CommonsZome::FoodKnowledge
    } else {
        CommonsZome::FoodProduction
    }
}

fn resolve_transport(qt: &str) -> CommonsZome {
    if qt.contains("share") || qt.contains("ride") || qt.contains("cargo") {
        CommonsZome::TransportSharing
    } else if qt.contains("impact")
        || qt.contains("carbon")
        || qt.contains("emission")
        || qt.contains("redeem")
        || qt.contains("balance")
        || qt.contains("redemption")
        || qt.contains("trip")
    {
        CommonsZome::TransportImpact
    } else {
        CommonsZome::TransportRoutes
    }
}

fn resolve_support(qt: &str) -> CommonsZome {
    if qt.contains("ticket")
        || qt.contains("alert")
        || qt.contains("preemptive")
        || qt.contains("satisfaction")
        || qt.contains("undo")
        || qt.contains("escalat")
        || qt.contains("comment")
        || qt.contains("action")
    {
        CommonsZome::SupportTickets
    } else if qt.contains("diagnostic")
        || qt.contains("helper")
        || qt.contains("availability")
        || qt.contains("cognitive")
        || qt.contains("privacy")
    {
        CommonsZome::SupportDiagnostics
    } else {
        CommonsZome::SupportKnowledge
    }
}

// ============================================================================
// Routing: Civic
// ============================================================================

/// Resolve a civic domain + query_type to a specific zome.
///
/// This is the type-safe replacement for the `resolve_domain_zome()` function
/// in `civic-bridge/coordinator/src/lib.rs` (lines 168-194).
///
/// Query type matching is case-insensitive (lowercased before comparison).
pub fn resolve_civic_zome(domain: BridgeDomain, query_type: &str) -> Option<CivicZome> {
    let qt = query_type.to_ascii_lowercase();
    match domain {
        BridgeDomain::Justice => Some(resolve_justice(&qt)),
        BridgeDomain::Emergency => Some(resolve_emergency(&qt)),
        BridgeDomain::Media => Some(resolve_media(&qt)),
        // Commons domains → None
        _ => None,
    }
}

fn resolve_justice(qt: &str) -> CivicZome {
    if qt.contains("evidence") {
        CivicZome::JusticeEvidence
    } else if qt.contains("arbitrat") {
        CivicZome::JusticeArbitration
    } else if qt.contains("restorative") || qt.contains("mediat") {
        CivicZome::JusticeRestorative
    } else if qt.contains("enforce") || qt.contains("sanction") {
        CivicZome::JusticeEnforcement
    } else {
        CivicZome::JusticeCases
    }
}

fn resolve_emergency(qt: &str) -> CivicZome {
    if qt.contains("triage") || qt.contains("priorit") {
        CivicZome::EmergencyTriage
    } else if qt.contains("resource") || qt.contains("supply") {
        CivicZome::EmergencyResources
    } else if qt.contains("coordinat") {
        CivicZome::EmergencyCoordination
    } else if qt.contains("shelter") {
        CivicZome::EmergencyShelters
    } else if qt.contains("comm") || qt.contains("alert") {
        CivicZome::EmergencyComms
    } else {
        CivicZome::EmergencyIncidents
    }
}

fn resolve_media(qt: &str) -> CivicZome {
    if qt.contains("attribution") || qt.contains("source") {
        CivicZome::MediaAttribution
    } else if qt.contains("fact") || qt.contains("check") || qt.contains("verify") {
        CivicZome::MediaFactcheck
    } else if qt.contains("curat") || qt.contains("recommend") {
        CivicZome::MediaCuration
    } else {
        CivicZome::MediaPublication
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- BridgeDomain --

    #[test]
    fn domain_from_str_loose_lowercase() {
        assert_eq!(
            BridgeDomain::from_str_loose("property"),
            Some(BridgeDomain::Property)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("housing"),
            Some(BridgeDomain::Housing)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("care"),
            Some(BridgeDomain::Care)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("mutualaid"),
            Some(BridgeDomain::Mutualaid)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("water"),
            Some(BridgeDomain::Water)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("food"),
            Some(BridgeDomain::Food)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("transport"),
            Some(BridgeDomain::Transport)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("support"),
            Some(BridgeDomain::Support)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("space"),
            Some(BridgeDomain::Space)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("justice"),
            Some(BridgeDomain::Justice)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("emergency"),
            Some(BridgeDomain::Emergency)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("media"),
            Some(BridgeDomain::Media)
        );
    }

    #[test]
    fn domain_from_str_case_insensitive() {
        assert_eq!(
            BridgeDomain::from_str_loose("Property"),
            Some(BridgeDomain::Property)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("HOUSING"),
            Some(BridgeDomain::Housing)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("Justice"),
            Some(BridgeDomain::Justice)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("eMeRgEnCy"),
            Some(BridgeDomain::Emergency)
        );
    }

    #[test]
    fn domain_from_str_mutualaid_aliases() {
        assert_eq!(
            BridgeDomain::from_str_loose("mutualaid"),
            Some(BridgeDomain::Mutualaid)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("mutual_aid"),
            Some(BridgeDomain::Mutualaid)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("mutual-aid"),
            Some(BridgeDomain::Mutualaid)
        );
        assert_eq!(
            BridgeDomain::from_str_loose("Mutual_Aid"),
            Some(BridgeDomain::Mutualaid)
        );
    }

    #[test]
    fn domain_from_str_unknown_returns_none() {
        assert_eq!(BridgeDomain::from_str_loose("unknown"), None);
        assert_eq!(BridgeDomain::from_str_loose(""), None);
        assert_eq!(BridgeDomain::from_str_loose("propert"), None);
    }

    #[test]
    fn domain_as_str_roundtrip() {
        let domains = [
            BridgeDomain::Property,
            BridgeDomain::Housing,
            BridgeDomain::Care,
            BridgeDomain::Mutualaid,
            BridgeDomain::Water,
            BridgeDomain::Food,
            BridgeDomain::Transport,
            BridgeDomain::Support,
            BridgeDomain::Space,
            BridgeDomain::Justice,
            BridgeDomain::Emergency,
            BridgeDomain::Media,
        ];
        for d in &domains {
            assert_eq!(
                BridgeDomain::from_str_loose(d.as_str()),
                Some(*d),
                "as_str roundtrip failed for {:?}",
                d
            );
        }
    }

    #[test]
    fn domain_is_commons_civic_partition() {
        let all = [
            BridgeDomain::Property,
            BridgeDomain::Housing,
            BridgeDomain::Care,
            BridgeDomain::Mutualaid,
            BridgeDomain::Water,
            BridgeDomain::Food,
            BridgeDomain::Transport,
            BridgeDomain::Support,
            BridgeDomain::Space,
            BridgeDomain::Justice,
            BridgeDomain::Emergency,
            BridgeDomain::Media,
        ];
        for d in &all {
            assert!(
                d.is_commons() ^ d.is_civic(),
                "{:?} should be exactly one of commons/civic",
                d
            );
        }
    }

    #[test]
    fn commons_domains_count() {
        assert_eq!(COMMONS_DOMAINS.len(), 9);
    }

    #[test]
    fn civic_domains_count() {
        assert_eq!(CIVIC_DOMAINS.len(), 3);
    }

    // -- BridgeDomain serde --

    #[test]
    fn domain_serde_roundtrip() {
        let domains = [
            BridgeDomain::Property,
            BridgeDomain::Housing,
            BridgeDomain::Justice,
            BridgeDomain::Emergency,
            BridgeDomain::Media,
        ];
        for d in &domains {
            let json = serde_json::to_string(d).unwrap();
            let d2: BridgeDomain = serde_json::from_str(&json).unwrap();
            assert_eq!(*d, d2, "serde roundtrip failed for {:?}", d);
        }
    }

    #[test]
    fn domain_serde_lowercase_format() {
        let json = serde_json::to_string(&BridgeDomain::Property).unwrap();
        assert_eq!(json, "\"property\"");
        let json = serde_json::to_string(&BridgeDomain::Mutualaid).unwrap();
        assert_eq!(json, "\"mutualaid\"");
    }

    // -- CommonsZome --

    #[test]
    fn commons_zome_count() {
        assert_eq!(CommonsZome::ALL.len(), 38);
    }

    #[test]
    fn commons_zome_as_str_unique() {
        let mut names: Vec<&str> = CommonsZome::ALL.iter().map(|z| z.as_str()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "Duplicate as_str values found");
    }

    #[test]
    fn commons_zome_serde_roundtrip_all() {
        for z in CommonsZome::ALL {
            let json = serde_json::to_string(z).unwrap();
            let z2: CommonsZome = serde_json::from_str(&json).unwrap();
            assert_eq!(*z, z2, "serde roundtrip failed for {:?}", z);
        }
    }

    #[test]
    fn commons_zome_land_care_partition() {
        let land_count = CommonsZome::ALL.iter().filter(|z| z.is_land()).count();
        let care_count = CommonsZome::ALL.iter().filter(|z| !z.is_land()).count();
        assert_eq!(land_count, 19, "Expected 19 land zomes");
        assert_eq!(care_count, 19, "Expected 19 care zomes");
    }

    #[test]
    fn commons_zome_default_for_each_commons_domain() {
        for d in COMMONS_DOMAINS {
            assert!(
                CommonsZome::default_for_domain(*d).is_some(),
                "Missing default for commons domain {:?}",
                d
            );
        }
    }

    #[test]
    fn commons_zome_default_none_for_civic_domains() {
        for d in CIVIC_DOMAINS {
            assert!(CommonsZome::default_for_domain(*d).is_none());
        }
    }

    // -- CivicZome --

    #[test]
    fn civic_zome_count() {
        assert_eq!(CivicZome::ALL.len(), 15);
    }

    #[test]
    fn civic_zome_as_str_unique() {
        let mut names: Vec<&str> = CivicZome::ALL.iter().map(|z| z.as_str()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "Duplicate as_str values found");
    }

    #[test]
    fn civic_zome_serde_roundtrip_all() {
        for z in CivicZome::ALL {
            let json = serde_json::to_string(z).unwrap();
            let z2: CivicZome = serde_json::from_str(&json).unwrap();
            assert_eq!(*z, z2, "serde roundtrip failed for {:?}", z);
        }
    }

    #[test]
    fn civic_zome_default_for_each_civic_domain() {
        for d in CIVIC_DOMAINS {
            assert!(
                CivicZome::default_for_domain(*d).is_some(),
                "Missing default for civic domain {:?}",
                d
            );
        }
    }

    #[test]
    fn civic_zome_default_none_for_commons_domains() {
        for d in COMMONS_DOMAINS {
            assert!(CivicZome::default_for_domain(*d).is_none());
        }
    }

    // -- CrossClusterRole --

    #[test]
    fn cross_cluster_role_serde_roundtrip() {
        let roles = [
            CrossClusterRole::Commons,
            CrossClusterRole::Civic,
            CrossClusterRole::Identity,
        ];
        for r in &roles {
            let json = serde_json::to_string(r).unwrap();
            let r2: CrossClusterRole = serde_json::from_str(&json).unwrap();
            assert_eq!(*r, r2);
        }
    }

    #[test]
    fn cross_cluster_role_as_str() {
        assert_eq!(CrossClusterRole::Commons.as_str(), "commons");
        assert_eq!(CrossClusterRole::Civic.as_str(), "civic");
        assert_eq!(CrossClusterRole::Identity.as_str(), "identity");
    }

    // -- resolve_commons_zome: Property routing --

    #[test]
    fn property_transfer_keywords() {
        let d = BridgeDomain::Property;
        assert_eq!(
            resolve_commons_zome(d, "transfer"),
            Some(CommonsZome::PropertyTransfer)
        );
        assert_eq!(
            resolve_commons_zome(d, "ownership"),
            Some(CommonsZome::PropertyTransfer)
        );
        assert_eq!(
            resolve_commons_zome(d, "ownership_transfer"),
            Some(CommonsZome::PropertyTransfer)
        );
    }

    #[test]
    fn property_disputes_keywords() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "dispute"),
            Some(CommonsZome::PropertyDisputes)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "boundary_dispute"),
            Some(CommonsZome::PropertyDisputes)
        );
    }

    #[test]
    fn property_registry_keywords() {
        let d = BridgeDomain::Property;
        assert_eq!(
            resolve_commons_zome(d, "encumbrance"),
            Some(CommonsZome::PropertyRegistry)
        );
        assert_eq!(
            resolve_commons_zome(d, "title"),
            Some(CommonsZome::PropertyRegistry)
        );
    }

    #[test]
    fn property_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "anything_else"),
            Some(CommonsZome::PropertyRegistry)
        );
    }

    // -- resolve_commons_zome: Housing routing --

    #[test]
    fn housing_clt_keywords() {
        let d = BridgeDomain::Housing;
        assert_eq!(
            resolve_commons_zome(d, "clt"),
            Some(CommonsZome::HousingClt)
        );
        assert_eq!(
            resolve_commons_zome(d, "lease"),
            Some(CommonsZome::HousingClt)
        );
        assert_eq!(
            resolve_commons_zome(d, "resale"),
            Some(CommonsZome::HousingClt)
        );
    }

    #[test]
    fn housing_membership_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Housing, "member_status"),
            Some(CommonsZome::HousingMembership)
        );
    }

    #[test]
    fn housing_finances_keywords() {
        let d = BridgeDomain::Housing;
        assert_eq!(
            resolve_commons_zome(d, "finance"),
            Some(CommonsZome::HousingFinances)
        );
        assert_eq!(
            resolve_commons_zome(d, "fee_schedule"),
            Some(CommonsZome::HousingFinances)
        );
    }

    #[test]
    fn housing_maintenance_keywords() {
        let d = BridgeDomain::Housing;
        assert_eq!(
            resolve_commons_zome(d, "maintenance"),
            Some(CommonsZome::HousingMaintenance)
        );
        assert_eq!(
            resolve_commons_zome(d, "repair_request"),
            Some(CommonsZome::HousingMaintenance)
        );
    }

    #[test]
    fn housing_governance_keywords() {
        let d = BridgeDomain::Housing;
        assert_eq!(
            resolve_commons_zome(d, "governance"),
            Some(CommonsZome::HousingGovernance)
        );
        assert_eq!(
            resolve_commons_zome(d, "proposal"),
            Some(CommonsZome::HousingGovernance)
        );
    }

    #[test]
    fn housing_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Housing, "listing"),
            Some(CommonsZome::HousingUnits)
        );
    }

    // -- resolve_commons_zome: Care routing --

    #[test]
    fn care_matching_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "match_provider"),
            Some(CommonsZome::CareMatching)
        );
    }

    #[test]
    fn care_circles_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "circle"),
            Some(CommonsZome::CareCircles)
        );
    }

    #[test]
    fn care_credentials_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "credential"),
            Some(CommonsZome::CareCredentials)
        );
    }

    #[test]
    fn care_plans_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "plan"),
            Some(CommonsZome::CarePlans)
        );
    }

    #[test]
    fn care_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "general"),
            Some(CommonsZome::CareTimebank)
        );
    }

    // -- resolve_commons_zome: Mutualaid routing --

    #[test]
    fn mutualaid_resources_keywords() {
        let d = BridgeDomain::Mutualaid;
        assert_eq!(
            resolve_commons_zome(d, "resource"),
            Some(CommonsZome::MutualaidResources)
        );
        assert_eq!(
            resolve_commons_zome(d, "booking"),
            Some(CommonsZome::MutualaidResources)
        );
    }

    #[test]
    fn mutualaid_needs_keywords() {
        let d = BridgeDomain::Mutualaid;
        assert_eq!(
            resolve_commons_zome(d, "need"),
            Some(CommonsZome::MutualaidNeeds)
        );
        assert_eq!(
            resolve_commons_zome(d, "handoff"),
            Some(CommonsZome::MutualaidNeeds)
        );
    }

    #[test]
    fn mutualaid_pools_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Mutualaid, "pool"),
            Some(CommonsZome::MutualaidPools)
        );
    }

    #[test]
    fn mutualaid_requests_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Mutualaid, "request"),
            Some(CommonsZome::MutualaidRequests)
        );
    }

    #[test]
    fn mutualaid_circles_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Mutualaid, "circle"),
            Some(CommonsZome::MutualaidCircles)
        );
    }

    #[test]
    fn mutualaid_governance_keywords() {
        let d = BridgeDomain::Mutualaid;
        assert_eq!(
            resolve_commons_zome(d, "governance"),
            Some(CommonsZome::MutualaidGovernance)
        );
        assert_eq!(
            resolve_commons_zome(d, "proposal"),
            Some(CommonsZome::MutualaidGovernance)
        );
    }

    #[test]
    fn mutualaid_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Mutualaid, "general"),
            Some(CommonsZome::MutualaidTimebank)
        );
    }

    // -- resolve_commons_zome: Water routing --

    #[test]
    fn water_purity_keywords() {
        let d = BridgeDomain::Water;
        assert_eq!(
            resolve_commons_zome(d, "purity"),
            Some(CommonsZome::WaterPurity)
        );
        assert_eq!(
            resolve_commons_zome(d, "quality"),
            Some(CommonsZome::WaterPurity)
        );
    }

    #[test]
    fn water_capture_keywords() {
        let d = BridgeDomain::Water;
        assert_eq!(
            resolve_commons_zome(d, "capture"),
            Some(CommonsZome::WaterCapture)
        );
        assert_eq!(
            resolve_commons_zome(d, "harvest"),
            Some(CommonsZome::WaterCapture)
        );
    }

    #[test]
    fn water_steward_keywords() {
        let d = BridgeDomain::Water;
        assert_eq!(
            resolve_commons_zome(d, "steward"),
            Some(CommonsZome::WaterSteward)
        );
        assert_eq!(
            resolve_commons_zome(d, "guardian"),
            Some(CommonsZome::WaterSteward)
        );
    }

    #[test]
    fn water_wisdom_keywords() {
        let d = BridgeDomain::Water;
        assert_eq!(
            resolve_commons_zome(d, "wisdom"),
            Some(CommonsZome::WaterWisdom)
        );
        assert_eq!(
            resolve_commons_zome(d, "knowledge"),
            Some(CommonsZome::WaterWisdom)
        );
    }

    #[test]
    fn water_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Water, "general"),
            Some(CommonsZome::WaterFlow)
        );
    }

    // -- resolve_commons_zome: Food routing --

    #[test]
    fn food_distribution_keywords() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "distribution"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "market"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "order"),
            Some(CommonsZome::FoodDistribution)
        );
    }

    #[test]
    fn food_preservation_keywords() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "preservation"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "batch"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "storage"),
            Some(CommonsZome::FoodPreservation)
        );
    }

    #[test]
    fn food_knowledge_keywords() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "knowledge"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "seed"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "recipe"),
            Some(CommonsZome::FoodKnowledge)
        );
    }

    #[test]
    fn food_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Food, "general"),
            Some(CommonsZome::FoodProduction)
        );
    }

    // -- resolve_commons_zome: Transport routing --

    #[test]
    fn transport_sharing_keywords() {
        let d = BridgeDomain::Transport;
        assert_eq!(
            resolve_commons_zome(d, "share"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "ride"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "cargo"),
            Some(CommonsZome::TransportSharing)
        );
    }

    #[test]
    fn transport_impact_keywords() {
        let d = BridgeDomain::Transport;
        assert_eq!(
            resolve_commons_zome(d, "impact"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "carbon"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "emission"),
            Some(CommonsZome::TransportImpact)
        );
    }

    #[test]
    fn transport_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Transport, "route_search"),
            Some(CommonsZome::TransportRoutes)
        );
    }

    // -- resolve_commons_zome: Support routing --

    #[test]
    fn support_tickets_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Support, "ticket"),
            Some(CommonsZome::SupportTickets)
        );
    }

    #[test]
    fn support_diagnostics_keyword() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Support, "diagnostic"),
            Some(CommonsZome::SupportDiagnostics)
        );
    }

    #[test]
    fn support_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Support, "general"),
            Some(CommonsZome::SupportKnowledge)
        );
    }

    // -- resolve_commons_zome: Space --

    #[test]
    fn space_always_resolves() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Space, "anything"),
            Some(CommonsZome::Space)
        );
    }

    // -- resolve_commons_zome: Civic domains return None --

    #[test]
    fn commons_routing_none_for_civic_domains() {
        assert_eq!(resolve_commons_zome(BridgeDomain::Justice, "cases"), None);
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Emergency, "incidents"),
            None
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Media, "publication"),
            None
        );
    }

    // -- resolve_commons_zome: Case insensitivity --

    #[test]
    fn commons_routing_case_insensitive() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "TRANSFER"),
            Some(CommonsZome::PropertyTransfer)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Housing, "Governance"),
            Some(CommonsZome::HousingGovernance)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Water, "PURITY_check"),
            Some(CommonsZome::WaterPurity)
        );
    }

    // -- resolve_civic_zome: Justice routing --

    #[test]
    fn justice_evidence_keyword() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "evidence"),
            Some(CivicZome::JusticeEvidence)
        );
    }

    #[test]
    fn justice_arbitration_keyword() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "arbitration"),
            Some(CivicZome::JusticeArbitration)
        );
        // Prefix match
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "arbitrat_request"),
            Some(CivicZome::JusticeArbitration)
        );
    }

    #[test]
    fn justice_restorative_keywords() {
        let d = BridgeDomain::Justice;
        assert_eq!(
            resolve_civic_zome(d, "restorative"),
            Some(CivicZome::JusticeRestorative)
        );
        assert_eq!(
            resolve_civic_zome(d, "mediation"),
            Some(CivicZome::JusticeRestorative)
        );
    }

    #[test]
    fn justice_enforcement_keywords() {
        let d = BridgeDomain::Justice;
        assert_eq!(
            resolve_civic_zome(d, "enforce"),
            Some(CivicZome::JusticeEnforcement)
        );
        assert_eq!(
            resolve_civic_zome(d, "sanction"),
            Some(CivicZome::JusticeEnforcement)
        );
    }

    #[test]
    fn justice_default() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "general"),
            Some(CivicZome::JusticeCases)
        );
    }

    // -- resolve_civic_zome: Emergency routing --

    #[test]
    fn emergency_triage_keywords() {
        let d = BridgeDomain::Emergency;
        assert_eq!(
            resolve_civic_zome(d, "triage"),
            Some(CivicZome::EmergencyTriage)
        );
        assert_eq!(
            resolve_civic_zome(d, "priority_assessment"),
            Some(CivicZome::EmergencyTriage)
        );
    }

    #[test]
    fn emergency_resources_keywords() {
        let d = BridgeDomain::Emergency;
        assert_eq!(
            resolve_civic_zome(d, "resource"),
            Some(CivicZome::EmergencyResources)
        );
        assert_eq!(
            resolve_civic_zome(d, "supply_status"),
            Some(CivicZome::EmergencyResources)
        );
    }

    #[test]
    fn emergency_coordination_keyword() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Emergency, "coordination"),
            Some(CivicZome::EmergencyCoordination)
        );
    }

    #[test]
    fn emergency_shelters_keyword() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Emergency, "shelter"),
            Some(CivicZome::EmergencyShelters)
        );
    }

    #[test]
    fn emergency_comms_keywords() {
        let d = BridgeDomain::Emergency;
        assert_eq!(
            resolve_civic_zome(d, "comm"),
            Some(CivicZome::EmergencyComms)
        );
        assert_eq!(
            resolve_civic_zome(d, "alert"),
            Some(CivicZome::EmergencyComms)
        );
    }

    #[test]
    fn emergency_default() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Emergency, "general"),
            Some(CivicZome::EmergencyIncidents)
        );
    }

    // -- resolve_civic_zome: Media routing --

    #[test]
    fn media_attribution_keywords() {
        let d = BridgeDomain::Media;
        assert_eq!(
            resolve_civic_zome(d, "attribution"),
            Some(CivicZome::MediaAttribution)
        );
        assert_eq!(
            resolve_civic_zome(d, "source_check"),
            Some(CivicZome::MediaAttribution)
        );
    }

    #[test]
    fn media_factcheck_keywords() {
        let d = BridgeDomain::Media;
        assert_eq!(
            resolve_civic_zome(d, "factcheck"),
            Some(CivicZome::MediaFactcheck)
        );
        assert_eq!(
            resolve_civic_zome(d, "check_claim"),
            Some(CivicZome::MediaFactcheck)
        );
        assert_eq!(
            resolve_civic_zome(d, "verify"),
            Some(CivicZome::MediaFactcheck)
        );
    }

    #[test]
    fn media_curation_keywords() {
        let d = BridgeDomain::Media;
        assert_eq!(
            resolve_civic_zome(d, "curation"),
            Some(CivicZome::MediaCuration)
        );
        assert_eq!(
            resolve_civic_zome(d, "recommend"),
            Some(CivicZome::MediaCuration)
        );
    }

    #[test]
    fn media_default() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Media, "general"),
            Some(CivicZome::MediaPublication)
        );
    }

    // -- resolve_civic_zome: Commons domains return None --

    #[test]
    fn civic_routing_none_for_commons_domains() {
        assert_eq!(resolve_civic_zome(BridgeDomain::Property, "transfer"), None);
        assert_eq!(resolve_civic_zome(BridgeDomain::Housing, "units"), None);
        assert_eq!(resolve_civic_zome(BridgeDomain::Care, "match"), None);
        assert_eq!(resolve_civic_zome(BridgeDomain::Space, "booking"), None);
    }

    // -- resolve_civic_zome: Case insensitivity --

    #[test]
    fn civic_routing_case_insensitive() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "EVIDENCE_review"),
            Some(CivicZome::JusticeEvidence)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Emergency, "TRIAGE"),
            Some(CivicZome::EmergencyTriage)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Media, "Attribution"),
            Some(CivicZome::MediaAttribution)
        );
    }

    // -- ALLOWED_ZOMES consistency checks --

    #[test]
    fn commons_zome_as_str_matches_allowed_zomes() {
        // The 38 ALLOWED_ZOMES in commons-bridge should match CommonsZome::ALL as_str values
        let expected: Vec<&str> = vec![
            "property_registry",
            "property_transfer",
            "property_disputes",
            "property_commons",
            "housing_units",
            "housing_membership",
            "housing_finances",
            "housing_maintenance",
            "housing_clt",
            "housing_governance",
            "care_timebank",
            "care_circles",
            "care_matching",
            "care_plans",
            "care_credentials",
            "mutualaid_needs",
            "mutualaid_circles",
            "mutualaid_governance",
            "mutualaid_pools",
            "mutualaid_requests",
            "mutualaid_resources",
            "mutualaid_timebank",
            "water_flow",
            "water_purity",
            "water_capture",
            "water_steward",
            "water_wisdom",
            "food_production",
            "food_distribution",
            "food_preservation",
            "food_knowledge",
            "transport_routes",
            "transport_sharing",
            "transport_impact",
            "support_knowledge",
            "support_tickets",
            "support_diagnostics",
            "space",
        ];
        let actual: Vec<&str> = CommonsZome::ALL.iter().map(|z| z.as_str()).collect();
        assert_eq!(actual.len(), expected.len());
        for name in &expected {
            assert!(
                actual.contains(name),
                "Expected zome '{}' not found in CommonsZome::ALL",
                name
            );
        }
    }

    #[test]
    fn civic_zome_as_str_matches_allowed_zomes() {
        let expected: Vec<&str> = vec![
            "justice_cases",
            "justice_evidence",
            "justice_arbitration",
            "justice_restorative",
            "justice_enforcement",
            "emergency_incidents",
            "emergency_triage",
            "emergency_resources",
            "emergency_coordination",
            "emergency_shelters",
            "emergency_comms",
            "media_publication",
            "media_attribution",
            "media_factcheck",
            "media_curation",
        ];
        let actual: Vec<&str> = CivicZome::ALL.iter().map(|z| z.as_str()).collect();
        assert_eq!(actual.len(), expected.len());
        for name in &expected {
            assert!(
                actual.contains(name),
                "Expected zome '{}' not found in CivicZome::ALL",
                name
            );
        }
    }

    // -- Routing matches default_for_domain --

    #[test]
    fn resolve_commons_empty_query_returns_default() {
        for d in COMMONS_DOMAINS {
            let default = CommonsZome::default_for_domain(*d);
            let resolved = resolve_commons_zome(*d, "");
            assert_eq!(
                resolved, default,
                "Empty query for {:?} should return default {:?}, got {:?}",
                d, default, resolved
            );
        }
    }

    #[test]
    fn resolve_civic_empty_query_returns_default() {
        for d in CIVIC_DOMAINS {
            let default = CivicZome::default_for_domain(*d);
            let resolved = resolve_civic_zome(*d, "");
            assert_eq!(
                resolved, default,
                "Empty query for {:?} should return default {:?}, got {:?}",
                d, default, resolved
            );
        }
    }

    // -- Case-insensitive keyword routing edge cases --

    #[test]
    fn commons_routing_mixed_case_keywords() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "Transfer_Deed"),
            Some(CommonsZome::PropertyTransfer)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Housing, "GoVeRnAnCe"),
            Some(CommonsZome::HousingGovernance)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Care, "CREDENTIAL_check"),
            Some(CommonsZome::CareCredentials)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Mutualaid, "Pool_Status"),
            Some(CommonsZome::MutualaidPools)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Water, "PURITY"),
            Some(CommonsZome::WaterPurity)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Food, "DISTRIBUTION"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Transport, "SHARE_ride"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Support, "TICKET_create"),
            Some(CommonsZome::SupportTickets)
        );
    }

    #[test]
    fn civic_routing_mixed_case_keywords() {
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "ARBITRATION_panel"),
            Some(CivicZome::JusticeArbitration)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "Restorative_Justice"),
            Some(CivicZome::JusticeRestorative)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Emergency, "SHELTER_capacity"),
            Some(CivicZome::EmergencyShelters)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Media, "FACTCHECK_Status"),
            Some(CivicZome::MediaFactcheck)
        );
    }

    // -- Unknown keyword falls through to default --

    #[test]
    fn unknown_keywords_fall_through_to_default() {
        // Every domain should route unknown keywords to its default zome
        let cases = [
            (
                BridgeDomain::Property,
                "xyzzy",
                CommonsZome::PropertyRegistry,
            ),
            (
                BridgeDomain::Housing,
                "nonexistent_query",
                CommonsZome::HousingUnits,
            ),
            (
                BridgeDomain::Care,
                "404_not_found",
                CommonsZome::CareTimebank,
            ),
            (
                BridgeDomain::Mutualaid,
                "random_gibberish",
                CommonsZome::MutualaidTimebank,
            ),
            (BridgeDomain::Water, "unicorn_data", CommonsZome::WaterFlow),
            (
                BridgeDomain::Food,
                "quantum_soup",
                CommonsZome::FoodProduction,
            ),
            (
                BridgeDomain::Transport,
                "teleportation",
                CommonsZome::TransportRoutes,
            ),
            (
                BridgeDomain::Support,
                "astrology",
                CommonsZome::SupportKnowledge,
            ),
        ];
        for (domain, keyword, expected) in &cases {
            assert_eq!(
                resolve_commons_zome(*domain, keyword),
                Some(*expected),
                "Unknown keyword '{}' for {:?} should return default {:?}",
                keyword,
                domain,
                expected
            );
        }
    }

    #[test]
    fn unknown_keywords_civic_fall_through_to_default() {
        let cases = [
            (BridgeDomain::Justice, "xyzzy", CivicZome::JusticeCases),
            (
                BridgeDomain::Emergency,
                "nonexistent",
                CivicZome::EmergencyIncidents,
            ),
            (
                BridgeDomain::Media,
                "gibberish",
                CivicZome::MediaPublication,
            ),
        ];
        for (domain, keyword, expected) in &cases {
            assert_eq!(
                resolve_civic_zome(*domain, keyword),
                Some(*expected),
                "Unknown keyword '{}' for {:?} should return default {:?}",
                keyword,
                domain,
                expected
            );
        }
    }

    // -- Whitespace and special characters --

    #[test]
    fn whitespace_in_query_type_falls_to_default() {
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "   "),
            Some(CommonsZome::PropertyRegistry)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "   "),
            Some(CivicZome::JusticeCases)
        );
    }

    #[test]
    fn query_type_with_keyword_embedded_in_whitespace() {
        // "  transfer  " contains "transfer"
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "  transfer  "),
            Some(CommonsZome::PropertyTransfer)
        );
    }

    #[test]
    fn query_type_with_special_characters() {
        // No keywords match — falls to default
        assert_eq!(
            resolve_commons_zome(BridgeDomain::Property, "!@#$%^&*()"),
            Some(CommonsZome::PropertyRegistry)
        );
        assert_eq!(
            resolve_civic_zome(BridgeDomain::Justice, "!@#$%^&*()"),
            Some(CivicZome::JusticeCases)
        );
    }

    // -- Phase 2-4 enrichment keyword routing --

    #[test]
    fn transport_impact_redeem_keywords() {
        let d = BridgeDomain::Transport;
        assert_eq!(
            resolve_commons_zome(d, "redeem_credits"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_agent_carbon_balance"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_my_redemptions"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "redemption_history"),
            Some(CommonsZome::TransportImpact)
        );
    }

    #[test]
    fn food_nutrient_keyword() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "add_nutrient_profile"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_nutrient_profile"),
            Some(CommonsZome::FoodKnowledge)
        );
    }

    #[test]
    fn food_allergen_keyword() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "search_allergen_safe"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "allergen_check"),
            Some(CommonsZome::FoodDistribution)
        );
    }

    #[test]
    fn support_diagnostics_helper_keywords() {
        let d = BridgeDomain::Support;
        assert_eq!(
            resolve_commons_zome(d, "register_helper"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "update_availability"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_available_helpers"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "publish_cognitive_update"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "set_privacy_preference"),
            Some(CommonsZome::SupportDiagnostics)
        );
    }

    #[test]
    fn support_tickets_escalation_keywords() {
        let d = BridgeDomain::Support;
        assert_eq!(
            resolve_commons_zome(d, "create_preemptive_alert"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "list_preemptive_alerts"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "promote_alert_to_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "escalate_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "submit_satisfaction"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_undo"),
            Some(CommonsZome::SupportTickets)
        );
    }

    // -- Phase 2-4 full function name dispatch coverage --

    #[test]
    fn phase2_food_function_names_dispatch_correctly() {
        let d = BridgeDomain::Food;
        // Seed exchange → FoodKnowledge (contains "seed")
        assert_eq!(
            resolve_commons_zome(d, "offer_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "request_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_available_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_open_seed_requests"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "match_seed_request"),
            Some(CommonsZome::FoodKnowledge)
        );
        // Seed quality ratings → FoodKnowledge (contains "rating" or "seed")
        assert_eq!(
            resolve_commons_zome(d, "rate_seed_exchange"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_exchange_ratings"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_grower_ratings"),
            Some(CommonsZome::FoodKnowledge)
        );
        // Garden membership → FoodProduction (default)
        assert_eq!(
            resolve_commons_zome(d, "add_garden_member"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_plot_members"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "remove_garden_member"),
            Some(CommonsZome::FoodProduction)
        );
    }

    #[test]
    fn phase2_transport_function_names_dispatch_correctly() {
        let d = BridgeDomain::Transport;
        assert_eq!(
            resolve_commons_zome(d, "review_ride"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_ride_reviews"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "find_nearby_rides"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "redeem_credits"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_my_redemptions"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_agent_carbon_balance"),
            Some(CommonsZome::TransportImpact)
        );
        // Maintenance/features → TransportRoutes (default)
        assert_eq!(
            resolve_commons_zome(d, "log_maintenance"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_vehicle_maintenance"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "set_vehicle_features"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_accessible_vehicles"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_driver_rating"),
            Some(CommonsZome::TransportRoutes)
        );
    }

    #[test]
    fn phase2_support_function_names_dispatch_correctly() {
        let d = BridgeDomain::Support;
        // Diagnostics
        assert_eq!(
            resolve_commons_zome(d, "run_diagnostic"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "register_helper"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "publish_cognitive_update"),
            Some(CommonsZome::SupportDiagnostics)
        );
        // Tickets
        assert_eq!(
            resolve_commons_zome(d, "create_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "escalate_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_preemptive_alert"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "submit_satisfaction"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_undo"),
            Some(CommonsZome::SupportTickets)
        );
        // Knowledge (default)
        assert_eq!(
            resolve_commons_zome(d, "create_article"),
            Some(CommonsZome::SupportKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "search_by_category"),
            Some(CommonsZome::SupportKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_resolution"),
            Some(CommonsZome::SupportKnowledge)
        );
    }

    // -- Domain from_str_loose edge cases --

    #[test]
    fn domain_from_str_loose_leading_trailing_spaces_not_trimmed() {
        // Intentionally NOT trimmed — caller should trim
        assert_eq!(BridgeDomain::from_str_loose(" property"), None);
        assert_eq!(BridgeDomain::from_str_loose("property "), None);
    }

    #[test]
    fn domain_from_str_loose_unicode_not_matched() {
        assert_eq!(BridgeDomain::from_str_loose("próperty"), None);
        assert_eq!(BridgeDomain::from_str_loose("justíce"), None);
    }

    // -- New keyword routing (post-enrichment) --

    #[test]
    fn food_practice_routes_to_knowledge() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "share_practice"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_practices_by_category"),
            Some(CommonsZome::FoodKnowledge)
        );
    }

    #[test]
    fn food_product_listing_routes_to_distribution() {
        let d = BridgeDomain::Food;
        assert_eq!(
            resolve_commons_zome(d, "list_product"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_market_listings"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_producer_listings"),
            Some(CommonsZome::FoodDistribution)
        );
    }

    #[test]
    fn transport_trip_routes_to_impact() {
        let d = BridgeDomain::Transport;
        assert_eq!(
            resolve_commons_zome(d, "create_trip"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_trip_carbon"),
            Some(CommonsZome::TransportImpact)
        );
    }

    #[test]
    fn support_comment_action_routes_to_tickets() {
        let d = BridgeDomain::Support;
        assert_eq!(
            resolve_commons_zome(d, "add_comment"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_ticket_comments"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "take_action"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "list_actions"),
            Some(CommonsZome::SupportTickets)
        );
    }

    // -- Exhaustive SDK function name dispatch coverage --

    #[test]
    fn food_client_all_function_names_dispatch_correctly() {
        let d = BridgeDomain::Food;
        // Production (default)
        assert_eq!(
            resolve_commons_zome(d, "register_plot"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_plot"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_all_plots"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "plant_crop"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "record_harvest"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_plot_crops"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_crop_yields"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_season_plan"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_season_plans"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "add_garden_member"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_plot_members"),
            Some(CommonsZome::FoodProduction)
        );
        assert_eq!(
            resolve_commons_zome(d, "remove_garden_member"),
            Some(CommonsZome::FoodProduction)
        );
        // Distribution
        assert_eq!(
            resolve_commons_zome(d, "create_market"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_all_markets"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "list_product"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_market_listings"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_producer_listings"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "place_order"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "fulfill_order"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "cancel_order"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_my_orders"),
            Some(CommonsZome::FoodDistribution)
        );
        assert_eq!(
            resolve_commons_zome(d, "search_allergen_safe"),
            Some(CommonsZome::FoodDistribution)
        );
        // Preservation
        assert_eq!(
            resolve_commons_zome(d, "start_batch"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "complete_batch"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_batch"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "register_storage"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_storage_inventory"),
            Some(CommonsZome::FoodPreservation)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_agent_batches"),
            Some(CommonsZome::FoodPreservation)
        );
        // Knowledge
        assert_eq!(
            resolve_commons_zome(d, "catalog_seed"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_seed"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_seeds_by_species"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "share_practice"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_practices_by_category"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "share_recipe"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_recipes_by_tag"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "search_knowledge"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "offer_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "request_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_available_seeds"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_open_seed_requests"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "match_seed_request"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "rate_seed_exchange"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_exchange_ratings"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_grower_ratings"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "add_nutrient_profile"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_nutrient_profile"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "update_seed_variety"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "update_traditional_practice"),
            Some(CommonsZome::FoodKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "update_recipe"),
            Some(CommonsZome::FoodKnowledge)
        );
    }

    #[test]
    fn transport_client_all_function_names_dispatch_correctly() {
        let d = BridgeDomain::Transport;
        // Routes (default)
        assert_eq!(
            resolve_commons_zome(d, "create_route"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_route"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "log_maintenance"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_vehicle_maintenance"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "set_vehicle_features"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_accessible_vehicles"),
            Some(CommonsZome::TransportRoutes)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_driver_rating"),
            Some(CommonsZome::TransportRoutes)
        );
        // Sharing
        assert_eq!(
            resolve_commons_zome(d, "offer_ride"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "find_nearby_rides"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "review_ride"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_ride_reviews"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "share_vehicle"),
            Some(CommonsZome::TransportSharing)
        );
        assert_eq!(
            resolve_commons_zome(d, "offer_cargo_space"),
            Some(CommonsZome::TransportSharing)
        );
        // Impact
        assert_eq!(
            resolve_commons_zome(d, "log_trip"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_trip_carbon"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "record_emission"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_carbon_balance"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "redeem_credits"),
            Some(CommonsZome::TransportImpact)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_my_redemptions"),
            Some(CommonsZome::TransportImpact)
        );
    }

    #[test]
    fn support_client_all_function_names_dispatch_correctly() {
        let d = BridgeDomain::Support;
        // Knowledge (default)
        assert_eq!(
            resolve_commons_zome(d, "create_article"),
            Some(CommonsZome::SupportKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "search_by_category"),
            Some(CommonsZome::SupportKnowledge)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_resolution"),
            Some(CommonsZome::SupportKnowledge)
        );
        // Tickets
        assert_eq!(
            resolve_commons_zome(d, "create_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "escalate_ticket"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_preemptive_alert"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "submit_satisfaction"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "create_undo"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "add_comment"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "get_ticket_comments"),
            Some(CommonsZome::SupportTickets)
        );
        assert_eq!(
            resolve_commons_zome(d, "take_action"),
            Some(CommonsZome::SupportTickets)
        );
        // Diagnostics
        assert_eq!(
            resolve_commons_zome(d, "run_diagnostic"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "register_helper"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "update_availability"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "publish_cognitive_update"),
            Some(CommonsZome::SupportDiagnostics)
        );
        assert_eq!(
            resolve_commons_zome(d, "set_privacy_preference"),
            Some(CommonsZome::SupportDiagnostics)
        );
    }
}
