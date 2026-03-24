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
];

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
pub const PERSONAL_LOCAL_ZOMES: &[&str] = &[
    "identity_vault",
    "health_vault",
    "credential_wallet",
];

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
];

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
const PERSONAL_TO_IDENTITY: &[&str] = &[
    "identity_bridge",
    "did_registry",
    "verifiable_credential",
];

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
pub const fn get_allowed_zomes(initiator: CrossClusterRole, target: CrossClusterRole) -> &'static [&'static str] {
    match (initiator, target) {
        // Commons outbound
        (CrossClusterRole::Commons, CrossClusterRole::Civic) => COMMONS_TO_CIVIC,
        (CrossClusterRole::Commons, CrossClusterRole::Identity) => COMMONS_TO_IDENTITY,
        (CrossClusterRole::Commons, CrossClusterRole::Finance) => COMMONS_TO_FINANCE,

        // Civic outbound
        (CrossClusterRole::Civic, CrossClusterRole::Commons) => CIVIC_TO_COMMONS,
        (CrossClusterRole::Civic, CrossClusterRole::Identity) => CIVIC_TO_IDENTITY,

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
        _ => None,
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
        assert_eq!(COMMONS_LOCAL_ZOMES.len(), 38);
    }

    #[test]
    fn civic_local_zome_count() {
        assert_eq!(CIVIC_LOCAL_ZOMES.len(), 16);
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
        assert_eq!(COMMONS_CARE_ZOMES.len(), 19);
    }

    #[test]
    fn commons_sub_clusters_partition_local() {
        // LAND + CARE must equal LOCAL (no overlap, no gaps)
        let mut combined: Vec<&&str> = COMMONS_LAND_ZOMES.iter().chain(COMMONS_CARE_ZOMES.iter()).collect();
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
    fn is_not_allowed_civic_to_finance() {
        // No civic→finance route is registered
        assert!(!is_allowed(
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
        assert_eq!(role_name(CrossClusterRole::Finance), "finance");
        assert_eq!(role_name(CrossClusterRole::Governance), "governance");
    }

    // ---- No duplicates in any allowlist ----

    #[test]
    fn no_duplicates_in_any_allowlist() {
        let lists: &[(&str, &[&str])] = &[
            ("COMMONS_LOCAL_ZOMES", COMMONS_LOCAL_ZOMES),
            ("CIVIC_LOCAL_ZOMES", CIVIC_LOCAL_ZOMES),
            ("HEARTH_LOCAL_ZOMES", HEARTH_LOCAL_ZOMES),
            ("PERSONAL_LOCAL_ZOMES", PERSONAL_LOCAL_ZOMES),
            ("COMMONS_LAND_ZOMES", COMMONS_LAND_ZOMES),
            ("COMMONS_CARE_ZOMES", COMMONS_CARE_ZOMES),
            ("COMMONS_TO_CIVIC", COMMONS_TO_CIVIC),
            ("COMMONS_TO_IDENTITY", COMMONS_TO_IDENTITY),
            ("COMMONS_TO_FINANCE", COMMONS_TO_FINANCE),
            ("CIVIC_TO_COMMONS", CIVIC_TO_COMMONS),
            ("CIVIC_TO_IDENTITY", CIVIC_TO_IDENTITY),
            ("HEARTH_TO_PERSONAL", HEARTH_TO_PERSONAL),
            ("HEARTH_TO_IDENTITY", HEARTH_TO_IDENTITY),
            ("HEARTH_TO_COMMONS", HEARTH_TO_COMMONS),
            ("HEARTH_TO_CIVIC", HEARTH_TO_CIVIC),
            ("PERSONAL_TO_COMMONS", PERSONAL_TO_COMMONS),
            ("PERSONAL_TO_CIVIC", PERSONAL_TO_CIVIC),
            ("PERSONAL_TO_GOVERNANCE", PERSONAL_TO_GOVERNANCE),
            ("PERSONAL_TO_IDENTITY", PERSONAL_TO_IDENTITY),
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
        let lists: &[(&str, &[&str])] = &[
            ("COMMONS_LOCAL_ZOMES", COMMONS_LOCAL_ZOMES),
            ("CIVIC_LOCAL_ZOMES", CIVIC_LOCAL_ZOMES),
            ("HEARTH_LOCAL_ZOMES", HEARTH_LOCAL_ZOMES),
            ("PERSONAL_LOCAL_ZOMES", PERSONAL_LOCAL_ZOMES),
            ("COMMONS_TO_CIVIC", COMMONS_TO_CIVIC),
            ("COMMONS_TO_IDENTITY", COMMONS_TO_IDENTITY),
            ("COMMONS_TO_FINANCE", COMMONS_TO_FINANCE),
            ("CIVIC_TO_COMMONS", CIVIC_TO_COMMONS),
            ("CIVIC_TO_IDENTITY", CIVIC_TO_IDENTITY),
            ("HEARTH_TO_PERSONAL", HEARTH_TO_PERSONAL),
            ("HEARTH_TO_IDENTITY", HEARTH_TO_IDENTITY),
            ("HEARTH_TO_COMMONS", HEARTH_TO_COMMONS),
            ("HEARTH_TO_CIVIC", HEARTH_TO_CIVIC),
            ("PERSONAL_TO_COMMONS", PERSONAL_TO_COMMONS),
            ("PERSONAL_TO_CIVIC", PERSONAL_TO_CIVIC),
            ("PERSONAL_TO_GOVERNANCE", PERSONAL_TO_GOVERNANCE),
            ("PERSONAL_TO_IDENTITY", PERSONAL_TO_IDENTITY),
        ];
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
        assert_eq!(get_local_zomes(CrossClusterRole::Commons).unwrap().len(), 38);
        assert_eq!(get_local_zomes(CrossClusterRole::Civic).unwrap().len(), 16);
        assert_eq!(get_local_zomes(CrossClusterRole::Hearth).unwrap().len(), 10);
        assert_eq!(get_local_zomes(CrossClusterRole::Personal).unwrap().len(), 3);
        assert!(get_local_zomes(CrossClusterRole::Identity).is_none());
        assert!(get_local_zomes(CrossClusterRole::Finance).is_none());
        assert!(get_local_zomes(CrossClusterRole::Governance).is_none());
    }

    // ---- Empty result for unregistered routes ----

    #[test]
    fn unregistered_routes_return_empty() {
        // Spot-check several unregistered routes
        assert!(get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Commons).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Finance, CrossClusterRole::Commons).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Personal).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Hearth).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Hearth).is_empty());
        assert!(get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Personal).is_empty());
    }
}
