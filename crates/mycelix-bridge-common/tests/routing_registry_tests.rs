// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for the routing registry.
//!
//! The inline tests in routing_registry.rs cover zome counts.
//! These tests cover the routing logic: is_allowed, get_local_zomes,
//! cross-cluster dispatch correctness, and security invariants.

use mycelix_bridge_common::routing::CrossClusterRole;
use mycelix_bridge_common::routing_registry::*;

// ============================================================================
// is_allowed — positive cases
// ============================================================================

#[test]
fn test_commons_to_civic_allowed_zomes() {
    // Justice zomes should be reachable from Commons
    assert!(is_allowed(CrossClusterRole::Commons, CrossClusterRole::Civic, "justice_cases"));
    assert!(is_allowed(CrossClusterRole::Commons, CrossClusterRole::Civic, "emergency_incidents"));
}

#[test]
fn test_civic_to_commons_allowed_zomes() {
    // Civic can reach commons property, water, care etc.
    assert!(is_allowed(CrossClusterRole::Civic, CrossClusterRole::Commons, "property_registry"));
    assert!(is_allowed(CrossClusterRole::Civic, CrossClusterRole::Commons, "water_flow"));
}

#[test]
fn test_commons_to_identity_allowed() {
    assert!(is_allowed(CrossClusterRole::Commons, CrossClusterRole::Identity, "did_registry"));
}

#[test]
fn test_commons_to_finance_allowed() {
    assert!(is_allowed(CrossClusterRole::Commons, CrossClusterRole::Finance, "payments"));
}

// ============================================================================
// is_allowed — negative cases (security)
// ============================================================================

#[test]
fn test_self_to_self_not_allowed() {
    // Self-referential cross-cluster calls should return empty
    assert!(!is_allowed(CrossClusterRole::Commons, CrossClusterRole::Commons, "property_registry"));
    assert!(!is_allowed(CrossClusterRole::Civic, CrossClusterRole::Civic, "justice_cases"));
}

#[test]
fn test_unknown_zome_not_allowed() {
    assert!(!is_allowed(CrossClusterRole::Commons, CrossClusterRole::Civic, "nonexistent_zome"));
    assert!(!is_allowed(CrossClusterRole::Commons, CrossClusterRole::Civic, ""));
    assert!(!is_allowed(CrossClusterRole::Commons, CrossClusterRole::Identity, "admin_override"));
}

#[test]
fn test_reverse_route_not_symmetric() {
    // Route from A->B existing doesn't mean B->A allows the same zomes
    let a_to_b = get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Civic);
    let b_to_a = get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Commons);
    // These should be different allowlists
    assert_ne!(a_to_b.len(), b_to_a.len(), "Routes should not be symmetric");
}

// ============================================================================
// get_local_zomes
// ============================================================================

#[test]
fn test_local_zomes_commons() {
    let zomes = get_local_zomes(CrossClusterRole::Commons).expect("Commons should have local zomes");
    assert!(zomes.contains(&"property_registry"));
    assert!(zomes.contains(&"water_flow"));
    assert!(zomes.contains(&"care_timebank"));
}

#[test]
fn test_local_zomes_civic() {
    let zomes = get_local_zomes(CrossClusterRole::Civic).expect("Civic should have local zomes");
    assert!(zomes.contains(&"justice_cases"));
    assert!(zomes.contains(&"emergency_incidents"));
    assert!(zomes.contains(&"mediation"));
}

#[test]
fn test_local_zomes_hearth() {
    let zomes = get_local_zomes(CrossClusterRole::Hearth).expect("Hearth should have local zomes");
    assert!(zomes.contains(&"hearth_kinship"));
    assert!(zomes.contains(&"hearth_gratitude"));
}

#[test]
fn test_local_zomes_personal() {
    let zomes = get_local_zomes(CrossClusterRole::Personal).expect("Personal should have local zomes");
    assert!(zomes.contains(&"identity_vault"));
    assert!(zomes.contains(&"health_vault"));
    assert!(zomes.contains(&"credential_wallet"));
    assert_eq!(zomes.len(), 3);
}

#[test]
fn test_local_zomes_music() {
    let zomes = get_local_zomes(CrossClusterRole::Music).expect("Music should have local zomes");
    assert!(zomes.contains(&"catalog"));
    assert!(zomes.contains(&"plays"));
}

#[test]
fn test_local_zomes_health() {
    let zomes = get_local_zomes(CrossClusterRole::Health).expect("Health should have local zomes");
    assert!(zomes.contains(&"patient"));
    assert!(zomes.contains(&"consent"));
    assert!(zomes.contains(&"telehealth"));
}

// ============================================================================
// Commons sub-cluster partitioning
// ============================================================================

#[test]
fn test_commons_land_contains_physical_infra() {
    assert!(COMMONS_LAND_ZOMES.contains(&"property_registry"));
    assert!(COMMONS_LAND_ZOMES.contains(&"housing_units"));
    assert!(COMMONS_LAND_ZOMES.contains(&"water_flow"));
    assert!(COMMONS_LAND_ZOMES.contains(&"food_production"));
}

#[test]
fn test_commons_care_contains_social_infra() {
    assert!(COMMONS_CARE_ZOMES.contains(&"care_timebank"));
    assert!(COMMONS_CARE_ZOMES.contains(&"mutualaid_needs"));
    assert!(COMMONS_CARE_ZOMES.contains(&"transport_routes"));
}

#[test]
fn test_commons_partition_no_overlap() {
    for zome in COMMONS_LAND_ZOMES {
        assert!(
            !COMMONS_CARE_ZOMES.contains(zome),
            "Zome '{}' should not be in both LAND and CARE",
            zome
        );
    }
}

#[test]
fn test_commons_partition_complete() {
    // Every local zome should be in exactly one sub-cluster
    for zome in COMMONS_LOCAL_ZOMES {
        let in_land = COMMONS_LAND_ZOMES.contains(zome);
        let in_care = COMMONS_CARE_ZOMES.contains(zome);
        assert!(
            in_land || in_care,
            "Zome '{}' missing from both sub-clusters",
            zome
        );
        assert!(
            !(in_land && in_care),
            "Zome '{}' in both sub-clusters",
            zome
        );
    }
}

// ============================================================================
// Local zome uniqueness (no duplicates)
// ============================================================================

#[test]
fn test_no_duplicate_local_zomes_commons() {
    let mut sorted: Vec<&str> = COMMONS_LOCAL_ZOMES.to_vec();
    sorted.sort();
    let len_before = sorted.len();
    sorted.dedup();
    assert_eq!(sorted.len(), len_before, "Commons has duplicate local zomes");
}

#[test]
fn test_no_duplicate_local_zomes_civic() {
    let mut sorted: Vec<&str> = CIVIC_LOCAL_ZOMES.to_vec();
    sorted.sort();
    let len_before = sorted.len();
    sorted.dedup();
    assert_eq!(sorted.len(), len_before, "Civic has duplicate local zomes");
}

#[test]
fn test_no_duplicate_local_zomes_health() {
    let mut sorted: Vec<&str> = HEALTH_LOCAL_ZOMES.to_vec();
    sorted.sort();
    let len_before = sorted.len();
    sorted.dedup();
    assert_eq!(sorted.len(), len_before, "Health has duplicate local zomes");
}

// ============================================================================
// All clusters have local zomes
// ============================================================================

#[test]
fn test_all_clusters_have_local_zomes() {
    let clusters = [
        CrossClusterRole::Commons,
        CrossClusterRole::Civic,
        CrossClusterRole::Hearth,
        CrossClusterRole::Personal,
        CrossClusterRole::Music,
        CrossClusterRole::Health,
        CrossClusterRole::Energy,
        CrossClusterRole::Knowledge,
        CrossClusterRole::Climate,
        CrossClusterRole::Identity,
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
    ];

    for cluster in &clusters {
        let zomes = get_local_zomes(*cluster);
        assert!(
            zomes.is_some(),
            "Cluster {:?} should have local zomes",
            cluster
        );
        assert!(
            !zomes.unwrap().is_empty(),
            "Cluster {:?} has empty local zomes",
            cluster
        );
    }
}
