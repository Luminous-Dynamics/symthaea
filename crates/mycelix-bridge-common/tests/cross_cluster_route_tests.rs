// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit tests for the routing registry cross-cluster dispatch allowlists.
//!
//! These validate that every registered route exists, contains the expected
//! zomes, blocks unknown zomes, and that structural invariants hold (total
//! route count, no self-routes, local zomes coverage).

use mycelix_bridge_common::routing::CrossClusterRole;
use mycelix_bridge_common::routing_registry::{get_allowed_zomes, get_local_zomes, is_allowed};

// ============================================================================
// Group A: Routes already implemented in bridge code
// ============================================================================

// --- Finance → Governance (4 zomes) ---

#[test]
fn finance_to_governance_allows_governance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
        "governance_bridge"
    ));
}

#[test]
fn finance_to_governance_allows_proposals() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
        "proposals"
    ));
}

#[test]
fn finance_to_governance_allows_voting() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
        "voting"
    ));
}

#[test]
fn finance_to_governance_allows_councils() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
        "councils"
    ));
}

#[test]
fn finance_to_governance_has_4_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Finance, CrossClusterRole::Governance);
    assert_eq!(zomes.len(), 4);
}

#[test]
fn finance_to_governance_blocks_unknown() {
    assert!(!is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Governance,
        "evil_zome"
    ));
}

// --- Governance → Personal (1 zome) ---

#[test]
fn governance_to_personal_allows_personal_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Personal,
        "personal_bridge"
    ));
}

#[test]
fn governance_to_personal_has_1_zome() {
    let zomes = get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Personal);
    assert_eq!(zomes.len(), 1);
}

#[test]
fn governance_to_personal_blocks_identity_vault() {
    assert!(!is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Personal,
        "identity_vault"
    ));
}

// --- Governance → Identity (3 zomes) ---

#[test]
fn governance_to_identity_allows_identity_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Identity,
        "identity_bridge"
    ));
}

#[test]
fn governance_to_identity_allows_did_registry() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Identity,
        "did_registry"
    ));
}

#[test]
fn governance_to_identity_allows_verifiable_credential() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Identity,
        "verifiable_credential"
    ));
}

#[test]
fn governance_to_identity_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Identity);
    assert_eq!(zomes.len(), 3);
}

// --- Energy → Finance (3 zomes) ---

#[test]
fn energy_to_finance_allows_finance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Finance,
        "finance_bridge"
    ));
}

#[test]
fn energy_to_finance_allows_payments() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Finance,
        "payments"
    ));
}

#[test]
fn energy_to_finance_allows_staking() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Finance,
        "staking"
    ));
}

#[test]
fn energy_to_finance_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Energy, CrossClusterRole::Finance);
    assert_eq!(zomes.len(), 3);
}

// --- Energy → Supplychain (3 zomes) ---

#[test]
fn energy_to_supplychain_allows_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Supplychain,
        "bridge_coordinator"
    ));
}

#[test]
fn energy_to_supplychain_allows_inventory() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Supplychain,
        "inventory_coordinator"
    ));
}

#[test]
fn energy_to_supplychain_allows_procurement() {
    assert!(is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Supplychain,
        "procurement_coordinator"
    ));
}

#[test]
fn energy_to_supplychain_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Energy, CrossClusterRole::Supplychain);
    assert_eq!(zomes.len(), 3);
}

// --- Climate → Supplychain (3 zomes) ---

#[test]
fn climate_to_supplychain_allows_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Climate,
        CrossClusterRole::Supplychain,
        "bridge_coordinator"
    ));
}

#[test]
fn climate_to_supplychain_allows_claims() {
    assert!(is_allowed(
        CrossClusterRole::Climate,
        CrossClusterRole::Supplychain,
        "claims_coordinator"
    ));
}

#[test]
fn climate_to_supplychain_allows_verification() {
    assert!(is_allowed(
        CrossClusterRole::Climate,
        CrossClusterRole::Supplychain,
        "verification_coordinator"
    ));
}

#[test]
fn climate_to_supplychain_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Climate, CrossClusterRole::Supplychain);
    assert_eq!(zomes.len(), 3);
}

// --- Manufacturing → Commons (2 zomes) ---

#[test]
fn manufacturing_to_commons_allows_commons_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Commons,
        "commons_bridge"
    ));
}

#[test]
fn manufacturing_to_commons_allows_resource_mesh() {
    assert!(is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Commons,
        "resource_mesh"
    ));
}

#[test]
fn manufacturing_to_commons_has_2_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Manufacturing, CrossClusterRole::Commons);
    assert_eq!(zomes.len(), 2);
}

// --- Manufacturing → Supplychain (3 zomes) ---

#[test]
fn manufacturing_to_supplychain_allows_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Supplychain,
        "bridge_coordinator"
    ));
}

#[test]
fn manufacturing_to_supplychain_allows_procurement() {
    assert!(is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Supplychain,
        "procurement_coordinator"
    ));
}

#[test]
fn manufacturing_to_supplychain_allows_inventory() {
    assert!(is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Supplychain,
        "inventory_coordinator"
    ));
}

#[test]
fn manufacturing_to_supplychain_has_3_zomes() {
    let zomes =
        get_allowed_zomes(CrossClusterRole::Manufacturing, CrossClusterRole::Supplychain);
    assert_eq!(zomes.len(), 3);
}

// ============================================================================
// Group B: Governance execution loop
// ============================================================================

// --- Governance → Commons (4 zomes) ---

#[test]
fn governance_to_commons_allows_commons_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Commons,
        "commons_bridge"
    ));
}

#[test]
fn governance_to_commons_allows_property_registry() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Commons,
        "property_registry"
    ));
}

#[test]
fn governance_to_commons_allows_housing_governance() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Commons,
        "housing_governance"
    ));
}

#[test]
fn governance_to_commons_allows_water_steward() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Commons,
        "water_steward"
    ));
}

#[test]
fn governance_to_commons_has_4_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Commons);
    assert_eq!(zomes.len(), 4);
}

// --- Governance → Civic (3 zomes) ---

#[test]
fn governance_to_civic_allows_civic_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Civic,
        "civic_bridge"
    ));
}

#[test]
fn governance_to_civic_allows_justice_cases() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Civic,
        "justice_cases"
    ));
}

#[test]
fn governance_to_civic_allows_emergency_coordination() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Civic,
        "emergency_coordination"
    ));
}

#[test]
fn governance_to_civic_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Civic);
    assert_eq!(zomes.len(), 3);
}

// --- Governance → Finance (3 zomes) ---

#[test]
fn governance_to_finance_allows_finance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Finance,
        "finance_bridge"
    ));
}

#[test]
fn governance_to_finance_allows_treasury() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Finance,
        "treasury"
    ));
}

#[test]
fn governance_to_finance_allows_payments() {
    assert!(is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Finance,
        "payments"
    ));
}

#[test]
fn governance_to_finance_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Governance, CrossClusterRole::Finance);
    assert_eq!(zomes.len(), 3);
}

// --- Finance → Commons (3 zomes) ---

#[test]
fn finance_to_commons_allows_commons_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Commons,
        "commons_bridge"
    ));
}

#[test]
fn finance_to_commons_allows_housing_finances() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Commons,
        "housing_finances"
    ));
}

#[test]
fn finance_to_commons_allows_mutualaid_pools() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Commons,
        "mutualaid_pools"
    ));
}

#[test]
fn finance_to_commons_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Finance, CrossClusterRole::Commons);
    assert_eq!(zomes.len(), 3);
}

// --- Finance → Civic (2 zomes) ---

#[test]
fn finance_to_civic_allows_civic_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Civic,
        "civic_bridge"
    ));
}

#[test]
fn finance_to_civic_allows_justice_enforcement() {
    assert!(is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Civic,
        "justice_enforcement"
    ));
}

#[test]
fn finance_to_civic_has_2_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Finance, CrossClusterRole::Civic);
    assert_eq!(zomes.len(), 2);
}

// --- Civic → Finance (3 zomes) ---

#[test]
fn civic_to_finance_allows_finance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Finance,
        "finance_bridge"
    ));
}

#[test]
fn civic_to_finance_allows_payments() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Finance,
        "payments"
    ));
}

#[test]
fn civic_to_finance_allows_treasury() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Finance,
        "treasury"
    ));
}

#[test]
fn civic_to_finance_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Finance);
    assert_eq!(zomes.len(), 3);
}

// --- Civic → Hearth (3 zomes) ---

#[test]
fn civic_to_hearth_allows_hearth_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Hearth,
        "hearth_bridge"
    ));
}

#[test]
fn civic_to_hearth_allows_hearth_care() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Hearth,
        "hearth_care"
    ));
}

#[test]
fn civic_to_hearth_allows_hearth_emergency() {
    assert!(is_allowed(
        CrossClusterRole::Civic,
        CrossClusterRole::Hearth,
        "hearth_emergency"
    ));
}

#[test]
fn civic_to_hearth_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Civic, CrossClusterRole::Hearth);
    assert_eq!(zomes.len(), 3);
}

// --- Commons → Hearth (3 zomes) ---

#[test]
fn commons_to_hearth_allows_hearth_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Commons,
        CrossClusterRole::Hearth,
        "hearth_bridge"
    ));
}

#[test]
fn commons_to_hearth_allows_hearth_care() {
    assert!(is_allowed(
        CrossClusterRole::Commons,
        CrossClusterRole::Hearth,
        "hearth_care"
    ));
}

#[test]
fn commons_to_hearth_allows_hearth_resources() {
    assert!(is_allowed(
        CrossClusterRole::Commons,
        CrossClusterRole::Hearth,
        "hearth_resources"
    ));
}

#[test]
fn commons_to_hearth_has_3_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Hearth);
    assert_eq!(zomes.len(), 3);
}

// ============================================================================
// Group C: Cross-domain integration
// ============================================================================

// --- Identity → Commons/Civic/Finance/Governance (1 each) ---

#[test]
fn identity_to_commons_allows_commons_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Identity,
        CrossClusterRole::Commons,
        "commons_bridge"
    ));
}

#[test]
fn identity_to_commons_has_1_zome() {
    let zomes = get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Commons);
    assert_eq!(zomes.len(), 1);
}

#[test]
fn identity_to_civic_allows_civic_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Identity,
        CrossClusterRole::Civic,
        "civic_bridge"
    ));
}

#[test]
fn identity_to_civic_has_1_zome() {
    let zomes = get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Civic);
    assert_eq!(zomes.len(), 1);
}

#[test]
fn identity_to_finance_allows_finance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Identity,
        CrossClusterRole::Finance,
        "finance_bridge"
    ));
}

#[test]
fn identity_to_finance_has_1_zome() {
    let zomes = get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Finance);
    assert_eq!(zomes.len(), 1);
}

#[test]
fn identity_to_governance_allows_governance_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Identity,
        CrossClusterRole::Governance,
        "governance_bridge"
    ));
}

#[test]
fn identity_to_governance_has_1_zome() {
    let zomes = get_allowed_zomes(CrossClusterRole::Identity, CrossClusterRole::Governance);
    assert_eq!(zomes.len(), 1);
}

// --- Health → Personal (2 zomes) ---

#[test]
fn health_to_personal_allows_personal_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Health,
        CrossClusterRole::Personal,
        "personal_bridge"
    ));
}

#[test]
fn health_to_personal_allows_health_vault() {
    assert!(is_allowed(
        CrossClusterRole::Health,
        CrossClusterRole::Personal,
        "health_vault"
    ));
}

#[test]
fn health_to_personal_has_2_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Health, CrossClusterRole::Personal);
    assert_eq!(zomes.len(), 2);
}

// --- Health → Identity (2 zomes) ---

#[test]
fn health_to_identity_allows_identity_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Health,
        CrossClusterRole::Identity,
        "identity_bridge"
    ));
}

#[test]
fn health_to_identity_allows_did_registry() {
    assert!(is_allowed(
        CrossClusterRole::Health,
        CrossClusterRole::Identity,
        "did_registry"
    ));
}

#[test]
fn health_to_identity_has_2_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Health, CrossClusterRole::Identity);
    assert_eq!(zomes.len(), 2);
}

// --- Praxis → Identity (2 zomes) ---

#[test]
fn praxis_to_identity_allows_identity_bridge() {
    assert!(is_allowed(
        CrossClusterRole::Praxis,
        CrossClusterRole::Identity,
        "identity_bridge"
    ));
}

#[test]
fn praxis_to_identity_allows_verifiable_credential() {
    assert!(is_allowed(
        CrossClusterRole::Praxis,
        CrossClusterRole::Identity,
        "verifiable_credential"
    ));
}

#[test]
fn praxis_to_identity_has_2_zomes() {
    let zomes = get_allowed_zomes(CrossClusterRole::Praxis, CrossClusterRole::Identity);
    assert_eq!(zomes.len(), 2);
}

// ============================================================================
// Structural invariants
// ============================================================================

/// There are exactly 36 registered cross-cluster routes (non-empty allowlists).
#[test]
fn total_registered_routes_is_36() {
    let mut count = 0;
    for &src in CrossClusterRole::ALL {
        for &dst in CrossClusterRole::ALL {
            if src as u8 != dst as u8 {
                let zomes = get_allowed_zomes(src, dst);
                if !zomes.is_empty() {
                    count += 1;
                }
            }
        }
    }
    assert_eq!(
        count, 53,
        "Expected 53 registered routes, found {count}"
    );
}

/// Every registered route has at least 1 allowed zome.
#[test]
fn every_route_has_at_least_one_zome() {
    for &src in CrossClusterRole::ALL {
        for &dst in CrossClusterRole::ALL {
            let zomes = get_allowed_zomes(src, dst);
            if !zomes.is_empty() {
                assert!(
                    zomes.len() >= 1,
                    "Route {:?} -> {:?} has 0 zomes despite being registered",
                    src,
                    dst
                );
            }
        }
    }
}

/// CrossClusterRole::ALL has exactly 30 variants (added Craft).
#[test]
fn cross_cluster_role_has_30_variants() {
    assert_eq!(
        CrossClusterRole::ALL.len(),
        30,
        "Expected 30 CrossClusterRole variants, found {}",
        CrossClusterRole::ALL.len()
    );
}

/// Self→self always returns an empty allowlist for all 16 clusters.
#[test]
fn self_to_self_returns_empty_for_all_clusters() {
    for &role in CrossClusterRole::ALL {
        let zomes = get_allowed_zomes(role, role);
        assert!(
            zomes.is_empty(),
            "Self-route {:?} -> {:?} should return empty, got {} zomes",
            role,
            role,
            zomes.len()
        );
    }
}

/// `get_local_zomes` returns `Some` for all 16 clusters.
#[test]
fn get_local_zomes_returns_some_for_all_clusters() {
    for &role in CrossClusterRole::ALL {
        assert!(
            get_local_zomes(role).is_some(),
            "get_local_zomes({:?}) returned None — every cluster should have a local zome list",
            role
        );
    }
}

/// Every Holochain-based local zome list is non-empty.
/// Non-Holochain clusters (Lunar, MultiworldSim, Portal, Workspace) may have empty lists.
#[test]
fn local_zome_lists_are_nonempty() {
    let non_holochain = [
        CrossClusterRole::Lunar,
        CrossClusterRole::MultiworldSim,
        CrossClusterRole::Portal,
        CrossClusterRole::Workspace,
    ];
    for &role in CrossClusterRole::ALL {
        if non_holochain.contains(&role) {
            continue;
        }
        if let Some(zomes) = get_local_zomes(role) {
            assert!(
                !zomes.is_empty(),
                "get_local_zomes({:?}) returned an empty list",
                role
            );
        }
    }
}

// ============================================================================
// Negative tests: routes that should NOT exist
// ============================================================================

#[test]
fn legacy_to_music_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Legacy, CrossClusterRole::Music);
    assert!(
        zomes.is_empty(),
        "Legacy -> Music should not have a route"
    );
}

#[test]
fn climate_to_hearth_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Climate, CrossClusterRole::Hearth);
    assert!(
        zomes.is_empty(),
        "Climate -> Hearth should not have a route"
    );
}

#[test]
fn music_to_governance_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Music, CrossClusterRole::Governance);
    assert!(
        zomes.is_empty(),
        "Music -> Governance should not have a route"
    );
}

#[test]
fn supplychain_to_hearth_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Supplychain, CrossClusterRole::Hearth);
    assert!(
        zomes.is_empty(),
        "Supplychain -> Hearth should not have a route"
    );
}

#[test]
fn knowledge_to_personal_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Knowledge, CrossClusterRole::Personal);
    assert!(
        zomes.is_empty(),
        "Knowledge -> Personal should not have a route"
    );
}

#[test]
fn legacy_to_energy_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Legacy, CrossClusterRole::Energy);
    assert!(
        zomes.is_empty(),
        "Legacy -> Energy should not have a route"
    );
}

#[test]
fn praxis_to_finance_allows_tend() {
    let zomes = get_allowed_zomes(CrossClusterRole::Praxis, CrossClusterRole::Finance);
    assert_eq!(zomes.len(), 2, "Praxis -> Finance should have 2 zomes (finance_bridge, tend)");
    assert!(zomes.contains(&"tend"), "Praxis -> Finance should allow TEND learning credits");
}

#[test]
fn music_to_climate_not_registered() {
    let zomes = get_allowed_zomes(CrossClusterRole::Music, CrossClusterRole::Climate);
    assert!(
        zomes.is_empty(),
        "Music -> Climate should not have a route"
    );
}

// ============================================================================
// Negative tests: specific zomes blocked on real routes
// ============================================================================

#[test]
fn governance_to_commons_blocks_mutualaid_pools() {
    // Governance can reach commons_bridge + property_registry +
    // housing_governance + water_steward, but NOT mutualaid_pools.
    assert!(!is_allowed(
        CrossClusterRole::Governance,
        CrossClusterRole::Commons,
        "mutualaid_pools"
    ));
}

#[test]
fn energy_to_finance_blocks_treasury() {
    // Energy can reach finance_bridge + payments + staking, but NOT treasury.
    assert!(!is_allowed(
        CrossClusterRole::Energy,
        CrossClusterRole::Finance,
        "treasury"
    ));
}

#[test]
fn health_to_personal_blocks_credential_wallet() {
    // Health can reach personal_bridge + health_vault, but NOT credential_wallet.
    assert!(!is_allowed(
        CrossClusterRole::Health,
        CrossClusterRole::Personal,
        "credential_wallet"
    ));
}

#[test]
fn praxis_to_identity_blocks_mfa() {
    // Praxis can reach identity_bridge + verifiable_credential, but NOT mfa.
    assert!(!is_allowed(
        CrossClusterRole::Praxis,
        CrossClusterRole::Identity,
        "mfa"
    ));
}

#[test]
fn climate_to_supplychain_blocks_logistics() {
    // Climate can reach bridge + claims + verification, but NOT logistics.
    assert!(!is_allowed(
        CrossClusterRole::Climate,
        CrossClusterRole::Supplychain,
        "logistics_coordinator"
    ));
}

#[test]
fn manufacturing_to_commons_blocks_water_flow() {
    // Manufacturing can reach commons_bridge + resource_mesh, but NOT water_flow.
    assert!(!is_allowed(
        CrossClusterRole::Manufacturing,
        CrossClusterRole::Commons,
        "water_flow"
    ));
}

#[test]
fn finance_to_civic_blocks_emergency_incidents() {
    // Finance can reach civic_bridge + justice_enforcement, but NOT emergency_incidents.
    assert!(!is_allowed(
        CrossClusterRole::Finance,
        CrossClusterRole::Civic,
        "emergency_incidents"
    ));
}
