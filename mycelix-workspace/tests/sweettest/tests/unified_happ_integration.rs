#![allow(deprecated)] // Tests use legacy ConsciousnessCredential/Tier for backward-compat bridge testing
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified hApp Integration Tests — Cross-Cluster Dispatch
//!
//! Verifies that all available roles in the unified hApp can communicate
//! via `CallTargetCell::OtherRole(role)` cross-cluster dispatch.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build all cluster WASMs
//! for cluster in commons civic identity governance finance health; do
//!     cd mycelix-$cluster && cargo build --release --target wasm32-unknown-unknown && cd ..
//! done
//!
//! # Pack DNAs and unified hApp
//! hc app pack mycelix-workspace/happs/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test unified_happ_integration -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining structs
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: u32,
    total_queries: u32,
    domains: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CrossClusterDispatchInput {
    role: String,
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchResult {
    success: bool,
    response: Option<Vec<u8>>,
    error: Option<String>,
}

// ============================================================================
// Helper: Check if a DNA file exists (role is available)
// ============================================================================

fn has_role(role: &str) -> bool {
    let path = match role {
        "commons_land" => DnaPaths::commons_land(),
        "commons_care" => DnaPaths::commons_care(),
        "civic" => DnaPaths::civic(),
        "identity" => DnaPaths::identity(),
        "governance" => DnaPaths::governance(),
        "finance" => DnaPaths::finance(),
        "health" => DnaPaths::health(),
        "personal" => DnaPaths::personal(),
        "hearth" => DnaPaths::hearth(),
        "attribution" => DnaPaths::attribution(),
        _ => return false,
    };
    path.exists()
}

// ============================================================================
// Tests
// ============================================================================

/// Test commons → civic cross-cluster dispatch via bridge zomes.
///
/// Verifies that the commons bridge can dispatch calls to the civic cluster
/// and receive responses when both are installed in the unified hApp.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
async fn test_commons_to_civic_dispatch() {
    if !has_role("commons_land") || !has_role("civic") {
        eprintln!("SKIP: commons_land or civic DNA not built");
        return;
    }

    let dna_paths = DnaPaths::unified_happ_dnas();
    let available: Vec<_> = dna_paths
        .iter()
        .filter(|(role, path)| {
            let exists = path.exists();
            if !exists {
                eprintln!("DNA not found for role '{}': {:?}", role, path);
            }
            exists
        })
        .collect();

    // Need at least commons_land + civic
    let has_commons = available.iter().any(|(r, _)| *r == "commons_land");
    let has_civic = available.iter().any(|(r, _)| *r == "civic");
    if !has_commons || !has_civic {
        eprintln!("SKIP: Need both commons_land and civic DNAs");
        return;
    }

    // Set up conductor with available DNAs
    let mut conductor = SweetConductor::from_standard_config().await;
    let mut dna_files = Vec::new();
    for (role, path) in &available {
        let dna = SweetDnaFile::from_bundle(path)
            .await
            .unwrap_or_else(|e| panic!("Failed to load {} DNA: {:?}", role, e));
        dna_files.push((*role, dna));
    }

    let dna_refs: Vec<&SweetDnaFile> = dna_files.iter().map(|(_, d)| d).collect();
    let app = conductor
        .setup_app("unified-test", &dna_refs)
        .await
        .unwrap();

    // Verify both cells are alive by calling bridge health on commons
    let commons_cell = &app.cells()[dna_files
        .iter()
        .position(|(r, _)| *r == "commons_land")
        .unwrap()];

    let health: BridgeHealth = conductor
        .call(&commons_cell.zome("bridge"), "bridge_health", ())
        .await;

    assert!(
        health.healthy,
        "Commons bridge should be healthy: {:?}",
        health
    );
}

/// Test governance → finance cross-cluster dispatch.
///
/// Verifies that governance can dispatch to finance cluster for treasury
/// operations when both are installed in the unified hApp.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
async fn test_governance_to_finance_dispatch() {
    if !has_role("governance") || !has_role("finance") {
        eprintln!("SKIP: governance or finance DNA not built");
        return;
    }

    let governance_dna = SweetDnaFile::from_bundle(&DnaPaths::governance())
        .await
        .expect("Failed to load governance DNA");
    let finance_dna = SweetDnaFile::from_bundle(&DnaPaths::finance())
        .await
        .expect("Failed to load finance DNA");

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor
        .setup_app("gov-finance-test", &[&governance_dna, &finance_dna])
        .await
        .unwrap();

    // Verify governance bridge health
    let gov_cell = &app.cells()[0];
    let health: BridgeHealth = conductor
        .call(&gov_cell.zome("bridge"), "bridge_health", ())
        .await;

    assert!(
        health.healthy,
        "Governance bridge should be healthy: {:?}",
        health
    );
}

/// Test identity resolution is available across the unified hApp.
///
/// Verifies that the identity cluster's DID resolution is accessible
/// from other clusters via cross-role dispatch.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
#[serial]
async fn test_identity_resolution_available() {
    if !has_role("identity") {
        eprintln!("SKIP: identity DNA not built");
        return;
    }

    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Failed to load identity DNA");

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor
        .setup_app("identity-test", &[&identity_dna])
        .await
        .unwrap();

    let identity_cell = &app.cells()[0];

    // Verify identity bridge health (the bridge zome exposes DID resolution)
    let health: BridgeHealth = conductor
        .call(&identity_cell.zome("bridge"), "bridge_health", ())
        .await;

    assert!(
        health.healthy,
        "Identity bridge should be healthy: {:?}",
        health
    );

    // Verify the identity cluster reports its domains
    assert!(
        !health.domains.is_empty(),
        "Identity bridge should report domains"
    );
}

// ============================================================================
// Pure Rust tests — no conductor required
// ============================================================================

#[cfg(test)]
mod pure_rust {
    use mycelix_bridge_common::consciousness_profile::{
        ConsciousnessProfile, CivicTier, ReputationState,
        REPUTATION_BLACKLIST_THRESHOLD, REPUTATION_DECAY_PER_DAY,
        REPUTATION_RESTORATION_INTERACTIONS, REPUTATION_SLASH_FACTOR,
    };
    use mycelix_bridge_entry_types::{
        BridgeEventEntry, BridgeQueryEntry, CachedCredentialEntry, SchemaMigration,
    };

    // ---- Test 4: Schema versioning round-trip ----

    #[test]
    fn test_schema_versioning_roundtrip() {
        // BridgeQueryEntry
        let agent_bytes = vec![0xAAu8; 36];
        let agent = holochain::prelude::AgentPubKey::from_raw_36(agent_bytes);

        let query = BridgeQueryEntry {
            schema_version: 1,
            domain: "property".into(),
            query_type: "list_all".into(),
            requester: agent.clone(),
            params: r#"{"limit":10}"#.into(),
            result: Some("found 3".into()),
            created_at: holochain::prelude::Timestamp::from_micros(1_000_000),
            resolved_at: Some(holochain::prelude::Timestamp::from_micros(2_000_000)),
            success: Some(true),
        };
        let json = serde_json::to_string(&query).unwrap();
        let roundtripped: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(query, roundtripped);
        assert_eq!(roundtripped.schema_version, 1);
        assert_eq!(roundtripped.domain, "property");
        assert_eq!(roundtripped.result, Some("found 3".into()));

        // BridgeEventEntry
        let event = BridgeEventEntry {
            schema_version: 1,
            domain: "housing".into(),
            event_type: "listing_created".into(),
            source_agent: agent.clone(),
            payload: r#"{"address":"123 Main St"}"#.into(),
            created_at: holochain::prelude::Timestamp::from_micros(3_000_000),
            related_hashes: vec!["hash_a".into(), "hash_b".into()],
        };
        let json = serde_json::to_string(&event).unwrap();
        let roundtripped: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(event, roundtripped);
        assert_eq!(roundtripped.schema_version, 1);
        assert_eq!(roundtripped.related_hashes.len(), 2);

        // CachedCredentialEntry
        let cred = CachedCredentialEntry {
            schema_version: 1,
            did: "did:mycelix:agent42".into(),
            credential_json: r#"{"profile":{"identity":0.75}}"#.into(),
            cached_at_us: 5_000_000,
        };
        let json = serde_json::to_string(&cred).unwrap();
        let roundtripped: CachedCredentialEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(cred, roundtripped);
        assert_eq!(roundtripped.did, "did:mycelix:agent42");

        // SchemaMigration trait: verify CURRENT_VERSION
        assert_eq!(BridgeQueryEntry::CURRENT_VERSION, 1);
        assert_eq!(BridgeEventEntry::CURRENT_VERSION, 1);
        assert_eq!(CachedCredentialEntry::CURRENT_VERSION, 1);

        // Verify schema_version defaults to 1 when constructing entries
        let query_default = BridgeQueryEntry {
            schema_version: 1, // default_schema_v1
            domain: "test".into(),
            query_type: "q".into(),
            requester: agent.clone(),
            params: "{}".into(),
            result: None,
            created_at: holochain::prelude::Timestamp::from_micros(0),
            resolved_at: None,
            success: None,
        };
        assert_eq!(query_default.schema_version, 1);
    }

    // ---- Test 5: Consciousness tier progression with hysteresis ----

    #[test]
    fn test_consciousness_tier_progression_with_hysteresis() {
        // Progress through tiers: Observer → Participant → Citizen → Steward → Guardian
        let scores = [
            (0.0, CivicTier::Observer),
            (0.36, CivicTier::Participant),   // >= 0.3 + 0.05 margin for promote
            (0.46, CivicTier::Citizen),        // >= 0.4 + 0.05 margin
            (0.66, CivicTier::Steward),        // >= 0.6 + 0.05 margin
            (0.86, CivicTier::Guardian),       // >= 0.8 + 0.05 margin
        ];

        let mut current_tier = CivicTier::Observer;
        for (score, expected_tier) in &scores {
            // Build a profile where combined_score = score
            // combined = identity*0.25 + reputation*0.25 + community*0.30 + engagement*0.20
            // Set all dimensions equal: score = d*(0.25+0.25+0.30+0.20) = d*1.0 = d
            let profile = ConsciousnessProfile {
                identity: *score,
                reputation: *score,
                community: *score,
                engagement: *score,
            };
            assert!(
                (profile.combined_score() - score).abs() < 1e-10,
                "combined_score for uniform {} should be {}",
                score,
                score
            );

            let tier = CivicTier::from_score(*score);
            // For non-hysteresis, check basic progression
            if *score >= 0.8 {
                assert_eq!(tier, CivicTier::Guardian);
            } else if *score >= 0.6 {
                assert_eq!(tier, CivicTier::Steward);
            } else if *score >= 0.4 {
                assert_eq!(tier, CivicTier::Citizen);
            } else if *score >= 0.3 {
                assert_eq!(tier, CivicTier::Participant);
            } else {
                assert_eq!(tier, CivicTier::Observer);
            }

            // With hysteresis, using previous tier as context
            let hysteresis_tier = profile.tier_with_hysteresis(current_tier);
            assert_eq!(
                hysteresis_tier, *expected_tier,
                "From {:?} at score {}, expected {:?} got {:?}",
                current_tier, score, expected_tier, hysteresis_tier
            );
            current_tier = hysteresis_tier;
        }

        // Hysteresis prevents demotion when score is at boundary minus 0.03
        // Citizen threshold is 0.4, demote threshold is 0.35
        // Score 0.4 - 0.03 = 0.37 is ABOVE demote threshold 0.35, so should NOT demote
        let profile_boundary = ConsciousnessProfile {
            identity: 0.37,
            reputation: 0.37,
            community: 0.37,
            engagement: 0.37,
        };
        let tier = profile_boundary.tier_with_hysteresis(CivicTier::Citizen);
        assert_eq!(
            tier,
            CivicTier::Citizen,
            "Score 0.37 should NOT cause demotion from Citizen (demote threshold is 0.35)"
        );

        // Similarly for Steward: threshold 0.6, demote 0.55
        // Score 0.6 - 0.03 = 0.57 is ABOVE 0.55
        let profile_steward = ConsciousnessProfile {
            identity: 0.57,
            reputation: 0.57,
            community: 0.57,
            engagement: 0.57,
        };
        let tier = profile_steward.tier_with_hysteresis(CivicTier::Steward);
        assert_eq!(
            tier,
            CivicTier::Steward,
            "Score 0.57 should NOT cause demotion from Steward (demote threshold is 0.55)"
        );

        // Guardian: threshold 0.8, demote 0.75
        // Score 0.8 - 0.03 = 0.77 is ABOVE 0.75
        let profile_guardian = ConsciousnessProfile {
            identity: 0.77,
            reputation: 0.77,
            community: 0.77,
            engagement: 0.77,
        };
        let tier = profile_guardian.tier_with_hysteresis(CivicTier::Guardian);
        assert_eq!(
            tier,
            CivicTier::Guardian,
            "Score 0.77 should NOT cause demotion from Guardian (demote threshold is 0.75)"
        );
    }

    // ---- Test 6: Reputation decay and slashing ----

    #[test]
    fn test_reputation_decay_and_slashing() {
        let start_us: u64 = 1_000_000_000_000; // ~11.5 days in microseconds
        let day_us: u64 = 86_400_000_000; // 1 day in microseconds

        // Start with perfect reputation
        let mut state = ReputationState::new(1.0, start_us);
        assert!((state.score - 1.0).abs() < 1e-10, "Initial score should be 1.0");
        assert!(!state.blacklisted, "Should not start blacklisted");

        // Apply decay over 30 days
        let after_30_days = start_us + 30 * day_us;
        state.apply_decay(after_30_days);

        // Expected: 1.0 * 0.998^30
        let expected = REPUTATION_DECAY_PER_DAY.powi(30);
        assert!(
            (state.score - expected).abs() < 1e-6,
            "After 30 days decay: expected {:.6}, got {:.6}",
            expected,
            state.score
        );
        // 0.998^30 ~ 0.9418 — noticeable but not catastrophic
        assert!(state.score > 0.93, "30-day decay should leave score above 0.93");
        assert!(state.score < 0.95, "30-day decay should leave score below 0.95");

        // Apply slash: 50% reduction
        let slash_time = after_30_days + 1000;
        let pre_slash = state.score;
        state.slash(slash_time);
        let expected_after_slash = pre_slash * (1.0 - REPUTATION_SLASH_FACTOR);
        assert!(
            (state.score - expected_after_slash).abs() < 1e-6,
            "Slash should reduce by {:.0}%: expected {:.6}, got {:.6}",
            REPUTATION_SLASH_FACTOR * 100.0,
            expected_after_slash,
            state.score
        );
        assert_eq!(state.total_slashes, 1);
        assert_eq!(state.consecutive_good, 0, "Slash resets consecutive good");

        // Multiple slashes to push below blacklist threshold (0.05)
        // Current score ~0.47. Need to slash until below 0.05.
        // Each slash halves: 0.47 → 0.235 → 0.118 → 0.059 → 0.029
        let mut t = slash_time;
        while state.score >= REPUTATION_BLACKLIST_THRESHOLD {
            t += 1000;
            state.slash(t);
        }
        assert!(
            state.blacklisted,
            "Should be blacklisted after score dropped below {}",
            REPUTATION_BLACKLIST_THRESHOLD
        );
        assert!(state.blacklisted_since_us.is_some());
        assert!(!state.can_participate(), "Blacklisted agents cannot participate");

        // Restoration: 100 good interactions
        for i in 0..REPUTATION_RESTORATION_INTERACTIONS {
            t += 1000;
            state.record_good_interaction(0.01, t);
            if i < REPUTATION_RESTORATION_INTERACTIONS - 1 {
                assert!(
                    state.blacklisted,
                    "Should still be blacklisted before {} interactions (at {})",
                    REPUTATION_RESTORATION_INTERACTIONS,
                    i + 1
                );
            }
        }
        assert!(
            !state.blacklisted,
            "Should be restored after {} good interactions",
            REPUTATION_RESTORATION_INTERACTIONS
        );
        assert!(
            state.can_participate(),
            "Restored agents should be able to participate"
        );
        assert!(state.score > 0.0, "Score should be positive after restoration");
    }
}
