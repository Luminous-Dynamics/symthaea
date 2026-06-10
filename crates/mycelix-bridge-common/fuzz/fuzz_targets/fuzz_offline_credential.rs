// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mycelix_bridge_common::consciousness_profile::{
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier,
};
use mycelix_bridge_common::offline_credential::{FreshnessAttestation, OfflineCredential};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Arbitrary input types
// ---------------------------------------------------------------------------

#[derive(Arbitrary, Debug)]
struct FuzzOfflineInput {
    // Credential base
    tier_idx: u8,
    issued_at: u64,
    ttl_us: u64,
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,

    // Offline credential fields
    last_online_verification: u64,
    degradation_enabled: bool,
    custom_grace_hours: Option<u64>,

    // Attestation (optional)
    has_attestation: bool,
    attester_did_bytes: Vec<u8>,
    attestation_timestamp: u64,
    attestation_tier_idx: u8,
    attestation_signature: Vec<u8>,
    attestation_key: [u8; 32],

    // Query times
    query_times: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tier_from_idx(idx: u8) -> ConsciousnessTier {
    match idx % 5 {
        0 => ConsciousnessTier::Observer,
        1 => ConsciousnessTier::Participant,
        2 => ConsciousnessTier::Citizen,
        3 => ConsciousnessTier::Steward,
        _ => ConsciousnessTier::Guardian,
    }
}

fn make_credential(input: &FuzzOfflineInput) -> ConsciousnessCredential {
    ConsciousnessCredential {
        did: "did:mycelix:fuzz".to_string(),
        profile: ConsciousnessProfile {
            identity: input.identity,
            reputation: input.reputation,
            community: input.community,
            engagement: input.engagement,
        },
        tier: tier_from_idx(input.tier_idx),
        issued_at: input.issued_at,
        expires_at: input.issued_at.saturating_add(input.ttl_us),
        issuer: "did:mycelix:fuzz-issuer".to_string(),
        trajectory_commitment: None,
        extensions: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Fuzz target
// ---------------------------------------------------------------------------

fuzz_target!(|input: FuzzOfflineInput| {
    let credential = make_credential(&input);

    // 1. Test OfflineCredential::new — must not panic
    let mut offline = OfflineCredential::new(credential.clone());
    offline.degradation_enabled = input.degradation_enabled;
    offline.last_online_verification = input.last_online_verification;

    // 2. Test custom grace period construction
    if let Some(hours) = input.custom_grace_hours {
        let custom = OfflineCredential::with_grace_hours(credential.clone(), hours);
        // Grace period must be capped at 720 hours (30 days)
        if let Some(gp) = custom.custom_grace_period {
            let max_grace_us = 720 * 3600 * 1_000_000u64;
            assert!(
                gp <= max_grace_us,
                "custom grace {} exceeds 30-day max {}",
                gp,
                max_grace_us
            );
        }

        // Exercise effective_tier on custom-grace credential
        for &t in &input.query_times {
            let _tier = custom.effective_tier(t);
            let _usable = custom.is_usable(t);
            let _hours = custom.hours_offline(t);
        }
    }

    // 3. Test effective_tier with arbitrary timestamps — must not panic
    for &now_us in &input.query_times {
        let tier = offline.effective_tier(now_us);

        // Invariant: effective tier can never exceed the original tier
        assert!(
            tier <= offline.credential.tier,
            "effective tier {:?} > original {:?} at time {}",
            tier,
            offline.credential.tier,
            now_us
        );

        // Invariant: if degradation disabled, tier must equal original
        if !offline.degradation_enabled {
            assert_eq!(
                tier, offline.credential.tier,
                "degradation disabled but tier changed"
            );
        }

        let _usable = offline.is_usable(now_us);
        let _hours = offline.hours_offline(now_us);
    }

    // 4. Test freshness attestation path
    if input.has_attestation {
        let attester_did =
            String::from_utf8_lossy(&input.attester_did_bytes[..input.attester_did_bytes.len().min(64)])
                .to_string();

        let mut attestation = FreshnessAttestation {
            attester_did,
            timestamp: input.attestation_timestamp,
            tier_at_attestation: tier_from_idx(input.attestation_tier_idx),
            signature: input.attestation_signature.clone(),
        };

        // Exercise canonical_bytes — must not panic
        let _canonical = attestation.canonical_bytes();

        // Exercise sign + verify round-trip
        attestation.sign_blake3(&input.attestation_key);
        let valid = attestation.verify_blake3(&input.attestation_key);
        assert!(valid, "BLAKE3 sign/verify round-trip failed");

        // Verify with wrong key must fail (unless key is all-zeros and happens to collide)
        let mut wrong_key = input.attestation_key;
        wrong_key[0] ^= 0xFF;
        if wrong_key != input.attestation_key {
            let invalid = attestation.verify_blake3(&wrong_key);
            assert!(!invalid, "BLAKE3 verify with wrong key should fail");
        }

        // Attach attestation and re-test effective_tier
        offline.attestation = Some(attestation);
        for &now_us in &input.query_times {
            let tier = offline.effective_tier(now_us);

            // Effective tier can never exceed original (even with attestation)
            assert!(
                tier <= offline.credential.tier,
                "attestation: effective tier {:?} > original {:?}",
                tier,
                offline.credential.tier
            );
        }
    }

    // 5. Test record_online_verification resets degradation
    if let Some(&reset_time) = input.query_times.first() {
        offline.record_online_verification(reset_time);
        assert_eq!(offline.last_online_verification, reset_time);
        let _ts = offline.freshness_timestamp();
    }

    // 6. Monotonicity: for a fixed credential, longer offline = same or lower tier
    //    (only when no attestation and degradation enabled, using default grace)
    //    NOTE: Monotonicity only holds when all query_times are >= reference_time.
    //    When reference_time is far in the future, queries before it trigger
    //    future-timestamp degradation, but queries after it see small elapsed time.
    //    This is by design (clock skew recovery). We restrict the assertion to
    //    query times that are all past the reference_time.
    if input.degradation_enabled && offline.attestation.is_none() && input.custom_grace_hours.is_none()
    {
        let base_offline = OfflineCredential::new(make_credential(&input));
        let ref_time = base_offline.last_online_verification;
        let mut sorted_times: Vec<u64> = input
            .query_times
            .iter()
            .copied()
            .filter(|&t| t >= ref_time)
            .collect();
        sorted_times.sort();
        let mut prev_tier = base_offline.credential.tier;
        for &t in &sorted_times {
            let tier = base_offline.effective_tier(t);
            assert!(
                tier <= prev_tier,
                "tier increased from {:?} to {:?} at time {} (issued_at={}, last_online={})",
                prev_tier,
                tier,
                t,
                base_offline.credential.issued_at,
                base_offline.last_online_verification,
            );
            prev_tier = tier;
        }
    }
});
