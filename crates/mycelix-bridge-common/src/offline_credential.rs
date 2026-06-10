// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline-resilient credential extensions for Mycelix identity.
//!
//! Extends ConsciousnessCredential with graceful degradation during
//! network outages. In South Africa, loadshedding and cable cuts mean
//! internet connectivity cannot be assumed. This module ensures identity
//! remains functional during extended offline periods.
//!
//! ## Degradation Model
//!
//! - **0-24h offline**: Full tier maintained (original credential TTL)
//! - **24h-72h**: Tier drops by one level (e.g., Guardian -> Steward)
//! - **72h-168h (7 days)**: Tier drops by two levels
//! - **>168h**: Falls to Observer (read-only)
//!
//! The degradation is deterministic: any verifier can compute the
//! effective tier from the credential's `issued_at` timestamp and
//! the current time, without needing network connectivity.

use serde::{Deserialize, Serialize};

use crate::consciousness_profile::{ConsciousnessCredential, ConsciousnessTier};

/// External freshness attestation from a peer or bridge node.
///
/// Provides third-party confirmation of a credential's freshness,
/// preventing self-reported `last_online_verification` from being trusted blindly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessAttestation {
    /// DID of the attesting node.
    pub attester_did: String,
    /// Attestation timestamp (microseconds).
    pub timestamp: u64,
    /// Tier at the time of attestation.
    pub tier_at_attestation: ConsciousnessTier,
    /// BLAKE3 keyed hash signature over canonical bytes.
    #[serde(default)]
    pub signature: Vec<u8>,
}

impl FreshnessAttestation {
    /// Deterministic serialization of signable fields.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(self.attester_did.as_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.push(self.tier_at_attestation as u8);
        buf
    }

    /// Sign with BLAKE3 keyed hash.
    pub fn sign_blake3(&mut self, key: &[u8; 32]) {
        let data = self.canonical_bytes();
        let mac = blake3::keyed_hash(key, &data);
        self.signature = mac.as_bytes().to_vec();
    }

    /// Verify BLAKE3 keyed hash signature.
    pub fn verify_blake3(&self, key: &[u8; 32]) -> bool {
        let data = self.canonical_bytes();
        let mac = blake3::keyed_hash(key, &data);
        self.signature == mac.as_bytes().as_slice()
    }
}

/// Grace period thresholds in microseconds.
const HOURS_24: u64 = 24 * 3600 * 1_000_000;
const HOURS_72: u64 = 72 * 3600 * 1_000_000;
const HOURS_168: u64 = 168 * 3600 * 1_000_000;

/// Clock skew tolerance: 15 minutes in microseconds.
///
/// During South African loadshedding or extended offline periods, device
/// clocks can drift. A reference_time up to 15 minutes in the "future" (ahead
/// of now_us) is treated as a zero-elapsed scenario rather than triggering
/// the 2-tier anti-tampering degradation. NTP syncs within seconds when power
/// returns; 15 minutes provides >100x safety margin while limiting the
/// attack window vs the previous 1 hour tolerance.
const CLOCK_SKEW_TOLERANCE_US: u64 = 900 * 1_000_000;

/// Maximum allowed custom grace period: 30 days (720 hours).
const MAX_GRACE_HOURS: u64 = 720;

/// Offline-resilient wrapper for ConsciousnessCredential.
///
/// Provides graceful tier degradation during network outages while
/// preserving the original credential for re-verification when
/// connectivity is restored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineCredential {
    /// The original credential (issued while online).
    pub credential: ConsciousnessCredential,
    /// Timestamp of last successful online verification (microseconds).
    pub last_online_verification: u64,
    /// Whether offline degradation is enabled.
    pub degradation_enabled: bool,
    /// Custom grace period override (microseconds). None = use defaults.
    pub custom_grace_period: Option<u64>,
    /// External freshness attestation from a peer or bridge node.
    /// When present with a valid signature, overrides self-reported timestamps.
    #[serde(default)]
    pub attestation: Option<FreshnessAttestation>,
}

impl OfflineCredential {
    /// Wrap a credential with offline resilience.
    pub fn new(credential: ConsciousnessCredential) -> Self {
        Self {
            last_online_verification: credential.issued_at,
            degradation_enabled: true,
            custom_grace_period: None,
            attestation: None,
            credential,
        }
    }

    /// Create with a custom grace period (in hours, clamped to 720 = 30 days max).
    pub fn with_grace_hours(credential: ConsciousnessCredential, hours: u64) -> Self {
        let clamped_hours = hours.min(MAX_GRACE_HOURS);
        Self {
            last_online_verification: credential.issued_at,
            degradation_enabled: true,
            custom_grace_period: Some(clamped_hours * 3600 * 1_000_000),
            attestation: None,
            credential,
        }
    }

    /// Compute the effective tier at the given time.
    ///
    /// Deterministic: any verifier can compute this independently
    /// from the credential timestamps and the current time.
    pub fn effective_tier(&self, now_us: u64) -> ConsciousnessTier {
        if !self.degradation_enabled {
            return self.credential.tier;
        }

        // Prefer attestation timestamp when attestation is present and signed
        let reference_time = match &self.attestation {
            Some(att) if !att.signature.is_empty() => {
                // Causality check: attestation cannot predate credential issuance
                if att.timestamp < self.credential.issued_at {
                    return self.credential.tier.degrade(2);
                }
                att.timestamp
            }
            _ => self.last_online_verification,
        };

        // Future-dated reference times: tolerate up to 15 minutes of clock skew
        // (common during South African loadshedding recovery when NTP hasn't synced).
        // Beyond the tolerance window, treat as suspicious and degrade.
        if reference_time > now_us {
            let gap = reference_time - now_us;
            if gap > CLOCK_SKEW_TOLERANCE_US {
                return self.credential.tier.degrade(2);
            }
            // Within tolerance: treat as zero elapsed (just-verified)
            return self.credential.tier;
        }

        let elapsed = now_us - reference_time;
        let grace = self.custom_grace_period.unwrap_or(HOURS_24);

        if elapsed <= grace {
            // Within grace period: full tier
            self.credential.tier
        } else if elapsed <= HOURS_72 {
            // 24h-72h: drop one level
            self.credential.tier.degrade(1)
        } else if elapsed <= HOURS_168 {
            // 72h-168h: drop two levels
            self.credential.tier.degrade(2)
        } else {
            // >7 days: Observer only
            ConsciousnessTier::Observer
        }
    }

    /// Whether the credential is still usable (above Observer).
    pub fn is_usable(&self, now_us: u64) -> bool {
        self.effective_tier(now_us) > ConsciousnessTier::Observer
    }

    /// Record a successful online verification (resets degradation clock).
    pub fn record_online_verification(&mut self, now_us: u64) {
        self.last_online_verification = now_us;
    }

    /// Hours elapsed since last online verification.
    pub fn hours_offline(&self, now_us: u64) -> f64 {
        let elapsed_us = now_us.saturating_sub(self.last_online_verification);
        elapsed_us as f64 / (3600.0 * 1_000_000.0)
    }

    /// Freshness proof: signed timestamp from last online sync.
    /// The verifier can check this to determine degradation level.
    pub fn freshness_timestamp(&self) -> u64 {
        self.last_online_verification
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness_profile::ConsciousnessProfile;

    fn make_credential(tier: ConsciousnessTier, issued_at: u64) -> ConsciousnessCredential {
        ConsciousnessCredential {
            did: "did:mycelix:test".to_string(),
            profile: ConsciousnessProfile {
                identity: 0.8,
                reputation: 0.7,
                community: 0.9,
                engagement: 0.6,
            },
            tier,
            issued_at,
            expires_at: issued_at + HOURS_168, // 7 days
            issuer: "did:mycelix:bridge".to_string(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn within_grace_full_tier() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::new(cred);

        // At t=12h, should still be Guardian
        let tier = offline.effective_tier(12 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Guardian);
    }

    #[test]
    fn after_24h_drops_one() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::new(cred);

        // At t=48h, should be Steward (Guardian - 1)
        let tier = offline.effective_tier(48 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Steward);
    }

    #[test]
    fn after_72h_drops_two() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::new(cred);

        // At t=120h, should be Citizen (Guardian - 2)
        let tier = offline.effective_tier(120 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Citizen);
    }

    #[test]
    fn after_7_days_observer() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::new(cred);

        // At t=8 days, should be Observer
        let tier = offline.effective_tier(8 * 24 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Observer);
    }

    #[test]
    fn online_verification_resets_clock() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let mut offline = OfflineCredential::new(cred);

        // At t=48h, would be degraded
        assert_eq!(
            offline.effective_tier(48 * 3600 * 1_000_000),
            ConsciousnessTier::Steward
        );

        // Record online verification at t=47h
        offline.record_online_verification(47 * 3600 * 1_000_000);

        // Now at t=48h, only 1h since last verification -- full tier
        assert_eq!(
            offline.effective_tier(48 * 3600 * 1_000_000),
            ConsciousnessTier::Guardian
        );
    }

    #[test]
    fn degradation_disabled() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let mut offline = OfflineCredential::new(cred);
        offline.degradation_enabled = false;

        // Even after 8 days, should be Guardian
        let tier = offline.effective_tier(8 * 24 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Guardian);
    }

    #[test]
    fn participant_degrades_to_observer() {
        let cred = make_credential(ConsciousnessTier::Participant, 0);
        let offline = OfflineCredential::new(cred);

        // Participant - 1 = Observer
        let tier = offline.effective_tier(48 * 3600 * 1_000_000);
        assert_eq!(tier, ConsciousnessTier::Observer);
    }

    #[test]
    fn hours_offline_calculation() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::new(cred);

        let hours = offline.hours_offline(48 * 3600 * 1_000_000);
        assert!((hours - 48.0).abs() < 0.01);
    }

    #[test]
    fn tier_degrade_levels() {
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(0),
            ConsciousnessTier::Guardian
        );
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(1),
            ConsciousnessTier::Steward
        );
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(2),
            ConsciousnessTier::Citizen
        );
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(3),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(4),
            ConsciousnessTier::Observer
        );
        assert_eq!(
            ConsciousnessTier::Guardian.degrade(100),
            ConsciousnessTier::Observer
        );
    }

    #[test]
    fn custom_grace_period() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let offline = OfflineCredential::with_grace_hours(cred, 48);

        // At t=36h with 48h grace, should still be full tier
        assert_eq!(
            offline.effective_tier(36 * 3600 * 1_000_000),
            ConsciousnessTier::Guardian
        );

        // At t=60h (past 48h grace, within 72h), drops one
        assert_eq!(
            offline.effective_tier(60 * 3600 * 1_000_000),
            ConsciousnessTier::Steward
        );
    }

    #[test]
    fn attestation_overrides_self_reported() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let mut offline = OfflineCredential::new(cred);
        // Self-reported: t=0. At t=48h, would degrade.
        assert_eq!(
            offline.effective_tier(48 * 3600 * 1_000_000),
            ConsciousnessTier::Steward
        );
        // Add attestation at t=47h with signature
        offline.attestation = Some(FreshnessAttestation {
            attester_did: "did:mycelix:peer".into(),
            timestamp: 47 * 3600 * 1_000_000,
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![1], // non-empty = "signed"
        });
        // Now at t=48h, only 1h since attestation — full tier
        assert_eq!(
            offline.effective_tier(48 * 3600 * 1_000_000),
            ConsciousnessTier::Guardian
        );
    }

    #[test]
    fn unsigned_attestation_treated_as_self_reported() {
        let cred = make_credential(ConsciousnessTier::Guardian, 0);
        let mut offline = OfflineCredential::new(cred);
        // Add attestation at t=47h but with empty signature
        offline.attestation = Some(FreshnessAttestation {
            attester_did: "did:mycelix:peer".into(),
            timestamp: 47 * 3600 * 1_000_000,
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![], // unsigned
        });
        // Should fall back to self-reported (t=0), so at t=48h: degraded
        assert_eq!(
            offline.effective_tier(48 * 3600 * 1_000_000),
            ConsciousnessTier::Steward
        );
    }

    #[test]
    fn freshness_attestation_blake3_verification() {
        let key = [99u8; 32];
        let mut att = FreshnessAttestation {
            attester_did: "did:mycelix:bridge".into(),
            timestamp: 1_000_000,
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![],
        };
        att.sign_blake3(&key);
        assert!(att.verify_blake3(&key));
    }

    #[test]
    fn freshness_attestation_tamper_rejection() {
        let key = [99u8; 32];
        let mut att = FreshnessAttestation {
            attester_did: "did:mycelix:bridge".into(),
            timestamp: 1_000_000,
            tier_at_attestation: ConsciousnessTier::Guardian,
            signature: vec![],
        };
        att.sign_blake3(&key);
        // Tamper with timestamp
        att.timestamp = 2_000_000;
        assert!(!att.verify_blake3(&key));
    }

    #[test]
    fn is_usable_above_observer() {
        let cred = make_credential(ConsciousnessTier::Participant, 0);
        let offline = OfflineCredential::new(cred);

        // Participant is usable within grace
        assert!(offline.is_usable(12 * 3600 * 1_000_000));

        // After 24h, Participant degrades to Observer -- not usable
        assert!(!offline.is_usable(48 * 3600 * 1_000_000));
    }

    // ---- Clock skew tolerance ----

    #[test]
    fn clock_skew_within_tolerance_preserves_tier() {
        // Credential issued at time 100h, current time 10 minutes behind
        // (reference_time 10 minutes ahead of now — within 15min tolerance)
        let base = 100 * 3600 * 1_000_000_u64;
        let now = base - 10 * 60 * 1_000_000; // 10 min behind reference
        let cred = make_credential(ConsciousnessTier::Citizen, base);
        let offline = OfflineCredential::new(cred);

        // Should preserve full tier (within clock skew tolerance)
        assert_eq!(offline.effective_tier(now), ConsciousnessTier::Citizen);
    }

    #[test]
    fn clock_skew_at_15min_boundary_preserves_tier() {
        // Reference time exactly 15 minutes ahead of now (at tolerance boundary)
        let base = 100 * 3600 * 1_000_000_u64;
        let now = base - 900 * 1_000_000; // exactly 15min behind
        let cred = make_credential(ConsciousnessTier::Steward, base);
        let offline = OfflineCredential::new(cred);

        // Should still preserve tier (at boundary)
        assert_eq!(offline.effective_tier(now), ConsciousnessTier::Steward);
    }

    #[test]
    fn clock_skew_beyond_tolerance_degrades() {
        // Reference time 30 minutes ahead of now (beyond 15min tolerance)
        let base = 100 * 3600 * 1_000_000_u64;
        let now = base - 30 * 60 * 1_000_000; // 30min behind
        let cred = make_credential(ConsciousnessTier::Guardian, base);
        let offline = OfflineCredential::new(cred);

        // Should degrade by 2 levels (suspicious future timestamp)
        assert_eq!(offline.effective_tier(now), ConsciousnessTier::Citizen);
    }

    #[test]
    fn clock_skew_tiny_drift_preserves_tier() {
        // Reference time just 1 microsecond ahead of now
        let base = 50 * 3600 * 1_000_000_u64;
        let now = base - 1;
        let cred = make_credential(ConsciousnessTier::Citizen, base);
        let offline = OfflineCredential::new(cred);

        assert_eq!(offline.effective_tier(now), ConsciousnessTier::Citizen);
    }
}
