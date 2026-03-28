// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! License enforcement for cross-cluster routing.
//!
//! Protects zomes backed by patented/licensed inventions, ensuring that
//! agents hold a valid license grant before dispatching calls to those
//! zomes.  Open-source licensed inventions pass through without checks.
//!
//! ## Design
//!
//! The `LicenseRegistry` holds:
//! 1. A list of `LicenseGrant` entries (who may use what, and under what terms).
//! 2. A mapping from zome names to required invention IDs (`protected_zomes`).
//!
//! When a cross-cluster call targets a protected zome, the caller invokes
//! `check_license()` which returns one of four outcomes: `Allowed`,
//! `RequiresLicense`, `Expired`, or `Revoked`.
//!
//! ## Integration
//!
//! License checks sit between routing allowlist validation and the actual
//! `call()` dispatch.  See `routing_registry.rs` for the integration point.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// License types
// ============================================================================

/// Type of license granted for an invention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrantedLicenseType {
    /// Open-source license (AGPL/MIT/Apache) — no enforcement needed.
    OpenSource,
    /// Paid commercial license.
    Commercial,
    /// Academic or non-commercial research use.
    Research,
    /// Time-limited evaluation/trial.
    Evaluation,
    /// Previously granted, now revoked.
    Revoked,
}

/// A license grant for a specific invention to a specific agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseGrant {
    /// Unique identifier of the invention (references `InventionClaim.id`).
    pub invention_id: String,
    /// DID of the licensed agent.
    pub licensee_did: String,
    /// Type of the granted license.
    pub license_type: GrantedLicenseType,
    /// Timestamp when the license was granted (microseconds).
    pub granted_at: u64,
    /// Expiry timestamp (microseconds). `None` means perpetual.
    pub expires_at: Option<u64>,
    /// Whether the licensee must attribute the inventor.
    pub attribution_required: bool,
    /// Optional royalty percentage (0.0–100.0).
    pub royalty_percentage: Option<f64>,
}

// ============================================================================
// Check result
// ============================================================================

/// Result of a license check against the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseCheckResult {
    /// The call is allowed (valid grant, open-source, or zome is unprotected).
    Allowed,
    /// The agent must obtain a license for the given invention.
    RequiresLicense { invention_id: String },
    /// The agent's license has expired.
    Expired { invention_id: String, expired_at: u64 },
    /// The agent's license was revoked.
    Revoked { invention_id: String },
}

// ============================================================================
// Registry
// ============================================================================

/// License registry holding grants and zome-to-invention protection mappings.
///
/// Constructed at startup and consulted on every cross-cluster dispatch to a
/// protected zome.  Unprotected zomes always pass through.
#[derive(Debug, Clone, Default)]
pub struct LicenseRegistry {
    /// All registered license grants.
    grants: Vec<LicenseGrant>,
    /// Map from zome name to the invention IDs it requires a license for.
    protected_zomes: HashMap<String, Vec<String>>,
}

impl LicenseRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new license grant.
    pub fn register_grant(&mut self, grant: LicenseGrant) {
        self.grants.push(grant);
    }

    /// Revoke a specific grant by invention and licensee.
    ///
    /// Marks **all** matching grants as `Revoked`.  Returns the number of
    /// grants that were revoked.
    pub fn revoke_grant(&mut self, invention_id: &str, licensee_did: &str) -> usize {
        let mut count = 0;
        for grant in &mut self.grants {
            if grant.invention_id == invention_id
                && grant.licensee_did == licensee_did
                && grant.license_type != GrantedLicenseType::Revoked
            {
                grant.license_type = GrantedLicenseType::Revoked;
                count += 1;
            }
        }
        count
    }

    /// Mark a zome as requiring license(s) for the given invention IDs.
    ///
    /// Calling this multiple times for the same zome appends to the list
    /// (duplicates are filtered during `check_license`).
    pub fn protect_zome(&mut self, zome_name: &str, invention_ids: Vec<String>) {
        self.protected_zomes
            .entry(zome_name.to_string())
            .or_default()
            .extend(invention_ids);
    }

    /// Check whether `licensee_did` may call `target_zome` at time `now`.
    ///
    /// Returns the first failing check, or `Allowed` if every required
    /// invention has a valid (non-expired, non-revoked) grant.
    pub fn check_license(
        &self,
        licensee_did: &str,
        target_zome: &str,
        now: u64,
    ) -> LicenseCheckResult {
        // Unprotected zomes always pass.
        let required = match self.protected_zomes.get(target_zome) {
            Some(ids) if !ids.is_empty() => ids,
            _ => return LicenseCheckResult::Allowed,
        };

        for invention_id in required {
            // Open-source inventions need no per-agent grant.
            if self.is_open_source(invention_id) {
                continue;
            }

            // Find the best matching grant for this agent + invention.
            let grant = self
                .grants
                .iter()
                .filter(|g| g.invention_id == *invention_id && g.licensee_did == licensee_did)
                .last(); // Most recently registered grant wins.

            match grant {
                None => {
                    return LicenseCheckResult::RequiresLicense {
                        invention_id: invention_id.clone(),
                    };
                }
                Some(g) if g.license_type == GrantedLicenseType::Revoked => {
                    return LicenseCheckResult::Revoked {
                        invention_id: invention_id.clone(),
                    };
                }
                Some(g) => {
                    if let Some(expires_at) = g.expires_at {
                        if now >= expires_at {
                            return LicenseCheckResult::Expired {
                                invention_id: invention_id.clone(),
                                expired_at: expires_at,
                            };
                        }
                    }
                    // Valid grant — continue to next required invention.
                }
            }
        }

        LicenseCheckResult::Allowed
    }

    /// Whether an invention is published under an open-source license.
    ///
    /// Returns `true` if **any** grant for this invention is `OpenSource`.
    /// Open-source inventions bypass per-agent enforcement entirely.
    pub fn is_open_source(&self, invention_id: &str) -> bool {
        self.grants.iter().any(|g| {
            g.invention_id == invention_id && g.license_type == GrantedLicenseType::OpenSource
        })
    }

    /// Return all grants held by a specific agent.
    pub fn get_grants_for_agent(&self, did: &str) -> Vec<&LicenseGrant> {
        self.grants
            .iter()
            .filter(|g| g.licensee_did == did)
            .collect()
    }

    /// Return all grants that have expired as of `now`.
    pub fn expired_grants(&self, now: u64) -> Vec<&LicenseGrant> {
        self.grants
            .iter()
            .filter(|g| {
                if let Some(exp) = g.expires_at {
                    now >= exp
                } else {
                    false
                }
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        1_000_000_000_000 // arbitrary timestamp
    }

    fn make_grant(
        invention_id: &str,
        licensee_did: &str,
        license_type: GrantedLicenseType,
        expires_at: Option<u64>,
    ) -> LicenseGrant {
        LicenseGrant {
            invention_id: invention_id.to_string(),
            licensee_did: licensee_did.to_string(),
            license_type,
            granted_at: now(),
            expires_at,
            attribution_required: false,
            royalty_percentage: None,
        }
    }

    // ---- Construction ----

    #[test]
    fn empty_registry() {
        let reg = LicenseRegistry::new();
        assert!(reg.grants.is_empty());
        assert!(reg.protected_zomes.is_empty());
    }

    // ---- Grant registration and lookup ----

    #[test]
    fn register_and_list_grants() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        reg.register_grant(make_grant(
            "inv-002",
            "did:mycelix:alice",
            GrantedLicenseType::Research,
            None,
        ));
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:bob",
            GrantedLicenseType::Evaluation,
            Some(now() + 1_000_000),
        ));

        let alice_grants = reg.get_grants_for_agent("did:mycelix:alice");
        assert_eq!(alice_grants.len(), 2);

        let bob_grants = reg.get_grants_for_agent("did:mycelix:bob");
        assert_eq!(bob_grants.len(), 1);

        let unknown = reg.get_grants_for_agent("did:mycelix:charlie");
        assert!(unknown.is_empty());
    }

    // ---- Unprotected zome always allowed ----

    #[test]
    fn unprotected_zome_always_allowed() {
        let reg = LicenseRegistry::new();
        let result = reg.check_license("did:mycelix:anyone", "some_zome", now());
        assert_eq!(result, LicenseCheckResult::Allowed);
    }

    // ---- Open-source invention passes without per-agent grant ----

    #[test]
    fn open_source_invention_passes() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-oss",
            "did:mycelix:community",
            GrantedLicenseType::OpenSource,
            None,
        ));
        reg.protect_zome("cool_zome", vec!["inv-oss".into()]);

        // Agent with no personal grant can still access because the invention is OSS.
        let result = reg.check_license("did:mycelix:random", "cool_zome", now());
        assert_eq!(result, LicenseCheckResult::Allowed);
        assert!(reg.is_open_source("inv-oss"));
    }

    // ---- Valid commercial grant ----

    #[test]
    fn valid_commercial_grant_allows() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None, // perpetual
        ));
        reg.protect_zome("patented_zome", vec!["inv-001".into()]);

        let result = reg.check_license("did:mycelix:alice", "patented_zome", now());
        assert_eq!(result, LicenseCheckResult::Allowed);
    }

    // ---- No grant → requires license ----

    #[test]
    fn no_grant_requires_license() {
        let mut reg = LicenseRegistry::new();
        reg.protect_zome("patented_zome", vec!["inv-001".into()]);

        let result = reg.check_license("did:mycelix:alice", "patented_zome", now());
        assert_eq!(
            result,
            LicenseCheckResult::RequiresLicense {
                invention_id: "inv-001".into()
            }
        );
    }

    // ---- Expired grant ----

    #[test]
    fn expired_grant_returns_expired() {
        let mut reg = LicenseRegistry::new();
        let expiry = now() + 500_000;
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Evaluation,
            Some(expiry),
        ));
        reg.protect_zome("trial_zome", vec!["inv-001".into()]);

        // Before expiry — allowed.
        let result = reg.check_license("did:mycelix:alice", "trial_zome", now());
        assert_eq!(result, LicenseCheckResult::Allowed);

        // At expiry — expired.
        let result = reg.check_license("did:mycelix:alice", "trial_zome", expiry);
        assert_eq!(
            result,
            LicenseCheckResult::Expired {
                invention_id: "inv-001".into(),
                expired_at: expiry,
            }
        );

        // After expiry — expired.
        let result = reg.check_license("did:mycelix:alice", "trial_zome", expiry + 1);
        assert_eq!(
            result,
            LicenseCheckResult::Expired {
                invention_id: "inv-001".into(),
                expired_at: expiry,
            }
        );
    }

    // ---- Revoked grant ----

    #[test]
    fn revoked_grant_returns_revoked() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        reg.protect_zome("patented_zome", vec!["inv-001".into()]);

        // Before revocation — allowed.
        assert_eq!(
            reg.check_license("did:mycelix:alice", "patented_zome", now()),
            LicenseCheckResult::Allowed,
        );

        // Revoke.
        let count = reg.revoke_grant("inv-001", "did:mycelix:alice");
        assert_eq!(count, 1);

        // After revocation — revoked.
        assert_eq!(
            reg.check_license("did:mycelix:alice", "patented_zome", now()),
            LicenseCheckResult::Revoked {
                invention_id: "inv-001".into()
            },
        );
    }

    // ---- Revoke idempotency ----

    #[test]
    fn revoke_idempotent() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));

        assert_eq!(reg.revoke_grant("inv-001", "did:mycelix:alice"), 1);
        // Second revoke should find 0 non-revoked grants.
        assert_eq!(reg.revoke_grant("inv-001", "did:mycelix:alice"), 0);
    }

    // ---- Multiple inventions required ----

    #[test]
    fn multiple_inventions_all_must_be_licensed() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        // Alice has inv-001 but NOT inv-002.
        reg.protect_zome("multi_zome", vec!["inv-001".into(), "inv-002".into()]);

        let result = reg.check_license("did:mycelix:alice", "multi_zome", now());
        assert_eq!(
            result,
            LicenseCheckResult::RequiresLicense {
                invention_id: "inv-002".into()
            }
        );

        // Grant inv-002 as well.
        reg.register_grant(make_grant(
            "inv-002",
            "did:mycelix:alice",
            GrantedLicenseType::Research,
            None,
        ));
        let result = reg.check_license("did:mycelix:alice", "multi_zome", now());
        assert_eq!(result, LicenseCheckResult::Allowed);
    }

    // ---- Multiple grants for same agent/invention ----

    #[test]
    fn latest_grant_wins() {
        let mut reg = LicenseRegistry::new();
        // First grant: evaluation that expires.
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Evaluation,
            Some(now() + 100),
        ));
        // Second grant: commercial perpetual (most recent).
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        reg.protect_zome("z", vec!["inv-001".into()]);

        // The perpetual commercial grant (latest) should win.
        let result = reg.check_license("did:mycelix:alice", "z", now() + 200);
        assert_eq!(result, LicenseCheckResult::Allowed);
    }

    // ---- Unknown agent ----

    #[test]
    fn unknown_agent_requires_license() {
        let mut reg = LicenseRegistry::new();
        reg.protect_zome("z", vec!["inv-001".into()]);

        let result = reg.check_license("did:mycelix:unknown", "z", now());
        assert_eq!(
            result,
            LicenseCheckResult::RequiresLicense {
                invention_id: "inv-001".into()
            }
        );
    }

    // ---- Unknown zome (unprotected) ----

    #[test]
    fn unknown_zome_always_allowed() {
        let mut reg = LicenseRegistry::new();
        reg.protect_zome("other", vec!["inv-001".into()]);

        // "nonexistent" is not in protected_zomes.
        let result = reg.check_license("did:mycelix:alice", "nonexistent", now());
        assert_eq!(result, LicenseCheckResult::Allowed);
    }

    // ---- expired_grants helper ----

    #[test]
    fn expired_grants_helper() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Evaluation,
            Some(now() + 100),
        ));
        reg.register_grant(make_grant(
            "inv-002",
            "did:mycelix:bob",
            GrantedLicenseType::Commercial,
            None, // perpetual — never expires
        ));
        reg.register_grant(make_grant(
            "inv-003",
            "did:mycelix:charlie",
            GrantedLicenseType::Research,
            Some(now() + 200),
        ));

        // At now+150: only inv-001 has expired.
        let expired = reg.expired_grants(now() + 150);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].invention_id, "inv-001");

        // At now+300: both time-limited grants expired.
        let expired = reg.expired_grants(now() + 300);
        assert_eq!(expired.len(), 2);
    }

    // ---- is_open_source ----

    #[test]
    fn non_existent_invention_is_not_open_source() {
        let reg = LicenseRegistry::new();
        assert!(!reg.is_open_source("inv-nonexistent"));
    }

    #[test]
    fn commercial_invention_is_not_open_source() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(make_grant(
            "inv-001",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        assert!(!reg.is_open_source("inv-001"));
    }

    // ---- protect_zome accumulates ----

    #[test]
    fn protect_zome_accumulates() {
        let mut reg = LicenseRegistry::new();
        reg.protect_zome("z", vec!["inv-001".into()]);
        reg.protect_zome("z", vec!["inv-002".into()]);

        // Both inventions are now required.
        let required = reg.protected_zomes.get("z").unwrap();
        assert_eq!(required.len(), 2);
    }

    // ---- Mixed: one OSS, one commercial ----

    #[test]
    fn mixed_oss_and_commercial_requires_commercial_grant() {
        let mut reg = LicenseRegistry::new();
        // inv-oss is open source.
        reg.register_grant(make_grant(
            "inv-oss",
            "did:mycelix:community",
            GrantedLicenseType::OpenSource,
            None,
        ));
        // inv-com is commercial — alice has a grant.
        reg.register_grant(make_grant(
            "inv-com",
            "did:mycelix:alice",
            GrantedLicenseType::Commercial,
            None,
        ));
        reg.protect_zome("hybrid_zome", vec!["inv-oss".into(), "inv-com".into()]);

        // Alice can access (OSS passes, commercial grant exists).
        assert_eq!(
            reg.check_license("did:mycelix:alice", "hybrid_zome", now()),
            LicenseCheckResult::Allowed,
        );

        // Bob cannot access (OSS passes, but no commercial grant).
        assert_eq!(
            reg.check_license("did:mycelix:bob", "hybrid_zome", now()),
            LicenseCheckResult::RequiresLicense {
                invention_id: "inv-com".into()
            },
        );
    }

    // ---- Serde round-trip ----

    #[test]
    fn grant_serde_roundtrip() {
        let grant = LicenseGrant {
            invention_id: "inv-001".into(),
            licensee_did: "did:mycelix:alice".into(),
            license_type: GrantedLicenseType::Commercial,
            granted_at: now(),
            expires_at: Some(now() + 1_000_000),
            attribution_required: true,
            royalty_percentage: Some(5.0),
        };

        let json = serde_json::to_string(&grant).unwrap();
        let round: LicenseGrant = serde_json::from_str(&json).unwrap();
        assert_eq!(round.invention_id, "inv-001");
        assert_eq!(round.license_type, GrantedLicenseType::Commercial);
        assert!(round.attribution_required);
        assert_eq!(round.royalty_percentage, Some(5.0));
    }

    // ---- Attribution and royalty fields ----

    #[test]
    fn attribution_and_royalty_preserved() {
        let mut reg = LicenseRegistry::new();
        reg.register_grant(LicenseGrant {
            invention_id: "inv-001".into(),
            licensee_did: "did:mycelix:alice".into(),
            license_type: GrantedLicenseType::Commercial,
            granted_at: now(),
            expires_at: None,
            attribution_required: true,
            royalty_percentage: Some(2.5),
        });

        let grants = reg.get_grants_for_agent("did:mycelix:alice");
        assert_eq!(grants.len(), 1);
        assert!(grants[0].attribution_required);
        assert_eq!(grants[0].royalty_percentage, Some(2.5));
    }
}
