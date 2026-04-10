// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign civic gating — 8D replacement for consciousness gating.
//!
//! `gate_civic()` is a drop-in replacement for `gate_consciousness()`.
//! It attempts to fetch a native `SovereignCredential` (8D) first. If the
//! local bridge supports it, evaluation uses the full 8-dimensional profile.
//! Otherwise, it falls back to the legacy `ConsciousnessCredential` (4D)
//! with automatic conversion to the 8D space.
//!
//! Returns the same `GovernanceEligibility` for backward compatibility.

pub use sovereign_profile::{
    civic_requirement_basic, civic_requirement_constitutional, civic_requirement_guardian,
    civic_requirement_proposal, civic_requirement_voting, CivicRequirement, CivicTier,
    SovereignCredential, SovereignDimension, SovereignProfile,
};
pub use sovereign_profile::compat::{LegacyProfile, LegacyTier};
pub use sovereign_profile::weights::DimensionWeights;
pub use sovereign_profile::i18n;

use crate::consciousness_profile::{
    ConsciousnessCredential, ConsciousnessTier, GovernanceEligibility, GovernanceRequirement,
};

// ---------------------------------------------------------------------------
// Conversion: ConsciousnessCredential → SovereignProfile
// ---------------------------------------------------------------------------

/// Convert a legacy `ConsciousnessCredential` to a `SovereignProfile`.
///
/// The 4 old dimensions are distributed across the 8 new dimensions:
/// - identity → epistemic_integrity + network_resilience
/// - reputation → economic_velocity + stewardship_care
/// - community → civic_participation + semantic_resonance
/// - engagement → thermodynamic_yield + domain_competence
pub fn sovereign_from_credential(credential: &ConsciousnessCredential) -> SovereignProfile {
    let legacy = LegacyProfile {
        identity: credential.profile.identity,
        reputation: credential.profile.reputation,
        community: credential.profile.community,
        engagement: credential.profile.engagement,
    };
    SovereignProfile::from(legacy)
}

/// Convert a `CivicRequirement` to the legacy `GovernanceRequirement`.
///
/// Maps per-dimension minimums back to the old identity/community minimums.
/// Dimensions without a clear mapping are dropped (conservative — never gates
/// more than the old system would).
pub fn governance_requirement_from_civic(civic: &CivicRequirement) -> GovernanceRequirement {
    let min_tier = match civic.min_tier {
        CivicTier::Observer => ConsciousnessTier::Observer,
        CivicTier::Participant => ConsciousnessTier::Participant,
        CivicTier::Citizen => ConsciousnessTier::Citizen,
        CivicTier::Steward => ConsciousnessTier::Steward,
        CivicTier::Guardian => ConsciousnessTier::Guardian,
    };

    let mut min_identity = None;
    let mut min_community = None;

    for &(dim, val) in &civic.min_dimensions {
        match dim {
            // Epistemic maps to identity in the old system
            SovereignDimension::EpistemicIntegrity | SovereignDimension::NetworkResilience => {
                min_identity = Some(min_identity.unwrap_or(0.0_f64).max(val));
            }
            // Civic participation maps to community in the old system
            SovereignDimension::CivicParticipation | SovereignDimension::SemanticResonance => {
                min_community = Some(min_community.unwrap_or(0.0_f64).max(val));
            }
            // Other dimensions have no legacy equivalent — silently ignored
            _ => {}
        }
    }

    GovernanceRequirement {
        min_tier,
        min_identity,
        min_community,
    }
}

// ---------------------------------------------------------------------------
// gate_civic — the main entry point
// ---------------------------------------------------------------------------

/// Map `CivicTier` to legacy `ConsciousnessTier` (1:1, same names).
fn civic_to_consciousness_tier(tier: CivicTier) -> ConsciousnessTier {
    match tier {
        CivicTier::Observer => ConsciousnessTier::Observer,
        CivicTier::Participant => ConsciousnessTier::Participant,
        CivicTier::Citizen => ConsciousnessTier::Citizen,
        CivicTier::Steward => ConsciousnessTier::Steward,
        CivicTier::Guardian => ConsciousnessTier::Guardian,
    }
}

/// Evaluate a `SovereignCredential` against a `CivicRequirement`.
///
/// Produces a `GovernanceEligibility` for backward compatibility.
fn evaluate_sovereign(
    cred: &SovereignCredential,
    requirement: &CivicRequirement,
    now_us: u64,
) -> GovernanceEligibility {
    if cred.expires_at <= now_us {
        return GovernanceEligibility {
            eligible: false,
            weight_bp: 0,
            tier: ConsciousnessTier::Observer,
            profile: sovereign_to_legacy_profile(&cred.profile),
            reasons: vec!["Sovereign credential expired".into()],
            restoration_progress: 1.0,
        };
    }

    let weights = DimensionWeights::governance();
    let tier = cred.profile.tier(&weights);
    let eligible = cred.profile.meets_requirement(requirement, &weights);
    let consciousness_tier = civic_to_consciousness_tier(tier);
    let weight_bp = consciousness_tier.vote_weight_bp();

    let reasons = if eligible {
        vec![]
    } else {
        let mut r = Vec::new();
        if tier < requirement.min_tier {
            r.push(format!(
                "CivicTier {:?} below required {:?}",
                tier, requirement.min_tier
            ));
        }
        for &(dim, min_val) in &requirement.min_dimensions {
            let actual = cred.profile.get(dim);
            if actual < min_val {
                r.push(format!(
                    "{:?} {:.3} below required {:.3}",
                    dim, actual, min_val
                ));
            }
        }
        r
    };

    GovernanceEligibility {
        eligible,
        weight_bp,
        tier: consciousness_tier,
        profile: sovereign_to_legacy_profile(&cred.profile),
        reasons,
        restoration_progress: 1.0,
    }
}

/// Convert `SovereignProfile` to legacy `ConsciousnessProfile` for backward compat.
fn sovereign_to_legacy_profile(
    profile: &SovereignProfile,
) -> crate::consciousness_profile::ConsciousnessProfile {
    let legacy = LegacyProfile::from(profile.clone());
    crate::consciousness_profile::ConsciousnessProfile {
        identity: legacy.identity,
        reputation: legacy.reputation,
        community: legacy.community,
        engagement: legacy.engagement,
    }
}

/// Gate a governance action using the 8D Sovereign Profile.
///
/// Drop-in replacement for `gate_consciousness()`. Tries to fetch a native
/// `SovereignCredential` from the local bridge first. Falls back to the legacy
/// `ConsciousnessCredential` path if the bridge doesn't support it yet.
///
/// Returns `Ok(GovernanceEligibility)` — eligible or ineligible with reasons.
#[cfg(feature = "hdk")]
pub fn gate_civic(
    bridge_zome: &str,
    requirement: &CivicRequirement,
    action_name: &str,
) -> hdk::prelude::ExternResult<GovernanceEligibility> {
    use hdk::prelude::*;

    let agent = agent_info()?.agent_initial_pubkey;
    let did = format!("did:mycelix:{}", agent);

    // --- Try native SovereignCredential path ---
    if let Ok(ZomeCallResponse::Ok(extern_io)) = call(
        CallTargetCell::Local,
        ZomeName::new(bridge_zome),
        FunctionName::new("get_sovereign_credential"),
        None,
        did.clone(),
    ) {
        if let Ok(cred) = extern_io.decode::<SovereignCredential>() {
            let now_us = sys_time()?.as_micros() as u64;
            let result = evaluate_sovereign(&cred, requirement, now_us);

            let tier_index = match result.tier {
                ConsciousnessTier::Observer => 0,
                ConsciousnessTier::Participant => 1,
                ConsciousnessTier::Citizen => 2,
                ConsciousnessTier::Steward => 3,
                ConsciousnessTier::Guardian => 4,
            };
            crate::metrics::record_gate_check(result.eligible, tier_index, 0);

            return Ok(result);
        }
    }

    // --- Fallback: legacy ConsciousnessCredential path ---
    let legacy_req = governance_requirement_from_civic(requirement);
    crate::consciousness_profile::gate_consciousness(bridge_zome, &legacy_req, action_name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness_profile::ConsciousnessProfile;

    #[test]
    fn sovereign_from_credential_maps_dimensions() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile {
                identity: 0.7,
                reputation: 0.5,
                community: 0.9,
                engagement: 0.3,
            },
            tier: ConsciousnessTier::Citizen,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "test".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };

        let sovereign = sovereign_from_credential(&cred);
        assert!((sovereign.epistemic_integrity - 0.7).abs() < 1e-10);
        assert!((sovereign.network_resilience - 0.7).abs() < 1e-10);
        assert!((sovereign.economic_velocity - 0.5).abs() < 1e-10);
        assert!((sovereign.stewardship_care - 0.5).abs() < 1e-10);
        assert!((sovereign.civic_participation - 0.9).abs() < 1e-10);
        assert!((sovereign.semantic_resonance - 0.9).abs() < 1e-10);
        assert!((sovereign.thermodynamic_yield - 0.3).abs() < 1e-10);
        assert!((sovereign.domain_competence - 0.3).abs() < 1e-10);
    }

    #[test]
    fn civic_requirement_basic_converts_to_participant() {
        let civic = civic_requirement_basic();
        let legacy = governance_requirement_from_civic(&civic);
        assert_eq!(legacy.min_tier, ConsciousnessTier::Participant);
        assert!(legacy.min_identity.is_none());
        assert!(legacy.min_community.is_none());
    }

    #[test]
    fn civic_requirement_voting_converts_with_identity_minimum() {
        let civic = civic_requirement_voting();
        let legacy = governance_requirement_from_civic(&civic);
        assert_eq!(legacy.min_tier, ConsciousnessTier::Citizen);
        assert_eq!(legacy.min_identity, Some(0.25));
    }

    #[test]
    fn civic_requirement_constitutional_converts_with_both_minimums() {
        let civic = civic_requirement_constitutional();
        let legacy = governance_requirement_from_civic(&civic);
        assert_eq!(legacy.min_tier, ConsciousnessTier::Steward);
        assert_eq!(legacy.min_identity, Some(0.5));
        assert_eq!(legacy.min_community, Some(0.3));
    }

    #[test]
    fn civic_requirement_guardian_converts_correctly() {
        let civic = civic_requirement_guardian();
        let legacy = governance_requirement_from_civic(&civic);
        assert_eq!(legacy.min_tier, ConsciousnessTier::Guardian);
        assert_eq!(legacy.min_identity, Some(0.7));
        assert_eq!(legacy.min_community, Some(0.5));
    }

    #[test]
    fn all_civic_presets_have_matching_legacy_presets() {
        // Verify civic presets produce equivalent legacy requirements
        use crate::consciousness_profile::*;

        let pairs: Vec<(CivicRequirement, GovernanceRequirement)> = vec![
            (civic_requirement_basic(), requirement_for_basic()),
            (civic_requirement_proposal(), requirement_for_proposal()),
            (civic_requirement_voting(), requirement_for_voting()),
            (civic_requirement_constitutional(), requirement_for_constitutional()),
            (civic_requirement_guardian(), requirement_for_guardian()),
        ];

        for (civic, expected) in pairs {
            let converted = governance_requirement_from_civic(&civic);
            assert_eq!(
                converted.min_tier, expected.min_tier,
                "Tier mismatch for {:?}", civic.min_tier
            );
            assert_eq!(
                converted.min_identity, expected.min_identity,
                "Identity minimum mismatch for {:?}", civic.min_tier
            );
            assert_eq!(
                converted.min_community, expected.min_community,
                "Community minimum mismatch for {:?}", civic.min_tier
            );
        }
    }
}
