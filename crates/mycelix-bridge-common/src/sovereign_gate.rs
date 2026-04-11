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

#[allow(deprecated)] // Needed for backward-compatible fallback path
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

    // -----------------------------------------------------------------------
    // evaluate_sovereign tests
    // -----------------------------------------------------------------------

    fn steward_profile() -> SovereignProfile {
        SovereignProfile {
            epistemic_integrity: 0.7,
            thermodynamic_yield: 0.6,
            network_resilience: 0.7,
            economic_velocity: 0.6,
            civic_participation: 0.8,
            stewardship_care: 0.6,
            semantic_resonance: 0.7,
            domain_competence: 0.5,
        }
    }

    fn steward_credential() -> SovereignCredential {
        SovereignCredential {
            did: "did:mycelix:test".into(),
            profile: steward_profile(),
            tier: CivicTier::Steward,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "did:mycelix:issuer".into(),
            extensions: vec![],
        }
    }

    #[test]
    fn evaluate_sovereign_eligible_basic() {
        let cred = steward_credential();
        let req = civic_requirement_basic();
        let result = evaluate_sovereign(&cred, &req, 1000);
        assert!(result.eligible);
        assert!(result.reasons.is_empty());
        assert!(result.weight_bp > 0);
        assert_eq!(result.tier, ConsciousnessTier::Steward);
    }

    #[test]
    fn evaluate_sovereign_eligible_with_dimensions() {
        let cred = steward_credential();
        let req = civic_requirement_constitutional();
        let result = evaluate_sovereign(&cred, &req, 1000);
        assert!(result.eligible, "Steward should pass constitutional: {:?}", result.reasons);
    }

    #[test]
    fn evaluate_sovereign_ineligible_tier_too_low() {
        let mut cred = steward_credential();
        cred.profile = SovereignProfile {
            epistemic_integrity: 0.1,
            thermodynamic_yield: 0.1,
            network_resilience: 0.1,
            economic_velocity: 0.1,
            civic_participation: 0.1,
            stewardship_care: 0.1,
            semantic_resonance: 0.1,
            domain_competence: 0.1,
        };
        let req = civic_requirement_voting(); // Citizen required
        let result = evaluate_sovereign(&cred, &req, 1000);
        assert!(!result.eligible);
        assert!(!result.reasons.is_empty());
    }

    #[test]
    fn evaluate_sovereign_ineligible_dimension_below_minimum() {
        let mut cred = steward_credential();
        // High overall score but one dimension below the constitutional minimum
        cred.profile.epistemic_integrity = 0.1;
        let req = CivicRequirement {
            min_tier: CivicTier::Participant,
            min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.5)],
        };
        let result = evaluate_sovereign(&cred, &req, 1000);
        assert!(!result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("EpistemicIntegrity")));
    }

    #[test]
    fn evaluate_sovereign_expired_credential() {
        let mut cred = steward_credential();
        cred.expires_at = 500; // expired before now_us=1000
        let req = civic_requirement_basic();
        let result = evaluate_sovereign(&cred, &req, 1000);
        assert!(!result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("expired")));
    }

    #[test]
    fn evaluate_sovereign_returns_backward_compatible_profile() {
        let cred = steward_credential();
        let req = civic_requirement_basic();
        let result = evaluate_sovereign(&cred, &req, 1000);
        // Legacy profile should be populated (not zero)
        assert!(result.profile.identity > 0.0);
        assert!(result.profile.reputation > 0.0);
        assert!(result.profile.community > 0.0);
    }

    #[test]
    fn civic_to_consciousness_tier_is_bijective() {
        let pairs = [
            (CivicTier::Observer, ConsciousnessTier::Observer),
            (CivicTier::Participant, ConsciousnessTier::Participant),
            (CivicTier::Citizen, ConsciousnessTier::Citizen),
            (CivicTier::Steward, ConsciousnessTier::Steward),
            (CivicTier::Guardian, ConsciousnessTier::Guardian),
        ];
        for (civic, expected) in pairs {
            assert_eq!(civic_to_consciousness_tier(civic), expected);
        }
    }

    #[test]
    fn sovereign_to_legacy_profile_round_trips() {
        // LegacyProfile → SovereignProfile → LegacyProfile should be identity
        let original = ConsciousnessProfile {
            identity: 0.6,
            reputation: 0.4,
            community: 0.8,
            engagement: 0.2,
        };
        let legacy_cred = ConsciousnessCredential {
            did: "test".into(),
            profile: original.clone(),
            tier: ConsciousnessTier::Citizen,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "test".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        let sovereign = sovereign_from_credential(&legacy_cred);
        let round_tripped = sovereign_to_legacy_profile(&sovereign);
        assert!((round_tripped.identity - original.identity).abs() < 1e-10);
        assert!((round_tripped.reputation - original.reputation).abs() < 1e-10);
        assert!((round_tripped.community - original.community).abs() < 1e-10);
        assert!((round_tripped.engagement - original.engagement).abs() < 1e-10);
    }

    #[test]
    fn evaluate_sovereign_weight_scales_with_tier() {
        // Higher-tier profiles should get higher vote weight
        let low = SovereignCredential {
            did: "low".into(),
            profile: SovereignProfile {
                epistemic_integrity: 0.3,
                thermodynamic_yield: 0.3,
                network_resilience: 0.3,
                economic_velocity: 0.3,
                civic_participation: 0.3,
                stewardship_care: 0.3,
                semantic_resonance: 0.3,
                domain_competence: 0.3,
            },
            tier: CivicTier::Participant,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "test".into(),
            extensions: vec![],
        };
        let high = steward_credential();
        let req = civic_requirement_basic();
        let low_result = evaluate_sovereign(&low, &req, 1000);
        let high_result = evaluate_sovereign(&high, &req, 1000);
        assert!(high_result.weight_bp >= low_result.weight_bp,
            "Steward weight {} should >= Participant weight {}",
            high_result.weight_bp, low_result.weight_bp);
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
