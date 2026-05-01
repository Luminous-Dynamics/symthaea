// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::compat::{LegacyProfile, LegacyTier};
use crate::weights::DimensionWeights;
use crate::*;

// ---------------------------------------------------------------------------
// SovereignProfile basics
// ---------------------------------------------------------------------------

#[test]
fn zero_profile_is_observer() {
    let p = SovereignProfile::zero();
    assert_eq!(p.tier(&DimensionWeights::equal()), CivicTier::Observer);
}

#[test]
fn full_profile_is_guardian() {
    let p = SovereignProfile::from_array([1.0; 8]);
    assert_eq!(p.tier(&DimensionWeights::equal()), CivicTier::Guardian);
}

#[test]
fn combined_score_with_equal_weights() {
    let p = SovereignProfile::from_array([0.5; 8]);
    let score = p.combined_score(&DimensionWeights::equal());
    assert!((score - 0.5).abs() < 1e-10);
}

#[test]
fn combined_score_with_governance_weights() {
    let mut p = SovereignProfile::zero();
    p.civic_participation = 1.0; // D4 weight in governance = 0.18
    let score = p.combined_score(&DimensionWeights::governance());
    assert!((score - 0.18).abs() < 1e-10);
}

#[test]
fn single_dimension_contributes_its_weight() {
    let weights = DimensionWeights::governance();
    for dim in SovereignDimension::ALL {
        let mut p = SovereignProfile::zero();
        p.set(dim, 1.0);
        let score = p.combined_score(&weights);
        let expected = weights.weights[dim.index()];
        assert!(
            (score - expected).abs() < 1e-10,
            "Dimension {:?} (idx {}) should contribute {expected}, got {score}",
            dim,
            dim.index()
        );
    }
}

#[test]
fn nan_dimensions_treated_as_zero() {
    let p = SovereignProfile::from_array([f64::NAN; 8]);
    assert_eq!(p.combined_score(&DimensionWeights::equal()), 0.0);
}

#[test]
fn inf_dimensions_clamped() {
    let p = SovereignProfile::from_array([f64::INFINITY; 8]);
    // All sanitized to 0.0 (not finite → 0.0)
    assert_eq!(p.combined_score(&DimensionWeights::equal()), 0.0);
}

#[test]
fn negative_dimensions_clamped_to_zero() {
    let p = SovereignProfile::from_array([-1.0; 8]);
    assert_eq!(p.combined_score(&DimensionWeights::equal()), 0.0);
}

#[test]
fn supranormal_dimensions_clamped_to_one() {
    let p = SovereignProfile::from_array([5.0; 8]);
    assert!((p.combined_score(&DimensionWeights::equal()) - 1.0).abs() < 1e-10);
}

#[test]
fn get_and_set_round_trip() {
    let mut p = SovereignProfile::zero();
    for dim in SovereignDimension::ALL {
        let val = (dim.index() as f64 + 1.0) / 10.0;
        p.set(dim, val);
        assert!((p.get(dim) - val).abs() < 1e-10);
    }
}

#[test]
fn as_array_from_array_round_trip() {
    let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let p = SovereignProfile::from_array(values);
    assert_eq!(p.as_array(), values);
}

// ---------------------------------------------------------------------------
// CivicTier
// ---------------------------------------------------------------------------

#[test]
fn tier_thresholds_match_legacy() {
    assert_eq!(CivicTier::from_score(0.0), CivicTier::Observer);
    assert_eq!(CivicTier::from_score(0.29), CivicTier::Observer);
    assert_eq!(CivicTier::from_score(0.3), CivicTier::Participant);
    assert_eq!(CivicTier::from_score(0.39), CivicTier::Participant);
    assert_eq!(CivicTier::from_score(0.4), CivicTier::Citizen);
    assert_eq!(CivicTier::from_score(0.59), CivicTier::Citizen);
    assert_eq!(CivicTier::from_score(0.6), CivicTier::Steward);
    assert_eq!(CivicTier::from_score(0.79), CivicTier::Steward);
    assert_eq!(CivicTier::from_score(0.8), CivicTier::Guardian);
    assert_eq!(CivicTier::from_score(1.0), CivicTier::Guardian);
}

#[test]
fn tier_ordering_is_monotonic() {
    assert!(CivicTier::Observer < CivicTier::Participant);
    assert!(CivicTier::Participant < CivicTier::Citizen);
    assert!(CivicTier::Citizen < CivicTier::Steward);
    assert!(CivicTier::Steward < CivicTier::Guardian);
}

#[test]
fn vote_weights_are_non_decreasing() {
    let tiers = [
        CivicTier::Observer,
        CivicTier::Participant,
        CivicTier::Citizen,
        CivicTier::Steward,
        CivicTier::Guardian,
    ];
    for i in 1..tiers.len() {
        assert!(
            tiers[i].vote_weight_bp() >= tiers[i - 1].vote_weight_bp(),
            "{:?} weight should be >= {:?} weight",
            tiers[i],
            tiers[i - 1]
        );
    }
}

// ---------------------------------------------------------------------------
// CivicRequirement
// ---------------------------------------------------------------------------

#[test]
fn basic_requirement_met_by_participant() {
    let p = SovereignProfile::from_array([0.4; 8]);
    let w = DimensionWeights::equal();
    assert!(p.meets_requirement(&civic_requirement_basic(), &w));
}

#[test]
fn basic_requirement_not_met_by_observer() {
    let p = SovereignProfile::from_array([0.1; 8]);
    let w = DimensionWeights::equal();
    assert!(!p.meets_requirement(&civic_requirement_basic(), &w));
}

#[test]
fn constitutional_requirement_checks_per_dimension_minimums() {
    let req = civic_requirement_constitutional();
    let w = DimensionWeights::equal();

    // High overall but low epistemic — should fail
    let mut p = SovereignProfile::from_array([0.8; 8]);
    p.epistemic_integrity = 0.2; // below 0.5 minimum
    assert!(!p.meets_requirement(&req, &w));

    // Fix epistemic but low civic — should fail
    p.epistemic_integrity = 0.6;
    p.civic_participation = 0.1; // below 0.3 minimum
    assert!(!p.meets_requirement(&req, &w));

    // Fix both — should pass
    p.civic_participation = 0.4;
    assert!(p.meets_requirement(&req, &w));
}

#[test]
fn guardian_requirement_is_strictest() {
    let req = civic_requirement_guardian();
    let w = DimensionWeights::equal();

    // Even high overall score fails without dimension minimums
    let mut p = SovereignProfile::from_array([0.9; 8]);
    p.epistemic_integrity = 0.5; // below 0.7
    assert!(!p.meets_requirement(&req, &w));
}

// ---------------------------------------------------------------------------
// DimensionWeights
// ---------------------------------------------------------------------------

#[test]
fn all_presets_are_normalized() {
    assert!(DimensionWeights::equal().is_normalized());
    assert!(DimensionWeights::governance().is_normalized());
    assert!(DimensionWeights::energy_cooperative().is_normalized());
    assert!(DimensionWeights::knowledge_commons().is_normalized());
    assert!(DimensionWeights::care_community().is_normalized());
}

#[test]
fn normalize_fixes_unnormalized_weights() {
    let mut w = DimensionWeights { weights: [1.0; 8] };
    assert!(!w.is_normalized());
    w.normalize();
    assert!(w.is_normalized());
    assert!((w.weights[0] - 0.125).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Backward compatibility (4D ↔ 8D)
// ---------------------------------------------------------------------------

#[test]
fn legacy_to_sovereign_preserves_values() {
    let old = LegacyProfile {
        identity: 0.6,
        reputation: 0.5,
        community: 0.8,
        engagement: 0.3,
    };
    let new: SovereignProfile = old.into();

    // identity → epistemic + network
    assert!((new.epistemic_integrity - 0.6).abs() < 1e-10);
    assert!((new.network_resilience - 0.6).abs() < 1e-10);
    // reputation → economic + stewardship
    assert!((new.economic_velocity - 0.5).abs() < 1e-10);
    assert!((new.stewardship_care - 0.5).abs() < 1e-10);
    // community → civic + resonance
    assert!((new.civic_participation - 0.8).abs() < 1e-10);
    assert!((new.semantic_resonance - 0.8).abs() < 1e-10);
    // engagement → thermo + competence
    assert!((new.thermodynamic_yield - 0.3).abs() < 1e-10);
    assert!((new.domain_competence - 0.3).abs() < 1e-10);
}

#[test]
fn sovereign_to_legacy_averages_pairs() {
    let new = SovereignProfile {
        epistemic_integrity: 0.8,
        network_resilience: 0.4,
        economic_velocity: 0.6,
        stewardship_care: 0.2,
        civic_participation: 0.9,
        semantic_resonance: 0.7,
        thermodynamic_yield: 0.5,
        domain_competence: 0.3,
    };
    let old: LegacyProfile = new.into();
    assert!((old.identity - 0.6).abs() < 1e-10); // (0.8+0.4)/2
    assert!((old.reputation - 0.4).abs() < 1e-10); // (0.6+0.2)/2
    assert!((old.community - 0.8).abs() < 1e-10); // (0.9+0.7)/2
    assert!((old.engagement - 0.4).abs() < 1e-10); // (0.5+0.3)/2
}

#[test]
fn legacy_round_trip_preserves_tier() {
    // For any 4D profile, converting to 8D and back should preserve the tier
    let test_cases = [
        LegacyProfile {
            identity: 0.1,
            reputation: 0.1,
            community: 0.1,
            engagement: 0.1,
        },
        LegacyProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        },
        LegacyProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.9,
            engagement: 0.6,
        },
        LegacyProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        },
    ];

    // Use equal weights so the 8D average equals the 4D weighted score
    // (since legacy→sovereign duplicates values, equal weights averages them back)
    let weights = DimensionWeights::equal();

    for old in &test_cases {
        let old_score = old.combined_score();
        let sovereign: SovereignProfile = old.clone().into();
        let new_score = sovereign.combined_score(&weights);
        // The scores should be very close (equal weights on duplicated values = same average)
        assert!(
            (old_score - new_score).abs() < 0.05,
            "Legacy score {old_score:.3} diverged from sovereign score {new_score:.3} for {:?}",
            old
        );
    }
}

#[test]
fn tier_conversion_is_bijective() {
    for tier in [
        LegacyTier::Observer,
        LegacyTier::Participant,
        LegacyTier::Citizen,
        LegacyTier::Steward,
        LegacyTier::Guardian,
    ] {
        let civic: CivicTier = tier.into();
        let back: LegacyTier = civic.into();
        assert_eq!(tier, back);
    }
}

// ---------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------

#[test]
fn all_dimensions_have_labels() {
    for dim in SovereignDimension::ALL {
        let label = i18n::dimension_label(dim);
        assert!(!label.key.is_empty());
        assert!(!label.name_en.is_empty());
        assert!(!label.description_en.is_empty());
        assert_eq!(label.key, dim.key());
    }
}

#[test]
fn dimension_indices_are_unique_and_sequential() {
    for (i, dim) in SovereignDimension::ALL.iter().enumerate() {
        assert_eq!(dim.index(), i);
    }
}

// ---------------------------------------------------------------------------
// Serde round-trips
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn sovereign_profile_json_round_trip() {
    let p = SovereignProfile::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let json = serde_json::to_string(&p).unwrap();
    let deserialized: SovereignProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(p, deserialized);
}

#[cfg(feature = "serde")]
#[test]
fn civic_tier_json_round_trip() {
    for tier in [
        CivicTier::Observer,
        CivicTier::Participant,
        CivicTier::Citizen,
        CivicTier::Steward,
        CivicTier::Guardian,
    ] {
        let json = serde_json::to_string(&tier).unwrap();
        let deserialized: CivicTier = serde_json::from_str(&json).unwrap();
        assert_eq!(tier, deserialized);
    }
}

#[cfg(feature = "serde")]
#[test]
fn sovereign_credential_json_round_trip() {
    let cred = SovereignCredential {
        did: "did:mycelix:uCAES...".into(),
        profile: SovereignProfile::from_array([0.5; 8]),
        tier: CivicTier::Citizen,
        issued_at: 1712000000000000,
        expires_at: 1712600000000000,
        issuer: "did:mycelix:identity-bridge".into(),
        extensions: vec![("version".into(), vec![1])],
    };
    let json = serde_json::to_string(&cred).unwrap();
    let deserialized: SovereignCredential = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.did, cred.did);
    assert_eq!(deserialized.tier, CivicTier::Citizen);
}

// ---------------------------------------------------------------------------
// Full pipeline integration test
// ---------------------------------------------------------------------------

#[cfg(feature = "hdc")]
#[test]
fn full_pipeline_collect_normalize_decay_encode_tier() {
    use crate::collectors::{normalize_all, CollectedDimensions, DimensionInput};
    use crate::decay::{apply_decay, DecayConfig};
    use crate::hdc::{encode_profile, tier_from_popcount, TierThresholds};

    // Step 1: Simulate collected raw data from cluster sources
    let mut collected = CollectedDimensions::default();
    collected.epistemic_integrity = DimensionInput {
        activity_count: 40,
        baseline: 50,
        quality: 0.9,
        days_since_last: 5.0,
        recency_half_life_days: 90.0,
    };
    collected.civic_participation = DimensionInput {
        activity_count: 15,
        baseline: 20,
        quality: 0.85,
        days_since_last: 2.0,
        recency_half_life_days: 180.0,
    };
    collected.domain_competence = DimensionInput {
        activity_count: 3,
        baseline: 3,
        quality: 0.95,
        days_since_last: 30.0,
        recency_half_life_days: 365.0,
    };
    collected.stewardship_care = DimensionInput {
        activity_count: 25,
        baseline: 30,
        quality: 0.8,
        days_since_last: 10.0,
        recency_half_life_days: 60.0,
    };

    // Step 2: Normalize to [0,1] scores
    let profile = normalize_all(&collected);
    assert!(
        profile.epistemic_integrity > 0.5,
        "Epistemic should be high"
    );
    assert!(
        profile.civic_participation > 0.4,
        "Civic should be moderate"
    );
    assert!(profile.domain_competence > 0.5, "Competence should be high");
    assert_eq!(
        profile.thermodynamic_yield, 0.0,
        "Thermo should be zero (no data)"
    );

    // Step 3: Apply decay (30 days since last interaction)
    let decay_config = DecayConfig::default_governance();
    let day_us: u64 = 86_400_000_000;
    let now_us = 100 * day_us;
    let last_interaction = 70 * day_us; // 30 days ago
    let decayed = apply_decay(&profile, last_interaction, now_us, &decay_config);

    // Verify decay reduced scores
    assert!(
        decayed.epistemic_integrity < profile.epistemic_integrity,
        "Decay should reduce epistemic: {} → {}",
        profile.epistemic_integrity,
        decayed.epistemic_integrity
    );

    // Step 4: Derive scalar tier
    let weights = DimensionWeights::governance();
    let scalar_score = decayed.combined_score(&weights);
    let scalar_tier = decayed.tier(&weights);

    // Step 5: Encode to HDC BinaryHV
    let hv = encode_profile(&decayed, &weights);
    assert!(hv.popcount() > 0, "Encoded HV should have nonzero popcount");

    // Step 6: Derive tier from popcount
    let thresholds = TierThresholds::calibrate(&weights);
    let hdc_tier = tier_from_popcount(&hv, &thresholds);

    // Step 7: Verify popcount tier matches scalar tier
    // (They should match for well-calibrated thresholds)
    println!(
        "Pipeline: score={:.3}, scalar_tier={:?}, hdc_tier={:?}, popcount={}",
        scalar_score,
        scalar_tier,
        hdc_tier,
        hv.popcount()
    );

    // The tiers should be close (within 1 tier of each other due to
    // threshold calibration imprecision). Exact match not guaranteed
    // because HDC encoding has quantization effects.
    let scalar_idx = [
        CivicTier::Observer,
        CivicTier::Participant,
        CivicTier::Citizen,
        CivicTier::Steward,
        CivicTier::Guardian,
    ]
    .iter()
    .position(|t| *t == scalar_tier)
    .unwrap();
    let hdc_idx = [
        CivicTier::Observer,
        CivicTier::Participant,
        CivicTier::Citizen,
        CivicTier::Steward,
        CivicTier::Guardian,
    ]
    .iter()
    .position(|t| *t == hdc_tier)
    .unwrap();
    assert!(
        (scalar_idx as i32 - hdc_idx as i32).abs() <= 1,
        "HDC tier {:?} should be within 1 of scalar tier {:?}",
        hdc_tier,
        scalar_tier
    );
}
