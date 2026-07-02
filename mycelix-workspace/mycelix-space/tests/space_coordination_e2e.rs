// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end integration tests for the space coordination flow.
//!
//! These tests verify the full pipeline: trust requirements → observation fusion →
//! conjunction screening → multi-party coordination types.
//! Runs without a Holochain conductor (library-level integration).

use std::collections::HashMap;

use chrono::{Duration, Utc};
use orbital_mechanics::{
    conjunction::{ConjunctionAnalyzer, RiskLevel},
    fusion::{FusionPipeline, SensorMeasurement, TrustWeighting},
    propagator::Propagator,
    state::{DataSource, StateVector},
    tle::TwoLineElement,
    CovarianceMatrix,
};
use mycelix_space_shared::{
    compute_staleness, QualityScore, StalenessConfig, SpaceTimestamp,
    requirement_for_observation, requirement_for_conjunction_creation,
    requirement_for_risk_update, requirement_for_bounty_creation,
    requirement_for_agreement_signing, requirement_for_tle_submission,
};

/// Sample ISS TLE for testing
const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

/// Fictional debris TLE at similar orbit (same epoch for screening)
const DEBRIS_TLE: &str = "DEBRIS
1 49863U 21999A   24001.50000000  .00001000  00000-0  50000-4 0  9992
2 49863  51.6500 247.5000 0007000 131.0000 325.0000 15.72000000 10005";

// =============================================================================
// Test 1: Trust fabric requirement tier ordering
// =============================================================================

/// Verify that higher-stakes operations require higher tiers, and that
/// identity/community requirements increase with tier.
#[test]
fn test_trust_requirement_tier_ordering() {
    let tle_sub = requirement_for_tle_submission();
    let observation = requirement_for_observation();
    let conjunction = requirement_for_conjunction_creation();
    let risk_update = requirement_for_risk_update();
    let bounty = requirement_for_bounty_creation();
    let agreement = requirement_for_agreement_signing();

    // Tier ordering: Observer < Participant < Citizen < Steward
    assert!(
        tle_sub.min_tier <= observation.min_tier,
        "TLE submission should require at most observation-level trust"
    );
    assert!(
        observation.min_tier < conjunction.min_tier,
        "Observation (Observer) should be lower than conjunction creation (Participant)"
    );
    assert!(
        conjunction.min_tier < risk_update.min_tier,
        "Conjunction creation (Participant) should be lower than risk update (Citizen)"
    );
    assert!(
        risk_update.min_tier < bounty.min_tier,
        "Risk update (Citizen) should be lower than bounty creation (Steward)"
    );
    assert_eq!(
        bounty.min_tier, agreement.min_tier,
        "Bounty creation and agreement signing should both be Steward"
    );

    // Observer-level requirements have no identity/community minimums
    assert_eq!(tle_sub.min_identity, None);
    assert_eq!(tle_sub.min_community, None);
    assert_eq!(observation.min_identity, None);
    assert_eq!(observation.min_community, None);

    // Steward-level requirements have both identity and community minimums
    assert!(
        bounty.min_identity.is_some(),
        "Bounty creation should require min_identity"
    );
    assert!(
        bounty.min_community.is_some(),
        "Bounty creation should require min_community"
    );
    assert!(
        agreement.min_identity.is_some(),
        "Agreement signing should require min_identity"
    );
    assert!(
        agreement.min_community.is_some(),
        "Agreement signing should require min_community"
    );

    // Identity requirements should increase with tier
    let conj_identity = conjunction.min_identity.unwrap_or(0.0);
    let bounty_identity = bounty.min_identity.unwrap_or(0.0);
    assert!(
        bounty_identity > conj_identity,
        "Steward identity requirement ({}) should exceed Participant ({})",
        bounty_identity,
        conj_identity
    );
}

// =============================================================================
// Test 2: Trust-weighted fusion changes output quality
// =============================================================================

/// Create two identical sensor measurements and fuse them with and without
/// trust weighting. Trust should change the fused quality.
#[test]
fn test_trust_weighted_fusion_quality_difference() {
    let now = Utc::now();

    let make_measurement = |sensor_id: &str, quality: f64| SensorMeasurement {
        time: now,
        state: StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        covariance: CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]),
        sensor_id: sensor_id.to_string(),
        data_source: DataSource::SpaceTrack,
        quality,
    };

    let measurements = vec![
        make_measurement("sensor-A", 0.8),
        make_measurement("sensor-B", 0.8),
    ];

    // Fuse without trust weighting
    let pipeline_plain = FusionPipeline::new();
    let result_plain = pipeline_plain.fuse(&measurements).expect("Plain fusion failed");

    // Fuse with trust weighting: sensor-A is fully trusted, sensor-B is unknown
    let mut trust = TrustWeighting::default();
    trust.sensor_trust.insert("sensor-A".to_string(), 1.0);
    // sensor-B not in map → uses trust_floor (0.3)

    let pipeline_trust = FusionPipeline::new().with_trust_weighting(trust);
    let result_trust = pipeline_trust.fuse(&measurements).expect("Trust fusion failed");

    // The fused quality should differ because sensor-B gets reduced weight
    assert!(
        (result_plain.fused_quality - result_trust.fused_quality).abs() > 1e-6,
        "Trust weighting should change fused quality: plain={}, trust={}",
        result_plain.fused_quality,
        result_trust.fused_quality
    );

    // Plain fusion should have higher overall quality (both sensors at full weight)
    assert!(
        result_plain.fused_quality > result_trust.fused_quality,
        "Plain quality ({}) should exceed trust-weighted quality ({}) when one sensor is untrusted",
        result_plain.fused_quality,
        result_trust.fused_quality
    );
}

// =============================================================================
// Test 3: Trust floor prevents zero contribution
// =============================================================================

/// An unknown sensor (not in trust map) should still contribute via the
/// trust floor, producing effective quality > 0.
#[test]
fn test_trust_floor_preserves_minimum_contribution() {
    let trust = TrustWeighting {
        trust_floor: 0.3,
        sensor_trust: HashMap::new(), // no sensors registered
    };

    let raw_quality = 0.9;
    let effective = trust.effective_quality(raw_quality, "unknown-sensor");

    // effective = 0.9 * (0.3 + (1 - 0.3) * 0.0) = 0.9 * 0.3 = 0.27
    assert!(
        effective > 0.0,
        "Trust floor should prevent zero contribution: got {}",
        effective
    );

    let expected = raw_quality * trust.trust_floor;
    assert!(
        (effective - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        effective
    );

    // A fully trusted sensor should get full quality
    let mut trust_with_known = trust.clone();
    trust_with_known
        .sensor_trust
        .insert("known-sensor".to_string(), 1.0);
    let effective_known = trust_with_known.effective_quality(raw_quality, "known-sensor");
    assert!(
        (effective_known - raw_quality).abs() < 1e-10,
        "Fully trusted sensor should get full quality: expected {}, got {}",
        raw_quality,
        effective_known
    );
}

// =============================================================================
// Test 4: Full screening pipeline with trust
// =============================================================================

/// Parse ISS and debris TLEs, propagate to a common epoch, and run
/// conjunction assessment. Verify the pipeline produces finite, bounded results.
#[test]
fn test_screening_with_trust_weighted_observations() {
    let iss_tle = TwoLineElement::parse(ISS_TLE).expect("Failed to parse ISS TLE");
    let debris_tle = TwoLineElement::parse(DEBRIS_TLE).expect("Failed to parse debris TLE");

    let iss_prop = Propagator::from_tle(&iss_tle).expect("ISS propagator failed");
    let debris_prop = Propagator::from_tle(&debris_tle).expect("Debris propagator failed");

    // Propagate both to 1 hour after epoch
    let screen_time = iss_tle.epoch + Duration::hours(1);
    let iss_state = iss_prop
        .propagate_to(screen_time)
        .expect("ISS propagation failed");
    let debris_state = debris_prop
        .propagate_to(screen_time)
        .expect("Debris propagation failed");

    // Run conjunction assessment
    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&iss_state, &debris_state);

    // Verify basic pipeline invariants
    assert_eq!(assessment.primary_norad_id, 25544);
    assert_eq!(assessment.secondary_norad_id, 49863);

    assert!(
        assessment.miss_distance_km.is_finite() && assessment.miss_distance_km >= 0.0,
        "Miss distance must be finite and non-negative: {}",
        assessment.miss_distance_km
    );
    assert!(
        assessment.relative_velocity_kms.is_finite() && assessment.relative_velocity_kms >= 0.0,
        "Relative velocity must be finite and non-negative: {}",
        assessment.relative_velocity_kms
    );

    let pc = assessment.collision_probability.pc;
    assert!(
        pc.is_finite() && pc >= 0.0 && pc <= 1.0,
        "Pc must be finite and in [0, 1]: {}",
        pc
    );

    // Risk level should be consistent with Pc
    let expected_risk = RiskLevel::from_pc(pc);
    assert_eq!(
        assessment.risk_level, expected_risk,
        "Risk level mismatch: assessed {:?} but Pc {} implies {:?}",
        assessment.risk_level, pc, expected_risk
    );

    // HBR should be positive
    assert!(
        assessment.hard_body_radius_m > 0.0,
        "Hard-body radius must be positive"
    );
}

// =============================================================================
// Test 5: Multi-party proposal status machine
// =============================================================================

/// Mirror of `MultiPartyProposalStatus` from
/// `traffic_control_integrity` — tests the state machine transitions without
/// importing the Holochain zome crate directly.
#[derive(Clone, Debug, PartialEq)]
enum ProposalStatus {
    Voting,
    Approved,
    Rejected,
    Expired,
    Executing,
    Completed,
}

impl ProposalStatus {
    fn is_valid_transition(&self, next: &Self) -> bool {
        matches!(
            (self, next),
            (Self::Voting, Self::Approved)
                | (Self::Voting, Self::Rejected)
                | (Self::Voting, Self::Expired)
                | (Self::Approved, Self::Executing)
                | (Self::Approved, Self::Expired)
                | (Self::Executing, Self::Completed)
        )
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Rejected | Self::Expired | Self::Completed
        )
    }
}

/// Verify the proposal state machine: valid transitions, rejected invalid
/// transitions, and terminal state enforcement.
#[test]
fn test_multi_party_proposal_lifecycle() {
    use ProposalStatus::*;

    // Valid forward transitions
    assert!(Voting.is_valid_transition(&Approved));
    assert!(Voting.is_valid_transition(&Rejected));
    assert!(Voting.is_valid_transition(&Expired));
    assert!(Approved.is_valid_transition(&Executing));
    assert!(Approved.is_valid_transition(&Expired));
    assert!(Executing.is_valid_transition(&Completed));

    // Invalid transitions
    assert!(!Voting.is_valid_transition(&Completed), "Cannot skip from Voting to Completed");
    assert!(!Voting.is_valid_transition(&Executing), "Cannot skip from Voting to Executing");
    assert!(!Approved.is_valid_transition(&Rejected), "Cannot go from Approved back to Rejected");
    assert!(!Executing.is_valid_transition(&Voting), "Cannot go from Executing back to Voting");
    assert!(!Completed.is_valid_transition(&Voting), "Terminal state should block transitions");

    // Terminal states block all further transitions
    let terminal_states = vec![Rejected, Expired, Completed];
    let all_states = vec![Voting, Approved, Rejected, Expired, Executing, Completed];
    for terminal in &terminal_states {
        assert!(terminal.is_terminal(), "{:?} should be terminal", terminal);
        for target in &all_states {
            assert!(
                !terminal.is_valid_transition(target),
                "Terminal state {:?} should not transition to {:?}",
                terminal,
                target
            );
        }
    }

    // Non-terminal states should not be terminal
    assert!(!Voting.is_terminal());
    assert!(!Approved.is_terminal());
    assert!(!Executing.is_terminal());
}

// =============================================================================
// Test 6: Vote weight accumulation
// =============================================================================

/// Simulate 5 operators with different weights voting on 3 options.
/// Verify winner selection and quorum calculation.
#[test]
fn test_vote_weight_tally() {
    struct Vote {
        option_index: u32,
        weight: f64,
    }

    let votes = vec![
        Vote { option_index: 0, weight: 0.9 },  // Verified operator, option A
        Vote { option_index: 1, weight: 0.6 },  // Established operator, option B
        Vote { option_index: 0, weight: 0.3 },  // BasicTrust, option A
        Vote { option_index: 2, weight: 1.0 },  // FoundingMember, option C
        Vote { option_index: 0, weight: 0.6 },  // Established operator, option A
    ];

    // Tally weighted votes per option
    let mut tallies: HashMap<u32, f64> = HashMap::new();
    let mut total_weight = 0.0;
    for vote in &votes {
        *tallies.entry(vote.option_index).or_default() += vote.weight;
        total_weight += vote.weight;
    }

    // Find winner
    let (winner, winner_weight) = tallies
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, v)| (*k, *v))
        .unwrap();

    // Option A: 0.9 + 0.3 + 0.6 = 1.8
    // Option B: 0.6
    // Option C: 1.0
    assert_eq!(winner, 0, "Option A should win with highest weighted total");
    assert!(
        (winner_weight - 1.8).abs() < 1e-10,
        "Winner weight should be 1.8, got {}",
        winner_weight
    );

    // Quorum: total cast weight / max possible weight (sum of all weights)
    let max_possible = total_weight; // all operators voted
    let quorum = total_weight / max_possible;
    assert!(
        (quorum - 1.0).abs() < 1e-10,
        "Full participation should give quorum of 1.0"
    );

    // Partial participation: if founding member abstains
    let partial_weight: f64 = votes.iter().filter(|v| v.weight < 1.0).map(|v| v.weight).sum();
    let partial_quorum = partial_weight / total_weight;
    assert!(
        partial_quorum < 1.0,
        "Partial participation ({}) should give quorum < 1.0",
        partial_quorum
    );
    // partial = (0.9 + 0.6 + 0.3 + 0.6) / 3.4 = 2.4 / 3.4 ≈ 0.706
    assert!(
        (partial_quorum - 2.4 / 3.4).abs() < 1e-10,
        "Expected ~0.706, got {}",
        partial_quorum
    );
}

// =============================================================================
// Test 7: Risk level classification consistency
// =============================================================================

/// Verify that `RiskLevel::from_pc()` produces the correct classification at
/// each threshold boundary.
#[test]
fn test_risk_level_thresholds() {
    // Well below negligible threshold
    assert_eq!(
        RiskLevel::from_pc(1e-10),
        RiskLevel::Negligible,
        "Pc=1e-10 should be Negligible"
    );

    // At the Negligible/Low boundary (1e-7)
    assert_eq!(
        RiskLevel::from_pc(0.5e-7),
        RiskLevel::Negligible,
        "Pc=0.5e-7 should be Negligible (below 1e-7)"
    );
    assert_eq!(
        RiskLevel::from_pc(1e-7),
        RiskLevel::Low,
        "Pc=1e-7 should be Low (at boundary)"
    );

    // Low range
    assert_eq!(
        RiskLevel::from_pc(1e-6),
        RiskLevel::Low,
        "Pc=1e-6 should be Low"
    );

    // At the Low/Medium boundary (1e-5)
    assert_eq!(
        RiskLevel::from_pc(1e-5),
        RiskLevel::Medium,
        "Pc=1e-5 should be Medium (at boundary)"
    );

    // Medium range
    assert_eq!(
        RiskLevel::from_pc(5e-5),
        RiskLevel::Medium,
        "Pc=5e-5 should be Medium"
    );

    // At the Medium/High boundary (1e-4)
    assert_eq!(
        RiskLevel::from_pc(1e-4),
        RiskLevel::High,
        "Pc=1e-4 should be High (at boundary)"
    );

    // High range
    assert_eq!(
        RiskLevel::from_pc(5e-4),
        RiskLevel::High,
        "Pc=5e-4 should be High"
    );

    // At the High/Emergency boundary (1e-3)
    assert_eq!(
        RiskLevel::from_pc(1e-3),
        RiskLevel::Emergency,
        "Pc=1e-3 should be Emergency (at boundary)"
    );

    // Emergency range
    assert_eq!(
        RiskLevel::from_pc(0.5),
        RiskLevel::Emergency,
        "Pc=0.5 should be Emergency"
    );

    // Ordering: Negligible < Low < Medium < High < Emergency
    assert!(RiskLevel::Negligible < RiskLevel::Low);
    assert!(RiskLevel::Low < RiskLevel::Medium);
    assert!(RiskLevel::Medium < RiskLevel::High);
    assert!(RiskLevel::High < RiskLevel::Emergency);
}

// =============================================================================
// Test 8: TLE staleness degrades conjunction confidence
// =============================================================================

/// Verify that stale TLEs produce lower confidence multipliers than fresh ones.
#[test]
fn test_staleness_degrades_screening_confidence() {
    let config = StalenessConfig::default();
    let quality = QualityScore::new(90);

    // Fresh TLE: epoch = now
    let fresh_epoch = SpaceTimestamp::now();
    let fresh = compute_staleness(&fresh_epoch, &quality, &config);
    assert!(
        !fresh.is_stale,
        "Fresh TLE should not be stale"
    );
    assert!(
        fresh.confidence > 0.9,
        "Fresh TLE confidence should be > 0.9, got {}",
        fresh.confidence
    );
    assert_eq!(
        fresh.effective_quality.value(),
        quality.value(),
        "Fresh TLE effective quality should match original"
    );

    // Moderately old TLE: 15 days old
    let moderate_epoch = SpaceTimestamp {
        micros: Utc::now().timestamp_micros() - 15 * 24 * 3600 * 1_000_000,
    };
    let moderate = compute_staleness(&moderate_epoch, &quality, &config);
    assert!(
        !moderate.is_stale,
        "15-day-old TLE should not be stale (threshold is 30 days)"
    );
    assert!(
        moderate.confidence < fresh.confidence,
        "Moderate TLE confidence ({}) should be less than fresh ({})",
        moderate.confidence,
        fresh.confidence
    );

    // Stale TLE: 35 days old (past the 30-day threshold)
    let stale_epoch = SpaceTimestamp {
        micros: Utc::now().timestamp_micros() - 35 * 24 * 3600 * 1_000_000,
    };
    let stale = compute_staleness(&stale_epoch, &quality, &config);
    assert!(
        stale.is_stale,
        "35-day-old TLE should be stale"
    );
    assert!(
        stale.confidence < moderate.confidence,
        "Stale TLE confidence ({}) should be less than moderate ({})",
        stale.confidence,
        moderate.confidence
    );
    assert!(
        stale.effective_quality.value() < quality.value(),
        "Stale TLE effective quality ({}) should be degraded from original ({})",
        stale.effective_quality.value(),
        quality.value()
    );

    // Very old TLE: 60 days old (2x threshold)
    let very_old_epoch = SpaceTimestamp {
        micros: Utc::now().timestamp_micros() - 60 * 24 * 3600 * 1_000_000,
    };
    let very_old = compute_staleness(&very_old_epoch, &quality, &config);
    assert!(
        very_old.is_stale,
        "60-day-old TLE should be stale"
    );
    assert!(
        very_old.confidence <= stale.confidence,
        "Very old TLE confidence ({}) should be <= stale ({})",
        very_old.confidence,
        stale.confidence
    );

    // Confidence should never drop below 0.1 (the minimum)
    assert!(
        very_old.confidence >= 0.1,
        "Confidence should not drop below 0.1, got {}",
        very_old.confidence
    );

    // Monotonic degradation: fresh > moderate > stale > very_old
    assert!(
        fresh.confidence > moderate.confidence
            && moderate.confidence > stale.confidence
            && stale.confidence >= very_old.confidence,
        "Confidence should degrade monotonically with age: {} > {} > {} >= {}",
        fresh.confidence,
        moderate.confidence,
        stale.confidence,
        very_old.confidence
    );
}
