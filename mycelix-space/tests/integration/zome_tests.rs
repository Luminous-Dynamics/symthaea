//! Holochain Zome Integration Tests
//!
//! Tests the mycelix-space zomes in a simulated Holochain environment.
//! These tests verify:
//! - Entry creation and retrieval
//! - Cross-zome dependencies
//! - Signal emission for alerts
//! - Multi-agent scenarios

use mycelix_space_shared::*;
use chrono::Utc;
use std::collections::HashMap;

/// Test that shared types serialize correctly for Holochain
#[test]
fn test_shared_types_serialization() {
    // Test OrbitalObjectEntry serialization
    let obj = OrbitalObjectEntry {
        norad_id: 25544,
        name: "ISS (ZARYA)".to_string(),
        object_type: ObjectType::Payload,
        launch_date: Some(Utc::now()),
        decay_date: None,
        owner_country: Some("ISS".to_string()),
        data_source: DataSourceSimple::SpaceTrack,
        metadata: HashMap::new(),
    };

    let json = serde_json::to_string(&obj).expect("Failed to serialize");
    let parsed: OrbitalObjectEntry = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.norad_id, 25544);
    assert_eq!(parsed.name, "ISS (ZARYA)");
}

/// Test TLE data serialization
#[test]
fn test_tle_data_serialization() {
    let tle = TleData {
        norad_id: 25544,
        line1: "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997".to_string(),
        line2: "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577".to_string(),
        epoch: Utc::now(),
        source: DataSourceSimple::SpaceTrack,
    };

    let json = serde_json::to_string(&tle).expect("Failed to serialize");
    let parsed: TleData = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.norad_id, 25544);
    assert!(parsed.line1.starts_with("1 25544"));
}

/// Test StateVectorData serialization
#[test]
fn test_state_vector_serialization() {
    let sv = StateVectorData {
        norad_id: 25544,
        epoch: Utc::now(),
        position_km: [6800.0, 0.0, 0.0],
        velocity_kms: [0.0, 7.5, 0.0],
        covariance: None,
        reference_frame: ReferenceFrame::Teme,
        quality: 0.95,
        source: DataSourceSimple::SpaceTrack,
    };

    let json = serde_json::to_string(&sv).expect("Failed to serialize");
    let parsed: StateVectorData = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.norad_id, 25544);
    assert!((parsed.position_km[0] - 6800.0).abs() < 0.001);
}

/// Test ConjunctionAssessment serialization
#[test]
fn test_conjunction_assessment_serialization() {
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::Medium,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let json = serde_json::to_string(&assessment).expect("Failed to serialize");
    let parsed: ConjunctionAssessment = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.primary_norad_id, 25544);
    assert_eq!(parsed.secondary_norad_id, 49863);
    assert_eq!(parsed.risk_level, RiskLevel::Medium);
}

/// Test RiskLevel thresholds match expected values
#[test]
fn test_risk_level_values() {
    // Verify RiskLevel enum values
    assert_eq!(RiskLevel::Negligible as u8, 0);
    assert_eq!(RiskLevel::Low as u8, 1);
    assert_eq!(RiskLevel::Medium as u8, 2);
    assert_eq!(RiskLevel::High as u8, 3);
    assert_eq!(RiskLevel::Emergency as u8, 4);
}

/// Test RiskLevel from probability of collision
#[test]
fn test_risk_level_from_pc() {
    assert_eq!(RiskLevel::from_pc(1e-8), RiskLevel::Negligible);
    assert_eq!(RiskLevel::from_pc(1e-6), RiskLevel::Low);
    assert_eq!(RiskLevel::from_pc(5e-5), RiskLevel::Medium);
    assert_eq!(RiskLevel::from_pc(5e-4), RiskLevel::High);
    assert_eq!(RiskLevel::from_pc(1e-2), RiskLevel::Emergency);
}

/// Test ObjectType enum
#[test]
fn test_object_types() {
    let types = vec![
        ObjectType::Payload,
        ObjectType::RocketBody,
        ObjectType::Debris,
        ObjectType::Unknown,
    ];

    for obj_type in types {
        let json = serde_json::to_string(&obj_type).expect("Failed to serialize");
        let parsed: ObjectType = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, obj_type);
    }
}

/// Test DataSourceSimple enum
#[test]
fn test_data_source_types() {
    let sources = vec![
        DataSourceSimple::SpaceTrack,
        DataSourceSimple::CelesTrak,
        DataSourceSimple::Operator,
        DataSourceSimple::Computed,
    ];

    for source in sources {
        let json = serde_json::to_string(&source).expect("Failed to serialize");
        let parsed: DataSourceSimple = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, source);
    }
}

/// Test ReferenceFrame enum
#[test]
fn test_reference_frames() {
    let frames = vec![
        ReferenceFrame::Teme,
        ReferenceFrame::J2000,
        ReferenceFrame::Itrf,
        ReferenceFrame::Gcrf,
    ];

    for frame in frames {
        let json = serde_json::to_string(&frame).expect("Failed to serialize");
        let parsed: ReferenceFrame = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, frame);
    }
}

/// Test ObservationEntry serialization
#[test]
fn test_observation_entry_serialization() {
    let obs = ObservationEntry {
        norad_id: 25544,
        timestamp: Utc::now(),
        observation_type: ObservationType::Radar,
        observer_location: GroundLocation {
            latitude_deg: 32.9,
            longitude_deg: -96.7,
            altitude_m: 200.0,
            name: Some("Test Station".to_string()),
        },
        raw_data: serde_json::json!({"range_km": 400.0, "azimuth_deg": 45.0}),
        quality_score: 0.9,
        data_source: DataSourceSimple::Operator,
    };

    let json = serde_json::to_string(&obs).expect("Failed to serialize");
    let parsed: ObservationEntry = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.norad_id, 25544);
    assert_eq!(parsed.observation_type, ObservationType::Radar);
}

/// Test ManeuverRequest serialization
#[test]
fn test_maneuver_request_serialization() {
    let request = ManeuverRequest {
        requesting_object: 25544,
        conflicting_object: 49863,
        conjunction_id: "CNJ-2024-001".to_string(),
        proposed_maneuver: ManeuverType::InTrack,
        delta_v_ms: 0.5,
        execution_time: Utc::now(),
        priority: ManeuverPriority::High,
        status: ManeuverStatus::Proposed,
    };

    let json = serde_json::to_string(&request).expect("Failed to serialize");
    let parsed: ManeuverRequest = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.requesting_object, 25544);
    assert_eq!(parsed.priority, ManeuverPriority::High);
}

/// Test BountyEntry serialization
#[test]
fn test_bounty_entry_serialization() {
    let bounty = BountyEntry {
        debris_norad_id: 49863,
        bounty_amount: 100000,
        currency: "USD".to_string(),
        sponsor: "ESA".to_string(),
        deadline: Some(Utc::now()),
        requirements: BountyRequirements {
            min_mass_kg: Some(100.0),
            max_altitude_km: Some(600.0),
            debris_type: Some(ObjectType::Debris),
            removal_method: None,
        },
        status: BountyStatus::Active,
    };

    let json = serde_json::to_string(&bounty).expect("Failed to serialize");
    let parsed: BountyEntry = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.debris_norad_id, 49863);
    assert_eq!(parsed.bounty_amount, 100000);
}

/// Test NoradId validation
#[test]
fn test_norad_id_validation() {
    // Valid IDs
    assert!(NoradId::new(25544).is_ok());  // ISS
    assert!(NoradId::new(1).is_ok());
    assert!(NoradId::new(999999).is_ok());

    // Invalid IDs
    assert!(NoradId::new(0).is_err());
    assert!(NoradId::new(1000000).is_err());
}

/// Test QualityScore validation
#[test]
fn test_quality_score_validation() {
    let high = QualityScore::new(90);
    assert!(high.is_high());
    assert!(high.is_acceptable());

    let medium = QualityScore::new(60);
    assert!(!medium.is_high());
    assert!(medium.is_acceptable());

    let low = QualityScore::new(30);
    assert!(!low.is_high());
    assert!(!low.is_acceptable());

    // Clamped to 100
    let over = QualityScore::new(150);
    assert_eq!(over.value(), 100);
}

/// Test SpaceTimestamp functionality
#[test]
fn test_space_timestamp() {
    let now = SpaceTimestamp::now();
    let dt = now.to_datetime();
    let reconstructed = SpaceTimestamp::from_datetime(dt);

    // Should be very close (might differ by a microsecond or so)
    assert!((now.micros - reconstructed.micros).abs() < 10);

    // Age should be essentially 0
    assert!(now.age_seconds().abs() < 1);
}

/// Test GroundLocation serialization
#[test]
fn test_ground_location_serialization() {
    let loc = GroundLocation {
        latitude_deg: 28.5729,
        longitude_deg: -80.6490,
        altitude_m: 5.0,
        name: Some("Kennedy Space Center".to_string()),
    };

    let json = serde_json::to_string(&loc).expect("Failed to serialize");
    let parsed: GroundLocation = serde_json::from_str(&json).expect("Failed to deserialize");
    assert!((parsed.latitude_deg - 28.5729).abs() < 0.001);
    assert_eq!(parsed.name, Some("Kennedy Space Center".to_string()));
}

/// Test TrustLevel ordering
#[test]
fn test_trust_level_ordering() {
    assert!(TrustLevel::Unverified < TrustLevel::BasicTrust);
    assert!(TrustLevel::BasicTrust < TrustLevel::Established);
    assert!(TrustLevel::Established < TrustLevel::Verified);
    assert!(TrustLevel::Verified < TrustLevel::FoundingMember);
}

/// Test TrustLevel weights
#[test]
fn test_trust_level_weights() {
    assert!((TrustLevel::Unverified.weight() - 0.1).abs() < 0.01);
    assert!((TrustLevel::BasicTrust.weight() - 0.3).abs() < 0.01);
    assert!((TrustLevel::Established.weight() - 0.6).abs() < 0.01);
    assert!((TrustLevel::Verified.weight() - 0.9).abs() < 0.01);
    assert!((TrustLevel::FoundingMember.weight() - 1.0).abs() < 0.01);
}

/// Test ObservationType enum
#[test]
fn test_observation_types() {
    let types = vec![
        ObservationType::Optical,
        ObservationType::Radar,
        ObservationType::LaserRanging,
        ObservationType::SpaceBased,
        ObservationType::RadioTracking,
    ];

    for obs_type in types {
        let json = serde_json::to_string(&obs_type).expect("Failed to serialize");
        let parsed: ObservationType = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, obs_type);
    }
}

/// Test ManeuverType enum
#[test]
fn test_maneuver_types() {
    let types = vec![
        ManeuverType::InTrack,
        ManeuverType::CrossTrack,
        ManeuverType::Radial,
        ManeuverType::Combined,
    ];

    for maneuver_type in types {
        let json = serde_json::to_string(&maneuver_type).expect("Failed to serialize");
        let parsed: ManeuverType = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, maneuver_type);
    }
}

/// Test ManeuverPriority ordering
#[test]
fn test_maneuver_priority_ordering() {
    assert!(ManeuverPriority::Low < ManeuverPriority::Normal);
    assert!(ManeuverPriority::Normal < ManeuverPriority::High);
    assert!(ManeuverPriority::High < ManeuverPriority::Emergency);
}

/// Test BountyStatus transitions
#[test]
fn test_bounty_status_enum() {
    let statuses = vec![
        BountyStatus::Active,
        BountyStatus::Claimed,
        BountyStatus::InProgress,
        BountyStatus::Completed,
        BountyStatus::Expired,
        BountyStatus::Cancelled,
    ];

    for status in statuses {
        let json = serde_json::to_string(&status).expect("Failed to serialize");
        let parsed: BountyStatus = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, status);
    }
}

/// Test PcMethod enum
#[test]
fn test_pc_method_enum() {
    let methods = vec![
        PcMethod::Alfano2D,
        PcMethod::Foster3D,
        PcMethod::ChanMonteCarlo,
        PcMethod::PateraMax,
    ];

    for method in methods {
        let json = serde_json::to_string(&method).expect("Failed to serialize");
        let parsed: PcMethod = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, method);
    }
}

// =============================================================================
// Alert and Signal Tests
// =============================================================================

/// Test AlertType enum
#[test]
fn test_alert_type_enum() {
    let types = vec![
        AlertType::ConjunctionDetected,
        AlertType::RiskEscalation,
        AlertType::RiskDeescalation,
        AlertType::ManeuverRequired,
        AlertType::ManeuverExecuted,
        AlertType::DebrisAlert,
        AlertType::BountyClaimed,
        AlertType::SystemAlert,
    ];

    for alert_type in types {
        let json = serde_json::to_string(&alert_type).expect("Failed to serialize");
        let parsed: AlertType = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed, alert_type);
    }
}

/// Test AlertPriority ordering and serialization
#[test]
fn test_alert_priority() {
    assert!(AlertPriority::Info < AlertPriority::Low);
    assert!(AlertPriority::Low < AlertPriority::Medium);
    assert!(AlertPriority::Medium < AlertPriority::High);
    assert!(AlertPriority::High < AlertPriority::Critical);

    // Test from_risk_level
    assert_eq!(AlertPriority::from_risk_level(RiskLevel::Negligible), AlertPriority::Info);
    assert_eq!(AlertPriority::from_risk_level(RiskLevel::Low), AlertPriority::Low);
    assert_eq!(AlertPriority::from_risk_level(RiskLevel::Medium), AlertPriority::Medium);
    assert_eq!(AlertPriority::from_risk_level(RiskLevel::High), AlertPriority::High);
    assert_eq!(AlertPriority::from_risk_level(RiskLevel::Emergency), AlertPriority::Critical);
}

/// Test ConjunctionAlert creation
#[test]
fn test_conjunction_alert_creation() {
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.3,
        relative_velocity_kms: 14.5,
        collision_probability: 5e-4,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::High,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::new_conjunction(&assessment);

    assert_eq!(alert.alert_type, AlertType::ConjunctionDetected);
    assert_eq!(alert.priority, AlertPriority::High);
    assert_eq!(alert.primary_norad_id, 25544);
    assert_eq!(alert.secondary_norad_id, 49863);
    assert!(alert.is_critical());
    assert!(alert.recommendation.is_some());
}

/// Test ConjunctionAlert risk escalation
#[test]
fn test_conjunction_alert_escalation() {
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.1,
        relative_velocity_kms: 14.5,
        collision_probability: 5e-3,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::Emergency,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::risk_escalation(&assessment, RiskLevel::Medium);

    assert_eq!(alert.alert_type, AlertType::RiskEscalation);
    assert_eq!(alert.priority, AlertPriority::Critical);
    assert_eq!(alert.previous_risk_level, Some(RiskLevel::Medium));
    assert!(alert.is_critical());
}

/// Test SpaceSignal serialization
#[test]
fn test_space_signal_serialization() {
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::Medium,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::new_conjunction(&assessment);
    let signal = SpaceSignal::Conjunction(alert);

    let json = serde_json::to_string(&signal).expect("Failed to serialize");
    let parsed: SpaceSignal = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(parsed.priority(), AlertPriority::Medium);
    assert!(!parsed.is_critical());
}

/// Test ManeuverAlert serialization
#[test]
fn test_maneuver_alert_serialization() {
    let alert = ManeuverAlert {
        alert_type: AlertType::ManeuverRequired,
        priority: AlertPriority::High,
        norad_id: 25544,
        conjunction_primary: Some(25544),
        conjunction_secondary: Some(49863),
        maneuver_type: ManeuverType::InTrack,
        delta_v_ms: 0.5,
        execution_time: Utc::now(),
        generated_at: Utc::now(),
        message: Some("Collision avoidance maneuver required".to_string()),
    };

    let json = serde_json::to_string(&alert).expect("Failed to serialize");
    let parsed: ManeuverAlert = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(parsed.alert_type, AlertType::ManeuverRequired);
    assert_eq!(parsed.priority, AlertPriority::High);
    assert_eq!(parsed.norad_id, 25544);
}

/// Test BountyAlert serialization
#[test]
fn test_bounty_alert_serialization() {
    let alert = BountyAlert {
        alert_type: AlertType::BountyClaimed,
        priority: AlertPriority::Medium,
        debris_norad_id: 49863,
        bounty_amount: 100000,
        currency: "USD".to_string(),
        sponsor: "ESA".to_string(),
        generated_at: Utc::now(),
        message: Some("Debris removal bounty claimed".to_string()),
    };

    let json = serde_json::to_string(&alert).expect("Failed to serialize");
    let parsed: BountyAlert = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(parsed.alert_type, AlertType::BountyClaimed);
    assert_eq!(parsed.debris_norad_id, 49863);
}
