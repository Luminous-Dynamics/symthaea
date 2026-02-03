//! Tests for conjunction analysis
//!
//! Tests close approach detection, collision probability calculation,
//! and risk assessment.

use chrono::Utc;
use orbital_mechanics::{
    conjunction::{ConjunctionAnalyzer, RiskLevel, screen_catalog},
    state::{OrbitalState, StateVector, DataSource},
    covariance::CovarianceMatrix,
};

/// Test basic conjunction detection between two objects
#[test]
fn test_conjunction_detection() {
    let now = Utc::now();

    // Create two objects 1 km apart
    let primary = OrbitalState::new(
        25544,  // ISS
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    let secondary = OrbitalState::new(
        99999,  // Debris
        now,
        StateVector::new(7001.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    assert!((assessment.miss_distance_km - 1.0).abs() < 0.01);
    assert!(assessment.collision_probability.has_covariance);
}

/// Test risk level assignment based on Pc
#[test]
fn test_risk_level_thresholds() {
    // Test all threshold boundaries
    assert_eq!(RiskLevel::from_pc(1e-8), RiskLevel::Negligible);
    assert_eq!(RiskLevel::from_pc(5e-7), RiskLevel::Low);
    assert_eq!(RiskLevel::from_pc(5e-5), RiskLevel::Medium);
    assert_eq!(RiskLevel::from_pc(5e-4), RiskLevel::High);
    assert_eq!(RiskLevel::from_pc(5e-3), RiskLevel::Emergency);
}

/// Test risk level boundary at 1e-7
#[test]
fn test_risk_level_low_boundary() {
    assert_eq!(RiskLevel::from_pc(1e-7), RiskLevel::Low);
    assert_eq!(RiskLevel::from_pc(9.9e-8), RiskLevel::Negligible);
}

/// Test risk level boundary at 1e-5
#[test]
fn test_risk_level_medium_boundary() {
    assert_eq!(RiskLevel::from_pc(1e-5), RiskLevel::Medium);
    assert_eq!(RiskLevel::from_pc(9.9e-6), RiskLevel::Low);
}

/// Test risk level boundary at 1e-4
#[test]
fn test_risk_level_high_boundary() {
    assert_eq!(RiskLevel::from_pc(1e-4), RiskLevel::High);
    assert_eq!(RiskLevel::from_pc(9.9e-5), RiskLevel::Medium);
}

/// Test risk level boundary at 1e-3
#[test]
fn test_risk_level_emergency_boundary() {
    assert_eq!(RiskLevel::from_pc(1e-3), RiskLevel::Emergency);
    assert_eq!(RiskLevel::from_pc(9.9e-4), RiskLevel::High);
}

/// Test risk level descriptions
#[test]
fn test_risk_level_descriptions() {
    assert!(!RiskLevel::Negligible.description().is_empty());
    assert!(!RiskLevel::Low.description().is_empty());
    assert!(!RiskLevel::Medium.description().is_empty());
    assert!(!RiskLevel::High.description().is_empty());
    assert!(!RiskLevel::Emergency.description().is_empty());
}

/// Test catalog screening with multiple objects
#[test]
fn test_catalog_screening() {
    let now = Utc::now();

    // Create a protected object
    let protected = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    // Create catalog objects at varying distances
    let catalog: Vec<OrbitalState> = (0..10)
        .map(|i| {
            OrbitalState::new(
                40000 + i,
                now,
                StateVector::new(7000.0 + (i as f64) * 0.5, 0.0, 0.0, 0.0, 7.5, 0.0),
                DataSource::SpaceTrack,
            )
        })
        .collect();

    // Screen for close approaches within 2 km
    let conjunctions = screen_catalog(&[protected], &catalog, 2.0);

    // Should find objects at 0.5, 1.0, 1.5 km (indices 1, 2, 3)
    assert!(conjunctions.len() >= 3);

    // Should be sorted by distance
    for window in conjunctions.windows(2) {
        assert!(window[0].2 <= window[1].2);
    }
}

/// Test collision probability without covariance
#[test]
fn test_pc_without_covariance() {
    let now = Utc::now();

    // Create objects without covariance
    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7001.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    // Should fall back to miss-distance-only method
    assert!(!assessment.collision_probability.has_covariance);
}

/// Test close approach with high relative velocity
#[test]
fn test_high_velocity_conjunction() {
    let now = Utc::now();

    // Create nearly head-on collision scenario
    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    // Retrograde orbit - opposite direction
    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7000.5, 0.0, 0.0, 0.0, -7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    // Relative velocity should be ~15 km/s (7.5 + 7.5)
    assert!(assessment.relative_velocity_kms > 14.0,
            "Relative velocity: {} km/s", assessment.relative_velocity_kms);
}

/// Test conjunction analyzer configuration
#[test]
fn test_analyzer_configuration() {
    let analyzer = ConjunctionAnalyzer::new()
        .with_hbr(50.0)  // 50m hard body radius
        .with_screening_threshold(10.0);  // 10km threshold

    let now = Utc::now();

    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7000.5, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let assessment = analyzer.assess(&primary, &secondary);

    // HBR should be what we configured
    assert_eq!(assessment.hard_body_radius_m, 50.0);
}

/// Test covariance matrix operations
#[test]
fn test_covariance_matrix() {
    let cov = CovarianceMatrix::diagonal([1.0, 2.0, 3.0, 0.001, 0.002, 0.003]);

    let pos_sigma = cov.position_sigma();
    let vel_sigma = cov.velocity_sigma();

    // Should have meaningful uncertainty values
    assert!(pos_sigma > 0.0);
    assert!(vel_sigma > 0.0);
}

/// Test assessment output fields
#[test]
fn test_assessment_fields() {
    let now = Utc::now();

    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    let secondary = OrbitalState::new(
        49863,
        now,
        StateVector::new(7000.2, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0);
    let assessment = analyzer.assess(&primary, &secondary);

    // Verify all required fields
    assert_eq!(assessment.primary_norad_id, 25544);
    assert_eq!(assessment.secondary_norad_id, 49863);
    assert!(assessment.miss_distance_km < 1.0);
    assert!(assessment.collision_probability.pc > 0.0);
    assert!(assessment.collision_probability.pc_lower <= assessment.collision_probability.pc);
    assert!(assessment.collision_probability.pc_upper >= assessment.collision_probability.pc);
    assert_eq!(assessment.hard_body_radius_m, 20.0);
}

/// Test very close approach
#[test]
fn test_very_close_approach() {
    let now = Utc::now();

    // Objects 100 meters apart
    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001]));

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7000.1, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001]));

    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0);
    let assessment = analyzer.assess(&primary, &secondary);

    // Should have high Pc and high/emergency risk
    assert!(assessment.miss_distance_km < 0.2);
    assert!(assessment.risk_level >= RiskLevel::High ||
            assessment.collision_probability.pc > 1e-4);
}

/// Test screening excludes self-conjunctions
#[test]
fn test_no_self_conjunction() {
    let now = Utc::now();

    let objects: Vec<OrbitalState> = (0..5)
        .map(|i| {
            OrbitalState::new(
                25544 + i, // Include ISS
                now,
                StateVector::new(7000.0 + (i as f64), 0.0, 0.0, 0.0, 7.5, 0.0),
                DataSource::SpaceTrack,
            )
        })
        .collect();

    // Screen all objects against all (including themselves)
    let conjunctions = screen_catalog(&objects, &objects, 2.0);

    // No object should be paired with itself
    for (primary_id, secondary_id, _) in &conjunctions {
        assert_ne!(primary_id, secondary_id, "Self-conjunction detected");
    }
}

/// Test miss distance precision
#[test]
fn test_miss_distance_precision() {
    let now = Utc::now();

    // Objects exactly 0.5 km apart in X
    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7000.5, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    assert!((assessment.miss_distance_km - 0.5).abs() < 0.001,
            "Expected 0.5 km, got {}", assessment.miss_distance_km);
}

/// Test relative velocity calculation
#[test]
fn test_relative_velocity_same_direction() {
    let now = Utc::now();

    // Objects moving in same direction at same speed
    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7000.5, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    );

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    // Relative velocity should be ~0
    assert!(assessment.relative_velocity_kms < 0.01,
            "Expected ~0 km/s, got {}", assessment.relative_velocity_kms);
}

/// Test Pc method selection
#[test]
fn test_pc_method_with_covariance() {
    let now = Utc::now();

    let primary = OrbitalState::new(
        25544,
        now,
        StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    let secondary = OrbitalState::new(
        99999,
        now,
        StateVector::new(7001.0, 0.0, 0.0, 0.0, 7.5, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&primary, &secondary);

    // Should use Alfano 2D method when covariance available
    use orbital_mechanics::conjunction::PcMethod;
    assert_eq!(assessment.collision_probability.method, PcMethod::Alfano2D);
}
