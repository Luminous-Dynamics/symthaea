// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the physics/ module.
//!
//! All tests are gated on the `physics` feature flag because the physics module
//! is conditionally compiled.
//!
//! Tests cover:
//! - PlasmaReading creation, normalization, and critical detection
//! - PlasmaHdcEncoder deterministic encoding from same seed
//! - PlasmaState to_readings conversion and has_critical detection
//! - CModSample feature vector extraction
//! - DisruptionLabel classification and numeric conversion
//! - StabilityRegime from_phi threshold classification
//! - SensorNormalizer normalization ranges
//! - Critical reading detection across sensor types
#![cfg(feature = "physics")]

use symthaea::physics::{
    CModSample, DisruptionLabel, PlasmaControlConfig, PlasmaHdcEncoder, PlasmaReading,
    PlasmaSensorType, PlasmaState, SensorNormalizer, StabilityRegime,
};

// ============================================================================
// PlasmaReading Tests
// ============================================================================

#[test]
fn test_plasma_reading_creation_and_normalization() {
    // PlasmaCurrent typical range is [0.5, 2.0]
    let reading = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.25, 0.0);

    assert_eq!(reading.sensor, PlasmaSensorType::PlasmaCurrent);
    assert!((reading.value - 1.25).abs() < f64::EPSILON);
    assert!((reading.timestamp - 0.0).abs() < f64::EPSILON);
    assert!(
        (reading.quality - 1.0).abs() < f64::EPSILON,
        "Default quality should be 1.0"
    );

    // Normalized: (1.25 - 0.5) / (2.0 - 0.5) = 0.75 / 1.5 = 0.5
    let norm = reading.normalized();
    assert!(
        (norm - 0.5).abs() < 0.01,
        "1.25 MA should normalize to ~0.5, got {}",
        norm
    );
}

#[test]
fn test_plasma_reading_normalization_clamping() {
    // Value below range should clamp to 0.0
    let below = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 0.0, 0.0);
    assert!(
        below.normalized() <= 0.01,
        "Value below range should clamp near 0.0, got {}",
        below.normalized()
    );

    // Value above range should clamp to 1.0
    let above = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 10.0, 0.0);
    assert!(
        above.normalized() >= 0.99,
        "Value above range should clamp near 1.0, got {}",
        above.normalized()
    );
}

#[test]
fn test_plasma_reading_critical_detection() {
    // PlasmaCurrent critical: < 0.3 or > 2.5
    let normal = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 0.0);
    assert!(!normal.is_critical(), "1.0 MA should not be critical");

    let low_critical = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 0.2, 0.0);
    assert!(
        low_critical.is_critical(),
        "0.2 MA should be critical (below 0.3)"
    );

    let high_critical = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 3.0, 0.0);
    assert!(
        high_critical.is_critical(),
        "3.0 MA should be critical (above 2.5)"
    );

    // SafetyFactor critical: < 2.0
    let q95_critical = PlasmaReading::new(PlasmaSensorType::SafetyFactor, 1.5, 0.0);
    assert!(
        q95_critical.is_critical(),
        "q95 = 1.5 should be critical (below 2.0)"
    );

    let q95_normal = PlasmaReading::new(PlasmaSensorType::SafetyFactor, 3.5, 0.0);
    assert!(
        !q95_normal.is_critical(),
        "q95 = 3.5 should not be critical"
    );

    // ToroidalField has no critical thresholds
    let bt = PlasmaReading::new(PlasmaSensorType::ToroidalField, 100.0, 0.0);
    assert!(
        !bt.is_critical(),
        "ToroidalField should never be critical (no thresholds)"
    );
}

// ============================================================================
// PlasmaHdcEncoder Tests
// ============================================================================

#[test]
fn test_encoder_deterministic_from_same_seed() {
    let encoder1 = PlasmaHdcEncoder::new(42);
    let encoder2 = PlasmaHdcEncoder::new(42);

    let reading = PlasmaReading::new(PlasmaSensorType::ElectronTemperature, 5.0, 1.0);

    let hv1 = encoder1.encode_sample(&reading);
    let hv2 = encoder2.encode_sample(&reading);

    assert_eq!(
        hv1, hv2,
        "Same seed should produce identical BinaryHV encodings"
    );
}

#[test]
fn test_encoder_different_seeds_produce_different_hvs() {
    let encoder_a = PlasmaHdcEncoder::new(1);
    let encoder_b = PlasmaHdcEncoder::new(2);

    let reading = PlasmaReading::new(PlasmaSensorType::ElectronDensity, 1.5, 0.0);

    let hv_a = encoder_a.encode_sample(&reading);
    let hv_b = encoder_b.encode_sample(&reading);

    let sim = hv_a.similarity(&hv_b);
    assert!(
        sim < 0.95,
        "Different seeds should generally produce different encodings, similarity: {}",
        sim
    );
}

#[test]
fn test_encoder_base_vectors_exist_for_all_sensors() {
    let encoder = PlasmaHdcEncoder::new(12345);

    for sensor in PlasmaSensorType::all() {
        let base = encoder.base_vector(sensor);
        let density = base.density();
        assert!(
            density > 0.3 && density < 0.7,
            "Base vector for {:?} should have reasonable density, got {}",
            sensor,
            density
        );
    }
}

#[test]
fn test_encoder_different_sensors_produce_different_hvs() {
    let encoder = PlasmaHdcEncoder::new(42);

    let ip_reading = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 0.0);
    let ne_reading = PlasmaReading::new(PlasmaSensorType::ElectronDensity, 1.0, 0.0);

    let hv_ip = encoder.encode_sample(&ip_reading);
    let hv_ne = encoder.encode_sample(&ne_reading);

    let sim = hv_ip.similarity(&hv_ne);
    assert!(
        sim < 0.7,
        "Different sensors should have low similarity, got {}",
        sim
    );
}

// ============================================================================
// PlasmaState Tests
// ============================================================================

#[test]
fn test_plasma_state_to_readings() {
    let state = PlasmaState::new(
        1.0, // timestamp
        1.0, // ip
        1.5, // ne
        5.0, // te
        1.0, // prad
        1.0, // vloop
        5.0, // bt
        0.5, // bp
        3.5, // q95
        2.0, // wmhd
        2.0, // beta
    );

    let readings = state.to_readings();
    assert_eq!(
        readings.len(),
        10,
        "PlasmaState should produce 10 sensor readings"
    );

    // Verify a few specific readings
    let ip_reading = readings
        .iter()
        .find(|r| r.sensor == PlasmaSensorType::PlasmaCurrent);
    assert!(ip_reading.is_some());
    assert!((ip_reading.unwrap().value - 1.0).abs() < f64::EPSILON);

    let te_reading = readings
        .iter()
        .find(|r| r.sensor == PlasmaSensorType::ElectronTemperature);
    assert!(te_reading.is_some());
    assert!((te_reading.unwrap().value - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_plasma_state_has_critical() {
    // A genuinely stable state (all values within normal operating range)
    let stable = PlasmaState::new(
        0.0, 1.0, // ip: within [0.5, 2.0], not critical (thresholds: <0.3 or >2.5)
        1.5, // ne: within [0.5, 3.0], not critical (threshold: >4.0)
        5.0, // te: within [1.0, 10.0], not critical (threshold: <0.5)
        0.5, // prad: within [0.1, 5.0], not critical (threshold: >0.8)
        1.0, // vloop: within [0.5, 3.0], not critical (threshold: >5.0)
        5.0, // bt: no critical thresholds
        0.5, // bp: no critical thresholds
        3.5, // q95: within [2.0, 5.0], not critical (threshold: <2.0)
        2.0, // wmhd: no critical thresholds
        2.0, // beta: within [0.5, 5.0], not critical (threshold: >3.5)
    );
    assert!(
        !stable.has_critical(),
        "Stable PlasmaState should not have critical readings"
    );

    // Disruption-like state with several critical values
    let critical = PlasmaState::new(
        0.0, 0.2, // ip < 0.3 -> critical
        4.5, // ne > 4.0 -> critical
        0.3, // te < 0.5 -> critical
        0.9, // prad > 0.8 -> critical
        6.0, // vloop > 5.0 -> critical
        5.0, // bt (no thresholds)
        0.5, // bp (no thresholds)
        1.5, // q95 < 2.0 -> critical
        0.1, // wmhd (no thresholds)
        4.0, // beta > 3.5 -> critical
    );
    assert!(
        critical.has_critical(),
        "Disruption state should have critical readings"
    );

    let critical_sensors = critical.critical_sensors();
    assert!(
        critical_sensors.contains(&PlasmaSensorType::PlasmaCurrent),
        "PlasmaCurrent should be critical"
    );
    assert!(
        critical_sensors.contains(&PlasmaSensorType::SafetyFactor),
        "SafetyFactor should be critical"
    );
    assert!(
        critical_sensors.contains(&PlasmaSensorType::ElectronTemperature),
        "ElectronTemperature should be critical"
    );
}

// ============================================================================
// CModSample Tests
// ============================================================================

#[test]
fn test_cmod_sample_creation_and_feature_vector() {
    let sample = CModSample::new(12345, 500.0);

    assert_eq!(sample.shot_id, 12345);
    assert!((sample.time_ms - 500.0).abs() < f64::EPSILON);
    assert!(
        sample.has_missing_data(),
        "New CModSample should have NaN (missing) sensor values"
    );

    // Feature vector should contain NaN values for a fresh sample
    let features = sample.to_feature_vector();
    assert_eq!(
        features.len(),
        8,
        "Feature vector should have 8 sensor values"
    );
    assert!(
        features.iter().all(|v| v.is_nan()),
        "All features should be NaN for a default sample"
    );
}

#[test]
fn test_cmod_sample_with_data() {
    let mut sample = CModSample::new(100, 250.0);
    sample.ip = 0.8;
    sample.ne = 2.0;
    sample.te = 3.0;
    sample.prad = 1.5;
    sample.vloop = 1.2;
    sample.q95 = 4.0;
    sample.wmhd = 0.15;
    sample.beta = 1.5;

    assert!(
        !sample.has_missing_data(),
        "Sample with all values set should not have missing data"
    );

    let features = sample.to_feature_vector();
    assert_eq!(features.len(), 8);
    assert!((features[0] - 0.8).abs() < f32::EPSILON, "ip should be 0.8");
    assert!((features[1] - 2.0).abs() < f32::EPSILON, "ne should be 2.0");
}

// ============================================================================
// DisruptionLabel Tests
// ============================================================================

#[test]
fn test_disruption_label_classification() {
    assert!((DisruptionLabel::Normal.to_numeric() - 0.0).abs() < f32::EPSILON);
    assert!((DisruptionLabel::Warning.to_numeric() - 0.5).abs() < f32::EPSILON);
    assert!((DisruptionLabel::Critical.to_numeric() - 1.0).abs() < f32::EPSILON);
    assert!((DisruptionLabel::PostDisruption.to_numeric() - (-1.0)).abs() < f32::EPSILON);

    // Color mapping
    assert_eq!(DisruptionLabel::Normal.color(), "green");
    assert_eq!(DisruptionLabel::Warning.color(), "yellow");
    assert_eq!(DisruptionLabel::Critical.color(), "red");
    assert_eq!(DisruptionLabel::PostDisruption.color(), "gray");
}

// ============================================================================
// StabilityRegime Tests
// ============================================================================

#[test]
fn test_stability_regime_from_phi_thresholds() {
    let config = PlasmaControlConfig::default();
    // Default thresholds: stable >= 0.6, warning >= 0.4, critical >= 0.25, else emergency

    assert!(matches!(
        StabilityRegime::from_phi(0.8, &config),
        StabilityRegime::Stable
    ));
    assert!(matches!(
        StabilityRegime::from_phi(0.5, &config),
        StabilityRegime::Warning
    ));
    assert!(matches!(
        StabilityRegime::from_phi(0.3, &config),
        StabilityRegime::Critical
    ));
    assert!(matches!(
        StabilityRegime::from_phi(0.1, &config),
        StabilityRegime::Emergency
    ));

    // Boundary values
    assert!(matches!(
        StabilityRegime::from_phi(0.6, &config),
        StabilityRegime::Stable
    ));
    assert!(matches!(
        StabilityRegime::from_phi(0.4, &config),
        StabilityRegime::Warning
    ));
    assert!(matches!(
        StabilityRegime::from_phi(0.25, &config),
        StabilityRegime::Critical
    ));
}

#[test]
fn test_stability_regime_urgency() {
    assert!((StabilityRegime::Stable.urgency() - 0.0).abs() < f32::EPSILON);
    assert!((StabilityRegime::Warning.urgency() - 0.3).abs() < f32::EPSILON);
    assert!((StabilityRegime::Critical.urgency() - 0.7).abs() < f32::EPSILON);
    assert!((StabilityRegime::Emergency.urgency() - 1.0).abs() < f32::EPSILON);
}

// ============================================================================
// SensorNormalizer Tests
// ============================================================================

#[test]
fn test_sensor_normalizer_ranges() {
    let normalizer = SensorNormalizer::default();

    // Normalize Ip: default range [0.0, 1.5]
    let norm_ip = normalizer.normalize_ip(0.75);
    assert!(
        (norm_ip - 0.5).abs() < 0.01,
        "0.75 MA in [0.0, 1.5] should normalize to 0.5, got {}",
        norm_ip
    );

    // Normalize ne: default range [0.0, 5.0]
    let norm_ne = normalizer.normalize_ne(2.5);
    assert!(
        (norm_ne - 0.5).abs() < 0.01,
        "2.5 in [0.0, 5.0] should normalize to 0.5, got {}",
        norm_ne
    );

    // Values at boundaries
    let at_min = normalizer.normalize_ip(0.0);
    assert!(
        at_min.abs() < 0.01,
        "Value at range minimum should normalize to ~0.0, got {}",
        at_min
    );

    let at_max = normalizer.normalize_ip(1.5);
    assert!(
        (at_max - 1.0).abs() < 0.01,
        "Value at range maximum should normalize to ~1.0, got {}",
        at_max
    );

    // Values outside range should clamp
    let above_max = normalizer.normalize_ip(10.0);
    assert!(
        (above_max - 1.0).abs() < 0.01,
        "Value above range should clamp to 1.0, got {}",
        above_max
    );

    let below_min = normalizer.normalize_ip(-1.0);
    assert!(
        below_min.abs() < 0.01,
        "Value below range should clamp to 0.0, got {}",
        below_min
    );
}

#[test]
fn test_sensor_normalizer_nan_handling() {
    let normalizer = SensorNormalizer::default();

    // NaN should return 0.5 (middle default)
    let nan_result = normalizer.normalize_ip(f32::NAN);
    assert!(
        (nan_result - 0.5).abs() < 0.01,
        "NaN input should normalize to 0.5, got {}",
        nan_result
    );
}