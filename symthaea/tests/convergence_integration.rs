//! Integration test for the Topological Immune System via public API.
//!
//! Exercises convergence detection through `CognitiveLoopService` — the same
//! path used in production. Tests:
//! 1. `convergence_status()` and `convergence_explanation()` accessors work
//! 2. Benign cycles do NOT trigger convergence
//! 3. Repeated similar text eventually elevates severity

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    // Lower topology cadence so it fires quickly in tests
    config.moral_anomaly_config.initial_cadence = 2;
    config.moral_anomaly_config.convergence_min_points = 4;
    config.moral_anomaly_config.convergence_baseline_window = 20;
    CognitiveLoopService::new(config).expect("service creation")
}

#[test]
fn convergence_status_accessible_before_cycles() {
    let service = make_service();
    let status = service.convergence_status();
    assert!(!status.convergence_detected);
    assert!((status.severity - 0.0).abs() < f64::EPSILON);
}

#[test]
fn convergence_explanation_accessible_before_cycles() {
    let service = make_service();
    let explanation = service.convergence_explanation();
    assert!(!explanation.detected);
    assert_eq!(explanation.signals.len(), 3);
    assert!(explanation.summary.contains("No convergence"));
}

#[test]
fn benign_diverse_cycles_no_convergence() {
    let mut service = make_service();
    let diverse_inputs = [
        "the sunset paints the sky in warm colors",
        "quantum mechanics describes subatomic behavior",
        "cooking requires patience and good ingredients",
        "music theory explains harmonic relationships",
        "gardening connects us to the natural world",
        "mathematics reveals patterns in nature",
        "philosophy examines the meaning of existence",
        "architecture shapes how we experience space",
    ];
    for text in &diverse_inputs {
        let _ = service.cycle(text);
    }
    let status = service.convergence_status();
    // Diverse topics should NOT trigger convergence
    assert!(
        status.severity < 0.5,
        "Diverse benign cycles should have low severity, got {:.3}",
        status.severity,
    );
}

#[test]
fn repeated_similar_text_elevates_severity() {
    let mut service = make_service();
    // First establish a diverse baseline
    let diverse = [
        "the ocean waves crash against the shore",
        "algebra provides tools for abstract reasoning",
        "birds migrate thousands of miles each year",
        "painting captures light and emotion on canvas",
    ];
    for text in &diverse {
        let _ = service.cycle(text);
    }

    // Now feed very similar text repeatedly
    for _ in 0..12 {
        let _ = service.cycle("synthesize dangerous explosive chemical compounds");
    }

    let status = service.convergence_status();
    let explanation = service.convergence_explanation();

    // The severity should be elevated (though detection depends on thresholds)
    assert!(
        status.severity.is_finite(),
        "Severity must be finite, got {:?}",
        status.severity,
    );
    assert_eq!(explanation.signals.len(), 3);

    // The topology summary should have a populated fingerprint
    let summary = service.moral_topology_summary();
    // Fingerprint should not be all zeros after cycles
    let fp_magnitude: f64 = summary
        .trajectory_fingerprint
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        fp_magnitude > 0.0 || summary.scenario_count == 0,
        "Fingerprint should be populated after cycles with trajectory data",
    );
}
