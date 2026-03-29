// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_phi_oracle::{
    CoherenceTrend, CovarianceEncoder, IntegrationOracle, OracleConfig, TimeSeriesEncoder,
};

/// End-to-end: encode random observations -> accumulate -> measure -> get report.
#[test]
fn test_end_to_end_random_system() {
    let encoder = TimeSeriesEncoder::new(8, 256, 42).with_name("test-random");

    let config = OracleConfig {
        window_size: 30,
        temporal_probes: vec![0.1, 1.0, 10.0],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();

    // Feed enough observations
    let mut rng: u64 = 7777;
    for _ in 0..60 {
        let obs: Vec<f64> = (0..8)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
            })
            .collect();
        oracle.observe(&obs).unwrap();
    }

    assert!(oracle.ready(), "should be ready after 60 observations");
    let report = oracle.measure().expect("should produce a report");

    assert!(
        report.integration_index >= 0.0,
        "integration index must be non-negative"
    );
    assert!(report.total_mutual_information >= 0.0);
    assert_eq!(report.num_observations, 60);

    // MIP should partition the 8 variables into two non-empty groups
    let (a, b) = &report.minimum_information_partition;
    assert!(
        !a.is_empty() && !b.is_empty(),
        "MIP should produce non-trivial partition"
    );
    assert_eq!(a.len() + b.len(), 8, "partition should cover all variables");

    // Spectral order should include all 8 variables
    assert_eq!(report.spectral_order.len(), 8);

    // Temporal coherence should be present (we configured temporal probes)
    let tc = report
        .temporal_coherence
        .as_ref()
        .expect("temporal coherence should exist");
    assert_eq!(tc.cv_by_tau.len(), 3);
    assert!(tc.dominant_timescale > 0.0);
}

/// Correlated system should have higher integration than uncorrelated.
#[test]
fn test_correlated_vs_uncorrelated() {
    fn measure_integration(correlated: bool, seed: u64) -> f64 {
        let encoder = TimeSeriesEncoder::new(4, 128, seed);
        let config = OracleConfig {
            window_size: 40,
            temporal_probes: vec![], // skip temporal for speed
            seed,
            ..Default::default()
        };

        let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();

        let mut rng = seed + 100;
        for t in 0..80 {
            let base = if correlated {
                // All variables share a common signal
                let common = (t as f64 * 0.1).sin();
                vec![common; 4]
            } else {
                vec![0.0; 4]
            };

            let obs: Vec<f64> = base
                .iter()
                .map(|&b| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
                    b + noise * 0.3
                })
                .collect();
            oracle.observe(&obs).unwrap();
        }

        oracle.measure().map(|r| r.integration_index).unwrap_or(0.0)
    }

    let correlated_phi = measure_integration(true, 42);
    let uncorrelated_phi = measure_integration(false, 99);

    // Correlated system should have strictly higher integration
    assert!(
        correlated_phi > uncorrelated_phi,
        "correlated ({correlated_phi:.6}) should exceed uncorrelated ({uncorrelated_phi:.6})"
    );
}

/// Covariance bypass: feed a pre-built covariance matrix and get a result.
#[test]
fn test_covariance_bypass() {
    let encoder = CovarianceEncoder::new(3);
    let mut oracle = IntegrationOracle::new(Box::new(encoder), OracleConfig::default()).unwrap();

    // 3x3 covariance matrix (row-major) with known structure:
    // Variables 0,1 are correlated; variable 2 is independent.
    #[rustfmt::skip]
    let cov = [
        1.0, 0.8, 0.0,
        0.8, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    oracle.observe_covariance(&cov, 3, 100).unwrap();
    assert!(oracle.ready());

    let report = oracle
        .measure()
        .expect("should produce report from covariance");
    assert!(report.integration_index >= 0.0);
    assert_eq!(report.num_observations, 100);

    // MIP should produce a non-trivial partition of the 3 variables.
    let (a, b) = &report.minimum_information_partition;
    assert!(
        !a.is_empty() && !b.is_empty(),
        "MIP should produce non-trivial partition"
    );
    assert_eq!(
        a.len() + b.len(),
        3,
        "partition should cover all 3 variables"
    );

    // Integration index should be positive — the system has structure
    // (variables 0 and 1 are correlated at r=0.8).
    assert!(
        report.integration_index > 0.0,
        "system with correlation should have positive integration index, got {:.6}",
        report.integration_index
    );
}

/// Temporal coherence: periodic signal should have a dominant timescale.
#[test]
fn test_temporal_periodic_signal() {
    let encoder = TimeSeriesEncoder::new(4, 64, 42);
    let config = OracleConfig {
        window_size: 40,
        temporal_probes: vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();

    // Feed a signal with period ~1.0s (sampled at dt=0.02s)
    for step in 0..100 {
        let t = step as f64 * 0.02;
        let obs: Vec<f64> = (0..4)
            .map(|i| (t * 2.0 * std::f64::consts::PI + i as f64 * 0.5).sin())
            .collect();
        oracle.observe(&obs).unwrap();
    }

    let report = oracle.measure().expect("should produce report");
    let tc = report
        .temporal_coherence
        .as_ref()
        .expect("temporal coherence should exist");

    // Should have CV values for all 6 probe timescales
    assert_eq!(tc.cv_by_tau.len(), 6);

    // Dominant timescale should be finite and positive
    assert!(tc.dominant_timescale > 0.0 && tc.dominant_timescale.is_finite());
}

/// Reset should clear all state.
#[test]
fn test_reset_clears_everything() {
    let encoder = TimeSeriesEncoder::new(4, 64, 42);
    let mut oracle = IntegrationOracle::with_defaults(Box::new(encoder)).unwrap();

    for i in 0..60 {
        let obs = vec![(i as f64 * 0.1).sin(); 4];
        oracle.observe(&obs).unwrap();
    }
    assert!(oracle.ready());

    oracle.reset();
    assert!(!oracle.ready());
    assert_eq!(oracle.num_observations(), 0);
    assert!(oracle.measure().is_none());
}

/// Covariance encoder via observe_covariance path should match direct
/// covariance input (smoke test that the bypass works).
#[test]
fn test_covariance_deterministic() {
    let encoder = CovarianceEncoder::new(4);
    let mut oracle = IntegrationOracle::new(Box::new(encoder), OracleConfig::default()).unwrap();

    // Identity covariance (all variables independent)
    #[rustfmt::skip]
    let cov_identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    oracle.observe_covariance(&cov_identity, 4, 200).unwrap();
    let report1 = oracle.measure().expect("should produce report");

    // Measure again — should be deterministic
    let report2 = oracle.measure().expect("should produce report again");
    assert!(
        (report1.integration_index - report2.integration_index).abs() < 1e-10,
        "repeated measurement should be deterministic"
    );

    // Integration of independent variables should be low
    // (identity covariance = zero mutual information between variables)
    assert!(
        report1.integration_index < 0.01,
        "independent variables should have near-zero integration, got {:.6}",
        report1.integration_index
    );
}

/// Regression test: the signal-loss bug is fixed.
///
/// In the old pipeline, `observe()` encoded N variables into 256-dim HVs and
/// pushed them into `SpectralMIPFinder`, which computed covariance across HV
/// dimensions (random projections), NOT original variables. Result: integration
/// index ~0.000004 for a clearly coupled 6-node system.
///
/// After the fix, covariance is computed on the original 6 variables, giving
/// meaningful integration.
#[test]
fn test_signal_loss_fixed() {
    let num_nodes = 6;
    let encoder = TimeSeriesEncoder::new(num_nodes, 256, 42).with_name("signal-loss-regression");

    let config = OracleConfig {
        window_size: 50,
        temporal_probes: vec![],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();

    // Simulate a coupled 6-node system (same as power_grid example)
    let coupling = [
        [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.3, 0.0, 0.6, 0.0],
        [0.0, 0.3, 0.0, 0.8, 0.0, 0.6],
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
    ];

    let dt = 0.02;
    let mut state = vec![0.0f64; num_nodes];
    let mut rng_state: u64 = 12345;

    for step in 0..200 {
        let mut forces = vec![0.0f64; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                forces[i] += coupling[i][j] * (state[j] - state[i]);
            }
            forces[i] -= 0.1 * state[i];
            if i == 0 || i == 3 {
                forces[i] += 0.5 * (step as f64 * dt * 2.0 * std::f64::consts::PI * 0.1).sin();
            }
        }

        for i in 0..num_nodes {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02;
            state[i] += forces[i] * dt + noise;
        }

        let observation: Vec<f64> = state.iter().map(|&dev| 60.0 + dev).collect();
        oracle.observe(&observation).unwrap();
    }

    let report = oracle.measure().expect("should produce report");

    // The old buggy pipeline gave ~0.000004. The fix should give >> 0.01.
    assert!(
        report.integration_index > 0.01,
        "signal-loss regression: integration index should be >> 0.01 for a coupled \
         6-node system, got {:.6} (old bug gave ~0.000004)",
        report.integration_index
    );
}

/// Trend tracking across multiple measurement windows.
#[test]
fn test_trend_tracking() {
    let encoder = TimeSeriesEncoder::new(4, 64, 42);
    let config = OracleConfig {
        window_size: 20,
        temporal_probes: vec![],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();
    let mut trend = CoherenceTrend::new(10);

    // First window: weakly correlated
    for t in 0..30 {
        let mut rng = 42u64.wrapping_add(t as u64 * 7919);
        let obs: Vec<f64> = (0..4)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
            })
            .collect();
        oracle.observe(&obs).unwrap();
    }
    if let Some(report) = oracle.measure() {
        trend.record(&report);
    }

    // Reset and second window: strongly correlated
    oracle.reset();
    for t in 0..30 {
        let common = (t as f64 * 0.2).sin();
        let obs = vec![common, common * 0.9, common * 0.8, common * 0.7];
        oracle.observe(&obs).unwrap();
    }
    if let Some(report) = oracle.measure() {
        trend.record(&report);
    }

    assert_eq!(trend.len(), 2);
    assert!(
        trend.trend_slope().is_some(),
        "should have slope with 2 points"
    );

    // Display should work
    let display = format!("{trend}");
    assert!(!display.is_empty());
}

/// Display impl produces readable output.
#[test]
fn test_display_integration_report() {
    let encoder = TimeSeriesEncoder::new(4, 64, 42);
    let config = OracleConfig {
        window_size: 20,
        temporal_probes: vec![0.1, 1.0],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();
    for t in 0..40 {
        let common = (t as f64 * 0.1).sin();
        let obs = vec![common, common * 0.8, common * -0.5, (t as f64 * 0.3).cos()];
        oracle.observe(&obs).unwrap();
    }

    let report = oracle.measure().expect("should produce report");
    let display = format!("{report}");

    assert!(display.contains("Integration Report"), "display: {display}");
    assert!(display.contains("Integration index:"), "display: {display}");
    assert!(display.contains("Normalized index:"), "display: {display}");
    assert!(display.contains("Part A:"), "display: {display}");
    assert!(display.contains("Part B:"), "display: {display}");
    assert!(display.contains("Temporal Coherence"), "display: {display}");
    assert!(
        display.contains("Variable contributions"),
        "display: {display}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Round 2 tests
// ═════════════════════════════════════════════════════════════════════════════

/// Normalized index should always be in [0, 1].
#[test]
fn test_normalized_index_bounds() {
    let mut oracle = IntegrationOracle::new_simple(
        4,
        OracleConfig {
            window_size: 20,
            temporal_probes: vec![],
            ..Default::default()
        },
    )
    .unwrap();

    for t in 0..40 {
        let common = (t as f64 * 0.1).sin();
        oracle
            .observe(&[common, common * 0.8, common * -0.5, (t as f64 * 0.3).cos()])
            .unwrap();
    }

    let report = oracle.measure().expect("should produce report");
    assert!(
        report.normalized_index >= 0.0 && report.normalized_index <= 1.0,
        "normalized_index out of bounds: {}",
        report.normalized_index
    );
}

/// new_simple constructor should work without an encoder.
#[test]
fn test_simple_constructor() {
    let mut oracle = IntegrationOracle::new_simple(6, OracleConfig::default()).unwrap();
    for t in 0..60 {
        let obs: Vec<f64> = (0..6).map(|i| (t as f64 * 0.1 + i as f64).sin()).collect();
        oracle.observe(&obs).unwrap();
    }
    let report = oracle.measure().expect("should produce report");
    assert!(report.integration_index >= 0.0);
    assert_eq!(report.variable_contributions.len(), 6);
}

/// new_simple rejects fewer than 2 variables.
#[test]
fn test_simple_constructor_too_few_vars() {
    let result = IntegrationOracle::new_simple(1, OracleConfig::default());
    assert!(result.is_err());
}

/// Variable contributions should have correct length and an independent
/// variable should contribute ~0.
#[test]
fn test_variable_contributions() {
    let encoder = CovarianceEncoder::new(4);
    let mut oracle = IntegrationOracle::new(Box::new(encoder), OracleConfig::default()).unwrap();

    // Variables 0,1,2 correlated; variable 3 independent.
    #[rustfmt::skip]
    let cov = [
        1.0, 0.7, 0.6, 0.0,
        0.7, 1.0, 0.5, 0.0,
        0.6, 0.5, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    oracle.observe_covariance(&cov, 4, 100).unwrap();

    let report = oracle.measure().expect("should produce report");
    assert_eq!(report.variable_contributions.len(), 4);

    // Variable 3 (independent) should have low contribution relative to correlated vars.
    // Note: contribution uses total_mi (MIP-based) minus pairwise MI of submatrix,
    // so small residual is expected from the different computation methods.
    let indep_contrib = report.variable_contributions[3];
    assert!(
        indep_contrib.abs() < 0.2,
        "independent variable should have low contribution, got {indep_contrib:.6}"
    );

    // Correlated variables should contribute positively
    assert!(
        report.variable_contributions[0] > 0.0,
        "correlated variable should contribute positively"
    );
}

/// Hierarchical measurement should produce multi-scale phi values.
#[test]
fn test_hierarchical_multi_scale() {
    let encoder = TimeSeriesEncoder::new(8, 128, 42);
    let config = OracleConfig {
        window_size: 30,
        temporal_probes: vec![],
        ..Default::default()
    };

    let mut oracle = IntegrationOracle::new(Box::new(encoder), config).unwrap();
    for t in 0..60 {
        let obs: Vec<f64> = (0..8)
            .map(|i| (t as f64 * 0.1 + i as f64 * 0.3).sin())
            .collect();
        oracle.observe(&obs).unwrap();
    }

    let hier = oracle
        .measure_hierarchical()
        .expect("should produce hierarchical report");

    // Should have at least 2 scales: 8 vars and 4 vars
    assert!(
        hier.scales.len() >= 2,
        "should have >= 2 scales, got {:?}",
        hier.scales
    );
    assert_eq!(hier.scales[0], 8);
    assert_eq!(hier.phi_by_scale.len(), hier.scales.len());

    // All phi values should be non-negative
    for &phi in &hier.phi_by_scale {
        assert!(phi >= 0.0, "phi should be non-negative, got {phi}");
    }

    // Display should work
    let display = format!("{hier}");
    assert!(display.contains("Hierarchical"), "display: {display}");
}
