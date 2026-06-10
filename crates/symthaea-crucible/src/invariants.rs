// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal invariants that must hold for ALL inputs to the moral systems.
//!
//! These are the non-negotiable mathematical properties. If any of these fail,
//! the moral system has a bug — regardless of scenario, regardless of substrate.

use crate::telemetry::CycleTelemetry;

/// Check all universal invariants against a single cycle's telemetry.
/// Returns a list of violation descriptions (empty = all passed).
pub fn check_universal(telemetry: &CycleTelemetry) -> Vec<String> {
    let mut violations = Vec::new();

    // 1. No NaN/Inf in any numeric output
    if !telemetry.moral_score.is_finite() {
        violations.push(format!(
            "cycle {}: moral_score is not finite ({})",
            telemetry.cycle, telemetry.moral_score
        ));
    }
    if !telemetry.moral_free_energy.is_finite() {
        violations.push(format!(
            "cycle {}: moral_free_energy is not finite ({})",
            telemetry.cycle, telemetry.moral_free_energy
        ));
    }
    if !telemetry.anomaly_score.is_finite() {
        violations.push(format!(
            "cycle {}: anomaly_score is not finite ({})",
            telemetry.cycle, telemetry.anomaly_score
        ));
    }
    for (i, &coord) in telemetry.harmony_coordinates.iter().enumerate() {
        if !coord.is_finite() {
            violations.push(format!(
                "cycle {}: harmony_coordinate[{i}] is not finite ({coord})",
                telemetry.cycle
            ));
        }
    }

    // 2. Moral score bounded [-1, 1]
    if telemetry.moral_score < -1.0 || telemetry.moral_score > 1.0 {
        violations.push(format!(
            "cycle {}: moral_score out of bounds ({})",
            telemetry.cycle, telemetry.moral_score
        ));
    }

    // 3. Anomaly score bounded [0, 1]
    if telemetry.anomaly_score < 0.0 || telemetry.anomaly_score > 1.0 {
        violations.push(format!(
            "cycle {}: anomaly_score out of bounds ({})",
            telemetry.cycle, telemetry.anomaly_score
        ));
    }

    // 4. Moral free energy non-negative (it's KL + entropy, both >= 0)
    if telemetry.moral_free_energy < -1e-9 {
        violations.push(format!(
            "cycle {}: moral_free_energy negative ({})",
            telemetry.cycle, telemetry.moral_free_energy
        ));
    }

    violations
}

/// Check trajectory-level invariants across a sequence of telemetry points.
/// Returns violation descriptions.
pub fn check_trajectory(telemetry: &[CycleTelemetry]) -> Vec<String> {
    let mut violations = Vec::new();

    if telemetry.is_empty() {
        return violations;
    }

    // 1. Harmony coordinates should not all be identical across the entire run
    //    (would indicate the system is stuck / not responding to input)
    if telemetry.len() >= 10 {
        let first = &telemetry[0].harmony_coordinates;
        let all_identical = telemetry.iter().all(|t| {
            t.harmony_coordinates
                .iter()
                .zip(first)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        });
        if all_identical {
            violations.push(
                "trajectory: all harmony coordinates identical — system may not be responding to input"
                    .to_string(),
            );
        }
    }

    // 2. Free energy should not monotonically increase for >50 consecutive cycles
    //    (would indicate unbounded moral surprise accumulation)
    let fe_values: Vec<f64> = telemetry.iter().map(|t| t.moral_free_energy).collect();
    let mut monotone_increase_run = 0usize;
    for window in fe_values.windows(2) {
        if window[1] > window[0] + 1e-9 {
            monotone_increase_run += 1;
            if monotone_increase_run > 50 {
                violations.push(format!(
                    "trajectory: free energy monotonically increasing for {} consecutive observations",
                    monotone_increase_run
                ));
                break;
            }
        } else {
            monotone_increase_run = 0;
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::AnomalyFlags;

    fn make_telemetry(cycle: u64, moral_score: f64, fe: f64) -> CycleTelemetry {
        CycleTelemetry {
            cycle,
            moral_score,
            moral_verdict: "Neutral".to_string(),
            consent_violation: false,
            harmony_coordinates: [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05],
            moral_free_energy: fe,
            anomaly_score: 0.0,
            anomaly_flags: AnomalyFlags::default(),
        }
    }

    #[test]
    fn test_valid_telemetry_passes() {
        let t = make_telemetry(7, 0.5, 0.3);
        assert!(check_universal(&t).is_empty());
    }

    #[test]
    fn test_nan_moral_score_detected() {
        let t = make_telemetry(7, f64::NAN, 0.3);
        let violations = check_universal(&t);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("moral_score"));
    }

    #[test]
    fn test_out_of_bounds_moral_score() {
        let t = make_telemetry(7, 1.5, 0.3);
        let violations = check_universal(&t);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("out of bounds"));
    }

    #[test]
    fn test_negative_free_energy_detected() {
        let t = make_telemetry(7, 0.5, -1.0);
        let violations = check_universal(&t);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("negative"));
    }

    #[test]
    fn test_trajectory_static_detected() {
        let telemetry: Vec<_> = (0..20).map(|i| make_telemetry(i, 0.5, 0.3)).collect();
        let violations = check_trajectory(&telemetry);
        assert!(
            violations.iter().any(|v| v.contains("identical")),
            "Should detect static trajectory: {violations:?}"
        );
    }

    #[test]
    fn test_trajectory_varying_passes() {
        let telemetry: Vec<_> = (0..20)
            .map(|i| {
                let mut t = make_telemetry(i, 0.5, 0.3);
                t.harmony_coordinates[0] = i as f64 * 0.01;
                t
            })
            .collect();
        let violations = check_trajectory(&telemetry);
        let static_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.contains("identical"))
            .collect();
        assert!(static_violations.is_empty());
    }
}
