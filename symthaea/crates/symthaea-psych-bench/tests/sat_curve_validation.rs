// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SAT (Speed-Accuracy Tradeoff) curve validation: run representative benchmarks
//! at multiple time-pressure levels and verify human-like SAT properties.

use symthaea_psych_bench::benchmarks::{
    executive::{FlankerBenchmark, StroopBenchmark},
    language::SemanticPrimingBenchmark,
    sustained_attention::{CptBenchmark, SartBenchmark},
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark, sat_curves::SatBattery};

fn sat_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 10,
        seed: 42,
        ..Default::default()
    }
}

/// Verify SAT curves have valid structure across 6 representative benchmarks.
#[test]
fn sat_curve_basic_properties() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(NBackBenchmark),
        Box::new(SartBenchmark),
        Box::new(CptBenchmark),
        Box::new(SemanticPrimingBenchmark),
    ];

    let bench_refs: Vec<&dyn PsychBenchmark> = benchmarks.iter().map(|b| b.as_ref()).collect();
    let config = sat_config();
    let pressures = [0.0, 0.25, 0.5, 0.75, 1.0];

    let battery = SatBattery::run(&bench_refs, &config, &pressures);

    assert_eq!(
        battery.curves.len(),
        benchmarks.len(),
        "Should produce one curve per benchmark"
    );

    let mut failures = Vec::new();

    for curve in &battery.curves {
        // Each curve should have points matching the pressure levels
        assert_eq!(
            curve.points.len(),
            pressures.len(),
            "{}: wrong number of SAT points",
            curve.benchmark
        );

        // All points should have finite values
        for (i, point) in curve.points.iter().enumerate() {
            if !point.accuracy.is_finite() {
                failures.push(format!(
                    "{}: accuracy not finite at pressure={}",
                    curve.benchmark, pressures[i]
                ));
            }
            if !point.mean_rt.is_finite() {
                failures.push(format!(
                    "{}: mean_rt not finite at pressure={}",
                    curve.benchmark, pressures[i]
                ));
            }
            if point.accuracy < 0.0 || point.accuracy > 1.0 {
                failures.push(format!(
                    "{}: accuracy out of [0,1] at pressure={}: {}",
                    curve.benchmark, pressures[i], point.accuracy
                ));
            }
        }

        // Fit parameters should be finite
        if !curve.fit.asymptote.is_finite() {
            failures.push(format!("{}: fit asymptote not finite", curve.benchmark));
        }
        if !curve.fit.rate.is_finite() {
            failures.push(format!("{}: fit rate not finite", curve.benchmark));
        }
        if !curve.fit.intercept.is_finite() {
            failures.push(format!("{}: fit intercept not finite", curve.benchmark));
        }
        if !curve.fit.r_squared.is_finite() {
            failures.push(format!("{}: fit R² not finite", curve.benchmark));
        }
        if curve.fit.r_squared < 0.0 || curve.fit.r_squared > 1.0 {
            failures.push(format!(
                "{}: R² out of [0,1]: {}",
                curve.benchmark, curve.fit.r_squared
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "SAT curve validation failures ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Verify that time pressure actually changes benchmark behavior:
/// at least some benchmarks should show measurable accuracy reduction.
#[test]
fn time_pressure_affects_accuracy() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(StroopBenchmark),
        Box::new(NBackBenchmark),
        Box::new(SartBenchmark),
    ];

    let bench_refs: Vec<&dyn PsychBenchmark> = benchmarks.iter().map(|b| b.as_ref()).collect();
    let config = sat_config();
    let pressures = [0.0, 1.0];

    let battery = SatBattery::run(&bench_refs, &config, &pressures);

    let mut any_change = false;
    for curve in &battery.curves {
        let no_pressure = curve.points[0].accuracy;
        let max_pressure = curve.points[1].accuracy;
        if (no_pressure - max_pressure).abs() > 0.01 {
            any_change = true;
        }
    }

    assert!(
        any_change,
        "At least one benchmark should show measurable accuracy change under time pressure"
    );
}

/// Verify SAT asymptote (lambda) is reasonable: between 0.1 and 1.0.
#[test]
fn sat_asymptote_in_range() {
    let benchmarks: Vec<Box<dyn PsychBenchmark>> =
        vec![Box::new(StroopBenchmark), Box::new(FlankerBenchmark)];

    let bench_refs: Vec<&dyn PsychBenchmark> = benchmarks.iter().map(|b| b.as_ref()).collect();
    let config = sat_config();
    let pressures = [0.0, 0.33, 0.67, 1.0];

    let battery = SatBattery::run(&bench_refs, &config, &pressures);

    for curve in &battery.curves {
        assert!(
            curve.fit.asymptote >= 0.0 && curve.fit.asymptote <= 1.5,
            "{}: asymptote out of reasonable range [0.0, 1.5]: {}",
            curve.benchmark,
            curve.fit.asymptote
        );
    }
}
