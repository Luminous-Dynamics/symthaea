//! SEMI-VM synthetic structural-causal-model reference fixtures.
//!
//! This is a transparent development oracle, not a production diagnostic or
//! sealed benchmark. Hidden mechanism parameters are visible in this test-only
//! fixture so known-answer intervention properties can be checked.

#[derive(Clone, Copy, Debug)]
struct ProcessWorld {
    process_offset: f64,
    common_process_offset: f64,
    sensor_gain: f64,
    sensor_bias: f64,
    metrology_bias: f64,
    sensor_available: bool,
}

#[derive(Clone, Copy, Debug)]
struct Wafer {
    recipe: f64,
    product_effect: f64,
    tool_effect: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Observation {
    sensor: Option<f64>,
    reported_metrology: f64,
    independent_reference: Option<f64>,
}

fn observe(world: ProcessWorld, wafer: Wafer, request_reference: bool) -> Observation {
    let physical_quality = 2.0 * wafer.recipe
        + wafer.product_effect
        + wafer.tool_effect
        + world.process_offset
        + world.common_process_offset;
    Observation {
        sensor: world
            .sensor_available
            .then_some(world.sensor_gain * physical_quality + world.sensor_bias),
        reported_metrology: physical_quality + world.metrology_bias,
        independent_reference: request_reference.then_some(physical_quality),
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-10
}

fn baseline_world() -> ProcessWorld {
    ProcessWorld {
        process_offset: 0.0,
        common_process_offset: 0.0,
        sensor_gain: 1.0,
        sensor_bias: 0.0,
        metrology_bias: 0.0,
        sensor_available: true,
    }
}

fn wafer(tool_effect: f64) -> Wafer {
    Wafer {
        recipe: 2.0,
        product_effect: 1.5,
        tool_effect,
    }
}

#[test]
fn physical_process_shift_moves_sensor_and_independent_reference_together() {
    let base = observe(baseline_world(), wafer(0.0), true);
    let shifted_world = ProcessWorld {
        process_offset: 3.0,
        ..baseline_world()
    };
    let shifted = observe(shifted_world, wafer(0.0), true);

    assert!(close(shifted.sensor.unwrap() - base.sensor.unwrap(), 3.0));
    assert!(close(
        shifted.reported_metrology - base.reported_metrology,
        3.0
    ));
    assert!(close(
        shifted.independent_reference.unwrap() - base.independent_reference.unwrap(),
        3.0
    ));
}

#[test]
fn sensor_only_drift_changes_telemetry_not_physical_quality_or_metrology() {
    let base = observe(baseline_world(), wafer(0.0), true);
    let sensor_drift = ProcessWorld {
        sensor_bias: 4.0,
        ..baseline_world()
    };
    let shifted = observe(sensor_drift, wafer(0.0), true);

    assert!(close(shifted.sensor.unwrap() - base.sensor.unwrap(), 4.0));
    assert!(close(shifted.reported_metrology, base.reported_metrology));
    assert_eq!(shifted.independent_reference, base.independent_reference);
}

#[test]
fn calibration_drift_changes_reported_label_not_sensor_or_independent_reference() {
    let base = observe(baseline_world(), wafer(0.0), true);
    let calibration_drift = ProcessWorld {
        metrology_bias: -2.5,
        ..baseline_world()
    };
    let shifted = observe(calibration_drift, wafer(0.0), true);

    assert_eq!(shifted.sensor, base.sensor);
    assert!(close(
        shifted.reported_metrology - base.reported_metrology,
        -2.5
    ));
    assert_eq!(shifted.independent_reference, base.independent_reference);
}

#[test]
fn common_cause_process_shift_has_same_sign_across_tools() {
    let world = ProcessWorld {
        common_process_offset: 2.0,
        ..baseline_world()
    };
    let tool_a = observe(world, wafer(0.0), true);
    let tool_b = observe(world, wafer(0.8), true);
    let baseline_a = observe(baseline_world(), wafer(0.0), true);
    let baseline_b = observe(baseline_world(), wafer(0.8), true);

    let change_a =
        tool_a.independent_reference.unwrap() - baseline_a.independent_reference.unwrap();
    let change_b =
        tool_b.independent_reference.unwrap() - baseline_b.independent_reference.unwrap();
    assert!(close(change_a, 2.0));
    assert!(close(change_b, 2.0));
}

#[test]
fn chamber_specific_aging_remains_local_to_the_affected_tool() {
    let aged_a = ProcessWorld {
        process_offset: 1.75,
        ..baseline_world()
    };
    let stable_b = baseline_world();
    let tool_a = observe(aged_a, wafer(0.0), true);
    let tool_b = observe(stable_b, wafer(0.0), true);
    let baseline = observe(baseline_world(), wafer(0.0), true);

    assert!(close(
        tool_a.independent_reference.unwrap() - baseline.independent_reference.unwrap(),
        1.75
    ));
    assert!(close(
        tool_b.independent_reference.unwrap() - baseline.independent_reference.unwrap(),
        0.0
    ));
}

#[test]
fn process_shift_and_metrology_bias_can_be_observationally_equivalent() {
    let mut no_sensor = baseline_world();
    no_sensor.sensor_available = false;
    let process_shift = ProcessWorld {
        process_offset: 2.0,
        ..no_sensor
    };
    let calibration_shift = ProcessWorld {
        metrology_bias: 2.0,
        ..no_sensor
    };

    // Without a sensor or independent reference, both worlds expose the same
    // single reported-label value. The correct diagnosis is NOT_IDENTIFIABLE.
    let observed_process = observe(process_shift, wafer(0.0), false);
    let observed_calibration = observe(calibration_shift, wafer(0.0), false);
    assert_eq!(observed_process.sensor, None);
    assert_eq!(observed_calibration.sensor, None);
    assert!(close(
        observed_process.reported_metrology,
        observed_calibration.reported_metrology
    ));
    assert_eq!(observed_process.independent_reference, None);
    assert_eq!(observed_calibration.independent_reference, None);
}

#[test]
fn independent_reference_assay_breaks_process_vs_calibration_equivalence() {
    let mut no_sensor = baseline_world();
    no_sensor.sensor_available = false;
    let process_shift = ProcessWorld {
        process_offset: 2.0,
        ..no_sensor
    };
    let calibration_shift = ProcessWorld {
        metrology_bias: 2.0,
        ..no_sensor
    };

    let process = observe(process_shift, wafer(0.0), true);
    let calibration = observe(calibration_shift, wafer(0.0), true);
    assert!(close(
        process.reported_metrology,
        calibration.reported_metrology
    ));
    assert!(close(
        process.independent_reference.unwrap(),
        calibration.independent_reference.unwrap() + 2.0
    ));
}

#[test]
fn product_mix_can_move_pooled_mean_without_within_family_process_drift() {
    let world = baseline_world();
    let family_a: Vec<f64> = (0..100)
        .map(|_| {
            observe(
                world,
                Wafer {
                    product_effect: 0.0,
                    ..wafer(0.0)
                },
                false,
            )
            .reported_metrology
        })
        .collect();
    let family_b: Vec<f64> = (0..100)
        .map(|_| {
            observe(
                world,
                Wafer {
                    product_effect: 5.0,
                    ..wafer(0.0)
                },
                false,
            )
            .reported_metrology
        })
        .collect();

    let mean_a = family_a.iter().sum::<f64>() / family_a.len() as f64;
    let mean_b = family_b.iter().sum::<f64>() / family_b.len() as f64;
    let mixed_before = (mean_a * 90.0 + mean_b * 10.0) / 100.0;
    let mixed_after = (mean_a * 10.0 + mean_b * 90.0) / 100.0;

    assert!(close(mean_a, 5.5));
    assert!(close(mean_b, 10.5));
    assert!(mixed_after > mixed_before);
    assert!(close(
        observe(
            world,
            Wafer {
                product_effect: 0.0,
                ..wafer(0.0)
            },
            false
        )
        .reported_metrology,
        mean_a
    ));
    assert!(close(
        observe(
            world,
            Wafer {
                product_effect: 5.0,
                ..wafer(0.0)
            },
            false
        )
        .reported_metrology,
        mean_b
    ));
}

#[test]
fn missing_sensor_is_missing_not_a_numeric_zero() {
    let unavailable = ProcessWorld {
        sensor_available: false,
        ..baseline_world()
    };
    let observation = observe(unavailable, wafer(0.0), false);

    assert_eq!(observation.sensor, None);
    assert!(observation.reported_metrology.is_finite());
}

#[test]
fn selected_metrology_subset_is_not_population_prevalence() {
    let population: Vec<f64> = (0..100).map(f64::from).collect();
    let true_mean = population.iter().sum::<f64>() / population.len() as f64;
    let selected: Vec<f64> = population
        .iter()
        .copied()
        .filter(|value| *value >= 75.0)
        .collect();
    let selected_mean = selected.iter().sum::<f64>() / selected.len() as f64;

    assert!(close(true_mean, 49.5));
    assert!(selected_mean > true_mean);
    assert_eq!(selected.len(), 25);
}

#[test]
fn fixed_seed_scm_replay_is_deterministic() {
    let world = ProcessWorld {
        process_offset: 0.75,
        common_process_offset: -0.25,
        sensor_gain: 1.1,
        sensor_bias: -0.4,
        metrology_bias: 0.15,
        sensor_available: true,
    };
    let first = observe(world, wafer(0.3), true);
    let second = observe(world, wafer(0.3), true);
    assert_eq!(first, second);
}
