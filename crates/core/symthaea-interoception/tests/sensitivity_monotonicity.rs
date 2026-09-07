use symthaea_interoception::{
    assess_allostasis_with_drive, assess_homeostasis, AllostaticConfig, InteroceptiveDrive,
    InteroceptiveDynamicsConfig, NativeInteroceptiveModel, NativeInteroceptiveState,
    ViabilityChannel, ViabilityVariable, CHANNEL_COUNT,
};

fn standard_variable(value: f32, precision: f32, importance: f32) -> ViabilityVariable {
    ViabilityVariable::new(value, 0.65, 0.85, 0.25, 1.0, precision, importance)
}

fn state_with_compute(value: f32, precision: f32, importance: f32) -> NativeInteroceptiveState {
    let neutral = standard_variable(0.75, precision, importance);
    let mut channels = [neutral; CHANNEL_COUNT];
    channels[ViabilityChannel::ComputeReserve.index()] =
        standard_variable(value, precision, importance);
    NativeInteroceptiveState::new(channels)
}

#[test]
fn narrower_preferred_band_cannot_reduce_same_state_deviation() {
    let value = 0.50;
    let wide = ViabilityVariable::new(value, 0.60, 0.90, 0.00, 1.00, 1.0, 1.0);
    let narrow = ViabilityVariable::new(value, 0.70, 0.80, 0.00, 1.00, 1.0, 1.0);

    assert!(narrow.normalized_deviation() >= wide.normalized_deviation());
}

#[test]
fn wider_viability_margin_cannot_increase_same_state_deviation() {
    let value = 0.50;
    let narrow_margin = ViabilityVariable::new(value, 0.60, 0.80, 0.40, 1.00, 1.0, 1.0);
    let wide_margin = ViabilityVariable::new(value, 0.60, 0.80, 0.00, 1.00, 1.0, 1.0);

    assert!(wide_margin.normalized_deviation() <= narrow_margin.normalized_deviation());
}

#[test]
fn uniform_weight_scaling_does_not_change_aggregate_deviation() {
    let low_scale = assess_homeostasis(&state_with_compute(0.45, 0.5, 1.0));
    let high_scale = assess_homeostasis(&state_with_compute(0.45, 2.0, 1.0));

    assert!((low_scale.weighted_deviation - high_scale.weighted_deviation).abs() < 1e-6);
}

#[test]
fn increasing_deviated_channel_weight_increases_aggregate_deviation() {
    let neutral = standard_variable(0.75, 1.0, 1.0);

    let mut low_channels = [neutral; CHANNEL_COUNT];
    low_channels[ViabilityChannel::ComputeReserve.index()] =
        standard_variable(0.45, 1.0, 0.25);
    let low = assess_homeostasis(&NativeInteroceptiveState::new(low_channels));

    let mut high_channels = [neutral; CHANNEL_COUNT];
    high_channels[ViabilityChannel::ComputeReserve.index()] =
        standard_variable(0.45, 1.0, 4.0);
    let high = assess_homeostasis(&NativeInteroceptiveState::new(high_channels));

    assert!(high.weighted_deviation > low.weighted_deviation);
}

#[test]
fn zero_weight_excludes_aggregate_influence_without_erasing_raw_breach_evidence() {
    let neutral = standard_variable(0.75, 1.0, 1.0);
    let mut channels = [neutral; CHANNEL_COUNT];
    channels[ViabilityChannel::ComputeReserve.index()] =
        standard_variable(0.10, 1.0, 0.0);
    let report = assess_homeostasis(&NativeInteroceptiveState::new(channels));

    assert_eq!(report.weighted_deviation, 0.0);
    assert!(report.peak_deviation > 1.0);
    assert_eq!(report.violated_channels, 1);
    assert!(report.channel_deviations[ViabilityChannel::ComputeReserve.index()] > 1.0);
}

#[test]
fn longer_deteriorating_forecast_does_not_look_healthier() {
    let model = NativeInteroceptiveModel::default();
    let drive =
        InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.03);
    let short = assess_allostasis_with_drive(
        &model,
        drive,
        AllostaticConfig {
            horizon_steps: 8,
            dt: 1.0,
            discount: 0.95,
        },
    );
    let long = assess_allostasis_with_drive(
        &model,
        drive,
        AllostaticConfig {
            horizon_steps: 16,
            dt: 1.0,
            discount: 0.95,
        },
    );

    assert!(long.discounted_debt >= short.discounted_debt);
    assert!(long.peak_projected_deviation >= short.peak_projected_deviation);
}

#[test]
fn greater_future_weight_cannot_reduce_debt_on_monotonic_deterioration() {
    let model = NativeInteroceptiveModel::default();
    let drive =
        InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.03);
    let near_weighted = assess_allostasis_with_drive(
        &model,
        drive,
        AllostaticConfig {
            horizon_steps: 16,
            dt: 1.0,
            discount: 0.50,
        },
    );
    let uniform = assess_allostasis_with_drive(
        &model,
        drive,
        AllostaticConfig {
            horizon_steps: 16,
            dt: 1.0,
            discount: 1.00,
        },
    );

    assert!(uniform.discounted_debt >= near_weighted.discounted_debt);
}

#[test]
fn stronger_passive_recovery_reduces_debt_under_same_declared_load() {
    let state = NativeInteroceptiveState::default();
    let drive =
        InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.03);
    let forecast = AllostaticConfig {
        horizon_steps: 16,
        dt: 1.0,
        discount: 0.95,
    };
    let slow = NativeInteroceptiveModel::new(
        state.clone(),
        InteroceptiveDynamicsConfig {
            step_dt: 1.0,
            recovery_rate: 0.01,
            min_value: 0.0,
            max_value: 1.0,
        },
    );
    let fast = NativeInteroceptiveModel::new(
        state,
        InteroceptiveDynamicsConfig {
            step_dt: 1.0,
            recovery_rate: 0.20,
            min_value: 0.0,
            max_value: 1.0,
        },
    );

    let slow_report = assess_allostasis_with_drive(&slow, drive, forecast);
    let fast_report = assess_allostasis_with_drive(&fast, drive, forecast);
    assert!(fast_report.discounted_debt <= slow_report.discounted_debt);
}

#[test]
fn longer_drive_persistence_leaves_no_better_terminal_regulatory_state() {
    fn run(driven_steps: usize) -> f32 {
        let mut model = NativeInteroceptiveModel::default();
        let drive =
            InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.03);
        for step in 0..16 {
            model.step(if step < driven_steps {
                drive
            } else {
                InteroceptiveDrive::ZERO
            });
        }
        assess_homeostasis(model.state()).weighted_deviation
    }

    assert!(run(12) >= run(4));
}
