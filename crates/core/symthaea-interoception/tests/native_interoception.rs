use symthaea_interoception::{
    apply_intervention, assess_allostasis, assess_allostasis_with_drive, assess_homeostasis,
    AllostaticConfig, InteroceptiveDrive, InteroceptiveIntervention, InteroceptiveSnapshot,
    NativeInteroceptiveModel, NativeInteroceptiveState, ViabilityChannel,
};

#[test]
fn default_state_has_zero_regulatory_deviation() {
    let state = NativeInteroceptiveState::default();
    let home = assess_homeostasis(&state);
    let allo = assess_allostasis(&state, AllostaticConfig::default());

    assert_eq!(home.weighted_deviation, 0.0);
    assert_eq!(home.peak_deviation, 0.0);
    assert!(home.is_within_viability());
    assert_eq!(allo.discounted_debt, 0.0);
    assert_eq!(allo.breach_exposures, 0);
    assert_eq!(allo.unique_breached_channels, 0);
    assert_eq!(allo.first_breach_step, None);
}

#[test]
fn deviation_is_zero_inside_preferred_band_and_normalized_outside_it() {
    let mut state = NativeInteroceptiveState::default();
    state.get_mut(ViabilityChannel::ComputeReserve).value = 0.45;

    let report = assess_homeostasis(&state);
    let deviation = report.channel_deviations[ViabilityChannel::ComputeReserve.index()];

    assert!((deviation - 0.5).abs() < 1e-6);
    assert!((report.peak_deviation - 0.5).abs() < 1e-6);
    assert!(report.is_within_viability());
}

#[test]
fn kinematic_allostasis_detects_future_deterioration_before_current_deviation() {
    let mut state = NativeInteroceptiveState::default();
    let reserve = state.get_mut(ViabilityChannel::ComputeReserve);
    reserve.value = 0.70;
    reserve.velocity = -0.08;

    let current = assess_homeostasis(&state);
    let future = assess_allostasis(
        &state,
        AllostaticConfig {
            horizon_steps: 8,
            dt: 1.0,
            discount: 1.0,
        },
    );

    assert_eq!(current.weighted_deviation, 0.0);
    assert!(future.discounted_debt > 0.0);
    assert!(future.terminal_deviation > 0.0);
    assert!(future.breach_exposures > 0);
    assert_eq!(future.unique_breached_channels, 1);
    assert!(future.first_breach_step.is_some());
}

#[test]
fn dynamics_aware_allostasis_uses_declared_future_drive() {
    let model = NativeInteroceptiveModel::default();
    let config = AllostaticConfig {
        horizon_steps: 8,
        dt: 1.0,
        discount: 1.0,
    };

    let kinematic = assess_allostasis(model.state(), config);
    assert_eq!(kinematic.discounted_debt, 0.0);

    let drive = InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.08);
    let rollout = assess_allostasis_with_drive(&model, drive, config);
    assert!(rollout.discounted_debt > 0.0);
    assert!(rollout.terminal_deviation > 0.0);
    assert!(rollout.breach_exposures > 0);
}

#[test]
fn zero_drive_recovers_a_perturbed_channel_toward_its_preferred_band() {
    let mut state = NativeInteroceptiveState::default();
    state.get_mut(ViabilityChannel::ComputeReserve).value = 0.30;
    let before = assess_homeostasis(&state).weighted_deviation;

    let mut model = NativeInteroceptiveModel::new(state, Default::default());
    let first = model.step(InteroceptiveDrive::ZERO);
    assert_eq!(first.restorative_channels, 1);
    assert_eq!(first.driven_channels, 0);
    assert_eq!(first.changed_channels, 1);

    for _ in 1..20 {
        model.step(InteroceptiveDrive::ZERO);
    }

    let after = assess_homeostasis(model.state()).weighted_deviation;
    assert!(after < before);
}

#[test]
fn rollout_forecast_represents_native_recovery() {
    let mut state = NativeInteroceptiveState::default();
    state.get_mut(ViabilityChannel::ComputeReserve).value = 0.30;
    let model = NativeInteroceptiveModel::new(state, Default::default());
    let current = assess_homeostasis(model.state()).weighted_deviation;
    let future = assess_allostasis_with_drive(
        &model,
        InteroceptiveDrive::ZERO,
        AllostaticConfig::default(),
    );

    assert!(future.terminal_deviation < current);
}

#[test]
fn zero_drive_preserves_states_already_inside_preferred_band() {
    let mut state = NativeInteroceptiveState::default();
    state.get_mut(ViabilityChannel::ComputeReserve).value = 0.68;
    state.get_mut(ViabilityChannel::NoveltyBalance).value = 0.57;
    let before = state.clone();

    let mut model = NativeInteroceptiveModel::new(state, Default::default());
    let report = model.step(InteroceptiveDrive::ZERO);

    assert_eq!(report.driven_channels, 0);
    assert_eq!(report.restorative_channels, 0);
    assert_eq!(report.clamped_channels, 0);
    assert_eq!(report.changed_channels, 0);

    for channel in ViabilityChannel::ALL {
        assert_eq!(
            model.state().get(channel).value,
            before.get(channel).value,
            "zero drive must not create an implicit midpoint setpoint for {channel:?}"
        );
        assert_eq!(model.state().get(channel).velocity, 0.0);
    }
}

#[test]
fn interventions_are_explicit_clamped_and_do_not_create_false_velocity() {
    let mut model = NativeInteroceptiveModel::default();
    model.step(InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::Integrity, -0.1));
    assert!(model.state().get(ViabilityChannel::Integrity).velocity < 0.0);

    let record = apply_intervention(
        &mut model,
        InteroceptiveIntervention::set(ViabilityChannel::Integrity, -1.0),
    );

    assert_eq!(record.cycle, 1);
    assert_eq!(record.before, 0.8);
    assert_eq!(record.requested, -1.0);
    assert_eq!(record.after, 0.0);
    assert!(record.clamped);
    assert_eq!(model.state().get(ViabilityChannel::Integrity).velocity, 0.0);
    assert!(!assess_homeostasis(model.state()).is_within_viability());
}

#[test]
fn identical_initial_state_and_drives_produce_identical_snapshots() {
    let mut left = NativeInteroceptiveModel::default();
    let mut right = NativeInteroceptiveModel::default();
    let drive = InteroceptiveDrive::ZERO
        .with_rate(ViabilityChannel::ComputeReserve, -0.015)
        .with_rate(ViabilityChannel::EpistemicResolution, -0.007)
        .with_rate(ViabilityChannel::NoveltyBalance, 0.004);

    for _ in 0..32 {
        left.step(drive);
        right.step(drive);
    }

    let left_snapshot = InteroceptiveSnapshot::capture(&left, AllostaticConfig::default());
    let right_snapshot = InteroceptiveSnapshot::capture(&right, AllostaticConfig::default());
    assert_eq!(left_snapshot, right_snapshot);
}

#[test]
fn core_source_contains_no_named_state_categories() {
    let source = [
        include_str!("../src/lib.rs"),
        include_str!("../src/state.rs"),
        include_str!("../src/homeostasis.rs"),
        include_str!("../src/allostasis.rs"),
        include_str!("../src/dynamics.rs"),
        include_str!("../src/intervention.rs"),
        include_str!("../src/snapshot.rs"),
    ]
    .join("\n")
    .to_ascii_lowercase();

    for forbidden in ["emotion", "fear", "joy", "anger", "sadness", "grief"] {
        assert!(
            !source.contains(forbidden),
            "core interoception source must remain category-free: found {forbidden}"
        );
    }
}
