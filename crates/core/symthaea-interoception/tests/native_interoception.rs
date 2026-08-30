use symthaea_interoception::{
    assess_allostasis, assess_homeostasis, AllostaticConfig, InteroceptiveDrive,
    InteroceptiveSnapshot, NativeInteroceptiveModel, NativeInteroceptiveState,
    ViabilityChannel,
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
    assert_eq!(allo.projected_viability_breaches, 0);
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
fn allostasis_detects_future_deterioration_before_current_deviation() {
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
    assert!(future.projected_viability_breaches > 0);
}

#[test]
fn zero_drive_recovers_a_perturbed_channel_toward_its_preferred_band() {
    let mut state = NativeInteroceptiveState::default();
    state.get_mut(ViabilityChannel::ComputeReserve).value = 0.30;
    let before = assess_homeostasis(&state).weighted_deviation;

    let mut model = NativeInteroceptiveModel::new(state, Default::default());
    for _ in 0..20 {
        model.step(InteroceptiveDrive::ZERO);
    }

    let after = assess_homeostasis(model.state()).weighted_deviation;
    assert!(after < before);
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
