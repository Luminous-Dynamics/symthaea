use std::collections::HashSet;

use symthaea_interoception::{
    assess_allostasis, assess_allostasis_with_drive, AllostaticConfig, InteroceptiveDrive,
    NativeInteroceptiveModel, NativeInteroceptiveState, ViabilityChannel, CHANNEL_COUNT,
};

#[test]
fn stable_channel_ids_are_unique_and_complete() {
    let ids: HashSet<_> = ViabilityChannel::ALL
        .iter()
        .map(|channel| channel.stable_id())
        .collect();
    assert_eq!(ids.len(), CHANNEL_COUNT);
}

#[test]
fn lower_viability_boundary_normalizes_to_one_for_every_channel() {
    for channel in ViabilityChannel::ALL {
        let baseline = NativeInteroceptiveState::default();
        let variable = baseline.get(channel);
        assert!(variable.viable_low() < variable.preferred_low());
        let state = baseline.with_value(channel, variable.viable_low());
        let deviation = state.get(channel).normalized_deviation();

        assert!(
            (deviation - 1.0).abs() < 1e-6,
            "unexpected lower-bound normalization for {channel:?}: {deviation}"
        );
    }
}

#[test]
fn undriven_recovery_never_increases_deviation_from_below_the_preferred_band() {
    for channel in ViabilityChannel::ALL {
        let baseline = NativeInteroceptiveState::default();
        let variable = baseline.get(channel);
        let initial = 0.5 * (variable.viable_low() + variable.preferred_low());
        let state = baseline.with_value(channel, initial);

        let mut model = NativeInteroceptiveModel::new(state, Default::default());
        let mut previous = model.state().get(channel).normalized_deviation();

        for _ in 0..64 {
            model.step(InteroceptiveDrive::ZERO);
            let current = model.state().get(channel).normalized_deviation();
            assert!(
                current <= previous + 1e-6,
                "undriven recovery increased deviation for {channel:?}: {previous} -> {current}"
            );
            previous = current;
        }
    }
}

#[test]
fn extreme_drives_are_bounded_and_finite() {
    for channel in ViabilityChannel::ALL {
        for rate in [-100.0_f32, 100.0_f32] {
            let mut model = NativeInteroceptiveModel::default();
            let report = model.step(InteroceptiveDrive::ZERO.with_rate(channel, rate));
            let value = model.state().get(channel).value();

            assert!(value.is_finite());
            assert!((0.0..=1.0).contains(&value));
            assert!(report.clamped_channels > 0);
            assert!(report.driven_channels > 0);
        }
    }
}

#[test]
fn allostatic_reports_are_finite_at_discount_extremes() {
    let state = NativeInteroceptiveState::default().with_observation(
        ViabilityChannel::ComputeReserve,
        0.75,
        -0.04,
    );

    for discount in [0.0_f32, 1.0_f32] {
        let config = AllostaticConfig {
            horizon_steps: 32,
            dt: 1.0,
            discount,
        };
        let report = assess_allostasis(&state, config);
        assert!(report.discounted_debt.is_finite());
        assert!(report.peak_projected_deviation.is_finite());
        assert!(report.terminal_deviation.is_finite());
        assert!(report.channel_debt.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn dynamics_aware_forecasts_are_deterministic_across_all_channels() {
    let config = AllostaticConfig {
        horizon_steps: 12,
        dt: 1.0,
        discount: 0.93,
    };

    for channel in ViabilityChannel::ALL {
        let drive = InteroceptiveDrive::ZERO.with_rate(channel, -0.03);
        let left = NativeInteroceptiveModel::default();
        let right = NativeInteroceptiveModel::default();

        assert_eq!(
            assess_allostasis_with_drive(&left, drive, config),
            assess_allostasis_with_drive(&right, drive, config),
            "forecast diverged for {channel:?}"
        );
    }
}
