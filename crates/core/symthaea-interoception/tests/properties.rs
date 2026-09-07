use proptest::prelude::*;
use symthaea_interoception::{
    assess_allostasis_with_drive, assess_homeostasis, AllostaticConfig, InteroceptiveDrive,
    InteroceptiveDynamicsConfig, NativeInteroceptiveModel, NativeInteroceptiveState,
    ViabilityChannel, ViabilityVariable,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn valid_viability_variables_have_finite_nonnegative_deviation(
        value_i in -200i32..=200,
        preferred_low_i in -100i32..=0,
        preferred_width_i in 1u32..=100,
        lower_margin_i in 1u32..=100,
        upper_margin_i in 1u32..=100,
        precision_i in 0u32..=200,
        importance_i in 0u32..=200,
    ) {
        let preferred_low = preferred_low_i as f32 / 100.0;
        let preferred_high = preferred_low + preferred_width_i as f32 / 100.0;
        let viable_low = preferred_low - lower_margin_i as f32 / 100.0;
        let viable_high = preferred_high + upper_margin_i as f32 / 100.0;
        let value = value_i as f32 / 100.0;
        let precision = precision_i as f32 / 100.0;
        let importance = importance_i as f32 / 100.0;

        let variable = ViabilityVariable::new(
            value,
            preferred_low,
            preferred_high,
            viable_low,
            viable_high,
            precision,
            importance,
        );
        let deviation = variable.normalized_deviation();

        prop_assert!(deviation.is_finite());
        prop_assert!(deviation >= 0.0);
        if value >= preferred_low && value <= preferred_high {
            prop_assert_eq!(deviation, 0.0);
        }
    }

    #[test]
    fn bounded_dynamics_stay_finite_under_arbitrary_finite_drives(
        dt_i in 1u32..=100,
        recovery_gain_i in 0u32..=100,
        drive_a_i in -1000i32..=1000,
        drive_b_i in -1000i32..=1000,
        steps in 1u32..=64,
    ) {
        let step_dt = dt_i as f32 / 100.0;
        let recovery_gain = recovery_gain_i as f32 / 100.0;
        let recovery_rate = recovery_gain / step_dt;
        let config = InteroceptiveDynamicsConfig {
            step_dt,
            recovery_rate,
            min_value: 0.0,
            max_value: 1.0,
        };
        let drive = InteroceptiveDrive::ZERO
            .with_rate(ViabilityChannel::ComputeReserve, drive_a_i as f32 / 100.0)
            .with_rate(ViabilityChannel::Integrity, drive_b_i as f32 / 100.0);
        let mut model = NativeInteroceptiveModel::new(NativeInteroceptiveState::default(), config);

        for _ in 0..steps {
            model.step(drive);
            for variable in model.state().channels() {
                prop_assert!(variable.value().is_finite());
                prop_assert!(variable.velocity().is_finite());
                prop_assert!(variable.value() >= config.min_value);
                prop_assert!(variable.value() <= config.max_value);
            }
        }
    }

    #[test]
    fn undriven_passive_recovery_never_increases_deviation(
        initial_i in 0u32..=64,
        dt_i in 1u32..=100,
        recovery_gain_i in 0u32..=100,
        steps in 1u32..=64,
    ) {
        let initial = initial_i as f32 / 100.0;
        let state = NativeInteroceptiveState::default()
            .with_value(ViabilityChannel::ComputeReserve, initial);
        let step_dt = dt_i as f32 / 100.0;
        let recovery_gain = recovery_gain_i as f32 / 100.0;
        let config = InteroceptiveDynamicsConfig {
            step_dt,
            recovery_rate: recovery_gain / step_dt,
            min_value: 0.0,
            max_value: 1.0,
        };
        let mut model = NativeInteroceptiveModel::new(state, config);
        let mut previous = assess_homeostasis(model.state()).channel_deviations
            [ViabilityChannel::ComputeReserve.index()];

        for _ in 0..steps {
            model.step(InteroceptiveDrive::ZERO);
            let current = assess_homeostasis(model.state()).channel_deviations
                [ViabilityChannel::ComputeReserve.index()];
            prop_assert!(current <= previous + 1e-6);
            previous = current;
        }
    }

    #[test]
    fn identical_rollouts_are_exactly_deterministic(
        dt_i in 1u32..=100,
        recovery_gain_i in 0u32..=100,
        drive_i in -100i32..=100,
        steps in 1u32..=64,
    ) {
        let step_dt = dt_i as f32 / 100.0;
        let recovery_gain = recovery_gain_i as f32 / 100.0;
        let config = InteroceptiveDynamicsConfig {
            step_dt,
            recovery_rate: recovery_gain / step_dt,
            min_value: 0.0,
            max_value: 1.0,
        };
        let drive = InteroceptiveDrive::ZERO
            .with_rate(ViabilityChannel::EpistemicResolution, drive_i as f32 / 100.0);
        let mut left = NativeInteroceptiveModel::new(NativeInteroceptiveState::default(), config);
        let mut right = left.clone();

        for _ in 0..steps {
            prop_assert_eq!(left.step(drive), right.step(drive));
        }
        prop_assert_eq!(left, right);
    }

    #[test]
    fn stronger_declared_degradation_never_has_lower_allostatic_debt(
        weak_i in 0u32..=20,
        extra_i in 0u32..=40,
        horizon in 1u16..=32,
        discount_i in 0u32..=100,
    ) {
        let model = NativeInteroceptiveModel::default();
        let weak = -(weak_i as f32 / 1000.0);
        let strong = -((weak_i + extra_i) as f32 / 1000.0);
        let config = AllostaticConfig {
            horizon_steps: horizon,
            dt: model.config().step_dt,
            discount: discount_i as f32 / 100.0,
        };
        let weak_report = assess_allostasis_with_drive(
            &model,
            InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, weak),
            config,
        );
        let strong_report = assess_allostasis_with_drive(
            &model,
            InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, strong),
            config,
        );

        prop_assert!(strong_report.discounted_debt + 1e-6 >= weak_report.discounted_debt);
        prop_assert!(strong_report.peak_projected_deviation + 1e-6 >= weak_report.peak_projected_deviation);
    }
}
