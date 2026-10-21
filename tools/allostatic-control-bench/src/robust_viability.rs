//! Interval-envelope viability checks for uncertain one-step dynamics.
//!
//! Unlike confidence-adjusted point forecasts, this module asks whether every
//! state/action effect inside declared bounds respects the next-step limits.
//! Rectangular intervals are conservative: they discard correlations between
//! effects, so richer polytopes or zonotopes may reduce conservatism later.
//!
//! A successful check is conditional on the declared envelope containing the
//! real state and transition. It is not a formal guarantee under omitted error.

use super::Action;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    pub fn new(lower: f64, upper: f64) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Self { lower, upper })
    }

    pub fn point(value: f64) -> Option<Self> {
        Self::new(value, value)
    }

    pub const fn lower(self) -> f64 {
        self.lower
    }

    pub const fn upper(self) -> f64 {
        self.upper
    }

    fn add(self, other: Self) -> Option<Self> {
        Self::new(self.lower + other.lower, self.upper + other.upper)
    }

    fn clamp(self, lower: f64, upper: f64) -> Option<Self> {
        Self::new(
            self.lower.clamp(lower, upper),
            self.upper.clamp(lower, upper),
        )
    }

    fn is_within(self, lower: f64, upper: f64) -> bool {
        self.lower >= lower && self.upper <= upper
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateEnvelope {
    reserve: Interval,
    load: Interval,
}

impl StateEnvelope {
    pub fn new(reserve: Interval, load: Interval) -> Option<Self> {
        if reserve.is_within(0.0, 1.0) && load.is_within(0.0, 1.0) {
            Some(Self { reserve, load })
        } else {
            None
        }
    }

    pub const fn reserve(self) -> Interval {
        self.reserve
    }

    pub const fn load(self) -> Interval {
        self.load
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionEffectBounds {
    reserve_delta: Interval,
    load_delta: Interval,
}

impl ActionEffectBounds {
    pub fn new(reserve_delta: Interval, load_delta: Interval) -> Option<Self> {
        if reserve_delta.is_within(-1.0, 1.0) && load_delta.is_within(-1.0, 1.0) {
            Some(Self {
                reserve_delta,
                load_delta,
            })
        } else {
            None
        }
    }

    pub const fn reserve_delta(self) -> Interval {
        self.reserve_delta
    }

    pub const fn load_delta(self) -> Interval {
        self.load_delta
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsBounds {
    push: ActionEffectBounds,
    maintain: ActionEffectBounds,
    emergency_recover: ActionEffectBounds,
}

impl DynamicsBounds {
    pub const fn new(
        push: ActionEffectBounds,
        maintain: ActionEffectBounds,
        emergency_recover: ActionEffectBounds,
    ) -> Self {
        Self {
            push,
            maintain,
            emergency_recover,
        }
    }

    /// Return the transition envelope for one available action.
    pub const fn effect_for_action(self, action: Action) -> ActionEffectBounds {
        match action {
            Action::Push => self.push,
            Action::Maintain => self.maintain,
            Action::EmergencyRecover => self.emergency_recover,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViabilityLimits {
    min_reserve: f64,
    max_load: f64,
}

impl ViabilityLimits {
    pub fn new(min_reserve: f64, max_load: f64) -> Option<Self> {
        (min_reserve.is_finite()
            && max_load.is_finite()
            && (0.0..=1.0).contains(&min_reserve)
            && (0.0..=1.0).contains(&max_load))
        .then_some(Self {
            min_reserve,
            max_load,
        })
    }

    pub const fn min_reserve(self) -> f64 {
        self.min_reserve
    }

    pub const fn max_load(self) -> f64 {
        self.max_load
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobustActionSet {
    push: bool,
    maintain: bool,
    emergency_recover: bool,
}

impl RobustActionSet {
    pub const fn push(self) -> bool {
        self.push
    }

    pub const fn maintain(self) -> bool {
        self.maintain
    }

    pub const fn emergency_recover(self) -> bool {
        self.emergency_recover
    }

    pub const fn count(self) -> usize {
        self.push as usize + self.maintain as usize + self.emergency_recover as usize
    }

    pub const fn contains(self, action: Action) -> bool {
        match action {
            Action::Push => self.push,
            Action::Maintain => self.maintain,
            Action::EmergencyRecover => self.emergency_recover,
        }
    }

    /// Keep the proposal when robustly viable. Otherwise use a deterministic
    /// task-preserving fallback. None means no action is certified viable.
    pub fn filter(self, proposed: Action) -> Option<Action> {
        if self.contains(proposed) {
            return Some(proposed);
        }
        if self.maintain {
            return Some(Action::Maintain);
        }
        if self.emergency_recover {
            return Some(Action::EmergencyRecover);
        }
        if self.push {
            return Some(Action::Push);
        }
        None
    }
}

/// Conservative one-step reachable envelope under independent interval bounds.
pub fn next_state_envelope(
    state: StateEnvelope,
    effects: ActionEffectBounds,
) -> Option<StateEnvelope> {
    let reserve = state.reserve.add(effects.reserve_delta)?.clamp(0.0, 1.0)?;
    let load = state.load.add(effects.load_delta)?.clamp(0.0, 1.0)?;
    StateEnvelope::new(reserve, load)
}

/// True only when the worst-case next reserve and load meet both limits.
pub fn robustly_viable(
    state: StateEnvelope,
    effects: ActionEffectBounds,
    limits: ViabilityLimits,
) -> bool {
    next_state_envelope(state, effects).is_some_and(|next| {
        next.reserve.lower >= limits.min_reserve && next.load.upper <= limits.max_load
    })
}

/// Compute the robustly viable action set. Emergency recovery is included only
/// when the controller has authority to invoke it. Invalid contracts fail closed.
pub fn evaluate_action_set(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
) -> RobustActionSet {
    RobustActionSet {
        push: robustly_viable(state, dynamics.effect_for_action(Action::Push), limits),
        maintain: robustly_viable(state, dynamics.effect_for_action(Action::Maintain), limits),
        emergency_recover: shield_can_recover
            && robustly_viable(state, dynamics.effect_for_action(Action::EmergencyRecover), limits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("test interval should be valid")
    }

    fn effects(reserve_delta: (f64, f64), load_delta: (f64, f64)) -> ActionEffectBounds {
        ActionEffectBounds::new(
            interval(reserve_delta.0, reserve_delta.1),
            interval(load_delta.0, load_delta.1),
        )
        .expect("test effects should be valid")
    }

    fn nominal_dynamics() -> DynamicsBounds {
        DynamicsBounds::new(
            effects((-0.06, -0.05), (0.02, 0.03)),
            effects((0.03, 0.06), (-0.05, -0.02)),
            effects((0.08, 0.12), (-0.12, -0.08)),
        )
    }

    fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
        StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("test state should be valid")
    }

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("test limits should be valid")
    }

    #[test]
    fn constructors_reject_nonfinite_inverted_and_out_of_range_bounds() {
        assert!(Interval::new(f64::NAN, 0.2).is_none());
        assert!(Interval::new(0.4, 0.3).is_none());
        assert!(StateEnvelope::new(interval(-0.1, 0.2), interval(0.0, 0.2)).is_none());
        assert!(ActionEffectBounds::new(interval(-1.1, -1.0), interval(0.0, 0.1)).is_none());
        assert!(ViabilityLimits::new(0.9, 0.2).is_some());
        assert!(ViabilityLimits::new(0.1, f64::INFINITY).is_none());
    }

    #[test]
    fn bounded_uncertainty_vetoes_an_action_that_is_safe_at_the_midpoint() {
        let state = state((0.80, 0.80), (0.84, 0.84));
        let narrow = effects((-0.06, -0.05), (0.02, 0.03));
        let wide = effects((-0.06, -0.05), (0.02, 0.06));

        assert!(robustly_viable(state, narrow, limits()));
        assert!(!robustly_viable(state, wide, limits()));
    }

    #[test]
    fn state_estimation_uncertainty_must_be_safe_at_the_worst_case() {
        let narrow = state((0.18, 0.18), (0.82, 0.82));
        let wide = state((0.14, 0.20), (0.82, 0.85));
        let push = effects((-0.01, -0.005), (0.01, 0.02));

        assert!(robustly_viable(narrow, push, limits()));
        assert!(!robustly_viable(wide, push, limits()));
    }

    #[test]
    fn widening_transition_bounds_cannot_add_robustly_safe_actions() {
        let current = state((0.80, 0.80), (0.84, 0.84));
        let narrow = DynamicsBounds::new(
            effects((-0.06, -0.05), (0.02, 0.03)),
            effects((0.03, 0.06), (-0.05, -0.02)),
            effects((0.08, 0.12), (-0.12, -0.08)),
        );
        let wide = DynamicsBounds::new(
            effects((-0.08, -0.04), (0.01, 0.06)),
            effects((0.01, 0.07), (-0.07, 0.02)),
            effects((0.05, 0.14), (-0.14, -0.05)),
        );
        let narrow_set = evaluate_action_set(current, narrow, limits(), true);
        let wide_set = evaluate_action_set(current, wide, limits(), true);

        assert!(!wide_set.push() || narrow_set.push());
        assert!(!wide_set.maintain() || narrow_set.maintain());
        assert!(!wide_set.emergency_recover() || narrow_set.emergency_recover());
    }

    #[test]
    fn emergency_action_is_not_counted_without_authority() {
        let current = state((0.16, 0.16), (0.87, 0.87));
        let dynamics = DynamicsBounds::new(
            effects((-0.08, -0.05), (0.02, 0.05)),
            effects((0.03, 0.07), (-0.02, 0.03)),
            effects((0.08, 0.12), (-0.12, -0.08)),
        );
        let unshielded = evaluate_action_set(current, dynamics, limits(), false);
        let shielded = evaluate_action_set(current, dynamics, limits(), true);

        assert!(!unshielded.emergency_recover());
        assert!(shielded.emergency_recover());
        assert_eq!(shielded.count(), unshielded.count() + 1);
    }

    #[test]
    fn absent_robust_action_is_not_fabricated_as_safe() {
        let current = state((0.15, 0.15), (0.88, 0.88));
        let dangerous = DynamicsBounds::new(
            effects((-0.10, -0.08), (0.02, 0.05)),
            effects((-0.01, 0.01), (0.02, 0.05)),
            effects((0.01, 0.03), (-0.01, 0.01)),
        );
        let set = evaluate_action_set(current, dangerous, limits(), true);

        assert_eq!(set.count(), 0);
        assert_eq!(set.filter(Action::Push), None);
    }

    #[test]
    fn filter_keeps_a_safe_proposal_and_uses_a_safe_fallback_when_needed() {
        let current = state((0.80, 0.80), (0.60, 0.60));
        let set = evaluate_action_set(current, nominal_dynamics(), limits(), true);
        assert_eq!(set.filter(Action::Push), Some(Action::Push));

        let stressed = state((0.16, 0.16), (0.87, 0.87));
        let stressed_dynamics = DynamicsBounds::new(
            effects((-0.08, -0.05), (0.02, 0.05)),
            effects((0.03, 0.07), (-0.02, 0.03)),
            effects((0.08, 0.12), (-0.12, -0.08)),
        );
        let stressed_set = evaluate_action_set(stressed, stressed_dynamics, limits(), true);
        assert_eq!(
            stressed_set.filter(Action::Push),
            Some(Action::EmergencyRecover)
        );
    }
}
