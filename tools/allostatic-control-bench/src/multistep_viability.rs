//! Finite-horizon robust viability using conservative interval tubes.
//!
//! A first action is certified only if there exists one open-loop continuation
//! sequence that keeps the entire reachable interval within the viability limits
//! at every intermediate step. This is sound under the declared rectangular
//! dynamics bounds, but conservative: it does not search state-contingent
//! continuation policies over partially observed state sets.

use super::robust_viability::{
    next_state_envelope, DynamicsBounds, StateEnvelope, ViabilityLimits,
};
use super::Action;

pub const MAX_CERTIFIED_HORIZON: usize = 8;

const ALL_ACTIONS: [Action; 3] = [Action::Push, Action::Maintain, Action::EmergencyRecover];
const NO_RECOVERY_ACTIONS: [Action; 2] = [Action::Push, Action::Maintain];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HorizonViableActionSet {
    horizon: usize,
    push: bool,
    maintain: bool,
    emergency_recover: bool,
}

impl HorizonViableActionSet {
    pub const fn horizon(self) -> usize {
        self.horizon
    }

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

    /// Preserve a proposal if it has a robustly safe full-horizon continuation.
    /// Otherwise choose only a certified alternative; never imply safety when
    /// no first action has a complete feasible continuation.
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

fn available_actions(shield_can_recover: bool) -> &'static [Action] {
    if shield_can_recover {
        &ALL_ACTIONS
    } else {
        &NO_RECOVERY_ACTIONS
    }
}

fn state_is_viable(state: StateEnvelope, limits: ViabilityLimits) -> bool {
    state.reserve().lower() >= limits.min_reserve()
        && state.load().upper() <= limits.max_load()
}

/// Return the next interval only if the action is safe for every state/effect in
/// the uncertainty envelope. Unsafe intermediate tubes cannot be repaired later.
fn safe_transition(
    state: StateEnvelope,
    action: Action,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
) -> Option<StateEnvelope> {
    let next = next_state_envelope(state, dynamics.effect_for_action(action))?;
    state_is_viable(next, limits).then_some(next)
}

/// Find one continuation sequence that keeps every interval-valued trajectory
/// safe for remaining_steps. This is depth-bounded exhaustive search over a
/// small discrete action set; it fails closed when no sequence exists.
fn has_safe_continuation(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    remaining_steps: usize,
    shield_can_recover: bool,
) -> bool {
    if remaining_steps == 0 {
        return true;
    }

    available_actions(shield_can_recover).iter().any(|action| {
        safe_transition(state, *action, dynamics, limits).is_some_and(|next| {
            has_safe_continuation(
                next,
                dynamics,
                limits,
                remaining_steps - 1,
                shield_can_recover,
            )
        })
    })
}

/// Evaluate first actions against a fixed horizon of robustly safe transitions.
/// Horizon one reduces to one-step viability. Invalid horizon inputs return None;
/// a valid but infeasible problem returns an empty set.
pub fn evaluate_horizon_action_set(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
) -> Option<HorizonViableActionSet> {
    if horizon == 0 || horizon > MAX_CERTIFIED_HORIZON {
        return None;
    }

    let mut set = HorizonViableActionSet {
        horizon,
        push: false,
        maintain: false,
        emergency_recover: false,
    };
    if !state_is_viable(state, limits) {
        return Some(set);
    }

    for action in available_actions(shield_can_recover) {
        let certified = safe_transition(state, *action, dynamics, limits).is_some_and(|next| {
            has_safe_continuation(
                next,
                dynamics,
                limits,
                horizon - 1,
                shield_can_recover,
            )
        });
        match action {
            Action::Push => set.push = certified,
            Action::Maintain => set.maintain = certified,
            Action::EmergencyRecover => {
                set.emergency_recover = shield_can_recover && certified
            }
        }
    }
    Some(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust_viability::{
        evaluate_action_set, ActionEffectBounds, Interval, ViabilityLimits,
    };

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid interval")
    }

    fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
        ActionEffectBounds::new(
            interval(reserve.0, reserve.1),
            interval(load.0, load.1),
        )
        .expect("valid action effect")
    }

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("valid limits")
    }

    fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
        StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid state")
    }

    /// Push is safe now but makes every second action unsafe. Maintain and
    /// recovery can continue safely for two steps, but no three-step sequence
    /// stays within the load ceiling.
    fn deadline_dynamics() -> DynamicsBounds {
        DynamicsBounds::new(
            effects((-0.005, -0.005), (0.05, 0.06)),
            effects((0.02, 0.03), (0.03, 0.04)),
            effects((0.02, 0.03), (0.02, 0.03)),
        )
    }

    fn boundary_state() -> StateEnvelope {
        state((0.17, 0.17), (0.80, 0.80))
    }

    #[test]
    fn horizon_one_matches_the_one_step_viability_set() {
        let state = boundary_state();
        let dynamics = deadline_dynamics();
        let one_step = evaluate_action_set(state, dynamics, limits(), true);
        let horizon_one =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 1).expect("valid horizon");

        assert_eq!(horizon_one.horizon(), 1);
        assert_eq!(horizon_one.count(), one_step.count());
        for action in ALL_ACTIONS {
            assert_eq!(horizon_one.contains(action), one_step.contains(action));
        }
    }

    #[test]
    fn one_step_safe_push_is_rejected_without_a_safe_continuation() {
        let state = boundary_state();
        let dynamics = deadline_dynamics();
        let one_step =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 1).expect("one-step set");
        let two_step =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 2).expect("two-step set");

        assert!(one_step.push());
        assert!(!two_step.push());
        assert!(two_step.maintain());
        assert!(two_step.emergency_recover());
        assert_eq!(two_step.filter(Action::Push), Some(Action::Maintain));
    }

    #[test]
    fn viable_first_action_sets_shrink_or_stay_equal_as_horizon_grows() {
        let state = boundary_state();
        let dynamics = deadline_dynamics();
        let sets: Vec<HorizonViableActionSet> = (1..=3)
            .map(|horizon| {
                evaluate_horizon_action_set(state, dynamics, limits(), true, horizon)
                    .expect("bounded valid horizon")
            })
            .collect();

        for adjacent in sets.windows(2) {
            for action in ALL_ACTIONS {
                assert!(
                    !adjacent[1].contains(action) || adjacent[0].contains(action),
                    "an action cannot become horizon-safe at a longer horizon: {action:?}"
                );
            }
        }
        assert_eq!(sets[2].count(), 0);
        assert_eq!(sets[2].filter(Action::Maintain), None);
    }

    #[test]
    fn emergency_recovery_requires_authority_for_the_entire_continuation() {
        let state = boundary_state();
        let dynamics = deadline_dynamics();
        let with_authority =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 2).expect("valid set");
        let without_authority =
            evaluate_horizon_action_set(state, dynamics, limits(), false, 2).expect("valid set");

        assert!(with_authority.emergency_recover());
        assert!(!without_authority.emergency_recover());
        assert_eq!(with_authority.count(), without_authority.count() + 1);
    }

    #[test]
    fn invalid_horizon_and_unsafe_initial_tube_fail_closed() {
        let initial_state = boundary_state();
        let dynamics = deadline_dynamics();
        assert!(evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 0).is_none());
        assert!(evaluate_horizon_action_set(
            initial_state,
            dynamics,
            limits(),
            true,
            MAX_CERTIFIED_HORIZON + 1,
        )
        .is_none());

        let unsafe_state = state((0.14, 0.18), (0.80, 0.82));
        let empty = evaluate_horizon_action_set(unsafe_state, dynamics, limits(), true, 2)
            .expect("well-formed state with no valid certificate");
        assert_eq!(empty.count(), 0);
        assert_eq!(empty.filter(Action::Maintain), None);
    }

    #[test]
    fn future_recovery_cannot_excuse_an_unsafe_intermediate_tube() {
        let state = boundary_state();
        let dynamics = DynamicsBounds::new(
            effects((-0.005, -0.005), (0.09, 0.10)),
            effects((0.02, 0.03), (0.03, 0.04)),
            effects((0.02, 0.03), (-0.30, -0.25)),
        );
        // Push immediately violates the load limit, while EmergencyRecover can
        // reduce load on the following step. The first action remains uncertified.
        let one_step =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 1).expect("one-step set");
        let two_step =
            evaluate_horizon_action_set(state, dynamics, limits(), true, 2).expect("two-step set");
        assert!(!one_step.push());
        assert!(!two_step.push());
    }
}
