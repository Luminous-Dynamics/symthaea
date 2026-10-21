//! Finite-horizon robust viability using conservative interval tubes.
//!
//! A first action is certified only if one open-loop continuation sequence keeps
//! the complete reachable interval within the limits at every intermediate step.
//! This is sound under the declared rectangular action-effect bounds, but
//! conservative: it does not search state-contingent continuation policies.

use super::robust_viability::{next_state_envelope, DynamicsBounds, StateEnvelope, ViabilityLimits};
use super::Action;

pub const MAX_CERTIFIED_HORIZON: usize = 8;
/// Search budget exceeds the exhaustive three-action tree through horizon 8.
pub const MAX_EXPANDED_SEARCH_NODES: usize = 10_000;

const ALL_ACTIONS: [Action; 3] = [Action::Push, Action::Maintain, Action::EmergencyRecover];
const NO_RECOVERY_ACTIONS: [Action; 2] = [Action::Push, Action::Maintain];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HorizonSearchStats {
    pub expanded_nodes: usize,
    pub transition_evaluations: usize,
    pub max_depth_reached: usize,
}

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

    /// Preserve a proposal when it has a full-horizon certificate. Otherwise
    /// return only a certified alternative; None means no action is certified.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HorizonEvaluation {
    pub action_set: HorizonViableActionSet,
    pub stats: HorizonSearchStats,
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

/// Return the next interval only if every state/effect in the uncertainty set is
/// safe. An unsafe intermediate interval cannot be repaired retroactively.
fn safe_transition(
    state: StateEnvelope,
    action: Action,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
) -> Option<StateEnvelope> {
    let next = next_state_envelope(state, dynamics.effect_for_action(action))?;
    state_is_viable(next, limits).then_some(next)
}

/// Find one continuation sequence that keeps all represented trajectories safe.
/// A resource-budget exhaustion returns None so callers cannot certify a partial
/// search result as safe.
fn has_safe_continuation(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    remaining_steps: usize,
    depth: usize,
    shield_can_recover: bool,
    max_expanded_nodes: usize,
    stats: &mut HorizonSearchStats,
) -> Option<bool> {
    stats.expanded_nodes += 1;
    stats.max_depth_reached = stats.max_depth_reached.max(depth);
    if stats.expanded_nodes > max_expanded_nodes {
        return None;
    }
    if remaining_steps == 0 {
        return Some(true);
    }

    for action in available_actions(shield_can_recover) {
        stats.transition_evaluations += 1;
        if let Some(next) = safe_transition(state, *action, dynamics, limits) {
            if has_safe_continuation(
                next,
                dynamics,
                limits,
                remaining_steps - 1,
                depth + 1,
                shield_can_recover,
                max_expanded_nodes,
                stats,
            )? {
                return Some(true);
            }
        }
    }
    Some(false)
}

/// Evaluate the first action against a fixed horizon. Horizon one is identical
/// to one-step viability. Invalid horizons or exhausted search budgets return
/// None; a valid but infeasible state returns an empty set.
pub fn evaluate_horizon_action_set(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
) -> Option<HorizonViableActionSet> {
    evaluate_horizon_action_set_with_budget(
        state,
        dynamics,
        limits,
        shield_can_recover,
        horizon,
        MAX_EXPANDED_SEARCH_NODES,
    )
    .map(|evaluation| evaluation.action_set)
}

/// As above, with explicit node accounting. The budget counts recursive search
/// nodes; if it is reached before proving a candidate sequence, evaluation fails
/// closed rather than returning an incomplete certificate.
pub fn evaluate_horizon_action_set_with_budget(
    state: StateEnvelope,
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
    max_expanded_nodes: usize,
) -> Option<HorizonEvaluation> {
    if horizon == 0 || horizon > MAX_CERTIFIED_HORIZON || max_expanded_nodes == 0 {
        return None;
    }

    let mut stats = HorizonSearchStats::default();
    let mut set = HorizonViableActionSet {
        horizon,
        push: false,
        maintain: false,
        emergency_recover: false,
    };
    if !state_is_viable(state, limits) {
        return Some(HorizonEvaluation {
            action_set: set,
            stats,
        });
    }

    for action in available_actions(shield_can_recover) {
        stats.transition_evaluations += 1;
        let certified = if let Some(next) = safe_transition(state, *action, dynamics, limits) {
            has_safe_continuation(
                next,
                dynamics,
                limits,
                horizon - 1,
                1,
                shield_can_recover,
                max_expanded_nodes,
                &mut stats,
            )?
        } else {
            false
        };
        match action {
            Action::Push => set.push = certified,
            Action::Maintain => set.maintain = certified,
            Action::EmergencyRecover => {
                set.emergency_recover = shield_can_recover && certified
            }
        }
    }

    Some(HorizonEvaluation {
        action_set: set,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust_viability::{evaluate_action_set, ActionEffectBounds, Interval, ViabilityLimits};

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
        let initial_state = boundary_state();
        let dynamics = deadline_dynamics();
        let one_step = evaluate_action_set(initial_state, dynamics, limits(), true);
        let horizon_one =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 1)
                .expect("valid horizon");

        assert_eq!(horizon_one.horizon(), 1);
        assert_eq!(horizon_one.count(), one_step.count());
        for action in ALL_ACTIONS {
            assert_eq!(horizon_one.contains(action), one_step.contains(action));
        }
    }

    #[test]
    fn one_step_safe_push_is_rejected_without_a_safe_continuation() {
        let initial_state = boundary_state();
        let dynamics = deadline_dynamics();
        let one_step =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 1)
                .expect("one-step set");
        let two_step =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 2)
                .expect("two-step set");

        assert!(one_step.push());
        assert!(!two_step.push());
        assert!(two_step.maintain());
        assert!(two_step.emergency_recover());
        assert_eq!(two_step.filter(Action::Push), Some(Action::Maintain));
    }

    #[test]
    fn viable_first_action_sets_shrink_or_stay_equal_as_horizon_grows() {
        let initial_state = boundary_state();
        let dynamics = deadline_dynamics();
        let sets: Vec<HorizonViableActionSet> = (1..=3)
            .map(|horizon| {
                evaluate_horizon_action_set(initial_state, dynamics, limits(), true, horizon)
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
        let initial_state = boundary_state();
        let dynamics = deadline_dynamics();
        let with_authority =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 2)
                .expect("valid set");
        let without_authority =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), false, 2)
                .expect("valid set");

        assert!(with_authority.emergency_recover());
        assert!(!without_authority.emergency_recover());
        assert_eq!(with_authority.count(), without_authority.count() + 1);
    }

    #[test]
    fn invalid_horizon_budget_and_unsafe_initial_tube_fail_closed() {
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
        assert!(evaluate_horizon_action_set_with_budget(
            initial_state,
            dynamics,
            limits(),
            true,
            MAX_CERTIFIED_HORIZON,
            1,
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
        let initial_state = boundary_state();
        let dynamics = DynamicsBounds::new(
            effects((-0.005, -0.005), (0.09, 0.10)),
            effects((0.02, 0.03), (0.03, 0.04)),
            effects((0.02, 0.03), (-0.30, -0.25)),
        );
        let unsafe_intermediate = next_state_envelope(
            initial_state,
            dynamics.effect_for_action(Action::Push),
        )
        .expect("well-formed but unsafe intermediate tube");
        assert!(!state_is_viable(unsafe_intermediate, limits()));
        let recovered = next_state_envelope(
            unsafe_intermediate,
            dynamics.effect_for_action(Action::EmergencyRecover),
        )
        .expect("recovery transition");
        assert!(state_is_viable(recovered, limits()));

        let one_step =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 1)
                .expect("one-step set");
        let two_step =
            evaluate_horizon_action_set(initial_state, dynamics, limits(), true, 2)
                .expect("two-step set");
        assert!(!one_step.push());
        assert!(!two_step.push());
    }
}
