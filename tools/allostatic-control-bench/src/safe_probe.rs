//! Safety-constrained active model discrimination.
//!
//! The module proposes an information-seeking first action only when that action
//! has a certified multi-step continuation under the rectangular hull of every
//! candidate dynamics hypothesis. This is a conservative sufficient condition:
//! it requires one common open-loop continuation to remain safe for every effect
//! in the hull. It must not be mistaken for a statistical guarantee that the
//! candidate hypothesis set contains the actual plant.

use super::multistep_viability::{evaluate_horizon_action_set, HorizonViableActionSet};
use super::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use super::Action;

const ACTIONS: [Action; 3] = [Action::Push, Action::Maintain, Action::EmergencyRecover];
const SCORE_EPSILON: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafeProbeRecommendation {
    horizon: usize,
    safe_actions: HorizonViableActionSet,
    candidate_hypotheses: usize,
    minimum_discrimination: f64,
    action_scores: [f64; 3],
    selected_action: Option<Action>,
    selected_score: f64,
}

impl SafeProbeRecommendation {
    pub const fn horizon(self) -> usize {
        self.horizon
    }

    pub const fn safe_actions(self) -> HorizonViableActionSet {
        self.safe_actions
    }

    pub const fn candidate_hypotheses(self) -> usize {
        self.candidate_hypotheses
    }

    pub const fn minimum_discrimination(self) -> f64 {
        self.minimum_discrimination
    }

    /// Scores are ordered Push, Maintain, EmergencyRecover. A score measures the
    /// minimum pairwise Euclidean gap between predicted reserve/load effect boxes.
    pub const fn action_scores(self) -> [f64; 3] {
        self.action_scores
    }

    /// None means no sufficiently discriminating action has a common robust
    /// multi-step certificate. It does not mean no ordinary safe action exists.
    pub const fn selected_action(self) -> Option<Action> {
        self.selected_action
    }

    pub const fn selected_score(self) -> f64 {
        self.selected_score
    }

    pub const fn selected_action_is_certified(self) -> bool {
        match self.selected_action {
            Some(action) => self.safe_actions.contains(action),
            None => false,
        }
    }
}

fn action_index(action: Action) -> usize {
    match action {
        Action::Push => 0,
        Action::Maintain => 1,
        Action::EmergencyRecover => 2,
    }
}

fn interval_gap(first: Interval, second: Interval) -> f64 {
    if first.upper() < second.lower() {
        second.lower() - first.upper()
    } else if second.upper() < first.lower() {
        first.lower() - second.upper()
    } else {
        0.0
    }
}

fn pairwise_effect_separation(first: ActionEffectBounds, second: ActionEffectBounds) -> f64 {
    let reserve_gap = interval_gap(first.reserve_delta(), second.reserve_delta());
    let load_gap = interval_gap(first.load_delta(), second.load_delta());
    reserve_gap.hypot(load_gap)
}

/// Return the worst-case pairwise separation between candidate effect boxes.
/// A value of zero means at least one model pair has overlapping prediction
/// boxes for this action, so that action cannot distinguish every hypothesis pair
/// under the present interval information.
fn maximin_discrimination(hypotheses: &[DynamicsBounds], action: Action) -> f64 {
    let mut minimum = f64::INFINITY;
    for first_index in 0..hypotheses.len() {
        for second_index in (first_index + 1)..hypotheses.len() {
            let first = hypotheses[first_index].effect_for_action(action);
            let second = hypotheses[second_index].effect_for_action(action);
            minimum = minimum.min(pairwise_effect_separation(first, second));
        }
    }
    if minimum.is_finite() { minimum } else { 0.0 }
}

fn hull_interval(intervals: impl Iterator<Item = Interval>) -> Option<Interval> {
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut any = false;
    for interval in intervals {
        any = true;
        lower = lower.min(interval.lower());
        upper = upper.max(interval.upper());
    }
    any.then_some(())?;
    Interval::new(lower, upper)
}

fn hull_effect(hypotheses: &[DynamicsBounds], action: Action) -> Option<ActionEffectBounds> {
    let reserve = hull_interval(
        hypotheses
            .iter()
            .map(|model| model.effect_for_action(action).reserve_delta()),
    )?;
    let load = hull_interval(
        hypotheses
            .iter()
            .map(|model| model.effect_for_action(action).load_delta()),
    )?;
    ActionEffectBounds::new(reserve, load)
}

/// Build a rectangular transition hull containing every supplied candidate model.
/// Rectangularization can add combinations not present in any candidate model,
/// making this safe-set computation conservative but straightforward to audit.
pub fn dynamics_hull(hypotheses: &[DynamicsBounds]) -> Option<DynamicsBounds> {
    if hypotheses.is_empty() {
        return None;
    }
    Some(DynamicsBounds::new(
        hull_effect(hypotheses, Action::Push)?,
        hull_effect(hypotheses, Action::Maintain)?,
        hull_effect(hypotheses, Action::EmergencyRecover)?,
    ))
}

/// Propose an informative action subject to a common multi-step robust certificate.
///
/// Safety is evaluated against the hull of all model hypotheses—not separately
/// per model—so the returned action must have a safe continuation under a single
/// conservative envelope. Information score is the minimum pairwise separation
/// between interval predictions. The selected action must exceed the predeclared
/// threshold; otherwise the result reports no probe and leaves ordinary safe
/// task-action selection to the caller.
///
/// Returns None for malformed input or exhausted horizon search. A None result
/// must never be interpreted as permission to bypass the existing safety filter.
pub fn recommend_safe_probe(
    state: StateEnvelope,
    hypotheses: &[DynamicsBounds],
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
    preferred_action: Action,
    minimum_discrimination: f64,
) -> Option<SafeProbeRecommendation> {
    if hypotheses.len() < 2
        || !minimum_discrimination.is_finite()
        || minimum_discrimination < 0.0
    {
        return None;
    }

    let hull = dynamics_hull(hypotheses)?;
    let safe_actions =
        evaluate_horizon_action_set(state, hull, limits, shield_can_recover, horizon)?;
    let action_scores = ACTIONS.map(|action| maximin_discrimination(hypotheses, action));

    let mut selected_action = None;
    let mut selected_score = 0.0_f64;
    for action in ACTIONS {
        let score = action_scores[action_index(action)];
        if !safe_actions.contains(action) || score <= minimum_discrimination {
            continue;
        }
        match selected_action {
            None => {
                selected_action = Some(action);
                selected_score = score;
            }
            Some(current) => {
                if score > selected_score + SCORE_EPSILON
                    || ((score - selected_score).abs() <= SCORE_EPSILON
                        && action == preferred_action
                        && current != preferred_action)
                {
                    selected_action = Some(action);
                    selected_score = score;
                }
            }
        }
    }

    Some(SafeProbeRecommendation {
        horizon,
        safe_actions,
        candidate_hypotheses: hypotheses.len(),
        minimum_discrimination,
        action_scores,
        selected_action,
        selected_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistep_viability::evaluate_horizon_action_set;
    use crate::robust_viability::evaluate_action_set;

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid interval")
    }

    fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
        ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid effect bounds")
    }

    fn model(
        push: ActionEffectBounds,
        maintain: ActionEffectBounds,
        recovery: ActionEffectBounds,
    ) -> DynamicsBounds {
        DynamicsBounds::new(push, maintain, recovery)
    }

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("valid limits")
    }

    fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
        StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid state")
    }

    fn informative_models() -> [DynamicsBounds; 2] {
        [
            model(
                effects((-0.02, -0.01), (0.01, 0.02)),
                effects((0.02, 0.03), (-0.02, -0.01)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
            model(
                effects((-0.09, -0.08), (0.09, 0.10)),
                effects((0.02, 0.03), (-0.02, -0.01)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
        ]
    }

    #[test]
    fn safe_informative_action_is_selected_when_all_hypotheses_support_it() {
        let models = informative_models();
        let current = state((0.50, 0.50), (0.50, 0.50));
        let recommendation = recommend_safe_probe(
            current,
            &models,
            limits(),
            true,
            3,
            Action::Maintain,
            0.02,
        )
        .expect("valid active-probe request");

        assert!(recommendation.safe_actions().push());
        assert_eq!(recommendation.selected_action(), Some(Action::Push));
        assert!(recommendation.selected_action_is_certified());
        assert!(recommendation.selected_score() > 0.02);
        assert_eq!(recommendation.candidate_hypotheses(), 2);
    }

    #[test]
    fn unsafe_high_information_action_is_never_selected() {
        let models = informative_models();
        let current = state((0.18, 0.18), (0.80, 0.80));
        let recommendation = recommend_safe_probe(
            current,
            &models,
            limits(),
            true,
            3,
            Action::Maintain,
            0.02,
        )
        .expect("valid active-probe request");

        assert!(!recommendation.safe_actions().push());
        assert!(recommendation.safe_actions().maintain());
        assert_ne!(recommendation.selected_action(), Some(Action::Push));
        assert_eq!(recommendation.selected_action(), None);
        assert!(!recommendation.selected_action_is_certified());
    }

    #[test]
    fn no_model_separation_returns_no_probe_without_claiming_no_safe_action() {
        let identical = model(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        );
        let models = [identical, identical];
        let recommendation = recommend_safe_probe(
            state((0.50, 0.50), (0.50, 0.50)),
            &models,
            limits(),
            true,
            2,
            Action::Maintain,
            0.0,
        )
        .expect("valid but uninformative request");

        assert!(recommendation.safe_actions().count() > 0);
        assert_eq!(recommendation.selected_action(), None);
        assert!(!recommendation.selected_action_is_certified());
    }

    #[test]
    fn horizon_filter_rejects_one_step_safe_but_dead_end_probe() {
        let current = state((0.16, 0.16), (0.87, 0.87));
        let dangerous_continuation = model(
            effects((-0.005, -0.005), (0.01, 0.01)),
            effects((0.0, 0.0), (0.005, 0.005)),
            effects((-0.02, -0.02), (-0.30, -0.25)),
        );
        let alternate = model(
            effects((-0.006, -0.006), (0.01, 0.01)),
            effects((0.0, 0.0), (0.005, 0.005)),
            effects((-0.02, -0.02), (-0.30, -0.25)),
        );
        let models = [dangerous_continuation, alternate];
        let hull = dynamics_hull(&models).expect("valid hull");
        let one_step = evaluate_action_set(current, hull, limits(), true);
        let horizon_two =
            evaluate_horizon_action_set(current, hull, limits(), true, 2).expect("valid search");

        assert!(one_step.push());
        assert!(!horizon_two.push());
        let recommendation = recommend_safe_probe(
            current,
            &models,
            limits(),
            true,
            2,
            Action::Push,
            0.0,
        )
        .expect("valid active-probe request");
        assert_ne!(recommendation.selected_action(), Some(Action::Push));
    }

    #[test]
    fn model_hull_contains_each_hypothesis_effect_box() {
        let models = informative_models();
        let hull = dynamics_hull(&models).expect("valid model hull");
        for action in ACTIONS {
            let combined = hull.effect_for_action(action);
            for candidate in models {
                let effect = candidate.effect_for_action(action);
                assert!(combined.reserve_delta().lower() <= effect.reserve_delta().lower());
                assert!(combined.reserve_delta().upper() >= effect.reserve_delta().upper());
                assert!(combined.load_delta().lower() <= effect.load_delta().lower());
                assert!(combined.load_delta().upper() >= effect.load_delta().upper());
            }
        }
    }

    #[test]
    fn insufficient_models_and_invalid_threshold_fail_closed() {
        let models = informative_models();
        assert!(recommend_safe_probe(
            state((0.50, 0.50), (0.50, 0.50)),
            &models[..1],
            limits(),
            true,
            2,
            Action::Maintain,
            0.02,
        )
        .is_none());
        assert!(recommend_safe_probe(
            state((0.50, 0.50), (0.50, 0.50)),
            &models,
            limits(),
            true,
            2,
            Action::Maintain,
            f64::NAN,
        )
        .is_none());
    }
}
