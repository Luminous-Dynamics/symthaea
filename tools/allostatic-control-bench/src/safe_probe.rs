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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeSelectionMode {
    /// Require the minimum noise-adjusted pairwise gap to exceed a threshold.
    FullPairwiseSeparation,
    /// Prefer an action that guarantees elimination of at least one hypothesis;
    /// rank first by worst-case elimination count, then by geometric separation.
    GuaranteedWorstCaseReduction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafeProbeRecommendation {
    horizon: usize,
    safe_actions: HorizonViableActionSet,
    candidate_hypotheses: usize,
    selection_mode: ProbeSelectionMode,
    minimum_discrimination: f64,
    measurement_radius: f64,
    action_scores: [f64; 3],
    worst_case_remaining_hypotheses: [usize; 3],
    guaranteed_hypothesis_eliminations: [usize; 3],
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

    pub const fn selection_mode(self) -> ProbeSelectionMode {
        self.selection_mode
    }

    pub const fn minimum_discrimination(self) -> f64 {
        self.minimum_discrimination
    }

    /// Per-coordinate bound on observation error. Pairwise score gaps are
    /// reduced by twice this radius before a probe can claim distinguishability.
    pub const fn measurement_radius(self) -> f64 {
        self.measurement_radius
    }

    /// True only when every pair of candidate effect boxes remains disjoint
    /// after expansion by the declared measurement uncertainty.
    pub const fn selected_action_guarantees_pairwise_distinguishability(self) -> bool {
        self.selected_action.is_some() && self.selected_score > 0.0
    }

    /// Maximum number of hypotheses that may remain after any observation for
    /// each action, assuming the declared prediction and measurement boxes hold.
    /// Scores are ordered Push, Maintain, EmergencyRecover.
    pub const fn worst_case_remaining_hypotheses(self) -> [usize; 3] {
        self.worst_case_remaining_hypotheses
    }

    /// Minimum number of hypotheses guaranteed to be eliminated by each action.
    pub const fn guaranteed_hypothesis_eliminations(self) -> [usize; 3] {
        self.guaranteed_hypothesis_eliminations
    }

    /// True when every observation compatible with the selected action is
    /// compatible with fewer than the original number of candidate hypotheses.
    pub fn selected_action_guarantees_hypothesis_reduction(self) -> bool {
        self.selected_action.is_some_and(|action| {
            self.guaranteed_hypothesis_eliminations[action_index(action)] > 0
        })
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

fn pairwise_effect_separation(
    first: ActionEffectBounds,
    second: ActionEffectBounds,
    measurement_radius: f64,
) -> f64 {
    // Expanding each prediction interval by +/- radius makes their separation
    // shrink by 2*radius. Clamp overlapping noisy observation sets to zero.
    let reserve_gap =
        (interval_gap(first.reserve_delta(), second.reserve_delta()) - 2.0 * measurement_radius)
            .max(0.0);
    let load_gap =
        (interval_gap(first.load_delta(), second.load_delta()) - 2.0 * measurement_radius)
            .max(0.0);
    reserve_gap.hypot(load_gap)
}

/// Maximum number of candidate observation sets that can contain the same
/// measured transition. This is the worst-case posterior cardinality under
/// interval/set-membership updating with the declared bounded observation noise.
fn worst_case_remaining_hypotheses(
    hypotheses: &[DynamicsBounds],
    action: Action,
    measurement_radius: f64,
) -> usize {
    if hypotheses.is_empty() {
        return 0;
    }
    // Effect coordinates are bounded by [-1, 1]. With radius >= 2 every
    // candidate observation box necessarily overlaps every other one.
    if measurement_radius >= 2.0 {
        return hypotheses.len();
    }

    let mut boxes = Vec::with_capacity(hypotheses.len());
    let mut x_edges = Vec::with_capacity(hypotheses.len() * 2);
    let mut y_edges = Vec::with_capacity(hypotheses.len() * 2);
    for hypothesis in hypotheses {
        let effect = hypothesis.effect_for_action(action);
        let reserve = effect.reserve_delta();
        let load = effect.load_delta();
        let x_lower = reserve.lower() - measurement_radius;
        let x_upper = reserve.upper() + measurement_radius;
        let y_lower = load.lower() - measurement_radius;
        let y_upper = load.upper() + measurement_radius;
        boxes.push((x_lower, x_upper, y_lower, y_upper));
        x_edges.push(x_lower);
        x_edges.push(x_upper);
        y_edges.push(y_lower);
        y_edges.push(y_upper);
    }

    // For axis-aligned closed rectangles, a maximum overlap depth is attained
    // at the product of rectangle-edge coordinates; evaluating those points
    // includes both positive-area overlaps and boundary-only intersections.
    let mut maximum = 0;
    for x in &x_edges {
        for y in &y_edges {
            let overlap = boxes
                .iter()
                .filter(|(x_lower, x_upper, y_lower, y_upper)| {
                    *x >= *x_lower && *x <= *x_upper && *y >= *y_lower && *y <= *y_upper
                })
                .count();
            maximum = maximum.max(overlap);
        }
    }
    maximum
}

fn maximin_discrimination(
    hypotheses: &[DynamicsBounds],
    action: Action,
    measurement_radius: f64,
) -> f64 {
    let mut minimum = f64::INFINITY;
    for first_index in 0..hypotheses.len() {
        for second_index in (first_index + 1)..hypotheses.len() {
            let first = hypotheses[first_index].effect_for_action(action);
            let second = hypotheses[second_index].effect_for_action(action);
            minimum = minimum.min(pairwise_effect_separation(first, second, measurement_radius));
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

/// One noisy observation of the reserve/load transition effects for a chosen action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedEffect {
    reserve_delta: f64,
    load_delta: f64,
    measurement_radius: f64,
}

impl ObservedEffect {
    pub fn new(reserve_delta: f64, load_delta: f64, measurement_radius: f64) -> Option<Self> {
        (reserve_delta.is_finite()
            && load_delta.is_finite()
            && measurement_radius.is_finite()
            && measurement_radius >= 0.0)
        .then_some(Self {
            reserve_delta,
            load_delta,
            measurement_radius,
        })
    }

    pub const fn reserve_delta(self) -> f64 {
        self.reserve_delta
    }

    pub const fn load_delta(self) -> f64 {
        self.load_delta
    }

    pub const fn measurement_radius(self) -> f64 {
        self.measurement_radius
    }
}

fn interval_intersects_observation(interval: Interval, observed: f64, radius: f64) -> bool {
    interval.upper() >= observed - radius && interval.lower() <= observed + radius
}

/// Retain only hypotheses whose predicted effect box intersects the measured
/// transition box. Some(empty) is meaningful: the observation falsified every
/// candidate model. Callers must then invalidate the prior certificate and fail
/// closed; they must not silently restore the original model family.
pub fn filter_compatible_hypotheses(
    hypotheses: &[DynamicsBounds],
    action: Action,
    observation: ObservedEffect,
) -> Option<Vec<DynamicsBounds>> {
    if hypotheses.is_empty() {
        return None;
    }

    let retained = hypotheses
        .iter()
        .copied()
        .filter(|model| {
            let effect = model.effect_for_action(action);
            interval_intersects_observation(
                effect.reserve_delta(),
                observation.reserve_delta,
                observation.measurement_radius,
            ) && interval_intersects_observation(
                effect.load_delta(),
                observation.load_delta,
                observation.measurement_radius,
            )
        })
        .collect();
    Some(retained)
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

/// Recommend a fully pairwise-discriminating action subject to a common
/// multi-step robust certificate. This legacy wrapper assumes zero measurement
/// error; use `recommend_safe_probe_with_noise` when a noise bound is known.
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
    recommend_safe_probe_with_noise(
        state,
        hypotheses,
        limits,
        shield_can_recover,
        horizon,
        preferred_action,
        minimum_discrimination,
        0.0,
    )
}

/// Recommend a fully discriminating action, discounted by a declared
/// per-coordinate observation error bound. The selected action's minimum pairwise
/// observation-set gap must exceed `minimum_discrimination`.
#[allow(clippy::too_many_arguments)]
pub fn recommend_safe_probe_with_noise(
    state: StateEnvelope,
    hypotheses: &[DynamicsBounds],
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
    preferred_action: Action,
    minimum_discrimination: f64,
    measurement_radius: f64,
) -> Option<SafeProbeRecommendation> {
    recommend_probe_internal(
        state,
        hypotheses,
        limits,
        shield_can_recover,
        horizon,
        preferred_action,
        minimum_discrimination,
        measurement_radius,
        ProbeSelectionMode::FullPairwiseSeparation,
    )
}

/// Choose a safe probe that guarantees reducing the finite candidate family even
/// when complete identification is impossible. Ranks certified actions first by
/// worst-case number of hypotheses eliminated under the bounded observation
/// model, then by noise-adjusted pairwise separation. This is set-membership
/// reasoning over a finite hypothesis list, not A-SLS or a continuous parameter
/// estimator.
#[allow(clippy::too_many_arguments)]
pub fn recommend_safe_probe_for_guaranteed_reduction(
    state: StateEnvelope,
    hypotheses: &[DynamicsBounds],
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
    preferred_action: Action,
    measurement_radius: f64,
) -> Option<SafeProbeRecommendation> {
    recommend_probe_internal(
        state,
        hypotheses,
        limits,
        shield_can_recover,
        horizon,
        preferred_action,
        0.0,
        measurement_radius,
        ProbeSelectionMode::GuaranteedWorstCaseReduction,
    )
}

#[allow(clippy::too_many_arguments)]
fn recommend_probe_internal(
    state: StateEnvelope,
    hypotheses: &[DynamicsBounds],
    limits: ViabilityLimits,
    shield_can_recover: bool,
    horizon: usize,
    preferred_action: Action,
    minimum_discrimination: f64,
    measurement_radius: f64,
    selection_mode: ProbeSelectionMode,
) -> Option<SafeProbeRecommendation> {
    if hypotheses.len() < 2
        || !minimum_discrimination.is_finite()
        || minimum_discrimination < 0.0
        || !measurement_radius.is_finite()
        || measurement_radius < 0.0
    {
        return None;
    }

    let hull = dynamics_hull(hypotheses)?;
    let safe_actions =
        evaluate_horizon_action_set(state, hull, limits, shield_can_recover, horizon)?;
    let action_scores =
        ACTIONS.map(|action| maximin_discrimination(hypotheses, action, measurement_radius));
    let worst_case_remaining_hypotheses =
        ACTIONS.map(|action| self::worst_case_remaining_hypotheses(hypotheses, action, measurement_radius));
    let guaranteed_hypothesis_eliminations = worst_case_remaining_hypotheses
        .map(|remaining| hypotheses.len().saturating_sub(remaining));

    let mut selected_action = None;
    let mut selected_score = 0.0_f64;
    let mut selected_eliminations = 0_usize;
    for action in ACTIONS {
        let index = action_index(action);
        let score = action_scores[index];
        let eliminations = guaranteed_hypothesis_eliminations[index];
        if !safe_actions.contains(action) {
            continue;
        }
        let eligible = match selection_mode {
            ProbeSelectionMode::FullPairwiseSeparation => score > minimum_discrimination,
            ProbeSelectionMode::GuaranteedWorstCaseReduction => eliminations > 0,
        };
        if !eligible {
            continue;
        }
        match selected_action {
            None => {
                selected_action = Some(action);
                selected_score = score;
                selected_eliminations = eliminations;
            }
            Some(current) => {
                let prefer_current_mode = match selection_mode {
                    ProbeSelectionMode::FullPairwiseSeparation => {
                        score > selected_score + SCORE_EPSILON
                            || ((score - selected_score).abs() <= SCORE_EPSILON
                                && action == preferred_action
                                && current != preferred_action)
                    }
                    ProbeSelectionMode::GuaranteedWorstCaseReduction => {
                        eliminations > selected_eliminations
                            || (eliminations == selected_eliminations
                                && score > selected_score + SCORE_EPSILON)
                            || (eliminations == selected_eliminations
                                && (score - selected_score).abs() <= SCORE_EPSILON
                                && action == preferred_action
                                && current != preferred_action)
                    }
                };
                if prefer_current_mode {
                    selected_action = Some(action);
                    selected_score = score;
                    selected_eliminations = eliminations;
                }
            }
        }
    }

    Some(SafeProbeRecommendation {
        horizon,
        safe_actions,
        candidate_hypotheses: hypotheses.len(),
        selection_mode,
        minimum_discrimination,
        measurement_radius,
        action_scores,
        worst_case_remaining_hypotheses,
        guaranteed_hypothesis_eliminations,
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
    fn observation_noise_can_remove_apparent_discriminability() {
        let models = informative_models();
        let current = state((0.50, 0.50), (0.50, 0.50));
        let low_noise = recommend_safe_probe_with_noise(
            current, &models, limits(), true, 3, Action::Maintain, 0.02, 0.005,
        ).expect("well-formed low-noise request");
        assert_eq!(low_noise.selected_action(), Some(Action::Push));
        assert!(low_noise.selected_action_guarantees_pairwise_distinguishability());

        let high_noise = recommend_safe_probe_with_noise(
            current, &models, limits(), true, 3, Action::Maintain, 0.0, 0.04,
        ).expect("well-formed high-noise request");
        assert!(high_noise.safe_actions().count() > 0);
        assert_eq!(high_noise.selected_action(), None);
        assert!(!high_noise.selected_action_guarantees_pairwise_distinguishability());
        assert_eq!(high_noise.measurement_radius(), 0.04);
    }

    #[test]
    fn invalid_measurement_noise_fails_closed() {
        let models = informative_models();
        assert!(recommend_safe_probe_with_noise(
            state((0.50, 0.50), (0.50, 0.50)), &models, limits(), true, 3,
            Action::Maintain, 0.0, f64::NAN,
        ).is_none());
        assert!(recommend_safe_probe_with_noise(
            state((0.50, 0.50), (0.50, 0.50)), &models, limits(), true, 3,
            Action::Maintain, 0.0, -0.1,
        ).is_none());
    }

    #[test]
    fn probe_can_guarantee_partial_learning_when_full_identification_is_impossible() {
        let shared = model(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        );
        let distinct_push = model(
            effects((-0.09, -0.08), (0.09, 0.10)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        );
        let models = [shared, shared, distinct_push];
        let recommendation = recommend_safe_probe_for_guaranteed_reduction(
            state((0.50, 0.50), (0.50, 0.50)),
            &models,
            limits(),
            true,
            3,
            Action::Maintain,
            0.005,
        )
        .expect("valid partial-learning request");

        assert_eq!(recommendation.selected_action(), Some(Action::Push));
        assert!(recommendation.selected_action_is_certified());
        assert!(!recommendation.selected_action_guarantees_pairwise_distinguishability());
        assert!(recommendation.selected_action_guarantees_hypothesis_reduction());
        assert_eq!(recommendation.worst_case_remaining_hypotheses()[0], 2);
        assert_eq!(recommendation.guaranteed_hypothesis_eliminations()[0], 1);
        assert_eq!(recommendation.worst_case_remaining_hypotheses()[1], 3);
        assert_eq!(recommendation.guaranteed_hypothesis_eliminations()[1], 0);
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
    fn observed_safe_probe_can_eliminate_a_model_and_expand_viability() {
        let models = [
            model(
                effects((-0.02, -0.02), (0.02, 0.02)),
                effects((0.02, 0.02), (-0.02, -0.02)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
            model(
                effects((-0.08, -0.08), (0.10, 0.10)),
                effects((0.04, 0.04), (-0.06, -0.06)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
        ];
        let current = state((0.30, 0.30), (0.80, 0.80));
        let hull = dynamics_hull(&models).expect("valid model hull");
        let before = evaluate_horizon_action_set(current, hull, limits(), true, 3)
            .expect("valid pre-probe certificate");
        assert!(!before.push());

        let recommendation = recommend_safe_probe(
            current,
            &models,
            limits(),
            true,
            3,
            Action::Maintain,
            0.01,
        )
        .expect("valid probe request");
        assert_eq!(recommendation.selected_action(), Some(Action::Maintain));
        assert!(recommendation.selected_action_is_certified());

        let observation =
            ObservedEffect::new(0.02, -0.02, 0.001).expect("valid measured transition");
        let posterior = filter_compatible_hypotheses(
            &models,
            Action::Maintain,
            observation,
        )
        .expect("nonempty prior");
        assert_eq!(posterior.len(), 1);
        assert_eq!(posterior[0], models[0]);

        let posterior_hull = dynamics_hull(&posterior).expect("posterior still nonempty");
        let after = evaluate_horizon_action_set(current, posterior_hull, limits(), true, 3)
            .expect("valid post-probe certificate");
        assert!(after.push());
        assert!(after.count() > before.count());
    }

    #[test]
    fn observation_that_falsifies_every_candidate_does_not_restore_the_prior() {
        let models = informative_models();
        let observation =
            ObservedEffect::new(0.99, 0.99, 0.001).expect("finite but out-of-family observation");
        let retained = filter_compatible_hypotheses(
            &models,
            Action::Maintain,
            observation,
        )
        .expect("valid nonempty prior");
        assert!(retained.is_empty());
        assert!(dynamics_hull(&retained).is_none());
    }

    #[test]
    fn ambiguous_observation_keeps_both_overlapping_hypotheses() {
        let models = [
            model(
                effects((-0.02, -0.02), (0.02, 0.02)),
                effects((0.02, 0.02), (-0.02, -0.02)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
            model(
                effects((-0.08, -0.08), (0.10, 0.10)),
                effects((0.04, 0.04), (-0.06, -0.06)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
        ];
        let observation =
            ObservedEffect::new(0.03, -0.04, 0.021).expect("valid ambiguous observation");
        let retained = filter_compatible_hypotheses(
            &models,
            Action::Maintain,
            observation,
        )
        .expect("valid nonempty prior");
        assert_eq!(retained.len(), 2);
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
