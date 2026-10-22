//! Robust viability under correlated joint uncertainty.
//!
//! This module contrasts an axis-aligned box outer approximation with the exact
//! support function of a zonotope for a coupled linear viability constraint.
//! For purely coordinate-wise box constraints, correlation alone cannot improve
//! the marginal extrema; the coupled constraint is deliberately explicit here.
//! It represents a safety budget such as reserve minus a weighted load term.
//!
//! This is an isolated synthetic research layer. It does not change the qualified
//! allostatic-control path or claim to reproduce conformalized SLS-MPC.

use super::Action;

pub const MAX_ZONOTOPE_GENERATORS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatePoint {
    reserve: f64,
    load: f64,
}

impl StatePoint {
    pub fn new(reserve: f64, load: f64) -> Option<Self> {
        (reserve.is_finite()
            && load.is_finite()
            && (0.0..=1.0).contains(&reserve)
            && (0.0..=1.0).contains(&load))
        .then_some(Self { reserve, load })
    }

    pub const fn reserve(self) -> f64 {
        self.reserve
    }

    pub const fn load(self) -> f64 {
        self.load
    }

    pub fn is_viable(self, limits: JointViabilityLimits) -> bool {
        self.reserve >= limits.min_reserve
            && self.load <= limits.max_load
            && self.reserve - limits.load_weight * self.load >= limits.min_joint_margin
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Delta2D {
    reserve: f64,
    load: f64,
}

impl Delta2D {
    pub fn new(reserve: f64, load: f64) -> Option<Self> {
        (reserve.is_finite() && load.is_finite()).then_some(Self { reserve, load })
    }

    pub const fn reserve(self) -> f64 {
        self.reserve
    }

    pub const fn load(self) -> f64 {
        self.load
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionEffectZonotope {
    center: Delta2D,
    generators: Vec<Delta2D>,
}

impl ActionEffectZonotope {
    pub fn new(center: Delta2D, generators: Vec<Delta2D>) -> Option<Self> {
        if generators.len() > MAX_ZONOTOPE_GENERATORS {
            return None;
        }
        Some(Self { center, generators })
    }

    pub const fn center(&self) -> Delta2D {
        self.center
    }

    pub fn generators(&self) -> &[Delta2D] {
        &self.generators
    }

    pub fn corner_count(&self) -> usize {
        1_usize << self.generators.len()
    }

    fn coordinate_radii(&self) -> (f64, f64) {
        self.generators.iter().fold((0.0, 0.0), |(reserve, load), g| {
            (reserve + g.reserve.abs(), load + g.load.abs())
        })
    }

    fn joint_radius(&self, load_weight: f64) -> f64 {
        self.generators
            .iter()
            .map(|g| (g.reserve - load_weight * g.load).abs())
            .sum()
    }

    fn box_joint_radius(&self, load_weight: f64) -> f64 {
        let (reserve_radius, load_radius) = self.coordinate_radii();
        reserve_radius + load_weight * load_radius
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointViabilityLimits {
    min_reserve: f64,
    max_load: f64,
    load_weight: f64,
    min_joint_margin: f64,
}

impl JointViabilityLimits {
    pub fn new(
        min_reserve: f64,
        max_load: f64,
        load_weight: f64,
        min_joint_margin: f64,
    ) -> Option<Self> {
        (min_reserve.is_finite()
            && max_load.is_finite()
            && load_weight.is_finite()
            && min_joint_margin.is_finite()
            && (0.0..=1.0).contains(&min_reserve)
            && (0.0..=1.0).contains(&max_load)
            && load_weight >= 0.0
            && load_weight <= 2.0
            && (-2.0..=1.0).contains(&min_joint_margin))
        .then_some(Self {
            min_reserve,
            max_load,
            load_weight,
            min_joint_margin,
        })
    }

    pub const fn min_reserve(self) -> f64 {
        self.min_reserve
    }

    pub const fn max_load(self) -> f64 {
        self.max_load
    }

    pub const fn load_weight(self) -> f64 {
        self.load_weight
    }

    pub const fn min_joint_margin(self) -> f64 {
        self.min_joint_margin
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViabilityMargins {
    min_reserve: f64,
    max_reserve: f64,
    min_load: f64,
    max_load: f64,
    joint_margin: f64,
}

impl ViabilityMargins {
    pub const fn min_reserve(self) -> f64 {
        self.min_reserve
    }

    pub const fn max_reserve(self) -> f64 {
        self.max_reserve
    }

    pub const fn min_load(self) -> f64 {
        self.min_load
    }

    pub const fn max_load(self) -> f64 {
        self.max_load
    }

    pub const fn joint_margin(self) -> f64 {
        self.joint_margin
    }

    pub fn minimum(self) -> f64 {
        self.min_reserve
            .min(self.max_reserve)
            .min(self.min_load)
            .min(self.max_load)
            .min(self.joint_margin)
    }

    pub fn is_safe(self) -> bool {
        self.minimum() >= 0.0
    }
}

fn next_center(state: StatePoint, effects: &ActionEffectZonotope) -> StatePoint {
    StatePoint {
        reserve: state.reserve + effects.center.reserve,
        load: state.load + effects.center.load,
    }
}

/// Margins when the zonotope is replaced with its axis-aligned bounding box.
/// This is a sound outer approximation, but loses correlation for the joint rule.
pub fn box_viability_margins(
    state: StatePoint,
    effects: &ActionEffectZonotope,
    limits: JointViabilityLimits,
) -> ViabilityMargins {
    let center = next_center(state, effects);
    let (reserve_radius, load_radius) = effects.coordinate_radii();
    let joint_lower =
        center.reserve - limits.load_weight * center.load
            - effects.box_joint_radius(limits.load_weight);

    ViabilityMargins {
        min_reserve: center.reserve - reserve_radius - limits.min_reserve,
        max_reserve: 1.0 - (center.reserve + reserve_radius),
        min_load: center.load - load_radius,
        max_load: limits.max_load - (center.load + load_radius),
        joint_margin: joint_lower - limits.min_joint_margin,
    }
}

/// Exact support-function margins for the declared zonotope and linear limits.
/// For a generator g and constraint normal c, its worst-case contribution is |c·g|.
pub fn zonotope_viability_margins(
    state: StatePoint,
    effects: &ActionEffectZonotope,
    limits: JointViabilityLimits,
) -> ViabilityMargins {
    let center = next_center(state, effects);
    let (reserve_radius, load_radius) = effects.coordinate_radii();
    let joint_lower = center.reserve - limits.load_weight * center.load
        - effects.joint_radius(limits.load_weight);

    ViabilityMargins {
        min_reserve: center.reserve - reserve_radius - limits.min_reserve,
        max_reserve: 1.0 - (center.reserve + reserve_radius),
        min_load: center.load - load_radius,
        max_load: limits.max_load - (center.load + load_radius),
        joint_margin: joint_lower - limits.min_joint_margin,
    }
}

/// Robust certification with the rectangular outer approximation.
pub fn box_certifies(
    state: StatePoint,
    effects: &ActionEffectZonotope,
    limits: JointViabilityLimits,
) -> bool {
    state.is_viable(limits) && box_viability_margins(state, effects, limits).is_safe()
}

/// Robust certification against the exact joint zonotope.
pub fn zonotope_certifies(
    state: StatePoint,
    effects: &ActionEffectZonotope,
    limits: JointViabilityLimits,
) -> bool {
    state.is_viable(limits) && zonotope_viability_margins(state, effects, limits).is_safe()
}

/// Independent vertex enumeration oracle for the linear constraints. Since a
/// zonotope is the convex hull of its vertices and each safety rule is linear,
/// every point is viable exactly when all enumerated vertices are viable.
pub fn corner_oracle_safe(
    state: StatePoint,
    effects: &ActionEffectZonotope,
    limits: JointViabilityLimits,
) -> bool {
    if !state.is_viable(limits) {
        return false;
    }

    for mask in 0..effects.corner_count() {
        let mut reserve = state.reserve + effects.center.reserve;
        let mut load = state.load + effects.center.load;
        for (index, generator) in effects.generators.iter().enumerate() {
            let sign = if (mask & (1_usize << index)) == 0 {
                -1.0
            } else {
                1.0
            };
            reserve += sign * generator.reserve;
            load += sign * generator.load;
        }
        let Some(vertex) = StatePoint::new(reserve, load) else {
            return false;
        };
        if !vertex.is_viable(limits) {
            return false;
        }
    }
    true
}

pub const fn action_name(action: Action) -> &'static str {
    match action {
        Action::Push => "push",
        Action::Maintain => "maintain",
        Action::EmergencyRecover => "emergency_recover",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        box_certifies, box_viability_margins, corner_oracle_safe, zonotope_certifies,
        zonotope_viability_margins, ActionEffectZonotope, Delta2D, JointViabilityLimits,
        StatePoint, MAX_ZONOTOPE_GENERATORS,
    };

    fn limits() -> JointViabilityLimits {
        JointViabilityLimits::new(0.15, 0.88, 0.5, 0.0).expect("valid limits")
    }

    fn point(reserve: f64, load: f64) -> StatePoint {
        StatePoint::new(reserve, load).expect("valid state point")
    }

    fn delta(reserve: f64, load: f64) -> Delta2D {
        Delta2D::new(reserve, load).expect("valid delta")
    }

    #[test]
    fn aligned_uncertainty_is_safe_as_a_zonotope_but_not_its_box_hull() {
        let state = point(0.32, 0.55);
        let effects = ActionEffectZonotope::new(delta(0.0, 0.0), vec![delta(0.06, 0.08)])
            .expect("valid zonotope");

        assert!(state.is_viable(limits()));
        assert!(!box_certifies(state, &effects, limits()));
        assert!(zonotope_certifies(state, &effects, limits()));
        assert!(corner_oracle_safe(state, &effects, limits()));
        assert!(zonotope_viability_margins(state, &effects, limits()).minimum() > 0.0);
        assert!(box_viability_margins(state, &effects, limits()).minimum() < 0.0);
    }

    #[test]
    fn anti_aligned_uncertainty_is_not_made_safe_by_geometry_optimism() {
        let state = point(0.32, 0.55);
        let effects = ActionEffectZonotope::new(delta(0.0, 0.0), vec![delta(0.06, -0.08)])
            .expect("valid zonotope");

        assert!(!box_certifies(state, &effects, limits()));
        assert!(!zonotope_certifies(state, &effects, limits()));
        assert!(!corner_oracle_safe(state, &effects, limits()));
    }

    #[test]
    fn support_function_matches_independent_vertex_oracle_across_generator_patterns() {
        let state = point(0.48, 0.48);
        for reserve_generator in [0.01, 0.03, 0.05] {
            for load_generator in [-0.06, -0.02, 0.02, 0.06] {
                let effects = ActionEffectZonotope::new(
                    delta(0.01, -0.01),
                    vec![
                        delta(reserve_generator, load_generator),
                        delta(-0.01, 0.015),
                    ],
                )
                .expect("valid zonotope");
                assert_eq!(
                    zonotope_certifies(state, &effects, limits()),
                    corner_oracle_safe(state, &effects, limits())
                );
            }
        }
    }

    #[test]
    fn box_certificate_never_exceeds_the_correlated_zonotope_certificate() {
        let state = point(0.4, 0.5);
        for r in [0.01, 0.03, 0.06] {
            for l in [0.02, 0.04, 0.08] {
                let effects = ActionEffectZonotope::new(
                    delta(0.0, 0.0),
                    vec![delta(r, l), delta(-r / 2.0, -l / 2.0)],
                )
                .expect("valid zonotope");
                if box_certifies(state, &effects, limits()) {
                    assert!(zonotope_certifies(state, &effects, limits()));
                }
            }
        }
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(StatePoint::new(f64::NAN, 0.5).is_none());
        assert!(Delta2D::new(0.0, f64::INFINITY).is_none());
        assert!(JointViabilityLimits::new(0.2, 0.8, f64::NAN, 0.0).is_none());

        let too_many = vec![delta(0.0, 0.0); MAX_ZONOTOPE_GENERATORS + 1];
        assert!(ActionEffectZonotope::new(delta(0.0, 0.0), too_many).is_none());
    }
}
