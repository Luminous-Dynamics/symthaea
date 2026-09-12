// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Planar Lie-group configuration geometry built on the shared `S^1` factor.

use crate::configuration_space::{CircleSpace, ConfigurationSpaceError, LieGroup};
use crate::state_space::{MetricSpace, StateSpace, StateSpaceError, StateSpaceProfile, StateValidity};

/// `SO(2)` is isomorphic to `S^1`; this alias preserves that exact topology and metric.
pub type So2Space = CircleSpace;

/// A rigid planar pose `(x, y, theta)` with angle in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidPose2 {
    /// Translation in the selected 2D frame.
    pub translation: [f64; 2],
    /// Orientation angle in radians.
    pub rotation: f64,
}

/// `SE(2)` with explicit translation and rotation metric weights.
#[derive(Clone, Debug)]
pub struct Se2Space {
    rotation: CircleSpace,
    translation_weight: f64,
    rotation_weight: f64,
    profile: StateSpaceProfile,
}

impl Se2Space {
    /// Construct a planar rigid-pose space.
    ///
    /// No universal metres-to-radians conversion is implied; both weights are
    /// explicit and bound into profile identity.
    pub fn new(
        translation_weight: f64,
        rotation_weight: f64,
    ) -> Result<Self, ConfigurationSpaceError> {
        for (name, value) in [
            ("translation", translation_weight),
            ("rotation", rotation_weight),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigurationSpaceError::InvalidMetricWeight {
                    name: name.to_string(),
                    value,
                });
            }
        }
        let rotation = CircleSpace::new();
        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&translation_weight.to_bits().to_le_bytes());
        parameters.extend_from_slice(&rotation_weight.to_bits().to_le_bytes());
        let profile = StateSpaceProfile::new(
            "se2-rigid-pose",
            Some(3),
            "weighted-r2-so2-product-v1",
            parameters,
            vec![rotation.profile().clone()],
        );
        Ok(Self {
            rotation,
            translation_weight,
            rotation_weight,
            profile,
        })
    }

    /// Translation metric weight.
    pub fn translation_weight(&self) -> f64 {
        self.translation_weight
    }

    /// Rotation metric weight.
    pub fn rotation_weight(&self) -> f64 {
        self.rotation_weight
    }
}

impl StateSpace for Se2Space {
    type State = RigidPose2;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        for (coordinate, value) in state.translation.iter().enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate };
            }
        }
        let rotation = self.rotation.validate_state(&state.rotation);
        if !rotation.is_valid() {
            return StateValidity::ComponentInvalid {
                component: 1,
                validity: Box::new(rotation),
            };
        }
        StateValidity::Valid
    }

    fn interpolate(
        &self,
        from: &Self::State,
        to: &Self::State,
        t: f64,
    ) -> Result<Self::State, StateSpaceError> {
        if !t.is_finite() || !(0.0..=1.0).contains(&t) {
            return Err(StateSpaceError::InvalidInterpolationParameter { t });
        }
        self.require_valid("from", from)?;
        self.require_valid("to", to)?;
        if t <= 0.0 {
            return Ok(*from);
        }
        if t >= 1.0 {
            return Ok(*to);
        }
        let translation = [
            (1.0 - t) * from.translation[0] + t * to.translation[0],
            (1.0 - t) * from.translation[1] + t * to.translation[1],
        ];
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(2) translation interpolation",
            });
        }
        let rotation = self.rotation.interpolate(&from.rotation, &to.rotation, t)?;
        Ok(RigidPose2 {
            translation,
            rotation,
        })
    }
}

impl MetricSpace for Se2Space {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        let translation = (a.translation[0] - b.translation[0])
            .hypot(a.translation[1] - b.translation[1]);
        let rotation = self.rotation.distance(&a.rotation, &b.rotation)?;
        let result = (self.translation_weight.sqrt() * translation)
            .hypot(self.rotation_weight.sqrt() * rotation);
        if result.is_finite() {
            Ok(result)
        } else {
            Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(2) weighted distance",
            })
        }
    }
}

impl LieGroup for Se2Space {
    fn identity(&self) -> Self::State {
        RigidPose2 {
            translation: [0.0; 2],
            rotation: 0.0,
        }
    }

    fn compose(
        &self,
        left: &Self::State,
        right: &Self::State,
    ) -> Result<Self::State, StateSpaceError> {
        self.require_valid("left", left)?;
        self.require_valid("right", right)?;
        let cos = left.rotation.cos();
        let sin = left.rotation.sin();
        let rotated = [
            cos * right.translation[0] - sin * right.translation[1],
            sin * right.translation[0] + cos * right.translation[1],
        ];
        let translation = [
            left.translation[0] + rotated[0],
            left.translation[1] + rotated[1],
        ];
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(2) composition translation",
            });
        }
        let rotation = self.rotation.compose(&left.rotation, &right.rotation)?;
        Ok(RigidPose2 {
            translation,
            rotation,
        })
    }

    fn inverse(&self, state: &Self::State) -> Result<Self::State, StateSpaceError> {
        self.require_valid("state", state)?;
        let rotation = self.rotation.inverse(&state.rotation)?;
        let cos = rotation.cos();
        let sin = rotation.sin();
        let negative = [-state.translation[0], -state.translation[1]];
        let translation = [
            cos * negative[0] - sin * negative[1],
            sin * negative[0] + cos * negative[1],
        ];
        Ok(RigidPose2 {
            translation,
            rotation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn so2_alias_preserves_circle_wraparound() {
        let space = So2Space::new();
        let a = 179.0_f64.to_radians();
        let b = (-179.0_f64).to_radians();
        close(
            space.distance(&a, &b).unwrap(),
            2.0_f64.to_radians(),
            1e-12,
        );
    }

    #[test]
    fn se2_inverse_cancels_pose() {
        let space = Se2Space::new(1.0, 1.0).unwrap();
        let pose = RigidPose2 {
            translation: [2.0, -1.0],
            rotation: 0.7,
        };
        let inverse = space.inverse(&pose).unwrap();
        let composed = space.compose(&pose, &inverse).unwrap();
        close(
            space.distance(&composed, &space.identity()).unwrap(),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn se2_metric_profile_binds_weighting() {
        let a = Se2Space::new(1.0, 1.0).unwrap();
        let b = Se2Space::new(2.0, 1.0).unwrap();
        assert_ne!(a.profile().identity(), b.profile().identity());
    }
}
