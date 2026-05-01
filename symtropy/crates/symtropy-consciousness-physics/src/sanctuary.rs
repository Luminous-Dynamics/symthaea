// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sanctuary zones: consciousness-created safe regions in physics space.
//!
//! When the Eight Harmonies "Sacred Stillness" activation exceeds 0.6 and
//! total harmony energy exceeds 2.0, a sanctuary zone forms around the entity.
//! Inside this zone, collision impulses are dampened by 90%+, creating a
//! consciousness-created safe haven in physics space.
//!
//! This couples consciousness directly to collision physics — not as a UI
//! effect, but as a real physical force modifier.

use nalgebra::SVector;
use symtropy_math::Point;

/// A sanctuary zone in D-dimensional space.
#[derive(Debug, Clone)]
pub struct SanctuaryZone<const D: usize> {
    /// Center of the sanctuary (entity position).
    pub center: Point<D>,
    /// Radius of the sanctuary effect.
    pub radius: f64,
    /// Dampening factor for collision impulses inside the zone [0.0, 1.0].
    /// 0.0 = no dampening, 1.0 = full absorption.
    pub dampening: f64,
    /// Whether the sanctuary is currently active.
    pub active: bool,
}

/// Conditions for sanctuary activation.
pub struct SanctuaryConditions {
    /// Sacred Stillness harmony activation [0.0, 1.0].
    pub stillness_activation: f64,
    /// Total harmony energy (sum of all 8 harmony activations).
    pub total_harmony_energy: f64,
    /// Current consciousness level (Φ).
    pub phi: f64,
}

impl<const D: usize> SanctuaryZone<D> {
    /// Create an inactive sanctuary zone at the given position.
    pub fn new(center: Point<D>, radius: f64) -> Self {
        Self {
            center,
            radius,
            dampening: 0.0,
            active: false,
        }
    }

    /// Update the sanctuary based on current harmony conditions.
    ///
    /// Activation requires:
    /// - Sacred Stillness > 0.6
    /// - Total harmony energy > 2.0
    /// - Φ > 0.3 (at least Yellow tier)
    pub fn update(&mut self, conditions: &SanctuaryConditions, new_center: Point<D>) {
        self.center = new_center;

        let should_activate = conditions.stillness_activation > 0.6
            && conditions.total_harmony_energy > 2.0
            && conditions.phi > 0.3;

        if should_activate {
            self.active = true;
            // Dampening scales with stillness and phi
            self.dampening = 0.9 * conditions.stillness_activation * conditions.phi.min(1.0);
        } else {
            self.active = false;
            self.dampening = 0.0;
        }
    }

    /// How much to dampen a collision impulse at the given point.
    ///
    /// Returns the impulse multiplier [0.0, 1.0]:
    /// - 1.0 = full impulse (outside sanctuary or inactive)
    /// - 0.0 = fully absorbed (center of active sanctuary)
    pub fn impulse_multiplier(&self, point: &SVector<f64, D>) -> f64 {
        if !self.active {
            return 1.0;
        }

        let dist = (Point(*point)).distance(&self.center);
        if dist >= self.radius {
            return 1.0;
        }

        // Linear falloff from center (max dampening) to edge (no dampening)
        let t = dist / self.radius; // 0 at center, 1 at edge
        1.0 - self.dampening * (1.0 - t)
    }

    /// Whether a point is inside the sanctuary zone.
    #[inline]
    pub fn contains(&self, point: &SVector<f64, D>) -> bool {
        self.active && (Point(*point)).distance(&self.center) < self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_conditions() -> SanctuaryConditions {
        SanctuaryConditions {
            stillness_activation: 0.8,
            total_harmony_energy: 4.0,
            phi: 0.7,
        }
    }

    fn inactive_conditions() -> SanctuaryConditions {
        SanctuaryConditions {
            stillness_activation: 0.3, // below threshold
            total_harmony_energy: 1.0,
            phi: 0.5,
        }
    }

    #[test]
    fn activation_requires_all_conditions() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&active_conditions(), Point::origin());
        assert!(zone.active);

        zone.update(&inactive_conditions(), Point::origin());
        assert!(!zone.active);
    }

    #[test]
    fn center_has_max_dampening() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&active_conditions(), Point::origin());

        let mult = zone.impulse_multiplier(&SVector::from([0.0, 0.0, 0.0]));
        // Should be heavily dampened (close to 0)
        assert!(
            mult < 0.5,
            "mult = {mult}, expected strong dampening at center"
        );
    }

    #[test]
    fn edge_has_no_dampening() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&active_conditions(), Point::origin());

        let mult = zone.impulse_multiplier(&SVector::from([10.0, 0.0, 0.0]));
        assert!(
            (mult - 1.0).abs() < 0.01,
            "mult = {mult}, expected ~1.0 at edge"
        );
    }

    #[test]
    fn outside_unaffected() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&active_conditions(), Point::origin());

        let mult = zone.impulse_multiplier(&SVector::from([20.0, 0.0, 0.0]));
        assert_eq!(mult, 1.0);
    }

    #[test]
    fn inactive_zone_no_effect() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&inactive_conditions(), Point::origin());

        let mult = zone.impulse_multiplier(&SVector::from([0.0, 0.0, 0.0]));
        assert_eq!(mult, 1.0);
    }

    #[test]
    fn dampening_is_monotonic_with_distance() {
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&active_conditions(), Point::origin());

        let m_center = zone.impulse_multiplier(&SVector::from([0.0, 0.0, 0.0]));
        let m_mid = zone.impulse_multiplier(&SVector::from([5.0, 0.0, 0.0]));
        let m_edge = zone.impulse_multiplier(&SVector::from([10.0, 0.0, 0.0]));

        assert!(
            m_center <= m_mid,
            "center ({m_center}) should be ≤ mid ({m_mid})"
        );
        assert!(m_mid <= m_edge, "mid ({m_mid}) should be ≤ edge ({m_edge})");
    }

    #[test]
    fn low_phi_prevents_sanctuary() {
        let conds = SanctuaryConditions {
            stillness_activation: 0.9,
            total_harmony_energy: 5.0,
            phi: 0.2, // Below 0.3 threshold
        };
        let mut zone = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone.update(&conds, Point::origin());
        assert!(!zone.active);
    }

    #[test]
    fn sanctuary_4d() {
        let mut zone = SanctuaryZone::<4>::new(Point::origin(), 5.0);
        zone.update(&active_conditions(), Point::origin());
        assert!(zone.contains(&SVector::from([1.0, 1.0, 1.0, 1.0]))); // dist=2, inside
        assert!(!zone.contains(&SVector::from([3.0, 3.0, 3.0, 3.0]))); // dist=6, outside
    }

    #[test]
    fn dampening_scales_with_phi() {
        let high_phi = SanctuaryConditions {
            stillness_activation: 0.8,
            total_harmony_energy: 4.0,
            phi: 0.9,
        };
        let low_phi = SanctuaryConditions {
            stillness_activation: 0.8,
            total_harmony_energy: 4.0,
            phi: 0.4,
        };

        let mut zone_high = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone_high.update(&high_phi, Point::origin());

        let mut zone_low = SanctuaryZone::<3>::new(Point::origin(), 10.0);
        zone_low.update(&low_phi, Point::origin());

        assert!(
            zone_high.dampening > zone_low.dampening,
            "high phi ({}) should dampen more than low phi ({})",
            zone_high.dampening,
            zone_low.dampening
        );
    }
}
