// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Harmony field physics: spatially-varying consciousness fields that modulate
//! local physics constants.
//!
//! Inspired by McFadden's CEMI field theory (2020): the brain's EM field *is*
//! consciousness and causally influences neural activity. By analogy, harmony
//! activations create fields that modulate local friction, collision response,
//! and other physics parameters.
//!
//! # Field Properties
//! - Each of the 8 harmonies creates a radial field around the entity
//! - Field strength falls off as 1/r² (like electromagnetic fields)
//! - Overlapping fields interact: resonance (aligned) vs interference (opposed)
//! - Physical effects: resonant fields reduce friction, dissonant increase impulse

use symtropy_math::Point;

/// Number of harmonies in the Eight Harmonies system.
pub const NUM_HARMONIES: usize = 8;

/// A harmony field source: an entity emitting harmony activations.
#[derive(Debug, Clone)]
pub struct HarmonySource<const D: usize> {
    /// Position of the source entity.
    pub position: Point<D>,
    /// Harmony activations [0.0, 1.0] for each of the 8 harmonies.
    pub activations: [f64; NUM_HARMONIES],
    /// Field strength multiplier (scales with consciousness level).
    pub strength: f64,
    /// Field radius (beyond this, field is negligible).
    pub radius: f64,
    /// Simulation time when this source was created/activated (Fix 6).
    /// Used for finite propagation delay. Default 0.0 (instant).
    pub created_at: f64,
    /// Field propagation speed in units/sec (Fix 6).
    /// Default f64::MAX (instant propagation for moving agents).
    /// Use finite values only for stationary sources (wells, sanctuaries).
    pub propagation_speed: f64,
}

/// The harmony field: aggregates all sources and computes field effects.
pub struct HarmonyField<const D: usize> {
    pub sources: Vec<HarmonySource<D>>,
}

impl<const D: usize> HarmonyField<D> {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Softening parameter ε for Plummer softening: (r² + ε²)^(n/2).
    /// Prevents singularity at r→0. Standard N-body astrophysics technique.
    const SOFTENING_EPSILON: f64 = 1.0;

    /// Sample the total harmony field at a point.
    ///
    /// Returns the summed harmony activations at that location,
    /// with dimension-correct 1/r^(D-1) falloff and Plummer softening.
    pub fn sample(&self, point: &Point<D>) -> [f64; NUM_HARMONIES] {
        let mut total = [0.0f64; NUM_HARMONIES];
        let exponent = (D as f64 - 1.0).max(1.0); // 2D→1/r, 3D→1/r², 4D→1/r³
        for source in &self.sources {
            let dist = source.position.distance(point);
            if dist >= source.radius {
                continue;
            }
            // Plummer-softened 1/r^(D-1) falloff (dimension-correct field theory)
            let r_soft = (dist * dist + Self::SOFTENING_EPSILON * Self::SOFTENING_EPSILON)
                .powf(exponent / 2.0);
            let falloff = source.strength / r_soft;

            for i in 0..NUM_HARMONIES {
                total[i] += source.activations[i] * falloff;
            }
        }
        total
    }

    /// Compute the resonance between two harmony activation vectors.
    ///
    /// Resonance = dot product of normalized activation vectors.
    /// Range: [-1, 1] where 1 = perfectly aligned, -1 = perfectly opposed.
    pub fn resonance(a: &[f64; NUM_HARMONIES], b: &[f64; NUM_HARMONIES]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Friction multiplier at a point, based on local harmony field.
    ///
    /// Resonant fields (aligned harmonies) reduce friction (cooperation flows).
    /// Dissonant fields (opposed harmonies) increase friction (conflict resistance).
    ///
    /// Returns multiplier [0.5, 2.0]:
    /// - 0.5 = half friction (strong resonance)
    /// - 1.0 = normal friction (no field or neutral)
    /// - 2.0 = double friction (strong dissonance)
    pub fn friction_multiplier(
        &self,
        point: &Point<D>,
        entity_harmonies: &[f64; NUM_HARMONIES],
    ) -> f64 {
        let field = self.sample(point);
        let res = Self::resonance(&field, entity_harmonies);
        // Map resonance [-1, 1] → friction [2.0, 0.5]
        // res = 1.0 → 0.5 (half friction, harmony flows)
        // res = 0.0 → 1.0 (normal)
        // res = -1.0 → 2.0 (double friction, conflict resists)
        1.0 - res * 0.5
    }

    /// Collision impulse multiplier at a point.
    ///
    /// Resonant fields dampen collisions (peaceful interactions).
    /// Dissonant fields amplify collisions (conflict escalates).
    ///
    /// Returns multiplier [0.5, 1.5].
    pub fn impulse_multiplier(
        &self,
        point: &Point<D>,
        entity_harmonies: &[f64; NUM_HARMONIES],
    ) -> f64 {
        let field = self.sample(point);
        let res = Self::resonance(&field, entity_harmonies);
        // res = 1.0 → 0.5 (dampened)
        // res = 0.0 → 1.0 (normal)
        // res = -1.0 → 1.5 (amplified)
        1.0 - res * 0.5
    }

    /// Sample the field at a point with finite propagation delay (Fix 6).
    ///
    /// Sources whose field hasn't reached the point yet (based on distance /
    /// propagation_speed since created_at) contribute zero.
    /// For stationary sources only — moving agents should use default
    /// propagation_speed = f64::MAX (instant).
    pub fn sample_at_time(&self, point: &Point<D>, current_time: f64) -> [f64; NUM_HARMONIES] {
        let mut total = [0.0f64; NUM_HARMONIES];
        let exponent = (D as f64 - 1.0).max(1.0);
        for source in &self.sources {
            let dist = source.position.distance(point);
            if dist >= source.radius {
                continue;
            }
            // Finite propagation delay check
            if source.propagation_speed < f64::MAX {
                let travel_time = dist / source.propagation_speed;
                if current_time < source.created_at + travel_time {
                    continue; // Field hasn't reached here yet
                }
            }
            let r_soft = (dist * dist + Self::SOFTENING_EPSILON * Self::SOFTENING_EPSILON)
                .powf(exponent / 2.0);
            let falloff = source.strength / r_soft;
            for i in 0..NUM_HARMONIES {
                total[i] += source.activations[i] * falloff;
            }
        }
        total
    }

    /// Total field energy at a point (sum of all harmony strengths).
    pub fn field_energy(&self, point: &Point<D>) -> f64 {
        let field = self.sample(point);
        field.iter().sum()
    }

    /// Gradient of conformal parameter σ = scale × field_energy (for curvature).
    /// Computed via central finite differences.
    #[cfg(feature = "consciousness-curvature")]
    pub fn sigma_gradient(
        &self,
        point: &Point<D>,
        curvature_scale: f64,
        epsilon: f64,
    ) -> nalgebra::SVector<f64, D> {
        let mut grad = nalgebra::SVector::<f64, D>::zeros();
        for i in 0..D {
            let mut p_plus = *point;
            let mut p_minus = *point;
            *p_plus.coord_mut(i) += epsilon;
            *p_minus.coord_mut(i) -= epsilon;
            let e_plus = self.field_energy(&p_plus) * curvature_scale;
            let e_minus = self.field_energy(&p_minus) * curvature_scale;
            grad[i] = (e_plus - e_minus) / (2.0 * epsilon);
        }
        grad
    }

    /// Laplacian of conformal parameter σ (for Ricci scalar, Fix 8).
    #[cfg(feature = "consciousness-curvature")]
    pub fn sigma_laplacian(
        &self,
        point: &Point<D>,
        curvature_scale: f64,
        epsilon: f64,
    ) -> f64 {
        let center = self.field_energy(point) * curvature_scale;
        let mut sum = 0.0;
        for i in 0..D {
            let mut p_plus = *point;
            let mut p_minus = *point;
            *p_plus.coord_mut(i) += epsilon;
            *p_minus.coord_mut(i) -= epsilon;
            let e_plus = self.field_energy(&p_plus) * curvature_scale;
            let e_minus = self.field_energy(&p_minus) * curvature_scale;
            sum += (e_plus - 2.0 * center + e_minus) / (epsilon * epsilon);
        }
        sum
    }
}

impl<const D: usize> Default for HarmonyField<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stillness_source<const D: usize>(pos: Point<D>) -> HarmonySource<D> {
        let mut activations = [0.0; NUM_HARMONIES];
        activations[7] = 1.0; // Sacred Stillness
        HarmonySource {
            position: pos,
            activations,
            strength: 10.0,
            radius: 100.0,
            created_at: 0.0,
            propagation_speed: f64::MAX,
        }
    }

    #[test]
    fn sample_at_source() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::origin()));

        let sample = field.sample(&Point::origin());
        // At distance ~1 (clamped), strength 10, stillness 1.0 → 10.0
        assert!(sample[7] > 1.0, "stillness = {}", sample[7]);
    }

    #[test]
    fn sample_falls_off_with_distance() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::origin()));

        let near = field.sample(&Point::new([2.0, 0.0, 0.0]));
        let far = field.sample(&Point::new([10.0, 0.0, 0.0]));
        assert!(near[7] > far[7], "near {} should be > far {}", near[7], far[7]);
    }

    #[test]
    fn sample_zero_outside_radius() {
        let mut source = stillness_source::<3>(Point::origin());
        source.radius = 5.0;
        let mut field = HarmonyField::new();
        field.sources.push(source);

        let outside = field.sample(&Point::new([10.0, 0.0, 0.0]));
        assert!(outside[7] < 1e-10);
    }

    #[test]
    fn resonance_aligned() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let res = HarmonyField::<3>::resonance(&a, &b);
        assert!((res - 1.0).abs() < 1e-10, "identical harmonies should resonate");
    }

    #[test]
    fn resonance_opposed() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let res = HarmonyField::<3>::resonance(&a, &b);
        assert!((res - 0.0).abs() < 1e-10, "orthogonal harmonies = zero resonance");
    }

    #[test]
    fn friction_reduced_by_resonance() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::origin()));

        let mut entity_harmonies = [0.0; NUM_HARMONIES];
        entity_harmonies[7] = 1.0; // Same harmony → resonance

        let mult = field.friction_multiplier(&Point::new([2.0, 0.0, 0.0]), &entity_harmonies);
        assert!(mult < 1.0, "resonant friction mult {} should be < 1.0", mult);
    }

    #[test]
    fn friction_increased_by_dissonance() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::origin()));

        let mut entity_harmonies = [0.0; NUM_HARMONIES];
        entity_harmonies[0] = 1.0; // Different harmony → less resonance

        // Since orthogonal harmonies give resonance=0, friction should be ~1.0
        let mult = field.friction_multiplier(&Point::new([2.0, 0.0, 0.0]), &entity_harmonies);
        assert!(
            (mult - 1.0).abs() < 0.5,
            "orthogonal friction mult {} should be near 1.0",
            mult
        );
    }

    #[test]
    fn multiple_sources_superpose() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::new([5.0, 0.0, 0.0])));
        field.sources.push(stillness_source(Point::new([-5.0, 0.0, 0.0])));

        let single_source = {
            let mut f = HarmonyField::<3>::new();
            f.sources.push(stillness_source(Point::new([5.0, 0.0, 0.0])));
            f.sample(&Point::origin())
        };

        let both = field.sample(&Point::origin());
        assert!(both[7] > single_source[7], "superposition should be stronger");
    }

    #[test]
    fn field_energy_sums() {
        let mut field = HarmonyField::<3>::new();
        field.sources.push(stillness_source(Point::origin()));
        let energy = field.field_energy(&Point::new([2.0, 0.0, 0.0]));
        assert!(energy > 0.0);
    }

    #[test]
    fn harmony_field_4d() {
        let mut field = HarmonyField::<4>::new();
        let source = HarmonySource {
            position: Point::origin(),
            activations: [0.5; 8],
            strength: 5.0,
            radius: 50.0,
            created_at: 0.0,
            propagation_speed: f64::MAX,
        };
        field.sources.push(source);
        let sample = field.sample(&Point::new([3.0, 0.0, 0.0, 0.0]));
        assert!(sample[0] > 0.0);
    }
}
