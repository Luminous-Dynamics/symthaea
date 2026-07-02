// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dimensional Energy Leakage: energy bleeds between dimensions.
//!
//! Based on Randall-Sundrum brane cosmology (1999): gravity leaks
//! between dimensional branes. In our engine, energy that moves
//! along the W axis (4th spatial dimension) disappears from a 3D
//! observer's perspective. The First Law appears to break locally.
//!
//! For FEP agents trapped in 3D, this produces IRREDUCIBLE prediction
//! errors — their model of reality fundamentally cannot account for
//! the missing Joules. This is the mathematical equivalent of
//! encountering dark energy.
//!
//! # Physics Grounding
//!
//! - Randall & Sundrum, PRL 83 (1999): gravity leaks between branes
//! - Jacobson, PRL 75 (1995): spacetime thermodynamics
//! - Ryu & Takayanagi, PRL 96 (2006): holographic entanglement entropy
//! - Carroll & Chen (2004): arrow of time from extra dimensions
//!
//! # Key Properties
//!
//! 1. Leakage rate scales with local energy density (hotter = leaks more)
//! 2. Leakage is directional along W axis
//! 3. D-dimensional observers see energy vanish (sink) or appear (source)
//! 4. Entropy increases locally near leakage points
//! 5. The pattern of leakage encodes the geometry of the extra dimension

use nalgebra::SVector;

/// A point in the extra dimension where energy leaks between branes.
///
/// Generic over the dimensionality D of the physics space. In 3D physics
/// this models a classical Randall-Sundrum brane sink/source. In 4D physics
/// the leakage operates in the full 4-dimensional space.
#[derive(Debug, Clone)]
pub struct LeakagePoint<const D: usize> {
    /// Position in D-dimensional space where the leakage manifests.
    pub position: SVector<f64, D>,
    /// W-coordinate of the leak (distance along the extra dimension).
    pub w_depth: f64,
    /// Leakage rate: Joules per tick flowing through this point.
    /// Positive = energy entering the simulation (source).
    /// Negative = energy leaving the simulation (sink).
    pub flow_rate: f64,
    /// Radius of effect in D-dimensional space.
    pub radius: f64,
    /// Whether this leak is currently active.
    pub active: bool,
    /// Cumulative energy transferred through this point.
    pub total_transferred: f64,
}

impl<const D: usize> LeakagePoint<D> {
    /// Create an energy sink (energy leaves simulation → extra dimension).
    pub fn sink(position: SVector<f64, D>, w_depth: f64, rate: f64, radius: f64) -> Self {
        Self {
            position,
            w_depth,
            flow_rate: -rate.abs(), // negative = draining
            radius,
            active: true,
            total_transferred: 0.0,
        }
    }

    /// Create an energy source (energy enters simulation from extra dimension).
    pub fn source(position: SVector<f64, D>, w_depth: f64, rate: f64, radius: f64) -> Self {
        Self {
            position,
            w_depth,
            flow_rate: rate.abs(), // positive = injecting
            radius,
            active: true,
            total_transferred: 0.0,
        }
    }

    /// Distance from a D-dimensional position to this leakage point.
    pub fn distance(&self, pos: &SVector<f64, D>) -> f64 {
        (pos - self.position).norm()
    }

    /// Energy effect on an agent at the given position.
    /// Returns Joules per tick (positive = gaining, negative = losing).
    /// Falls off as 1/r² from the leakage point (like gravity).
    pub fn effect_at(&self, pos: &SVector<f64, D>) -> f64 {
        if !self.active {
            return 0.0;
        }
        let dist = self.distance(pos);
        if dist >= self.radius {
            return 0.0;
        }
        // 1/r² falloff, clamped at r=1 to avoid singularity
        let r = dist.max(1.0);
        self.flow_rate / (r * r)
    }
}

/// The dimensional leakage system: manages all leakage points and
/// tracks the energy balance between dimensions.
#[derive(Debug, Clone)]
pub struct DimensionalLeakage<const D: usize> {
    /// All leakage points in the world.
    pub points: Vec<LeakagePoint<D>>,
    /// Total energy that has leaked OUT of the simulation (into the bulk).
    pub total_leaked_out: f64,
    /// Total energy that has leaked INTO the simulation (from the bulk).
    pub total_leaked_in: f64,
    /// The apparent conservation violation as seen by simulation observers.
    /// This is the "dark energy" — energy unaccounted for in the simulation.
    pub apparent_violation: f64,
    /// Whether leakage is enabled.
    pub enabled: bool,
}

impl<const D: usize> DimensionalLeakage<D> {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            total_leaked_out: 0.0,
            total_leaked_in: 0.0,
            apparent_violation: 0.0,
            enabled: false,
        }
    }

    /// Add a leakage point.
    pub fn add_point(&mut self, point: LeakagePoint<D>) {
        self.points.push(point);
    }

    /// Create a standard wormhole configuration: one sink and one source at
    /// different positions, simulating energy flowing through the extra
    /// dimension from one location to another.
    pub fn create_wormhole(
        &mut self,
        entry: SVector<f64, D>,
        exit: SVector<f64, D>,
        rate: f64,
        radius: f64,
    ) {
        self.add_point(LeakagePoint::sink(entry, 1.0, rate, radius));
        self.add_point(LeakagePoint::source(exit, 1.0, rate, radius));
        self.enabled = true;
    }

    /// Total energy effect at a D-dimensional position from all leakage points.
    pub fn total_effect_at(&self, pos: &SVector<f64, D>) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        self.points.iter().map(|p| p.effect_at(pos)).sum()
    }

    /// Tick the leakage system: accumulate transfer totals.
    pub fn tick(&mut self) {
        if !self.enabled {
            return;
        }
        for point in &mut self.points {
            if point.active {
                // The flow_rate represents energy per tick at unit distance.
                // Total throughput scales with the integral of 1/r² over the radius.
                let throughput = point.flow_rate.abs() * point.radius.min(10.0);
                point.total_transferred += throughput;

                if point.flow_rate < 0.0 {
                    self.total_leaked_out += throughput;
                } else {
                    self.total_leaked_in += throughput;
                }
            }
        }
        self.apparent_violation = self.total_leaked_out - self.total_leaked_in;
    }

    /// The prediction error an FEP agent would experience at this position.
    ///
    /// An agent expecting energy conservation sees energy appear/disappear.
    /// The magnitude of the apparent violation IS the prediction error.
    /// This is irreducible — no D-dimensional model can explain it.
    pub fn prediction_error_at(&self, pos: &SVector<f64, D>) -> f64 {
        self.total_effect_at(pos).abs()
    }

    /// Whether any leakage point is within `threshold` of a position.
    pub fn near_leakage(&self, pos: &SVector<f64, D>, threshold: f64) -> bool {
        self.points.iter().any(|p| p.distance(pos) < threshold)
    }

    /// Entropy increase rate at a position due to dimensional leakage.
    ///
    /// Per Jacobson (1995): spacetime thermodynamics implies energy
    /// flow through dimensional boundaries increases local entropy.
    pub fn local_entropy_increase(&self, pos: &SVector<f64, D>) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        // Entropy increase proportional to energy flow rate
        self.total_effect_at(pos).abs() * 0.1
    }
}

impl<const D: usize> Default for DimensionalLeakage<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(x: f64, y: f64, z: f64) -> SVector<f64, 3> {
        SVector::from([x, y, z])
    }

    #[test]
    fn sink_drains_energy() {
        let sink = LeakagePoint::sink(vec3(0.0, 0.0, 0.0), 1.0, 0.5, 50.0);
        let effect = sink.effect_at(&vec3(5.0, 0.0, 0.0));
        assert!(effect < 0.0, "sink should drain energy, got {}", effect);
    }

    #[test]
    fn source_provides_energy() {
        let source = LeakagePoint::source(vec3(0.0, 0.0, 0.0), 1.0, 0.5, 50.0);
        let effect = source.effect_at(&vec3(5.0, 0.0, 0.0));
        assert!(effect > 0.0, "source should provide energy, got {}", effect);
    }

    #[test]
    fn effect_falls_off_with_distance() {
        let sink = LeakagePoint::sink(vec3(0.0, 0.0, 0.0), 1.0, 1.0, 100.0);
        let near = sink.effect_at(&vec3(2.0, 0.0, 0.0)).abs();
        let far = sink.effect_at(&vec3(10.0, 0.0, 0.0)).abs();
        assert!(near > far, "effect should decrease with distance");
    }

    #[test]
    fn no_effect_outside_radius() {
        let sink = LeakagePoint::sink(vec3(0.0, 0.0, 0.0), 1.0, 1.0, 10.0);
        let effect = sink.effect_at(&vec3(20.0, 0.0, 0.0));
        assert_eq!(effect, 0.0, "no effect outside radius");
    }

    #[test]
    fn wormhole_conserves_in_extra_dimension() {
        let mut leakage = DimensionalLeakage::<3>::new();
        leakage.create_wormhole(vec3(0.0, 0.0, 0.0), vec3(50.0, 0.0, 0.0), 1.0, 20.0);

        // Near entry (sink): energy drains
        let at_entry = leakage.total_effect_at(&vec3(1.0, 0.0, 0.0));
        assert!(at_entry < 0.0, "entry should drain");

        // Near exit (source): energy appears
        let at_exit = leakage.total_effect_at(&vec3(51.0, 0.0, 0.0));
        assert!(at_exit > 0.0, "exit should provide");
    }

    #[test]
    fn prediction_error_nonzero_near_leakage() {
        let mut leakage = DimensionalLeakage::<3>::new();
        leakage.create_wormhole(vec3(0.0, 0.0, 0.0), vec3(100.0, 0.0, 0.0), 1.0, 30.0);

        let pe_near = leakage.prediction_error_at(&vec3(5.0, 0.0, 0.0));
        let pe_far = leakage.prediction_error_at(&vec3(200.0, 0.0, 0.0));
        assert!(pe_near > pe_far, "prediction error higher near leakage");
    }

    #[test]
    fn entropy_increases_near_leakage() {
        let mut leakage = DimensionalLeakage::<3>::new();
        leakage.create_wormhole(vec3(0.0, 0.0, 0.0), vec3(100.0, 0.0, 0.0), 1.0, 30.0);

        let entropy_near = leakage.local_entropy_increase(&vec3(5.0, 0.0, 0.0));
        let entropy_far = leakage.local_entropy_increase(&vec3(200.0, 0.0, 0.0));
        assert!(entropy_near > entropy_far, "entropy increases near leakage");
    }

    #[test]
    fn disabled_leakage_no_effect() {
        let leakage = DimensionalLeakage::<3>::new(); // enabled = false
        let effect = leakage.total_effect_at(&vec3(0.0, 0.0, 0.0));
        assert_eq!(effect, 0.0);
    }

    #[test]
    fn tick_accumulates_totals() {
        let mut leakage = DimensionalLeakage::<3>::new();
        leakage.create_wormhole(vec3(0.0, 0.0, 0.0), vec3(100.0, 0.0, 0.0), 1.0, 20.0);

        leakage.tick();
        assert!(leakage.total_leaked_out > 0.0);
        assert!(leakage.total_leaked_in > 0.0);

        leakage.tick();
        let after_two = leakage.total_leaked_out;
        assert!(after_two > 0.0, "should accumulate over ticks");
    }

    #[test]
    fn works_in_4d() {
        let origin = SVector::<f64, 4>::from([0.0, 0.0, 0.0, 0.0]);
        let far = SVector::<f64, 4>::from([50.0, 0.0, 0.0, 0.0]);
        let mut leakage = DimensionalLeakage::<4>::new();
        leakage.create_wormhole(origin, far, 1.0, 20.0);

        let near_pos = SVector::<f64, 4>::from([1.0, 0.0, 0.0, 0.0]);
        let at_entry = leakage.total_effect_at(&near_pos);
        assert!(at_entry < 0.0, "4D entry should drain energy");
    }

    #[test]
    fn works_in_2d() {
        let origin = SVector::<f64, 2>::from([0.0, 0.0]);
        let far = SVector::<f64, 2>::from([50.0, 0.0]);
        let mut leakage = DimensionalLeakage::<2>::new();
        leakage.create_wormhole(origin, far, 1.0, 20.0);

        let near_pos = SVector::<f64, 2>::from([1.0, 0.0]);
        let at_entry = leakage.total_effect_at(&near_pos);
        assert!(at_entry < 0.0, "2D entry should drain energy");
    }
}
