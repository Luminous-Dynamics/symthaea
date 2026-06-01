// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic state-coupled physics field.
//!
//! [`SimpleCoupledField`] implements `PhysicsCallback<D>` using ANY user-provided
//! metric instead of requiring IIT or the Master Consciousness Equation.
//!
//! The metric is a `f64` in [0, 1] per body. It couples to physics through the
//! same 5-channel framework as `ConsciousnessField`:
//!
//! 1. **Metric → Force**: NRC 4-tier safety gates motor authority
//! 2. **Metric → Energy**: maintenance cost scales with metric value
//! 3. **Harmony → Impulse**: sanctuary zones dampen collisions
//! 4. **Harmony → Friction**: CEMI-inspired fields modulate friction
//! 5. **Collision → Metric**: prediction error reduces motor precision
//!
//! # Use Cases
//!
//! - **Game dev**: health-coupled physics (low health = weak forces)
//! - **Economics**: wealth-coupled dynamics (broke agents can't move)
//! - **Social science**: trust-coupled friction (aligned agents slide, opposed resist)
//! - **Robotics**: skill-coupled motor authority (novice robot = reduced force)
//! - **Biology**: fitness-coupled energy budgets (weak organisms collapse)
//!
//! # Example
//!
//! ```ignore
//! let mut field = SimpleCoupledField::<2>::new();
//! field.register(handle, 100.0, 10.0);
//! field.set_metric(handle, 0.8);  // high metric = full force authority
//! world.step_with_callback(0.016, &mut field);
//! ```

use nalgebra::SVector;
use std::collections::BTreeMap;
use symtropy_math::Point;

use crate::energy::EnergyBudget;
use crate::harmony_field::{HarmonyField, HarmonySource, NUM_HARMONIES};
use crate::safety::SafetyTier;
use crate::sanctuary::{SanctuaryConditions, SanctuaryZone};
use crate::thermodynamics::{ThermodynamicConstants, ThermodynamicLedger};
use symtropy_physics::body::BodyHandle;
use symtropy_physics::world::PhysicsCallback;

/// Per-entity state for the simple coupled field.
pub struct SimpleEntity {
    /// The user-provided state metric [0, 1].
    /// 0.0 = Red (no motor authority), 1.0 = Green (full authority).
    pub metric: f64,
    /// Safety tier derived from metric.
    pub safety_tier: SafetyTier,
    /// Energy budget (persistent, depletes through actions).
    pub energy: EnergyBudget,
    /// Harmony activations [0, 1] for each of the 9 harmony dimensions.
    pub harmony_activations: [f64; NUM_HARMONIES],
    /// Prediction error from unexpected collisions.
    pub prediction_error: f64,
    /// Motor precision = 1 / (1 + prediction_error).
    pub motor_precision: f64,
    /// Prediction error decay rate per tick.
    pub prediction_decay: f64,
}

impl SimpleEntity {
    pub fn new(max_energy: f64) -> Self {
        Self {
            metric: 0.5,
            safety_tier: SafetyTier::Yellow,
            energy: EnergyBudget::new(max_energy),
            harmony_activations: [0.0; NUM_HARMONIES],
            prediction_error: 0.0,
            motor_precision: 1.0,
            prediction_decay: 0.05,
        }
    }

    /// Set the state metric and update derived fields.
    pub fn set_metric(&mut self, value: f64) {
        self.metric = value.clamp(0.0, 1.0);
        self.safety_tier = if self.energy.is_collapsed() {
            SafetyTier::Red
        } else {
            SafetyTier::from_phi(self.metric)
        };
    }

    /// Effective motor gain: safety tier * motor precision.
    pub fn effective_motor_gain(&self) -> f64 {
        self.safety_tier.motor_gain() * self.motor_precision
    }

    /// Register a collision — spikes prediction error.
    pub fn on_collision(&mut self, impulse_magnitude: f64) {
        let spike = (impulse_magnitude * 0.01).min(2.0);
        self.prediction_error += spike;
        self.motor_precision = 1.0 / (1.0 + self.prediction_error);
    }

    /// Decay prediction error (call each tick).
    pub fn tick_prediction_error(&mut self) {
        self.prediction_error *= 1.0 - self.prediction_decay;
        if self.prediction_error < 1e-6 {
            self.prediction_error = 0.0;
        }
        self.motor_precision = 1.0 / (1.0 + self.prediction_error);
    }

    /// Sacred Stillness activation (harmony index 7).
    pub fn stillness(&self) -> f64 {
        self.harmony_activations[7]
    }

    /// Total harmony energy.
    pub fn total_harmony_energy(&self) -> f64 {
        self.harmony_activations.iter().sum()
    }

    /// Sanctuary conditions from current state.
    pub fn sanctuary_conditions(&self) -> SanctuaryConditions {
        SanctuaryConditions {
            stillness_activation: self.stillness(),
            total_harmony_energy: self.total_harmony_energy(),
            phi: self.metric,
        }
    }
}

/// Generic state-coupled physics field.
///
/// Drop-in replacement for `ConsciousnessField` that accepts any [0, 1] metric
/// per body instead of requiring the Master Consciousness Equation.
pub struct SimpleCoupledField<const D: usize> {
    /// Per-entity states.
    pub entities: BTreeMap<BodyHandle, SimpleEntity>,
    /// Sanctuary zones for impulse dampening.
    pub sanctuaries: BTreeMap<BodyHandle, SanctuaryZone<D>>,
    /// Harmony field for Channel 4 (friction coupling).
    pub harmony_field: HarmonyField<D>,
    /// Cached positions for phi-gravity and harmony rebuilds.
    entity_positions: BTreeMap<BodyHandle, SVector<f64, D>>,
    /// Collective metric (average across all entities).
    pub collective_metric: f64,
    /// Thermodynamic ledger.
    pub ledger: ThermodynamicLedger,
    /// Tunable constants.
    pub constants: ThermodynamicConstants,
    /// Sanctuary absorption accumulator (thread-safe).
    sanctuary_absorbed: std::sync::atomic::AtomicU64,
}

impl<const D: usize> SimpleCoupledField<D> {
    pub fn new() -> Self {
        Self {
            entities: BTreeMap::new(),
            sanctuaries: BTreeMap::new(),
            harmony_field: HarmonyField::new(),
            entity_positions: BTreeMap::new(),
            collective_metric: 0.0,
            ledger: ThermodynamicLedger::new(),
            constants: ThermodynamicConstants::default(),
            sanctuary_absorbed: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Register a new entity.
    pub fn register(&mut self, handle: BodyHandle, max_energy: f64, sanctuary_radius: f64) {
        self.entities.insert(handle, SimpleEntity::new(max_energy));
        self.sanctuaries.insert(
            handle,
            SanctuaryZone::new(Point::origin(), sanctuary_radius),
        );
    }

    /// Set the state metric for an entity.
    pub fn set_metric(&mut self, handle: BodyHandle, value: f64) {
        if let Some(entity) = self.entities.get_mut(&handle) {
            entity.set_metric(value);
        }
        self.recompute_collective();
    }

    /// Get the current metric for an entity.
    pub fn metric(&self, handle: BodyHandle) -> f64 {
        self.entities.get(&handle).map(|e| e.metric).unwrap_or(0.0)
    }

    /// Get safety tier.
    pub fn safety_tier(&self, handle: BodyHandle) -> SafetyTier {
        self.entities
            .get(&handle)
            .map(|e| e.safety_tier)
            .unwrap_or(SafetyTier::Green)
    }

    /// Whether entity has energy.
    pub fn has_energy(&self, handle: BodyHandle) -> bool {
        self.entities
            .get(&handle)
            .map(|e| e.energy.has_energy())
            .unwrap_or(false)
    }

    /// Consume energy from an entity.
    pub fn consume_energy(&mut self, handle: BodyHandle, amount: f64) -> f64 {
        let metric = self.metric(handle);
        let consumed = self
            .entities
            .get_mut(&handle)
            .map(|e| e.energy.consume(amount))
            .unwrap_or(0.0);
        if consumed > 0.0 {
            self.ledger.record_action(consumed, metric);
        }
        consumed
    }

    /// Tick prediction error decay for all entities.
    pub fn tick_prediction_errors(&mut self) {
        for entity in self.entities.values_mut() {
            entity.tick_prediction_error();
        }
    }

    /// Rebuild harmony field from entity positions.
    pub fn rebuild_harmony_field(&mut self, positions: &[(BodyHandle, Point<D>)]) {
        self.harmony_field.sources.clear();
        self.entity_positions.clear();
        for (handle, pos) in positions {
            self.entity_positions.insert(*handle, pos.to_vector());
            if let Some(entity) = self.entities.get(handle) {
                self.harmony_field.sources.push(HarmonySource {
                    position: *pos,
                    activations: entity.harmony_activations,
                    strength: entity.metric.max(0.1),
                    radius: self.constants.harmony_range,
                    created_at: 0.0,
                    propagation_speed: f64::MAX,
                });
            }
        }
    }

    /// Update sanctuary zone for an entity.
    pub fn update_sanctuary(&mut self, handle: BodyHandle, position: Point<D>) {
        if let Some(entity) = self.entities.get(&handle) {
            let conditions = entity.sanctuary_conditions();
            if let Some(sanctuary) = self.sanctuaries.get_mut(&handle) {
                sanctuary.update(&conditions, position);
            }
        }
    }

    /// Resource regeneration multiplier based on collective metric.
    pub fn resource_regeneration_multiplier(&self) -> f64 {
        0.5 + 1.5 * self.collective_metric
    }

    fn drain_sanctuary_absorption(&mut self) {
        let bits = self
            .sanctuary_absorbed
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let absorbed = f64::from_bits(bits);
        if absorbed > 1e-15 {
            self.ledger.record_dissipation(absorbed);
        }
    }

    fn recompute_collective(&mut self) {
        if self.entities.is_empty() {
            self.collective_metric = 0.0;
            return;
        }
        let sum: f64 = self.entities.values().map(|e| e.metric).sum();
        self.collective_metric = sum / self.entities.len() as f64;
    }
}

impl<const D: usize> Default for SimpleCoupledField<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// PhysicsCallback implementation — same 5-channel coupling as ConsciousnessField.
impl<const D: usize> PhysicsCallback<D> for SimpleCoupledField<D> {
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        let gain = self
            .entities
            .get(&body)
            .map(|e| e.effective_motor_gain())
            .unwrap_or(1.0);
        force * gain
    }

    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64 {
        let mut result = impulse;
        for sanctuary in self.sanctuaries.values() {
            result *= sanctuary.impulse_multiplier(contact_point);
        }

        let dampened = result;
        let absorbed = impulse - dampened;
        if absorbed > 1e-10 {
            use std::sync::atomic::Ordering;
            let absorbed_energy = absorbed * 0.5;
            let _ = self.sanctuary_absorbed.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |bits| Some((f64::from_bits(bits) + absorbed_energy).to_bits()),
            );
        }
        dampened
    }

    fn friction_multiplier(&self, contact_point: &SVector<f64, D>, body: BodyHandle) -> f64 {
        if self.harmony_field.sources.is_empty() {
            return 1.0;
        }
        let harmonies = self
            .entities
            .get(&body)
            .map(|e| e.harmony_activations)
            .unwrap_or([0.0; NUM_HARMONIES]);
        let point = Point(*contact_point);
        self.harmony_field.friction_multiplier(&point, &harmonies)
    }

    fn on_collision(&mut self, event: &symtropy_physics::CollisionEvent<D>) {
        self.drain_sanctuary_absorption();
        let drain_rate = self.constants.collision_energy_drain;

        let resonance = {
            let ha = self
                .entities
                .get(&event.body_a)
                .map(|e| e.harmony_activations);
            let hb = self
                .entities
                .get(&event.body_b)
                .map(|e| e.harmony_activations);
            match (ha, hb) {
                (Some(a), Some(b)) => HarmonyField::<D>::resonance(&a, &b),
                _ => 0.0,
            }
        };

        if let Some(entity) = self.entities.get_mut(&event.body_a) {
            let surprise = (1.0 - resonance).max(0.1);
            entity.on_collision(event.impulse * surprise);
            let consumed = entity.energy.consume(event.impulse * drain_rate);
            entity.energy.dissipate_heat(consumed * 0.5);
        }
        if let Some(entity) = self.entities.get_mut(&event.body_b) {
            let surprise = (1.0 - resonance).max(0.1);
            entity.on_collision(event.impulse * surprise);
            let consumed = entity.energy.consume(event.impulse * drain_rate);
            entity.energy.dissipate_heat(consumed * 0.5);
        }
    }

    fn record_dissipation(&mut self, energy: f64) {
        self.drain_sanctuary_absorption();
        self.ledger.record_dissipation(energy);
        let n = self.entities.len();
        if n > 0 && energy > 1e-15 {
            let per_entity = energy / n as f64;
            for entity in self.entities.values_mut() {
                entity.energy.dissipate_heat(per_entity);
            }
        }
    }

    fn record_work(&mut self, body: BodyHandle, work_joules: f64) {
        if let Some(entity) = self.entities.get_mut(&body) {
            if work_joules > 0.0 {
                let actual_consumed = entity.energy.consume(work_joules);
                self.ledger.record_action(actual_consumed, 1.0);
                entity.energy.dissipate_heat(actual_consumed * 0.1);
            } else {
                let recovered = work_joules.abs() * 0.4;
                entity.energy.regenerate(recovered);
                self.ledger.lifetime_energy -= recovered;
            }
        }
    }

    fn apply_trauma(&mut self, _: &symtropy_physics::CollisionEvent<D>) {}
}

impl<const D: usize> SimpleCoupledField<D> {
    /// Finalize this tick's energy balance.
    pub fn tick_thermodynamics(&mut self) -> crate::thermodynamics::TickBalance {
        self.drain_sanctuary_absorption();
        self.ledger.tick_balance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_physics::world::PhysicsCallback;

    #[test]
    fn high_metric_full_force() {
        let mut field = SimpleCoupledField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.set_metric(h, 0.9);

        let force = SVector::from([10.0, 0.0, 0.0]);
        let modulated = PhysicsCallback::<3>::modulate_force(&field, h, &force);
        // High metric (0.9) → Green tier → gain = 1.0
        assert!(
            modulated[0] > 5.0,
            "high metric should give strong force, got {}",
            modulated[0]
        );
    }

    #[test]
    fn low_metric_reduced_force() {
        let mut field = SimpleCoupledField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.set_metric(h, 0.05);

        let force = SVector::from([10.0, 0.0, 0.0]);
        let modulated = PhysicsCallback::<3>::modulate_force(&field, h, &force);
        // Low metric (0.05) → Red tier → gain = 0.0
        assert!(
            modulated[0] < 1.0,
            "low metric should reduce force, got {}",
            modulated[0]
        );
    }

    #[test]
    fn energy_depletes_and_collapses() {
        let mut field = SimpleCoupledField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.set_metric(h, 0.8);

        assert!(field.has_energy(h));
        field.consume_energy(h, 100.0);
        assert!(!field.has_energy(h));

        // Collapsed entity should be Red tier regardless of metric
        field.set_metric(h, 0.9);
        assert_eq!(field.safety_tier(h), SafetyTier::Red);
    }

    #[test]
    fn collective_metric_averages() {
        let mut field = SimpleCoupledField::<3>::new();
        field.register(BodyHandle(0), 100.0, 10.0);
        field.register(BodyHandle(1), 100.0, 10.0);
        field.set_metric(BodyHandle(0), 0.8);
        field.set_metric(BodyHandle(1), 0.4);

        assert!((field.collective_metric - 0.6).abs() < 1e-10);
    }

    #[test]
    fn prediction_error_reduces_precision() {
        let mut field = SimpleCoupledField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.set_metric(h, 0.8);

        let gain_before = field.entities.get(&h).unwrap().effective_motor_gain();

        // Simulate collision
        field.entities.get_mut(&h).unwrap().on_collision(100.0);

        let gain_after = field.entities.get(&h).unwrap().effective_motor_gain();
        assert!(
            gain_after < gain_before,
            "collision should reduce motor gain"
        );

        // Decay recovers precision
        for _ in 0..40 {
            field.entities.get_mut(&h).unwrap().tick_prediction_error();
        }
        let gain_recovered = field.entities.get(&h).unwrap().effective_motor_gain();
        assert!(gain_recovered > gain_after, "precision should recover");
    }

    #[test]
    fn works_as_physics_callback() {
        use symtropy_physics::PhysicsWorld;

        let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
        let h = world.add_sphere(symtropy_math::Point::new([0.0, 10.0]), 1.0, 1.0);

        let mut field = SimpleCoupledField::<2>::new();
        field.register(h, 100.0, 10.0);
        field.set_metric(h, 0.7);

        // Step with the SimpleCoupledField as callback — should not panic
        world.step_with_callback(0.016, &mut field);

        // Body should have moved (gravity)
        let pos = world.body(h).unwrap().position();
        assert!(pos.coord(1) < 10.0, "body should fall under gravity");
    }
}
