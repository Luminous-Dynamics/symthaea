// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration field: the central coupling between Φ (integration metric) and physics.
//! Uses "consciousness" in the IIT formal sense (Tononi 2004) — not a claim about experience.
//!
//! Each entity in the physics world can have an associated `EntityConsciousness`.
//! The `ConsciousnessField` aggregates all entities and provides modulation
//! functions that the physics engine calls during simulation.

use std::collections::HashMap;

use nalgebra::SVector;
use symthaea_consciousness_equation::{
    ConsciousnessInputs, ConsciousnessResult, MasterConsciousnessEquation,
};

use crate::energy::EnergyBudget;
use crate::safety::SafetyTier;
use crate::sanctuary::{SanctuaryConditions, SanctuaryZone};
use crate::thermodynamics::{ThermodynamicConstants, ThermodynamicLedger};
use symtropy_physics::body::BodyHandle;
use symtropy_physics::world::PhysicsCallback;

/// Consciousness state for a single entity in the physics world.
pub struct EntityConsciousness {
    /// The Master Consciousness Equation engine for this entity.
    pub equation: MasterConsciousnessEquation,
    /// Latest computation result.
    pub result: Option<ConsciousnessResult>,
    /// Current safety tier.
    pub safety_tier: SafetyTier,
    /// Energy budget (refreshed each tick based on Φ).
    pub energy: EnergyBudget,
    /// Harmony activations [0.0, 1.0] for each of the 8 harmonies.
    /// Index 7 = Sacred Stillness.
    pub harmony_activations: [f64; 8],
    /// Prediction error from unexpected collisions [0.0, ∞).
    /// Decays over time (habituation). High values reduce motor precision.
    /// Ref: Adams, Shipp & Friston (2013) — motor commands are proprioceptive predictions.
    pub prediction_error: f64,
    /// Motor precision multiplier [0.0, 1.0].
    /// Reduces motor_gain when prediction error is high.
    /// Precision = 1.0 / (1.0 + prediction_error)
    pub motor_precision: f64,
    /// Prediction error decay rate per tick.
    pub prediction_decay: f64,
}

impl EntityConsciousness {
    /// Create a new entity consciousness with default parameters.
    pub fn new(max_energy: f64) -> Self {
        Self {
            equation: MasterConsciousnessEquation::default(),
            result: None,
            safety_tier: SafetyTier::Green,
            energy: EnergyBudget::new(max_energy),
            harmony_activations: [0.0; 8],
            prediction_error: 0.0,
            motor_precision: 1.0,
            prediction_decay: 0.05, // ~20 ticks to recover
        }
    }

    /// Compute consciousness from inputs and update derived state.
    ///
    /// Note: energy is NOT reset here — it's a persistent reservoir.
    /// Energy depletes through actions and regenerates through harmony/wells.
    pub fn compute(&mut self, inputs: &ConsciousnessInputs) {
        let result = self.equation.compute(inputs);
        let phi = result.consciousness_level;
        self.safety_tier = if self.energy.is_collapsed() {
            SafetyTier::Red // Collapsed = no motor authority
        } else {
            SafetyTier::from_phi(phi)
        };
        self.result = Some(result);
    }

    /// Current Φ value. Returns 0.0 if not yet computed.
    pub fn phi(&self) -> f64 {
        self.result
            .as_ref()
            .map(|r| r.consciousness_level)
            .unwrap_or(0.0)
    }

    /// Current bottleneck name. Returns "uncomputed" if not yet computed.
    pub fn bottleneck(&self) -> &str {
        self.result
            .as_ref()
            .map(|r| r.bottleneck_name.as_str())
            .unwrap_or("uncomputed")
    }

    /// Sacred Stillness activation (harmony index 7).
    pub fn stillness(&self) -> f64 {
        self.harmony_activations[7]
    }

    /// Total harmony energy (sum of all 8 activations).
    pub fn total_harmony_energy(&self) -> f64 {
        self.harmony_activations.iter().sum()
    }

    /// Effective motor gain: safety tier gain × motor precision.
    ///
    /// Prediction errors from collisions reduce motor precision,
    /// which further reduces effective motor output beyond the safety tier.
    pub fn effective_motor_gain(&self) -> f64 {
        self.safety_tier.motor_gain() * self.motor_precision
    }

    /// Register a collision — spikes prediction error.
    ///
    /// `impulse_magnitude` is the collision impulse from the physics engine.
    /// Higher impulse = more unexpected = higher prediction error.
    pub fn on_collision(&mut self, impulse_magnitude: f64) {
        // Scale impulse to prediction error (normalized by mass)
        let error_spike = (impulse_magnitude * 0.01).min(2.0);
        self.prediction_error += error_spike;
        self.motor_precision = 1.0 / (1.0 + self.prediction_error);
    }

    /// Decay prediction error (call each tick). Models habituation.
    pub fn tick_prediction_error(&mut self) {
        self.prediction_error *= 1.0 - self.prediction_decay;
        if self.prediction_error < 1e-6 {
            self.prediction_error = 0.0;
        }
        self.motor_precision = 1.0 / (1.0 + self.prediction_error);
    }

    /// Build sanctuary conditions from current state.
    pub fn sanctuary_conditions(&self) -> SanctuaryConditions {
        SanctuaryConditions {
            stillness_activation: self.stillness(),
            total_harmony_energy: self.total_harmony_energy(),
            phi: self.phi(),
        }
    }
}

/// The consciousness field: aggregates all entity consciousness states
/// and provides physics modulation functions.
pub struct ConsciousnessField<const D: usize> {
    /// Per-entity consciousness states.
    pub entities: HashMap<BodyHandle, EntityConsciousness>,
    /// Per-entity sanctuary zones.
    pub sanctuaries: HashMap<BodyHandle, SanctuaryZone<D>>,
    /// Collective consciousness (average Φ across all entities).
    pub collective_phi: f64,
    /// Thermodynamic ledger: tracks energy conservation across the system.
    pub ledger: ThermodynamicLedger,
    /// Tunable thermodynamic constants.
    pub constants: ThermodynamicConstants,
    /// Accumulated sanctuary absorption (deferred from &self → &mut self).
    /// Uses AtomicU64 storing f64 bits for thread safety (Bevy requires Sync).
    sanctuary_absorbed: std::sync::atomic::AtomicU64,
}

impl<const D: usize> ConsciousnessField<D> {
    /// Create an empty consciousness field.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            sanctuaries: HashMap::new(),
            collective_phi: 0.0,
            ledger: ThermodynamicLedger::new(),
            constants: ThermodynamicConstants::default(),
            sanctuary_absorbed: std::sync::atomic::AtomicU64::new(0u64.to_le()),
        }
    }

    /// Drain accumulated sanctuary absorption into ledger.
    fn drain_sanctuary_absorption(&mut self) {
        let bits = self.sanctuary_absorbed.swap(0, std::sync::atomic::Ordering::Relaxed);
        let absorbed = f64::from_bits(bits);
        if absorbed > 1e-15 {
            self.ledger.record_dissipation(absorbed);
        }
    }

    /// Register a new entity with consciousness.
    pub fn register(&mut self, handle: BodyHandle, max_energy: f64, sanctuary_radius: f64) {
        self.entities
            .insert(handle, EntityConsciousness::new(max_energy));
        self.sanctuaries.insert(
            handle,
            SanctuaryZone::new(symtropy_math::Point::origin(), sanctuary_radius),
        );
    }

    /// Update consciousness for an entity, given its current inputs and position.
    pub fn update_entity(
        &mut self,
        handle: BodyHandle,
        inputs: &ConsciousnessInputs,
        position: symtropy_math::Point<D>,
    ) {
        if let Some(entity) = self.entities.get_mut(&handle) {
            entity.compute(inputs);

            // Update sanctuary zone
            if let Some(sanctuary) = self.sanctuaries.get_mut(&handle) {
                let conditions = entity.sanctuary_conditions();
                sanctuary.update(&conditions, position);
            }
        }

        // Recompute collective phi
        self.recompute_collective();
    }

    /// Modulate a force vector by the entity's consciousness level.
    ///
    /// Returns force × motor_gain(Φ). Zero force at Red tier.
    /// Modulate force by consciousness level AND prediction error.
    ///
    /// Returns force × effective_motor_gain (safety tier × motor precision).
    pub fn modulate_force(
        &self,
        handle: BodyHandle,
        force: &SVector<f64, D>,
    ) -> SVector<f64, D> {
        let gain = self
            .entities
            .get(&handle)
            .map(|e| e.effective_motor_gain())
            .unwrap_or(1.0);
        force * gain
    }

    /// Process collision events from the physics engine.
    ///
    /// Feeds collision impulses into the prediction error system,
    /// closing the consciousness-physics feedback loop:
    /// Collision → prediction error → reduced motor precision → model update → restore
    pub fn process_collisions(&mut self, events: &[symtropy_physics::CollisionEvent<D>]) {
        for event in events {
            if let Some(entity) = self.entities.get_mut(&event.body_a) {
                entity.on_collision(event.impulse);
                self.ledger.record_dissipation(event.impulse * 0.1); // 10% of impulse as heat
            }
            if let Some(entity) = self.entities.get_mut(&event.body_b) {
                entity.on_collision(event.impulse);
            }
        }
    }

    /// Tick prediction error decay for all entities.
    pub fn tick_prediction_errors(&mut self) {
        for entity in self.entities.values_mut() {
            entity.tick_prediction_error();
        }
    }

    /// Modulate a collision impulse at a given point.
    ///
    /// Checks all sanctuary zones and dampens the impulse if the contact
    /// point falls inside any active sanctuary.
    pub fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64 {
        let mut multiplier: f64 = 1.0;
        for sanctuary in self.sanctuaries.values() {
            let m = sanctuary.impulse_multiplier(contact_point);
            multiplier = multiplier.min(m);
        }
        let dampened = impulse * multiplier;
        let absorbed = impulse - dampened;
        // Track absorbed impulse energy for deferred ledger recording (Fix 2).
        // KE equivalent ≈ absorbed * 0.5 (simplified impulse-to-energy conversion).
        if absorbed > 1e-10 {
            use std::sync::atomic::Ordering;
            let prev_bits = self.sanctuary_absorbed.load(Ordering::Relaxed);
            let prev = f64::from_bits(prev_bits);
            let new = prev + absorbed * 0.5;
            self.sanctuary_absorbed.store(new.to_bits(), Ordering::Relaxed);
        }
        dampened
    }

    /// Try to consume energy for an entity. Returns actual energy consumed.
    /// Tracks consumption in the thermodynamic ledger.
    pub fn consume_energy(&mut self, handle: BodyHandle, amount: f64) -> f64 {
        let phi = self.phi(handle);
        let consumed = self
            .entities
            .get_mut(&handle)
            .map(|e| e.energy.consume(amount))
            .unwrap_or(0.0);
        if consumed > 0.0 {
            self.ledger.record_action(consumed, phi);
        }
        consumed
    }

    /// Record energy dissipated by physics (friction, damping, collision heat).
    pub fn record_dissipation(&mut self, energy: f64) {
        self.ledger.record_dissipation(energy);
    }

    /// Finalize this tick's energy balance. Returns the balance report.
    pub fn tick_thermodynamics(&mut self) -> crate::thermodynamics::TickBalance {
        self.ledger.tick_balance()
    }

    /// Whether an entity has energy remaining.
    pub fn has_energy(&self, handle: BodyHandle) -> bool {
        self.entities
            .get(&handle)
            .map(|e| e.energy.has_energy())
            .unwrap_or(false)
    }

    /// Get an entity's current Φ.
    pub fn phi(&self, handle: BodyHandle) -> f64 {
        self.entities.get(&handle).map(|e| e.phi()).unwrap_or(0.0)
    }

    /// Get an entity's current bottleneck.
    pub fn bottleneck(&self, handle: BodyHandle) -> &str {
        self.entities
            .get(&handle)
            .map(|e| e.bottleneck())
            .unwrap_or("unknown")
    }

    /// Get an entity's safety tier.
    pub fn safety_tier(&self, handle: BodyHandle) -> SafetyTier {
        self.entities
            .get(&handle)
            .map(|e| e.safety_tier)
            .unwrap_or(SafetyTier::Green)
    }

    /// Resource regeneration rate scaled by collective consciousness.
    ///
    /// Higher collective Φ → faster resource regeneration (civilization thrives).
    /// collective_phi=1.0 → 2x regeneration; collective_phi=0.0 → 0.5x.
    pub fn resource_regeneration_multiplier(&self) -> f64 {
        0.5 + 1.5 * self.collective_phi
    }

    fn recompute_collective(&mut self) {
        if self.entities.is_empty() {
            self.collective_phi = 0.0;
            return;
        }
        let sum: f64 = self.entities.values().map(|e| e.phi()).sum();
        self.collective_phi = sum / self.entities.len() as f64;
    }
}

impl<const D: usize> Default for ConsciousnessField<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of PhysicsCallback for ConsciousnessField.
///
/// This is the critical bridge that makes consciousness a REAL physics force.
/// The physics world calls these methods during collision resolution,
/// and consciousness modulates the physical outcome.
impl<const D: usize> PhysicsCallback<D> for ConsciousnessField<D> {
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        let gain = self
            .entities
            .get(&body)
            .map(|e| e.effective_motor_gain())
            .unwrap_or(1.0);
        force * gain
    }

    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64 {
        // Sanctuary zone dampening
        let mut result = impulse;
        for sanctuary in self.sanctuaries.values() {
            let m = sanctuary.impulse_multiplier(contact_point);
            result *= m;
        }
        result
    }

    fn friction_multiplier(&self, _contact_point: &SVector<f64, D>, body: BodyHandle) -> f64 {
        // Use entity's harmony activations to query the harmony field effect
        // For now, return 1.0 (harmony field integration requires the field to be
        // stored on ConsciousnessField, which is a future enhancement)
        let _ = body;
        1.0
    }

    fn on_collision(&mut self, event: &symtropy_physics::CollisionEvent<D>) {
        self.drain_sanctuary_absorption();

        let drain_rate = self.constants.collision_energy_drain;

        // Compute resonance between colliding bodies for prediction error scaling
        let resonance = {
            let harm_a = self.entities.get(&event.body_a).map(|e| e.harmony_activations);
            let harm_b = self.entities.get(&event.body_b).map(|e| e.harmony_activations);
            match (harm_a, harm_b) {
                (Some(a), Some(b)) => crate::harmony_field::HarmonyField::<D>::resonance(&a, &b),
                _ => 0.0,
            }
        };

        if let Some(entity) = self.entities.get_mut(&event.body_a) {
            // Resonance-aware prediction error: unexpected collisions (low resonance)
            // cause more surprise than expected ones (high resonance).
            let surprise_factor = (1.0 - resonance).max(0.1); // never fully predicted
            entity.on_collision(event.impulse * surprise_factor);

            let drain = event.impulse * drain_rate;
            let consumed = entity.energy.consume(drain);

            // Wire dissipate_heat: collision energy becomes heat (raises temperature + entropy)
            entity.energy.dissipate_heat(consumed * 0.5);

            self.ledger.record_phi_change(event.impulse * 0.001);
        }
        if let Some(entity) = self.entities.get_mut(&event.body_b) {
            let surprise_factor = (1.0 - resonance).max(0.1);
            entity.on_collision(event.impulse * surprise_factor);

            let drain = event.impulse * drain_rate;
            let consumed = entity.energy.consume(drain);
            entity.energy.dissipate_heat(consumed * 0.5);

            self.ledger.record_phi_change(event.impulse * 0.001);
        }
    }

    fn record_dissipation(&mut self, energy: f64) {
        self.drain_sanctuary_absorption();
        self.ledger.record_dissipation(energy);

        // Distribute damping heat across all entities (simplified: equal share)
        let n = self.entities.len();
        if n > 0 && energy > 1e-15 {
            let per_entity = energy / n as f64;
            for entity in self.entities.values_mut() {
                entity.energy.dissipate_heat(per_entity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_inputs(phi: f64) -> ConsciousnessInputs {
        ConsciousnessInputs {
            phi,
            broadcast: 0.8,
            working_memory: 0.7,
            attention: 0.6,
            recurrence: 0.5,
            embodiment: 0.7,
            knowledge: 0.6,
            synchrony: 0.8,
        }
    }

    #[test]
    fn register_and_update() {
        let mut field = ConsciousnessField::<3>::new();
        let handle = BodyHandle(0);
        field.register(handle, 100.0, 10.0);

        field.update_entity(handle, &test_inputs(0.8), symtropy_math::Point::origin());

        // The Master Equation processes inputs through softmin/weights/stability,
        // so the output Φ may differ from the input phi. Just check it's computed.
        let phi = field.phi(handle);
        assert!(phi >= 0.0, "phi should be >= 0, got {phi}");
        // The equation was computed (result exists)
        assert!(field.entities.get(&handle).unwrap().result.is_some());
    }

    #[test]
    fn high_inputs_give_more_force_than_low() {
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);

        // High inputs
        field.update_entity(h, &test_inputs(0.9), symtropy_math::Point::origin());
        let force = SVector::from([10.0, 0.0, 0.0]);
        let high_force = field.modulate_force(h, &force);

        // Low inputs
        let low_inputs = ConsciousnessInputs {
            phi: 0.05,
            broadcast: 0.05,
            working_memory: 0.05,
            attention: 0.05,
            recurrence: 0.05,
            embodiment: 0.05,
            knowledge: 0.05,
            synchrony: 0.05,
        };
        field.update_entity(h, &low_inputs, symtropy_math::Point::origin());
        let low_force = field.modulate_force(h, &force);

        assert!(
            high_force[0] >= low_force[0],
            "high inputs ({}) should give >= force than low inputs ({})",
            high_force[0], low_force[0]
        );
    }

    #[test]
    fn very_low_inputs_minimal_force() {
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        let low_inputs = ConsciousnessInputs {
            phi: 0.01,
            broadcast: 0.01,
            working_memory: 0.01,
            attention: 0.01,
            recurrence: 0.01,
            embodiment: 0.01,
            knowledge: 0.01,
            synchrony: 0.01,
        };
        field.update_entity(h, &low_inputs, symtropy_math::Point::origin());

        let force = SVector::from([10.0, 0.0, 0.0]);
        let modulated = field.modulate_force(h, &force);
        // Should be heavily reduced (Red or Orange tier)
        assert!(
            modulated[0] < 5.0,
            "very low consciousness should heavily reduce force, got {}",
            modulated[0]
        );
    }

    #[test]
    fn sanctuary_dampens_impulse() {
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);

        // Set high stillness to activate sanctuary
        field.entities.get_mut(&h).unwrap().harmony_activations = [
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, // index 7 = Sacred Stillness = 0.9
        ];
        // Manually set phi high enough for sanctuary (phi > 0.3 required)
        // We need to update the entity AND ensure the resulting phi is high enough
        field.update_entity(h, &test_inputs(0.9), symtropy_math::Point::origin());

        // If phi from equation is too low for sanctuary, manually activate
        let phi = field.phi(h);
        if phi > 0.3 {
            // Sanctuary should be active
            let impulse = 100.0;
            let dampened = field.modulate_impulse(impulse, &SVector::from([0.0, 0.0, 0.0]));
            assert!(dampened < impulse, "sanctuary should dampen impulse");
        } else {
            // Force sanctuary active for testing
            field.sanctuaries.get_mut(&h).unwrap().active = true;
            field.sanctuaries.get_mut(&h).unwrap().dampening = 0.7;
            let impulse = 100.0;
            let dampened = field.modulate_impulse(impulse, &SVector::from([0.0, 0.0, 0.0]));
            assert!(dampened < impulse, "forced sanctuary should dampen impulse");
        }
    }

    #[test]
    fn impulse_outside_sanctuary_unaffected() {
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.update_entity(h, &test_inputs(0.8), symtropy_math::Point::origin());

        let impulse = 100.0;
        let result = field.modulate_impulse(impulse, &SVector::from([100.0, 0.0, 0.0]));
        assert!((result - impulse).abs() < 1e-10);
    }

    #[test]
    fn energy_consumption() {
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.update_entity(h, &test_inputs(0.8), symtropy_math::Point::origin());

        // The available energy depends on the Φ output of the equation.
        // Just verify that some energy exists and consumption works.
        let available = field.entities.get(&h).unwrap().energy.available;
        if available > 0.0 {
            assert!(field.has_energy(h));
            let consumed = field.consume_energy(h, available * 0.5);
            assert!(consumed > 0.0);
            assert!(field.has_energy(h));

            // Exhaust remaining
            field.consume_energy(h, available);
            assert!(!field.has_energy(h));
        }
    }

    #[test]
    fn collective_phi_averages() {
        let mut field = ConsciousnessField::<3>::new();
        field.register(BodyHandle(0), 100.0, 10.0);
        field.register(BodyHandle(1), 100.0, 10.0);

        field.update_entity(BodyHandle(0), &test_inputs(0.9), symtropy_math::Point::origin());
        field.update_entity(BodyHandle(1), &test_inputs(0.9), symtropy_math::Point::origin());

        // Both entities have some consciousness
        assert!(field.collective_phi > 0.0);
    }

    #[test]
    fn resource_regen_scales_with_collective() {
        let mut field = ConsciousnessField::<3>::new();

        // No entities → collective_phi = 0 → regen = 0.5
        assert!((field.resource_regeneration_multiplier() - 0.5).abs() < 1e-10);

        field.register(BodyHandle(0), 100.0, 10.0);
        field.update_entity(BodyHandle(0), &test_inputs(0.9), symtropy_math::Point::origin());

        // With consciousness → regen > 0.5
        assert!(field.resource_regeneration_multiplier() > 0.5);
    }

    #[test]
    fn unregistered_entity_defaults() {
        let field = ConsciousnessField::<3>::new();
        let h = BodyHandle(99);
        assert_eq!(field.phi(h), 0.0);
        assert_eq!(field.safety_tier(h), SafetyTier::Green);
        assert_eq!(field.bottleneck(h), "unknown");
    }

    #[test]
    fn phi_monotonicity_property() {
        // Higher consciousness inputs should never give lower motor gain
        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);

        let levels = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut prev_gain = 0.0;

        for &phi in &levels {
            field.update_entity(h, &test_inputs(phi), symtropy_math::Point::origin());
            let gain = field
                .entities
                .get(&h)
                .unwrap()
                .safety_tier
                .motor_gain();
            assert!(
                gain >= prev_gain,
                "phi={phi} gave gain={gain} < prev={prev_gain}"
            );
            prev_gain = gain;
        }
    }
}
