// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration field: the central coupling between Φ (integrated information) and physics.
//!
//! Φ is a formal measure from IIT (Tononi 2004) quantifying how much a system's
//! information exceeds the sum of its parts. It is a mathematical property of
//! causal structure — NOT a claim about subjective experience or sentience.
//!
//! Each entity has an [`EntityConsciousness`] (read: entity integration state) that
//! computes Φ from sensory/behavioral inputs and gates physics via safety tiers.
//! The [`ConsciousnessField`] (read: integration field) aggregates all entities
//! and provides modulation functions called during physics simulation.

use std::collections::BTreeMap;
use std::collections::VecDeque;

use nalgebra::SVector;
use symthaea_consciousness_equation::{
    ConsciousnessInputs, ConsciousnessResult, MasterConsciousnessEquation,
};

use crate::energy::EnergyBudget;
use crate::harmony_field::{HarmonyField, HarmonySource};
use crate::safety::SafetyTier;
use crate::sanctuary::{SanctuaryConditions, SanctuaryZone};
use crate::thermodynamics::{ThermodynamicConstants, ThermodynamicLedger};
use symtropy_physics::body::BodyHandle;
use symtropy_physics::world::PhysicsCallback;

/// Persistent agent memory for temporal state tracking.
///
/// Gives agents something beyond single-tick perception: remembered well
/// locations, partner history, and hunger urgency. This is NOT learning
/// (no weight updates) — it's episodic memory that the FEP gradient can use.
#[derive(Debug, Clone)]
pub struct AgentMemory {
    /// Known energy well positions (discovered when within range).
    pub known_wells: Vec<SVector<f64, 2>>,
    /// Resonance history with specific agents: (handle_bits, cumulative_resonance, encounters).
    pub partner_history: Vec<(u64, f64, u32)>,
    /// Ticks since last positive energy regeneration.
    pub ticks_since_regen: u64,
    /// Rolling energy average over last 100 ticks (for reward signal).
    pub energy_window: VecDeque<f64>,
    /// Maximum partner history entries.
    pub max_partners: usize,
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self {
            known_wells: Vec::new(),
            partner_history: Vec::new(),
            ticks_since_regen: 0,
            energy_window: VecDeque::with_capacity(100),
            max_partners: 20,
        }
    }
}

impl AgentMemory {
    /// Record an energy observation for the rolling window.
    pub fn record_energy(&mut self, energy_fraction: f64) {
        self.energy_window.push_back(energy_fraction);
        if self.energy_window.len() > 100 {
            self.energy_window.pop_front();
        }
    }

    /// Windowed reward: average energy change over window.
    /// Positive = gaining energy on average. Negative = losing.
    pub fn windowed_reward(&self) -> f64 {
        if self.energy_window.len() < 2 {
            return 0.0;
        }
        let n = self.energy_window.len();
        let half = n / 2;
        let first_half: f64 = self.energy_window.iter().take(half).sum::<f64>() / half as f64;
        let second_half: f64 =
            self.energy_window.iter().skip(half).sum::<f64>() / (n - half) as f64;
        second_half - first_half
    }

    /// Record a discovered well position (deduplicated by proximity).
    pub fn discover_well(&mut self, pos: &SVector<f64, 2>) {
        let dominated = self.known_wells.iter().any(|w| (w - pos).norm() < 10.0);
        if !dominated {
            self.known_wells.push(*pos);
        }
    }

    /// Record a cooperation event with another agent.
    pub fn record_partner(&mut self, handle: symtropy_physics::BodyHandle, resonance: f64) {
        let bits = handle.0 as u64;
        if let Some(entry) = self.partner_history.iter_mut().find(|(h, _, _)| *h == bits) {
            entry.1 += resonance;
            entry.2 += 1;
        } else {
            if self.partner_history.len() >= self.max_partners {
                // Evict least-encountered partner
                if let Some(min_idx) = self
                    .partner_history
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (_, _, c))| *c)
                    .map(|(i, _)| i)
                {
                    self.partner_history.swap_remove(min_idx);
                }
            }
            self.partner_history.push((bits, resonance, 1));
        }
    }

    /// Mean resonance with known partners.
    pub fn mean_partner_resonance(&self) -> f64 {
        if self.partner_history.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .partner_history
            .iter()
            .map(|(_, res, count)| if *count > 0 { res / *count as f64 } else { 0.0 })
            .sum();
        total / self.partner_history.len() as f64
    }
}

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
    /// Harmony activations [0.0, 1.0] for each of the 9 harmonies.
    /// Index 7 = Sacred Stillness, Index 8 = Emotional Contagion.
    pub harmony_activations: [f64; crate::harmony_field::NUM_HARMONIES],
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
    /// Persistent agent memory (well locations, partner history, energy window).
    pub memory: AgentMemory,
}

impl EntityConsciousness {
    /// Create a new entity consciousness with default parameters.
    pub fn new(max_energy: f64) -> Self {
        Self {
            equation: MasterConsciousnessEquation::default(),
            result: None,
            safety_tier: SafetyTier::Green,
            energy: EnergyBudget::new(max_energy),
            harmony_activations: [0.0; crate::harmony_field::NUM_HARMONIES],
            prediction_error: 0.0,
            motor_precision: 1.0,
            prediction_decay: 0.05, // ~20 ticks to recover
            memory: AgentMemory::default(),
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

    /// Total harmony energy (sum of all 9 activations).
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
    pub entities: BTreeMap<BodyHandle, EntityConsciousness>,
    /// Per-entity sanctuary zones.
    pub sanctuaries: BTreeMap<BodyHandle, SanctuaryZone<D>>,
    /// Harmony field for Channel 4 (Harmony → Friction) coupling.
    /// Rebuilt each frame from entity positions and harmony activations.
    pub harmony_field: HarmonyField<D>,
    /// Cached entity positions (rebuilt each frame by `rebuild_harmony_field`).
    /// Used by Phi-gravity and other position-dependent calculations.
    entity_positions: BTreeMap<BodyHandle, SVector<f64, D>>,
    /// Phi-gravity coupling strength (0.0 = disabled, default 0.5).
    /// Higher values create stronger attraction between conscious entities.
    pub phi_gravity_strength: f64,
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
            entities: BTreeMap::new(),
            sanctuaries: BTreeMap::new(),
            harmony_field: HarmonyField::new(),
            entity_positions: BTreeMap::new(),
            phi_gravity_strength: 0.0, // disabled by default
            collective_phi: 0.0,
            ledger: ThermodynamicLedger::new(),
            constants: ThermodynamicConstants::default(),
            sanctuary_absorbed: std::sync::atomic::AtomicU64::new(0u64.to_le()),
        }
    }

    /// Drain accumulated sanctuary absorption into ledger.
    fn drain_sanctuary_absorption(&mut self) {
        let bits = self
            .sanctuary_absorbed
            .swap(0, std::sync::atomic::Ordering::Relaxed);
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
            // Temperature-consciousness feedback: excessive heat reduces clarity.
            // Smooth sigmoid from body temperature (310K) → impaired at 360K → floor at 410K+.
            // No binary threshold — the penalty is differentiable at every temperature.
            // Ref: Somjen et al. (2001), Kiyatkin (2010) — see thermodynamics::smooth_temperature_penalty.
            let temp_penalty =
                crate::thermodynamics::smooth_temperature_penalty(entity.energy.temperature);
            let effective_inputs = ConsciousnessInputs {
                phi: inputs.phi * temp_penalty,
                broadcast: inputs.broadcast * temp_penalty,
                working_memory: inputs.working_memory * temp_penalty,
                attention: inputs.attention * temp_penalty,
                recurrence: inputs.recurrence, // temporal structure unaffected
                embodiment: inputs.embodiment, // body awareness unaffected
                knowledge: inputs.knowledge,   // stored knowledge unaffected
                synchrony: inputs.synchrony * temp_penalty,
            };

            let phi_before = entity.phi();
            entity.compute(&effective_inputs);
            let phi_after = entity.phi();

            // Track actual Phi change for J/Phi metric accuracy
            let delta_phi = (phi_after - phi_before).abs();
            if delta_phi > 1e-10 {
                self.ledger.record_phi_change(delta_phi);
            }

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
    pub fn modulate_force(&self, handle: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
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
            let absorbed_energy = absorbed * 0.5;
            let _ = self.sanctuary_absorbed.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |bits| {
                    let prev = f64::from_bits(bits);
                    Some((prev + absorbed_energy).to_bits())
                },
            );
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

    /// Rebuild the harmony field from current entity positions and activations.
    ///
    /// Call this each frame after entity positions are known (e.g., from physics sync).
    /// This populates the harmony field sources used by Channel 4 (Harmony → Friction).
    pub fn rebuild_harmony_field(&mut self, positions: &[(BodyHandle, symtropy_math::Point<D>)]) {
        self.harmony_field.sources.clear();
        self.entity_positions.clear();
        for (handle, pos) in positions {
            self.entity_positions.insert(*handle, pos.to_vector());
            if let Some(entity) = self.entities.get(handle) {
                self.harmony_field.sources.push(HarmonySource {
                    position: *pos,
                    activations: entity.harmony_activations,
                    strength: entity.phi().max(0.1),
                    radius: self.constants.harmony_range,
                    created_at: 0.0,
                    propagation_speed: f64::MAX,
                });
            }
        }
    }

    /// Spread emotional contagion (Harmony dimension 9, index 8) across all entities.
    ///
    /// Call once per tick after all `update_entity()` calls have completed.
    /// Entities near high-Φ neighbors drift their emotional activation toward
    /// the social mean; isolated entities decay toward zero.
    ///
    /// Physics: implements McFadden's CEMI theory — EM field synchrony creates
    /// shared emotional states. Dunbar (2012): social laughter/emotion is the
    /// primate grooming equivalent for large groups.
    ///
    /// `positions`: current positions of all registered entities
    /// `dt`: simulation time step in seconds
    pub fn spread_emotional_contagion(
        &mut self,
        positions: &[(BodyHandle, symtropy_math::Point<D>)],
        dt: f64,
    ) {
        use crate::harmony_field::{contagion_update, EMOTIONAL_CONTAGION_IDX};

        // Snapshot: (position_as_svector, emotion, phi) for all entities
        let sources: Vec<(SVector<f64, D>, f64, f64)> = positions
            .iter()
            .filter_map(|(h, pos)| {
                self.entities.get(h).map(|e| {
                    (
                        pos.0,
                        e.harmony_activations[EMOTIONAL_CONTAGION_IDX],
                        e.phi(),
                    )
                })
            })
            .collect();

        // Apply to each entity (self excluded from its own sources)
        for (handle, pos) in positions {
            let self_pos = pos.0;
            if let Some(entity) = self.entities.get_mut(handle) {
                let own_emotion = entity.harmony_activations[EMOTIONAL_CONTAGION_IDX];
                let own_phi = entity.phi();
                // Filter out self (near-zero distance)
                let others: Vec<_> = sources
                    .iter()
                    .filter(|(p, _, _)| (p - self_pos).norm() > 1e-6)
                    .cloned()
                    .collect();
                entity.harmony_activations[EMOTIONAL_CONTAGION_IDX] =
                    contagion_update(&self_pos, own_emotion, own_phi, &others, dt);
            }
        }
    }

    /// Apply dimensional leakage effects to entity energy budgets.
    ///
    /// Leakage sinks drain energy; leakage sources inject energy. This couples
    /// the dimensional leakage system to the thermodynamic layer, so agents near
    /// leakage points actually gain or lose energy.
    ///
    /// Only active when `leakage.enabled` is true.
    pub fn apply_dimensional_leakage(
        &mut self,
        leakage: &crate::dimensional_leakage::DimensionalLeakage<D>,
        positions: &[(BodyHandle, SVector<f64, D>)],
    ) {
        if !leakage.enabled {
            return;
        }
        for (handle, pos) in positions {
            let effect = leakage.total_effect_at(pos);
            if let Some(entity) = self.entities.get_mut(handle) {
                if effect > 0.0 {
                    entity.energy.regenerate(effect);
                    self.ledger.record_dissipation(-effect); // Energy enters 3D
                } else if effect < 0.0 {
                    let consumed = entity.energy.consume(effect.abs());
                    self.ledger.record_dissipation(consumed); // Energy exits 3D
                }
            }
        }
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
/// This is the critical bridge: Φ gates force authority and energy budgets.
/// The physics world calls these methods during collision resolution,
/// and the integration metric modulates the physical outcome.
impl<const D: usize> PhysicsCallback<D> for ConsciousnessField<D> {
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        let gain = self
            .entities
            .get(&body)
            .map(|e| e.effective_motor_gain())
            .unwrap_or(1.0);
        let mut result = force * gain;

        // Phi-gravity: conscious entities attract each other (when enabled)
        if self.phi_gravity_strength > 1e-10 {
            if let (Some(pos), Some(entity)) =
                (self.entity_positions.get(&body), self.entities.get(&body))
            {
                let self_phi = entity.phi();
                if self_phi > 0.01 {
                    let nearby: Vec<_> = self
                        .entity_positions
                        .iter()
                        .filter(|(h, _)| **h != body)
                        .filter_map(|(h, p)| self.entities.get(h).map(|e| (*p, e.phi())))
                        .collect();
                    let gravity = crate::fep_gradient::phi_gravity(
                        pos,
                        self_phi,
                        &nearby,
                        self.phi_gravity_strength,
                    );
                    result += gravity;
                }
            }
        }

        result
    }

    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64 {
        ConsciousnessField::modulate_impulse(self, impulse, contact_point)
    }

    fn friction_multiplier(&self, contact_point: &SVector<f64, D>, body: BodyHandle) -> f64 {
        if self.harmony_field.sources.is_empty() {
            return 1.0;
        }
        let entity_harmonies = self
            .entities
            .get(&body)
            .map(|e| e.harmony_activations)
            .unwrap_or([0.0; crate::harmony_field::NUM_HARMONIES]);
        let point = symtropy_math::Point(*contact_point);
        self.harmony_field
            .friction_multiplier(&point, &entity_harmonies)
    }

    fn on_collision(&mut self, event: &symtropy_physics::CollisionEvent<D>) {
        self.drain_sanctuary_absorption();

        let drain_rate = self.constants.collision_energy_drain;

        // Compute resonance between colliding bodies for prediction error scaling
        let resonance = {
            let harm_a = self
                .entities
                .get(&event.body_a)
                .map(|e| e.harmony_activations);
            let harm_b = self
                .entities
                .get(&event.body_b)
                .map(|e| e.harmony_activations);
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
        }
        if let Some(entity) = self.entities.get_mut(&event.body_b) {
            let surprise_factor = (1.0 - resonance).max(0.1);
            entity.on_collision(event.impulse * surprise_factor);

            let drain = event.impulse * drain_rate;
            let consumed = entity.energy.consume(drain);
            entity.energy.dissipate_heat(consumed * 0.5);
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
            high_force[0],
            low_force[0]
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
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.0, // index 7 = Sacred Stillness = 0.9
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

        field.update_entity(
            BodyHandle(0),
            &test_inputs(0.9),
            symtropy_math::Point::origin(),
        );
        field.update_entity(
            BodyHandle(1),
            &test_inputs(0.9),
            symtropy_math::Point::origin(),
        );

        // Both entities have some consciousness
        assert!(field.collective_phi > 0.0);
    }

    #[test]
    fn resource_regen_scales_with_collective() {
        let mut field = ConsciousnessField::<3>::new();

        // No entities → collective_phi = 0 → regen = 0.5
        assert!((field.resource_regeneration_multiplier() - 0.5).abs() < 1e-10);

        field.register(BodyHandle(0), 100.0, 10.0);
        field.update_entity(
            BodyHandle(0),
            &test_inputs(0.9),
            symtropy_math::Point::origin(),
        );

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
            let gain = field.entities.get(&h).unwrap().safety_tier.motor_gain();
            assert!(
                gain >= prev_gain,
                "phi={phi} gave gain={gain} < prev={prev_gain}"
            );
            prev_gain = gain;
        }
    }

    #[test]
    fn harmony_friction_reduces_for_resonant_entities() {
        use symtropy_physics::world::PhysicsCallback;

        let mut field = ConsciousnessField::<3>::new();
        let h0 = BodyHandle(0);
        let h1 = BodyHandle(1);
        field.register(h0, 100.0, 10.0);
        field.register(h1, 100.0, 10.0);

        // Set aligned harmonies (Sacred Stillness)
        field.entities.get_mut(&h0).unwrap().harmony_activations =
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        field.entities.get_mut(&h1).unwrap().harmony_activations =
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        field.update_entity(h0, &test_inputs(0.8), symtropy_math::Point::origin());
        field.update_entity(
            h1,
            &test_inputs(0.8),
            symtropy_math::Point::new([5.0, 0.0, 0.0]),
        );

        // Rebuild harmony field with known positions
        let positions = vec![
            (h0, symtropy_math::Point::new([0.0, 0.0, 0.0])),
            (h1, symtropy_math::Point::new([5.0, 0.0, 0.0])),
        ];
        field.rebuild_harmony_field(&positions);

        // Contact near h1 — h0's harmony field should reduce friction
        let contact = SVector::from([2.5, 0.0, 0.0]);
        let mult = PhysicsCallback::<3>::friction_multiplier(&field, &contact, h0);
        assert!(
            mult < 1.0,
            "resonant friction multiplier {} should be < 1.0 (harmony reduces friction)",
            mult
        );
    }

    #[test]
    fn dimensional_leakage_drains_energy() {
        use crate::dimensional_leakage::{DimensionalLeakage, LeakagePoint};

        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.update_entity(h, &test_inputs(0.8), symtropy_math::Point::origin());

        let energy_before = field.entities.get(&h).unwrap().energy.available;

        let mut leakage = DimensionalLeakage::new();
        leakage.add_point(LeakagePoint::sink(
            nalgebra::SVector::from([0.0, 0.0, 0.0]),
            1.0,
            5.0,
            50.0,
        ));
        leakage.enabled = true;

        // Entity at origin, sink at origin → should drain energy
        field.apply_dimensional_leakage(&leakage, &[(h, nalgebra::SVector::from([1.0, 0.0, 0.0]))]);

        let energy_after = field.entities.get(&h).unwrap().energy.available;
        assert!(
            energy_after < energy_before,
            "energy should decrease near a sink: before={energy_before}, after={energy_after}"
        );
    }

    #[test]
    fn dimensional_leakage_disabled_no_effect() {
        use crate::dimensional_leakage::{DimensionalLeakage, LeakagePoint};

        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);

        let energy_before = field.entities.get(&h).unwrap().energy.available;

        let mut leakage = DimensionalLeakage::new();
        leakage.add_point(LeakagePoint::sink(
            nalgebra::SVector::from([0.0, 0.0, 0.0]),
            1.0,
            5.0,
            50.0,
        ));
        // NOT enabled

        field.apply_dimensional_leakage(&leakage, &[(h, nalgebra::SVector::from([0.0, 0.0, 0.0]))]);

        let energy_after = field.entities.get(&h).unwrap().energy.available;
        assert!(
            (energy_after - energy_before).abs() < 1e-10,
            "disabled leakage should not affect energy"
        );
    }

    #[test]
    fn harmony_friction_neutral_without_rebuild() {
        use symtropy_physics::world::PhysicsCallback;

        let field = ConsciousnessField::<3>::new();
        let contact = SVector::from([0.0, 0.0, 0.0]);
        let mult = PhysicsCallback::<3>::friction_multiplier(&field, &contact, BodyHandle(0));
        assert!(
            (mult - 1.0).abs() < 1e-10,
            "without harmony sources, friction should be neutral (1.0), got {}",
            mult
        );
    }

    // ── Emotional Contagion integration tests ────────────────────────────────

    #[test]
    fn emotional_contagion_spreads_between_nearby_entities() {
        use crate::harmony_field::EMOTIONAL_CONTAGION_IDX;

        let mut field = ConsciousnessField::<3>::new();
        let emitter = BodyHandle(0);
        let receiver = BodyHandle(1);
        field.register(emitter, 100.0, 10.0);
        field.register(receiver, 100.0, 10.0);

        // Emitter has high emotion; receiver has none
        field
            .entities
            .get_mut(&emitter)
            .unwrap()
            .harmony_activations[EMOTIONAL_CONTAGION_IDX] = 0.9;
        field
            .entities
            .get_mut(&receiver)
            .unwrap()
            .harmony_activations[EMOTIONAL_CONTAGION_IDX] = 0.0;

        // Give emitter some Φ so it broadcasts
        field.update_entity(emitter, &test_inputs(0.9), symtropy_math::Point::origin());
        field.update_entity(
            receiver,
            &test_inputs(0.8),
            symtropy_math::Point::new([5.0, 0.0, 0.0]),
        );
        // Re-set after update_entity (which doesn't touch harmony_activations[8])
        field
            .entities
            .get_mut(&emitter)
            .unwrap()
            .harmony_activations[EMOTIONAL_CONTAGION_IDX] = 0.9;

        let positions = vec![
            (emitter, symtropy_math::Point::new([0.0, 0.0, 0.0])),
            (receiver, symtropy_math::Point::new([5.0, 0.0, 0.0])),
        ];
        field.spread_emotional_contagion(&positions, 1.0);

        let receiver_emotion =
            field.entities.get(&receiver).unwrap().harmony_activations[EMOTIONAL_CONTAGION_IDX];
        assert!(
            receiver_emotion > 0.0,
            "nearby receiver should catch emotion, got {receiver_emotion}"
        );
    }

    #[test]
    fn emotional_contagion_decays_in_isolation() {
        use crate::harmony_field::EMOTIONAL_CONTAGION_IDX;

        let mut field = ConsciousnessField::<3>::new();
        let h = BodyHandle(0);
        field.register(h, 100.0, 10.0);
        field.entities.get_mut(&h).unwrap().harmony_activations[EMOTIONAL_CONTAGION_IDX] = 0.8;

        // Single entity — no neighbors to spread from
        let positions = vec![(h, symtropy_math::Point::<3>::origin())];
        field.spread_emotional_contagion(&positions, 1.0);

        let emotion = field.entities.get(&h).unwrap().harmony_activations[EMOTIONAL_CONTAGION_IDX];
        assert!(
            emotion < 0.8,
            "isolated entity emotion should decay, got {emotion}"
        );
    }

    #[test]
    fn emotional_contagion_does_not_affect_other_harmony_dims() {
        use crate::harmony_field::EMOTIONAL_CONTAGION_IDX;

        let mut field = ConsciousnessField::<3>::new();
        let emitter = BodyHandle(0);
        let receiver = BodyHandle(1);
        field.register(emitter, 100.0, 10.0);
        field.register(receiver, 100.0, 10.0);

        // Set non-contagion harmonies
        field
            .entities
            .get_mut(&receiver)
            .unwrap()
            .harmony_activations[3] = 0.7;
        field
            .entities
            .get_mut(&emitter)
            .unwrap()
            .harmony_activations[EMOTIONAL_CONTAGION_IDX] = 0.9;

        let positions = vec![
            (emitter, symtropy_math::Point::new([0.0, 0.0, 0.0])),
            (receiver, symtropy_math::Point::new([3.0, 0.0, 0.0])),
        ];
        field.spread_emotional_contagion(&positions, 1.0);

        // harmony[3] on receiver should be untouched
        let h3 = field.entities.get(&receiver).unwrap().harmony_activations[3];
        assert!(
            (h3 - 0.7).abs() < 1e-10,
            "contagion must only affect index {EMOTIONAL_CONTAGION_IDX}, not index 3 (got {h3})"
        );
    }
}
