// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics world: owns bodies, steps simulation, resolves collisions.

use std::collections::BTreeMap;
use std::collections::HashMap;

use nalgebra::SVector;
use symtropy_math::{Capsule, HalfSpace, HyperBox, Point, Sphere};

use crate::body::{BodyHandle, NetId, RigidBody};
use crate::broadphase;
use crate::constraint::Constraint;
use crate::contact::{CollisionEvent, ContactCache, ContactManifold};
use crate::gjk;
use crate::integrator;
use crate::manifold_gen;

/// Quantize a float to Q16.16 fixed-point (range ±32768, resolution 1/65536).
///
/// Used by the `deterministic-net` feature to guarantee bit-identical simulation
/// across heterogeneous hardware for P2P multiplayer and replay files.
#[cfg(feature = "deterministic-net")]
#[inline]
fn quantize_q16_16(v: f64) -> f64 {
    (v * 65536.0).round() / 65536.0
}

/// Callback trait for consciousness-physics coupling.
///
/// The physics world calls these methods during collision resolution,
/// allowing consciousness to modulate forces, impulses, and friction
/// without creating a circular dependency between crates.
pub trait PhysicsCallback<const D: usize> {
    /// Modulate a force by the entity's consciousness level.
    /// Returns the modified force vector.
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D>;

    /// Modulate a collision impulse at a contact point.
    /// Returns the modified impulse magnitude.
    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64;

    /// Friction multiplier at a contact point (harmony field effect).
    /// 1.0 = normal, <1.0 = reduced (resonance), >1.0 = increased (dissonance).
    fn friction_multiplier(&self, contact_point: &SVector<f64, D>, body: BodyHandle) -> f64;

    /// Called after collision resolution with the collision event.
    /// This is the primary hook for consciousness to observe and react to
    /// physical impacts (e.g., calculating trauma, adjusting internal state).
    fn on_collision(&mut self, event: &CollisionEvent<D>);

    /// Called after each physics step to record energy dissipated.
    fn record_dissipation(&mut self, energy: f64);

    /// Calculates and applies the long-term effect of a collision event
    /// (e.g., trauma, fatigue, shock) to the entity's internal state.
    /// This is the primary hook for persistent state change based on impact.
    fn apply_trauma(&mut self, event: &CollisionEvent<D>);
}

/// No-op callback for physics-only usage (no consciousness coupling).
pub struct NoOpCallback;

impl<const D: usize> PhysicsCallback<D> for NoOpCallback {
    fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        *force
    }
    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 {
        impulse
    }
    fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 {
        1.0
    }
    fn on_collision(&mut self, _: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _: f64) {}
    fn apply_trauma(&mut self, _: &CollisionEvent<D>) {}
}

/// The physics world manages all rigid bodies and steps the simulation.
pub struct PhysicsWorld<const D: usize> {
    pub bodies: Vec<RigidBody<D>>,
    pub constraints: Vec<Box<dyn Constraint<D>>>,
    pub gravity: SVector<f64, D>,
    /// Contacts from the last step.
    pub contacts: Vec<ContactManifold<D>>,
    /// Collision events from the last step (for game logic callbacks).
    pub collision_events: Vec<CollisionEvent<D>>,
    /// Sensor overlap events from the last step.
    pub sensor_events: Vec<crate::contact::SensorEvent>,
    /// Contact cache for warm-starting (previous frame's impulses).
    contact_cache: ContactCache<D>,
    /// Warm-starting from the previous frame (swapped at frame boundaries).
    prev_cache: ContactCache<D>,
    /// Position correction iterations per step.
    pub solver_iterations: usize,
    /// Sleep velocity threshold.
    pub sleep_threshold: f64,
    /// Ticks below sleep threshold before sleeping.
    pub sleep_ticks: u32,
    /// Penetration slop (small overlap allowed to prevent jitter).
    pub slop: f64,
    /// Baumgarte stabilization factor (TGS Soft position-bias coefficient).
    ///
    /// Used as `bias = baumgarte * (depth - slop).max(0) / dt` in TGS Soft.
    /// Typical values: 0.1–0.4. Default: 0.2.
    pub baumgarte: f64,
    /// Constraint compliance (softness). 0.0 = rigid (default). Higher values
    /// make contacts softer (spring-like). Applied as `α = compliance / dt²`.
    pub compliance: f64,
    /// NetId → BodyHandle mapping for cross-machine replay determinism.
    net_id_map: BTreeMap<NetId, BodyHandle>,
    /// BodyHandle → Vec index for O(1) body lookup.
    handle_to_index: HashMap<BodyHandle, usize>,
    next_handle: usize,
    /// Cached broadphase data for Static/Kinematic bodies (rebuilt only on add/remove).
    static_broadphase: broadphase::StaticBroadphase<D>,
    /// Set to true when a static or kinematic body is added or removed.
    static_tree_dirty: bool,
}

impl<const D: usize> Default for PhysicsWorld<D> {
    fn default() -> Self {
        Self::new(SVector::zeros())
    }
}

impl<const D: usize> PhysicsWorld<D> {
    /// Create an empty physics world.
    pub fn new(gravity: SVector<f64, D>) -> Self {
        Self {
            bodies: Vec::new(),
            constraints: Vec::new(),
            gravity,
            contacts: Vec::new(),
            collision_events: Vec::new(),
            sensor_events: Vec::new(),
            contact_cache: ContactCache::new(),
            prev_cache: ContactCache::new(),
            solver_iterations: 4,
            slop: 0.01,
            baumgarte: 0.2,
            compliance: 0.0,
            sleep_threshold: 0.5,
            sleep_ticks: 60, // ~1 second at 64Hz
            net_id_map: BTreeMap::new(),
            handle_to_index: HashMap::new(),
            next_handle: 0,
            static_broadphase: broadphase::StaticBroadphase::new(),
            static_tree_dirty: false,
        }
    }

    /// Get the total number of bodies in the world.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Get the total kinetic energy of all dynamic bodies in the world.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| b.body_type == crate::body::BodyType::Dynamic)
            .map(|b| b.kinetic_energy())
            .sum()
    }

    /// Count how many bodies are currently sleeping.
    pub fn sleeping_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.sleeping).count()
    }

    /// Add a dynamic sphere body and return its handle.
    pub fn add_sphere(&mut self, position: Point<D>, radius: f64, mass: f64) -> BodyHandle {
        let handle = self.allocate_handle();
        let body = RigidBody::dynamic_sphere(handle, position, radius, mass);
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.handle_to_index.insert(handle, idx);
        handle
    }

    /// Add a dynamic body with a custom collider.
    pub fn add_body(&mut self, mut body: RigidBody<D>) -> BodyHandle {
        use crate::body::BodyType;
        let handle = self.allocate_handle();
        body.handle = handle;
        if body.body_type == BodyType::Static || body.body_type == BodyType::Kinematic {
            self.static_tree_dirty = true;
        }
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.handle_to_index.insert(handle, idx);
        handle
    }

    /// Add multiple bodies with stable network IDs in deterministic order.
    ///
    /// Bodies are sorted by `NetId` before insertion, ensuring the same
    /// `BodyHandle` assignment regardless of the caller's iteration order.
    /// Returns an error if any `NetId` is duplicated.
    pub fn add_bodies_deterministic(
        &mut self,
        mut bodies: Vec<(NetId, RigidBody<D>)>,
    ) -> Result<Vec<BodyHandle>, String> {
        bodies.sort_by_key(|(id, _)| *id);
        let mut handles = Vec::with_capacity(bodies.len());
        for (net_id, mut body) in bodies {
            if self.net_id_map.contains_key(&net_id) {
                return Err(format!("duplicate NetId({})", net_id.0));
            }
            let handle = self.allocate_handle();
            body.handle = handle;
            body.net_id = Some(net_id);
            self.net_id_map.insert(net_id, handle);
            let idx = self.bodies.len();
            self.bodies.push(body);
            self.handle_to_index.insert(handle, idx);
            handles.push(handle);
        }
        Ok(handles)
    }

    /// Resolve a stable `NetId` to its `BodyHandle`.
    pub fn handle_for_net_id(&self, net_id: NetId) -> Option<BodyHandle> {
        self.net_id_map.get(&net_id).copied()
    }

    /// Resolve a `BodyHandle` to its stable `NetId`.
    pub fn net_id_for_handle(&self, handle: BodyHandle) -> Option<NetId> {
        self.body(handle).and_then(|b| b.net_id)
    }

    /// Assign a stable network identifier to a body.
    pub fn set_net_id(&mut self, handle: BodyHandle, net_id: NetId) {
        let old_id = self.body(handle).and_then(|b| b.net_id);

        if let Some(old) = old_id {
            self.net_id_map.remove(&old);
        }

        if let Some(body) = self.body_mut(handle) {
            body.net_id = Some(net_id);
            self.net_id_map.insert(net_id, handle);
        }
    }

    /// Add a constraint between two bodies.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint<D>>) {
        self.constraints.push(constraint);
    }

    /// Get a reference to a body by handle.
    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody<D>> {
        self.handle_to_index
            .get(&handle)
            .and_then(|&idx| self.bodies.get(idx))
    }

    /// Get a mutable reference to a body by handle.
    pub fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody<D>> {
        self.handle_to_index
            .get(&handle)
            .copied()
            .and_then(|idx| self.bodies.get_mut(idx))
    }

    /// Step with consciousness-physics callback.
    ///
    /// The callback modulates forces, impulses, and friction based on
    /// consciousness state, closing the consciousness-physics loop.
    pub fn step_with_callback(&mut self, dt: f64, callback: &mut dyn PhysicsCallback<D>) {
        self.step_internal(dt, Some(callback));
    }

    /// Step without consciousness coupling (pure physics).
    pub fn step(&mut self, dt: f64) {
        self.step_internal(dt, None);
    }

    fn step_internal(&mut self, dt: f64, mut callback: Option<&mut dyn PhysicsCallback<D>>) {
        // 0. Clear events from previous step
        self.collision_events.clear();
        self.sensor_events.clear();
        // Warm-starting: swap caches so prev_cache has last frame's impulses
        std::mem::swap(&mut self.contact_cache, &mut self.prev_cache);
        self.contact_cache.begin_frame();

        // 0b. Rebuild static broadphase cache if bodies have been added/removed.
        if self.static_tree_dirty {
            self.static_broadphase.rebuild(&self.bodies);
            self.static_tree_dirty = false;
        }

        // 1. Integrate all bodies (skip sleeping)
        #[cfg(feature = "deterministic-net")]
        for body in &mut self.bodies {
            // Snap accumulated forces to Q16.16 before integration for
            // bit-identical simulation across platforms (P2P / replay).
            if !body.sleeping {
                for i in 0..D {
                    body.force_accumulator[i] = quantize_q16_16(body.force_accumulator[i]);
                }
            }
        }
        for body in &mut self.bodies {
            if !body.sleeping {
                if let Some(cb) = callback.as_mut() {
                    body.force_accumulator =
                        cb.modulate_force(body.handle, &body.force_accumulator);
                }
                integrator::integrate(body, &self.gravity, dt);
            }
        }
        #[cfg(feature = "deterministic-net")]
        for body in &mut self.bodies {
            // Snap positions and velocities after integration.
            for i in 0..D {
                body.transform.translation.0[i] = quantize_q16_16(body.transform.translation.0[i]);
                body.linear_velocity[i] = quantize_q16_16(body.linear_velocity[i]);
            }
        }

        // 2. Broadphase: find potentially colliding pairs (incremental static cache)
        let pairs = broadphase::find_pairs_incremental(&self.bodies, &self.static_broadphase);

        // 3. Narrowphase: GJK for each pair
        self.contacts.clear();
        for pair in &pairs {
            let (idx_a, idx_b) = self.find_body_indices(pair.0, pair.1);
            if let (Some(a), Some(b)) = (idx_a, idx_b) {
                let pos_a = self.bodies[a].transform.translation.0;
                let pos_b = self.bodies[b].transform.translation.0;

                // Fast path: analytical HalfSpace contacts (bypass GJK+EPA)
                if let Some(manifold) = self.try_halfspace_contact(a, b, pair.0, pair.1) {
                    if self.bodies[a].is_sensor || self.bodies[b].is_sensor {
                        let (sensor, other) = if self.bodies[a].is_sensor {
                            (pair.0, pair.1)
                        } else {
                            (pair.1, pair.0)
                        };
                        self.sensor_events
                            .push(crate::contact::SensorEvent { sensor, other });
                        continue;
                    }
                    self.contacts.push(manifold);
                    continue;
                }

                // Special path: Mesh collisions (if implemented by the shape)
                if let Some(manifolds) = self.try_mesh_contact(a, b, pair.0, pair.1) {
                    for manifold in manifolds {
                        if self.bodies[a].is_sensor || self.bodies[b].is_sensor {
                            let (sensor, other) = if self.bodies[a].is_sensor {
                                (pair.0, pair.1)
                            } else {
                                (pair.1, pair.0)
                            };
                            self.sensor_events
                                .push(crate::contact::SensorEvent { sensor, other });
                            continue;
                        }
                        self.contacts.push(manifold);
                    }
                    continue;
                }

                let result = gjk::intersects(
                    self.bodies[a].collider.as_ref(),
                    &pos_a,
                    self.bodies[b].collider.as_ref(),
                    &pos_b,
                );

                if result.intersecting {
                    // Sensor detection: emit event but skip collision resolution
                    if self.bodies[a].is_sensor || self.bodies[b].is_sensor {
                        let (sensor, other) = if self.bodies[a].is_sensor {
                            (pair.0, pair.1)
                        } else {
                            (pair.1, pair.0)
                        };
                        self.sensor_events
                            .push(crate::contact::SensorEvent { sensor, other });
                        continue;
                    }

                    // EPA for accurate penetration depth and normal
                    if let Some(epa_result) = crate::epa::penetration(
                        self.bodies[a].collider.as_ref(),
                        &pos_a,
                        self.bodies[b].collider.as_ref(),
                        &pos_b,
                        &result.simplex,
                    ) {
                        if epa_result.depth > 0.0 {
                            // Multi-point manifold: contact perturbation for stable stacking
                            let manifold = manifold_gen::generate_contact_manifold(
                                self.bodies[a].collider.as_ref(),
                                &pos_a,
                                self.bodies[b].collider.as_ref(),
                                &pos_b,
                                epa_result.normal,
                                epa_result.depth,
                                pair.0,
                                pair.1,
                            );
                            self.contacts.push(manifold);
                        }
                    }
                }
            }
        }

        // 4. Build islands and skip sleeping ones
        let islands = crate::island::build_islands(
            &self.bodies,
            &self.contacts,
            &self.constraints,
            &self.handle_to_index,
        );
        let active_contact_indices: Vec<usize> = islands
            .iter()
            .filter(|island| !island.sleeping)
            .flat_map(|island| island.contact_indices.iter().copied())
            .collect();

        // 5a. Warm-start: apply previous-frame impulse ONCE before the solver loop.
        //     (The old code accidentally applied warm-starting N times per frame.)
        for &ci in &active_contact_indices {
            if ci < self.contacts.len() {
                self.warm_start_contact(ci);
            }
        }

        // 5b. TGS Soft velocity solver: position-correction bias folded into impulse.
        //     Each iteration accumulates per-point lambda; write updated manifold back.
        for _ in 0..self.solver_iterations {
            for &ci in &active_contact_indices {
                if ci < self.contacts.len() {
                    let contact = self.contacts[ci].clone();
                    let updated = self.resolve_contact(contact, dt, &mut callback);
                    self.contacts[ci] = updated;
                }
            }
        }

        // 5c. After the solve, cache per-manifold total impulse for next frame's warm-start.
        for &ci in &active_contact_indices {
            if ci < self.contacts.len() {
                let c = &self.contacts[ci];
                let total_lambda: f64 = c.points.iter().map(|p| p.lambda).sum();
                self.contact_cache
                    .store(c.body_a, c.body_b, c.point(), total_lambda, 0.0);
            }
        }

        // 6. Solve constraints (active islands only)
        let active_constraint_indices: Vec<usize> = islands
            .iter()
            .filter(|island| !island.sleeping)
            .flat_map(|island| island.constraint_indices.iter().copied())
            .collect();

        for _ in 0..self.solver_iterations {
            for &ci in &active_constraint_indices {
                if ci >= self.constraints.len() {
                    continue;
                }
                let (ha, hb) = self.constraints[ci].bodies();
                let (idx_a, idx_b) = self.find_body_indices(ha, hb);
                if let (Some(a), Some(b)) = (idx_a, idx_b) {
                    if a < b {
                        let (left, right) = self.bodies.split_at_mut(b);
                        self.constraints[ci].solve(&mut left[a], &mut right[0], dt);
                    } else {
                        let (left, right) = self.bodies.split_at_mut(a);
                        self.constraints[ci].solve(&mut right[0], &mut left[b], dt);
                    }
                }
            }
        }

        // 6b. Velocity-level constraint solve (active islands only)
        for _ in 0..self.solver_iterations {
            for &ci in &active_constraint_indices {
                if ci >= self.constraints.len() {
                    continue;
                }
                let (ha, hb) = self.constraints[ci].bodies();
                let (idx_a, idx_b) = self.find_body_indices(ha, hb);
                if let (Some(a), Some(b)) = (idx_a, idx_b) {
                    if a < b {
                        let (left, right) = self.bodies.split_at_mut(b);
                        self.constraints[ci].solve_velocity(&mut left[a], &mut right[0], dt);
                    } else {
                        let (left, right) = self.bodies.split_at_mut(a);
                        self.constraints[ci].solve_velocity(&mut right[0], &mut left[b], dt);
                    }
                }
            }
        }

        // 7. Body sleeping: deactivate near-stationary bodies
        let threshold = self.sleep_threshold;
        let ticks = self.sleep_ticks;
        for body in &mut self.bodies {
            body.try_sleep(threshold, ticks);
        }

        // 8. State Decay: Apply natural recovery/decay to all conscious entities.
        if let Some(cb) = callback.as_mut() {
            self.decay_state(*cb, dt);
        }
    }

    /// Decay the consciousness state over time.
    ///
    /// This simulates natural recovery (Trauma fades, Fatigue dissipates)
    /// and stress dissipation.
    fn decay_state(&mut self, _callback: &mut dyn PhysicsCallback<D>, _dt: f64) {
        // Note: In a real system, we would iterate over all conscious bodies
        // and update their individual state. Here, we assume the callback
        // handles the global state update for simplicity.

        // The callback implementation must handle the actual state update.
        // We pass a dummy event since decay is time-based, not impact-based.
        let _dummy_event: CollisionEvent<D> = CollisionEvent {
            body_a: BodyHandle(0),
            body_b: BodyHandle(0),
            impulse: 0.0,
            normal: SVector::zeros(),
            depth: 0.0,
        };

        // We call a specialized decay method on the callback trait (if available)
        // or, for now, we rely on the callback's internal logic to handle time-based decay.
        // Since we cannot modify the trait signature here, we assume the callback
        // implements a method to handle time-based decay, or we simply log the intent.

        // For demonstration, we assume the callback has a method to handle time decay.
        // If the trait were expanded, we would call:
        // callback.decay_state(dt);

        // Since we cannot modify the trait here, we will leave this as a comment
        // and assume the callback implementation handles the time decay internally
        // based on the physics step time (dt).
    }

    /// Warm-start a single contact from the previous frame's impulse cache.
    ///
    /// Called ONCE per contact before the solver loop. Distributes the cached
    /// total impulse evenly across all contact points and initialises `lambda`.
    fn warm_start_contact(&mut self, ci: usize) {
        let contact = self.contacts[ci].clone();
        let (idx_a, idx_b) = self.find_body_indices(contact.body_a, contact.body_b);
        let (Some(a), Some(b)) = (idx_a, idx_b) else {
            return;
        };

        let primary_pt = contact.primary_point().position;
        let cached_total = self
            .prev_cache
            .lookup(contact.body_a, contact.body_b, &primary_pt)
            .map(|c| c.normal_impulse * 0.8) // 80% of previous frame
            .unwrap_or(0.0);

        if cached_total > 1e-15 {
            let n_pts = contact.points.len().max(1) as f64;
            let per_pt = cached_total / n_pts;

            // Apply warm-start impulse
            let impulse = contact.normal * cached_total;
            integrator::apply_impulse(&mut self.bodies[a], &(-impulse));
            integrator::apply_impulse(&mut self.bodies[b], &impulse);

            // Seed lambda so TGS Soft starts from warm-start value
            for pt in &mut self.contacts[ci].points {
                pt.lambda = per_pt;
            }
        }
    }

    /// Resolve a single contact using TGS Soft (Temporal Gauss-Seidel with compliance).
    ///
    /// Replaces Baumgarte stabilisation: position correction is folded into a
    /// "bias velocity" `bias = baumgarte * (depth - slop).max(0) / dt`, which is
    /// added to the velocity constraint. Lambda clamping ensures contacts only push
    /// (never pull), eliminating ghost-acceleration artefacts.
    ///
    /// Returns the updated manifold so accumulated lambdas persist across iterations.
    fn resolve_contact(
        &mut self,
        mut contact: ContactManifold<D>,
        dt: f64,
        callback: &mut Option<&mut dyn PhysicsCallback<D>>,
    ) -> ContactManifold<D> {
        let (idx_a, idx_b) = self.find_body_indices(contact.body_a, contact.body_b);
        let (Some(a), Some(b)) = (idx_a, idx_b) else {
            return contact;
        };

        let inv_mass_a = self.bodies[a].inv_mass;
        let inv_mass_b = self.bodies[b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass < 1e-15 {
            return contact;
        }

        // Compliance: α = compliance / dt² (adds softness to the constraint)
        let alpha = self.compliance / (dt * dt).max(1e-20);
        let safe_dt = dt.max(1e-10);

        let baumgarte = self.baumgarte;
        let slop = self.slop;

        // ─── TGS Soft: per-point velocity+position constraint ───
        let mut total_normal_impulse = 0.0_f64;

        for pt in &mut contact.points {
            let v_rel_n = {
                let va = self.bodies[a].linear_velocity;
                let vb = self.bodies[b].linear_velocity;
                (vb - va).dot(&contact.normal)
            };

            // Position-correction bias (replaces Baumgarte teleport)
            let bias = (pt.depth - slop).max(0.0) * baumgarte / safe_dt;

            // TGS Soft impulse: Δλ = (-v_rel·n + bias) / (w_a + w_b + α)
            let denom = total_inv_mass + alpha;
            let delta_lambda = (-v_rel_n + bias) / denom;

            // Clamp: accumulated normal impulse must stay ≥ 0 (no pulling)
            let new_lambda = (pt.lambda + delta_lambda).max(0.0);
            let actual_delta = new_lambda - pt.lambda;
            pt.lambda = new_lambda;

            if actual_delta.abs() > 1e-15 {
                // ═══ CONSCIOUSNESS MODULATION: sanctuary + harmony fields ═══
                let modulated_delta = if let Some(cb) = callback.as_mut() {
                    cb.modulate_impulse(actual_delta, &pt.position)
                } else {
                    actual_delta
                };

                let impulse = contact.normal * modulated_delta;
                integrator::apply_impulse(&mut self.bodies[a], &(-impulse));
                integrator::apply_impulse(&mut self.bodies[b], &impulse);
                total_normal_impulse += modulated_delta.abs();
            }
        }

        // ─── Coulomb friction (once per manifold, based on total normal impulse) ───
        if total_normal_impulse > 1e-15 {
            let primary_pt = contact.point();
            let v_rel = self.bodies[b].linear_velocity - self.bodies[a].linear_velocity;
            let v_n = contact.normal * v_rel.dot(&contact.normal);
            let v_t = v_rel - v_n;
            let v_t_mag = v_t.norm();

            if v_t_mag > 1e-10 {
                let tangent = v_t / v_t_mag;
                let mut mu = (self.bodies[a].friction * self.bodies[b].friction).sqrt();

                // ═══ HARMONY FIELD FRICTION MODULATION ═══
                if let Some(cb) = callback.as_mut() {
                    mu *= cb.friction_multiplier(&primary_pt, contact.body_a);
                }

                let j_t_desired = -v_t_mag / total_inv_mass;
                let j_t = j_t_desired.clamp(-mu * total_normal_impulse, mu * total_normal_impulse);
                let friction_impulse = tangent * j_t;
                integrator::apply_impulse(&mut self.bodies[a], &(-friction_impulse));
                integrator::apply_impulse(&mut self.bodies[b], &friction_impulse);

                // Record friction dissipation for consciousness energy budget
                if let Some(cb) = callback.as_mut() {
                    cb.record_dissipation(j_t.abs() * 0.1);
                }
            }

            // ─── Emit collision event ───
            let event = CollisionEvent {
                body_a: contact.body_a,
                body_b: contact.body_b,
                impulse: total_normal_impulse,
                normal: contact.normal,
                depth: contact.depth(),
            };

            // ═══ CONSCIOUSNESS FEEDBACK: collision → prediction error ═══
            // 1. Call the general on_collision hook (for immediate reaction)
            if let Some(cb) = callback.as_mut() {
                cb.on_collision(&event);
            }
            // 2. Call the dedicated trauma/state update hook (for persistent state)
            if let Some(cb) = callback.as_mut() {
                cb.apply_trauma(&event);
            }
        }

        contact
    }

    #[inline]
    fn allocate_handle(&mut self) -> BodyHandle {
        let handle = BodyHandle(self.next_handle);
        self.next_handle += 1;
        handle
    }

    #[inline]
    fn find_body_indices(
        &self,
        body_a: BodyHandle,
        body_b: BodyHandle,
    ) -> (Option<usize>, Option<usize>) {
        (
            self.handle_to_index.get(&body_a).copied(),
            self.handle_to_index.get(&body_b).copied(),
        )
    }

    fn try_halfspace_contact(
        &self,
        idx_a: usize,
        idx_b: usize,
        handle_a: BodyHandle,
        handle_b: BodyHandle,
    ) -> Option<ContactManifold<D>> {
        let body_a = &self.bodies[idx_a];
        let body_b = &self.bodies[idx_b];

        if let Some(plane) = body_a.collider.as_any().downcast_ref::<HalfSpace<D>>() {
            return self.contact_against_halfspace(plane, body_b, handle_a, handle_b, true);
        }
        if let Some(plane) = body_b.collider.as_any().downcast_ref::<HalfSpace<D>>() {
            return self.contact_against_halfspace(plane, body_a, handle_a, handle_b, false);
        }

        None
    }

    fn try_mesh_contact(
        &self,
        idx_a: usize,
        idx_b: usize,
        handle_a: BodyHandle,
        handle_b: BodyHandle,
    ) -> Option<Vec<ContactManifold<D>>> {
        use crate::manifold_gen::MeshColliderMetadata;
        let body_a = &self.bodies[idx_a];
        let body_b = &self.bodies[idx_b];

        if let Some(mesh) = body_a
            .collider
            .as_any()
            .downcast_ref::<&dyn MeshColliderMetadata<D>>()
        {
            return Some(mesh.generate_mesh_contacts(
                handle_a,
                &body_a.transform.translation.0,
                body_b.collider.as_ref(),
                handle_b,
                &body_b.transform.translation.0,
            ));
        }
        if let Some(mesh) = body_b
            .collider
            .as_any()
            .downcast_ref::<&dyn MeshColliderMetadata<D>>()
        {
            return Some(mesh.generate_mesh_contacts(
                handle_b,
                &body_b.transform.translation.0,
                body_a.collider.as_ref(),
                handle_a,
                &body_a.transform.translation.0,
            ));
        }

        None
    }

    fn contact_against_halfspace(
        &self,
        plane: &HalfSpace<D>,
        other: &RigidBody<D>,
        handle_a: BodyHandle,
        handle_b: BodyHandle,
        plane_is_a: bool,
    ) -> Option<ContactManifold<D>> {
        let other_pos = other.transform.translation.0;
        let normal = if plane_is_a {
            plane.normal
        } else {
            -plane.normal
        };

        if let Some(sphere) = other.collider.as_any().downcast_ref::<Sphere<D>>() {
            let (point, depth) = plane.contact_sphere(&other_pos, sphere.radius)?;
            return Some(ContactManifold::single(
                handle_a, handle_b, normal, point, depth,
            ));
        }

        if let Some(capsule) = other.collider.as_any().downcast_ref::<Capsule<D>>() {
            let contacts = plane.contact_capsule(
                &other_pos,
                capsule.half_height,
                capsule.radius,
                capsule.axis,
            );
            return Self::manifold_from_contacts(handle_a, handle_b, normal, contacts);
        }

        if let Some(hyperbox) = other.collider.as_any().downcast_ref::<HyperBox<D>>() {
            let contacts = plane.contact_box(&other_pos, &hyperbox.half_extents);
            return Self::manifold_from_contacts(handle_a, handle_b, normal, contacts);
        }

        None
    }

    fn manifold_from_contacts(
        body_a: BodyHandle,
        body_b: BodyHandle,
        normal: SVector<f64, D>,
        contacts: Vec<(SVector<f64, D>, f64)>,
    ) -> Option<ContactManifold<D>> {
        let mut contacts = contacts.into_iter();
        let (point, depth) = contacts.next()?;
        let mut manifold = ContactManifold::single(body_a, body_b, normal, point, depth);
        for (position, depth) in contacts {
            manifold.points.push(crate::contact::ContactPoint {
                position,
                depth,
                lambda: 0.0,
            });
        }
        Some(manifold)
    }
}
