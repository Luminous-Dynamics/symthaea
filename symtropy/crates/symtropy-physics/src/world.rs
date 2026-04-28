// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics world: owns bodies, steps simulation, resolves collisions.

use std::collections::BTreeMap;
use std::collections::HashMap;

use nalgebra::SVector;
use symtropy_math::Point;

use crate::body::{BodyHandle, NetId, RigidBody};
use crate::broadphase;
use crate::contact::{CollisionEvent, ContactCache, ContactManifold};
use crate::constraint::Constraint;
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
    fn on_collision(&mut self, event: &CollisionEvent<D>);

    /// Called after each physics step to record energy dissipated.
    fn record_dissipation(&mut self, energy: f64);
}

/// No-op callback for physics-only usage (no consciousness coupling).
pub struct NoOpCallback;

impl<const D: usize> PhysicsCallback<D> for NoOpCallback {
    fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> { *force }
    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 { impulse }
    fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 { 1.0 }
    fn on_collision(&mut self, _: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _: f64) {}
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

    /// Add a dynamic sphere body and return its handle.
    pub fn add_sphere(
        &mut self,
        position: Point<D>,
        radius: f64,
        mass: f64,
    ) -> BodyHandle {
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
                if let Some(ref cb) = callback {
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
                        self.sensor_events.push(crate::contact::SensorEvent { sensor, other });
                        continue;
                    }
                    self.contacts.push(manifold);
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
                        self.sensor_events.push(crate::contact::SensorEvent { sensor, other });
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
        let active_contact_indices: Vec<usize> = islands.iter()
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
                self.contact_cache.store(
                    c.body_a, c.body_b, c.point(), total_lambda, 0.0,
                );
            }
        }

        // 6. Solve constraints (active islands only)
        let active_constraint_indices: Vec<usize> = islands.iter()
            .filter(|island| !island.sleeping)
            .flat_map(|island| island.constraint_indices.iter().copied())
            .collect();

        for _ in 0..self.solver_iterations {
            for &ci in &active_constraint_indices {
                if ci >= self.constraints.len() { continue; }
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
                if ci >= self.constraints.len() { continue; }
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
    }

    /// Warm-start a single contact from the previous frame's impulse cache.
    ///
    /// Called ONCE per contact before the solver loop. Distributes the cached
    /// total impulse evenly across all contact points and initialises `lambda`.
    fn warm_start_contact(&mut self, ci: usize) {
        let contact = self.contacts[ci].clone();
        let (idx_a, idx_b) = self.find_body_indices(contact.body_a, contact.body_b);
        let (Some(a), Some(b)) = (idx_a, idx_b) else { return; };

        let primary_pt = contact.primary_point().position;
        let cached_total = self.prev_cache
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
                (va - vb).dot(&contact.normal)
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
                let modulated_delta = if let Some(ref cb) = callback {
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
                if let Some(ref cb) = callback {
                    mu *= cb.friction_multiplier(&primary_pt, contact.body_a);
                }

                let j_t_desired = -v_t_mag / total_inv_mass;
                let j_t = j_t_desired.clamp(-mu * total_normal_impulse, mu * total_normal_impulse);
                let friction_impulse = tangent * j_t;
                integrator::apply_impulse(&mut self.bodies[a], &(-friction_impulse));
                integrator::apply_impulse(&mut self.bodies[b], &friction_impulse);

                // Record friction dissipation for consciousness energy budget
                if let Some(ref mut cb) = callback {
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
            if let Some(ref mut cb) = callback {
                cb.on_collision(&event);
            }

            self.collision_events.push(event);

            // Wake both bodies
            self.bodies[a].wake();
            self.bodies[b].wake();
        }

        contact
    }

    /// Try analytical HalfSpace contact. Returns None if neither body is a HalfSpace.
    fn try_halfspace_contact(
        &self,
        idx_a: usize,
        idx_b: usize,
        handle_a: BodyHandle,
        handle_b: BodyHandle,
    ) -> Option<ContactManifold<D>> {
        use symtropy_math::HalfSpace;

        // Check if body A is a HalfSpace
        if let Some(hs) = self.bodies[idx_a].collider.as_any().downcast_ref::<HalfSpace<D>>() {
            let (center_b, radius_b) = self.bodies[idx_b].collider.bounding_sphere();
            let world_center_b = self.bodies[idx_b].transform.transform_point(&center_b).0;
            if let Some((contact_pt, depth)) = hs.contact_sphere(&world_center_b, radius_b) {
                return Some(ContactManifold::single(
                    handle_a, handle_b, hs.normal, contact_pt, depth,
                ));
            }
            return None;
        }

        // Check if body B is a HalfSpace
        if let Some(hs) = self.bodies[idx_b].collider.as_any().downcast_ref::<HalfSpace<D>>() {
            let (center_a, radius_a) = self.bodies[idx_a].collider.bounding_sphere();
            let world_center_a = self.bodies[idx_a].transform.transform_point(&center_a).0;
            if let Some((contact_pt, depth)) = hs.contact_sphere(&world_center_a, radius_a) {
                return Some(ContactManifold::single(
                    handle_a, handle_b, -hs.normal, contact_pt, depth,
                ));
            }
            return None;
        }

        None // Neither body is a HalfSpace
    }

    fn find_body_indices(&self, a: BodyHandle, b: BodyHandle) -> (Option<usize>, Option<usize>) {
        (
            self.handle_to_index.get(&a).copied(),
            self.handle_to_index.get(&b).copied(),
        )
    }

    fn allocate_handle(&mut self) -> BodyHandle {
        let h = BodyHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Number of bodies in the world.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Total kinetic energy across all bodies.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(|b| b.kinetic_energy()).sum()
    }

    /// Collision events from the last step.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent<D>> {
        std::mem::take(&mut self.collision_events)
    }

    /// Number of sleeping bodies.
    pub fn sleeping_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.sleeping).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ForceScaleCallback {
        gain: f64,
    }

    impl<const D: usize> PhysicsCallback<D> for ForceScaleCallback {
        fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
            force * self.gain
        }

        fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 {
            impulse
        }

        fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 {
            1.0
        }

        fn on_collision(&mut self, _: &CollisionEvent<D>) {}

        fn record_dissipation(&mut self, _: f64) {}
    }

    #[test]
    fn empty_world_steps() {
        let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));
        world.step(0.01);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn falling_sphere() {
        let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));
        let h = world.add_sphere(Point::new([0.0, 10.0, 0.0]), 0.5, 1.0);

        for _ in 0..100 {
            world.step(0.01);
        }

        let y = world.body(h).unwrap().transform.translation.coord(1);
        assert!(y < 10.0, "sphere should have fallen, y = {y}");
        assert!(y > 0.0, "sphere fell too far, y = {y}");
    }

    #[test]
    fn two_spheres_collide() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let a = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let b = world.add_sphere(Point::new([3.0, 0.0, 0.0]), 1.0, 1.0);

        // Push them toward each other
        world.body_mut(a).unwrap().linear_velocity = SVector::from([5.0, 0.0, 0.0]);
        world.body_mut(b).unwrap().linear_velocity = SVector::from([-5.0, 0.0, 0.0]);

        for _ in 0..100 {
            world.step(0.01);
        }

        // After collision they should have bounced — velocities should have swapped/reduced
        let va = world.body(a).unwrap().linear_velocity[0];
        let vb = world.body(b).unwrap().linear_velocity[0];
        // A was moving right, should now be moving left (or slower)
        assert!(va < 5.0, "va = {va}, should have changed after collision");
    }

    #[test]
    fn sphere_bounces_off_static() {
        let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));

        // Static floor: small box at y=-0.5 (top face at y=0)
        let floor = RigidBody::<3>::static_body(
            BodyHandle(999),
            Point::new([0.0, -0.5, 0.0]),
            Box::new(symtropy_math::Sphere::<3>::new(symtropy_math::Point::origin(), 5.0)),
        );
        world.add_body(floor);

        // Falling sphere starting at y=3
        let sphere = world.add_sphere(Point::new([0.0, 3.0, 0.0]), 0.5, 1.0);

        for _ in 0..200 {
            world.step(0.01);
        }

        // Sphere should not have fallen far below the floor
        let y = world.body(sphere).unwrap().transform.translation.coord(1);
        assert!(y > -3.0, "sphere fell through floor, y = {y}");
    }

    #[test]
    fn physics_4d() {
        let mut world = PhysicsWorld::<4>::new(SVector::from([0.0, -9.81, 0.0, 0.0]));
        let h = world.add_sphere(Point::new([0.0, 10.0, 0.0, 0.0]), 1.0, 1.0);

        for _ in 0..100 {
            world.step(0.01);
        }

        let y = world.body(h).unwrap().transform.translation.coord(1);
        assert!(y < 10.0, "4D sphere should fall, y = {y}");
    }

    #[test]
    fn physics_2d() {
        let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
        let h = world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);

        for _ in 0..100 {
            world.step(0.01);
        }

        let y = world.body(h).unwrap().transform.translation.coord(1);
        assert!(y < 10.0, "2D sphere should fall, y = {y}");
    }

    #[test]
    fn energy_decreases_with_damping() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h = world.add_sphere(Point::origin(), 1.0, 1.0);
        world.body_mut(h).unwrap().linear_velocity = SVector::from([10.0, 0.0, 0.0]);

        let initial_ke = world.total_kinetic_energy();

        for _ in 0..100 {
            world.step(0.01);
        }

        let final_ke = world.total_kinetic_energy();
        assert!(final_ke < initial_ke, "KE should decrease with damping");
    }

    #[test]
    fn callback_modulates_applied_forces_before_integration() {
        let dt = 0.1;

        let mut blocked = PhysicsWorld::<3>::new(SVector::zeros());
        let h_blocked = blocked.add_sphere(Point::origin(), 1.0, 1.0);
        blocked
            .body_mut(h_blocked)
            .unwrap()
            .apply_force(SVector::from([10.0, 0.0, 0.0]));
        let mut red_tier = ForceScaleCallback { gain: 0.0 };
        blocked.step_with_callback(dt, &mut red_tier);
        assert!(
            blocked.body(h_blocked).unwrap().linear_velocity.norm() < 1e-12,
            "zero-gain callback should block intended motor force"
        );

        let mut allowed = PhysicsWorld::<3>::new(SVector::zeros());
        let h_allowed = allowed.add_sphere(Point::origin(), 1.0, 1.0);
        allowed
            .body_mut(h_allowed)
            .unwrap()
            .apply_force(SVector::from([10.0, 0.0, 0.0]));
        let mut green_tier = ForceScaleCallback { gain: 1.0 };
        allowed.step_with_callback(dt, &mut green_tier);
        assert!(
            allowed.body(h_allowed).unwrap().linear_velocity[0] > 0.9,
            "unit-gain callback should preserve intended motor force"
        );
    }

    #[test]
    fn add_bodies_deterministic_assigns_net_ids() {
        use symtropy_math::{Point, Sphere};
        let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));

        let bodies = vec![
            (NetId(42), RigidBody::dynamic_sphere(BodyHandle(0), Point::new([0.0, 5.0, 0.0]), 1.0, 1.0)),
            (NetId(7), RigidBody::dynamic_sphere(BodyHandle(0), Point::new([3.0, 5.0, 0.0]), 1.0, 2.0)),
            (NetId(99), RigidBody::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::<3>::new(Point::origin(), 10.0)))),
        ];

        let handles = world.add_bodies_deterministic(bodies).unwrap();
        assert_eq!(handles.len(), 3);
        assert_eq!(world.body_count(), 3);

        // Verify NetId lookup works
        let h7 = world.handle_for_net_id(NetId(7)).unwrap();
        let h42 = world.handle_for_net_id(NetId(42)).unwrap();
        let h99 = world.handle_for_net_id(NetId(99)).unwrap();

        // Bodies are sorted by NetId, so NetId(7) gets handle first
        assert!(h7 < h42, "NetId(7) inserted before NetId(42)");
        assert!(h42 < h99, "NetId(42) inserted before NetId(99)");

        // Verify the bodies have correct net_id stored
        assert_eq!(world.body(h42).unwrap().net_id, Some(NetId(42)));

        // Unknown NetId returns None
        assert!(world.handle_for_net_id(NetId(999)).is_none());
    }

    #[test]
    fn add_bodies_deterministic_rejects_duplicates() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let bodies = vec![
            (NetId(1), RigidBody::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0)),
            (NetId(1), RigidBody::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0)),
        ];
        assert!(world.add_bodies_deterministic(bodies).is_err());
    }

    #[test]
    fn collision_groups_prevent_collision() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());

        // Two overlapping spheres in different groups
        let h1 = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = world.add_sphere(Point::new([0.5, 0.0, 0.0]), 1.0, 1.0);

        // Put them in non-overlapping groups
        world.body_mut(h1).unwrap().collision_group = 0x01;
        world.body_mut(h1).unwrap().collision_mask = 0x01;
        world.body_mut(h2).unwrap().collision_group = 0x02;
        world.body_mut(h2).unwrap().collision_mask = 0x02;

        // Step — they overlap but shouldn't collide
        world.step(0.016);
        assert!(
            world.collision_events.is_empty(),
            "bodies in different groups should not collide"
        );
    }

    #[test]
    fn collision_groups_allow_matching() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());

        // Two overlapping spheres in the same group
        let h1 = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = world.add_sphere(Point::new([0.5, 0.0, 0.0]), 1.0, 1.0);

        world.body_mut(h1).unwrap().collision_group = 0x01;
        world.body_mut(h1).unwrap().collision_mask = 0x01;
        world.body_mut(h2).unwrap().collision_group = 0x01;
        world.body_mut(h2).unwrap().collision_mask = 0x01;

        // Give converging velocity so impulse is generated
        world.body_mut(h1).unwrap().linear_velocity = SVector::from([1.0, 0.0, 0.0]);
        world.body_mut(h2).unwrap().linear_velocity = SVector::from([-1.0, 0.0, 0.0]);

        world.step(0.016);
        // Check contacts were generated (impulse events require relative velocity)
        assert!(
            !world.contacts.is_empty(),
            "bodies in matching groups should generate contacts"
        );
    }

    #[test]
    fn sensor_detects_overlap_without_impulse() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());

        let h1 = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = world.add_sphere(Point::new([0.5, 0.0, 0.0]), 1.0, 1.0);

        // Mark h1 as a sensor
        world.body_mut(h1).unwrap().is_sensor = true;

        world.step(0.016);

        // Should have sensor event, not collision event
        assert!(
            world.collision_events.is_empty(),
            "sensor should not produce collision events"
        );
        assert!(
            !world.sensor_events.is_empty(),
            "sensor should produce sensor events"
        );
        assert_eq!(world.sensor_events[0].sensor, h1);
        assert_eq!(world.sensor_events[0].other, h2);
    }
}
