//! Physics world: owns bodies, steps simulation, resolves collisions.

use nalgebra::SVector;
use symtropy_math::Point;

use crate::body::{BodyHandle, RigidBody};
use crate::broadphase;
use crate::contact::{CollisionEvent, ContactManifold};
use crate::constraint::Constraint;
use crate::gjk;
use crate::integrator;

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
    /// Position correction iterations per step.
    pub solver_iterations: usize,
    /// Sleep velocity threshold.
    pub sleep_threshold: f64,
    /// Ticks below sleep threshold before sleeping.
    pub sleep_ticks: u32,
    /// Penetration slop (small overlap allowed to prevent jitter).
    pub slop: f64,
    /// Baumgarte stabilization factor.
    pub baumgarte: f64,
    next_handle: usize,
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
            solver_iterations: 4,
            slop: 0.01,
            baumgarte: 0.2,
            sleep_threshold: 0.5,
            sleep_ticks: 60, // ~1 second at 64Hz
            next_handle: 0,
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
        self.bodies.push(body);
        handle
    }

    /// Add a dynamic body with a custom collider.
    pub fn add_body(&mut self, mut body: RigidBody<D>) -> BodyHandle {
        let handle = self.allocate_handle();
        body.handle = handle;
        self.bodies.push(body);
        handle
    }

    /// Add a constraint between two bodies.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint<D>>) {
        self.constraints.push(constraint);
    }

    /// Get a reference to a body by handle.
    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody<D>> {
        self.bodies.iter().find(|b| b.handle == handle)
    }

    /// Get a mutable reference to a body by handle.
    pub fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody<D>> {
        self.bodies.iter_mut().find(|b| b.handle == handle)
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

        // 1. Integrate all bodies (skip sleeping)
        for body in &mut self.bodies {
            if !body.sleeping {
                integrator::integrate(body, &self.gravity, dt);
            }
        }

        // 2. Broadphase: find potentially colliding pairs
        let pairs = broadphase::find_pairs(&self.bodies);

        // 3. Narrowphase: GJK for each pair
        self.contacts.clear();
        for pair in &pairs {
            let (idx_a, idx_b) = self.find_body_indices(pair.0, pair.1);
            if let (Some(a), Some(b)) = (idx_a, idx_b) {
                let pos_a = self.bodies[a].transform.translation.0;
                let pos_b = self.bodies[b].transform.translation.0;

                let result = gjk::intersects(
                    self.bodies[a].collider.as_ref(),
                    &pos_a,
                    self.bodies[b].collider.as_ref(),
                    &pos_b,
                );

                if result.intersecting {
                    // EPA for accurate penetration depth and normal
                    if let Some(epa_result) = crate::epa::penetration(
                        self.bodies[a].collider.as_ref(),
                        &pos_a,
                        self.bodies[b].collider.as_ref(),
                        &pos_b,
                        &result.simplex,
                    ) {
                        if epa_result.depth > 0.0 {
                            let (_, ra) = self.bodies[a].collider.bounding_sphere();
                            self.contacts.push(ContactManifold {
                                body_a: pair.0,
                                body_b: pair.1,
                                normal: epa_result.normal,
                                depth: epa_result.depth,
                                point: pos_a + epa_result.normal * ra,
                            });
                        }
                    }
                }
            }
        }

        // 4. Resolve collisions (with consciousness modulation if callback present)
        for _ in 0..self.solver_iterations {
            let contacts: Vec<_> = self.contacts.clone();
            for contact in contacts {
                self.resolve_contact(contact, dt, &mut callback);
            }
        }

        // 5. Solve constraints
        for _ in 0..self.solver_iterations {
            for ci in 0..self.constraints.len() {
                let (ha, hb) = self.constraints[ci].bodies();
                let (idx_a, idx_b) = self.find_body_indices(ha, hb);
                if let (Some(a), Some(b)) = (idx_a, idx_b) {
                    // Safety: we know a != b (different handles)
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

        // 6. Body sleeping: deactivate near-stationary bodies
        let threshold = self.sleep_threshold;
        let ticks = self.sleep_ticks;
        for body in &mut self.bodies {
            body.try_sleep(threshold, ticks);
        }
    }

    /// Resolve a single contact with optional consciousness modulation.
    fn resolve_contact(
        &mut self,
        contact: ContactManifold<D>,
        _dt: f64,
        callback: &mut Option<&mut dyn PhysicsCallback<D>>,
    ) {
        let (idx_a, idx_b) = self.find_body_indices(contact.body_a, contact.body_b);
        let (Some(a), Some(b)) = (idx_a, idx_b) else {
            return;
        };

        let inv_mass_a = self.bodies[a].inv_mass;
        let inv_mass_b = self.bodies[b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass < 1e-15 {
            return;
        }

        // Position correction (Baumgarte stabilization)
        let correction_mag = (contact.depth - self.slop).max(0.0) * self.baumgarte;
        let correction = contact.normal * correction_mag;

        if self.bodies[a].is_dynamic() {
            integrator::separate(&mut self.bodies[a], &(-correction * (inv_mass_a / total_inv_mass)));
        }
        if self.bodies[b].is_dynamic() {
            integrator::separate(&mut self.bodies[b], &(correction * (inv_mass_b / total_inv_mass)));
        }

        // Normal velocity resolution
        let relative_vel = self.bodies[b].linear_velocity - self.bodies[a].linear_velocity;
        let restitution = (self.bodies[a].restitution + self.bodies[b].restitution) * 0.5;

        let mut j = contact.impulse_magnitude(&relative_vel, inv_mass_a, inv_mass_b, restitution);

        // ═══ CONSCIOUSNESS MODULATION ═══
        // Modulate impulse via sanctuary zones and harmony fields
        if let Some(ref cb) = callback {
            j = cb.modulate_impulse(j, &contact.point);
        }

        if j > 0.0 {
            let impulse = contact.normal * j;
            integrator::apply_impulse(&mut self.bodies[a], &(-impulse));
            integrator::apply_impulse(&mut self.bodies[b], &impulse);

            // Coulomb friction with consciousness-modulated coefficient
            let v_rel = self.bodies[b].linear_velocity - self.bodies[a].linear_velocity;
            let v_n = contact.normal * v_rel.dot(&contact.normal);
            let v_t = v_rel - v_n;
            let v_t_mag = v_t.norm();
            if v_t_mag > 1e-10 {
                let tangent = v_t / v_t_mag;
                let mut mu = (self.bodies[a].friction * self.bodies[b].friction).sqrt();

                // ═══ HARMONY FIELD FRICTION MODULATION ═══
                if let Some(ref cb) = callback {
                    mu *= cb.friction_multiplier(&contact.point, contact.body_a);
                }

                let j_t_desired = -v_t_mag / total_inv_mass;
                let j_t = j_t_desired.clamp(-mu * j, mu * j);
                let friction_impulse = tangent * j_t;
                integrator::apply_impulse(&mut self.bodies[a], &(-friction_impulse));
                integrator::apply_impulse(&mut self.bodies[b], &friction_impulse);

                // Record friction dissipation
                if let Some(ref mut cb) = callback {
                    cb.record_dissipation(j_t.abs() * 0.1);
                }
            }

            // Emit collision event and notify consciousness callback
            let event = CollisionEvent {
                body_a: contact.body_a,
                body_b: contact.body_b,
                impulse: j,
                normal: contact.normal,
                depth: contact.depth,
            };

            // ═══ CONSCIOUSNESS FEEDBACK: collision → prediction error ═══
            if let Some(ref mut cb) = callback {
                cb.on_collision(&event);
            }

            self.collision_events.push(event);

            // Wake both bodies on collision
            self.bodies[a].wake();
            self.bodies[b].wake();
        }
    }

    fn find_body_indices(&self, a: BodyHandle, b: BodyHandle) -> (Option<usize>, Option<usize>) {
        let mut ia = None;
        let mut ib = None;
        for (i, body) in self.bodies.iter().enumerate() {
            if body.handle == a {
                ia = Some(i);
            }
            if body.handle == b {
                ib = Some(i);
            }
        }
        (ia, ib)
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
}
