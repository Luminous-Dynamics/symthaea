//! Independent analytical reference functions for a bounded AUV model.
//!
//! This crate deliberately does not call the production symthaea-auv
//! hydrodynamics helpers. It supplies hand-checkable references for
//! relative-flow drag, hydrostatics, diagonal effective inertia, and
//! mechanical power at a force/velocity boundary.
//!
//! It is not CFD, a full Fossen 6-DOF solver, a propulsor map, or a physical
//! calibration. Numerical agreement with these equations establishes only
//! equation-level consistency for the declared profile.

#![forbid(unsafe_code)]

const QUATERNION_NORM_TOLERANCE: f64 = 1.0e-6;

#[derive(Clone, Debug, PartialEq)]
pub struct HydroReferenceProfile {
    pub fluid_density_kg_m3: f64,
    /// Dimensionless translation drag coefficients for surge, sway and heave.
    pub drag_coefficient_dimensionless: [f64; 3],
    /// Reference area for surge, sway and heave, in square metres.
    pub reference_area_m2: [f64; 3],
    pub dry_mass_kg: f64,
    pub displaced_volume_m3: f64,
    /// Rigid-body dry inertia about the body axes, in kg m².
    pub dry_inertia_kg_m2: [f64; 3],
    /// Translational added mass for surge, sway and heave, in kg.
    pub added_mass_kg: [f64; 3],
    /// Added rotational inertia for roll, pitch and yaw, in kg m².
    pub added_inertia_kg_m2: [f64; 3],
    pub gravitational_acceleration_m_s2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowCase {
    /// Vehicle velocity relative to Earth, expressed in body axes, m/s.
    pub vehicle_velocity_body_m_s: [f64; 3],
    /// Water velocity relative to Earth, expressed in world axes, m/s.
    pub water_velocity_world_m_s: [f64; 3],
    /// Body-to-world quaternion in [w, x, y, z] order.
    pub body_to_world_wxyz: [f64; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowEvaluation {
    pub water_velocity_body_m_s: [f64; 3],
    pub relative_velocity_body_m_s: [f64; 3],
    /// Hydrodynamic drag in body axes, Newtons.
    pub drag_force_body_n: [f64; 3],
    pub drag_magnitude_n: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HydrostaticEvaluation {
    pub buoyancy_up_n: f64,
    pub weight_down_n: f64,
    /// Positive means the net static force is upward.
    pub net_force_up_n: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectiveInertia {
    /// Dry mass plus translational added mass, kg.
    pub translation_kg: [f64; 3],
    /// Dry rigid-body inertia plus added rotational inertia, kg m².
    pub rotation_kg_m2: [f64; 3],
}

fn finite3(values: [f64; 3]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn finite4(values: [f64; 4]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn validate_unit_quaternion(q: [f64; 4]) -> Result<(), String> {
    if !finite4(q) {
        return Err("body-to-world quaternion contains a non-finite value".to_owned());
    }
    let norm = q[0].hypot(q[1]).hypot(q[2]).hypot(q[3]);
    if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
        return Err("body-to-world quaternion has zero or invalid norm".to_owned());
    }
    if (norm - 1.0).abs() > QUATERNION_NORM_TOLERANCE {
        return Err("body-to-world quaternion is outside unit-norm tolerance".to_owned());
    }
    Ok(())
}

impl HydroReferenceProfile {
    pub fn validate(&self) -> Result<(), String> {
        if !self.fluid_density_kg_m3.is_finite() || self.fluid_density_kg_m3 <= 0.0 {
            return Err("fluid density must be finite and positive".to_owned());
        }
        if !self.gravitational_acceleration_m_s2.is_finite()
            || self.gravitational_acceleration_m_s2 <= 0.0
        {
            return Err("gravitational acceleration must be finite and positive".to_owned());
        }
        if !self.dry_mass_kg.is_finite() || self.dry_mass_kg <= 0.0 {
            return Err("dry mass must be finite and positive".to_owned());
        }
        if !self.displaced_volume_m3.is_finite() || self.displaced_volume_m3 < 0.0 {
            return Err("displaced volume must be finite and nonnegative".to_owned());
        }
        if !finite3(self.drag_coefficient_dimensionless)
            || self.drag_coefficient_dimensionless.iter().any(|value| *value < 0.0)
        {
            return Err("drag coefficients must be finite and nonnegative".to_owned());
        }
        if !finite3(self.reference_area_m2)
            || self.reference_area_m2.iter().any(|value| *value <= 0.0)
        {
            return Err("reference areas must be finite and positive".to_owned());
        }
        if !finite3(self.dry_inertia_kg_m2)
            || self.dry_inertia_kg_m2.iter().any(|value| *value <= 0.0)
        {
            return Err("dry rigid-body inertia must be finite and positive on every axis".to_owned());
        }
        if !finite3(self.added_mass_kg) || self.added_mass_kg.iter().any(|value| *value < 0.0) {
            return Err("added masses must be finite and nonnegative".to_owned());
        }
        if !finite3(self.added_inertia_kg_m2)
            || self.added_inertia_kg_m2.iter().any(|value| *value < 0.0)
        {
            return Err("added rotational inertia must be finite and nonnegative".to_owned());
        }
        Ok(())
    }
}

/// Rotate a world-frame vector into body axes using the conjugate quaternion.
pub fn world_vector_to_body(
    body_to_world_wxyz: [f64; 4],
    world_vector: [f64; 3],
) -> Result<[f64; 3], String> {
    validate_unit_quaternion(body_to_world_wxyz)?;
    if !finite3(world_vector) {
        return Err("world vector contains a non-finite value".to_owned());
    }

    let [w, x, y, z] = body_to_world_wxyz;
    let (w, x, y, z) = (w, -x, -y, -z);
    let [vx, vy, vz] = world_vector;
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    let result = [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ];
    if !finite3(result) {
        return Err("world-to-body vector rotation overflowed".to_owned());
    }
    Ok(result)
}

/// Evaluate translation drag using velocity relative to the water, not
/// vehicle velocity relative to Earth:
///
/// F_i = -0.5 * rho * Cd_i * A_i * |u_vehicle,i - u_water,i|
///       * (u_vehicle,i - u_water,i)
pub fn evaluate_relative_flow_drag(
    profile: &HydroReferenceProfile,
    case: &FlowCase,
) -> Result<FlowEvaluation, String> {
    profile.validate()?;
    if !finite3(case.vehicle_velocity_body_m_s) {
        return Err("vehicle body velocity contains a non-finite value".to_owned());
    }

    let water_body =
        world_vector_to_body(case.body_to_world_wxyz, case.water_velocity_world_m_s)?;
    let relative = [
        case.vehicle_velocity_body_m_s[0] - water_body[0],
        case.vehicle_velocity_body_m_s[1] - water_body[1],
        case.vehicle_velocity_body_m_s[2] - water_body[2],
    ];
    if !finite3(relative) {
        return Err("relative water velocity overflowed".to_owned());
    }

    let mut drag = [0.0; 3];
    for axis in 0..3 {
        drag[axis] = -0.5
            * profile.fluid_density_kg_m3
            * profile.drag_coefficient_dimensionless[axis]
            * profile.reference_area_m2[axis]
            * relative[axis].abs()
            * relative[axis];
    }
    if !finite3(drag) {
        return Err("relative-flow drag calculation overflowed".to_owned());
    }
    let magnitude = drag[0].hypot(drag[1]).hypot(drag[2]);
    if !magnitude.is_finite() {
        return Err("drag magnitude calculation overflowed".to_owned());
    }

    Ok(FlowEvaluation {
        water_velocity_body_m_s: water_body,
        relative_velocity_body_m_s: relative,
        drag_force_body_n: drag,
        drag_magnitude_n: magnitude,
    })
}

/// Compute static buoyancy and weight using the world z-up sign convention.
pub fn evaluate_hydrostatics(
    profile: &HydroReferenceProfile,
) -> Result<HydrostaticEvaluation, String> {
    profile.validate()?;
    let buoyancy = profile.fluid_density_kg_m3
        * profile.gravitational_acceleration_m_s2
        * profile.displaced_volume_m3;
    let weight = profile.dry_mass_kg * profile.gravitational_acceleration_m_s2;
    let net = buoyancy - weight;
    if !buoyancy.is_finite() || !weight.is_finite() || !net.is_finite() {
        return Err("hydrostatic force calculation overflowed".to_owned());
    }
    Ok(HydrostaticEvaluation {
        buoyancy_up_n: buoyancy,
        weight_down_n: weight,
        net_force_up_n: net,
    })
}

/// Combine dry rigid-body inertia with diagonal added mass/inertia.
///
/// Rotational effective inertia must include the rigid body's own inertia;
/// added rotational inertia alone is not a complete angular mass term.
pub fn effective_inertia(
    profile: &HydroReferenceProfile,
) -> Result<EffectiveInertia, String> {
    profile.validate()?;
    let translation = [
        profile.dry_mass_kg + profile.added_mass_kg[0],
        profile.dry_mass_kg + profile.added_mass_kg[1],
        profile.dry_mass_kg + profile.added_mass_kg[2],
    ];
    let rotation = [
        profile.dry_inertia_kg_m2[0] + profile.added_inertia_kg_m2[0],
        profile.dry_inertia_kg_m2[1] + profile.added_inertia_kg_m2[1],
        profile.dry_inertia_kg_m2[2] + profile.added_inertia_kg_m2[2],
    ];
    if !finite3(translation) || !finite3(rotation) {
        return Err("effective inertia calculation overflowed".to_owned());
    }
    Ok(EffectiveInertia {
        translation_kg: translation,
        rotation_kg_m2: rotation,
    })
}

/// Mechanical power crossing a force/velocity boundary, in watts.
///
/// This is F dot v. It is not shaft or electrical input power; those require
/// a propulsor/drive map and separately declared efficiencies.
pub fn mechanical_power_w(
    force_n: [f64; 3],
    velocity_m_s: [f64; 3],
) -> Result<f64, String> {
    if !finite3(force_n) || !finite3(velocity_m_s) {
        return Err("force or velocity contains a non-finite value".to_owned());
    }
    let power = force_n[0] * velocity_m_s[0]
        + force_n[1] * velocity_m_s[1]
        + force_n[2] * velocity_m_s[2];
    if !power.is_finite() {
        return Err("mechanical power calculation overflowed".to_owned());
    }
    Ok(power)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> HydroReferenceProfile {
        HydroReferenceProfile {
            fluid_density_kg_m3: 998.0,
            drag_coefficient_dimensionless: [50.0, 100.0, 100.0],
            reference_area_m2: [0.03, 0.15, 0.15],
            dry_mass_kg: 50.0,
            displaced_volume_m3: 0.05,
            dry_inertia_kg_m2: [1.0, 2.0, 3.0],
            added_mass_kg: [10.0, 30.0, 30.0],
            added_inertia_kg_m2: [0.5, 5.0, 5.0],
            gravitational_acceleration_m_s2: 9.81,
        }
    }

    fn flow_case(vehicle: [f64; 3], water_world: [f64; 3]) -> FlowCase {
        FlowCase {
            vehicle_velocity_body_m_s: vehicle,
            water_velocity_world_m_s: water_world,
            body_to_world_wxyz: [1.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn known_answer_surge_drag_matches_declared_equation() {
        let result =
            evaluate_relative_flow_drag(&profile(), &flow_case([1.0, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        assert!((result.drag_force_body_n[0] + 748.5).abs() < 1.0e-10);
    }

    #[test]
    fn quarter_speed_drag_is_one_sixteenth_of_surge_reference() {
        let result =
            evaluate_relative_flow_drag(&profile(), &flow_case([0.25, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        assert!((result.drag_force_body_n[0] + 46.78125).abs() < 1.0e-10);
    }

    #[test]
    fn doubling_relative_speed_quadruples_quadratic_drag() {
        let one =
            evaluate_relative_flow_drag(&profile(), &flow_case([1.0, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        let two =
            evaluate_relative_flow_drag(&profile(), &flow_case([2.0, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        assert!((two.drag_force_body_n[0] / one.drag_force_body_n[0] - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn drag_reverses_sign_when_relative_velocity_reverses() {
        let forward =
            evaluate_relative_flow_drag(&profile(), &flow_case([1.0, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        let reverse =
            evaluate_relative_flow_drag(&profile(), &flow_case([-1.0, 0.0, 0.0], [0.0; 3]))
                .unwrap();
        assert!(forward.drag_force_body_n[0] < 0.0);
        assert!(reverse.drag_force_body_n[0] > 0.0);
        assert!((forward.drag_force_body_n[0] + reverse.drag_force_body_n[0]).abs() < 1.0e-12);
    }

    #[test]
    fn zero_relative_velocity_produces_zero_drag() {
        let result =
            evaluate_relative_flow_drag(&profile(), &flow_case([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
                .unwrap();
        assert_eq!(result.relative_velocity_body_m_s, [0.0, 0.0, 0.0]);
        assert_eq!(result.drag_force_body_n, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn water_velocity_is_rotated_from_world_into_body_axes() {
        let half = std::f64::consts::FRAC_PI_4;
        let case = FlowCase {
            vehicle_velocity_body_m_s: [0.0; 3],
            water_velocity_world_m_s: [1.0, 0.0, 0.0],
            body_to_world_wxyz: [half.cos(), 0.0, 0.0, half.sin()],
        };
        let result = evaluate_relative_flow_drag(&profile(), &case).unwrap();
        assert!(result.water_velocity_body_m_s[0].abs() < 1.0e-12);
        assert!((result.water_velocity_body_m_s[1] + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn buoyancy_and_weight_sign_match_default_profile_numbers() {
        let result = evaluate_hydrostatics(&profile()).unwrap();
        assert!((result.buoyancy_up_n - 489.519).abs() < 1.0e-9);
        assert!((result.weight_down_n - 490.5).abs() < 1.0e-9);
        assert!((result.net_force_up_n + 0.981).abs() < 1.0e-9);
    }

    #[test]
    fn neutral_buoyancy_has_zero_net_static_force() {
        let mut test_profile = profile();
        test_profile.displaced_volume_m3 = test_profile.dry_mass_kg / test_profile.fluid_density_kg_m3;
        let result = evaluate_hydrostatics(&test_profile).unwrap();
        assert!(result.net_force_up_n.abs() < 1.0e-10);
    }

    #[test]
    fn effective_rotational_inertia_includes_dry_and_added_inertia() {
        let result = effective_inertia(&profile()).unwrap();
        assert_eq!(result.translation_kg, [60.0, 80.0, 80.0]);
        assert_eq!(result.rotation_kg_m2, [1.5, 7.0, 8.0]);
    }

    #[test]
    fn zero_dry_rotation_inertia_is_rejected() {
        let mut test_profile = profile();
        test_profile.dry_inertia_kg_m2[0] = 0.0;
        assert!(effective_inertia(&test_profile).is_err());
    }

    #[test]
    fn mechanical_power_is_force_dot_velocity() {
        assert_eq!(mechanical_power_w([10.0, 0.0, 0.0], [2.0, 0.0, 0.0]).unwrap(), 20.0);
        assert_eq!(mechanical_power_w([-10.0, 0.0, 0.0], [2.0, 0.0, 0.0]).unwrap(), -20.0);
    }

    #[test]
    fn non_finite_velocity_is_rejected() {
        assert!(evaluate_relative_flow_drag(
            &profile(),
            &flow_case([f64::NAN, 0.0, 0.0], [0.0; 3])
        )
        .is_err());
    }

    #[test]
    fn non_unit_attitude_is_rejected() {
        let mut case = flow_case([0.0; 3], [1.0, 0.0, 0.0]);
        case.body_to_world_wxyz = [2.0, 0.0, 0.0, 0.0];
        assert!(evaluate_relative_flow_drag(&profile(), &case).is_err());
    }

    #[test]
    fn invalid_density_and_negative_area_are_rejected() {
        let mut test_profile = profile();
        test_profile.fluid_density_kg_m3 = 0.0;
        assert!(evaluate_hydrostatics(&test_profile).is_err());
        test_profile = profile();
        test_profile.reference_area_m2[0] = -0.01;
        assert!(evaluate_relative_flow_drag(&test_profile, &flow_case([1.0, 0.0, 0.0], [0.0; 3])).is_err());
    }

    #[test]
    fn power_rejects_non_finite_boundary_values() {
        assert!(mechanical_power_w([f64::INFINITY, 0.0, 0.0], [1.0, 0.0, 0.0]).is_err());
    }
}
