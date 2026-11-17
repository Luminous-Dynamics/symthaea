//! Minimal deterministic T0 kernel for orbital-ring research.
//!
//! This crate deliberately provides only checked SI boundary types, an ideal
//! two-body circular-orbit reference calculation, and conservation-ledger
//! residuals. It does not model rotor dynamics, magnetic support, structural
//! stability, thermal behaviour, debris exposure, or ring feasibility.
//!
//! Keplerian reference values are rejected for architectures with different
//! support physics. Passing tests here is not evidence of ring equilibrium.

#![deny(unsafe_code)]

use std::{error::Error, fmt};

/// WGS-84 equatorial semi-major axis, metres.
pub const EARTH_EQUATORIAL_RADIUS_M: f64 = 6_378_137.0;
/// Earth gravitational parameter for the ideal point-mass model, m^3/s^2.
pub const EARTH_MU_M3_S2: f64 = 3.986_004_418e14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureMode {
    /// A structure whose centreline follows a freely orbiting trajectory.
    FreelyOrbitingStructure,
    /// A ring intended to remain stationary using dynamic support.
    DynamicallySupportedStationaryRing,
    /// Localized, elevated launch-loop architecture.
    LaunchLoop,
    /// Rotating momentum-exchange tether or skyhook.
    MomentumExchangeTether,
    /// Particle accelerator mounted to a ring; requires separate beam physics.
    RingMountedAccelerator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// Earth-centred inertial frame, treated as inertial for this T0 contract.
    EarthCenteredInertial,
    EarthFixed,
    BodyFixed,
    LocalOrbital,
}

impl ReferenceFrame {
    fn is_t0_inertial(self) -> bool {
        matches!(self, Self::EarthCenteredInertial)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityModel {
    EarthPointMass,
    EarthJ2,
    EarthJ2AndThirdBodies,
}

/// Checked non-negative length in metres.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Meters(f64);

impl Meters {
    pub fn new(value: f64) -> Result<Self, T0Error> {
        if !value.is_finite() {
            return Err(T0Error::NonFiniteInput("length_m"));
        }
        if value < 0.0 {
            return Err(T0Error::NegativeInput("length_m"));
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Checked positive mass in kilograms.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Kilograms(f64);

impl Kilograms {
    pub fn new(value: f64) -> Result<Self, T0Error> {
        if !value.is_finite() {
            return Err(T0Error::NonFiniteInput("mass_kg"));
        }
        if value <= 0.0 {
            return Err(T0Error::NonPositiveInput("mass_kg"));
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Checked non-negative duration in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Seconds(f64);

impl Seconds {
    pub fn new(value: f64) -> Result<Self, T0Error> {
        if !value.is_finite() {
            return Err(T0Error::NonFiniteInput("duration_s"));
        }
        if value < 0.0 {
            return Err(T0Error::NegativeInput("duration_s"));
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Checked finite energy in joules. Negative values are allowed for signed
/// energy terms and residuals.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Joules(f64);

impl Joules {
    pub fn new(value: f64) -> Result<Self, T0Error> {
        if !value.is_finite() {
            return Err(T0Error::NonFiniteInput("energy_j"));
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Scenario assumptions for T0 screening.
///
/// Altitude is measured above the WGS-84 equatorial reference radius; the
/// closed-form model does not perform a geodetic/ellipsoid conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct T0ScenarioSpec {
    architecture: ArchitectureMode,
    reference_frame: ReferenceFrame,
    gravity_model: GravityModel,
    atmospheric_drag_enabled: bool,
    altitude_m: Meters,
}

impl T0ScenarioSpec {
    pub const fn new(
        architecture: ArchitectureMode,
        reference_frame: ReferenceFrame,
        gravity_model: GravityModel,
        atmospheric_drag_enabled: bool,
        altitude_m: Meters,
    ) -> Self {
        Self {
            architecture,
            reference_frame,
            gravity_model,
            atmospheric_drag_enabled,
            altitude_m,
        }
    }

    pub const fn architecture(self) -> ArchitectureMode {
        self.architecture
    }

    pub const fn reference_frame(self) -> ReferenceFrame {
        self.reference_frame
    }

    pub const fn gravity_model(self) -> GravityModel {
        self.gravity_model
    }

    pub const fn atmospheric_drag_enabled(self) -> bool {
        self.atmospheric_drag_enabled
    }

    pub const fn altitude_m(self) -> Meters {
        self.altitude_m
    }
}

/// Ideal circular-orbit reference values, all dimensional outputs in SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularOrbitReference {
    pub altitude_m: Meters,
    pub geocentric_radius_m: Meters,
    pub circumference_m: Meters,
    pub circular_speed_m_s: f64,
    pub period_s: Seconds,
    /// Ideal circular-orbit specific kinetic energy, J/kg.
    pub specific_kinetic_energy_j_kg: f64,
    /// Ideal circular-orbit specific total mechanical energy, J/kg.
    pub specific_orbital_energy_j_kg: f64,
}

/// Calculate circular two-body orbit values for an explicitly freely orbiting
/// structure. This refuses to treat a stationary supported ring, launch loop,
/// momentum-exchange tether, or ring-mounted accelerator as a Keplerian orbit.
pub fn circular_orbit_reference(
    spec: T0ScenarioSpec,
) -> Result<CircularOrbitReference, T0Error> {
    if spec.architecture != ArchitectureMode::FreelyOrbitingStructure {
        return Err(T0Error::UnsupportedArchitecture(spec.architecture));
    }
    if !spec.reference_frame.is_t0_inertial() {
        return Err(T0Error::NonInertialFrame(spec.reference_frame));
    }
    if spec.gravity_model != GravityModel::EarthPointMass {
        return Err(T0Error::UnsupportedGravityModel(spec.gravity_model));
    }
    if spec.atmospheric_drag_enabled {
        return Err(T0Error::AtmosphericDragNotSupported);
    }

    let altitude = spec.altitude_m.value();
    let radius = EARTH_EQUATORIAL_RADIUS_M + altitude;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(T0Error::NonFiniteResult("geocentric_radius_m"));
    }

    let circumference = std::f64::consts::TAU * radius;
    let speed = (EARTH_MU_M3_S2 / radius).sqrt();
    let period = std::f64::consts::TAU
        * (radius.powi(3) / EARTH_MU_M3_S2).sqrt();
    let specific_kinetic_energy = 0.5 * speed * speed;
    let specific_orbital_energy = -EARTH_MU_M3_S2 / (2.0 * radius);

    for (name, value) in [
        ("circumference_m", circumference),
        ("circular_speed_m_s", speed),
        ("period_s", period),
        ("specific_kinetic_energy_j_kg", specific_kinetic_energy),
        ("specific_orbital_energy_j_kg", specific_orbital_energy),
    ] {
        if !value.is_finite() {
            return Err(T0Error::NonFiniteResult(name));
        }
    }

    Ok(CircularOrbitReference {
        altitude_m: spec.altitude_m,
        geocentric_radius_m: Meters::new(radius)?,
        circumference_m: Meters::new(circumference)?,
        circular_speed_m_s: speed,
        period_s: Seconds::new(period)?,
        specific_kinetic_energy_j_kg: specific_kinetic_energy,
        specific_orbital_energy_j_kg: specific_orbital_energy,
    })
}

/// Momentum vector components in kg·m/s, in one declared reference frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumVector {
    components_kg_m_s: [f64; 3],
}

impl MomentumVector {
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, T0Error> {
        let components = [x, y, z];
        if components.iter().any(|value| !value.is_finite()) {
            return Err(T0Error::NonFiniteInput("momentum_kg_m_s"));
        }
        Ok(Self { components_kg_m_s: components })
    }

    pub const fn components_kg_m_s(self) -> [f64; 3] {
        self.components_kg_m_s
    }

    pub fn magnitude_kg_m_s(self) -> f64 {
        let [x, y, z] = self.components_kg_m_s;
        x.hypot(y).hypot(z)
    }

    fn add(self, rhs: Self) -> Result<Self, T0Error> {
        Self::new(
            self.components_kg_m_s[0] + rhs.components_kg_m_s[0],
            self.components_kg_m_s[1] + rhs.components_kg_m_s[1],
            self.components_kg_m_s[2] + rhs.components_kg_m_s[2],
        )
    }

    fn subtract(self, rhs: Self) -> Result<Self, T0Error> {
        Self::new(
            self.components_kg_m_s[0] - rhs.components_kg_m_s[0],
            self.components_kg_m_s[1] - rhs.components_kg_m_s[1],
            self.components_kg_m_s[2] - rhs.components_kg_m_s[2],
        )
    }
}

/// Energy ledger with a common, declared inertial frame.
///
/// Sign convention: initial + external input = final + dissipated + exported.
/// System boundaries and energy terms must be mutually compatible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyLedger {
    frame: ReferenceFrame,
    initial_j: Joules,
    external_input_j: Joules,
    final_j: Joules,
    dissipated_j: Joules,
    exported_j: Joules,
}

impl EnergyLedger {
    pub const fn new(
        frame: ReferenceFrame,
        initial_j: Joules,
        external_input_j: Joules,
        final_j: Joules,
        dissipated_j: Joules,
        exported_j: Joules,
    ) -> Self {
        Self { frame, initial_j, external_input_j, final_j, dissipated_j, exported_j }
    }

    /// Positive residual means more energy was supplied than accounted for.
    pub fn residual_j(&self) -> Result<Joules, T0Error> {
        self.require_inertial_frame()?;
        let residual = self.initial_j.value() + self.external_input_j.value()
            - self.final_j.value() - self.dissipated_j.value() - self.exported_j.value();
        Joules::new(residual).map_err(|_| T0Error::NonFiniteResult("energy_residual_j"))
    }

    pub fn is_balanced(&self, tolerance_j: Joules) -> Result<bool, T0Error> {
        if tolerance_j.value() < 0.0 {
            return Err(T0Error::NegativeInput("energy_tolerance_j"));
        }
        Ok(self.residual_j()?.value().abs() <= tolerance_j.value())
    }

    fn require_inertial_frame(&self) -> Result<(), T0Error> {
        if !self.frame.is_t0_inertial() {
            return Err(T0Error::NonInertialFrame(self.frame));
        }
        Ok(())
    }
}

/// Momentum ledger with a common declared inertial frame.
///
/// Sign convention: initial + external impulse = final + exported impulse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumLedger {
    frame: ReferenceFrame,
    initial: MomentumVector,
    external_impulse: MomentumVector,
    final_momentum: MomentumVector,
    exported_impulse: MomentumVector,
}

impl MomentumLedger {
    pub const fn new(
        frame: ReferenceFrame,
        initial: MomentumVector,
        external_impulse: MomentumVector,
        final_momentum: MomentumVector,
        exported_impulse: MomentumVector,
    ) -> Self {
        Self { frame, initial, external_impulse, final_momentum, exported_impulse }
    }

    pub fn residual_kg_m_s(&self) -> Result<MomentumVector, T0Error> {
        if !self.frame.is_t0_inertial() {
            return Err(T0Error::NonInertialFrame(self.frame));
        }
        self.initial.add(self.external_impulse)?
            .subtract(self.final_momentum)?
            .subtract(self.exported_impulse)
    }

    pub fn is_balanced(&self, tolerance_kg_m_s: f64) -> Result<bool, T0Error> {
        if !tolerance_kg_m_s.is_finite() {
            return Err(T0Error::NonFiniteInput("momentum_tolerance_kg_m_s"));
        }
        if tolerance_kg_m_s < 0.0 {
            return Err(T0Error::NegativeInput("momentum_tolerance_kg_m_s"));
        }
        let magnitude = self.residual_kg_m_s()?.magnitude_kg_m_s();
        if !magnitude.is_finite() {
            return Err(T0Error::NonFiniteResult("momentum_residual_magnitude"));
        }
        Ok(magnitude <= tolerance_kg_m_s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T0Error {
    NonFiniteInput(&'static str),
    NegativeInput(&'static str),
    NonPositiveInput(&'static str),
    UnsupportedArchitecture(ArchitectureMode),
    NonInertialFrame(ReferenceFrame),
    UnsupportedGravityModel(GravityModel),
    AtmosphericDragNotSupported,
    NonFiniteResult(&'static str),
}

impl fmt::Display for T0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput(name) => write!(f, "{name} must be finite"),
            Self::NegativeInput(name) => write!(f, "{name} must be non-negative"),
            Self::NonPositiveInput(name) => write!(f, "{name} must be positive"),
            Self::UnsupportedArchitecture(mode) => {
                write!(f, "circular-orbit reference is not valid evidence for {mode:?}")
            }
            Self::NonInertialFrame(frame) => {
                write!(f, "calculation requires EarthCenteredInertial, got {frame:?}")
            }
            Self::UnsupportedGravityModel(model) => {
                write!(f, "T0 circular-orbit reference does not support {model:?}")
            }
            Self::AtmosphericDragNotSupported => {
                write!(f, "T0 circular-orbit reference does not include atmospheric drag")
            }
            Self::NonFiniteResult(name) => write!(f, "calculation produced non-finite {name}"),
        }
    }
}

impl Error for T0Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_spec(altitude_m: f64) -> T0ScenarioSpec {
        T0ScenarioSpec::new(
            ArchitectureMode::FreelyOrbitingStructure,
            ReferenceFrame::EarthCenteredInertial,
            GravityModel::EarthPointMass,
            false,
            Meters::new(altitude_m).expect("valid test altitude"),
        )
    }

    #[test]
    fn reference_case_at_400_km_matches_documented_values() {
        let result = circular_orbit_reference(valid_spec(400_000.0)).unwrap();
        assert!((result.geocentric_radius_m.value() / 1_000.0 - 6_778.137).abs() < 1e-9);
        assert!((result.circumference_m.value() / 1_000.0 - 42_588.2908).abs() < 0.001);
        assert!((result.circular_speed_m_s / 1_000.0 - 7.6685582).abs() < 1e-6);
        assert!((result.period_s.value() / 60.0 - 92.5604045).abs() < 1e-5);
        assert!((result.specific_kinetic_energy_j_kg / 1e6 - 29.4033922).abs() < 1e-5);
        assert!(result.specific_orbital_energy_j_kg < 0.0);
    }

    #[test]
    fn reference_grid_is_finite_and_ordered() {
        let results: Vec<_> = [200_000.0, 300_000.0, 400_000.0, 500_000.0]
            .into_iter()
            .map(|h| circular_orbit_reference(valid_spec(h)).unwrap())
            .collect();
        for result in &results {
            assert!(result.circular_speed_m_s.is_finite());
            assert!(result.period_s.value().is_finite());
            assert!(result.circumference_m.value().is_finite());
        }
        assert!(results.windows(2).all(|p| {
            p[0].circular_speed_m_s > p[1].circular_speed_m_s
                && p[0].period_s.value() < p[1].period_s.value()
                && p[0].circumference_m.value() < p[1].circumference_m.value()
        }));
    }

    #[test]
    fn rejects_architectures_that_are_not_free_keplerian_orbits() {
        for mode in [
            ArchitectureMode::DynamicallySupportedStationaryRing,
            ArchitectureMode::LaunchLoop,
            ArchitectureMode::MomentumExchangeTether,
            ArchitectureMode::RingMountedAccelerator,
        ] {
            let spec = T0ScenarioSpec::new(
                mode,
                ReferenceFrame::EarthCenteredInertial,
                GravityModel::EarthPointMass,
                false,
                Meters::new(400_000.0).unwrap(),
            );
            assert_eq!(circular_orbit_reference(spec).unwrap_err(), T0Error::UnsupportedArchitecture(mode));
        }
    }

    #[test]
    fn rejects_rotating_frames_unimplemented_gravity_and_drag() {
        for frame in [ReferenceFrame::EarthFixed, ReferenceFrame::BodyFixed, ReferenceFrame::LocalOrbital] {
            let spec = T0ScenarioSpec::new(
                ArchitectureMode::FreelyOrbitingStructure, frame, GravityModel::EarthPointMass,
                false, Meters::new(400_000.0).unwrap(),
            );
            assert_eq!(circular_orbit_reference(spec).unwrap_err(), T0Error::NonInertialFrame(frame));
        }
        let j2 = T0ScenarioSpec::new(
            ArchitectureMode::FreelyOrbitingStructure, ReferenceFrame::EarthCenteredInertial,
            GravityModel::EarthJ2, false, Meters::new(400_000.0).unwrap(),
        );
        assert_eq!(circular_orbit_reference(j2).unwrap_err(), T0Error::UnsupportedGravityModel(GravityModel::EarthJ2));
        let drag = T0ScenarioSpec::new(
            ArchitectureMode::FreelyOrbitingStructure, ReferenceFrame::EarthCenteredInertial,
            GravityModel::EarthPointMass, true, Meters::new(400_000.0).unwrap(),
        );
        assert_eq!(circular_orbit_reference(drag).unwrap_err(), T0Error::AtmosphericDragNotSupported);
    }

    #[test]
    fn invalid_units_and_non_finite_results_fail_closed() {
        assert_eq!(Meters::new(f64::NAN).unwrap_err(), T0Error::NonFiniteInput("length_m"));
        assert_eq!(Meters::new(-1.0).unwrap_err(), T0Error::NegativeInput("length_m"));
        assert_eq!(Kilograms::new(0.0).unwrap_err(), T0Error::NonPositiveInput("mass_kg"));
        assert!(Seconds::new(f64::INFINITY).is_err());
        assert!(Joules::new(f64::NEG_INFINITY).is_err());
        assert!(MomentumVector::new(0.0, f64::NAN, 0.0).is_err());
        assert_eq!(
            circular_orbit_reference(valid_spec(f64::MAX)).unwrap_err(),
            T0Error::NonFiniteResult("circumference_m")
        );
    }

    #[test]
    fn energy_ledger_balances_or_exposes_residual() {
        let balanced = EnergyLedger::new(
            ReferenceFrame::EarthCenteredInertial,
            Joules::new(100.0).unwrap(), Joules::new(25.0).unwrap(),
            Joules::new(90.0).unwrap(), Joules::new(20.0).unwrap(), Joules::new(15.0).unwrap(),
        );
        assert_eq!(balanced.residual_j().unwrap().value(), 0.0);
        assert!(balanced.is_balanced(Joules::new(1e-9).unwrap()).unwrap());

        let missing = EnergyLedger::new(
            ReferenceFrame::EarthCenteredInertial,
            Joules::new(100.0).unwrap(), Joules::new(0.0).unwrap(),
            Joules::new(90.0).unwrap(), Joules::new(0.0).unwrap(), Joules::new(0.0).unwrap(),
        );
        assert_eq!(missing.residual_j().unwrap().value(), 10.0);
        assert!(!missing.is_balanced(Joules::new(0.1).unwrap()).unwrap());
        assert!(balanced.is_balanced(Joules::new(-1.0).unwrap()).is_err());
    }

    #[test]
    fn energy_ledger_rejects_non_inertial_frame() {
        let ledger = EnergyLedger::new(
            ReferenceFrame::EarthFixed, Joules::new(1.0).unwrap(), Joules::new(0.0).unwrap(),
            Joules::new(1.0).unwrap(), Joules::new(0.0).unwrap(), Joules::new(0.0).unwrap(),
        );
        assert_eq!(ledger.residual_j().unwrap_err(), T0Error::NonInertialFrame(ReferenceFrame::EarthFixed));
    }

    #[test]
    fn momentum_ledger_balances_componentwise_and_exposes_missing_impulse() {
        let balanced = MomentumLedger::new(
            ReferenceFrame::EarthCenteredInertial,
            MomentumVector::new(10.0, 20.0, 30.0).unwrap(),
            MomentumVector::new(1.0, -2.0, 3.0).unwrap(),
            MomentumVector::new(8.0, 15.0, 20.0).unwrap(),
            MomentumVector::new(3.0, 3.0, 13.0).unwrap(),
        );
        assert!(balanced.is_balanced(1e-12).unwrap());

        let missing = MomentumLedger::new(
            ReferenceFrame::EarthCenteredInertial,
            MomentumVector::new(1.0, 0.0, 0.0).unwrap(),
            MomentumVector::new(0.0, 0.0, 0.0).unwrap(),
            MomentumVector::new(0.5, 0.0, 0.0).unwrap(),
            MomentumVector::new(0.0, 0.0, 0.0).unwrap(),
        );
        assert!(!missing.is_balanced(1e-6).unwrap());
        assert_eq!(missing.residual_kg_m_s().unwrap().components_kg_m_s(), [0.5, 0.0, 0.0]);
    }

    #[test]
    fn momentum_ledger_rejects_non_inertial_frame_and_invalid_tolerance() {
        let zero = MomentumVector::new(0.0, 0.0, 0.0).unwrap();
        let ledger = MomentumLedger::new(ReferenceFrame::LocalOrbital, zero, zero, zero, zero);
        assert_eq!(ledger.residual_kg_m_s().unwrap_err(), T0Error::NonInertialFrame(ReferenceFrame::LocalOrbital));
        assert!(ledger.is_balanced(f64::NAN).is_err());
        assert!(ledger.is_balanced(-1.0).is_err());
    }

    #[test]
    fn overflowing_momentum_residual_magnitude_fails_closed() {
        let ledger = MomentumLedger::new(
            ReferenceFrame::EarthCenteredInertial,
            MomentumVector::new(f64::MAX, f64::MAX, 0.0).unwrap(),
            MomentumVector::new(0.0, 0.0, 0.0).unwrap(),
            MomentumVector::new(0.0, 0.0, 0.0).unwrap(),
            MomentumVector::new(0.0, 0.0, 0.0).unwrap(),
        );
        assert_eq!(
            ledger.is_balanced(1.0).unwrap_err(),
            T0Error::NonFiniteResult("momentum_residual_magnitude")
        );
    }

    #[test]
    fn repeated_reference_calculation_is_deterministic() {
        let spec = valid_spec(400_000.0);
        assert_eq!(circular_orbit_reference(spec).unwrap(), circular_orbit_reference(spec).unwrap());
    }
}
