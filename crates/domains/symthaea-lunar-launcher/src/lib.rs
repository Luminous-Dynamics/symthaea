// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lunar launcher/catcher architecture vocabulary and first-order launcher math.
//!
//! This crate intentionally stops at evidence-bearing architecture semantics and
//! transparent analytic reference calculations. It does not implement orbital
//! guidance, release authority, electromagnetic field control, catcher control,
//! or any hardware command path.

#![deny(unsafe_code)]

/// Evidence class attached to launcher/catcher assumptions and envelopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EvidenceLevel {
    Assumption,
    Derived,
    Simulated,
    Measured,
    Qualified,
}

/// Broad launcher role. Detailed physics remains outside the neutral transport graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LauncherClass {
    SurfaceHop,
    RegionalSurface,
    SurfaceToSpace,
}

/// Candidate catcher architecture families for Phase-0 trade studies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CatcherClass {
    SurfaceCaptureRail,
    PassiveL2,
    ActiveL2,
    DistributedLowLunarOrbit,
    CatcherTug,
    DomainSpecific(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvidenceRef(pub String);

impl EvidenceRef {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn is_well_formed(&self) -> bool {
        !self.0.trim().is_empty()
    }
}

/// Stable reference to a launcher/catcher site without defining its frame transform.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteRef {
    pub frame: String,
    pub location: String,
    pub evidence: Vec<EvidenceRef>,
}

impl SiteRef {
    pub fn is_well_formed(&self) -> bool {
        !self.frame.trim().is_empty()
            && !self.location.trim().is_empty()
            && self.evidence.iter().all(EvidenceRef::is_well_formed)
    }
}

/// Research smart-pod description. No propulsion or capture mechanism is implied.
#[derive(Debug, Clone, PartialEq)]
pub struct LaunchPod {
    pub pod_id: String,
    pub gross_mass_kg: f64,
    pub cargo_mass_kg: f64,
    pub max_axial_acceleration_m_s2: f64,
    pub max_lateral_acceleration_m_s2: f64,
    pub optional_terminal_delta_v_m_s: Option<f64>,
    pub reusable: bool,
    pub evidence_level: EvidenceLevel,
    pub evidence: Vec<EvidenceRef>,
}

impl LaunchPod {
    pub fn is_well_formed(&self) -> bool {
        !self.pod_id.trim().is_empty()
            && finite_positive(self.gross_mass_kg)
            && self.cargo_mass_kg.is_finite()
            && self.cargo_mass_kg >= 0.0
            && self.cargo_mass_kg <= self.gross_mass_kg
            && finite_positive(self.max_axial_acceleration_m_s2)
            && finite_positive(self.max_lateral_acceleration_m_s2)
            && self
                .optional_terminal_delta_v_m_s
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && self.evidence.iter().all(EvidenceRef::is_well_formed)
    }

    pub fn dry_mass_kg(&self) -> Option<f64> {
        self.is_well_formed()
            .then_some(self.gross_mass_kg - self.cargo_mass_kg)
    }

    pub fn cargo_fraction(&self) -> Option<f64> {
        self.is_well_formed()
            .then_some(self.cargo_mass_kg / self.gross_mass_kg)
    }
}

/// Evidence-bearing launcher site description. Track geometry and EM design are out of scope.
#[derive(Debug, Clone, PartialEq)]
pub struct LauncherSite {
    pub launcher_id: String,
    pub class: LauncherClass,
    pub site: SiteRef,
    pub max_payload_mass_kg: f64,
    pub max_exit_speed_m_s: f64,
    pub max_axial_acceleration_m_s2: f64,
    pub safe_miss_corridor_ref: String,
    pub evidence_level: EvidenceLevel,
    pub evidence: Vec<EvidenceRef>,
}

impl LauncherSite {
    pub fn is_well_formed(&self) -> bool {
        !self.launcher_id.trim().is_empty()
            && self.site.is_well_formed()
            && finite_positive(self.max_payload_mass_kg)
            && finite_positive(self.max_exit_speed_m_s)
            && finite_positive(self.max_axial_acceleration_m_s2)
            && !self.safe_miss_corridor_ref.trim().is_empty()
            && self.evidence.iter().all(EvidenceRef::is_well_formed)
    }

    /// Architecture-level compatibility only. This is not launch authorization.
    pub fn supports_pod(&self, pod: &LaunchPod) -> bool {
        self.is_well_formed()
            && pod.is_well_formed()
            && pod.gross_mass_kg <= self.max_payload_mass_kg
            && self.max_axial_acceleration_m_s2 <= pod.max_axial_acceleration_m_s2
    }
}

/// Catcher envelope used for architecture comparison, not capture control.
#[derive(Debug, Clone, PartialEq)]
pub struct CatcherEnvelope {
    pub catcher_id: String,
    pub class: CatcherClass,
    pub site: SiteRef,
    pub max_pod_mass_kg: f64,
    pub max_relative_speed_m_s: f64,
    pub max_capture_impulse_n_s: f64,
    pub regenerative_capture: bool,
    pub safe_miss_contingency_ref: String,
    pub evidence_level: EvidenceLevel,
    pub evidence: Vec<EvidenceRef>,
}

impl CatcherEnvelope {
    pub fn is_well_formed(&self) -> bool {
        !self.catcher_id.trim().is_empty()
            && self.site.is_well_formed()
            && finite_positive(self.max_pod_mass_kg)
            && finite_positive(self.max_relative_speed_m_s)
            && finite_positive(self.max_capture_impulse_n_s)
            && !self.safe_miss_contingency_ref.trim().is_empty()
            && self.evidence.iter().all(EvidenceRef::is_well_formed)
    }
}

/// Inputs to the LL-002 constant-acceleration analytic reference model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticLaunchInput {
    /// Gross launched mass including pod and cargo.
    pub mass_kg: f64,
    /// Speed entering the modeled acceleration section.
    pub initial_speed_m_s: f64,
    /// Requested exit speed from the modeled section.
    pub exit_speed_m_s: f64,
    /// Constant longitudinal acceleration magnitude used for the reference case.
    pub acceleration_m_s2: f64,
    /// Electrical-to-payload-kinetic-energy efficiency, exclusive of downstream capture.
    pub electrical_efficiency: f64,
    /// Minimum time between launch events for long-run average-power accounting.
    pub launch_interval_s: f64,
}

impl AnalyticLaunchInput {
    pub fn validate(&self) -> Result<(), LauncherError> {
        let values = [
            self.mass_kg,
            self.initial_speed_m_s,
            self.exit_speed_m_s,
            self.acceleration_m_s2,
            self.electrical_efficiency,
            self.launch_interval_s,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(LauncherError::NonFiniteInput);
        }
        if self.mass_kg <= 0.0
            || self.initial_speed_m_s < 0.0
            || self.exit_speed_m_s <= self.initial_speed_m_s
            || self.acceleration_m_s2 <= 0.0
            || self.launch_interval_s <= 0.0
        {
            return Err(LauncherError::InvalidInput);
        }
        if !(0.0..=1.0).contains(&self.electrical_efficiency)
            || self.electrical_efficiency == 0.0
        {
            return Err(LauncherError::InvalidEfficiency);
        }
        Ok(())
    }
}

/// Transparent outputs from the LL-002 first-order reference model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticLaunchResult {
    pub delta_v_m_s: f64,
    pub acceleration_time_s: f64,
    pub minimum_track_length_m: f64,
    pub average_force_n: f64,
    pub payload_kinetic_energy_gain_j: f64,
    pub electrical_input_energy_j: f64,
    pub average_power_during_acceleration_w: f64,
    pub long_run_average_power_w: f64,
    pub launches_per_hour: f64,
    pub energy_per_kg_j: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LauncherError {
    NonFiniteInput,
    InvalidInput,
    InvalidEfficiency,
    CargoAccelerationExceeded,
}

/// Solve the LL-002 constant-acceleration reference case.
///
/// Relations are intentionally elementary and auditable:
/// `v^2 = v0^2 + 2 a L`, `delta_KE = 1/2 m (v^2-v0^2)`, and
/// `E_electrical = delta_KE / efficiency`.
///
/// This is not an electromagnetic field, thermal, structural, trajectory, or
/// release-safety model.
pub fn analytic_launch_reference(
    input: AnalyticLaunchInput,
) -> Result<AnalyticLaunchResult, LauncherError> {
    input.validate()?;

    let delta_v_m_s = input.exit_speed_m_s - input.initial_speed_m_s;
    let acceleration_time_s = delta_v_m_s / input.acceleration_m_s2;
    let minimum_track_length_m =
        (input.exit_speed_m_s.powi(2) - input.initial_speed_m_s.powi(2))
            / (2.0 * input.acceleration_m_s2);
    let average_force_n = input.mass_kg * input.acceleration_m_s2;
    let payload_kinetic_energy_gain_j = 0.5
        * input.mass_kg
        * (input.exit_speed_m_s.powi(2) - input.initial_speed_m_s.powi(2));
    let electrical_input_energy_j = payload_kinetic_energy_gain_j / input.electrical_efficiency;
    let average_power_during_acceleration_w = electrical_input_energy_j / acceleration_time_s;
    let long_run_average_power_w = electrical_input_energy_j / input.launch_interval_s;
    let launches_per_hour = 3600.0 / input.launch_interval_s;
    let energy_per_kg_j = electrical_input_energy_j / input.mass_kg;

    let result = AnalyticLaunchResult {
        delta_v_m_s,
        acceleration_time_s,
        minimum_track_length_m,
        average_force_n,
        payload_kinetic_energy_gain_j,
        electrical_input_energy_j,
        average_power_during_acceleration_w,
        long_run_average_power_w,
        launches_per_hour,
        energy_per_kg_j,
    };

    if [
        result.delta_v_m_s,
        result.acceleration_time_s,
        result.minimum_track_length_m,
        result.average_force_n,
        result.payload_kinetic_energy_gain_j,
        result.electrical_input_energy_j,
        result.average_power_during_acceleration_w,
        result.long_run_average_power_w,
        result.launches_per_hour,
        result.energy_per_kg_j,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(LauncherError::InvalidInput);
    }

    Ok(result)
}

/// Run the reference model while enforcing the pod's declared axial acceleration limit.
pub fn analytic_launch_for_pod(
    pod: &LaunchPod,
    mut input: AnalyticLaunchInput,
) -> Result<AnalyticLaunchResult, LauncherError> {
    if !pod.is_well_formed() {
        return Err(LauncherError::InvalidInput);
    }
    input.mass_kg = pod.gross_mass_kg;
    if input.acceleration_m_s2 > pod.max_axial_acceleration_m_s2 {
        return Err(LauncherError::CargoAccelerationExceeded);
    }
    analytic_launch_reference(input)
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }

    fn pod(max_accel: f64) -> LaunchPod {
        LaunchPod {
            pod_id: "pod-001".into(),
            gross_mass_kg: 100.0,
            cargo_mass_kg: 80.0,
            max_axial_acceleration_m_s2: max_accel,
            max_lateral_acceleration_m_s2: 20.0,
            optional_terminal_delta_v_m_s: Some(25.0),
            reusable: true,
            evidence_level: EvidenceLevel::Simulated,
            evidence: vec![EvidenceRef::new("ll-pod-v0")],
        }
    }

    #[test]
    fn closed_form_reference_matches_basic_case() {
        let input = AnalyticLaunchInput {
            mass_kg: 100.0,
            initial_speed_m_s: 0.0,
            exit_speed_m_s: 100.0,
            acceleration_m_s2: 10.0,
            electrical_efficiency: 0.5,
            launch_interval_s: 20.0,
        };
        let result = analytic_launch_reference(input).unwrap();

        close(result.delta_v_m_s, 100.0, 1.0e-12);
        close(result.acceleration_time_s, 10.0, 1.0e-12);
        close(result.minimum_track_length_m, 500.0, 1.0e-12);
        close(result.average_force_n, 1_000.0, 1.0e-12);
        close(result.payload_kinetic_energy_gain_j, 500_000.0, 1.0e-9);
        close(result.electrical_input_energy_j, 1_000_000.0, 1.0e-9);
        close(result.average_power_during_acceleration_w, 100_000.0, 1.0e-9);
        close(result.long_run_average_power_w, 50_000.0, 1.0e-9);
        close(result.launches_per_hour, 180.0, 1.0e-12);
        close(result.energy_per_kg_j, 10_000.0, 1.0e-9);
    }

    #[test]
    fn nonzero_initial_speed_uses_energy_and_distance_difference() {
        let input = AnalyticLaunchInput {
            mass_kg: 20.0,
            initial_speed_m_s: 50.0,
            exit_speed_m_s: 150.0,
            acceleration_m_s2: 20.0,
            electrical_efficiency: 0.8,
            launch_interval_s: 60.0,
        };
        let result = analytic_launch_reference(input).unwrap();

        close(result.acceleration_time_s, 5.0, 1.0e-12);
        close(result.minimum_track_length_m, 500.0, 1.0e-12);
        close(result.payload_kinetic_energy_gain_j, 200_000.0, 1.0e-9);
        close(result.electrical_input_energy_j, 250_000.0, 1.0e-9);
    }

    #[test]
    fn higher_acceleration_shortens_track_without_changing_ideal_energy() {
        let slow = analytic_launch_reference(AnalyticLaunchInput {
            mass_kg: 100.0,
            initial_speed_m_s: 0.0,
            exit_speed_m_s: 200.0,
            acceleration_m_s2: 10.0,
            electrical_efficiency: 0.75,
            launch_interval_s: 100.0,
        })
        .unwrap();
        let fast = analytic_launch_reference(AnalyticLaunchInput {
            acceleration_m_s2: 40.0,
            ..AnalyticLaunchInput {
                mass_kg: 100.0,
                initial_speed_m_s: 0.0,
                exit_speed_m_s: 200.0,
                acceleration_m_s2: 10.0,
                electrical_efficiency: 0.75,
                launch_interval_s: 100.0,
            }
        })
        .unwrap();

        close(fast.minimum_track_length_m, slow.minimum_track_length_m / 4.0, 1.0e-12);
        close(
            fast.electrical_input_energy_j,
            slow.electrical_input_energy_j,
            1.0e-9,
        );
        assert!(fast.average_power_during_acceleration_w > slow.average_power_during_acceleration_w);
    }

    #[test]
    fn pod_acceleration_limit_fails_closed() {
        let p = pod(30.0);
        let input = AnalyticLaunchInput {
            mass_kg: 1.0,
            initial_speed_m_s: 0.0,
            exit_speed_m_s: 100.0,
            acceleration_m_s2: 40.0,
            electrical_efficiency: 0.9,
            launch_interval_s: 30.0,
        };
        assert_eq!(
            analytic_launch_for_pod(&p, input),
            Err(LauncherError::CargoAccelerationExceeded)
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let bad_efficiency = AnalyticLaunchInput {
            mass_kg: 10.0,
            initial_speed_m_s: 0.0,
            exit_speed_m_s: 10.0,
            acceleration_m_s2: 1.0,
            electrical_efficiency: 0.0,
            launch_interval_s: 10.0,
        };
        assert_eq!(
            analytic_launch_reference(bad_efficiency),
            Err(LauncherError::InvalidEfficiency)
        );

        let nan = AnalyticLaunchInput {
            mass_kg: f64::NAN,
            ..bad_efficiency
        };
        assert_eq!(
            analytic_launch_reference(nan),
            Err(LauncherError::NonFiniteInput)
        );
    }

    #[test]
    fn architecture_vocabulary_validates_without_granting_authority() {
        let launcher = LauncherSite {
            launcher_id: "south-pole-demo".into(),
            class: LauncherClass::RegionalSurface,
            site: SiteRef {
                frame: "lunar-fixed".into(),
                location: "site-a".into(),
                evidence: vec![EvidenceRef::new("survey-v0")],
            },
            max_payload_mass_kg: 500.0,
            max_exit_speed_m_s: 500.0,
            max_axial_acceleration_m_s2: 25.0,
            safe_miss_corridor_ref: "corridor-a-v0".into(),
            evidence_level: EvidenceLevel::Simulated,
            evidence: vec![EvidenceRef::new("ll-site-v0")],
        };
        assert!(launcher.is_well_formed());
        assert!(launcher.supports_pod(&pod(30.0)));
    }
}
