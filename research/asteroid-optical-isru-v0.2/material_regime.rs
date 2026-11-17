// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Physics-informed B1 optical-mining regime model. This is a calibratable
// surrogate, not a validated thermal/particle transport solver.
//
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/material_regime.rs -o /tmp/isru-material-regime
//   /tmp/isru-material-regime

#![forbid(unsafe_code)]

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialParameters {
    pub bulk_density_kg_m3: f64,
    pub specific_heat_j_kg_k: f64,
    pub active_thermal_depth_m: f64,
    pub water_mass_fraction: f64,
    pub organic_mass_fraction: f64,
    pub salt_mass_fraction: f64,
    pub spallation_onset_k: f64,
    pub volatile_release_onset_k: f64,
    pub brittle_ductile_transition_k: f64,
    pub melt_onset_k: f64,
    pub melt_transition_width_k: f64,
    pub spallation_yield_per_pulse: f64,
    pub volatile_release_efficiency: f64,
    pub melt_growth_time_constant_s: f64,
    pub melt_suppression_exponent: f64,
    pub ductile_spallation_floor: f64,
    pub cooling_time_constant_s: f64,
    pub surface_refresh_efficiency: f64,
    pub surface_depletion_rate: f64,
    pub organic_deposition_fraction: f64,
    pub airborne_salt_fraction: f64,
    pub airborne_dust_fraction: f64,
}

impl MaterialParameters {
    pub fn validate(&self) -> Result<(), RegimeError> {
        for (name, value) in [
            ("bulk_density_kg_m3", self.bulk_density_kg_m3),
            ("specific_heat_j_kg_k", self.specific_heat_j_kg_k),
            ("active_thermal_depth_m", self.active_thermal_depth_m),
            ("water_mass_fraction", self.water_mass_fraction),
            ("organic_mass_fraction", self.organic_mass_fraction),
            ("salt_mass_fraction", self.salt_mass_fraction),
            ("spallation_onset_k", self.spallation_onset_k),
            ("volatile_release_onset_k", self.volatile_release_onset_k),
            (
                "brittle_ductile_transition_k",
                self.brittle_ductile_transition_k,
            ),
            ("melt_onset_k", self.melt_onset_k),
            ("melt_transition_width_k", self.melt_transition_width_k),
            (
                "spallation_yield_per_pulse",
                self.spallation_yield_per_pulse,
            ),
            (
                "volatile_release_efficiency",
                self.volatile_release_efficiency,
            ),
            (
                "melt_growth_time_constant_s",
                self.melt_growth_time_constant_s,
            ),
            ("melt_suppression_exponent", self.melt_suppression_exponent),
            ("ductile_spallation_floor", self.ductile_spallation_floor),
            ("cooling_time_constant_s", self.cooling_time_constant_s),
            (
                "surface_refresh_efficiency",
                self.surface_refresh_efficiency,
            ),
            ("surface_depletion_rate", self.surface_depletion_rate),
            (
                "organic_deposition_fraction",
                self.organic_deposition_fraction,
            ),
            ("airborne_salt_fraction", self.airborne_salt_fraction),
            ("airborne_dust_fraction", self.airborne_dust_fraction),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RegimeError::InvalidParameter(name));
            }
        }
        if self.bulk_density_kg_m3 == 0.0
            || self.specific_heat_j_kg_k == 0.0
            || self.active_thermal_depth_m == 0.0
            || self.melt_transition_width_k == 0.0
            || self.melt_growth_time_constant_s == 0.0
            || self.cooling_time_constant_s == 0.0
        {
            return Err(RegimeError::InvalidParameter("zero_physical_scale"));
        }
        if self.water_mass_fraction + self.organic_mass_fraction + self.salt_mass_fraction >= 1.0 {
            return Err(RegimeError::InvalidParameter("volatile_mass_fractions"));
        }
        for (name, value) in [
            (
                "spallation_yield_per_pulse",
                self.spallation_yield_per_pulse,
            ),
            (
                "volatile_release_efficiency",
                self.volatile_release_efficiency,
            ),
            ("ductile_spallation_floor", self.ductile_spallation_floor),
            (
                "surface_refresh_efficiency",
                self.surface_refresh_efficiency,
            ),
            ("surface_depletion_rate", self.surface_depletion_rate),
            (
                "organic_deposition_fraction",
                self.organic_deposition_fraction,
            ),
            ("airborne_salt_fraction", self.airborne_salt_fraction),
            ("airborne_dust_fraction", self.airborne_dust_fraction),
        ] {
            if value > 1.0 {
                return Err(RegimeError::InvalidParameter(name));
            }
        }
        if !(self.spallation_onset_k < self.volatile_release_onset_k
            && self.volatile_release_onset_k < self.brittle_ductile_transition_k
            && self.brittle_ductile_transition_k < self.melt_onset_k)
        {
            return Err(RegimeError::InvalidRegimeOrdering);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialState {
    pub ambient_temperature_k: f64,
    pub surface_temperature_k: f64,
    /// Total remaining bulk sample mass, including volatile inventories.
    pub remaining_solid_kg: f64,
    pub remaining_water_kg: f64,
    pub remaining_organic_kg: f64,
    pub remaining_salt_kg: f64,
    /// Fractional depletion of the presently exposed surface layer.
    pub surface_depletion_fraction: f64,
    /// Persistent melt/crust layer fraction. Cleaning the optical path does
    /// not change this material state.
    pub melt_layer_fraction: f64,
    pub elapsed_seconds: u64,
}

impl MaterialState {
    pub fn validate(&self) -> Result<(), RegimeError> {
        for (name, value) in [
            ("ambient_temperature_k", self.ambient_temperature_k),
            ("surface_temperature_k", self.surface_temperature_k),
            ("remaining_solid_kg", self.remaining_solid_kg),
            ("remaining_water_kg", self.remaining_water_kg),
            ("remaining_organic_kg", self.remaining_organic_kg),
            ("remaining_salt_kg", self.remaining_salt_kg),
            (
                "surface_depletion_fraction",
                self.surface_depletion_fraction,
            ),
            ("melt_layer_fraction", self.melt_layer_fraction),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RegimeError::InvalidState(name));
            }
        }
        if self.ambient_temperature_k == 0.0
            || self.surface_temperature_k == 0.0
            || self.surface_depletion_fraction > 1.0
            || self.melt_layer_fraction > 1.0
        {
            return Err(RegimeError::InvalidState("state_out_of_range"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaserPulse {
    /// Incident irradiance at the input reference plane, W/cm².
    pub incident_irradiance_w_cm2: f64,
    pub beam_area_cm2: f64,
    pub on_duration_seconds: f64,
    pub cooling_pause_seconds: f64,
    /// Current measured/estimated transmission of the process optical path.
    pub process_optic_transmission: f64,
}

impl LaserPulse {
    pub fn validate(&self) -> Result<(), RegimeError> {
        for (name, value) in [
            ("incident_irradiance_w_cm2", self.incident_irradiance_w_cm2),
            ("beam_area_cm2", self.beam_area_cm2),
            ("on_duration_seconds", self.on_duration_seconds),
            ("cooling_pause_seconds", self.cooling_pause_seconds),
            (
                "process_optic_transmission",
                self.process_optic_transmission,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RegimeError::InvalidPulse(name));
            }
        }
        if self.beam_area_cm2 == 0.0
            || self.on_duration_seconds == 0.0
            || self.process_optic_transmission > 1.0
        {
            return Err(RegimeError::InvalidPulse("pulse_out_of_range"));
        }
        Ok(())
    }

    pub fn incident_energy_j(&self) -> f64 {
        self.incident_irradiance_w_cm2 * self.beam_area_cm2 * self.on_duration_seconds
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrapAndGasMeasurement {
    pub trap_capture_efficiency: f64,
    pub purge_gas_flow_kg_s: f64,
    pub purge_gas_water_mass_fraction: f64,
}

impl TrapAndGasMeasurement {
    pub fn validate(&self) -> Result<(), RegimeError> {
        if !self.trap_capture_efficiency.is_finite()
            || !(0.0..=1.0).contains(&self.trap_capture_efficiency)
            || !self.purge_gas_flow_kg_s.is_finite()
            || self.purge_gas_flow_kg_s < 0.0
            || !self.purge_gas_water_mass_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.purge_gas_water_mass_fraction)
        {
            return Err(RegimeError::InvalidGasMeasurement);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialRegime {
    BelowSpallation,
    ThermalSpallation,
    DuctileOrSintering,
    Melting,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialPulseResult {
    pub simulation_only: bool,
    pub regime: MaterialRegime,
    pub incident_energy_j: f64,
    pub absorbed_energy_j: f64,
    pub peak_temperature_k: f64,
    pub end_temperature_k: f64,
    pub spalled_solid_mass_kg: f64,
    pub released_water_vapor_kg: f64,
    pub released_organic_vapor_kg: f64,
    pub released_salt_vapor_kg: f64,
    pub captured_water_kg: f64,
    /// Gas-analyzer signal includes purge moisture and is not product mass.
    pub gas_analyzer_water_signal_kg: f64,
    pub surface_refresh_fraction: f64,
    pub melt_layer_fraction: f64,
    /// Species order: organic film, volatile salt, mineral dust. These are
    /// estimated emitted/deposited areal loads at the optic reference plane;
    /// apply independently calibrated transfer factors for witness and QCM.
    pub optical_deposition_estimate_ug_cm2: [f64; 3],
    pub elapsed_seconds: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegimeError {
    InvalidParameter(&'static str),
    InvalidState(&'static str),
    InvalidPulse(&'static str),
    InvalidRegimeOrdering,
    InvalidGasMeasurement,
    SampleExhausted,
    TimeOverflow,
}

pub struct MaterialRegimeModel {
    parameters: MaterialParameters,
    state: MaterialState,
    beam_area_cm2: f64,
    optical_surface_area_cm2: f64,
}

impl MaterialRegimeModel {
    pub fn new(
        parameters: MaterialParameters,
        state: MaterialState,
        beam_area_cm2: f64,
        optical_surface_area_cm2: f64,
    ) -> Result<Self, RegimeError> {
        parameters.validate()?;
        state.validate()?;
        if !beam_area_cm2.is_finite()
            || beam_area_cm2 <= 0.0
            || !optical_surface_area_cm2.is_finite()
            || optical_surface_area_cm2 <= 0.0
        {
            return Err(RegimeError::InvalidParameter("areas"));
        }
        Ok(Self {
            parameters,
            state,
            beam_area_cm2,
            optical_surface_area_cm2,
        })
    }

    pub fn apply_pulse(
        &mut self,
        pulse: &LaserPulse,
        gas: &TrapAndGasMeasurement,
    ) -> Result<MaterialPulseResult, RegimeError> {
        pulse.validate()?;
        gas.validate()?;
        if (pulse.beam_area_cm2 - self.beam_area_cm2).abs() > 1.0e-9 {
            return Err(RegimeError::InvalidPulse("beam_area_mismatch"));
        }
        if self.state.remaining_solid_kg <= 0.0 {
            return Err(RegimeError::SampleExhausted);
        }

        let incident_energy = pulse.incident_energy_j();
        let absorbed_energy = incident_energy * pulse.process_optic_transmission;
        let active_surface_mass = (self.beam_area_cm2 * 1.0e-4)
            * self.parameters.active_thermal_depth_m
            * self.parameters.bulk_density_kg_m3;
        let thermal_mass = active_surface_mass.min(self.state.remaining_solid_kg);
        let heat_capacity = thermal_mass * self.parameters.specific_heat_j_kg_k;
        let delta_temperature = if heat_capacity > 0.0 {
            absorbed_energy / heat_capacity
        } else {
            0.0
        };
        let peak_temperature = self.state.surface_temperature_k + delta_temperature;

        let regime = if peak_temperature < self.parameters.spallation_onset_k {
            MaterialRegime::BelowSpallation
        } else if peak_temperature < self.parameters.brittle_ductile_transition_k {
            MaterialRegime::ThermalSpallation
        } else if peak_temperature < self.parameters.melt_onset_k {
            MaterialRegime::DuctileOrSintering
        } else {
            MaterialRegime::Melting
        };

        let melt_excess_fraction = ((peak_temperature - self.parameters.melt_onset_k)
            / self.parameters.melt_transition_width_k)
            .clamp(0.0, 1.0);
        let melt_kinetics =
            1.0 - (-pulse.on_duration_seconds / self.parameters.melt_growth_time_constant_s).exp();
        let melt_increment = melt_excess_fraction * melt_kinetics;
        let melt_layer_fraction =
            1.0 - (1.0 - self.state.melt_layer_fraction) * (1.0 - melt_increment);

        let spall_temperature_factor = ((peak_temperature - self.parameters.spallation_onset_k)
            / (self.parameters.brittle_ductile_transition_k - self.parameters.spallation_onset_k))
            .clamp(0.0, 1.0);
        let ductile_fraction = ((peak_temperature - self.parameters.brittle_ductile_transition_k)
            / (self.parameters.melt_onset_k - self.parameters.brittle_ductile_transition_k))
            .clamp(0.0, 1.0);
        let ductile_factor =
            1.0 - ductile_fraction * (1.0 - self.parameters.ductile_spallation_floor);
        let melt_suppression =
            (1.0 - melt_layer_fraction).powf(self.parameters.melt_suppression_exponent);
        let unrefreshed_surface = 1.0 - self.state.surface_depletion_fraction;
        let proposed_spalled_mass = thermal_mass
            * self.parameters.spallation_yield_per_pulse
            * spall_temperature_factor
            * ductile_factor
            * melt_suppression
            * unrefreshed_surface;

        let volatile_release_fraction = ((peak_temperature
            - self.parameters.volatile_release_onset_k)
            / (self.parameters.brittle_ductile_transition_k
                - self.parameters.volatile_release_onset_k))
            .clamp(0.0, 1.0)
            * self.parameters.volatile_release_efficiency
            * melt_suppression
            * unrefreshed_surface;

        let available_water =
            (thermal_mass * self.parameters.water_mass_fraction).min(self.state.remaining_water_kg);
        let available_organic = (thermal_mass * self.parameters.organic_mass_fraction)
            .min(self.state.remaining_organic_kg);
        let available_salt =
            (thermal_mass * self.parameters.salt_mass_fraction).min(self.state.remaining_salt_kg);

        let released_water = available_water * volatile_release_fraction;
        let released_organic = available_organic * volatile_release_fraction;
        let released_salt = available_salt * volatile_release_fraction;
        let released_volatile_mass = released_water + released_organic + released_salt;
        // Keep the bulk inventory as total remaining sample mass. Volatiles
        // leave that inventory as vapour, while spalled solid leaves as
        // particles; the two sinks must never exceed remaining bulk mass.
        let available_for_spallation =
            (self.state.remaining_solid_kg - released_volatile_mass).max(0.0);
        let spalled_mass = proposed_spalled_mass.min(available_for_spallation);
        let captured_water = released_water * gas.trap_capture_efficiency;
        let purge_moisture =
            gas.purge_gas_flow_kg_s * pulse.on_duration_seconds * gas.purge_gas_water_mass_fraction;
        let gas_analyzer_signal = captured_water + purge_moisture;

        let refresh_fraction = if thermal_mass > 0.0 {
            (spalled_mass / thermal_mass * self.parameters.surface_refresh_efficiency)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let depletion_increment =
            volatile_release_fraction * self.parameters.surface_depletion_rate;
        let surface_depletion = (self.state.surface_depletion_fraction + depletion_increment
            - refresh_fraction)
            .clamp(0.0, 1.0);

        let deposited_organic = released_organic * self.parameters.organic_deposition_fraction;
        let deposited_salt = released_salt * self.parameters.airborne_salt_fraction;
        let deposited_dust = spalled_mass * self.parameters.airborne_dust_fraction;
        let areal_load_scale = 1.0e9 / self.optical_surface_area_cm2;
        let optical_deposition_estimate = [
            deposited_organic * areal_load_scale,
            deposited_salt * areal_load_scale,
            deposited_dust * areal_load_scale,
        ];

        let end_temperature = self.state.ambient_temperature_k
            + (peak_temperature - self.state.ambient_temperature_k)
                * (-pulse.cooling_pause_seconds / self.parameters.cooling_time_constant_s).exp();
        let elapsed = (pulse.on_duration_seconds + pulse.cooling_pause_seconds).ceil();
        if !elapsed.is_finite() || elapsed > u64::MAX as f64 {
            return Err(RegimeError::TimeOverflow);
        }
        let elapsed_seconds = elapsed as u64;
        let next_elapsed = self
            .state
            .elapsed_seconds
            .checked_add(elapsed_seconds)
            .ok_or(RegimeError::TimeOverflow)?;

        self.state.surface_temperature_k = end_temperature;
        self.state.remaining_solid_kg =
            (self.state.remaining_solid_kg - spalled_mass - released_volatile_mass).max(0.0);
        self.state.remaining_water_kg = (self.state.remaining_water_kg - released_water).max(0.0);
        self.state.remaining_organic_kg =
            (self.state.remaining_organic_kg - released_organic).max(0.0);
        self.state.remaining_salt_kg = (self.state.remaining_salt_kg - released_salt).max(0.0);
        self.state.surface_depletion_fraction = surface_depletion;
        self.state.melt_layer_fraction = melt_layer_fraction;
        self.state.elapsed_seconds = next_elapsed;

        Ok(MaterialPulseResult {
            simulation_only: true,
            regime,
            incident_energy_j: incident_energy,
            absorbed_energy_j: absorbed_energy,
            peak_temperature_k: peak_temperature,
            end_temperature_k: end_temperature,
            spalled_solid_mass_kg: spalled_mass,
            released_water_vapor_kg: released_water,
            released_organic_vapor_kg: released_organic,
            released_salt_vapor_kg: released_salt,
            captured_water_kg: captured_water,
            gas_analyzer_water_signal_kg: gas_analyzer_signal,
            surface_refresh_fraction: refresh_fraction,
            melt_layer_fraction,
            optical_deposition_estimate_ug_cm2: optical_deposition_estimate,
            elapsed_seconds: next_elapsed,
        })
    }

    /// Advance campaign time during idle/maintenance and relax the material
    /// surface toward ambient temperature without applying laser energy.
    pub fn advance_idle(&mut self, duration_seconds: u64) -> Result<(), RegimeError> {
        let next_elapsed = self
            .state
            .elapsed_seconds
            .checked_add(duration_seconds)
            .ok_or(RegimeError::TimeOverflow)?;
        let cooling = (-(duration_seconds as f64)
            / self.parameters.cooling_time_constant_s)
            .exp();
        self.state.surface_temperature_k = self.state.ambient_temperature_k
            + (self.state.surface_temperature_k - self.state.ambient_temperature_k) * cooling;
        self.state.elapsed_seconds = next_elapsed;
        Ok(())
    }

    pub fn state(&self) -> &MaterialState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> MaterialParameters {
        MaterialParameters {
            bulk_density_kg_m3: 3_000.0,
            specific_heat_j_kg_k: 800.0,
            active_thermal_depth_m: 0.001,
            water_mass_fraction: 0.10,
            organic_mass_fraction: 0.02,
            salt_mass_fraction: 0.01,
            spallation_onset_k: 550.0,
            volatile_release_onset_k: 600.0,
            brittle_ductile_transition_k: 850.0,
            melt_onset_k: 1_100.0,
            melt_transition_width_k: 250.0,
            spallation_yield_per_pulse: 0.10,
            volatile_release_efficiency: 0.80,
            melt_growth_time_constant_s: 2.0,
            melt_suppression_exponent: 2.0,
            ductile_spallation_floor: 0.05,
            cooling_time_constant_s: 3.0,
            surface_refresh_efficiency: 1.0,
            surface_depletion_rate: 0.25,
            organic_deposition_fraction: 0.02,
            airborne_salt_fraction: 0.01,
            airborne_dust_fraction: 0.005,
        }
    }

    fn state() -> MaterialState {
        MaterialState {
            ambient_temperature_k: 300.0,
            surface_temperature_k: 300.0,
            remaining_solid_kg: 0.01,
            remaining_water_kg: 0.001,
            remaining_organic_kg: 0.0002,
            remaining_salt_kg: 0.0001,
            surface_depletion_fraction: 0.0,
            melt_layer_fraction: 0.0,
            elapsed_seconds: 0,
        }
    }

    fn pulse(irradiance: f64, on: f64, off: f64, transmission: f64) -> LaserPulse {
        LaserPulse {
            incident_irradiance_w_cm2: irradiance,
            beam_area_cm2: 1.0,
            on_duration_seconds: on,
            cooling_pause_seconds: off,
            process_optic_transmission: transmission,
        }
    }

    fn trap() -> TrapAndGasMeasurement {
        TrapAndGasMeasurement {
            trap_capture_efficiency: 0.8,
            purge_gas_flow_kg_s: 0.000_001,
            purge_gas_water_mass_fraction: 0.0,
        }
    }

    fn model() -> MaterialRegimeModel {
        MaterialRegimeModel::new(parameters(), state(), 1.0, 100.0).unwrap()
    }

    #[test]
    fn incident_and_absorbed_energy_are_separate() {
        let mut value = model();
        let result = value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 0.5), &trap())
            .unwrap();
        assert_eq!(result.incident_energy_j, 100.0);
        assert_eq!(result.absorbed_energy_j, 50.0);
        assert!(result.simulation_only);
    }

    #[test]
    fn lower_optical_transmission_reduces_heating_and_spallation() {
        let mut clean = model();
        let mut fouled = model();
        let clean_result = clean
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        let fouled_result = fouled
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 0.5), &trap())
            .unwrap();
        assert!(clean_result.peak_temperature_k > fouled_result.peak_temperature_k);
        assert!(clean_result.spalled_solid_mass_kg > fouled_result.spalled_solid_mass_kg);
    }

    #[test]
    fn melt_and_ductile_transition_can_reduce_spallation_at_higher_irradiance() {
        let mut efficient = model();
        let mut overheated = model();
        let spalling = efficient
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        let melt = overheated
            .apply_pulse(&pulse(300.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        assert_eq!(spalling.regime, MaterialRegime::ThermalSpallation);
        assert_eq!(melt.regime, MaterialRegime::Melting);
        assert!(melt.peak_temperature_k > spalling.peak_temperature_k);
        assert!(spalling.spalled_solid_mass_kg > melt.spalled_solid_mass_kg);
        assert!(melt.melt_layer_fraction > 0.0);
    }

    #[test]
    fn pulsed_delivery_can_preserve_spallation_better_than_equal_energy_continuous_delivery() {
        let mut pulsed = model();
        let mut continuous = model();
        let mut pulsed_total = 0.0;
        for _ in 0..3 {
            pulsed_total += pulsed
                .apply_pulse(&pulse(100.0, 1.0, 10.0, 1.0), &trap())
                .unwrap()
                .spalled_solid_mass_kg;
        }
        let continuous_total = continuous
            .apply_pulse(&pulse(300.0, 1.0, 0.0, 1.0), &trap())
            .unwrap()
            .spalled_solid_mass_kg;
        assert!(pulsed_total > continuous_total);
        assert!(pulsed.state().melt_layer_fraction < continuous.state().melt_layer_fraction);
    }

    #[test]
    fn wet_purge_inflates_gas_signal_but_not_captured_water() {
        let mut dry = model();
        let mut wet = model();
        let dry_gas = trap();
        let wet_gas = TrapAndGasMeasurement {
            purge_gas_water_mass_fraction: 0.05,
            ..trap()
        };
        let dry_result = dry
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &dry_gas)
            .unwrap();
        let wet_result = wet
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &wet_gas)
            .unwrap();
        assert_eq!(dry_result.captured_water_kg, wet_result.captured_water_kg);
        assert!(wet_result.gas_analyzer_water_signal_kg > dry_result.gas_analyzer_water_signal_kg);
    }

    #[test]
    fn water_release_and_captured_water_are_distinct() {
        let mut value = model();
        let result = value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        assert!(result.released_water_vapor_kg >= result.captured_water_kg);
        assert!(result.released_water_vapor_kg > 0.0);
        assert!(result.captured_water_kg > 0.0);
    }

    #[test]
    fn organic_and_particle_release_create_optic_deposition_estimates() {
        let mut value = model();
        let result = value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        assert!(result.optical_deposition_estimate_ug_cm2[0] > 0.0);
        assert!(result.optical_deposition_estimate_ug_cm2[2] > 0.0);
    }

    #[test]
    fn surface_refresh_reduces_exposed_surface_depletion() {
        let mut value = model();
        let before = value.state().surface_depletion_fraction;
        let result = value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        assert!(result.surface_refresh_fraction > 0.0);
        assert!(value.state().surface_depletion_fraction < before + 0.1);
    }

    #[test]
    fn idle_time_cools_material_and_advances_campaign_clock() {
        let mut value = model();
        let hot = value.apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap()).unwrap();
        let before = value.state().surface_temperature_k;
        value.advance_idle(10).unwrap();
        assert!(value.state().surface_temperature_k < before);
        assert_eq!(value.state().elapsed_seconds, hot.elapsed_seconds + 10);
    }

    #[test]
    fn volatile_release_and_spallation_conserve_bulk_sample_mass() {
        let mut value = model();
        let initial_mass = value.state().remaining_solid_kg;
        let result = value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        let removed_mass = result.spalled_solid_mass_kg
            + result.released_water_vapor_kg
            + result.released_organic_vapor_kg
            + result.released_salt_vapor_kg;
        let remaining_mass = value.state().remaining_solid_kg;
        assert!((initial_mass - remaining_mass - removed_mass).abs() < 1.0e-12);
    }

    #[test]
    fn sample_inventory_is_depleted_and_exhaustion_is_explicit() {
        let mut params = parameters();
        params.spallation_yield_per_pulse = 1.0;
        params.melt_suppression_exponent = 0.0;
        params.ductile_spallation_floor = 1.0;
        let mut state = state();
        state.remaining_solid_kg = 0.000_3;
        let mut value = MaterialRegimeModel::new(params, state, 1.0, 100.0).unwrap();
        value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        value
            .apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap())
            .unwrap();
        assert_eq!(
            value.apply_pulse(&pulse(100.0, 1.0, 0.0, 1.0), &trap()),
            Err(RegimeError::SampleExhausted)
        );
    }

    #[test]
    fn invalid_regime_ordering_and_pulse_inputs_fail_closed() {
        let mut params = parameters();
        params.melt_onset_k = params.brittle_ductile_transition_k;
        assert_eq!(
            MaterialRegimeModel::new(params, state(), 1.0, 100.0).err(),
            Some(RegimeError::InvalidRegimeOrdering)
        );
        let mut value = model();
        assert_eq!(
            value
                .apply_pulse(&pulse(-1.0, 1.0, 0.0, 1.0), &trap())
                .err(),
            Some(RegimeError::InvalidPulse("incident_irradiance_w_cm2"))
        );
    }

    #[test]
    fn gas_signal_does_not_change_physical_water_inventory_or_capture() {
        let mut value = model();
        let before = value.state().remaining_water_kg;
        let result = value
            .apply_pulse(
                &pulse(100.0, 1.0, 0.0, 1.0),
                &TrapAndGasMeasurement {
                    purge_gas_water_mass_fraction: 0.2,
                    ..trap()
                },
            )
            .unwrap();
        assert!(value.state().remaining_water_kg < before);
        assert!(result.gas_analyzer_water_signal_kg >= result.captured_water_kg);
    }
}
