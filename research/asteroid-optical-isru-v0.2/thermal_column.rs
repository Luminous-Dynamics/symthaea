//! Conservative one-dimensional finite-volume thermal column for optical ISRU.
//!
//! This module resolves a depth-dependent temperature profile with explicit
//! cell geometry, exponentially attenuated optical deposition, conductive
//! exchange, radiative surface exchange and a caller-supplied latent-heat sink.
//! It is a verification-grade numerical reference, not a calibrated asteroid
//! regolith or volatile-transport model. Material properties and boundary
//! inputs must be measured for the target rig/material before prediction use.

const STEFAN_BOLTZMANN_W_M2_K4: f64 = 5.670_374_419e-8;
const MAX_SUBSTEPS: usize = 1_000_000;
const STABILITY_FACTOR: f64 = 0.4;
const ENERGY_BALANCE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalColumnParameters {
    pub bulk_density_kg_m3: f64,
    pub specific_heat_j_kg_k: f64,
    pub conductivity_w_m_k: f64,
    pub cell_thickness_m: f64,
    pub optical_absorption_depth_m: f64,
    pub surface_emissivity: f64,
    pub ambient_temperature_k: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalColumnState {
    /// Cell temperatures, ordered from the illuminated surface into the bed.
    pub temperatures_k: Vec<f64>,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalPulse {
    pub incident_irradiance_w_m2: f64,
    pub duration_seconds: f64,
    /// Measured or independently estimated transmission of the process path.
    pub optical_transmission: f64,
    pub surface_absorptivity: f64,
    /// Bounded by a separate material-inventory/kinetics model.
    pub volatile_mass_flux_kg_m2_s: f64,
    pub latent_heat_j_kg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalStepReceipt {
    pub incident_energy_j_m2: f64,
    pub energy_delivered_after_optics_j_m2: f64,
    pub optical_path_loss_j_m2: f64,
    pub absorbed_energy_in_column_j_m2: f64,
    pub unabsorbed_energy_j_m2: f64,
    /// Positive means net energy radiated outward; negative means net input.
    pub net_radiative_exchange_j_m2: f64,
    pub latent_heat_removed_j_m2: f64,
    pub internal_energy_change_j_m2: f64,
    pub energy_residual_j_m2: f64,
    pub energy_residual_fraction: f64,
    pub substeps: usize,
    pub surface_temperature_before_k: f64,
    pub surface_temperature_after_k: f64,
    pub peak_column_temperature_k: f64,
    pub simulation_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThermalColumnError {
    InvalidParameters,
    InvalidState,
    InvalidPulse,
    TooManySubsteps,
    NonFiniteIntermediate,
    NonPhysicalTemperature,
    EnergyBalanceFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalColumn {
    parameters: ThermalColumnParameters,
    state: ThermalColumnState,
}

impl ThermalColumn {
    pub fn new(
        parameters: ThermalColumnParameters,
        state: ThermalColumnState,
    ) -> Result<Self, ThermalColumnError> {
        validate_parameters(parameters)?;
        validate_state(&state)?;
        if state.temperatures_k.len() < 2 {
            return Err(ThermalColumnError::InvalidState);
        }
        Ok(Self { parameters, state })
    }

    pub fn state(&self) -> &ThermalColumnState {
        &self.state
    }

    pub fn parameters(&self) -> ThermalColumnParameters {
        self.parameters
    }

    /// Apply one exposure/cooling interval transactionally. Any invalid
    /// intermediate leaves the original state untouched.
    pub fn apply_pulse(
        &mut self,
        pulse: ThermalPulse,
    ) -> Result<ThermalStepReceipt, ThermalColumnError> {
        validate_pulse(pulse)?;
        let parameters = self.parameters;
        let cell_count = self.state.temperatures_k.len();
        let initial_temperatures = self.state.temperatures_k.clone();
        let mut temperatures = initial_temperatures.clone();
        let surface_temperature_before_k = temperatures[0];
        let areal_heat_capacity =
            parameters.bulk_density_kg_m3
                * parameters.specific_heat_j_kg_k
                * parameters.cell_thickness_m;
        let conductivity_per_length =
            parameters.conductivity_w_m_k / parameters.cell_thickness_m;
        let incident_energy_j_m2 =
            pulse.incident_irradiance_w_m2 * pulse.duration_seconds;
        let energy_delivered_after_optics_j_m2 =
            incident_energy_j_m2 * pulse.optical_transmission;
        let optical_path_loss_j_m2 =
            incident_energy_j_m2 - energy_delivered_after_optics_j_m2;
        let total_depth_m = cell_count as f64 * parameters.cell_thickness_m;
        let absorbed_fraction = pulse.surface_absorptivity
            * (1.0 - (-total_depth_m / parameters.optical_absorption_depth_m).exp());
        let expected_absorbed_energy_j_m2 =
            energy_delivered_after_optics_j_m2 * absorbed_fraction;
        let unabsorbed_energy_j_m2 =
            energy_delivered_after_optics_j_m2 - expected_absorbed_energy_j_m2;

        let mut absorbed_energy_j_m2 = 0.0;
        let mut net_radiative_exchange_j_m2 = 0.0;
        let mut latent_heat_removed_j_m2 = 0.0;
        let mut remaining_seconds = pulse.duration_seconds;
        let mut substeps = 0usize;

        while remaining_seconds > 0.0 {
            if substeps >= MAX_SUBSTEPS {
                return Err(ThermalColumnError::TooManySubsteps);
            }

            let surface_temperature = temperatures[0];
            let radiative_derivative = 4.0
                * parameters.surface_emissivity
                * STEFAN_BOLTZMANN_W_M2_K4
                * surface_temperature.powi(3);
            let maximum_row_coupling =
                2.0 * conductivity_per_length + radiative_derivative;
            if !maximum_row_coupling.is_finite() || maximum_row_coupling <= 0.0 {
                return Err(ThermalColumnError::NonFiniteIntermediate);
            }
            let stable_step_seconds =
                STABILITY_FACTOR * areal_heat_capacity / maximum_row_coupling;
            if !stable_step_seconds.is_finite() || stable_step_seconds <= 0.0 {
                return Err(ThermalColumnError::NonFiniteIntermediate);
            }
            let step_seconds = remaining_seconds.min(stable_step_seconds);
            if step_seconds <= 0.0 {
                return Err(ThermalColumnError::NonFiniteIntermediate);
            }

            let radiative_flux_w_m2 = parameters.surface_emissivity
                * STEFAN_BOLTZMANN_W_M2_K4
                * (surface_temperature.powi(4) - parameters.ambient_temperature_k.powi(4));
            let latent_flux_w_m2 =
                pulse.volatile_mass_flux_kg_m2_s * pulse.latent_heat_j_kg;
            if !radiative_flux_w_m2.is_finite() || !latent_flux_w_m2.is_finite() {
                return Err(ThermalColumnError::NonFiniteIntermediate);
            }

            let mut next_temperatures = vec![0.0; cell_count];
            for cell_index in 0..cell_count {
                let depth_top_m = cell_index as f64 * parameters.cell_thickness_m;
                let depth_bottom_m = depth_top_m + parameters.cell_thickness_m;
                let absorption_fraction = (-depth_top_m
                    / parameters.optical_absorption_depth_m)
                    .exp()
                    - (-depth_bottom_m / parameters.optical_absorption_depth_m).exp();
                let absorbed_power_w_m2 = energy_delivered_after_optics_j_m2
                    / pulse.duration_seconds
                    * pulse.surface_absorptivity
                    * absorption_fraction;

                let conductive_power_w_m2 = if cell_index == 0 {
                    conductivity_per_length * (temperatures[1] - temperatures[0])
                        - radiative_flux_w_m2
                        - latent_flux_w_m2
                } else if cell_index + 1 == cell_count {
                    conductivity_per_length
                        * (temperatures[cell_index - 1] - temperatures[cell_index])
                } else {
                    conductivity_per_length
                        * (temperatures[cell_index - 1] - temperatures[cell_index])
                        + conductivity_per_length
                            * (temperatures[cell_index + 1] - temperatures[cell_index])
                };

                let net_power_w_m2 = absorbed_power_w_m2 + conductive_power_w_m2;
                let temperature_change_k =
                    net_power_w_m2 * step_seconds / areal_heat_capacity;
                let next_temperature = temperatures[cell_index] + temperature_change_k;
                if !absorbed_power_w_m2.is_finite()
                    || !next_temperature.is_finite()
                    || next_temperature <= 0.0
                {
                    return Err(ThermalColumnError::NonPhysicalTemperature);
                }
                next_temperatures[cell_index] = next_temperature;
                absorbed_energy_j_m2 += absorbed_power_w_m2 * step_seconds;
            }

            net_radiative_exchange_j_m2 += radiative_flux_w_m2 * step_seconds;
            latent_heat_removed_j_m2 += latent_flux_w_m2 * step_seconds;
            temperatures = next_temperatures;
            remaining_seconds = (remaining_seconds - step_seconds).max(0.0);
            substeps += 1;
        }

        let internal_energy_change_j_m2 = areal_heat_capacity
            * temperatures
                .iter()
                .zip(&initial_temperatures)
                .map(|(after, before)| after - before)
                .sum::<f64>();
        let energy_residual_j_m2 = absorbed_energy_j_m2
            - net_radiative_exchange_j_m2
            - latent_heat_removed_j_m2
            - internal_energy_change_j_m2;
        let energy_scale = absorbed_energy_j_m2
            .abs()
            .max(net_radiative_exchange_j_m2.abs())
            .max(latent_heat_removed_j_m2.abs())
            .max(internal_energy_change_j_m2.abs())
            .max(1.0);
        let energy_residual_fraction = energy_residual_j_m2.abs() / energy_scale;
        if !energy_residual_fraction.is_finite() {
            return Err(ThermalColumnError::NonFiniteIntermediate);
        }
        if energy_residual_fraction > ENERGY_BALANCE_TOLERANCE {
            return Err(ThermalColumnError::EnergyBalanceFailure);
        }

        let peak_column_temperature_k = temperatures
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let receipt = ThermalStepReceipt {
            incident_energy_j_m2,
            energy_delivered_after_optics_j_m2,
            optical_path_loss_j_m2,
            absorbed_energy_in_column_j_m2: absorbed_energy_j_m2,
            unabsorbed_energy_j_m2,
            net_radiative_exchange_j_m2,
            latent_heat_removed_j_m2,
            internal_energy_change_j_m2,
            energy_residual_j_m2,
            energy_residual_fraction,
            substeps,
            surface_temperature_before_k,
            surface_temperature_after_k: temperatures[0],
            peak_column_temperature_k,
            simulation_only: true,
        };

        self.state.temperatures_k = temperatures;
        self.state.elapsed_seconds += pulse.duration_seconds;
        Ok(receipt)
    }
}

fn validate_parameters(
    parameters: ThermalColumnParameters,
) -> Result<(), ThermalColumnError> {
    if !parameters.bulk_density_kg_m3.is_finite()
        || parameters.bulk_density_kg_m3 <= 0.0
        || !parameters.specific_heat_j_kg_k.is_finite()
        || parameters.specific_heat_j_kg_k <= 0.0
        || !parameters.conductivity_w_m_k.is_finite()
        || parameters.conductivity_w_m_k <= 0.0
        || !parameters.cell_thickness_m.is_finite()
        || parameters.cell_thickness_m <= 0.0
        || !parameters.optical_absorption_depth_m.is_finite()
        || parameters.optical_absorption_depth_m <= 0.0
        || !parameters.surface_emissivity.is_finite()
        || !(0.0..=1.0).contains(&parameters.surface_emissivity)
        || !parameters.ambient_temperature_k.is_finite()
        || parameters.ambient_temperature_k <= 0.0
    {
        return Err(ThermalColumnError::InvalidParameters);
    }
    Ok(())
}

fn validate_state(state: &ThermalColumnState) -> Result<(), ThermalColumnError> {
    if !state.elapsed_seconds.is_finite()
        || state.elapsed_seconds < 0.0
        || state
            .temperatures_k
            .iter()
            .any(|temperature| !temperature.is_finite() || *temperature <= 0.0)
    {
        return Err(ThermalColumnError::InvalidState);
    }
    Ok(())
}

fn validate_pulse(pulse: ThermalPulse) -> Result<(), ThermalColumnError> {
    if !pulse.incident_irradiance_w_m2.is_finite()
        || pulse.incident_irradiance_w_m2 < 0.0
        || !pulse.duration_seconds.is_finite()
        || pulse.duration_seconds <= 0.0
        || !pulse.optical_transmission.is_finite()
        || !(0.0..=1.0).contains(&pulse.optical_transmission)
        || !pulse.surface_absorptivity.is_finite()
        || !(0.0..=1.0).contains(&pulse.surface_absorptivity)
        || !pulse.volatile_mass_flux_kg_m2_s.is_finite()
        || pulse.volatile_mass_flux_kg_m2_s < 0.0
        || !pulse.latent_heat_j_kg.is_finite()
        || pulse.latent_heat_j_kg < 0.0
    {
        return Err(ThermalColumnError::InvalidPulse);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> ThermalColumnParameters {
        ThermalColumnParameters {
            bulk_density_kg_m3: 1_500.0,
            specific_heat_j_kg_k: 800.0,
            conductivity_w_m_k: 0.1,
            cell_thickness_m: 0.001,
            optical_absorption_depth_m: 0.001,
            surface_emissivity: 0.8,
            ambient_temperature_k: 200.0,
        }
    }

    fn column(initial_temperature_k: f64) -> ThermalColumn {
        ThermalColumn::new(
            parameters(),
            ThermalColumnState {
                temperatures_k: vec![initial_temperature_k; 20],
                elapsed_seconds: 0.0,
            },
        )
        .unwrap()
    }

    fn pulse() -> ThermalPulse {
        ThermalPulse {
            incident_irradiance_w_m2: 10_000.0,
            duration_seconds: 2.0,
            optical_transmission: 1.0,
            surface_absorptivity: 0.9,
            volatile_mass_flux_kg_m2_s: 0.0,
            latent_heat_j_kg: 0.0,
        }
    }

    #[test]
    fn pulsed_heating_resolves_a_surface_to_depth_gradient() {
        let mut world = column(200.0);
        world.apply_pulse(pulse()).unwrap();
        let temperatures = &world.state().temperatures_k;
        assert!(temperatures[0] > temperatures[1]);
        assert!(temperatures[1] > temperatures[5]);
        assert!(temperatures[5] > temperatures[19]);
    }

    #[test]
    fn optical_path_attenuation_reduces_absorbed_column_energy() {
        let mut clean_path = column(200.0);
        let mut fouled_path = column(200.0);
        let clean = clean_path.apply_pulse(pulse()).unwrap();
        let fouled = fouled_path
            .apply_pulse(ThermalPulse {
                optical_transmission: 0.5,
                ..pulse()
            })
            .unwrap();
        assert!(
            (fouled.absorbed_energy_in_column_j_m2
                / clean.absorbed_energy_in_column_j_m2
                - 0.5)
                .abs()
                < 1.0e-9
        );
        assert!(fouled.optical_path_loss_j_m2 > 0.0);
    }

    #[test]
    fn energy_accounting_closes_for_exposure_with_latent_sink() {
        let mut world = column(200.0);
        let receipt = world
            .apply_pulse(ThermalPulse {
                volatile_mass_flux_kg_m2_s: 1.0e-5,
                latent_heat_j_kg: 2.0e6,
                ..pulse()
            })
            .unwrap();
        assert!(receipt.energy_residual_fraction < ENERGY_BALANCE_TOLERANCE);
        assert!(receipt.latent_heat_removed_j_m2 > 0.0);
        assert!(receipt.simulation_only);
    }

    #[test]
    fn radiation_cools_a_hot_column_toward_the_environment() {
        let mut world = column(500.0);
        let receipt = world
            .apply_pulse(ThermalPulse {
                incident_irradiance_w_m2: 0.0,
                duration_seconds: 1.0,
                optical_transmission: 1.0,
                surface_absorptivity: 0.9,
                volatile_mass_flux_kg_m2_s: 0.0,
                latent_heat_j_kg: 0.0,
            })
            .unwrap();
        assert!(receipt.net_radiative_exchange_j_m2 > 0.0);
        assert!(receipt.surface_temperature_after_k < 500.0);
        assert!(world.state().elapsed_seconds > 0.0);
    }

    #[test]
    fn invalid_input_rolls_back_the_entire_state() {
        let mut world = column(200.0);
        let before = world.state().clone();
        let result = world.apply_pulse(ThermalPulse {
            optical_transmission: 1.1,
            ..pulse()
        });
        assert_eq!(result, Err(ThermalColumnError::InvalidPulse));
        assert_eq!(world.state(), &before);
    }

    #[test]
    fn identical_initial_conditions_and_pulses_replay_exactly() {
        let mut first = column(200.0);
        let mut second = column(200.0);
        let first_receipt = first.apply_pulse(pulse()).unwrap();
        let second_receipt = second.apply_pulse(pulse()).unwrap();
        assert_eq!(first_receipt, second_receipt);
        assert_eq!(first.state(), second.state());
    }

    #[test]
    fn exposure_receipt_separates_incident_delivery_and_absorption() {
        let mut world = column(200.0);
        let receipt = world
            .apply_pulse(ThermalPulse {
                optical_transmission: 0.8,
                ..pulse()
            })
            .unwrap();
        assert!(
            (receipt.incident_energy_j_m2
                - receipt.energy_delivered_after_optics_j_m2
                - receipt.optical_path_loss_j_m2)
                .abs()
                < 1.0e-8
        );
        assert!(
            (receipt.energy_delivered_after_optics_j_m2
                - receipt.absorbed_energy_in_column_j_m2
                - receipt.unabsorbed_energy_j_m2)
                .abs()
                < 1.0e-8
        );
    }

    #[test]
    fn excessive_latent_heat_fails_without_committing_partial_state() {
        let mut world = column(200.0);
        let before = world.state().clone();
        let result = world.apply_pulse(ThermalPulse {
            incident_irradiance_w_m2: 0.0,
            duration_seconds: 10.0,
            optical_transmission: 1.0,
            surface_absorptivity: 0.9,
            volatile_mass_flux_kg_m2_s: 10.0,
            latent_heat_j_kg: 1.0e6,
        });
        assert!(result.is_err());
        assert_eq!(world.state(), &before);
    }
}
