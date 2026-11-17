// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Coupled B1 pulse transaction: the current production-optic state attenuates
// the next pulse, and material emissions update optic, witness and QCM state.
//
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/coupled_campaign.rs -o /tmp/isru-coupled
//   /tmp/isru-coupled

#![forbid(unsafe_code)]

#[path = "material_regime.rs"]
mod material_regime;
#[path = "optical_health.rs"]
mod optical_health;

use material_regime::{
    LaserPulse, MaterialParameters, MaterialRegimeModel, MaterialState, RegimeError,
    TrapAndGasMeasurement,
};
use optical_health::{
    DepositionEvent, DepositionTransfer, HealthAction, OpticalHealthError, OpticalHealthModel,
    OpticalHealthReading, OpticalHealthThresholds, SpectralBand, SurfaceState,
};


#[derive(Clone, Debug, PartialEq)]
pub enum CoupledError {
    Material(RegimeError),
    Optical(OpticalHealthError),
    CampaignBlocked(HealthAction),
    InvalidPulseIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPulseReceipt {
    pub pulse_id: String,
    pub pre_pulse_health: OpticalHealthReading,
    pub applied_transmission: f64,
    pub absorbed_energy_j: f64,
    pub captured_water_kg: f64,
    pub gas_analyzer_water_signal_kg: f64,
    pub deposited_flux_ug_cm2: [f64; 3],
    pub material_elapsed_seconds: u64,
    pub simulation_only: bool,
}

/// Simulation-only coupled campaign. It cannot create physical calibration,
/// assay, custody or authority evidence.
#[derive(Clone)]
pub struct CoupledCampaign {
    material: MaterialRegimeModel,
    optics: OpticalHealthModel,
    thresholds: OpticalHealthThresholds,
    transfer: DepositionTransfer,
    process_band: SpectralBand,
}

impl CoupledCampaign {
    pub fn new(
        material: MaterialRegimeModel,
        optics: OpticalHealthModel,
        thresholds: OpticalHealthThresholds,
        transfer: DepositionTransfer,
        process_band: SpectralBand,
    ) -> Result<Self, OpticalHealthError> {
        transfer.validate()?;
        Ok(Self {
            material,
            optics,
            thresholds,
            transfer,
            process_band,
        })
    }

    /// A pulse is atomic across material and optical state: either the material
    /// response and its deposition both validate and commit, or neither does.
    pub fn apply_pulse(
        &mut self,
        pulse_id: &str,
        incident_irradiance_w_cm2: f64,
        beam_area_cm2: f64,
        on_duration_seconds: f64,
        cooling_pause_seconds: f64,
        gas: &TrapAndGasMeasurement,
        source_temperature_k: f64,
        seed: u64,
    ) -> Result<CoupledPulseReceipt, CoupledError> {
        if pulse_id.trim().is_empty() {
            return Err(CoupledError::InvalidPulseIdentity);
        }

        let mut staged_material = self.material.clone();
        let mut staged_optics = self.optics.clone();
        let observed_at = staged_material.state().elapsed_seconds;
        let reading = staged_optics
            .inspect(&format!("{pulse_id}-pre"), observed_at, seed)
            .map_err(CoupledError::Optical)?;
        let action = staged_optics
            .decide(&reading, &self.thresholds)
            .map_err(CoupledError::Optical)?;
        if action != HealthAction::Continue {
            return Err(CoupledError::CampaignBlocked(action));
        }

        let transmission = staged_optics.process_transmission()[self.process_band.index()];
        let pulse = LaserPulse {
            incident_irradiance_w_cm2,
            beam_area_cm2,
            on_duration_seconds,
            cooling_pause_seconds,
            process_optic_transmission: transmission,
        };
        let result = staged_material
            .apply_pulse(&pulse, gas)
            .map_err(CoupledError::Material)?;

        let deposition = DepositionEvent {
            event_id: format!("{pulse_id}-deposition"),
            deposited_flux_ug_cm2: result.optical_deposition_estimate_ug_cm2,
            transfer: self.transfer.clone(),
            source_temperature_k,
            recorded_at_seconds: result.elapsed_seconds,
        };
        staged_optics
            .apply_deposition(deposition)
            .map_err(CoupledError::Optical)?;

        let receipt = CoupledPulseReceipt {
            pulse_id: pulse_id.to_string(),
            pre_pulse_health: reading,
            applied_transmission: transmission,
            absorbed_energy_j: result.absorbed_energy_j,
            captured_water_kg: result.captured_water_kg,
            gas_analyzer_water_signal_kg: result.gas_analyzer_water_signal_kg,
            deposited_flux_ug_cm2: result.optical_deposition_estimate_ug_cm2,
            material_elapsed_seconds: result.elapsed_seconds,
            simulation_only: true,
        };

        self.material = staged_material;
        self.optics = staged_optics;
        Ok(receipt)
    }

    pub fn material_state(&self) -> &MaterialState {
        self.material.state()
    }

    pub fn process_transmission(&self) -> [f64; 4] {
        self.optics.process_transmission()
    }

    pub fn witness_transmission(&self) -> [f64; 4] {
        self.optics.witness_transmission()
    }

    pub fn qcm_areal_mass_ug_cm2(&self) -> f64 {
        self.optics.qcm_areal_mass_ug_cm2()
    }

    pub fn is_qualified(&self) -> bool {
        self.optics.qualified()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material() -> MaterialRegimeModel {
        MaterialRegimeModel::new(
            MaterialParameters {
                bulk_density_kg_m3: 1_000.0,
                specific_heat_j_kg_k: 500.0,
                active_thermal_depth_m: 0.01,
                water_mass_fraction: 0.10,
                organic_mass_fraction: 0.05,
                salt_mass_fraction: 0.02,
                spallation_onset_k: 400.0,
                volatile_release_onset_k: 450.0,
                brittle_ductile_transition_k: 700.0,
                melt_onset_k: 900.0,
                melt_transition_width_k: 100.0,
                spallation_yield_per_pulse: 0.2,
                volatile_release_efficiency: 0.8,
                melt_growth_time_constant_s: 5.0,
                melt_suppression_exponent: 1.0,
                ductile_spallation_floor: 0.1,
                cooling_time_constant_s: 30.0,
                surface_refresh_efficiency: 0.5,
                surface_depletion_rate: 0.1,
                organic_deposition_fraction: 0.8,
                airborne_salt_fraction: 0.2,
                airborne_dust_fraction: 0.1,
            },
            MaterialState {
                ambient_temperature_k: 300.0,
                surface_temperature_k: 300.0,
                remaining_solid_kg: 10.0,
                remaining_water_kg: 1.0,
                remaining_organic_kg: 0.5,
                remaining_salt_kg: 0.2,
                surface_depletion_fraction: 0.0,
                melt_layer_fraction: 0.0,
                elapsed_seconds: 0,
            },
            1.0,
            100.0,
        )
        .unwrap()
    }

    fn optics() -> OpticalHealthModel {
        let surface = |id: &str, location: &str, area: f64| SurfaceState {
            surface_id: id.to_string(),
            location: location.to_string(),
            temperature_k: 300.0,
            area_cm2: area,
            loading_ug_cm2: [0.0; 3],
            roughness_nm: 1.0,
        };
        OpticalHealthModel::new(
            surface("process", "beam path", 100.0),
            surface("witness", "representative witness plane", 10.0),
            surface("qcm", "near-optic monitor", 1.0),
            [1.0; 4],
            [[0.01; 4], [0.005; 4], [0.002; 4]],
            100.0,
            0.000_001,
        )
        .unwrap()
    }

    fn campaign() -> CoupledCampaign {
        CoupledCampaign::new(
            material(),
            optics(),
            OpticalHealthThresholds {
                minimum_process_transmission: [0.01; 4],
                maximum_witness_loss: [0.99; 4],
                maximum_qcm_mass_ug_cm2: 1.0e9,
            },
            DepositionTransfer {
                process_optic_factor: 1.0,
                optical_witness_factor: 0.5,
                qcm_factor: 0.25,
            },
            SpectralBand::Visible,
        )
        .unwrap()
    }

    fn gas() -> TrapAndGasMeasurement {
        TrapAndGasMeasurement {
            trap_capture_efficiency: 0.8,
            purge_gas_flow_kg_s: 0.0,
            purge_gas_water_mass_fraction: 0.0,
        }
    }

    #[test]
    fn deposition_from_one_pulse_attenuates_the_next_pulse() {
        let mut world = campaign();
        let first = world
            .apply_pulse("pulse-1", 100.0, 1.0, 1.0, 0.0, &gas(), 500.0, 7)
            .unwrap();
        let after_first = world.process_transmission()[SpectralBand::Visible.index()];
        let second = world
            .apply_pulse("pulse-2", 100.0, 1.0, 1.0, 0.0, &gas(), 500.0, 8)
            .unwrap();
        assert!(first.deposited_flux_ug_cm2.iter().any(|mass| *mass > 0.0));
        assert!(after_first < 1.0);
        assert!(second.applied_transmission < first.applied_transmission);
        assert!(second.absorbed_energy_j < first.absorbed_energy_j);
    }

    #[test]
    fn witness_and_qcm_remain_distinct_observations() {
        let mut world = campaign();
        world
            .apply_pulse("pulse-1", 100.0, 1.0, 1.0, 0.0, &gas(), 500.0, 7)
            .unwrap();
        assert!(world.qcm_areal_mass_ug_cm2() > 0.0);
        assert!(world.witness_transmission()[SpectralBand::Visible.index()] < 1.0);
        assert!(world.process_transmission()[SpectralBand::Visible.index()] < 1.0);
        assert_ne!(
            world.qcm_areal_mass_ug_cm2(),
            1.0 - world.witness_transmission()[SpectralBand::Visible.index()]
        );
    }

    #[test]
    fn failed_deposition_rolls_back_material_state() {
        let mut world = campaign();
        let before = world.material_state().clone();
        let result = world.apply_pulse("rollback", 100.0, 1.0, 1.0, 0.0, &gas(), -1.0, 7);
        assert!(matches!(result, Err(CoupledError::Optical(_))));
        assert_eq!(*world.material_state(), before);
        assert_eq!(world.process_transmission(), [1.0; 4]);
    }

    #[test]
    fn wet_purge_changes_signal_not_captured_product() {
        let mut dry = campaign();
        let mut wet = campaign();
        let dry_receipt = dry
            .apply_pulse("dry", 100.0, 1.0, 1.0, 0.0, &gas(), 500.0, 7)
            .unwrap();
        let wet_gas = TrapAndGasMeasurement {
            purge_gas_flow_kg_s: 0.01,
            purge_gas_water_mass_fraction: 0.1,
            ..gas()
        };
        let wet_receipt = wet
            .apply_pulse("wet", 100.0, 1.0, 1.0, 0.0, &wet_gas, 500.0, 7)
            .unwrap();
        assert_eq!(dry_receipt.captured_water_kg, wet_receipt.captured_water_kg);
        assert!(
            wet_receipt.gas_analyzer_water_signal_kg > dry_receipt.gas_analyzer_water_signal_kg
        );
    }

    #[test]
    fn receipt_never_claims_physical_measurement() {
        let mut world = campaign();
        let receipt = world
            .apply_pulse("sim-only", 100.0, 1.0, 1.0, 0.0, &gas(), 500.0, 7)
            .unwrap();
        assert!(receipt.simulation_only);
        assert!(receipt.pre_pulse_health.simulation_only);
    }
}
