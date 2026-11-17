// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Stateful optical contamination model for the B1 optical-ISRU twin.
// This module keeps production-optic throughput, witness optics, QCM
// deposition mass and source reference signals as separate observations.
//
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/optical_health.rs -o /tmp/isru-optical-health
//   /tmp/isru-optical-health

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

const BAND_COUNT: usize = 4;
const SPECIES_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpectralBand {
    Ultraviolet,
    Visible,
    NearInfrared,
    ShortwaveInfrared,
}

impl SpectralBand {
    fn index(self) -> usize {
        match self {
            Self::Ultraviolet => 0,
            Self::Visible => 1,
            Self::NearInfrared => 2,
            Self::ShortwaveInfrared => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContaminantSpecies {
    OrganicFilm,
    VolatileSalt,
    MineralDust,
}

impl ContaminantSpecies {
    fn index(self) -> usize {
        match self {
            Self::OrganicFilm => 0,
            Self::VolatileSalt => 1,
            Self::MineralDust => 2,
        }
    }

    const ALL: [Self; SPECIES_COUNT] = [Self::OrganicFilm, Self::VolatileSalt, Self::MineralDust];
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceState {
    pub surface_id: String,
    pub location: String,
    pub temperature_k: f64,
    pub area_cm2: f64,
    /// Areal loading by contaminant species in µg/cm².
    pub loading_ug_cm2: [f64; SPECIES_COUNT],
    pub roughness_nm: f64,
}

impl SurfaceState {
    pub fn validate(&self) -> Result<(), OpticalHealthError> {
        if self.surface_id.trim().is_empty() || self.location.trim().is_empty() {
            return Err(OpticalHealthError::EmptyIdentity("surface"));
        }
        for (name, value) in [
            ("temperature_k", self.temperature_k),
            ("area_cm2", self.area_cm2),
            ("roughness_nm", self.roughness_nm),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OpticalHealthError::InvalidSurfaceParameter(name));
            }
        }
        if self.temperature_k == 0.0 || self.area_cm2 == 0.0 {
            return Err(OpticalHealthError::InvalidSurfaceParameter(
                "temperature_or_area_zero",
            ));
        }
        if self
            .loading_ug_cm2
            .iter()
            .any(|loading| !loading.is_finite() || *loading < 0.0)
        {
            return Err(OpticalHealthError::InvalidSurfaceParameter(
                "contaminant_loading",
            ));
        }
        Ok(())
    }

    pub fn total_loading_ug_cm2(&self) -> f64 {
        self.loading_ug_cm2.iter().sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepositionTransfer {
    /// Dimensionless relative areal-loading factors. They represent measured
    /// geometry/temperature transfer functions, not universal constants.
    pub process_optic_factor: f64,
    pub optical_witness_factor: f64,
    pub qcm_factor: f64,
}

impl DepositionTransfer {
    pub fn validate(&self) -> Result<(), OpticalHealthError> {
        for (name, value) in [
            ("process_optic_factor", self.process_optic_factor),
            ("optical_witness_factor", self.optical_witness_factor),
            ("qcm_factor", self.qcm_factor),
        ] {
            if !value.is_finite() || value < 0.0 || value > 100.0 {
                return Err(OpticalHealthError::InvalidTransferFactor(name));
            }
        }
        if self.process_optic_factor == 0.0
            || self.optical_witness_factor == 0.0
            || self.qcm_factor == 0.0
        {
            return Err(OpticalHealthError::InvalidTransferFactor(
                "zero_transfer_factor",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepositionEvent {
    pub event_id: String,
    pub deposited_flux_ug_cm2: [f64; SPECIES_COUNT],
    pub transfer: DepositionTransfer,
    pub source_temperature_k: f64,
    pub recorded_at_seconds: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpticalHealthReading {
    pub reading_id: String,
    pub source_reference_power_w: [f64; BAND_COUNT],
    pub process_channel_power_w: [f64; BAND_COUNT],
    /// Ratio-normalized process-path transmission relative to the clean
    /// baseline, indexed by SpectralBand.
    pub normalized_process_transmission: [f64; BAND_COUNT],
    /// Witness sample transmittance is a separate surface observation.
    pub witness_transmission: [f64; BAND_COUNT],
    /// QCM local areal mass. This is never used as an optical-transmission
    /// substitute.
    pub qcm_areal_mass_ug_cm2: f64,
    pub observed_at_seconds: u64,
    pub simulation_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpticalHealthThresholds {
    pub minimum_process_transmission: [f64; BAND_COUNT],
    pub maximum_witness_loss: [f64; BAND_COUNT],
    pub maximum_qcm_mass_ug_cm2: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthAction {
    Continue,
    RequestIndependentInspection,
    CleanAndRequalify,
    PauseCampaign,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CleaningProtocol {
    pub protocol_id: String,
    /// Fraction removed by contaminant species; independently qualified for
    /// each substrate and geometry in physical use.
    pub removal_fraction: [f64; SPECIES_COUNT],
    /// Signed change in surface roughness; positive values may represent
    /// cleaning damage. This model does not assume cleaning is harmless.
    pub roughness_delta_nm: f64,
    pub duration_seconds: u64,
    pub energy_wh: f64,
    pub consumables_g: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CleaningReceipt {
    pub protocol_id: String,
    pub process_loading_before_ug_cm2: f64,
    pub process_loading_after_ug_cm2: f64,
    pub witness_loading_before_ug_cm2: f64,
    pub witness_loading_after_ug_cm2: f64,
    pub qcm_loading_before_ug_cm2: f64,
    pub qcm_loading_after_ug_cm2: f64,
    pub roughness_delta_nm: f64,
    pub duration_seconds: u64,
    pub energy_wh: f64,
    pub consumables_g: f64,
    pub requalification_required: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpticalHealthError {
    EmptyIdentity(&'static str),
    InvalidSurfaceParameter(&'static str),
    InvalidTransferFactor(&'static str),
    InvalidDeposition(String),
    InvalidOpticalModel(&'static str),
    InvalidNoise,
    InvalidThreshold(&'static str),
    InvalidReading(&'static str),
    StaleReading,
    InvalidCleaningProtocol(&'static str),
    DuplicateDepositionEvent(String),
    TimeWentBackwards,
    RequalificationNotDue,
    RequalificationFailed,
}

pub struct OpticalHealthModel {
    process_optic: SurfaceState,
    witness: SurfaceState,
    qcm: SurfaceState,
    clean_baseline_transmission: [f64; BAND_COUNT],
    extinction_coefficients_cm2_per_ug: [[f64; BAND_COUNT]; SPECIES_COUNT],
    source_power_w: f64,
    sensor_noise_stddev: f64,
    last_event_time_seconds: u64,
    last_observation_time_seconds: u64,
    seen_deposition_events: BTreeMap<String, u64>,
    requalification_required: bool,
    qualified: bool,
}

impl OpticalHealthModel {
    pub fn new(
        process_optic: SurfaceState,
        witness: SurfaceState,
        qcm: SurfaceState,
        clean_baseline_transmission: [f64; BAND_COUNT],
        extinction_coefficients_cm2_per_ug: [[f64; BAND_COUNT]; SPECIES_COUNT],
        source_power_w: f64,
        sensor_noise_stddev: f64,
    ) -> Result<Self, OpticalHealthError> {
        process_optic.validate()?;
        witness.validate()?;
        qcm.validate()?;
        for (name, value) in [
            ("source_power_w", source_power_w),
            ("sensor_noise_stddev", sensor_noise_stddev),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(OpticalHealthError::InvalidOpticalModel(name));
            }
        }
        if clean_baseline_transmission
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
        {
            return Err(OpticalHealthError::InvalidOpticalModel(
                "clean_baseline_transmission",
            ));
        }
        if extinction_coefficients_cm2_per_ug
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(OpticalHealthError::InvalidOpticalModel(
                "extinction_coefficients",
            ));
        }

        Ok(Self {
            process_optic,
            witness,
            qcm,
            clean_baseline_transmission,
            extinction_coefficients_cm2_per_ug,
            source_power_w,
            sensor_noise_stddev,
            last_event_time_seconds: 0,
            last_observation_time_seconds: 0,
            seen_deposition_events: BTreeMap::new(),
            requalification_required: false,
            qualified: true,
        })
    }

    pub fn apply_deposition(&mut self, event: DepositionEvent) -> Result<(), OpticalHealthError> {
        if event.event_id.trim().is_empty() {
            return Err(OpticalHealthError::EmptyIdentity("event_id"));
        }
        if self.seen_deposition_events.contains_key(&event.event_id) {
            return Err(OpticalHealthError::DuplicateDepositionEvent(event.event_id));
        }
        if event.recorded_at_seconds < self.last_event_time_seconds
            || event.recorded_at_seconds < self.last_observation_time_seconds
        {
            return Err(OpticalHealthError::TimeWentBackwards);
        }
        event.transfer.validate()?;
        if !event.source_temperature_k.is_finite() || event.source_temperature_k <= 0.0 {
            return Err(OpticalHealthError::InvalidDeposition(
                "source_temperature".to_string(),
            ));
        }
        if event
            .deposited_flux_ug_cm2
            .iter()
            .any(|mass| !mass.is_finite() || *mass < 0.0)
        {
            return Err(OpticalHealthError::InvalidDeposition(
                "deposited_flux".to_string(),
            ));
        }

        add_scaled_loading(
            &mut self.process_optic.loading_ug_cm2,
            event.deposited_flux_ug_cm2,
            event.transfer.process_optic_factor,
        );
        add_scaled_loading(
            &mut self.witness.loading_ug_cm2,
            event.deposited_flux_ug_cm2,
            event.transfer.optical_witness_factor,
        );
        add_scaled_loading(
            &mut self.qcm.loading_ug_cm2,
            event.deposited_flux_ug_cm2,
            event.transfer.qcm_factor,
        );
        self.last_event_time_seconds = event.recorded_at_seconds;
        self.seen_deposition_events
            .insert(event.event_id, event.recorded_at_seconds);
        Ok(())
    }

    /// Change source output independently of the optical path. A clean
    /// reference channel observes the same source and allows ratio
    /// normalization to separate source drift from path transmission loss.
    pub fn set_source_power(&mut self, power_w: f64) -> Result<(), OpticalHealthError> {
        if !power_w.is_finite() || power_w <= 0.0 {
            return Err(OpticalHealthError::InvalidOpticalModel("source_power_w"));
        }
        self.source_power_w = power_w;
        Ok(())
    }

    pub fn inspect(
        &mut self,
        reading_id: &str,
        observed_at_seconds: u64,
        seed: u64,
    ) -> Result<OpticalHealthReading, OpticalHealthError> {
        if reading_id.trim().is_empty() {
            return Err(OpticalHealthError::EmptyIdentity("reading_id"));
        }
        if observed_at_seconds < self.last_event_time_seconds
            || observed_at_seconds < self.last_observation_time_seconds
        {
            return Err(OpticalHealthError::TimeWentBackwards);
        }

        let process_truth = self.process_transmission();
        let witness_truth = self.witness_transmission();
        let mut source_reference = [0.0; BAND_COUNT];
        let mut process_channel = [0.0; BAND_COUNT];
        let mut normalized_process_transmission = [0.0; BAND_COUNT];
        let mut witness_transmission = [0.0; BAND_COUNT];

        for band in 0..BAND_COUNT {
            let band_seed = seed ^ ((band as u64 + 1) * 0x9e37_79b9);
            source_reference[band] = self.source_power_w
                * (1.0 + deterministic_noise(band_seed ^ 0x1111, self.sensor_noise_stddev));
            process_channel[band] = self.source_power_w
                * process_truth[band]
                * (1.0 + deterministic_noise(band_seed ^ 0x2222, self.sensor_noise_stddev));
            normalized_process_transmission[band] = (process_channel[band]
                / source_reference[band])
                / self.clean_baseline_transmission[band];
            witness_transmission[band] = witness_truth[band]
                * (1.0 + deterministic_noise(band_seed ^ 0x3333, self.sensor_noise_stddev));
        }

        let qcm_mass = self.qcm.total_loading_ug_cm2()
            * (1.0 + deterministic_noise(seed ^ 0x4444, self.sensor_noise_stddev));

        let reading = OpticalHealthReading {
            reading_id: reading_id.to_string(),
            source_reference_power_w: source_reference,
            process_channel_power_w: process_channel,
            normalized_process_transmission,
            witness_transmission,
            qcm_areal_mass_ug_cm2: qcm_mass.max(0.0),
            observed_at_seconds,
            simulation_only: true,
        };
        self.last_observation_time_seconds = observed_at_seconds;
        Ok(reading)
    }

    pub fn decide(
        &self,
        reading: &OpticalHealthReading,
        thresholds: &OpticalHealthThresholds,
    ) -> Result<HealthAction, OpticalHealthError> {
        validate_reading(reading)?;
        if reading.observed_at_seconds < self.last_event_time_seconds
            || reading.observed_at_seconds < self.last_observation_time_seconds
        {
            return Err(OpticalHealthError::StaleReading);
        }
        validate_thresholds(thresholds)?;
        if !self.qualified || self.requalification_required {
            return Ok(HealthAction::PauseCampaign);
        }
        evaluate_health_thresholds(reading, thresholds)
    }

    pub fn clean(
        &mut self,
        protocol: &CleaningProtocol,
        at_seconds: u64,
    ) -> Result<CleaningReceipt, OpticalHealthError> {
        if protocol.protocol_id.trim().is_empty() {
            return Err(OpticalHealthError::EmptyIdentity("protocol_id"));
        }
        if protocol.duration_seconds == 0
            || !protocol.energy_wh.is_finite()
            || protocol.energy_wh < 0.0
            || !protocol.consumables_g.is_finite()
            || protocol.consumables_g < 0.0
            || !protocol.roughness_delta_nm.is_finite()
            || protocol
                .removal_fraction
                .iter()
                .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err(OpticalHealthError::InvalidCleaningProtocol(
                "invalid_cost_or_removal_fraction",
            ));
        }
        if at_seconds < self.last_event_time_seconds
            || at_seconds < self.last_observation_time_seconds
        {
            return Err(OpticalHealthError::TimeWentBackwards);
        }

        let process_before = self.process_optic.total_loading_ug_cm2();
        let witness_before = self.witness.total_loading_ug_cm2();
        let qcm_before = self.qcm.total_loading_ug_cm2();

        remove_contamination(&mut self.process_optic, protocol);
        remove_contamination(&mut self.witness, protocol);
        remove_contamination(&mut self.qcm, protocol);

        self.process_optic.roughness_nm =
            (self.process_optic.roughness_nm + protocol.roughness_delta_nm).max(0.0);
        self.witness.roughness_nm =
            (self.witness.roughness_nm + protocol.roughness_delta_nm).max(0.0);
        self.qcm.roughness_nm = (self.qcm.roughness_nm + protocol.roughness_delta_nm).max(0.0);
        self.last_event_time_seconds = at_seconds;
        self.requalification_required = true;
        self.qualified = false;

        Ok(CleaningReceipt {
            protocol_id: protocol.protocol_id.clone(),
            process_loading_before_ug_cm2: process_before,
            process_loading_after_ug_cm2: self.process_optic.total_loading_ug_cm2(),
            witness_loading_before_ug_cm2: witness_before,
            witness_loading_after_ug_cm2: self.witness.total_loading_ug_cm2(),
            qcm_loading_before_ug_cm2: qcm_before,
            qcm_loading_after_ug_cm2: self.qcm.total_loading_ug_cm2(),
            roughness_delta_nm: protocol.roughness_delta_nm,
            duration_seconds: protocol.duration_seconds,
            energy_wh: protocol.energy_wh,
            consumables_g: protocol.consumables_g,
            requalification_required: true,
        })
    }

    /// Requalification must follow a cleaning action and requires an observed
    /// post-clean reading that meets the process-optic and witness limits.
    pub fn requalify(
        &mut self,
        reading: &OpticalHealthReading,
        thresholds: &OpticalHealthThresholds,
    ) -> Result<(), OpticalHealthError> {
        if !self.requalification_required {
            return Err(OpticalHealthError::RequalificationNotDue);
        }
        validate_reading(reading)?;
        if reading.observed_at_seconds < self.last_event_time_seconds
            || reading.observed_at_seconds < self.last_observation_time_seconds
        {
            return Err(OpticalHealthError::StaleReading);
        }
        validate_thresholds(thresholds)?;
        if evaluate_health_thresholds(reading, thresholds)? != HealthAction::Continue {
            return Err(OpticalHealthError::RequalificationFailed);
        }
        self.requalification_required = false;
        self.qualified = true;
        Ok(())
    }

    pub fn process_transmission(&self) -> [f64; BAND_COUNT] {
        spectral_transmission(
            &self.process_optic,
            &self.extinction_coefficients_cm2_per_ug,
        )
    }

    pub fn witness_transmission(&self) -> [f64; BAND_COUNT] {
        spectral_transmission(&self.witness, &self.extinction_coefficients_cm2_per_ug)
    }

    pub fn qcm_areal_mass_ug_cm2(&self) -> f64 {
        self.qcm.total_loading_ug_cm2()
    }

    pub fn qualified(&self) -> bool {
        self.qualified && !self.requalification_required
    }

    pub fn source_power_w(&self) -> f64 {
        self.source_power_w
    }
}

fn validate_thresholds(
    thresholds: &OpticalHealthThresholds,
) -> Result<(), OpticalHealthError> {
    if thresholds
        .minimum_process_transmission
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        || thresholds
            .maximum_witness_loss
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(OpticalHealthError::InvalidThreshold("spectral_thresholds"));
    }
    if !thresholds.maximum_qcm_mass_ug_cm2.is_finite()
        || thresholds.maximum_qcm_mass_ug_cm2 < 0.0
    {
        return Err(OpticalHealthError::InvalidThreshold("qcm_mass"));
    }
    Ok(())
}

fn evaluate_health_thresholds(
    reading: &OpticalHealthReading,
    thresholds: &OpticalHealthThresholds,
) -> Result<HealthAction, OpticalHealthError> {
    let process_below_limit = reading
        .normalized_process_transmission
        .iter()
        .zip(thresholds.minimum_process_transmission.iter())
        .any(|(observed, minimum)| observed < minimum);
    if process_below_limit {
        return Ok(HealthAction::CleanAndRequalify);
    }

    let witness_loss_exceeded = reading
        .witness_transmission
        .iter()
        .zip(thresholds.maximum_witness_loss.iter())
        .any(|(observed, maximum_loss)| (1.0 - observed) > *maximum_loss);
    if witness_loss_exceeded
        || reading.qcm_areal_mass_ug_cm2 > thresholds.maximum_qcm_mass_ug_cm2
    {
        return Ok(HealthAction::RequestIndependentInspection);
    }

    Ok(HealthAction::Continue)
}

fn validate_reading(reading: &OpticalHealthReading) -> Result<(), OpticalHealthError> {
    if reading.reading_id.trim().is_empty() {
        return Err(OpticalHealthError::EmptyIdentity("reading_id"));
    }
    if !reading.simulation_only {
        return Err(OpticalHealthError::InvalidReading("not_simulation_only"));
    }
    if reading
        .source_reference_power_w
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || reading
            .process_channel_power_w
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || reading
            .normalized_process_transmission
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || reading
            .witness_transmission
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || !reading.qcm_areal_mass_ug_cm2.is_finite()
        || reading.qcm_areal_mass_ug_cm2 < 0.0
    {
        return Err(OpticalHealthError::InvalidReading(
            "non_finite_or_negative_value",
        ));
    }
    Ok(())
}

fn add_scaled_loading(target: &mut [f64; SPECIES_COUNT], flux: [f64; SPECIES_COUNT], factor: f64) {
    for species in ContaminantSpecies::ALL {
        let index = species.index();
        target[index] += flux[index] * factor;
    }
}

fn remove_contamination(surface: &mut SurfaceState, protocol: &CleaningProtocol) {
    for species in ContaminantSpecies::ALL {
        let index = species.index();
        surface.loading_ug_cm2[index] *= 1.0 - protocol.removal_fraction[index];
    }
}

fn spectral_transmission(
    surface: &SurfaceState,
    extinction: &[[f64; BAND_COUNT]; SPECIES_COUNT],
) -> [f64; BAND_COUNT] {
    let mut output = [1.0; BAND_COUNT];
    for band in 0..BAND_COUNT {
        let mut optical_depth = surface.roughness_nm * 0.000_001;
        for species in ContaminantSpecies::ALL {
            let index = species.index();
            optical_depth += surface.loading_ug_cm2[index] * extinction[index][band];
        }
        output[band] = (-optical_depth).exp().clamp(0.0, 1.0);
    }
    output
}

/// Reproducible bounded noise for B1. This is deliberately not described as
/// Gaussian; physical sensor-noise models must be estimated from instrument
/// calibration data before a B2 claim.
fn deterministic_noise(seed: u64, relative_bound: f64) -> f64 {
    let mut state = if seed == 0 {
        0xa076_1d64_78bd_642f
    } else {
        seed
    };
    let value = next_random(&mut state);
    let uniform = (value >> 11) as f64 / ((1_u64 << 53) as f64);
    (2.0 * uniform - 1.0) * relative_bound
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (*state).wrapping_mul(0x2545_f491_4f6c_dd1d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(id: &str, location: &str, temp: f64) -> SurfaceState {
        SurfaceState {
            surface_id: id.to_string(),
            location: location.to_string(),
            temperature_k: temp,
            area_cm2: 10.0,
            loading_ug_cm2: [0.0; SPECIES_COUNT],
            roughness_nm: 0.0,
        }
    }

    fn model() -> OpticalHealthModel {
        OpticalHealthModel::new(
            surface("production-optic", "main beam path", 300.0),
            surface("witness", "representative witness plane", 300.0),
            surface("qcm", "QCM monitor plane", 300.0),
            [1.0; BAND_COUNT],
            [
                [0.2, 0.1, 0.05, 0.01],
                [0.1, 0.05, 0.2, 0.3],
                [0.05, 0.1, 0.15, 0.2],
            ],
            10.0,
            0.001,
        )
        .unwrap()
    }

    fn transfer() -> DepositionTransfer {
        DepositionTransfer {
            process_optic_factor: 1.0,
            optical_witness_factor: 0.9,
            qcm_factor: 1.2,
        }
    }

    fn event(id: &str, time: u64) -> DepositionEvent {
        DepositionEvent {
            event_id: id.to_string(),
            deposited_flux_ug_cm2: [2.0, 1.0, 0.5],
            transfer: transfer(),
            source_temperature_k: 600.0,
            recorded_at_seconds: time,
        }
    }

    fn thresholds() -> OpticalHealthThresholds {
        OpticalHealthThresholds {
            minimum_process_transmission: [0.4; BAND_COUNT],
            maximum_witness_loss: [0.2; BAND_COUNT],
            maximum_qcm_mass_ug_cm2: 10.0,
        }
    }

    #[test]
    fn clean_system_starts_qualified_and_transmits() {
        let value = model();
        assert!(value.qualified());
        assert_eq!(value.process_transmission(), [1.0; BAND_COUNT]);
    }

    #[test]
    fn deposition_changes_optical_and_qcm_surfaces_independently() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        assert!(value.process_optic.total_loading_ug_cm2() > 0.0);
        assert!(value.witness.total_loading_ug_cm2() > 0.0);
        assert!(value.qcm.total_loading_ug_cm2() > 0.0);
        assert_ne!(
            value.process_optic.total_loading_ug_cm2(),
            value.qcm.total_loading_ug_cm2()
        );
    }

    #[test]
    fn qcm_mass_is_not_used_as_a_proxy_for_optical_transmission() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        let qcm_mass = value.qcm_areal_mass_ug_cm2();
        let process = value.process_transmission();
        assert!(qcm_mass > 0.0);
        assert!(process.iter().all(|transmission| *transmission < 1.0));
        assert_ne!(qcm_mass, 1.0 - process[0]);
    }

    #[test]
    fn source_drift_is_normalized_by_independent_reference_channel() {
        let mut value = model();
        let before = value.inspect("reading-1", 1, 7).unwrap();
        value.set_source_power(5.0).unwrap();
        let after = value.inspect("reading-2", 2, 7).unwrap();
        for band in 0..BAND_COUNT {
            assert!(
                (before.normalized_process_transmission[band]
                    - after.normalized_process_transmission[band])
                    .abs()
                    < 0.01
            );
        }
        assert!(after.source_reference_power_w[0] < before.source_reference_power_w[0]);
    }

    #[test]
    fn wavelength_dependent_contamination_is_preserved() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        let transmission = value.process_transmission();
        assert!(transmission[0] < transmission[3]);
    }

    #[test]
    fn health_decision_pauses_when_process_optic_throughput_falls_below_limit() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        let reading = value.inspect("reading-1", 10, 9).unwrap();
        let mut tight = thresholds();
        tight.minimum_process_transmission = [0.95; BAND_COUNT];
        assert_eq!(
            value.decide(&reading, &tight),
            Ok(HealthAction::CleanAndRequalify)
        );
    }

    #[test]
    fn qcm_threshold_triggers_independent_inspection_not_optical_substitution() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        let reading = value.inspect("reading-1", 10, 9).unwrap();
        let mut qcm_sensitive = thresholds();
        qcm_sensitive.maximum_qcm_mass_ug_cm2 = 0.1;
        assert_eq!(
            value.decide(&reading, &qcm_sensitive),
            Ok(HealthAction::RequestIndependentInspection)
        );
    }

    #[test]
    fn cleaning_records_residuals_and_requires_requalification() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        let receipt = value
            .clean(
                &CleaningProtocol {
                    protocol_id: "clean-v1".to_string(),
                    removal_fraction: [0.9, 0.8, 0.5],
                    roughness_delta_nm: 2.0,
                    duration_seconds: 60,
                    energy_wh: 5.0,
                    consumables_g: 1.0,
                },
                20,
            )
            .unwrap();
        assert!(receipt.process_loading_after_ug_cm2 < receipt.process_loading_before_ug_cm2);
        assert!(receipt.requalification_required);
        assert!(!value.qualified());
        assert_eq!(
            {
                let reading = value.inspect("post-clean", 80, 10).unwrap();
                value.decide(&reading, &thresholds())
            },
            Ok(HealthAction::PauseCampaign)
        );
    }

    #[test]
    fn successful_post_clean_inspection_is_required_before_requalification() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        value
            .clean(
                &CleaningProtocol {
                    protocol_id: "clean-v1".to_string(),
                    removal_fraction: [1.0, 1.0, 1.0],
                    roughness_delta_nm: 0.0,
                    duration_seconds: 10,
                    energy_wh: 1.0,
                    consumables_g: 0.1,
                },
                20,
            )
            .unwrap();
        let reading = value.inspect("post-clean", 30, 1).unwrap();
        value.requalify(&reading, &thresholds()).unwrap();
        assert!(value.qualified());
    }

    #[test]
    fn failed_requalification_keeps_campaign_unqualified() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        value
            .clean(
                &CleaningProtocol {
                    protocol_id: "clean-v1".to_string(),
                    removal_fraction: [0.0, 0.0, 0.0],
                    roughness_delta_nm: 10.0,
                    duration_seconds: 10,
                    energy_wh: 1.0,
                    consumables_g: 0.1,
                },
                20,
            )
            .unwrap();
        let reading = value.inspect("post-clean", 30, 1).unwrap();
        assert_eq!(
            value.requalify(&reading, &thresholds()),
            Err(OpticalHealthError::RequalificationFailed)
        );
        assert!(!value.qualified());
    }

    #[test]
    fn deposition_cannot_be_backdated_before_an_already_observed_reading() {
        let mut value = model();
        let reading = value.inspect("reading-1", 20, 4).unwrap();
        assert_eq!(reading.observed_at_seconds, 20);
        assert_eq!(
            value.apply_deposition(event("late-event", 19)),
            Err(OpticalHealthError::TimeWentBackwards)
        );
        value.apply_deposition(event("valid-event", 21)).unwrap();
    }

    #[test]
    fn an_older_reading_cannot_override_a_newer_inspection() {
        let mut value = model();
        let older = value.inspect("reading-old", 20, 4).unwrap();
        let newer = value.inspect("reading-new", 21, 4).unwrap();
        assert_eq!(
            value.decide(&older, &thresholds()),
            Err(OpticalHealthError::StaleReading)
        );
        assert_eq!(
            value.decide(&newer, &thresholds()),
            Ok(HealthAction::Continue)
        );
    }

    #[test]
    fn inspection_time_cannot_move_backwards() {
        let mut value = model();
        value.inspect("reading-1", 20, 4).unwrap();
        assert_eq!(
            value.inspect("reading-2", 19, 4),
            Err(OpticalHealthError::TimeWentBackwards)
        );
    }

    #[test]
    fn duplicate_and_backdated_deposition_events_are_rejected() {
        let mut value = model();
        value.apply_deposition(event("dep-1", 10)).unwrap();
        assert_eq!(
            value.apply_deposition(event("dep-1", 11)),
            Err(OpticalHealthError::DuplicateDepositionEvent(
                "dep-1".to_string()
            ))
        );
        assert_eq!(
            value.apply_deposition(event("dep-2", 9)),
            Err(OpticalHealthError::TimeWentBackwards)
        );
    }

    #[test]
    fn surface_locations_and_temperature_are_explicit() {
        let value = model();
        assert_eq!(value.process_optic.location, "main beam path");
        assert_eq!(value.witness.location, "representative witness plane");
        assert_eq!(value.qcm.location, "QCM monitor plane");
    }
}
