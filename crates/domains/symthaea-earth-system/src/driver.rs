// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-neutral environmental-driver exports for downstream models.
//!
//! The exported records preserve model state, units, and diagnostics without
//! inventing ecological productivity, disturbance, or demographic responses.
//! Those assumptions belong in the receiving ecology model.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::hydrology::HydrologySample;
use crate::latitude::LatitudinalEnergyBalanceModel;
use crate::nutrient::NutrientSample;
use crate::productivity::{ProductivityLedger, ProductivityLimitation};
use crate::soil_carbon::SoilCarbonSample;
use crate::transient::{OneBoxSample, TwoBoxSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDriverSample {
    pub time_seconds: f64,
    pub temperature: f64,
    pub temperature_anomaly: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrologyDriverSample {
    pub time_days: f64,
    pub storage_mm: f64,
    pub soil_moisture_fraction: f64,
    pub water_deficit_fraction: f64,
    pub actual_evapotranspiration_mm_per_day: f64,
    pub runoff_mm_per_day: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilCarbonDriverSample {
    pub time_years: f64,
    pub temperature_k: f64,
    pub fast_carbon: f64,
    pub slow_carbon: f64,
    pub total_carbon: f64,
    pub respiration_rate_per_year: f64,
    pub budget_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientDriverSample {
    pub time: f64,
    pub organic_pool: f64,
    pub mineral_pool: f64,
    pub mineralization_flux: f64,
    pub uptake_flux: f64,
    pub leaching_flux: f64,
    pub budget_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductivityDriverSample {
    pub duration: f64,
    pub environmental_multiplier: f64,
    pub gross_primary_production: f64,
    pub net_primary_production: f64,
    pub retained_biomass_carbon: f64,
    pub litter_carbon: f64,
    pub nutrient_uptake: f64,
    pub remaining_mineral_nutrient: f64,
    pub limitation: ProductivityLimitation,
    pub carbon_budget_residual: f64,
    pub nutrient_budget_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatitudeBandDriverSample {
    pub band: usize,
    pub latitude_radians: f64,
    /// Equal-area fraction of the globe represented by this band.
    pub area_fraction: f64,
    pub temperature: f64,
    pub temperature_anomaly: f64,
}

/// Export one equal-area zonal temperature state without inventing ecological
/// productivity or disturbance fields.
pub fn latitude_temperature_drivers(
    model: &LatitudinalEnergyBalanceModel,
    temperatures: &[f64],
) -> Result<Vec<LatitudeBandDriverSample>, ModelError> {
    model.validate()?;
    model.global_mean_temperature(temperatures)?;
    let area_fraction = 1.0 / model.bands as f64;
    Ok(temperatures
        .iter()
        .enumerate()
        .map(|(band, temperature)| LatitudeBandDriverSample {
            band,
            latitude_radians: model.latitude_radians(band),
            area_fraction,
            temperature: *temperature,
            temperature_anomaly: *temperature - model.reference_temperature,
        })
        .collect())
}

/// Export conserved bucket states without assigning biological meaning to
/// water stress. Soil-moisture fraction is bounded in `[0, 1]`.
pub fn hydrology_drivers(
    samples: &[HydrologySample],
) -> Result<Vec<HydrologyDriverSample>, ModelError> {
    if samples.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "hydrology trajectory",
        });
    }
    let mut output = Vec::with_capacity(samples.len());
    let mut previous_time = None;
    for (index, sample) in samples.iter().enumerate() {
        require_non_negative("time_days", sample.time_days)?;
        require_non_negative("storage_mm", sample.storage_mm)?;
        require_non_negative(
            "actual_evapotranspiration_mm_per_day",
            sample.actual_evapotranspiration_mm_per_day,
        )?;
        require_non_negative("runoff_mm_per_day", sample.runoff_mm_per_day)?;
        if !(0.0..=1.0).contains(&sample.soil_moisture_fraction) {
            return Err(ModelError::OutOfRange {
                parameter: "soil_moisture_fraction",
                value: sample.soil_moisture_fraction,
                min: 0.0,
                max: 1.0,
            });
        }
        if let Some(previous) = previous_time
            && sample.time_days <= previous
        {
            return Err(ModelError::NonMonotonicTime {
                index,
                previous,
                current: sample.time_days,
            });
        }
        previous_time = Some(sample.time_days);
        output.push(HydrologyDriverSample {
            time_days: sample.time_days,
            storage_mm: sample.storage_mm,
            soil_moisture_fraction: sample.soil_moisture_fraction,
            water_deficit_fraction: 1.0 - sample.soil_moisture_fraction,
            actual_evapotranspiration_mm_per_day: sample.actual_evapotranspiration_mm_per_day,
            runoff_mm_per_day: sample.runoff_mm_per_day,
        });
    }
    Ok(output)
}

/// Export soil-carbon state and respiration diagnostics without converting them
/// into productivity or atmospheric concentration.
pub fn soil_carbon_drivers(
    samples: &[SoilCarbonSample],
) -> Result<Vec<SoilCarbonDriverSample>, ModelError> {
    if samples.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "soil-carbon trajectory",
        });
    }
    let mut output = Vec::with_capacity(samples.len());
    let mut previous_time = None;
    for (index, sample) in samples.iter().enumerate() {
        require_non_negative("time_years", sample.time_years)?;
        require_positive("temperature_k", sample.temperature_k)?;
        require_non_negative("fast_carbon", sample.fast_carbon)?;
        require_non_negative("slow_carbon", sample.slow_carbon)?;
        require_non_negative("total_carbon", sample.total_carbon)?;
        require_non_negative(
            "respiration_rate_per_year",
            sample.respiration_rate_per_year,
        )?;
        require_finite("budget_residual", sample.budget_residual)?;
        if let Some(previous) = previous_time
            && sample.time_years <= previous
        {
            return Err(ModelError::NonMonotonicTime {
                index,
                previous,
                current: sample.time_years,
            });
        }
        previous_time = Some(sample.time_years);
        output.push(SoilCarbonDriverSample {
            time_years: sample.time_years,
            temperature_k: sample.temperature_k,
            fast_carbon: sample.fast_carbon,
            slow_carbon: sample.slow_carbon,
            total_carbon: sample.total_carbon,
            respiration_rate_per_year: sample.respiration_rate_per_year,
            budget_residual: sample.budget_residual,
        });
    }
    Ok(output)
}

/// Export nutrient-cycle states and fluxes without assigning a species-level
/// response to mineral availability.
pub fn nutrient_drivers(
    samples: &[NutrientSample],
) -> Result<Vec<NutrientDriverSample>, ModelError> {
    if samples.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "nutrient trajectory",
        });
    }
    let mut output = Vec::with_capacity(samples.len());
    let mut previous_time = None;
    for (index, sample) in samples.iter().enumerate() {
        require_non_negative("nutrient_time", sample.time)?;
        require_non_negative("organic_pool", sample.state.organic_pool)?;
        require_non_negative("mineral_pool", sample.state.mineral_pool)?;
        require_non_negative("mineralization_flux", sample.mineralization_flux)?;
        require_non_negative("uptake_flux", sample.uptake_flux)?;
        require_non_negative("leaching_flux", sample.leaching_flux)?;
        require_finite("nutrient_budget_residual", sample.budget_residual)?;
        if let Some(previous) = previous_time
            && sample.time <= previous
        {
            return Err(ModelError::NonMonotonicTime {
                index,
                previous,
                current: sample.time,
            });
        }
        previous_time = Some(sample.time);
        output.push(NutrientDriverSample {
            time: sample.time,
            organic_pool: sample.state.organic_pool,
            mineral_pool: sample.state.mineral_pool,
            mineralization_flux: sample.mineralization_flux,
            uptake_flux: sample.uptake_flux,
            leaching_flux: sample.leaching_flux,
            budget_residual: sample.budget_residual,
        });
    }
    Ok(output)
}

/// Export a finite-interval productivity ledger without converting it into a
/// demographic growth rate or carrying capacity.
pub fn productivity_driver(
    ledger: ProductivityLedger,
) -> Result<ProductivityDriverSample, ModelError> {
    require_non_negative("productivity_duration", ledger.duration)?;
    crate::error::require_fraction("environmental_multiplier", ledger.environmental_multiplier)?;
    require_non_negative("gross_primary_production", ledger.gross_primary_production)?;
    require_non_negative("net_primary_production", ledger.net_primary_production)?;
    require_non_negative("retained_biomass_carbon", ledger.retained_biomass_carbon)?;
    require_non_negative("litter_carbon", ledger.litter_carbon)?;
    require_non_negative("nutrient_uptake", ledger.nutrient_uptake)?;
    require_non_negative(
        "remaining_mineral_nutrient",
        ledger.remaining_mineral_nutrient,
    )?;
    require_finite("carbon_budget_residual", ledger.carbon_budget_residual)?;
    require_finite("nutrient_budget_residual", ledger.nutrient_budget_residual)?;
    Ok(ProductivityDriverSample {
        duration: ledger.duration,
        environmental_multiplier: ledger.environmental_multiplier,
        gross_primary_production: ledger.gross_primary_production,
        net_primary_production: ledger.net_primary_production,
        retained_biomass_carbon: ledger.retained_biomass_carbon,
        litter_carbon: ledger.litter_carbon,
        nutrient_uptake: ledger.nutrient_uptake,
        remaining_mineral_nutrient: ledger.remaining_mineral_nutrient,
        limitation: ledger.limitation,
        carbon_budget_residual: ledger.carbon_budget_residual,
        nutrient_budget_residual: ledger.nutrient_budget_residual,
    })
}

pub fn one_box_temperature_drivers(
    samples: &[OneBoxSample],
    reference_temperature: f64,
) -> Result<Vec<TemperatureDriverSample>, ModelError> {
    require_positive("reference_temperature", reference_temperature)?;
    export_temperature_drivers(
        samples
            .iter()
            .map(|sample| (sample.time_seconds, sample.temperature)),
        samples.len(),
        reference_temperature,
    )
}

pub fn two_box_surface_temperature_drivers(
    samples: &[TwoBoxSample],
    reference_temperature: f64,
) -> Result<Vec<TemperatureDriverSample>, ModelError> {
    require_positive("reference_temperature", reference_temperature)?;
    export_temperature_drivers(
        samples
            .iter()
            .map(|sample| (sample.time_seconds, sample.state.surface_temperature)),
        samples.len(),
        reference_temperature,
    )
}

fn export_temperature_drivers<I>(
    values: I,
    length: usize,
    reference_temperature: f64,
) -> Result<Vec<TemperatureDriverSample>, ModelError>
where
    I: IntoIterator<Item = (f64, f64)>,
{
    if length == 0 {
        return Err(ModelError::EmptySeries {
            series: "temperature trajectory",
        });
    }
    let mut output = Vec::with_capacity(length);
    let mut previous_time = None;
    for (index, (time_seconds, temperature)) in values.into_iter().enumerate() {
        crate::error::require_non_negative("time_seconds", time_seconds)?;
        require_positive("temperature", temperature)?;
        if let Some(previous) = previous_time
            && time_seconds <= previous
        {
            return Err(ModelError::NonMonotonicTime {
                index,
                previous,
                current: time_seconds,
            });
        }
        previous_time = Some(time_seconds);
        output.push(TemperatureDriverSample {
            time_seconds,
            temperature,
            temperature_anomaly: temperature - reference_temperature,
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_box_export_preserves_time_and_anomaly() {
        let samples = [
            OneBoxSample {
                time_seconds: 0.0,
                temperature: 288.0,
                forcing: 0.0,
                radiative_imbalance: 0.0,
            },
            OneBoxSample {
                time_seconds: 1.0,
                temperature: 289.5,
                forcing: 2.0,
                radiative_imbalance: 0.2,
            },
        ];
        let drivers = one_box_temperature_drivers(&samples, 288.0).unwrap();
        assert_eq!(drivers.len(), 2);
        assert_eq!(drivers[0].temperature_anomaly, 0.0);
        assert!((drivers[1].temperature_anomaly - 1.5).abs() < 1e-12);
    }

    #[test]
    fn non_monotonic_export_fails_closed() {
        let samples = [
            OneBoxSample {
                time_seconds: 1.0,
                temperature: 288.0,
                forcing: 0.0,
                radiative_imbalance: 0.0,
            },
            OneBoxSample {
                time_seconds: 1.0,
                temperature: 289.0,
                forcing: 0.0,
                radiative_imbalance: 0.0,
            },
        ];
        assert!(matches!(
            one_box_temperature_drivers(&samples, 288.0),
            Err(ModelError::NonMonotonicTime { .. })
        ));
    }

    #[test]
    fn latitude_export_preserves_equal_area_and_zonal_anomalies() {
        let model = LatitudinalEnergyBalanceModel::earthlike(8).unwrap();
        let temperatures: Vec<_> = (0..model.bands).map(|band| 280.0 + band as f64).collect();
        let drivers = latitude_temperature_drivers(&model, &temperatures).unwrap();
        assert_eq!(drivers.len(), model.bands);
        assert!(
            (drivers
                .iter()
                .map(|driver| driver.area_fraction)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1.0e-12
        );
        assert_eq!(drivers[3].temperature, 283.0);
        assert_eq!(
            drivers[3].temperature_anomaly,
            283.0 - model.reference_temperature
        );
        assert!(drivers.first().unwrap().latitude_radians < 0.0);
        assert!(drivers.last().unwrap().latitude_radians > 0.0);
    }

    #[test]
    fn hydrology_export_preserves_moisture_and_deficit_complement() {
        let model = crate::hydrology::HydrologyBucket::try_new(100.0, 5.0).unwrap();
        let samples = model.exact_trajectory(50.0, 2.5, 1.0, 3).unwrap();
        let drivers = hydrology_drivers(&samples).unwrap();
        assert_eq!(drivers.len(), 4);
        assert!(drivers.iter().all(|driver| {
            (driver.soil_moisture_fraction + driver.water_deficit_fraction - 1.0).abs() < 1.0e-12
        }));
    }

    #[test]
    fn soil_carbon_export_preserves_budget_evidence() {
        let model = crate::soil_carbon::TwoPoolSoilCarbon::illustrative();
        let samples = model
            .exact_trajectory(10.0, 90.0, 3.0, 283.15, 1.0, 4)
            .unwrap();
        let drivers = soil_carbon_drivers(&samples).unwrap();
        assert_eq!(drivers.len(), 5);
        assert!(
            drivers
                .iter()
                .all(|driver| driver.budget_residual.abs() < 1.0e-10)
        );
    }

    #[test]
    fn nutrient_export_preserves_flux_and_budget_evidence() {
        let model = crate::nutrient::TwoPoolNutrientCycle::try_new(0.2, 0.08, 0.02).unwrap();
        let samples = model
            .exact_trajectory(
                crate::nutrient::NutrientState {
                    organic_pool: 5.0,
                    mineral_pool: 2.0,
                },
                4.0,
                1.0,
                1.0,
                4,
            )
            .unwrap();
        let drivers = nutrient_drivers(&samples).unwrap();
        assert_eq!(drivers.len(), 5);
        assert!(
            drivers
                .iter()
                .all(|driver| driver.budget_residual.abs() < 1.0e-10)
        );
        assert!(drivers.last().unwrap().uptake_flux > 0.0);
    }

    #[test]
    fn productivity_export_preserves_explicit_limitation() {
        let ledger =
            crate::productivity::EcosystemProductivityModel::try_new(10.0, 20.0, 0.4, 0.25)
                .unwrap()
                .account_interval(10.0, 1.0, 1.0)
                .unwrap();
        let driver = productivity_driver(ledger).unwrap();
        assert_eq!(driver.limitation, ProductivityLimitation::Nutrient);
        assert!(driver.carbon_budget_residual.abs() < 1.0e-12);
        assert!(driver.nutrient_budget_residual.abs() < 1.0e-12);
    }
}
