//! Plasma-specific metadata wrapper around the shared decision diagnosability
//! and sensor-cut oracle.
//!
//! This module does not implement another sensor-cut solver. It binds generic
//! structural signatures to a declared machine/regime context and preserves
//! missing engineering metadata as unknown. Lifecycle cost is reported only
//! when every selected diagnostic has a declared cost.

use serde::{Deserialize, Serialize};
use symthaea_observability::{
    CommonModeCutSet, DecisionWorld, DiagnosabilityError, DiagnosabilityModel,
    DiagnosabilityResult, DiagnosticSensor, SensorCutSet, SensorSet,
};

pub const PLASMA_DIAGNOSTIC_PORTFOLIO_SCHEMA: &str =
    "luminous.plasma-diagnostic-portfolio.v0.1";
pub const PLASMA_PORTFOLIO_CLAIM_CEILING: &str =
    "SYNTHETIC_STRUCTURAL_DECISION_DIAGNOSABILITY_ONLY_NO_PHYSICAL_SENSOR_OR_RELIABILITY_CLAIM";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlasmaPortfolioError {
    pub message: String,
}

impl PlasmaPortfolioError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PlasmaPortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PlasmaPortfolioError {}

impl From<DiagnosabilityError> for PlasmaPortfolioError {
    fn from(value: DiagnosabilityError) -> Self {
        Self::new(value.message)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaDiagnosticCandidate {
    pub diagnostic_system_id: String,
    pub canonical_quantity: String,
    pub units: String,
    pub machine_ids: Vec<String>,
    pub regime_ids: Vec<String>,
    /// None means no defensible sampling-rate value is available.
    pub sampling_rate_hz: Option<f64>,
    /// None means no defensible end-to-end diagnostic latency is available.
    pub measurement_latency_ns: Option<u64>,
    /// None means lifecycle cost is unknown, not zero.
    pub lifecycle_cost_usd: Option<f64>,
    pub calibration_dependency_ids: Vec<String>,
    pub assumption_reference: String,
    pub sensor: DiagnosticSensor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaDiagnosticPortfolioSpec {
    pub schema_version: String,
    pub catalogue_id: String,
    pub machine_id: String,
    pub regime_id: String,
    pub claim_ceiling: String,
    pub worlds: Vec<DecisionWorld>,
    pub candidates: Vec<PlasmaDiagnosticCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlasmaPortfolioSelection {
    pub sensor_ids: Vec<String>,
    /// None if any sensor in the set has unknown lifecycle cost.
    pub estimated_lifecycle_cost_usd: Option<f64>,
    pub sensors_with_unknown_cost: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlasmaDiagnosticPortfolioAnalysis {
    pub catalogue_id: String,
    pub machine_id: String,
    pub regime_id: String,
    pub baseline_diagnosability: DiagnosabilityResult,
    pub minimal_sufficient_portfolios: Vec<PlasmaPortfolioSelection>,
    pub minimum_sensor_cut_sets: Vec<SensorCutSet>,
    pub minimum_common_mode_cut_sets: Vec<CommonModeCutSet>,
    pub candidate_sensors_with_unknown_cost: Vec<String>,
}

impl PlasmaDiagnosticPortfolioSpec {
    /// Validate plasma context and metadata, then delegate structural signature
    /// validation to the shared generic oracle.
    pub fn validate(&self) -> Result<(), PlasmaPortfolioError> {
        if self.schema_version != PLASMA_DIAGNOSTIC_PORTFOLIO_SCHEMA {
            return Err(PlasmaPortfolioError::new(
                "unsupported plasma diagnostic portfolio schema",
            ));
        }
        for (value, field) in [
            (self.catalogue_id.as_str(), "catalogue_id"),
            (self.machine_id.as_str(), "machine_id"),
            (self.regime_id.as_str(), "regime_id"),
            (self.claim_ceiling.as_str(), "claim_ceiling"),
        ] {
            if value.trim().is_empty() {
                return Err(PlasmaPortfolioError::new(format!(
                    "{field} must be non-empty",
                )));
            }
        }
        if self.claim_ceiling != PLASMA_PORTFOLIO_CLAIM_CEILING {
            return Err(PlasmaPortfolioError::new(
                "plasma portfolio claim ceiling is immutable",
            ));
        }
        if self.candidates.is_empty() {
            return Err(PlasmaPortfolioError::new(
                "at least one diagnostic candidate is required",
            ));
        }

        for candidate in &self.candidates {
            for (value, field) in [
                (candidate.diagnostic_system_id.as_str(), "diagnostic_system_id"),
                (candidate.canonical_quantity.as_str(), "canonical_quantity"),
                (candidate.units.as_str(), "units"),
                (candidate.assumption_reference.as_str(), "assumption_reference"),
            ] {
                if value.trim().is_empty() {
                    return Err(PlasmaPortfolioError::new(format!(
                        "candidate {} has empty {field}",
                        candidate.sensor.sensor_id,
                    )));
                }
            }
            if !candidate.machine_ids.iter().any(|id| id == &self.machine_id) {
                return Err(PlasmaPortfolioError::new(format!(
                    "candidate {} is not declared for machine {}",
                    candidate.sensor.sensor_id, self.machine_id
                )));
            }
            if !candidate.regime_ids.iter().any(|id| id == &self.regime_id) {
                return Err(PlasmaPortfolioError::new(format!(
                    "candidate {} is not declared for regime {}",
                    candidate.sensor.sensor_id, self.regime_id
                )));
            }
            if candidate.machine_ids.iter().any(|id| id.trim().is_empty())
                || candidate.regime_ids.iter().any(|id| id.trim().is_empty())
            {
                return Err(PlasmaPortfolioError::new(
                    "machine/regime applicability IDs must be non-empty",
                ));
            }
            if let Some(rate) = candidate.sampling_rate_hz {
                if !rate.is_finite() || rate <= 0.0 {
                    return Err(PlasmaPortfolioError::new(format!(
                        "candidate {} has invalid sampling rate",
                        candidate.sensor.sensor_id
                    )));
                }
            }
            if let Some(cost) = candidate.lifecycle_cost_usd {
                if !cost.is_finite() || cost < 0.0 {
                    return Err(PlasmaPortfolioError::new(format!(
                        "candidate {} has invalid lifecycle cost",
                        candidate.sensor.sensor_id
                    )));
                }
            }

            let mut calibration_ids = std::collections::BTreeSet::new();
            for calibration_id in &candidate.calibration_dependency_ids {
                if calibration_id.trim().is_empty()
                    || !calibration_ids.insert(calibration_id.as_str())
                {
                    return Err(PlasmaPortfolioError::new(format!(
                        "candidate {} has an empty or duplicate calibration dependency",
                        candidate.sensor.sensor_id
                    )));
                }
            }
        }

        self.structural_model().validate()?;
        Ok(())
    }

    /// Build the generic structural model. No numeric reliability, noise,
    /// failure probability or instrument calibration is inferred here.
    pub fn structural_model(&self) -> DiagnosabilityModel {
        DiagnosabilityModel {
            worlds: self.worlds.clone(),
            sensors: self
                .candidates
                .iter()
                .map(|candidate| candidate.sensor.clone())
                .collect(),
        }
    }

    /// Analyze a declared baseline portfolio and return minimal sufficient
    /// portfolios and loss cuts. Missing candidate costs stay explicitly
    /// unknown; the function never substitutes zero.
    pub fn analyze(
        &self,
        baseline_active_sensor_ids: &[&str],
    ) -> Result<PlasmaDiagnosticPortfolioAnalysis, PlasmaPortfolioError> {
        self.validate()?;
        let model = self.structural_model();
        let baseline_diagnosability = model.diagnose(baseline_active_sensor_ids)?;
        let minimal_sets = model.minimal_sufficient_sensor_sets()?;
        let minimum_sensor_cut_sets =
            model.minimal_sensor_cut_sets(baseline_active_sensor_ids)?;
        let minimum_common_mode_cut_sets =
            model.minimal_common_mode_cut_sets(baseline_active_sensor_ids)?;

        let minimal_sufficient_portfolios = minimal_sets
            .into_iter()
            .map(|set| self.portfolio_selection(set))
            .collect::<Result<Vec<_>, _>>()?;

        let mut candidate_sensors_with_unknown_cost: Vec<String> = self
            .candidates
            .iter()
            .filter(|candidate| candidate.lifecycle_cost_usd.is_none())
            .map(|candidate| candidate.sensor.sensor_id.clone())
            .collect();
        candidate_sensors_with_unknown_cost.sort();

        Ok(PlasmaDiagnosticPortfolioAnalysis {
            catalogue_id: self.catalogue_id.clone(),
            machine_id: self.machine_id.clone(),
            regime_id: self.regime_id.clone(),
            baseline_diagnosability,
            minimal_sufficient_portfolios,
            minimum_sensor_cut_sets,
            minimum_common_mode_cut_sets,
            candidate_sensors_with_unknown_cost,
        })
    }

    fn portfolio_selection(
        &self,
        set: SensorSet,
    ) -> Result<PlasmaPortfolioSelection, PlasmaPortfolioError> {
        let mut known_total = 0.0;
        let mut sensors_with_unknown_cost = Vec::new();

        for sensor_id in &set.sensor_ids {
            let candidate = self
                .candidates
                .iter()
                .find(|candidate| candidate.sensor.sensor_id == *sensor_id)
                .ok_or_else(|| {
                    PlasmaPortfolioError::new(format!(
                        "minimal sensor set references unknown candidate {sensor_id}",
                    ))
                })?;
            match candidate.lifecycle_cost_usd {
                Some(cost) => known_total += cost,
                None => sensors_with_unknown_cost.push(sensor_id.clone()),
            }
        }

        let estimated_lifecycle_cost_usd =
            sensors_with_unknown_cost.is_empty().then_some(known_total);
        Ok(PlasmaPortfolioSelection {
            sensor_ids: set.sensor_ids,
            estimated_lifecycle_cost_usd,
            sensors_with_unknown_cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn fixture() -> PlasmaDiagnosticPortfolioSpec {
        let source =
            include_str!("../../../../benchmarks/plasma/diagnostic-portfolio-structural-v0.json");
        serde_json::from_str(source).expect("synthetic plasma portfolio fixture should parse")
    }

    #[test]
    fn frozen_synthetic_plasma_catalogue_is_decision_diagnosable() {
        let spec = fixture();
        spec.validate().expect("fixture should validate");
        let baseline: Vec<&str> = spec
            .candidates
            .iter()
            .map(|candidate| candidate.sensor.sensor_id.as_str())
            .collect();
        let analysis = spec.analyze(&baseline).expect("portfolio analysis");

        assert!(analysis.baseline_diagnosability.diagnosable);
        assert_eq!(analysis.catalogue_id, "PLASMA-PORTFOLIO-STRUCTURAL-SYNTHETIC-001");
        assert_eq!(analysis.machine_id, "synthetic-mast-like");
        assert_eq!(analysis.minimal_sufficient_portfolios.len(), 4);
        assert_eq!(analysis.minimum_sensor_cut_sets.len(), 2);
        assert_eq!(analysis.minimum_common_mode_cut_sets.len(), 2);
    }

    #[test]
    fn minimal_portfolios_preserve_unknown_costs_instead_of_zero_filling() {
        let spec = fixture();
        let baseline: Vec<&str> = spec
            .candidates
            .iter()
            .map(|candidate| candidate.sensor.sensor_id.as_str())
            .collect();
        let analysis = spec.analyze(&baseline).expect("portfolio analysis");

        for portfolio in &analysis.minimal_sufficient_portfolios {
            assert_eq!(portfolio.estimated_lifecycle_cost_usd, None);
            assert_eq!(portfolio.sensors_with_unknown_cost.len(), 2);
        }
        assert_eq!(analysis.candidate_sensors_with_unknown_cost.len(), 5);
    }

    #[test]
    fn minimal_sufficient_portfolios_are_one_density_path_plus_one_mhd_path() {
        let spec = fixture();
        let baseline: Vec<&str> = spec
            .candidates
            .iter()
            .map(|candidate| candidate.sensor.sensor_id.as_str())
            .collect();
        let analysis = spec.analyze(&baseline).expect("portfolio analysis");
        let actual: BTreeSet<Vec<String>> = analysis
            .minimal_sufficient_portfolios
            .iter()
            .map(|portfolio| portfolio.sensor_ids.clone())
            .collect();
        let expected = [
            vec![
                "interferometer_density_a".to_owned(),
                "flux_loop_mhd_b".to_owned(),
            ],
            vec![
                "interferometer_density_a".to_owned(),
                "mirnov_mhd_a".to_owned(),
            ],
            vec![
                "thomson_density_b".to_owned(),
                "flux_loop_mhd_b".to_owned(),
            ],
            vec![
                "thomson_density_b".to_owned(),
                "mirnov_mhd_a".to_owned(),
            ],
        ]
        .into_iter()
        .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn unknown_imaging_control_is_not_counted_as_a_sufficient_sensor() {
        let mut spec = fixture();
        let candidate = spec
            .candidates
            .iter_mut()
            .find(|candidate| candidate.sensor.sensor_id == "unknown_imaging_control")
            .expect("unknown control");
        candidate.sensor.outcomes.iter_mut().for_each(|outcome| {
            outcome.observation = symthaea_observability::SensorObservation::Unknown;
        });

        let model = spec.structural_model();
        let result = model
            .diagnose(&["unknown_imaging_control", "interferometer_density_a"])
            .expect("structural diagnosis");
        assert!(!result.diagnosable);
    }

    #[test]
    fn applicability_and_cost_metadata_fail_closed_when_malformed() {
        let mut wrong_machine = fixture();
        wrong_machine.candidates[0].machine_ids = vec!["another-machine".to_owned()];
        assert!(wrong_machine.validate().is_err());

        let mut invalid_cost = fixture();
        invalid_cost.candidates[0].lifecycle_cost_usd = Some(f64::NAN);
        assert!(invalid_cost.validate().is_err());

        let mut invalid_rate = fixture();
        invalid_rate.candidates[0].sampling_rate_hz = Some(0.0);
        assert!(invalid_rate.validate().is_err());
    }
}
