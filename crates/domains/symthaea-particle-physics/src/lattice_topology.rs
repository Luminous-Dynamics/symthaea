// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive topology/slow-mode diagnostics for lattice Monte Carlo histories.
//!
//! This module consumes an externally defined topological-charge history. It
//! does not define a clover operator, gradient flow, cooling prescription, or
//! any universal threshold for declaring topology adequately sampled.
//!
//! A sampler can therefore have excellent plaquette ESS while this module still
//! exposes a constant/slow topological history as a separate evidence stream.

use crate::lattice_statistics::{
    LatticeStatisticsError, integrated_autocorrelation_time, sample_mean, sample_variance,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyDefinitionLineage {
    /// Versioned topological-charge operator identity, e.g. a clover definition.
    pub operator_id: String,
    /// Versioned smoothing/flow prescription, or `none` for an unsmoothed diagnostic.
    pub smoothing_id: String,
    /// Flow/smoothing time in the convention defined by `smoothing_id`.
    pub smoothing_time: f64,
    pub evidence_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopologySample {
    pub cycle: u64,
    pub charge: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopologyAutocorrelationDiagnostic {
    /// `None` when the observed history has zero variance.
    pub tau_int: Option<f64>,
    /// `None` when the observed history has zero variance.
    pub effective_sample_size: Option<f64>,
    pub positive_lag_count: usize,
    /// Descriptive statement about the observed finite history, not proof that
    /// the underlying Markov chain is mathematically frozen.
    pub zero_variance_observed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectorOccupancy {
    pub sector: i64,
    pub samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyDiagnostics {
    pub definition: TopologyDefinitionLineage,
    pub sample_count: usize,
    pub mean_charge: f64,
    pub mean_charge_squared: f64,
    /// Connected susceptibility `(E[Q^2] - E[Q]^2) / V_4` in lattice units.
    pub connected_susceptibility_lattice_units: f64,
    pub assigned_sector_samples: usize,
    pub ambiguous_sector_samples: usize,
    pub distinct_assigned_sectors: usize,
    pub sector_occupancy: Vec<SectorOccupancy>,
    /// Changes between immediately adjacent, unambiguous assigned sectors.
    /// Ambiguous samples break continuity rather than being bridged across.
    pub adjacent_sector_changes: usize,
    /// Longest run of the same immediately adjacent assigned sector.
    pub max_assigned_sector_dwell: usize,
    pub charge_autocorrelation: TopologyAutocorrelationDiagnostic,
    pub charge_squared_autocorrelation: TopologyAutocorrelationDiagnostic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopologyDiagnosticsError {
    Statistics(LatticeStatisticsError),
    TooFewSamples(usize),
    NonMonotonicCycle { previous: u64, current: u64 },
    NonFiniteCharge { index: usize, value: f64 },
    InvalidSectorTolerance(f64),
    InvalidSmoothingTime(f64),
    InvalidFourVolume(usize),
    EmptyLineageField(&'static str),
    SectorOutOfRange(f64),
}

impl From<LatticeStatisticsError> for TopologyDiagnosticsError {
    fn from(value: LatticeStatisticsError) -> Self {
        Self::Statistics(value)
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), TopologyDiagnosticsError> {
    if value.trim().is_empty() {
        Err(TopologyDiagnosticsError::EmptyLineageField(field))
    } else {
        Ok(())
    }
}

impl TopologyDefinitionLineage {
    pub fn validate(&self) -> Result<(), TopologyDiagnosticsError> {
        require_nonempty(&self.operator_id, "operator_id")?;
        require_nonempty(&self.smoothing_id, "smoothing_id")?;
        require_nonempty(&self.evidence_id, "evidence_id")?;
        if !self.smoothing_time.is_finite() || self.smoothing_time < 0.0 {
            return Err(TopologyDiagnosticsError::InvalidSmoothingTime(
                self.smoothing_time,
            ));
        }
        Ok(())
    }
}

fn autocorrelation_diagnostic(
    values: &[f64],
    max_lag: usize,
) -> Result<TopologyAutocorrelationDiagnostic, TopologyDiagnosticsError> {
    let variance = sample_variance(values)?;
    if variance == 0.0 {
        return Ok(TopologyAutocorrelationDiagnostic {
            tau_int: None,
            effective_sample_size: None,
            positive_lag_count: 0,
            zero_variance_observed: true,
        });
    }
    let (tau_int, positive_lag_count) = integrated_autocorrelation_time(values, max_lag)?;
    Ok(TopologyAutocorrelationDiagnostic {
        tau_int: Some(tau_int),
        effective_sample_size: Some(values.len() as f64 / (2.0 * tau_int)),
        positive_lag_count,
        zero_variance_observed: false,
    })
}

fn assign_sector(charge: f64, tolerance: f64) -> Result<Option<i64>, TopologyDiagnosticsError> {
    let rounded = charge.round();
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(TopologyDiagnosticsError::SectorOutOfRange(charge));
    }
    Ok(((charge - rounded).abs() <= tolerance).then_some(rounded as i64))
}

/// Analyze a retained topological-charge history.
///
/// `sector_tolerance` must be in `[0, 0.5)`. Values farther from the nearest
/// integer are marked ambiguous; no attempt is made to force every sample into
/// an integer sector. No minimum tunneling count or ESS threshold is encoded.
pub fn analyze_topology_history(
    samples: &[TopologySample],
    definition: TopologyDefinitionLineage,
    sector_tolerance: f64,
    max_lag: usize,
    lattice_four_volume: usize,
) -> Result<TopologyDiagnostics, TopologyDiagnosticsError> {
    definition.validate()?;
    if samples.len() < 3 {
        return Err(TopologyDiagnosticsError::TooFewSamples(samples.len()));
    }
    if !sector_tolerance.is_finite() || !(0.0..0.5).contains(&sector_tolerance) {
        return Err(TopologyDiagnosticsError::InvalidSectorTolerance(
            sector_tolerance,
        ));
    }
    if lattice_four_volume == 0 {
        return Err(TopologyDiagnosticsError::InvalidFourVolume(0));
    }

    let mut charges = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        if index > 0 && sample.cycle <= samples[index - 1].cycle {
            return Err(TopologyDiagnosticsError::NonMonotonicCycle {
                previous: samples[index - 1].cycle,
                current: sample.cycle,
            });
        }
        if !sample.charge.is_finite() {
            return Err(TopologyDiagnosticsError::NonFiniteCharge {
                index,
                value: sample.charge,
            });
        }
        charges.push(sample.charge);
    }

    let mean_charge = sample_mean(&charges)?;
    let charge_squared: Vec<f64> = charges.iter().map(|q| q * q).collect();
    let mean_charge_squared = sample_mean(&charge_squared)?;
    let connected_susceptibility_lattice_units =
        (mean_charge_squared - mean_charge * mean_charge) / lattice_four_volume as f64;

    let mut occupancy = BTreeMap::<i64, usize>::new();
    let mut ambiguous_sector_samples = 0usize;
    let mut adjacent_sector_changes = 0usize;
    let mut max_assigned_sector_dwell = 0usize;
    let mut previous_sector: Option<i64> = None;
    let mut current_dwell = 0usize;

    for charge in &charges {
        match assign_sector(*charge, sector_tolerance)? {
            Some(sector) => {
                *occupancy.entry(sector).or_default() += 1;
                match previous_sector {
                    Some(previous) if previous == sector => {
                        current_dwell += 1;
                    }
                    Some(_) => {
                        adjacent_sector_changes += 1;
                        current_dwell = 1;
                    }
                    None => {
                        current_dwell = 1;
                    }
                }
                max_assigned_sector_dwell = max_assigned_sector_dwell.max(current_dwell);
                previous_sector = Some(sector);
            }
            None => {
                ambiguous_sector_samples += 1;
                previous_sector = None;
                current_dwell = 0;
            }
        }
    }

    let assigned_sector_samples = samples.len() - ambiguous_sector_samples;
    let sector_occupancy = occupancy
        .into_iter()
        .map(|(sector, samples)| SectorOccupancy { sector, samples })
        .collect::<Vec<_>>();

    Ok(TopologyDiagnostics {
        definition,
        sample_count: samples.len(),
        mean_charge,
        mean_charge_squared,
        connected_susceptibility_lattice_units,
        assigned_sector_samples,
        ambiguous_sector_samples,
        distinct_assigned_sectors: sector_occupancy.len(),
        sector_occupancy,
        adjacent_sector_changes,
        max_assigned_sector_dwell,
        charge_autocorrelation: autocorrelation_diagnostic(&charges, max_lag)?,
        charge_squared_autocorrelation: autocorrelation_diagnostic(&charge_squared, max_lag)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> TopologyDefinitionLineage {
        TopologyDefinitionLineage {
            operator_id: "clover_q_v1".into(),
            smoothing_id: "wilson_flow_v1".into(),
            smoothing_time: 1.0,
            evidence_id: "LQCD-topology-operator".into(),
        }
    }

    fn samples(values: &[f64]) -> Vec<TopologySample> {
        values
            .iter()
            .enumerate()
            .map(|(index, charge)| TopologySample {
                cycle: (index + 1) as u64,
                charge: *charge,
            })
            .collect()
    }

    #[test]
    fn sector_changes_and_occupancy_are_descriptive() {
        let history = samples(&[
            0.02, 0.04, 0.98, 1.03, 0.01, -0.97, -1.02, 0.03,
        ]);
        let diagnostic = analyze_topology_history(&history, definition(), 0.10, 3, 256).unwrap();
        assert_eq!(diagnostic.ambiguous_sector_samples, 0);
        assert_eq!(diagnostic.distinct_assigned_sectors, 3);
        assert_eq!(diagnostic.adjacent_sector_changes, 4);
        assert_eq!(diagnostic.max_assigned_sector_dwell, 2);
        assert_eq!(diagnostic.assigned_sector_samples, 8);
    }

    #[test]
    fn ambiguity_breaks_sector_continuity_instead_of_inventing_tunnel() {
        let history = samples(&[0.02, 0.60, 1.02, 1.01]);
        let diagnostic = analyze_topology_history(&history, definition(), 0.20, 2, 64).unwrap();
        assert_eq!(diagnostic.ambiguous_sector_samples, 1);
        assert_eq!(diagnostic.adjacent_sector_changes, 0);
        assert_eq!(diagnostic.max_assigned_sector_dwell, 2);
    }

    #[test]
    fn constant_history_reports_zero_variance_without_fake_ess() {
        let history = samples(&[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]);
        let diagnostic = analyze_topology_history(&history, definition(), 0.10, 2, 64).unwrap();
        assert!(diagnostic.charge_autocorrelation.zero_variance_observed);
        assert_eq!(diagnostic.charge_autocorrelation.tau_int, None);
        assert_eq!(diagnostic.charge_autocorrelation.effective_sample_size, None);
    }

    #[test]
    fn correlated_topology_has_finite_separate_q_and_q2_diagnostics() {
        let history = samples(&[
            -1.0, -0.9, -0.8, -0.7, -0.5, -0.2, 0.1, 0.4,
            0.7, 0.9, 1.0, 0.8, 0.5, 0.2, -0.1, -0.4,
        ]);
        let diagnostic = analyze_topology_history(&history, definition(), 0.25, 6, 256).unwrap();
        assert!(diagnostic.charge_autocorrelation.tau_int.unwrap() >= 0.5);
        assert!(diagnostic.charge_squared_autocorrelation.tau_int.unwrap() >= 0.5);
    }

    #[test]
    fn connected_susceptibility_uses_declared_four_volume() {
        let history = samples(&[-1.0, 0.0, 1.0, 0.0]);
        let diagnostic = analyze_topology_history(&history, definition(), 0.10, 2, 8).unwrap();
        // E[Q] = 0, E[Q^2] = 1/2, so chi = 1/16 in lattice units.
        assert!((diagnostic.connected_susceptibility_lattice_units - 0.0625).abs() < 1.0e-14);
    }

    #[test]
    fn nonmonotonic_cycles_fail_closed() {
        let history = [
            TopologySample { cycle: 1, charge: 0.0 },
            TopologySample { cycle: 3, charge: 0.1 },
            TopologySample { cycle: 2, charge: 0.2 },
        ];
        assert!(matches!(
            analyze_topology_history(&history, definition(), 0.20, 1, 64),
            Err(TopologyDiagnosticsError::NonMonotonicCycle { .. })
        ));
    }
}
