// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic uncertainty and boundary campaign design.
//!
//! A handful of nominal seeds does not exercise parameter uncertainty. This
//! module generates explicit nominal, per-axis boundary, global-corner, and
//! Latin-hypercube-like stratified cases from declared parameter ranges. It also
//! reports axis and pairwise-bin coverage without pretending that coverage alone
//! proves safety.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignAxis {
    pub name: String,
    pub unit: String,
    pub minimum: f64,
    pub nominal: f64,
    pub maximum: f64,
    pub coverage_bins: usize,
}

impl CampaignAxis {
    fn validate(&self) -> Result<(), CampaignDesignError> {
        if self.name.trim().is_empty() || self.unit.trim().is_empty() {
            return Err(CampaignDesignError::InvalidAxis);
        }
        if !self.minimum.is_finite()
            || !self.nominal.is_finite()
            || !self.maximum.is_finite()
            || self.maximum <= self.minimum
            || !(self.minimum..=self.maximum).contains(&self.nominal)
            || self.coverage_bins < 2
        {
            return Err(CampaignDesignError::InvalidAxis);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignCaseClass {
    Nominal,
    AxisLowerBoundary,
    AxisUpperBoundary,
    GlobalLowerCorner,
    GlobalUpperCorner,
    Stratified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignAxisValue {
    pub axis: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignCase {
    pub case_id: String,
    pub seed: u64,
    pub class: CampaignCaseClass,
    pub values: Vec<CampaignAxisValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CampaignDesignConfig {
    pub seed: u64,
    pub stratified_samples: usize,
    pub include_axis_boundaries: bool,
    pub include_global_corners: bool,
    pub minimum_pairwise_coverage_fraction: f64,
}

impl Default for CampaignDesignConfig {
    fn default() -> Self {
        Self {
            seed: 0x5a17_2026_cafe_babe,
            stratified_samples: 16,
            include_axis_boundaries: true,
            include_global_corners: true,
            minimum_pairwise_coverage_fraction: 0.25,
        }
    }
}

impl CampaignDesignConfig {
    fn validate(&self) -> Result<(), CampaignDesignError> {
        if self.stratified_samples == 0
            || !self.minimum_pairwise_coverage_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_pairwise_coverage_fraction)
        {
            return Err(CampaignDesignError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CampaignDesignError {
    EmptyCampaignIdentity,
    EmptyAxes,
    InvalidAxis,
    DuplicateAxis,
    InvalidConfiguration,
    InvalidCase,
    DuplicateCase,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignPlan {
    pub schema_version: String,
    pub campaign_id: String,
    pub base_scenario_id: String,
    pub base_scenario_digest: String,
    pub axes: Vec<CampaignAxis>,
    pub cases: Vec<CampaignCase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisCoverage {
    pub axis: String,
    pub minimum_observed: Option<f64>,
    pub maximum_observed: Option<f64>,
    pub nominal_seen: bool,
    pub lower_boundary_seen: bool,
    pub upper_boundary_seen: bool,
    pub occupied_bins: usize,
    pub total_bins: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseCoverage {
    pub first_axis: String,
    pub second_axis: String,
    pub occupied_cells: usize,
    pub total_cells: usize,
    pub coverage_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignCoverageReport {
    pub campaign_id: String,
    pub complete: bool,
    pub axis_coverage: Vec<AxisCoverage>,
    pub pairwise_coverage: Vec<PairwiseCoverage>,
    pub duplicate_case_ids: Vec<String>,
    pub invalid_case_ids: Vec<String>,
    pub missing_boundary_axes: Vec<String>,
    pub undercovered_pairs: Vec<(String, String)>,
}

impl CampaignPlan {
    pub fn generate(
        campaign_id: impl Into<String>,
        base_scenario_id: impl Into<String>,
        base_scenario_digest: impl Into<String>,
        axes: Vec<CampaignAxis>,
        config: CampaignDesignConfig,
    ) -> Result<Self, CampaignDesignError> {
        let campaign_id = campaign_id.into();
        let base_scenario_id = base_scenario_id.into();
        let base_scenario_digest = base_scenario_digest.into();
        if campaign_id.trim().is_empty()
            || base_scenario_id.trim().is_empty()
            || !valid_digest_identifier(&base_scenario_digest)
        {
            return Err(CampaignDesignError::EmptyCampaignIdentity);
        }
        if axes.is_empty() {
            return Err(CampaignDesignError::EmptyAxes);
        }
        config.validate()?;
        let mut names = BTreeSet::new();
        for axis in &axes {
            axis.validate()?;
            if !names.insert(axis.name.clone()) {
                return Err(CampaignDesignError::DuplicateAxis);
            }
        }

        let mut cases = Vec::new();
        cases.push(make_case(
            "nominal",
            mix_seed(config.seed, 0),
            CampaignCaseClass::Nominal,
            &axes,
            |axis, _| axis.nominal,
        ));
        if config.include_axis_boundaries {
            for (index, axis) in axes.iter().enumerate() {
                cases.push(make_case(
                    &format!("axis-{}-low", sanitize_id(&axis.name)),
                    mix_seed(config.seed, 1 + index as u64 * 2),
                    CampaignCaseClass::AxisLowerBoundary,
                    &axes,
                    |candidate, candidate_index| {
                        if candidate_index == index {
                            candidate.minimum
                        } else {
                            candidate.nominal
                        }
                    },
                ));
                cases.push(make_case(
                    &format!("axis-{}-high", sanitize_id(&axis.name)),
                    mix_seed(config.seed, 2 + index as u64 * 2),
                    CampaignCaseClass::AxisUpperBoundary,
                    &axes,
                    |candidate, candidate_index| {
                        if candidate_index == index {
                            candidate.maximum
                        } else {
                            candidate.nominal
                        }
                    },
                ));
            }
        }
        if config.include_global_corners {
            cases.push(make_case(
                "global-low",
                mix_seed(config.seed, 10_000),
                CampaignCaseClass::GlobalLowerCorner,
                &axes,
                |axis, _| axis.minimum,
            ));
            cases.push(make_case(
                "global-high",
                mix_seed(config.seed, 10_001),
                CampaignCaseClass::GlobalUpperCorner,
                &axes,
                |axis, _| axis.maximum,
            ));
        }

        let permutations: Vec<Vec<usize>> = axes
            .iter()
            .enumerate()
            .map(|(axis_index, _)| {
                deterministic_permutation(
                    config.stratified_samples,
                    mix_seed(config.seed, 20_000 + axis_index as u64),
                )
            })
            .collect();
        for sample_index in 0..config.stratified_samples {
            let values = axes
                .iter()
                .enumerate()
                .map(|(axis_index, axis)| {
                    let stratum = permutations[axis_index][sample_index];
                    let fraction = (stratum as f64 + 0.5) / config.stratified_samples as f64;
                    CampaignAxisValue {
                        axis: axis.name.clone(),
                        value: axis.minimum + fraction * (axis.maximum - axis.minimum),
                    }
                })
                .collect();
            cases.push(CampaignCase {
                case_id: format!("stratified-{sample_index:04}"),
                seed: mix_seed(config.seed, 30_000 + sample_index as u64),
                class: CampaignCaseClass::Stratified,
                values,
            });
        }

        let plan = Self {
            schema_version: "symthaea-helicopter-uncertainty-campaign-v1".into(),
            campaign_id,
            base_scenario_id,
            base_scenario_digest,
            axes,
            cases,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), CampaignDesignError> {
        if self.schema_version.trim().is_empty()
            || self.campaign_id.trim().is_empty()
            || self.base_scenario_id.trim().is_empty()
            || !valid_digest_identifier(&self.base_scenario_digest)
            || self.axes.is_empty()
        {
            return Err(CampaignDesignError::EmptyCampaignIdentity);
        }
        let mut axis_names = BTreeSet::new();
        for axis in &self.axes {
            axis.validate()?;
            if !axis_names.insert(axis.name.clone()) {
                return Err(CampaignDesignError::DuplicateAxis);
            }
        }
        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            if !case_ids.insert(case.case_id.clone()) {
                return Err(CampaignDesignError::DuplicateCase);
            }
            if !case_is_valid(case, &self.axes) {
                return Err(CampaignDesignError::InvalidCase);
            }
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, CampaignDesignError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| CampaignDesignError::SerializationFailed)
    }

    pub fn coverage(
        &self,
        minimum_pairwise_fraction: f64,
    ) -> Result<CampaignCoverageReport, CampaignDesignError> {
        if !minimum_pairwise_fraction.is_finite()
            || !(0.0..=1.0).contains(&minimum_pairwise_fraction)
        {
            return Err(CampaignDesignError::InvalidConfiguration);
        }
        let mut duplicate_case_ids = Vec::new();
        let mut seen_cases = BTreeSet::new();
        let mut invalid_case_ids = Vec::new();
        for case in &self.cases {
            if !seen_cases.insert(case.case_id.clone()) {
                duplicate_case_ids.push(case.case_id.clone());
            }
            if !case_is_valid(case, &self.axes) {
                invalid_case_ids.push(case.case_id.clone());
            }
        }

        let mut axis_coverage = Vec::new();
        let mut missing_boundary_axes = Vec::new();
        for (axis_index, axis) in self.axes.iter().enumerate() {
            let values: Vec<_> = self
                .cases
                .iter()
                .filter_map(|case| case.values.get(axis_index).map(|value| value.value))
                .collect();
            let minimum_observed = values.iter().copied().reduce(f64::min);
            let maximum_observed = values.iter().copied().reduce(f64::max);
            let nominal_seen = values
                .iter()
                .any(|value| nearly_equal(*value, axis.nominal));
            let lower_boundary_seen = values
                .iter()
                .any(|value| nearly_equal(*value, axis.minimum));
            let upper_boundary_seen = values
                .iter()
                .any(|value| nearly_equal(*value, axis.maximum));
            let mut occupied = BTreeSet::new();
            for value in values {
                occupied.insert(bin_index(axis, value));
            }
            if !lower_boundary_seen || !upper_boundary_seen {
                missing_boundary_axes.push(axis.name.clone());
            }
            axis_coverage.push(AxisCoverage {
                axis: axis.name.clone(),
                minimum_observed,
                maximum_observed,
                nominal_seen,
                lower_boundary_seen,
                upper_boundary_seen,
                occupied_bins: occupied.len(),
                total_bins: axis.coverage_bins,
            });
        }

        let mut pairwise_coverage = Vec::new();
        let mut undercovered_pairs = Vec::new();
        for first in 0..self.axes.len() {
            for second in first + 1..self.axes.len() {
                let first_axis = &self.axes[first];
                let second_axis = &self.axes[second];
                let mut cells = BTreeSet::new();
                for case in &self.cases {
                    if let (Some(first_value), Some(second_value)) =
                        (case.values.get(first), case.values.get(second))
                    {
                        cells.insert((
                            bin_index(first_axis, first_value.value),
                            bin_index(second_axis, second_value.value),
                        ));
                    }
                }
                let total_cells = first_axis.coverage_bins * second_axis.coverage_bins;
                let coverage_fraction = cells.len() as f64 / total_cells.max(1) as f64;
                if coverage_fraction < minimum_pairwise_fraction {
                    undercovered_pairs.push((first_axis.name.clone(), second_axis.name.clone()));
                }
                pairwise_coverage.push(PairwiseCoverage {
                    first_axis: first_axis.name.clone(),
                    second_axis: second_axis.name.clone(),
                    occupied_cells: cells.len(),
                    total_cells,
                    coverage_fraction,
                });
            }
        }

        let complete = duplicate_case_ids.is_empty()
            && invalid_case_ids.is_empty()
            && missing_boundary_axes.is_empty()
            && undercovered_pairs.is_empty()
            && axis_coverage.iter().all(|coverage| coverage.nominal_seen);
        Ok(CampaignCoverageReport {
            campaign_id: self.campaign_id.clone(),
            complete,
            axis_coverage,
            pairwise_coverage,
            duplicate_case_ids,
            invalid_case_ids,
            missing_boundary_axes,
            undercovered_pairs,
        })
    }
}

fn make_case<F>(
    id: &str,
    seed: u64,
    class: CampaignCaseClass,
    axes: &[CampaignAxis],
    mut value: F,
) -> CampaignCase
where
    F: FnMut(&CampaignAxis, usize) -> f64,
{
    CampaignCase {
        case_id: id.to_string(),
        seed,
        class,
        values: axes
            .iter()
            .enumerate()
            .map(|(index, axis)| CampaignAxisValue {
                axis: axis.name.clone(),
                value: value(axis, index),
            })
            .collect(),
    }
}

fn case_is_valid(case: &CampaignCase, axes: &[CampaignAxis]) -> bool {
    !case.case_id.trim().is_empty()
        && case.values.len() == axes.len()
        && case.values.iter().zip(axes).all(|(value, axis)| {
            value.axis == axis.name
                && value.value.is_finite()
                && (axis.minimum..=axis.maximum).contains(&value.value)
        })
}

fn deterministic_permutation(length: usize, seed: u64) -> Vec<usize> {
    let mut values: Vec<_> = (0..length).collect();
    let mut state = nonzero_seed(seed);
    for index in (1..length).rev() {
        state = xorshift64(state);
        let swap = (state as usize) % (index + 1);
        values.swap(index, swap);
    }
    values
}

fn mix_seed(seed: u64, stream: u64) -> u64 {
    nonzero_seed(seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0xa5a5_5a5a_d3c4_b2e1
    } else {
        seed
    }
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    nonzero_seed(state)
}

fn bin_index(axis: &CampaignAxis, value: f64) -> usize {
    if nearly_equal(value, axis.maximum) {
        return axis.coverage_bins - 1;
    }
    let fraction = ((value - axis.minimum) / (axis.maximum - axis.minimum)).clamp(0.0, 1.0);
    (fraction * axis.coverage_bins as f64).floor() as usize
}

fn nearly_equal(first: f64, second: f64) -> bool {
    (first - second).abs() <= 1.0e-10 * first.abs().max(second.abs()).max(1.0)
}

fn sanitize_id(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn valid_digest_identifier(value: &str) -> bool {
    let Some((algorithm, digest)) = value.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && digest.len() >= 8
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<CampaignAxis> {
        vec![
            CampaignAxis {
                name: "air-density".into(),
                unit: "kg/m3".into(),
                minimum: 0.9,
                nominal: 1.1,
                maximum: 1.3,
                coverage_bins: 4,
            },
            CampaignAxis {
                name: "payload-mass".into(),
                unit: "kg".into(),
                minimum: 0.0,
                nominal: 50.0,
                maximum: 100.0,
                coverage_bins: 4,
            },
            CampaignAxis {
                name: "crosswind".into(),
                unit: "m/s".into(),
                minimum: 0.0,
                nominal: 5.0,
                maximum: 20.0,
                coverage_bins: 4,
            },
        ]
    }

    #[test]
    fn generation_is_deterministic_and_in_range() {
        let first = CampaignPlan::generate(
            "campaign-a",
            "scenario-a",
            "sha256:11111111",
            axes(),
            CampaignDesignConfig::default(),
        )
        .unwrap();
        let second = CampaignPlan::generate(
            "campaign-a",
            "scenario-a",
            "sha256:11111111",
            axes(),
            CampaignDesignConfig::default(),
        )
        .unwrap();
        assert_eq!(first.cases, second.cases);
        first.validate().unwrap();
    }

    #[test]
    fn plan_contains_nominal_axis_boundaries_and_global_corners() {
        let plan = CampaignPlan::generate(
            "campaign-a",
            "scenario-a",
            "sha256:11111111",
            axes(),
            CampaignDesignConfig::default(),
        )
        .unwrap();
        assert!(
            plan.cases
                .iter()
                .any(|case| case.class == CampaignCaseClass::Nominal)
        );
        assert!(
            plan.cases
                .iter()
                .any(|case| case.class == CampaignCaseClass::GlobalLowerCorner)
        );
        assert!(
            plan.cases
                .iter()
                .any(|case| case.class == CampaignCaseClass::GlobalUpperCorner)
        );
        let coverage = plan.coverage(0.20).unwrap();
        assert!(coverage.missing_boundary_axes.is_empty());
        assert!(coverage.axis_coverage.iter().all(|axis| axis.nominal_seen));
    }

    #[test]
    fn coverage_fails_when_boundary_cases_are_removed() {
        let mut plan = CampaignPlan::generate(
            "campaign-a",
            "scenario-a",
            "sha256:11111111",
            axes(),
            CampaignDesignConfig {
                include_axis_boundaries: false,
                include_global_corners: false,
                ..CampaignDesignConfig::default()
            },
        )
        .unwrap();
        plan.cases.retain(|case| {
            case.class == CampaignCaseClass::Stratified || case.class == CampaignCaseClass::Nominal
        });
        let coverage = plan.coverage(0.0).unwrap();
        assert!(!coverage.complete);
        assert_eq!(coverage.missing_boundary_axes.len(), 3);
    }

    #[test]
    fn insufficient_stratified_samples_expose_pairwise_gap() {
        let plan = CampaignPlan::generate(
            "campaign-a",
            "scenario-a",
            "sha256:11111111",
            axes(),
            CampaignDesignConfig {
                stratified_samples: 2,
                include_axis_boundaries: false,
                include_global_corners: false,
                minimum_pairwise_coverage_fraction: 0.5,
                ..CampaignDesignConfig::default()
            },
        )
        .unwrap();
        let coverage = plan.coverage(0.5).unwrap();
        assert!(!coverage.undercovered_pairs.is_empty());
    }
}
