// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Common-cause failure analysis for nominally redundant flight lanes.
//!
//! Two lanes are not independent merely because they have different component
//! identifiers. Shared power, time, software images, data buses, cooling, or
//! environmental exposure can defeat both lanes at once. This module makes
//! those dependencies explicit and evaluates whether declared critical
//! functions retain both capacity and lane diversity after a common-cause
//! event.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RedundancyLane {
    LaneA,
    LaneB,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommonCauseDomain {
    ElectricalPower,
    TimeReference,
    SoftwareImage,
    DataBus,
    Cooling,
    EnvironmentalExposure,
    HumanConfiguration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RedundantAssetRole {
    NavigationSensor,
    FlightComputer,
    ActuatorLane,
    EngineControl,
    EvidenceRecorder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedundantAsset {
    pub asset_id: String,
    pub lane: RedundancyLane,
    pub role: RedundantAssetRole,
    /// Failure of any listed domain makes the asset unavailable.
    pub common_dependencies: Vec<CommonCauseDomain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalFunctionRequirement {
    pub function_id: String,
    pub role: RedundantAssetRole,
    pub minimum_surviving_assets: usize,
    pub minimum_independent_lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CommonCauseEvent {
    pub failed_domains: Vec<CommonCauseDomain>,
    pub failed_assets: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommonCauseFunctionStatus {
    Tolerant,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonCauseFunctionAssessment {
    pub function_id: String,
    pub role: RedundantAssetRole,
    pub status: CommonCauseFunctionStatus,
    pub surviving_assets: Vec<String>,
    pub surviving_lanes: Vec<RedundancyLane>,
    pub required_assets: usize,
    pub required_lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonCauseAssessment {
    pub unavailable_assets: Vec<String>,
    pub affected_domains: Vec<CommonCauseDomain>,
    pub function_assessments: Vec<CommonCauseFunctionAssessment>,
    pub all_critical_functions_tolerant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonCauseError {
    EmptyArchitecture,
    EmptyAssetId,
    DuplicateAsset,
    EmptyFunctionId,
    DuplicateFunction,
    InvalidRequirement,
    UnknownFailedAsset,
}

#[derive(Debug, Clone)]
pub struct CommonCauseAnalyzer {
    assets: Vec<RedundantAsset>,
    requirements: Vec<CriticalFunctionRequirement>,
}

impl CommonCauseAnalyzer {
    pub fn new(
        assets: Vec<RedundantAsset>,
        requirements: Vec<CriticalFunctionRequirement>,
    ) -> Result<Self, CommonCauseError> {
        if assets.is_empty() || requirements.is_empty() {
            return Err(CommonCauseError::EmptyArchitecture);
        }

        let mut asset_ids = BTreeSet::new();
        for asset in &assets {
            if asset.asset_id.trim().is_empty() {
                return Err(CommonCauseError::EmptyAssetId);
            }
            if !asset_ids.insert(asset.asset_id.clone()) {
                return Err(CommonCauseError::DuplicateAsset);
            }
        }

        let mut function_ids = BTreeSet::new();
        for requirement in &requirements {
            if requirement.function_id.trim().is_empty() {
                return Err(CommonCauseError::EmptyFunctionId);
            }
            if !function_ids.insert(requirement.function_id.clone()) {
                return Err(CommonCauseError::DuplicateFunction);
            }
            if requirement.minimum_surviving_assets == 0
                || requirement.minimum_independent_lanes == 0
                || requirement.minimum_independent_lanes > requirement.minimum_surviving_assets
            {
                return Err(CommonCauseError::InvalidRequirement);
            }
            let role_assets = assets
                .iter()
                .filter(|asset| asset.role == requirement.role)
                .count();
            if role_assets < requirement.minimum_surviving_assets {
                return Err(CommonCauseError::InvalidRequirement);
            }
        }

        Ok(Self {
            assets,
            requirements,
        })
    }

    pub fn assess(
        &self,
        event: &CommonCauseEvent,
    ) -> Result<CommonCauseAssessment, CommonCauseError> {
        let known_ids: BTreeSet<_> = self
            .assets
            .iter()
            .map(|asset| asset.asset_id.as_str())
            .collect();
        if event
            .failed_assets
            .iter()
            .any(|asset| !known_ids.contains(asset.as_str()))
        {
            return Err(CommonCauseError::UnknownFailedAsset);
        }

        let failed_domains: BTreeSet<_> = event.failed_domains.iter().copied().collect();
        let explicit_failures: BTreeSet<_> =
            event.failed_assets.iter().map(String::as_str).collect();
        let unavailable: BTreeSet<_> = self
            .assets
            .iter()
            .filter(|asset| {
                explicit_failures.contains(asset.asset_id.as_str())
                    || asset
                        .common_dependencies
                        .iter()
                        .any(|domain| failed_domains.contains(domain))
            })
            .map(|asset| asset.asset_id.clone())
            .collect();

        let assets_by_role: BTreeMap<_, Vec<_>> = self.assets.iter().fold(
            BTreeMap::<RedundantAssetRole, Vec<&RedundantAsset>>::new(),
            |mut map, asset| {
                map.entry(asset.role).or_default().push(asset);
                map
            },
        );

        let mut function_assessments = Vec::with_capacity(self.requirements.len());
        for requirement in &self.requirements {
            let mut surviving_assets: Vec<_> = assets_by_role
                .get(&requirement.role)
                .into_iter()
                .flatten()
                .filter(|asset| !unavailable.contains(&asset.asset_id))
                .map(|asset| asset.asset_id.clone())
                .collect();
            surviving_assets.sort();

            let mut surviving_lanes: Vec<_> = self
                .assets
                .iter()
                .filter(|asset| {
                    asset.role == requirement.role && surviving_assets.contains(&asset.asset_id)
                })
                .map(|asset| asset.lane)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            surviving_lanes.sort();

            let enough_assets = surviving_assets.len() >= requirement.minimum_surviving_assets;
            let enough_lanes = surviving_lanes.len() >= requirement.minimum_independent_lanes;
            let status = if enough_assets && enough_lanes {
                CommonCauseFunctionStatus::Tolerant
            } else if !surviving_assets.is_empty() {
                CommonCauseFunctionStatus::Degraded
            } else {
                CommonCauseFunctionStatus::Failed
            };
            function_assessments.push(CommonCauseFunctionAssessment {
                function_id: requirement.function_id.clone(),
                role: requirement.role,
                status,
                surviving_assets,
                surviving_lanes,
                required_assets: requirement.minimum_surviving_assets,
                required_lanes: requirement.minimum_independent_lanes,
            });
        }

        let all_critical_functions_tolerant = function_assessments
            .iter()
            .all(|assessment| assessment.status == CommonCauseFunctionStatus::Tolerant);

        Ok(CommonCauseAssessment {
            unavailable_assets: unavailable.into_iter().collect(),
            affected_domains: failed_domains.into_iter().collect(),
            function_assessments,
            all_critical_functions_tolerant,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> CommonCauseAnalyzer {
        CommonCauseAnalyzer::new(
            vec![
                RedundantAsset {
                    asset_id: "fc-a".into(),
                    lane: RedundancyLane::LaneA,
                    role: RedundantAssetRole::FlightComputer,
                    common_dependencies: vec![CommonCauseDomain::ElectricalPower],
                },
                RedundantAsset {
                    asset_id: "fc-b".into(),
                    lane: RedundancyLane::LaneB,
                    role: RedundantAssetRole::FlightComputer,
                    common_dependencies: vec![CommonCauseDomain::ElectricalPower],
                },
            ],
            vec![CriticalFunctionRequirement {
                function_id: "flight-control".into(),
                role: RedundantAssetRole::FlightComputer,
                minimum_surviving_assets: 2,
                minimum_independent_lanes: 2,
            }],
        )
        .unwrap()
    }

    #[test]
    fn shared_power_defeats_nominal_redundancy() {
        let assessment = analyzer()
            .assess(&CommonCauseEvent {
                failed_domains: vec![CommonCauseDomain::ElectricalPower],
                failed_assets: vec![],
            })
            .unwrap();
        assert!(!assessment.all_critical_functions_tolerant);
        assert_eq!(assessment.unavailable_assets.len(), 2);
        assert_eq!(
            assessment.function_assessments[0].status,
            CommonCauseFunctionStatus::Failed
        );
    }

    #[test]
    fn a_single_lane_failure_is_degraded_not_hidden() {
        let assessment = analyzer()
            .assess(&CommonCauseEvent {
                failed_domains: vec![],
                failed_assets: vec!["fc-a".into()],
            })
            .unwrap();
        assert_eq!(
            assessment.function_assessments[0].status,
            CommonCauseFunctionStatus::Degraded
        );
        assert_eq!(assessment.function_assessments[0].surviving_lanes.len(), 1);
    }

    #[test]
    fn unknown_failed_assets_are_rejected() {
        assert_eq!(
            analyzer().assess(&CommonCauseEvent {
                failed_domains: vec![],
                failed_assets: vec!["missing".into()],
            }),
            Err(CommonCauseError::UnknownFailedAsset)
        );
    }
}
