// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Five-platform legacy IT portfolio V2.
//!
//! V2 preserves the V1 portfolio lineage and layers in the measured AIX/IBM i
//! networking depth plus six public networking scenarios. It does not rewrite V1
//! and does not equate increased coverage with qualification.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_networking_depth::{
    enrich_legacy_aix_ibmi_networking_v1, LegacyNetworkingDepthErrorV1,
};
use crate::legacy_networking_scenarios::{
    register_legacy_networking_qualification_cases_v1,
    seed_legacy_networking_qualification_scenarios_v1, LegacyNetworkingScenarioErrorV1,
};
use crate::legacy_portfolio::{
    build_legacy_five_platform_portfolio_v1, LegacyPortfolioErrorV1, LegacyPortfolioSummaryV1,
};
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, exhaustive_legacy_qualification_profile_v1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const LEGACY_PORTFOLIO_SCHEMA_V2: &str = "symthaea-it-legacy-five-platform-portfolio-v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyPortfolioSummaryV2 {
    pub schema_version: String,
    pub base_v1: LegacyPortfolioSummaryV1,
    pub added_networking_mechanisms: usize,
    pub added_networking_cases: usize,
    pub total_mechanisms: usize,
    pub total_public_cases: usize,
    pub aix_networking_cases: usize,
    pub ibmi_networking_cases: usize,
    pub qualification_requirements: usize,
    pub ready_requirements: usize,
    pub source_qualification_ready: bool,
}

pub fn build_legacy_five_platform_portfolio_v2(
    fetched_at_unix_ms: u64,
) -> Result<(LegacyComputingPackV1, ItQualificationMatrixV1, LegacyPortfolioSummaryV2), LegacyPortfolioV2ErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyPortfolioV2ErrorV1::InvalidInput(
            "portfolio V2 source timestamp must be non-zero".into(),
        ));
    }

    let (mut pack, mut matrix, base_v1) =
        build_legacy_five_platform_portfolio_v1(fetched_at_unix_ms)?;
    let networking_timestamp = fetched_at_unix_ms
        .checked_add(6)
        .ok_or_else(|| LegacyPortfolioV2ErrorV1::InvalidInput("portfolio V2 timestamp overflow".into()))?;
    let networking = enrich_legacy_aix_ibmi_networking_v1(&mut pack, networking_timestamp)?;

    let scenarios = seed_legacy_networking_qualification_scenarios_v1(&pack)?;
    let aix_networking_cases = scenarios
        .iter()
        .filter(|scenario| scenario.platform == LegacyPlatformV1::Aix)
        .count();
    let ibmi_networking_cases = scenarios
        .iter()
        .filter(|scenario| scenario.platform == LegacyPlatformV1::IbmI)
        .count();
    let added_networking_cases =
        register_legacy_networking_qualification_cases_v1(&pack, &mut matrix)?;

    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix)
        .map_err(|err| LegacyPortfolioV2ErrorV1::Qualification(err.to_string()))?;

    for platform in [LegacyPlatformV1::Aix, LegacyPlatformV1::IbmI] {
        let requirement = assessment
            .requirements
            .iter()
            .find(|requirement| {
                requirement.platform == platform
                    && requirement.area == LegacyKnowledgeAreaV1::Networking
            })
            .ok_or(LegacyPortfolioV2ErrorV1::MissingNetworkingRequirement(platform))?;
        if requirement.matching_active_cases < 3 {
            return Err(LegacyPortfolioV2ErrorV1::NetworkingCoverageRegression {
                platform,
                observed_cases: requirement.matching_active_cases,
            });
        }
    }

    let total_mechanisms = base_v1
        .mechanism_count
        .checked_add(networking.mechanisms.len())
        .ok_or_else(|| LegacyPortfolioV2ErrorV1::InvalidInput("mechanism count overflow".into()))?;
    let total_public_cases = base_v1
        .public_case_count
        .checked_add(added_networking_cases)
        .ok_or_else(|| LegacyPortfolioV2ErrorV1::InvalidInput("case count overflow".into()))?;

    let summary = LegacyPortfolioSummaryV2 {
        schema_version: LEGACY_PORTFOLIO_SCHEMA_V2.into(),
        base_v1,
        added_networking_mechanisms: networking.mechanisms.len(),
        added_networking_cases,
        total_mechanisms,
        total_public_cases,
        aix_networking_cases,
        ibmi_networking_cases,
        qualification_requirements: assessment.requirements.len(),
        ready_requirements: assessment.ready_requirements,
        source_qualification_ready: assessment.source_qualification_ready,
    };

    Ok((pack, matrix, summary))
}

#[derive(Debug)]
pub enum LegacyPortfolioV2ErrorV1 {
    Base(LegacyPortfolioErrorV1),
    Networking(LegacyNetworkingDepthErrorV1),
    Scenarios(LegacyNetworkingScenarioErrorV1),
    Qualification(String),
    InvalidInput(String),
    MissingNetworkingRequirement(LegacyPlatformV1),
    NetworkingCoverageRegression {
        platform: LegacyPlatformV1,
        observed_cases: usize,
    },
}

impl fmt::Display for LegacyPortfolioV2ErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(err) => write!(f, "legacy portfolio V1 error: {err}"),
            Self::Networking(err) => write!(f, "legacy networking depth error: {err}"),
            Self::Scenarios(err) => write!(f, "legacy networking scenario error: {err}"),
            Self::Qualification(err) => write!(f, "legacy portfolio V2 qualification error: {err}"),
            Self::InvalidInput(message) => write!(f, "invalid legacy portfolio V2 input: {message}"),
            Self::MissingNetworkingRequirement(platform) => {
                write!(f, "legacy portfolio V2 missing networking requirement for {platform:?}")
            }
            Self::NetworkingCoverageRegression {
                platform,
                observed_cases,
            } => write!(
                f,
                "legacy portfolio V2 networking coverage regressed for {platform:?}: {observed_cases} active cases"
            ),
        }
    }
}

impl Error for LegacyPortfolioV2ErrorV1 {}

impl From<LegacyPortfolioErrorV1> for LegacyPortfolioV2ErrorV1 {
    fn from(value: LegacyPortfolioErrorV1) -> Self {
        Self::Base(value)
    }
}

impl From<LegacyNetworkingDepthErrorV1> for LegacyPortfolioV2ErrorV1 {
    fn from(value: LegacyNetworkingDepthErrorV1) -> Self {
        Self::Networking(value)
    }
}

impl From<LegacyNetworkingScenarioErrorV1> for LegacyPortfolioV2ErrorV1 {
    fn from(value: LegacyNetworkingScenarioErrorV1) -> Self {
        Self::Scenarios(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile::LegacyQualificationBlockerV1;

    #[test]
    fn v2_preserves_v1_and_adds_exact_networking_depth() {
        let (_, _, summary) =
            build_legacy_five_platform_portfolio_v2(1_800_000_000_000).unwrap();
        assert_eq!(summary.schema_version, LEGACY_PORTFOLIO_SCHEMA_V2);
        assert_eq!(summary.base_v1.mechanism_count, 35);
        assert_eq!(summary.base_v1.public_case_count, 33);
        assert_eq!(summary.added_networking_mechanisms, 4);
        assert_eq!(summary.added_networking_cases, 6);
        assert_eq!(summary.total_mechanisms, 39);
        assert_eq!(summary.total_public_cases, 39);
        assert_eq!(summary.aix_networking_cases, 3);
        assert_eq!(summary.ibmi_networking_cases, 3);
    }

    #[test]
    fn more_coverage_does_not_become_qualification() {
        let (pack, matrix, summary) =
            build_legacy_five_platform_portfolio_v2(1_800_000_000_000).unwrap();
        assert_eq!(summary.qualification_requirements, 50);
        assert_eq!(summary.ready_requirements, 0);
        assert!(!summary.source_qualification_ready);

        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        for platform in [LegacyPlatformV1::Aix, LegacyPlatformV1::IbmI] {
            let requirement = assessment
                .requirements
                .iter()
                .find(|requirement| {
                    requirement.platform == platform
                        && requirement.area == LegacyKnowledgeAreaV1::Networking
                })
                .unwrap();
            assert!(requirement.matching_active_cases >= 3);
            assert!(requirement
                .blockers
                .contains(&LegacyQualificationBlockerV1::SourceNotContentDigestBound));
            assert!(!requirement.ready_for_evaluation());
        }
    }

    #[test]
    fn v2_is_deterministic_for_same_capture_epoch() {
        let first = build_legacy_five_platform_portfolio_v2(1_800_000_000_000)
            .unwrap()
            .2;
        let second = build_legacy_five_platform_portfolio_v2(1_800_000_000_000)
            .unwrap()
            .2;
        assert_eq!(first, second);
    }
}
