// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrated five-platform legacy IT portfolio.
//!
//! This layer composes the independently reviewable z/OS, IBM i, AIX, Solaris,
//! and HP-UX foundations without turning assembly into a competence claim. It
//! also proves that platform-specific claims remain inapplicable to every other
//! canonical legacy platform identity.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_aix::enrich_legacy_aix_foundation_v1;
use crate::legacy_aix_scenarios::register_aix_qualification_cases_v1;
use crate::legacy_computing::{seed_legacy_computing_pack_v1, LegacyComputingPackV1, LegacyPlatformV1};
use crate::legacy_hpux::enrich_legacy_hpux_foundation_v1;
use crate::legacy_hpux_scenarios::register_hpux_qualification_cases_v1;
use crate::legacy_ibmi::enrich_legacy_ibmi_foundation_v1;
use crate::legacy_ibmi_scenarios::register_ibmi_qualification_cases_v1;
use crate::legacy_platform_identity::legacy_platform_identity_v1;
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, exhaustive_legacy_qualification_profile_v1,
};
use crate::legacy_solaris::enrich_legacy_solaris_foundation_v1;
use crate::legacy_solaris_scenarios::register_solaris_qualification_cases_v1;
use crate::legacy_zos::enrich_legacy_zos_foundation_v1;
use crate::legacy_zos_scenarios::register_zos_qualification_cases_v1;
use crate::standards_registry::TechnicalClaimIdV1;
use crate::technology::ApplicabilityStatusV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_PORTFOLIO_SCHEMA_V1: &str = "symthaea-it-legacy-five-platform-portfolio-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyPortfolioSummaryV1 {
    pub schema_version: String,
    pub platform_count: usize,
    pub mechanism_count: usize,
    pub public_case_count: usize,
    pub platform_case_counts: BTreeMap<LegacyPlatformV1, usize>,
    pub distinct_platform_claims: usize,
    pub cross_platform_isolation_checks: usize,
    pub qualification_requirements: usize,
    pub ready_requirements: usize,
    pub source_qualification_ready: bool,
}

pub fn build_legacy_five_platform_portfolio_v1(
    fetched_at_unix_ms: u64,
) -> Result<(LegacyComputingPackV1, ItQualificationMatrixV1, LegacyPortfolioSummaryV1), LegacyPortfolioErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyPortfolioErrorV1::InvalidInput(
            "portfolio source timestamp must be non-zero".into(),
        ));
    }

    let mut pack = seed_legacy_computing_pack_v1(fetched_at_unix_ms)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;

    let zos = enrich_legacy_zos_foundation_v1(&mut pack, offset(fetched_at_unix_ms, 1)?)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;
    let ibmi = enrich_legacy_ibmi_foundation_v1(&mut pack, offset(fetched_at_unix_ms, 2)?)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;
    let aix = enrich_legacy_aix_foundation_v1(&mut pack, offset(fetched_at_unix_ms, 3)?)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;
    let solaris = enrich_legacy_solaris_foundation_v1(&mut pack, offset(fetched_at_unix_ms, 4)?)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;
    let hpux = enrich_legacy_hpux_foundation_v1(&mut pack, offset(fetched_at_unix_ms, 5)?)
        .map_err(|err| LegacyPortfolioErrorV1::Platform(err.to_string()))?;

    let mut claims = BTreeMap::<LegacyPlatformV1, BTreeSet<TechnicalClaimIdV1>>::new();
    collect_claims(&mut claims, LegacyPlatformV1::Zos, zos.mechanisms.iter().flat_map(|m| m.source_claims.iter()));
    collect_claims(&mut claims, LegacyPlatformV1::IbmI, ibmi.mechanisms.iter().flat_map(|m| m.source_claims.iter()));
    collect_claims(&mut claims, LegacyPlatformV1::Aix, aix.mechanisms.iter().flat_map(|m| m.source_claims.iter()));
    collect_claims(&mut claims, LegacyPlatformV1::Solaris, solaris.mechanisms.iter().flat_map(|m| m.source_claims.iter()));
    collect_claims(&mut claims, LegacyPlatformV1::HpUx, hpux.mechanisms.iter().flat_map(|m| m.source_claims.iter()));
    let cross_platform_isolation_checks = verify_cross_platform_isolation(&pack, &claims)?;

    let mut matrix = ItQualificationMatrixV1::new();
    let mut platform_case_counts = BTreeMap::new();
    platform_case_counts.insert(
        LegacyPlatformV1::Zos,
        register_zos_qualification_cases_v1(&pack, &mut matrix)
            .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?,
    );
    platform_case_counts.insert(
        LegacyPlatformV1::IbmI,
        register_ibmi_qualification_cases_v1(&pack, &mut matrix)
            .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?,
    );
    platform_case_counts.insert(
        LegacyPlatformV1::Aix,
        register_aix_qualification_cases_v1(&pack, &mut matrix)
            .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?,
    );
    platform_case_counts.insert(
        LegacyPlatformV1::Solaris,
        register_solaris_qualification_cases_v1(&pack, &mut matrix)
            .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?,
    );
    platform_case_counts.insert(
        LegacyPlatformV1::HpUx,
        register_hpux_qualification_cases_v1(&pack, &mut matrix)
            .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?,
    );

    let profile = exhaustive_legacy_qualification_profile_v1();
    let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix)
        .map_err(|err| LegacyPortfolioErrorV1::Qualification(err.to_string()))?;

    let summary = LegacyPortfolioSummaryV1 {
        schema_version: LEGACY_PORTFOLIO_SCHEMA_V1.into(),
        platform_count: LegacyPlatformV1::ALL.len(),
        mechanism_count: zos.mechanisms.len()
            + ibmi.mechanisms.len()
            + aix.mechanisms.len()
            + solaris.mechanisms.len()
            + hpux.mechanisms.len(),
        public_case_count: platform_case_counts.values().sum(),
        platform_case_counts,
        distinct_platform_claims: claims.values().map(BTreeSet::len).sum(),
        cross_platform_isolation_checks,
        qualification_requirements: assessment.requirements.len(),
        ready_requirements: assessment.ready_requirements,
        source_qualification_ready: assessment.source_qualification_ready,
    };

    Ok((pack, matrix, summary))
}

fn offset(base: u64, delta: u64) -> Result<u64, LegacyPortfolioErrorV1> {
    base.checked_add(delta)
        .ok_or_else(|| LegacyPortfolioErrorV1::InvalidInput("portfolio timestamp overflow".into()))
}

fn collect_claims<'a>(
    target: &mut BTreeMap<LegacyPlatformV1, BTreeSet<TechnicalClaimIdV1>>,
    platform: LegacyPlatformV1,
    claims: impl Iterator<Item = &'a TechnicalClaimIdV1>,
) {
    target
        .entry(platform)
        .or_default()
        .extend(claims.cloned());
}

fn verify_cross_platform_isolation(
    pack: &LegacyComputingPackV1,
    claims: &BTreeMap<LegacyPlatformV1, BTreeSet<TechnicalClaimIdV1>>,
) -> Result<usize, LegacyPortfolioErrorV1> {
    let mut checks = 0usize;
    for (owner, claim_ids) in claims {
        for claim_id in claim_ids {
            let claim = pack
                .sources
                .claim(claim_id)
                .ok_or_else(|| LegacyPortfolioErrorV1::UnknownClaim(claim_id.clone()))?;
            let applicability = claim.applicability.as_ref().ok_or_else(|| {
                LegacyPortfolioErrorV1::MissingApplicability(claim_id.clone())
            })?;
            for observed_platform in LegacyPlatformV1::ALL {
                if observed_platform == *owner {
                    continue;
                }
                let observed = legacy_platform_identity_v1(observed_platform);
                let assessment = applicability
                    .assess(&observed)
                    .map_err(|err| LegacyPortfolioErrorV1::Isolation(err.to_string()))?;
                checks += 1;
                if assessment.status != ApplicabilityStatusV1::Inapplicable {
                    return Err(LegacyPortfolioErrorV1::CrossPlatformLeak {
                        claim: claim_id.clone(),
                        owner: *owner,
                        observed: observed_platform,
                        status: assessment.status,
                    });
                }
            }
        }
    }
    Ok(checks)
}

#[derive(Debug)]
pub enum LegacyPortfolioErrorV1 {
    InvalidInput(String),
    Platform(String),
    Qualification(String),
    Isolation(String),
    UnknownClaim(TechnicalClaimIdV1),
    MissingApplicability(TechnicalClaimIdV1),
    CrossPlatformLeak {
        claim: TechnicalClaimIdV1,
        owner: LegacyPlatformV1,
        observed: LegacyPlatformV1,
        status: ApplicabilityStatusV1,
    },
}

impl fmt::Display for LegacyPortfolioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid legacy portfolio input: {message}"),
            Self::Platform(message) => write!(f, "legacy portfolio platform error: {message}"),
            Self::Qualification(message) => write!(f, "legacy portfolio qualification error: {message}"),
            Self::Isolation(message) => write!(f, "legacy portfolio isolation error: {message}"),
            Self::UnknownClaim(id) => write!(f, "legacy portfolio references unknown claim {}", id.0),
            Self::MissingApplicability(id) => write!(f, "legacy portfolio claim {} lacks applicability", id.0),
            Self::CrossPlatformLeak { claim, owner, observed, status } => write!(
                f,
                "legacy claim {} owned by {owner:?} leaked to {observed:?} with status {status:?}",
                claim.0
            ),
        }
    }
}

impl Error for LegacyPortfolioErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_composes_five_platforms_thirty_five_mechanisms_and_thirty_three_cases() {
        let (_pack, _matrix, summary) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        assert_eq!(summary.platform_count, 5);
        assert_eq!(summary.mechanism_count, 35);
        assert_eq!(summary.public_case_count, 33);
        assert_eq!(summary.platform_case_counts[&LegacyPlatformV1::Zos], 6);
        assert_eq!(summary.platform_case_counts[&LegacyPlatformV1::IbmI], 6);
        assert_eq!(summary.platform_case_counts[&LegacyPlatformV1::Aix], 7);
        assert_eq!(summary.platform_case_counts[&LegacyPlatformV1::Solaris], 7);
        assert_eq!(summary.platform_case_counts[&LegacyPlatformV1::HpUx], 7);
    }

    #[test]
    fn portfolio_claims_are_isolated_from_all_other_platform_identities() {
        let (_pack, _matrix, summary) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        assert!(summary.distinct_platform_claims >= 30);
        assert_eq!(
            summary.cross_platform_isolation_checks,
            summary.distinct_platform_claims * 4
        );
    }

    #[test]
    fn assembled_portfolio_still_does_not_claim_qualification() {
        let (_pack, _matrix, summary) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        assert_eq!(summary.qualification_requirements, 50);
        assert_eq!(summary.ready_requirements, 0);
        assert!(!summary.source_qualification_ready);
    }

    #[test]
    fn same_portfolio_build_is_deterministic_for_same_capture_time() {
        let (_, _, first) = build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let (_, _, second) = build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        assert_eq!(first, second);
    }
}
