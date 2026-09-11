// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical identity and applicability boundaries for legacy enterprise systems.
//!
//! Vendor identity is deliberately insufficient. IBM owns z/OS, IBM i, and AIX,
//! so platform-scoped knowledge must bind ecosystem + vendor + product + version.

use crate::legacy_computing::LegacyPlatformV1;
use crate::technology::{
    ApplicabilityAssessmentV1, ApplicabilityScopeV1, StringSelectorV1,
    TechnologyIdentityError, TechnologyIdentityV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const LEGACY_PLATFORM_IDENTITY_SCHEMA_V1: &str = "symthaea-it-legacy-platform-identity-v1";
pub const LEGACY_PLATFORM_ECOSYSTEM_V1: &str = "legacy-enterprise-os";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyPlatformIdentitySpecV1 {
    pub schema_version: String,
    pub platform: LegacyPlatformV1,
    pub ecosystem: String,
    pub vendor: String,
    pub product: String,
    pub version_family: String,
}

impl LegacyPlatformIdentitySpecV1 {
    pub fn validate(&self) -> Result<(), TechnologyIdentityError> {
        let identity = self.identity();
        identity.validate()?;
        self.scope().validate()?;
        Ok(())
    }

    pub fn identity(&self) -> TechnologyIdentityV1 {
        TechnologyIdentityV1 {
            ecosystem: Some(self.ecosystem.clone()),
            vendor: Some(self.vendor.clone()),
            product: self.product.clone(),
            edition: None,
            version: Some(self.version_family.clone()),
            build: None,
            architecture: None,
            platform: None,
            profile: None,
            observed_features: BTreeSet::new(),
        }
    }

    pub fn scope(&self) -> ApplicabilityScopeV1 {
        ApplicabilityScopeV1 {
            ecosystem: StringSelectorV1::Exact(self.ecosystem.clone()),
            vendor: StringSelectorV1::Exact(self.vendor.clone()),
            product: StringSelectorV1::Exact(self.product.clone()),
            version: StringSelectorV1::Prefix(self.version_family.clone()),
            ..ApplicabilityScopeV1::default()
        }
    }
}

pub fn legacy_platform_identity_spec_v1(platform: LegacyPlatformV1) -> LegacyPlatformIdentitySpecV1 {
    let (vendor, product, version_family) = match platform {
        LegacyPlatformV1::Zos => ("IBM", "z/OS", "3.2"),
        LegacyPlatformV1::IbmI => ("IBM", "IBM i", "7.6"),
        LegacyPlatformV1::Aix => ("IBM", "AIX", "7.3"),
        LegacyPlatformV1::Solaris => ("Oracle", "Oracle Solaris", "11.4"),
        LegacyPlatformV1::HpUx => ("HPE", "HP-UX", "11i v3"),
    };
    LegacyPlatformIdentitySpecV1 {
        schema_version: LEGACY_PLATFORM_IDENTITY_SCHEMA_V1.into(),
        platform,
        ecosystem: LEGACY_PLATFORM_ECOSYSTEM_V1.into(),
        vendor: vendor.into(),
        product: product.into(),
        version_family: version_family.into(),
    }
}

pub fn legacy_platform_identity_v1(platform: LegacyPlatformV1) -> TechnologyIdentityV1 {
    legacy_platform_identity_spec_v1(platform).identity()
}

pub fn legacy_platform_scope_v1(platform: LegacyPlatformV1) -> ApplicabilityScopeV1 {
    legacy_platform_identity_spec_v1(platform).scope()
}

pub fn assess_legacy_platform_identity_v1(
    expected: LegacyPlatformV1,
    observed: &TechnologyIdentityV1,
) -> Result<ApplicabilityAssessmentV1, TechnologyIdentityError> {
    legacy_platform_scope_v1(expected).assess(observed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_computing::seed_legacy_computing_pack_v1;
    use crate::technology::ApplicabilityStatusV1;

    #[test]
    fn all_specs_validate_and_share_one_canonical_ecosystem() {
        for platform in LegacyPlatformV1::ALL {
            let spec = legacy_platform_identity_spec_v1(platform);
            spec.validate().unwrap();
            assert_eq!(spec.schema_version, LEGACY_PLATFORM_IDENTITY_SCHEMA_V1);
            assert_eq!(spec.ecosystem, LEGACY_PLATFORM_ECOSYSTEM_V1);
        }
    }

    #[test]
    fn full_five_by_five_platform_matrix_is_strictly_isolated() {
        let mut applicable = 0usize;
        let mut inapplicable = 0usize;
        for expected in LegacyPlatformV1::ALL {
            for observed_platform in LegacyPlatformV1::ALL {
                let observed = legacy_platform_identity_v1(observed_platform);
                let assessment = assess_legacy_platform_identity_v1(expected, &observed).unwrap();
                if expected == observed_platform {
                    assert_eq!(assessment.status, ApplicabilityStatusV1::Applicable);
                    applicable += 1;
                } else {
                    assert_eq!(assessment.status, ApplicabilityStatusV1::Inapplicable);
                    inapplicable += 1;
                }
            }
        }
        assert_eq!(applicable, 5);
        assert_eq!(inapplicable, 20);
    }

    #[test]
    fn shared_ibm_vendor_never_collapses_product_identity() {
        let aix = legacy_platform_identity_v1(LegacyPlatformV1::Aix);
        let ibmi = legacy_platform_identity_v1(LegacyPlatformV1::IbmI);
        let zos = legacy_platform_identity_v1(LegacyPlatformV1::Zos);
        assert_eq!(aix.vendor.as_deref(), Some("IBM"));
        assert_eq!(ibmi.vendor.as_deref(), Some("IBM"));
        assert_eq!(zos.vendor.as_deref(), Some("IBM"));
        assert_eq!(
            assess_legacy_platform_identity_v1(LegacyPlatformV1::Aix, &ibmi)
                .unwrap()
                .status,
            ApplicabilityStatusV1::Inapplicable
        );
        assert_eq!(
            assess_legacy_platform_identity_v1(LegacyPlatformV1::Zos, &aix)
                .unwrap()
                .status,
            ApplicabilityStatusV1::Inapplicable
        );
    }

    #[test]
    fn missing_version_is_indeterminate_not_applicable() {
        let mut observed = legacy_platform_identity_v1(LegacyPlatformV1::Aix);
        observed.version = None;
        let assessment = assess_legacy_platform_identity_v1(LegacyPlatformV1::Aix, &observed).unwrap();
        assert_eq!(assessment.status, ApplicabilityStatusV1::Indeterminate);
        assert!(!assessment.unknowns.is_empty());
    }

    #[test]
    fn wrong_version_family_is_inapplicable() {
        let mut observed = legacy_platform_identity_v1(LegacyPlatformV1::Solaris);
        observed.version = Some("10".into());
        assert_eq!(
            assess_legacy_platform_identity_v1(LegacyPlatformV1::Solaris, &observed)
                .unwrap()
                .status,
            ApplicabilityStatusV1::Inapplicable
        );
    }

    #[test]
    fn seeded_profiles_match_canonical_version_families() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        for platform in LegacyPlatformV1::ALL {
            let profile = pack.profile(platform).unwrap();
            let spec = legacy_platform_identity_spec_v1(platform);
            assert_eq!(profile.version_family, spec.version_family);
        }
    }

    #[test]
    fn seeded_procedures_are_applicable_only_to_their_own_platform_identity() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        for procedure in &pack.procedures {
            for observed_platform in LegacyPlatformV1::ALL {
                let observed = legacy_platform_identity_v1(observed_platform);
                let assessment = procedure.applicability.assess(&observed).unwrap();
                let expected_status = if observed_platform == procedure.platform {
                    ApplicabilityStatusV1::Applicable
                } else {
                    ApplicabilityStatusV1::Inapplicable
                };
                assert_eq!(assessment.status, expected_status);
            }
        }
    }
}
