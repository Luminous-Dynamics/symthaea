// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Version- and environment-aware technology identity for IT reasoning.
//!
//! A claim about a technology is only useful when Symthaea can state the
//! implementation context to which that claim applies. These types intentionally
//! avoid pretending every vendor's version string has SemVer ordering.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TechnologyIdentityV1 {
    /// Ecosystem or standards family, e.g. `linux`, `windows`, `kubernetes`, `tls`.
    pub ecosystem: Option<String>,
    /// Vendor/project responsible for the implementation, if applicable.
    pub vendor: Option<String>,
    /// Canonical product/project name. This is the minimum required identity.
    pub product: String,
    pub edition: Option<String>,
    pub version: Option<String>,
    pub build: Option<String>,
    pub architecture: Option<String>,
    pub platform: Option<String>,
    /// Optional profile/flavor such as `fips`, `server-core`, or `managed`.
    pub profile: Option<String>,
    /// Explicitly observed capabilities/features. Absence from this set does not
    /// prove a feature is unsupported.
    #[serde(default)]
    pub observed_features: BTreeSet<String>,
}

impl TechnologyIdentityV1 {
    pub fn validate(&self) -> Result<(), TechnologyIdentityError> {
        validate_required(&self.product, "product")?;
        for (name, value) in [
            ("ecosystem", self.ecosystem.as_deref()),
            ("vendor", self.vendor.as_deref()),
            ("edition", self.edition.as_deref()),
            ("version", self.version.as_deref()),
            ("build", self.build.as_deref()),
            ("architecture", self.architecture.as_deref()),
            ("platform", self.platform.as_deref()),
            ("profile", self.profile.as_deref()),
        ] {
            if value.is_some_and(|v| v.trim().is_empty()) {
                return Err(TechnologyIdentityError::EmptyField(name));
            }
        }
        if self
            .observed_features
            .iter()
            .any(|feature| feature.trim().is_empty())
        {
            return Err(TechnologyIdentityError::EmptyField("observed_feature"));
        }
        Ok(())
    }

    /// Stable comparison key for exact identity within V1. It is deliberately
    /// not advertised as a vendor-global identifier.
    pub fn canonical_key(&self) -> Result<String, TechnologyIdentityError> {
        self.validate()?;
        Ok([
            normalize_opt(self.ecosystem.as_deref()),
            normalize_opt(self.vendor.as_deref()),
            normalize_required(&self.product),
            normalize_opt(self.edition.as_deref()),
            normalize_opt(self.version.as_deref()),
            normalize_opt(self.build.as_deref()),
            normalize_opt(self.architecture.as_deref()),
            normalize_opt(self.platform.as_deref()),
            normalize_opt(self.profile.as_deref()),
        ]
        .join("|"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringSelectorV1 {
    /// No constraint on this dimension.
    Any,
    Exact(String),
    /// Useful for vendor version families where lexical range ordering would be unsafe.
    Prefix(String),
    OneOf(BTreeSet<String>),
}

impl Default for StringSelectorV1 {
    fn default() -> Self {
        Self::Any
    }
}

impl StringSelectorV1 {
    fn evaluate(&self, observed: Option<&str>, dimension: &'static str) -> DimensionMatchV1 {
        match self {
            Self::Any => DimensionMatchV1::Match,
            Self::Exact(expected) => match observed {
                Some(actual) if eq_norm(actual, expected) => DimensionMatchV1::Match,
                Some(actual) => DimensionMatchV1::Mismatch(format!(
                    "{dimension}: expected {expected:?}, observed {actual:?}"
                )),
                None => DimensionMatchV1::Unknown(format!(
                    "{dimension}: exact {expected:?} required but observation is missing"
                )),
            },
            Self::Prefix(prefix) => match observed {
                Some(actual) if normalize_required(actual).starts_with(&normalize_required(prefix)) => {
                    DimensionMatchV1::Match
                }
                Some(actual) => DimensionMatchV1::Mismatch(format!(
                    "{dimension}: expected prefix {prefix:?}, observed {actual:?}"
                )),
                None => DimensionMatchV1::Unknown(format!(
                    "{dimension}: prefix {prefix:?} required but observation is missing"
                )),
            },
            Self::OneOf(expected) => match observed {
                Some(actual) if expected.iter().any(|candidate| eq_norm(actual, candidate)) => {
                    DimensionMatchV1::Match
                }
                Some(actual) => DimensionMatchV1::Mismatch(format!(
                    "{dimension}: observed {actual:?} is outside the permitted set"
                )),
                None => DimensionMatchV1::Unknown(format!(
                    "{dimension}: constrained value required but observation is missing"
                )),
            },
        }
    }

    fn validate(&self, dimension: &'static str) -> Result<(), TechnologyIdentityError> {
        match self {
            Self::Any => Ok(()),
            Self::Exact(value) | Self::Prefix(value) => validate_required(value, dimension),
            Self::OneOf(values) if values.is_empty() => {
                Err(TechnologyIdentityError::EmptySelector(dimension))
            }
            Self::OneOf(values) => {
                for value in values {
                    validate_required(value, dimension)?;
                }
                Ok(())
            }
        }
    }
}

/// Applicability scope attached to a technical claim, procedure, or test.
///
/// V1 supports equality/prefix/set selectors instead of generic version ranges;
/// a future adapter may add ecosystem-specific version ordering only when that
/// ordering is well-defined and qualified.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicabilityScopeV1 {
    pub ecosystem: StringSelectorV1,
    pub vendor: StringSelectorV1,
    pub product: StringSelectorV1,
    pub edition: StringSelectorV1,
    pub version: StringSelectorV1,
    pub build: StringSelectorV1,
    pub architecture: StringSelectorV1,
    pub platform: StringSelectorV1,
    pub profile: StringSelectorV1,
    /// Features that must have been positively observed. Missing feature evidence
    /// produces `Indeterminate`, not `Inapplicable`.
    #[serde(default)]
    pub required_features: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicabilityStatusV1 {
    Applicable,
    Inapplicable,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicabilityAssessmentV1 {
    pub status: ApplicabilityStatusV1,
    pub mismatches: Vec<String>,
    pub unknowns: Vec<String>,
}

impl ApplicabilityScopeV1 {
    pub fn validate(&self) -> Result<(), TechnologyIdentityError> {
        for (dimension, selector) in [
            ("ecosystem", &self.ecosystem),
            ("vendor", &self.vendor),
            ("product", &self.product),
            ("edition", &self.edition),
            ("version", &self.version),
            ("build", &self.build),
            ("architecture", &self.architecture),
            ("platform", &self.platform),
            ("profile", &self.profile),
        ] {
            selector.validate(dimension)?;
        }
        if self
            .required_features
            .iter()
            .any(|feature| feature.trim().is_empty())
        {
            return Err(TechnologyIdentityError::EmptyField("required_feature"));
        }
        Ok(())
    }

    pub fn assess(
        &self,
        identity: &TechnologyIdentityV1,
    ) -> Result<ApplicabilityAssessmentV1, TechnologyIdentityError> {
        self.validate()?;
        identity.validate()?;

        let mut mismatches = Vec::new();
        let mut unknowns = Vec::new();
        let dimensions = [
            self.ecosystem
                .evaluate(identity.ecosystem.as_deref(), "ecosystem"),
            self.vendor.evaluate(identity.vendor.as_deref(), "vendor"),
            self.product.evaluate(Some(&identity.product), "product"),
            self.edition.evaluate(identity.edition.as_deref(), "edition"),
            self.version.evaluate(identity.version.as_deref(), "version"),
            self.build.evaluate(identity.build.as_deref(), "build"),
            self.architecture
                .evaluate(identity.architecture.as_deref(), "architecture"),
            self.platform
                .evaluate(identity.platform.as_deref(), "platform"),
            self.profile.evaluate(identity.profile.as_deref(), "profile"),
        ];

        for result in dimensions {
            match result {
                DimensionMatchV1::Match => {}
                DimensionMatchV1::Mismatch(reason) => mismatches.push(reason),
                DimensionMatchV1::Unknown(reason) => unknowns.push(reason),
            }
        }

        for required in &self.required_features {
            if !identity
                .observed_features
                .iter()
                .any(|actual| eq_norm(actual, required))
            {
                unknowns.push(format!(
                    "feature {required:?} has not been positively observed"
                ));
            }
        }

        let status = if !mismatches.is_empty() {
            ApplicabilityStatusV1::Inapplicable
        } else if !unknowns.is_empty() {
            ApplicabilityStatusV1::Indeterminate
        } else {
            ApplicabilityStatusV1::Applicable
        };

        Ok(ApplicabilityAssessmentV1 {
            status,
            mismatches,
            unknowns,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DimensionMatchV1 {
    Match,
    Mismatch(String),
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TechnologyIdentityError {
    EmptyField(&'static str),
    EmptySelector(&'static str),
}

impl fmt::Display for TechnologyIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "technology identity field {field} is empty"),
            Self::EmptySelector(field) => write!(f, "selector {field} contains no candidates"),
        }
    }
}

impl Error for TechnologyIdentityError {}

fn validate_required(value: &str, field: &'static str) -> Result<(), TechnologyIdentityError> {
    if value.trim().is_empty() {
        Err(TechnologyIdentityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn normalize_required(value: &str) -> String {
    value.trim().to_lowercase()
}

fn normalize_opt(value: Option<&str>) -> String {
    value.map(normalize_required).unwrap_or_else(|| "?".to_string())
}

fn eq_norm(left: &str, right: &str) -> bool {
    normalize_required(left) == normalize_required(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_identity() -> TechnologyIdentityV1 {
        TechnologyIdentityV1 {
            ecosystem: Some("Linux".into()),
            vendor: Some("NixOS".into()),
            product: "NixOS".into(),
            edition: None,
            version: Some("26.05".into()),
            build: Some("system-generation-412".into()),
            architecture: Some("x86_64".into()),
            platform: Some("bare-metal".into()),
            profile: None,
            observed_features: BTreeSet::from(["systemd".into(), "nftables".into()]),
        }
    }

    #[test]
    fn canonical_key_preserves_unknown_dimensions() {
        let key = linux_identity().canonical_key().unwrap();
        assert_eq!(
            key,
            "linux|nixos|nixos|?|26.05|system-generation-412|x86_64|bare-metal|?"
        );
    }

    #[test]
    fn exact_mismatch_is_inapplicable() {
        let scope = ApplicabilityScopeV1 {
            product: StringSelectorV1::Exact("Windows Server".into()),
            ..Default::default()
        };
        let result = scope.assess(&linux_identity()).unwrap();
        assert_eq!(result.status, ApplicabilityStatusV1::Inapplicable);
        assert_eq!(result.mismatches.len(), 1);
    }

    #[test]
    fn missing_constrained_dimension_is_indeterminate() {
        let scope = ApplicabilityScopeV1 {
            profile: StringSelectorV1::Exact("fips".into()),
            ..Default::default()
        };
        let result = scope.assess(&linux_identity()).unwrap();
        assert_eq!(result.status, ApplicabilityStatusV1::Indeterminate);
        assert!(result.mismatches.is_empty());
        assert_eq!(result.unknowns.len(), 1);
    }

    #[test]
    fn missing_feature_is_unknown_not_false() {
        let scope = ApplicabilityScopeV1 {
            product: StringSelectorV1::Exact("NixOS".into()),
            required_features: BTreeSet::from(["selinux".into()]),
            ..Default::default()
        };
        let result = scope.assess(&linux_identity()).unwrap();
        assert_eq!(result.status, ApplicabilityStatusV1::Indeterminate);
    }

    #[test]
    fn matching_scope_is_applicable() {
        let scope = ApplicabilityScopeV1 {
            ecosystem: StringSelectorV1::Exact("linux".into()),
            product: StringSelectorV1::Exact("nixos".into()),
            version: StringSelectorV1::Prefix("26.".into()),
            required_features: BTreeSet::from(["NFTABLES".into()]),
            ..Default::default()
        };
        let result = scope.assess(&linux_identity()).unwrap();
        assert_eq!(result.status, ApplicabilityStatusV1::Applicable);
        assert!(result.unknowns.is_empty());
    }
}
