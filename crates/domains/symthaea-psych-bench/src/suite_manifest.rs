// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Declarative benchmark-suite manifests for psych-bench evidence surfaces.
//!
//! This module describes an *intended* benchmark population. It is deliberately
//! separate from [`crate::suite_identity`], which derives the *observed runnable*
//! population from actual [`crate::harness::PsychBenchmark`] objects.
//!
//! A valid manifest does not prove that the observed population equals the
//! declared population, that any benchmark executed, or that any result is
//! scientifically valid. Those are later evidence layers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const BENCHMARK_SUITE_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MANIFEST_DOMAIN: &[u8] = b"symthaea-psych-bench-suite-manifest-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteOrderingSemantics {
    OrderIndependent,
    Ordered,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteMemberStatus {
    Included,
    Excluded { reason: String },
}

/// Claim metadata is identity-bearing when supplied, but grants no authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimClassBinding {
    Unbound,
    Bound { schema_id: String, class_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteMember {
    pub benchmark_id: String,
    pub status: SuiteMemberStatus,
    pub claim_class: ClaimClassBinding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteManifest {
    pub schema_version: u32,
    pub suite_id: String,
    pub purpose: String,
    pub ordering: SuiteOrderingSemantics,
    pub members: Vec<BenchmarkSuiteMember>,
}

/// Human-facing alias bound to one exact manifest digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteAlias {
    pub alias_id: String,
    pub suite_digest: String,
}

/// Binds a declarative manifest to an intended code subject.
///
/// This is *not* an execution receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteCodeBinding {
    pub suite_digest: String,
    pub code_subject: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkSuiteManifestError {
    UnsupportedSchema { found: u32 },
    EmptyMembers,
    NonCanonicalComponent { field: &'static str, value: String },
    DuplicateBenchmarkId(String),
    MissingExecutable(String),
    AliasDigestMismatch,
    SuiteDigestMismatch,
}

impl fmt::Display for BenchmarkSuiteManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { found } => {
                write!(
                    f,
                    "unsupported benchmark-suite manifest schema version: {found}"
                )
            }
            Self::EmptyMembers => write!(f, "suite manifest must contain at least one member"),
            Self::NonCanonicalComponent { field, value } => {
                write!(f, "non-canonical {field}: {value:?}")
            }
            Self::DuplicateBenchmarkId(id) => write!(f, "duplicate benchmark id: {id}"),
            Self::MissingExecutable(id) => {
                write!(f, "included benchmark is not resolvable: {id}")
            }
            Self::AliasDigestMismatch => write!(f, "suite alias digest does not match manifest"),
            Self::SuiteDigestMismatch => {
                write!(f, "code binding suite digest does not match manifest")
            }
        }
    }
}

impl std::error::Error for BenchmarkSuiteManifestError {}

impl BenchmarkSuiteManifest {
    pub fn validate(&self) -> Result<(), BenchmarkSuiteManifestError> {
        if self.schema_version != BENCHMARK_SUITE_MANIFEST_SCHEMA_VERSION {
            return Err(BenchmarkSuiteManifestError::UnsupportedSchema {
                found: self.schema_version,
            });
        }
        validate_component("suite_id", &self.suite_id)?;
        validate_component("purpose", &self.purpose)?;
        if self.members.is_empty() {
            return Err(BenchmarkSuiteManifestError::EmptyMembers);
        }

        let mut seen = BTreeSet::new();
        for member in &self.members {
            validate_component("benchmark_id", &member.benchmark_id)?;
            if !seen.insert(member.benchmark_id.as_str()) {
                return Err(BenchmarkSuiteManifestError::DuplicateBenchmarkId(
                    member.benchmark_id.clone(),
                ));
            }

            if let SuiteMemberStatus::Excluded { reason } = &member.status {
                validate_component("exclusion_reason", reason)?;
            }

            if let ClaimClassBinding::Bound {
                schema_id,
                class_id,
            } = &member.claim_class
            {
                validate_component("claim_schema_id", schema_id)?;
                validate_component("claim_class_id", class_id)?;
            }
        }
        Ok(())
    }

    /// Canonical language-neutral bytes used by [`Self::digest`].
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BenchmarkSuiteManifestError> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MANIFEST_DOMAIN);
        push_u32(&mut bytes, self.schema_version);
        push_str(&mut bytes, &self.suite_id);
        push_str(&mut bytes, &self.purpose);
        bytes.push(match self.ordering {
            SuiteOrderingSemantics::OrderIndependent => 0,
            SuiteOrderingSemantics::Ordered => 1,
        });
        push_u64(&mut bytes, self.members.len() as u64);

        let mut members: Vec<&BenchmarkSuiteMember> = self.members.iter().collect();
        if self.ordering == SuiteOrderingSemantics::OrderIndependent {
            members.sort_by(|left, right| left.benchmark_id.cmp(&right.benchmark_id));
        }

        for (index, member) in members.into_iter().enumerate() {
            if self.ordering == SuiteOrderingSemantics::Ordered {
                push_u64(&mut bytes, index as u64);
            }
            push_str(&mut bytes, &member.benchmark_id);
            match &member.status {
                SuiteMemberStatus::Included => bytes.push(0),
                SuiteMemberStatus::Excluded { reason } => {
                    bytes.push(1);
                    push_str(&mut bytes, reason);
                }
            }
            match &member.claim_class {
                ClaimClassBinding::Unbound => bytes.push(0),
                ClaimClassBinding::Bound {
                    schema_id,
                    class_id,
                } => {
                    bytes.push(1);
                    push_str(&mut bytes, schema_id);
                    push_str(&mut bytes, class_id);
                }
            }
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<String, BenchmarkSuiteManifestError> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }

    /// Proves only that every *included manifest ID* occurs in the supplied
    /// resolver set. Extra supplied IDs are intentionally ignored here.
    /// Exact manifest ↔ observed-population equality belongs to the later
    /// compatibility theorem tracked by #3386.
    pub fn validate_included_ids_are_resolvable<'a, I>(
        &self,
        executable_ids: I,
    ) -> Result<(), BenchmarkSuiteManifestError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.validate()?;
        let executable: BTreeSet<&str> = executable_ids.into_iter().collect();
        for member in &self.members {
            if matches!(&member.status, SuiteMemberStatus::Included)
                && !executable.contains(member.benchmark_id.as_str())
            {
                return Err(BenchmarkSuiteManifestError::MissingExecutable(
                    member.benchmark_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn included_ids(&self) -> Result<Vec<&str>, BenchmarkSuiteManifestError> {
        self.validate()?;
        Ok(self
            .members
            .iter()
            .filter_map(|member| {
                matches!(&member.status, SuiteMemberStatus::Included)
                    .then_some(member.benchmark_id.as_str())
            })
            .collect())
    }
}

impl BenchmarkSuiteAlias {
    pub fn validate_against(
        &self,
        manifest: &BenchmarkSuiteManifest,
    ) -> Result<(), BenchmarkSuiteManifestError> {
        validate_component("alias_id", &self.alias_id)?;
        validate_digest(&self.suite_digest)?;
        if self.suite_digest != manifest.digest()? {
            return Err(BenchmarkSuiteManifestError::AliasDigestMismatch);
        }
        Ok(())
    }
}

impl BenchmarkSuiteCodeBinding {
    pub fn bind(
        manifest: &BenchmarkSuiteManifest,
        code_subject: impl Into<String>,
    ) -> Result<Self, BenchmarkSuiteManifestError> {
        let code_subject = code_subject.into();
        validate_component("code_subject", &code_subject)?;
        Ok(Self {
            suite_digest: manifest.digest()?,
            code_subject,
        })
    }

    pub fn validate_against(
        &self,
        manifest: &BenchmarkSuiteManifest,
    ) -> Result<(), BenchmarkSuiteManifestError> {
        validate_component("code_subject", &self.code_subject)?;
        validate_digest(&self.suite_digest)?;
        if self.suite_digest != manifest.digest()? {
            return Err(BenchmarkSuiteManifestError::SuiteDigestMismatch);
        }
        Ok(())
    }
}

fn validate_component(field: &'static str, value: &str) -> Result<(), BenchmarkSuiteManifestError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_digest(digest: &str) -> Result<(), BenchmarkSuiteManifestError> {
    let canonical = digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if canonical {
        Ok(())
    } else {
        Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
            field: "suite_digest",
            value: digest.to_string(),
        })
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn included(id: &str) -> BenchmarkSuiteMember {
        BenchmarkSuiteMember {
            benchmark_id: id.to_owned(),
            status: SuiteMemberStatus::Included,
            claim_class: ClaimClassBinding::Unbound,
        }
    }

    fn manifest(ordering: SuiteOrderingSemantics) -> BenchmarkSuiteManifest {
        BenchmarkSuiteManifest {
            schema_version: BENCHMARK_SUITE_MANIFEST_SCHEMA_VERSION,
            suite_id: "paper-battery-v1".into(),
            purpose: "paper-battery".into(),
            ordering,
            members: vec![included("Executive::Stroop"), included("WorM::NBack")],
        }
    }

    #[test]
    fn order_independent_manifest_is_declaration_order_invariant() {
        let first = manifest(SuiteOrderingSemantics::OrderIndependent);
        let mut second = first.clone();
        second.members.reverse();
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn ordered_manifest_binds_declaration_order() {
        let first = manifest(SuiteOrderingSemantics::Ordered);
        let mut second = first.clone();
        second.members.reverse();
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn membership_exclusion_and_claim_binding_are_identity_bearing() {
        let first = manifest(SuiteOrderingSemantics::OrderIndependent);

        let mut membership = first.clone();
        membership.members.push(included("Executive::Flanker"));
        assert_ne!(first.digest().unwrap(), membership.digest().unwrap());

        let mut exclusion = first.clone();
        exclusion.members[0].status = SuiteMemberStatus::Excluded {
            reason: "not-supported-by-profile".into(),
        };
        assert_ne!(first.digest().unwrap(), exclusion.digest().unwrap());

        let mut claim = first.clone();
        claim.members[0].claim_class = ClaimClassBinding::Bound {
            schema_id: "psych-bench-claim-contract-v1".into(),
            class_id: "behavioral_measurement".into(),
        };
        assert_ne!(first.digest().unwrap(), claim.digest().unwrap());
    }

    #[test]
    fn duplicate_and_empty_manifests_fail_closed() {
        let mut duplicate = manifest(SuiteOrderingSemantics::OrderIndependent);
        duplicate.members.push(included("Executive::Stroop"));
        assert!(matches!(
            duplicate.validate(),
            Err(BenchmarkSuiteManifestError::DuplicateBenchmarkId(id))
                if id == "Executive::Stroop"
        ));

        let mut empty = manifest(SuiteOrderingSemantics::OrderIndependent);
        empty.members.clear();
        assert_eq!(
            empty.validate(),
            Err(BenchmarkSuiteManifestError::EmptyMembers)
        );
    }

    #[test]
    fn every_identity_bearing_text_component_must_be_canonical() {
        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.suite_id = " paper-battery-v1".into();
        assert!(matches!(
            suite.validate(),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "suite_id",
                ..
            })
        ));

        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.purpose = "paper-battery\nlegacy".into();
        assert!(matches!(
            suite.validate(),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "purpose",
                ..
            })
        ));

        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.members[0].benchmark_id = "Executive::Stroop ".into();
        assert!(matches!(
            suite.validate(),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "benchmark_id",
                ..
            })
        ));

        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.members[0].status = SuiteMemberStatus::Excluded {
            reason: "legacy\tcondition".into(),
        };
        assert!(matches!(
            suite.validate(),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "exclusion_reason",
                ..
            })
        ));

        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.members[0].claim_class = ClaimClassBinding::Bound {
            schema_id: " claim-v1".into(),
            class_id: "behavioral_measurement".into(),
        };
        assert!(matches!(
            suite.validate(),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "claim_schema_id",
                ..
            })
        ));
    }

    #[test]
    fn resolvability_is_not_exact_population_compatibility() {
        let mut suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        suite.members[1].status = SuiteMemberStatus::Excluded {
            reason: "profile-exclusion".into(),
        };
        suite
            .validate_included_ids_are_resolvable([
                "Executive::Stroop",
                "Extra::RunnableButNotInManifest",
            ])
            .unwrap();
        assert!(matches!(
            suite.validate_included_ids_are_resolvable(["Extra::RunnableButNotInManifest"]),
            Err(BenchmarkSuiteManifestError::MissingExecutable(id))
                if id == "Executive::Stroop"
        ));
    }

    #[test]
    fn alias_and_code_binding_require_canonical_identity() {
        let suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        let alias = BenchmarkSuiteAlias {
            alias_id: "paper-current".into(),
            suite_digest: suite.digest().unwrap(),
        };
        alias.validate_against(&suite).unwrap();

        let mut bad_alias = alias.clone();
        bad_alias.alias_id = " paper-current".into();
        assert!(matches!(
            bad_alias.validate_against(&suite),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "alias_id",
                ..
            })
        ));

        let binding = BenchmarkSuiteCodeBinding::bind(&suite, "git:0123456789abcdef").unwrap();
        binding.validate_against(&suite).unwrap();
        assert_eq!(binding.suite_digest, suite.digest().unwrap());
        assert!(matches!(
            BenchmarkSuiteCodeBinding::bind(&suite, " git:0123456789abcdef"),
            Err(BenchmarkSuiteManifestError::NonCanonicalComponent {
                field: "code_subject",
                ..
            })
        ));
    }

    #[test]
    fn alias_cannot_retarget_changed_membership() {
        let first = manifest(SuiteOrderingSemantics::OrderIndependent);
        let alias = BenchmarkSuiteAlias {
            alias_id: "paper-current".into(),
            suite_digest: first.digest().unwrap(),
        };
        let mut changed = first.clone();
        changed.members.push(included("Executive::Flanker"));
        assert_eq!(
            alias.validate_against(&changed),
            Err(BenchmarkSuiteManifestError::AliasDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_manifest_identity() {
        let suite = manifest(SuiteOrderingSemantics::OrderIndependent);
        let bytes = serde_json::to_vec(&suite).unwrap();
        let decoded: BenchmarkSuiteManifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(suite, decoded);
        assert_eq!(suite.digest().unwrap(), decoded.digest().unwrap());
    }
}
