// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compatibility bridge between declared and observed benchmark populations.
//!
//! [`crate::suite_manifest`] describes the *intended* suite while
//! [`crate::suite_identity`] derives the *observed runnable* population from
//! actual [`crate::harness::PsychBenchmark`] objects. This module proves one
//! narrow relation between those already-independent identities:
//!
//! ```text
//! order-independent manifest included IDs == observed executable IDs
//! ```
//!
//! Matching populations do not prove that any benchmark executed, completed,
//! produced a result, used valid metrics, or supports a scientific claim.
//! Ordered manifests are refused because the observed executable identity
//! deliberately canonicalizes membership lexicographically and therefore carries
//! no runtime sequence authority.

use crate::suite_identity::{BenchmarkSuiteIdentityError, ExecutableBenchmarkSuiteIdentity};
use crate::suite_manifest::{
    BenchmarkSuiteCodeBinding, BenchmarkSuiteManifest, BenchmarkSuiteManifestError,
    SuiteMemberStatus, SuiteOrderingSemantics,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const SUITE_COMPATIBILITY_SCHEMA_VERSION: u32 = 1;
const COMPATIBILITY_DOMAIN: &[u8] = b"symthaea-psych-bench-suite-compatibility-v1\0";

/// Content-addressed proof that one order-independent declared population equals
/// one observed runnable population for the declared code binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteCompatibilityReceipt {
    pub schema_version: u32,
    pub manifest_digest: String,
    pub observed_suite_digest: String,
    pub code_subject: String,
    pub included_member_count: usize,
    pub compatibility_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkSuiteCompatibilityError {
    Manifest(BenchmarkSuiteManifestError),
    Observed(BenchmarkSuiteIdentityError),
    UnsupportedSchema { found: u32 },
    OrderedManifestRequiresSequenceReceipt,
    MissingObservedMember(String),
    UnexpectedObservedMember(String),
    ExcludedMemberObserved(String),
    ManifestDigestMismatch,
    ObservedSuiteDigestMismatch,
    CodeSubjectMismatch,
    IncludedMemberCountMismatch { declared: usize, expected: usize },
    CompatibilityDigestMismatch,
}

impl From<BenchmarkSuiteManifestError> for BenchmarkSuiteCompatibilityError {
    fn from(value: BenchmarkSuiteManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<BenchmarkSuiteIdentityError> for BenchmarkSuiteCompatibilityError {
    fn from(value: BenchmarkSuiteIdentityError) -> Self {
        Self::Observed(value)
    }
}

impl fmt::Display for BenchmarkSuiteCompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "invalid suite manifest: {error}"),
            Self::Observed(error) => write!(f, "invalid observed executable suite: {error:?}"),
            Self::UnsupportedSchema { found } => {
                write!(f, "unsupported suite compatibility schema version: {found}")
            }
            Self::OrderedManifestRequiresSequenceReceipt => write!(
                f,
                "ordered manifests require a sequence-bearing execution receipt"
            ),
            Self::MissingObservedMember(id) => {
                write!(
                    f,
                    "manifest included member is absent from observed suite: {id}"
                )
            }
            Self::UnexpectedObservedMember(id) => {
                write!(f, "observed suite contains undeclared member: {id}")
            }
            Self::ExcludedMemberObserved(id) => {
                write!(
                    f,
                    "observed suite contains explicitly excluded member: {id}"
                )
            }
            Self::ManifestDigestMismatch => {
                write!(
                    f,
                    "compatibility receipt manifest digest does not match manifest"
                )
            }
            Self::ObservedSuiteDigestMismatch => write!(
                f,
                "compatibility receipt observed-suite digest does not match observed suite"
            ),
            Self::CodeSubjectMismatch => {
                write!(
                    f,
                    "compatibility receipt code subject does not match code binding"
                )
            }
            Self::IncludedMemberCountMismatch { declared, expected } => write!(
                f,
                "included-member count mismatch: declared {declared}, expected {expected}"
            ),
            Self::CompatibilityDigestMismatch => {
                write!(f, "compatibility receipt digest does not revalidate")
            }
        }
    }
}

impl std::error::Error for BenchmarkSuiteCompatibilityError {}

impl BenchmarkSuiteCompatibilityReceipt {
    /// Bind an already-valid order-independent manifest to an already-valid
    /// observed executable population and the manifest's code binding.
    pub fn bind(
        manifest: &BenchmarkSuiteManifest,
        observed: &ExecutableBenchmarkSuiteIdentity,
        code_binding: &BenchmarkSuiteCodeBinding,
    ) -> Result<Self, BenchmarkSuiteCompatibilityError> {
        manifest.validate()?;
        observed.validate()?;
        code_binding.validate_against(manifest)?;

        if manifest.ordering != SuiteOrderingSemantics::OrderIndependent {
            return Err(BenchmarkSuiteCompatibilityError::OrderedManifestRequiresSequenceReceipt);
        }

        validate_population_equality(manifest, observed)?;

        let manifest_digest = manifest.digest()?;
        let observed_suite_digest = observed.digest.clone();
        let code_subject = code_binding.code_subject.clone();
        let included_member_count = manifest
            .members
            .iter()
            .filter(|member| matches!(&member.status, SuiteMemberStatus::Included))
            .count();
        let compatibility_digest = compute_compatibility_digest(
            &manifest_digest,
            &observed_suite_digest,
            &code_subject,
            included_member_count,
        );

        Ok(Self {
            schema_version: SUITE_COMPATIBILITY_SCHEMA_VERSION,
            manifest_digest,
            observed_suite_digest,
            code_subject,
            included_member_count,
            compatibility_digest,
        })
    }

    /// Revalidate serialized compatibility against all three source identities.
    ///
    /// This proves record consistency only. It does not prove that a benchmark
    /// process executed or that this receipt was produced at an execution boundary.
    pub fn validate_against(
        &self,
        manifest: &BenchmarkSuiteManifest,
        observed: &ExecutableBenchmarkSuiteIdentity,
        code_binding: &BenchmarkSuiteCodeBinding,
    ) -> Result<(), BenchmarkSuiteCompatibilityError> {
        if self.schema_version != SUITE_COMPATIBILITY_SCHEMA_VERSION {
            return Err(BenchmarkSuiteCompatibilityError::UnsupportedSchema {
                found: self.schema_version,
            });
        }

        manifest.validate()?;
        observed.validate()?;
        code_binding.validate_against(manifest)?;
        if manifest.ordering != SuiteOrderingSemantics::OrderIndependent {
            return Err(BenchmarkSuiteCompatibilityError::OrderedManifestRequiresSequenceReceipt);
        }
        validate_population_equality(manifest, observed)?;

        let manifest_digest = manifest.digest()?;
        if self.manifest_digest != manifest_digest {
            return Err(BenchmarkSuiteCompatibilityError::ManifestDigestMismatch);
        }
        if self.observed_suite_digest != observed.digest {
            return Err(BenchmarkSuiteCompatibilityError::ObservedSuiteDigestMismatch);
        }
        if self.code_subject != code_binding.code_subject {
            return Err(BenchmarkSuiteCompatibilityError::CodeSubjectMismatch);
        }

        let expected_count = manifest
            .members
            .iter()
            .filter(|member| matches!(&member.status, SuiteMemberStatus::Included))
            .count();
        if self.included_member_count != expected_count {
            return Err(
                BenchmarkSuiteCompatibilityError::IncludedMemberCountMismatch {
                    declared: self.included_member_count,
                    expected: expected_count,
                },
            );
        }

        let expected_digest = compute_compatibility_digest(
            &manifest_digest,
            &observed.digest,
            &code_binding.code_subject,
            expected_count,
        );
        if self.compatibility_digest != expected_digest {
            return Err(BenchmarkSuiteCompatibilityError::CompatibilityDigestMismatch);
        }
        Ok(())
    }
}

fn validate_population_equality(
    manifest: &BenchmarkSuiteManifest,
    observed: &ExecutableBenchmarkSuiteIdentity,
) -> Result<(), BenchmarkSuiteCompatibilityError> {
    let mut included = BTreeSet::new();
    let mut excluded = BTreeSet::new();
    for member in &manifest.members {
        match &member.status {
            SuiteMemberStatus::Included => {
                included.insert(member.benchmark_id.as_str());
            }
            SuiteMemberStatus::Excluded { .. } => {
                excluded.insert(member.benchmark_id.as_str());
            }
        }
    }

    let observed_ids: BTreeSet<&str> = observed.benchmark_ids.iter().map(String::as_str).collect();

    if let Some(id) = observed_ids.intersection(&excluded).next() {
        return Err(BenchmarkSuiteCompatibilityError::ExcludedMemberObserved(
            (*id).to_string(),
        ));
    }
    if let Some(id) = included.difference(&observed_ids).next() {
        return Err(BenchmarkSuiteCompatibilityError::MissingObservedMember(
            (*id).to_string(),
        ));
    }
    if let Some(id) = observed_ids.difference(&included).next() {
        return Err(BenchmarkSuiteCompatibilityError::UnexpectedObservedMember(
            (*id).to_string(),
        ));
    }
    Ok(())
}

fn compute_compatibility_digest(
    manifest_digest: &str,
    observed_suite_digest: &str,
    code_subject: &str,
    included_member_count: usize,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COMPATIBILITY_DOMAIN);
    hasher.update(&SUITE_COMPATIBILITY_SCHEMA_VERSION.to_le_bytes());
    update_len_prefixed(&mut hasher, manifest_digest.as_bytes());
    update_len_prefixed(&mut hasher, observed_suite_digest.as_bytes());
    update_len_prefixed(&mut hasher, code_subject.as_bytes());
    hasher.update(&(included_member_count as u64).to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

fn update_len_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BenchmarkConfig, BenchmarkResult, PsychBenchmark};
    use crate::suite_identity::BenchmarkSuitePurpose;
    use crate::suite_manifest::{
        BENCHMARK_SUITE_MANIFEST_SCHEMA_VERSION, BenchmarkSuiteMember, ClaimClassBinding,
    };

    struct NamedBenchmark(&'static str);

    impl PsychBenchmark for NamedBenchmark {
        fn name(&self) -> &str {
            self.0
        }

        fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
            BenchmarkResult::new(self.0, None)
        }
    }

    fn included(id: &str) -> BenchmarkSuiteMember {
        BenchmarkSuiteMember {
            benchmark_id: id.into(),
            status: SuiteMemberStatus::Included,
            claim_class: ClaimClassBinding::Unbound,
        }
    }

    fn excluded(id: &str) -> BenchmarkSuiteMember {
        BenchmarkSuiteMember {
            benchmark_id: id.into(),
            status: SuiteMemberStatus::Excluded {
                reason: "profile-exclusion".into(),
            },
            claim_class: ClaimClassBinding::Unbound,
        }
    }

    fn manifest() -> BenchmarkSuiteManifest {
        BenchmarkSuiteManifest {
            schema_version: BENCHMARK_SUITE_MANIFEST_SCHEMA_VERSION,
            suite_id: "paper-battery-v1".into(),
            purpose: "paper-battery".into(),
            ordering: SuiteOrderingSemantics::OrderIndependent,
            members: vec![included("A"), included("B"), excluded("X")],
        }
    }

    fn observed(ids: &[&'static str]) -> ExecutableBenchmarkSuiteIdentity {
        let benchmarks: Vec<NamedBenchmark> = ids.iter().map(|id| NamedBenchmark(id)).collect();
        let refs: Vec<&dyn PsychBenchmark> = benchmarks
            .iter()
            .map(|benchmark| benchmark as &dyn PsychBenchmark)
            .collect();
        ExecutableBenchmarkSuiteIdentity::from_benchmarks(BenchmarkSuitePurpose::PaperBattery, refs)
            .unwrap()
    }

    fn binding(manifest: &BenchmarkSuiteManifest) -> BenchmarkSuiteCodeBinding {
        BenchmarkSuiteCodeBinding::bind(manifest, "git:0123456789abcdef").unwrap()
    }

    #[test]
    fn exact_order_independent_population_binds_all_three_identities() {
        let manifest = manifest();
        let observed = observed(&["B", "A"]);
        let binding = binding(&manifest);
        let receipt =
            BenchmarkSuiteCompatibilityReceipt::bind(&manifest, &observed, &binding).unwrap();

        assert_eq!(receipt.manifest_digest, manifest.digest().unwrap());
        assert_eq!(receipt.observed_suite_digest, observed.digest);
        assert_eq!(receipt.code_subject, binding.code_subject);
        assert_eq!(receipt.included_member_count, 2);
        receipt
            .validate_against(&manifest, &observed, &binding)
            .unwrap();
    }

    #[test]
    fn missing_extra_and_excluded_observed_members_fail_closed() {
        let manifest = manifest();
        let binding = binding(&manifest);

        assert!(matches!(
            BenchmarkSuiteCompatibilityReceipt::bind(
                &manifest,
                &observed(&["A"]),
                &binding,
            ),
            Err(BenchmarkSuiteCompatibilityError::MissingObservedMember(id)) if id == "B"
        ));

        assert!(matches!(
            BenchmarkSuiteCompatibilityReceipt::bind(
                &manifest,
                &observed(&["A", "B", "C"]),
                &binding,
            ),
            Err(BenchmarkSuiteCompatibilityError::UnexpectedObservedMember(id)) if id == "C"
        ));

        assert!(matches!(
            BenchmarkSuiteCompatibilityReceipt::bind(
                &manifest,
                &observed(&["A", "B", "X"]),
                &binding,
            ),
            Err(BenchmarkSuiteCompatibilityError::ExcludedMemberObserved(id)) if id == "X"
        ));
    }

    #[test]
    fn ordered_manifest_is_refused_without_sequence_authority() {
        let mut manifest = manifest();
        manifest.ordering = SuiteOrderingSemantics::Ordered;
        let observed = observed(&["A", "B"]);
        let binding = binding(&manifest);
        assert_eq!(
            BenchmarkSuiteCompatibilityReceipt::bind(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::OrderedManifestRequiresSequenceReceipt)
        );
    }

    #[test]
    fn serialized_receipt_revalidates_and_every_identity_is_bound() {
        let manifest = manifest();
        let observed = observed(&["A", "B"]);
        let binding = binding(&manifest);
        let original =
            BenchmarkSuiteCompatibilityReceipt::bind(&manifest, &observed, &binding).unwrap();
        let bytes = serde_json::to_vec(&original).unwrap();
        let decoded: BenchmarkSuiteCompatibilityReceipt = serde_json::from_slice(&bytes).unwrap();
        decoded
            .validate_against(&manifest, &observed, &binding)
            .unwrap();

        let mut wrong_manifest_digest = decoded.clone();
        wrong_manifest_digest.manifest_digest = "0".repeat(64);
        assert_eq!(
            wrong_manifest_digest.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::ManifestDigestMismatch)
        );

        let mut wrong_observed_digest = decoded.clone();
        wrong_observed_digest.observed_suite_digest = "0".repeat(64);
        assert_eq!(
            wrong_observed_digest.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::ObservedSuiteDigestMismatch)
        );

        let mut wrong_subject = decoded.clone();
        wrong_subject.code_subject = "git:fedcba9876543210".into();
        assert_eq!(
            wrong_subject.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::CodeSubjectMismatch)
        );

        let mut wrong_count = decoded.clone();
        wrong_count.included_member_count += 1;
        assert!(matches!(
            wrong_count.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::IncludedMemberCountMismatch { .. })
        ));

        let mut wrong_digest = decoded;
        wrong_digest.compatibility_digest = "0".repeat(64);
        assert_eq!(
            wrong_digest.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::CompatibilityDigestMismatch)
        );
    }

    #[test]
    fn manifest_identity_drift_invalidates_existing_code_binding() {
        let mut manifest = manifest();
        let observed = observed(&["A", "B"]);
        let binding = binding(&manifest);
        let receipt =
            BenchmarkSuiteCompatibilityReceipt::bind(&manifest, &observed, &binding).unwrap();

        manifest.members[0].claim_class = ClaimClassBinding::Bound {
            schema_id: "claim-v1".into(),
            class_id: "behavioral-measurement".into(),
        };
        assert!(matches!(
            receipt.validate_against(&manifest, &observed, &binding),
            Err(BenchmarkSuiteCompatibilityError::Manifest(
                BenchmarkSuiteManifestError::SuiteDigestMismatch
            ))
        ));
    }
}
