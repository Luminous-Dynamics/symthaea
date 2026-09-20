// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical configuration identity for executor-identity provider policy.
//!
//! This crate is deliberately configuration-only. It does not verify providers,
//! create live executor state, perform I/O, or grant authority.
//!
//! ```text
//! ExecutorProviderPolicyManifestV1
//!     != provider verification
//!     != currentness proof
//!     != same-live-subject proof
//!     != live executor identity
//!     != execution authority
//! ```

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_executor_identity::{
    ExecutorIdentityDimension, ExecutorIdentityRequirement, ExecutorVerifierProfileId,
};
use symthaea_executor_subject_graph::{SubjectGraphRelationPolicyV1, SubjectRelationClass};
use symthaea_interaction_core::Digest32;
use thiserror::Error;

pub const EXECUTOR_PROVIDER_POLICY_SCHEMA_VERSION: u16 = 1;

const ENTRY_DOMAIN: &[u8] = b"symthaea.executor.provider-policy.entry.v1\0";
const ENTRY_SET_DOMAIN: &[u8] = b"symthaea.executor.provider-policy.entry-set.v1\0";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.executor.provider-policy.manifest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProviderPolicyTargetV1 {
    Identity(ExecutorIdentityDimension),
    Relation(SubjectRelationClass),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProviderPolicyError {
    #[error("{0} must not use an all-zero digest")]
    ZeroDigest(&'static str),
    #[error("evidence schema version must be non-zero")]
    ZeroEvidenceSchemaVersion,
    #[error("ExecutorProfile is local composition configuration, not a provider-evidence target")]
    UnsupportedExecutorProfileTarget,
    #[error("duplicate provider-policy entry {0:?}")]
    DuplicateEntry(ProviderPolicyEntryIdV1),
    #[error("target {target:?} repeats verifier profile {verifier_profile:?} with different policy semantics")]
    AmbiguousVerifierProfile {
        target: ProviderPolicyTargetV1,
        verifier_profile: ExecutorVerifierProfileId,
    },
    #[error("provider-policy identity target {0:?} is not required by this composition profile")]
    UnexpectedIdentityTarget(ExecutorIdentityDimension),
    #[error("provider-policy relation target {0:?} is not required by this composition profile")]
    UnexpectedRelationTarget(SubjectRelationClass),
    #[error("required provider-policy identity target {0:?} is missing")]
    MissingIdentityTarget(ExecutorIdentityDimension),
    #[error("required provider-policy relation target {0:?} is missing")]
    MissingRelationTarget(SubjectRelationClass),
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), ProviderPolicyError> {
    if value.as_bytes() == &[0; 32] {
        Err(ProviderPolicyError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderPolicyEntryIdV1(Digest32);

impl ProviderPolicyEntryIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutorProviderPolicyManifestIdV1(Digest32);

impl ExecutorProviderPolicyManifestIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// One ordinary provider-policy configuration entry.
///
/// Every commitment is configuration identity only. Construction does not
/// authenticate trust roots, appraisal policy, currentness, or provider code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderPolicyEntryV1 {
    target: ProviderPolicyTargetV1,
    verifier_profile: ExecutorVerifierProfileId,
    evidence_schema_commitment: Digest32,
    evidence_schema_version: u16,
    trust_anchor_commitment: Digest32,
    appraisal_policy_commitment: Digest32,
    currentness_profile_commitment: Digest32,
    minimum_epoch: u64,
    id: ProviderPolicyEntryIdV1,
}

impl ProviderPolicyEntryV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: ProviderPolicyTargetV1,
        verifier_profile: ExecutorVerifierProfileId,
        evidence_schema_commitment: Digest32,
        evidence_schema_version: u16,
        trust_anchor_commitment: Digest32,
        appraisal_policy_commitment: Digest32,
        currentness_profile_commitment: Digest32,
        minimum_epoch: u64,
    ) -> Result<Self, ProviderPolicyError> {
        if matches!(
            target,
            ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::ExecutorProfile)
        ) {
            return Err(ProviderPolicyError::UnsupportedExecutorProfileTarget);
        }
        if evidence_schema_version == 0 {
            return Err(ProviderPolicyError::ZeroEvidenceSchemaVersion);
        }
        reject_zero("evidence schema commitment", evidence_schema_commitment)?;
        reject_zero("trust anchor commitment", trust_anchor_commitment)?;
        reject_zero("appraisal policy commitment", appraisal_policy_commitment)?;
        reject_zero(
            "currentness profile commitment",
            currentness_profile_commitment,
        )?;

        let mut transcript = Transcript::new(ENTRY_DOMAIN);
        transcript.u16(EXECUTOR_PROVIDER_POLICY_SCHEMA_VERSION);
        transcript.target(target);
        transcript.digest(verifier_profile.digest());
        transcript.digest(evidence_schema_commitment);
        transcript.u16(evidence_schema_version);
        transcript.digest(trust_anchor_commitment);
        transcript.digest(appraisal_policy_commitment);
        transcript.digest(currentness_profile_commitment);
        transcript.u64(minimum_epoch);
        let id = ProviderPolicyEntryIdV1(transcript.finish());

        Ok(Self {
            target,
            verifier_profile,
            evidence_schema_commitment,
            evidence_schema_version,
            trust_anchor_commitment,
            appraisal_policy_commitment,
            currentness_profile_commitment,
            minimum_epoch,
            id,
        })
    }

    pub const fn target(&self) -> ProviderPolicyTargetV1 {
        self.target
    }

    pub const fn verifier_profile(&self) -> ExecutorVerifierProfileId {
        self.verifier_profile
    }

    pub const fn evidence_schema_commitment(&self) -> Digest32 {
        self.evidence_schema_commitment
    }

    pub const fn evidence_schema_version(&self) -> u16 {
        self.evidence_schema_version
    }

    pub const fn trust_anchor_commitment(&self) -> Digest32 {
        self.trust_anchor_commitment
    }

    pub const fn appraisal_policy_commitment(&self) -> Digest32 {
        self.appraisal_policy_commitment
    }

    pub const fn currentness_profile_commitment(&self) -> Digest32 {
        self.currentness_profile_commitment
    }

    pub const fn minimum_epoch(&self) -> u64 {
        self.minimum_epoch
    }

    pub const fn id(&self) -> ProviderPolicyEntryIdV1 {
        self.id
    }
}

/// Canonical exact provider-policy manifest for one executor identity
/// requirement and one D3A relation-assurance policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorProviderPolicyManifestV1 {
    requirement: Digest32,
    relation_policy: Digest32,
    entries: Vec<ProviderPolicyEntryV1>,
    entry_set: Digest32,
    id: ExecutorProviderPolicyManifestIdV1,
}

impl ExecutorProviderPolicyManifestV1 {
    pub fn new(
        requirement: &ExecutorIdentityRequirement,
        relation_policy: &SubjectGraphRelationPolicyV1,
        mut entries: Vec<ProviderPolicyEntryV1>,
    ) -> Result<Self, ProviderPolicyError> {
        let required_identity = required_identity_targets(requirement);
        let required_relations = required_relation_targets(requirement);

        let mut seen_ids = BTreeSet::new();
        let mut seen_profile_by_target = BTreeMap::new();
        let mut identity_coverage = BTreeSet::new();
        let mut relation_coverage = BTreeSet::new();

        for entry in &entries {
            if !seen_ids.insert(entry.id()) {
                return Err(ProviderPolicyError::DuplicateEntry(entry.id()));
            }

            let profile_key = (entry.target(), entry.verifier_profile().digest());
            if let Some(previous) = seen_profile_by_target.insert(profile_key, entry.id()) {
                if previous != entry.id() {
                    return Err(ProviderPolicyError::AmbiguousVerifierProfile {
                        target: entry.target(),
                        verifier_profile: entry.verifier_profile(),
                    });
                }
            }

            match entry.target() {
                ProviderPolicyTargetV1::Identity(dimension) => {
                    if !required_identity.contains(&dimension) {
                        return Err(ProviderPolicyError::UnexpectedIdentityTarget(dimension));
                    }
                    identity_coverage.insert(dimension);
                }
                ProviderPolicyTargetV1::Relation(class) => {
                    if !required_relations.contains(&class) {
                        return Err(ProviderPolicyError::UnexpectedRelationTarget(class));
                    }
                    relation_coverage.insert(class);
                }
            }
        }

        for dimension in required_identity {
            if !identity_coverage.contains(&dimension) {
                return Err(ProviderPolicyError::MissingIdentityTarget(dimension));
            }
        }
        for class in required_relations {
            if !relation_coverage.contains(&class) {
                return Err(ProviderPolicyError::MissingRelationTarget(class));
            }
        }

        entries.sort_by_key(ProviderPolicyEntryV1::id);

        let mut set_transcript = Transcript::new(ENTRY_SET_DOMAIN);
        set_transcript.u16(EXECUTOR_PROVIDER_POLICY_SCHEMA_VERSION);
        set_transcript.u32(entries.len() as u32);
        for entry in &entries {
            set_transcript.digest(entry.id().digest());
        }
        let entry_set = set_transcript.finish();

        let requirement_digest = requirement.digest();
        let relation_policy_digest = relation_policy.digest();
        let mut transcript = Transcript::new(MANIFEST_DOMAIN);
        transcript.u16(EXECUTOR_PROVIDER_POLICY_SCHEMA_VERSION);
        transcript.digest(requirement_digest);
        transcript.digest(relation_policy_digest);
        transcript.digest(entry_set);
        let id = ExecutorProviderPolicyManifestIdV1(transcript.finish());

        Ok(Self {
            requirement: requirement_digest,
            relation_policy: relation_policy_digest,
            entries,
            entry_set,
            id,
        })
    }

    pub const fn requirement_digest(&self) -> Digest32 {
        self.requirement
    }

    pub const fn relation_policy_digest(&self) -> Digest32 {
        self.relation_policy
    }

    pub fn entries(&self) -> &[ProviderPolicyEntryV1] {
        &self.entries
    }

    pub const fn entry_set_digest(&self) -> Digest32 {
        self.entry_set
    }

    pub const fn id(&self) -> ExecutorProviderPolicyManifestIdV1 {
        self.id
    }

    pub const fn digest(&self) -> Digest32 {
        self.id.0
    }
}

fn required_identity_targets(
    requirement: &ExecutorIdentityRequirement,
) -> BTreeSet<ExecutorIdentityDimension> {
    use ExecutorIdentityDimension::{
        Device, Embodiment, ExecutorProfile, Operator, SessionPeer, Software, Workload,
    };

    let dimensions = requirement.required_dimensions();
    let mut required = BTreeSet::new();
    for dimension in [SessionPeer, Operator, Workload, Software, Device, Embodiment] {
        if dimensions.contains(dimension) {
            required.insert(dimension);
        }
    }

    if dimensions.contains(SessionPeer)
        || dimensions.contains(Software)
        || dimensions.contains(Device)
        || dimensions.contains(Embodiment)
    {
        required.insert(Workload);
    }
    if dimensions.contains(Embodiment) {
        required.insert(Device);
    }

    debug_assert!(!required.contains(&ExecutorProfile));
    required
}

fn required_relation_targets(
    requirement: &ExecutorIdentityRequirement,
) -> BTreeSet<SubjectRelationClass> {
    use ExecutorIdentityDimension::{Device, Embodiment, SessionPeer, Software};
    use SubjectRelationClass::{
        DeviceEmbodiment, EndpointWorkload, WorkloadDevice, WorkloadSoftware,
    };

    let dimensions = requirement.required_dimensions();
    let mut required = BTreeSet::new();
    if dimensions.contains(SessionPeer) {
        required.insert(EndpointWorkload);
    }
    if dimensions.contains(Software) {
        required.insert(WorkloadSoftware);
    }
    if dimensions.contains(Device) || dimensions.contains(Embodiment) {
        required.insert(WorkloadDevice);
    }
    if dimensions.contains(Embodiment) {
        required.insert(DeviceEmbodiment);
    }
    required
}

const fn dimension_code(dimension: ExecutorIdentityDimension) -> u16 {
    match dimension {
        ExecutorIdentityDimension::SessionPeer => 0,
        ExecutorIdentityDimension::Operator => 1,
        ExecutorIdentityDimension::Workload => 2,
        ExecutorIdentityDimension::Software => 3,
        ExecutorIdentityDimension::Device => 4,
        ExecutorIdentityDimension::Embodiment => 5,
        ExecutorIdentityDimension::ExecutorProfile => 6,
    }
}

const fn relation_code(class: SubjectRelationClass) -> u16 {
    match class {
        SubjectRelationClass::EndpointWorkload => 0,
        SubjectRelationClass::WorkloadSoftware => 1,
        SubjectRelationClass::WorkloadDevice => 2,
        SubjectRelationClass::DeviceEmbodiment => 3,
    }
}

struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn target(&mut self, target: ProviderPolicyTargetV1) {
        match target {
            ProviderPolicyTargetV1::Identity(dimension) => {
                self.u16(0);
                self.u16(dimension_code(dimension));
            }
            ProviderPolicyTargetV1::Relation(class) => {
                self.u16(1);
                self.u16(relation_code(class));
            }
        }
    }

    fn finish(self) -> Digest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Digest32::new(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_executor_identity::{
        ExecutorIdentityDimensionSet, ExecutorIdentityProfile, ExecutorVerifierProfileId,
    };
    use symthaea_executor_subject_graph::SubjectRelationAssurance;

    const EXPECTED_ENTRIES: [&str; 5] = [
        "e21e9e30fcbbf65163634e9b8acb7e96f48242a6e1ce06ce75f5c6210adfd8c8",
        "aafac6a6a0d45f3d9306e25e591d432dfa08aea9a4b6660123067cc2256d2e00",
        "028d1d54654e8d885172717a48684ca0bc46dc307420f56544eda99d3ad76d8a",
        "90d233ee33e696df55296cdd1d96a1005d375fe8388fbdc9b32dd1847de00c7b",
        "afe1024b2d446430b303e8a0ddfef205a03c29bc7fc6a375e3111505f2d416bc",
    ];
    const EXPECTED_ENTRY_SET: &str =
        "d3d800d226ba364d28d5d186fa5eb946be5163cc6e129ffefe458ede6d1240ec";
    const EXPECTED_MANIFEST: &str =
        "09afb27f405b4c549bb37afa7fb6c010279bab7a9b71426ed68babd4703c86b4";

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn requirement() -> ExecutorIdentityRequirement {
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect("requirement")
    }

    fn relation_policy() -> SubjectGraphRelationPolicyV1 {
        SubjectGraphRelationPolicyV1::new(
            SubjectRelationAssurance::AuthenticatedProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::HardwareAnchoredProcess,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn entry(
        target: ProviderPolicyTargetV1,
        verifier: u8,
        schema: u8,
        trust: u8,
        appraisal: u8,
        currentness: u8,
        epoch: u64,
    ) -> ProviderPolicyEntryV1 {
        ProviderPolicyEntryV1::new(
            target,
            ExecutorVerifierProfileId::new(digest(verifier)).expect("verifier profile"),
            digest(schema),
            1,
            digest(trust),
            digest(appraisal),
            digest(currentness),
            epoch,
        )
        .expect("entry")
    }

    fn entries() -> Vec<ProviderPolicyEntryV1> {
        vec![
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::SessionPeer),
                0x11,
                0x31,
                0x41,
                0x51,
                0x61,
                7,
            ),
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Workload),
                0x12,
                0x32,
                0x42,
                0x52,
                0x62,
                8,
            ),
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Software),
                0x13,
                0x33,
                0x43,
                0x53,
                0x63,
                9,
            ),
            entry(
                ProviderPolicyTargetV1::Relation(SubjectRelationClass::EndpointWorkload),
                0x21,
                0x34,
                0x44,
                0x54,
                0x64,
                10,
            ),
            entry(
                ProviderPolicyTargetV1::Relation(SubjectRelationClass::WorkloadSoftware),
                0x22,
                0x35,
                0x45,
                0x55,
                0x65,
                11,
            ),
        ]
    }

    #[test]
    fn frozen_entry_set_and_manifest_vectors() {
        let entries = entries();
        for (entry, expected) in entries.iter().zip(EXPECTED_ENTRIES) {
            assert_eq!(entry.id().digest().to_hex(), expected);
        }
        let manifest = ExecutorProviderPolicyManifestV1::new(
            &requirement(),
            &relation_policy(),
            entries,
        )
        .expect("manifest");
        assert_eq!(manifest.entry_set_digest().to_hex(), EXPECTED_ENTRY_SET);
        assert_eq!(manifest.digest().to_hex(), EXPECTED_MANIFEST);
    }

    #[test]
    fn entry_order_is_non_semantic() {
        let forward = ExecutorProviderPolicyManifestV1::new(
            &requirement(),
            &relation_policy(),
            entries(),
        )
        .unwrap();
        let mut reversed_entries = entries();
        reversed_entries.reverse();
        let reversed = ExecutorProviderPolicyManifestV1::new(
            &requirement(),
            &relation_policy(),
            reversed_entries,
        )
        .unwrap();
        assert_eq!(forward.entry_set_digest(), reversed.entry_set_digest());
        assert_eq!(forward.digest(), reversed.digest());
    }

    #[test]
    fn duplicate_and_ambiguous_entries_fail_closed() {
        let mut duplicate = entries();
        duplicate.push(duplicate[0].clone());
        assert!(matches!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                duplicate
            ),
            Err(ProviderPolicyError::DuplicateEntry(_))
        ));

        let mut ambiguous = entries();
        ambiguous.push(entry(
            ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::SessionPeer),
            0x11,
            0x31,
            0x99,
            0x51,
            0x61,
            7,
        ));
        assert!(matches!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                ambiguous
            ),
            Err(ProviderPolicyError::AmbiguousVerifierProfile { .. })
        ));
    }

    #[test]
    fn missing_identity_and_relation_coverage_fail_closed() {
        let missing_software: Vec<_> = entries()
            .into_iter()
            .filter(|entry| {
                entry.target()
                    != ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Software)
            })
            .collect();
        assert_eq!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                missing_software
            ),
            Err(ProviderPolicyError::MissingIdentityTarget(
                ExecutorIdentityDimension::Software
            ))
        );

        let missing_relation: Vec<_> = entries()
            .into_iter()
            .filter(|entry| {
                entry.target()
                    != ProviderPolicyTargetV1::Relation(SubjectRelationClass::WorkloadSoftware)
            })
            .collect();
        assert_eq!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                missing_relation
            ),
            Err(ProviderPolicyError::MissingRelationTarget(
                SubjectRelationClass::WorkloadSoftware
            ))
        );
    }

    #[test]
    fn unexpected_targets_fail_closed() {
        let mut unexpected_identity = entries();
        unexpected_identity.push(entry(
            ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Device),
            0x14,
            0x36,
            0x46,
            0x56,
            0x66,
            12,
        ));
        assert_eq!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                unexpected_identity
            ),
            Err(ProviderPolicyError::UnexpectedIdentityTarget(
                ExecutorIdentityDimension::Device
            ))
        );

        let mut unexpected_relation = entries();
        unexpected_relation.push(entry(
            ProviderPolicyTargetV1::Relation(SubjectRelationClass::WorkloadDevice),
            0x23,
            0x37,
            0x47,
            0x57,
            0x67,
            13,
        ));
        assert_eq!(
            ExecutorProviderPolicyManifestV1::new(
                &requirement(),
                &relation_policy(),
                unexpected_relation
            ),
            Err(ProviderPolicyError::UnexpectedRelationTarget(
                SubjectRelationClass::WorkloadDevice
            ))
        );
    }

    #[test]
    fn embodiment_support_closure_requires_device_policy() {
        let requirement = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::Embodiment,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect("requirement");

        let sparse = vec![
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Workload),
                0x12,
                0x32,
                0x42,
                0x52,
                0x62,
                8,
            ),
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Software),
                0x13,
                0x33,
                0x43,
                0x53,
                0x63,
                9,
            ),
            entry(
                ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Embodiment),
                0x15,
                0x38,
                0x48,
                0x58,
                0x68,
                14,
            ),
            entry(
                ProviderPolicyTargetV1::Relation(SubjectRelationClass::WorkloadSoftware),
                0x22,
                0x35,
                0x45,
                0x55,
                0x65,
                11,
            ),
        ];

        assert_eq!(
            ExecutorProviderPolicyManifestV1::new(&requirement, &relation_policy(), sparse),
            Err(ProviderPolicyError::MissingIdentityTarget(
                ExecutorIdentityDimension::Device
            ))
        );
    }

    #[test]
    fn executor_profile_and_zero_commitments_reject() {
        let unsupported = ProviderPolicyEntryV1::new(
            ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::ExecutorProfile),
            ExecutorVerifierProfileId::new(digest(0x11)).unwrap(),
            digest(0x31),
            1,
            digest(0x41),
            digest(0x51),
            digest(0x61),
            0,
        );
        assert_eq!(
            unsupported,
            Err(ProviderPolicyError::UnsupportedExecutorProfileTarget)
        );

        let zero = ProviderPolicyEntryV1::new(
            ProviderPolicyTargetV1::Identity(ExecutorIdentityDimension::Workload),
            ExecutorVerifierProfileId::new(digest(0x12)).unwrap(),
            Digest32::new([0; 32]),
            1,
            digest(0x42),
            digest(0x52),
            digest(0x62),
            0,
        );
        assert_eq!(
            zero,
            Err(ProviderPolicyError::ZeroDigest(
                "evidence schema commitment"
            ))
        );
    }
}
