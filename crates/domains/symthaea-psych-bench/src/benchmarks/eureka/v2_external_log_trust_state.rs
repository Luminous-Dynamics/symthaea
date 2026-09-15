// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure candidate mechanics for EUREKA-002 V2 transparency-log trust state.
//!
//! This module deliberately stops before trust verification or admission. It
//! models four different objects that later phases must keep distinct:
//!
//! 1. a candidate TUF metadata/target state;
//! 2. a stable transparency-log identity;
//! 3. a candidate binding saying that one TUF state names one log identity; and
//! 4. an append-only EUREKA admission candidate for that stable log identity.
//!
//! The durable checkpoint binds all four identities and provides only a
//! multi-dimensional anti-rollback *classification*. A numerically newer
//! candidate is never accepted here. A later qualified TUF verifier and a later
//! EUREKA admission transition must authenticate/authorize the corresponding
//! state changes before any production trust token can exist.
//!
//! None of these types means "TUF verified", "EUREKA admitted", "externally
//! included", "consistency witnessed", or "execution authorized".

use super::v2_qualifier_admission_manifest::V2_REPOSITORY_IDENTITY;

pub(super) const V2_TUF_TRUST_STATE_DESCRIPTOR_SCHEMA: &str =
    "EUREKA.002.V2.TUF_TRUST_STATE_DESCRIPTOR.v1";
pub(super) const V2_TUF_TRUST_STATE_DESCRIPTOR_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.TUF_TRUST_STATE_DESCRIPTOR_COMMITMENT.v1";
pub(super) const V2_TRANSPARENCY_LOG_IDENTITY_DESCRIPTOR_SCHEMA: &str =
    "EUREKA.002.V2.TRANSPARENCY_LOG_IDENTITY_DESCRIPTOR.v1";
pub(super) const V2_TRANSPARENCY_LOG_IDENTITY_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.TRANSPARENCY_LOG_IDENTITY_COMMITMENT.v1";
pub(super) const V2_TUF_LOG_BINDING_CANDIDATE_SCHEMA: &str =
    "EUREKA.002.V2.TUF_LOG_BINDING_CANDIDATE.v1";
pub(super) const V2_TUF_LOG_BINDING_CANDIDATE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.TUF_LOG_BINDING_CANDIDATE_COMMITMENT.v1";
pub(super) const V2_EXTERNAL_LOG_ADMISSION_CANDIDATE_SCHEMA: &str =
    "EUREKA.002.V2.EXTERNAL_LOG_ADMISSION_CANDIDATE.v1";
pub(super) const V2_EXTERNAL_LOG_ADMISSION_CANDIDATE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.EXTERNAL_LOG_ADMISSION_CANDIDATE_COMMITMENT.v1";
pub(super) const V2_EXTERNAL_LOG_TRUST_CHECKPOINT_SCHEMA: &str =
    "EUREKA.002.V2.EXTERNAL_LOG_TRUST_CHECKPOINT.v1";
pub(super) const V2_EXTERNAL_LOG_TRUST_CHECKPOINT_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.EXTERNAL_LOG_TRUST_CHECKPOINT_COMMITMENT.v1";
pub(super) const V2_EXTERNALITY_POLICY_REVISION: &str =
    "EUREKA.002.V2.EXTERNALITY_POLICY.transparency-log-inclusion.v1";
pub(super) const V2_TRANSPARENCY_PROVIDER: &str = "sigstore-rekor-v2";
pub(super) const V2_TRANSPARENCY_ROLE: &str = "transparency-log-inclusion";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2VersionedMetadataIdentity {
    version: u64,
    sha256: [u8; 32],
}

impl V2VersionedMetadataIdentity {
    pub(super) fn candidate(version: u64, sha256: [u8; 32]) -> Result<Self, V2ExternalLogTrustError> {
        if version == 0 {
            return Err(V2ExternalLogTrustError::InvalidTufState);
        }
        Ok(Self { version, sha256 })
    }

    pub(super) const fn version(self) -> u64 {
        self.version
    }

    pub(super) const fn sha256(self) -> [u8; 32] {
        self.sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2TufTrustStateCandidateInput {
    pub(super) bootstrap_root_version: u64,
    pub(super) bootstrap_root_sha256: [u8; 32],
    pub(super) current_root: V2VersionedMetadataIdentity,
    pub(super) timestamp: V2VersionedMetadataIdentity,
    pub(super) snapshot: V2VersionedMetadataIdentity,
    pub(super) targets: V2VersionedMetadataIdentity,
    pub(super) trusted_root_target_name: String,
    pub(super) trusted_root_target_sha256: [u8; 32],
    pub(super) signing_config_target_name: String,
    pub(super) signing_config_target_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2TufTrustStateDescriptor {
    bootstrap_root_version: u64,
    bootstrap_root_sha256: [u8; 32],
    current_root: V2VersionedMetadataIdentity,
    timestamp: V2VersionedMetadataIdentity,
    snapshot: V2VersionedMetadataIdentity,
    targets: V2VersionedMetadataIdentity,
    trusted_root_target_name: String,
    trusted_root_target_sha256: [u8; 32],
    signing_config_target_name: String,
    signing_config_target_sha256: [u8; 32],
    commitment: [u8; 32],
}

impl V2TufTrustStateDescriptor {
    pub(super) fn candidate(
        input: V2TufTrustStateCandidateInput,
    ) -> Result<Self, V2ExternalLogTrustError> {
        if input.bootstrap_root_version == 0
            || input.current_root.version() < input.bootstrap_root_version
        {
            return Err(V2ExternalLogTrustError::InvalidTufState);
        }
        validate_atom(&input.trusted_root_target_name)?;
        validate_atom(&input.signing_config_target_name)?;
        let mut descriptor = Self {
            bootstrap_root_version: input.bootstrap_root_version,
            bootstrap_root_sha256: input.bootstrap_root_sha256,
            current_root: input.current_root,
            timestamp: input.timestamp,
            snapshot: input.snapshot,
            targets: input.targets,
            trusted_root_target_name: input.trusted_root_target_name,
            trusted_root_target_sha256: input.trusted_root_target_sha256,
            signing_config_target_name: input.signing_config_target_name,
            signing_config_target_sha256: input.signing_config_target_sha256,
            commitment: [0_u8; 32],
        };
        descriptor.commitment = tuf_state_commitment(&descriptor);
        Ok(descriptor)
    }

    pub(super) const fn bootstrap_root_version(&self) -> u64 {
        self.bootstrap_root_version
    }

    pub(super) const fn bootstrap_root_sha256(&self) -> [u8; 32] {
        self.bootstrap_root_sha256
    }

    pub(super) const fn current_root(&self) -> V2VersionedMetadataIdentity {
        self.current_root
    }

    pub(super) const fn timestamp(&self) -> V2VersionedMetadataIdentity {
        self.timestamp
    }

    pub(super) const fn snapshot(&self) -> V2VersionedMetadataIdentity {
        self.snapshot
    }

    pub(super) const fn targets(&self) -> V2VersionedMetadataIdentity {
        self.targets
    }

    pub(super) fn trusted_root_target_name(&self) -> &str {
        &self.trusted_root_target_name
    }

    pub(super) const fn trusted_root_target_sha256(&self) -> [u8; 32] {
        self.trusted_root_target_sha256
    }

    pub(super) fn signing_config_target_name(&self) -> &str {
        &self.signing_config_target_name
    }

    pub(super) const fn signing_config_target_sha256(&self) -> [u8; 32] {
        self.signing_config_target_sha256
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::from_utf8(self.canonical_body_bytes())
            .expect("TUF trust-state descriptor body is ASCII");
        push_field(&mut output, "descriptor_commitment", &hex32(self.commitment));
        output.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        push_field(
            &mut output,
            "descriptor_schema_revision",
            V2_TUF_TRUST_STATE_DESCRIPTOR_SCHEMA,
        );
        push_field(&mut output, "repository", V2_REPOSITORY_IDENTITY);
        push_field(
            &mut output,
            "bootstrap_root_version",
            &self.bootstrap_root_version.to_string(),
        );
        push_field(
            &mut output,
            "bootstrap_root_sha256",
            &hex32(self.bootstrap_root_sha256),
        );
        push_versioned_metadata(&mut output, "current_root", self.current_root);
        push_versioned_metadata(&mut output, "timestamp", self.timestamp);
        push_versioned_metadata(&mut output, "snapshot", self.snapshot);
        push_versioned_metadata(&mut output, "targets", self.targets);
        push_field(
            &mut output,
            "trusted_root_target_name",
            &self.trusted_root_target_name,
        );
        push_field(
            &mut output,
            "trusted_root_target_sha256",
            &hex32(self.trusted_root_target_sha256),
        );
        push_field(
            &mut output,
            "signing_config_target_name",
            &self.signing_config_target_name,
        );
        push_field(
            &mut output,
            "signing_config_target_sha256",
            &hex32(self.signing_config_target_sha256),
        );
        push_false_authority_fields(&mut output);
        output.into_bytes()
    }

    pub(super) fn parse_canonical(input: &[u8]) -> Result<Self, V2ExternalLogTrustError> {
        let lines = canonical_lines(input, 21)?;
        if field(lines[0], "descriptor_schema_revision")?
            != V2_TUF_TRUST_STATE_DESCRIPTOR_SCHEMA
        {
            return Err(V2ExternalLogTrustError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY {
            return Err(V2ExternalLogTrustError::WrongRepository);
        }
        let descriptor = Self::candidate(V2TufTrustStateCandidateInput {
            bootstrap_root_version: canonical_u64(field(lines[2], "bootstrap_root_version")?)?,
            bootstrap_root_sha256: decode_hex32(field(lines[3], "bootstrap_root_sha256")?)?,
            current_root: parse_versioned_metadata(&lines[4..6], "current_root")?,
            timestamp: parse_versioned_metadata(&lines[6..8], "timestamp")?,
            snapshot: parse_versioned_metadata(&lines[8..10], "snapshot")?,
            targets: parse_versioned_metadata(&lines[10..12], "targets")?,
            trusted_root_target_name: field(lines[12], "trusted_root_target_name")?.to_owned(),
            trusted_root_target_sha256: decode_hex32(field(
                lines[13],
                "trusted_root_target_sha256",
            )?)?,
            signing_config_target_name: field(lines[14], "signing_config_target_name")?.to_owned(),
            signing_config_target_sha256: decode_hex32(field(
                lines[15],
                "signing_config_target_sha256",
            )?)?,
        })?;
        require_false_authority_fields(&lines[16..20])?;
        let claimed = decode_hex32(field(lines[20], "descriptor_commitment")?)?;
        if descriptor.commitment != claimed {
            return Err(V2ExternalLogTrustError::CommitmentMismatch);
        }
        if descriptor.canonical_bytes() != input {
            return Err(V2ExternalLogTrustError::NonCanonicalEncoding);
        }
        Ok(descriptor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2TransparencyLogIdentityDescriptor {
    log_origin: String,
    public_key_sha256: [u8; 32],
    checkpoint_key_id: [u8; 32],
    key_metadata_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2TransparencyLogIdentityDescriptor {
    pub(super) fn candidate(
        log_origin: &str,
        public_key_sha256: [u8; 32],
        checkpoint_key_id: [u8; 32],
        key_metadata_commitment: [u8; 32],
    ) -> Result<Self, V2ExternalLogTrustError> {
        validate_atom(log_origin)?;
        let mut descriptor = Self {
            log_origin: log_origin.to_owned(),
            public_key_sha256,
            checkpoint_key_id,
            key_metadata_commitment,
            commitment: [0_u8; 32],
        };
        descriptor.commitment = log_identity_commitment(&descriptor);
        Ok(descriptor)
    }

    pub(super) fn log_origin(&self) -> &str {
        &self.log_origin
    }

    pub(super) const fn public_key_sha256(&self) -> [u8; 32] {
        self.public_key_sha256
    }

    pub(super) const fn checkpoint_key_id(&self) -> [u8; 32] {
        self.checkpoint_key_id
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::from_utf8(self.canonical_body_bytes())
            .expect("transparency-log identity descriptor body is ASCII");
        push_field(&mut output, "descriptor_commitment", &hex32(self.commitment));
        output.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        push_field(
            &mut output,
            "descriptor_schema_revision",
            V2_TRANSPARENCY_LOG_IDENTITY_DESCRIPTOR_SCHEMA,
        );
        push_field(&mut output, "repository", V2_REPOSITORY_IDENTITY);
        push_field(&mut output, "provider", V2_TRANSPARENCY_PROVIDER);
        push_field(&mut output, "provider_role", V2_TRANSPARENCY_ROLE);
        push_field(&mut output, "log_origin", &self.log_origin);
        push_field(
            &mut output,
            "public_key_sha256",
            &hex32(self.public_key_sha256),
        );
        push_field(
            &mut output,
            "checkpoint_key_id",
            &hex32(self.checkpoint_key_id),
        );
        push_field(
            &mut output,
            "key_metadata_commitment",
            &hex32(self.key_metadata_commitment),
        );
        push_false_authority_fields(&mut output);
        output.into_bytes()
    }

    pub(super) fn parse_canonical(input: &[u8]) -> Result<Self, V2ExternalLogTrustError> {
        let lines = canonical_lines(input, 13)?;
        if field(lines[0], "descriptor_schema_revision")?
            != V2_TRANSPARENCY_LOG_IDENTITY_DESCRIPTOR_SCHEMA
        {
            return Err(V2ExternalLogTrustError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY
            || field(lines[2], "provider")? != V2_TRANSPARENCY_PROVIDER
            || field(lines[3], "provider_role")? != V2_TRANSPARENCY_ROLE
        {
            return Err(V2ExternalLogTrustError::WrongIdentity);
        }
        let descriptor = Self::candidate(
            field(lines[4], "log_origin")?,
            decode_hex32(field(lines[5], "public_key_sha256")?)?,
            decode_hex32(field(lines[6], "checkpoint_key_id")?)?,
            decode_hex32(field(lines[7], "key_metadata_commitment")?)?,
        )?;
        require_false_authority_fields(&lines[8..12])?;
        let claimed = decode_hex32(field(lines[12], "descriptor_commitment")?)?;
        if descriptor.commitment != claimed {
            return Err(V2ExternalLogTrustError::CommitmentMismatch);
        }
        if descriptor.canonical_bytes() != input {
            return Err(V2ExternalLogTrustError::NonCanonicalEncoding);
        }
        Ok(descriptor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2TufLogBindingCandidate {
    tuf_state_commitment: [u8; 32],
    log_identity_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2TufLogBindingCandidate {
    pub(super) fn candidate(
        tuf_state: &V2TufTrustStateDescriptor,
        log_identity: &V2TransparencyLogIdentityDescriptor,
    ) -> Self {
        let mut binding = Self {
            tuf_state_commitment: tuf_state.commitment(),
            log_identity_commitment: log_identity.commitment(),
            commitment: [0_u8; 32],
        };
        binding.commitment = tuf_log_binding_commitment(&binding);
        binding
    }

    pub(super) const fn tuf_state_commitment(&self) -> [u8; 32] {
        self.tuf_state_commitment
    }

    pub(super) const fn log_identity_commitment(&self) -> [u8; 32] {
        self.log_identity_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::from_utf8(self.canonical_body_bytes())
            .expect("TUF/log binding candidate body is ASCII");
        push_field(&mut output, "candidate_commitment", &hex32(self.commitment));
        output.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        push_field(
            &mut output,
            "candidate_schema_revision",
            V2_TUF_LOG_BINDING_CANDIDATE_SCHEMA,
        );
        push_field(&mut output, "repository", V2_REPOSITORY_IDENTITY);
        push_field(&mut output, "provider", V2_TRANSPARENCY_PROVIDER);
        push_field(&mut output, "provider_role", V2_TRANSPARENCY_ROLE);
        push_field(
            &mut output,
            "tuf_state_commitment",
            &hex32(self.tuf_state_commitment),
        );
        push_field(
            &mut output,
            "log_identity_commitment",
            &hex32(self.log_identity_commitment),
        );
        push_false_authority_fields(&mut output);
        output.into_bytes()
    }

    pub(super) fn parse_canonical(input: &[u8]) -> Result<Self, V2ExternalLogTrustError> {
        let lines = canonical_lines(input, 11)?;
        if field(lines[0], "candidate_schema_revision")? != V2_TUF_LOG_BINDING_CANDIDATE_SCHEMA {
            return Err(V2ExternalLogTrustError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY
            || field(lines[2], "provider")? != V2_TRANSPARENCY_PROVIDER
            || field(lines[3], "provider_role")? != V2_TRANSPARENCY_ROLE
        {
            return Err(V2ExternalLogTrustError::WrongIdentity);
        }
        let mut binding = Self {
            tuf_state_commitment: decode_hex32(field(lines[4], "tuf_state_commitment")?)?,
            log_identity_commitment: decode_hex32(field(lines[5], "log_identity_commitment")?)?,
            commitment: [0_u8; 32],
        };
        require_false_authority_fields(&lines[6..10])?;
        binding.commitment = tuf_log_binding_commitment(&binding);
        let claimed = decode_hex32(field(lines[10], "candidate_commitment")?)?;
        if binding.commitment != claimed {
            return Err(V2ExternalLogTrustError::CommitmentMismatch);
        }
        if binding.canonical_bytes() != input {
            return Err(V2ExternalLogTrustError::NonCanonicalEncoding);
        }
        Ok(binding)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2ExternalLogAdmissionCandidate {
    sequence: u64,
    predecessor_commitment: Option<[u8; 32]>,
    log_identity_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2ExternalLogAdmissionCandidate {
    pub(super) fn genesis(log_identity: &V2TransparencyLogIdentityDescriptor) -> Self {
        Self::new(1, None, log_identity.commitment())
    }

    pub(super) fn successor(
        previous: &Self,
        log_identity: &V2TransparencyLogIdentityDescriptor,
    ) -> Result<Self, V2ExternalLogTrustError> {
        if log_identity.commitment() == previous.log_identity_commitment {
            return Err(V2ExternalLogTrustError::NoOpAdmissionRotation);
        }
        let sequence = previous
            .sequence
            .checked_add(1)
            .ok_or(V2ExternalLogTrustError::SequenceExhausted)?;
        Ok(Self::new(
            sequence,
            Some(previous.commitment),
            log_identity.commitment(),
        ))
    }

    fn new(
        sequence: u64,
        predecessor_commitment: Option<[u8; 32]>,
        log_identity_commitment: [u8; 32],
    ) -> Self {
        let mut candidate = Self {
            sequence,
            predecessor_commitment,
            log_identity_commitment,
            commitment: [0_u8; 32],
        };
        candidate.commitment = admission_candidate_commitment(&candidate);
        candidate
    }

    pub(super) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) const fn predecessor_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_commitment
    }

    pub(super) const fn log_identity_commitment(&self) -> [u8; 32] {
        self.log_identity_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::from_utf8(self.canonical_body_bytes())
            .expect("external-log admission candidate body is ASCII");
        push_field(&mut output, "candidate_commitment", &hex32(self.commitment));
        output.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        push_field(
            &mut output,
            "candidate_schema_revision",
            V2_EXTERNAL_LOG_ADMISSION_CANDIDATE_SCHEMA,
        );
        push_field(&mut output, "repository", V2_REPOSITORY_IDENTITY);
        push_field(
            &mut output,
            "externality_policy_revision",
            V2_EXTERNALITY_POLICY_REVISION,
        );
        push_field(&mut output, "provider", V2_TRANSPARENCY_PROVIDER);
        push_field(&mut output, "provider_role", V2_TRANSPARENCY_ROLE);
        push_field(&mut output, "admission_sequence", &self.sequence.to_string());
        push_field(
            &mut output,
            "predecessor_admission_commitment",
            &optional_hex(self.predecessor_commitment),
        );
        push_field(
            &mut output,
            "log_identity_commitment",
            &hex32(self.log_identity_commitment),
        );
        push_false_authority_fields(&mut output);
        output.into_bytes()
    }

    pub(super) fn parse_canonical(input: &[u8]) -> Result<Self, V2ExternalLogTrustError> {
        let lines = canonical_lines(input, 13)?;
        if field(lines[0], "candidate_schema_revision")?
            != V2_EXTERNAL_LOG_ADMISSION_CANDIDATE_SCHEMA
        {
            return Err(V2ExternalLogTrustError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY
            || field(lines[2], "externality_policy_revision")? != V2_EXTERNALITY_POLICY_REVISION
            || field(lines[3], "provider")? != V2_TRANSPARENCY_PROVIDER
            || field(lines[4], "provider_role")? != V2_TRANSPARENCY_ROLE
        {
            return Err(V2ExternalLogTrustError::WrongIdentity);
        }
        let sequence = canonical_u64(field(lines[5], "admission_sequence")?)?;
        let predecessor_commitment = optional_commitment(field(
            lines[6],
            "predecessor_admission_commitment",
        )?)?;
        if sequence == 0 || (sequence == 1) != predecessor_commitment.is_none() {
            return Err(V2ExternalLogTrustError::InvalidAdmissionCandidate);
        }
        let log_identity_commitment =
            decode_hex32(field(lines[7], "log_identity_commitment")?)?;
        require_false_authority_fields(&lines[8..12])?;
        let candidate = Self::new(sequence, predecessor_commitment, log_identity_commitment);
        let claimed = decode_hex32(field(lines[12], "candidate_commitment")?)?;
        if candidate.commitment != claimed {
            return Err(V2ExternalLogTrustError::CommitmentMismatch);
        }
        if candidate.canonical_bytes() != input {
            return Err(V2ExternalLogTrustError::NonCanonicalEncoding);
        }
        Ok(candidate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2ExternalLogTrustCheckpoint {
    checkpoint_sequence: u64,
    bootstrap_root_version: u64,
    bootstrap_root_sha256: [u8; 32],
    current_root: V2VersionedMetadataIdentity,
    timestamp: V2VersionedMetadataIdentity,
    snapshot: V2VersionedMetadataIdentity,
    targets: V2VersionedMetadataIdentity,
    trusted_root_target_name: String,
    trusted_root_target_sha256: [u8; 32],
    signing_config_target_name: String,
    signing_config_target_sha256: [u8; 32],
    tuf_state_commitment: [u8; 32],
    log_identity_commitment: [u8; 32],
    tuf_log_binding_commitment: [u8; 32],
    admission_sequence: u64,
    admission_commitment: [u8; 32],
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2ExternalLogTrustCheckpoint {
    pub(super) fn genesis_candidate(
        tuf_state: &V2TufTrustStateDescriptor,
        log_identity: &V2TransparencyLogIdentityDescriptor,
        binding: &V2TufLogBindingCandidate,
        admission: &V2ExternalLogAdmissionCandidate,
    ) -> Result<Self, V2ExternalLogTrustError> {
        require_candidate_relations(tuf_state, log_identity, binding, admission)?;
        if admission.sequence() != 1 || admission.predecessor_commitment().is_some() {
            return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
        }
        Ok(Self::new(
            1,
            tuf_state,
            log_identity,
            binding,
            admission,
            None,
        ))
    }

    pub(super) fn successor_candidate(
        previous: &Self,
        tuf_state: &V2TufTrustStateDescriptor,
        log_identity: &V2TransparencyLogIdentityDescriptor,
        binding: &V2TufLogBindingCandidate,
        admission: &V2ExternalLogAdmissionCandidate,
    ) -> Result<Self, V2ExternalLogTrustError> {
        require_candidate_relations(tuf_state, log_identity, binding, admission)?;
        if tuf_state.bootstrap_root_version() != previous.bootstrap_root_version
            || tuf_state.bootstrap_root_sha256() != previous.bootstrap_root_sha256
        {
            return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
        }
        match admission.sequence().cmp(&previous.admission_sequence) {
            std::cmp::Ordering::Less => {
                return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
            }
            std::cmp::Ordering::Equal => {
                if admission.commitment() != previous.admission_commitment
                    || log_identity.commitment() != previous.log_identity_commitment
                {
                    return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
                }
            }
            std::cmp::Ordering::Greater => {
                let expected = previous
                    .admission_sequence
                    .checked_add(1)
                    .ok_or(V2ExternalLogTrustError::SequenceExhausted)?;
                if admission.sequence() != expected
                    || admission.predecessor_commitment() != Some(previous.admission_commitment)
                    || log_identity.commitment() == previous.log_identity_commitment
                {
                    return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
                }
            }
        }
        let checkpoint_sequence = previous
            .checkpoint_sequence
            .checked_add(1)
            .ok_or(V2ExternalLogTrustError::SequenceExhausted)?;
        let candidate = Self::new(
            checkpoint_sequence,
            tuf_state,
            log_identity,
            binding,
            admission,
            Some(previous.commitment),
        );
        match candidate.classify_against(previous) {
            V2ExternalLogTrustDisposition::AheadRequiresVerifiedTransition => Ok(candidate),
            V2ExternalLogTrustDisposition::ExactCandidateState => {
                Err(V2ExternalLogTrustError::NoOpCheckpoint)
            }
            _ => Err(V2ExternalLogTrustError::InvalidCheckpointShape),
        }
    }

    fn new(
        checkpoint_sequence: u64,
        tuf_state: &V2TufTrustStateDescriptor,
        log_identity: &V2TransparencyLogIdentityDescriptor,
        binding: &V2TufLogBindingCandidate,
        admission: &V2ExternalLogAdmissionCandidate,
        predecessor_checkpoint_commitment: Option<[u8; 32]>,
    ) -> Self {
        let mut checkpoint = Self {
            checkpoint_sequence,
            bootstrap_root_version: tuf_state.bootstrap_root_version(),
            bootstrap_root_sha256: tuf_state.bootstrap_root_sha256(),
            current_root: tuf_state.current_root(),
            timestamp: tuf_state.timestamp(),
            snapshot: tuf_state.snapshot(),
            targets: tuf_state.targets(),
            trusted_root_target_name: tuf_state.trusted_root_target_name().to_owned(),
            trusted_root_target_sha256: tuf_state.trusted_root_target_sha256(),
            signing_config_target_name: tuf_state.signing_config_target_name().to_owned(),
            signing_config_target_sha256: tuf_state.signing_config_target_sha256(),
            tuf_state_commitment: tuf_state.commitment(),
            log_identity_commitment: log_identity.commitment(),
            tuf_log_binding_commitment: binding.commitment(),
            admission_sequence: admission.sequence(),
            admission_commitment: admission.commitment(),
            predecessor_checkpoint_commitment,
            commitment: [0_u8; 32],
        };
        checkpoint.commitment = trust_checkpoint_commitment(&checkpoint);
        checkpoint
    }

    pub(super) const fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
    }

    pub(super) const fn admission_sequence(&self) -> u64 {
        self.admission_sequence
    }

    pub(super) const fn log_identity_commitment(&self) -> [u8; 32] {
        self.log_identity_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) const fn predecessor_checkpoint_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_checkpoint_commitment
    }

    pub(super) fn classify_against(&self, accepted: &Self) -> V2ExternalLogTrustDisposition {
        if self.bootstrap_root_version != accepted.bootstrap_root_version
            || self.bootstrap_root_sha256 != accepted.bootstrap_root_sha256
        {
            return V2ExternalLogTrustDisposition::Incomparable;
        }

        let metadata_equivocation =
            same_version_different_digest(self.current_root, accepted.current_root)
                || same_version_different_digest(self.timestamp, accepted.timestamp)
                || same_version_different_digest(self.snapshot, accepted.snapshot)
                || same_version_different_digest(self.targets, accepted.targets);
        let target_equivocation = self.targets.version() == accepted.targets.version()
            && (self.trusted_root_target_name != accepted.trusted_root_target_name
                || self.trusted_root_target_sha256 != accepted.trusted_root_target_sha256
                || self.signing_config_target_name != accepted.signing_config_target_name
                || self.signing_config_target_sha256 != accepted.signing_config_target_sha256);
        let admission_equivocation = self.admission_sequence == accepted.admission_sequence
            && (self.admission_commitment != accepted.admission_commitment
                || self.log_identity_commitment != accepted.log_identity_commitment);
        let exact_tuf_components = self.current_root == accepted.current_root
            && self.timestamp == accepted.timestamp
            && self.snapshot == accepted.snapshot
            && self.targets == accepted.targets
            && self.trusted_root_target_name == accepted.trusted_root_target_name
            && self.trusted_root_target_sha256 == accepted.trusted_root_target_sha256
            && self.signing_config_target_name == accepted.signing_config_target_name
            && self.signing_config_target_sha256 == accepted.signing_config_target_sha256;
        let tuf_commitment_equivocation = exact_tuf_components
            && self.tuf_state_commitment != accepted.tuf_state_commitment;
        let binding_equivocation = self.tuf_state_commitment == accepted.tuf_state_commitment
            && self.log_identity_commitment == accepted.log_identity_commitment
            && self.tuf_log_binding_commitment != accepted.tuf_log_binding_commitment;
        let checkpoint_equivocation = self.checkpoint_sequence == accepted.checkpoint_sequence
            && self.commitment != accepted.commitment;

        if metadata_equivocation
            || target_equivocation
            || admission_equivocation
            || tuf_commitment_equivocation
            || binding_equivocation
            || checkpoint_equivocation
        {
            return V2ExternalLogTrustDisposition::Equivocation;
        }

        let semantic_exact = exact_tuf_components
            && self.tuf_state_commitment == accepted.tuf_state_commitment
            && self.log_identity_commitment == accepted.log_identity_commitment
            && self.tuf_log_binding_commitment == accepted.tuf_log_binding_commitment
            && self.admission_sequence == accepted.admission_sequence
            && self.admission_commitment == accepted.admission_commitment;
        if semantic_exact
            && self.checkpoint_sequence == accepted.checkpoint_sequence
            && self.commitment == accepted.commitment
        {
            return V2ExternalLogTrustDisposition::ExactCandidateState;
        }
        if semantic_exact {
            return V2ExternalLogTrustDisposition::Equivocation;
        }

        let order = [
            self.checkpoint_sequence.cmp(&accepted.checkpoint_sequence),
            self.current_root.version().cmp(&accepted.current_root.version()),
            self.timestamp.version().cmp(&accepted.timestamp.version()),
            self.snapshot.version().cmp(&accepted.snapshot.version()),
            self.targets.version().cmp(&accepted.targets.version()),
            self.admission_sequence.cmp(&accepted.admission_sequence),
        ];
        let any_less = order.iter().any(|value| value.is_lt());
        let any_greater = order.iter().any(|value| value.is_gt());
        match (any_less, any_greater) {
            (true, true) => V2ExternalLogTrustDisposition::Incomparable,
            (true, false) => V2ExternalLogTrustDisposition::Rollback,
            (false, true) => V2ExternalLogTrustDisposition::AheadRequiresVerifiedTransition,
            (false, false) => V2ExternalLogTrustDisposition::Equivocation,
        }
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::from_utf8(self.canonical_body_bytes())
            .expect("external-log trust checkpoint body is ASCII");
        push_field(&mut output, "checkpoint_commitment", &hex32(self.commitment));
        output.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        push_field(
            &mut output,
            "checkpoint_schema_revision",
            V2_EXTERNAL_LOG_TRUST_CHECKPOINT_SCHEMA,
        );
        push_field(&mut output, "repository", V2_REPOSITORY_IDENTITY);
        push_field(
            &mut output,
            "checkpoint_sequence",
            &self.checkpoint_sequence.to_string(),
        );
        push_field(
            &mut output,
            "bootstrap_root_version",
            &self.bootstrap_root_version.to_string(),
        );
        push_field(
            &mut output,
            "bootstrap_root_sha256",
            &hex32(self.bootstrap_root_sha256),
        );
        push_versioned_metadata(&mut output, "current_root", self.current_root);
        push_versioned_metadata(&mut output, "timestamp", self.timestamp);
        push_versioned_metadata(&mut output, "snapshot", self.snapshot);
        push_versioned_metadata(&mut output, "targets", self.targets);
        push_field(
            &mut output,
            "trusted_root_target_name",
            &self.trusted_root_target_name,
        );
        push_field(
            &mut output,
            "trusted_root_target_sha256",
            &hex32(self.trusted_root_target_sha256),
        );
        push_field(
            &mut output,
            "signing_config_target_name",
            &self.signing_config_target_name,
        );
        push_field(
            &mut output,
            "signing_config_target_sha256",
            &hex32(self.signing_config_target_sha256),
        );
        push_field(
            &mut output,
            "tuf_state_commitment",
            &hex32(self.tuf_state_commitment),
        );
        push_field(
            &mut output,
            "log_identity_commitment",
            &hex32(self.log_identity_commitment),
        );
        push_field(
            &mut output,
            "tuf_log_binding_commitment",
            &hex32(self.tuf_log_binding_commitment),
        );
        push_field(
            &mut output,
            "admission_sequence",
            &self.admission_sequence.to_string(),
        );
        push_field(
            &mut output,
            "admission_commitment",
            &hex32(self.admission_commitment),
        );
        push_field(
            &mut output,
            "predecessor_checkpoint_commitment",
            &optional_hex(self.predecessor_checkpoint_commitment),
        );
        push_false_authority_fields(&mut output);
        output.into_bytes()
    }

    pub(super) fn parse_canonical(input: &[u8]) -> Result<Self, V2ExternalLogTrustError> {
        let lines = canonical_lines(input, 28)?;
        if field(lines[0], "checkpoint_schema_revision")?
            != V2_EXTERNAL_LOG_TRUST_CHECKPOINT_SCHEMA
        {
            return Err(V2ExternalLogTrustError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY {
            return Err(V2ExternalLogTrustError::WrongRepository);
        }
        let checkpoint_sequence = canonical_u64(field(lines[2], "checkpoint_sequence")?)?;
        let bootstrap_root_version = canonical_u64(field(lines[3], "bootstrap_root_version")?)?;
        let bootstrap_root_sha256 = decode_hex32(field(lines[4], "bootstrap_root_sha256")?)?;
        let current_root = parse_versioned_metadata(&lines[5..7], "current_root")?;
        let timestamp = parse_versioned_metadata(&lines[7..9], "timestamp")?;
        let snapshot = parse_versioned_metadata(&lines[9..11], "snapshot")?;
        let targets = parse_versioned_metadata(&lines[11..13], "targets")?;
        let trusted_root_target_name = field(lines[13], "trusted_root_target_name")?.to_owned();
        let trusted_root_target_sha256 =
            decode_hex32(field(lines[14], "trusted_root_target_sha256")?)?;
        let signing_config_target_name = field(lines[15], "signing_config_target_name")?.to_owned();
        let signing_config_target_sha256 =
            decode_hex32(field(lines[16], "signing_config_target_sha256")?)?;
        validate_atom(&trusted_root_target_name)?;
        validate_atom(&signing_config_target_name)?;
        let tuf_state_commitment = decode_hex32(field(lines[17], "tuf_state_commitment")?)?;
        let log_identity_commitment = decode_hex32(field(lines[18], "log_identity_commitment")?)?;
        let tuf_log_binding_commitment =
            decode_hex32(field(lines[19], "tuf_log_binding_commitment")?)?;
        let admission_sequence = canonical_u64(field(lines[20], "admission_sequence")?)?;
        let admission_commitment = decode_hex32(field(lines[21], "admission_commitment")?)?;
        let predecessor_checkpoint_commitment = optional_commitment(field(
            lines[22],
            "predecessor_checkpoint_commitment",
        )?)?;
        if checkpoint_sequence == 0
            || bootstrap_root_version == 0
            || current_root.version() < bootstrap_root_version
            || admission_sequence == 0
            || (checkpoint_sequence == 1) != predecessor_checkpoint_commitment.is_none()
        {
            return Err(V2ExternalLogTrustError::InvalidCheckpointShape);
        }
        require_false_authority_fields(&lines[23..27])?;
        let claimed = decode_hex32(field(lines[27], "checkpoint_commitment")?)?;
        let mut checkpoint = Self {
            checkpoint_sequence,
            bootstrap_root_version,
            bootstrap_root_sha256,
            current_root,
            timestamp,
            snapshot,
            targets,
            trusted_root_target_name,
            trusted_root_target_sha256,
            signing_config_target_name,
            signing_config_target_sha256,
            tuf_state_commitment,
            log_identity_commitment,
            tuf_log_binding_commitment,
            admission_sequence,
            admission_commitment,
            predecessor_checkpoint_commitment,
            commitment: [0_u8; 32],
        };
        checkpoint.commitment = trust_checkpoint_commitment(&checkpoint);
        if checkpoint.commitment != claimed {
            return Err(V2ExternalLogTrustError::CommitmentMismatch);
        }
        if checkpoint.canonical_bytes() != input {
            return Err(V2ExternalLogTrustError::NonCanonicalEncoding);
        }
        Ok(checkpoint)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalLogTrustDisposition {
    ExactCandidateState,
    Rollback,
    Equivocation,
    AheadRequiresVerifiedTransition,
    Incomparable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalLogTrustError {
    InvalidTufState,
    InvalidAtom,
    InvalidAdmissionCandidate,
    CandidateBindingMismatch,
    SequenceExhausted,
    NoOpAdmissionRotation,
    InvalidCheckpointShape,
    NoOpCheckpoint,
    InvalidUtf8,
    MissingTrailingNewline,
    WrongFieldCount,
    UnsupportedSchema,
    WrongRepository,
    WrongIdentity,
    InvalidNumber,
    InvalidHex,
    AuthorityEscalation,
    CommitmentMismatch,
    NonCanonicalEncoding,
}

fn require_candidate_relations(
    tuf_state: &V2TufTrustStateDescriptor,
    log_identity: &V2TransparencyLogIdentityDescriptor,
    binding: &V2TufLogBindingCandidate,
    admission: &V2ExternalLogAdmissionCandidate,
) -> Result<(), V2ExternalLogTrustError> {
    if binding.tuf_state_commitment() != tuf_state.commitment()
        || binding.log_identity_commitment() != log_identity.commitment()
        || admission.log_identity_commitment() != log_identity.commitment()
    {
        return Err(V2ExternalLogTrustError::CandidateBindingMismatch);
    }
    Ok(())
}

fn same_version_different_digest(
    candidate: V2VersionedMetadataIdentity,
    accepted: V2VersionedMetadataIdentity,
) -> bool {
    candidate.version() == accepted.version() && candidate.sha256() != accepted.sha256()
}

fn tuf_state_commitment(descriptor: &V2TufTrustStateDescriptor) -> [u8; 32] {
    domain_commitment(
        V2_TUF_TRUST_STATE_DESCRIPTOR_COMMITMENT_REVISION,
        &descriptor.canonical_body_bytes(),
    )
}

fn log_identity_commitment(descriptor: &V2TransparencyLogIdentityDescriptor) -> [u8; 32] {
    domain_commitment(
        V2_TRANSPARENCY_LOG_IDENTITY_COMMITMENT_REVISION,
        &descriptor.canonical_body_bytes(),
    )
}

fn tuf_log_binding_commitment(binding: &V2TufLogBindingCandidate) -> [u8; 32] {
    domain_commitment(
        V2_TUF_LOG_BINDING_CANDIDATE_COMMITMENT_REVISION,
        &binding.canonical_body_bytes(),
    )
}

fn admission_candidate_commitment(candidate: &V2ExternalLogAdmissionCandidate) -> [u8; 32] {
    domain_commitment(
        V2_EXTERNAL_LOG_ADMISSION_CANDIDATE_COMMITMENT_REVISION,
        &candidate.canonical_body_bytes(),
    )
}

fn trust_checkpoint_commitment(checkpoint: &V2ExternalLogTrustCheckpoint) -> [u8; 32] {
    domain_commitment(
        V2_EXTERNAL_LOG_TRUST_CHECKPOINT_COMMITMENT_REVISION,
        &checkpoint.canonical_body_bytes(),
    )
}

fn domain_commitment(revision: &str, body: &[u8]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, revision.as_bytes());
    encode_bytes(&mut bytes, body);
    *blake3::hash(&bytes).as_bytes()
}

fn push_versioned_metadata(
    output: &mut String,
    prefix: &str,
    identity: V2VersionedMetadataIdentity,
) {
    push_field(
        output,
        &format!("{prefix}_version"),
        &identity.version().to_string(),
    );
    push_field(
        output,
        &format!("{prefix}_sha256"),
        &hex32(identity.sha256()),
    );
}

fn parse_versioned_metadata(
    lines: &[&str],
    prefix: &str,
) -> Result<V2VersionedMetadataIdentity, V2ExternalLogTrustError> {
    if lines.len() != 2 {
        return Err(V2ExternalLogTrustError::WrongFieldCount);
    }
    let version_key = format!("{prefix}_version");
    let digest_key = format!("{prefix}_sha256");
    V2VersionedMetadataIdentity::candidate(
        canonical_u64(dynamic_field(lines[0], &version_key)?)?,
        decode_hex32(dynamic_field(lines[1], &digest_key)?)?,
    )
}

fn push_false_authority_fields(output: &mut String) {
    push_field(output, "tuf_verification_complete", "false");
    push_field(output, "eureka_external_log_admitted", "false");
    push_field(output, "externality_verified", "false");
    push_field(output, "execution_authority_granted", "false");
}

fn require_false_authority_fields(lines: &[&str]) -> Result<(), V2ExternalLogTrustError> {
    if lines.len() != 4
        || field(lines[0], "tuf_verification_complete")? != "false"
        || field(lines[1], "eureka_external_log_admitted")? != "false"
        || field(lines[2], "externality_verified")? != "false"
        || field(lines[3], "execution_authority_granted")? != "false"
    {
        return Err(V2ExternalLogTrustError::AuthorityEscalation);
    }
    Ok(())
}

fn canonical_lines(
    input: &[u8],
    expected: usize,
) -> Result<Vec<&str>, V2ExternalLogTrustError> {
    let text = std::str::from_utf8(input).map_err(|_| V2ExternalLogTrustError::InvalidUtf8)?;
    if !text.ends_with('\n') {
        return Err(V2ExternalLogTrustError::MissingTrailingNewline);
    }
    let lines: Vec<&str> = text[..text.len() - 1].split('\n').collect();
    if lines.len() != expected {
        return Err(V2ExternalLogTrustError::WrongFieldCount);
    }
    Ok(lines)
}

fn field<'a>(line: &'a str, expected_key: &str) -> Result<&'a str, V2ExternalLogTrustError> {
    dynamic_field(line, expected_key)
}

fn dynamic_field<'a>(
    line: &'a str,
    expected_key: &str,
) -> Result<&'a str, V2ExternalLogTrustError> {
    let Some((key, value)) = line.split_once('=') else {
        return Err(V2ExternalLogTrustError::WrongIdentity);
    };
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(V2ExternalLogTrustError::WrongIdentity);
    }
    Ok(value)
}

fn validate_atom(value: &str) -> Result<(), V2ExternalLogTrustError> {
    if value.is_empty()
        || value.len() > 256
        || !value.is_ascii()
        || value
            .bytes()
            .any(|byte| byte <= 0x20 || byte == b'=' || byte == 0x7f)
    {
        return Err(V2ExternalLogTrustError::InvalidAtom);
    }
    Ok(())
}

fn canonical_u64(value: &str) -> Result<u64, V2ExternalLogTrustError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(V2ExternalLogTrustError::InvalidNumber);
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2ExternalLogTrustError::InvalidNumber)?;
    if parsed.to_string() != value {
        return Err(V2ExternalLogTrustError::InvalidNumber);
    }
    Ok(parsed)
}

fn optional_commitment(value: &str) -> Result<Option<[u8; 32]>, V2ExternalLogTrustError> {
    if value == "none" {
        Ok(None)
    } else {
        decode_hex32(value).map(Some)
    }
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2ExternalLogTrustError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2ExternalLogTrustError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2ExternalLogTrustError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2ExternalLogTrustError::InvalidHex)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn optional_hex(value: Option<[u8; 32]>) -> String {
    value.map(hex32).unwrap_or_else(|| "none".to_owned())
}

fn hex32(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn push_field(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('=');
    output.push_str(value);
    output.push('\n');
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn metadata(version: u64, byte: u8) -> V2VersionedMetadataIdentity {
        V2VersionedMetadataIdentity::candidate(version, digest(byte)).unwrap()
    }

    fn tuf(
        root: u64,
        timestamp: u64,
        snapshot: u64,
        targets: u64,
        target_byte: u8,
    ) -> V2TufTrustStateDescriptor {
        V2TufTrustStateDescriptor::candidate(V2TufTrustStateCandidateInput {
            bootstrap_root_version: 10,
            bootstrap_root_sha256: digest(0x10),
            current_root: metadata(root, u8::try_from(root).unwrap()),
            timestamp: metadata(timestamp, target_byte.wrapping_add(10)),
            snapshot: metadata(snapshot, target_byte.wrapping_add(11)),
            targets: metadata(targets, target_byte.wrapping_add(12)),
            trusted_root_target_name: "trusted_root.json".to_owned(),
            trusted_root_target_sha256: digest(target_byte),
            signing_config_target_name: "signing_config.v2.json".to_owned(),
            signing_config_target_sha256: digest(target_byte.wrapping_add(1)),
        })
        .unwrap()
    }

    fn log(key_byte: u8) -> V2TransparencyLogIdentityDescriptor {
        V2TransparencyLogIdentityDescriptor::candidate(
            "rekor-v2.example.invalid",
            digest(key_byte),
            digest(key_byte.wrapping_add(1)),
            digest(key_byte.wrapping_add(2)),
        )
        .unwrap()
    }

    fn genesis_parts() -> (
        V2TufTrustStateDescriptor,
        V2TransparencyLogIdentityDescriptor,
        V2TufLogBindingCandidate,
        V2ExternalLogAdmissionCandidate,
        V2ExternalLogTrustCheckpoint,
    ) {
        let state = tuf(12, 20, 18, 15, 0x30);
        let identity = log(0x40);
        let binding = V2TufLogBindingCandidate::candidate(&state, &identity);
        let admission = V2ExternalLogAdmissionCandidate::genesis(&identity);
        let checkpoint = V2ExternalLogTrustCheckpoint::genesis_candidate(
            &state,
            &identity,
            &binding,
            &admission,
        )
        .unwrap();
        (state, identity, binding, admission, checkpoint)
    }

    #[test]
    fn all_phase_a_bytes_remain_non_authorizing_and_round_trip() {
        let (state, identity, binding, admission, checkpoint) = genesis_parts();
        assert_eq!(
            V2TufTrustStateDescriptor::parse_canonical(&state.canonical_bytes()).unwrap(),
            state
        );
        assert_eq!(
            V2TransparencyLogIdentityDescriptor::parse_canonical(&identity.canonical_bytes())
                .unwrap(),
            identity
        );
        assert_eq!(
            V2TufLogBindingCandidate::parse_canonical(&binding.canonical_bytes()).unwrap(),
            binding
        );
        assert_eq!(
            V2ExternalLogAdmissionCandidate::parse_canonical(&admission.canonical_bytes()).unwrap(),
            admission
        );
        assert_eq!(
            V2ExternalLogTrustCheckpoint::parse_canonical(&checkpoint.canonical_bytes()).unwrap(),
            checkpoint
        );

        for bytes in [
            state.canonical_bytes(),
            identity.canonical_bytes(),
            binding.canonical_bytes(),
            admission.canonical_bytes(),
            checkpoint.canonical_bytes(),
        ] {
            let text = String::from_utf8(bytes).unwrap();
            assert!(text.contains("tuf_verification_complete=false\n"));
            assert!(text.contains("eureka_external_log_admitted=false\n"));
            assert!(text.contains("externality_verified=false\n"));
            assert!(text.contains("execution_authority_granted=false\n"));
        }
    }

    #[test]
    fn stable_log_identity_is_separate_from_tuf_refresh_state() {
        let (state, identity, binding, admission, accepted) = genesis_parts();
        let refreshed = tuf(12, 21, 19, 16, 0x31);
        let refreshed_binding = V2TufLogBindingCandidate::candidate(&refreshed, &identity);
        assert_ne!(binding.commitment(), refreshed_binding.commitment());
        assert_eq!(admission.log_identity_commitment(), identity.commitment());

        let candidate = V2ExternalLogTrustCheckpoint::successor_candidate(
            &accepted,
            &refreshed,
            &identity,
            &refreshed_binding,
            &admission,
        )
        .unwrap();
        assert_eq!(candidate.admission_sequence(), 1);
        assert_eq!(candidate.log_identity_commitment(), identity.commitment());
        assert_eq!(candidate.checkpoint_sequence(), 2);
        assert_eq!(
            candidate.classify_against(&accepted),
            V2ExternalLogTrustDisposition::AheadRequiresVerifiedTransition
        );
        assert_eq!(state.bootstrap_root_sha256(), refreshed.bootstrap_root_sha256());
    }

    #[test]
    fn provider_key_rotation_requires_admission_rotation() {
        let (state, first_log, _, admission, accepted) = genesis_parts();
        let second_log = log(0x50);
        let second_binding = V2TufLogBindingCandidate::candidate(&state, &second_log);

        assert_eq!(
            V2ExternalLogTrustCheckpoint::successor_candidate(
                &accepted,
                &state,
                &second_log,
                &second_binding,
                &admission,
            ),
            Err(V2ExternalLogTrustError::CandidateBindingMismatch)
        );

        let rotated = V2ExternalLogAdmissionCandidate::successor(&admission, &second_log).unwrap();
        let candidate = V2ExternalLogTrustCheckpoint::successor_candidate(
            &accepted,
            &state,
            &second_log,
            &second_binding,
            &rotated,
        )
        .unwrap();
        assert_eq!(candidate.admission_sequence(), 2);
        assert_ne!(candidate.log_identity_commitment(), first_log.commitment());
        assert_eq!(
            candidate.classify_against(&accepted),
            V2ExternalLogTrustDisposition::AheadRequiresVerifiedTransition
        );
    }

    #[test]
    fn checkpoint_rejects_cross_state_cross_log_and_cross_admission_pairing() {
        let (state, identity, binding, admission, _) = genesis_parts();
        let other_state = tuf(13, 21, 19, 16, 0x31);
        let other_log = log(0x50);
        let other_binding = V2TufLogBindingCandidate::candidate(&other_state, &other_log);
        let other_admission = V2ExternalLogAdmissionCandidate::genesis(&other_log);

        assert_eq!(
            V2ExternalLogTrustCheckpoint::genesis_candidate(
                &state,
                &identity,
                &other_binding,
                &admission,
            ),
            Err(V2ExternalLogTrustError::CandidateBindingMismatch)
        );
        assert_eq!(
            V2ExternalLogTrustCheckpoint::genesis_candidate(
                &state,
                &identity,
                &binding,
                &other_admission,
            ),
            Err(V2ExternalLogTrustError::CandidateBindingMismatch)
        );
    }

    #[test]
    fn high_water_comparison_separates_all_five_dispositions() {
        let (state, identity, binding, admission, accepted) = genesis_parts();
        assert_eq!(
            accepted.classify_against(&accepted),
            V2ExternalLogTrustDisposition::ExactCandidateState
        );

        let rollback_state = tuf(11, 19, 17, 14, 0x30);
        let rollback_binding = V2TufLogBindingCandidate::candidate(&rollback_state, &identity);
        let rollback = V2ExternalLogTrustCheckpoint::new(
            0,
            &rollback_state,
            &identity,
            &rollback_binding,
            &admission,
            None,
        );
        assert_eq!(
            rollback.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Rollback
        );

        let mut equivocation = accepted.clone();
        equivocation.current_root = metadata(12, 0xee);
        equivocation.commitment = trust_checkpoint_commitment(&equivocation);
        assert_eq!(
            equivocation.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Equivocation
        );

        let ahead_state = tuf(13, 21, 19, 16, 0x31);
        let ahead_binding = V2TufLogBindingCandidate::candidate(&ahead_state, &identity);
        let ahead = V2ExternalLogTrustCheckpoint::new(
            2,
            &ahead_state,
            &identity,
            &ahead_binding,
            &admission,
            Some(accepted.commitment()),
        );
        assert_eq!(
            ahead.classify_against(&accepted),
            V2ExternalLogTrustDisposition::AheadRequiresVerifiedTransition
        );

        let mixed_state = tuf(13, 19, 19, 16, 0x31);
        let mixed_binding = V2TufLogBindingCandidate::candidate(&mixed_state, &identity);
        let mixed = V2ExternalLogTrustCheckpoint::new(
            2,
            &mixed_state,
            &identity,
            &mixed_binding,
            &admission,
            Some(accepted.commitment()),
        );
        assert_eq!(
            mixed.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Incomparable
        );

        assert_eq!(state.commitment(), binding.tuf_state_commitment());
    }

    #[test]
    fn same_version_changed_metadata_or_target_identity_is_equivocation() {
        let (_, _, _, _, accepted) = genesis_parts();

        let mut metadata_changed = accepted.clone();
        metadata_changed.current_root = metadata(12, 0xee);
        metadata_changed.commitment = trust_checkpoint_commitment(&metadata_changed);
        assert_eq!(
            metadata_changed.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Equivocation
        );

        let mut target_changed = accepted.clone();
        target_changed.trusted_root_target_name = "alternate-trusted-root.json".to_owned();
        target_changed.commitment = trust_checkpoint_commitment(&target_changed);
        assert_eq!(
            target_changed.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Equivocation
        );
    }

    #[test]
    fn different_bootstrap_root_is_incomparable_even_at_higher_versions() {
        let (_, identity, _, admission, accepted) = genesis_parts();
        let foreign_state = V2TufTrustStateDescriptor::candidate(V2TufTrustStateCandidateInput {
            bootstrap_root_version: 10,
            bootstrap_root_sha256: digest(0x99),
            current_root: metadata(99, 0x98),
            timestamp: metadata(99, 0x97),
            snapshot: metadata(99, 0x96),
            targets: metadata(99, 0x95),
            trusted_root_target_name: "trusted_root.json".to_owned(),
            trusted_root_target_sha256: digest(0x30),
            signing_config_target_name: "signing_config.v2.json".to_owned(),
            signing_config_target_sha256: digest(0x31),
        })
        .unwrap();
        let foreign_binding = V2TufLogBindingCandidate::candidate(&foreign_state, &identity);
        let foreign = V2ExternalLogTrustCheckpoint::new(
            99,
            &foreign_state,
            &identity,
            &foreign_binding,
            &admission,
            None,
        );
        assert_eq!(
            foreign.classify_against(&accepted),
            V2ExternalLogTrustDisposition::Incomparable
        );
    }

    #[test]
    fn parser_rejects_authority_escalation_and_noncanonical_numbers() {
        let (state, _, _, _, _) = genesis_parts();
        let text = String::from_utf8(state.canonical_bytes()).unwrap();
        let escalated = text.replacen(
            "tuf_verification_complete=false",
            "tuf_verification_complete=true",
            1,
        );
        assert_eq!(
            V2TufTrustStateDescriptor::parse_canonical(escalated.as_bytes()),
            Err(V2ExternalLogTrustError::AuthorityEscalation)
        );
        let leading_zero = text.replacen("current_root_version=12", "current_root_version=012", 1);
        assert_eq!(
            V2TufTrustStateDescriptor::parse_canonical(leading_zero.as_bytes()),
            Err(V2ExternalLogTrustError::InvalidNumber)
        );
    }

    #[test]
    fn source_has_no_network_verifier_private_key_externality_or_execution_surface() {
        let source = include_str!("v2_external_log_trust_state.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let forbidden = [
            ["req", "west"].concat(),
            ["std::", "net"].concat(),
            ["tough", "::"].concat(),
            ["tuf", "::client"].concat(),
            ["Signing", "Key"].concat(),
            ["PRIVATE", " KEY"].concat(),
            ["TufVerified", "TransparencyLogIdentity"].concat(),
            ["EurekaAdmitted", "TransparencyLogIdentity"].concat(),
            ["ExternallyIncluded", "Publication"].concat(),
            ["tuf_verification_complete=", "true"].concat(),
            ["eureka_external_log_admitted=", "true"].concat(),
            ["externality_verified=", "true"].concat(),
            ["execution_authority_granted=", "true"].concat(),
            ["predict_", "ticket"].concat(),
            ["reveal", "("].concat(),
            ["score_", "consequence"].concat(),
        ];
        for forbidden in forbidden {
            assert!(
                !source.contains(&forbidden),
                "forbidden Phase A trust-state surface: {forbidden}"
            );
        }
    }
}
