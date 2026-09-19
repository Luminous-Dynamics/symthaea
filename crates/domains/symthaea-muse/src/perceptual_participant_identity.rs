// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1DTR: participant pseudonym generation and schedule isolation.
//!
//! Historical P1D correctly freezes randomization over pseudonymous participant
//! tokens, but accepts arbitrary nonempty unique strings. This additive repair
//! defines the registered token-generator/output contract and a single-
//! participant schedule projection without rewriting P1D's randomization math.
//!
//! A participant token is an analytical pseudonym, not an authentication secret.
//! The full P1D schedule book is arm-semantically public evidence, not a
//! participant-visible cohort directory. Production claims that the registered
//! OS CSPRNG actually executed remain qualification/execution claims.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
        PublicParticipantScheduleV1, validate_perceptual_participant_schedule,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Write as _;

pub const PARTICIPANT_TOKEN_GENERATION_RECEIPT_VERSION: &str =
    "mel003-participant-token-generation-receipt-v1";
pub const PARTICIPANT_TOKEN_GENERATOR_VERSION: &str =
    "mel003-participant-token-generator-v1";
pub const PARTICIPANT_IDENTITY_BOUNDARY_POLICY_VERSION: &str =
    "mel003-participant-identity-boundary-policy-v1";
pub const PARTICIPANT_SCHEDULE_PROJECTION_VERSION: &str =
    "mel003-participant-schedule-projection-v1";
pub const PARTICIPANT_TOKEN_RANDOM_BYTES: usize = 16;
pub const PARTICIPANT_TOKEN_RANDOM_BITS: u16 = 128;
pub const PARTICIPANT_TOKEN_HEX_CHARS: usize = PARTICIPANT_TOKEN_RANDOM_BYTES * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantTokenRandomnessSourceV1 {
    /// The registered implementation requests token bytes from the operating
    /// system CSPRNG. Whether a production execution actually consumed that
    /// source is an execution/qualification fact, not proven by JSON shape.
    OperatingSystemCsprng,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantTokenEncodingV1 {
    LowerHex128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantTokenGeneratorIdentityV1 {
    pub source_revision: String,
    pub binary_sha256: String,
    pub environment_sha256: String,
    pub generator_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualParticipantTokenGenerationReceiptV1 {
    pub receipt_version: String,
    pub protocol_sha256: String,
    pub cohort_sha256: String,
    pub cohort_id: String,
    pub participant_count: usize,
    pub randomness_source: ParticipantTokenRandomnessSourceV1,
    pub random_bits_per_token: u16,
    pub token_encoding: ParticipantTokenEncodingV1,
    pub generator: ParticipantTokenGeneratorIdentityV1,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenParticipantIdentityBoundaryPolicyV1 {
    pub policy_version: String,
    pub token_generation_receipt_sha256: String,
    /// A stable analytical pseudonym may identify a record but must never be
    /// the sole bearer credential granting access to participant state.
    pub stable_participant_token_bearer_access_prohibited: bool,
    /// Client surfaces must receive one schedule projection, not the cohort book.
    pub full_schedule_book_client_exposure_prohibited: bool,
    pub cohort_enumeration_prohibited: bool,
    /// Names/contact details and their token linkage stay outside study evidence.
    pub recruitment_linkage_in_study_evidence_prohibited: bool,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenParticipantScheduleProjectionV1 {
    pub projection_version: String,
    pub schedule_book_sha256: String,
    pub token_generation_receipt_sha256: String,
    /// Exactly one participant's already-blinded P1D schedule.
    pub schedule: PublicParticipantScheduleV1,
    pub projection_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualParticipantIdentityIssueV1 {
    InvalidProtocol,
    InvalidParticipantSchedule,
    WrongReceiptVersion,
    WrongGeneratorVersion,
    WrongRandomnessSource,
    WrongRandomBitCount { found: u16, expected: u16 },
    WrongTokenEncoding,
    InvalidGeneratorDigest { field: String },
    EmptyGeneratorField { field: String },
    CohortCountMismatch { found: usize, expected: usize },
    EmptyCohortId,
    CohortIdMismatch,
    InvalidTokenFormat { index: usize },
    DuplicateParticipantToken { token: String },
    ProtocolDigestMismatch,
    CohortDigestMismatch,
    InvalidReceiptDigest,
    ReceiptDigestMismatch,
    WrongBoundaryPolicyVersion,
    BoundaryReceiptDigestMismatch,
    MissingBoundaryProtection { field: String },
    InvalidBoundaryPolicyDigest,
    BoundaryPolicyDigestMismatch,
    WrongProjectionVersion,
    ScheduleBookDigestMismatch,
    ProjectionReceiptDigestMismatch,
    ParticipantScheduleNotFound,
    DuplicateParticipantSchedule,
    ProjectedScheduleMismatch,
    InvalidProjectionDigest,
    ProjectionDigestMismatch,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TokenGenerationReceiptCommitment<'a> {
    receipt_version: &'a str,
    protocol_sha256: &'a str,
    cohort_sha256: &'a str,
    cohort_id: &'a str,
    participant_count: usize,
    randomness_source: ParticipantTokenRandomnessSourceV1,
    random_bits_per_token: u16,
    token_encoding: ParticipantTokenEncodingV1,
    generator: &'a ParticipantTokenGeneratorIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ParticipantIdentityBoundaryPolicyCommitment<'a> {
    policy_version: &'a str,
    token_generation_receipt_sha256: &'a str,
    stable_participant_token_bearer_access_prohibited: bool,
    full_schedule_book_client_exposure_prohibited: bool,
    cohort_enumeration_prohibited: bool,
    recruitment_linkage_in_study_evidence_prohibited: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ParticipantScheduleProjectionCommitment<'a> {
    projection_version: &'a str,
    schedule_book_sha256: &'a str,
    token_generation_receipt_sha256: &'a str,
    schedule: &'a PublicParticipantScheduleV1,
}

pub fn cohort_slots_commitment(
    cohort: &PerceptualCohortSlotsV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(cohort)
}

pub fn token_generation_receipt_commitment(
    receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&TokenGenerationReceiptCommitment {
        receipt_version: &receipt.receipt_version,
        protocol_sha256: &receipt.protocol_sha256,
        cohort_sha256: &receipt.cohort_sha256,
        cohort_id: &receipt.cohort_id,
        participant_count: receipt.participant_count,
        randomness_source: receipt.randomness_source,
        random_bits_per_token: receipt.random_bits_per_token,
        token_encoding: receipt.token_encoding,
        generator: &receipt.generator,
    })
}

pub fn identity_boundary_policy_commitment(
    policy: &FrozenParticipantIdentityBoundaryPolicyV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ParticipantIdentityBoundaryPolicyCommitment {
        policy_version: &policy.policy_version,
        token_generation_receipt_sha256: &policy.token_generation_receipt_sha256,
        stable_participant_token_bearer_access_prohibited: policy
            .stable_participant_token_bearer_access_prohibited,
        full_schedule_book_client_exposure_prohibited: policy
            .full_schedule_book_client_exposure_prohibited,
        cohort_enumeration_prohibited: policy.cohort_enumeration_prohibited,
        recruitment_linkage_in_study_evidence_prohibited: policy
            .recruitment_linkage_in_study_evidence_prohibited,
    })
}

pub fn participant_schedule_projection_commitment(
    projection: &FrozenParticipantScheduleProjectionV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ParticipantScheduleProjectionCommitment {
        projection_version: &projection.projection_version,
        schedule_book_sha256: &projection.schedule_book_sha256,
        token_generation_receipt_sha256: &projection.token_generation_receipt_sha256,
        schedule: &projection.schedule,
    })
}

/// Generate the complete maximum-enrollment pseudonym set with the OS CSPRNG.
///
/// The resulting receipt is a structural/source-side statement. A production
/// claim that this exact generator execution really used the registered entropy
/// source must be joined to executable qualification evidence.
pub fn generate_perceptual_cohort_slots_os_rng(
    protocol: &FrozenPerceptualStudyProtocolV1,
    cohort_id: impl Into<String>,
    generator: ParticipantTokenGeneratorIdentityV1,
) -> Result<
    (
        PerceptualCohortSlotsV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    ),
    Vec<PerceptualParticipantIdentityIssueV1>,
> {
    let cohort_id = cohort_id.into();
    if cohort_id.trim().is_empty() {
        return Err(vec![PerceptualParticipantIdentityIssueV1::EmptyCohortId]);
    }
    let generator_issues = validate_generator_identity(&generator);
    if !generator_issues.is_empty() {
        return Err(generator_issues);
    }
    if !protocol.validate().is_empty() {
        return Err(vec![PerceptualParticipantIdentityIssueV1::InvalidProtocol]);
    }

    let count = protocol.sample_size.maximum_enrolled_participants;
    let mut rng = OsRng;
    let mut tokens = BTreeSet::new();
    while tokens.len() < count {
        let mut bytes = [0u8; PARTICIPANT_TOKEN_RANDOM_BYTES];
        rng.fill_bytes(&mut bytes);
        tokens.insert(lower_hex(&bytes));
    }
    let cohort = PerceptualCohortSlotsV1 {
        cohort_id: cohort_id.clone(),
        participant_tokens: tokens.into_iter().collect(),
    };
    let protocol_sha256 = canonical_json_sha256(protocol)
        .map_err(|_| vec![PerceptualParticipantIdentityIssueV1::SerializationFailed])?;
    let cohort_sha256 = cohort_slots_commitment(&cohort)
        .map_err(|_| vec![PerceptualParticipantIdentityIssueV1::SerializationFailed])?;
    let mut receipt = FrozenPerceptualParticipantTokenGenerationReceiptV1 {
        receipt_version: PARTICIPANT_TOKEN_GENERATION_RECEIPT_VERSION.into(),
        protocol_sha256,
        cohort_sha256,
        cohort_id,
        participant_count: count,
        randomness_source: ParticipantTokenRandomnessSourceV1::OperatingSystemCsprng,
        random_bits_per_token: PARTICIPANT_TOKEN_RANDOM_BITS,
        token_encoding: ParticipantTokenEncodingV1::LowerHex128,
        generator,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = token_generation_receipt_commitment(&receipt)
        .map_err(|_| vec![PerceptualParticipantIdentityIssueV1::SerializationFailed])?;
    let issues = validate_participant_token_generation_receipt(protocol, &cohort, &receipt);
    if issues.is_empty() {
        Ok((cohort, receipt))
    } else {
        Err(issues)
    }
}

pub fn validate_participant_token_generation_receipt(
    protocol: &FrozenPerceptualStudyProtocolV1,
    cohort: &PerceptualCohortSlotsV1,
    receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
) -> Vec<PerceptualParticipantIdentityIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidProtocol);
    }
    if receipt.receipt_version != PARTICIPANT_TOKEN_GENERATION_RECEIPT_VERSION {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongReceiptVersion);
    }
    if receipt.generator.generator_version != PARTICIPANT_TOKEN_GENERATOR_VERSION {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongGeneratorVersion);
    }
    if receipt.randomness_source != ParticipantTokenRandomnessSourceV1::OperatingSystemCsprng {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongRandomnessSource);
    }
    if receipt.random_bits_per_token != PARTICIPANT_TOKEN_RANDOM_BITS {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongRandomBitCount {
            found: receipt.random_bits_per_token,
            expected: PARTICIPANT_TOKEN_RANDOM_BITS,
        });
    }
    if receipt.token_encoding != ParticipantTokenEncodingV1::LowerHex128 {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongTokenEncoding);
    }
    issues.extend(validate_generator_identity(&receipt.generator));

    let expected_count = protocol.sample_size.maximum_enrolled_participants;
    if cohort.participant_tokens.len() != expected_count {
        issues.push(PerceptualParticipantIdentityIssueV1::CohortCountMismatch {
            found: cohort.participant_tokens.len(),
            expected: expected_count,
        });
    }
    if receipt.participant_count != cohort.participant_tokens.len() {
        issues.push(PerceptualParticipantIdentityIssueV1::CohortCountMismatch {
            found: receipt.participant_count,
            expected: cohort.participant_tokens.len(),
        });
    }
    if cohort.cohort_id.trim().is_empty() || receipt.cohort_id.trim().is_empty() {
        issues.push(PerceptualParticipantIdentityIssueV1::EmptyCohortId);
    }
    if receipt.cohort_id != cohort.cohort_id {
        issues.push(PerceptualParticipantIdentityIssueV1::CohortIdMismatch);
    }

    let mut seen = BTreeSet::new();
    for (index, token) in cohort.participant_tokens.iter().enumerate() {
        if !is_canonical_participant_token(token) {
            issues.push(PerceptualParticipantIdentityIssueV1::InvalidTokenFormat { index });
        }
        if !seen.insert(token.as_str()) {
            issues.push(PerceptualParticipantIdentityIssueV1::DuplicateParticipantToken {
                token: token.clone(),
            });
        }
    }

    match canonical_json_sha256(protocol) {
        Ok(value) if value == receipt.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    match cohort_slots_commitment(cohort) {
        Ok(value) if value == receipt.cohort_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::CohortDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    if !is_sha256(&receipt.receipt_sha256) {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidReceiptDigest);
    }
    match token_generation_receipt_commitment(receipt) {
        Ok(value) if value == receipt.receipt_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::ReceiptDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    issues
}

pub fn seal_participant_identity_boundary_policy(
    policy: &mut FrozenParticipantIdentityBoundaryPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = identity_boundary_policy_commitment(policy)?;
    Ok(())
}

pub fn validate_participant_identity_boundary_policy(
    receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    policy: &FrozenParticipantIdentityBoundaryPolicyV1,
) -> Vec<PerceptualParticipantIdentityIssueV1> {
    let mut issues = Vec::new();
    if policy.policy_version != PARTICIPANT_IDENTITY_BOUNDARY_POLICY_VERSION {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongBoundaryPolicyVersion);
    }
    if policy.token_generation_receipt_sha256 != receipt.receipt_sha256 {
        issues.push(PerceptualParticipantIdentityIssueV1::BoundaryReceiptDigestMismatch);
    }
    for (field, protected) in [
        (
            "stable_participant_token_bearer_access_prohibited",
            policy.stable_participant_token_bearer_access_prohibited,
        ),
        (
            "full_schedule_book_client_exposure_prohibited",
            policy.full_schedule_book_client_exposure_prohibited,
        ),
        ("cohort_enumeration_prohibited", policy.cohort_enumeration_prohibited),
        (
            "recruitment_linkage_in_study_evidence_prohibited",
            policy.recruitment_linkage_in_study_evidence_prohibited,
        ),
    ] {
        if !protected {
            issues.push(PerceptualParticipantIdentityIssueV1::MissingBoundaryProtection {
                field: field.into(),
            });
        }
    }
    if !is_sha256(&policy.policy_sha256) {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidBoundaryPolicyDigest);
    }
    match identity_boundary_policy_commitment(policy) {
        Ok(value) if value == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::BoundaryPolicyDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    issues
}

#[allow(clippy::too_many_arguments)]
pub fn project_participant_schedule(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    boundary_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    book: &PerceptualParticipantScheduleBookV1,
    participant_token: &str,
) -> Result<FrozenParticipantScheduleProjectionV1, Vec<PerceptualParticipantIdentityIssueV1>> {
    let mut issues = validate_participant_token_generation_receipt(protocol, cohort, token_receipt);
    issues.extend(validate_participant_identity_boundary_policy(
        token_receipt,
        boundary_policy,
    ));
    if !validate_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        book,
        None,
    )
    .is_empty()
    {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidParticipantSchedule);
    }
    let mut matches = book
        .schedules
        .iter()
        .filter(|schedule| schedule.participant_token == participant_token);
    let Some(schedule) = matches.next() else {
        issues.push(PerceptualParticipantIdentityIssueV1::ParticipantScheduleNotFound);
        return Err(issues);
    };
    if matches.next().is_some() {
        issues.push(PerceptualParticipantIdentityIssueV1::DuplicateParticipantSchedule);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut projection = FrozenParticipantScheduleProjectionV1 {
        projection_version: PARTICIPANT_SCHEDULE_PROJECTION_VERSION.into(),
        schedule_book_sha256: canonical_json_sha256(book)
            .map_err(|_| vec![PerceptualParticipantIdentityIssueV1::SerializationFailed])?,
        token_generation_receipt_sha256: token_receipt.receipt_sha256.clone(),
        schedule: schedule.clone(),
        projection_sha256: String::new(),
    };
    projection.projection_sha256 = participant_schedule_projection_commitment(&projection)
        .map_err(|_| vec![PerceptualParticipantIdentityIssueV1::SerializationFailed])?;
    let validation = validate_participant_schedule_projection(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        boundary_policy,
        book,
        &projection,
    );
    if validation.is_empty() {
        Ok(projection)
    } else {
        Err(validation)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_participant_schedule_projection(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    boundary_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    book: &PerceptualParticipantScheduleBookV1,
    projection: &FrozenParticipantScheduleProjectionV1,
) -> Vec<PerceptualParticipantIdentityIssueV1> {
    let mut issues = validate_participant_token_generation_receipt(protocol, cohort, token_receipt);
    issues.extend(validate_participant_identity_boundary_policy(
        token_receipt,
        boundary_policy,
    ));
    if !validate_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        book,
        None,
    )
    .is_empty()
    {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidParticipantSchedule);
    }
    if projection.projection_version != PARTICIPANT_SCHEDULE_PROJECTION_VERSION {
        issues.push(PerceptualParticipantIdentityIssueV1::WrongProjectionVersion);
    }
    match canonical_json_sha256(book) {
        Ok(value) if value == projection.schedule_book_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::ScheduleBookDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    if projection.token_generation_receipt_sha256 != token_receipt.receipt_sha256 {
        issues.push(PerceptualParticipantIdentityIssueV1::ProjectionReceiptDigestMismatch);
    }

    let matches: Vec<_> = book
        .schedules
        .iter()
        .filter(|schedule| schedule.participant_token == projection.schedule.participant_token)
        .collect();
    match matches.as_slice() {
        [] => issues.push(PerceptualParticipantIdentityIssueV1::ParticipantScheduleNotFound),
        [expected] if **expected == projection.schedule => {}
        [_] => issues.push(PerceptualParticipantIdentityIssueV1::ProjectedScheduleMismatch),
        _ => issues.push(PerceptualParticipantIdentityIssueV1::DuplicateParticipantSchedule),
    }
    if !is_sha256(&projection.projection_sha256) {
        issues.push(PerceptualParticipantIdentityIssueV1::InvalidProjectionDigest);
    }
    match participant_schedule_projection_commitment(projection) {
        Ok(value) if value == projection.projection_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantIdentityIssueV1::ProjectionDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantIdentityIssueV1::SerializationFailed),
    }
    issues
}

fn validate_generator_identity(
    generator: &ParticipantTokenGeneratorIdentityV1,
) -> Vec<PerceptualParticipantIdentityIssueV1> {
    let mut issues = Vec::new();
    for (field, value) in [
        ("source_revision", generator.source_revision.as_str()),
        ("generator_version", generator.generator_version.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(PerceptualParticipantIdentityIssueV1::EmptyGeneratorField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("binary_sha256", generator.binary_sha256.as_str()),
        ("environment_sha256", generator.environment_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualParticipantIdentityIssueV1::InvalidGeneratorDigest {
                field: field.into(),
            });
        }
    }
    issues
}

fn is_canonical_participant_token(token: &str) -> bool {
    token.len() == PARTICIPANT_TOKEN_HEX_CHARS
        && token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn generator() -> ParticipantTokenGeneratorIdentityV1 {
        ParticipantTokenGeneratorIdentityV1 {
            source_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            binary_sha256: DIGEST_A.into(),
            environment_sha256: DIGEST_B.into(),
            generator_version: PARTICIPANT_TOKEN_GENERATOR_VERSION.into(),
        }
    }

    #[test]
    fn canonical_token_shape_rejects_predictable_labels() {
        assert!(!is_canonical_participant_token("participant-001"));
        assert!(!is_canonical_participant_token(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ));
        assert!(is_canonical_participant_token(
            "0123456789abcdef0123456789abcdef"
        ));
    }

    #[test]
    fn cohort_commitment_binds_exact_token_set() {
        let left = PerceptualCohortSlotsV1 {
            cohort_id: "cohort-a".into(),
            participant_tokens: vec!["0123456789abcdef0123456789abcdef".into()],
        };
        let mut right = left.clone();
        right.participant_tokens[0] = "fedcba9876543210fedcba9876543210".into();
        assert_ne!(
            cohort_slots_commitment(&left).unwrap(),
            cohort_slots_commitment(&right).unwrap()
        );
    }

    #[test]
    fn boundary_policy_rejects_participant_token_as_bearer_credential() {
        let receipt = FrozenPerceptualParticipantTokenGenerationReceiptV1 {
            receipt_version: PARTICIPANT_TOKEN_GENERATION_RECEIPT_VERSION.into(),
            protocol_sha256: DIGEST_A.into(),
            cohort_sha256: DIGEST_B.into(),
            cohort_id: "cohort-a".into(),
            participant_count: 1,
            randomness_source: ParticipantTokenRandomnessSourceV1::OperatingSystemCsprng,
            random_bits_per_token: PARTICIPANT_TOKEN_RANDOM_BITS,
            token_encoding: ParticipantTokenEncodingV1::LowerHex128,
            generator: generator(),
            receipt_sha256: DIGEST_A.into(),
        };
        let mut policy = FrozenParticipantIdentityBoundaryPolicyV1 {
            policy_version: PARTICIPANT_IDENTITY_BOUNDARY_POLICY_VERSION.into(),
            token_generation_receipt_sha256: receipt.receipt_sha256.clone(),
            stable_participant_token_bearer_access_prohibited: false,
            full_schedule_book_client_exposure_prohibited: true,
            cohort_enumeration_prohibited: true,
            recruitment_linkage_in_study_evidence_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_participant_identity_boundary_policy(&mut policy).unwrap();
        let issues = validate_participant_identity_boundary_policy(&receipt, &policy);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualParticipantIdentityIssueV1::MissingBoundaryProtection { field }
                if field == "stable_participant_token_bearer_access_prohibited"
        )));
    }

    #[test]
    fn generator_identity_requires_registered_version() {
        let mut identity = generator();
        assert!(validate_generator_identity(&identity).is_empty());
        identity.generator_version.clear();
        assert!(!validate_generator_identity(&identity).is_empty());
    }
}
