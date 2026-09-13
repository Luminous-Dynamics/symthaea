// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Second-stage witness diversity verification.
//!
//! The base continuity verifier proves that distinct Holochain agent keys observed the
//! exact Xenia checkpoint retained by Mycelix. This module deliberately asks a different
//! question: do enough of those keys belong to independently governed/control domains?
//!
//! Domain bindings are accepted only from a deployment-specific resolver that has already
//! verified the relevant Mycelix verifiable credential, issuer policy, revocation state,
//! subject binding, schema/type and validity window. Symthaea then re-checks the compact
//! verified view and counts canonical `control_domain_id`s rather than self-declared labels.

use std::collections::BTreeSet;
use std::error::Error as StdError;

use thiserror::Error;

use crate::IndependentlyWitnessedCheckpoint;

/// Interoperability schema expected from Mycelix Identity's generic VC machinery.
pub const WITNESS_DOMAIN_CREDENTIAL_SCHEMA_ID: &str =
    "mycelix:schema:governance:continuity-witness-domain:v1";
/// Required W3C VC type in addition to `VerifiableCredential`.
pub const WITNESS_DOMAIN_CREDENTIAL_TYPE: &str = "ContinuityWitnessDomainCredential";
/// Purpose bound into the verified credential/presentation.
pub const WITNESS_DOMAIN_PURPOSE: &str =
    "symthaea.episodic-continuity.external-witness.v1";

const MAX_TEXT_BYTES: usize = 1024;
const MAX_WITNESSES: usize = 4096;

/// Explicit policy for administrative diversity among already-distinct witness keys.
///
/// There are intentionally no defaults. A deployment must choose its own thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessDiversityPolicy {
    /// Minimum number of distinct canonical administrative/control roots.
    pub minimum_distinct_control_domains: u32,
    /// Minimum number of distinct trusted credential issuers represented among counted bindings.
    pub minimum_distinct_credential_issuers: u32,
    /// If true, every base witness key must have a valid domain binding.
    pub require_every_witness_bound: bool,
}

impl WitnessDiversityPolicy {
    pub fn validate(self) -> Result<(), WitnessDiversityPolicyError> {
        if self.minimum_distinct_control_domains == 0 {
            return Err(WitnessDiversityPolicyError::ZeroControlDomainThreshold);
        }
        if self.minimum_distinct_credential_issuers == 0 {
            return Err(WitnessDiversityPolicyError::ZeroIssuerThreshold);
        }
        if self.minimum_distinct_control_domains as usize > MAX_WITNESSES {
            return Err(WitnessDiversityPolicyError::ControlDomainThresholdTooLarge {
                actual: self.minimum_distinct_control_domains,
                max: MAX_WITNESSES as u32,
            });
        }
        if self.minimum_distinct_credential_issuers as usize > MAX_WITNESSES {
            return Err(WitnessDiversityPolicyError::IssuerThresholdTooLarge {
                actual: self.minimum_distinct_credential_issuers,
                max: MAX_WITNESSES as u32,
            });
        }
        Ok(())
    }
}

/// Compact output of a fully verified Mycelix witness-domain credential.
///
/// `domain_id` may identify a council, organization, jurisdiction, notary group, hardware
/// administrative domain, etc. `control_domain_id` MUST identify the canonical common control
/// root used for independence counting. Two differently named domains under one operator/admin
/// root therefore share the same `control_domain_id` and count once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedWitnessDomainBinding {
    pub witness_id: String,
    pub subject_did: String,
    pub domain_id: String,
    pub control_domain_id: String,
    pub credential_id: String,
    pub issuer_did: String,
    pub schema_id: String,
    pub credential_type: String,
    pub purpose: String,
    pub valid_from_unix_secs: u64,
    pub valid_until_unix_secs: Option<u64>,
    /// When revocation/status was checked by the resolver.
    pub revocation_checked_at_unix_secs: u64,
}

/// Boundary to Mycelix Identity / external VC verification.
///
/// Implementations MUST, before returning `Some(binding)`:
/// - cryptographically verify the credential or presentation;
/// - enforce trusted issuer policy;
/// - verify credential status/revocation;
/// - require the exact witness DID as subject;
/// - require [`WITNESS_DOMAIN_CREDENTIAL_SCHEMA_ID`],
///   [`WITNESS_DOMAIN_CREDENTIAL_TYPE`] and [`WITNESS_DOMAIN_PURPOSE`];
/// - verify the domain/control-root claims under the issuer's policy;
/// - validate the credential at `checkpoint_timestamp_unix_secs`.
///
/// `None` means no acceptable binding exists for that witness. It is not equivalent to error.
pub trait VerifiedWitnessDomainResolver {
    type Error: StdError + Send + Sync + 'static;

    fn resolve_verified_domain(
        &self,
        witness_id: &str,
        expected_subject_did: &str,
        checkpoint_timestamp_unix_secs: u64,
    ) -> Result<Option<VerifiedWitnessDomainBinding>, Self::Error>;
}

/// Evidence that the checkpoint passed both distinct-key and administrative-diversity policy.
///
/// This remains evidence only. It does not grant capability, consent, governance authority or
/// execution permission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainDiverseWitnessedCheckpoint {
    pub checkpoint: IndependentlyWitnessedCheckpoint,
    pub verified_bindings: Vec<VerifiedWitnessDomainBinding>,
    pub distinct_control_domain_ids: Vec<String>,
    pub distinct_credential_issuer_dids: Vec<String>,
    pub unbound_witness_ids: Vec<String>,
}

/// Apply verified administrative-domain diversity to already-authenticated checkpoint evidence.
pub fn verify_witness_domain_diversity<R>(
    checkpoint: &IndependentlyWitnessedCheckpoint,
    policy: WitnessDiversityPolicy,
    resolver: &R,
) -> Result<DomainDiverseWitnessedCheckpoint, WitnessDiversityError<R::Error>>
where
    R: VerifiedWitnessDomainResolver,
{
    policy.validate()?;
    if checkpoint.revision == 0 {
        return Err(WitnessDiversityError::ZeroRevision);
    }
    if checkpoint.distinct_witness_ids.is_empty() {
        return Err(WitnessDiversityError::NoWitnesses);
    }
    if checkpoint.distinct_witness_ids.len() > MAX_WITNESSES {
        return Err(WitnessDiversityError::TooManyWitnesses {
            actual: checkpoint.distinct_witness_ids.len(),
            max: MAX_WITNESSES,
        });
    }

    let mut seen_witnesses = BTreeSet::new();
    let mut verified_bindings = Vec::new();
    let mut control_domains = BTreeSet::new();
    let mut issuers = BTreeSet::new();
    let mut unbound = Vec::new();

    for witness_id in &checkpoint.distinct_witness_ids {
        validate_text("witness_id", witness_id)?;
        if !seen_witnesses.insert(witness_id.clone()) {
            return Err(WitnessDiversityError::DuplicateBaseWitness {
                witness_id: witness_id.clone(),
            });
        }

        let expected_subject_did = format!("did:mycelix:{witness_id}");
        let binding = resolver
            .resolve_verified_domain(
                witness_id,
                &expected_subject_did,
                checkpoint.xenia_checkpoint_timestamp_unix_secs,
            )
            .map_err(WitnessDiversityError::Resolver)?;

        let Some(binding) = binding else {
            if policy.require_every_witness_bound {
                return Err(WitnessDiversityError::UnboundWitness {
                    witness_id: witness_id.clone(),
                });
            }
            unbound.push(witness_id.clone());
            continue;
        };

        validate_binding(
            witness_id,
            &expected_subject_did,
            checkpoint.xenia_checkpoint_timestamp_unix_secs,
            &binding,
        )?;
        control_domains.insert(binding.control_domain_id.clone());
        issuers.insert(binding.issuer_did.clone());
        verified_bindings.push(binding);
    }

    if control_domains.len() < policy.minimum_distinct_control_domains as usize {
        return Err(WitnessDiversityError::InsufficientControlDomains {
            required: policy.minimum_distinct_control_domains,
            actual: control_domains.len() as u32,
        });
    }
    if issuers.len() < policy.minimum_distinct_credential_issuers as usize {
        return Err(WitnessDiversityError::InsufficientCredentialIssuers {
            required: policy.minimum_distinct_credential_issuers,
            actual: issuers.len() as u32,
        });
    }

    Ok(DomainDiverseWitnessedCheckpoint {
        checkpoint: checkpoint.clone(),
        verified_bindings,
        distinct_control_domain_ids: control_domains.into_iter().collect(),
        distinct_credential_issuer_dids: issuers.into_iter().collect(),
        unbound_witness_ids: unbound,
    })
}

fn validate_binding<E>(
    expected_witness_id: &str,
    expected_subject_did: &str,
    checkpoint_timestamp_unix_secs: u64,
    binding: &VerifiedWitnessDomainBinding,
) -> Result<(), WitnessDiversityError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    for (field, value) in [
        ("binding.witness_id", binding.witness_id.as_str()),
        ("binding.subject_did", binding.subject_did.as_str()),
        ("binding.domain_id", binding.domain_id.as_str()),
        ("binding.control_domain_id", binding.control_domain_id.as_str()),
        ("binding.credential_id", binding.credential_id.as_str()),
        ("binding.issuer_did", binding.issuer_did.as_str()),
        ("binding.schema_id", binding.schema_id.as_str()),
        ("binding.credential_type", binding.credential_type.as_str()),
        ("binding.purpose", binding.purpose.as_str()),
    ] {
        validate_text(field, value)?;
    }

    if binding.witness_id != expected_witness_id {
        return Err(WitnessDiversityError::WitnessBindingMismatch);
    }
    if binding.subject_did != expected_subject_did {
        return Err(WitnessDiversityError::SubjectDidMismatch);
    }
    if binding.schema_id != WITNESS_DOMAIN_CREDENTIAL_SCHEMA_ID {
        return Err(WitnessDiversityError::CredentialSchemaMismatch);
    }
    if binding.credential_type != WITNESS_DOMAIN_CREDENTIAL_TYPE {
        return Err(WitnessDiversityError::CredentialTypeMismatch);
    }
    if binding.purpose != WITNESS_DOMAIN_PURPOSE {
        return Err(WitnessDiversityError::CredentialPurposeMismatch);
    }
    if !binding.issuer_did.starts_with("did:") {
        return Err(WitnessDiversityError::InvalidIssuerDid);
    }
    if binding.valid_from_unix_secs > checkpoint_timestamp_unix_secs {
        return Err(WitnessDiversityError::CredentialNotYetValid);
    }
    if let Some(valid_until) = binding.valid_until_unix_secs {
        if valid_until < checkpoint_timestamp_unix_secs {
            return Err(WitnessDiversityError::CredentialExpired);
        }
        if valid_until < binding.valid_from_unix_secs {
            return Err(WitnessDiversityError::InvalidCredentialValidityWindow);
        }
    }
    if binding.revocation_checked_at_unix_secs < checkpoint_timestamp_unix_secs {
        return Err(WitnessDiversityError::StaleRevocationCheck);
    }
    Ok(())
}

fn validate_text<E>(field: &'static str, value: &str) -> Result<(), WitnessDiversityError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        Err(WitnessDiversityError::InvalidText { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WitnessDiversityPolicyError {
    #[error("witness control-domain threshold must be non-zero")]
    ZeroControlDomainThreshold,
    #[error("witness credential-issuer threshold must be non-zero")]
    ZeroIssuerThreshold,
    #[error("control-domain threshold too large: actual={actual}, max={max}")]
    ControlDomainThresholdTooLarge { actual: u32, max: u32 },
    #[error("credential-issuer threshold too large: actual={actual}, max={max}")]
    IssuerThresholdTooLarge { actual: u32, max: u32 },
}

#[derive(Debug, Error)]
pub enum WitnessDiversityError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Policy(#[from] WitnessDiversityPolicyError),
    #[error("invalid witness-diversity text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("checkpoint revision must be non-zero")]
    ZeroRevision,
    #[error("checkpoint contains no distinct witness identities")]
    NoWitnesses,
    #[error("too many checkpoint witnesses: actual={actual}, max={max}")]
    TooManyWitnesses { actual: usize, max: usize },
    #[error("base checkpoint repeated witness identity `{witness_id}`")]
    DuplicateBaseWitness { witness_id: String },
    #[error("witness-domain resolver failed: {0}")]
    Resolver(E),
    #[error("witness `{witness_id}` has no acceptable verified domain binding")]
    UnboundWitness { witness_id: String },
    #[error("verified witness-domain binding identifies a different witness")]
    WitnessBindingMismatch,
    #[error("verified witness-domain binding subject DID does not match witness key")]
    SubjectDidMismatch,
    #[error("verified witness-domain credential schema mismatch")]
    CredentialSchemaMismatch,
    #[error("verified witness-domain credential type mismatch")]
    CredentialTypeMismatch,
    #[error("verified witness-domain credential purpose mismatch")]
    CredentialPurposeMismatch,
    #[error("verified witness-domain credential issuer is not a DID")]
    InvalidIssuerDid,
    #[error("verified witness-domain credential was not yet valid at checkpoint time")]
    CredentialNotYetValid,
    #[error("verified witness-domain credential expired before checkpoint time")]
    CredentialExpired,
    #[error("verified witness-domain credential has an invalid validity window")]
    InvalidCredentialValidityWindow,
    #[error("credential revocation/status check predates checkpoint time")]
    StaleRevocationCheck,
    #[error("insufficient independent control domains: required={required}, actual={actual}")]
    InsufficientControlDomains { required: u32, actual: u32 },
    #[error("insufficient distinct credential issuers: required={required}, actual={actual}")]
    InsufficientCredentialIssuers { required: u32, actual: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndependentlyWitnessedCheckpoint;
    use std::collections::BTreeMap;
    use std::fmt;
    use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MockError;

    impl fmt::Display for MockError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "mock resolver error")
        }
    }

    impl StdError for MockError {}

    struct MockResolver {
        values: BTreeMap<String, VerifiedWitnessDomainBinding>,
        fail_for: Option<String>,
    }

    impl VerifiedWitnessDomainResolver for MockResolver {
        type Error = MockError;

        fn resolve_verified_domain(
            &self,
            witness_id: &str,
            _expected_subject_did: &str,
            _checkpoint_timestamp_unix_secs: u64,
        ) -> Result<Option<VerifiedWitnessDomainBinding>, Self::Error> {
            if self.fail_for.as_deref() == Some(witness_id) {
                return Err(MockError);
            }
            Ok(self.values.get(witness_id).cloned())
        }
    }

    fn checkpoint(witnesses: &[&str]) -> IndependentlyWitnessedCheckpoint {
        IndependentlyWitnessedCheckpoint {
            object_id: "symthaea:self:episodic-memory".into(),
            revision: 7,
            object_fingerprint: [0x11; 32],
            anchor_fingerprint: [0x22; 32],
            checkpoint_digest: Sha256Digest([0x33; 32]),
            checkpoint_action_ref: "uhCkk-checkpoint".into(),
            distinct_witness_ids: witnesses.iter().map(|v| (*v).to_string()).collect(),
            xenia_checkpoint_timestamp_unix_secs: 1_800_000_000,
        }
    }

    fn binding(witness: &str, domain: &str, root: &str, issuer: &str) -> VerifiedWitnessDomainBinding {
        VerifiedWitnessDomainBinding {
            witness_id: witness.into(),
            subject_did: format!("did:mycelix:{witness}"),
            domain_id: domain.into(),
            control_domain_id: root.into(),
            credential_id: format!("urn:mycelix:credential:{witness}"),
            issuer_did: issuer.into(),
            schema_id: WITNESS_DOMAIN_CREDENTIAL_SCHEMA_ID.into(),
            credential_type: WITNESS_DOMAIN_CREDENTIAL_TYPE.into(),
            purpose: WITNESS_DOMAIN_PURPOSE.into(),
            valid_from_unix_secs: 1_700_000_000,
            valid_until_unix_secs: Some(1_900_000_000),
            revocation_checked_at_unix_secs: 1_800_000_100,
        }
    }

    fn policy(domains: u32, issuers: u32) -> WitnessDiversityPolicy {
        WitnessDiversityPolicy {
            minimum_distinct_control_domains: domains,
            minimum_distinct_credential_issuers: issuers,
            require_every_witness_bound: true,
        }
    }

    #[test]
    fn distinct_control_roots_pass() {
        let cp = checkpoint(&["uhCAk-a", "uhCAk-b"]);
        let resolver = MockResolver {
            values: BTreeMap::from([
                (
                    "uhCAk-a".into(),
                    binding("uhCAk-a", "council:technical", "operator:alpha", "did:issuer:a"),
                ),
                (
                    "uhCAk-b".into(),
                    binding("uhCAk-b", "org:archive", "operator:beta", "did:issuer:b"),
                ),
            ]),
            fail_for: None,
        };
        let out = verify_witness_domain_diversity(&cp, policy(2, 2), &resolver).unwrap();
        assert_eq!(out.distinct_control_domain_ids.len(), 2);
        assert_eq!(out.distinct_credential_issuer_dids.len(), 2);
    }

    #[test]
    fn different_labels_under_same_control_root_count_once() {
        let cp = checkpoint(&["uhCAk-a", "uhCAk-b"]);
        let resolver = MockResolver {
            values: BTreeMap::from([
                (
                    "uhCAk-a".into(),
                    binding("uhCAk-a", "council:technical", "operator:alpha", "did:issuer:a"),
                ),
                (
                    "uhCAk-b".into(),
                    binding("uhCAk-b", "org:archive", "operator:alpha", "did:issuer:b"),
                ),
            ]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(2, 1), &resolver),
            Err(WitnessDiversityError::InsufficientControlDomains { actual: 1, .. })
        ));
    }

    #[test]
    fn issuer_diversity_is_independent_policy_axis() {
        let cp = checkpoint(&["uhCAk-a", "uhCAk-b"]);
        let resolver = MockResolver {
            values: BTreeMap::from([
                (
                    "uhCAk-a".into(),
                    binding("uhCAk-a", "domain:a", "root:a", "did:issuer:shared"),
                ),
                (
                    "uhCAk-b".into(),
                    binding("uhCAk-b", "domain:b", "root:b", "did:issuer:shared"),
                ),
            ]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(2, 2), &resolver),
            Err(WitnessDiversityError::InsufficientCredentialIssuers { actual: 1, .. })
        ));
    }

    #[test]
    fn unbound_witness_fails_when_policy_requires_complete_binding() {
        let cp = checkpoint(&["uhCAk-a", "uhCAk-b"]);
        let resolver = MockResolver {
            values: BTreeMap::from([(
                "uhCAk-a".into(),
                binding("uhCAk-a", "domain:a", "root:a", "did:issuer:a"),
            )]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(1, 1), &resolver),
            Err(WitnessDiversityError::UnboundWitness { .. })
        ));
    }

    #[test]
    fn optional_unbound_witness_is_audited_but_not_counted() {
        let cp = checkpoint(&["uhCAk-a", "uhCAk-b"]);
        let resolver = MockResolver {
            values: BTreeMap::from([(
                "uhCAk-a".into(),
                binding("uhCAk-a", "domain:a", "root:a", "did:issuer:a"),
            )]),
            fail_for: None,
        };
        let mut p = policy(1, 1);
        p.require_every_witness_bound = false;
        let out = verify_witness_domain_diversity(&cp, p, &resolver).unwrap();
        assert_eq!(out.unbound_witness_ids, vec!["uhCAk-b".to_string()]);
        assert_eq!(out.distinct_control_domain_ids, vec!["root:a".to_string()]);
    }

    #[test]
    fn subject_did_substitution_is_rejected() {
        let cp = checkpoint(&["uhCAk-a"]);
        let mut value = binding("uhCAk-a", "domain:a", "root:a", "did:issuer:a");
        value.subject_did = "did:mycelix:someone-else".into();
        let resolver = MockResolver {
            values: BTreeMap::from([("uhCAk-a".into(), value)]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(1, 1), &resolver),
            Err(WitnessDiversityError::SubjectDidMismatch)
        ));
    }

    #[test]
    fn expired_credential_is_rejected() {
        let cp = checkpoint(&["uhCAk-a"]);
        let mut value = binding("uhCAk-a", "domain:a", "root:a", "did:issuer:a");
        value.valid_until_unix_secs = Some(1_799_999_999);
        let resolver = MockResolver {
            values: BTreeMap::from([("uhCAk-a".into(), value)]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(1, 1), &resolver),
            Err(WitnessDiversityError::CredentialExpired)
        ));
    }

    #[test]
    fn stale_revocation_check_is_rejected() {
        let cp = checkpoint(&["uhCAk-a"]);
        let mut value = binding("uhCAk-a", "domain:a", "root:a", "did:issuer:a");
        value.revocation_checked_at_unix_secs = 1_799_999_999;
        let resolver = MockResolver {
            values: BTreeMap::from([("uhCAk-a".into(), value)]),
            fail_for: None,
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(1, 1), &resolver),
            Err(WitnessDiversityError::StaleRevocationCheck)
        ));
    }

    #[test]
    fn resolver_failure_is_fatal() {
        let cp = checkpoint(&["uhCAk-a"]);
        let resolver = MockResolver {
            values: BTreeMap::new(),
            fail_for: Some("uhCAk-a".into()),
        };
        assert!(matches!(
            verify_witness_domain_diversity(&cp, policy(1, 1), &resolver),
            Err(WitnessDiversityError::Resolver(MockError))
        ));
    }
}
