// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lineage-bearing transitions for safety-profile authorization.
//!
//! [`crate::authorization::SafetyProfileAuthorizationSubject`] is the exact
//! claim about one profile authorization. This module adds the separate lineage
//! contract that must be authenticated for runtime admission: generation 1 is an
//! explicit bootstrap transition, while every later transition binds the digest
//! of its exact predecessor.
//!
//! This closes the equivocation seam left by generation counters alone. Two
//! conflicting generation-N claims can exist, but a generation-(N+1) transition
//! cannot be replayed over both branches because it names one exact predecessor
//! digest.

use crate::authorization::{
    ProfileAuthorityRootDigest, SafetyProfileAuthorizationError,
    SafetyProfileAuthorizationSubject,
};
use serde::{Deserialize, Serialize};
use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

pub const SAFETY_PROFILE_AUTHORIZATION_TRANSITION_SCHEMA_V1: &str =
    "symthaea-safety-profile-authorization-transition-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-profile-authorization-transition:v1\0";

/// Exact digest of one canonical authorization transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyProfileAuthorizationTransitionDigest {
    Blake3_256([u8; 32]),
}

impl SafetyProfileAuthorizationTransitionDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        let digest = ConfigurationDigest::blake3_256(bytes).into_blake3_256();
        Self::Blake3_256(digest)
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Exact lineage input for one authorization transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyProfileAuthorizationPredecessor {
    /// Generation 1 only. The authority root itself must already have been
    /// established by trusted provisioning or a separate root-transition policy.
    Bootstrap,
    /// Exact digest of the immediately preceding authorization transition.
    Previous(SafetyProfileAuthorizationTransitionDigest),
}

/// Canonical transition that must be authenticated before a profile authorization
/// can enter runtime state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyProfileAuthorizationTransition {
    schema_version: String,
    predecessor: SafetyProfileAuthorizationPredecessor,
    subject: SafetyProfileAuthorizationSubject,
}

impl SafetyProfileAuthorizationTransition {
    /// Build the only valid bootstrap transition: generation 1 under an already
    /// externally trusted authority root.
    pub fn bootstrap(
        subject: SafetyProfileAuthorizationSubject,
    ) -> Result<Self, SafetyProfileAuthorizationTransitionError> {
        subject.validate()?;
        if subject.generation() != 1 {
            return Err(
                SafetyProfileAuthorizationTransitionError::BootstrapGenerationMustBeOne {
                    observed: subject.generation(),
                },
            );
        }

        Ok(Self {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_TRANSITION_SCHEMA_V1.to_owned(),
            predecessor: SafetyProfileAuthorizationPredecessor::Bootstrap,
            subject,
        })
    }

    /// Build an exact successor transition.
    ///
    /// The successor must preserve the authorization chain's node and authority
    /// root, advance generation by exactly one, and bind the exact predecessor
    /// transition digest. Authority-root rotation is intentionally not expressible
    /// here; it requires a separate predecessor-authorized root transition.
    pub fn successor(
        subject: SafetyProfileAuthorizationSubject,
        predecessor: &SafetyProfileAuthorizationTransition,
    ) -> Result<Self, SafetyProfileAuthorizationTransitionError> {
        predecessor.validate()?;
        subject.validate()?;

        let expected_generation = predecessor
            .subject
            .generation()
            .checked_add(1)
            .ok_or(SafetyProfileAuthorizationTransitionError::GenerationExhausted {
                current: predecessor.subject.generation(),
            })?;

        if subject.generation() != expected_generation {
            return Err(
                SafetyProfileAuthorizationTransitionError::GenerationNotSuccessor {
                    current: predecessor.subject.generation(),
                    expected: expected_generation,
                    observed: subject.generation(),
                },
            );
        }
        if subject.subject_node_id() != predecessor.subject.subject_node_id() {
            return Err(SafetyProfileAuthorizationTransitionError::SubjectNodeChanged {
                predecessor: predecessor.subject.subject_node_id().to_owned(),
                successor: subject.subject_node_id().to_owned(),
            });
        }
        if subject.authority_root_id() != predecessor.subject.authority_root_id() {
            return Err(SafetyProfileAuthorizationTransitionError::AuthorityRootIdChanged {
                predecessor: predecessor.subject.authority_root_id().to_owned(),
                successor: subject.authority_root_id().to_owned(),
            });
        }
        if subject.authority_root_digest() != predecessor.subject.authority_root_digest() {
            return Err(SafetyProfileAuthorizationTransitionError::AuthorityRootChanged);
        }

        Ok(Self {
            schema_version: SAFETY_PROFILE_AUTHORIZATION_TRANSITION_SCHEMA_V1.to_owned(),
            predecessor: SafetyProfileAuthorizationPredecessor::Previous(
                predecessor.transition_digest()?,
            ),
            subject,
        })
    }

    pub fn validate(&self) -> Result<(), SafetyProfileAuthorizationTransitionError> {
        if self.schema_version != SAFETY_PROFILE_AUTHORIZATION_TRANSITION_SCHEMA_V1 {
            return Err(
                SafetyProfileAuthorizationTransitionError::UnsupportedSchemaVersion(
                    self.schema_version.clone(),
                ),
            );
        }
        self.subject.validate()?;

        match (self.subject.generation(), self.predecessor) {
            (1, SafetyProfileAuthorizationPredecessor::Bootstrap) => Ok(()),
            (1, SafetyProfileAuthorizationPredecessor::Previous(_)) => {
                Err(SafetyProfileAuthorizationTransitionError::GenerationOneHasPredecessor)
            }
            (_, SafetyProfileAuthorizationPredecessor::Bootstrap) => Err(
                SafetyProfileAuthorizationTransitionError::NonInitialGenerationUsesBootstrap {
                    generation: self.subject.generation(),
                },
            ),
            (_, SafetyProfileAuthorizationPredecessor::Previous(_)) => Ok(()),
        }
    }

    /// Fixed-order, domain-separated bytes that the external authority verifier
    /// must authenticate.
    ///
    /// V1 encoding:
    /// - transition domain separator,
    /// - transition schema as big-endian u32 length-prefixed UTF-8,
    /// - predecessor tag: `0` for Bootstrap, `1` for Previous,
    /// - for Previous only: digest algorithm tag `1` then 32 digest bytes,
    /// - complete canonical authorization-subject bytes as big-endian u32
    ///   length-prefixed opaque bytes.
    pub fn canonical_signing_bytes(
        &self,
    ) -> Result<Vec<u8>, SafetyProfileAuthorizationTransitionError> {
        self.validate()?;
        let subject_bytes = self.subject.canonical_signing_bytes()?;
        let mut out = Vec::with_capacity(384);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_bytes(
            &mut out,
            "schema_version",
            self.schema_version.as_bytes(),
        )?;
        push_predecessor(&mut out, self.predecessor);
        push_bytes(&mut out, "subject", &subject_bytes)?;
        Ok(out)
    }

    pub fn transition_digest(
        &self,
    ) -> Result<SafetyProfileAuthorizationTransitionDigest, SafetyProfileAuthorizationTransitionError>
    {
        Ok(SafetyProfileAuthorizationTransitionDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    pub fn predecessor(&self) -> SafetyProfileAuthorizationPredecessor {
        self.predecessor
    }

    pub fn subject(&self) -> &SafetyProfileAuthorizationSubject {
        &self.subject
    }

    pub fn generation(&self) -> u64 {
        self.subject.generation()
    }

    pub fn authority_root_digest(&self) -> ProfileAuthorityRootDigest {
        self.subject.authority_root_digest()
    }
}

fn push_bytes(
    out: &mut Vec<u8>,
    field: &'static str,
    bytes: &[u8],
) -> Result<(), SafetyProfileAuthorizationTransitionError> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| SafetyProfileAuthorizationTransitionError::FieldTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn push_predecessor(out: &mut Vec<u8>, predecessor: SafetyProfileAuthorizationPredecessor) {
    match predecessor {
        SafetyProfileAuthorizationPredecessor::Bootstrap => out.push(0),
        SafetyProfileAuthorizationPredecessor::Previous(digest) => {
            out.push(1);
            match digest {
                SafetyProfileAuthorizationTransitionDigest::Blake3_256(bytes) => {
                    out.push(1);
                    out.extend_from_slice(&bytes);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationTransitionError {
    #[error(transparent)]
    Subject(#[from] SafetyProfileAuthorizationError),
    #[error("unsupported safety-profile authorization transition schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("bootstrap safety-profile authorization generation must be 1, observed {observed}")]
    BootstrapGenerationMustBeOne { observed: u64 },
    #[error("generation 1 safety-profile authorization must not carry a predecessor")]
    GenerationOneHasPredecessor,
    #[error("safety-profile authorization generation {generation} cannot use bootstrap predecessor")]
    NonInitialGenerationUsesBootstrap { generation: u64 },
    #[error("safety-profile authorization generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error("safety-profile authorization generation must be exact successor of {current}: expected {expected}, observed {observed}")]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
    },
    #[error("safety-profile authorization subject node changed across transition: predecessor {predecessor}, successor {successor}")]
    SubjectNodeChanged {
        predecessor: String,
        successor: String,
    },
    #[error("safety-profile authorization authority-root id changed across transition: predecessor {predecessor}, successor {successor}")]
    AuthorityRootIdChanged {
        predecessor: String,
        successor: String,
    },
    #[error("safety-profile authorization authority root changed without a root-transition policy")]
    AuthorityRootChanged,
    #[error("safety-profile authorization transition field {0} exceeds canonical u32 length")]
    FieldTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::ProfileAuthorityRootDigest;
    use crate::{ComponentRequirement, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1, SafetyConfigurationProfile};

    fn root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "test-profile-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn subject(
        authorization_id: &str,
        generation: u64,
        root_digest: ProfileAuthorityRootDigest,
        node: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            authorization_id,
            "root-1",
            root_digest,
            node,
            generation,
            1_000,
            2_000,
            profile,
        )
        .unwrap()
    }

    #[test]
    fn bootstrap_is_generation_one_only() {
        let profile = profile();
        let first = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-1",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();
        assert_eq!(first.generation(), 1);
        assert_eq!(
            first.predecessor(),
            SafetyProfileAuthorizationPredecessor::Bootstrap
        );

        assert_eq!(
            SafetyProfileAuthorizationTransition::bootstrap(subject(
                "auth-2",
                2,
                root(0x33),
                "node",
                &profile,
            )),
            Err(
                SafetyProfileAuthorizationTransitionError::BootstrapGenerationMustBeOne {
                    observed: 2,
                }
            )
        );
    }

    #[test]
    fn successor_binds_exact_predecessor_digest() {
        let profile = profile();
        let first = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-1",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();
        let second = SafetyProfileAuthorizationTransition::successor(
            subject("auth-2", 2, root(0x33), "node", &profile),
            &first,
        )
        .unwrap();

        assert_eq!(
            second.predecessor(),
            SafetyProfileAuthorizationPredecessor::Previous(first.transition_digest().unwrap())
        );
        assert_ne!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn skipped_or_replayed_generation_is_rejected() {
        let profile = profile();
        let first = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-1",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();

        for observed in [1, 3] {
            assert!(matches!(
                SafetyProfileAuthorizationTransition::successor(
                    subject("candidate", observed, root(0x33), "node", &profile),
                    &first,
                ),
                Err(SafetyProfileAuthorizationTransitionError::GenerationNotSuccessor { .. })
            ));
        }
    }

    #[test]
    fn node_or_authority_root_cannot_change_inside_profile_chain() {
        let profile = profile();
        let first = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-1",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();

        assert!(matches!(
            SafetyProfileAuthorizationTransition::successor(
                subject("auth-2", 2, root(0x33), "other-node", &profile),
                &first,
            ),
            Err(SafetyProfileAuthorizationTransitionError::SubjectNodeChanged { .. })
        ));
        assert_eq!(
            SafetyProfileAuthorizationTransition::successor(
                subject("auth-2", 2, root(0x44), "node", &profile),
                &first,
            ),
            Err(SafetyProfileAuthorizationTransitionError::AuthorityRootChanged)
        );
    }

    #[test]
    fn conflicting_predecessors_produce_distinct_successor_lineages() {
        let profile = profile();
        let first_a = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-a",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();
        let first_b = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-b",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();
        assert_ne!(
            first_a.transition_digest().unwrap(),
            first_b.transition_digest().unwrap()
        );

        let candidate = subject("auth-2", 2, root(0x33), "node", &profile);
        let second_a = SafetyProfileAuthorizationTransition::successor(candidate.clone(), &first_a)
            .unwrap();
        let second_b = SafetyProfileAuthorizationTransition::successor(candidate, &first_b).unwrap();

        assert_ne!(second_a.predecessor(), second_b.predecessor());
        assert_ne!(
            second_a.canonical_signing_bytes().unwrap(),
            second_b.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn bootstrap_canonical_bytes_have_pinned_cross_tool_v1_vector() {
        let profile = profile();
        let transition = SafetyProfileAuthorizationTransition::bootstrap(subject(
            "auth-1",
            1,
            root(0x33),
            "node",
            &profile,
        ))
        .unwrap();

        let expected = hex_bytes(
            "73796d74686165613a7361666574792d70726f66696c652d617574686f72697a6174696f6e2d7472616e736974696f6e3a7631000000003373796d74686165612d7361666574792d70726f66696c652d617574686f72697a6174696f6e2d7472616e736974696f6e2d763100000000de73796d74686165613a7361666574792d70726f66696c652d617574686f72697a6174696f6e3a7631000000002873796d74686165612d7361666574792d70726f66696c652d617574686f72697a6174696f6e2d763100000006617574682d3100000006726f6f742d31000000046e6f6465000000000000000100000000000003e800000000000007d00000000f746573742d70726f66696c652d7631013333333333333333333333333333333333333333333333333333333333333333012222222222222222222222222222222222222222222222222222222222222222",
        );
        assert_eq!(transition.canonical_signing_bytes().unwrap(), expected);
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| (from_hex(pair[0]) << 4) | from_hex(pair[1]))
            .collect()
    }

    fn from_hex(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => panic!("invalid hex byte"),
        }
    }
}
