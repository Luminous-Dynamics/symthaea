//! Canonical persistence encoding and checked rehydration for proposed occurrences.
//!
//! Persisted bytes are historical identity material only. Decoding never proves that an
//! occurrence committed, remains current, or carries scientific/action authority.

use super::{
    COMMIT_OPERATION_DOMAIN, ControlPlaneCommitOperationIdV1, ControlPlaneOccurrenceStoreBindingV1,
    OCCURRENCE_DOMAIN, ProposedControlPlaneOccurrenceV1, hash_with, put_commitment,
    put_optional_commitment, put_text, put_u64,
};
use symthaea_scientific_view_profile::Commitment32;
use thiserror::Error;

pub const OCCURRENCE_WIRE_DOMAIN_V1: &[u8] =
    b"symthaea.science.view.control-plane.occurrence-wire.v1";

impl ProposedControlPlaneOccurrenceV1 {
    /// Canonical language-neutral V1 persistence bytes.
    ///
    /// This is an identity/replay representation only. Persisting these bytes does not prove
    /// that the occurrence committed or is current.
    pub fn to_canonical_bytes_v1(&self) -> Vec<u8> {
        let predecessor_len = if self.predecessor_occurrence.is_some() { 32 } else { 0 };
        let mut bytes = Vec::with_capacity(8 + OCCURRENCE_WIRE_DOMAIN_V1.len() + 137 + predecessor_len);
        bytes.extend_from_slice(&(OCCURRENCE_WIRE_DOMAIN_V1.len() as u64).to_be_bytes());
        bytes.extend_from_slice(OCCURRENCE_WIRE_DOMAIN_V1);
        bytes.extend_from_slice(self.store_binding_commitment.as_bytes());
        bytes.extend_from_slice(&self.sequence.to_be_bytes());
        match self.predecessor_occurrence {
            None => bytes.push(0),
            Some(predecessor) => {
                bytes.push(1);
                bytes.extend_from_slice(predecessor.as_bytes());
            }
        }
        bytes.extend_from_slice(self.candidate_transition_commitment.as_bytes());
        bytes.extend_from_slice(self.operation_id.commitment().as_bytes());
        bytes.extend_from_slice(self.commitment.as_bytes());
        bytes
    }

    /// Rehydrate exact V1 persistence bytes only after re-deriving every identity under the
    /// caller's expected occurrence-store binding.
    pub fn from_canonical_bytes_v1_checked(
        expected_store_binding: &ControlPlaneOccurrenceStoreBindingV1,
        bytes: &[u8],
    ) -> Result<Self, ControlPlaneOccurrenceWireError> {
        let mut cursor = WireCursor::new(bytes);

        let domain_len = cursor.read_u64()?;
        let expected_domain_len = u64::try_from(OCCURRENCE_WIRE_DOMAIN_V1.len())
            .map_err(|_| ControlPlaneOccurrenceWireError::LengthOverflow)?;
        if domain_len != expected_domain_len {
            return Err(ControlPlaneOccurrenceWireError::UnsupportedDomain);
        }
        if cursor.read_exact(OCCURRENCE_WIRE_DOMAIN_V1.len())? != OCCURRENCE_WIRE_DOMAIN_V1 {
            return Err(ControlPlaneOccurrenceWireError::UnsupportedDomain);
        }

        let encoded_store_binding = cursor.read_commitment()?;
        if encoded_store_binding != expected_store_binding.commitment() {
            return Err(ControlPlaneOccurrenceWireError::StoreBindingMismatch);
        }

        let sequence = cursor.read_u64()?;
        if sequence == 0 {
            return Err(ControlPlaneOccurrenceWireError::SequenceZero);
        }

        let predecessor_occurrence = match cursor.read_u8()? {
            0 => None,
            1 => Some(cursor.read_commitment()?),
            value => {
                return Err(ControlPlaneOccurrenceWireError::InvalidPredecessorDiscriminator(
                    value,
                ));
            }
        };

        match (sequence, predecessor_occurrence) {
            (1, None) => {}
            (1, Some(_)) | (_, None) => {
                return Err(ControlPlaneOccurrenceWireError::InvalidOccurrenceShape);
            }
            _ => {}
        }

        let candidate_transition_commitment = cursor.read_commitment()?;
        let encoded_operation_id = cursor.read_commitment()?;
        let encoded_occurrence_commitment = cursor.read_commitment()?;

        if !cursor.is_finished() {
            return Err(ControlPlaneOccurrenceWireError::TrailingBytes);
        }

        let recomputed_operation_id = hash_with(COMMIT_OPERATION_DOMAIN, |hasher| {
            put_text(hasher, expected_store_binding.deployment_id());
            put_text(hasher, expected_store_binding.view_namespace());
            put_commitment(hasher, expected_store_binding.commitment());
            put_optional_commitment(hasher, predecessor_occurrence);
            put_commitment(hasher, candidate_transition_commitment);
            put_commitment(
                hasher,
                expected_store_binding.persistence_profile_commitment(),
            );
        });
        if recomputed_operation_id != encoded_operation_id {
            return Err(ControlPlaneOccurrenceWireError::OperationIdMismatch);
        }

        let recomputed_occurrence_commitment = hash_with(OCCURRENCE_DOMAIN, |hasher| {
            put_commitment(hasher, expected_store_binding.commitment());
            put_u64(hasher, sequence);
            put_optional_commitment(hasher, predecessor_occurrence);
            put_commitment(hasher, candidate_transition_commitment);
            put_commitment(hasher, recomputed_operation_id);
        });
        if recomputed_occurrence_commitment != encoded_occurrence_commitment {
            return Err(ControlPlaneOccurrenceWireError::OccurrenceCommitmentMismatch);
        }

        Ok(Self {
            store_binding_commitment: encoded_store_binding,
            sequence,
            predecessor_occurrence,
            candidate_transition_commitment,
            operation_id: ControlPlaneCommitOperationIdV1(recomputed_operation_id),
            commitment: recomputed_occurrence_commitment,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ControlPlaneOccurrenceWireError {
    #[error("control-plane occurrence wire record is truncated")]
    Truncated,
    #[error("control-plane occurrence wire length cannot be represented")]
    LengthOverflow,
    #[error("unsupported control-plane occurrence wire domain/version")]
    UnsupportedDomain,
    #[error("control-plane occurrence wire store binding does not match expected binding")]
    StoreBindingMismatch,
    #[error("control-plane occurrence wire sequence must be nonzero")]
    SequenceZero,
    #[error("invalid control-plane occurrence predecessor discriminator {0}")]
    InvalidPredecessorDiscriminator(u8),
    #[error("control-plane occurrence sequence/predecessor shape is invalid")]
    InvalidOccurrenceShape,
    #[error("control-plane occurrence wire operation id does not recompute")]
    OperationIdMismatch,
    #[error("control-plane occurrence wire occurrence commitment does not recompute")]
    OccurrenceCommitmentMismatch,
    #[error("control-plane occurrence wire record contains trailing bytes")]
    TrailingBytes,
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WireCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], ControlPlaneOccurrenceWireError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ControlPlaneOccurrenceWireError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ControlPlaneOccurrenceWireError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, ControlPlaneOccurrenceWireError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u64(&mut self) -> Result<u64, ControlPlaneOccurrenceWireError> {
        let raw: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| ControlPlaneOccurrenceWireError::Truncated)?;
        Ok(u64::from_be_bytes(raw))
    }

    fn read_commitment(&mut self) -> Result<Commitment32, ControlPlaneOccurrenceWireError> {
        let raw: [u8; 32] = self
            .read_exact(32)?
            .try_into()
            .map_err(|_| ControlPlaneOccurrenceWireError::Truncated)?;
        Ok(Commitment32::from_bytes(raw))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GENESIS_WIRE_HEX: &str = "000000000000003673796d74686165612e736369656e63652e766965772e636f6e74726f6c2d706c616e652e6f6363757272656e63652d776972652e7631cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05000000000000000100dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8dfc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0";
    const SUCCESSOR_WIRE_HEX: &str = "000000000000003673796d74686165612e736369656e63652e766965772e636f6e74726f6c2d706c616e652e6f6363757272656e63652d776972652e7631cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05000000000000000201f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0";

    fn c(byte: u8) -> Commitment32 {
        Commitment32::from_bytes([byte; 32])
    }

    fn commitment(hex: &str) -> Commitment32 {
        let bytes = hex_bytes(hex);
        Commitment32::from_bytes(bytes.try_into().unwrap())
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
            .collect()
    }

    fn binding() -> ControlPlaneOccurrenceStoreBindingV1 {
        ControlPlaneOccurrenceStoreBindingV1::new(
            "deployment/site01-a",
            "lunar/site01",
            "control-plane/store-a",
            7,
            c(71),
        )
        .unwrap()
    }

    fn genesis() -> ProposedControlPlaneOccurrenceV1 {
        ProposedControlPlaneOccurrenceV1 {
            store_binding_commitment: commitment(
                "cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05",
            ),
            sequence: 1,
            predecessor_occurrence: None,
            candidate_transition_commitment: commitment(
                "dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8d",
            ),
            operation_id: ControlPlaneCommitOperationIdV1(commitment(
                "fc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0",
            )),
            commitment: commitment(
                "f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0",
            ),
        }
    }

    fn successor() -> ProposedControlPlaneOccurrenceV1 {
        ProposedControlPlaneOccurrenceV1 {
            store_binding_commitment: binding().commitment(),
            sequence: 2,
            predecessor_occurrence: Some(genesis().commitment()),
            candidate_transition_commitment: commitment(
                "a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf",
            ),
            operation_id: ControlPlaneCommitOperationIdV1(commitment(
                "31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428",
            )),
            commitment: commitment(
                "629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0",
            ),
        }
    }

    #[test]
    fn genesis_wire_vector_round_trips_exactly() {
        let occurrence = genesis();
        let bytes = occurrence.to_canonical_bytes_v1();
        assert_eq!(bytes.len(), 199);
        assert_eq!(bytes, hex_bytes(GENESIS_WIRE_HEX));
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap(),
            occurrence
        );
    }

    #[test]
    fn successor_wire_vector_round_trips_exactly() {
        let occurrence = successor();
        let bytes = occurrence.to_canonical_bytes_v1();
        assert_eq!(bytes.len(), 231);
        assert_eq!(bytes, hex_bytes(SUCCESSOR_WIRE_HEX));
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap(),
            occurrence
        );
    }

    #[test]
    fn wrong_store_epoch_is_rejected() {
        let wrong = ControlPlaneOccurrenceStoreBindingV1::new(
            "deployment/site01-a",
            "lunar/site01",
            "control-plane/store-a",
            8,
            c(71),
        )
        .unwrap();
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(
                &wrong,
                &genesis().to_canonical_bytes_v1(),
            )
            .unwrap_err(),
            ControlPlaneOccurrenceWireError::StoreBindingMismatch
        );
    }

    #[test]
    fn mutated_candidate_with_old_ids_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[103] ^= 1;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::OperationIdMismatch
        );
    }

    #[test]
    fn mutated_predecessor_with_old_ids_is_rejected() {
        let mut bytes = successor().to_canonical_bytes_v1();
        bytes[103] ^= 1;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::OperationIdMismatch
        );
    }

    #[test]
    fn mutated_operation_id_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[135] ^= 1;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::OperationIdMismatch
        );
    }

    #[test]
    fn mutated_occurrence_commitment_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[167] ^= 1;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::OccurrenceCommitmentMismatch
        );
    }

    #[test]
    fn zero_sequence_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[94..102].fill(0);
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::SequenceZero
        );
    }

    #[test]
    fn successor_without_predecessor_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[94..102].copy_from_slice(&2u64.to_be_bytes());
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::InvalidOccurrenceShape
        );
    }

    #[test]
    fn genesis_with_predecessor_is_rejected() {
        let mut bytes = successor().to_canonical_bytes_v1();
        bytes[94..102].copy_from_slice(&1u64.to_be_bytes());
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::InvalidOccurrenceShape
        );
    }

    #[test]
    fn invalid_predecessor_discriminator_is_rejected() {
        let mut bytes = genesis().to_canonical_bytes_v1();
        bytes[102] = 2;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(&binding(), &bytes)
                .unwrap_err(),
            ControlPlaneOccurrenceWireError::InvalidPredecessorDiscriminator(2)
        );
    }

    #[test]
    fn truncated_trailing_and_unknown_domain_records_are_rejected() {
        let mut truncated = genesis().to_canonical_bytes_v1();
        truncated.pop();
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(
                &binding(),
                &truncated,
            )
            .unwrap_err(),
            ControlPlaneOccurrenceWireError::Truncated
        );

        let mut trailing = genesis().to_canonical_bytes_v1();
        trailing.push(0);
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(
                &binding(),
                &trailing,
            )
            .unwrap_err(),
            ControlPlaneOccurrenceWireError::TrailingBytes
        );

        let mut wrong_domain = genesis().to_canonical_bytes_v1();
        wrong_domain[8] ^= 1;
        assert_eq!(
            ProposedControlPlaneOccurrenceV1::from_canonical_bytes_v1_checked(
                &binding(),
                &wrong_domain,
            )
            .unwrap_err(),
            ControlPlaneOccurrenceWireError::UnsupportedDomain
        );
    }
}
