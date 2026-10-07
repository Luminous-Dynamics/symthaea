//! Holochain-facing projection for durable Symthaea evidence.
//!
//! This module deliberately has no Holochain crate dependency.  It defines the
//! deterministic, content-addressed entry shape that a thin conductor adapter
//! can commit through an app/zome.  Realtime Iroh messages remain ephemeral;
//! this projection is the durable semantic boundary.
//!
//! A projection is an anchor, not proof by itself.  The referenced evidence,
//! VDS proof, and signatures must still be independently verified before the
//! anchor is admitted by a Holochain integrity zome.

use uuid::Uuid;

pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/holochain-evidence-anchor-v1";
pub const MAX_SELECTION_POLICY_BYTES: usize = 256;
pub const MAX_CONTEXT_BYTES: usize = 8 * 1024;
pub const HOLOCHAIN_ACTION_HASH_BYTES: usize = 39;
/// RFC 9942 header 394 receipt collections are capped at 16 receipts.
pub const MAX_RECEIPT_SELECTION_CANDIDATES: u32 = 16;
/// Holochain 0.7 ActionHash primitive prefix (`uhCkk`), in raw bytes.
pub const HOLOCHAIN_ACTION_HASH_PREFIX: [u8; 3] = [0x84, 0x29, 0x24];

/// Opaque native Holochain ActionHash bytes.
///
/// Symthaea deliberately does not depend on holo_hash; the future conductor
/// adapter can convert this exact 39-byte representation into the native
/// ActionHash and retrieve the dependency with must_get_valid_record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HolochainActionHash([u8; HOLOCHAIN_ACTION_HASH_BYTES]);

impl HolochainActionHash {
    pub fn from_raw(bytes: [u8; HOLOCHAIN_ACTION_HASH_BYTES]) -> Result<Self, HolochainProjectionError> {
        if bytes == [0; HOLOCHAIN_ACTION_HASH_BYTES] {
            return Err(HolochainProjectionError::ZeroDigest);
        }
        if bytes[..3] != HOLOCHAIN_ACTION_HASH_PREFIX {
            return Err(HolochainProjectionError::InvalidActionHashType);
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; HOLOCHAIN_ACTION_HASH_BYTES] {
        &self.0
    }
}

/// The kind of durable object being anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceAnchorKind {
    Observation,
    Claim,
    Decision,
    Outcome,
    Attestation,
}

impl EvidenceAnchorKind {
    fn tag(self) -> u8 {
        match self {
            Self::Observation => 1,
            Self::Claim => 2,
            Self::Decision => 3,
            Self::Outcome => 4,
            Self::Attestation => 5,
        }
    }
}

/// Explicit receipt-selection provenance.
///
/// RFC 9942 receipt arrays are priority ordered, so the selected index is
/// semantic context, not interchangeable set membership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptSelectionContext {
    pub collection_sha256: [u8; 32],
    pub collection_len: u32,
    pub selected_index: u32,
    pub selected_receipt_sha256: [u8; 32],
    /// Digest of the complete selection decision, including rejected and
    /// not-evaluated candidates. This is an application digest, not a
    /// Holochain EntryHash; a conductor adapter must map the decision to a
    /// native addressable entry when validation needs to retrieve it.
    pub selection_decision_sha256: [u8; 32],
    pub selection_policy: String,
    pub selection_policy_version: u16,
}

impl ReceiptSelectionContext {
    /// Build a durable projection directly from a fully evaluated RFC 9942
    /// selection decision, preventing the compact anchor from disagreeing with
    /// the decision it references.
    #[cfg(feature = "semantic-receipts")]
    pub fn from_decision(
        decision: &crate::rfc9942_selection::ReceiptSelectionDecision,
    ) -> Result<Self, HolochainProjectionError> {
        let selection_decision_sha256 = decision
            .validated_digest()
            .map_err(|_| HolochainProjectionError::InvalidReceiptSelection)?;

        let selected_index = decision
            .selected_index
            .ok_or(HolochainProjectionError::InvalidReceiptSelection)?;
        let selected_receipt_sha256 = decision
            .selected_receipt_sha256
            .ok_or(HolochainProjectionError::InvalidReceiptSelection)?;

        Ok(Self {
            collection_sha256: decision.collection_sha256,
            collection_len: decision.collection_len,
            selected_index,
            selected_receipt_sha256,
            selection_decision_sha256,
            selection_policy: decision.policy_id.to_owned(),
            selection_policy_version: decision.policy_version,
        })
    }

    fn validate(&self) -> Result<(), HolochainProjectionError> {
        if self.collection_len == 0
            || self.collection_len > MAX_RECEIPT_SELECTION_CANDIDATES
            || self.selected_index >= self.collection_len
        {
            return Err(HolochainProjectionError::InvalidReceiptSelection);
        }
        if self.selection_policy.is_empty()
            || self.selection_policy.len() > MAX_SELECTION_POLICY_BYTES
        {
            return Err(HolochainProjectionError::FieldTooLarge(
                "selection_policy",
            ));
        }
        if self.collection_sha256 == [0; 32]
            || self.selected_receipt_sha256 == [0; 32]
            || self.selection_decision_sha256 == [0; 32]
        {
            return Err(HolochainProjectionError::ZeroDigest);
        }
        Ok(())
    }
}

/// A compact durable reference to evidence that has already been verified by
/// the Symthaea evidence pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HolochainEvidenceAnchor {
    pub anchor_id: Uuid,
    pub origin_node_id: Uuid,
    pub kind: EvidenceAnchorKind,
    pub evidence_digest: [u8; 32],
    pub parent_evidence_digest: Option<[u8; 32]>,
    /// Native ActionHash for the parent evidence entry, when already published.
    pub parent_action_hash: Option<HolochainActionHash>,
    pub vds_root: Option<[u8; 32]>,
    pub receipt_selection: Option<ReceiptSelectionContext>,
    /// Native ActionHash for the durable selection decision entry, when already published.
    pub selection_decision_action_hash: Option<HolochainActionHash>,
    /// Human/application context is carried as bytes but is never interpreted
    /// by the canonical encoder.  The integrity zome can impose its own schema.
    pub context: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HolochainProjectionError {
    ZeroDigest,
    InvalidReceiptSelection,
    UnaddressableDependency(&'static str),
    InvalidActionHashType,
    FieldTooLarge(&'static str),
}

impl HolochainEvidenceAnchor {
    pub fn validate(&self) -> Result<(), HolochainProjectionError> {
        if self.anchor_id.is_nil() || self.origin_node_id.is_nil() {
            return Err(HolochainProjectionError::ZeroDigest);
        }
        if self.evidence_digest == [0; 32] {
            return Err(HolochainProjectionError::ZeroDigest);
        }
        if self.context.len() > MAX_CONTEXT_BYTES {
            return Err(HolochainProjectionError::FieldTooLarge("context"));
        }
        if let Some(parent) = self.parent_evidence_digest {
            if parent == [0; 32] {
                return Err(HolochainProjectionError::ZeroDigest);
            }
        }
        if self.parent_evidence_digest.is_some() && self.parent_action_hash.is_none() {
            return Err(HolochainProjectionError::UnaddressableDependency("parent_evidence"));
        }
        if let Some(root) = self.vds_root {
            if root == [0; 32] {
                return Err(HolochainProjectionError::ZeroDigest);
            }
        }
        if let Some(selection) = &self.receipt_selection {
            selection.validate()?;
        }
        if self.receipt_selection.is_some() && self.selection_decision_action_hash.is_none() {
            return Err(HolochainProjectionError::UnaddressableDependency("selection_decision"));
        }
        Ok(())
    }

    /// Canonical bytes suitable for hashing and deterministic integrity-zome
    /// validation.  UUIDs and fixed-size digests have fixed width; variable
    /// fields are length-prefixed.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, HolochainProjectionError> {
        self.validate()?;
        let mut out = Vec::with_capacity(256 + self.context.len());
        put_bytes(&mut out, DOMAIN);
        put_u16(&mut out, VERSION);
        put_uuid(&mut out, self.anchor_id);
        put_uuid(&mut out, self.origin_node_id);
        out.push(self.kind.tag());
        out.extend_from_slice(&self.evidence_digest);

        match self.parent_evidence_digest {
            Some(digest) => {
                out.push(1);
                out.extend_from_slice(&digest);
            }
            None => out.push(0),
        }

        match self.parent_action_hash {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(hash.as_bytes());
            }
            None => out.push(0),
        }

        match self.vds_root {
            Some(root) => {
                out.push(1);
                out.extend_from_slice(&root);
            }
            None => out.push(0),
        }

        match &self.receipt_selection {
            Some(selection) => {
                out.push(1);
                out.extend_from_slice(&selection.collection_sha256);
                put_u32(&mut out, selection.collection_len);
                put_u32(&mut out, selection.selected_index);
                out.extend_from_slice(&selection.selected_receipt_sha256);
                out.extend_from_slice(&selection.selection_decision_sha256);
                put_string(&mut out, &selection.selection_policy);
                put_u16(&mut out, selection.selection_policy_version);
            }
            None => out.push(0),
        }

        match self.selection_decision_action_hash {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(hash.as_bytes());
            }
            None => out.push(0),
        }

        put_bytes(&mut out, &self.context);
        Ok(out)
    }
}

fn put_uuid(out: &mut Vec<u8>, value: Uuid) {
    out.extend_from_slice(value.as_bytes());
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_action_hash(fill: u8) -> HolochainActionHash {
        let mut bytes = [fill; HOLOCHAIN_ACTION_HASH_BYTES];
        bytes[..3].copy_from_slice(&HOLOCHAIN_ACTION_HASH_PREFIX);
        HolochainActionHash::from_raw(bytes).unwrap()
    }

    fn anchor() -> HolochainEvidenceAnchor {
        HolochainEvidenceAnchor {
            anchor_id: Uuid::from_u128(1),
            origin_node_id: Uuid::from_u128(2),
            kind: EvidenceAnchorKind::Attestation,
            evidence_digest: [1; 32],
            parent_evidence_digest: Some([2; 32]),
            parent_action_hash: Some(valid_action_hash(7)),
            vds_root: Some([3; 32]),
            receipt_selection: Some(ReceiptSelectionContext {
                collection_sha256: [4; 32],
                collection_len: 2,
                selected_index: 1,
                selected_receipt_sha256: [5; 32],
                selection_decision_sha256: [6; 32],
                selection_policy: "rfc9942/priority-first-valid-v1".into(),
                selection_policy_version: 1,
            }),
            selection_decision_action_hash: Some(valid_action_hash(8)),
            context: b"qualification".to_vec(),
        }
    }

    #[test]
    fn canonical_encoding_is_deterministic() {
        assert_eq!(anchor().canonical_bytes(), anchor().canonical_bytes());
    }

    #[test]
    fn receipt_selection_is_not_a_set() {
        let mut first = anchor();
        let first_bytes = first.canonical_bytes().unwrap();
        first.receipt_selection.as_mut().unwrap().selected_index = 0;
        let second_bytes = first.canonical_bytes().unwrap();
        assert_ne!(first_bytes, second_bytes);
    }

    #[test]
    fn collection_fingerprint_is_witnessed() {
        let mut changed = anchor();
        let before = changed.canonical_bytes().unwrap();
        changed
            .receipt_selection
            .as_mut()
            .unwrap()
            .collection_sha256[0] ^= 1;
        assert_ne!(before, changed.canonical_bytes().unwrap());
    }

    #[test]
    fn selection_context_rejects_collection_length_above_rfc9942_ceiling() {
        let mut projected = anchor();
        projected.receipt_selection.as_mut().unwrap().collection_len =
            MAX_RECEIPT_SELECTION_CANDIDATES + 1;
        assert_eq!(
            projected.validate(),
            Err(HolochainProjectionError::InvalidReceiptSelection)
        );
        assert!(projected.canonical_bytes().is_err());
    }

    #[test]
    fn invalid_selection_cannot_be_projected() {
        let mut invalid = anchor();
        invalid.receipt_selection.as_mut().unwrap().selected_index = 2;
        assert_eq!(
            invalid.canonical_bytes(),
            Err(HolochainProjectionError::InvalidReceiptSelection)
        );
    }

    #[test]
    fn zero_digest_is_fail_closed() {
        let mut invalid = anchor();
        invalid.evidence_digest = [0; 32];
        assert_eq!(
            invalid.validate(),
            Err(HolochainProjectionError::ZeroDigest)
        );
    }

    #[test]
    fn claimed_durable_dependencies_must_be_addressable() {
        let mut invalid = anchor();
        invalid.parent_action_hash = None;
        assert_eq!(
            invalid.validate(),
            Err(HolochainProjectionError::UnaddressableDependency("parent_evidence"))
        );
        let mut invalid_selection = anchor();
        invalid_selection.selection_decision_action_hash = None;
        assert_eq!(
            invalid_selection.validate(),
            Err(HolochainProjectionError::UnaddressableDependency("selection_decision"))
        );
    }

    #[test]
    fn selection_decision_digest_is_bound() {
        let mut first = anchor();
        let before = first.canonical_bytes().unwrap();
        first
            .receipt_selection
            .as_mut()
            .unwrap()
            .selection_decision_sha256[0] ^= 1;
        assert_ne!(before, first.canonical_bytes().unwrap());
    }

    #[test]
    fn native_action_hash_binding_is_canonical() {
        let mut first = anchor();
        let before = first.canonical_bytes().unwrap();
        first.parent_action_hash = Some(valid_action_hash(9));
        assert_ne!(before, first.canonical_bytes().unwrap());
        assert!(HolochainActionHash::from_raw([0; HOLOCHAIN_ACTION_HASH_BYTES]).is_err());
        assert_eq!(
            HolochainActionHash::from_raw([9; HOLOCHAIN_ACTION_HASH_BYTES]),
            Err(HolochainProjectionError::InvalidActionHashType)
        );
    }

    #[test]
    fn selection_dependency_identity_is_canonical() {
        let mut first = anchor();
        let before = first.canonical_bytes().unwrap();
        first.selection_decision_action_hash = Some(valid_action_hash(9));
        assert_ne!(before, first.canonical_bytes().unwrap());
    }

    #[test]
    fn context_is_length_framed() {
        let mut first = anchor();
        first.context = b"a".to_vec();
        let mut second = anchor();
        second.context = b"ab".to_vec();
        assert_ne!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
    }
}
