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

pub const VERSION: u16 = 2;
pub const DOMAIN: &[u8] = b"symthaea-swarm/holochain-evidence-anchor-v2";
/// Stable identifiers for the application-level PQ-bound assurance schema.
pub const HYBRID_ASSURANCE_PROFILE_ID: &str =
    "symthaea-swarm/rfc9942-pq-bound-mldsa65-v1";
pub const HYBRID_ASSURANCE_PROFILE_VERSION: u16 = 1;
pub const HYBRID_ASSURANCE_ML_DSA_65_ALGORITHM_ID: i64 = -49;
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
    /// Identity of the cryptographically verified Receipt capability selected
    /// by the decision. This is not itself an authorization or truth claim.
    pub verified_capability_sha256: [u8; 32],
    /// Identity of the verified outer Signature_With_Receipt composition that
    /// carried the selected Receipt and caused the atomic selection result.
    /// This is a provenance capability, not an authorization or truth claim.
    pub verified_composition_capability_sha256: [u8; 32],
    /// Digest of the complete selection decision, including rejected and
    /// not-evaluated candidates. This is an application digest, not a
    /// Holochain EntryHash; a conductor adapter must map the decision to a
    /// native addressable entry when validation needs to retrieve it.
    pub selection_decision_sha256: [u8; 32],
    pub selection_policy: String,
    pub selection_policy_version: u16,
    /// Whether policy required the stronger PQ-bound assurance at admission.
    /// Read-only externally: it is set by the assurance admission constructor.
    hybrid_required: bool,
    /// PQ attestation identity, retained in canonical durable projection.
    /// Read-only externally: it is derived from a privately constructed verified capability.
    hybrid_assurance: Option<HybridReceiptAssuranceContext>,
}

/// Canonical durable identity for an application-level PQ-bound receipt
/// attestation. This is metadata to be independently checked by validators,
/// not a replacement for retaining or verifying the source cryptographic bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridReceiptAssuranceContext {
    profile_id: String,
    profile_version: u16,
    pq_algorithm_id: i64,
    hybrid_capability_sha256: [u8; 32],
    key_policy_digest_sha256: [u8; 32],
    evaluation_time_unix_seconds: u64,
    pq_key_id: [u8; 16],
    verifying_key_sha256: [u8; 32],
    pq_signature_sha256: [u8; 32],
    receipt_sha256: [u8; 32],
    classical_capability_sha256: [u8; 32],
    transcript_sha256: [u8; 32],
}

impl HybridReceiptAssuranceContext {
    pub fn profile_id(&self) -> &str { &self.profile_id }
    pub const fn profile_version(&self) -> u16 { self.profile_version }
    pub const fn pq_algorithm_id(&self) -> i64 { self.pq_algorithm_id }
    pub const fn hybrid_capability_sha256(&self) -> [u8; 32] { self.hybrid_capability_sha256 }
    pub const fn key_policy_digest_sha256(&self) -> [u8; 32] { self.key_policy_digest_sha256 }
    pub const fn evaluation_time_unix_seconds(&self) -> u64 { self.evaluation_time_unix_seconds }
    pub const fn pq_key_id(&self) -> [u8; 16] { self.pq_key_id }
    pub const fn verifying_key_sha256(&self) -> [u8; 32] { self.verifying_key_sha256 }
    pub const fn pq_signature_sha256(&self) -> [u8; 32] { self.pq_signature_sha256 }
    pub const fn receipt_sha256(&self) -> [u8; 32] { self.receipt_sha256 }
    pub const fn classical_capability_sha256(&self) -> [u8; 32] { self.classical_capability_sha256 }
    pub const fn transcript_sha256(&self) -> [u8; 32] { self.transcript_sha256 }

    #[cfg(feature = "semantic-receipts")]
    fn from_verified(
        hybrid: &crate::rfc9942_hybrid::Rfc9942HybridVerifiedReceipt,
    ) -> Self {
        let attestation = hybrid.pq_attestation();
        let transcript = attestation.transcript();
        Self {
            profile_id: HYBRID_ASSURANCE_PROFILE_ID.to_owned(),
            profile_version: HYBRID_ASSURANCE_PROFILE_VERSION,
            pq_algorithm_id: HYBRID_ASSURANCE_ML_DSA_65_ALGORITHM_ID,
            hybrid_capability_sha256: hybrid.hybrid_capability_sha256(),
            key_policy_digest_sha256: hybrid.key_policy_digest_sha256(),
            evaluation_time_unix_seconds:
                hybrid.key_authorization_evaluation_time_unix_seconds(),
            pq_key_id: attestation.key_id().bytes(),
            verifying_key_sha256: attestation.verifying_key_sha256(),
            pq_signature_sha256: attestation.signature_sha256(),
            receipt_sha256: transcript.receipt_sha256(),
            classical_capability_sha256: transcript.classical_capability_sha256(),
            transcript_sha256: transcript.transcript_sha256(),
        }
    }

    fn validate(
        &self,
        selected_receipt_sha256: [u8; 32],
        verified_capability_sha256: [u8; 32],
    ) -> Result<(), HolochainProjectionError> {
        if self.profile_id != HYBRID_ASSURANCE_PROFILE_ID
            || self.profile_id.len() > MAX_SELECTION_POLICY_BYTES
            || self.profile_version != HYBRID_ASSURANCE_PROFILE_VERSION
            || self.pq_algorithm_id != HYBRID_ASSURANCE_ML_DSA_65_ALGORITHM_ID
            || self.hybrid_capability_sha256 == [0; 32]
            || self.key_policy_digest_sha256 == [0; 32]
            || self.pq_key_id == [0; 16]
            || self.verifying_key_sha256 == [0; 32]
            || self.pq_signature_sha256 == [0; 32]
            || self.receipt_sha256 == [0; 32]
            || self.classical_capability_sha256 == [0; 32]
            || self.transcript_sha256 == [0; 32]
            || self.receipt_sha256 != selected_receipt_sha256
            || self.classical_capability_sha256 != verified_capability_sha256
        {
            return Err(HolochainProjectionError::InvalidHybridAssurance);
        }
        Ok(())
    }
}

impl ReceiptSelectionContext {
    /// Whether the assurance policy explicitly required PQ-bound verification.
    pub const fn hybrid_required(&self) -> bool {
        self.hybrid_required
    }

    /// Read-only reference to the hybrid assurance metadata, if admitted.
    pub fn hybrid_assurance(&self) -> Option<&HybridReceiptAssuranceContext> {
        self.hybrid_assurance.as_ref()
    }

    /// Internal structural conversion used only after both source-collection
    /// and verified-capability binding have succeeded.
    #[cfg(feature = "semantic-receipts")]
    fn from_decision(
        decision: &crate::rfc9942_selection::ReceiptSelectionDecision,
        verified_capability_sha256: [u8; 32],
        verified_composition_capability_sha256: [u8; 32],
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
            verified_capability_sha256,
            verified_composition_capability_sha256,
            selection_decision_sha256,
            selection_policy: decision.policy_id.to_owned(),
            selection_policy_version: decision.policy_version,
            hybrid_required: false,
            hybrid_assurance: None,
        })
    }

    /// Build the durable selection context from the proof-carrying witness.
    ///
    /// The witness carries both the inner verified Receipt capability and the
    /// verified outer Signature_With_Receipt composition, so neither causal
    /// provenance layer is discarded when crossing into the durable model.
    #[cfg(feature = "semantic-receipts")]
    pub fn from_verified_selection(
        selection: &crate::rfc9942_selection::Rfc9942VerifiedReceiptSelection,
    ) -> Result<Self, HolochainProjectionError> {
        Self::from_decision(
            selection.decision(),
            selection.verified_capability_sha256(),
            selection.verified_composition_capability_sha256(),
        )
    }

    /// Build a durable context from the explicit assurance admission gate.
    /// This preserves whether hybrid assurance was required, and refuses to
    /// serialize a hybrid-required context without the bound PQ capability.
    #[cfg(feature = "semantic-receipts")]
    pub fn from_assurance_admission(
        admission: &crate::rfc9942_hybrid::Rfc9942SelectionAssuranceAdmission,
    ) -> Result<Self, HolochainProjectionError> {
        let selection = admission.selection();
        let mut context = Self::from_decision(
            selection.decision(),
            selection.verified_capability_sha256(),
            selection.verified_composition_capability_sha256(),
        )?;
        context.hybrid_required = admission.requirement()
            == crate::rfc9942_hybrid::Rfc9942HybridRequirement::HybridRequired;
        context.hybrid_assurance = admission
            .hybrid()
            .map(HybridReceiptAssuranceContext::from_verified);
        context.validate()?;
        Ok(context)
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
        if self.hybrid_required && self.hybrid_assurance.is_none() {
            return Err(HolochainProjectionError::HybridAssuranceRequired);
        }
        if let Some(hybrid) = &self.hybrid_assurance {
            hybrid.validate(
                self.selected_receipt_sha256,
                self.verified_capability_sha256,
            )?;
        }
        if self.collection_sha256 == [0; 32]
            || self.selected_receipt_sha256 == [0; 32]
            || self.verified_capability_sha256 == [0; 32]
            || self.verified_composition_capability_sha256 == [0; 32]
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
    HybridAssuranceRequired,
    InvalidHybridAssurance,
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
                out.extend_from_slice(&selection.verified_capability_sha256);
                out.extend_from_slice(&selection.verified_composition_capability_sha256);
                out.extend_from_slice(&selection.selection_decision_sha256);
                put_string(&mut out, &selection.selection_policy);
                put_u16(&mut out, selection.selection_policy_version);
                out.push(u8::from(selection.hybrid_required));
                match &selection.hybrid_assurance {
                    Some(hybrid) => {
                        out.push(1);
                        put_string(&mut out, &hybrid.profile_id);
                        put_u16(&mut out, hybrid.profile_version);
                        out.extend_from_slice(&hybrid.pq_algorithm_id.to_be_bytes());
                        out.extend_from_slice(&hybrid.hybrid_capability_sha256);
                        out.extend_from_slice(&hybrid.key_policy_digest_sha256);
                        out.extend_from_slice(
                            &hybrid.evaluation_time_unix_seconds.to_be_bytes(),
                        );
                        out.extend_from_slice(&hybrid.pq_key_id);
                        out.extend_from_slice(&hybrid.verifying_key_sha256);
                        out.extend_from_slice(&hybrid.pq_signature_sha256);
                        out.extend_from_slice(&hybrid.receipt_sha256);
                        out.extend_from_slice(&hybrid.classical_capability_sha256);
                        out.extend_from_slice(&hybrid.transcript_sha256);
                    }
                    None => out.push(0),
                }
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
                verified_capability_sha256: [6; 32],
                verified_composition_capability_sha256: [9; 32],
                selection_decision_sha256: [7; 32],
                selection_policy: "rfc9942/priority-first-valid-v1".into(),
                selection_policy_version: 1,
                hybrid_required: false,
                hybrid_assurance: None,
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

    #[cfg(feature = "semantic-receipts")]
    #[test]
    fn projection_profile_constants_match_hybrid_verifier_profile() {
        assert_eq!(
            HYBRID_ASSURANCE_PROFILE_ID,
            crate::rfc9942_hybrid::HYBRID_POLICY_ID
        );
        assert_eq!(
            HYBRID_ASSURANCE_PROFILE_VERSION,
            crate::rfc9942_hybrid::HYBRID_POLICY_VERSION
        );
        assert_eq!(
            HYBRID_ASSURANCE_ML_DSA_65_ALGORITHM_ID,
            crate::rfc9942_hybrid::ML_DSA_65_COSE_ALGORITHM_ID
        );
    }

    #[test]
    fn hybrid_required_projection_fails_closed_without_hybrid_capability() {
        let mut projected = anchor();
        projected.receipt_selection.as_mut().unwrap().hybrid_required = true;
        assert_eq!(
            projected.validate(),
            Err(HolochainProjectionError::HybridAssuranceRequired)
        );
        assert!(projected.canonical_bytes().is_err());
    }

    #[test]
    fn hybrid_attestation_identity_is_canonical_and_bound_to_selection() {
        let mut projected = anchor();
        let selection = projected.receipt_selection.as_mut().unwrap();
        selection.hybrid_assurance = Some(HybridReceiptAssuranceContext {
            profile_id: crate::rfc9942_hybrid::HYBRID_POLICY_ID.to_owned(),
            profile_version: crate::rfc9942_hybrid::HYBRID_POLICY_VERSION,
            pq_algorithm_id: crate::rfc9942_hybrid::ML_DSA_65_COSE_ALGORITHM_ID,
            hybrid_capability_sha256: [10; 32],
            key_policy_digest_sha256: [11; 32],
            evaluation_time_unix_seconds: 1234,
            pq_key_id: [12; 16],
            verifying_key_sha256: [13; 32],
            pq_signature_sha256: [14; 32],
            receipt_sha256: [5; 32],
            classical_capability_sha256: [6; 32],
            transcript_sha256: [15; 32],
        });
        let before = projected.canonical_bytes().unwrap();
        projected.receipt_selection.as_mut().unwrap()
            .hybrid_assurance.as_mut().unwrap()
            .hybrid_capability_sha256[0] ^= 1;
        assert_ne!(before, projected.canonical_bytes().unwrap());

        let mut wrong_receipt = anchor();
        wrong_receipt.receipt_selection.as_mut().unwrap().hybrid_assurance =
            projected.receipt_selection.as_ref().unwrap().hybrid_assurance.clone();
        assert_eq!(
            wrong_receipt.validate(),
            Err(HolochainProjectionError::InvalidHybridAssurance)
        );
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
    fn verified_capability_identity_is_bound() {
        let mut first = anchor();
        let before = first.canonical_bytes().unwrap();
        first
            .receipt_selection
            .as_mut()
            .unwrap()
            .verified_capability_sha256[0] ^= 1;
        assert_ne!(before, first.canonical_bytes().unwrap());
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
