// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compact Merkle membership proofs for federation manifest source records.
//!
//! The proof authenticates membership in the exact canonical source-record set
//! of a manifest. It does not authenticate scientific truth or authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evidence_lineage_dkg::DkgNodeType;
use crate::evidence_lineage_federation::{
    EvidenceLineageDkgFederationManifest, FederationSourceRecord,
};

const MERKLE_LEAF_DOMAIN: &[u8] = b"symthaea:evidence-lineage-federation-merkle-leaf:v1\0";
const MERKLE_PARENT_DOMAIN: &[u8] = b"symthaea:evidence-lineage-federation-merkle-parent:v1\0";
const PROOF_DIGEST_DOMAIN: &[u8] = b"symthaea:evidence-lineage-federation-merkle-proof:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MerkleSide {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleSibling {
    pub side: MerkleSide,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDkgMerkleMembershipProof {
    pub proof_version: String,
    pub digest_algorithm: String,
    pub manifest_digest: String,
    pub projection_digest: String,
    pub node_id: String,
    pub node_type: DkgNodeType,
    pub record_digest: String,
    pub leaf_index: u64,
    pub leaf_count: u64,
    pub merkle_root: String,
    pub siblings: Vec<MerkleSibling>,
    pub proof_digest: String,
}

impl EvidenceLineageDkgMerkleMembershipProof {
    pub fn prove(
        manifest: &EvidenceLineageDkgFederationManifest,
        node_id: &str,
        node_type: DkgNodeType,
        record_digest: &str,
    ) -> Result<Self, MerkleProofError> {
        manifest.verify_source_records()
            .map_err(MerkleProofError::InvalidManifest)?;

        let target = FederationSourceRecord {
            node_id: node_id.to_owned(),
            node_type,
            record_digest: record_digest.to_owned(),
        };
        let index = manifest.source_records.iter().position(|r| r == &target)
            .ok_or(MerkleProofError::MemberNotFound)?;

        let leaves: Vec<[u8; 32]> = manifest.source_records.iter().map(leaf_hash).collect();
        let root = merkle_root(&leaves)?;
        let siblings = merkle_siblings(&leaves, index)?;

        let mut proof = Self {
            proof_version: "1.0.0".into(),
            digest_algorithm: "sha256".into(),
            manifest_digest: manifest.manifest_digest.clone(),
            projection_digest: manifest.projection_digest.clone(),
            node_id: node_id.into(),
            node_type,
            record_digest: record_digest.into(),
            leaf_index: index as u64,
            leaf_count: leaves.len() as u64,
            merkle_root: hex_digest(root),
            siblings,
            proof_digest: String::new(),
        };
        proof.proof_digest = proof.compute_digest();
        Ok(proof)
    }

    pub fn verify_against(
        &self,
        manifest: &EvidenceLineageDkgFederationManifest,
    ) -> Result<(), MerkleProofError> {
        manifest.verify_source_records()
            .map_err(MerkleProofError::InvalidManifest)?;
        if self.proof_version != "1.0.0" {
            return Err(MerkleProofError::ProofVersionMismatch);
        }
        if self.digest_algorithm != "sha256" {
            return Err(MerkleProofError::DigestAlgorithmMismatch);
        }
        if self.manifest_digest != manifest.manifest_digest {
            return Err(MerkleProofError::ManifestDigestMismatch);
        }
        if self.projection_digest != manifest.projection_digest {
            return Err(MerkleProofError::ProjectionDigestMismatch);
        }
        if self.leaf_count != manifest.source_records.len() as u64
            || self.leaf_index >= self.leaf_count {
            return Err(MerkleProofError::LeafPositionMismatch);
        }
        let target = FederationSourceRecord {
            node_id: self.node_id.clone(),
            node_type: self.node_type,
            record_digest: self.record_digest.clone(),
        };
        let actual_index = manifest.source_records.iter().position(|r| r == &target)
            .ok_or(MerkleProofError::MemberNotFound)?;
        if actual_index as u64 != self.leaf_index {
            return Err(MerkleProofError::LeafPositionMismatch);
        }

        let mut digest = leaf_hash(&target);
        let mut index = self.leaf_index as usize;
        for sibling in &self.siblings {
            digest = match sibling.side {
                MerkleSide::Left => parent_hash(&sibling.digest, &hex_digest(digest)),
                MerkleSide::Right => parent_hash(&hex_digest(digest), &sibling.digest),
            };
            index /= 2;
        }
        if hex_digest(digest) != self.merkle_root {
            return Err(MerkleProofError::MerkleRootMismatch);
        }
        let expected_root = merkle_root(&manifest.source_records.iter().map(leaf_hash).collect::<Vec<_>>())?;
        if hex_digest(expected_root) != self.merkle_root {
            return Err(MerkleProofError::MerkleRootMismatch);
        }
        if self.proof_digest != self.compute_digest() {
            return Err(MerkleProofError::ProofDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(PROOF_DIGEST_DOMAIN);
        put(&mut h, &self.proof_version);
        put(&mut h, &self.digest_algorithm);
        put(&mut h, &self.manifest_digest);
        put(&mut h, &self.projection_digest);
        put(&mut h, &self.node_id);
        put(&mut h, node_type(self.node_type));
        put(&mut h, &self.record_digest);
        h.update(self.leaf_index.to_be_bytes());
        h.update(self.leaf_count.to_be_bytes());
        put(&mut h, &self.merkle_root);
        h.update((self.siblings.len() as u64).to_be_bytes());
        for sibling in &self.siblings {
            put(&mut h, match sibling.side { MerkleSide::Left => "left", MerkleSide::Right => "right" });
            put(&mut h, &sibling.digest);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn leaf_hash(record: &FederationSourceRecord) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(MERKLE_LEAF_DOMAIN);
    put(&mut h, &record.node_id);
    put(&mut h, node_type(record.node_type));
    put(&mut h, &record.record_digest);
    h.finalize().into()
}

fn parent_hash(left: &str, right: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(MERKLE_PARENT_DOMAIN);
    put(&mut h, left);
    put(&mut h, right);
    h.finalize().into()
}

fn merkle_root(leaves: &[[u8; 32]]) -> Result<[u8; 32], MerkleProofError> {
    if leaves.is_empty() {
        return Err(MerkleProofError::EmptySourceSet);
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            let left = hex_digest(pair[0]);
            let right = if pair.len() == 2 { hex_digest(pair[1]) } else { left.clone() };
            next.push(parent_hash(&left, &right));
        }
        level = next;
    }
    Ok(level[0])
}

fn merkle_siblings(
    leaves: &[[u8; 32]],
    target: usize,
) -> Result<Vec<MerkleSibling>, MerkleProofError> {
    if leaves.is_empty() || target >= leaves.len() {
        return Err(MerkleProofError::LeafPositionMismatch);
    }
    let mut level = leaves.to_vec();
    let mut index = target;
    let mut siblings = Vec::new();
    while level.len() > 1 {
        let is_right = index % 2 == 1;
        let sibling_index = if is_right { index - 1 } else { index + 1 };
        let sibling = if sibling_index < level.len() {
            level[sibling_index]
        } else {
            level[index]
        };
        siblings.push(MerkleSibling {
            side: if is_right { MerkleSide::Left } else { MerkleSide::Right },
            digest: hex_digest(sibling),
        });
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            let left = hex_digest(pair[0]);
            let right = if pair.len() == 2 { hex_digest(pair[1]) } else { left.clone() };
            next.push(parent_hash(&left, &right));
        }
        level = next;
        index /= 2;
    }
    Ok(siblings)
}

fn hex_digest(bytes: [u8; 32]) -> String { format!("{:x}", bytes) }

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn node_type(value: DkgNodeType) -> &'static str {
    match value {
        DkgNodeType::CandidateCommitment => "candidate_commitment",
        DkgNodeType::ExternalObservation => "external_observation",
        DkgNodeType::IndependentAssessment => "independent_assessment",
        DkgNodeType::Replication => "replication",
        DkgNodeType::CriterionEvidence => "criterion_evidence",
        DkgNodeType::EvidenceChainAudit => "evidence_chain_audit",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleProofError {
    InvalidManifest(crate::evidence_lineage_federation::FederationManifestError),
    EmptySourceSet,
    MemberNotFound,
    ProofVersionMismatch,
    DigestAlgorithmMismatch,
    ManifestDigestMismatch,
    ProjectionDigestMismatch,
    LeafPositionMismatch,
    MerkleRootMismatch,
    ProofDigestMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_lineage_dkg::{DkgEdge, DkgEdgeType, DkgNode};
    use crate::evidence_lineage_federation::EvidenceLineageDkgFederationManifest;

    fn manifest() -> EvidenceLineageDkgFederationManifest {
        let p = crate::evidence_lineage_dkg::EvidenceLineageDkgProjection::new(
            vec![
                DkgNode { node_id:"commitment:1".into(), node_type:DkgNodeType::CandidateCommitment, record_digest:"sha256:a".into(), supersedes_node_id:None },
                DkgNode { node_id:"observation:1".into(), node_type:DkgNodeType::ExternalObservation, record_digest:"sha256:b".into(), supersedes_node_id:None },
                DkgNode { node_id:"assessment:1".into(), node_type:DkgNodeType::IndependentAssessment, record_digest:"sha256:c".into(), supersedes_node_id:None },
                DkgNode { node_id:"replication:1".into(), node_type:DkgNodeType::Replication, record_digest:"sha256:d".into(), supersedes_node_id:None },
                DkgNode { node_id:"evidence:1".into(), node_type:DkgNodeType::CriterionEvidence, record_digest:"sha256:e".into(), supersedes_node_id:None },
            ],
            vec![
                DkgEdge {source_node_id:"observation:1".into(),edge_type:DkgEdgeType::ObservedFrom,target_node_id:"commitment:1".into()},
                DkgEdge {source_node_id:"assessment:1".into(),edge_type:DkgEdgeType::Assesses,target_node_id:"observation:1".into()},
                DkgEdge {source_node_id:"replication:1".into(),edge_type:DkgEdgeType::Replicates,target_node_id:"assessment:1".into()},
                DkgEdge {source_node_id:"evidence:1".into(),edge_type:DkgEdgeType::EligibleFor,target_node_id:"replication:1".into()},
            ],
        );
        EvidenceLineageDkgFederationManifest::from_projection(&p,None).unwrap()
    }

    #[test]
    fn membership_proof_verifies() {
        let m=manifest();
        let p=EvidenceLineageDkgMerkleMembershipProof::prove(&m,"observation:1",DkgNodeType::ExternalObservation,"sha256:b").unwrap();
        assert!(p.verify_against(&m).is_ok());
    }

    #[test]
    fn wrong_member_is_rejected() {
        let m=manifest();
        assert_eq!(EvidenceLineageDkgMerkleMembershipProof::prove(&m,"missing",DkgNodeType::ExternalObservation,"sha256:b"),Err(MerkleProofError::MemberNotFound));
    }

    #[test]
    fn tampered_sibling_is_rejected() {
        let m=manifest();
        let mut p=EvidenceLineageDkgMerkleMembershipProof::prove(&m,"observation:1",DkgNodeType::ExternalObservation,"sha256:b").unwrap();
        p.siblings[0].digest="00".repeat(32);
        assert_eq!(p.verify_against(&m),Err(MerkleProofError::MerkleRootMismatch));
    }

    #[test]
    fn tampered_root_is_rejected() {
        let m=manifest();
        let mut p=EvidenceLineageDkgMerkleMembershipProof::prove(&m,"observation:1",DkgNodeType::ExternalObservation,"sha256:b").unwrap();
        p.merkle_root="00".repeat(32);
        assert_eq!(p.verify_against(&m),Err(MerkleProofError::MerkleRootMismatch));
    }

    #[test]
    fn tampered_binding_is_rejected() {
        let m=manifest();
        let mut p=EvidenceLineageDkgMerkleMembershipProof::prove(&m,"observation:1",DkgNodeType::ExternalObservation,"sha256:b").unwrap();
        p.manifest_digest="sha256:tampered".into();
        assert_eq!(p.verify_against(&m),Err(MerkleProofError::ManifestDigestMismatch));
    }

    #[test]
    fn serde_roundtrip_preserves_proof() {
        let m=manifest();
        let p=EvidenceLineageDkgMerkleMembershipProof::prove(&m,"observation:1",DkgNodeType::ExternalObservation,"sha256:b").unwrap();
        let bytes=serde_json::to_vec(&p).unwrap();
        let decoded: EvidenceLineageDkgMerkleMembershipProof=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(p,decoded);
        assert!(decoded.verify_against(&m).is_ok());
    }
}