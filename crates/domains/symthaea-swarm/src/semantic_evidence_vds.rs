// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit boundary between local history witnesses and VDS consistency proofs.
//!
//! The local chained EvidenceHistory is deliberately not treated as a Merkle
//! VDS. This module also contains a concrete RFC 9162 SHA-256 verifier over an
//! independent ordered leaf sequence. Callers must explicitly map evidence
//! records into VDS leaves.

use crate::semantic_evidence_history::HistoryCheckpoint;
use sha2::{Digest, Sha256};

pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-vds";
pub const RFC9162_VDS_NAME: &str = "RFC9162_SHA256";
pub const RFC9162_VDS_ID: u64 = 1;
pub const RFC9162_INCLUSION_PROOF_ID: i64 = -1;
pub const RFC9162_CONSISTENCY_PROOF_ID: i64 = -2;
pub const RFC9942_RECEIPTS_HEADER_LABEL: i64 = 394;
pub const RFC9942_VDS_HEADER_LABEL: i64 = 395;
pub const RFC9942_VDP_HEADER_LABEL: i64 = 396;

/// RFC 9942 proof type carried in the vdp header map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rfc9942ProofKind {
    Inclusion,
    Consistency,
}

impl Rfc9942ProofKind {
    pub const fn label(self) -> i64 {
        match self {
            Self::Inclusion => RFC9162_INCLUSION_PROOF_ID,
            Self::Consistency => RFC9162_CONSISTENCY_PROOF_ID,
        }
    }

    pub const fn from_label(label: i64) -> Option<Self> {
        match label {
            RFC9162_INCLUSION_PROOF_ID => Some(Self::Inclusion),
            RFC9162_CONSISTENCY_PROOF_ID => Some(Self::Consistency),
            _ => None,
        }
    }
}

/// RFC 9942 vdp (label 396) value for the RFC9162_SHA256 VDS.
///
/// The RFC 9942 `vds` parameter (label 395) is a separate protected-header
/// field. This object intentionally does not serialize that identifier.
/// This models the proof collection, not a COSE receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc9942Vdp {
    kind: Rfc9942ProofKind,
    proofs: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rfc9942VdpError {
    #[error("proof collection must contain at least one proof")]
    EmptyProofCollection,
    #[error("RFC 9942 receipt payload must be exactly 32 bytes for SHA-256")]
    InvalidPayloadLength,
    #[error("RFC 9942 vds header value {0} does not identify RFC9162_SHA256")]
    VdsMismatch(u64),
    #[error("proof collection contains an invalid RFC 9162 proof: {0}")]
    InvalidProof(#[from] Rfc9162ProofDecodeError),
    #[error("invalid RFC 9942 VDP structure")]
    InvalidStructure,
    #[error("invalid RFC 9942 VDP CBOR encoding")]
    InvalidEncoding,
    #[error("trailing bytes after RFC 9942 VDP")]
    TrailingBytes,
    #[error("RFC 9942 VDP proof kind does not match the requested verification")]
    WrongProofKind,
    #[error("none of the supplied RFC 9942 proofs verifies against the expected VDS state")]
    NoMatchingProof,
}

impl Rfc9942Vdp {
    pub fn new(kind: Rfc9942ProofKind, proofs: Vec<Vec<u8>>) -> Result<Self, Rfc9942VdpError> {
        if proofs.is_empty() {
            return Err(Rfc9942VdpError::EmptyProofCollection);
        }
        for proof in &proofs {
            match kind {
                Rfc9942ProofKind::Inclusion => {
                    let decoded = Rfc9162InclusionProof::from_cbor(proof)?;
                    if decoded.inclusion_path.is_empty() {
                        return Err(Rfc9942VdpError::InvalidProof(Rfc9162ProofDecodeError::InvalidStructure));
                    }
                }
                Rfc9942ProofKind::Consistency => { Rfc9162ConsistencyProof::from_cbor(proof)?; }
            }
        }
        Ok(Self { kind, proofs })
    }

    /// The VDS identifier this concrete VDP implementation is defined for.
    /// It is metadata, not part of this VDP's CBOR bytes; callers must bind it
    /// to the separate RFC 9942 `vds` protected-header value.
    pub const fn vds_id(&self) -> u64 { RFC9162_VDS_ID }

    /// Bind the VDP to the RFC 9942 vds protected-header value before proof
    /// verification. The actual protected-header/COSE parser remains external.
    pub fn validate_vds_id(&self, vds_id: u64) -> Result<(), Rfc9942VdpError> {
        if vds_id == self.vds_id() { Ok(()) } else { Err(Rfc9942VdpError::VdsMismatch(vds_id)) }
    }

    pub const fn kind(&self) -> Rfc9942ProofKind { self.kind }
    pub fn proofs(&self) -> &[Vec<u8>] { &self.proofs }

    /// Verify inclusion after explicitly binding the RFC 9942 `vds` header value.
    /// The collection may contain multiple proofs; at least one must verify.
    pub fn verify_inclusion_for_vds(
        &self,
        vds_id: u64,
        candidate_entry: &[u8],
        expected_head: VdsTreeHead,
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        self.validate_vds_id(vds_id)?;
        self.verify_inclusion(candidate_entry, expected_head)
    }

    /// Verify inclusion against the 32-byte signed/detached receipt payload root.
    /// COSE signature verification and detached-payload resolution remain external.
    pub fn verify_inclusion_with_payload(
        &self,
        candidate_entry: &[u8],
        payload: &[u8],
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        if self.kind != Rfc9942ProofKind::Inclusion {
            return Err(Rfc9942VdpError::WrongProofKind);
        }
        if payload.len() != 32 {
            return Err(Rfc9942VdpError::InvalidPayloadLength);
        }
        let mut root = [0u8; 32];
        root.copy_from_slice(payload);
        let vds = Rfc9162Sha256Vds;
        for proof_bytes in &self.proofs {
            let proof = Rfc9162InclusionProof::from_cbor(proof_bytes)?;
            let head = VdsTreeHead::new(proof.tree_size, root);
            if vds.verify_inclusion(candidate_entry, root, &proof) {
                return Ok(head);
            }
        }
        Err(Rfc9942VdpError::NoMatchingProof)
    }

    pub fn verify_inclusion(
        &self,
        candidate_entry: &[u8],
        expected_head: VdsTreeHead,
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        if self.kind != Rfc9942ProofKind::Inclusion {
            return Err(Rfc9942VdpError::WrongProofKind);
        }
        let vds = Rfc9162Sha256Vds;
        for proof in &self.proofs {
            if vds.verify_rfc9942_inclusion_cbor(candidate_entry, expected_head, proof).is_ok() {
                return Ok(expected_head);
            }
        }
        Err(Rfc9942VdpError::NoMatchingProof)
    }

    /// Verify consistency after explicitly binding the RFC 9942 `vds` header value.
    /// The collection may contain multiple proofs; at least one must verify.
    pub fn verify_consistency_for_vds(
        &self,
        vds_id: u64,
        older: VdsTreeHead,
        newer: VdsTreeHead,
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        self.validate_vds_id(vds_id)?;
        self.verify_consistency(older, newer)
    }

    /// Verify consistency against the 32-byte signed/detached newer-tree root.
    /// The older tree head is supplied separately; COSE remains external.
    pub fn verify_consistency_with_payload(
        &self,
        older: VdsTreeHead,
        payload: &[u8],
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        if self.kind != Rfc9942ProofKind::Consistency {
            return Err(Rfc9942VdpError::WrongProofKind);
        }
        if payload.len() != 32 {
            return Err(Rfc9942VdpError::InvalidPayloadLength);
        }
        let mut root = [0u8; 32];
        root.copy_from_slice(payload);
        let vds = Rfc9162Sha256Vds;
        for proof_bytes in &self.proofs {
            let proof = Rfc9162ConsistencyProof::from_cbor(proof_bytes)?;
            if proof.first != older.tree_size() {
                continue;
            }
            let newer = VdsTreeHead::new(proof.second, root);
            if vds.verify(older.root(), root, &proof) {
                return Ok(newer);
            }
        }
        Err(Rfc9942VdpError::NoMatchingProof)
    }

    pub fn verify_consistency(
        &self,
        older: VdsTreeHead,
        newer: VdsTreeHead,
    ) -> Result<VdsTreeHead, Rfc9942VdpError> {
        if self.kind != Rfc9942ProofKind::Consistency {
            return Err(Rfc9942VdpError::WrongProofKind);
        }
        let vds = Rfc9162Sha256Vds;
        for proof in &self.proofs {
            if vds.verify_rfc9942_consistency_cbor(older, newer, proof).is_ok() {
                return Ok(newer);
            }
        }
        Err(Rfc9942VdpError::NoMatchingProof)
    }

    pub fn to_cbor(&self) -> Vec<u8> {
        let mut out = Vec::new();
        cbor_map_len(&mut out, 1);
        cbor_int(&mut out, self.kind.label());
        cbor_array_len(&mut out, self.proofs.len() as u64);
        for proof in &self.proofs { cbor_bytes(&mut out, proof); }
        out
    }

    pub fn from_cbor(bytes: &[u8]) -> Result<Self, Rfc9942VdpError> {
        let mut reader = CborReader::new(bytes);
        let map_len = reader.read_map_len().map_err(|_| Rfc9942VdpError::InvalidEncoding)?;
        if map_len != 1 { return Err(Rfc9942VdpError::InvalidStructure); }
        let label = reader.read_i64().map_err(|_| Rfc9942VdpError::InvalidEncoding)?;
        let kind = Rfc9942ProofKind::from_label(label).ok_or(Rfc9942VdpError::InvalidStructure)?;
        let count = reader.read_array_len().map_err(|_| Rfc9942VdpError::InvalidEncoding)?;
        if count == 0 { return Err(Rfc9942VdpError::EmptyProofCollection); }
        let mut proofs = Vec::with_capacity(count);
        for _ in 0..count {
            proofs.push(reader.read_bstr().map_err(|_| Rfc9942VdpError::InvalidEncoding)?);
        }
        match reader.finish() {
            Ok(()) => Self::new(kind, proofs),
            Err(Rfc9162ProofDecodeError::TrailingBytes) => Err(Rfc9942VdpError::TrailingBytes),
            Err(_) => Err(Rfc9942VdpError::InvalidEncoding),
        }
    }
}

/// A VDS-native tree head binds an ordered tree size to its Merkle root.
/// It is intentionally independent of the local chained HistoryCheckpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VdsTreeHead {
    tree_size: u64,
    root: [u8; 32],
}

impl VdsTreeHead {
    pub fn new(tree_size: u64, root: [u8; 32]) -> Self { Self { tree_size, root } }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn root(&self) -> [u8; 32] { self.root }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyStatus {
    Valid,
    Invalid,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsistencyRequest {
    older: HistoryCheckpoint,
    newer: HistoryCheckpoint,
}

impl ConsistencyRequest {
    pub fn new(older: HistoryCheckpoint, newer: HistoryCheckpoint) -> Self { Self { older, newer } }
    pub fn older(&self) -> HistoryCheckpoint { self.older }
    pub fn newer(&self) -> HistoryCheckpoint { self.newer }
    pub fn is_strict_extension_request(&self) -> bool { self.older.length() < self.newer.length() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyProof {
    vds: &'static str,
    version: u16,
    bytes: Vec<u8>,
}

impl ConsistencyProof {
    pub fn new(vds: &'static str, version: u16, bytes: Vec<u8>) -> Self {
        Self { vds, version, bytes }
    }
    pub fn vds(&self) -> &'static str { self.vds }
    pub fn version(&self) -> u16 { self.version }
    pub fn bytes(&self) -> &[u8] { &self.bytes }
}

pub trait HistoryVds {
    fn vds_name(&self) -> &'static str;
    fn prove_consistency(&self, _request: &ConsistencyRequest) -> Result<ConsistencyProof, ConsistencyError> {
        Err(ConsistencyError::Unsupported)
    }
    fn verify_consistency(&self, request: &ConsistencyRequest, proof: &ConsistencyProof) -> ConsistencyStatus;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConsistencyError {
    #[error("this VDS adapter does not support consistency proofs")]
    Unsupported,
    #[error("a consistency proof cannot be generated for the supplied request")]
    CannotGenerate,
}

/// The current chained local witness remains explicitly outside the VDS layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainedHistoryVds;

impl HistoryVds for ChainedHistoryVds {
    fn vds_name(&self) -> &'static str { "symthaea-chained-history-v1" }
    fn verify_consistency(
        &self,
        _request: &ConsistencyRequest,
        _proof: &ConsistencyProof,
    ) -> ConsistencyStatus {
        ConsistencyStatus::Unsupported
    }
}

/// RFC 9162 consistency proof represented in semantic form.
///
/// RFC 9942 maps this to CBOR as [old_size, new_size, consistency_path].
/// Encoding is intentionally separate from this verifier so it can later be
/// bound to the RFC 9942 receipt layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc9162ConsistencyProof {
    pub first: u64,
    pub second: u64,
    pub consistency_path: Vec<[u8; 32]>,
}

impl Rfc9162ConsistencyProof {
    pub fn new(first: u64, second: u64, consistency_path: Vec<[u8; 32]>) -> Self {
        Self { first, second, consistency_path }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rfc9162ProofVerificationError {
    #[error("invalid RFC 9942 proof encoding: {0}")]
    Decode(#[from] Rfc9162ProofDecodeError),
    #[error("proof tree size does not match expected tree head")]
    TreeSizeMismatch,
    #[error("proof root does not match expected tree head")]
    RootMismatch,
    #[error("RFC 9162 proof verification failed")]
    InvalidProof,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rfc9162ProofDecodeError {
    #[error("unexpected end of CBOR input")]
    UnexpectedEof,
    #[error("invalid CBOR major type or non-preferred encoding")]
    InvalidEncoding,
    #[error("invalid proof structure")]
    InvalidStructure,
    #[error("invalid hash length")]
    InvalidHashLength,
    #[error("trailing bytes after proof")]
    TrailingBytes,
}

struct CborReader<'a> { bytes: &'a [u8], offset: usize }
impl<'a> CborReader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, offset: 0 } }
    fn read_u64(&mut self) -> Result<u64, Rfc9162ProofDecodeError> {
        let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?; self.offset+=1;
        if initial>>5 != 0 { return Err(Rfc9162ProofDecodeError::InvalidEncoding); }
        let ai=initial&0x1f;
        match ai { 0..=23 => Ok(ai as u64), 24 => self.read_uint(1,24), 25=>self.read_uint(2,256), 26=>self.read_uint(4,65536), 27=>self.read_uint(8,4294967296), _=>Err(Rfc9162ProofDecodeError::InvalidEncoding) }
    }
    fn read_uint(&mut self,n:usize,min:u64)->Result<u64,Rfc9162ProofDecodeError>{ if self.offset+n>self.bytes.len(){return Err(Rfc9162ProofDecodeError::UnexpectedEof)} let mut v=0u64; for b in &self.bytes[self.offset..self.offset+n]{v=(v<<8)|*b as u64;} self.offset+=n; if v<min{return Err(Rfc9162ProofDecodeError::InvalidEncoding)} Ok(v) }
    fn read_i64(&mut self) -> Result<i64, Rfc9162ProofDecodeError> {
        let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?;
        let major=initial>>5;
        if major!=0 && major!=1 { return Err(Rfc9162ProofDecodeError::InvalidEncoding); }
        self.offset+=1;
        let ai=initial&0x1f;
        let argument=match ai {
            0..=23=>ai as u64,
            24=>self.read_uint(1,24)?,
            25=>self.read_uint(2,256)?,
            26=>self.read_uint(4,65_536)?,
            27=>self.read_uint(8,4_294_967_296)?,
            _=>return Err(Rfc9162ProofDecodeError::InvalidEncoding),
        };
        if major==0 {
            i64::try_from(argument).map_err(|_| Rfc9162ProofDecodeError::InvalidStructure)
        } else {
            let value=argument.checked_add(1).ok_or(Rfc9162ProofDecodeError::InvalidStructure)?;
            if value>i64::MAX as u64+1 { return Err(Rfc9162ProofDecodeError::InvalidStructure); }
            if value==i64::MAX as u64+1 { Ok(i64::MIN) } else { Ok(-(value as i64)) }
        }
    }

    fn read_map_len(&mut self) -> Result<usize, Rfc9162ProofDecodeError> {
        let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?;
        self.offset+=1;
        if initial>>5!=5 { return Err(Rfc9162ProofDecodeError::InvalidEncoding); }
        let ai=initial&0x1f;
        let n=match ai {
            0..=23=>ai as u64,
            24=>self.read_uint(1,24)?,
            25=>self.read_uint(2,256)?,
            26=>self.read_uint(4,65_536)?,
            27=>self.read_uint(8,4_294_967_296)?,
            _=>return Err(Rfc9162ProofDecodeError::InvalidEncoding),
        };
        usize::try_from(n).map_err(|_| Rfc9162ProofDecodeError::InvalidStructure)
    }

    fn read_bstr(&mut self) -> Result<Vec<u8>, Rfc9162ProofDecodeError> {
        let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?;
        self.offset+=1;
        if initial>>5!=2 { return Err(Rfc9162ProofDecodeError::InvalidEncoding); }
        let ai=initial&0x1f;
        let n=match ai {
            0..=23=>ai as u64,
            24=>self.read_uint(1,24)?,
            25=>self.read_uint(2,256)?,
            26=>self.read_uint(4,65_536)?,
            27=>self.read_uint(8,4_294_967_296)?,
            _=>return Err(Rfc9162ProofDecodeError::InvalidEncoding),
        };
        let n=usize::try_from(n).map_err(|_| Rfc9162ProofDecodeError::InvalidStructure)?;
        let end=self.offset.checked_add(n).ok_or(Rfc9162ProofDecodeError::InvalidStructure)?;
        if end>self.bytes.len() { return Err(Rfc9162ProofDecodeError::UnexpectedEof); }
        let bytes=self.bytes[self.offset..end].to_vec();
        self.offset=end;
        Ok(bytes)
    }
    fn read_array_len(&mut self)->Result<usize,Rfc9162ProofDecodeError>{ let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?; self.offset+=1; if initial>>5!=4{return Err(Rfc9162ProofDecodeError::InvalidEncoding)} let ai=initial&0x1f; let n=match ai{0..=23=>ai as u64,24=>self.read_uint(1,24)?,25=>self.read_uint(2,256)?,26=>self.read_uint(4,65536)?,27=>self.read_uint(8,4294967296)?,_=>return Err(Rfc9162ProofDecodeError::InvalidEncoding)}; usize::try_from(n).map_err(|_|Rfc9162ProofDecodeError::InvalidStructure) }
    fn read_bstr32(&mut self)->Result<[u8;32],Rfc9162ProofDecodeError>{ let initial=*self.bytes.get(self.offset).ok_or(Rfc9162ProofDecodeError::UnexpectedEof)?; self.offset+=1; if initial>>5!=2{return Err(Rfc9162ProofDecodeError::InvalidEncoding)} let ai=initial&0x1f; let n=match ai{0..=23=>ai as u64,24=>self.read_uint(1,24)?,25=>self.read_uint(2,256)?,26=>self.read_uint(4,65536)?,27=>self.read_uint(8,4294967296)?,_=>return Err(Rfc9162ProofDecodeError::InvalidEncoding)}; if n!=32{return Err(Rfc9162ProofDecodeError::InvalidHashLength)} let end=self.offset.checked_add(32).ok_or(Rfc9162ProofDecodeError::InvalidStructure)?; if end>self.bytes.len(){return Err(Rfc9162ProofDecodeError::UnexpectedEof)} let mut out=[0u8;32]; out.copy_from_slice(&self.bytes[self.offset..end]); self.offset=end; Ok(out) }
    fn finish(self)->Result<(),Rfc9162ProofDecodeError>{ if self.offset==self.bytes.len(){Ok(())}else{Err(Rfc9162ProofDecodeError::TrailingBytes)} }
}

impl Rfc9162ConsistencyProof {
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, Rfc9162ProofDecodeError> {
        let mut r=CborReader::new(bytes); if r.read_array_len()? != 3{return Err(Rfc9162ProofDecodeError::InvalidStructure)}
        let first=r.read_u64()?; let second=r.read_u64()?; let n=r.read_array_len()?; let mut path=Vec::with_capacity(n); for _ in 0..n{path.push(r.read_bstr32()?)} r.finish()?;
        if first==0 || first>=second || path.is_empty(){return Err(Rfc9162ProofDecodeError::InvalidStructure)}
        Ok(Self::new(first,second,path))
    }
}

impl Rfc9162InclusionProof {
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, Rfc9162ProofDecodeError> {
        let mut r=CborReader::new(bytes); if r.read_array_len()? != 3{return Err(Rfc9162ProofDecodeError::InvalidStructure)}
        let tree_size=r.read_u64()?; let leaf_index=r.read_u64()?; let n=r.read_array_len()?; let mut path=Vec::with_capacity(n); for _ in 0..n{path.push(r.read_bstr32()?)} r.finish()?;
        if tree_size==0 || leaf_index>=tree_size{return Err(Rfc9162ProofDecodeError::InvalidStructure)}
        Ok(Self::new(tree_size,leaf_index,path))
    }
}

impl Rfc9162Sha256Vds {
    /// Verify RFC 9942 inclusion-proof content against an expected tree head.
    ///
    /// This performs the proof step only. COSE signature, signer identity,
    /// authorization, and receipt-policy checks remain outside this VDS layer.
    pub fn verify_rfc9942_inclusion_cbor(
        &self,
        candidate_entry: &[u8],
        expected_head: VdsTreeHead,
        proof_cbor: &[u8],
    ) -> Result<VdsTreeHead, Rfc9162ProofVerificationError> {
        let proof = Rfc9162InclusionProof::from_cbor(proof_cbor)?;
        if proof.tree_size != expected_head.tree_size() {
            return Err(Rfc9162ProofVerificationError::TreeSizeMismatch);
        }
        if !self.verify_inclusion(candidate_entry, expected_head.root(), &proof) {
            return Err(Rfc9162ProofVerificationError::InvalidProof);
        }
        Ok(expected_head)
    }

    /// Verify RFC 9942 inclusion proof content for an EvidenceDigest leaf.
    pub fn verify_rfc9942_evidence_inclusion_cbor(
        &self,
        digest: crate::semantic_evidence_digest::EvidenceDigest,
        expected_head: VdsTreeHead,
        proof_cbor: &[u8],
    ) -> Result<VdsTreeHead, Rfc9162ProofVerificationError> {
        let leaf = EvidenceVdsLeaf::from_evidence_digest(digest);
        self.verify_rfc9942_inclusion_cbor(leaf.as_bytes(), expected_head, proof_cbor)
    }

    /// Verify RFC 9942 consistency-proof content against both tree heads.
    ///
    /// The returned head is the newer head, matching RFC 9942's detached
    /// payload semantics for a consistency receipt.
    pub fn verify_rfc9942_consistency_cbor(
        &self,
        older: VdsTreeHead,
        newer: VdsTreeHead,
        proof_cbor: &[u8],
    ) -> Result<VdsTreeHead, Rfc9162ProofVerificationError> {
        let proof = Rfc9162ConsistencyProof::from_cbor(proof_cbor)?;
        if proof.first != older.tree_size() || proof.second != newer.tree_size() {
            return Err(Rfc9162ProofVerificationError::TreeSizeMismatch);
        }
        if !self.verify(older.root(), newer.root(), &proof) {
            return Err(Rfc9162ProofVerificationError::InvalidProof);
        }
        Ok(newer)
    }
}

impl Rfc9162ConsistencyProof {
    /// Encode the RFC 9942 proof-content array using deterministic CBOR.
    ///
    /// This is proof content only; it is not a COSE receipt and carries no
    /// signer, issuer, or authorization semantics.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut out = Vec::new();
        cbor_array_len(&mut out, 3);
        cbor_uint(&mut out, self.first);
        cbor_uint(&mut out, self.second);
        cbor_array_len(&mut out, self.consistency_path.len() as u64);
        for hash in &self.consistency_path { cbor_bytes(&mut out, hash); }
        out
    }
}

impl Rfc9162InclusionProof {
    /// Encode the RFC 9942 proof-content array using deterministic CBOR.
    ///
    /// This is proof content only; it is not a COSE receipt and carries no
    /// signer, issuer, or authorization semantics.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut out = Vec::new();
        cbor_array_len(&mut out, 3);
        cbor_uint(&mut out, self.tree_size);
        cbor_uint(&mut out, self.leaf_index);
        cbor_array_len(&mut out, self.inclusion_path.len() as u64);
        for hash in &self.inclusion_path { cbor_bytes(&mut out, hash); }
        out
    }
}

fn cbor_map_len(out: &mut Vec<u8>, len: u64) {
    match len {
        0..=23=>out.push(0xa0|len as u8),
        24..=255=>out.extend_from_slice(&[0xb8,len as u8]),
        256..=65_535=>{out.push(0xb9);out.extend_from_slice(&(len as u16).to_be_bytes());},
        65_536..=4_294_967_295=>{out.push(0xba);out.extend_from_slice(&(len as u32).to_be_bytes());},
        _=>{out.push(0xbb);out.extend_from_slice(&len.to_be_bytes());},
    }
}

fn cbor_int(out: &mut Vec<u8>, value: i64) {
    if value>=0 { cbor_uint(out,value as u64); } else {
        let encoded=(-1-value) as u64;
        match encoded {
            0..=23=>out.push(0x20|encoded as u8),
            24..=255=>out.extend_from_slice(&[0x38,encoded as u8]),
            256..=65_535=>{out.push(0x39);out.extend_from_slice(&(encoded as u16).to_be_bytes());},
            65_536..=4_294_967_295=>{out.push(0x3a);out.extend_from_slice(&(encoded as u32).to_be_bytes());},
            _=>{out.push(0x3b);out.extend_from_slice(&encoded.to_be_bytes());},
        }
    }
}
fn cbor_uint(out: &mut Vec<u8>, value: u64) {
    match value {
        0..=23 => out.push(value as u8),
        24..=255 => { out.extend_from_slice(&[0x18, value as u8]); }
        256..=65_535 => { out.push(0x19); out.extend_from_slice(&(value as u16).to_be_bytes()); }
        65_536..=4_294_967_295 => { out.push(0x1a); out.extend_from_slice(&(value as u32).to_be_bytes()); }
        _ => { out.push(0x1b); out.extend_from_slice(&value.to_be_bytes()); }
    }
}

fn cbor_array_len(out: &mut Vec<u8>, len: u64) {
    match len {
        0..=23 => out.push(0x80 | len as u8),
        24..=255 => out.extend_from_slice(&[0x98, len as u8]),
        256..=65_535 => { out.push(0x99); out.extend_from_slice(&(len as u16).to_be_bytes()); }
        65_536..=4_294_967_295 => { out.push(0x9a); out.extend_from_slice(&(len as u32).to_be_bytes()); }
        _ => { out.push(0x9b); out.extend_from_slice(&len.to_be_bytes()); }
    }
}

fn cbor_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    match bytes.len() as u64 {
        0..=23 => out.push(0x40 | bytes.len() as u8),
        24..=255 => out.extend_from_slice(&[0x58, bytes.len() as u8]),
        256..=65_535 => { out.push(0x59); out.extend_from_slice(&(bytes.len() as u16).to_be_bytes()); }
        65_536..=4_294_967_295 => { out.push(0x5a); out.extend_from_slice(&(bytes.len() as u32).to_be_bytes()); }
        _ => { out.push(0x5b); out.extend_from_slice(&(bytes.len() as u64).to_be_bytes()); }
    }
    out.extend_from_slice(bytes);
}

/// Canonical VDS leaf projection for a semantic evidence digest.
///
/// This is deliberately a projection: the EvidenceDigest remains the semantic
/// identity, while the VDS receives a versioned, domain-separated leaf input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceVdsLeaf([u8; 32]);

impl EvidenceVdsLeaf {
    pub fn from_evidence_digest(digest: crate::semantic_evidence_digest::EvidenceDigest) -> Self {
        let mut input = Vec::with_capacity(1 + 2 + 2 + DOMAIN.len() + 32);
        input.extend_from_slice(&(DOMAIN.len() as u16).to_be_bytes());
        input.extend_from_slice(DOMAIN);
        input.extend_from_slice(&VERSION.to_be_bytes());
        input.extend_from_slice(&1u16.to_be_bytes());
        input.extend_from_slice(digest.as_bytes());
        Self(sha256(&input))
    }

    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
    pub fn as_vec(&self) -> Vec<u8> { self.0.to_vec() }
}

/// RFC 9162 inclusion proof for one projected evidence leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc9162InclusionProof {
    pub tree_size: u64,
    pub leaf_index: u64,
    pub inclusion_path: Vec<[u8; 32]>,
}

impl Rfc9162InclusionProof {
    pub fn new(tree_size: u64, leaf_index: u64, inclusion_path: Vec<[u8; 32]>) -> Self {
        Self { tree_size, leaf_index, inclusion_path }
    }
}

/// Concrete RFC 9162 SHA-256 Merkle VDS operations.
///
/// This VDS consumes an explicit ordered leaf sequence. It does not consume
/// HistoryCheckpoint because the local chained witness has different semantics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rfc9162Sha256Vds;

impl Rfc9162Sha256Vds {
    pub fn vds_name(&self) -> &'static str { RFC9162_VDS_NAME }
    pub fn root(&self, leaves: &[Vec<u8>]) -> [u8; 32] { merkle_tree_hash(leaves) }

    pub fn tree_head(&self, leaves: &[Vec<u8>]) -> VdsTreeHead {
        VdsTreeHead::new(leaves.len() as u64, self.root(leaves))
    }

    /// Generate the RFC 9162 minimal consistency proof for the first `first`
    /// leaves of the supplied ordered sequence.
    pub fn prove(&self, leaves: &[Vec<u8>], first: usize) -> Option<Rfc9162ConsistencyProof> {
        if first == 0 || first >= leaves.len() { return None; }
        let path = consistency_subproof(first, leaves, true);
        Some(Rfc9162ConsistencyProof::new(first as u64, leaves.len() as u64, path))
    }

    pub fn inclusion_proof(&self, leaves: &[Vec<u8>], leaf_index: usize) -> Option<Rfc9162InclusionProof> {
        if leaf_index >= leaves.len() { return None; }
        let path = inclusion_path(leaf_index, leaves);
        Some(Rfc9162InclusionProof::new(leaves.len() as u64, leaf_index as u64, path))
    }

    pub fn verify_inclusion(
        &self,
        leaf: &[u8],
        root: [u8; 32],
        proof: &Rfc9162InclusionProof,
    ) -> bool {
        verify_rfc9162_inclusion(leaf, root, proof)
    }

    pub fn verify_evidence_inclusion(
        &self,
        digest: crate::semantic_evidence_digest::EvidenceDigest,
        root: [u8; 32],
        proof: &Rfc9162InclusionProof,
    ) -> bool {
        let leaf = EvidenceVdsLeaf::from_evidence_digest(digest);
        self.verify_inclusion(leaf.as_bytes(), root, proof)
    }

    pub fn verify(
        &self,
        first_root: [u8; 32],
        second_root: [u8; 32],
        proof: &Rfc9162ConsistencyProof,
    ) -> bool {
        verify_rfc9162_consistency(first_root, second_root, proof)
    }

    pub fn verify_tree_heads(&self, older: VdsTreeHead, newer: VdsTreeHead, proof: &Rfc9162ConsistencyProof) -> bool {
        older.tree_size() == proof.first
            && newer.tree_size() == proof.second
            && self.verify(older.root(), newer.root(), proof)
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn leaf_hash(data: &[u8]) -> [u8; 32] {
    let mut input = Vec::with_capacity(1 + data.len());
    input.push(0x00);
    input.extend_from_slice(data);
    sha256(&input)
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut input = [0u8; 65];
    input[0] = 0x01;
    input[1..33].copy_from_slice(left);
    input[33..].copy_from_slice(right);
    sha256(&input)
}

fn merkle_tree_hash(leaves: &[Vec<u8>]) -> [u8; 32] {
    match leaves.len() {
        0 => sha256(&[]),
        1 => leaf_hash(&leaves[0]),
        n => {
            let k = largest_power_of_two_less_than(n);
            let left = merkle_tree_hash(&leaves[..k].to_vec());
            let right = merkle_tree_hash(&leaves[k..].to_vec());
            node_hash(&left, &right)
        }
    }
}

fn largest_power_of_two_less_than(n: usize) -> usize {
    debug_assert!(n > 1);
    let highest = 1usize << (usize::BITS - 1 - n.leading_zeros());
    if highest == n { highest >> 1 } else { highest }
}

fn inclusion_path(index: usize, leaves: &[Vec<u8>]) -> Vec<[u8; 32]> {
    if leaves.len() <= 1 { return Vec::new(); }
    let k = largest_power_of_two_less_than(leaves.len());
    if index < k {
        let mut path = inclusion_path(index, &leaves[..k]);
        path.push(merkle_tree_hash(&leaves[k..].to_vec()));
        path
    } else {
        let mut path = inclusion_path(index - k, &leaves[k..]);
        path.push(merkle_tree_hash(&leaves[..k].to_vec()));
        path
    }
}

fn verify_rfc9162_inclusion(
    leaf: &[u8],
    root: [u8; 32],
    proof: &Rfc9162InclusionProof,
) -> bool {
    if proof.tree_size == 0 || proof.leaf_index >= proof.tree_size { return false; }
    let mut fn_ = proof.leaf_index;
    let mut sn = proof.tree_size - 1;
    let mut r = leaf_hash(leaf);

    for p in &proof.inclusion_path {
        if sn == 0 { return false; }
        if (fn_ & 1) == 1 || fn_ == sn {
            r = node_hash(p, &r);
            if fn_ & 1 == 0 {
                while fn_ & 1 == 0 && fn_ != 0 {
                    fn_ >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            r = node_hash(&r, p);
        }
        fn_ >>= 1;
        sn >>= 1;
    }

    sn == 0 && r == root
}

fn consistency_subproof(m: usize, leaves: &[Vec<u8>], complete: bool) -> Vec<[u8; 32]> {
    if m == leaves.len() {
        return if complete { Vec::new() } else { vec![merkle_tree_hash(leaves)] };
    }

    let k = largest_power_of_two_less_than(leaves.len());
    let mut proof = if m <= k {
        consistency_subproof(m, &leaves[..k], complete)
    } else {
        consistency_subproof(m - k, &leaves[k..], false)
    };

    if m <= k {
        proof.push(merkle_tree_hash(&leaves[k..].to_vec()));
    } else {
        proof.push(merkle_tree_hash(&leaves[..k].to_vec()));
    }
    proof
}

fn verify_rfc9162_consistency(
    first_root: [u8; 32],
    second_root: [u8; 32],
    proof: &Rfc9162ConsistencyProof,
) -> bool {
    if proof.first == 0 || proof.first >= proof.second || proof.consistency_path.is_empty() {
        return false;
    }

    if proof.first.is_power_of_two() {
        let mut path = Vec::with_capacity(proof.consistency_path.len() + 1);
        path.push(first_root);
        path.extend_from_slice(&proof.consistency_path);
        verify_consistency_path(first_root, second_root, proof.first, proof.second, &path)
    } else {
        verify_consistency_path(
            first_root,
            second_root,
            proof.first,
            proof.second,
            &proof.consistency_path,
        )
    }
}

fn verify_consistency_path(
    first_root: [u8; 32],
    second_root: [u8; 32],
    first: u64,
    second: u64,
    path: &[[u8; 32]],
) -> bool {
    let mut fn_ = first - 1;
    let mut sn = second - 1;

    if fn_ & 1 == 1 {
        while fn_ & 1 == 1 {
            fn_ >>= 1;
            sn >>= 1;
        }
    }

    let Some(first_node) = path.first().copied() else { return false };
    let mut fr = first_node;
    let mut sr = first_node;

    for c in &path[1..] {
        if sn == 0 { return false; }

        if (fn_ & 1) == 1 || fn_ == sn {
            fr = node_hash(c, &fr);
            sr = node_hash(c, &sr);

            if fn_ & 1 == 0 {
                while fn_ & 1 == 0 && fn_ != 0 {
                    fn_ >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            sr = node_hash(&sr, c);
        }

        fn_ >>= 1;
        sn >>= 1;
    }

    fr == first_root && sr == second_root && sn == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc9162_root_is_deterministic_and_order_sensitive() {
        let vds = Rfc9162Sha256Vds;
        let a = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let b = vec![b"b".to_vec(), b"a".to_vec(), b"c".to_vec()];
        assert_eq!(vds.root(&a), vds.root(&a));
        assert_ne!(vds.root(&a), vds.root(&b));
    }

    #[test]
    fn empty_and_singleton_roots_are_distinct() {
        let vds = Rfc9162Sha256Vds;
        assert_ne!(vds.root(&[]), vds.root(&[b"a".to_vec()]));
    }

    #[test]
    fn known_two_leaf_root_matches_definition() {
        let vds = Rfc9162Sha256Vds;
        let leaves = vec![b"a".to_vec(), b"b".to_vec()];
        let expected = node_hash(&leaf_hash(b"a"), &leaf_hash(b"b"));
        assert_eq!(vds.root(&leaves), expected);
    }

    #[test]
    fn rfc9942_proof_cbor_round_trips_through_strict_decoder() {
        let inclusion = Rfc9162InclusionProof::new(20, 17, vec![[0x11; 32], [0x22; 32]]);
        assert_eq!(Rfc9162InclusionProof::from_cbor(&inclusion.to_cbor()).unwrap(), inclusion);
        let consistency = Rfc9162ConsistencyProof::new(20, 104, vec![[0x33; 32], [0x44; 32]]);
        assert_eq!(Rfc9162ConsistencyProof::from_cbor(&consistency.to_cbor()).unwrap(), consistency);
    }

    #[test]
    fn rfc9942_proof_decoder_rejects_noncanonical_and_trailing_input() {
        let inclusion = Rfc9162InclusionProof::new(20, 17, vec![[0x11; 32]]);
        let mut encoded = inclusion.to_cbor();
        encoded.push(0);
        assert_eq!(
            Rfc9162InclusionProof::from_cbor(&encoded),
            Err(Rfc9162ProofDecodeError::TrailingBytes)
        );
        let noncanonical = vec![0x83, 0x18, 0x14, 0x11, 0x80];
        assert_eq!(
            Rfc9162InclusionProof::from_cbor(&noncanonical),
            Err(Rfc9162ProofDecodeError::InvalidEncoding)
        );
    }

    #[test]
    fn rfc9942_inclusion_and_consistency_cbor_shapes_are_deterministic() {
        let inclusion = Rfc9162InclusionProof::new(20, 17, vec![[0x11; 32], [0x22; 32]]);
        assert_eq!(
            inclusion.to_cbor(),
            vec![0x83, 0x14, 0x11, 0x82, 0x58, 0x20]
                .into_iter()
                .chain([0x11; 32])
                .chain([0x58, 0x20])
                .chain([0x22; 32])
                .collect::<Vec<_>>()
        );

        let consistency = Rfc9162ConsistencyProof::new(20, 104, vec![[0x33; 32]]);
        let expected = vec![0x83, 0x14, 0x18, 0x68, 0x81, 0x58, 0x20]
            .into_iter()
            .chain([0x33; 32])
            .collect::<Vec<_>>();
        assert_eq!(consistency.to_cbor(), expected);
    }

    #[test]
    fn rfc9942_vdp_encoding_matches_the_registered_label_shapes() {
        let inclusion=Rfc9162InclusionProof::new(2,0,vec![[0x11;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![inclusion]).unwrap();
        let mut expected=vec![0xa1,0x20,0x81,0x58,0x26];
        expected.extend_from_slice(&Rfc9162InclusionProof::new(2,0,vec![[0x11;32]]).to_cbor());
        assert_eq!(vdp.to_cbor(),expected);

        let consistency=Rfc9162ConsistencyProof::new(1,2,vec![[0x22;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Consistency,vec![consistency]).unwrap();
        let encoded=vdp.to_cbor();
        assert_eq!(&encoded[..3],&[0xa1,0x21,0x81]);
        assert_eq!(encoded[3],0x58);
        assert_eq!(encoded[4],0x26);
    }

    #[test]
    fn rfc9942_vdp_round_trips_and_preserves_proof_kind() {
        let inclusion=Rfc9162InclusionProof::new(4,2,vec![[0x11;32],[0x22;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![inclusion.clone()]).unwrap();
        let decoded=Rfc9942Vdp::from_cbor(&vdp.to_cbor()).unwrap();
        assert_eq!(decoded.vds_id(),RFC9162_VDS_ID);
        assert_eq!(decoded.kind(),Rfc9942ProofKind::Inclusion);
        assert_eq!(decoded.proofs(),&[inclusion]);

        let consistency=Rfc9162ConsistencyProof::new(1,2,vec![[0x22;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Consistency,vec![consistency.clone()]).unwrap();
        let decoded=Rfc9942Vdp::from_cbor(&vdp.to_cbor()).unwrap();
        assert_eq!(decoded.kind(),Rfc9942ProofKind::Consistency);
        assert_eq!(decoded.proofs(),&[consistency]);
    }

    #[test]
    fn rfc9942_vdp_rejects_rfc9162_singleton_inclusion_proof() {
        let singleton=Rfc9162InclusionProof::new(1,0,Vec::new()).to_cbor();
        assert_eq!(Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![singleton]),Err(Rfc9942VdpError::InvalidProof(Rfc9162ProofDecodeError::InvalidStructure)));
    }

    #[test]
    fn rfc9942_vdp_binds_inclusion_to_receipt_payload_root() {
        let vds=Rfc9162Sha256Vds; let leaves=vec![b"a".to_vec(),b"b".to_vec()]; let head=vds.tree_head(&leaves);
        let proof=vds.inclusion_proof(&leaves,0).unwrap().to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![proof]).unwrap();
        assert_eq!(vdp.verify_inclusion_with_payload(b"a",&head.root()).unwrap(),head);
        assert_eq!(vdp.verify_inclusion_with_payload(b"a",&[0xAA;32]),Err(Rfc9942VdpError::NoMatchingProof));
        assert_eq!(vdp.verify_inclusion_with_payload(b"a",&[0xAA;31]),Err(Rfc9942VdpError::InvalidPayloadLength));
    }

    #[test]
    fn rfc9942_vdp_binds_consistency_to_newer_receipt_payload_root() {
        let vds=Rfc9162Sha256Vds; let leaves:Vec<Vec<u8>>=(0..4).map(|i|format!("leaf-{i}").into_bytes()).collect();
        let older=vds.tree_head(&leaves[..2].to_vec()); let newer=vds.tree_head(&leaves);
        let proof=vds.prove(&leaves,2).unwrap().to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Consistency,vec![proof]).unwrap();
        assert_eq!(vdp.verify_consistency_with_payload(older,&newer.root()).unwrap(),newer);
        assert_eq!(vdp.verify_consistency_with_payload(older,&[0xAA;32]),Err(Rfc9942VdpError::NoMatchingProof));
        assert_eq!(vdp.verify_consistency_with_payload(older,&[0xAA;31]),Err(Rfc9942VdpError::InvalidPayloadLength));
    }

    #[test]
    fn rfc9942_vdp_verification_requires_matching_vds_id() {
        let vds=Rfc9162Sha256Vds;
        let leaves=vec![b"a".to_vec(),b"b".to_vec()];
        let head=vds.tree_head(&leaves);
        let proof=vds.inclusion_proof(&leaves,0).unwrap().to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![proof]).unwrap();
        assert_eq!(vdp.verify_inclusion_for_vds(2,b"a",head),Err(Rfc9942VdpError::VdsMismatch(2)));
        assert_eq!(vdp.verify_inclusion_for_vds(1,b"a",head),Ok(head));
    }

    #[test]
    fn rfc9942_vdp_requires_explicit_vds_binding() {
        let inclusion=Rfc9162InclusionProof::new(2,0,vec![[0x11;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![inclusion]).unwrap();
        assert_eq!(vdp.validate_vds_id(RFC9162_VDS_ID),Ok(()));
        assert_eq!(vdp.validate_vds_id(2),Err(Rfc9942VdpError::VdsMismatch(2)));
    }

    #[test]
    fn rfc9942_vdp_can_carry_multiple_proofs() {
        let first=Rfc9162InclusionProof::new(4,1,vec![[0x11;32],[0x22;32]]).to_cbor();
        let second=Rfc9162InclusionProof::new(4,3,vec![[0x33;32],[0x44;32]]).to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![first,second]).unwrap();
        assert_eq!(Rfc9942Vdp::from_cbor(&vdp.to_cbor()).unwrap().proofs().len(),2);
    }

    #[test]
    fn rfc9942_vdp_strict_decoder_rejects_wrong_label_empty_collection_and_trailing_bytes() {
        let inclusion=Rfc9162InclusionProof::new(2,0,vec![[0x11;32]]).to_cbor();
        let mut wrong_label=vec![0xa1,0x22,0x81];
        cbor_bytes(&mut wrong_label,&inclusion);
        assert_eq!(Rfc9942Vdp::from_cbor(&wrong_label),Err(Rfc9942VdpError::InvalidStructure));
        assert_eq!(Rfc9942Vdp::from_cbor(&[0xa1,0x20,0x80]),Err(Rfc9942VdpError::EmptyProofCollection));
        let mut trailing=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![inclusion]).unwrap().to_cbor();
        trailing.push(0);
        assert_eq!(Rfc9942Vdp::from_cbor(&trailing),Err(Rfc9942VdpError::TrailingBytes));
    }

    #[test]
    fn rfc9942_vdp_rejects_noncanonical_negative_label_encoding() {
        let encoded=vec![0xa1,0x38,0x00,0x81,0x40];
        assert_eq!(Rfc9942Vdp::from_cbor(&encoded),Err(Rfc9942VdpError::InvalidEncoding));
    }

    #[test]
    fn rfc9942_vdp_verification_binds_kind_and_tree_head() {
        let vds=Rfc9162Sha256Vds;
        let leaves: Vec<Vec<u8>>=(0..4).map(|i|format!("leaf-{i}").into_bytes()).collect();
        let head=vds.tree_head(&leaves);
        let proof=vds.inclusion_proof(&leaves,2).unwrap().to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion,vec![proof]).unwrap();
        assert_eq!(vdp.verify_inclusion(&leaves[2],head).unwrap(),head);
        assert_eq!(vdp.verify_inclusion(b"tampered",head),Err(Rfc9942VdpError::NoMatchingProof));
        let consistency=Rfc9942Vdp::new(
            Rfc9942ProofKind::Consistency,
            vec![vds.prove(&leaves,2).unwrap().to_cbor()],
        ).unwrap();
        assert_eq!(consistency.verify_inclusion(&leaves[2],head),Err(Rfc9942VdpError::WrongProofKind));
    }

    #[test]
    fn rfc9942_vdp_consistency_verification_accepts_append_only_transition() {
        let vds=Rfc9162Sha256Vds;
        let leaves: Vec<Vec<u8>>=(0..8).map(|i|format!("leaf-{i}").into_bytes()).collect();
        let older=vds.tree_head(&leaves[..4].to_vec());
        let newer=vds.tree_head(&leaves);
        let proof=vds.prove(&leaves,4).unwrap().to_cbor();
        let vdp=Rfc9942Vdp::new(Rfc9942ProofKind::Consistency,vec![proof]).unwrap();
        assert_eq!(vdp.verify_consistency(older,newer).unwrap(),newer);
        let forged=VdsTreeHead::new(newer.tree_size(),[0xAA;32]);
        assert_eq!(vdp.verify_consistency(older,forged),Err(Rfc9942VdpError::NoMatchingProof));
    }

    #[test]
    fn rfc9942_inclusion_verification_binds_candidate_and_tree_head() {
        let vds = Rfc9162Sha256Vds;
        let leaves: Vec<Vec<u8>> = (0..4).map(|i| format!("leaf-{i}").into_bytes()).collect();
        let head = vds.tree_head(&leaves);
        let proof = vds.inclusion_proof(&leaves, 2).expect("proof").to_cbor();
        assert_eq!(
            vds.verify_rfc9942_inclusion_cbor(&leaves[2], head, &proof).unwrap(),
            head
        );
        assert_eq!(
            vds.verify_rfc9942_inclusion_cbor(b"tampered", head, &proof),
            Err(Rfc9162ProofVerificationError::InvalidProof)
        );
        let wrong_size = VdsTreeHead::new(head.tree_size() + 1, head.root());
        assert_eq!(
            vds.verify_rfc9942_inclusion_cbor(&leaves[2], wrong_size, &proof),
            Err(Rfc9162ProofVerificationError::TreeSizeMismatch)
        );
    }

    #[test]
    fn rfc9942_consistency_verification_binds_both_tree_heads() {
        let vds = Rfc9162Sha256Vds;
        let leaves: Vec<Vec<u8>> = (0..8).map(|i| format!("leaf-{i}").into_bytes()).collect();
        let older = vds.tree_head(&leaves[..4].to_vec());
        let newer = vds.tree_head(&leaves);
        let proof = vds.prove(&leaves, 4).expect("proof").to_cbor();
        assert_eq!(
            vds.verify_rfc9942_consistency_cbor(older, newer, &proof).unwrap(),
            newer
        );
        let wrong_root = VdsTreeHead::new(newer.tree_size(), [0xAA; 32]);
        assert_eq!(
            vds.verify_rfc9942_consistency_cbor(older, wrong_root, &proof),
            Err(Rfc9162ProofVerificationError::InvalidProof)
        );
        let wrong_size = VdsTreeHead::new(3, older.root());
        assert_eq!(
            vds.verify_rfc9942_consistency_cbor(wrong_size, newer, &proof),
            Err(Rfc9162ProofVerificationError::TreeSizeMismatch)
        );
    }

    #[test]
    fn rfc9162_inclusion_rejects_out_of_range_leaf_index() {
        let vds = Rfc9162Sha256Vds;
        let leaves = vec![b"a".to_vec(), b"b".to_vec()];
        let proof = Rfc9162InclusionProof::new(2, 2, Vec::new());
        assert!(!vds.verify_inclusion(&leaves[0], vds.root(&leaves), &proof));
    }

    #[test]
    fn rfc9162_consistency_covers_smallest_valid_extension() {
        let vds = Rfc9162Sha256Vds;
        let leaves = vec![b"a".to_vec(), b"b".to_vec()];
        let older = vds.tree_head(&leaves[..1].to_vec());
        let newer = vds.tree_head(&leaves);
        let proof = vds.prove(&leaves, 1).expect("proof");
        assert_eq!(proof.consistency_path.len(), 1);
        assert!(vds.verify_tree_heads(older, newer, &proof));
    }

    #[test]
    fn tree_heads_bind_size_to_root_and_verify_consistency() {
        let vds = Rfc9162Sha256Vds;
        let leaves: Vec<Vec<u8>> = (0..8).map(|i| format!("leaf-{i}").into_bytes()).collect();
        let older = vds.tree_head(&leaves[..4].to_vec());
        let newer = vds.tree_head(&leaves);
        let proof = vds.prove(&leaves, 4).expect("proof");
        assert!(vds.verify_tree_heads(older, newer, &proof));
        assert!(!vds.verify_tree_heads(VdsTreeHead::new(3, older.root()), newer, &proof));
        assert!(!vds.verify_tree_heads(older, VdsTreeHead::new(9, newer.root()), &proof));
    }

    #[test]
    fn generated_consistency_proofs_round_trip() {
        let vds = Rfc9162Sha256Vds;
        for n in 2..=12 {
            let leaves: Vec<Vec<u8>> = (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect();
            let new_root = vds.root(&leaves);
            for first in 1..n {
                let proof = vds.prove(&leaves, first).expect("valid proof request");
                let old_root = vds.root(&leaves[..first].to_vec());
                assert!(vds.verify(old_root, new_root, &proof), "n={n}, first={first}");
            }
        }
    }

    #[test]
    fn generated_inclusion_proofs_round_trip() {
        let vds = Rfc9162Sha256Vds;
        for n in 1..=12 {
            let leaves: Vec<Vec<u8>> = (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect();
            let root = vds.root(&leaves);
            for index in 0..n {
                let proof = vds.inclusion_proof(&leaves, index).expect("valid inclusion request");
                assert!(vds.verify_inclusion(&leaves[index], root, &proof), "n={n}, index={index}");
                let mut tampered = leaves[index].clone();
                tampered.push(b'!');
                assert!(!vds.verify_inclusion(&tampered, root, &proof), "tampered n={n}, index={index}");
            }
        }
    }

    #[test]
    fn evidence_digest_projection_is_domain_separated_and_verifiable() {
        let digest = crate::semantic_evidence_digest::EvidenceDigest([7u8; 32]);
        let leaf = EvidenceVdsLeaf::from_evidence_digest(digest);
        let vds = Rfc9162Sha256Vds;
        let leaves = vec![leaf.as_vec(), b"other".to_vec()];
        let root = vds.root(&leaves);
        let proof = vds.inclusion_proof(&leaves, 0).expect("leaf proof");
        assert!(vds.verify_evidence_inclusion(digest, root, &proof));
        assert_ne!(leaf.as_bytes(), digest.as_bytes());
    }

    #[test]
    fn malformed_consistency_proof_is_rejected() {
        let vds = Rfc9162Sha256Vds;
        let old = vds.root(&[b"a".to_vec()]);
        let new = vds.root(&[b"a".to_vec(), b"b".to_vec()]);
        let proof = Rfc9162ConsistencyProof::new(0, 2, vec![[0; 32]]);
        assert!(!vds.verify(old, new, &proof));
    }

    #[test]
    fn chained_history_remains_explicitly_unsupported() {
        let adapter = ChainedHistoryVds;
        let proof = ConsistencyProof::new(adapter.vds_name(), VERSION, Vec::new());
        let history = crate::semantic_evidence_history::EvidenceHistory::new();
        let checkpoint = history.checkpoint();
        let request = ConsistencyRequest::new(checkpoint, checkpoint);
        assert_eq!(
            adapter.verify_consistency(&request, &proof),
            ConsistencyStatus::Unsupported
        );
    }

    #[test]
    fn unsupported_is_not_invalid() {
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Invalid);
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Valid);
    }
}
