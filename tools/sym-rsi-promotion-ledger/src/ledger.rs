// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent append-only promotion history for adaptive RSI evaluation.
//!
//! This crate is deliberately separate from the editable Symthaea agent. It
//! reuses the production REUSE allocation, paired sign test and confidence
//! certificate, but owns the durable decision history and its signatures.
//!
//! The file format is fixed-width and versioned. Every manifest and event is
//! signed by an externally supplied signer; the event hash chain binds every
//! later decision to the signed manifest and previous event. A partial record,
//! invalid signature, changed experiment context, reset history or stale writer
//! lock fails closed. The ledger stores digests and the coarse round decision,
//! not per-task outcomes or confidence scores.
//!
//! This is a ledger core, not yet the complete evaluator service. The caller
//! must admit candidate receipts, source/build provenance and audit task
//! outcomes through the independent verification owners before passing them
//! here. Process credentials, network isolation and sealed-holdout enforcement
//! remain deployment responsibilities.

use crate::crypto_digest::{Sha256Digest, sha256};
use crate::reuse_error_budget::{
    ReuseCandidateEvidence, ReuseRoundAudit, ReuseRoundDecision, evaluate_reuse_round_audited,
    MAX_REUSE_ROUND,
};
use crate::reuse_confidence_certificate::ReuseConfidenceCertificate;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const MAX_LEDGER_ROUNDS: u64 = MAX_REUSE_ROUND as u64;
const MANIFEST_MAGIC: &[u8; 8] = b"RSIPRM01";
const EVENT_MAGIC: &[u8; 8] = b"RSIEVT01";
const MANIFEST_BODY_LEN: usize = 324;
const MANIFEST_RECORD_LEN: usize = MANIFEST_BODY_LEN + 32 + 64 + 32;
const EVENT_BODY_LEN: usize = 320;
const EVENT_RECORD_LEN: usize = EVENT_BODY_LEN + 32 + 64 + 32;

const MANIFEST_HASH_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.manifest.v1\0";
const MANIFEST_RECORD_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.manifest-record.v1\0";
const EVENT_HASH_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.event.v1\0";
const EVENT_RECORD_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.event-record.v1\0";
const CANDIDATE_ID_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.candidate-id.v1\0";
const CANDIDATE_MANIFEST_DOMAIN: &[u8] =
    b"symthaea.rsi.promotion-ledger.candidate-manifest.v1\0";
const CANDIDATE_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.rsi.promotion-ledger.candidate-evidence.v1\0";
const OUTCOMES_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.paired-outcomes.v1\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.rsi.promotion-ledger.certificate.v1\0";

fn zero_digest() -> Sha256Digest {
    Sha256Digest([0; 32])
}

fn digest_is_zero(value: Sha256Digest) -> bool {
    value.0 == [0; 32]
}

fn hash_domain(domain: &[u8], bytes: &[u8]) -> Sha256Digest {
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(bytes);
    sha256(&input)
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_digest(output: &mut Vec<u8>, value: Sha256Digest) {
    output.extend_from_slice(&value.0);
}

fn take<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    length: usize,
) -> Result<&'a [u8], LedgerError> {
    let end = offset
        .checked_add(length)
        .ok_or(LedgerError::InvalidRecordLength)?;
    let slice = bytes
        .get(*offset..end)
        .ok_or(LedgerError::InvalidRecordLength)?;
    *offset = end;
    Ok(slice)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, LedgerError> {
    let slice = take(bytes, offset, 4)?;
    Ok(u32::from_le_bytes(slice.try_into().expect("four-byte slice")))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, LedgerError> {
    let slice = take(bytes, offset, 8)?;
    Ok(u64::from_le_bytes(slice.try_into().expect("eight-byte slice")))
}

fn read_digest(bytes: &[u8], offset: &mut usize) -> Result<Sha256Digest, LedgerError> {
    let slice = take(bytes, offset, 32)?;
    let mut digest = [0u8; 32];
    digest.copy_from_slice(slice);
    Ok(Sha256Digest(digest))
}

/// Normally implemented by an HSM/KMS adapter owned by the evaluator service.
/// This crate never stores or receives private key material.
pub trait LedgerSigner {
    fn key_id_digest(&self) -> Sha256Digest;
    fn sign_digest(&self, message: &Sha256Digest) -> Result<[u8; 64], String>;
}

/// Must be backed by the independently managed trust owner. A hash chain
/// without signature verification is not an authenticated promotion history.
pub trait LedgerVerifier {
    fn verify_digest(
        &self,
        key_id_digest: Sha256Digest,
        message: &Sha256Digest,
        signature: &[u8; 64],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum LedgerError {
    Io(String),
    LedgerAlreadyExists,
    LedgerNotInitialized,
    WriterLocked,
    InvalidConfig(&'static str),
    InvalidCandidate(&'static str),
    InvalidRecordLength,
    UnsupportedFormat,
    InvalidManifest,
    InvalidSignature,
    SignatureProvider(String),
    SignatureVerifier(String),
    CorruptLedger(&'static str),
    SequenceOverflow,
    RoundLimitExceeded,
    CandidateCountMismatch { expected: usize, actual: usize },
    InvalidCumulativeLowerBound,
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "promotion ledger I/O failed: {reason}"),
            Self::LedgerAlreadyExists => write!(f, "promotion ledger already exists"),
            Self::LedgerNotInitialized => write!(f, "promotion ledger is not initialized"),
            Self::WriterLocked => write!(
                f,
                "promotion ledger has a writer lock; inspect the owner and recover manually"
            ),
            Self::InvalidConfig(field) => write!(f, "invalid frozen ledger config: {field}"),
            Self::InvalidCandidate(field) => write!(f, "invalid candidate evidence: {field}"),
            Self::InvalidRecordLength => write!(f, "ledger record has an invalid byte length"),
            Self::UnsupportedFormat => write!(f, "unsupported promotion ledger format"),
            Self::InvalidManifest => write!(f, "promotion ledger manifest is invalid"),
            Self::InvalidSignature => write!(f, "promotion ledger signature verification failed"),
            Self::SignatureProvider(reason) => write!(f, "ledger signer failed: {reason}"),
            Self::SignatureVerifier(reason) => write!(f, "ledger verifier failed: {reason}"),
            Self::CorruptLedger(reason) => write!(f, "promotion ledger is corrupt: {reason}"),
            Self::SequenceOverflow => write!(f, "promotion ledger sequence overflow"),
            Self::RoundLimitExceeded => write!(f, "maximum reviewed REUSE round limit exceeded"),
            Self::CandidateCountMismatch { expected, actual } => {
                write!(f, "expected {expected} candidate slots, received {actual}")
            }
            Self::InvalidCumulativeLowerBound => {
                write!(f, "cumulative lower bound is non-finite, negative or regressed")
            }
        }
    }
}

impl std::error::Error for LedgerError {}

impl From<std::io::Error> for LedgerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

/// Immutable identity and statistical contract for one adaptive experiment.
/// The signing key and algorithm identities are frozen with the trust snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerConfig {
    experiment_digest: Sha256Digest,
    preregistration_digest: Sha256Digest,
    task_universe_digest: Sha256Digest,
    evaluator_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    service_identity_digest: Sha256Digest,
    initial_incumbent_digest: Sha256Digest,
    signer_key_id_digest: Sha256Digest,
    signature_algorithm_digest: Sha256Digest,
    alpha: f64,
    rho: f64,
    candidates_per_round: u32,
    maximum_rounds: u64,
}

impl LedgerConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        experiment_digest: Sha256Digest,
        preregistration_digest: Sha256Digest,
        task_universe_digest: Sha256Digest,
        evaluator_digest: Sha256Digest,
        trust_snapshot_digest: Sha256Digest,
        service_identity_digest: Sha256Digest,
        initial_incumbent_digest: Sha256Digest,
        signer_key_id_digest: Sha256Digest,
        signature_algorithm_digest: Sha256Digest,
        alpha: f64,
        rho: f64,
        candidates_per_round: u32,
    ) -> Result<Self, LedgerError> {
        let config = Self {
            experiment_digest,
            preregistration_digest,
            task_universe_digest,
            evaluator_digest,
            trust_snapshot_digest,
            service_identity_digest,
            initial_incumbent_digest,
            signer_key_id_digest,
            signature_algorithm_digest,
            alpha,
            rho,
            candidates_per_round,
            maximum_rounds: MAX_LEDGER_ROUNDS,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), LedgerError> {
        for (name, digest) in [
            ("experiment_digest", self.experiment_digest),
            ("preregistration_digest", self.preregistration_digest),
            ("task_universe_digest", self.task_universe_digest),
            ("evaluator_digest", self.evaluator_digest),
            ("trust_snapshot_digest", self.trust_snapshot_digest),
            ("service_identity_digest", self.service_identity_digest),
            ("initial_incumbent_digest", self.initial_incumbent_digest),
            ("signer_key_id_digest", self.signer_key_id_digest),
            ("signature_algorithm_digest", self.signature_algorithm_digest),
        ] {
            if digest_is_zero(digest) {
                return Err(LedgerError::InvalidConfig(name));
            }
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(LedgerError::InvalidConfig("alpha"));
        }
        if !self.rho.is_finite() || self.rho <= 0.0 || self.rho >= 1.0 {
            return Err(LedgerError::InvalidConfig("rho"));
        }
        if self.candidates_per_round == 0 {
            return Err(LedgerError::InvalidConfig("candidates_per_round"));
        }
        if self.maximum_rounds == 0 || self.maximum_rounds > MAX_LEDGER_ROUNDS {
            return Err(LedgerError::InvalidConfig("maximum_rounds"));
        }
        Ok(())
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn rho(&self) -> f64 {
        self.rho
    }

    pub fn candidates_per_round(&self) -> u32 {
        self.candidates_per_round
    }

    pub fn experiment_digest(&self) -> Sha256Digest {
        self.experiment_digest
    }

    pub fn preregistration_digest(&self) -> Sha256Digest {
        self.preregistration_digest
    }

    pub fn task_universe_digest(&self) -> Sha256Digest {
        self.task_universe_digest
    }

    pub fn evaluator_digest(&self) -> Sha256Digest {
        self.evaluator_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
}

/// Identity commitments for one candidate slot. The paired-outcome digest is
/// recomputed from the actual vector before the REUSE selector runs. Receipt and
/// provenance digests must have been admitted by the trusted receipt/provenance
/// adapters before this object is constructed; this crate does not duplicate
/// those verification owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateIdentity {
    candidate_id_digest: Sha256Digest,
    system_identity_digest: Sha256Digest,
    source_tree_digest: Sha256Digest,
    executable_digest: Sha256Digest,
    parent_lineage_digest: Sha256Digest,
    composed_receipt_digest: Sha256Digest,
    evidence_set_digest: Sha256Digest,
    paired_outcomes_digest: Sha256Digest,
}

impl CandidateIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        candidate_id_digest: Sha256Digest,
        system_identity_digest: Sha256Digest,
        source_tree_digest: Sha256Digest,
        executable_digest: Sha256Digest,
        parent_lineage_digest: Sha256Digest,
        composed_receipt_digest: Sha256Digest,
        evidence_set_digest: Sha256Digest,
        paired_outcomes_digest: Sha256Digest,
    ) -> Result<Self, LedgerError> {
        let identity = Self {
            candidate_id_digest,
            system_identity_digest,
            source_tree_digest,
            executable_digest,
            parent_lineage_digest,
            composed_receipt_digest,
            evidence_set_digest,
            paired_outcomes_digest,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), LedgerError> {
        for (name, digest) in [
            ("candidate_id_digest", self.candidate_id_digest),
            ("system_identity_digest", self.system_identity_digest),
            ("source_tree_digest", self.source_tree_digest),
            ("executable_digest", self.executable_digest),
            ("parent_lineage_digest", self.parent_lineage_digest),
            ("composed_receipt_digest", self.composed_receipt_digest),
            ("evidence_set_digest", self.evidence_set_digest),
            ("paired_outcomes_digest", self.paired_outcomes_digest),
        ] {
            if digest_is_zero(digest) {
                return Err(LedgerError::InvalidCandidate(name));
            }
        }
        Ok(())
    }

    pub fn paired_outcomes_digest(paired_differences: &[i8]) -> Result<Sha256Digest, LedgerError> {
        if paired_differences.is_empty() {
            return Err(LedgerError::InvalidCandidate("empty paired outcome vector"));
        }
        if paired_differences.iter().any(|value| !(-1..=1).contains(value)) {
            return Err(LedgerError::InvalidCandidate("paired outcome outside {-1,0,+1}"));
        }
        let bytes: Vec<u8> = paired_differences
            .iter()
            .map(|value| (*value as i16 + 1) as u8)
            .collect();
        Ok(hash_domain(OUTCOMES_DOMAIN, &bytes))
    }

    fn slot_digest(&self) -> Sha256Digest {
        let mut bytes = Vec::with_capacity(8 * 32);
        for digest in [
            self.candidate_id_digest,
            self.system_identity_digest,
            self.source_tree_digest,
            self.executable_digest,
            self.parent_lineage_digest,
            self.composed_receipt_digest,
            self.evidence_set_digest,
            self.paired_outcomes_digest,
        ] {
            push_digest(&mut bytes, digest);
        }
        hash_domain(CANDIDATE_ID_DOMAIN, &bytes)
    }
}

/// Full paired outcomes are private evaluator input and never included in the
/// ledger. Only their digest and the coarse final decision are committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSubmission {
    identity: CandidateIdentity,
    paired_differences: Vec<i8>,
}

impl CandidateSubmission {
    pub fn new(
        identity: CandidateIdentity,
        paired_differences: Vec<i8>,
    ) -> Result<Self, LedgerError> {
        identity.validate()?;
        let actual = CandidateIdentity::paired_outcomes_digest(&paired_differences)?;
        if actual != identity.paired_outcomes_digest {
            return Err(LedgerError::InvalidCandidate(
                "paired outcome digest does not match admitted candidate identity",
            ));
        }
        Ok(Self {
            identity,
            paired_differences,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotionLedgerReport {
    pub rounds: u64,
    pub promotions: u64,
    pub incumbent_digest: Sha256Digest,
    pub cumulative_lower_bound_bits: u64,
    pub head_digest: Sha256Digest,
}

impl PromotionLedgerReport {
    pub fn cumulative_lower_bound(&self) -> f64 {
        f64::from_bits(self.cumulative_lower_bound_bits)
    }

    pub fn intact(&self) -> bool {
        let value = self.cumulative_lower_bound();
        value.is_finite() && value >= 0.0
    }
}

pub type PromotionRoundDecision = ReuseRoundDecision;

/// Independent append-only promotion ledger. Every append reloads and verifies
/// the full chain under an exclusive create-new writer lock; state is never
/// trusted from an in-memory cache.
#[derive(Debug, Clone)]
pub struct PromotionLedger {
    path: PathBuf,
    config: LedgerConfig,
}

impl PromotionLedger {
    /// Create a new ledger directory. Existing directories are never overwritten.
    pub fn initialize(
        path: impl AsRef<Path>,
        config: LedgerConfig,
        signer: &dyn LedgerSigner,
    ) -> Result<Self, LedgerError> {
        config.validate()?;
        if signer.key_id_digest() != config.signer_key_id_digest {
            return Err(LedgerError::InvalidConfig("signer key id does not match config"));
        }

        let path = path.as_ref().to_path_buf();
        fs::create_dir(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                LedgerError::LedgerAlreadyExists
            } else {
                LedgerError::from(error)
            }
        })?;

        let manifest = encode_manifest(&config, signer)?;
        let mut manifest_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("manifest.bin"))?;
        manifest_file.write_all(&manifest)?;
        manifest_file.sync_all()?;

        let events_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("events.bin"))?;
        events_file.sync_all()?;
        sync_directory(&path)?;

        Ok(Self { path, config })
    }

    /// Open and authenticate an existing ledger. A verifier backed by the
    /// independent trust owner is mandatory; hash-chain integrity alone is not
    /// sufficient to authorize further evaluation.
    pub fn open(
        path: impl AsRef<Path>,
        verifier: &dyn LedgerVerifier,
    ) -> Result<Self, LedgerError> {
        let path = path.as_ref().to_path_buf();
        let _lock = WriterLock::acquire(&path)?;
        let loaded = load_verified(&path, verifier)?;
        Ok(Self {
            path,
            config: loaded.config,
        })
    }

    /// Evaluate a complete round from full candidate outcome vectors, persist
    /// the signed decision and return only y_t. The signer/verifier are external
    /// authority adapters. The caller must already have independently validated
    /// source/build and composed receipt digests in every CandidateIdentity.
    ///
    /// This method does not return the selected certificate, per-task outcomes,
    /// scores, p-values or cumulative lower bound.
    pub fn evaluate_and_commit(
        &self,
        timestamp_unix_ms: u64,
        candidates: &[CandidateSubmission],
        signer: &dyn LedgerSigner,
        verifier: &dyn LedgerVerifier,
    ) -> Result<PromotionRoundDecision, LedgerError> {
        if signer.key_id_digest() != self.config.signer_key_id_digest {
            return Err(LedgerError::InvalidConfig("signer key id does not match frozen config"));
        }
        if candidates.len() != self.config.candidates_per_round as usize {
            return Err(LedgerError::CandidateCountMismatch {
                expected: self.config.candidates_per_round as usize,
                actual: candidates.len(),
            });
        }

        let _lock = WriterLock::acquire(&self.path)?;
        let loaded = load_verified(&self.path, verifier)?;
        if loaded.config != self.config {
            return Err(LedgerError::CorruptLedger("frozen manifest changed after open"));
        }
        let state = derive_state(&loaded)?;
        if state.rounds >= self.config.maximum_rounds {
            return Err(LedgerError::RoundLimitExceeded);
        }
        if timestamp_unix_ms < state.last_timestamp_unix_ms {
            return Err(LedgerError::CorruptLedger("timestamp regressed"));
        }

        let mut identities = BTreeSet::new();
        let mut slot_digests = Vec::with_capacity(candidates.len());
        let mut candidate_evidence = Vec::with_capacity(candidates.len());
        let mut evidence_commitment = Vec::with_capacity(candidates.len() * 96);
        for candidate in candidates {
            candidate.identity.validate()?;
            let outcome_digest =
                CandidateIdentity::paired_outcomes_digest(&candidate.paired_differences)?;
            if outcome_digest != candidate.identity.paired_outcomes_digest {
                return Err(LedgerError::InvalidCandidate(
                    "paired outcome digest changed after admission",
                ));
            }
            let slot_digest = candidate.identity.slot_digest();
            if !identities.insert(slot_digest) {
                return Err(LedgerError::InvalidCandidate("duplicate candidate identity"));
            }
            push_digest(&mut evidence_commitment, candidate.identity.evidence_set_digest);
            push_digest(&mut evidence_commitment, candidate.identity.composed_receipt_digest);
            push_digest(&mut evidence_commitment, outcome_digest);
            slot_digests.push(slot_digest);
            candidate_evidence.push(
                ReuseCandidateEvidence::new(candidate.paired_differences.clone())
                    .map_err(|_| LedgerError::InvalidCandidate("REUSE candidate outcome vector"))?,
            );
        }
        let candidate_manifest_digest =
            hash_domain(CANDIDATE_MANIFEST_DOMAIN, &flatten_digests(&slot_digests));
        let candidate_evidence_digest =
            hash_domain(CANDIDATE_EVIDENCE_DOMAIN, &evidence_commitment);

        let audit: ReuseRoundAudit = evaluate_reuse_round_audited(
            self.config.alpha,
            state.rounds as usize + 1,
            state.promotions as usize,
            self.config.candidates_per_round as usize,
            self.config.rho,
            &candidate_evidence,
            state.cumulative_lower_bound,
        )
        .map_err(|_| LedgerError::CorruptLedger("REUSE statistical kernel rejected the round"))?;

        let decision = audit.decision();
        let (selected_candidate_digest, selected_evidence_digest, certificate_digest, next_incumbent) =
            match decision {
                ReuseRoundDecision::RejectAll => (
                    zero_digest(),
                    zero_digest(),
                    zero_digest(),
                    state.incumbent_digest,
                ),
                ReuseRoundDecision::PromoteCandidate { candidate_number } => {
                    let selected = candidates
                        .get(candidate_number - 1)
                        .ok_or(LedgerError::CorruptLedger("selector returned candidate outside K"))?;
                    let selected_identity_digest = selected.identity.slot_digest();
                    let selected_evidence = selected_evidence_digest(&selected.identity);
                    let certificate = audit
                        .selected_certificate()
                        .ok_or(LedgerError::CorruptLedger("promotion has no confidence certificate"))?;
                    let certificate_digest = digest_certificate(certificate);
                    (selected_identity_digest, selected_evidence, certificate_digest, selected_identity_digest)
                }
            };

        let next_cumulative_lower_bound = audit.cumulative_lower_bound();
        if !next_cumulative_lower_bound.is_finite()
            || next_cumulative_lower_bound < state.cumulative_lower_bound
        {
            return Err(LedgerError::InvalidCumulativeLowerBound);
        }

        let body = EventBody {
            sequence: state.rounds + 1,
            round: state.rounds + 1,
            prior_promotions: state.promotions,
            timestamp_unix_ms,
            candidate_count: self.config.candidates_per_round,
            decision_number: decision_number(decision),
            previous_incumbent_digest: state.incumbent_digest,
            next_incumbent_digest: next_incumbent,
            candidate_manifest_digest,
            candidate_evidence_digest,
            selected_candidate_digest,
            selected_evidence_digest,
            selected_certificate_digest: certificate_digest,
            prior_cumulative_lower_bound_bits: state.cumulative_lower_bound.to_bits(),
            next_cumulative_lower_bound_bits: next_cumulative_lower_bound.to_bits(),
            previous_record_hash: state.head_digest,
        };
        let record = sign_event(body, signer)?;
        append_record(&self.path, &record)?;
        sync_directory(&self.path)?;
        Ok(decision)
    }

    /// Verify the chain and return public accounting state. No detailed
    /// evaluation values or certificates are exposed by this report.
    pub fn report(
        &self,
        verifier: &dyn LedgerVerifier,
    ) -> Result<PromotionLedgerReport, LedgerError> {
        let _lock = WriterLock::acquire(&self.path)?;
        let loaded = load_verified(&self.path, verifier)?;
        if loaded.config != self.config {
            return Err(LedgerError::CorruptLedger("frozen manifest changed after open"));
        }
        let state = derive_state(&loaded)?;
        Ok(PromotionLedgerReport {
            rounds: state.rounds,
            promotions: state.promotions,
            incumbent_digest: state.incumbent_digest,
            cumulative_lower_bound_bits: state.cumulative_lower_bound.to_bits(),
            head_digest: state.head_digest,
        })
    }
}

#[derive(Debug, Clone)]
struct EventBody {
    sequence: u64,
    round: u64,
    prior_promotions: u64,
    timestamp_unix_ms: u64,
    candidate_count: u32,
    decision_number: u32,
    previous_incumbent_digest: Sha256Digest,
    next_incumbent_digest: Sha256Digest,
    candidate_manifest_digest: Sha256Digest,
    candidate_evidence_digest: Sha256Digest,
    selected_candidate_digest: Sha256Digest,
    selected_evidence_digest: Sha256Digest,
    selected_certificate_digest: Sha256Digest,
    prior_cumulative_lower_bound_bits: u64,
    next_cumulative_lower_bound_bits: u64,
    previous_record_hash: Sha256Digest,
}

#[derive(Debug, Clone)]
struct SignedEvent {
    body: EventBody,
    record_hash: Sha256Digest,
}

#[derive(Debug, Clone)]
struct LoadedLedger {
    config: LedgerConfig,
    manifest_record_hash: Sha256Digest,
    events: Vec<SignedEvent>,
}

#[derive(Debug, Clone, Copy)]
struct DerivedState {
    rounds: u64,
    promotions: u64,
    incumbent_digest: Sha256Digest,
    cumulative_lower_bound: f64,
    last_timestamp_unix_ms: u64,
    head_digest: Sha256Digest,
}

fn encode_config_body(config: &LedgerConfig) -> Vec<u8> {
    let mut output = Vec::with_capacity(MANIFEST_BODY_LEN);
    output.extend_from_slice(MANIFEST_MAGIC);
    push_u32(&mut output, 1);
    push_u64(&mut output, config.alpha.to_bits());
    push_u64(&mut output, config.rho.to_bits());
    push_u32(&mut output, config.candidates_per_round);
    push_u32(&mut output, config.maximum_rounds as u32);
    for digest in [
        config.experiment_digest,
        config.preregistration_digest,
        config.task_universe_digest,
        config.evaluator_digest,
        config.trust_snapshot_digest,
        config.service_identity_digest,
        config.initial_incumbent_digest,
        config.signer_key_id_digest,
        config.signature_algorithm_digest,
    ] {
        push_digest(&mut output, digest);
    }
    output
}

fn encode_manifest(
    config: &LedgerConfig,
    signer: &dyn LedgerSigner,
) -> Result<Vec<u8>, LedgerError> {
    let body = encode_config_body(config);
    if body.len() != MANIFEST_BODY_LEN {
        return Err(LedgerError::InvalidManifest);
    }
    let manifest_hash = hash_domain(MANIFEST_HASH_DOMAIN, &body);
    let signature = signer
        .sign_digest(&manifest_hash)
        .map_err(LedgerError::SignatureProvider)?;
    let record_hash = hash_signed_record(
        MANIFEST_RECORD_DOMAIN,
        manifest_hash,
        config.signer_key_id_digest,
        &signature,
    );
    let mut encoded = body;
    push_digest(&mut encoded, manifest_hash);
    encoded.extend_from_slice(&signature);
    push_digest(&mut encoded, record_hash);
    Ok(encoded)
}

fn decode_and_verify_manifest(
    bytes: &[u8],
    verifier: &dyn LedgerVerifier,
) -> Result<(LedgerConfig, Sha256Digest), LedgerError> {
    if bytes.len() != MANIFEST_RECORD_LEN {
        return Err(LedgerError::InvalidManifest);
    }
    let body = &bytes[..MANIFEST_BODY_LEN];
    if &body[..8] != MANIFEST_MAGIC {
        return Err(LedgerError::UnsupportedFormat);
    }
    let mut offset = 8;
    if read_u32(body, &mut offset)? != 1 {
        return Err(LedgerError::UnsupportedFormat);
    }
    let alpha = f64::from_bits(read_u64(body, &mut offset)?);
    let rho = f64::from_bits(read_u64(body, &mut offset)?);
    let candidates_per_round = read_u32(body, &mut offset)?;
    let maximum_rounds = read_u32(body, &mut offset)? as u64;
    let config = LedgerConfig {
        experiment_digest: read_digest(body, &mut offset)?,
        preregistration_digest: read_digest(body, &mut offset)?,
        task_universe_digest: read_digest(body, &mut offset)?,
        evaluator_digest: read_digest(body, &mut offset)?,
        trust_snapshot_digest: read_digest(body, &mut offset)?,
        service_identity_digest: read_digest(body, &mut offset)?,
        initial_incumbent_digest: read_digest(body, &mut offset)?,
        signer_key_id_digest: read_digest(body, &mut offset)?,
        signature_algorithm_digest: read_digest(body, &mut offset)?,
        alpha,
        rho,
        candidates_per_round,
        maximum_rounds,
    };
    if offset != MANIFEST_BODY_LEN {
        return Err(LedgerError::InvalidManifest);
    }
    config.validate()?;

    let mut offset = MANIFEST_BODY_LEN;
    let manifest_hash = read_digest(bytes, &mut offset)?;
    let signature_slice = take(bytes, &mut offset, 64)?;
    let mut signature = [0u8; 64];
    signature.copy_from_slice(signature_slice);
    let record_hash = read_digest(bytes, &mut offset)?;
    if offset != bytes.len() {
        return Err(LedgerError::InvalidManifest);
    }
    if hash_domain(MANIFEST_HASH_DOMAIN, body) != manifest_hash {
        return Err(LedgerError::InvalidManifest);
    }
    let valid = verifier
        .verify_digest(config.signer_key_id_digest, &manifest_hash, &signature)
        .map_err(LedgerError::SignatureVerifier)?;
    if !valid
        || hash_signed_record(
            MANIFEST_RECORD_DOMAIN,
            manifest_hash,
            config.signer_key_id_digest,
            &signature,
        ) != record_hash
    {
        return Err(LedgerError::InvalidSignature);
    }
    Ok((config, record_hash))
}

fn encode_event_body(body: &EventBody) -> Vec<u8> {
    let mut output = Vec::with_capacity(EVENT_BODY_LEN);
    output.extend_from_slice(EVENT_MAGIC);
    push_u64(&mut output, body.sequence);
    push_u64(&mut output, body.round);
    push_u64(&mut output, body.prior_promotions);
    push_u64(&mut output, body.timestamp_unix_ms);
    push_u32(&mut output, body.candidate_count);
    push_u32(&mut output, body.decision_number);
    for digest in [
        body.previous_incumbent_digest,
        body.next_incumbent_digest,
        body.candidate_manifest_digest,
        body.candidate_evidence_digest,
        body.selected_candidate_digest,
        body.selected_evidence_digest,
        body.selected_certificate_digest,
    ] {
        push_digest(&mut output, digest);
    }
    push_u64(&mut output, body.prior_cumulative_lower_bound_bits);
    push_u64(&mut output, body.next_cumulative_lower_bound_bits);
    push_digest(&mut output, body.previous_record_hash);
    output
}

fn decode_event_body(bytes: &[u8]) -> Result<EventBody, LedgerError> {
    if bytes.len() != EVENT_BODY_LEN || &bytes[..8] != EVENT_MAGIC {
        return Err(LedgerError::InvalidRecordLength);
    }
    let mut offset = 8;
    let body = EventBody {
        sequence: read_u64(bytes, &mut offset)?,
        round: read_u64(bytes, &mut offset)?,
        prior_promotions: read_u64(bytes, &mut offset)?,
        timestamp_unix_ms: read_u64(bytes, &mut offset)?,
        candidate_count: read_u32(bytes, &mut offset)?,
        decision_number: read_u32(bytes, &mut offset)?,
        previous_incumbent_digest: read_digest(bytes, &mut offset)?,
        next_incumbent_digest: read_digest(bytes, &mut offset)?,
        candidate_manifest_digest: read_digest(bytes, &mut offset)?,
        candidate_evidence_digest: read_digest(bytes, &mut offset)?,
        selected_candidate_digest: read_digest(bytes, &mut offset)?,
        selected_evidence_digest: read_digest(bytes, &mut offset)?,
        selected_certificate_digest: read_digest(bytes, &mut offset)?,
        prior_cumulative_lower_bound_bits: read_u64(bytes, &mut offset)?,
        next_cumulative_lower_bound_bits: read_u64(bytes, &mut offset)?,
        previous_record_hash: read_digest(bytes, &mut offset)?,
    };
    if offset != EVENT_BODY_LEN {
        return Err(LedgerError::InvalidRecordLength);
    }
    Ok(body)
}

fn sign_event(body: EventBody, signer: &dyn LedgerSigner) -> Result<SignedEvent, LedgerError> {
    let bytes = encode_event_body(&body);
    let body_hash = hash_domain(EVENT_HASH_DOMAIN, &bytes);
    let signature = signer
        .sign_digest(&body_hash)
        .map_err(LedgerError::SignatureProvider)?;
    let record_hash = hash_signed_record(
        EVENT_RECORD_DOMAIN,
        body_hash,
        signer.key_id_digest(),
        &signature,
    );
    Ok(SignedEvent { body, record_hash })
}

fn encode_event_record(record: &SignedEvent, signer: &dyn LedgerSigner) -> Result<Vec<u8>, LedgerError> {
    let body = encode_event_body(&record.body);
    let body_hash = hash_domain(EVENT_HASH_DOMAIN, &body);
    // The event was already signed; obtain the same deterministic signature
    // from the signer to encode it. The signer contract must be deterministic.
    let signature = signer
        .sign_digest(&body_hash)
        .map_err(LedgerError::SignatureProvider)?;
    let record_hash = hash_signed_record(
        EVENT_RECORD_DOMAIN,
        body_hash,
        signer.key_id_digest(),
        &signature,
    );
    if record_hash != record.record_hash {
        return Err(LedgerError::SignatureProvider(
            "signer returned a non-deterministic signature".into(),
        ));
    }
    let mut output = body;
    push_digest(&mut output, body_hash);
    output.extend_from_slice(&signature);
    push_digest(&mut output, record_hash);
    Ok(output)
}

fn decode_and_verify_event(
    bytes: &[u8],
    config: &LedgerConfig,
    verifier: &dyn LedgerVerifier,
) -> Result<SignedEvent, LedgerError> {
    if bytes.len() != EVENT_RECORD_LEN {
        return Err(LedgerError::InvalidRecordLength);
    }
    let body_bytes = &bytes[..EVENT_BODY_LEN];
    let body = decode_event_body(body_bytes)?;
    let mut offset = EVENT_BODY_LEN;
    let body_hash = read_digest(bytes, &mut offset)?;
    let signature_slice = take(bytes, &mut offset, 64)?;
    let mut signature = [0u8; 64];
    signature.copy_from_slice(signature_slice);
    let record_hash = read_digest(bytes, &mut offset)?;
    if offset != bytes.len() {
        return Err(LedgerError::InvalidRecordLength);
    }
    if hash_domain(EVENT_HASH_DOMAIN, body_bytes) != body_hash {
        return Err(LedgerError::CorruptLedger("event body hash mismatch"));
    }
    let valid = verifier
        .verify_digest(config.signer_key_id_digest, &body_hash, &signature)
        .map_err(LedgerError::SignatureVerifier)?;
    if !valid
        || hash_signed_record(
            EVENT_RECORD_DOMAIN,
            body_hash,
            config.signer_key_id_digest,
            &signature,
        ) != record_hash
    {
        return Err(LedgerError::InvalidSignature);
    }
    Ok(SignedEvent { body, record_hash })
}

fn hash_signed_record(
    domain: &[u8],
    body_hash: Sha256Digest,
    key_id_digest: Sha256Digest,
    signature: &[u8; 64],
) -> Sha256Digest {
    let mut bytes = Vec::with_capacity(128);
    push_digest(&mut bytes, body_hash);
    push_digest(&mut bytes, key_id_digest);
    bytes.extend_from_slice(signature);
    hash_domain(domain, &bytes)
}

fn load_verified(
    path: &Path,
    verifier: &dyn LedgerVerifier,
) -> Result<LoadedLedger, LedgerError> {
    let manifest_bytes = fs::read(path.join("manifest.bin")).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            LedgerError::LedgerNotInitialized
        } else {
            LedgerError::from(error)
        }
    })?;
    let (config, manifest_record_hash) =
        decode_and_verify_manifest(&manifest_bytes, verifier)?;
    let event_bytes = fs::read(path.join("events.bin")).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            LedgerError::LedgerNotInitialized
        } else {
            LedgerError::from(error)
        }
    })?;
    if event_bytes.len() % EVENT_RECORD_LEN != 0 {
        return Err(LedgerError::CorruptLedger("truncated or partial event record"));
    }
    let mut events = Vec::with_capacity(event_bytes.len() / EVENT_RECORD_LEN);
    for chunk in event_bytes.chunks_exact(EVENT_RECORD_LEN) {
        events.push(decode_and_verify_event(chunk, &config, verifier)?);
    }
    let loaded = LoadedLedger {
        config,
        manifest_record_hash,
        events,
    };
    verify_event_transitions(&loaded)?;
    Ok(loaded)
}

fn verify_event_transitions(loaded: &LoadedLedger) -> Result<(), LedgerError> {
    let mut promotions = 0u64;
    let mut incumbent = loaded.config.initial_incumbent_digest;
    let mut cumulative = 0.0_f64;
    let mut timestamp = 0u64;
    let mut previous_hash = loaded.manifest_record_hash;

    if loaded.events.len() as u64 > loaded.config.maximum_rounds {
        return Err(LedgerError::RoundLimitExceeded);
    }
    for (index, record) in loaded.events.iter().enumerate() {
        let body = &record.body;
        let expected = index as u64 + 1;
        if body.sequence != expected || body.round != expected {
            return Err(LedgerError::CorruptLedger("round or event sequence mismatch"));
        }
        if body.prior_promotions != promotions {
            return Err(LedgerError::CorruptLedger("promotion history mismatch"));
        }
        if body.candidate_count != loaded.config.candidates_per_round
            || body.decision_number > loaded.config.candidates_per_round
        {
            return Err(LedgerError::CorruptLedger("candidate count or decision is out of range"));
        }
        if body.timestamp_unix_ms < timestamp {
            return Err(LedgerError::CorruptLedger("timestamp regressed"));
        }
        if body.previous_record_hash != previous_hash {
            return Err(LedgerError::CorruptLedger("event hash chain is broken"));
        }
        if body.previous_incumbent_digest != incumbent {
            return Err(LedgerError::CorruptLedger("incumbent lineage is broken"));
        }
        if f64::from_bits(body.prior_cumulative_lower_bound_bits).to_bits() != cumulative.to_bits() {
            return Err(LedgerError::CorruptLedger("cumulative lower bound chain is broken"));
        }
        if digest_is_zero(body.candidate_manifest_digest)
            || digest_is_zero(body.candidate_evidence_digest)
        {
            return Err(LedgerError::CorruptLedger("candidate/evidence commitment is missing"));
        }

        let next_cumulative = f64::from_bits(body.next_cumulative_lower_bound_bits);
        if !next_cumulative.is_finite() || next_cumulative < cumulative || next_cumulative < 0.0 {
            return Err(LedgerError::InvalidCumulativeLowerBound);
        }
        if body.decision_number == 0 {
            if !digest_is_zero(body.selected_candidate_digest)
                || !digest_is_zero(body.selected_evidence_digest)
                || !digest_is_zero(body.selected_certificate_digest)
                || body.next_incumbent_digest != incumbent
                || next_cumulative.to_bits() != cumulative.to_bits()
            {
                return Err(LedgerError::CorruptLedger("rejection changed promotion state"));
            }
        } else {
            if digest_is_zero(body.selected_candidate_digest)
                || digest_is_zero(body.selected_evidence_digest)
                || digest_is_zero(body.selected_certificate_digest)
                || body.next_incumbent_digest != body.selected_candidate_digest
            {
                return Err(LedgerError::CorruptLedger("promotion lacks bound candidate/certificate"));
            }
            promotions = promotions.checked_add(1).ok_or(LedgerError::SequenceOverflow)?;
        }

        incumbent = body.next_incumbent_digest;
        cumulative = next_cumulative;
        timestamp = body.timestamp_unix_ms;
        previous_hash = record.record_hash;
    }
    Ok(())
}

fn derive_state(loaded: &LoadedLedger) -> Result<DerivedState, LedgerError> {
    verify_event_transitions(loaded)?;
    let promotions = loaded
        .events
        .iter()
        .filter(|record| record.body.decision_number > 0)
        .count() as u64;
    let last = loaded.events.last();
    Ok(DerivedState {
        rounds: loaded.events.len() as u64,
        promotions,
        incumbent_digest: last
            .map(|record| record.body.next_incumbent_digest)
            .unwrap_or(loaded.config.initial_incumbent_digest),
        cumulative_lower_bound: last
            .map(|record| f64::from_bits(record.body.next_cumulative_lower_bound_bits))
            .unwrap_or(0.0),
        last_timestamp_unix_ms: last.map(|record| record.body.timestamp_unix_ms).unwrap_or(0),
        head_digest: last
            .map(|record| record.record_hash)
            .unwrap_or(loaded.manifest_record_hash),
    })
}

fn append_record(
    path: &Path,
    record: &SignedEvent,
    signer: &dyn LedgerSigner,
) -> Result<(), LedgerError> {
    let mut file = OpenOptions::new()
        .append(true)
        .write(true)
        .open(path.join("events.bin"))?;
    file.write_all(&encode_event_record(record, signer)?)?;
    file.sync_all()?;
    Ok(())
}

struct WriterLock {
    path: PathBuf,
}

impl WriterLock {
    fn acquire(ledger_path: &Path) -> Result<Self, LedgerError> {
        let path = ledger_path.join("writer.lock");
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(LedgerError::WriterLocked);
            }
            Err(error) => return Err(error.into()),
        };
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        sync_directory(ledger_path)?;
        Ok(Self { path })
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // A stale lock after a crash is deliberately not removed by startup.
        // Recovery is an explicit operator action after inspecting the owner.
        let _ = fs::remove_file(&self.path);
        if let Some(parent) = self.path.parent() {
            let _ = sync_directory(parent);
        }
    }
}

fn sync_directory(path: &Path) -> Result<(), LedgerError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn flatten_digests(values: &[Sha256Digest]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for value in values {
        push_digest(&mut bytes, *value);
    }
    bytes
}

fn selected_evidence_digest(identity: &CandidateIdentity) -> Sha256Digest {
    let mut bytes = Vec::with_capacity(96);
    push_digest(&mut bytes, identity.evidence_set_digest);
    push_digest(&mut bytes, identity.composed_receipt_digest);
    push_digest(&mut bytes, identity.paired_outcomes_digest);
    hash_domain(CANDIDATE_EVIDENCE_DOMAIN, &bytes)
}

fn decision_number(decision: ReuseRoundDecision) -> u32 {
    match decision {
        ReuseRoundDecision::RejectAll => 0,
        ReuseRoundDecision::PromoteCandidate { candidate_number } => candidate_number as u32,
    }
}

fn digest_certificate(certificate: &ReuseConfidenceCertificate) -> Sha256Digest {
    let mut bytes = Vec::with_capacity(11 * 8);
    push_u64(&mut bytes, certificate.sample_count() as u64);
    push_u64(&mut bytes, certificate.candidate_wins() as u64);
    push_u64(&mut bytes, certificate.incumbent_wins() as u64);
    push_u64(&mut bytes, certificate.discordant_count() as u64);
    push_u64(&mut bytes, certificate.empirical_improvement().to_bits());
    push_u64(&mut bytes, certificate.conditional_win_lower().to_bits());
    push_u64(&mut bytes, certificate.discordance_rate_lower().to_bits());
    push_u64(&mut bytes, certificate.sign_lower_bound().to_bits());
    push_u64(&mut bytes, certificate.empirical_bernstein_lower_bound().to_bits());
    push_u64(&mut bytes, certificate.lower_bound().to_bits());
    push_u64(&mut bytes, certificate.cumulative_increment().to_bits());
    hash_domain(CERTIFICATE_DOMAIN, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    struct TestSigner {
        key: [u8; 32],
        key_id: Sha256Digest,
    }

    impl TestSigner {
        fn new() -> Self {
            let key = [0x5a; 32];
            Self {
                key,
                key_id: sha256(&key),
            }
        }

        fn signature(&self, message: &Sha256Digest) -> [u8; 64] {
            let mut left = Vec::from(self.key);
            push_digest(&mut left, *message);
            let mut right = Vec::from(message.0);
            right.extend_from_slice(&self.key);
            let mut signature = [0u8; 64];
            signature[..32].copy_from_slice(&sha256(&left).0);
            signature[32..].copy_from_slice(&sha256(&right).0);
            signature
        }
    }

    impl LedgerSigner for TestSigner {
        fn key_id_digest(&self) -> Sha256Digest {
            self.key_id
        }

        fn sign_digest(&self, message: &Sha256Digest) -> Result<[u8; 64], String> {
            Ok(self.signature(message))
        }
    }

    impl LedgerVerifier for TestSigner {
        fn verify_digest(
            &self,
            key_id_digest: Sha256Digest,
            message: &Sha256Digest,
            signature: &[u8; 64],
        ) -> Result<bool, String> {
            Ok(key_id_digest == self.key_id && *signature == self.signature(message))
        }
    }

    fn digest(label: &[u8]) -> Sha256Digest {
        sha256(label)
    }

    fn config(signer: &TestSigner) -> LedgerConfig {
        LedgerConfig::new(
            digest(b"experiment-1"),
            digest(b"preregistration-1"),
            digest(b"task-universe-1"),
            digest(b"evaluator-1"),
            digest(b"trust-snapshot-1"),
            digest(b"service-identity-1"),
            digest(b"incumbent-0"),
            signer.key_id_digest(),
            digest(b"Ed25519"),
            0.05,
            0.5,
            2,
        )
        .unwrap()
    }

    fn temp_dir() -> PathBuf {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sym-rsi-promotion-ledger-{}-{sequence}",
            std::process::id()
        ))
    }

    fn candidate(id: &[u8], outcomes: Vec<i8>) -> CandidateSubmission {
        let outcomes_digest = CandidateIdentity::paired_outcomes_digest(&outcomes).unwrap();
        let identity = CandidateIdentity::new(
            digest(id),
            digest(b"system-identity"),
            digest(b"source-tree"),
            digest(b"executable"),
            digest(b"parent-lineage"),
            digest(b"composed-receipt"),
            digest(b"evidence-set"),
            outcomes_digest,
        )
        .unwrap();
        CandidateSubmission::new(identity, outcomes).unwrap()
    }

    #[test]
    fn initializes_and_opens_signed_empty_ledger() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let reopened = PromotionLedger::open(&path, &signer).unwrap();
        let report = reopened.report(&signer).unwrap();
        assert_eq!(report.rounds, 0);
        assert_eq!(report.promotions, 0);
        assert_eq!(report.incumbent_digest, digest(b"incumbent-0"));
        assert_eq!(report.cumulative_lower_bound(), 0.0);
        assert!(report.intact());
        drop(ledger);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn promotion_is_derived_from_full_outcomes_and_persisted_as_one_based_decision() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let candidate_a = candidate(b"candidate-a", vec![1; 32]);
        let candidate_b = candidate(b"candidate-b", vec![-1; 32]);

        let decision = ledger
            .evaluate_and_commit(100, &[candidate_a, candidate_b], &signer, &signer)
            .unwrap();
        assert_eq!(
            decision,
            ReuseRoundDecision::PromoteCandidate {
                candidate_number: 1
            }
        );

        let report = PromotionLedger::open(&path, &signer)
            .unwrap()
            .report(&signer)
            .unwrap();
        assert_eq!(report.rounds, 1);
        assert_eq!(report.promotions, 1);
        assert_ne!(report.incumbent_digest, digest(b"incumbent-0"));
        assert!(report.cumulative_lower_bound() > 0.0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejection_preserves_incumbent_and_cumulative_lower_bound() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let candidates = [
            candidate(b"candidate-a", vec![0; 32]),
            candidate(b"candidate-b", vec![0; 32]),
        ];
        assert_eq!(
            ledger
                .evaluate_and_commit(100, &candidates, &signer, &signer)
                .unwrap(),
            ReuseRoundDecision::RejectAll
        );
        let report = ledger.report(&signer).unwrap();
        assert_eq!(report.rounds, 1);
        assert_eq!(report.promotions, 0);
        assert_eq!(report.incumbent_digest, digest(b"incumbent-0"));
        assert_eq!(report.cumulative_lower_bound(), 0.0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_candidate_outcome_digest_mismatch_before_evaluation() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let identity = CandidateIdentity::new(
            digest(b"candidate-a"),
            digest(b"system"),
            digest(b"source"),
            digest(b"binary"),
            digest(b"lineage"),
            digest(b"receipt"),
            digest(b"evidence"),
            digest(b"not-the-outcomes"),
        )
        .unwrap();
        assert!(matches!(
            CandidateSubmission::new(identity, vec![1; 32]),
            Err(LedgerError::InvalidCandidate(_))
        ));
        assert_eq!(ledger.report(&signer).unwrap().rounds, 0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_duplicate_candidate_identity_in_one_round() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let one = candidate(b"same", vec![1; 32]);
        let two = one.clone();
        assert!(matches!(
            ledger.evaluate_and_commit(100, &[one, two], &signer, &signer),
            Err(LedgerError::InvalidCandidate("duplicate candidate identity"))
        ));
        assert_eq!(ledger.report(&signer).unwrap().rounds, 0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn refuses_stale_writer_lock_without_removing_it() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        fs::write(path.join("writer.lock"), b"pid=unknown\n").unwrap();
        assert_eq!(ledger.report(&signer), Err(LedgerError::WriterLocked));
        assert!(path.join("writer.lock").exists());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn detects_partial_trailing_event_record() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        drop(ledger);
        let mut events = OpenOptions::new()
            .append(true)
            .open(path.join("events.bin"))
            .unwrap();
        events.write_all(b"partial").unwrap();
        events.sync_all().unwrap();
        assert!(matches!(
            PromotionLedger::open(&path, &signer),
            Err(LedgerError::CorruptLedger("truncated or partial event record"))
        ));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn detects_event_tampering_and_signature_failure() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let candidates = [
            candidate(b"candidate-a", vec![1; 32]),
            candidate(b"candidate-b", vec![-1; 32]),
        ];
        ledger
            .evaluate_and_commit(100, &candidates, &signer, &signer)
            .unwrap();
        let mut bytes = fs::read(path.join("events.bin")).unwrap();
        bytes[80] ^= 0x01;
        fs::write(path.join("events.bin"), bytes).unwrap();
        assert!(PromotionLedger::open(&path, &signer).is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_manifest_signature_from_wrong_trust_owner() {
        let signer = TestSigner::new();
        let wrong = TestSigner {
            key: [0xa5; 32],
            key_id: digest(b"wrong-key-id"),
        };
        let path = temp_dir();
        PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        assert!(matches!(
            PromotionLedger::open(&path, &wrong),
            Err(LedgerError::InvalidSignature)
        ));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn append_history_derives_promotion_count_and_round_index() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let first = [
            candidate(b"candidate-a", vec![0; 32]),
            candidate(b"candidate-b", vec![0; 32]),
        ];
        assert_eq!(
            ledger.evaluate_and_commit(100, &first, &signer, &signer).unwrap(),
            ReuseRoundDecision::RejectAll
        );
        let second = [
            candidate(b"candidate-c", vec![1; 64]),
            candidate(b"candidate-d", vec![-1; 64]),
        ];
        assert_eq!(
            ledger.evaluate_and_commit(101, &second, &signer, &signer).unwrap(),
            ReuseRoundDecision::PromoteCandidate {
                candidate_number: 1
            }
        );
        let report = PromotionLedger::open(&path, &signer).unwrap().report(&signer).unwrap();
        assert_eq!(report.rounds, 2);
        assert_eq!(report.promotions, 1);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_regressing_timestamp_and_keeps_prior_head() {
        let signer = TestSigner::new();
        let path = temp_dir();
        let ledger = PromotionLedger::initialize(&path, config(&signer), &signer).unwrap();
        let candidates = [
            candidate(b"candidate-a", vec![0; 32]),
            candidate(b"candidate-b", vec![0; 32]),
        ];
        ledger
            .evaluate_and_commit(100, &candidates, &signer, &signer)
            .unwrap();
        assert!(matches!(
            ledger.evaluate_and_commit(99, &candidates, &signer, &signer),
            Err(LedgerError::CorruptLedger("timestamp regressed"))
        ));
        assert_eq!(ledger.report(&signer).unwrap().rounds, 1);
        fs::remove_dir_all(path).unwrap();
    }
}
