//! As-of evidence replay and filtration-admission receipts for connectomics.
//!
//! The operational filtration is ordered by global availability sequence,
//! never by biological event time. The verifier audits chronology and
//! receipt-binding semantics. It does not authenticate artifacts or prove
//! conditional supermartingale validity / an adjuster theorem.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditError(pub String);

impl fmt::Display for AuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for AuditError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn decode_sha256(value: &str) -> Result<[u8; 32], AuditError> {
    if !valid_sha256(value) {
        return Err(AuditError("malformed SHA-256 digest".into()));
    }
    let mut decoded = [0_u8; 32];
    for (index, slot) in decoded.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| AuditError("malformed SHA-256 digest".into()))?;
    }
    Ok(decoded)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Deterministic binary writer: big-endian fixed-width integers and u64
/// length-prefixed UTF-8 strings/byte sequences. Never uses JSON or Debug.
#[derive(Default)]
struct CanonicalWriter(Vec<u8>);

impl CanonicalWriter {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), AuditError> {
        let length = u64::try_from(value.len())
            .map_err(|_| AuditError("canonical byte field exceeds u64 length".into()))?;
        self.u64(length);
        self.0.extend_from_slice(value);
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), AuditError> {
        self.bytes(value.as_bytes())
    }

    fn digest(&mut self, value: &str) -> Result<(), AuditError> {
        self.0.extend_from_slice(&decode_sha256(value)?);
        Ok(())
    }

    fn optional_string(&mut self, value: Option<&str>) -> Result<(), AuditError> {
        match value {
            Some(value) => {
                self.u8(1);
                self.string(value)
            }
            None => {
                self.u8(0);
                Ok(())
            }
        }
    }

    fn optional_i64(&mut self, value: Option<i64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.i64(value);
            }
            None => self.u8(0),
        }
    }

    fn strings(&mut self, values: &[String]) -> Result<(), AuditError> {
        self.u64(
            u64::try_from(values.len())
                .map_err(|_| AuditError("canonical list exceeds u64 length".into()))?,
        );
        for value in values {
            self.string(value)?;
        }
        Ok(())
    }
}

const LEDGER_SCHEMA_VERSION: u32 = 1;
const LEDGER_HASH_SUITE: &str = "SHA-256";
const LEDGER_GENESIS_DOMAIN: &[u8] = b"SYTHAEA/CONNECTOME/ASOF-LEDGER-GENESIS/V1\0";
const LEDGER_ENTRY_DOMAIN: &[u8] = b"SYTHAEA/CONNECTOME/ASOF-LEDGER-ENTRY/V1\0";
const INPUT_PREFIX_DOMAIN: &[u8] = b"SYTHAEA/CONNECTOME/ASOF-INPUT-PREFIX/V1\0";
const HISTORY_PREFIX_DOMAIN: &[u8] = b"SYTHAEA/CONNECTOME/ASOF-HISTORY-PREFIX/V1\0";
const CHECKPOINT_DOMAIN: &[u8] = b"SYTHAEA/CONNECTOME/ASOF-CHECKPOINT/V1\0";

fn canonical_payload(payload: &LedgerPayload) -> Result<(u8, Vec<u8>), AuditError> {
    let mut writer = CanonicalWriter::default();
    match payload {
        LedgerPayload::Evidence(event) => {
            writer.u8(1);
            writer.string(&event.event_id)?;
            writer.string(&event.evidence_family_id)?;
            writer.string(&event.source_stream)?;
            writer.digest(&event.source_artifact_sha256)?;
            writer.u32(event.revision);
            writer.optional_string(event.supersedes_event_id.as_deref())?;
            writer.optional_i64(event.event_time_ms);
            writer.optional_i64(event.acquired_at_ms);
            writer.optional_i64(event.annotated_at_ms);
            writer.optional_i64(event.qc_at_ms);
            writer.u8(match event.availability_state {
                AvailabilityState::Approved => 0,
                AvailabilityState::Provisional => 1,
                AvailabilityState::Disputed => 2,
                AvailabilityState::Failed => 3,
                AvailabilityState::Missing => 4,
                AvailabilityState::OutOfSupport => 5,
                AvailabilityState::NotAssessed => 6,
            });
            writer.u8(match event.split {
                EvidenceSplit::Exploration => 0,
                EvidenceSplit::Fit => 1,
                EvidenceSplit::Calibration => 2,
                EvidenceSplit::Confirmation => 3,
            });
            writer.optional_string(event.confirmatory_family_id.as_deref())?;
            Ok((1, writer.0))
        }
        LedgerPayload::FamilyFreeze(freeze) => {
            writer.u8(2);
            writer.string(&freeze.family_id)?;
            writer.digest(&freeze.family_commitment_sha256)?;
            writer.u8(u8::from(freeze.confirmation_lockbox_closed));
            Ok((2, writer.0))
        }
        LedgerPayload::Decision(decision) => {
            writer.u8(3);
            writer.string(&decision.decision_id)?;
            writer.string(&decision.campaign_id)?;
            writer.u8(match decision.kind {
                DecisionKind::Discovery => 0,
                DecisionKind::Confirmation => 1,
            });
            writer.digest(&decision.policy_version_sha256)?;
            writer.digest(&decision.input_prefix_sha256)?;
            writer.digest(&decision.driver_receipt_sha256)?;
            writer.digest(&decision.history_prefix_sha256)?;
            writer.digest(&decision.action_sha256)?;
            writer.digest(&decision.feasibility_receipt_sha256)?;
            writer.strings(&decision.available_event_ids)?;
            writer.strings(&decision.consumed_event_ids)?;
            writer.optional_string(decision.confirmatory_family_id.as_deref())?;
            Ok((3, writer.0))
        }
    }
}

fn genesis_root(ledger: &GlobalLedger) -> Result<String, AuditError> {
    let mut writer = CanonicalWriter::default();
    writer.u32(LEDGER_SCHEMA_VERSION);
    writer.string(LEDGER_HASH_SUITE)?;
    writer.string(&ledger.campaign_id)?;
    writer.string(&ledger.global_filtration_id)?;
    writer.digest(&ledger.global_filtration_sha256)?;
    writer.string(&ledger.ledger_epoch_id)?;
    let mut preimage = LEDGER_GENESIS_DOMAIN.to_vec();
    preimage.extend_from_slice(&writer.0);
    Ok(sha256_hex(&preimage))
}

fn entry_root(
    ledger: &GlobalLedger,
    sequence: u64,
    previous_root: &str,
    payload: &LedgerPayload,
) -> Result<String, AuditError> {
    let (payload_kind, payload_bytes) = canonical_payload(payload)?;
    let mut writer = CanonicalWriter::default();
    writer.u32(LEDGER_SCHEMA_VERSION);
    writer.string(LEDGER_HASH_SUITE)?;
    writer.string(&ledger.campaign_id)?;
    writer.string(&ledger.global_filtration_id)?;
    writer.digest(&ledger.global_filtration_sha256)?;
    writer.string(&ledger.ledger_epoch_id)?;
    writer.u64(sequence);
    writer.digest(previous_root)?;
    writer.u8(payload_kind);
    writer.bytes(&payload_bytes)?;
    let mut preimage = LEDGER_ENTRY_DOMAIN.to_vec();
    preimage.extend_from_slice(&writer.0);
    Ok(sha256_hex(&preimage))
}

fn prefix_digest(
    domain: &[u8],
    event_ids: &[String],
    event_roots: &BTreeMap<String, String>,
) -> Result<String, AuditError> {
    let mut writer = CanonicalWriter::default();
    writer.u32(LEDGER_SCHEMA_VERSION);
    writer.u64(
        u64::try_from(event_ids.len())
            .map_err(|_| AuditError("prefix event list exceeds u64 length".into()))?,
    );
    for event_id in event_ids {
        writer.string(event_id)?;
        match event_roots.get(event_id) {
            Some(root) => {
                writer.u8(1);
                writer.digest(root)?;
            }
            None => writer.u8(0),
        }
    }
    let mut preimage = domain.to_vec();
    preimage.extend_from_slice(&writer.0);
    Ok(sha256_hex(&preimage))
}

/// Producer-side helper. Consumers must independently verify with audit_ledger.
pub fn seal_ledger(ledger: &mut GlobalLedger) -> Result<(), AuditError> {
    if ledger.campaign_id.is_empty()
        || ledger.global_filtration_id.is_empty()
        || ledger.ledger_epoch_id.is_empty()
    {
        return Err(AuditError(
            "campaign, filtration, and ledger epoch IDs are required".into(),
        ));
    }
    if !valid_sha256(&ledger.global_filtration_sha256) {
        return Err(AuditError("malformed global filtration digest".into()));
    }
    // Copy only immutable ledger context so the entries can be mutated below
    // without aliasing the whole GlobalLedger during entry-root calculation.
    let context = GlobalLedger {
        campaign_id: ledger.campaign_id.clone(),
        global_filtration_id: ledger.global_filtration_id.clone(),
        ledger_epoch_id: ledger.ledger_epoch_id.clone(),
        global_filtration_sha256: ledger.global_filtration_sha256.clone(),
        ledger_chain_root_sha256: String::new(),
        entries: Vec::new(),
    };
    let mut previous_root = genesis_root(&context)?;
    let mut event_roots = BTreeMap::new();
    let mut ordered_event_ids = Vec::new();
    for (index, entry) in ledger.entries.iter_mut().enumerate() {
        let sequence = index as u64 + 1;
        if entry.sequence != sequence {
            return Err(AuditError(format!(
                "global ledger sequence gap: expected {sequence}, found {}",
                entry.sequence
            )));
        }
        if let LedgerPayload::Decision(decision) = &mut entry.payload {
            decision.input_prefix_sha256 =
                prefix_digest(INPUT_PREFIX_DOMAIN, &ordered_event_ids, &event_roots)?;
            decision.history_prefix_sha256 = prefix_digest(
                HISTORY_PREFIX_DOMAIN,
                &decision.consumed_event_ids,
                &event_roots,
            )?;
        }
        let root = entry_root(&context, entry.sequence, &previous_root, &entry.payload)?;
        entry.entry_commitment_sha256.clone_from(&root);
        if let LedgerPayload::Evidence(event) = &entry.payload {
            if event_roots
                .insert(event.event_id.clone(), root.clone())
                .is_some()
            {
                return Err(AuditError(format!(
                    "duplicate evidence event ID {} during ledger sealing",
                    event.event_id
                )));
            }
            ordered_event_ids.push(event.event_id.clone());
        }
        previous_root = root;
    }
    ledger.ledger_chain_root_sha256 = previous_root;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LedgerIntegrityDisposition {
    InternallyCommittedUnanchored,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedTerminalCheckpoint {
    pub campaign_id: String,
    pub global_filtration_id: String,
    pub ledger_epoch_id: String,
    pub terminal_sequence: u64,
    pub ledger_chain_root_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalCheckpointDisposition {
    MatchesIndependentlyPinnedCheckpoint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalCheckpointAudit {
    pub disposition: ExternalCheckpointDisposition,
    pub checkpoint_sha256: String,
    pub terminal_sequence: u64,
    pub ledger_chain_root_sha256: String,
}

pub fn expected_terminal_checkpoint_sha256(
    checkpoint: &ExpectedTerminalCheckpoint,
) -> Result<String, AuditError> {
    let mut writer = CanonicalWriter::default();
    writer.u32(LEDGER_SCHEMA_VERSION);
    writer.string(LEDGER_HASH_SUITE)?;
    writer.string(&checkpoint.campaign_id)?;
    writer.string(&checkpoint.global_filtration_id)?;
    writer.string(&checkpoint.ledger_epoch_id)?;
    writer.u64(checkpoint.terminal_sequence);
    writer.digest(&checkpoint.ledger_chain_root_sha256)?;
    let mut preimage = CHECKPOINT_DOMAIN.to_vec();
    preimage.extend_from_slice(&writer.0);
    Ok(sha256_hex(&preimage))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AvailabilityState {
    Approved,
    Provisional,
    Disputed,
    Failed,
    Missing,
    OutOfSupport,
    NotAssessed,
}

impl AvailabilityState {
    fn consumable_as_typed_state(self) -> bool {
        matches!(
            self,
            Self::Approved | Self::Failed | Self::Missing | Self::OutOfSupport | Self::NotAssessed
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceSplit {
    Exploration,
    Fit,
    Calibration,
    Confirmation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceEvent {
    pub event_id: String,
    pub evidence_family_id: String,
    pub source_stream: String,
    pub source_artifact_sha256: String,
    pub revision: u32,
    pub supersedes_event_id: Option<String>,
    /// Biological event time. It never determines operational availability.
    pub event_time_ms: Option<i64>,
    pub acquired_at_ms: Option<i64>,
    pub annotated_at_ms: Option<i64>,
    pub qc_at_ms: Option<i64>,
    pub availability_state: AvailabilityState,
    pub split: EvidenceSplit,
    /// Required for confirmation-split data; binds it to the frozen family.
    pub confirmatory_family_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyFreeze {
    pub family_id: String,
    pub family_commitment_sha256: String,
    /// The lockbox is still closed at the instant of this freeze event.
    pub confirmation_lockbox_closed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionKind {
    Discovery,
    Confirmation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionReceipt {
    pub decision_id: String,
    pub campaign_id: String,
    pub kind: DecisionKind,
    pub policy_version_sha256: String,
    pub input_prefix_sha256: String,
    pub driver_receipt_sha256: String,
    pub history_prefix_sha256: String,
    pub action_sha256: String,
    pub feasibility_receipt_sha256: String,
    /// Exact set of evidence made available to the policy before this decision.
    pub available_event_ids: Vec<String>,
    /// Exact evidence inputs reported by the driver as consumed.
    pub consumed_event_ids: Vec<String>,
    pub confirmatory_family_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerPayload {
    Evidence(EvidenceEvent),
    FamilyFreeze(FamilyFreeze),
    Decision(DecisionReceipt),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEntry {
    /// One-based global availability/order sequence; all entries are unique.
    pub sequence: u64,
    pub entry_commitment_sha256: String,
    pub payload: LedgerPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalLedger {
    pub campaign_id: String,
    pub global_filtration_id: String,
    /// Immutable identity for one append-only availability-ledger epoch.
    pub ledger_epoch_id: String,
    /// Filtration-definition identity; distinct from the evolving ledger root.
    pub global_filtration_sha256: String,
    /// Recomputed terminal append-chain root over all entries in this epoch.
    pub ledger_chain_root_sha256: String,
    pub entries: Vec<LedgerEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionAuditStatus {
    Accepted,
    LookaheadDetected,
    PrefixMismatch,
    LockboxLeakage,
    InvalidReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionAudit {
    pub decision_id: String,
    pub sequence: u64,
    pub status: DecisionAuditStatus,
    pub available_prefix_ids: Vec<String>,
    pub consumed_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerAudit {
    pub campaign_id: String,
    pub ledger_epoch_id: String,
    pub final_sequence: u64,
    pub global_filtration_sha256: String,
    pub ledger_chain_root_sha256: String,
    pub integrity_disposition: LedgerIntegrityDisposition,
    pub decisions: Vec<DecisionAudit>,
    pub evidence_ids: Vec<String>,
}

/// Replay the immutable ledger in availability order and independently
/// recompute every content-bound entry commitment and decision-time prefix.
/// Biological event time never determines ordering or lookahead.
pub fn audit_ledger(ledger: &GlobalLedger) -> Result<LedgerAudit, AuditError> {
    if ledger.campaign_id.is_empty()
        || ledger.global_filtration_id.is_empty()
        || ledger.ledger_epoch_id.is_empty()
    {
        return Err(AuditError(
            "campaign, global filtration, and ledger epoch IDs are required".into(),
        ));
    }
    if !valid_sha256(&ledger.global_filtration_sha256)
        || !valid_sha256(&ledger.ledger_chain_root_sha256)
    {
        return Err(AuditError(
            "malformed filtration-definition or ledger-chain digest".into(),
        ));
    }
    if ledger.entries.is_empty() {
        return Err(AuditError(
            "global ledger must contain at least one entry".into(),
        ));
    }

    let mut evidence: BTreeMap<String, (u64, EvidenceEvent)> = BTreeMap::new();
    let mut event_roots: BTreeMap<String, String> = BTreeMap::new();
    let mut families: BTreeMap<String, (u64, FamilyFreeze)> = BTreeMap::new();
    let mut decisions = Vec::new();
    let mut ordered_evidence_ids = Vec::new();
    let mut previous_root = genesis_root(ledger)?;

    for (index, entry) in ledger.entries.iter().enumerate() {
        let expected_sequence = index as u64 + 1;
        if entry.sequence != expected_sequence {
            return Err(AuditError(format!(
                "global ledger sequence gap: expected {expected_sequence}, found {}",
                entry.sequence
            )));
        }
        if !valid_sha256(&entry.entry_commitment_sha256) {
            return Err(AuditError(format!(
                "malformed entry commitment at sequence {}",
                entry.sequence
            )));
        }

        let expected_entry_root =
            entry_root(ledger, entry.sequence, &previous_root, &entry.payload)?;
        if entry.entry_commitment_sha256 != expected_entry_root {
            return Err(AuditError(format!(
                "entry commitment mismatch at global sequence {}",
                entry.sequence
            )));
        }
        previous_root.clone_from(&expected_entry_root);

        match &entry.payload {
            LedgerPayload::Evidence(event) => {
                if event.event_id.is_empty()
                    || event.evidence_family_id.is_empty()
                    || event.source_stream.is_empty()
                {
                    return Err(AuditError(format!(
                        "invalid evidence identity at sequence {}",
                        entry.sequence
                    )));
                }
                if evidence.contains_key(&event.event_id) {
                    return Err(AuditError(format!(
                        "duplicate or rewritten event ID {}",
                        event.event_id
                    )));
                }
                match event.split {
                    EvidenceSplit::Confirmation => {
                        let Some(family_id) = event.confirmatory_family_id.as_ref() else {
                            return Err(AuditError(format!(
                                "confirmation event {} lacks a family binding",
                                event.event_id
                            )));
                        };
                        let Some((freeze_sequence, _)) = families.get(family_id) else {
                            return Err(AuditError(format!(
                                "confirmation event {} arrived before its family freeze",
                                event.event_id
                            )));
                        };
                        if *freeze_sequence >= entry.sequence {
                            return Err(AuditError(format!(
                                "confirmation event {} is not after its family freeze",
                                event.event_id
                            )));
                        }
                    }
                    _ if event.confirmatory_family_id.is_some() => {
                        return Err(AuditError(format!(
                            "non-confirmation event {} has a confirmatory family binding",
                            event.event_id
                        )));
                    }
                    _ => {}
                }
                match (&event.supersedes_event_id, event.revision) {
                    (None, 0) => {}
                    (Some(previous_id), revision) if revision > 0 => {
                        let (previous_sequence, previous) =
                            evidence.get(previous_id).ok_or_else(|| {
                                AuditError(format!(
                                    "{} supersedes an unknown or future event {previous_id}",
                                    event.event_id
                                ))
                            })?;
                        if *previous_sequence >= entry.sequence
                            || previous.evidence_family_id != event.evidence_family_id
                            || previous.source_stream != event.source_stream
                            || previous.split != event.split
                            || previous.confirmatory_family_id != event.confirmatory_family_id
                            || previous.revision.checked_add(1) != Some(revision)
                        {
                            return Err(AuditError(format!(
                                "invalid revision/supersession chain for {}",
                                event.event_id
                            )));
                        }
                    }
                    _ => {
                        return Err(AuditError(format!(
                            "revision zero must have no supersession; later revisions must name a parent ({})",
                            event.event_id
                        )));
                    }
                }
                evidence.insert(event.event_id.clone(), (entry.sequence, event.clone()));
                event_roots.insert(event.event_id.clone(), expected_entry_root);
                ordered_evidence_ids.push(event.event_id.clone());
            }
            LedgerPayload::FamilyFreeze(freeze) => {
                if freeze.family_id.is_empty()
                    || !freeze.confirmation_lockbox_closed
                    || families.contains_key(&freeze.family_id)
                {
                    return Err(AuditError(format!(
                        "invalid or duplicate family freeze at sequence {}",
                        entry.sequence
                    )));
                }
                families.insert(freeze.family_id.clone(), (entry.sequence, freeze.clone()));
            }
            LedgerPayload::Decision(decision) => {
                let mut status = DecisionAuditStatus::Accepted;
                if decision.campaign_id != ledger.campaign_id || decision.decision_id.is_empty() {
                    status = DecisionAuditStatus::InvalidReceipt;
                }
                for digest in [
                    &decision.policy_version_sha256,
                    &decision.input_prefix_sha256,
                    &decision.driver_receipt_sha256,
                    &decision.history_prefix_sha256,
                    &decision.action_sha256,
                    &decision.feasibility_receipt_sha256,
                ] {
                    if !valid_sha256(digest) {
                        status = DecisionAuditStatus::InvalidReceipt;
                    }
                }

                let expected_input_prefix =
                    prefix_digest(INPUT_PREFIX_DOMAIN, &ordered_evidence_ids, &event_roots)?;
                let expected_history_prefix = prefix_digest(
                    HISTORY_PREFIX_DOMAIN,
                    &decision.consumed_event_ids,
                    &event_roots,
                )?;
                if decision.available_event_ids != ordered_evidence_ids
                    || decision.input_prefix_sha256 != expected_input_prefix
                    || decision.history_prefix_sha256 != expected_history_prefix
                {
                    status = DecisionAuditStatus::PrefixMismatch;
                }
                match decision.kind {
                    DecisionKind::Discovery if decision.confirmatory_family_id.is_some() => {
                        status = DecisionAuditStatus::InvalidReceipt;
                    }
                    DecisionKind::Confirmation if decision.confirmatory_family_id.is_none() => {
                        status = DecisionAuditStatus::LockboxLeakage;
                    }
                    _ => {}
                }

                let mut consumed = BTreeSet::new();
                for event_id in &decision.consumed_event_ids {
                    if !consumed.insert(event_id.as_str()) {
                        status = DecisionAuditStatus::InvalidReceipt;
                        continue;
                    }
                    let Some((available_sequence, event)) = evidence.get(event_id) else {
                        status = DecisionAuditStatus::LookaheadDetected;
                        continue;
                    };
                    if *available_sequence >= entry.sequence {
                        status = DecisionAuditStatus::LookaheadDetected;
                    }
                    if !event.availability_state.consumable_as_typed_state() {
                        status = DecisionAuditStatus::InvalidReceipt;
                    }
                    match decision.kind {
                        DecisionKind::Discovery => {
                            if event.split == EvidenceSplit::Confirmation {
                                status = DecisionAuditStatus::LockboxLeakage;
                            }
                        }
                        DecisionKind::Confirmation => {
                            if event.split == EvidenceSplit::Confirmation {
                                let Some(family_id) = decision.confirmatory_family_id.as_ref()
                                else {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                    continue;
                                };
                                let Some((freeze_sequence, _)) = families.get(family_id) else {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                    continue;
                                };
                                if *freeze_sequence >= entry.sequence
                                    || *available_sequence <= *freeze_sequence
                                    || event.confirmatory_family_id.as_deref()
                                        != Some(family_id.as_str())
                                {
                                    status = DecisionAuditStatus::LockboxLeakage;
                                }
                            }
                        }
                    }
                }
                if decision.kind == DecisionKind::Confirmation {
                    let valid_freeze = decision
                        .confirmatory_family_id
                        .as_ref()
                        .and_then(|id| families.get(id))
                        .is_some_and(|(freeze_sequence, _)| *freeze_sequence < entry.sequence);
                    if !valid_freeze {
                        status = DecisionAuditStatus::LockboxLeakage;
                    }
                }
                decisions.push(DecisionAudit {
                    decision_id: decision.decision_id.clone(),
                    sequence: entry.sequence,
                    status,
                    available_prefix_ids: decision.available_event_ids.clone(),
                    consumed_ids: decision.consumed_event_ids.clone(),
                });
            }
        }
    }

    if ledger.ledger_chain_root_sha256 != previous_root {
        return Err(AuditError(
            "ledger terminal chain root does not match recomputed final entry".into(),
        ));
    }

    Ok(LedgerAudit {
        campaign_id: ledger.campaign_id.clone(),
        ledger_epoch_id: ledger.ledger_epoch_id.clone(),
        final_sequence: ledger.entries.len() as u64,
        global_filtration_sha256: ledger.global_filtration_sha256.clone(),
        ledger_chain_root_sha256: previous_root,
        integrity_disposition: LedgerIntegrityDisposition::InternallyCommittedUnanchored,
        decisions,
        evidence_ids: ordered_evidence_ids,
    })
}

/// Match a committed ledger to a terminal checkpoint whose commitment was
/// pinned through an independent channel. This does not verify a signature or
/// witness quorum; trust in the pinned digest remains external.
pub fn audit_ledger_against_external_checkpoint(
    ledger: &GlobalLedger,
    checkpoint: &ExpectedTerminalCheckpoint,
    independently_pinned_checkpoint_sha256: &str,
) -> Result<ExternalCheckpointAudit, AuditError> {
    let audit = audit_ledger(ledger)?;
    let computed = expected_terminal_checkpoint_sha256(checkpoint)?;
    if computed != independently_pinned_checkpoint_sha256 {
        return Err(AuditError(
            "terminal checkpoint does not match the independently pinned checkpoint digest".into(),
        ));
    }
    if checkpoint.campaign_id != ledger.campaign_id
        || checkpoint.global_filtration_id != ledger.global_filtration_id
        || checkpoint.ledger_epoch_id != ledger.ledger_epoch_id
        || checkpoint.terminal_sequence != audit.final_sequence
        || checkpoint.ledger_chain_root_sha256 != audit.ledger_chain_root_sha256
    {
        return Err(AuditError(
            "ledger does not match the externally pinned terminal checkpoint".into(),
        ));
    }
    Ok(ExternalCheckpointAudit {
        disposition: ExternalCheckpointDisposition::MatchesIndependentlyPinnedCheckpoint,
        checkpoint_sha256: computed,
        terminal_sequence: audit.final_sequence,
        ledger_chain_root_sha256: audit.ledger_chain_root_sha256,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionMode {
    DirectGlobalSupermartingale,
    AdjusterLifted,
    LocalOnly,
    IncomparableOrUnknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdjusterReceipt {
    pub adjuster_id: String,
    pub adjuster_version: String,
    pub source_process_sha256: String,
    pub source_filtration_sha256: String,
    pub target_filtration_sha256: String,
    pub parameter_sha256: String,
    pub theorem_assumptions_sha256: String,
    pub lifted_output_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiltrationAdmission {
    pub campaign_id: String,
    pub process_id: String,
    pub null_spec_sha256: String,
    pub input_process_sha256: String,
    pub output_process_sha256: String,
    pub source_filtration_id: String,
    pub source_filtration_sha256: String,
    pub target_filtration_id: String,
    pub target_filtration_sha256: String,
    pub observed_event_ids: Vec<String>,
    /// The process output is committed after this availability prefix.
    pub process_output_sequence: u64,
    pub mode: AdmissionMode,
    /// Independent proof-review receipt; this verifier does not check its theorem.
    pub proof_receipt_sha256: Option<String>,
    pub assumption_sha256: Option<String>,
    pub adjuster: Option<AdjusterReceipt>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionStatus {
    AdmittedDirectGlobal,
    AdmittedAfterLift,
    NotCombinableLocalOnly,
    NotCombinableIncomparable,
    InvalidReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionAudit {
    pub process_id: String,
    pub status: AdmissionStatus,
    pub proof_receipt_sha256: Option<String>,
    pub source_filtration_sha256: String,
    pub target_filtration_sha256: String,
}

/// Structural gate only. Admission means that a matching external proof
/// receipt is present and bound to these exact artifacts; it is not a claim
/// that this crate proved the supermartingale/adjuster theorem.
pub fn audit_admission(ledger: &GlobalLedger, admission: &FiltrationAdmission) -> AdmissionAudit {
    let invalid = |reason_status| AdmissionAudit {
        process_id: admission.process_id.clone(),
        status: reason_status,
        proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
        source_filtration_sha256: admission.source_filtration_sha256.clone(),
        target_filtration_sha256: admission.target_filtration_sha256.clone(),
    };
    if admission.campaign_id != ledger.campaign_id
        || admission.process_id.is_empty()
        || admission.source_filtration_id.is_empty()
        || admission.target_filtration_id != ledger.global_filtration_id
        || admission.target_filtration_sha256 != ledger.global_filtration_sha256
        || !valid_sha256(&admission.null_spec_sha256)
        || !valid_sha256(&admission.input_process_sha256)
        || !valid_sha256(&admission.output_process_sha256)
        || !valid_sha256(&admission.source_filtration_sha256)
        || !valid_sha256(&admission.target_filtration_sha256)
        || admission.observed_event_ids.is_empty()
        || admission.process_output_sequence == 0
        || admission.process_output_sequence > ledger.entries.len() as u64
    {
        return invalid(AdmissionStatus::InvalidReceipt);
    }

    let evidence_map: BTreeMap<_, _> = ledger
        .entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            LedgerPayload::Evidence(event) => {
                Some((event.event_id.as_str(), (entry.sequence, event)))
            }
            _ => None,
        })
        .collect();
    let mut observed = BTreeSet::new();
    for event_id in &admission.observed_event_ids {
        let Some((available_sequence, event)) = evidence_map.get(event_id.as_str()) else {
            return invalid(AdmissionStatus::InvalidReceipt);
        };
        if !observed.insert(event_id.as_str())
            || *available_sequence >= admission.process_output_sequence
            || !event.availability_state.consumable_as_typed_state()
        {
            return invalid(AdmissionStatus::InvalidReceipt);
        }
    }

    match admission.mode {
        AdmissionMode::DirectGlobalSupermartingale => {
            if admission.source_filtration_id != ledger.global_filtration_id
                || admission.source_filtration_sha256 != ledger.global_filtration_sha256
                || admission
                    .proof_receipt_sha256
                    .as_deref()
                    .is_none_or(|d| !valid_sha256(d))
                || admission
                    .assumption_sha256
                    .as_deref()
                    .is_none_or(|d| !valid_sha256(d))
                || admission.adjuster.is_some()
            {
                return invalid(AdmissionStatus::InvalidReceipt);
            }
            AdmissionAudit {
                process_id: admission.process_id.clone(),
                status: AdmissionStatus::AdmittedDirectGlobal,
                proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
                source_filtration_sha256: admission.source_filtration_sha256.clone(),
                target_filtration_sha256: admission.target_filtration_sha256.clone(),
            }
        }
        AdmissionMode::AdjusterLifted => {
            let Some(adjuster) = admission.adjuster.as_ref() else {
                return invalid(AdmissionStatus::InvalidReceipt);
            };
            if admission.source_filtration_sha256 == ledger.global_filtration_sha256
                || admission
                    .proof_receipt_sha256
                    .as_deref()
                    .is_none_or(|d| !valid_sha256(d))
                || admission
                    .assumption_sha256
                    .as_deref()
                    .is_none_or(|d| !valid_sha256(d))
                || adjuster.adjuster_id.is_empty()
                || adjuster.adjuster_version.is_empty()
                || adjuster.source_process_sha256 != admission.input_process_sha256
                || adjuster.source_filtration_sha256 != admission.source_filtration_sha256
                || adjuster.target_filtration_sha256 != admission.target_filtration_sha256
                || admission.assumption_sha256.as_deref()
                    != Some(adjuster.theorem_assumptions_sha256.as_str())
                || !valid_sha256(&adjuster.source_process_sha256)
                || !valid_sha256(&adjuster.source_filtration_sha256)
                || !valid_sha256(&adjuster.target_filtration_sha256)
                || !valid_sha256(&adjuster.parameter_sha256)
                || !valid_sha256(&adjuster.theorem_assumptions_sha256)
                || !valid_sha256(&adjuster.lifted_output_sha256)
                || adjuster.lifted_output_sha256 != admission.output_process_sha256
            {
                return invalid(AdmissionStatus::InvalidReceipt);
            }
            AdmissionAudit {
                process_id: admission.process_id.clone(),
                status: AdmissionStatus::AdmittedAfterLift,
                proof_receipt_sha256: admission.proof_receipt_sha256.clone(),
                source_filtration_sha256: admission.source_filtration_sha256.clone(),
                target_filtration_sha256: admission.target_filtration_sha256.clone(),
            }
        }
        AdmissionMode::LocalOnly => invalid(AdmissionStatus::NotCombinableLocalOnly),
        AdmissionMode::IncomparableOrUnknown => invalid(AdmissionStatus::NotCombinableIncomparable),
    }
}

/// Replayable as-of snapshots for prefix-invariance metamorphic tests.
pub fn decision_prefixes(
    ledger: &GlobalLedger,
) -> Result<Vec<(String, Vec<String>, String)>, AuditError> {
    let audit = audit_ledger(ledger)?;
    if audit
        .decisions
        .iter()
        .any(|decision| decision.status != DecisionAuditStatus::Accepted)
    {
        return Err(AuditError(
            "cannot produce prefixes from an invalid decision ledger".into(),
        ));
    }
    Ok(audit
        .decisions
        .iter()
        .map(|decision| {
            let action_sha = ledger
                .entries
                .iter()
                .find_map(|entry| match &entry.payload {
                    LedgerPayload::Decision(receipt)
                        if receipt.decision_id == decision.decision_id =>
                    {
                        Some(receipt.action_sha256.clone())
                    }
                    _ => None,
                })
                .expect("audited decision exists");
            (
                decision.decision_id.clone(),
                decision.consumed_ids.clone(),
                action_sha,
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn event(
        id: &str,
        family: &str,
        event_time: i64,
        state: AvailabilityState,
        split: EvidenceSplit,
    ) -> LedgerPayload {
        LedgerPayload::Evidence(EvidenceEvent {
            event_id: id.into(),
            evidence_family_id: family.into(),
            source_stream: "synthetic-stream".into(),
            source_artifact_sha256: d('a'),
            revision: 0,
            supersedes_event_id: None,
            event_time_ms: Some(event_time),
            acquired_at_ms: Some(event_time + 1),
            annotated_at_ms: Some(event_time + 2),
            qc_at_ms: Some(event_time + 3),
            availability_state: state,
            split,
            confirmatory_family_id: None,
        })
    }

    fn decision(
        id: &str,
        available: &[&str],
        consumed: &[&str],
        kind: DecisionKind,
        family: Option<&str>,
    ) -> LedgerPayload {
        LedgerPayload::Decision(DecisionReceipt {
            decision_id: id.into(),
            campaign_id: "campaign-1".into(),
            kind,
            policy_version_sha256: d('b'),
            input_prefix_sha256: d('c'),
            driver_receipt_sha256: d('d'),
            history_prefix_sha256: d('e'),
            action_sha256: d('f'),
            feasibility_receipt_sha256: d('1'),
            available_event_ids: available.iter().map(|x| (*x).into()).collect(),
            consumed_event_ids: consumed.iter().map(|x| (*x).into()).collect(),
            confirmatory_family_id: family.map(str::to_owned),
        })
    }

    fn entry(sequence: u64, payload: LedgerPayload) -> LedgerEntry {
        LedgerEntry {
            sequence,
            entry_commitment_sha256: d('9'),
            payload,
        }
    }

    fn fixture() -> GlobalLedger {
        let mut ledger = GlobalLedger {
            campaign_id: "campaign-1".into(),
            global_filtration_id: "connectome-global-v1".into(),
            ledger_epoch_id: "epoch-1".into(),
            global_filtration_sha256: d('8'),
            ledger_chain_root_sha256: d('0'),
            entries: vec![
                entry(
                    1,
                    event(
                        "S1",
                        "structural",
                        100,
                        AvailabilityState::Approved,
                        EvidenceSplit::Fit,
                    ),
                ),
                entry(
                    2,
                    decision("D1", &["S1"], &["S1"], DecisionKind::Discovery, None),
                ),
                entry(
                    3,
                    event(
                        "T7",
                        "behaviour-trial-7",
                        90,
                        AvailabilityState::Approved,
                        EvidenceSplit::Exploration,
                    ),
                ),
                entry(
                    4,
                    decision(
                        "D2",
                        &["S1", "T7"],
                        &["S1", "T7"],
                        DecisionKind::Discovery,
                        None,
                    ),
                ),
                entry(
                    5,
                    event(
                        "T8-failed",
                        "behaviour-trial-8",
                        105,
                        AvailabilityState::Failed,
                        EvidenceSplit::Exploration,
                    ),
                ),
            ],
        };
        seal_ledger(&mut ledger).expect("synthetic ledger seals");
        ledger
    }

    fn confirmation_event() -> LedgerPayload {
        let mut payload = event(
            "T7",
            "behaviour-trial-7",
            90,
            AvailabilityState::Approved,
            EvidenceSplit::Confirmation,
        );
        if let LedgerPayload::Evidence(ref mut value) = payload {
            value.confirmatory_family_id = Some("family-1".into());
        }
        payload
    }

    fn admission(mode: AdmissionMode) -> FiltrationAdmission {
        FiltrationAdmission {
            campaign_id: "campaign-1".into(),
            process_id: "P-S".into(),
            null_spec_sha256: d('a'),
            input_process_sha256: d('b'),
            output_process_sha256: d('c'),
            source_filtration_id: "connectome-global-v1".into(),
            source_filtration_sha256: d('8'),
            target_filtration_id: "connectome-global-v1".into(),
            target_filtration_sha256: d('8'),
            observed_event_ids: vec!["S1".into(), "T7".into()],
            process_output_sequence: 4,
            mode,
            proof_receipt_sha256: Some(d('d')),
            assumption_sha256: Some(d('e')),
            adjuster: None,
        }
    }

    #[test]
    fn known_answer_uses_availability_not_event_time() {
        let report = audit_ledger(&fixture()).unwrap();
        assert_eq!(report.final_sequence, 5);
        assert_eq!(report.decisions.len(), 2);
        assert_eq!(report.decisions[0].status, DecisionAuditStatus::Accepted);
        assert_eq!(report.decisions[1].status, DecisionAuditStatus::Accepted);
        assert_eq!(report.decisions[0].available_prefix_ids, vec!["S1"]);
        assert_eq!(report.decisions[1].available_prefix_ids, vec!["S1", "T7"]);
        // T7's biological event time (90) predates D1, but its availability
        // sequence (3) follows D1 (2), so D1 cannot consume it.
    }

    #[test]
    fn detects_lookahead_even_when_biological_event_time_is_earlier() {
        let mut ledger = fixture();
        ledger.entries[1] = entry(
            2,
            decision(
                "D1",
                &["S1", "T7"],
                &["S1", "T7"],
                DecisionKind::Discovery,
                None,
            ),
        );
        // T7 has not yet been delivered at sequence 2.
        seal_ledger(&mut ledger).unwrap();
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(
            audit.decisions[0].status,
            DecisionAuditStatus::LookaheadDetected
        );
    }

    #[test]
    fn rejects_provisional_and_disputed_inputs() {
        let mut ledger = fixture();
        ledger.entries[0] = entry(
            1,
            event(
                "S1",
                "structural",
                100,
                AvailabilityState::Provisional,
                EvidenceSplit::Fit,
            ),
        );
        seal_ledger(&mut ledger).unwrap();
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(
            audit.decisions[0].status,
            DecisionAuditStatus::InvalidReceipt
        );
    }

    #[test]
    fn rejects_noncontiguous_global_order() {
        let mut ledger = fixture();
        ledger.entries[2].sequence = 7;
        assert!(audit_ledger(&ledger).is_err());
    }

    #[test]
    fn rejects_duplicate_event_ids_and_in_place_rewrites() {
        let mut ledger = fixture();
        ledger.entries.push(entry(
            6,
            event(
                "S1",
                "structural",
                101,
                AvailabilityState::Approved,
                EvidenceSplit::Fit,
            ),
        ));
        assert!(audit_ledger(&ledger).is_err());
    }

    #[test]
    fn accepts_immutable_revision_chain_as_new_events() {
        let mut ledger = fixture();
        let revised = EvidenceEvent {
            event_id: "T7-v2".into(),
            evidence_family_id: "behaviour-trial-7".into(),
            source_stream: "synthetic-stream".into(),
            source_artifact_sha256: d('2'),
            revision: 1,
            supersedes_event_id: Some("T7".into()),
            event_time_ms: Some(90),
            acquired_at_ms: Some(91),
            annotated_at_ms: Some(110),
            qc_at_ms: Some(111),
            availability_state: AvailabilityState::Approved,
            split: EvidenceSplit::Exploration,
            confirmatory_family_id: None,
        };
        ledger
            .entries
            .push(entry(6, LedgerPayload::Evidence(revised)));
        seal_ledger(&mut ledger).unwrap();
        assert!(audit_ledger(&ledger).is_ok());
    }

    #[test]
    fn rejects_discovery_consuming_confirmation_lockbox_data() {
        let mut ledger = fixture();
        ledger.entries.insert(
            1,
            entry(
                2,
                LedgerPayload::FamilyFreeze(FamilyFreeze {
                    family_id: "family-1".into(),
                    family_commitment_sha256: d('7'),
                    confirmation_lockbox_closed: true,
                }),
            ),
        );
        ledger.entries[2] = entry(3, confirmation_event());
        ledger.entries[3] = entry(
            4,
            decision(
                "D2",
                &["S1", "T7"],
                &["S1", "T7"],
                DecisionKind::Discovery,
                None,
            ),
        );
        for (index, item) in ledger.entries.iter_mut().enumerate() {
            item.sequence = index as u64 + 1;
        }
        seal_ledger(&mut ledger).unwrap();
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(
            audit.decisions[0].status,
            DecisionAuditStatus::LockboxLeakage
        );
    }

    #[test]
    fn confirmation_requires_a_prior_freeze_and_postfreeze_lockbox_data() {
        let mut ledger = fixture();
        ledger.entries.insert(
            1,
            entry(
                2,
                LedgerPayload::FamilyFreeze(FamilyFreeze {
                    family_id: "family-1".into(),
                    family_commitment_sha256: d('7'),
                    confirmation_lockbox_closed: true,
                }),
            ),
        );
        ledger.entries[2] = entry(
            3,
            decision("D1", &["S1"], &["S1"], DecisionKind::Discovery, None),
        );
        ledger.entries[3] = entry(4, confirmation_event());
        ledger.entries[4] = entry(
            5,
            decision(
                "D2",
                &["S1", "T7"],
                &["S1", "T7"],
                DecisionKind::Confirmation,
                Some("family-1"),
            ),
        );
        for (idx, e) in ledger.entries.iter_mut().enumerate() {
            e.sequence = idx as u64 + 1;
        }
        seal_ledger(&mut ledger).unwrap();
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(audit.decisions[1].status, DecisionAuditStatus::Accepted);
    }

    #[test]
    fn rejects_confirmation_event_that_predates_family_freeze() {
        let mut ledger = fixture();
        let structural = ledger.entries[0].clone();
        let mut prefreeze_event = confirmation_event();
        if let LedgerPayload::Evidence(ref mut value) = prefreeze_event {
            value.event_id = "T7-prefreeze".into();
        }
        ledger.entries = vec![
            structural,
            entry(2, prefreeze_event),
            entry(
                3,
                LedgerPayload::FamilyFreeze(FamilyFreeze {
                    family_id: "family-1".into(),
                    family_commitment_sha256: d('7'),
                    confirmation_lockbox_closed: true,
                }),
            ),
        ];
        seal_ledger(&mut ledger).unwrap();
        assert!(audit_ledger(&ledger).is_err());
    }

    #[test]
    fn direct_global_admission_requires_exact_global_filtration_and_review_receipt() {
        let ledger = fixture();
        let report = audit_admission(
            &ledger,
            &admission(AdmissionMode::DirectGlobalSupermartingale),
        );
        assert_eq!(report.status, AdmissionStatus::AdmittedDirectGlobal);
    }

    #[test]
    fn process_output_must_follow_all_observed_evidence() {
        let ledger = fixture();
        let mut receipt = admission(AdmissionMode::DirectGlobalSupermartingale);
        receipt.process_output_sequence = 3; // T7 is first available at sequence 3.
        assert_eq!(
            audit_admission(&ledger, &receipt).status,
            AdmissionStatus::InvalidReceipt
        );
    }

    #[test]
    fn local_only_stream_is_not_combinable_by_digest_relabelling() {
        let ledger = fixture();
        let mut local = admission(AdmissionMode::LocalOnly);
        local.source_filtration_id = "neural-local".into();
        local.source_filtration_sha256 = d('4');
        let report = audit_admission(&ledger, &local);
        assert_eq!(report.status, AdmissionStatus::NotCombinableLocalOnly);

        let mut forged = admission(AdmissionMode::DirectGlobalSupermartingale);
        forged.source_filtration_id = "neural-local".into();
        forged.source_filtration_sha256 = d('4');
        assert_eq!(
            audit_admission(&ledger, &forged).status,
            AdmissionStatus::InvalidReceipt
        );
    }

    #[test]
    fn lifted_stream_must_bind_adjuster_and_exact_output() {
        let ledger = fixture();
        let mut lifted = admission(AdmissionMode::AdjusterLifted);
        lifted.source_filtration_id = "neural-local".into();
        lifted.source_filtration_sha256 = d('4');
        lifted.adjuster = Some(AdjusterReceipt {
            adjuster_id: "reviewed-adjuster".into(),
            adjuster_version: "1".into(),
            source_process_sha256: d('b'),
            source_filtration_sha256: d('4'),
            target_filtration_sha256: d('8'),
            parameter_sha256: d('5'),
            theorem_assumptions_sha256: d('e'),
            lifted_output_sha256: d('c'),
        });
        assert_eq!(
            audit_admission(&ledger, &lifted).status,
            AdmissionStatus::AdmittedAfterLift
        );
        lifted.adjuster.as_mut().unwrap().lifted_output_sha256 = d('7');
        assert_eq!(
            audit_admission(&ledger, &lifted).status,
            AdmissionStatus::InvalidReceipt
        );
    }

    #[test]
    fn incomparable_filtration_is_explicitly_not_combinable() {
        let ledger = fixture();
        assert_eq!(
            audit_admission(&ledger, &admission(AdmissionMode::IncomparableOrUnknown)).status,
            AdmissionStatus::NotCombinableIncomparable
        );
    }

    #[test]
    fn future_only_extension_preserves_prior_decision_prefixes() {
        let ledger = fixture();
        let before = decision_prefixes(&ledger).unwrap();
        let mut extended = fixture();
        extended.entries.push(entry(
            6,
            event(
                "future-track",
                "later-trial",
                -999,
                AvailabilityState::Approved,
                EvidenceSplit::Exploration,
            ),
        ));
        seal_ledger(&mut extended).unwrap();
        let after = decision_prefixes(&extended).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn typed_failed_and_out_of_support_events_are_not_zero_filled() {
        let mut ledger = fixture();
        ledger.entries[4] = entry(
            5,
            event(
                "T8-failed",
                "behaviour-trial-8",
                105,
                AvailabilityState::OutOfSupport,
                EvidenceSplit::Exploration,
            ),
        );
        seal_ledger(&mut ledger).unwrap();
        let report = audit_ledger(&ledger).unwrap();
        assert_eq!(report.evidence_ids.last().unwrap(), "T8-failed");
    }

    #[test]
    fn standard_sha256_known_answer_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn ledger_audit_recomputes_each_entry_and_terminal_root() {
        let ledger = fixture();
        let audit = audit_ledger(&ledger).unwrap();
        assert_eq!(
            audit.integrity_disposition,
            LedgerIntegrityDisposition::InternallyCommittedUnanchored
        );
        assert_eq!(audit.ledger_epoch_id, "epoch-1");
        assert_eq!(audit.global_filtration_sha256, d('8'));
        assert_eq!(
            audit.ledger_chain_root_sha256,
            ledger.ledger_chain_root_sha256
        );
        assert_ne!(
            audit.ledger_chain_root_sha256,
            audit.global_filtration_sha256
        );

        let mut entry_tamper = ledger.clone();
        entry_tamper.entries[0].entry_commitment_sha256 = d('9');
        assert!(audit_ledger(&entry_tamper).is_err());

        let mut payload_tamper = ledger.clone();
        if let LedgerPayload::Evidence(event) = &mut payload_tamper.entries[0].payload {
            event.source_stream = "rewritten-stream".into();
        }
        assert!(audit_ledger(&payload_tamper).is_err());

        let mut root_tamper = ledger;
        root_tamper.ledger_chain_root_sha256 = d('7');
        assert!(audit_ledger(&root_tamper).is_err());
    }

    #[test]
    fn decision_prefix_hashes_are_recomputed_from_exact_ordered_evidence_roots() {
        let mut ledger = fixture();
        let original = match &ledger.entries[3].payload {
            LedgerPayload::Decision(decision) => decision.input_prefix_sha256.clone(),
            _ => unreachable!(),
        };
        if let LedgerPayload::Decision(decision) = &mut ledger.entries[3].payload {
            decision.input_prefix_sha256 = d('6');
        }
        // The receipt field is included in the entry itself, so even a
        // syntactically valid replacement invalidates its append commitment.
        assert!(audit_ledger(&ledger).is_err());

        if let LedgerPayload::Decision(decision) = &mut ledger.entries[3].payload {
            decision.input_prefix_sha256 = original;
        }
        seal_ledger(&mut ledger).unwrap();
        assert_eq!(
            audit_ledger(&ledger).unwrap().decisions[1].status,
            DecisionAuditStatus::Accepted
        );
    }

    #[test]
    fn canonical_payload_distinguishes_unicode_option_timestamps_and_enum_values() {
        let base = EvidenceEvent {
            event_id: "trial-µ".into(),
            evidence_family_id: "family-α".into(),
            source_stream: "kinematics/左".into(),
            source_artifact_sha256: d('a'),
            revision: 0,
            supersedes_event_id: None,
            event_time_ms: Some(-7),
            acquired_at_ms: None,
            annotated_at_ms: Some(0),
            qc_at_ms: None,
            availability_state: AvailabilityState::Provisional,
            split: EvidenceSplit::Calibration,
            confirmatory_family_id: None,
        };
        let (base_kind, base_bytes) =
            canonical_payload(&LedgerPayload::Evidence(base.clone())).unwrap();
        assert_eq!(base_kind, 1);
        assert!(
            base_bytes
                .windows("trial-µ".len())
                .any(|w| w == "trial-µ".as_bytes())
        );

        let mut present_option = base.clone();
        present_option.supersedes_event_id = Some("prior-1".into());
        assert_ne!(
            base_bytes,
            canonical_payload(&LedgerPayload::Evidence(present_option))
                .unwrap()
                .1
        );

        let mut timestamp_changed = base.clone();
        timestamp_changed.event_time_ms = None;
        assert_ne!(
            base_bytes,
            canonical_payload(&LedgerPayload::Evidence(timestamp_changed))
                .unwrap()
                .1
        );

        let mut enum_changed = base;
        enum_changed.availability_state = AvailabilityState::Disputed;
        assert_ne!(
            base_bytes,
            canonical_payload(&LedgerPayload::Evidence(enum_changed))
                .unwrap()
                .1
        );
    }

    #[test]
    fn payload_kind_discriminants_are_domain_distinct() {
        let evidence = event(
            "E1",
            "family",
            -1,
            AvailabilityState::Approved,
            EvidenceSplit::Exploration,
        );
        let freeze = LedgerPayload::FamilyFreeze(FamilyFreeze {
            family_id: "family".into(),
            family_commitment_sha256: d('b'),
            confirmation_lockbox_closed: true,
        });
        let decision = decision("D1", &[], &[], DecisionKind::Discovery, None);
        let (e_kind, e_bytes) = canonical_payload(&evidence).unwrap();
        let (f_kind, f_bytes) = canonical_payload(&freeze).unwrap();
        let (d_kind, d_bytes) = canonical_payload(&decision).unwrap();
        assert_eq!((e_kind, f_kind, d_kind), (1, 2, 3));
        assert_ne!(e_bytes, f_bytes);
        assert_ne!(e_bytes, d_bytes);
        assert_ne!(f_bytes, d_bytes);
    }

    #[test]
    fn independent_checkpoint_detects_a_resealed_truncated_prefix() {
        let complete = fixture();
        let complete_audit = audit_ledger(&complete).unwrap();
        let checkpoint = ExpectedTerminalCheckpoint {
            campaign_id: complete.campaign_id.clone(),
            global_filtration_id: complete.global_filtration_id.clone(),
            ledger_epoch_id: complete.ledger_epoch_id.clone(),
            terminal_sequence: complete_audit.final_sequence,
            ledger_chain_root_sha256: complete_audit.ledger_chain_root_sha256.clone(),
        };
        let pinned_sha = expected_terminal_checkpoint_sha256(&checkpoint).unwrap();
        let matched =
            audit_ledger_against_external_checkpoint(&complete, &checkpoint, &pinned_sha).unwrap();
        assert_eq!(
            matched.disposition,
            ExternalCheckpointDisposition::MatchesIndependentlyPinnedCheckpoint
        );

        let mut truncated = complete.clone();
        truncated.entries.pop();
        seal_ledger(&mut truncated).unwrap();
        let truncated_audit = audit_ledger(&truncated).unwrap();
        assert_eq!(
            truncated_audit.integrity_disposition,
            LedgerIntegrityDisposition::InternallyCommittedUnanchored
        );
        assert!(
            audit_ledger_against_external_checkpoint(&truncated, &checkpoint, &pinned_sha).is_err()
        );
    }

    #[test]
    fn external_checkpoint_digest_itself_must_match_the_pinned_value() {
        let ledger = fixture();
        let checkpoint = ExpectedTerminalCheckpoint {
            campaign_id: ledger.campaign_id.clone(),
            global_filtration_id: ledger.global_filtration_id.clone(),
            ledger_epoch_id: ledger.ledger_epoch_id.clone(),
            terminal_sequence: ledger.entries.len() as u64,
            ledger_chain_root_sha256: ledger.ledger_chain_root_sha256.clone(),
        };
        assert!(audit_ledger_against_external_checkpoint(&ledger, &checkpoint, &d('0')).is_err());
    }

    #[test]
    fn every_entry_payload_mutation_changes_its_commitment() {
        let ledger = fixture();
        let original = ledger.entries[0].entry_commitment_sha256.clone();
        let mut changed = ledger.clone();
        if let LedgerPayload::Evidence(event) = &mut changed.entries[0].payload {
            event.qc_at_ms = Some(-1);
        }
        seal_ledger(&mut changed).unwrap();
        assert_ne!(original, changed.entries[0].entry_commitment_sha256);
        assert_ne!(
            ledger.ledger_chain_root_sha256,
            changed.ledger_chain_root_sha256
        );
    }

    #[test]
    fn independent_checkpoint_rejects_old_epoch_and_wrong_sequence() {
        let ledger = fixture();
        let checkpoint = ExpectedTerminalCheckpoint {
            campaign_id: ledger.campaign_id.clone(),
            global_filtration_id: ledger.global_filtration_id.clone(),
            ledger_epoch_id: "prior-epoch".into(),
            terminal_sequence: ledger.entries.len() as u64,
            ledger_chain_root_sha256: ledger.ledger_chain_root_sha256.clone(),
        };
        let pin = expected_terminal_checkpoint_sha256(&checkpoint).unwrap();
        assert!(audit_ledger_against_external_checkpoint(&ledger, &checkpoint, &pin).is_err());

        let mut wrong_sequence = checkpoint;
        wrong_sequence.ledger_epoch_id = ledger.ledger_epoch_id.clone();
        wrong_sequence.terminal_sequence += 1;
        let pin = expected_terminal_checkpoint_sha256(&wrong_sequence).unwrap();
        assert!(audit_ledger_against_external_checkpoint(&ledger, &wrong_sequence, &pin).is_err());
    }

    #[test]
    fn empty_ledger_is_not_a_committed_evidence_history() {
        let mut ledger = fixture();
        ledger.entries.clear();
        ledger.ledger_chain_root_sha256 = genesis_root(&ledger).unwrap();
        assert!(audit_ledger(&ledger).is_err());
    }
}
