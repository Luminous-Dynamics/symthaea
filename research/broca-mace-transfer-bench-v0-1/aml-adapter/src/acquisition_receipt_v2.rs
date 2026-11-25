// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Versioned MACE acquisition receipts. V2 records retrieval observations only;
//! it cannot qualify evidence or establish decision sufficiency.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Mutex,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"luminous:mace:retrieval-acquisition:v2\0";
const SCHEMA: &str = "mace_retrieval_acquisition_receipt_v2";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcquisitionStatusV2 {
    CandidatesReturned,
    NoCandidateEvidence,
    BudgetExhausted,
    ToolError,
    Abstained,
}

impl AcquisitionStatusV2 {
    fn as_str(self) -> &'static str {
        match self {
            Self::CandidatesReturned => "candidates_returned",
            Self::NoCandidateEvidence => "no_candidate_evidence",
            Self::BudgetExhausted => "budget_exhausted",
            Self::ToolError => "tool_error",
            Self::Abstained => "abstained",
        }
    }
    fn tag(self) -> u8 {
        match self {
            Self::CandidatesReturned => 1,
            Self::NoCandidateEvidence => 2,
            Self::BudgetExhausted => 3,
            Self::ToolError => 4,
            Self::Abstained => 5,
        }
    }
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "candidates_returned" => Self::CandidatesReturned,
            "no_candidate_evidence" => Self::NoCandidateEvidence,
            "budget_exhausted" => Self::BudgetExhausted,
            "tool_error" => Self::ToolError,
            "abstained" => Self::Abstained,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationOutcomeV2 {
    EvidenceObserved,
    NoEvidence,
    ToolError,
}

impl OperationOutcomeV2 {
    fn as_str(self) -> &'static str {
        match self {
            Self::EvidenceObserved => "evidence_observed",
            Self::NoEvidence => "no_evidence",
            Self::ToolError => "tool_error",
        }
    }
    fn tag(self) -> u8 {
        match self {
            Self::EvidenceObserved => 1,
            Self::NoEvidence => 2,
            Self::ToolError => 3,
        }
    }
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "evidence_observed" => Self::EvidenceObserved,
            "no_evidence" => Self::NoEvidence,
            "tool_error" => Self::ToolError,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcquisitionOperationV2 {
    pub tool_id: String,
    pub tool_revision_sha256: String,
    pub action_template_sha256: String,
    pub observed_evidence_ids: Vec<String>,
    pub outcome: OperationOutcomeV2,
    pub elapsed_ns: u64,
    pub inspected_bytes: u64,
    pub inspected_bytes_measured: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcquisitionReceiptV2 {
    pub run_id: String,
    pub query_invocation_id: String,
    pub domain_id: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    pub query_sha256: String,
    pub candidate_set_sha256: Option<String>,
    pub operations: Vec<AcquisitionOperationV2>,
    pub acquisition_status: AcquisitionStatusV2,
    pub tool_call_budget: u32,
    pub tool_calls_used: u32,
    pub elapsed_ns: u64,
    pub inspected_bytes: u64,
    pub inspected_bytes_complete: bool,
    pub created_at_unix: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiptError {
    InvalidObject,
    InvalidField,
    InvalidDigest,
    InvalidState,
    Io,
    CandidateMismatch,
    InvocationConflict,
    Expired,
    Revoked,
    CorruptHistory,
}

/// Verify an externally produced candidate batch against this acquisition receipt.
/// The existing V1 candidate-set commitment already covers sorted records,
/// cardinality, evidence identity, payload digest, scope, purpose and candidate ID.
pub fn verify_candidate_batch(
    receipt: &AcquisitionReceiptV2,
    batch: &crate::candidate_evidence::CandidateEvidenceBatchV1,
) -> Result<(), ReceiptError> {
    receipt.validate()?;
    let recomputed = crate::candidate_evidence::encode_candidate_evidence_jsonl(&batch.evidence)
        .map_err(|_| ReceiptError::CandidateMismatch)?;
    // CandidateEvidenceBatchV1 has public fields. Recompute from the records and
    // compare the entire envelope, not only the fields copied into the receipt.
    // Otherwise callers can mutate backend/scope/purpose metadata while retaining
    // the same evidence vector and candidate-set digest.
    if &recomputed != batch
        || recomputed.run_id != receipt.run_id
        || recomputed.candidate_id != receipt.query_invocation_id
        || recomputed.query_sha256 != receipt.query_sha256
        || Some(&recomputed.candidate_set_sha256) != receipt.candidate_set_sha256.as_ref()
    {
        return Err(ReceiptError::CandidateMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvocationWrite {
    Inserted,
    IdempotentReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InvocationBinding {
    receipt_sha256: String,
    purpose: String,
    scope: String,
    expires_at_unix: u64,
}

/// Thread-safe invocation binding ledger. Revocation is explicit and fail-closed.
/// Durable authority still depends on the caller persisting this state in a
/// separately protected store/checkpoint.
#[derive(Default)]
struct InvocationRegistryState {
    bindings: BTreeMap<(String, String), InvocationBinding>,
    revoked: BTreeSet<(String, String)>,
}

/// The binding and revocation sets share one lock so a revocation cannot race
/// between the registration check and insertion. This is process-local state;
/// durable authority still requires a separately protected persistent store.
#[derive(Default)]
pub struct InvocationRegistry {
    state: Mutex<InvocationRegistryState>,
}

impl InvocationRegistry {
    pub fn record(
        &self,
        receipt: &AcquisitionReceiptV2,
        purpose: &str,
        scope: &str,
        expires_at_unix: u64,
        now_unix: u64,
    ) -> Result<InvocationWrite, ReceiptError> {
        receipt.validate()?;
        if purpose.trim().is_empty() || scope.trim().is_empty() {
            return Err(ReceiptError::InvalidField);
        }
        if expires_at_unix <= now_unix {
            return Err(ReceiptError::Expired);
        }
        let key = (receipt.run_id.clone(), receipt.query_invocation_id.clone());
        let mut state = self.state.lock().map_err(|_| ReceiptError::Io)?;
        if state.revoked.contains(&key) {
            return Err(ReceiptError::Revoked);
        }
        let value = InvocationBinding {
            receipt_sha256: receipt.receipt_sha256(),
            purpose: purpose.into(),
            scope: scope.into(),
            expires_at_unix,
        };
        match state.bindings.get(&key) {
            Some(old) if old == &value => Ok(InvocationWrite::IdempotentReplay),
            Some(_) => Err(ReceiptError::InvocationConflict),
            None => {
                state.bindings.insert(key, value);
                Ok(InvocationWrite::Inserted)
            }
        }
    }

    pub fn revoke(&self, run_id: &str, invocation_id: &str) -> Result<(), ReceiptError> {
        self.state
            .lock()
            .map_err(|_| ReceiptError::Io)?
            .revoked
            .insert((run_id.into(), invocation_id.into()));
        Ok(())
    }

    /// Check the exact registered receipt, rather than accepting only a matching
    /// run/invocation key. Reusing an invocation identifier with altered receipt
    /// content is a conflict even when purpose, scope and expiry still match.
    pub fn check(
        &self,
        receipt: &AcquisitionReceiptV2,
        purpose: &str,
        scope: &str,
        now_unix: u64,
    ) -> Result<(), ReceiptError> {
        receipt.validate()?;
        let key = (receipt.run_id.clone(), receipt.query_invocation_id.clone());
        let state = self.state.lock().map_err(|_| ReceiptError::Io)?;
        if state.revoked.contains(&key) {
            return Err(ReceiptError::Revoked);
        }
        let binding = state
            .bindings
            .get(&key)
            .ok_or(ReceiptError::InvocationConflict)?;
        if binding.receipt_sha256 != receipt.receipt_sha256()
            || binding.purpose != purpose
            || binding.scope != scope
        {
            return Err(ReceiptError::InvocationConflict);
        }
        if now_unix >= binding.expires_at_unix {
            return Err(ReceiptError::Expired);
        }
        Ok(())
    }
}

fn hash_field(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}
fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn str_field<'a>(v: &'a Value, k: &str) -> Result<&'a str, ReceiptError> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or(ReceiptError::InvalidField)
}
fn u64_field(v: &Value, k: &str) -> Result<u64, ReceiptError> {
    v.get(k)
        .and_then(Value::as_u64)
        .ok_or(ReceiptError::InvalidField)
}

impl AcquisitionReceiptV2 {
    pub fn validate(&self) -> Result<(), ReceiptError> {
        if self.run_id.is_empty()
            || self.query_invocation_id.is_empty()
            || self.domain_id.is_empty()
            || self.operations.is_empty()
            || self.operations.len() > 256
            || self.tool_call_budget == 0
            || self.tool_calls_used == 0
            || self.tool_calls_used as usize != self.operations.len()
            || self.tool_calls_used > self.tool_call_budget
        {
            return Err(ReceiptError::InvalidState);
        }
        for h in [
            &self.environment_revision_sha256,
            &self.tool_profile_sha256,
            &self.query_sha256,
        ] {
            if !valid_hash(h) {
                return Err(ReceiptError::InvalidField);
            }
        }
        if self
            .candidate_set_sha256
            .as_ref()
            .is_some_and(|h| !valid_hash(h))
        {
            return Err(ReceiptError::InvalidField);
        }
        for op in &self.operations {
            if op.tool_id.is_empty()
                || !valid_hash(&op.tool_revision_sha256)
                || !valid_hash(&op.action_template_sha256)
                || op.observed_evidence_ids.len() > 10_000
            {
                return Err(ReceiptError::InvalidField);
            }
            let mut ids = BTreeSet::new();
            for id in &op.observed_evidence_ids {
                if id.is_empty() || !ids.insert(id) {
                    return Err(ReceiptError::InvalidField);
                }
            }
        }
        let has_evidence = self.operations.iter().any(|o| {
            o.outcome == OperationOutcomeV2::EvidenceObserved && !o.observed_evidence_ids.is_empty()
        });
        match self.acquisition_status {
            AcquisitionStatusV2::CandidatesReturned
                if !has_evidence || self.candidate_set_sha256.is_none() =>
            {
                return Err(ReceiptError::InvalidState);
            }
            AcquisitionStatusV2::NoCandidateEvidence
                if has_evidence || self.candidate_set_sha256.is_some() =>
            {
                return Err(ReceiptError::InvalidState);
            }
            AcquisitionStatusV2::ToolError
                if !self
                    .operations
                    .iter()
                    .any(|o| o.outcome == OperationOutcomeV2::ToolError) =>
            {
                return Err(ReceiptError::InvalidState);
            }
            _ => {}
        }
        Ok(())
    }

    /// Domain-separated canonical binary digest. JSON key order and whitespace do not affect it.
    pub fn receipt_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for s in [
            &self.run_id,
            &self.query_invocation_id,
            &self.domain_id,
            &self.environment_revision_sha256,
            &self.tool_profile_sha256,
            &self.query_sha256,
        ] {
            hash_field(&mut h, s);
        }
        match &self.candidate_set_sha256 {
            Some(s) => {
                h.update([1]);
                hash_field(&mut h, s);
            }
            None => h.update([0]),
        }
        h.update([self.acquisition_status.tag()]);
        h.update(self.tool_call_budget.to_be_bytes());
        h.update(self.tool_calls_used.to_be_bytes());
        h.update(self.elapsed_ns.to_be_bytes());
        h.update(self.inspected_bytes.to_be_bytes());
        h.update([u8::from(self.inspected_bytes_complete)]);
        h.update(self.created_at_unix.to_be_bytes());
        h.update((self.operations.len() as u64).to_be_bytes());
        for op in &self.operations {
            hash_field(&mut h, &op.tool_id);
            hash_field(&mut h, &op.tool_revision_sha256);
            hash_field(&mut h, &op.action_template_sha256);
            h.update([op.outcome.tag()]);
            h.update(op.elapsed_ns.to_be_bytes());
            h.update(op.inspected_bytes.to_be_bytes());
            h.update([u8::from(op.inspected_bytes_measured)]);
            let mut ids = op.observed_evidence_ids.clone();
            ids.sort();
            h.update((ids.len() as u64).to_be_bytes());
            for id in ids {
                hash_field(&mut h, &id);
            }
        }
        let bytes = h.finalize();
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn to_json_value(&self) -> Value {
        json!({"schema_version":SCHEMA,"run_id":self.run_id,"query_invocation_id":self.query_invocation_id,"domain_id":self.domain_id,
            "environment_revision_sha256":self.environment_revision_sha256,"tool_profile_sha256":self.tool_profile_sha256,"query_sha256":self.query_sha256,
            "candidate_set_sha256":self.candidate_set_sha256,"operations":self.operations.iter().map(|o| json!({"tool_id":o.tool_id,"tool_revision_sha256":o.tool_revision_sha256,
                "action_template_sha256":o.action_template_sha256,"observed_evidence_ids":o.observed_evidence_ids,"outcome":o.outcome.as_str(),"elapsed_ns":o.elapsed_ns,
                "inspected_bytes":o.inspected_bytes,"inspected_bytes_measured":o.inspected_bytes_measured})).collect::<Vec<_>>(),
            "acquisition_status":self.acquisition_status.as_str(),"tool_call_budget":self.tool_call_budget,"tool_calls_used":self.tool_calls_used,"elapsed_ns":self.elapsed_ns,
            "inspected_bytes":self.inspected_bytes,"inspected_bytes_complete":self.inspected_bytes_complete,"created_at_unix":self.created_at_unix,"receipt_sha256":self.receipt_sha256()})
    }

    pub fn from_json_value(v: &Value) -> Result<Self, ReceiptError> {
        let o = v.as_object().ok_or(ReceiptError::InvalidObject)?;
        const FIELDS: [&str; 17] = [
            "schema_version",
            "run_id",
            "query_invocation_id",
            "domain_id",
            "environment_revision_sha256",
            "tool_profile_sha256",
            "query_sha256",
            "candidate_set_sha256",
            "operations",
            "acquisition_status",
            "tool_call_budget",
            "tool_calls_used",
            "elapsed_ns",
            "inspected_bytes",
            "inspected_bytes_complete",
            "created_at_unix",
            "receipt_sha256",
        ];
        if o.len() != FIELDS.len()
            || FIELDS.iter().any(|k| !o.contains_key(*k))
            || str_field(v, "schema_version")? != SCHEMA
        {
            return Err(ReceiptError::InvalidObject);
        }
        let candidate_set_sha256 = match &o["candidate_set_sha256"] {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            _ => return Err(ReceiptError::InvalidField),
        };
        let mut operations = Vec::new();
        for item in o["operations"]
            .as_array()
            .ok_or(ReceiptError::InvalidField)?
        {
            let x = item.as_object().ok_or(ReceiptError::InvalidField)?;
            if x.len() != 8 {
                return Err(ReceiptError::InvalidObject);
            }
            operations.push(AcquisitionOperationV2 {
                tool_id: str_field(item, "tool_id")?.into(),
                tool_revision_sha256: str_field(item, "tool_revision_sha256")?.into(),
                action_template_sha256: str_field(item, "action_template_sha256")?.into(),
                observed_evidence_ids: item["observed_evidence_ids"]
                    .as_array()
                    .ok_or(ReceiptError::InvalidField)?
                    .iter()
                    .map(|x| {
                        x.as_str()
                            .map(str::to_owned)
                            .ok_or(ReceiptError::InvalidField)
                    })
                    .collect::<Result<_, _>>()?,
                outcome: OperationOutcomeV2::parse(str_field(item, "outcome")?)
                    .ok_or(ReceiptError::InvalidField)?,
                elapsed_ns: u64_field(item, "elapsed_ns")?,
                inspected_bytes: u64_field(item, "inspected_bytes")?,
                inspected_bytes_measured: item["inspected_bytes_measured"]
                    .as_bool()
                    .ok_or(ReceiptError::InvalidField)?,
            });
        }
        let budget = u32::try_from(u64_field(v, "tool_call_budget")?)
            .map_err(|_| ReceiptError::InvalidField)?;
        let used = u32::try_from(u64_field(v, "tool_calls_used")?)
            .map_err(|_| ReceiptError::InvalidField)?;
        let r = Self {
            run_id: str_field(v, "run_id")?.into(),
            query_invocation_id: str_field(v, "query_invocation_id")?.into(),
            domain_id: str_field(v, "domain_id")?.into(),
            environment_revision_sha256: str_field(v, "environment_revision_sha256")?.into(),
            tool_profile_sha256: str_field(v, "tool_profile_sha256")?.into(),
            query_sha256: str_field(v, "query_sha256")?.into(),
            candidate_set_sha256,
            operations,
            acquisition_status: AcquisitionStatusV2::parse(str_field(v, "acquisition_status")?)
                .ok_or(ReceiptError::InvalidField)?,
            tool_call_budget: budget,
            tool_calls_used: used,
            elapsed_ns: u64_field(v, "elapsed_ns")?,
            inspected_bytes: u64_field(v, "inspected_bytes")?,
            inspected_bytes_complete: v["inspected_bytes_complete"]
                .as_bool()
                .ok_or(ReceiptError::InvalidField)?,
            created_at_unix: u64_field(v, "created_at_unix")?,
        };
        r.validate()?;
        let supplied = str_field(v, "receipt_sha256")?;
        if !valid_hash(supplied) || supplied != r.receipt_sha256() {
            return Err(ReceiptError::InvalidDigest);
        }
        Ok(r)
    }
}

/// Explicit protocol dispatch. V1 remains an opaque historical payload here; callers
/// must use its existing V1 decoder and digest rules rather than upgrade its meaning.
pub fn dispatch_schema_version(v: &Value) -> Result<&'static str, ReceiptError> {
    match v.get("schema_version").and_then(Value::as_str) {
        Some(SCHEMA) => Ok(SCHEMA),
        Some("mace_retrieval_trace_receipt_v1") => Ok("mace_retrieval_trace_receipt_v1"),
        _ => Err(ReceiptError::InvalidObject),
    }
}

/// Append one V2 receipt per line. Existing bytes are retained, making this suitable
/// for local restart/replay qualification; atomic directory-level durability is a
/// deployment concern and is intentionally not claimed by this prototype.
pub fn append_jsonl(path: &Path, receipt: &AcquisitionReceiptV2) -> Result<(), ReceiptError> {
    receipt.validate()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|_| ReceiptError::Io)?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .append(true)
        .open(path)
        .map_err(|_| ReceiptError::Io)?;
    f.lock().map_err(|_| ReceiptError::Io)?;

    let mut history = Vec::new();
    f.seek(SeekFrom::Start(0)).map_err(|_| ReceiptError::Io)?;
    f.read_to_end(&mut history).map_err(|_| ReceiptError::Io)?;

    // Treat only a non-newline-terminated final fragment as an interrupted append.
    // Validate all complete records before mutating the journal.
    let incomplete_tail = !history.is_empty() && *history.last().unwrap() != b'\n';
    if incomplete_tail {
        let end = history
            .iter()
            .rposition(|b| *b == b'\n')
            .map_or(0, |i| i + 1);
        history.truncate(end);
    }

    let key = (receipt.run_id.clone(), receipt.query_invocation_id.clone());
    let incoming_digest = receipt.receipt_sha256();
    let mut invocations = BTreeMap::<(String, String), String>::new();
    let mut idempotent_replay = false;

    for line in history
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
    {
        let value: Value =
            serde_json::from_slice(line).map_err(|_| ReceiptError::CorruptHistory)?;
        let (old_key, digest) = match dispatch_schema_version(&value)? {
            SCHEMA => {
                let old = AcquisitionReceiptV2::from_json_value(&value)
                    .map_err(|_| ReceiptError::CorruptHistory)?;
                (
                    (old.run_id.clone(), old.query_invocation_id.clone()),
                    old.receipt_sha256(),
                )
            }
            "mace_retrieval_trace_receipt_v1" => {
                let old = crate::retrieval_policy_memory::RetrievalTraceReceiptV1::from_json_value(
                    &value,
                )
                .map_err(|_| ReceiptError::CorruptHistory)?;
                (
                    (old.run_id.clone(), old.query_invocation_id.clone()),
                    old.receipt_sha256(),
                )
            }
            _ => return Err(ReceiptError::CorruptHistory),
        };

        // Match recovery semantics: the entire existing journal must have unique
        // invocation identities, not merely a unique key for the incoming record.
        if let Some(previous_digest) = invocations.insert(old_key.clone(), digest.clone()) {
            return Err(if previous_digest == digest {
                ReceiptError::CorruptHistory
            } else {
                ReceiptError::InvocationConflict
            });
        }

        if old_key == key {
            if digest != incoming_digest {
                return Err(ReceiptError::InvocationConflict);
            }
            idempotent_replay = true;
        }
    }

    // Persist tail repair even when the incoming receipt is an idempotent replay.
    if incomplete_tail {
        f.set_len(history.len() as u64)
            .map_err(|_| ReceiptError::Io)?;
        f.sync_all().map_err(|_| ReceiptError::Io)?;
    }

    if idempotent_replay {
        f.unlock().map_err(|_| ReceiptError::Io)?;
        return Ok(());
    }

    f.seek(SeekFrom::End(0)).map_err(|_| ReceiptError::Io)?;
    let mut encoded = serde_json::to_vec(&receipt.to_json_value()).map_err(|_| ReceiptError::Io)?;
    encoded.push(b'\n');
    let result = f
        .write_all(&encoded)
        .and_then(|()| f.sync_all())
        .map_err(|_| ReceiptError::Io);
    let unlock = f.unlock().map_err(|_| ReceiptError::Io);
    result.and(unlock)
}

/// Validate complete history records and discard only an incomplete final line.
/// A malformed newline-terminated record is corruption and is never skipped.
pub fn recover_jsonl(path: &Path) -> Result<Vec<Value>, ReceiptError> {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|_| ReceiptError::Io)?;
    f.lock().map_err(|_| ReceiptError::Io)?;
    let mut bytes = Vec::new();
    f.seek(SeekFrom::Start(0)).map_err(|_| ReceiptError::Io)?;
    f.read_to_end(&mut bytes).map_err(|_| ReceiptError::Io)?;
    if !bytes.is_empty() && *bytes.last().unwrap() != b'\n' {
        let last_newline = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        f.set_len(last_newline as u64)
            .map_err(|_| ReceiptError::Io)?;
        bytes.truncate(last_newline);
        f.sync_all().map_err(|_| ReceiptError::Io)?;
    }
    let mut records = Vec::new();
    let mut invocations = BTreeMap::<(String, String), String>::new();
    for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        let value: Value =
            serde_json::from_slice(line).map_err(|_| ReceiptError::CorruptHistory)?;
        match dispatch_schema_version(&value)? {
            SCHEMA => {
                let r = AcquisitionReceiptV2::from_json_value(&value)
                    .map_err(|_| ReceiptError::CorruptHistory)?;
                let key = (r.run_id.clone(), r.query_invocation_id.clone());
                let digest = r.receipt_sha256();
                if let Some(old) = invocations.get(&key) {
                    return Err(if old == &digest {
                        ReceiptError::CorruptHistory
                    } else {
                        ReceiptError::InvocationConflict
                    });
                }
                invocations.insert(key, digest);
            }
            "mace_retrieval_trace_receipt_v1" => {
                let r = crate::retrieval_policy_memory::RetrievalTraceReceiptV1::from_json_value(
                    &value,
                )
                .map_err(|_| ReceiptError::CorruptHistory)?;
                let key = (r.run_id.clone(), r.query_invocation_id.clone());
                let digest = r.receipt_sha256();
                if let Some(old) = invocations.get(&key) {
                    return Err(if old == &digest {
                        ReceiptError::CorruptHistory
                    } else {
                        ReceiptError::InvocationConflict
                    });
                }
                invocations.insert(key, digest);
            }
            _ => return Err(ReceiptError::CorruptHistory),
        }
        records.push(value);
    }
    let _ = f.unlock();
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_evidence::{
        CandidateEvidenceV1, EvidenceModalityV1, encode_candidate_evidence_jsonl,
    };
    fn h() -> String {
        "a".repeat(64)
    }
    fn receipt() -> AcquisitionReceiptV2 {
        AcquisitionReceiptV2 {
            run_id: "r".into(),
            query_invocation_id: "i".into(),
            domain_id: "d".into(),
            environment_revision_sha256: h(),
            tool_profile_sha256: h(),
            query_sha256: h(),
            candidate_set_sha256: Some("b".repeat(64)),
            operations: vec![AcquisitionOperationV2 {
                tool_id: "search".into(),
                tool_revision_sha256: h(),
                action_template_sha256: h(),
                observed_evidence_ids: vec!["e1".into()],
                outcome: OperationOutcomeV2::EvidenceObserved,
                elapsed_ns: 7,
                inspected_bytes: 9,
                inspected_bytes_measured: true,
            }],
            acquisition_status: AcquisitionStatusV2::CandidatesReturned,
            tool_call_budget: 1,
            tool_calls_used: 1,
            elapsed_ns: 7,
            inspected_bytes: 9,
            inspected_bytes_complete: true,
            created_at_unix: 10,
        }
    }

    fn candidate() -> CandidateEvidenceV1 {
        CandidateEvidenceV1 {
            run_id: "r".into(),
            candidate_id: "i".into(),
            backend_id: "search".into(),
            backend_revision_sha256: "a".repeat(64),
            query_scope: "scope".into(),
            query_purpose: "purpose".into(),
            query_sha256: "a".repeat(64),
            evidence_id: "e1".into(),
            source_id: "source".into(),
            source_sha256: "b".repeat(64),
            payload_sha256: "c".repeat(64),
            payload_ref: "opaque:payload".into(),
            modality: EvidenceModalityV1::Text,
            retrieval_channel: "bm25".into(),
            retrieval_rank: 1,
            retrieval_score: None,
            parent_evidence_ids: Vec::new(),
            derivation_receipt_sha256: None,
        }
    }

    #[test]
    fn candidate_commitment_binds_identity_payload_count_and_invocation() {
        let mut r = receipt();
        let records = vec![candidate()];
        let batch = encode_candidate_evidence_jsonl(&records).unwrap();
        r.candidate_set_sha256 = Some(batch.candidate_set_sha256.clone());
        verify_candidate_batch(&r, &batch).unwrap();
        let mut changed = records.clone();
        changed[0].payload_sha256 = "d".repeat(64);
        let substituted = encode_candidate_evidence_jsonl(&changed).unwrap();
        assert_eq!(
            verify_candidate_batch(&r, &substituted),
            Err(ReceiptError::CandidateMismatch)
        );
        let mut omitted = records.clone();
        omitted[0].evidence_id = "different-id".into();
        let omitted = encode_candidate_evidence_jsonl(&omitted).unwrap();
        assert_eq!(
            verify_candidate_batch(&r, &omitted),
            Err(ReceiptError::CandidateMismatch)
        );
        let mut duplicate = records.clone();
        duplicate.push(duplicate[0].clone());
        assert!(encode_candidate_evidence_jsonl(&duplicate).is_err());
        let mut added = records.clone();
        let mut second = added[0].clone();
        second.evidence_id = "e2".into();
        second.retrieval_rank = 2;
        added.push(second);
        let added = encode_candidate_evidence_jsonl(&added).unwrap();
        assert_eq!(
            verify_candidate_batch(&r, &added),
            Err(ReceiptError::CandidateMismatch)
        );
        let mut rebound = records;
        rebound[0].candidate_id = "other-invocation".into();
        let rebound = encode_candidate_evidence_jsonl(&rebound).unwrap();
        assert_eq!(
            verify_candidate_batch(&r, &rebound),
            Err(ReceiptError::CandidateMismatch)
        );
    }

    #[test]
    fn candidate_envelope_metadata_cannot_diverge_from_committed_records() {
        let mut r = receipt();
        let batch = encode_candidate_evidence_jsonl(&[candidate()]).unwrap();
        r.candidate_set_sha256 = Some(batch.candidate_set_sha256.clone());
        verify_candidate_batch(&r, &batch).unwrap();

        let mut changed_backend = batch.clone();
        changed_backend.backend_id = "substituted-backend".into();
        assert_eq!(
            verify_candidate_batch(&r, &changed_backend),
            Err(ReceiptError::CandidateMismatch)
        );

        let mut changed_revision = batch.clone();
        changed_revision.backend_revision_sha256 = "d".repeat(64);
        assert_eq!(
            verify_candidate_batch(&r, &changed_revision),
            Err(ReceiptError::CandidateMismatch)
        );

        let mut changed_scope = batch.clone();
        changed_scope.query_scope = "other-scope".into();
        assert_eq!(
            verify_candidate_batch(&r, &changed_scope),
            Err(ReceiptError::CandidateMismatch)
        );

        let mut changed_purpose = batch.clone();
        changed_purpose.query_purpose = "other-purpose".into();
        assert_eq!(
            verify_candidate_batch(&r, &changed_purpose),
            Err(ReceiptError::CandidateMismatch)
        );

        let mut changed_digest = batch;
        changed_digest.candidate_set_sha256 = "e".repeat(64);
        assert_eq!(
            verify_candidate_batch(&r, &changed_digest),
            Err(ReceiptError::CandidateMismatch)
        );
    }

    #[test]
    fn invocation_registry_is_idempotent_and_rejects_receipt_rebinding_expiry_and_revocation() {
        let registry = InvocationRegistry::default();
        let r = receipt();
        assert_eq!(
            registry.record(&r, "answer", "tenant-1", 100, 10),
            Ok(InvocationWrite::Inserted)
        );
        assert_eq!(
            registry.record(&r, "answer", "tenant-1", 100, 10),
            Ok(InvocationWrite::IdempotentReplay)
        );
        assert_eq!(
            registry.check(&r, "answer", "tenant-1", 10),
            Ok(())
        );

        let mut rebound = r.clone();
        rebound.elapsed_ns += 1;
        assert_eq!(
            registry.check(&rebound, "answer", "tenant-1", 10),
            Err(ReceiptError::InvocationConflict)
        );
        assert_eq!(
            registry.record(&r, "different-purpose", "tenant-1", 100, 10),
            Err(ReceiptError::InvocationConflict)
        );
        assert_eq!(
            registry.check(&r, "answer", "tenant-2", 10),
            Err(ReceiptError::InvocationConflict)
        );
        assert_eq!(
            registry.check(&r, "answer", "tenant-1", 100),
            Err(ReceiptError::Expired)
        );
        registry.revoke("r", "i").unwrap();
        assert_eq!(
            registry.check(&r, "answer", "tenant-1", 10),
            Err(ReceiptError::Revoked)
        );
        assert_eq!(
            registry.record(&r, "answer", "tenant-1", 100, 10),
            Err(ReceiptError::Revoked)
        );
    }
    #[test]
    fn v2_round_trip_and_digest_domain_are_stable() {
        let r = receipt();
        r.validate().unwrap();
        assert_eq!(
            r.receipt_sha256(),
            "81d1dc85c850db8b4cbce896459e00e95b9d91b7bda8659b05f2d59252805f56"
        );
        let v = r.to_json_value();
        assert_eq!(AcquisitionReceiptV2::from_json_value(&v), Ok(r));
        assert_eq!(dispatch_schema_version(&v), Ok(SCHEMA));
    }
    #[test]
    fn malformed_and_unknown_versions_fail_closed() {
        let mut v = receipt().to_json_value();
        v["extra"] = json!(true);
        assert_eq!(
            AcquisitionReceiptV2::from_json_value(&v),
            Err(ReceiptError::InvalidObject)
        );
        assert_eq!(
            dispatch_schema_version(&json!({"schema_version":"mace_retrieval_trace_receipt_v3"})),
            Err(ReceiptError::InvalidObject)
        );
    }
    #[test]
    fn candidates_cannot_be_encoded_as_decision_sufficiency() {
        assert!(
            !receipt()
                .to_json_value()
                .to_string()
                .contains("evidence_sufficient")
        );
        let mut r = receipt();
        r.candidate_set_sha256 = None;
        assert_eq!(r.validate(), Err(ReceiptError::InvalidState));
    }
    #[test]
    fn persistence_replays_v2_without_rewriting_prior_records() {
        let p = std::env::temp_dir().join(format!("mace-v2-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&p);
        let r = receipt();
        append_jsonl(&p, &r).unwrap();
        let bytes = fs::read(&p).unwrap();
        let v: Value = serde_json::from_slice(bytes.strip_suffix(b"\n").unwrap()).unwrap();
        assert_eq!(AcquisitionReceiptV2::from_json_value(&v), Ok(r));
        let _ = fs::remove_file(p);
    }

    #[test]
    fn persistence_recovers_partial_tail_rejects_corruption_and_is_idempotent() {
        let p = std::env::temp_dir().join(format!("mace-v2-recover-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&p);
        let r = receipt();
        append_jsonl(&p, &r).unwrap();
        let before = fs::read(&p).unwrap();
        append_jsonl(&p, &r).unwrap();
        assert_eq!(fs::read(&p).unwrap(), before);
        OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{partial")
            .unwrap();
        assert_eq!(recover_jsonl(&p).unwrap().len(), 1);
        let recovered = fs::read(&p).unwrap();
        assert_eq!(recovered, before);
        OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{broken}\n")
            .unwrap();
        assert_eq!(recover_jsonl(&p), Err(ReceiptError::CorruptHistory));
        let _ = fs::remove_file(p);
    }

    #[test]
    fn append_rejects_duplicate_history_for_an_unrelated_invocation() {
        let p = std::env::temp_dir().join(format!(
            "mace-v2-duplicate-history-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&p);

        let first = receipt();
        let mut unrelated = receipt();
        unrelated.query_invocation_id = "other-invocation".into();
        let mut conflicting = unrelated.clone();
        conflicting.elapsed_ns += 1;
        let incoming = {
            let mut value = receipt();
            value.query_invocation_id = "new-invocation".into();
            value
        };

        let mut bytes = serde_json::to_vec(&first.to_json_value()).unwrap();
        bytes.push(b'\n');
        bytes.extend(serde_json::to_vec(&unrelated.to_json_value()).unwrap());
        bytes.push(b'\n');
        bytes.extend(serde_json::to_vec(&conflicting.to_json_value()).unwrap());
        bytes.push(b'\n');
        fs::write(&p, &bytes).unwrap();

        assert_eq!(
            append_jsonl(&p, &incoming),
            Err(ReceiptError::InvocationConflict)
        );
        assert_eq!(fs::read(&p).unwrap(), bytes);
        let _ = fs::remove_file(p);
    }

    #[test]
    fn idempotent_append_durably_repairs_an_incomplete_tail() {
        let p = std::env::temp_dir().join(format!(
            "mace-v2-idempotent-tail-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&p);
        let r = receipt();
        append_jsonl(&p, &r).unwrap();
        let complete = fs::read(&p).unwrap();

        OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{interrupted")
            .unwrap();

        assert_eq!(
            append_jsonl(&p, &r),
            Ok(())
        );
        assert_eq!(fs::read(&p).unwrap(), complete);
        let _ = fs::remove_file(p);
    }

    #[test]
    fn concurrent_duplicate_appends_commit_once() {
        let p =
            std::env::temp_dir().join(format!("mace-v2-concurrent-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&p);
        let p = std::sync::Arc::new(p);
        let threads = (0..8)
            .map(|_| {
                let p = p.clone();
                std::thread::spawn(move || append_jsonl(&p, &receipt()).unwrap())
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(recover_jsonl(&p).unwrap().len(), 1);
        let _ = fs::remove_file(&*p);
    }

    #[test]
    fn concurrent_conflicting_invocation_writes_have_one_winner() {
        let p = std::sync::Arc::new(
            std::env::temp_dir().join(format!("mace-v2-conflict-{}.jsonl", std::process::id())),
        );
        let _ = fs::remove_file(&*p);
        let a = receipt();
        let mut b = receipt();
        b.elapsed_ns += 1;
        let p1 = p.clone();
        let t1 = std::thread::spawn(move || append_jsonl(&p1, &a));
        let p2 = p.clone();
        let t2 = std::thread::spawn(move || append_jsonl(&p2, &b));
        let results = [t1.join().unwrap(), t2.join().unwrap()];
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|r| **r == Err(ReceiptError::InvocationConflict))
                .count(),
            1
        );
        assert_eq!(recover_jsonl(&p).unwrap().len(), 1);
        let _ = fs::remove_file(&*p);
    }

    #[test]
    fn mixed_legacy_v1_and_v2_history_replays_without_upgrading_v1() {
        use crate::retrieval_policy_memory::{
            RetrievalOperationOutcomeV1, RetrievalOperationV1, RetrievalStopReasonV1,
            RetrievalTraceReceiptV1,
        };
        let p = std::env::temp_dir().join(format!("mace-v2-mixed-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&p);
        let legacy = RetrievalTraceReceiptV1 {
            run_id: "r".into(),
            query_invocation_id: "legacy-i".into(),
            domain_id: "d".into(),
            environment_revision_sha256: h(),
            tool_profile_sha256: h(),
            query_sha256: h(),
            candidate_set_sha256: Some("b".repeat(64)),
            operations: vec![RetrievalOperationV1 {
                tool_id: "search".into(),
                tool_revision_sha256: h(),
                action_template_sha256: h(),
                observed_evidence_ids: vec!["legacy-e".into()],
                outcome: RetrievalOperationOutcomeV1::EvidenceObserved,
                elapsed_ns: 1,
                inspected_bytes: 0,
                inspected_bytes_measured: false,
            }],
            stop_reason: RetrievalStopReasonV1::EvidenceSufficient,
            tool_call_budget: 1,
            tool_calls_used: 1,
            elapsed_ns: 1,
            inspected_bytes: 0,
            inspected_bytes_complete: false,
            created_at_unix: 1,
        };
        let mut bytes = serde_json::to_vec(&legacy.to_json_value()).unwrap();
        bytes.push(b'\n');
        fs::write(&p, bytes).unwrap();
        let current = AcquisitionReceiptV2 {
            query_invocation_id: "v2-i".into(),
            candidate_set_sha256: None,
            operations: vec![AcquisitionOperationV2 {
                tool_id: "search".into(),
                tool_revision_sha256: h(),
                action_template_sha256: h(),
                observed_evidence_ids: vec![],
                outcome: OperationOutcomeV2::NoEvidence,
                elapsed_ns: 1,
                inspected_bytes: 0,
                inspected_bytes_measured: false,
            }],
            acquisition_status: AcquisitionStatusV2::NoCandidateEvidence,
            ..receipt()
        };
        append_jsonl(&p, &current).unwrap();
        let records = recover_jsonl(&p).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0]["schema_version"],
            "mace_retrieval_trace_receipt_v1"
        );
        assert_eq!(records[0]["stop_reason"], "evidence_sufficient");
        assert_eq!(records[1]["schema_version"], SCHEMA);
        let _ = fs::remove_file(p);
    }
}
