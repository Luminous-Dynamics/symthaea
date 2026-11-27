// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Single-writer, crash-safe AML Add/Search store backed by redb.
//!
//! The database opens with redb's default ExclusiveWriter mode. A second
//! process cannot open the file while this store is live. The hosting HTTP
//! service must also serialize calls through one DurableAmlServiceV1 instance.
//! This backend deliberately does not advertise distributed multi-writer use.
//!
//! On open, only durable Reserved ledger entries left by a prior process are
//! reclaimed. ExclusiveWriter proves the previous writer process is gone.
//! Written-but-not-finalized rows keep their receipt and are completed through
//! the coordinator's normal reconcile, verify, read-back and finalize path.
//!
//! Search uses deterministic BM25 lexical ranking over each user's own rows.
//! It is an internal lexical baseline/fallback, not the AML Academic embedding
//! profile and not an external SOTA claim.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use redb::{Builder, Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    add_payload_digest, AddRequest, AddVisibilityReceiptV1, AddVisibilityVerifierV1,
    AmlReject, ContentPart, DurableAddKeyV1, DurableAddReconciliationV1,
    DurableAddReservationV1, DurableMemoryRecordV1, DurableScopedSearchItemV1,
    DurableStoreErrorV1, DurableAmlStoreV1, FrozenAdapterWordCounter,
    MessageContent, SearchItem, SearchRequest, SearchResponse, Track,
};

const ADD_LEDGER: TableDefinition<&str, &[u8]> =
    TableDefinition::new("mace_aml_add_ledger_v1");
const MEMORY_RECORDS: TableDefinition<&str, &[u8]> =
    TableDefinition::new("mace_aml_memory_records_v1");
const CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredReceiptV1 {
    track: String,
    request_id: String,
    user_id: String,
    session_id: String,
    request_payload_digest: String,
    write_receipt_digest: String,
    search_visibility_receipt_digest: String,
    persisted_at_ms: i64,
    searchable_at_ms: i64,
    visible_memory_ids: Vec<String>,
}

impl From<&AddVisibilityReceiptV1> for StoredReceiptV1 {
    fn from(receipt: &AddVisibilityReceiptV1) -> Self {
        Self {
            track: track_name(receipt.track).to_owned(),
            request_id: receipt.request_id.clone(),
            user_id: receipt.user_id.clone(),
            session_id: receipt.session_id.clone(),
            request_payload_digest: receipt.request_payload_digest.clone(),
            write_receipt_digest: receipt.write_receipt_digest.clone(),
            search_visibility_receipt_digest: receipt.search_visibility_receipt_digest.clone(),
            persisted_at_ms: receipt.persisted_at_ms,
            searchable_at_ms: receipt.searchable_at_ms,
            visible_memory_ids: receipt.visible_memory_ids.clone(),
        }
    }
}

impl TryFrom<StoredReceiptV1> for AddVisibilityReceiptV1 {
    type Error = DurableStoreErrorV1;

    fn try_from(value: StoredReceiptV1) -> Result<Self, Self::Error> {
        Ok(Self {
            track: parse_track(&value.track)?,
            request_id: value.request_id,
            user_id: value.user_id,
            session_id: value.session_id,
            request_payload_digest: value.request_payload_digest,
            write_receipt_digest: value.write_receipt_digest,
            search_visibility_receipt_digest: value.search_visibility_receipt_digest,
            persisted_at_ms: value.persisted_at_ms,
            searchable_at_ms: value.searchable_at_ms,
            visible_memory_ids: value.visible_memory_ids,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredLedgerEntryV1 {
    idempotency_digest: String,
    state: String,
    receipt: Option<StoredReceiptV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum StoredContentV1 {
    Text(String),
    Parts(Vec<StoredPartV1>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum StoredPartV1 {
    Text(String),
    ImageDataUri(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredMemoryRecordV1 {
    id: String,
    user_id: String,
    origin_request_id: String,
    source_request_payload_digest: String,
    track: String,
    content: StoredContentV1,
}

impl StoredMemoryRecordV1 {
    fn from_durable(track: Track, record: &DurableMemoryRecordV1) -> Self {
        Self {
            id: record.id.clone(),
            user_id: record.user_id.clone(),
            origin_request_id: record.origin_request_id.clone(),
            source_request_payload_digest: record.source_request_payload_digest.clone(),
            track: track_name(track).to_owned(),
            content: content_to_stored(&record.content),
        }
    }

    fn into_durable(self) -> DurableMemoryRecordV1 {
        DurableMemoryRecordV1 {
            id: self.id,
            user_id: self.user_id,
            origin_request_id: self.origin_request_id,
            source_request_payload_digest: self.source_request_payload_digest,
            content: stored_to_content(self.content),
        }
    }
}

pub struct RedbDurableAmlStoreV1 {
    database: Database,
}

impl RedbDurableAmlStoreV1 {
    /// Open or create a store. The redb default ExclusiveWriter mode excludes
    /// other processes. Startup reclaims only reservations that have no write
    /// receipt; committed writes remain recoverable.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DurableStoreErrorV1> {
        let mut builder = Builder::new();
        builder.set_cache_size(CACHE_BYTES);
        builder.set_check_integrity_on_open(true);
        let database = builder.create(path).map_err(backend_error)?;
        {
            let transaction = database.begin_write().map_err(backend_error)?;
            {
                let _ledger = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
                let _records = transaction.open_table(MEMORY_RECORDS).map_err(backend_error)?;
            }
            transaction.commit().map_err(backend_error)?;
        }
        let store = Self { database };
        store.reclaim_abandoned_reservations()?;
        Ok(store)
    }

    fn reclaim_abandoned_reservations(&self) -> Result<(), DurableStoreErrorV1> {
        // ExclusiveWriter excludes every other process. Any Reserved row when
        // this process opens the file belongs to an exited prior writer.
        let transaction = self.database.begin_write().map_err(backend_error)?;
        let abandoned = {
            let table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            let mut keys = Vec::new();
            for item in table.iter().map_err(backend_error)? {
                let (key, value) = item.map_err(backend_error)?;
                let entry: StoredLedgerEntryV1 =
                    serde_json::from_slice(value.value()).map_err(backend_error)?;
                if entry.state == "reserved" && entry.receipt.is_none() {
                    keys.push(key.value().to_owned());
                }
            }
            keys
        };
        if !abandoned.is_empty() {
            let mut table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            for key in abandoned {
                table.remove(key.as_str()).map_err(backend_error)?;
            }
        }
        transaction.commit().map_err(backend_error)?;
        Ok(())
    }

    fn ledger_key(key: &DurableAddKeyV1) -> String {
        format!(
            "{}:{}:{}",
            hex_component(track_name(key.track).as_bytes()),
            hex_component(key.user_id.as_bytes()),
            hex_component(key.request_id.as_bytes())
        )
    }

    fn memory_key(user_id: &str, memory_id: &str) -> String {
        format!(
            "{}:{}",
            hex_component(user_id.as_bytes()),
            hex_component(memory_id.as_bytes())
        )
    }

    fn memory_user_prefix(user_id: &str) -> String {
        format!("{}:", hex_component(user_id.as_bytes()))
    }

    fn load_ledger(
        &self,
        key: &DurableAddKeyV1,
    ) -> Result<Option<StoredLedgerEntryV1>, DurableStoreErrorV1> {
        let transaction = self.database.begin_read().map_err(backend_error)?;
        let table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
        let stored = table.get(Self::ledger_key(key).as_str()).map_err(backend_error)?;
        stored
            .map(|guard| serde_json::from_slice(guard.value()).map_err(backend_error))
            .transpose()
    }

    fn read_memory_record(
        &self,
        user_id: &str,
        memory_id: &str,
    ) -> Result<Option<DurableMemoryRecordV1>, DurableStoreErrorV1> {
        let transaction = self.database.begin_read().map_err(backend_error)?;
        let table = transaction.open_table(MEMORY_RECORDS).map_err(backend_error)?;
        let stored = table
            .get(Self::memory_key(user_id, memory_id).as_str())
            .map_err(backend_error)?;
        stored.map(|guard| {
            let record: StoredMemoryRecordV1 =
                serde_json::from_slice(guard.value()).map_err(backend_error)?;
            if record.user_id != user_id || record.id != memory_id {
                return Err(DurableStoreErrorV1::Backend(
                    "memory key/value tenant or identifier mismatch".into(),
                ));
            }
            Ok(record.into_durable())
        }).transpose()
    }

    fn visible_records_for_user(
        &self,
        user_id: &str,
        track: Track,
    ) -> Result<Vec<StoredMemoryRecordV1>, DurableStoreErrorV1> {
        let transaction = self.database.begin_read().map_err(backend_error)?;
        let table = transaction.open_table(MEMORY_RECORDS).map_err(backend_error)?;
        let lower = Self::memory_user_prefix(user_id);
        // Memory-key suffixes are lowercase hexadecimal. The sentinel 'g'
        // is greater than every allowed suffix character, so the range is
        // restricted to this exact user partition.
        let upper = format!("{lower}g");
        let range = table.range(lower.as_str()..upper.as_str()).map_err(backend_error)?;
        let mut records = Vec::new();
        for item in range {
            let (_key, value) = item.map_err(backend_error)?;
            let record: StoredMemoryRecordV1 =
                serde_json::from_slice(value.value()).map_err(backend_error)?;
            if record.user_id != user_id {
                return Err(DurableStoreErrorV1::Backend(
                    "user-partition index returned another tenant".into(),
                ));
            }
            if record.track == track_name(track) {
                records.push(record);
            }
        }
        Ok(records)
    }
}

impl DurableAmlStoreV1 for RedbDurableAmlStoreV1 {
    fn reserve_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<DurableAddReservationV1, DurableStoreErrorV1> {
        let ledger_key = Self::ledger_key(key);
        let transaction = self.database.begin_write().map_err(backend_error)?;
        let result = {
            let mut table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            match table.get(ledger_key.as_str()).map_err(backend_error)? {
                Some(existing) => {
                    let entry: StoredLedgerEntryV1 =
                        serde_json::from_slice(existing.value()).map_err(backend_error)?;
                    if entry.idempotency_digest != key.idempotency_digest {
                        return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
                    }
                    match entry.state.as_str() {
                        "finalized" => {
                            let receipt = entry.receipt.ok_or_else(|| {
                                DurableStoreErrorV1::Backend(
                                    "finalized ledger entry has no receipt".into(),
                                )
                            })?;
                            DurableAddReservationV1::Committed(receipt.try_into()?)
                        }
                        "reserved" | "written" => DurableAddReservationV1::InFlight,
                        other => {
                            return Err(DurableStoreErrorV1::Backend(format!(
                                "unknown Add ledger state {other}"
                            )));
                        }
                    }
                }
                None => {
                    let entry = StoredLedgerEntryV1 {
                        idempotency_digest: key.idempotency_digest.clone(),
                        state: "reserved".into(),
                        receipt: None,
                    };
                    let bytes = serde_json::to_vec(&entry).map_err(backend_error)?;
                    table.insert(ledger_key.as_str(), bytes.as_slice()).map_err(backend_error)?;
                    DurableAddReservationV1::New
                }
            }
        };
        transaction.commit().map_err(backend_error)?;
        Ok(result)
    }

    fn persist_and_index(
        &mut self,
        key: &DurableAddKeyV1,
        track: Track,
        request: &AddRequest,
    ) -> Result<AddVisibilityReceiptV1, DurableStoreErrorV1> {
        if track != key.track || request.user_id != key.user_id || request.request_id != key.request_id {
            return Err(DurableStoreErrorV1::Backend(
                "persist call does not match durable Add key".into(),
            ));
        }
        let ledger_key = Self::ledger_key(key);
        let transaction = self.database.begin_write().map_err(backend_error)?;
        let (receipt, memory_rows) = {
            let mut ledger = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            let existing = ledger
                .get(ledger_key.as_str())
                .map_err(backend_error)?
                .ok_or_else(|| DurableStoreErrorV1::Backend(
                    "Add persistence without a durable reservation".into(),
                ))?;
            let entry: StoredLedgerEntryV1 =
                serde_json::from_slice(existing.value()).map_err(backend_error)?;
            if entry.idempotency_digest != key.idempotency_digest {
                return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
            }
            if entry.state == "written" || entry.state == "finalized" {
                let receipt: AddVisibilityReceiptV1 = entry.receipt.ok_or_else(|| {
                    DurableStoreErrorV1::Backend("written ledger entry has no receipt".into())
                })?.try_into()?;
                return Ok(receipt);
            }
            if entry.state != "reserved" {
                return Err(DurableStoreErrorV1::Backend(format!(
                    "cannot persist from Add state {}", entry.state
                )));
            }

            let payload_digest = add_payload_digest(track, request);
            let memory_rows: Vec<DurableMemoryRecordV1> = request
                .messages
                .iter()
                .enumerate()
                .map(|(index, message)| DurableMemoryRecordV1 {
                    id: format!("mace-{}-{index}", &payload_digest[..24]),
                    user_id: request.user_id.clone(),
                    origin_request_id: request.request_id.clone(),
                    source_request_payload_digest: payload_digest.clone(),
                    content: message.content.clone(),
                })
                .collect();
            let visible_memory_ids: Vec<String> =
                memory_rows.iter().map(|row| row.id.clone()).collect();
            let now = now_ms()?;
            let receipt = AddVisibilityReceiptV1 {
                track,
                request_id: request.request_id.clone(),
                user_id: request.user_id.clone(),
                session_id: request.session_id.clone(),
                request_payload_digest: payload_digest.clone(),
                write_receipt_digest: receipt_digest(
                    b"luminous:mace:redb-write:v1\0",
                    &payload_digest,
                    now,
                    &visible_memory_ids,
                ),
                search_visibility_receipt_digest: receipt_digest(
                    b"luminous:mace:redb-visible:v1\0",
                    &payload_digest,
                    now,
                    &visible_memory_ids,
                ),
                persisted_at_ms: now,
                // Memory and visibility become atomic at the same database
                // commit. Millisecond precision may represent both events
                // with the same timestamp.
                searchable_at_ms: now,
                visible_memory_ids,
            };
            let stored = StoredLedgerEntryV1 {
                idempotency_digest: key.idempotency_digest.clone(),
                state: "written".into(),
                receipt: Some(StoredReceiptV1::from(&receipt)),
            };
            let bytes = serde_json::to_vec(&stored).map_err(backend_error)?;
            ledger.insert(ledger_key.as_str(), bytes.as_slice()).map_err(backend_error)?;
            (receipt, memory_rows)
        };
        {
            let mut table = transaction.open_table(MEMORY_RECORDS).map_err(backend_error)?;
            for row in &memory_rows {
                let stored = StoredMemoryRecordV1::from_durable(track, row);
                let bytes = serde_json::to_vec(&stored).map_err(backend_error)?;
                let memory_key = Self::memory_key(&row.user_id, &row.id);
                table.insert(memory_key.as_str(), bytes.as_slice()).map_err(backend_error)?;
            }
        }
        transaction.commit().map_err(backend_error)?;
        Ok(receipt)
    }

    fn reconcile_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<DurableAddReconciliationV1, DurableStoreErrorV1> {
        let Some(entry) = self.load_ledger(key)? else {
            return Ok(DurableAddReconciliationV1::NotCommitted);
        };
        if entry.idempotency_digest != key.idempotency_digest {
            return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
        }
        match entry.state.as_str() {
            "reserved" => Ok(DurableAddReconciliationV1::NotCommitted),
            "written" | "finalized" => {
                let receipt = entry.receipt.ok_or_else(|| {
                    DurableStoreErrorV1::Backend("written ledger entry has no receipt".into())
                })?;
                Ok(DurableAddReconciliationV1::Committed(receipt.try_into()?))
            }
            other => Err(DurableStoreErrorV1::Backend(format!(
                "unknown Add ledger state {other}"
            ))),
        }
    }

    fn read_visible_by_ids(
        &mut self,
        user_id: &str,
        memory_ids: &[String],
    ) -> Result<Vec<DurableMemoryRecordV1>, DurableStoreErrorV1> {
        let mut records = Vec::with_capacity(memory_ids.len());
        for id in memory_ids {
            if let Some(record) = self.read_memory_record(user_id, id)? {
                records.push(record);
            }
        }
        Ok(records)
    }

    fn finalize_add(
        &mut self,
        key: &DurableAddKeyV1,
        receipt: &AddVisibilityReceiptV1,
    ) -> Result<(), DurableStoreErrorV1> {
        let ledger_key = Self::ledger_key(key);
        let transaction = self.database.begin_write().map_err(backend_error)?;
        {
            let mut table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            let existing = table
                .get(ledger_key.as_str())
                .map_err(backend_error)?
                .ok_or_else(|| DurableStoreErrorV1::Backend(
                    "finalize without a durable reservation".into(),
                ))?;
            let mut entry: StoredLedgerEntryV1 =
                serde_json::from_slice(existing.value()).map_err(backend_error)?;
            if entry.idempotency_digest != key.idempotency_digest {
                return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
            }
            if entry.receipt.as_ref() != Some(&StoredReceiptV1::from(receipt)) {
                return Err(DurableStoreErrorV1::Backend(
                    "finalize receipt does not match committed Add".into(),
                ));
            }
            if entry.state != "written" && entry.state != "finalized" {
                return Err(DurableStoreErrorV1::Backend(
                    "cannot finalize without a committed write".into(),
                ));
            }
            entry.state = "finalized".into();
            let bytes = serde_json::to_vec(&entry).map_err(backend_error)?;
            table.insert(ledger_key.as_str(), bytes.as_slice()).map_err(backend_error)?;
        }
        transaction.commit().map_err(backend_error)?;
        Ok(())
    }

    fn release_uncommitted_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<(), DurableStoreErrorV1> {
        let ledger_key = Self::ledger_key(key);
        let transaction = self.database.begin_write().map_err(backend_error)?;
        {
            let mut table = transaction.open_table(ADD_LEDGER).map_err(backend_error)?;
            let Some(existing) = table
                .get(ledger_key.as_str())
                .map_err(backend_error)?
            else {
                return Err(DurableStoreErrorV1::Backend(
                    "release without reservation".into(),
                ));
            };
            let entry: StoredLedgerEntryV1 =
                serde_json::from_slice(existing.value()).map_err(backend_error)?;
            if entry.idempotency_digest != key.idempotency_digest
                || entry.state != "reserved"
                || entry.receipt.is_some()
            {
                return Err(DurableStoreErrorV1::Backend(
                    "refusing to release committed, written or rebound Add".into(),
                ));
            }
            table.remove(ledger_key.as_str()).map_err(backend_error)?;
        }
        transaction.commit().map_err(backend_error)?;
        Ok(())
    }

    fn search_with_user_partition(
        &mut self,
        track: Track,
        request: &SearchRequest,
    ) -> Result<Vec<DurableScopedSearchItemV1>, DurableStoreErrorV1> {
        let records = self.visible_records_for_user(&request.user_id, track)?;
        let query_tokens = content_tokens(&request.query);
        let n = records.len() as f64;
        let mut document_frequency = BTreeMap::<String, usize>::new();
        let mut tokenized = Vec::with_capacity(records.len());
        let mut total_length = 0usize;

        for record in &records {
            let tokens = content_tokens(&stored_to_content(record.content.clone()));
            let unique: BTreeSet<&str> = tokens.iter().map(String::as_str).collect();
            for token in unique {
                *document_frequency.entry(token.to_owned()).or_default() += 1;
            }
            total_length = total_length.saturating_add(tokens.len());
            tokenized.push(tokens);
        }

        let average_length = if records.is_empty() {
            0.0
        } else {
            total_length as f64 / n
        };
        let query_terms: BTreeSet<String> = query_tokens.into_iter().collect();
        let mut ranked = Vec::with_capacity(records.len());
        for (record, tokens) in records.into_iter().zip(tokenized) {
            let mut term_frequency = HashMap::<&str, usize>::new();
            for token in &tokens {
                *term_frequency.entry(token.as_str()).or_default() += 1;
            }
            let document_length = tokens.len() as f64;
            let mut score = 0.0_f64;
            for term in &query_terms {
                let frequency = *term_frequency.get(term.as_str()).unwrap_or(&0) as f64;
                if frequency == 0.0 {
                    continue;
                }
                let df = *document_frequency.get(term).unwrap_or(&0) as f64;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let k1 = 1.2;
                let b = 0.75;
                let norm = if average_length == 0.0 {
                    1.0
                } else {
                    k1 * (1.0 - b + b * document_length / average_length)
                };
                score += idf * (frequency * (k1 + 1.0)) / (frequency + norm);
            }
            ranked.push((score, record));
        }
        ranked.sort_by(|(left_score, left_record), (right_score, right_record)| {
            right_score.total_cmp(left_score)
                .then_with(|| left_record.id.cmp(&right_record.id))
        });
        Ok(ranked
            .into_iter()
            .take(request.top_k)
            .map(|(score, stored)| {
                let owner_user_id = stored.user_id.clone();
                let record = stored.into_durable();
                DurableScopedSearchItemV1 {
                    owner_user_id,
                    item: SearchItem {
                        id: record.id,
                        content: record.content,
                        score: Some(score),
                        created_at: None,
                    },
                }
            })
            .collect())
    }
}

/// This verifier checks local cryptographic hashes over the committed write
/// and visibility receipt. It is not a remote provider signature or an
/// independently witnessed storage attestation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RedbVisibilityVerifierV1;

impl AddVisibilityVerifierV1 for RedbVisibilityVerifierV1 {
    fn verify(&self, receipt: &AddVisibilityReceiptV1) -> Result<(), AmlReject> {
        let expected_write = receipt_digest(
            b"luminous:mace:redb-write:v1\0",
            &receipt.request_payload_digest,
            receipt.persisted_at_ms,
            &receipt.visible_memory_ids,
        );
        let expected_search = receipt_digest(
            b"luminous:mace:redb-visible:v1\0",
            &receipt.request_payload_digest,
            receipt.searchable_at_ms,
            &receipt.visible_memory_ids,
        );
        if receipt.write_receipt_digest != expected_write
            || receipt.search_visibility_receipt_digest != expected_search
            || receipt.searchable_at_ms < receipt.persisted_at_ms
        {
            return Err(AmlReject::AddVisibilityVerificationFailed);
        }
        Ok(())
    }
}

fn receipt_digest(domain: &[u8], payload_digest: &str, timestamp_ms: i64, ids: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((payload_digest.len() as u64).to_be_bytes());
    hasher.update(payload_digest.as_bytes());
    hasher.update(timestamp_ms.to_be_bytes());
    let encoded_ids = serde_json::to_vec(ids).unwrap_or_default();
    hasher.update((encoded_ids.len() as u64).to_be_bytes());
    hasher.update(encoded_ids);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_component(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn track_name(track: Track) -> &'static str {
    match track {
        Track::Textual => "textual",
        Track::Coding => "coding",
        Track::Multimodal => "multimodal",
    }
}

fn parse_track(value: &str) -> Result<Track, DurableStoreErrorV1> {
    match value {
        "textual" => Ok(Track::Textual),
        "coding" => Ok(Track::Coding),
        "multimodal" => Ok(Track::Multimodal),
        other => Err(DurableStoreErrorV1::Backend(format!(
            "unknown stored track {other}"
        ))),
    }
}

fn now_ms() -> Result<i64, DurableStoreErrorV1> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(backend_error)?;
    i64::try_from(duration.as_millis()).map_err(backend_error)
}

fn backend_error(error: impl std::fmt::Display) -> DurableStoreErrorV1 {
    DurableStoreErrorV1::Backend(error.to_string())
}

fn content_tokens(content: &MessageContent) -> Vec<String> {
    let mut text = String::new();
    match content {
        MessageContent::Text(value) => text.push_str(value),
        MessageContent::Parts(parts) => {
            for part in parts {
                if let ContentPart::Text(value) = part {
                    text.push(' ');
                    text.push_str(value);
                }
            }
        }
    }
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn content_to_stored(content: &MessageContent) -> StoredContentV1 {
    match content {
        MessageContent::Text(text) => StoredContentV1::Text(text.clone()),
        MessageContent::Parts(parts) => StoredContentV1::Parts(
            parts.iter().map(|part| match part {
                ContentPart::Text(text) => StoredPartV1::Text(text.clone()),
                ContentPart::ImageDataUri(uri) => StoredPartV1::ImageDataUri(uri.clone()),
            }).collect()
        ),
    }
}

fn stored_to_content(content: StoredContentV1) -> MessageContent {
    match content {
        StoredContentV1::Text(text) => MessageContent::Text(text),
        StoredContentV1::Parts(parts) => MessageContent::Parts(
            parts.into_iter().map(|part| match part {
                StoredPartV1::Text(text) => ContentPart::Text(text),
                StoredPartV1::ImageDataUri(uri) => ContentPart::ImageDataUri(uri),
            }).collect()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::super::{
        idempotency_payload_digest, DurableAmlServiceV1, DurableAmlServiceErrorV1,
        FrozenAdapterWordCounter, Message, Role,
    };

    static NEXT_FILE: AtomicU64 = AtomicU64::new(1);

    struct TestCounter;

    impl FrozenAdapterWordCounter for TestCounter {
        fn adapter_version(&self) -> &str {
            "redb-test-counter-v1"
        }
        fn count_words(&self, messages: &[Message]) -> usize {
            messages.iter().map(|message| match &message.content {
                MessageContent::Text(text) => text.split_whitespace().count(),
                MessageContent::Parts(parts) => parts.iter().map(|part| match part {
                    ContentPart::Text(text) => text.split_whitespace().count(),
                    ContentPart::ImageDataUri(_) => 0,
                }).sum(),
            }).sum()
        }
    }

    fn temp_db_path() -> PathBuf {
        let index = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "mace-redb-test-{}-{index}.redb",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    fn request(request_id: &str, user_id: &str, content: &str) -> AddRequest {
        AddRequest {
            request_id: request_id.into(),
            messages: vec![Message {
                role: Role::User,
                timestamp_ms: Some(1_700_000_000_000),
                content: MessageContent::Text(content.into()),
            }],
            user_id: user_id.into(),
            session_id: format!("session:{request_id}"),
        }
    }

    fn service(
        store: RedbDurableAmlStoreV1,
    ) -> DurableAmlServiceV1<RedbDurableAmlStoreV1, RedbVisibilityVerifierV1, TestCounter> {
        DurableAmlServiceV1::new(store, RedbVisibilityVerifierV1, TestCounter)
    }

    fn durable_key(request: &AddRequest) -> DurableAddKeyV1 {
        DurableAddKeyV1 {
            track: Track::Textual,
            user_id: request.user_id.clone(),
            request_id: request.request_id.clone(),
            idempotency_digest: idempotency_payload_digest(
                Track::Textual,
                request,
                Some(TestCounter.adapter_version()),
            ),
            word_counter_version: Some(TestCounter.adapter_version().to_owned()),
        }
    }

    #[test]
    fn add_and_idempotent_replay_survive_database_reopen() {
        let path = temp_db_path();
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            let write = request("redb-1", "user-a", "remember the Mars mission date");
            assert!(service.add(Track::Textual, &write).unwrap().success);
        }
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            let write = request("redb-1", "user-a", "remember the Mars mission date");
            assert!(service.add(Track::Textual, &write).unwrap().success);
            let response = service.search(
                Track::Textual,
                &SearchRequest {
                    query: MessageContent::Text("Mars mission".into()),
                    options: None,
                    user_id: "user-a".into(),
                    top_k: 100,
                },
            ).unwrap();
            assert_eq!(response.data.len(), 1);
            assert_eq!(
                response.data[0].content,
                MessageContent::Text("remember the Mars mission date".into())
            );
        }
        cleanup(&path);
    }

    #[test]
    fn restart_reclaims_only_unwritten_reservations() {
        let path = temp_db_path();
        let write = request("redb-pending", "user-a", "pending message");
        {
            let mut store = RedbDurableAmlStoreV1::open(&path).unwrap();
            let key = durable_key(&write);
            assert_eq!(store.reserve_add(&key).unwrap(), DurableAddReservationV1::New);
        }
        {
            let mut store = RedbDurableAmlStoreV1::open(&path).unwrap();
            let key = durable_key(&write);
            assert_eq!(store.reserve_add(&key).unwrap(), DurableAddReservationV1::New);
        }
        cleanup(&path);
    }

    #[test]
    fn written_before_finalize_is_reconciled_after_restart_without_duplicate() {
        let path = temp_db_path();
        let write = request("redb-written", "user-a", "write reached durable commit");
        {
            let mut store = RedbDurableAmlStoreV1::open(&path).unwrap();
            let key = durable_key(&write);
            assert_eq!(store.reserve_add(&key).unwrap(), DurableAddReservationV1::New);
            store.persist_and_index(&key, Track::Textual, &write).unwrap();
        }
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            assert!(service.add(Track::Textual, &write).unwrap().success);
            let response = service.search(
                Track::Textual,
                &SearchRequest {
                    query: MessageContent::Text("durable commit".into()),
                    options: None,
                    user_id: "user-a".into(),
                    top_k: 100,
                },
            ).unwrap();
            assert_eq!(response.data.len(), 1);
        }
        cleanup(&path);
    }

    #[test]
    fn bm25_search_is_tenant_partitioned_and_ranks_relevant_text_first() {
        let path = temp_db_path();
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            service.add(
                Track::Textual,
                &request("bm25-mars", "user-a", "Mars mission launch window and orbital transfer"),
            ).unwrap();
            service.add(
                Track::Textual,
                &request("bm25-garden", "user-a", "tomato garden irrigation and soil nutrients"),
            ).unwrap();
            service.add(
                Track::Textual,
                &request("bm25-other-user", "user-b", "Mars mission confidential notes"),
            ).unwrap();
            let response = service.search(
                Track::Textual,
                &SearchRequest {
                    query: MessageContent::Text("Mars mission orbital".into()),
                    options: None,
                    user_id: "user-a".into(),
                    top_k: 100,
                },
            ).unwrap();
            assert_eq!(response.data.len(), 2);
            assert_eq!(
                response.data[0].content,
                MessageContent::Text("Mars mission launch window and orbital transfer".into())
            );
        }
        cleanup(&path);
    }

    #[test]
    fn request_id_rebinding_fails_across_store_reopen() {
        let path = temp_db_path();
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            service.add(
                Track::Textual,
                &request("redb-conflict", "user-a", "original")
            ).unwrap();
        }
        {
            let mut service = service(RedbDurableAmlStoreV1::open(&path).unwrap());
            let changed = request("redb-conflict", "user-a", "different content");
            assert_eq!(
                service.add(Track::Textual, &changed),
                Err(DurableAmlServiceErrorV1::Contract(
                    AmlReject::RequestIdPayloadConflict
                ))
            );
        }
        cleanup(&path);
    }
}
