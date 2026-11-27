// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Durable orchestration boundary for the AML Add/Search contract.
//!
//! This module is storage- and HTTP-framework-neutral. A hosted adapter must
//! implement DurableAmlStoreV1 with transactional persistence, restart-safe
//! idempotency and a user-partitioned search index. A successful write call
//! alone is not enough for HTTP 200: every returned memory ID is read back
//! under the exact request user before the durable idempotency row is finalized.
//!
//! The store owns transaction isolation and crash recovery. Its reservation,
//! persistence, reconciliation and finalization operations must be durable
//! and idempotent across process restarts.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    validate_add_request, validate_add_response, validate_search_request_for_track,
    validate_search_response_for_track, validate_textual_add_response, validate_textual_chunk,
    AddRequest, AddResponse, AddVisibilityReceiptV1, AddVisibilityVerifierV1, AmlReject,
    FrozenAdapterWordCounter, MessageContent, SearchRequest, SearchResponse, Track,
    MAX_SEARCH_RESPONSE_IMAGE_BYTES,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DurableAddKeyV1 {
    pub track: Track,
    pub user_id: String,
    pub request_id: String,
    /// Binds the canonical request and, for Textual, the frozen Adapter counter.
    pub idempotency_digest: String,
    pub word_counter_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DurableAddReservationV1 {
    /// The caller owns a newly reserved durable idempotency key.
    New,
    /// A prior call may have written or crashed before finalization. Reconcile;
    /// do not blindly issue another write.
    InFlight,
    /// A prior call completed. Visibility is re-read before replaying HTTP 200.
    Committed(AddVisibilityReceiptV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DurableAddReconciliationV1 {
    /// The store has positively established that the write did not commit.
    NotCommitted,
    /// The store found a durable receipt. Verify it and read the IDs back.
    Committed(AddVisibilityReceiptV1),
    /// The commit state cannot be established. Fail closed.
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DurableMemoryRecordV1 {
    pub id: String,
    pub user_id: String,
    pub origin_request_id: String,
    pub source_request_payload_digest: String,
    pub content: MessageContent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DurableStoreErrorV1 {
    RequestIdPayloadConflict,
    Backend(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DurableAmlServiceErrorV1 {
    Contract(AmlReject),
    Store(DurableStoreErrorV1),
    AddStillInFlight,
    CommitOutcomeUnknown,
    SearchVisibilityMismatch,
    SearchScopeViolation,
}

/// A result returned by the internal store with its tenant identity intact.
/// The public AML response strips this field only after the coordinator has
/// verified that every result belongs to the request user.
#[derive(Clone, Debug, PartialEq)]
pub struct DurableScopedSearchItemV1 {
    pub owner_user_id: String,
    pub item: super::SearchItem,
}

/// Implement this trait with a durable database and a user-partitioned index.
///
/// Required backend invariants:
/// - reserve_add atomically reserves (track, user_id, request_id) and rejects
///   reuse with a different idempotency_digest.
/// - persist_and_index persists logical memories and makes their IDs searchable
///   before returning. Repeating the same key/digest returns the same receipt.
/// - reconcile_add is read-only and may return Unknown; it never starts a write.
/// - read_visible_by_ids applies the exact user_id partition in the storage/index
///   query itself, before scoring or content retrieval.
/// - finalize_add is durable and idempotent for the same verified receipt.
/// - search_with_user_partition applies isolation before ranking and preserves
///   the resulting ranked order.
/// - Committed state must survive process restart; an in-memory cache alone
///   cannot establish durable success.
pub trait DurableAmlStoreV1 {
    fn reserve_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<DurableAddReservationV1, DurableStoreErrorV1>;

    fn persist_and_index(
        &mut self,
        key: &DurableAddKeyV1,
        track: Track,
        request: &AddRequest,
    ) -> Result<AddVisibilityReceiptV1, DurableStoreErrorV1>;

    fn reconcile_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<DurableAddReconciliationV1, DurableStoreErrorV1>;

    fn read_visible_by_ids(
        &mut self,
        user_id: &str,
        memory_ids: &[String],
    ) -> Result<Vec<DurableMemoryRecordV1>, DurableStoreErrorV1>;

    /// Atomically release an uncommitted reservation after reconcile_add has
    /// returned NotCommitted. Must reject if a write/receipt appeared meanwhile.
    fn release_uncommitted_add(
        &mut self,
        key: &DurableAddKeyV1,
    ) -> Result<(), DurableStoreErrorV1>;

    fn finalize_add(
        &mut self,
        key: &DurableAddKeyV1,
        receipt: &AddVisibilityReceiptV1,
    ) -> Result<(), DurableStoreErrorV1>;

    fn search_with_user_partition(
        &mut self,
        track: Track,
        request: &SearchRequest,
    ) -> Result<Vec<DurableScopedSearchItemV1>, DurableStoreErrorV1>;
}

pub struct DurableAmlServiceV1<S, V, C> {
    store: S,
    verifier: V,
    word_counter: C,
}

impl<S, V, C> DurableAmlServiceV1<S, V, C>
where
    S: DurableAmlStoreV1,
    V: AddVisibilityVerifierV1,
    C: FrozenAdapterWordCounter,
{
    pub fn new(store: S, verifier: V, word_counter: C) -> Self {
        Self {
            store,
            verifier,
            word_counter,
        }
    }

    /// Execute a synchronous Add. An HTTP adapter may send 200 only after this
    /// method returns Ok.
    pub fn add(
        &mut self,
        track: Track,
        request: &AddRequest,
    ) -> Result<AddResponse, DurableAmlServiceErrorV1> {
        let word_counter_version = if track == Track::Textual {
            validate_textual_chunk(request, &self.word_counter)
                .map_err(DurableAmlServiceErrorV1::Contract)?;
            Some(self.word_counter.adapter_version().to_owned())
        } else {
            validate_add_request(track, request)
                .map_err(DurableAmlServiceErrorV1::Contract)?;
            None
        };

        let idempotency_digest = super::idempotency_payload_digest(
            track,
            request,
            word_counter_version.as_deref(),
        );
        let key = DurableAddKeyV1 {
            track,
            user_id: request.user_id.clone(),
            request_id: request.request_id.clone(),
            idempotency_digest,
            word_counter_version,
        };

        let receipt = match self.store.reserve_add(&key).map_err(map_store_error)? {
            DurableAddReservationV1::New => {
                match self.store.persist_and_index(&key, track, request) {
                    Ok(receipt) => receipt,
                    Err(write_error) => {
                        // A database/transport error does not prove rollback.
                        // Reconcile before returning or permitting another write.
                        match self.store.reconcile_add(&key).map_err(map_store_error)? {
                            DurableAddReconciliationV1::Committed(receipt) => receipt,
                            DurableAddReconciliationV1::NotCommitted => {
                                self.store.release_uncommitted_add(&key).map_err(map_store_error)?;
                                return Err(map_store_error(write_error));
                            }
                            DurableAddReconciliationV1::Unknown => {
                                return Err(DurableAmlServiceErrorV1::CommitOutcomeUnknown);
                            }
                        }
                    }
                }
            }
            DurableAddReservationV1::InFlight => {
                match self.store.reconcile_add(&key).map_err(map_store_error)? {
                    DurableAddReconciliationV1::Committed(receipt) => receipt,
                    DurableAddReconciliationV1::NotCommitted => {
                        self.store.release_uncommitted_add(&key).map_err(map_store_error)?;
                        return Err(DurableAmlServiceErrorV1::AddStillInFlight);
                    }
                    DurableAddReconciliationV1::Unknown => {
                        return Err(DurableAmlServiceErrorV1::CommitOutcomeUnknown);
                    }
                }
            }
            DurableAddReservationV1::Committed(receipt) => receipt,
        };

        self.verify_receipt_and_visibility(track, request, &key, &receipt)?;
        Ok(AddResponse {
            success: true,
            request_id: request.request_id.clone(),
            user_id: request.user_id.clone(),
            session_id: request.session_id.clone(),
        })
    }

    /// Return retrieved memory records, never a generated final answer.
    pub fn search(
        &mut self,
        track: Track,
        request: &SearchRequest,
    ) -> Result<SearchResponse, DurableAmlServiceErrorV1> {
        validate_search_request_for_track(track, request)
            .map_err(DurableAmlServiceErrorV1::Contract)?;
        let scoped = self
            .store
            .search_with_user_partition(track, request)
            .map_err(map_store_error)?;
        let mut data = Vec::with_capacity(scoped.len());
        for result in scoped {
            if result.owner_user_id != request.user_id {
                return Err(DurableAmlServiceErrorV1::SearchScopeViolation);
            }
            data.push(result.item);
        }
        let response = SearchResponse { data };
        validate_search_response_for_track(track, request, &response)
            .map_err(DurableAmlServiceErrorV1::Contract)?;
        Ok(response)
    }

    pub fn into_store(self) -> S {
        self.store
    }

    fn verify_receipt_and_visibility(
        &mut self,
        track: Track,
        request: &AddRequest,
        key: &DurableAddKeyV1,
        receipt: &AddVisibilityReceiptV1,
    ) -> Result<(), DurableAmlServiceErrorV1> {
        let response = AddResponse {
            success: true,
            request_id: request.request_id.clone(),
            user_id: request.user_id.clone(),
            session_id: request.session_id.clone(),
        };
        let verified = if track == Track::Textual {
            validate_textual_add_response(
                request,
                &response,
                receipt,
                &self.verifier,
                &self.word_counter,
            )
        } else {
            validate_add_response(track, request, &response, receipt, &self.verifier)
        }
        .map_err(DurableAmlServiceErrorV1::Contract)?;

        let records = self
            .store
            .read_visible_by_ids(&request.user_id, verified.visible_memory_ids())
            .map_err(map_store_error)?;
        verify_visible_records(
            track,
            &request.user_id,
            &request.request_id,
            verified.request_payload_digest(),
            verified.visible_memory_ids(),
            &records,
        )?;

        // A failure here must not yield HTTP 200. Finalization must be durable
        // and idempotent so an identical retry can reconcile and finish.
        self.store
            .finalize_add(key, verified.receipt())
            .map_err(map_store_error)?;
        Ok(())
    }
}

fn map_store_error(error: DurableStoreErrorV1) -> DurableAmlServiceErrorV1 {
    match error {
        DurableStoreErrorV1::RequestIdPayloadConflict => {
            DurableAmlServiceErrorV1::Contract(AmlReject::RequestIdPayloadConflict)
        }
        other => DurableAmlServiceErrorV1::Store(other),
    }
}

fn verify_visible_records(
    track: Track,
    expected_user_id: &str,
    expected_request_id: &str,
    expected_payload_digest: &str,
    expected_ids: &[String],
    records: &[DurableMemoryRecordV1],
) -> Result<(), DurableAmlServiceErrorV1> {
    if expected_ids.is_empty() || records.len() != expected_ids.len() {
        return Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch);
    }
    let expected: BTreeSet<&str> = expected_ids.iter().map(String::as_str).collect();
    if expected.len() != expected_ids.len() {
        return Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch);
    }

    let mut found = BTreeMap::<&str, usize>::new();
    let mut total_image_bytes = 0usize;
    for record in records {
        if record.user_id != expected_user_id
            || record.origin_request_id != expected_request_id
            || record.source_request_payload_digest != expected_payload_digest
            || !expected.contains(record.id.as_str())
        {
            return Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch);
        }
        if found.insert(record.id.as_str(), 1).is_some() {
            return Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch);
        }
        let image_bytes = super::validate_content(track, &record.content)
            .map_err(DurableAmlServiceErrorV1::Contract)?;
        total_image_bytes = total_image_bytes.saturating_add(image_bytes);
    }
    if found.len() != expected.len()
        || (track == Track::Multimodal && total_image_bytes > MAX_SEARCH_RESPONSE_IMAGE_BYTES)
    {
        return Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use super::super::{
        add_payload_digest, sha256_hex, AddVisibilityVerifierV1, FrozenAdapterWordCounter,
        Message, MessageContent, Role, SearchItem,
    };

    struct TestCounter;

    impl FrozenAdapterWordCounter for TestCounter {
        fn adapter_version(&self) -> &str {
            "aml-frozen-test-counter-v1"
        }

        fn count_words(&self, messages: &[Message]) -> usize {
            messages
                .iter()
                .map(|message| match &message.content {
                    MessageContent::Text(text) => text.split_whitespace().count(),
                    MessageContent::Parts(_) => 0,
                })
                .sum()
        }
    }

    struct TestVerifier;

    impl AddVisibilityVerifierV1 for TestVerifier {
        fn verify(&self, receipt: &AddVisibilityReceiptV1) -> Result<(), AmlReject> {
            let expected_write = sha256_hex(
                format!(
                    "durable-fixture-write:{}:{}",
                    receipt.request_payload_digest, receipt.persisted_at_ms
                )
                .as_bytes(),
            );
            let expected_search = sha256_hex(
                format!(
                    "durable-fixture-search:{}:{}",
                    receipt.request_payload_digest, receipt.searchable_at_ms
                )
                .as_bytes(),
            );
            if receipt.write_receipt_digest != expected_write
                || receipt.search_visibility_receipt_digest != expected_search
            {
                return Err(AmlReject::AddVisibilityVerificationFailed);
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Entry {
        digest: String,
        receipt: Option<AddVisibilityReceiptV1>,
        finalized: bool,
    }

    #[derive(Default)]
    struct FakeDurableStore {
        entries: BTreeMap<(Track, String, String), Entry>,
        visible: Vec<DurableMemoryRecordV1>,
        lose_write_reply_after_commit: bool,
        unknown_reconcile: bool,
        hide_visibility: bool,
        inject_cross_user_search_result: bool,
        fail_write_before_commit: bool,
    }

    impl FakeDurableStore {
        fn entry_key(key: &DurableAddKeyV1) -> (Track, String, String) {
            (key.track, key.user_id.clone(), key.request_id.clone())
        }

        fn visibility_receipt(
            key: &DurableAddKeyV1,
            request: &AddRequest,
        ) -> (AddVisibilityReceiptV1, Vec<DurableMemoryRecordV1>) {
            let request_digest = add_payload_digest(key.track, request);
            let persisted_at_ms = 100;
            let searchable_at_ms = 101;
            let records: Vec<DurableMemoryRecordV1> = request
                .messages
                .iter()
                .enumerate()
                .map(|(index, message)| DurableMemoryRecordV1 {
                    id: format!("{}:{}:{index}", request.user_id, request.request_id),
                    user_id: request.user_id.clone(),
                    origin_request_id: request.request_id.clone(),
                    source_request_payload_digest: add_payload_digest(key.track, request),
                    content: message.content.clone(),
                })
                .collect();
            let visible_memory_ids: Vec<String> =
                records.iter().map(|record| record.id.clone()).collect();
            let receipt = AddVisibilityReceiptV1 {
                track: key.track,
                request_id: request.request_id.clone(),
                user_id: request.user_id.clone(),
                session_id: request.session_id.clone(),
                request_payload_digest: request_digest.clone(),
                write_receipt_digest: sha256_hex(
                    format!("durable-fixture-write:{request_digest}:{persisted_at_ms}").as_bytes(),
                ),
                search_visibility_receipt_digest: sha256_hex(
                    format!("durable-fixture-search:{request_digest}:{searchable_at_ms}").as_bytes(),
                ),
                persisted_at_ms,
                searchable_at_ms,
                visible_memory_ids,
            };
            (receipt, records)
        }
    }

    impl DurableAmlStoreV1 for FakeDurableStore {
        fn reserve_add(
            &mut self,
            key: &DurableAddKeyV1,
        ) -> Result<DurableAddReservationV1, DurableStoreErrorV1> {
            let entry_key = Self::entry_key(key);
            match self.entries.get(&entry_key) {
                Some(entry) if entry.digest != key.idempotency_digest => {
                    Err(DurableStoreErrorV1::RequestIdPayloadConflict)
                }
                Some(entry) if entry.finalized => Ok(DurableAddReservationV1::Committed(
                    entry.receipt.clone().ok_or_else(|| {
                        DurableStoreErrorV1::Backend("finalized without receipt".into())
                    })?,
                )),
                Some(_) => Ok(DurableAddReservationV1::InFlight),
                None => {
                    self.entries.insert(
                        entry_key,
                        Entry {
                            digest: key.idempotency_digest.clone(),
                            receipt: None,
                            finalized: false,
                        },
                    );
                    Ok(DurableAddReservationV1::New)
                }
            }
        }

        fn persist_and_index(
            &mut self,
            key: &DurableAddKeyV1,
            _track: Track,
            request: &AddRequest,
        ) -> Result<AddVisibilityReceiptV1, DurableStoreErrorV1> {
            let entry_key = Self::entry_key(key);
            let entry = self.entries.get_mut(&entry_key).ok_or_else(|| {
                DurableStoreErrorV1::Backend("write without reservation".into())
            })?;
            if entry.digest != key.idempotency_digest {
                return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
            }
            if self.fail_write_before_commit {
                self.fail_write_before_commit = false;
                return Err(DurableStoreErrorV1::Backend(
                    "simulated failure before commit".into(),
                ));
            }
            let (receipt, records) = Self::visibility_receipt(key, request);
            entry.receipt = Some(receipt.clone());
            self.visible.extend(records);
            if self.lose_write_reply_after_commit {
                self.lose_write_reply_after_commit = false;
                return Err(DurableStoreErrorV1::Backend(
                    "simulated lost reply after durable commit".into(),
                ));
            }
            Ok(receipt)
        }

        fn reconcile_add(
            &mut self,
            key: &DurableAddKeyV1,
        ) -> Result<DurableAddReconciliationV1, DurableStoreErrorV1> {
            if self.unknown_reconcile {
                return Ok(DurableAddReconciliationV1::Unknown);
            }
            let Some(entry) = self.entries.get(&Self::entry_key(key)) else {
                return Ok(DurableAddReconciliationV1::NotCommitted);
            };
            if entry.digest != key.idempotency_digest {
                return Err(DurableStoreErrorV1::RequestIdPayloadConflict);
            }
            Ok(match &entry.receipt {
                Some(receipt) => DurableAddReconciliationV1::Committed(receipt.clone()),
                None => DurableAddReconciliationV1::NotCommitted,
            })
        }

        fn read_visible_by_ids(
            &mut self,
            user_id: &str,
            memory_ids: &[String],
        ) -> Result<Vec<DurableMemoryRecordV1>, DurableStoreErrorV1> {
            if self.hide_visibility {
                return Ok(Vec::new());
            }
            let wanted: BTreeSet<&str> = memory_ids.iter().map(String::as_str).collect();
            Ok(self
                .visible
                .iter()
                .filter(|record| record.user_id == user_id && wanted.contains(record.id.as_str()))
                .cloned()
                .collect())
        }

        fn release_uncommitted_add(
            &mut self,
            key: &DurableAddKeyV1,
        ) -> Result<(), DurableStoreErrorV1> {
            let entry_key = Self::entry_key(key);
            let Some(entry) = self.entries.get(&entry_key) else {
                return Err(DurableStoreErrorV1::Backend(
                    "release without reservation".into(),
                ));
            };
            if entry.digest != key.idempotency_digest || entry.receipt.is_some() {
                return Err(DurableStoreErrorV1::Backend(
                    "refusing to release a committed or rebound reservation".into(),
                ));
            }
            self.entries.remove(&entry_key);
            Ok(())
        }

        fn finalize_add(
            &mut self,
            key: &DurableAddKeyV1,
            receipt: &AddVisibilityReceiptV1,
        ) -> Result<(), DurableStoreErrorV1> {
            let entry = self.entries.get_mut(&Self::entry_key(key)).ok_or_else(|| {
                DurableStoreErrorV1::Backend("finalize without reservation".into())
            })?;
            if entry.digest != key.idempotency_digest || entry.receipt.as_ref() != Some(receipt) {
                return Err(DurableStoreErrorV1::Backend(
                    "finalization binding mismatch".into(),
                ));
            }
            entry.finalized = true;
            Ok(())
        }

        fn search_with_user_partition(
            &mut self,
            _track: Track,
            request: &SearchRequest,
        ) -> Result<Vec<DurableScopedSearchItemV1>, DurableStoreErrorV1> {
            let mut results: Vec<DurableScopedSearchItemV1> = self
                .visible
                .iter()
                .filter(|record| record.user_id == request.user_id)
                .take(request.top_k)
                .map(|record| DurableScopedSearchItemV1 {
                    owner_user_id: record.user_id.clone(),
                    item: SearchItem {
                        id: record.id.clone(),
                        content: record.content.clone(),
                        score: Some(1.0),
                        created_at: None,
                    },
                })
                .collect();
            if self.inject_cross_user_search_result {
                if let Some(record) = self.visible.iter().find(|record| record.user_id != request.user_id) {
                    results.push(DurableScopedSearchItemV1 {
                        owner_user_id: record.user_id.clone(),
                        item: SearchItem {
                            id: record.id.clone(),
                            content: record.content.clone(),
                            score: Some(1.0),
                            created_at: None,
                        },
                    });
                }
            }
            Ok(results)
        }
    }

    fn add_request(request_id: &str, user_id: &str, content: &str) -> AddRequest {
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
        store: FakeDurableStore,
    ) -> DurableAmlServiceV1<FakeDurableStore, TestVerifier, TestCounter> {
        DurableAmlServiceV1::new(store, TestVerifier, TestCounter)
    }

    #[test]
    fn add_waits_for_exact_user_scoped_visibility_before_success() {
        let mut service = service(FakeDurableStore::default());
        let request = add_request("durable-1", "user-a", "remember alpha");
        let response = service.add(Track::Textual, &request).unwrap();
        assert!(response.success);
        assert_eq!(response.request_id, request.request_id);
        assert_eq!(response.user_id, request.user_id);
        assert_eq!(response.session_id, request.session_id);
        let store = service.into_store();
        assert_eq!(store.visible.len(), 1);
        assert!(store.entries.values().all(|entry| entry.finalized));
    }

    #[test]
    fn exact_retry_replays_receipt_without_duplicate_write() {
        let mut service = service(FakeDurableStore::default());
        let request = add_request("durable-2", "user-a", "remember beta");
        let first = service.add(Track::Textual, &request).unwrap();
        let before = service.store.visible.len();
        let second = service.add(Track::Textual, &request).unwrap();
        let after = service.store.visible.len();
        assert_eq!(first, second);
        assert_eq!(before, after);
    }

    #[test]
    fn request_id_payload_rebinding_is_a_hard_conflict() {
        let mut service = service(FakeDurableStore::default());
        let first = add_request("durable-3", "user-a", "original");
        let changed = add_request("durable-3", "user-a", "changed");
        service.add(Track::Textual, &first).unwrap();
        assert_eq!(
            service.add(Track::Textual, &changed),
            Err(DurableAmlServiceErrorV1::Contract(AmlReject::RequestIdPayloadConflict))
        );
    }

    #[test]
    fn lost_write_reply_reconciles_before_acknowledging() {
        let mut store = FakeDurableStore::default();
        store.lose_write_reply_after_commit = true;
        let mut service = service(store);
        let request = add_request("durable-4", "user-a", "committed before reply was lost");
        let response = service.add(Track::Textual, &request).unwrap();
        assert!(response.success);
        assert_eq!(service.store.visible.len(), 1);
    }

    #[test]
    fn reconciled_not_committed_write_can_be_retried_without_stuck_reservation() {
        let mut store = FakeDurableStore::default();
        store.fail_write_before_commit = true;
        let mut service = service(store);
        let request = add_request("durable-retry", "user-a", "retry after proven rollback");
        assert!(matches!(
            service.add(Track::Textual, &request),
            Err(DurableAmlServiceErrorV1::Store(DurableStoreErrorV1::Backend(_)))
        ));
        let response = service.add(Track::Textual, &request).unwrap();
        assert!(response.success);
        assert_eq!(service.store.visible.len(), 1);
        assert!(service.store.entries.values().all(|entry| entry.finalized));
    }

    #[test]
    fn unknown_commit_outcome_never_returns_false_success() {
        let mut store = FakeDurableStore::default();
        store.lose_write_reply_after_commit = true;
        store.unknown_reconcile = true;
        let mut service = service(store);
        let request = add_request("durable-5", "user-a", "ambiguous outcome");
        assert_eq!(
            service.add(Track::Textual, &request),
            Err(DurableAmlServiceErrorV1::CommitOutcomeUnknown)
        );
    }

    #[test]
    fn absent_search_visibility_blocks_success_acknowledgement() {
        let mut store = FakeDurableStore::default();
        store.hide_visibility = true;
        let mut service = service(store);
        let request = add_request("durable-6", "user-a", "not visible yet");
        assert_eq!(
            service.add(Track::Textual, &request),
            Err(DurableAmlServiceErrorV1::SearchVisibilityMismatch)
        );
    }

    #[test]
    fn search_rejects_cross_user_candidate_before_public_serialization() {
        let mut store = FakeDurableStore::default();
        let mut service = service(store);
        service.add(Track::Textual, &add_request("user-a-2", "user-a", "alpha")).unwrap();
        service.add(Track::Textual, &add_request("user-b-2", "user-b", "beta")).unwrap();
        service.store.inject_cross_user_search_result = true;
        let result = service.search(
            Track::Textual,
            &SearchRequest {
                query: MessageContent::Text("query".into()),
                options: None,
                user_id: "user-a".into(),
                top_k: 100,
            },
        );
        assert_eq!(result, Err(DurableAmlServiceErrorV1::SearchScopeViolation));
    }

    #[test]
    fn search_is_partitioned_by_user_and_returns_memories_not_answers() {
        let mut service = service(FakeDurableStore::default());
        service.add(Track::Textual, &add_request("user-a-1", "user-a", "alpha")).unwrap();
        service.add(Track::Textual, &add_request("user-b-1", "user-b", "beta")).unwrap();
        let response = service
            .search(
                Track::Textual,
                &SearchRequest {
                    query: MessageContent::Text("query".into()),
                    options: None,
                    user_id: "user-a".into(),
                    top_k: 100,
                },
            )
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].content, MessageContent::Text("alpha".into()));
    }
}
