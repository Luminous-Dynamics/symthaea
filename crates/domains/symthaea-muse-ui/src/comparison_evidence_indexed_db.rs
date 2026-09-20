// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Browser-local IndexedDB persistence for artifact-bound comparison evidence.
//!
//! This module is intentionally `wasm32`-only and contains no network path.
//! Local persistence is not consent to upload, aggregate, or enroll a listener
//! in research. Every value read from IndexedDB crosses the pure validation
//! boundary in `comparison_evidence_store` before it is returned to callers.
//!
//! Two invariants are deliberately stronger than a typical browser cache:
//!
//! 1. writes are create-only (`IDBObjectStore::add`, never `put`), and
//! 2. reads validate both the stored envelope and the key/value binding.
//!
//! A valid envelope stored under the wrong IndexedDB key is corruption, not a
//! second alias for the same evidence.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use futures::channel::oneshot;
use js_sys::Array;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::JsValue;
use wasm_bindgen::JsCast;
use web_sys::{
    DomException, IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransaction,
    IdbTransactionMode,
};

use crate::comparison_evidence_envelope::{
    ArtifactBoundBlindComparisonEnvelopeV1, ValidatedArtifactBoundBlindComparisonEnvelopeV1,
};
use crate::comparison_evidence_store::{
    ComparisonEvidenceCreateResolution, ComparisonEvidenceStoreContractError,
    prepare_comparison_evidence_write, resolve_create_once_comparison_evidence,
    validate_stored_comparison_evidence_json,
};

pub const COMPARISON_EVIDENCE_DATABASE_NAME: &str = "melothaea-private-evidence";
pub const COMPARISON_EVIDENCE_DATABASE_VERSION: u32 = 1;
pub const COMPARISON_EVIDENCE_OBJECT_STORE: &str = "blind-comparisons-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedDbComparisonEvidenceCreateOutcome {
    Created,
    ExactReplay,
}

#[derive(Debug, PartialEq)]
pub enum IndexedDbComparisonEvidenceError {
    StorageUnavailable,
    OpenBlocked,
    SchemaUpgrade(String),
    RequestFailed {
        name: String,
        message: String,
    },
    TransactionAborted {
        name: Option<String>,
        message: Option<String>,
    },
    TransactionChannelClosed,
    RequestChannelClosed,
    QuotaExceeded,
    ConstraintWithoutStoredRecord,
    UnexpectedStoredKey,
    UnexpectedStoredValue,
    KeyValueCountMismatch {
        keys: u32,
        values: u32,
    },
    Js(String),
    Contract(ComparisonEvidenceStoreContractError),
}

impl fmt::Display for IndexedDbComparisonEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageUnavailable => {
                write!(f, "IndexedDB is unavailable in this browser context")
            }
            Self::OpenBlocked => write!(
                f,
                "IndexedDB open/upgrade is blocked by another live database connection"
            ),
            Self::SchemaUpgrade(message) => {
                write!(f, "comparison evidence database schema upgrade failed: {message}")
            }
            Self::RequestFailed { name, message } => {
                write!(f, "IndexedDB request failed ({name}): {message}")
            }
            Self::TransactionAborted { name, message } => write!(
                f,
                "IndexedDB transaction aborted{}{}",
                name.as_ref()
                    .map(|value| format!(" ({value})"))
                    .unwrap_or_default(),
                message
                    .as_ref()
                    .map(|value| format!(": {value}"))
                    .unwrap_or_default()
            ),
            Self::TransactionChannelClosed => {
                write!(f, "IndexedDB transaction completion channel closed unexpectedly")
            }
            Self::RequestChannelClosed => {
                write!(f, "IndexedDB request completion channel closed unexpectedly")
            }
            Self::QuotaExceeded => write!(f, "browser storage quota was exceeded"),
            Self::ConstraintWithoutStoredRecord => write!(
                f,
                "IndexedDB reported a uniqueness conflict but no stored record was readable afterward"
            ),
            Self::UnexpectedStoredKey => write!(
                f,
                "comparison evidence object store contained a non-string primary key"
            ),
            Self::UnexpectedStoredValue => write!(
                f,
                "comparison evidence object store contained a non-string value"
            ),
            Self::KeyValueCountMismatch { keys, values } => write!(
                f,
                "comparison evidence object store returned {keys} keys for {values} values"
            ),
            Self::Js(message) => write!(f, "IndexedDB JavaScript binding error: {message}"),
            Self::Contract(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for IndexedDbComparisonEvidenceError {}

impl From<ComparisonEvidenceStoreContractError> for IndexedDbComparisonEvidenceError {
    fn from(value: ComparisonEvidenceStoreContractError) -> Self {
        Self::Contract(value)
    }
}

pub struct IndexedDbComparisonEvidenceStore {
    database: IdbDatabase,
    // This callback must stay alive with the database connection. Closing on
    // `versionchange` prevents this tab from blocking later schema upgrades.
    _on_version_change: Closure<dyn FnMut()>,
}

impl IndexedDbComparisonEvidenceStore {
    pub async fn open() -> Result<Self, IndexedDbComparisonEvidenceError> {
        let window =
            web_sys::window().ok_or(IndexedDbComparisonEvidenceError::StorageUnavailable)?;
        let factory = window
            .indexed_db()
            .map_err(js_binding_error)?
            .ok_or(IndexedDbComparisonEvidenceError::StorageUnavailable)?;
        let open = factory
            .open_with_u32(
                COMPARISON_EVIDENCE_DATABASE_NAME,
                COMPARISON_EVIDENCE_DATABASE_VERSION,
            )
            .map_err(js_binding_error)?;

        let upgrade_failure = Rc::new(RefCell::new(None::<String>));
        let upgrade_failure_callback = upgrade_failure.clone();
        let upgrade_request = open.clone();
        let on_upgrade = Closure::wrap(Box::new(move || {
            let request: IdbRequest = upgrade_request.clone().unchecked_into();
            let upgrade_result = (|| -> Result<(), JsValue> {
                let database: IdbDatabase = request.result()?.dyn_into()?;
                if !database
                    .object_store_names()
                    .contains(COMPARISON_EVIDENCE_OBJECT_STORE)
                {
                    database.create_object_store(COMPARISON_EVIDENCE_OBJECT_STORE)?;
                }
                Ok(())
            })();

            if let Err(error) = upgrade_result {
                *upgrade_failure_callback.borrow_mut() = Some(js_value_message(&error));
                if let Some(transaction) = request.transaction() {
                    let _ = transaction.abort();
                }
            }
        }) as Box<dyn FnMut()>);
        let upgrade_guard = OpenUpgradeGuard::new(open.clone(), on_upgrade);

        let result = OpenRequestWaiter::new(open.clone()).wait().await;
        drop(upgrade_guard);

        if let Some(message) = upgrade_failure.borrow_mut().take() {
            return Err(IndexedDbComparisonEvidenceError::SchemaUpgrade(message));
        }

        let database: IdbDatabase = result?.dyn_into().map_err(js_binding_error)?;
        if !database
            .object_store_names()
            .contains(COMPARISON_EVIDENCE_OBJECT_STORE)
        {
            return Err(IndexedDbComparisonEvidenceError::SchemaUpgrade(
                "required blind-comparisons-v1 object store is missing after open".into(),
            ));
        }

        let database_for_version_change = database.clone();
        let on_version_change = Closure::wrap(Box::new(move || {
            database_for_version_change.close();
        }) as Box<dyn FnMut()>);
        database.set_onversionchange(Some(on_version_change.as_ref().unchecked_ref()));

        Ok(Self {
            database,
            _on_version_change: on_version_change,
        })
    }

    /// Create one comparison record without overwrite semantics.
    ///
    /// `IDBObjectStore::add` is the atomic uniqueness boundary. A constraint
    /// failure never falls back to `put`: the existing record is re-read and
    /// passed through the pure replay/conflict resolver from MEL-CMP-004L1.
    pub async fn create_once(
        &self,
        envelope: ArtifactBoundBlindComparisonEnvelopeV1,
    ) -> Result<IndexedDbComparisonEvidenceCreateOutcome, IndexedDbComparisonEvidenceError> {
        let candidate = prepare_comparison_evidence_write(envelope)?;
        let transaction = self.readwrite_transaction()?;
        let store = transaction
            .object_store(COMPARISON_EVIDENCE_OBJECT_STORE)
            .map_err(js_binding_error)?;

        // Create the request before arming transaction callbacks. If the binding
        // throws synchronously, no Rust closure remains registered on the live
        // transaction after this stack frame returns.
        let request = store
            .add_with_key(
                &JsValue::from_str(candidate.json()),
                &JsValue::from_str(candidate.trial_instance_id()),
            )
            .map_err(js_binding_error)?;
        let transaction_waiter = TransactionWaiter::new(transaction);

        match RequestWaiter::new(request).wait().await {
            Ok(_) => {
                // Request success is not durable transaction success. Wait for
                // the read-write transaction itself to complete before claiming
                // that local evidence was created.
                transaction_waiter.wait().await?;
                Ok(IndexedDbComparisonEvidenceCreateOutcome::Created)
            }
            Err(RequestFailure { name, .. }) if name == "ConstraintError" => {
                // A failed `add` normally aborts the transaction. Drain its
                // terminal signal, then inspect the existing value through a
                // fresh read transaction; never replace it with `put`.
                let _ = transaction_waiter.wait().await;
                let Some(existing_json) = self.read_raw(candidate.trial_instance_id()).await? else {
                    return Err(
                        IndexedDbComparisonEvidenceError::ConstraintWithoutStoredRecord,
                    );
                };
                match resolve_create_once_comparison_evidence(Some(&existing_json), candidate)? {
                    ComparisonEvidenceCreateResolution::ExactReplay => {
                        Ok(IndexedDbComparisonEvidenceCreateOutcome::ExactReplay)
                    }
                    ComparisonEvidenceCreateResolution::Insert(_) => Err(
                        IndexedDbComparisonEvidenceError::ConstraintWithoutStoredRecord,
                    ),
                }
            }
            Err(failure) => {
                // Request error is usually more specific, but preserve a quota
                // classification surfaced only by the transaction as well.
                if matches!(
                    transaction_waiter.wait().await,
                    Err(IndexedDbComparisonEvidenceError::QuotaExceeded)
                ) {
                    return Err(IndexedDbComparisonEvidenceError::QuotaExceeded);
                }
                Err(map_request_failure(failure))
            }
        }
    }

    /// Read and validate one trial. Missing keys are truthful absence; malformed
    /// stored values and key/value mismatches are errors and are never rewritten.
    pub async fn read(
        &self,
        trial_instance_id: &str,
    ) -> Result<
        Option<ValidatedArtifactBoundBlindComparisonEnvelopeV1>,
        IndexedDbComparisonEvidenceError,
    > {
        let Some(json) = self.read_raw(trial_instance_id).await? else {
            return Ok(None);
        };
        Ok(Some(validate_keyed_stored_json(trial_instance_id, &json)?))
    }

    /// Return every locally stored comparison only after validating both each
    /// envelope and its primary-key binding.
    ///
    /// Keys and values are requested in the same readonly transaction so the two
    /// arrays describe one transactionally consistent object-store snapshot.
    pub async fn list(
        &self,
    ) -> Result<
        Vec<ValidatedArtifactBoundBlindComparisonEnvelopeV1>,
        IndexedDbComparisonEvidenceError,
    > {
        let transaction = self.readonly_transaction()?;
        let store = transaction
            .object_store(COMPARISON_EVIDENCE_OBJECT_STORE)
            .map_err(js_binding_error)?;

        // Queue both requests before yielding so the readonly transaction cannot
        // auto-complete between fetching values and their keys.
        let values_request = store.get_all().map_err(js_binding_error)?;
        let keys_request = store.get_all_keys().map_err(js_binding_error)?;
        let transaction_waiter = TransactionWaiter::new(transaction);
        let values_waiter = RequestWaiter::new(values_request);
        let keys_waiter = RequestWaiter::new(keys_request);

        let values_value = values_waiter.wait().await.map_err(map_request_failure)?;
        let keys_value = keys_waiter.wait().await.map_err(map_request_failure)?;
        transaction_waiter.wait().await?;

        if !Array::is_array(&values_value) {
            return Err(IndexedDbComparisonEvidenceError::UnexpectedStoredValue);
        }
        if !Array::is_array(&keys_value) {
            return Err(IndexedDbComparisonEvidenceError::UnexpectedStoredKey);
        }

        let values = Array::from(&values_value);
        let keys = Array::from(&keys_value);
        if values.length() != keys.length() {
            return Err(IndexedDbComparisonEvidenceError::KeyValueCountMismatch {
                keys: keys.length(),
                values: values.length(),
            });
        }

        let mut validated = Vec::with_capacity(values.length() as usize);
        for index in 0..values.length() {
            let key = keys
                .get(index)
                .as_string()
                .ok_or(IndexedDbComparisonEvidenceError::UnexpectedStoredKey)?;
            let json = values
                .get(index)
                .as_string()
                .ok_or(IndexedDbComparisonEvidenceError::UnexpectedStoredValue)?;
            validated.push(validate_keyed_stored_json(&key, &json)?);
        }
        Ok(validated)
    }

    /// Explicitly delete one already-validated local trial.
    ///
    /// Deletion is a local user action only; it says nothing about copies that
    /// may have been explicitly exported to another destination.
    pub async fn delete(
        &self,
        evidence: &ValidatedArtifactBoundBlindComparisonEnvelopeV1,
    ) -> Result<(), IndexedDbComparisonEvidenceError> {
        let transaction = self.readwrite_transaction()?;
        let store = transaction
            .object_store(COMPARISON_EVIDENCE_OBJECT_STORE)
            .map_err(js_binding_error)?;
        let request = store
            .delete(&JsValue::from_str(evidence.trial_instance_id()))
            .map_err(js_binding_error)?;
        let transaction_waiter = TransactionWaiter::new(transaction);
        RequestWaiter::new(request)
            .wait()
            .await
            .map_err(map_request_failure)?;
        transaction_waiter.wait().await
    }

    async fn read_raw(
        &self,
        trial_instance_id: &str,
    ) -> Result<Option<String>, IndexedDbComparisonEvidenceError> {
        let transaction = self.readonly_transaction()?;
        let store = transaction
            .object_store(COMPARISON_EVIDENCE_OBJECT_STORE)
            .map_err(js_binding_error)?;
        let request = store
            .get(&JsValue::from_str(trial_instance_id))
            .map_err(js_binding_error)?;
        let transaction_waiter = TransactionWaiter::new(transaction);
        let value = RequestWaiter::new(request)
            .wait()
            .await
            .map_err(map_request_failure)?;
        transaction_waiter.wait().await?;

        if value.is_undefined() {
            return Ok(None);
        }
        value
            .as_string()
            .map(Some)
            .ok_or(IndexedDbComparisonEvidenceError::UnexpectedStoredValue)
    }

    fn readonly_transaction(&self) -> Result<IdbTransaction, IndexedDbComparisonEvidenceError> {
        self.database
            .transaction_with_str_and_mode(
                COMPARISON_EVIDENCE_OBJECT_STORE,
                IdbTransactionMode::Readonly,
            )
            .map_err(js_binding_error)
    }

    fn readwrite_transaction(&self) -> Result<IdbTransaction, IndexedDbComparisonEvidenceError> {
        self.database
            .transaction_with_str_and_mode(
                COMPARISON_EVIDENCE_OBJECT_STORE,
                IdbTransactionMode::Readwrite,
            )
            .map_err(js_binding_error)
    }
}

impl Drop for IndexedDbComparisonEvidenceStore {
    fn drop(&mut self) {
        self.database.set_onversionchange(None);
        self.database.close();
    }
}

fn validate_keyed_stored_json(
    expected_trial_instance_id: &str,
    json: &str,
) -> Result<ValidatedArtifactBoundBlindComparisonEnvelopeV1, IndexedDbComparisonEvidenceError> {
    let validated = validate_stored_comparison_evidence_json(json)?;
    if validated.trial_instance_id() != expected_trial_instance_id {
        return Err(ComparisonEvidenceStoreContractError::ExistingKeyMismatch {
            expected_trial_instance_id: expected_trial_instance_id.to_string(),
            found_trial_instance_id: validated.trial_instance_id().to_string(),
        }
        .into());
    }
    Ok(validated)
}

/// Keeps `onupgradeneeded` alive and, critically, detaches it if the async open
/// future is cancelled before the request reaches a terminal event.
struct OpenUpgradeGuard {
    request: IdbOpenDbRequest,
    _callback: Closure<dyn FnMut()>,
}

impl OpenUpgradeGuard {
    fn new(request: IdbOpenDbRequest, callback: Closure<dyn FnMut()>) -> Self {
        request.set_onupgradeneeded(Some(callback.as_ref().unchecked_ref()));
        Self {
            request,
            _callback: callback,
        }
    }
}

impl Drop for OpenUpgradeGuard {
    fn drop(&mut self) {
        self.request.set_onupgradeneeded(None);
    }
}

#[derive(Clone, Debug)]
struct RequestFailure {
    name: String,
    message: String,
}

/// Cancellation-safe bridge for a single `IDBRequest`.
///
/// Drop detaches JS handlers before the Rust closures are destroyed, so tearing
/// down a Leptos task cannot leave an event target pointing at a dead closure.
struct RequestWaiter {
    request: IdbRequest,
    receiver: oneshot::Receiver<Result<JsValue, RequestFailure>>,
    _on_success: Closure<dyn FnMut()>,
    _on_error: Closure<dyn FnMut()>,
}

impl RequestWaiter {
    fn new(request: IdbRequest) -> Self {
        let (sender, receiver) = oneshot::channel();
        let sender = Rc::new(RefCell::new(Some(sender)));

        let success_sender = sender.clone();
        let success_request = request.clone();
        let on_success = Closure::wrap(Box::new(move || {
            let result = success_request.result().map_err(|error| RequestFailure {
                name: "JsBindingError".into(),
                message: js_value_message(&error),
            });
            if let Some(sender) = success_sender.borrow_mut().take() {
                let _ = sender.send(result);
            }
        }) as Box<dyn FnMut()>);

        let error_sender = sender;
        let error_request = request.clone();
        let on_error = Closure::wrap(Box::new(move || {
            let failure = request_failure(&error_request);
            if let Some(sender) = error_sender.borrow_mut().take() {
                let _ = sender.send(Err(failure));
            }
        }) as Box<dyn FnMut()>);

        request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        request.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        Self {
            request,
            receiver,
            _on_success: on_success,
            _on_error: on_error,
        }
    }

    async fn wait(mut self) -> Result<JsValue, RequestFailure> {
        (&mut self.receiver).await.unwrap_or_else(|_| {
            Err(RequestFailure {
                name: "ChannelClosed".into(),
                message: "IndexedDB request completion channel closed unexpectedly".into(),
            })
        })
    }
}

impl Drop for RequestWaiter {
    fn drop(&mut self) {
        self.request.set_onsuccess(None);
        self.request.set_onerror(None);
    }
}

/// Cancellation-safe bridge for `IDBOpenDBRequest` terminal events.
struct OpenRequestWaiter {
    request: IdbOpenDbRequest,
    receiver: oneshot::Receiver<Result<JsValue, IndexedDbComparisonEvidenceError>>,
    _on_success: Closure<dyn FnMut()>,
    _on_error: Closure<dyn FnMut()>,
    _on_blocked: Closure<dyn FnMut()>,
}

impl OpenRequestWaiter {
    fn new(request: IdbOpenDbRequest) -> Self {
        let (sender, receiver) = oneshot::channel();
        let sender = Rc::new(RefCell::new(Some(sender)));

        let success_sender = sender.clone();
        let success_request: IdbRequest = request.clone().unchecked_into();
        let on_success = Closure::wrap(Box::new(move || {
            let result = success_request.result().map_err(js_binding_error);
            if let Some(sender) = success_sender.borrow_mut().take() {
                let _ = sender.send(result);
            }
        }) as Box<dyn FnMut()>);

        let error_sender = sender.clone();
        let error_request: IdbRequest = request.clone().unchecked_into();
        let on_error = Closure::wrap(Box::new(move || {
            let failure = request_failure(&error_request);
            if let Some(sender) = error_sender.borrow_mut().take() {
                let _ = sender.send(Err(map_request_failure(failure)));
            }
        }) as Box<dyn FnMut()>);

        let blocked_sender = sender;
        let on_blocked = Closure::wrap(Box::new(move || {
            if let Some(sender) = blocked_sender.borrow_mut().take() {
                let _ = sender.send(Err(IndexedDbComparisonEvidenceError::OpenBlocked));
            }
        }) as Box<dyn FnMut()>);

        request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        request.set_onblocked(Some(on_blocked.as_ref().unchecked_ref()));

        Self {
            request,
            receiver,
            _on_success: on_success,
            _on_error: on_error,
            _on_blocked: on_blocked,
        }
    }

    async fn wait(mut self) -> Result<JsValue, IndexedDbComparisonEvidenceError> {
        (&mut self.receiver)
            .await
            .map_err(|_| IndexedDbComparisonEvidenceError::RequestChannelClosed)?
    }
}

impl Drop for OpenRequestWaiter {
    fn drop(&mut self) {
        self.request.set_onsuccess(None);
        self.request.set_onerror(None);
        self.request.set_onblocked(None);
    }
}

/// Cancellation-safe bridge for transaction completion/abort/error events.
struct TransactionWaiter {
    transaction: IdbTransaction,
    receiver: oneshot::Receiver<Result<(), IndexedDbComparisonEvidenceError>>,
    _on_complete: Closure<dyn FnMut()>,
    _on_abort: Closure<dyn FnMut()>,
    _on_error: Closure<dyn FnMut()>,
}

impl TransactionWaiter {
    fn new(transaction: IdbTransaction) -> Self {
        let (sender, receiver) = oneshot::channel();
        let sender = Rc::new(RefCell::new(Some(sender)));

        let complete_sender = sender.clone();
        let on_complete = Closure::wrap(Box::new(move || {
            if let Some(sender) = complete_sender.borrow_mut().take() {
                let _ = sender.send(Ok(()));
            }
        }) as Box<dyn FnMut()>);

        let abort_sender = sender.clone();
        let abort_transaction = transaction.clone();
        let on_abort = Closure::wrap(Box::new(move || {
            if let Some(sender) = abort_sender.borrow_mut().take() {
                let _ = sender.send(Err(transaction_failure(&abort_transaction)));
            }
        }) as Box<dyn FnMut()>);

        let error_sender = sender;
        let error_transaction = transaction.clone();
        let on_error = Closure::wrap(Box::new(move || {
            if let Some(sender) = error_sender.borrow_mut().take() {
                let _ = sender.send(Err(transaction_failure(&error_transaction)));
            }
        }) as Box<dyn FnMut()>);

        transaction.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
        transaction.set_onabort(Some(on_abort.as_ref().unchecked_ref()));
        transaction.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        Self {
            transaction,
            receiver,
            _on_complete: on_complete,
            _on_abort: on_abort,
            _on_error: on_error,
        }
    }

    async fn wait(mut self) -> Result<(), IndexedDbComparisonEvidenceError> {
        (&mut self.receiver)
            .await
            .map_err(|_| IndexedDbComparisonEvidenceError::TransactionChannelClosed)?
    }
}

impl Drop for TransactionWaiter {
    fn drop(&mut self) {
        self.transaction.set_oncomplete(None);
        self.transaction.set_onabort(None);
        self.transaction.set_onerror(None);
    }
}

fn request_failure(request: &IdbRequest) -> RequestFailure {
    match request.error() {
        Ok(Some(error)) => RequestFailure {
            name: error.name(),
            message: error.message(),
        },
        Ok(None) => RequestFailure {
            name: "UnknownError".into(),
            message: "IndexedDB request failed without a DOMException".into(),
        },
        Err(error) => RequestFailure {
            name: "JsBindingError".into(),
            message: js_value_message(&error),
        },
    }
}

fn map_request_failure(failure: RequestFailure) -> IndexedDbComparisonEvidenceError {
    match failure.name.as_str() {
        "QuotaExceededError" => IndexedDbComparisonEvidenceError::QuotaExceeded,
        "ChannelClosed" => IndexedDbComparisonEvidenceError::RequestChannelClosed,
        _ => IndexedDbComparisonEvidenceError::RequestFailed {
            name: failure.name,
            message: failure.message,
        },
    }
}

fn transaction_failure(transaction: &IdbTransaction) -> IndexedDbComparisonEvidenceError {
    let error: Option<DomException> = transaction.error();
    if error
        .as_ref()
        .is_some_and(|error| error.name() == "QuotaExceededError")
    {
        return IndexedDbComparisonEvidenceError::QuotaExceeded;
    }
    IndexedDbComparisonEvidenceError::TransactionAborted {
        name: error.as_ref().map(DomException::name),
        message: error.as_ref().map(DomException::message),
    }
}

fn js_binding_error(error: JsValue) -> IndexedDbComparisonEvidenceError {
    IndexedDbComparisonEvidenceError::Js(js_value_message(&error))
}

fn js_value_message(value: &JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| format!("{value:?}"))
}
