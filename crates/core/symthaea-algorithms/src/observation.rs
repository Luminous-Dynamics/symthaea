// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded content-addressed objects for discovery observations.
//!
//! [`DiscoveryLedger`](crate::ledger::DiscoveryLedger) intentionally stores only an
//! `observation_id` on each event. This module supplies an optional canonical object archive for
//! those IDs so negative results can be reconstructed without coupling the ledger to Forge,
//! compiler diagnostics, counterexample formats, benchmark harnesses, or future generators.
//!
//! Observation objects are descriptive evidence containers only. Presence in this store does not
//! establish correctness, performance, replication, promotion, or runtime authority.

use crate::discovery::DiscoveryRun;
use crate::ledger::DiscoveryLedger;
use crate::ContentId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Keep core observation objects small. Large logs/traces should be retained externally and
/// represented here by a compact descriptor containing their own content identity and metadata.
pub const MAX_INLINE_OBSERVATION_BYTES: usize = 64 * 1024;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ObservationError {
    #[error("observation schema must be non-empty canonical single-line text")]
    InvalidSchema,
    #[error("observation payload must not be empty")]
    EmptyPayload,
    #[error("observation payload exceeds inline limit: {observed} > {maximum} bytes")]
    PayloadTooLarge { observed: usize, maximum: usize },
    #[error("observation identity does not match its canonical fields")]
    IdentityMismatch,
    #[error("observation store contains a duplicate object identity")]
    DuplicateObservation,
    #[error("observation store key does not match object identity")]
    StoreKeyMismatch,
    #[error("ledger references an observation that is absent from the store: {0}")]
    MissingObservation(String),
    #[error("discovery ledger is invalid for the supplied run: {0}")]
    InvalidLedger(String),
}

/// Encoding of the exact payload bytes. `Json` means the producer promises canonical JSON bytes;
/// this core crate does not parse or normalize JSON itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationEncoding {
    Utf8,
    Json,
    Binary,
}

impl ObservationEncoding {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Utf8 => b"utf8",
            Self::Json => b"json",
            Self::Binary => b"binary",
        }
    }
}

/// One bounded immutable observation payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationObject {
    id: ContentId,
    schema: String,
    encoding: ObservationEncoding,
    payload: Vec<u8>,
}

impl ObservationObject {
    pub fn new(
        schema: impl Into<String>,
        encoding: ObservationEncoding,
        payload: Vec<u8>,
    ) -> Result<Self, ObservationError> {
        let schema = schema.into();
        validate_schema(&schema)?;
        if payload.is_empty() {
            return Err(ObservationError::EmptyPayload);
        }
        if payload.len() > MAX_INLINE_OBSERVATION_BYTES {
            return Err(ObservationError::PayloadTooLarge {
                observed: payload.len(),
                maximum: MAX_INLINE_OBSERVATION_BYTES,
            });
        }
        let byte_len = (payload.len() as u64).to_be_bytes();
        let id = ContentId::derive(
            "symthaea.discovery-observation-object.v1",
            [
                schema.as_bytes(),
                encoding.tag(),
                byte_len.as_slice(),
                payload.as_slice(),
            ],
        );
        Ok(Self {
            id,
            schema,
            encoding,
            payload,
        })
    }

    pub fn utf8(
        schema: impl Into<String>,
        payload: impl Into<String>,
    ) -> Result<Self, ObservationError> {
        Self::new(schema, ObservationEncoding::Utf8, payload.into().into_bytes())
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn encoding(&self) -> ObservationEncoding {
        self.encoding
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let rebuilt = Self::new(self.schema.clone(), self.encoding, self.payload.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ObservationError::IdentityMismatch)
        }
    }
}

fn validate_schema(schema: &str) -> Result<(), ObservationError> {
    if schema.trim().is_empty()
        || schema.trim() != schema
        || schema.chars().any(char::is_control)
    {
        return Err(ObservationError::InvalidSchema);
    }
    Ok(())
}

/// Deduplicated validated observation archive. The map key is the canonical string form of the
/// object's content identity so this type does not impose ordering requirements on `ContentId`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ObservationStore {
    objects: BTreeMap<String, ObservationObject>,
}

impl ObservationStore {
    pub fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
        }
    }

    /// Rehydrate persisted objects only after every object and key relationship validates.
    /// Duplicate object IDs are rejected as noncanonical persistence rather than silently folded.
    pub fn from_objects(
        objects: impl IntoIterator<Item = ObservationObject>,
    ) -> Result<Self, ObservationError> {
        let mut store = Self::new();
        for object in objects {
            object.validate()?;
            let key = object.id().as_str().to_string();
            if store.objects.insert(key, object).is_some() {
                return Err(ObservationError::DuplicateObservation);
            }
        }
        store.validate()?;
        Ok(store)
    }

    /// Insert one canonical object. Re-inserting the exact same content-addressed object is an
    /// idempotent no-op and returns `false`.
    pub fn insert(&mut self, object: ObservationObject) -> Result<bool, ObservationError> {
        object.validate()?;
        let key = object.id().as_str().to_string();
        if self.objects.contains_key(&key) {
            return Ok(false);
        }
        self.objects.insert(key, object);
        Ok(true)
    }

    pub fn get(&self, id: &ContentId) -> Option<&ObservationObject> {
        self.objects.get(id.as_str())
    }

    pub fn contains(&self, id: &ContentId) -> bool {
        self.objects.contains_key(id.as_str())
    }

    pub fn objects(&self) -> impl Iterator<Item = &ObservationObject> {
        self.objects.values()
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        for (key, object) in &self.objects {
            object.validate()?;
            if key != object.id().as_str() {
                return Err(ObservationError::StoreKeyMismatch);
            }
        }
        Ok(())
    }

    /// Require every event in an already-valid ledger to have a reconstructable observation
    /// object in this store. Extra objects are permitted because one store may cover multiple
    /// snapshots or generators.
    pub fn validate_complete_for_ledger(
        &self,
        run: &DiscoveryRun,
        ledger: &DiscoveryLedger,
    ) -> Result<(), ObservationError> {
        self.validate()?;
        ledger
            .validate_for(run)
            .map_err(|error| ObservationError::InvalidLedger(error.to_string()))?;
        for event in ledger.events() {
            if !self.contains(event.observation_id()) {
                return Err(ObservationError::MissingObservation(
                    event.observation_id().as_str().to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Deterministic identity of the current sorted object set.
    pub fn snapshot_id(&self) -> Result<ContentId, ObservationError> {
        self.validate()?;
        let mut parts = Vec::with_capacity(self.objects.len() + 1);
        parts.push((self.objects.len() as u64).to_be_bytes().to_vec());
        parts.extend(self.objects.keys().map(|key| key.as_bytes().to_vec()));
        Ok(ContentId::derive(
            "symthaea.discovery-observation-store.v1",
            parts.iter().map(Vec::as_slice),
        ))
    }
}

impl Default for ObservationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{DiscoveryPolicy, SearchBudget};
    use crate::ledger::{DiscoveryEventKind, DiscoveryLedger};
    use crate::{DeterminismRequirement, DiscoveryRisk, ProblemSpec, SemanticGuarantee};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn run() -> DiscoveryRun {
        let problem = ProblemSpec::new(
            "observation-store-test",
            "Return the exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            cid("generator", "observation-test"),
            "abc123",
            SearchBudget::new(4, 2, 4).unwrap(),
            7,
        )
        .unwrap()
    }

    #[test]
    fn object_identity_binds_schema_encoding_and_exact_bytes() {
        let a = ObservationObject::utf8("forge.compile-failure.v1", "same payload").unwrap();
        let b = ObservationObject::utf8("forge.test-failure.v1", "same payload").unwrap();
        let c = ObservationObject::new(
            "forge.compile-failure.v1",
            ObservationEncoding::Binary,
            b"same payload".to_vec(),
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
        assert!(a.validate().is_ok());
    }

    #[test]
    fn oversized_inline_payload_is_rejected() {
        let payload = vec![0u8; MAX_INLINE_OBSERVATION_BYTES + 1];
        assert!(matches!(
            ObservationObject::new("large.v1", ObservationEncoding::Binary, payload),
            Err(ObservationError::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn store_proves_complete_ledger_observation_coverage() {
        let run = run();
        let generated = ObservationObject::utf8("test.generated.v1", "generated").unwrap();
        let rejected = ObservationObject::utf8("test.rejected.v1", "counterexample").unwrap();
        let completed = ObservationObject::utf8("test.completed.v1", "complete").unwrap();
        let artifact = cid("artifact", "candidate-a");

        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::CandidateGenerated,
                Some(artifact.clone()),
                generated.id().clone(),
            )
            .unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::RejectedCorrectness,
                Some(artifact),
                rejected.id().clone(),
            )
            .unwrap();
        ledger.complete(&run, completed.id().clone()).unwrap();

        let complete = ObservationStore::from_objects(vec![
            generated.clone(),
            rejected.clone(),
            completed.clone(),
        ])
        .unwrap();
        assert!(complete.validate_complete_for_ledger(&run, &ledger).is_ok());

        let incomplete = ObservationStore::from_objects(vec![generated, completed]).unwrap();
        assert!(matches!(
            incomplete.validate_complete_for_ledger(&run, &ledger),
            Err(ObservationError::MissingObservation(_))
        ));
    }

    #[test]
    fn insertion_is_idempotent_but_persisted_duplicates_are_noncanonical() {
        let object = ObservationObject::utf8("test.v1", "payload").unwrap();
        let mut store = ObservationStore::new();
        assert!(store.insert(object.clone()).unwrap());
        assert!(!store.insert(object.clone()).unwrap());
        assert!(matches!(
            ObservationStore::from_objects(vec![object.clone(), object]),
            Err(ObservationError::DuplicateObservation)
        ));
    }

    #[test]
    fn snapshot_identity_is_insertion_order_independent() {
        let a = ObservationObject::utf8("a.v1", "alpha").unwrap();
        let b = ObservationObject::utf8("b.v1", "beta").unwrap();
        let first = ObservationStore::from_objects(vec![a.clone(), b.clone()]).unwrap();
        let second = ObservationStore::from_objects(vec![b, a]).unwrap();
        assert_eq!(first.snapshot_id().unwrap(), second.snapshot_id().unwrap());
    }
}
