//! Independently retained fork-evidence protocol primitives for CIV-014.
//!
//! This module specifies the event, frontier, receipt, and remote append contract.
//! It intentionally supplies no production witness backend and does not replace
//! the current adapter's local-only fork recording. Integrating these primitives
//! with a durable local pending journal and crash/recovery coordinator is a
//! separate step; until that exists, no external fork-audit guarantee is claimed.

use crate::Digest;
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;

const FORK_EVENT_SCHEMA_VERSION: u16 = 1;
const FORK_EVENT_DOMAIN: &[u8] = b"mycelix-civ014-fork-event-v1\0";
const FORK_RECEIPT_DOMAIN: &[u8] = b"mycelix-civ014-fork-receipt-v1\0";
const ZERO_DIGEST: Digest = [0; 32];

/// A monotonic frontier retained by a separately operated fork witness.
///
/// An empty frontier is valid only when explicitly provisioned. An unknown
/// log/epoch is unavailable, never implicitly initialized to genesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkFrontier {
    pub log_id: String,
    pub witness_epoch: u64,
    pub event_count: u64,
    pub tail_digest: Digest,
}

impl ForkFrontier {
    pub fn empty(log_id: impl Into<String>, witness_epoch: u64) -> Self {
        Self {
            log_id: log_id.into(),
            witness_epoch,
            event_count: 0,
            tail_digest: ZERO_DIGEST,
        }
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        if self.log_id.is_empty() {
            return Err(ForkWitnessError::InvalidInput("log ID is empty"));
        }
        if self.event_count == 0 && self.tail_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidFrontier(
                "empty frontier must use the zero tail digest",
            ));
        }
        Ok(())
    }

    fn after_event(&self, event: &ForkEvent) -> Result<Self, ForkWitnessError> {
        self.validate()?;
        event.validate_for(self)?;
        let event_count = self
            .event_count
            .checked_add(1)
            .ok_or(ForkWitnessError::FrontierOverflow)?;
        Ok(Self {
            log_id: self.log_id.clone(),
            witness_epoch: self.witness_epoch,
            event_count,
            tail_digest: event.event_digest,
        })
    }
}

/// Canonical fork event. The event digest is also the stable event identity.
///
/// The event records the accepted-head position observed when the conflict was
/// discovered, but it does not advance that accepted head. The previous fork
/// frontier count and tail are included in the hashed payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkEvent {
    pub schema_version: u16,
    pub log_id: String,
    pub witness_epoch: u64,
    pub previous_event_count: u64,
    pub previous_event_digest: Digest,
    pub accepted_generation: u64,
    pub accepted_head_digest: Digest,
    pub fork_generation: u64,
    pub first_record_digest: Digest,
    pub conflicting_record_digest: Digest,
    pub event_digest: Digest,
}

impl ForkEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        frontier: &ForkFrontier,
        accepted_generation: u64,
        accepted_head_digest: Digest,
        fork_generation: u64,
        first_record_digest: Digest,
        conflicting_record_digest: Digest,
    ) -> Result<Self, ForkWitnessError> {
        frontier.validate()?;
        if fork_generation == 0 {
            return Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            ));
        }
        if accepted_generation == 0 && accepted_head_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            ));
        }
        if fork_generation > accepted_generation.saturating_add(1) {
            return Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            ));
        }
        if first_record_digest == conflicting_record_digest {
            return Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            ));
        }
        let mut event = Self {
            schema_version: FORK_EVENT_SCHEMA_VERSION,
            log_id: frontier.log_id.clone(),
            witness_epoch: frontier.witness_epoch,
            previous_event_count: frontier.event_count,
            previous_event_digest: frontier.tail_digest,
            accepted_generation,
            accepted_head_digest,
            fork_generation,
            first_record_digest,
            conflicting_record_digest,
            event_digest: ZERO_DIGEST,
        };
        event.event_digest = event.calculate_digest()?;
        Ok(event)
    }

    /// Return the exact domain-separated canonical event encoding.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ForkWitnessError> {
        if self.schema_version != FORK_EVENT_SCHEMA_VERSION {
            return Err(ForkWitnessError::InvalidInput(
                "unsupported fork-event schema version",
            ));
        }
        if self.log_id.is_empty() {
            return Err(ForkWitnessError::InvalidInput("log ID is empty"));
        }
        if self.fork_generation == 0 {
            return Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            ));
        }
        if self.accepted_generation == 0 && self.accepted_head_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            ));
        }
        if self.fork_generation > self.accepted_generation.saturating_add(1) {
            return Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            ));
        }
        if self.first_record_digest == self.conflicting_record_digest {
            return Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            ));
        }
        if self.previous_event_count == 0 && self.previous_event_digest != ZERO_DIGEST {
            return Err(ForkWitnessError::InvalidInput(
                "empty previous frontier must use zero digest",
            ));
        }

        let mut bytes = Vec::with_capacity(FORK_EVENT_DOMAIN.len() + self.log_id.len() + 202);
        bytes.extend_from_slice(FORK_EVENT_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        encode_field(&mut bytes, self.log_id.as_bytes());
        bytes.extend_from_slice(&self.witness_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.previous_event_count.to_be_bytes());
        bytes.extend_from_slice(&self.previous_event_digest);
        bytes.extend_from_slice(&self.accepted_generation.to_be_bytes());
        bytes.extend_from_slice(&self.accepted_head_digest);
        bytes.extend_from_slice(&self.fork_generation.to_be_bytes());
        bytes.extend_from_slice(&self.first_record_digest);
        bytes.extend_from_slice(&self.conflicting_record_digest);
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        if self.calculate_digest()? != self.event_digest {
            return Err(ForkWitnessError::InvalidEventDigest);
        }
        Ok(())
    }

    pub fn validate_for(&self, frontier: &ForkFrontier) -> Result<(), ForkWitnessError> {
        frontier.validate()?;
        self.validate()?;
        if self.log_id != frontier.log_id || self.witness_epoch != frontier.witness_epoch {
            return Err(ForkWitnessError::FrontierScopeMismatch);
        }
        if self.previous_event_count != frontier.event_count
            || self.previous_event_digest != frontier.tail_digest
        {
            return Err(ForkWitnessError::FrontierConflict);
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Result<Digest, ForkWitnessError> {
        Ok(Sha256::digest(self.canonical_bytes()?).into())
    }
}

/// Durable receipt returned by the independent fork witness.
///
/// The receipt commits to the exact event, prior frontier, and new frontier.
/// Callers must validate it against their pending local event before reporting
/// a completed append.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForkAppendReceipt {
    pub event: ForkEvent,
    pub previous_frontier: ForkFrontier,
    pub frontier_after: ForkFrontier,
    pub receipt_digest: Digest,
}

impl ForkAppendReceipt {
    pub fn build(
        previous_frontier: ForkFrontier,
        event: ForkEvent,
    ) -> Result<Self, ForkWitnessError> {
        previous_frontier.validate()?;
        event.validate_for(&previous_frontier)?;
        let frontier_after = previous_frontier.after_event(&event)?;
        let mut receipt = Self {
            event,
            previous_frontier,
            frontier_after,
            receipt_digest: ZERO_DIGEST,
        };
        receipt.receipt_digest = receipt.calculate_digest()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), ForkWitnessError> {
        self.previous_frontier.validate()?;
        self.event.validate_for(&self.previous_frontier)?;
        let expected_after = self.previous_frontier.after_event(&self.event)?;
        if self.frontier_after != expected_after {
            return Err(ForkWitnessError::InvalidReceipt(
                "receipt frontier does not advance the prior frontier by one exact event",
            ));
        }
        if self.calculate_digest()? != self.receipt_digest {
            return Err(ForkWitnessError::InvalidReceiptDigest);
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Result<Digest, ForkWitnessError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FORK_RECEIPT_DOMAIN);
        encode_field(&mut bytes, self.previous_frontier.log_id.as_bytes());
        bytes.extend_from_slice(&self.previous_frontier.witness_epoch.to_be_bytes());
        bytes.extend_from_slice(&self.previous_frontier.event_count.to_be_bytes());
        bytes.extend_from_slice(&self.previous_frontier.tail_digest);
        bytes.extend_from_slice(&self.event.event_digest);
        bytes.extend_from_slice(&self.frontier_after.event_count.to_be_bytes());
        bytes.extend_from_slice(&self.frontier_after.tail_digest);
        Ok(Sha256::digest(bytes).into())
    }
}

/// Separate capability from accepted-head CAS. Implementations must retain full
/// event payloads and receipts outside the local SQLite rollback domain.
///
/// append_event is linearizable and idempotent for the exact same event. If the
/// response is ambiguous, find_event resolves the stable event digest; read_event
/// lets recovery reconstruct a local journal whose fork rows were erased. A
/// frontier-only service cannot provide that reconstruction.
pub trait IndependentForkWitness: Send + Sync {
    fn current_frontier(
        &self,
        log_id: &str,
        witness_epoch: u64,
    ) -> Result<ForkFrontier, ForkWitnessError>;

    fn append_event(
        &self,
        expected_frontier: &ForkFrontier,
        event: &ForkEvent,
    ) -> Result<ForkAppendReceipt, ForkWitnessError>;

    fn find_event(
        &self,
        log_id: &str,
        witness_epoch: u64,
        event_digest: Digest,
    ) -> Result<Option<ForkAppendReceipt>, ForkWitnessError>;

    /// Sequence is one-based. A successful implementation retains the full
    /// event body, not just its digest, for local restoration after erasure.
    fn read_event(
        &self,
        log_id: &str,
        witness_epoch: u64,
        sequence: u64,
    ) -> Result<Option<ForkEvent>, ForkWitnessError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForkWitnessError {
    Unavailable,
    FrontierConflict,
    FrontierOverflow,
    FrontierScopeMismatch,
    EventIdentityConflict,
    InvalidEventDigest,
    InvalidFrontier(&'static str),
    InvalidInput(&'static str),
    InvalidReceipt(&'static str),
    InvalidReceiptDigest,
    Other(String),
}

impl fmt::Display for ForkWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "independent fork witness unavailable"),
            Self::FrontierConflict => write!(f, "independent fork-witness frontier conflict"),
            Self::FrontierOverflow => write!(f, "independent fork-witness frontier overflow"),
            Self::FrontierScopeMismatch => write!(f, "fork event is bound to another log or epoch"),
            Self::EventIdentityConflict => write!(f, "fork event identity conflicts with retained payload"),
            Self::InvalidEventDigest => write!(f, "fork-event digest does not match canonical payload"),
            Self::InvalidFrontier(message) => write!(f, "invalid fork-witness frontier: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid fork-witness input: {message}"),
            Self::InvalidReceipt(message) => write!(f, "invalid fork-witness receipt: {message}"),
            Self::InvalidReceiptDigest => write!(f, "fork-witness receipt digest mismatch"),
            Self::Other(message) => write!(f, "independent fork-witness error: {message}"),
        }
    }
}

impl std::error::Error for ForkWitnessError {}

fn encode_field(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryForkWitness {
        events: Mutex<HashMap<(String, u64), Vec<ForkAppendReceipt>>>,
    }

    impl MemoryForkWitness {
        fn provision(&self, log_id: &str, witness_epoch: u64) {
            self.events
                .lock()
                .expect("fork-witness mutex")
                .entry((log_id.to_owned(), witness_epoch))
                .or_default();
        }
    }

    impl IndependentForkWitness for MemoryForkWitness {
        fn current_frontier(
            &self,
            log_id: &str,
            witness_epoch: u64,
        ) -> Result<ForkFrontier, ForkWitnessError> {
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            Ok(receipts
                .last()
                .map(|receipt| receipt.frontier_after.clone())
                .unwrap_or_else(|| ForkFrontier::empty(log_id, witness_epoch)))
        }

        fn append_event(
            &self,
            expected_frontier: &ForkFrontier,
            event: &ForkEvent,
        ) -> Result<ForkAppendReceipt, ForkWitnessError> {
            expected_frontier.validate()?;
            event.validate_for(expected_frontier)?;
            let mut events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get_mut(&(expected_frontier.log_id.clone(), expected_frontier.witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;

            if let Some(existing) = receipts
                .iter()
                .find(|receipt| receipt.event.event_digest == event.event_digest)
            {
                if existing.event != *event {
                    return Err(ForkWitnessError::EventIdentityConflict);
                }
                existing.validate()?;
                return Ok(existing.clone());
            }

            let actual_frontier = receipts
                .last()
                .map(|receipt| receipt.frontier_after.clone())
                .unwrap_or_else(|| ForkFrontier::empty(
                    &expected_frontier.log_id,
                    expected_frontier.witness_epoch,
                ));
            if actual_frontier != *expected_frontier {
                return Err(ForkWitnessError::FrontierConflict);
            }
            let receipt = ForkAppendReceipt::build(expected_frontier.clone(), event.clone())?;
            receipt.validate()?;
            receipts.push(receipt.clone());
            Ok(receipt)
        }

        fn find_event(
            &self,
            log_id: &str,
            witness_epoch: u64,
            event_digest: Digest,
        ) -> Result<Option<ForkAppendReceipt>, ForkWitnessError> {
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            let found = receipts
                .iter()
                .find(|receipt| receipt.event.event_digest == event_digest)
                .cloned();
            if let Some(receipt) = found.as_ref() {
                receipt.validate()?;
            }
            Ok(found)
        }

        fn read_event(
            &self,
            log_id: &str,
            witness_epoch: u64,
            sequence: u64,
        ) -> Result<Option<ForkEvent>, ForkWitnessError> {
            if sequence == 0 {
                return Err(ForkWitnessError::InvalidInput(
                    "fork-event sequence is one-based",
                ));
            }
            let index = usize::try_from(sequence - 1)
                .map_err(|_| ForkWitnessError::InvalidInput("fork-event sequence is too large"))?;
            let events = self
                .events
                .lock()
                .map_err(|_| ForkWitnessError::Other("fork-witness mutex poisoned".into()))?;
            let receipts = events
                .get(&(log_id.to_owned(), witness_epoch))
                .ok_or(ForkWitnessError::Unavailable)?;
            Ok(receipts.get(index).map(|receipt| receipt.event.clone()))
        }
    }

    fn d(byte: u8) -> Digest {
        [byte; 32]
    }

    #[test]
    fn fork_event_and_receipt_are_canonical_and_bound_to_frontier() {
        let frontier = ForkFrontier::empty("log-a", 7);
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6))
            .expect("build canonical event");
        event.validate().expect("event digest verifies");
        let receipt = ForkAppendReceipt::build(frontier.clone(), event.clone())
            .expect("build receipt");
        receipt.validate().expect("receipt binds prior and next frontiers");
        assert_eq!(receipt.frontier_after.event_count, 1);
        assert_eq!(receipt.frontier_after.tail_digest, event.event_digest);
        assert_ne!(receipt.receipt_digest, event.event_digest);
    }

    #[test]
    fn exact_remote_append_retry_returns_one_stable_receipt() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let before = witness.current_frontier("log-a", 7).expect("provisioned frontier");
        let event = ForkEvent::build(&before, 4, d(4), 4, d(5), d(6)).expect("build event");
        let first = witness.append_event(&before, &event).expect("append event");
        let retry = witness
            .append_event(&before, &event)
            .expect("exact retry returns existing receipt");
        assert_eq!(retry, first);
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            1,
            "idempotent retry cannot duplicate remote evidence",
        );
    }

    #[test]
    fn stale_frontier_cannot_append_a_different_event() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let empty = witness.current_frontier("log-a", 7).expect("empty frontier");
        let first = ForkEvent::build(&empty, 4, d(4), 4, d(5), d(6)).expect("first event");
        witness.append_event(&empty, &first).expect("first append");
        let competing = ForkEvent::build(&empty, 4, d(4), 4, d(7), d(8))
            .expect("competing event is canonical for stale frontier");
        assert_eq!(
            witness.append_event(&empty, &competing),
            Err(ForkWitnessError::FrontierConflict),
        );
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            1,
        );
    }

    #[test]
    fn invalid_event_digest_is_rejected_before_remote_append() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let frontier = witness.current_frontier("log-a", 7).expect("empty frontier");
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let mut tampered = event.clone();
        tampered.accepted_head_digest = d(9);
        assert_eq!(
            witness.append_event(&frontier, &tampered),
            Err(ForkWitnessError::InvalidEventDigest),
        );
        assert_eq!(
            witness.current_frontier("log-a", 7).expect("read frontier").event_count,
            0,
        );
    }

    #[test]
    fn remote_receipt_can_resolve_an_ambiguous_append_and_restore_payload() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        let frontier = witness.current_frontier("log-a", 7).expect("empty frontier");
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let committed = witness.append_event(&frontier, &event).expect("remote commit");
        let found = witness
            .find_event("log-a", 7, event.event_digest)
            .expect("read back by stable identity")
            .expect("event exists remotely");
        assert_eq!(found, committed);
        let restored = witness
            .read_event("log-a", 7, 1)
            .expect("read retained payload")
            .expect("remote witness retains full body");
        assert_eq!(restored, event);
    }

    #[test]
    fn missing_remote_log_is_unavailable_not_implicit_genesis() {
        let witness = MemoryForkWitness::default();
        assert_eq!(
            witness.current_frontier("log-not-provisioned", 7),
            Err(ForkWitnessError::Unavailable),
        );
    }

    #[test]
    fn event_rejects_generation_zero_and_identical_competitors() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 0, ZERO_DIGEST, 0, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "fork evidence cannot use generation zero",
            )),
        );
        assert_eq!(
            ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(5)),
            Err(ForkWitnessError::InvalidInput(
                "fork event must bind two distinct record digests",
            )),
        );
    }

    #[test]
    fn event_rejects_a_malformed_genesis_head_digest() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 0, d(9), 1, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "genesis accepted head must use the zero digest",
            )),
        );
    }

    #[test]
    fn event_rejects_a_fork_generation_ahead_by_more_than_one() {
        let frontier = ForkFrontier::empty("log-a", 7);
        assert_eq!(
            ForkEvent::build(&frontier, 4, d(4), 6, d(5), d(6)),
            Err(ForkWitnessError::InvalidInput(
                "fork generation cannot exceed the next accepted-head generation",
            )),
        );
    }

    #[test]
    fn receipt_tampering_is_detected() {
        let frontier = ForkFrontier::empty("log-a", 7);
        let event = ForkEvent::build(&frontier, 4, d(4), 4, d(5), d(6)).expect("build event");
        let mut receipt = ForkAppendReceipt::build(frontier, event).expect("build receipt");
        receipt.frontier_after.event_count = 2;
        assert_eq!(
            receipt.validate(),
            Err(ForkWitnessError::InvalidReceipt(
                "receipt frontier does not advance the prior frontier by one exact event",
            )),
        );
    }

    #[test]
    fn event_sequence_reads_are_one_based() {
        let witness = MemoryForkWitness::default();
        witness.provision("log-a", 7);
        assert_eq!(
            witness.read_event("log-a", 7, 0),
            Err(ForkWitnessError::InvalidInput("fork-event sequence is one-based")),
        );
    }
}
