//! Compare-and-swap resource reservation reference ledger for preflighted NAS actions.
//!
//! This module closes the read-only-preflight race with an optimistic revision head.
//! A privately constructible VerifiedMeasurementReservation can only be obtained by
//! running the complete measurement preflight in nas_action_authorization. The
//! reservation ledger then checks that the preflight was made against its exact current
//! head, validates the complete cost vector, and atomically mutates its in-process model
//! only after a full candidate-state integrity replay succeeds.
//!
//! State transitions:
//!   RESERVED -> COMMITTED | RELEASED | EXPIRED
//!
//! COMMITTED is terminal and is charged permanently against the budget. RELEASED and
//! EXPIRED release only uncommitted headroom. An idempotent retry with the same scope
//! digest and exact evidence returns the existing state without appending another event.
//!
//! This is a deterministic reference state machine, not a distributed transaction
//! service. The &mut self API is sequential only inside one process. Production must
//! implement the same expected-head compare-and-swap within durable serializable storage
//! (or a dedicated reservation service), with trusted lease time, crash recovery, and
//! authenticated callers. A Git commit or SHA-256 digest is not a trusted clock or
//! external authorization signature.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::multi_resource_budget::{
    remaining_resource_headroom, validate_resource_registry, ResourceBudget,
    ResourceCostVector, ResourceHeadroom, ResourceRegistry,
};
use crate::nas_action_authorization::{
    compute_preflight_receipt_sha256, verify_action_preflight_receipt_digest,
    NasActionKind, NasActionPreflightReceipt,
    VerifiedMeasurementReservation,
};

const LEDGER_SCHEMA: &str = "symthaea.nas.resource-reservation-ledger";
const LEDGER_VERSION: &str = "0.1.0";
const GENESIS_DOMAIN: &[u8] = b"symthaea.nas.resource-reservation-genesis.v0.1";
const LEDGER_DOMAIN: &[u8] = b"symthaea.nas.resource-reservation-ledger.v0.1";
const EVENT_DOMAIN: &[u8] = b"symthaea.nas.resource-reservation-event.v0.1";
const RECORD_DOMAIN: &[u8] = b"symthaea.nas.resource-reservation-record.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.resource-reservation-receipt.v0.1";
const BUDGET_DOMAIN: &[u8] = b"symthaea.nas.resource-budget.v0.1";
const COST_DOMAIN: &[u8] = b"symthaea.nas.resource-cost-vector.v0.1";
const MAX_RESERVATIONS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationState {
    Reserved,
    Committed,
    Released,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationEventKind {
    Reserved,
    Committed,
    Released,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceReservationRecord {
    /// Idempotency key: exact action scope SHA-256.
    pub reservation_id: String,
    pub action_id: String,
    pub preflight_receipt_sha256: String,
    pub preflight_receipt: NasActionPreflightReceipt,
    pub resource_registry_sha256: String,
    pub resource_budget_sha256: String,
    pub resource_cost_sha256: String,
    pub resource_cost: ResourceCostVector,
    pub reserved_at_epoch: u64,
    pub expires_at_epoch: u64,
    pub state: ReservationState,
    pub latest_event_sha256: String,
    pub record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceReservationEvent {
    pub sequence: u64,
    pub kind: ReservationEventKind,
    pub reservation_id: String,
    pub action_id: String,
    pub preflight_receipt_sha256: String,
    pub resource_cost_sha256: String,
    pub epoch: u64,
    pub expires_at_epoch: u64,
    pub reason_code: String,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceReservationReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub reservation_id: String,
    pub action_id: String,
    pub state: ReservationState,
    pub preflight_receipt_sha256: String,
    pub resource_registry_sha256: String,
    pub resource_budget_sha256: String,
    pub resource_cost_sha256: String,
    pub reserved_at_epoch: u64,
    pub expires_at_epoch: u64,
    pub latest_event_sha256: String,
    pub ledger_head_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceReservationError {
    pub field: String,
    pub message: String,
}

impl ResourceReservationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ResourceReservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for ResourceReservationError {}

/// Reference reservation ledger. All mutable fields are private; mutation is only
/// possible through revision-checked state transitions and every transition is replayed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceReservationLedger {
    schema: String,
    version: String,
    run_id: String,
    resource_registry: ResourceRegistry,
    initial_budget: ResourceBudget,
    max_lease_epochs: u64,
    current_epoch: u64,
    genesis_sha256: String,
    reservations: Vec<ResourceReservationRecord>,
    events: Vec<ResourceReservationEvent>,
    ledger_sha256: String,
}

impl ResourceReservationLedger {
    pub fn new(
        run_id: impl Into<String>,
        resource_registry: ResourceRegistry,
        initial_budget: ResourceBudget,
        max_lease_epochs: u64,
    ) -> Result<Self, ResourceReservationError> {
        let run_id = run_id.into();
        if run_id.trim().is_empty() {
            return Err(ResourceReservationError::new("run_id", "must be non-empty"));
        }
        if max_lease_epochs == 0 {
            return Err(ResourceReservationError::new(
                "max_lease_epochs",
                "must be strictly positive",
            ));
        }
        validate_resource_registry(&resource_registry)
            .map_err(|error| ResourceReservationError::new("resource_registry", error.to_string()))?;
        remaining_resource_headroom(&resource_registry, &initial_budget)
            .map_err(|error| ResourceReservationError::new("initial_budget", error.to_string()))?;

        let genesis_sha256 = compute_genesis_sha256(
            &run_id,
            &resource_registry,
            &initial_budget,
            max_lease_epochs,
        )?;
        let mut ledger = Self {
            schema: LEDGER_SCHEMA.to_owned(),
            version: LEDGER_VERSION.to_owned(),
            run_id,
            resource_registry,
            initial_budget,
            max_lease_epochs,
            current_epoch: 0,
            genesis_sha256,
            reservations: Vec::new(),
            events: Vec::new(),
            ledger_sha256: String::new(),
        };
        ledger.ledger_sha256 = compute_ledger_sha256(&ledger)?;
        Ok(ledger)
    }

    pub fn head_sha256(&self) -> &str {
        &self.ledger_sha256
    }

    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    pub fn reservations(&self) -> &[ResourceReservationRecord] {
        &self.reservations
    }

    pub fn events(&self) -> &[ResourceReservationEvent] {
        &self.events
    }

    /// Current spendable headroom after base spending/reserves and this ledger's
    /// active reservations and committed actions.
    pub fn available_headroom(&self) -> Result<Vec<ResourceHeadroom>, ResourceReservationError> {
        verify_resource_reservation_ledger(self)?;
        let mut headroom = remaining_resource_headroom(
            &self.resource_registry,
            &self.initial_budget,
        )
        .map_err(|error| ResourceReservationError::new("initial_budget", error.to_string()))?;

        for record in &self.reservations {
            if !matches!(record.state, ReservationState::Reserved | ReservationState::Committed) {
                continue;
            }
            for (index, quantity) in record.resource_cost.resources.iter().enumerate() {
                let available = &mut headroom[index].available_units;
                *available = available.checked_sub(quantity.units).ok_or_else(|| {
                    ResourceReservationError::new(
                        format!("reservations[{}].resource_cost.resources[{index}]", record.reservation_id),
                        "active reservations and committed consumption exceed available headroom",
                    )
                })?;
            }
        }
        Ok(headroom)
    }

    /// Reserve a measurement action only when its privately issued preflight capability
    /// is bound to the exact current ledger head. The first request from a given head
    /// wins; a stale competing request must refresh and re-run preflight.
    pub fn reserve_measurement(
        &mut self,
        expected_head_sha256: &str,
        proof: VerifiedMeasurementReservation,
        expires_at_epoch: u64,
    ) -> Result<ResourceReservationReceipt, ResourceReservationError> {
        verify_resource_reservation_ledger(self)?;
        validate_measurement_proof(self, &proof)?;

        // Idempotent retry is checked before CAS so a lost response can be safely retried
        // with the original expected head. A mismatch on the same idempotency key fails.
        if let Some(existing) = self
            .reservations
            .iter()
            .find(|record| record.reservation_id == proof.scope_sha256)
        {
            if existing.action_id != proof.preflight_receipt.action_id
                || existing.preflight_receipt_sha256 != proof.preflight_receipt.receipt_sha256
                || existing.resource_cost != proof.resource_cost
                || existing.expires_at_epoch != expires_at_epoch
            {
                return Err(ResourceReservationError::new(
                    "idempotency_key",
                    "the same action scope was previously used with different action, receipt, or cost evidence",
                ));
            }
            return self.receipt_for(existing);
        }

        self.check_expected_head(expected_head_sha256)?;
        if proof.reservation_ledger_head_sha256 != self.ledger_sha256 {
            return Err(ResourceReservationError::new(
                "reservation_ledger_head_sha256",
                "preflight capability was created against a different reservation-ledger revision",
            ));
        }
        if expires_at_epoch <= self.current_epoch {
            return Err(ResourceReservationError::new(
                "expires_at_epoch",
                "lease expiry must be later than the current monotone epoch",
            ));
        }
        let max_expiry = self
            .current_epoch
            .checked_add(self.max_lease_epochs)
            .ok_or_else(|| ResourceReservationError::new("expires_at_epoch", "lease horizon overflow"))?;
        if expires_at_epoch > max_expiry {
            return Err(ResourceReservationError::new(
                "expires_at_epoch",
                "lease exceeds the maximum predeclared reservation duration",
            ));
        }
        if self.reservations.len() >= MAX_RESERVATIONS {
            return Err(ResourceReservationError::new(
                "reservations",
                "reservation count exceeds the configured safety limit",
            ));
        }

        validate_cost_vector(&self.resource_registry, &proof.resource_cost)?;
        ensure_cost_fits_headroom(self, &proof.resource_cost)?;

        let mut candidate = self.clone();
        let resource_cost_sha256 = compute_cost_sha256(&proof.resource_cost)?;
        let mut record = ResourceReservationRecord {
            reservation_id: proof.scope_sha256.clone(),
            action_id: proof.preflight_receipt.action_id.clone(),
            preflight_receipt_sha256: proof.preflight_receipt.receipt_sha256.clone(),
            preflight_receipt: proof.preflight_receipt.clone(),
            resource_registry_sha256: self.resource_registry.registry_sha256.clone(),
            resource_budget_sha256: proof.resource_budget_sha256.clone(),
            resource_cost_sha256,
            resource_cost: proof.resource_cost.clone(),
            reserved_at_epoch: self.current_epoch,
            expires_at_epoch,
            state: ReservationState::Reserved,
            latest_event_sha256: String::new(),
            record_sha256: String::new(),
        };
        record.record_sha256 = compute_record_sha256(&record)?;
        candidate.reservations.push(record);
        let index = candidate.reservations.len() - 1;
        append_event(
            &mut candidate,
            index,
            ReservationState::Reserved,
            self.current_epoch,
            "reserve".to_owned(),
        )?;
        candidate.ledger_sha256 = compute_ledger_sha256(&candidate)?;
        verify_resource_reservation_ledger(&candidate)?;
        *self = candidate;

        let created = self
            .reservations
            .last()
            .ok_or_else(|| ResourceReservationError::new("reservations", "new reservation missing"))?;
        self.receipt_for(created)
    }

    /// Commit is a one-way conservative budget debit. It must happen immediately before
    /// the external operation. Once committed, resource quantity is permanently counted
    /// even if the downstream process crashes or the physical action fails.
    pub fn commit_measurement(
        &mut self,
        expected_head_sha256: &str,
        reservation_id: &str,
    ) -> Result<ResourceReservationReceipt, ResourceReservationError> {
        verify_resource_reservation_ledger(self)?;
        let index = self.find_reservation(reservation_id)?;
        if self.reservations[index].state == ReservationState::Committed {
            return self.receipt_for(&self.reservations[index]);
        }
        self.check_expected_head(expected_head_sha256)?;
        let record = &self.reservations[index];
        if record.state != ReservationState::Reserved {
            return Err(ResourceReservationError::new(
                "state",
                "only an active RESERVED measurement can be committed",
            ));
        }
        if self.current_epoch >= record.expires_at_epoch {
            return Err(ResourceReservationError::new(
                "expires_at_epoch",
                "reservation lease has expired; advance the ledger epoch before retry",
            ));
        }

        let mut candidate = self.clone();
        append_event(
            &mut candidate,
            index,
            ReservationState::Committed,
            self.current_epoch,
            "commit".to_owned(),
        )?;
        candidate.ledger_sha256 = compute_ledger_sha256(&candidate)?;
        verify_resource_reservation_ledger(&candidate)?;
        *self = candidate;
        self.receipt_for(&self.reservations[index])
    }

    /// Release an unused reservation. A committed action is never refunded.
    pub fn release_measurement(
        &mut self,
        expected_head_sha256: &str,
        reservation_id: &str,
        reason_code: &str,
    ) -> Result<ResourceReservationReceipt, ResourceReservationError> {
        verify_resource_reservation_ledger(self)?;
        if reason_code.trim().is_empty() {
            return Err(ResourceReservationError::new(
                "reason_code",
                "release reason must be non-empty",
            ));
        }
        let index = self.find_reservation(reservation_id)?;
        if self.reservations[index].state == ReservationState::Released {
            return self.receipt_for(&self.reservations[index]);
        }
        self.check_expected_head(expected_head_sha256)?;
        let record = &self.reservations[index];
        if record.state != ReservationState::Reserved {
            return Err(ResourceReservationError::new(
                "state",
                "only an active RESERVED measurement can be released",
            ));
        }
        if self.current_epoch >= record.expires_at_epoch {
            return Err(ResourceReservationError::new(
                "expires_at_epoch",
                "expired lease must transition through EXPIRED, not manual release",
            ));
        }

        let mut candidate = self.clone();
        append_event(
            &mut candidate,
            index,
            ReservationState::Released,
            self.current_epoch,
            reason_code.to_owned(),
        )?;
        candidate.ledger_sha256 = compute_ledger_sha256(&candidate)?;
        verify_resource_reservation_ledger(&candidate)?;
        *self = candidate;
        self.receipt_for(&self.reservations[index])
    }

    /// Advance a caller-supplied monotone logical epoch and expire every due RESERVED
    /// lease. The caller/production adapter must supply a trusted monotonic clock.
    pub fn advance_epoch(
        &mut self,
        expected_head_sha256: &str,
        new_epoch: u64,
    ) -> Result<(), ResourceReservationError> {
        verify_resource_reservation_ledger(self)?;
        self.check_expected_head(expected_head_sha256)?;
        if new_epoch <= self.current_epoch {
            return Err(ResourceReservationError::new(
                "new_epoch",
                "logical epoch must advance strictly monotonically",
            ));
        }

        let mut candidate = self.clone();
        candidate.current_epoch = new_epoch;
        let expired_indices: Vec<usize> = candidate
            .reservations
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                (record.state == ReservationState::Reserved
                    && record.expires_at_epoch <= new_epoch)
                    .then_some(index)
            })
            .collect();
        for index in expired_indices {
            append_event(
                &mut candidate,
                index,
                ReservationState::Expired,
                new_epoch,
                "lease_expired".to_owned(),
            )?;
        }
        candidate.ledger_sha256 = compute_ledger_sha256(&candidate)?;
        verify_resource_reservation_ledger(&candidate)?;
        *self = candidate;
        Ok(())
    }

    pub fn verify_integrity(&self) -> Result<(), ResourceReservationError> {
        verify_resource_reservation_ledger(self)
    }

    fn check_expected_head(&self, expected_head_sha256: &str) -> Result<(), ResourceReservationError> {
        if expected_head_sha256 != self.ledger_sha256 {
            return Err(ResourceReservationError::new(
                "expected_head_sha256",
                "stale reservation-ledger revision; refresh and re-run preflight",
            ));
        }
        Ok(())
    }

    fn find_reservation(&self, reservation_id: &str) -> Result<usize, ResourceReservationError> {
        self.reservations
            .iter()
            .position(|record| record.reservation_id == reservation_id)
            .ok_or_else(|| {
                ResourceReservationError::new(
                    "reservation_id",
                    "unknown idempotency key / action scope digest",
                )
            })
    }

    fn receipt_for(
        &self,
        record: &ResourceReservationRecord,
    ) -> Result<ResourceReservationReceipt, ResourceReservationError> {
        let mut receipt = ResourceReservationReceipt {
            schema: "symthaea.nas.resource-reservation-receipt".to_owned(),
            version: "0.1.0".to_owned(),
            run_id: self.run_id.clone(),
            reservation_id: record.reservation_id.clone(),
            action_id: record.action_id.clone(),
            state: record.state,
            preflight_receipt_sha256: record.preflight_receipt_sha256.clone(),
            resource_registry_sha256: record.resource_registry_sha256.clone(),
            resource_budget_sha256: record.resource_budget_sha256.clone(),
            resource_cost_sha256: record.resource_cost_sha256.clone(),
            reserved_at_epoch: record.reserved_at_epoch,
            expires_at_epoch: record.expires_at_epoch,
            latest_event_sha256: record.latest_event_sha256.clone(),
            ledger_head_sha256: self.ledger_sha256.clone(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = compute_reservation_receipt_sha256(&receipt)?;
        Ok(receipt)
    }
}

/// Recompute and verify a reservation receipt's own canonical digest.
/// This verifies integrity only; callers should also verify the full ledger.
pub fn verify_resource_reservation_receipt(
    receipt: &ResourceReservationReceipt,
) -> Result<(), ResourceReservationError> {
    if !is_sha256(&receipt.receipt_sha256)
        || compute_reservation_receipt_sha256(receipt)? != receipt.receipt_sha256
    {
        return Err(ResourceReservationError::new(
            "receipt_sha256",
            "reservation receipt digest does not match canonical receipt fields",
        ));
    }
    Ok(())
}

/// Replay the event history, reconstruct states and resource use, and verify the full ledger.
pub fn verify_resource_reservation_ledger(
    ledger: &ResourceReservationLedger,
) -> Result<(), ResourceReservationError> {
    if ledger.schema != LEDGER_SCHEMA || ledger.version != LEDGER_VERSION {
        return Err(ResourceReservationError::new(
            "schema_version",
            "unsupported resource reservation ledger schema/version",
        ));
    }
    if ledger.run_id.trim().is_empty() || ledger.max_lease_epochs == 0 {
        return Err(ResourceReservationError::new(
            "genesis",
            "run ID and maximum lease duration must be valid",
        ));
    }
    validate_resource_registry(&ledger.resource_registry)
        .map_err(|error| ResourceReservationError::new("resource_registry", error.to_string()))?;
    remaining_resource_headroom(&ledger.resource_registry, &ledger.initial_budget)
        .map_err(|error| ResourceReservationError::new("initial_budget", error.to_string()))?;
    let budget_sha = compute_budget_sha256(&ledger.initial_budget)?;
    if compute_genesis_sha256(
        &ledger.run_id,
        &ledger.resource_registry,
        &ledger.initial_budget,
        ledger.max_lease_epochs,
    )? != ledger.genesis_sha256
    {
        return Err(ResourceReservationError::new(
            "genesis_sha256",
            "run, registry, budget, or lease config differs from ledger genesis",
        ));
    }

    let mut records = HashMap::with_capacity(ledger.reservations.len());
    for (index, record) in ledger.reservations.iter().enumerate() {
        verify_action_preflight_receipt_digest(&record.preflight_receipt)
            .map_err(|error| ResourceReservationError::new(
                format!("reservations[{index}].preflight_receipt"),
                error.to_string(),
            ))?;
        if record.preflight_receipt.receipt_sha256 != record.preflight_receipt_sha256
            || record.preflight_receipt.scope_sha256 != record.reservation_id
            || record.preflight_receipt.run_id != ledger.run_id
            || record.preflight_receipt.action_id != record.action_id
            || record.preflight_receipt.action_kind != NasActionKind::AcquireMeasurement
            || record.preflight_receipt.disposition != "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION"
            || record.preflight_receipt.resource_registry_sha256.as_deref()
                != Some(ledger.resource_registry.registry_sha256.as_str())
            || record.preflight_receipt.resource_budget_sha256.as_deref()
                != Some(budget_sha.as_str())
            || record.preflight_receipt.resource_cost_sha256.as_deref()
                != Some(record.resource_cost_sha256.as_str())
        {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}].preflight_receipt"),
                "stored preflight receipt does not bind the reservation run, action, scope, registry, budget, and cost",
            ));
        }
        if !is_sha256(&record.reservation_id)
            || !is_sha256(&record.preflight_receipt_sha256)
            || record.action_id.trim().is_empty()
            || record.resource_registry_sha256 != ledger.resource_registry.registry_sha256
            || record.resource_budget_sha256 != budget_sha
            || record.resource_cost_sha256 != compute_cost_sha256(&record.resource_cost)?
            || record.record_sha256 != compute_record_sha256(record)?
        {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}]"),
                "reservation record identity, cost, or digest is invalid",
            ));
        }
        validate_cost_vector(&ledger.resource_registry, &record.resource_cost)?;
        if record.resource_cost.action_id != record.action_id
            || record.expires_at_epoch <= record.reserved_at_epoch
            || record.expires_at_epoch
                > record.reserved_at_epoch.saturating_add(ledger.max_lease_epochs)
        {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}]"),
                "action cost identity or lease interval is invalid",
            ));
        }
        if records.insert(record.reservation_id.as_str(), (index, record)).is_some() {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}].reservation_id"),
                "duplicate idempotency key",
            ));
        }
    }
    if ledger.reservations.len() > MAX_RESERVATIONS {
        return Err(ResourceReservationError::new(
            "reservations",
            "reservation count exceeds configured maximum",
        ));
    }

    let mut replay: HashMap<String, (ReservationState, String, u64, u64)> = HashMap::new();
    let mut previous_event = ledger.genesis_sha256.clone();
    let mut previous_epoch = 0u64;
    for (index, event) in ledger.events.iter().enumerate() {
        if event.sequence != index as u64
            || event.previous_event_sha256 != previous_event
            || event.epoch < previous_epoch
            || event.epoch > ledger.current_epoch
            || event.event_sha256 != compute_event_sha256(event)?
        {
            return Err(ResourceReservationError::new(
                format!("events[{index}]"),
                "event sequence, epoch, prior hash, or event digest is invalid",
            ));
        }
        let (_, record) = records.get(event.reservation_id.as_str()).ok_or_else(|| {
            ResourceReservationError::new(
                format!("events[{index}].reservation_id"),
                "event refers to an unknown reservation",
            )
        })?;
        if event.action_id != record.action_id
            || event.preflight_receipt_sha256 != record.preflight_receipt_sha256
            || event.resource_cost_sha256 != record.resource_cost_sha256
            || event.expires_at_epoch != record.expires_at_epoch
        {
            return Err(ResourceReservationError::new(
                format!("events[{index}]"),
                "event identity differs from immutable reservation data",
            ));
        }

        match replay.get(&event.reservation_id) {
            None => {
                if event.kind != ReservationEventKind::Reserved
                    || event.epoch != record.reserved_at_epoch
                    || event.reason_code != "reserve"
                {
                    return Err(ResourceReservationError::new(
                        format!("events[{index}]"),
                        "first event for a reservation must be RESERVED at its creation epoch",
                    ));
                }
                replay.insert(
                    event.reservation_id.clone(),
                    (
                        ReservationState::Reserved,
                        event.event_sha256.clone(),
                        record.reserved_at_epoch,
                        record.expires_at_epoch,
                    ),
                );
            }
            Some((prior_state, _, _, expires_at)) => {
                if *prior_state != ReservationState::Reserved {
                    return Err(ResourceReservationError::new(
                        format!("events[{index}]"),
                        "terminal reservation state cannot transition again",
                    ));
                }
                let next_state = match event.kind {
                    ReservationEventKind::Committed if event.epoch < *expires_at => {
                        if event.reason_code != "commit" {
                            return Err(ResourceReservationError::new(
                                format!("events[{index}].reason_code"),
                                "COMMITTED event must use the commit reason",
                            ));
                        }
                        ReservationState::Committed
                    }
                    ReservationEventKind::Released if event.epoch < *expires_at => {
                        if event.reason_code.trim().is_empty() || event.reason_code == "reserve" {
                            return Err(ResourceReservationError::new(
                                format!("events[{index}].reason_code"),
                                "RELEASED event requires a non-empty explicit reason",
                            ));
                        }
                        ReservationState::Released
                    }
                    ReservationEventKind::Expired if event.epoch >= *expires_at => {
                        if event.reason_code != "lease_expired" {
                            return Err(ResourceReservationError::new(
                                format!("events[{index}].reason_code"),
                                "EXPIRED event must use the lease_expired reason",
                            ));
                        }
                        ReservationState::Expired
                    }
                    _ => {
                        return Err(ResourceReservationError::new(
                            format!("events[{index}].kind"),
                            "invalid reservation state transition or lease timing",
                        ));
                    }
                };
                replay.insert(
                    event.reservation_id.clone(),
                    (
                        next_state,
                        event.event_sha256.clone(),
                        record.reserved_at_epoch,
                        record.expires_at_epoch,
                    ),
                );
            }
        }

        previous_event = event.event_sha256.clone();
        previous_epoch = event.epoch;
    }

    if replay.len() != ledger.reservations.len() {
        return Err(ResourceReservationError::new(
            "events",
            "every reservation must have exactly one initial RESERVED event",
        ));
    }
    for (index, record) in ledger.reservations.iter().enumerate() {
        let (state, latest_event, _, _) = replay.get(&record.reservation_id).ok_or_else(|| {
            ResourceReservationError::new(
                format!("reservations[{index}]"),
                "reservation has no event history",
            )
        })?;
        if *state != record.state || latest_event != &record.latest_event_sha256 {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}]"),
                "record state or latest-event pointer differs from replayed history",
            ));
        }
        if record.state == ReservationState::Reserved
            && ledger.current_epoch >= record.expires_at_epoch
        {
            return Err(ResourceReservationError::new(
                format!("reservations[{index}].state"),
                "a reservation whose lease has elapsed must be transitioned to EXPIRED",
            ));
        }
    }

    let base_headroom = remaining_resource_headroom(
        &ledger.resource_registry,
        &ledger.initial_budget,
    )
    .map_err(|error| ResourceReservationError::new("initial_budget", error.to_string()))?;
    let mut charged = vec![0u64; base_headroom.len()];
    for record in &ledger.reservations {
        if matches!(record.state, ReservationState::Reserved | ReservationState::Committed) {
            for (index, quantity) in record.resource_cost.resources.iter().enumerate() {
                charged[index] = charged[index]
                    .checked_add(quantity.units)
                    .ok_or_else(|| {
                        ResourceReservationError::new(
                            "resource_accounting",
                            "reservation total overflowed u64",
                        )
                    })?;
            }
        }
    }
    for (index, used) in charged.iter().enumerate() {
        if *used > base_headroom[index].available_units {
            return Err(ResourceReservationError::new(
                format!("resource_accounting[{index}]"),
                "active and committed reservations exceed unprotected initial headroom",
            ));
        }
    }

    if ledger.ledger_sha256 != compute_ledger_sha256(ledger)? {
        return Err(ResourceReservationError::new(
            "ledger_sha256",
            "ledger head does not match the canonical ledger contents",
        ));
    }
    Ok(())
}

fn validate_measurement_proof(
    ledger: &ResourceReservationLedger,
    proof: &VerifiedMeasurementReservation,
) -> Result<(), ResourceReservationError> {
    verify_action_preflight_receipt_digest(&proof.preflight_receipt)
        .map_err(|error| ResourceReservationError::new("preflight_receipt", error.to_string()))?;
    let receipt = &proof.preflight_receipt;
    if receipt.action_kind != NasActionKind::AcquireMeasurement
        || receipt.disposition != "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION"
    {
        return Err(ResourceReservationError::new(
            "preflight_receipt",
            "only an eligible measurement preflight may reserve resources",
        ));
    }
    if receipt.run_id != ledger.run_id
        || proof.resource_cost.action_id != receipt.action_id
        || receipt.scope_sha256 != proof.scope_sha256
        || receipt.resource_registry_sha256.as_deref()
            != Some(ledger.resource_registry.registry_sha256.as_str())
        || receipt.resource_budget_sha256.as_deref()
            != Some(proof.resource_budget_sha256.as_str())
        || receipt.resource_cost_sha256.as_deref()
            != Some(compute_cost_sha256(&proof.resource_cost)?.as_str())
        || proof.resource_registry_sha256 != ledger.resource_registry.registry_sha256
    {
        return Err(ResourceReservationError::new(
            "preflight_scope",
            "preflight receipt does not bind this run, scope, registry, budget, and cost vector",
        ));
    }
    if compute_budget_sha256(&ledger.initial_budget)? != proof.resource_budget_sha256 {
        return Err(ResourceReservationError::new(
            "resource_budget_sha256",
            "preflight budget does not match the reservation ledger's frozen initial budget",
        ));
    }
    validate_cost_vector(&ledger.resource_registry, &proof.resource_cost)
}

fn validate_cost_vector(
    registry: &ResourceRegistry,
    cost: &ResourceCostVector,
) -> Result<(), ResourceReservationError> {
    if cost.action_id.trim().is_empty()
        || cost.resource_registry_sha256 != registry.registry_sha256
        || cost.resources.len() != registry.resources.len()
    {
        return Err(ResourceReservationError::new(
            "resource_cost",
            "cost vector must bind the registry and have one explicit coordinate per resource",
        ));
    }
    let mut any_positive = false;
    for (index, (quantity, definition)) in cost
        .resources
        .iter()
        .zip(&registry.resources)
        .enumerate()
    {
        if quantity.resource_id != definition.resource_id
            || quantity.unit != definition.unit
            || quantity.cost_model_sha256 != definition.cost_model_sha256
        {
            return Err(ResourceReservationError::new(
                format!("resource_cost.resources[{index}]"),
                "resource ID, unit, coordinate order, or cost-model digest differs from registry",
            ));
        }
        any_positive |= quantity.units > 0;
    }
    if !any_positive {
        return Err(ResourceReservationError::new(
            "resource_cost",
            "measurement action must reserve at least one positive resource quantity",
        ));
    }
    Ok(())
}

fn ensure_cost_fits_headroom(
    ledger: &ResourceReservationLedger,
    cost: &ResourceCostVector,
) -> Result<(), ResourceReservationError> {
    let headroom = ledger.available_headroom()?;
    for (index, (quantity, remaining)) in cost.resources.iter().zip(&headroom).enumerate() {
        if quantity.units > remaining.available_units {
            return Err(ResourceReservationError::new(
                format!("resource_cost.resources[{index}]"),
                format!(
                    "requested {} {} exceeds spendable headroom {}",
                    quantity.units, quantity.unit, remaining.available_units
                ),
            ));
        }
    }
    Ok(())
}

fn append_event(
    ledger: &mut ResourceReservationLedger,
    reservation_index: usize,
    state: ReservationState,
    epoch: u64,
    reason_code: String,
) -> Result<(), ResourceReservationError> {
    let record = ledger
        .reservations
        .get(reservation_index)
        .ok_or_else(|| ResourceReservationError::new("reservation_index", "unknown record"))?;
    let kind = match state {
        ReservationState::Reserved => ReservationEventKind::Reserved,
        ReservationState::Committed => ReservationEventKind::Committed,
        ReservationState::Released => ReservationEventKind::Released,
        ReservationState::Expired => ReservationEventKind::Expired,
    };
    let previous_event_sha256 = ledger
        .events
        .last()
        .map(|event| event.event_sha256.clone())
        .unwrap_or_else(|| ledger.genesis_sha256.clone());
    let mut event = ResourceReservationEvent {
        sequence: ledger.events.len() as u64,
        kind,
        reservation_id: record.reservation_id.clone(),
        action_id: record.action_id.clone(),
        preflight_receipt_sha256: record.preflight_receipt_sha256.clone(),
        resource_cost_sha256: record.resource_cost_sha256.clone(),
        epoch,
        expires_at_epoch: record.expires_at_epoch,
        reason_code,
        previous_event_sha256,
        event_sha256: String::new(),
    };
    event.event_sha256 = compute_event_sha256(&event)?;
    let event_sha256 = event.event_sha256.clone();
    ledger.events.push(event);
    let record = &mut ledger.reservations[reservation_index];
    record.state = state;
    record.latest_event_sha256 = event_sha256;
    record.record_sha256 = compute_record_sha256(record)?;
    Ok(())
}

fn compute_genesis_sha256(
    run_id: &str,
    registry: &ResourceRegistry,
    budget: &ResourceBudget,
    max_lease_epochs: u64,
) -> Result<String, ResourceReservationError> {
    let registry_sha = &registry.registry_sha256;
    let budget_sha = compute_budget_sha256(budget)?;
    digest_serde(
        GENESIS_DOMAIN,
        &(run_id, registry_sha, budget_sha, max_lease_epochs),
    )
}

fn compute_budget_sha256(budget: &ResourceBudget) -> Result<String, ResourceReservationError> {
    digest_serde(BUDGET_DOMAIN, budget)
}

fn compute_cost_sha256(cost: &ResourceCostVector) -> Result<String, ResourceReservationError> {
    digest_serde(COST_DOMAIN, cost)
}

fn compute_record_sha256(
    record: &ResourceReservationRecord,
) -> Result<String, ResourceReservationError> {
    let mut canonical = record.clone();
    canonical.record_sha256.clear();
    digest_serde(RECORD_DOMAIN, &canonical)
}

fn compute_event_sha256(
    event: &ResourceReservationEvent,
) -> Result<String, ResourceReservationError> {
    let mut canonical = event.clone();
    canonical.event_sha256.clear();
    digest_serde(EVENT_DOMAIN, &canonical)
}

fn compute_ledger_sha256(
    ledger: &ResourceReservationLedger,
) -> Result<String, ResourceReservationError> {
    let mut canonical = ledger.clone();
    canonical.ledger_sha256.clear();
    digest_serde(LEDGER_DOMAIN, &canonical)
}

fn compute_reservation_receipt_sha256(
    receipt: &ResourceReservationReceipt,
) -> Result<String, ResourceReservationError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    digest_serde(RECEIPT_DOMAIN, &canonical)
}

fn digest_serde<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, ResourceReservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ResourceReservationError::new("serialization", error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    Ok(encode_sha256(hasher.finalize().as_slice()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_resource_budget::{
        compute_resource_registry_sha256, ResourceBudgetLimit, ResourceClass,
        ResourceDefinition, ResourceQuantity,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture() -> (ResourceRegistry, ResourceBudget, ResourceCostVector) {
        let mut registry = ResourceRegistry {
            schema: "symthaea.nas.resource-registry".into(),
            version: "0.1.0".into(),
            resources: vec![
                ResourceDefinition {
                    resource_id: "device_ms".into(),
                    unit: "ms".into(),
                    class: ResourceClass::TargetDeviceTime,
                    cost_model_sha256: sha('a'),
                },
                ResourceDefinition {
                    resource_id: "instrument_ms".into(),
                    unit: "ms".into(),
                    class: ResourceClass::InstrumentTime,
                    cost_model_sha256: sha('b'),
                },
            ],
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = compute_resource_registry_sha256(&registry).unwrap();
        let budget = ResourceBudget {
            resource_registry_sha256: registry.registry_sha256.clone(),
            limits: vec![
                ResourceBudgetLimit {
                    resource_id: "device_ms".into(),
                    unit: "ms".into(),
                    total_units: 100,
                    already_spent_units: 10,
                    protected_reserve_units: 20,
                },
                ResourceBudgetLimit {
                    resource_id: "instrument_ms".into(),
                    unit: "ms".into(),
                    total_units: 1_000,
                    already_spent_units: 100,
                    protected_reserve_units: 100,
                },
            ],
        };
        let cost = ResourceCostVector {
            action_id: "measure-001".into(),
            resource_registry_sha256: registry.registry_sha256.clone(),
            resources: vec![
                ResourceQuantity {
                    resource_id: "device_ms".into(),
                    unit: "ms".into(),
                    units: 30,
                    cost_model_sha256: sha('a'),
                },
                ResourceQuantity {
                    resource_id: "instrument_ms".into(),
                    unit: "ms".into(),
                    units: 300,
                    cost_model_sha256: sha('b'),
                },
            ],
        };
        (registry, budget, cost)
    }

    fn proof(
        ledger: &ResourceReservationLedger,
        action_id: &str,
        scope_sha256: &str,
        cost: ResourceCostVector,
        receipt_char: char,
    ) -> VerifiedMeasurementReservation {
        let cost_sha = compute_cost_sha256(&cost).unwrap();
        let mut receipt = NasActionPreflightReceipt {
            schema: "symthaea.nas.action-preflight".into(),
            version: "0.1.0".into(),
            action_kind: NasActionKind::AcquireMeasurement,
            disposition: "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION".into(),
            run_id: ledger.run_id.clone(),
            target_commit_id: std::iter::repeat_n(receipt_char, 40).collect(),
            action_id: action_id.into(),
            candidate_id: "candidate-A".into(),
            scope_sha256: scope_sha256.into(),
            confidence_family_manifest_sha256: None,
            confidence_ledger_sha256: None,
            confidence_receipt_sha256: None,
            portfolio_decision_sha256: None,
            objective_registry_sha256: None,
            hardware_profile_sha256: None,
            resource_registry_sha256: Some(ledger.resource_registry.registry_sha256.clone()),
            resource_budget_sha256: Some(compute_budget_sha256(&ledger.initial_budget).unwrap()),
            resource_cost_sha256: Some(cost_sha),
            resource_frontier_sha256: None,
            pruning_risk_genesis_sha256: None,
            pruning_risk_receipt_sha256: None,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = compute_preflight_receipt_sha256(&receipt).unwrap();
        VerifiedMeasurementReservation {
            preflight_receipt: receipt,
            scope_sha256: scope_sha256.into(),
            reservation_ledger_head_sha256: ledger.head_sha256().into(),
            resource_registry_sha256: ledger.resource_registry.registry_sha256.clone(),
            resource_budget_sha256: compute_budget_sha256(&ledger.initial_budget).unwrap(),
            resource_cost: cost,
        }
    }

    fn ledger() -> ResourceReservationLedger {
        let (registry, budget, _) = fixture();
        ResourceReservationLedger::new("run-reserve-001", registry, budget, 10).unwrap()
    }

    #[test]
    fn genesis_and_empty_ledger_verify() {
        let ledger = ledger();
        ledger.verify_integrity().unwrap();
        assert_eq!(ledger.current_epoch(), 0);
        assert!(ledger.available_headroom().unwrap().iter().all(|r| r.available_units > 0));
    }

    #[test]
    fn reserves_against_exact_head_and_charges_all_dimensions() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        let head = ledger.head_sha256().to_owned();
        let receipt = ledger.reserve_measurement(&head, proof, 5).unwrap();
        assert_eq!(receipt.state, ReservationState::Reserved);
        let headroom = ledger.available_headroom().unwrap();
        assert_eq!(headroom[0].available_units, 40);
        assert_eq!(headroom[1].available_units, 500);
        ledger.verify_integrity().unwrap();
    }

    #[test]
    fn stale_compare_and_swap_allows_only_one_distinct_reservation() {
        let mut ledger = ledger();
        let (_, _, first_cost) = fixture();
        let (_, _, mut second_cost) = fixture();
        second_cost.action_id = "measure-002".into();
        let original_head = ledger.head_sha256().to_owned();
        let first = proof(&ledger, "measure-001", &sha('c'), first_cost, 'd');
        let second = proof(&ledger, "measure-002", &sha('e'), second_cost, 'f');
        ledger.reserve_measurement(&original_head, first, 5).unwrap();
        assert!(ledger.reserve_measurement(&original_head, second, 5).is_err());
        assert_eq!(ledger.reservations().len(), 1);
    }

    #[test]
    fn same_scope_retry_is_idempotent_and_does_not_append_event() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let scope = sha('c');
        let proof = proof(&ledger, "measure-001", &scope, cost, 'd');
        let head = ledger.head_sha256().to_owned();
        let first = ledger.reserve_measurement(&head, proof, 5).unwrap();
        let events_before = ledger.events().len();
        // Retry is issued from the same preflight capability and original expected head.
        let (_, _, cost_again) = fixture();
        let mut retry_proof = proof(&ledger, "measure-001", &scope, cost_again, 'd');
        retry_proof.reservation_ledger_head_sha256 = head.clone();
        let retry = ledger
            .reserve_measurement(&head, retry_proof, 5)
            .unwrap();
        assert_eq!(first.reservation_id, retry.reservation_id);
        assert_eq!(first.receipt_sha256, retry.receipt_sha256);
        assert_eq!(ledger.events().len(), events_before);
        let (_, _, cost_changed_lease) = fixture();
        let mut conflicting_retry = proof(&ledger, "measure-001", &scope, cost_changed_lease, 'd');
        conflicting_retry.reservation_ledger_head_sha256 = head.clone();
        assert!(ledger
            .reserve_measurement(&head, conflicting_retry, 6)
            .is_err());
        assert_eq!(ledger.events().len(), events_before);
    }

    #[test]
    fn changed_cost_on_same_scope_fails_idempotency() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let head = ledger.head_sha256().to_owned();
        let first = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        ledger.reserve_measurement(&head, first, 5).unwrap();
        let (_, _, mut changed) = fixture();
        changed.resources[0].units += 1;
        let retry = proof(&ledger, "measure-001", &sha('c'), changed, 'e');
        assert!(ledger.reserve_measurement(&head, retry, 5).is_err());
    }

    #[test]
    fn rejects_budget_overcommit_and_protected_reserve_breach() {
        let mut ledger = ledger();
        let (_, _, mut cost) = fixture();
        cost.resources[0].units = 71;
        let proof = proof(&ledger, "measure-too-large", &sha('c'), cost, 'd');
        assert!(ledger.reserve_measurement(ledger.head_sha256(), proof, 5).is_err());
        assert!(ledger.reservations().is_empty());
    }

    #[test]
    fn committed_reservations_are_never_refunded() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        let reservation = ledger
            .reserve_measurement(ledger.head_sha256(), proof, 5)
            .unwrap();
        let committed = ledger
            .commit_measurement(ledger.head_sha256(), &reservation.reservation_id)
            .unwrap();
        assert_eq!(committed.state, ReservationState::Committed);
        assert!(ledger
            .release_measurement(ledger.head_sha256(), &reservation.reservation_id, "unused")
            .is_err());
        assert_eq!(ledger.available_headroom().unwrap()[0].available_units, 40);
    }

    #[test]
    fn release_before_commit_frees_capacity() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        let reservation = ledger
            .reserve_measurement(ledger.head_sha256(), proof, 5)
            .unwrap();
        ledger
            .release_measurement(ledger.head_sha256(), &reservation.reservation_id, "cancelled")
            .unwrap();
        assert_eq!(ledger.available_headroom().unwrap()[0].available_units, 70);
    }

    #[test]
    fn expiry_is_monotone_and_reclaims_only_uncommitted_reservations() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        ledger
            .reserve_measurement(ledger.head_sha256(), proof, 3)
            .unwrap();
        ledger.advance_epoch(ledger.head_sha256(), 2).unwrap();
        assert_eq!(ledger.available_headroom().unwrap()[0].available_units, 40);
        ledger.advance_epoch(ledger.head_sha256(), 3).unwrap();
        assert_eq!(ledger.available_headroom().unwrap()[0].available_units, 70);
        assert_eq!(ledger.reservations()[0].state, ReservationState::Expired);
        assert!(ledger.advance_epoch(ledger.head_sha256(), 2).is_err());
    }

    #[test]
    fn rejects_commit_after_expiry_until_expiry_transition_is_recorded() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        let reservation = ledger
            .reserve_measurement(ledger.head_sha256(), proof, 2)
            .unwrap();
        assert!(ledger
            .advance_epoch(ledger.head_sha256(), 2)
            .is_ok());
        assert!(ledger
            .commit_measurement(ledger.head_sha256(), &reservation.reservation_id)
            .is_err());
    }

    #[test]
    fn rejects_registry_unit_cost_model_and_budget_identity_mismatch() {
        let mut ledger = ledger();
        let (_, _, mut cost) = fixture();
        cost.resources[0].unit = "seconds".into();
        let bad_unit = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        assert!(ledger.reserve_measurement(ledger.head_sha256(), bad_unit, 5).is_err());

        let (_, _, mut cost2) = fixture();
        cost2.resources[1].cost_model_sha256 = sha('f');
        let bad_model = proof(&ledger, "measure-002", &sha('e'), cost2, 'f');
        assert!(ledger.reserve_measurement(ledger.head_sha256(), bad_model, 5).is_err());
    }


    #[test]
    fn integrity_verifier_rejects_untransitioned_expired_reservation() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        ledger
            .reserve_measurement(ledger.head_sha256(), proof, 2)
            .unwrap();
        ledger.current_epoch = 2;
        ledger.ledger_sha256 = compute_ledger_sha256(&ledger).unwrap();
        assert!(ledger.verify_integrity().is_err());
    }

    #[test]
    fn detects_event_record_and_ledger_tampering() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        ledger
            .reserve_measurement(ledger.head_sha256(), proof, 5)
            .unwrap();
        let mut changed = ledger.clone();
        changed.events[0].reason_code = "forged".into();
        assert!(changed.verify_integrity().is_err());

        let mut changed_record = ledger.clone();
        changed_record.reservations[0].expires_at_epoch += 1;
        assert!(changed_record.verify_integrity().is_err());
    }

    #[test]
    fn json_round_trip_preserves_full_reservation_history() {
        let mut ledger = ledger();
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        ledger
            .reserve_measurement(ledger.head_sha256(), proof, 5)
            .unwrap();
        ledger
            .release_measurement(ledger.head_sha256(), &sha('c'), "cancelled")
            .unwrap();
        let encoded = serde_json::to_string(&ledger).unwrap();
        let decoded: ResourceReservationLedger = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, ledger);
        decoded.verify_integrity().unwrap();
    }

    #[test]
    fn rejects_unknown_fields_and_invalid_lease() {
        let mut ledger = ledger();
        let mut encoded = serde_json::to_value(&ledger).unwrap();
        encoded.as_object_mut().unwrap().insert(
            "unknown_unsafe_field".into(),
            serde_json::Value::Bool(true),
        );
        assert!(serde_json::from_value::<ResourceReservationLedger>(encoded).is_err());
        let (_, _, cost) = fixture();
        let proof = proof(&ledger, "measure-001", &sha('c'), cost, 'd');
        assert!(ledger.reserve_measurement(ledger.head_sha256(), proof, 11).is_err());
    }
}
