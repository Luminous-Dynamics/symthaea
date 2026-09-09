// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-7 conservative inference resource/quota guard.
//!
//! Provider quota observations are evidence, not authority. They may tighten
//! locally authorized availability but can never replenish it within an epoch.
//! Only explicit local quota-epoch advancement can mint fresh request/token
//! authority.

#[cfg(not(test))]
use super::inference_receipt::InferenceWireEvidence;
#[cfg(test)]
use crate::inference_receipt::InferenceWireEvidence;

use std::collections::HashSet;
use std::fmt;

/// Locally authorized resource envelope for one provider/account scope.
///
/// `None` budgets mean the local deployment deliberately imposes no numeric cap;
/// they do not mean that an unknown provider quota has been proven unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceResourceAuthority {
    quota_epoch: u64,
    credential_epoch: u64,
    request_budget: Option<u64>,
    token_budget: Option<u64>,
}

impl InferenceResourceAuthority {
    pub const fn new(
        quota_epoch: u64,
        credential_epoch: u64,
        request_budget: Option<u64>,
        token_budget: Option<u64>,
    ) -> Self {
        Self {
            quota_epoch,
            credential_epoch,
            request_budget,
            token_budget,
        }
    }

    pub const fn quota_epoch(&self) -> u64 {
        self.quota_epoch
    }

    pub const fn credential_epoch(&self) -> u64 {
        self.credential_epoch
    }
}

/// Guard timing/backoff policy. All units are monotonic milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceResourceGuardConfig {
    pub default_rate_limit_backoff_millis: u64,
    pub failure_threshold: u32,
    pub failure_base_backoff_millis: u64,
    pub max_backoff_millis: u64,
}

impl Default for InferenceResourceGuardConfig {
    fn default() -> Self {
        Self {
            default_rate_limit_backoff_millis: 60_000,
            failure_threshold: 3,
            failure_base_backoff_millis: 1_000,
            max_backoff_millis: 15 * 60_000,
        }
    }
}

/// One local resource reservation. It is intentionally non-Clone/non-Serde.
/// Dropping it without settlement keeps the reservation charged conservatively.
/// Call `abandon()` when the caller knows the attempt will never be settled; that
/// retires the capability without refunding any budget.
pub struct InferenceResourceReservation {
    guard_instance_id: [u8; 32],
    generation: u64,
    quota_epoch: u64,
    credential_epoch: u64,
    reserved_tokens: u64,
}

impl fmt::Debug for InferenceResourceReservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceResourceReservation")
            .field("generation", &self.generation)
            .field("quota_epoch", &self.quota_epoch)
            .field("credential_epoch", &self.credential_epoch)
            .field("reserved_tokens", &self.reserved_tokens)
            .finish_non_exhaustive()
    }
}

/// Provider/account availability state owned by the local runtime.
#[derive(Debug)]
pub struct InferenceResourceGuard {
    authority: InferenceResourceAuthority,
    config: InferenceResourceGuardConfig,
    /// Caller-supplied non-secret unique identity for this in-memory guard instance.
    /// It prevents a reservation minted by another guard from settling this one.
    instance_id: [u8; 32],

    local_remaining_requests: Option<u64>,
    local_remaining_tokens: Option<u64>,

    /// Provider observations can only decrease these ceilings in one quota epoch.
    provider_remaining_requests_ceiling: Option<u64>,
    provider_remaining_tokens_ceiling: Option<u64>,

    /// Informational reset hint only. It does not mint quota.
    last_observed_reset_after_millis: Option<u64>,

    cooldown_until_millis: Option<u64>,
    credential_blocked_epoch: Option<u64>,
    consecutive_failures: u32,

    next_generation: u64,
    active_reservations: HashSet<u64>,
}

impl InferenceResourceGuard {
    pub fn new(
        authority: InferenceResourceAuthority,
        config: InferenceResourceGuardConfig,
        instance_id: [u8; 32],
    ) -> Result<Self, InferenceResourceGuardError> {
        validate_config(config)?;
        if instance_id == [0; 32] {
            return Err(InferenceResourceGuardError::InvalidInstanceId);
        }
        Ok(Self {
            authority,
            config,
            instance_id,
            local_remaining_requests: authority.request_budget,
            local_remaining_tokens: authority.token_budget,
            provider_remaining_requests_ceiling: None,
            provider_remaining_tokens_ceiling: None,
            last_observed_reset_after_millis: None,
            cooldown_until_millis: None,
            credential_blocked_epoch: None,
            consecutive_failures: 0,
            next_generation: 0,
            active_reservations: HashSet::new(),
        })
    }

    pub const fn authority(&self) -> InferenceResourceAuthority {
        self.authority
    }

    pub fn effective_remaining_requests(&self) -> Option<u64> {
        min_optional(
            self.local_remaining_requests,
            self.provider_remaining_requests_ceiling,
        )
    }

    pub fn effective_remaining_tokens(&self) -> Option<u64> {
        min_optional(
            self.local_remaining_tokens,
            self.provider_remaining_tokens_ceiling,
        )
    }

    pub const fn cooldown_until_millis(&self) -> Option<u64> {
        self.cooldown_until_millis
    }

    pub const fn last_observed_reset_after_millis(&self) -> Option<u64> {
        self.last_observed_reset_after_millis
    }

    pub const fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    pub fn active_reservation_count(&self) -> usize {
        self.active_reservations.len()
    }

    pub fn is_credential_blocked(&self) -> bool {
        self.credential_blocked_epoch == Some(self.authority.credential_epoch)
    }

    /// Reserve one request plus the conservative maximum token exposure before
    /// dispatch. This prevents concurrent callers from oversubscribing a shared
    /// provider scope.
    pub fn reserve(
        &mut self,
        reserved_tokens: u64,
        now_millis: u64,
    ) -> Result<InferenceResourceReservation, InferenceResourceGuardError> {
        if self.is_credential_blocked() {
            return Err(InferenceResourceGuardError::CredentialBlocked);
        }
        if self
            .cooldown_until_millis
            .is_some_and(|until| now_millis < until)
        {
            return Err(InferenceResourceGuardError::CoolingDown);
        }

        if self.effective_remaining_requests() == Some(0) {
            return Err(InferenceResourceGuardError::RequestBudgetExhausted);
        }
        if self
            .effective_remaining_tokens()
            .is_some_and(|remaining| remaining < reserved_tokens)
        {
            return Err(InferenceResourceGuardError::TokenBudgetExhausted);
        }

        decrement_optional(&mut self.local_remaining_requests, 1)?;
        decrement_optional(&mut self.local_remaining_tokens, reserved_tokens)?;
        decrement_optional(&mut self.provider_remaining_requests_ceiling, 1)?;
        decrement_optional(
            &mut self.provider_remaining_tokens_ceiling,
            reserved_tokens,
        )?;

        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(InferenceResourceGuardError::GenerationOverflow)?;
        self.next_generation = generation;
        self.active_reservations.insert(generation);

        Ok(InferenceResourceReservation {
            guard_instance_id: self.instance_id,
            generation,
            quota_epoch: self.authority.quota_epoch,
            credential_epoch: self.authority.credential_epoch,
            reserved_tokens,
        })
    }

    /// Retire a reservation that will never reach a provider settlement path.
    /// Budget remains charged; this operation only prevents an abandoned active
    /// capability from permanently blocking a later explicit epoch transition.
    pub fn abandon(
        &mut self,
        reservation: InferenceResourceReservation,
    ) -> Result<(), InferenceResourceGuardError> {
        self.consume_reservation(reservation)
    }

    /// Settle a successful provider attempt and conservatively assimilate wire
    /// quota observations. Success clears transient failure backoff, but it does
    /// not refund unused reserved tokens from provider-declared usage.
    pub fn settle_success(
        &mut self,
        reservation: InferenceResourceReservation,
        evidence: &InferenceWireEvidence,
    ) -> Result<(), InferenceResourceGuardError> {
        self.consume_reservation(reservation)?;
        self.assimilate_wire_evidence(evidence);
        self.consecutive_failures = 0;
        Ok(())
    }

    /// Settle a 429/rate-limit event. Provider Retry-After can lengthen a cooldown
    /// only up to the locally configured maximum; it can never mint fresh quota.
    pub fn settle_rate_limited(
        &mut self,
        reservation: InferenceResourceReservation,
        evidence: &InferenceWireEvidence,
        now_millis: u64,
    ) -> Result<(), InferenceResourceGuardError> {
        self.consume_reservation(reservation)?;
        self.assimilate_wire_evidence(evidence);
        let requested = evidence
            .rate_limits()
            .retry_after_millis
            .unwrap_or(self.config.default_rate_limit_backoff_millis);
        self.extend_cooldown(now_millis, requested);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        Ok(())
    }

    /// Settle an authentication/authorization rejection. The current credential
    /// epoch stays blocked until an explicit newer credential epoch is installed.
    pub fn settle_auth_rejected(
        &mut self,
        reservation: InferenceResourceReservation,
    ) -> Result<(), InferenceResourceGuardError> {
        self.consume_reservation(reservation)?;
        self.credential_blocked_epoch = Some(self.authority.credential_epoch);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        Ok(())
    }

    /// Settle a provider/server or transport failure. Repeated failures open a
    /// bounded local circuit breaker. No provider observation can bypass it.
    pub fn settle_failure(
        &mut self,
        reservation: InferenceResourceReservation,
        now_millis: u64,
    ) -> Result<(), InferenceResourceGuardError> {
        self.consume_reservation(reservation)?;
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures >= self.config.failure_threshold {
            let exponent = self
                .consecutive_failures
                .saturating_sub(self.config.failure_threshold)
                .min(31);
            let multiplier = 1u64.checked_shl(exponent).unwrap_or(u64::MAX);
            let requested = self
                .config
                .failure_base_backoff_millis
                .saturating_mul(multiplier);
            self.extend_cooldown(now_millis, requested);
        }
        Ok(())
    }

    /// Install a new locally authorized quota epoch. This is the only operation
    /// that replenishes request/token authority.
    pub fn advance_quota_epoch(
        &mut self,
        new_epoch: u64,
        request_budget: Option<u64>,
        token_budget: Option<u64>,
    ) -> Result<(), InferenceResourceGuardError> {
        if new_epoch <= self.authority.quota_epoch {
            return Err(InferenceResourceGuardError::QuotaEpochNotAdvanced);
        }
        if !self.active_reservations.is_empty() {
            return Err(InferenceResourceGuardError::ActiveReservationsPresent);
        }

        self.authority.quota_epoch = new_epoch;
        self.authority.request_budget = request_budget;
        self.authority.token_budget = token_budget;
        self.local_remaining_requests = request_budget;
        self.local_remaining_tokens = token_budget;
        self.provider_remaining_requests_ceiling = None;
        self.provider_remaining_tokens_ceiling = None;
        self.last_observed_reset_after_millis = None;
        self.cooldown_until_millis = None;
        self.consecutive_failures = 0;
        Ok(())
    }

    /// Install a newer credential epoch. This clears an authentication block but
    /// deliberately does not replenish request/token authority. Active attempts
    /// must be settled or explicitly abandoned first so old-credential work cannot
    /// be mistaken for work under the new credential epoch.
    pub fn advance_credential_epoch(
        &mut self,
        new_epoch: u64,
    ) -> Result<(), InferenceResourceGuardError> {
        if new_epoch <= self.authority.credential_epoch {
            return Err(InferenceResourceGuardError::CredentialEpochNotAdvanced);
        }
        if !self.active_reservations.is_empty() {
            return Err(InferenceResourceGuardError::ActiveReservationsPresent);
        }
        self.authority.credential_epoch = new_epoch;
        self.credential_blocked_epoch = None;
        Ok(())
    }

    fn consume_reservation(
        &mut self,
        reservation: InferenceResourceReservation,
    ) -> Result<(), InferenceResourceGuardError> {
        if reservation.guard_instance_id != self.instance_id {
            return Err(InferenceResourceGuardError::ReservationGuardMismatch);
        }
        if reservation.quota_epoch != self.authority.quota_epoch
            || reservation.credential_epoch != self.authority.credential_epoch
        {
            return Err(InferenceResourceGuardError::ReservationEpochMismatch);
        }
        if !self.active_reservations.remove(&reservation.generation) {
            return Err(InferenceResourceGuardError::InactiveReservation);
        }
        Ok(())
    }

    fn assimilate_wire_evidence(&mut self, evidence: &InferenceWireEvidence) {
        let limits = evidence.rate_limits();
        tighten_optional(
            &mut self.provider_remaining_requests_ceiling,
            limits.request_remaining,
        );
        tighten_optional(
            &mut self.provider_remaining_tokens_ceiling,
            limits.token_remaining,
        );

        // Reset is informational only. Keep the shortest observed horizon to avoid
        // presenting a more optimistic provider reset than previously observed.
        let reset = min_optional(
            limits.request_reset_after_millis,
            limits.token_reset_after_millis,
        );
        tighten_optional(&mut self.last_observed_reset_after_millis, reset);
    }

    fn extend_cooldown(&mut self, now_millis: u64, requested_millis: u64) {
        let bounded = requested_millis.min(self.config.max_backoff_millis);
        let until = now_millis.saturating_add(bounded);
        self.cooldown_until_millis = Some(
            self.cooldown_until_millis
                .map_or(until, |existing| existing.max(until)),
        );
    }
}

fn validate_config(config: InferenceResourceGuardConfig) -> Result<(), InferenceResourceGuardError> {
    if config.default_rate_limit_backoff_millis == 0
        || config.failure_threshold == 0
        || config.failure_base_backoff_millis == 0
        || config.max_backoff_millis == 0
    {
        return Err(InferenceResourceGuardError::InvalidConfig);
    }
    Ok(())
}

fn min_optional(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn tighten_optional(slot: &mut Option<u64>, observed: Option<u64>) {
    let Some(observed) = observed else {
        return;
    };
    *slot = Some(slot.map_or(observed, |existing| existing.min(observed)));
}

fn decrement_optional(
    slot: &mut Option<u64>,
    amount: u64,
) -> Result<(), InferenceResourceGuardError> {
    if let Some(value) = slot {
        *value = value
            .checked_sub(amount)
            .ok_or(InferenceResourceGuardError::ReservationUnderflow)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceResourceGuardError {
    InvalidConfig,
    InvalidInstanceId,
    CoolingDown,
    CredentialBlocked,
    RequestBudgetExhausted,
    TokenBudgetExhausted,
    ReservationUnderflow,
    GenerationOverflow,
    InactiveReservation,
    ReservationGuardMismatch,
    ReservationEpochMismatch,
    QuotaEpochNotAdvanced,
    CredentialEpochNotAdvanced,
    ActiveReservationsPresent,
}

impl fmt::Display for InferenceResourceGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "inference resource guard config is invalid"),
            Self::InvalidInstanceId => write!(f, "resource guard instance id must be non-zero"),
            Self::CoolingDown => write!(f, "provider scope is cooling down"),
            Self::CredentialBlocked => write!(f, "provider credential epoch is blocked"),
            Self::RequestBudgetExhausted => write!(f, "request budget is exhausted"),
            Self::TokenBudgetExhausted => write!(f, "token budget is exhausted"),
            Self::ReservationUnderflow => write!(f, "resource reservation would underflow budget"),
            Self::GenerationOverflow => write!(f, "resource reservation generation overflow"),
            Self::InactiveReservation => write!(f, "resource reservation is inactive or consumed"),
            Self::ReservationGuardMismatch => {
                write!(f, "resource reservation belongs to a different guard instance")
            }
            Self::ReservationEpochMismatch => {
                write!(f, "resource reservation belongs to a different authority epoch")
            }
            Self::QuotaEpochNotAdvanced => write!(f, "quota epoch must increase"),
            Self::CredentialEpochNotAdvanced => write!(f, "credential epoch must increase"),
            Self::ActiveReservationsPresent => {
                write!(f, "cannot advance authority epoch with active reservations")
            }
        }
    }
}

impl std::error::Error for InferenceResourceGuardError {}
