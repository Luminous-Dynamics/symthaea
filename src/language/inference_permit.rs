// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Process-local inference execution permit boundary.
//!
//! This module deliberately binds digests rather than duplicating IF-0 semantic
//! schemas. A future adapter will canonicalize the admitted request/route/policy
//! and provider state and supply their exact digests here.
//!
//! Permits are private-field, non-Clone, non-Serde values. Preparing execution
//! consumes the permit even when a freshness or TOCTOU recheck fails.

use std::collections::HashMap;
use std::fmt;

/// Non-zero 32-byte digest used to bind an exact external semantic artifact.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingDigest([u8; 32]);

impl BindingDigest {
    pub fn new(bytes: [u8; 32]) -> Result<Self, InferencePermitError> {
        if bytes == [0; 32] {
            return Err(InferencePermitError::ZeroDigest);
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for BindingDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BindingDigest({:02x}{:02x}{:02x}{:02x}…)", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// Exact state that must remain unchanged between route admission and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceExecutionBinding {
    pub request_digest: BindingDigest,
    pub route_digest: BindingDigest,
    pub policy_digest: BindingDigest,
    pub provider_state_digest: BindingDigest,
    pub credential_state_digest: BindingDigest,
    pub quota_state_digest: BindingDigest,
}

/// Caller-supplied unpredictable nonce. It is an identity/replay value, not a secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PermitNonce([u8; 32]);

impl PermitNonce {
    fn new(bytes: [u8; 32]) -> Result<Self, InferencePermitError> {
        if bytes == [0; 32] {
            return Err(InferencePermitError::ZeroNonce);
        }
        Ok(Self(bytes))
    }
}

/// Opaque process-local execution permit.
///
/// Intentionally not Clone, Copy, Serialize, or Deserialize.
pub struct InferencePermit {
    binding: InferenceExecutionBinding,
    generation: u64,
    nonce: PermitNonce,
    issued_at_tick: u64,
    expires_at_tick: u64,
}

impl fmt::Debug for InferencePermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferencePermit")
            .field("generation", &self.generation)
            .field("issued_at_tick", &self.issued_at_tick)
            .field("expires_at_tick", &self.expires_at_tick)
            .finish_non_exhaustive()
    }
}

/// Authority-bearing preparation result for a future executor wrapper.
///
/// Also non-Clone/non-Serde so production integration can require ownership transfer
/// directly into the external-call boundary.
pub struct PreparedInferenceExecution {
    binding: InferenceExecutionBinding,
    generation: u64,
    prepared_at_tick: u64,
}

impl fmt::Debug for PreparedInferenceExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedInferenceExecution")
            .field("generation", &self.generation)
            .field("prepared_at_tick", &self.prepared_at_tick)
            .finish_non_exhaustive()
    }
}

impl PreparedInferenceExecution {
    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn prepared_at_tick(&self) -> u64 {
        self.prepared_at_tick
    }
}

/// Process-local permit issuer/replay guard.
///
/// A durable/distributed implementation is explicitly future work. This object is
/// sufficient to establish the ownership, freshness, and TOCTOU semantics first.
#[derive(Debug, Default)]
pub struct InferencePermitIssuer {
    next_generation: u64,
    active: HashMap<PermitNonce, u64>,
}

impl InferencePermitIssuer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn issue(
        &mut self,
        binding: InferenceExecutionBinding,
        issued_at_tick: u64,
        ttl_ticks: u64,
        nonce: [u8; 32],
    ) -> Result<InferencePermit, InferencePermitError> {
        if ttl_ticks == 0 {
            return Err(InferencePermitError::ZeroTtl);
        }
        let expires_at_tick = issued_at_tick
            .checked_add(ttl_ticks)
            .ok_or(InferencePermitError::TickOverflow)?;
        let nonce = PermitNonce::new(nonce)?;
        if self.active.contains_key(&nonce) {
            return Err(InferencePermitError::DuplicateActiveNonce);
        }

        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(InferencePermitError::GenerationOverflow)?;
        self.next_generation = generation;
        self.active.insert(nonce, generation);

        Ok(InferencePermit {
            binding,
            generation,
            nonce,
            issued_at_tick,
            expires_at_tick,
        })
    }

    /// Consume one permit and recheck the exact current execution binding.
    ///
    /// The replay-guard entry is removed before semantic checks. Therefore a stale,
    /// expired, future-dated, or raced permit cannot be retried after state changes.
    pub fn prepare_execution(
        &mut self,
        permit: InferencePermit,
        current_binding: InferenceExecutionBinding,
        now_tick: u64,
    ) -> Result<PreparedInferenceExecution, InferencePermitError> {
        let active_generation = self
            .active
            .remove(&permit.nonce)
            .ok_or(InferencePermitError::InactiveOrConsumedPermit)?;
        if active_generation != permit.generation {
            return Err(InferencePermitError::GenerationMismatch);
        }
        if now_tick < permit.issued_at_tick {
            return Err(InferencePermitError::PermitFromFuture);
        }
        // Half-open validity interval [issued_at, expires_at).
        if now_tick >= permit.expires_at_tick {
            return Err(InferencePermitError::PermitExpired);
        }
        if current_binding != permit.binding {
            return Err(InferencePermitError::BindingChanged);
        }

        Ok(PreparedInferenceExecution {
            binding: permit.binding,
            generation: permit.generation,
            prepared_at_tick: now_tick,
        })
    }

    pub fn active_permit_count(&self) -> usize {
        self.active.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePermitError {
    ZeroDigest,
    ZeroNonce,
    ZeroTtl,
    TickOverflow,
    GenerationOverflow,
    DuplicateActiveNonce,
    InactiveOrConsumedPermit,
    GenerationMismatch,
    PermitFromFuture,
    PermitExpired,
    BindingChanged,
}

impl fmt::Display for InferencePermitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDigest => write!(f, "binding digest must be non-zero"),
            Self::ZeroNonce => write!(f, "permit nonce must be non-zero"),
            Self::ZeroTtl => write!(f, "permit TTL must be non-zero"),
            Self::TickOverflow => write!(f, "permit expiry tick overflow"),
            Self::GenerationOverflow => write!(f, "permit generation overflow"),
            Self::DuplicateActiveNonce => write!(f, "permit nonce is already active"),
            Self::InactiveOrConsumedPermit => write!(f, "permit is inactive or already consumed"),
            Self::GenerationMismatch => write!(f, "permit generation does not match replay guard"),
            Self::PermitFromFuture => write!(f, "permit issuance tick is in the future"),
            Self::PermitExpired => write!(f, "permit has expired"),
            Self::BindingChanged => write!(f, "inference execution binding changed after admission"),
        }
    }
}

impl std::error::Error for InferencePermitError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> BindingDigest {
        BindingDigest::new([byte; 32]).unwrap()
    }

    fn binding() -> InferenceExecutionBinding {
        InferenceExecutionBinding {
            request_digest: digest(1),
            route_digest: digest(2),
            policy_digest: digest(3),
            provider_state_digest: digest(4),
            credential_state_digest: digest(5),
            quota_state_digest: digest(6),
        }
    }

    #[test]
    fn exact_fresh_binding_prepares_once() {
        let mut issuer = InferencePermitIssuer::new();
        let permit = issuer.issue(binding(), 100, 10, [7; 32]).unwrap();
        assert_eq!(issuer.active_permit_count(), 1);

        let prepared = issuer.prepare_execution(permit, binding(), 105).unwrap();
        assert_eq!(prepared.binding(), &binding());
        assert_eq!(prepared.prepared_at_tick(), 105);
        assert_eq!(issuer.active_permit_count(), 0);
    }

    #[test]
    fn binding_race_fails_and_consumes_permit() {
        let mut issuer = InferencePermitIssuer::new();
        let permit = issuer.issue(binding(), 100, 10, [8; 32]).unwrap();
        let mut raced = binding();
        raced.provider_state_digest = digest(9);

        let err = issuer.prepare_execution(permit, raced, 101).unwrap_err();
        assert_eq!(err, InferencePermitError::BindingChanged);
        assert_eq!(issuer.active_permit_count(), 0);
    }

    #[test]
    fn expiry_is_half_open_and_consumes_permit() {
        let mut issuer = InferencePermitIssuer::new();
        let permit = issuer.issue(binding(), 100, 10, [9; 32]).unwrap();
        let err = issuer.prepare_execution(permit, binding(), 110).unwrap_err();
        assert_eq!(err, InferencePermitError::PermitExpired);
        assert_eq!(issuer.active_permit_count(), 0);
    }

    #[test]
    fn future_dated_use_fails_closed() {
        let mut issuer = InferencePermitIssuer::new();
        let permit = issuer.issue(binding(), 100, 10, [10; 32]).unwrap();
        let err = issuer.prepare_execution(permit, binding(), 99).unwrap_err();
        assert_eq!(err, InferencePermitError::PermitFromFuture);
        assert_eq!(issuer.active_permit_count(), 0);
    }

    #[test]
    fn duplicate_active_nonce_is_rejected() {
        let mut issuer = InferencePermitIssuer::new();
        let _permit = issuer.issue(binding(), 1, 10, [11; 32]).unwrap();
        let err = issuer.issue(binding(), 1, 10, [11; 32]).unwrap_err();
        assert_eq!(err, InferencePermitError::DuplicateActiveNonce);
    }

    #[test]
    fn zero_nonce_ttl_and_digest_are_rejected() {
        assert_eq!(
            BindingDigest::new([0; 32]).unwrap_err(),
            InferencePermitError::ZeroDigest
        );

        let mut issuer = InferencePermitIssuer::new();
        assert!(matches!(
            issuer.issue(binding(), 1, 0, [1; 32]),
            Err(InferencePermitError::ZeroTtl)
        ));
        assert!(matches!(
            issuer.issue(binding(), 1, 1, [0; 32]),
            Err(InferencePermitError::ZeroNonce)
        ));
    }

    #[test]
    fn generations_are_monotonic() {
        let mut issuer = InferencePermitIssuer::new();
        let first = issuer.issue(binding(), 1, 10, [12; 32]).unwrap();
        let second = issuer.issue(binding(), 1, 10, [13; 32]).unwrap();
        assert_eq!(first.generation + 1, second.generation);
    }
}
