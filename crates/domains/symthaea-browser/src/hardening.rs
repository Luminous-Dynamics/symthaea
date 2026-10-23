// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded browser-execution primitives.
//!
//! These types deliberately separate browser policy from runtime admission.
//! A policy may allow an operation while the runtime still denies it because
//! an exact approval is missing, a budget has been exhausted, or the failure
//! circuit breaker is open.
//!
//! `BrowserApproval` is an exact, process-local approval token. It is not a
//! cryptographic credential and must not be represented as one. A later Xenia
//! integration can authenticate the same approval request/decision semantics.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::actions::BrowserAction;

const ACTION_DIGEST_DOMAIN: &[u8] = b"symthaea.browser.action.v1\0";
const TRACE_DIGEST_DOMAIN: &[u8] = b"symthaea.browser.trace.v1\0";

/// Coarse consequence class used for approval and budget semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserConsequence {
    /// Passive observation or viewport-only movement.
    Passive,
    /// Page or history navigation.
    Navigation,
    /// Activation of a page control.
    Interaction,
    /// Entry of caller-controlled text into page state.
    TextEntry,
}

impl BrowserConsequence {
    /// Whether current browser hardening requires an exact approval.
    pub const fn requires_exact_approval(self) -> bool {
        matches!(self, Self::Interaction | Self::TextEntry)
    }
}

/// Classify an action by its externally meaningful consequence.
pub fn consequence_of(action: &BrowserAction) -> BrowserConsequence {
    match action {
        BrowserAction::Click { .. } => BrowserConsequence::Interaction,
        BrowserAction::Type { .. } => BrowserConsequence::TextEntry,
        BrowserAction::Navigate { .. } | BrowserAction::GoBack | BrowserAction::GoForward => {
            BrowserConsequence::Navigation
        }
        BrowserAction::NoOp
        | BrowserAction::Screenshot
        | BrowserAction::ExtractText { .. }
        | BrowserAction::ScrollTo { .. } => BrowserConsequence::Passive,
    }
}

/// Stable semantic label that intentionally excludes page-controlled payloads.
pub fn action_kind(action: &BrowserAction) -> &'static str {
    match action {
        BrowserAction::Navigate { .. } => "navigate",
        BrowserAction::Click { .. } => "click",
        BrowserAction::Type { .. } => "type",
        BrowserAction::ScrollTo { .. } => "scroll-to",
        BrowserAction::GoBack => "go-back",
        BrowserAction::GoForward => "go-forward",
        BrowserAction::ExtractText { .. } => "extract-text",
        BrowserAction::Screenshot => "screenshot",
        BrowserAction::NoOp => "noop",
    }
}

/// Domain-separated digest of the exact serialized action.
///
/// Typed text and URLs may therefore affect the commitment without appearing
/// directly in a durable receipt or trace record.
pub fn action_digest(action: &BrowserAction) -> String {
    let encoded = serde_json::to_vec(action).expect("BrowserAction serialization must succeed");
    let mut hasher = blake3::Hasher::new();
    hasher.update(ACTION_DIGEST_DOMAIN);
    hasher.update(&encoded);
    hasher.finalize().to_hex().to_string()
}

/// One exact approval request emitted by the executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserApprovalRequest {
    pub execution_session_id: Uuid,
    pub action_digest: String,
    pub consequence: BrowserConsequence,
}

/// Exact, expiring approval for one browser action in one executor session.
///
/// This structure is deliberately not signed in the browser crate. Its current
/// purpose is to close lower-level approval bypasses and provide the semantic
/// object that a future Xenia/authority adapter can authenticate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserApproval {
    pub execution_session_id: Uuid,
    pub action_digest: String,
    pub consequence: BrowserConsequence,
    pub expires_at_unix_ms: u64,
}

impl BrowserApproval {
    /// Materialize an approval for a previously emitted exact request.
    pub fn approve(request: BrowserApprovalRequest, expires_at_unix_ms: u64) -> Self {
        Self {
            execution_session_id: request.execution_session_id,
            action_digest: request.action_digest,
            consequence: request.consequence,
            expires_at_unix_ms,
        }
    }

    pub(crate) fn validate(
        &self,
        request: &BrowserApprovalRequest,
        now_unix_ms: u64,
    ) -> Result<(), BrowserRuntimeDenial> {
        if self.expires_at_unix_ms < now_unix_ms {
            return Err(BrowserRuntimeDenial::ApprovalExpired);
        }
        if self.execution_session_id != request.execution_session_id {
            return Err(BrowserRuntimeDenial::ApprovalSessionMismatch);
        }
        if self.action_digest != request.action_digest {
            return Err(BrowserRuntimeDenial::ApprovalActionMismatch);
        }
        if self.consequence != request.consequence {
            return Err(BrowserRuntimeDenial::ApprovalConsequenceMismatch);
        }
        Ok(())
    }
}

/// Bounded autonomy envelope for one executor session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserRuntimeLimits {
    pub max_actions: u32,
    pub max_mutating_actions: u32,
    pub max_consecutive_failures: u32,
}

impl Default for BrowserRuntimeLimits {
    fn default() -> Self {
        Self {
            max_actions: 100,
            max_mutating_actions: 20,
            max_consecutive_failures: 3,
        }
    }
}

impl BrowserRuntimeLimits {
    pub fn validate(self) -> Result<Self, BrowserRuntimeDenial> {
        if self.max_actions == 0
            || self.max_mutating_actions > self.max_actions
            || self.max_consecutive_failures == 0
        {
            return Err(BrowserRuntimeDenial::InvalidRuntimeLimits);
        }
        Ok(self)
    }
}

/// Stable runtime-denial reasons independent of ordinary browser policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserRuntimeDenial {
    InvalidRuntimeLimits,
    ExactApprovalRequired,
    ApprovalExpired,
    ApprovalSessionMismatch,
    ApprovalActionMismatch,
    ApprovalConsequenceMismatch,
    TotalActionBudgetExhausted,
    MutatingActionBudgetExhausted,
    FailureCircuitOpen,
}

/// Read-only runtime accounting snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserRuntimeSnapshot {
    pub actions_reserved: u32,
    pub mutating_actions_reserved: u32,
    pub consecutive_failures: u32,
    pub circuit_open: bool,
}

#[derive(Debug)]
pub(crate) struct BrowserRuntimeState {
    limits: BrowserRuntimeLimits,
    actions_reserved: u32,
    mutating_actions_reserved: u32,
    consecutive_failures: u32,
    circuit_open: bool,
    previous_trace_hash: String,
}

impl BrowserRuntimeState {
    pub(crate) fn new(limits: BrowserRuntimeLimits) -> Result<Self, BrowserRuntimeDenial> {
        let limits = limits.validate()?;
        Ok(Self {
            limits,
            actions_reserved: 0,
            mutating_actions_reserved: 0,
            consecutive_failures: 0,
            circuit_open: false,
            previous_trace_hash: String::new(),
        })
    }

    /// Reserve budget before dispatch so crashes cannot make an in-flight
    /// mutation free from accounting.
    pub(crate) fn reserve(&mut self, action: &BrowserAction) -> Result<(), BrowserRuntimeDenial> {
        if self.circuit_open {
            return Err(BrowserRuntimeDenial::FailureCircuitOpen);
        }
        if self.actions_reserved >= self.limits.max_actions {
            return Err(BrowserRuntimeDenial::TotalActionBudgetExhausted);
        }
        let mutating = !action.is_read_only();
        if mutating && self.mutating_actions_reserved >= self.limits.max_mutating_actions {
            return Err(BrowserRuntimeDenial::MutatingActionBudgetExhausted);
        }

        self.actions_reserved = self.actions_reserved.saturating_add(1);
        if mutating {
            self.mutating_actions_reserved = self.mutating_actions_reserved.saturating_add(1);
        }
        Ok(())
    }

    pub(crate) fn record_result(&mut self, succeeded: bool) {
        if succeeded {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            if self.consecutive_failures >= self.limits.max_consecutive_failures {
                self.circuit_open = true;
            }
        }
    }

    pub(crate) fn snapshot(&self) -> BrowserRuntimeSnapshot {
        BrowserRuntimeSnapshot {
            actions_reserved: self.actions_reserved,
            mutating_actions_reserved: self.mutating_actions_reserved,
            consecutive_failures: self.consecutive_failures,
            circuit_open: self.circuit_open,
        }
    }

    pub(crate) fn next_trace_hash(
        &mut self,
        action_digest: &str,
        outcome_tag: &str,
        output_digest: Option<&str>,
    ) -> (String, String) {
        let previous = self.previous_trace_hash.clone();
        let mut hasher = blake3::Hasher::new();
        hasher.update(TRACE_DIGEST_DOMAIN);
        hasher.update(previous.as_bytes());
        hasher.update(&[0]);
        hasher.update(action_digest.as_bytes());
        hasher.update(&[0]);
        hasher.update(outcome_tag.as_bytes());
        hasher.update(&[0]);
        if let Some(output_digest) = output_digest {
            hasher.update(output_digest.as_bytes());
        }
        let next = hasher.finalize().to_hex().to_string();
        self.previous_trace_hash = next.clone();
        (previous, next)
    }
}

pub(crate) fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ElementSelector;

    fn click() -> BrowserAction {
        BrowserAction::Click {
            selector: ElementSelector::Css("#submit".into()),
        }
    }

    #[test]
    fn exact_approval_rejects_action_substitution() {
        let session = Uuid::new_v4();
        let request = BrowserApprovalRequest {
            execution_session_id: session,
            action_digest: action_digest(&click()),
            consequence: BrowserConsequence::Interaction,
        };
        let approval = BrowserApproval::approve(request.clone(), 10_000);
        let substituted = BrowserApprovalRequest {
            action_digest: action_digest(&BrowserAction::Click {
                selector: ElementSelector::Css("#delete".into()),
            }),
            ..request
        };
        assert_eq!(
            approval.validate(&substituted, 1),
            Err(BrowserRuntimeDenial::ApprovalActionMismatch)
        );
    }

    #[test]
    fn budget_is_reserved_before_result() {
        let mut state = BrowserRuntimeState::new(BrowserRuntimeLimits {
            max_actions: 2,
            max_mutating_actions: 1,
            max_consecutive_failures: 2,
        })
        .unwrap();
        state.reserve(&click()).unwrap();
        assert_eq!(
            state.reserve(&click()),
            Err(BrowserRuntimeDenial::MutatingActionBudgetExhausted)
        );
    }

    #[test]
    fn failure_circuit_latches() {
        let limits = BrowserRuntimeLimits {
            max_actions: 10,
            max_mutating_actions: 10,
            max_consecutive_failures: 2,
        };
        let mut state = BrowserRuntimeState::new(limits).unwrap();
        state.record_result(false);
        assert!(!state.snapshot().circuit_open);
        state.record_result(false);
        assert!(state.snapshot().circuit_open);
        assert_eq!(
            state.reserve(&BrowserAction::Screenshot),
            Err(BrowserRuntimeDenial::FailureCircuitOpen)
        );
    }

    #[test]
    fn trace_chain_changes_with_outcome() {
        let mut left = BrowserRuntimeState::new(BrowserRuntimeLimits::default()).unwrap();
        let mut right = BrowserRuntimeState::new(BrowserRuntimeLimits::default()).unwrap();
        let digest = action_digest(&click());
        let (_, left_hash) = left.next_trace_hash(&digest, "executed", None);
        let (_, right_hash) = right.next_trace_hash(&digest, "failed", None);
        assert_ne!(left_hash, right_hash);
    }
}
