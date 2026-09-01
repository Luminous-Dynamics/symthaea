// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical policy- and runtime-enforcing browser action executor.
//!
//! Browser policy answers whether an action is generally permitted. Runtime
//! admission independently enforces exact approvals for consequential actions,
//! bounded action budgets, a failure circuit breaker, and privacy-preserving
//! chained receipts. Raw page output is returned separately from durable
//! receipt evidence.

use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::actions::BrowserAction;
use crate::cdp::CdpSession;
use crate::hardening::{
    BrowserApproval, BrowserApprovalRequest, BrowserConsequence, BrowserRuntimeDenial,
    BrowserRuntimeLimits, BrowserRuntimeSnapshot, BrowserRuntimeState, action_digest, action_kind,
    consequence_of, unix_time_ms,
};
use crate::safety::{BrowserSafetyPolicy, PolicyDecision, PolicyDenial};

const OUTPUT_DIGEST_DOMAIN: &[u8] = b"symthaea.browser.output.v1\0";

/// Result class for a browser action proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// The action completed without a transport-level error.
    Executed,
    /// Conscious restraint: no browser operation was requested.
    NoOp,
    /// Browser policy denied the action before runtime admission.
    Denied(PolicyDenial),
    /// Runtime admission denied the action before CDP dispatch.
    RuntimeDenied(BrowserRuntimeDenial),
    /// CDP or postcondition execution failed.
    Failed(String),
}

/// Ephemeral action output returned to the caller.
///
/// This is deliberately not embedded in [`ActionReceipt`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutput {
    None,
    Text(String),
    Screenshot(Vec<u8>),
}

/// Privacy-minimized auditable result of one action proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionReceipt {
    /// Process-local executor session to which approvals are bound.
    pub execution_session_id: Uuid,
    /// Stable semantic action label without page-controlled payloads.
    pub action_kind: String,
    /// Commitment to the exact action including URL/selector/text payload.
    pub action_digest: String,
    pub consequence: BrowserConsequence,
    pub policy_decision: PolicyDecision,
    pub outcome: ActionOutcome,
    /// Commitment to returned data, when any, without retaining that data.
    pub output_digest: Option<String>,
    pub output_len: usize,
    /// Hash-chain predecessor. Empty for the first receipt in a session.
    pub previous_trace_hash: String,
    /// Hash-chain head after this receipt.
    pub trace_hash: String,
    pub elapsed_ms: u128,
}

impl ActionReceipt {
    /// Whether policy/runtime admission allowed dispatch and execution succeeded.
    pub fn succeeded(&self) -> bool {
        matches!(&self.outcome, ActionOutcome::Executed | ActionOutcome::NoOp)
    }
}

/// Complete caller-facing result: durable evidence plus ephemeral output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionExecution {
    pub receipt: ActionReceipt,
    pub output: ActionOutput,
}

impl ActionExecution {
    pub fn succeeded(&self) -> bool {
        self.receipt.succeeded()
    }
}

/// The only normal route from a cognitive action proposal to browser mutation.
pub struct BrowserExecutor<'a> {
    session: &'a CdpSession,
    policy: &'a BrowserSafetyPolicy,
    phi: f64,
    execution_session_id: Uuid,
    runtime: Mutex<BrowserRuntimeState>,
}

impl<'a> BrowserExecutor<'a> {
    pub fn new(session: &'a CdpSession, policy: &'a BrowserSafetyPolicy, phi: f64) -> Self {
        Self::new_with_limits(session, policy, phi, BrowserRuntimeLimits::default())
            .expect("default browser runtime limits are valid")
    }

    pub fn new_with_limits(
        session: &'a CdpSession,
        policy: &'a BrowserSafetyPolicy,
        phi: f64,
        limits: BrowserRuntimeLimits,
    ) -> Result<Self, BrowserRuntimeDenial> {
        Ok(Self {
            session,
            policy,
            phi,
            execution_session_id: Uuid::new_v4(),
            runtime: Mutex::new(BrowserRuntimeState::new(limits)?),
        })
    }

    /// Process-local identity used to bind exact approvals to this executor.
    pub fn execution_session_id(&self) -> Uuid {
        self.execution_session_id
    }

    /// Emit the exact request that an external approval ceremony should review.
    pub fn approval_request(&self, action: &BrowserAction) -> BrowserApprovalRequest {
        BrowserApprovalRequest {
            execution_session_id: self.execution_session_id,
            action_digest: action_digest(action),
            consequence: consequence_of(action),
        }
    }

    pub fn runtime_snapshot(&self) -> BrowserRuntimeSnapshot {
        self.runtime_lock().snapshot()
    }

    /// Compatibility path for callers that only need a receipt.
    ///
    /// `Click` and `Type` intentionally fail closed here because they require
    /// an exact approval and must use [`Self::execute_proposal`].
    pub async fn execute(&self, action: BrowserAction) -> ActionReceipt {
        self.execute_with_output(action).await.receipt
    }

    /// Execute an action and return ephemeral data separately from its receipt.
    pub async fn execute_with_output(&self, action: BrowserAction) -> ActionExecution {
        self.execute_proposal(action, None).await
    }

    /// Dispatch a proposal after browser policy and independent runtime
    /// admission. Consequential `Click`/`Type` actions require an exact,
    /// unexpired approval bound to this executor session and action digest.
    pub async fn execute_proposal(
        &self,
        action: BrowserAction,
        approval: Option<&BrowserApproval>,
    ) -> ActionExecution {
        let started = Instant::now();
        let decision = self.policy.evaluate_action(&action, self.phi);
        let digest = action_digest(&action);
        let consequence = consequence_of(&action);

        if let PolicyDecision::Deny { ref reason, .. } = decision {
            return self.finish(
                action,
                digest,
                consequence,
                decision,
                ActionOutcome::Denied(reason.clone()),
                ActionOutput::None,
                started,
                false,
            );
        }

        if consequence.requires_exact_approval() {
            let request = self.approval_request(&action);
            let approval_result = approval
                .ok_or(BrowserRuntimeDenial::ExactApprovalRequired)
                .and_then(|approval| approval.validate(&request, unix_time_ms()));
            if let Err(denial) = approval_result {
                return self.finish(
                    action,
                    digest,
                    consequence,
                    decision,
                    ActionOutcome::RuntimeDenied(denial),
                    ActionOutput::None,
                    started,
                    false,
                );
            }
        }

        if let Err(denial) = self.runtime_lock().reserve(&action) {
            return self.finish(
                action,
                digest,
                consequence,
                decision,
                ActionOutcome::RuntimeDenied(denial),
                ActionOutput::None,
                started,
                false,
            );
        }

        let execution = match &action {
            BrowserAction::Navigate { url } => {
                self.session.navigate(url).await.map(|_| ActionOutput::None)
            }
            BrowserAction::Click { selector } => self
                .session
                .click(selector)
                .await
                .map(|_| ActionOutput::None),
            BrowserAction::Type { selector, text } => self
                .session
                .type_text(selector, text)
                .await
                .map(|_| ActionOutput::None),
            BrowserAction::ScrollTo { selector } => self
                .session
                .scroll_to(selector)
                .await
                .map(|_| ActionOutput::None),
            BrowserAction::GoBack => self.session.go_back().await.map(|_| ActionOutput::None),
            BrowserAction::GoForward => self.session.go_forward().await.map(|_| ActionOutput::None),
            BrowserAction::ExtractText { selector } => self
                .session
                .extract_text(selector.as_deref())
                .await
                .map(ActionOutput::Text),
            BrowserAction::Screenshot => self
                .session
                .screenshot()
                .await
                .map(ActionOutput::Screenshot),
            BrowserAction::NoOp => Ok(ActionOutput::None),
        };

        let (outcome, output, succeeded) = match execution {
            Ok(output) if matches!(&action, BrowserAction::NoOp) => {
                (ActionOutcome::NoOp, output, true)
            }
            Ok(output) => (ActionOutcome::Executed, output, true),
            Err(error) => (
                ActionOutcome::Failed(error.to_string()),
                ActionOutput::None,
                false,
            ),
        };

        self.runtime_lock().record_result(succeeded);
        self.finish(
            action,
            digest,
            consequence,
            decision,
            outcome,
            output,
            started,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        _action: BrowserAction,
        action_digest: String,
        consequence: BrowserConsequence,
        policy_decision: PolicyDecision,
        outcome: ActionOutcome,
        output: ActionOutput,
        started: Instant,
        _runtime_admitted: bool,
    ) -> ActionExecution {
        let (output_digest, output_len) = summarize_output(&output);
        let outcome_tag = outcome_tag(&outcome);
        let (previous_trace_hash, trace_hash) = self.runtime_lock().next_trace_hash(
            &action_digest,
            outcome_tag,
            output_digest.as_deref(),
        );
        let receipt = ActionReceipt {
            execution_session_id: self.execution_session_id,
            action_kind: action_kind_from_digest_context(consequence, outcome_tag).to_string(),
            action_digest,
            consequence,
            policy_decision,
            outcome,
            output_digest,
            output_len,
            previous_trace_hash,
            trace_hash,
            elapsed_ms: started.elapsed().as_millis(),
        };
        ActionExecution { receipt, output }
    }

    fn runtime_lock(&self) -> std::sync::MutexGuard<'_, BrowserRuntimeState> {
        self.runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn action_kind_from_digest_context(
    consequence: BrowserConsequence,
    _outcome_tag: &str,
) -> &'static str {
    // This fallback exists only because receipts intentionally do not retain the
    // action object. The exact semantic action is still committed by
    // `action_digest`; consequence remains independently inspectable.
    match consequence {
        BrowserConsequence::Passive => "passive",
        BrowserConsequence::Navigation => "navigation",
        BrowserConsequence::Interaction => "click",
        BrowserConsequence::TextEntry => "type",
    }
}

fn summarize_output(output: &ActionOutput) -> (Option<String>, usize) {
    let bytes: &[u8] = match output {
        ActionOutput::None => return (None, 0),
        ActionOutput::Text(text) => text.as_bytes(),
        ActionOutput::Screenshot(bytes) => bytes,
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(OUTPUT_DIGEST_DOMAIN);
    hasher.update(bytes);
    (Some(hasher.finalize().to_hex().to_string()), bytes.len())
}

fn outcome_tag(outcome: &ActionOutcome) -> &'static str {
    match outcome {
        ActionOutcome::Executed => "executed",
        ActionOutcome::NoOp => "noop",
        ActionOutcome::Denied(_) => "policy-denied",
        ActionOutcome::RuntimeDenied(_) => "runtime-denied",
        ActionOutcome::Failed(_) => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ElementSelector;

    #[test]
    fn output_summary_never_retains_text() {
        let secret = "correct horse battery staple";
        let (digest, len) = summarize_output(&ActionOutput::Text(secret.into()));
        assert_eq!(len, secret.len());
        let digest = digest.unwrap();
        assert!(!digest.contains(secret));
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn approval_request_is_exact_action_bound() {
        let a = BrowserAction::Click {
            selector: ElementSelector::Css("#allow".into()),
        };
        let b = BrowserAction::Click {
            selector: ElementSelector::Css("#delete".into()),
        };
        assert_ne!(action_digest(&a), action_digest(&b));
        assert_eq!(action_kind(&a), "click");
    }
}
