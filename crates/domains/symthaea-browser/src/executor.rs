// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical policy-enforcing browser action executor.
//!
//! Callers submit a [`BrowserAction`] to this layer rather than invoking CDP
//! mutation methods directly. Every attempt yields an [`ActionReceipt`] that
//! records authorization, outcome, output, and elapsed time.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::actions::BrowserAction;
use crate::cdp::CdpSession;
use crate::safety::{BrowserSafetyPolicy, PolicyDecision, PolicyDenial};

/// Result class for a dispatched browser action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// The action completed without a transport-level error.
    Executed,
    /// Conscious restraint: no browser operation was requested.
    NoOp,
    /// Policy denied the action before CDP dispatch.
    Denied(PolicyDenial),
    /// CDP or postcondition execution failed.
    Failed(String),
}

/// Optional action output returned to the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutput {
    None,
    Text(String),
    Screenshot(Vec<u8>),
}

/// Auditable result of one action proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionReceipt {
    pub action: BrowserAction,
    pub policy_decision: PolicyDecision,
    pub outcome: ActionOutcome,
    pub output: ActionOutput,
    pub elapsed_ms: u128,
}

impl ActionReceipt {
    /// Whether policy allowed dispatch and execution succeeded.
    pub fn succeeded(&self) -> bool {
        matches!(&self.outcome, ActionOutcome::Executed | ActionOutcome::NoOp)
    }
}

/// The only normal route from a cognitive action proposal to browser mutation.
pub struct BrowserExecutor<'a> {
    session: &'a CdpSession,
    policy: &'a BrowserSafetyPolicy,
    phi: f64,
}

impl<'a> BrowserExecutor<'a> {
    pub fn new(session: &'a CdpSession, policy: &'a BrowserSafetyPolicy, phi: f64) -> Self {
        Self {
            session,
            policy,
            phi,
        }
    }

    /// Dispatch an action after evaluating capability, Phi, and URL policy.
    pub async fn execute(&self, action: BrowserAction) -> ActionReceipt {
        let started = Instant::now();
        let decision = self.policy.evaluate_action(&action, self.phi);

        if let PolicyDecision::Deny { ref reason, .. } = decision {
            return ActionReceipt {
                action,
                policy_decision: decision,
                outcome: ActionOutcome::Denied(reason.clone()),
                output: ActionOutput::None,
                elapsed_ms: started.elapsed().as_millis(),
            };
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

        let (outcome, output) = match execution {
            Ok(output) if matches!(&action, BrowserAction::NoOp) => (ActionOutcome::NoOp, output),
            Ok(output) => (ActionOutcome::Executed, output),
            Err(error) => (ActionOutcome::Failed(error.to_string()), ActionOutput::None),
        };

        ActionReceipt {
            action,
            policy_decision: decision,
            outcome,
            output,
            elapsed_ms: started.elapsed().as_millis(),
        }
    }
}
