// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical policy-enforcing browser action executor.
//!
//! Callers submit a [`BrowserAction`] to this layer rather than invoking CDP
//! mutation methods directly. Every attempt yields an [`ActionReceipt`] that
//! records authorization, outcome, output, and elapsed time.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::actions::BrowserAction;
use crate::cdp::CdpSession;
use crate::safety::{BrowserSafetyPolicy, PolicyDecision, PolicyDenial};
use crate::observation::PageObservation;

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

/// Actions that read or manipulate current page state must revalidate the
/// current URL. Explicit navigation and checked history navigation have their
/// own destination-policy gates; NoOp does not touch the browser.
fn requires_current_page_url_check(action: &BrowserAction) -> bool {
    matches!(
        action,
        BrowserAction::Click { .. }
            | BrowserAction::Type { .. }
            | BrowserAction::ScrollTo { .. }
            | BrowserAction::ExtractText { .. }
            | BrowserAction::Screenshot
    )
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

    /// Observe the current page through the executor's capability, Phi and URL checks.
    ///
    /// Perception loops should use this method rather than reading directly from
    /// the low-level CDP session. It serializes against other actions and refuses
    /// to expose page content when the current URL is denied by policy.
    pub async fn observe(&self) -> Result<PageObservation> {
        let _action_guard = self.session.begin_action().await;
        let policy_action = BrowserAction::ExtractText { selector: None };
        if let PolicyDecision::Deny { reason, .. } =
            self.policy.evaluate_action(&policy_action, self.phi)
        {
            bail!("browser observation denied by action policy: {reason:?}");
        }

        let current_url = self.session.current_url().await?;
        if let PolicyDecision::Deny { reason, .. } = self.policy.evaluate_url(&current_url) {
            bail!("browser observation denied by current URL policy: {reason:?}");
        }

        let mut observation = self
            .session
            .observe()
            .await
            .context("Policy-checked accessibility observation failed")?;

        // Return only the redacted URL to external callers; a permitted host
        // can still place credentials or secret-bearing query parameters in the
        // landed URL. Keep the raw URL inside the CDP session for policy checks.
        observation.url = observation.redacted_url();
        Ok(observation)
    }

    /// Dispatch an action after evaluating capability, Phi, and URL policy.
    pub async fn execute(&self, action: BrowserAction) -> ActionReceipt {
        let started = Instant::now();
        // Hold this guard from policy preflight through action dispatch so two
        // executor calls cannot interleave current-URL validation and mutation.
        let _action_guard = self.session.begin_action().await;
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

        // A prior navigation may have been redirected by the browser or site.
        // Re-check the currently displayed page before any observation or
        // interaction that could expose data or cause a side effect. History
        // actions perform their own destination check and remain available as
        // a path away from an unsafe current page.
        if requires_current_page_url_check(&action) {
            let current_url = match self.session.current_url().await {
                Ok(url) => url,
                Err(error) => {
                    return ActionReceipt {
                        action,
                        policy_decision: decision,
                        outcome: ActionOutcome::Failed(error.to_string()),
                        output: ActionOutput::None,
                        elapsed_ms: started.elapsed().as_millis(),
                    };
                }
            };
            let current_url_decision = self.policy.evaluate_url(&current_url);
            if let PolicyDecision::Deny { ref reason, .. } = current_url_decision {
                let reason = reason.clone();
                return ActionReceipt {
                    action,
                    policy_decision: current_url_decision,
                    outcome: ActionOutcome::Denied(reason),
                    output: ActionOutput::None,
                    elapsed_ms: started.elapsed().as_millis(),
                };
            }
        }

        let execution = match &action {
            BrowserAction::Navigate { url } => {
                match self.session.navigate(url).await {
                    Err(error) => Err(error),
                    Ok(()) => match self.session.current_url().await {
                        Err(error) => Err(error),
                        Ok(landed_url) => {
                            let landing_decision = self.policy.evaluate_url(&landed_url);
                            if !landing_decision.is_allowed() {
                                return ActionReceipt {
                                    action: action.clone(),
                                    policy_decision: landing_decision,
                                    outcome: ActionOutcome::Failed(
                                        "blocked post-navigation URL; request may have occurred"
                                            .to_string(),
                                    ),
                                    output: ActionOutput::None,
                                    elapsed_ms: started.elapsed().as_millis(),
                                };
                            }
                            Ok(ActionOutput::None)
                        }
                    },
                }
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
            BrowserAction::GoBack => {
                match self
                    .session
                    .go_back_with_url_policy(|url| self.policy.evaluate_url(url))
                    .await
                {
                    Ok(target_decision) if !target_decision.is_allowed() => {
                        let reason = target_decision
                            .denial_reason()
                            .cloned()
                            .unwrap_or(PolicyDenial::InvalidUrl);
                        return ActionReceipt {
                            action: action.clone(),
                            policy_decision: target_decision,
                            outcome: ActionOutcome::Denied(reason),
                            output: ActionOutput::None,
                            elapsed_ms: started.elapsed().as_millis(),
                        };
                    }
                    Ok(_) => Ok(ActionOutput::None),
                    Err(error) => Err(error),
                }
            }
            BrowserAction::GoForward => {
                match self
                    .session
                    .go_forward_with_url_policy(|url| self.policy.evaluate_url(url))
                    .await
                {
                    Ok(target_decision) if !target_decision.is_allowed() => {
                        let reason = target_decision
                            .denial_reason()
                            .cloned()
                            .unwrap_or(PolicyDenial::InvalidUrl);
                        return ActionReceipt {
                            action: action.clone(),
                            policy_decision: target_decision,
                            outcome: ActionOutcome::Denied(reason),
                            output: ActionOutput::None,
                            elapsed_ms: started.elapsed().as_millis(),
                        };
                    }
                    Ok(_) => Ok(ActionOutput::None),
                    Err(error) => Err(error),
                }
            },
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

#[cfg(test)]
mod current_page_policy_tests {
    use super::*;
    use crate::actions::ElementSelector;

    #[test]
    fn current_url_is_required_for_page_reads_and_interactions() {
        assert!(requires_current_page_url_check(&BrowserAction::Screenshot));
        assert!(requires_current_page_url_check(&BrowserAction::ExtractText { selector: None }));
        assert!(requires_current_page_url_check(&BrowserAction::ScrollTo {
            selector: ElementSelector::Css("main".into())
        }));
        assert!(requires_current_page_url_check(&BrowserAction::Click {
            selector: ElementSelector::Css("button".into())
        }));
        assert!(requires_current_page_url_check(&BrowserAction::Type {
            selector: ElementSelector::Css("input".into()),
            text: "test".into()
        }));
    }

    #[test]
    fn destination_checked_navigation_and_noop_use_their_own_policy_path() {
        assert!(!requires_current_page_url_check(&BrowserAction::Navigate {
            url: "https://example.com".into()
        }));
        assert!(!requires_current_page_url_check(&BrowserAction::GoBack));
        assert!(!requires_current_page_url_check(&BrowserAction::GoForward));
        assert!(!requires_current_page_url_check(&BrowserAction::NoOp));
    }
}
