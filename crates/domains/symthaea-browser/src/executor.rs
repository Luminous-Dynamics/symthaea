// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical policy-enforcing browser action executor.
//!
//! Persistable audit receipts contain bounded, secret-safe summaries only.
//! Immediate execution output and detailed diagnostics are returned separately
//! in a type that deliberately does not implement Serialize or Debug.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::actions::{BrowserAction, ElementSelector};
use crate::cdp::CdpSession;
use crate::observation::PageObservation;
use crate::safety::{BrowserSafetyPolicy, PolicyDecision};

const MAX_AUDIT_ORIGIN_CHARS: usize = 512;

/// Stable, bounded kind of action; never contains caller-provided strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserActionKind {
    Navigate,
    Click,
    Type,
    ScrollTo,
    GoBack,
    GoForward,
    ExtractText,
    Screenshot,
    NoOp,
}

/// The kind of selector used, without its value or accessible name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectorKind {
    BackendNodeId,
    Css,
    Accessible,
}

/// Safe action metadata suitable for audit persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSummary {
    pub kind: BrowserActionKind,
    pub selector_kind: Option<SelectorKind>,
    /// HTTP(S) origin only. Userinfo, path, query and fragment are excluded.
    pub navigation_origin: Option<String>,
    /// Length metadata only; the text entered is never stored here.
    pub text_len_bytes: Option<usize>,
    pub text_len_chars: Option<usize>,
}

impl ActionSummary {
    pub fn from_action(action: &BrowserAction) -> Self {
        let (kind, selector_kind, navigation_origin, text_len_bytes, text_len_chars) =
            match action {
                BrowserAction::Navigate { url } => (
                    BrowserActionKind::Navigate,
                    None,
                    safe_navigation_origin(url),
                    None,
                    None,
                ),
                BrowserAction::Click { selector } => (
                    BrowserActionKind::Click,
                    Some(selector_kind(selector)),
                    None,
                    None,
                    None,
                ),
                BrowserAction::Type { selector, text } => (
                    BrowserActionKind::Type,
                    Some(selector_kind(selector)),
                    None,
                    Some(text.len()),
                    Some(text.chars().count()),
                ),
                BrowserAction::ScrollTo { selector } => (
                    BrowserActionKind::ScrollTo,
                    Some(selector_kind(selector)),
                    None,
                    None,
                    None,
                ),
                BrowserAction::GoBack => (
                    BrowserActionKind::GoBack,
                    None,
                    None,
                    None,
                    None,
                ),
                BrowserAction::GoForward => (
                    BrowserActionKind::GoForward,
                    None,
                    None,
                    None,
                    None,
                ),
                BrowserAction::ExtractText { selector } => (
                    BrowserActionKind::ExtractText,
                    selector.as_ref().map(|_| SelectorKind::Css),
                    None,
                    None,
                    None,
                ),
                BrowserAction::Screenshot => (
                    BrowserActionKind::Screenshot,
                    None,
                    None,
                    None,
                    None,
                ),
                BrowserAction::NoOp => (
                    BrowserActionKind::NoOp,
                    None,
                    None,
                    None,
                    None,
                ),
            };

        Self {
            kind,
            selector_kind,
            navigation_origin,
            text_len_bytes,
            text_len_chars,
        }
    }
}

fn selector_kind(selector: &ElementSelector) -> SelectorKind {
    match selector {
        ElementSelector::BackendNodeId(_) => SelectorKind::BackendNodeId,
        ElementSelector::Css(_) => SelectorKind::Css,
        ElementSelector::Accessible { .. } => SelectorKind::Accessible,
    }
}

fn safe_navigation_origin(value: &str) -> Option<String> {
    let parsed = Url::parse(value).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }

    let origin = parsed.origin().ascii_serialization();
    if origin == "null" || origin.len() > MAX_AUDIT_ORIGIN_CHARS {
        return None;
    }
    Some(origin)
}

/// Bounded action outcome class; contains no arbitrary error or page text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    Executed,
    NoOp,
    Denied,
    Failed,
}

/// Stable reason codes for auditing. Detailed error strings never enter receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionReasonCode {
    PolicyDenied,
    CurrentUrlUnavailable,
    CurrentUrlDenied,
    NavigationLandedOnDeniedUrl,
    BrowserOperationFailed,
}

/// Persistable audit metadata. This intentionally excludes the original
/// BrowserAction, extracted page text, screenshot bytes, and free-form errors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionReceipt {
    pub action: ActionSummary,
    pub policy_decision: PolicyDecision,
    pub outcome: ActionOutcome,
    pub reason_code: Option<ActionReasonCode>,
    pub elapsed_ms: u128,
}

impl ActionReceipt {
    /// Whether policy allowed dispatch and execution succeeded.
    pub fn succeeded(&self) -> bool {
        matches!(self.outcome, ActionOutcome::Executed | ActionOutcome::NoOp)
    }
}

/// Raw output is transient caller data, not audit metadata.
///
/// This type deliberately has no Serialize or Debug implementation. Callers
/// that receive text or screenshots must make a separate, explicit retention
/// decision and apply their own access-control and redaction policy.
pub enum ActionOutput {
    None,
    Text(String),
    Screenshot(Vec<u8>),
}

/// Immediate result of an execution attempt.
///
/// The nested receipt is safe to persist. Output and diagnostic details are
/// kept outside it. This type deliberately does not implement Serialize,
/// Clone, or Debug; cancelling the execute future yields no result and does
/// not implicitly persist a receipt.
pub struct ActionExecution {
    receipt: ActionReceipt,
    /// Immediate caller output; may contain sensitive page content.
    pub output: ActionOutput,
    diagnostic: Option<String>,
}

impl ActionExecution {
    /// Read-only access to the safe audit receipt.
    pub fn receipt(&self) -> &ActionReceipt {
        &self.receipt
    }

    pub fn succeeded(&self) -> bool {
        self.receipt.succeeded()
    }

    /// Access detailed diagnostics only for controlled, transient debugging.
    /// The string may contain sensitive URLs or remote error text; never log
    /// or persist it without application-specific redaction.
    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
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

fn build_execution(
    action: &BrowserAction,
    policy_decision: PolicyDecision,
    outcome: ActionOutcome,
    output: ActionOutput,
    reason_code: Option<ActionReasonCode>,
    elapsed_ms: u128,
    diagnostic: Option<String>,
) -> ActionExecution {
    let receipt = ActionReceipt {
        action: ActionSummary::from_action(action),
        policy_decision,
        outcome,
        reason_code,
        elapsed_ms,
    };
    ActionExecution {
        receipt,
        output,
        diagnostic,
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
    ///
    /// The returned receipt contains safe audit metadata only. Text, screenshots
    /// and detailed failure information are transient fields on ActionExecution.
    pub async fn execute(&self, action: BrowserAction) -> ActionExecution {
        let started = Instant::now();
        // Hold this guard from policy preflight through action dispatch so two
        // executor calls cannot interleave current-URL validation and mutation.
        let _action_guard = self.session.begin_action().await;
        let decision = self.policy.evaluate_action(&action, self.phi);

        if matches!(&decision, PolicyDecision::Deny { .. }) {
            return build_execution(
                &action,
                decision,
                ActionOutcome::Denied,
                ActionOutput::None,
                Some(ActionReasonCode::PolicyDenied),
                started.elapsed().as_millis(),
                None,
            );
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
                    return build_execution(
                        &action,
                        decision,
                        ActionOutcome::Failed,
                        ActionOutput::None,
                        Some(ActionReasonCode::CurrentUrlUnavailable),
                        started.elapsed().as_millis(),
                        Some(error.to_string()),
                    );
                }
            };
            let current_url_decision = self.policy.evaluate_url(&current_url);
            if matches!(&current_url_decision, PolicyDecision::Deny { .. }) {
                return build_execution(
                    &action,
                    current_url_decision,
                    ActionOutcome::Denied,
                    ActionOutput::None,
                    Some(ActionReasonCode::CurrentUrlDenied),
                    started.elapsed().as_millis(),
                    None,
                );
            }
        }

        let execution: Result<ActionOutput> = match &action {
            BrowserAction::Navigate { url } => {
                match self.session.navigate(url).await {
                    Err(error) => Err(error),
                    Ok(()) => match self.session.current_url().await {
                        Err(error) => Err(error),
                        Ok(landed_url) => {
                            let landing_decision = self.policy.evaluate_url(&landed_url);
                            if !landing_decision.is_allowed() {
                                return build_execution(
                                    &action,
                                    landing_decision,
                                    ActionOutcome::Failed,
                                    ActionOutput::None,
                                    Some(ActionReasonCode::NavigationLandedOnDeniedUrl),
                                    started.elapsed().as_millis(),
                                    Some(
                                        "Navigation landed on a URL denied by policy; request may already have occurred."
                                            .to_string(),
                                    ),
                                );
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
                        return build_execution(
                            &action,
                            target_decision,
                            ActionOutcome::Denied,
                            ActionOutput::None,
                            Some(ActionReasonCode::PolicyDenied),
                            started.elapsed().as_millis(),
                            None,
                        );
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
                        return build_execution(
                            &action,
                            target_decision,
                            ActionOutcome::Denied,
                            ActionOutput::None,
                            Some(ActionReasonCode::PolicyDenied),
                            started.elapsed().as_millis(),
                            None,
                        );
                    }
                    Ok(_) => Ok(ActionOutput::None),
                    Err(error) => Err(error),
                }
            }
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

        match execution {
            Ok(output) => {
                let outcome = if matches!(&action, BrowserAction::NoOp) {
                    ActionOutcome::NoOp
                } else {
                    ActionOutcome::Executed
                };
                build_execution(
                    &action,
                    decision,
                    outcome,
                    output,
                    None,
                    started.elapsed().as_millis(),
                    None,
                )
            }
            Err(error) => build_execution(
                &action,
                decision,
                ActionOutcome::Failed,
                ActionOutput::None,
                Some(ActionReasonCode::BrowserOperationFailed),
                started.elapsed().as_millis(),
                Some(error.to_string()),
            ),
        }
    }
}

#[cfg(test)]
mod current_page_policy_tests {
    use super::*;

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

#[cfg(test)]
mod receipt_privacy_tests {
    use super::*;
    use crate::safety::PolicyDenial;

    const PASSWORD: &str = "supersecret-password";
    const OTP: &str = "481516";
    const BEARER: &str = "bearer-secret-token";
    const EMAIL: &str = "person@example.net";
    const SELECTOR_SECRET: &str = "#selector-secret";
    const QUERY_SECRET: &str = "query-secret";
    const FRAGMENT_SECRET: &str = "fragment-secret";
    const PAGE_SECRET: &str = "page-output-secret";
    const DIAGNOSTIC_SECRET: &str = "diagnostic-output-secret";
    const SCREENSHOT_SECRET: &[u8] = b"screenshot-payload-secret";

    fn allowed() -> PolicyDecision {
        PolicyDecision::Allow {
            required_phi: Some(0.5),
        }
    }

    fn assert_no_literals(serialized: &str, values: &[&str]) {
        for value in values {
            assert!(
                !serialized.contains(value),
                "receipt leaked a sensitive literal"
            );
        }
    }

    #[test]
    fn receipt_summarizes_type_action_without_serializing_input_selector_or_output() {
        let input = format!(
            "password={PASSWORD} otp={OTP} bearer={BEARER} email={EMAIL}"
        );
        let action = BrowserAction::Type {
            selector: ElementSelector::Css(SELECTOR_SECRET.to_string()),
            text: input.clone(),
        };
        let execution = build_execution(
            &action,
            allowed(),
            ActionOutcome::Failed,
            ActionOutput::Text(PAGE_SECRET.to_string()),
            Some(ActionReasonCode::BrowserOperationFailed),
            3,
            Some(DIAGNOSTIC_SECRET.to_string()),
        );

        let serialized =
            serde_json::to_string(&execution.receipt).expect("safe receipt serializes");
        assert_no_literals(
            &serialized,
            &[
                PASSWORD,
                OTP,
                BEARER,
                EMAIL,
                SELECTOR_SECRET,
                PAGE_SECRET,
                DIAGNOSTIC_SECRET,
            ],
        );
        assert!(serialized.contains("text_len_bytes"));
        assert!(serialized.contains("text_len_chars"));
        assert!(serialized.contains("BrowserOperationFailed"));
        assert!(matches!(
            &execution.output,
            ActionOutput::Text(text) if text == PAGE_SECRET
        ));
        assert!(matches!(
            &action,
            BrowserAction::Type { text, .. } if text == input
        ));
    }

    #[test]
    fn receipt_navigation_origin_excludes_userinfo_path_query_and_fragment() {
        let action = BrowserAction::Navigate {
            url: format!(
                "https://user:{PASSWORD}@example.com/private/path?token={QUERY_SECRET}#{FRAGMENT_SECRET}"
            ),
        };
        let execution = build_execution(
            &action,
            allowed(),
            ActionOutcome::Denied,
            ActionOutput::None,
            Some(ActionReasonCode::PolicyDenied),
            1,
            None,
        );
        let serialized =
            serde_json::to_string(&execution.receipt).expect("safe receipt serializes");

        assert!(serialized.contains("https://example.com"));
        assert_no_literals(
            &serialized,
            &[PASSWORD, QUERY_SECRET, FRAGMENT_SECRET, "/private/path", "user:"],
        );
    }

    #[test]
    fn receipt_never_contains_screenshot_bytes_on_denial_or_failure() {
        let action = BrowserAction::Screenshot;
        for outcome in [ActionOutcome::Denied, ActionOutcome::Failed] {
            let (decision, reason_code) = match outcome {
                ActionOutcome::Denied => (
                    PolicyDecision::Deny {
                        reason: PolicyDenial::InvalidUrl,
                        required_phi: Some(0.05),
                    },
                    ActionReasonCode::PolicyDenied,
                ),
                ActionOutcome::Failed => (allowed(), ActionReasonCode::BrowserOperationFailed),
                _ => unreachable!("the test only exercises denied and failed outcomes"),
            };
            let execution = build_execution(
                &action,
                decision,
                outcome,
                ActionOutput::Screenshot(SCREENSHOT_SECRET.to_vec()),
                Some(reason_code),
                0,
                Some(DIAGNOSTIC_SECRET.to_string()),
            );
            let serialized =
                serde_json::to_string(&execution.receipt).expect("safe receipt serializes");
            assert_no_literals(
                &serialized,
                &[std::str::from_utf8(SCREENSHOT_SECRET).unwrap(), DIAGNOSTIC_SECRET],
            );
        }
    }

    #[test]
    fn summary_is_secret_safe_without_execution_output_for_abandoned_work() {
        // A cancelled execution future returns no ActionExecution. If a caller
        // needs a cancellation record, this summary is safe to persist without
        // retaining the raw action or output.
        let action = BrowserAction::Type {
            selector: ElementSelector::Css(SELECTOR_SECRET.to_string()),
            text: format!("{PASSWORD} {OTP} {BEARER} {EMAIL}"),
        };
        let summary = ActionSummary::from_action(&action);
        let serialized = serde_json::to_string(&summary).expect("safe summary serializes");
        assert_no_literals(
            &serialized,
            &[PASSWORD, OTP, BEARER, EMAIL, SELECTOR_SECRET],
        );
    }

}
