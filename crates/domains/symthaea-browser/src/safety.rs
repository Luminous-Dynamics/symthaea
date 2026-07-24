// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser safety policy and auditable policy decisions.
//!
//! The safety layer sits between action selection and CDP actuation. It
//! validates policy parameters, consciousness state, URL structure, origin
//! scope, and local-network access before an action is dispatched.
//!
//! Phi is a caution signal, not an authority grant. A sufficient Phi can pass
//! an action's coherence threshold, but it never overrides a URL denial.

use std::net::IpAddr;

use serde::{Deserialize, Serialize};
use tracing::warn;
use url::{Host, Url};

use crate::actions::{BrowserAction, BrowserCapability};

/// Stable denial reasons suitable for telemetry, tests, and audit receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyDenial {
    /// A policy parameter is non-finite or outside its documented range.
    InvalidPolicy,
    /// The supplied Phi value is non-finite or outside `[0.0, 1.0]`.
    InvalidPhi,
    /// The policy does not grant the action's required authority.
    MissingCapability,
    /// Phi did not meet the action's effective threshold.
    InsufficientPhi,
    /// The URL could not be parsed as an absolute URL.
    InvalidUrl,
    /// The URL scheme is not explicitly permitted.
    DisallowedScheme,
    /// User information was embedded in the URL.
    UrlCredentialsDenied,
    /// The URL has no host even though the scheme requires one.
    MissingHost,
    /// Loopback, link-local, private, multicast, or otherwise local addressing
    /// was denied by the policy.
    PrivateNetworkDenied,
    /// The canonical host matched the blocklist.
    BlocklistedHost,
    /// An allowlist exists and the canonical host did not match it.
    HostNotAllowlisted,
}

/// Result of evaluating a URL or browser action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// The operation is permitted. `required_phi` is populated for actions and
    /// omitted for URL-only checks.
    Allow { required_phi: Option<f64> },
    /// The operation is denied for a stable, inspectable reason.
    Deny {
        reason: PolicyDenial,
        required_phi: Option<f64>,
    },
}

impl PolicyDecision {
    /// Whether this decision permits the requested operation.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }

    /// Denial reason, when denied.
    pub fn denial_reason(&self) -> Option<&PolicyDenial> {
        match self {
            Self::Deny { reason, .. } => Some(reason),
            Self::Allow { .. } => None,
        }
    }
}

/// Safety policy for browser actions.
///
/// `#[serde(default)]` keeps stored configurations forward-compatible when new
/// safety fields are introduced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserSafetyPolicy {
    /// If non-empty, only these hosts and their subdomains are allowed.
    /// Entries may be bare hosts (`example.com`) or absolute URLs.
    pub url_allowlist: Vec<String>,

    /// Hosts that are always blocked, even when also allowlisted.
    /// Entries may be bare hosts or absolute URLs.
    pub url_blocklist: Vec<String>,

    /// URL schemes that may be navigated. Defaults to HTTP and HTTPS only.
    pub allowed_schemes: Vec<String>,

    /// Permit loopback, RFC1918, link-local, and other local-network targets.
    /// Disabled by default to reduce SSRF and browser-to-host pivoting risk.
    pub allow_private_networks: bool,

    /// Permit `username:password@host` URL syntax. Disabled by default to avoid
    /// credential confusion and accidental secret disclosure.
    pub allow_url_credentials: bool,

    /// Explicit browser authorities granted to this agent. The default policy
    /// permits observation and navigation, but not clicking or text entry.
    pub granted_capabilities: Vec<BrowserCapability>,

    /// Global minimum Phi required for any action. Must be finite and in
    /// `[0.0, 1.0]`.
    pub global_phi_floor: f64,

    /// Multiplier applied to each action's `required_phi()`. Must be finite and
    /// non-negative. Values above `1.0` make the agent more cautious.
    pub phi_multiplier: f64,
}

impl Default for BrowserSafetyPolicy {
    fn default() -> Self {
        Self {
            url_allowlist: Vec::new(),
            url_blocklist: Vec::new(),
            allowed_schemes: vec!["http".to_string(), "https".to_string()],
            allow_private_networks: false,
            allow_url_credentials: false,
            granted_capabilities: vec![BrowserCapability::Observe, BrowserCapability::Navigate],
            global_phi_floor: 0.0,
            phi_multiplier: 1.0,
        }
    }
}

impl BrowserSafetyPolicy {
    /// Evaluate a URL and return an auditable decision.
    ///
    /// This performs syntactic origin and IP-literal checks. A production
    /// network boundary should additionally pin DNS resolution and re-check the
    /// resolved address on redirects to prevent DNS rebinding.
    pub fn evaluate_url(&self, raw_url: &str) -> PolicyDecision {
        if !self.policy_values_valid() {
            return Self::deny(PolicyDenial::InvalidPolicy, None);
        }

        let parsed = match Url::parse(raw_url) {
            Ok(url) => url,
            Err(_) => return Self::deny(PolicyDenial::InvalidUrl, None),
        };

        let scheme = parsed.scheme().to_ascii_lowercase();
        if !self
            .allowed_schemes
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&scheme))
        {
            return Self::deny(PolicyDenial::DisallowedScheme, None);
        }

        if !self.allow_url_credentials
            && (!parsed.username().is_empty() || parsed.password().is_some())
        {
            return Self::deny(PolicyDenial::UrlCredentialsDenied, None);
        }

        let host = match parsed.host() {
            Some(host) => host,
            None => return Self::deny(PolicyDenial::MissingHost, None),
        };

        let canonical_host = canonical_host(&host);
        if !self.allow_private_networks && host_is_local_or_private(&host, &canonical_host) {
            return Self::deny(PolicyDenial::PrivateNetworkDenied, None);
        }

        if self
            .url_blocklist
            .iter()
            .filter_map(|entry| host_pattern(entry))
            .any(|pattern| host_matches(&canonical_host, &pattern))
        {
            return Self::deny(PolicyDenial::BlocklistedHost, None);
        }

        if !self.url_allowlist.is_empty()
            && !self
                .url_allowlist
                .iter()
                .filter_map(|entry| host_pattern(entry))
                .any(|pattern| host_matches(&canonical_host, &pattern))
        {
            return Self::deny(PolicyDenial::HostNotAllowlisted, None);
        }

        PolicyDecision::Allow { required_phi: None }
    }

    /// Evaluate an action under the current Phi value.
    pub fn evaluate_action(&self, action: &BrowserAction, phi: f64) -> PolicyDecision {
        if !self.policy_values_valid() {
            return Self::deny(PolicyDenial::InvalidPolicy, None);
        }
        if !phi.is_finite() || !(0.0..=1.0).contains(&phi) {
            return Self::deny(PolicyDenial::InvalidPhi, None);
        }

        if !self
            .granted_capabilities
            .contains(&action.required_capability())
        {
            return Self::deny(PolicyDenial::MissingCapability, None);
        }

        let threshold = (action.required_phi() * self.phi_multiplier).max(self.global_phi_floor);
        if phi < threshold {
            return Self::deny(PolicyDenial::InsufficientPhi, Some(threshold));
        }

        if let BrowserAction::Navigate { url } = action {
            let decision = self.evaluate_url(url);
            if let PolicyDecision::Deny { reason, .. } = decision {
                return Self::deny(reason, Some(threshold));
            }
        }

        PolicyDecision::Allow {
            required_phi: Some(threshold),
        }
    }

    /// Compatibility helper for callers that only need a boolean.
    pub fn is_url_allowed(&self, url: &str) -> bool {
        let decision = self.evaluate_url(url);
        if let PolicyDecision::Deny { ref reason, .. } = decision {
            warn!(url, ?reason, "URL denied by browser policy");
        }
        decision.is_allowed()
    }

    /// Compatibility helper for callers that only need a boolean.
    pub fn is_action_allowed(&self, action: &BrowserAction, phi: f64) -> bool {
        let decision = self.evaluate_action(action, phi);
        if let PolicyDecision::Deny {
            ref reason,
            required_phi,
        } = decision
        {
            warn!(
                ?action,
                phi,
                ?required_phi,
                ?reason,
                "Browser action denied"
            );
        }
        decision.is_allowed()
    }

    /// Create a restrictive policy that only allows specified hosts.
    pub fn allowlist_only(domains: Vec<String>) -> Self {
        Self {
            url_allowlist: domains,
            ..Default::default()
        }
    }

    /// Create a policy with all currently defined browser capabilities. URL
    /// and local-network restrictions remain unchanged.
    pub fn interactive() -> Self {
        Self {
            granted_capabilities: vec![
                BrowserCapability::Observe,
                BrowserCapability::Navigate,
                BrowserCapability::Interact,
                BrowserCapability::EnterText,
            ],
            ..Default::default()
        }
    }

    /// Grant a capability if it is not already present.
    pub fn grant(&mut self, capability: BrowserCapability) {
        if !self.granted_capabilities.contains(&capability) {
            self.granted_capabilities.push(capability);
        }
    }

    fn policy_values_valid(&self) -> bool {
        self.global_phi_floor.is_finite()
            && (0.0..=1.0).contains(&self.global_phi_floor)
            && self.phi_multiplier.is_finite()
            && self.phi_multiplier >= 0.0
            && !self.allowed_schemes.is_empty()
            && self
                .allowed_schemes
                .iter()
                .all(|scheme| is_valid_scheme_token(scheme))
    }

    fn deny(reason: PolicyDenial, required_phi: Option<f64>) -> PolicyDecision {
        PolicyDecision::Deny {
            reason,
            required_phi,
        }
    }
}

fn is_valid_scheme_token(scheme: &str) -> bool {
    !scheme.is_empty()
        && scheme.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphabetic()
                || (index > 0 && (byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')))
        })
}

fn canonical_host(host: &Host<&str>) -> String {
    match host {
        Host::Domain(domain) => domain.trim_end_matches('.').to_ascii_lowercase(),
        Host::Ipv4(ip) => ip.to_string(),
        Host::Ipv6(ip) => ip.to_string(),
    }
}

fn host_is_local_or_private(host: &Host<&str>, canonical: &str) -> bool {
    match host {
        Host::Ipv4(ip) => is_disallowed_ip(IpAddr::V4(*ip)),
        Host::Ipv6(ip) => is_disallowed_ip(IpAddr::V6(*ip)),
        Host::Domain(_) => {
            canonical == "localhost"
                || canonical.ends_with(".localhost")
                || canonical.ends_with(".local")
        }
    }
}

fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_multicast()
                || ip.is_unspecified()
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

fn host_pattern(entry: &str) -> Option<String> {
    let trimmed = entry.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(url) = Url::parse(trimmed) {
        return url
            .host_str()
            .map(|host| host.trim_end_matches('.').to_ascii_lowercase());
    }

    let without_dot = trimmed.trim_start_matches('.').trim_end_matches('.');
    if without_dot.is_empty()
        || without_dot.contains('/')
        || without_dot.contains('@')
        || without_dot.contains(':')
    {
        return None;
    }
    Some(without_dot.to_ascii_lowercase())
}

fn host_matches(host: &str, pattern: &str) -> bool {
    host == pattern
        || host
            .strip_suffix(pattern)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_blocks_non_web_schemes_and_local_targets() {
        let policy = BrowserSafetyPolicy::default();
        assert!(!policy.is_url_allowed("file:///etc/passwd"));
        assert!(!policy.is_url_allowed("javascript:alert(1)"));
        assert!(!policy.is_url_allowed("chrome://settings"));
        assert!(!policy.is_url_allowed("http://127.0.0.1:8080"));
        assert!(!policy.is_url_allowed("http://169.254.169.254/latest/meta-data"));
        assert!(!policy.is_url_allowed("http://localhost:3000"));
        assert!(policy.is_url_allowed("https://example.com"));
    }

    #[test]
    fn local_targets_require_explicit_opt_in() {
        let policy = BrowserSafetyPolicy {
            allow_private_networks: true,
            ..Default::default()
        };
        assert!(policy.is_url_allowed("http://127.0.0.1:8080"));
        assert!(policy.is_url_allowed("http://localhost:3000"));
    }

    #[test]
    fn allowlist_matches_host_boundary_only() {
        let policy = BrowserSafetyPolicy::allowlist_only(vec!["example.com".into()]);
        assert!(policy.is_url_allowed("https://example.com/page"));
        assert!(policy.is_url_allowed("https://sub.example.com"));
        assert!(!policy.is_url_allowed("https://example.com.evil.test"));
        assert!(!policy.is_url_allowed("https://evil.test/?next=example.com"));
        assert!(!policy.is_url_allowed("https://example.com@evil.test"));
    }

    #[test]
    fn blocklist_wins_over_allowlist() {
        let policy = BrowserSafetyPolicy {
            url_allowlist: vec!["example.com".into()],
            url_blocklist: vec!["admin.example.com".into()],
            ..Default::default()
        };
        assert!(policy.is_url_allowed("https://example.com"));
        assert!(!policy.is_url_allowed("https://admin.example.com"));
    }

    #[test]
    fn url_credentials_are_denied_by_default() {
        let policy = BrowserSafetyPolicy::default();
        assert_eq!(
            policy
                .evaluate_url("https://user:secret@example.com")
                .denial_reason(),
            Some(&PolicyDenial::UrlCredentialsDenied)
        );
    }

    #[test]
    fn action_phi_gating_returns_reason() {
        let policy = BrowserSafetyPolicy::interactive();
        assert!(policy.is_action_allowed(&BrowserAction::NoOp, 0.0));

        let type_action = BrowserAction::Type {
            selector: crate::actions::ElementSelector::Css("input".into()),
            text: "test".into(),
        };
        let decision = policy.evaluate_action(&type_action, 0.3);
        assert_eq!(
            decision.denial_reason(),
            Some(&PolicyDenial::InsufficientPhi)
        );
        assert!(policy.is_action_allowed(&type_action, 0.6));
    }

    #[test]
    fn non_finite_phi_fails_closed() {
        let policy = BrowserSafetyPolicy::default();
        assert_eq!(
            policy.evaluate_action(&BrowserAction::NoOp, f64::NAN),
            PolicyDecision::Deny {
                reason: PolicyDenial::InvalidPhi,
                required_phi: None,
            }
        );
        assert!(!policy.is_action_allowed(&BrowserAction::NoOp, f64::INFINITY));
    }

    #[test]
    fn invalid_policy_fails_closed() {
        let policy = BrowserSafetyPolicy {
            phi_multiplier: f64::NAN,
            ..Default::default()
        };
        assert_eq!(
            policy.evaluate_action(&BrowserAction::NoOp, 0.5),
            PolicyDecision::Deny {
                reason: PolicyDenial::InvalidPolicy,
                required_phi: None,
            }
        );
    }

    #[test]
    fn navigate_checks_real_url() {
        let policy = BrowserSafetyPolicy::default();
        let nav = BrowserAction::Navigate {
            url: "file:///etc/passwd".into(),
        };
        assert!(!policy.is_action_allowed(&nav, 1.0));
    }

    #[test]
    fn phi_cannot_create_missing_authority() {
        let policy = BrowserSafetyPolicy::default();
        let click = BrowserAction::Click {
            selector: crate::actions::ElementSelector::Css("button".into()),
        };
        assert_eq!(
            policy.evaluate_action(&click, 1.0).denial_reason(),
            Some(&PolicyDenial::MissingCapability)
        );

        let mut interactive = policy;
        interactive.grant(BrowserCapability::Interact);
        assert!(interactive.is_action_allowed(&click, 1.0));
    }
}
