// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser safety policy: URL filtering and Phi-gated action approval.
//!
//! The safety layer sits between the cognitive loop's action selection and
//! the CDP session. It enforces:
//! 1. URL allowlist/blocklist (domain-level filtering)
//! 2. Per-action Phi thresholds (consciousness-gated permissions)
//! 3. Global Phi floor (minimum consciousness for any action)

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::actions::BrowserAction;

/// Safety policy for browser actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserSafetyPolicy {
    /// If non-empty, only these URL patterns are allowed.
    /// Patterns are matched as domain suffixes (e.g. "example.com" matches
    /// "sub.example.com" and "example.com/path").
    pub url_allowlist: Vec<String>,

    /// These URL patterns are always blocked, even if on the allowlist.
    pub url_blocklist: Vec<String>,

    /// Global minimum Phi required for any action (including NoOp reads).
    /// Default: 0.0 (no floor).
    pub global_phi_floor: f64,

    /// Multiplier applied to each action's `required_phi()`.
    /// Values > 1.0 make the agent more cautious; < 1.0 more permissive.
    /// Default: 1.0.
    pub phi_multiplier: f64,
}

impl Default for BrowserSafetyPolicy {
    fn default() -> Self {
        Self {
            url_allowlist: Vec::new(),
            url_blocklist: vec![
                // Sensible defaults: block dangerous domains
                "chrome://".to_string(),
                "file://".to_string(),
                "javascript:".to_string(),
                "data:".to_string(),
            ],
            global_phi_floor: 0.0,
            phi_multiplier: 1.0,
        }
    }
}

impl BrowserSafetyPolicy {
    /// Check whether a URL is permitted under this policy.
    pub fn is_url_allowed(&self, url: &str) -> bool {
        let lower = url.to_lowercase();

        // Blocklist always wins
        for pattern in &self.url_blocklist {
            if lower.starts_with(&pattern.to_lowercase()) || lower.contains(&pattern.to_lowercase())
            {
                warn!(url, pattern, "URL blocked by blocklist");
                return false;
            }
        }

        // If allowlist is set, URL must match at least one pattern
        if !self.url_allowlist.is_empty() {
            let allowed = self.url_allowlist.iter().any(|pattern| {
                let p = pattern.to_lowercase();
                lower.contains(&p)
            });
            if !allowed {
                warn!(url, "URL not on allowlist");
                return false;
            }
        }

        true
    }

    /// Check whether an action is permitted given the current Phi level.
    pub fn is_action_allowed(&self, action: &BrowserAction, phi: f64) -> bool {
        let threshold = (action.required_phi() * self.phi_multiplier).max(self.global_phi_floor);
        if phi < threshold {
            warn!(?action, phi, threshold, "Action blocked: insufficient Phi");
            return false;
        }

        // URL-specific checks for navigation
        if let BrowserAction::Navigate { url } = action {
            return self.is_url_allowed(url);
        }

        true
    }

    /// Create a restrictive policy that only allows specified domains.
    pub fn allowlist_only(domains: Vec<String>) -> Self {
        Self {
            url_allowlist: domains,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocklist_default() {
        let policy = BrowserSafetyPolicy::default();
        assert!(!policy.is_url_allowed("file:///etc/passwd"));
        assert!(!policy.is_url_allowed("javascript:alert(1)"));
        assert!(!policy.is_url_allowed("chrome://settings"));
        assert!(policy.is_url_allowed("https://example.com"));
    }

    #[test]
    fn test_allowlist_only() {
        let policy = BrowserSafetyPolicy::allowlist_only(vec!["example.com".into()]);
        assert!(policy.is_url_allowed("https://example.com/page"));
        assert!(policy.is_url_allowed("https://sub.example.com"));
        assert!(!policy.is_url_allowed("https://evil.com"));
    }

    #[test]
    fn test_action_phi_gating() {
        let policy = BrowserSafetyPolicy::default();

        // NoOp always allowed (phi=0.0)
        assert!(policy.is_action_allowed(&BrowserAction::NoOp, 0.0));

        // Type requires phi=0.50
        let type_action = BrowserAction::Type {
            selector: crate::actions::ElementSelector::Css("input".into()),
            text: "test".into(),
        };
        assert!(!policy.is_action_allowed(&type_action, 0.3));
        assert!(policy.is_action_allowed(&type_action, 0.6));
    }

    #[test]
    fn test_phi_multiplier() {
        let policy = BrowserSafetyPolicy {
            phi_multiplier: 2.0,
            ..Default::default()
        };

        // Click normally requires 0.40, but with 2x multiplier needs 0.80
        let click = BrowserAction::Click {
            selector: crate::actions::ElementSelector::Css("button".into()),
        };
        assert!(!policy.is_action_allowed(&click, 0.6));
        assert!(policy.is_action_allowed(&click, 0.9));
    }

    #[test]
    fn test_navigate_checks_url() {
        let policy = BrowserSafetyPolicy::default();
        let nav = BrowserAction::Navigate {
            url: "file:///etc/passwd".into(),
        };
        // Even with sufficient Phi, blocked URL is blocked
        assert!(!policy.is_action_allowed(&nav, 1.0));
    }
}
