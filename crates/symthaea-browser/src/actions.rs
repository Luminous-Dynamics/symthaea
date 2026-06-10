// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser actions with Phi-gated safety thresholds.
//!
//! Each action declares a minimum consciousness level (`required_phi`) that
//! must be met before the cognitive loop is allowed to dispatch it. Read-only
//! actions (extract text, screenshot) have low thresholds; mutating actions
//! (click, type, navigate) require higher Phi.

use serde::{Deserialize, Serialize};

/// Selector for targeting a DOM element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementSelector {
    /// CDP backend node ID (most reliable after accessibility tree query).
    BackendNodeId(i64),
    /// CSS selector string.
    Css(String),
    /// Accessibility-based selector: match by ARIA role and name.
    Accessible { role: String, name: String },
}

/// A browser action that the cognitive loop can dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrowserAction {
    /// Navigate to a URL.
    Navigate { url: String },
    /// Click an element.
    Click { selector: ElementSelector },
    /// Type text into an element (focus + key events).
    Type {
        selector: ElementSelector,
        text: String,
    },
    /// Scroll an element into view.
    ScrollTo { selector: ElementSelector },
    /// Navigate back in history.
    GoBack,
    /// Navigate forward in history.
    GoForward,
    /// Extract text content from the page or a specific element.
    ExtractText { selector: Option<String> },
    /// Capture a screenshot (PNG bytes).
    Screenshot,
    /// Do nothing this cycle (conscious restraint or low Phi).
    NoOp,
}

impl BrowserAction {
    /// Minimum Phi required to execute this action.
    ///
    /// Higher values demand more conscious engagement before the system
    /// is allowed to perform the action. Scale: 0.0 (always allowed) to
    /// 1.0 (requires peak consciousness).
    ///
    /// | Action      | Phi   | Rationale                              |
    /// |-------------|-------|----------------------------------------|
    /// | NoOp        | 0.00  | Inaction is always safe                |
    /// | Screenshot  | 0.05  | Passive observation                    |
    /// | ExtractText | 0.10  | Read-only content extraction           |
    /// | GoBack/Fwd  | 0.20  | Minor navigation, reversible           |
    /// | ScrollTo    | 0.20  | View adjustment, no mutation           |
    /// | Navigate    | 0.30  | Page load — side effects possible      |
    /// | Click       | 0.40  | DOM mutation, form submission possible |
    /// | Type        | 0.50  | Content creation, highest agency       |
    pub fn required_phi(&self) -> f64 {
        match self {
            BrowserAction::NoOp => 0.00,
            BrowserAction::Screenshot => 0.05,
            BrowserAction::ExtractText { .. } => 0.10,
            BrowserAction::GoBack | BrowserAction::GoForward => 0.20,
            BrowserAction::ScrollTo { .. } => 0.20,
            BrowserAction::Navigate { .. } => 0.30,
            BrowserAction::Click { .. } => 0.40,
            BrowserAction::Type { .. } => 0.50,
        }
    }

    /// Whether this action only reads state (no side effects).
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            BrowserAction::NoOp
                | BrowserAction::Screenshot
                | BrowserAction::ExtractText { .. }
                | BrowserAction::ScrollTo { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_phi_ordering() {
        // Read-only actions should require less Phi than mutating ones.
        let noop = BrowserAction::NoOp;
        let screenshot = BrowserAction::Screenshot;
        let navigate = BrowserAction::Navigate {
            url: "https://example.com".into(),
        };
        let click = BrowserAction::Click {
            selector: ElementSelector::Css("button".into()),
        };
        let type_action = BrowserAction::Type {
            selector: ElementSelector::Css("input".into()),
            text: "hello".into(),
        };

        assert!(noop.required_phi() < screenshot.required_phi());
        assert!(screenshot.required_phi() < navigate.required_phi());
        assert!(navigate.required_phi() < click.required_phi());
        assert!(click.required_phi() < type_action.required_phi());
    }

    #[test]
    fn test_is_read_only() {
        assert!(BrowserAction::NoOp.is_read_only());
        assert!(BrowserAction::Screenshot.is_read_only());
        assert!(BrowserAction::ExtractText { selector: None }.is_read_only());
        assert!(!BrowserAction::Navigate { url: "x".into() }.is_read_only());
        assert!(
            !BrowserAction::Click {
                selector: ElementSelector::Css("a".into())
            }
            .is_read_only()
        );
        assert!(
            !BrowserAction::Type {
                selector: ElementSelector::Css("input".into()),
                text: "x".into()
            }
            .is_read_only()
        );
    }

    #[test]
    fn test_phi_thresholds_bounded() {
        // All thresholds must be in [0.0, 1.0].
        let actions = vec![
            BrowserAction::NoOp,
            BrowserAction::Screenshot,
            BrowserAction::ExtractText { selector: None },
            BrowserAction::GoBack,
            BrowserAction::GoForward,
            BrowserAction::ScrollTo {
                selector: ElementSelector::Css("div".into()),
            },
            BrowserAction::Navigate {
                url: "https://example.com".into(),
            },
            BrowserAction::Click {
                selector: ElementSelector::Css("a".into()),
            },
            BrowserAction::Type {
                selector: ElementSelector::Css("input".into()),
                text: "test".into(),
            },
        ];
        for action in &actions {
            let phi = action.required_phi();
            assert!(
                (0.0..=1.0).contains(&phi),
                "{:?} has out-of-range phi: {}",
                action,
                phi
            );
        }
    }
}
