// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Page observation model: accessibility tree snapshot for the cognitive loop.
//!
//! The accessibility tree provides a semantically rich, vision-independent
//! representation of page content. Each element carries role, name, value,
//! and state — enough for the HDC encoder to build a consciousness-compatible
//! hypervector without needing pixel-level rendering.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single accessible element from the page's accessibility tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibleElement {
    /// CDP backend node ID for reliable targeting.
    pub backend_node_id: i64,
    /// ARIA role (e.g. "button", "link", "textbox", "heading").
    pub role: String,
    /// Accessible name (visible label or aria-label).
    pub name: String,
    /// Current value (for inputs, selects, etc.).
    pub value: Option<String>,
    /// Accessible description (aria-describedby, title, etc.).
    pub description: Option<String>,
    /// Whether this element currently has focus.
    pub focused: bool,
    /// Whether this element is disabled.
    pub disabled: bool,
}

impl AccessibleElement {
    /// Render this element as a compact text line for cognitive processing.
    ///
    /// Format: `[role] "name" (value) {focused} {disabled}`
    pub fn to_text_line(&self) -> String {
        let mut parts = vec![format!("[{}]", self.role)];
        if !self.name.is_empty() {
            parts.push(format!("\"{}\"", self.name));
        }
        if let Some(ref v) = self.value {
            if !v.is_empty() {
                parts.push(format!("({})", v));
            }
        }
        if self.focused {
            parts.push("{focused}".to_string());
        }
        if self.disabled {
            parts.push("{disabled}".to_string());
        }
        parts.join(" ")
    }
}

/// Snapshot of a page's state as seen through the accessibility tree.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageObservation {
    /// Current page URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// Accessible elements from the tree (flattened, depth-first order).
    pub elements: Vec<AccessibleElement>,
    /// Index into `elements` of the currently focused element, if any.
    pub focused_element: Option<usize>,
}

impl PageObservation {
    /// Render the observation as structured text for the cognitive loop.
    ///
    /// This is the primary interface between the browser and Symthaea's
    /// language/reasoning systems. Format:
    ///
    /// ```text
    /// PAGE: <title>
    /// URL: <url>
    /// FOCUS: <index>
    /// ELEMENTS:
    ///   0: [button] "Submit" {focused}
    ///   1: [textbox] "Search" (current value)
    ///   ...
    /// ```
    pub fn to_cognitive_text(&self) -> String {
        let mut lines = Vec::with_capacity(self.elements.len() + 4);
        lines.push(format!("PAGE: {}", self.title));
        lines.push(format!("URL: {}", self.url));
        match self.focused_element {
            Some(idx) => lines.push(format!("FOCUS: {}", idx)),
            None => lines.push("FOCUS: none".to_string()),
        }
        lines.push("ELEMENTS:".to_string());
        for (i, elem) in self.elements.iter().enumerate() {
            lines.push(format!("  {}: {}", i, elem.to_text_line()));
        }
        lines.join("\n")
    }

    /// Number of interactive elements (not disabled, with a role that
    /// suggests user interaction).
    pub fn interactive_count(&self) -> usize {
        self.elements
            .iter()
            .filter(|e| {
                !e.disabled
                    && matches!(
                        e.role.as_str(),
                        "button"
                            | "link"
                            | "textbox"
                            | "checkbox"
                            | "radio"
                            | "combobox"
                            | "menuitem"
                            | "tab"
                            | "slider"
                    )
            })
            .count()
    }
}

impl fmt::Display for PageObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_cognitive_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_observation() -> PageObservation {
        PageObservation {
            url: "https://example.com".into(),
            title: "Example Domain".into(),
            focused_element: Some(0),
            elements: vec![
                AccessibleElement {
                    backend_node_id: 1,
                    role: "button".into(),
                    name: "Submit".into(),
                    value: None,
                    description: None,
                    focused: true,
                    disabled: false,
                },
                AccessibleElement {
                    backend_node_id: 2,
                    role: "textbox".into(),
                    name: "Search".into(),
                    value: Some("hello".into()),
                    description: Some("Search the site".into()),
                    focused: false,
                    disabled: false,
                },
                AccessibleElement {
                    backend_node_id: 3,
                    role: "heading".into(),
                    name: "Welcome".into(),
                    value: None,
                    description: None,
                    focused: false,
                    disabled: false,
                },
            ],
        }
    }

    #[test]
    fn test_cognitive_text_formatting() {
        let obs = sample_observation();
        let text = obs.to_cognitive_text();

        assert!(text.contains("PAGE: Example Domain"));
        assert!(text.contains("URL: https://example.com"));
        assert!(text.contains("FOCUS: 0"));
        assert!(text.contains("[button] \"Submit\" {focused}"));
        assert!(text.contains("[textbox] \"Search\" (hello)"));
        assert!(text.contains("[heading] \"Welcome\""));
    }

    #[test]
    fn test_interactive_count() {
        let obs = sample_observation();
        // button + textbox = 2 interactive (heading is not interactive)
        assert_eq!(obs.interactive_count(), 2);
    }

    #[test]
    fn test_disabled_not_interactive() {
        let obs = PageObservation {
            url: "https://example.com".into(),
            title: "Test".into(),
            focused_element: None,
            elements: vec![AccessibleElement {
                backend_node_id: 1,
                role: "button".into(),
                name: "Disabled".into(),
                value: None,
                description: None,
                focused: false,
                disabled: true,
            }],
        };
        assert_eq!(obs.interactive_count(), 0);
    }
}
