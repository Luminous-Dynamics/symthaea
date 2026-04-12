// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phone action types with Phi-gated safety thresholds.

/// Actions the phone embodiment can take via ADB.
#[derive(Debug, Clone)]
pub enum PhoneAction {
    /// No action (always safe).
    NoOp,
    /// Capture a screenshot (passive observation).
    Screenshot,
    /// Press the back button.
    Back,
    /// Press the home button.
    Home,
    /// Open a URL in the default browser.
    OpenUrl { url: String },
    /// Swipe gesture.
    Swipe {
        x1: u32,
        y1: u32,
        x2: u32,
        y2: u32,
        duration_ms: u32,
    },
    /// Tap at screen coordinates.
    Tap { x: u32, y: u32 },
    /// Type text into the focused field.
    Type { text: String },
}

impl PhoneAction {
    /// Minimum consciousness level (Phi) required for this action.
    ///
    /// Graduated from passive observation (0.05) to content creation (0.50).
    /// Follows the same pattern as `BrowserAction` in symthaea-browser.
    pub fn required_phi(&self) -> f64 {
        match self {
            Self::NoOp => 0.0,
            Self::Screenshot => 0.05,
            Self::Back | Self::Home => 0.20,
            Self::OpenUrl { .. } => 0.30,
            Self::Swipe { .. } => 0.35,
            Self::Tap { .. } => 0.40,
            Self::Type { .. } => 0.50,
        }
    }

    /// Whether this action mutates the phone state.
    pub fn is_mutating(&self) -> bool {
        !matches!(self, Self::NoOp | Self::Screenshot)
    }

    /// Human-readable label for confirmation dialogs.
    pub fn label(&self) -> String {
        match self {
            Self::NoOp => "no action".into(),
            Self::Screenshot => "capture screenshot".into(),
            Self::Back => "press back".into(),
            Self::Home => "press home".into(),
            Self::OpenUrl { url } => format!("open URL: {url}"),
            Self::Swipe { x1, y1, x2, y2, .. } => {
                format!("swipe ({x1},{y1})→({x2},{y2})")
            }
            Self::Tap { x, y } => format!("tap at ({x},{y})"),
            Self::Type { text } => format!("type: \"{text}\""),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_thresholds_are_monotonic() {
        // More impactful actions require higher Phi
        assert!(PhoneAction::NoOp.required_phi() < PhoneAction::Screenshot.required_phi());
        assert!(PhoneAction::Screenshot.required_phi() < PhoneAction::Back.required_phi());
        assert!(
            PhoneAction::Back.required_phi()
                < PhoneAction::OpenUrl { url: String::new() }.required_phi()
        );
        assert!(
            PhoneAction::OpenUrl { url: String::new() }.required_phi()
                < PhoneAction::Tap { x: 0, y: 0 }.required_phi()
        );
        assert!(
            PhoneAction::Tap { x: 0, y: 0 }.required_phi()
                < PhoneAction::Type {
                    text: String::new()
                }
                .required_phi()
        );
    }

    #[test]
    fn test_screenshot_is_not_mutating() {
        assert!(!PhoneAction::Screenshot.is_mutating());
        assert!(!PhoneAction::NoOp.is_mutating());
    }

    #[test]
    fn test_tap_is_mutating() {
        assert!(PhoneAction::Tap { x: 100, y: 200 }.is_mutating());
        assert!(PhoneAction::Type {
            text: "hello".into()
        }
        .is_mutating());
    }
}
