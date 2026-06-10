// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! G2: Theming System
//!
//! Color schemes and styling for the Symthaea shell and GUI.

use std::collections::HashMap;

/// ANSI color code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create from hex string (e.g., "#ff0000" or "ff0000")
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;

        Some(Self { r, g, b })
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Get ANSI escape sequence for foreground
    pub fn to_ansi_fg(&self) -> String {
        format!("\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }

    /// Get ANSI escape sequence for background
    pub fn to_ansi_bg(&self) -> String {
        format!("\x1b[48;2;{};{};{}m", self.r, self.g, self.b)
    }

    // Common colors
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    pub const RED: Color = Color::rgb(255, 0, 0);
    pub const GREEN: Color = Color::rgb(0, 255, 0);
    pub const BLUE: Color = Color::rgb(0, 0, 255);
    pub const YELLOW: Color = Color::rgb(255, 255, 0);
    pub const CYAN: Color = Color::rgb(0, 255, 255);
    pub const MAGENTA: Color = Color::rgb(255, 0, 255);
}

/// Semantic color roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorRole {
    /// Primary accent color
    Primary,
    /// Secondary accent color
    Secondary,
    /// Background
    Background,
    /// Surface (slightly elevated background)
    Surface,
    /// Primary text
    Text,
    /// Muted/secondary text
    TextMuted,
    /// Success/safe indicator
    Success,
    /// Warning indicator
    Warning,
    /// Error/danger indicator
    Error,
    /// Info indicator
    Info,
    /// High consciousness (Phi >= 0.7)
    ConsciousnessHigh,
    /// Medium consciousness (Phi >= 0.4)
    ConsciousnessMedium,
    /// Low consciousness (Phi < 0.4)
    ConsciousnessLow,
    /// Safe operation
    SafetyGreen,
    /// Caution needed
    SafetyYellow,
    /// Danger/blocked
    SafetyRed,
    /// Prompt text
    Prompt,
    /// Command input
    Input,
    /// Command output
    Output,
    /// Completion highlight
    CompletionHighlight,
    /// Selection highlight
    Selection,
    /// Cursor
    Cursor,
}

/// A complete color theme
#[derive(Debug, Clone)]
pub struct Theme {
    /// Theme name
    pub name: String,
    /// Description
    pub description: String,
    /// Color mappings
    pub colors: HashMap<ColorRole, Color>,
    /// Whether this is a dark theme
    pub is_dark: bool,
}

impl Theme {
    /// Create a new theme
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            colors: HashMap::new(),
            is_dark: true,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set color for role
    pub fn with_color(mut self, role: ColorRole, color: Color) -> Self {
        self.colors.insert(role, color);
        self
    }

    /// Set whether dark theme
    pub fn dark(mut self, is_dark: bool) -> Self {
        self.is_dark = is_dark;
        self
    }

    /// Get color for role (with fallback)
    pub fn get(&self, role: ColorRole) -> Color {
        self.colors
            .get(&role)
            .copied()
            .unwrap_or_else(|| self.fallback(role))
    }

    fn fallback(&self, role: ColorRole) -> Color {
        use ColorRole::*;
        match role {
            Primary => Color::rgb(100, 149, 237),  // Cornflower blue
            Secondary => Color::rgb(138, 43, 226), // Blue violet
            Background => {
                if self.is_dark {
                    Color::rgb(30, 30, 30)
                } else {
                    Color::rgb(250, 250, 250)
                }
            }
            Surface => {
                if self.is_dark {
                    Color::rgb(45, 45, 45)
                } else {
                    Color::rgb(240, 240, 240)
                }
            }
            Text => {
                if self.is_dark {
                    Color::rgb(220, 220, 220)
                } else {
                    Color::rgb(30, 30, 30)
                }
            }
            TextMuted => {
                if self.is_dark {
                    Color::rgb(150, 150, 150)
                } else {
                    Color::rgb(100, 100, 100)
                }
            }
            Success | ConsciousnessHigh | SafetyGreen => Color::rgb(50, 205, 50),
            Warning | ConsciousnessMedium | SafetyYellow => Color::rgb(255, 200, 0),
            Error | ConsciousnessLow | SafetyRed => Color::rgb(220, 50, 50),
            Info => Color::rgb(30, 144, 255),
            Prompt => self.get(Primary),
            Input => self.get(Text),
            Output => self.get(TextMuted),
            CompletionHighlight => Color::rgb(70, 130, 180),
            Selection => Color::rgb(60, 60, 100),
            Cursor => self.get(Primary),
        }
    }

    /// Get ANSI escape for role foreground
    pub fn fg(&self, role: ColorRole) -> String {
        self.get(role).to_ansi_fg()
    }

    /// Get ANSI escape for role background
    pub fn bg(&self, role: ColorRole) -> String {
        self.get(role).to_ansi_bg()
    }

    /// ANSI reset sequence
    pub fn reset() -> &'static str {
        "\x1b[0m"
    }
}

/// Built-in themes
pub struct Themes;

impl Themes {
    /// Default dark theme - Luminous
    pub fn luminous() -> Theme {
        Theme::new("Luminous")
            .with_description("Default dark theme with consciousness-aware coloring")
            .dark(true)
            .with_color(ColorRole::Primary, Color::rgb(138, 180, 248)) // Light blue
            .with_color(ColorRole::Secondary, Color::rgb(187, 134, 252)) // Light purple
            .with_color(ColorRole::Background, Color::rgb(18, 18, 18))
            .with_color(ColorRole::Surface, Color::rgb(30, 30, 30))
            .with_color(ColorRole::Text, Color::rgb(232, 234, 237))
            .with_color(ColorRole::TextMuted, Color::rgb(154, 160, 166))
            .with_color(ColorRole::ConsciousnessHigh, Color::rgb(129, 201, 149))
            .with_color(ColorRole::ConsciousnessMedium, Color::rgb(251, 188, 4))
            .with_color(ColorRole::ConsciousnessLow, Color::rgb(242, 139, 130))
            .with_color(ColorRole::SafetyGreen, Color::rgb(129, 201, 149))
            .with_color(ColorRole::SafetyYellow, Color::rgb(251, 188, 4))
            .with_color(ColorRole::SafetyRed, Color::rgb(242, 139, 130))
            .with_color(ColorRole::Prompt, Color::rgb(138, 180, 248))
            .with_color(ColorRole::CompletionHighlight, Color::rgb(66, 133, 244))
    }

    /// Light theme
    pub fn daylight() -> Theme {
        Theme::new("Daylight")
            .with_description("Light theme for daytime use")
            .dark(false)
            .with_color(ColorRole::Primary, Color::rgb(25, 118, 210))
            .with_color(ColorRole::Secondary, Color::rgb(123, 31, 162))
            .with_color(ColorRole::Background, Color::rgb(255, 255, 255))
            .with_color(ColorRole::Surface, Color::rgb(245, 245, 245))
            .with_color(ColorRole::Text, Color::rgb(32, 33, 36))
            .with_color(ColorRole::TextMuted, Color::rgb(95, 99, 104))
            .with_color(ColorRole::ConsciousnessHigh, Color::rgb(52, 168, 83))
            .with_color(ColorRole::ConsciousnessMedium, Color::rgb(251, 188, 4))
            .with_color(ColorRole::ConsciousnessLow, Color::rgb(234, 67, 53))
    }

    /// Nord-inspired theme
    pub fn aurora() -> Theme {
        Theme::new("Aurora")
            .with_description("Nord-inspired arctic theme")
            .dark(true)
            .with_color(ColorRole::Primary, Color::rgb(136, 192, 208)) // Nord frost
            .with_color(ColorRole::Secondary, Color::rgb(180, 142, 173)) // Nord aurora purple
            .with_color(ColorRole::Background, Color::rgb(46, 52, 64)) // Nord polar night
            .with_color(ColorRole::Surface, Color::rgb(59, 66, 82))
            .with_color(ColorRole::Text, Color::rgb(236, 239, 244))
            .with_color(ColorRole::TextMuted, Color::rgb(216, 222, 233))
            .with_color(ColorRole::Success, Color::rgb(163, 190, 140)) // Nord green
            .with_color(ColorRole::Warning, Color::rgb(235, 203, 139)) // Nord yellow
            .with_color(ColorRole::Error, Color::rgb(191, 97, 106)) // Nord red
    }

    /// High contrast theme
    pub fn contrast() -> Theme {
        Theme::new("Contrast")
            .with_description("High contrast accessibility theme")
            .dark(true)
            .with_color(ColorRole::Primary, Color::rgb(255, 255, 0)) // Pure yellow
            .with_color(ColorRole::Secondary, Color::rgb(0, 255, 255)) // Pure cyan
            .with_color(ColorRole::Background, Color::rgb(0, 0, 0)) // Pure black
            .with_color(ColorRole::Surface, Color::rgb(20, 20, 20))
            .with_color(ColorRole::Text, Color::rgb(255, 255, 255)) // Pure white
            .with_color(ColorRole::TextMuted, Color::rgb(200, 200, 200))
            .with_color(ColorRole::Success, Color::rgb(0, 255, 0))
            .with_color(ColorRole::Warning, Color::rgb(255, 255, 0))
            .with_color(ColorRole::Error, Color::rgb(255, 0, 0))
    }

    /// Solarized dark
    pub fn solarized_dark() -> Theme {
        Theme::new("Solarized Dark")
            .with_description("Solarized dark color scheme")
            .dark(true)
            .with_color(ColorRole::Primary, Color::rgb(38, 139, 210)) // Blue
            .with_color(ColorRole::Secondary, Color::rgb(108, 113, 196)) // Violet
            .with_color(ColorRole::Background, Color::rgb(0, 43, 54)) // Base03
            .with_color(ColorRole::Surface, Color::rgb(7, 54, 66)) // Base02
            .with_color(ColorRole::Text, Color::rgb(131, 148, 150)) // Base0
            .with_color(ColorRole::TextMuted, Color::rgb(88, 110, 117)) // Base01
            .with_color(ColorRole::Success, Color::rgb(133, 153, 0)) // Green
            .with_color(ColorRole::Warning, Color::rgb(181, 137, 0)) // Yellow
            .with_color(ColorRole::Error, Color::rgb(220, 50, 47)) // Red
    }

    /// Get all built-in themes
    pub fn all() -> Vec<Theme> {
        vec![
            Self::luminous(),
            Self::daylight(),
            Self::aurora(),
            Self::contrast(),
            Self::solarized_dark(),
        ]
    }
}

/// Theme manager for runtime theme switching
pub struct ThemeManager {
    /// Current theme
    current: Theme,
    /// Available themes
    themes: HashMap<String, Theme>,
}

impl ThemeManager {
    /// Create with default theme
    pub fn new() -> Self {
        let mut themes = HashMap::new();
        for theme in Themes::all() {
            themes.insert(theme.name.to_lowercase(), theme);
        }

        Self {
            current: Themes::luminous(),
            themes,
        }
    }

    /// Get current theme
    pub fn current(&self) -> &Theme {
        &self.current
    }

    /// Set theme by name
    pub fn set_theme(&mut self, name: &str) -> bool {
        if let Some(theme) = self.themes.get(&name.to_lowercase()) {
            self.current = theme.clone();
            true
        } else {
            false
        }
    }

    /// Add custom theme
    pub fn add_theme(&mut self, theme: Theme) {
        self.themes.insert(theme.name.to_lowercase(), theme);
    }

    /// List available themes
    pub fn list_themes(&self) -> Vec<&str> {
        self.themes.keys().map(|s| s.as_str()).collect()
    }

    /// Get color for role
    pub fn get(&self, role: ColorRole) -> Color {
        self.current.get(role)
    }

    /// Get ANSI foreground for role
    pub fn fg(&self, role: ColorRole) -> String {
        self.current.fg(role)
    }

    /// Get ANSI background for role
    pub fn bg(&self, role: ColorRole) -> String {
        self.current.bg(role)
    }

    /// Format consciousness indicator with theme colors
    pub fn format_phi(&self, phi: f64) -> String {
        let role = if phi >= 0.7 {
            ColorRole::ConsciousnessHigh
        } else if phi >= 0.4 {
            ColorRole::ConsciousnessMedium
        } else {
            ColorRole::ConsciousnessLow
        };

        format!("{}Φ:{:.2}{}", self.fg(role), phi, Theme::reset())
    }

    /// Format safety level with theme colors
    pub fn format_safety(&self, level: &str) -> String {
        let role = match level.to_uppercase().as_str() {
            "GREEN" | "SAFE" => ColorRole::SafetyGreen,
            "YELLOW" | "CAUTION" => ColorRole::SafetyYellow,
            "RED" | "DANGER" | "BLOCKED" => ColorRole::SafetyRed,
            _ => ColorRole::Text,
        };

        format!("{}{}{}", self.fg(role), level, Theme::reset())
    }
}

impl Default for ThemeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#ff0000").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);

        let color = Color::from_hex("00ff00").unwrap();
        assert_eq!(color.g, 255);
    }

    #[test]
    fn test_color_to_hex() {
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.to_hex(), "#ff8040");
    }

    #[test]
    fn test_theme_colors() {
        let theme = Themes::luminous();
        let primary = theme.get(ColorRole::Primary);
        assert_eq!(primary, Color::rgb(138, 180, 248));
    }

    #[test]
    fn test_theme_manager() {
        let mut mgr = ThemeManager::new();

        assert!(mgr.list_themes().contains(&"luminous"));
        assert!(mgr.set_theme("aurora"));
        assert_eq!(mgr.current().name, "Aurora");
    }

    #[test]
    fn test_phi_formatting() {
        let mgr = ThemeManager::new();

        let high = mgr.format_phi(0.85);
        assert!(high.contains("0.85"));

        let low = mgr.format_phi(0.2);
        assert!(low.contains("0.20"));
    }
}
