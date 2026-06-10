// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Color primitives: HSL↔RGB conversion, neuromodulator-driven palette generation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// RGBA color with f32 components in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create from HSL (h in [0,360), s/l in [0,1]).
    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        Self::from_hsla(h, s, l, 1.0)
    }

    /// Create from HSLA (h in [0,360), s/l/a in [0,1]).
    pub fn from_hsla(h: f32, s: f32, l: f32, a: f32) -> Self {
        let h = h.rem_euclid(360.0);
        let s = s.clamp(0.0, 1.0);
        let l = l.clamp(0.0, 1.0);

        if s == 0.0 {
            return Self {
                r: l,
                g: l,
                b: l,
                a,
            };
        }

        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;
        let h_norm = h / 360.0;

        Self {
            r: hue_to_rgb(p, q, h_norm + 1.0 / 3.0),
            g: hue_to_rgb(p, q, h_norm),
            b: hue_to_rgb(p, q, h_norm - 1.0 / 3.0),
            a,
        }
    }

    /// Convert to HSL. Returns (h in [0,360), s in [0,1], l in [0,1]).
    pub fn to_hsl(&self) -> (f32, f32, f32) {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;

        if (max - min).abs() < 1e-6 {
            return (0.0, 0.0, l);
        }

        let d = max - min;
        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };

        let h = if (max - self.r).abs() < 1e-6 {
            let mut h = (self.g - self.b) / d;
            if self.g < self.b {
                h += 6.0;
            }
            h
        } else if (max - self.g).abs() < 1e-6 {
            (self.b - self.r) / d + 2.0
        } else {
            (self.r - self.g) / d + 4.0
        };

        (h * 60.0, s, l)
    }

    /// Hex string like "#e8c547" or "rgba(232,197,71,0.5)" if alpha < 1.
    pub fn to_css(&self) -> String {
        if (self.a - 1.0).abs() < 1e-3 {
            format!(
                "#{:02x}{:02x}{:02x}",
                (self.r * 255.0).round() as u8,
                (self.g * 255.0).round() as u8,
                (self.b * 255.0).round() as u8,
            )
        } else {
            format!(
                "rgba({},{},{},{:.2})",
                (self.r * 255.0).round() as u8,
                (self.g * 255.0).round() as u8,
                (self.b * 255.0).round() as u8,
                self.a,
            )
        }
    }

    /// Linearly interpolate toward `other` by `t` in [0,1].
    pub fn lerp(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a.clamp(0.0, 1.0);
        self
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_css())
    }
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// A 4-color palette derived from the neuromodulator bath.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub ambient: Color,
}

impl Palette {
    /// Derive palette from neuromodulator levels (all in [0,1]).
    pub fn from_neuromodulators(
        dopamine: f32,
        serotonin: f32,
        noradrenaline: f32,
        oxytocin: f32,
        gaba: f32,
    ) -> Self {
        let da = dopamine.clamp(0.0, 1.0);
        let ht = serotonin.clamp(0.0, 1.0);
        let ne = noradrenaline.clamp(0.0, 1.0);
        let oxy = oxytocin.clamp(0.0, 1.0);
        let gaba_c = gaba.clamp(0.0, 1.0);
        let calm = (oxy + gaba_c) / 2.0;

        Self {
            // Dopamine → gold
            primary: Color::from_hsl(45.0, 0.8, 0.4 + da * 0.2),
            // Serotonin → warm amber
            secondary: Color::from_hsl(25.0 + ht * 15.0, 0.7, 0.5),
            // Noradrenaline → cool blue
            accent: Color::from_hsl(200.0, 0.6 + ne * 0.3, 0.5),
            // Oxytocin+GABA → lavender
            ambient: Color::from_hsl(300.0, 0.3, 0.6 + calm * 0.2),
        }
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self::from_neuromodulators(0.5, 0.5, 0.5, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_rgb_roundtrip() {
        let cases = [
            (0.0, 1.0, 0.5),   // red
            (120.0, 1.0, 0.5), // green
            (240.0, 1.0, 0.5), // blue
            (45.0, 0.8, 0.5),  // gold
            (0.0, 0.0, 0.5),   // grey
        ];
        for (h, s, l) in cases {
            let c = Color::from_hsl(h, s, l);
            let (h2, s2, l2) = c.to_hsl();
            if s > 0.01 {
                // Handle 0°/360° wrap-around
                let h_diff = (h - h2).abs().min(360.0 - (h - h2).abs());
                assert!(h_diff < 1.0, "h mismatch: {h} vs {h2}");
            }
            assert!((s - s2).abs() < 0.02, "s mismatch: {s} vs {s2}");
            assert!((l - l2).abs() < 0.02, "l mismatch: {l} vs {l2}");
        }
    }

    #[test]
    fn css_hex_output() {
        let white = Color::rgb(1.0, 1.0, 1.0);
        assert_eq!(white.to_css(), "#ffffff");
        let black = Color::rgb(0.0, 0.0, 0.0);
        assert_eq!(black.to_css(), "#000000");
    }

    #[test]
    fn css_rgba_output() {
        let c = Color::rgba(1.0, 0.0, 0.0, 0.5);
        assert!(c.to_css().starts_with("rgba("));
        assert!(c.to_css().contains("0.50"));
    }

    #[test]
    fn lerp_endpoints() {
        let a = Color::rgb(0.0, 0.0, 0.0);
        let b = Color::rgb(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
    }

    #[test]
    fn palette_from_neuromodulators_ranges() {
        let p = Palette::from_neuromodulators(0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(p.primary.r >= 0.0 && p.primary.r <= 1.0);
        let p2 = Palette::from_neuromodulators(1.0, 1.0, 1.0, 1.0, 1.0);
        // High dopamine → brighter primary
        assert!(p2.primary.g > p.primary.g || p2.primary.r > p.primary.r);
    }

    #[test]
    fn palette_default_is_mid() {
        let p = Palette::default();
        // Should produce valid colors
        assert!(p.primary.r >= 0.0 && p.primary.r <= 1.0);
        assert!(p.accent.b >= 0.0 && p.accent.b <= 1.0);
    }
}
