// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Solarpunk color palette for the mycelial colonization animation.
///
/// Colors drawn from the Pulse dashboard aesthetic — earth tones, living greens,
/// solar golds. Each color is RGBA (u8, u8, u8, u8).

/// Deep moss background — the dark substrate from which mycelium emerges.
pub const MOSS_DEEP: Rgba = Rgba(0x1a, 0x2e, 0x22, 0xff);

/// Lichen grey — subtle structural elements, dormant threads.
pub const LICHEN_GREY: Rgba = Rgba(0x5a, 0x6b, 0x5e, 0xff);

/// Clay — warm earth tone for secondary structures.
pub const CLAY: Rgba = Rgba(0x8b, 0x6f, 0x4e, 0xff);

/// Leaf green — primary mycelial thread color, alive and growing.
pub const LEAF_GREEN: Rgba = Rgba(0x7e, 0xc8, 0xa0, 0xff);

/// Solar gold — node pulse color, fired on derivation completion events.
pub const SOLAR_GOLD: Rgba = Rgba(0xe8, 0xc5, 0x47, 0xff);

/// Mycelial white — final convergence flash, pure completion.
pub const MYCELIAL_WHITE: Rgba = Rgba(0xf0, 0xf0, 0xf0, 0xff);

/// Pure black — initial void.
pub const BLACK: Rgba = Rgba(0x00, 0x00, 0x00, 0xff);

/// RGBA pixel value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    /// Pack into a 32-bit value for the framebuffer (XRGB8888 format).
    /// DRM dumb buffers typically use XRGB8888: [B, G, R, X] in memory (little-endian u32).
    #[inline]
    pub fn to_xrgb8888(self) -> u32 {
        (self.0 as u32) << 16 | (self.1 as u32) << 8 | (self.2 as u32)
    }

    /// Linear interpolation between two colors.
    /// `t` is clamped to [0.0, 1.0].
    pub fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Rgba(
            (a.0 as f32 * inv + b.0 as f32 * t) as u8,
            (a.1 as f32 * inv + b.1 as f32 * t) as u8,
            (a.2 as f32 * inv + b.2 as f32 * t) as u8,
            (a.3 as f32 * inv + b.3 as f32 * t) as u8,
        )
    }

    /// Return a copy with modified alpha (opacity 0.0-1.0).
    pub fn with_opacity(self, opacity: f32) -> Rgba {
        Rgba(
            self.0,
            self.1,
            self.2,
            (opacity.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    /// Scale brightness by a factor (0.0 = black, 1.0 = unchanged, >1.0 = brighter).
    pub fn brighten(self, factor: f32) -> Rgba {
        Rgba(
            (self.0 as f32 * factor).min(255.0) as u8,
            (self.1 as f32 * factor).min(255.0) as u8,
            (self.2 as f32 * factor).min(255.0) as u8,
            self.3,
        )
    }

    /// Alpha-composite `self` over `dst`.
    pub fn over(self, dst: Rgba) -> Rgba {
        let sa = self.3 as f32 / 255.0;
        let da = dst.3 as f32 / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a < 0.001 {
            return Rgba(0, 0, 0, 0);
        }
        let blend =
            |s: u8, d: u8| -> u8 { ((s as f32 * sa + d as f32 * da * (1.0 - sa)) / out_a) as u8 };
        Rgba(
            blend(self.0, dst.0),
            blend(self.1, dst.1),
            blend(self.2, dst.2),
            (out_a * 255.0) as u8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_endpoints() {
        assert_eq!(Rgba::lerp(BLACK, MYCELIAL_WHITE, 0.0), BLACK);
        assert_eq!(Rgba::lerp(BLACK, MYCELIAL_WHITE, 1.0), MYCELIAL_WHITE);
    }

    #[test]
    fn test_xrgb8888_packing() {
        let red = Rgba(0xff, 0x00, 0x00, 0xff);
        assert_eq!(red.to_xrgb8888(), 0x00ff0000);
        let green = Rgba(0x00, 0xff, 0x00, 0xff);
        assert_eq!(green.to_xrgb8888(), 0x0000ff00);
        let blue = Rgba(0x00, 0x00, 0xff, 0xff);
        assert_eq!(blue.to_xrgb8888(), 0x000000ff);
    }

    #[test]
    fn test_brighten_clamps() {
        let bright = MYCELIAL_WHITE.brighten(2.0);
        assert_eq!(bright.0, 255);
        assert_eq!(bright.1, 255);
        assert_eq!(bright.2, 255);
    }

    #[test]
    fn test_over_opaque_replaces() {
        let result = LEAF_GREEN.over(MOSS_DEEP);
        // Fully opaque source should replace destination
        assert_eq!(result, LEAF_GREEN);
    }

    #[test]
    fn test_over_transparent_preserves() {
        let transparent = Rgba(0xff, 0x00, 0x00, 0x00);
        let result = transparent.over(MOSS_DEEP);
        assert_eq!(result, MOSS_DEEP);
    }
}
