// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tiny dependency-free bitmap microtype for factual early-boot labels.
//!
//! Spore intentionally does not pull a font stack into the DRM boot path. This
//! module provides only the compact uppercase alphabet and punctuation needed by
//! labels such as `SPORE`, `GERMINATION`, `RELIGHTING`, and `RECOVERY`.

use crate::color::Rgba;

const GLYPH_WIDTH: usize = 5;
const GLYPH_HEIGHT: usize = 7;

#[derive(Debug, Clone, Copy)]
pub struct TextMetrics {
    pub width: usize,
    pub height: usize,
}

pub fn measure(text: &str, scale: usize, tracking: usize) -> TextMetrics {
    let scale = scale.max(1);
    let glyphs = text.chars().count();
    let width = if glyphs == 0 {
        0
    } else {
        glyphs * GLYPH_WIDTH * scale + (glyphs - 1) * tracking
    };
    TextMetrics {
        width,
        height: GLYPH_HEIGHT * scale,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_text(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    scale: usize,
    tracking: usize,
    color: Rgba,
) {
    if width == 0 || height == 0 || buffer.len() < width * height {
        return;
    }
    let scale = scale.max(1);
    let mut cursor = x;
    for character in text.chars() {
        let glyph = glyph(character.to_ascii_uppercase());
        draw_glyph(buffer, width, height, cursor, y, glyph, scale, color);
        cursor = cursor.saturating_add(GLYPH_WIDTH * scale + tracking);
        if cursor >= width {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_glyph(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    rows: [u8; GLYPH_HEIGHT],
    scale: usize,
    color: Rgba,
) {
    for (row_index, row) in rows.iter().copied().enumerate() {
        for column in 0..GLYPH_WIDTH {
            let bit = 1u8 << (GLYPH_WIDTH - 1 - column);
            if row & bit == 0 {
                continue;
            }
            let px = x.saturating_add(column * scale);
            let py = y.saturating_add(row_index * scale);
            for oy in 0..scale {
                for ox in 0..scale {
                    let dx = px + ox;
                    let dy = py + oy;
                    if dx < width && dy < height {
                        blend_pixel(buffer, width, dx, dy, color);
                    }
                }
            }
        }
    }
}

fn blend_pixel(buffer: &mut [u32], width: usize, x: usize, y: usize, src: Rgba) {
    let index = y * width + x;
    let value = buffer[index];
    let dst = Rgba(
        ((value >> 16) & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        (value & 0xff) as u8,
        0xff,
    );
    buffer[index] = src.over(dst).to_xrgb8888();
}

fn glyph(c: char) -> [u8; GLYPH_HEIGHT] {
    match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '.' => [0, 0, 0, 0, 0, 0b00110, 0b00110],
        ':' => [0, 0b00110, 0b00110, 0, 0b00110, 0b00110, 0],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        ' ' => [0; GLYPH_HEIGHT],
        _ => [0; GLYPH_HEIGHT],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_tracks_scale_and_tracking() {
        let metrics = measure("SPORE", 2, 1);
        assert_eq!(metrics.width, 5 * 5 * 2 + 4);
        assert_eq!(metrics.height, 14);
    }

    #[test]
    fn rendering_is_deterministic_and_nonempty() {
        let mut a = vec![0u32; 160 * 90];
        let mut b = vec![0u32; 160 * 90];
        draw_text(
            &mut a,
            160,
            90,
            8,
            8,
            "SPORE",
            2,
            1,
            Rgba(255, 255, 255, 255),
        );
        draw_text(
            &mut b,
            160,
            90,
            8,
            8,
            "SPORE",
            2,
            1,
            Rgba(255, 255, 255, 255),
        );
        assert_eq!(a, b);
        assert!(a.iter().any(|pixel| *pixel != 0));
    }
}
