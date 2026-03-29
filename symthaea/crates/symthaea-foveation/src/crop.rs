// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pixel crop extraction from full-resolution frames using PatchGrid coordinates.
//!
//! Extracts rectangular regions from raw pixel buffers, with optional
//! context padding (1 patch in each direction) and bilinear downscaling
//! when crops exceed the configured pixel budget.

use crate::types::FrameBuffer;

/// Extract a pixel crop from a frame buffer at the given grid position.
///
/// The crop is centered on the patch at `(grid_row, grid_col)` and padded
/// by `pad` patches in each direction (clamped to frame edges).
///
/// # Arguments
/// * `frame` — Full-resolution frame buffer.
/// * `grid_row`, `grid_col` — Patch grid coordinates.
/// * `patch_size` — Size of each patch in pixels (square).
/// * `pad` — Number of extra patches to include as context margin.
/// * `max_crop_pixels` — Maximum total pixels before downscaling.
///
/// # Returns
/// `(pixels, width, height)` — the cropped (and possibly downscaled) pixel data.
pub fn extract_crop(
    frame: &FrameBuffer,
    grid_row: usize,
    grid_col: usize,
    patch_size: usize,
    pad: usize,
    max_crop_pixels: usize,
) -> (Vec<u8>, u32, u32) {
    let fw = frame.width as usize;
    let fh = frame.height as usize;
    let ch = frame.channels;

    // Compute padded bounding box in pixel space
    let x0 = grid_col.saturating_sub(pad) * patch_size;
    let y0 = grid_row.saturating_sub(pad) * patch_size;
    let x1 = ((grid_col + 1 + pad) * patch_size).min(fw);
    let y1 = ((grid_row + 1 + pad) * patch_size).min(fh);

    let crop_w = x1.saturating_sub(x0);
    let crop_h = y1.saturating_sub(y0);

    if crop_w == 0 || crop_h == 0 {
        return (Vec::new(), 0, 0);
    }

    // Extract raw crop
    let mut crop = Vec::with_capacity(crop_w * crop_h * ch);
    for row in y0..y1 {
        let row_start = (row * fw + x0) * ch;
        let row_end = (row * fw + x1) * ch;
        if row_end <= frame.pixels.len() {
            crop.extend_from_slice(&frame.pixels[row_start..row_end]);
        } else {
            // Partial row (shouldn't happen with valid frames, but be safe)
            let available = frame.pixels.len().saturating_sub(row_start);
            crop.extend_from_slice(&frame.pixels[row_start..row_start + available]);
            crop.resize(crop.len() + (row_end - row_start - available), 0);
        }
    }

    let total_pixels = crop_w * crop_h;
    if total_pixels > max_crop_pixels && max_crop_pixels > 0 {
        // Downscale using bilinear interpolation
        let scale = (max_crop_pixels as f32 / total_pixels as f32).sqrt();
        let new_w = ((crop_w as f32 * scale).round() as usize).max(1);
        let new_h = ((crop_h as f32 * scale).round() as usize).max(1);
        let downscaled = bilinear_downscale(&crop, crop_w, crop_h, ch, new_w, new_h);
        (downscaled, new_w as u32, new_h as u32)
    } else {
        (crop, crop_w as u32, crop_h as u32)
    }
}

/// Bilinear downscale of a pixel buffer.
///
/// Simple implementation with no external dependencies — adequate for
/// the moderate downscale ratios we encounter (typically 2–4x).
fn bilinear_downscale(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * channels];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel to source coordinates
            let sx = (dx as f32 + 0.5) * (src_w as f32 / dst_w as f32) - 0.5;
            let sy = (dy as f32 + 0.5) * (src_h as f32 / dst_h as f32) - 0.5;

            let x0 = (sx.floor() as isize).max(0) as usize;
            let y0 = (sy.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let fx = fx.clamp(0.0, 1.0);
            let fy = fy.clamp(0.0, 1.0);

            for c in 0..channels {
                let v00 = src[(y0 * src_w + x0) * channels + c] as f32;
                let v10 = src[(y0 * src_w + x1) * channels + c] as f32;
                let v01 = src[(y1 * src_w + x0) * channels + c] as f32;
                let v11 = src[(y1 * src_w + x1) * channels + c] as f32;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                dst[(dy * dst_w + dx) * channels + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gray_frame(w: u32, h: u32, val: u8) -> FrameBuffer {
        FrameBuffer {
            pixels: vec![val; (w * h) as usize],
            width: w,
            height: h,
            channels: 1,
            frame_id: 0,
            timestamp_us: 0,
        }
    }

    fn make_gradient_frame(w: u32, h: u32) -> FrameBuffer {
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((x + y) % 256) as u8);
            }
        }
        FrameBuffer {
            pixels,
            width: w,
            height: h,
            channels: 1,
            frame_id: 1,
            timestamp_us: 1000,
        }
    }

    fn make_rgb_frame(w: u32, h: u32) -> FrameBuffer {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((x * 3) % 256) as u8); // R
                pixels.push(((y * 5) % 256) as u8); // G
                pixels.push(((x + y) % 256) as u8); // B
            }
        }
        FrameBuffer {
            pixels,
            width: w,
            height: h,
            channels: 3,
            frame_id: 2,
            timestamp_us: 2000,
        }
    }

    #[test]
    fn test_basic_crop_no_padding() {
        let frame = make_gray_frame(64, 64, 128);
        let (crop, w, h) = extract_crop(&frame, 2, 3, 8, 0, usize::MAX);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(crop.len(), 64); // 8*8*1
        assert!(crop.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_crop_with_padding() {
        let frame = make_gray_frame(64, 64, 100);
        // Patch at (2,3) with pad=1 → (1..4) rows, (2..5) cols → 3×3 patches = 24×24
        let (crop, w, h) = extract_crop(&frame, 2, 3, 8, 1, usize::MAX);
        assert_eq!(w, 24);
        assert_eq!(h, 24);
        assert_eq!(crop.len(), 24 * 24);
    }

    #[test]
    fn test_crop_padding_clamps_at_edges() {
        let frame = make_gray_frame(64, 64, 50);
        // Patch at (0,0) with pad=1 → can't go negative, starts at 0
        let (crop, w, h) = extract_crop(&frame, 0, 0, 8, 1, usize::MAX);
        // Row: max(0-1,0)=0 to min(0+1+1,8)=2 → 2 patches = 16
        // Col: max(0-1,0)=0 to min(0+1+1,8)=2 → 2 patches = 16
        assert_eq!(w, 16);
        assert_eq!(h, 16);
        assert_eq!(crop.len(), 16 * 16);
    }

    #[test]
    fn test_crop_padding_clamps_at_bottom_right() {
        let frame = make_gray_frame(64, 64, 50);
        // Patch at (7,7) (last patch in 8×8 grid) with pad=1
        // Row: 6..9 → clamped to 6..8 (rows in pixels: 48..64) = 16px
        // Col: 6..9 → clamped to 6..8 (cols in pixels: 48..64) = 16px
        let (_crop, w, h) = extract_crop(&frame, 7, 7, 8, 1, usize::MAX);
        assert_eq!(w, 16);
        assert_eq!(h, 16);
    }

    #[test]
    fn test_crop_gradient_preserves_values() {
        let frame = make_gradient_frame(64, 64);
        // Patch at (1,1) no padding → pixels from (8,8) to (16,16)
        let (crop, w, h) = extract_crop(&frame, 1, 1, 8, 0, usize::MAX);
        assert_eq!(w, 8);
        assert_eq!(h, 8);

        // Verify first pixel matches gradient: (8+8)%256 = 16
        assert_eq!(crop[0], 16);
    }

    #[test]
    fn test_crop_rgb_correct_channels() {
        let frame = make_rgb_frame(64, 64);
        let (crop, w, h) = extract_crop(&frame, 2, 3, 8, 0, usize::MAX);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(crop.len(), 8 * 8 * 3); // RGB
    }

    #[test]
    fn test_downscale_triggers() {
        let frame = make_gray_frame(256, 256, 200);
        // Patch size 32, pad=1 → 3×3 patches = 96×96 = 9216 pixels
        // Max 4096 → must downscale
        let (_crop, w, h) = extract_crop(&frame, 4, 4, 32, 1, 4096);
        let total = (w as usize) * (h as usize);
        assert!(
            total <= 4096 + 100, // Allow small rounding overshoot
            "Downscaled crop {w}×{h}={total} should be near 4096"
        );
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn test_downscale_preserves_approximate_values() {
        // Solid color frame → downscaled should still be the same color
        let frame = make_gray_frame(128, 128, 150);
        let (crop, w, h) = extract_crop(&frame, 2, 2, 16, 1, 1024);
        assert!(w > 0 && h > 0);
        // All pixels should be very close to 150
        for &v in &crop {
            assert!(
                (v as i32 - 150).unsigned_abs() <= 1,
                "Downscaled solid should preserve value, got {v}"
            );
        }
    }

    #[test]
    fn test_empty_crop_zero_size() {
        // Frame too small for requested grid position
        let frame = make_gray_frame(4, 4, 100);
        // Grid row 10 with patch_size 8 → y0 = 80, y1 = min(88,4)=4, crop_h=0
        let (crop, w, h) = extract_crop(&frame, 10, 10, 8, 0, usize::MAX);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
        assert!(crop.is_empty());
    }

    #[test]
    fn test_bilinear_downscale_identity() {
        // Same size → should be near-identical
        let src = vec![100u8, 150, 200, 50];
        let dst = bilinear_downscale(&src, 2, 2, 1, 2, 2);
        assert_eq!(dst.len(), 4);
        // Values should be close to originals
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!(
                (*s as i32 - *d as i32).unsigned_abs() <= 1,
                "Identity downscale should preserve values: {s} vs {d}"
            );
        }
    }

    #[test]
    fn test_bilinear_downscale_halves() {
        // 4×4 solid → 2×2 should still be solid
        let src = vec![180u8; 16];
        let dst = bilinear_downscale(&src, 4, 4, 1, 2, 2);
        assert_eq!(dst.len(), 4);
        for &v in &dst {
            assert_eq!(v, 180);
        }
    }

    #[test]
    fn test_bilinear_downscale_rgb() {
        // 4×4 RGB solid → 2×2 RGB
        let src: Vec<u8> = vec![100, 150, 200].repeat(16);
        let dst = bilinear_downscale(&src, 4, 4, 3, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
        for chunk in dst.chunks(3) {
            assert!((chunk[0] as i32 - 100).unsigned_abs() <= 1);
            assert!((chunk[1] as i32 - 150).unsigned_abs() <= 1);
            assert!((chunk[2] as i32 - 200).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn test_crop_entire_frame_single_patch() {
        // Frame exactly one patch
        let frame = make_gray_frame(8, 8, 77);
        let (crop, w, h) = extract_crop(&frame, 0, 0, 8, 0, usize::MAX);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(crop.len(), 64);
        assert!(crop.iter().all(|&v| v == 77));
    }
}
