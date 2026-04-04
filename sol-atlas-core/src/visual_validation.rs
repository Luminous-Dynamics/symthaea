// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Visual validation for Sol Atlas — automated checks on rendered frames.
//!
//! Phase 1: Statistical checks on PNG images (brightness, color distribution).
//! Phase 2: HDC encoding + cosine similarity for regression detection.
//! Phase 3: Symthaea FEP agent perception-action loop for autonomous QA.

/// Frame statistics for visual validation.
#[derive(Debug, Clone)]
pub struct FrameStats {
    pub width: u32,
    pub height: u32,
    pub mean_brightness: f32,
    pub mean_red: f32,
    pub mean_green: f32,
    pub mean_blue: f32,
    pub non_black_fraction: f32,
}

/// Compute basic statistics from raw RGBA pixel data.
pub fn compute_frame_stats(pixels: &[u8], width: u32, height: u32) -> FrameStats {
    let total = (width * height) as f64;
    if total == 0.0 || pixels.len() < (total as usize * 4) {
        return FrameStats {
            width, height,
            mean_brightness: 0.0, mean_red: 0.0, mean_green: 0.0, mean_blue: 0.0,
            non_black_fraction: 0.0,
        };
    }

    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut non_black = 0u64;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f64 / 255.0;
        let g = chunk[1] as f64 / 255.0;
        let b = chunk[2] as f64 / 255.0;
        sum_r += r;
        sum_g += g;
        sum_b += b;
        if r > 0.02 || g > 0.02 || b > 0.02 {
            non_black += 1;
        }
    }

    let mean_r = (sum_r / total) as f32;
    let mean_g = (sum_g / total) as f32;
    let mean_b = (sum_b / total) as f32;

    FrameStats {
        width,
        height,
        mean_brightness: (mean_r + mean_g + mean_b) / 3.0,
        mean_red: mean_r,
        mean_green: mean_g,
        mean_blue: mean_b,
        non_black_fraction: non_black as f32 / total as f32,
    }
}

/// Visual validation checks — returns list of issues found.
pub fn validate_frame(stats: &FrameStats) -> Vec<String> {
    let mut issues = Vec::new();

    // Check 1: Frame is not completely black
    if stats.non_black_fraction < 0.05 {
        issues.push(format!(
            "Frame nearly black: only {:.1}% non-black pixels",
            stats.non_black_fraction * 100.0
        ));
    }

    // Check 2: Globe should make at least 10% of frame non-black
    if stats.non_black_fraction < 0.10 {
        issues.push(format!(
            "Globe may not be rendering: {:.1}% non-black (expected >10%)",
            stats.non_black_fraction * 100.0
        ));
    }

    // Check 3: Teal/cyan bias expected for holographic mode
    if stats.mean_green < stats.mean_red && stats.mean_blue < stats.mean_red {
        issues.push("Color bias: red dominant (expected teal/cyan for holographic)".into());
    }

    // Check 4: Not overexposed
    if stats.mean_brightness > 0.5 {
        issues.push(format!(
            "Overexposed: mean brightness {:.2} (expected <0.5 for space scene)",
            stats.mean_brightness
        ));
    }

    issues
}

/// Simple HDC-inspired frame fingerprint — encodes spatial color distribution
/// as a fixed-size vector for fast similarity comparison.
/// Not full 16,384D BinaryHV (requires symthaea-core dep), but captures the
/// same concept: perceptual similarity without pixel-level comparison.
pub struct FrameFingerprint {
    /// 64 bins: 4x4 spatial grid × 4 color channels (R, G, B, brightness)
    pub bins: [f32; 64],
}

impl FrameFingerprint {
    /// Encode a frame into a 64-bin fingerprint.
    pub fn from_pixels(pixels: &[u8], width: u32, height: u32) -> Self {
        let mut bins = [0.0f32; 64];
        let total = (width * height) as f32;
        if total == 0.0 || pixels.len() < (total as usize * 4) {
            return Self { bins };
        }

        // 4x4 spatial grid
        let gw = width / 4;
        let gh = height / 4;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) as usize) * 4;
                if idx + 3 >= pixels.len() { continue; }
                let r = pixels[idx] as f32 / 255.0;
                let g = pixels[idx + 1] as f32 / 255.0;
                let b = pixels[idx + 2] as f32 / 255.0;

                let gx = (x / gw).min(3) as usize;
                let gy = (y / gh).min(3) as usize;
                let grid_idx = (gy * 4 + gx) * 4;

                bins[grid_idx] += r;
                bins[grid_idx + 1] += g;
                bins[grid_idx + 2] += b;
                bins[grid_idx + 3] += (r + g + b) / 3.0;
            }
        }

        // Normalize by cell pixel count
        let cell_count = (gw * gh) as f32;
        if cell_count > 0.0 {
            for bin in bins.iter_mut() {
                *bin /= cell_count;
            }
        }

        Self { bins }
    }

    /// Cosine similarity between two fingerprints (-1 to 1).
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0f32;
        let mut mag_a = 0.0f32;
        let mut mag_b = 0.0f32;
        for i in 0..64 {
            dot += self.bins[i] * other.bins[i];
            mag_a += self.bins[i] * self.bins[i];
            mag_b += other.bins[i] * other.bins[i];
        }
        let denom = mag_a.sqrt() * mag_b.sqrt();
        if denom < 1e-10 { return 0.0; }
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_frame_detected() {
        let pixels = vec![0u8; 100 * 100 * 4]; // all black
        let stats = compute_frame_stats(&pixels, 100, 100);
        assert!(stats.non_black_fraction < 0.01);
        let issues = validate_frame(&stats);
        assert!(!issues.is_empty());
        assert!(issues[0].contains("black"));
    }

    #[test]
    fn normal_frame_passes() {
        // Simulate a frame with ~30% globe coverage (teal on black)
        let mut pixels = vec![0u8; 100 * 100 * 4];
        for i in 0..3000 {
            let idx = i * 4;
            pixels[idx] = 20;    // R: low
            pixels[idx + 1] = 80; // G: teal
            pixels[idx + 2] = 90; // B: cyan
            pixels[idx + 3] = 255;
        }
        let stats = compute_frame_stats(&pixels, 100, 100);
        assert!(stats.non_black_fraction > 0.25);
        assert!(stats.mean_green > stats.mean_red);
        let issues = validate_frame(&stats);
        assert!(issues.is_empty(), "Expected no issues, got: {:?}", issues);
    }

    #[test]
    fn overexposed_detected() {
        let pixels = vec![200u8; 100 * 100 * 4]; // very bright
        let stats = compute_frame_stats(&pixels, 100, 100);
        let issues = validate_frame(&stats);
        assert!(issues.iter().any(|i| i.contains("Overexposed")));
    }

    #[test]
    fn fingerprint_self_similarity() {
        let mut pixels = vec![0u8; 100 * 100 * 4];
        for i in 0..3000 { pixels[i * 4 + 1] = 80; pixels[i * 4 + 2] = 90; }
        let fp = FrameFingerprint::from_pixels(&pixels, 100, 100);
        assert!((fp.similarity(&fp) - 1.0).abs() < 0.001, "Self-similarity should be 1.0");
    }

    #[test]
    fn fingerprint_different_frames() {
        // Teal frame
        let mut teal = vec![0u8; 100 * 100 * 4];
        for i in 0..5000 { teal[i * 4 + 1] = 80; teal[i * 4 + 2] = 90; }
        // Red frame
        let mut red = vec![0u8; 100 * 100 * 4];
        for i in 0..5000 { red[i * 4] = 200; }

        let fp_teal = FrameFingerprint::from_pixels(&teal, 100, 100);
        let fp_red = FrameFingerprint::from_pixels(&red, 100, 100);
        let sim = fp_teal.similarity(&fp_red);
        assert!(sim < 0.8, "Different color frames should have low similarity: {}", sim);
    }

    #[test]
    fn fingerprint_similar_frames() {
        // Two teal frames with slight variation
        let mut frame_a = vec![0u8; 100 * 100 * 4];
        let mut frame_b = vec![0u8; 100 * 100 * 4];
        for i in 0..5000 {
            frame_a[i * 4 + 1] = 80; frame_a[i * 4 + 2] = 90;
            frame_b[i * 4 + 1] = 78; frame_b[i * 4 + 2] = 92; // slight variation
        }
        let fp_a = FrameFingerprint::from_pixels(&frame_a, 100, 100);
        let fp_b = FrameFingerprint::from_pixels(&frame_b, 100, 100);
        let sim = fp_a.similarity(&fp_b);
        assert!(sim > 0.95, "Similar frames should have high similarity: {}", sim);
    }
}
