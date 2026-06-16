// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frame - A single video frame with timestamp
//!
//! Represents a captured frame from a video source with precise timing
//! information for temporal analysis.

use std::time::{Duration, Instant};

/// Pixel format for frame data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    /// RGB24 - 3 bytes per pixel (Red, Green, Blue)
    Rgb24,
    /// RGBA32 - 4 bytes per pixel (Red, Green, Blue, Alpha)
    Rgba32,
    /// Grayscale - 1 byte per pixel
    Gray8,
}

impl FrameFormat {
    /// Bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            FrameFormat::Rgb24 => 3,
            FrameFormat::Rgba32 => 4,
            FrameFormat::Gray8 => 1,
        }
    }
}

/// A single video frame with timing information
#[derive(Clone)]
pub struct Frame {
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Pixel format
    pub format: FrameFormat,
    /// When this frame was captured (monotonic time)
    pub timestamp: Instant,
    /// Frame sequence number (for ordering)
    pub sequence: u64,
}

impl Frame {
    /// Create a new frame
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: FrameFormat, sequence: u64) -> Self {
        Self {
            data,
            width,
            height,
            format,
            timestamp: Instant::now(),
            sequence,
        }
    }

    /// Create a grayscale frame from RGB24 data
    pub fn to_grayscale(&self) -> Frame {
        if self.format == FrameFormat::Gray8 {
            return self.clone();
        }

        let bpp = self.format.bytes_per_pixel();
        let pixel_count = (self.width * self.height) as usize;
        let mut gray_data = Vec::with_capacity(pixel_count);

        for i in 0..pixel_count {
            let offset = i * bpp;
            if offset + 2 < self.data.len() {
                let r = self.data[offset] as f32;
                let g = self.data[offset + 1] as f32;
                let b = self.data[offset + 2] as f32;
                // Luminance formula: 0.299*R + 0.587*G + 0.114*B
                let gray = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                gray_data.push(gray);
            }
        }

        Frame {
            data: gray_data,
            width: self.width,
            height: self.height,
            format: FrameFormat::Gray8,
            timestamp: self.timestamp,
            sequence: self.sequence,
        }
    }

    /// Get pixel at (x, y) as (R, G, B)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<(u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let bpp = self.format.bytes_per_pixel();
        let offset = ((y * self.width + x) as usize) * bpp;

        match self.format {
            FrameFormat::Rgb24 | FrameFormat::Rgba32 => {
                if offset + 2 < self.data.len() {
                    Some((
                        self.data[offset],
                        self.data[offset + 1],
                        self.data[offset + 2],
                    ))
                } else {
                    None
                }
            }
            FrameFormat::Gray8 => {
                if offset < self.data.len() {
                    let g = self.data[offset];
                    Some((g, g, g))
                } else {
                    None
                }
            }
        }
    }

    /// Calculate average brightness (0.0 to 1.0)
    pub fn average_brightness(&self) -> f32 {
        let gray = self.to_grayscale();
        if gray.data.is_empty() {
            return 0.0;
        }

        let sum: u64 = gray.data.iter().map(|&p| p as u64).sum();
        (sum as f32) / (gray.data.len() as f32 * 255.0)
    }

    /// Downsample to target size (simple box filter)
    pub fn downsample(&self, target_width: u32, target_height: u32) -> Frame {
        if target_width >= self.width && target_height >= self.height {
            return self.clone();
        }

        let scale_x = self.width as f32 / target_width as f32;
        let scale_y = self.height as f32 / target_height as f32;
        let bpp = self.format.bytes_per_pixel();

        let mut new_data = Vec::with_capacity((target_width * target_height) as usize * bpp);

        for y in 0..target_height {
            for x in 0..target_width {
                let src_x = ((x as f32 + 0.5) * scale_x) as u32;
                let src_y = ((y as f32 + 0.5) * scale_y) as u32;
                let src_x = src_x.min(self.width - 1);
                let src_y = src_y.min(self.height - 1);

                let src_offset = ((src_y * self.width + src_x) as usize) * bpp;
                for b in 0..bpp {
                    if src_offset + b < self.data.len() {
                        new_data.push(self.data[src_offset + b]);
                    } else {
                        new_data.push(0);
                    }
                }
            }
        }

        Frame {
            data: new_data,
            width: target_width,
            height: target_height,
            format: self.format,
            timestamp: self.timestamp,
            sequence: self.sequence,
        }
    }

    /// Time since this frame was captured
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Expected data size based on dimensions and format
    pub fn expected_size(&self) -> usize {
        (self.width * self.height) as usize * self.format.bytes_per_pixel()
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("sequence", &self.sequence)
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Frame {
        let size = (width * height * 3) as usize;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        Frame::new(data, width, height, FrameFormat::Rgb24, 0)
    }

    #[test]
    fn test_frame_creation() {
        let frame = create_test_frame(64, 64);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.format, FrameFormat::Rgb24);
        assert_eq!(frame.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_grayscale_conversion() {
        let frame = create_test_frame(64, 64);
        let gray = frame.to_grayscale();
        assert_eq!(gray.format, FrameFormat::Gray8);
        assert_eq!(gray.data.len(), 64 * 64);
    }

    #[test]
    fn test_downsample() {
        let frame = create_test_frame(128, 128);
        let downsampled = frame.downsample(64, 64);
        assert_eq!(downsampled.width, 64);
        assert_eq!(downsampled.height, 64);
        assert_eq!(downsampled.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_brightness() {
        // Create a white frame
        let white_data = vec![255u8; 64 * 64 * 3];
        let white_frame = Frame::new(white_data, 64, 64, FrameFormat::Rgb24, 0);
        assert!((white_frame.average_brightness() - 1.0).abs() < 0.01);

        // Create a black frame
        let black_data = vec![0u8; 64 * 64 * 3];
        let black_frame = Frame::new(black_data, 64, 64, FrameFormat::Rgb24, 0);
        assert!((black_frame.average_brightness() - 0.0).abs() < 0.01);
    }
}
