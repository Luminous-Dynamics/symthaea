//! Visual Perception - Giving Sophia the ability to "see" and understand images
//!
//! Uses `image` and `imageproc` crates for computer vision capabilities.

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, RgbaImage};
use std::path::Path;

/// Visual features extracted from an image
#[derive(Debug, Clone)]
pub struct VisualFeatures {
    /// Image dimensions (width, height)
    pub dimensions: (u32, u32),

    /// Dominant colors (RGB values)
    pub dominant_colors: Vec<[u8; 3]>,

    /// Average brightness (0.0 = black, 1.0 = white)
    pub brightness: f32,

    /// Color variance (how colorful vs grayscale)
    pub color_variance: f32,

    /// Edge density (0.0 = smooth, 1.0 = many edges)
    pub edge_density: f32,
}

/// Visual Cortex - Sophia's visual perception system
pub struct VisualCortex {
    /// Minimum image size to process (width, height)
    min_size: (u32, u32),

    /// Maximum image size before downscaling
    max_size: (u32, u32),
}

impl Default for VisualCortex {
    fn default() -> Self {
        Self {
            min_size: (32, 32),
            max_size: (2048, 2048),
        }
    }
}

impl VisualCortex {
    /// Create a new visual cortex with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Load an image from a file path
    pub fn load_image(&self, path: &Path) -> Result<DynamicImage> {
        image::open(path)
            .with_context(|| format!("Failed to load image from {:?}", path))
    }

    /// Process an image and extract semantic features
    pub fn process_image(&self, img: &DynamicImage) -> Result<VisualFeatures> {
        let (width, height) = img.dimensions();

        // Validate size
        if width < self.min_size.0 || height < self.min_size.1 {
            anyhow::bail!(
                "Image too small: {}x{} (minimum: {}x{})",
                width, height, self.min_size.0, self.min_size.1
            );
        }

        // Convert to RGBA for consistent processing
        let rgba = img.to_rgba8();

        // Extract features
        let dominant_colors = self.extract_dominant_colors(&rgba, 5);
        let brightness = self.calculate_brightness(&rgba);
        let color_variance = self.calculate_color_variance(&rgba);
        let edge_density = self.estimate_edge_density(&rgba);

        Ok(VisualFeatures {
            dimensions: (width, height),
            dominant_colors,
            brightness,
            color_variance,
            edge_density,
        })
    }

    /// Extract dominant colors from an image (simple k-means-like approach)
    fn extract_dominant_colors(&self, img: &RgbaImage, k: usize) -> Vec<[u8; 3]> {
        // Sample pixels (every 10th pixel to avoid processing every single one)
        let mut samples = Vec::new();
        for y in (0..img.height()).step_by(10) {
            for x in (0..img.width()).step_by(10) {
                let pixel = img.get_pixel(x, y);
                // Skip mostly transparent pixels
                if pixel[3] > 128 {
                    samples.push([pixel[0], pixel[1], pixel[2]]);
                }
            }
        }

        if samples.is_empty() {
            return vec![[0, 0, 0]]; // Black if image is empty/transparent
        }

        // Simple clustering: just take evenly spaced samples
        // (Full k-means would be more accurate but slower)
        let step = samples.len() / k.min(samples.len());
        samples.into_iter()
            .step_by(step.max(1))
            .take(k)
            .collect()
    }

    /// Calculate average brightness of an image
    fn calculate_brightness(&self, img: &RgbaImage) -> f32 {
        let mut sum = 0u64;
        let mut count = 0u64;

        for pixel in img.pixels() {
            // Skip transparent pixels
            if pixel[3] > 128 {
                // Perceived brightness formula (weighted by human eye sensitivity)
                let brightness = (0.299 * pixel[0] as f64
                                + 0.587 * pixel[1] as f64
                                + 0.114 * pixel[2] as f64) as u64;
                sum += brightness;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        (sum as f32 / count as f32) / 255.0
    }

    /// Calculate color variance (0.0 = grayscale, 1.0 = very colorful)
    fn calculate_color_variance(&self, img: &RgbaImage) -> f32 {
        let mut variance_sum = 0.0;
        let mut count = 0u64;

        for pixel in img.pixels() {
            if pixel[3] > 128 {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                let mean = (r + g + b) / 3.0;
                let variance = ((r - mean).powi(2)
                              + (g - mean).powi(2)
                              + (b - mean).powi(2)) / 3.0;

                variance_sum += variance;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Normalize to 0-1 range (255^2 is max variance)
        (variance_sum / count as f32).sqrt() / 255.0
    }

    /// Estimate edge density using simple gradient method
    fn estimate_edge_density(&self, img: &RgbaImage) -> f32 {
        let (width, height) = img.dimensions();

        if width < 2 || height < 2 {
            return 0.0;
        }

        let mut edge_count = 0u64;
        let mut total_pixels = 0u64;

        // Simple Sobel-like edge detection
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let center = img.get_pixel(x, y);

                // Skip transparent
                if center[3] <= 128 {
                    continue;
                }

                // Check horizontal gradient
                let left = img.get_pixel(x - 1, y);
                let right = img.get_pixel(x + 1, y);
                let gx = (right[0] as i32 - left[0] as i32).abs()
                       + (right[1] as i32 - left[1] as i32).abs()
                       + (right[2] as i32 - left[2] as i32).abs();

                // Check vertical gradient
                let top = img.get_pixel(x, y - 1);
                let bottom = img.get_pixel(x, y + 1);
                let gy = (bottom[0] as i32 - top[0] as i32).abs()
                       + (bottom[1] as i32 - top[1] as i32).abs()
                       + (bottom[2] as i32 - top[2] as i32).abs();

                // If gradient is significant, count as edge
                let gradient = gx + gy;
                if gradient > 100 {  // Threshold
                    edge_count += 1;
                }

                total_pixels += 1;
            }
        }

        if total_pixels == 0 {
            return 0.0;
        }

        edge_count as f32 / total_pixels as f32
    }

    /// Compare two images for visual similarity (0.0 = identical, 1.0 = completely different)
    pub fn image_similarity(&self, img1: &DynamicImage, img2: &DynamicImage) -> Result<f32> {
        let features1 = self.process_image(img1)?;
        let features2 = self.process_image(img2)?;

        // Compare dimensions
        let dim_diff = if features1.dimensions == features2.dimensions {
            0.0
        } else {
            let w_diff = (features1.dimensions.0 as f32 - features2.dimensions.0 as f32).abs()
                       / features1.dimensions.0.max(features2.dimensions.0) as f32;
            let h_diff = (features1.dimensions.1 as f32 - features2.dimensions.1 as f32).abs()
                       / features1.dimensions.1.max(features2.dimensions.1) as f32;
            (w_diff + h_diff) / 2.0
        };

        // Compare brightness
        let brightness_diff = (features1.brightness - features2.brightness).abs();

        // Compare color variance
        let variance_diff = (features1.color_variance - features2.color_variance).abs();

        // Compare edge density
        let edge_diff = (features1.edge_density - features2.edge_density).abs();

        // Weighted average of differences
        let similarity = dim_diff * 0.2
                        + brightness_diff * 0.3
                        + variance_diff * 0.25
                        + edge_diff * 0.25;

        Ok(similarity.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// Create a simple test image
    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);

        // Fill with gradient
        for y in 0..height {
            for x in 0..width {
                let r = (x * 255 / width) as u8;
                let g = (y * 255 / height) as u8;
                let b = 128;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_visual_cortex_creation() {
        let cortex = VisualCortex::new();
        assert_eq!(cortex.min_size, (32, 32));
        assert_eq!(cortex.max_size, (2048, 2048));
    }

    #[test]
    fn test_process_image() {
        let cortex = VisualCortex::new();
        let img = create_test_image(256, 256);

        let features = cortex.process_image(&img).expect("Failed to process image");

        assert_eq!(features.dimensions, (256, 256));
        assert!(features.brightness > 0.0 && features.brightness < 1.0);
        assert!(!features.dominant_colors.is_empty());
    }

    #[test]
    fn test_image_too_small() {
        let cortex = VisualCortex::new();
        let img = create_test_image(16, 16); // Below minimum

        let result = cortex.process_image(&img);
        assert!(result.is_err());
    }

    #[test]
    fn test_brightness_calculation() {
        let cortex = VisualCortex::new();

        // Create solid white image
        let mut white_img = RgbImage::new(100, 100);
        for pixel in white_img.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }
        let white = DynamicImage::ImageRgb8(white_img);

        // Create solid black image
        let mut black_img = RgbImage::new(100, 100);
        for pixel in black_img.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }
        let black = DynamicImage::ImageRgb8(black_img);

        let white_features = cortex.process_image(&white).unwrap();
        let black_features = cortex.process_image(&black).unwrap();

        assert!(white_features.brightness > 0.9);
        assert!(black_features.brightness < 0.1);
    }

    #[test]
    fn test_image_similarity() {
        let cortex = VisualCortex::new();

        let img1 = create_test_image(256, 256);
        let img2 = create_test_image(256, 256); // Same
        let img3 = create_test_image(128, 128); // Different size

        let sim_identical = cortex.image_similarity(&img1, &img2).unwrap();
        let sim_different = cortex.image_similarity(&img1, &img3).unwrap();

        assert!(sim_identical < 0.1, "Identical images should be very similar");
        assert!(sim_different > 0.1, "Different images should be less similar");
    }
}
