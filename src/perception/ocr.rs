// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OCR (Optical Character Recognition) - Reading text from images
//!
//! Dual-strategy approach for maximum accuracy:
//! 1. **rten + ocrs**: Pure Rust, lightweight (8MB), fast, no external dependencies
//! 2. **Tesseract**: Fallback for complex/noisy text, multilingual support
//!
//! Strategy selection based on:
//! - Image quality (clean vs noisy)
//! - Text complexity (simple vs handwritten)
//! - Confidence scores
//! - Performance requirements

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use std::time::Instant;

/// Recognized text from an image
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Extracted text
    pub text: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Recognition method used
    pub method: OcrMethod,

    /// Time taken for recognition
    pub duration_ms: u64,

    /// Individual word results (if available)
    pub words: Vec<OcrWord>,
}

/// Individual word recognition result
#[derive(Debug, Clone)]
pub struct OcrWord {
    /// Recognized word text
    pub text: String,

    /// Confidence for this word (0.0 to 1.0)
    pub confidence: f32,

    /// Bounding box (x, y, width, height)
    pub bbox: Option<(u32, u32, u32, u32)>,
}

/// OCR method used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrMethod {
    /// Pure Rust OCR (rten + ocrs)
    RustOcr,

    /// Tesseract fallback
    Tesseract,

    /// No text detected
    None,
}

impl std::fmt::Display for OcrMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrMethod::RustOcr => write!(f, "Rust OCR (rten)"),
            OcrMethod::Tesseract => write!(f, "Tesseract"),
            OcrMethod::None => write!(f, "No text detected"),
        }
    }
}

/// Image quality assessment for OCR strategy selection
#[derive(Debug, Clone)]
pub struct ImageQuality {
    /// Estimated quality score (0.0 = poor, 1.0 = excellent)
    pub score: f32,

    /// Is image resolution sufficient?
    pub resolution_ok: bool,

    /// Is contrast sufficient?
    pub contrast_ok: bool,

    /// Is image too noisy?
    pub too_noisy: bool,
}

/// Pure Rust OCR engine using rten + ocrs
pub struct RustOcrEngine {
    /// Whether the engine is initialized
    initialized: bool,

    /// Minimum confidence threshold (0.0 to 1.0)
    min_confidence: f32,
}

impl Default for RustOcrEngine {
    fn default() -> Self {
        Self {
            initialized: false,
            min_confidence: 0.6, // Default: 60% confidence minimum
        }
    }
}

impl RustOcrEngine {
    /// Create a new Rust OCR engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the OCR engine (load models).
    ///
    /// rten/ocrs model loading is deferred (blocked on `perception` feature
    /// stabilization), so this engine cannot recognize anything yet and
    /// refuses to report itself ready. (Until 2026-07-15 it marked itself
    /// initialized and `recognize()` returned an empty `Ok` result — so the
    /// auto strategy silently produced no text on precisely the
    /// highest-quality images.)
    pub fn initialize(&mut self) -> Result<()> {
        self.initialized = false;
        anyhow::bail!("Rust OCR (rten/ocrs) models not integrated yet — engine unavailable")
    }

    /// Whether the engine can actually recognize text.
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Recognize text from an image
    pub fn recognize(&self, _image: &DynamicImage) -> Result<OcrResult> {
        anyhow::bail!(
            "Rust OCR (rten/ocrs) models not integrated yet — use Tesseract or OcrMethod::None"
        )
    }

    /// Set minimum confidence threshold
    pub fn set_min_confidence(&mut self, threshold: f32) {
        self.min_confidence = threshold.clamp(0.0, 1.0);
    }
}

/// Tesseract OCR engine (external dependency)
pub struct TesseractEngine {
    /// Whether Tesseract is available on system
    available: bool,

    /// Language to use (default: "eng")
    language: String,
}

impl Default for TesseractEngine {
    fn default() -> Self {
        Self {
            available: false,
            language: "eng".to_string(),
        }
    }
}

impl TesseractEngine {
    /// Create a new Tesseract engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if Tesseract is available on the system
    pub fn check_available(&mut self) -> bool {
        // Check if tesseract command exists using which/where
        self.available = std::process::Command::new("tesseract")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        self.available
    }

    /// Set language for recognition
    pub fn set_language(&mut self, lang: &str) {
        self.language = lang.to_string();
    }

    /// Recognize text using Tesseract
    pub fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        if !self.available {
            anyhow::bail!("Tesseract not available on system");
        }

        let start = Instant::now();

        // Create temporary file for image
        let temp_dir = std::env::temp_dir();
        let img_path = temp_dir.join(format!("symthaea_ocr_{}.png", std::process::id()));
        let out_base = temp_dir.join(format!("symthaea_ocr_{}_out", std::process::id()));

        // Save image as PNG
        image
            .save(&img_path)
            .context("Failed to save temporary image for OCR")?;

        // Run tesseract with TSV output for confidence scores
        let output = std::process::Command::new("tesseract")
            .arg(&img_path)
            .arg(&out_base)
            .arg("-l")
            .arg(&self.language)
            .arg("--psm")
            .arg("3") // Fully automatic page segmentation
            .arg("-c")
            .arg("tessedit_create_tsv=1")
            .output()
            .context("Failed to run tesseract")?;

        // Clean up image file
        let _ = std::fs::remove_file(&img_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_file(out_base.with_extension("tsv"));
            anyhow::bail!("Tesseract failed: {stderr}");
        }

        // Read TSV output
        let tsv_path = out_base.with_extension("tsv");
        let tsv_content = std::fs::read_to_string(&tsv_path).unwrap_or_default();
        let _ = std::fs::remove_file(&tsv_path);

        // Parse TSV for text and confidence
        let mut text_parts: Vec<String> = Vec::new();
        let mut total_confidence: f32 = 0.0;
        let mut word_count: usize = 0;
        let mut words: Vec<OcrWord> = Vec::new();

        for line in tsv_content.lines().skip(1) {
            // Skip header
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() >= 12 {
                let word_text = cols[11].trim();
                if !word_text.is_empty() && word_text != "-1" {
                    let conf: f32 = cols[10].parse().unwrap_or(0.0);
                    let x: u32 = cols[6].parse().unwrap_or(0);
                    let y: u32 = cols[7].parse().unwrap_or(0);
                    let w: u32 = cols[8].parse().unwrap_or(0);
                    let h: u32 = cols[9].parse().unwrap_or(0);

                    text_parts.push(word_text.to_string());
                    total_confidence += conf;
                    word_count += 1;

                    words.push(OcrWord {
                        text: word_text.to_string(),
                        confidence: conf / 100.0,
                        bbox: Some((x, y, w, h)),
                    });
                }
            }
        }

        let text = text_parts.join(" ");
        let confidence = if word_count > 0 {
            total_confidence / word_count as f32 / 100.0 // Tesseract uses 0-100 scale
        } else {
            0.0
        };

        let result = OcrResult {
            text,
            confidence,
            method: OcrMethod::Tesseract,
            duration_ms: start.elapsed().as_millis() as u64,
            words,
        };

        Ok(result)
    }
}

/// Main OCR system with intelligent strategy selection
pub struct OcrSystem {
    /// Rust OCR engine (primary)
    rust_ocr: RustOcrEngine,

    /// Tesseract engine (fallback)
    tesseract: TesseractEngine,

    /// Strategy: prefer Rust OCR, fallback to Tesseract
    prefer_rust: bool,

    /// Automatically assess quality and choose best method
    auto_strategy: bool,
}

impl Default for OcrSystem {
    fn default() -> Self {
        Self {
            rust_ocr: RustOcrEngine::new(),
            tesseract: TesseractEngine::new(),
            prefer_rust: true,
            auto_strategy: true,
        }
    }
}

impl OcrSystem {
    /// Create a new OCR system
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the OCR system.
    ///
    /// Engine availability is graceful: the Rust OCR engine currently has no
    /// models integrated and reports itself unavailable; Tesseract is probed
    /// on the system. Strategy selection only routes to engines that are
    /// actually available.
    pub fn initialize(&mut self) -> Result<()> {
        if let Err(e) = self.rust_ocr.initialize() {
            tracing::debug!(error = %e, "Rust OCR engine unavailable");
        }

        // Check if Tesseract is available (optional)
        self.tesseract.check_available();

        Ok(())
    }

    /// Assess image quality for OCR
    pub fn assess_quality(&self, image: &DynamicImage) -> ImageQuality {
        let (width, height) = image.dimensions();

        // Check resolution (minimum 100x30 for readable text)
        let resolution_ok = width >= 100 && height >= 30;

        // Convert to grayscale for analysis
        let gray = image.to_luma8();

        // Calculate contrast (std deviation of pixel values)
        let pixels: Vec<u8> = gray.pixels().map(|p| p[0]).collect();
        let n = pixels.len().max(1) as f32;
        let mean: f32 = pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
        let variance: f32 = pixels
            .iter()
            .map(|&p| (p as f32 - mean).powi(2))
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        // Good contrast: std dev > 30 (on 0-255 scale)
        let contrast_ok = std_dev > 30.0;

        // Estimate noise (high-frequency content)
        // Simple approximation: count edges
        let mut edge_count = 0;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = gray.get_pixel(x, y)[0] as i32;
                let left = gray.get_pixel(x - 1, y)[0] as i32;
                let right = gray.get_pixel(x + 1, y)[0] as i32;
                let diff = (center - left).abs() + (center - right).abs();
                if diff > 50 {
                    edge_count += 1;
                }
            }
        }

        let edge_density = edge_count as f32 / (width * height) as f32;
        let too_noisy = edge_density > 0.3; // More than 30% edge pixels = very noisy

        // Calculate overall quality score
        let mut score = 0.0;
        if resolution_ok {
            score += 0.4;
        }
        if contrast_ok {
            score += 0.4;
        }
        if !too_noisy {
            score += 0.2;
        }

        ImageQuality {
            score,
            resolution_ok,
            contrast_ok,
            too_noisy,
        }
    }

    /// Select best OCR strategy based on image quality.
    ///
    /// Only routes to engines that are actually available. (Until 2026-07-15
    /// this routed high-quality images to the model-less Rust OCR stub, which
    /// silently returned empty text with `Ok` — the best inputs produced
    /// nothing while low-quality ones fell back to working Tesseract.)
    pub fn select_strategy(&self, quality: &ImageQuality) -> OcrMethod {
        if !self.auto_strategy {
            // Manual strategy: prefer Rust OCR if enabled AND available
            if self.prefer_rust && self.rust_ocr.is_available() {
                return OcrMethod::RustOcr;
            } else if self.tesseract.available {
                return OcrMethod::Tesseract;
            } else {
                return OcrMethod::None;
            }
        }

        // Auto strategy: choose based on quality, among available engines
        if quality.score >= 0.7 && self.rust_ocr.is_available() {
            // High quality: Rust OCR is fast and accurate
            OcrMethod::RustOcr
        } else if self.tesseract.available {
            // Low quality (or no Rust OCR): Tesseract is more robust
            OcrMethod::Tesseract
        } else if self.rust_ocr.is_available() {
            // No good option: try Rust OCR anyway
            OcrMethod::RustOcr
        } else {
            OcrMethod::None
        }
    }

    /// Recognize text from an image using the best strategy
    pub fn recognize(&mut self, image: &DynamicImage) -> Result<OcrResult> {
        // Assess image quality
        let quality = self.assess_quality(image);

        // Select strategy
        let strategy = self.select_strategy(&quality);

        // Execute OCR
        match strategy {
            OcrMethod::RustOcr => self.rust_ocr.recognize(image),
            OcrMethod::Tesseract => self.tesseract.recognize(image),
            OcrMethod::None => Ok(OcrResult {
                text: String::new(),
                confidence: 0.0,
                method: OcrMethod::None,
                duration_ms: 0,
                words: Vec::new(),
            }),
        }
    }

    /// Recognize text with fallback strategy
    ///
    /// Tries Rust OCR first (if available), falls back to Tesseract if
    /// confidence is low. With neither engine available, returns an
    /// `OcrMethod::None` empty result rather than fabricating output.
    pub fn recognize_with_fallback(&mut self, image: &DynamicImage) -> Result<OcrResult> {
        let rust_result = if self.rust_ocr.is_available() {
            Some(self.rust_ocr.recognize(image)?)
        } else {
            None
        };

        // If confidence is high enough, return it
        if let Some(ref r) = rust_result {
            if r.confidence >= 0.7 {
                return Ok(rust_result.unwrap());
            }
        }

        // If Tesseract is available, try it as fallback
        if self.tesseract.available {
            let tesseract_result = self.tesseract.recognize(image)?;

            // Return whichever has higher confidence
            match rust_result {
                Some(r) if r.confidence > tesseract_result.confidence => Ok(r),
                _ => Ok(tesseract_result),
            }
        } else if let Some(r) = rust_result {
            // No fallback available
            Ok(r)
        } else {
            Ok(OcrResult {
                text: String::new(),
                confidence: 0.0,
                method: OcrMethod::None,
                duration_ms: 0,
                words: Vec::new(),
            })
        }
    }

    /// Check if Tesseract fallback is available
    pub fn has_tesseract(&self) -> bool {
        self.tesseract.available
    }

    /// Enable or disable automatic strategy selection
    pub fn set_auto_strategy(&mut self, enabled: bool) {
        self.auto_strategy = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 255, 255]); // White background
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_ocr_system_creation() {
        let ocr = OcrSystem::new();
        assert!(ocr.prefer_rust);
        assert!(ocr.auto_strategy);
    }

    #[test]
    fn test_rust_ocr_engine_creation() {
        let engine = RustOcrEngine::new();
        assert!(!engine.initialized);
        assert_eq!(engine.min_confidence, 0.6);
    }

    #[test]
    fn test_tesseract_engine_creation() {
        let engine = TesseractEngine::new();
        assert!(!engine.available);
        assert_eq!(engine.language, "eng");
    }

    #[test]
    fn test_ocr_method_display() {
        assert_eq!(format!("{}", OcrMethod::RustOcr), "Rust OCR (rten)");
        assert_eq!(format!("{}", OcrMethod::Tesseract), "Tesseract");
        assert_eq!(format!("{}", OcrMethod::None), "No text detected");
    }

    #[test]
    fn test_quality_assessment_resolution() {
        let ocr = OcrSystem::new();

        // Good resolution
        let good_img = create_test_image(640, 480);
        let quality = ocr.assess_quality(&good_img);
        assert!(quality.resolution_ok);

        // Poor resolution
        let poor_img = create_test_image(50, 20);
        let quality = ocr.assess_quality(&poor_img);
        assert!(!quality.resolution_ok);
    }

    #[test]
    fn test_quality_assessment_contrast() {
        let ocr = OcrSystem::new();

        // Create high-contrast image (black and white)
        let mut high_contrast = RgbImage::new(200, 200);
        for y in 0..200 {
            for x in 0..200 {
                let color = if (x + y) % 20 < 10 {
                    Rgb([0, 0, 0]) // Black
                } else {
                    Rgb([255, 255, 255]) // White
                };
                high_contrast.put_pixel(x, y, color);
            }
        }

        let quality = ocr.assess_quality(&DynamicImage::ImageRgb8(high_contrast));
        assert!(
            quality.contrast_ok,
            "High contrast image should have good contrast"
        );
    }

    #[test]
    fn test_strategy_selection_never_routes_to_unavailable_engine() {
        let ocr = OcrSystem::new();

        // Rust OCR has no models integrated and Tesseract has not been
        // probed — with no available engine, every quality level must
        // select None rather than an engine that silently returns nothing.
        let high_quality = ImageQuality {
            score: 0.9,
            resolution_ok: true,
            contrast_ok: true,
            too_noisy: false,
        };
        assert_eq!(ocr.select_strategy(&high_quality), OcrMethod::None);

        let low_quality = ImageQuality {
            score: 0.3,
            resolution_ok: true,
            contrast_ok: false,
            too_noisy: true,
        };
        assert_eq!(ocr.select_strategy(&low_quality), OcrMethod::None);
    }

    #[test]
    fn test_rust_ocr_refuses_to_report_ready_without_models() {
        let mut engine = RustOcrEngine::new();
        assert!(engine.initialize().is_err());
        assert!(!engine.is_available());
        assert!(engine.recognize(&create_test_image(200, 100)).is_err());
    }

    #[test]
    fn test_set_min_confidence() {
        let mut engine = RustOcrEngine::new();

        engine.set_min_confidence(0.8);
        assert_eq!(engine.min_confidence, 0.8);

        // Test clamping
        engine.set_min_confidence(1.5);
        assert_eq!(engine.min_confidence, 1.0);

        engine.set_min_confidence(-0.5);
        assert_eq!(engine.min_confidence, 0.0);
    }

    #[test]
    fn test_set_language() {
        let mut engine = TesseractEngine::new();

        engine.set_language("fra");
        assert_eq!(engine.language, "fra");

        engine.set_language("deu");
        assert_eq!(engine.language, "deu");
    }

    #[test]
    fn test_ocr_system_initialization() {
        let mut ocr = OcrSystem::new();
        let result = ocr.initialize();

        // Should succeed (marks Rust OCR as initialized)
        assert!(result.is_ok());
    }
}
