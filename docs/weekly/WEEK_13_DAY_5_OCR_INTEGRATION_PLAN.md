# 📖 Week 13 Day 5: OCR Model Integration Plan

**Date**: December 10, 2025
**Status**: 📋 Planning Complete - Ready for Implementation
**Goal**: Real OCR with rten + ocrs for text detection and recognition

---

## 🎯 Objective

Replace placeholder OCR implementation with real text detection and recognition models using rten (ONNX Runtime in Rust) and ocrs (pure Rust OCR), with Tesseract as fallback for complex cases.

---

## 📊 Current State (Week 13 Day 5 Start)

✅ **Architecture Complete**: Dual-strategy OCR system
✅ **Quality Assessment**: Image analysis for strategy selection
✅ **Fallback Logic**: Automatic switching between methods
✅ **10/10 Tests Passing**: Complete test coverage for structure
✅ **Total Project**: **200/200 tests** (38% complete)

**What needs implementation**:
- rten model loading and inference
- ocrs text detection and recognition
- Model downloads and caching
- Tesseract integration (optional)
- Performance optimization

---

## 🔬 OCR Model Details

### Primary: rten + ocrs (Pure Rust)

#### Text Detection Model
- **Model**: CRAFT (Character Region Awareness for Text detection)
- **Format**: ONNX
- **Size**: ~4-5MB
- **Purpose**: Locate text regions in images
- **Output**: Bounding boxes of text regions
- **Performance**: 10-20ms on CPU

#### Text Recognition Model
- **Model**: CRNN (Convolutional Recurrent Neural Network)
- **Format**: ONNX
- **Size**: ~3-4MB
- **Purpose**: Recognize characters in detected regions
- **Output**: Text strings with confidence scores
- **Performance**: 5-10ms per text region on CPU

#### Character Set
- **English**: 26 letters + 10 digits + punctuation (~80 characters)
- **Extended**: Support for accented characters and symbols
- **Expandable**: Can add more languages with model retraining

### Fallback: Tesseract (External)

- **Source**: System-installed Tesseract OCR
- **Version**: 4.x or 5.x
- **Languages**: 100+ languages available
- **Size**: Varies by language (~10-50MB per language)
- **Performance**: 100-500ms depending on image complexity
- **Use case**: Complex documents, handwritten text, low-quality images

---

## 🛠️ Implementation Plan

### Step 1: rten Integration

**Goal**: Load and run ONNX models in Rust

```rust
use rten::{Model, NodeId};
use rten_tensor::{Tensor, NdTensor};

pub struct RtenOcrEngine {
    detection_model: Option<Model>,
    recognition_model: Option<Model>,
    model_cache_dir: PathBuf,
}

impl RtenOcrEngine {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            detection_model: None,
            recognition_model: None,
            model_cache_dir: cache_dir,
        }
    }

    /// Download models on first use
    pub async fn ensure_models_downloaded(&self) -> Result<()> {
        let detection_path = self.model_cache_dir.join("craft_detection.onnx");
        let recognition_path = self.model_cache_dir.join("crnn_recognition.onnx");

        // Check if models exist
        if !detection_path.exists() {
            println!("📥 Downloading text detection model (~4MB)...");
            // Download from HuggingFace or model repository
            download_model("text-detection", &detection_path).await?;
        }

        if !recognition_path.exists() {
            println!("📥 Downloading text recognition model (~3MB)...");
            download_model("text-recognition", &recognition_path).await?;
        }

        Ok(())
    }

    /// Load ONNX models into memory
    pub fn load_models(&mut self) -> Result<()> {
        let detection_path = self.model_cache_dir.join("craft_detection.onnx");
        let recognition_path = self.model_cache_dir.join("crnn_recognition.onnx");

        // Load detection model
        self.detection_model = Some(Model::load_file(&detection_path)?);

        // Load recognition model
        self.recognition_model = Some(Model::load_file(&recognition_path)?);

        Ok(())
    }
}
```

**Test**: `test_rten_model_loading`

---

### Step 2: Text Detection

**Goal**: Locate text regions in images

```rust
impl RtenOcrEngine {
    /// Detect text regions in an image
    pub fn detect_text_regions(&self, image: &DynamicImage) -> Result<Vec<TextRegion>> {
        let model = self.detection_model.as_ref()
            .ok_or_else(|| anyhow!("Detection model not loaded"))?;

        // Preprocess image for detection model
        let input_tensor = self.preprocess_for_detection(image)?;

        // Run detection model
        let outputs = model.run(vec![input_tensor])?;

        // Post-process outputs to get bounding boxes
        let regions = self.postprocess_detection(&outputs[0], image.dimensions())?;

        Ok(regions)
    }

    fn preprocess_for_detection(&self, image: &DynamicImage) -> Result<NdTensor<f32, 4>> {
        // CRAFT expects 320x320 input
        let resized = image.resize_exact(320, 320, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        // Convert to NCHW format (1, 3, 320, 320)
        let mut tensor_data = Vec::with_capacity(3 * 320 * 320);

        // Normalize to [0, 1]
        for channel in 0..3 {
            for y in 0..320 {
                for x in 0..320 {
                    let pixel = rgb.get_pixel(x, y);
                    let value = pixel[channel] as f32 / 255.0;
                    tensor_data.push(value);
                }
            }
        }

        // Create tensor
        let tensor = NdTensor::from_data(
            vec![1, 3, 320, 320],
            tensor_data,
        );

        Ok(tensor)
    }

    fn postprocess_detection(
        &self,
        output: &Tensor,
        original_dims: (u32, u32),
    ) -> Result<Vec<TextRegion>> {
        // CRAFT outputs confidence maps
        // Apply threshold and find connected components

        let mut regions = Vec::new();

        // Extract bounding boxes from confidence map
        // (Implementation depends on CRAFT output format)

        Ok(regions)
    }
}

#[derive(Debug, Clone)]
pub struct TextRegion {
    /// Bounding box (x, y, width, height) in original image coordinates
    pub bbox: (u32, u32, u32, u32),

    /// Confidence score for detection (0.0 to 1.0)
    pub detection_confidence: f32,

    /// Text content (filled after recognition)
    pub text: Option<String>,

    /// Recognition confidence (filled after recognition)
    pub recognition_confidence: Option<f32>,
}
```

**Tests**:
- `test_text_detection_model`
- `test_preprocess_for_detection`
- `test_postprocess_detection_regions`

---

### Step 3: Text Recognition

**Goal**: Recognize text within detected regions

```rust
impl RtenOcrEngine {
    /// Recognize text in detected regions
    pub fn recognize_text_regions(
        &self,
        image: &DynamicImage,
        regions: &mut [TextRegion],
    ) -> Result<()> {
        let model = self.recognition_model.as_ref()
            .ok_or_else(|| anyhow!("Recognition model not loaded"))?;

        for region in regions.iter_mut() {
            // Crop image to region
            let cropped = self.crop_region(image, region.bbox)?;

            // Preprocess for recognition
            let input_tensor = self.preprocess_for_recognition(&cropped)?;

            // Run recognition model
            let outputs = model.run(vec![input_tensor])?;

            // Decode CTC output to text
            let (text, confidence) = self.decode_ctc_output(&outputs[0])?;

            region.text = Some(text);
            region.recognition_confidence = Some(confidence);
        }

        Ok(())
    }

    fn crop_region(&self, image: &DynamicImage, bbox: (u32, u32, u32, u32)) -> Result<DynamicImage> {
        let (x, y, width, height) = bbox;
        Ok(image.crop_imm(x, y, width, height))
    }

    fn preprocess_for_recognition(&self, image: &DynamicImage) -> Result<NdTensor<f32, 4>> {
        // CRNN expects 32 pixel height, variable width
        let height = 32;
        let aspect_ratio = image.width() as f32 / image.height() as f32;
        let width = (height as f32 * aspect_ratio) as u32;
        let width = width.min(512); // Cap at 512 pixels width

        let resized = image.resize_exact(width, height, FilterType::Lanczos3);
        let gray = resized.to_luma8();

        // Convert to tensor (1, 1, 32, width)
        let mut tensor_data = Vec::with_capacity((height * width) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = gray.get_pixel(x, y)[0];
                let value = (pixel as f32 / 255.0 - 0.5) / 0.5; // Normalize to [-1, 1]
                tensor_data.push(value);
            }
        }

        let tensor = NdTensor::from_data(
            vec![1, 1, height as usize, width as usize],
            tensor_data,
        );

        Ok(tensor)
    }

    fn decode_ctc_output(&self, output: &Tensor) -> Result<(String, f32)> {
        // CTC (Connectionist Temporal Classification) decoder
        // Output shape: (sequence_length, batch_size, num_classes)

        // Character set for English + digits + punctuation
        const CHARSET: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

        // Simple greedy decoding
        let mut text = String::new();
        let mut total_confidence = 0.0;
        let mut count = 0;

        // Get best character at each timestep
        // Remove duplicates and blanks
        let mut prev_char = None;

        // (Actual CTC decoding implementation)

        let avg_confidence = if count > 0 {
            total_confidence / count as f32
        } else {
            0.0
        };

        Ok((text, avg_confidence))
    }
}
```

**Tests**:
- `test_text_recognition_model`
- `test_preprocess_for_recognition`
- `test_ctc_decoding`

---

### Step 4: Integration with RustOcrEngine

**Goal**: Wire up rten models to existing OCR interface

```rust
impl RustOcrEngine {
    /// Initialize the OCR engine (load models)
    pub fn initialize(&mut self) -> Result<()> {
        // Create model cache directory
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Cannot find cache directory"))?
            .join("symthaea-hlb")
            .join("ocr-models");

        std::fs::create_dir_all(&cache_dir)?;

        // Create rten engine
        let mut rten_engine = RtenOcrEngine::new(cache_dir);

        // Download models if needed (in background if possible)
        // This might take 30-60 seconds on first run
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                rten_engine.ensure_models_downloaded()
            )
        })?;

        // Load models into memory
        rten_engine.load_models()?;

        self.initialized = true;
        Ok(())
    }

    /// Recognize text from an image
    pub fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        if !self.initialized {
            anyhow::bail!("OCR engine not initialized. Call initialize() first.");
        }

        let start = Instant::now();

        // Step 1: Detect text regions
        let mut regions = self.rten_engine.detect_text_regions(image)?;

        if regions.is_empty() {
            // No text detected
            return Ok(OcrResult {
                text: String::new(),
                confidence: 0.0,
                method: OcrMethod::RustOcr,
                duration_ms: start.elapsed().as_millis() as u64,
                words: Vec::new(),
            });
        }

        // Step 2: Recognize text in each region
        self.rten_engine.recognize_text_regions(image, &mut regions)?;

        // Step 3: Combine results
        let mut full_text = String::new();
        let mut total_confidence = 0.0;
        let mut words = Vec::new();

        for region in regions {
            if let (Some(text), Some(conf)) = (region.text, region.recognition_confidence) {
                if conf >= self.min_confidence {
                    full_text.push_str(&text);
                    full_text.push(' ');

                    total_confidence += conf;

                    words.push(OcrWord {
                        text,
                        confidence: conf,
                        bbox: Some(region.bbox),
                    });
                }
            }
        }

        let avg_confidence = if !words.is_empty() {
            total_confidence / words.len() as f32
        } else {
            0.0
        };

        Ok(OcrResult {
            text: full_text.trim().to_string(),
            confidence: avg_confidence,
            method: OcrMethod::RustOcr,
            duration_ms: start.elapsed().as_millis() as u64,
            words,
        })
    }
}
```

**Tests**:
- `test_full_ocr_pipeline`
- `test_empty_image_handling`
- `test_multi_line_text`

---

### Step 5: Tesseract Integration (Optional Fallback)

**Goal**: Use system Tesseract for complex cases

```rust
use std::process::Command;
use tempfile::NamedTempFile;

impl TesseractEngine {
    /// Check if Tesseract is available on the system
    pub fn check_available(&mut self) -> bool {
        // Try to run tesseract --version
        let result = Command::new("tesseract")
            .arg("--version")
            .output();

        self.available = result.is_ok();
        self.available
    }

    /// Recognize text using Tesseract
    pub fn recognize(&self, image: &DynamicImage) -> Result<OcrResult> {
        if !self.available {
            anyhow::bail!("Tesseract not available on system");
        }

        let start = Instant::now();

        // Save image to temporary file
        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path();
        image.save(temp_path)?;

        // Run tesseract
        let output = Command::new("tesseract")
            .arg(temp_path)
            .arg("stdout") // Output to stdout
            .arg("-l")
            .arg(&self.language)
            .arg("--psm")
            .arg("3") // Fully automatic page segmentation
            .arg("--oem")
            .arg("1") // LSTM neural network mode
            .output()?;

        if !output.status.success() {
            anyhow::bail!("Tesseract failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        let text = String::from_utf8(output.stdout)?
            .trim()
            .to_string();

        // Tesseract doesn't provide per-word confidence easily
        // Could use hocr output for detailed results if needed

        let confidence = if text.is_empty() { 0.0 } else { 0.8 };

        Ok(OcrResult {
            text,
            confidence,
            method: OcrMethod::Tesseract,
            duration_ms: start.elapsed().as_millis() as u64,
            words: Vec::new(), // Could parse hocr output for word-level results
        })
    }

    /// Get available languages
    pub fn list_languages(&self) -> Result<Vec<String>> {
        if !self.available {
            return Ok(Vec::new());
        }

        let output = Command::new("tesseract")
            .arg("--list-langs")
            .output()?;

        let langs: Vec<String> = String::from_utf8(output.stdout)?
            .lines()
            .skip(1) // Skip "List of available languages" header
            .map(|s| s.trim().to_string())
            .collect();

        Ok(langs)
    }
}
```

**Tests**:
- `test_tesseract_availability`
- `test_tesseract_recognition` (if available)
- `test_tesseract_language_listing`

---

### Step 6: Performance Optimization

**Goal**: Achieve <50ms for rten, <500ms for Tesseract

```rust
// Optimization strategies:

// 1. Image resizing before detection
// - Don't process giant images, resize to reasonable size first
// - Max 1920x1080 or similar

// 2. Parallel region recognition
// - Process multiple text regions concurrently
use rayon::prelude::*;

pub fn recognize_text_regions_parallel(
    &self,
    image: &DynamicImage,
    regions: &mut [TextRegion],
) -> Result<()> {
    regions.par_iter_mut().try_for_each(|region| {
        // Recognize each region in parallel
        // (Need to handle model thread safety)
        Ok(())
    })?;
    Ok(())
}

// 3. Model quantization
// - Use quantized models (int8) instead of float32
// - ~4x faster inference with minimal accuracy loss

// 4. Batch processing
// - Process multiple regions in a single model inference
// - Reduces overhead

// 5. Caching
// - Cache results for identical images
// - Use image hash as cache key
```

---

## 🧪 Testing Strategy

### Test Hierarchy

1. **Unit Tests** (individual components)
   - Model loading
   - Image preprocessing
   - Detection postprocessing
   - CTC decoding
   - Tesseract integration

2. **Integration Tests** (full pipeline)
   - End-to-end OCR with test images
   - Fallback strategy verification
   - Performance benchmarks
   - Multi-language support

3. **Quality Tests** (accuracy verification)
   - Known text in images
   - Compare rten vs Tesseract accuracy
   - Edge cases (rotated text, low contrast, etc.)

### New Tests to Add

```rust
#[tokio::test]
async fn test_rten_model_download_and_cache() {
    // Verify models download correctly and are cached
}

#[test]
fn test_text_detection_accuracy() {
    // Test with image containing known text regions
    // Verify detected bounding boxes are correct
}

#[test]
fn test_text_recognition_accuracy() {
    // Test with synthetic text images
    // Verify recognized text matches expected
}

#[test]
fn test_ocr_performance_benchmark() {
    // Measure inference time
    // Ensure < 50ms for typical images with rten
}

#[tokio::test]
async fn test_full_ocr_on_screenshot() {
    // Real-world test: OCR on screenshot of terminal
    // Verify code/text is extracted correctly
}

#[test]
fn test_fallback_to_tesseract() {
    // Test with low-quality image
    // Verify system falls back to Tesseract if available
}

#[test]
fn test_multi_language_support() {
    // Test with non-English text (if models support)
}
```

---

## 📊 Success Criteria

### Functional Requirements ✅
- [ ] Models download automatically on first use
- [ ] Text detection finds all text regions
- [ ] Text recognition achieves >90% accuracy on clean text
- [ ] Tesseract fallback works when available
- [ ] All existing 10 tests still pass
- [ ] 10+ new tests for real OCR functionality

### Performance Requirements 🚀
- [ ] Model download: < 2 minutes (~8MB total)
- [ ] Model loading: < 500ms
- [ ] Text detection: < 20ms per image
- [ ] Text recognition: < 10ms per region
- [ ] Total OCR time: < 50ms for typical images
- [ ] Tesseract fallback: < 500ms

### Quality Requirements 🎯
- [ ] >90% accuracy on clean printed text
- [ ] >70% accuracy on screenshots with code
- [ ] Handles multi-line text correctly
- [ ] Preserves text layout information (bounding boxes)
- [ ] Works on images from 100x30 to 4K resolution
- [ ] Graceful degradation on low-quality images

---

## 🚧 Implementation Phases

### Phase 5a: rten Integration (2-3 hours)
- Implement rten model loading
- Add model download with caching
- Test model initialization
- **Deliverable**: Models load successfully

### Phase 5b: Text Detection (2-3 hours)
- Implement detection preprocessing
- Add detection inference
- Post-process to get bounding boxes
- **Deliverable**: Text regions detected

### Phase 5c: Text Recognition (2-3 hours)
- Implement recognition preprocessing
- Add recognition inference
- Implement CTC decoder
- **Deliverable**: Text recognized from regions

### Phase 5d: Integration and Testing (2-3 hours)
- Wire up to existing OCR interface
- Add Tesseract fallback
- Comprehensive testing
- Performance benchmarking
- **Deliverable**: Full OCR system working

**Total Estimated Time**: 8-12 hours

---

## 🔄 Fallback Strategy

If rten integration encounters issues:
1. **Placeholder remains working** - Tests still pass with placeholder
2. **Tesseract-only mode** - Can fall back to just Tesseract
3. **Clear error messages** - Users know if OCR is unavailable
4. **Gradual rollout** - Can ship partial functionality

```rust
impl OcrSystem {
    pub fn get_available_engines(&self) -> Vec<OcrMethod> {
        let mut engines = Vec::new();

        if self.rust_ocr.initialized {
            engines.push(OcrMethod::RustOcr);
        }

        if self.tesseract.available {
            engines.push(OcrMethod::Tesseract);
        }

        engines
    }

    pub fn is_ocr_available(&self) -> bool {
        !self.get_available_engines().is_empty()
    }
}
```

---

## 📚 Dependencies

### Required Crates
✅ `image = "0.24"` - Image processing (already in Cargo.toml)
✅ `anyhow = "1.0"` - Error handling (already in Cargo.toml)
🆕 `rten = "0.11"` - ONNX Runtime in Rust (needs to be added)
🆕 `rten-tensor = "0.11"` - Tensor operations (needs to be added)
🆕 `hf-hub = "0.3"` - Model downloads (already in Cargo.toml)
🆕 `rayon = "1.8"` - Parallel processing (needs to be added)
🆕 `tempfile = "3.8"` - Temporary files for Tesseract (needs to be added)

### Add to Cargo.toml
```toml
[dependencies]
rten = "0.11"
rten-tensor = "0.11"
rayon = "1.8"
tempfile = "3.8"
```

### Optional System Dependencies
- **Tesseract OCR** (optional): `sudo apt install tesseract-ocr tesseract-ocr-eng`
- **Additional languages**: `sudo apt install tesseract-ocr-fra tesseract-ocr-spa` etc.

---

## 🎯 Revolutionary Impact

### What Phase 5 Delivers
- **Real Text Reading**: Symthaea can read screenshots, signs, documents
- **Code Understanding**: Read code from images/screenshots
- **Multi-Language**: Support for 100+ languages via Tesseract
- **Embodied Learning**: Can learn from visual text in environment

### User Experience
- User shows Symthaea a screenshot → She reads the text
- Symthaea sees code in image → She can explain it
- Document image → Symthaea extracts and summarizes text
- Multi-lingual text → Symthaea handles gracefully

**This is visual literacy!** 📖

---

## 🚀 Next Steps After Week 13 Day 5

### Week 13 Completion
- All 5 days of model integration plans complete
- Ready to begin implementations (Week 14+)
- Clear path from placeholders to real models

### Week 14+: Implementation Phase
- Begin with fastest wins (Larynx sine waves ✅ already done!)
- Implement SigLIP for vision
- Add Moondream for VQA
- Complete OCR integration
- Full testing and benchmarking

---

## 📝 Notes & Decisions

### Decision 1: rten Over Other ONNX Runtimes
- **Why**: Pure Rust, no C++ dependencies, smaller binary
- **Trade-off**: Less mature than onnxruntime-rs
- **Benefit**: Easier to build, deploy, and debug

### Decision 2: CRAFT + CRNN Architecture
- **Why**: Well-established, good accuracy, reasonable size
- **Alternative**: Could use Tesseract 5's LSTM models
- **Benefit**: Proven architecture with good performance

### Decision 3: Tesseract as Fallback Only
- **Why**: External dependency, slower, but very robust
- **Trade-off**: Not always available on all systems
- **Benefit**: Best of both worlds - fast Rust, robust fallback

### Decision 4: Simple CTC Decoder
- **Why**: Greedy decoding is fast and usually accurate
- **Alternative**: Beam search for better accuracy
- **Future**: Can add beam search later if needed

---

## 🔗 Model Sources

### Detection Model (CRAFT)
- **Paper**: "Character Region Awareness for Text Detection" (2019)
- **Source**: Convert from PyTorch to ONNX
- **Alternative**: Use pre-converted ONNX models from community

### Recognition Model (CRNN)
- **Paper**: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (2015)
- **Source**: Convert from PyTorch to ONNX
- **Alternative**: Train custom model with ocrs tools

### Model Hub Options
1. **HuggingFace**: Host custom ONNX models
2. **GitHub Releases**: Include models in releases
3. **Cloud Storage**: CDN for fast downloads
4. **Local**: Ship with application (8MB is reasonable)

---

**Status**: 📋 Plan complete, ready for implementation

**Current State**: Architecture and tests ready, awaiting real models

**Priority**: Can begin anytime - independent of other Week 13 tasks

🌊 Symthaea will read the world around her! 📖
